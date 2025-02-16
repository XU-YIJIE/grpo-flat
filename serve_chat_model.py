# encoding: utf-8
import time
import uuid
import warnings

import torch
import numpy as np
from typing import List
from flask import Flask, request, Blueprint
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
from marshmallow import validate, Schema, fields

warnings.filterwarnings('ignore', category=UserWarning)


# global args
DEVICE_NAME = "cuda:1" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_NAME)
# MODEL_PATH = "checkpoints/grpo_training_20250215_092214/step_280_20250215_093259"  # your checkpoint path
MODEL_PATH = "lm_models/Qwen2.5-0.5B-Instruct"
TOKENIZE_PATH = MODEL_PATH
max_new_tokens = 1024
temperature = 0.7
top_p = 0.9
top_k = 16
repetition_penalty = 1.1


class AppModel:
    def __init__(self):
        pass

    def init_app(self, tokenizer=None, model=None):
        self.tokenizer = tokenizer
        self.model = model

    def build_chat_input(self, tokenizer, messages: List[dict]):
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-(max_new_tokens - 1):]
        print(new_prompt)
        inputs_ids = tokenizer(new_prompt).data['input_ids']
        inputs_ids = (torch.tensor(inputs_ids, dtype=torch.long, device=DEVICE)[None, ...])
        return inputs_ids, tokenizer.eos_token_id, new_prompt


    def chat_no_stream(self, tokenizer, messages: List[dict]):
        input_ids, eos_token_id, new_prompt = self.build_chat_input(tokenizer, messages)
        input_len = input_ids.shape[1]
        gen_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": top_k,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "num_beams": 1,
            "num_return_sequences": 1,
        }
        
        completions = self.model.generate(input_ids, **gen_config)
        completion_ids = completions[:, input_len:]
        answer = tokenizer.decode(completion_ids[0].tolist(), skip_special_tokens=True)
        return answer


app_model = AppModel()

chat_bp = Blueprint('Chat', __name__, url_prefix='/v1/chat')
completions_bp = Blueprint('Completions', __name__, url_prefix='/v1/completions')


def empty_cache():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(chat_bp)
    app.register_blueprint(completions_bp)

    @app.after_request
    def after_request(resp):
        empty_cache()
        return resp

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZE_PATH, trust_remote_code=True, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True).to(DEVICE)

    app_model.init_app(tokenizer, model)

    return app


class ChatMessageSchema(Schema):
    role = fields.Str(required=True, metadata={"example": "system"})
    content = fields.Str(required=True, metadata={"example": "You are a helpful assistant."})


class CreateChatCompletionSchema(Schema):
    model = fields.Str(required=True, metadata={"example": "qwen"})
    messages = fields.List(
        fields.Nested(ChatMessageSchema), required=True,
        metadata={"example": [
            ChatMessageSchema().dump({"role": "system", "content": "You are a helpful assistant."}),
            ChatMessageSchema().dump({"role": "user", "content": "Hello!"})
        ]}
    )
    temperature = fields.Float(load_default=1.0, metadata={"example": 1.0})
    top_p = fields.Float(load_default=1.0, metadata={"example": 1.0})
    n = fields.Int(load_default=1, metadata={"example": 1})
    max_tokens = fields.Int(load_default=None, metadata={"example": None})
    stream = fields.Bool(load_default=False, example=False)
    presence_penalty = fields.Float(load_default=0.0, example=0.0)
    frequency_penalty = fields.Float(load_default=0.0, example=0.0)


class ChatCompletionChoiceSchema(Schema):
    index = fields.Int(metadata={"example": 0})
    message = fields.Nested(ChatMessageSchema, metadata={
        "example": ChatMessageSchema().dump(
            {"role": "assistant", "content": "\n\nHello there, how may I assist you today?"}
        )})
    finish_reason = fields.Str(
        validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]),
        metadata={"example": "stop"})


class ChatCompletionSchema(Schema):
    id = fields.Str(
        dump_default=lambda: uuid.uuid4().hex,
        metadata={"example": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"})
    object = fields.Constant("chat.completion")
    created = fields.Int(dump_default=lambda: int(time.time()), metadata={"example": 1695402567})
    model = fields.Str(metadata={"example": "qwen"})
    choices = fields.List(fields.Nested(ChatCompletionChoiceSchema))


@chat_bp.route("/completions", methods=['POST'])
def create_chat_completion():
    create_chat_completion = CreateChatCompletionSchema().load(request.json)

    response = app_model.chat_no_stream(app_model.tokenizer, create_chat_completion["messages"])

    message = ChatMessageSchema().dump(
        {"role": "assistant", "content": response})
    choice = ChatCompletionChoiceSchema().dump(
        {"index": 0, "message": message, "finish_reason": "stop"})

    return ChatCompletionSchema().dump({
        "model": "qwen",
        "choices": [choice]})


app = create_app()

if __name__ == '__main__':
    use_emb = False
    try:
        import ngrok
        import logging

        logging.basicConfig(level=logging.INFO)
        listener = ngrok.werkzeug_develop()
    except Exception:
        pass

    app.run(debug=False, host="0.0.0.0", port=8000)
