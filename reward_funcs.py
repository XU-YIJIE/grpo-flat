import json
from typing import List
import re
import requests
from concurrent.futures import ThreadPoolExecutor
import jieba
from collections import Counter
import torch
import numpy as np

def reward_punish_too_long(completions, punish_length=100, **kwargs):
    '''
    Reward function that gives higher scores to completions that are close to 20 tokens.
    '''
    return [-abs(punish_length - len(completion.split(" ")))/100 for completion in completions]


def reward_hard_thinking_format(completions, **kwargs) -> list[float]:
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]


def reward_soft_thinking_format(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def reward_think_mark(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    return [think_mark_num(response) for response in responses]

def reward_unbias(completions, responses, **kwargs):
    import nltk
    return [nltk.bleu(response.split(' '), completion.split(' '), weights=(1,0,0,0)) for completion, response in zip(completions, responses)]

def extract_think(text):
    if "<think>" not in text or "</think>" not in text:
        return ""
    think = text.split("<think>")[-1]
    think = think.split("</think>")[0]
    return think.strip()

def think_mark_num(text):
    reward = 0
    if text.count("<think>\n") == 1:
        reward += 0.125
        
    if text.count("</think>\n") == 1:
        reward += 0.125
        
    if text.count("<answer>\n") == 1:
        reward += 0.125
        
    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward

def request_ollama(prompt: str, model: str = "qwen2.5:7b") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()  # 检查响应状态
    return response.json()["response"]

    
def parallel_batch_request(prompts, model="qwen2.5:7b", max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(request_ollama, prompt, model)
            for prompt in prompts
        ]
        return [f.result() for f in futures]


def llm_rater_reward(completions, **kwargs):
    print(completions)
    prompt = "你需要对一个夸夸机器人的回复进行打分，分值范围1-10分，越浮夸的回复分数越高。对不通顺的内容直接打0分。仅输出分数数字即可，不要输出任何其他内容。\n输入文本：{}，分数："
    prompts = [prompt.format(completion) for completion in completions]
    responses = parallel_batch_request(prompts)
    scores = []
    for response in responses:
        matches = re.findall(r'\b([1-9]|10)\b', response)
        score = int(matches[0]) if matches else 0
        scores.append(score)
    print(scores)
    return scores


def perplexity_reward(completions, model, tokenizer, **kwargs) -> float:
    encodings = tokenizer(completions, return_tensors="pt", add_special_tokens=True, padding=True)
    input_ids = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)
    labels = torch.where(attention_mask == 1, input_ids, torch.tensor(-100).to(model.device))
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        batch_size, seq_len, vocab_size = logits.shape
        shift_logits = logits[:, :-1, :]  # (b, seq_len-1, vocab_size)
        shift_labels = labels[:, 1:]    # (b, seq_len-1)
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        loss = loss.reshape(batch_size, seq_len-1).mean(dim=1)  # batch_size
        perplexity = torch.exp(loss).cpu()
        
    max_score = 10  # 控制得分区间为 0-10
    return max_score * (1 - (torch.log(perplexity) - 1) / 10).clamp(0, 1)  # perplexity越低，得分越高


# def get_ngram_scores(completions, **kwargs) -> List[float]:
#     scores = []
#     for completion in completions:
#         # 分词
#         words = ' '.join(jieba.cut(text))
#         score = self.kenlm_model.score(words)
#         scores.append(score)
#     return scores

def repetition_reward(completions, **kwargs) -> List[float]:
    """重复率得分，重复情况越多得分越低"""
    scores = []
    for completion in completions:
        words = list(jieba.cut(completion))
        word_counts = Counter(words)
        repetition_rate = len(word_counts) / len(words)
        scores.append(repetition_rate * 10)
    return scores

def length_reward(completions, **kwargs) -> List[float]:
    """长度得分，长度越接近ideal length得分越高"""
    ideal_length = 200  # ideal length
    scores = []
    for completion in completions:
        length = len(completion)
        score = np.exp(-((length - ideal_length) ** 2) / (2 * 50 ** 2))
        scores.append(score * 10)
    return scores

def chinese_char_ratio_reward(completions, **kwargs) -> List[float]:
    """中文字符比例得分，中文字符比例越高得分越高"""
    scores = []
    for completion in completions:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', completion))
        total_chars = len(completion)
        ratio = chinese_chars / total_chars if total_chars > 0 else 0
        scores.append(ratio * 10)
    return scores