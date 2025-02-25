# 🚀 grpo-flat with zero dataset

Based on qwen-0.5b model, low-resource zero-shot GRPO training that changes the model's output style in 15 minutes with single GPU.
基于qwen-0.5b模型，低资源0样本grpo训练，单卡训练15分钟改变原模型输出风格

## 📢 News
[2025/02/21]  policy_model/ref_model splitted, serve your ref_model separately with ref_fastapi.py

[2025/02/19]  multi-gpu training supported, old version moved to grpo_vanilla.py

[2025/02/18]  GRPO supports 8bit/4bit quantized training, supports lora/qlora

[2025/02/16]  GRPO refactored, old version moved to grpo_vanilla.py

## 🤖 Models

| Model Name | Purpose | Description |
|------------|---------|-------------|
| Qwen2.5-0.5B-Instruct | Policy Model | Base model used for training |
| Qwen2.5-7B | LLM Rater | Prompt based reward model used for scoring |
| [xyj787878/Qwen2.5-0.5B-GRPO-kuakua](https://huggingface.co/xyj787878/Qwen2.5-0.5B-GRPO-kuakua) | Trained Model | A compliment bot trained on Qwen2.5-0.5B-Instruct. 基于Qwen2.5-0.5B-Instruct训练的夸夸机器人 |

## 🏗️ Structure

```
grpo-flat/
├── grpo.py            # GRPO main program
├── grpo_trainer.py    # Flat implementation of GRPO trainer
├── reward_funcs.py    # GRPO reward function library
├── ref_fastapi.py    # serve your reference model
├── serve_chat_model.py    # serve your chat model
└── chat_with_model.py    # quick test your chat model in console
```

## 🎯 Reward Functions

| Function Name | Description | Purpose |
|--------|------|------|
| `llm_rater_reward` | Use larger language model to score sampling results | Evaluate overall quality of generated text |
| `perplexity_reward` | Score based on reference model perplexity | Ensure fluency of generated text |
| `repetition_reward` | Repetition penalty scoring | Avoid text repetition |
| `length_reward` | Length control scoring | Control length of generated text |
| `chinese_char_ratio_reward` | Chinese character ratio scoring | Ensure output is primarily in Chinese |

## 💻 Device Requirements

| Purpose | GPU | Description |
|------|-----|------|
| Policy Model Training | NVIDIA RTX 4090 × 1 | For policy model training |
| LLM Rater | NVIDIA RTX 2070S × 1 | For running Qwen2.5-7B-int8 rating model |

## 🛠️ Environment Setup

### Setup with requirements.txt
```bash
conda create -n grpo-flat python=3.10
pip install -r requirements.txt
```

### Or setup with ngc container (to avoid any environment issues)
```bash
docker pull nvcr.io/nvidia/pytorch:24.01-py3

docker run -itd --name ngc_pytorch_2401 -v /[your_local_workspace]/:/home/workspace  --workdir /home/workspace -u 1000:1000 -p 6000:22 -p 6001:9001 -p 6002:9002 -p 6003:9003 --gpus all --privileged --ipc=host [your_ngc_image_id] bash

docker exec -it ngc_pytorch_2401 bash

# and enjoy
```

## 🌟 Zero-shot Training of a Compliment Bot. 0样本训练一个夸夸机器人
```
# design your grpo generation prompt
{"prompt": [
        [
            {'role': 'system', 'content': "你是一个夸夸机器人"},
            {'role': 'user', 'content': "尝试用尽量浮夸的语气夸我"}
        ]
    ]
}
```

```
# llm rater prompt
prompt = "你需要对一个夸夸机器人的回复进行打分，分值范围1-10分，越浮夸的回复分数越高。对不通顺的内容直接打0分。仅输出分数数字即可，不要输出任何其他内容。\n输入文本：{}，分数："
```

#### Deploy your llm rater model
```
# Deploy your llm rater model on idle device (2070s in my case)
docker run -d --gpus 'device=1' -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Execute your llm rater model (qwen2.5:7b in my case)
docker exec -it ollama ollama run qwen2.5:7b
```

#### Quick start with grpo_vanilla.py
```
# run grpo_vanilla.py with one gpu
CUDA_VISIBLE_DEVICES=0 accelerate launch grpo_vanilla.py
```

#### Split policy_model ref_model, serve your ref_model separately
```
CUDA_VISIBLE_DEVICES=1 python ref_fastapi.py \
    --model_name_or_path "lm_models/Qwen2.5-0.5B-Instruct" \
    --max_parallel_requests 2 \
    --max_wait_time 0.1
```

#### Full parameters with grpo.py
```
# train policy model with one gpu and full parameters, with policy_model ref_model splitted
CUDA_VISIBLE_DEVICES=0 accelerate launch grpo.py \
    --model_name_or_path "lm_models/Qwen2.5-0.5B-Instruct" \
    --num_epochs 300 \
    --batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --log_steps 1 \
    --save_steps 10 \
    --max_grad_norm 1 \
    --max_save 3 \
    --group_num 8 \
    --mini_batch_size 1 \
    --wandb_project "grpo_training" \
    --ref_model_url "http://localhost:8000/predict" \
    --ref_from_remote true

# train policy model with one gpu and full parameters
CUDA_VISIBLE_DEVICES=0 accelerate launch grpo.py \
    --model_name_or_path "lm_models/Qwen2.5-0.5B-Instruct" \
    --num_epochs 300 \
    --batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --log_steps 1 \
    --save_steps 10 \
    --max_grad_norm 1 \
    --max_save 3 \
    --group_num 8 \
    --mini_batch_size 1 \
    --wandb_project "grpo_training"
```

#### lora/qlora with 8bit/4bit quantization
```
# train policy model with one gpu with 8bit quantization and lora
CUDA_VISIBLE_DEVICES=0 accelerate launch grpo.py \
    --model_name_or_path "lm_models/Qwen2.5-0.5B-Instruct" \
    --num_epochs 300 \
    --batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --log_steps 1 \
    --save_steps 10 \
    --max_grad_norm 1 \
    --max_save 3 \
    --group_num 8 \
    --mini_batch_size 1 \
    --wandb_project "grpo_training" \
    --use_peft True \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --use_8bit True \
    --target_modules "q_proj,v_proj,lm_head"

# train policy model with one gpu with qlora
CUDA_VISIBLE_DEVICES=0 accelerate launch grpo.py \
    --model_name_or_path "lm_models/Qwen2.5-0.5B-Instruct" \
    --num_epochs 300 \
    --batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --log_steps 1 \
    --save_steps 10 \
    --max_grad_norm 1 \
    --max_save 3 \
    --group_num 8 \
    --mini_batch_size 1 \
    --wandb_project "grpo_training" \
    --use_peft True \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --use_4bit True \
    --qlora True \
    --target_modules "q_proj,v_proj,lm_head"
```

#### Train policy model with multiple gpus if you have ...
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch grpo.py \
    --model_name_or_path "lm_models/Qwen2.5-0.5B-Instruct" \
    --num_epochs 300 \
    --batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --log_steps 1 \
    --save_steps 10 \
    --max_grad_norm 1 \
    --max_save 3 \
    --group_num 8 \
    --mini_batch_size 1 \
    --wandb_project "grpo_training" \
    --use_peft True \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --use_4bit True \
    --qlora True \
    --target_modules "q_proj,v_proj,lm_head"
```
[[training log]](https://drive.google.com/file/d/1Lv8gGAUBP-YaPcYM4FiVqAPFWoT3BNBn/view?usp=sharing)

## 📊 wandb log
### reward curve
![wandblog](./assets/reward.png)
### training curve
![wandblog](./assets/train.png)

[wandb report](https://wandb.ai/freejack7878-individual/grpo_training/reports/grpo-flat--VmlldzoxMTM2NjcyMw)


## 💬 Chat with your tuned model
```
# Serve your model
python serve_chat_model.py

# Chat with your model
python chat_with_model.py
```

## 🎮 Demo

with qwen2.5-0.5b-instruct

![qwen2.5-0.5b-instruct](./assets/qwen.png)
![qwen2.5-0.5b-instruct](./assets/qwen2.png)

with 夸夸模型

![kuakua](./assets/kuakua.png)
![kuakua](./assets/kuakua2.png)

## 📝 Citation

If you use this code in your research or project, please cite it as follows:

```bibtex
@misc{grpo-flat,
  author = {Xu, Yijie},
  title = {GRPO-flat: Zero-shot GRPO Training Framework with Limited Resources},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/xyj787890/grpo-flat}},
  note = {A zero-shot GRPO training implementation based on Qwen2.5-0.5B model}
}
```
