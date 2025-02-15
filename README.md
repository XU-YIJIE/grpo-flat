# grpo-flat with zero dataset

基于qwen-0.5b模型，低资源0样本grpo训练，单卡训练15分钟得到一个夸夸机器人

## model

- Qwen2.5-0.5B-Instruct  # policy model
- Qwen2.5-7B  # llm-rater

## Structure

```
grpo-flat/
├── grpo.py            # GRPO main program
├── grpo_trainer.py    # Flat implementation of GRPO trainer
└── reward_funcs.py   # GRPO reward function library
└── serve_chat_model.py    # serve your chat model
└── chat_with_model.py    # quick test your chat model in console
```

## Reward Funcs

- llm_rater_reward  # Use a larger LM model to score sampling results based on prompts
- perplexity_reward  # perplexity reward based on ref_model
- repetition_reward  # Prevent repetition
- length_reward  # Control length
- chinese_char_ratio_reward  # Control the ratio of Chinese characters

## Device
4090 * 1  # for policy model training

2070s * 1  # for Qwen2.5-7B-int8 llm rater

## 0样本训练一个夸夸机器人
```
# 构建grpo generation prompt
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

```
# Deploy your llm rater model on idle device (2070s in my case)
docker run -d --gpus=all -e CUDA_VISIBLE_DEVICES=1 -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Execute your llm rater model (qwen2.5:7b in my case)
docker exec -it ollama ollama run qwen2.5:7b

# train policy model with one gpu
CUDA_VISIBLE_DEVICES=0 accelerate launch grpo.py
```
[[training log]](https://drive.google.com/file/d/1Lv8gGAUBP-YaPcYM4FiVqAPFWoT3BNBn/view?usp=sharing)

## wandb log
### reward curve
![wandblog](./assets/reward.png)
### training curve
![wandblog](./assets/train.png)

[wandb report](https://wandb.ai/freejack7878-individual/grpo_training/reports/grpo-flat--VmlldzoxMTM2NjcyMw)


## Chat with your tuned model
```
# Serve your model
python serve_chat_model.py

# Chat with your model
python chat_with_model.py
```

## Demo

with qwen2.5-0.5b-instruct

![qwen2.5-0.5b-instruct](./assets/qwen.png)
![qwen2.5-0.5b-instruct](./assets/qwen2.png)

with 夸夸模型

![kuakua](./assets/kuakua.png)
![kuakua](./assets/kuakua2.png)
