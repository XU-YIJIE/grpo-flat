from ast import List
import shutil
from pyparsing import Any
import torch
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM,
                          get_linear_schedule_with_warmup,
                          PreTrainedTokenizer)
from grpo_trainer import GRPOTrainer, GRPOConfig
from reward_funcs import (reward_punish_too_long, 
                          reward_unbias, 
                          llm_rater_reward, 
                          perplexity_reward,
                          repetition_reward, 
                          length_reward, 
                          chinese_char_ratio_reward)
import os
from loguru import logger
from accelerate import Accelerator
from transformers.data.data_collator import DataCollatorForSeq2Seq
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import datetime
import wandb
from datasets import Dataset
from typing import Dict, List, Any

# from datasets import load_dataset
# from data.aligner import convert_tldr
# from data.preprocess import preprocess_rl_dataset_v1
# from constants import IGNORE_INDEX

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def preprocess_rl_dataset_v1(
    examples: Dict[str, List[Any]], 
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, List[List[int]]]:
    model_inputs = {"prompt": [], "response":[]}
    for i in range(len(examples["prompt"])):
        prompt = examples["prompt"][i]
        response = examples["response"][i]
        input_str = tokenizer.apply_chat_template(prompt, template=tokenizer.chat_template, tokenize=False, add_generation_prompt=True)
        model_inputs["prompt"].append(input_str)
        model_inputs["response"].append(response[0]['content'])
    return model_inputs


def get_demo_data():
    data = {
        "prompt": [
            [
                {'role': 'system', 'content': "你是一个夸夸机器人"},
                {'role': 'user', 'content': "尝试用尽量浮夸的语气夸我"}
            ]
        ],
        "response": [
            [
                {'role': 'assistant', 'content': ""}
            ]
        ]
    }
    dataset = Dataset.from_dict(data)
    dataset.set_format(type="torch", columns=["prompt", "response"])
    return dataset


def main():
    def save_manager(current_epoch, current_steps, current_avg_reward, max_save=None, prefix=None):
        checkpoints_dir = f"checkpoints/{wandb_run_name}"
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        
        if max_save is not None:
            step_dirs = [d for d in os.listdir(checkpoints_dir) 
                        if os.path.isdir(os.path.join(checkpoints_dir, d)) and d.startswith(prefix)]
            step_dirs.sort(key=lambda x: os.path.getctime(os.path.join(checkpoints_dir, x)))
            
            while len(step_dirs) >= max_save:
                oldest_dir = os.path.join(checkpoints_dir, step_dirs[0])
                shutil.rmtree(oldest_dir)
                step_dirs.pop(0)
        
        save_dir = os.path.join(checkpoints_dir, f"{prefix}_{current_steps}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(save_dir, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        training_state = {
            'step': current_steps,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'reward': current_avg_reward,
            'epoch': current_epoch,
        }
        torch.save(training_state, os.path.join(save_dir, "training_state.pt"))        
        
        
    model_name_or_path = "lm_models/Qwen2.5-0.5B-Instruct"  # 使用Qwen2.5-0.5B-Instruct作为基础模型
    dataset_dir = "dataset/tldr"
    
    learning_rate = 1e-6
    group_num = 8
    mini_batch_size = 1
    batch_size = 4  # 每个global_steps更新 batch_size / mini_batch_size 次
    gradient_accumulation_steps = 1
    # 每 gradient_accumulation_steps / (batch_size / mini_batch_size) 个global_steps反向传播一次
    num_epochs = 300
    log_steps = 1
    save_steps = 10

    max_grad_norm = 1
    seed = 1024
    max_save = 3
    
    resume = False
    
    # wandb
    wandb_project = "grpo_training"
    wandb_run_name = f"{wandb_project.split('/')[-1]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_dir = f"./wandb/{wandb_run_name}"
    
    # config
    config = GRPOConfig(
        learning_rate=learning_rate,
        group_num=group_num,  # 每个输入采样group_num个候选回复
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        seed=seed
    )
    
    # accelerator init
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    # wandb init
    if accelerator.is_main_process:
        if not os.path.exists(wandb_dir):
            os.makedirs(wandb_dir)
        
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            dir=wandb_dir,
            config={
                "model_name_or_path": model_name_or_path,
                "dataset": dataset_dir,
                "batch_size": batch_size,
                "mini_batch_size": mini_batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "group_num": group_num,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "max_grad_norm": max_grad_norm,
                "seed": seed,
                "wandb_dir": wandb_dir,
            }
        )
        logger.info(f"Wandb local dir: {wandb_dir}")
    
    # model
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    # 注意padding区分left/right
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, padding_side="left")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    ref_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).cpu()
    for param in ref_model.parameters():
        param.requires_grad = False
        
    # # dataprocess
    # dataset = load_dataset(path=dataset_dir, split="train")
    # # dataset = dataset.select(range(100))
    # dataset = convert_tldr(dataset)
    
    dataset = get_demo_data()
    
    column_names = list(next(iter(dataset)).keys())
    preprocess_func = partial(
        preprocess_rl_dataset_v1,
        tokenizer=tokenizer,
    )
    with accelerator.main_process_first():
        dataset = dataset.map(
            preprocess_func,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
            desc="Preprocessing dataset"
        )
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # optimizer & lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(dataloader) * (batch_size // mini_batch_size) // gradient_accumulation_steps
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    
    # resume training state from checkpoint
    if resume:
        training_state = torch.load(os.path.join(model_name_or_path, "training_state.pt"), map_location="cpu" )
        optimizer.load_state_dict(training_state['optimizer_state_dict'])
        lr_scheduler.load_state_dict(training_state['scheduler_state_dict'])
        
        start_epoch = training_state['epoch']
        total_steps_count = training_state['step']
        best_reward = training_state['reward']
        logger.info(f"Resuming from epoch {start_epoch}, step {total_steps_count}")
    else:
        start_epoch = 0
        total_steps_count = 0
        best_reward = float('-inf')
    

    model, optimizer, dataloader, lr_scheduler, ref_model = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler, ref_model)
    
    trainer = GRPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    
    # define reward funcs
    perplexity_reward_func = partial(perplexity_reward, model=ref_model, tokenizer=tokenizer)
    perplexity_reward_func.__name__ = "perplexity_reward"
    reward_funcs = [
                    perplexity_reward_func, 
                    llm_rater_reward, 
                    repetition_reward, 
                    length_reward, 
                    chinese_char_ratio_reward
                ]
    
    max_train_steps = num_epochs * len(dataloader)
    progress_bar = tqdm(range(max_train_steps), desc="Training Steps", disable=not accelerator.is_local_main_process)
    progress_bar.update(total_steps_count)  # 更新进度条到恢复的步数
    
    for epoch in range(start_epoch, num_epochs):
        metrics = {}
        for step, batch in enumerate(dataloader):
            prompts = batch["prompt"]
            prompt_ids = tokenizer(prompts, return_tensors="pt", add_special_tokens=True, padding=True)["input_ids"].to(accelerator.device)
            prompt_len = prompt_ids.shape[1]
            # reject sampling
            gen_config = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
                "num_beams": 1,
                "num_return_sequences": group_num,
            }
            completions = trainer.generate(
                prompt_ids,
                **gen_config
            )
            torch.cuda.empty_cache()

            completion_ids = completions[:, prompt_len:]
            completion_texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)  # group_num * batch_size
            
            # get reward kwargs
            reward_kwargs = {key: [] for key in batch.keys() if key not in ["prompt", "completions"]}
            for key in reward_kwargs:
                for example in batch[key]:
                    # Repeat each value in the column for `num_generations` times
                    reward_kwargs[key].extend([example] * config.group_num)
                    
            # call reward funcs
            all_rewards = []
            for reward_func in reward_funcs:
                rewards = np.array(reward_func(completion_texts, **reward_kwargs))  # length: group_num * batch_size
                all_rewards.append(rewards)  # length: func_num
            all_rewards = np.array(all_rewards)  # func_num, group_num * batch_size
            reward_per_func = all_rewards.mean(axis=1)  # func_num
            reward_all_funcs = all_rewards.sum(axis=0)  # group_num * batch_size
            
            for i, reward_func in enumerate(reward_funcs):
                reward_func_name = reward_func.__name__
                metrics[f"rewards/{reward_func_name}"] = reward_per_func[i].item()
                
            # expand prompt_ids, align with length of completions
            prompt_ids = prompt_ids.unsqueeze(1).expand(-1, config.group_num, -1).reshape(-1, prompt_ids.size(-1))
            reward_all_funcs = torch.tensor(reward_all_funcs, device=accelerator.device)
            
            # grpo step
            trainer.step(
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                reward_scores=reward_all_funcs
            )
            total_steps_count += 1
            current_lr = lr_scheduler.get_last_lr()[0]
            avg_reward_score = reward_all_funcs.mean().item()
            
            progress_bar.update(1)
            progress_bar.set_postfix({
                'avg_reward_score': f"{avg_reward_score:.3f}",
                'lr': f"{current_lr:.5e}",
            })

            # wandb logging
            if accelerator.is_main_process and total_steps_count % log_steps == 0:
                lr_scale = current_lr / learning_rate
                
                metrics.update({
                    "train/learning_rate": current_lr,
                    "train/lr_scale": lr_scale,
                    "train/epoch": epoch + (step + 1) / len(dataloader),
                    "train/global_step": total_steps_count,
                    "train/avg_reward_score": avg_reward_score,
                })
                wandb.log(metrics, step=total_steps_count)

                if avg_reward_score > best_reward and total_steps_count % save_steps == 0:
                    best_reward = avg_reward_score
                    metrics["best_reward"] = best_reward
                    save_manager(epoch + 1, total_steps_count, avg_reward_score, max_save=max_save, prefix="step")
    
    

if __name__ == "__main__":
    main()