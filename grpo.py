import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
from grpo_trainer import GRPOTrainer, GRPOConfig
from reward_funcs import (
    perplexity_reward,
    repetition_reward,
    length_reward,
    chinese_char_ratio_reward
)
import os
from loguru import logger
from accelerate import Accelerator
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import datetime
import wandb
from datasets import Dataset, load_dataset
from typing import Dict, List, Any
import sys
import gc
import argparse
from peft import LoraConfig, TaskType, get_peft_model
from arguments import parse_args
import shutil

logger = logger.bind(name="grpo")
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    filter=lambda record: record["level"].name in ["INFO", "ERROR"]
)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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

def find_all_linear_names(model, int4=False, int8=False):
    """Find all linear layer names in the model."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer:
    def __init__(self, args: argparse.Namespace):
        # data
        self.dataset_dir = args.dataset_dir
        self.preprocessing_num_workers = args.preprocessing_num_workers
        
        # model
        self.model_name_or_path = args.model_name_or_path
        self.tokenizer_name_or_path = args.tokenizer_name_or_path
        self.use_8bit = args.use_8bit
        self.use_4bit = args.use_4bit
        self.torch_dtype = args.torch_dtype
        self.device = args.device
        
        # training
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.num_warmup_steps = args.num_warmup_steps
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.max_grad_norm = args.max_grad_norm
        self.log_steps = args.log_steps
        self.save_steps = args.save_steps
        self.max_save = args.max_save
        self.group_num = args.group_num
        self.mini_batch_size = args.mini_batch_size
        
        # peft
        self.use_peft = args.use_peft
        self.lora_rank = args.lora_rank
        self.lora_alpha = args.lora_alpha
        self.lora_dropout = args.lora_dropout
        self.target_modules = args.target_modules
        self.qlora = args.qlora
        
        # wandb
        self.wandb_project = args.wandb_project
        self.wandb_run_name = args.wandb_run_name or f"{args.wandb_project.split('/')[-1]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.wandb_dir = args.wandb_dir or f"./wandb/{self.wandb_run_name}"
        
        # config
        self.grpo_config = GRPOConfig(
            group_num=self.group_num,
            mini_batch_size=self.mini_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_grad_norm=self.max_grad_norm,
        )
        
        self.accelerator = Accelerator(gradient_accumulation_steps=self.gradient_accumulation_steps)
        self.initializer()
        
        if self.accelerator.is_main_process:
            if not os.path.exists(self.wandb_dir):
                os.makedirs(self.wandb_dir)
            
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                dir=self.wandb_dir,
                config={
                    "model_name": self.model_name_or_path,
                    "dataset": self.dataset_dir,
                    "batch_size": self.batch_size,
                    "mini_batch_size": self.mini_batch_size,
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "max_grad_norm": self.max_grad_norm,
                    "group_num": self.group_num,
                    "use_8bit": self.use_8bit,
                    "use_4bit": self.use_4bit,
                    "wandb_dir": self.wandb_dir,
                }
            )
            logger.info(f"Wandb local dir: {self.wandb_dir}")

    def initializer(self):
        torch.cuda.empty_cache()
        
        # Quantization config
        quantization_config = None
        if self.use_4bit and self.use_8bit:
            raise ValueError("Error, use_4bit and use_8bit cannot be set at the same time")
        elif self.use_8bit or self.use_4bit:
            logger.info(f"Quantizing model, use_4bit: {self.use_4bit}, use_8bit: {self.use_8bit}")
            if self.use_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif self.use_4bit:
                if self.qlora:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=self.torch_dtype,
                    )
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=self.torch_dtype,
                    )
        
        # Model
        kwargs = {
            "pretrained_model_name_or_path": self.model_name_or_path,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if quantization_config:
            kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(**kwargs)
        
        if self.use_peft:
            target_modules = self.target_modules.split(',') if self.target_modules else None
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(self.model, int4=self.use_4bit, int8=self.use_8bit)
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules or ["q_proj", "v_proj"],
                inference_mode=False,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                use_rslora=True if self.use_4bit or self.use_8bit else False,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        if self.accelerator.is_main_process:
            logger.info(f"Total trainable parameters: {count_parameters(self.model)/1e9:.5f}B")
        
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path or self.model_name_or_path, 
            trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Reference model
        self.ref_model = AutoModelForCausalLM.from_pretrained(**kwargs)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.create_dataloader()
        self.create_optimizer()
        
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler, self.ref_model = self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader, self.lr_scheduler, self.ref_model)
        
        self.grpo_trainer = GRPOTrainer(
            config=self.grpo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            accelerator=self.accelerator,
        )

    def create_dataloader(self):
        # dataset = load_dataset(path=self.dataset_dir, split="train")
        # dataset = convert_tldr(dataset)
        
        dataset = get_demo_data()
        
        column_names = list(next(iter(dataset)).keys())
        preprocess_func = partial(preprocess_rl_dataset_v1, tokenizer=self.tokenizer)
        
        with self.accelerator.main_process_first():
            dataset = dataset.map(
                preprocess_func,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                remove_columns=column_names,
                desc="Preprocessing dataset"
            )
        
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False
        )

    def create_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if self.use_8bit or self.use_4bit:
            if self.use_4bit:
                from torchao.prototype.low_bit_optim import AdamW4bit
                self.optimizer = AdamW4bit(optimizer_grouped_parameters, lr=self.learning_rate)
            else:
                from bitsandbytes.optim import AdEMAMix
                self.optimizer = AdEMAMix(optimizer_grouped_parameters, lr=self.learning_rate, optim_bits=8)
        else:
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        num_update_steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
        max_train_steps = self.num_epochs * num_update_steps_per_epoch
        self.lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=max_train_steps)

    def save_manager(self, current_epoch, current_steps, current_reward, max_save=None, prefix=None):
        checkpoints_dir = f"checkpoints/{self.wandb_run_name}"
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
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        training_state = {
            'step': current_steps,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'reward': current_reward,
            'epoch': current_epoch,
        }
        torch.save(training_state, os.path.join(save_dir, "training_state.pt"))

    def train(self):
        total_steps = 0
        best_reward = float('-inf')
        
        # Define reward functions
        perplexity_reward_func = partial(perplexity_reward, model=self.ref_model, tokenizer=self.tokenizer)
        perplexity_reward_func.__name__ = "perplexity_reward"
        reward_funcs = [
            perplexity_reward_func,
            repetition_reward,
            length_reward,
            # chinese_char_ratio_reward
        ]
        
        max_train_steps = self.num_epochs * len(self.train_dataloader)
        progress_bar = tqdm(range(max_train_steps), desc="Training Steps", disable=not self.accelerator.is_local_main_process)
        
        for epoch in range(self.num_epochs):
            metrics = {}
            for step, batch in enumerate(self.train_dataloader):
                prompts = batch["prompt"]
                prompt_ids = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding=True
                )["input_ids"].to(self.accelerator.device)
                prompt_len = prompt_ids.shape[1]
                
                # Generate completions
                gen_config = {
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repetition_penalty": 1.1,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "use_cache": True,
                    "num_beams": 1,
                    "num_return_sequences": self.group_num,
                }
                completions = self.grpo_trainer.generate(prompt_ids, **gen_config)
                torch.cuda.empty_cache()
                
                completion_ids = completions[:, prompt_len:]
                completion_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)  # group_num * batch_size
                
                # get reward kwargs
                reward_kwargs = {key: [] for key in batch.keys() if key not in ["prompt", "completions"]}
                for key in reward_kwargs:
                    for example in batch[key]:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example] * self.group_num)
                
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
                prompt_ids = prompt_ids.unsqueeze(1).expand(-1, self.group_num, -1).reshape(-1, prompt_ids.size(-1))
                reward_all_funcs = torch.tensor(reward_all_funcs, device=self.accelerator.device)
                
                # GRPO training step
                self.grpo_trainer.step(
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    reward_scores=reward_all_funcs
                )
                
                total_steps += 1
                current_lr = self.lr_scheduler.get_last_lr()[0]
                avg_reward_score = reward_all_funcs.mean().item()
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'avg_reward_score': f"{avg_reward_score:.3f}",
                    'lr': f"{current_lr:.5e}",
                    'epoch': f"{epoch + (step + 1) / len(self.train_dataloader):.3f}",
                })
                
                # Logging
                if self.accelerator.is_main_process and total_steps % self.log_steps == 0:
                    lr_scale = current_lr / self.learning_rate
                    
                    metrics.update({
                        "train/learning_rate": current_lr,
                        "train/lr_scale": lr_scale,
                        "train/epoch": epoch + (step + 1) / len(self.train_dataloader),
                        "train/global_step": total_steps,
                        "train/avg_reward_score": avg_reward_score,
                    })
                    
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        metrics["train/grad_norm"] = grad_norm
                    
                    if avg_reward_score > best_reward and total_steps % self.save_steps == 0:
                        best_reward = avg_reward_score
                        metrics["best_reward"] = best_reward
                        self.save_manager(epoch + 1, total_steps, avg_reward_score, max_save=self.max_save, prefix="step")
                    
                    wandb.log(metrics, step=total_steps)
                
                # Memory cleanup
                if step % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            self.accelerator.wait_for_everyone()
            torch.cuda.empty_cache()
            gc.collect()
            
            # # Save checkpoint after each epoch
            # if self.accelerator.is_main_process:
            #     self.save_manager(epoch + 1, total_steps, avg_reward_score, max_save=self.max_save, prefix="epoch")
            #     logger.info(f"Epoch {epoch+1}/{self.num_epochs}:")
            #     logger.info(f"  Average Reward: {avg_reward_score:.4f}")
            #     logger.info(f"  Learning Rate: {current_lr:.2e}")
        
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            self.save_manager(self.num_epochs, total_steps, avg_reward_score, max_save=self.max_save, prefix="final")
            wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    
    # Override some arguments for GRPO training
    args.model_name_or_path = "lm_models/Qwen2.5-0.5B-Instruct"
    args.learning_rate = 1e-6
    args.batch_size = 2
    args.gradient_accumulation_steps = 1
    args.num_epochs = 300
    args.log_steps = 1
    args.save_steps = 10
    args.max_grad_norm = 1
    args.max_save = 3
    args.wandb_project = "grpo_training"
    
    args.group_num = 8
    args.mini_batch_size = 1
    
    args.use_peft = True
    args.lora_rank = 16
    args.lora_alpha = 16
    args.lora_dropout = 0.1
    args.use_8bit = True
    # args.use_4bit = True
    # args.qlora = True
    args.target_modules = "q_proj,v_proj,lm_head"
    
    trainer = Trainer(args)
    trainer.train()