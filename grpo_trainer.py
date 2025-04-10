import argparse
import os
import requests
import torch
import torch.nn.functional as F
from typing import List, Optional, Union
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling
from accelerate import Accelerator
import warnings
import time
import numpy as np
from tqdm import tqdm
import json
from typing import Callable
from loguru import logger


def expand_indice(input_tensor: torch.Tensor, group_num: int) -> torch.Tensor:
    result = []
    for start in input_tensor:
        sequence = torch.arange(start * group_num, start * group_num + group_num)
        result.append(sequence)
    
    return torch.cat(result)


class GRPOTrainer:
    def __init__(
        self, args: argparse.Namespace, model, ref_model, tokenizer, optimizer, lr_scheduler, accelerator: Accelerator
    ):
        self.args = args
        
        # grpo
        self.group_num = args.group_num
        self.mini_batch_size = args.mini_batch_size
        self.beta = args.beta
        self.cliprange = args.cliprange
        self.max_grad_norm = args.max_grad_norm
        self.ref_model_url = args.ref_model_url
        self.ref_from_remote = args.ref_from_remote
        
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.accelerator = accelerator
        self.current_device = self.accelerator.device
        

    def get_per_token_logps(self, model, input_ids, num_logits_to_keep):
        # We add 1 to `num_logits_to_keep` because the last logits of the sequence is later excluded
        # num_logits_to_keep 仅作用在lm_head上，控制logits长度
        logits = model(input_ids=input_ids, num_logits_to_keep=num_logits_to_keep + 1).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids[:, -num_logits_to_keep:]):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def get_all_logprobs(
        self,
        model,
        prompt_ids: torch.Tensor,  # 输入padding后的
        completion_ids: torch.Tensor,  # 输入padding后的
        from_remote: bool = False,
    ):
        batch_size = prompt_ids.shape[0] // self.group_num
        all_logprobs = []
        
        for i in range(0, batch_size, self.mini_batch_size):
            mini_prompt_ids = prompt_ids[i * self.group_num: (i + self.mini_batch_size) * self.group_num]
            mini_completion_ids = completion_ids[i * self.group_num: (i + self.mini_batch_size) * self.group_num]
            input_ids = torch.cat([mini_prompt_ids, mini_completion_ids], dim=1)
            if not from_remote:
                logprobs = self.get_per_token_logps(model, input_ids, num_logits_to_keep=completion_ids.shape[1])  # (mini_batch_size * group_num, seq_len)
            else:
                logprobs = self.get_logps_from_remote_model(input_ids, num_logits_to_keep=completion_ids.shape[1])
            all_logprobs.append(logprobs)
            
            del input_ids, logprobs, mini_prompt_ids, mini_completion_ids
            torch.cuda.empty_cache()
            
        return torch.cat(all_logprobs)
        
    def get_logps_from_remote_model(self, input_ids, num_logits_to_keep):
        response = requests.post(
            self.ref_model_url,
            json={"input_ids": input_ids.cpu().numpy().tolist(),
                  "num_logits_to_keep": num_logits_to_keep})
        if response.status_code == 200:
            return torch.tensor(response.json()["log_probs"]).to(self.current_device)
        else:
            raise Exception(f"Error: {response.text}")

    def grpo_advantage(self, rewards: torch.FloatTensor, epsilon: float = 1e-4):
        rewards = rewards.to(torch.float32)
        batch_size = rewards.shape[0] // self.group_num
        rewards_grouped = rewards.view(batch_size, self.group_num)
        mean = rewards_grouped.mean(dim=1, keepdim=True)  # [batch_size, 1]
        std = rewards_grouped.std(dim=1, keepdim=True)    # [batch_size, 1]
        advantages_grouped = (rewards_grouped - mean) / (std + epsilon)
        advantages = advantages_grouped.view(-1)
        return advantages

    def step(
        self,
        prompt_ids: torch.LongTensor,
        completion_ids: torch.LongTensor,
        reward_scores: torch.FloatTensor,
    ):
        """
        prompt_ids: group_num * batch_size, prompt_len
        completion_ids: group_num * batch_size, completion_len
        """
        self.model.train()
        
        batch_size = prompt_ids.shape[0] // self.group_num
        completions_mask = completion_ids != self.tokenizer.pad_token_id
        advantages = self.grpo_advantage(reward_scores)  # (group_num * batch_size)
        
        old_logprobs = self.get_all_logprobs(self.model, prompt_ids, completion_ids)  
        # (group_num * batch_size, completion_len)
        ref_logprobs = self.get_all_logprobs(self.ref_model, prompt_ids, completion_ids, from_remote=self.ref_from_remote)

        batch_indice = torch.arange(batch_size)
        for mini_batch_start in range(0, batch_size, self.mini_batch_size):
            mini_batch_end = mini_batch_start + self.mini_batch_size
            mini_batch_inds = batch_indice[mini_batch_start:mini_batch_end]
            mini_batch_inds = expand_indice(mini_batch_inds, self.group_num)
            
            # start = mini_batch_start * self.group_num
            # end = (start + self.mini_batch_size) * self.group_num

            mini_prompt_ids, mini_completion_ids, mini_old_logprobs, mini_ref_logprobs, mini_completions_mask, mini_advantages = prompt_ids[mini_batch_inds], completion_ids[mini_batch_inds], old_logprobs[mini_batch_inds], ref_logprobs[mini_batch_inds], completions_mask[mini_batch_inds], advantages[mini_batch_inds]  # mini_old_logprobs: group_num * mini_batch_size, completion_len
            
            with self.accelerator.accumulate(self.model):
                new_logprobs = self.get_all_logprobs(self.model, mini_prompt_ids, mini_completion_ids)
                del mini_prompt_ids, mini_completion_ids
                
                ratio = torch.exp(new_logprobs - mini_old_logprobs.detach())
                del mini_old_logprobs
                
                ratio_clip = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange)
                pg_loss = torch.min(mini_advantages.unsqueeze(dim=1) * ratio, mini_advantages.unsqueeze(dim=1) * ratio_clip)
                # pg_loss = mini_advantages.unsqueeze(dim=1) * ratio
                del mini_advantages
                
                unbiased_kl = torch.exp(mini_ref_logprobs - new_logprobs) - (mini_ref_logprobs - new_logprobs) - 1
                
                pg_loss = - pg_loss + self.beta * unbiased_kl
                loss = ((pg_loss * mini_completions_mask).sum(dim=1) / mini_completions_mask.sum(dim=1)).mean()
                del mini_completions_mask, unbiased_kl
                
                self.accelerator.backward(loss)
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
    def generate(
        self,
        prompt_tensor: Union[torch.Tensor],
        **generation_kwargs
    ):  
        unwrapped_model = self.accelerator.unwrap_model(self.model).eval()
        with torch.inference_mode():
            outputs = unwrapped_model.generate(
                input_ids=prompt_tensor,
                **generation_kwargs
            )
        del unwrapped_model
        return outputs