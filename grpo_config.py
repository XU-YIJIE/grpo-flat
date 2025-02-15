from dataclasses import dataclass

@dataclass
class GRPOConfig:
    learning_rate: float = 1e-5
    batch_size: int = 8
    group_num: int = 4  # 每个输入的采样数量
    mini_batch_size: int = 1  # minibatch更新的batch_size
    micro_batch_size: int = 1  # logprobs计算时的batch_size，用于降低显存峰值占用
    gradient_accumulation_steps: int = 1
    
    max_grad_norm: float = 1.0
    max_grad_norm: float = 1.0
    
    # KL惩罚相关
    init_kl_coef: float = 0.2
    beta: float = 0.01
    
    epochs: int = 4
    cliprange: float = 0.2
    
    seed: int = 42