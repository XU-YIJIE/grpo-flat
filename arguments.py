import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model")
    # data process
    parser.add_argument("--input_jsonl", type=list, default=None, help="input_jsonl")
    parser.add_argument("--dataset_dir", type=str, default="dataset/sharegpt_gpt4", help="dataset_dir")
    parser.add_argument("--dataset_name", type=str, default="sharegpt_gpt4", help="dataset_name")
    parser.add_argument("--template", type=str, default="qwen", help="template")
    parser.add_argument("--cutoff_len", type=int, default=1024, help="cutoff_len")
    parser.add_argument("--preprocessing_num_workers", type=int, default=8, help="preprocessing_num_workers")
    # model
    parser.add_argument("--model_name_or_path", type=str, default="lm_models/Qwen2.5-0.5B-Instruct", help="model_name_or_path")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="lm_models/Qwen2.5-0.5B-Instruct", help="tokenizer_name_or_path")
    parser.add_argument("--use_8bit", type=bool, default=False, help="use_8bit")
    parser.add_argument("--use_4bit", type=bool, default=False, help="use_4bit")
    parser.add_argument("--torch_dtype", type=str, default="float16", help="torch_dtype", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--model_tag", type=str, default=None, help="model_tag")
    parser.add_argument("--from_scratch", type=bool, default=True, help="if to train from scratch")
    parser.add_argument("--model_out_dir", type=str, default="model_ckpts", help="model_out_dir")
    parser.add_argument("--max_save", type=int, default=3, help="max save checkpoints")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    
    # training
    parser.add_argument("--num_epochs", type=int, default=1, help="num_epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="batch_size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="gradient_accumulation_steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max_grad_norm")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning_rate")
    parser.add_argument("--seed", type=int, default=1024, help="transformer_random_seed")
    parser.add_argument("--resume", type=bool, default=False, help="resume training")
    
    # peft
    parser.add_argument("--use_peft", type=bool, default=False, help="if to use peft")
    parser.add_argument("--lora_rank", type=int, default=8, help="lora rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="lora dropout")
    parser.add_argument("--target_modules", type=str, default="all", help="The names of the modules to apply Lora to.")
    parser.add_argument("--modules_to_save", type=str, default="lm_head", help="List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. ")
    parser.add_argument("--qlora", type=bool, default=False, help="if to use qlora")
    
    # grpo
    parser.add_argument("--group_num", type=int, default=8, help="group_num")
    parser.add_argument("--mini_batch_size", type=int, default=1, help="mini_batch_size")
    
    # optimizer 
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="learning rate warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="optimizer weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam beta2")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="adam epsilon")
    
    # logging
    parser.add_argument("--wandb_project", type=str, default="sft_training", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--wandb_dir", type=str, default=None, help="wandb local dir, default is ./wandb/")
    parser.add_argument("--log_steps", type=int, default=10, help="log every n steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="save every n steps")
    parser.add_argument("--eval_steps", type=int, default=1000, help="save every n steps")
    args = parser.parse_args()
    return args