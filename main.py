import os
import argparse
import torch
import yaml
import deepspeed
from prepare.loader import load_model_and_tokenizer, initialize_deepspeed_model
from prepare.preprocess import load_nonmember_data, load_pile_data, broadcast_data
from prepare.train import train_on_dataset, train_on_streaming_dataset
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', -1)))
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # ✅ DeepSpeed自动处理分布式初始化
    deepspeed.init_distributed()
    
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    print(f"Rank {rank}/{world_size}, Local Rank: {local_rank}")
    
    # 加载模型
    if rank == 0:
        print(f"Loading model: {config['model']['name']}")
    
    model, tokenizer = load_model_and_tokenizer(
        config['model']['name'], 
        local_rank
    )
    
    # ✅ 初始化DeepSpeed
    model_engine, optimizer, lr_scheduler = initialize_deepspeed_model(model, config)
    
    if rank == 0:
        print("✅ DeepSpeed ZeRO-2 initialized successfully")
        print(f"Train batch size: {model_engine.train_batch_size()}")
        print(f"Micro batch size per GPU: {model_engine.train_micro_batch_size_per_gpu()}")
        print(f"Gradient accumulation steps: {model_engine.gradient_accumulation_steps()}")
    
    # 加载数据
    nonmember_texts = load_nonmember_data(rank)
    pile_data_iterator = load_pile_data(
        config['data']['pile_max_samples'], 
        rank
    )
    
    # 广播数据到所有进程
    if local_rank != -1:
        nonmember_texts = broadcast_data(
            nonmember_texts, 
            device=model_engine.local_rank, 
            local_rank=local_rank
        )
    
    # 创建输出目录
    os.makedirs(config['output']['dir'], exist_ok=True)
    
    # 训练阶段1：Non-Member训练
    if rank == 0:
        print(f"Training on non-member samples for {config['training']['nonmember_epochs']} epochs...")
    
    train_on_dataset(
        model_engine, tokenizer, nonmember_texts, config, 
        local_rank, rank, "Non-Member Training"
    )
    
    torch.distributed.barrier()
    
    # 训练阶段2：Pile训练
    if rank == 0:
        print("Training on Pile dataset to simulate forgetting...")
    
    train_on_streaming_dataset(
        model_engine, tokenizer, pile_data_iterator, config, 
        local_rank, rank, "Pile Training"
    )
    
    torch.distributed.barrier()
    
    # 保存模型
    save_model(model_engine, tokenizer, config['output']['dir'], config, rank)

def save_model(model_engine, tokenizer, save_path, config, rank):
    """
    ✅ DeepSpeed模型保存
    """
    if rank == 0:  # 只在主进程保存
        save_path = Path(save_path) / config['model']['name'].replace("/", "_")
        os.makedirs(save_path, exist_ok=True)
        model = model_engine.module
        model = model.to(torch.float16)
        # DeepSpeed保存
        model = model.cpu()
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()