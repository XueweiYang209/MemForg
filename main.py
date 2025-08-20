import os
import argparse
import torch
import torch.distributed as dist
import yaml
from model.loader import load_model_and_tokenizer
from data.preprocess import load_nonmember_data, load_pile_data, broadcast_data
from train.train_ddp import train_on_dataset, train_on_streaming_dataset
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', -1)))
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_distributed(local_rank):
    """初始化分布式环境"""
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda", local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0
    return device, world_size, rank

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # 初始化分布式环境
    device, world_size, rank = setup_distributed(args.local_rank)
    
    print(f"local_rank: {args.local_rank}, world_size: {world_size}, rank: {rank}, device: {device}")
    # 加载模型
    if rank == 0:
        print(f"Loading model: {config['model']['name']}")
    model, tokenizer = load_model_and_tokenizer(
        config['model']['name'], 
        args.local_rank, 
        device
    )
    
    # 加载数据
    nonmember_texts = load_nonmember_data(rank)

    pile_data_iterator = load_pile_data(
        config['data']['pile_max_samples'], 
        rank
    )
    
    # 广播数据到所有进程
    if args.local_rank != -1:
        nonmember_texts = broadcast_data(nonmember_texts, device=device, local_rank=args.local_rank)
    
    # 创建输出目录
    os.makedirs(config['output']['dir'], exist_ok=True)
    
    # 2. 在Non-Member样本上训练
    if rank == 0:
        print(f"Training on non-member samples for {config['training']['nonmember_epochs']} epochs...")
    train_on_dataset(model, tokenizer, nonmember_texts, config, 
                    args.local_rank, rank, "Non-Member Training")
    
    if args.local_rank != -1:
        dist.barrier()
    
    # 4. 在Pile数据集上流式训练模拟遗忘
    if rank == 0:
        print("Training on Pile dataset to simulate forgetting...")
    train_on_streaming_dataset(model, tokenizer, pile_data_iterator, config, 
                              args.local_rank, rank, "Pile Training")
    
    if args.local_rank != -1:
        dist.barrier()
    save_model(model, tokenizer, config['output']['dir'], config, args.local_rank)
    
    if args.local_rank != -1:
        dist.destroy_process_group()


# 在训练完成后添加模型保存
def save_model(model, tokenizer, save_path, config, local_rank):
    """保存模型和tokenizer"""
    save_path = Path(save_path) / config['model']['name']
    if local_rank <= 0:  # 只在主进程保存
        os.makedirs(save_path, exist_ok=True)
        if hasattr(model, 'module'):
            model.module.save_pretrained(save_path)
        else:
            model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()