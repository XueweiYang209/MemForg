import torch
from torch.utils.data import DataLoader, distributed
from tqdm import tqdm
import os
from data.dataset import TextDataset
from data.preprocess import broadcast_data

def train_on_dataset(model, tokenizer, texts, config, local_rank=-1, rank=0, desc="Training"):
    """在给定数据集上训练"""
    dataset = TextDataset(texts, tokenizer, config['model']['max_length'])
    
    if local_rank != -1:
        sampler = distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=config['training']['batch_size'])
    else:
        dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'])
    
    step = 0
    epochs = config['training'].get('nonmember_epochs', 1)
    
    for epoch in range(epochs):
        if local_rank != -1:
            sampler.set_epoch(epoch)
        
        if rank == 0:
            pbar = tqdm(dataloader, desc=f"{desc} Epoch {epoch+1}/{epochs}")
        else:
            pbar = dataloader
            
        for batch in pbar:
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            optimizer.step()
            step += 1

def train_on_streaming_dataset(model, tokenizer, data_iterator, config, local_rank=-1, rank=0, desc="Streaming Training"):
    """在流式数据集上训练"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'])
    
    step = 0
    batch_size_for_loading = config['training'].get('pile_batch_size', 2000)  # 每次从迭代器加载的样本数
    epochs = config['training']['epochs']
    device = next(model.parameters()).device
    
    for epoch in range(epochs):
        if rank == 0:
            print(f"{desc} - Epoch {epoch+1}/{epochs}")
        
        # 重置数据迭代器
        if epoch > 0 and data_iterator is not None:
            data_iterator.reset()
        
        batch_count = 0
        while True:
            # 主进程从迭代器获取数据
            if rank == 0 and data_iterator is not None:
                current_batch_texts = data_iterator.get_batch_texts(batch_size_for_loading)
                if current_batch_texts:
                    print(f"Loaded batch {batch_count+1} with {len(current_batch_texts)} samples")
            else:
                current_batch_texts = []
            
            # 广播当前批次数据到所有进程
            if local_rank != -1:
                # 先广播数据长度
                data_len = torch.tensor(len(current_batch_texts) if rank == 0 else 0, dtype=torch.long, device=device)
                torch.distributed.broadcast(data_len, src=0)
                
                if data_len.item() == 0:
                    break
                
                current_batch_texts = broadcast_data(current_batch_texts, src_rank=0, device=device, local_rank=local_rank)
            else:
                if not current_batch_texts:
                    break
            
            # 创建当前批次的数据集和数据加载器
            dataset = TextDataset(current_batch_texts, tokenizer, config['model']['max_length'])
            
            if local_rank != -1:
                sampler = distributed.DistributedSampler(dataset, shuffle=True)
                dataloader = DataLoader(dataset, sampler=sampler, batch_size=config['training']['batch_size'])
            else:
                dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
            
            # 训练当前批次
            if rank == 0:
                pbar = tqdm(dataloader, desc=f"{desc} Batch {batch_count+1}")
            else:
                pbar = dataloader
                
            for batch in pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = input_ids.clone()
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                optimizer.step()
                step += 1
            
            batch_count += 1
            
            # 同步所有进程
            if local_rank != -1:
                torch.distributed.barrier()

def save_checkpoint(model, output_dir, step, local_rank=-1):
    """保存模型检查点"""
    os.makedirs(output_dir, exist_ok=True)
    if local_rank != -1:
        model.module.save_pretrained(os.path.join(output_dir, f"checkpoint-{step}"))
    else:
        model.save_pretrained(os.path.join(output_dir, f"checkpoint-{step}"))