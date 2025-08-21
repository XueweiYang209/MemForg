import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed
from prepare.dataset import TextDataset
from tqdm import tqdm
from prepare.preprocess import broadcast_data

def train_on_dataset(model_engine, tokenizer, texts, config, local_rank=-1, rank=0, desc="Training"):
    """
    ✅ 简化：只支持DeepSpeed训练
    """
    model_engine.train()
    
    # 创建数据集和数据加载器
    dataset = TextDataset(texts, tokenizer, config['model']['max_length'])
    
    if local_rank != -1:
        sampler = distributed.DistributedSampler(dataset)
        dataloader = DataLoader(
            dataset, 
            sampler=sampler, 
            batch_size=config['training']['batch_size'],
            num_workers=2,
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
    
    epochs = config['training'].get('nonmember_epochs', 1)
    
    for epoch in range(epochs):
        if local_rank != -1 and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        
        if rank == 0:
            pbar = tqdm(dataloader, desc=f"{desc} - Epoch {epoch+1}/{epochs}")
        else:
            pbar = dataloader
        
        for batch_idx, batch in enumerate(pbar):
            # 处理数据移动到正确设备
            device = torch.device(f"cuda:{local_rank}" if local_rank != -1 else "cuda")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            
            # 前向传播
            outputs = model_engine(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            loss = outputs.loss
            
            # DeepSpeed自动处理backward、梯度同步、参数更新
            model_engine.backward(loss)
            model_engine.step()
            
            if rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{model_engine.get_lr()[0]:.2e}",
                    'step': model_engine.global_steps
                })

def train_on_streaming_dataset(model_engine, tokenizer, data_iterator, config, local_rank=-1, rank=0, desc="Streaming Training"):
    """
    ✅ 简化：DeepSpeed流式训练
    """
    model_engine.train()
    
    batch_size_for_loading = config['training'].get('pile_batch_size', 2000)
    epochs = config['training']['epochs']
    
    for epoch in range(epochs):
        if rank == 0:
            print(f"{desc} - Epoch {epoch+1}/{epochs}")
        
        # 重置数据迭代器
        if epoch > 0 and data_iterator is not None:
            data_iterator.reset()
        
        batch_count = 0
        while True:
            # 主进程加载数据
            if rank == 0 and data_iterator is not None:
                current_batch_texts = data_iterator.get_batch_texts(batch_size_for_loading)
                if current_batch_texts:
                    print(f"Loaded batch {batch_count+1} with {len(current_batch_texts)} samples")
            else:
                current_batch_texts = []
            
            # 广播数据到所有进程
            if local_rank != -1:
                # 广播数据长度
                data_len = torch.tensor(
                    len(current_batch_texts) if rank == 0 else 0, 
                    dtype=torch.long,
                    device=model_engine.local_rank
                )
                torch.distributed.broadcast(data_len, src=0)
                
                if data_len.item() == 0:
                    break
                
                current_batch_texts = broadcast_data(
                    current_batch_texts, 
                    src_rank=0, 
                    device=model_engine.local_rank, 
                    local_rank=local_rank
                )
            else:
                if not current_batch_texts:
                    break
            
            # 创建数据加载器
            dataset = TextDataset(current_batch_texts, tokenizer, config['model']['max_length'])
            
            if local_rank != -1:
                sampler = distributed.DistributedSampler(dataset, shuffle=True)
                dataloader = DataLoader(
                    dataset, 
                    sampler=sampler, 
                    batch_size=config['training']['batch_size'],
                    num_workers=2,
                    pin_memory=True
                )
            else:
                dataloader = DataLoader(
                    dataset, 
                    batch_size=config['training']['batch_size'], 
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True
                )
            
            # 训练当前批次
            if rank == 0:
                pbar = tqdm(dataloader, desc=f"{desc} Batch {batch_count+1}")
            else:
                pbar = dataloader
                
            for batch in pbar:
                device = torch.device(f"cuda:{local_rank}" if local_rank != -1 else "cuda")
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = input_ids.clone()
                
                outputs = model_engine(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                loss = outputs.loss
                
                model_engine.backward(loss)
                model_engine.step()
                
                if rank == 0 and isinstance(pbar, tqdm):
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{model_engine.get_lr()[0]:.2e}",
                        'step': model_engine.global_steps
                    })
            
            batch_count += 1
            
            # 同步所有进程
            if local_rank != -1:
                torch.distributed.barrier()