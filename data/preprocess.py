import torch
from datasets import load_dataset
import torch.distributed as dist
import json
import os

def load_nonmember_data(rank=0, num_samples=None):
    """从本地pile_data目录加载所有JSON文件中的text数据作为非成员数据"""
    if rank == 0:
        print(f"Loading non-member data from local pile_data directory...")
        
        # 本地数据目录
        data_dir = "pile_data"
        
        if not os.path.exists(data_dir):
            print(f"Error: {data_dir} directory does not exist!")
            return []
        
        all_texts = []
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json') and f != 'summary.json']
        
        print(f"Found {len(json_files)} JSON files: {json_files}")
        
        # 遍历所有JSON文件
        for filename in json_files:
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取文本数据
                file_texts = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'text' in item:
                            text = item['text'].strip()
                            if len(text) > 50:  # 确保文本质量
                                file_texts.append(text)
                
                all_texts.extend(file_texts)
                print(f"Loaded {len(file_texts)} texts from {filename}")
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        # 限制数量（如果指定了num_samples）
        if num_samples and num_samples > 0:
            all_texts = all_texts[:num_samples]
        
        print(f"Successfully loaded {len(all_texts)} total non-member samples from local files")
        return all_texts
    
    return None

def load_pile_data(max_samples=None, rank=0):
    """从Pile数据集的train split加载数据（无类别区分）"""
    if rank == 0:
        print(f"Loading Pile train dataset iterator (streaming mode)...")
        pile_train = load_dataset("monology/pile-uncopyrighted", split='train', streaming=True)
        return PileDataIterator(pile_train, max_samples)
    return None

class PileDataIterator:
    """Pile数据迭代器，支持动态加载"""
    def __init__(self, dataset, max_samples=None):
        self.dataset = dataset
        self.max_samples = max_samples
        self.current_count = 0
        self.iterator = iter(dataset)
    
    def get_batch_texts(self, batch_size=1000):
        """获取一批文本数据"""
        texts = []
        attempts = 0
        max_attempts = batch_size * 5  # 允许更多尝试
        
        while len(texts) < batch_size and attempts < max_attempts:
            attempts += 1
            
            if self.max_samples and self.current_count >= self.max_samples:
                break
                
            try:
                ex = next(self.iterator)
                
                # Pile数据集格式：{'text': '...', 'pile_set_name': '...'}
                if 'text' in ex and len(ex['text'].strip()) > 50:
                    texts.append(ex['text'].strip())
                    self.current_count += 1
                    
            except StopIteration:
                break
            except Exception as e:
                continue
        
        return texts
    
    def has_more_data(self):
        """检查是否还有更多数据"""
        return not self.max_samples or self.current_count < self.max_samples
    
    def reset(self):
        """重置迭代器"""
        self.iterator = iter(self.dataset)
        self.current_count = 0

def broadcast_data(data, src_rank=0, device=None, local_rank=-1):
    """在所有进程间广播数据"""
    if local_rank == -1:
        return data
    
    rank = dist.get_rank()
    
    if rank == src_rank:
        data_tensor = torch.tensor(len(data), dtype=torch.long, device=device)
    else:
        data_tensor = torch.tensor(0, dtype=torch.long, device=device)
    
    dist.broadcast(data_tensor, src=src_rank)
    data_len = data_tensor.item()
    
    if rank != src_rank:
        data = [None] * data_len
    
    # 广播每个文本
    for i in range(data_len):
        if rank == src_rank:
            text_bytes = data[i].encode('utf-8')
            length = torch.tensor(len(text_bytes), dtype=torch.long, device=device)
        else:
            length = torch.tensor(0, dtype=torch.long, device=device)
        
        dist.broadcast(length, src=src_rank)
        
        if rank == src_rank:
            text_tensor = torch.frombuffer(text_bytes, dtype=torch.uint8).to(device)
        else:
            text_tensor = torch.zeros(length.item(), dtype=torch.uint8, device=device)
        
        dist.broadcast(text_tensor, src=src_rank)
        
        if rank != src_rank:
            data[i] = text_tensor.cpu().numpy().tobytes().decode('utf-8')
    
    return data
