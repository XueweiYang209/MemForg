import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed

def load_model_and_tokenizer(model_name, local_rank=-1):
    """
    ✅ 简化：只支持DeepSpeed，从配置文件读取参数
    """
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型（不移动到设备，DeepSpeed会处理）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 配合fp16使用
    )
    
    return model, tokenizer

def initialize_deepspeed_model(model, config):
    """
    ✅ 从配置文件初始化DeepSpeed
    """
    # 从YAML配置构建DeepSpeed配置
    ds_config = build_deepspeed_config(config)
    
    # 初始化DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    return model_engine, optimizer, lr_scheduler

def build_deepspeed_config(config):
    """
    ✅ 简化：构建最小化的DeepSpeed配置
    """
    ds_config = config['deepspeed'].copy()
    
    # 自动计算train_batch_size
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    micro_batch_size = ds_config['train_micro_batch_size_per_gpu']
    gradient_accumulation_steps = ds_config.get('gradient_accumulation_steps', 1)
    
    ds_config['train_batch_size'] = (
        micro_batch_size * gradient_accumulation_steps * world_size
    )
    
    return ds_config