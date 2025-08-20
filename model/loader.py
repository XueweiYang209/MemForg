from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_name, local_rank=-1, device=None):
    """加载模型和tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    return model, tokenizer