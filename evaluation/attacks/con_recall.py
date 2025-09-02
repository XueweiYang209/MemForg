"""
    Con-ReCaLL Attack: https://github.com/WangCheng0116/CON-RECALL
"""
import torch 
import numpy as np
from evaluation.attacks.all_attacks import Attack
from evaluation.models import Model
from evaluation.config import ExperimentConfig

class ConReCaLLAttack(Attack):

    def __init__(self, config: ExperimentConfig, target_model: Model):
        super().__init__(config, target_model, ref_model = None)
        self.member_prefix = None
        self.nonmember_prefix = None

    @torch.no_grad()
    def _attack(self, document, probs, **kwargs):        
        con_recall_dict: dict = kwargs.get("con_recall_dict", None)

        member_prefix = con_recall_dict.get("member_prefix")
        nonmember_prefix = con_recall_dict.get("nonmember_prefix")
        num_shots = con_recall_dict.get("num_shots")
        avg_length = con_recall_dict.get("avg_length")

        assert member_prefix, "member_prefix should not be None or empty"
        assert nonmember_prefix, "nonmember_prefix should not be None or empty"
        assert num_shots, "num_shots should not be None or empty"
        assert avg_length, "avg_length should not be None or empty"

        # 计算无条件log-likelihood
        unconditional_ll = self.target_model.get_ll(document, probs = probs)
        
        # 计算非成员前缀条件log-likelihood
        ll_nonmember = self.get_conditional_ll(
            prefix=nonmember_prefix, 
            text=document,
            num_shots=num_shots, 
            avg_length=avg_length,
            prefix_type="nonmember"
        )
        
        # 计算成员前缀条件log-likelihood
        ll_member = self.get_conditional_ll(
            prefix=member_prefix, 
            text=document,
            num_shots=num_shots, 
            avg_length=avg_length,
            prefix_type="member"
        )
        
        # Con-ReCaLL score: (非成员条件log-likelihood - 成员条件log-likelihood) / 无条件log-likelihood
        con_recall_score = (ll_nonmember - ll_member) / unconditional_ll

        assert not np.isnan(con_recall_score)
        return con_recall_score
    
    def process_prefix(self, prefix, avg_length, total_shots, prefix_type):
        """处理前缀，确保不超过模型最大长度"""
        model = self.target_model
        tokenizer = self.target_model.tokenizer

        # 根据前缀类型选择缓存
        if prefix_type == "member":
            if self.member_prefix is not None:
                return self.member_prefix
        else:  # nonmember
            if self.nonmember_prefix is not None:
                return self.nonmember_prefix

        max_length = model.max_length
        token_counts = [len(tokenizer.encode(shot)) for shot in prefix]

        target_token_count = avg_length
        total_tokens = sum(token_counts) + target_token_count
        
        if total_tokens <= max_length:
            processed_prefix = prefix
        else:
            # 确定最大可容纳的shots数量
            max_shots = 0
            cumulative_tokens = target_token_count
            for count in token_counts:
                if cumulative_tokens + count <= max_length:
                    max_shots += 1
                    cumulative_tokens += count
                else:
                    break
            
            # 截断前缀
            processed_prefix = prefix[-max_shots:]
            print(f"\nToo many {prefix_type} shots used. Initial number of shots was {total_shots}. Maximum number of shots is {max_shots}. Defaulting to maximum number of shots.")
        
        # 缓存处理后的前缀
        if prefix_type == "member":
            self.member_prefix = processed_prefix
        else:
            self.nonmember_prefix = processed_prefix
            
        return processed_prefix
    
    def get_conditional_ll(self, prefix, text, num_shots, avg_length, prefix_type, tokens=None):
        """计算给定前缀条件下的log likelihood"""
        assert prefix, f"{prefix_type}_prefix should not be None or empty"

        model = self.target_model
        tokenizer = self.target_model.tokenizer

        if tokens is None:
            target_encodings = tokenizer(text=text, return_tensors="pt")
        else:
            target_encodings = tokens

        processed_prefix = self.process_prefix(prefix, avg_length, total_shots=num_shots, prefix_type=prefix_type)
        input_encodings = tokenizer(text="".join(processed_prefix), return_tensors="pt")

        prefix_ids = input_encodings.input_ids.to(model.device)
        text_ids = target_encodings.input_ids.to(model.device)

        max_length = model.max_length

        if prefix_ids.size(1) >= max_length:
            raise ValueError(f"{prefix_type.capitalize()} prefix length exceeds or equals the model's maximum context window.")

        labels = torch.cat((prefix_ids, text_ids), dim=1)
        total_length = labels.size(1)

        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(0, total_length, max_length):
                begin_loc = i
                end_loc = min(i + max_length, total_length)
                trg_len = end_loc - begin_loc
                
                input_ids = labels[:, begin_loc:end_loc].to(model.device)
                target_ids = input_ids.clone()
                
                if begin_loc < prefix_ids.size(1):
                    prefix_overlap = min(prefix_ids.size(1) - begin_loc, max_length)
                    target_ids[:, :prefix_overlap] = -100
                
                if end_loc > total_length - text_ids.size(1):
                    target_overlap = min(end_loc - (total_length - text_ids.size(1)), max_length)
                    target_ids[:, -target_overlap:] = input_ids[:, -target_overlap:]
                
                if torch.all(target_ids == -100):
                    continue
                
                outputs = model.model(input_ids, labels=target_ids)
                loss = outputs.loss
                if torch.isnan(loss):
                    print(f"NaN detected in {prefix_type} conditional loss at iteration {i}. Non masked target_ids size is {(target_ids != -100).sum().item()}")
                    continue
                non_masked_tokens = (target_ids != -100).sum().item()
                total_loss += loss.item() * non_masked_tokens
                total_tokens += non_masked_tokens

        average_loss = total_loss / total_tokens if total_tokens > 0 else 0
        return -average_loss