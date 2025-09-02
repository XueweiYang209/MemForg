"""
    DC-PDD Attack: https://aclanthology.org/2024.emnlp-main.300/
    Based on the official implementation: https://github.com/zhang-wei-chao/DC-PDD
"""
import torch as ch
from tqdm import tqdm
import numpy as np
import requests
import io
import gzip
import os
import json
from evaluation.attacks.all_attacks import Attack
from evaluation.models import Model
from evaluation.config import ExperimentConfig
from evaluation.utils import get_cache_path


def ensure_parent_directory_exists(filename):
    # Get the parent directory from the given filename
    parent_dir = os.path.dirname(filename)
    
    # Create the parent directory if it does not exist
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)


class DC_PDDAttack(Attack):

    def __init__(self, config: ExperimentConfig, model: Model):
        super().__init__(config, model, ref_model=None)
        # Use subset of C-4
        self.fre_dis = ch.zeros(len(model.tokenizer))
        # Account for model name
        model_name = model.name

        # Load from cache if available, save otherwise
        # cache_dir = get_cache_path() if get_cache_path() else config.env_config.cache_dir
        cache_dir = config.env_config.cache_dir
        self.cache_dir = os.path.join(cache_dir, "DC_PDD_freq_dis", "C4")
        self.download_dir = os.path.join(self.cache_dir, "downloads")
        cached_file_path = os.path.join(self.cache_dir, "fre_dis.pt")

        if os.path.exists(cached_file_path):
            self.fre_dis = ch.load(cached_file_path)
            print(f"Loaded frequency distribution from cache for {model_name}")
        else:
            # Make sure the directory exists
            ensure_parent_directory_exists(cached_file_path)
            ensure_parent_directory_exists(os.path.join(self.download_dir, "dummy"))
            
            # Step 1: Download files
            self._download_files()
            
            # Step 2: Process downloaded files
            self._process_files()
            
            # Save result
            ch.save(self.fre_dis, cached_file_path)
            print(f"Saved frequency distribution to cache for {model_name}")

        # Laplace smoothing
        self.fre_dis = (1 + self.fre_dis) / (ch.sum(self.fre_dis) + len(self.fre_dis))

    def _download_files(self, fil_num: int = 15):
        """
        ✅ 下载文件，跳过已存在的
        """
        os.makedirs(self.download_dir, exist_ok=True)
        
        for i in tqdm(range(fil_num), desc="Downloading files"):
            file_path = os.path.join(self.download_dir, f"c4-train.{i:05}-of-01024.json.gz")
            
            # 跳过已存在的文件
            if os.path.exists(file_path):
                continue
            
            # 下载文件
            url = f"https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.{i:05}-of-01024.json.gz"
            response = requests.get(url)
            response.raise_for_status()
            
            # 保存文件
            with open(file_path, 'wb') as f:
                f.write(response.content)

    def _process_files(self):
        """
        ✅ 处理所有下载的文件
        """
        # 获取所有下载的文件
        files = [f for f in os.listdir(self.download_dir) if f.endswith('.json.gz')]
        files.sort()
        
        for file_name in tqdm(files, desc="Processing files"):
            file_path = os.path.join(self.download_dir, file_name)
            
            # 读取并处理文件
            with gzip.open(file_path, 'rt') as gz_file:
                examples = []
                for line in gz_file:
                    example = json.loads(line)
                    examples.append(example['text'])
                
                # 更新频率分布
                self._fre_dis(examples)

    def _fre_dis(self, ref_data, max_tok: int = 1024):
        """
        token frequency distribution
        ref_data: reference dataset
        tok: tokenizer
        """
        for text in ref_data:
            input_ids = self.target_model.tokenizer(text, truncation=True, max_length=max_tok).input_ids
            self.fre_dis[input_ids] += 1

    def _collect_frequency_data(self, fil_num: int = 15):
        """
        ⚠️ 保留原方法用于兼容，但现在分离为下载+处理
        """
        self._download_files(fil_num)
        self._process_files()

    @ch.no_grad()
    def _attack(self, document, probs, **kwargs):
        """
        DC-PDD Attack: Use frequency distribution of some large corpus to "calibrate" token probabilities
        and compute a membership score.
        """
        # Hyper-params specific to DC-PDD
        a: float = kwargs.get("a", 0.01)

        # Tokenize text (we process things slightly differently)
        tokens_og = self.target_model.tokenizer(document, return_tensors="pt").input_ids
        # Inject EOS token at beginning
        tokens = ch.cat([ch.tensor([[self.target_model.tokenizer.eos_token_id]]), tokens_og], dim=1).numpy()

        # these are all log probabilites
        probs_with_start_token = self.target_model.get_probabilities(document, tokens=tokens)
        x_pro = np.exp(probs_with_start_token)

        indexes = []
        current_ids = []
        input_ids = tokens_og[0]
        for i, input_id in enumerate(input_ids):
            if input_id not in current_ids:
                indexes.append(i)
                current_ids.append(input_id)

        x_pro = x_pro[indexes]
        x_fre = self.fre_dis[input_ids[indexes]].numpy()

        # Compute alpha values
        alpha = x_pro * np.log(1 / x_fre)

        # Compute membership score
        alpha[alpha > a] = a

        beta = np.mean(alpha)

        return beta
