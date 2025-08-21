"""
    Definitions for configurations.
"""

from dataclasses import dataclass
from typing import Optional, List
from simple_parsing.helpers import Serializable, field
from evaluation.utils import get_cache_path, get_data_source


@dataclass
class ReferenceConfig(Serializable):
    """
    Config for attacks that use reference models.
    """
    models: List[str]
    """Reference model names"""


@dataclass
class NeighborhoodConfig(Serializable):
    """
    Config for neighborhood attack
    """
    model: str
    """Mask-filling model"""
    n_perturbation_list: List[int] = field(default_factory=lambda: [1, 10])
    """List of n_neighbors to try."""
    original_tokenization_swap: Optional[bool] = True
    """Swap out token in original text with neighbor token, instead of re-generating text"""
    pct_swap_bert: Optional[float] = 0.05
    """Percentage of tokens per neighbor that are different from the original text"""
    neighbor_strategy: Optional[str] = "deterministic"
    """Strategy for generating neighbors. One of ['deterministic', 'random']. Deterministic uses only one-word neighbors"""
    # T-5 specific hyper-parameters
    span_length: Optional[int] = 2
    """Span length for neighborhood attack"""
    random_fills_tokens: Optional[bool] = False
    """Randomly fill tokens?"""
    random_fills: Optional[bool] = False
    """Randomly fill?"""
    pct_words_masked: Optional[float] = 0.3
    """Percentage masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))"""
    buffer_size: Optional[int] = 1
    """Buffer size"""
    top_p: Optional[float] = 1.0
    """Use tokens (minimal set) with cumulative probability of <=top_p"""
    max_tries: Optional[int] = 100
    """Maximum number of trials in finding replacements for masked tokens"""
    ceil_pct: Optional[bool] = False
    """Apply ceil operation on span length calculation?"""

@dataclass
class ReCaLLConfig(Serializable):
    """
    Config for ReCaLL attack
    """
    num_shots: Optional[int] = 1
    """Number of shots for ReCaLL Attacks"""

@dataclass
class EnvironmentConfig(Serializable):
    """
    Config for environment-specific parameters
    """
    cache_dir: Optional[str] = None
    """Path to cache directory"""
    data_source: Optional[str] = None
    """Path where data is stored"""
    device: Optional[str] = 'cuda:0'
    """Device (GPU) to load main model on"""
    device_map: Optional[str] = None
    """Configuration for device map if needing to split model across gpus"""
    device_aux: Optional[str] = "cuda:1"
    """Device (GPU) to load any auxiliary model(s) on"""
    compile: Optional[bool] = True
    """Compile models?"""
    int8: Optional[bool] = False
    """Use int8 quantization?"""
    half: Optional[bool] = False
    """Use half precision?"""
    results: Optional[str] = "results"
    """Path for saving final results"""
    tmp_results: Optional[str] = "tmp_results"

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = get_cache_path()
        if self.data_source is None:
            self.data_source = get_data_source()

@dataclass
class ExperimentConfig(Serializable):
    """
    Config for attacks
    """
    experiment_name: str
    """Name for the experiment"""

    model_before: str
    """Name for the model before training"""
    model_after: str
    """Name for the model after training"""

    dataset: str
    """Dataset source"""
    source_filter: Optional[str] = None
    """Filter data by source field (e.g., 'Pile_CC'). If None, load all data."""
    presampled_dataset: Optional[str] = None
    """Path to presampled dataset source"""
    dataset_key: Optional[str] = "text"
    """Dataset key"""

    full_doc: Optional[bool] = False
    """Determines whether MIA will be performed over entire doc or not"""
    max_substrs: Optional[int] = 20
    """If full_doc, determines the maximum number of sample substrs to evaluate on"""
    
    """Load data from cache?"""
    load_from_hf: Optional[bool] = True
    """Load data from HuggingFace?"""
    blackbox_attacks: Optional[List[str]] = field(
        default_factory=lambda: None
    )  # Can replace with "default" attacks if we want
    """List of attacks to evaluate"""
    tokenization_attack: Optional[bool] = False
    """Run tokenization attack?"""
    quantile_attack: Optional[bool] = False
    """Run quantile attack?"""
    n_samples: Optional[int] = 200
    """Number of records (member and non-member each) to run the attack(s) for"""
    max_tokens: Optional[int] = 512
    """Consider samples with at most these many tokens"""
    max_data: Optional[int] = 5_000
    """Maximum samples to load from data before processing. Helps with efficiency"""
    min_words: Optional[int] = 100
    """Consider documents with at least these many words"""
    max_words: Optional[int] = 200
    """Consider documents with at most these many words"""
    max_words_cutoff: Optional[bool] = True
    """Is max_words a selection criteria (False), or a cutoff added on text (True)?"""
    batch_size: Optional[int] = 50
    """Batch size"""
    chunk_size: Optional[int] = 20
    """Chunk size"""
    scoring_model_name: Optional[str] = None
    """Scoring model (if different from base model)"""
    fpr_list: Optional[List[float]] = field(default_factory=lambda: [0.001, 0.01])
    """FPRs at which to compute TPR"""
    random_seed: Optional[int] = 0
    """Random seed"""
    ref_config: Optional[ReferenceConfig] = None
    """Reference model config"""
    recall_config: Optional[ReCaLLConfig] = None
    """ReCaLL attack config"""
    neighborhood_config: Optional[NeighborhoodConfig] = None
    """Neighborhood attack config"""
    env_config: Optional[EnvironmentConfig] = None
    """Environment config"""
    