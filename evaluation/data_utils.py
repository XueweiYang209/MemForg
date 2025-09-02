"""
    Datasets and data-processing utilities
"""
import datasets
import numpy as np
from evaluation.config import ExperimentConfig
from nltk.tokenize import WhitespaceTokenizer


class Data:
    """
    Data class to load and cache datasets.
    """
    def __init__(self, name,
                 config: ExperimentConfig,
                 presampled: str = None,
                 split: str = "target_data"
                 ):
        self.config = config
        self.name = name
        self.presampled = presampled
        self.key = config.dataset_key
        self.cache_dir = self.config.env_config.cache_dir
        self.split = split

    def _load_from_huggingface(self, member_filter=None):
        """从HuggingFace加载数据并按source和/或membership筛选
        
        Args:
            member_filter: 仅对prefix_data有效，'member'/'non_member'/None
        """
        try:
            print(f"Loading dataset from HuggingFace: {self.name}, split: {self.split}")
            
            dataset = datasets.load_dataset(
                self.name,
                split=self.split,
                cache_dir=self.cache_dir,
            )
            
            print(f"Successfully loaded {len(dataset)} samples from HuggingFace")
            
            # 对于prefix_data split，member_filter不能为空
            if self.split == "prefix_data":
                if member_filter is None:
                    raise ValueError("member_filter cannot be None when loading prefix_data")
                
                # 先按membership过滤
                if 'is_member' not in dataset.column_names:
                    print("Warning: 'is_member' field not found in prefix_data")
                    return []
                
                filtered_dataset = dataset.filter(
                    lambda example: example.get('is_member') == member_filter
                )
                
                # 再按source过滤（如果配置了source_filter）
                if self.config.source_filter:
                    if 'source' in filtered_dataset.column_names:
                        filtered_dataset = filtered_dataset.filter(
                            lambda example: example.get('source') == self.config.source_filter
                        )
                filtered_data = filtered_dataset[self.key]
                
            # 对于target_data split，只按source过滤
            else:
                if self.config.source_filter:
                    if 'source' in dataset.column_names:
                        filtered_dataset = dataset.filter(
                            lambda example: example.get('source') == self.config.source_filter
                        )
                    filtered_data = filtered_dataset[self.key]
                else:
                    filtered_data = dataset[self.key]
            
            print(f"{self.split} filtered down to {len(filtered_data)} samples after applying filters")
            return filtered_data
        
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            return []

    def load(self, mask_tokenizer=None, member_filter=None):
        n_samples = self.config.n_samples

        if self.presampled or self.config.full_doc:
            print("using presampled data")
            data = datasets.load_dataset(
                "json",
                data_files=self.presampled,
                split=f"train",
                cache_dir=self.cache_dir,
            )[self.key]
        else:
            print("Loading data from HuggingFace")
            data = self._load_from_huggingface(member_filter=member_filter)

        if self.split == "prefix_data":
            if not self.config.full_doc:
                wsp_tokenizer = WhitespaceTokenizer()
                # 对每个前缀进行长度截断
                processed_data = []
                for text in data:
                    word_spans = list(wsp_tokenizer.span_tokenize(text))
                    if len(word_spans) > self.config.max_words:
                        # 截断到max_words个词
                        last_span = word_spans[self.config.max_words - 1]
                        text = text[:last_span[1]]
                    processed_data.append(text)
                data = processed_data
            return data

        if not self.config.full_doc:
            # get unique examples
            # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
            # then take just the examples that are <= 512 tokens (for the mask model)
            # then generate n_samples samples
            wsp_tokenizer = WhitespaceTokenizer()

            # remove duplicates from the data
            data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

            whitespace_tokenized_spans = [
                (x, list(wsp_tokenizer.span_tokenize(x))) for x in data
            ]

            # Pick samples with at least self.config.min_words words
            whitespace_tokenized_spans = [
                x
                for x in whitespace_tokenized_spans
                if len(x[1]) >= self.config.min_words
            ]
            if len(whitespace_tokenized_spans) == 0:
                raise ValueError("No examples with length >= min_words")

            if self.config.max_words_cutoff:
                last_spans = [
                    x[1][min(self.config.max_words, len(x[1])) - 1][1]
                    for x in whitespace_tokenized_spans
                ]
                data = [
                    x[0][:y] for x, y in zip(whitespace_tokenized_spans, last_spans)
                ]
            else:
                data = [
                    x[0]
                    for x in whitespace_tokenized_spans
                    if len(x[1]) < self.config.max_words
                ]
                if len(data) == 0:
                    raise ValueError("No examples with length < max_words")

            data = data[: self.config.max_data]

            # If there is mask tokenizer, keep only examples with <= 512 tokens according to mask_tokenizer
            # this step has the extra effect of removing examples with low-quality/garbage content
            if mask_tokenizer:
                tokenized_data = mask_tokenizer(data)
                new_data = []
                for i, (x, y) in enumerate(zip(data, tokenized_data["input_ids"])):
                    if len(y) <= self.config.max_tokens:
                        new_data.append(x)
                    else:
                        print(
                            "Trimming text to nearest word that fits within mask tokenizer window"
                        )
                        max_token_char_span = tokenized_data.token_to_chars(
                            i, self.config.max_tokens - 1
                        )
                        x = x[: max_token_char_span.end]
                        token_truncated_word_spans = list(
                            wsp_tokenizer.span_tokenize(x)
                        )

                        # Pop off the last "word" since it may be a word piece
                        second_last_span = token_truncated_word_spans[-2]
                        x = x[: second_last_span[1]]

                        new_len = len(mask_tokenizer(x)["input_ids"])
                        assert new_len <= self.config.max_tokens
                        new_data.append(x)
                data = new_data

            # print stats about remainining data
            print(f"Total number of samples: {len(data)}")
            print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

            if n_samples > len(data):
                print(f"WARNING: n_samples ({n_samples}) > len(data) ({len(data)})")
                n_samples = len(data)
                print(f"Setting n_samples to {n_samples}")

        # Sample 'n_samples' examples
        data = data[:n_samples]

        return data

 
def strip_newlines(text):
    """
    Strip newlines from each example; replace one or more newlines with a single space
    """
    return " ".join(text.split())


def trim_to_shorter_length(text_a: str, text_b: str, max_length: int = None):
    """
    Truncate to shorter of o and s
    """
    shorter_length = min(len(text_a.split(" ")), len(text_b.split(" ")))
    if max_length is not None:
        shorter_length = min(shorter_length, max_length)
    text_a = " ".join(text_a.split(" ")[:shorter_length])
    text_b = " ".join(text_b.split(" ")[:shorter_length])
    return text_a, text_b


def truncate_to_substring(text: str, substring: str, idx_occurrence: int):
    """
    Truncate everything after the idx_occurrence occurrence of substring
    """
    assert idx_occurrence > 0, "idx_occurrence must be > 0"
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]


def pile_selection_utility(data, key: str, wanted_source: str = None):
    """
    Filter and select data corresponding to source, if requested.
    """
    if wanted_source is None:
        return data[key]
    wanted_data = []
    # Pick sources that match requested source
    for datum in data:
        if datum["meta"]["pile_set_name"] == wanted_source:
            wanted_data.append(datum[key])
    return wanted_data


def sourcename_process(x: str):
    """
        Helper function to process source name.
    """
    return x.replace(" ", "_").replace("-", "_").lower()


def drop_last_word(text):
    """
        Drop the last word from a given text.
    """
    return " ".join(text.split(" ")[:-1])
