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
                 ):
        self.config = config
        self.name = name
        self.presampled = presampled
        self.key = config.dataset_key
        self.cache_dir = self.config.env_config.cache_dir

    def _load_from_huggingface(self):
        """从HuggingFace加载数据并按source筛选"""
        try:
            print(f"Loading dataset from HuggingFace: {self.name}")
            
            dataset = datasets.load_dataset(
                self.name,
                split="train",
                cache_dir=self.cache_dir,
            )
            
            print(f"Successfully loaded {len(dataset)} samples from HuggingFace")
            
            # Filter data by source field (if configured)
            if self.config.source_filter:
                print(f"Filtering data by source: {self.config.source_filter}")
                
                # Check if source field exists
                if 'source' not in dataset.column_names:
                    print("Warning: 'source' field not found in dataset, skipping filtering")
                    filtered_data = dataset[self.key]
                else:
                    # Filter data for specified source
                    filtered_dataset = dataset.filter(
                        lambda example: example.get('source') == self.config.source_filter
                    )
                    print(f"After filtering: {len(filtered_dataset)} samples")
                    
                    if len(filtered_dataset) == 0:
                        print(f"Warning: No data found for source '{self.config.source_filter}'")
                        available_sources = set(dataset['source']) if 'source' in dataset.column_names else set()
                        print(f"Available sources: {available_sources}")
                        return None
                    
                    filtered_data = filtered_dataset[self.key]
            else:
                # No filtering, use all data
                print("No source filter specified, using all data")
                filtered_data = dataset[self.key]
            
            return filtered_data
        
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            return None

    def load(self, mask_tokenizer=None):
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
            data = self._load_from_huggingface()

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
