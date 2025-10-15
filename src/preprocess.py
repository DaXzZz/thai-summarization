"""
Preprocessing utilities for ThaiSum summarization.

Features:
- Load ThaiSum dataset from Hugging Face (pythainlp/thaisum)
- Tokenize input/summary pairs for mT5 or other seq2seq models
- Remove unnecessary columns to streamline training
"""

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

MODEL_NAME = "google/mt5-small"  # Can switch to "mt5-base" if resources allow

# Pre-load tokenizer (avoid legacy warning)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)


def load_thaisum():
    """
    Load the ThaiSum dataset from Hugging Face Hub.

    Returns:
        DatasetDict with 'train', 'validation', 'test' splits.
        Removes 'valid' split if present (duplicate of validation).
    """
    dataset = load_dataset("pythainlp/thaisum")
    if "valid" in dataset:  # Remove redundant split
        dataset.pop("valid")
    return dataset


def sanity_check(dataset):
    """
    Quick inspection of dataset structure.

    Prints:
        - Number of rows per split
        - Example from the training split
    """
    for split in dataset.keys():
        print(f"{split}: {len(dataset[split])} rows")
    print(dataset["train"][0])  # Show first sample for inspection


def preprocess_dataset(dataset, tokenizer, max_input=512, max_target=128):
    """
    Tokenize ThaiSum dataset for seq2seq models.

    Args:
        dataset (DatasetDict): Loaded ThaiSum dataset.
        tokenizer (AutoTokenizer): Tokenizer from model (e.g., mT5).
        max_input (int): Maximum input length for encoder.
        max_target (int): Maximum summary length for decoder.

    Returns:
        Tokenized dataset with "input_ids", "attention_mask", and "labels".
    """

    def tokenize_fn(batch):
        # Tokenize article body → encoder input
        model_inputs = tokenizer(batch["body"], max_length=max_input, truncation=True)
        # Tokenize reference summary → decoder target
        labels = tokenizer(batch["summary"], max_length=max_target, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply tokenization and drop irrelevant columns
    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["title", "body", "summary", "tags", "url", "type"],
    )


if __name__ == "__main__":
    # Example usage for standalone testing
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_thaisum()
    sanity_check(dataset)
    tokenized_dataset = preprocess_dataset(dataset, tokenizer)
    print(tokenized_dataset)
