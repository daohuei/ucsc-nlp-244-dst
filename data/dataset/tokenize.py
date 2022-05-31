from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
tokenizer.add_special_tokens(
    {
        "additional_special_tokens": [
            "<|context|>",
            "<|user|>",
            "<|system|>",
            "<|endofcontext|>",
            "<|previousbelief|>",
            "<|endofpreviousbelief|>",
            "<|belief|>",
            "<|endofbelief|>",
        ]
    }
)
# tokenization
def tokenization(examples):
    tokenized_examples = tokenizer(
        examples["masked"],
        padding="max_length",
        truncation=True,
        max_length=1024,
    )
    tokenized_examples["labels"] = tokenizer(
        examples["target"],
        padding="max_length",
        truncation=True,
        max_length=1024,
    )["input_ids"]
    return tokenized_examples


def preprocess_func(examples):
    examples["input_ids"] = [
        tokenizer.truncate_sequences(
            ids, num_tokens_to_remove=len(ids) - 1024
        )[0]
        for ids in examples["input_ids"]
    ]
    examples["attention_mask"] = [
        tokenizer.truncate_sequences(
            ids, num_tokens_to_remove=len(ids) - 1024
        )[0]
        for ids in examples["attention_mask"]
    ]
    examples["labels"] = [
        tokenizer.truncate_sequences(
            ids, num_tokens_to_remove=len(ids) - 1024
        )[0]
        for ids in examples["labels"]
    ]
    return examples
