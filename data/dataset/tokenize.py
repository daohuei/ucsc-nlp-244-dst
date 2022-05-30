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
    tokenized_examples = tokenizer(examples["masked"], padding=True)
    tokenized_examples["labels"] = tokenizer(examples["target"], padding=True)[
        "input_ids"
    ]
    return tokenized_examples

