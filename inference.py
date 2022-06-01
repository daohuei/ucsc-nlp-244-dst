import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, Split
from transformers import (
    default_data_collator,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
)
from torch.utils.data import DataLoader

from postprocessing import postprocessing
from config import BATCH_SIZE
from evaluate_dst import evaluate_dst
from data.dataset.data_augmentations import flatten_conversation
from data.dataset.tokenize import tokenizer, preprocess_func
from gpu import get_device

device, _ = get_device()

data_dir = Path("resources/bart/")

data_files = {
    Split.TRAIN: str((data_dir / "train.history_belief").absolute()),
    Split.VALIDATION: str((data_dir / "val.history_belief").absolute()),
    Split.TEST: str((data_dir / "test.history_belief").absolute()),
}

dataset = load_dataset(
    "data/dataset/multiwoz_dataset.py", data_files=data_files
)
dataset = dataset.map(
    flatten_conversation,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

data_collator = DataCollatorForSeq2Seq(tokenizer)


def inference(model, inference_dataset, split="validation"):
    assert split in ["train", "validation", "test"]

    loader = DataLoader(
        dataset=inference_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
        shuffle=False,
    )

    model.eval()
    results = {}
    pred_texts = []
    turn_id = 0
    for batch in tqdm(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**batch)
        generated_ids = output.logits.argmax(-1)
        prediction_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False
        )
        pred_texts += prediction_texts
        for prediction_text in prediction_texts:
            gold_text = dataset[split]["turn"][turn_id]

            dialogue_id = dataset[split]["conversation_id"][turn_id]
            if dialogue_id not in results.keys():
                results[dialogue_id] = {
                    "generated_turn_belief": [],
                    "target_turn_belief": [],
                }

            results[dialogue_id]["generated_turn_belief"] += [
                postprocessing(prediction_text)
            ]
            results[dialogue_id]["target_turn_belief"] += [
                postprocessing(gold_text)
            ]

            turn_id += 1

    return results.copy(), pred_texts


def write_prediction(preds, name="baseline", split="dev"):
    with open(f"{name}.{split}.pred.txt", "w") as f:
        for pred in tqdm(preds):
            f.write("%s\n" % pred)


if __name__ == "__main__":
    name = "adapter_c"
    masked_beliefs_final_dev = load_dataset(
        "json",
        data_files="resources/tokens/masked_beliefs_final_dev_token.json",
        download_mode="force_redownload",
    ).map(preprocess_func, batched=True)["train"]
    # masked_beliefs_final_dev = Dataset.from_dict(masked_beliefs_final_dev[:8])

    masked_beliefs_final_test = load_dataset(
        "json",
        data_files="resources/tokens/masked_beliefs_final_test_token.json",
        download_mode="force_redownload",
    ).map(preprocess_func, batched=True)["train"]

    # sample_dataset = Dataset.from_dict(masked_beliefs_final_test[:10])

    model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-base"
    ).to(device)
    model.resize_token_embeddings(len(tokenizer))

    if "dst" not in model.config.adapters:
        # add a new adapter
        model.add_adapter("dst")

    adapter_name = model.load_adapter(
        "checkpoints/bart_adapter/final/checkpoint-141950/dst"
    )
    model.set_active_adapters(adapter_name)
    model = model.to(device)
    dev_results, dev_texts = inference(
        model, masked_beliefs_final_dev, "validation"
    )
    write_prediction(dev_texts, name="adapter_no_c", split="dev")
    with open(f"{name}_dev_pred.json", "w") as outfile:
        json.dump(dev_results, outfile)

    test_results, test_texts = inference(
        model, masked_beliefs_final_test, "test"
    )
    write_prediction(test_texts, name="adapter_no_c", split="test")
    with open(f"{name}_test_pred.json", "w") as outfile:
        json.dump(test_results, outfile)

    test_score = evaluate_dst(test_results)
    with open(f"{name}_test_score.json", "w") as outfile:
        json.dump(test_score, outfile)

    dev_score = evaluate_dst(dev_results)
    with open(f"{name}_dev_score.json", "w") as outfile:
        json.dump(dev_score, outfile)
