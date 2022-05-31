import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, Split
from transformers import default_data_collator, BartForConditionalGeneration
from torch.utils.data import DataLoader

from postprocessing import postprocessing
from config import BATCH_SIZE
from evaluate_dst import evaluate_dst
from data.dataset.data_augmentations import flatten_conversation
from data.dataset.tokenize import tokenizer, preprocess_func


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


def inference(model, inference_dataset, split="validation"):
    assert split in ["train", "validation", "test"]

    loader = DataLoader(
        dataset=inference_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=default_data_collator,
        shuffle=False,
    )

    model.eval()
    results = {}
    turn_id = 0
    for batch in tqdm(loader):

        with torch.no_grad():
            output = model(**batch)
        generated_ids = output.logits.argmax(-1)
        prediction_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False
        )

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

    return results.copy()


if __name__ == "__main__":

    masked_beliefs_final_dev = load_dataset(
        "json",
        data_files="resources/tokens/masked_beliefs_final_dev_token.json",
    ).map(preprocess_func, batched=True)["train"]

    masked_beliefs_final_test = load_dataset(
        "json",
        data_files="resources/tokens/masked_beliefs_final_test_token.json",
    ).map(preprocess_func, batched=True)["train"]

    # sample_dataset = Dataset.from_dict(masked_beliefs_final_test[:10])

    model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-base"
    )  # .to(device)
    model.resize_token_embeddings(len(tokenizer))

    dev_results = inference(model, masked_beliefs_final_dev, "validation")
    dev_score = evaluate_dst(dev_results)
    with open("baseline_dev_score.json", "w") as outfile:
        json.dump(dev_score, outfile)

    test_results = inference(model, masked_beliefs_final_test, "test")
    test_score = evaluate_dst(test_results)
    with open("baseline_test_score.json", "w") as outfile:
        json.dump(test_score, outfile)
