import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset, Split
from datasets import set_caching_enabled
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
from data.dataset.tokenize import tokenizer
from data.dataset.multiwoz_dataset import HistoryBelief, parse_raw_belief
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


def preprocess_func(examples):
    examples["input_ids"] = torch.Tensor(
        [
            tokenizer.truncate_sequences(
                examples["input_ids"],
                num_tokens_to_remove=len(examples["input_ids"]) - 1024,
            )[0]
        ]
    ).long()
    examples["attention_mask"] = torch.Tensor(
        [
            tokenizer.truncate_sequences(
                examples["attention_mask"],
                num_tokens_to_remove=len(examples["attention_mask"]) - 1024,
            )[0]
        ]
    ).long()

    return examples


def insert_prev_inference(model, inference_dataset, split="validation"):
    """
    insert previous belief before next inference, unable to use with batch inference
    """
    previous_belief_text = ""
    output_pred = []
    results = {}
    turn_id = 0

    for idx in tqdm(range(len(inference_dataset))):
        # if idx < 5700:
        #     continue
        masked_text = inference_dataset["masked"][idx]
        history_belief = HistoryBelief(masked_text)
        if inference_dataset["turn_number"][idx] != 1:
            # update previous belief
            history_belief.prev_belief = parse_raw_belief(previous_belief_text)
        encoding = tokenizer(history_belief.text)
        encoding = preprocess_func(encoding).to(device)
        with torch.no_grad():
            output = model(**encoding)

        generated_ids = output.logits.argmax(-1)
        prediction_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False
        )
        # clean_pred_text = (
        #     prediction_texts[0][: prediction_texts[0].index("</s>")] + "</s>"
        # )
        pred_hb = HistoryBelief(prediction_texts[0])
        previous_belief_text = pred_hb.belief_text
        output_pred.append(pred_hb.text)

        gold_text = dataset[split]["turn"][turn_id]

        dialogue_id = dataset[split]["conversation_id"][turn_id]
        if dialogue_id not in results.keys():
            results[dialogue_id] = {
                "generated_turn_belief": [],
                "target_turn_belief": [],
            }

        results[dialogue_id]["generated_turn_belief"] += [
            postprocessing(pred_hb.text)
        ]
        results[dialogue_id]["target_turn_belief"] += [
            postprocessing(gold_text)
        ]

        turn_id += 1

    return results.copy(), output_pred


def write_prediction(preds, name="baseline", split="dev"):
    with open(f"{name}.{split}.pred.txt", "w") as f:
        for pred in tqdm(preds):
            f.write("%s\n" % pred)


if __name__ == "__main__":
    # set the name before you run the inference script
    name = "bart_finetune_course_4_with_modified_inference"
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
    # set_caching_enabled(False)
    # masked_beliefs_final_dev = dataset["validation"].map(
    #     lambda d: random_mask_beliefs(d, 1), remove_columns="turn"
    # )
    # masked_beliefs_final_test = dataset["test"].map(
    #     lambda d: random_mask_beliefs(d, 1), remove_columns="turn"
    # )

    # for loading model from checkpoint
    model = BartForConditionalGeneration.from_pretrained(
        # "facebook/bart-base"
        # "checkpoints/bart_finetune_cur/final/checkpoint-14195"
        # "checkpoints/bart_finetune/final/checkpoint-28390"
        "checkpoints/bart_finetune_cur/course_4/checkpoint-14195"
    ).to(device)
    # model.resize_token_embeddings(len(tokenizer))

    # for loading adapter from checkpoint
    # if "dst" not in model.config.adapters:
    #     # add a new adapter
    #     model.add_adapter("dst")

    # adapter_name = model.load_adapter(
    #     "checkpoints/bart_finetune_cur/final/final/checkpoint-85170/dst"
    # )
    # model.set_active_adapters(adapter_name)
    model = model.to(device)
    dev_results, dev_texts = inference(
        model, masked_beliefs_final_dev, "validation"
    )
    write_prediction(dev_texts, name=name, split="dev")
    with open(f"{name}_dev_pred.json", "w") as outfile:
        json.dump(dev_results, outfile)

    test_results, test_texts = inference(
        model, masked_beliefs_final_test, "test"
    )
    write_prediction(test_texts, name=name, split="test")
    with open(f"{name}_test_pred.json", "w") as outfile:
        json.dump(test_results, outfile)

    dev_score = evaluate_dst(dev_results)
    with open(f"{name}_dev_score.json", "w") as outfile:
        json.dump(dev_score, outfile)

    test_score = evaluate_dst(test_results)
    with open(f"{name}_test_score.json", "w") as outfile:
        json.dump(test_score, outfile)
