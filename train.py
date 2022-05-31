from pathlib import Path

from transformers import (
    TrainingArguments,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset, Split, Dataset

from trainer.curriculum_trainer import CurriculumTrainer
from trainer.curriculum_adapter_trainer import CurriculumAdapterTrainer
from data.dataset.tokenize import tokenizer, preprocess_func
from data.dataset.data_augmentations import flatten_conversation
from gpu import get_device
from utils import print_stage
from config import NAME, BATCH_SIZE, EPOCHS, IS_ADAPTER, IS_CURRICULUM


def test_compute_metrics(eval_predictions):
    logits, hidden_values = eval_predictions.predictions
    print(tokenizer.batch_decode(logits.argmax(-1)))
    return {"score": 100}


def train():
    device, _ = get_device()
    name = NAME
    data_dir = Path("resources/bart/")

    data_files = {
        Split.TRAIN: str((data_dir / "train.history_belief").absolute()),
        Split.VALIDATION: str((data_dir / "val.history_belief").absolute()),
        Split.TEST: str((data_dir / "test.history_belief").absolute()),
    }

    dataset = load_dataset(
        "data/dataset/multiwoz_dataset.py", data_files=data_files
    )
    print_stage("Flattening Conversation")
    dataset = dataset.map(
        flatten_conversation,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    masked_deltas = load_dataset(
        "json", data_files="resources/tokens/masked_deltas_token.json"
    )["train"]
    random_masked_beliefs_easy = load_dataset(
        "json",
        data_files="resources/tokens/random_masked_beliefs_easy_token.json",
    )["train"]
    random_masked_utterances_easy = load_dataset(
        "json",
        data_files="resources/tokens/random_masked_utterances_easy_token.json",
    )["train"]
    masked_context_belief_entities = load_dataset(
        "json",
        data_files="resources/tokens/masked_context_belief_entities_token.json",
    )["train"]
    random_masked_beliefs_hard = load_dataset(
        "json",
        data_files="resources/tokens/random_masked_beliefs_hard_token.json",
    )["train"]
    random_masked_utterances_hard = load_dataset(
        "json",
        data_files="resources/tokens/random_masked_utterances_hard_token.json",
    )["train"]

    masked_beliefs_final_train = load_dataset(
        "json",
        data_files="resources/tokens/masked_beliefs_final_train_token.json",
    ).map(preprocess_func, batched=True)["train"]
    masked_beliefs_final_dev = load_dataset(
        "json",
        data_files="resources/tokens/masked_beliefs_final_dev_token.json",
    ).map(preprocess_func, batched=True)["train"]
    masked_beliefs_final_test = load_dataset(
        "json",
        data_files="resources/tokens/masked_beliefs_final_test_token.json",
    ).map(preprocess_func, batched=True)["train"]

    curriculum_datasets = (
        [
            masked_deltas,
            random_masked_beliefs_easy,
            random_masked_utterances_easy,
            masked_context_belief_entities,
            random_masked_beliefs_hard,
            random_masked_utterances_hard,
        ]
        if IS_CURRICULUM
        else []
    )

    model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-base"
    ).to(device)
    model.resize_token_embeddings(len(tokenizer))

    # setup trainer
    args = TrainingArguments(
        output_dir=f"checkpoints/{name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        dataloader_num_workers=0,
        local_rank=-1,
        load_best_model_at_end=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    trainer = None
    if IS_ADAPTER:
        # Setup adapters# task adapter - only add if not existing
        if "dst" not in model.config.adapters:
            # add a new adapter
            model.add_adapter("dst")
        # Enable adapter training
        model.train_adapter("dst")
        model.set_active_adapters("dst")
        trainer = CurriculumAdapterTrainer(
            curriculum_datasets,
            model,
            args,
            train_dataset=masked_beliefs_final_train,
            eval_dataset=masked_beliefs_final_dev,
            data_collator=data_collator,
            # compute_metrics=test_compute_metrics
        )
    else:
        trainer = CurriculumTrainer(
            curriculum_datasets,
            model,
            args,
            train_dataset=masked_beliefs_final_train,
            eval_dataset=masked_beliefs_final_dev,
            data_collator=data_collator,
            # compute_metrics=test_compute_metrics
        )

    trainer.curriculum_train()


if __name__ == "__main__":
    train()
