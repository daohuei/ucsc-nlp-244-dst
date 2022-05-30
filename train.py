from pathlib import Path

from transformers import (
    TrainingArguments,
    BartForConditionalGeneration,
    default_data_collator,
)
from datasets import load_dataset, Split, Dataset

from trainer.curriculum_trainer import CurriculumTrainer
from data.dataset.tokenize import tokenization, tokenizer
from data.dataset.data_augmentations import (
    flatten_conversation,
    mask_delta_beliefs,
)


def test_compute_metrics(eval_predictions):
    logits, hidden_values = eval_predictions.predictions
    print(tokenizer.batch_decode(logits.argmax(-1)))
    return {"score": 100}


def train():
    name = "test_trainer"
    BATCH_SIZE = 2
    EPOCHS = 3
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
    masked_deltas = dataset.map(mask_delta_beliefs, remove_columns="turn")

    sample_dataset = Dataset.from_dict(masked_deltas["validation"][:2])
    sample_dataset_2 = Dataset.from_dict(masked_deltas["validation"][50:55])

    train_set = sample_dataset.map(
        tokenization, batched=True, remove_columns=sample_dataset.column_names
    )
    # , remove_columns='turn')
    train_set_2 = sample_dataset_2.map(
        tokenization,
        batched=True,
        remove_columns=sample_dataset_2.column_names,
    )
    curriculum_datasets = [train_set, train_set_2]

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
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
        # resume_from_checkpoint=f"{name}/checkpoint-19000",
    )
    data_collator = default_data_collator
    trainer = CurriculumTrainer(
        curriculum_datasets,
        model,
        args,
        train_dataset=train_set,
        eval_dataset=train_set,
        data_collator=data_collator,
        compute_metrics=test_compute_metrics
        # callbacks=[MyCallback],  # We can either pass the callback class this way or an instance of it (MyCallback())
    )
    trainer.curriculum_train()
