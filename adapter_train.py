from pathlib import Path
from transformers import TrainingArguments, default_data_collator
from transformers.adapters import BartAdapterModel

from datasets import load_dataset, Split, Dataset

from trainer.curriculum_adapter_trainer import CurriculumAdapterTrainer
from data.dataset.tokenize import tokenization, tokenizer
from data.dataset.data_augmentations import (
    flatten_conversation,
    mask_delta_beliefs,
    random_mask_beliefs,
    mask_context_belief_entities,
    random_mask_utterance,
)
from gpu import get_device
from utils import print_stage


def test_compute_metrics(eval_predictions):
    logits, hidden_values = eval_predictions.predictions
    print(tokenizer.batch_decode(logits.argmax(-1)))
    return {"score": 100}


def train():
    device, _ = get_device()
    name = "bart_finetune_cur"
    BATCH_SIZE = 8
    EPOCHS = 1
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
    
    print_stage("Masking Difference of Dialogue States")
    masked_deltas = dataset["train"].map(
        mask_delta_beliefs, remove_columns="turn"
    )
    masked_deltas = masked_deltas.map(
        tokenization, batched=True, remove_columns=masked_deltas.column_names,
    )

    print_stage("Masking Beliefs (Easy)")
    random_masked_beliefs_easy = dataset["train"].map(
        lambda d: random_mask_beliefs(d, 0.15), remove_columns="turn"
    )
    random_masked_beliefs_easy = random_masked_beliefs_easy.map(
        tokenization,
        batched=True,
        remove_columns=random_masked_beliefs_easy.column_names,
    )
    
    print_stage("Masking Utterances (Easy)")
    random_masked_utterances_easy = dataset["train"].map(
        lambda d: random_mask_utterance(d, 0.15), remove_columns="turn"
    )
    random_masked_utterances_easy = random_masked_utterances_easy.map(
        tokenization,
        batched=True,
        remove_columns=random_masked_utterances_easy.column_names,
    )

    print_stage("Masking Belief Entities in the Context")
    masked_context_belief_entities = dataset["train"].map(
        mask_context_belief_entities, remove_columns="turn"
    )
    masked_context_belief_entities = masked_context_belief_entities.map(
        tokenization,
        batched=True,
        remove_columns=masked_context_belief_entities.column_names,
    )


    print_stage("Masking Beliefs (Hard)")
    random_masked_beliefs_hard = dataset["train"].map(
        lambda d: random_mask_beliefs(d, 0.5), remove_columns="turn"
    )
    random_masked_beliefs_hard = random_masked_beliefs_hard.map(
        tokenization,
        batched=True,
        remove_columns=random_masked_beliefs_hard.column_names,
    )
    
    print_stage("Masking Utterances (Hard)")
    random_masked_utterances_hard = dataset["train"].map(
        lambda d: random_mask_utterance(d, 0.5), remove_columns="turn"
    )
    random_masked_utterances_hard = random_masked_utterances_hard.map(
        tokenization,
        batched=True,
        remove_columns=random_masked_utterances_hard.column_names,
    )
    
    print_stage("Masking All Belief Values")
    masked_beliefs_final = dataset.map(
        lambda d: random_mask_beliefs(d, 1), remove_columns="turn"
    )
    masked_beliefs_final = masked_beliefs_final.map(
        tokenization,
        batched=True,
        remove_columns=masked_beliefs_final.column_names,   # this removes ['train'], ['val'] ['test']
    )
    # sample_dataset = Dataset.from_dict(masked_deltas["validation"][:2])
    # sample_dataset_2 = Dataset.from_dict(random_masked_beliefs_easy["validation"][50:55])
    # sample_dataset_3 = Dataset.from_dict(random_masked_utterances_easy["validation"][50:55])
    # sample_dataset_4 = Dataset.from_dict(masked_context_belief_entities["validation"][50:55])

    # train_set = sample_dataset.map(
    #     tokenization, batched=True, remove_columns=sample_dataset.column_names
    # )
    # # , remove_columns='turn')
    # train_set_2 = sample_dataset_2.map(
    #     tokenization,
    #     batched=True,
    #     remove_columns=sample_dataset_2.column_names,
    # )

    # train_set_3 = sample_dataset_3.map(
    #     tokenization,
    #     batched=True,
    #     remove_columns=sample_dataset_3.column_names,
    # )

    # train_set_4 = sample_dataset_4.map(
    #     tokenization,
    #     batched=True,
    #     remove_columns=sample_dataset_4.column_names,
    # )
    curriculum_datasets = [
        masked_deltas,
        random_masked_beliefs_easy,
        random_masked_utterances_easy,
        masked_context_belief_entities,
        random_masked_beliefs_hard,
        random_masked_utterances_hard,
    ]

    model = BartAdapterModel.from_pretrained(
        "facebook/bart-base"
    ).to(device)
    model.resize_token_embeddings(len(tokenizer))

    # add and activate adapter
    model.add_adapter("dst")
    model.train_adapter("dst")

    # setup trainer
    # same as huggingface trainer
    args = TrainingArguments(
        output_dir=f"checkpoints/{name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5, # smaller lr
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
    trainer = CurriculumAdapterTrainer(
        curriculum_datasets,
        model,
        args,
        train_dataset=masked_beliefs_final["train"],
        eval_dataset=masked_beliefs_final["validation"],
        data_collator=data_collator,
        # compute_metrics=test_compute_metrics
        # callbacks=[MyCallback],  # We can either pass the callback class this way or an instance of it (MyCallback())
    )
    trainer.curriculum_train()


if __name__ == "__main__":
    train()