from transformers import Trainer

from data.dataset.tokenize import preprocess_func


class CurriculumTrainer(Trainer):
    def __init__(self, curriculum_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_dataset = curriculum_dataset

    def curriculum_train(self):
        self.train()

