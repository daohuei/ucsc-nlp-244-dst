from transformers import AdapterTrainer, Trainer


class CurriculumAdapterTrainer(AdapterTrainer, Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
