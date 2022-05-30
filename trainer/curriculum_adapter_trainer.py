from transformers import AdapterTrainer

class CurriculumAdapterTrainer(AdapterTrainer):    
    def __init__(self, curriculum_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_dataset = curriculum_dataset

    def curriculum_train(self):

        self.total_epochs = self.args.num_train_epochs
        self.final_train_dataset = self.train_dataset
        self.model_name = self.args.output_dir

        for idx, c_dataset in enumerate(self.curriculum_dataset):
            # only train for 1 epoch during curriculum training
            self.args.num_train_epochs = 1
            self.args.output_dir = f"{self.model_name}/course_{idx}"
            self.train_dataset = c_dataset
            self.train()

        # switch back to the final training on the most difficult
        self.args.num_train_epochs = self.total_epochs
        self.args.output_dir = f"{self.model_name}/final"
        self.train_dataset = self.final_train_dataset
        self.train()