# ucsc-nlp-244-dst
UCSC NLP244 Final Project

## Usage
### Create Dataset
```bash
bash create_dataset.sh
```
### Tokenization
- use `output_tokenization.ipynb` for generating tokens.
> Just use first few cells for generating tokens
### Training
- Modify `config.py` for changing tunable hyperparameters.
- Run `python train.py` for training.
### Inference & Evaluation
- Modify setup in `inference.py` sciprt, such as checkpoint loading code.
- Run `python inference.py` script for inference and evaluation