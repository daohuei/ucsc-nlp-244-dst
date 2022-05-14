# preprocess multiwoz with delexicalized responses
python3 preprocess_multiwoz.py delex

# preprocess multiwoz with lexicalized responses
python3 preprocess_multiwoz.py lexical

# create dataset for language modeling with SimpleTOD
python3 prepare_simpletod_data.py
