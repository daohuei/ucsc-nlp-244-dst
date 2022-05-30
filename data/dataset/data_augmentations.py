from more_itertools import pairwise
import random
import re

from .multiwoz_dataset import HistoryBelief

MASK_TOKEN = "<mask>"

def insert_previous_belief(data):
    """insert previous turn's belief into language model text
    apply this method before flatten_conversation and use batch=False"""
    processed = [HistoryBelief(data['text'][0]).text]
    for text_1, text_2 in pairwise(data['text']):
        hb1, hb2 = HistoryBelief(text_1), HistoryBelief(text_2)
        hb2.prev_belief = hb1.belief
        processed.append(hb2.text)
    return {'text': processed}


def flatten_conversation(data):
    """flattens the dataset so that every turn is one example
    use batch=True for this"""
    turns = []
    conversation_ids = []
    turn_numbers = []
    for text, conversation_id in zip(data['text'], data['conversation_id']):
        for i, t in enumerate(text, start=1):
            turn_numbers.append(i)
            conversation_ids.append(conversation_id)
            turns.append(t)
    return {'turn': turns, 'conversation_id': conversation_ids, 'turn_number': turn_numbers}


def mask_delta_beliefs(data):
    """masks current belief values for a turn if it is different from the previous belief
    apply this after insert_previous_belief and flatten_conversation"""
    hb = HistoryBelief(data['turn'])
    prev_belief = hb.prev_belief
    for key in prev_belief:
        if hb.belief[key] != prev_belief[key]:
            hb.belief[key] = MASK_TOKEN
    return {'masked': hb.text, 'target': data['turn']}


def random_mask_beliefs(data, r):
    """randomly masks current belief values at proportion 0 <= r <= 1
    apply this after flatten_conversation"""
    hb = HistoryBelief(data['turn'])
    n = round(r*len(hb.belief))
    mask_keys = random.sample(hb.belief.keys(), n)
    for key in mask_keys:
        hb.belief[key] = MASK_TOKEN
    return {'masked': hb.text, 'target': data['turn']}


def remove_repeating_masks(string):
    """replaces contiguous masks with a single mask for text infilling"""
    tokens = string.split(' ')
    repeated_mask_idxs = [i for i, (tok1, tok2) in enumerate(pairwise(tokens), start=1) 
                            if tok1 == MASK_TOKEN and tok2 == MASK_TOKEN]
    for i in repeated_mask_idxs[::-1]:
        del tokens[i]
    return ' ' + ' '.join(tokens) + ' '


def mask_context_belief_entities(data):
    """masks entities in the context of the language modeling string if that entitiy appears as a belief
    apply this after flatten_conversation"""
    hb = HistoryBelief(data['turn'])
    values_or = ' | '.join(set(v for v in hb.belief.values() if v != 'not mentioned'))
    if values_or:
        hb.context = re.subn(values_or, f" {MASK_TOKEN} ", hb.context)[0]
        hb.context = remove_repeating_masks(hb.context)
    return {'masked': hb.text, 'target': data['turn']}


def random_mask_utterance(data, r):
    """randomly masks tokens in the context at proportion 0 <= r <= 1
    apply this after flatten_conversation"""
    hb = HistoryBelief(data['turn'])
    tokens = hb.context.strip().split(' ')
    content_idxs = [i for i, tok in enumerate(tokens) if tok != '<|system|>' and tok != '<|user|>']
    n = round(r*len(content_idxs))
    mask_idxs = random.sample(content_idxs, n)
    for i in mask_idxs:
        tokens[i] = MASK_TOKEN
    hb.context = ' '.join(tokens)
    hb.context = remove_repeating_masks(hb.context)
    return {'masked': hb.text, 'target': data['turn']}
