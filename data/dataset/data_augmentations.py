import random
import re

from .multiwoz_dataset import HistoryBelief


def flatten_conversation(data):
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
    hb = HistoryBelief(data['turn'])
    prev_belief = hb.prev_belief
    for key in prev_belief:
        if hb.belief[key] != prev_belief[key]:
            hb.belief[key] = 'MASK'
    return {'masked': hb.text, 'target': data['turn']}


def random_mask_beliefs(data, r):
    hb = HistoryBelief(data['turn'])
    n = round(r*len(hb.belief))
    mask_keys = random.sample(hb.belief.keys(), n)
    for key in mask_keys:
        hb.belief[key] = 'MASK'
    return {'masked': hb.text, 'target': data['turn']}


def remove_repeating_masks(string):
    tokens = string.split(' ')
    repeated_mask_idxs = []
    for i in range(1, len(tokens)):
        if tokens[i] == 'MASK' and tokens[i-1] == 'MASK':
            repeated_mask_idxs.append(i)
    for i in sorted(repeated_mask_idxs, reverse=True):
        del tokens[i]
    return ' '.join(tokens)


def mask_context_belief_entities(data):
    hb = HistoryBelief(data['turn'])
    values_or = ' | '.join(set(v for v in hb.belief.values() if v != 'not mentioned'))
    if values_or:
        hb.context = re.subn(values_or, ' MASK ', hb.context)[0]
        hb.context = ' ' + remove_repeating_masks(hb.context)
    return {'masked': hb.text, 'target': data['turn']}


def hit_syntax(tokens, idxs):
    for i in idxs:
        if tokens[i] == '<|system|>' or tokens[i] == '<|user|>':
            return True
    else:
        return False


def random_mask_utterance(data, r):
    hb = HistoryBelief(data['turn'])
    tokens = hb.context.strip().split(' ')
    content_tokens = list(filter(lambda tok: tok != '<|system|>' and tok != '<|user|>', tokens))
    n = round(r*len(content_tokens))
    idxs = [0]
    while hit_syntax(tokens, idxs):
        idxs = random.sample(range(len(tokens)), n)
    for i in idxs:
        tokens[i] = 'MASK'
    hb.context = ' '.join(tokens)
    hb.context = ' ' + remove_repeating_masks(hb.context)
    return {'masked': hb.text, 'target': data['turn']}
