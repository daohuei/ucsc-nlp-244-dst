from collections import OrderedDict
import itertools
from more_itertools import pairwise
import re

import datasets


SLOT_VALS = [
    "attraction area",
    "attraction name",
    "attraction type",
    "hospital department",
    "hotel area",
    "hotel book day",
    "hotel book people",
    "hotel book stay",
    "hotel internet",
    "hotel name",
    "hotel parking",
    "hotel pricerange",
    "hotel stars",
    "hotel type",
    "restaurant area",
    "restaurant book day",
    "restaurant book people",
    "restaurant book time",
    "restaurant food",
    "restaurant name",
    "restaurant pricerange",
    "taxi arriveby",
    "taxi departure",
    "taxi destination",
    "taxi leaveat",
    "train arriveby",
    "train book people",
    "train day",
    "train departure",
    "train destination",
    "train leaveat",
]


def empty_belief():
    return OrderedDict([(sv, "not mentioned") for sv in SLOT_VALS])


def parse_raw_belief(text):
    """take in belief text without <|belief|> and <|endofbelief|>. Return dict of belief state"""
    belief_state = empty_belief()
    if "," not in text:
        return belief_state
    for field in text.split(","):
        split = field.strip().split(" ")
        slot_name = (
            split[:3] if " ".join(split[:3]) in SLOT_VALS else split[:2]
        )
        slot_value = split[len(slot_name) :]
        belief_state[" ".join(slot_name)] = " ".join(slot_value)
    return belief_state


def belief_to_text(belief):
    """take in belief dict and format to text (without <|belief|> and <|endofbelief|>)"""
    return " , ".join(key + " " + value for key, value in belief.items())


class HistoryBelief:
    """parses simpletod serialized history belief lines. Interally keeps track of belief states in a dict for later modifications"""

    raw_text: str

    def __init__(self, raw_text):
        # replace belief text with {belief} or {previousbelief}
        # used later on in text property
        self.raw_text = raw_text
        self.text_fmt, n_subs = re.subn(
            r"<\|([^\|]*?)belief\|>.*<\|endof\1belief\|>",
            r"<|\1belief|> {\1belief} <|endof\1belief|>",
            self.raw_text,
        )
        if n_subs == 1:
            # if we are dealing with the raw data there is no previous belief state yet
            self.text_fmt = re.sub(
                "<\|belief\|>",
                r"<|previousbelief|> {previousbelief} <|endofpreviousbelief|> <|belief|>",
                self.text_fmt,
            )
        # also sub context
        self.text_fmt = re.sub(
            r"<\|context\|>(.*)<\|endofcontext\|>",
            "<|context|>{context}<|endofcontext|>",
            self.text_fmt,
        )
        # parse the context in case we want to use it later
        self.context = re.search(
            "<\|context\|>(.*)<\|endofcontext\|>", self.raw_text
        ).group(1)
        # parse the belief in to an internal dict
        belief_match = re.search(
            "<\|belief\|>(.*)<\|endofbelief\|>", self.raw_text
        )
        if belief_match:
            raw_belief = belief_match.group(1)
            self.belief = parse_raw_belief(raw_belief)
        else:
            self.belief = empty_belief()
        # also parse the previous belief if it's there
        prev_belief_match = re.search(
            "<\|previousbelief\|>(.*)<\|endofpreviousbelief\|>", self.raw_text
        )
        if prev_belief_match:
            raw_prev_belief = prev_belief_match.group(1)
            self.prev_belief = parse_raw_belief(raw_prev_belief)
        else:
            # everything is not mentioned if no previous belief
            self.prev_belief = empty_belief()
        # get he first utterance from the context to check if two history beliefs are from the same conversation
        self.first_utterance = re.search(
            "<\|user\|>([^<]*)<\|", self.raw_text
        ).group(1)

    @property
    def belief_text(self):
        return belief_to_text(self.belief)

    @property
    def previous_belief_text(self):
        return belief_to_text(self.prev_belief)

    @property
    def text(self):
        return self.text_fmt.format(
            context=self.context,
            previousbelief=self.previous_belief_text,
            belief=self.belief_text,
        )


EMPTY_HB = """\
<s> \
<|context|> <|user|> <|endofcontext|> \
<|belief|> <|endofbelief|> \
</s>
"""


def is_same_conversation(hb1, hb2):
    return hb1.first_utterance == hb2.first_utterance


class MultiWozDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        # data is separated first by conversation, then by line
        features = datasets.Features(
            {
                "text": datasets.features.Sequence(datasets.Value("string")),
                "conversation_id": datasets.Value("int32"),
            }
        )
        return datasets.DatasetInfo(features=features, supervised_keys=None,)

    def _split_generators(self, dl_manager):
        data_files = self.config.data_files
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files[datasets.Split.TRAIN]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_files[datasets.Split.VALIDATION]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files[datasets.Split.TEST]},
            ),
        ]

    def _generate_examples(self, filepath):
        text_stream = filepath[0].open().readlines()
        # add in an empty line at the beginning so that we process the first line in the file
        padded_text_stream = itertools.chain.from_iterable(
            [[EMPTY_HB], text_stream]
        )
        # parse all lines with HistoryBelief class
        history_belief_generator = (
            HistoryBelief(raw_text.strip()) for raw_text in padded_text_stream
        )
        conversation = []
        _id = 1
        for prev_hb, hb in pairwise(history_belief_generator):
            if is_same_conversation(prev_hb, hb):
                hb.prev_belief = prev_hb.belief
            elif conversation:
                # when we move onto new conversation, return prev conversation
                # also don't want to return empty conversation
                yield _id, {"text": conversation, "conversation_id": _id}
                _id += 1
                conversation = []
            conversation.append(hb.raw_text)
        # return last conversation in queue
        yield _id, {"text": conversation, "conversation_id": _id}
