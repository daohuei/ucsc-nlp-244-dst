from tqdm.auto import tqdm

from utils.Constants import SLOT_VALS
from utils.dst import ignore_none, default_cleaning

slot_template = {slot: "" for slot in SLOT_VALS}


def get_slot_map(slot_triplet_str_list):
    slot_map = slot_template.copy()
    for slot_triplet_str in slot_triplet_str_list:
        slot_triplets = slot_triplet_str.split()
        key = slot_triplets[0] + " " + slot_triplets[1]
        val = slot_triplets[2:]
        if key not in SLOT_VALS:
            continue
        slot_map[key] = val
    return slot_map


def get_unique_slot_map(preds, targets):
    unique_slots = set()
    pred_map = {}
    target_map = {}

    for pred_str in preds:
        triplet = pred_str.split()
        key = triplet[0] + " " + triplet[1]
        val = triplet[2:]
        if key not in SLOT_VALS:
            continue
        pred_map[key] = val
        unique_slots.add(key)

    for target_str in targets:
        triplet = target_str.split()
        key = triplet[0] + " " + triplet[1]
        val = triplet[2:]
        if key not in SLOT_VALS:
            continue
        target_map[key] = val
        unique_slots.add(key)

    return unique_slots.copy(), pred_map.copy(), target_map.copy()


def evaluate_dst(input_results):
    num_turns = 0
    joint_acc = 0
    slot_acc = 0
    r_slot_acc = 0

    num_slots = len(SLOT_VALS)
    num_r_slots = 0

    clean_tokens = ["<s>", "</s>"]

    results = input_results.copy()

    for dial in tqdm(results.keys()):
        dialogue_pred = results[dial]["generated_turn_belief"]
        dialogue_target = results[dial]["target_turn_belief"]

        for turn_id, (turn_target, turn_pred) in enumerate(
            zip(dialogue_target, dialogue_pred)
        ):

            # clean
            for bs in turn_pred:
                if bs in clean_tokens + ["", " "] or bs.split()[-1] == "none":
                    turn_pred.remove(bs)

            new_turn_pred = []
            for bs in turn_pred:
                for tok in clean_tokens:
                    bs = bs.replace(tok, "").strip()
                    new_turn_pred.append(bs)
            turn_pred = new_turn_pred

            turn_pred, turn_target = ignore_none(turn_pred, turn_target)
            # print(turn_pred, turn_target)
            turn_pred, turn_target = default_cleaning(turn_pred, turn_target)

            join_flag = False

            # calculate joint accuracy
            if set(turn_target) == set(turn_pred):
                joint_acc += 1
                join_flag = True

            pred_slot_map = get_slot_map(turn_pred)
            target_slot_map = get_slot_map(turn_target)

            # calculate slot accuracy
            for slot_key in SLOT_VALS:
                if target_slot_map[slot_key] == pred_slot_map[slot_key]:
                    slot_acc += 1

            # calculate relative slot accuracy
            (
                unique_slots,
                unique_pred_map,
                unique_target_map,
            ) = get_unique_slot_map(turn_pred, turn_target)
            for slot_key in unique_slots:
                if slot_key not in unique_target_map.keys():
                    continue
                if slot_key not in unique_pred_map.keys():
                    continue
                if unique_target_map[slot_key] == unique_pred_map[slot_key]:
                    r_slot_acc += 1
            num_r_slots += len(unique_slots)

            num_turns += 1
    joint_acc /= num_turns
    slot_acc /= num_slots * num_turns
    r_slot_acc /= num_r_slots
    print("joint accuracy: {}".format(joint_acc))
    print("slot accuracy: {}".format(slot_acc))
    print("relative slot accuracy: {}".format(r_slot_acc))
    return {
        "joint_acc": joint_acc,
        "slot_acc": slot_acc,
        "r_slot_acc": r_slot_acc,
    }
