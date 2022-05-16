from utils.args_parser import ArgsParser
from data.dataset.multiwoz import MultiWozDataset
from utils.multiwoz import dbPointer
import json
import os

from transformers import GPT2Tokenizer, BartTokenizer
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
multiwoz_data = json.load(open('resources/multi-woz/lex.json', 'r'))

save_dir = './resources/bart'
os.makedirs(save_dir, exist_ok=True)

for split in ['val', 'train', 'test']:
    opt = ArgsParser().parse()
    opt.use_knowledge = True
    opt.use_action = True
    opt.context_knowledge = True
    opt.lexical = True

    data = MultiWozDataset(opt, split=split, shuffle=False)
    
    opt_delex = ArgsParser().parse()
    data_delex = MultiWozDataset(opt_delex, split=split, shuffle=False)
    
    history_raw_new = []
    belief_raw_new = []
    belief_raw_none_new = []
    action_raw_new = []
    output_raw_new = []
    output_raw_delex_new = []
    db_search_raw = []
    db_nmatch_raw = []
    
    if split == 'test':
        test_dict = {}

    # construct into dictionary
    lex_dict = {}
    delex_dict = {}
    for d in data:
        lex_dict[d['name']] = d

    for d in data_delex:
        delex_dict[d['name']] = d

    # go through every dialogues
    for key in lex_dict:
        d_lex = lex_dict[key]
        d_delex = delex_dict[key]
        inp = d_lex['input_raw']
        out = d_lex['target_raw']
        out_delex = d_delex['target_raw']
  
        # go over each turn
        for i, (usr, sys) in enumerate(zip(inp, out)):
            if i == 0:
                history_new = '<|context|> <|user|> {} <|endofcontext|>'.format(usr)
            else:
                tmp_new = ['<|context|>']
                # include the history
                for k in range(i):

                    tmp_new.append('<|user|> ' + inp[k])
                    tmp_new.append('<|system|> ' + out[k])

                tmp_new.append('<|user|> ' + usr + '<|endofcontext|>')
                history_new = ' '.join(tmp_new)

            sys_delex = out_delex[i]
            history_raw_new.append(history_new)
            output_raw_new.append('<|response|> ' + sys + ' <|endofresponse|>')

            output_raw_delex_new.append('<|response|> ' + sys_delex.strip() + ' <|endofresponse|>')

        # go over every belief in turn
        belief = d_lex['belief_raw']
        for bs in belief:
            # bs -> a list of triplets
            tmp_bs_new = []
            for i, b in enumerate(bs):
                # value
                if b[-1] in ['not mentioned']: # comment this for DST task
                    continue
                if i == len(bs) - 1:
                    tmp_bs_new.append(' '.join(b))
                else:
                    tmp_bs_new.append(' '.join(b))

            if len(tmp_bs_new) == 0:
                tmp_bs_new.append(' ')

            tmp_new = '<|belief|> {} <|endofbelief|>'.format(' , '.join(tmp_bs_new))
            belief_raw_new.append(tmp_new)

        # belief for DST task (include none)
        # simply included not mentioned value
        for bs in belief:
            tmp_bs_new = []
            for i, b in enumerate(bs):
                if i == len(bs) - 1:
                    tmp_bs_new.append(' '.join(b))
                else:
                    tmp_bs_new.append(' '.join(b))

            if len(tmp_bs_new) == 0:
                tmp_bs_new.append(' ')

            tmp_new = '<|belief|> {} <|endofbelief|>'.format(' , '.join(tmp_bs_new))
            belief_raw_none_new.append(tmp_new)

        action = d_lex['action_raw']
        # for action sequence: simply a triplet for actions:(domain, action_type, value)
        for act in action:
            tmp_act_new = []
            for i, a in enumerate(act):
                if i == len(act) - 1:
                    tmp_act_new.append(' '.join(a))
                else:
                    tmp_act_new.append(' '.join(a))
            if len(tmp_act_new) == 0:
                tmp_act_new.append(' ')

            tmp_new = '<|action|> {} <|endofaction|>'.format(' , '.join(tmp_act_new))
            action_raw_new.append(tmp_new)


    # history only
    with open('{}/{}.history'.format(save_dir, split), 'wt') as f:
        for l in history_raw_new:
            f.write('{} {}\n'.format(bart_tokenizer._bos_token, l.lower()))

    tmp = []
    for hist, bs in zip(history_raw_new, belief_raw_none_new):
        tmp.append(' '.join([hist.lower(), bs.lower()]))

    # history and belief
    with open('{}/{}.history_belief'.format(save_dir, split),
              'wt') as f:
        for l in tmp:
            f.write('{} {} {}\n'.format(bart_tokenizer._bos_token, l.lower(), bart_tokenizer._eos_token))

    tmp = []
    for hist, bs, act, trg in zip(history_raw_new, belief_raw_new, action_raw_new, output_raw_delex_new):
        tmp.append(' '.join([hist.lower(), bs.lower(), act, trg]))
    # history, belief, action, and delex reponses
    with open('{}/{}.history_belief_action_sys_delex'.format(save_dir, split), 'wt') as f:
        for l in tmp:
            f.write('{} {} {}\n'.format(bart_tokenizer._bos_token, l.lower(), bart_tokenizer._eos_token))
