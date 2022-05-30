import re

def postprocessing(ds):
  '''
  input ds: dialog state for the current turn as string
  return prediction: valued prediction as list of triplet
  '''
  
  current_belief = re.search('<\|belief\|> (.*) <\|endofbelief\|>', ds)
  current_belief = current_belief.group(1)

  belief_list = re.split(', ', current_belief)  # here can split by ", domain", in case using commas alone is not enough

  prediction = []
  for x in belief_list:
    not_mentioned = re.search('not mentioned', x)
    if not_mentioned == None:
      precition.append(x)
  
  return prediction
