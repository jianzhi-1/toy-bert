from os import listdir
from os.path import isfile, join
import numpy as np
import jax.numpy as jnp

def getData(batch_size=1, mask=0.15):
  """ 
  batch_size: size of the batch
  mask: the percentage of input replaced with mask, default 15%

  RETURNS
  data_pairs: a list of dictionaries, each element of the form
    {
      "obs": jnp.array,
      "target": jnp.array
    }
  vocab_size: the total size of vocabulary, including auxiliary tokens
  
  """

  direc = './data' # directory
  allfiles = [join(direc, f) for f in listdir(direc) if isfile(join(direc, f))]
  data_pairs = []

  vocab_size = 5 # initially, start with <MASK>, <CLS>, <SEP>, <ISNEXT>, <NOTNEXT>
  vocab_map = {"<MASK>": 0, "<CLS>": 1, "<SEP>": 2, "<ISNEXT>": 3, "<NOTNEXT>": 4}
  for file_name in allfiles:
    with open(file_name, 'r') as f:
      lines = f.readlines()
      for line in lines:
        tokens = line.split()
        for token in tokens:
          if token in vocab_map: continue
          vocab_map[token] = vocab_size
          vocab_size += 1

  for file_name in allfiles:
    all_obs = []
    all_target = []
    with open(file_name, 'r') as f:
      lines = f.readlines()
      for i in range(len(lines) - 1):
        follow = 1 if np.random.random() > 0.5 else 0 # whether latter sentence follows former
        if follow:
          tokenized_sentence_1 = lines[i].split()
          tokenized_sentence_2 = lines[i + 1].split()
        else:
          tokenized_sentence_1 = lines[i + 1].split()
          tokenized_sentence_2 = lines[i].split()
        
        label = ["<ISNEXT>"] if follow else ["<NOTNEXT>"]
        
        target = label + tokenized_sentence_1 + ["<SEP>"] + tokenized_sentence_2
        target = [vocab_map[token] for token in target]

        mask_sentence_1 = (np.random.random(len(tokenized_sentence_1)) < mask)
        for j in range(len(mask_sentence_1)):
          if mask_sentence_1[j]: tokenized_sentence_1[j] = "<MASK>"
        mask_sentence_2 = (np.random.random(len(tokenized_sentence_2)) < mask)
        for j in range(len(mask_sentence_2)):
          if mask_sentence_2[j]: tokenized_sentence_2[j] = "<MASK>"
        
        obs = ["<CLS>"] + tokenized_sentence_1 + ["<SEP>"] + tokenized_sentence_2
        obs = [vocab_map[token] for token in obs]
        all_obs.append(obs)
        all_target.append(target)
  
  assert len(all_obs) == len(all_target)
  for i in range(0, len(all_obs), batch_size):
    obs_batch = []
    target_batch = []
    for j in range(i, i + batch_size):
      obs_batch.append(all_obs[j])
      target_batch.append(all_target[j])
    cur_dict = dict()
    cur_dict["obs"] = jnp.array(obs_batch)
    cur_dict["target"] = jnp.array(target_batch)
    data_pairs.append(cur_dict)
  return data_pairs, vocab_size
