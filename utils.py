from torch.nn.utils.rnn import pad_sequence
import numpy as np 
import clip
import random
p = "/home/jessica/personalized_driving_toyota/scenarios/scenarios.txt"
with open(p) as file:
    lines = [line.strip() for line in file]
scenarios = lines
scenarios_tokens = clip.tokenize(scenarios)

p = "/home/jessica/personalized_driving_toyota/scenarios/scenarios_nuscenes.txt"
with open(p) as file:
    lines = [line.strip() for line in file]
scenarios = lines

p = "/home/jessica/personalized_driving_toyota/scenarios/scenarios_small_300.txt"
with open(p) as file:
    lines = [line.strip() for line in file]
scenarios = lines

#scenarios = random.sample(scenarios, 300)
#with open("scenarios_small_100.txt", "w") as file:
#    for item in scenarios:
#        file.write(str(item) + "\n")
scenarios_tokens = clip.tokenize(scenarios)

def pad_collate(batch):
    '''just in case if there were different sequence lengths, 
    but currently all lengths should be the same when batching'''
    meta, img, vego, angle, dist = zip(*batch)
    m_lens = [len(x) for x in meta]
    i_lens = [len(y) for y in img]
    s_lens = [len(x) for x in vego]
    a_lens = [len(y) for y in angle]
    d_lens = [len(y) for y in dist] if dist[0] != None else None 

    m_pad = pad_sequence(meta, batch_first=True, padding_value=0)
    i_pad = pad_sequence(img, batch_first=True, padding_value=0)
    vego_pad = pad_sequence(vego, batch_first=True, padding_value=0)
    a_pad = pad_sequence(angle, batch_first=True, padding_value=0)
    d_pad = pad_sequence(dist, batch_first=True, padding_value=0) if dist[0] != None else None 

    return m_pad, i_pad, vego_pad, a_pad,d_pad, m_lens, i_lens, s_lens, a_lens, d_lens

def get_reduced_sample(ccenabled, reinstate=False, window = 10):
    switch_indices = np.where(np.diff(ccenabled.squeeze().int()) == (1 if reinstate else -1))[0] + 1
    intervention = np.zeros_like(ccenabled.squeeze(), dtype=bool)
    intervention_before = np.zeros_like(ccenabled.squeeze(), dtype=bool)
    intervention_after = np.zeros_like(ccenabled.squeeze(), dtype=bool)
    expanded_indices_before = []
    expanded_indices_after = []
    for i in switch_indices:
        expanded_indices_before.extend(list(range(max(i-window, 0), i)))
        expanded_indices_after.extend(list(range(i+1, min(i+window, len(ccenabled.squeeze())))))
        break
    intervention[switch_indices] = True
    intervention_before[expanded_indices_before] = True
    intervention_after[expanded_indices_after] = True
    return intervention, intervention_before, intervention_after