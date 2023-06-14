import pandas as pd
import numpy
import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append("..")
from utils import pad_collate
from dataloader_comma import CommaDataset
from dataloader_nuscenes import NUScenesDataset
from collections import Counter
from model import VTN
import matplotlib.pyplot as plt 
from PIL import Image
import glob
import os
from utils import * 
import re
gpu_num = 0
gpu = f'cuda:{gpu_num}'
multitask = 'distance'
backbone = 'none'
concept_features = True

scenarios_tokens = scenarios_tokens.to()

def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else -1

class UnNormalize(object):
    '''Since the dataloader returns normalized images, we migth need to unnormalize them again'''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))



def get_aligned_attention(attention, seq_len):
    attention = attention[:, 1:-1]
    sequence_length = seq_len
    window_size = 8
    overlap = 4
    global_attended_token = 1
    padding_tokens = 8
    # Calculate the number of chunks
    number_of_chunks = np.ceil((sequence_length - window_size + overlap + 1) / (window_size - overlap)).astype(int)

    # Create an empty alignment array
    alignment_array = np.zeros((sequence_length + 2 * window_size , sequence_length + 2 * window_size), dtype=float)

    # Iterate over each chunk and extract attended token index
    for chunk_idx in range(number_of_chunks):
        # Calculate the start and end indices of the chunk
        start_index = chunk_idx * (window_size - overlap)
        end_index = start_index + window_size
        alignment_array[chunk_idx, start_index:end_index] = attention[chunk_idx]
    return alignment_array


def moving_average(signal, window_size):
    window = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, window, mode='same')
    return smoothed_signal

def get_regular_ckpt_from_lightning_checkpoint(state_dict):
    for key in list(state_dict.keys()):
        oldkey = key
        if oldkey[0:6] == 'model.':
            key = oldkey[6:]
            state_dict[key] = state_dict.pop(oldkey)
    return state_dict

commda_ds = CommaDataset(dataset_type="test",
        multitask="distance",
        ground_truth="normal", dataset_path='/data1/jessica/data/toyota/')
nuscenes_ds = NUScenesDataset(dataset_type="test",
        multitask="distance",
        ground_truth="normal", dataset_path='/data1/jessica/data/toyota/' )
dataloader_comma = DataLoader(commda_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=pad_collate)
dataloader_nuscenes = DataLoader(nuscenes_ds, batch_size=1, shuffle=False, num_workers=0)

model = VTN(multitask=multitask, backbone=backbone, concept_features=concept_features, device = f"cuda:{gpu_num}", return_concepts=True)
checkpoint_path_distance = '/data1/jessica/data/toyota/ckpts_final/ckpts_final_comma_distance_none/lightning_logs/version_0/checkpoints//epoch=50-step=3162.ckpt'
checkpoint_path_angle = '/data1/jessica/data/toyota/ckpts_final/ckpts_final_comma_distance_none/lightning_logs/version_0/checkpoints//epoch=50-step=3162.ckpt'
checkpoint_path = checkpoint_path_angle

ckpt = torch.load(checkpoint_path, map_location=gpu)
state_dict = ckpt['state_dict']
state_dict = get_regular_ckpt_from_lightning_checkpoint(state_dict)
model.load_state_dict(state_dict)
model.eval()
model = model.to(gpu)