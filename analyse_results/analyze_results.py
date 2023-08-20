import pandas as pd
import numpy
import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
import glob
import os
import matplotlib.pyplot as plt 
from PIL import Image
from collections import Counter
import warnings 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import sys
sys.path.append("..")
from utils import pad_collate
from dataloader_comma import CommaDataset
from dataloader_nuscenes import NUScenesDataset
from model import VTN

gpu_num = 1
gpu = f'cuda:{gpu_num}'
multitask = 'distance'
backbone = 'none'
concept_features = True
'''
nuscenes_ds = NUScenesDataset(dataset_type="test",
        multitask=multitask, max_len=20,
        ground_truth="normal", dataset_path='/data1/jessica/data/toyota/')'''

commda_ds = CommaDataset(dataset_type="test",
        multitask=multitask,
        ground_truth="normal", dataset_path='/data1/jessica/data/toyota/')
dataloader_comma = DataLoader(commda_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=pad_collate)

model = VTN(multitask=multitask, backbone=backbone, concept_features=concept_features, device = f"cuda:{gpu_num}", return_concepts=True)
checkpoint_path_distance = '/data1/jessica/data/toyota/ckpts_final/ckpts_final_comma_distance_none/lightning_logs/version_0/checkpoints/epoch=50-step=3162.ckpt'
checkpoint_path_angle = '/data1/jessica/data/toyota/ckpts_final/ckpts_final_nuscenes_distance_none/lightning_logs/version_10/checkpoints/epoch=519-step=14560.ckpt'
checkpoint_path = checkpoint_path_distance

ckpt = torch.load(checkpoint_path)
state_dict = ckpt['state_dict']
print('done')