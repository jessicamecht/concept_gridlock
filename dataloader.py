from torch.utils.data import Dataset, DataLoader  # For custom data-sets
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import os
import math 
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch
import pandas as pd
from collections import namedtuple
import h5py
import matplotlib.pyplot as plt
import time
import random

class ONCEDataset(Dataset):
    def __init__(
        self,
        dataset_type="train",
        out_size=(240, 320),
        use_transform=False,
    ):
        assert dataset_type in ["train", "val", "test"]
        self.dataset_type = dataset_type
        self.max_len = 150
        self.min_angle, self.max_angle, self.range_angle = (2.1073424e-08, 0.102598816, 0.102598794)
        self.out_size = out_size
        self.use_transform = use_transform
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if dataset_type == "train":
            data_path = "/data1/jessica/data/toyota/once_data_w_lanes_compressed_train.hfd5" 
        else:
            data_path = "/data1/jessica/data/toyota/once_data_w_lanes_compressed.hfd5"
        self.people_seqs = []
        with h5py.File(data_path, "r") as f:
            for seq_key in list(f.keys()):
                iter_dict = {}
                keys_ = f[seq_key].keys()
                for key in keys_:
                    ds_obj = f[seq_key][key][()]
                    iter_dict[key] = ds_obj
                self.people_seqs.append(iter_dict)

    def __len__(self):
        return len(self.people_seqs)

    def __getitem__(self, idx):
        
        sequences = self.people_seqs[idx]#keys are 'angle', 'id', 'image_array', 'lanes_2d', 'lanes_3d', 'meta', 'pos', 'segm_masks', 'seq_name_x', 'speed', 'times'
        rint = 0#random.randint(0,max(0, len(sequences['image_array'])-(self.max_len+1))) to randomize sequence
        start = rint if len(sequences['image_array']) > self.max_len else 0
        end = rint+self.max_len if len(sequences['image_array']) > self.max_len else -1
        images = torch.from_numpy(sequences['image_array'])[start:end].permute(0,3,1,2)
        masks = torch.from_numpy(sequences['segm_masks'])[start:end].permute(0,3,1,2)
        images = F.resize(self.normalize(images), (224, 224))
        masks = F.resize(masks, (224, 224))
        angles = torch.from_numpy(sequences['angle'])[start:end]*(180/np.pi)
        angles = angles - self.min_angle/self.range_angle
        
        return torch.zeros(len(sequences['angle']))[start:end], images,  masks,  angles