from torch.utils.data import Dataset, DataLoader  # For custom data-sets
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import os
import numpy as np
from PIL import Image
import torch
import pandas as pd
from collections import namedtuple
import h5py
import matplotlib.pyplot as plt
import time
import random
DATA_PATH = "../processed/once_data_w_lanes_compressed.hfd5" 

class ONCEDataset(Dataset):
    def __init__(
        self,
        dataset_type="train",
        in_size=(480, 640),
        out_size=(240, 320),
        data_path=DATA_PATH,
        max_depth=0,
        use_transform=False,
    ):
        assert dataset_type in ["train", "test"]
        if dataset_type == "test":
            dataset_type = "val"
        self.dataset_type = dataset_type
        self.out_size = out_size
        self.data_path = data_path
        self.use_transform = use_transform
        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )
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
        return sequences['meta'], sequences['image_array'], sequences['segm_masks'], sequences['angle']