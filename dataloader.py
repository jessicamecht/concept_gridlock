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
DATA_PATH = "/home/jechterh/data1/jessica/data/toyota/once_data_w_lanes_compressed.hfd5" 

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
        self.resize = transforms.Resize(224)
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
        print('len')
        return len(self.people_seqs)

    def __getitem__(self, idx):
        print('gettt')
        rint = random.randint(10,15)
        sequences = self.people_seqs[idx]#keys are 'angle', 'id', 'image_array', 'lanes_2d', 'lanes_3d', 'meta', 'pos', 'segm_masks', 'seq_name_x', 'speed', 'times'
        images = torch.from_numpy(sequences['image_array'])[::rint].permute(0,3,1,2)
        masks = torch.from_numpy(sequences['segm_masks'])[::rint].permute(0,3,1,2)
        print('images', images.shape)
        images = F.resize(images, (224, 224))
        masks = F.resize(masks, (224, 224))
        return torch.from_numpy(sequences['meta'])[0:10], images,  masks,  torch.from_numpy(sequences['angle'])[0:10]