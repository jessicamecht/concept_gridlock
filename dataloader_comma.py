from torch.utils.data import Dataset  # For custom data-sets
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import torch
import h5py
from hampel import hampel 
import pandas as pd
from scipy import signal
from scipy.spatial.transform import Rotation
from decimal import Decimal
import torchvision.transforms.functional as TF
import random

class CommaDataset(Dataset):
    def __init__(
        self,
        dataset_type="train",
        out_size=(240, 320),
        use_transform=False,
        multitask="angle"
    ):
        assert dataset_type in ["train", "val", "test"]
        self.dataset_type = dataset_type
        self.max_len = 240
        self.multitask = multitask
        self.min_angle, self.max_angle, self.range_angle = (2.1073424e-08, 0.102598816, 0.102598794)
        self.out_size = out_size
        self.use_transform = use_transform
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.normalize_values = False
        if dataset_type == "train":
            data_path = "/data1/jessica/data/toyota/train_comma_1.hfd5"
            data_path2 = "/data1/jessica/data/toyota/train_comma_2.hfd5"
        elif dataset_type == "test":
            data_path = "/data1/jessica/data/toyota/test_comma_1.hfd5"
        elif dataset_type == "val":
            data_path = "/data1/jessica/data/toyota/val_comma_1.hfd5"
        self.people_seqs = []
        with h5py.File(data_path, "r") as f:
                self.all_angle_max = 0 
                self.all_angle_min = 100000
                self.all_dist_max = 0 
                self.all_dist_min = 100000
                for i, seq_key in enumerate(list(f.keys())):
                    person_seq = {}
                    keys_ = f[seq_key].keys()#'angle', 'brake', 'dist', 'gas', 'image', 'time', 'vEgo'
                    for key in keys_:                        
                        seq = f[seq_key][key][()]
                        if len(seq) < 100:
                            break
                        person_seq[key] = seq
                    if len(person_seq.keys()) == 7:
                        self.people_seqs.append(person_seq)
                    else: 
                        print(len(person_seq.keys()), 'failt')
        if dataset_type == "train":
            with h5py.File(data_path2, "r") as f:
                self.all_angle_max = 0 
                self.all_angle_min = 100000
                self.all_dist_max = 0 
                self.all_dist_min = 100000
                for i, seq_key in enumerate(list(f.keys())):
                    person_seq = {}
                    keys_ = f[seq_key].keys()#'angle', 'brake', 'dist', 'gas', 'image', 'time', 'vEgo'
                    for key in keys_:                        
                        seq = f[seq_key][key][()]
                        if len(seq) < 100:
                            break
                        person_seq[key] = seq
                    if len(person_seq.keys()) == 7:
                        self.people_seqs.append(person_seq)
                    else: 
                        print(len(person_seq.keys()), 'failt')
        print("len",len(self.people_seqs))

    def __len__(self):
        return len(self.people_seqs)

    def __getitem__(self, idx):
        sequences = self.people_seqs[idx]
        #rint = random.randint(0,max(0, len(sequences['image_array'])-(self.max_len+1))) #to randomize sequence
        start = 0#rint if len(sequences['image_array']) > self.max_len else 0
        end = self.max_len#rint+self.max_len if len(sequences['image_array']) > self.max_len else -1
        images = torch.from_numpy(np.array(sequences['image']).astype(float)).permute(0,3,1,2)[1::5][start:end]#[start:end]
        masks = 0#torch.from_numpy(sequences['segm_masks'].astype(int))[start:end].permute(0,3,1,2)
        #images = F.resize(self.normalize(images.type(torch.float)), (224, 224))
        #masks = F.resize(masks, (224, 224))
        vego = torch.from_numpy(np.array(sequences['vEgo'])[1::5].astype(float))[start:end]
        angles = torch.from_numpy(sequences['angle'].astype(float))[1::5][start:end]#[start:end]#*(180/np.pi)
        distances = torch.from_numpy(np.array(sequences['dist']))[start:end]
        max_dist = 70
        min_dist = 0
        distances[distances > max_dist] = 0
        #distances = ((distances - min_dist) / (max_dist - min_dist))
        #distances = torch.from_numpy(signal.resample(distances, len(images)))#[start:end]
        

        #angles = angles - self.min_angle/self.range_angle
        #images, masks = my_segmentation_transforms(images, masks)
        if self.normalize_values: 
            angles = 2*((angles - self.all_angle_min)/(self.all_angle_max-self.all_angle_min))-1
            distances = 2*((angles - self.all_dist_min)/(self.all_dist_max-self.all_dist_min))-1
        res = torch.zeros(len(sequences['angle']))[start:end], images.type(torch.float32),  vego.type(torch.float32),  angles.type(torch.float32), distances.type(torch.float32)
        if self.multitask == "distance":
            res = torch.zeros(len(sequences['angle']))[start:end], images.type(torch.float32),  vego.type(torch.float32), distances, angles.type(torch.float32)
        #print(torch.zeros(len(sequences['angle']))[start:end].shape, images.type(torch.float32).shape,  images.type(torch.float32).shape, distances.type(torch.float32).shape, angles.type(torch.float32).shape)
        return res 