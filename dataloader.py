from torch.utils.data import Dataset  # For custom data-sets
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import torch
import h5py
from scipy.spatial.transform import Rotation
from decimal import Decimal
import random

def get_speed(vel):
    assert(len(vel) == 3)
    return (Decimal(vel[0]) ** Decimal(2) + Decimal(vel[1]) ** Decimal(2) + Decimal(vel[2]) ** Decimal(2)) ** Decimal(0.5)

def angle(v1, v2):
    v1 = [Decimal(v1[0].item()), Decimal(v1[1].item()), Decimal(v1[2].item())]
    v2 = [Decimal(v2[0].item()), Decimal(v2[1].item()), Decimal(v2[2].item())]
    # [0, np.pi]
    num = np.dot(v1, v2)
    denom = (get_speed(v1) * get_speed(v2))
    return np.arccos(float(num / denom)) 

class ONCEDataset(Dataset):
    def __init__(
        self,
        dataset_type="train",
        out_size=(240, 320),
        use_transform=False,
        multitask="angle"
    ):
        assert dataset_type in ["train", "val", "test"]
        self.dataset_type = dataset_type
        self.max_len = 250
        self.multitask = multitask
        self.min_angle, self.max_angle, self.range_angle = (2.1073424e-08, 0.102598816, 0.102598794)
        self.out_size = out_size
        self.use_transform = use_transform
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if dataset_type == "train":
            data_path = "/data1/jessica/data/toyota/once_w_lanes_compressed_raw_small_multitask_all_train.hfd5"
            paths = [data_path]
        elif dataset_type == "test":
            data_path = "/data1/jessica/data/toyota/once_w_lanes_compressed_raw_small_multitask_all_test.hfd5"
            paths = [data_path]     
        elif dataset_type == "val":
            data_path = "/data1/jessica/data/toyota/once_w_lanes_compressed_raw_small_multitask_all_val.hfd5"
            paths = [data_path]
        self.people_seqs = []
        for data_path in paths:
            with h5py.File(data_path, "r") as f:
                for i, seq_key in enumerate(list(f.keys())):
                    iter_dict = {}
                    keys_ = f[seq_key].keys()
                    for key in keys_:
                        if key == "angle": continue
                        ds_obj = f[seq_key][key][()]#[0::subsample]
                        if key == "pos":
                            new_angles = []
                            for i in range(1, len(ds_obj)):
                                rotation = Rotation.from_quat(ds_obj[i][:4]).as_euler('zyx', degrees=True)    
                                prev_rotation = Rotation.from_quat(ds_obj[i-1][:4]).as_euler('zyx', degrees=True)       
                                rot = rotation-prev_rotation
                                new_angles.append(rot)
                            ds_obj = np.array(new_angles)
                            iter_dict["angle"] = ds_obj
                            continue
                        iter_dict[key] = ds_obj
                    self.people_seqs.append(iter_dict)

    def __len__(self):
        return len(self.people_seqs)

    def __getitem__(self, idx):
        sequences = self.people_seqs[idx]#keys are 'angle', 'id', 'image_array', 'lanes_2d', 'lanes_3d', 'meta', 'pos', 'segm_masks', 'seq_name_x', 'speed', 'times'
        rint = random.randint(0,max(0, len(sequences['image_array'])-(self.max_len+1))) #to randomize sequence
        start = rint if len(sequences['image_array']) > self.max_len else 0
        end = rint+self.max_len if len(sequences['image_array']) > self.max_len else -1
        images = torch.from_numpy(sequences['image_array'].astype(int))[start:end].permute(0,3,1,2)
        masks = torch.from_numpy(sequences['segm_masks'].astype(int))[start:end].permute(0,3,1,2)
        images = F.resize(self.normalize(images), (224, 224))
        masks = F.resize(masks, (224, 224))
        angles = torch.from_numpy(sequences['angle'])[start:end]#*(180/np.pi)
        distances = torch.from_numpy(sequences['distance'])[start:end]
        #angles = angles - self.min_angle/self.range_angle
        res = torch.zeros(len(sequences['angle']))[start:end], images,  masks,  angles.type(torch.float32), distances 
        if self.multitask == "distance":
            res = torch.zeros(len(sequences['angle']))[start:end], images,  masks, distances, angles.type(torch.float32)
        
        return res 