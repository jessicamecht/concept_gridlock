from torch.utils.data import Dataset  # For custom data-sets
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import torch
import h5py
from scipy.spatial.transform import Rotation
from decimal import Decimal
import torchvision.transforms.functional as TF
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
def get_angles(seq):
    new_angles = []
    for i in range(1, len(seq)):
        rotation = Rotation.from_quat(seq[i][:4]).as_euler('zyx', degrees=True)    
        prev_rotation = Rotation.from_quat(seq[i-1][:4]).as_euler('zyx', degrees=True)       
        rot = rotation-prev_rotation
        new_angles.append(rot[0])
    new_angles.append(0)
    return np.array(new_angles)

def my_segmentation_transforms(image, segmentation):
    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)
        segmentation = TF.rotate(segmentation, angle)
    # more transforms ...
    image =  TF.RandomGrayscale(image)
    image = TF.gaussian_blur(image, 3)
    return image, segmentation

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
            data_path1 = "/data1/jessica/data/toyota/once_w_lanes_compressed_raw_small_multitask_all_test.hfd5"
            data_path = "/data1/jessica/data/toyota/once_w_lanes_compressed_raw_small_multitask_all_val.hfd5"
            paths = [data_path, data_path1]  
        elif dataset_type == "val":
            data_path1 = "/data1/jessica/data/toyota/once_w_lanes_compressed_raw_small_multitask_all_test.hfd5"
            data_path = "/data1/jessica/data/toyota/once_w_lanes_compressed_raw_small_multitask_all_val.hfd5"
            paths = [data_path, data_path1]
        self.people_seqs = []
        for data_path in paths:
            with h5py.File(data_path, "r") as f:
                self.all_angle_max = 0 
                self.all_angle_min = 100000
                for i, seq_key in enumerate(list(f.keys())):
                    person_seq = {}
                    keys_ = f[seq_key].keys()
                    for key in keys_:
                        if key == "angle": continue
                        seq = f[seq_key][key][()]#[0::subsample]
                        #maxseq = seq.max()
                        #minseq = seq.min()
                        #self.all_angle_max = maxseq if maxseq > self.all_angle_max else self.all_angle_max
                        #self.all_angle_min = minseq if minseq < self.all_angle_min else self.all_angle_min
                        if key == "pos":
                            seq = get_angles(seq)
                            person_seq["angle"] = seq
                            continue
                        person_seq[key] = seq
                    self.people_seqs.append(person_seq)
                    for i in range(1, len(person_seq['angle'])):
                        person_seq_new = person_seq
                        if len(person_seq[key][i:i+self.max_len+1]) < self.max_len: break
                        for key in person_seq.keys():
                            person_seq_new[key] = person_seq[key][i:i+self.max_len+1]
                        self.people_seqs.append(person_seq_new)
        print("len",len(self.people_seqs))

    def __len__(self):
        return len(self.people_seqs)

    def __getitem__(self, idx):
        sequences = self.people_seqs[idx]#keys are 'angle', 'id', 'image_array', 'lanes_2d', 'lanes_3d', 'meta', 'pos', 'segm_masks', 'seq_name_x', 'speed', 'times'
        rint = random.randint(0,max(0, len(sequences['image_array'])-(self.max_len+1))) #to randomize sequence
        start = 0#rint if len(sequences['image_array']) > self.max_len else 0
        end = self.max_len#rint+self.max_len if len(sequences['image_array']) > self.max_len else -1
        images = torch.from_numpy(sequences['image_array'].astype(float))[start:end].permute(0,3,1,2)
        masks = torch.from_numpy(sequences['segm_masks'].astype(int))[start:end].permute(0,3,1,2)
        images = F.resize(self.normalize(images.type(torch.float)), (224, 224))
        masks = F.resize(masks, (224, 224))
        angles = torch.from_numpy(sequences['angle'])[start:end]#*(180/np.pi)
        distances = torch.from_numpy(sequences['distance'])[start:end]
        #angles = angles - self.min_angle/self.range_angle'
        #images, masks = my_segmentation_transforms(images, masks)
        res = torch.zeros(len(sequences['angle']))[start:end], images.type(torch.float32),  masks.type(torch.float32),  angles.type(torch.float32), distances .type(torch.float32)
        if self.multitask == "distance":
            res = torch.zeros(len(sequences['angle']))[start:end], images.type(torch.float32),  masks.type(torch.float32), distances.type(torch.float32), angles.type(torch.float32)
        
        return res 