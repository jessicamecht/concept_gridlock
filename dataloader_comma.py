from torch.utils.data import Dataset  # For custom data-sets
import torchvision.transforms as transforms
import numpy as np
import torch
import h5py
from scipy import ndimage
import cv2

class CommaDataset(Dataset):
    def __init__(
        self,
        dataset_type="train",
        use_transform=False,
        multitask="angle",
        ground_truth="desired",
        return_full=False, 
        dataset_path = None,
        dataset_fraction=1.0
    ):
        assert dataset_type in ["train", "val", "test"]
        if dataset_type == "val":
            dataset_type = "test" 
        if dataset_type == "test":
            dataset_type = "val" 
        
        self.dataset_type = dataset_type
        self.dataset_fraction = dataset_fraction
        self.max_len = 240
        self.ground_truth = ground_truth
        self.multitask = multitask
        self.use_transform = use_transform
        self.return_full = return_full
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.resize = transforms.Resize((224,224))
        #/data1/shared/jessica/data1/data/
        data_path = f"{dataset_path}/comma/comma_{dataset_type}_filtered.h5py" if ground_truth == "regular" else f"{dataset_path}/comma/comma_{dataset_type}_w_desired_filtered.h5py"
        self.people_seqs = []
        self.h5_file = h5py.File(data_path, "r")
        corrupt_idx = 62
        self.keys = list(self.h5_file.keys())
        if dataset_type == "train":
            self.keys.pop(corrupt_idx)
           
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        person_seq = {}
        seq_key  = self.keys[idx]
        keys_ = self.h5_file[seq_key].keys()#'angle', 'brake', 'dist', 'gas', 'image', 'time', 'vEgo'
        file = self.h5_file
        
        for key in keys_:                        
            seq = file[seq_key][key][()]
            seq = seq if len(seq) <= 241 else seq[1::5]
            person_seq[key] = torch.from_numpy(np.array(seq[0:self.max_len]).astype(float)).type(torch.float32)
        sequences = person_seq
        distances = sequences['dist']
        distances = ndimage.median_filter(distances, size=128, mode='nearest')

        steady_state = ~np.array(sequences['gaspressed']).astype(bool) & ~np.array(sequences['brakepressed']).astype(bool) & ~np.array(sequences['leftBlinker']).astype(bool) & ~np.array(sequences['rightBlinker']).astype(bool)
        last_idx = 0
        desired_gap = np.zeros(distances.shape)

        for i in range(len(steady_state)-1):
            if steady_state[i] == True:
                desired_gap[last_idx:i] = int(distances[i])
                last_idx = i
        desired_gap[-12:] = distances[-12:].mean().item()

        distances = sequences['dist'] if self.ground_truth else desired_gap
        images = sequences['image']
        images = images[:,0:160, :,:]#crop the image to remove the view of the inside car console
        images = images.permute(0,3,1,2)
        if not self.return_full:
            images = self.normalize(images/255.0)
            
        else:
            images = images/255.0
        images = self.resize(images)
        images_cropped = images
        intervention = np.array(sequences['gaspressed']).astype(bool) | np.array(sequences['brakepressed']).astype(bool) 
        res = images_cropped, images_cropped,  sequences['vEgo'],  sequences['angle'], distances
        if self.return_full: 
            return images_cropped,  sequences['vEgo'],  sequences['angle'], distances, np.array(sequences['gaspressed']).astype(bool),  np.array(sequences['brakepressed']).astype(bool) , np.array(sequences['CruiseStateenabled']).astype(bool)
        if self.multitask == "distance":
            res = images_cropped, images_cropped, sequences['vEgo'], distances, sequences['angle']
        if self.multitask == "intervention":
            res = images_cropped, images_cropped, sequences['vEgo'], distances, torch.tensor(np.array(sequences['gaspressed']).astype(bool) | np.array(sequences['brakepressed']).astype(bool))
        return res 