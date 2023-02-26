from torch.utils.data import Dataset  # For custom data-sets
import torchvision.transforms as transforms
import numpy as np
import torch
import h5py
import cv2

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
        self.max_dist = 70
        self.min_dist = 0
        self.multitask = multitask
        self.min_angle, self.max_angle, self.range_angle = (2.1073424e-08, 0.102598816, 0.102598794)
        self.out_size = out_size
        self.use_transform = use_transform
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.resize = transforms.Resize((224,224))
        self.normalize_values = False
        if dataset_type == "train":
            data_path = "/data1/jessica/data/toyota/comma_train_filtered.h5py"
        elif dataset_type == "test":
            data_path = "/data1/jessica/data/toyota/comma_test_filtered.h5py"
        elif dataset_type == "val":
            data_path = "/data1/jessica/data/toyota/comma_val_filtered.h5py"
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
        images = sequences['image']
        images = images[:,0:160, :,:]#crop the image to remove the view of the inside car console
        images = self.normalize(images.permute(0,3,1,2)/255.0)
        images = self.resize(images)
        images_cropped = images
        #distances[distances > self.max_dist] = 0
        #distances = ((distances - min_dist) / (max_dist - min_dist))
        res = images_cropped, images_cropped,  sequences['vEgo'],  sequences['angle'], distances
        if self.multitask == "distance":
            res = images_cropped, images_cropped, sequences['vEgo'], distances, sequences['angle']
        return res 