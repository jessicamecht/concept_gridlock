from torch.utils.data import Dataset  # For custom data-sets
import torchvision.transforms as transforms
import numpy as np
import torch
import h5py

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
        self.normalize_values = False
        if dataset_type == "train":
            data_path = "/data1/jessica/data/toyota/train_comma_1.hfd5"
            data_path2 = "/data1/jessica/data/toyota/train_comma_2.hfd5"
        elif dataset_type == "test":
            data_path = "/data1/jessica/data/toyota/test_comma_1.hfd5"
        elif dataset_type == "val":
            data_path = "/data1/jessica/data/toyota/val_comma_1.hfd5"
        self.people_seqs = []
        self.h5_file = h5py.File(data_path, "r")
        self.keys = list(self.h5_file.keys())
        self.keys.remove('10')
        self.keys.remove('17')
        if dataset_type == "train":
            self.keys.remove('37')
            self.keys.remove('53')
            self.keys.remove('55')
            self.keys.remove('58')
            self.h5_file2 = h5py.File(data_path2, "r")
            self.keys2 = list(self.h5_file2.keys())
        else:
            self.keys2 = []

    def __len__(self):
        return len(self.keys) + len(self.keys2)

    def __getitem__(self, idx):
        person_seq = {}
        if idx < len(self.keys):
            seq_key  = self.keys[idx]
            keys_ = self.h5_file[seq_key].keys()#'angle', 'brake', 'dist', 'gas', 'image', 'time', 'vEgo'
        else:
            seq_key  = self.keys2[idx - len(self.keys)]
            keys_ = self.h5_file2[seq_key].keys()#'angle', 'brake', 'dist', 'gas', 'image', 'time', 'vEgo'
        for key in keys_:                        
            seq = self.h5_file[seq_key][key][()]
            seq = seq if len(seq) <= 240 else seq[1::5]
            person_seq[key] = torch.from_numpy(np.array(seq[0:self.max_len]).astype(float)).type(torch.float32)
        sequences = person_seq
        distances = sequences['dist']
        images = self.normalize(sequences['image'].permute(0,3,1,2))
        
        distances[distances > self.max_dist] = 0
        #distances = ((distances - min_dist) / (max_dist - min_dist))
        res = images, images,  sequences['vEgo'],  sequences['angle'], distances
        if self.multitask == "distance":
            res = images, images, sequences['vEgo'], distances, sequences['angle']
        return res 