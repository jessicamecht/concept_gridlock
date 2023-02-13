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
            data_path = "/data1/jessica/data/toyota/train_comma_1.hfd5"
            data_path2 = "/data1/jessica/data/toyota/train_comma_2.hfd5"
            data_path3 = "/data1/jessica/data/toyota/train_comma_c2.hfd5"
            data_path4 = "/data1/jessica/data/toyota/train_comma_c3.hfd5"
        elif dataset_type == "test":
            data_path = "/data1/jessica/data/toyota/test_comma_1.hfd5"
            data_path2 = "/data1/jessica/data/toyota/test_comma_c2.hfd5"
            data_path3 = "/data1/jessica/data/toyota/test_comma_c3.hfd5"
        elif dataset_type == "val":
            data_path = "/data1/jessica/data/toyota/val_comma_1.hfd5"
            data_path2 = "/data1/jessica/data/toyota/val_comma_c2.hfd5"
            data_path3 = "/data1/jessica/data/toyota/val_comma_c3.hfd5"
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
            good_keys = [0, 1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20, 21, 24, 25, 27, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 45, 46, 50, 52, 53]#[0, 1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 45, 46, 50, 52, 53]
            gk2 = np.array([55, 56, 57, 60, 62, 64, 65, 66, 67, 68, 69, 70, 72, 73, 76, 77, 78, 81, 82, 84, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99, 100, 103, 104, 105, 106]) - 55# np.array([55, 56, 57, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 75, 76, 77, 78, 81, 82, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 103, 104, 105, 106]) -55
            self.keys = list(np.array(self.keys)[good_keys])
            self.keys2 = list(np.array(self.keys2)[gk2])[:-1]
            self.h5_file3 = h5py.File(data_path3, "r")
            self.keys3 = list(self.h5_file3.keys())
            self.h5_file4 = h5py.File(data_path4, "r")
            self.keys4 = list(self.h5_file4.keys())
        else:
            self.keys4 = []

        if dataset_type == "val":
            good_keys = [1, 6, 9, 10, 11, 12, 14, 15]
            self.keys = np.array(self.keys)[good_keys]
            self.h5_file2 = h5py.File(data_path2, "r")
            self.keys2 = list(self.h5_file2.keys())
            self.h5_file3 = h5py.File(data_path3, "r")
            self.keys3 = list(self.h5_file3.keys())
            
        if dataset_type == "test":
            good_keys = [0, 6, 8, 9, 10, 13]
            self.keys = np.array(self.keys)[good_keys]
            self.h5_file2 = h5py.File(data_path2, "r")
            self.keys2 = list(self.h5_file2.keys())[0:3] + list(self.h5_file2.keys())[5:]
            self.h5_file3 = h5py.File(data_path3, "r")
            self.keys3 = list(self.h5_file3.keys())[:-1]
           

    def __len__(self):
        return len(self.keys) + len(self.keys2) + len(self.keys3) + len(self.keys4)

    def __getitem__(self, idx):
        person_seq = {}
        if idx < len(self.keys):
            seq_key  = self.keys[idx]
            keys_ = self.h5_file[seq_key].keys()#'angle', 'brake', 'dist', 'gas', 'image', 'time', 'vEgo'
            file = self.h5_file
        elif idx < len(self.keys) + len(self.keys2):
            seq_key  = self.keys2[idx - len(self.keys)]
            keys_ = self.h5_file2[seq_key].keys()
            file = self.h5_file2
        elif idx < len(self.keys) + len(self.keys2) + len(self.keys3):
            seq_key  = self.keys3[idx - len(self.keys) - len(self.keys2)]
            keys_ = self.h5_file3[seq_key].keys()
            file = self.h5_file3
        else:
            seq_key  = self.keys4[idx - len(self.keys) - len(self.keys2) - len(self.keys3)]
            keys_ = self.h5_file4[seq_key].keys()
            file = self.h5_file4

        for key in keys_:                        
            seq = file[seq_key][key][()]
            seq = seq if len(seq) <= 241 else seq[1::5]
            person_seq[key] = torch.from_numpy(np.array(seq[0:self.max_len]).astype(float)).type(torch.float32)
        sequences = person_seq
        distances = sequences['dist']
        images = sequences['image']
        images = images[:,0:160, :,:]
        images = self.normalize(images.permute(0,3,1,2)/255.0)
        images = self.resize(images)
        images_cropped = images
        #distances[distances > self.max_dist] = 0
        #distances = ((distances - min_dist) / (max_dist - min_dist))
        res = images_cropped, images_cropped,  sequences['vEgo'],  sequences['angle'], distances
        if self.multitask == "distance":
            res = images_cropped, images_cropped, sequences['vEgo'], distances, sequences['angle']
        return res 