'''
extract visual features with mini batches
'''

import json 
import os 
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import sys
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader 
import numpy as np
from tqdm import tqdm

def find_imgs(root_path, img_names):
    img_paths = []
    for img_name in img_names:
        # print (img_name)
        img_path = os.path.join(root_path, 'images', img_name[:6], img_name[6:8], img_name[8:10], img_name[10:12], img_name+'.jpg')
        file_dir = img_path.replace('images', 'features')
        file_dir = file_dir.replace('.jpg', '.npy')
        if os.path.exists(file_dir): ## if image feature already extracted, exclude it
            continue
        if os.path.exists(img_path): ## if raw image exists
            save_dir = os.path.dirname(file_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            img_paths.append(img_path)
    return img_paths


def readimgs(img_path):
    img_path = os.path.join(img_path)
    try:
        img = Image.open(img_path).convert('RGB')
    except:
        img = None
    return img 

class VisualExtractor(nn.Module):
    def __init__(self):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = 'resnet101'
        self.pretrained = True
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)
        # self.avg_fnt = torch.nn.AdaptiveMaxPool2d((4,4))

    def forward(self, images):
        mb = images.size(0)
        patch_feats = self.model(images)
        avg_feats = patch_feats.squeeze()

        ## if use feature_map
        # avg_feats = self.avg_fnt(patch_feats).reshape(mb, patch_feats.size(1), -1)
        # avg_feats = avg_feats.permute(0, 2, 1)
        return avg_feats

class image_dataset(Dataset):
    def __init__(self, root_path):

        all_data = np.load(root_path+"old_metadata.npz")

        self.img_paths = ["/home/ayan/toyota/"+d.split('driving_')[1] for d in all_data['image_paths']]
        # for split in ['train', 'val', 'test']:
        print ('total imgs', len(self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = readimgs(img_path)
        img = img_transform(img)
        file_dir = img_path.replace('once/data/', 'once/features/')
        file_dir = file_dir.replace('.jpg', '.npy')
        return img, file_dir


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    root_path = '../data/once/'

    img_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    
    img_data = image_dataset(root_path)
    loader = DataLoader(dataset=img_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=False)
    
    # model = VisualExtractor().to(device)
    # model.eval()

    # with torch.no_grad():
    #     all_imgfeats = []
    #     for (imgs, file_dirs) in tqdm(loader):
    #         # print (type(imgs))
    #         # print (imgs.shape, imgs.dtype)
    #         imgs = imgs.to(device)
    #         feats = model(imgs)
    #         npyfeats = feats.cpu().detach().numpy()
    #         # print (npyfeats.shape)
    #         # sys.exit()
    #         # for i in range(len(npyfeats)):
    #         #   all_imgfeats.append(npyfeats[i])

    #         for i, path in enumerate(file_dirs):
    #             # print (npyfeats[i].shape, path)
    #             # sys.exit()
    #             save_dir = os.path.dirname(path)
    #             if not os.path.exists(save_dir):
    #                 os.makedirs(save_dir)
    #             np.save(path, npyfeats[i])



    data = np.load("../data/once/old_metadata.npz")
    flat_data = data['data'] 
    img_paths = ["/home/ayan/toyota/"+d.split('driving_')[1].replace('once/data//', 'once/features/').replace('.jpg', '.npy')
     for d in data['image_paths']]
    print (img_paths[0])

    with open("../data/once/metadata.npz", 'wb') as f: 
        np.savez_compressed(f, data = flat_data, image_paths = img_paths)






