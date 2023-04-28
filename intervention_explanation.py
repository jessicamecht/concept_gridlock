import torch
import clip
from PIL import Image
from utils import *
import pandas as pd
from tqdm import tqdm
from dataloader_comma import *
import json
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = "cuda:1"
model, preprocess = clip.load("ViT-L/14", device=device)
multitask = "distance"
dataset_type = 'train'
dataset = CommaDataset(dataset_type=dataset_type, multitask=multitask, return_full=True)
loader = DataLoader(dataset, batch_size=1, num_workers=5)
save_path = '/data3/jessica/data/toyota/explanation/'

text = clip.tokenize(scenarios).to(device)
scenarios_tokens = scenarios_tokens.to(device)
with torch.no_grad():
    for j, batch in tqdm(enumerate(loader)):
        image_array, vego, angle, distance, gas, brake = batch
        intervention = gas | brake
        if intervention.sum() == 0: 
            continue
        
        images = image_array.squeeze()[intervention.squeeze().tolist()]
        if images.shape[0] == 0: 
            del image_array, vego, angle, distance, gas, brake
            continue

        
        img = images
        s = img.shape#[batch_size, seq_len, h,w,c]
        logits_per_image, logits_per_text = model(img.to(device), scenarios_tokens.to(device))
        probs = logits_per_image.detach()
        probs = logits_per_image.softmax(dim=-1).cpu().detach()
        df = pd.DataFrame([scenarios[i.item()] for i in probs.argmax(dim=-1)])
        df.to_csv(f'{save_path}/{j}.csv')
        
        for i, img in enumerate(images):
            fig, ax = plt.subplots()
            ax.imshow(img.permute(1,2,0).int())
            fig.suptitle(str(df.iloc[i]))
            name = save_path + f"{i}_{j}" + "_image.png"
            result_dict = {scenarios[k]: str(probs.squeeze()[i][k].item()) for k in range(len(scenarios))}
            with open(name.replace('_image.png', ".json"), 'w') as f:
                json.dump(result_dict, f)
            fig.savefig(name)
        
