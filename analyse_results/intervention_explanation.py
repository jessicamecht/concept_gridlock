import torch
import clip
from PIL import Image
from utils import *
import pandas as pd
from tqdm import tqdm
import numpy as np
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
save_path = '/data1/shared/jessica/data3/data/toyota/explanation/'
dfs = []
text = clip.tokenize(scenarios).to(device)
scenarios_tokens = scenarios_tokens.to(device)
with torch.no_grad():
    for j, batch in tqdm(enumerate(loader)):
        image_array, vego, angle, distance, gas, brake, cruisestateenabled = batch
        intervention, intervention_before, intervention_after = get_reduced_sample(cruisestateenabled)
        if intervention.sum() == 0: 
            continue
        
        images_before = image_array.squeeze()[intervention_before.squeeze().tolist()]
        images_after = image_array.squeeze()[intervention_after.squeeze().tolist()]
        images_intervention = image_array.squeeze()[intervention.squeeze().tolist()]
        if images_before.shape[0] == 0 or images_after.shape[0] == 0: 
            del image_array, vego, angle, distance, gas, brake
            continue


        print("process")
        logits_per_image_before, logits_per_text_before = model(images_before.to(device), scenarios_tokens.to(device))
        logits_per_image_after, logits_per_text_after = model(images_after.to(device), scenarios_tokens.to(device))
        logits_per_image, logits_per_text = model(images_intervention.to(device), scenarios_tokens.to(device))
        probs_before = logits_per_image_before.softmax(dim=-1).cpu().detach()
        probs_after = logits_per_image_after.softmax(dim=-1).cpu().detach()
        #get the mean of the probabilities
        probs_before = probs_before.mean(dim=0).float()
        probs_after = probs_after.mean(dim=0).float()
        delta = probs_after - probs_before
        print(delta.shape, probs_before.shape, probs_after.shape)
        indic = torch.abs(delta).topk(5, dim=-1).indices
        df = pd.DataFrame([scenarios[i.item()] for i in indic])
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 4))
        axes = axes.flatten()
        print(len(axes), images_before.shape, images_after.shape)
        for i, image in enumerate(axes):
            if i < len(images_before):
                axes[i].imshow((images_before[i].permute(1,2,0)*255).int())
            axes[i].xaxis.set_tick_params(labelbottom=False)
            axes[i].yaxis.set_tick_params(labelleft=False)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
        name = save_path + f"{j}" + "_before.png"
        plt.savefig(name)

        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 4))
        axes = axes.flatten()
        for i, image in enumerate(axes):
            if i < len(images_after):
                axes[i].imshow((images_after[i].permute(1,2,0)*255).int())
            axes[i].xaxis.set_tick_params(labelbottom=False)
            axes[i].yaxis.set_tick_params(labelleft=False)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
        name = save_path + f"{j}" + "_after.png"
        plt.savefig(name)

        df.to_csv(f'{save_path}/{j}.csv')
        
'''for i, img in enumerate(images):
fig, ax = plt.subplots()
ax.imshow(img.permute(1,2,0).int())
fig.suptitle(str(df.iloc[i]))
name = save_path + f"{i}_{j}" + "_image.png"
result_dict = {scenarios[k]: str(probs.squeeze()[i][k].item()) for k in range(len(scenarios))}
with open(name.replace('_image.png', ".json"), 'w') as f:
    json.dump(result_dict, f)
fig.savefig(name)'''
        
