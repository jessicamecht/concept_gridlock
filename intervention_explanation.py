import torch
import clip
from PIL import Image
from utils import *
from tqdm import tqdm
from dataloader_comma import *
import json
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = "cuda:2"
model, preprocess = clip.load("ViT-L/14", device=device)
multitask = "distance"
dataset_type = 'train'
dataset = CommaDataset(dataset_type=dataset_type, multitask=multitask, return_full=True)
loader = DataLoader(dataset, batch_size=1, num_workers=5)
save_path = '/data3/jessica/data/toyota/explanation/'

text = clip.tokenize(scenarios).to(device)

for j, batch in tqdm(enumerate(loader)):
    image_array, vego, angle, distance, gas, brake = batch
    intervention = gas | brake
    if intervention.sum() == 0: 
        continue
    
    images = image_array.squeeze()[intervention.squeeze().tolist()]
    if images.shape[0] == 0: 
        del image_array, vego, angle, distance, gas, brake
        continue

    #fig, ax = plt.subplots()
    img = images
    print(';lkjhgfghjkl', images.shape)
    s = img.shape#[batch_size, seq_len, h,w,c]
    logits_per_image, logits_per_text = model(img.to(device), scenarios_tokens.to(device))
    probs = logits_per_image.detach()
    probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
    #ax.imshow(image.permute(1,2,0).int())
    #fig.suptitle(scenarios[probs.argmax()])
    #name = save_path + f"{i}_{j}" + "_image.png"

     #fig.savefig(name)
    #result_dict = {scenarios[i]: str(scenarios_tokens[probs.argmax()]) for i in range(len(scenarios))}
    with open(f"{j}.json", 'w') as f:
        json.dump(scenarios_tokens[probs.argmax()], f)
    del image_array, vego, angle, distance, gas, brake
