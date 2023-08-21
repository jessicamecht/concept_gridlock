import pandas as pd
import numpy
import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append("..")
from utils import pad_collate
from dataloader_comma import CommaDataset
from dataloader_nuscenes import NUScenesDataset
from collections import Counter
import imageio
from model import VTN
import matplotlib.pyplot as plt 
from PIL import Image
import glob
plt.rcParams.update({'font.size': 26}) 
import os
from utils import * 
import re
from vis_utils import * 
from tqdm import tqdm
import warnings 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
print('1')
ckpt = torch.load(checkpoint_path)
print('2')
state_dict = ckpt['state_dict']
state_dict = get_regular_ckpt_from_lightning_checkpoint(state_dict)
model.load_state_dict(state_dict)
model.eval()
model = model.to(gpu)
print('3')

def split_string(string):
    words = string.replace("a photo of driving on a highway with", "").replace("a photo of", "").replace("driving on a highway", "").replace("past", "").replace("a street with", "").split()  # Split the string into a list of words
    result = []
    current_line = ""
    
    for word in words:
        if len(current_line) + len(word) <= 33:
            current_line += word + " "
        else:
            result.append(current_line.strip())
            current_line = word + " "
    
    if current_line:
        result.append(current_line.strip())
    
    return "\n".join(result)

for j, batch in tqdm(enumerate(dataloader_comma)):
    image_array,  vego, angle, distance, g, s, l = batch
    print(image_array.shape,  vego.shape, angle.shape, distance.shape)
    img = image_array
    max_len = 240
    img, angle, distance, vego = img.to(gpu), angle.to(gpu), distance.to(gpu), vego.to(gpu)
    (logits, attns), concepts = model(img, angle, distance, vego)
    top5_indices = torch.tensor(concepts.squeeze()).topk(10).indices
    s = img.shape
    angle, distance, vego, logits, concepts = angle.to("cpu"), distance.to("cpu"), vego.to("cpu"), logits.detach().cpu().to("cpu"), concepts.detach().cpu().to("cpu")
    
    f = []
    inter = []
    for i, elem0 in enumerate(top5_indices):
        inter = []
        for elem in top5_indices[max(i-20, 0):min(i+20,len(top5_indices))]:
            l = elem.cpu().numpy().tolist()
            if 131 in l:
                l.remove(131)
            if 55 in l:
                l.remove(55)
            inter.extend(l)
        count_dict = Counter(inter)
        # Get the top 5 most occurring numbers
        top_5 = count_dict.most_common(3)
        intermediate = []
        for a in top_5: 
            intermediate.append(scenarios[a[0]])
        f.append(intermediate)

    fig, axes = plt.subplots(1, 1, figsize=(12, 16))#,gridspec_kw= {'height_ratios': [20, 1]})

    plt_idx = 0
    for i, image in tqdm(enumerate(img[0][10:120])):
    
        image_frame = (image).cpu().permute(1, 2, 0)#unorm(image).cpu().permute(1, 2, 0)

        # Display the image frame
        axes.imshow((np.array(image_frame) * 255).astype(np.uint8))
        
        title = '\n'.join([split_string("\u2022 " + h) for h in f[i]])
        axes.set_title(title)
        axes.set_aspect('equal')
        axes.set_xticks([])
        axes.set_yticks([])

        # Remove borders
        axes.spines['top'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.spines['left'].set_visible(False)
        axes.spines['right'].set_visible(False)

        plt.savefig(f"/home/jessica/personalized_driving_toyota/result_images/mturk/{i}.png")

    
    image_directory = '/home/jessica/personalized_driving_toyota/result_images/mturk/'

    # Set the output GIF file path
    output_path = lambda x: f'/home/jessica/personalized_driving_toyota/result_images/mturk/att/attention_comma_{j}.{x}'

    # Set the duration (in milliseconds) for each frame in the GIF
    frame_duration = 700

    # Get a sorted list of image files in the directory
    image_files = sorted(glob.glob(f'{image_directory}/*.png'), key=extract_number)  # Adjust the file extension if necessary

    # Create a list to store the frames of the GIF
    frames = []

    # Iterate over each image file
    for image_file in image_files:
        # Open the image file
        image = Image.open(image_file)

        # Add the image to the list of frames
        frames.append(image)

    # Save the frames as a GIF
    frames[0].save(output_path("gif"), format='GIF', append_images=frames[1:], save_all=True,
                duration=frame_duration, loop=0)
    imageio.mimsave(output_path("mp4"), frames, fps=4)
    break
    if j > 50: break