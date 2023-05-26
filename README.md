# Personalized Driving Project

To execute the training (or testing respectively -- make sure to set the testing checkpoint path accordingly), run 

`python3 main.py -task distance -train -gpu_num 1 -dataset comma -backbone resnet -bs 4 -ground_truth normal`

where backbone can be one of { clip, resnet, none }, backbone can be one of { normal, desired }, task  can be one of { distance, multitask, angle }, dataset can be one of { comma, nuscenes }, -train is a flag that should not be set if you want to run inference. -concept_features flag should only be set together with -backbone none

I am using Python version 3.10.9.

It is based on pytorch lightning. Most of the training and testing logic is defined in **module.py** as a pytorch lightning module. 
The model logic for the Visual Transformer Network is defined in **model.py**.

The ONCE dataloader can be found in **dataloader.py** (deprecated).

The Comma 2k19 dataloader can be found in **dataloader_comma.py**.

The NuScenes dataloader can be found in **dataloader_nuschenes.py**.

The gradient visualization can be found in **gradram.ipynb**.

Analysis of the results can be found in **/data_clean_up/analyse_results.ipynb**.

**/scenarios/** has the different scenarios for the clip evaluation.

The checkpoints for the comma dataset are in the google drive folder and can be used to run/continue training or testing.

### Dependencies: 
All requirements can be found in **requirements.txt**

______

# Pre-processing Comma Data
comma_preprocess/raw_readers.ipynb reads the chunks from the comma dataset and combines them to h5py file
It uses components of Comma openpilot (https://github.com/commaai/openpilot)


You can download the comma data here: https://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb
