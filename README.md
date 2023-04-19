# Personalized Driving Project

To execute the training (or testing respectively -- make sure to set the testing checkpoint path accordingly), run 

`python main.py -task angle -train True -gpu_num 0 -dataset comma -backbone resnet -bs 3 -dev_run False` 

It is based on pytorch lightning. Most of the training and testing logic is defined in **module.py** as a pytorch lightning module. 
The model logic for the Visual Transformer Network is defined in **model.py**.

The ONCE dataloader can be found in **dataloader.py** (deprecated).

The Comma 2k19 dataloader can be found in **dataloader_comma.py**.

The gradient visualization can be found in **gradram.ipynb**.

The checkpoints are in the google drive folder and can be used to run/continue training or testing.

### Dependencies: 
```
pytorch_lightning
transformers
timm
numpy
pandas
h5py
scipy
opencv-python
tensorboard
```

______

# Pre-processing Comma Data
comma_preprocess/raw_readers.ipynb reads the chunks from the comma dataset and combines them to h5py file
It uses components of Comma openpilot (https://github.com/commaai/openpilot)


You can download the comma data here: https://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb
