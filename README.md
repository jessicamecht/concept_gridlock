# personalized_driving_toyota

To execute the training (or testing respectively -- make sure to set the testing checkpoint path accordingly), run 

`python main.py -task angle -train True -gpu_num 0 -dataset comma -backbone resnet -bs 3 -dev_run False` 

It is based on pytorch lightning. Most of the training and testing logic is defined in **module.py** as a pytorch lightning module. 
The model logic for the Visual Transformer Network is defined in **model.py**.

The ONCE dataloader can be found in **dataloader.py** (deprecated).

The Comma 2k19 dataloader can be found in **dataloader_comma.py**.

The gradient visualization can be found in **gradram.ipynb**.

The checkpoints are in the google drive folder and can be used to run/continue training or testing.


