import pytorch_lightning as pl
from model import *
from module import * 
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-task', default="angle", type=str)  
    parser.add_argument('-gpu_num', default=0, type=int)  
    args = parser.parse_args()
    multitask = args.task
    model = VTN(multitask=multitask)
    module = LaneModule(model, multitask=multitask)
    trainer = pl.Trainer(
        #fast_dev_run=True,
        accelerator="gpu",
        devices=[args.gpu_num] if torch.cuda.is_available() else None, 
        resume_from_checkpoint = "/home/jessica/personalized_driving_toyota/checkpoints_multitask/lightning_logs/version_6/checkpoints/epoch=148-step=26969.ckpt",
        max_epochs=500,
        default_root_dir=f"./checkpoints_{args.task}" ,
        callbacks=[TQDMProgressBar(refresh_rate=20)],#, EarlyStopping(monitor="train_loss", mode="min")],
        )
    #trainer.fit(module)
    trainer.test(module)