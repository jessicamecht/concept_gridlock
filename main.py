import pytorch_lightning as pl
from model import *
from module import * 
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-task', default="angle", type=str)  
    parser.add_argument('-gpu_num', default=0, type=int) 
    parser.add_argument('-dataset', default="comma", type=str)  
    args = parser.parse_args()
    multitask = args.task
    early_stop_callback = EarlyStopping(monitor="val_loss_accumulated", min_delta=0.05, patience=5, verbose=False, mode="max")
    model = VTN(multitask=multitask)
    module = LaneModule(model, multitask=multitask, dataset = args.dataset)
    ckpt_pth = f"./checkpoints_{args.dataset}_{args.task}"
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_pth, save_top_k=2, monitor="val_loss_accumulated")

    trainer = pl.Trainer(
        #fast_dev_run=True,
        accelerator="gpu",
        devices=[args.gpu_num] if torch.cuda.is_available() else None, 
        #resume_from_checkpoint = "/home/jessica/personalized_driving_toyota/checkpoints_comma_angle/lightning_logs/version_0/checkpoints/epoch=439-step=9240.ckpt",
        max_epochs=500,
        default_root_dir=ckpt_pth ,
        callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback],#, EarlyStopping(monitor="train_loss", mode="min")],
        )
    trainer.fit(module)
    trainer.test(module)