import pytorch_lightning as pl
from model import *
from module import * 
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from  pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pathlib import Path
import pandas as pd 

def save_preds(logits, target, save_name):
    b, s = target.shape
    df = pd.DataFrame()
    df['logits'] = logits.squeeze().reshape(b*s).tolist()
    df['target'] = target.squeeze().reshape(b*s).tolist()
    Path(f'./{logger.log_dir}').mkdir(parents=True, exist_ok=True)
    df.to_csv(f'./{logger.log_dir}/{save_name}.csv', mode='a', index=False, header=False)

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', default="angle", type=str)  
    parser.add_argument('-train', default=True, type=bool)  
    parser.add_argument('-gpu_num', default=0, type=int) 
    parser.add_argument('-dataset', default="comma", type=str)  
    parser.add_argument('-backbone', default="resnet", type=str) 
    parser.add_argument('-concept_features', default=False, type=bool) 
    parser.add_argument('-bs', default=1, type=int) 
    parser.add_argument('-ground_truth', default="desired", type=str) 
    parser.add_argument('-dev_run', default=False, type=bool) 
    parser.add_argument('-checkpoint_path', default='/home/jessica/personalized_driving_toyota/checkpoints_comma_angle/lightning_logs/version_17/checkpoints/epoch=21-step=1694.ckpt', type=str)
    return parser

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    multitask = args.task
    early_stop_callback = EarlyStopping(monitor="val_loss_accumulated", min_delta=0.05, patience=5, verbose=False, mode="max")
    model = VTN(multitask=multitask, backbone=args.backbone, concept_features=args.concept_features, device = f"cuda:{args.gpu_num}")
    module = LaneModule(model, multitask=multitask, dataset = args.dataset, bs=args.bs, ground_truth=args.ground_truth)

    ckpt_pth = f"/data1/jessica/data/toyota/ckpts/ckpts_desired{args.dataset}_{args.task}"
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val_loss_accumulated")
    logger = TensorBoardLogger(save_dir=ckpt_pth)

    trainer = pl.Trainer(
        fast_dev_run=args.dev_run,
        accelerator="gpu",
        devices=[args.gpu_num] if torch.cuda.is_available() else None, 
        logger=logger,
        max_epochs=100,
        default_root_dir=ckpt_pth ,
        callbacks=[TQDMProgressBar(refresh_rate=5), checkpoint_callback],
        #, EarlyStopping(monitor="train_loss", mode="min")],#in case we want early stopping
        )
    if args.train:
        trainer.fit(module)
    else:
        ckpt_path=args.checkpoint_path
        preds = trainer.test(module, ckpt_path=ckpt_path)
        preds = trainer.predict(module, ckpt_path=ckpt_path)
        for pred in preds:
            if args.task != "multitask":
                predictions, preds_1, preds_2 = pred[0], pred[1], pred[2] 
                save_preds(predictions, preds_1, args.task)
            else:
                preds, angle, dist = pred[0], pred[1], pred[2]
                preds_angle, preds_dist = preds[0], preds[1]
                save_preds(preds_angle, angle, "angle_multi")
                save_preds(preds_dist, dist, "dist_multi")