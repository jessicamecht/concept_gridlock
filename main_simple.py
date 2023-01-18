import pytorch_lightning as pl
from model_simple import *
from module import * 
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from  pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pathlib import Path
def save_preds(logits, target, save_name):
    b, s = target.shape
    df = pd.DataFrame()
    df['logits'] = logits.squeeze().reshape(b*s).tolist()
    df['target'] = target.squeeze().reshape(b*s).tolist()
    Path(f'./{logger.log_dir}').mkdir(parents=True, exist_ok=True)
    df.to_csv(f'./{logger.log_dir}/{save_name}.csv', mode='a', index=False, header=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-task', default="angle", type=str)  
    parser.add_argument('-gpu_num', default=0, type=int) 
    parser.add_argument('-dataset', default="comma", type=str)  
    args = parser.parse_args()
    multitask = args.task
    early_stop_callback = EarlyStopping(monitor="val_loss_accumulated", min_delta=0.05, patience=5, verbose=False, mode="max")
    model = SeqLSTM(multitask=multitask)
    module = LaneModule(model, multitask=multitask, dataset = args.dataset, bs=2)
    ckpt_pth = f"./checkpoints_{args.dataset}_{args.task}"
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val_loss_accumulated")
    logger = TensorBoardLogger(save_dir=ckpt_pth)


    trainer = pl.Trainer(
        #fast_dev_run=True,
        accelerator="gpu",
        devices=[args.gpu_num] if torch.cuda.is_available() else None, 
        logger=logger,
        max_epochs=200,
        default_root_dir=ckpt_pth ,
        callbacks=[TQDMProgressBar(refresh_rate=5), checkpoint_callback],#, EarlyStopping(monitor="train_loss", mode="min")],
        )
    trainer.fit(module)
    ckpt_path='best'#/home/jessica/personalized_driving_toyota/checkpoints_comma_distance/lightning_logs/version_17/checkpoints/epoch=53-step=810.ckpt'#"best"

    preds = trainer.test(module, ckpt_path=ckpt_path)
    if args.task == "angle":
        preds = trainer.predict(module, ckpt_path=ckpt_path)
        for pred in preds:
            predictions, angle, distance = pred[0], pred[1], pred[2]
            save_preds(predictions, angle, "angle")
    elif args.task == "distance":
        preds = trainer.predict(module, ckpt_path=ckpt_path)
        for pred in preds:
            predictions, distance, angle = pred[0], pred[1], pred[2]        
            save_preds(predictions, distance, "dist")
    elif args.task == "multitask":
        preds = trainer.predict(module, ckpt_path=ckpt_path)
        for pred in preds:
            preds_angle, preds_dist, angle, dist = pred[0], pred[1], pred[2], pred[3]
            save_preds(preds_angle, angle, "angle_multi")
            save_preds(preds_dist, dist, "dist_multi")