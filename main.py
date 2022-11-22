import pytorch_lightning as pl
from model import *
from module import * 
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


if __name__ == "__main__":
    model = VTN(multitask=False)
    module = LaneModule(model)
    trainer = pl.Trainer(
        #fast_dev_run=True,
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None, 
        max_epochs=10000,
        default_root_dir="./checkpoints",
        callbacks=[TQDMProgressBar(refresh_rate=20)],#, EarlyStopping(monitor="train_loss", mode="min")],
        )
    trainer.fit(module)
    trainer.test(module)