import pytorch_lightning as pl
from model import *
from module import * 
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch 

if __name__ == "__main__":
    model = VTN()
    module = LaneModule(model)
    trainer = pl.Trainer(
        #fast_dev_run=True,
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=60,
        default_root_dir="/home/jessica/toyota/personalized_driving_toyota/checkpoints",
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        )
    trainer.fit(module)
    trainer.test(module)