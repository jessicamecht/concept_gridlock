import pytorch_lightning as pl
from dataloader import *
from model import *
from module import * 
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch 

if __name__ == "__main__":
    model = VTN()
    dataset = ONCEDataset()
    module = LaneModule(model, dataset)
    trainer = pl.Trainer(
        accelerator="mps",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=3,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        )
    trainer.fit(module, dataset)