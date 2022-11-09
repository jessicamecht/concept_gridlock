import pytorch_lightning as pl
import torch
from dataloader import *
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class LaneModule(pl.LightningModule):

    def __init__(self, model, bs=1):
        super(LaneModule, self).__init__()
        self.model = model
        self.num_workers = 10
        self.bs = bs
        self.loss = nn.MSELoss()
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        meta, image_array, segm_masks, angle, m_lens, i_lens, s_lens, a_lens = batch
        logits = self(image_array)
        loss = self.loss(logits.squeeze(), angle.squeeze())
        self.log_dict({"loss1": loss})
        return loss
    def validation_step(self, batch, batch_idx):
        meta, image_array, segm_masks, angle, m_lens, i_lens, s_lens, a_lens = batch
        logits = self(image_array)
        loss = self.loss(logits.squeeze(), angle.squeeze())
        return loss
    def test_step(self, batch, batch_idx):
        meta, image_array, segm_masks, angle, m_lens, i_lens, s_lens, a_lens = batch
        logits = self(image_array)
        loss = self.loss(logits.squeeze(), angle.squeeze())
        return loss
    def training_epoch_end(self, outputs):
        losses = torch.mean(torch.stack([x['loss'] for x in outputs]))
        self.log_dict({"loss_epoch1": losses })

    def train_dataloader(self):
        return self.get_dataloader(dataset_type="train")

    def val_dataloader(self):
        return self.get_dataloader(dataset_type="val")

    def test_dataloader(self):
        return self.get_dataloader(dataset_type="test")

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        return g_opt

    def get_dataloader(self, dataset_type):
        return DataLoader(ONCEDataset(dataset_type=dataset_type), batch_size=self.bs, num_workers=self.num_workers, collate_fn=pad_collate)

def pad_collate(batch):
    meta, img, segm, angle = zip(*batch)
    m_lens = [len(x) for x in meta]
    i_lens = [len(y) for y in img]
    s_lens = [len(x) for x in segm]
    a_lens = [len(y) for y in angle]

    m_pad = pad_sequence(meta, batch_first=True, padding_value=0)
    i_pad = pad_sequence(img, batch_first=True, padding_value=0)
    segm_pad = pad_sequence(segm, batch_first=True, padding_value=0)
    a_pad = pad_sequence(angle, batch_first=True, padding_value=0)

    return m_pad, i_pad, segm_pad, a_pad, m_lens, i_lens, s_lens, a_lens