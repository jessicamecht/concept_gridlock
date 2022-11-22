import pytorch_lightning as pl
import pytorch_lightning as pl
import torch
from dataloader import *
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class LaneModule(pl.LightningModule):

    def __init__(self, model, bs=1, multitask=False):
        super(LaneModule, self).__init__()
        self.model = model
        self.num_workers = 10
        self.multitask = multitask
        self.bs = bs
        self.loss = nn.MSELoss()
        if self.multitask:
            self.distanceloss = nn.MSELoss()
    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, logits, angle, distance):
        if self.multitask:
            logits_angle, logits_dist = logits
            loss_angle = self.loss(logits_angle.squeeze(), angle.squeeze())
            loss_distance = self.loss(logits_distance.squeeze(), distance.squeeze())
            loss = (loss_angle + loss_distance)/2
        else:
            loss = self.loss(logits.squeeze(), angle.squeeze())
        return loss
    def training_step(self, batch, batch_idx):
        meta, image_array, segm_masks, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        logits = self(image_array)
        print('logits', logits.dtype, angle.dtype)
        loss = self.calculate_loss(logits, angle, distance)
        self.log_dict({"train_loss": loss}, on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx):
        meta, image_array, segm_masks, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        logits = self(image_array)
        print('logits', logits.dtype, angle.dtype)
        loss = self.calculate_loss(logits, angle, distance)
        self.log_dict({"val_loss": loss}, on_epoch=True)
        return loss
    def test_step(self, batch, batch_idx):
        meta, image_array, segm_masks, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        logits = self(image_array)
        print('logits', logits.dtype, angle.dtype)
        loss = self.calculate_loss(logits, angle, distance)
        self.log_dict({"test_loss": loss}, on_epoch=True)
        return loss 

    def training_epoch_end(self, outputs):
        losses = torch.mean(torch.stack([x['loss'] for x in outputs]))
        print("train losses", losses)
        self.log_dict({"train_loss_accumulated": losses })
    def validation_epoch_end(self, outputs):
        print(outputs)
        losses = torch.mean(torch.stack([x for x in outputs]))
        print("val losses", losses)
        self.log_dict({"val_loss_accumulated": losses })
    def test_epoch_end(self, outputs):
        losses = torch.mean(torch.stack([x for x in outputs]))
        print("test losses", losses)
        self.log_dict({"test_loss_accumulated": losses })

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

def pad_collate1(batch):
    meta, img, segm, angle, dist = zip(*batch)
    pads = ()
    for l in [meta, img, segm, angle, dist]:
        lens = [len(x) for x in l] if l[0] != None else None
        pads = pads + (pad_sequence(lens, batch_first=True, padding_value=0))
    return pads

def pad_collate(batch):
    meta, img, segm, angle, dist = zip(*batch)
    m_lens = [len(x) for x in meta]
    i_lens = [len(y) for y in img]
    s_lens = [len(x) for x in segm]
    a_lens = [len(y) for y in angle]
    d_lens = [len(y) for y in dist] if dist[0] != None else None 


    m_pad = pad_sequence(meta, batch_first=True, padding_value=0)
    i_pad = pad_sequence(img, batch_first=True, padding_value=0)
    segm_pad = pad_sequence(segm, batch_first=True, padding_value=0)
    a_pad = pad_sequence(angle, batch_first=True, padding_value=0)
    d_pad = pad_sequence(dist, batch_first=True, padding_value=0) if dist[0] != None else None 
    return m_pad, i_pad, segm_pad, a_pad,d_pad, m_lens, i_lens, s_lens, a_lens, d_lens