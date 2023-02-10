import pytorch_lightning as pl
import pytorch_lightning as pl
import pandas as pd
import torch
from dataloader import *
from dataloader_comma import *
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class LaneModule(pl.LightningModule):
    '''Pytorch lightning module to train angle, distance or multitask procedures'''
    def __init__(self, model, bs, multitask="angle", dataset="comma"):
        super(LaneModule, self).__init__()
        self.model = model
        self.dataset = dataset,
        self.num_workers = 10
        self.multitask = multitask
        self.bs = bs
        self.i = 0
        self.loss = self.mse_loss
        if self.multitask  == "multitask":
            self.distanceloss = self.mse_loss

    def forward(self, x, angle, distance, vego):
        return self.model(x, angle, distance, vego)

    def mse_loss(self, input, target, mask, reduction="mean", task='distance'):
        out = (input[~mask]-target[~mask])**2
        return out.mean() if reduction == "mean" else out 

    def calculate_loss(self, logits, angle, distance, save_name):
        if self.multitask == "multitask":
            logits_angle, logits_dist = logits
            mask = distance.squeeze() == 0.0
            loss_angle = torch.sqrt(self.loss(logits_angle.squeeze(), angle.squeeze(), mask, task="angle"))
            loss_distance = torch.sqrt(self.loss(logits_dist.squeeze(), distance.squeeze(), mask, task="distance"))
            if loss_angle.isnan() or loss_distance.isnan():
                print(loss_angle, loss_distance)
            loss = loss_angle, loss_distance
            self.log_dict({"train_loss_angle": loss_angle}, on_epoch=True, batch_size=self.bs)
            self.log_dict({"train_loss_distance": loss_distance}, on_epoch=True, batch_size=self.bs)
            return loss_angle, loss_distance
        else:
            mask = distance.squeeze() == 0.0
            loss = torch.sqrt(self.loss(logits.squeeze(), angle.squeeze(), mask, task=self.multitask))
            return loss

    def training_step(self, batch, batch_idx):
        _, image_array, vego, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        logits = self(image_array, angle, distance, vego)
        loss = self.calculate_loss(logits, angle, distance, "train")
        if self.multitask == "multitask":
            loss_angle, loss_dist = loss
            loss = (loss_angle + loss_dist)/2
            self.log_dict({"val_loss_dist": loss_dist}, on_epoch=True, batch_size=self.bs)
            self.log_dict({"val_loss_angle": loss_angle}, on_epoch=True, batch_size=self.bs)
        self.log_dict({"train_loss": loss}, on_epoch=True, batch_size=self.bs)
        return loss

    def predict_step(self, batch, batch_idx):
        _, image_array, vego, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        logits = self(image_array, angle, distance, vego)
        return logits, angle, distance

    def validation_step(self, batch, batch_idx):
        _, image_array, vego, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        logits = self(image_array, angle, distance, vego)
        loss = self.calculate_loss(logits, angle, distance, "val")
        
        if self.multitask == "multitask":
            loss_angle, loss_dist = loss
            loss = (loss_angle + loss_dist)/2
            self.log_dict({"val_loss_dist": loss_dist}, on_epoch=True, batch_size=self.bs)
            self.log_dict({"val_loss_angle": loss_angle}, on_epoch=True, batch_size=self.bs)
        self.log_dict({"val_loss": loss}, on_epoch=True, batch_size=self.bs)
        
        return loss

    def test_step(self, batch, batch_idx):
        _, image_array, vego, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        logits = self(image_array, angle, distance, vego)
        loss = self.calculate_loss(logits, angle, distance, "test")

        if self.multitask == "multitask":
            loss_angle, loss_dist = loss
            loss = (loss_angle + loss_dist)/2
            self.log_dict({"test_loss_dist": loss_dist}, on_epoch=True, batch_size=self.bs)
            self.log_dict({"test_loss_angle": loss_angle}, on_epoch=True, batch_size=self.bs)
        self.log_dict({"test_loss": loss}, on_epoch=True, batch_size=self.bs)
        return loss 

    def training_epoch_end(self, outputs):
        losses = torch.mean(torch.stack([x['loss'] for x in outputs]))
        self.log_dict({"train_loss_accumulated": losses }, batch_size=self.bs)

    def validation_epoch_end(self, outputs):
        losses = torch.mean(torch.stack([x for x in outputs]))
        self.log_dict({"val_loss_accumulated": losses }, batch_size=self.bs)

    def test_epoch_end(self, outputs):
        losses = torch.mean(torch.stack([x for x in outputs]))
        self.log_dict({"test_loss_accumulated": losses }, batch_size=self.bs)

    def train_dataloader(self):
        return self.get_dataloader(dataset_type="train")

    def val_dataloader(self):
        return self.get_dataloader(dataset_type="val")

    def test_dataloader(self):
        return self.get_dataloader(dataset_type="test")

    def predict_dataloader(self):
        return self.get_dataloader(dataset_type="test")

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=1e-5)
        return g_opt

    def get_dataloader(self, dataset_type):
        ds = ONCEDataset(dataset_type=dataset_type, multitask=self.multitask) if self.dataset == "once" else CommaDataset(dataset_type=dataset_type, multitask=self.multitask)
        return DataLoader(ds, batch_size=self.bs, num_workers=self.num_workers, collate_fn=pad_collate)
        
def pad_collate(batch):
    '''just in case if there were different sequence lengths, 
    but currently all lengths should be the same when batching'''
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