import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np


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



class LaneModule(pl.LightningModule):

    def __init__(self, model, dataset, bs=1):
        super(LaneModule, self).__init__()
        self.dataset = dataset
        self.model = model
        self.bs = bs
    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)
    def training_step(self, batch, batch_idx):
        meta, image_array, segm_masks, angle, m_lens, i_lens, s_lens, a_lens = batch
        logits = self(image_array)
        loss = F.nll_loss(logits, angle)
        return loss
    def validation_step(self, batch, batch_idx):
        meta, image_array, segm_masks, angle, m_lens, i_lens, s_lens, a_lens = batch
        logits = self(image_array)
        loss = F.nll_loss(logits, angle)
        return loss
    def test_step(self, batch, batch_idx):
        meta, image_array, segm_masks, angle, m_lens, i_lens, s_lens, a_lens = batch
        logits = self(image_array)
        loss = F.nll_loss(logits, angle)
        return loss
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.bs, collate_fn=pad_collate)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.bs, collate_fn=pad_collate)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.bs, collate_fn=pad_collate)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        return g_opt