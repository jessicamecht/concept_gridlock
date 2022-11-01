from ast import Mult
import torch
import numpy as np
import torch.nn as nn

from torch_utils import LongTensor, FloatTensor, BoolTensor, is_cuda_available
from pytorch_models.Transformer import MultiHeadAttention, PointWiseFeedForward, Transformer

class GapFormer(Transformer):
    def __init__(self, hyper_params):
        super(GapFormer, self).__init__(hyper_params)
        self.hyper_params = hyper_params
        latent_size = self.hyper_params['latent_size']

        # Initial Embedding
        self.first_layer = nn.Linear(1 * (self.hyper_params['context_size'] + 1), latent_size, bias = True)
        nn.init.xavier_uniform_(self.first_layer.weight)
        self.emb_dropout = torch.nn.Dropout(p=hyper_params['dropout'])

        # RNN
        self.gru = nn.GRU(
            1 * (self.hyper_params['context_size'] + 1), latent_size, 
            batch_first = True, num_layers = 1
        )

        # SASRec
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        for _ in range(hyper_params['num_blocks']):
            if hyper_params['linear_attention']:
                new_attn_layer = MultiHeadAttention(hyper_params)
            else:
                new_attn_layer =  torch.nn.MultiheadAttention(
                    latent_size,
                    hyper_params['num_heads'],
                    hyper_params['dropout'],
                    bias = True
                )
                
            self.attention_layers.append(new_attn_layer)

            new_fwd_layer = PointWiseFeedForward(latent_size, hyper_params['dropout'])
            self.forward_layers.append(new_fwd_layer)

        if not hyper_params['linear_attention']:
            tl = hyper_params['max_seq_len'] # time dim len for enforce causality
            temp = torch.ones((tl, tl), dtype=torch.bool)
            if is_cuda_available: temp = temp.cuda()
            self.attention_mask = ~torch.tril(temp)          # [ seq_len x seq_len ]
            for i in range(tl): self.attention_mask[i, :max(0, i - hyper_params['transformer_seq_len'])] = True

        # Fusion
        self.decode = nn.Sequential(
            nn.Dropout(hyper_params['dropout']),
            nn.Linear(2 * latent_size if hyper_params['gapformer_fusion'] == 'concat' else latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, hyper_params['horizon_K'] if hyper_params['horizon'] else 1)
        )

        if self.hyper_params['gapformer_fusion'] in { 'global_softmax_attention', 'global_raw_attention' }:
            self.weights = nn.Parameter(torch.ones(2) * 0.5)
            self.softmax = nn.Softmax(dim = 0)

        for m in self.decode:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)

    def log2feats(self, log_seqs, timeline_mask):
        seqs = self.first_layer(log_seqs)           # [ bsz x seq_len x latent_size ]
        seqs = self.emb_dropout(seqs)

        # I think this is for handling padding!
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        for i in range(len(self.attention_layers)):
            if self.hyper_params['linear_attention']:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs)
                seqs = mha_outputs + seqs
            else:
                seqs = torch.transpose(seqs, 0, 1)
                mha_outputs, _ = self.attention_layers[i](
                    seqs, seqs, seqs, 
                    attn_mask=self.attention_mask[:seqs.shape[0], :seqs.shape[0]]
                )
                seqs = mha_outputs + seqs               # [ seq_len x bsz x latent_size ]
                seqs = torch.transpose(seqs, 0, 1)      # [ bsz x seq_len x latent_size ]

            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)    # [ bsz x seq_len x latent_size ]

        return seqs

    def one_step(self, input, scale_delta, eval = False):
        last_dist = input[:, :, -1]

        # RNN Part
        rnn_out, _ = self.gru(input)              # [ bsz x seq_len x latent_size ]

        # SASRec Part
        timeline_mask = last_dist == self.hyper_params['pad_with']
        log_feats = self.log2feats(input, timeline_mask)           # [ bsz x seq_len x features ]

        # Fusion part
        if self.hyper_params['gapformer_fusion'] == 'concat':
            middle = torch.cat([ rnn_out, log_feats ], axis = -1)

        elif self.hyper_params['gapformer_fusion'] == 'add':
            middle = (rnn_out + log_feats) / 2.0

        elif self.hyper_params['gapformer_fusion'] == 'global_softmax_attention':
            rnn_weight, sasrec_weight = self.softmax(self.weights)
            middle = (rnn_weight * rnn_out) + (sasrec_weight * log_feats)

        elif self.hyper_params['gapformer_fusion'] == 'global_raw_attention':
            rnn_weight, sasrec_weight = self.weights
            middle = (rnn_weight * rnn_out) + (sasrec_weight * log_feats)

        predicted_delta = self.decode(middle) / scale_delta
        
        if self.hyper_params['horizon']:
            if eval: return predicted_delta[:, :, 0] + last_dist
            else: return predicted_delta + last_dist.unsqueeze(-1)
        else:
            return predicted_delta.squeeze(-1) + last_dist
