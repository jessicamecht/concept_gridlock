import time
import torch
import numpy as np
import torch.nn as nn

import sys ; sys.path.insert(0, "../")
from torch_utils import FloatTensor

class RNN(nn.Module):
    def __init__(self, hyper_params):
        super(RNN, self).__init__()
        self.hyper_params = hyper_params
        latent_size = hyper_params['latent_size']
        
        self.gru = nn.GRU(
            1 * (self.hyper_params['context_size'] + 1), latent_size, 
            batch_first = True, num_layers = 1
        )

        self.decode = nn.Sequential(
            nn.Dropout(hyper_params['dropout']),
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, hyper_params['horizon_K'] if hyper_params['horizon'] else 1)
        )
        for m in self.decode:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)

    def embed(self, features):
        return features

    def one_step(self, input, eval = False):
        last_dist = input[:, :, -1]
        
        rnn_out, _ = self.gru(input)              # [ bsz x seq_len x latent_size ]
        predicted_delta = self.decode(rnn_out) / 2.0       # [ bsz * seq_len x 1]
        
        if self.hyper_params['horizon']:
            if eval: return predicted_delta[:, :, 0] + last_dist
            else: return predicted_delta + last_dist.unsqueeze(-1)
        else:
            return predicted_delta.squeeze(-1) + last_dist

    def forward(self, dist_seq, context_seq, eval = False, scale_delta = 1.0):
        in_shape = dist_seq.shape                           # [ bsz x seq_len ]
        
        context_embed = self.embed(context_seq)             # [ bsz x seq_len x features ]
        curr_distance_embed = dist_seq.unsqueeze(-1)        # [ bsz x seq_len x 1 ]

        if not eval:
            return self.one_step(torch.cat([ context_embed, curr_distance_embed ], axis = -1))

        else:
            eval_K = self.hyper_params['eval_K']
            output = torch.FloatTensor(np.zeros([ in_shape[0], in_shape[1] - eval_K, eval_K ]))
            in_shape = output.shape                             # [ bsz x seq_len - eval_K x input_size ]

            for t in range(self.hyper_params['eval_K']):

                if t == 0: now = curr_distance_embed
                else: now = output[:, :, t-1].unsqueeze(-1)

                # Can't predict eval_K timesteps for last eval_K timesteps
                now = now[:, :in_shape[1], :]

                output[:, :, t] = self.one_step(
                    torch.cat([ context_embed[:, t:, :][:, :in_shape[1], :], now ], axis = -1), 
                    eval = True
                )
            
            return output
