import torch
import numpy as np
import torch.nn as nn

import sys ; sys.path.insert(0, "../")
from torch_utils import FloatTensor

class MLP(nn.Module):
    def __init__(self, hyper_params):
        super(MLP, self).__init__()
        self.hyper_params = hyper_params
        latent_size = hyper_params['latent_size']

		# Baselines
        if hyper_params['model_type'] in { "copy", "linear" }: return

        self.decode = nn.Sequential(
            nn.Dropout(hyper_params['dropout']),
            nn.Linear(2 * (self.hyper_params['context_size'] + 1), latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, hyper_params['horizon_K'] if hyper_params['horizon'] else 1)
        )
        for m in self.decode:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)

    def embed(self, features):
        return features

    def one_step(self, input, eval = False):
        last_dist = input[:, -1]

        predicted_delta = self.decode(input) / 2.0

        if self.hyper_params['horizon']:
            if eval: return predicted_delta[:, 0] + last_dist
            else: return predicted_delta + last_dist.unsqueeze(-1)
        else:
            return predicted_delta.squeeze(-1) + last_dist

    def forward(self, user, curr_distance, prev_distance, context, prev_context, eval = False, scale_delta = 1.0):
        if self.hyper_params['model_type'] == "copy": 
            if eval: return curr_distance.unsqueeze(-1).repeat(1, self.hyper_params['eval_K'])
            else: return curr_distance
        
        if self.hyper_params['model_type'] == 'linear':
            delta = (curr_distance - prev_distance)
            if eval: 
                final = curr_distance.unsqueeze(-1).repeat(1, self.hyper_params['eval_K'])
                for i in range(self.hyper_params['eval_K']): final[:, i] += delta * (i + 1)
                return final
            else: return curr_distance + delta

        prev_context_embed = self.embed(prev_context)        
        context_embed = self.embed(context)

        if not eval:
            return self.one_step(torch.cat([
                prev_context_embed,
                prev_distance.unsqueeze(-1),
                context_embed,
                curr_distance.unsqueeze(-1)
            ], axis = -1))
        
        else:
            output = FloatTensor(np.zeros([ prev_distance.shape[0], self.hyper_params['eval_K'] ]))
            for t in range(self.hyper_params['eval_K']):
                if t == 0: prev, now = prev_distance, curr_distance
                elif t == 1: prev, now = curr_distance, output[:, 0]
                else: prev, now = output[:, t-2], output[:, t-1]

                output[:, t] = self.one_step(torch.cat([
                    prev_context_embed[:, t],
                    prev.unsqueeze(-1),
                    context_embed[:, t],
                    now.unsqueeze(-1)
                ], axis = -1), eval = True)

            return output
