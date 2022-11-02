import torch
import numpy as np
import torch.nn as nn

from torch_utils import LongTensor, FloatTensor, BoolTensor, is_cuda_available
from local_attention import LocalAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, hyper_params):
        super(MultiHeadAttention, self).__init__()

        self.attention_mask = torch.ones((1, hyper_params['max_seq_len']), dtype=torch.bool)
        if is_cuda_available: self.attention_mask = self.attention_mask.cuda()

        self.num_heads = hyper_params['num_heads']

        # We assume d_v always equals d_k
        self.d_k = hyper_params['latent_size'] // hyper_params['num_heads']
        self.h = hyper_params['num_heads']

        self.output_linear = nn.Linear(hyper_params['latent_size'], hyper_params['latent_size'])
        
        self.attention = LocalAttention(
            dim =self.d_k,           # dimension of each head (you need to pass this in for relative positional encoding)
            window_size = hyper_params['transformer_seq_len'],       # window size. 512 is optimal, but 256 or 128 yields good enough results
            causal = True,           # auto-regressive or not
            # shared_qk = True,        # For shared queries and keys -- NOT SURE IF THIS HELPS
            autopad = True,          # For varying sequence lengths
            look_backward = 1,       # each window looks at the window before
            look_forward = 0,        # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
            dropout = hyper_params['dropout'],           # post-attention dropout
            exact_windowsize = False # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
        )

    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k) #.transpose(1, 2)
        #                      for l, x in zip(self.linear_layers, (query, key, value))]
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k) #.transpose(1, 2)
                             for x in [query, key, value] ]

        # 2) Apply attention on all the projected vectors in batch.
        x = torch.cat([
            self.attention(
                query[:, :, i, :], key[:, :, i, :], value[:, :, i, :], 
                input_mask = self.attention_mask[:, :query.shape[1]]
            ).unsqueeze(2)
            for i in range(self.num_heads)
        ], dim = 2)

        # 3) "Concat" using a view and apply a final linear.
        x = x.contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), None

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class Transformer(torch.nn.Module):
    def __init__(self, hyper_params):
        super(Transformer, self).__init__()
        self.hyper_params = hyper_params

        # My additions
        self.first_layer = nn.Linear(1 * (self.hyper_params['context_size'] + 1), hyper_params['latent_size'], bias = True)
        
        self.decode = nn.Linear(
            hyper_params['latent_size'], 
            hyper_params['horizon_K'] if hyper_params['horizon'] else 1, 
            bias = True
        )

        for m in [ self.first_layer, self.decode ]:
            nn.init.xavier_uniform_(m.weight)

        # self.pos_emb = torch.nn.Embedding(hyper_params['max_seq_len'], hyper_params['latent_size']) # TO IMPROVE
        # self.pos_emb.weight.data.uniform_(-1, 1)

        self.emb_dropout = torch.nn.Dropout(p=hyper_params['dropout'])

        # self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        # self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        # self.last_layernorm = torch.nn.LayerNorm(hyper_params['latent_size'], eps=1e-8)

        # self.LINEAR_ATTENTION = False # Use linear attention vs. quadratic attention

        for _ in range(hyper_params['num_blocks']):
            # new_attn_layernorm = torch.nn.LayerNorm(hyper_params['latent_size'], eps=1e-8)
            # self.attention_layernorms.append(new_attn_layernorm)

            if self.hyper_params['linear_attention']:
                new_attn_layer = MultiHeadAttention(hyper_params)
            else:
                new_attn_layer =  torch.nn.MultiheadAttention(
                    hyper_params['latent_size'],
                    hyper_params['num_heads'],
                    hyper_params['dropout'],
                    bias = True
                )
                
            self.attention_layers.append(new_attn_layer)

            # new_fwd_layernorm = torch.nn.LayerNorm(hyper_params['latent_size'], eps=1e-8)
            # self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hyper_params['latent_size'], hyper_params['dropout'])
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs, timeline_mask):
        # seqs = self.item_emb(log_seqs)
        seqs = self.first_layer(log_seqs)
        # seqs *= self.hyper_params['latent_size'] ** 0.5
        # positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        # seqs += self.pos_emb(LongTensor(positions))
        seqs = self.emb_dropout(seqs)

        # I think this is for handling padding!
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        temp = torch.ones((tl, tl), dtype=torch.bool)
        if is_cuda_available: temp = temp.cuda()
        attention_mask = ~torch.tril(temp)

        for i in range(len(self.attention_layers)):
            if self.hyper_params['linear_attention']:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs)
                seqs = mha_outputs + seqs
            else:
                seqs = torch.transpose(seqs, 0, 1)
                mha_outputs, _ = self.attention_layers[i](
                    seqs, seqs, seqs, 
                    attn_mask=attention_mask
                )
                seqs = mha_outputs + seqs               # [ seq_len x bsz x latent_size ]
                seqs = torch.transpose(seqs, 0, 1)      # [ bsz x seq_len x latent_size ]

            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = seqs

        return log_feats

    def one_step(self, input, scale_delta, eval = False):
        last_dist = input[:, :, -1]
        timeline_mask = last_dist == self.hyper_params['pad_with']
        
        # Embed sequence
        log_feats = self.log2feats(input, timeline_mask)           # [ bsz x seq_len x features ]
        # print ('log_feats', log_feats.shape)
        predicted_delta = self.decode(log_feats) / scale_delta
        
        if self.hyper_params['horizon']:
            if eval: return predicted_delta[:, :, 0] + last_dist
            else: return predicted_delta + last_dist.unsqueeze(-1)
        else:
            return predicted_delta.squeeze(-1) + last_dist

    def forward(self, dist_seq, context_seq, eval = False, scale_delta = 1.0):
        in_shape = dist_seq.shape                           # [ bsz x seq_len ]
        
        context_embed = context_seq                         # [ bsz x seq_len x features ]
        curr_distance_embed = dist_seq.unsqueeze(-1)        # [ bsz x seq_len x 1 ]
        # print (context_embed.shape)
        # print (curr_distance_embed.shape)
        # print ("eval", eval)
        if not eval:
            return self.one_step(
                torch.cat([ context_embed, curr_distance_embed ], axis = -1),
                scale_delta
            )

        else:
            eval_K = self.hyper_params['eval_K']
            output = FloatTensor(np.zeros([ in_shape[0], in_shape[1] - eval_K, eval_K ]))
            in_shape = output.shape                             # [ bsz x seq_len - eval_K x input_size ]

            for t in range(self.hyper_params['eval_K']):

                if t == 0: now = curr_distance_embed
                else: now = output[:, :, t-1].unsqueeze(-1)

                # Can't predict eval_K timesteps for last eval_K timesteps
                now = now[:, :in_shape[1], :]

                output[:, :, t] = self.one_step(
                    torch.cat([ context_embed[:, t:, :][:, :in_shape[1], :], now ], axis = -1), 
                    scale_delta,
                    eval = True
                )
            
            return output
