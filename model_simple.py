import torch 
import torch.nn as nn 
from transformers import LongformerModel, LongformerConfig
import torch.nn.functional as F
from timm.models.vision_transformer import vit_base_patch16_224
from torchvision.models import resnet18, ResNet18_Weights

def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)

class LSTMModel(nn.Module):
    def __init__(self,
                 inp_size, 
                 hidden_size, 
                 num_layers=3):

        self.lstm = nn.LSTM(inp_size, hidden_size, num_layers)
    def forward(self, x):
        return self.lstm(x)


def pad_to_window_size_local(input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor,
                             one_sided_window_size: int, pad_token_id: int):
    '''A helper function to pad tokens and mask to work with the sliding_chunks implementation of Longformer self-attention.
    Based on _pad_to_window_size from https://github.com/huggingface/transformers:
    https://github.com/huggingface/transformers/blob/71bdc076dd4ba2f3264283d4bc8617755206dccd/src/transformers/models/longformer/modeling_longformer.py#L1516
    Input:
        input_ids = torch.Tensor(bsz x seqlen): ids of wordpieces
        attention_mask = torch.Tensor(bsz x seqlen): attention mask
        one_sided_window_size = int: window size on one side of each token
        pad_token_id = int: tokenizer.pad_token_id
    Returns
        (input_ids, attention_mask) padded to length divisible by 2 * one_sided_window_size
    '''
    w = 2 * one_sided_window_size
    seqlen = input_ids.size(1)
    padding_len = (w - seqlen % w) % w
    input_ids = F.pad(input_ids.permute(0, 2, 1), (0, padding_len), value=pad_token_id).permute(0, 2, 1)
    attention_mask = F.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
    position_ids = F.pad(position_ids, (1, padding_len), value=False)  # no attention on the padding tokens
    return input_ids, attention_mask, position_ids


class SeqLSTM(nn.Module):
    """
    VTN model builder. It uses ViT-Base as the backbone.
    Daniel Neimark, Omri Bar, Maya Zohar and Dotan Asselmann.
    "Video Transformer Network."
    https://arxiv.org/abs/2102.00719
    """

    def __init__(self,multitask="angle", backbone="resnet"):
        super(SeqLSTM, self).__init__()
        self._construct_network(multitask, backbone)

    def _construct_network(self, multitask, backbone):
        if backbone == "vit":
            self.backbone = vit_base_patch16_224(pretrained=True,num_classes=0,drop_path_rate=0.0,drop_rate=0.0)
            embed_dim = self.backbone.embed_dim
            num_attention_heads=12
            mlp_size = 768+2
        elif backbone== "resnet":
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
            embed_dim = 512+2
            num_attention_heads=2
            mlp_size = 128

        dfs_freeze(self.backbone)
        self.multitask = multitask
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        #self.pe = PositionalEncoding(embed_dim) #TODO add positional encoding

        self.temporal_encoder = LSTMModel(embed_dim, mlp_size)
        num_classes = 1
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mlp_size),
            nn.Linear(mlp_size, mlp_size),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_size, num_classes)
        )
        if self.multitask:
            self.mlp_head_2 = nn.Sequential(
                nn.LayerNorm(mlp_size),
                nn.Linear(mlp_size, mlp_size),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(mlp_size, num_classes)
            )

    def forward(self, x, angle, distance):
        angle = torch.roll(angle, shifts=1, dims=1)
        angle[:,0] = angle[:,1]
        distance = torch.roll(distance, shifts=1, dims=1)
        distance[:,0] = distance[:,1]
        # spatial backbone
        B, F, C, H, W = x.shape
        x = x.reshape(B * F, C, H, W)
        x = self.backbone(x)
        x = x.reshape(B, F, -1)
        x = torch.cat((x, angle.unsqueeze(-1)), dim=-1)
        x = torch.cat((x, distance.unsqueeze(-1)), dim=-1)

        # temporal encoder (Longformer)
        B, D, E = x.shape
        attention_mask = torch.ones((B, D), dtype=torch.long, device=x.device)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        cls_atten = torch.ones(1).expand(B, -1).to(x.device)
        attention_mask = torch.cat((attention_mask, cls_atten), dim=1)
        attention_mask[:, 0] = 2
        x, attention_mask, position_ids = pad_to_window_size_local(
            x,
            attention_mask,
            x,#position_ids, TODO add position_ids
            self.temporal_encoder.config.attention_window[0],
            self.temporal_encoder.config.pad_token_id)
        token_type_ids = torch.zeros(x.size()[:-1], dtype=torch.long, device=x.device)
        token_type_ids[:, 0] = 1

        # TODO add position_ids
        '''position_ids = position_ids.long()
        mask = attention_mask.ne(0).int()
        max_position_embeddings = self.temporal_encoder.config.max_position_embeddings
        position_ids = position_ids % (max_position_embeddings - 2)
        position_ids[:, 0] = max_position_embeddings - 2
        position_ids[mask == 0] = max_position_embeddings - 1'''

        x = self.temporal_encoder(token_type_ids)
        # MLP head
        x = x
        b, s, e = x.shape
        x = x#.reshape(b*s, e)
        if self.multitask:
            x2 = self.mlp_head_2(x)
            #x = x.reshape(b*s, e)
        x = self.mlp_head(x)
        if self.multitask != "multitask":
            return x[:,1:F+1,:]
        else:
            return x[:,1:F+1,:], x2[:,1:F+1,:]