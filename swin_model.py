import torch
from fairscale.nn.misc import checkpoint_wrapper
import random
import torch
from torch import nn
import sys 
sys.path.append('/home/jessica/ADAPT/')
from src.layers.bert.modeling_bert import BertEncoder
from src.layers.bert import BertConfig, BertEncoder

def get_sensor_pred_model(args):
    return Sensor_Pred_Head(args)


class Sensor_Pred_Head(torch.nn.Module):
    """ This is the Control Signal Prediction head that performs sensor regression """
    def __init__(self, args):
        """ Initializes the prediction head.
        A simple transformer that performs sensor regression. 
        We simply use a transformer to regress the whole signals of a video, which is superficial and could be optimized to a large extent.
        """
        super(Sensor_Pred_Head, self).__init__()

        self.img_feature_dim = int(args['img_feature_dim'])
        self.use_grid_feat = args['grid_feat']

        # Motion Transformer implemented by bert
        self.config = BertConfig.from_pretrained(args['config_name'] if args['config_name'] else \
            args['model_name_or_path'], num_labels=2, finetuning_task='image_captioning')
        self.encoder = BertEncoder(self.config)

        # type number of control signals to be used
        # TODO: Set this variable as an argument, corresponging to the control signal in dataloader
        
        self.sensor_dim = len(args['signal_types'])
        self.sensor_embedding = torch.nn.Linear(self.sensor_dim, self.config.hidden_size)
        self.sensor_dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # a mlp to transform the dimension of video feature 
        self.img_dim = self.img_feature_dim
        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.img_dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # a sample regression decoder
        self.decoder = nn.Linear(self.config.hidden_size, self.sensor_dim)


    def forward(self, *args, **kwargs):
        """The forward process.
        Parameters:
            img_feats: video features extracted by video swin
            car_info: ground truth of control signals
        """
        vid_feats = kwargs['img_feats']
        car_info  = kwargs['dist']
        angle  = kwargs['angle']
        vego  = kwargs['vego']

        '''angle = torch.roll(angle, shifts=1, dims=1)
        angle[:,0] = angle[:,1]
        distance = torch.roll(distance, shifts=1, dims=1)
        distance[:,0] = distance[:,1]
        vego = torch.roll(vego, shifts=1, dims=1)
        vego[:,0] = vego[:,1]'''

        car_info = car_info.unsqueeze(-1)
        #car_info = car_info.permute(0, 2, 1)
       

        B, S, C = car_info.shape
        #assert C == self.sensor_dim, f"{C}, {self.sensor_dim}"
        frame_num = S

        img_embedding_output = self.img_embedding(vid_feats)
        
        img_embedding_output = self.img_dropout(img_embedding_output)


        extended_attention_mask = self.get_attn_mask(img_embedding_output)

        encoder_outputs = self.encoder(img_embedding_output,
                                        extended_attention_mask)
        sequence_output = encoder_outputs[0][:, :frame_num, :]

        pred_tensor = self.decoder(sequence_output)


        loss = self.get_l2_loss(pred_tensor, car_info)

        return pred_tensor, loss#loss, pred_tensor

    def get_attn_mask(self, img_embedding_output):
        """Get attention mask that should be passed to motion transformer."""
        device = img_embedding_output.device
        bsz = img_embedding_output.shape[0]
        img_len = img_embedding_output.shape[1]


        attention_mask = torch.ones((bsz, img_len, img_len), dtype=torch.long)


        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask.to(device)

    def get_l2_loss(self, pred, targ):
        loss_func = nn.MSELoss()
        return loss_func(pred, targ)


class SignalVideoTransformer(torch.nn.Module):
    """ This is the one head module that performs Control Signal Prediction. """
    def __init__(self, args, config, swin, transformer_encoder):
        """ Initializes the model.
        Parameters:
            args: basic args of ADAPT, mostly defined in `src/configs/VidSwinBert/BDDX_multi_default.json` and input args
            config: config of transformer_encoder, mostly defined in `models/captioning/bert-base-uncased/config.json`
            swin: torch module of the backbone to be used. See `src/modeling/load_swin.py`
            transformer_encoder: torch module of the transformer architecture. See `src/modeling/load_bert.py`
        """
        super(SignalVideoTransformer, self).__init__()
        self.config = config
        self.use_checkpoint = args['use_checkpoint'] and not args['freeze_backbone']
        if self.use_checkpoint:
            self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        else:
            self.swin = swin
        self.trans_encoder = transformer_encoder
        self.img_feature_dim = int(args['img_feature_dim'])
        self.use_grid_feat = args['grid_feat']
        self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)
        self.compute_mask_on_the_fly = False # deprecated
        self.mask_prob = args['mask_prob']
        self.mask_token_id = -1
        self.max_img_seq_length = args['max_img_seq_length']

        # get Control Signal Prediction Head
        self.sensor_pred_head = get_sensor_pred_model(args)

        # if only_signal is True, it means we 
        # remove Driving Caption Generation head and only use Control Signal Prediction head 
        self.only_signal = True
        self.args = args
        assert self.only_signal


    def forward(self, img, angle, distance, vego):
        """ The forward process of Control Signal Prediction Head, 
        Parameters:
            input_ids: word tokens of input sentences tokenized by tokenizer
            attention_mask: multimodal attention mask in Vision-Language transformer
            token_type_ids: typen tokens of input sentences, 
                            0 means it is a narration sentence and 1 means a reasoning sentence, same size with input_ids
            img_feats: preprocessed frames of the video
            car_info: control signals of ego car in the video
        """
        # video swin to extract video features
        #images = kwargs['img_feats']
        
        images = img
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        
        images = images.permute(0, 2, 1, 3, 4)
        vid_feats = self.swin(images)

        # tokenize video features to video tokens
        if self.use_grid_feat==True:
            vid_feats = vid_feats.permute(0, 2, 3, 4, 1)
        vid_feats = vid_feats.view(B, -1, self.latent_feat_size)

        # use an mlp to transform video token dimension
        vid_feats = self.fc(vid_feats)

        # prepare VL transformer inputs
        kwargs = {
            'img_feats': vid_feats, 
            'dist': distance,
            'angle': angle,
            'vego': vego,
        }

        # only Control Signal Prediction head 
        sensor_outputs = self.sensor_pred_head(*self.args, **kwargs)        
        return sensor_outputs

    
    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(video_attention)))
        return sparsity_loss

    def reload_attn_mask(self, pretrain_attn_mask): 
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        scale_factor = 1
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens*i:pretrained_num_tokens*(i+1), 
                            pretrained_num_tokens*i:pretrained_num_tokens*(i+1)] = pretrained_learn_att 

    def freeze_backbone(self, freeze=True):
        for _, p in self.swin.named_parameters():
            p.requires_grad =  not freeze
