import pytorch_lightning as pl
from model import *
from module import * 
from swin_model import *
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch 
import yaml
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from  pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pathlib import Path
import pandas as pd 
import sys 
sys.path.append('/home/jessica/ADAPT/')
from src.modeling.load_swin import get_swin_model, reload_pretrained_swin
from src.modeling.load_bert import get_bert_model
import os

def save_preds(logits, target, save_name, p):
    b, s = target.shape
    df = pd.DataFrame()
    df['logits'] = logits.squeeze().reshape(b*s).tolist()
    df['target'] = target.squeeze().reshape(b*s).tolist()
    df.to_csv(f'{p}/{save_name}.csv', mode='a', index=False, header=False)

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', default="", type=str)  
    parser.add_argument('-train', action=argparse.BooleanOptionalAction)  
    parser.add_argument('-test', action=argparse.BooleanOptionalAction)
    
    parser.add_argument('-gpu_num', default=0, type=int) 
    parser.add_argument('-train_concepts', action=argparse.BooleanOptionalAction) 
    parser.add_argument('-n_scenarios', default=643, type=int) 
    parser.add_argument('-scenario_type', default="not_specified", type=str) 
    
    parser.add_argument('-dataset_fraction', default=1, type=float) 
    parser.add_argument('-curated', default="human", type=str)  
    parser.add_argument('-dataset', default="comma", type=str)  
    parser.add_argument('-backbone', default="resnet", type=str) 
    parser.add_argument('-dataset_path', default="/data1/jessica/data/toyota/", type=str) 
    parser.add_argument('-concept_features', action=argparse.BooleanOptionalAction) 
    parser.add_argument('-new_version', action=argparse.BooleanOptionalAction) 
    parser.add_argument('-intervention_prediction', action=argparse.BooleanOptionalAction) 
    parser.add_argument('-swin_baseline', action=argparse.BooleanOptionalAction) 
    parser.add_argument('-save_path', default="", type=str) 
    parser.add_argument('-max_epochs', default=1, type=int) 
    parser.add_argument('-bs', default=1, type=int) 
    parser.add_argument('-ground_truth', default="normal", type=str) 
    parser.add_argument('-dev_run', default=False, type=bool) 
    parser.add_argument('-checkpoint_path', default='', type=str)
    return parser

if __name__ == "__main__":
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50"

    if torch.cuda.device_count() > 0 and torch.cuda.get_device_capability()[0] >= 7:
        # Set the float32 matrix multiplication precision to 'high'
        torch.set_float32_matmul_precision('high')

    parser = get_arg_parser()
    args = parser.parse_args()
    multitask = args.task
   
    early_stop_callback = EarlyStopping(monitor="val_loss_accumulated", min_delta=0.05, patience=5, verbose=False, mode="max")
    if args.swin_baseline:
        args1 = {'model_name_or_path': '/data4/jessica/data/toyota/baselines/bert-base-uncased/', 
        'config_name': '', 'tokenizer_name': '', 'num_hidden_layers': -1, 
        'hidden_size': -1, 'num_attention_heads': -1, 'intermediate_size': -1, 
        'img_feature_dim': 512, 'load_partial_weights': False, 'freeze_embedding': False, 
        'drop_out': 0.1, 'max_seq_length': 30, 'max_seq_a_length': 30, 'max_img_seq_length': 196, 
        'do_lower_case': True, 'add_od_labels': False, 'od_label_conf': 0.0, 'use_asr': False, 
        'use_sep_cap': False, 'use_swap_cap': False, 'use_car_sensor': False, 'multitask': False, 
        'only_signal': True, 'signal_types': ['course'], 'unique_labels_on': False, 'no_sort_by_conf': False,
        'mask_prob': 0.5, 'max_masked_tokens': 45, 'attn_mask_type': 'seq2seq', 'text_mask_type': 'random', 
        'tag_to_mask': ['noun', 'verb'], 'mask_tag_prob': -1, 'random_mask_prob': 0, 
        'on_memory': False, 'effective_batch_size': 1, 'per_gpu_train_batch_size': 1, 'num_workers': 10, 'limited_samples': -1, 'learning_rate': 0.0003,
         'weight_decay': 0.05, 'adam_epsilon': 1e-08, 'max_grad_norm': 1.0, 'warmup_ratio': 0.1, 'scheduler': 'warmup_linear', 'gradient_accumulation_steps': 1, 
         'num_train_epochs': 1, 'logging_steps': 20, 'save_steps': 2000, 'restore_ratio': -1, 'device': f"cuda:{args.gpu_num}", 'seed': 88, 'local_rank': 0,
          'mixed_precision_method': 'deepspeed', 'zero_opt_stage': -1, 'amp_opt_level': 0, 'deepspeed_fp16': True, 'fairscale_fp16': False, 'pretrained_checkpoint': '', 
          'debug': False, 'debug_speed': False, 'config': 'src/configs/VidSwinBert/BDDX_multi_default.json', 'eval_model_dir': '',
        'do_train': True, 'do_test': False, 'do_eval': False, 'do_signal_eval': False, 'evaluate_during_training': True, 
          'per_gpu_eval_batch_size': 1, 'mask_img_feat': False, 'max_masked_img_tokens': 10, 'tie_weights': False, 'label_smoothing': 0, 'drop_worst_ratio': 0, 
          'drop_worst_after': 0, 'max_gen_length': 15, 'output_hidden_states': False, 'num_return_sequences': 1, 'num_beams': 1, 'num_keep_best': 1, 'temperature': 1, 
          'top_k': 0, 'top_p': 1, 'repetition_penalty': 1, 'length_penalty': 1, 'use_cbs': False, 'min_constraints_to_satisfy': 2, 'use_hypo': False,
           'decoding_constraint': False, 'remove_bad_endings': False, 'scst': False, 'sc_train_sample_n': 5, 'sc_baseline_type': 'greedy', 'cider_cached_tokens': 
           'coco_caption/gt/coco-train-words.p', 'max_num_frames': 8, 'img_res': 224, 'patch_size': 32, 'grid_feat': True, 'kinetics': '600', 'pretrained_2d': False, 
           'vidswin_size': 'base', 'freeze_backbone': True, 'use_checkpoint': True, 'backbone_coef_lr': 0.05, 'reload_pretrained_swin': False, 'learn_mask_enabled': False, 
           'loss_sparse_w': 0, 'loss_sensor_w': 0, 'sparse_mask_soft2hard': False, 'transfer_method': -1, 'att_mask_expansion': -1, 'resume_checkpoint': 'None', 
           'use_clip_model': True, 'num_gpus': 4, 'distributed': False}

        swin_model = get_swin_model(args1)
        bert_model, config, tokenizer = get_bert_model(args1)

        for name, param in swin_model.named_parameters():
            if not param.requires_grad:
                print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
        model = SignalVideoTransformer(args1, config, swin_model, bert_model)
        model.freeze_backbone(freeze=True)

    else:
        model = VTN(multitask=multitask, backbone=args.backbone, concept_features=args.concept_features, device = f"cuda:{args.gpu_num}", train_concepts=args.train_concepts)

    module = LaneModule(model, multitask=multitask, dataset = args.dataset, bs=args.bs, ground_truth=args.ground_truth, intervention=args.intervention_prediction, dataset_path=args.dataset_path, dataset_fraction=args.dataset_fraction)

    ckpt_pth = f"{args.dataset_path}/ckpts_final/ckpts_final_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}_{args.dataset_fraction}/"
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val_loss_accumulated")
    logger = TensorBoardLogger(save_dir=ckpt_pth)

    path = ckpt_pth + "/lightning_logs/" 
    if not os.path.exists(path):
        os.makedirs(path)
    vs = os.listdir(path)
    filt = []
    f_name, resume_path = 'None', 'None'
    if not args.new_version and not args.test:
        for elem1 in vs: 
            if 'version' in elem1:
                filt.append(elem1)
        versions =[elem.split("_")[-1]for elem in filt]
        versions = sorted(versions)
        version = f"version_{versions[-1]}"
        resume_path = path + version + "/checkpoints/"
        files = os.listdir(resume_path)
        for f in files: 
            if "ckpt" in f:
                f_name = f
                break
            else: 
                f_name = None
    resume = None if args.new_version or args.test and f_name != None else resume_path + f_name
    print(f"RESUME FROM: {resume}")
    trainer = pl.Trainer(
        fast_dev_run=args.dev_run,
        #gpus=2, 
        accelerator='gpu',
        devices=[args.gpu_num] if torch.cuda.is_available() else None, 
        logger=logger,
        resume_from_checkpoint= resume, 
        max_epochs=args.max_epochs,
        default_root_dir=ckpt_pth ,
        callbacks=[TQDMProgressBar(refresh_rate=5), checkpoint_callback],
        #, EarlyStopping(monitor="train_loss", mode="min")],#in case we want early stopping
        )
    save_path = args.save_path
    if args.train:
        trainer.fit(module)
        save_path = "/".join(checkpoint_callback.best_model_path.split("/")[:-1])
        print(f'saving hparams at {save_path}')
        with open(f'{save_path}/hparams.yaml', 'w') as f:
            yaml.dump(args, f)
    ckpt_path=args.checkpoint_path
    p = "/".join(ckpt_path.split("/")[:-2])
    #preds = trainer.test(module, ckpt_path=ckpt_path if ckpt_path != '' else "best")
    preds = trainer.predict(module, ckpt_path=ckpt_path if ckpt_path != '' else "best")
    #save_path =  "."
    for pred in preds:
        if args.task != "multitask":
            predictions, preds_1, preds_2 = pred[0], pred[1], pred[2] 
            save_preds(predictions, preds_1, f"{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}_{args.n_scenarios}_{args.curated}", save_path)
        else:
            preds, angle, dist = pred[0], pred[1], pred[2]
            preds_angle, preds_dist = preds[0], preds[1]
            save_preds(preds_angle, angle, f"angle_multi_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}", save_path)
            save_preds(preds_dist, dist, f"dist_multi_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}", save_path)