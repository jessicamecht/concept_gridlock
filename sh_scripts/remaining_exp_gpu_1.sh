
#python3 main.py -dataset nuscenes -backbone vit -ground_truth normal  -test -gpu_num 0  -task  multitask -bs 1 -checkpoint_path /data1/jessica/data/toyota/ckpts_final/ckpts_final_nuscenes_multitask_vit/lightning_logs/version_9/checkpoints/epoch=231-step=9478.ckpt
#python3 main.py -dataset nuscenes -backbone vit -ground_truth normal -test  -gpu_num 0 -task  distance -bs 1 -checkpoint_path /data1/jessica/data/toyota/ckpts_final/ckpts_final_nuscenes_distance_vit/lightning_logs/version_1/checkpoints/epoch=559-step=9795.ckpt
#python3 main.py -dataset nuscenes -backbone vit -ground_truth normal -test  -gpu_num 0  -task  angle -bs 1 -checkpoint_path /data1/jessica/data/toyota/ckpts_final/ckpts_final_nuscenes_angle_vit/lightning_logs/version_8/checkpoints/epoch=351-step=24893.ckpt


python3 main.py -dataset nuscenes -backbone resnet -ground_truth normal -test  -gpu_num 0  -task  multitask -bs 1 -checkpoint_path /data1/jessica/data/toyota/ckpts_final/ckpts_final_nuscenes_multitask_resnet/lightning_logs/version_6/checkpoints/epoch=535-step=7332.ckpt
python3 main.py -dataset nuscenes -backbone resnet -ground_truth normal -test  -gpu_num 0 -task  distance -bs 1 -checkpoint_path /data1/jessica/data/toyota/ckpts_final/ckpts_final_nuscenes_distance_resnet/lightning_logs/version_4/checkpoints/epoch=581-step=8562.ckpt
python3 main.py -dataset nuscenes -backbone resnet -ground_truth normal -test  -gpu_num 0  -task  angle -bs 1 -checkpoint_path /data1/jessica/data/toyota/ckpts_final/ckpts_final_nuscenes_angle_resnet/lightning_logs/version_7/checkpoints/epoch=434-step=3786.ckpt


python3 main.py -dataset nuscenes -backbone clip -ground_truth normal -test  -gpu_num 0  -task  multitask -bs 1 -checkpoint_path /data1/jessica/data/toyota/ckpts_final/ckpts_final_nuscenes_multitask_clip/lightning_logs/version_7/checkpoints/epoch=576-step=9048.ckpt
python3 main.py -dataset nuscenes -backbone clip -ground_truth normal -test  -gpu_num 0 -task  distance -bs 1 -checkpoint_path /data1/jessica/data/toyota/ckpts_final/ckpts_final_nuscenes_distance_clip/lightning_logs/version_4/checkpoints/epoch=421-step=6009.ckpt
python3 main.py -dataset nuscenes -backbone clip -ground_truth normal -test  -gpu_num 0  -task  angle -bs 1 -checkpoint_path /data1/jessica/data/toyota/ckpts_final/ckpts_final_nuscenes_angle_clip/lightning_logs/version_9/checkpoints/epoch=398-step=27930.ckpt


#python3 main.py -dataset nuscenes -backbone none -concept_features -test -ground_truth normal  -gpu_num 0  -task  multitask -bs 1 -checkpoint_path /data1/jessica/data/toyota/ckpts_final/ckpts_final_nuscenes_multitask_none/lightning_logs/version_5/checkpoints/epoch=585-step=8830.ckpt
#python3 main.py -dataset nuscenes -backbone none -concept_features -test -ground_truth normal  -gpu_num 0 -task  distance -bs 1 -checkpoint_path /data1/jessica/data/toyota/ckpts_final/ckpts_final_nuscenes_distance_none/lightning_logs/version_15/checkpoints/epoch=425-step=29820.ckpt
#python3 main.py -dataset nuscenes -backbone none -concept_features -test -ground_truth normal  -gpu_num 0  -task  angle -bs 1 -checkpoint_path /data1/jessica/data/toyota/ckpts_final/ckpts_final_nuscenes_angle_none/lightning_logs/version_8/checkpoints/epoch=592-step=9004.ckpt

