python3 main.py -dataset nuscenes -backbone vit -ground_truth normal -train -gpu_num 1  -max_epochs 400 -task  multitask -bs 12
python3 main.py -dataset nuscenes -backbone vit -ground_truth normal -train -gpu_num 1 -max_epochs 400 -task  distance -bs 12
python3 main.py -dataset nuscenes -backbone vit -ground_truth normal -train -gpu_num 1  -max_epochs 400 -task  angle -bs 12