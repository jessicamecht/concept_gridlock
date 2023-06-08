python3 main.py -dataset nuscenes -backbone vit -ground_truth normal -train -gpu_num 3  -max_epochs 400 -task  multitask -bs 1
python3 main.py -dataset nuscenes -backbone vit -ground_truth normal -train -gpu_num 3 -max_epochs 400 -task  distance -bs 1
python3 main.py -dataset nuscenes -backbone vit -ground_truth normal -train -gpu_num 3  -max_epochs 400 -task  angle -bs 1
