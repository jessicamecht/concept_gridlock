python3 main.py -dataset comma -backbone clip -ground_truth normal -train -gpu_num 0 -max_epochs 200 -task multitask -bs 1
python3 main.py -dataset nuscenes -backbone clip -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task multitask -bs 1
ython3 main.py -dataset nuscenes -backbone clip -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task angle -bs 1
python3 main.py -dataset comma -backbone clip -ground_truth normal -train -gpu_num 0 -max_epochs 200 -task angle -bs 1
python3 main.py -dataset comma -backbone clip -ground_truth normal -train -gpu_num 0 -max_epochs 200 -task distance -bs 1
python3 main.py -dataset nuscenes -backbone clip -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task distance -bs 1