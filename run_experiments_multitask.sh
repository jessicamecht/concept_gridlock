python3 main.py -dataset nuscenes -backbone resnet -ground_truth normal -train -gpu_num 3  -max_epochs 200 -task multitask -bs 1
python3 main.py -dataset comma -backbone resnet -ground_truth normal -train -gpu_num 3  -max_epochs 200 -task multitask -bs 1
python3 main.py -dataset comma -backbone clip -ground_truth normal -train -gpu_num 3 -max_epochs 200 -task multitask -bs 1
python3 main.py -dataset nuscenes -backbone clip -ground_truth normal -train -gpu_num 3  -max_epochs 200 -task multitask -bs 1

python3 main.py -dataset nuscenes -backbone vit -ground_truth normal -train -gpu_num 3  -max_epochs 200 -task multitask -bs 1
python3 main.py -dataset nuscenes -backbone none -concept_features -ground_truth normal -train -gpu_num 3  -max_epochs 200 -task multitask -bs 1

python3 main.py -dataset comma -backbone vit -ground_truth normal -train -gpu_num 3  -max_epochs 200 -task multitask -bs 1
python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -gpu_num 3  -max_epochs 200 -task multitask -bs 1
