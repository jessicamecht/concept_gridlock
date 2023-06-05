#python3 main.py -dataset nuscenes -backbone none -concept_features -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task distance -bs 4
python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task distance -bs 4

#python3 main.py -dataset nuscenes -backbone none -concept_features -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task multitask -bs 4
python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task multitask -bs 4

#python3 main.py -dataset nuscenes -backbone none -concept_features -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task angle -bs 4
python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task angle -bs 4






