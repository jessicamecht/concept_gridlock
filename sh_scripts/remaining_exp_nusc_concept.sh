python3 main.py -dataset nuscenes -backbone none -concept_features -train_concepts -ground_truth normal -train -gpu_num 1  -max_epochs 200 -task  angle -bs 1 -new_version
python3 main.py -dataset nuscenes -backbone none -concept_features -train_concepts -ground_truth normal -train -gpu_num 1  -max_epochs 200 -task  multitask -bs 1 -new_version
python3 main.py -dataset nuscenes -backbone none -concept_features -train_concepts -ground_truth normal -train -gpu_num 3  -max_epochs 200 -task  distance -bs 1 -new_version

python3 main.py -dataset comma -backbone none -concept_features -train_concepts -ground_truth normal -train -gpu_num 1  -max_epochs 600 -task  angle -bs 1 -new_version
python3 main.py -dataset comma -backbone none -concept_features -train_concepts -ground_truth normal -train -gpu_num 1  -max_epochs 600 -task  multitask -bs 1 -new_version
python3 main.py -dataset comma -backbone none -concept_features -train_concepts -ground_truth normal -train -gpu_num 1  -max_epochs 600 -task  distance -bs 1 
