
python3 main.py -dataset comma -backbone resnet -concept_features -ground_truth normal -train -gpu_num 1  -max_epochs 200 -task  angle -bs 1 -new_version
python3 main.py -dataset comma -backbone resnet -concept_features -ground_truth normal -train -gpu_num 1  -max_epochs 200 -task  multitask -bs 1 -new_version
python3 main.py -dataset comma -backbone resnet -concept_features -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task  distance -bs 1 -new_version
