python3 main.py -dataset nuscenes -backbone vit -ground_truth normal  -gpu_num 3  -task  multitask -bs 1
python3 main.py -dataset nuscenes -backbone vit -ground_truth normal  -gpu_num 3 -task  distance -bs 1
python3 main.py -dataset nuscenes -backbone vit -ground_truth normal  -gpu_num 3  -task  angle -bs 1


python3 main.py -dataset nuscenes -backbone resnet -ground_truth normal  -gpu_num 3  -task  multitask -bs 1
python3 main.py -dataset nuscenes -backbone resnet -ground_truth normal  -gpu_num 3 -task  distance -bs 1
python3 main.py -dataset nuscenes -backbone resnet -ground_truth normal  -gpu_num 3  -task  angle -bs 1


python3 main.py -dataset nuscenes -backbone clip -ground_truth normal  -gpu_num 3  -task  multitask -bs 1
python3 main.py -dataset nuscenes -backbone clip -ground_truth normal  -gpu_num 3 -task  distance -bs 1
python3 main.py -dataset nuscenes -backbone clip -ground_truth normal  -gpu_num 3  -task  angle -bs 1


python3 main.py -dataset nuscenes -backbone none -concept_features -ground_truth normal  -gpu_num 3  -task  multitask -bs 1
python3 main.py -dataset nuscenes -backbone none -concept_features -ground_truth normal  -gpu_num 3 -task  distance -bs 1
python3 main.py -dataset nuscenes -backbone none -concept_features -ground_truth normal  -gpu_num 3  -task  angle -bs 1
