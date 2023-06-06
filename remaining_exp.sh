
python3 main.py -dataset nuscenes -backbone vit -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task  multitask -bs 1  -dataset_path /data1/shared/jessica/data1/data/toyota/
python3 main.py -dataset comma -backbone vit -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task  distance -bs 1  -dataset_path /data1/shared/jessica/data1/data/toyota/
python3 main.py -dataset comma -backbone vit -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task  multitask -bs 1  -dataset_path /data1/shared/jessica/data1/data/toyota/

python3 main.py -dataset nuscenes -backbone none -concept_features -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task  multitask -bs 1  -dataset_path /data1/shared/jessica/data1/data/toyota/
