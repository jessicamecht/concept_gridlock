python3 main.py -dataset comma -backbone clip -ground_truth normal -train -gpu_num 1  -max_epochs 200 -task  angle -bs 2  -dataset_path /data1/shared/jessica/data1/data/toyota/
python3 main.py -dataset comma -backbone clip -ground_truth normal -train -gpu_num 1  -max_epochs 200 -task  distance -bs 2  -dataset_path /data1/shared/jessica/data1/data/toyota/
python3 main.py -dataset nuscenes -backbone clip -ground_truth normal -train -gpu_num 1  -max_epochs 200 -task  multitask -bs 32  -dataset_path /data1/shared/jessica/data1/data/toyota/

python3 main.py -dataset nuscenes -backbone resnet -ground_truth normal -train -gpu_num 1  -max_epochs 200 -task  multitask -bs 1  -dataset_path /data1/shared/jessica/data1/data/toyota/
