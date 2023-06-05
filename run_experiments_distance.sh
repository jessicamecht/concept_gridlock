python3 main.py -dataset nuscenes -backbone resnet -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task distance -bs 4
python3 main.py -dataset nuscenes -backbone vit -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task distance -bs 4
python3 main.py -dataset nuscenes -backbone none -concept_features -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task distance -bs 4
python3 main.py -dataset nuscenes -backbone clip -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task distance -bs 4

python3 main.py -dataset comma -backbone resnet -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task distance -bs 4
python3 main.py -dataset comma -backbone vit -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task distance -bs 4
python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -gpu_num 0  -max_epochs 200 -task distance -bs 4
python3 main.py -dataset comma -backbone clip -ground_truth normal -train -gpu_num 0 -max_epochs 200 -task distance -bs 4