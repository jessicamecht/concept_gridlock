import gc
import os
import copy
import time
import datetime
import traceback
from tqdm import tqdm
import multiprocessing
from collections import defaultdict

from main import main
from utils import get_common_path

common_hyper_params = {
    ## Some model hyper-params
    'weight_decay':         list(map(float, [ 1e-6 ])),
	'horizon': 				[ True ],
	'staggered_training': 	[ False ],
	'seq_step_size': 		[ 100 ], # NOTE: will not be used if staggered_training == False
    'image_feature':        [ True],
    ## Transformer / GapFormer
    'num_heads':            [ 1],
    'num_blocks':           [ 1],
	'linear_attention': 	[ True ],

    ## Future search
    'horizon_K':              [ 5],
    'horizon_decay':         [
        None,
        # 'linear',
        # 'inverse',
    ],

    ## General stuff
	'eval_K':               100,
    'pad_with':             0.0,
    'validate_every':       3,
    'early_stop':           20, # Stop if perf. doesn't increase for `early_stop` X `validate_every` epochs
}


once_hyper_params = {
    'dataset': 'once',

    'latent_size':          [256],
    'epochs':               50,
    
    'max_seq_len':          [ 500],
    'transformer_seq_len':  [ 50],
}

baselines = {
    'model_type':           [ 'copy', 'linear' ],
    'batch_size': 64,
}

non_sequential_models = {
    'model_type':           [ 'MLP' ],
    'batch_size':           [ 64 ],
    'lr':                   [ 0.001 ],
    'dropout':              [ 0.0]
}

rnn_models = {
    'model_type':           [ 'RNN' ],
    'batch_size':           [ 2 ], 
	'lr':                   [ 0.0005 ],
	'dropout':              [ 0.0],
}

transformer_models = {
    'model_type':           [ 'Transformer' ],
    'batch_size':           [ 2 ], 
	'lr':                   [ 0.0005 ],
	'dropout':              [ 0.0],
}

gapformer_models = {
    'model_type':           [ 'GapFormer' ],
    'batch_size':           [ 2 ], 
	'lr':                   [ 0.0005 ],
	'dropout':              [ 0.0],
    'gapformer_fusion':     [ 
        'concat', 
        'add', 
        'global_softmax_attention', 
        'global_raw_attention' 
    ],
}

final_search = [
    [ common_hyper_params ],
    
	# Datasets to train on
	[ 
        # toyota_hyper_params, 
        # openacc_hyper_params,
		once_hyper_params
    ],

	# Methods to train
    [ 
        baselines,
        non_sequential_models, 
        # rnn_models, 
        transformer_models,
        gapformer_models,
    ]
]

# NOTE: This vector dictates how many processes to run on each GPU/CPU in parallel
'''
For e.g. for a machine with 2 GPUs, and if gpu_ids = [ -1, -1, 0, 0, 1, 1 ]
6 configurations will be trained in parallel:
- 2 on the CPU (indicated by -1)
- 2 on the 1st GPU (GPU_ID = 0) (indicated by 0)
- 2 on the 2nd GPU (GPU_ID = 1) (indicated by 1)
Note that if provided GPU_ID > #GPUs available; the configuration will be trained on the CPU
'''
gpu_ids = [ 1, 2, 2, 3 ]

################## CONFIGURATION INPUT ENDS ###################

# STEP-1: Count processes 
def get_all_jobs_recursive(task):
    ret, single_proc = [], True

    for key in task:
        if type(task[key]) != list: continue

        single_proc = False
        for val in task[key]:
            send = copy.deepcopy(task) ; send[key] = val
            ret += get_all_jobs_recursive(send)

        break # All sub-jobs are already counted

    return ret if not single_proc else [ task ]

def get_all_jobs(already, final_search):
    if len(final_search) == 0: return get_all_jobs_recursive(already)

    ret = []
    for at, i in enumerate(final_search):

        for j in i:
            send = copy.deepcopy(already) ; send.update(j)
            ret += get_all_jobs(send, final_search[at + 1:])

        break # All sub-jobs are already counted
    return ret

duplicate_tasks = get_all_jobs({}, final_search)
print("Total processes before unique:", len(duplicate_tasks))

all_tasks, temp = [], set()
for task in tqdm(duplicate_tasks):
    log_file = get_common_path(task)

    if log_file is None: continue
    if task['dataset'] + log_file in temp: continue

    temp.add(task['dataset'] + log_file)

    ##### TEMP: Checking if job has already been done
    final_path = "../results/{}/logs/{}.txt".format(task['dataset'], log_file)
    if os.path.exists(final_path):
        f = open(final_path, 'r')
        lines = f.readlines() ; f.close()
        exists = sum(map(lambda x: int('TEST' in x.strip()), lines))
        if exists != 0: continue

    all_tasks.append(task)
print("Total processes after unique:", len(temp))
print("Total processes after removing already finished jobs:", len(all_tasks))
temp = defaultdict(int)
for t in all_tasks: temp[t['model_type']] += 1
print(dict(temp))
# exit()

# STEP-2: Assign individual GPU processes
gpu_jobs = [ [] for _ in range(len(gpu_ids)) ]
for i, task in enumerate(all_tasks): gpu_jobs[i % len(gpu_ids)].append(task)

# Step-3: Spawn jobs parallely
def file_write(log_file, s):
    f = open(log_file, 'a')
    f.write(s+'\n')
    f.close()

def run_tasks(tasks, gpu_id):
    start_time = time.time()
    for num, task in enumerate(tasks):
        try: 
            main(task, gpu_id = gpu_id)
            percent_done = max(0.00001, float(num + 1) / float(len(tasks)))
            time_elapsed = time.time() - start_time
            file_write(
                "../results/run_log.txt", 
                str(task) + "\nGPU_ID = " + str(gpu_id) + "; [{} / {}] ".format(num + 1, len(tasks)) +
                str(round(100.0 * percent_done, 2)) + "% done; " +
                "ETA = " + str(datetime.timedelta(seconds=int((time_elapsed / percent_done) - time_elapsed)))
            )
        except Exception as e:
            file_write(
                "../results/run_log.txt", "GPU_ID = " + str(gpu_id) + \
                "; ERROR [" + str(num) + "/" + str(len(tasks)) + "]\nJOB: " + str(task) + "\n" + str(traceback.format_exc())
            )
        gc.collect()

for gpu in range(len(gpu_ids)):
    p = multiprocessing.Process(target=run_tasks, args=(gpu_jobs[gpu], gpu_ids[gpu], ))
    p.start()
