INF = float(1e6)

def get_data_loader_class(hyper_params):
    from data_loaders import MLP, RNN

    return {
        # Baselines
        "copy": (MLP.TrainDataset, MLP.TestDataset),
        "linear": (MLP.TrainDataset, MLP.TestDataset),
        
        # Non-sequential
        "MLP": (MLP.TrainDataset, MLP.TestDataset),
        
        # Sequential
        "RNN": (RNN.TrainDataset, RNN.TestDataset),
        "Transformer": (RNN.TrainDataset, RNN.TestDataset),
        "GapFormer": (RNN.TrainDataset, RNN.TestDataset),
    }[hyper_params['model_type']]

def valid_hyper_params(hyper_params):
    is_valid = True

    if hyper_params['model_type'] in [ "Transformer", "GapFormer" ]:
        is_valid = is_valid and (hyper_params['latent_size'] >= hyper_params['num_heads'])
        is_valid = is_valid and (hyper_params['latent_size'] % hyper_params['num_heads'] == 0)

    return is_valid

def get_common_path(hyper_params):
    if not valid_hyper_params(hyper_params): return None

    if hyper_params['model_type'] in { 'copy', 'linear' }: return hyper_params['model_type']

    def get(key): return hyper_params[key]

    common_path = get('model_type') 
    if get('image_feature') == True:
        common_path += '_wimg'
    common_path += {
        "MLP": 			lambda: "_latent_size_{}_dropout_{}".format(
                            get('latent_size'), get('dropout')
                        ),

        "RNN":			lambda: "_latent_size_{}_dropout_{}_max_seq_len_{}".format(
                            get('latent_size'), 
                            get('dropout'), get('max_seq_len')
                        ),

        "Transformer": 	lambda: "_latent_size_{}_dropout_{}_heads_{}_blocks_{}_max_seq_len_{}".format(
                            get('latent_size'), get('dropout'), 
                            get('num_heads'), get('num_blocks'),
                            get('max_seq_len')
                        ),

        "GapFormer": 	lambda: "_latent_size_{}_dropout_{}_heads_{}_blocks_{}_max_seq_len_{}_fusion_{}".format(
                            get('latent_size'), get('dropout'), 
                            get('num_heads'), get('num_blocks'),
                            get('max_seq_len'), get('gapformer_fusion')
                        ),
    }[get('model_type')]()

    if get('model_type') in { 'Transformer', 'GapFormer' }:
        common_path += "_linear_attention_{}".format(get('linear_attention'))
        if get('linear_attention') == True: common_path += "_transformer_len_{}".format(get('transformer_seq_len'))

    if get('horizon') == True: common_path += "_horizon_K_{}_decay_{}".format(get('horizon_K'), get('horizon_decay'))

    if get('model_type') != 'MLP' and get('staggered_training') == True: common_path += "_seq_step_{}".format(get('seq_step_size'))


    common_path += "_bsz_{}_wd_{}_lr_{}".format(
        get('batch_size'), get('weight_decay'), get('lr')
    )
    
    return common_path

def remap_items(data):
    item_map = {}
    for user_data in data:
        for item, rating, time in user_data:
            if item not in item_map: item_map[item] = len(item_map) + 1

    for u in range(len(data)):
        data[u] = list(map(lambda x: [ item_map[x[0]], x[1], x[2] ], data[u]))

    return data

def file_write(log_file, s, dont_print=False):
    if dont_print == False: print(s)
    f = open(log_file, 'a')
    f.write(s+'\n')
    f.close()

def clear_log_file(log_file):
    f = open(log_file, 'w')
    f.write('')
    f.close()

def pretty_print(h):
    print("{")
    for key in h:
        print(' ' * 4 + str(key) + ': ' + h[key])
    print('}\n')

def log_end_epoch(hyper_params, metrics, epoch, time_elpased, metrics_on = '(VAL)', dont_print = False):
    import os, psutil, humanize
    import torch

    # log memory details
    process = psutil.Process(os.getpid())
    string1 = " | CPU_RAM = {}".format(humanize.naturalsize(process.memory_info().rss))
    if torch.cuda.is_available():
        string1 += " | GPU_RAM = {}".format(humanize.naturalsize(torch.cuda.max_memory_reserved()))

    string2 = ""
    for m in metrics: string2 += " | " + m + ' = ' + str(metrics[m])
    string2 += ' ' + metrics_on

    ss  = '-' * 89
    ss += '\n| end of epoch {} | time = {:5.2f}'.format(epoch, time_elpased)
    ss += string1
    ss += string2
    ss += '\n'
    ss += '-' * 89
    file_write(hyper_params['log_file'], ss, dont_print = dont_print)
