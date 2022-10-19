import time
import torch
import numpy as np
from numba import jit, int64, float32, float64
from sklearn.metrics import roc_auc_score

from utils import INF
from torch_utils import LongTensor, BoolTensor

def evaluate(model, criterion, reader, hyper_params, test = False):
    metrics = {}

    for i in range(hyper_params['eval_K']):
        metrics['RMSE_{}'.format(i + 1)] = 0.0
    divide_by = 0.0

    model.eval() ; is_cpu = False
	# Force RNN evaluation on CPU -- is faster
    if hyper_params['model_type'] == "RNN": is_cpu, model = True, model.cpu()

    eval_time, total = time.time(), 0
    with torch.no_grad():
        for data, y in reader:
            output = model(*data, eval = True)
            
            # Un-normalize output while predicting
            output = reader.un_normalize_dist(output)
            
            if hyper_params['model_type'] in { 'RNN', 'Transformer', 'GapFormer' }: 
                y_mask, y = y

                all_losses = criterion(output,  y, return_mean = False).data
                all_losses *= y_mask.unsqueeze(-1).expand(all_losses.size())
                all_losses = torch.sum(torch.sum(all_losses, dim = 0), dim = 0)
                for k in range(hyper_params['eval_K']):
                    metrics['RMSE_{}'.format(k + 1)] += float(all_losses[k])
                divide_by += float(torch.sum(y_mask))
            else:
                for i in range(hyper_params['eval_K']):
                    metrics['RMSE_{}'.format(i + 1)] += torch.sum(criterion(output[:, i], y[:, i], return_mean = False).data)
                divide_by += float(output.shape[0])
            
    for i in range(hyper_params['eval_K']):
        metrics['RMSE_{}'.format(i + 1)] = float(metrics['RMSE_{}'.format(i + 1)]) / divide_by
        metrics['RMSE_{}'.format(i + 1)] = round(metrics['RMSE_{}'.format(i + 1)] ** 0.5, 4)
    
    metrics['RMSE_mean'] = round(
        np.mean(list([ metrics['RMSE_{}'.format(i + 1)] for i in range(hyper_params['eval_K']) ])), 4
    )

    if is_cpu: model = model.cuda()
    
    return metrics
