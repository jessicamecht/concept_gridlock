import torch
import torch.nn.functional as F
import numpy as np

from torch_utils import FloatTensor

class CustomLoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super(CustomLoss, self).__init__()
        
        self.forward = self.mse
        self.hyper_params = hyper_params

        self.temp = None
        if hyper_params['horizon'] == True:
            if hyper_params['horizon_decay'] == 'linear':
                self.temp = FloatTensor(np.arange(1.0, 0.0, -1.0 / float(hyper_params['horizon_K']))).unsqueeze(0).unsqueeze(0)
            
            elif hyper_params['horizon_decay'] == 'inverse':
                self.temp = FloatTensor([ 1.0 / float(i+1) for i in range(hyper_params['horizon_K']) ]).unsqueeze(0).unsqueeze(0)

            else:
                self.temp = None

    def mse(self, output, y, return_mean = True):
        mse = torch.pow(output - y, 2)

        if self.temp is not None and len(output.shape) == 3 and output.shape[-1] == self.hyper_params['horizon_K']: 
            mse *= self.temp
            mse = torch.sum(mse, axis = -1)

        if return_mean: return torch.mean(mse)
        return mse
