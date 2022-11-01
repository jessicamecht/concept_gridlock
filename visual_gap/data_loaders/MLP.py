import torch
import numpy as np

from data_loaders.base import BaseTrainDataset, BaseTestDataset
from torch_utils import is_cuda_available

class TrainDataset(BaseTrainDataset):
    def __init__(self, data, hyper_params):
        super(TrainDataset, self).__init__(data, hyper_params)

        self.users_cpu, self.items, self.previous_items = [], [], []
        self.context, self.previous_context = [], []
        self.y = []

        for u in data:
            user_hist = data[u]
            if hyper_params['horizon']: user_hist = data[u][:-hyper_params['horizon_K']]

            for at, (i, policy_dep_context, policy_indep_context) in enumerate(user_hist):
                # No previous context or something to predict
                if at == 0 or at == len(data[u]) - 1: continue

                self.users_cpu.append(u)
                self.items.append(i)
                self.previous_items.append(data[u][at-1][0])
                self.context.append(policy_indep_context)
                self.previous_context.append(data[u][at-1][2])
                
                if hyper_params['horizon']:
                    self.y.append(list(map(lambda x: x[0], data[u][at+1:][:hyper_params['horizon_K']])))
                else:
                    self.y.append(data[u][at+1][0])

        # Copying ENTIRE dataset to the GPU
        self.users = torch.LongTensor(self.users_cpu)
        self.context = torch.FloatTensor(self.context)
        self.previous_context = torch.FloatTensor(self.previous_context)
        
        self.items = self.normalize_dist(torch.FloatTensor(self.items))
        self.previous_items = self.normalize_dist(torch.FloatTensor(self.previous_items))
        self.y = self.normalize_dist(torch.FloatTensor(self.y))

        self.device = "cuda:0" if is_cuda_available else "cpu"

        self.num_interactions = len(self.users_cpu)

    def shuffle_data(self):
        indices = np.arange(self.num_interactions)
        np.random.shuffle(indices)

        return list(map(lambda x: x[indices], [
            self.users, self.items, self.previous_items, \
            self.context, self.previous_context, \
            self.y
        ]))

    def __iter__(self):
        # Important for optimal and stable performance
        users, items, prev_items, context, prev_context, y = self.shuffle_data()

        for i in range(0, self.num_interactions, self.batch_size):
            yield list(map(lambda x: x.to(self.device), [ 
                users[i:i+self.batch_size], 
                items[i:i+self.batch_size],
                prev_items[i:i+self.batch_size],
                context[i:i+self.batch_size],
                prev_context[i:i+self.batch_size]
            ])), y[i:i+self.batch_size].to(self.device)

class TestDataset(BaseTestDataset):
    def __init__(self, data, train_data, hyper_params, val_data = None):
        super(TestDataset, self).__init__(data, train_data, hyper_params, val_data)

        self.users_cpu, self.items, self.previous_items = [], [], [] 
        self.context, self.previous_context = [], []
        self.y = []

        for u in data:
            for at, (i, policy_dep_context, policy_indep_context) in enumerate(data[u][:-self.hyper_params['eval_K']]):
                
                self.users_cpu.append(u)
                
                # Recognize the previous point
                if at == 0:
                    if self.is_test_set: prev_point = val_data[u][-1]
                    else: prev_point = train_data[u][-1]
                else:
                    prev_point = data[u][at-1]

                # Distance
                self.items.append(i)
                self.previous_items.append(prev_point[0])
                
                # Context (policy_indep_context ONLY)
                self.context.append([
                    data[u][at+i][2] for i in range(self.hyper_params['eval_K'])
                ])
                self.previous_context.append([
                    ([ prev_point ] + data[u][at:])[i][2] for i in range(self.hyper_params['eval_K'])
                ])

                # Next time-step distance
                self.y.append([
                    data[u][at+1+i][0] for i in range(self.hyper_params['eval_K'])
                ])

        # Copying ENTIRE dataset to the GPU
        self.users = torch.LongTensor(self.users_cpu)
        self.context = torch.FloatTensor(self.context)
        self.previous_context = torch.FloatTensor(self.previous_context)
        
        self.items = self.normalize_dist(torch.FloatTensor(self.items))
        self.previous_items = self.normalize_dist(torch.FloatTensor(self.previous_items))
        self.y = torch.FloatTensor(self.y) # Don't normalize while predicting

        self.device = "cuda:0" if is_cuda_available else "cpu"

        self.num_interactions = len(self.users_cpu)

    def __iter__(self):
        for i in range(0, self.num_interactions, self.batch_size):
            yield list(map(lambda x: x.to(self.device), [ 
                self.users[i:i+self.batch_size], 
                self.items[i:i+self.batch_size],
                self.previous_items[i:i+self.batch_size],
                self.context[i:i+self.batch_size],
                self.previous_context[i:i+self.batch_size]
            ])), self.y[i:i+self.batch_size].to(self.device)
