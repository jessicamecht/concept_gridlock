import math
import torch
import numpy as np

from data_loaders.base import BaseTrainDataset, BaseTestDataset
from torch_utils import is_cuda_available

class TrainDataset(BaseTrainDataset):
    def __init__(self, data, hyper_params):
        super(TrainDataset, self).__init__(data, hyper_params)
        
        self.item_sequence, self.context_sequence, self.y = [], [], []

        for u in data:
            user_item_sequence = self.normalize_dist(
                list(map(lambda x: x[0], data[u]))
            )
            user_context_sequence = list(map(lambda x: x[2], data[u]))

            # Append to `y` before since we need the entire sequence
            if hyper_params['horizon']:
                # self.y.append([ user_item_sequence[at+2:][:hyper_params['horizon_K']] for at in range(len(user_item_sequence) - hyper_params['horizon_K'] - 2) ])
                y = [ user_item_sequence[at:][:hyper_params['horizon_K']] for at in range(1, len(user_item_sequence) - (hyper_params['horizon_K'] - 1)) ]
            else:
                y = user_item_sequence[1:]

            # Now crop and append to `x`
            if hyper_params['horizon'] and hyper_params['horizon_K'] > 1:
                user_item_sequence = user_item_sequence[:-(hyper_params['horizon_K']-1)]
                user_context_sequence = user_context_sequence[:-(hyper_params['horizon_K']-1)]

            item_sequence = user_item_sequence[:-1]
            context_sequence = user_context_sequence[:-1]

            if hyper_params['staggered_training']:
                for i in range(0, len(item_sequence) - hyper_params['max_seq_len'], hyper_params['seq_step_size']):
                    # If it's the last step, make sure the model trains on the actual last sequence block
                    if i + hyper_params['seq_step_size'] >= len(item_sequence) - hyper_params['max_seq_len']:
                        i = len(item_sequence) - hyper_params['max_seq_len']

                    self.item_sequence.append(item_sequence[i:i+hyper_params['max_seq_len']])
                    self.context_sequence.append(context_sequence[i:i+hyper_params['max_seq_len']])
                    self.y.append(y[i:i+hyper_params['max_seq_len']])
            else: 
                self.item_sequence.append(item_sequence)
                self.context_sequence.append(context_sequence)
                self.y.append(y)

        if hyper_params['staggered_training']: target_fn = torch.FloatTensor
        else: target_fn = lambda x: torch.FloatTensor(self.sequential_pad(x, hyper_params))

        self.item_sequence, self.context_sequence, self.y = list(map(
            target_fn, 
            [ self.item_sequence, self.context_sequence, self.y ]
        ))

        self.device = "cuda:0" if is_cuda_available else "cpu"

        self.num_interactions = len(self.item_sequence)

    def shuffle_data(self):
        indices = np.arange(self.num_interactions)
        np.random.shuffle(indices)

        return list(map(lambda x: x[indices], [
            self.item_sequence, self.context_sequence, self.y
        ]))

    def __iter__(self):
        # Important for optimal and stable performance
        item_sequence, context_sequence, y = self.shuffle_data()

        for i in range(0, self.num_interactions, self.batch_size):
            yield [ 
                item_sequence[i:i+self.batch_size].to(self.device), 
                context_sequence[i:i+self.batch_size].to(self.device)
            ], y[i:i+self.batch_size].to(self.device)

    def __len__(self): 
        return math.ceil(float(self.num_interactions) / float(self.batch_size))

class TestDataset(BaseTestDataset):
    def __init__(self, data, train_data, hyper_params, val_data = None):
        super(TestDataset, self).__init__(data, train_data, hyper_params, val_data)

        self.batch_size = 64

        self.item_sequence = [] 
        self.context_sequence = []
        self.y, self.prediction_mask = [], []

        self.test_lengths = []

        for u in data:
            if self.is_test_set:
                main_sequence = train_data[u] + val_data[u] + data[u]
                predict_after = len(train_data[u]) + len(val_data[u])
            else:
                main_sequence = train_data[u] + data[u]
                predict_after = len(train_data[u])

            user_item_sequence = self.normalize_dist(
                list(map(lambda x: x[0], main_sequence))
            )
            user_context_sequence = list(map(lambda x: x[2], main_sequence))

            self.test_lengths.append(len(main_sequence) - predict_after)

            self.item_sequence.append(user_item_sequence[:-1])
            self.context_sequence.append(user_context_sequence[:-1])
            
            y_seq = user_item_sequence[1:]
            
            self.y.append([
                y_seq[i:i+hyper_params['eval_K']] for i in range(len(y_seq) - hyper_params['eval_K'])
            ])
            
            prediction_mask = np.zeros(len(y_seq) - hyper_params['eval_K'])
            prediction_mask[predict_after-1:] = 1.0
            self.prediction_mask.append(prediction_mask.tolist())

        # NOTE: Take the last `max_seq_len` points in each sequence for 
        # the evaluation set since SASRec can't handle arbitrary len sequences

        # Step-1: Take last `max_seq_len` points in each sequence
        self.item_sequence, self.context_sequence, self.y, self.prediction_mask = list(map(
            lambda x: torch.FloatTensor(self.sequential_pad(x, hyper_params)), 
            [ self.item_sequence, self.context_sequence, self.y, self.prediction_mask ]
        ))

        # Step-2: `y` doesn't have `max_seq_len` points ; it has 
        # `max_seq_len - eval_K` points in each sequence
        self.prediction_mask = self.prediction_mask.type(torch.BoolTensor)
        self.y, self.prediction_mask = list(map(
            lambda x: x[:, -(hyper_params['max_seq_len'] - hyper_params['eval_K']):],
            [ self.y, self.prediction_mask ]
        ))

        # While prediction, calculate the actual MSE
        self.y = self.un_normalize_dist(self.y)

        self.device = "cuda:0" if is_cuda_available else "cpu"
        self.num_interactions = len(self.item_sequence)

    def __iter__(self):
        for i in range(0, self.num_interactions, self.batch_size):

            if self.hyper_params['model_type'] == "RNN": copy_fn = lambda x: x # RNN evaluation on CPU
            else: copy_fn = lambda x: x.to(self.device)

            yield list(map(copy_fn, [ 
                self.item_sequence[i:i+self.batch_size],
                self.context_sequence[i:i+self.batch_size]
            ])), list(map(copy_fn, [
                self.prediction_mask[i:i+self.batch_size],
                self.y[i:i+self.batch_size]
            ]))

    def __len__(self): 
        return math.ceil(self.num_interactions / self.batch_size)
