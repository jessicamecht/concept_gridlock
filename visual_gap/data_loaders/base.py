import torch

class CombinedBase:
    def __init__(self, hyper_params):
        self.hyper_params = hyper_params
        self.batch_size = hyper_params['batch_size']
        self.num_users, self.num_items = hyper_params['total_users'], hyper_params['total_items']

    def __len__(self): return (self.num_interactions // self.batch_size) + 1

    def __del__(self):
        try:
            self.p.terminate() ; self.p.join()
        except: pass

    def pad(self, arr, max_len = None, pad_with = -1, side = 'right'):
        seq_len = max_len if max_len is not None else max(map(len, arr))

        for i in range(len(arr)):
            while len(arr[i]) < seq_len: 
                if side == 'right':
                    pad_elem = arr[i][-1] if pad_with == -1 else pad_with
                    if type(arr[i][-1]) == list: pad_elem = [ pad_elem ] * len(arr[i][-1])
                    arr[i].append(pad_elem)
                else:
                    pad_elem = arr[i][0] if pad_with == -1 else pad_with
                    if type(arr[i][0]) == list: pad_elem = [ pad_elem ] * len(arr[i][0])
                    arr[i] = [ pad_elem ] + arr[i]
            
            arr[i] = arr[i][-seq_len:] # Keep last `seq_len` items
        return arr

    def sequential_pad(self, arr, hyper_params):
        # Padding left side so that we can simply take out [:, -1, :] in the output
        return self.pad(arr, 
            max_len = hyper_params['max_seq_len'], 
            pad_with = hyper_params['pad_with'], 
            side = 'left'
        )

    def scatter(self, batch, tensor_kind, last_dimension):
        ret = tensor_kind(len(batch), last_dimension).zero_()

        if not torch.is_tensor(batch):
            if ret.is_cuda: batch = torch.cuda.LongTensor(batch)
            else: batch = torch.LongTensor(batch)

        return ret.scatter_(1, batch, 1)

    def normalize_dist(self, dist_tensor):
        if type(dist_tensor) == list:
            return list(map(
                lambda x: (x - self.hyper_params['mean_dist']) / self.hyper_params['std_dist'], 
                dist_tensor
            ))
        return (dist_tensor - self.hyper_params['mean_dist']) / self.hyper_params['std_dist']

    def un_normalize_dist(self, dist_tensor):
        return (dist_tensor * self.hyper_params['std_dist']) + self.hyper_params['mean_dist']

class BaseTrainDataset(CombinedBase):
    def __init__(self, data, hyper_params):
        super(BaseTrainDataset, self).__init__(hyper_params)

        self.data = data

class BaseTestDataset(CombinedBase):
    def __init__(self, data, train_data, hyper_params, val_data):
        super(BaseTestDataset, self).__init__(hyper_params)

        self.is_test_set = val_data is not None
        self.data, self.train_data = data, train_data
