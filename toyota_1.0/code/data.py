import numpy as np
from matplotlib import pyplot as plt

from preprocess import rating_data_toyota, rating_data_openacc, rating_data_once
from utils import get_data_loader_class

def load_data(hyper_params):
    train_loader_class, test_loader_class = get_data_loader_class(hyper_params)

    data_holder = DataHolder(hyper_params)
    print("# of users: {}\n# of items: {}".format(data_holder.num_users, data_holder.num_items))

    hyper_params['total_users']  = data_holder.num_users
    hyper_params['total_items']  = data_holder.num_items
    hyper_params['mean_dist']    = data_holder.mean_dist
    hyper_params['std_dist']     = data_holder.std_dist
    hyper_params['context_size'] = data_holder.context_size

    train_loader = train_loader_class(data_holder.train, hyper_params)
    val_loader = test_loader_class(data_holder.val, data_holder.train, hyper_params)
    test_loader = test_loader_class(
        data_holder.test, data_holder.train, hyper_params, val_data = data_holder.val
    )

    return train_loader, test_loader, val_loader, hyper_params

class DataHolder:
    def __init__(self, hyper_params):
        if hyper_params['dataset'].lower() == 'openacc':
            data_object = rating_data_openacc(hyper_params)
        elif hyper_params['dataset'].lower() == 'toyota':
            data_object = rating_data_toyota(hyper_params)
        elif hyper_params['dataset'].lower() == 'once':
            data_object = rating_data_once(hyper_params)
        
        data_object.load_index()
        self.data, self.index = data_object.data, data_object.index
        
        self.num_items = 0
		
        # Since these could be arbitrary features
        self.mean_dist, self.std_dist = self.normalize_features()
        
        self.num_users = len(self.data)

        self.context_size = len(self.data[0][0][2]) # Policy independent context only

    def normalize_features(self):
        all_dist = []
        for u in range(len(self.data)):
            for i, _, _ in self.data[u]:
                all_dist.append(i)
        all_dist = np.array(all_dist)

        # temp = sorted(all_dist)
        # plt.scatter(list(range(len(temp))), temp)
        # plt.savefig("distances.png")

        return np.mean(all_dist), np.std(all_dist)

    def remap(self):
        ## Counting number of unique items before
        valid_items = set()

        at = 0
        for u in range(len(self.data)):
            for i, _ in self.data[u]:
                if self.index[at] != -1: valid_items.add(i)
                at += 1
        assert at == len(self.index)

        ## Map creation done!
        item_map = dict(zip(list(valid_items), list(range(len(valid_items)))))

        new_data, new_index = [], []
        at = 0
        for u in range(len(self.data)):
            temp = []
            for i, policy_dep_context, policy_indep_context in self.data[u]:
                if self.index[at] == -1: 
                    at += 1
                    continue
                temp.append([ item_map[i], policy_dep_context, policy_indep_context ])
                new_index.append(self.index[at])
                at += 1
            new_data.append(temp)

        self.data = new_data
        self.index = new_index
        self.num_items = len(valid_items)

    def select(self, index_val):
        ret, at = {}, 0
        for u in range(len(self.data)):
            temp = []
            for i, policy_dep_context, policy_indep_context in self.data[u]:
                if self.index[at] == index_val: 
                    temp.append([ i, policy_dep_context, policy_indep_context ])
                at += 1
            if len(temp) > 0: ret[u] = temp
        
        assert at == len(self.index)
        return ret

    @property
    def train(self): return self.select(0)

    @property
    def val(self): return self.select(1)

    @property
    def test(self): return self.select(2)

    @property
    def num_train_interactions(self): return int(sum(map(lambda x: x == 0, self.index)))

    @property
    def num_val_interactions(self): return int(sum(map(lambda x: x == 1, self.index)))

    @property
    def num_test_interactions(self): return int(sum(map(lambda x: x == 2, self.index)))
