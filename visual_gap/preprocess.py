import os
import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from hyper_params import hyper_params

BASE_DATA_PATH = "../data/"

def get_data_path(dataset):
    return "{}/{}/preprocessed/total_data.hdf5".format(BASE_DATA_PATH, dataset)

def get_index_path(hyper_params):
    return "{}/{}/preprocessed/last_10_percent/index.npz".format(
        BASE_DATA_PATH,
        hyper_params['dataset'], 
    )

def get_all_files(dataset): 
    directory = "{}/{}/raw/".format(BASE_DATA_PATH, dataset)
    
    if dataset == "toyota":
        file_names = os.listdir(directory)
        all_files = list(map(lambda x: directory + x, file_names))
    elif dataset == "openacc":
        all_files = []
        for name in os.listdir(directory):
            if os.path.isdir(directory + name) and name not in { 'One_vehicle_multiple_drivers_on_road_campaign' }:
                dataset = directory + name + "/"
                all_files += [ 
                    dataset + file for file in os.listdir(dataset) if file.endswith(".csv") and not file.endswith("cations.csv") 
                ]

    return all_files

class rating_data_base:
    def __init__(self, hyper_params, dataset_name):
        self.hyper_params = hyper_params
        self.dataset_name = dataset_name
        self.load_data()

        self.index = [] # 0: train, 1: validation, 2: test, -1: removed/ignore
        for user_data in self.data:
            for _ in range(len(user_data)): self.index.append(42)

        self.complete_data_stats = None

    def train_test_split(self):
        at = 0
        for user in range(len(self.data)):
            # Split each user history into 80/10/10% train/val/test
            first_split_point = int(0.8 * len(self.data[user]))
            second_split_point = int(0.9 * len(self.data[user]))

            for timestep, _ in enumerate(self.data[user]):
                if timestep < first_split_point: self.index[at] = 0
                elif timestep < second_split_point: self.index[at] = 1
                else: self.index[at] = 2
                at += 1

        assert at == len(self.index)
        self.complete_data_stats = None

    def measure_data_stats(self):
        num_users, num_items, num_interactions, num_test, num_val = set(), set(), 0, 0, 0
        at = 0
        for u in range(len(self.data)):
            for i, _, _ in self.data[u]:
                if self.index[at] == 0: num_interactions += 1
                if self.index[at] == 1: num_val += 1
                if self.index[at] == 2: num_test += 1

                if self.index[at] != -1:
                    num_users.add(u)
                    num_items.add(i)
                at += 1

        data_stats = {}
        data_stats["num_users"] = len(num_users)
        data_stats["num_items"] = len(num_items)
        data_stats["num_train_interactions"] = num_interactions
        data_stats["num_test"] = num_test
        data_stats["num_val"] = num_val

        return data_stats

    def save_index(self, statistics = True):
        with open(get_index_path(self.hyper_params), "wb") as f: 
            if statistics:
                np.savez_compressed(f, data = self.index, stats = self.measure_data_stats())
            else:
                np.savez_compressed(f, data = self.index)

    def load_index(self):
        self.index = np.load(get_index_path(self.hyper_params))['data']
        if self.complete_data_stats is None: self.complete_data_stats = self.measure_data_stats()

    def save_data(self):
        flat_data = []
        for u in range(len(self.data)):
            flat_data += list(map(lambda x: [ u, x[0] ] + x[1] + x[2], self.data[u]))
        flat_data = np.array(flat_data)

        print(flat_data[0])

        shape = [ len(flat_data) ]

        with h5py.File(get_data_path(self.dataset_name), 'w') as file:
            dset = {}
            dset['user'] = file.create_dataset("user", shape, dtype = 'i4', maxshape = shape, compression="gzip")
            # dset['disc_dist'] = file.create_dataset("disc_dist", shape, dtype = 'i4', maxshape = shape, compression="gzip")
            dset['dist'] = file.create_dataset("dist", shape, dtype = 'f', maxshape = shape, compression="gzip")
            dset['time_gap'] = file.create_dataset("time_gap", shape, dtype = 'f', maxshape = shape, compression="gzip")
            dset['speed'] = file.create_dataset("speed", shape, dtype = 'f', maxshape = shape, compression="gzip")
            dset['curr_acc'] = file.create_dataset("curr_acc", shape, dtype = 'f', maxshape = shape, compression="gzip")
            dset['lead_speed'] = file.create_dataset("lead_speed", shape, dtype = 'f', maxshape = shape, compression="gzip")
            dset['weather'] = file.create_dataset("weather", shape, dtype = 'i4', maxshape = shape, compression="gzip")
            dset['vehicle'] = file.create_dataset("vehicle", shape, dtype = 'i4', maxshape = shape, compression="gzip")

            dset['user'][:] = flat_data[:, 0]
            # dset['disc_dist'][:] = flat_data[:, 1]
            dset['dist'][:] = flat_data[:, 1]
            dset['time_gap'][:] = flat_data[:, 2]
            dset['speed'][:] = flat_data[:, 3]
            dset['curr_acc'][:] = flat_data[:, 4]
            dset['lead_speed'][:] = flat_data[:, 5]
            dset['weather'][:] = flat_data[:, 6]
            dset['vehicle'][:] = flat_data[:, 7]

    def load_data(self):
        if os.path.exists(get_data_path(self.dataset_name)): 
            print ("Data exists!!! ***")
            with h5py.File(get_data_path(self.dataset_name), 'r') as f:
                flat_data = np.array(list(zip(
                    f['user'][:], 
                    # f['disc_dist'][:], 
                    f['dist'][:],
                    # Logging policy dependent context
                    f['time_gap'][:], f['speed'][:], f['curr_acc'][:], 
                    # Logging policy independent context
                    f['lead_speed'][:], f['weather'][:], f['vehicle'][:]
                )))

            self.data = [ [] for _ in range(int(max(flat_data[:, 0])) + 1) ]
            for i in range(len(flat_data)):
                self.data[int(flat_data[i, 0])].append([
                    flat_data[i, 1], 
                    flat_data[i, 2:5], # Logging policy dependent context
                    flat_data[i, 5:] # Logging policy independent context
                ])
                
        else: 
            print ("Non exists!!! ***")
            self.read_raw_data()
            # self.save_data()

        # Convert weather and vehicle to one-hot
        self.convert_to_one_hot()
        self.normalize_context()

    def convert_to_one_hot(self):
        # Count number of unique weathers and vehicles
        total_weathers, total_vehicles = {}, {}
        for u in range(len(self.data)):
            for _, _, policy_indep_context in self.data[u]:
                weather, vehicle = policy_indep_context[1], policy_indep_context[2]
                if vehicle not in total_vehicles: total_vehicles[vehicle] = len(total_vehicles)
                if weather not in total_weathers: total_weathers[weather] = len(total_weathers)

        # Convert each context's weather and vehicle to one-hot
        def convert(context):
            lead_speed, weather, vehicle = context

            one_hot_weather = np.zeros(len(total_weathers)) ; one_hot_weather[total_weathers[weather]] = 1.0
            one_hot_vehicle = np.zeros(len(total_vehicles)) ; one_hot_vehicle[total_vehicles[vehicle]] = 1.0
            
            return [
                lead_speed,
                *one_hot_weather.tolist(),
                *one_hot_vehicle.tolist()
            ]

        # Final
        for u in range(len(self.data)):
            for at in range(len(self.data[u])):
                self.data[u][at][2][:3] = convert(self.data[u][at][2][:3])

    def normalize_context(self):
        all_policy_dep_context, lead_speeds = [], []
        for u in range(len(self.data)):
            for _, policy_dep_context, policy_indep_context in self.data[u]:
                all_policy_dep_context.append(policy_dep_context)
                lead_speeds.append(policy_indep_context[0])
        all_policy_dep_context = np.array(all_policy_dep_context)
        lead_speeds = np.array(lead_speeds)

        mean, std = np.mean(all_policy_dep_context, axis = 0), np.std(all_policy_dep_context, axis = 0)
        mean_lead_speed, std_lead_speed = np.mean(lead_speeds), np.std(lead_speeds)

        for u in range(len(self.data)):
            for at in range(len(self.data[u])):

                for i in range(len(mean)):
                    self.data[u][at][1][i] -= mean[i]
                    if std[i] != 0.0: self.data[u][at][1][i] /= std[i]
                
                self.data[u][at][2][0] -= mean_lead_speed
                if std_lead_speed != 0.0: self.data[u][at][2][0] /= std_lead_speed

class rating_data_toyota(rating_data_base):
    def __init__(self, hyper_params):
        super(rating_data_toyota, self).__init__(hyper_params, "toyota")

    def read_raw_data(self):
        all_files = get_all_files("toyota")
        
        self.data = [] 
        for file_path in tqdm(all_files):
            # Each file is a user history
            user_history, line_number = [], 0
            # Each file could have a different column mapping
            header_map_for_this_file = {}

            for line in open(file_path, 'r'):
                line_number += 1

                if line_number == 1: 
                    headers = list(map(lambda x: x.strip(), line.strip().split(",")))
                    # Mapping from key to index in line
                    header_map_for_this_file = dict(zip(headers, list(range(len(headers)))))
                else:
                    temp = list(map(float, line.strip().split(",")))
                    user_history.append([
                        # self.discretize_distance(temp[header_map_for_this_file['distance-gap']]), # discretized-distance
                        temp[header_map_for_this_file['distance-gap']],
                        
                        # Logging policy dependent
                        [
                            temp[header_map_for_this_file['time-gap']],
                            temp[header_map_for_this_file['current-speed']],
                            temp[header_map_for_this_file['current-acceleration']],
                        ],

                        # Independent of the logging policy
                        [
                            temp[header_map_for_this_file['lead-speed']],
                            temp[header_map_for_this_file['weather']],
                            temp[header_map_for_this_file['vehicle']]
                        ],

                        # Timestamp just for sorting
                        temp[header_map_for_this_file['timestamp']]
                    ])

            # The file is already sorted by the timestamp, but just making sure
            user_history = sorted(user_history, key = lambda x: x[-1])
            # Removing the timestamp now as we don't need it anymore
            user_history = list(map(lambda x: x[:-1], user_history))

            # Append this user sequence
            self.data.append(user_history)

class rating_data_openacc(rating_data_base):
    def __init__(self, hyper_params):
        super(rating_data_openacc, self).__init__(hyper_params, "openacc")

    def get_manual_driven_files(self, all_files):
        manual_files, manual_cars_in_each_file = [], []

        for file_path in all_files:
            f = open(file_path, 'r') ; lines = f.readlines() ; f.close()
        
            drive_type_indices = [ at for at, col in enumerate(lines[5].strip().split(",")) if col.startswith("Driver") ]
            drive_type_indices = drive_type_indices[1:] # Leader driver doesn't matter as it doesn't have a valid gap
            driver_types = [ set() for _ in drive_type_indices ]
            
            given_manual = False
            for line in lines[:6]:
                line = line.strip().split(",")
                if line[0] == "ACC" and line[1] in { '0', '2' }: 
                    given_manual = True
                    break
            
            for line in lines[6:]:
                line = line.strip().split(",")
                for at, i in enumerate(drive_type_indices):
                    driver_types[at].add(line[i])
        
            atleast_one_human, human_drive_indices = False, []
            for at, d in enumerate(driver_types):
                if len(d) == 1 and list(d)[0] == 'Human': 
                    atleast_one_human = True
                    human_drive_indices.append(at)
                    
            if given_manual and not atleast_one_human:
                human_drive_indices.append(0)
                    
            if atleast_one_human or given_manual:
                manual_files.append(file_path)
                manual_cars_in_each_file.append(human_drive_indices)

        return manual_files, manual_cars_in_each_file
    
    def read_raw_data(self):
        all_files = get_all_files("openacc")
        all_files, manual_cars_in_each_file = self.get_manual_driven_files(all_files)
        
        self.data = [] 
        for file_num, file_path in enumerate(tqdm(all_files)):
            f = open(file_path, 'r') ; lines = f.readlines() ; f.close()

            gap_indices = [ at for at, col in enumerate(lines[5].strip().split(",")) if col.startswith("IVS") ]
            speed_indices = [ at for at, col in enumerate(lines[5].strip().split(",")) if col.startswith("Speed") ]
            vehicles = list(filter(lambda x: len(x) > 0, lines[1].strip().split(",")[1:]))

            # NOTE: The ZalaZone dataset has some problems. Check it's notes for more clarity.
            if not (len(gap_indices) + 1 == len(speed_indices) == len(vehicles)): continue 
            
            this_trajectories = [ [] for _ in gap_indices ] # Leader trajectory NOT stored
            
            for line in lines[6:]:
                line = line.strip().split(",")
                
                for at in range(len(gap_indices)):
                    try:
                        this_trajectories[at].append([
                            float(line[gap_indices[at]]), 

                            # Logging policy dependent
                            [
                                0, # time-gap NOTE: need to calculate using physics
                                float(line[speed_indices[at + 1]]), # self-speed
                                0 # current self-acceleration
                            ],

                            # Independent of the logging policy
                            [
                                float(line[speed_indices[at]]), # Leader-speed
                                0, # weather information not available,
                                vehicles[at + 1] # self-vehicle-name
                            ]
                        ])
                    except Exception as e:
                        # NOTE: There are missing values in several files, for several drivers 
                        # in different time periods. This is due to data acquisition issues.
                        continue

            # Append this user sequence
            for car_index in manual_cars_in_each_file[file_num]:
                self.data.append(this_trajectories[car_index])

        self.convert_vehicle_name_to_id()

    def convert_vehicle_name_to_id(self):
        # Count number of unique weathers and vehicles
        total_vehicles = {}
        for u in range(len(self.data)):
            for _, _, policy_indep_context in self.data[u]:
                vehicle = policy_indep_context[-1]
                if vehicle not in total_vehicles: total_vehicles[vehicle] = len(total_vehicles)

        # Final
        for u in range(len(self.data)):
            for at in range(len(self.data[u])):
                self.data[u][at][2][-1] = total_vehicles[self.data[u][at][2][-1]]

class rating_data_once(rating_data_base):
    def __init__(self, hyper_params):
        super(rating_data_once, self).__init__(hyper_params, "once")
    
    def read_raw_data(self):
        raw_data = np.load("{}/once/metadata.npz".format(BASE_DATA_PATH))
        flat_data = raw_data['data']
        image_paths = raw_data['image_paths'] 

        dict_data = defaultdict(list)

        weather_map = {}
        for (sequence_id, timestamp, gap, speed, weather, image_index), image_path in zip(flat_data, image_paths):
            if weather not in weather_map: weather_map[weather] = len(weather_map)
        print(weather_map)

        i = 0
        for (sequence_id, timestamp, gap, speed, weather, image_index), image_path in zip(flat_data, image_paths):
            if self.hyper_params['image_feature'] is True:
                dict_data[sequence_id].append([
                    float(gap),

                    # Logging policy dependent
                    [
                        0, # time-gap NOTE: need to calculate using physics
                        float(speed), # self-speed
                        0 # current self-acceleration
                    ],

                    # Independent of the logging policy
                    [
                        0, # Leader-speed
                        weather_map[weather], # weather information not available,
                        0 # self-vehicle-name
                    ]+np.load(image_path).tolist(), 
                    
                    timestamp        
                ])
            else: 
                dict_data[sequence_id].append([
                    float(gap),

                    # Logging policy dependent
                    [
                        0, # time-gap NOTE: need to calculate using physics
                        float(speed), # self-speed
                        0 # current self-acceleration
                    ],

                    # Independent of the logging policy
                    [
                        0, # Leader-speed
                        weather_map[weather], # weather information not available,
                        0 # self-vehicle-name
                    ], 
                    
                    timestamp        
                ])

        self.data = []
        for sequence_id in dict_data:
            # The file is already sorted by the timestamp, but just making sure
            user_history = sorted(dict_data[sequence_id], key = lambda x: x[-1])
            # Removing the timestamp now as we don't need it anymore
            user_history = list(map(lambda x: x[:-1], user_history))
            self.data.append(user_history)

        # print('sample data', self.data[0][:2])
        # sys.exit()

if __name__ == "__main__":
    print("Preprocessing {}...".format(hyper_params['dataset']))
    if hyper_params['dataset'] == 'toyota':
        data_object = rating_data_toyota(hyper_params)
    elif hyper_params['dataset'] == 'openacc':
        data_object = rating_data_openacc(hyper_params)
    elif hyper_params['dataset'] == 'once':
        data_object = rating_data_once(hyper_params)

    data_object.train_test_split()
    data_object.save_index()
