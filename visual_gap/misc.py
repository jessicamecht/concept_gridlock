'''
misc code to understand data, try out libraries
'''
import sys
'''
processed file
'''

# import h5py

# path = "/data1/jessica/data/toyota/once_data_w_lanes_compressed.hfd5"
# # path = "/data1/jessica/data/toyota/once_data_w_lanes_compressed_train.hfd5"
# # path = "/data1/jessica/data/toyota/once_data_w_lanes_compressed_train_resized_images.hfd5"

# with h5py.File(path, "r") as f:
#     for i, seq_key in enumerate(list(f.keys())):
#         iter_dict = {}
#         keys_ = f[seq_key].keys()
#         print (seq_key)
#         print (keys_)
#         # print (f[seq_key]['pos'])
#         # print (f[seq_key]['speed'])
#         # # print (len(f[seq_key]['angle'][()]))
#         # print (f[seq_key]['image_array'])
#         # # print (f[seq_key]['seq_name_x'][()])
#         # print (f[seq_key]['segm_masks'])
#         if i>0:
#             break
#         # for key in keys_:
#         #     ds_obj = f[seq_key][key][()]
#         #     iter_dict[key] = ds_obj
#         # self.people_seqs.append(iter_dict)

'''
read json
'''
# import json 

# def load_json(path_to_file):
#     with open(path_to_file) as p:
#         return json.load(p)

# root_path = "../data/annotations/train/000076/000076.json"
# a = load_json(root_path)
# print(a.keys())
# print(a['meta_info'])
# print(len(a['frames']))
# print(a['frames'][0].keys())
# print(a['frames'][1]['frame_id'])
# print(a['frames'][0]['pose'])
# print(a['frames'][0]['annos'].keys())
# print(len(a['frames'][0]['annos']['names']))
# print(len(a['frames'][0]['annos']['boxes_3d']))
# print(a['frames'][0]['annos']['boxes_2d'].keys())
# print(a['frames'][0]['annos']['boxes_2d']['cam01'][0])


'''
read once data processed by Noveen
'''
# import numpy as np
# from collections import defaultdict
# flat_data = np.load("../data/once/metadata.npz")['data']     
# dict_data = defaultdict(list)
# print (len(flat_data))
# weather_map = {}
# for sequence_id, timestamp, gap, speed, weather, image_index in flat_data:
#     if weather not in weather_map: weather_map[weather] = len(weather_map)
# print(weather_map)
# for sequence_id, timestamp, gap, speed, weather, image_index in flat_data:
#     dict_data[sequence_id].append([
#         float(gap),

#         # Logging policy dependent
#         [
#             0, # time-gap NOTE: need to calculate using physics
#             float(speed), # self-speed
#             0 # current self-acceleration
#         ],

#         # Independent of the logging policy
#         [
#             0, # Leader-speed
#             weather_map[weather], # weather information not available,
#             0 # self-vehicle-name
#         ],

#         timestamp,       
#     ])

# data = []
# for sequence_id in dict_data:
#     # The file is already sorted by the timestamp, but just making sure
#     user_history = sorted(dict_data[sequence_id], key = lambda x: x[-1])
#     # Removing the timestamp now as we don't need it anymore
#     user_history = list(map(lambda x: x[:-1], user_history))
#     data.append(user_history)

# print(data[0][:10])
# x = [ len(i) for i in data]
# print ('seq_len', x)
# print (sum(x))
# image_paths = np.load("../data/once/metadata.npz")['image_paths']   
# print (len(image_paths))
# # print(len(dict_data))
# # print(dict_data.keys())
# total = 0
# for i in x:
#     total += i
#     print (image_paths[total-1])


### find sequence in h5py

# sequence_ids = ['000076', '000080', '000092', '000104', '000113', '000121', '000027', '000028', '000112', '000201']
# import h5py

# path = "/data1/jessica/data/toyota/once_data_w_lanes_compressed.hfd5"
# # # path = "/data1/jessica/data/toyota/once_data_w_lanes_compressed_train.hfd5"
# # path = "/data1/jessica/data/toyota/once_data_w_lanes_compressed_train_resized_images.hfd5"

# with h5py.File(path, "r") as f:
#     print ([s for s in f.keys()])
#     for i, seq_key in enumerate(sequence_ids):
#         iter_dict = {}
#         keys_ = f[seq_key].keys()
#         print (seq_key)
#         print (f[seq_key]['pos'])


'''

'''
import numpy as np
from collections import defaultdict
data = np.load("../data/once/old_metadata.npz")
flat_data = data['data'] 
image_paths = data['image_paths'] 

print (flat_data[0])










