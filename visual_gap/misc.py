'''
misc code to understand data, try out libraries
'''

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

























