'''
Analyze data
'''
import h5py

root_dir = "/data3/jiacheng/deepz_data/comma/"
split = 'test'
data_path = root_dir+f"comma_{split}_w_desired_filtered.h5py"

h5_file = h5py.File(data_path, "r")
keys = list(h5_file.keys())
seq_key  = keys[0]

print (h5_file[seq_key].keys())