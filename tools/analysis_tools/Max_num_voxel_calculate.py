import sys

from trimesh import PointCloud
sys.path.append('.')
from networkx import predecessor
import onnxruntime
import argparse
import numpy as np
import os
from onnx_utils.preprocess import Preprocess
import torch
import pickle as pkl
import json
import matplotlib.pyplot as plt
import numpy as np

# point_cloud = [-16, -16, -4, 80, 16, 3]
point_cloud = [-24.0, -32.0, -4, 48.0, 32.0, 3]
voxel_size = [0.2, 0.2, point_cloud[5] - point_cloud[2]]
dx = point_cloud[3] - point_cloud[0]
dy = point_cloud[4] - point_cloud[1]
feat_size = [round(dy / voxel_size[1]), round(dx / voxel_size[0]), 1]

data_path = '/work/share/acuzow79dd/data/lidar_data/batch_0_pretrain_data/pkls/batch_0_pretrain_data_infos_val.pkl'
bin_path = '/work/share/acuzow79dd/data/lidar_data/batch_0_pretrain_data/cloud_bin'
with open(data_path,'rb') as f:
    data = pkl.load(f)
data = data['data_list']


nums_txt = 'num_points_v2_1.txt'
nums = []
with open(nums_txt,'w') as f:
    for each_data in data:
        bin = each_data['lidar_points']['lidar_path']
        input = bin_path +'/' + bin
        preprocessor = Preprocess(points_path=input, 
                                    load_dim=4, 
                                    use_dim=4,
                                    point_cloud_range=point_cloud,
                                    max_num_points=32,
                                    voxel_size=voxel_size,
                                    max_voxels=50000
                                    )        
        pre_data = preprocessor.preprocess()
        num_points = pre_data['num_points'].shape[0]
        f.write(str(num_points) +'\n')
        print(num_points)
        nums.append(num_points)
x = range(len(nums))
y = np.array(nums)
plt.plot(x,y)
plt.savefig('num_points_v2_1.jpg')
plt.show()

if __name__ == "__main__":
    pass