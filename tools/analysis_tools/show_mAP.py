from cProfile import label
import json
import matplotlib.pyplot as plt
from yaml import KeyToken

# 读取JSON文件
json_path = 'work_dirs/batch0_pretrain_dataset_pointpillars_tiny_3class_cyclic_ep80/20240530_141539/vis_data/scalars.json'
save_path = 'work_dirs/batch0_pretrain_dataset_pointpillars_tiny_3class_cyclic_ep80/20240530_141539/vis_data/mAP40.jpg'
key_param1 = []
key_param2 = []
with open(json_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        key_tmp = 'custom 3d metric/pred_instances_3d/KITTI/Overall_BEV_AP40_hard'
        if key_tmp in data:
            key_param1.append(data[key_tmp])
        key_tmp = 'custom 3d metric/pred_instances_3d/KITTI/Overall_3D_AP40_hard'
        if key_tmp in data:
            key_param2.append(data[key_tmp])

# 绘制图像
plt.figure(figsize=(10, 5))
plt.plot(key_param1,label='mAP40_bev')
plt.plot(key_param2,label='mAP40_3d')
plt.title('mAP40')
plt.xlabel('epoch')
plt.ylabel('mAP')
plt.legend(loc='upper left')
plt.savefig(save_path)
plt.show()