import os
import numpy as np
from tqdm import tqdm

anno_path = "/work/home/acuzow79dd/data/lidar_data/batch_0_pretrain_data/label_txt"

filenames = os.listdir(anno_path)

pedestrian_files = []

for fname in tqdm(filenames):
    full_path = os.path.join(anno_path, fname)
    with open(full_path, "r") as f:
        lines = f.readlines()
        f.close()
    
    lines = [line.strip("\n").strip(" ") for line in lines]
    obj_names = [line.split(" ")[0] for line in lines]
    if "Pedestrian" in obj_names or "pedestrian" in obj_names:
        pedestrian_files.append(fname)

overal_context = "\n".join(pedestrian_files)

with open("/work/home/acuzow79dd/data/lidar_data/batch_0_pretrain_data/pedestrian_files.txt", "w") as f:
    f.write(overal_context)
    f.close()