import pickle 
import argparse
from pypcd import pypcd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for new data")
    
    parser.add_argument("--labels",
                        type=str,
                        default="/work/home/acuzow79dd/data/lidar_data/batch_0_pretrain_data/label_txt")
    
    parser.add_argument("--points",
                        type=str, 
                        default="/work/home/acuzow79dd/data/lidar_data/batch_0_pretrain_data/cloud_bin")
    
    args = parser.parse_args()
    return args


def create_infos(args):
    labels_path = args.labels
    points_path = args.points
    # class_names = ['Pedestrian', 'Cyclist', 'Car']
    ped_txt_file = "/work/home/acuzow79dd/data/lidar_data/batch_0_pretrain_data/pedestrian_files.txt"
    
    with open (ped_txt_file, "r") as f:
        ped_files = f.readlines()
        ped_files = [pf.strip("\n").strip(" ") for pf in ped_files]
        ped_files = [pf.split(".")[0] for pf in ped_files]
        f.close()
    
    classes_group = {"Pedestrian":0,
                     "Cyclist":1,
                     "Car":2,
                     "Truck":2}
    
    label_filenames = os.listdir(labels_path)
    pcd_filenames = os.listdir(points_path)
    
    raw_names = [fname.split(".")[0] for fname in label_filenames]
    
    train_annos, test_annos = train_test_split(raw_names, test_size=0.2)
    
    # data_info_train = {"data_list":[]}
    # data_info_test = {"data_list":[]}
    ped_info = {"data_list":[]}
    
    for sample_name in tqdm(raw_names):
        if sample_name in ped_files:
            temp_dict = {"lidar_points":{"lidar_path":None},
                        "instances":None}
            # if sample_name in train_annos:
            #     mode = "train"
            # elif sample_name in test_annos:
            #     mode = "test"
            
            anno_full_path = os.path.join(labels_path, sample_name + ".txt")
            lidar_full_path = os.path.join(points_path, sample_name + ".bin")
            temp_dict["lidar_points"]["lidar_path"] = lidar_full_path
            
            with open(anno_full_path, "r") as f:
                content = f.readlines()
                f.close()
                
            instances = []
            
            for line in content:
                instance_dict = {"bbox_3d":None, "bbox_label_3d":None}
                line = line.strip(" ")
                line_array = line.split(" ")
                
                if len(line_array) == 9:
                    # 1. Old Result Version (Length = 9):
                    obj_name        = line_array[0]
                    dimensions  = [float(line_array[1]), float(line_array[2]), float(line_array[3])]
                    locations   = [float(line_array[4]), float(line_array[5]), float(line_array[6])]
                    rotation_y  = [float(line_array[7])]
                    score       = float(line_array[8])
                    box3d       = locations + dimensions + rotation_y
                elif len(line_array) == 11:
                    # 2. Old label Version (Length = 11):
                    obj_name = line_array[0]
                    dimensions = [float(line_array[5]), float(line_array[4]), float(line_array[6])]
                    locations = [float(line_array[7]), float(line_array[8]), float(line_array[9])]
                    rotation_y = [float(line_array[10])]
                    box3d = locations + dimensions + rotation_y
                elif len(line_array) == 15:
                    # 3. KITTI Standard Label Version (Length = 15):
                    obj_name        = line_array[0]
                    dimensions  = [float(line_array[9]), float(line_array[10]), float(line_array[8])]
                    locations   = [float(line_array[11]), float(line_array[12]), float(line_array[13])]
                    rotation_y  = [float(line_array[14])]
                    box3d       = locations + dimensions + rotation_y
                elif len(line_array) == 16:
                    # 4. KITTI Standard Result Version (Length = 16):
                    obj_name        = line_array[0]
                    dimensions  = [float(line_array[9]), float(line_array[10]), float(line_array[8])]
                    locations   = [float(line_array[11]), float(line_array[12]), float(line_array[13])]
                    rotation_y  = [float(line_array[14])]
                    score       = float(line_array[8])
                    box3d       = locations + dimensions + rotation_y
                elif len(line_array) == 20:
                    # 5. KITTI FP Result Version (Length = 20, for false positive):
                    obj_name        = line_array[0]
                    dimensions  = [float(line_array[9]), float(line_array[10]), float(line_array[8])]
                    locations   = [float(line_array[11]), float(line_array[12]), float(line_array[13])]
                    rotation_y  = [float(line_array[14])]
                    score       = float(line_array[8])
                    ious        = [float(line_array[16]), float(line_array[17]), float(line_array[18]), float(line_array[19])]
                    box3d       = locations + dimensions + rotation_y
                elif len(line_array) == 19:
                    # 6. KITTI FN Label Version (Length = 19, for false negative):
                    obj_name = line_array[0]
                    dimensions = [float(line_array[9]), float(line_array[10]), float(line_array[8])]
                    locations = [float(line_array[11]), float(line_array[12]), float(line_array[13])]
                    rotation_y = [float(line_array[14])]
                    ious = [float(line_array[15]), float(line_array[16]), float(line_array[17]), float(line_array[18])]
                    box3d = locations + dimensions + rotation_y
                
                
                        
                
                
                label3d = classes_group[obj_name]
                
                instance_dict["bbox_3d"] = box3d
                instance_dict["bbox_label_3d"] = label3d
                instances.append(instance_dict)
            
            temp_dict["instances"] = instances

        # if mode == "train":
        #     data_info_train["data_list"].append(temp_dict)
        # elif mode == "test":
        #     data_info_test["data_list"].append(temp_dict)
    
    # with open("train_info.pkl", "wb") as f:
    #     pickle.dump(data_info_train, f)
    #     f.close()
    
    # with open("test_info.pkl", "wb") as f:
    #     pickle.dump(data_info_test, f)
    #     f.close()
    
    with open("/work/home/acuzow79dd/data/lidar_data/batch_0_pretrain_data/pedestrian_info.pkl", "wb") as f:
        pickle.dump(ped_info, f)
        f.close()

        
if __name__ == "__main__":
    args = parse_args()
    create_infos(args=args)
    