import numpy as np
import math
import json 
import pickle
import os 
from mmdet3d.structures.bbox_3d.lidar_box3d import LiDARInstance3DBoxes
import torch
import argparse
from tqdm import tqdm
from mmdet3d.structures.ops import box_np_ops
import mmengine

META_INFO = {
    'categories': 
        {
        'Pedestrian': 0,
        'Cyclist': 1,
        'Car': 2,
        'Truck' : 3,
        'Misc' : 4,
          },
'dataset': 'custom',
'info_version': '1.0'
 }


CLASS_MAPPING = {
    'Pedestrian/Pedestrian': 0,
    
    'Cyclist/Bicyclist': 1,
    'Cyclist/Bike': 1,
    'Cyclist/Box-Tricycle': 1,
    'Cyclist/Box-Tricyclist': 1,
    'Cyclist/Motor': 1,
    'Cyclist/MotorCyclist': 1,
    'Cyclist/Tricycle': 1,
    'Cyclist/Tricyclist': 1,
    'Cyclist/mini-electronic-vehicle' : 1,
    'Cyclist/pram-trolley' : 1, 
    
    'Vehicle/Car': 2,
    'Vehicle/MPV': 2,
    'Vehicle/SUV': 2,
    'Vehicle/Electric-cruiser' : 2,
    'Vehicle/OtherVehicle' : 2,
    'Vehicle/PickupCar' : 2,
    

    
    'Vehicle/Engineering-Vehicle': 2,
    'Vehicle/Bus': 2,
    'Vehicle/SmallTruck': 2,
    'Vehicle/Truck': 2,
    'Vehicle/smallbus' : 2,
    'Vehicle/Trailer-head' :2,
    'Vehicle/Trailer' :2,
    
    'Misc/Cone': 4,
    'Misc/Fence' : 4,
    'Misc/Barrier-pier' : 4,
    'Misc/railroadgate' : 4,
    'Misc/Traffic-Pole' : 4 ,
    'Misc/no_parking_board' : 4,
    'Misc/Construction_board' : 4,
    'Misc/Barricade':4,
    'Misc/CrashBarrier' :4,
    # Ignore class
}

CAMERA_INFOS ={
    'CAM0': {'cam2img': [[1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]],
    'lidar2img': [[1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]]},
    'CAM1': {'cam2imFg': [[1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]],
    'lidar2img': [[1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]]},
    'CAM2': {'img_path': 'demo.jpg',
    'height': 427,
    'width': 640,
    'cam2img': [[1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]],
    'lidar2img': [[1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]],
    'lidar2cam': [[1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]]},
    'CAM3': {'cam2img': [[1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]],
    'lidar2img': [[1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]]},
    'R0_rect': [[1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]]
    }

LIDAR_INFOS = {
    'num_pts_feats': 4,
    'lidar_path': [],
    'Tr_velo_to_cam': [[1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]],
    'Tr_imu_to_velo': [[1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]]
 }
def quaternion_to_yaw(qw, qx, qy, qz):
    # Calculate yaw from quaternion
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    # yaw += np.pi / 2
    return yaw

def _calculate_num_points_in_gt(point_root,
                                annos :dict,
                                name,
                                num_features=4,
                                ):
    pc_path = point_root.joinpath(name+'.bin')
    
    points_v = np.fromfile(
        pc_path, dtype=np.float32, count=-1).reshape([-1, num_features])

    bboxes_3d = np.array([ann['bbox_3d'] for ann in annos])
    
    indices = box_np_ops.points_in_rbbox(points_v[:, :3], bboxes_3d)
    num_points_in_gt = indices.sum(0)
    
    for i, ann in enumerate(annos) : 
        ann.update({'num_lidar_pts' : num_points_in_gt[i]})
    return annos


def convert_to_bbox3d_format(data, index):
    """
    Convert a single sample dictionary into the desired format.

    Args:
        data (dict): A dictionary containing bounding box information.
        index (int): The index of the sample (for `index` field in output).
        group_id (int): The group ID for the sample.

    Returns:
        dict: A dictionary in the desired output format.
    """
    # Convert quaternion to yaw
    yaw = quaternion_to_yaw(*data['rotation'])


    # Extract the required fields
    x, y, z = data['coordinate']
    
    dx, dy, dz = data['size']
    # z = z - (dz / 2)
    # Create 3D bounding box array
    bbox_3d = [x, y, z, dy, dx, dz, yaw]
    # Create LiDARInstance3DBoxes object
    bbox_3d = LiDARInstance3DBoxes(torch.tensor([bbox_3d]),
                                   with_yaw=True, origin=(0.5, 0.5, 0.5)
    ).numpy()[0]
    # Get category label
    category = data['category']  # Extract class name from category string
    bbox_label_3d = CLASS_MAPPING.get(category, -1)  # Map category to label index
    if bbox_label_3d == 2 and bbox_3d[3] >= 6:
        bbox_label_3d = 3 
        
    # Compute additional fields
    depth = z  # Typically, depth is the z-coordinate in many systems
    # Compute the bearing angle between the camera and the object
    bearing_angle = np.arctan2( x,y)  # Notice we use (x, z)

    # Calculate alpha (alpha = yaw - bearing_angle)
    alpha = yaw - bearing_angle
    # Normalize alpha to the range [-pi, pi]
    alpha = (alpha - np.pi/2) % (2 * np.pi) - np.pi


    # num_lidar_pts = data['num_lidar_pts'] 
    # num_lidar_pts = _calculate_num_points_in_gt ()
    
    occluded = data['Occlusion']
    truncated = data['Truncation']
    group_id = index
    # Assemble the final dictionary
    result = {
        'bbox': [0.0, 0.0, 0.0, 0.0],  # 2D bounding box set to zero since we're focusing on 3D
        'bbox_label': bbox_label_3d,  # Assume bbox_label is the same as bbox_label_3d
        'bbox_3d': bbox_3d,
        'bbox_label_3d': bbox_label_3d,
        'depth': depth,
        'center_2d': [0.0, 0.0],  # Placeholder for 2D center
        # 'num_lidar_pts': num_lidar_pts,
        'difficulty': 0,  # Placeholder for difficulty
        'truncated': truncated,
        'occluded': occluded,
        'alpha': alpha,
        'score': 0.0,  # Assuming score is 0 as it's not provided
        'index': index,
        'group_id': group_id,
        'gt_ori_labels' : category
    }

    return result

def parse_json_infos(path,point_root,crowd=False):
    with open(path, 'r') as f :
        data = json.load(f)
        f.close()
    
    gt_info = {}
    # Extract Object information to standard format
    objects = []
    for idx, obj in enumerate(data['objects']): # TODO
        if 'crowd' not in obj: # FIXME
            gt_data_info = convert_to_bbox3d_format(obj, idx)
            if gt_data_info['bbox_label_3d']>-1:
                gt_data_info.update({'crowd':0})
                objects.append(gt_data_info)
        elif obj['crowd'] == 1 :
            gt_data_info = convert_to_bbox3d_format(obj, idx)
            if gt_data_info['bbox_label_3d']>-1:
                gt_data_info.update({'crowd':0})
                objects.append(gt_data_info)
        else :
            if crowd :
                gt_data_info = convert_to_bbox3d_format(obj, idx)
                if gt_data_info['bbox_label_3d']>-1:
                    gt_data_info.update({'crowd':1})
                    objects.append(gt_data_info)
    
    
    # update lidar_points    
    if len(objects) >0 :
        objects = _calculate_num_points_in_gt(point_root,
                                            objects,
                                            name=path.stem
                                                )
        
    sample_idx = path.name # split name with '/' 
    sample_idx = sample_idx.replace('.json','.bin')
    # sample_idx = sample_idx.split('.')
    gt_info['instances']= objects
    gt_info['images'] = CAMERA_INFOS
    # LIDAR_INFOS.update({'lidar_path':sample_idx})
    lidar_infos = LIDAR_INFOS.copy()
    lidar_infos['lidar_path'] = sample_idx
    gt_info['lidar_points']=lidar_infos
    gt_info['sample_idx'] = sample_idx
    
    return gt_info

from pathlib import Path
def create_pickle_file(args):
    
    # output_dir = Path(args.output_dir)
    json_path = Path(args.json_path)
    output_dir = Path(args.out_dir)
    
    split_sets = Path(args.split_sets)
    split_sets = list(split_sets.glob('*.txt'))
    
    for split in split_sets : 
        
        with open(split, 'r')  as f :
            names = f.read()
            f.close()
        names = names.split('\n')
        names = [n for n in names if n.endswith('.bin')]
        
        point_root = Path(args.point_root)
        data_list =[]
        for name in tqdm(names) : 
            path = json_path.joinpath(name.replace('.bin', '.json'))
            ann = parse_json_infos(path,point_root, args.crowd_removal)
            # if len(ann['instances']) > 0 :
            data_list.append(ann)
        
        data_infos = {}
        data_infos.update({'metainfo' : META_INFO})
        data_infos.update({'data_list' : data_list})    
        
        output_file = output_dir.joinpath(split.stem + '_data_infos.pkl')    
        with open(output_file, 'wb') as f:
            pickle.dump(data_infos, f)
            f.close()
        saved_name = output_file.resolve().absolute().as_posix()
        print(f"{split.stem} data information saved on {saved_name}")
# j=0        
# for d in data['data_list'] :
#     for i in d['instances']:
#         if i['bbox_3d'][5]<0 :
#             j+=1
                
                
# print(j)           
    
    

def parse_args():
    parser = argparse.ArgumentParser(description="Get directories for training, validation, and data root.")

    parser.add_argument('--out_dir','-o', type=str, required=False, help='Path to the training directory.')
    parser.add_argument('--json_path','-j', type=str, required=False, help='Path to the root data directory.')
    parser.add_argument('--split_sets', '-s', type=str, required=True, help='Path the json root')    
    parser.add_argument('--point_root', '-p', type=str, required=True, help='Path the point cloud root')    
    parser.add_argument('--crowd_removal', '-c', action="store_true", help='Consider crowd lables')    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print("Batch 6 data converting")
    args = parse_args()
    print(args)
    data_infos = create_pickle_file(args)
    
    