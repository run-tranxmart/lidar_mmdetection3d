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
from pathlib import Path
from  pypcd import pypcd
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

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



LABEL_CONVERSION = {
    
    # Pedestrian
    'Person': 'Pedestrian',
    
    # Cyclist
    'Bike': 'Cyclist',
    'Motor': 'Cyclist',
    'MotorCyclist': 'Cyclist',  
    'BoxTypeTricycle' : 'Cyclist' , 
    'Trolley' : 'Cyclist',
    
    # Car and Truck
    'Suv': 'Car',
    'Mpv': 'Car',
    'Car': 'Car',
    'OtherCar': 'Car',
    'PickupCar': 'Car',    
    'Truck': 'Car',
    'BigBoxTruck': 'Car',
    'SmallBoxTruck': 'Car',
    'EngineeringVehicle': 'Car',
    'Bus': 'Car',
    
    # Misc
    'CrashBarrier' : 'Misc',
    'crashBarrel' : 'Misc',
    'SafetyBarrier' : 'Misc',
    'Sign' : 'Misc',
    'Hoarding' : 'Misc',
    'TrafficCone' : 'Misc',
    'TriangleRoadblock' : 'Misc',
    'WarningPost' : 'Misc',
    'StopBar' : 'Misc',
    
    'WarningTriangle':'Misc',
    
    # Other classes not mapped will default to -1 later
    'Door' : -1,
    'Ignore' : -1,
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


def _calculate_num_points_in_gt(point_root,
                                annos :dict,
                                name,
                                num_features=4,
                                ):
    pc_path = point_root.joinpath(name)
    # breakpoint()
    points_v = np.fromfile(
        pc_path, dtype=np.float32, count=-1).reshape([-1, num_features])

    bboxes_3d = np.array([ann['bbox_3d'] for ann in annos])
    
    indices = box_np_ops.points_in_rbbox(points_v[:, :3], bboxes_3d)
    num_points_in_gt = indices.sum(0)
    
    filtered_annos = []  # A new list to hold filtered annotations
    for i, ann in enumerate(annos):
        num_points = num_points_in_gt[i]
        if num_points > 0:  # Check if there are LiDAR points
            ann.update({'num_lidar_pts': num_points})
            filtered_annos.append(ann)

    annos = filtered_annos
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
    yaw = data['rotation_y']


    # Extract the required fields
    x, y, z = data['x'], data['y'], data['z']
    # z = z + 0.4
    
    l, w, h = data['length'], data['width'], data['height']
    # Create 3D bounding box array
    bbox_3d = [x, y, z, l, w, h, yaw]
    # Create LiDARInstance3DBoxes object
    bbox_3d = LiDARInstance3DBoxes(torch.tensor([bbox_3d]),
                                   with_yaw=True, origin=(0.5, 0.5, 0)
    )
    # bbox_3d.rotate(1.57)
    bbox_3d = bbox_3d.numpy()[0]
    # Get category label
    category = data['original_lbl']  # Extract class name from category string
    bbox_label_3d = META_INFO['categories'].get(LABEL_CONVERSION.get(category, -1), -1)  # Map category to label index
    if bbox_label_3d == 2 and bbox_3d[3] >= 6:
        bbox_label_3d = 3 
        
    # Compute additional fields
    depth = z  # Typically, depth is the z-coordinate in many systems
    # Compute the bearing angle between the camera and the object
    bearing_angle = np.arctan2(x,y)  # Notice we use (x, z)

    # Calculate alpha (alpha = yaw - bearing_angle)
    alpha = yaw - bearing_angle
    # Normalize alpha to the range [-pi, pi]
    alpha = (alpha - np.pi/2) % (2 * np.pi) - np.pi


    num_lidar_pts = data['num_points'] 
    
    occluded = 0
    truncated = 0
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
        'gt_ori_labels': data['original_lbl']
        
    }
    

    return result

def parse_infos(data):
    
    gt_info = {}
    # Extract Object information to standard format
    objects = []
    # breakpoint()
    for idx, obj in enumerate(data): 
        # 
        gt_data_info = convert_to_bbox3d_format(obj, idx)
        
        if gt_data_info['bbox_label_3d']>-1 :
            if obj['is_crowd'] == False: 
                gt_data_info.update({'crowd':0})
            else : 
                gt_data_info.update({'crowd':1})
            objects.append(gt_data_info)
    
    
       
    return objects
        
    

def process_annotation(ann,annos, args, errors):
    sample_idx = ann + '.bin'
    err_pcd = ann + '.pcd'
    bin_path = Path(args.bin_dir.joinpath(sample_idx))
    
    if bin_path.is_file() and err_pcd not in errors:
        gt_info = {}
        instances = parse_infos(annos[ann])

        # Update lidar points only if there are valid instances.
        if len(instances) > 0:
            instances = _calculate_num_points_in_gt(
                Path(args.bin_dir).resolve(),
                instances,
                name=sample_idx
            )

        gt_info['instances'] = instances
        gt_info['images'] = CAMERA_INFOS

        lidar_infos = LIDAR_INFOS.copy()
        lidar_infos['lidar_path'] = sample_idx
        gt_info['lidar_points'] = lidar_infos
        gt_info['sample_idx'] = sample_idx
        
        if len(gt_info['instances']) > 0 : 
            return gt_info # Return the processed annotation.
        # returnne  
    return None

def create_pickle_file(args, errors=[]):
    base_dir = Path(args.base_dir).name
    annos_path = Path(args.annos_path)
    output_dir = Path(args.out_dir)

    # Load the annotations.
    with open(annos_path, 'rb') as f:
        annos = pickle.load(f)

    data_list = []

    # Use ThreadPoolExecutor for concurrent processing.
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_annotation, ann, annos, args, errors): ann for ann in annos}

        # Collect results as they complete.
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                data_list.append(result)  # Append valid results.

    # Prepare data for saving.
    data_infos = {
        'metainfo': META_INFO,
        'data_list': data_list
    }

    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Save the results as a pickle file.
    output_file = output_dir.joinpath(base_dir + '_data_infos.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(data_infos, f)

    saved_name = output_file.resolve().absolute().as_posix()
    print(f"{base_dir} data information saved on {saved_name}")
    


def process_pcd_file(pcd_name, pcd_dir, bin_dir, history, height_offset, errors):
    if not pcd_name.endswith(".pcd"):
        return f"Frame {pcd_name} is not a .pcd file!"

    try:
        if pcd_name[:-4] in history:
            bin_name = pcd_name.replace(".pcd", ".bin")
            pcd_path = os.path.join(pcd_dir, pcd_name)
            bin_path = os.path.join(bin_dir, bin_name)

            # Load and process the point cloud data.
            pcd_data = pypcd.PointCloud.from_path(pcd_path)
            points = np.zeros([pcd_data.width, 4], dtype=np.float32)
            points[:, 0] = pcd_data.pc_data["y"].copy()
            points[:, 1] = -pcd_data.pc_data["x"].copy()
            points[:, 2] = pcd_data.pc_data["z"].copy()
            points[:, 3] = pcd_data.pc_data["intensity"].copy().astype(np.float32)

            # Apply height offset.
            points[:, 2] += height_offset
            points.tofile(bin_path)
            # breakpoint()
    
    except Exception as e:
        # Collect error information.
        errors.append(pcd_name + '\n')
        return f"Error processing {pcd_name}: {e}"

    return None  # Indicate successful processing

def pcd2bin_multithreaded(pcd_dir, bin_dir, history_path, height_offset, max_threads=8):
    os.makedirs(bin_dir, exist_ok=True)
    errors = []

    # Load history file.
    with open(history_path, 'r') as f:
        history = json.load(f)
    history = history.keys()

    # Get the list of .pcd files.
    pcd_list = sorted(os.listdir(pcd_dir))

    # Use ThreadPoolExecutor for multithreading.
    with ThreadPoolExecutor(max_threads) as executor:
        futures = {
            executor.submit(process_pcd_file, pcd_name, pcd_dir, bin_dir, history, height_offset, errors): pcd_name
            for pcd_name in pcd_list
        }

        # Display progress with tqdm.
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                print(result)  # Print any warnings or errors.

    return errors

def parse_args():
    parser = argparse.ArgumentParser(description="Get directories for training, validation, and data root.")
    
    parser.add_argument('base_dir', type=str, help='Path to the training directory.')

    parser.add_argument('--out_dir','-o',default='pkls/', type=str,  help='Path to the training directory.')
    parser.add_argument('--history_path',default='history_frames.json', type=str,  help='Path to the root data directory.')
    parser.add_argument('--annos_path', '-a',default='annos.pkl', type=str,  help='Path the json root')    
    parser.add_argument('--pcd_dir', '-p',default='pcd_data', type=str, help='Path the point cloud root')    
    parser.add_argument('--bin_dir', '-b',default='cloud_bin', type=str, help='Path the point cloud root')    

    parser.add_argument('--crowd_removal', '-c', action="store_true", help='Consider crowd lables')    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print("Batch 6 data converting")
    args = parse_args()
    print(args)
    base_dir = Path(args.base_dir)
    args.annos_path = base_dir.joinpath(args.annos_path)
    args.out_dir = base_dir.joinpath(args.out_dir)
    args.history_path = base_dir.joinpath(args.history_path)
    args.pcd_dir = base_dir.joinpath(args.pcd_dir)
    args.bin_dir = base_dir.joinpath(args.bin_dir)
    
    
    
    
    errors = pcd2bin_multithreaded(args.pcd_dir,args.bin_dir,args.history_path,0)
    print("errors")
    print(errors)
    data_infos = create_pickle_file(args,errors)
    
    with open(args.base_dir+'/errors.txt', 'w') as f : 
        f.writelines(errors)
    
    