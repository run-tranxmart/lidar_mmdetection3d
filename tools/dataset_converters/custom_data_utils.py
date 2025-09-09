import argparse
from collections import OrderedDict
from concurrent import futures as futures
import json
import numpy as np
from pathlib import Path
import pickle
from PIL import Image
import os
from os import path as osp
from skimage import io

from tqdm import tqdm
import warnings
import mmengine


def _get_json_anno(json_path):
    obj_cls_dict = {
        'vehicle': 'Car',
        'pedestrian': 'Pedestrian',
        'rider': 'Cyclist',
        'bicycle': 'Cyclist',
        'tricycle': 'Cyclist',
        'wheelchair': 'Cyclist',
        'stroller': 'Cyclist',
        'service vehicle': 'Cyclist',
    }
    ignore_cls_set = {'unknown', 'traffic_facility', 'bicycles', 'animal'}
    annos = {}
    annos.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'score': []
    })
    with open(json_path, 'r') as f:
        json_label = json.load(f)
        json_cloud = json_label['pcd_file']
        cloud_name = json_cloud.rsplit('/')[-1]
        frame_name = cloud_name.rsplit('.')[0]
        if frame_name not in json_path:
            print(f'Frame frame {frame_name} not in json path {json_path}')
        json_objs = json_label['items']
        for obj in json_objs:
            obj_category = obj['category']
            obj_subcategory = obj['subcategory']
            obj_name = None
            if obj_category in obj_cls_dict:
                obj_name = obj_cls_dict[obj_category]
            elif obj_subcategory in obj_cls_dict:
                obj_name = obj_cls_dict[obj_subcategory]
            elif (obj_category in ignore_cls_set) or \
                    (obj_subcategory in ignore_cls_set):
                continue
            if obj_name is None:
                print(f"{obj_category} - {obj_subcategory} not in dict")
                continue
            obj_bbox3d = obj['bbox3d']
            dimensions = [
                obj_bbox3d['height'],
                obj_bbox3d['length'],
                obj_bbox3d['width'],
            ]
            location = [
                obj_bbox3d['position_y'],
                -obj_bbox3d['position_x'],
                obj_bbox3d['position_z'] + 0.8 - obj_bbox3d['height'] / 2.0,
            ]
            rotation_y = float(obj_bbox3d['yaw']) + np.pi / 2.0 + np.pi
            alpha = -np.arctan2(-location[1], location[0]) + rotation_y
            box2d = [0.0, 0.0, 0.0, 0.0]

            annos['name'].append(obj_name)
            annos['truncated'].append(0.0)
            annos['occluded'].append(0.0)
            annos['alpha'].append(alpha)
            annos['bbox'].append(box2d)
            annos['dimensions'].append(dimensions)
            annos['location'].append(location)
            annos['rotation_y'].append(rotation_y)
            annos['score'].append(0.0)

    num_gt = len(annos['name'])
    index = list(range(num_gt))
    annos['name'] = np.array(annos['name'])
    annos['truncated'] = np.array(annos['truncated'])
    annos['occluded'] = np.array(annos['occluded'])
    annos['alpha'] = np.array(annos['alpha']).reshape(-1)
    annos['bbox'] = np.array(annos['bbox']).reshape(-1, 4)
    annos['dimensions'] = np.array(annos['dimensions']).reshape(-1, 3)
    annos['location'] = np.array(annos['location']).reshape(-1, 3)
    annos['rotation_y'] = np.array(annos['rotation_y']).reshape(-1)
    annos['score'] = np.array(annos['score'])
    annos['index'] = np.array(index, dtype=np.int32)
    annos['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annos


def _get_json_anno_v2(json_path):
    obj_cls_dict = {
        'vehicle': 'Car',
        'pedestrian': 'Pedestrian',
        'rider': 'Cyclist',
        'bicycle': 'Cyclist',
        'tricycle': 'Cyclist',
        'wheelchair': 'Cyclist',
        'stroller': 'Cyclist',
        'service vehicle': 'Cyclist',
    }
    ignore_cls_set = {'unknown', 'traffic_facility', 'bicycles', 'animal'}
    annos = {}
    annos.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'score': []
    })
    with open(json_path, 'r') as f:
        json_label = json.load(f)
        json_cloud = json_label['pcd_file']
        cloud_name = json_cloud.rsplit('/')[-1]
        frame_name = cloud_name.rsplit('.')[0]
        if frame_name not in json_path:
            print(f'Frame frame {frame_name} not in json path {json_path}')
        json_objs = json_label['items']
        for obj in json_objs:
            obj_category = obj['category']
            obj_subcategory = obj['subcategory']
            obj_name = None
            if obj_category in obj_cls_dict:
                obj_name = obj_cls_dict[obj_category]
            elif obj_subcategory in obj_cls_dict:
                obj_name = obj_cls_dict[obj_subcategory]
            elif (obj_category in ignore_cls_set) or \
                    (obj_subcategory in ignore_cls_set):
                continue
            if obj_name is None:
                print(f"{obj_category} - {obj_subcategory} not in dict")
                continue
            obj_bbox3d = obj['bbox3d']
            dimensions = [
                obj_bbox3d['height'],
                obj_bbox3d['length'],
                obj_bbox3d['width'],
            ]
            location = [
                obj_bbox3d['position_x'],
                obj_bbox3d['position_y'],
                obj_bbox3d['position_z'] - obj_bbox3d['height'] / 2.0,
            ]
            rotation_y = float(obj_bbox3d['yaw'])
            alpha = -np.arctan2(-location[1], location[0]) + rotation_y
            box2d = [0.0, 0.0, 0.0, 0.0]

            annos['name'].append(obj_name)
            annos['truncated'].append(0.0)
            annos['occluded'].append(0.0)
            annos['alpha'].append(alpha)
            annos['bbox'].append(box2d)
            annos['dimensions'].append(dimensions)
            annos['location'].append(location)
            annos['rotation_y'].append(rotation_y)
            annos['score'].append(0.0)

    num_gt = len(annos['name'])
    index = list(range(num_gt))
    annos['name'] = np.array(annos['name'])
    annos['truncated'] = np.array(annos['truncated'])
    annos['occluded'] = np.array(annos['occluded'])
    annos['alpha'] = np.array(annos['alpha']).reshape(-1)
    annos['bbox'] = np.array(annos['bbox']).reshape(-1, 4)
    annos['dimensions'] = np.array(annos['dimensions']).reshape(-1, 3)
    annos['location'] = np.array(annos['location']).reshape(-1, 3)
    annos['rotation_y'] = np.array(annos['rotation_y']).reshape(-1)
    annos['score'] = np.array(annos['score'])
    annos['index'] = np.array(index, dtype=np.int32)
    annos['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annos


def _get_label_anno(label_path):
    annos = {}
    annos.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'score': []
    })
    with open(label_path, 'r') as f:
        content = f.readlines()
        for line in content:
            instance_dict = {"bbox_3d": None, "bbox_label_3d": None}
            line_array = line.strip().split()
            if len(line_array) == 11:
                # Txt: l, h, w, x, y, z, r
                # Pkl: x, y, z, w, l, h, r
                obj_name = line_array[0]
                # dimensions = [float(line_array[4]), float(line_array[5]), float(line_array[6])]
                dimensions = [
                    float(line_array[6]),
                    float(line_array[4]),
                    float(line_array[5])
                ]
                location = [
                    float(line_array[7]),
                    float(line_array[8]),
                    float(line_array[9])
                ]
                rotation_y = float(line_array[10])
                alpha = -np.arctan2(-location[1], location[0]) + rotation_y
                box2d = [0.0, 0.0, 0.0, 0.0]
            elif len(line_array) == 15:
                # Txt: h, w, l, x, y, z, r
                obj_name = line_array[0]
                dimensions = [
                    float(line_array[8]),
                    float(line_array[10]),
                    float(line_array[9])
                ]
                location = [
                    float(line_array[11]),
                    float(line_array[12]),
                    float(line_array[13])
                ]
                rotation_y = -float(line_array[14]) - np.pi / 2.0
                alpha = -np.arctan2(-location[1], location[0]) + rotation_y
                box2d = [
                    float(line_array[4]),
                    float(line_array[5]),
                    float(line_array[6]),
                    float(line_array[7])
                ]
            if obj_name == 'Truck':
                obj_name = 'Car'
            annos['name'].append(obj_name)
            annos['truncated'].append(0.0)
            annos['occluded'].append(0.0)
            annos['alpha'].append(alpha)
            annos['bbox'].append(box2d)
            annos['dimensions'].append(dimensions)
            annos['location'].append(location)
            annos['rotation_y'].append(rotation_y)
            annos['score'].append(0.0)
    num_gt = len(annos['name'])
    index = list(range(num_gt))
    annos['name'] = np.array(annos['name'])
    annos['truncated'] = np.array(annos['truncated'])
    annos['occluded'] = np.array(annos['occluded'])
    annos['alpha'] = np.array(annos['alpha']).reshape(-1)
    annos['bbox'] = np.array(annos['bbox']).reshape(-1, 4)
    annos['dimensions'] = np.array(annos['dimensions']).reshape(-1, 3)
    annos['location'] = np.array(annos['location']).reshape(-1, 3)
    annos['rotation_y'] = np.array(annos['rotation_y']).reshape(-1)
    annos['score'] = np.array(annos['score'])
    annos['index'] = np.array(index, dtype=np.int32)
    annos['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annos


def get_custom_frame_info(path,
                          frame_list,
                          label_info=True,
                          num_worker=8,
                          relative_path=True):
    root_path = Path(path)
    if not isinstance(frame_list, list):
        raise TypeError('Invalid frame list')

    def map_func(idx):
        info = {}
        pc_info = {'num_features': 4}
        image_info = {'image_idx': idx}
        frame_name = idx
        if ('.bin' in frame_name) or ('.txt' in frame_name):
            frame_name = frame_name.rsplit('.', 1)[0]

        annotations = None
        if relative_path:
            pc_info['velodyne_path'] = osp.join('cloud_bin',
                                                frame_name + '.bin')
            image_info['image_path'] = osp.join('image', 'demo.jpg')
        else:
            pc_info['velodyne_path'] = osp.join(root_path, 'cloud_bin',
                                                frame_name + '.bin')
            image_info['image_path'] = osp.join(root_path, 'image', 'demo.jpg')

        image_info['image_shape'] = np.array([427, 640], dtype=np.int32)
        if label_info:
            json_path = osp.join(root_path, 'raw_label', frame_name + '.json')
            txt_path = osp.join(root_path, 'label_txt', frame_name + '.txt')
            if osp.exists(json_path):
                # annotations = _get_json_anno(json_path)
                annotations = _get_json_anno_v2(json_path)
            elif osp.exists(txt_path):
                annotations = _get_label_anno(txt_path)
            else:
                raise FileNotFoundError('Cannot find label file: ' +
                                        json_path + ' or ' + txt_path)
        info['image'] = image_info
        info['point_cloud'] = pc_info
        if annotations is not None:
            info['annos'] = annotations
            info['annos']['difficulty'] = np.zeros_like(
                info['annos']['truncated']).astype(np.int32)
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        frame_infos = executor.map(map_func, frame_list)

    return list(frame_infos)
