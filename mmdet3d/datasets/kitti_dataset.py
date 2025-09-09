# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Union

import numpy as np
from mmengine.fileio import join_path, list_from_file, load
from mmengine.utils import is_abs
from mmdet3d.registry import DATASETS
from mmdet3d.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class KittiDataset(Det3DDataset):
    r"""KITTI Dataset.

    This class serves as the API for experiments on the `KITTI Dataset
    <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_lidar=True).
        default_cam_key (str): The default camera name adopted.
            Defaults to 'CAM2'.
        load_type (str): Type of loading mode. Defaults to 'frame_based'.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
              to convert to the FOV-based data type to support image-based
              detector.
            - 'fov_image_based': Only load the instances inside the default
              cam, and need to convert to the FOV-based data type to support
              image-based detector.
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes.
            Defaults to [0, -40, -3, 70.4, 40, 0.0].
    """
    # TODO: use full classes of kitti
    METAINFO = {
        'classes': ('Pedestrian', 'Cyclist', 'Car', 'Truck', 'Misc'),
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42),
                    (197, 226, 255), (255, 77, 255)]  # Adjust colors as needed
    }

    def __init__(self,
                 data_root: str,
                 ann_file: Union[str, List],
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True),
                 default_cam_key: str = 'CAM2',
                 load_type: str = 'frame_based',
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 convert_cam_to_lidar: bool = True,
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
                 **kwargs) -> None:

        self.pcd_limit_range = pcd_limit_range
        self.convert_cam_to_lidar = convert_cam_to_lidar
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')
        self.load_type = load_type
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert self.modality is not None
        assert box_type_3d.lower() in ('lidar', 'camera')

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.modality['use_lidar']:
            if 'plane' in info:
                # convert ground plane to velodyne coordinates
                plane = np.array(info['plane'])
                lidar2cam = np.array(
                    info['images']['CAM2']['lidar2cam'], dtype=np.float32)
                reverse = np.linalg.inv(lidar2cam)

                (plane_norm_cam, plane_off_cam) = (plane[:3],
                                                   -plane[:3] * plane[3])
                plane_norm_lidar = \
                    (reverse[:3, :3] @ plane_norm_cam[:, None])[:, 0]
                plane_off_lidar = (
                    reverse[:3, :3] @ plane_off_cam[:, None][:, 0] +
                    reverse[:3, 3])
                plane_lidar = np.zeros_like(plane_norm_lidar, shape=(4, ))
                plane_lidar[:3] = plane_norm_lidar
                plane_lidar[3] = -plane_norm_lidar.T @ plane_off_lidar
            else:
                plane_lidar = None

            info['plane'] = plane_lidar

        if self.load_type == 'fov_image_based' and self.load_eval_anns:
            info['instances'] = info['cam_instances'][self.default_cam_key]

        info = super().parse_data_info(info)

        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - bbox_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                  0, 1, 2 represent xxxxx respectively.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

            if self.load_type in ['fov_image_based', 'mv_image_based']:
                ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
                ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
                ann_info['depths'] = np.zeros((0), dtype=np.float32)

        ann_info = self._remove_dontcare(ann_info)
        if self.convert_cam_to_lidar:
            # in kitti, lidar2cam = R0_rect @ Tr_velo_to_cam
            lidar2cam = np.array(info['images']['CAM2']['lidar2cam'])
            # convert gt_bboxes_3d to velodyne coordinates with `lidar2cam`
            gt_bboxes_3d = CameraInstance3DBoxes(
                ann_info['gt_bboxes_3d']).convert_to(self.box_mode_3d,
                                                     np.linalg.inv(lidar2cam))
            ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        else:
            # directly init the gt_bboxes_3d
            # l, h, w -> l, w, h
            # ann_info['gt_bboxes_3d'][:, [3, 4, 5]] = ann_info['gt_bboxes_3d'][:, [3, 5, 4]]
            gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
            ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info

    def _join_prefix(self):
        """Join ``self.data_root`` with ``self.data_prefix`` and
        ``self.ann_file``.

        Examples:
            >>> # self.data_prefix contains relative paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='a/b/c/d/e')
            >>> self.ann_file
            'a/b/c/f'
            >>> # self.data_prefix contains absolute paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='/d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='/d/e')
            >>> self.ann_file
            'a/b/c/f'
        """
        # Automatically join annotation file path with `self.root` if
        # `self.ann_file` is not an absolute path.
        if self.ann_file and self.data_root:
            if isinstance(self.ann_file, str):
                self.ann_file = [self.ann_file]
            for i, ann_file in enumerate(self.ann_file):
                if not is_abs(ann_file):
                    self.ann_file[i] = join_path(self.data_root, ann_file)

        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if not isinstance(prefix, str):
                raise TypeError('prefix should be a string, but got '
                                f'{type(prefix)}')
            if not is_abs(prefix) and self.data_root:
                self.data_prefix[data_key] = join_path(self.data_root, prefix)
            else:
                self.data_prefix[data_key] = prefix

    def load_data_list(self) -> List[dict]:
        '''Overload the loading functions from an annotation file or list'''
        annotations = {
            'metainfo': [],
            'data_list': [],
        }
        if isinstance(self.ann_file, str):
            annotations = load(self.ann_file)
        elif isinstance(self.ann_file, List):
            for anno_pth in self.ann_file:
                single_annotations = load(anno_pth)
                src_frame = len(annotations['data_list'])
                annotations['metainfo'] = single_annotations['metainfo']
                annotations['data_list'].extend(
                    single_annotations['data_list'])
                des_frame = len(annotations['data_list'])
                print(f'Frame {src_frame} to {des_frame}')

        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                             'keys')
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For image tasks, `data_info` should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_list.append(data_info)
            elif isinstance(data_info, list):
                # For video tasks, `data_info` could contain image
                # information of multiple frames, such as
                # [dict(video_path='xxx', timestamps=...),
                #  dict(video_path='xxx', timestamps=...)]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')
        return data_list


@DATASETS.register_module()
class Kitti_4Classes_Dataset(KittiDataset):
    # Customize the METAINFO for your 4 classes
    METAINFO = {
        'classes': ('Pedestrian', 'Cyclist', 'Car', 'Truck'),
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42),
                    (197, 226, 255)]  # Adjust colors as needed
    }
    
@DATASETS.register_module()
class Kitti_5Class_Dataset(KittiDataset):
    # Customize the METAINFO for your 5 classes
    METAINFO = {
        'classes': ('Pedestrian', 'Cyclist', 'Car', 'Truck', 'Misc'),
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42),
                    (197, 226, 255), (255, 77, 255)]  # Adjust colors as needed
    }
    
@DATASETS.register_module()
class Kitti_6Class_Dataset(KittiDataset):
    # Customize the METAINFO for your 5 classes
    METAINFO = {
        'classes': ('Pedestrian', 'Cyclist', 'Car', 'Truck', 'Cone', 'Barricade'),
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42),
                    (197, 226, 255), (255, 77, 255), (0, 0, 142)]  # Adjust colors as needed
    }