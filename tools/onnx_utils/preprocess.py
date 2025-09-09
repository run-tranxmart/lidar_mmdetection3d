from mmcv.utils import ext_loader
from mmdet3d.datasets.transforms.loading import LoadPointsFromFile

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn import functional as F
from typing import Any, List, Optional, Tuple, Union

ext_module = ext_loader.load_ext('_ext', [
    'dynamic_voxelize_forward', 'hard_voxelize_forward',
    'dynamic_point_to_voxel_forward', 'dynamic_point_to_voxel_backward'
])


class _Voxelization(Function):

    @staticmethod
    def forward(
            ctx: Any,
            points: torch.Tensor,
            voxel_size: Union[tuple, float],
            coors_range: Union[tuple, float],
            max_points: int = 35,
            max_voxels: int = 20000,
            deterministic: bool = True) -> Union[Tuple[torch.Tensor], Tuple]:
        """Convert kitti points(N, >=3) to voxels.

        Args:
            points (torch.Tensor): [N, ndim]. Points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity.
            voxel_size (tuple or float): The size of voxel with the shape of
                [3].
            coors_range (tuple or float): The coordinate range of voxel with
                the shape of [6].
            max_points (int, optional): maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize. Default: 35.
            max_voxels (int, optional): maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
                Default: 20000.
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.

        Returns:
            tuple[torch.Tensor]: tuple[torch.Tensor]: A tuple contains three
            elements. The first one is the output voxels with the shape of
            [M, max_points, n_dim], which only contain points and returned
            when max_points != -1. The second is the voxel coordinates with
            shape of [M, 3]. The last is number of point per voxel with the
            shape of [M], which only returned when max_points != -1.
        """
        if max_points == -1 or max_voxels == -1:
            coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
            ext_module.dynamic_voxelize_forward(
                points,
                torch.tensor(voxel_size, dtype=torch.float),
                torch.tensor(coors_range, dtype=torch.float),
                coors,
                NDim=3)
            return coors
        else:
            voxels = points.new_zeros(
                size=(max_voxels, max_points, points.size(1)))
            coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
            num_points_per_voxel = points.new_zeros(
                size=(max_voxels, ), dtype=torch.int)
            voxel_num = torch.zeros(size=(), dtype=torch.long)
            ext_module.hard_voxelize_forward(
                points,
                torch.tensor(voxel_size, dtype=torch.float),
                torch.tensor(coors_range, dtype=torch.float),
                voxels,
                coors,
                num_points_per_voxel,
                voxel_num,
                max_points=max_points,
                max_voxels=max_voxels,
                NDim=3,
                deterministic=deterministic)

            # select the valid voxels
            voxels_out = voxels[:voxel_num]
            coors_out = coors[:voxel_num]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
            return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.apply


class Preprocess:

    def __init__(
        self,
        points_path: str,
        cfg,
        shuffle_points=False,
        cloud_loader=None,) -> None:

        self.points_path = points_path
        
        # Access the test pipeline from the dataset config file
        test_pipeline = cfg.get('test_pipeline', None)
    
        for step in test_pipeline:
            if step['type'] == 'LoadPointsFromFile':
                load_dim = step.get('load_dim', None)
                use_dim = step.get('use_dim', None)
        
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.point_cloud_range = cfg['point_cloud_range']
        voxel_layer_params = cfg['model']['data_preprocessor']['voxel_layer']
        self.max_num_points = voxel_layer_params['max_num_points']
        self.voxel_size = voxel_layer_params['voxel_size']
        self.max_voxels = voxel_layer_params['max_voxels'][1]

        self._with_distance = False
        self._with_cluster_center = True
        self._with_voxel_center = True
        self.shuffle_points = shuffle_points
        self.legacy = True
        self.vx = self.voxel_size[0]
        self.vy = self.voxel_size[1]
        self.vz = self.voxel_size[2]
        self.x_offset = self.vx / 2 + self.point_cloud_range[0]
        self.y_offset = self.vy / 2 + self.point_cloud_range[1]
        self.z_offset = self.vz / 2 + self.point_cloud_range[2]
        # init the cloud loader
        self.points = None
        self.cloud_loader = cloud_loader
        if self.cloud_loader is None:
            self.cloud_loader = LoadPointsFromFile(
                coord_type='LIDAR',
                load_dim=self.load_dim,
                use_dim=self.use_dim)

    def load_points(self):
        self.points = self.cloud_loader.load_points(self.points_path)
        if self.shuffle_points:
            idx = torch.randperm(len(self.points))
            self.points = self.points[idx]

    def filter_range_3d(self):
        in_range_flags = ((self.points[:, 0] > self.point_cloud_range[0])
                          & (self.points[:, 1] > self.point_cloud_range[1])
                          & (self.points[:, 2] > self.point_cloud_range[2])
                          & (self.points[:, 0] < self.point_cloud_range[3])
                          & (self.points[:, 1] < self.point_cloud_range[4])
                          & (self.points[:, 2] < self.point_cloud_range[5]))
        self.points = self.points[in_range_flags]

    def create_voxel_grid(self):
        voxel_size = torch.tensor(self.voxel_size, dtype=torch.float32)
        point_cloud_range = torch.tensor(self.point_cloud_range, dtype=torch.float32)
        grid_shape = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_shape = torch.round(grid_shape).long().tolist()
        self.grid_shape = grid_shape

        points = torch.tensor(self.points, dtype=torch.float32)
        voxels_out, coors_out, num_points_per_voxel_out = voxelization(
            points, voxel_size, point_cloud_range,
            self.max_num_points, self.max_voxels)
        return voxels_out, coors_out, num_points_per_voxel_out

    def get_paddings_indicator(self,
                               actual_num: Tensor,
                               max_num: Tensor,
                               axis: int = 0) -> Tensor:
        """Create boolean mask by actually number of a padded tensor.

        Args:
            actual_num (torch.Tensor): Actual number of points in each voxel.
            max_num (int): Max number of points in each voxel

        Returns:
            torch.Tensor: Mask indicates which points are valid inside a voxel.
        """
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(
            max_num, dtype=torch.int,
            device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]
        return paddings_indicator

    def dist_PCenter(self, feat_inputs: Tensor, num_points: Tensor,
                     coors: Tensor):
        features_ls = [feat_inputs]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = feat_inputs[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(feat_inputs).view(
                    -1, 1, 1)
            f_cluster = feat_inputs[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = feat_inputs.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(feat_inputs[:, :, :3])
                f_center[:, :, 0] = feat_inputs[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = feat_inputs[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = feat_inputs[:, :, 2] - (
                    coors[:, 1].to(dtype).unsqueeze(1) * self.vz +
                    self.z_offset)
            else:
                f_center = feat_inputs[:, :, :3]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(feat_inputs).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(feat_inputs).unsqueeze(1) * self.vy +
                    self.y_offset)
                f_center[:, :, 2] = f_center[:, :, 2] - (
                    coors[:, 1].type_as(feat_inputs).unsqueeze(1) * self.vz +
                    self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(feat_inputs[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        feats = torch.cat(features_ls, dim=-1)
        voxel_count = feats.shape[1]
        mask = self.get_paddings_indicator(num_points, voxel_count)
        mask = torch.unsqueeze(mask, -1).type_as(feats)
        feats *= mask
        feat_output = feats
        return feat_output

    def preprocess(self):
        self.load_points()
        self.filter_range_3d()
        coors, num_points = [], []
        res_voxels, res_coors, res_num_points = self.create_voxel_grid()

        res_coors = F.pad(res_coors, (1, 0), mode='constant', value=0)
        num_points = res_num_points
        coors = res_coors
        feat_pfns_input = self.dist_PCenter(res_voxels, num_points, coors)

        res_dict = {
            "num_points": num_points,
            "pfns_input": feat_pfns_input,
            "coors": coors
        }
        return res_dict
