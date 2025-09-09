from imp import init_builtin
from unittest import result
import warnings
import numpy as np
import pdb
import torch
from torch import nn as nn
from torch import Tensor
from typing import List, Dict, Tuple, Union

from mmcv.ops import nms, nms_rotated
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures.ops.iou3d.iou3d_v2 import nms_gpu, nms_normal_gpu

point_cloud = [-16, -16, -4, 80, 16, 3]
voxel_size = [0.2, 0.2, point_cloud[5] - point_cloud[2]]
dx = point_cloud[3] - point_cloud[0]
dy = point_cloud[4] - point_cloud[1]
feat_size = [round(dy / voxel_size[1]), round(dx / voxel_size[0]), 1]

generator_ranges = [
    # Ped: 14=9+4+1
    # 9
    [
        point_cloud[0] + 0. / 6, point_cloud[1] + 0. / 6, -0.6,
        point_cloud[3] - voxel_size[0] + 0. / 6,
        point_cloud[4] - voxel_size[1] + 0. / 6, -0.6
    ],
    [
        point_cloud[0] + 1. / 6, point_cloud[1] + 0. / 6, -0.6,
        point_cloud[3] - voxel_size[0] + 1. / 6,
        point_cloud[4] - voxel_size[1] + 0. / 6, -0.6
    ],
    [
        point_cloud[0] + 2. / 6, point_cloud[1] + 0. / 6, -0.6,
        point_cloud[3] - voxel_size[0] + 2. / 6,
        point_cloud[4] - voxel_size[1] + 0. / 6, -0.6
    ],
    [
        point_cloud[0] + 0. / 6, point_cloud[1] + 1. / 6, -0.6,
        point_cloud[3] - voxel_size[0] + 0. / 6,
        point_cloud[4] - voxel_size[1] + 1. / 6, -0.6
    ],
    [
        point_cloud[0] + 1. / 6, point_cloud[1] + 1. / 6, -0.6,
        point_cloud[3] - voxel_size[0] + 1. / 6,
        point_cloud[4] - voxel_size[1] + 1. / 6, -0.6
    ],
    [
        point_cloud[0] + 2. / 6, point_cloud[1] + 1. / 6, -0.6,
        point_cloud[3] - voxel_size[0] + 2. / 6,
        point_cloud[4] - voxel_size[1] + 1. / 6, -0.6
    ],
    [
        point_cloud[0] + 0. / 6, point_cloud[1] + 2. / 6, -0.6,
        point_cloud[3] - voxel_size[0] + 0. / 6,
        point_cloud[4] - voxel_size[1] + 2. / 6, -0.6
    ],
    [
        point_cloud[0] + 1. / 6, point_cloud[1] + 2. / 6, -0.6,
        point_cloud[3] - voxel_size[0] + 1. / 6,
        point_cloud[4] - voxel_size[1] + 2. / 6, -0.6
    ],
    [
        point_cloud[0] + 2. / 6, point_cloud[1] + 2. / 6, -0.6,
        point_cloud[3] - voxel_size[0] + 2. / 6,
        point_cloud[4] - voxel_size[1] + 2. / 6, -0.6
    ],
    # 4
    [
        point_cloud[0], point_cloud[1], -0.6, point_cloud[3] - voxel_size[0],
        point_cloud[4] - voxel_size[1], -0.6
    ],
    [
        point_cloud[0] + voxel_size[0], point_cloud[1], -0.6, point_cloud[3],
        point_cloud[4] - voxel_size[1], -0.6
    ],
    [
        point_cloud[0], point_cloud[1] + voxel_size[1], -0.6,
        point_cloud[3] - voxel_size[0], point_cloud[4], -0.6
    ],
    [
        point_cloud[0] + voxel_size[0], point_cloud[1] + voxel_size[1], -0.6,
        point_cloud[3], point_cloud[4], -0.6
    ],
    # 1
    [
        point_cloud[0], point_cloud[1], -0.6, point_cloud[3] - voxel_size[0],
        point_cloud[4] - voxel_size[1], -0.6
    ],

    # Cyc: 16=8+8
    [
        point_cloud[0], point_cloud[1], -0.6, point_cloud[3] - voxel_size[0],
        point_cloud[4] - voxel_size[1], -0.6
    ],
    [
        point_cloud[0] + voxel_size[0], point_cloud[1], -0.6, point_cloud[3],
        point_cloud[4] - voxel_size[1], -0.6
    ],
    [
        point_cloud[0], point_cloud[1] + voxel_size[1], -0.6,
        point_cloud[3] - voxel_size[0], point_cloud[4], -0.6
    ],
    [
        point_cloud[0] + voxel_size[0], point_cloud[1] + voxel_size[1], -0.6,
        point_cloud[3], point_cloud[4], -0.6
    ],
    [
        point_cloud[0], point_cloud[1], -0.6, point_cloud[3] - voxel_size[0],
        point_cloud[4] - voxel_size[1], -0.6
    ],
    [
        point_cloud[0] + voxel_size[0], point_cloud[1], -0.6, point_cloud[3],
        point_cloud[4] - voxel_size[1], -0.6
    ],
    [
        point_cloud[0], point_cloud[1] + voxel_size[1], -0.6,
        point_cloud[3] - voxel_size[0], point_cloud[4], -0.6
    ],
    [
        point_cloud[0] + voxel_size[0], point_cloud[1] + voxel_size[1], -0.6,
        point_cloud[3], point_cloud[4], -0.6
    ],
    [
        point_cloud[0], point_cloud[1], -0.6, point_cloud[3] - voxel_size[0],
        point_cloud[4] - voxel_size[1], -0.6
    ],
    [
        point_cloud[0] + voxel_size[0], point_cloud[1], -0.6, point_cloud[3],
        point_cloud[4] - voxel_size[1], -0.6
    ],
    [
        point_cloud[0], point_cloud[1] + voxel_size[1], -0.6,
        point_cloud[3] - voxel_size[0], point_cloud[4], -0.6
    ],
    [
        point_cloud[0] + voxel_size[0], point_cloud[1] + voxel_size[1], -0.6,
        point_cloud[3], point_cloud[4], -0.6
    ],
    [
        point_cloud[0], point_cloud[1], -0.6, point_cloud[3] - voxel_size[0],
        point_cloud[4] - voxel_size[1], -0.6
    ],
    [
        point_cloud[0] + voxel_size[0], point_cloud[1], -0.6, point_cloud[3],
        point_cloud[4] - voxel_size[1], -0.6
    ],
    [
        point_cloud[0], point_cloud[1] + voxel_size[1], -0.6,
        point_cloud[3] - voxel_size[0], point_cloud[4], -0.6
    ],
    [
        point_cloud[0] + voxel_size[0], point_cloud[1] + voxel_size[1], -0.6,
        point_cloud[3], point_cloud[4], -0.6
    ],

    # Car: 6
    [
        point_cloud[0], point_cloud[1], -1.78, point_cloud[3] - voxel_size[0],
        point_cloud[4] - voxel_size[1], -1.78
    ],
    [
        point_cloud[0], point_cloud[1], -1.78, point_cloud[3] - voxel_size[0],
        point_cloud[4] - voxel_size[1], -1.78
    ],
    [
        point_cloud[0], point_cloud[1], -1.78, point_cloud[3] - voxel_size[0],
        point_cloud[4] - voxel_size[1], -1.78
    ],
    [
        point_cloud[0], point_cloud[1], -1.78, point_cloud[3] - voxel_size[0],
        point_cloud[4] - voxel_size[1], -1.78
    ],
    [
        point_cloud[0], point_cloud[1], -1.78, point_cloud[3] - voxel_size[0],
        point_cloud[4] - voxel_size[1], -1.78
    ],
    [
        point_cloud[0], point_cloud[1], -1.78, point_cloud[3] - voxel_size[0],
        point_cloud[4] - voxel_size[1], -1.78
    ]
]
generator_sizes = [
    # [length, width, height, rotation, cls]
    # Pedestrian: 13=9 + 4 + 1
    [0.4, 0.4, 1.58567389, 0, 0],
    [0.4, 0.4, 1.58567389, 0, 0],
    [0.4, 0.4, 1.58567389, 0, 0],
    [0.4, 0.4, 1.58567389, 0, 0],
    [0.4, 0.4, 1.58567389, 0, 0],
    [0.4, 0.4, 1.58567389, 0, 0],
    [0.4, 0.4, 1.58567389, 0, 0],
    [0.4, 0.4, 1.58567389, 0, 0],
    [0.4, 0.4, 1.58567389, 0, 0],
    [0.6, 0.6, 1.58567389, 0, 0],
    [0.6, 0.6, 1.58567389, 0, 0],
    [0.6, 0.6, 1.58567389, 0, 0],
    [0.6, 0.6, 1.58567389, 0, 0],
    [0.8, 0.8, 1.58567389, 0, 0],

    # Cyclist: 8 + 8
    [1.62955035, 0.54944008, 1.19224497, 0, 1],
    [1.62955035, 0.54944008, 1.19224497, 0, 1],
    [1.62955035, 0.54944008, 1.19224497, 0, 1],
    [1.62955035, 0.54944008, 1.19224497, 0, 1],
    [1.97689939, 0.87673796, 1.57529756, 0, 1],
    [1.97689939, 0.87673796, 1.57529756, 0, 1],
    [1.97689939, 0.87673796, 1.57529756, 0, 1],
    [1.97689939, 0.87673796, 1.57529756, 0, 1],
    [1.62955035, 0.54944008, 1.19224497, 1.57, 1],
    [1.62955035, 0.54944008, 1.19224497, 1.57, 1],
    [1.62955035, 0.54944008, 1.19224497, 1.57, 1],
    [1.62955035, 0.54944008, 1.19224497, 1.57, 1],
    [1.97689939, 0.87673796, 1.57529756, 1.57, 1],
    [1.97689939, 0.87673796, 1.57529756, 1.57, 1],
    [1.97689939, 0.87673796, 1.57529756, 1.57, 1],
    [1.97689939, 0.87673796, 1.57529756, 1.57, 1],

    # Car: 6
    [4.36032103, 1.85728288, 1.62119677, 0, 2],
    [4.36032103, 1.85728288, 1.62119677, 1.57, 2],
    [6.98833714, 2.55313427, 2.92700137, 0, 2],
    [6.98833714, 2.55313427, 2.92700137, 1.57, 2],
    [12.78522061, 3.02172211, 3.85993345, 0, 2],
    [12.78522061, 3.02172211, 3.85993345, 1.57, 2]
]


class Postprocess():

    def __init__(self):
        super().__init__()
        # Head params
        self.in_channels = 384
        self.num_classes = 3
        self.feat_channels = 384
        self.diff_rad_by_sin = True
        self.use_direction_classifier = True
        self.assigner_per_size = False
        self.assign_per_class = True
        self.dir_offset = 0.7854
        self.dir_limit_offset = 0
        self.long_branch_num = 1
        self.assign_per_level = True
        self.conv_bias = True
        self.use_sigmoid_cls = True
        # self.nms_score_thr = 0.1
        self.nms_score_thr = 0.5
        # self.score_thr = 0.1
        self.score_thr = 0.5
        self.nms_max_num = 50
        # Generator Params
        self.generator_ranges = generator_ranges,
        self.generator_sizes = generator_sizes,
        self.generator_scales = [1],
        self.generator_featmap_sizes = [[80, 240]]
        self.generator_custom_values = [],
        self.generator_reshape_out = False,
        self.generator_size_per_range = True
        # DeltaXYZWLHRBBoxCoder Params
        self.box_code_size = 7

    def anchors_single_range(self,
                             feature_size,
                             anchor_range,
                             scale=1,
                             sizes=[[1.6, 3.9, 1.56]],
                             device='cpu'):
        """Generate anchors in a single range.
        Args:
            feature_size (list[float] | tuple[float]): Feature map size. It is
                either a list of a tuple of [D, H, W](in order of z, y, and x).
            anchor_range (torch.Tensor | list[float]): Range of anchors with
                shape [6]. The order is consistent with that of anchors, i.e.,
                (x_min, y_min, z_min, x_max, y_max, z_max).
            scale (float | int, optional): The scale factor of anchors.
            sizes (list[list] | np.ndarray | torch.Tensor): Anchor size with
                shape [N, 3], in order of x, y, z.
            rotations (list[float] | np.ndarray | torch.Tensor): Rotations of
                anchors in a single feature grid.
            device (str): Devices that the anchors will be put on.
        Returns:
            torch.Tensor: Anchors with shape \
                [*feature_size, num_sizes, num_rots, 7].
        """
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        z_centers = torch.linspace(
            anchor_range[2], anchor_range[5], feature_size[0], device=device)
        y_centers = torch.linspace(
            anchor_range[1], anchor_range[4], feature_size[1], device=device)
        x_centers = torch.linspace(
            anchor_range[0], anchor_range[3], feature_size[2], device=device)
        # # # Correcting center calculations in Python to match the adjusted C++ logic
        # x_centers = torch.linspace(
        #             anchor_range[0], anchor_range[3], feature_size[2] - 1, device=device)
        # y_centers = torch.linspace(
        #             anchor_range[1], anchor_range[4], feature_size[1] - 1, device=device)


        sizes = torch.tensor(sizes).reshape(-1, 5)  # * scale
        sizes[:, :3] *= scale[0]

        rets = torch.meshgrid(x_centers, y_centers, z_centers)
        rets = list(rets)
        tile_shape = [1] * 4
        tile_shape[-1] = int(sizes.shape[0])

        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-1).repeat(tile_shape).unsqueeze(-1)
        sizes = sizes.reshape([1, 1, 1, -1, 5])
        tile_size_shape = list(rets[0].shape)

        tile_size_shape[3] = 1
        sizes = sizes.repeat(tile_size_shape)

        rets.insert(3, sizes)
        ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4]).unsqueeze(-2)
        # [1, 200, 176, N, 2, 7] for kitti after permute
        if len(self.generator_custom_values) > 0:
            custom_ndim = len(self.generator_custom_values)
            custom = ret.new_zeros([*ret.shape[:-1], custom_ndim])
            # custom[:] = self.custom_values
            ret = torch.cat([ret, custom], dim=-1)
            # [1, 200, 176, N, 2, 9] for nus dataset after permute
        return ret

    def single_level_grid_anchors(self, featmap_size, scale, device='cpu'):
        """Generate grid anchors of a single level feature map.
        This function is usually called by method ``self.grid_anchors``.
        Args:
            featmap_size (tuple[int]): Size of the feature map.
            scale (float): Scale factor of the anchors in the current level.
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.
        Returns:
            torch.Tensor: Anchors in the overall feature map.
        """
        # We reimplement the anchor generator using torch in cuda
        # torch: 0.6975 s for 1000 times
        # numpy: 4.3345 s for 1000 times
        # which is ~5 times faster than the numpy implementation
        if not self.generator_size_per_range:
            return self.anchors_single_range(
                featmap_size,
                self.generator_ranges[0],
                scale,
                self.sizes,
                device=device)

        mr_anchors = []
        for anchor_range, anchor_size in zip(self.generator_ranges[0],
                                             self.generator_sizes[0]):
            mr_anchors.append(
                self.anchors_single_range(
                    featmap_size,
                    anchor_range,
                    scale,
                    anchor_size,
                    device=device))
        mr_anchors = torch.cat(mr_anchors, dim=-3)
        mr_labels = mr_anchors[0, 0, 0, :, 0, -1].long()
        ret = []
        for i in range(max(mr_labels).item() + 1):
            ret.append(mr_anchors[:, :, :, mr_labels == i, :, :-1])
        return ret

    def grid_anchors(self, featmap_sizes, device='cpu'):
        """Generate grid anchors in multiple feature levels.
        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            device (str): Device where the anchors will be put on.
        Returns:
            list[torch.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature lavel, \
                num_base_anchors is the number of anchors for that level.
        """
        num_levels = len(self.generator_scales)
        assert num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.single_level_grid_anchors(
                featmap_sizes[i], self.generator_scales[i], device=device)
            anchors = torch.stack(anchors)
            if self.generator_reshape_out:
                anchors = anchors.reshape(-1, anchors.size(-1))
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def boxes_decode(self, anchors: Tensor, deltas: Tensor) -> Tensor:
        """Apply transformation `deltas` (dx, dy, dz, dx_size, dy_size,
        dz_size, dr, dv*) to `boxes`.

        Args:
            anchors (torch.Tensor): Parameters of anchors with shape (N, 7).
            deltas (torch.Tensor): Encoded boxes with shape
                (N, 7+n) [x, y, z, x_size, y_size, z_size, r, velo*].

        Returns:
            torch.Tensor: Decoded boxes.
        """
        cas, cts = [], []
        box_ndim = anchors.shape[-1]
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(deltas, 1, dim=-1)
        else:
            xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim=-1)

        za = za + ha / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)

    def xywhr2xyxyr(
            self, boxes_xywhr: Union[Tensor,
                                     np.ndarray]) -> Union[Tensor, np.ndarray]:
        """Convert a rotated boxes in XYWHR format to XYXYR format.

        Args:
            boxes_xywhr (Tensor or np.ndarray): Rotated boxes in XYWHR format.

        Returns:
            Tensor or np.ndarray: Converted boxes in XYXYR format.
        """
        boxes = torch.zeros_like(boxes_xywhr)
        half_w = boxes_xywhr[..., 2] / 2
        half_h = boxes_xywhr[..., 3] / 2

        boxes[..., 0] = boxes_xywhr[..., 0] - half_w
        boxes[..., 1] = boxes_xywhr[..., 1] - half_h
        boxes[..., 2] = boxes_xywhr[..., 0] + half_w
        boxes[..., 3] = boxes_xywhr[..., 1] + half_h
        boxes[..., 4] = boxes_xywhr[..., 4]
        return boxes

    def nms_bev(self,
                boxes: Tensor,
                scores: Tensor,
                thresh: float,
                pre_max_size=None,
                post_max_size=None) -> Tensor:
        """NMS function GPU implementation (for BEV boxes). The overlap of two
        boxes for IoU calculation is defined as the exact overlapping area of the
        two boxes. In this function, one can also set ``pre_max_size`` and
        ``post_max_size``.

        Args:
            boxes (Tensor): Input boxes with the shape of [N, 5]
                ([x1, y1, x2, y2, ry]).
            scores (Tensor): Scores of boxes with the shape of [N].
            thresh (float): Overlap threshold of NMS.
            pre_max_size (int, optional): Max size of boxes before NMS.
                Defaults to None.
            post_max_size (int, optional): Max size of boxes after NMS.
                Defaults to None.

        Returns:
            Tensor: Indexes after NMS.
        """
        assert boxes.size(1) == 5, 'Input boxes shape should be [N, 5]'
        order = scores.sort(0, descending=True)[1]
        if pre_max_size is not None:
            order = order[:pre_max_size]
        boxes = boxes[order].contiguous()
        scores = scores[order]

        # xyxyr -> back to xywhr
        # note: better skip this step before nms_bev call in the future
        boxes = torch.stack(
            ((boxes[:, 0] + boxes[:, 2]) / 2,
             (boxes[:, 1] + boxes[:, 3]) / 2, boxes[:, 2] - boxes[:, 0],
             boxes[:, 3] - boxes[:, 1], boxes[:, 4]),
            dim=-1)

        keep = nms_rotated(boxes, scores, thresh)[1]
        keep = order[keep]
        if post_max_size is not None:
            keep = keep[:post_max_size]
        return keep
    

    def box3d_multiclass_nms(self,
                             mlvl_bboxes,
                             mlvl_bboxes_for_nms,
                             mlvl_scores,
                             score_thr,
                             max_num,
                             cfg=None,
                             mlvl_dir_scores=None,
                             mlvl_attr_scores=None,
                             mlvl_bboxes2d=None,
                             mlvl_dir_preds=None,
                             mlvl_anchors=None,
                             mlvl_features=None):
        """Multi-class nms for 3D boxes.

        Args:
            mlvl_bboxes (torch.Tensor): Multi-level boxes with shape (N, M).
                M is the dimensions of boxes.
            mlvl_bboxes_for_nms (torch.Tensor): Multi-level boxes with shape
                (N, 5) ([x1, y1, x2, y2, ry]). N is the number of boxes.
            mlvl_scores (torch.Tensor): Multi-level boxes with shape
                (N, C + 1). N is the number of boxes. C is the number of classes.
            score_thr (float): Score thredhold to filter boxes with low
                confidence.
            max_num (int): Maximum number of boxes will be kept.
            cfg (dict): Configuration dict of NMS.
            mlvl_dir_scores (torch.Tensor, optional): Multi-level scores
                of direction classifier. Defaults to None.
            mlvl_attr_scores (torch.Tensor, optional): Multi-level scores
                of attribute classifier. Defaults to None.
            mlvl_bboxes2d (torch.Tensor, optional): Multi-level 2D bounding
                boxes. Defaults to None.

        Returns:
            tuple[torch.Tensor]: Return results after nms, including 3D \
                bounding boxes, scores, labels, direction scores, attribute \
                scores (optional) and 2D bounding boxes (optional).
        """
        # do multi class nms
        # the fg class id range: [0, num_classes-1]
        num_classes = mlvl_scores.shape[1] - 1
        bboxes = []
        scores = []
        labels = []
        dir_scores = []
        dir_preds = []
        anchors = []
        features = []
        attr_scores = []
        bboxes2d = []
        for i in range(num_classes):
            cls_inds = mlvl_scores[:, i] > score_thr
            if not cls_inds.any():
                continue

            _scores = mlvl_scores[cls_inds, i]
            _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]

            # Perform NMS and get the indices of the selected boxes
            outputs = nms_gpu(_bboxes_for_nms, _scores, self.nms_score_thr)
            selected = outputs[1] if isinstance(outputs, tuple) else outputs
            # selected = selected.long()

            # Ensure the indices and bounding boxes are on the same device
            if _bboxes_for_nms.is_cuda:
                selected = selected.cuda()
            else:
                selected = selected.cpu()

            _mlvl_bboxes = mlvl_bboxes[cls_inds, :]
            bboxes.append(_mlvl_bboxes[selected])
            scores.append(_scores[selected])
            cls_label = mlvl_bboxes.new_full((len(selected),), i, dtype=torch.long)
            labels.append(cls_label)
            

            if mlvl_dir_scores is not None:
                _mlvl_dir_scores = mlvl_dir_scores[cls_inds]
                dir_scores.append(_mlvl_dir_scores[selected])
            if mlvl_dir_preds is not None:
                _mlvl_dir_preds = mlvl_dir_preds[cls_inds]
                dir_preds.append(_mlvl_dir_preds[selected])
            if mlvl_anchors is not None:
                _mlvl_anchors = mlvl_anchors[cls_inds]
                anchors.append(_mlvl_anchors[selected])
            if mlvl_features is not None:
                _mlvl_features = mlvl_features[cls_inds]
                features.append(_mlvl_features[selected])
            if mlvl_attr_scores is not None:
                _mlvl_attr_scores = mlvl_attr_scores[cls_inds]
                attr_scores.append(_mlvl_attr_scores[selected])
            if mlvl_bboxes2d is not None:
                _mlvl_bboxes2d = mlvl_bboxes2d[cls_inds]
                bboxes2d.append(_mlvl_bboxes2d[selected])

        if bboxes:
            bboxes = torch.cat(bboxes, dim=0)
            scores = torch.cat(scores, dim=0)
            labels = torch.cat(labels, dim=0)
            if mlvl_dir_scores is not None:
                dir_scores = torch.cat(dir_scores, dim=0)
            if mlvl_dir_preds is not None:
                dir_preds = torch.cat(dir_preds, dim=0)
            if mlvl_anchors is not None:
                anchors = torch.cat(anchors, dim=0)
            if mlvl_features is not None:
                features = torch.cat(features, dim=0)
            if mlvl_attr_scores is not None:
                attr_scores = torch.cat(attr_scores, dim=0)
            if mlvl_bboxes2d is not None:
                bboxes2d = torch.cat(bboxes2d, dim=0)
            if bboxes.shape[0] > max_num:
                _, inds = scores.sort(descending=True)
                inds = inds[:max_num]
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                scores = scores[inds]
                if mlvl_dir_scores is not None:
                    dir_scores = dir_scores[inds]
                if mlvl_dir_preds is not None:
                    dir_preds = dir_preds[inds]
                if mlvl_attr_scores is not None:
                    attr_scores = attr_scores[inds]
                if mlvl_bboxes2d is not None:
                    bboxes2d = bboxes2d[inds]
                if mlvl_features is not None:
                    features = features[inds]
                if mlvl_anchors is not None:
                    anchors = mlvl_anchors[inds]
        else:
            bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)))
            scores = mlvl_scores.new_zeros((0, ))
            labels = mlvl_scores.new_zeros((0, ), dtype=torch.long)
            if mlvl_dir_scores is not None:
                dir_scores = mlvl_scores.new_zeros((0, ))
            if mlvl_dir_preds is not None:
                dir_preds = mlvl_scores.new_zeros((0, 2))
            if mlvl_attr_scores is not None:
                attr_scores = mlvl_scores.new_zeros((0, ))
            if mlvl_bboxes2d is not None:
                bboxes2d = mlvl_scores.new_zeros((0, 4))
            if mlvl_features is not None:
                features = mlvl_scores.new_zeros((0, mlvl_features.size(-1)))
            if mlvl_anchors is not None:
                anchors = mlvl_anchors.new_zeros((0, mlvl_anchors.size(-1)))

        results = (bboxes, scores, labels)

        if mlvl_dir_scores is not None:
            results = results + (dir_scores, )
        if mlvl_attr_scores is not None:
            results = results + (attr_scores, )
        if mlvl_bboxes2d is not None:
            results = results + (bboxes2d, )
        if mlvl_dir_preds is not None:
            results = results + (dir_preds, )
        if mlvl_features is not None:
            results = results + (features, )
        if mlvl_anchors is not None:
            results = results + (anchors, )

        return results

    def limit_period(self,
                     val: Union[np.ndarray, Tensor],
                     offset: float = 0.5,
                     period: float = np.pi) -> Union[np.ndarray, Tensor]:
        """Limit the value into a period for periodic function.

        Args:
            val (np.ndarray or Tensor): The value to be converted.
            offset (float): Offset to set the value range. Defaults to 0.5.
            period (float): Period of the value. Defaults to np.pi.

        Returns:
            np.ndarray or Tensor: Value in the range of
            [-offset * period, (1-offset) * period].
        """
        limited_val = val - torch.floor(val / period + offset) * period
        return limited_val

    def filter_results(self, src_instances: Dict):
        valid_inds = []
        bboxes = src_instances['bboxes_3d']
        scores = src_instances['scores_3d']
        labels = src_instances['labels_3d']
        dirs = src_instances['dir_3d']

        if len(bboxes) == 0:
            return src_instances

        bboxes_3d = LiDARInstance3DBoxes(bboxes)
        limit_range = bboxes_3d.tensor.new_tensor(point_cloud) * 1.05
        valid_inds = ((bboxes_3d.center > limit_range[:3]) &
                      (bboxes_3d.center < limit_range[3:]))
        valid_inds = valid_inds.all(-1)
        valid_bboxes = bboxes[valid_inds]
        valid_scores = scores[valid_inds]
        valid_labels = labels[valid_inds]
        valid_dirs = dirs[valid_inds]

        des_instances = {
            'bboxes_3d': valid_bboxes,
            'scores_3d': valid_scores,
            'labels_3d': valid_labels,
            'dir_3d': valid_dirs
        }
        return des_instances

    def predict_by_feat(
        self,
        cls_scores,
        bbox_preds,
        dir_preds,
        ind_topk,
    ):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): Whether th rescale bbox.

        Returns:
            list[tuple]: Prediction resultes of batches.
        """

        # device = cls_scores[0]

        # anchors & bboxes
        mlvl_anchors = self.grid_anchors(
            self.generator_featmap_sizes, device='cpu')
        if isinstance(mlvl_anchors[0], list):
            mlvl_anchors = [
                torch.cat(anchor, dim=-3).reshape(-1, self.box_code_size + 1)
                for anchor in mlvl_anchors
            ]
        else:
            mlvl_anchors = [
                anchor.reshape(-1, self.box_code_size + 1)
                for anchor in mlvl_anchors
            ]
        mlvl_anchors = mlvl_anchors[0]
        anchors_topk = mlvl_anchors[ind_topk, :]
        bboxes = self.boxes_decode(anchors_topk, bbox_preds)
        bboxes_for_nms = self.xywhr2xyxyr(bboxes[:, [0, 1, 3, 4, 6]])
        bboxes_for_nms[..., 4] += np.pi / 2

        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = cls_scores.new_zeros(cls_scores.shape[0], 1)
            cls_scores = torch.cat([cls_scores, padding], dim=1)

        score_thr = self.score_thr 

        dir_scores = torch.max(dir_preds, dim=-1)[1]
        results = self.box3d_multiclass_nms(
            bboxes,
            bboxes_for_nms,
            cls_scores,
            score_thr,
            self.nms_max_num,
            mlvl_dir_scores=dir_scores,
            mlvl_dir_preds=dir_preds)
        # the following code use the box3d_multiclass_nms from mmdet3d
        '''
        from mmdet3d.models.layers import box3d_multiclass_nms
        cfg = Config(cfg_dict={'use_rotate_nms': True, 'nms_thr': 0.3})
        bboxes = bboxes.cuda()
        bboxes_for_nms = bboxes_for_nms.cuda()
        cls_scores = cls_scores.cuda()
        dir_scores = dir_scores.cuda()
        dir_preds = dir_preds.cuda()
        results = box3d_multiclass_nms(
            bboxes,
            bboxes_for_nms,
            cls_scores,
            score_thr,
            self.nms_max_num,
            cfg,
            mlvl_dir_scores=dir_scores,
            mlvl_dir_preds=dir_preds)
        '''

        bboxes, scores, labels, dir_scores, dir_preds = results

        # Recover correct value
        if bboxes.shape[0] > 0:
            dir_rot = self.limit_period(bboxes[..., 6] - self.dir_offset,
                                        self.dir_limit_offset, np.pi)
            bboxes[..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * dir_scores.to(bboxes.dtype))

        result_dict = {
            'bboxes_3d': bboxes,
            'scores_3d': scores,
            'labels_3d': labels,
            'dir_3d': dir_preds,
        }
        filtered_dict = self.filter_results(result_dict)
        final_results = {
            'bboxes_3d': filtered_dict['bboxes_3d'].cpu().numpy().tolist(),
            'scores_3d': filtered_dict['scores_3d'].cpu().numpy().tolist(),
            'labels_3d': filtered_dict['labels_3d'].cpu().numpy().tolist(),
            'dir': filtered_dict['dir_3d'].cpu().numpy().tolist()
        }
        return final_results
