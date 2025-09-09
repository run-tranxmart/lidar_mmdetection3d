# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import os.path as osp
import pdb

import pickle as pkl
import torch
from torch import nn as nn
from torch import Tensor
from typing import List, Tuple

from mmengine.structures import InstanceData
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models.utils import multi_apply
from mmdet3d.models.task_modules.samplers import PseudoSampler
from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.structures import (limit_period, xywhr2xyxyr)
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.utils.typing_utils import (ConfigType, InstanceList,
                                        OptMultiConfig, OptInstanceList)

from .base_3d_dense_head import Base3DDenseHead
from .train_mixins import AnchorTrainMixin
# from mmcv.runner import BaseModule, force_fp32
# from mmengine.runner import amp
# from mmdet.utils.memory import cast_tensor_type
# from mmdet.core import (build_assigner, build_bbox_coder, build_prior_generator, build_sampler, multi_apply)
# from mmdet3d.models.test_time_augs import merge_aug_bboxes_3d
# from mmdet.models import HEADS
# from ..builder import build_loss


@MODELS.register_module()
class Anchor3DHead(Base3DDenseHead, AnchorTrainMixin):
    """Anchor head for SECOND/PointPillars/MVXNet/PartA2.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        train_cfg (dict): Train configs.
        test_cfg (dict): Test configs.
        feat_channels (int): Number of channels of the feature map.
        use_direction_classifier (bool): Whether to add a direction classifier.
        anchor_generator(dict): Config dict of anchor generator.
        assigner_per_size (bool): Whether to do assignment for each separate
            anchor size.
        assign_per_class (bool): Whether to do assignment for each class.
        diff_rad_by_sin (bool): Whether to change the difference into sin
            difference for box regression loss.
        dir_offset (float | int): The offset of BEV rotation angles.
            (TODO: may be moved into box coder)
        dir_limit_offset (float | int): The limited range of BEV
            rotation angles. (TODO: may be moved into box coder)
        bbox_coder (dict): Config dict of box coders.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_dir (dict): Config of direction classifier loss.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 train_cfg,
                 test_cfg,
                 feat_channels=256,
                 use_direction_classifier=True,
                 anchor_generator=dict(
                     type='Anchor3DRangeGenerator',
                     range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
                     strides=[2],
                     sizes=[[1.6, 3.9, 1.56]],
                     rotations=[0, 1.57],
                     custom_values=[],
                     reshape_out=False),
                 assigner_per_size=False,
                 assign_per_class=False,
                 assign_per_level=True,
                 diff_rad_by_sin=True,
                 dir_offset=0,
                 dir_limit_offset=1,
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_dir=dict(type='CrossEntropyLoss', loss_weight=0.2),
                 loss_iou=None,
                 echo_mismatch_gt=False,
                 init_cfg=None,
                 long_branch_num=0,
                 norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
                 conv_bias=True):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.diff_rad_by_sin = diff_rad_by_sin
        self.use_direction_classifier = use_direction_classifier
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner_per_size = assigner_per_size
        self.assign_per_class = assign_per_class
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.long_branch_num = long_branch_num
        self.assign_per_level = assign_per_level
        self.norm_cfg = norm_cfg
        self.conv_bias = conv_bias
        self.echo_mismatch_gt = echo_mismatch_gt

        # build anchor generator
        self.prior_generator = TASK_UTILS.build(anchor_generator)
        self.multi_level_anchors_cache = None
        # In 3D detection, the anchor stride is connected with anchor size
        self.num_anchors = self.prior_generator.num_base_anchors
        # build box coder
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.box_code_size = self.bbox_coder.code_size

        # build loss function
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in [
            'mmdet.FocalLoss', 'mmdet.GHMC'
        ]
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_dir = MODELS.build(loss_dir)
        self.loss_iou = None if loss_iou is None else MODELS.build(loss_dir)

        self._init_layers()
        self._init_assigner_sampler()

        if init_cfg is None:
            self.init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=dict(
                    type='Normal', name='conv_cls', std=0.01, bias_prob=0.01))

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return
        print("Train cfg: ", self.train_cfg)
        if self.sampling:
            self.bbox_sampler = TASK_UTILS.build(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                TASK_UTILS.build(res) for res in self.train_cfg.assigner
            ]

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.cls_out_channels = self.num_anchors * self.num_classes
        if self.long_branch_num <= 0:
            self.conv_cls = nn.Conv2d(self.feat_channels,
                                      self.cls_out_channels, 1)
            self.conv_reg = nn.Conv2d(self.feat_channels,
                                      self.num_anchors * self.box_code_size, 1)
            if self.use_direction_classifier:
                self.conv_dir_cls = nn.Conv2d(self.feat_channels,
                                              self.num_anchors * 2, 1)
            return
        in_channels = self.feat_channels
        cls_blocks = []
        reg_blocks = []
        dir_cls_blocks = []
        for _ in range(self.long_branch_num):
            cls_blocks.extend([
                build_conv_layer(
                    dict(type='Conv2d', bias=self.conv_bias),
                    in_channels,
                    256,
                    3,
                    stride=1,
                    padding=1),
                build_norm_layer(self.norm_cfg, 256)[1],
                nn.SiLU(inplace=True),
            ])
            reg_blocks.extend([
                build_conv_layer(
                    dict(type='Conv2d', bias=self.conv_bias),
                    in_channels,
                    256,
                    3,
                    stride=1,
                    padding=1),
                build_norm_layer(self.norm_cfg, 256)[1],
                nn.SiLU(inplace=True),
            ])
            if self.use_direction_classifier:
                dir_cls_blocks.extend([
                    build_conv_layer(
                        dict(type='Conv2d', bias=self.conv_bias),
                        in_channels,
                        256,
                        3,
                        stride=1,
                        padding=1),
                    build_norm_layer(self.norm_cfg, 256)[1],
                    nn.SiLU(inplace=True),
                ])
            in_channels = 256
        cls_blocks.append(nn.Conv2d(in_channels, self.cls_out_channels, 1))
        reg_blocks.append(
            nn.Conv2d(in_channels, self.num_anchors * self.box_code_size, 1))

        self.conv_cls = nn.Sequential(*cls_blocks)
        self.conv_reg = nn.Sequential(*reg_blocks)
        if self.use_direction_classifier:
            dir_cls_blocks.append(
                nn.Conv2d(in_channels, self.num_anchors * 2, 1))
            self.conv_dir_cls = nn.Sequential(*dir_cls_blocks)

    def forward_single(self, x):
        """Forward function on a single-scale feature map.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            tuple[torch.Tensor]: Contain score of each class, bbox \
                regression and direction classification predictions.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_preds = None
        if self.use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_preds

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple[list[torch.Tensor]]: Multi-level class score, bbox \
                and direction predictions.
        """
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, input_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            input_metas (list[dict]): contain pcd and img's meta info.
            device (str): device of current module.

        Returns:
            list[list[torch.Tensor]]: Anchors of each image, valid flags \
                of each image.
        """
        num_imgs = len(input_metas)
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        if self.multi_level_anchors_cache is None:
            self.multi_level_anchors_cache = self.prior_generator.grid_anchors(
                featmap_sizes, device=device)
        anchor_list = [self.multi_level_anchors_cache for _ in range(num_imgs)]
        return anchor_list

    def _loss_by_feat_single(self, cls_score, bbox_pred, dir_cls_preds, labels,
                             label_weights, bbox_targets, bbox_weights,
                             dir_targets, dir_weights, num_total_samples):
        """Calculate loss of Single-level results.

        Args:
            cls_score (torch.Tensor): Class score in single-level.
            bbox_pred (torch.Tensor): Bbox prediction in single-level.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single-level.
            labels (torch.Tensor): Labels of class.
            label_weights (torch.Tensor): Weights of class loss.
            bbox_targets (torch.Tensor): Targets of bbox predictions.
            bbox_weights (torch.Tensor): Weights of bbox loss.
            dir_targets (torch.Tensor): Targets of direction predictions.
            dir_weights (torch.Tensor): Weights of direction loss.
            num_total_samples (int): The number of valid samples.

        Returns:
            tuple[torch.Tensor]: Losses of class, bbox \
                and direction, respectively.
        """
        # classification loss
        if num_total_samples is None:
            num_total_samples = int(cls_score.shape[0])
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        assert labels.max().item() <= self.num_classes
        # print(cls_score.shape,labels.shape)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, self.box_code_size)
        bbox_targets = bbox_targets.reshape(-1, self.box_code_size)
        bbox_weights = bbox_weights.reshape(-1, self.box_code_size)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero(
            as_tuple=False).reshape(-1)
        num_pos = len(pos_inds)

        pos_bbox_pred = bbox_pred[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_bbox_weights = bbox_weights[pos_inds]

        # dir loss
        if self.use_direction_classifier:
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).reshape(-1, 2)
            dir_targets = dir_targets.reshape(-1)
            dir_weights = dir_weights.reshape(-1)
            pos_dir_cls_preds = dir_cls_preds[pos_inds]
            pos_dir_targets = dir_targets[pos_inds]
            pos_dir_weights = dir_weights[pos_inds]

        if num_pos > 0:
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                pos_bbox_weights = pos_bbox_weights * bbox_weights.new_tensor(
                    code_weight)
            if self.diff_rad_by_sin:
                pos_bbox_pred, pos_bbox_targets = self.add_sin_difference(
                    pos_bbox_pred, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_targets,
                pos_bbox_weights,
                avg_factor=num_total_samples)
            if self.loss_iou is not None:
                loss_iou = self.loss_iou(
                    pos_bbox_pred,
                    pos_bbox_targets,
                    pos_bbox_weights,
                    avg_factor=num_total_samples)
                loss_bbox += loss_iou

            # direction classification loss
            loss_dir = None
            if self.use_direction_classifier:
                loss_dir = self.loss_dir(
                    pos_dir_cls_preds,
                    pos_dir_targets,
                    pos_dir_weights,
                    avg_factor=num_total_samples)
        else:
            loss_bbox = pos_bbox_pred.sum()
            if self.use_direction_classifier:
                loss_dir = pos_dir_cls_preds.sum()

        return loss_cls, loss_bbox, loss_dir

    @staticmethod
    def add_sin_difference(boxes1, boxes2):
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th \
                dimensions are changed.
        """
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7])
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[...,
                                                                         6:7])
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                           dim=-1)
        return boxes1, boxes2

    def loss_by_feat(self,
                     cls_scores,
                     bbox_preds,
                     dir_cls_preds,
                     batch_gt_instances_3d,
                     batch_input_metas,
                     batch_gt_instances_ignore=None) -> dict:
        # gt_bboxes,
        # gt_labels,
        # input_metas,
        # gt_bboxes_ignore=None):
        """Calculate losses.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d``
                and ``labels_3d`` attributes.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, list[torch.Tensor]]: Classification, bbox, and \
                direction losses of each level.

                - loss_cls (list[torch.Tensor]): Classification losses.
                - loss_bbox (list[torch.Tensor]): Box regression losses.
                - loss_dir (list[torch.Tensor]): Direction classification \
                    losses.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list = self.get_anchors(
            featmap_sizes, batch_input_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        # Parse the gt bboxes and gt labels
        gt_labels = [
            batch_gt_instances_3d[i]['labels_3d']
            for i in range(len(batch_gt_instances_3d))
        ]
        gt_bboxes = [
            batch_gt_instances_3d[i]['bboxes_3d']
            for i in range(len(batch_gt_instances_3d))
        ]

        # for i in range(len(gt_bboxes)):
        #     if 'labels_3d' in gt_bboxes[0]:
        #         gt_labels.append(gt_bboxes[0]['labels_3d'])

        cls_reg_targets = self.anchor_target_3d(
            anchor_list,
            cls_scores,
            bbox_preds,
            gt_bboxes,
            batch_input_metas,
            gt_bboxes_ignore_list=batch_gt_instances_ignore,
            gt_labels_list=gt_labels,
            num_classes=self.num_classes,
            label_channels=label_channels,
            sampling=self.sampling)

        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         dir_targets_list, dir_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # # Extract the assigned anchors
        # ped_label_indices = torch.where(labels_list[0]==0)[1]
        # cyc_label_indices = torch.where(labels_list[0]==1)[1]
        # car_label_indices = torch.where(labels_list[0]==2)[1]
        # if len(ped_label_indices) > 0:
        #     ped_label_weight = label_weights_list[0][:, ped_label_indices]
        #     ped_bboxes = bbox_targets_list[0][:, ped_label_indices, :]
        # if len(cyc_label_indices) > 0:
        #     cyc_label_weight = label_weights_list[0][:, cyc_label_indices]
        #     cyc_bboxes = bbox_targets_list[0][:, cyc_label_indices, :]
        # if len(car_label_indices) > 0:
        #     car_label_weight = label_weights_list[0][:, car_label_indices]
        #     car_bboxes = bbox_targets_list[0][:, car_label_indices, :]

        # print(len(cls_scores),cls_scores[0].shape)#,,dir_cls_preds)
        # print(len(bbox_preds),bbox_preds[0].shape)#,bbox_preds,dir_cls_preds)
        # print(len(dir_cls_preds),dir_cls_preds[0].shape)#,bbox_preds,dir_cls_preds)
        # print(labels_list[0].shape,bbox_targets_list[0].shape,dir_weights_list[0].shape)
        # num_total_samples = None
        losses_cls, losses_bbox, losses_dir = multi_apply(
            self._loss_by_feat_single,
            cls_scores,
            bbox_preds,
            dir_cls_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            dir_targets_list,
            dir_weights_list,
            num_total_samples=num_total_samples)
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dir=losses_dir)

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the 3D detection head and predict
        detection results on the features of the upstream network.

        Args:
            feats_dict (dict): Contains features from the first stage.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(x)
        predictions = self.predict_by_feat(
            *outs, batch_input_metas=batch_input_metas, rescale=rescale)

        return predictions

    def predict_by_feat(self,
                        cls_scores,
                        bbox_preds,
                        dir_cls_preds,
                        batch_input_metas,
                        cfg=None,
                        rescale=False,
                        onnx_decode=False):
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
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        device = cls_scores[0].device
        # breakpoint()

        mlvl_anchors = self.prior_generator.grid_anchors(
            featmap_sizes, device=device)
        if isinstance(mlvl_anchors[0], list):
            mlvl_anchors = [
                torch.cat(anchor, dim=-3).reshape(-1, self.box_code_size)
                for anchor in mlvl_anchors
            ]
        else:
            mlvl_anchors = [
                anchor.reshape(-1, self.box_code_size)
                for anchor in mlvl_anchors
            ]

        result_list = []
        for img_id in range(len(batch_input_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            dir_cls_pred_list = [
                dir_cls_preds[i][img_id].detach() for i in range(num_levels)
            ]

            input_meta = batch_input_metas[img_id]
            proposals = self._predict_by_feat_single(cls_score_list,
                                                     bbox_pred_list,
                                                     dir_cls_pred_list,
                                                     mlvl_anchors, input_meta,
                                                     cfg, rescale, onnx_decode)
            result_list.append(proposals)
        return result_list

    def _predict_by_feat_single(self,
                                cls_scores,
                                bbox_preds,
                                dir_cls_preds,
                                mlvl_anchors,
                                input_meta,
                                cfg=None,
                                rescale=False,
                                onnx_decode=False):
        """Get bboxes of single branch.

        Args:
            cls_scores (torch.Tensor): Class score in single batch.
            bbox_preds (torch.Tensor): Bbox prediction in single batch.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single batch.
            mlvl_anchors (List[torch.Tensor]): Multi-level anchors
                in single batch.
            input_meta (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): whether th rescale bbox.

        Returns:
            tuple: Contain predictions of single batch.

                - bboxes (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
                - scores (torch.Tensor): Class score of each bbox.
                - labels (torch.Tensor): Label of each bbox.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_dir_preds = []
        for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
                cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            # sample_path = input_meta['lidar_path']
            # if ('3_33' in sample_path):
            #     frame_name = osp.basename(sample_path)
            #     dump_scores_pth = f'./head_scores_sigmoid_{frame_name}.pkl'
            #     with open(dump_scores_pth, 'wb') as f:
            #         pkl.dump(scores, f)

            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.box_code_size)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)

                _, topk_inds = max_scores.topk(nms_pre)

                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]
                dir_cls_pred = dir_cls_pred[topk_inds]

            # decode is finished in onnx already
            if not onnx_decode:
                bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            else:
                bboxes = bbox_pred

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_dir_preds.append(dir_cls_pred)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.box_code_size).bev)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        mlvl_dir_preds = torch.cat(mlvl_dir_preds)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = cfg.get('score_thr', 0)
        # rotate to x positive
        mlvl_bboxes_for_nms[..., 4] += np.pi / 2

        results = box3d_multiclass_nms(
            mlvl_bboxes,
            mlvl_bboxes_for_nms,
            mlvl_scores,
            score_thr,
            cfg.max_num,
            cfg,
            mlvl_dir_scores,
            mlvl_dir_preds=mlvl_dir_preds)

        bboxes, scores, labels, dir_scores, dir_preds = results

        # Recover correct value
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * dir_scores.to(bboxes.dtype))
        bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)

        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels

        return results
    

    
@MODELS.register_module()
class MultiClassAnchor3DHead(Base3DDenseHead):
    
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 use_direction_classifier=True,
                 class_head_configs=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.use_direction_classifier = use_direction_classifier
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # Create class-specific heads
        self.class_heads = nn.ModuleList()
        self.class_to_head_idx = {}  # Maps class index to head index
        current_class_idx = 0
        
        for head_idx, head_config in enumerate(class_head_configs):
            head_classes = head_config['classes']
            head_type = head_config['type']
            config = head_config.copy()
            config.pop('classes')
            config.pop('type')
            print(head_type,head_classes)
            head = MODELS.build(dict(
                type=head_type,
                num_classes=len(head_classes),
                in_channels=in_channels,
                feat_channels=feat_channels,
                use_direction_classifier=use_direction_classifier,
                train_cfg=train_cfg,
                test_cfg=test_cfg,
                **config
            ))
            
            self.class_heads.append(head)
            
            # Map classes to this head
            for class_name in head_classes:
                self.class_to_head_idx[current_class_idx] = head_idx
                current_class_idx += 1
    def forward(self, feats):

        x = []
        for head in self.class_heads:
            a = head(feats)
            x.append(a)
        return x

    

    def predict(self,
                x,
                batch_data_samples,
                rescale=False):
        data_samples_metainfo = [data_samples.metainfo for data_samples in batch_data_samples]
        outs = self(x)
        head_results = []

        for idx, head in enumerate(self.class_heads) : 
            head_result = head.predict_by_feat(*outs[idx], data_samples_metainfo, rescale=rescale)[0]

            head_classes = [class_idx for class_idx, h_idx 
                                    in self.class_to_head_idx.items() 
                                    if h_idx == idx]
            class_mapping = {i: original_idx 
                                    for i, original_idx in enumerate(head_classes)}
            head_result.labels_3d = torch.tensor(
                    [class_mapping[label.item()] 
                     for label in head_result.labels_3d],
                    device=head_result.labels_3d.device)
            head_results.append(head_result)
                
               
        combined_results = InstanceData() 
       
        if len(head_results) > 0:
                combined_results.bboxes_3d = LiDARInstance3DBoxes(torch.cat(
                    [pred.bboxes_3d.tensor for pred in head_results]))
                combined_results.scores_3d = torch.cat(
                    [pred.scores_3d for pred in head_results])
                combined_results.labels_3d = torch.cat(
                    [pred.labels_3d for pred in head_results])
        
        return combined_results
    def loss_by_feat(self,
                     cls_scores,
                     bbox_preds,
                     dir_cls_preds,
                     batch_gt_instances_3d,
                     batch_input_metas,
                     batch_gt_instances_ignore=None):
        
        losses = dict()
        
        for head_idx, head in enumerate(self.class_heads):
            head_gt_instances = []
            
            for batch_idx, gt_instances in enumerate(batch_gt_instances_3d):
                head_classes = [class_idx for class_idx, h_idx 
                              in self.class_to_head_idx.items() 
                              if h_idx == head_idx]
                
                mask = torch.tensor([label in head_classes 
                                   for label in gt_instances.labels_3d])
                
                head_instance = InstanceData()
                head_instance.bboxes_3d = gt_instances.bboxes_3d[mask]
                head_instance.labels_3d = gt_instances.labels_3d[mask]
                head_gt_instances.append(head_instance)
            
            # Calculate loss for this head
            head_losses = head.loss_by_feat(
                [cls_scores[head_idx]],
                [bbox_preds[head_idx]],
                [dir_cls_preds[head_idx]],
                head_gt_instances,
                batch_input_metas,
                batch_gt_instances_ignore
            )
            
            # Add prefix to distinguish between heads
            for k, v in head_losses.items():
                losses[f'head{head_idx}.{k}'] = v
                
        return losses

    