# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmengine.structures import InstanceData
from mmdet.models.utils import images_to_levels, multi_apply
from mmdet3d.structures import limit_period
from mmdet3d.models.task_modules.assigners import ATSS3DAssigner


class AnchorTrainMixin(object):
    """Mixin class for target assigning of dense heads."""

    def anchor_target_3d(
            self,
            anchor_list,
            pred_scores_list,  # bs*(anchor_num*3)*h*w
            pred_bboxes_list,  # bs*(anchor_num*7)*h*w
            gt_bboxes_list,
            input_metas,
            gt_bboxes_ignore_list=None,
            gt_labels_list=None,
            label_channels=1,
            num_classes=1,
            sampling=True):
        """Compute regression and classification targets for anchors.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            gt_bboxes_list (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                bboxes of each image.
            input_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (None | list): Ignore list of gt bboxes.
            gt_labels_list (list[torch.Tensor]): Gt labels of batches.
            label_channels (int): The channel of labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple (list, list, list, list, list, list, int, int):
                Anchor targets, including labels, label weights,
                bbox targets, bbox weights, direction targets,
                direction weights, number of postive anchors and
                number of negative anchors.
        """
        num_imgs = len(input_metas)
        assert len(anchor_list) == num_imgs

        if isinstance(anchor_list[0][0], list):
            # sizes of anchors are different
            # anchor number of a single level
            num_level_anchors = [
                sum([anchor.view(-1, self.box_code_size).size(0) for anchor in anchors]) for anchors in anchor_list[0]
            ]
            pred_scores_list = [_.detach() for _ in pred_scores_list]
            pred_scores_list = [[
                _[idx].reshape(-1, self.num_classes, _.shape[2], _.shape[3]).permute(2, 3, 0, 1)
                for _ in pred_scores_list
            ] for idx in range(num_imgs)]
            pred_bboxes_list = [_.detach() for _ in pred_bboxes_list]
            pred_bboxes_list = [[
                _[idx].reshape(-1, self.box_code_size, _.shape[2], _.shape[3]).permute(2, 3, 0, 1)
                for _ in pred_bboxes_list
            ] for idx in range(num_imgs)]
        else:
            # anchor number of multi levels
            num_level_anchors = [anchors.view(-1, self.box_code_size).size(0) for anchors in anchor_list[0]]
            # concat all level anchors and flags to a single tensor
            for i in range(num_imgs):
                anchor_list[i] = torch.cat(anchor_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, all_dir_targets, all_dir_weights,
         pos_inds_list, neg_inds_list) = multi_apply(self.anchor_target_3d_single,
                                                     anchor_list,
                                                     pred_scores_list,
                                                     pred_bboxes_list,
                                                     gt_bboxes_list,
                                                     gt_bboxes_ignore_list,
                                                     gt_labels_list,
                                                     input_metas,
                                                     label_channels=label_channels,
                                                     num_classes=num_classes,
                                                     sampling=sampling)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        dir_targets_list = images_to_levels(all_dir_targets, num_level_anchors)
        dir_weights_list = images_to_levels(all_dir_weights, num_level_anchors)
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, dir_targets_list,
                dir_weights_list, num_total_pos, num_total_neg)

    # flake8: noqa: C901
    def anchor_target_3d_single(self,
                                anchors,
                                pred_scores,
                                pred_bboxes,
                                gt_bboxes,
                                gt_bboxes_ignore,
                                gt_labels,
                                input_meta,
                                label_channels=1,
                                num_classes=1,
                                sampling=True):
        """Compute targets of anchors in single batch.
        Args:
            anchors (torch.Tensor): Concatenated multi-level anchor.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Gt bboxes.
            gt_bboxes_ignore (torch.Tensor): Ignored gt bboxes.
            gt_labels (torch.Tensor): Gt class labels.
            input_meta (dict): Meta info of each image.
            label_channels (int): The channel of labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple[torch.Tensor]: Anchor targets.
        """

        if isinstance(self.bbox_assigner, list) and (not isinstance(anchors, list)):
            feat_size = anchors.size(0) * anchors.size(1) * anchors.size(2)
            rot_angles = anchors.size(-2)
            assert len(self.bbox_assigner) == anchors.size(-3)
            (total_labels, total_label_weights, total_bbox_targets, total_bbox_weights, total_dir_targets,
             total_dir_weights, total_pos_inds, total_neg_inds) = [], [], [], [], [], [], [], []
            current_anchor_num = 0
            for i, assigner in enumerate(self.bbox_assigner):
                current_anchors = anchors[..., i, :, :].reshape(-1, self.box_code_size)
                current_anchor_num += current_anchors.size(0)
                if self.assign_per_class:
                    gt_per_cls = (gt_labels == i)
                    anchor_targets = self.anchor_target_single_assigner(assigner, current_anchors,
                                                                        gt_bboxes[gt_per_cls, :], gt_bboxes_ignore,
                                                                        gt_labels[gt_per_cls], input_meta, num_classes,
                                                                        sampling)
                else:
                    anchor_targets = self.anchor_target_single_assigner(assigner, current_anchors, gt_bboxes,
                                                                        gt_bboxes_ignore, gt_labels, input_meta,
                                                                        num_classes, sampling)

                (labels, label_weights, bbox_targets, bbox_weights, dir_targets, dir_weights, pos_inds,
                 neg_inds) = anchor_targets
                total_labels.append(labels.reshape(feat_size, 1, rot_angles))
                total_label_weights.append(label_weights.reshape(feat_size, 1, rot_angles))
                total_bbox_targets.append(bbox_targets.reshape(feat_size, 1, rot_angles, anchors.size(-1)))
                total_bbox_weights.append(bbox_weights.reshape(feat_size, 1, rot_angles, anchors.size(-1)))
                total_dir_targets.append(dir_targets.reshape(feat_size, 1, rot_angles))
                total_dir_weights.append(dir_weights.reshape(feat_size, 1, rot_angles))
                total_pos_inds.append(pos_inds)
                total_neg_inds.append(neg_inds)

            total_labels = torch.cat(total_labels, dim=-2).reshape(-1)
            total_label_weights = torch.cat(total_label_weights, dim=-2).reshape(-1)
            total_bbox_targets = torch.cat(total_bbox_targets, dim=-3).reshape(-1, anchors.size(-1))
            total_bbox_weights = torch.cat(total_bbox_weights, dim=-3).reshape(-1, anchors.size(-1))
            total_dir_targets = torch.cat(total_dir_targets, dim=-2).reshape(-1)
            total_dir_weights = torch.cat(total_dir_weights, dim=-2).reshape(-1)
            total_pos_inds = torch.cat(total_pos_inds, dim=0).reshape(-1)
            total_neg_inds = torch.cat(total_neg_inds, dim=0).reshape(-1)
            return (total_labels, total_label_weights, total_bbox_targets, total_bbox_weights, total_dir_targets,
                    total_dir_weights, total_pos_inds, total_neg_inds)
        elif isinstance(self.bbox_assigner, list) and isinstance(anchors[0], list):
            # import pdb; pdb.set_trace()
            # this code branch is using!

            # ###############xiaohan: This code branch is using!
            # ###### Single scale/Multi scale
            # ###### Class-Aware Anchor
            # ###### Per-class Assigner
            assert len(self.bbox_assigner) == self.num_classes
            (total_labels_list, total_label_weights_list, total_bbox_targets_list, total_bbox_weights_list,
             total_dir_targets_list, total_dir_weights_list, total_pos_inds_list,
             total_neg_inds_list) = [], [], [], [], [], [], [], []
            if self.assign_per_level:
                for level_i, level_anchors in enumerate(anchors):
                    (total_labels, total_label_weights, total_bbox_targets, total_bbox_weights, total_dir_targets,
                     total_dir_weights, total_pos_inds, total_neg_inds) = [], [], [], [], [], [], [], []
                    current_anchor_num = 0
                    for i, assigner in enumerate(self.bbox_assigner):
                        current_anchors = level_anchors[i]
                        current_pred_scores = pred_scores[level_i][..., current_anchor_num:current_anchor_num
                                                                   + current_anchors.shape[-3], :self.num_classes]
                        current_pred_bboxes = pred_bboxes[level_i][..., current_anchor_num:current_anchor_num
                                                                   + current_anchors.shape[-3], :self.box_code_size]
                        current_anchor_num += current_anchors.shape[-3]
                        if self.assign_per_class:
                            gt_per_cls = (gt_labels == i)
                            anchor_targets = self.anchor_target_single_assigner(assigner, current_anchors,
                                                                                current_pred_scores,
                                                                                current_pred_bboxes,
                                                                                gt_bboxes[gt_per_cls, :],
                                                                                gt_bboxes_ignore, gt_labels[gt_per_cls],
                                                                                input_meta, num_classes, sampling)
                        else:
                            anchor_targets = self.anchor_target_single_assigner(assigner, current_anchors,
                                                                                current_pred_scores,
                                                                                current_pred_bboxes, gt_bboxes,
                                                                                gt_bboxes_ignore, gt_labels, input_meta,
                                                                                num_classes, sampling)

                        (labels, label_weights, bbox_targets, bbox_weights, dir_targets, dir_weights, pos_inds,
                         neg_inds) = anchor_targets
                        total_labels.append(labels.reshape(current_anchors.shape[1], current_anchors.shape[2], -1))
                        total_label_weights.append(
                            label_weights.reshape(current_anchors.shape[1], current_anchors.shape[2], -1))
                        total_bbox_targets.append(
                            bbox_targets.reshape(current_anchors.shape[1], current_anchors.shape[2], -1,
                                                 level_anchors[i].size(-1)))
                        total_bbox_weights.append(
                            bbox_weights.reshape(current_anchors.shape[1], current_anchors.shape[2], -1,
                                                 level_anchors[i].size(-1)))
                        total_dir_targets.append(
                            dir_targets.reshape(current_anchors.shape[1], current_anchors.shape[2], -1))
                        total_dir_weights.append(
                            dir_weights.reshape(current_anchors.shape[1], current_anchors.shape[2], -1))
                        total_pos_inds.append(pos_inds)
                        total_neg_inds.append(neg_inds)

                    total_labels = torch.cat(total_labels, dim=2).reshape(-1)
                    total_label_weights = torch.cat(total_label_weights, dim=2).reshape(-1)
                    total_bbox_targets = torch.cat(total_bbox_targets,
                                                   dim=2).reshape(-1, total_bbox_targets[0].size(-1))
                    total_bbox_weights = torch.cat(total_bbox_weights,
                                                   dim=2).reshape(-1, total_bbox_weights[0].size(-1))
                    total_dir_targets = torch.cat(total_dir_targets, dim=2).reshape(-1)
                    total_dir_weights = torch.cat(total_dir_weights, dim=2).reshape(-1)
                    total_pos_inds = torch.cat(total_pos_inds, dim=0).reshape(-1)
                    total_neg_inds = torch.cat(total_neg_inds, dim=0).reshape(-1)
                    total_labels_list.append(total_labels)
                    total_label_weights_list.append(total_label_weights)
                    total_bbox_targets_list.append(total_bbox_targets)
                    total_bbox_weights_list.append(total_bbox_weights)
                    total_dir_targets_list.append(total_dir_targets)
                    total_dir_weights_list.append(total_dir_weights)
                    total_pos_inds_list.append(total_pos_inds)
                    total_neg_inds_list.append(total_neg_inds)
            else:
                (total_labels, total_label_weights, total_bbox_targets, total_bbox_weights, total_dir_targets,
                 total_dir_weights, total_pos_inds, total_neg_inds) = [[] for _ in anchors],\
                                                                      [[] for _ in anchors],\
                                                                      [[] for _ in anchors],\
                                                                      [[] for _ in anchors],\
                                                                      [[] for _ in anchors],\
                                                                      [[] for _ in anchors],\
                                                                      [[] for _ in anchors],\
                                                                      [[] for _ in anchors]
                current_anchor_nums = [[0 for __ in range(len(anchors))] for _ in range(len(self.bbox_assigner))]
                for i, assigner in enumerate(self.bbox_assigner):
                    current_pred_scores = []
                    current_pred_bboxes = []
                    current_anchors = []
                    level_anchor_nums = []
                    for level_i, level_anchors in enumerate(anchors):
                        if level_anchors[i].shape[-3] == 0:
                            level_anchor_nums.append(0)
                            continue
                        current_anchor_num = current_anchor_nums[level_i][i]
                        current_anchors.append(level_anchors[i])
                        current_pred_scores.append(
                            pred_scores[level_i][..., current_anchor_num:current_anchor_num
                                                 + level_anchors[i].shape[-3], :self.num_classes])
                        current_pred_bboxes.append(
                            pred_bboxes[level_i][..., current_anchor_num:current_anchor_num
                                                 + level_anchors[i].shape[-3], :self.box_code_size])
                        current_anchor_nums[level_i][i] += level_anchors[i].shape[-3]
                        level_anchor_nums.append(
                            [current_pred_scores[-1].shape[:-1].numel(), current_pred_scores[-1].shape[-2]])
                        # level_anchor_nums.append(current_pred_scores[-1].numel()//self.num_classes)
                    current_anchors = torch.cat([x.reshape(-1, 1, self.box_code_size) for x in current_anchors], dim=0)
                    current_pred_scores = torch.cat([x.reshape(-1, self.num_classes) for x in current_pred_scores],
                                                    dim=0)
                    current_pred_bboxes = torch.cat([x.reshape(-1, self.box_code_size) for x in current_pred_bboxes],
                                                    dim=0)
                    if self.assign_per_class:
                        gt_per_cls = (gt_labels == i)
                        anchor_targets = self.anchor_target_single_assigner(assigner, current_anchors,
                                                                            current_pred_scores, current_pred_bboxes,
                                                                            gt_bboxes[gt_per_cls, :], gt_bboxes_ignore,
                                                                            gt_labels[gt_per_cls], input_meta,
                                                                            num_classes, sampling)
                    else:
                        anchor_targets = self.anchor_target_single_assigner(assigner, current_anchors,
                                                                            current_pred_scores, current_pred_bboxes,
                                                                            gt_bboxes, gt_bboxes_ignore, gt_labels,
                                                                            input_meta, num_classes, sampling)

                    (labels, label_weights, bbox_targets, bbox_weights, dir_targets, dir_weights, pos_inds,
                     neg_inds) = anchor_targets

                    num_start_idx = 0
                    for level_i, level_anchors in enumerate(anchors):
                        if level_anchors[i].shape[-3] == 0:
                            continue
                        current_anchor_num, anchor_cnt = level_anchor_nums[level_i]  # anchor_num:1280*256 anchor_cnt: 2
                        total_labels[level_i].append(labels[num_start_idx:num_start_idx + current_anchor_num].reshape(
                            -1, anchor_cnt, 1))
                        total_label_weights[level_i].append(
                            label_weights[num_start_idx:num_start_idx + current_anchor_num].reshape(-1, anchor_cnt, 1))
                        total_bbox_targets[level_i].append(bbox_targets[num_start_idx:num_start_idx
                                                                        + current_anchor_num].reshape(
                                                                            -1, anchor_cnt, self.box_code_size))
                        total_bbox_weights[level_i].append(bbox_weights[num_start_idx:num_start_idx
                                                                        + current_anchor_num].reshape(
                                                                            -1, anchor_cnt, self.box_code_size))
                        total_dir_targets[level_i].append(dir_targets[num_start_idx:num_start_idx
                                                                      + current_anchor_num].reshape(-1, anchor_cnt, 1))
                        total_dir_weights[level_i].append(dir_weights[num_start_idx:num_start_idx
                                                                      + current_anchor_num].reshape(-1, anchor_cnt, 1))
                        total_pos_inds[level_i].append(pos_inds[num_start_idx:num_start_idx + current_anchor_num])
                        total_neg_inds[level_i].append(neg_inds[num_start_idx:num_start_idx + current_anchor_num])
                        num_start_idx += current_anchor_num

                total_labels = torch.cat([torch.cat(x, dim=-2).reshape(-1, 1) for x in total_labels], dim=0).reshape(-1)
                total_label_weights = torch.cat([torch.cat(x, dim=-2).reshape(-1, 1) for x in total_label_weights],
                                                dim=0).reshape(-1)
                total_bbox_targets = torch.cat(
                    [torch.cat(x, dim=-2).reshape(-1, self.box_code_size) for x in total_bbox_targets],
                    dim=0).reshape(-1, self.box_code_size)
                total_bbox_weights = torch.cat(
                    [torch.cat(x, dim=-2).reshape(-1, self.box_code_size) for x in total_bbox_weights],
                    dim=0).reshape(-1, self.box_code_size)
                total_dir_targets = torch.cat([torch.cat(x, dim=-2).reshape(-1, 1) for x in total_dir_targets],
                                              dim=0).reshape(-1)
                total_dir_weights = torch.cat([torch.cat(x, dim=-2).reshape(-1, 1) for x in total_dir_weights],
                                              dim=0).reshape(-1)
                total_pos_inds = torch.cat([torch.cat(x, dim=0) for x in total_pos_inds], dim=0).reshape(-1)
                total_neg_inds = torch.cat([torch.cat(x, dim=0) for x in total_neg_inds], dim=0).reshape(-1)

                total_labels_list.append(total_labels)
                total_label_weights_list.append(total_label_weights)
                total_bbox_targets_list.append(total_bbox_targets)
                total_bbox_weights_list.append(total_bbox_weights)
                total_dir_targets_list.append(total_dir_targets)
                total_dir_weights_list.append(total_dir_weights)
                total_pos_inds_list.append(total_pos_inds)
                total_neg_inds_list.append(total_neg_inds)

            total_labels_list = torch.cat(total_labels_list)
            total_label_weights_list = torch.cat(total_label_weights_list)
            total_bbox_targets_list = torch.cat(total_bbox_targets_list, dim=0)
            total_bbox_weights_list = torch.cat(total_bbox_weights_list, dim=0)
            total_dir_targets_list = torch.cat(total_dir_targets_list, dim=0)
            total_dir_weights_list = torch.cat(total_dir_weights_list, dim=0)
            total_pos_inds_list = torch.cat(total_pos_inds_list, dim=0)
            total_neg_inds_list = torch.cat(total_neg_inds_list, dim=0)
            return (total_labels_list, total_label_weights_list, total_bbox_targets_list, total_bbox_weights_list,
                    total_dir_targets_list, total_dir_weights_list, total_pos_inds_list, total_neg_inds_list)
        elif isinstance(self.bbox_assigner, list) and isinstance(anchors, list):
            # class-aware anchors with different feature map sizes
            assert len(self.bbox_assigner) == len(anchors), \
                'The number of bbox assigners and anchors should be the same.'
            (total_labels, total_label_weights, total_bbox_targets, total_bbox_weights, total_dir_targets,
             total_dir_weights, total_pos_inds, total_neg_inds) = [], [], [], [], [], [], [], []
            current_anchor_num = 0
            for i, assigner in enumerate(self.bbox_assigner):
                current_anchors = anchors[i]
                current_anchor_num += current_anchors.size(0)
                if self.assign_per_class:
                    gt_per_cls = (gt_labels == i)
                    anchor_targets = self.anchor_target_single_assigner(assigner, current_anchors,
                                                                        gt_bboxes[gt_per_cls, :], gt_bboxes_ignore,
                                                                        gt_labels[gt_per_cls], input_meta, num_classes,
                                                                        sampling)
                else:
                    anchor_targets = self.anchor_target_single_assigner(assigner, current_anchors, gt_bboxes,
                                                                        gt_bboxes_ignore, gt_labels, input_meta,
                                                                        num_classes, sampling)

                (labels, label_weights, bbox_targets, bbox_weights, dir_targets, dir_weights, pos_inds,
                 neg_inds) = anchor_targets
                total_labels.append(labels.reshape(current_anchors.shape[1], current_anchors.shape[2], -1))
                total_label_weights.append(label_weights.reshape(current_anchors.shape[1], current_anchors.shape[2],
                                                                 -1))
                total_bbox_targets.append(
                    bbox_targets.reshape(current_anchors.shape[1], current_anchors.shape[2], -1, anchors[i].size(-1)))
                total_bbox_weights.append(
                    bbox_weights.reshape(current_anchors.shape[1], current_anchors.shape[2], -1, anchors[i].size(-1)))
                total_dir_targets.append(dir_targets.reshape(current_anchors.shape[1], current_anchors.shape[2], -1))
                total_dir_weights.append(dir_weights.reshape(current_anchors.shape[1], current_anchors.shape[2], -1))
                total_pos_inds.append(pos_inds)
                total_neg_inds.append(neg_inds)

            total_labels = torch.cat(total_labels, dim=2).reshape(-1)
            total_label_weights = torch.cat(total_label_weights, dim=2).reshape(-1)
            total_bbox_targets = torch.cat(total_bbox_targets, dim=2).reshape(-1, total_bbox_targets[0].size(-1))
            total_bbox_weights = torch.cat(total_bbox_weights, dim=2).reshape(-1, total_bbox_weights[0].size(-1))
            total_dir_targets = torch.cat(total_dir_targets, dim=2).reshape(-1)
            total_dir_weights = torch.cat(total_dir_weights, dim=2).reshape(-1)
            total_pos_inds = torch.cat(total_pos_inds, dim=0).reshape(-1)
            total_neg_inds = torch.cat(total_neg_inds, dim=0).reshape(-1)
            return (total_labels, total_label_weights, total_bbox_targets, total_bbox_weights, total_dir_targets,
                    total_dir_weights, total_pos_inds, total_neg_inds)
        else:
            return self.anchor_target_single_assigner(self.bbox_assigner, anchors, gt_bboxes, gt_bboxes_ignore,
                                                      gt_labels, input_meta, num_classes, sampling)
    

    def anchor_target_single_assigner(self,
                                      bbox_assigner,
                                      anchors,
                                      pred_scores,
                                      pred_bboxes,
                                      gt_bboxes,
                                      gt_bboxes_ignore,
                                      gt_labels,
                                      input_meta,
                                      num_classes=1,
                                      sampling=True):
        """Assign anchors and encode positive anchors.

        Args:
            bbox_assigner (BaseAssigner): assign positive and negative boxes.
            anchors (torch.Tensor): Concatenated multi-level anchor.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Gt bboxes.
            gt_bboxes_ignore (torch.Tensor): Ignored gt bboxes.
            gt_labels (torch.Tensor): Gt class labels.
            input_meta (dict): Meta info of each image.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple[torch.Tensor]: Anchor targets.
        """

        def assign(bbox_assigner):
            if isinstance(bbox_assigner, ATSS3DAssigner):
                return bbox_assigner.assign(anchors,
                                            gt_bboxes,
                                            gt_bboxes_ignore,
                                            gt_labels,
                                            current_anchor_num=current_anchor_num)
            else:
                assert 1 == 0, f'interface for {bbox_assigner} is not implement in train_mixins'

        current_anchor_num = anchors.shape[-2] * anchors.shape[-3]
        anchors = anchors.reshape(-1, anchors.size(-1))
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        dir_targets = anchors.new_zeros((anchors.shape[0]), dtype=torch.long)
        dir_weights = anchors.new_zeros((anchors.shape[0]), dtype=torch.float)
        labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        if len(gt_bboxes) > 0:
            if not isinstance(gt_bboxes, torch.Tensor):
                gt_bboxes = gt_bboxes.tensor.to(anchors.device)

            pred_instance_3d = InstanceData(priors=anchors)
            assign_result = assign(bbox_assigner)
            if self.echo_mismatch_gt:
                miss_cnt_ = assign_result.num_gts - assign_result.gt_inds.unique().shape[0] + 1
                if miss_cnt_ != 0:
                    print(f"Warning! {input_meta['pts_filename']} have {miss_cnt_} unmatched {gt_labels[0]} gt, please check!")
                    unmatch_gts = set(np.arange(assign_result.num_gts + 1)) - set(assign_result.gt_inds.unique().cpu().numpy())
                    for unmatch_idx_, unmatch_gt_index in enumerate(unmatch_gts):
                        print(f'{unmatch_idx_:02d}: {gt_bboxes[unmatch_gt_index-1]}')
            
            sampling_result = self.bbox_sampler.sample(assign_result, pred_instance_3d, gt_bboxes)
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds
        else:
            pos_inds = torch.nonzero(anchors.new_zeros((anchors.shape[0], ), dtype=torch.bool) > 0,
                                     as_tuple=False).squeeze(-1).unique()
            neg_inds = torch.nonzero(anchors.new_zeros((anchors.shape[0], ), dtype=torch.bool) == 0,
                                     as_tuple=False).squeeze(-1).unique()

        if gt_labels is not None:
            labels += num_classes
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            pos_dir_targets = get_direction_target(sampling_result.pos_bboxes,
                                                   pos_bbox_targets,
                                                   self.dir_offset,
                                                   one_hot=False)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            dir_targets[pos_inds] = pos_dir_targets
            dir_weights[pos_inds] = 1.0

            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        return (labels, label_weights, bbox_targets, bbox_weights, dir_targets, dir_weights, pos_inds, neg_inds)



def get_direction_target(anchors,
                         reg_targets,
                         dir_offset=0,
                         dir_limit_offset=0,
                         num_bins=2,
                         one_hot=True):
    """Encode direction to 0 ~ num_bins-1.

    Args:
        anchors (torch.Tensor): Concatenated multi-level anchor.
        reg_targets (torch.Tensor): Bbox regression targets.
        dir_offset (int): Direction offset.
        num_bins (int): Number of bins to divide 2*PI.
        one_hot (bool): Whether to encode as one hot.

    Returns:
        torch.Tensor: Encoded direction targets.
    """
    rot_gt = reg_targets[..., 6] + anchors[..., 6]
    offset_rot = limit_period(rot_gt - dir_offset, dir_limit_offset, 2 * np.pi)
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    if one_hot:
        dir_targets = torch.zeros(
            *list(dir_cls_targets.shape),
            num_bins,
            dtype=anchors.dtype,
            device=dir_cls_targets.device)
        dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
        dir_cls_targets = dir_targets
    return dir_cls_targets
