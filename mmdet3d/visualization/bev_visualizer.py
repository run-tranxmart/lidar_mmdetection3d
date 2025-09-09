# Copyright (c) SoftMotion. All rights reserved.
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from typing import List, Optional, Sequence, Tuple, Union

import mmcv
from mmdet.visualization import DetLocalVisualizer
from mmengine.dist import master_only
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer as MMENGINE_Visualizer
from mmengine.visualization.utils import tensor2ndarray
from mmdet3d.registry import VISUALIZERS
from mmdet3d.structures import (Det3DDataSample, LiDARInstance3DBoxes)


@VISUALIZERS.register_module()
class BEVLocalVisualizer(DetLocalVisualizer):
    """BEV Local Visualizer.
    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): The origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (List[dict], optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
            Defaults to None.
        bbox_color (str or Tuple[int], optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        text_color (str or Tuple[int]): Color of texts. The tuple of color
            should be in BGR order. Defaults to (200, 200, 200).
        mask_color (str or Tuple[int], optional): Color of masks. The tuple of
            color should be in BGR order. Defaults to None.
        line_width (int or float): The linewidth of lines. Defaults to 2.
        frame_cfg (dict): The coordinate frame config while Open3D
            visualization initialization.
            Defaults to dict(size=1, origin=[0, 0, 0]).
        alpha (int or float): The transparency of bboxes or mask.
            Defaults to 0.8.
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[List[dict]] = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = None,
                 text_color: Union[str, Tuple[int]] = (200, 200, 200),
                 mask_color: Optional[Union[str, Tuple[int]]] = None,
                 line_width: Union[int, float] = 3,
                 alpha: Union[int, float] = 0.8,
                 class_names: Optional[List[str]] = None,
                 score_thresh: Optional[List[float]] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                 voxel_size: Optional[Tuple[float]] = (0.12, 0.12, 0.2),
                 area_scope: Optional[List[float]] = [[-72, 92], [-72, 72], [-5, 5]],
                 det_scope: Optional[List[float]] = [[-60, 60], [-30, 30]],
                 cloud_rotation: float = 0.0):
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            line_width=line_width,
            alpha=alpha)
        # (dx, dy, dz)
        self.voxel_size = voxel_size
        # [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        self.area_scope = np.array(area_scope)
        self.cloud_rotation = cloud_rotation

        # set the detection scope
        self.det_scope = np.array(det_scope).flatten()
        if len(self.det_scope) == 6:
            '''
                [xmin, ymin, zmin, xmax, ymax, zmax] ->
                [[xmin, ymin, zmin],
                 [xmax, ymax, zmax]] ->
                [[xmin, ymin]
                 [xmax, ymax]]
            '''
            self.det_scope = self.det_scope.reshape(2, 3)
            self.det_scope = self.det_scope[:, :2]
            self.det_scope = self.det_scope.T
        elif len(self.det_scope) == 4:
            self.det_scope = self.det_scope.reshape(2, 2)
        else:
            raise ValueError('Unsupported detection scope')

        # load the class_names
        self.class_names = class_names
        if class_names is None:
            self.class_names = ['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Misc']
        self.class2id, self.id2class = {}, {}
        for cls_id, cls_name in enumerate(self.class_names):
            self.class2id[cls_name] = cls_id
            self.id2class[cls_id] = cls_name

        # get the score thresholds
        self.score_thresh = score_thresh
        if isinstance(score_thresh, float):
            self.score_thresh = [score_thresh] * len(self.class_names)
        elif isinstance(score_thresh, List):
            if len(score_thresh) < len(self.class_names):
                raise IndexError(
                    f'Input score thresholds {len(score_thresh)} less than \
                    class number {len(self.class_names)}')
            else:
                self.score_thresh = score_thresh[:len(self.class_names)]
        else:
            raise TypeError(
                f'Unsupported input thresholds type: {type(score_thresh)}')

        # define the palatte
        self.palette = [
            [255, 51, 51],  # 0 Pedestrian: Red
            [255, 204, 45],  # 1 Cyclist: Gold Orange
            [68, 255, 117],  # 2 Car: Green
            [142, 118, 255],  # 3 Truck: Purple
            [224, 224, 224],  # 4 Misc: Light Grey
            [224, 224, 224],  # 5 Unknown: Light Grey
            [190, 190, 190],  # 6 DontCare: Grey
            [255, 215, 0],  # 7 Traffic_Warning_Object: Gold
            [255, 192, 203],  # 8 Traffic_Warning_Sign: Pink
            [255, 127, 36],  # 9 Road_Falling_Object: Chocolate1
            [255, 64, 64],  # 10 Road_Intrusion_Object: Brown1
            [255, 0, 255],  # 11 Animal: Magenta
        ]

    def _draw_instances_bev(self,
                            bev_map,
                            instances: InstanceData,
                            crowd: List = [],
                            draw_score: bool = False,
                            thickness: int = 2) -> dict:
        """Draw BEV instances of GT or prediction.

        Args:
            data_input (dict): The input dict to draw.
            instances (:obj:`InstanceData`): Data structure for instance-level
                annotations or predictions.
            thickness: box thickness

        Returns:
            dict: The drawn point cloud and image whose channel is RGB.
        """
        # Only visualize when there is at least one instance
        if not len(instances) > 0:
            return None
        if crowd is None:
            crowd = []
        bboxes_3d = instances.bboxes_3d  # LiDARInstance3DBoxes
        labels_3d = tensor2ndarray(instances.labels_3d)
        scores_3d = tensor2ndarray(instances.scores_3d) if hasattr(instances, 'scores_3d') and instances.scores_3d is not None else None
        corners_3d = tensor2ndarray(bboxes_3d.corners)
        centers_3d = tensor2ndarray(bboxes_3d.center)
        if len(crowd) > 0:
            crowd = np.array(crowd, dtype=np.int16)
            if len(labels_3d) != len(crowd):
                raise IndexError(f'labels {len(labels_3d)} != \
                    crowd {len(crowd)}')
        corners_bev = self._corners3d_to_bev_corners(corners_3d)
        centers_pixel = self._pt2pix(centers_3d)

        for cur_label in range(labels_3d.min(), labels_3d.max() + 1):
            if scores_3d is None:
                mask = (labels_3d == cur_label)
            else:
                mask = (labels_3d == cur_label) & (
                    scores_3d > self.score_thresh[cur_label])
            if mask.sum() == 0:
                continue
                        
            bev_color_map = np.asarray(self.palette)
            bev_color_map = bev_color_map[:, [2, 1, 0]]
            if len(labels_3d) == len(crowd):
                cur_crowd = crowd[mask]
            else:
                cur_crowd = []
            bev_map = self._draw_box_lines(
                bev_map,
                corners_bev[mask],
                # cur_crowd,
                color=tuple(bev_color_map[cur_label]),
                thickness=thickness)
            if draw_score and (scores_3d is not None):
                bev_map = self._draw_box_scores(bev_map,
                                                centers_pixel[mask],
                                                scores_3d[mask])

        return bev_map

    def _boxes3d_to_corners3d(self, boxes3d):
        """
        :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
        :param rotate:
        :return: corners3d: (N, 8, 3)
        """
        boxes_num = boxes3d.shape[0]
        w, l, h = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
        x_corners = np.array([
            w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.
        ],
                             dtype=np.float32).T
        y_corners = np.array([
            -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2.
        ],
                             dtype=np.float32).T
        z_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        z_corners[:, 4:8] = h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)

        ry = boxes3d[:, 6]
        zeros, ones = np.zeros(
            ry.size, dtype=np.float32), np.ones(
                ry.size, dtype=np.float32)
        rot_list = np.array([[np.cos(ry), -np.sin(ry), zeros],
                             [np.sin(ry), np.cos(ry), zeros],
                             [zeros, zeros, ones]])  # (3, 3, N)
        R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

        temp_corners = np.concatenate(
            (x_corners.reshape(-1, 8, 1), y_corners.reshape(
                -1, 8, 1), z_corners.reshape(-1, 8, 1)),
            axis=2)  # (N, 8, 3)
        rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
        x_corners, y_corners, z_corners = rotated_corners[:, :,
                                                          0], rotated_corners[:, :,
                                                                              1], rotated_corners[:, :,
                                                                                                  2]

        x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

        x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
        y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
        z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

        corners = np.concatenate(
            (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)),
            axis=2)

        return corners.astype(np.float32)

    def _corners3d_to_bev_corners(self, corners3d):
        """
        :param corners3d: (N, 8, 3)
        :return:
            bev_corners: (N, 4, 2)
        """
        voxel_idxs = np.floor(corners3d[:, :, 0:3] / self.voxel_size).astype(
            np.int32)

        min_voxel_coords = np.floor(self.area_scope[:, 0] /
                                    self.voxel_size).astype(np.int32)
        max_voxel_coords = np.ceil(self.area_scope[:, 1] / self.voxel_size -
                                   1).astype(np.int32)
        voxelized_shape = ((max_voxel_coords - min_voxel_coords) + 1).astype(
            np.int32)

        # Check the points are bounded by the image scope
        # assert (min_voxel_coords <= np.amin(voxel_idxs, axis=0)).all(), 'Shape: %s' % (str(voxel_idxs.shape))
        # assert (max_voxel_coords >= np.amax(voxel_idxs, axis=0)).all(), 'Shape: %s' % (str(voxel_idxs.shape))
        voxel_idxs = voxel_idxs - min_voxel_coords

        select_idxs = [0, 2, 6, 4]
        # voxel_idxs = voxel_idxs[:, 0:4, 0:2]
        voxel_idxs = voxel_idxs[:, select_idxs, 0:2]
        x_idxs, y_idxs = voxel_idxs[:, :, 0].copy(), voxel_idxs[:, :, 1].copy()
        voxel_idxs[:, :, 0] = voxelized_shape[1] - y_idxs
        voxel_idxs[:, :, 1] = voxelized_shape[0] - x_idxs
        return voxel_idxs

    def _pt2pix(self, pt):
        """
        :param corners3d: (N, 3)
        :return:
            bev_corners: (N, 2)
        """
        voxel_size = np.array(self.voxel_size)
        area_scope = np.array(self.area_scope)

        voxel_idxs = np.floor(pt / voxel_size).astype(np.int32)

        min_voxel_coords = np.floor(area_scope[:, 0] / voxel_size).astype(
            np.int32)
        max_voxel_coords = np.ceil(area_scope[:, 1] / voxel_size - 1).astype(
            np.int32)
        voxelized_shape = ((max_voxel_coords - min_voxel_coords) + 1).astype(
            np.int32)

        # Check the points are bounded by the image scope
        voxel_idxs = voxel_idxs - min_voxel_coords
        voxel_idxs = voxel_idxs[:, 0:2]
        x_idxs, y_idxs = voxel_idxs[:, 0].copy(), voxel_idxs[:, 1].copy()
        voxel_idxs[:, 0] = voxelized_shape[1] - y_idxs
        voxel_idxs[:, 1] = voxelized_shape[0] - x_idxs
        return voxel_idxs

    def draw_distance_circle(self,
                             image,
                             circle_delta=10,
                             circle_range=None,
                             color=(200, 200, 200),
                             thickness=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        circle_center = np.array([[0., 0., 0.]], dtype=np.float32)
        circle_center = self._pt2pix(circle_center)
        center_x, center_y = circle_center[0, 0], circle_center[0, 1]
        if circle_range is None:
            circle_range = abs(np.asarray(self.area_scope)).max()
        for rad in np.arange(
                circle_delta,
                circle_delta + circle_range,
                circle_delta,
                dtype=np.float32):
            circle_point = np.array([[rad, 0, 0]], dtype=np.float32)
            circle_point = self._pt2pix(circle_point)
            circle_point_x = circle_point[0, 0]
            circle_point_y = circle_point[0, 1]
            circle_radius = np.abs(circle_point_y - center_y)
            circle_text = str(int(rad)) + 'm'
            cv2.circle(
                image, (center_x, center_y),
                circle_radius,
                color,
                thickness=thickness)
            cv2.putText(image, '%s' % circle_text,
                        (circle_point_x - 10, circle_point_y), font, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)
        return image

    def _draw_dashed_line(self,
                          img,
                          start_point,
                          end_point,
                          color,
                          thickness=1,
                          dash_length=20):
        d = np.linalg.norm(np.array(end_point) - np.array(start_point))
        dashes = int(d / dash_length)
        for i in range(dashes):
            start = start_point + (np.array(end_point) -
                                   np.array(start_point)) * (
                                       i / dashes)
            end = start_point + (np.array(end_point) -
                                 np.array(start_point)) * ((i + 1) / dashes)
            if i % 2 == 0:
                cv2.line(img, tuple(np.int32(start)), tuple(np.int32(end)),
                         color, thickness)

    def draw_bev_det_scope(self,
                           image,
                           dashed=True,
                           color=(255, 105, 180),
                           thickness=1):
        det_box = np.array([
            [self.det_scope[0][0], self.det_scope[1][0], 0.],
            [self.det_scope[0][1], self.det_scope[1][1], 0.],
        ],
                           dtype=np.float32)
        det_box = self._pt2pix(det_box)
        x1, y1 = det_box[0][0], det_box[0][1]
        x2, y2 = det_box[1][0], det_box[1][1]
        if dashed:
            self._draw_dashed_line(image, (x1, y1), (x1, y2), color, thickness)
            self._draw_dashed_line(image, (x1, y2), (x2, y2), color, thickness)
            self._draw_dashed_line(image, (x2, y2), (x2, y1), color, thickness)
            self._draw_dashed_line(image, (x2, y1), (x1, y1), color, thickness)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        return image

    def _draw_box_scores(self,
                         image,
                         box_centers,
                         box_scores,
                         center_offset=(-10, 0),
                         color=(255, 255, 255),
                         font=cv2.FONT_HERSHEY_SIMPLEX,
                         size=0.4,
                         thickness=1):
        for k in range(box_centers.shape[0]):
            center = np.round(box_centers[k, :]).astype(np.int32)
            center += center_offset
            plot_score = format(box_scores[k], '.3f')
            cv2.putText(image, text=plot_score, org=tuple(center), 
                        fontFace=font, fontScale=size,
                        color=color, thickness=thickness,
                        lineType=cv2.LINE_AA)
        return image

    def _draw_box_lines(self,
                        image,
                        bev_corners,
                        crowd=[],
                        color=(0, 255, 0),
                        thickness=2,
                        arrow=True):
        """
        :param image: (H, W)
        :param bev_corners: (N, 4, 2)
        :return:
        """
        # font = cv2.FONT_HERSHEY_SIMPLEX
        color = tuple([int(x) for x in color])
        for k in range(bev_corners.shape[0]):
            crossline = False
            if len(crowd) == bev_corners.shape[0]:
                if crowd[k]:
                    crossline = True
            for j in range(0, 4):
                x1, y1 = bev_corners[k, j, 0], bev_corners[k, j, 1]
                x2, y2 = bev_corners[k, (j + 1) % 4, 0], bev_corners[k, (j + 1) % 4, 1]
                cv2.line(image, (x1, y1), (x2, y2), color, thickness=thickness)
            if crossline:
                for j in range(0, 2):
                    x1, y1 = bev_corners[k, j, 0], bev_corners[k, j, 1]
                    x2, y2 = bev_corners[k, (j + 2) % 4, 0], bev_corners[k, (j + 2) % 4, 1]
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness=thickness) 

            center = bev_corners[k].mean(axis=0)

            head = (bev_corners[k][2] + bev_corners[k][3]) / 2
            head = (head - center) * 1.5 + center
            center = np.round(center).astype(np.int32)
            head = np.round(head).astype(np.int32)
            if (arrow) and (not crossline):
                image = cv2.arrowedLine(
                    image,
                    tuple(center),
                    tuple(head),
                    color,
                    thickness=thickness)

        return image

    def get_part_color_by_offset(self, pts_offset):
        """
        :param pts_offset: (N, 3) offset in xyz, 0~1
        :return:
        """
        color_st = np.array([60, 60, 60], dtype=np.uint8)
        color_ed = np.array([230, 230, 230], dtype=np.uint8)
        pts_color = pts_offset * (color_ed - color_st).astype(
            np.float32) + color_st.astype(np.float32)
        pts_color = np.clip(np.round(pts_color), a_min=0, a_max=255)
        pts_color = pts_color.astype(np.uint8)
        return pts_color

    def _create_scope_filter(self, pts):
        """
        :param pts: (N, 3) point cloud in rect camera coords
        :area_scope: (3, 2), area to keep [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
        """
        pts = pts.transpose()
        x_scope = self.area_scope[0]
        y_scope = self.area_scope[1]
        z_scope = self.area_scope[2]
        scope_filter = (pts[0] > x_scope[0]) & (pts[0] < x_scope[1]) \
                        & (pts[1] > y_scope[0]) & (pts[1] < y_scope[1]) \
                        & (pts[2] > z_scope[0]) & (pts[2] < z_scope[1])

        return scope_filter

    def _convert_pts_to_bev_map(self, points):
        """
        :param pts: (N, 3 or 4) point cloud in rect camera coords
        """
        voxel_size = np.array(self.voxel_size)
        area_scope = np.array(self.area_scope)
        scope_filter = self._create_scope_filter(points)
        pts_val = points[scope_filter]
        voxel_idxs = np.floor(pts_val[:, 0:3] / voxel_size).astype(np.int32)

        min_voxel_coords = np.floor(area_scope[:, 0] / voxel_size).astype(
            np.int32)
        max_voxel_coords = np.ceil(area_scope[:, 1] / voxel_size - 1).astype(
            np.int32)

        # Check the points are bounded by the image scope
        assert (min_voxel_coords <= np.amin(voxel_idxs,
                                            axis=0)).all(), 'Shape: %s' % (
                                                str(voxel_idxs.shape))
        assert (max_voxel_coords >= np.amax(voxel_idxs,
                                            axis=0)).all(), 'Shape: %s' % (
                                                str(voxel_idxs.shape))
        voxel_idxs = voxel_idxs - min_voxel_coords

        voxelized_shape = ((max_voxel_coords - min_voxel_coords) + 1).astype(
            np.int32)
        if points.shape[1] == 4:
            voxelized_shape[2] += 1  # intensity channel

        voxel_map = np.zeros(voxelized_shape, dtype=np.float32)
        voxel_map[voxel_idxs[:, 0], voxel_idxs[:, 1], voxel_idxs[:, 2]] = 1.0

        if points.shape[1] >= 4:
            intensity = points[:, 3]
            # if np.amax(intensity) <= 1.0:
            #     intensity *= 255.0
            intensity = intensity[scope_filter]
            for i in range(intensity.shape[0]):
                # Save biggest intensity in each voxel
                if intensity[i] > voxel_map[voxel_idxs[i, 0], voxel_idxs[i, 1],
                                            -1]:
                    voxel_map[voxel_idxs[i, 0], voxel_idxs[i, 1],
                              -1] = intensity[i]

        voxel_map = np.flip(np.flip(voxel_map, axis=0), axis=1)
        return voxel_map

    def draw_points(self, points, draw_intensity=False):
        voxel_map = self._convert_pts_to_bev_map(points)
        bev_map = voxel_map.sum(axis=2)
        if draw_intensity:
            # Get the intensity map from last channel
            intensity_map = voxel_map[:, :, -1]
            intensity_map = intensity_map.astype(np.uint8)
            intensity_map = cv2.equalizeHist(intensity_map)

            cmap = plt.get_cmap('hot')
            bev_map = cmap(intensity_map)
            bev_map = np.delete(bev_map, 3, axis=2)
            bev_map[:, :, [0, 1, 2]] = bev_map[:, :, [2, 1, 0]] * 255
            bev_map = bev_map.astype(np.uint8)
        else:
            bev_index = bev_map > 0
            bev_map = np.zeros([bev_map.shape[0], bev_map.shape[1], 3],
                               dtype=np.uint8)
            bev_map[bev_index] = (228, 197, 85)
        return bev_map

    @master_only
    def add_datasample(self,
                       name: str,
                       data_input: dict,
                       data_sample: Optional[Det3DDataSample] = None,
                       data_crowd: Optional[List[float]] = None,
                       pred_score_thr: Optional[Union[float, List[float]]] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       draw_score: bool = True,
                       draw_distance: bool = True,
                       draw_det_scope: bool = True,
                       show: bool = False,
                       wait_time: float = 0,
                       save_file: Optional[str] = None,
                       vis_task: str = 'lidar_det',
                       draw_intensity: bool = False) -> None:
        """Draw datasample and save to all backends.
        - If GT and prediction are plotted at the same time, the GT bounding-boxes
        are plotted by green color, and the predicted bounding-boxes are plotted 
        in orange color. 
        - If ``show`` is True, all storage backends are ignored, and the images
          will be displayed in a local window.
        - If ``save_file`` is specified, the drawn image will be saved to
          ``save_file``. It is usually used when the display is not available.

        Args:
            name (str): The image identifier.
            data_input (dict): It should include the point clouds or image
                to draw.
            data_sample (:obj:`Det3DDataSample`, optional): Prediction
                Det3DDataSample. Defaults to None.
            data_crowd (List, optional): crowd flag. Defaults to None
            pred_score_thr (float) : indicate the score threshold instead
                of the initialized values
            draw_gt (bool): Whether to draw GT Det3DDataSample.
                Defaults to True.
            draw_pred (bool): Whether to draw Prediction Det3DDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn point clouds and image.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            save_file (str, optional): Path to output file. Defaults to None.
            vis_task (str): Visualization task. Defaults to 'mono_det'.
            draw_intensity (bool): Whether to show RGB point cloud based on 
                the intensity. Defaults to False.
        """

        assert vis_task in ('lidar_det'), f'unexpected vis_task {vis_task}.'
        # Step 0: if given pred_score_thr, update the score thresholds
        if pred_score_thr is not None:
            # get the score thresholds
            if isinstance(pred_score_thr, float):
                self.score_thresh = [pred_score_thr] * len(self.class_names)
            elif isinstance(pred_score_thr, List):
                if len(pred_score_thr) < len(self.class_names):
                    raise IndexError(
                        f'Input score thresholds {len(pred_score_thr)} less\
                            than class number {len(self.class_names)}')
                else:
                    self.score_thresh = pred_score_thr[:len(self.class_names)]
            else:
                raise TypeError(
                    f'Unsupported input thresholds type: {type(pred_score_thr)}'
                )

        # Step 1: draw points on BEV map
        bev_map = self.draw_points(
            data_input['points'], draw_intensity=draw_intensity)

        # Step 2: draw distance circle and detectino scope on BEV map
        if draw_distance:
            bev_map = self.draw_distance_circle(bev_map)
        if draw_det_scope:
            bev_map = self.draw_bev_det_scope(bev_map)

        # Step 3: draw detection scope on BEV map
        if data_sample is not None:
            if draw_gt and draw_pred:
                pass
            elif draw_pred:
                if 'pred_instances_3d' in data_sample:
                    pred_instances_3d = data_sample.pred_instances_3d
                    self._draw_instances_bev(bev_map, pred_instances_3d, 
                                             data_crowd, draw_score=draw_score)
            elif draw_gt:
                if 'gt_instances_3d' in data_sample:
                    gt_instances_3d = data_sample.gt_instances_3d
                    self._draw_instances_bev(bev_map, gt_instances_3d,
                                             data_crowd, draw_score=False)

        # Step 4: show the image in local window
        if show:
            mmcv.imshow(bev_map, win_name=name, wait_time=wait_time)

        # Step 5: save the bev image
        if save_file is not None:
            # check the suffix of the name of image file
            if not (save_file.endswith('.png') or save_file.endswith('.jpg')):
                save_file = f'{save_file}.png'
            mmcv.imwrite(bev_map, save_file)
