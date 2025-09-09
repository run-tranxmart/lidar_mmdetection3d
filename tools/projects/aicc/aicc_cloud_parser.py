# Copyright (c) SoftMotion. All rights reserved.
import argparse
import numpy as np
import os
import os.path as osp
from tqdm import tqdm


def determine_pcd_fields(src_frame_path):
    with open(src_frame_path, "rb") as f:
        header = []
        field_num = -1
        while True:
            ln = f.readline().strip()
            header.append(ln)
            if ln.startswith(b"FIELDS"):
                field_str = ln.strip().split()
                field_num = len(field_str) - 1
                break
    return field_num


def save_cloud(save_path, cloud_points):
    if save_path.endswith(".bin"):
        cloud_points = cloud_points.astype(np.float32)
        cloud_points.tofile(save_path)
    elif save_path.endswith(".pcd"):
        point_type = np.dtype([
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("intensity", np.float32),
        ])
        points = np.zeros(cloud_points.shape[0], dtype=point_type)
        points["x"] = cloud_points[:, 0]
        points["y"] = cloud_points[:, 1]
        points["z"] = cloud_points[:, 2]
        points["intensity"] = cloud_points[:, 3]
        points_num = cloud_points.shape[0]
        # Write the header
        with open(save_path, "w") as fp:
            fp.write(
                "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1"
            )
            fp.write("\nWIDTH " + str(points_num))
            fp.write("\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0")
            fp.write("\nPOINTS " + str(points_num))
            fp.write("\nDATA binary")
            fp.write("\n")
        # Write the points
        with open(save_path, "ab+") as fp:
            pc_data = np.array(points, dtype=point_type)
            fp.write(pc_data.tobytes("C"))
    else:
        raise TypeError("Only support the .pcd and .bin suffix")


def parse_custom_fields4(src_frame_path):
    """PCD Header
    VERSION 0.7
    FIELDS x y z intensity
    SIZE 4 4 4 4
    TYPE F F F F
    COUNT 1 1 1 1
    WIDTH 236250
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS 236250
    DATA ascii
    """
    points = []
    with open(src_frame_path, "rb") as f:
        header = []
        while True:
            ln = f.readline().strip()
            header.append(ln)
            if ln.startswith(b"DATA"):
                break

        points_lines = f.readlines()
        for ln in points_lines:
            pt = np.fromstring(ln, dtype=np.float32, sep=" ")
            if not np.isnan(pt).any():
                points.append(pt)
        points = np.asarray(points, dtype=np.float32)

        # Filter zero values
        points_xyz = points[:, :3]
        points_intensity = points[:, 3][:, np.newaxis]
        zero_index = np.where(points_xyz == [0., 0., 0.])
        points_xyz = np.delete(points_xyz, zero_index, axis=0)
        points_intensity = np.delete(points_intensity, zero_index, axis=0)
        points = np.hstack((points_xyz, points_intensity))
        # Filter NaN values
        points = points[~np.isnan(points).any(axis=1), :]
        # Filter duplicate points
        points = np.unique(points, axis=0)
    return points


def parse_custom_fields15(src_frame_path):
    """PCD Header
    VERSION 0.7
    FIELDS x y z __12 __13 __14 __15 intensity __17 ring __20 __21 __22 __23 timestamp
    SIZE 4 4 4 1 1 1 1 1 1 2 1 1 1 1 8
    TYPE F F F U U U U U U U U U U U F
    COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    WIDTH 236250
    HEIGHT 1
    VIEWPOINT 0.0 0.0 0.0 1.0 0.0 0.0 0.0
    """
    point_dtype = np.dtype([
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("__12", np.uint8),
        ("__13", np.uint8),
        ("__14", np.uint8),
        ("__15", np.uint8),
        ("intensity", np.uint8),
        ("__17", np.uint8),
        ("ring", np.uint16),
        ("__20", np.uint8),
        ("__21", np.uint8),
        ("__22", np.uint8),
        ("__23", np.uint8),
        ("timestamp", np.float64),
    ])
    points = []
    point_num = 0
    with open(src_frame_path, "rb") as f:
        header = []
        while True:
            ln = f.readline().strip()
            header.append(ln)
            if ln.startswith(b"POINTS"):
                point_num = int(ln.split()[-1])
            if ln.startswith(b"DATA"):
                break
        rowstep = point_num * point_dtype.itemsize
        buf = f.read(rowstep)
        src_points = np.frombuffer(buf, dtype=point_dtype)
        points_xyz = np.vstack(
            (src_points["x"], src_points["y"], src_points["z"])).transpose()
        points_intensity = src_points["intensity"].astype(
            np.float32)[:, np.newaxis]
        # Filter zero values
        zero_index = np.where(points_xyz == [0., 0., 0.])
        points_xyz = np.delete(points_xyz, zero_index, axis=0)
        points_intensity = np.delete(points_intensity, zero_index, axis=0)
        points = np.hstack((points_xyz, points_intensity))
        # Filter NaN values
        points = points[~np.isnan(points).any(axis=1), :]
        # Filter duplicate values
        points = np.unique(points, axis=0)
    return points


def parse_custom_pcd(src_path,
                     des_path,
                     save_pcd: bool = True,
                     save_bin: bool = True,
                     z_offset: float = 0.0,
                     rotate: float = 0.0):
    cloud_count = 0
    cloud_list = []
    if os.path.isdir(src_path):
        src_pcd_dir = src_path
        cloud_list = sorted(os.listdir(src_pcd_dir))
        cloud_list = [
            osp.join(src_pcd_dir, cloud_fn) for cloud_fn in cloud_list
        ]
    elif os.path.isfile(src_path):
        cloud_list = [src_path]
    else:
        raise FileNotFoundError(f"Source path {src_path} is not available")
    # create the save directory
    if save_pcd:
        des_pcd_dir = osp.join(des_path, 'cloud_pcd')
        os.makedirs(des_pcd_dir, exist_ok=True)
    if save_bin:
        des_bin_dir = osp.join(des_path, 'cloud_bin')
        os.makedirs(des_bin_dir, exist_ok=True)

    for i in tqdm(range(len(cloud_list))):
        src_pcd_path = cloud_list[i]
        if src_pcd_path.endswith(".pcd"):
            field_num = determine_pcd_fields(src_pcd_path)
            if field_num == 4:
                points = parse_custom_fields4(src_pcd_path)
                points[:, 2] += z_offset
                cloud_count += 1
            elif field_num == 15:
                points = parse_custom_fields15(src_pcd_path)
                points[:, 2] += z_offset
                cloud_count += 1
            else:
                raise TypeError("Unsupport PCD fields")
        else:
            continue

        # rotate the points if need
        if np.fabs(rotate - 90) < 1e-6:
            # clockwise 90 -> x=-y, y=x
            temp_x = points[:, 0].copy()
            temp_y = points[:, 1].copy()
            points[:, 0] = -temp_y
            points[:, 1] = temp_x
        elif np.fabs(rotate + 90) < 1e-6:
            # anti-clockwise 90 -> y=-x, x=y
            temp_x = points[:, 0].copy()
            temp_y = points[:, 1].copy()
            points[:, 0] = temp_y
            points[:, 1] = -1.0 * temp_x
        elif (np.fabs(rotate - 180) < 1e-6) or (np.fabs(rotate + 180) < 1e-6):
            #  clockwise 180
            points[:, 0] = -points[:, 0]
            points[:, 1] = -points[:, 1]
        # save the cloud
        cloud_fn = src_pcd_path.split("/")[-1]
        frame_name = cloud_fn.rsplit(".", 1)[0]
        if save_pcd:
            des_pcd_path = osp.join(des_pcd_dir, frame_name + ".pcd")
            save_cloud(des_pcd_path, points)
        if save_bin:
            des_bin_path = osp.join(des_bin_dir, frame_name + ".bin")
            save_cloud(des_bin_path, points)
    return cloud_count


def parse_args():
    parser = argparse.ArgumentParser(description="Parse custom cloud")
    parser.add_argument(
        "--src_cloud",
        "-s",
        type=str,
        default=None,
        help="Path of source pcd cloud file or directory",
    )
    parser.add_argument(
        "--des_cloud",
        "-d",
        type=str,
        default=None,
        help="Path of destination cloud file or directory",
    )
    parser.add_argument(
        "--z_offset",
        type=float,
        default=0.8,
        help="Z-offset",
    )
    parser.add_argument(
        "--rotate",
        type=float,
        default=-90.0,
        help="Rotate angle (+clockwise, -anticlockwise)",
    )
    parser.add_argument(
        "--save_pcd",
        action="store_true",
        default=False,
        help="Save PCD file",
    )
    parser.add_argument(
        "--save_bin",
        action="store_true",
        default=False,
        help="Save BIN file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    parse_custom_pcd(args.src_cloud, args.des_cloud, args.save_pcd,
                     args.save_bin, args.z_offset, args.rotate)
