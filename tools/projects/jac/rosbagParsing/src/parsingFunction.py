import os
import sys

from pypcd import pypcd
import rosbag
from tqdm import tqdm
from datetime import datetime
import cv2
import struct
import numpy as np
import csv
from math import pi

import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

camera_topic = [
    "/Camera_4",
    "/Camera_5",
    "/Camera_6",
    "/Camera_7",
    "/Camera_8",
    "/Camera_9",
    "/Camera_10",
    "/Camera_11",
]
# camera_topic = ['/Camera_8']
lidar_topics = [
    "/hesai_left_lidar",
    "/hesai_right_lidar",
    "/innovusion_lidar_front",
    "/innovusion_lidar_front2",
    "/innovusion_lidar_front3",
    "/innovusion_lidar_front4",
]
offset_addr = {
    "CX": 0x40,
    "CY": 0x48,
    "FX": 0x50,
    "FY": 0x58,
    "K1": 0xA0,
    "K2": 0xA8,
    "K3": 0xB0,
    "K4": 0xB8,
    "P1": 0xD0,
    "P2": 0xD8,
    "IW": 0xF8,
    "IH": 0xFC,
}


def hello():
    print("hello world")


def save_pic(frame, save_dir, save_type, i):
    # function to save pictures, need to do special process when save_type is yuv
    if save_type == "yuv":
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        with open(save_dir + "/image" + str(i) + "." + save_type, "wb") as f:
            f.write(yuv_frame.tobytes())
    else:
        cv2.imwrite(save_dir + "/image" + str(i) + "." + save_type, frame)


def LidarParsing(bag_path, save_dir, lidar_topics=lidar_topics):
    # parsing lidar data from bag, but i'm not sure that if there are only '/hesai_left_lidar/' in lidar topic of bag
    # if there were other topic, it should be written as a list like camera topic
    try:
        bag = rosbag.Bag(bag_path, "r")
    except Exception as e:
        print(f"wrong with {e}")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for lidar in lidar_topics:
        save_dir_tmp = save_dir + lidar
        if not os.path.exists(save_dir_tmp):
            os.makedirs(save_dir_tmp)

        for topic, msg, t in tqdm(bag.read_messages(topics=lidar)):

            stamp_sec = msg.header.stamp.secs
            stamp_nsec = msg.header.stamp.nsecs
            stamp_nsec = stamp_nsec * 1e-9
            stamp_nsec = "{:.9f}".format(stamp_nsec)
            stamp_nsec = stamp_nsec.split(".")[-1]

            # point cloud data -> numpy array
            gen = pc2.read_points(msg)
            points = np.array(list(gen))
            pcd_data = np.zeros(
                (points.shape[0],),
                dtype=[
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                    ("intensity", np.float32),
                    ("ring", np.int32),
                    ("timestamp", np.float32),
                ],
            )
            pcd_data["x"] = points[:, 0]
            pcd_data["y"] = points[:, 1]
            pcd_data["z"] = points[:, 2]
            try:
                pcd_data["intensity"] = points[:, 3]
            except IndexError:
                print("Points have no intensity attribute")
            try:
                pcd_data["ring"] = points[:, 4].astype(np.int32)  # ring should be int
            except:
                print("Points have no ring attribute")
            try:
                pcd_data["timestamp"] = points[:, 5]
            except:
                print("Points have no timestamp attribute")
            cloud = pypcd.PointCloud.from_array(pcd_data)
            cloud.save_pcd(
                save_dir_tmp + "/" + str(stamp_sec) + "." + stamp_nsec + ".pcd"
            )

            # Save to bin data
            # with open(save_dir_tmp+'/'+str(stamp_sec)+'.' +stamp_nsec+'.bin','wb') as f:
            #     f.write(pcd_data)

            # Point cloud -> numpy array
            # gen = pc2.read_points(msg)
            # points = np.array(list(gen))
            # Create open3d object to get cloud
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            # pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)
            # Save the point-cloud file named as timestamps
            # o3d.io.write_point_cloud(save_dir_tmp+'/'+str(stamp_sec)+'.' +stamp_nsec+'.pcd', pcd)

        print(f"ok for {lidar} parsing")


def Mp4Parsing(
    bag_path, save_dir, save_type=None, distortion=None, camera_topic=camera_topic
):
    # read head of mp4 and collect data from bag to produce a mp4 file
    # and depending the parameter to decide do save picture and undistortion or not
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open("./src/overlay.h265", "rb") as f:
        header = f.read()

    try:
        bag = rosbag.Bag(bag_path, "r")
    except Exception as e:
        print(f"wrong with {e}")

    datetime_obj = datetime.fromtimestamp(bag.get_start_time())
    time_str = datetime_obj.strftime("%Y-%m-%d-%H-%M-%S")

    res = {}
    stamp = {}
    print("collecting the mp4 data")
    for topic, msg, t in tqdm(bag.read_messages()):
        if topic in camera_topic:

            if topic not in res.keys():
                res[topic] = b""
                res[topic] = header
                # res[topic] += msg.data
            else:
                res[topic] += msg.data

            if len(str(msg.header.stamp.nsecs)) == 8:
                length_with_zero = "0" + str(msg.header.stamp.nsecs)
            elif len(str(msg.header.stamp.nsecs)) == 7:
                length_with_zero = "00" + str(msg.header.stamp.nsecs)
            elif len(str(msg.header.stamp.nsecs)) == 6:
                length_with_zero = "000" + str(msg.header.stamp.nsecs)
            else:
                length_with_zero = str(msg.header.stamp.nsecs)

            if topic not in stamp.keys():
                stamp[topic] = []
                stamp[topic].append(str(msg.header.stamp.secs) + "." + length_with_zero)
            else:
                stamp[topic].append(str(msg.header.stamp.secs) + "." + length_with_zero)

    # print(res.keys())
    for key in res.keys():
        file_path = save_dir + key
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        file_path_tmp = file_path + "/" + time_str + ".mp4"
        with open(file_path_tmp, "wb") as f:
            f.write(res[key])
        if save_type is not None:
            if distortion is not None:
                # print(file_path_tmp, file_path, save_type, key[1:])
                PicParsing(
                    file_path_tmp, file_path, save_type, stamp[key], disortion=key[1:]
                )
            else:
                PicParsing(file_path_tmp, file_path, save_type, stamp[key])


def PicParsing(mp4_path, save_dir, save_type, timestamp, disortion=None):
    # use a while loop, its too slow, and can't get pre infomation so it can not apply multi thread to speed up it
    cap = cv2.VideoCapture(mp4_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if cap.isOpened():
        ret = True
        i = 0
        while ret:
            try:
                time = timestamp[i]
            except:
                time = i
            i += 1
            ret, frame = cap.read()
            if frame is not None:
                if disortion is None:
                    save_pic(frame, save_dir, save_type, time)
                else:
                    result = read_internal_data_from_bin(
                        "./src/car_camera_internal_data/" + disortion + ".bin",
                        offset_addr,
                    )
                    undistorted_img = _distortion(frame, result)
                    save_pic(undistorted_img, save_dir, save_type, time)
                print(f"save the {disortion} {i} pictures", end="\r")
    else:
        raise ValueError("video file can not be opened")
    print("")
    print("ok", save_dir)


def read_internal_data_from_bin(file_name, offsets):
    # read camera internal data
    base_addr = 0x835
    with open(file_name, "rb") as file:
        result = {}
        for key, offset in offsets.items():
            file.seek(base_addr + offset)
            if key == "IW" or key == "IH":
                data_bytes = file.read(4)
                data_float = struct.unpack("i", data_bytes)[0]
            else:
                data_bytes = file.read(8)
                data_float = struct.unpack("d", data_bytes)[0]
            result[key] = data_float
        return result


def _distortion(frame, result):
    # input a img(numpy) output a undistortion img(numpy)
    camera_matrix = np.array(
        [[result["FX"], 0, result["CX"]], [0, result["FY"], result["CY"]], [0, 0, 1]]
    )
    distortion = np.array(
        [
            result["K1"],
            result["K2"],
            result["P1"],
            result["P2"],
            result["K3"],
            result["K4"],
            0,
            0,
        ]
    )  # example values
    optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion, frame.shape[1::-1], 1, frame.shape[1::-1]
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, distortion, None, optimal_camera_matrix, frame.shape[1::-1], 5
    )
    undistorted_img = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    return undistorted_img


def distortion(filepath, save_path, internal):
    # undistortion function
    result = read_internal_data_from_bin(
        "./src/car_camera_internal_data/" + internal + ".bin", offset_addr
    )
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if os.path.isdir(filepath):
        supported_formats = (".jpg", ".png")
        for root, dirs, files in os.walk(filepath):
            for file in tqdm(files):
                if file.endswith(supported_formats):
                    frame = cv2.imread(os.path.join(root, file))
                    undistorted_img = _distortion(frame, result)
                    cv2.imwrite(save_path + "/undistorted_" + file, undistorted_img)
    elif os.path.isfile(filepath):
        frame = cv2.imread(filepath)
        undistorted_img = _distortion(frame, result)

        cv2.imwrite(save_path + "/undistorted_img.png", undistorted_img)


def canParsing(bag_path, save_dir):

    csv_first_row = "gps_seconds,gps_nseconds,lat(deg),log(deg),alt(m),vn(m/s),ve(m/s),vu(m/s),roll(rad),pitch(rad),yaw(rad),ins_status\n"

    try:
        bag = rosbag.Bag(bag_path, "r")
    except Exception as e:
        print(f"wrong with {e}")
        return

    bag_name = bag_path.split("/")[-1].split(".")[0]
    save_path = save_dir + "/" + bag_name

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + "/ins_info.csv", "w") as f:
        writer = csv.writer(f)
        f.write(csv_first_row)
        for topic, msg, t in tqdm(bag.read_messages(topics="/can_data")):
            # value the csv need
            gps_seconds = msg.INS_SOL_RESULT_STATUS_Signals.timestamp_s
            gps_nseconds = msg.INS_SOL_RESULT_STATUS_Signals.timestamp_ns
            lat = msg.INS_SOL_RESULT_LATITUDE_Signals.INS_latitude
            log = msg.INS_SOL_RESULT_LONGITUDE_Signals.INS_longitude
            alt = msg.INS_SOL_RESULT_ALTITUDE_Signals.INS_altitude
            vn = msg.INS_SOL_RESULT_VEL_Signals.INS_velocityN
            ve = msg.INS_SOL_RESULT_VEL_Signals.INS_velocityE
            vu = msg.INS_SOL_RESULT_VEL_Signals.INS_velocityU
            roll = msg.INS_SOL_RESULT_ANGLE_Signals.INS_roll * pi / 180
            pitch = msg.INS_SOL_RESULT_ANGLE_Signals.INS_pitch * pi / 180
            yaw = msg.INS_SOL_RESULT_ANGLE_Signals.INS_yawAngle * pi / 180
            ins_status = msg.INS_SOL_RESULT_STATUS_Signals.INS_insStatus

            csv_data = [
                gps_seconds,
                gps_nseconds,
                lat,
                log,
                alt,
                vn,
                ve,
                vu,
                roll,
                pitch,
                yaw,
                ins_status,
            ]
            writer.writerow(csv_data)


if __name__ == "__main__":
    bag_path = "/root/data/2024-04-29-16-30-03.bag"
    save_dir = "/root/data/test"

    # LidarParsing(bag_path, save_dir)
    # Mp4Parsing(bag_path, save_dir)
    # distortion('/root/rosbagParsing/test/Camera_5', './test/result', 'Camera_5')
    bag = rosbag.Bag(bag_path, "r")
    a = {}
    for topic, msg, t in tqdm(bag.read_messages()):
        a[topic] = 1
    print(a)
