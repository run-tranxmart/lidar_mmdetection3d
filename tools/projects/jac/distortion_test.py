import os
import pandas
import warnings
import argparse
import json
import numpy as np
import open3d as o3d
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from pyproj import Proj

utm_zone = '51' # ‘51r？’
_myProj = Proj('+proj=utm +zone={} +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs'.format(utm_zone))

def lnglat_to_utm(points):
	if isinstance(points, list):
		pts = np.array(points)
		x, y = _myProj(pts[:, 0], pts[:, 1])
		return list(zip(x, y))
	elif isinstance(points, np.ndarray):
		x, y = _myProj(points[:, 0], points[:, 1])
		x = x.reshape(-1, 1)
		y = y.reshape(-1, 1)
		utm_points = np.concatenate((x, y), axis = 1)
		return utm_points
	else:
		raise TypeError

def load_novatel_csv(csv_path):
	'''
	return a dict wich time as key a (4, 4), matrix as item for each key
	'''
	csv_data = pandas.read_csv(csv_path)
	csv_stime = csv_data['gps_seconds'].to_list()
	csv_nstime = csv_data['gps_nseconds'].to_list()
	gps_time = []

	for i in range(len(csv_stime)):
		gps_time.append(csv_stime[i] + csv_nstime[i]*1e-9)

	csv_lng = csv_data['log(deg)'].to_numpy()[:, np.newaxis]
	csv_lat = csv_data['lat(deg)'].to_numpy()[:, np.newaxis]
	csv_alt = csv_data['alt(m)'].to_numpy()[:, np.newaxis]
	csv_roll = csv_data['roll(rad)'].to_numpy()[:, np.newaxis]
	csv_pitch = csv_data['pitch(rad)'].to_numpy()[:, np.newaxis]
	csv_yaw = csv_data['yaw(rad)'].to_numpy()[:, np.newaxis]
	csv_eulr = np.hstack((csv_roll, csv_pitch, csv_yaw))

	# Transform the LLA to ENU
	csv_lnglat = np.hstack((csv_lng, csv_lat))
	csv_utm = lnglat_to_utm(csv_lnglat)
	csv_enu = np.hstack((csv_utm, csv_alt))

	gps_mat_dict = {}

	# Transform the roll, pitch and yaw to matrix
	for i in range(csv_roll.shape[0]):
		transformation_matrix = np.zeros((4, 4), dtype = np.float32)
		rotation_matrix = R.from_rotvec(csv_eulr[i, :]).as_matrix()
		transformation_matrix[:3, :3] = rotation_matrix
		transformation_matrix[:3, 3] = csv_enu[i, :]
		transformation_matrix[3, 3] = 1.0
		gps_mat_dict[gps_time[i]] = transformation_matrix

	return gps_mat_dict
	
def load_calib(lidar):
	'''
	return a (4, 4) matrix
	'''
	calib_mat_dict = {'innovusion_lidar_front': 
       np.array([[-0.0104882, -0.0356707, 0.9993086, 1.975],
                 [0.0003945, -0.9993636, -0.0356685, 0.0],
                 [0.9999449, 0.0000202, 0.0104956, 1.88774],
                 [0.0, 0.0, 0.0, 1.0]]),
       'hesai_right_lidar': 
       np.array([[0.00603560, 0.99992682, -0.01048769, 1.99275241],
                 [-0.99998168, 0.00604009, 0.00039471, -0.50544088],
                 [0.00045810, 0.01048511, 0.99994491, 0.94497671],
                 [0.0, 0.0, 0.0, 1.0]]),
       'hesai_left_lidar':
       np.array([[-0.00424963, -0.99993611, -0.01048821, 1.88206128],
                 [0.99999098, -0.00425402, 0.00039458, 0.62278553],
                 [-0.00043924, -0.01048644, 0.99994490, 0.89401779],
                 [0.0, 0.0, 0.0, 1.0]]),
       'innovusion_lidar_front2':
       np.array([[-0.01958916, -0.02855625, -0.99940099, -0.30532075],
                 [0.02282286, 0.99931838, -0.02900174, -0.04743182],
                 [0.99954770, -0.02337738, -0.01892414, 1.60583384],
                 [0.0, 0.0 ,0.0 ,1.0]]),
        'innovusion_lidar_front3':
        np.array([[-0.02304702, 0.84410612, -0.53568182, 0.64519125],
                  [0.01388421 ,0.53604188 ,0.84407634 ,0.56403038],
                  [0.99963815 ,0.01201595, -0.02407402, 1.60983901],
                  [0.0, 0.0, 0.0, 1.0]]),
        'innovusion_lidar_front4':
        np.array([[-0.011806, -0.844124 ,-0.536018, 0.758444],
                  [0.0753758 ,0.533779 ,-0.842258, -0.612709],
                  [0.997085 ,-0.0503465, 0.0573247, 1.778417],
                  [0.0, 0.0 ,0.0 ,1.0]])}

	if lidar in calib_mat_dict.keys():
		# cause we get the lidar2imu matrix, so we need to apply Inverse to get the gnss2lidar matrix
		# question, dose it really? the result that don't apply inverse are better than apply
		# gnss2lidar_mat = np.linalg.inv(calib_mat_dict[lidar])
		gnss2lidar_mat = calib_mat_dict[lidar]
	else:
		gnss2lidar_mat = np.array(
					[[1, 0, 0, 0],
					[0, 1, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1]], dtype=np.float32
				)
	return gnss2lidar_mat

def get_pose_mat(pose_dict, quety_time):
	# output: the closest pose dict and closest timestamp index
	pose_time_array = np.array(list(pose_dict.keys()))
	timestamp_interval = np.fabs(pose_time_array - quety_time)
	closest_index = np.argmin(timestamp_interval)
	closest_timestamp = pose_time_array[closest_index]
	# print(closest_timestamp)
	return pose_dict[closest_timestamp], closest_index

def load_cloud(cloud_path):
	'''
		Load the cloud to the nx4 array
		return a (n, 4) numpy data. the 4 means x, y, z, intensity
	'''
	cloud_points = np.zeros((0, 4), dtype = np.float32)
	if not os.path.exists(cloud_path):
		warnings.warn('Cloud path does not exist!'.format(cloud_path))
	if cloud_path.endswith('.pcd'):
		# cloud = pypcd.PointCloud.from_path(cloud_path)
		# cloud_xyz = np.concatenate((cloud.pc_data['x'], cloud.pc_data['y'], cloud.pc_data['z']), axis=0)
		cloud = o3d.t.io.read_point_cloud(cloud_path)
		print(cloud)
		cloud.remove_non_finite_points()
		cloud_xyz = cloud.point['positions'].numpy()
		# cloud_xyz = np.concatenate(np.asarray(cloud.points["x"]),np.asarray(cloud.points["y"]),np.asarray(cloud.points["z"]),axis=0)
		# cloud_xyz = cloud_xyz.reshape(3, int(cloud_xyz.shape[0] / 3)).T
		# cloud_intensity = cloud.pc_data['intensity'] / 255.0
		cloud_intensity = cloud.point["intensity"].numpy()/255.0
		cloud_intensity = cloud_intensity.reshape(-1,1)
		cloud_points = np.hstack((cloud_xyz, cloud_intensity))
		cloud_points = cloud_points.astype(np.float32)
	elif cloud_path.endswith('.bin'):
		cloud_points = np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 4)
	return cloud_points

def trans_cloud(pts_3d, trans_mat):
	'''
		Input: cloud_points: nx3 data
		return: cloud_points: nx3 data
	'''
	# if trans_mat.shape[0] == 3:
	# 	trans_mat = np.vstack((trans_mat, np.array(0., 0., 0., 1.)))
	pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
	pts_3d_proj = np.transpose(np.dot(trans_mat, np.transpose(pts_3d_hom)))
	return pts_3d_proj[:, 0:3]

def save_pcd_cloud(pcd_path, cloud):
	point_num = cloud.shape[0]
	point_type = np.dtype(
		[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
	)
	points = np.zeros(point_num, dtype=point_type)
	points['x'] = cloud[:, 0]
	points['y'] = cloud[:, 1]
	points['z'] = cloud[:, 2]
	points['intensity'] = cloud[:, 3] * 255.0
	# Write the header
	with open(pcd_path, 'w') as fp:
		fp.write(
			'# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1')
		fp.write('\nWIDTH ' + str(point_num))
		fp.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
		fp.write('\nPOINTS ' + str(point_num))
		fp.write('\nDATA binary')
		fp.write('\n')
        # Write the points
	with open(pcd_path, 'ab+') as fp:
		pc_data = np.array(points, dtype=point_type)
		fp.write(pc_data.tobytes('C'))	

csv_path = '/root/data/dataParsing/can/2024-07-13-01-00-22/ins_info.csv'
source_cloud = '/root/data/dataParsing/lidar/innovusion_lidar_front/1698305888.810535904.pcd'
destination_cloud = '/root/data/result.pcd'
# lidar = 'Front_Innuv'

def undisortionfunc(source_cloud, csv_path):
	# Load the pose file
	pose_dict = load_novatel_csv(csv_path)
	pose_time_array = np.array(list(pose_dict.keys()))

	# read the lidar and the time
	lidar = source_cloud.split('/')[-2]
	# print(lidar)
	tmp = source_cloud.split('/')[-1]
	timestmap = tmp.split('.')[0]
	timestmap_n = tmp.split('.')[1]
	timestmap = float(timestmap)
	timestmap_n = float(timestmap_n)
	
	# Load the calib file and get the matrix from GNSS to LiDAR
	gnss2lidar_mat = load_calib(lidar)
	gnss2lidar_mat_inv = np.linalg.inv(gnss2lidar_mat)
	
	# compute the pose matrix for points
	source_timestamp = timestmap + timestmap_n * 1e-9
	target_timestamp = source_timestamp + 0.1
	source_pose_mat, source_index = get_pose_mat(pose_dict, source_timestamp)
	_, target_index = get_pose_mat(pose_dict, target_timestamp)
	
	target_pose_mat_list = []
	time_ratio = []

	if target_index - source_index == 1:
		target_pose_mat_list.append(source_pose_mat)
		target_pose_mat_list.append(pose_dict[pose_time_array[target_index]])
		time_ratio = [1, 1]
	else:
		target_pose_mat_list.append(source_pose_mat)
		target_pose_mat_list.append(pose_dict[pose_time_array[target_index-1]])
		target_pose_mat_list.append(pose_dict[pose_time_array[target_index]])
		time_ratio = [1, 2, 1]

	trans_mat_list = []

	for target_pose_mat in target_pose_mat_list:
		# source_pose_mat_inv = np.linalg.inv(source_pose_mat)
		target_pose_mat_inv = np.linalg.inv(target_pose_mat)
		
		trans_mat =  gnss2lidar_mat @ target_pose_mat_inv @ source_pose_mat @ gnss2lidar_mat_inv
		trans_mat_list.append(trans_mat)
	
	src_cloud = load_cloud(source_cloud)
	src_point = src_cloud[:, 0:3]
	src_intensity = src_cloud[:, 3][:, np.newaxis]

	# split the cloud
	total = sum(time_ratio)
	chunks = [int(src_point.shape[0] * r / total) for r in time_ratio]
	chunks[-1] += src_point.shape[0] - sum(chunks)
	src_point_list = np.split(src_point, np.cumsum(chunks)[:-1], axis=0)

	# undistortion
	des_point_list = []
	for i in range(len(src_point_list)):
		des_point_tmp = trans_cloud(src_point_list[i], trans_mat_list[i])
		des_point_list.append(des_point_tmp)
	
	# save cloud
	des_point = np.vstack(des_point_list)
	des_cloud = np.hstack((des_point, src_intensity))
	save_pcd_cloud(destination_cloud, des_cloud)

undisortionfunc(source_cloud, csv_path)

