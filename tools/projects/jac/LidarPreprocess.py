import argparse
import os,glob
from pathlib import Path,PurePath
import numpy as np
# from pypcd import pypcd
import warnings
# from copy import deepcopy
import open3d as o3d
import time

mat = {'innovusion_lidar_front': 
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

def load_cloud(cloud_path):
		'''
			Load the cloud to the nx4 array
		'''
		cloud_points = np.zeros((0, 4), dtype = np.float32)
		if not os.path.exists(cloud_path):
			warnings.warn('Cloud path does not exist!'.format(cloud_path))
		if cloud_path.endswith('.pcd'):
			# cloud = pypcd.PointCloud.from_path(cloud_path)
			# cloud_xyz = np.concatenate((cloud.pc_data['x'], cloud.pc_data['y'], cloud.pc_data['z']), axis=0)
			cloud = o3d.t.io.read_point_cloud(cloud_path)
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
		'''
		# if trans_mat.shape[0] == 3:
		# 	trans_mat = np.vstack((trans_mat, np.array(0., 0., 0., 1.)))
		pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
		pts_3d_proj = np.transpose(np.dot(trans_mat, np.transpose(pts_3d_hom)))
		return pts_3d_proj[:, 0:3]

def extract_batch(sorted_pcds: list, merge_seconds: float, base_lidar: str):
    Batches = []
    tmp_batch = []
    tmp_batch_lidars = []
    tmp_batch_times = []
    base_time = None  # Initialize base time for reference

    for pcd in sorted_pcds:
        current_time = pcd['secs'] + pcd['nanosecs'] * 1e-9

        # Add the LiDAR to the current batch if it's not already included
        if pcd['lidar'] not in tmp_batch_lidars:
            tmp_batch_lidars.append(pcd['lidar'])
            tmp_batch.append(pcd)
            tmp_batch_times.append(current_time)

            # Set base_time when encountering base_lidar in the batch
            if pcd['lidar'] == base_lidar:
                base_time = current_time
        else:
            # Check time differences against the base_time if it is set
            if base_time is not None:
                time_diffs = [t - base_time for t in tmp_batch_times]

                # Check if all time differences are within merge_seconds
                if all(abs(diff) <= merge_seconds for diff in time_diffs):
                    Batches.append(tmp_batch)
                else:
                    print("Base lidar time:", base_time)
                    print("Timestamps:", tmp_batch_times)
                    print("Time differences:", time_diffs)
                    warnings.warn(f"Batch with base lidar at {base_time} exceeds {merge_seconds} seconds; batch skipped.")
            
            # Reset batch information when starting a new batch
            tmp_batch = [pcd]
            tmp_batch_lidars = [pcd['lidar']]
            tmp_batch_times = [current_time]
            
            # Reset base_time if the new batch has base_lidar, otherwise reset it to None
            if pcd['lidar'] == base_lidar:
                base_time = current_time
            else:
                base_time = None  # Reset base_time if base_lidar is not in the new batch

    return Batches




def sort_pcd(base_path:str, merge_seconds: float, base_lidar: str):
	base_path = Path(base_path)
	folders = [f for f in base_path.iterdir() if f.is_dir()]
	pcds_all = []
	for f in folders:
		pcds= f.glob('*.pcd')
		pcds_all += pcds
	pcds_dict = []
	for p in pcds_all:
		timestamp = p.name
		lidar = p.parent.name
		secs,nanosecs,_ = timestamp.split('.')
		pcds_dict.append({'pcd':p,'nanosecs':int(nanosecs),'secs':int(secs),'lidar':lidar})
	pcds_dict  = sorted(pcds_dict,key=lambda x:x['secs']*1e9+x['nanosecs'])
	with open(Path.joinpath(base_path,'sort_pcd.txt'),'w') as f:
		for p in pcds_dict:
			f.write(str(p['pcd'])+'\n')
	pcds_batches = extract_batch(pcds_dict, merge_seconds, base_lidar)
	return pcds_batches

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
	


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Baidu Parser')
	parser.add_argument('--dir_path', type = str, default='/home/linali/Downloads/pcd_fromShahab/transformed',
					  help='path to lidars')	
	parser.add_argument('--save_path', type=str, default='/home/linali/Downloads/pcd_fromShahab/transformed_merge', 
					 help='path to save')	
	parser.add_argument('--merge_seconds', type=float, default=0.05, help='Merge seconds') # 50 milliseconds = 0.05 seconds
	parser.add_argument('--warp_frame', type=int, default=6, help='Warping frames')
	parser.add_argument('--base_lidar', type=str, default='innovusion_lidar_front',
                        help='Name of the base lidar topic')
	args = parser.parse_args()
	dirpath = args.dir_path
	savepath = args.save_path
	base_lidar = args.base_lidar 
	merge_seconds = args.merge_seconds

	if not Path.exists(Path(savepath)):
		Path.mkdir(Path(savepath)) 
		
	pcd_batches = sort_pcd(dirpath, merge_seconds, base_lidar)

	for count,batch in enumerate(pcd_batches):
		if count < 0:
			continue
		# 1 Stitch the pcd
		pcd_stitch = np.zeros((0,4),dtype=np.float32)
		base_lidar_name = None

		for pcd in batch:
			 # Check if the lidar topic is 'innovusion_lidar_front'
			if pcd['lidar'] == base_lidar:
				# Extract the name without extension for the base lidar
				base_lidar_name = os.path.basename(str(pcd['pcd'])).rsplit('.', 1)[0]

			# print(str(pcd['pcd']))
			data_pcd = load_cloud(str(pcd['pcd']))
			time.sleep(2)
			tran_mat = mat[pcd['lidar']]
			data_point = trans_cloud(data_pcd[:,0:3],tran_mat)
			data_new = np.hstack((data_point,data_pcd[:,3][:, np.newaxis]))
			pcd_stitch = np.concatenate((pcd_stitch,data_new),axis=0)
			
		# Ensure we found the 'innovusion_lidar_front' topic
		if base_lidar_name is not None:
			save_name = f"{savepath}/{base_lidar_name}_stitch.pcd"
		else:
			warnings.warn(f"Warning: The specified base lidar topic '{base_lidar}' was not found in the current batch.")
			continue  # Skip the rest of the code for this iteration and move to the next batch

		# save_pcd_cloud(save_name, pcd_stitch)


		# 2 Denoiseing
		# 2.1 remove 0 intensity points
		index_zero = np.where(pcd_stitch[:,3]==0)
		pcd_stitch = np.delete(pcd_stitch,index_zero,axis=0)
		# save_name = savepath + '/' + str(count) +'_rm0.pcd'
		# save_pcd_cloud(save_name, pcd_stitch)


		# 2.2 remove duplicate points
		# uniq_xyz,uniq_index,dup_counts = np.unique(pcd_stitch[:,:-1],axis=0,return_index=True,return_counts=True)
		# uniq_intensity = pcd_stitch[uniq_index,-1]
		# dup_pcd_toprocess = np.where(dup_counts>1)[0]
		# for dup_ind in dup_pcd_toprocess:
		# 	ind_in_stitch = uniq_index[dup_ind]
		# 	dup_xyz= np.unique(np.where(pcd_stitch[:,:-1] == pcd_stitch[ind_in_stitch,:-1])[0])
		# 	dup_intensity = np.max(pcd_stitch[dup_xyz,-1])
		# 	uniq_intensity[dup_ind] = dup_intensity
		# uniq_pcd = np.concatenate((uniq_xyz,uniq_intensity.reshape(len(uniq_xyz),1)),axis=1)
		# save_name = savepath + '/' + str(count) +'_uniq.pcd'
		# save_pcd_cloud(save_name, uniq_pcd)
		uniq_pcd = np.unique(pcd_stitch,axis=0)

		# # 3 height normalization
		uniq_pcd[:,2] -= 1.35
		# save_name = savepath + '/' + str(count) +'_H_norm.pcd'
		# save_pcd_cloud(save_name, uniq_pcd)

		# save files
		pcd_final = uniq_pcd.astype(np.float32)
		# save_name = savepath + '/' + str(count) +'.pcd'
		save_pcd_cloud(save_name, pcd_final)
		save_name_bin = save_name +'.bin'
		with open(save_name_bin,'wb') as f:
			f.write(pcd_final)

		# print(count)
        
			
