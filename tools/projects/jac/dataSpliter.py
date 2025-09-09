import os
import argparse
from pathlib import Path,PurePath,PosixPath
from tqdm import tqdm
import shutil

camera_list = ['Camera_8','Camera_9','Camera_10','Camera_11']
lidar_list = ['hesai_left_lidar', 'hesai_right_lidar', 
                'innovusion_lidar_front', 'innovusion_lidar_front2', 'innovusion_lidar_front3',
                'innovusion_lidar_front4']

class file_time:
    # class for store the timestmap
    def __init__(self, seconds:int, nanoseconds:int) -> None:
        self.seconds = seconds
        self.nanoseconds = nanoseconds

    def __time_compute1(self):
        return self.seconds * 1e9 + self.nanoseconds

    def __time_compute2(self):
        return self.seconds + self.nanoseconds * 1e-9

    def __sub__(self, other):
        if isinstance(other, file_time):
            return self.__time_compute2() - other.__time_compute2()
        return NotImplemented

def check_batch(batch, postfix:str):

    if postfix == 'pcd':
        if len(batch) != 6:
            return False
    elif postfix == 'png':
        if len(batch) != 4:
            return False

    first_second = batch[0]['secs']
    first_nanosecond = str(batch[0]['nanosecs'])

    for b in batch:
        b_nano = str(b['nanosecs'])

        if b['secs'] != first_second:
            return False
        if len(b_nano) != len(first_nanosecond):
            return False
        else:
            if b_nano[0] != first_nanosecond[0]:
                return False
    return True

def extract_batch(sorted_files:list, postfix:str):
    # collect the all lidar datas as a batch
    Batches = []
    tmp_batch = []
    tmp_batch_items = []
    item = ''
    recorded_index = None

    if postfix == 'pcd':
        item = 'lidar'
    elif postfix == 'png':
        item = 'camera'
    
    i = 0
    while i < len(sorted_files):
        f = sorted_files[i]
        
        if f[f'{item}'] not in tmp_batch_items:
            tmp_batch_items.append(f[f'{item}'])
            tmp_batch.append(f)
            if len(tmp_batch) == 1:  # 记录tmp为空后的第一个索引
                recorded_index = i
        else:
            if check_batch(tmp_batch, postfix):
                Batches.append(tmp_batch)
                tmp_batch = []
                tmp_batch_items = []
                tmp_batch_items.append(f[f'{item}'])
                tmp_batch.append(f)
            else:
                i = recorded_index + 1
                tmp_batch = []
                tmp_batch_items = []
                recorded_index = None  # 清空记录的索引
                continue  # 继续处理从recorded_index开始的数据
        i += 1

    return Batches


def sort_files(base_path:str, itemList:list, postfix:str):
    # get all folders
    folders = [Path(base_path+'/'+f) for f in itemList]
    # print(folders)

    all_files = []
    for f in folders:
        pcds= f.glob(f'*.{postfix}')
        # add pcd filenames to pcds_all list
        all_files += pcds
    # print(all_files)

    files_dict=[]
    for f in all_files:
        # get timestamps and file
        if postfix == 'pcd':
            timestamp = f.name
            lidar = f.parent.name
            secs,nanosecs,_ = timestamp.split('.')
            files_dict.append({'pcd':f,'nanosecs':int(nanosecs),'secs':int(secs),'lidar':lidar})
        elif postfix == 'png':
            timestamp = f.name.replace('image', '')
            image = f.parent.name
            secs,nanosecs,_ = timestamp.split('.')
            files_dict.append({'img':f,'nanosecs':int(nanosecs),'secs':int(secs),'camera':image})
        else:
            print('unsupport type')
            return
    # print(files_dict)

    # sorted by time
    # need to add condition
    files_dict  = sorted(files_dict,key=lambda x:x['secs']*1e9+x['nanosecs'])
    with open(Path.joinpath(Path(base_path),'sorted.txt'),'w') as f:
        for p in files_dict:
            if postfix == 'pcd':
                f.write(str(p['pcd'])+'\n')
            elif postfix == 'png':
                f.write(str(p['img'])+'\n')
    files_batches = extract_batch(files_dict, postfix)
    # print(files_batches)
    return files_batches

def find_nearest(pb_time, img_batchs, idx):
    target_batch = None
    diff = 10000
    i = idx

    for ib in img_batchs[idx:]:
        ib_time = file_time(ib[0]['secs'], ib[0]['nanosecs'])
        diff_tmp = pb_time - ib_time
        if diff_tmp <= 0.0:
            diff = diff_tmp
            target_batch = ib
            break
        i += 1

    return target_batch, i, diff

def matching(pcd_batchs:list, img_batchs:list, time_delay:float=2.0):
    # do matching from pcd to img
    # pcd_times = [item[0] for item in self.pcd_time_path]
    # image_times = [item[0] for item in self.image_time_path]

    chosen = []
    start_time = file_time(-1,0)
    max_diff = 0.15
    idx = 0

    for i, pb in tqdm(enumerate(pcd_batchs)):
        # print(i,pb)
        pb_time = file_time(pb[0]['secs'], pb[0]['nanosecs'])
        # print(pb_time - start_time)

        if i == 0 or pb_time - start_time > time_delay:
            start_time = pb_time
            if i != 0:
                ib, idx_r, diff = find_nearest(pb_time, img_batchs, idx) # get nearest img_batch
                idx = idx_r
                if diff < max_diff:
                    chosen.append([pb,ib])
            else:
                ib, idx_r, diff = find_nearest(pb_time, img_batchs, idx) # get nearest img_batch
                idx = idx_r
                if diff < max_diff:
                    chosen.append([pb,ib])
    # print(chosen)
    return chosen

def parse_args():

    parser = argparse.ArgumentParser(description='a tool for rosbag parsing.')

    # position parameter
    parser.add_argument('--lidar_dir', type=str, help='the route of lidar file', default='/root/data/annotation_data/data_parsing_0805_maxiangguang/Test-087-20240729-0950/2024-07-29-11-04-34/lidar')
    parser.add_argument('--camera_dir', type=str, help='the route of camera file', default='/root/data/annotation_data/data_parsing_0805_maxiangguang/Test-087-20240729-0950/2024-07-29-11-04-34/camera')
    parser.add_argument('--output_dir', type=str, help='the route for stroe the result', default='/root/data/annotation_data/data_parsing_0805_maxiangguang/Test-087-20240729-0950/2024-07-29-11-04-34/result')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    lidar_dir = args.lidar_dir
    camera_dir = args.camera_dir
    save_path_base = args.output_dir

    # get sorted batchs
    pcd_batchs = sort_files(lidar_dir, lidar_list, 'pcd')
    img_batchs = sort_files(camera_dir, camera_list, 'png')

    print('-------------start matching--------------')
    # bind the lidar to img
    chosen = matching(pcd_batchs, img_batchs)

    print('-------------start moving--------------')
    for c in tqdm(chosen):
        innovusion_lidar_front_pcd = None
        for pcd in c[0]:
            if pcd['lidar'] == 'innovusion_lidar_front':
                innovusion_lidar_front_pcd = pcd['pcd']
                break
        if innovusion_lidar_front_pcd:
            folder_name = innovusion_lidar_front_pcd.stem
            # print(folder_name)
            destination_folder = Path(save_path_base) / folder_name
            os.makedirs(destination_folder, exist_ok=True)

            for lidar_item in c[0]:
                save_path = str(destination_folder) + '/' + lidar_item['lidar'] + lidar_item['pcd'].name
                shutil.copy(lidar_item['pcd'], save_path)

            for img_item in c[1]:
                save_path = str(destination_folder) + '/' + img_item['camera'] + img_item['img'].name
                shutil.copy(img_item['img'], save_path)
