import argparse
import json
import os
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser(description='Split the data of AICC')
    parser.add_argument(
        '--base_dir',
        type=str,
        default='./data/batch_2_aicc_data',
        help='Path of cloud directory')
    parser.add_argument(
        '--cloud_dir', type=str, default=None, help='Path of cloud directory')
    parser.add_argument(
        '--cloud_list',
        type=str,
        default='list/raw_cloud.txt',
        help='Path of cloud list')
    parser.add_argument(
        '--label_dir', type=str, default='raw_labels', help='Path of label')
    parser.add_argument(
        '--split_dir',
        type=str,
        default='split_set',
        help='Path of cloud directory')
    parser.add_argument(
        '--split_ratio',
        type=int,
        default=8,
        help='1/split_ratio as validation set')
    args = parser.parse_args()
    return args


def get_label_num(json_pth):
    label_num = 0
    with open(json_pth, 'r') as f:
        json_label = json.load(f)
        label_num = len(json_label['items'])
    return label_num


if __name__ == "__main__":
    args = parse_args()

    # Get the cloud frames from list or dir
    cloud_frames = []

    cloud_list_pth = osp.join(args.base_dir, args.cloud_list)
    if osp.exists(cloud_list_pth):
        with open(cloud_list_pth, 'r') as f:
            list_lines = f.readlines()
            for ln in sorted(list_lines):
                if '_' in ln:
                    ln = ln.strip()
                    cloud_name = ln.split('/')[-1]
                    cloud_frames.append(cloud_name)
    elif args.cloud_dir is not None:
        cloud_dir = osp.join(args.base_dir, args.cloud_dir)
        if osp.isdir(cloud_dir):
            cloud_list = os.listdir(cloud_dir)
    else:
        raise FileExistsError(
            f'Both {cloud_list_pth} and {args.cloud_dir} do not exist!')

    # Divide the cloud frames to sequence dict
    seq_dict = {}
    for cloud_name in cloud_frames:
        if '.pcd' in cloud_name:
            cloud_name = cloud_name.replace('.pcd', '.bin')
        # get the label path and check it
        label_name = cloud_name.rsplit('.', 1)[0] + '.json'
        label_pth = osp.join(args.base_dir, args.label_dir, label_name)
        if osp.exists(label_pth):
            if get_label_num(label_pth) == 0:
                print(f'Cloud {cloud_name} has no labels')
                continue
        else:
            print(f'Label pth {label_pth} does not exist!')
            continue

        seq_name = cloud_name.split('_')[0]
        if seq_name in seq_dict:
            seq_dict[seq_name].append(cloud_name)
        else:
            seq_dict[seq_name] = [cloud_name]

    # Split the frame: 80% used for training, 20% used for evaluation
    seq_id = 0
    train_frames = []
    val_frames = []
    for seq_name, seq_frames in seq_dict.items():
        if seq_id % 8 == 0:
            val_frames.extend(seq_frames)
            # print(f"Seq: {seq_name} , Count : {len(seq_frames)} -> Val set ")
        else:
            train_frames.extend(seq_frames)
            # print(f"Seq: {seq_name} , Count : {len(seq_frames)} -> Train set ")
        seq_id += 1
    trainval_frames = train_frames + val_frames

    # write the frames to split set
    split_dir = osp.join(args.base_dir, args.split_dir)
    os.makedirs(split_dir, exist_ok=True)
    train_list_pth = osp.join(split_dir, "train.txt")
    val_list_pth = osp.join(split_dir, "val.txt")
    trainval_list_pth = osp.join(split_dir, "trainval.txt")
    with open(train_list_pth, 'w') as f:
        for frame_name in train_frames:
            f.write(frame_name + '\n')
    with open(val_list_pth, 'w') as f:
        for frame_name in val_frames:
            f.write(frame_name + '\n')
    with open(trainval_list_pth, 'w') as f:
        for frame_name in trainval_frames:
            f.write(frame_name + '\n')
