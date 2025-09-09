# Copyright (c) Softmotion. All rights reserved.
import argparse
from os import path as osp
import mmengine
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Update pkl annos with data root")
    parser.add_argument(
        "--src_pkl_path", type=str, default=None, help="Source pickle path"
    )
    parser.add_argument(
        "--des_pkl_path", type=str, default=None, help="Destination pickle path"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        required=True,
        help="Prefix of lidar point path",
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        default="gt_database/gt_database_3cls",
        required=False,
        help="Directory name of gt database",
    )
    parser.add_argument(
        "--cloud_dir",
        type=str,
        default="cloud_bin",
        required=False,
        help="Cloud directory name",
    )

    parser.add_argument(
        "--dbinfos", action="store_true", default=False, help="Update db infos"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.src_pkl_path is not None:
        if osp.exists(args.src_pkl_path):
            src_labels = mmengine.load(args.src_pkl_path)
            des_labels = {}
            if not args.dbinfos:
                des_labels["metainfo"] = src_labels["metainfo"]
                des_labels["data_list"] = []
                for i, frame_label in enumerate(tqdm(src_labels["data_list"])):
                    try:
                        src_path = frame_label["sample_idx"]
                        des_path = osp.join(
                            args.prefix, args.cloud_dir, frame_label["sample_idx"]
                        )
                        frame_label["sample_idx"] = des_path
                        frame_label["lidar_points"]["lidar_path"] = des_path
                        des_labels["data_list"].append(frame_label)
                    except:
                        print(f"Frame {i} has no sample_idx")
            else:
                for cls_name, src_cls_infos in src_labels.items():
                    des_labels[cls_name] = []
                    for cls_db in tqdm(src_cls_infos, desc=cls_name):
                        src_db_pth = cls_db["path"]
                        src_db_name = src_db_pth.rsplit("/")[-1]
                        des_db_pth = osp.join(args.prefix, args.db_dir, src_db_name)
                        cls_db["path"] = des_db_pth
                        des_labels[cls_name].append(cls_db)

        if args.des_pkl_path is None:
            pkl_name = args.src_pkl_path.rsplit(".", 1)[0]
            args.des_pkl_path = pkl_name + ".update_prefix.pkl"
            mmengine.dump(des_labels, args.des_pkl_path, "pkl")
