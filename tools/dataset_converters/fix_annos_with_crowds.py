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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.src_pkl_path is None:
        raise FileExistsError("Src pkl path is None!")
    if not osp.exists(args.src_pkl_path):
        raise FileExistsError("Src pkl path does not exist!")
    if args.des_pkl_path is None:
        pkl_name = args.src_pkl_path.rsplit(".", 1)[0]
        args.des_pkl_path = pkl_name + ".fix_crowd.pkl"

    src_labels = mmengine.load(args.src_pkl_path)
    des_labels = {}

    des_labels["metainfo"] = src_labels["metainfo"]
    des_labels["data_list"] = []

    for i, frame_label in enumerate(tqdm(src_labels["data_list"])):
        if len(frame_label["instances"]) == 0:
            print("Frame {} has no labels!".format(frame_label["sample_idx"]))
            continue
        for k, label_obj in enumerate(frame_label["instances"]):
            if label_obj["gt_ori_labels"] == "Non_vehicle/bicycles":
                if label_obj["crowd"] == 0:
                    print(
                        "Frame {} : convert crowd ...".format(frame_label["sample_idx"])
                    )
                    label_obj["crowd"] = 1
                else:
                    print("Frame {} : has crowd obj ...".format(frame_label["sample_idx"]))
            # # if label_obj["gt_ori_labels"] == "Non_vehicle/bicycle":
            # if label_obj["bbox_label"] == 1:
            #     if "1732587484.482383013" in frame_label["sample_idx"]:
            #         obj_ori_label = label_obj["gt_ori_labels"]
            #         obj_length = label_obj["bbox_3d"][3]
            #         obj_width = label_obj["bbox_3d"][4]

            #         print("obj {} , length: {} , width: {}".format(obj_ori_label, obj_length, obj_width))

            #     # if label_obj["crowd"] == 0:
            #     #     print(
            #     #         "Frame {} : convert crowd ...".format(frame_label["sample_idx"])
            #     #     )
            #     #     label_obj["crowd"] = 1
        des_labels["data_list"].append(frame_label)
        frame_instances = frame_label["instances"]

    mmengine.dump(des_labels, args.des_pkl_path, "pkl")