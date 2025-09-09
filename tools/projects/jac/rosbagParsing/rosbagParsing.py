import argparse
import sys
import os
import src


def parse_args():

    parser = argparse.ArgumentParser(description="a tool for rosbag parsing.")

    # position parameter
    parser.add_argument("input_file", type=str, help="the route of input file")
    parser.add_argument("output_file", type=str, help="the route for stroe the result")

    parser.add_argument("-l", "--lidar", action="store_true", help="parsing lidar data")

    parser.add_argument("-m", "--mp4", action="store_true", help="parsing mp4 data.")
    parser.add_argument(
        "-p",
        "--pic",
        type=str,
        choices=["png", "jpg", "yuv"],
        nargs="?",
        const="png",
        help="parsing pic data. needing input a mp4 file.",
    )
    parser.add_argument(
        "-d",
        "--distortion",
        help="do distortion need input the camera to make sure internal",
        nargs="?",
        const="True",
        choices=[
            "True",
            "Camera_4",
            "Camera_5",
            "Camera_6",
            "Camera_7",
            "Camera_8",
            "Camera_9",
            "Camera_10",
            "Camera_11",
        ],
    )
    parser.add_argument(
        "-c",
        "--can",
        action="store_true",
        help="parsing can data for point cloud undisortion.",
    )
    parser.add_argument("--camera", nargs="+", type=str, help="List of camera names")
    parser.add_argument("--lidarList", nargs="+", type=str, help="List of lidar names")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.input_file):
        print(f"wrong:intput file {args.input_file} not exist。")
        sys.exit(1)

    # output_dir = os.path.dirname(args.output_file)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    camera_list = args.camera
    lidar_list = args.lidarList

    # if do lidar
    if args.lidar:
        print("lidar parsing")
        if lidar_list:
            src.LidarParsing(args.input_file, args.output_file, lidar_topics=lidar_list)
        else:
            src.LidarParsing(args.input_file, args.output_file)

    if args.can:
        print("can parsing")
        src.canParsing(args.input_file, args.output_file)

    # select function with different check
    if args.mp4 and args.pic and args.distortion:
        # parsing mp4, use mp4 to parse pic with distortion
        # input file must be a bag file
        print(args.mp4, args.pic, args.distortion, 1)
        if camera_list:
            src.Mp4Parsing(
                args.input_file,
                args.output_file,
                save_type=args.pic,
                distortion=args.distortion,
                camera_topic=camera_list,
            )
        else:
            src.Mp4Parsing(
                args.input_file,
                args.output_file,
                save_type=args.pic,
                distortion=args.distortion,
            )
    elif args.mp4 and args.pic and not args.distortion:
        # parsing mp4, use mp4 to parse pic without distortion
        # input file must be a bag file
        print(args.mp4, args.pic, args.distortion, 2)
        if camera_list:
            src.Mp4Parsing(
                args.input_file,
                args.output_file,
                save_type=args.pic,
                camera_topic=camera_list,
            )
        else:
            src.Mp4Parsing(args.input_file, args.output_file, save_type=args.pic)
    elif args.mp4 and not args.pic and not args.distortion:
        # parsing mp4 only
        # input file must be a bag file
        print(args.mp4, args.pic, args.distortion, 3)
        if camera_list:
            src.Mp4Parsing(args.input_file, args.output_file, camera_topic=camera_list)
        else:
            src.Mp4Parsing(args.input_file, args.output_file)
    elif args.mp4 and not args.pic and args.distortion:
        # parsing mp4 only and report a warning of dont de distortion
        # input file must be a bag file
        print(args.mp4, args.pic, args.distortion, 4)
        print("wraning: mp4 file don't need to distortion")
        if camera_list:
            src.Mp4Parsing(args.input_file, args.output_file, camera_topic=camera_list)
        else:
            src.Mp4Parsing(args.input_file, args.output_file)
    elif not args.mp4 and args.pic and args.distortion:
        # parsing pic from mp4 file with distortion
        # input file must be a mp4 file
        print(args.mp4, args.pic, args.distortion, 5)
        if args.distortion == "True":
            print("warning: please assign the internal of these pictures")
        else:
            src.PicParsing(
                args.input_file, args.output_file, args.pic, disortion=args.distortion
            )
    elif not args.mp4 and args.pic and not args.distortion:
        # parsing pic from mp4 file without distortion
        # input file must be a mp4 file
        print(args.mp4, args.pic, args.distortion, 6)
        src.PicParsing(args.input_file, args.output_file, args.pic)
    elif not args.mp4 and not args.pic and args.distortion:
        # do distortion for pic
        # input file must be a pic file
        print(args.mp4, args.pic, args.distortion, 7)
        if args.distortion == "True":
            print("warning: please assign the internal of these pictures")
        else:
            src.distortion(args.input_file, args.output_file, args.distortion)
    else:
        print("please input the mode")
    # print(args.mp4, args.pic, args.distortion)
