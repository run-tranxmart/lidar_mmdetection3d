import argparse
import pickle
import numpy as np
from mmengine import load

def parse_args():
    parser = argparse.ArgumentParser(description="Process data and update truck labels.")
    parser.add_argument("--input-file", required=True, help="Path to the input .pkl file")
    parser.add_argument("--output-file", required=True, help="Path to save the modified .pkl file")
    parser.add_argument("--range", nargs=3, type=float, default=[6.0, 0, 0],
                        help="A list of [x, y, z] length for the new class")
    parser.add_argument("--name", required=True,
                        help="Name of the new class ('split_truck_from_annos' or 'split_truck_from_vehicle')")
    parser.add_argument("--number", type=int, required=True, help="Label for the new class (e.g., 3)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load data from the input .pkl file
    data_infos = load(args.input_file)
    data_infos_with_new = data_infos.copy()

    _range = np.array(args.range)
    new_count, car_count = 0, 0

    # count_classes = len(data_infos_with_truck['meta_info']['categories'])
    for data in data_infos_with_new['data_list']:
        for ann in data['instances']:
            object_size = np.array(ann['bbox_3d'][3:6])
            if (object_size > _range).all():
                ann['bbox_label_3d'] = args.number
                ann['bbox_label'] = args.number
                new_count += 1
            else:
                car_count += 1

    print(f"new class count = {new_count}, and other count = {car_count}")
    new_class = {args.name: args.number}
    data_infos_with_new['metainfo']['categories'].update(new_class)
    print(data_infos_with_new['metainfo']['categories'])

    # Save the modified data to the output .pkl file
    with open(args.output_file, "wb") as f:
        pickle.dump(data_infos_with_new, f)
        print(f"Modified data saved to {args.output_file}")


if __name__ == "__main__":
    
    main()
    print('labels annotated.')
