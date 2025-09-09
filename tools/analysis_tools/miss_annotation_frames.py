
import os
import json
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(
        description="Filter JSON files by score threshold and save their names in a text file."
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        required=True,
        help='Path to the folder containing result JSON files.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to the output folder where the filtered file names will be saved.'
    )
    parser.add_argument(
        '--score_threshold',
        type=float,
        required=True,
        help='Score threshold for filtering; JSON files with any score > threshold will be recorded.'
    )
    args = parser.parse_args()

  
    os.makedirs(args.output_dir, exist_ok=True)
    

    output_txt_path = os.path.join(args.output_dir, "filtered_files.txt")
    
   
    json_files = glob.glob(os.path.join(args.result_dir, '*.json'))
    if not json_files:
        print("No JSON files found in the specified result directory.")
        return
    
    filtered_files = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
        
        # Check if the file contains the key "scores_3d"
        if "scores_3d" not in data:
            print(f"File {json_file} does not contain key 'scores_3d'. Skipping.")
            continue
        
        scores = data["scores_3d"]
        
        # If any score exceeds the threshold, store the filename
        if any(score > args.score_threshold for score in scores):
            filtered_files.append(os.path.basename(json_file))
            print(f"Selected {json_file}")
    
    # Save the filtered filenames to the text file
    if filtered_files:
        with open(output_txt_path, 'w') as f:
            for file_name in filtered_files:
                f.write(file_name + '\n')
        print(f"Filtered file names saved to {output_txt_path}")
    else:
        print("No files met the filtering criteria.")

if __name__ == "__main__":
    main()

