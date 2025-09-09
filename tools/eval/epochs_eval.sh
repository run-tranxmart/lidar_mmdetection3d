#!/bin/bash

# Check the number of input arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <config_file> <epoch_dir> <gpu_num> [<log_dir>]"
    exit 1
fi

config_file="$1"    # Args 1: config file
target_dir="$2"     # Args 2: epoch directories
gpu_num="$3"        # Args 3: gpu number

# Check config file exists or not. If not, exit
if [! -f "$config_file" ]; then
    echo "Config file $config_file does not exist. Please Check"
    exit 1
fi

# Extract the experiment name from config file
exper_name=$(basename "$config_file")
exper_name="${exper_name%.py}"
echo "Exper file name: $exper_name"

# Set the log directory based on the input argument or default value
if [ $# -eq 3 ]; then
    log_dir="work_dirs/$exper_name/eval_epochs"
    echo "Log dir: $log_dir"
    # Create the directory of logs (if does not exists)
    mkdir -p "$log_dir"
else
    log_dir="$4"
    echo "Log dir: $log_dir"
    # Create the directory of logs (if does not exists)
    mkdir -p "$log_dir"
fi

# Initialize an empty list to save the checkpoints
file_list=()

# Find the checkpoints which start with "epoch_" and end with ".pth"
while IFS= read -r file_path; do
    file_list+=("$file_path")
done < <(find "$target_dir" -type f -name "epoch_*.pth")

sorted_list=($(
    printf "%s\n" "${file_list[@]}" | sort -rV
))

# Enumerate the checkpoints
log_list=()
for file in "${sorted_list[@]}"; do
    model_name=$(basename "$file")
    log_path="${log_dir}/${model_name}.eval.log"

    # Run the evaluation command and get the PID
    dist_test_command="./tools/dist_test.sh $config_file $file $gpu_num"
    echo "Evaluation Command: $dist_test_command"
    echo "Log path: ${log_path}"
    log_list+=("$log_path")
    $dist_test_command | tee -a "${log_path}"
    pid=$!
    # Wait the process finish
    wait $pid
    echo "Evaluation $file finish! Next..."
done


declare -A epoch_ap_bev
declare -A epoch_ap_3d
# Enumerate the log file and extract the AP values
for log_file in "${log_list[@]}"; do
    echo "Parse Log file: $log_file"
    ap_bev_list=()
    ap_3d_list=()
    while IFS= read -r line; do
        # use grep to select bev AP values
        if echo "$line" | grep -q 'bev  AP: [0-9.]*'; then
            # echo "line: $line"
            value=$(echo "$line" | awk '{print $3}' | tr -d '[:space:]')
            # echo "BEV mAP: $value"
            ap_bev_list+=("$value")
        fi
        # use grep to select bev AP values
        if echo "$line" | grep -q '3d   AP: [0-9.]*'; then
            # echo "line: $line"
            value=$(echo "$line" | awk '{print $3}' | tr -d '[:space:]')
            ap_3d_list+=("$value")
        fi
    done < <(cat "$log_file")
    epoch_ap_bev["$log_file"]=${ap_bev_list[-1]}
    epoch_ap_3d["$log_file"]=${ap_3d_list[-1]}
    echo "BEV mAP: ${ap_bev_list[-1]}"
    echo "3D  mAP: ${ap_3d_list[-1]}"
done

# Get maximum bev mAP
max_bev_ap=0
max_bev_ap_file=""
for file in "${!epoch_ap_bev[@]}"; do
    current_ap=${epoch_ap_bev[$file]}
    if (( $(echo "$current_ap > $max_bev_ap" | bc -l) )); then
        max_bev_ap=$current_ap
        max_bev_ap_file=$file
    fi
done
echo "Max BEV mAP $max_bev_ap - File $max_bev_ap_file"

# Get maximum 3d mAP
max_3d_ap=0
max_3d_ap_file=""
for file in "${!epoch_ap_3d[@]}"; do
    current_ap=${epoch_ap_3d[$file]}
    if (( $(echo "$current_ap > $max_3d_ap" | bc -l) )); then
        max_3d_ap=$current_ap
        max_3d_ap_file=$file
    fi
done
echo "Max 3D mAP $max_3d_ap - File $max_3d_ap_file"