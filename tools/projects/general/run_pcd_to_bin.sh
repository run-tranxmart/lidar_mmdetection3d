BASE_DIR=batch_12_aicc_data
PCD_DIR="$BASE_DIR/raw_cloud"
BIN_DIR="$BASE_DIR/cloud_bin"
HEIGHT_OFFSET=-1.3
IS_MULTIPROCESS=0
# Write a code if the IS_MULTIPROCESS is 1 return --is_multiprocess, else return empty string
if [ $IS_MULTIPROCESS -eq 1 ]; then
    IS_MULTIPROCESS_STR="--is_multiprocess"
else
    IS_MULTIPROCESS_STR=""
fi

python pcd_to_bin.py --pcd_dir $PCD_DIR --bin_dir $BIN_DIR --height_offset $HEIGHT_OFFSET $IS_MULTIPROCESS_STR