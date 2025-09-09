BASE_PATH=batch_1_robosense_3M1_data
OUT_DIR="$BASE_PATH/pkls/5class"
RAW_PICKLE_PATH="$BASE_PATH/raw_label/3M1-lidar-PVB.pkl"
SPLIT_SETS="$BASE_PATH/split_set"
PCD_DIR="$BASE_PATH/cloud_bin"

python analytics_batch1.py --out_dir $OUT_DIR --raw_pickle_path $RAW_PICKLE_PATH --split_sets $SPLIT_SETS --pcd_dir $PCD_DIR