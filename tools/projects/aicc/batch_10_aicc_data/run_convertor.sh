BASE_PATH=batch_10_aicc_data
OUT_DIR="$BASE_PATH/pkls/5class"
JSON_DIR="$BASE_PATH/raw_label"
SPLIT_SETS="$BASE_PATH/split_set"
PCD_DIR="$BASE_PATH/cloud_bin"

python analytics.py --out_dir $OUT_DIR --json_dir $JSON_DIR --split_sets $SPLIT_SETS --pcd_dir $PCD_DIR