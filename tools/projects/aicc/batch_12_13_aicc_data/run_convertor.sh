BASE_PATH=$1
OUT_DIR="$BASE_PATH/pkls/test"
JSON_DIR="$BASE_PATH/raw_label"
SPLIT_SETS="$BASE_PATH/split_set"
PCD_DIR="$BASE_PATH/cloud_bin"
Z_OFFSET=0.0

python convertor.py --out_dir $OUT_DIR --json_dir $JSON_DIR --split_sets $SPLIT_SETS --pcd_dir $PCD_DIR --z_offset $Z_OFFSET
