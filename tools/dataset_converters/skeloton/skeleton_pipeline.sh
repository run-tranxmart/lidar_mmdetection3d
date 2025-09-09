#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Output path provided as the first argument when running the script
OUTPUT_PATH=${1:-"default_output_path"}

# Ensure the output directory exists
mkdir -p "${OUTPUT_PATH}"

# Array of parameters
NUM_BATCHES_LIST=(5 10 20 30 50  100)
LIMITS=(500 1000 2000)
RULES=("categories" "specific_labels" "instances_and_labels" "categories_and_labels")
TARGET_LABELS="[0]"

# Create/Check logs.csv header if it doesn't exist
LOG_FILE="${OUTPUT_PATH}/logs.csv"
if [ ! -f "$LOG_FILE" ]; then
    echo "timestamp,rule,num_batches,limit,truck,cyclist,car,misc,pedestrian" > "$LOG_FILE"
fi

# Function to process one combination
process_combination() {
    local RULE=$1
    local NUM_BATCHES=$2
    local LIMIT=$3
    local TARGET_LABELS=$4
    
    # Create save directory
    local SAVE_DIR="${OUTPUT_PATH}/${NUM_BATCHES}_splits/${RULE}"
    mkdir -p "${SAVE_DIR}"
    
    # Construct save path
    local SAVE_PATH="${SAVE_DIR}/${NUM_BATCHES}_splits_skeloton_${LIMIT}.pkl"
    
    echo "Processing: Rule=${RULE}, Batches=${NUM_BATCHES}, Limit=${LIMIT}, Labels=${TARGET_LABELS}"
    
    # Execute the Python script and capture its output
    OUTPUT=$(python "${SCRIPT_DIR}/skeloton_dataset.py" \
        --save_path "${SAVE_PATH}" \
        --limit "${LIMIT}" \
        --num_batches "${NUM_BATCHES}" \
        --rule "${RULE}" \
        --target-labels "${TARGET_LABELS}")
    
    # Extract numbers using grep and sed
    TRUCK=$(echo "$OUTPUT" | grep "Truck:" | sed 's/Truck: //')
    CYCLIST=$(echo "$OUTPUT" | grep "Cyclist:" | sed 's/Cyclist: //')
    CAR=$(echo "$OUTPUT" | grep "Car:" | sed 's/Car: //')
    MISC=$(echo "$OUTPUT" | grep -E "(Static|Misc):" | sed -E 's/(Static|Misc): //')
    PEDESTRIAN=$(echo "$OUTPUT" | grep "Pedestrian:" | sed 's/Pedestrian: //')
    
    # Get current timestamp
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Append to logs.csv (using flock for thread safety)
    flock "$LOG_FILE" bash -c "echo \"$TIMESTAMP,$RULE,$NUM_BATCHES,$LIMIT,$TRUCK,$CYCLIST,$CAR,$MISC,$PEDESTRIAN\" >> \"$LOG_FILE\""
    
    # Print to console for monitoring
    echo "----------------------------------------"
    echo "Timestamp: $TIMESTAMP"
    echo "Rule: $RULE"
    echo "Num Batches: $NUM_BATCHES"
    echo "Limit: $LIMIT"
    echo "Results:"
    echo "$OUTPUT"
    echo "========================================"
}
export -f process_combination
export OUTPUT_PATH SCRIPT_DIR LOG_FILE

# Maximum number of parallel processes
MAX_JOBS=8  # Adjust this number based on your system's capabilities

# Counter for managing parallel processes
current_jobs=0

# Process all combinations
for RULE in "${RULES[@]}"; do
    for NUM_BATCHES in "${NUM_BATCHES_LIST[@]}"; do
        for LIMIT in "${LIMITS[@]}"; do
            # Wait if we've reached max parallel jobs
            if [[ $current_jobs -ge $MAX_JOBS ]]; then
                wait -n  # Wait for any child process to finish
                ((current_jobs--))
            fi
            
            # Run the process in background
            process_combination "$RULE" "$NUM_BATCHES" "$LIMIT" "$TARGET_LABELS" &
            
            ((current_jobs++))
            echo "Started job: $current_jobs"
        done
    done
done

# Wait for all remaining processes to complete
wait
echo "All jobs completed!"