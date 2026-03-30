#!/bin/bash

# Check if correct number of arguments provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 <MODEL_DIR> <INPUT_DIR> <OUTPUT_DIR> <DEVICE>"
    echo "Example: $0 /path/to/model /path/to/fasta /path/to/output cuda:0"
    exit 1
fi


MODEL_DIR="$1"
INPUT_DIR="$2" 
OUTPUT_DIR="$3"
DEVICE="$4"


if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory '$MODEL_DIR' does not exist"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create combined results file with header
echo "Protein Name	DBP prediction probability	DBP prediction result	TF prediction probability	TF prediction result" > "${OUTPUT_DIR}/all_predictions.res"

# Process each FASTA file
for fasta_file in "${INPUT_DIR}"/*.fasta; do
    # Check if any .fasta files exist
    if [ ! -e "$fasta_file" ]; then
        echo "No .fasta files found in $INPUT_DIR"
        exit 1
    fi
    
    echo "Processing: $(basename "$fasta_file")"
    
    # Run prediction
    CUDA_VISIBLE_DEVICES=1 python prediction.py "$MODEL_DIR" "$fasta_file" "$OUTPUT_DIR" "$DEVICE"
    
    # Check if prediction was successful
    if [ $? -ne 0 ]; then
        echo "Error: Prediction failed for $(basename "$fasta_file")"
        continue
    fi
    
    # Append results (skip header) to combined file
    if [ -f "${OUTPUT_DIR}/DBP_TF_prediction.res" ]; then
        tail -n +2 "${OUTPUT_DIR}/DBP_TF_prediction.res" >> "${OUTPUT_DIR}/all_predictions.res"
    else
        echo "Warning: Output file ${OUTPUT_DIR}/DBP_TF_prediction.res not found"
    fi
done

echo "All predictions completed! Results in: ${OUTPUT_DIR}/all_predictions.res"