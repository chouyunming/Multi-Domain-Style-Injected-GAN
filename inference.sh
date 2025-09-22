#!/bin/bash

# Multi-Domain Style-Injected GAN Inference Script
# Processes multiple checkpoints across all target domains with specified style modes

# --- Configuration ---
SAVE_DIR_BASE="./results"
EXPERIMENT_NAME="multidomain_exp1"

BASE_CHECKPOINT_DIR="${SAVE_DIR_BASE}/${EXPERIMENT_NAME}/checkpoints"
BASE_OUTPUT_DIR="${SAVE_DIR_BASE}/${EXPERIMENT_NAME}/output"
SOURCE_INPUT_DIR="./data/target/Tomato_healthy"
REF_DOMAINS_DIR="./data/ref"

GPU_ID=0
IMAGE_SIZE=256

# Style modes to run - Available: average, random, interpolate, noise, specific
STYLE_MODES=("interpolate")

# Optional: Specify checkpoint epochs to process (leave empty for all)
SPECIFIC_EPOCHS=()

echo "======================================================================"
echo "Starting Multi-Domain Inference: $EXPERIMENT_NAME"
echo "Style modes: ${STYLE_MODES[*]}"
echo "======================================================================"

# --- Validation ---
if [ ! -d "$BASE_CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $BASE_CHECKPOINT_DIR"
    exit 1
fi

if [ ! -d "$SOURCE_INPUT_DIR" ]; then
    echo "ERROR: Source directory not found: $SOURCE_INPUT_DIR"
    exit 1
fi

if [ ! -d "$REF_DOMAINS_DIR" ]; then
    echo "ERROR: Reference directory not found: $REF_DOMAINS_DIR"
    exit 1
fi

# --- Auto-discover target domains ---
TARGET_DOMAINS=()
for domain_path in "$REF_DOMAINS_DIR"/*; do
    if [ -d "$domain_path" ]; then
        domain_name=$(basename "$domain_path")
        if [[ ! "$domain_name" =~ ^\. ]] && [[ "$domain_name" != "__pycache__" ]]; then
            TARGET_DOMAINS+=("$domain_name")
        fi
    fi
done

if [ ${#TARGET_DOMAINS[@]} -eq 0 ]; then
    echo "ERROR: No target domains found in $REF_DOMAINS_DIR"
    exit 1
fi

echo "Found ${#TARGET_DOMAINS[@]} domains: ${TARGET_DOMAINS[*]}"

# --- Find checkpoint directories ---
CHECKPOINT_DIRS=()
for checkpoint_path in "$BASE_CHECKPOINT_DIR"/*; do
    if [ -d "$checkpoint_path" ]; then
        epoch_dir_name=$(basename "$checkpoint_path")
        if [ ${#SPECIFIC_EPOCHS[@]} -eq 0 ] || [[ " ${SPECIFIC_EPOCHS[*]} " =~ " $epoch_dir_name " ]]; then
            CHECKPOINT_DIRS+=("$epoch_dir_name")
        fi
    fi
done

if [ ${#CHECKPOINT_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoint directories found"
    exit 1
fi

echo "Processing ${#CHECKPOINT_DIRS[@]} checkpoints: ${CHECKPOINT_DIRS[*]}"

total_combinations=$((${#CHECKPOINT_DIRS[@]} * ${#TARGET_DOMAINS[@]} * ${#STYLE_MODES[@]}))
echo "Total combinations: $total_combinations"
echo ""

# --- Track progress ---
total_success=0
total_failed=0
current_combination=0

# --- Main processing loop ---
for epoch_dir in "${CHECKPOINT_DIRS[@]}"; do
    checkpoint_path="$BASE_CHECKPOINT_DIR/$epoch_dir"
    
    # Verify checkpoint file exists
    checkpoint_file="${checkpoint_path}/checkpoint.pth"
    if [ ! -f "$checkpoint_file" ]; then
        echo "WARNING: Checkpoint file not found: $checkpoint_file"
        continue
    fi
    
    echo "----------------------------------------------------------------------"
    echo "Processing checkpoint: $epoch_dir"
    echo "----------------------------------------------------------------------"
    
    for domain in "${TARGET_DOMAINS[@]}"; do
        domain_ref_path="${REF_DOMAINS_DIR}/${domain}"
        
        # Check domain directory
        if [ ! -d "$domain_ref_path" ]; then
            echo "WARNING: Domain directory not found: $domain_ref_path"
            continue
        fi
        
        for style_mode in "${STYLE_MODES[@]}"; do
            ((current_combination++))
            
            # Create output directory
            output_dir="${BASE_OUTPUT_DIR}/${epoch_dir}/${domain}/${style_mode}"
            mkdir -p "$output_dir"
            
            # Show domain header
            printf "[%3d/%3d] %-30s (%s)\n" $current_combination $total_combinations "$domain" "$style_mode"
            
            # Create log file for verbose output
            log_file="${output_dir}/inference_log.txt"
            
            # Run inference - redirect verbose output to log, but let tqdm show in terminal
            # stdout goes to log file, stderr (where tqdm displays) goes to terminal
            python3 inference.py \
                --input_dir "$SOURCE_INPUT_DIR" \
                --ref_domains_dir "$REF_DOMAINS_DIR" \
                --checkpoint_dir "$checkpoint_path" \
                --output_dir "$output_dir" \
                --target_domain "$domain" \
                --gpu $GPU_ID \
                --image_size $IMAGE_SIZE \
                --style_mode "$style_mode" \
                --noise_level 0.1 > "$log_file"
            
            # Get exit code and wait a moment for files to be fully written
            exit_code=$?
            sleep 0.5  # Brief pause to ensure files are written to disk
            
            # Check results
            if [ $exit_code -eq 0 ]; then
                # Count generated images - include both uppercase and lowercase extensions
                img_count=$(find "$output_dir" -maxdepth 1 \( -name "*.png" -o -name "*.PNG" -o -name "*.jpg" -o -name "*.JPG" -o -name "*.jpeg" -o -name "*.JPEG" \) 2>/dev/null | wc -l)
                
                if [ $img_count -gt 0 ]; then
                    printf "✓ Success (%d images)\n\n" $img_count
                    ((total_success++))
                else
                    printf "✗ No images generated (check log: %s)\n\n" "$log_file"
                    ((total_failed++))
                fi
            else
                printf "✗ Failed (exit code: %d, check log: %s)\n\n" $exit_code "$log_file"
                ((total_failed++))
            fi
        done
    done
    echo ""
done

echo "======================================================================"
echo "Inference completed!"
echo "Successful runs: $total_success"
echo "Failed runs: $total_failed"
if [ $((total_success + total_failed)) -gt 0 ]; then
    success_rate=$(echo "scale=1; $total_success * 100 / ($total_success + $total_failed)" | bc -l 2>/dev/null || echo "N/A")
    echo "Success rate: $success_rate%"
fi
echo "Results saved to: $BASE_OUTPUT_DIR"
echo "======================================================================"

# --- Generate summary report ---
SUMMARY_FILE="${BASE_OUTPUT_DIR}/inference_summary.txt"
echo "INFERENCE SUMMARY" > "$SUMMARY_FILE"
echo "=================================" >> "$SUMMARY_FILE"
echo "Generated on: $(date)" >> "$SUMMARY_FILE"
echo "Experiment: $EXPERIMENT_NAME" >> "$SUMMARY_FILE"
echo "Target domains: ${TARGET_DOMAINS[*]}" >> "$SUMMARY_FILE"
echo "Style modes: ${STYLE_MODES[*]}" >> "$SUMMARY_FILE"
echo "Successful runs: $total_success" >> "$SUMMARY_FILE"
echo "Failed runs: $total_failed" >> "$SUMMARY_FILE"
echo "Total combinations: $total_combinations" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "Images generated per configuration:" >> "$SUMMARY_FILE"
for epoch_dir in "${CHECKPOINT_DIRS[@]}"; do
    echo "" >> "$SUMMARY_FILE"
    echo "$epoch_dir:" >> "$SUMMARY_FILE"
    
    for domain in "${TARGET_DOMAINS[@]}"; do
        for style_mode in "${STYLE_MODES[@]}"; do
            output_dir="${BASE_OUTPUT_DIR}/${epoch_dir}/${domain}/${style_mode}"
            if [ -d "$output_dir" ]; then
                img_count=$(find "$output_dir" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
                echo "  $domain/$style_mode: $img_count images" >> "$SUMMARY_FILE"
            fi
        done
    done
done

echo "Summary saved to: $SUMMARY_FILE"

if [ $total_failed -gt 0 ]; then
    echo ""
    echo "⚠️  Some runs failed. Check individual log files for details:"
    echo "   find $BASE_OUTPUT_DIR -name 'inference_log.txt' -exec grep -l 'Error\|Failed' {} \;"
fi