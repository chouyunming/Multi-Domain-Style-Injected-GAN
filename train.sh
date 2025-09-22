#!/bin/bash

SOURCE_DIR="./data/src/Tomato_Healthy"
TARGET_DIR="./data/ref"
EPOCHS=200
BATCH_SIZE=4

# # ========================= EXPERIMENT 1 ==========================
echo "--- Preparing Experiment 1: ---"

EXP1_NAME='multidomain_exp1'
EXP1_LR_G=2e-4
EXP1_LR_D=1e-4
EXP1_WEIGHTS='{"gan": 1.0, "cycle": 10.0, "identity": 5.0, "style": 1.0, "content": 1.0}'

python main.py \
  --exp_name "$EXP1_NAME" \
  --source_dir "$SOURCE_DIR" \
  --target_dir "$TARGET_DIR" \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr_g $EXP1_LR_G \
  --lr_d $EXP1_LR_D \
  --loss_weights "$EXP1_WEIGHTS" \
  --wandb

echo "--- Experiment 1 Finished ---"
echo ""

# # ========================= EXPERIMENT 2 ==========================
echo "--- Preparing Experiment 2: ---"

EXP2_NAME='multidomain_exp1_all'
EXP2_LR_G=2e-4
EXP2_LR_D=1e-4
EXP2_WEIGHTS='{"gan": 1.0, "cycle": 10.0, "identity": 5.0, "style": 1.0, "content": 1.0}'

python main.py \
  --exp_name "$EXP2_NAME" \
  --source_dir "$SOURCE_DIR" \
  --target_dir "./data/ref_all" \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr_g $EXP2_LR_G \
  --lr_d $EXP2_LR_D \
  --loss_weights "$EXP2_WEIGHTS" \
  --wandb

echo "--- Experiment 2 Finished ---"
echo ""

echo "All experiments completed."