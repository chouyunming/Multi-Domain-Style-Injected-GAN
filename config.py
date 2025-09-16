# -*- coding: utf-8 -*-
# ===================================================================
# Multi-Domain Style-Injected GAN Training Settings
# ===================================================================

# Data paths
SOURCE_DIR = "./data/src/Tomato_Healthy"  # Single source domain
TARGET_DIR = "./data/ref2"  # Parent directory containing all target domains
GPU = 0
IMAGE_SIZE = 256

# Model architecture
STYLE_DIM = 64  # Dimension of style code
# N_RESIDUAL_BLOCKS = 6  # Number of residual blocks in generator

# Training settings
SAVE_DIR_BASE = './results'
EXPERIMENT_NAME = 'multidomain_exp'
NUM_EPOCHS = 200
BATCH_SIZE = 4
SAVE_FREQ = 100

N_RESIDUAL_BLOCKS = 8
STYLE_DIM = 256  # Style dimension

# Learning rates
LEARNING_RATE_G = 2e-4  
LEARNING_RATE_D = 1e-4  

# Loss weights
LOSS_WEIGHTS = {
    'gan': 1.0,
    'cycle': 10.0,        # Cycle consistency loss
    'identity': 5.0,     # Identity loss'
    'content': 1.0,       # Content preservation (from VGG)
    'style': 1.0          # Style matching (from VGG)
}

# Training options
TRAINING_USE_EMA = True
RESUME_CHECKPOINT = None

# ===================================================================
# Multi-Domain StyleCycleGAN Inference Settings
# ===================================================================
INFERENCE_INPUT_DIR = './experiments/plant_village_raw/synthetic_target/Tomato_healthy'
INFERENCE_TARGET_DOMAINS_DIR = './stylecyclegan/dataset/target_domains'
INFERENCE_CHECKPOINT_DIR = './stylecyclegan/results/multi_domain_experiment/checkpoints/epoch_180'
INFERENCE_OUTPUT_DIR = './stylecyclegan/output/multi_domain/interpolate'
INFERENCE_TARGET_DOMAIN = 1  # Which target domain to translate to (1, 2, 3, ...)

INFERENCE_USE_EMA = True

# ===================================================================
# Style Extracting Mode for Multi-Domain
# choices=['average', 'random', 'interpolate', 'specific_domain']
# ===================================================================
INFERENCE_STYLE_MODE = 'specific_domain'
INFERENCE_DOMAIN_ID = 1  # Specific domain to use for style extraction
INFERENCE_NOISE_LEVEL = 0.1

# ===================================================================
# Metrics Per Epoch Experiment Settings
# ===================================================================
METRICS_INPUT_DIR = './stylecyclegan/output/multi_domain_exp/interpolate'
METRICS_TARGET_DIR = './experiments/plant_village_raw/train/Tomato_Bacterial_spot'

# VAE settings (optional, for style sampling)
INFERENCE_VAE_CHECKPOINT = None
INFERENCE_VAE_LATENT_DIM = 16
INFERENCE_STYLE_DIM = 64