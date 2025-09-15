# -*- coding: utf-8 -*-
# ===================================================================
# Multi-Domain Style-Injected GAN Training Settings
# ===================================================================

# Data paths
SOURCE_DIR = "./data/src/Tomato_Healthy"  # Single source domain
TARGET_DOMAINS_DIR = "./data/ref"  # Parent directory containing all target domains
GPU = 0
IMAGE_SIZE = 256

# Model architecture
STYLE_DIM = 64  # Dimension of style code
# N_RESIDUAL_BLOCKS = 6  # Number of residual blocks in generator

# Training settings
SAVE_DIR_BASE = './results'
EXPERIMENT_NAME = 'multidomain_exp2'
NUM_EPOCHS = 200
BATCH_SIZE = 4
SAVE_FREQ = 100

# Learning rates
LEARNING_RATE_G = 1e-4  
LEARNING_RATE_D = 1e-4  

# Loss weights
LOSS_WEIGHTS = {
    'gan': 1.0,
    'style_recon': 1.0,  # Style reconstruction loss
    'style_div': 1.0,     # Style diversification loss
    'cycle': 1.0,        # Cycle consistency loss
    'content': 0.0,       # Content preservation (from VGG)
    'style': 0.0          # Style matching (from VGG)
}

# Training options
TRAINING_USE_EMA = True
RESUME_CHECKPOINT = None
NUM_WORKERS = 4

# ===================================================================
# Multi-Domain Inference Settings
# ===================================================================
INFERENCE_INPUT_DIR = './test_images/source'
INFERENCE_REF_DOMAINS_DIR = './data/ref'  # Directory containing all reference domain folders
INFERENCE_CHECKPOINT_DIR = './results/multidomain_experiment/checkpoints/epoch_200'
INFERENCE_OUTPUT_DIR = './results/multidomain_experiment/output'
INFERENCE_TARGET_DOMAIN = 'class_name_2'  # Specify which domain to translate to

INFERENCE_USE_EMA = True

# Style extracting mode for inference
# choices=['average', 'random', 'interpolate', 'noise', 'specific']
INFERENCE_STYLE_MODE = 'interpolate'
INFERENCE_NOISE_LEVEL = 0.1

# VAE settings (optional, for style sampling)
INFERENCE_VAE_CHECKPOINT = None
INFERENCE_VAE_LATENT_DIM = 16
INFERENCE_STYLE_DIM = 64