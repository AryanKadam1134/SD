# config.py

# Configuration settings for the Stress Detection & Recommendation system

# Path to the AffectNet dataset
DATASET_PATH = 'c:/D/Aryan/Projects/final/stress-detection-app/dataset_root'

# Model parameters
MODEL_PATH = 'models/best_model.pth'
IMAGE_SIZE = (224, 224)  # Size for input images
BATCH_SIZE = 16  # Reduced for 2GB VRAM
NUM_CLASSES = 8  # Number of emotion classes
LEARNING_RATE = 0.001
EPOCHS = 50

# Training settings
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Threshold for stress detection
STRESS_THRESHOLD = 0.5

# Other settings
DEBUG = True

# GPU Settings
USE_GPU = True
CUDA_DEVICE = 0
MIXED_PRECISION = True
NUM_WORKERS = 2  # Optimized for MX330
PIN_MEMORY = True
GRADIENT_ACCUMULATION_STEPS = 4  # To simulate larger batch sizes
