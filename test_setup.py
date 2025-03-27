import torch
import numpy as np
import cv2
import torchvision
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_setup():
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Torchvision version: {torchvision.__version__}")
    logger.info(f"OpenCV version: {cv2.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        # Test CUDA tensor creation
        x = torch.rand(5, 3).cuda()
        logger.info("Successfully created CUDA tensor")

if __name__ == "__main__":
    test_setup()