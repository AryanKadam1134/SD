import sys
import torch
import matplotlib
import pandas
import seaborn
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_installations():
    """Verify all required packages are installed correctly"""
    try:
        logger.info("=== Package Versions ===")
        logger.info(f"Python: {sys.version.split()[0]}")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Matplotlib: {matplotlib.__version__}")
        logger.info(f"Pandas: {pandas.__version__}")
        logger.info(f"Seaborn: {seaborn.__version__}")
        logger.info(f"OpenCV: {cv2.__version__}")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            logger.info(f"\nGPU: {torch.cuda.get_device_name(0)}")
            x = torch.rand(3, 3).cuda()
            logger.info("CUDA tensor test: Success")
            
        # Test matplotlib
        import matplotlib.pyplot as plt
        plt.figure()
        plt.close()
        logger.info("Matplotlib test: Success")
        
        logger.info("\n✓ All packages installed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Setup check failed: {str(e)}")
        return False

if __name__ == "__main__":
    check_installations()