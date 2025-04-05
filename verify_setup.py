import os
from pathlib import Path
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_setup():
    """Verify all required files and directories exist"""
    try:
        # Required paths
        required_paths = {
            'model': Path('models/model_epoch_0_acc_0.1250.pth'),
            'templates': Path('src/webapp/templates'),
            'static': Path('src/webapp/static'),
            'dataset': Path('dataset_root')
        }
        
        # Check each path
        for name, path in required_paths.items():
            if not path.exists():
                logger.error(f"{name.title()} not found at: {path}")
                return False
            logger.info(f"✓ Found {name} at {path}")
            
        # Check specific required files
        required_files = {
            'index template': Path('src/webapp/templates/index.html'),
            'results template': Path('src/webapp/templates/results.html'),
            'model weights': Path('models/model_epoch_0_acc_0.1250.pth'),
            'configuration': Path('config.py')
        }
        
        for name, file_path in required_files.items():
            if not file_path.exists():
                logger.error(f"{name.title()} not found at: {file_path}")
                return False
            logger.info(f"✓ Found {name} at {file_path}")
        
        # Verify model checkpoint
        model_path = Path('models/model_epoch_0_acc_0.1250.pth')
        if not verify_model_checkpoint(model_path):
            logger.error("Model checkpoint verification failed")
            return False
            
        logger.info("✓ All components verified successfully")
        return True
        
    except Exception as e:
        logger.error(f"Setup verification failed: {str(e)}")
        return False

def verify_model_checkpoint(model_path):
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        logger.info("Checking model checkpoint structure...")
        
        if 'model_state_dict' in checkpoint:
            logger.info("✓ Found model_state_dict in checkpoint")
            state_dict = checkpoint['model_state_dict']
        else:
            logger.info("❌ No model_state_dict found in checkpoint")
            return False
            
        # Verify key model layers exist
        required_keys = [
            "features.0.weight", "features.0.bias",
            "features.3.weight", "features.3.bias",
            "features.6.weight", "features.6.bias",
            "features.9.weight", "features.9.bias",
            "classifier.1.weight", "classifier.1.bias",
            "classifier.4.weight", "classifier.4.bias"
        ]
        
        missing_keys = [key for key in required_keys if key not in state_dict]
        if missing_keys:
            logger.error(f"Missing keys in model: {missing_keys}")
            return False
            
        logger.info("✓ Model checkpoint structure verified")
        return True
        
    except Exception as e:
        logger.error(f"Failed to verify model checkpoint: {str(e)}")
        return False

if __name__ == "__main__":
    if verify_setup():
        logger.info("\n✅ System ready to start")
    else:
        logger.error("\n❌ Please fix missing components before starting")