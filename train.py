import torch
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from src.models.stress_detector import EmotionCNN
from src.data.data_loader import create_data_loaders
from src.training.trainer import ModelTrainer
from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for model checkpoints and visualizations"""
    Path('models').mkdir(exist_ok=True)
    Path('visualizations').mkdir(exist_ok=True)

def main():
    try:
        setup_directories()
        
        # Set device and memory settings for GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        logger.info(f"Using device: {device}")

        # Create model and move to device
        model = EmotionCNN().to(device)
        logger.info(f"Model created on {device}")

        # Create data loaders with error handling
        try:
            train_loader, val_loader, test_loader = create_data_loaders(
                DATASET_PATH, 
                batch_size=BATCH_SIZE
            )
            logger.info(f"Data loaders created with batch size: {BATCH_SIZE}")
        except Exception as e:
            logger.error(f"Failed to create data loaders: {str(e)}")
            raise

        # Initialize trainer with progress tracking
        trainer = ModelTrainer(
            model=model,
            train_dataset=train_loader.dataset,
            val_dataset=val_loader.dataset,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE
        )
        logger.info("Trainer initialized successfully")

        # Train model with progress visualization
        history = trainer.train(epochs=EPOCHS, save_path='models')
        
        # Plot training results
        if history:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Loss History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='Train Accuracy')
            plt.plot(history['val_acc'], label='Validation Accuracy')
            plt.title('Accuracy History')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('visualizations/training_history.png')
            plt.close()
            
            logger.info("Training visualization saved")
        
        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()