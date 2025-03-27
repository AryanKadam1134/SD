import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_dataset_structure(dataset_path: str):
    """Verify AffectNet dataset structure and count samples"""
    
    splits = ['train', 'test', 'val']
    emotions = ['fear', 'anger', 'happy', 'sad', 'neutral', 'disgust', 'contempt', 'surprised']
    dataset_stats = {}

    try:
        for split in splits:
            split_path = Path(dataset_path) / split
            if not split_path.exists():
                logger.error(f"Split directory missing: {split}")
                continue
                
            split_stats = {}
            total_images = 0
            
            for emotion in emotions:
                emotion_path = split_path / emotion
                if not emotion_path.exists():
                    logger.error(f"Emotion directory missing: {split}/{emotion}")
                    continue
                    
                # Count images in emotion directory
                images = list(emotion_path.glob('*.jpg')) + list(emotion_path.glob('*.png'))
                num_images = len(images)
                split_stats[emotion] = num_images
                total_images += num_images
                
                logger.info(f"{split}/{emotion}: {num_images} images")
            
            dataset_stats[split] = {
                'per_emotion': split_stats,
                'total': total_images
            }
            logger.info(f"\nTotal {split} images: {total_images}\n")

        return dataset_stats

    except Exception as e:
        logger.error(f"Error verifying dataset: {str(e)}")
        return None

if __name__ == "__main__":
    DATASET_PATH = 'c:/D/Aryan/Projects/final/stress-detection-app/dataset_root'
    logger.info(f"Verifying dataset at: {DATASET_PATH}")
    stats = verify_dataset_structure(DATASET_PATH)
    
    if stats:
        logger.info("\nDataset Summary:")
        for split, data in stats.items():
            logger.info(f"\n{split.upper()} SET:")
            for emotion, count in data['per_emotion'].items():
                logger.info(f"{emotion}: {count} images")
            logger.info(f"Total: {data['total']} images")