import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        # CNN architecture for MX330 (2GB VRAM)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 8)  # 8 emotion classes
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class StressDetector:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EmotionCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.emotions = ['fear', 'anger', 'happy', 'sad', 'neutral', 
                        'disgust', 'contempt', 'surprised']
        self.current_stress_level = 0.0
        
    def predict_stress(self, face_img):
        """Predict stress level from face image"""
        image = self.transform(face_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probs = torch.softmax(outputs, dim=1)
            emotion_idx = torch.argmax(probs).item()
            confidence = probs[0][emotion_idx].item()
            
        # Calculate stress level (simplified)
        stress_emotions = {'fear': 0.8, 'anger': 0.7, 'sad': 0.6, 
                         'disgust': 0.5, 'contempt': 0.4}
        emotion = self.emotions[emotion_idx]
        stress_level = stress_emotions.get(emotion, 0.1) * confidence
        
        self.current_stress_level = stress_level
        return stress_level, emotion, confidence