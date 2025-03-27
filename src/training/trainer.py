import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from torch.cuda.amp import autocast, GradScaler

class ModelTrainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size=16, learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            # Memory optimization for MX330
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
        
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', 
                                           factor=0.1, patience=5, verbose=True)
        self.scaler = GradScaler()  # For mixed precision training
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # Initialize other attributes
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
        self.best_acc = 0.0
        self.patience = 10
        self.patience_counter = 0

    def train(self, epochs=50, save_path='models'):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Mixed precision training
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Log progress
                if batch_idx % 50 == 0:
                    mem_used = torch.cuda.memory_allocated() / 1024**2
                    self.logger.info(
                        f'Epoch: {epoch+1} | Batch: {batch_idx}/{len(self.train_loader)} | '
                        f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | '
                        f'GPU Mem: {mem_used:.1f}MB'
                    )
            
            # Validation and early stopping logic
            val_acc = self.validate()
            self.scheduler.step(val_acc)
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(save_path, epoch, val_acc)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.info(f'Early stopping triggered after {epoch+1} epochs')
                    break

    def validate(self):
        """Validate the model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return correct / total
    
    def validate_metrics(self):
        self.model.eval()
        val_loss = 0
        confusion_matrix = torch.zeros(8, 8)
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                    
        return val_loss / len(self.val_loader), confusion_matrix
    
    def plot_training_history(self):
        """Plot training loss and validation accuracy history"""
        plt.figure(figsize=(12, 4))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.title('Training Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        
    def load_best_model(self, model_path):
        """Load the best model checkpoint"""
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['val_acc']

    def save_checkpoint(self, save_path, epoch, val_acc):
        """Save model checkpoint"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
        }
        
        path = os.path.join(save_path, f'model_epoch_{epoch}_acc_{val_acc:.4f}.pth')
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")