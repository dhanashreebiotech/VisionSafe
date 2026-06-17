"""
Train Activity Model — Enhanced for higher accuracy
Trains the Conv1D-BiLSTM Activity Classifier with:
- More epochs (100)
- Early stopping
- Data augmentation (noise injection)
- Learning rate scheduling
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import ActivityClassifier
import json

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "dataset_artifacts")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "..", "activity_model.pth")
EPOCHS = 100
BATCH_SIZE = 16  # Smaller batch for better generalization
LR = 0.0005
PATIENCE = 15  # Early stopping patience

class PoseDataset(Dataset):
    def __init__(self, npz_path, augment=False):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data['X']).float()
        self.y = torch.from_numpy(data['y']).long()
        self.augment = augment
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            # Add small noise for regularization
            x += torch.randn_like(x) * 0.01
            # Random temporal shift (shift sequence by 1-2 frames)
            if torch.rand(1).item() > 0.5:
                shift = torch.randint(1, 3, (1,)).item()
                x = torch.roll(x, shift, dims=0)
        return x, self.y[idx]

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    train_path = os.path.join(DATA_DIR, "train.npz")
    val_path = os.path.join(DATA_DIR, "val.npz")
    test_path = os.path.join(DATA_DIR, "test.npz")
    
    if not os.path.exists(train_path):
        print(f"Data not found at {train_path}!")
        return

    train_ds = PoseDataset(train_path, augment=True)
    val_ds = PoseDataset(val_path, augment=False)
    test_ds = PoseDataset(test_path, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Calculate Class Weights for Imbalance
    all_labels = train_ds.y.numpy()
    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    weights = total_samples / (len(class_counts) * class_counts.astype(float))
    class_weights = torch.FloatTensor(weights).to(device)
    print(f"Class Counts: {class_counts}")
    print(f"Class Weights: {weights}")
    
    model = ActivityClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
        train_acc = correct / total
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
                
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={total_loss:.4f} Acc={train_acc:.4f} | Val Loss={avg_val_loss:.4f} Acc={val_acc:.4f} | LR={scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
            print(f"  ✓ New best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break
            
    print(f"\n{'='*50}")
    print(f"Training Complete. Best Val Acc: {best_val_acc:.4f}")
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Test Evaluation
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    test_correct = 0
    test_total = 0
    
    all_preds = []
    all_labels_test = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            test_total += y.size(0)
            test_correct += (predicted == y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels_test.extend(y.cpu().numpy())
            
    print(f"\nTest Accuracy: {test_correct/test_total:.4f}")
    
    from sklearn.metrics import classification_report, confusion_matrix
    classes = ["Sitting", "Standing", "Walking", "Yoga"]
    print("\n" + classification_report(all_labels_test, all_preds, target_names=classes))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels_test, all_preds))

if __name__ == "__main__":
    train()
