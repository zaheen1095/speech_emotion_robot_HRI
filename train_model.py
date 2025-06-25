import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from models.cnn_bilstm import CNNBiLSTM
from config import FEATURES_DIR, CLASSES, FEATURE_SETTINGS, MODEL_DIR, BATCH_SIZE
from sklearn.model_selection import train_test_split

# --- Custom Dataset Loader ---
class FeatureDataset(Dataset):
    def __init__(self, feature_paths, labels):
        self.feature_paths = feature_paths
        self.labels = labels

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        x = np.load(self.feature_paths[idx])
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# --- Load Training Data ---
def load_data():
    X = []
    y = []
    for idx, emotion in enumerate(CLASSES):
        emotion_dir = FEATURES_DIR / 'train' / emotion
        for file in os.listdir(emotion_dir):
            if file.endswith(".npy"):
                X.append(str(emotion_dir / file))
                y.append(idx)
    return train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# --- Training Function ---
def train():
    print("\n🚀 Loading data...")
    X_train, X_val, y_train, y_val = load_data()
    train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FeatureDataset(X_val, y_val), batch_size=BATCH_SIZE)

    print("\n🧠 Initializing model...")
    model = CNNBiLSTM(input_dim=39, num_classes=len(CLASSES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 30
    best_val_acc = 0.0
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n🏋️ Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- Validation ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            print("✅ Best model saved.")

if __name__ == "__main__":
    train()