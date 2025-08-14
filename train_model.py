import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from models.cnn_bilstm import CNNBiLSTM
from config import FEATURES_DIR, CLASSES, FEATURE_SETTINGS, MODEL_DIR, BATCH_SIZE, CLASS_WEIGHTS, MONITOR_METRIC, LABEL_SMOOTHING
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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
    X, y = [], []
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

    print("\n Initializing model...")
    model = CNNBiLSTM(input_dim=39, num_classes=len(CLASSES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # **class weights on the same device as the model/inputs**
    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    writer = SummaryWriter()

    EPOCHS = 30
    best_monitor_metric = -1.0
    patience = 5
    patience_counter = 0
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n Training...")
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        # --- Validation ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        np_preds = np.array(all_preds)
        np_labels = np.array(all_labels)
        val_acc = (np_preds == np_labels).mean()

        # --- Choose monitor metric from config ---
        if MONITOR_METRIC == "f1_score_macro":
            monitor_value = f1_score(np_labels, np_preds, average='macro')
        elif MONITOR_METRIC == "recall_sad":
            sad = 1
            tp = int(((np_preds == sad) & (np_labels == sad)).sum())
            fn = int(((np_preds != sad) & (np_labels == sad)).sum())
            monitor_value = tp / (tp + fn) if (tp + fn) else 0.0
        else:  # default to validation accuracy
            monitor_value = val_acc

        scheduler.step(monitor_value)

        # for logging: always compute recall_sad for the print
        sad = 1
        tp = int(((np_preds == sad) & (np_labels == sad)).sum())
        fn = int(((np_preds != sad) & (np_labels == sad)).sum())
        recall_sad = tp / (tp + fn) if (tp + fn) else 0.0

        writer.add_scalar('Loss/train', total_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar(f'Monitor Metric/{MONITOR_METRIC}', monitor_value, epoch)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.2f} | Val Recall(sad): {recall_sad:.2f}")

        # --- Early stopping / checkpoint on monitor metric ---
        if monitor_value > best_monitor_metric:
            best_monitor_metric = monitor_value
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            print("Best model saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(" Early stopping!")
                break

    writer.close()

if __name__ == "__main__":
    train()
