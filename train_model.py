# train_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
from models.cnn_bilstm import CNNBiLSTM
from config import FEATURES_DIR, CLASSES, FEATURE_SETTINGS, MODEL_DIR, BATCH_SIZE, CLASS_WEIGHTS, MONITOR_METRIC, LABEL_SMOOTHING
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

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
    X, y, groups = [], [], []
    for idx, emotion in enumerate(CLASSES):
        emotion_dir = FEATURES_DIR / 'train' / emotion
        for file in os.listdir(emotion_dir):
            if file.endswith(".npy"):
                X.append(str(emotion_dir / file))
                y.append(idx)
                # group by speaker or prefix (prevents speaker leakage)
                groups.append(file.split('_', 1)[0].lower())
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))
    return [X[i] for i in train_idx], [X[i] for i in val_idx], [y[i] for i in train_idx], [y[i] for i in val_idx]

def _plot_confusion_matrix(cm, class_names):
    import itertools
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Val Confusion Matrix')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    return fig

def _plot_training_curves(hist, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss
    fig1, ax1 = plt.subplots(figsize=(5,3), dpi=120)
    ax1.plot(hist['loss_train'], label='train')
    ax1.plot(hist['loss_val'], label='val')
    ax1.set_title('Loss vs Epoch')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / 'curves_loss.png'); plt.close(fig1)

    # Metrics
    fig2, ax2 = plt.subplots(figsize=(5,3), dpi=120)
    ax2.plot(hist['acc_val'], label='val_acc')
    ax2.plot(hist['recall_sad'], label='val_recall_sad')
    ax2.set_title('Validation Metrics vs Epoch')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Score'); ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / 'curves_metrics.png'); plt.close(fig2)

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

    # class weights on same device
    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    writer = SummaryWriter()
    hist = {'loss_train': [], 'loss_val': [], 'acc_val': [], 'recall_sad': []}

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

        avg_train_loss = total_loss / max(1, len(train_loader))

        # --- Validation ---
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss_val_batch = criterion(outputs, labels)
                val_loss += float(loss_val_batch.item())
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_loss /= max(1, len(val_loader))

        np_preds = np.array(all_preds)
        np_labels = np.array(all_labels)
        val_acc = (np_preds == np_labels).mean()

        # compute always for logs/plots
        sad = 1
        tp = int(((np_preds == sad) & (np_labels == sad)).sum())
        fn = int(((np_preds != sad) & (np_labels == sad)).sum())
        recall_sad = tp / (tp + fn) if (tp + fn) else 0.0
        f1_macro = f1_score(np_labels, np_preds, average='macro')

        # --- Choose monitor metric from config ---
        if MONITOR_METRIC == "f1_score_macro":
            monitor_value = f1_macro
        elif MONITOR_METRIC == "recall_sad":
            monitor_value = recall_sad
        else:
            monitor_value = val_acc

        scheduler.step(monitor_value)

        # --- TensorBoard scalars ---
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)                      # NEW
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/macro_val', f1_macro, epoch)                  # NEW
        writer.add_scalar(f'Monitor Metric/{MONITOR_METRIC}', monitor_value, epoch)

        # --- TensorBoard val confusion matrix image (optional) ---
        cm = confusion_matrix(np_labels, np_preds, labels=list(range(len(CLASSES))))
        fig_cm = _plot_confusion_matrix(cm, class_names=CLASSES)
        writer.add_figure('ConfusionMatrix/val', fig_cm, epoch)             # NEW
        plt.close(fig_cm)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f} | Val Recall(sad): {recall_sad:.2f} | "
              f"F1(macro): {f1_macro:.2f}")

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

    # save static PNG curves like in papers
    _plot_training_curves(hist, MODEL_DIR)                                  # NEW
    print(f"Saved plots to: {MODEL_DIR}")

    writer.close()

if __name__ == "__main__":
    train()
