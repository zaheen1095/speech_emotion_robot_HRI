# train_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import numpy as np

import torch.nn.functional as F
from pathlib import Path
from models.cnn_bilstm import CNNBiLSTM
from config import FEATURES_DIR, CLASSES, FEATURE_SETTINGS, MODEL_DIR, BATCH_SIZE, CLASS_WEIGHTS, MONITOR_METRIC, LABEL_SMOOTHING
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
from augmentations import spec_augment   # + (optionally later) augment_waveform

# --- Custom Dataset Loader ---
class FeatureDataset(Dataset):
    def __init__(self, feature_paths, labels, split: str = "train", augment: bool = False):
        self.feature_paths = feature_paths
        self.labels = labels
         # new flags (default keep old behavior if you don't pass them)
        self.split = split
        self.augment = augment

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        x = np.load(self.feature_paths[idx])
        y = self.labels[idx]

        # train-only SpecAugment (feature-level masks)
        if self.split == "train" and self.augment:
            x = spec_augment(x.T, p=0.5).T  # 50% chance; returns same shape as input
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# --- Load Training Data WITH validation split ---
def load_data():
    X_all, y_all = [], []
    for idx, emotion in enumerate(CLASSES):
        emotion_dir = FEATURES_DIR / 'train' / emotion
        if not emotion_dir.exists():
            continue
        for file in os.listdir(emotion_dir):
            if file.endswith(".npy"):
                X_all.append(str(emotion_dir / file))
                y_all.append(idx)

    if not X_all:
        raise SystemExit(f"No .npy features found under {FEATURES_DIR}/train/<{','.join(CLASSES)}>")

    # 80/20 split inside the training set -> true validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all,
        test_size=0.20,
        stratify=y_all,
        random_state=42
    )
    return X_train, y_train, X_val, y_val

def _plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    import itertools
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
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

def train():
    print("\n🚀 Loading data with proper validation split...")
    X_train, y_train, X_val, y_val = load_data()

    train_loader = DataLoader(FeatureDataset(X_train, y_train, augment = True), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(FeatureDataset(X_val,   y_val, split="val",   augment=False),   batch_size=BATCH_SIZE, shuffle=False)

    print(" Augmentation (SpecAugment) on train:", True)
    print(" Augmentation on val/test:", False)


    print("\n Initializing model...")
    # input_dim = FEATURE_SETTINGS['n_mfcc'] * (
    #     1 + int(FEATURE_SETTINGS.get('use_delta', False)) +
    #     1 + int(FEATURE_SETTINGS.get('use_delta_delta', False)) - 1  # compact way to add deltas if True
    # )
    input_dim = FEATURE_SETTINGS['n_mfcc'] * (
        1 + int(FEATURE_SETTINGS.get('use_delta', False)) +
        int(FEATURE_SETTINGS.get('use_delta_delta', False))
    )
    # (equivalently) input_dim = FEATURE_SETTINGS['n_mfcc'] * (1 + bool(...) + bool(...))

    model = CNNBiLSTM(input_dim=input_dim, num_classes=len(CLASSES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(" Train balance:", np.bincount(np.array(y_train), minlength=len(CLASSES)))
    print(" Val balance:  ", np.bincount(np.array(y_val),   minlength=len(CLASSES)))

    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-6, verbose=True)

    writer = SummaryWriter()
    hist = {'loss_train': [], 'loss_val': [], 'acc_val': [], 'recall_sad': []}

    EPOCHS = 30
    best_monitor_metric = -1.0
    patience = 5
    patience_counter = 0
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n Training...")
    for epoch in range(EPOCHS):
        # ----- Train -----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += float(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / max(1, len(train_loader))
        train_acc = train_correct / max(1, train_total)

        # ----- Validate -----
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        all_logits = []  # NEW

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
                all_logits.append(outputs.cpu())  # NEW

        val_loss /= max(1, len(val_loader))
        np_preds = np.array(all_preds)
        np_labels = np.array(all_labels)
        val_acc = (np_preds == np_labels).mean()

        sad_idx = CLASSES.index('sad')
        tp = int(((np_preds == sad_idx) & (np_labels == sad_idx)).sum())
        fn = int(((np_preds != sad_idx) & (np_labels == sad_idx)).sum())
        recall_sad = tp / (tp + fn) if (tp + fn) else 0.0

        from sklearn.metrics import f1_score, confusion_matrix
        f1_macro = f1_score(np_labels, np_preds, average='macro')
        pred_dist = np.bincount(np_preds, minlength=len(CLASSES))
        cm = confusion_matrix(np_labels, np_preds, labels=list(range(len(CLASSES))))

        

        logits_val = torch.cat(all_logits, dim=0)          # [N, 2]
        probs_val = F.softmax(logits_val, dim=1).numpy()   # [N, 2]
        p_sad = probs_val[:, CLASSES.index('sad')]         # [N]

        # simple grid search; tighten if you like
        grid = np.linspace(0.35, 0.70, 36)  # 0.35..0.70 step 0.01
        best_thr, best_f1 = 0.50, -1.0
        for thr in grid:
            preds_thr = (p_sad >= thr).astype(int)  # 1=sad, 0=happy (assumes CLASSES=['happy','sad'])
            f1_thr = f1_score(np_labels, preds_thr, average='macro')
            if f1_thr > best_f1:
                best_f1 = f1_thr
                best_thr = float(thr)

        # Choose monitor metric
        if MONITOR_METRIC == "f1_score_macro":
            monitor_value = f1_macro
        elif MONITOR_METRIC == "recall_sad":
            monitor_value = recall_sad
        else:
            monitor_value = val_acc

        hist['loss_train'].append(avg_train_loss)
        hist['loss_val'].append(val_loss)
        hist['acc_val'].append(val_acc)
        hist['recall_sad'].append(recall_sad)

        scheduler.step(monitor_value)

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Recall/sad', recall_sad, epoch)
        writer.add_scalar('F1/macro_val', f1_macro, epoch)

        fig_cm = _plot_confusion_matrix(cm, class_names=CLASSES, title=f'Val CM - Epoch {epoch+1}')
        writer.add_figure('ConfusionMatrix/val', fig_cm, epoch)
        plt.close(fig_cm)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | "
            f"Val Recall(sad): {recall_sad:.3f} | "
            f"F1: {f1_macro:.3f} | Dist: {pred_dist.tolist()}"
        )

        # Early stopping + checkpoint
        if monitor_value > best_monitor_metric:
            best_monitor_metric = monitor_value
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_monitor_metric,
                'val_acc': val_acc,
                'recall_sad': recall_sad,
                'sad_threshold': best_thr,          # NEW
            }, MODEL_DIR / "best_model.pt")
            print(f"✓ Best model saved (metric: {best_monitor_metric:.3f}, sad_thr={best_thr:.2f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Optional: warn for overfitting
        if epoch > 10 and train_acc - val_acc > 0.15:
            print(f"Warning: possible overfitting (train {train_acc:.3f} vs val {val_acc:.3f}).")

    _plot_training_curves(hist, MODEL_DIR)
    print(f"Training complete. Best {MONITOR_METRIC}: {best_monitor_metric:.3f}")
    print(f"Saved model and plots to: {MODEL_DIR}")
    writer.close()

if __name__ == "__main__":
    train()
