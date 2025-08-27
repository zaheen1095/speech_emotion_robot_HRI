# train_model.py
import os
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from models.cnn_bilstm import CNNBiLSTM
from config import (
    FEATURES_DIR, CLASSES, FEATURE_SETTINGS, MODEL_DIR, USE_ATTENTION, USE_EXTRA_FEATURES, EXTRA_FEATURES,
    BATCH_SIZE, CLASS_WEIGHTS, MONITOR_METRIC, LABEL_SMOOTHING
)
from augmentations import spec_augment

# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# -------------------------
# Dataset
# -------------------------
class FeatureDataset(Dataset):
    def __init__(self, feature_paths, labels, split: str = "train", augment: bool = False):
        self.feature_paths = feature_paths
        self.labels = labels
        self.split = split
        self.augment = augment

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        x = np.load(self.feature_paths[idx])  # (T, D)
        y = self.labels[idx]
        # train-only SpecAugment (feature-level)
        if self.split == "train" and self.augment:
            x = spec_augment(x.T, p=0.5).T  # keep (T, D)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# ---- Optional Focal Loss (B3) ----
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # cross-entropy with optional label smoothing
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none',
                             label_smoothing=self.label_smoothing)
        # convert CE to pt = exp(-CE)
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal

# -------------------------
# Utilities
# -------------------------
def _plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    import itertools
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True'); ax.set_xlabel('Predicted')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    return fig

def _plot_training_curves(hist, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss
    fig1, ax1 = plt.subplots(figsize=(5, 3), dpi=120)
    ax1.plot(hist['loss_train'], label='train')
    ax1.plot(hist['loss_val'], label='val')
    ax1.set_title('Loss vs Epoch')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / 'curves_loss.png'); plt.close(fig1)

    # Metrics
    fig2, ax2 = plt.subplots(figsize=(5, 3), dpi=120)
    ax2.plot(hist['acc_val'], label='val_acc')
    ax2.plot(hist['recall_sad'], label='val_recall_sad')
    ax2.set_title('Validation Metrics vs Epoch')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Score'); ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / 'curves_metrics.png'); plt.close(fig2)

# -------------------------
# Data loading
# -------------------------
# Markers used by your offline_augmentation—adjust if your filenames differ.
# Any file whose *filename* contains one of these tokens will be treated as augmented.
AUG_TOKENS = (
    ".aug", "_aug", "-aug",  # generic
    "noise", "reverb", "rir", "pitch", "tempo", "speed", "stretch"  # common ops
)

def _is_augmented_file(path_str: str) -> bool:
    name = os.path.basename(path_str).lower()
    return any(tok in name for tok in AUG_TOKENS)

def load_data():
    X_all, y_all, groups = [], [], []
    for idx, emotion in enumerate(CLASSES):
        emotion_dir = FEATURES_DIR / 'train' / emotion
        if not emotion_dir.exists():
            continue
        for file in os.listdir(emotion_dir):
            if file.endswith(".npy"):
                p = str(emotion_dir / file)
                X_all.append(p)
                y_all.append(idx)
                # group key = basename without any "__aug-..." suffix
                base = Path(file).stem
                group_key = base.split("__aug-")[0]
                groups.append(f"{emotion}/{group_key}")  # include class to be safe

    if not X_all:
        raise SystemExit(f"No .npy features found under {FEATURES_DIR}/train/<{','.join(CLASSES)}>")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(X_all, y_all, groups))

    X_train = [X_all[i] for i in train_idx]
    y_train = [y_all[i] for i in train_idx]
      # val: keep originals only (drop augs)
    X_val = [X_all[i] for i in val_idx if not _is_augmented_file(X_all[i])]
    y_val = [y_all[i] for i in val_idx if not _is_augmented_file(X_all[i])]


    # (optional) sanity print:
    # print("Unique groups in TRAIN:", len(set([groups[i] for i in train_idx])))
    # print("Unique groups in VAL  :", len(set([groups[i] for i in val_idx])))

    return X_train, y_train, X_val, y_val

# -------------------------
# Training
# -------------------------
def train():
    print("\n🚀 Loading data with clean validation (no augmented items in val)...")
    X_train, y_train, X_val, y_val = load_data()

    train_loader = DataLoader(
        FeatureDataset(X_train, y_train, split="train", augment=True),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        FeatureDataset(X_val, y_val, split="val", augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(" SpecAugment on train:", True)
    print(" Augmentation on val/test:", False)

    print("\n Initializing model...")

    with torch.no_grad():
        sample_x, _ = next(iter(val_loader)) if len(val_loader) > 0 else next(iter(train_loader))
    input_dim = sample_x.shape[-1]

    model = CNNBiLSTM(input_dim=input_dim, num_classes=len(CLASSES), use_attention=USE_ATTENTION)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(" Train balance:", np.bincount(np.array(y_train), minlength=len(CLASSES)))
    print(" Val balance:  ", np.bincount(np.array(y_val),   minlength=len(CLASSES)))

    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-6, verbose=True)

    USE_FOCAL = os.environ.get("USE_FOCAL_LOSS", "0") == "1"   # opt-in via env var
    FOCAL_GAMMA = float(os.environ.get("FOCAL_GAMMA", "2.0"))

    if USE_FOCAL:
        criterion = FocalLoss(weight=weights, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    writer = SummaryWriter()
    hist = {'loss_train': [], 'loss_val': [], 'acc_val': [], 'recall_sad': []}

    EPOCHS = 50
    best_monitor_metric = -1.0
    patience = 5
    patience_counter = 0
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    happy_idx = CLASSES.index('happy')
    sad_idx   = CLASSES.index('sad')

    print("\n Training...")
    for epoch in range(EPOCHS):
        # ----- Train -----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += float(loss.item())
            preds = outputs.argmax(dim=1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        avg_train_loss = train_loss / max(1, len(train_loader))
        train_acc = train_correct / max(1, train_total)

        # ----- Validate -----
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        all_logits = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                loss_val_batch = criterion(outputs, labels)
                val_loss += float(loss_val_batch.item())

                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_logits.append(outputs.cpu())

        val_loss /= max(1, len(val_loader))
        np_preds = np.array(all_preds, dtype=int)
        np_labels = np.array(all_labels, dtype=int)
        val_acc = (np_preds == np_labels).mean()
        f1_macro = f1_score(np_labels, np_preds, average='macro')
        tp = int(((np_preds == sad_idx) & (np_labels == sad_idx)).sum())
        fn = int(((np_preds != sad_idx) & (np_labels == sad_idx)).sum())
        recall_sad = tp / (tp + fn) if (tp + fn) else 0.0

        # f1_macro = f1_score(np_labels, np_preds, average='macro')
        f1_per_class = f1_score(np_labels, np_preds, labels=[happy_idx, sad_idx], average=None, zero_division=0)
        f1_happy, f1_sad = float(f1_per_class[0]), float(f1_per_class[1])
        f1_gap = abs(f1_happy - f1_sad)
        pred_dist = np.bincount(np_preds, minlength=len(CLASSES))
        cm = confusion_matrix(np_labels, np_preds, labels=list(range(len(CLASSES))))

        # ---- Threshold sweep (class-order safe) ----
        logits_val = torch.cat(all_logits, dim=0)         # [N, 2]
        probs_val = F.softmax(logits_val, dim=1).numpy()  # [N, 2]
        p_sad = probs_val[:, sad_idx]                     # [N]

        grid = np.linspace(0.35, 0.70, 36)  # 0.35..0.70 step 0.01
        best_thr, best_f1 = 0.50, -1.0
        for thr in grid:
            preds_thr_idx = np.where(p_sad >= thr, sad_idx, happy_idx)
            f1_thr = f1_score(np_labels, preds_thr_idx, average='macro')
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
        writer.add_scalar('F1/happy_val', f1_happy, epoch)
        writer.add_scalar('F1/sad_val', f1_sad, epoch)
        writer.add_scalar('F1/gap_val', f1_gap, epoch)
        fig_cm = _plot_confusion_matrix(cm, class_names=CLASSES, title=f'Val CM - Epoch {epoch+1}')
        writer.add_figure('ConfusionMatrix/val', fig_cm, epoch)
        plt.close(fig_cm)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | "
            f"Val Recall(sad): {recall_sad:.3f} | "
            f"F1: {f1_macro:.3f} | Dist: {pred_dist.tolist()} | Thr*={best_thr:.2f} | "
            f"F1_gap={f1_gap:.3f}"
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
                'sad_threshold': best_thr,
            }, Path(MODEL_DIR) / "best_model.pt")
             # persist a PNG of the best CM
            fig_best = _plot_confusion_matrix(cm, class_names=CLASSES, title=f'Best Val CM (epoch {epoch+1})')
            Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
            fig_best.savefig(Path(MODEL_DIR) / "best_val_cm.png", dpi=140, bbox_inches='tight')
            plt.close(fig_best)
            print(f"✓ Best model saved (metric: {best_monitor_metric:.3f}, sad_thr={best_thr:.2f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        if epoch > 10 and train_acc - val_acc > 0.15:
            print(f"Warning: possible overfitting (train {train_acc:.3f} vs val {val_acc:.3f}).")

    _plot_training_curves(hist, MODEL_DIR)

    # Small summary for bookkeeping
    summary = {
        "best_metric": float(best_monitor_metric),
        "monitor": MONITOR_METRIC,
        "classes": CLASSES,
        "input_dim": int(input_dim),
        "used_attention": bool(USE_ATTENTION),
        "used_extra_features": bool(USE_EXTRA_FEATURES),
        "extra_features": EXTRA_FEATURES,
        "use_focal_loss": bool(USE_FOCAL),
        "focal_gamma": float(FOCAL_GAMMA) if USE_FOCAL else None,
    }
    with open(Path(MODEL_DIR) / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Training complete. Best {MONITOR_METRIC}: {best_monitor_metric:.3f}")
    print(f"Saved model and plots to: {MODEL_DIR}")
    writer.close()

if __name__ == "__main__":
    train()
