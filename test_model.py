# test_model.py
import os
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from models.cnn_bilstm import CNNBiLSTM
from config import FEATURES_DIR, CLASSES, MODEL_DIR, BATCH_SIZE, FEATURE_SETTINGS

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

# --- Load Test Data ---
def load_test_data():
    X, y = [], []
    for idx, emotion in enumerate(CLASSES):
        emotion_dir = FEATURES_DIR / 'test' / emotion
        if not emotion_dir.exists():
            continue
        for file in os.listdir(emotion_dir):
            if file.endswith(".npy"):
                X.append(str(emotion_dir / file))
                y.append(idx)
    if not X:
        raise SystemExit(f"No test features found under {FEATURES_DIR}/test/<{','.join(CLASSES)}> (.npy)")
    return X, y

def _compute_input_dim():
    return FEATURE_SETTINGS['n_mfcc'] * (
        1 + int(FEATURE_SETTINGS.get('use_delta', False)) +
        int(FEATURE_SETTINGS.get('use_delta_delta', False))
    )

# --- Testing Function ---
def test():
    print("\n Loading test data...")
    X_test, y_test = load_test_data()
    test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    print("\n Loading trained model...")
    input_dim = _compute_input_dim()
    model = CNNBiLSTM(input_dim=input_dim, num_classes=len(CLASSES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = MODEL_DIR / "best_model.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    # handle both formats: full dict or plain state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint (epoch={checkpoint.get('epoch','?')}, best_metric={checkpoint.get('best_metric','?')})")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    print("\nRunning inference on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)              # logits
            _, preds = torch.max(outputs, 1)     # argmax
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    # Reports
    os.makedirs("results", exist_ok=True)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)

    print("\n" + "="*50)
    print(" CLASSIFICATION REPORT (Test)")
    print("="*50)
    print(report)
    print(f"\nOverall Accuracy: {acc:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")

    with open("results/classification_report.txt", "w") as f:
        f.write(report)
        f.write(f"\n\nOverall Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score (Macro): {f1_macro:.4f}\n")
        f.write(f"F1 Score (Weighted): {f1_weighted:.4f}\n")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.show()

    print("\nResults saved to 'results/'")

if __name__ == "__main__":
    test()
