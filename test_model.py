import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models.cnn_bilstm import CNNBiLSTM
from config import FEATURES_DIR, CLASSES, MODEL_DIR, BATCH_SIZE
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Custom Dataset ---
class TestDataset(Dataset):
    def __init__(self):
        self.files = []
        self.labels = []
        for idx, emotion in enumerate(CLASSES):
            path = FEATURES_DIR / 'test' / emotion
            for file in os.listdir(path):
                if file.endswith(".npy"):
                    self.files.append(path / file)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx])
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# --- Evaluation ---
def evaluate():
    dataset = TestDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    model = CNNBiLSTM(input_dim=39, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location=torch.device('cpu')))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\n🎯 Test Accuracy: {acc:.2f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()
