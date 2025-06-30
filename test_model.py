import os
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from models.cnn_bilstm import CNNBiLSTM
from config import FEATURES_DIR, CLASSES, MODEL_DIR, BATCH_SIZE

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
    X = []
    y = []
    for idx, emotion in enumerate(CLASSES):
        emotion_dir = FEATURES_DIR / 'test' / emotion
        for file in os.listdir(emotion_dir):
            if file.endswith(".npy"):
                X.append(str(emotion_dir / file))
                y.append(idx)
    return X, y

# --- Testing Function ---
def test():
    print("\n Loading test data...")
    X_test, y_test = load_test_data()
    test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=BATCH_SIZE)

    print("\n Loading trained model...")
    model = CNNBiLSTM(input_dim=39, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location=torch.device('cpu')))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    test()