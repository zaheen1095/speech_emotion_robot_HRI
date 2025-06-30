import torch
from models.cnn_bilstm import CNNBiLSTM
from config import MODEL_DIR
import os

# Initialize model
model = CNNBiLSTM(input_dim=39, num_classes=2)
model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location=torch.device('cpu')))
model.eval()

# Dummy input (1 batch, time, features)
dummy_input = torch.randn(1, 200, 39)  # Adjust time dimension if needed

# Export
onnx_path = MODEL_DIR / "best_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size", 1: "sequence"}},
    opset_version=11,
    verbose=False
)

print(f"✅ Exported ONNX model to {onnx_path}")