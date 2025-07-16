import torch
from models.cnn_bilstm import CNNBiLSTM
from config import FEATURE_SETTINGS, CLASSES, MODEL_DIR
import os

# Initialize model
model = CNNBiLSTM(
    # input_dim=FEATURE_SETTINGS['n_mfcc'], 
    input_dim=FEATURE_SETTINGS['n_mfcc'] * (1 + FEATURE_SETTINGS['use_delta'] + FEATURE_SETTINGS['use_delta_delta']),
    num_classes=len(CLASSES)
)
checkpoint = torch.load(
    MODEL_DIR / "best_model.pt",
    map_location=torch.device('cpu')
)
model.load_state_dict(checkpoint)
model.eval()

feature_dim = FEATURE_SETTINGS['n_mfcc'] \
              * (1 + int(FEATURE_SETTINGS['use_delta']) \
                    + int(FEATURE_SETTINGS['use_delta_delta']))

dummy_input = torch.randn(
    1,
    FEATURE_SETTINGS.get('max_len', 150),
    feature_dim
) 

# Export
onnx_path = MODEL_DIR / "best_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size", 1: "seq_len"}, "output": {0: "batch_size", 1: "seq_len"}},
    opset_version=12,
    do_constant_folding=True,
    export_params=True,
    verbose=False
)

print(f"✅ Exported ONNX model to {onnx_path}")