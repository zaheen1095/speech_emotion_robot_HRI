# export_to_onnx.py
import argparse
from pathlib import Path
import numpy as np
import torch
from models.cnn_bilstm import CNNBiLSTM
from config import FEATURES_DIR, CLASSES, MODEL_DIR, FEATURE_SETTINGS, USE_ATTENTION, FEATURES_DIR_SSL

def infer_input_dim_from_features(root_dir: Path):
    # search_dirs = [FEATURES_DIR / "test", FEATURES_DIR / "train"]
    search_dirs = [root_dir / "test", root_dir / "train"]
    for root in search_dirs:
        for cls in CLASSES:
            pdir = root / cls
            if pdir.exists():
                for f in pdir.glob("*.npy"):
                    x = np.load(f)
                    dim = int(x.shape[-1])
                    print(f"[infer] sample feature: {f.name} → D={dim}")
                    return dim
    # config fallback if no features found
    d = FEATURE_SETTINGS['n_mfcc'] * (
        1 + int(FEATURE_SETTINGS.get('use_delta', False)) +
        int(FEATURE_SETTINGS.get('use_delta_delta', False))
    )
    return int(d)

def load_checkpoint_weights(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=str(MODEL_DIR / "best_model.pt"))
    ap.add_argument("--out", type=str, default=str(MODEL_DIR / "model_c0.onnx"))
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--features_root", type=str, default=str(FEATURES_DIR_SSL),
                   help="Root of .npy features (e.g., datasets/features_ssl)")
    ap.add_argument("--seq_len", type=int, default=int(FEATURE_SETTINGS.get("max_len", 150)))
    ap.add_argument("--expect_dim", type=int, default=None, help="Optional safety check: assert exported input_dim equals this (e.g., 768 for SSL, 39/45 for MFCC).")
    args = ap.parse_args()

    device = torch.device("cpu")
    input_dim = infer_input_dim_from_features(Path(args.features_root))

    model = CNNBiLSTM(input_dim=input_dim, num_classes=len(CLASSES), use_attention=USE_ATTENTION)
    state = load_checkpoint_weights(args.checkpoint, device)
    # model.load_state_dict(state, strict=True)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise SystemExit(f"[abort] state_dict load failed. This usually means checkpoint " f"D (conv in-channels) doesn't match inferred D={input_dim} "  f"from {args.features_root}. Details:\n{e}")
    model.eval()

    T = int(args.seq_len)   # seq length used in features
    dummy = torch.randn(1, T, input_dim, dtype=torch.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 1: "seq_len"}, "output": {0: "batch_size"}},
        opset_version=args.opset,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )
    print(f"✅ Exported ONNX model to {out_path}")
    print(f"[summary] seq_len={T}, input_dim={input_dim}, classes={len(CLASSES)}, opset={args.opset}")

if __name__ == "__main__":
    main()
