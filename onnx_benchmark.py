# onnx_benchmark.py
import argparse, os, time, numpy as np
import onnxruntime as ort

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to ONNX model, e.g. models/model_c0.onnx")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--T", type=int, default=150, help="sequence length (frames)")
    ap.add_argument("--D", type=int, default=39, help="feature dim (check your exported D)")
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        raise SystemExit(f"ONNX file not found: {args.onnx}")

    print(f"[info] Loading ONNX: {args.onnx}")
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    in_name = in_meta.name

    dummy = np.random.randn(1, args.T, args.D).astype("float32")

    for _ in range(max(1, args.warmup)):
        sess.run(None, {in_name: dummy})

    t0 = time.time()
    for _ in range(args.iters):
        sess.run(None, {in_name: dummy})
    dt = (time.time() - t0) / args.iters
    print(f"[bench] mean latency ≈ {dt*1000:.2f} ms per clip (T={args.T}, D={args.D})")

if __name__ == "__main__":
    main()
