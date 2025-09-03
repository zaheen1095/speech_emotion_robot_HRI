# RUNBOOK

## Phase A: Offline augmentation → Resample
```bash
python offline_augmentation.py --in datasets/raw_audio --out datasets/augmented_audio
python resample_audio.py --in datasets/augmented_audio --sr 16000 --out datasets/resampled_audio
```

## Phase B1: Feature extraction (enriched)
```bash
python extract_features.py --in datasets/resampled_audio --out datasets/features --use-extra
```

## Phase B2/B3: Train (attention + focal + threshold sweep)
```bash
python train_model.py --config config.py
```

## Phase B4: Holdout + LOCO + Stress
```bash
python test_model.py --checkpoint models/best_model_B4_final1.pt --report-dir results/B4_holdout_ALL
python stress_eval.py --checkpoint models/best_model_B4_final1.pt --out results/C2_stress_ALL.csv
```

## Phase C: Export + Benchmark + RT inference
```bash
python export_to_onnx.py --checkpoint models/best_model_B4_final1.pt --out models/model_c0.onnx
python onnx_benchmark.py --model models/model_c0.onnx
python onnx_inference.py --model models/model_c0.onnx --audio <wav or mic>
```