# ASR/MoE Simulation Runbook

This runbook matches the current code in this workspace:
- `process_data.py`
- `prepare_simulation_manifests.py`
- `precompute_features.py`
- `train_arch_moe.py`
- `train_dme_sim.py`

## 1. Environment

```bash
cd /mnt/data/khanhtl/MoE
conda activate ai
python -V
pip install -r ./requirements.txt
```

## 2. Export raw datasets

### 2.1 SPRINGLab/IndicTTS-Hindi

```bash
python ./process_data.py \
  --dataset SPRINGLab/IndicTTS-Hindi \
  --output-dir ./processed_data_indic \
  --val-size 0.1 \
  --test-size 0.1 \
  --sample-rate 22050 \
  --default-language hi
```

### 2.2 dianavdavidson/MUCS-Hinglish

```bash
python ./process_data.py \
  --dataset dianavdavidson/MUCS-Hinglish \
  --output-dir ./processed_data_mucs \
  --val-size 0.1 \
  --test-size 0.0 \
  --sample-rate 22050 \
  --default-language hinglish
```

Notes:
- MUCS keeps its original `test` split when present.
- `process_data.py` auto-detects text/id columns.

## 3. Build simulation manifests

### 3.1 Indic

```bash
python ./prepare_simulation_manifests.py \
  --input-dir ./processed_data_indic/manifests \
  --output-dir ./simulation_manifests_indic \
  --train-variants clean,speed_0.9,speed_1.1,noise_0.005 \
  --eval-variants clean \
  --language-tag auto
```

### 3.2 MUCS

```bash
python ./prepare_simulation_manifests.py \
  --input-dir ./processed_data_mucs/manifests \
  --output-dir ./simulation_manifests_mucs \
  --train-variants clean,speed_0.9,speed_1.1,noise_0.005 \
  --eval-variants clean \
  --language-tag auto
```

## 4. Optional: merge Indic + MUCS

```bash
python - <<'PY'
from pathlib import Path

src_dirs = [
    Path("./simulation_manifests_indic"),
    Path("./simulation_manifests_mucs"),
]
out_dir = Path("./simulation_manifests_mix")
out_dir.mkdir(parents=True, exist_ok=True)

for split in ("train", "validation", "test"):
    out_path = out_dir / f"{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as writer:
        for src in src_dirs:
            path = src / f"{split}.jsonl"
            if path.exists():
                writer.write(path.read_text(encoding="utf-8"))
print(f"Merged manifests written to: {out_dir}")
PY
```

Possible raw manifest roots:
- `./simulation_manifests_indic`
- `./simulation_manifests_mucs`
- `./simulation_manifests_mix`

## 5. Recommended: precompute cached features

This is the preferred path when GPU memory is occupied but GPU utilization is low. It removes audio decode, resample, and log-mel extraction from the train hot path.

Each cache directory contains:
- `manifest.jsonl`
- `features/*.pt`
- `vocab.json`
- `summary.json`

### 5.1 Cache Indic

```bash
python ./precompute_features.py \
  --manifest ./simulation_manifests_indic/train.jsonl \
  --output-dir ./cache_indic/train

python ./precompute_features.py \
  --manifest ./simulation_manifests_indic/validation.jsonl \
  --output-dir ./cache_indic/validation \
  --vocab-json ./cache_indic/train/vocab.json

python ./precompute_features.py \
  --manifest ./simulation_manifests_indic/test.jsonl \
  --output-dir ./cache_indic/test \
  --vocab-json ./cache_indic/train/vocab.json
```

### 5.2 Cache MUCS

```bash
python ./precompute_features.py \
  --manifest ./simulation_manifests_mucs/train.jsonl \
  --output-dir ./cache_mucs/train

python ./precompute_features.py \
  --manifest ./simulation_manifests_mucs/validation.jsonl \
  --output-dir ./cache_mucs/validation \
  --vocab-json ./cache_mucs/train/vocab.json

python ./precompute_features.py \
  --manifest ./simulation_manifests_mucs/test.jsonl \
  --output-dir ./cache_mucs/test \
  --vocab-json ./cache_mucs/train/vocab.json
```

### 5.3 Cache merged manifests

```bash
python ./precompute_features.py \
  --manifest ./simulation_manifests_mix/train.jsonl \
  --output-dir ./cache_mix/train

python ./precompute_features.py \
  --manifest ./simulation_manifests_mix/validation.jsonl \
  --output-dir ./cache_mix/validation \
  --vocab-json ./cache_mix/train/vocab.json

python ./precompute_features.py \
  --manifest ./simulation_manifests_mix/test.jsonl \
  --output-dir ./cache_mix/test \
  --vocab-json ./cache_mix/train/vocab.json
```

Notes:
- Validation/test should reuse the training vocabulary via `--vocab-json`.
- In cached mode, deterministic preprocessing is frozen into the cache.
- `noise_*` variants are also frozen when cached. If you need fresh random noise every epoch, use `--data-mode raw`.

## 6. Train CA-SAMoE from cached features

Example with Indic cache:

```bash
python ./train_arch_moe.py \
  --train-manifest ./cache_indic/train/manifest.jsonl \
  --valid-manifest ./cache_indic/validation/manifest.jsonl \
  --test-manifest ./cache_indic/test/manifest.jsonl \
  --output-dir ./runs/ca_samoe_indic_cached \
  --data-mode cached \
  --encoder-type conformer \
  --ffn-type shared_adapter_moe \
  --num-experts 4 \
  --epochs 20 \
  --batch-size 32 \
  --num-workers 4 \
  --pin-memory on \
  --persistent-workers on \
  --prefetch-factor 4 \
  --competition-weight 0.05 \
  --competition-interval-steps 4 \
  --competition-warmup-epochs 1 \
  --competition-batches 1 \
  --eval-every-epochs 1 \
  --early-stop-patience 5 \
  --amp on \
  --profile-performance \
  --log-timing-every 20 \
  --wandb-run-name arch-casamoe-indic
```

Example with merged cache:

```bash
python ./train_arch_moe.py \
  --train-manifest ./cache_mix/train/manifest.jsonl \
  --valid-manifest ./cache_mix/validation/manifest.jsonl \
  --test-manifest ./cache_mix/test/manifest.jsonl \
  --output-dir ./runs/ca_samoe_mix_cached \
  --data-mode cached \
  --encoder-type conformer \
  --ffn-type shared_adapter_moe \
  --num-experts 4 \
  --epochs 20 \
  --batch-size 8 \
  --num-workers 4 \
  --pin-memory on \
  --persistent-workers on \
  --prefetch-factor 4 \
  --competition-weight 0.05 \
  --competition-interval-steps 4 \
  --competition-warmup-epochs 1 \
  --competition-batches 1 \
  --amp on \
  --profile-performance \
  --wandb-mode disabled
```

## 7. Train baseline from cached features

```bash
python ./train_dme_sim.py \
  --train-manifest ./cache_mix/train/manifest.jsonl \
  --valid-manifest ./cache_mix/validation/manifest.jsonl \
  --test-manifest ./cache_mix/test/manifest.jsonl \
  --output-dir ./runs/dme_mix_cached \
  --data-mode cached \
  --model-type smear \
  --num-experts 4 \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 4 \
  --pin-memory on \
  --persistent-workers on \
  --prefetch-factor 4 \
  --amp on \
  --profile-performance \
  --log-timing-every 20 \
  --wandb-mode disabled
```

## 8. Fallback: train directly from raw manifests

Use this only when you want online waveform-domain randomness such as fresh `noise_*` every epoch.

```bash
python ./train_arch_moe.py \
  --train-manifest ./simulation_manifests_indic/train.jsonl \
  --valid-manifest ./simulation_manifests_indic/validation.jsonl \
  --test-manifest ./simulation_manifests_indic/test.jsonl \
  --output-dir ./runs/ca_samoe_indic_raw \
  --data-mode raw \
  --encoder-type conformer \
  --ffn-type shared_adapter_moe \
  --num-experts 4 \
  --epochs 10 \
  --batch-size 8 \
  --num-workers 4 \
  --pin-memory on \
  --persistent-workers on \
  --prefetch-factor 4 \
  --amp on \
  --wandb-mode disabled
```

## 9. Performance options

- `--data-mode cached`: recommended for throughput.
- `--preload-cache`: preload cached `.pt` samples into host RAM.
- `--preload-to-gpu`: try to preload cached features onto GPU. Use only when the cache is small enough to fit beside the model and optimizer states.
- `--pin-memory on`: recommended when training on CUDA.
- `--persistent-workers on`: recommended when `--num-workers > 0`.
- `--profile-performance`: prints average `data`, `transfer`, `forward`, `backward`, `optimizer` timings. CA-SAMoE also prints `competition`.

Practical guidance:
- Start with `--data-mode cached --num-workers 4 --pin-memory on --persistent-workers on`.
- Add `--preload-cache` if you have enough RAM and still see `data` time dominating.
- Only try `--preload-to-gpu` for small caches or smoke tests.

## 10. Monitoring and outputs

- Training uses tqdm progress bars (`train eX`, `valid eX`, `test`).
- Loss is printed every `--log-interval` steps.
- Timing is printed every `--log-timing-every` steps when profiling is enabled.
- Best checkpoint is saved to `runs/.../best.pt`.
- Training history is saved to `runs/.../train_history.json`.
- CA-SAMoE expert evolution events are saved to `runs/.../expert_evolution_epoch_*.json` when enabled.

## 11. Troubleshooting

- `GPU memory is high but GPU utilization is low`:
  switch to cached mode and inspect timing logs. If `data` time is larger than `forward + backward`, the input pipeline is still the bottleneck.
- `FileNotFoundError` for cached features:
  make sure `--train-manifest` points to the cache `manifest.jsonl`, not the raw manifest.
- `Validation/test cache uses wrong vocabulary`:
  rebuild valid/test with `--vocab-json <train_cache>/vocab.json`.
- `preload-to-gpu` falls back:
  this means the cache does not fit in VRAM; keep cached mode but remove `--preload-to-gpu`.
- Missing `validation.jsonl` or `test.jsonl`:
  check `--val-size` and `--test-size`, then inspect `processed_data_*/dataset_summary.json`.
