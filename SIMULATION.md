# ASR/MoE Training Runbook

## 1. Environment Setup

```bash
cd C:\Users\Admin\Documents\MoE
conda activate ai
python -V
pip install -r ./requirements.txt
```

## 2. Data Preparation

### 2.1 Export raw datasets

**IndicTTS (Telugu):**
```bash
python ./process_data.py `
  --dataset SPRINGLab/IndicTTS_Telugu `
  --output-dir ./processed_data_indic `
  --val-size 0.1 `
  --test-size 0.1 `
  --sample-rate 22050 `
  --default-language telugu
```

**MUCS-Hinglish:**
```bash
python ./process_data.py `
  --dataset dianavdavidson/MUCS-Hinglish `
  --output-dir ./processed_data_mucs `
  --val-size 0.1 `
  --test-size 0.0 `
  --sample-rate 22050 `
  --default-language hinglish
```

### 2.2 Build simulation manifests

**Indic:**
```bash
python ./prepare_simulation_manifests.py `
  --input-dir ./processed_data_indic/manifests `
  --output-dir ./simulation_manifests_indic `
  --train-variants clean,speed_0.9,speed_1.1,noise_0.005 `
  --eval-variants clean `
  --language-tag auto
```

**MUCS:**
```bash
python ./prepare_simulation_manifests.py `
  --input-dir ./processed_data_mucs/manifests `
  --output-dir ./simulation_manifests_mucs `
  --train-variants clean,speed_0.9,speed_1.1,noise_0.005 `
  --eval-variants clean `
  --language-tag auto
```

### 2.3 (Optional) Merge Indic + MUCS

```bash
python -c "
from pathlib import Path
src_dirs = [Path('./simulation_manifests_indic'), Path('./simulation_manifests_mucs')]
out_dir = Path('./simulation_manifests_mix')
out_dir.mkdir(parents=True, exist_ok=True)
for split in ('train', 'validation', 'test'):
    out_path = out_dir / f'{split}.jsonl'
    with out_path.open('w', encoding='utf-8') as writer:
        for src in src_dirs:
            path = src / f'{split}.jsonl'
            if path.exists():
                writer.write(path.read_text(encoding='utf-8'))
print(f'Merged manifests written to: {out_dir}')
"
```

## 3. Precompute Cached Features (Recommended)

Caching removes audio decode/resample/log-mel from the training hot path.

### 3.1 Indic cache

```bash
python ./precompute_features.py `
  --manifest ./simulation_manifests_indic/train.jsonl `
  --output-dir ./cache_indic/train

python ./precompute_features.py `
  --manifest ./simulation_manifests_indic/validation.jsonl `
  --output-dir ./cache_indic/validation `
  --vocab-json ./cache_indic/train/vocab.json

python ./precompute_features.py `
  --manifest ./simulation_manifests_indic/test.jsonl `
  --output-dir ./cache_indic/test `
  --vocab-json ./cache_indic/train/vocab.json
```

### 3.2 MUCS cache

```bash
python ./precompute_features.py `
  --manifest ./simulation_manifests_mucs/train.jsonl `
  --output-dir ./cache_mucs/train

python ./precompute_features.py `
  --manifest ./simulation_manifests_mucs/validation.jsonl `
  --output-dir ./cache_mucs/validation `
  --vocab-json ./cache_mucs/train/vocab.json

python ./precompute_features.py `
  --manifest ./simulation_manifests_mucs/test.jsonl `
  --output-dir ./cache_mucs/test `
  --vocab-json ./cache_mucs/train/vocab.json
```

### 3.3 Merged cache

```bash
python ./precompute_features.py `
  --manifest ./simulation_manifests_mix/train.jsonl `
  --output-dir ./cache_mix/train

python ./precompute_features.py `
  --manifest ./simulation_manifests_mix/validation.jsonl `
  --output-dir ./cache_mix/validation `
  --vocab-json ./cache_mix/train/vocab.json

python ./precompute_features.py `
  --manifest ./simulation_manifests_mix/test.jsonl `
  --output-dir ./cache_mix/test `
  --vocab-json ./cache_mix/train/vocab.json
```

## 4. Train Models

### 4.1 Train CA-SAMoE (Competitive-Attractive SharedAdapterMoE)

**With Indic cache:**
```bash
python train_arch_moe.py --train-manifest processed_data_librispeech/manifests/train.jsonl --valid-manifest processed_data_librispeech/manifests/validation.jsonl --test-manifest processed_data_librispeech/manifests/test.jsonl --output-dir runs/exp1 --encoder-type conformer --ffn-type shared_adapter_moe --num-experts 4 --epochs 15 --batch-size 4 --lr 3e-4 --device cuda:0 --max-tokens-per-batch 50000 --pretrained-encoder facebook/wav2vec2-base --wandb-mode disabled --allow-existing-output-dir

```

**With merged cache:**
```bash
python ./train_arch_moe.py `
  --train-manifest ./cache_mix/train/manifest.jsonl `
  --valid-manifest ./cache_mix/validation/manifest.jsonl `
  --test-manifest ./cache_mix/test/manifest.jsonl `
  --output-dir ./runs/ca_samoe_mix_cached `
  --data-mode cached `
  --encoder-type conformer `
  --ffn-type shared_adapter_moe `
  --num-experts 4 `
  --epochs 20 `
  --batch-size 8 `
  --num-workers 4 `
  --pin-memory on `
  --persistent-workers on `
  --prefetch-factor 4 `
  --competition-weight 0.05 `
  --competition-interval-steps 4 `
  --competition-warmup-epochs 1 `
  --competition-batches 1 `
  --amp on `
  --profile-performance `
  --wandb-mode disabled
```

### 4.2 Train DME baseline

```bash
python ./train_dme_sim.py `
  --train-manifest ./cache_mix/train/manifest.jsonl `
  --valid-manifest ./cache_mix/validation/manifest.jsonl `
  --test-manifest ./cache_mix/test/manifest.jsonl `
  --output-dir ./runs/dme_mix_cached `
  --data-mode cached `
  --model-type smear `
  --num-experts 4 `
  --epochs 50 `
  --batch-size 8 `
  --num-workers 4 `
  --pin-memory on `
  --persistent-workers on `
  --prefetch-factor 4 `
  --amp on `
  --profile-performance `
  --log-timing-every 20 `
  --wandb-mode disabled
```

### 4.3 Train from raw manifests (fallback)

Dùng khi cần online waveform-domain randomness (noise tươi mỗi epoch):

```bash
python ./train_arch_moe.py `
  --train-manifest ./simulation_manifests_indic/train.jsonl `
  --valid-manifest ./simulation_manifests_indic/validation.jsonl `
  --test-manifest ./simulation_manifests_indic/test.jsonl `
  --output-dir ./runs/ca_samoe_indic_raw `
  --data-mode raw `
  --encoder-type conformer `
  --ffn-type shared_adapter_moe `
  --num-experts 4 `
  --epochs 10 `
  --batch-size 8 `
  --num-workers 4 `
  --pin-memory on `
  --persistent-workers on `
  --prefetch-factor 4 `
  --amp on `
  --wandb-mode disabled
```

## 5. Performance Tips

| Flag | Khi nào dùng |
|------|-------------|
| `--data-mode cached` | Luôn dùng nếu đã precompute features |
| `--num-workers 4 --pin-memory on --persistent-workers on` | Training trên CUDA |
| `--preload-cache` | RAM đủ lớn và `data` time chiếm nhiều hơn `forward+backward` |
| `--preload-to-gpu` | Cache nhỏ, vừa VRAM cùng model + optimizer |
| `--profile-performance` | In timing `data/transfer/forward/backward/optimizer` mỗi N step |

## 6. Outputs

- Best checkpoint: `runs/.../best.pt`
- Training history: `runs/.../train_history.json`
- Expert evolution events (CA-SAMoE): `runs/.../expert_evolution_epoch_*.json`
- Weights & Biases: `https://wandb.ai` (nếu bật)

## 7. Troubleshooting

- **GPU memory cao nhưng GPU utilization thấp**: Chuyển sang cached mode, check timing log. Nếu `data` > `forward+backward` thì bottleneck là input pipeline.
- **FileNotFoundError cho cached features**: Đảm bảo `--train-manifest` trỏ đến `manifest.jsonl` trong cache, không phải raw manifest.
- **Validation/test cache sai vocabulary**: Rebuild với `--vocab-json <train_cache>/vocab.json`.
- **Thiếu validation/test manifest**: Kiểm tra `--val-size`/`--test-size` khi chạy `process_data.py`.
