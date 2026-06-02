# Parakeet Fine-Tune Environment

Ready-to-use Docker image for fine-tuning NVIDIA Parakeet ASR models.

## Quick Start

```bash
# Pull the pre-built image
docker pull ghcr.io/YOUR_USER/parakeet-finetune-env:latest

# Run with your data
docker run --gpus all -v /path/to/your/data:/data \
  ghcr.io/YOUR_USER/parakeet-finetune-env \
  python /opt/NeMo/examples/asr/speech_to_text_finetune.py \
    +init_from_pretrained_model=nvidia/parakeet-tdt-0.6b-v3 \
    model.train_ds.manifest_filepath=/data/manifest.jsonl \
    model.validation_ds.manifest_filepath=/data/manifest.jsonl \
    ...
```

## Build Locally

```bash
docker build -t parakeet-finetune-env .
```

## Vast.ai Template

Search for "nemo-parakeet-finetune" on vast.ai to launch pre-configured instances.

## Included Patches

All NeMo compatibility patches pre-applied:
- torchaudio ABI mismatch fix
- pytorch_lightning → lightning alias
- OneLogger/NeptuneLogger/wandb fallbacks
- Lhotse `__len__` float return fix
- TensorBoardLogger import fix

## Contents

- NeMo 2.7.3 with CUDA 12 support
- Official NeMo fine-tune scripts at `/opt/NeMo/`
- PyTorch 2.6.0 + torchaudio 2.6.0
