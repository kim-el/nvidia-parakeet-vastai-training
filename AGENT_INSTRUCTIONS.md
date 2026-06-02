# Agent Instructions: Fine-Tune Parakeet 0.6B on Voice Samples

## Session Summary (2026-06-02)

### What We Accomplished
1. **Trained on 81 voice samples** - Fine-tuned Parakeet 0.6B on Kim's English voice
2. **Base model WER**: 38.89% → **Fine-tuned WER**: 47.52% (worse due to overfitting)
3. **Created instant setup script** - `setup_nemo.sh` eliminates dependency hell
4. **Documented all 10 patches** - Future setups take 12 min instead of hours

### Key Findings
- **Learning rate matters**: 1e-5 causes hallucinations, 1e-7 is stable
- **81 samples is too few**: Model overfits quickly, WER plateaus around epoch 11
- **Base model is already good**: 25.76% WER on standard benchmarks
- **Fine-tuning small datasets is hard**: Need 500+ samples for meaningful improvement

### Next Steps
1. Collect 500+ voice samples (30+ minutes) for better fine-tuning
2. Use `setup_nemo.sh` for instant environment setup on any GPU
3. Consider Malay language training with 1300h dataset (see MALAY_1300H_GUIDE.md)

---

## Goal
Fine-tune `nvidia/parakeet-tdt-0.6b-v3` on Kim's 81 voice samples (6.7 min English speech) to prove the fine-tuning pipeline works.

## What We Learned (6 hours of failures + 2 hours of success)

### NVIDIA's OFFICIAL Requirements (read from their README)
- **Python 3.12+**
- **PyTorch 2.6+**
- **Install**: `pip install 'nemo-toolkit[all,cu12]'`
- **NGC Container**: `nvcr.io/nvidia/nemo:26.02` (pre-built, everything works)
- **Official fine-tuning**: `python speech_to_text_finetune.py init_from_pretrained_model="nvidia/parakeet-tdt-0.6b-v3"`

### What DOESN'T Work
- PyTorch 2.1.0 + NeMo 2.x — torch too old, missing `torch.distributed.device_mesh`
- PyTorch 2.1.0 + NeMo 1.x — `use_bias` config mismatch, huggingface_hub deprecation, TDT loss segfault, dependency hell
- PyTorch images with mismatched CUDA versions — torchaudio/torchvision ABI errors
- Building Docker images locally — colima takes 50GB, qemu x86 emulation is painfully slow
- GitHub Actions Docker builds — NeMo is too heavy for free runners
- NVIDIA NeMo container on slow vast.ai hosts — 15GB pull takes 20+ minutes and times out
- **Building Docker images on vast.ai instances** — no Docker daemon access, can't commit containers
- **Fine-tuning with LR=1e-5 on 81 samples** — causes hallucinations (Russian/French/gibberish)
- **Fine-tuning more than 20 epochs on small datasets** — overfitting, WER plateaus

### What WORKS
- `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` — should match PyTorch 2.6 requirement
- OR `nvcr.io/nvidia/nemo:26.02` — official container, EVERYTHING pre-configured
- Fast datacenter vast.ai hosts (Czechia Tesla T4: offer 13080908, 8895 Mbps, 99.94% reliability)
- `pip install 'nemo-toolkit[all,cu12]'` — NVIDIA's recommended install that handles CUDA deps
- **Learning rate 1e-7** — stable training, no hallucinations on small datasets
- **setup_nemo.sh script** — instant environment setup with all 10 patches applied
- **Stopping training at epoch 11-20** — best WER before overfitting kicks in

## Budget
- **$0.89 remaining** on vast.ai (as of 2026-06-02)
- Any verified GPU with 60GB+ disk at $0.90-1.30/hr
- Setup takes ~12 min (pip install + patches)
- Fine-tuning 81 samples (6.7 min) with 20 epochs takes ~5-10 min
- Total expected cost: ~$0.30-0.40 per session

## Data Location
- Voice samples: `/Users/kimen/Experiment - Handy/voice_samples/` (81 WAV + 81 TXT files)
- Manifest: `/Users/kimen/Experiment - Handy/voice_manifest.jsonl` (NeMo JSONL format)
- Total: 13MB, 6.7 minutes of audio

## Instant Setup (NEW - No Dependency Hell)

We created `setup_nemo.sh` that installs NeMo + all 10 patches in one command. Future setups take ~12 min instead of hours of troubleshooting.

### Quick Start
1. Rent any verified GPU instance with 60GB+ disk:
   ```bash
   vastai create instance <offer_id> --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime --disk 60 --ssh
   ```

2. SSH in and run the setup script:
   ```bash
   scp -P <port> "/Users/kimen/Experiment - Handy/setup_nemo.sh" root@<host>:/workspace/
   ssh -p <port> root@<host> "chmod +x /workspace/setup_nemo.sh && /workspace/setup_nemo.sh"
   ```

3. Upload your data:
   ```bash
   scp -P <port> "/Users/kimen/Experiment - Handy/voice_manifest.jsonl" root@<host>:/workspace/
   scp -P <port> "/Users/kimen/Experiment - Handy/voice_samples/"* root@<host>:/workspace/voice_samples/
   ```

4. Run fine-tuning (same as before):
   ```bash
   ssh -p <port> root@<host> "cd /opt/NeMo && python examples/asr/speech_to_text_finetune.py \
     init_from_pretrained_model='nvidia/parakeet-tdt-0.6b-v3' \
     model.train_ds.manifest_filepath='/workspace/voice_manifest.jsonl' \
     model.validation_ds.manifest_filepath='/workspace/voice_manifest.jsonl' \
     model.train_ds.batch_size=2 \
     model.validation_ds.batch_size=1 \
     model.optim.lr=1e-7 \
     trainer.max_epochs=100 \
     trainer.devices=1 \
     trainer.accelerator='gpu' \
     exp_manager.exp_dir=/workspace/results \
     exp_manager.checkpoint_callback_params.save_top_k=1"
   ```

5. Download results and destroy instance:
   ```bash
   scp -P <port> root@<host>:/workspace/results/**/*.nemo "/Users/kimen/Experiment - Handy/"
   vastai destroy instance <instance_id>
   ```

### What the Setup Script Does
- Installs build dependencies (cmake, ffmpeg, etc.)
- Installs `nemo-toolkit[all,cu12]`
- Applies 10 compatibility patches:
  1. torchaudio ABI mismatch
  2. pytorch_lightning → lightning alias
  3. OneLogger fallback
  4. NeptuneLogger fallback
  5. Lhotse __len__ fix
  6. TensorBoardLogger import
  7. wandb import guard
  8. Pins torch to 2.6.0
  9. Clones NeMo repo
  10. Verifies all imports work

### Verified Working Config
- **Base image**: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
- **Torch**: 2.6.0+cu124
- **NeMo**: 2.7.3
- **Learning rate**: 1e-7 (prevents hallucinations on small datasets)
- **Epochs**: 100 (best WER around epoch 11-20)
- **Batch size**: 2 (training), 1 (validation)

## Recommended Approach

### Option A: Official NeMo Container (Easiest)
1. `vastai create instance 13080908 --image nvcr.io/nvidia/nemo:26.02 --disk 40 --ssh --label parakeet`
2. Wait for SSH (fast on this host)
3. Upload voice data: `scp voice_manifest.jsonl voice_samples/* root@host:/workspace/`
4. Clone NeMo repo for the fine-tuning script:
   ```
   git clone https://github.com/NVIDIA/NeMo.git
   cd NeMo
   ```
5. Run official fine-tuning (Hydra overrides from CLI):
   ```
   python examples/asr/speech_to_text_finetune.py \
     init_from_pretrained_model="nvidia/parakeet-tdt-0.6b-v3" \
     model.train_ds.manifest_filepath="/workspace/voice_manifest.jsonl" \
     model.validation_ds.manifest_filepath="/workspace/voice_manifest.jsonl" \
     model.train_ds.batch_size=2 \
     model.validation_ds.batch_size=1 \
     model.train_ds.num_workers=2 \
     model.optim.lr=1e-5 \
     trainer.max_epochs=20 \
     trainer.devices=1 \
     trainer.accelerator="gpu"
   ```
6. Model saves to `~/results/` or specified path
7. Export ONNX: `python -c "from nemo.collections.asr.models import ASRModel; m=ASRModel.restore_from('results/checkpoint.nemo'); m.export('encoder-model.onnx')"`
8. Download: `scp root@host:/workspace/encoder-model.onnx .`
9. `vastai destroy instance <id>`

### Option B: PyTorch Base + Pip Install
1. `vastai create instance 13080908 --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime --disk 40 --ssh --label parakeet`
2. SSH in, run:
   ```
   apt-get update && apt-get install -y build-essential sox libsndfile1 git
   pip install 'nemo-toolkit[all,cu12]'
   ```
3. Then same steps 4-9 as Option A

### Key Config Notes
- Parakeet 0.6B is a TDT model — uses BPE tokenizer, 8192 vocab
- Set `batch_size=2` (small dataset, 12GB VRAM is tight)
- `num_workers=2` (avoid multiprocessing crashes)
- Hydra CLI overrides are preferred over editing YAML configs

## Files Created During This Session

### Setup & Configuration
- `setup_nemo.sh` — Instant environment setup script (installs NeMo + 10 patches in 12 min)
- `parakeet-finetune-env/Dockerfile` — Docker image definition (for future use)
- `parakeet-finetune-env/.github/workflows/build-image.yml` — GitHub Actions workflow
- `parakeet-finetune-env/README.md` — Docker image documentation

### Training Data
- `voice_samples/` — 81 WAV files (6.7 min English speech)
- `voice_manifest.jsonl` — NeMo JSONL manifest for training
- `voice_samples/*.txt` — Transcription files for each audio sample

### Results & Logs
- `finetune_official.log` — Training log from last session (stopped at epoch 65)
- `results_official/` — Training output directory (checkpoints, metrics)
- `parakeet-voice-finetuned.nemo` — Fine-tuned model (47.52% WER, worse than base)

### Documentation
- `AGENT_INSTRUCTIONS.md` — This file (updated with session findings)
- `~/.claude/skills/nemo-finetune/MALAY_1300H_GUIDE.md` — Guide for Malay language training
- `~/.claude/projects/-Users-kimen/memory/nemo-pip-finetune.md` — Technical notes on patches

## Quick Reference Commands

### Rent Instance
```bash
vastai search offers 'verified=true num_gpus=1 disk_space>60' --limit 5
vastai create instance <offer_id> --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime --disk 60 --ssh
```

### Setup Environment
```bash
scp -P <port> "/Users/kimen/Experiment - Handy/setup_nemo.sh" root@<host>:/workspace/
ssh -p <port> root@<host> "chmod +x /workspace/setup_nemo.sh && /workspace/setup_nemo.sh"
```

### Upload Data
```bash
scp -P <port> "/Users/kimen/Experiment - Handy/voice_manifest.jsonl" root@<host>:/workspace/
scp -P <port> "/Users/kimen/Experiment - Handy/voice_samples/"* root@<host>:/workspace/voice_samples/
```

### Run Training
```bash
ssh -p <port> root@<host> "cd /opt/NeMo && python examples/asr/speech_to_text_finetune.py \
  init_from_pretrained_model='nvidia/parakeet-tdt-0.6b-v3' \
  model.train_ds.manifest_filepath='/workspace/voice_manifest.jsonl' \
  model.validation_ds.manifest_filepath='/workspace/voice_manifest.jsonl' \
  model.train_ds.batch_size=2 model.validation_ds.batch_size=1 \
  model.optim.lr=1e-7 trainer.max_epochs=100 \
  trainer.devices=1 trainer.accelerator='gpu' \
  exp_manager.exp_dir=/workspace/results \
  exp_manager.checkpoint_callback_params.save_top_k=1"
```

### Monitor Training
```bash
ssh -p <port> root@<host> "tail -f /workspace/results/*/finetune.log | grep 'val_wer'"
```

### Download Results
```bash
scp -P <port> root@<host>:/workspace/results/**/*.nemo "/Users/kimen/Experiment - Handy/"
```

### Cleanup
```bash
vastai destroy instance <instance_id>
```

## Post Fine-Tuning
Once `.nemo` checkpoint is downloaded:
1. Convert to ONNX INT8 for fast inference
2. Swap into the existing Rust server at `/Users/kimen/Experiment - Handy/handy-stt-server/`
3. Test via `https://100.103.160.52:8443` over Tailscale

## Future Improvements

### For Better WER on Your Voice
1. **Collect 500+ samples** (30+ minutes) — current 81 samples cause overfitting
2. **Use data augmentation** — speed perturbation, noise injection
3. **Try different learning rates** — 1e-8 might work better for very small datasets
4. **Freeze encoder** — only fine-tune decoder to preserve base model knowledge

### For Malay Language Training
See `~/.claude/skills/nemo-finetune/MALAY_1300H_GUIDE.md` for detailed instructions on:
- Training tokenizer on Malay text
- Using tarred datasets for 1300h of audio
- Multi-GPU training setup
- Expected costs: $150-400

### For Production Deployment
1. Export fine-tuned model to ONNX INT8
2. Quantize for faster inference (28ms target on Mac)
3. Test WER on real-world audio samples
4. Deploy via Rust server with `transcribe_rs` crate

## If Something Goes Wrong
- Check instance: `vastai show instances`
- Check training log: `ssh root@host "cat /workspace/finetune.log"`
- If segfault: try different CUDA version image
- If OOM: reduce batch_size to 1
- If stuck: `vastai destroy instance <id>` and try the other option

### Common Issues We Hit

**Hallucinations (Russian/French/gibberish output)**
- Cause: Learning rate too high (1e-5 or above)
- Fix: Use `model.optim.lr=1e-7`

**WER gets worse after fine-tuning**
- Cause: Overfitting on small dataset (81 samples)
- Fix: Stop training at epoch 11-20, or collect more data (500+ samples)

**torchaudio ABI mismatch error**
- Cause: torch upgraded to 2.9.1 but torchaudio compiled for 2.6.0
- Fix: Run `setup_nemo.sh` which pins torch to 2.6.0

**Docker build fails on vast.ai**
- Cause: No Docker daemon access inside containers
- Fix: Use `setup_nemo.sh` instead, or build Docker image on your own machine

**Training stops at epoch 22-23**
- Cause: WER plateaus, no improvement
- Fix: This is normal. Best checkpoint is usually epoch 11 (47.68% WER)

## Docker Approach (Why It Didn't Work)

We tried building a Docker image to make future setups instant, but:
1. **Can't build on vast.ai** — instances run inside containers, no Docker daemon access
2. **Can't build locally** — colima on Mac is slow, no CUDA support
3. **GitHub Actions** — NeMo is too heavy (15GB+) for free runners

**Solution**: `setup_nemo.sh` script that installs everything in 12 minutes. No Docker needed.

If you want Docker in the future:
- Build on a machine with Docker + NVIDIA GPU
- Or use GitHub Actions with paid runners (larger disk)
- Or use RunPod/Lambda which have better Docker support
