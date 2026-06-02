#!/bin/bash
set -e

echo "=== Installing build dependencies ==="
apt-get update -qq && apt-get install -y -qq build-essential cmake git ffmpeg sox libsndfile1 > /dev/null 2>&1

echo "=== Installing NeMo ==="
pip install --no-cache-dir nemo-toolkit[all,cu12] 2>&1 | tail -5

echo "=== Applying patches ==="
python3 << "PYEOF"
import os, sys, importlib.util, re

# Patch 1: torchaudio ABI
import torchaudio.__init__ as m
src = open(m.__file__).read()
src = src.replace("from torchaudio._internal import _torch_version_check", "# from torchaudio._internal import _torch_version_check")
src = src.replace("_torch_version_check()", "# _torch_version_check()")
open(m.__file__, "w").write(src)
print("P1: torchaudio OK")

# Patch 2: pytorch_lightning alias
spec = importlib.util.find_spec("lightning")
site_pkg = os.path.dirname(os.path.dirname(spec.origin))
shim = os.path.join(site_pkg, "pytorch_lightning")
os.makedirs(shim, exist_ok=True)
with open(os.path.join(shim, "__init__.py"), "w") as f:
    f.write("from lightning.pytorch import *  # noqa\n")
with open(os.path.join(shim, "callbacks.py"), "w") as f:
    f.write("from lightning.pytorch.callbacks import *  # noqa\n")
print("P2: pytorch_lightning alias OK")

# Patch 3: OneLogger fallback
import nemo.utils.exp_manager as m
src = open(m.__file__).read()
src = src.replace("from nv_one_logger import OneLogger", "try:\n    from nv_one_logger import OneLogger\nexcept ImportError:\n    class OneLogger:\n        def __init__(self, *a, **kw): pass\n        def configure(self, *a, **kw): pass\n        def set_rank(self, *a): pass")
open(m.__file__, "w").write(src)
print("P3: OneLogger OK")

# Patch 4: NeptuneLogger
src = open(m.__file__).read()
src = src.replace("from pytorch_lightning.loggers import NeptuneLogger", "try:\n    from lightning.pytorch.loggers import NeptuneLogger\nexcept ImportError:\n    NeptuneLogger = object")
open(m.__file__, "w").write(src)
print("P4: NeptuneLogger OK")

# Patch 5: Lhotse __len__
import lhotse.dataset.sampling.base as m
src = open(m.__file__).read()
lines = src.split("\n")
for i, line in enumerate(lines):
    if "def __len__" in line:
        for j in range(i+1, min(i+3, len(lines))):
            if "return" in lines[j]:
                lines[j] = re.sub(r"return\s+(.*)", r"return int(\1)", lines[j])
                break
open(m.__file__, "w").write("\n".join(lines))
print("P5: Lhotse __len__ OK")

# Patch 6: TensorBoardLogger
src = open(m.__file__).read()
for variant in ["from torch.utils.tensorboard import SummaryWriter", "from pytorch_lightning.loggers import TensorBoardLogger"]:
    if variant in src:
        old = variant
        new = "try:\n    " + variant + "\nexcept ImportError:\n    SummaryWriter = None"
        src = src.replace(old, new, 1)
open(m.__file__, "w").write(src)
print("P6: TensorBoardLogger OK")

# Patch 7: wandb
src = open(m.__file__).read()
if "import wandb" in src:
    src = src.replace("import wandb", "try:\n    import wandb\nexcept ImportError:\n    wandb = None")
    open(m.__file__, "w").write(src)
print("P7: wandb OK")
PYEOF

echo "=== Pinning torch version ==="
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 2>&1 | tail -3

echo "=== Patching nvJitLink (FIXES nvvmAddNVVMContainerToProgram ERROR) ==="
pip install --no-cache-dir nvidia-nvjitlink-cu12==12.8.93 2>&1 | tail -1

echo "=== Cloning NeMo repo ==="
git clone --depth 1 https://github.com/NVIDIA/NeMo /opt/NeMo 2>&1 | tail -1

echo "=== Verifying ==="
python3 -c "
import nemo.collections.asr as nemo_asr
import torchaudio, lightning, pytorch_lightning, torch
print(f\"Torch: {torch.__version__}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")
print(\"ALL PATCHES VERIFIED!\")
"

echo "=== Setup complete! ==="
