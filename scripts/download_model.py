"""Download the exact checkpoint used by the whole group (pinned revision)."""
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REVISION = "267b167dc01f0e4e61923ea244e8b988f84deb80"
path = snapshot_download(
    "google/ddpm-cifar10-32", revision=REVISION,
    local_dir=ROOT / "shared_framework" / "checkpoints" / "google-ddpm-cifar10-32",
    allow_patterns=["config.json", "model_index.json", "scheduler_config.json", "diffusion_pytorch_model.bin"],
)
print("model ready:", path)
