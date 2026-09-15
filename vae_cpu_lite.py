"""Small latent VAE training path used by the Action Flow trainer.

The codec deliberately stays local and dependency-free: it is small enough to
train on a CPU and stores its weights with every latent-flow checkpoint.
"""
import json
import math
import time
from pathlib import Path


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


class TinyVAE:  # wrapped rather than subclassed at import time to keep GUI startup light
    def __new__(cls, *args, **kwargs):
        torch, nn = _torch()

        class _TinyVAE(nn.Module):
            latent_channels = 4
            downscale_factor = 4

            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1), nn.SiLU(),
                    nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
                    nn.Conv2d(64, 4, 3, 1, 1), nn.Tanh(),
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(4, 64, 4, 2, 1), nn.SiLU(),
                    nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
                    nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh(),
                )

            def encode(self, image):
                return self.encoder(image)

            def decode(self, latent):
                return self.decoder(latent)

            def forward(self, image):
                latent = self.encode(image)
                return self.decode(latent), latent

        return _TinyVAE()


def save_vae(vae, directory):
    torch, _ = _torch()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(vae.state_dict(), directory / "pytorch_model.bin")
    (directory / "config.json").write_text(json.dumps({
        "architecture": "oasis_tiny_vae_v1", "latent_channels": 4,
        "downscale_factor": 4,
    }, indent=2), encoding="utf-8")


def load_vae(directory, device, dtype=None):
    torch, _ = _torch()
    vae = TinyVAE()
    weight_path = Path(directory) / "vae" / "pytorch_model.bin"
    try:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before the safe weights_only argument existed.
        state = torch.load(weight_path, map_location="cpu")
    vae.load_state_dict(state)
    vae.to(device=device, dtype=dtype or torch.float32).eval()
    return vae


def train_vae_cpu_lite(args, dataset, event, action_maps, action_dim):
    """Train a compact codec, then an action-conditioned rectified flow in its latent space."""
    import torch
    import torch.nn.functional as F
    from diffusers import UNet2DModel
    from torch.utils.data import DataLoader

    requested = getattr(args, "device", "auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("GPU was selected, but CUDA is not available. Choose Auto or CPU.")
    device = torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")
    height, width = dataset.height, dataset.width
    if height % 4 or width % 4:
        raise ValueError("VAE CPU Lite needs a frame size divisible by 4.")
    batch_size = max(1, int(args.batch_size))
    workers = 0 if device.type == "cpu" else max(0, int(args.workers))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                        pin_memory=device.type == "cuda")
    if not len(loader):
        raise ValueError("No usable batches were found for VAE training.")
    amp = device.type == "cuda" and args.mixed_precision == "fp16"
    dtype = torch.float16 if amp else torch.float32
    vae = TinyVAE().to(device)
    vae_epochs = max(1, min(20, int(getattr(args, "vae_epochs", 5))))
    vae_opt = torch.optim.AdamW(vae.parameters(), lr=max(1e-5, float(args.learning_rate) * 5), weight_decay=1e-5)
    event(type="start", transitions=len(dataset), training=len(dataset), validation=0,
          device=str(device), total_updates=vae_epochs * len(loader) + int(args.epochs) * len(loader),
          initialization="new Tiny VAE + latent action flow", frame_gap=args.frame_gap,
          action_aggregation=args.action_aggregation, optimizer="AdamW", contrast_samples=0,
          contrast_every=0, balance_actions=False, action_counts=getattr(args, "dataset_action_counts", {}),
          camera_encoding=getattr(args, "camera_encoding", "unknown"), camera_input_source=getattr(args, "camera_input_source", "unknown"))
    update = 0
    started = time.perf_counter()
    for epoch in range(1, vae_epochs + 1):
        vae.train()
        for _previous, target, _action in loader:
            target = target.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                reconstruction, latent = vae(target)
                loss = F.l1_loss(reconstruction, target) + 0.01 * latent.square().mean()
            vae_opt.zero_grad(set_to_none=True); loss.backward(); vae_opt.step(); update += 1
            elapsed = max(1e-6, time.perf_counter() - started)
            event(type="progress", epoch=epoch, epochs=vae_epochs + int(args.epochs), update=update,
                  total_updates=vae_epochs * len(loader) + int(args.epochs) * len(loader), loss=float(loss.item()),
                  lr=vae_opt.param_groups[0]["lr"], seconds_per_update=elapsed / update,
                  raw_seconds_per_update=elapsed / update, eta=None, contrast_active=False)
    vae.eval()
    latent_h, latent_w = height // 4, width // 4
    model = UNet2DModel(sample_size=(latent_h, latent_w), in_channels=8 + int(action_dim), out_channels=4,
                        layers_per_block=1, block_out_channels=(32, 64, 96),
                        down_block_types=("DownBlock2D",) * 3, up_block_types=("UpBlock2D",) * 3,
                        norm_num_groups=8, act_fn="silu", attention_head_dim=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4)
    total = vae_epochs * len(loader) + int(args.epochs) * len(loader)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for previous, target, action in loader:
            previous, target, action = previous.to(device), target.to(device), action.to(device)
            with torch.no_grad():
                previous = vae.encode(previous)
                target = vae.encode(target)
            noise = torch.randn_like(target)
            t = torch.rand((target.shape[0], 1, 1, 1), device=device)
            mixed = (1.0 - t) * noise + t * target
            maps = action_maps(action, latent_h, latent_w, dtype=target.dtype, scale=args.action_input_scale)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                predicted = model(torch.cat([mixed, previous, maps], dim=1), (t.flatten() * 1000.0)).sample
                loss = F.mse_loss(predicted, target - noise)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); update += 1
            elapsed = max(1e-6, time.perf_counter() - started)
            event(type="progress", epoch=vae_epochs + epoch, epochs=vae_epochs + int(args.epochs), update=update,
                  total_updates=total, loss=float(loss.item()), lr=opt.param_groups[0]["lr"],
                  seconds_per_update=elapsed / update, raw_seconds_per_update=elapsed / update,
                  eta=None, contrast_active=False)
        output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output / "unet", safe_serialization=True); save_vae(vae, output / "vae")
        info = {
            "model_type": "action_conditioned_latent_vae_flow_video", "format_version": 1,
            "name": args.model_name, "resolution": f"{width}x{height}", "width": width, "height": height,
            "latent_channels": 4, "latent_downscale_factor": 4, "engine": "vae_cpu_lite",
            "action_names": list(getattr(args, "action_names", [])), "action_input_scale": float(args.action_input_scale),
            "enabled_action_names": list(getattr(args, "enabled_action_names", [])), "frame_gap": int(args.frame_gap),
            "capture_fps": getattr(args, "capture_fps", None), "epochs_completed": epoch,
            "vae_epochs_completed": vae_epochs, "created_or_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (output / "action_flow_model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        event(type="saved", epoch=epoch)
    event(type="complete", output_dir=str(args.output_dir))
