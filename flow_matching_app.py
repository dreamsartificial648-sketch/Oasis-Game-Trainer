"""Simple unconditional Rectified Flow / Flow Matching image trainer and generator.

Designed as a standalone companion to the user's DDPM Tkinter application.
Models created here are intentionally stored in ``output_flow_models`` and are
not compatible with DDPMPipeline folders.

Dependencies:
    pip install torch torchvision diffusers pillow safetensors
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "output_flow_models"
GENERATIONS_DIR = APP_DIR / "flow_generations"
VIDEO_MODELS_DIR = APP_DIR / "output_flow_video_models"
VIDEO_GENERATIONS_DIR = APP_DIR / "flow_video_generations"
DREAM_CYCLE_DIR = APP_DIR / "flow_dream_cycles"
SETTINGS_FILE = APP_DIR / "flow_matching_settings.json"
STOP_FILE_NAME = "stop_flow_training.flag"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

BG = "#15171d"
PANEL = "#20232b"
FIELD = "#2b2f39"
FG = "#eef1f7"
DIM = "#aab0be"
ACCENT = "#6f98ff"

ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "4:3 (Landscape)": (4, 3),
    "3:4 (Portrait)": (3, 4),
    "3:2 (Landscape)": (3, 2),
    "2:3 (Portrait)": (2, 3),
    "16:9 (Widescreen)": (16, 9),
    "9:16 (Vertical)": (9, 16),
}


def event(**values):
    print("FLOW_EVENT:" + json.dumps(values), flush=True)


def human_time(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "calculating..."
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def model_is_valid(path):
    path = Path(path)
    try:
        info = json.loads((path / "flow_model_info.json").read_text(encoding="utf-8"))
        return info.get("model_type") == "rectified_flow" and (path / "unet" / "config.json").is_file()
    except Exception:
        return False


def find_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted((p for p in MODELS_DIR.iterdir() if p.is_dir() and model_is_valid(p)), key=lambda p: p.stat().st_mtime, reverse=True)


def scan_images(folder):
    paths = []
    for p in Path(folder).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(p)
    return sorted(paths)


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def video_model_is_valid(path):
    path = Path(path)
    try:
        info = json.loads((path / "flow_video_model_info.json").read_text(encoding="utf-8"))
        return info.get("model_type") == "autoregressive_rectified_flow_video" and (path / "unet" / "config.json").is_file()
    except Exception:
        return False


def find_video_models():
    VIDEO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted((p for p in VIDEO_MODELS_DIR.iterdir() if p.is_dir() and video_model_is_valid(p)),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def scan_videos(path):
    path = Path(path)
    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    return []


def load_reference_video_frames(video_path, frame_count, resolution):
    """Load evenly spaced reference frames, letterboxed to the model resolution."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Reference video guidance requires OpenCV. Install it with: pip install opencv-python") from exc
    from PIL import Image

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open reference video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError("The reference video does not contain readable frames.")

    wanted = max(1, int(frame_count))
    if wanted == 1:
        indices = [0]
    else:
        indices = [round(i * (total - 1) / (wanted - 1)) for i in range(wanted)]

    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame).convert("RGB")
        image.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (resolution, resolution))
        canvas.paste(image, ((resolution - image.width) // 2, (resolution - image.height) // 2))
        frames.append(canvas)
    cap.release()

    if not frames:
        raise ValueError("No readable frames could be extracted from the reference video.")
    while len(frames) < wanted:
        frames.append(frames[-1].copy())
    return frames[:wanted]


def build_video_unet(resolution, gradient_checkpointing=False):
    from diffusers import UNet2DModel
    model = UNet2DModel(
        sample_size=resolution,
        in_channels=6,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
    return model


def load_video_unet(model_dir, device=None, dtype=None):
    from diffusers import UNet2DModel
    kwargs = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = UNet2DModel.from_pretrained(str(model_dir), subfolder="unet", **kwargs)
    if int(model.config.in_channels) != 6:
        raise ValueError("The selected model is not a 6-channel autoregressive Flow video model.")
    if device is not None:
        model.to(device)
    return model


def normalized_frame_difference(image_a, image_b):
    import numpy as np
    a = np.asarray(image_a, dtype=np.float32)
    b = np.asarray(image_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Frame shapes do not match for difference calculation: {a.shape} vs {b.shape}")
    return float(np.mean(np.abs(a - b)) / 255.0)


def build_filtered_pairs(extracted, min_frame_diff=0.0, max_frame_diff=1.0):
    from PIL import Image

    extracted = [Path(p) for p in extracted]
    stats = {
        "candidate_pairs": max(0, len(extracted) - 1),
        "kept_pairs": 0,
        "skipped_too_similar": 0,
        "skipped_too_different": 0,
    }
    if not extracted:
        return [], stats

    if max_frame_diff is None or max_frame_diff <= 0:
        max_frame_diff = 1.0
    max_frame_diff = max(0.0, float(max_frame_diff))
    min_frame_diff = max(0.0, float(min_frame_diff))

    pairs = []
    for a_path, b_path in zip(extracted[:-1], extracted[1:]):
        with Image.open(a_path) as image_a, Image.open(b_path) as image_b:
            diff = normalized_frame_difference(image_a.convert("RGB"), image_b.convert("RGB"))
        if diff < min_frame_diff:
            stats["skipped_too_similar"] += 1
            continue
        if max_frame_diff < 1.0 and diff > max_frame_diff:
            stats["skipped_too_different"] += 1
            continue
        pairs.append((a_path, b_path))
        stats["kept_pairs"] += 1
    return pairs, stats


def prepare_video_frame_cache(source, cache_dir, resolution, frame_stride=1, max_frames=0,
                              min_frame_diff=0.0, max_frame_diff=1.0):
    """Extract videos or normalize frame folders into numbered JPEG sequences."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Video features require OpenCV. Install it with: pip install opencv-python") from exc
    from PIL import Image

    source = Path(source)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    pair_stats = {
        "candidate_pairs": 0,
        "kept_pairs": 0,
        "skipped_too_similar": 0,
        "skipped_too_different": 0,
    }
    sequence_index = 0

    # Treat each image-containing folder as its own sequence.
    image_groups = []
    if source.is_dir():
        by_parent = {}
        for image_path in scan_images(source):
            by_parent.setdefault(image_path.parent, []).append(image_path)
        image_groups = [sorted(group) for group in by_parent.values() if len(group) >= 2]

    for group in image_groups:
        seq_dir = cache_dir / f"frames_{sequence_index:04d}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        for idx, path in enumerate(group[::max(1, frame_stride)]):
            if max_frames and idx >= max_frames:
                break
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (resolution, resolution))
                canvas.paste(image, ((resolution-image.width)//2, (resolution-image.height)//2))
                out = seq_dir / f"{idx:06d}.jpg"
                canvas.save(out, quality=95)
                extracted.append(out)
        seq_pairs, seq_stats = build_filtered_pairs(extracted, min_frame_diff, max_frame_diff)
        pairs.extend(seq_pairs)
        for key, value in seq_stats.items():
            pair_stats[key] += value
        sequence_index += 1

    for video_path in scan_videos(source):
        seq_dir = cache_dir / f"video_{sequence_index:04d}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        extracted = []
        raw_index = 0
        kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if raw_index % max(1, frame_stride) == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                image.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (resolution, resolution))
                canvas.paste(image, ((resolution-image.width)//2, (resolution-image.height)//2))
                out = seq_dir / f"{kept:06d}.jpg"
                canvas.save(out, quality=95)
                extracted.append(out)
                kept += 1
                if max_frames and kept >= max_frames:
                    break
            raw_index += 1
        cap.release()
        seq_pairs, seq_stats = build_filtered_pairs(extracted, min_frame_diff, max_frame_diff)
        pairs.extend(seq_pairs)
        for key, value in seq_stats.items():
            pair_stats[key] += value
        sequence_index += 1
    return [(Path(a), Path(b)) for a, b in pairs], pair_stats


class FlowVideoPairDataset:
    def __init__(self, pairs, condition_noise=0.05, random_flip=False):
        from torchvision import transforms
        self.pairs = [(str(a), str(b)) for a, b in pairs]
        self.condition_noise = float(condition_noise)
        self.flip = bool(random_flip)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        from PIL import Image
        import torch
        prev_path, next_path = self.pairs[index]
        with Image.open(prev_path) as a, Image.open(next_path) as b:
            previous = self.to_tensor(a.convert("RGB"))
            target = self.to_tensor(b.convert("RGB"))
        if self.flip and random.random() < 0.5:
            previous = torch.flip(previous, dims=[2])
            target = torch.flip(target, dims=[2])
        if self.condition_noise > 0:
            previous = (previous + torch.randn_like(previous) * self.condition_noise).clamp(-1, 1)
        return previous, target


def save_video_model(model, output_dir, args, epoch, stopped=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "unet", safe_serialization=True)
    info = {
        "model_type": "autoregressive_rectified_flow_video",
        "format_version": 1,
        "name": args.model_name,
        "resolution": int(model.config.sample_size),
        "conditioning_frames": 1,
        "condition_channels": 3,
        "epochs_completed": int(epoch),
        "stopped_early": bool(stopped),
        "created_or_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_path": "linear_interpolation",
        "velocity_target": "next_frame_minus_noise",
    }
    (output_dir / "flow_video_model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def sample_next_video_frame(model, previous, steps, device, dtype, generator, method="Heun", initial_noise=None):
    import torch
    if initial_noise is None:
        x = torch.randn(previous.shape, generator=generator, device=device, dtype=dtype)
    else:
        x = initial_noise.to(device=device, dtype=dtype).clone()
    dt = 1.0 / int(steps)
    model.eval()
    with torch.inference_mode():
        for i in range(int(steps)):
            t = i / int(steps)
            ts = torch.full((previous.shape[0],), t * 1000.0, device=device, dtype=dtype)
            velocity = model(torch.cat([x, previous], dim=1), ts).sample
            if method.lower() == "heun" and i < int(steps) - 1:
                predicted = x + dt * velocity
                ts_next = torch.full((previous.shape[0],), (t + dt) * 1000.0, device=device, dtype=dtype)
                velocity_next = model(torch.cat([predicted, previous], dim=1), ts_next).sample
                x = x + 0.5 * dt * (velocity + velocity_next)
            else:
                x = x + dt * velocity
    return x.clamp(-1, 1)


def noise_transition_alpha(position, curve="Cosine"):
    """Map a 0..1 transition position to a smooth interpolation amount."""
    position = max(0.0, min(1.0, float(position)))
    if str(curve).lower() == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * position)
    return position


def slerp_noise(first, second, alpha):
    """Spherical interpolation that keeps Gaussian noise energy nearly constant."""
    import torch
    alpha = max(0.0, min(1.0, float(alpha)))
    if alpha <= 0.0:
        return first.clone()
    if alpha >= 1.0:
        return second.clone()
    a = first.float()
    b = second.float()
    batch = a.shape[0]
    a_flat = a.reshape(batch, -1)
    b_flat = b.reshape(batch, -1)
    a_norm = torch.linalg.vector_norm(a_flat, dim=1, keepdim=True).clamp_min(1e-8)
    b_norm = torch.linalg.vector_norm(b_flat, dim=1, keepdim=True).clamp_min(1e-8)
    dot = ((a_flat / a_norm) * (b_flat / b_norm)).sum(dim=1).clamp(-0.9995, 0.9995)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(1e-6)
    w_a = (torch.sin((1.0 - alpha) * omega) / sin_omega).reshape(batch, *([1] * (a.ndim - 1)))
    w_b = (torch.sin(alpha * omega) / sin_omega).reshape(batch, *([1] * (a.ndim - 1)))
    mixed = w_a * a + w_b * b
    # Match the expected standard deviation of fresh N(0,1) noise.
    dims = tuple(range(1, mixed.ndim))
    mixed = mixed / mixed.std(dim=dims, keepdim=True).clamp_min(1e-6)
    return mixed.to(dtype=first.dtype)


def hybrid_transition_noise(first, second, position, curve, refresh_strength, generator):
    """Move between random noise anchors while retaining some fresh per-frame randomness."""
    import torch
    alpha = noise_transition_alpha(position, curve)
    base = slerp_noise(first, second, alpha)
    refresh = max(0.0, min(1.0, float(refresh_strength)))
    if refresh > 0.0:
        fresh = torch.randn(base.shape, generator=generator, device=base.device, dtype=base.dtype)
        # Variance-preserving mixture: coefficients lie on a unit circle.
        base = math.sqrt(max(0.0, 1.0 - refresh * refresh)) * base + refresh * fresh
        dims = tuple(range(1, base.ndim))
        base = base / base.float().std(dim=dims, keepdim=True).clamp_min(1e-6).to(base.dtype)
    return base, alpha


def optical_flow_interpolate_pair(first, second, alpha):
    import cv2
    import numpy as np

    alpha = float(alpha)
    first_rgb = pil_to_cv_rgb(first)
    second_rgb = pil_to_cv_rgb(second)
    first_gray = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2GRAY)
    second_gray = cv2.cvtColor(second_rgb, cv2.COLOR_RGB2GRAY)

    flow_ab = cv2.calcOpticalFlowFarneback(first_gray, second_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_ba = cv2.calcOpticalFlowFarneback(second_gray, first_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    h, w = first_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    map_ax = grid_x - alpha * flow_ab[..., 0]
    map_ay = grid_y - alpha * flow_ab[..., 1]
    warped_a = cv2.remap(first_rgb, map_ax, map_ay, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    back = 1.0 - alpha
    map_bx = grid_x - back * flow_ba[..., 0]
    map_by = grid_y - back * flow_ba[..., 1]
    warped_b = cv2.remap(second_rgb, map_bx, map_by, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    blended = ((1.0 - alpha) * warped_a.astype(np.float32) + alpha * warped_b.astype(np.float32))
    return cv_rgb_to_pil(blended)


def interpolate_pil_frames(frames, transition_frames=0):
    transition_frames = max(0, int(transition_frames))
    if transition_frames <= 0 or len(frames) <= 1:
        return list(frames)

    smoothed = [frames[0].copy()]
    for first, second in zip(frames[:-1], frames[1:]):
        for i in range(1, transition_frames + 1):
            alpha = i / float(transition_frames + 1)
            smoothed.append(optical_flow_interpolate_pair(first, second, alpha))
        smoothed.append(second.copy())
    return smoothed


def constrain_pil_frame_difference(previous, current, max_difference=1.0):
    max_difference = float(max_difference)
    if max_difference <= 0 or max_difference >= 1.0:
        return current, normalized_frame_difference(previous, current), 1.0

    diff = normalized_frame_difference(previous, current)
    if diff <= max_difference:
        return current, diff, 1.0

    alpha = max(0.0, min(1.0, max_difference / max(diff, 1e-8)))
    adjusted = optical_flow_interpolate_pair(previous, current, alpha)
    adjusted_diff = normalized_frame_difference(previous, adjusted)
    return adjusted, adjusted_diff, alpha


def normalized_tensor_frame_difference(previous, current):
    import torch

    a = previous.detach().float()
    b = current.detach().float()
    if a.shape != b.shape:
        raise ValueError(f"Frame shapes do not match for tensor difference calculation: {tuple(a.shape)} vs {tuple(b.shape)}")
    return float(torch.mean(torch.abs(a - b)).item() / 2.0)


def constrain_tensor_frame_difference(previous, current, max_difference=1.0):
    max_difference = float(max_difference)
    if max_difference <= 0 or max_difference >= 1.0:
        return current.clamp(-1, 1), normalized_tensor_frame_difference(previous, current), 1.0

    diff = normalized_tensor_frame_difference(previous, current)
    if diff <= max_difference:
        return current.clamp(-1, 1), diff, 1.0

    alpha = max(0.0, min(1.0, max_difference / max(diff, 1e-8)))
    adjusted = (previous + (current - previous) * alpha).clamp(-1, 1)
    adjusted_diff = normalized_tensor_frame_difference(previous, adjusted)
    return adjusted, adjusted_diff, alpha


def video_training_worker(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.benchmark = True
    else:
        event(type="warning", message="CUDA is unavailable. Video Flow training on CPU will be extremely slow.")

    amp_enabled = device.type == "cuda" and args.mixed_precision == "fp16"
    cache_dir = Path(args.output_dir) / "video_frame_cache"
    pairs, pair_stats = prepare_video_frame_cache(
        args.data_dir,
        cache_dir,
        args.resolution,
        args.frame_stride,
        args.max_frames,
        args.min_frame_diff,
        args.max_frame_diff,
    )
    if not pairs:
        raise ValueError("No consecutive frame pairs were found. Choose a video file or a folder containing videos/frame sequences.")
    loader_kwargs = dict(batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                         pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(FlowVideoPairDataset(pairs, args.condition_noise, args.random_flip), **loader_kwargs)

    if args.continue_model:
        model = load_video_unet(args.continue_model)
        if int(model.config.sample_size) != args.resolution:
            raise ValueError("The existing video model resolution does not match the selected resolution.")
        if args.gradient_checkpointing:
            model.enable_gradient_checkpointing()
    else:
        model = build_video_unet(args.resolution, args.gradient_checkpointing)
    model.to(device)
    if device.type == "cuda": model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total_updates = args.epochs * math.ceil(len(loader) / args.gradient_accumulation)
    warmup_steps = max(1, int(total_updates * 0.05))
    def lr_lambda(step):
        if step < warmup_steps: return step / warmup_steps
        q = (step - warmup_steps) / max(1, total_updates - warmup_steps)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * q)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    stop_path = Path(args.output_dir) / STOP_FILE_NAME
    stop_path.unlink(missing_ok=True)
    optimizer.zero_grad(set_to_none=True)
    update = 0; started = time.time(); completed_epoch = 0
    event(
        type="start",
        pairs=len(pairs),
        batches=len(loader),
        total_updates=total_updates,
        device=str(device),
        candidate_pairs=pair_stats.get("candidate_pairs", len(pairs)),
        kept_pairs=pair_stats.get("kept_pairs", len(pairs)),
        skipped_too_similar=pair_stats.get("skipped_too_similar", 0),
        skipped_too_different=pair_stats.get("skipped_too_different", 0),
    )
    for epoch_num in range(1, args.epochs + 1):
        model.train(); loss_sum = 0.0
        for batch_index, (previous, clean) in enumerate(loader, 1):
            if stop_path.exists():
                save_video_model(model.float(), args.output_dir, args, completed_epoch, stopped=True)
                event(type="stopped", output_dir=args.output_dir); return
            previous = previous.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            if device.type == "cuda":
                previous = previous.to(memory_format=torch.channels_last)
                clean = clean.to(memory_format=torch.channels_last)
            noise = torch.randn_like(clean)
            b = clean.shape[0]
            t = torch.rand((b,1,1,1), device=device)
            x_t = (1-t)*noise + t*clean
            timesteps = t.flatten()*1000.0
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                pred = model(torch.cat([x_t, previous], dim=1), timesteps).sample
                flow_loss = F.mse_loss(pred.float(), (clean-noise).float())
                reconstructed = x_t + (1.0-t)*pred
                temporal_loss = F.l1_loss(reconstructed.float(), clean.float())
                loss = (flow_loss + args.temporal_loss_weight * temporal_loss) / args.gradient_accumulation
            scaler.scale(loss).backward(); loss_sum += float(loss.item())*args.gradient_accumulation
            if batch_index % args.gradient_accumulation == 0 or batch_index == len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                update += 1
                eta = (time.time()-started)/max(1,update)*max(0,total_updates-update)
                event(type="progress", epoch=epoch_num, epochs=args.epochs, update=update,
                      total_updates=total_updates, loss=loss_sum/batch_index, lr=scheduler.get_last_lr()[0], eta=eta)
        completed_epoch = epoch_num
        if epoch_num % args.save_every == 0 or epoch_num == args.epochs:
            save_video_model(model, args.output_dir, args, epoch_num)
            model.to(device)
            if device.type == "cuda": model.to(memory_format=torch.channels_last)
            event(type="saved", epoch=epoch_num, output_dir=args.output_dir)
    event(type="complete", output_dir=args.output_dir)


class FlowImageDataset:
    """Picklable image dataset for Windows DataLoader worker processes.

    Windows starts each worker as a fresh Python process, so dataset classes
    must live at module scope rather than inside ``training_worker``.
    """
    def __init__(self, files, resolution, random_flip=False):
        from torchvision import transforms

        self.files = [str(path) for path in files]
        operations = [
            transforms.Resize(int(resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(int(resolution)),
        ]
        if random_flip:
            operations.append(transforms.RandomHorizontalFlip())
        operations += [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        from PIL import Image

        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB"))


def build_unet(resolution, gradient_checkpointing=False):
    from diffusers import UNet2DModel
    # This is close to the proven DDPM app backbone, but one stage smaller to
    # keep 256px Flow Matching practical on a 12 GB RTX 3060.
    model = UNet2DModel(
        sample_size=resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
    return model


def load_unet(model_dir, device=None, dtype=None):
    from diffusers import UNet2DModel
    kwargs = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = UNet2DModel.from_pretrained(str(model_dir), subfolder="unet", **kwargs)
    if device is not None:
        model.to(device)
    return model


def tensor_to_pil(tensor):
    from torchvision.transforms.functional import to_pil_image
    return to_pil_image(((tensor.detach().float().cpu().clamp(-1, 1) + 1) / 2).clamp(0, 1))


def pil_to_normalized_tensor(image, device=None, dtype=None):
    import torch
    from torchvision import transforms
    tensor = transforms.ToTensor()(image.convert("RGB"))
    tensor = transforms.Normalize([0.5] * 3, [0.5] * 3)(tensor).unsqueeze(0)
    if device is not None or dtype is not None:
        tensor = tensor.to(device=device, dtype=dtype)
    return tensor


def pil_to_cv_rgb(image):
    import numpy as np
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def cv_rgb_to_pil(array):
    from PIL import Image
    return Image.fromarray(array.clip(0, 255).astype("uint8"), mode="RGB")


def flow_output_size(model, aspect_ratio="1:1 (Square)"):
    """Keep approximately the trained pixel area while changing shape."""
    sample_size = model.config.sample_size
    if isinstance(sample_size, (list, tuple)):
        base_h, base_w = int(sample_size[0]), int(sample_size[1])
    else:
        base_h = base_w = int(sample_size)
    rw, rh = ASPECT_RATIOS.get(aspect_ratio, (1, 1))
    area = base_h * base_w
    width = math.sqrt(area * rw / rh)
    height = math.sqrt(area * rh / rw)
    # This U-Net has four spatial downsampling operations, so both dimensions
    # must be divisible by 16 to keep skip-connection tensors aligned.
    width = max(16, int(round(width / 16)) * 16)
    height = max(16, int(round(height / 16)) * 16)
    return height, width


def sample_flow(model, batch_size, steps, device, dtype, seed, method="Heun", progress=None,
                aspect_ratio="1:1 (Square)"):
    """Integrate dx/dt=v(x,t), t=0 noise -> t=1 image."""
    import torch
    height, width = flow_output_size(model, aspect_ratio)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn((batch_size, 3, height, width), generator=generator, device=device, dtype=dtype)
    dt = 1.0 / int(steps)
    model.eval()
    with torch.inference_mode():
        for i in range(int(steps)):
            t = i / int(steps)
            # diffusers UNet time embeddings accept continuous float timesteps.
            ts = torch.full((batch_size,), t * 1000.0, device=device, dtype=dtype)
            velocity = model(x, ts).sample
            if method.lower() == "heun" and i < int(steps) - 1:
                predicted = x + dt * velocity
                ts_next = torch.full((batch_size,), (t + dt) * 1000.0, device=device, dtype=dtype)
                velocity_next = model(predicted, ts_next).sample
                x = x + 0.5 * dt * (velocity + velocity_next)
            else:
                x = x + dt * velocity
            if progress:
                progress(i + 1, int(steps))
    return [tensor_to_pil(item) for item in x]


def sample_flow_from_noise(model, initial_noise, steps, method="Heun"):
    """Integrate one or more caller-supplied noise states into images."""
    import torch
    x = initial_noise.clone()
    steps = max(1, int(steps))
    dt = 1.0 / steps
    model.eval()
    with torch.inference_mode():
        for i in range(steps):
            t = i / steps
            ts = torch.full((x.shape[0],), t * 1000.0, device=x.device, dtype=x.dtype)
            velocity = model(x, ts).sample
            if str(method).lower() == "heun" and i < steps - 1:
                predicted = x + dt * velocity
                ts_next = torch.full((x.shape[0],), (t + dt) * 1000.0, device=x.device, dtype=x.dtype)
                velocity_next = model(predicted, ts_next).sample
                x = x + 0.5 * dt * (velocity + velocity_next)
            else:
                x = x + dt * velocity
    return [tensor_to_pil(item) for item in x]


def generate_flow_dream_cycle(model, frame_count, steps, seed, method="Heun",
                              variation_strength=0.10, transition_frames=30,
                              curve="Cosine Ease", progress=None):
    """Walk locally through an image Flow model's input-noise space."""
    import torch
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    height, width = flow_output_size(model)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    shape = (1, 3, height, width)

    def normalize(noise):
        dims = tuple(range(1, noise.ndim))
        return noise / noise.square().mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-6)

    start = normalize(torch.randn(shape, generator=generator, device=device, dtype=dtype))

    def nearby(origin):
        offset = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return normalize(origin + float(variation_strength) * offset)

    target = nearby(start)
    frames = []
    transition_frames = max(2, int(transition_frames))
    for index in range(max(1, int(frame_count))):
        position = index % transition_frames
        if index and position == 0:
            start, target = target, nearby(target)
        amount = position / float(transition_frames)
        if not str(curve).lower().startswith("linear"):
            amount = 0.5 - 0.5 * math.cos(math.pi * amount)
        noise = normalize((1.0 - amount) * start + amount * target)
        frame = sample_flow_from_noise(model, noise, steps, method)[0]
        frames.append(frame)
        if progress:
            progress(index + 1, frame)
    return frames


def interpolate_dream_frames(frames, output_count):
    from PIL import Image
    if not frames:
        return []
    output_count = max(1, int(output_count))
    if len(frames) == 1:
        return [frames[0].copy() for _ in range(output_count)]
    result = []
    for index in range(output_count):
        position = index * (len(frames) - 1) / max(1, output_count - 1)
        left = int(position); right = min(left + 1, len(frames) - 1)
        amount = position - left
        result.append(frames[left].copy() if right == left else Image.blend(
            frames[left].convert("RGB"), frames[right].convert("RGB"), amount))
    return result


def save_dream_cycle(frames, path, fps):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        duration = max(1, round(1000 / max(1, int(fps))))
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    else:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("MP4 export requires OpenCV. Install it with: pip install opencv-python") from exc
        width, height = frames[0].size
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not create the MP4 video writer.")
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(pil_to_cv_rgb(frame), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
    return path


def save_model(model, output_dir, args, epoch, stopped=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "unet", safe_serialization=True)
    info = {
        "model_type": "rectified_flow",
        "format_version": 1,
        "name": args.model_name,
        "resolution": int(model.config.sample_size),
        "epochs_completed": int(epoch),
        "stopped_early": bool(stopped),
        "created_or_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_path": "linear_interpolation",
        "velocity_target": "data_minus_noise",
    }
    (output_dir / "flow_model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def training_worker(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        event(type="warning", message="CUDA is unavailable. Flow training on CPU will be extremely slow.")

    amp_enabled = device.type == "cuda" and args.mixed_precision == "fp16"
    # Keep trainable/master weights in FP32. Autocast performs suitable CUDA
    # operations in FP16, while GradScaler safely scales the FP32 gradients.
    # Converting the model itself with model.half() makes GradScaler reject the
    # native FP16 gradients during unscale_.
    training_dtype = torch.float32
    paths = scan_images(args.data_dir)
    if not paths:
        raise ValueError("No supported images were found in the selected dataset folder.")

    loader_kwargs = dict(batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                         pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(
        FlowImageDataset(paths, args.resolution, args.random_flip),
        **loader_kwargs,
    )

    if args.continue_model:
        model = load_unet(args.continue_model)
        loaded_res = int(model.config.sample_size)
        if loaded_res != args.resolution:
            raise ValueError(f"Existing model is {loaded_res}px, but training is set to {args.resolution}px.")
        if args.gradient_checkpointing:
            model.enable_gradient_checkpointing()
    else:
        model = build_unet(args.resolution, gradient_checkpointing=args.gradient_checkpointing)
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total_updates = args.epochs * math.ceil(len(loader) / args.gradient_accumulation)

    # Cosine annealing with a short linear warmup (first 5% of steps).
    warmup_steps = max(1, int(0.05 * total_updates))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_updates - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    update = 0
    started = time.time()
    stop_path = Path(args.output_dir) / STOP_FILE_NAME
    stop_path.unlink(missing_ok=True)
    event(type="start", images=len(paths), batches=len(loader), total_updates=total_updates, device=str(device))

    optimizer.zero_grad(set_to_none=True)
    completed_epoch = 0
    for epoch_num in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch_index, clean in enumerate(loader, 1):
            if stop_path.exists():
                save_model(model.float(), args.output_dir, args, completed_epoch, stopped=True)
                event(type="stopped", output_dir=args.output_dir)
                return
            clean = clean.to(device=device, dtype=training_dtype, non_blocking=True)
            if device.type == "cuda":
                clean = clean.to(memory_format=torch.channels_last)
            noise = torch.randn_like(clean)
            b = clean.shape[0]
            # Linear conditional path: x_t=(1-t)z+t*x; target velocity=x-z.
            t = torch.rand((b, 1, 1, 1), device=device, dtype=training_dtype)
            x_t = (1.0 - t) * noise + t * clean
            timesteps = (t.flatten() * 1000.0).to(dtype=training_dtype)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predicted_velocity = model(x_t, timesteps).sample
                loss = F.mse_loss(predicted_velocity.float(), (clean - noise).float()) / args.gradient_accumulation
            scaler.scale(loss).backward()
            loss_sum += float(loss.item()) * args.gradient_accumulation
            if batch_index % args.gradient_accumulation == 0 or batch_index == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                elapsed = time.time() - started
                eta = elapsed / max(1, update) * max(0, total_updates - update)
                current_lr = scheduler.get_last_lr()[0]
                event(type="progress", epoch=epoch_num, epochs=args.epochs, update=update, lr=current_lr,
                      total_updates=total_updates, loss=loss_sum / batch_index, eta=eta)
        completed_epoch = epoch_num
        if epoch_num % args.preview_every == 0 or epoch_num == args.epochs:
            preview_dir = Path(args.output_dir) / "gui_previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_image = sample_flow(
                model, 1, args.preview_steps, device, torch.float32,
                seed=12345, method="Euler",
            )[0]
            preview_path = preview_dir / f"preview_epoch_{epoch_num:04d}.png"
            preview_image.save(preview_path)
            event(type="preview", epoch=epoch_num, path=str(preview_path))
        if epoch_num % args.save_every == 0 or epoch_num == args.epochs:
            save_model(model, args.output_dir, args, epoch_num)
            model.to(device=device)
            if device.type == "cuda":
                model.to(memory_format=torch.channels_last)
            event(type="saved", epoch=epoch_num, output_dir=args.output_dir)

    event(type="complete", output_dir=args.output_dir)


class Field(ttk.Frame):
    def __init__(self, parent, label, default="", values=None, width=16):
        super().__init__(parent, style="Card.TFrame")
        ttk.Label(self, text=label, style="Meta.TLabel").pack(anchor="w")
        self.var = tk.StringVar(value=str(default))
        if values:
            self.widget = ttk.Combobox(self, textvariable=self.var, values=values, state="readonly", width=width)
        else:
            self.widget = ttk.Entry(self, textvariable=self.var, width=width, style="Field.TEntry")
        self.widget.pack(fill="x", pady=(3, 0))
    def get(self):
        return self.var.get()


def scroll_panel(parent):
    canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
    bar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=bar.set)
    bar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    inner = ttk.Frame(canvas, style="Panel.TFrame")
    window = canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
    return inner


class TrainTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, style="Panel.TFrame")
        self.app = app
        self.process = None
        self.events = queue.Queue()
        self._build()

    def _build(self):
        content = scroll_panel(self)
        ttk.Label(content, text="Train a Rectified Flow", style="Heading.TLabel").pack(anchor="w", padx=18, pady=(18, 4))
        ttk.Label(content, text="Learns a continuous velocity from random noise to your image dataset.", style="Body.TLabel").pack(anchor="w", padx=18, pady=(0, 14))
        card = ttk.Frame(content, style="Card.TFrame", padding=14)
        card.pack(fill="x", padx=18, pady=4)

        self.data_var = tk.StringVar()
        self.name_var = tk.StringVar(value="My Flow Model")
        self.output_var = tk.StringVar(value=str(MODELS_DIR / "My Flow Model"))
        self.continue_var = tk.StringVar()
        for label, var, command in [("Dataset folder", self.data_var, self.browse_data),
                                    ("Continue Flow model (optional)", self.continue_var, self.browse_continue),
                                    ("Save model to", self.output_var, self.browse_output)]:
            row = ttk.Frame(card, style="Card.TFrame"); row.pack(fill="x", pady=5)
            ttk.Label(row, text=label, style="Meta.TLabel").pack(anchor="w")
            line = ttk.Frame(row, style="Card.TFrame"); line.pack(fill="x", pady=(3, 0))
            ttk.Entry(line, textvariable=var, style="Field.TEntry").pack(side="left", fill="x", expand=True)
            ttk.Button(line, text="Browse...", command=command).pack(side="left", padx=(7, 0))
        ttk.Label(card, text="Model name", style="Meta.TLabel").pack(anchor="w", pady=(6, 0))
        name_entry = ttk.Entry(card, textvariable=self.name_var, style="Field.TEntry")
        name_entry.pack(fill="x", pady=(3, 6))
        name_entry.bind("<FocusOut>", self.sync_name)

        grid = ttk.Frame(card, style="Card.TFrame"); grid.pack(fill="x")
        self.epochs = Field(grid, "Epochs", "100")
        self.resolution = Field(grid, "Resolution", "256", ["64", "128", "256"])
        self.batch = Field(grid, "Batch size", "16", ["1", "2", "4", "8", "12", "16", "24", "32"])
        self.lr = Field(grid, "Learning rate", "0.0002", ["0.0001", "0.0002", "0.0003", "0.0005"])
        self.workers = Field(grid, "Data workers", "6", ["0", "2", "4", "6", "8", "12"])
        self.grad = Field(grid, "Gradient accumulation", "1", ["1", "2", "4"])
        self.precision = Field(grid, "Mixed precision", "fp16", ["fp16", "no"])
        self.save_every = Field(grid, "Save every N epochs", "10")
        self.preview_every = Field(grid, "Preview every N epochs", "10")
        self.preview_steps = Field(grid, "Preview flow steps", "10", ["4", "8", "10", "20"])
        fields = [self.epochs, self.resolution, self.batch, self.lr, self.workers, self.grad,
                  self.precision, self.save_every, self.preview_every, self.preview_steps]
        for i, field in enumerate(fields):
            field.grid(row=i // 2, column=i % 2, sticky="ew", padx=(0 if i % 2 == 0 else 8, 8 if i % 2 == 0 else 0), pady=6)
        grid.columnconfigure(0, weight=1); grid.columnconfigure(1, weight=1)
        self.flip_var = tk.BooleanVar(value=True)
        self.tf32_var = tk.BooleanVar(value=True)
        self.grad_ckpt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(card, text="Random horizontal flip", variable=self.flip_var).pack(anchor="w", pady=(8, 2))
        ttk.Checkbutton(card, text="Use TF32 acceleration (RTX 30-series+)", variable=self.tf32_var).pack(anchor="w", pady=(0, 2))
        ttk.Checkbutton(card, text="Gradient checkpointing (saves VRAM, allows larger batches)", variable=self.grad_ckpt_var).pack(anchor="w")

        buttons = ttk.Frame(content, style="Panel.TFrame"); buttons.pack(fill="x", padx=18, pady=12)
        self.start_btn = ttk.Button(buttons, text="Start Training", command=self.start, style="Accent.TButton")
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(buttons, text="Stop && Save", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        self.progress = ttk.Progressbar(content, mode="determinate")
        self.progress.pack(fill="x", padx=18, pady=(0, 6))
        self.status = ttk.Label(content, text="Ready.", style="Body.TLabel")
        self.status.pack(anchor="w", padx=18)
        preview_box = ttk.Frame(content, style="Card.TFrame", padding=10)
        preview_box.pack(fill="x", padx=18, pady=(8, 0))
        ttk.Label(preview_box, text="Training Preview", style="CardHeading.TLabel").pack(anchor="w", pady=(0, 6))
        self.preview_label = tk.Label(preview_box, text="A preview will appear here after the first preview interval.",
                                      bg=FIELD, fg=DIM, height=12)
        self.preview_label.pack(fill="both", expand=True)
        self.log = tk.Text(content, height=12, bg=FIELD, fg=FG, insertbackground=FG, relief="flat", wrap="word")
        self.log.pack(fill="both", expand=True, padx=18, pady=(8, 18))

    def sync_name(self, _event=None):
        if self.name_var.get().strip():
            self.output_var.set(str(MODELS_DIR / self.name_var.get().strip()))
    def browse_data(self):
        value = filedialog.askdirectory(title="Select image dataset")
        if value:
            self.data_var.set(value)
            if not self.name_var.get().strip() or self.name_var.get() == "My Flow Model":
                self.name_var.set(Path(value).name + " Flow")
                self.sync_name()
    def browse_continue(self):
        value = filedialog.askdirectory(initialdir=MODELS_DIR, title="Select an existing Flow model")
        if value:
            if not model_is_valid(value):
                messagebox.showerror("Not a Flow model", "Choose a folder containing flow_model_info.json and an unet folder.")
            else:
                self.continue_var.set(value)
    def browse_output(self):
        value = filedialog.askdirectory(initialdir=MODELS_DIR)
        if value: self.output_var.set(value)
    def append(self, text):
        self.log.insert("end", text.rstrip() + "\n"); self.log.see("end")

    def start(self):
        try:
            data = Path(self.data_var.get())
            if not data.is_dir(): raise ValueError("Select a valid dataset folder.")
            if not scan_images(data): raise ValueError("That folder contains no supported images.")
            values = dict(epochs=int(self.epochs.get()), resolution=int(self.resolution.get()), batch=int(self.batch.get()),
                          lr=float(self.lr.get()), workers=int(self.workers.get()), grad=int(self.grad.get()),
                          save=int(self.save_every.get()), preview=int(self.preview_every.get()),
                          preview_steps=int(self.preview_steps.get()))
            if min(values.values()) <= 0: raise ValueError("All numeric settings must be greater than zero.")
            output = Path(self.output_var.get())
            if output.exists() and any(output.iterdir()) and not self.continue_var.get():
                if not messagebox.askyesno("Replace model?", "The save folder is not empty. Replace its Flow model files?"): return
            output.mkdir(parents=True, exist_ok=True)
            (output / STOP_FILE_NAME).unlink(missing_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--train-worker", "--data-dir", str(data),
                       "--output-dir", str(output), "--model-name", self.name_var.get().strip() or output.name,
                       "--epochs", str(values["epochs"]), "--resolution", str(values["resolution"]),
                       "--batch-size", str(values["batch"]), "--learning-rate", str(values["lr"]),
                       "--workers", str(values["workers"]), "--gradient-accumulation", str(values["grad"]),
                       "--mixed-precision", self.precision.get(), "--save-every", str(values["save"]),
                       "--preview-every", str(values["preview"]), "--preview-steps", str(values["preview_steps"])]
            if self.continue_var.get(): command += ["--continue-model", self.continue_var.get()]
            if self.flip_var.get(): command.append("--random-flip")
            if self.tf32_var.get(): command.append("--tf32")
            if self.grad_ckpt_var.get(): command.append("--gradient-checkpointing")
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                            encoding="utf-8", errors="replace", bufsize=1, creationflags=flags)
            self.start_btn.config(state="disabled"); self.stop_btn.config(state="normal")
            self.app.set_busy(True); self.status.config(text="Training is starting..."); self.progress["value"] = 0
            threading.Thread(target=self.read_output, daemon=True).start()
            self.after(100, self.poll)
        except Exception as exc:
            messagebox.showerror("Cannot start training", str(exc))

    def read_output(self):
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            self.events.put(line.rstrip())
        self.events.put(("exit", self.process.wait()))
    def poll(self):
        try:
            while True:
                item = self.events.get_nowait()
                if isinstance(item, tuple):
                    self.finished(item[1]); return
                if item.startswith("FLOW_EVENT:"):
                    self.handle_event(json.loads(item.split(":", 1)[1]))
                else: self.append(item)
        except queue.Empty: pass
        if self.process: self.after(100, self.poll)
    def handle_event(self, info):
        kind = info.get("type")
        if kind == "progress":
            self.progress["maximum"] = info["total_updates"]; self.progress["value"] = info["update"]
            lr_str = f"  •  lr {info['lr']:.2e}" if "lr" in info else ""
            self.status.config(text=f"Epoch {info['epoch']}/{info['epochs']}  •  loss {info['loss']:.4f}{lr_str}  •  ETA {human_time(info['eta'])}")
        elif kind == "start": self.append(f"Training {info['images']} images on {info['device']} ({info['total_updates']} optimizer steps).")
        elif kind == "saved": self.append(f"Saved Flow model at epoch {info['epoch']}.")
        elif kind == "preview":
            self.show_training_preview(info.get("path"), info.get("epoch"))
        elif kind == "warning": self.append("WARNING: " + info.get("message", ""))
        elif kind == "stopped": self.status.config(text="Stopped safely; current model was saved.")
        elif kind == "complete": self.status.config(text="Training complete!")
    def show_training_preview(self, path, epoch):
        try:
            from PIL import Image, ImageTk
            image = Image.open(path).convert("RGB")
            image.thumbnail((420, 320))
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview_label.config(image=self.preview_photo, text="", height=0)
            self.append(f"Generated training preview for epoch {epoch}.")
        except Exception as exc:
            self.append(f"Could not display training preview: {exc}")
    def stop(self):
        if self.process:
            Path(self.output_var.get(), STOP_FILE_NAME).touch()
            self.stop_btn.config(state="disabled"); self.status.config(text="Stopping after the current batch, then saving...")
    def finished(self, code):
        self.process = None; self.start_btn.config(state="normal"); self.stop_btn.config(state="disabled"); self.app.set_busy(False)
        self.app.generate_tab.refresh_models()
        if hasattr(self.app, "dream_cycle_tab"): self.app.dream_cycle_tab.refresh_models()
        if code != 0:
            self.status.config(text=f"Training stopped with error code {code}. See the log.")
            messagebox.showerror("Training error", "Flow training stopped with an error. The details are in the training log.")


class GenerateTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, style="Panel.TFrame")
        self.app = app; self.model_paths = {}; self.last_images = []; self.last_generation = None
        self.loaded_key = None; self.loaded_model = None
        self._build(); self.refresh_models()
    def _build(self):
        main = ttk.Frame(self, style="Panel.TFrame", padding=18); main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1); main.rowconfigure(2, weight=1)
        controls = ttk.Frame(main, style="Card.TFrame", padding=12); controls.grid(row=0, column=0, rowspan=3, sticky="nsw", padx=(0, 14))
        ttk.Label(controls, text="Generation", style="CardHeading.TLabel").pack(anchor="w", pady=(0, 8))
        self.model_var = tk.StringVar()
        self.model_box = ttk.Combobox(controls, textvariable=self.model_var, state="readonly", width=30)
        self.model_box.pack(fill="x", pady=4)
        ttk.Button(controls, text="Refresh Models", command=self.refresh_models).pack(fill="x", pady=4)
        self.steps = Field(controls, "Flow steps", "20", ["1", "2", "4", "8", "10", "20", "30", "50"]); self.steps.pack(fill="x", pady=5)
        self.method = Field(controls, "ODE method", "Heun", ["Heun", "Euler"]); self.method.pack(fill="x", pady=5)
        self.count = Field(controls, "Images", "1", ["1", "2", "4", "8"]); self.count.pack(fill="x", pady=5)
        self.aspect = Field(controls, "Aspect ratio", "1:1 (Square)", list(ASPECT_RATIOS)); self.aspect.pack(fill="x", pady=5)
        self.display_size = Field(controls, "Preview size (display only)", "512", ["256", "384", "512", "640", "768"]); self.display_size.pack(fill="x", pady=5)
        self.display_size.widget.bind("<<ComboboxSelected>>", lambda _e: self.show_last_preview())
        self.seed = Field(controls, "Seed", str(random.randrange(2**31))); self.seed.pack(fill="x", pady=5)
        ttk.Button(controls, text="Randomize Seed", command=lambda: self.seed.var.set(str(random.randrange(2**31)))).pack(fill="x", pady=4)
        self.auto_seed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Auto-randomize seed on each generation", variable=self.auto_seed_var).pack(anchor="w", pady=(0, 4))
        self.output_var = tk.StringVar(value=str(GENERATIONS_DIR))
        ttk.Label(controls, text="Save images to", style="Meta.TLabel").pack(anchor="w", pady=(7, 0))
        ttk.Entry(controls, textvariable=self.output_var, style="Field.TEntry").pack(fill="x", pady=3)
        ttk.Button(controls, text="Browse...", command=self.browse_output).pack(fill="x", pady=4)
        self.generate_btn = ttk.Button(controls, text="Generate", command=self.generate, style="Accent.TButton")
        self.generate_btn.pack(fill="x", pady=(12, 4))
        self.save_btn = ttk.Button(controls, text="Save Last Image(s)", command=self.save_last, state="disabled")
        self.save_btn.pack(fill="x", pady=4)
        ttk.Label(controls, text="Heun usually looks smoother; it evaluates the model twice per step. Start around 20 steps.", style="CardBody.TLabel", wraplength=230, justify="left").pack(fill="x", pady=5)
        self.status = ttk.Label(main, text="Ready.", style="Body.TLabel"); self.status.grid(row=0, column=1, sticky="w")
        self.progress = ttk.Progressbar(main, mode="determinate"); self.progress.grid(row=1, column=1, sticky="ew", pady=8)
        frame = ttk.Frame(main, style="Card.TFrame"); frame.grid(row=2, column=1, sticky="nsew")
        frame.rowconfigure(0, weight=1); frame.columnconfigure(0, weight=1)
        self.image_label = tk.Label(frame, text="Generated image will appear here", bg=FIELD, fg=DIM)
        self.image_label.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=2, pady=2)
        # Navigation bar — shown only when more than one image was generated
        self.nav_frame = ttk.Frame(frame, style="Card.TFrame")
        self.nav_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        self.prev_btn = ttk.Button(self.nav_frame, text="◀ Prev", command=self._prev_image)
        self.prev_btn.pack(side="left", padx=8)
        self.img_counter_var = tk.StringVar(value="")
        ttk.Label(self.nav_frame, textvariable=self.img_counter_var, style="CardBody.TLabel").pack(side="left", expand=True)
        self.next_btn = ttk.Button(self.nav_frame, text="Next ▶", command=self._next_image)
        self.next_btn.pack(side="right", padx=8)
        self.nav_frame.grid_remove()
        self._preview_index = 0
    def refresh_models(self):
        models = find_models(); self.model_paths = {p.name: p for p in models}; self.model_box["values"] = list(self.model_paths)
        if models and self.model_var.get() not in self.model_paths: self.model_var.set(models[0].name)
    def browse_output(self):
        value = filedialog.askdirectory(initialdir=GENERATIONS_DIR)
        if value: self.output_var.set(value)
    def generate(self):
        try:
            key = self.model_var.get()
            if key not in self.model_paths: raise ValueError("Select a trained Flow model.")
            if self.auto_seed_var.get():
                self.seed.var.set(str(random.randrange(2**31)))
            steps, count, seed = int(self.steps.get()), int(self.count.get()), int(self.seed.get())
            if steps < 1 or count < 1: raise ValueError("Steps and image count must be positive.")
            self.generate_btn.config(state="disabled"); self.progress["maximum"] = steps; self.progress["value"] = 0
            self.status.config(text="Loading model and generating..."); self.app.set_generation_busy(True)
            threading.Thread(target=self.generate_worker,
                             args=(key, steps, count, seed, self.method.get(), self.aspect.get()),
                             daemon=True).start()
        except Exception as exc: messagebox.showerror("Cannot generate", str(exc))
    def generate_worker(self, key, steps, count, seed, method, aspect_ratio):
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            if self.loaded_key != key or self.loaded_model is None:
                self.loaded_model = None
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                self.loaded_model = load_unet(self.model_paths[key], device=device, dtype=dtype)
                self.loaded_key = key
            def progress(done, total): self.after(0, lambda: self.progress.configure(value=done, maximum=total))
            images = sample_flow(self.loaded_model, count, steps, device, dtype, seed, method, progress,
                                 aspect_ratio=aspect_ratio)
            self.after(0, lambda: self.generation_done(images, key, seed, aspect_ratio))
        except Exception:
            error = traceback.format_exc()
            self.after(0, lambda: self.generation_failed(error))
    def generation_done(self, images, key, seed, aspect_ratio):
        self.last_images = images
        self.last_generation = {"model": key, "seed": seed, "aspect_ratio": aspect_ratio}
        self._preview_index = 0
        self.show_last_preview()
        width, height = images[0].size
        self.status.config(text=f"Generated {len(images)} image(s) at {width}×{height}. Click Save Last Image(s) to keep them.")
        self.save_btn.config(state="normal")
        self.generate_btn.config(state="normal"); self.app.set_generation_busy(False)
    def show_last_preview(self):
        if not self.last_images:
            return
        from PIL import Image, ImageTk
        # Clamp index in case image count changed
        self._preview_index = max(0, min(self._preview_index, len(self.last_images) - 1))
        size = max(128, int(self.display_size.get()))
        source = self.last_images[self._preview_index]
        scale = size / max(source.width, source.height)
        preview = source.resize(
            (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
            Image.Resampling.LANCZOS,
        )
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.image_label.config(image=self.preview_photo, text="")
        # Show/hide nav bar and update counter
        if len(self.last_images) > 1:
            self.img_counter_var.set(f"Image {self._preview_index + 1} of {len(self.last_images)}")
            self.prev_btn.config(state="normal" if self._preview_index > 0 else "disabled")
            self.next_btn.config(state="normal" if self._preview_index < len(self.last_images) - 1 else "disabled")
            self.nav_frame.grid()
        else:
            self.nav_frame.grid_remove()
    def _prev_image(self):
        if self._preview_index > 0:
            self._preview_index -= 1
            self.show_last_preview()
    def _next_image(self):
        if self._preview_index < len(self.last_images) - 1:
            self._preview_index += 1
            self.show_last_preview()
    def save_last(self):
        if not self.last_images or not self.last_generation:
            messagebox.showinfo("Nothing to save", "Generate an image first.")
            return
        try:
            base_out = Path(self.output_var.get())
            key = self.last_generation["model"]

            # Keep generations organized by model, matching the Stable Diffusion app.
            # Example: flow_generations/My Flow Model/*.png
            safe_model_name = "".join(
                c if c not in '<>:"/\\|?*' else "_"
                for c in key
            ).strip().rstrip(".") or "Unnamed Flow Model"
            out = base_out / safe_model_name
            out.mkdir(parents=True, exist_ok=True)

            stamp = time.strftime("%Y%m%d_%H%M%S")
            seed = self.last_generation["seed"]
            paths = []
            for i, image in enumerate(self.last_images, 1):
                path = out / f"{safe_model_name}_flow_{stamp}_seed{seed}_{i}.png"
                image.save(path)
                paths.append(path)
            self.status.config(text=f"Saved {len(paths)} image(s) to {out}")
        except Exception as exc:
            messagebox.showerror("Could not save images", str(exc))
    def generation_failed(self, error):
        self.status.config(text="Generation failed."); self.generate_btn.config(state="normal"); self.app.set_generation_busy(False)
        messagebox.showerror("Generation error", error[-1800:])


class DreamCycleTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, style="Panel.TFrame")
        self.app = app; self.model_paths = {}; self.loaded_key = None; self.loaded_model = None
        self.preview_photo = None; self.last_video_path = None
        self._build(); self.refresh_models()

    def _build(self):
        main = ttk.Frame(self, style="Panel.TFrame", padding=18); main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1); main.rowconfigure(2, weight=1)
        controls = ttk.Frame(main, style="Card.TFrame", padding=12); controls.grid(row=0, column=0, rowspan=3, sticky="nsw", padx=(0,14))
        ttk.Label(controls, text="Flow Dream Cycler", style="CardHeading.TLabel").pack(anchor="w", pady=(0,8))
        ttk.Label(controls, text="GAN-style local noise walk", style="Meta.TLabel").pack(anchor="w")
        self.model_var=tk.StringVar(); self.model_box=ttk.Combobox(controls,textvariable=self.model_var,state="readonly",width=30)
        self.model_box.pack(fill="x",pady=4)
        row=ttk.Frame(controls,style="Card.TFrame"); row.pack(fill="x")
        ttk.Button(row,text="Refresh",command=self.refresh_models).pack(side="left",fill="x",expand=True)
        ttk.Button(row,text="Use Generation Model",command=self.use_generation_model).pack(side="left",padx=(6,0))
        ttk.Button(controls,text="Use Video Tab Model",command=self.use_video_model).pack(fill="x",pady=(4,0))
        self.start_image_var=tk.StringVar(); ttk.Label(controls,text="Starting image (required for video models)",style="Meta.TLabel").pack(anchor="w",pady=(6,0))
        startrow=ttk.Frame(controls,style="Card.TFrame"); startrow.pack(fill="x",pady=3)
        ttk.Entry(startrow,textvariable=self.start_image_var,style="Field.TEntry").pack(side="left",fill="x",expand=True)
        ttk.Button(startrow,text="Browse...",command=self.browse_start_image).pack(side="left",padx=(6,0))
        self.seconds=Field(controls,"Length (seconds)","6",["2","4","6","8","10"]); self.seconds.pack(fill="x",pady=5)
        self.dream_fps=Field(controls,"Dream frames / second","6",["2","4","6","8"]); self.dream_fps.pack(fill="x",pady=5)
        self.output_fps=Field(controls,"Output FPS","24",["12","24","30","60"]); self.output_fps.pack(fill="x",pady=5)
        self.steps=Field(controls,"Flow steps","20",["4","8","10","20","30","50"]); self.steps.pack(fill="x",pady=5)
        self.method=Field(controls,"ODE method","Heun",["Heun","Euler"]); self.method.pack(fill="x",pady=5)
        self.variation=Field(controls,"Variation per waypoint (%)","10",["3","5","8","10","15","25"]); self.variation.pack(fill="x",pady=5)
        self.transition=Field(controls,"Dream frames per transition","30",["8","16","24","30","48","60"]); self.transition.pack(fill="x",pady=5)
        self.curve=Field(controls,"Movement curve","Cosine Ease",["Cosine Ease","Linear"]); self.curve.pack(fill="x",pady=5)
        self.seed=Field(controls,"Seed",str(random.randrange(2**31))); self.seed.pack(fill="x",pady=5)
        ttk.Button(controls,text="Randomize Seed",command=lambda:self.seed.var.set(str(random.randrange(2**31)))).pack(fill="x",pady=3)
        self.output_var=tk.StringVar(value=str(DREAM_CYCLE_DIR)); ttk.Label(controls,text="Save dream cycles to",style="Meta.TLabel").pack(anchor="w",pady=(5,0))
        ttk.Entry(controls,textvariable=self.output_var,style="Field.TEntry").pack(fill="x",pady=3)
        outrow=ttk.Frame(controls,style="Card.TFrame"); outrow.pack(fill="x")
        ttk.Button(outrow,text="Browse...",command=self.browse_output).pack(side="left",fill="x",expand=True)
        self.format_var=tk.StringVar(value="mp4"); ttk.Combobox(outrow,textvariable=self.format_var,values=["mp4","gif"],state="readonly",width=6).pack(side="left",padx=(6,0))
        self.generate_btn=ttk.Button(controls,text="Generate Dream Cycle",command=self.generate,style="Accent.TButton"); self.generate_btn.pack(fill="x",pady=(10,4))
        ttk.Label(controls,text="Image models map nearby noise inputs independently. Video models also condition every output on the preceding frame, using the starting image as frame zero.",style="CardBody.TLabel",wraplength=245,justify="left").pack(fill="x",pady=5)
        self.status=ttk.Label(main,text="Ready for a local noise-space walk.",style="Body.TLabel"); self.status.grid(row=0,column=1,sticky="w")
        self.progress=ttk.Progressbar(main,mode="determinate"); self.progress.grid(row=1,column=1,sticky="ew",pady=8)
        preview=ttk.Frame(main,style="Card.TFrame"); preview.grid(row=2,column=1,sticky="nsew"); preview.rowconfigure(0,weight=1); preview.columnconfigure(0,weight=1)
        self.preview=tk.Label(preview,text="Dream-cycle frames will appear here",bg=FIELD,fg=DIM); self.preview.grid(row=0,column=0,sticky="nsew",padx=2,pady=2)

    def refresh_models(self):
        image_models=find_models(); video_models=find_video_models(); self.model_paths={}
        for path in image_models: self.model_paths[f"[Image] {path.name}"]=("image",path)
        for path in video_models: self.model_paths[f"[Video] {path.name}"]=("video",path)
        labels=list(self.model_paths); self.model_box["values"]=labels
        if labels and self.model_var.get() not in self.model_paths: self.model_var.set(labels[0])

    def use_generation_model(self):
        key=self.app.generate_tab.model_var.get()
        label=f"[Image] {key}"
        if label in self.model_paths: self.model_var.set(label); self.status.config(text=f"Using Generation model: {key}")
        else: messagebox.showerror("No model selected","Select a trained model in the Generation tab first.")

    def use_video_model(self):
        key=self.app.video_tab.v_model_var.get(); label=f"[Video] {key}"
        if label in self.model_paths: self.model_var.set(label); self.status.config(text=f"Using Video model: {key}")
        else: messagebox.showerror("No video model selected","Select a trained model in the Video tab first.")

    def browse_start_image(self):
        value=filedialog.askopenfilename(filetypes=[("Images","*.png *.jpg *.jpeg *.webp *.bmp")])
        if value: self.start_image_var.set(value)

    def browse_output(self):
        value=filedialog.askdirectory(initialdir=self.output_var.get() or DREAM_CYCLE_DIR)
        if value: self.output_var.set(value)

    def generate(self):
        try:
            key=self.model_var.get()
            if key not in self.model_paths: raise ValueError("Select a trained Flow image or video model.")
            model_type,_model_path=self.model_paths[key]
            start=Path(self.start_image_var.get()) if self.start_image_var.get().strip() else None
            if model_type=="video" and (start is None or not start.is_file()): raise ValueError("Video Flow models require a starting image.")
            seconds=float(self.seconds.get()); dream_fps=float(self.dream_fps.get()); output_fps=int(self.output_fps.get())
            steps=int(self.steps.get()); variation=float(self.variation.get())/100.0; transition=int(self.transition.get()); seed=int(self.seed.get())
            if min(seconds,dream_fps,output_fps,steps,variation,transition)<=0: raise ValueError("All numeric settings must be greater than zero.")
            frame_count=max(2,round(seconds*dream_fps)); output_count=max(2,round(seconds*output_fps))
            if frame_count>240 and not messagebox.askyesno("Large render",f"This will generate {frame_count} full Flow frames. Continue?"): return
            self.generate_btn.config(state="disabled"); self.progress.config(value=0,maximum=frame_count)
            self.status.config(text=f"Generating dream frame 0/{frame_count}..."); self.app.set_dream_busy(True)
            args=(key,start,frame_count,output_count,output_fps,steps,variation,transition,seed,self.method.get(),self.curve.get(),self.format_var.get())
            threading.Thread(target=self._worker,args=args,daemon=True).start()
        except Exception as exc: messagebox.showerror("Cannot generate Dream Cycle",str(exc))

    def _worker(self,key,start_image,frame_count,output_count,output_fps,steps,variation,transition,seed,method,curve,extension):
        try:
            import torch
            from PIL import Image
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); dtype=torch.float16 if device.type=="cuda" else torch.float32
            model_type,model_path=self.model_paths[key]
            if self.loaded_key!=key or self.loaded_model is None:
                self.loaded_model=None
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                loader=load_video_unet if model_type=="video" else load_unet
                self.loaded_model=loader(model_path,device=device,dtype=dtype); self.loaded_key=key
            def update(done,frame): self.after(0,lambda:self._show_progress(done,frame_count,frame))
            if model_type=="image":
                frames=generate_flow_dream_cycle(self.loaded_model,frame_count,steps,seed,method,variation,transition,curve,update)
            else:
                res=int(self.loaded_model.config.sample_size)
                image=Image.open(start_image).convert("RGB"); image.thumbnail((res,res),Image.Resampling.LANCZOS)
                canvas=Image.new("RGB",(res,res)); canvas.paste(image,((res-image.width)//2,(res-image.height)//2))
                previous=pil_to_normalized_tensor(canvas,device=device,dtype=dtype)
                generator=torch.Generator(device=device).manual_seed(seed)
                anchor_a=torch.randn(previous.shape,generator=generator,device=device,dtype=dtype)
                anchor_b=slerp_noise(anchor_a,anchor_a+variation*torch.randn(previous.shape,generator=generator,device=device,dtype=dtype),1.0)
                frames=[canvas]; update(1,canvas)
                for index in range(1,frame_count):
                    position=index%transition
                    if position==0:
                        anchor_a=anchor_b
                        anchor_b=slerp_noise(anchor_a,anchor_a+variation*torch.randn(previous.shape,generator=generator,device=device,dtype=dtype),1.0)
                    amount=position/float(transition)
                    if not str(curve).lower().startswith("linear"): amount=0.5-0.5*math.cos(math.pi*amount)
                    noise=slerp_noise(anchor_a,anchor_b,amount)
                    previous=sample_next_video_frame(self.loaded_model,previous,steps,device,dtype,generator,method,initial_noise=noise).detach()
                    frame=tensor_to_pil(previous[0]); frames.append(frame); update(index+1,frame)
            output_frames=interpolate_dream_frames(frames,output_count)
            display_name=model_path.name; safe="".join(c if c not in '<>:"/\\|?*' else '_' for c in display_name).strip() or "Flow Model"
            folder=Path(self.output_var.get())/safe; stamp=time.strftime("%Y%m%d_%H%M%S")
            path=folder/f"{safe}_dream_cycle_{stamp}_seed{seed}.{extension}"
            save_dream_cycle(output_frames,path,output_fps)
            info={"model":display_name,"model_type":model_type,"starting_image":str(start_image) if start_image else None,"seed":seed,"dream_frames":frame_count,"output_frames":output_count,"output_fps":output_fps,"flow_steps":steps,"method":method,"variation":variation,"transition_frames":transition,"curve":curve}
            (folder/f"{path.stem}.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
            self.after(0,lambda:self._done(path,frame_count,output_count,output_fps))
        except Exception:
            error=traceback.format_exc(); self.after(0,lambda:self._failed(error))

    def _show_progress(self,done,total,frame):
        from PIL import Image,ImageTk
        self.progress.config(value=done,maximum=total); self.status.config(text=f"Generating dream frame {done}/{total}...")
        scale=min(512/max(frame.size),1.0); image=frame.resize((max(1,round(frame.width*scale)),max(1,round(frame.height*scale))),Image.Resampling.LANCZOS)
        self.preview_photo=ImageTk.PhotoImage(image); self.preview.config(image=self.preview_photo,text="")

    def _done(self,path,dream_count,output_count,fps):
        self.last_video_path=path; self.generate_btn.config(state="normal"); self.app.set_dream_busy(False)
        self.status.config(text=f"Saved {dream_count} Flow frames interpolated to {output_count} frames at {fps} FPS: {path}")

    def _failed(self,error):
        self.generate_btn.config(state="normal"); self.app.set_dream_busy(False); self.status.config(text="Dream Cycle generation failed.")
        messagebox.showerror("Dream Cycle error",error[-1800:])


class VideoTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, style="Panel.TFrame")
        self.app = app; self.process = None; self.events = queue.Queue()
        self.model_paths = {}; self.loaded_key = None; self.loaded_model = None
        self.last_video_dir = None; self.last_video_path = None
        self._build(); self.refresh_models()

    def _build(self):
        content = scroll_panel(self)
        ttk.Label(content, text="Autoregressive Flow Video", style="Heading.TLabel").pack(anchor="w", padx=18, pady=(18,4))
        ttk.Label(content, text="Train on consecutive frames, then generate each new frame from the previous one.", style="Body.TLabel").pack(anchor="w", padx=18, pady=(0,12))
        train = ttk.Frame(content, style="Card.TFrame", padding=14); train.pack(fill="x", padx=18, pady=5)
        ttk.Label(train, text="Video Training", style="CardHeading.TLabel").pack(anchor="w", pady=(0,8))
        self.video_data_var = tk.StringVar(); self.video_name_var = tk.StringVar(value="My Flow Video Model")
        self.video_output_var = tk.StringVar(value=str(VIDEO_MODELS_DIR / "My Flow Video Model")); self.video_continue_var = tk.StringVar()
        for label,var,cmd in [("Video file or frame/video folder",self.video_data_var,self.browse_video_data),
                              ("Continue video model (optional)",self.video_continue_var,self.browse_video_continue),
                              ("Save video model to",self.video_output_var,self.browse_video_output)]:
            ttk.Label(train,text=label,style="Meta.TLabel").pack(anchor="w",pady=(5,0))
            row=ttk.Frame(train,style="Card.TFrame"); row.pack(fill="x",pady=(3,0))
            ttk.Entry(row,textvariable=var,style="Field.TEntry").pack(side="left",fill="x",expand=True)
            ttk.Button(row,text="Browse...",command=cmd).pack(side="left",padx=(7,0))
        ttk.Label(train,text="Model name",style="Meta.TLabel").pack(anchor="w",pady=(7,0))
        ttk.Entry(train,textvariable=self.video_name_var,style="Field.TEntry").pack(fill="x",pady=(3,5))
        grid=ttk.Frame(train,style="Card.TFrame"); grid.pack(fill="x")
        self.v_epochs=Field(grid,"Epochs","100"); self.v_res=Field(grid,"Resolution","128",["64","128","384","256","512","640","768"])
        self.v_batch=Field(grid,"Batch size","8",["1","2","4","8","12","16"]); self.v_lr=Field(grid,"Learning rate","0.0002",["0.0001","0.0002","0.0003"])
        self.v_workers=Field(grid,"Data workers","4",["0","2","4","6","8","12"]); self.v_grad=Field(grid,"Gradient accumulation","1",["1","2","4"])
        self.v_stride=Field(grid,"Frame stride","1",["1","2","3","4","6"]); self.v_max_frames=Field(grid,"Max frames per sequence (0 = all)","0")
        self.v_cond_noise=Field(grid,"Condition corruption","0.05",["0.0","0.02","0.05","0.1","0.15"]); self.v_temp_loss=Field(grid,"Temporal loss weight","0.1",["0.0","0.05","0.1","0.2"])
        self.v_min_diff=Field(grid,"Min frame diff","0.0",["0.0","0.002","0.005","0.01","0.02"]); self.v_max_diff=Field(grid,"Max frame diff","1.0",["0.05","0.1","0.15","0.2","0.3","0.5","1.0"])
        self.v_save=Field(grid,"Save every N epochs","10"); self.v_precision=Field(grid,"Mixed precision","fp16",["fp16","no"])
        fs=[self.v_epochs,self.v_res,self.v_batch,self.v_lr,self.v_workers,self.v_grad,self.v_stride,self.v_max_frames,self.v_cond_noise,self.v_temp_loss,self.v_min_diff,self.v_max_diff,self.v_save,self.v_precision]
        for i,f in enumerate(fs): f.grid(row=i//2,column=i%2,sticky="ew",padx=(0 if i%2==0 else 8,8 if i%2==0 else 0),pady=5)
        grid.columnconfigure(0,weight=1); grid.columnconfigure(1,weight=1)
        self.v_flip=tk.BooleanVar(value=True); self.v_tf32=tk.BooleanVar(value=True); self.v_ckpt=tk.BooleanVar(value=False)
        ttk.Checkbutton(train,text="Random horizontal flip",variable=self.v_flip).pack(anchor="w")
        ttk.Checkbutton(train,text="Use TF32 acceleration",variable=self.v_tf32).pack(anchor="w")
        ttk.Checkbutton(train,text="Gradient checkpointing",variable=self.v_ckpt).pack(anchor="w")
        brow=ttk.Frame(train,style="Card.TFrame"); brow.pack(fill="x",pady=(10,0))
        self.v_start=ttk.Button(brow,text="Start Video Training",command=self.start_video_training,style="Accent.TButton"); self.v_start.pack(side="left")
        self.v_stop=ttk.Button(brow,text="Stop && Save",command=self.stop_video_training,state="disabled"); self.v_stop.pack(side="left",padx=8)
        self.v_progress=ttk.Progressbar(train,mode="determinate"); self.v_progress.pack(fill="x",pady=(10,5))
        self.v_status=ttk.Label(train,text="Ready.",style="CardBody.TLabel"); self.v_status.pack(anchor="w")
        self.v_log=tk.Text(train,height=8,bg=FIELD,fg=FG,insertbackground=FG,relief="flat",wrap="word"); self.v_log.pack(fill="x",pady=(8,0))

        gen=ttk.Frame(content,style="Card.TFrame",padding=14); gen.pack(fill="x",padx=18,pady=(10,18))
        ttk.Label(gen,text="Video Generation",style="CardHeading.TLabel").pack(anchor="w",pady=(0,8))
        self.v_model_var=tk.StringVar(); self.v_model_box=ttk.Combobox(gen,textvariable=self.v_model_var,state="readonly"); self.v_model_box.pack(fill="x",pady=3)
        ttk.Button(gen,text="Refresh Video Models",command=self.refresh_models).pack(anchor="w",pady=3)
        self.start_image_var=tk.StringVar(); ttk.Label(gen,text="Starting image",style="Meta.TLabel").pack(anchor="w",pady=(6,0))
        row=ttk.Frame(gen,style="Card.TFrame"); row.pack(fill="x",pady=3)
        ttk.Entry(row,textvariable=self.start_image_var,style="Field.TEntry").pack(side="left",fill="x",expand=True)
        ttk.Button(row,text="Browse...",command=self.browse_start_image).pack(side="left",padx=(7,0))
        self.reference_video_var=tk.StringVar(); ttk.Label(gen,text="Reference video (optional)",style="Meta.TLabel").pack(anchor="w",pady=(6,0))
        row=ttk.Frame(gen,style="Card.TFrame"); row.pack(fill="x",pady=3)
        ttk.Entry(row,textvariable=self.reference_video_var,style="Field.TEntry").pack(side="left",fill="x",expand=True)
        ttk.Button(row,text="Browse...",command=self.browse_reference_video).pack(side="left",padx=(7,0))
        ttk.Button(row,text="Clear",command=lambda:self.reference_video_var.set("")).pack(side="left",padx=(7,0))
        ggrid=ttk.Frame(gen,style="Card.TFrame"); ggrid.pack(fill="x")
        self.v_steps=Field(ggrid,"Flow steps per frame","12",["4","8","12","20","30"]); self.v_frames=Field(ggrid,"Frames to generate","48",["12","24","48","72","120","240","480","960"])
        self.v_fps=Field(ggrid,"Output FPS","12",["6","8","12","15","24","30"]); self.v_method=Field(ggrid,"ODE method","Euler",["Euler","Heun"])
        self.v_seed=Field(ggrid,"Seed",str(random.randrange(2**31))); self.v_feedback=Field(ggrid,"Previous-frame feedback","1.0",["0.85","0.9","0.95","1.0"])
        self.v_reference_strength=Field(ggrid,"Reference guidance strength","0.35",["0.0","0.1","0.2","0.35","0.5","0.65","0.8"])
        self.v_noise_interval=Field(ggrid,"Frames per random-noise transition","6",["2","3","4","6","8","12","24"])
        self.v_noise_curve=Field(ggrid,"Noise transition curve","Cosine",["Cosine","Linear"])
        self.v_noise_refresh=Field(ggrid,"Fresh noise per frame","0.15",["0.0","0.05","0.1","0.15","0.2","0.3","0.5"])
        self.v_interp=Field(ggrid,"Post-export transition frames","0",["0","1","2","3"])
        self.v_max_gen_diff=Field(ggrid,"Max generated frame diff","1.0",["0.02","0.03","0.05","0.08","0.1","0.15","0.2","1.0"])
        generation_fields=[self.v_steps,self.v_frames,self.v_fps,self.v_method,self.v_seed,self.v_feedback,self.v_reference_strength,self.v_noise_interval,self.v_noise_curve,self.v_noise_refresh,self.v_interp,self.v_max_gen_diff]
        for i,f in enumerate(generation_fields): f.grid(row=i//2,column=i%2,sticky="ew",padx=(0 if i%2==0 else 8,8 if i%2==0 else 0),pady=5)
        ggrid.columnconfigure(0,weight=1); ggrid.columnconfigure(1,weight=1)
        seed_row=ttk.Frame(gen,style="Card.TFrame"); seed_row.pack(fill="x",pady=(5,0))
        ttk.Button(seed_row,text="Randomize Seed Now",command=lambda:self.v_seed.var.set(str(random.randrange(2**31)))).pack(side="left")
        self.v_auto_seed_var=tk.BooleanVar(value=True)
        ttk.Checkbutton(seed_row,text="Auto-randomize seed on each video",variable=self.v_auto_seed_var).pack(side="left",padx=(12,0))
        ttk.Label(gen,text="Hybrid noise path keeps noise energy stable. An optional reference video can guide the conditioning frames so the generated dream follows its rough layout and movement without directly replacing the model output. Reference guidance strength controls how closely it follows.",style="CardBody.TLabel",wraplength=720,justify="left").pack(anchor="w",pady=(6,0))
        self.video_gen_output=tk.StringVar(value=str(VIDEO_GENERATIONS_DIR)); ttk.Label(gen,text="Save videos to",style="Meta.TLabel").pack(anchor="w",pady=(6,0))
        row=ttk.Frame(gen,style="Card.TFrame"); row.pack(fill="x",pady=3)
        ttk.Entry(row,textvariable=self.video_gen_output,style="Field.TEntry").pack(side="left",fill="x",expand=True)
        ttk.Button(row,text="Browse...",command=self.browse_video_gen_output).pack(side="left",padx=(7,0))
        self.v_generate=ttk.Button(gen,text="Generate Hybrid Noise Video",command=self.generate_video,style="Accent.TButton"); self.v_generate.pack(anchor="w",pady=(10,4))
        self.v_gen_progress=ttk.Progressbar(gen,mode="determinate"); self.v_gen_progress.pack(fill="x",pady=5)
        self.v_gen_status=ttk.Label(gen,text="Select a trained video model and starting image.",style="CardBody.TLabel"); self.v_gen_status.pack(anchor="w")

    def browse_video_data(self):
        value=filedialog.askopenfilename(title="Select video",filetypes=[("Video files","*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),("All files","*.*")])
        if not value: value=filedialog.askdirectory(title="Select frame or video folder")
        if value: self.video_data_var.set(value)
    def browse_video_continue(self):
        value=filedialog.askdirectory(initialdir=VIDEO_MODELS_DIR)
        if value:
            if video_model_is_valid(value): self.video_continue_var.set(value)
            else: messagebox.showerror("Not a video Flow model","Choose a folder containing flow_video_model_info.json and an unet folder.")
    def browse_video_output(self):
        value=filedialog.askdirectory(initialdir=VIDEO_MODELS_DIR)
        if value: self.video_output_var.set(value)
    def browse_start_image(self):
        value=filedialog.askopenfilename(filetypes=[("Images","*.png *.jpg *.jpeg *.webp *.bmp")])
        if value: self.start_image_var.set(value)
    def browse_reference_video(self):
        value=filedialog.askopenfilename(title="Select optional reference video",filetypes=[("Video files","*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),("All files","*.*")])
        if value: self.reference_video_var.set(value)
    def browse_video_gen_output(self):
        value=filedialog.askdirectory(initialdir=VIDEO_GENERATIONS_DIR)
        if value: self.video_gen_output.set(value)
    def append(self,text): self.v_log.insert("end",text.rstrip()+"\n"); self.v_log.see("end")
    def refresh_models(self):
        models=find_video_models(); self.model_paths={p.name:p for p in models}; self.v_model_box["values"]=list(self.model_paths)
        if models and self.v_model_var.get() not in self.model_paths: self.v_model_var.set(models[0].name)
    def start_video_training(self):
        try:
            source=Path(self.video_data_var.get())
            if not source.exists(): raise ValueError("Select a valid video file or frame/video folder.")
            output=Path(self.video_output_var.get()); output.mkdir(parents=True,exist_ok=True); (output/STOP_FILE_NAME).unlink(missing_ok=True)
            cmd=[sys.executable,str(Path(__file__).resolve()),"--train-video-worker","--data-dir",str(source),"--output-dir",str(output),"--model-name",self.video_name_var.get().strip() or output.name,
                 "--epochs",self.v_epochs.get(),"--resolution",self.v_res.get(),"--batch-size",self.v_batch.get(),"--learning-rate",self.v_lr.get(),"--workers",self.v_workers.get(),
                 "--gradient-accumulation",self.v_grad.get(),"--frame-stride",self.v_stride.get(),"--max-frames",self.v_max_frames.get(),"--condition-noise",self.v_cond_noise.get(),
                 "--temporal-loss-weight",self.v_temp_loss.get(),"--min-frame-diff",self.v_min_diff.get(),"--max-frame-diff",self.v_max_diff.get(),"--save-every",self.v_save.get(),"--mixed-precision",self.v_precision.get()]
            if self.video_continue_var.get(): cmd += ["--continue-model",self.video_continue_var.get()]
            if self.v_flip.get(): cmd.append("--random-flip")
            if self.v_tf32.get(): cmd.append("--tf32")
            if self.v_ckpt.get(): cmd.append("--gradient-checkpointing")
            flags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
            self.process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1,creationflags=flags)
            self.v_start.config(state="disabled"); self.v_stop.config(state="normal"); self.app.set_video_busy(True); self.v_status.config(text="Preparing frames and starting training...")
            threading.Thread(target=self._read_output,daemon=True).start(); self.after(100,self._poll)
        except Exception as exc: messagebox.showerror("Cannot start video training",str(exc))
    def _read_output(self):
        for line in self.process.stdout: self.events.put(line.rstrip())
        self.events.put(("exit",self.process.wait()))
    def _poll(self):
        try:
            while True:
                item=self.events.get_nowait()
                if isinstance(item,tuple): self._finished(item[1]); return
                if item.startswith("FLOW_EVENT:"):
                    info=json.loads(item.split(":",1)[1]); kind=info.get("type")
                    if kind=="progress":
                        self.v_progress["maximum"]=info["total_updates"]; self.v_progress["value"]=info["update"]
                        self.v_status.config(text=f"Epoch {info['epoch']}/{info['epochs']} • loss {info['loss']:.4f} • ETA {human_time(info['eta'])}")
                    elif kind=="start":
                        self.append(
                            f"Training on {info['pairs']} consecutive frame pairs using {info['device']}. "
                            f"Kept {info.get('kept_pairs', info['pairs'])}/{info.get('candidate_pairs', info['pairs'])} pairs "
                            f"(skipped similar: {info.get('skipped_too_similar', 0)}, skipped large jumps: {info.get('skipped_too_different', 0)})."
                        )
                    elif kind=="saved": self.append(f"Saved video model at epoch {info['epoch']}.")
                    elif kind=="complete": self.v_status.config(text="Video training complete!")
                    elif kind=="stopped": self.v_status.config(text="Stopped safely; current video model was saved.")
                    elif kind=="warning": self.append("WARNING: "+info.get("message",""))
                else: self.append(item)
        except queue.Empty: pass
        if self.process: self.after(100,self._poll)
    def stop_video_training(self):
        if self.process: Path(self.video_output_var.get(),STOP_FILE_NAME).touch(); self.v_stop.config(state="disabled"); self.v_status.config(text="Stopping after the current batch...")
    def _finished(self,code):
        self.process=None; self.v_start.config(state="normal"); self.v_stop.config(state="disabled"); self.app.set_video_busy(False); self.refresh_models()
        if hasattr(self.app,"dream_cycle_tab"): self.app.dream_cycle_tab.refresh_models()
        if code!=0: self.v_status.config(text=f"Video training failed with code {code}."); messagebox.showerror("Video training error","See the Video tab log for details.")
    def generate_video(self):
        try:
            key=self.v_model_var.get(); start=Path(self.start_image_var.get())
            if key not in self.model_paths: raise ValueError("Select a trained video model.")
            if not start.is_file(): raise ValueError("Select a starting image.")
            if self.v_auto_seed_var.get():
                self.v_seed.var.set(str(random.randrange(2**31)))
            steps=int(self.v_steps.get()); frames=int(self.v_frames.get()); fps=int(self.v_fps.get()); seed=int(self.v_seed.get()); feedback=float(self.v_feedback.get())
            reference_path=Path(self.reference_video_var.get()) if self.reference_video_var.get().strip() else None
            reference_strength=float(self.v_reference_strength.get())
            if reference_path is not None and not reference_path.is_file(): raise ValueError("Select a valid reference video, or clear the optional field.")
            if not 0.0 <= reference_strength <= 1.0: raise ValueError("Reference guidance strength must be between 0.0 and 1.0.")
            noise_interval=int(self.v_noise_interval.get()); noise_curve=self.v_noise_curve.get(); noise_refresh=float(self.v_noise_refresh.get())
            transition_frames=int(self.v_interp.get()); max_generated_frame_diff=float(self.v_max_gen_diff.get())
            if min(steps,frames,fps,noise_interval)<=0: raise ValueError("Steps, frames, FPS, and noise transition length must be positive.")
            if not 0.0 <= noise_refresh <= 1.0: raise ValueError("Fresh noise per frame must be between 0.0 and 1.0.")
            if transition_frames < 0: raise ValueError("Transition frames must be zero or greater.")
            if max_generated_frame_diff <= 0 and max_generated_frame_diff != 1.0: raise ValueError("Max generated frame diff must be positive, or 1.0 to disable it.")
            self.v_generate.config(state="disabled"); self.v_gen_progress["maximum"]=frames; self.v_gen_progress["value"]=0; self.app.set_video_busy(True)
            threading.Thread(target=self._generate_worker,args=(key,start,steps,frames,fps,seed,self.v_method.get(),feedback,reference_path,reference_strength,noise_interval,noise_curve,noise_refresh,transition_frames,max_generated_frame_diff),daemon=True).start()
        except Exception as exc: messagebox.showerror("Cannot generate video",str(exc))
    def _generate_worker(self,key,start,steps,frames,fps,seed,method,feedback,reference_path,reference_strength,noise_interval,noise_curve,noise_refresh,transition_frames,max_generated_frame_diff):
        try:
            import torch, cv2, numpy as np
            from PIL import Image
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); dtype=torch.float16 if device.type=="cuda" else torch.float32
            if self.loaded_key!=key or self.loaded_model is None:
                self.loaded_model=None
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                self.loaded_model=load_video_unet(self.model_paths[key],device=device,dtype=dtype); self.loaded_key=key
            res=int(self.loaded_model.config.sample_size)
            image=Image.open(start).convert("RGB"); image.thumbnail((res,res),Image.Resampling.LANCZOS)
            canvas=Image.new("RGB",(res,res)); canvas.paste(image,((res-image.width)//2,(res-image.height)//2))
            previous=pil_to_normalized_tensor(canvas, device=device, dtype=dtype)
            reference_tensors=[]
            if reference_path is not None and reference_strength > 0.0:
                reference_frames=load_reference_video_frames(reference_path, frames, res)
                reference_tensors=[pil_to_normalized_tensor(frame,device=device,dtype=dtype) for frame in reference_frames]
            generator=torch.Generator(device=device).manual_seed(seed)
            noise_anchor_a=torch.randn(previous.shape,generator=generator,device=device,dtype=dtype)
            noise_anchor_b=torch.randn(previous.shape,generator=generator,device=device,dtype=dtype)
            out_root=Path(self.video_gen_output.get()); safe="".join(c if c not in '<>:"/\\|?*' else '_' for c in key).strip() or "Flow Video"
            stamp=time.strftime("%Y%m%d_%H%M%S"); out_dir=out_root/safe/f"{stamp}_seed{seed}"; out_dir.mkdir(parents=True,exist_ok=True)
            pil_frames=[canvas]; canvas.save(out_dir/"frame_000000.png")
            generation_stats=[]
            self.after(0,lambda:self.v_gen_progress.configure(value=1))
            self.after(0,lambda:self.v_gen_status.config(text=f"Generated frame 1/{frames}..."))
            for i in range(1,frames):
                segment_position=((i - 1) % noise_interval + 1) / float(noise_interval)
                frame_noise, noise_alpha = hybrid_transition_noise(
                    noise_anchor_a, noise_anchor_b, segment_position, noise_curve, noise_refresh, generator
                )
                conditioning_previous=previous
                if reference_tensors:
                    reference_condition=reference_tensors[min(i - 1, len(reference_tensors) - 1)]
                    conditioning_previous=((1.0-reference_strength)*previous+reference_strength*reference_condition).clamp(-1,1)
                generated=sample_next_video_frame(self.loaded_model,conditioning_previous,steps,device,dtype,generator,method,initial_noise=frame_noise)
                if feedback<1.0:
                    generated=(feedback*generated+(1-feedback)*previous).clamp(-1,1)
                generated, generated_diff, diff_scale = constrain_tensor_frame_difference(previous, generated, max_generated_frame_diff)
                previous=generated.detach().clone()
                pil=tensor_to_pil(generated[0]); pil.save(out_dir/f"frame_{i:06d}.png"); pil_frames.append(pil)
                generation_stats.append({
                    "frame_index": int(i),
                    "frame_difference": round(generated_diff, 6),
                    "difference_scale_applied": round(diff_scale, 6),
                    "feedback": round(float(feedback), 6),
                    "reference_guidance_strength": round(float(reference_strength if reference_tensors else 0.0), 6),
                    "noise_transition_alpha": round(float(noise_alpha), 6),
                    "noise_segment_frame": int(((i - 1) % noise_interval) + 1),
                })
                if i % noise_interval == 0:
                    noise_anchor_a = noise_anchor_b
                    noise_anchor_b = torch.randn(previous.shape,generator=generator,device=device,dtype=dtype)
                self.after(0,lambda n=i+1:self.v_gen_progress.configure(value=n))
                self.after(0,lambda n=i+1:self.v_gen_status.config(text=f"Generated frame {n}/{frames}..."))
            export_frames = interpolate_pil_frames(pil_frames, transition_frames)
            video_path=out_dir/f"{safe}_{stamp}_seed{seed}.mp4"
            writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(res,res))
            if not writer.isOpened(): raise RuntimeError("OpenCV could not create the MP4 file.")
            for pil in export_frames:
                writer.write(cv2.cvtColor(np.array(pil.convert("RGB")),cv2.COLOR_RGB2BGR))
            writer.release()
            average_diff = sum(item["frame_difference"] for item in generation_stats) / max(1, len(generation_stats))
            info = {
                "generation_mode": "hybrid_slerp_noise_autoregressive",
                "raw_model_frames": len(pil_frames),
                "transition_frames_between_outputs": int(transition_frames),
                "interpolation_mode": "optical_flow_farneback" if int(transition_frames) > 0 else "none",
                "previous_frame_feedback": float(feedback),
                "reference_video": str(reference_path) if reference_path is not None else None,
                "reference_guidance_strength": float(reference_strength if reference_tensors else 0.0),
                "reference_guidance_mode": "conditioning_frame_blend" if reference_tensors else "none",
                "noise_transition_frames": int(noise_interval),
                "noise_transition_curve": str(noise_curve),
                "fresh_noise_per_frame": float(noise_refresh),
                "noise_interpolation": "spherical_variance_preserving",
                "max_generated_frame_diff": float(max_generated_frame_diff),
                "exported_video_frames": len(export_frames),
                "fps": int(fps),
                "output_seconds": round(len(export_frames) / max(1, fps), 3),
                "average_frame_difference": round(average_diff, 6),
                "generation_frame_stats": generation_stats,
            }
            (out_dir / "video_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            self.last_video_dir=out_dir; self.last_video_path=video_path
            self.after(0,lambda:self._video_done(video_path,out_dir, len(pil_frames), len(export_frames), transition_frames, max_generated_frame_diff))
        except Exception:
            err=traceback.format_exc(); self.after(0,lambda:self._video_failed(err))
    def _video_done(self,path,out_dir, raw_frames=None, export_frames=None, transition_frames=0, max_generated_frame_diff=1.0):
        if raw_frames is not None and export_frames is not None:
            self.v_gen_status.config(text=f"Saved MP4 and PNG frames to {out_dir} • raw frames: {raw_frames}, exported video frames: {export_frames} (transition frames: {transition_frames}, max generated diff: {max_generated_frame_diff})")
        else:
            self.v_gen_status.config(text=f"Saved MP4 and PNG frames to {out_dir}")
        self.v_generate.config(state="normal"); self.app.set_video_busy(False)
    def _video_failed(self,error):
        self.v_gen_status.config(text="Video generation failed."); self.v_generate.config(state="normal"); self.app.set_video_busy(False); messagebox.showerror("Video generation error",error[-1800:])


class App:
    def __init__(self, root):
        self.root = root; self.training_busy = False; self.generation_busy = False; self.video_busy = False; self.dream_busy = False
        root.title("Flow Matching Image Lab"); root.geometry("980x820"); root.minsize(820, 650); root.configure(bg=BG)
        self.style()
        notebook = ttk.Notebook(root); notebook.pack(fill="both", expand=True)
        self.train_tab = TrainTab(notebook, self); self.generate_tab = GenerateTab(notebook, self)
        self.video_tab = VideoTab(notebook, self); self.dream_cycle_tab = DreamCycleTab(notebook, self)
        notebook.add(self.train_tab, text="  Training  "); notebook.add(self.generate_tab, text="  Generation  ")
        notebook.add(self.video_tab, text="  Video  "); notebook.add(self.dream_cycle_tab, text="  Dream Cycle  ")
        root.after(400, self.cuda_warning)
    def style(self):
        s = ttk.Style();
        try: s.theme_use("clam")
        except tk.TclError: pass
        s.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))
        s.configure("Panel.TFrame", background=BG); s.configure("Card.TFrame", background=PANEL)
        s.configure("Heading.TLabel", background=BG, foreground=FG, font=("Segoe UI", 16, "bold"))
        s.configure("Body.TLabel", background=BG, foreground=DIM)
        s.configure("Meta.TLabel", background=PANEL, foreground=DIM, font=("Segoe UI", 9))
        s.configure("CardHeading.TLabel", background=PANEL, foreground=FG, font=("Segoe UI", 13, "bold"))
        s.configure("CardBody.TLabel", background=PANEL, foreground=DIM)
        s.configure("Field.TEntry", fieldbackground=FIELD, foreground=FG, insertcolor=FG)
        s.configure("Accent.TButton", background=ACCENT, foreground="#10131a", font=("Segoe UI", 10, "bold"), padding=8)
        s.map("Accent.TButton", background=[("active", "#88aaff")])
        s.configure("TNotebook", background=BG); s.configure("TNotebook.Tab", background=PANEL, foreground=DIM, padding=(16, 9))
        s.map("TNotebook.Tab", background=[("selected", BG)], foreground=[("selected", FG)])
        s.configure("TCombobox", fieldbackground=FIELD, foreground=FG)
        s.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=FIELD)
    def set_busy(self, busy):
        self.training_busy = busy
        self.generate_tab.generate_btn.config(state="disabled" if busy or self.generation_busy or self.video_busy or self.dream_busy else "normal")
        if hasattr(self, "video_tab"):
            self.video_tab.v_start.config(state="disabled" if busy or self.video_busy else "normal")
            self.video_tab.v_generate.config(state="disabled" if busy or self.video_busy else "normal")
        if hasattr(self, "dream_cycle_tab"):
            self.dream_cycle_tab.generate_btn.config(state="disabled" if busy or self.dream_busy else "normal")
    def set_generation_busy(self, busy):
        self.generation_busy = busy
        self.train_tab.start_btn.config(state="disabled" if busy or self.training_busy or self.video_busy or self.dream_busy else "normal")
        if hasattr(self, "video_tab"):
            self.video_tab.v_start.config(state="disabled" if busy or self.video_busy else "normal")
            self.video_tab.v_generate.config(state="disabled" if busy or self.video_busy else "normal")
        if hasattr(self, "dream_cycle_tab"):
            self.dream_cycle_tab.generate_btn.config(state="disabled" if busy or self.dream_busy else "normal")
    def set_video_busy(self, busy):
        self.video_busy = busy
        self.train_tab.start_btn.config(state="disabled" if busy or self.training_busy or self.generation_busy or self.dream_busy else "normal")
        self.generate_tab.generate_btn.config(state="disabled" if busy or self.training_busy or self.generation_busy or self.dream_busy else "normal")
        if hasattr(self, "video_tab"):
            self.video_tab.v_start.config(state="disabled" if busy or self.training_busy or self.generation_busy else "normal")
            self.video_tab.v_generate.config(state="disabled" if busy or self.training_busy or self.generation_busy else "normal")

    def set_dream_busy(self, busy):
        self.dream_busy = busy
        blocked = busy or self.training_busy or self.generation_busy or self.video_busy
        self.train_tab.start_btn.config(state="disabled" if blocked else "normal")
        self.generate_tab.generate_btn.config(state="disabled" if blocked else "normal")
        self.video_tab.v_start.config(state="disabled" if blocked else "normal")
        self.video_tab.v_generate.config(state="disabled" if blocked else "normal")
        self.dream_cycle_tab.generate_btn.config(state="disabled" if blocked else "normal")

    def cuda_warning(self):
        try:
            import torch
            if not torch.cuda.is_available():
                messagebox.showwarning("CUDA GPU unavailable", "PyTorch cannot see a CUDA GPU. The app can run on CPU, but training will be extremely slow. Check that your CUDA-enabled PyTorch installation is active.")
        except Exception as exc:
            messagebox.showerror("Missing dependency", f"PyTorch could not be imported:\n\n{exc}\n\nInstall: pip install torch torchvision diffusers pillow safetensors")


def parse_worker_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-worker", action="store_true")
    p.add_argument("--train-video-worker", action="store_true")
    p.add_argument("--data-dir"); p.add_argument("--output-dir"); p.add_argument("--model-name")
    p.add_argument("--continue-model", default=""); p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--resolution", type=int, default=256); p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=1); p.add_argument("--mixed-precision", choices=["fp16", "no"], default="fp16")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--preview-every", type=int, default=10)
    p.add_argument("--preview-steps", type=int, default=10)
    p.add_argument("--random-flip", action="store_true"); p.add_argument("--tf32", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--condition-noise", type=float, default=0.05)
    p.add_argument("--temporal-loss-weight", type=float, default=0.1)
    p.add_argument("--min-frame-diff", type=float, default=0.0)
    p.add_argument("--max-frame-diff", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    worker_args = parse_worker_args()
    if worker_args.train_worker:
        try: training_worker(worker_args)
        except Exception:
            traceback.print_exc(); sys.exit(1)
    elif worker_args.train_video_worker:
        try: video_training_worker(worker_args)
        except Exception:
            traceback.print_exc(); sys.exit(1)
    else:
        MODELS_DIR.mkdir(parents=True, exist_ok=True); GENERATIONS_DIR.mkdir(parents=True, exist_ok=True); VIDEO_MODELS_DIR.mkdir(parents=True, exist_ok=True); VIDEO_GENERATIONS_DIR.mkdir(parents=True, exist_ok=True); DREAM_CYCLE_DIR.mkdir(parents=True, exist_ok=True)
        root = tk.Tk(); App(root); root.mainloop()
