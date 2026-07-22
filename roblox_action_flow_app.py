from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
import math
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

APP_DIR = Path(__file__).resolve().parent
SOURCE_APP_NAMES = ["flow_matching_app(13).py", "flow_matching_app.py"]
ACTION_MODELS_DIR = APP_DIR / "output_action_flow_models"
ACTION_GENERATIONS_DIR = APP_DIR / "action_flow_captures"
STOP_FILE_NAME = "stop_action_training.flag"
TRAINING_SETTINGS_FILE = APP_DIR / "roblox_action_flow_training_settings.json"

BG = "#15171d"
PANEL = "#20232b"
FIELD = "#2b2f39"
FG = "#eef1f7"
DIM = "#aab0be"
ACCENT = "#6f98ff"
ACTION_NAMES = ["w", "a", "s", "d", "jump", "mouse_dx", "mouse_dy", "zoom"]
ACTION_DIM = len(ACTION_NAMES)
BINARY_ACTION_NAMES = ACTION_NAMES[:5]
ACTION_DISPLAY_NAMES = {
    "w": "Up", "a": "Left", "s": "Down", "d": "Right", "jump": "Jump",
    "mouse_dx": "Camera Yaw", "mouse_dy": "Camera Pitch", "zoom": "Zoom",
}
KEY_TO_ACTION = {
    "w": "w", "up": "w",
    "a": "a", "left": "a",
    "s": "s", "down": "s",
    "d": "d", "right": "d",
    "space": "space",
}
PLAYER_ACTION_GUIDANCE = 2.0
PLAYER_MOUSE_SCALE = 120.0
PLAYER_ZOOM_SCALE = 3.0
DEFAULT_MOTION_NOISE_REFRESH = 0.06
DEFAULT_ACTIVE_NOISE_MULTIPLIER = 1.0


def parse_frame_size(value):
    """Return (height, width) for a true 16:9 model/training size."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
    else:
        text = str(value).lower().replace("×", "x").strip()
        if "x" in text:
            width, height = (int(part.strip()) for part in text.split("x", 1))
        else:
            # Legacy UI values represented a square side. Map the supported values
            # to their new 16:9 equivalents instead of quietly producing a square.
            width = int(text)
            height = round(width * 9 / 16)
    if width <= 0 or height <= 0:
        raise ValueError("Frame width and height must be greater than zero.")
    if width * 9 != height * 16:
        raise ValueError(f"Frame size {width}x{height} is not exactly 16:9.")
    if width % 16 or height % 16:
        raise ValueError("16:9 width and height must both be divisible by 16 (for example 256x144 or 512x288).")
    return height, width


def set_model_frame_size(model, resolution):
    """Retarget spatially reusable convolutional weights to a 16:9 canvas."""
    height, width = parse_frame_size(resolution)
    model.register_to_config(sample_size=(height, width))
    return model


def model_frame_size(model_or_config):
    config = getattr(model_or_config, "config", model_or_config)
    sample_size = getattr(config, "sample_size", None)
    if sample_size is None and isinstance(config, dict):
        sample_size = config.get("sample_size")
    if isinstance(sample_size, (list, tuple)):
        return int(sample_size[0]), int(sample_size[1])
    side = int(sample_size)
    return side, side


def event(**values):
    print("ACTION_FLOW_EVENT:" + json.dumps(values), flush=True)


def write_json_atomic(path, values):
    """Replace a JSON report only after the complete new file is on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def publish_training_metrics(output_dir, snapshot, append_history=False):
    """Publish the latest machine-readable state and optionally preserve a sample."""
    output_dir = Path(output_dir)
    snapshot = dict(snapshot)
    snapshot["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json_atomic(output_dir / "metrics_latest.json", snapshot)
    if append_history:
        history_path = output_dir / "metrics_history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
            handle.flush()


def system_telemetry(device=None):
    """Best-effort host/GPU snapshot; unavailable sensors stay null."""
    result = {
        "cpu_percent": None, "ram_used_gb": None, "ram_percent": None,
        "gpu_clock_mhz": None, "gpu_power_watts": None,
    }
    try:
        import psutil
        memory = psutil.virtual_memory()
        result.update(
            cpu_percent=float(psutil.cpu_percent(interval=None)),
            ram_used_gb=float(memory.used / (1024 ** 3)),
            ram_percent=float(memory.percent),
        )
    except (ImportError, OSError):
        pass
    if device is not None and getattr(device, "type", None) == "cuda":
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.gr,power.draw", "--format=csv,noheader,nounits", "-i", str(device.index or 0)],
                capture_output=True, text=True, timeout=2, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            clock, power = completed.stdout.strip().splitlines()[0].split(",")[:2]
            result["gpu_clock_mhz"] = float(clock.strip())
            result["gpu_power_watts"] = float(power.strip())
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
    return result


def action_is_idle(action):
    return all(
        abs(float(value)) <= (0.5 if index < 5 else 0.02)
        for index, value in enumerate(action)
    )


def validation_split_indexes(pairs, validation_count, frame_gap):
    """Build a reproducible holdout containing both idle and active scenes when possible."""
    total = len(pairs)
    validation_count = min(max(1, int(validation_count)), total - 1)
    val = list(range(total - validation_count, total))
    is_idle = lambda i: action_is_idle(pairs[i][2])
    desired = []
    if any(is_idle(i) for i in range(total)) and not any(is_idle(i) for i in val):
        desired.append(max(i for i in range(total - validation_count) if is_idle(i)))
    if any(not is_idle(i) for i in range(total)) and not any(not is_idle(i) for i in val):
        desired.append(max(i for i in range(total - validation_count) if not is_idle(i)))
    for index in desired:
        replace = next((i for i in val if is_idle(i) != is_idle(index)), val[0])
        val.remove(replace)
        val.append(index)
    val = sorted(set(val))
    # Validation may be capped to its first few batches; put representatives first.
    idle_first = next((i for i in val if is_idle(i)), None)
    action_first = next((i for i in val if not is_idle(i)), None)
    representatives = [i for i in (idle_first, action_first) if i is not None]
    val = representatives + [i for i in val if i not in representatives]
    guard = max(1, int(frame_gap))
    blocked = {j for i in val for j in range(max(0, i - guard), min(total, i + guard + 1))}
    train = [i for i in range(total) if i not in blocked]
    if not train:
        train = [next(i for i in range(total) if i not in val)]
    return train, val


def human_time(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "calculating..."
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def analyze_reference_motion(video_path, max_samples=180):
    """Turn a gameplay clip into a conservative temporal-noise target.

    This estimates visible movement only; it cannot infer game physics or teach a
    missing control mapping. Median statistics keep a cutscene or explosion from
    selecting an unsafe noise level.
    """
    import cv2
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Could not open that video file.")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, math.ceil(frame_count / max(1, int(max_samples)))) if frame_count else 1
    previous = None
    differences = []
    flows = []
    index = 0
    while len(differences) < max_samples:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride:
            index += 1
            continue
        index += 1
        gray = cv2.cvtColor(cv2.resize(frame, (128, 72)), cv2.COLOR_BGR2GRAY)
        if previous is not None:
            differences.append(float(cv2.absdiff(gray, previous).mean()) / 255.0)
            flow = cv2.calcOpticalFlowFarneback(
                previous, gray, None, 0.5, 2, 15, 2, 5, 1.2, 0,
            )
            magnitude, _angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flows.append(float(magnitude.mean()))
        previous = gray
    capture.release()
    if len(differences) < 5:
        raise ValueError("The video needs at least six readable frames.")

    def percentile(values, fraction):
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))]

    median_difference = percentile(differences, 0.50)
    high_difference = percentile(differences, 0.75)
    median_flow = percentile(flows, 0.50)
    # The cap is deliberately modest: refresh introduces visual variation but too
    # much fresh noise overwhelms the action-conditioned previous-frame signal.
    refresh = max(0.02, min(0.12, 0.015 + high_difference * 0.55 + median_flow * 0.004))
    return {
        "sampled_transitions": len(differences),
        "source_fps": fps,
        "median_difference": median_difference,
        "high_difference": high_difference,
        "median_flow": median_flow,
        "motion_noise_refresh": refresh,
    }


def find_source_app():
    for name in SOURCE_APP_NAMES:
        candidate = APP_DIR / name
        if candidate.is_file():
            return candidate
    matches = sorted(APP_DIR.glob("flow_matching_app*.py"), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    raise FileNotFoundError("Place this app beside flow_matching_app(13).py or another flow_matching_app*.py file.")


def load_source_module():
    source = find_source_app()
    spec = importlib.util.spec_from_file_location("action_flow_source", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def base_video_model_is_valid(path):
    path = Path(path)
    try:
        info = json.loads((path / "flow_video_model_info.json").read_text(encoding="utf-8"))
        return info.get("model_type") == "autoregressive_rectified_flow_video" and (path / "unet" / "config.json").is_file()
    except Exception:
        return False


def action_model_is_valid(path):
    path = Path(path)
    try:
        info = json.loads((path / "action_flow_model_info.json").read_text(encoding="utf-8"))
        return info.get("model_type") == "action_conditioned_rectified_flow_video" and (path / "unet" / "config.json").is_file()
    except Exception:
        return False


def expand_unet_for_actions(model, action_dim=ACTION_DIM):
    """Expand a trained 6-channel video U-Net to [flow RGB, previous RGB, action maps]."""
    import torch
    import torch.nn as nn

    old = model.conv_in
    expected = 6 + int(action_dim)
    if old.in_channels == expected:
        model.register_to_config(in_channels=expected)
        return model
    if old.in_channels != 6:
        raise ValueError(f"Expected a 6-channel base video U-Net, but found {old.in_channels} channels.")

    new = nn.Conv2d(
        expected,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    ).to(device=old.weight.device, dtype=old.weight.dtype)

    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :6].copy_(old.weight)
        # Small non-zero initialization lets action channels begin learning immediately
        # without disrupting the already-trained visual pathway.
        nn.init.normal_(new.weight[:, 6:], mean=0.0, std=0.002)
        if old.bias is not None:
            new.bias.copy_(old.bias)

    model.conv_in = new
    model.register_to_config(in_channels=expected)
    return model


def build_action_unet_from_scratch(resolution):
    """Create a new action-conditioned video U-Net without pretrained visual weights."""
    from diffusers import UNet2DModel

    height, width = parse_frame_size(resolution)

    # This remains small enough for a 12 GB GPU while still learning RGB appearance,
    # previous-frame conditioning, and the eight action channels from the dataset.
    return UNet2DModel(
        sample_size=(height, width),
        in_channels=6 + ACTION_DIM,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 192, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        norm_num_groups=32,
        act_fn="silu",
        attention_head_dim=8,
    )


def load_action_unet(model_dir, device=None, dtype=None):
    from diffusers import UNet2DModel
    kwargs = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = UNet2DModel.from_pretrained(str(model_dir), subfolder="unet", **kwargs)
    channels = int(model.config.in_channels)
    if channels == 11 and ACTION_DIM == 8:
        # Upgrade an older five-action checkpoint while preserving its learned weights.
        import torch
        import torch.nn as nn
        old = model.conv_in
        expanded = nn.Conv2d(14, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None).to(old.weight)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :11].copy_(old.weight)
            nn.init.normal_(expanded.weight[:, 11:], mean=0.0, std=0.002)
            if old.bias is not None:
                expanded.bias.copy_(old.bias)
        model.conv_in = expanded
        model.register_to_config(in_channels=14)
    elif channels != 6 + ACTION_DIM:
        raise ValueError("The selected checkpoint is not a compatible action-conditioned video model.")
    if device is not None:
        model.to(device)
    return model


def aggregate_action_window(window_rows, method="window"):
    """Aggregate controls between the previous frame and a farther target frame.

    Binary controls use max/any in the default window mode so a short press is not
    lost. Continuous mouse/zoom controls use their mean to avoid scaling them simply
    because a larger frame gap was selected.
    """
    if not window_rows:
        return [0.0] * ACTION_DIM

    method = str(method).strip().lower()
    vectors = [[float(row.get(name, 0.0)) for name in ACTION_NAMES] for row in window_rows]

    if method == "last":
        return vectors[-1]

    if method == "mean":
        return [sum(vector[i] for vector in vectors) / len(vectors) for i in range(ACTION_DIM)]

    # Default: preserve short presses, but never create impossible opposing controls
    # merely because the player changed direction inside the temporal window.
    binary = [max(vector[i] for vector in vectors) for i in range(5)]
    for first, second in ((0, 2), (1, 3)):
        first_total = sum(vector[first] for vector in vectors)
        second_total = sum(vector[second] for vector in vectors)
        if first_total > second_total:
            binary[first], binary[second] = 1.0, 0.0
        elif second_total > first_total:
            binary[first], binary[second] = 0.0, 1.0
        elif first_total > 0:
            # On an exact tie, the most recent direction best describes the target.
            for vector in reversed(vectors):
                if vector[first] != vector[second]:
                    binary[first] = 1.0 if vector[first] > vector[second] else 0.0
                    binary[second] = 1.0 - binary[first]
                    break
    degree_encoded = any(
        str(row.get("camera_encoding", "")).lower() == "relative_degrees_v1"
        or "camera_yaw_delta_degrees" in row
        for row in window_rows
    )
    if degree_encoded:
        # Relative camera rotations compose across a temporal gap. Summing preserves
        # the total yaw/pitch that caused the target view; averaging would make a
        # three-frame turn appear three times weaker than it really was.
        analog = [
            max(-1.0, min(1.0, sum(vector[i] for vector in vectors)))
            for i in range(5, ACTION_DIM)
        ]
    else:
        # Legacy pixel-normalized datasets retain their original aggregation.
        analog = [sum(vector[i] for vector in vectors) / len(vectors) for i in range(5, ACTION_DIM)]
    return binary + analog


def recommended_frame_gap(capture_fps, target_seconds=0.25):
    """Choose a horizon with enough visible motion while retaining responsive controls."""
    try:
        fps = float(capture_fps)
    except (TypeError, ValueError):
        return None
    if fps <= 0:
        return None
    return max(1, min(12, round(fps * float(target_seconds))))


def observed_action_profiles(pairs):
    """Describe only controls and binary combinations genuinely present in a dataset."""
    counts = {
        name: sum(abs(float(action[i])) > (0.5 if i < 5 else 0.02)
                  for _previous, _target, action in pairs)
        for i, name in enumerate(ACTION_NAMES)
    }
    minimum_examples = max(2, math.ceil(len(pairs) * 0.005))
    enabled = [name for name in ACTION_NAMES if counts[name] >= minimum_examples]
    enabled_binary_indexes = {
        index for index, name in enumerate(BINARY_ACTION_NAMES) if name in enabled
    }
    binary_vectors = sorted({
        tuple(1.0 if i in enabled_binary_indexes and float(action[i]) > 0.5 else 0.0
              for i in range(5))
        for _previous, _target, action in pairs
    })
    if (0.0,) * 5 not in binary_vectors:
        binary_vectors.insert(0, (0.0,) * 5)
    enabled_indexes = {index for index, name in enumerate(ACTION_NAMES) if name in enabled}
    full_profiles = sorted({
        tuple(
            (
                (1.0 if float(action[i]) > 0.5 else 0.0) if i < 5
                else (0.5 if float(action[i]) > 0.02 else -0.5 if float(action[i]) < -0.02 else 0.0)
            ) if i in enabled_indexes else 0.0
            for i in range(ACTION_DIM)
        )
        for _previous, _target, action in pairs
    })
    return (
        counts,
        enabled,
        [list(vector) for vector in binary_vectors],
        [list(vector) for vector in full_profiles],
    )


def counterfactual_actions(actions, observed_action_prototypes):
    """Return a guaranteed-different observed control profile for every row."""
    import torch
    prototypes = torch.as_tensor(
        observed_action_prototypes, device=actions.device, dtype=actions.dtype,
    )
    if prototypes.ndim != 2 or prototypes.shape[1] != ACTION_DIM:
        raise ValueError(f"Observed action profiles must have {ACTION_DIM} values each.")
    if prototypes.shape[0] < 2:
        raise ValueError("Counterfactual training needs at least two distinct observed action profiles.")
    current = torch.cat([
        (actions[:, :5] > 0.5).to(dtype=actions.dtype),
        torch.where(
            actions[:, 5:] > 0.02,
            torch.full_like(actions[:, 5:], 0.5),
            torch.where(actions[:, 5:] < -0.02, torch.full_like(actions[:, 5:], -0.5), torch.zeros_like(actions[:, 5:])),
        ),
    ], dim=1)
    different = torch.any(prototypes[None, :, :] != current[:, None, :], dim=2)
    choices = []
    for row in range(actions.shape[0]):
        candidates = torch.nonzero(different[row], as_tuple=False).flatten()
        selected = candidates[torch.randint(candidates.numel(), (1,), device=actions.device)]
        choices.append(prototypes[selected].squeeze(0))
    return torch.stack(choices, dim=0)


def inspect_action_dataset(dataset_dir, frame_gap=1, action_aggregation="window"):
    """Profile recorder data without changing it, so bad captures are visible before training."""
    dataset_dir = Path(dataset_dir)
    frame_gap = max(1, int(frame_gap))
    metadata = dataset_dir / "actions.jsonl"
    frames_dir = dataset_dir / "frames"
    if not metadata.is_file():
        raise ValueError("The dataset folder does not contain actions.jsonl.")
    if not frames_dir.is_dir():
        raise ValueError("The dataset folder does not contain a frames folder.")

    rows = []
    dataset_info = {}
    info_path = dataset_dir / "dataset_info.json"
    if info_path.is_file():
        try:
            dataset_info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            dataset_info = {}
    capture_fps = dataset_info.get("capture_fps")
    report = {
        "metadata_rows": 0, "valid_rows": 0, "missing_images": 0,
        "invalid_json": 0, "duplicate_indices": 0, "sessions": 0,
        "legacy_segments": 0, "valid_transitions": 0,
        "action_counts": {name: 0 for name in ACTION_NAMES},
        "idle_rows": 0, "estimated_minutes": 0.0,
        "capture_fps": capture_fps,
        "recommended_frame_gap": recommended_frame_gap(capture_fps),
        "enabled_action_names": [],
        "observed_binary_actions": [],
        "observed_action_prototypes": [],
        "camera_encoding": dataset_info.get("camera_encoding", "legacy_pixels"),
        "camera_input_source": dataset_info.get("camera_input_source", "unknown"),
        "camera_degree_rows": 0,
        "camera_active_rows": 0,
        "right_mouse_rows": 0,
        "absolute_yaw_degrees": 0.0,
        "absolute_pitch_degrees": 0.0,
        "yaw_counts_per_360_degrees": dataset_info.get("yaw_counts_per_360_degrees"),
        "pitch_counts_per_180_degrees": dataset_info.get("pitch_counts_per_180_degrees"),
        "max_yaw_degrees_per_frame": dataset_info.get("max_yaw_degrees_per_frame", 45.0),
        "max_pitch_degrees_per_frame": dataset_info.get("max_pitch_degrees_per_frame", 30.0),
        "require_right_mouse_for_camera": bool(dataset_info.get("require_right_mouse_for_camera", False)),
    }
    last_legacy_index = None
    last_legacy_timestamp = None
    legacy_segment = 0
    seen_indices = {}
    for line_number, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        report["metadata_rows"] += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            report["invalid_json"] += 1
            continue
        try:
            frame_index = int(row.get("frame_index", -1))
            timestamp = float(row.get("timestamp_seconds", 0.0))
        except (TypeError, ValueError):
            report["invalid_json"] += 1
            continue
        session_id = str(row.get("session_id", "")).strip()
        if not session_id:
            # Older recordings have no session_id. A frame-index reset, repeat, or
            # timestamp reset is a safe boundary rather than a fake camera teleport.
            if (last_legacy_index is not None and
                    (frame_index <= last_legacy_index or timestamp < last_legacy_timestamp)):
                legacy_segment += 1
            session_id = f"legacy-{legacy_segment}"
            last_legacy_index, last_legacy_timestamp = frame_index, timestamp
        row["_session_id"] = session_id
        row["_line_number"] = line_number
        path = frames_dir / str(row.get("filename", ""))
        if path.is_file():
            if str(row.get("camera_encoding", "")).lower() == "relative_degrees_v1":
                report["camera_degree_rows"] += 1
            report["camera_active_rows"] += int(bool(row.get("camera_active", False)))
            report["right_mouse_rows"] += int(bool(row.get("right_mouse", False)))
            report["absolute_yaw_degrees"] += abs(float(row.get("camera_yaw_delta_degrees", 0.0)))
            report["absolute_pitch_degrees"] += abs(float(row.get("camera_pitch_delta_degrees", 0.0)))
            row["path"] = path
            rows.append(row)
            report["valid_rows"] += 1
            key = (session_id, frame_index)
            seen_indices[key] = seen_indices.get(key, 0) + 1
        else:
            report["missing_images"] += 1

    report["duplicate_indices"] = sum(count - 1 for count in seen_indices.values() if count > 1)
    report["sessions"] = len({row["_session_id"] for row in rows if not row["_session_id"].startswith("legacy-")})
    report["legacy_segments"] = len({row["_session_id"] for row in rows if row["_session_id"].startswith("legacy-")})

    grouped = {}
    for row in rows:
        grouped.setdefault(row["_session_id"], []).append(row)
        active = False
        for name in ACTION_NAMES:
            value = float(row.get(name, 0.0))
            if abs(value) > (0.5 if name in ACTION_NAMES[:5] else 0.02):
                report["action_counts"][name] += 1
                active = True
        if not active:
            report["idle_rows"] += 1
    duration_seconds = 0.0
    for session_rows in grouped.values():
        timestamps = [float(row.get("timestamp_seconds", 0.0)) for row in session_rows]
        if timestamps:
            duration_seconds += max(0.0, max(timestamps) - min(timestamps))
    report["estimated_minutes"] = duration_seconds / 60.0
    pairs = []
    for session_rows in grouped.values():
        session_rows.sort(key=lambda item: (int(item.get("frame_index", -1)), item["_line_number"]))
        # Ambiguous duplicate frame labels are excluded; guessing one can train a
        # contradictory transition.
        unique_rows = [row for row in session_rows if seen_indices[(row["_session_id"], int(row["frame_index"]))] == 1]
        for start in range(0, len(unique_rows) - frame_gap):
            previous = unique_rows[start]
            target = unique_rows[start + frame_gap]
            prev_index = int(previous.get("frame_index", -1))
            target_index = int(target.get("frame_index", -1))
            if target_index != prev_index + frame_gap:
                continue
            window_rows = unique_rows[start + 1:start + frame_gap + 1]
            if any(int(row.get("frame_index", -1)) != prev_index + offset
                   for offset, row in enumerate(window_rows, 1)):
                continue
            action = aggregate_action_window(window_rows, action_aggregation)
            pairs.append((previous["path"], target["path"], action))
    report["valid_transitions"] = len(pairs)
    pair_counts, enabled_actions, binary_profiles, action_prototypes = observed_action_profiles(pairs)
    report["transition_action_counts"] = pair_counts
    report["enabled_action_names"] = enabled_actions
    report["observed_binary_actions"] = binary_profiles
    report["observed_action_prototypes"] = action_prototypes
    if report["camera_degree_rows"] and capture_fps:
        report["recommended_frame_gap"] = recommended_frame_gap(capture_fps, target_seconds=0.12)
    return report, pairs


def read_action_dataset(dataset_dir, frame_gap=1, action_aggregation="window"):
    report, pairs = inspect_action_dataset(dataset_dir, frame_gap, action_aggregation)
    if not pairs:
        raise ValueError(
            f"No valid labelled transitions were found for frame gap {frame_gap}. "
            "Try a smaller gap or inspect the dataset for missing images, duplicate indexes, or session boundaries."
        )
    return pairs


def select_training_chunk(pairs, chunk_size=0, chunk_offset=0, mode="balanced", seed=1234):
    """Select a repeatable chunk without always taking one contiguous recording slice."""
    chunk_size = int(chunk_size)
    if chunk_size <= 0 or chunk_size >= len(pairs):
        return list(pairs)

    mode = str(mode).strip().lower()
    if mode == "sequential":
        start = max(0, min(int(chunk_offset), max(0, len(pairs) - chunk_size)))
        return list(pairs[start:start + chunk_size])

    rng = random.Random(int(seed) + int(chunk_offset))
    if mode == "random":
        indexes = rng.sample(range(len(pairs)), chunk_size)
        return [pairs[index] for index in indexes]

    # Weighted sampling without replacement. Rare controls receive a moderate
    # inverse-square-root boost, while W is still allowed to remain naturally common.
    counts = [sum(float(pair[2][i]) > 0.5 for pair in pairs) for i in range(5)]
    reference = max(1, max(counts))
    ranked = []
    for index, pair in enumerate(pairs):
        active = [i for i, value in enumerate(pair[2][:5]) if float(value) > 0.5]
        if active:
            weight = max(math.sqrt(reference / max(1, counts[i])) for i in active)
        elif any(abs(float(value)) > 0.02 for value in pair[2][5:]):
            weight = 1.5
        else:
            weight = 1.0
        # Efraimidis-Spirakis weighted sampling key.
        key = rng.random() ** (1.0 / max(1e-6, weight))
        ranked.append((key, index))
    ranked.sort(reverse=True)
    selected = sorted(index for _key, index in ranked[:chunk_size])
    return [pairs[index] for index in selected]


class ActionPairDataset:
    def __init__(self, pairs, resolution, condition_noise=0.03):
        from torchvision import transforms
        self.pairs = [(str(a), str(b), list(action)) for a, b, action in pairs]
        self.height, self.width = parse_frame_size(resolution)
        self.condition_noise = float(condition_noise)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        import torch
        from PIL import Image, ImageOps
        prev_path, next_path, action = self.pairs[index]
        with Image.open(prev_path) as image:
            fitted = ImageOps.fit(
                image.convert("RGB"), (self.width, self.height),
                method=Image.Resampling.BILINEAR, centering=(0.5, 0.5),
            )
            previous = self.transform(fitted)
        with Image.open(next_path) as image:
            fitted = ImageOps.fit(
                image.convert("RGB"), (self.width, self.height),
                method=Image.Resampling.BILINEAR, centering=(0.5, 0.5),
            )
            target = self.transform(fitted)
        if self.condition_noise > 0:
            previous = (previous + torch.randn_like(previous) * self.condition_noise).clamp(-1, 1)
        return previous, target, torch.tensor(action, dtype=torch.float32)


def action_maps(action, height, width, dtype=None, scale=1.0):
    action = action * float(scale)
    maps = action[:, :, None, None].expand(-1, -1, height, width)
    return maps.to(dtype=dtype) if dtype is not None else maps


def save_action_model(model, output_dir, args, epoch, stopped=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "unet", safe_serialization=True)
    info = {
        "model_type": "action_conditioned_rectified_flow_video",
        "format_version": 3,
        "name": args.model_name,
        "resolution": f"{model_frame_size(model)[1]}x{model_frame_size(model)[0]}",
        "width": model_frame_size(model)[1],
        "height": model_frame_size(model)[0],
        "aspect_ratio": "16:9",
        "in_channels": int(model.config.in_channels),
        "action_names": ACTION_NAMES,
        "action_encoding": "broadcast_spatial_channels_with_neutral_dropout",
        "continuous_controls": ["mouse_dx", "mouse_dy", "zoom"],
        "enabled_action_names": list(getattr(args, "enabled_action_names", ACTION_NAMES)),
        "action_counts": dict(getattr(args, "dataset_action_counts", {})),
        "observed_binary_actions": list(getattr(args, "observed_binary_actions", [])),
        "observed_action_prototypes": list(getattr(args, "observed_action_prototypes", [])),
        "capture_fps": getattr(args, "capture_fps", None),
        "recommended_frame_gap": getattr(args, "recommended_frame_gap", None),
        "camera_encoding": str(getattr(args, "camera_encoding", "legacy_pixels")),
        "camera_input_source": str(getattr(args, "camera_input_source", "unknown")),
        "yaw_counts_per_360_degrees": getattr(args, "yaw_counts_per_360_degrees", None),
        "pitch_counts_per_180_degrees": getattr(args, "pitch_counts_per_180_degrees", None),
        "max_yaw_degrees_per_frame": float(getattr(args, "max_yaw_degrees_per_frame", 45.0)),
        "max_pitch_degrees_per_frame": float(getattr(args, "max_pitch_degrees_per_frame", 30.0)),
        "require_right_mouse_for_camera": bool(getattr(args, "require_right_mouse_for_camera", False)),
        "action_input_scale": float(getattr(args, "action_input_scale", 1.0)),
        "neutral_action_dropout": float(getattr(args, "neutral_action_dropout", 0.0)),
        "motion_loss_weight": float(getattr(args, "motion_loss_weight", 0.0)),
        "action_contrast_weight": float(getattr(args, "action_contrast_weight", 0.0)),
        "action_contrast_margin": float(getattr(args, "action_contrast_margin", 0.0)),
        "contrast_every": int(getattr(args, "contrast_every", 1)),
        "contrast_samples": int(getattr(args, "contrast_samples", 2)),
        "chunk_size": int(getattr(args, "chunk_size", 0)),
        "chunk_mode": str(getattr(args, "chunk_mode", "balanced")),
        "chunk_seed": int(getattr(args, "chunk_seed", 1234)),
        "balance_actions": bool(getattr(args, "balance_actions", False)),
        "frame_gap": int(getattr(args, "frame_gap", 1)),
        "action_aggregation": str(getattr(args, "action_aggregation", "window")),
        "validation_split_strategy": "idle_action_stratified_tail_with_gap_guard",
        "base_model": str(args.base_model),
        "initialization": (
            "continued_action_model" if getattr(args, "continue_action_model", "")
            else "pretrained_video_model" if getattr(args, "base_model", "")
            else "scratch_action_model"
        ),
        "epochs_completed": int(epoch),
        "stopped_early": bool(stopped),
        "created_or_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "action_flow_model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def save_recovery_checkpoint(model, output_dir, args, epoch, batch_index):
    """Atomically rotate an interruption-safe in-progress checkpoint."""
    output_dir = Path(output_dir)
    recovery_dir = output_dir / "recovery_checkpoint"
    previous_dir = output_dir / "recovery_checkpoint_previous"
    staging_dir = output_dir / "recovery_checkpoint_writing"
    shutil.rmtree(staging_dir, ignore_errors=True)
    save_action_model(model, staging_dir, args, epoch, stopped=True)
    info_path = staging_dir / "action_flow_model_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.update({
        "recovery_checkpoint": True,
        "recovery_batch": int(batch_index),
        "recovery_note": "Safe in-progress checkpoint. Select this folder as Continue action model to resume.",
    })
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    # Keep the prior recovery folder until the new one has been written completely.
    shutil.rmtree(previous_dir, ignore_errors=True)
    if recovery_dir.exists():
        recovery_dir.replace(previous_dir)
    staging_dir.replace(recovery_dir)


def sample_action_frame(model, previous, action, steps, device, dtype, generator, method="Euler", initial_noise=None, action_input_scale=1.0):
    import torch
    if initial_noise is None:
        x = torch.randn(previous.shape, generator=generator, device=device, dtype=dtype)
    else:
        x = initial_noise.to(device=device, dtype=dtype).clone()
    action = action.to(device=device, dtype=dtype)
    maps = action_maps(action, previous.shape[-2], previous.shape[-1], dtype=dtype, scale=action_input_scale)
    dt = 1.0 / max(1, int(steps))
    model.eval()
    with torch.inference_mode():
        for i in range(max(1, int(steps))):
            t = i / max(1, int(steps))
            ts = torch.full((previous.shape[0],), t * 1000.0, device=device, dtype=dtype)
            velocity = model(torch.cat([x, previous, maps], dim=1), ts).sample
            if method.lower() == "heun" and i < int(steps) - 1:
                predicted = x + dt * velocity
                ts_next = torch.full((previous.shape[0],), (t + dt) * 1000.0, device=device, dtype=dtype)
                velocity_next = model(torch.cat([predicted, previous, maps], dim=1), ts_next).sample
                x = x + 0.5 * dt * (velocity + velocity_next)
            else:
                x = x + dt * velocity
    # Prevent a single unstable ODE step from poisoning all later autoregressive frames.
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    return x.clamp(-1.0, 1.0)


def sample_guided_action_frame(
    model, previous, action, steps, device, dtype, generator,
    method="Euler", initial_noise=None, action_input_scale=1.0,
    guidance=PLAYER_ACTION_GUIDANCE,
):
    """Generate using the model's action-only difference from its neutral prediction.

    Neutral and conditioned inputs are evaluated together as a batch. This makes the
    pressed control visible without UI-side RGB blending, anchors, or smoothing.
    """
    import torch
    if initial_noise is None:
        x = torch.randn(previous.shape, generator=generator, device=device, dtype=dtype)
    else:
        x = initial_noise.to(device=device, dtype=dtype).clone()
    action = action.to(device=device, dtype=dtype)
    maps = action_maps(action, previous.shape[-2], previous.shape[-1], dtype=dtype, scale=action_input_scale)
    neutral_maps = torch.zeros_like(maps)
    dt = 1.0 / max(1, int(steps))

    def guided_velocity(state, ts):
        model_input = torch.cat([
            torch.cat([state, previous, neutral_maps], dim=1),
            torch.cat([state, previous, maps], dim=1),
        ], dim=0)
        both_ts = torch.cat([ts, ts], dim=0)
        neutral, conditioned = model(model_input, both_ts).sample.chunk(2, dim=0)
        return neutral + float(guidance) * (conditioned - neutral)

    model.eval()
    with torch.inference_mode():
        for i in range(max(1, int(steps))):
            t = i / max(1, int(steps))
            ts = torch.full((previous.shape[0],), t * 1000.0, device=device, dtype=dtype)
            velocity = guided_velocity(x, ts)
            if method.lower() == "heun" and i < int(steps) - 1:
                predicted = x + dt * velocity
                ts_next = torch.full((previous.shape[0],), (t + dt) * 1000.0, device=device, dtype=dtype)
                velocity_next = guided_velocity(predicted, ts_next)
                x = x + 0.5 * dt * (velocity + velocity_next)
            else:
                x = x + dt * velocity
    return torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def evaluate_action_model(model, val_loader, device, amp_enabled, args):
    """Measure held-out prediction quality and whether controls affect the model."""
    import torch
    import torch.nn.functional as F

    totals = {"flow": 0.0, "temporal": 0.0, "idle": 0.0, "action": 0.0,
              "counterfactual_correct": 0.0, "counterfactual_advantage": 0.0}
    counts = {"batches": 0, "idle": 0, "action": 0, "counterfactual": 0}
    generator = torch.Generator(device=device).manual_seed(24680)
    model.eval()
    diagnostic_previous = None
    diagnostic_x_t = None
    diagnostic_t = None

    with torch.inference_mode():
        for previous, clean, actions in val_loader:
            previous = previous.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            if device.type == "cuda":
                previous = previous.to(memory_format=torch.channels_last)
                clean = clean.to(memory_format=torch.channels_last)
            noise = torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype)
            t = torch.rand((clean.shape[0], 1, 1, 1), generator=generator, device=device)
            x_t = (1.0 - t) * noise + t * clean
            timesteps = t.flatten() * 1000.0
            maps = action_maps(actions, clean.shape[-2], clean.shape[-1], dtype=clean.dtype,
                                 scale=args.action_input_scale)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                target_velocity = (clean - noise).float()
                pred = model(torch.cat([x_t, previous, maps], dim=1), timesteps).sample.float()
                if len(args.observed_action_prototypes) >= 2:
                    wrong_actions = counterfactual_actions(actions, args.observed_action_prototypes)
                    wrong_maps = action_maps(
                        wrong_actions, clean.shape[-2], clean.shape[-1],
                        dtype=clean.dtype, scale=args.action_input_scale,
                    )
                    wrong_pred = model(
                        torch.cat([x_t, previous, wrong_maps], dim=1), timesteps,
                    ).sample.float()
            reconstructed = x_t.float() + (1.0 - t) * pred
            per_sample = torch.mean((pred - target_velocity) ** 2, dim=(1, 2, 3))
            if len(args.observed_action_prototypes) >= 2:
                wrong_error = torch.mean((wrong_pred - target_velocity) ** 2, dim=(1, 2, 3))
                totals["counterfactual_correct"] += float((per_sample < wrong_error).sum().item())
                totals["counterfactual_advantage"] += float((wrong_error - per_sample).sum().item())
                counts["counterfactual"] += int(clean.shape[0])
            idle_mask = (
                torch.all(actions[:, :5] < 0.5, dim=1)
                & torch.all(torch.abs(actions[:, 5:]) <= 0.02, dim=1)
            )
            action_mask = ~idle_mask
            totals["flow"] += float(per_sample.sum().item())
            totals["temporal"] += float(F.l1_loss(reconstructed, clean.float(), reduction="sum").item())
            counts["batches"] += clean.shape[0]
            if idle_mask.any():
                totals["idle"] += float(per_sample[idle_mask].sum().item())
                counts["idle"] += int(idle_mask.sum().item())
            if action_mask.any():
                totals["action"] += float(per_sample[action_mask].sum().item())
                counts["action"] += int(action_mask.sum().item())
            if diagnostic_previous is None:
                diagnostic_previous = previous[:1]
                diagnostic_x_t = x_t[:1]
                diagnostic_t = t[:1]

    pixel_count = max(1, counts["batches"] * diagnostic_previous.shape[1]
                      * diagnostic_previous.shape[2] * diagnostic_previous.shape[3])
    validation = {
        "flow_loss": totals["flow"] / max(1, counts["batches"]),
        "temporal_loss": totals["temporal"] / pixel_count,
        "idle_error": totals["idle"] / counts["idle"] if counts["idle"] else None,
        "action_error": totals["action"] / counts["action"] if counts["action"] else None,
        "counterfactual_accuracy": (
            totals["counterfactual_correct"] / counts["counterfactual"]
            if counts["counterfactual"] else None
        ),
        "counterfactual_error_advantage": (
            totals["counterfactual_advantage"] / counts["counterfactual"]
            if counts["counterfactual"] else None
        ),
        "samples": counts["batches"],
    }
    validation["total_loss"] = (
        validation["flow_loss"]
        + float(args.temporal_loss_weight) * validation["temporal_loss"]
    )

    # Compare velocity predictions for canonical controls using the exact same
    # previous frame, noise, and timestep. This isolates conditioning response.
    canonical_specs = [
        (name, index, 1.0) for index, name in enumerate(BINARY_ACTION_NAMES)
        if name in args.enabled_action_names
    ]
    if "mouse_dx" in args.enabled_action_names:
        canonical_specs.extend([("yaw_left", 5, -0.5), ("yaw_right", 5, 0.5)])
    if "mouse_dy" in args.enabled_action_names:
        canonical_specs.extend([("pitch_down", 6, -0.5), ("pitch_up", 6, 0.5)])
    if "zoom" in args.enabled_action_names:
        canonical_specs.extend([("zoom_out", 7, -0.5), ("zoom_in", 7, 0.5)])
    canonical = torch.zeros((1 + len(canonical_specs), ACTION_DIM), device=device,
                            dtype=diagnostic_previous.dtype)
    for row, (_label, index, value) in enumerate(canonical_specs, 1):
        canonical[row, index] = value
    sample_count = canonical.shape[0]
    previous_many = diagnostic_previous.expand(sample_count, -1, -1, -1)
    x_t_many = diagnostic_x_t.expand(sample_count, -1, -1, -1)
    t_many = diagnostic_t.expand(sample_count, -1, -1, -1)
    maps = action_maps(canonical, previous_many.shape[-2], previous_many.shape[-1],
                       dtype=previous_many.dtype, scale=args.action_input_scale)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=amp_enabled
    ):
        predictions = model(torch.cat([x_t_many, previous_many, maps], dim=1),
                            t_many.flatten() * 1000.0).sample.float()
    differences = torch.mean(torch.abs(predictions[1:] - predictions[:1]), dim=(1, 2, 3))
    sensitivity = {
        label: float(differences[row].item())
        for row, (label, _action_index, _value) in enumerate(canonical_specs)
    }
    sensitivity["mean"] = float(differences.mean().item()) if differences.numel() else 0.0
    return validation, sensitivity


def training_worker(args):
    import torch
    import torch.nn.functional as F
    from diffusers import UNet2DModel
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

    args.neutral_action_dropout = max(0.0, min(0.9, float(args.neutral_action_dropout)))
    args.motion_loss_weight = max(0.0, float(args.motion_loss_weight))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        event(type="warning", message="CUDA is unavailable. Training will be extremely slow.")

    amp_enabled = device.type == "cuda" and args.mixed_precision == "fp16"
    dataset_report, pairs = inspect_action_dataset(
        args.dataset_dir, args.frame_gap, args.action_aggregation,
    )
    if not pairs:
        raise ValueError(f"No valid labelled transitions were found for frame gap {args.frame_gap}.")
    pairs = select_training_chunk(
        pairs, args.chunk_size, args.chunk_offset,
        mode=args.chunk_mode, seed=args.chunk_seed,
    )
    if len(pairs) < 2:
        raise ValueError("The selected training chunk contains fewer than two transitions.")
    action_counts, enabled_action_names, observed_binary_actions, observed_action_prototypes = observed_action_profiles(pairs)
    if len(observed_action_prototypes) < 2:
        raise ValueError(
            "This dataset has fewer than two distinct observed control profiles. "
            "Record idle plus at least one active control before action training."
        )
    args.dataset_action_counts = action_counts
    args.enabled_action_names = enabled_action_names
    args.observed_binary_actions = observed_binary_actions
    args.observed_action_prototypes = observed_action_prototypes
    args.capture_fps = dataset_report.get("capture_fps")
    args.recommended_frame_gap = dataset_report.get("recommended_frame_gap")
    args.camera_encoding = dataset_report.get("camera_encoding", "legacy_pixels")
    args.camera_input_source = dataset_report.get("camera_input_source", "unknown")
    args.yaw_counts_per_360_degrees = dataset_report.get("yaw_counts_per_360_degrees")
    args.pitch_counts_per_180_degrees = dataset_report.get("pitch_counts_per_180_degrees")
    args.max_yaw_degrees_per_frame = dataset_report.get("max_yaw_degrees_per_frame", 45.0)
    args.max_pitch_degrees_per_frame = dataset_report.get("max_pitch_degrees_per_frame", 30.0)
    args.require_right_mouse_for_camera = dataset_report.get("require_right_mouse_for_camera", False)
    dataset = ActionPairDataset(pairs, args.resolution, args.condition_noise)

    validation_count = max(1, int(len(dataset) * args.validation_split)) if len(dataset) >= 10 else 1
    validation_count = min(validation_count, len(dataset) - 1)
    train_indexes, val_indexes = validation_split_indexes(pairs, validation_count, args.frame_gap)
    train_dataset = Subset(dataset, train_indexes)
    val_dataset = Subset(dataset, val_indexes)
    train_count = len(train_dataset)
    validation_count = len(val_dataset)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    sampler = None
    if args.balance_actions:
        # Oversample transitions containing less common controls (including signed
        # camera/zoom movement) without
        # rewriting files or metadata. A cap avoids turning a handful of rare clips
        # into almost every batch.
        selected_actions = [pairs[index][2] for index in train_dataset.indices]
        counts = [
            sum(abs(float(action[i])) > (0.5 if i < 5 else 0.02) for action in selected_actions)
            for i in range(ACTION_DIM)
        ]
        reference = max(1, max(counts))
        weights = []
        for action in selected_actions:
            active = [
                i for i, value in enumerate(action)
                if abs(float(value)) > (0.5 if i < 5 else 0.02)
            ]
            if active:
                weight = max(min(8.0, math.sqrt(reference / max(1, counts[i]))) for i in active)
            else:
                weight = 1.0
            weights.append(weight)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_dataset), replacement=True,
            generator=torch.Generator().manual_seed(4321),
        )
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = True
    loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    if args.continue_action_model:
        if not action_model_is_valid(args.continue_action_model):
            raise ValueError("The continuation folder is not a compatible action-conditioned model.")
        model = load_action_unet(args.continue_action_model)
        if model_frame_size(model) != parse_frame_size(args.resolution):
            actual_h, actual_w = model_frame_size(model)
            requested_h, requested_w = parse_frame_size(args.resolution)
            event(
                type="warning",
                message=(
                    f"Retargeting continuation model canvas from {actual_w}x{actual_h} to "
                    f"{requested_w}x{requested_h}. The learned convolutional weights are preserved."
                ),
            )
            set_model_frame_size(model, args.resolution)
        initialization = "continued action model"
    elif args.base_model:
        if not base_video_model_is_valid(args.base_model):
            raise ValueError("The selected optional Roblox Flow video model is invalid.")
        model = UNet2DModel.from_pretrained(str(args.base_model), subfolder="unet")
        if model_frame_size(model) != parse_frame_size(args.resolution):
            actual_h, actual_w = model_frame_size(model)
            requested_h, requested_w = parse_frame_size(args.resolution)
            event(
                type="warning",
                message=(
                    f"Retargeting base model canvas from {actual_w}x{actual_h} to "
                    f"{requested_w}x{requested_h}. The learned convolutional weights are preserved."
                ),
            )
            set_model_frame_size(model, args.resolution)
        model = expand_unet_for_actions(model)
        initialization = "pretrained Roblox Flow video model"
    else:
        model = build_action_unet_from_scratch(args.resolution)
        initialization = "new action model from scratch"
        event(
            type="warning",
            message=(
                "No Roblox Flow video model was supplied. A new action-conditioned U-Net was initialized "
                "from random weights and will learn both Roblox appearance and controls only from this action dataset."
            ),
        )

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)

    optimizer_kwargs = dict(lr=args.learning_rate, weight_decay=1e-4)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), fused=device.type == "cuda", **optimizer_kwargs)
        optimizer_mode = "fused" if device.type == "cuda" else "standard"
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        optimizer_mode = "standard"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total_updates = args.epochs * math.ceil(len(loader) / args.gradient_accumulation)
    warmup = max(1, int(total_updates * 0.05))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_updates - warmup)
        return max(0.10, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    stop_path = Path(args.output_dir) / STOP_FILE_NAME
    stop_path.unlink(missing_ok=True)
    optimizer.zero_grad(set_to_none=True)
    update = 0
    completed_epoch = 0
    started = time.perf_counter()
    last_update_finished = started
    last_recovery_time = started
    smoothed_update_seconds = None
    smoothed_normal_seconds = None
    smoothed_contrast_seconds = None
    telemetry_snapshot = system_telemetry(device)
    run_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    latest_metrics = {
        "schema_version": 2,
        "run_id": run_id,
        "status": "training",
        "model_name": args.model_name,
        "epoch": 0,
        "epochs": args.epochs,
        "update": 0,
        "total_updates": total_updates,
        "training": {"loss": None, "flow_loss": None, "temporal_loss": None,
                     "contrast_loss": None, "learning_rate": args.learning_rate},
        "speed": {"seconds_per_update": None, "normal_seconds_per_update": None,
                  "contrast_seconds_per_update": None, "data_load_seconds": None,
                  "eta_seconds": None},
        "telemetry": system_telemetry(device),
        "validation": None,
        "action_sensitivity": None,
        "dataset": {"transitions": len(pairs), "training": train_count,
                    "validation": validation_count,
                    "validation_idle": sum(action_is_idle(pairs[i][2]) for i in val_indexes),
                    "validation_action": sum(not action_is_idle(pairs[i][2]) for i in val_indexes),
                    "capture_fps": args.capture_fps,
                    "recommended_frame_gap": args.recommended_frame_gap,
                    "camera_encoding": args.camera_encoding,
                    "camera_input_source": args.camera_input_source,
                    "enabled_action_names": enabled_action_names,
                    "action_counts": action_counts},
        "settings": {
            "resolution": str(args.resolution),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "temporal_loss_weight": args.temporal_loss_weight,
            "action_contrast_weight": args.action_contrast_weight,
            "action_contrast_margin": args.action_contrast_margin,
            "neutral_action_dropout": args.neutral_action_dropout,
            "motion_loss_weight": args.motion_loss_weight,
            "frame_gap": args.frame_gap,
        },
    }
    publish_training_metrics(args.output_dir, latest_metrics, append_history=True)

    event(
        type="start",
        transitions=len(pairs),
        training=train_count,
        validation=validation_count,
        batches=len(loader),
        total_updates=total_updates,
        device=str(device),
        action_counts=action_counts,
        enabled_action_names=enabled_action_names,
        observed_binary_actions=observed_binary_actions,
        observed_action_prototypes=observed_action_prototypes,
        camera_encoding=args.camera_encoding,
        camera_input_source=args.camera_input_source,
        capture_fps=args.capture_fps,
        recommended_frame_gap=args.recommended_frame_gap,
        frame_gap=args.frame_gap,
        action_aggregation=args.action_aggregation,
        initialization=initialization,
        chunk_mode=args.chunk_mode,
        optimizer=optimizer_mode,
        contrast_every=args.contrast_every,
        contrast_samples=args.contrast_samples,
        balance_actions=bool(args.balance_actions),
        benchmark_only=bool(args.benchmark_only),
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        flow_loss_sum = 0.0
        temporal_loss_sum = 0.0
        contrast_loss_sum = 0.0
        loader_iterator = iter(loader)
        batch_index = 0
        while True:
            load_started = time.perf_counter()
            try:
                previous, clean, actions = next(loader_iterator)
            except StopIteration:
                break
            data_load_seconds = time.perf_counter() - load_started
            batch_index += 1
            if stop_path.exists():
                if not args.benchmark_only:
                    save_action_model(model.float(), args.output_dir, args, completed_epoch, stopped=True)
                    latest_metrics["status"] = "stopped"
                    publish_training_metrics(args.output_dir, latest_metrics, append_history=True)
                event(type="stopped", output_dir=args.output_dir, benchmark_only=bool(args.benchmark_only))
                return

            previous = previous.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            if device.type == "cuda":
                previous = previous.to(memory_format=torch.channels_last)
                clean = clean.to(memory_format=torch.channels_last)

            noise = torch.randn_like(clean)
            b = clean.shape[0]
            t = torch.rand((b, 1, 1, 1), device=device)
            x_t = (1.0 - t) * noise + t * clean
            timesteps = t.flatten() * 1000.0
            training_actions = actions.clone()
            if args.neutral_action_dropout > 0:
                neutral_mask = torch.rand((b,), device=device) < float(args.neutral_action_dropout)
                training_actions[neutral_mask] = 0.0
            maps = action_maps(
                training_actions, clean.shape[-2], clean.shape[-1],
                dtype=clean.dtype, scale=args.action_input_scale,
            )
            visible_motion = torch.mean(torch.abs(clean.float() - previous.float()), dim=1, keepdim=True)
            motion_baseline = visible_motion.mean(dim=(2, 3), keepdim=True).clamp_min(1e-4)
            motion_emphasis = (visible_motion / motion_baseline).clamp(0.0, 4.0)
            pixel_weights = 1.0 + float(args.motion_loss_weight) * motion_emphasis
            pixel_weights = pixel_weights / pixel_weights.mean().clamp_min(1e-6)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                target_velocity = (clean - noise).float()
                pred = model(torch.cat([x_t, previous, maps], dim=1), timesteps).sample
                flow_error = torch.mean((pred.float() - target_velocity) ** 2, dim=1, keepdim=True)
                flow_loss = torch.mean(flow_error * pixel_weights)
                reconstructed = x_t + (1.0 - t) * pred
                temporal_error = torch.mean(torch.abs(reconstructed.float() - clean.float()), dim=1, keepdim=True)
                temporal_loss = torch.mean(temporal_error * pixel_weights)

                contrast_loss = pred.float().new_tensor(0.0)
                contrast_active = (
                    args.action_contrast_weight > 0
                    and len(args.observed_action_prototypes) > 1
                    and (batch_index - 1) % max(1, args.contrast_every) == 0
                )
                if contrast_active:
                    # Compare guaranteed-different controls for the exact same state,
                    # noise, timestep, and target. Dataset-valid prototypes avoid teaching
                    # arbitrary responses for controls that were never recorded.
                    contrast_count = min(max(1, int(args.contrast_samples)), actions.shape[0])
                    contrast_indexes = torch.randperm(actions.shape[0], device=actions.device)[:contrast_count]
                    true_actions = actions[contrast_indexes]
                    wrong_actions = counterfactual_actions(true_actions, args.observed_action_prototypes)
                    true_maps = action_maps(true_actions, clean.shape[-2], clean.shape[-1],
                                            dtype=clean.dtype, scale=args.action_input_scale)
                    wrong_maps = action_maps(wrong_actions, clean.shape[-2], clean.shape[-1],
                                             dtype=clean.dtype, scale=args.action_input_scale)
                    paired_inputs = torch.cat([
                        torch.cat([x_t[contrast_indexes], previous[contrast_indexes], true_maps], dim=1),
                        torch.cat([x_t[contrast_indexes], previous[contrast_indexes], wrong_maps], dim=1),
                    ], dim=0)
                    paired_times = torch.cat([timesteps[contrast_indexes], timesteps[contrast_indexes]], dim=0)
                    true_pred, wrong_pred = model(paired_inputs, paired_times).sample.chunk(2, dim=0)
                    true_target = target_velocity[contrast_indexes]
                    true_err = torch.mean((true_pred.float() - true_target) ** 2, dim=(1, 2, 3))
                    wrong_err = torch.mean((wrong_pred.float() - true_target) ** 2, dim=(1, 2, 3))
                    margin = float(args.action_contrast_margin)
                    # Compensate for running contrast periodically so the expected
                    # contribution stays near the configured contrast weight.
                    contrast_loss = torch.relu(margin + true_err - wrong_err).mean() * max(1, args.contrast_every)

                loss = (flow_loss
                        + args.temporal_loss_weight * temporal_loss
                        + args.action_contrast_weight * contrast_loss) / args.gradient_accumulation

            scaler.scale(loss).backward()
            loss_sum += float(loss.item()) * args.gradient_accumulation
            flow_loss_sum += float(flow_loss.item())
            temporal_loss_sum += float(temporal_loss.item())
            contrast_loss_sum += float(contrast_loss.item())

            if batch_index % args.gradient_accumulation == 0 or batch_index == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                # CUDA work is asynchronous. Synchronize before timing so the graph
                # reports completed GPU batches rather than CPU command-queue timing.
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                now = time.perf_counter()
                update_seconds = now - last_update_finished
                last_update_finished = now
                # Ignore the first few startup-heavy updates. Use a gentle rolling
                # average rather than a median window: a median creates misleading
                # plateaus and sudden steps in the graph as old batches fall out.
                if update > 3:
                    if smoothed_update_seconds is None:
                        smoothed_update_seconds = update_seconds
                    else:
                        smoothed_update_seconds = (
                            0.15 * update_seconds + 0.85 * smoothed_update_seconds
                        )
                    if contrast_active:
                        smoothed_contrast_seconds = (update_seconds if smoothed_contrast_seconds is None else
                                                     0.15 * update_seconds + 0.85 * smoothed_contrast_seconds)
                    else:
                        smoothed_normal_seconds = (update_seconds if smoothed_normal_seconds is None else
                                                   0.15 * update_seconds + 0.85 * smoothed_normal_seconds)
                eta = None
                seconds_per_update = None
                if update > 7 and smoothed_update_seconds is not None:
                    seconds_per_update = smoothed_update_seconds
                    eta = seconds_per_update * max(0, total_updates - update)
                if update == 1 or update % max(1, args.metrics_every) == 0:
                    telemetry_snapshot = system_telemetry(device)
                event(
                    type="progress",
                    epoch=epoch,
                    epochs=args.epochs,
                    update=update,
                    total_updates=total_updates,
                    loss=loss_sum / batch_index,
                    lr=scheduler.get_last_lr()[0],
                    eta=eta,
                    seconds_per_update=seconds_per_update,
                    raw_seconds_per_update=update_seconds,
                    normal_seconds_per_update=smoothed_normal_seconds,
                    contrast_seconds_per_update=smoothed_contrast_seconds,
                    data_load_seconds=data_load_seconds,
                    contrast_active=contrast_active,
                    telemetry=telemetry_snapshot,
                )
                latest_metrics.update({"epoch": epoch, "update": update})
                latest_metrics["training"] = {
                    "loss": loss_sum / batch_index,
                    "flow_loss": flow_loss_sum / batch_index,
                    "temporal_loss": temporal_loss_sum / batch_index,
                    "contrast_loss": contrast_loss_sum / batch_index,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                latest_metrics["speed"] = {
                    "seconds_per_update": seconds_per_update,
                    "raw_seconds_per_update": update_seconds,
                    "normal_seconds_per_update": smoothed_normal_seconds,
                    "contrast_seconds_per_update": smoothed_contrast_seconds,
                    "data_load_seconds": data_load_seconds,
                    "eta_seconds": eta,
                }
                latest_metrics["telemetry"] = telemetry_snapshot
                if update == 1 or update % max(1, args.metrics_every) == 0:
                    publish_training_metrics(args.output_dir, latest_metrics, append_history=True)

                recovery_seconds = max(0, int(args.recovery_minutes)) * 60
                if (
                    not args.benchmark_only
                    and recovery_seconds
                    and now - last_recovery_time >= recovery_seconds
                ):
                    save_recovery_checkpoint(model, args.output_dir, args, completed_epoch, batch_index)
                    model.to(device)
                    if device.type == "cuda":
                        model.to(memory_format=torch.channels_last)
                    last_recovery_time = time.perf_counter()
                    # Saving a checkpoint is deliberate downtime, not a slow training batch.
                    last_update_finished = last_recovery_time
                    event(
                        type="recovery_saved",
                        epoch=completed_epoch,
                        batch=batch_index,
                        output_dir=str(Path(args.output_dir) / "recovery_checkpoint"),
                    )

                if args.benchmark_only and update >= max(1, args.benchmark_batches):
                    latest_metrics["status"] = "benchmark_complete"
                    latest_metrics["benchmark"] = {
                        "batches": update,
                        "seconds_per_update": seconds_per_update,
                        "normal_seconds_per_update": smoothed_normal_seconds,
                        "contrast_seconds_per_update": smoothed_contrast_seconds,
                        "estimated_epoch_seconds": (seconds_per_update * len(loader)) if seconds_per_update else None,
                    }
                    publish_training_metrics(args.output_dir, latest_metrics, append_history=True)
                    event(
                        type="benchmark_complete",
                        batches=update,
                        seconds_per_update=seconds_per_update,
                        estimated_epoch_seconds=(seconds_per_update * len(loader)) if seconds_per_update else None,
                        contrast_every=args.contrast_every,
                        contrast_samples=args.contrast_samples,
                        normal_seconds_per_update=smoothed_normal_seconds,
                        contrast_seconds_per_update=smoothed_contrast_seconds,
                    )
                    return

        completed_epoch = epoch
        validation_loader = val_loader
        if args.validation_batches > 0:
            from itertools import islice
            validation_loader = islice(val_loader, args.validation_batches)
        validation, sensitivity = evaluate_action_model(
            model, validation_loader, device, amp_enabled, args
        )
        latest_metrics["validation"] = validation
        latest_metrics["action_sensitivity"] = sensitivity
        latest_metrics["epoch"] = epoch
        publish_training_metrics(args.output_dir, latest_metrics, append_history=True)
        event(type="validation", epoch=epoch, validation=validation,
              action_sensitivity=sensitivity)
        last_update_finished = time.perf_counter()

        if epoch % args.preview_every == 0 or epoch == args.epochs:
            model.eval()
            previous, target, action = val_dataset[0]
            previous_b = previous.unsqueeze(0).to(device)
            action_b = action.unsqueeze(0).to(device)
            preview_generator = torch.Generator(device=device).manual_seed(12345)
            generated = sample_action_frame(
                model,
                previous_b,
                action_b,
                args.preview_steps,
                device,
                torch.float16 if amp_enabled else torch.float32,
                preview_generator,
                method="Euler",
                action_input_scale=args.action_input_scale,
            )
            from torchvision.transforms.functional import to_pil_image
            from PIL import Image, ImageDraw
            def to_pil(tensor):
                return to_pil_image(((tensor.detach().float().cpu().clamp(-1, 1) + 1) / 2).clamp(0, 1))
            images = [to_pil(previous), to_pil(generated[0]), to_pil(target)]
            frame_h, frame_w = parse_frame_size(args.resolution)
            panel = Image.new("RGB", (frame_w * 3, frame_h + 24), (20, 20, 24))
            labels = ["Previous", "Generated", "Target"]
            draw = ImageDraw.Draw(panel)
            for i, image in enumerate(images):
                panel.paste(image, (i * frame_w, 24))
                draw.text((i * frame_w + 5, 5), labels[i], fill=(240, 240, 240))
            preview_dir = Path(args.output_dir) / "gui_previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_dir / f"action_preview_epoch_{epoch:04d}.png"
            panel.save(preview_path)
            event(type="preview", epoch=epoch, path=str(preview_path))
            # Do not let preview generation distort the next batch's speed sample.
            last_update_finished = time.perf_counter()

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_action_model(model, args.output_dir, args, epoch)
            model.to(device)
            if device.type == "cuda":
                model.to(memory_format=torch.channels_last)
            event(type="saved", epoch=epoch, output_dir=args.output_dir)
            # Disk saving is also deliberate downtime, not training throughput.
            last_update_finished = time.perf_counter()

    latest_metrics["status"] = "complete"
    latest_metrics["epoch"] = completed_epoch
    publish_training_metrics(args.output_dir, latest_metrics, append_history=True)
    event(type="complete", output_dir=args.output_dir)


class Field(ttk.Frame):
    def __init__(self, parent, label, default="", values=None):
        super().__init__(parent, style="Card.TFrame")
        ttk.Label(self, text=label, style="Meta.TLabel").pack(anchor="w")
        self.var = tk.StringVar(value=str(default))
        if values:
            self.widget = ttk.Combobox(self, textvariable=self.var, values=values, state="readonly")
        else:
            self.widget = ttk.Entry(self, textvariable=self.var, style="Field.TEntry")
        self.widget.pack(fill="x", pady=(3, 0))

    def get(self):
        return self.var.get()


class TrainingSpeedChart(tk.Canvas):
    """Small dependency-free timeline for explaining changing training ETAs."""
    def __init__(self, parent):
        super().__init__(parent, height=175, bg=FIELD, highlightthickness=0)
        self.points = []
        self.started_at = None
        self.latest_eta = None
        self.bind("<Configure>", lambda _event: self.redraw())
        self.redraw()

    def reset(self):
        self.points = []
        self.started_at = time.perf_counter()
        self.latest_eta = None
        self.redraw()

    def add(self, seconds_per_batch, eta=None, contrast_active=False):
        if seconds_per_batch is None:
            return
        if self.started_at is None:
            self.started_at = time.perf_counter()
        elapsed = time.perf_counter() - self.started_at
        self.points.append((elapsed, float(seconds_per_batch), bool(contrast_active)))
        self.points = self.points[-600:]
        self.latest_eta = eta
        self.redraw()

    def redraw(self):
        self.delete("all")
        width = max(300, self.winfo_width())
        height = max(120, self.winfo_height())
        left, right, top, bottom = 52, 18, 30, 28
        plot_w, plot_h = width - left - right, height - top - bottom
        self.create_text(10, 10, anchor="w", fill=FG, font=("Segoe UI", 10, "bold"), text="Training speed over time")
        if not self.points:
            self.create_text(width / 2, height / 2, fill=DIM, text="Speed points will appear after the warm-up batches.")
            return

        max_time = max(60.0, self.points[-1][0])
        speeds = [point[1] for point in self.points]
        min_speed = min(speeds)
        max_speed = max(speeds)
        if max_speed - min_speed < 0.1:
            min_speed = max(0.0, min_speed - 0.5)
            max_speed += 0.5
        else:
            padding = (max_speed - min_speed) * 0.15
            min_speed = max(0.0, min_speed - padding)
            max_speed += padding

        for fraction in (0.0, 0.5, 1.0):
            y = top + plot_h * (1.0 - fraction)
            self.create_line(left, y, width - right, y, fill="#3a3f4d")
            value = min_speed + (max_speed - min_speed) * fraction
            self.create_text(left - 6, y, anchor="e", fill=DIM, font=("Segoe UI", 8), text=f"{value:.1f}s")

        colors = {False: ACCENT, True: "#ffb454"}
        for contrast in (False, True):
            coords = []
            for elapsed, speed, point_contrast in self.points:
                if point_contrast != contrast:
                    continue
                x = left + (elapsed / max_time) * plot_w
                y = top + (1.0 - (speed - min_speed) / (max_speed - min_speed)) * plot_h
                coords.extend((x, y))
            if len(coords) >= 4:
                self.create_line(*coords, fill=colors[contrast], width=2, smooth=True)
            elif coords:
                self.create_oval(coords[0] - 2, coords[1] - 2, coords[0] + 2, coords[1] + 2,
                                 fill=colors[contrast], outline="")

        self.create_text(left, height - 10, anchor="w", fill=DIM, font=("Segoe UI", 8), text="start")
        self.create_text(width - right, height - 10, anchor="e", fill=DIM, font=("Segoe UI", 8), text=f"{max_time / 60:.1f} min")
        latest = speeds[-1]
        eta_text = human_time(self.latest_eta) if self.latest_eta is not None else "calculating"
        self.create_text(width - right, 10, anchor="e", fill=DIM, font=("Segoe UI", 9), text=f"Normal blue  |  contrast orange  |  Latest {latest:.2f}s  |  remaining {eta_text}")


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

    def wheel(event):
        canvas.yview_scroll(int(-event.delta / 120), "units")

    canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", wheel))
    canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))
    return inner


class TrainTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, style="Panel.TFrame")
        self.app = app
        self.process = None
        self.pending_training_after_benchmark = False
        self.last_run_was_preflight = False
        self.events = queue.Queue()
        self.preview_photo = None
        self._settings_loaded = False
        self._build()
        self.load_training_settings()
        self._install_settings_autosave()

    def _build(self):
        content = scroll_panel(self)
        ttk.Label(content, text="Action-Conditioned Fine-Tuning", style="Heading.TLabel").pack(anchor="w", padx=18, pady=(18, 4))
        ttk.Label(
            content,
            text="Teaches controls using farther-ahead frame targets, aggregated action windows, boosted action channels, and contrast loss. The Roblox Flow video model is optional: leave it blank to train a new action-conditioned model entirely from the recorded action dataset.",
            style="Body.TLabel",
        ).pack(anchor="w", padx=18, pady=(0, 12))

        card = ttk.Frame(content, style="Card.TFrame", padding=14)
        card.pack(fill="x", padx=18, pady=5)

        self.dataset_var = tk.StringVar()
        self.base_model_var = tk.StringVar()
        self.continue_var = tk.StringVar()
        self.name_var = tk.StringVar(value="Roblox Action Flow")
        self.output_var = tk.StringVar(value=str(ACTION_MODELS_DIR / "Roblox Action Flow"))

        rows = [
            ("Recorded action dataset", self.dataset_var, self.browse_dataset),
            ("Roblox Flow video model (optional)", self.base_model_var, self.browse_base_model),
            ("Continue action model (optional)", self.continue_var, self.browse_continue),
            ("Save action model to", self.output_var, self.browse_output),
        ]
        for label, variable, command in rows:
            ttk.Label(card, text=label, style="Meta.TLabel").pack(anchor="w", pady=(5, 0))
            line = ttk.Frame(card, style="Card.TFrame")
            line.pack(fill="x", pady=(3, 0))
            ttk.Entry(line, textvariable=variable, style="Field.TEntry").pack(side="left", fill="x", expand=True)
            ttk.Button(line, text="Browse...", command=command).pack(side="left", padx=(7, 0))

        ttk.Label(
            card,
            text="Starting priority: Continue Action Model → optional Roblox Flow model → new random Action Model. Leave both model fields blank for the dataset-only experiment.",
            style="Meta.TLabel",
            wraplength=900,
            justify="left",
        ).pack(anchor="w", pady=(8, 2))
        ttk.Button(card, text="Clear Optional Roblox Flow Model", command=lambda: self.base_model_var.set("")).pack(anchor="w", pady=(0, 4))

        ttk.Label(card, text="Model name", style="Meta.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Entry(card, textvariable=self.name_var, style="Field.TEntry").pack(fill="x", pady=(3, 6))

        settings_buttons = ttk.Frame(card, style="Card.TFrame")
        settings_buttons.pack(fill="x", pady=(4, 8))
        ttk.Button(settings_buttons, text="Check Dataset", command=self.check_dataset).pack(side="left")
        ttk.Button(settings_buttons, text="Import Settings TXT/JSON", command=self.import_training_settings).pack(side="left")
        ttk.Button(settings_buttons, text="Export Current Settings", command=self.export_training_settings).pack(side="left", padx=(8, 0))
        ttk.Label(settings_buttons, text="Settings also autosave when changed.", style="Meta.TLabel").pack(side="left", padx=(12, 0))

        grid = ttk.Frame(card, style="Card.TFrame")
        grid.pack(fill="x")
        self.epochs = Field(grid, "Epochs", "30")
        self.resolution = Field(
            grid, "Training frame size (16:9)", "256x144", ["256x144", "512x288"]
        )
        self.batch = Field(grid, "Batch size", "2", ["1", "2", "4", "5"])
        self.lr = Field(grid, "Learning rate", "0.00002", ["0.00001", "0.00002", "0.00005", "0.0001"])
        self.workers = Field(grid, "Data workers", "4", ["0", "2", "4", "6", "8", "12"])
        self.grad = Field(grid, "Gradient accumulation", "1", ["1", "2", "4"])
        self.condition_noise = Field(grid, "Previous-frame corruption", "0.03", ["0.0", "0.01", "0.03", "0.05"])
        self.temporal_weight = Field(grid, "Temporal loss weight", "0.1", ["0.0", "0.05", "0.1", "0.2"])
        self.validation_split = Field(grid, "Validation split", "0.1", ["0.05", "0.1", "0.15"])
        self.frame_gap = Field(grid, "Training frame gap", "3", ["1", "2", "3", "4", "5", "6", "8"])
        self.action_aggregation = Field(
            grid,
            "Action aggregation across gap",
            "window",
            ["window", "mean", "last"],
        )
        self.action_scale = Field(grid, "Action input scale", "8.0", ["1.0", "2.0", "5.0", "8.0", "12.0"])
        self.neutral_dropout = Field(grid, "Neutral action dropout", "0.15", ["0.0", "0.1", "0.15", "0.2", "0.3"])
        self.motion_loss_weight = Field(grid, "Visible-motion loss boost", "2.0", ["0.0", "1.0", "2.0", "3.0", "4.0"])
        self.contrast_weight = Field(grid, "Action contrast weight", "0.35", ["0.0", "0.05", "0.1", "0.2", "0.35", "0.5"])
        self.contrast_margin = Field(grid, "Action contrast margin", "0.02", ["0.005", "0.01", "0.02", "0.05"])
        self.contrast_every = Field(grid, "Contrast every N batches", "4", ["1", "2", "4", "8"])
        self.contrast_samples = Field(grid, "Contrast samples on scheduled batch", "2", ["2", "3", "4", "5"])
        self.chunk_size = Field(grid, "Transitions per training chunk (0 = all)", "0", ["0", "384", "1000", "2000", "3000", "4000", "5000"])
        self.chunk_mode = Field(grid, "Chunk selection", "balanced", ["balanced", "random", "sequential"])
        self.chunk_offset = Field(grid, "Chunk seed/offset", "0")
        self.chunk_seed = Field(grid, "Chunk random seed", "1234")
        self.benchmark_batches = Field(grid, "Speed-test batches", "8", ["5", "8", "12", "20"])
        self.recovery_minutes = Field(grid, "Emergency recovery every minutes (0 = off)", "30", ["0", "15", "30", "45", "60"])
        self.save_every = Field(grid, "Save every N epochs", "10")
        self.preview_every = Field(grid, "Preview every N epochs", "5")
        self.preview_steps = Field(grid, "Preview flow steps", "1", ["1", "2", "4", "8"])
        self.preview_display_width = Field(
            grid, "Preview display width (UI only)", "1000",
            ["640", "820", "1000", "1200", "1400"],
        )
        self.precision = Field(grid, "Mixed precision", "fp16", ["fp16", "no"])
        fields = [
            self.epochs, self.resolution, self.batch, self.lr, self.workers, self.grad,
            self.condition_noise, self.temporal_weight, self.validation_split,
            self.frame_gap, self.action_aggregation,
            self.action_scale, self.neutral_dropout, self.motion_loss_weight,
            self.contrast_weight, self.contrast_margin,
            self.contrast_every, self.contrast_samples,
            self.chunk_size, self.chunk_mode, self.chunk_offset, self.chunk_seed,
            self.benchmark_batches, self.recovery_minutes,
            self.save_every, self.preview_every, self.preview_steps,
            self.preview_display_width, self.precision,
        ]
        for i, field in enumerate(fields):
            field.grid(row=i // 2, column=i % 2, sticky="ew", padx=(0 if i % 2 == 0 else 8, 8 if i % 2 == 0 else 0), pady=5)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        self.tf32_var = tk.BooleanVar(value=True)
        self.ckpt_var = tk.BooleanVar(value=False)
        self.balance_actions_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(card, text="Use TF32 acceleration", variable=self.tf32_var).pack(anchor="w", pady=(8, 2))
        ttk.Checkbutton(card, text="Gradient checkpointing", variable=self.ckpt_var).pack(anchor="w")
        ttk.Checkbutton(
            card,
            text="Balance movement, camera, and zoom sampling (does not alter dataset files)",
            variable=self.balance_actions_var,
        ).pack(anchor="w", pady=(2, 0))

        buttons = ttk.Frame(card, style="Card.TFrame")
        buttons.pack(fill="x", pady=(12, 0))
        self.start_btn = ttk.Button(buttons, text="Start Action Training", command=self.start, style="Accent.TButton")
        self.start_btn.pack(side="left")
        self.benchmark_btn = ttk.Button(buttons, text="Benchmark 8 Batches", command=lambda: self.start(benchmark=True))
        self.benchmark_btn.pack(side="left", padx=(8, 0))
        self.stop_btn = ttk.Button(buttons, text="Stop && Save", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)

        self.progress = ttk.Progressbar(card, mode="determinate")
        self.progress.pack(fill="x", pady=(10, 5))
        self.status = ttk.Label(card, text="Ready.", style="Body.TLabel")
        self.status.pack(anchor="w")
        self.speed_chart = TrainingSpeedChart(card)
        self.speed_chart.pack(fill="x", pady=(8, 0))

        preview_card = ttk.Frame(content, style="Card.TFrame", padding=10)
        preview_card.pack(fill="x", padx=18, pady=(8, 0))
        ttk.Label(preview_card, text="Previous / Generated / Target", style="CardHeading.TLabel").pack(anchor="w", pady=(0, 6))
        self.preview_label = tk.Label(preview_card, text="A comparison preview will appear during training.", bg=FIELD, fg=DIM, height=13)
        self.preview_label.pack(fill="both", expand=True)

        self.log = tk.Text(content, height=12, bg=FIELD, fg=FG, insertbackground=FG, relief="flat", wrap="word")
        self.log.pack(fill="both", expand=True, padx=18, pady=(8, 18))

    def _settings_variables(self):
        return {
            "dataset_dir": self.dataset_var,
            "base_model": self.base_model_var,
            "continue_action_model": self.continue_var,
            "model_name": self.name_var,
            "output_dir": self.output_var,
            "epochs": self.epochs.var,
            "resolution": self.resolution.var,
            "batch_size": self.batch.var,
            "learning_rate": self.lr.var,
            "workers": self.workers.var,
            "gradient_accumulation": self.grad.var,
            "condition_noise": self.condition_noise.var,
            "temporal_loss_weight": self.temporal_weight.var,
            "validation_split": self.validation_split.var,
            "frame_gap": self.frame_gap.var,
            "action_aggregation": self.action_aggregation.var,
            "action_input_scale": self.action_scale.var,
            "neutral_action_dropout": self.neutral_dropout.var,
            "motion_loss_weight": self.motion_loss_weight.var,
            "action_contrast_weight": self.contrast_weight.var,
            "action_contrast_margin": self.contrast_margin.var,
            "contrast_every": self.contrast_every.var,
            "contrast_samples": self.contrast_samples.var,
            "chunk_size": self.chunk_size.var,
            "chunk_mode": self.chunk_mode.var,
            "chunk_offset": self.chunk_offset.var,
            "chunk_seed": self.chunk_seed.var,
            "benchmark_batches": self.benchmark_batches.var,
            "recovery_minutes": self.recovery_minutes.var,
            "save_every": self.save_every.var,
            "preview_every": self.preview_every.var,
            "preview_steps": self.preview_steps.var,
            "preview_display_width": self.preview_display_width.var,
            "mixed_precision": self.precision.var,
            "tf32": self.tf32_var,
            "gradient_checkpointing": self.ckpt_var,
            "balance_actions": self.balance_actions_var,
        }

    def collect_training_settings(self):
        values = {}
        for key, variable in self._settings_variables().items():
            values[key] = variable.get()
        values["format"] = "roblox_action_flow_training_settings_v2"
        return values

    def apply_training_settings(self, settings):
        aliases = {
            "batch": "batch_size", "lr": "learning_rate", "grad": "gradient_accumulation",
            "condition_corruption": "condition_noise", "previous_frame_corruption": "condition_noise",
            "temporal_weight": "temporal_loss_weight", "action_scale": "action_input_scale",
            "contrast_weight": "action_contrast_weight", "contrast_margin": "action_contrast_margin",
            "contrast_interval": "contrast_every", "contrast_batch": "contrast_samples",
            "precision": "mixed_precision", "chunk": "chunk_size", "offset": "chunk_offset",
            "dataset": "dataset_dir", "continue_model": "continue_action_model", "output": "output_dir",
        }
        normalized = {}
        for key, value in settings.items():
            clean = str(key).strip().lower().replace("-", "_").replace(" ", "_")
            normalized[aliases.get(clean, clean)] = value

        variables = self._settings_variables()
        applied = []
        for key, value in normalized.items():
            if key not in variables or key == "format":
                continue
            variable = variables[key]
            if isinstance(variable, tk.BooleanVar):
                if isinstance(value, str):
                    value = value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
                variable.set(bool(value))
            else:
                variable.set(str(value))
            applied.append(key)
        return applied

    def save_training_settings(self, silent=True):
        try:
            payload = self.collect_training_settings()
            temp = TRAINING_SETTINGS_FILE.with_suffix(".tmp")
            temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temp.replace(TRAINING_SETTINGS_FILE)
            if not silent:
                self.status.config(text=f"Saved training settings to {TRAINING_SETTINGS_FILE.name}.")
            return True
        except Exception as exc:
            if not silent:
                messagebox.showerror("Could not save settings", str(exc))
            return False

    def load_training_settings(self):
        self._settings_loaded = False
        try:
            if TRAINING_SETTINGS_FILE.is_file():
                settings = json.loads(TRAINING_SETTINGS_FILE.read_text(encoding="utf-8"))
                self.apply_training_settings(settings)
                self.status.config(text=f"Restored autosaved settings from {TRAINING_SETTINGS_FILE.name}.")
        except Exception as exc:
            self.status.config(text=f"Could not restore saved settings: {exc}")
        finally:
            self._settings_loaded = True

    def _install_settings_autosave(self):
        self._autosave_after_id = None

        def changed(*_args):
            if not self._settings_loaded:
                return
            if self._autosave_after_id is not None:
                try:
                    self.after_cancel(self._autosave_after_id)
                except Exception:
                    pass
            self._autosave_after_id = self.after(600, lambda: self.save_training_settings(silent=True))

        for variable in self._settings_variables().values():
            variable.trace_add("write", changed)

    @staticmethod
    def _parse_settings_text(raw_text):
        stripped = raw_text.strip()
        if not stripped:
            raise ValueError("The selected settings file is empty.")
        if stripped.startswith("{"):
            data = json.loads(stripped)
            if not isinstance(data, dict):
                raise ValueError("JSON settings must contain one object of key/value pairs.")
            return data

        data = {}
        for line_number, original in enumerate(raw_text.splitlines(), 1):
            line = original.strip()
            if not line or line.startswith(("#", ";", "//")):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
            elif ":" in line:
                key, value = line.split(":", 1)
            else:
                raise ValueError(f"Line {line_number} must use key=value or key: value format.")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                raise ValueError(f"Line {line_number} has no setting name.")
            data[key] = value
        if not data:
            raise ValueError("No settings were found in the selected file.")
        return data

    def import_training_settings(self):
        path = filedialog.askopenfilename(
            title="Import action-training settings",
            filetypes=[("Settings files", "*.txt *.json *.cfg *.ini"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            settings = self._parse_settings_text(Path(path).read_text(encoding="utf-8-sig"))
            applied = self.apply_training_settings(settings)
            if not applied:
                raise ValueError("The file did not contain any recognized training setting names.")
            self.save_training_settings(silent=True)
            self.status.config(text=f"Imported {len(applied)} settings from {Path(path).name}.")
            messagebox.showinfo("Settings imported", f"Imported {len(applied)} training settings.\n\nUnknown entries, if any, were ignored.")
        except Exception as exc:
            messagebox.showerror("Could not import settings", str(exc))

    def export_training_settings(self):
        path = filedialog.asksaveasfilename(
            title="Export action-training settings",
            defaultextension=".txt",
            initialfile="action_training_settings.txt",
            filetypes=[("Text settings", "*.txt"), ("JSON settings", "*.json")],
        )
        if not path:
            return
        try:
            settings = self.collect_training_settings()
            target = Path(path)
            if target.suffix.lower() == ".json":
                target.write_text(json.dumps(settings, indent=2), encoding="utf-8")
            else:
                lines = [
                    "# Roblox Action Flow training settings",
                    "# Import this file with the Import Settings TXT/JSON button.",
                ]
                for key, value in settings.items():
                    if key != "format":
                        lines.append(f"{key}={value}")
                target.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.status.config(text=f"Exported settings to {target.name}.")
        except Exception as exc:
            messagebox.showerror("Could not export settings", str(exc))

    def browse_dataset(self):
        value = filedialog.askdirectory(title="Select the recorder dataset folder")
        if value:
            self.dataset_var.set(value)

    def check_dataset(self):
        try:
            dataset = Path(self.dataset_var.get())
            if not dataset.is_dir():
                raise ValueError("Select the recorded action dataset folder first.")
            report, _pairs = inspect_action_dataset(dataset, int(self.frame_gap.get()), self.action_aggregation.get())
            problems = []
            warnings = []
            if report["missing_images"]:
                problems.append(f"{report['missing_images']:,} labels point to missing image files")
            if report["duplicate_indices"]:
                problems.append(f"{report['duplicate_indices']:,} duplicate frame indexes were excluded")
            if report["invalid_json"]:
                problems.append(f"{report['invalid_json']:,} invalid metadata lines were skipped")
            usable = max(1, report["valid_rows"])
            idle_pct = 100.0 * report["idle_rows"] / usable
            binary_rates = {
                name: 100.0 * report["action_counts"][name] / usable
                for name in ACTION_NAMES[:5]
            }
            for name, rate in binary_rates.items():
                if rate == 0.0:
                    continue
                if rate < 5.0:
                    warnings.append(f"{name.upper()} is rare ({rate:.1f}% of frames); test it carefully or record more clean examples")
            camera_rate = 100.0 * report["camera_active_rows"] / usable
            right_mouse_rate = 100.0 * report["right_mouse_rows"] / usable
            if report["right_mouse_rows"] and not report["camera_active_rows"]:
                problems.append("right mouse was recorded, but no camera yaw/pitch labels were captured")
            elif report["camera_degree_rows"] and 0.0 < camera_rate < 5.0:
                warnings.append(f"Degree-labelled camera movement is rare ({camera_rate:.1f}% of frames)")
            if idle_pct > 30.0:
                warnings.append(f"Idle input is common ({idle_pct:.1f}% of frames)")
            quality = "READY" if not problems else "NEEDS CLEANUP"
            summary = (
                f"{quality}\n\n"
                f"Metadata rows: {report['metadata_rows']:,}\n"
                f"Usable frame rows: {report['valid_rows']:,}\n"
                f"Valid transitions at gap {int(self.frame_gap.get())}: {report['valid_transitions']:,}\n"
                f"New-format sessions: {report['sessions']:,}\n"
                f"Legacy recording segments: {report['legacy_segments']:,}\n\n"
                f"Estimated recorded time: {report['estimated_minutes']:.1f} minutes\n"
                f"Capture FPS: {report['capture_fps'] or 'unknown'}\n"
                f"Recommended frame gap: {report['recommended_frame_gap'] or 'unknown'} "
                f"(current: {int(self.frame_gap.get())})\n"
                f"Idle: {idle_pct:.1f}%\n"
                f"Enabled controls: {', '.join(ACTION_DISPLAY_NAMES.get(name, name) for name in report['enabled_action_names']) or 'none'}\n"
                f"Camera encoding: {report['camera_encoding']} ({report['camera_input_source']})\n"
                f"Camera-labelled frames: {camera_rate:.1f}% | right mouse: {right_mouse_rate:.1f}%\n"
                f"Recorded rotation: yaw {report['absolute_yaw_degrees']:.1f}° | pitch {report['absolute_pitch_degrees']:.1f}°\n"
                + "Action coverage: "
                + ", ".join(f"{name.upper()} {rate:.1f}%" for name, rate in binary_rates.items())
                + "\n\n"
                + ("Issues:\n- " + "\n- ".join(problems) if problems else "No missing images, duplicate labels, or invalid JSON found.")
                + ("\n\nWarnings:\n- " + "\n- ".join(warnings) if warnings else "")
            )
            self.status.config(text=f"Dataset check: {quality.lower()} — {report['valid_transitions']:,} valid transitions.")
            self.append("DATASET CHECK\n" + summary)
            messagebox.showinfo("Action dataset check", summary)
        except Exception as exc:
            messagebox.showerror("Dataset check failed", str(exc))

    def browse_base_model(self):
        value = filedialog.askdirectory(title="Select an optional Roblox Flow video model")
        if value:
            if not base_video_model_is_valid(value):
                messagebox.showerror("Invalid model", "Choose a folder containing flow_video_model_info.json and an unet folder.")
            else:
                self.base_model_var.set(value)
                try:
                    info = json.loads((Path(value) / "flow_video_model_info.json").read_text(encoding="utf-8"))
                    resolution = info.get("resolution")
                    if not resolution and info.get("width") and info.get("height"):
                        resolution = f"{info['width']}x{info['height']}"
                    self.resolution.var.set(str(resolution or "256x144"))
                except Exception:
                    pass

    def browse_continue(self):
        value = filedialog.askdirectory(initialdir=ACTION_MODELS_DIR, title="Continue an action-conditioned model")
        if value:
            if not action_model_is_valid(value):
                messagebox.showerror("Invalid model", "Choose a saved action-conditioned Flow model.")
            else:
                self.continue_var.set(value)

    def browse_output(self):
        value = filedialog.askdirectory(initialdir=ACTION_MODELS_DIR)
        if value:
            self.output_var.set(value)

    def append(self, text):
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")

    def start(self, benchmark=False, preflight=False):
        if not benchmark and not preflight:
            self.pending_training_after_benchmark = True
            self.append("Running the same-settings speed benchmark before full training...")
            return self.start(benchmark=True, preflight=True)
        try:
            self.save_training_settings(silent=True)
            dataset = Path(self.dataset_var.get())
            if not dataset.is_dir():
                raise ValueError("Select the recorded action dataset folder.")
            frame_gap = int(self.frame_gap.get())
            action_aggregation = self.action_aggregation.get()
            report, pairs = inspect_action_dataset(dataset, frame_gap, action_aggregation)
            if len(pairs) < 10:
                raise ValueError("The action dataset contains too few valid transitions.")
            recommended_gap = report.get("recommended_frame_gap")
            if recommended_gap and recommended_gap != frame_gap:
                self.append(
                    f"FRAME-GAP NOTE: this {report.get('capture_fps')} FPS capture recommends gap "
                    f"{recommended_gap}; current gap is {frame_gap}. Camera degree datasets use a shorter horizon."
                )
            if report["missing_images"] or report["duplicate_indices"] or report["invalid_json"]:
                proceed = messagebox.askyesno(
                    "Dataset has issues",
                    f"The trainer will safely exclude {report['missing_images']:,} missing-image labels, "
                    f"{report['duplicate_indices']:,} duplicate labels, and {report['invalid_json']:,} invalid lines.\n\n"
                    f"{len(pairs):,} valid transitions remain. Train with the clean subset?",
                )
                if not proceed:
                    return
            base_value = self.base_model_var.get().strip()
            continue_value = self.continue_var.get().strip()
            if continue_value and not action_model_is_valid(continue_value):
                raise ValueError("The selected continuation Action Model is invalid.")
            if (not continue_value) and base_value and not base_video_model_is_valid(base_value):
                raise ValueError("The selected optional Roblox Flow video model is invalid. Clear the field to train from scratch.")

            values = {
                "epochs": int(self.epochs.get()),
                "resolution": self.resolution.get().strip(),
                "batch": int(self.batch.get()),
                "lr": float(self.lr.get()),
                "workers": int(self.workers.get()),
                "grad": int(self.grad.get()),
                "condition_noise": float(self.condition_noise.get()),
                "temporal_weight": float(self.temporal_weight.get()),
                "validation_split": float(self.validation_split.get()),
                "frame_gap": frame_gap,
                "action_aggregation": action_aggregation,
                "action_scale": float(self.action_scale.get()),
                "neutral_dropout": max(0.0, min(0.9, float(self.neutral_dropout.get()))),
                "motion_loss_weight": max(0.0, float(self.motion_loss_weight.get())),
                "contrast_weight": float(self.contrast_weight.get()),
                "contrast_margin": float(self.contrast_margin.get()),
                "contrast_every": max(1, int(self.contrast_every.get())),
                "contrast_samples": max(2, int(self.contrast_samples.get())),
                "chunk_size": int(self.chunk_size.get()),
                "chunk_mode": self.chunk_mode.get().strip().lower(),
                "chunk_offset": int(self.chunk_offset.get()),
                "chunk_seed": int(self.chunk_seed.get()),
                "benchmark_batches": max(1, int(self.benchmark_batches.get())),
                "recovery_minutes": max(0, int(self.recovery_minutes.get())),
                "save_every": int(self.save_every.get()),
                "preview_every": int(self.preview_every.get()),
                "preview_steps": int(self.preview_steps.get()),
                "balance_actions": bool(self.balance_actions_var.get()),
            }
            # Validate here so mistakes are reported in the UI before a worker starts.
            parse_frame_size(values["resolution"])
            output = Path(self.output_var.get())
            output.mkdir(parents=True, exist_ok=True)
            (output / STOP_FILE_NAME).unlink(missing_ok=True)

            command = [
                sys.executable, str(Path(__file__).resolve()), "--train-worker",
                "--dataset-dir", str(dataset),
                "--base-model", base_value,
                "--output-dir", str(output),
                "--model-name", self.name_var.get().strip() or output.name,
                "--epochs", str(values["epochs"]),
                "--resolution", str(values["resolution"]),
                "--batch-size", str(values["batch"]),
                "--learning-rate", str(values["lr"]),
                "--workers", str(values["workers"]),
                "--gradient-accumulation", str(values["grad"]),
                "--condition-noise", str(values["condition_noise"]),
                "--temporal-loss-weight", str(values["temporal_weight"]),
                "--validation-split", str(values["validation_split"]),
                "--frame-gap", str(values["frame_gap"]),
                "--action-aggregation", str(values["action_aggregation"]),
                "--action-input-scale", str(values["action_scale"]),
                "--neutral-action-dropout", str(values["neutral_dropout"]),
                "--motion-loss-weight", str(values["motion_loss_weight"]),
                "--action-contrast-weight", str(values["contrast_weight"]),
                "--action-contrast-margin", str(values["contrast_margin"]),
                "--contrast-every", str(values["contrast_every"]),
                "--contrast-samples", str(values["contrast_samples"]),
                "--chunk-size", str(values["chunk_size"]),
                "--chunk-mode", str(values["chunk_mode"]),
                "--chunk-offset", str(values["chunk_offset"]),
                "--chunk-seed", str(values["chunk_seed"]),
                "--benchmark-batches", str(values["benchmark_batches"]),
                "--recovery-minutes", str(values["recovery_minutes"]),
                "--save-every", str(values["save_every"]),
                "--preview-every", str(values["preview_every"]),
                "--preview-steps", str(values["preview_steps"]),
                "--mixed-precision", self.precision.get(),
            ]
            if continue_value:
                command += ["--continue-action-model", continue_value]
            if self.tf32_var.get():
                command.append("--tf32")
            if self.ckpt_var.get():
                command.append("--gradient-checkpointing")
            if values["balance_actions"]:
                command.append("--balance-actions")
            if benchmark:
                command.append("--benchmark-only")

            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=flags,
            )
            self.last_run_was_preflight = bool(preflight)
            self.start_btn.config(state="disabled")
            self.benchmark_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status.config(text="Speed benchmark is starting..." if benchmark else "Action training is starting...")
            self.progress["value"] = 0
            self.speed_chart.reset()
            threading.Thread(target=self.read_output, daemon=True).start()
            self.after(100, self.poll)
        except Exception as exc:
            messagebox.showerror("Cannot start action training", str(exc))

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
                    self.finished(item[1])
                    return
                if item.startswith("ACTION_FLOW_EVENT:"):
                    self.handle_event(json.loads(item.split(":", 1)[1]))
                else:
                    self.append(item)
        except queue.Empty:
            pass
        if self.process:
            self.after(100, self.poll)

    def handle_event(self, info):
        kind = info.get("type")
        if kind == "start":
            self.append(
                f"Loaded {info['transitions']} transitions: {info['training']} train, "
                f"{info['validation']} validation, on {info['device']}."
            )
            self.append(
                f"Frame gap: {info.get('frame_gap', 1)}  |  "
                f"Action aggregation: {info.get('action_aggregation', 'window')}"
            )
            self.append(f"Initialization: {info.get('initialization', 'unknown')}")
            self.append(
                f"Chunk selection: {info.get('chunk_mode', 'sequential')}  |  "
                f"Optimizer: {info.get('optimizer', 'standard')}  |  "
                f"Contrast: {info.get('contrast_samples', '?')} samples every "
                f"{info.get('contrast_every', 1)} batches"
            )
            self.append(
                "Control balancing: " + (
                    "enabled (rare controls are oversampled)" if info.get("balance_actions")
                    else "disabled"
                )
            )
            self.append(
                f"Camera labels: {info.get('camera_encoding', 'legacy_pixels')}  |  "
                f"source: {info.get('camera_input_source', 'unknown')}"
            )
            self.append("Action counts: " + ", ".join(f"{k.upper()}={v}" for k, v in info.get("action_counts", {}).items()))
        elif kind == "progress":
            self.progress["maximum"] = info["total_updates"]
            self.progress["value"] = info["update"]
            speed = info.get("seconds_per_update")
            speed_text = "warming up" if speed is None else f"{speed:.2f}s/batch"
            self.status.config(
                text=f"Epoch {info['epoch']}/{info['epochs']}  |  loss {info['loss']:.4f}  |  "
                     f"lr {info['lr']:.2e}  |  {speed_text}  |  ETA {human_time(info.get('eta'))}"
            )
            self.speed_chart.add(info.get("raw_seconds_per_update"), info.get("eta"), info.get("contrast_active", False))
        elif kind == "benchmark_complete":
            seconds = info.get("seconds_per_update")
            epoch_seconds = info.get("estimated_epoch_seconds")
            if seconds is None:
                summary = "Benchmark finished, but too few stable samples were collected. Increase Speed-test batches."
            else:
                summary = (
                    f"Benchmark complete: {seconds:.2f} seconds per batch; "
                    f"estimated epoch {human_time(epoch_seconds)} with the current settings."
                )
                normal = info.get("normal_seconds_per_update")
                contrast = info.get("contrast_seconds_per_update")
                if normal is not None or contrast is not None:
                    summary += f" Normal {normal if normal is not None else float('nan'):.2f}s; contrast {contrast if contrast is not None else float('nan'):.2f}s."
            self.status.config(text=summary)
            self.append(summary)
        elif kind == "validation":
            validation = info.get("validation", {})
            sensitivity = info.get("action_sensitivity", {})
            idle = validation.get("idle_error")
            action = validation.get("action_error")
            idle_text = "n/a" if idle is None else f"{idle:.4f}"
            action_text = "n/a" if action is None else f"{action:.4f}"
            counterfactual_accuracy = validation.get("counterfactual_accuracy")
            counterfactual_text = ("n/a" if counterfactual_accuracy is None
                                   else f"{100.0 * counterfactual_accuracy:.1f}%")
            self.append(
                f"Validation epoch {info['epoch']}: loss {validation.get('total_loss', 0):.4f}, "
                f"idle {idle_text}, action {action_text}, "
                f"counterfactual accuracy {counterfactual_text}, "
                f"mean control response {sensitivity.get('mean', 0):.5f}."
            )
            self.append(
                "Control response: " + ", ".join(
                    f"{ACTION_DISPLAY_NAMES.get(name, name.upper())}={value:.5f}"
                    for name, value in sensitivity.items() if name != "mean"
                )
            )
        elif kind == "preview":
            self.show_preview(info["path"], info["epoch"])
        elif kind == "saved":
            self.append(f"Saved action model at epoch {info['epoch']}.")
        elif kind == "recovery_saved":
            self.append(f"Emergency recovery saved at batch {info['batch']}. It can resume from recovery_checkpoint.")
        elif kind == "warning":
            self.append("WARNING: " + info.get("message", ""))
        elif kind == "stopped":
            if info.get("benchmark_only"):
                self.pending_training_after_benchmark = False
            self.status.config(text="Stopped safely" + ("." if info.get("benchmark_only") else " and saved."))
        elif kind == "complete":
            self.status.config(text="Action training complete!")

    def show_preview(self, path, epoch):
        try:
            from PIL import Image, ImageTk
            image = Image.open(path).convert("RGB")
            display_width = max(320, min(2000, int(self.preview_display_width.get())))
            # This resizes only the Tk preview; the saved preview and model tensors
            # retain the selected training dimensions.
            display_height = max(180, round(display_width * image.height / image.width))
            image.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview_label.config(image=self.preview_photo, text="", height=0)
            self.append(f"Generated comparison preview at epoch {epoch}.")
        except Exception as exc:
            self.append(f"Could not display preview: {exc}")

    def stop(self):
        if self.process:
            Path(self.output_var.get(), STOP_FILE_NAME).touch()
            self.stop_btn.config(state="disabled")
            self.status.config(text="Stopping after the current batch, then saving...")

    def finished(self, code):
        launch_training = code == 0 and self.last_run_was_preflight and self.pending_training_after_benchmark
        self.process = None
        self.start_btn.config(state="normal")
        self.benchmark_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.app.player_tab.refresh_models()
        if code != 0:
            self.pending_training_after_benchmark = False
            self.status.config(text=f"Training exited with code {code}. See the log.")
            messagebox.showerror("Action training error", "Training stopped with an error. See the log.")
        elif launch_training:
            self.pending_training_after_benchmark = False
            self.last_run_was_preflight = False
            self.append("Preflight passed. Starting the full run with those settings.")
            self.after(100, lambda: self.start(benchmark=False, preflight=True))


class PlayerTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, style="Panel.TFrame")
        self.app = app
        self.models = {}
        self.loaded_key = None
        self.loaded_stamp = None
        self.loaded_model = None
        self.device = None
        self.dtype = None
        self.current_frame = None
        self.display_frame = None
        self.preview_photo = None
        self.running = False
        self.run_id = 0
        self.worker = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.ui_queue = queue.Queue(maxsize=16)
        self.runtime_settings = {}
        self.keys = set()
        self.keyboard_listener = None
        self.global_keyboard_enabled = True
        self.mouse_listener = None
        self.last_mouse_position = None
        self.pending_mouse_dx = 0.0
        self.pending_mouse_dy = 0.0
        self.pending_zoom = 0.0
        self.right_mouse_held = False
        self.control_lock = threading.Lock()
        self.frame_index = 0
        self.noise_state = None
        self.player_seed = random.randrange(1_000_000_000)
        self.motion_noise_refresh = DEFAULT_MOTION_NOISE_REFRESH
        self.active_noise_multiplier = DEFAULT_ACTIVE_NOISE_MULTIPLIER
        self.motion_profile_summary = "Balanced movement profile"
        self.compare_photo = None
        self._last_settings_signature = None
        self._build()
        self.refresh_models()
        self.root_bindings()
        self.start_keyboard_listener()
        self.start_mouse_listener()
        self.after(30, self.poll_frames)

    def root_bindings(self):
        self.app.root.bind_all("<KeyPress>", self.on_key_down)
        self.app.root.bind_all("<KeyRelease>", self.on_key_up)

    def _build(self):
        main = ttk.Frame(self, style="Panel.TFrame", padding=16)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        controls_host = ttk.Frame(main, style="Panel.TFrame")
        controls_host.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        try:
            controls_host.configure(width=340)
            controls_host.grid_propagate(False)
        except Exception:
            pass
        controls_scroller = scroll_panel(controls_host)
        controls = ttk.Frame(controls_scroller, style="Card.TFrame", padding=12)
        controls.pack(fill="both", expand=True)
        ttk.Label(controls, text="Action Player", style="CardHeading.TLabel").pack(anchor="w", pady=(0, 8))

        self.model_var = tk.StringVar()
        self.model_box = ttk.Combobox(controls, textvariable=self.model_var, state="readonly", width=31)
        self.model_box.pack(fill="x", pady=3)
        self.model_box.bind("<<ComboboxSelected>>", self.on_model_selected)
        ttk.Button(controls, text="Refresh Models", command=self.refresh_models).pack(fill="x", pady=3)

        self.start_image_var = tk.StringVar()
        ttk.Label(controls, text="Starting image", style="Meta.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.start_image_var, style="Field.TEntry").pack(fill="x", pady=3)
        ttk.Button(controls, text="Choose Starting Image", command=self.choose_start_image).pack(fill="x", pady=3)

        self.steps = Field(controls, "Flow steps", "4", ["1", "2", "4", "6", "8", "10", "12"])
        self.steps.pack(fill="x", pady=5)
        self.method = Field(controls, "Denoise method", "Euler", ["Euler", "Heun"])
        self.method.pack(fill="x", pady=5)
        self.guidance = Field(controls, "Action guidance", str(PLAYER_ACTION_GUIDANCE), ["1.0", "1.5", "2.0", "2.5", "3.0", "3.5"])
        self.guidance.pack(fill="x", pady=5)
        self.active_noise = Field(controls, "Active motion noise multiplier", str(DEFAULT_ACTIVE_NOISE_MULTIPLIER), ["0.25", "0.5", "0.75", "1.0"])
        self.active_noise.pack(fill="x", pady=5)
        self.require_right_mouse_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="Camera input requires right-mouse drag",
            variable=self.require_right_mouse_var,
        ).pack(anchor="w", pady=(3, 3))
        ttk.Button(controls, text="Calibrate Movement from Video", command=self.calibrate_motion_from_video).pack(fill="x", pady=(7, 2))
        self.motion_profile_var = tk.StringVar(value=self.motion_profile_summary)
        ttk.Label(controls, textvariable=self.motion_profile_var, style="Meta.TLabel", wraplength=250).pack(anchor="w", pady=(0, 6))

        self.start_btn = ttk.Button(controls, text="Start Action Player", command=self.toggle, style="Accent.TButton")
        self.start_btn.pack(fill="x", pady=(12, 4))
        ttk.Button(controls, text="Reset Frame", command=self.reset_frame).pack(fill="x", pady=3)
        ttk.Button(controls, text="Capture Frame", command=self.capture_frame).pack(fill="x", pady=3)
        ttk.Button(controls, text="Compare Actions (same noise)", command=self.compare_actions).pack(fill="x", pady=3)

        ttk.Label(
            controls,
            text=(
                "Use W/A/S/D or the arrow keys, plus Space. Idle holds the current frame; an active control shows the "
                "model's raw, action-guided next frame. Right-drag controls degree-calibrated yaw/pitch; the wheel "
                "controls zoom. A reference video can calibrate movement frequency, but not teach missing controls. "
                "No display smoothing, anchors, or RGB blending are applied."
            ),
            style="Meta.TLabel",
            wraplength=250,
            justify="left",
        ).pack(fill="x", pady=(10, 0))

        display = ttk.Frame(main, style="Card.TFrame", padding=10)
        display.grid(row=0, column=1, sticky="nsew")
        display.rowconfigure(1, weight=1)
        display.columnconfigure(0, weight=1)

        top = ttk.Frame(display, style="Card.TFrame")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        self.status = ttk.Label(top, text="Choose an action model and starting image.", style="Meta.TLabel")
        self.status.pack(side="left")
        self.fps_var = tk.StringVar(value="AI FPS: — | Direct action guidance")
        ttk.Label(top, textvariable=self.fps_var, style="Meta.TLabel").pack(side="right")

        self.canvas = tk.Canvas(display, bg="#08090c", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _e: self.show_frame(self.display_frame if self.display_frame is not None else self.current_frame))
        self.input_var = tk.StringVar(value="Input: IDLE")
        tk.Label(
            display, textvariable=self.input_var, bg=PANEL, fg=ACCENT,
            font=("Segoe UI", 12, "bold"), anchor="w",
        ).grid(row=2, column=0, sticky="ew", pady=(7, 0))

    def refresh_models(self):
        ACTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        paths = sorted((p for p in ACTION_MODELS_DIR.iterdir() if p.is_dir() and action_model_is_valid(p)), key=lambda p: p.stat().st_mtime, reverse=True)
        self.models = {p.name: p for p in paths}
        self.model_box["values"] = list(self.models)
        if paths and self.model_var.get() not in self.models:
            self.model_var.set(paths[0].name)

    def model_stamp(self, key):
        """Identify the actual checkpoint, including models overwritten in the same folder."""
        path = self.models[key]
        candidates = [
            path / "unet" / "diffusion_pytorch_model.safetensors",
            path / "unet" / "diffusion_pytorch_model.bin",
            path / "action_flow_model_info.json",
        ]
        values = []
        for candidate in candidates:
            if candidate.is_file():
                stat = candidate.stat()
                values.append((str(candidate), stat.st_mtime_ns, stat.st_size))
        return tuple(values)

    def unload_loaded_model(self):
        self.loaded_model = None
        self.loaded_key = None
        self.loaded_stamp = None
        self.noise_state = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def selected_model_description(self, key):
        try:
            info = json.loads((self.models[key] / "action_flow_model_info.json").read_text(encoding="utf-8"))
            enabled = info.get("enabled_action_names")
            controls = (", ".join(ACTION_DISPLAY_NAMES.get(name, name) for name in enabled)
                        if isinstance(enabled, list) else "legacy/all controls")
            return (f"{key} | epoch {info.get('epochs_completed', '?')} | "
                    f"gap {info.get('frame_gap', 1)} | controls {controls} | {self.models[key]}")
        except Exception:
            return f"{key} | {self.models[key]}"

    def on_model_selected(self, _event=None):
        key = self.model_var.get()
        if key not in self.models:
            return
        was_running = self.running
        if was_running:
            self.running = False
            self.run_id += 1
            self.start_btn.config(text="Start Action Player")
        self.unload_loaded_model()
        self.sync_player_to_model()
        # Reset correlated noise so two checkpoints are not compared through the
        # latent state created by the previously loaded checkpoint. Keep the same
        # visible starting/current frame unless the user presses Reset Frame.
        self.status.config(text="Selected model: " + self.selected_model_description(key))
        if was_running:
            self.after(100, self.start)

    def selected_model_info(self, key=None):
        key = key or self.model_var.get()
        if key not in self.models:
            return {}
        try:
            return json.loads((self.models[key] / "action_flow_model_info.json").read_text(encoding="utf-8"))
        except Exception:
            return {}

    def enabled_actions_for_model(self, key=None):
        """Older checkpoints remain permissive; v2 checkpoints expose trained controls."""
        info = self.selected_model_info(key)
        enabled = info.get("enabled_action_names")
        return set(enabled) if isinstance(enabled, list) else set(ACTION_NAMES)

    def validate_inference_settings(self, emit_console=True):
        """Validate only the two user-facing generation choices."""
        info = self.selected_model_info()
        trained_scale = float(info.get("action_input_scale", 1.0))
        frame_gap = int(info.get("frame_gap", 1))
        issues = []
        signature = (
            self.model_var.get(), int(self.steps.get()), self.method.get(),
            float(self.guidance.get()), float(self.active_noise.get()),
        )
        if emit_console and signature != self._last_settings_signature:
            print("\n--- Direct Action-Guided Player ---", flush=True)
            print(f"Model: {self.model_var.get()}", flush=True)
            print(f"Training frame gap: {frame_gap}", flush=True)
            print(f"Trained action scale: {trained_scale:.3f}", flush=True)
            print(f"Action guidance: {float(self.guidance.get()):.2f}", flush=True)
            print(f"Active motion noise multiplier: {float(self.active_noise.get()):.2f}", flush=True)
            print(f"Flow steps / denoise method: {int(self.steps.get())} / {self.method.get()}", flush=True)
            print("Idle is frozen; generated frames are displayed raw.\n", flush=True)
            self._last_settings_signature = signature

        return issues, frame_gap, trained_scale

    def sync_player_to_model(self):
        """The checkpoint's stored action scale is used automatically."""
        self.validate_inference_settings(emit_console=False)

    def choose_start_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if path:
            self.start_image_var.set(path)
            self.reset_frame()

    def calibrate_motion_from_video(self):
        path = filedialog.askopenfilename(
            title="Choose a gameplay reference video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.webm"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.status.config(text="Analyzing reference gameplay motion...")
            self.update_idletasks()
            profile = analyze_reference_motion(path)
            self.motion_noise_refresh = profile["motion_noise_refresh"]
            self.motion_profile_summary = (
                f"Reference profile: {profile['motion_noise_refresh']:.3f} movement refresh "
                f"from {profile['sampled_transitions']} frame transitions"
            )
            self.motion_profile_var.set(self.motion_profile_summary)
            self.status.config(text="Reference movement profile ready.")
            messagebox.showinfo(
                "Movement profile ready",
                f"Measured {profile['sampled_transitions']} transitions at source FPS {profile['source_fps']:.1f}.\n\n"
                f"Median frame change: {profile['median_difference']:.3f}\n"
                f"Median optical flow: {profile['median_flow']:.3f}\n"
                f"Automatic movement refresh: {profile['motion_noise_refresh']:.3f}\n\n"
                "This changes temporal variation during active inputs only. It does not replace action training.",
            )
        except Exception as exc:
            self.status.config(text="Could not analyze reference video.")
            messagebox.showerror("Reference video analysis failed", str(exc))

    def get_frame_size(self):
        key = self.model_var.get()
        if key not in self.models:
            return 144, 256
        info = json.loads((self.models[key] / "action_flow_model_info.json").read_text(encoding="utf-8"))
        if info.get("width") and info.get("height"):
            return int(info["height"]), int(info["width"])
        resolution = info.get("resolution", "256x144")
        if isinstance(resolution, str) and "x" in resolution.lower():
            return parse_frame_size(resolution)
        # Compatibility for older square action checkpoints.
        side = int(resolution)
        return side, side

    def reset_frame(self):
        try:
            from PIL import Image, ImageOps
            path = Path(self.start_image_var.get())
            if not path.is_file():
                raise ValueError("Choose a starting screenshot from the same Roblox game.")
            height, width = self.get_frame_size()
            with Image.open(path) as image:
                # Match training exactly: aspect-preserving resize plus centered crop.
                canvas = ImageOps.fit(
                    image.convert("RGB"), (width, height),
                    method=Image.Resampling.LANCZOS, centering=(0.5, 0.5),
                )
                self.current_frame = canvas
                self.display_frame = canvas.copy()
            self.frame_index = 0
            self.noise_state = None
            self.player_seed = random.randrange(1_000_000_000)
            self.show_frame(self.display_frame)
            self.status.config(text="Frame reset. Press Start Action Player.")
        except Exception as exc:
            messagebox.showerror("Could not reset frame", str(exc))

    def start_mouse_listener(self):
        try:
            from pynput import mouse
        except ImportError:
            return

        def on_move(x, y):
            if not self.running:
                self.last_mouse_position = (x, y)
                return
            with self.control_lock:
                if self.last_mouse_position is not None:
                    self.pending_mouse_dx += float(x - self.last_mouse_position[0])
                    self.pending_mouse_dy += float(y - self.last_mouse_position[1])
                self.last_mouse_position = (x, y)

        def on_scroll(_x, _y, _dx, dy):
            if not self.running:
                return
            with self.control_lock:
                self.pending_zoom += float(dy)

        def on_click(_x, _y, button, pressed):
            if button == mouse.Button.right:
                with self.control_lock:
                    self.right_mouse_held = bool(pressed)

        self.mouse_listener = mouse.Listener(on_move=on_move, on_scroll=on_scroll, on_click=on_click)
        self.mouse_listener.daemon = True
        self.mouse_listener.start()

    def start_keyboard_listener(self):
        """Let the player receive controls even after focus moves to its preview."""
        try:
            from pynput import keyboard
        except ImportError:
            return

        def name_for(key):
            try:
                char = key.char.lower()
            except Exception:
                char = None
            if char in {"w", "a", "s", "d"}:
                return char
            arrow_actions = {
                keyboard.Key.up: "w", keyboard.Key.left: "a",
                keyboard.Key.down: "s", keyboard.Key.right: "d",
            }
            if key in arrow_actions:
                return arrow_actions[key]
            if key == keyboard.Key.space:
                return "space"
            return None

        def on_press(key):
            if not self.running or not self.global_keyboard_enabled:
                return
            name = name_for(key)
            if name:
                with self.control_lock:
                    self.keys.add(name)

        def on_release(key):
            name = name_for(key)
            if name:
                with self.control_lock:
                    self.keys.discard(name)

        self.keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.keyboard_listener.daemon = True
        self.keyboard_listener.start()

    def snapshot_runtime_settings(self):
        """Publish the two UI choices plus checkpoint-owned action scaling."""
        model_key = self.model_var.get()
        info = self.selected_model_info(model_key)
        if self.runtime_settings.get("model_key") == model_key:
            action_scale = float(self.runtime_settings.get("action_input_scale", 1.0))
        else:
            action_scale = float(info.get("action_input_scale", 1.0))
        return {
            "model_key": model_key,
            "seed": self.player_seed,
            "steps": max(1, int(self.steps.get())),
            "method": self.method.get(),
            "action_input_scale": action_scale,
            "motion_noise_refresh": self.motion_noise_refresh,
            "active_noise_multiplier": max(0.0, min(1.0, float(self.active_noise.get()))),
            "guidance": max(0.0, float(self.guidance.get())),
            "camera_encoding": str(info.get("camera_encoding", "legacy_pixels")),
            "yaw_counts_per_360_degrees": float(info.get("yaw_counts_per_360_degrees") or 2400.0),
            "pitch_counts_per_180_degrees": float(info.get("pitch_counts_per_180_degrees") or 1200.0),
            "max_yaw_degrees_per_frame": float(info.get("max_yaw_degrees_per_frame", 45.0)),
            "max_pitch_degrees_per_frame": float(info.get("max_pitch_degrees_per_frame", 30.0)),
            "require_right_mouse": bool(self.require_right_mouse_var.get()),
        }

    def publish_runtime_settings(self):
        """Atomically publish a fresh settings snapshot to the worker."""
        self.runtime_settings = self.snapshot_runtime_settings()

    def action_snapshot(self, settings, reset_mouse=True):
        with self.control_lock:
            camera_allowed = not settings.get("require_right_mouse", True) or self.right_mouse_held
            if str(settings.get("camera_encoding", "legacy_pixels")).lower() == "relative_degrees_v1":
                yaw_counts = max(1.0, float(settings.get("yaw_counts_per_360_degrees", 2400.0)))
                pitch_counts = max(1.0, float(settings.get("pitch_counts_per_180_degrees", 1200.0)))
                max_yaw = max(1.0, float(settings.get("max_yaw_degrees_per_frame", 45.0)))
                max_pitch = max(1.0, float(settings.get("max_pitch_degrees_per_frame", 30.0)))
                yaw_degrees = self.pending_mouse_dx * (360.0 / yaw_counts) if camera_allowed else 0.0
                pitch_degrees = -self.pending_mouse_dy * (180.0 / pitch_counts) if camera_allowed else 0.0
                mouse_dx = max(-1.0, min(1.0, yaw_degrees / max_yaw))
                mouse_dy = max(-1.0, min(1.0, pitch_degrees / max_pitch))
            else:
                mouse_dx = max(-1.0, min(1.0, self.pending_mouse_dx / PLAYER_MOUSE_SCALE)) if camera_allowed else 0.0
                mouse_dy = max(-1.0, min(1.0, self.pending_mouse_dy / PLAYER_MOUSE_SCALE)) if camera_allowed else 0.0
            zoom = max(-1.0, min(1.0, self.pending_zoom / PLAYER_ZOOM_SCALE))
            if reset_mouse:
                self.pending_mouse_dx = 0.0
                self.pending_mouse_dy = 0.0
                self.pending_zoom = 0.0
            keys = set(self.keys)
        values = [
            1.0 if "w" in keys else 0.0,
            1.0 if "a" in keys else 0.0,
            1.0 if "s" in keys else 0.0,
            1.0 if "d" in keys else 0.0,
            1.0 if "space" in keys else 0.0,
            mouse_dx,
            mouse_dy,
            zoom,
        ]
        enabled = self.enabled_actions_for_model(settings.get("model_key"))
        values = [value if ACTION_NAMES[index] in enabled else 0.0
                  for index, value in enumerate(values)]
        # Ignore tiny mouse jitter so idle genuinely holds the current frame.
        moving = any(v > 0.5 for v in values[:5]) or abs(mouse_dx) > 0.02 or abs(mouse_dy) > 0.02 or abs(zoom) > 0.02
        if not moving:
            values[5:] = [0.0, 0.0, 0.0]
        return values, moving

    def blend_noise(self, previous_shape, refresh, generator):
        """Return temporally correlated *noise* without ever feeding generated RGB back as noise."""
        import torch
        refresh = max(0.0, min(1.0, float(refresh)))
        if self.noise_state is None or tuple(self.noise_state.shape) != tuple(previous_shape):
            self.noise_state = torch.randn(previous_shape, generator=generator, device=self.device, dtype=self.dtype)
        elif refresh > 0.0:
            fresh = torch.randn(previous_shape, generator=generator, device=self.device, dtype=self.dtype)
            # Variance-preserving interpolation. The old linear mix followed by a raw
            # standard-deviation division could amplify biased/non-noise tensors.
            keep = math.sqrt(max(0.0, 1.0 - refresh * refresh))
            mixed = self.noise_state * keep + fresh * refresh
            dims = tuple(range(1, mixed.ndim))
            mean = mixed.float().mean(dim=dims, keepdim=True)
            std = mixed.float().std(dim=dims, keepdim=True).clamp_min(1e-6)
            self.noise_state = ((mixed.float() - mean) / std).to(dtype=self.dtype).clamp(-4.0, 4.0)
        return self.noise_state.detach().clone()
    def on_key_down(self, event):
        key = KEY_TO_ACTION.get(event.keysym.lower())
        if key:
            with self.control_lock:
                self.keys.add(key)
            self.update_input()

    def on_key_up(self, event):
        key = KEY_TO_ACTION.get(event.keysym.lower())
        with self.control_lock:
            if key:
                self.keys.discard(key)
        self.update_input()

    def update_input(self):
        with self.control_lock:
            keys = set(self.keys)
        labels = [name.upper() if name != "space" else "SPACE" for name in ["w", "a", "s", "d", "space"] if name in keys]
        self.input_var.set("Input: " + (" + ".join(labels) if labels else "IDLE"))

    def toggle(self):
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self):
        try:
            if self.model_var.get() not in self.models:
                raise ValueError("Select a trained action model.")
            if self.current_frame is None:
                self.reset_frame()
            if self.current_frame is None:
                raise ValueError("Choose a starting image.")
            issues, frame_gap, trained_scale = self.validate_inference_settings(emit_console=True)
            self.publish_runtime_settings()
            self.run_id += 1
            run_id = self.run_id
            self.running = True
            self.noise_state = None
            if self.display_frame is None and self.current_frame is not None:
                self.display_frame = self.current_frame.copy()
            self.start_btn.config(text="Stop Action Player")
            self.status.config(text="Loading action model...")
            self.worker = threading.Thread(target=self.generation_loop, args=(run_id,), daemon=True)
            self.worker.start()
        except Exception as exc:
            messagebox.showerror("Cannot start player", str(exc))

    def stop(self):
        self.running = False
        self.run_id += 1
        self.start_btn.config(text="Start Action Player")
        self.status.config(text="Action player paused.")

    def generation_loop(self, run_id):
        try:
            import torch
            source, _ = load_source_module()
            initial_settings = dict(self.runtime_settings)
            key = initial_settings["model_key"]
            selected_stamp = self.model_stamp(key)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            if (self.loaded_key != key or self.loaded_model is None or
                    self.loaded_stamp != selected_stamp):
                self.loaded_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.loaded_model = load_action_unet(self.models[key], self.device, self.dtype)
                self.loaded_key = key
                self.loaded_stamp = selected_stamp
                self.noise_state = None
            generator = torch.Generator(device=self.device).manual_seed(initial_settings["seed"])
            description = self.selected_model_description(key)
            self.ui_queue.put(("status", "Running: " + description))

            while self.running and run_id == self.run_id:
                started = time.perf_counter()
                settings = dict(self.runtime_settings)
                previous = source.pil_to_normalized_tensor(self.current_frame, self.device, self.dtype)
                action_values, moving = self.action_snapshot(settings, reset_mouse=True)
                input_text = (
                    "Guided input: " + (" + ".join([name.upper() for name, value in zip(ACTION_NAMES[:5], action_values[:5]) if value > 0.5]) or "IDLE")
                    + f"  |  yaw {action_values[5]:+.2f}  pitch {action_values[6]:+.2f}  zoom {action_values[7]:+.2f}"
                    + f"  |  guidance {settings['guidance']:.1f}x"
                    + f"  |  motion {settings['motion_noise_refresh'] * settings['active_noise_multiplier']:.3f}"
                )

                if not moving:
                    model_result = self.current_frame.copy()
                    time.sleep(0.02)
                else:
                    # Persistent same-noise inference makes action differences visible;
                    # only the previous frame and pressed control change between steps.
                    active_refresh = settings["motion_noise_refresh"] * settings["active_noise_multiplier"]
                    initial_noise = self.blend_noise(previous.shape, active_refresh, generator)
                    action = torch.tensor([action_values], device=self.device, dtype=self.dtype)
                    generated = sample_guided_action_frame(
                        self.loaded_model,
                        previous,
                        action,
                        settings["steps"],
                        self.device,
                        self.dtype,
                        generator,
                        method=settings["method"],
                        initial_noise=initial_noise,
                        action_input_scale=settings["action_input_scale"],
                        guidance=settings["guidance"],
                    )
                    model_result = source.tensor_to_pil(generated[0])

                self.current_frame = model_result
                self.display_frame = model_result.copy()
                self.frame_index += 1
                elapsed = max(1e-6, time.perf_counter() - started)
                try:
                    while self.frame_queue.qsize() > 0:
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait((self.display_frame.copy(), 1.0 / elapsed, input_text, "Direct guided"))
                except queue.Full:
                    pass
        except Exception:
            error = traceback.format_exc()
            if run_id == self.run_id:
                self.running = False
                try:
                    self.ui_queue.put_nowait(("error", error))
                except queue.Full:
                    pass

    def compare_actions(self):
        try:
            import torch
            from PIL import Image, ImageDraw, ImageTk
            if self.model_var.get() not in self.models:
                raise ValueError("Select a trained action model.")
            if self.current_frame is None:
                self.reset_frame()
            if self.current_frame is None:
                raise ValueError("Choose a starting image.")
            self.validate_inference_settings(emit_console=True)
            source, _ = load_source_module()
            if self.device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            key = self.model_var.get()
            selected_stamp = self.model_stamp(key)
            if (self.loaded_key != key or self.loaded_model is None or
                    self.loaded_stamp != selected_stamp):
                self.loaded_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.loaded_model = load_action_unet(self.models[key], self.device, self.dtype)
                self.loaded_key = key
                self.loaded_stamp = selected_stamp
                self.noise_state = None
            previous = source.pil_to_normalized_tensor(self.current_frame, self.device, self.dtype)
            info = self.selected_model_info(key)
            enabled = self.enabled_actions_for_model(key)
            compare_actions = [("Idle", [0.0] * ACTION_DIM)]
            for index, name in enumerate(BINARY_ACTION_NAMES):
                if name in enabled:
                    values = [0.0] * ACTION_DIM
                    values[index] = 1.0
                    compare_actions.append((ACTION_DISPLAY_NAMES.get(name, name.upper()), values))
            for label, index, value, control_name in (
                ("Yaw Left", 5, -0.5, "mouse_dx"),
                ("Yaw Right", 5, 0.5, "mouse_dx"),
                ("Pitch Up", 6, 0.5, "mouse_dy"),
                ("Pitch Down", 6, -0.5, "mouse_dy"),
            ):
                if control_name in enabled:
                    values = [0.0] * ACTION_DIM
                    values[index] = value
                    compare_actions.append((label, values))
            for profile in info.get("observed_binary_actions", []):
                active = [BINARY_ACTION_NAMES[i] for i, value in enumerate(profile[:5]) if float(value) > 0.5]
                if len(active) < 2 or any(name not in enabled for name in active):
                    continue
                values = [0.0] * ACTION_DIM
                values[:5] = [float(value) for value in profile[:5]]
                label = "+".join(ACTION_DISPLAY_NAMES.get(name, name.upper()) for name in active)
                if label not in {existing_label for existing_label, _values in compare_actions}:
                    compare_actions.append((label, values))
                if len(compare_actions) >= 9:
                    break
            gen = torch.Generator(device=self.device).manual_seed(self.player_seed)
            base_noise = self.blend_noise(previous.shape, 0.0, gen)
            action_scale = float(self.selected_model_info(key).get("action_input_scale", 1.0))
            height, width = self.get_frame_size()
            columns = 3
            rows = math.ceil(len(compare_actions) / columns)
            panel = Image.new("RGB", (width * columns, (height + 24) * rows), (20, 20, 24))
            draw = ImageDraw.Draw(panel)
            for i, (label, values) in enumerate(compare_actions):
                action = torch.tensor([values], device=self.device, dtype=self.dtype)
                if label == "Idle":
                    generated = sample_action_frame(
                        self.loaded_model, previous, action, int(self.steps.get()),
                        self.device, self.dtype, gen, method=self.method.get(),
                        initial_noise=base_noise.clone(),
                        action_input_scale=action_scale,
                    )
                else:
                    generated = sample_guided_action_frame(
                        self.loaded_model, previous, action, int(self.steps.get()),
                        self.device, self.dtype, gen, method=self.method.get(),
                        initial_noise=base_noise.clone(),
                        action_input_scale=action_scale,
                        guidance=max(0.0, float(self.guidance.get())),
                    )
                image = source.tensor_to_pil(generated[0].clamp(-1, 1))
                column = i % columns
                row = i // columns
                x = column * width
                y = row * (height + 24)
                panel.paste(image, (x, y + 24))
                draw.text((x + 5, y + 5), label, fill=(240, 240, 240))
            ACTION_GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
            out = ACTION_GENERATIONS_DIR / f"action_compare_{time.strftime('%Y%m%d_%H%M%S')}.png"
            panel.save(out)
            popup = tk.Toplevel(self)
            popup.title("Same-noise Action Comparison")
            preview = panel.copy()
            preview.thumbnail((min(1600, panel.width), 900), Image.Resampling.NEAREST)
            self.compare_photo = ImageTk.PhotoImage(preview)
            label = tk.Label(popup, image=self.compare_photo, bg="#111318")
            label.pack(fill="both", expand=True)
            tk.Label(popup, text=f"Saved: {out}", bg="#111318", fg="#eef1f7").pack(fill="x")
            self.status.config(text=f"Saved same-noise comparison: {out.name}")
        except Exception as exc:
            messagebox.showerror("Action comparison failed", str(exc))

    def poll_frames(self):
        if self.running:
            try:
                self.publish_runtime_settings()
            except (TypeError, ValueError, tk.TclError):
                # Keep the last valid snapshot while the user is editing a field.
                pass
        try:
            while True:
                kind, value = self.ui_queue.get_nowait()
                if kind == "status":
                    self.status.config(text=value)
                elif kind == "error":
                    self.failed(value)
        except queue.Empty:
            pass
        try:
            while True:
                frame, fps, input_text, mode = self.frame_queue.get_nowait()
                self.show_frame(frame)
                self.input_var.set(input_text)
                self.fps_var.set(f"AI FPS: {fps:.2f}  |  Display: {mode}")
        except queue.Empty:
            pass
        self.after(30, self.poll_frames)

    def show_frame(self, frame):
        if frame is None:
            return
        from PIL import Image, ImageTk
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        scale = min(cw / frame.width, ch / frame.height)
        display_width = max(1, round(frame.width * scale))
        display_height = max(1, round(frame.height * scale))
        preview = frame.resize((display_width, display_height), Image.Resampling.NEAREST)
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.preview_photo, anchor="center")

    def capture_frame(self):
        if self.current_frame is None:
            return
        ACTION_GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = ACTION_GENERATIONS_DIR / f"action_frame_{time.strftime('%Y%m%d_%H%M%S')}.png"
        self.current_frame.save(path)
        self.status.config(text=f"Captured {path.name}")

    def failed(self, error):
        self.start_btn.config(text="Start Action Player")
        self.status.config(text="Action player failed.")
        messagebox.showerror("Action player error", error[-2200:])


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Game Action Flow Trainer + Controls")
        self.root.geometry("1180x820")
        self.root.minsize(960, 680)
        self.root.configure(bg=BG)
        self._style()

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)
        self.train_tab = TrainTab(notebook, self)
        self.player_tab = PlayerTab(notebook, self)
        notebook.add(self.train_tab, text="Action Training")
        notebook.add(self.player_tab, text="Action Player")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        try:
            self.train_tab.save_training_settings(silent=True)
            try:
                self.player_tab.running = False
                for listener in (self.player_tab.keyboard_listener, self.player_tab.mouse_listener):
                    if listener is not None:
                        listener.stop()
            except Exception:
                pass
        finally:
            self.root.destroy()

    def _style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("Panel.TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("Heading.TLabel", background=BG, foreground=FG, font=("Segoe UI", 17, "bold"))
        style.configure("Body.TLabel", background=BG, foreground=DIM)
        style.configure("CardHeading.TLabel", background=PANEL, foreground=FG, font=("Segoe UI", 12, "bold"))
        style.configure("Meta.TLabel", background=PANEL, foreground=DIM)
        style.configure("Field.TEntry", fieldbackground=FIELD, foreground=FG, insertcolor=FG)
        style.configure("Accent.TButton", background=ACCENT, foreground="#10131a", font=("Segoe UI", 10, "bold"))
        style.configure("TCombobox", fieldbackground=FIELD, foreground=FG)
        style.configure("TCheckbutton", background=PANEL, foreground=FG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-worker", action="store_true")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--continue-action-model", default="")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-name", default="Roblox Action Flow")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--resolution", default="256x144")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--condition-noise", type=float, default=0.03)
    parser.add_argument("--temporal-loss-weight", type=float, default=0.1)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--metrics-every", type=int, default=25)
    parser.add_argument("--frame-gap", type=int, default=3)
    parser.add_argument("--action-aggregation", choices=["window", "mean", "last"], default="window")
    parser.add_argument("--action-input-scale", type=float, default=8.0)
    parser.add_argument("--neutral-action-dropout", type=float, default=0.15)
    parser.add_argument("--motion-loss-weight", type=float, default=2.0)
    parser.add_argument("--action-contrast-weight", type=float, default=0.35)
    parser.add_argument("--action-contrast-margin", type=float, default=0.02)
    parser.add_argument("--contrast-every", type=int, default=4)
    parser.add_argument("--contrast-samples", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--chunk-mode", choices=["balanced", "random", "sequential"], default="balanced")
    parser.add_argument("--chunk-offset", type=int, default=0)
    parser.add_argument("--chunk-seed", type=int, default=1234)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-batches", type=int, default=8)
    parser.add_argument("--recovery-minutes", type=int, default=30)
    parser.add_argument("--balance-actions", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--preview-every", type=int, default=5)
    parser.add_argument("--preview-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", choices=["fp16", "no"], default="fp16")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ACTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ACTION_GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    if args.train_worker:
        try:
            training_worker(args)
        except Exception:
            traceback.print_exc()
            raise
        return

    root = tk.Tk()
    try:
        App(root)
    except Exception as exc:
        root.withdraw()
        messagebox.showerror("Could not start app", f"{exc}\n\n{traceback.format_exc()[-2000:]}")
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
