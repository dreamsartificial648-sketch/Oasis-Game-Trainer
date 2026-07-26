"""Experimental video-to-action pseudo-labeller.

This is deliberately a *review-first* tool.  It does not claim to recover a
player's keyboard state from pixels.  Instead it uses motion between adjacent
video frames to create tentative W/A/S/D labels, then writes the same frame and
actions.jsonl layout used by roblox_action_dataset_recorder.py.  Treat its
output as candidates to inspect before adding it to a training dataset.

Requires: opencv-python, Pillow (both are already in requirements.txt).
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as exc:  # Gives a useful error instead of an obscure crash.
    raise SystemExit("Install dependencies first: pip install opencv-python Pillow") from exc


APP_DIR = Path(__file__).resolve().parent
BG, PANEL, FG, DIM, ACCENT = "#10131a", "#1a202b", "#f4f6fb", "#aeb8c8", "#6da8ff"


@dataclass
class Prediction:
    label: str
    confidence: float
    motion_x: float
    motion_y: float
    expansion: float


def resize_to_16_9(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Centre-crop a video frame to the recorder's 16:9 output convention."""
    source_h, source_w = frame.shape[:2]
    source_ratio, target_ratio = source_w / source_h, width / height
    if source_ratio > target_ratio:
        crop_w = round(source_h * target_ratio)
        left = (source_w - crop_w) // 2
        frame = frame[:, left:left + crop_w]
    elif source_ratio < target_ratio:
        crop_h = round(source_w / target_ratio)
        top = (source_h - crop_h) // 2
        frame = frame[top:top + crop_h, :]
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


class MotionLabeler:
    """Converts dense optical flow into conservative, explainable pseudo-labels."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def predict(self, previous: np.ndarray | None, current: np.ndarray) -> Prediction:
        if previous is None:
            return Prediction("IDLE", 0.0, 0.0, 0.0, 0.0)

        # A modest image size keeps this usable on long recordings and removes UI/HUD noise.
        small_size = (160, 90)
        before = cv2.cvtColor(cv2.resize(previous, small_size), cv2.COLOR_BGR2GRAY)
        after = cv2.cvtColor(cv2.resize(current, small_size), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            before, after, None, 0.5, 3, 21, 3, 5, 1.2, 0,
        )

        h, w = before.shape
        # Ignore 8% around the edge: HUDs and video borders otherwise dominate the motion.
        margin_x, margin_y = int(w * 0.08), int(h * 0.08)
        inner = flow[margin_y:h - margin_y, margin_x:w - margin_x]
        motion_x = float(np.median(inner[..., 0]))
        motion_y = float(np.median(inner[..., 1]))

        yy, xx = np.mgrid[margin_y:h - margin_y, margin_x:w - margin_x]
        xx = xx - (w - 1) / 2.0
        yy = yy - (h - 1) / 2.0
        radius = np.sqrt(xx * xx + yy * yy) + 1e-6
        radial = (inner[..., 0] * xx + inner[..., 1] * yy) / radius
        expansion = float(np.median(radial))
        magnitude = math.hypot(motion_x, motion_y)

        # First-person forward movement usually expands the scene, backwards contracts it.
        # Side labels describe the *apparent scene flow*, so they are intentionally marked
        # tentative; camera movement and third-person games may reverse this relationship.
        score_x, score_y, score_z = abs(motion_x), abs(motion_y), abs(expansion)
        strongest = max(score_x, score_y, score_z)
        if strongest < self.threshold:
            return Prediction("IDLE", max(0.0, 1.0 - strongest / max(self.threshold, 1e-6)), motion_x, motion_y, expansion)
        if score_z >= score_x and score_z >= score_y:
            label = "W" if expansion > 0 else "S"
        elif score_x >= score_y:
            # With an image moving left, the camera/player generally moved right.
            label = "D" if motion_x < 0 else "A"
        else:
            # Vertical flow is not a recorder movement control; keep it as IDLE.
            label = "IDLE"
        confidence = min(0.99, strongest / (strongest + self.threshold))
        return Prediction(label, confidence, motion_x, motion_y, expansion)


class VideoToActionLabelsApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Video to Action Labels — Experimental")
        root.configure(bg=BG)
        root.minsize(800, 610)
        self.video_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(APP_DIR / "action_predictions"))
        self.fps_var = tk.StringVar(value="5")
        self.resolution_var = tk.StringVar(value="256x144")
        self.threshold_var = tk.StringVar(value="0.45")
        self.minimum_confidence_var = tk.StringVar(value="0.55")
        self.keep_idle_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Choose a gameplay video to begin.")
        self.progress_var = tk.StringVar(value="0 frames written")
        self.running = False
        self.cancel_requested = False
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.preview_photo = None
        self._build_ui()
        root.after(80, self._drain_events)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=PANEL, foreground=FG)
        style.configure("Muted.TLabel", background=PANEL, foreground=DIM)
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Video to Action Labels", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Experimental pseudo-labels from visible motion. Review before training.", foreground=DIM).pack(anchor="w", pady=(2, 14))
        card = ttk.Frame(outer, style="Card.TFrame", padding=14)
        card.pack(fill="x")
        self._path_row(card, "Gameplay video", self.video_var, self._browse_video, "Video")
        self._path_row(card, "Prediction folder", self.output_var, self._browse_output, "Folder")
        fields = ttk.Frame(card, style="Card.TFrame")
        fields.pack(fill="x", pady=(10, 0))
        for text, variable, width in (("Sample FPS", self.fps_var, 8), ("Frame size", self.resolution_var, 10), ("Motion threshold", self.threshold_var, 10), ("Min. confidence", self.minimum_confidence_var, 10)):
            group = ttk.Frame(fields, style="Card.TFrame")
            group.pack(side="left", padx=(0, 14))
            ttk.Label(group, text=text, style="Muted.TLabel").pack(anchor="w")
            ttk.Entry(group, textvariable=variable, width=width).pack(anchor="w", pady=(3, 0))
        ttk.Checkbutton(card, text="Keep IDLE / low-confidence frames for review", variable=self.keep_idle_var).pack(anchor="w", pady=(12, 0))
        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=12)
        self.start_button = ttk.Button(actions, text="Create Predictions", style="Accent.TButton", command=self.start)
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=8)
        ttk.Label(actions, textvariable=self.progress_var, foreground=DIM).pack(side="right")
        preview_card = ttk.Frame(outer, style="Card.TFrame", padding=12)
        preview_card.pack(fill="both", expand=True)
        ttk.Label(preview_card, textvariable=self.status_var, style="Card.TLabel", wraplength=730).pack(anchor="w")
        self.preview = ttk.Label(preview_card, text="Preview will appear here", style="Muted.TLabel", anchor="center")
        self.preview.pack(fill="both", expand=True, pady=(10, 0))

    def _path_row(self, parent, label, variable, command, button_text) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, style="Muted.TLabel", width=18).pack(side="left")
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True, padx=(0, 7))
        ttk.Button(row, text=f"Choose {button_text}", command=command).pack(side="right")

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(title="Choose gameplay video", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")])
        if path:
            self.video_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Choose prediction folder")
        if path:
            self.output_var.set(path)

    def start(self) -> None:
        video = Path(self.video_var.get()).expanduser()
        if not video.is_file():
            messagebox.showerror("Video not found", "Choose an existing gameplay video.")
            return
        try:
            sample_fps = float(self.fps_var.get())
            width, height = (int(value.strip()) for value in self.resolution_var.get().lower().split("x", 1))
            threshold = float(self.threshold_var.get())
            minimum_confidence = float(self.minimum_confidence_var.get())
            if sample_fps <= 0 or width <= 0 or height <= 0 or width * 9 != height * 16 or threshold <= 0 or not 0 <= minimum_confidence <= 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid settings", "Use a positive FPS and threshold, exact 16:9 frame size, and confidence from 0 to 1.")
            return
        self.running, self.cancel_requested = True, False
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status_var.set("Reading video and estimating motion...")
        args = (video, Path(self.output_var.get()).expanduser(), sample_fps, width, height, threshold, minimum_confidence, self.keep_idle_var.get())
        threading.Thread(target=self._process_video, args=args, daemon=True).start()

    def cancel(self) -> None:
        self.cancel_requested = True
        self.status_var.set("Cancelling after the current frame...")

    def _process_video(self, video: Path, output: Path, sample_fps: float, width: int, height: int, threshold: float, minimum_confidence: float, keep_idle: bool) -> None:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            self.events.put(("error", "OpenCV could not open this video."))
            return
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_step = max(1, round(source_fps / sample_fps))
        actual_fps = source_fps / frame_step
        root, frames = output, output / "frames"
        try:
            frames.mkdir(parents=True, exist_ok=True)
            existing = list(frames.glob("frame_*.png"))
            try:
                next_index = max((int(path.stem.split("_")[1]) for path in existing), default=-1) + 1
            except (IndexError, ValueError):
                next_index = len(existing)
            written, labeler, previous, source_index = 0, MotionLabeler(threshold), None, 0
            session_id = uuid.uuid4().hex
            started = time.strftime("%Y-%m-%d %H:%M:%S")
            metadata_path = root / "actions.jsonl"
            with metadata_path.open("a", encoding="utf-8") as metadata:
                while not self.cancel_requested:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if source_index % frame_step:
                        source_index += 1
                        continue
                    timestamp = source_index / source_fps
                    resized = resize_to_16_9(frame, width, height)
                    prediction = labeler.predict(previous, resized)
                    previous = resized.copy()
                    source_index += 1
                    should_write = keep_idle or (prediction.label != "IDLE" and prediction.confidence >= minimum_confidence)
                    if should_write:
                        filename = f"frame_{next_index:08d}_{prediction.label}.png"
                        cv2.imwrite(str(frames / filename), resized, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                        record = {
                            "session_id": session_id, "session_started_at": started, "frame_index": next_index,
                            "filename": filename, "timestamp_seconds": round(timestamp, 6),
                            "w": int(prediction.label == "W"), "a": int(prediction.label == "A"),
                            "s": int(prediction.label == "S"), "d": int(prediction.label == "D"), "jump": 0,
                            "mouse_dx": 0.0, "mouse_dy": 0.0, "mouse_raw_dx": 0.0, "mouse_raw_dy": 0.0,
                            "camera_encoding": "unavailable_from_video", "camera_yaw_delta_degrees": 0.0,
                            "camera_pitch_delta_degrees": 0.0, "camera_active": 0, "right_mouse": 0, "zoom": 0.0,
                            "move_x": int(prediction.label == "D") - int(prediction.label == "A"),
                            "move_y": int(prediction.label == "W") - int(prediction.label == "S"),
                            "prediction_confidence": round(prediction.confidence, 6),
                            "optical_flow_x": round(prediction.motion_x, 6), "optical_flow_y": round(prediction.motion_y, 6),
                            "optical_flow_expansion": round(prediction.expansion, 6), "label_source": "experimental_optical_flow",
                        }
                        metadata.write(json.dumps(record) + "\n")
                        next_index += 1
                        written += 1
                    if written and (written % 5 == 0 or written == 1):
                        progress = f"{written} frames written" + (f" / about {total_frames // frame_step}" if total_frames else "")
                        self.events.put(("progress", (progress, resized.copy(), prediction)))
            info = {
                "format_version": 3, "created_at": started, "source_video": str(video),
                "capture_fps": round(actual_fps, 6), "output_resolution": f"{width}x{height}",
                "output_width": width, "output_height": height, "aspect_ratio": "16:9",
                "action_dimensions": ["w", "a", "s", "d", "jump", "mouse_dx", "mouse_dy", "zoom", "move_x", "move_y"],
                "label_source": "experimental optical-flow pseudo-labels; review before training",
                "motion_threshold": threshold, "minimum_confidence_for_non_idle_export": minimum_confidence,
                "transition_alignment": "Row t is a tentative label inferred from visible motion between sampled frames t-1 and t.",
            }
            (root / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            self.events.put(("done", f"{'Cancelled' if self.cancel_requested else 'Finished'}: {written} frames saved to\n{root}"))
        except Exception as exc:
            self.events.put(("error", f"Could not create predictions: {exc}"))
        finally:
            cap.release()

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    text, frame, pred = payload
                    self.progress_var.set(text)
                    self.status_var.set(f"Latest tentative label: {pred.label} ({pred.confidence:.0%}) | flow x={pred.motion_x:.2f}, expansion={pred.expansion:.2f}")
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb)
                    image.thumbnail((720, 405))
                    self.preview_photo = ImageTk.PhotoImage(image)
                    self.preview.configure(image=self.preview_photo, text="")
                elif kind == "done":
                    self._finish(str(payload), False)
                elif kind == "error":
                    self._finish(str(payload), True)
        except queue.Empty:
            pass
        self.root.after(80, self._drain_events)

    def _finish(self, text: str, is_error: bool) -> None:
        self.running = False
        self.start_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.status_var.set(text)
        if is_error:
            messagebox.showerror("Video to Action Labels", text)
        else:
            messagebox.showinfo("Video to Action Labels", text + "\n\nReview the labels before using this as training data.")


if __name__ == "__main__":
    app_root = tk.Tk()
    VideoToActionLabelsApp(app_root)
    app_root.mainloop()
