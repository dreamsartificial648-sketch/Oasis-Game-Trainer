from __future__ import annotations

import importlib.util
import json
import math
import queue
import random
import threading
import time
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

APP_DIR = Path(__file__).resolve().parent
MODEL_DROP_DIR = APP_DIR / "1 VIDEO MODEL IN HERE"
CAPTURE_DIR = APP_DIR / "oasis_captures"
SOURCE_APP_NAMES = ["flow_matching_app(12).py", "flow_matching_app.py"]

BG = "#15171d"
PANEL = "#20232b"
FIELD = "#2b2f39"
FG = "#eef1f7"
DIM = "#aab0be"
ACCENT = "#6f98ff"


def find_source_app() -> Path:
    for name in SOURCE_APP_NAMES:
        candidate = APP_DIR / name
        if candidate.is_file():
            return candidate
    matches = sorted(APP_DIR.glob("flow_matching_app*.py"), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Place this app beside flow_matching_app(12).py (or another flow_matching_app*.py file)."
    )


def load_flow_module():
    source = find_source_app()
    spec = importlib.util.spec_from_file_location("flow_oasis_source", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper functions from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def is_video_model(path: Path) -> bool:
    try:
        info = json.loads((path / "flow_video_model_info.json").read_text(encoding="utf-8"))
        return (
            info.get("model_type") == "autoregressive_rectified_flow_video"
            and (path / "unet" / "config.json").is_file()
        )
    except Exception:
        return False


def discover_models() -> list[Path]:
    MODEL_DROP_DIR.mkdir(parents=True, exist_ok=True)
    found: list[Path] = []
    if is_video_model(MODEL_DROP_DIR):
        found.append(MODEL_DROP_DIR)
    for path in MODEL_DROP_DIR.iterdir():
        if path.is_dir() and is_video_model(path):
            found.append(path)
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def fit_image(image, size: int):
    from PIL import Image

    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def make_default_world(size: int, seed: int):
    import numpy as np
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(seed)
    base = rng.normal(127, 42, (size, size, 3)).clip(0, 255).astype("uint8")
    image = Image.fromarray(base, "RGB").filter(ImageFilter.GaussianBlur(max(1, size / 64)))
    return image


def create_soft_mask(size: int, center_x: float, center_y: float, radius: float, feather: float):
    import numpy as np
    from PIL import Image

    yy, xx = np.mgrid[0:size, 0:size]
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    feather = max(1.0, feather)
    mask = 1.0 - np.clip((distance - radius) / feather, 0.0, 1.0)
    mask = (mask * 255.0).astype("uint8")
    return Image.fromarray(mask, "L")


def pil_mask_to_tensor(mask, device, dtype):
    import torch
    import numpy as np

    array = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)


def affine_camera_warp(image, forward: float, turn: float, vertical: float):
    """Fake camera motion using zoom, horizontal shift, and a slight perspective-like shear."""
    from PIL import Image

    width, height = image.size
    zoom = max(0.86, min(1.18, 1.0 + forward))
    scaled_w = max(8, round(width * zoom))
    scaled_h = max(8, round(height * zoom))
    scaled = image.resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    x = round((width - scaled_w) / 2 + turn * width)
    y = round((height - scaled_h) / 2 + vertical * height)
    canvas.paste(scaled, (x, y))

    # A subtle shear makes A/D feel more like turning than flat strafing.
    shear = max(-0.12, min(0.12, -turn * 0.9))
    return canvas.transform(
        (width, height),
        Image.Transform.AFFINE,
        (1.0, shear, -shear * height / 2, 0.0, 1.0, 0.0),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
    )


def exposed_region_mask(size: int, forward: float, turn: float, vertical: float):
    """Estimate which borders were uncovered by the camera warp."""
    from PIL import Image, ImageDraw, ImageFilter

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    border = max(2, round(size * (abs(turn) * 1.8 + max(0.0, -forward) * 1.5)))
    if turn > 0:
        draw.rectangle((0, 0, min(size, border), size), fill=255)
    elif turn < 0:
        draw.rectangle((max(0, size - border), 0, size, size), fill=255)
    if forward < 0:
        draw.rectangle((0, 0, size, border), fill=255)
        draw.rectangle((0, size - border, size, size), fill=255)
    if vertical > 0:
        draw.rectangle((0, 0, size, max(2, round(size * vertical * 1.5))), fill=255)
    elif vertical < 0:
        draw.rectangle((0, size - max(2, round(size * abs(vertical) * 1.5)), size, size), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(max(1, size / 48)))


class OasisApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Roblox Oasis - Hybrid Treadmill")
        self.root.geometry("1120x790")
        self.root.minsize(940, 680)
        self.root.configure(bg=BG)

        self.flow, self.source_app = load_flow_module()
        self.models: dict[str, Path] = {}
        self.loaded_model = None
        self.loaded_key = None
        self.device = None
        self.dtype = None
        self.current_frame = None
        self.player_anchor = None
        self.mask_image = None
        self.preview_photo = None
        self.running = False
        self.worker_thread = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.keys: set[str] = set()
        self.jump_phase = 0.0
        self.noise_world = None
        self.noise_x = 0
        self.noise_y = 0
        self.frame_counter = 0
        self.last_frame_time = None
        self.mask_mode = False
        self.mask_center = (0.5, 0.56)

        # Lightweight hybrid world-state: keep the original noise treadmill, but add
        # gentle inertial motion, weak viewpoint memory, and uncertainty-guided noise.
        self.pose_x = 0.0
        self.pose_z = 0.0
        self.pose_yaw = 0.0
        self.forward_velocity = 0.0
        self.turn_velocity = 0.0
        self.view_memory: dict[tuple[int, int, int], dict] = {}
        self.confidence_image = None
        self.last_motion_time = time.perf_counter()
        self.active_flow_steps = 4

        self._style()
        self._build()
        self.refresh_models()
        self.root.bind_all("<KeyPress>", self.on_key_down)
        self.root.bind_all("<KeyRelease>", self.on_key_up)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(30, self.poll_frames)

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

    def _build(self):
        outer = ttk.Frame(self.root, style="Panel.TFrame", padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Roblox Oasis — Hybrid Treadmill", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text=f"Model drop folder: {MODEL_DROP_DIR.name}  •  Uses helpers from {self.source_app.name}",
            style="Body.TLabel",
        ).pack(anchor="w", pady=(2, 12))

        body = ttk.Frame(outer, style="Panel.TFrame")
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Scrollable left control panel so the lower buttons remain reachable.
        control_shell = ttk.Frame(body, style="Card.TFrame")
        control_shell.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        control_shell.rowconfigure(0, weight=1)
        control_shell.columnconfigure(0, weight=1)

        control_canvas = tk.Canvas(
            control_shell,
            width=292,
            bg=PANEL,
            highlightthickness=0,
            borderwidth=0,
        )
        control_scrollbar = ttk.Scrollbar(
            control_shell,
            orient="vertical",
            command=control_canvas.yview,
        )
        controls = ttk.Frame(control_canvas, style="Card.TFrame", padding=12)
        controls_window = control_canvas.create_window((0, 0), window=controls, anchor="nw")

        control_canvas.configure(yscrollcommand=control_scrollbar.set)
        control_canvas.grid(row=0, column=0, sticky="nsew")
        control_scrollbar.grid(row=0, column=1, sticky="ns")

        def update_control_scrollregion(_event=None):
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))

        def resize_control_contents(event):
            control_canvas.itemconfigure(controls_window, width=event.width)

        def scroll_controls(event):
            if event.delta:
                control_canvas.yview_scroll(int(-event.delta / 120) * 3, "units")
            elif getattr(event, "num", None) == 4:
                control_canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                control_canvas.yview_scroll(3, "units")
            return "break"

        def bind_control_wheel(_event=None):
            control_canvas.bind_all("<MouseWheel>", scroll_controls)
            control_canvas.bind_all("<Button-4>", scroll_controls)
            control_canvas.bind_all("<Button-5>", scroll_controls)

        def unbind_control_wheel(_event=None):
            control_canvas.unbind_all("<MouseWheel>")
            control_canvas.unbind_all("<Button-4>")
            control_canvas.unbind_all("<Button-5>")

        controls.bind("<Configure>", update_control_scrollregion)
        control_canvas.bind("<Configure>", resize_control_contents)
        control_canvas.bind("<Enter>", bind_control_wheel)
        control_canvas.bind("<Leave>", unbind_control_wheel)
        controls.bind("<Enter>", bind_control_wheel)
        controls.bind("<Leave>", unbind_control_wheel)

        ttk.Label(controls, text="World Controls", style="CardHeading.TLabel").pack(anchor="w", pady=(0, 8))

        self.model_var = tk.StringVar()
        self.model_box = ttk.Combobox(controls, textvariable=self.model_var, state="readonly", width=31)
        self.model_box.pack(fill="x", pady=3)
        ttk.Button(controls, text="Refresh Folder", command=self.refresh_models).pack(fill="x", pady=3)
        ttk.Button(controls, text="Open Model Folder", command=self.open_model_folder).pack(fill="x", pady=3)

        self.start_path_var = tk.StringVar()
        ttk.Label(controls, text="Starting screenshot (optional)", style="Meta.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.start_path_var, style="Field.TEntry").pack(fill="x", pady=3)
        ttk.Button(controls, text="Choose Starting Image", command=self.choose_start_image).pack(fill="x", pady=3)

        self.steps_var = tk.StringVar(value="4")
        self.method_var = tk.StringVar(value="Euler")
        self.speed_var = tk.DoubleVar(value=0.018)
        self.turn_var = tk.DoubleVar(value=0.025)
        self.noise_var = tk.DoubleVar(value=0.30)
        self.feedback_var = tk.DoubleVar(value=0.18)
        self.preserve_var = tk.DoubleVar(value=0.94)
        self.mask_radius_var = tk.DoubleVar(value=0.16)
        self.mask_feather_var = tk.DoubleVar(value=0.09)
        self.inertia_var = tk.DoubleVar(value=0.68)
        self.memory_strength_var = tk.DoubleVar(value=0.18)
        self.stability_var = tk.DoubleVar(value=0.22)
        self.memory_enabled_var = tk.BooleanVar(value=True)
        self.uncertainty_enabled_var = tk.BooleanVar(value=True)
        self.dynamic_steps_var = tk.BooleanVar(value=True)
        self.moving_steps_var = tk.StringVar(value="1")
        self.idle_steps_var = tk.StringVar(value="6")
        self.idle_delay_var = tk.DoubleVar(value=0.65)
        self.seed_var = tk.StringVar(value=str(random.randrange(1, 1_000_000_000)))

        self._combo(controls, "Interactive flow steps", self.steps_var, ["1", "2", "3", "4", "6", "8", "10"])
        ttk.Checkbutton(controls, text="Dynamic moving/idle flow steps", variable=self.dynamic_steps_var).pack(anchor="w", pady=(5, 1))
        self._combo(controls, "Steps while moving", self.moving_steps_var, ["1", "2", "3", "4"])
        self._combo(controls, "Steps while standing still", self.idle_steps_var, ["2", "3", "4", "6", "8", "10"])
        self._scale(controls, "Idle refinement delay", self.idle_delay_var, 0.15, 2.0)
        self._combo(controls, "ODE method", self.method_var, ["Euler", "Heun"])
        self._scale(controls, "Movement strength", self.speed_var, 0.004, 0.050)
        self._scale(controls, "Turning strength", self.turn_var, 0.005, 0.070)
        self._scale(controls, "Directional noise", self.noise_var, 0.0, 0.85)
        self._scale(controls, "Old-frame feedback", self.feedback_var, 0.0, 0.65)
        self._scale(controls, "Player preservation", self.preserve_var, 0.0, 1.0)
        self._scale(controls, "Camera inertia", self.inertia_var, 0.0, 0.95)
        self._scale(controls, "View-memory strength", self.memory_strength_var, 0.0, 0.45)
        self._scale(controls, "Stable-region preservation", self.stability_var, 0.0, 0.55)
        ttk.Checkbutton(controls, text="Persistent viewpoint memory", variable=self.memory_enabled_var).pack(anchor="w", pady=(7, 1))
        ttk.Checkbutton(controls, text="Adaptive uncertainty noise", variable=self.uncertainty_enabled_var).pack(anchor="w", pady=(1, 4))
        self._scale(controls, "Player mask radius", self.mask_radius_var, 0.04, 0.35, self.update_mask)
        self._scale(controls, "Mask feather", self.mask_feather_var, 0.01, 0.25, self.update_mask)

        ttk.Label(controls, text="Seed", style="Meta.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=self.seed_var, style="Field.TEntry").pack(fill="x", pady=3)
        ttk.Button(controls, text="Randomize Seed", command=lambda: self.seed_var.set(str(random.randrange(1, 1_000_000_000)))).pack(fill="x", pady=3)

        self.mask_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Protect selected player region", variable=self.mask_enabled_var).pack(anchor="w", pady=(8, 2))
        ttk.Button(controls, text="Select Player on Preview", command=self.enable_mask_selection).pack(fill="x", pady=3)
        ttk.Button(controls, text="Reset World", command=self.reset_world).pack(fill="x", pady=(10, 3))

        self.play_btn = ttk.Button(controls, text="Start Oasis", command=self.toggle_play, style="Accent.TButton")
        self.play_btn.pack(fill="x", pady=(10, 3))
        ttk.Button(controls, text="Capture Frame", command=self.capture_frame).pack(fill="x", pady=3)

        ttk.Label(
            controls,
            text="Controls: W/S move forward/back, A/D turn, Space jumps. This hybrid build keeps the original noise treadmill, then gently adds inertia, weak viewpoint memory, adaptive noise, and optional low-step movement with higher-step idle refinement.",
            style="Meta.TLabel",
            wraplength=245,
            justify="left",
        ).pack(fill="x", pady=(10, 0))

        display_card = ttk.Frame(body, style="Card.TFrame", padding=10)
        display_card.grid(row=0, column=1, sticky="nsew")
        display_card.rowconfigure(1, weight=1)
        display_card.columnconfigure(0, weight=1)

        top = ttk.Frame(display_card, style="Card.TFrame")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        self.status_var = tk.StringVar(value="Ready. Add one trained video model to the model folder.")
        ttk.Label(top, textvariable=self.status_var, style="Meta.TLabel").pack(side="left")
        self.fps_var = tk.StringVar(value="AI FPS: —")
        ttk.Label(top, textvariable=self.fps_var, style="Meta.TLabel").pack(side="right")

        self.canvas = tk.Canvas(display_card, bg="#08090c", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", lambda _e: self.show_frame(self.current_frame))

        self.key_var = tk.StringVar(value="Input: idle")
        ttk.Label(display_card, textvariable=self.key_var, style="Meta.TLabel").grid(row=2, column=0, sticky="w", pady=(7, 0))

    def _combo(self, parent, label, variable, values):
        ttk.Label(parent, text=label, style="Meta.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly").pack(fill="x", pady=3)

    def _scale(self, parent, label, variable, low, high, command=None):
        ttk.Label(parent, text=label, style="Meta.TLabel").pack(anchor="w", pady=(5, 0))
        scale = ttk.Scale(parent, variable=variable, from_=low, to=high, command=command)
        scale.pack(fill="x", pady=(1, 2))

    def refresh_models(self):
        paths = discover_models()
        self.models = {p.name: p for p in paths}
        self.model_box["values"] = list(self.models)
        if paths and self.model_var.get() not in self.models:
            self.model_var.set(paths[0].name)
        if len(paths) == 1:
            self.status_var.set(f"Found model: {paths[0].name}")
        elif len(paths) > 1:
            self.status_var.set(f"Found {len(paths)} models. Choose one from the dropdown.")
        else:
            self.status_var.set(f"No model found. Put a video model folder inside: {MODEL_DROP_DIR}")

    def open_model_folder(self):
        import os
        MODEL_DROP_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(MODEL_DROP_DIR) if hasattr(os, "startfile") else None

    def choose_start_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if path:
            self.start_path_var.set(path)
            self.reset_world()

    def get_resolution(self) -> int:
        key = self.model_var.get()
        if key not in self.models:
            return 256
        info = json.loads((self.models[key] / "flow_video_model_info.json").read_text(encoding="utf-8"))
        return int(info.get("resolution", 256))

    def reset_world(self):
        try:
            size = self.get_resolution()
            seed = int(self.seed_var.get())
            if self.start_path_var.get() and Path(self.start_path_var.get()).is_file():
                from PIL import Image
                with Image.open(self.start_path_var.get()) as image:
                    self.current_frame = fit_image(image, size)
            else:
                self.current_frame = make_default_world(size, seed)
            self.player_anchor = self.current_frame.copy()
            self.noise_world = None
            self.noise_x = self.noise_y = 0
            self.frame_counter = 0
            self.pose_x = 0.0
            self.pose_z = 0.0
            self.pose_yaw = 0.0
            self.forward_velocity = 0.0
            self.turn_velocity = 0.0
            self.last_motion_time = time.perf_counter()
            self.active_flow_steps = int(self.steps_var.get())
            self.view_memory.clear()
            from PIL import Image
            self.confidence_image = Image.new("L", (size, size), 170)
            self.update_mask()
            self.show_frame(self.current_frame)
            self.status_var.set("World reset. Select the player region, then press Start Oasis.")
        except Exception as exc:
            messagebox.showerror("Could not reset world", str(exc))

    def update_mask(self, _value=None):
        if self.current_frame is None:
            return
        size = self.current_frame.width
        cx = self.mask_center[0] * size
        cy = self.mask_center[1] * size
        radius = self.mask_radius_var.get() * size
        feather = self.mask_feather_var.get() * size
        self.mask_image = create_soft_mask(size, cx, cy, radius, feather)
        self.show_frame(self.current_frame)

    def enable_mask_selection(self):
        self.mask_mode = True
        self.status_var.set("Click the center of the player in the preview.")

    def on_canvas_click(self, event):
        if not self.mask_mode or self.current_frame is None:
            return
        bbox = self._image_bbox()
        if bbox is None:
            return
        x0, y0, x1, y1 = bbox
        if not (x0 <= event.x <= x1 and y0 <= event.y <= y1):
            return
        self.mask_center = ((event.x - x0) / max(1, x1 - x0), (event.y - y0) / max(1, y1 - y0))
        self.player_anchor = self.current_frame.copy()
        self.mask_mode = False
        self.update_mask()
        self.status_var.set("Protected player region selected with a soft Gaussian edge.")

    def _image_bbox(self):
        if self.current_frame is None:
            return None
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        scale = min(cw / self.current_frame.width, ch / self.current_frame.height)
        w, h = self.current_frame.width * scale, self.current_frame.height * scale
        return ((cw - w) / 2, (ch - h) / 2, (cw + w) / 2, (ch + h) / 2)

    def show_frame(self, frame):
        if frame is None:
            return
        from PIL import Image, ImageTk, ImageDraw

        preview = frame.copy()
        if self.mask_image is not None and self.mask_enabled_var.get():
            overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
            edge = self.mask_image.point(lambda p: 120 if 20 < p < 235 else 0)
            color = Image.new("RGBA", preview.size, (100, 150, 255, 0))
            color.putalpha(edge)
            overlay.alpha_composite(color)
            preview = Image.alpha_composite(preview.convert("RGBA"), overlay).convert("RGB")

        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        preview.thumbnail((cw, ch), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.preview_photo, anchor="center")

    def on_key_down(self, event):
        key = event.keysym.lower()
        if key in {"w", "a", "s", "d", "space"}:
            self.keys.add(key)
            self.update_input_label()

    def on_key_up(self, event):
        self.keys.discard(event.keysym.lower())
        self.update_input_label()

    def update_input_label(self):
        order = [k.upper() if k != "space" else "SPACE" for k in ["w", "a", "s", "d", "space"] if k in self.keys]
        self.key_var.set("Input: " + (" + ".join(order) if order else "idle"))

    def toggle_play(self):
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self):
        try:
            key = self.model_var.get()
            if key not in self.models:
                raise ValueError(f"Put a valid trained video model inside {MODEL_DROP_DIR.name}, then refresh.")
            if self.current_frame is None:
                self.reset_world()
            if self.current_frame is None:
                raise ValueError("The world could not be initialized.")
            steps = int(self.steps_var.get())
            if steps < 1:
                raise ValueError("Flow steps must be at least 1.")
            if self.dynamic_steps_var.get():
                if int(self.moving_steps_var.get()) < 1 or int(self.idle_steps_var.get()) < 1:
                    raise ValueError("Moving and idle flow steps must be at least 1.")
            self.running = True
            self.play_btn.config(text="Stop Oasis")
            self.status_var.set("Loading model and entering the generated world...")
            self.worker_thread = threading.Thread(target=self.generation_loop, daemon=True)
            self.worker_thread.start()
        except Exception as exc:
            messagebox.showerror("Cannot start Oasis", str(exc))

    def stop(self):
        self.running = False
        self.play_btn.config(text="Start Oasis")
        self.status_var.set("Oasis paused.")

    def ensure_noise_world(self, size, seed):
        import torch
        if self.noise_world is None or self.noise_world.shape[-1] != size * 4:
            generator = torch.Generator(device=self.device).manual_seed(seed + 99173)
            self.noise_world = torch.randn((1, 3, size * 4, size * 4), generator=generator, device=self.device, dtype=self.dtype)
            self.noise_x = size * 3 // 2
            self.noise_y = size * 3 // 2

    def noise_view(self, size, dx, dy):
        max_offset = size * 3
        self.noise_x = int(max(0, min(max_offset, self.noise_x + dx)))
        self.noise_y = int(max(0, min(max_offset, self.noise_y + dy)))
        return self.noise_world[:, :, self.noise_y:self.noise_y + size, self.noise_x:self.noise_x + size]

    def pose_key(self):
        return (round(self.pose_x / 0.22), round(self.pose_z / 0.22), round(self.pose_yaw / 0.20))

    def retrieve_view_memory(self, size):
        if not self.memory_enabled_var.get():
            return None, None
        entry = self.view_memory.get(self.pose_key())
        if not entry:
            return None, None
        image = entry["image"]
        confidence = entry["confidence"]
        if image.size != (size, size):
            image = image.resize((size, size))
            confidence = confidence.resize((size, size))
        return image, confidence

    def store_view_memory(self, image, confidence):
        if not self.memory_enabled_var.get():
            return
        from PIL import Image, ImageChops
        key = self.pose_key()
        old = self.view_memory.get(key)
        if old is None:
            self.view_memory[key] = {"image": image.copy(), "confidence": confidence.copy(), "age": 0}
        else:
            update_mask = ImageChops.invert(old["confidence"]).point(lambda p: min(145, 24 + p))
            old["image"] = Image.composite(image, old["image"], update_mask)
            old["confidence"] = ImageChops.lighter(old["confidence"], confidence)
            old["age"] = 0
        for memory_key in list(self.view_memory):
            if memory_key == key:
                continue
            self.view_memory[memory_key]["age"] += 1
            if self.view_memory[memory_key]["age"] > 480:
                del self.view_memory[memory_key]

    def build_uncertainty_mask(self, previous_pil, warped_pil, exposed):
        from PIL import ImageChops, ImageFilter
        changed = ImageChops.difference(previous_pil.convert("RGB"), warped_pil.convert("RGB")).convert("L")
        changed = changed.point(lambda p: min(255, int(p * 3.8)))
        uncertain = ImageChops.lighter(changed, exposed)
        if self.confidence_image is not None:
            low_confidence = ImageChops.invert(self.confidence_image).point(lambda p: int(p * 0.65))
            uncertain = ImageChops.lighter(uncertain, low_confidence)
        return uncertain.filter(ImageFilter.GaussianBlur(max(1, previous_pil.width / 96)))

    def generation_loop(self):
        try:
            import torch
            from PIL import Image

            key = self.model_var.get()
            model_path = self.models[key]
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            if self.loaded_key != key or self.loaded_model is None:
                self.loaded_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.loaded_model = self.flow.load_video_unet(model_path, device=self.device, dtype=self.dtype)
                self.loaded_key = key

            size = int(self.loaded_model.config.sample_size)
            if isinstance(self.loaded_model.config.sample_size, (list, tuple)):
                size = int(self.loaded_model.config.sample_size[0])
            seed = int(self.seed_var.get())
            generator = torch.Generator(device=self.device).manual_seed(seed)
            self.ensure_noise_world(size, seed)
            self.root.after(0, lambda: self.status_var.set("Oasis running. Use WASD and Space."))

            while self.running:
                started = time.perf_counter()
                keys = set(self.keys)
                move = self.speed_var.get()
                turn_strength = self.turn_var.get()
                target_forward = move if "w" in keys else (-move if "s" in keys else 0.0)
                target_turn = turn_strength if "a" in keys else (-turn_strength if "d" in keys else 0.0)

                inertia = self.inertia_var.get()
                self.forward_velocity = self.forward_velocity * inertia + target_forward * (1.0 - inertia)
                self.turn_velocity = self.turn_velocity * inertia + target_turn * (1.0 - inertia)
                forward = self.forward_velocity
                turn = self.turn_velocity

                moving_now = (
                    bool(keys.intersection({"w", "a", "s", "d", "space"}))
                    or abs(forward) > max(0.00035, move * 0.08)
                    or abs(turn) > max(0.00035, turn_strength * 0.08)
                )
                now = time.perf_counter()
                if moving_now:
                    self.last_motion_time = now
                if self.dynamic_steps_var.get():
                    idle_ready = (not moving_now) and (now - self.last_motion_time >= self.idle_delay_var.get())
                    flow_steps = int(self.idle_steps_var.get()) if idle_ready else int(self.moving_steps_var.get())
                else:
                    flow_steps = int(self.steps_var.get())
                self.active_flow_steps = flow_steps

                self.pose_yaw += turn * 1.05
                self.pose_x += math.sin(self.pose_yaw) * forward * 2.0
                self.pose_z += math.cos(self.pose_yaw) * forward * 2.0

                if "space" in keys and self.jump_phase <= 0.0:
                    self.jump_phase = 0.001
                vertical = 0.0
                if self.jump_phase > 0.0:
                    self.jump_phase += 0.16
                    vertical = -math.sin(min(math.pi, self.jump_phase)) * 0.045
                    if self.jump_phase >= math.pi:
                        self.jump_phase = 0.0

                previous_pil = self.current_frame.copy()
                warped_pil = affine_camera_warp(previous_pil, forward, turn, vertical)
                previous = self.flow.pil_to_normalized_tensor(warped_pil, self.device, self.dtype)

                scroll = max(1, round(size * 0.025))
                noise_dx = int(round((-turn / max(0.001, self.turn_var.get())) * scroll)) if abs(turn) > 1e-6 else 0
                noise_dy = int(round((-forward / max(0.001, self.speed_var.get())) * scroll)) if abs(forward) > 1e-6 else 0
                base_noise = self.noise_view(size, noise_dx, noise_dy).clone()

                exposed = exposed_region_mask(size, forward, turn, vertical)
                exposed_tensor = pil_mask_to_tensor(exposed, self.device, self.dtype)
                uncertainty = self.build_uncertainty_mask(previous_pil, warped_pil, exposed)
                uncertainty_tensor = pil_mask_to_tensor(uncertainty, self.device, self.dtype)
                directional = self.noise_var.get()
                fresh = torch.randn(base_noise.shape, generator=generator, device=self.device, dtype=self.dtype)

                base_noise = base_noise * (1.0 - exposed_tensor * directional * 0.45) + fresh * (exposed_tensor * directional * 0.45)
                if self.uncertainty_enabled_var.get() and directional > 0:
                    adaptive = (uncertainty_tensor * directional * 0.95).clamp(0, 1)
                    base_noise = base_noise * (1.0 - adaptive) + fresh * adaptive

                memory_image, memory_confidence = self.retrieve_view_memory(size)
                if memory_image is not None:
                    memory_tensor = self.flow.pil_to_normalized_tensor(memory_image, self.device, self.dtype)
                    memory_mask = pil_mask_to_tensor(memory_confidence, self.device, self.dtype)
                    memory_amount = (memory_mask * self.memory_strength_var.get()).clamp(0, 1)
                    previous = previous * (1.0 - memory_amount) + memory_tensor * memory_amount
                    base_noise = base_noise * (1.0 - memory_amount * 0.55) + memory_tensor * (memory_amount * 0.55)

                if self.mask_enabled_var.get() and self.mask_image is not None:
                    protect = pil_mask_to_tensor(self.mask_image, self.device, self.dtype)
                    base_noise = base_noise * (1.0 - protect * self.preserve_var.get()) + previous * (protect * self.preserve_var.get())

                generated = self.flow.sample_next_video_frame(
                    self.loaded_model,
                    previous,
                    flow_steps,
                    self.device,
                    self.dtype,
                    generator,
                    method=self.method_var.get(),
                    initial_noise=base_noise,
                )

                feedback = self.feedback_var.get()
                if feedback > 0:
                    generated = (generated * (1.0 - feedback) + previous * feedback).clamp(-1, 1)

                if self.uncertainty_enabled_var.get():
                    stable_mask = ((1.0 - uncertainty_tensor) * self.stability_var.get()).clamp(0, 1)
                    generated = generated * (1.0 - stable_mask) + previous * stable_mask
                    generated = generated.clamp(-1, 1)

                result = self.flow.tensor_to_pil(generated[0])
                if self.mask_enabled_var.get() and self.mask_image is not None:
                    source = previous_pil if self.player_anchor is None else self.player_anchor
                    preserve = self.preserve_var.get()
                    if preserve < 1.0:
                        from PIL import ImageChops
                        weak_mask = self.mask_image.point(lambda p: int(p * preserve))
                    else:
                        weak_mask = self.mask_image
                    result = Image.composite(source, result, weak_mask)
                    self.player_anchor = Image.blend(source, result, 0.03)

                from PIL import ImageChops
                confidence = ImageChops.invert(uncertainty).point(lambda p: int(55 + p * 0.70))
                if self.confidence_image is None:
                    self.confidence_image = confidence
                else:
                    self.confidence_image = Image.blend(self.confidence_image, confidence, 0.16)
                self.store_view_memory(result, self.confidence_image)

                self.current_frame = result
                self.frame_counter += 1
                elapsed = max(1e-6, time.perf_counter() - started)
                try:
                    while self.frame_queue.qsize() > 0:
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait((result.copy(), 1.0 / elapsed))
                except queue.Full:
                    pass
        except Exception:
            error = traceback.format_exc()
            self.running = False
            self.root.after(0, lambda: self._generation_failed(error))

    def poll_frames(self):
        try:
            while True:
                frame, fps = self.frame_queue.get_nowait()
                self.show_frame(frame)
                self.fps_var.set(f"AI FPS: {fps:.2f}  •  Steps: {self.active_flow_steps}")
        except queue.Empty:
            pass
        self.root.after(30, self.poll_frames)

    def capture_frame(self):
        if self.current_frame is None:
            messagebox.showinfo("Nothing to capture", "Start or reset the world first.")
            return
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        path = CAPTURE_DIR / f"oasis_{time.strftime('%Y%m%d_%H%M%S')}.png"
        self.current_frame.save(path)
        self.status_var.set(f"Captured frame to {path.name}")

    def _generation_failed(self, error):
        self.play_btn.config(text="Start Oasis")
        self.status_var.set("Oasis stopped because generation failed.")
        messagebox.showerror("Oasis generation error", error[-2200:])

    def close(self):
        self.running = False
        self.root.destroy()


def main():
    MODEL_DROP_DIR.mkdir(parents=True, exist_ok=True)
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    try:
        OasisApp(root)
    except Exception as exc:
        root.withdraw()
        messagebox.showerror("Could not start Roblox Oasis", f"{exc}\n\n{traceback.format_exc()[-1800:]}")
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
