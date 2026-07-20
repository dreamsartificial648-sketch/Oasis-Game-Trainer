
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import mss
except ImportError:
    mss = None

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

try:
    from pynput import keyboard, mouse
except ImportError:
    keyboard = None
    mouse = None


APP_BG = "#15171d"
PANEL = "#20232b"
FIELD = "#2b2f39"
FG = "#eef1f7"
DIM = "#aab0be"
ACCENT = "#6f98ff"


@dataclass
class ActionState:
    w: int = 0
    a: int = 0
    s: int = 0
    d: int = 0
    jump: int = 0
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    zoom: float = 0.0

    def set_direction(self, direction: str, pressed: int) -> None:
        """Set a canonical direction shared by WASD and arrow-key controls."""
        setattr(self, direction, pressed)

    @property
    def move_x(self) -> int:
        return self.d - self.a

    @property
    def move_y(self) -> int:
        return self.w - self.s

    def compact_label(self) -> str:
        labels = []
        if self.w:
            labels.append("W")
        if self.a:
            labels.append("A")
        if self.s:
            labels.append("S")
        if self.d:
            labels.append("D")
        if self.jump:
            labels.append("J")
        if abs(self.mouse_dx) > 0.02 or abs(self.mouse_dy) > 0.02:
            labels.append("CAM")
        if abs(self.zoom) > 0.02:
            labels.append("ZOOM")
        return "-".join(labels) if labels else "IDLE"


class RegionSelector(tk.Toplevel):
    def __init__(self, parent: tk.Tk, callback):
        super().__init__(parent)
        self.callback = callback
        self.start_x = None
        self.start_y = None
        self.rect = None

        self.attributes("-fullscreen", True)
        self.attributes("-alpha", 0.35)
        self.attributes("-topmost", True)
        self.configure(bg="black")
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(
            self.winfo_screenwidth() // 2,
            40,
            text="Drag over the game viewport. Press Esc to cancel.",
            fill="white",
            font=("Segoe UI", 18, "bold"),
        )
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<Escape>", lambda _e: self.destroy())

    def on_press(self, event):
        self.start_x = event.x_root
        self.start_y = event.y_root
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="#6f98ff", width=3
        )

    def on_drag(self, event):
        if self.start_x is None:
            return
        x0 = self.start_x
        y0 = self.start_y
        self.canvas.coords(
            self.rect,
            x0,
            y0,
            event.x_root,
            event.y_root
        )

    def on_release(self, event):
        if self.start_x is None:
            return

        x0 = min(self.start_x, event.x_root)
        y0 = min(self.start_y, event.y_root)
        x1 = max(self.start_x, event.x_root)
        y1 = max(self.start_y, event.y_root)

        width = x1 - x0
        height = y1 - y0

        if width < 32 or height < 32:
            messagebox.showwarning("Region too small", "Select a larger area.")
            return

        self.destroy()
        self.callback({
            "left": int(x0),
            "top": int(y0),
            "width": int(width),
            "height": int(height),
        })


class RobloxActionRecorder:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Game Action Dataset Recorder")
        self.root.geometry("980x720")
        self.root.minsize(840, 620)
        self.root.configure(bg=APP_BG)

        self.output_dir = Path.cwd() / "roblox_action_dataset"
        self.capture_region = None
        self.recording = False
        self.capture_thread = None
        self.keyboard_listener = None
        self.mouse_listener = None
        self.last_mouse_position = None
        self.pending_mouse_dx = 0.0
        self.pending_mouse_dy = 0.0
        self.pending_zoom = 0.0
        self.action_lock = threading.Lock()
        self.action_state = ActionState()
        self.frame_index = 0
        self.start_time = 0.0
        self.preview_photo = None
        self.last_preview = None

        self._style()
        self._build()
        self._start_keyboard_listener()
        self._start_mouse_listener()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=APP_BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("Panel.TFrame", background=APP_BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("Heading.TLabel", background=APP_BG, foreground=FG, font=("Segoe UI", 17, "bold"))
        style.configure("Body.TLabel", background=APP_BG, foreground=DIM)
        style.configure("CardHeading.TLabel", background=PANEL, foreground=FG, font=("Segoe UI", 12, "bold"))
        style.configure("Meta.TLabel", background=PANEL, foreground=DIM)
        style.configure("Value.TLabel", background=PANEL, foreground=FG)
        style.configure("Field.TEntry", fieldbackground=FIELD, foreground=FG, insertcolor=FG)
        style.configure("Accent.TButton", background=ACCENT, foreground="#10131a", font=("Segoe UI", 10, "bold"))
        style.configure("TCheckbutton", background=PANEL, foreground=FG)
        style.configure("TCombobox", fieldbackground=FIELD, foreground=FG)

    def _build(self):
        outer = ttk.Frame(self.root, style="Panel.TFrame", padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Game Action Dataset Recorder",
            style="Heading.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            outer,
            text="Records aligned W/A/S/D or arrow-key directions, Space, mouse movement, and zoom labels.",
            style="Body.TLabel"
        ).pack(anchor="w", pady=(2, 12))

        body = ttk.Frame(outer, style="Panel.TFrame")
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        controls = ttk.Frame(body, style="Card.TFrame", padding=14)
        controls.grid(row=0, column=0, sticky="ns", padx=(0, 14))

        ttk.Label(controls, text="Capture Settings", style="CardHeading.TLabel").pack(anchor="w", pady=(0, 10))

        self.output_var = tk.StringVar(value=str(self.output_dir))
        ttk.Label(controls, text="Dataset output folder", style="Meta.TLabel").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.output_var, width=36, style="Field.TEntry").pack(fill="x", pady=4)
        ttk.Button(controls, text="Choose Output Folder", command=self.choose_output_folder).pack(fill="x", pady=3)

        self.region_var = tk.StringVar(value="No region selected")
        ttk.Label(controls, text="Capture region", style="Meta.TLabel").pack(anchor="w", pady=(10, 0))
        ttk.Label(controls, textvariable=self.region_var, style="Value.TLabel", wraplength=260).pack(anchor="w", pady=(3, 5))
        ttk.Button(controls, text="Select Game Region", command=self.select_region).pack(fill="x", pady=3)

        self.fps_var = tk.IntVar(value=12)
        ttk.Label(controls, text="Capture FPS", style="Meta.TLabel").pack(anchor="w", pady=(10, 0))
        ttk.Spinbox(controls, from_=1, to=30, textvariable=self.fps_var, width=8).pack(fill="x", pady=4)

        self.resolution_var = tk.StringVar(value="256")
        ttk.Label(controls, text="Output resolution", style="Meta.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            state="readonly",
            textvariable=self.resolution_var,
            values=["128", "192", "256", "320", "384"],
        ).pack(fill="x", pady=4)

        self.crop_mode_var = tk.StringVar(value="Center crop")
        ttk.Label(controls, text="Resize method", style="Meta.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            state="readonly",
            textvariable=self.crop_mode_var,
            values=["Center crop", "Letterbox"],
        ).pack(fill="x", pady=4)

        self.include_idle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="Record idle frames",
            variable=self.include_idle_var
        ).pack(anchor="w", pady=(10, 2))

        self.filename_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="Include action label in filename",
            variable=self.filename_labels_var
        ).pack(anchor="w", pady=2)

        self.start_delay_var = tk.DoubleVar(value=3.0)
        ttk.Label(controls, text="Start delay (seconds)", style="Meta.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Spinbox(
            controls,
            from_=0.0,
            to=10.0,
            increment=0.5,
            textvariable=self.start_delay_var
        ).pack(fill="x", pady=4)

        self.record_mouse_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="Record mouse yaw/pitch and wheel zoom",
            variable=self.record_mouse_var
        ).pack(anchor="w", pady=(8, 2))

        self.hide_while_recording_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="Hide recorder while recording (recommended)",
            variable=self.hide_while_recording_var,
        ).pack(anchor="w", pady=(6, 2))

        self.mouse_scale_var = tk.DoubleVar(value=120.0)
        ttk.Label(controls, text="Mouse normalization (pixels = full input)", style="Meta.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Spinbox(controls, from_=20.0, to=1000.0, increment=10.0, textvariable=self.mouse_scale_var).pack(fill="x", pady=4)

        self.zoom_scale_var = tk.DoubleVar(value=3.0)
        ttk.Label(controls, text="Zoom normalization (wheel ticks = full input)", style="Meta.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Spinbox(controls, from_=1.0, to=20.0, increment=1.0, textvariable=self.zoom_scale_var).pack(fill="x", pady=4)

        ttk.Button(
            controls,
            text="Start Recording",
            command=self.toggle_recording,
            style="Accent.TButton"
        ).pack(fill="x", pady=(14, 4))
        self.record_btn = controls.winfo_children()[-1]

        ttk.Button(
            controls,
            text="Open Dataset Folder",
            command=self.open_dataset_folder
        ).pack(fill="x", pady=3)

        ttk.Label(
            controls,
            text=(
                "Recommended: 10-15 FPS with a training gap of 2-3. The live preview on the right confirms the capture. "
                "For capture with no recorder in the frames, leave hiding enabled and press F8 to stop."
            ),
            style="Meta.TLabel",
            wraplength=270,
            justify="left",
        ).pack(fill="x", pady=(12, 0))

        preview_card = ttk.Frame(body, style="Card.TFrame", padding=10)
        preview_card.grid(row=0, column=1, sticky="nsew")
        preview_card.rowconfigure(1, weight=1)
        preview_card.columnconfigure(0, weight=1)

        top = ttk.Frame(preview_card, style="Card.TFrame")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.status_var, style="Value.TLabel").pack(side="left")

        self.counter_var = tk.StringVar(value="Frames: 0")
        ttk.Label(top, textvariable=self.counter_var, style="Meta.TLabel").pack(side="right")

        self.preview_canvas = tk.Canvas(preview_card, bg="#08090c", highlightthickness=0)
        self.preview_canvas.grid(row=1, column=0, sticky="nsew")
        self.preview_canvas.bind("<Configure>", lambda _e: self.show_preview(self.last_preview))

        info = ttk.Frame(preview_card, style="Card.TFrame")
        info.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        self.action_var = tk.StringVar(value="Input: IDLE")
        ttk.Label(info, textvariable=self.action_var, style="Value.TLabel").pack(side="left")

        self.elapsed_var = tk.StringVar(value="Elapsed: 00:00")
        ttk.Label(info, textvariable=self.elapsed_var, style="Meta.TLabel").pack(side="right")

        # F8 is handled by the global pynput listener, including while hidden.

    def _start_keyboard_listener(self):
        if keyboard is None:
            messagebox.showerror(
                "Missing dependency",
                "Install dependencies with:\n\npip install pynput mss pillow"
            )
            return

        def on_press(key):
            if key == keyboard.Key.f8:
                self.root.after(0, self.toggle_recording)
                return

            with self.action_lock:
                try:
                    char = key.char.lower()
                except Exception:
                    char = None

                direction = {
                    "w": "w", "a": "a", "s": "s", "d": "d",
                    keyboard.Key.up: "w", keyboard.Key.left: "a",
                    keyboard.Key.down: "s", keyboard.Key.right: "d",
                }.get(char if char is not None else key)
                if direction:
                    self.action_state.set_direction(direction, 1)
                elif key == keyboard.Key.space:
                    self.action_state.jump = 1

                self._queue_action_label()

        def on_release(key):
            with self.action_lock:
                try:
                    char = key.char.lower()
                except Exception:
                    char = None

                direction = {
                    "w": "w", "a": "a", "s": "s", "d": "d",
                    keyboard.Key.up: "w", keyboard.Key.left: "a",
                    keyboard.Key.down: "s", keyboard.Key.right: "d",
                }.get(char if char is not None else key)
                if direction:
                    self.action_state.set_direction(direction, 0)
                elif key == keyboard.Key.space:
                    self.action_state.jump = 0

                self._queue_action_label()

        self.keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.keyboard_listener.daemon = True
        self.keyboard_listener.start()

    def _start_mouse_listener(self):
        if mouse is None:
            return

        def on_move(x, y):
            if not self.recording or not self.record_mouse_var.get():
                self.last_mouse_position = (x, y)
                return
            with self.action_lock:
                if self.last_mouse_position is not None:
                    self.pending_mouse_dx += float(x - self.last_mouse_position[0])
                    self.pending_mouse_dy += float(y - self.last_mouse_position[1])
                self.last_mouse_position = (x, y)

        def on_scroll(_x, _y, _dx, dy):
            if not self.recording or not self.record_mouse_var.get():
                return
            with self.action_lock:
                self.pending_zoom += float(dy)

        self.mouse_listener = mouse.Listener(on_move=on_move, on_scroll=on_scroll)
        self.mouse_listener.daemon = True
        self.mouse_listener.start()

    def _queue_action_label(self):
        label = self.action_state.compact_label()
        self.root.after(0, lambda: self.action_var.set(f"Input: {label}"))

    def choose_output_folder(self):
        path = filedialog.askdirectory(initialdir=self.output_var.get() or str(Path.cwd()))
        if path:
            self.output_var.set(path)

    def select_region(self):
        self.root.withdraw()

        def selected(region):
            self.capture_region = region
            self.region_var.set(
                f"x={region['left']}, y={region['top']}, "
                f"{region['width']}×{region['height']}"
            )
            self.root.deiconify()
            self.root.lift()

        selector = RegionSelector(self.root, selected)
        selector.protocol("WM_DELETE_WINDOW", lambda: (selector.destroy(), self.root.deiconify()))

    def _dataset_paths(self):
        root = Path(self.output_var.get()).expanduser()
        frames = root / "frames"
        metadata = root / "actions.jsonl"
        config = root / "dataset_info.json"
        return root, frames, metadata, config

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if mss is None or Image is None:
            messagebox.showerror(
                "Missing dependencies",
                "Install dependencies with:\n\npip install pynput mss pillow"
            )
            return

        if self.capture_region is None:
            messagebox.showwarning("No capture region", "Select the game viewport first.")
            return

        try:
            fps = int(self.fps_var.get())
            resolution = int(self.resolution_var.get())
            delay = float(self.start_delay_var.get())
        except ValueError:
            messagebox.showerror("Invalid settings", "FPS, resolution, and delay must be numeric.")
            return

        if fps < 1:
            messagebox.showerror("Invalid FPS", "Capture FPS must be at least 1.")
            return

        root, frames, metadata, config = self._dataset_paths()
        frames.mkdir(parents=True, exist_ok=True)

        config_data = {
            "format_version": 2,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "capture_fps": fps,
            "output_resolution": resolution,
            "resize_method": self.crop_mode_var.get(),
            "action_dimensions": ["w", "a", "s", "d", "jump", "mouse_dx", "mouse_dy", "zoom", "move_x", "move_y"],
            "mouse_normalization_pixels": float(self.mouse_scale_var.get()),
            "zoom_normalization_ticks": float(self.zoom_scale_var.get()),
            "transition_alignment": (
                "Each action row describes the held controls associated with that captured frame. "
                "For next-frame training, use row t as the action for frame t-1 -> frame t."
            ),
            "session_handling": "Each recording start writes a unique session_id. Training never crosses session boundaries.",
            "capture_region": self.capture_region,
        }
        config.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

        existing = sorted(frames.glob("frame_*.png"))
        if existing:
            try:
                last_number = max(int(p.stem.split("_")[1]) for p in existing)
            except Exception:
                last_number = len(existing)
            self.frame_index = last_number + 1
        else:
            self.frame_index = 0

        with self.action_lock:
            self.pending_mouse_dx = 0.0
            self.pending_mouse_dy = 0.0
            self.pending_zoom = 0.0
            self.last_mouse_position = None
        self.recording = True
        # A new session prevents the trainer from learning a fake transition when
        # recording is stopped, the game world changes, and recording resumes.
        self.session_id = uuid.uuid4().hex
        self.session_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.record_btn.config(text="Stop Recording")
        self.status_var.set(f"Recording starts in {delay:.1f}s...")
        self.capture_thread = threading.Thread(
            target=self.capture_loop,
            args=(fps, resolution, delay, frames, metadata),
            daemon=True
        )
        self.capture_thread.start()

    def stop_recording(self):
        self.recording = False
        self.record_btn.config(text="Start Recording")
        self.status_var.set("Recording stopped")
        self.root.after(0, self._show_recorder)

    def _hide_recorder(self):
        if self.recording and self.hide_while_recording_var.get():
            self.root.withdraw()

    def _show_recorder(self):
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        except tk.TclError:
            pass

    def process_frame(self, image, resolution: int):
        image = image.convert("RGB")

        if self.crop_mode_var.get() == "Letterbox":
            image.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (resolution, resolution), (0, 0, 0))
            x = (resolution - image.width) // 2
            y = (resolution - image.height) // 2
            canvas.paste(image, (x, y))
            return canvas

        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        return image.resize((resolution, resolution), Image.Resampling.LANCZOS)

    def capture_loop(self, fps, resolution, delay, frames_dir, metadata_path):
        try:
            time.sleep(max(0.0, delay))
            if not self.recording:
                return

            self.start_time = time.perf_counter()
            self.root.after(0, self._hide_recorder)
            self.root.after(0, lambda: self.status_var.set("Recording"))
            interval = 1.0 / fps
            next_capture = time.perf_counter()

            with mss.mss() as sct, metadata_path.open("a", encoding="utf-8") as metadata_file:
                while self.recording:
                    now = time.perf_counter()
                    if now < next_capture:
                        time.sleep(min(0.002, next_capture - now))
                        continue

                    capture_timestamp = time.perf_counter()
                    screenshot = sct.grab(self.capture_region)
                    image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                    image = self.process_frame(image, resolution)

                    with self.action_lock:
                        mouse_scale = max(1.0, float(self.mouse_scale_var.get()))
                        zoom_scale = max(1.0, float(self.zoom_scale_var.get()))
                        action = ActionState(**asdict(self.action_state))
                        if self.record_mouse_var.get():
                            action.mouse_dx = max(-1.0, min(1.0, self.pending_mouse_dx / mouse_scale))
                            action.mouse_dy = max(-1.0, min(1.0, self.pending_mouse_dy / mouse_scale))
                            action.zoom = max(-1.0, min(1.0, self.pending_zoom / zoom_scale))
                        self.pending_mouse_dx = 0.0
                        self.pending_mouse_dy = 0.0
                        self.pending_zoom = 0.0

                    if not self.include_idle_var.get() and action.compact_label() == "IDLE":
                        next_capture += interval
                        continue

                    label = action.compact_label()
                    if self.filename_labels_var.get():
                        filename = f"frame_{self.frame_index:08d}_{label}.png"
                    else:
                        filename = f"frame_{self.frame_index:08d}.png"

                    image_path = frames_dir / filename
                    image.save(image_path, compress_level=1)

                    elapsed = capture_timestamp - self.start_time
                    record = {
                        "session_id": self.session_id,
                        "session_started_at": self.session_started_at,
                        "frame_index": self.frame_index,
                        "filename": filename,
                        "timestamp_seconds": round(elapsed, 6),
                        "w": action.w,
                        "a": action.a,
                        "s": action.s,
                        "d": action.d,
                        "jump": action.jump,
                        "mouse_dx": round(action.mouse_dx, 6),
                        "mouse_dy": round(action.mouse_dy, 6),
                        "zoom": round(action.zoom, 6),
                        "move_x": action.move_x,
                        "move_y": action.move_y,
                    }
                    metadata_file.write(json.dumps(record) + "\n")
                    metadata_file.flush()

                    self.frame_index += 1
                    self.last_preview = image.copy()

                    self.root.after(0, lambda img=image.copy(): self.show_preview(img))
                    self.root.after(0, lambda: self.counter_var.set(f"Frames: {self.frame_index}"))
                    minutes = int(elapsed // 60)
                    seconds = int(elapsed % 60)
                    self.root.after(
                        0,
                        lambda m=minutes, s=seconds: self.elapsed_var.set(f"Elapsed: {m:02d}:{s:02d}")
                    )

                    next_capture += interval
                    if next_capture < time.perf_counter() - interval:
                        next_capture = time.perf_counter() + interval

        except Exception:
            error = traceback.format_exc()
            self.recording = False
            self.root.after(0, self._show_recorder)
            self.root.after(0, lambda: self.record_btn.config(text="Start Recording"))
            self.root.after(0, lambda: self.status_var.set("Recording failed"))
            self.root.after(
                0,
                lambda: messagebox.showerror("Recorder error", error[-2400:])
            )

    def show_preview(self, image):
        if image is None or ImageTk is None:
            return

        cw = max(1, self.preview_canvas.winfo_width())
        ch = max(1, self.preview_canvas.winfo_height())
        preview = image.copy()
        preview.thumbnail((cw, ch), Image.Resampling.NEAREST)
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(
            cw // 2, ch // 2,
            image=self.preview_photo,
            anchor="center"
        )

    def open_dataset_folder(self):
        import os

        root, _, _, _ = self._dataset_paths()
        root.mkdir(parents=True, exist_ok=True)

        if hasattr(os, "startfile"):
            os.startfile(root)
        else:
            messagebox.showinfo("Dataset folder", str(root))

    def close(self):
        self.recording = False
        if self.keyboard_listener is not None:
            try:
                self.keyboard_listener.stop()
            except Exception:
                pass
        if self.mouse_listener is not None:
            try:
                self.mouse_listener.stop()
            except Exception:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        RobloxActionRecorder(root)
    except Exception as exc:
        root.withdraw()
        messagebox.showerror(
            "Could not start recorder",
            f"{exc}\n\n{traceback.format_exc()[-2000:]}"
        )
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
