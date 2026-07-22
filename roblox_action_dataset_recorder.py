
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

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
CAMERA_ENCODING = "relative_degrees_v1"
DEFAULT_YAW_COUNTS_PER_360 = 2400.0
DEFAULT_PITCH_COUNTS_PER_180 = 1200.0
MAX_YAW_DEGREES_PER_FRAME = 45.0
MAX_PITCH_DEGREES_PER_FRAME = 30.0
APP_DIR = Path(__file__).resolve().parent
CAPTURE_PRESETS_FILE = APP_DIR / "roblox_action_capture_regions.json"


class WindowsRawMouseInput:
    """Receive relative mouse counts even when a game locks/re-centers the cursor."""
    def __init__(self, root, callback):
        self.available = False
        self.error = "Windows Raw Input is only available on Windows."
        self._old_proc = None
        self._wndproc = None
        self._hwnd = None
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes

            class RAWINPUTDEVICE(ctypes.Structure):
                _fields_ = [
                    ("usUsagePage", wintypes.USHORT),
                    ("usUsage", wintypes.USHORT),
                    ("dwFlags", wintypes.DWORD),
                    ("hwndTarget", wintypes.HWND),
                ]

            class RAWINPUTHEADER(ctypes.Structure):
                _fields_ = [
                    ("dwType", wintypes.DWORD),
                    ("dwSize", wintypes.DWORD),
                    ("hDevice", wintypes.HANDLE),
                    ("wParam", wintypes.WPARAM),
                ]

            class BUTTON_FIELDS(ctypes.Structure):
                _fields_ = [("usButtonFlags", wintypes.USHORT), ("usButtonData", wintypes.USHORT)]

            class BUTTONS(ctypes.Union):
                _anonymous_ = ("fields",)
                _fields_ = [("ulButtons", wintypes.ULONG), ("fields", BUTTON_FIELDS)]

            class RAWMOUSE(ctypes.Structure):
                _anonymous_ = ("buttons",)
                _fields_ = [
                    ("usFlags", wintypes.USHORT),
                    ("buttons", BUTTONS),
                    ("ulRawButtons", wintypes.ULONG),
                    ("lLastX", wintypes.LONG),
                    ("lLastY", wintypes.LONG),
                    ("ulExtraInformation", wintypes.ULONG),
                ]

            class RAWDATA(ctypes.Union):
                _fields_ = [("mouse", RAWMOUSE), ("padding", ctypes.c_byte * 24)]

            class RAWINPUT(ctypes.Structure):
                _fields_ = [("header", RAWINPUTHEADER), ("data", RAWDATA)]

            user32 = ctypes.windll.user32
            WM_INPUT = 0x00FF
            RID_INPUT = 0x10000003
            RIM_TYPEMOUSE = 0
            RIDEV_INPUTSINK = 0x00000100
            GWLP_WNDPROC = -4
            WNDPROC = ctypes.WINFUNCTYPE(
                ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
            )
            get_window_long = user32.GetWindowLongPtrW
            set_window_long = user32.SetWindowLongPtrW
            get_window_long.argtypes = [wintypes.HWND, ctypes.c_int]
            get_window_long.restype = ctypes.c_void_p
            set_window_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
            set_window_long.restype = ctypes.c_void_p
            user32.CallWindowProcW.argtypes = [
                ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
            ]
            user32.CallWindowProcW.restype = ctypes.c_ssize_t
            user32.GetRawInputData.argtypes = [
                wintypes.HANDLE, wintypes.UINT, ctypes.c_void_p,
                ctypes.POINTER(wintypes.UINT), wintypes.UINT,
            ]

            root.update_idletasks()
            self._hwnd = wintypes.HWND(root.winfo_id())
            self._old_proc = get_window_long(self._hwnd, GWLP_WNDPROC)
            if not self._old_proc:
                raise OSError("Could not read the recorder window procedure.")

            def handle_raw_input(lparam):
                size = wintypes.UINT(0)
                header_size = ctypes.sizeof(RAWINPUTHEADER)
                user32.GetRawInputData(lparam, RID_INPUT, None, ctypes.byref(size), header_size)
                if size.value < ctypes.sizeof(RAWINPUT):
                    return
                buffer = ctypes.create_string_buffer(size.value)
                copied = user32.GetRawInputData(lparam, RID_INPUT, buffer, ctypes.byref(size), header_size)
                if copied == 0xFFFFFFFF:
                    return
                raw = ctypes.cast(buffer, ctypes.POINTER(RAWINPUT)).contents
                # Absolute-coordinate devices are not useful as relative camera input.
                if raw.header.dwType == RIM_TYPEMOUSE and not (raw.data.mouse.usFlags & 0x0001):
                    dx, dy = int(raw.data.mouse.lLastX), int(raw.data.mouse.lLastY)
                    if dx or dy:
                        callback(dx, dy)

            @WNDPROC
            def wndproc(hwnd, message, wparam, lparam):
                if message == WM_INPUT:
                    handle_raw_input(lparam)
                return user32.CallWindowProcW(
                    ctypes.c_void_p(self._old_proc), hwnd, message, wparam, lparam
                )

            self._wndproc = wndproc
            set_window_long(self._hwnd, GWLP_WNDPROC, ctypes.cast(wndproc, ctypes.c_void_p))
            device = RAWINPUTDEVICE(0x01, 0x02, RIDEV_INPUTSINK, self._hwnd)
            if not user32.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(device)):
                raise ctypes.WinError()
            self.available = True
            self.error = ""
        except Exception as exc:
            self.error = str(exc)
            self.close()

    def close(self):
        if os.name == "nt" and self._hwnd and self._old_proc:
            try:
                import ctypes
                ctypes.windll.user32.SetWindowLongPtrW(
                    self._hwnd, -4, ctypes.c_void_p(self._old_proc)
                )
            except Exception:
                pass
        self.available = False
        self._wndproc = None
        self._old_proc = None


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
    right_mouse: int = 0
    camera_active: int = 0
    camera_yaw_delta_degrees: float = 0.0
    camera_pitch_delta_degrees: float = 0.0

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
        self.root.geometry("1020x840")
        self.root.minsize(840, 620)
        self.root.configure(bg=APP_BG)

        self.output_dir = Path.cwd() / "roblox_action_dataset"
        self.capture_region = None
        self.capture_presets = self._load_capture_presets()
        self.recording = False
        self.capture_thread = None
        self.keyboard_listener = None
        self.mouse_listener = None
        self.last_mouse_position = None
        self.pending_mouse_dx = 0.0
        self.pending_mouse_dy = 0.0
        self.pending_zoom = 0.0
        # Do not subclass Tk's native window procedure from Python. WM_INPUT can
        # invoke a ctypes callback without the GIL on Python 3.11, which can crash
        # the recorder while Pillow is processing a capture. Keep the stable
        # pynput listener path here; the camera diagnostics will flag missing data.
        self.raw_mouse_input = None
        self.raw_mouse_events = 0
        self.camera_labeled_frames = 0
        self.right_mouse_frames = 0
        self.session_frame_count = 0
        self.calibrating_camera = False
        self.calibration_raw_dx = 0.0
        self.calibration_raw_dy = 0.0
        self.action_lock = threading.Lock()
        self.action_state = ActionState()
        self.frame_index = 0
        self.start_time = 0.0
        self.preview_photo = None
        self.last_preview = None

        self._style()
        self._build()
        self.camera_source_var.set("Camera source: safe cursor listener (camera labels are checked after capture)")
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

        controls_shell = ttk.Frame(body, style="Card.TFrame")
        controls_shell.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        self.controls_canvas = tk.Canvas(controls_shell, bg=PANEL, highlightthickness=0, width=310)
        controls_scrollbar = ttk.Scrollbar(controls_shell, orient="vertical", command=self.controls_canvas.yview)
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        self.controls_canvas.grid(row=0, column=0, sticky="ns")
        controls_scrollbar.grid(row=0, column=1, sticky="ns")
        controls = ttk.Frame(self.controls_canvas, style="Card.TFrame", padding=14)
        self.controls_window = self.controls_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind("<Configure>", self._on_controls_configure)
        self.controls_canvas.bind("<Configure>", self._on_controls_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._scroll_controls, add="+")

        ttk.Label(controls, text="Capture Settings", style="CardHeading.TLabel").pack(anchor="w", pady=(0, 10))

        self.output_var = tk.StringVar(value=str(self.output_dir))
        ttk.Label(controls, text="Dataset output folder", style="Meta.TLabel").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.output_var, width=36, style="Field.TEntry").pack(fill="x", pady=4)
        ttk.Button(controls, text="Choose Output Folder", command=self.choose_output_folder).pack(fill="x", pady=3)

        self.region_var = tk.StringVar(value="No region selected")
        ttk.Label(controls, text="Capture region", style="Meta.TLabel").pack(anchor="w", pady=(10, 0))
        ttk.Label(controls, textvariable=self.region_var, style="Value.TLabel", wraplength=260).pack(anchor="w", pady=(3, 5))
        ttk.Button(controls, text="Select Game Region", command=self.select_region).pack(fill="x", pady=3)
        self.region_preset_var = tk.StringVar()
        ttk.Label(controls, text="Saved game regions", style="Meta.TLabel").pack(anchor="w", pady=(8, 0))
        self.region_preset_box = ttk.Combobox(controls, state="readonly", textvariable=self.region_preset_var)
        self.region_preset_box.pack(fill="x", pady=(3, 3))
        self.region_preset_box.bind("<<ComboboxSelected>>", self.apply_capture_preset)
        ttk.Button(controls, text="Use Saved Region", command=self.apply_capture_preset).pack(fill="x", pady=3)
        self._refresh_capture_preset_choices()

        self.fps_var = tk.IntVar(value=12)
        ttk.Label(controls, text="Capture FPS", style="Meta.TLabel").pack(anchor="w", pady=(10, 0))
        ttk.Spinbox(controls, from_=1, to=30, textvariable=self.fps_var, width=8).pack(fill="x", pady=4)

        self.resolution_var = tk.StringVar(value="256x144")
        ttk.Label(controls, text="Output frame size (16:9)", style="Meta.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            state="readonly",
            textvariable=self.resolution_var,
            values=["256x144", "512x288"],
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

        self.require_right_mouse_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="Only label camera turns while right mouse is held",
            variable=self.require_right_mouse_var,
        ).pack(anchor="w", pady=(2, 2))

        self.hide_while_recording_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="Hide recorder while recording (recommended)",
            variable=self.hide_while_recording_var,
        ).pack(anchor="w", pady=(6, 2))

        self.mouse_scale_var = tk.DoubleVar(value=120.0)  # legacy metadata compatibility
        self.yaw_counts_var = tk.DoubleVar(value=DEFAULT_YAW_COUNTS_PER_360)
        ttk.Label(controls, text="Horizontal mouse counts for a 360° turn", style="Meta.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Spinbox(controls, from_=100.0, to=20000.0, increment=100.0, textvariable=self.yaw_counts_var).pack(fill="x", pady=4)

        self.pitch_counts_var = tk.DoubleVar(value=DEFAULT_PITCH_COUNTS_PER_180)
        ttk.Label(controls, text="Vertical mouse counts for a 180° turn", style="Meta.TLabel").pack(anchor="w", pady=(4, 0))
        ttk.Spinbox(controls, from_=100.0, to=20000.0, increment=100.0, textvariable=self.pitch_counts_var).pack(fill="x", pady=4)
        ttk.Button(
            controls,
            text="Calibrate 360° Camera Turn (finish with F7)",
            command=self.start_camera_calibration,
        ).pack(fill="x", pady=(2, 4))

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

        self.camera_source_var = tk.StringVar(value="Camera source: checking...")
        ttk.Label(
            preview_card, textvariable=self.camera_source_var,
            style="Meta.TLabel", anchor="w",
        ).grid(row=3, column=0, sticky="ew", pady=(5, 0))

        # F8 stops recording and F7 finishes camera calibration, including while hidden.

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
            if key == keyboard.Key.f7:
                self.root.after(0, self.finish_camera_calibration)
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
            if (not self.recording and not self.calibrating_camera) or not self.record_mouse_var.get():
                self.last_mouse_position = (x, y)
                return
            # Raw Input supplies true relative counts even when Roblox locks the
            # pointer. Absolute cursor deltas are only a compatibility fallback.
            if self.raw_mouse_input is not None and self.raw_mouse_input.available:
                self.last_mouse_position = (x, y)
                return
            with self.action_lock:
                if self.last_mouse_position is not None:
                    dx = float(x - self.last_mouse_position[0])
                    dy = float(y - self.last_mouse_position[1])
                    if self.calibrating_camera:
                        self.calibration_raw_dx += dx
                        self.calibration_raw_dy += dy
                    if self.recording:
                        self.pending_mouse_dx += dx
                        self.pending_mouse_dy += dy
                self.last_mouse_position = (x, y)

        def on_scroll(_x, _y, _dx, dy):
            if not self.recording or not self.record_mouse_var.get():
                return
            with self.action_lock:
                self.pending_zoom += float(dy)

        def on_click(_x, _y, button, pressed):
            if button == mouse.Button.right:
                with self.action_lock:
                    self.action_state.right_mouse = 1 if pressed else 0
                self._queue_action_label()

        self.mouse_listener = mouse.Listener(on_move=on_move, on_scroll=on_scroll, on_click=on_click)
        self.mouse_listener.daemon = True
        self.mouse_listener.start()

    def _on_raw_mouse_delta(self, dx, dy):
        if (not self.recording and not self.calibrating_camera) or not self.record_mouse_var.get():
            return
        with self.action_lock:
            if self.calibrating_camera:
                self.calibration_raw_dx += float(dx)
                self.calibration_raw_dy += float(dy)
            if self.recording:
                self.pending_mouse_dx += float(dx)
                self.pending_mouse_dy += float(dy)
            self.raw_mouse_events += 1

    def start_camera_calibration(self):
        if self.recording:
            messagebox.showinfo("Stop recording first", "Finish the current recording before camera calibration.")
            return
        with self.action_lock:
            self.calibrating_camera = True
            self.calibration_raw_dx = 0.0
            self.calibration_raw_dy = 0.0
            self.last_mouse_position = None
        self.status_var.set("Camera calibration: rotate exactly 360°, then press F7")
        self.camera_source_var.set("Calibration armed: perform one complete horizontal camera turn")
        messagebox.showinfo(
            "Camera calibration armed",
            "Return to Roblox and rotate the camera exactly one complete 360° horizontal turn. "
            "You may hold right mouse while turning. Press F7 when the view reaches its original direction.",
        )
        self.root.withdraw()

    def finish_camera_calibration(self):
        if not self.calibrating_camera:
            return
        with self.action_lock:
            horizontal_counts = abs(self.calibration_raw_dx)
            self.calibrating_camera = False
            self.calibration_raw_dx = 0.0
            self.calibration_raw_dy = 0.0
        self._show_recorder()
        if horizontal_counts < 100.0:
            self.status_var.set("Camera calibration failed")
            messagebox.showwarning(
                "Camera calibration was too small",
                f"Only {horizontal_counts:.0f} horizontal counts were detected. Confirm Raw Input is ready, "
                "turn a complete 360°, and try again.",
            )
            return
        self.yaw_counts_var.set(round(horizontal_counts, 1))
        self.pitch_counts_var.set(round(horizontal_counts / 2.0, 1))
        self.status_var.set("Camera calibration ready")
        self.camera_source_var.set(
            f"Calibrated: {horizontal_counts:.0f} counts/360°; {horizontal_counts / 2.0:.0f} counts/180°"
        )
        messagebox.showinfo(
            "Camera calibration complete",
            f"Horizontal calibration: {horizontal_counts:.0f} counts for 360°.\n"
            f"Vertical scale: {horizontal_counts / 2.0:.0f} counts for 180°.\n\n"
            "These values will be written into the next dataset and saved into its trained model.",
        )

    def _queue_action_label(self):
        label = self.action_state.compact_label()
        if self.action_state.right_mouse:
            label = label + (" + RMB" if label != "IDLE" else "RMB")
        self.root.after(0, lambda: self.action_var.set(f"Input: {label}"))

    def choose_output_folder(self):
        path = filedialog.askdirectory(initialdir=self.output_var.get() or str(Path.cwd()))
        if path:
            self.output_var.set(path)

    def _load_capture_presets(self):
        try:
            values = json.loads(CAPTURE_PRESETS_FILE.read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                raise ValueError("saved regions must be an object")
            presets = {}
            for name, region in values.items():
                if not isinstance(name, str) or not isinstance(region, dict):
                    continue
                candidate = {key: int(region[key]) for key in ("left", "top", "width", "height")}
                if candidate["width"] > 0 and candidate["height"] > 0:
                    presets[name] = candidate
            return presets
        except FileNotFoundError:
            return {}
        except Exception:
            return {}

    def _save_capture_presets(self):
        temporary = CAPTURE_PRESETS_FILE.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.capture_presets, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(CAPTURE_PRESETS_FILE)

    def _refresh_capture_preset_choices(self):
        if hasattr(self, "region_preset_box"):
            self.region_preset_box["values"] = sorted(self.capture_presets, key=str.casefold)

    def _set_capture_region(self, region, preset_name=None):
        self.capture_region = {key: int(region[key]) for key in ("left", "top", "width", "height")}
        prefix = f"{preset_name}: " if preset_name else ""
        self.region_var.set(
            f"{prefix}x={self.capture_region['left']}, y={self.capture_region['top']}, "
            f"{self.capture_region['width']}×{self.capture_region['height']}"
        )

    def apply_capture_preset(self, _event=None):
        name = self.region_preset_var.get().strip()
        region = self.capture_presets.get(name)
        if not region:
            messagebox.showinfo("Choose a saved region", "Select a saved game region first.")
            return
        self._set_capture_region(region, name)
        self.status_var.set(f"Using saved region: {name}")

    def _offer_to_save_capture_region(self, region):
        if not messagebox.askyesno(
            "Save game region?",
            "Save this capture region for next time? You can choose it from Saved game regions.",
            parent=self.root,
        ):
            return
        name = simpledialog.askstring("Name saved region", "Name for this game region:", parent=self.root)
        if name is None:
            return
        name = name.strip()
        if not name:
            messagebox.showwarning("Name required", "The region was selected but not saved because it has no name.")
            return
        if name in self.capture_presets and not messagebox.askyesno(
            "Replace saved region?", f"Replace the existing saved region named {name!r}?", parent=self.root,
        ):
            return
        self.capture_presets[name] = {key: int(region[key]) for key in ("left", "top", "width", "height")}
        try:
            self._save_capture_presets()
        except Exception as exc:
            messagebox.showerror("Could not save region", str(exc))
            return
        self._refresh_capture_preset_choices()
        self.region_preset_var.set(name)
        self._set_capture_region(region, name)
        self.status_var.set(f"Saved game region: {name}")

    def select_region(self):
        self.root.withdraw()

        def selected(region):
            self.root.deiconify()
            self.root.lift()
            self._set_capture_region(region)
            self._offer_to_save_capture_region(region)

        selector = RegionSelector(self.root, selected)
        selector.protocol("WM_DELETE_WINDOW", lambda: (selector.destroy(), self.root.deiconify()))

    def _on_controls_configure(self, _event=None):
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _on_controls_canvas_configure(self, event):
        self.controls_canvas.itemconfigure(self.controls_window, width=event.width)

    def _scroll_controls(self, event):
        if not hasattr(self, "controls_canvas"):
            return
        x = self.root.winfo_pointerx() - self.controls_canvas.winfo_rootx()
        y = self.root.winfo_pointery() - self.controls_canvas.winfo_rooty()
        if 0 <= x < self.controls_canvas.winfo_width() and 0 <= y < self.controls_canvas.winfo_height():
            self.controls_canvas.yview_scroll(-max(1, event.delta // 120), "units")
            return "break"

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
        if self.calibrating_camera:
            messagebox.showinfo("Finish camera calibration", "Press F7 to finish the active 360° calibration first.")
            return
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
            width, height = (int(part.strip()) for part in self.resolution_var.get().lower().split("x", 1))
            delay = float(self.start_delay_var.get())
            yaw_counts = float(self.yaw_counts_var.get())
            pitch_counts = float(self.pitch_counts_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid settings",
                "FPS, a 16:9 frame size such as 256x144, delay, and camera calibration values must be numeric.",
            )
            return

        if fps < 1:
            messagebox.showerror("Invalid FPS", "Capture FPS must be at least 1.")
            return
        if width <= 0 or height <= 0 or width * 9 != height * 16 or width % 16 or height % 16:
            messagebox.showerror(
                "Invalid frame size",
                "Choose an exact 16:9 size whose dimensions are divisible by 16, such as 256x144 or 512x288.",
            )
            return
        if yaw_counts <= 0 or pitch_counts <= 0:
            messagebox.showerror("Invalid camera calibration", "Camera counts-per-turn values must be greater than zero.")
            return

        root, frames, metadata, config = self._dataset_paths()
        frames.mkdir(parents=True, exist_ok=True)

        existing = sorted(frames.glob("frame_*.png"))
        if existing:
            try:
                with Image.open(existing[0]) as sample:
                    existing_size = sample.size
            except Exception as exc:
                messagebox.showerror("Could not inspect dataset", f"Could not read {existing[0].name}: {exc}")
                return
            if existing_size != (width, height):
                messagebox.showerror(
                    "Dataset frame-size mismatch",
                    f"This folder already contains {existing_size[0]}x{existing_size[1]} frames, but the recorder "
                    f"is set to {width}x{height}. Choose a new dataset output folder so square and 16:9 frames "
                    "are never mixed together.",
                )
                return

        config_data = {
            "format_version": 3,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "capture_fps": fps,
            "output_resolution": f"{width}x{height}",
            "output_width": width,
            "output_height": height,
            "aspect_ratio": "16:9",
            "resize_method": self.crop_mode_var.get(),
            "action_dimensions": [
                "w", "a", "s", "d", "jump", "mouse_dx", "mouse_dy", "zoom", "move_x", "move_y",
                "right_mouse", "camera_active", "camera_yaw_delta_degrees", "camera_pitch_delta_degrees",
            ],
            "mouse_normalization_pixels": float(self.mouse_scale_var.get()),
            "camera_input_source": (
                "windows_raw_input" if self.raw_mouse_input and self.raw_mouse_input.available
                else "absolute_cursor_fallback"
            ),
            "camera_encoding": CAMERA_ENCODING,
            "yaw_counts_per_360_degrees": yaw_counts,
            "pitch_counts_per_180_degrees": pitch_counts,
            "max_yaw_degrees_per_frame": MAX_YAW_DEGREES_PER_FRAME,
            "max_pitch_degrees_per_frame": MAX_PITCH_DEGREES_PER_FRAME,
            "require_right_mouse_for_camera": bool(self.require_right_mouse_var.get()),
            "zoom_normalization_ticks": float(self.zoom_scale_var.get()),
            "transition_alignment": (
                "Each action row describes the held controls associated with that captured frame. "
                "For next-frame training, use row t as the action for frame t-1 -> frame t."
            ),
            "session_handling": "Each recording start writes a unique session_id. Training never crosses session boundaries.",
            "capture_region": self.capture_region,
        }
        config.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

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
            self.raw_mouse_events = 0
            self.camera_labeled_frames = 0
            self.right_mouse_frames = 0
            self.session_frame_count = 0
        self.recording = True
        # A new session prevents the trainer from learning a fake transition when
        # recording is stopped, the game world changes, and recording resumes.
        self.session_id = uuid.uuid4().hex
        self.session_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.record_btn.config(text="Stop Recording")
        self.status_var.set(f"Recording starts in {delay:.1f}s...")
        self.capture_thread = threading.Thread(
            target=self.capture_loop,
            args=(fps, (width, height), delay, frames, metadata),
            daemon=True
        )
        self.capture_thread.start()

    def stop_recording(self):
        self.recording = False
        self.record_btn.config(text="Start Recording")
        self.status_var.set("Recording stopped")
        self.root.after(0, self._show_recorder)
        self.root.after(200, self._show_camera_capture_summary)

    def _show_camera_capture_summary(self):
        if not self.record_mouse_var.get() or self.session_frame_count < 10:
            return
        source = "Raw Input" if self.raw_mouse_input and self.raw_mouse_input.available else "cursor fallback"
        self.camera_source_var.set(
            f"Camera: {source} | labelled {self.camera_labeled_frames}/{self.session_frame_count} frames | "
            f"RMB {self.right_mouse_frames} frames"
        )
        if self.right_mouse_frames >= 6 and self.camera_labeled_frames == 0:
            messagebox.showwarning(
                "No camera turns were labelled",
                "Right mouse was held, but the recorder captured no yaw/pitch movement. "
                "Do not train this session as camera data. Try running the recorder normally (not elevated), "
                "confirm Windows Raw Input says ready, and record a short camera calibration session first.",
            )

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

    def process_frame(self, image, frame_size):
        image = image.convert("RGB")
        target_width, target_height = frame_size

        if self.crop_mode_var.get() == "Letterbox":
            image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
            x = (target_width - image.width) // 2
            y = (target_height - image.height) // 2
            canvas.paste(image, (x, y))
            return canvas

        width, height = image.size
        source_ratio = width / height
        target_ratio = target_width / target_height
        if source_ratio > target_ratio:
            crop_width = round(height * target_ratio)
            left = (width - crop_width) // 2
            image = image.crop((left, 0, left + crop_width, height))
        elif source_ratio < target_ratio:
            crop_height = round(width / target_ratio)
            top = (height - crop_height) // 2
            image = image.crop((0, top, width, top + crop_height))
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

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

            with mss.MSS() as sct, metadata_path.open("a", encoding="utf-8") as metadata_file:
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
                        zoom_scale = max(1.0, float(self.zoom_scale_var.get()))
                        yaw_counts = max(1.0, float(self.yaw_counts_var.get()))
                        pitch_counts = max(1.0, float(self.pitch_counts_var.get()))
                        action = ActionState(**asdict(self.action_state))
                        raw_dx = self.pending_mouse_dx
                        raw_dy = self.pending_mouse_dy
                        if self.record_mouse_var.get():
                            camera_allowed = not self.require_right_mouse_var.get() or bool(action.right_mouse)
                            yaw_degrees = raw_dx * (360.0 / yaw_counts) if camera_allowed else 0.0
                            # Positive pitch means looking up; Windows mouse Y is positive downward.
                            pitch_degrees = -raw_dy * (180.0 / pitch_counts) if camera_allowed else 0.0
                            action.camera_yaw_delta_degrees = yaw_degrees
                            action.camera_pitch_delta_degrees = pitch_degrees
                            action.mouse_dx = max(-1.0, min(1.0, yaw_degrees / MAX_YAW_DEGREES_PER_FRAME))
                            action.mouse_dy = max(-1.0, min(1.0, pitch_degrees / MAX_PITCH_DEGREES_PER_FRAME))
                            action.zoom = max(-1.0, min(1.0, self.pending_zoom / zoom_scale))
                            action.camera_active = int(
                                abs(action.mouse_dx) > 0.002 or abs(action.mouse_dy) > 0.002
                            )
                        self.pending_mouse_dx = 0.0
                        self.pending_mouse_dy = 0.0
                        self.pending_zoom = 0.0
                        self.session_frame_count += 1
                        self.camera_labeled_frames += int(action.camera_active)
                        self.right_mouse_frames += int(action.right_mouse)

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
                        "mouse_raw_dx": round(raw_dx, 3),
                        "mouse_raw_dy": round(raw_dy, 3),
                        "camera_encoding": CAMERA_ENCODING,
                        "camera_yaw_delta_degrees": round(action.camera_yaw_delta_degrees, 6),
                        "camera_pitch_delta_degrees": round(action.camera_pitch_delta_degrees, 6),
                        "camera_active": action.camera_active,
                        "right_mouse": action.right_mouse,
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
                    camera_source = "Raw" if self.raw_mouse_input and self.raw_mouse_input.available else "Fallback"
                    self.root.after(
                        0,
                        lambda source=camera_source, yaw=action.camera_yaw_delta_degrees,
                               pitch=action.camera_pitch_delta_degrees, rmb=action.right_mouse:
                        self.camera_source_var.set(
                            f"Camera {source}: yaw {yaw:+.2f}° | pitch {pitch:+.2f}° | RMB {'held' if rmb else 'up'}"
                        )
                    )
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
        if self.raw_mouse_input is not None:
            self.raw_mouse_input.close()
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
