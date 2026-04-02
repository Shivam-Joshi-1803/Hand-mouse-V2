"""
Hand Gesture Mouse Controller v4.0  —  Right Hand (+ Optional Left Modifier)
=============================================================================
  ☝️  Index Only      →  Move Cursor
  🤏  Pinch (short)   →  Left Click
  🤏  Pinch (hold)    →  Drag & Drop
  ✌️  Victory / Peace →  Right Click
  👌  OK Sign         →  Double Click
  ✊  Fist            →  Scroll Up / Down
  👍  Thumb Up        →  Volume Up
  👎  Thumb Down      →  Volume Down
  🤟  Three Fingers   →  Alt+Tab
  🖐️  Open Palm       →  Neutral / Reset
  ─────────────────────────────────────
  P        →  Pause / Resume
  S        →  Settings Panel
  C        →  Calibration Mode
  Q / ESC  →  Quit
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import json
import os
import logging
import threading
import ctypes
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import tkinter as tk
from tkinter import ttk

# ── Optional: PIL for emoji rendering
try:
    from PIL import Image, ImageDraw, ImageFont
    _EMOJI_FONT_PATH = r"C:\Windows\Fonts\seguiemj.ttf"
    _emoji_font_sm = ImageFont.truetype(_EMOJI_FONT_PATH, 18) if os.path.exists(_EMOJI_FONT_PATH) else None
    _emoji_font_lg = ImageFont.truetype(_EMOJI_FONT_PATH, 22) if os.path.exists(_EMOJI_FONT_PATH) else None
    USE_PIL = _emoji_font_sm is not None
except ImportError:
    USE_PIL = False

# ── Optional: pystray for system tray
try:
    import pystray
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False

# ── Optional: winsound for audio feedback
try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

# ─────────────────────────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE    = os.path.join(_DIR, "hand_controller.log")
CONFIG_FILE = os.path.join(_DIR, "config.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("HandMouse")

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION  (persists to config.json)
# ─────────────────────────────────────────────────────────────────
@dataclass
class Cfg:
    # Camera
    cam_idx:     int   = 0
    cam_w:       int   = 1280
    cam_h:       int   = 720
    cam_fps:     int   = 30
    # MediaPipe
    det_conf:    float = 0.75
    trk_conf:    float = 0.65
    complexity:  int   = 1
    # Cursor mapping
    margin:      float = 0.25
    # Kalman
    kp:          float = 0.008
    km:          float = 0.055
    dead_px:     int   = 2
    # Gesture thresholds
    pinch_thr:   float = 0.33
    ok_thr:      float = 0.36
    vic_spread:  float = 0.22
    thumb_thr:   float = 0.12
    # State machine
    enter_f:     int   = 2
    exit_f:      int   = 3
    # Cooldowns
    cd_click:    float = 0.35
    cd_rclick:   float = 0.40
    cd_dclick:   float = 0.50
    cd_scroll:   float = 0.10
    cd_thumb:    float = 0.60
    cd_three:    float = 0.80
    # Drag
    drag_hold:   float = 1.20
    # Scroll
    scroll_sens: float = 5.0
    scroll_dead: float = 0.022
    scroll_pinch_guard: float = 0.70
    scroll_max:  int   = 40
    # UI
    sidebar_w:   int   = 200
    ui_alpha:    float = 0.62
    # Feature toggles
    use_sound:   bool  = True
    show_history: bool = True
    left_hand_mod: bool = False
    start_paused: bool = False

    def save(self, path: str = CONFIG_FILE):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
            log.info(f"Config saved → {path}")
        except Exception as e:
            log.error(f"Config save failed: {e}")

    @classmethod
    def load(cls, path: str = CONFIG_FILE) -> "Cfg":
        if not os.path.exists(path):
            log.info("No config.json — using defaults")
            return cls()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            obj = cls()
            for k, v in data.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            log.info(f"Config loaded ← {path}")
            return obj
        except Exception as e:
            log.error(f"Config load failed: {e} — using defaults")
            return cls()


C = Cfg.load()

# ─────────────────────────────────────────────────────────────────
#  PIL EMOJI HELPER
# ─────────────────────────────────────────────────────────────────
def put_emoji_text(frame, text: str, pos: Tuple[int, int],
                   font=None, color=(255, 255, 255)) -> np.ndarray:
    if not USE_PIL or font is None:
        return frame
    pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    r, g, b = color[2], color[1], color[0]
    draw.text(pos, text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ─────────────────────────────────────────────────────────────────
#  EMA LANDMARK SMOOTHER  (reduces jitter on raw landmarks)
# ─────────────────────────────────────────────────────────────────
class LandmarkEMA:
    """
    Exponential Moving Average smoother for 21 hand landmarks.
    Reduces per-frame jitter without adding latency like a window.
    alpha=1.0 = no smoothing.  alpha=0.3 = heavy smoothing.
    """
    def __init__(self, alpha: float = 0.45, n: int = 21):
        self.alpha = alpha
        self._buf: Optional[list] = None

    def update(self, landmarks):
        """Returns smoothed landmark list (same interface as hand_lm.landmark)."""
        if self._buf is None:
            self._buf = [(lm.x, lm.y, lm.z) for lm in landmarks]
            return landmarks
        a = self.alpha
        self._buf = [
            (a*lm.x + (1-a)*b[0],
             a*lm.y + (1-a)*b[1],
             a*lm.z + (1-a)*b[2])
            for lm, b in zip(landmarks, self._buf)
        ]
        return _EMAProxy(self._buf)

    def reset(self):
        self._buf = None


class _EMAProxy:
    """Wraps smoothed (x,y,z) tuples to look like MediaPipe landmarks."""
    class _Pt:
        __slots__ = ('x', 'y', 'z')
        def __init__(self, t): self.x, self.y, self.z = t
    def __init__(self, buf):
        self._pts = [self._Pt(t) for t in buf]
    def __getitem__(self, i): return self._pts[i]
    def __iter__(self): return iter(self._pts)
    def __len__(self): return len(self._pts)


# ─────────────────────────────────────────────────────────────────
#  KALMAN FILTER 1-D  (fixed: float precision, int only at moveTo)
# ─────────────────────────────────────────────────────────────────
class Kalman1D:
    def __init__(self, q, r):
        self.q = q; self.r = r
        self.x = 0.0; self.p = 1.0; self.ready = False

    def update(self, z: float) -> float:
        if not self.ready:
            self.x = z; self.ready = True; return z
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p = (1 - k) * self.p
        return self.x

    def reset(self):
        self.ready = False

# ─────────────────────────────────────────────────────────────────
#  GESTURE STATE MACHINE  (hysteresis)
# ─────────────────────────────────────────────────────────────────
class GSM:
    def __init__(self, enter=3, exit_n=4):
        self.enter = enter; self.exit_n = exit_n
        self.active = "neutral"
        self.cand = "neutral"; self.cand_c = 0; self.exit_c = 0

    def update(self, raw: str) -> str:
        if raw == self.active:
            self.exit_c = 0; self.cand = raw; self.cand_c = self.enter
        elif raw == self.cand:
            self.cand_c += 1; self.exit_c = 0
            if self.cand_c >= self.enter:
                self.active = self.cand
        else:
            self.exit_c += 1
            if self.exit_c >= self.exit_n:
                self.cand = raw; self.cand_c = 1; self.exit_c = 0
        return self.active

    @property
    def confidence(self) -> float:
        """0.0→1.0: how settled the current gesture is."""
        return min(self.cand_c / max(self.enter, 1), 1.0)

    def reset(self):
        self.active = "neutral"; self.cand = "neutral"
        self.cand_c = 0; self.exit_c = 0

# ─────────────────────────────────────────────────────────────────
#  LANDMARK INDICES  (MediaPipe hand 21-point model)
# ─────────────────────────────────────────────────────────────────
W   = 0                        # Wrist
T1  = 1;  T2  = 2;  T4  = 4   # Thumb CMC / Low / Tip
I5  = 5;  I6  = 6;  I8  = 8   # Index  MCP / PIP / Tip
M9  = 9;  M10 = 10; M12 = 12  # Middle MCP / PIP / Tip
R13 = 13; R14 = 14; R16 = 16  # Ring   MCP / PIP / Tip
P17 = 17; P18 = 18; P20 = 20  # Pinky  MCP / PIP / Tip

# ─────────────────────────────────────────────────────────────────
#  GESTURE CLASSIFIER  — precise, conflict-free
# ─────────────────────────────────────────────────────────────────
def _d(lm, a, b):
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

def _scale(lm):
    return _d(lm, W, M9) + 1e-6

def _nd(lm, a, b):
    return _d(lm, a, b) / _scale(lm)

def _up(lm, tip, pip):
    """True if finger tip is above its PIP joint (finger extended upward)."""
    return lm[tip].y < lm[pip].y

def _curl_score(lm, tip, mcp) -> float:
    """Normalized curl: positive = tip below MCP (curled), negative = extended."""
    return (lm[tip].y - lm[mcp].y) / (_scale(lm) + 1e-6)

def _strict_fist(lm) -> bool:
    """
    All four fingers clearly curled past their MCP joints.
    Uses BOTH MCP anchor (not just PIP) for a tight, unambiguous check.
    A loose fist or partially open hand will NOT pass.
    """
    CURL_MIN = 0.10   # tip must be this far below MCP in normalised units
    return (
        _curl_score(lm, I8,  I5)  > CURL_MIN and
        _curl_score(lm, M12, M9)  > CURL_MIN and
        _curl_score(lm, R16, R13) > CURL_MIN and
        _curl_score(lm, P20, P17) > CURL_MIN
    )

def _thumb_spread(lm) -> bool:
    """
    Thumb tip is clearly spread away from the index MCP (not tucked into fist).
    Required before thumb_up / thumb_down can fire.
    """
    return _nd(lm, T4, I5) > 0.38

def classify(lm) -> str:
    idx = _up(lm, I8,  I6)
    mid = _up(lm, M12, M10)
    rng = _up(lm, R16, R14)
    pky = _up(lm, P20, P18)

    ti = _nd(lm, T4, I8)
    im = _nd(lm, I8, M12)
    tm = _nd(lm, T4, M12)

    # ── 1. STRICT FIST — checked FIRST so it can never be confused
    #    with thumb_up/thumb_down when all fingers are curled.
    if _strict_fist(lm):
        return "fist"

    # ── 2. Thumb Up / Thumb Down
    #    Extra guard: thumb must be visibly spread (not tucked),
    #    ALL other fingers must be curled (using strict MCP check).
    fingers_curled = not idx and not mid and not rng and not pky
    if fingers_curled and _thumb_spread(lm):
        thumb_up_delta   = lm[W].y - lm[T4].y   # > 0  → tip above wrist
        thumb_down_delta = lm[T4].y - lm[W].y   # > 0  → tip below wrist
        if thumb_up_delta > C.thumb_thr and lm[T4].y < lm[T2].y:
            return "thumb_up"
        if thumb_down_delta > C.thumb_thr and lm[T4].y > lm[T2].y:
            return "thumb_down"

    # ── 3. Pinch (index + thumb close, other fingers relaxed)
    if ti < C.pinch_thr and not mid and not rng and not pky:
        return "pinch"

    # ── 4. OK sign
    if ti < C.ok_thr and mid and (rng or pky) and tm > 0.40:
        return "ok_sign"

    # ── 5. Three fingers (index + middle + ring up, pinky down)
    if idx and mid and rng and not pky:
        return "three_fingers"

    # ── 6. Index pointing — tip must be clearly above MCP, not just PIP
    if idx and not mid and not rng and not pky:
        if lm[I8].y < lm[I5].y - 0.025:
            return "index_only"

    # ── 7. Victory (two fingers spread)
    if idx and mid and not rng and not pky and im > C.vic_spread:
        return "victory"

    # ── 8. Open palm
    if sum([idx, mid, rng, pky]) >= 4:
        return "open_hand"

    return "neutral"

def classify_modifier(lm) -> str:
    """Left-hand modifier: fist=Ctrl, open=Alt, victory=Shift."""
    idx = _up(lm, I8,  I6)
    mid = _up(lm, M12, M10)
    rng = _up(lm, R16, R14)
    pky = _up(lm, P20, P18)
    if _strict_fist(lm):
        return "mod_ctrl"
    if sum([idx, mid, rng, pky]) >= 4:
        return "mod_alt"
    if idx and mid and not rng and not pky:
        return "mod_shift"
    return "mod_none"

# ─────────────────────────────────────────────────────────────────
#  UI DATA
# ─────────────────────────────────────────────────────────────────
GC = {
    "index_only":    (90,  220,  90),
    "pinch":         (90,  160, 255),
    "victory":       (255, 170,  60),
    "ok_sign":       (255,  90, 190),
    "fist":          (90,   90, 255),
    "open_hand":     (180, 180, 180),
    "thumb_up":      (60,  220, 120),
    "thumb_down":    (60,  120, 220),
    "three_fingers": (220, 180,  60),
    "neutral":       (110, 110, 110),
}

GUIDE = [
    ("index_only",    "☝️",  "1F", "Move Cursor"),
    ("pinch",         "🤏",  "PN", "Click / Drag"),
    ("victory",       "✌️",  "VC", "Right Click"),
    ("ok_sign",       "👌",  "OK", "Double Click"),
    ("fist",          "✊",  "FT", "Scroll"),
    ("thumb_up",      "👍",  "TU", "Volume Up"),
    ("thumb_down",    "👎",  "TD", "Volume Down"),
    ("three_fingers", "🤟",  "3F", "Alt+Tab"),
    ("open_hand",     "🖐️", "OP", "Neutral"),
]

# ─────────────────────────────────────────────────────────────────
#  VOLUME CONTROL  (Windows virtual key codes)
# ─────────────────────────────────────────────────────────────────
VK_VOL_UP   = 0xAF
VK_VOL_DOWN = 0xAE

def _press_vk(vk: int):
    try:
        ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.01)
        ctypes.windll.user32.keybd_event(vk, 0, 2, 0)
    except Exception as e:
        log.warning(f"VK press failed: {e}")

# ─────────────────────────────────────────────────────────────────
#  SOUND FEEDBACK
# ─────────────────────────────────────────────────────────────────
def _beep(freq: int = 800, dur: int = 40):
    if not C.use_sound or not HAS_SOUND:
        return
    threading.Thread(target=winsound.Beep, args=(freq, dur), daemon=True).start()

# ─────────────────────────────────────────────────────────────────
#  SETTINGS GUI  (Tkinter — runs in its own daemon thread)
# ─────────────────────────────────────────────────────────────────
class SettingsWindow:
    def __init__(self, cfg: Cfg, on_apply):
        self.cfg = cfg
        self.on_apply = on_apply
        self._open = False

    def open(self):
        if self._open:
            return
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        self._open = True
        win = tk.Tk()
        win.title("Hand Mouse  ⚙  Settings")
        win.geometry("440x560")
        win.resizable(False, False)
        win.configure(bg="#0d0f1a")
        win.protocol("WM_DELETE_WINDOW", lambda: self._close(win))

        style = ttk.Style(win)
        style.theme_use("clam")
        style.configure("TLabel",      background="#0d0f1a", foreground="#c0c8e0", font=("Segoe UI", 9))
        style.configure("TCheckbutton",background="#0d0f1a", foreground="#c0c8e0", font=("Segoe UI", 9))
        style.configure("TScale",      background="#0d0f1a")
        style.configure("Accent.TButton", background="#1a3a6a", foreground="#fff", font=("Segoe UI", 9, "bold"))
        style.map("Accent.TButton",    background=[("active", "#2456a4")])

        self._vars: dict = {}

        # Scrollable canvas
        outer = tk.Frame(win, bg="#0d0f1a")
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, bg="#0d0f1a", highlightthickness=0, bd=0)
        vsb    = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg="#0d0f1a")
        cwin  = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_cfg(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(cwin, width=canvas.winfo_width())
        inner.bind("<Configure>", _on_cfg)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(cwin, width=e.width))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        def section(label):
            tk.Frame(inner, bg="#253050", height=1).pack(fill="x", padx=10, pady=(10, 0))
            tk.Label(inner, text=f"  {label}", bg="#0d0f1a", fg="#4a90d9",
                     font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10)

        def slider(label, key, lo, hi, is_int=False, fmt=None):
            val = tk.DoubleVar(value=getattr(self.cfg, key))
            self._vars[key] = (val, is_int)
            row = tk.Frame(inner, bg="#0d0f1a"); row.pack(fill="x", padx=12, pady=2)
            tk.Label(row, text=label, bg="#0d0f1a", fg="#a0aac0", font=("Segoe UI", 9),
                     width=24, anchor="w").pack(side="left")
            disp_fmt = fmt or ("{:.0f}" if is_int else "{:.2f}")
            lbl = tk.Label(row, text=disp_fmt.format(val.get()), bg="#0d0f1a",
                           fg="#70b0ff", font=("Segoe UI", 9, "bold"), width=5)
            lbl.pack(side="right")
            def _upd(*_): lbl.config(text=disp_fmt.format(val.get()))
            ttk.Scale(row, from_=lo, to=hi, variable=val, orient="horizontal",
                      length=170, command=_upd).pack(side="right", padx=4)

        def check(label, key):
            val = tk.BooleanVar(value=getattr(self.cfg, key))
            self._vars[key] = (val, False)
            ttk.Checkbutton(inner, text=f"  {label}", variable=val).pack(anchor="w", padx=14, pady=2)

        # ── Sections ──────────────────────────────────────────────
        section("Gesture Thresholds")
        slider("Pinch threshold",     "pinch_thr",  0.15, 0.60)
        slider("OK sign threshold",   "ok_thr",     0.20, 0.60)
        slider("Victory spread",      "vic_spread", 0.10, 0.40)
        slider("Thumb threshold",     "thumb_thr",  0.05, 0.30)

        section("Timing & Cooldowns")
        slider("Click cooldown (s)",  "cd_click",   0.10, 1.0)
        slider("Drag hold time (s)",  "drag_hold",  0.3,  3.0)
        slider("Scroll cooldown (s)", "cd_scroll",  0.02, 0.5)
        slider("Volume cooldown (s)", "cd_thumb",   0.2,  2.0)

        section("Cursor & Scroll")
        slider("Screen margin",       "margin",     0.0,  0.20)
        slider("Scroll sensitivity",  "scroll_sens",1.0, 20.0)
        slider("Dead zone (px)",      "dead_px",    0,    12,   is_int=True)
        slider("Scroll max amount",   "scroll_max", 5,    80,   is_int=True)

        section("Features")
        check("Sound feedback",        "use_sound")
        check("Show action history",   "show_history")
        check("Left hand modifier",    "left_hand_mod")

        # ── Buttons ───────────────────────────────────────────────
        bf = tk.Frame(win, bg="#0d0f1a"); bf.pack(fill="x", padx=12, pady=8)

        def apply_cfg():
            for k, (v, is_int) in self._vars.items():
                raw = v.get()
                if is_int:
                    raw = int(round(raw))
                setattr(self.cfg, k, raw)
            self.on_apply()
            log.info("Settings applied (hot-reload)")

        def save_close():
            apply_cfg(); self.cfg.save(); self._close(win)

        ttk.Button(bf, text="✔  Apply", style="Accent.TButton", command=apply_cfg).pack(side="left",  padx=4)
        ttk.Button(bf, text="💾  Save & Close", style="Accent.TButton", command=save_close).pack(side="left", padx=4)
        ttk.Button(bf, text="✘  Cancel", command=lambda: self._close(win)).pack(side="right", padx=4)

        win.mainloop()
        self._open = False

    def _close(self, win):
        try:
            win.destroy()
        except Exception:
            pass
        self._open = False

# ─────────────────────────────────────────────────────────────────
#  SYSTEM TRAY  (optional pystray)
# ─────────────────────────────────────────────────────────────────
def _start_tray(controller):
    if not HAS_TRAY:
        return None
    try:
        from PIL import Image as _Img, ImageDraw as _Drw
        img = _Img.new("RGBA", (64, 64), (0, 0, 0, 0))
        d   = _Drw.Draw(img)
        d.ellipse([4, 4, 60, 60], fill=(70, 140, 255, 220))
        menu = pystray.Menu(
            pystray.MenuItem("Pause / Resume", lambda *_: controller.toggle_pause()),
            pystray.MenuItem("Settings",       lambda *_: controller.settings_win.open()),
            pystray.MenuItem("Quit",           lambda *_: controller.request_quit()),
        )
        icon = pystray.Icon("HandMouse", img, "Hand Mouse v4.0", menu)
        threading.Thread(target=icon.run, daemon=True).start()
        log.info("System tray started")
        return icon
    except Exception as e:
        log.warning(f"Tray failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────────
#  CONTROLLER
# ─────────────────────────────────────────────────────────────────
class HandMouseController:
    def __init__(self):
        self.mph = mp.solutions.hands
        self.hands = None
        self._init_mediapipe()

        self.sw, self.sh = pyautogui.size()
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE    = 0.002

        self.cap = None
        self._open_camera()

        self.kx = Kalman1D(C.kp, C.km)
        self.ky = Kalman1D(C.kp, C.km)
        self.cx, self.cy = pyautogui.position()

        self.gsm      = GSM(C.enter_f, C.exit_f)
        self.gsm_left = GSM(C.enter_f, C.exit_f)
        self.gesture  = "neutral"
        self.prev_ges = "neutral"
        self.left_mod = "mod_none"

        self.pinch_t0: Optional[float] = None
        self.dragging       = False
        self.pinch_clicked  = False
        self.last_pinch_end = 0.0

        self.scroll_ref: Optional[float] = None
        self.scroll_buf = deque(maxlen=6)

        self.t_click  = self.t_rclick = self.t_dclick = 0.0
        self.t_scroll = self.t_thumb  = self.t_three  = 0.0

        # ── Cross-gesture mutual exclusion timers ─────────────────
        # Prevents Vol and Scroll from co-firing (confirmed bug in logs)
        self.last_thumb_end  = 0.0   # when thumb_up/down last EXITED
        self.last_scroll_end = 0.0   # when fist/scroll last EXITED
        self.last_drag_end   = 0.0   # cooldown after drag drop
        _CROSS_BLOCK         = 0.9   # seconds each blocks the other
        self._CROSS_BLOCK    = _CROSS_BLOCK

        # ── EMA landmark smoother ─────────────────────────────────
        self.ema = LandmarkEMA(alpha=0.45)

        self.action_txt     = ""
        self.action_t       = 0.0
        self.action_history: deque = deque(maxlen=5)

        self.fps   = 0.0
        self.fps_n = 0
        self.fps_t = time.time()

        self.paused    = C.start_paused
        self.quit_flag = False

        self.settings_win = SettingsWindow(C, self._on_settings_apply)
        self._tray = _start_tray(self)

        log.info(self._welcome())

    # ── MediaPipe ─────────────────────────────────────────────────
    def _init_mediapipe(self):
        if self.hands:
            self.hands.close()
        self.hands = self.mph.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=C.det_conf,
            min_tracking_confidence=C.trk_conf,
            model_complexity=C.complexity,
        )

    # ── Camera ────────────────────────────────────────────────────
    def _open_camera(self) -> bool:
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(C.cam_idx)
        for prop, val in [
            (cv2.CAP_PROP_FRAME_WIDTH,  C.cam_w),
            (cv2.CAP_PROP_FRAME_HEIGHT, C.cam_h),
            (cv2.CAP_PROP_FPS,          C.cam_fps),
            (cv2.CAP_PROP_BUFFERSIZE,   1),
        ]:
            self.cap.set(prop, val)
        ok = self.cap.isOpened()
        if ok:
            self.fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info(f"Camera {C.cam_idx}: {self.fw}×{self.fh}")
        else:
            log.error(f"Cannot open camera {C.cam_idx}")
        return ok

    # ── Settings hot-reload ───────────────────────────────────────
    def _on_settings_apply(self):
        self.kx = Kalman1D(C.kp, C.km)
        self.ky = Kalman1D(C.kp, C.km)
        self._init_mediapipe()
        log.info("Settings hot-reloaded")

    # ── Screen mapping (Kalman on float → int at end) ─────────────
    def to_screen(self, lx, ly) -> Tuple[int, int]:
        m  = C.margin
        nx = float(np.clip((lx - m) / (1 - 2*m), 0.0, 1.0))
        
        # Shift Y-axis tracking upward: bottom of screen is reached at 70% down the camera frame
        # This ensures you never have to pull your wrist uncomfortably far back/down.
        bot_y = 0.70
        ny = float(np.clip((ly - m) / (max(0.01, bot_y - m)), 0.0, 1.0))
        sx = int(self.kx.update(nx * self.sw))
        sy = int(self.ky.update(ny * self.sh))
        if abs(sx - self.cx) < C.dead_px and abs(sy - self.cy) < C.dead_px:
            return self.cx, self.cy
        self.cx, self.cy = sx, sy
        return sx, sy

    # ── Helpers ───────────────────────────────────────────────────
    def _log(self, txt: str):
        self.action_txt = txt
        self.action_t   = time.time()
        self.action_history.appendleft(f"{time.strftime('%H:%M:%S')}  {txt}")
        log.info(txt)

    def _end_drag(self):
        if self.dragging:
            pyautogui.mouseUp()
            self.dragging  = False
            self.last_drag_end = time.time()
            self._log("🖱️  Drag → DROP")
            _beep(600, 60)

    def _reset_pinch(self):  self.pinch_t0 = None
    def _reset_scroll(self):
        self.scroll_ref = None
        self.scroll_buf.clear()

    def _reset_kalman(self):
        self.kx.reset(); self.ky.reset()
        self.ema.reset()

    def toggle_pause(self):
        self.paused = not self.paused
        _beep(400 if self.paused else 900, 80)
        log.info("PAUSED" if self.paused else "RESUMED")

    def request_quit(self):
        self.quit_flag = True

    # ── Execute gesture ───────────────────────────────────────────
    def execute(self, gesture: str, lm, now: float):
        prev = self.prev_ges
        CB   = self._CROSS_BLOCK

        # ── Track gesture group exits for cross-locking ───────────
        if prev in ("thumb_up", "thumb_down") and gesture not in ("thumb_up", "thumb_down"):
            self.last_thumb_end = now
        if prev == "fist" and gesture != "fist":
            self.last_scroll_end = now
            self._reset_scroll()  # always wipe on fist exit

        # ── Pinch ENTRY → click immediately
        if prev != "pinch" and gesture == "pinch":
            self.pinch_t0      = now
            self.pinch_clicked = False
            # Block click for post-drag cooldown (0.4s)
            if (not self.dragging
                    and now - self.t_click      > C.cd_click
                    and now - self.last_drag_end > 0.40):
                pyautogui.click()
                self.t_click       = now
                self.pinch_clicked = True
                self._log("🤏  Click")
                _beep(900, 35)

        # ── Pinch EXIT → drop drag if active
        elif prev == "pinch" and gesture != "pinch":
            if self.dragging:
                self._end_drag()
            self.last_pinch_end = now
            self._reset_pinch()

        # ── Index: move cursor
        if gesture == "index_only":
            tip = lm[I8]
            sx, sy = self.to_screen(tip.x, tip.y)
            pyautogui.moveTo(sx, sy)
            self._reset_scroll()

        # ── Pinch HOLD → drag
        elif gesture == "pinch":
            held   = (now - self.pinch_t0) if self.pinch_t0 else 0
            tip    = lm[I8]
            sx, sy = self.to_screen(tip.x, tip.y)
            if held >= C.drag_hold:
                if not self.dragging:
                    pyautogui.mouseDown()
                    self.dragging = True
                    self._log("🤏  Drag START")
                    _beep(500, 80)
                else:
                    pyautogui.moveTo(sx, sy)
            self._reset_scroll()

        # ── Victory: right click on entry
        elif gesture == "victory":
            if prev != "victory" and now - self.t_rclick > C.cd_rclick:
                pyautogui.rightClick()
                self.t_rclick = now
                self._log("✌️  Right Click")
                _beep(700, 40)

        # ── OK sign: double click on entry
        elif gesture == "ok_sign":
            if prev != "ok_sign" and now - self.t_dclick > C.cd_dclick:
                pyautogui.doubleClick()
                self.t_dclick = now
                self._log("👌  Double Click")
                _beep(900, 35)

        # ── Fist: scroll
        #    Cross-lock: blocked for CB seconds after any thumb gesture
        elif gesture == "fist":
            scroll_blocked = (
                now - self.last_pinch_end  < C.scroll_pinch_guard or
                now - self.last_thumb_end  < CB or        # ← NEW cross-lock
                now - self.last_drag_end   < 0.5           # ← NEW post-drag
            )
            if scroll_blocked:
                self._reset_scroll()
            else:
                wy = lm[W].y
                if self.scroll_ref is None:
                    self.scroll_ref = wy
                delta = self.scroll_ref - wy
                self.scroll_buf.append(delta)
                if len(self.scroll_buf) >= 3:
                    avg = float(np.mean(self.scroll_buf))
                    if abs(avg) > C.scroll_dead and now - self.t_scroll > C.cd_scroll:
                        amount = int(np.clip(abs(avg) * 150 * C.scroll_sens, 1, C.scroll_max))
                        up = avg > 0
                        pyautogui.scroll(amount if up else -amount)
                        self.t_scroll   = now
                        self.scroll_ref = wy
                        self.scroll_buf.clear()
                        self._log(f"✊  Scroll {'↑' if up else '↓'} ({amount})")

        # ── Thumb Up: volume up
        #    Cross-lock: blocked for CB seconds after scroll was active
        elif gesture == "thumb_up":
            if (prev != "thumb_up"
                    and now - self.t_thumb       > C.cd_thumb
                    and now - self.last_scroll_end > CB        # ← NEW cross-lock
                    and now - self.last_drag_end   > 0.5):     # ← NEW post-drag
                _press_vk(VK_VOL_UP)
                self.t_thumb = now
                self._log("👍  Volume Up")
                _beep(1000, 40)

        # ── Thumb Down: volume down
        elif gesture == "thumb_down":
            if (prev != "thumb_down"
                    and now - self.t_thumb       > C.cd_thumb
                    and now - self.last_scroll_end > CB
                    and now - self.last_drag_end   > 0.5):
                _press_vk(VK_VOL_DOWN)
                self.t_thumb = now
                self._log("👎  Volume Down")
                _beep(600, 40)

        # ── Three fingers: Alt+Tab
        elif gesture == "three_fingers":
            if prev != "three_fingers" and now - self.t_three > C.cd_three:
                pyautogui.hotkey('alt', 'tab')
                self.t_three = now
                self._log("🤟  Alt+Tab")
                _beep(750, 50)

        # ── Open palm / neutral: reset
        elif gesture in ("open_hand", "neutral"):
            self._reset_scroll()
            if prev not in ("open_hand", "neutral"):
                self._reset_kalman()

        # Don't reset Kalman during active drag or cursor movement
        if gesture not in ("index_only", "pinch") and not self.dragging:
            self._reset_kalman()

    # ── Draw skeleton ─────────────────────────────────────────────
    def draw_hand(self, frame, hand_lm):
        h, w = frame.shape[:2]
        col  = GC.get(self.gesture, (150, 150, 150))
        lm   = hand_lm.landmark

        for a, b in self.mph.HAND_CONNECTIONS:
            p1 = (int(lm[a].x*w), int(lm[a].y*h))
            p2 = (int(lm[b].x*w), int(lm[b].y*h))
            cv2.line(frame, p1, p2, col, 2, cv2.LINE_AA)

        for i, pt in enumerate(lm):
            cx2, cy2 = int(pt.x*w), int(pt.y*h)
            r = 7 if i in (4, 8, 12, 16, 20) else 4
            cv2.circle(frame, (cx2, cy2), r,   col,    -1, cv2.LINE_AA)
            cv2.circle(frame, (cx2, cy2), r+1, (0,0,0), 1, cv2.LINE_AA)

        if self.gesture == "index_only":
            ix, iy = int(lm[I8].x*w), int(lm[I8].y*h)
            cv2.circle(frame, (ix, iy), 22, col, 2, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy),  5, (255, 255, 255), -1, cv2.LINE_AA)

        elif self.gesture == "pinch":
            mx = int((lm[T4].x + lm[I8].x) / 2 * w)
            my = int((lm[T4].y + lm[I8].y) / 2 * h)
            if self.dragging:
                cv2.circle(frame, (mx, my), 28, col, -1, cv2.LINE_AA)
                cv2.putText(frame, "DRAG", (mx-26, my+46),
                            cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                prog = min((time.time() - self.pinch_t0) / C.drag_hold, 1.0) if self.pinch_t0 else 0
                cv2.circle(frame, (mx, my), 22, col, 2, cv2.LINE_AA)
                if self.pinch_clicked:
                    cv2.circle(frame, (mx, my), 8, col, -1, cv2.LINE_AA)
                    cv2.putText(frame, "CLICK", (mx-28, my-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
                if prog > 0.05:
                    cv2.ellipse(frame, (mx, my), (22, 22), -90, 0,
                                int(360*prog), (255, 180, 60), 3)
                    cv2.putText(frame, f"DRAG {int(prog*100)}%", (mx-34, my+44),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 180, 60), 1, cv2.LINE_AA)

        elif self.gesture == "fist":
            wx_, wy_ = int(lm[W].x*w), int(lm[W].y*h)
            guard = C.scroll_pinch_guard - (time.time() - self.last_pinch_end)
            if guard > 0:
                cv2.circle(frame, (wx_, wy_), 28, (60, 60, 60), 2, cv2.LINE_AA)
                cv2.putText(frame, f"wait {guard:.1f}s", (wx_-38, wy_-34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 100, 100), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (wx_, wy_), 28, col, 2, cv2.LINE_AA)

        elif self.gesture in ("thumb_up", "thumb_down"):
            tx, ty  = int(lm[T4].x*w), int(lm[T4].y*h)
            lbl = "▲ VOL+" if self.gesture == "thumb_up" else "▼ VOL-"
            cv2.circle(frame, (tx, ty), 20, col, -1, cv2.LINE_AA)
            cv2.putText(frame, lbl, (tx-38, ty-28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        elif self.gesture == "three_fingers":
            ix, iy = int(lm[I8].x*w), int(lm[I8].y*h)
            cv2.putText(frame, "ALT+TAB", (ix-40, iy-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 2, cv2.LINE_AA)

    # ── Draw UI ───────────────────────────────────────────────────
    def draw_ui(self, frame, has_hand: bool):
        h, w = frame.shape[:2]
        sw   = C.sidebar_w
        ov   = frame.copy()
        now  = time.time()

        # ── Sidebar background with subtle gradient vignette
        cv2.rectangle(ov, (0, 0), (sw, h), (6, 8, 16), -1)
        cv2.addWeighted(ov[:, :sw], C.ui_alpha,
                        frame[:, :sw], 1-C.ui_alpha, 0, frame[:, :sw])
        # Sidebar border — two-tone glow
        cv2.line(frame, (sw, 0), (sw, h), (50, 80, 160), 2)
        cv2.line(frame, (sw+1, 0), (sw+1, h), (20, 30, 60), 1)

        # ── Logo block
        cv2.putText(frame, "HAND",  (12, 30), cv2.FONT_HERSHEY_DUPLEX,  0.92, (70, 160, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "MOUSE", (12, 54), cv2.FONT_HERSHEY_DUPLEX,  0.85, (70, 160, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "v4.0",  (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (50, 90, 160),  1, cv2.LINE_AA)
        cv2.line(frame, (8, 78), (sw-8, 78), (30, 50, 100), 1)

        # ── Gesture guide rows with confidence bar on active row
        conf = self.gsm.confidence
        for i, (g, emoji, code, desc) in enumerate(GUIDE):
            yb  = 97 + i * 42
            col = GC.get(g, (150, 150, 150))
            act = (self.gesture == g)

            if act:
                # Background fill
                dim = tuple(max(0, int(v * 0.16)) for v in col)
                cv2.rectangle(frame, (3, yb-20), (sw-3, yb+22), dim, -1)
                # Border
                cv2.rectangle(frame, (3, yb-20), (sw-3, yb+22), col, 1)
                # Active indicator dot
                cv2.circle(frame, (sw-11, yb+1), 5, col, -1, cv2.LINE_AA)
                # Confidence bar (bottom of row)
                bar_x0, bar_y  = 8, yb + 17
                bar_w = int((sw - 16) * conf)
                cv2.rectangle(frame, (bar_x0, bar_y), (sw-8, bar_y+3), (20, 24, 40), -1)
                cv2.rectangle(frame, (bar_x0, bar_y), (bar_x0 + bar_w, bar_y+3), col, -1)

            tc = col if act else tuple(int(v * 0.40) for v in col)
            dc = col if act else (68, 68, 68)

            if USE_PIL:
                font = _emoji_font_lg if act else _emoji_font_sm
                frame = put_emoji_text(frame, emoji, (10, yb-15), font, tc)
                cv2.putText(frame, desc, (42, yb+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44 if act else 0.38, dc, 1, cv2.LINE_AA)
            else:
                ts = 0.56 if act else 0.46
                wt = 2 if act else 1
                cv2.putText(frame, code, (12, yb+5),
                            cv2.FONT_HERSHEY_DUPLEX, ts, tc, wt, cv2.LINE_AA)
                cv2.putText(frame, desc, (44, yb+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.41 if act else 0.35, dc, 1, cv2.LINE_AA)

        # Top bar
        cv2.rectangle(ov, (sw, 0), (w, 52), (10, 12, 20), -1)
        cv2.addWeighted(ov[0:52, sw:], C.ui_alpha,
                        frame[0:52, sw:], 1-C.ui_alpha, 0, frame[0:52, sw:])

        # PAUSED overlay
        if self.paused:
            pause_ov = frame.copy()
            cv2.rectangle(pause_ov, (sw, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(pause_ov[0:h, sw:], 0.6,
                            frame[0:h, sw:], 0.4, 0, frame[0:h, sw:])
            cx_p = sw + (w - sw) // 2
            cv2.putText(frame, "PAUSED", (cx_p-110, h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 2.0, (80, 80, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "Press  P  to resume", (cx_p-130, h//2 + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (110, 110, 220), 1, cv2.LINE_AA)

        # ── Status badges (top bar)
        badge_x = sw + 10

        # Hand detection badge — pulsing border when active
        hc    = (80, 220, 80) if has_hand else (50, 50, 50)
        hc_bg = (14, 30, 14) if has_hand else (16, 16, 18)
        ht    = "RIGHT HAND" if has_hand else "No Hand"
        cv2.rectangle(frame, (badge_x, 8), (badge_x+170, 42), hc_bg, -1)
        cv2.rectangle(frame, (badge_x, 8), (badge_x+170, 42), hc, 1)
        # Pulse dot
        pulse_r = 5 + int(2 * math.sin(now * 6)) if has_hand else 4
        cv2.circle(frame, (badge_x+14, 25), pulse_r, hc, -1, cv2.LINE_AA)
        cv2.putText(frame, ht, (badge_x+28, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, hc, 1, cv2.LINE_AA)
        badge_x += 180

        # Current gesture badge
        if self.gesture not in ("neutral",):
            gcol = GC.get(self.gesture, (120, 120, 120))
            gname = self.gesture.replace("_", " ").upper()
            tw2, _ = cv2.getTextSize(gname, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0]
            bw = tw2 + 22
            cv2.rectangle(frame, (badge_x, 8), (badge_x+bw, 42),
                          tuple(max(0,int(v*0.12)) for v in gcol), -1)
            cv2.rectangle(frame, (badge_x, 8), (badge_x+bw, 42), gcol, 1)
            cv2.putText(frame, gname, (badge_x+10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, gcol, 1, cv2.LINE_AA)
            badge_x += bw + 8

        # Drag badge
        if self.dragging:
            cv2.rectangle(frame, (badge_x, 8), (badge_x+100, 42), (14, 36, 90), -1)
            cv2.rectangle(frame, (badge_x, 8), (badge_x+100, 42), (80, 140, 255), 1)
            cv2.putText(frame, "DRAGGING", (badge_x+6, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 160, 255), 1, cv2.LINE_AA)
            badge_x += 108

        # Left modifier badge
        if C.left_hand_mod and self.left_mod != "mod_none":
            _mlbl = {"mod_ctrl": "CTRL", "mod_alt": "ALT", "mod_shift": "SHIFT"}
            ml = _mlbl.get(self.left_mod, "")
            if ml:
                cv2.rectangle(frame, (badge_x, 8), (badge_x+68, 42), (36, 14, 60), -1)
                cv2.rectangle(frame, (badge_x, 8), (badge_x+68, 42), (180, 80, 255), 1)
                cv2.putText(frame, ml, (badge_x+8, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 110, 255), 1, cv2.LINE_AA)

        # FPS — right-aligned
        fps_s = f"{self.fps:.0f} FPS"
        tw, _ = cv2.getTextSize(fps_s, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1)[0]
        cv2.putText(frame, fps_s, (w-tw-12, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (100, 100, 120), 1, cv2.LINE_AA)

        # Action flash (fades)
        age = now - self.action_t
        if age < 1.4:
            af = max(0.0, 1.0 - age/1.4)
            fc = tuple(int(v*af) for v in (80, 240, 180))
            if USE_PIL:
                frame = put_emoji_text(frame, self.action_txt, (sw+340, 10), _emoji_font_lg, fc)
            else:
                cv2.putText(frame, self.action_txt, (sw+330, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, fc, 2, cv2.LINE_AA)

        # Action history
        if C.show_history and self.action_history:
            for j, hist_txt in enumerate(self.action_history):
                af = max(0.10, 1.0 - j * 0.22)
                hcol = (int(200*af), int(220*af), int(180*af))
                cv2.putText(frame, hist_txt, (sw+12, h - 40 - j*18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, hcol, 1, cv2.LINE_AA)

        # Control zone indicator
        if has_hand and not self.paused:
            m  = C.margin
            x0 = sw + int(m*(w-sw)); y0 = int(m*h)
            x1 = w  - int(m*(w-sw)); y1 = h - int(m*h)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (50, 90, 50), 1)
            cv2.putText(frame, "Control Zone", (x0+4, y0+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (50, 90, 50), 1, cv2.LINE_AA)

        # Bottom help bar
        cv2.rectangle(ov, (sw, h-28), (w, h), (10, 12, 20), -1)
        cv2.addWeighted(ov[h-28:h, sw:], C.ui_alpha,
                        frame[h-28:h, sw:], 1-C.ui_alpha, 0, frame[h-28:h, sw:])
        cv2.putText(frame, "P=Pause  S=Settings  Q/ESC=Quit",
                    (sw+12, h-9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90, 90, 90), 1, cv2.LINE_AA)

        return frame

    # ── Always-on-top helper (Windows only) ───────────────────────
    @staticmethod
    def _set_always_on_top(win_title: str):
        """Pin the OpenCV window to always sit above other windows."""
        try:
            HWND_TOPMOST   = -1
            SWP_NOSIZE     = 0x0001
            SWP_NOMOVE     = 0x0002
            SWP_NOACTIVATE = 0x0010
            hwnd = ctypes.windll.user32.FindWindowW(None, win_title)
            if hwnd:
                ctypes.windll.user32.SetWindowPos(
                    hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE
                )
                log.info("Window pinned always-on-top")
        except Exception as e:
            log.warning(f"always-on-top failed: {e}")

    # ── Main loop ─────────────────────────────────────────────────
    def run(self):
        WIN_TITLE = "Hand Mouse Controller"
        cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_TITLE, 780, 480)   # compact window

        # Wait one frame so the OS creates the handle, then pin it
        cv2.waitKey(1)
        self._set_always_on_top(WIN_TITLE)

        reconnect_t = 0.0
        no_cam_msg  = False

        while not self.quit_flag:
            # ── Camera reconnect logic ─────────────────────────────
            if self.cap is None or not self.cap.isOpened():
                now = time.time()
                if now - reconnect_t > 3.0:
                    log.warning("Camera lost — attempting reconnect …")
                    self._open_camera()
                    reconnect_t = now
                # Show blank frame with message
                blank = np.zeros((self.fh if hasattr(self, 'fh') else 480,
                                  self.fw if hasattr(self, 'fw') else 640, 3), np.uint8)
                cv2.putText(blank, "Camera disconnected — retrying …",
                            (40, 240), cv2.FONT_HERSHEY_DUPLEX, 0.9, (60, 60, 255), 2)
                cv2.imshow(WIN_TITLE, blank)
                if cv2.waitKey(200) & 0xFF in (ord('q'), ord('Q'), 27):
                    break
                continue

            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                continue

            frame   = cv2.flip(frame, 1)
            now     = time.time()

            right_hand_lm = None
            left_hand_lm  = None

            if not self.paused:
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_lm, handedness in zip(results.multi_hand_landmarks,
                                                   results.multi_handedness):
                        label = handedness.classification[0].label
                        if label == "Right":
                            right_hand_lm = hand_lm
                        elif label == "Left":
                            left_hand_lm = hand_lm

            has_hand = right_hand_lm is not None

            if has_hand and not self.paused:
                # Apply EMA smoother before classification
                lm            = self.ema.update(right_hand_lm.landmark)
                raw           = classify(lm)
                self.prev_ges = self.gesture
                self.gesture  = self.gsm.update(raw)
                self.execute(self.gesture, lm, now)
                self.draw_hand(frame, right_hand_lm)

                # Left-hand modifier
                if C.left_hand_mod and left_hand_lm:
                    raw_left   = classify_modifier(left_hand_lm.landmark)
                    self.left_mod = self.gsm_left.update(raw_left)
                    self.draw_hand(frame, left_hand_lm)
            else:
                if not self.paused:
                    self.prev_ges = self.gesture
                    self.gesture  = self.gsm.update("neutral")
                    self._end_drag()
                    self._reset_pinch()
                    self._reset_scroll()
                    self._reset_kalman()
                    cv2.putText(frame, "Show RIGHT hand",
                                (C.sidebar_w + 20, 110),
                                cv2.FONT_HERSHEY_DUPLEX, 0.80, (60, 60, 60), 2, cv2.LINE_AA)

            # FPS counter
            self.fps_n += 1
            dt = now - self.fps_t
            if dt >= 1.0:
                self.fps   = self.fps_n / dt
                self.fps_n = 0
                self.fps_t = now

            frame = self.draw_ui(frame, has_hand)
            cv2.imshow(WIN_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('p'), ord('P')):
                self.toggle_pause()
            elif key in (ord('s'), ord('S')):
                self.settings_win.open()

        self.cleanup()

    def cleanup(self):
        self._end_drag()
        if self.hands:
            self.hands.close()          # FIX: was missing — MediaPipe resource leak
        if self.cap and self.cap.isOpened():
            self.cap.release()
        C.save()                        # Auto-save config on exit
        cv2.destroyAllWindows()
        log.info("Stopped. Config auto-saved.")

    @staticmethod
    def _welcome():
        return (
            "\n╔══════════════════════════════════════════════════════╗\n"
            "║   Hand Mouse Controller  v4.0  —  Right Hand Only    ║\n"
            "╠══════════════════════════════════════════════════════╣\n"
            "║  ☝️  Index Only      →  Move Cursor                   ║\n"
            "║  🤏  Pinch (short)   →  Left Click                    ║\n"
            "║  🤏  Pinch (hold)    →  Drag & Drop                   ║\n"
            "║  ✌️  Victory         →  Right Click                   ║\n"
            "║  👌  OK Sign         →  Double Click                  ║\n"
            "║  ✊  Fist            →  Scroll Up / Down              ║\n"
            "║  👍  Thumb Up        →  Volume Up                     ║\n"
            "║  👎  Thumb Down      →  Volume Down                   ║\n"
            "║  🤟  Three Fingers   →  Alt+Tab                       ║\n"
            "║  🖐️  Open Palm       →  Neutral / Reset               ║\n"
            "║  P → Pause   S → Settings   Q/ESC → Quit             ║\n"
            "╚══════════════════════════════════════════════════════╝\n"
        )


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        HandMouseController().run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("Done.")