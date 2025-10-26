# bot.py - Universal Game Automation Bot core
import cv2
import numpy as np
import subprocess
import time
import random
import yaml
import re
import os
import copy
import logging
import sqlite3
from logging.handlers import TimedRotatingFileHandler
from collections import Counter, deque, defaultdict
import sys
from pathlib import Path


# --- TK GUI ---
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox, simpledialog
import threading
import queue
from datetime import datetime
from PIL import Image, ImageTk
from typing import Dict, Any, List, Optional, Set, Tuple
import json
import ast
import ctypes

from task_engine import (
    TaskManager,
    TaskDefinition,
    TaskResult,
    load_tasks_from_dict,
    TaskError,
)


# --- Keyboard hook for pause/resume ---
try:
    import keyboard
except ImportError:
    print("The 'keyboard' package is required for the pause feature.")
    print("Install it via: pip install keyboard")
    sys.exit(1)

# --- Windows control helpers ---
try:
    from PIL import ImageGrab
    import pygetwindow as gw
    import pyautogui
except ImportError:
    # Delay warning until Windows mode is actually requested.
    pass

# --- Farbige Konsolenausgabe (Optional) ---
try:
    import colorama

    colorama.just_fix_windows_console()
    GREEN = colorama.Fore.GREEN
    YELLOW = colorama.Fore.YELLOW
    RED = colorama.Fore.RED
    CYAN = colorama.Fore.CYAN
    BOLD = colorama.Style.BRIGHT
    RESET = colorama.Style.RESET_ALL
except ImportError:
    # Fallback, falls colorama nicht installiert ist
    GREEN = YELLOW = RED = CYAN = BOLD = RESET = ""

# --- Profile management ---
BASE_DIR = Path(__file__).resolve().parent
PROFILES_DIR = BASE_DIR / "profiles"
ACTIVE_PROFILE_FILE = PROFILES_DIR / "active_profile.txt"
DEFAULT_PROFILE_NAME = "lastwar"
DATA_DIR = BASE_DIR / "data"
STATS_DB_PATH = DATA_DIR / "stats.db"
STATS_DB: Optional[sqlite3.Connection] = None


def init_stats_storage() -> None:
    global STATS_DB
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        STATS_DB = sqlite3.connect(
            STATS_DB_PATH,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
        )
        STATS_DB.execute(
            """
            CREATE TABLE IF NOT EXISTS series_events (
                metric_key TEXT NOT NULL,
                ts REAL NOT NULL,
                value REAL NOT NULL
            )
            """
        )
        STATS_DB.execute(
            "CREATE INDEX IF NOT EXISTS idx_series_events_key_ts ON series_events(metric_key, ts)"
        )
        STATS_DB.commit()
    except Exception as exc:
        logging.getLogger(__name__).error("Failed to initialize stats storage: %s", exc)
        STATS_DB = None


def prune_old_series_data(cutoff_ts: float) -> None:
    if not STATS_DB:
        return
    try:
        STATS_DB.execute("DELETE FROM series_events WHERE ts < ?", (cutoff_ts,))
        STATS_DB.commit()
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to prune stats storage: %s", exc)


def ensure_profiles_dir() -> None:
    """Make sure the profiles directory exists."""
    try:
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def list_profiles() -> List[str]:
    """Return sorted list of profile names that ship with config.yaml."""
    ensure_profiles_dir()
    names = []
    try:
        for entry in PROFILES_DIR.iterdir():
            if entry.is_dir() and (entry / "config.yaml").exists():
                names.append(entry.name)
    except FileNotFoundError:
        pass
    return sorted(names)


def read_active_profile() -> Optional[str]:
    """Read the currently active profile name."""
    ensure_profiles_dir()
    available = list_profiles()
    if ACTIVE_PROFILE_FILE.exists():
        try:
            name = ACTIVE_PROFILE_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            name = ""
        if name in available:
            return name
    if DEFAULT_PROFILE_NAME in available:
        return DEFAULT_PROFILE_NAME
    return available[0] if available else None


def write_active_profile(name: str) -> None:
    """Persist the active profile selection."""
    ensure_profiles_dir()
    ACTIVE_PROFILE_FILE.write_text(name.strip(), encoding="utf-8")


ACTIVE_PROFILE_NAME = read_active_profile()
PROFILE_ROOT = PROFILES_DIR / ACTIVE_PROFILE_NAME if ACTIVE_PROFILE_NAME else None


def resolve_profile_path(path_str: str) -> Optional[str]:
    """Resolve a path relative to the active profile (or repo base)."""
    if not path_str:
        return None
    expanded = Path(os.path.expanduser(path_str))
    if expanded.is_absolute():
        return str(expanded)
    base = PROFILE_ROOT if PROFILE_ROOT else BASE_DIR
    return str((base / expanded).resolve())


def make_profile_relative(path_str: str) -> str:
    """Return a profile-relative path when possible."""
    if not path_str:
        return path_str
    try:
        base = PROFILE_ROOT if PROFILE_ROOT else BASE_DIR
        return str(Path(path_str).resolve().relative_to(base))
    except Exception:
        return path_str


def resolve_template_path(path_str: str) -> Optional[str]:
    """Resolve template-relative paths, preferring the profile templates directory."""
    if not path_str:
        return None
    expanded = Path(os.path.expanduser(path_str))
    if expanded.is_absolute():
        return str(expanded)
    # Try relative to the configured template directory first
    try:
        candidate = PROFILE_TEMPLATES_DIR / expanded
        if candidate.exists():
            return str(candidate.resolve())
    except Exception:
        pass
    # Fall back to general profile-relative path
    return resolve_profile_path(path_str)


# Translation resources for multi-language support
TRANSLATIONS: Dict[str, Dict[str, Any]] = {}
TRANSLATION_STRINGS: Dict[str, Dict[str, str]] = {}
TRANSLATION_COMMENTS: Dict[str, Dict[str, str]] = {}
TRANSLATION_META: Dict[str, Dict[str, Any]] = {}


def _deep_merge_dict(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``source`` into ``target``."""
    for key, value in source.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge_dict(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load a YAML file, returning an empty dict on failure."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Failed to read translation file %s: %s", path, exc
        )
    return {}


def load_translations() -> Dict[str, Dict[str, Any]]:
    """Load core and profile-specific translations from YAML files."""
    translations: Dict[str, Dict[str, Any]] = {}

    # Core translations in repository root
    base_dir = BASE_DIR / "translations"
    if base_dir.exists():
        for path in sorted(base_dir.glob("*.yaml")):
            if not path.is_file():
                continue
            lang = path.stem.lower()
            translations[lang] = _load_yaml_file(path)

    # Profile-specific overrides
    if PROFILE_ROOT:
        profile_dir = PROFILE_ROOT / "translations"
        if profile_dir.exists():
            for path in sorted(profile_dir.glob("*.yaml")):
                if not path.is_file():
                    continue
                lang = path.stem.lower()
                base = translations.setdefault(lang, {})
                overrides = _load_yaml_file(path)
                _deep_merge_dict(base, overrides)

    # Ensure at least English entry exists
    translations.setdefault("en", {"strings": {}})
    return translations


TRANSLATIONS = load_translations()
for lang_code, payload in TRANSLATIONS.items():
    strings = payload.get("strings", {})
    comments = payload.get("setting_comments", {})
    meta = payload.get("meta", {})
    TRANSLATION_STRINGS[lang_code] = (
        strings if isinstance(strings, dict) else {}
    )
    TRANSLATION_COMMENTS[lang_code] = (
        comments if isinstance(comments, dict) else {}
    )
    TRANSLATION_META[lang_code] = meta if isinstance(meta, dict) else {}

AVAILABLE_LANG_CODES: List[str] = sorted(TRANSLATION_STRINGS.keys())


def get_language_display_name(code: str) -> str:
    """Return a human-friendly language name."""
    meta = TRANSLATION_META.get(code, {})
    label = meta.get("name")
    if isinstance(label, str) and label.strip():
        return label
    return code.upper()


def translate_key(lang: str, key: str) -> str:
    """Return the translated string for ``key`` with fallback to English."""
    lang = (lang or "en").lower()
    strings = TRANSLATION_STRINGS.get(lang, {})
    if key in strings:
        return strings[key]
    return TRANSLATION_STRINGS.get("en", {}).get(key, key)


# Regex removing ANSI escape codes when calculating string length
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Global pause flag
is_paused = False
bot_is_running = True


# --- Logging Konfiguration ---
def setup_logging():
    """Konfiguriert das Logging-System."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger("LastWarBot")
    logger.setLevel(logging.DEBUG)

    # Verhindert, dass Logs mehrfach ausgegeben werden, falls das Skript neu geladen wird
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter for log messages
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Daily rotating file handler keeping the last 30 days
    handler = TimedRotatingFileHandler(
        os.path.join(log_dir, "bot.log"),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    handler.setFormatter(log_format)
    logger.addHandler(handler)

    # Console output handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger


LOGGER = setup_logging()


# --- Config laden & Hilfsfunktionen ---
def _is_comment_wrapper(node: dict) -> bool:
    """Check if a dict represents a value/comment wrapper."""
    if not isinstance(node, dict):
        return False
    keys = set(node.keys())
    return "value" in node and keys.issubset({"value", "comment"})


def _separate_config_comments(node, path=(), comments=None):
    """Split config node into pure values plus comment dictionary."""
    if comments is None:
        comments = {}

    if isinstance(node, dict):
        if _is_comment_wrapper(node):
            key_path = ".".join(path)
            comments[key_path] = node.get("comment", "")
            value = node.get("value")
            if isinstance(value, dict):
                return _separate_config_comments(value, path, comments)
            if isinstance(value, list):
                return copy.deepcopy(value)
            return value

        result = {}
        for key, value in node.items():
            sub_path = path + (key,)
            result[key] = _separate_config_comments(value, sub_path, comments)
        return result

    if isinstance(node, list):
        # Lists are treated as plain data; deep-copy to avoid accidental mutation.
        return copy.deepcopy(node)

    return copy.deepcopy(node)


def separate_config_comments(data):
    """Return (values, comments) tuple extracted from a YAML config structure."""
    comments = {}
    values = _separate_config_comments(data, (), comments)
    return values, comments


def merge_config_with_comments(values, comments, path=()):
    """Merge values with comments back into YAML-serializable structure."""
    if isinstance(values, dict):
        result = {}
        for key, value in values.items():
            sub_path = path + (key,)
            result[key] = merge_config_with_comments(value, comments, sub_path)

        key_path = ".".join(path)
        comment_text = comments.get(key_path, "")
        if key_path and comment_text:
            return {"value": result, "comment": comment_text}
        return result

    if isinstance(values, list):
        return copy.deepcopy(values)

    key_path = ".".join(path)
    comment_text = comments.get(key_path, "")
    if comment_text:
        return {"value": values, "comment": comment_text}
    return copy.deepcopy(values)

DEFAULT_SETTING_COMMENTS = copy.deepcopy(TRANSLATION_COMMENTS.get("en", {}))

CFG_COMMENTS = {}


def load_config(path="config.yaml"):
    """Load the configuration file from disk."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        values, comments = separate_config_comments(raw)
        # Populate default comments for missing entries without mutating saved data yet.
        for key, text in DEFAULT_SETTING_COMMENTS.items():
            comments.setdefault(key, text)
        global CFG_COMMENTS
        CFG_COMMENTS = comments
        return values
    except FileNotFoundError:
        LOGGER.error(f"Configuration file not found at: {path}. Bot will exit.")
        sys.exit(1)
    except Exception as e:
        LOGGER.error(f"Failed to load configuration file: {e}")
        sys.exit(1)


def persist_language_to_config(path: str, language: str) -> None:
    """Persist the selected GUI language into the config YAML."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}
    except Exception as exc:
        LOGGER.error("Unable to read config for language persistence: %s", exc)
        return

    lang_node = data.get("language")
    if isinstance(lang_node, dict):
        lang_node["value"] = language
    else:
        data["language"] = {"value": language}

    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    except Exception as exc:
        LOGGER.error("Unable to persist language selection: %s", exc)

CONFIG_PATH = (resolve_profile_path("config.yaml") if PROFILE_ROOT else str((BASE_DIR / "config.yaml").resolve()))
if not CONFIG_PATH or not Path(CONFIG_PATH).exists():
    CONFIG_PATH = str((BASE_DIR / "config.yaml").resolve())
CFG = load_config(CONFIG_PATH)
ACTIVE_LANGUAGE = str(CFG.get("language", "en")).lower()
if ACTIVE_LANGUAGE in TRANSLATION_COMMENTS:
    for key, text in TRANSLATION_COMMENTS[ACTIVE_LANGUAGE].items():
        CFG_COMMENTS.setdefault(key, text)

# Cached absolute directories for the active profile
PROFILE_TEMPLATES_DIR = Path(
    resolve_profile_path(CFG.get("template_dir", "templates") or "templates")
)
PROFILE_TASKS_PATH = resolve_profile_path(
    CFG.get("tasks_file", "tasks.yaml") or "tasks.yaml"
)
try:
    PROFILE_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

init_stats_storage()

# Lade Konfigurationswerte mit Fallback-Defaults
JITTER = CFG.get("jitter_px", 8)
BOT_TITLE = CFG.get("bot_title", "Universal Game Automation Bot")

for strings in TRANSLATION_STRINGS.values():
    strings["title"] = BOT_TITLE

# Template matching thresholds
THRESH = CFG.get("threshold_default", 0.87)
THRESH_LOOSE = CFG.get("threshold_loose", 0.84)

# This dictionary will now store the entire template object from the config
TEMPLATES_CONFIG = CFG.get("templates", {})
LOADED_TEMPLATES = {}  # This will store the loaded image data

# Task engine configuration
TASKS_PATH = PROFILE_TASKS_PATH or str((BASE_DIR / "tasks.yaml").resolve())
TASKS_CONFIG: Dict[str, Any] = {}
TASK_MANAGER: Optional[TaskManager] = None
TASKS_MTIME: float = 0.0
CONFIG_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*config\.([a-zA-Z0-9_\.]+)\s*\}\}")
STATS: Optional["Stats"] = None
MISSING_CRITICAL_TASKS: Set[str] = set()


def _get_config_value(path: str) -> Any:
    """Resolve a dotted config key (e.g. 'threshold_loose') against CFG."""
    parts = [p for p in path.split(".") if p]
    value: Any = CFG
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return copy.deepcopy(value)


def _resolve_config_placeholders(text: str) -> Any:
    """Replace config placeholders within a string."""
    stripped = text.strip()
    match = CONFIG_PLACEHOLDER_PATTERN.fullmatch(stripped)
    if match:
        resolved = _get_config_value(match.group(1))
        if resolved is None:
            LOGGER.warning(
                "Unknown configuration placeholder '%s' in tasks file.",
                match.group(1),
            )
        return resolved

    def repl(m):
        resolved = _get_config_value(m.group(1))
        if resolved is None:
            LOGGER.warning(
                "Unknown configuration placeholder '%s' in tasks file.",
                m.group(1),
            )
            return ""
        return str(resolved)

    return CONFIG_PLACEHOLDER_PATTERN.sub(repl, text)


def _resolve_task_structure(data: Any) -> Any:
    """Recursively replace config placeholders for runtime execution."""
    if isinstance(data, dict):
        return {k: _resolve_task_structure(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_task_structure(item) for item in data]
    if isinstance(data, str):
        return _resolve_config_placeholders(data)
    return copy.deepcopy(data)


def _build_task_action_env() -> Dict[str, Any]:
    """Return the runtime environment provided to task actions."""
    return {
        "logger": LOGGER,
        "find_and_tap": find_and_tap,
        "wait_and_tap": wait_and_tap,
        "template_exists": template_exists,
        "swipe": swipe,
        "key_back": key_back,
        "threshold_default": CFG.get("threshold_default", 0.85),
        "threshold_loose": CFG.get("threshold_loose", 0.8),
    }


def load_task_definitions(path: Optional[str]) -> None:
    """Load workflow definitions from YAML and initialise the task manager."""
    global TASKS_CONFIG, TASK_MANAGER, TASKS_MTIME, PROTECTED_TASK_IDS

    if not path:
        LOGGER.warning("No tasks file configured; task automation disabled.")
        TASKS_CONFIG = {"version": 1, "tasks": []}
        TASK_MANAGER = None
        TASKS_MTIME = 0.0
        return

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw_data = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        LOGGER.warning(f"Tasks file not found at {path}. No workflows loaded.")
        TASKS_CONFIG = {"version": 1, "tasks": []}
        TASK_MANAGER = None
        TASKS_MTIME = 0.0
        return
    except Exception as exc:
        LOGGER.error(f"Failed to read tasks file {path}: {exc}")
        TASKS_CONFIG = {"version": 1, "tasks": []}
        TASK_MANAGER = None
        TASKS_MTIME = 0.0
        return

    TASKS_CONFIG = copy.deepcopy(raw_data)
    resolved_data = _resolve_task_structure(raw_data)
    try:
        task_definitions = load_tasks_from_dict(resolved_data)
    except TaskError as exc:
        LOGGER.error(f"Invalid task configuration in {path}: {exc}")
        TASK_MANAGER = None
        return

    attr_protected = {
        task.id
        for task in task_definitions
        if getattr(task, "protected_system_task", False)
    }
    if attr_protected:
        PROTECTED_TASK_IDS = attr_protected
    else:
        PROTECTED_TASK_IDS = {
            task.id for task in task_definitions if task.id in DEFAULT_PROTECTED_TASKS
        }

    action_env = _build_task_action_env()
    TASK_MANAGER = TaskManager(task_definitions, action_env)
    TASK_MANAGER.prepare()
    if STATS:
        try:
            STATS.configure_tasks(task_definitions)
        except Exception as exc:
            LOGGER.warning("Failed to configure stats metadata: %s", exc)

    try:
        TASKS_MTIME = os.path.getmtime(path)
    except OSError:
        TASKS_MTIME = time.time()

    LOGGER.info(
        f"Loaded {len(task_definitions)} workflow(s) from {os.path.basename(path)}."
    )


def maybe_reload_tasks(path: Optional[str]) -> None:
    """Reload tasks when the backing YAML file has changed."""
    global TASKS_MTIME

    if not path:
        return
    try:
        current_mtime = os.path.getmtime(path)
    except OSError:
        return

    if TASKS_MTIME == 0.0 or current_mtime > TASKS_MTIME:
        LOGGER.info("Detected update to tasks configuration. Reloading...")
        load_task_definitions(path)


# GUI-Konfiguration
GUI_ENABLED = CFG.get("gui_enabled", False)
GUI_SHOW_ROI = CFG.get("gui_show_roi", False)
GUI_HISTORY_SIZE = CFG.get("gui_history_size", 15)
gui_heatmap_alpha = CFG.get("gui_heatmap_alpha", 0.6)
GUI_HISTORY_ONLY_SUCCESSFUL = CFG.get("gui_history_only_successful", False)
GUI_INITIAL_WIDTH = CFG.get("gui_initial_width")
GUI_INITIAL_HEIGHT = CFG.get("gui_initial_height")


# --- Control abstraction layer ---
class ControlInterface:
    """Interface for different control backends."""

    def __init__(self):
        self.screen_width = 0
        self.screen_height = 0

    def screenshot_bgr(self):
        """Return a BGR screenshot."""
        raise NotImplementedError

    def tap(self, x, y, jitter=0):
        """Perform a click at the given coordinates."""
        raise NotImplementedError

    def swipe(self, x1, y1, x2, y2, dur_ms=400):
        """Perform a swipe gesture."""
        raise NotImplementedError

    def key_back(self):
        """Simulate the platform-specific 'back' action."""
        raise NotImplementedError

    def get_screen_size(self):
        """Return the target screen size."""
        if self.screen_width == 0 or self.screen_height == 0:
            screen = self.screenshot_bgr()
            if screen is not None:
                self.screen_height, self.screen_width = screen.shape[:2]
        return self.screen_width, self.screen_height


class AdbControl(ControlInterface):
    """Control implementation that drives Android via ADB."""

    def __init__(self, serial):
        super().__init__()
        self.serial = serial
        self.adb_path = ["adb"] + (["-s", self.serial] if self.serial else [])
        LOGGER.info(
            f"ADB controller initialised for device: {self.serial or 'default'}"
        )

    def _adb_cmd(self, args, binary=False, timeout=15):
        try:
            res = subprocess.run(
                self.adb_path + args, capture_output=True, timeout=timeout, check=True
            )
            return res.stdout if binary else res.stdout.decode(errors="ignore")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode(errors="ignore").strip()
            LOGGER.error(
                f"ADB command failed: {' '.join(self.adb_path + args)}. Error: {error_msg}"
            )
            raise RuntimeError(f"ADB error: {error_msg}")
        except subprocess.TimeoutExpired:
            LOGGER.error(f"ADB command timed out: {' '.join(self.adb_path + args)}")
            raise RuntimeError("ADB Timeout")

    def screenshot_bgr(self):
        try:
            raw = self._adb_cmd(["exec-out", "screencap", "-p"], binary=True)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("ADB screenshot could not be decoded (empty image).")
            return img
        except Exception as e:
            LOGGER.error(f"ADB screenshot failed: {e}")
            raise

    def tap(self, x, y, jitter=JITTER):
        dx = random.randint(-jitter, jitter)
        dy = random.randint(-jitter, jitter)
        final_x, final_y = max(1, x + dx), max(1, y + dy)
        LOGGER.debug(f"ADB tap at ({final_x}, {final_y})")
        self._adb_cmd(["shell", "input", "tap", str(final_x), str(final_y)])
        time.sleep(random.uniform(0.3, 0.7))

    def swipe(self, x1, y1, x2, y2, dur_ms=400):
        LOGGER.debug(f"ADB swipe from ({x1}, {y1}) to ({x2}, {y2}) in {dur_ms}ms")
        self._adb_cmd(
            ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(dur_ms)]
        )
        time.sleep(random.uniform(0.5, 1.0))

    def key_back(self):
        LOGGER.debug("Sending Android BACK key via ADB")
        self._adb_cmd(["shell", "input", "keyevent", "4"])
        time.sleep(random.uniform(1.2, 1.8))


class WindowsControl(ControlInterface):
    """Control implementation that targets a Windows application window."""

    def __init__(self, window_title):
        super().__init__()
        self.window = self._find_window(window_title)
        if self.window is None:
            raise RuntimeError(
                f"Window with title '{window_title}' could not be found."
            )

        try:
            if self.window.isMinimized:
                self.window.restore()
            self.window.activate()
            time.sleep(0.5)
            LOGGER.info(
                f"Windows controller initialised for window '{self.window.title}'."
            )
        except Exception as e:
            LOGGER.warning(
                "Unable to bring the window to the foreground: %s. Ensure the script has sufficient permissions and the window is visible.",
                e,
            )

    def _find_window(self, title):
        try:
            windows = gw.getWindowsWithTitle(title)
            if windows:
                return windows[0]
        except Exception as e:
            LOGGER.error(f"Error locating window with pygetwindow: {e}")
        return None

    def screenshot_bgr(self):
        try:
            if not self.window or not self.window.visible:
                self.window = self._find_window(self.window.title)
                if not self.window:
                    raise RuntimeError("Game window lost and could not be found again.")

            bbox = (
                self.window.left,
                self.window.top,
                self.window.right,
                self.window.bottom,
            )
            img_pil = ImageGrab.grab(bbox=bbox, all_screens=True)
            img_np = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            return img_bgr
        except Exception as e:
            LOGGER.error(f"Windows screenshot failed: {e}")
            # Try to reactivate the window
            self.window.activate()
            time.sleep(0.5)
            raise

    def tap(self, x, y, jitter=JITTER):
        dx = random.randint(-jitter, jitter)
        dy = random.randint(-jitter, jitter)

        abs_x = self.window.left + x + dx
        abs_y = self.window.top + y + dy

        LOGGER.debug(f"Windows click at ({abs_x}, {abs_y})")
        pyautogui.click(abs_x, abs_y)
        time.sleep(random.uniform(0.3, 0.7))

    def swipe(self, x1, y1, x2, y2, dur_ms=400):
        start_x, start_y = self.window.left + x1, self.window.top + y1
        end_x, end_y = self.window.left + x2, self.window.top + y2
        duration_sec = dur_ms / 1000.0

        LOGGER.debug(
            f"Windows swipe from ({start_x}, {start_y}) to ({end_x}, {end_y}) in {duration_sec:.2f}s"
        )
        pyautogui.moveTo(start_x, start_y)
        pyautogui.dragTo(end_x, end_y, duration=duration_sec, button="left")
        time.sleep(random.uniform(0.5, 1.0))

    def key_back(self):
        LOGGER.debug("Sending ESCAPE key press to Windows client")
        pyautogui.press("esc")
        time.sleep(random.uniform(1.2, 1.8))


# --- Control factory ---
def get_control_instance(config):
    """Create and return the proper control backend."""
    mode = config.get("control_mode", "android")

    if mode == "windows":
        try:
            # Import lazily so Windows dependencies are only required when needed
            from PIL import ImageGrab
            import pygetwindow as gw
            import pyautogui
        except ImportError:
            LOGGER.error(
                "Windows mode requires the packages Pillow, pygetwindow, and pyautogui."
            )
            LOGGER.error("Install them with: pip install Pillow pygetwindow pyautogui")
            sys.exit(1)

        window_title = config.get("windows_title")
        if not window_title:
            LOGGER.error(
                "'control_mode: windows' requires 'windows_title' to be set in config.yaml."
            )
            sys.exit(1)
        return WindowsControl(window_title)

    elif mode == "android":
        serial = config.get("device_serial")
        return AdbControl(serial)

    else:
        raise ValueError(
            f"Unknown control_mode: '{mode}'. Expected 'android' or 'windows'."
        )


# Global control backend instance
control = get_control_instance(CFG)


# --- Wrapper around control backends ---
def tap(x, y, jitter=JITTER):
    """Wrapper redirecting to control.tap."""
    control.tap(x, y, jitter)


def longtap(x, y, duration=2000, jitter=JITTER):
    """Simulate a long tap via swipe with identical start and end points."""
    dx = random.randint(-jitter, jitter)
    dy = random.randint(-jitter, jitter)
    final_x = max(1, int(x + dx))
    final_y = max(1, int(y + dy))

    try:
        dur_value = float(duration)
    except (TypeError, ValueError):
        LOGGER.warning(
            "Invalid long-tap duration %r for (%s, %s); falling back to default.",
            duration,
            final_x,
            final_y,
        )
        dur_value = 2000.0

    dur_ms = max(1, int(round(dur_value)))
    LOGGER.debug(
        "Executing long tap at (%s, %s) for %d ms (jitter=%d).",
        final_x,
        final_y,
        dur_ms,
        jitter,
    )
    control.swipe(final_x, final_y, final_x, final_y, dur_ms)


def swipe(x1, y1, x2, y2, dur_ms=400):
    """Wrapper redirecting to control.swipe."""
    control.swipe(x1, y1, x2, y2, dur_ms)


def key_back():
    """Wrapper redirecting to control.key_back."""
    control.key_back()


def screenshot_bgr():
    """Wrapper redirecting to control.screenshot_bgr."""
    return control.screenshot_bgr()


# --- GUI Tabs ---


class MonitorTab(ttk.Frame):
    """Tab Monitor with OpenCV overlay and heatmap visualization"""

    def __init__(self, parent, gui_manager):
        super().__init__(parent)
        self.gui_manager = gui_manager
        self.current_image = None
        self.photo_image = None

        self.setup_ui()

    def setup_ui(self):
        # Control panel at the top
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.pause_btn = ttk.Button(
            control_frame, text=self.gui_manager.t("pause"), command=self.toggle_pause
        )
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            control_frame, text=self.gui_manager.t("stop"), command=self.stop_bot
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = ttk.Label(
            control_frame,
            text=f"{self.gui_manager.t('status')}: {self.gui_manager.t('running')}",
        )
        self.status_label.pack(side=tk.RIGHT, padx=5)

        # Canvas for image display
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg="black")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Inform the OpenCV renderer about the current available size in the Monitor tab
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        # Hook mouse interactions: left-click for pause/step buttons; wheel for sidebar scroll
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<MouseWheel>", self._on_canvas_wheel)

        scrollbar_y = ttk.Scrollbar(
            canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        scrollbar_x = ttk.Scrollbar(
            self, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.configure(
            xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set
        )

        self.is_paused = False

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.configure(text=self.gui_manager.t("resume"))
            self.status_label.configure(
                text=f"{self.gui_manager.t('status')}: {self.gui_manager.t('paused')}"
            )
            self.gui_manager.send_command("pause")
        else:
            self.pause_btn.configure(text=self.gui_manager.t("pause"))
            self.status_label.configure(
                text=f"{self.gui_manager.t('status')}: {self.gui_manager.t('running')}"
            )
            self.gui_manager.send_command("resume")

    def stop_bot(self):
        self.status_label.configure(
            text=f"{self.gui_manager.t('status')}: {self.gui_manager.t('stopped')}"
        )
        self.gui_manager.send_command("stop")

    def update_image(self, image_data):
        """Update canvas with new image"""
        try:
            if isinstance(image_data, str):
                # Load from file path
                image = Image.open(image_data)
            else:
                # Assume PIL Image object
                image = image_data

            # Resize to fit current canvas while keeping aspect ratio
            cw = max(1, self.canvas.winfo_width())
            ch = max(1, self.canvas.winfo_height())
            if cw > 1 and ch > 1:
                # Create a copy to avoid mutating original
                img_copy = image.copy()
                img_copy.thumbnail((cw, ch), Image.Resampling.LANCZOS)
                image = img_copy

            self.current_image = image
            self.photo_image = ImageTk.PhotoImage(image)

            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        except Exception as e:
            print(f"Error updating image: {e}")

    def set_paused(self, paused: bool):
        """Set paused UI state to match global paused state."""
        self.is_paused = bool(paused)
        if self.is_paused:
            self.pause_btn.configure(text=self.gui_manager.t("resume"))
            self.status_label.configure(
                text=f"{self.gui_manager.t('status')}: {self.gui_manager.t('paused')}"
            )
        else:
            self.pause_btn.configure(text=self.gui_manager.t("pause"))
            self.status_label.configure(
                text=f"{self.gui_manager.t('status')}: {self.gui_manager.t('running')}"
            )

    def refresh_texts(self):
        """Refresh visible texts according to current language."""
        if self.is_paused:
            self.pause_btn.configure(text=self.gui_manager.t("resume"))
            self.status_label.configure(
                text=f"{self.gui_manager.t('status')}: {self.gui_manager.t('paused')}"
            )
        else:
            self.pause_btn.configure(text=self.gui_manager.t("pause"))
            self.status_label.configure(
                text=f"{self.gui_manager.t('status')}: {self.gui_manager.t('running')}"
            )
        self.stop_btn.configure(text=self.gui_manager.t("stop"))

    # --- MonitorTab integration points for the OpenCV renderer ---
    def _on_canvas_resize(self, event):
        """Forward available drawing area to the OpenCV renderer (Tk backend)."""
        try:
            from builtins import globals as _g  # noqa: F401 (document intent)
        except Exception:
            pass
        # Use global gui_controller if available
        try:
            global gui_controller
            if gui_controller is not None and hasattr(
                gui_controller, "set_target_size"
            ):
                gui_controller.set_target_size(event.width, event.height)
        except Exception:
            pass

    def _on_canvas_click(self, event):
        """Send click to the OpenCV overlay to trigger pause/step buttons."""
        try:
            global gui_controller
            if gui_controller is not None and hasattr(
                gui_controller, "process_click_from_tk"
            ):
                gui_controller.process_click_from_tk(event.x, event.y)
        except Exception:
            pass

    def _on_canvas_wheel(self, event):
        """Send wheel scroll to the overlay sidebar (Windows uses event.delta ±120)."""
        try:
            global gui_controller
            if gui_controller is not None and hasattr(
                gui_controller, "process_wheel_from_tk"
            ):
                gui_controller.process_wheel_from_tk(event.delta, event.x, event.y)
        except Exception:
            pass


class DashboardTab(ttk.Frame):
    """Tab Statistics dashboard"""

    def __init__(self, parent, gui_manager):
        super().__init__(parent)
        self.gui_manager = gui_manager
        self.stats = {"actions": 0, "success": 0, "runtime": 0}
        self.dynamic_counter_vars: Dict[str, tk.StringVar] = {}
        self._dynamic_counter_keys: List[str] = []
        self.metrics_placeholder: Optional[ttk.Label] = None
        self.metric_series_data: Dict[str, Dict[str, Any]] = {}
        self.metric_key_map: Dict[str, str] = {}
        self.metric_combo_var = tk.StringVar()
        self.metric_range_var = tk.StringVar()
        self.range_definitions: List[Tuple[str, Optional[int]]] = [
            ("metric_range_1h", 3600),
            ("metric_range_6h", 21600),
            ("metric_range_24h", 86400),
            ("metric_range_7d", 604800),
            ("metric_range_all", None),
        ]
        self._scroll_bound = False
        self.setup_ui()

    def setup_ui(self):
        self._canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.configure(yscrollcommand=scrollbar.set)

        self._content = ttk.Frame(self._canvas)
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._content, anchor=tk.NW
        )
        self._content.columnconfigure(0, weight=1)
        self._content.rowconfigure(0, weight=1)
        self._content.rowconfigure(1, weight=1)
        self._content.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.bind(
            "<Configure>",
            lambda e: self._canvas.itemconfigure(self._canvas_window, width=e.width),
        )
        self._canvas.bind("<Enter>", lambda _e: self._bind_scroll())
        self._canvas.bind("<Leave>", lambda _e: self._unbind_scroll())

        top_frame = ttk.Frame(self._content)
        top_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)
        top_frame.rowconfigure(0, weight=1)

        stats_frame = ttk.LabelFrame(
            top_frame, text=self.gui_manager.t("statistics"), padding=10
        )
        stats_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ttk.Label(
            stats_frame,
            text=self.gui_manager.t("actions_performed") + ":",
            font=("Arial", 10, "bold"),
        ).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.actions_label = ttk.Label(stats_frame, text="0", font=("Arial", 10))
        self.actions_label.grid(row=0, column=1, sticky=tk.W, padx=20, pady=5)

        ttk.Label(
            stats_frame,
            text=self.gui_manager.t("runtime") + ":",
            font=("Arial", 10, "bold"),
        ).grid(row=3, column=0, sticky=tk.W, pady=5)
        self.runtime_label = ttk.Label(stats_frame, text="00:00:00", font=("Arial", 10))
        self.runtime_label.grid(row=3, column=1, sticky=tk.W, padx=20, pady=5)

        self.metrics_frame = ttk.LabelFrame(
            stats_frame,
            text=self.gui_manager.t("task_metrics"),
            padding=5,
        )
        self.metrics_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=(10, 0))
        self.metrics_frame.columnconfigure(0, weight=1)
        self.metrics_frame.columnconfigure(1, weight=1)
        self.dynamic_stats_container = ttk.Frame(self.metrics_frame)
        self.dynamic_stats_container.grid(row=0, column=0, columnspan=2, sticky=tk.EW)
        self._show_metrics_placeholder()

        self.charts_frame = ttk.LabelFrame(
            top_frame, text=self.gui_manager.t("metric_history"), padding=10
        )
        self.charts_frame.grid(row=0, column=1, sticky="nsew")
        selector_frame = ttk.Frame(self.charts_frame)
        selector_frame.pack(fill=tk.X, pady=(0, 6))
        self.metric_select_label = ttk.Label(
            selector_frame, text=self.gui_manager.t("metric_select_label")
        )
        self.metric_select_label.pack(side=tk.LEFT)
        self.metric_combo = ttk.Combobox(
            selector_frame,
            textvariable=self.metric_combo_var,
            state="readonly",
            width=28,
        )
        self.metric_combo.pack(side=tk.LEFT, padx=5)
        self.metric_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._draw_selected_series()
        )
        self.metric_range_label = ttk.Label(
            selector_frame, text=self.gui_manager.t("metric_range_label")
        )
        self.metric_range_label.pack(side=tk.LEFT, padx=(12, 0))
        self.metric_range_combo = ttk.Combobox(
            selector_frame,
            textvariable=self.metric_range_var,
            state="readonly",
            width=16,
        )
        self.metric_range_combo.pack(side=tk.LEFT, padx=5)
        self.metric_range_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._draw_selected_series()
        )
        self.metric_canvas = tk.Canvas(
            self.charts_frame,
            height=180,
            background="#1f1f1f",
            highlightthickness=0,
        )
        self.metric_canvas.pack(fill=tk.BOTH, expand=True)
        self.metric_canvas.bind("<Configure>", lambda _e: self._draw_selected_series())
        self.metric_series_empty_label = ttk.Label(
            self.charts_frame,
            text=self.gui_manager.t("metric_history_empty"),
            foreground="gray",
        )
        self.metric_series_empty_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.metric_range_var.set(self.gui_manager.t("metric_range_all"))
        self._refresh_range_options()

        # Recent actions frame
        self.actions_frame = ttk.LabelFrame(
            self._content, text=self.gui_manager.t("recent_actions"), padding=10
        )
        self.actions_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        self.actions_tree = ttk.Treeview(
            self.actions_frame,
            columns=("time", "action", "result"),
            show="headings",
            height=10,
        )
        self.actions_tree.heading(
            "time", text=self.gui_manager.t("recent_actions_time")
        )
        self.actions_tree.heading(
            "action", text=self.gui_manager.t("recent_actions_action")
        )
        self.actions_tree.heading(
            "result", text=self.gui_manager.t("recent_actions_result")
        )

        self.actions_tree.column("time", width=150)
        self.actions_tree.column("action", width=300)
        self.actions_tree.column("result", width=100)

        actions_scrollbar = ttk.Scrollbar(
            self.actions_frame, orient=tk.VERTICAL, command=self.actions_tree.yview
        )
        self.actions_tree.configure(yscrollcommand=actions_scrollbar.set)

        self.actions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        actions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _show_metrics_placeholder(self):
        for child in self.dynamic_stats_container.winfo_children():
            child.destroy()
        placeholder = ttk.Label(
            self.dynamic_stats_container,
            text=self.gui_manager.t("no_metrics_configured"),
            foreground="gray",
        )
        placeholder.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)
        self.metrics_placeholder = placeholder
        self.dynamic_counter_vars.clear()
        self._dynamic_counter_keys = []

    def _bind_scroll(self):
        if self._scroll_bound:
            return
        try:
            self._canvas.bind_all("<MouseWheel>", self._on_scroll)
            self._canvas.bind_all("<Button-4>", self._on_scroll)
            self._canvas.bind_all("<Button-5>", self._on_scroll)
            self._scroll_bound = True
        except Exception:
            pass

    def _unbind_scroll(self):
        if not self._scroll_bound:
            return
        try:
            self._canvas.unbind_all("<MouseWheel>")
            self._canvas.unbind_all("<Button-4>")
            self._canvas.unbind_all("<Button-5>")
        except Exception:
            pass
        finally:
            self._scroll_bound = False

    def _on_scroll(self, event):
        if not getattr(self, "_canvas", None):
            return
        try:
            if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
                self._canvas.yview_scroll(-1, "units")
            else:
                self._canvas.yview_scroll(1, "units")
        except Exception:
            pass

    def _format_counter_value(self, metric: Dict[str, Any]) -> str:
        value = metric.get("value")
        if isinstance(value, float):
            if abs(value - round(value)) < 0.01:
                value = int(round(value))
            else:
                value = round(value, 2)
        unit = metric.get("unit")
        text = "-" if value is None else str(value)
        if unit and value is not None:
            text = f"{text} {unit}"
        last_ts = metric.get("last_ts")
        if last_ts:
            try:
                dt = datetime.fromtimestamp(last_ts)
                text = f"{text} ({dt.strftime('%H:%M')})"
            except Exception:
                pass
        return text

    def _rebuild_dynamic_counters(self, metrics: List[Dict[str, Any]]):
        for child in self.dynamic_stats_container.winfo_children():
            child.destroy()
        self.dynamic_counter_vars = {}
        self._dynamic_counter_keys = []
        if not metrics:
            self._show_metrics_placeholder()
            return
        self.metrics_placeholder = None
        for idx, metric in enumerate(metrics):
            key = metric.get("key")
            if not key:
                continue
            label_text = str(metric.get("label") or key)
            ttk.Label(
                self.dynamic_stats_container, text=label_text + ":"
            ).grid(row=idx, column=0, sticky=tk.W, padx=4, pady=2)
            value_var = tk.StringVar(value=self._format_counter_value(metric))
            ttk.Label(
                self.dynamic_stats_container,
                textvariable=value_var,
                anchor=tk.E,
                font=("Arial", 10, "bold"),
            ).grid(row=idx, column=1, sticky=tk.E, padx=4, pady=2)
            self.dynamic_counter_vars[key] = value_var
            self._dynamic_counter_keys.append(key)
        self.dynamic_stats_container.grid_columnconfigure(0, weight=1)
        self.dynamic_stats_container.grid_columnconfigure(1, weight=1)

    def _refresh_range_options(self):
        values: List[str] = []
        self.range_value_map: Dict[str, Optional[int]] = {}
        for key, seconds in self.range_definitions:
            label = self.gui_manager.t(key)
            values.append(label)
            self.range_value_map[label] = seconds
        if not values:
            self.metric_range_combo.configure(values=[])
            self.metric_range_var.set("")
            self.metric_range_combo.set("")
            return
        self.metric_range_combo.configure(values=values)
        current = self.metric_range_var.get()
        if current not in values:
            default_label = self.gui_manager.t("metric_range_all")
            selected = default_label if default_label in values else values[0]
            self.metric_range_var.set(selected)
        self.metric_range_combo.set(self.metric_range_var.get())

    def _update_series_selector(self, series: List[Dict[str, Any]]):
        self.metric_series_data = {}
        self.metric_key_map = {}
        options: List[str] = []
        for metric in series:
            key = metric.get("key")
            if not key:
                continue
            label = str(metric.get("label") or key)
            self.metric_series_data[key] = metric
            self.metric_key_map[label] = key
            options.append(label)
        self.metric_combo.configure(values=options)
        current_label = self.metric_combo_var.get()
        if not options:
            self.metric_combo_var.set("")
            self.metric_combo.set("")
            self._clear_series_canvas(empty_message=True)
            return
        if current_label not in options:
            current_label = options[0]
            self.metric_combo_var.set(current_label)
        self.metric_combo.set(self.metric_combo_var.get())
        self._draw_selected_series()

    def _clear_series_canvas(self, empty_message: bool = False):
        if not hasattr(self, "metric_canvas"):
            return
        self.metric_canvas.delete("all")
        if empty_message:
            self.metric_series_empty_label.config(
                text=self.gui_manager.t("metric_history_empty")
            )
            self.metric_series_empty_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        else:
            self.metric_series_empty_label.place_forget()

    def _get_selected_series_key(self) -> Optional[str]:
        label = self.metric_combo_var.get()
        return self.metric_key_map.get(label)

    def _get_selected_range_seconds(self) -> Optional[int]:
        label = self.metric_range_var.get()
        return self.range_value_map.get(label)

    def _draw_selected_series(self):
        if not hasattr(self, "metric_canvas"):
            return
        key = self._get_selected_series_key()
        entry = self.metric_series_data.get(key) if key else None
        if not entry:
            self._clear_series_canvas(empty_message=bool(self.metric_series_data))
            return
        points = entry.get("points") or []
        range_seconds = self._get_selected_range_seconds()
        now_ts = time.time()
        if range_seconds is not None:
            cutoff = now_ts - range_seconds
            points = [
                p
                for p in points
                if isinstance(p, dict) and p.get("ts") is not None and p["ts"] >= cutoff
            ]
        else:
            points = [p for p in points if isinstance(p, dict) and p.get("ts") is not None]
        if not points:
            self._clear_series_canvas(empty_message=True)
            return
        self.metric_series_empty_label.place_forget()
        canvas = self.metric_canvas
        canvas.delete("all")
        width = max(int(canvas.winfo_width()), 160)
        height = max(int(canvas.winfo_height()), 120)
        margin = 24
        usable_width = max(width - 2 * margin, 1)
        usable_height = max(height - 2 * margin, 1)

        sorted_points = sorted(points, key=lambda p: p["ts"])
        cumulative = []
        total = 0.0
        for point in sorted_points:
            try:
                value = float(point.get("value", 0))
            except (TypeError, ValueError):
                value = 0.0
            if value <= 0:
                continue
            total += value
            cumulative.append((point["ts"], total))
        if not cumulative:
            self._clear_series_canvas(empty_message=True)
            return
        min_ts = cumulative[0][0]
        max_ts = cumulative[-1][0]
        if max_ts == min_ts:
            max_ts += 1.0
        min_val = 0.0
        max_val = cumulative[-1][1]
        if max_val == min_val:
            max_val = min_val + 1.0

        coords: List[float] = []
        for ts, value in cumulative:
            x = margin + (ts - min_ts) / (max_ts - min_ts) * usable_width
            y_ratio = (value - min_val) / (max_val - min_val)
            y = margin + (1 - y_ratio) * usable_height
            coords.extend([x, y])

        canvas.create_rectangle(
            margin,
            margin,
            margin + usable_width,
            margin + usable_height,
            outline="#333333",
        )
        if len(coords) >= 4:
            canvas.create_line(*coords, fill="#3b82f6", width=2, smooth=True)
            for idx in range(0, len(coords), 2):
                canvas.create_oval(
                    coords[idx] - 2,
                    coords[idx + 1] - 2,
                    coords[idx] + 2,
                    coords[idx + 1] + 2,
                    fill="#60a5fa",
                    outline="",
                )

        start_text = datetime.fromtimestamp(min_ts).strftime("%H:%M")
        end_text = datetime.fromtimestamp(max_ts).strftime("%H:%M")
        canvas.create_text(
            margin,
            height - 6,
            text=start_text,
            anchor=tk.SW,
            fill="#bbbbbb",
            font=("Arial", 8),
        )
        canvas.create_text(
            width - margin,
            height - 6,
            text=end_text,
            anchor=tk.SE,
            fill="#bbbbbb",
            font=("Arial", 8),
        )
        total_text = self.gui_manager.t("metric_total_events").format(count=int(total))
        canvas.create_text(
            width - margin,
            margin,
            text=total_text,
            anchor=tk.NE,
            fill="#dddddd",
            font=("Arial", 9, "bold"),
        )

    def update_stats(self, stats_data):
        """Update statistics display using live global stats."""
        total_actions = int(stats_data.get("total_actions", 0))
        uptime = int(stats_data.get("uptime_secs", 0))

        self.actions_label.configure(text=str(total_actions))

        hours = uptime // 3600
        minutes = (uptime % 3600) // 60
        seconds = uptime % 60
        self.runtime_label.configure(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")

        metrics = stats_data.get("dynamic_counters") or []
        metric_keys = [m.get("key") for m in metrics if m.get("key")]
        if metric_keys != self._dynamic_counter_keys:
            self._rebuild_dynamic_counters(metrics)
        for metric in metrics:
            key = metric.get("key")
            var = self.dynamic_counter_vars.get(key)
            if var is not None:
                var.set(self._format_counter_value(metric))

        series = stats_data.get("series") or []
        self._update_series_selector(series)

        # Rebuild recent actions list if provided
        try:
            if "history" in stats_data:
                for item in self.actions_tree.get_children():
                    self.actions_tree.delete(item)
                for entry in stats_data.get("history", []):
                    ok = entry.get("ok", None)
                    detail_text = entry.get("detail", "").strip()
                    if ok is True:
                        base_result = self.gui_manager.t("result_ok")
                    elif ok is False:
                        base_result = self.gui_manager.t("result_fail")
                    else:
                        base_result = self.gui_manager.t("result_info")
                    result = (
                        f"{base_result} - {detail_text}"
                        if detail_text
                        else base_result
                    )
                    self.actions_tree.insert(
                        "",
                        0,
                        values=(entry.get("time", ""), entry.get("action", ""), result),
                    )
        except Exception:
            pass

    def refresh_texts(self):
        """Refresh static labels (minimal, as most are numeric)."""
        try:
            self.actions_frame.configure(text=self.gui_manager.t("recent_actions"))
            self.actions_tree.heading(
                "time", text=self.gui_manager.t("recent_actions_time")
            )
            self.actions_tree.heading(
                "action", text=self.gui_manager.t("recent_actions_action")
            )
            self.actions_tree.heading(
                "result", text=self.gui_manager.t("recent_actions_result")
            )
            self.actions_tree.column("result", width=120)
            self.metrics_frame.configure(text=self.gui_manager.t("task_metrics"))
            if self.metrics_placeholder:
                self.metrics_placeholder.configure(
                    text=self.gui_manager.t("no_metrics_configured")
                )
            self.series_frame.configure(text=self.gui_manager.t("series_metrics"))
            for col in ("name", "label", "source", "max_points", "unit"):
                self.series_tree.heading(col, text=self.gui_manager.t(f"series_{col}"))
            self.add_series_button.configure(text=self.gui_manager.t("add_series"))
            self.edit_series_button.configure(text=self.gui_manager.t("edit_series"))
            self.remove_series_button.configure(text=self.gui_manager.t("remove_series"))
            self._refresh_series_tree()
            self.charts_frame.configure(text=self.gui_manager.t("metric_history"))
            self.metric_select_label.configure(
                text=self.gui_manager.t("metric_select_label")
            )
            self.metric_range_label.configure(
                text=self.gui_manager.t("metric_range_label")
            )
            self.metric_series_empty_label.configure(
                text=self.gui_manager.t("metric_history_empty")
            )
            self._refresh_range_options()
            self.metric_combo.configure(values=list(self.metric_key_map.keys()))
            if self.metric_combo_var.get() in self.metric_key_map:
                self.metric_combo.set(self.metric_combo_var.get())
            self._draw_selected_series()
        except Exception:
            pass

        # runtime is updated in update_stats

    def add_action(self, action, result):
        """Add action to recent actions list"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.actions_tree.insert("", 0, values=(timestamp, action, result))

        # Keep only last 100 actions
        items = self.actions_tree.get_children()
        if len(items) > 100:
            self.actions_tree.delete(items[-1])


DEFAULT_PROTECTED_TASKS = {"preflight_guard", "stabilize_home"}
PROTECTED_TASK_IDS: Set[str] = set()


STEP_FIELD_SCHEMAS = {
    "sleep": [
        {"name": "seconds", "widget": "entry", "label": "Seconds"},
        {"name": "min", "widget": "entry", "label": "Min"},
        {"name": "max", "widget": "entry", "label": "Max"},
    ],
    "tap_template": [
        {"name": "template", "widget": "template", "label": "Template", "required": True,},
        {"name": "threshold", "widget": "entry", "label": "Threshold"},
        {"name": "use_roi", "widget": "bool", "label": "Use ROI", "default": True},
        {"name": "duration_ms", "widget": "entry", "label": "Duration (ms)"},
        {"name": "required", "widget": "bool", "label": "Required", "default": True},
        {"name": "stop_on_fail", "widget": "bool", "label": "Stop on fail", "default": True,},
        {"name": "success_detail", "widget": "entry", "label": "Success detail"},
        {"name": "fail_detail", "widget": "entry", "label": "Fail detail"},
        {"name": "success_steps", "widget": "steps", "label": "Success steps"},
        {"name": "failure_steps", "widget": "steps", "label": "Failure steps"},
    ],
    "wait_tap_template": [
        {"name": "template", "widget": "template", "label": "Template", "required": True,},
        {"name": "threshold", "widget": "entry", "label": "Threshold"},
        {"name": "use_roi", "widget": "bool", "label": "Use ROI", "default": True},
        {"name": "duration_ms", "widget": "entry", "label": "Duration (ms)"},
        {"name": "timeout", "widget": "entry", "label": "Timeout"},
        {"name": "sleep_interval", "widget": "entry", "label": "Sleep interval"},
        {"name": "success_detail", "widget": "entry", "label": "Success detail"},
        {"name": "fail_detail", "widget": "entry", "label": "Fail detail"},
        {"name": "required", "widget": "bool", "label": "Required", "default": True},
        {"name": "stop_on_fail", "widget": "bool", "label": "Stop on fail", "default": True,},
        {"name": "success_steps", "widget": "steps", "label": "Success steps"},
        {"name": "failure_steps", "widget": "steps", "label": "Failure steps"},
    ],
    "if": [
        {"name": "condition", "widget": "condition", "label": "Condition", "required": True,},
        {"name": "then_steps", "widget": "steps", "label": "Then steps"},
        {"name": "else_steps", "widget": "steps", "label": "Else steps"},
    ],
    "loop": [
        {"name": "condition", "widget": "condition", "label": "Condition"},
        {"name": "max_iterations", "widget": "entry", "label": "Max iterations"},
        {"name": "break_on_fail", "widget": "bool", "label": "Break on fail", "default": True,},
        {"name": "sleep_between", "widget": "entry", "label": "Sleep between"},
        {"name": "steps", "widget": "steps", "label": "Loop steps", "required": True},
    ],
    "log_message": [
        {"name": "level", "widget": "combo", "label": "Level", "values": ["info", "warning", "error", "debug"], "default": "info",},
        {"name": "message", "widget": "entry", "label": "Message", "required": True},
    ],
    "set_flag": [
        {"name": "name", "widget": "entry", "label": "Flag name", "required": True},
        {"name": "value", "widget": "entry", "label": "Value", "default": True},
    ],
    "set_detail": [
        {"name": "text", "widget": "entry", "label": "Detail", "required": True},
        {"name": "append", "widget": "bool", "label": "Append", "default": False},
    ],
    "set_success": [
        {"name": "value", "widget": "entry", "label": "Success value"},
        {"name": "from_flag", "widget": "entry", "label": "From flag"},
        {"name": "default", "widget": "entry", "label": "Default"},
    ],
    "set_executed": [
        {"name": "value", "widget": "bool", "label": "Executed", "default": True},
    ],
    "press_back": [
        {"name": "sleep_after", "widget": "entry", "label": "Sleep after"},
    ],
    "call_task": [
        {"name": "task_id", "widget": "task", "label": "Task", "required": True},
        {"name": "propagate_success", "widget": "bool", "label": "Propagate success", "default": True,},
        {"name": "stop_on_fail", "widget": "bool", "label": "Stop on fail", "default": True,},
        {"name": "detail", "widget": "entry", "label": "Detail"},
    ],
    "stop_task": [
        {"name": "success", "widget": "entry", "label": "Success"},
        {"name": "detail", "widget": "entry", "label": "Detail"},
    ],
}


def summarize_step_description(step: Dict[str, Any]) -> str:
    step_type = step.get("type", "")
    if step_type in ("tap_template", "wait_tap_template"):
        label = step.get("template", "")
        outcome = []
        if step.get("success_steps"):
            outcome.append("then")
        if step.get("failure_steps"):
            outcome.append("else")
        suffix = f" ({'/'.join(outcome)})" if outcome else ""
        return f"{label}{suffix}"
    if step_type == "sleep":
        seconds = step.get("seconds")
        if seconds:
            return f"sleep {seconds}"
        min_v = step.get("min")
        max_v = step.get("max")
        if min_v or max_v:
            return f"sleep {min_v or ''}-{max_v or ''}"
    if step_type == "log_message":
        return f"{step.get('level', 'info')}: {step.get('message', '')}"
    if step_type == "set_flag":
        return step.get("name", "")
    if step_type == "set_detail":
        return step.get("text", "")
    if step_type == "set_success":
        if "value" in step:
            return f"success={step.get('value')}"
        if step.get("from_flag"):
            return f"success from {step.get('from_flag')}"
    if step_type == "if":
        cond = step.get("condition", {}).get("type", "condition")
        return f"if {cond}"
    if step_type == "loop":
        cond = step.get("condition", {}).get("type") or "count"
        return f"loop {cond}"
    if step_type == "call_task":
        return step.get("task_id", "")
    if step_type == "stop_task":
        return step.get("detail", "")
    return ", ".join(f"{k}={v}" for k, v in step.items() if k != "type")


class TaskStepDialog:
    def __init__(
        self,
        parent,
        gui_manager,
        template_names: Optional[List[str]] = None,
        task_ids: Optional[List[str]] = None,
        initial=None,
    ):
        self.parent = parent
        self.gui_manager = gui_manager
        self.template_names = sorted(template_names or [])
        self.task_ids = sorted(task_ids or [])
        self.initial = initial or {}
        self.result = None
        self.field_widgets: Dict[str, Any] = {}

        self.top = tk.Toplevel(parent)
        self.top.title(self.gui_manager.t("edit_step"))
        self.top.transient(parent)
        self.top.resizable(False, False)
        self.top.grab_set()

        self.type_var = tk.StringVar(value=self.initial.get("type", "tap_template"))
        type_frame = ttk.Frame(self.top)
        type_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        ttk.Label(type_frame, text=self.gui_manager.t("step_type") + ":").pack(
            side=tk.LEFT
        )
        self.type_combo = ttk.Combobox(
            type_frame,
            textvariable=self.type_var,
            values=list(STEP_FIELD_SCHEMAS.keys()),
            state="readonly",
            width=24,
        )
        self.type_combo.pack(side=tk.LEFT, padx=5)
        self.type_combo.bind("<<ComboboxSelected>>", lambda _e: self.render_fields())

        self.fields_frame = ttk.Frame(self.top)
        self.fields_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        button_frame = ttk.Frame(self.top)
        button_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        ttk.Button(
            button_frame, text=self.gui_manager.t("save"), command=self.on_save
        ).pack(side=tk.RIGHT, padx=5)
        ttk.Button(
            button_frame, text=self.gui_manager.t("cancel"), command=self.on_cancel
        ).pack(side=tk.RIGHT)

        self.top.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.render_fields()

    def render_fields(self):
        for child in self.fields_frame.winfo_children():
            child.destroy()
        self.field_widgets.clear()

        schema = STEP_FIELD_SCHEMAS.get(self.type_var.get(), [])
        for row, spec in enumerate(schema):
            label = ttk.Label(
                self.fields_frame,
                text=f"{spec.get('label', spec['name']).title()}",
            )
            label.grid(row=row, column=0, sticky=tk.W, pady=2)

            initial_value = self.initial.get(spec["name"])
            widget_type = spec.get("widget", "entry")

            if widget_type == "bool":
                var = tk.BooleanVar(
                    value=bool(initial_value)
                    if initial_value is not None
                    else spec.get("default", False)
                )
                widget = ttk.Checkbutton(self.fields_frame, variable=var)
                widget.var = var
            elif widget_type == "combo":
                values = spec.get("values", [])
                var = tk.StringVar(
                    value=initial_value
                    if initial_value is not None
                    else spec.get("default", "")
                )
                widget = ttk.Combobox(
                    self.fields_frame,
                    textvariable=var,
                    values=values,
                    state="readonly" if values else "normal",
                    width=28,
                )
                widget.var = var
            elif widget_type == "template":
                var = tk.StringVar(
                    value=initial_value
                    if initial_value is not None
                    else (self.template_names[0] if self.template_names else "")
                )
                widget = ttk.Combobox(
                    self.fields_frame,
                    textvariable=var,
                    values=self.template_names,
                    state="readonly" if self.template_names else "normal",
                    width=28,
                )
                widget.var = var
            elif widget_type == "task":
                var = tk.StringVar(
                    value=initial_value
                    if initial_value is not None
                    else (self.task_ids[0] if self.task_ids else "")
                )
                widget = ttk.Combobox(
                    self.fields_frame,
                    textvariable=var,
                    values=self.task_ids,
                    state="readonly" if self.task_ids else "normal",
                    width=28,
                )
                widget.var = var
            elif widget_type == "steps":
                steps_value = (
                    copy.deepcopy(initial_value)
                    if isinstance(initial_value, list)
                    else []
                )
                var = {"value": steps_value}

                def open_steps_editor(var=var, spec=spec):
                    editor = StepsEditorDialog(
                        self.top,
                        self.gui_manager,
                        self.template_names,
                        self.task_ids,
                        var["value"],
                    )
                    result = editor.show()
                    if result is not None:
                        var["value"] = result
                        summary_var.set(self._format_steps_summary(var["value"]))

                summary_var = tk.StringVar(
                    value=self._format_steps_summary(var["value"])
                )
                button = ttk.Button(
                    self.fields_frame,
                    text=f"{self.gui_manager.t('edit')}...",
                    command=open_steps_editor,
                )
                button.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                summary_label = ttk.Label(self.fields_frame, textvariable=summary_var)
                summary_label.grid(row=row, column=2, sticky=tk.W, padx=5, pady=2)
                widget = button
                widget.var = var
                widget.summary_var = summary_var
            elif widget_type == "condition":
                condition_value = (
                    copy.deepcopy(initial_value)
                    if isinstance(initial_value, dict)
                    else None
                )
                var = {"value": condition_value}

                def open_condition_editor(var=var):
                    editor = ConditionEditorDialog(
                        self.top,
                        self.gui_manager,
                        self.template_names,
                        initial_condition=var["value"],
                    )
                    result = editor.show()
                    if result is not None:
                        var["value"] = result
                        summary_var.set(self._format_condition_summary(var["value"]))

                summary_var = tk.StringVar(
                    value=self._format_condition_summary(var["value"])
                )
                button = ttk.Button(
                    self.fields_frame,
                    text=f"{self.gui_manager.t('edit')}...",
                    command=open_condition_editor,
                )
                button.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                summary_label = ttk.Label(self.fields_frame, textvariable=summary_var)
                summary_label.grid(row=row, column=2, sticky=tk.W, padx=5, pady=2)
                widget = button
                widget.var = var
                widget.summary_var = summary_var
            else:
                var = tk.StringVar()
                if initial_value is not None:
                    var.set(str(initial_value))
                elif spec.get("default") is not None:
                    var.set(str(spec["default"]))
                widget = ttk.Entry(self.fields_frame, textvariable=var, width=30)
                widget.var = var

            if widget_type not in {"steps", "condition"}:
                widget.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.field_widgets[spec["name"]] = (spec, widget)

    def _format_steps_summary(self, steps: List[Dict[str, Any]]) -> str:
        count = len(steps or [])
        if count == 0:
            return "(empty)"
        first = summarize_step_description(steps[0])
        suffix = "" if count == 1 else f" +{count - 1}"
        return f"{first}{suffix}"

    def _format_condition_summary(self, condition: Optional[Dict[str, Any]]) -> str:
        if not condition:
            return "(none)"
        cond_type = condition.get("type", "condition")
        template = condition.get("template")
        if template:
            return f"{cond_type}: {template}"
        flag = condition.get("flag")
        if flag:
            return f"{cond_type}: {flag}"
        return cond_type

    def on_save(self):
        step_type = self.type_var.get()
        data = {"type": step_type}
        for name, (spec, widget) in self.field_widgets.items():
            widget_type = spec.get("widget", "entry")
            if widget_type == "bool":
                value = bool(widget.var.get())
            elif widget_type in {"combo", "template", "task", "entry"}:
                value = widget.var.get().strip()
                if value == "":
                    value = None
            elif widget_type == "steps":
                value = copy.deepcopy(widget.var["value"])
                if not value:
                    value = []
            elif widget_type == "condition":
                value = copy.deepcopy(widget.var["value"])
            else:
                value = widget.var.get().strip()
                if value == "":
                    value = None

            if value is None and spec.get("required"):
                messagebox.showwarning(
                    "Warning",
                    f"{spec.get('label', name)} is required.",
                    parent=self.top,
                )
                return
            if value is not None:
                data[name] = copy.deepcopy(value)

        self.result = data
        self.top.grab_release()
        self.top.destroy()

    def on_cancel(self):
        self.result = None
        self.top.grab_release()
        self.top.destroy()

    def show(self) -> Optional[Dict[str, Any]]:
        self.parent.wait_window(self.top)
        return self.result


class StepsEditorDialog:
    def __init__(
        self,
        parent,
        gui_manager,
        template_names: List[str],
        task_ids: List[str],
        initial_steps: Optional[List[Dict[str, Any]]],
    ):
        self.parent = parent
        self.gui_manager = gui_manager
        self.template_names = template_names
        self.task_ids = task_ids
        self.steps = copy.deepcopy(initial_steps or [])
        self.result = None

        self.top = tk.Toplevel(parent)
        self.top.title(self.gui_manager.t("steps"))
        self.top.transient(parent)
        self.top.resizable(False, False)
        self.top.grab_set()

        frame = ttk.Frame(self.top)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.listbox = tk.Listbox(frame, height=10, width=50)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.configure(yscrollcommand=scrollbar.set)

        btn_frame = ttk.Frame(self.top)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(
            btn_frame, text=self.gui_manager.t("add"), command=self.add_step
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            btn_frame, text=self.gui_manager.t("edit"), command=self.edit_step
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            btn_frame, text=self.gui_manager.t("delete"), command=self.delete_step
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            btn_frame,
            text=self.gui_manager.t("move_up"),
            command=lambda: self.move_step(-1),
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            btn_frame,
            text=self.gui_manager.t("move_down"),
            command=lambda: self.move_step(1),
        ).pack(side=tk.LEFT, padx=3)

        action_frame = ttk.Frame(self.top)
        action_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(
            action_frame, text=self.gui_manager.t("save"), command=self.on_save
        ).pack(side=tk.RIGHT, padx=5)
        ttk.Button(
            action_frame, text=self.gui_manager.t("cancel"), command=self.on_cancel
        ).pack(side=tk.RIGHT)

        self.top.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self._refresh_listbox()

    def _refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for step in self.steps:
            self.listbox.insert(tk.END, summarize_step_description(step))

    def _current_index(self) -> Optional[int]:
        selection = self.listbox.curselection()
        if not selection:
            return None
        return selection[0]

    def _task_ids_for_picker(self) -> List[str]:
        return [task.get("id", "") for task in self.tasks if task.get("id")]

    def add_step(self):
        dialog = TaskStepDialog(
            self.top,
            self.gui_manager,
            self.template_names,
            self.task_ids,
        )
        step = dialog.show()
        if step:
            self.steps.append(step)
            self._refresh_listbox()

    def edit_step(self):
        idx = self._current_index()
        if idx is None:
            return
        dialog = TaskStepDialog(
            self.top,
            self.gui_manager,
            self.template_names,
            self.task_ids,
            initial=self.steps[idx],
        )
        step = dialog.show()
        if step:
            self.steps[idx] = step
            self._refresh_listbox()

    def delete_step(self):
        idx = self._current_index()
        if idx is None:
            return
        del self.steps[idx]
        self._refresh_listbox()

    def move_step(self, direction: int):
        idx = self._current_index()
        if idx is None:
            return
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(self.steps):
            return
        self.steps[idx], self.steps[new_idx] = self.steps[new_idx], self.steps[idx]
        self._refresh_listbox()
        self.listbox.selection_set(new_idx)

    def on_save(self):
        self.result = copy.deepcopy(self.steps)
        self.top.grab_release()
        self.top.destroy()

    def on_cancel(self):
        self.result = None
        self.top.grab_release()
        self.top.destroy()

    def show(self) -> Optional[List[Dict[str, Any]]]:
        self.parent.wait_window(self.top)
        return self.result


class ConditionEditorDialog:
    def __init__(
        self,
        parent,
        gui_manager,
        template_names: List[str],
        initial_condition: Optional[Dict[str, Any]] = None,
    ):
        self.parent = parent
        self.gui_manager = gui_manager
        self.template_names = template_names
        self.condition = copy.deepcopy(initial_condition) or {"type": "always_true"}
        self.result = None

        self.top = tk.Toplevel(parent)
        self.top.title(self.gui_manager.t("condition"))
        self.top.transient(parent)
        self.top.resizable(False, False)
        self.top.grab_set()

        frame = ttk.Frame(self.top)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(frame, text=self.gui_manager.t("condition") + ":").grid(
            row=0, column=0, sticky=tk.W
        )
        self.type_var = tk.StringVar(value=self._initial_type())
        type_combo = ttk.Combobox(
            frame,
            textvariable=self.type_var,
            values=[
                "template_present",
                "template_missing",
                "flag_equals",
                "flag_not_equals",
                "always_true",
                "always_false",
            ],
            state="readonly",
            width=24,
        )
        type_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        type_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_fields())

        self.fields_frame = ttk.Frame(frame)
        self.fields_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)

        action_frame = ttk.Frame(self.top)
        action_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(
            action_frame, text=self.gui_manager.t("save"), command=self.on_save
        ).pack(side=tk.RIGHT, padx=5)
        ttk.Button(
            action_frame, text=self.gui_manager.t("cancel"), command=self.on_cancel
        ).pack(side=tk.RIGHT)

        self.top.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self._refresh_fields()

    def _initial_type(self) -> str:
        cond_type = self.condition.get("type")
        if cond_type == "template":
            return (
                "template_missing"
                if self.condition.get("negate")
                else "template_present"
            )
        if cond_type == "flag":
            return "flag_not_equals" if self.condition.get("negate") else "flag_equals"
        if cond_type in {"always_true", "always_false"}:
            return cond_type
        return "always_true"

    def _refresh_fields(self):
        for child in self.fields_frame.winfo_children():
            child.destroy()

        cond_type = self.type_var.get()
        if cond_type in {"template_present", "template_missing"}:
            ttk.Label(self.fields_frame, text="Template:").grid(
                row=0, column=0, sticky=tk.W
            )
            self.template_var = tk.StringVar(
                value=self.condition.get(
                    "template", self.template_names[0] if self.template_names else ""
                )
            )
            ttk.Combobox(
                self.fields_frame,
                textvariable=self.template_var,
                values=self.template_names,
                state="readonly" if self.template_names else "normal",
                width=24,
            ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

            ttk.Label(self.fields_frame, text="Threshold:").grid(
                row=1, column=0, sticky=tk.W
            )
            self.threshold_var = tk.StringVar(
                value=str(self.condition.get("threshold", ""))
            )
            ttk.Entry(
                self.fields_frame, textvariable=self.threshold_var, width=26
            ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

            self.use_roi_var = tk.BooleanVar(value=self.condition.get("use_roi", True))
            ttk.Checkbutton(
                self.fields_frame,
                text="Use ROI",
                variable=self.use_roi_var,
            ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        elif cond_type in {"flag_equals", "flag_not_equals"}:
            ttk.Label(self.fields_frame, text="Flag:").grid(
                row=0, column=0, sticky=tk.W
            )
            self.flag_var = tk.StringVar(value=self.condition.get("flag", ""))
            ttk.Entry(self.fields_frame, textvariable=self.flag_var, width=26).grid(
                row=0, column=1, sticky=tk.W, padx=5, pady=2
            )

            ttk.Label(self.fields_frame, text="Value:").grid(
                row=1, column=0, sticky=tk.W
            )
            self.flag_value_var = tk.StringVar(
                value=str(self.condition.get("value", ""))
            )
            ttk.Entry(
                self.fields_frame, textvariable=self.flag_value_var, width=26
            ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        else:
            ttk.Label(self.fields_frame, text="No additional parameters.").grid(
                row=0, column=0, sticky=tk.W
            )

    def on_save(self):
        cond_type = self.type_var.get()
        if cond_type in {"template_present", "template_missing"}:
            template = self.template_var.get().strip()
            if not template:
                messagebox.showwarning(
                    "Warning", "Template is required.", parent=self.top
                )
                return
            threshold = self.threshold_var.get().strip()
            condition = {
                "type": "template",
                "template": template,
                "use_roi": bool(self.use_roi_var.get()),
            }
            if threshold:
                condition["threshold"] = threshold
            if cond_type == "template_missing":
                condition["negate"] = True
        elif cond_type in {"flag_equals", "flag_not_equals"}:
            flag = self.flag_var.get().strip()
            if not flag:
                messagebox.showwarning(
                    "Warning", "Flag name is required.", parent=self.top
                )
                return
            value = self.flag_value_var.get()
            condition = {
                "type": "flag",
                "flag": flag,
                "value": value,
                "negate": cond_type == "flag_not_equals",
            }
        else:
            condition = {"type": cond_type}

        self.result = condition
        self.top.grab_release()
        self.top.destroy()

    def on_cancel(self):
        self.result = None
        self.top.grab_release()
        self.top.destroy()

    def show(self) -> Optional[Dict[str, Any]]:
        self.parent.wait_window(self.top)
        return self.result


class CounterEditorDialog:
    def __init__(
        self,
        parent,
        gui_manager,
        initial: Optional[Dict[str, Any]] = None,
    ):
        self.parent = parent
        self.gui_manager = gui_manager
        data = initial or {}
        self.result: Optional[Dict[str, Any]] = None

        self.top = tk.Toplevel(parent)
        self.top.title(self.gui_manager.t("counter_editor_title"))
        self.top.transient(parent)
        self.top.resizable(False, False)
        self.top.grab_set()

        frame = ttk.Frame(self.top, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=self.gui_manager.t("counter_name") + ":").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.name_var = tk.StringVar(value=data.get("name", ""))
        ttk.Entry(frame, textvariable=self.name_var, width=30).grid(
            row=0, column=1, sticky=tk.W, pady=2
        )

        ttk.Label(frame, text=self.gui_manager.t("counter_label") + ":").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.label_var = tk.StringVar(value=data.get("label", ""))
        ttk.Entry(frame, textvariable=self.label_var, width=30).grid(
            row=1, column=1, sticky=tk.W, pady=2
        )

        ttk.Label(frame, text=self.gui_manager.t("counter_source") + ":").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        self.source_var = tk.StringVar(value=data.get("source", ""))
        ttk.Entry(frame, textvariable=self.source_var, width=30).grid(
            row=2, column=1, sticky=tk.W, pady=2
        )

        ttk.Label(frame, text=self.gui_manager.t("counter_accumulate") + ":").grid(
            row=3, column=0, sticky=tk.W, pady=2
        )
        accumulate_map = {
            "sum": self.gui_manager.t("counter_accumulate_sum"),
            "set": self.gui_manager.t("counter_accumulate_set"),
        }
        self.accumulate_map = accumulate_map
        reverse_map = {v: k for k, v in accumulate_map.items()}
        current_acc = data.get("accumulate", "sum").lower()
        display_value = accumulate_map.get(current_acc, accumulate_map["sum"])
        self.accumulate_var = tk.StringVar(value=display_value)
        self.accumulate_combo = ttk.Combobox(
            frame,
            textvariable=self.accumulate_var,
            values=list(accumulate_map.values()),
            state="readonly",
            width=28,
        )
        self.accumulate_combo.grid(row=3, column=1, sticky=tk.W, pady=2)
        self.accumulate_reverse_map = reverse_map

        ttk.Label(frame, text=self.gui_manager.t("counter_unit") + ":").grid(
            row=4, column=0, sticky=tk.W, pady=2
        )
        self.unit_var = tk.StringVar(value=data.get("unit", ""))
        ttk.Entry(frame, textvariable=self.unit_var, width=30).grid(
            row=4, column=1, sticky=tk.W, pady=2
        )

        actions = ttk.Frame(self.top, padding=(10, 0, 10, 10))
        actions.pack(fill=tk.X)
        ttk.Button(actions, text=self.gui_manager.t("save"), command=self.on_save).pack(
            side=tk.RIGHT, padx=5
        )
        ttk.Button(
            actions, text=self.gui_manager.t("cancel"), command=self.on_cancel
        ).pack(side=tk.RIGHT)

        self.top.protocol("WM_DELETE_WINDOW", self.on_cancel)

    def on_save(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning(
                "Warning", self.gui_manager.t("counter_name_required"), parent=self.top
            )
            return
        label = self.label_var.get().strip()
        source = self.source_var.get().strip()
        accumulate_display = self.accumulate_var.get()
        accumulate = self.accumulate_reverse_map.get(accumulate_display, "sum")
        unit = self.unit_var.get().strip()
        self.result = {
            "name": name,
            "label": label,
            "source": source,
            "accumulate": accumulate,
            "unit": unit,
        }
        self.top.grab_release()
        self.top.destroy()

    def on_cancel(self):
        self.result = None
        self.top.grab_release()
        self.top.destroy()

    def show(self) -> Optional[Dict[str, Any]]:
        self.parent.wait_window(self.top)
        return self.result


class SeriesEditorDialog:
    def __init__(
        self,
        parent,
        gui_manager,
        initial: Optional[Dict[str, Any]] = None,
    ):
        self.parent = parent
        self.gui_manager = gui_manager
        data = initial or {}
        self.result: Optional[Dict[str, Any]] = None

        self.top = tk.Toplevel(parent)
        self.top.title(self.gui_manager.t("series_editor_title"))
        self.top.transient(parent)
        self.top.resizable(False, False)
        self.top.grab_set()

        frame = ttk.Frame(self.top, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=self.gui_manager.t("series_name") + ":").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.name_var = tk.StringVar(value=data.get("name", ""))
        ttk.Entry(frame, textvariable=self.name_var, width=30).grid(
            row=0, column=1, sticky=tk.W, pady=2
        )

        ttk.Label(frame, text=self.gui_manager.t("series_label") + ":").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.label_var = tk.StringVar(value=data.get("label", ""))
        ttk.Entry(frame, textvariable=self.label_var, width=30).grid(
            row=1, column=1, sticky=tk.W, pady=2
        )

        ttk.Label(frame, text=self.gui_manager.t("series_source") + ":").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        self.source_var = tk.StringVar(value=data.get("source", ""))
        ttk.Entry(frame, textvariable=self.source_var, width=30).grid(
            row=2, column=1, sticky=tk.W, pady=2
        )

        ttk.Label(frame, text=self.gui_manager.t("series_max_points") + ":").grid(
            row=3, column=0, sticky=tk.W, pady=2
        )
        self.max_points_var = tk.StringVar(value=str(data.get("max_points", 50)))
        ttk.Entry(frame, textvariable=self.max_points_var, width=12).grid(
            row=3, column=1, sticky=tk.W, pady=2
        )

        ttk.Label(frame, text=self.gui_manager.t("series_unit") + ":").grid(
            row=4, column=0, sticky=tk.W, pady=2
        )
        self.unit_var = tk.StringVar(value=data.get("unit", ""))
        ttk.Entry(frame, textvariable=self.unit_var, width=30).grid(
            row=4, column=1, sticky=tk.W, pady=2
        )

        actions = ttk.Frame(self.top, padding=(10, 0, 10, 10))
        actions.pack(fill=tk.X)
        ttk.Button(actions, text=self.gui_manager.t("save"), command=self.on_save).pack(
            side=tk.RIGHT, padx=5
        )
        ttk.Button(
            actions, text=self.gui_manager.t("cancel"), command=self.on_cancel
        ).pack(side=tk.RIGHT)

        self.top.protocol("WM_DELETE_WINDOW", self.on_cancel)

    def on_save(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning(
                "Warning", self.gui_manager.t("series_name_required"), parent=self.top
            )
            return
        label = self.label_var.get().strip()
        source = self.source_var.get().strip()
        if not source:
            messagebox.showwarning(
                "Warning", self.gui_manager.t("series_source_required"), parent=self.top
            )
            return
        try:
            max_points = int(self.max_points_var.get().strip() or 50)
        except ValueError:
            messagebox.showwarning(
                "Warning", self.gui_manager.t("series_max_points_invalid"), parent=self.top
            )
            return
        unit = self.unit_var.get().strip()
        self.result = {
            "name": name,
            "label": label,
            "source": source,
            "max_points": max_points,
            "unit": unit,
        }
        self.top.grab_release()
        self.top.destroy()

    def on_cancel(self):
        self.result = None
        self.top.grab_release()
        self.top.destroy()

    def show(self) -> Optional[Dict[str, Any]]:
        self.parent.wait_window(self.top)
        return self.result


class TasksTab(ttk.Frame):
    """Tab Tasks for creating and editing Workflows."""

    def __init__(self, parent, gui_manager):
        super().__init__(parent)
        self.gui_manager = gui_manager
        self.tasks: List[Dict[str, Any]] = []
        self.tasks_version = 1
        self.current_index: Optional[int] = None
        self.current_steps: List[Dict[str, Any]] = []
        self.current_counters: List[Dict[str, Any]] = []
        self.current_series: List[Dict[str, Any]] = []
        self._suspend = False
        self._right_mousewheel_bound = False

        self._entry_labels: List[ttk.Label] = []

        self.setup_ui()
        self.load_tasks_from_file()

    def setup_ui(self):
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.list_label = ttk.Label(
            left, text=self.gui_manager.t("task_list"), font=("Arial", 10, "bold")
        )
        self.list_label.pack(anchor=tk.W)

        self.task_listbox = tk.Listbox(left, height=18, exportselection=False)
        self.task_listbox.pack(fill=tk.BOTH, expand=True, pady=(5, 5))
        self.task_listbox.bind("<<ListboxSelect>>", self.on_task_select)

        task_btn_frame = ttk.Frame(left)
        task_btn_frame.pack(fill=tk.X)
        self.add_task_button = ttk.Button(
            task_btn_frame, text=self.gui_manager.t("add_task"), command=self.add_task
        )
        self.add_task_button.pack(fill=tk.X, pady=1)
        self.duplicate_task_button = ttk.Button(
            task_btn_frame,
            text=self.gui_manager.t("duplicate_task"),
            command=self.duplicate_task,
        )
        self.duplicate_task_button.pack(fill=tk.X, pady=1)
        self.delete_task_button = ttk.Button(
            task_btn_frame,
            text=self.gui_manager.t("delete_task"),
            command=self.delete_task,
        )
        self.delete_task_button.pack(fill=tk.X, pady=1)

        move_frame = ttk.Frame(left)
        move_frame.pack(fill=tk.X, pady=(6, 0))
        self.move_up_button = ttk.Button(
            move_frame,
            text=self.gui_manager.t("move_up"),
            command=lambda: self.move_task(-1),
        )
        self.move_up_button.pack(fill=tk.X, pady=1)
        self.move_down_button = ttk.Button(
            move_frame,
            text=self.gui_manager.t("move_down"),
            command=lambda: self.move_task(1),
        )
        self.move_down_button.pack(fill=tk.X, pady=1)

        self.protected_list_hint = ttk.Label(
            left,
            text=self.gui_manager.t("task_protected_list_hint"),
            wraplength=220,
            justify=tk.LEFT,
            foreground="orange",
        )
        self.protected_list_hint.pack(fill=tk.X, pady=(6, 0))

        right_container = ttk.Frame(main)
        right_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_canvas = tk.Canvas(right_container, highlightthickness=0)
        right_scrollbar = ttk.Scrollbar(
            right_container, orient=tk.VERTICAL, command=self.right_canvas.yview
        )
        self.right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_canvas.configure(yscrollcommand=right_scrollbar.set)

        self.right_frame = ttk.Frame(self.right_canvas)
        self._right_canvas_window = self.right_canvas.create_window(
            (0, 0), window=self.right_frame, anchor=tk.NW
        )

        self.right_frame.bind(
            "<Configure>",
            lambda e: self.right_canvas.configure(
                scrollregion=self.right_canvas.bbox("all")
            ),
        )
        self.right_canvas.bind(
            "<Configure>",
            lambda e: self.right_canvas.itemconfigure(
                self._right_canvas_window, width=e.width
            ),
        )
        self.right_canvas.bind("<Enter>", lambda _e: self._bind_right_mousewheel())
        self.right_canvas.bind("<Leave>", lambda _e: self._unbind_right_mousewheel())

        top_frame = ttk.Frame(self.right_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)

        self.details_frame = ttk.LabelFrame(
            top_frame, text=self.gui_manager.t("task"), padding=10
        )
        self.details_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.task_id_var = tk.StringVar()
        self.task_name_var = tk.StringVar()
        self.task_desc_var = tk.StringVar()
        self.task_enabled_var = tk.BooleanVar(value=True)

        self.task_id_entry = self._add_entry_row(
            self.details_frame, 0, self.gui_manager.t("task_id"), self.task_id_var
        )
        self.task_name_entry = self._add_entry_row(
            self.details_frame, 1, self.gui_manager.t("task_name"), self.task_name_var
        )
        self.task_desc_entry = self._add_entry_row(
            self.details_frame,
            2,
            self.gui_manager.t("task_description"),
            self.task_desc_var,
        )
        self.enabled_check = ttk.Checkbutton(
            self.details_frame,
            text=self.gui_manager.t("task_enabled"),
            variable=self.task_enabled_var,
        )
        self.enabled_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=4)
        self.protected_hint_var = tk.StringVar(value="")
        self.protected_hint_label = ttk.Label(
            self.details_frame,
            textvariable=self.protected_hint_var,
            foreground="orange",
            wraplength=360,
            justify=tk.LEFT,
        )
        self.protected_hint_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))

        self.trigger_frame = ttk.LabelFrame(
            top_frame, text=self.gui_manager.t("trigger"), padding=10
        )
        self.trigger_frame.grid(row=0, column=1, sticky="nsew")

        self.trigger_type_var = tk.StringVar(value="interval")
        self.trigger_min_var = tk.StringVar()
        self.trigger_max_var = tk.StringVar()
        self.trigger_run_var = tk.BooleanVar(value=False)

        self.trigger_type_label = ttk.Label(
            self.trigger_frame, text=self.gui_manager.t("trigger_type") + ":"
        )
        self.trigger_type_label.grid(row=0, column=0, sticky=tk.W, pady=2)
        self.trigger_type_combo = ttk.Combobox(
            self.trigger_frame,
            textvariable=self.trigger_type_var,
            values=["interval", "scheduled"],
            state="readonly",
            width=18,
        )
        self.trigger_type_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        self.trigger_type_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._update_trigger_fields()
        )

        self.trigger_min_label = ttk.Label(
            self.trigger_frame, text=self.gui_manager.t("trigger_min_seconds") + ":"
        )
        self.trigger_min_label.grid(row=1, column=0, sticky=tk.W, pady=2)
        self.trigger_min_entry = ttk.Entry(
            self.trigger_frame, textvariable=self.trigger_min_var, width=20
        )
        self.trigger_min_entry.grid(row=1, column=1, sticky=tk.W, padx=5)

        self.trigger_max_label = ttk.Label(
            self.trigger_frame, text=self.gui_manager.t("trigger_max_seconds") + ":"
        )
        self.trigger_max_label.grid(row=2, column=0, sticky=tk.W, pady=2)
        self.trigger_max_entry = ttk.Entry(
            self.trigger_frame, textvariable=self.trigger_max_var, width=20
        )
        self.trigger_max_entry.grid(row=2, column=1, sticky=tk.W, padx=5)

        self.trigger_times_var = tk.StringVar()
        self.trigger_times_label = ttk.Label(
            self.trigger_frame, text=self.gui_manager.t("trigger_times") + ":"
        )
        self.trigger_times_entry = ttk.Entry(
            self.trigger_frame, textvariable=self.trigger_times_var, width=20
        )

        self.trigger_run_check = ttk.Checkbutton(
            self.trigger_frame,
            text=self.gui_manager.t("trigger_run_at_start"),
            variable=self.trigger_run_var,
        )
        self.trigger_run_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)

        self._update_trigger_fields()

        self.stats_frame = ttk.LabelFrame(
            self.right_frame, text=self.gui_manager.t("task_metrics"), padding=10
        )
        self.stats_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        counters_container = ttk.Frame(self.stats_frame)
        counters_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.counters_tree = ttk.Treeview(
            counters_container,
            columns=("name", "label", "source", "accumulate", "unit"),
            show="headings",
            height=5,
        )
        for col in ("name", "label", "source", "accumulate", "unit"):
            self.counters_tree.heading(col, text=self.gui_manager.t(f"counter_{col}"))
        self.counters_tree.column("name", width=120, anchor=tk.W)
        self.counters_tree.column("label", width=160, anchor=tk.W)
        self.counters_tree.column("source", width=200, anchor=tk.W)
        self.counters_tree.column("accumulate", width=110, anchor=tk.W)
        self.counters_tree.column("unit", width=80, anchor=tk.W)
        counters_scroll = ttk.Scrollbar(
            counters_container, orient=tk.VERTICAL, command=self.counters_tree.yview
        )
        self.counters_tree.configure(yscrollcommand=counters_scroll.set)
        self.counters_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        counters_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        counter_buttons = ttk.Frame(self.stats_frame)
        counter_buttons.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.add_counter_button = ttk.Button(
            counter_buttons,
            text=self.gui_manager.t("add_counter"),
            command=self.add_counter,
        )
        self.add_counter_button.pack(fill=tk.X, pady=2)
        self.edit_counter_button = ttk.Button(
            counter_buttons,
            text=self.gui_manager.t("edit_counter"),
            command=self.edit_counter,
        )
        self.edit_counter_button.pack(fill=tk.X, pady=2)
        self.remove_counter_button = ttk.Button(
            counter_buttons,
            text=self.gui_manager.t("remove_counter"),
            command=self.remove_counter,
        )
        self.remove_counter_button.pack(fill=tk.X, pady=2)
        self._refresh_counters_tree()

        self.series_frame = ttk.LabelFrame(
            self.right_frame, text=self.gui_manager.t("series_metrics"), padding=10
        )
        self.series_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        series_container = ttk.Frame(self.series_frame)
        series_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.series_tree = ttk.Treeview(
            series_container,
            columns=("name", "label", "source", "max_points", "unit"),
            show="headings",
            height=5,
        )
        for col in ("name", "label", "source", "max_points", "unit"):
            self.series_tree.heading(col, text=self.gui_manager.t(f"series_{col}"))
        self.series_tree.column("name", width=120, anchor=tk.W)
        self.series_tree.column("label", width=160, anchor=tk.W)
        self.series_tree.column("source", width=200, anchor=tk.W)
        self.series_tree.column("max_points", width=90, anchor=tk.W)
        self.series_tree.column("unit", width=80, anchor=tk.W)
        series_scroll = ttk.Scrollbar(
            series_container, orient=tk.VERTICAL, command=self.series_tree.yview
        )
        self.series_tree.configure(yscrollcommand=series_scroll.set)
        self.series_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        series_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        series_buttons = ttk.Frame(self.series_frame)
        series_buttons.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.add_series_button = ttk.Button(
            series_buttons,
            text=self.gui_manager.t("add_series"),
            command=self.add_series,
        )
        self.add_series_button.pack(fill=tk.X, pady=2)
        self.edit_series_button = ttk.Button(
            series_buttons,
            text=self.gui_manager.t("edit_series"),
            command=self.edit_series,
        )
        self.edit_series_button.pack(fill=tk.X, pady=2)
        self.remove_series_button = ttk.Button(
            series_buttons,
            text=self.gui_manager.t("remove_series"),
            command=self.remove_series,
        )
        self.remove_series_button.pack(fill=tk.X, pady=2)
        self._refresh_series_tree()

        self.steps_frame = ttk.LabelFrame(
            self.right_frame, text=self.gui_manager.t("steps"), padding=10
        )
        self.steps_frame.pack(fill=tk.BOTH, expand=True)

        self.steps_tree = ttk.Treeview(
            self.steps_frame,
            columns=("type", "summary"),
            show="headings",
            height=8,
        )
        self.steps_tree.heading("type", text=self.gui_manager.t("step_type"))
        self.steps_tree.heading("summary", text=self.gui_manager.t("step_summary"))
        self.steps_tree.column("type", width=120)
        self.steps_tree.column("summary", width=320)
        self.steps_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        steps_scroll = ttk.Scrollbar(
            self.steps_frame, orient=tk.VERTICAL, command=self.steps_tree.yview
        )
        self.steps_tree.configure(yscrollcommand=steps_scroll.set)
        steps_scroll.pack(side=tk.LEFT, fill=tk.Y)

        step_buttons = ttk.Frame(self.steps_frame)
        step_buttons.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        self.add_step_button = ttk.Button(
            step_buttons, text=self.gui_manager.t("add_step"), command=self.add_step
        )
        self.add_step_button.pack(fill=tk.X, pady=2)
        self.edit_step_button = ttk.Button(
            step_buttons, text=self.gui_manager.t("edit_step"), command=self.edit_step
        )
        self.edit_step_button.pack(fill=tk.X, pady=2)
        self.remove_step_button = ttk.Button(
            step_buttons,
            text=self.gui_manager.t("remove_step"),
            command=self.remove_step,
        )
        self.remove_step_button.pack(fill=tk.X, pady=2)
        self.step_move_up_button = ttk.Button(
            step_buttons,
            text=self.gui_manager.t("move_up"),
            command=lambda: self.move_step(-1),
        )
        self.step_move_up_button.pack(fill=tk.X, pady=(8, 2))
        self.step_move_down_button = ttk.Button(
            step_buttons,
            text=self.gui_manager.t("move_down"),
            command=lambda: self.move_step(1),
        )
        self.step_move_down_button.pack(fill=tk.X, pady=2)

        bottom = ttk.Frame(self.right_frame)
        bottom.pack(fill=tk.X, pady=10)
        self.reload_button = ttk.Button(
            bottom,
            text=self.gui_manager.t("load_tasks"),
            command=self.load_tasks_from_file,
        )
        self.reload_button.pack(side=tk.LEFT)
        self.save_button = ttk.Button(
            bottom,
            text=self.gui_manager.t("save_tasks"),
            command=self.save_tasks_to_file,
        )
        self.save_button.pack(side=tk.RIGHT)

    def _add_entry_row(self, parent, row, label, variable):
        lbl = ttk.Label(parent, text=label + ":")
        lbl.grid(row=row, column=0, sticky=tk.W, pady=2)
        entry = ttk.Entry(parent, textvariable=variable, width=40)
        entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        self._entry_labels.append(lbl)
        return entry

    def _task_ids_for_picker(self) -> List[str]:
        """Return task IDs for nested pickers."""
        return [task.get("id", "") for task in self.tasks if task.get("id")]

    def _update_trigger_fields(self):
        trigger_type = (self.trigger_type_var.get() or "interval").lower()
        if trigger_type == "sheduled":
            trigger_type = "scheduled"
            self.trigger_type_var.set("scheduled")

        if trigger_type == "scheduled":
            self.trigger_min_label.grid_remove()
            self.trigger_min_entry.grid_remove()
            self.trigger_max_label.grid_remove()
            self.trigger_max_entry.grid_remove()
            self.trigger_times_label.grid(row=1, column=0, sticky=tk.W, pady=2)
            self.trigger_times_entry.grid(row=1, column=1, sticky=tk.W, padx=5)
        else:
            self.trigger_times_label.grid_remove()
            self.trigger_times_entry.grid_remove()
            self.trigger_min_label.grid(row=1, column=0, sticky=tk.W, pady=2)
            self.trigger_min_entry.grid(row=1, column=1, sticky=tk.W, padx=5)
            self.trigger_max_label.grid(row=2, column=0, sticky=tk.W, pady=2)
            self.trigger_max_entry.grid(row=2, column=1, sticky=tk.W, padx=5)

        self.trigger_run_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)

    def _validate_scheduled_times(self, times: List[str]) -> bool:
        if not times:
            return False
        for value in times:
            cleaned = value.strip()
            if not cleaned:
                return False
            for fmt in ("%H:%M:%S", "%H:%M"):
                try:
                    datetime.strptime(cleaned, fmt)
                    break
                except ValueError:
                    continue
            else:
                return False
        return True

    def _normalize_counters(self, counters) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if isinstance(counters, dict):
            for name, cfg in counters.items():
                entry: Dict[str, Any] = {"name": str(name)}
                if isinstance(cfg, dict):
                    for key in ("label", "source", "accumulate", "unit"):
                        if cfg.get(key) is not None:
                            entry[key] = str(cfg.get(key))
                elif cfg is not None:
                    entry["label"] = str(cfg)
                normalized.append(entry)
        elif isinstance(counters, list):
            for item in counters:
                if isinstance(item, dict):
                    entry = dict(item)
                    entry["name"] = str(
                        entry.get("name") or entry.get("key") or entry.get("id") or ""
                    )
                    normalized.append(entry)
        cleaned: List[Dict[str, Any]] = []
        for entry in normalized:
            name = (entry.get("name") or "").strip()
            if not name:
                continue
            cleaned.append(
                {
                    "name": name,
                    "label": (entry.get("label") or "").strip(),
                    "source": (entry.get("source") or "").strip(),
                    "accumulate": (entry.get("accumulate") or "sum")
                    .strip()
                    .lower()
                    or "sum",
                    "unit": (entry.get("unit") or "").strip(),
                }
            )
        return cleaned

    def _load_counters_from_task(self, task: Dict[str, Any]) -> None:
        stats = task.get("stats", {}) if isinstance(task, dict) else {}
        counters = stats.get("counters")
        self.current_counters = self._normalize_counters(counters)
        self._refresh_counters_tree()

    def _refresh_counters_tree(self) -> None:
        if not hasattr(self, "counters_tree"):
            return
        for item in self.counters_tree.get_children():
            self.counters_tree.delete(item)
        for entry in self.current_counters:
            display_accumulate = entry.get("accumulate", "sum")
            if display_accumulate == "sum":
                display_accumulate = self.gui_manager.t("counter_accumulate_sum")
            elif display_accumulate == "set":
                display_accumulate = self.gui_manager.t("counter_accumulate_set")
            self.counters_tree.insert(
                "",
                tk.END,
                values=(
                    entry.get("name", ""),
                    entry.get("label", ""),
                    entry.get("source", ""),
                    display_accumulate,
                    entry.get("unit", ""),
                ),
            )

    def _current_counter_index(self) -> Optional[int]:
        selection = self.counters_tree.selection()
        if not selection:
            return None
        try:
            return self.counters_tree.index(selection[0])
        except Exception:
            return None

    def add_counter(self):
        dialog = CounterEditorDialog(self, self.gui_manager)
        result = dialog.show()
        if not result:
            return
        name = result.get("name", "")
        if any(counter.get("name") == name for counter in self.current_counters):
            messagebox.showwarning(
                "Warning",
                self.gui_manager.t("counter_duplicate"),
                parent=self,
            )
            return
        self.current_counters.append(result)
        self._refresh_counters_tree()

    def edit_counter(self):
        idx = self._current_counter_index()
        if idx is None:
            return
        dialog = CounterEditorDialog(
            self, self.gui_manager, initial=self.current_counters[idx]
        )
        result = dialog.show()
        if not result:
            return
        name = result.get("name", "")
        for i, counter in enumerate(self.current_counters):
            if i != idx and counter.get("name") == name:
                messagebox.showwarning(
                    "Warning",
                    self.gui_manager.t("counter_duplicate"),
                    parent=self,
                )
                return
        self.current_counters[idx] = result
        self._refresh_counters_tree()

    def remove_counter(self):
        idx = self._current_counter_index()
        if idx is None:
            return
        del self.current_counters[idx]
        self._refresh_counters_tree()

    def _bind_right_mousewheel(self):
        if self._right_mousewheel_bound or not getattr(self, "right_canvas", None):
            return
        try:
            self.right_canvas.bind_all("<MouseWheel>", self._on_right_mousewheel)
            self.right_canvas.bind_all("<Button-4>", self._on_right_mousewheel)
            self.right_canvas.bind_all("<Button-5>", self._on_right_mousewheel)
            self._right_mousewheel_bound = True
        except Exception:
            pass

    def _unbind_right_mousewheel(self):
        if not self._right_mousewheel_bound or not getattr(self, "right_canvas", None):
            return
        try:
            self.right_canvas.unbind_all("<MouseWheel>")
            self.right_canvas.unbind_all("<Button-4>")
            self.right_canvas.unbind_all("<Button-5>")
        except Exception:
            pass
        finally:
            self._right_mousewheel_bound = False

    def _on_right_mousewheel(self, event):
        if not getattr(self, "right_canvas", None):
            return
        try:
            if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
                self.right_canvas.yview_scroll(-1, "units")
            else:
                self.right_canvas.yview_scroll(1, "units")
        except Exception:
            pass

    def _normalize_series(self, series) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if isinstance(series, dict):
            for name, cfg in series.items():
                entry: Dict[str, Any] = {"name": str(name)}
                if isinstance(cfg, dict):
                    for key in ("label", "source", "max_points", "unit"):
                        if cfg.get(key) is not None:
                            entry[key] = cfg.get(key)
                normalized.append(entry)
        elif isinstance(series, list):
            for item in series:
                if isinstance(item, dict):
                    entry = dict(item)
                    entry["name"] = str(
                        entry.get("name") or entry.get("key") or entry.get("id") or ""
                    )
                    normalized.append(entry)
        cleaned: List[Dict[str, Any]] = []
        for entry in normalized:
            name = (entry.get("name") or "").strip()
            if not name:
                continue
            try:
                max_points = int(entry.get("max_points") or 50)
            except Exception:
                max_points = 50
            cleaned.append(
                {
                    "name": name,
                    "label": (entry.get("label") or "").strip(),
                    "source": (entry.get("source") or "").strip(),
                    "max_points": max_points,
                    "unit": (entry.get("unit") or "").strip(),
                }
            )
        return cleaned

    def _load_series_from_task(self, task: Dict[str, Any]) -> None:
        stats = task.get("stats", {}) if isinstance(task, dict) else {}
        series = stats.get("series")
        self.current_series = self._normalize_series(series)
        self._refresh_series_tree()

    def _refresh_series_tree(self) -> None:
        if not hasattr(self, "series_tree"):
            return
        for item in self.series_tree.get_children():
            self.series_tree.delete(item)
        for entry in self.current_series:
            self.series_tree.insert(
                "",
                tk.END,
                values=(
                    entry.get("name", ""),
                    entry.get("label", ""),
                    entry.get("source", ""),
                    entry.get("max_points", 50),
                    entry.get("unit", ""),
                ),
            )

    def _current_series_index(self) -> Optional[int]:
        selection = self.series_tree.selection()
        if not selection:
            return None
        try:
            return self.series_tree.index(selection[0])
        except Exception:
            return None

    def add_series(self):
        dialog = SeriesEditorDialog(self, self.gui_manager)
        result = dialog.show()
        if not result:
            return
        name = result.get("name")
        if any(series.get("name") == name for series in self.current_series):
            messagebox.showwarning(
                "Warning",
                self.gui_manager.t("series_duplicate"),
                parent=self,
            )
            return
        self.current_series.append(result)
        self._refresh_series_tree()

    def edit_series(self):
        idx = self._current_series_index()
        if idx is None:
            return
        dialog = SeriesEditorDialog(
            self, self.gui_manager, initial=self.current_series[idx]
        )
        result = dialog.show()
        if not result:
            return
        name = result.get("name")
        for i, series in enumerate(self.current_series):
            if i != idx and series.get("name") == name:
                messagebox.showwarning(
                    "Warning",
                    self.gui_manager.t("series_duplicate"),
                    parent=self,
                )
                return
        self.current_series[idx] = result
        self._refresh_series_tree()

    def remove_series(self):
        idx = self._current_series_index()
        if idx is None:
            return
        del self.current_series[idx]
        self._refresh_series_tree()


    def _is_protected_task(self, task_id: Optional[str]) -> bool:
        if not task_id:
            return False
        if task_id in PROTECTED_TASK_IDS:
            return True
        for task in self.tasks:
            if task.get("id") == task_id:
                if task.get("protected-system-task") or task.get("protected_system_task"):
                    return True
                break
        return False

    def _apply_task_protection(self, task_id: Optional[str]):
        protected = self._is_protected_task(task_id)
        id_state = "readonly" if protected else "normal"
        name_state = "readonly" if protected else "normal"
        if getattr(self, "task_id_entry", None):
            self.task_id_entry.configure(state=id_state)
        if getattr(self, "task_name_entry", None):
            self.task_name_entry.configure(state=name_state)
        if getattr(self, "delete_task_button", None):
            self.delete_task_button.configure(
                state=tk.DISABLED if protected else tk.NORMAL
            )
        state = tk.DISABLED if protected else tk.NORMAL
        if getattr(self, "add_counter_button", None):
            self.add_counter_button.configure(state=state)
        if getattr(self, "edit_counter_button", None):
            self.edit_counter_button.configure(state=state)
        if getattr(self, "remove_counter_button", None):
            self.remove_counter_button.configure(state=state)
        if getattr(self, "add_series_button", None):
            self.add_series_button.configure(state=state)
        if getattr(self, "edit_series_button", None):
            self.edit_series_button.configure(state=state)
        if getattr(self, "remove_series_button", None):
            self.remove_series_button.configure(state=state)
        self._update_protected_hint(protected)

    def _update_protected_hint(self, protected: bool):
        if getattr(self, "protected_hint_var", None) is None:
            return
        if protected:
            self.protected_hint_var.set(self.gui_manager.t("task_protected_hint"))
        else:
            self.protected_hint_var.set("")

    def load_tasks_from_file(self):
        try:
            load_task_definitions(TASKS_PATH)
            data = (
                copy.deepcopy(TASKS_CONFIG)
                if TASKS_CONFIG
                else {"version": 1, "tasks": []}
            )
            self.tasks_version = data.get("version", 1)
            self.tasks = copy.deepcopy(data.get("tasks", []))
        except Exception as exc:
            self.gui_manager.log_message(f"Error loading tasks: {exc}", "ERROR")
            self.tasks = []
        self.populate_task_list()
        if self.tasks:
            self.task_listbox.selection_set(0)
            self.on_task_select()
        else:
            self.clear_form()

    def populate_task_list(self):
        self._suspend = True
        self.task_listbox.delete(0, tk.END)
        for task in self.tasks:
            label = task.get("name") or task.get("id") or "task"
            enabled = task.get("enabled", True)
            is_protected = bool(
                task.get("protected-system-task")
                or task.get("protected_system_task")
                or (task.get("id") in PROTECTED_TASK_IDS)
            )
            prefix = "[SYS] " if is_protected else ""
            self.task_listbox.insert(
                tk.END, f"{'✓ ' if enabled else ''}{prefix}{label}"
            )
        self._suspend = False

    def clear_form(self):
        self.current_index = None
        self.task_id_var.set("")
        self.task_name_var.set("")
        self.task_desc_var.set("")
        self.task_enabled_var.set(True)
        self.trigger_type_var.set("interval")
        self.trigger_min_var.set("")
        self.trigger_max_var.set("")
        self.trigger_run_var.set(False)
        self.trigger_times_var.set("")
        self.current_steps = []
        self.update_steps_tree()
        self.current_counters = []
        self._refresh_counters_tree()
        self.current_series = []
        self._refresh_series_tree()
        self._update_trigger_fields()
        self._apply_task_protection(None)

    def on_task_select(self, _event=None):
        if self._suspend:
            return
        idx = self._current_task_index()
        if idx is None:
            return
        if self.current_index is not None and self.current_index != idx:
            if not self.save_current_task():
                self.task_listbox.selection_clear(0, tk.END)
                if self.current_index is not None:
                    self.task_listbox.selection_set(self.current_index)
                return
        self.current_index = idx
        task = copy.deepcopy(self.tasks[idx])
        self.task_id_var.set(task.get("id", ""))
        self.task_name_var.set(task.get("name", ""))
        self.task_desc_var.set(task.get("description", ""))
        self.task_enabled_var.set(bool(task.get("enabled", True)))

        trigger = task.get("trigger", {}) or {}
        trigger_type = str(trigger.get("type", "interval")).lower()
        if trigger_type == "sheduled":
            trigger_type = "scheduled"
        self.trigger_type_var.set(trigger_type)
        self.trigger_min_var.set(
            self._value_to_text(trigger.get("min_seconds", ""))
        )
        self.trigger_max_var.set(
            self._value_to_text(trigger.get("max_seconds", ""))
        )
        self.trigger_run_var.set(bool(trigger.get("run_at_start", False)))
        if trigger_type == "scheduled":
            times = trigger.get("times", [])
            if isinstance(times, list):
                self.trigger_times_var.set(", ".join(str(t) for t in times))
            elif isinstance(times, str):
                self.trigger_times_var.set(times)
            else:
                self.trigger_times_var.set("")
        else:
            self.trigger_times_var.set("")
        self._update_trigger_fields()

        self.current_steps = copy.deepcopy(task.get("steps", []))
        self.update_steps_tree()
        self._load_counters_from_task(task)
        self._load_series_from_task(task)
        self._apply_task_protection(self.task_id_var.get())

    def save_current_task(self) -> bool:
        if self.current_index is None:
            return True
        task_data = self._collect_task_data()
        if not task_data:
            return False
        self.tasks[self.current_index] = task_data
        self.populate_task_list()
        self.task_listbox.selection_set(self.current_index)
        return True

    def _collect_task_data(self) -> Optional[Dict[str, Any]]:
        task_id = self.task_id_var.get().strip()
        if not task_id:
            messagebox.showwarning("Warning", "Task ID is required", parent=self)
            return None
        name = self.task_name_var.get().strip() or task_id
        description = self.task_desc_var.get().strip()
        enabled = bool(self.task_enabled_var.get())

        original_task = (
            self.tasks[self.current_index] if self.current_index is not None else None
        )
        if original_task and self._is_protected_task(original_task.get("id")):
            protected_id = original_task.get("id")
            if task_id != protected_id:
                messagebox.showwarning(
                    "Warning", self.gui_manager.t("task_protected"), parent=self
                )
                self.task_id_var.set(protected_id)
                return None
            original_name = original_task.get("name", protected_id)
            if name != original_name:
                messagebox.showwarning(
                    "Warning", self.gui_manager.t("task_protected"), parent=self
                )
                self.task_name_var.set(original_name)
                return None

        trigger_type = self.trigger_type_var.get().strip().lower() or "interval"
        if trigger_type == "sheduled":
            trigger_type = "scheduled"
        trigger = {"type": trigger_type}
        if trigger_type == "scheduled":
            times_text = self.trigger_times_var.get().strip()
            times = [t.strip() for t in re.split(r"[,\s]+", times_text) if t.strip()]
            if not times:
                messagebox.showwarning(
                    "Warning",
                    self.gui_manager.t("trigger_times_invalid"),
                    parent=self,
                )
                return None
            if not self._validate_scheduled_times(times):
                messagebox.showwarning(
                    "Warning",
                    self.gui_manager.t("trigger_times_invalid"),
                    parent=self,
                )
                return None
            trigger["times"] = times
        else:
            min_text = self.trigger_min_var.get().strip()
            max_text = self.trigger_max_var.get().strip()
            if min_text:
                trigger["min_seconds"] = self._text_to_value(min_text)
            if max_text:
                trigger["max_seconds"] = self._text_to_value(max_text)
        if self.trigger_run_var.get():
            trigger["run_at_start"] = True

        counters_copy = copy.deepcopy(self.current_counters)
        original_stats = (
            copy.deepcopy(original_task.get("stats", {})) if original_task else {}
        )
        stats_block: Dict[str, Any] = {
            key: value
            for key, value in (original_stats or {}).items()
            if key not in {"counters", "series"}
        }
        counters_dict: Dict[str, Any] = {}
        for index, counter in enumerate(counters_copy):
            counter_name = counter.get("name") or f"metric_{index + 1}"
            counter_name = counter_name.strip()
            if not counter_name:
                continue
            cfg: Dict[str, Any] = {}
            if counter.get("label"):
                cfg["label"] = counter["label"]
            if counter.get("source"):
                cfg["source"] = counter["source"]
            accumulate = counter.get("accumulate") or "sum"
            if accumulate not in {"sum", "set"}:
                accumulate = "sum"
            cfg["accumulate"] = accumulate
            if counter.get("unit"):
                cfg["unit"] = counter["unit"]
            counters_dict[counter_name] = cfg
        if counters_dict:
            stats_block["counters"] = counters_dict
        elif "counters" in stats_block:
            stats_block.pop("counters", None)

        series_dict: Dict[str, Any] = {}
        for index, series in enumerate(copy.deepcopy(self.current_series)):
            series_name = series.get("name") or f"series_{index + 1}"
            series_name = series_name.strip()
            if not series_name:
                continue
            cfg: Dict[str, Any] = {}
            if series.get("label"):
                cfg["label"] = series["label"]
            if series.get("source"):
                cfg["source"] = series["source"]
            try:
                cfg["max_points"] = int(series.get("max_points") or 50)
            except Exception:
                cfg["max_points"] = 50
            if series.get("unit"):
                cfg["unit"] = series["unit"]
            series_dict[series_name] = cfg
        if series_dict:
            stats_block["series"] = series_dict
        elif "series" in stats_block:
            stats_block.pop("series", None)

        protected_flag = False
        if original_task and (
            original_task.get("protected-system-task")
            or original_task.get("protected_system_task")
        ):
            protected_flag = True
        elif task_id in PROTECTED_TASK_IDS:
            protected_flag = True

        task_data = {
            "id": task_id,
            "name": name,
            "description": description,
            "enabled": enabled,
            "trigger": trigger,
            "steps": copy.deepcopy(self.current_steps),
        }
        if stats_block:
            task_data["stats"] = stats_block
        if protected_flag:
            task_data["protected-system-task"] = True
        return task_data

    def update_steps_tree(self):
        self.steps_tree.delete(*self.steps_tree.get_children())
        for idx, step in enumerate(self.current_steps):
            self.steps_tree.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(step.get("type", ""), self._summarize_step(step)),
            )

    def add_task(self):
        if not self.save_current_task():
            return
        base = "task"
        existing = {task.get("id") for task in self.tasks}
        suffix = 1
        new_id = f"{base}_{suffix}"
        while new_id in existing:
            suffix += 1
            new_id = f"{base}_{suffix}"
        new_task = {
            "id": new_id,
            "name": f"Task {suffix}",
            "description": "",
            "enabled": True,
            "trigger": {
                "type": "interval",
                "min_seconds": "{{config.random_pause_min}}",
                "max_seconds": "{{config.random_pause_max}}",
            },
            "steps": [],
        }
        self.tasks.append(new_task)
        self.populate_task_list()
        self.task_listbox.selection_clear(0, tk.END)
        index = len(self.tasks) - 1
        self.task_listbox.selection_set(index)
        self.current_index = index
        self.on_task_select()

    def duplicate_task(self):
        idx = self._current_task_index()
        if idx is None:
            return
        if not self.save_current_task():
            return
        dup = copy.deepcopy(self.tasks[idx])
        dup_id = dup.get("id", "task")
        dup.pop("protected-system-task", None)
        dup.pop("protected_system_task", None)
        suffix = 1
        existing = {task.get("id") for task in self.tasks}
        new_id = f"{dup_id}_copy"
        while new_id in existing:
            suffix += 1
            new_id = f"{dup_id}_copy{suffix}"
        dup["id"] = new_id
        dup["name"] = dup.get("name", dup_id) + " (copy)"
        self.tasks.insert(idx + 1, dup)
        self.populate_task_list()
        self.task_listbox.selection_clear(0, tk.END)
        self.task_listbox.selection_set(idx + 1)
        self.current_index = idx + 1
        self.on_task_select()

    def delete_task(self):
        idx = self._current_task_index()
        if idx is None:
            return
        task_id = self.tasks[idx].get("id")
        if self._is_protected_task(task_id):
            messagebox.showwarning(
                "Warning",
                self.gui_manager.t("task_protected"),
                parent=self,
            )
            return
        if not messagebox.askyesno(
            "Confirm",
            f"Delete task '{self.tasks[idx].get('name', self.tasks[idx].get('id', 'task'))}'?",
            parent=self,
        ):
            return
        del self.tasks[idx]
        self.populate_task_list()
        if self.tasks:
            new_idx = min(idx, len(self.tasks) - 1)
            self.task_listbox.selection_set(new_idx)
            self.current_index = new_idx
            self.on_task_select()
        else:
            self.clear_form()

    def move_task(self, direction: int):
        idx = self._current_task_index()
        if idx is None:
            return
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(self.tasks):
            return
        if not self.save_current_task():
            return
        self.tasks[idx], self.tasks[new_idx] = self.tasks[new_idx], self.tasks[idx]
        self.populate_task_list()
        self.task_listbox.selection_set(new_idx)
        self.current_index = new_idx
        self.on_task_select()

    def add_step(self):
        dialog = TaskStepDialog(
            self,
            self.gui_manager,
            self.gui_manager.get_template_names(),
            self._task_ids_for_picker(),
        )
        step = dialog.show()
        if step:
            self.current_steps.append(step)
            self.update_steps_tree()

    def edit_step(self):
        idx = self._current_step_index()
        if idx is None:
            return
        dialog = TaskStepDialog(
            self,
            self.gui_manager,
            self.gui_manager.get_template_names(),
            self._task_ids_for_picker(),
            initial=self.current_steps[idx],
        )
        step = dialog.show()
        if step:
            self.current_steps[idx] = step
            self.update_steps_tree()

    def remove_step(self):
        idx = self._current_step_index()
        if idx is None:
            return
        del self.current_steps[idx]
        self.update_steps_tree()

    def move_step(self, direction: int):
        idx = self._current_step_index()
        if idx is None:
            return
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(self.current_steps):
            return
        self.current_steps[idx], self.current_steps[new_idx] = (
            self.current_steps[new_idx],
            self.current_steps[idx],
        )
        self.update_steps_tree()
        self.steps_tree.selection_set(str(new_idx))

    def save_tasks_to_file(self):
        if not self.save_current_task():
            return
        global TASKS_CONFIG
        data = copy.deepcopy(TASKS_CONFIG) if TASKS_CONFIG else {}
        data["version"] = self.tasks_version
        data["tasks"] = copy.deepcopy(self.tasks)
        try:
            os.makedirs(os.path.dirname(TASKS_PATH), exist_ok=True)
            with open(TASKS_PATH, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            TASKS_CONFIG = copy.deepcopy(data)
            load_task_definitions(TASKS_PATH)
            self.gui_manager.log_message("Tasks saved", "INFO")
            self.load_tasks_from_file()
        except Exception as exc:
            self.gui_manager.log_message(f"Error saving tasks: {exc}", "ERROR")

    def refresh_texts(self):
        self.list_label.configure(text=self.gui_manager.t("task_list"))
        self.details_frame.configure(text=self.gui_manager.t("task"))
        self.trigger_frame.configure(text=self.gui_manager.t("trigger"))
        self.steps_frame.configure(text=self.gui_manager.t("steps"))
        self.add_task_button.configure(text=self.gui_manager.t("add_task"))
        self.duplicate_task_button.configure(text=self.gui_manager.t("duplicate_task"))
        self.delete_task_button.configure(text=self.gui_manager.t("delete_task"))
        self.move_up_button.configure(text=self.gui_manager.t("move_up"))
        self.move_down_button.configure(text=self.gui_manager.t("move_down"))
        self.add_step_button.configure(text=self.gui_manager.t("add_step"))
        self.edit_step_button.configure(text=self.gui_manager.t("edit_step"))
        self.remove_step_button.configure(text=self.gui_manager.t("remove_step"))
        self.step_move_up_button.configure(text=self.gui_manager.t("move_up"))
        self.step_move_down_button.configure(text=self.gui_manager.t("move_down"))
        self.reload_button.configure(text=self.gui_manager.t("load_tasks"))
        self.save_button.configure(text=self.gui_manager.t("save_tasks"))
        self.enabled_check.configure(text=self.gui_manager.t("task_enabled"))
        self.trigger_type_combo.configure(values=["interval", "scheduled"])
        self.protected_list_hint.configure(text=self.gui_manager.t("task_protected_list_hint"))
        steps_headings = {
            "type": self.gui_manager.t("step_type"),
            "summary": self.gui_manager.t("step_summary"),
        }
        self.steps_tree.heading("type", text=steps_headings["type"])
        self.steps_tree.heading("summary", text=steps_headings["summary"])
        label_keys = [
            "task_id",
            "task_name",
            "task_description",
        ]
        for lbl, key in zip(self._entry_labels, label_keys):
            lbl.configure(text=self.gui_manager.t(key) + ":")
        self.trigger_type_label.configure(text=self.gui_manager.t("trigger_type") + ":")
        self.trigger_min_label.configure(
            text=self.gui_manager.t("trigger_min_seconds") + ":"
        )
        self.trigger_max_label.configure(
            text=self.gui_manager.t("trigger_max_seconds") + ":"
        )
        self.trigger_run_check.configure(
            text=self.gui_manager.t("trigger_run_at_start")
        )
        self.stats_frame.configure(text=self.gui_manager.t("task_metrics"))
        self.add_counter_button.configure(text=self.gui_manager.t("add_counter"))
        self.edit_counter_button.configure(text=self.gui_manager.t("edit_counter"))
        self.remove_counter_button.configure(text=self.gui_manager.t("remove_counter"))
        for col in ("name", "label", "source", "accumulate", "unit"):
            self.counters_tree.heading(col, text=self.gui_manager.t(f"counter_{col}"))
        self._refresh_counters_tree()
        self._update_protected_hint(self._is_protected_task(self.task_id_var.get()))
        self.populate_task_list()
        self.update_steps_tree()

    def _summarize_step(self, step: Dict[str, Any]) -> str:
        return summarize_step_description(step)

    def _text_to_value(self, text: str) -> Any:
        if not text:
            return ""
        if text.startswith("{{") and text.endswith("}}"):  # placeholder
            return text
        try:
            if "." in text:
                return float(text)
            return int(text)
        except ValueError:
            return text

    def _value_to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return str(value)
        return str(value)

    def _current_task_index(self) -> Optional[int]:
        selection = self.task_listbox.curselection()
        if not selection:
            return None
        return selection[0]

    def _current_step_index(self) -> Optional[int]:
        selection = self.steps_tree.selection()
        if not selection:
            return None
        return int(selection[0])


class LogTab(ttk.Frame):
    """Tab Log viewer"""

    def __init__(self, parent, gui_manager):
        super().__init__(parent)
        self.gui_manager = gui_manager
        self.auto_scroll = True

        self.setup_ui()

    def setup_ui(self):
        # Control panel
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(control_frame, text=self.gui_manager.t("log_level") + ":").pack(
            side=tk.LEFT, padx=5
        )

        self.log_level_var = tk.StringVar(value="ALL")
        log_level_combo = ttk.Combobox(
            control_frame,
            textvariable=self.log_level_var,
            values=["ALL", "DEBUG", "INFO", "WARNING", "ERROR"],
            state="readonly",
            width=10,
        )
        log_level_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_frame, text=self.gui_manager.t("clear_log"), command=self.clear_log
        ).pack(side=tk.LEFT, padx=5)

        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            control_frame,
            text=self.gui_manager.t("auto_scroll"),
            variable=self.auto_scroll_var,
            command=self.toggle_auto_scroll,
        ).pack(side=tk.LEFT, padx=5)

        # Log text widget
        log_frame = ttk.Frame(self)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Configure tags for different log levels
        self.log_text.tag_config("DEBUG", foreground="gray")
        self.log_text.tag_config("INFO", foreground="black")
        self.log_text.tag_config("WARNING", foreground="orange")
        self.log_text.tag_config("ERROR", foreground="red")

    def add_log(self, message, level="INFO"):
        """Add log entry"""
        log_level = self.log_level_var.get()

        levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        message_level = levels.get(level, 1)
        filter_level = levels.get(log_level, 0)

        if log_level != "ALL" and message_level < filter_level:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"

        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry, level)
        self.log_text.configure(state=tk.DISABLED)

        if self.auto_scroll:
            self.log_text.see(tk.END)

    def clear_log(self):
        """Clear log text"""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def toggle_auto_scroll(self):
        """Toggle auto-scroll"""
        self.auto_scroll = self.auto_scroll_var.get()

    def refresh_texts(self):
        """Refresh visible strings for current language."""
        self.profile_label.config(text=self.gui_manager.t("profile_label"))
        self.profile_apply_btn.config(text=self.gui_manager.t("profile_apply"))
        if not self.gui_manager.available_profiles:
            self.profile_status.config(
                text=self.gui_manager.t("profile_none_available")
            )


class SettingsTab(ttk.Frame):
    """Tab Settings editor"""

    def __init__(self, parent, gui_manager):
        super().__init__(parent)
        self.gui_manager = gui_manager
        self.config = {}
        self.comments = {}
        self.config_widgets = {}
        self.comment_vars = {}
        self.canvas = None
        self._mousewheel_bound = False

        self.setup_ui()

    def setup_ui(self):
        # Profile selection header
        profile_frame = ttk.Frame(self)
        profile_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.profile_label = ttk.Label(
            profile_frame, text=self.gui_manager.t("profile_label")
        )
        self.profile_label.pack(side=tk.LEFT, padx=5)
        self.profile_var = tk.StringVar(value=self.gui_manager.profile_name or "")
        self.profile_combo = ttk.Combobox(
            profile_frame,
            textvariable=self.profile_var,
            values=self.gui_manager.available_profiles,
            state="readonly",
            width=25,
        )
        self.profile_combo.pack(side=tk.LEFT, padx=5)
        self.profile_apply_btn = ttk.Button(
            profile_frame,
            text=self.gui_manager.t("profile_apply"),
            command=self._on_profile_change,
        )
        self.profile_apply_btn.pack(side=tk.LEFT, padx=5)
        self.profile_status = ttk.Label(profile_frame, text="", foreground="orange")
        self.profile_status.pack(side=tk.LEFT, padx=10)
        if not self.gui_manager.available_profiles:
            self.profile_combo.configure(state="disabled")
            self.profile_status.configure(
                text=self.gui_manager.t("profile_none_available"),
                foreground="red",
            )

        # Scrollable frame for settings
        self.canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)

        self.settings_frame = ttk.Frame(self.canvas)
        self.settings_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.settings_frame, anchor=tk.NW)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)
        self.settings_frame.bind("<Enter>", self._bind_mousewheel)
        self.settings_frame.bind("<Leave>", self._unbind_mousewheel)

        # Button frame at bottom
        button_frame = ttk.Frame(self)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        ttk.Button(
            button_frame, text=self.gui_manager.t("save"), command=self.save_config
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame, text=self.gui_manager.t("load"), command=self.load_config
        ).pack(side=tk.LEFT, padx=5)

    def _on_profile_change(self):
        """Handle profile selection changes from the Settings tab."""
        selected = self.profile_var.get().strip()
        success, message = self.gui_manager.apply_profile_selection(selected)
        self.profile_status.configure(
            text=message, foreground="green" if success else "red"
        )
        # Refresh available profiles in case the set changed
        updated_profiles = list_profiles()
        self.gui_manager.available_profiles = updated_profiles
        self.profile_combo.configure(values=updated_profiles)
        self.profile_combo.configure(
            state="readonly" if updated_profiles else "disabled"
        )
        if success:
            if self.gui_manager.profile_name:
                self.profile_var.set(self.gui_manager.profile_name)
        else:
            self.profile_var.set(self.gui_manager.profile_name or "")

    def _bind_mousewheel(self, _event=None):
        if self.canvas is None or self._mousewheel_bound:
            return
        try:
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
            self.canvas.bind_all("<Button-4>", self._on_mousewheel)
            self.canvas.bind_all("<Button-5>", self._on_mousewheel)
            self._mousewheel_bound = True
        except Exception:
            pass

    def _unbind_mousewheel(self, _event=None):
        if self.canvas is None or not self._mousewheel_bound:
            return
        try:
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
        except Exception:
            pass
        finally:
            self._mousewheel_bound = False

    def _on_mousewheel(self, event):
        if self.canvas is None:
            return
        try:
            if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
                self.canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
                self.canvas.yview_scroll(1, "units")
        except Exception:
            pass

    def load_config(self, config_path=None):
        """Load configuration from file"""
        path = config_path or self.gui_manager.active_config_path()
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                self.config, self.comments = separate_config_comments(raw)
            else:
                self.config = self.get_default_config()
                self.comments = {}

            self._apply_default_comments()

            self.display_config()
            self.gui_manager.log_message(self.gui_manager.t("config_loaded"), "INFO")
        except Exception as e:
            self.gui_manager.log_message(
                f"{self.gui_manager.t('error_loading')}: {e}", "ERROR"
            )

    def save_config(self, config_path=None):
        """Save configuration to file"""
        global CFG_COMMENTS
        path = config_path or self.gui_manager.active_config_path()
        try:
            self.collect_config_values()

            data_to_save = merge_config_with_comments(self.config, self.comments)

            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    data_to_save, f, default_flow_style=False, sort_keys=False
                )

            try:
                CFG.update(copy.deepcopy(self.config))
                CFG_COMMENTS = dict(self.comments)
            except Exception:
                pass

            self.gui_manager.log_message(self.gui_manager.t("config_saved"), "INFO")
        except Exception as e:
            self.gui_manager.log_message(
                f"{self.gui_manager.t('error_saving')}: {e}", "ERROR"
            )

    def _apply_default_comments(self):
        if self.comments is None:
            self.comments = {}
        for key, text in DEFAULT_SETTING_COMMENTS.items():
            self.comments.setdefault(key, text)

    def get_default_config(self):
        """Return default configuration"""
        return {
            "general": {
                "bot_enabled": True,
                "check_interval": 5,
                "max_retries": 3,
                "timeout": 30,
            },
            "monitor": {
                "refresh_rate": 1000,
            },
        }

    def display_config(self):
        """Display configuration in UI"""
        # Clear existing widgets
        for widget in self.settings_frame.winfo_children():
            widget.destroy()
        self.config_widgets.clear()
        self.comment_vars.clear()

        skip_sections = {"templates", "rois"}
        general_items = {
            key: value
            for key, value in self.config.items()
            if key not in skip_sections and not isinstance(value, dict)
        }
        section_items = [
            (section, settings)
            for section, settings in self.config.items()
            if section not in skip_sections and isinstance(settings, dict)
        ]

        row = 0

        if general_items:
            general_frame = ttk.LabelFrame(
                self.settings_frame,
                text=self.gui_manager.t("general_settings"),
                padding=10,
            )
            general_frame.grid(row=row, column=0, sticky=tk.EW, padx=5, pady=5)
            self.settings_frame.columnconfigure(0, weight=1)
            row += 1

            for key, value in general_items.items():
                self.create_config_widget(general_frame, None, key, value)

        for section, settings in section_items:
            section_frame = ttk.LabelFrame(
                self.settings_frame, text=section.replace("_", " ").title(), padding=10
            )
            section_frame.grid(row=row, column=0, sticky=tk.EW, padx=5, pady=5)
            self.settings_frame.columnconfigure(0, weight=1)
            row += 1

            for key, value in settings.items():
                self.create_config_widget(section_frame, section, key, value)

    def refresh_texts(self):
        """Recreate section headers and labels with current language (minimal)."""
        try:
            self.profile_label.config(text=self.gui_manager.t("profile_label"))
            self.profile_apply_btn.config(text=self.gui_manager.t("profile_apply"))
            self.profile_combo.configure(values=self.gui_manager.available_profiles)
            self.profile_combo.configure(
                state="readonly" if self.gui_manager.available_profiles else "disabled"
            )
            if self.gui_manager.profile_name:
                self.profile_var.set(self.gui_manager.profile_name)
            self.display_config()
        except Exception:
            pass

    def create_config_widget(self, parent, section, key, value):
        """Create widget for configuration value"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        label_text = key.replace("_", " ").title()
        ttk.Label(frame, text=label_text + ":", width=20).pack(side=tk.LEFT, padx=5)

        widget_key = f"{section}.{key}" if section else key
        comment_text = self.comments.get(widget_key, "")
        self.comments.setdefault(widget_key, comment_text)

        if widget_key == "language":
            code_value = str(value) if value is not None else self.gui_manager.language
            available_codes = (
                list(self.gui_manager.available_languages)
                if getattr(self.gui_manager, "available_languages", None)
                else list(AVAILABLE_LANG_CODES)
            )
            if code_value and code_value not in available_codes:
                available_codes.append(code_value)
            display_map = {}
            display_values = []
            for code in available_codes:
                display = f"{get_language_display_name(code)} ({code})"
                display_map[display] = code
                if display not in display_values:
                    display_values.append(display)
            current_display = next(
                (disp for disp, code in display_map.items() if code == code_value),
                None,
            )
            if current_display is None:
                current_display = code_value or "en"
                display_map[current_display] = code_value or "en"
                display_values.append(current_display)
            var_value = tk.StringVar(value=code_value)
            display_var = tk.StringVar(value=current_display)

            def _sync_language_selection(_event=None):
                selected = display_var.get()
                resolved = display_map.get(selected, selected)
                if resolved:
                    var_value.set(resolved)

            widget = ttk.Combobox(
                frame,
                textvariable=display_var,
                values=display_values,
                state="readonly",
                width=25,
            )
            widget.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            widget.bind("<<ComboboxSelected>>", _sync_language_selection)
            _sync_language_selection()
        elif widget_key == "control_mode":
            var_value = tk.StringVar(value=str(value))
            widget = ttk.Combobox(
                frame,
                textvariable=var_value,
                values=["android", "windows"],
                state="readonly",
                width=15,
            )
            widget.pack(side=tk.LEFT, padx=5)
        elif isinstance(value, bool):
            var_value = tk.BooleanVar(value=value)
            widget = ttk.Checkbutton(frame, variable=var_value)
            widget.pack(side=tk.LEFT, padx=5)
        elif isinstance(value, int):
            var_value = tk.IntVar(value=value)
            widget = ttk.Spinbox(
                frame, from_=0, to=10000, textvariable=var_value, width=15
            )
            widget.pack(side=tk.LEFT, padx=5)
        elif isinstance(value, float):
            var_value = tk.DoubleVar(value=value)
            widget = ttk.Entry(frame, textvariable=var_value, width=15)
            widget.pack(side=tk.LEFT, padx=5)
        else:
            var_value = tk.StringVar(value=str(value))
            widget = ttk.Entry(frame, textvariable=var_value, width=30)
            widget.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.config_widgets[widget_key] = var_value

        comment_var = tk.StringVar(value=comment_text)
        comment_label = ttk.Label(
            frame,
            textvariable=comment_var,
            foreground="gray",
            wraplength=400,
            justify=tk.LEFT,
        )
        comment_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.comment_vars[widget_key] = comment_var

    def _set_path_value(self, container, path_str, value):
        keys = path_str.split(".") if path_str else []
        if not keys:
            return
        target = container
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    def collect_config_values(self):
        """Collect values from widgets back into config dict"""
        config_copy = copy.deepcopy(self.config)

        for widget_key, var in self.config_widgets.items():
            try:
                value = var.get()
            except Exception:
                value = var.get()
            self._set_path_value(config_copy, widget_key, value)

        for widget_key, var in self.comment_vars.items():
            self.comments[widget_key] = var.get().strip()

        self.config = config_copy


class ROITab(ttk.Frame):
    """Tab for managing Regions of Interest."""

    def __init__(self, parent, gui_manager):
        super().__init__(parent)
        self.gui_manager = gui_manager
        self.rois = {}
        self.current_roi = None
        self.preview_image = None
        self._preview_pil = None
        self._latest_screen_pil = None
        self._preview_roi_rect: Optional[Tuple[int, int, int, int]] = None
        self._preview_base_size: Tuple[int, int] = (0, 0)

        # Crop tool state
        self._crop_window = None
        self._crop_canvas = None
        self._crop_img_pil = None
        self._crop_img_tk = None
        self._crop_rect = None
        self._crop_sel = None
        self._crop_start = None
        self._crop_base_scale = 1.0
        self._crop_zoom = 1.0
        self._crop_zoom_var = None

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.list_label = ttk.Label(
            left_frame, text=self.gui_manager.t("roi_list"), font=("Arial", 10, "bold")
        )
        self.list_label.pack(pady=5)

        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.roi_listbox = tk.Listbox(list_frame, width=30)
        self.roi_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.roi_listbox.bind("<<ListboxSelect>>", self.on_roi_select)

        scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.roi_listbox.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.roi_listbox.configure(yscrollcommand=scrollbar.set)

        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            button_frame, text=self.gui_manager.t("add"), command=self.add_roi
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            button_frame, text=self.gui_manager.t("delete"), command=self.delete_roi
        ).pack(side=tk.LEFT, padx=2)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.details_frame = ttk.LabelFrame(
            right_frame, text=self.gui_manager.t("roi_details"), padding=10
        )
        self.details_frame.pack(fill=tk.X, pady=5)

        name_frame = ttk.Frame(self.details_frame)
        name_frame.pack(fill=tk.X, pady=2)
        self.name_label = ttk.Label(
            name_frame, text=self.gui_manager.t("roi_name") + ":", width=15
        )
        self.name_label.pack(side=tk.LEFT)
        self.name_var = tk.StringVar()
        ttk.Entry(name_frame, textvariable=self.name_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5
        )

        coords_frame = ttk.Frame(self.details_frame)
        coords_frame.pack(fill=tk.X, pady=2)
        self.coords_label = ttk.Label(
            coords_frame, text=self.gui_manager.t("roi_coordinates") + ":", width=15
        )
        self.coords_label.pack(side=tk.LEFT)

        coords_container = ttk.Frame(coords_frame)
        coords_container.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.coord_vars = []
        coord_names = ["x1", "y1", "x2", "y2"]
        for coord in coord_names:
            field = ttk.Frame(coords_container)
            field.pack(side=tk.LEFT, padx=2)
            ttk.Label(field, text=coord.upper()).pack()
            var = tk.DoubleVar(value=0.0)
            ttk.Entry(field, textvariable=var, width=8).pack()
            self.coord_vars.append(var)

        button_row = ttk.Frame(self.details_frame)
        button_row.pack(fill=tk.X, pady=5)
        self.capture_button = ttk.Button(
            button_row, text=self.gui_manager.t("capture_roi"), command=self.capture_roi
        )
        self.capture_button.pack(side=tk.LEFT, padx=2)
        self.save_button = ttk.Button(
            button_row,
            text=self.gui_manager.t("save"),
            command=lambda: self.save_current_roi(),
        )
        self.save_button.pack(side=tk.LEFT, padx=2)

        self.preview_frame = ttk.LabelFrame(
            right_frame, text=self.gui_manager.t("preview"), padding=10
        )
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.preview_canvas = tk.Canvas(
            self.preview_frame,
            width=400,
            height=300,
            bg="#333333",
            highlightthickness=0,
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas.bind("<Configure>", lambda _e: self._redraw_preview())

        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill=tk.X, pady=5)
        self.save_config_button = ttk.Button(
            action_frame,
            text=self.gui_manager.t("save_to_config"),
            command=self.save_rois_to_config,
        )
        self.save_config_button.pack(side=tk.RIGHT, padx=2)

    def load_rois(self, config_path=None):
        """Load ROI definitions from configuration."""
        path = config_path or self.gui_manager.active_config_path()
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            else:
                data = {}
            self.rois = data.get("rois", {}).copy()
            self.populate_list()
            if self.gui_manager:
                self.gui_manager.update_rois(self.rois)
                if (
                    hasattr(self.gui_manager, "templates_tab")
                    and self.gui_manager.templates_tab
                ):
                    self.gui_manager.templates_tab.update_roi_choices(
                        list(self.rois.keys())
                    )
        except Exception as e:
            self.gui_manager.log_message(f"Error loading ROIs: {e}", "ERROR")

    def populate_list(self, selected=None):
        """Populate ROI listbox with current entries."""
        self.roi_listbox.delete(0, tk.END)
        names = sorted(self.rois.keys())
        for name in names:
            self.roi_listbox.insert(tk.END, name)

        if selected and selected in names:
            index = names.index(selected)
            self.roi_listbox.selection_set(index)
            self.roi_listbox.see(index)
            self.on_roi_select(None)
        elif names:
            self.roi_listbox.selection_set(0)
            self.roi_listbox.see(0)
            self.on_roi_select(None)
        else:
            self.clear_form()

    def clear_form(self):
        """Reset ROI detail form."""
        self.current_roi = None
        self.name_var.set("")
        for var in self.coord_vars:
            var.set(0.0)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_text(
            200,
            150,
            text="",
            fill="white",
            font=("Arial", 12),
        )

    def on_roi_select(self, _event):
        """Handle ROI selection from list."""
        selection = self.roi_listbox.curselection()
        if not selection:
            return
        roi_name = self.roi_listbox.get(selection[0])
        self.current_roi = roi_name
        self.name_var.set(roi_name)
        coords = self.rois.get(roi_name, [0.0, 0.0, 1.0, 1.0])
        for var, value in zip(self.coord_vars, coords):
            var.set(float(value))
        self.update_preview()

    def add_roi(self):
        """Add a new ROI placeholder."""
        new_name = simpledialog.askstring("ROI", "Enter new ROI name:")
        if not new_name:
            return
        new_name = new_name.strip()
        if not new_name:
            return
        if new_name in self.rois:
            messagebox.showwarning("Warning", f"ROI '{new_name}' already exists.")
            return
        self.rois[new_name] = [0.0, 0.0, 1.0, 1.0]
        self.gui_manager.log_message(f"ROI added: {new_name}", "INFO")
        self.populate_list(selected=new_name)

    def delete_roi(self):
        """Delete the selected ROI."""
        selection = self.roi_listbox.curselection()
        if not selection:
            return
        roi_name = self.roi_listbox.get(selection[0])
        if not messagebox.askyesno("Confirm Delete", f"Delete ROI '{roi_name}'?"):
            return
        self.rois.pop(roi_name, None)
        if self.current_roi == roi_name:
            self.current_roi = None
        self.gui_manager.log_message(f"ROI deleted: {roi_name}", "INFO")
        self.populate_list()
        self.gui_manager.update_rois(self.rois)
        if (
            hasattr(self.gui_manager, "templates_tab")
            and self.gui_manager.templates_tab
        ):
            self.gui_manager.templates_tab.update_roi_choices(list(self.rois.keys()))

    def save_current_roi(self, silent=False):
        """Persist current ROI details into the collection."""
        name = self.name_var.get().strip()
        if not name:
            if not silent:
                messagebox.showwarning("Warning", "Please provide a ROI name.")
            return

        try:
            coords = [float(var.get()) for var in self.coord_vars]
        except (tk.TclError, ValueError):
            if not silent:
                messagebox.showerror("Error", "Invalid coordinate values.")
            return

        x1, y1, x2, y2 = coords
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        if x2 - x1 <= 0 or y2 - y1 <= 0:
            if not silent:
                messagebox.showerror(
                    "Error", "ROI must have a positive width and height."
                )
            return

        coords = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]

        # Handle renaming
        old_name = self.current_roi
        if old_name and old_name != name and name in self.rois:
            if not silent:
                messagebox.showerror("Error", f"ROI '{name}' already exists.")
            return

        if old_name and old_name != name:
            self.rois.pop(old_name, None)

        self.rois[name] = coords
        self.current_roi = name
        self.gui_manager.update_rois(self.rois)
        if (
            hasattr(self.gui_manager, "templates_tab")
            and self.gui_manager.templates_tab
        ):
            self.gui_manager.templates_tab.update_roi_choices(list(self.rois.keys()))

        self.populate_list(selected=name)
        self.update_preview()
        self.gui_manager.log_message(f"ROI saved: {name} -> {coords}", "INFO")

    def _get_current_coords(self):
        if self.current_roi and self.current_roi in self.rois:
            return self.rois[self.current_roi]
        try:
            return [float(var.get()) for var in self.coord_vars]
        except (tk.TclError, ValueError):
            return None

    def capture_roi(self):
        """Capture a screenshot and open crop tool to define ROI."""
        if not self.current_roi and not self.name_var.get().strip():
            messagebox.showwarning("Warning", "Select or name an ROI before capturing.")
            return

        try:
            img_bgr = screenshot_bgr()
            if img_bgr is None:
                self.gui_manager.log_message("No screenshot available", "ERROR")
                messagebox.showerror("Error", "No screenshot available.")
                return
            img_pil = self._bgr_to_pil(img_bgr)
            if img_pil is None:
                self.gui_manager.log_message("Error converting screenshot", "ERROR")
                messagebox.showerror("Error", "Unable to convert screenshot.")
                return
            self._latest_screen_pil = img_pil.copy()
            self.open_crop_tool(img_pil)
        except Exception as e:
            self.gui_manager.log_message(f"ROI capture failed: {e}", "ERROR")
            messagebox.showerror("Error", f"ROI capture failed: {e}")

    def _bgr_to_pil(self, img_bgr):
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        except Exception:
            return None

    def open_crop_tool(self, img_pil):
        """Open cropping window for ROI selection."""
        try:
            if self._crop_window and tk.Toplevel.winfo_exists(self._crop_window):
                self._crop_window.destroy()
        except Exception:
            pass

        self._crop_img_pil = img_pil
        self._crop_rect = None
        self._crop_sel = None
        self._crop_start = None
        self._crop_base_scale = 1.0
        self._crop_zoom = 1.0

        self._crop_window = tk.Toplevel(self)
        self._crop_window.title("Select ROI Area")
        self._crop_window.geometry("1000x700")
        self._crop_window.minsize(400, 300)

        self._crop_canvas = tk.Canvas(
            self._crop_window, bg="black", highlightthickness=0
        )
        self._crop_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        controls = ttk.Frame(self._crop_window)
        controls.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Button(
            controls, text=self.gui_manager.t("save"), command=self._save_crop
        ).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(controls, text="Cancel", command=self._crop_window.destroy).pack(
            side=tk.LEFT, padx=5, pady=5
        )

        self._crop_zoom_var = tk.DoubleVar(value=1.0)
        zoom_scale = ttk.Scale(
            controls,
            from_=0.5,
            to=4.0,
            orient=tk.HORIZONTAL,
            variable=self._crop_zoom_var,
            command=lambda _e=None: self._on_zoom_change(),
        )
        zoom_scale.pack(side=tk.RIGHT, padx=5, pady=5)
        ttk.Label(controls, text="Zoom").pack(side=tk.RIGHT)

        self._crop_canvas.bind("<ButtonPress-1>", self._on_crop_press)
        self._crop_canvas.bind("<B1-Motion>", self._on_crop_drag)
        self._crop_canvas.bind("<ButtonRelease-1>", self._on_crop_release)
        self._crop_canvas.bind("<Configure>", self._on_crop_canvas_resize)
        self._crop_window.bind("<Control-MouseWheel>", self._on_mousewheel_zoom)

        self._render_crop_image()

    def _on_crop_press(self, event):
        if not self._crop_img_pil:
            return
        scale = self._crop_effective_scale()
        ix = int(max(0, min(self._crop_img_pil.width, event.x / scale)))
        iy = int(max(0, min(self._crop_img_pil.height, event.y / scale)))
        self._crop_start = (ix, iy)
        self._crop_sel = (ix, iy, ix, iy)
        self._draw_crop_rect()

    def _on_crop_drag(self, event):
        if not self._crop_start or not self._crop_img_pil:
            return
        scale = self._crop_effective_scale()
        x0, y0 = self._crop_start
        x1 = int(max(0, min(self._crop_img_pil.width, event.x / scale)))
        y1 = int(max(0, min(self._crop_img_pil.height, event.y / scale)))
        self._crop_sel = (x0, y0, x1, y1)
        self._draw_crop_rect()

    def _on_crop_release(self, _event):
        self._draw_crop_rect()

    def _save_crop(self):
        if not (self._crop_sel and self._crop_img_pil):
            self.gui_manager.log_message("No crop selection to save.", "WARNING")
            return

        x0, y0, x1, y1 = self._crop_sel
        x0, x1 = sorted([int(x0), int(x1)])
        y0, y1 = sorted([int(y0), int(y1)])
        if x1 <= x0 or y1 <= y0:
            self.gui_manager.log_message("Invalid ROI selection.", "ERROR")
            messagebox.showerror("Error", "Invalid ROI selection.")
            return

        width, height = self._crop_img_pil.width, self._crop_img_pil.height
        coords = [
            round(max(0, min(width, x0)) / width, 4),
            round(max(0, min(height, y0)) / height, 4),
            round(max(0, min(width, x1)) / width, 4),
            round(max(0, min(height, y1)) / height, 4),
        ]

        for var, value in zip(self.coord_vars, coords):
            var.set(value)

        self.save_current_roi(silent=True)
        self.gui_manager.log_message(
            f"ROI selection captured: {self.current_roi}", "INFO"
        )
        if self._crop_window:
            self._crop_window.destroy()
        if self._latest_screen_pil:
            self._update_preview_from_image(self._latest_screen_pil)

    def _crop_effective_scale(self):
        return max(0.01, float(self._crop_base_scale) * float(self._crop_zoom))

    def _compute_canvas_scale(self):
        if not (self._crop_canvas and self._crop_img_pil):
            return
        cw = max(1, int(self._crop_canvas.winfo_width() or 1))
        ch = max(1, int(self._crop_canvas.winfo_height() or 1))
        iw = max(1, int(self._crop_img_pil.width))
        ih = max(1, int(self._crop_img_pil.height))
        self._crop_base_scale = min(cw / iw, ch / ih)

    def _render_crop_image(self):
        if not (self._crop_canvas and self._crop_img_pil):
            return
        self._compute_canvas_scale()
        scale = self._crop_effective_scale()
        iw, ih = self._crop_img_pil.width, self._crop_img_pil.height
        disp_w = max(1, int(iw * scale))
        disp_h = max(1, int(ih * scale))

        disp_img = self._crop_img_pil.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self._crop_img_tk = ImageTk.PhotoImage(disp_img)
        self._crop_canvas.delete("all")
        self._crop_canvas.create_image(0, 0, anchor=tk.NW, image=self._crop_img_tk)
        self._crop_canvas.config(scrollregion=(0, 0, disp_w, disp_h))
        self._draw_crop_rect()

    def _draw_crop_rect(self):
        if not (self._crop_sel and self._crop_canvas):
            if self._crop_rect:
                try:
                    self._crop_canvas.delete(self._crop_rect)
                except Exception:
                    pass
                self._crop_rect = None
            return

        x0, y0, x1, y1 = self._crop_sel
        scale = self._crop_effective_scale()
        cx0, cy0 = x0 * scale, y0 * scale
        cx1, cy1 = x1 * scale, y1 * scale

        if self._crop_rect:
            try:
                self._crop_canvas.coords(self._crop_rect, cx0, cy0, cx1, cy1)
            except Exception:
                self._crop_canvas.delete(self._crop_rect)
                self._crop_rect = None
        if not self._crop_rect:
            self._crop_rect = self._crop_canvas.create_rectangle(
                cx0, cy0, cx1, cy1, outline="red", width=2
            )

    def _on_crop_canvas_resize(self, _event):
        self._render_crop_image()

    def _on_zoom_change(self):
        try:
            self._crop_zoom = float(self._crop_zoom_var.get())
        except Exception:
            self._crop_zoom = 1.0
        self._render_crop_image()

    def _on_mousewheel_zoom(self, event):
        delta = 0.1 if event.delta > 0 else -0.1
        try:
            val = float(self._crop_zoom_var.get()) + delta
        except Exception:
            val = 1.0
        val = max(0.5, min(4.0, val))
        self._crop_zoom_var.set(val)
        self._on_zoom_change()

    def _update_preview_from_image(self, img_pil):
        coords = self._get_current_coords()
        self._preview_base_size = (img_pil.width, img_pil.height)
        self._preview_roi_rect = None
        if coords:
            x1, y1, x2, y2 = coords
            width, height = self._preview_base_size
            left = int(x1 * width)
            top = int(y1 * height)
            right = int(x2 * width)
            bottom = int(y2 * height)
            if right > left and bottom > top:
                self._preview_roi_rect = (left, top, right, bottom)
        self._preview_pil = img_pil.copy()
        self._render_preview_image()

    def _render_preview_image(self):
        if not self.preview_canvas or not self._preview_pil:
            return
        cw = max(1, int(self.preview_canvas.winfo_width() or 1))
        ch = max(1, int(self.preview_canvas.winfo_height() or 1))
        img = self._preview_pil.copy()
        orig_w, orig_h = img.size
        img.thumbnail((cw, ch), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(img)
        self.preview_canvas.delete("all")
        x = (cw - img.width) // 2
        y = (ch - img.height) // 2
        self.preview_canvas.create_image(x, y, anchor=tk.NW, image=self.preview_image)
        if self._preview_roi_rect and orig_w and orig_h:
            scale_x = img.width / orig_w
            scale_y = img.height / orig_h
            left, top, right, bottom = self._preview_roi_rect
            rect_left = x + int(left * scale_x)
            rect_top = y + int(top * scale_y)
            rect_right = x + int(right * scale_x)
            rect_bottom = y + int(bottom * scale_y)
            self.preview_canvas.create_rectangle(
                rect_left,
                rect_top,
                rect_right,
                rect_bottom,
                outline="red",
                width=2,
            )

    def _redraw_preview(self):
        if self._preview_pil:
            self._render_preview_image()

    def update_preview(self):
        """Refresh preview image for the current ROI."""
        coords = self._get_current_coords()
        if not coords:
            return
        if self._latest_screen_pil:
            self._update_preview_from_image(self._latest_screen_pil)
            return
        try:
            img_bgr = screenshot_bgr()
            if img_bgr is None:
                self.preview_canvas.delete("all")
                self.preview_canvas.create_text(
                    200,
                    150,
                    text="Preview unavailable",
                    fill="white",
                    font=("Arial", 12),
                )
                return
            img_pil = self._bgr_to_pil(img_bgr)
            if img_pil is None:
                return
            self._latest_screen_pil = img_pil
            self._update_preview_from_image(img_pil)
        except Exception as e:
            self.gui_manager.log_message(f"Preview update failed: {e}", "ERROR")

    def save_rois_to_config(self, config_path=None):
        """Persist ROI definitions into configuration file."""
        self.save_current_roi(silent=True)
        target_path = config_path or self.gui_manager.active_config_path()
        try:
            if os.path.exists(target_path):
                with open(target_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            else:
                data = {}
            data["rois"] = {name: coords for name, coords in self.rois.items()}
            with open(target_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            CFG["rois"] = data.get("rois", {}).copy()
            self.gui_manager.update_rois(self.rois)
            if (
                hasattr(self.gui_manager, "templates_tab")
                and self.gui_manager.templates_tab
            ):
                self.gui_manager.templates_tab.update_roi_choices(
                    list(self.rois.keys())
                )
            self.gui_manager.log_message(self.gui_manager.t("config_saved"), "INFO")
        except Exception as e:
            self.gui_manager.log_message(
                f"{self.gui_manager.t('error_saving')}: {e}", "ERROR"
            )

    def refresh_texts(self):
        """Refresh static texts for localization."""
        try:
            self.list_label.config(text=self.gui_manager.t("roi_list"))
            self.details_frame.config(text=self.gui_manager.t("roi_details"))
            self.name_label.config(text=self.gui_manager.t("roi_name") + ":")
            self.coords_label.config(text=self.gui_manager.t("roi_coordinates") + ":")
            self.capture_button.config(text=self.gui_manager.t("capture_roi"))
            self.save_button.config(text=self.gui_manager.t("save"))
            self.preview_frame.config(text=self.gui_manager.t("preview"))
            self.save_config_button.config(text=self.gui_manager.t("save_to_config"))
        except Exception:
            pass


class TemplatesTab(ttk.Frame):
    """Tab Template management"""

    MAX_TEMPLATE_IMAGES = 10

    def __init__(self, parent, gui_manager):
        super().__init__(parent)
        self.gui_manager = gui_manager
        self.templates = {}
        self.current_template = None
        self.preview_image = None
        self.current_paths: List[str] = []
        self._capture_mode = "auto"
        # --- Crop tool state (initialized to avoid attribute errors) ---
        self._crop_window = None
        self._crop_canvas = None
        self._crop_img_pil = None
        self._crop_img_tk = None
        self._crop_image_id = None
        self._crop_rect = None  # Canvas rectangle item id
        self._crop_sel = None  # Selection in image coords (x0, y0, x1, y1)
        self._crop_start = None  # Drag start in image coords
        self._crop_base_scale = 1.0  # Fit-to-window scale
        self._crop_zoom = 1.0  # User zoom multiplier
        self._crop_zoom_var = None  # Tk variable for zoom slider

        self.setup_ui()

    def setup_ui(self):
        # Left panel - template list
        left_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)

        self.templates_label = ttk.Label(
            left_frame, text=self.gui_manager.t("templates"), font=("Arial", 10, "bold")
        )
        self.templates_label.pack(pady=5)

        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.template_listbox = tk.Listbox(list_frame, width=30)
        self.template_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.template_listbox.bind("<<ListboxSelect>>", self.on_template_select)

        scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.template_listbox.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.template_listbox.configure(yscrollcommand=scrollbar.set)

        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=5)

        self.add_button = ttk.Button(
            button_frame, text=self.gui_manager.t("add"), command=self.add_template
        )
        self.add_button.pack(side=tk.LEFT, padx=2)

        self.delete_button = ttk.Button(
            button_frame,
            text=self.gui_manager.t("delete"),
            command=self.delete_template,
        )
        self.delete_button.pack(side=tk.LEFT, padx=2)

        # Right panel - template editor
        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Template details
        details_frame = ttk.LabelFrame(right_frame, text="Template Details", padding=10)
        details_frame.pack(fill=tk.X, pady=5)

        # Name
        name_frame = ttk.Frame(details_frame)
        name_frame.pack(fill=tk.X, pady=2)
        ttk.Label(
            name_frame, text=self.gui_manager.t("template_name") + ":", width=15
        ).pack(side=tk.LEFT)
        self.name_var = tk.StringVar()
        ttk.Entry(name_frame, textvariable=self.name_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5
        )

        # Template images
        self.images_frame = ttk.LabelFrame(
            details_frame, text=self.gui_manager.t("template_images"), padding=8
        )
        self.images_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        list_container = ttk.Frame(self.images_frame)
        list_container.pack(fill=tk.BOTH, expand=True)

        self.image_listbox = tk.Listbox(list_container, height=6)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_listbox.bind("<<ListboxSelect>>", self.on_image_select)

        image_scrollbar = ttk.Scrollbar(
            list_container, orient=tk.VERTICAL, command=self.image_listbox.yview
        )
        image_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.image_listbox.configure(yscrollcommand=image_scrollbar.set)

        controls_column = ttk.Frame(list_container)
        controls_column.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.add_image_button = ttk.Button(
            controls_column,
            text=self.gui_manager.t("add_image"),
            command=self.add_image_path_from_dialog,
        )
        self.add_image_button.pack(fill=tk.X, pady=1)

        self.add_capture_button = ttk.Button(
            controls_column,
            text=self.gui_manager.t("add_image_capture"),
            command=lambda: self.capture_and_crop(mode="append"),
        )
        self.add_capture_button.pack(fill=tk.X, pady=1)

        self.replace_image_button = ttk.Button(
            controls_column,
            text=self.gui_manager.t("replace_image"),
            command=self.replace_image_path,
        )
        self.replace_image_button.pack(fill=tk.X, pady=1)

        self.replace_capture_button = ttk.Button(
            controls_column,
            text=self.gui_manager.t("replace_image_capture"),
            command=lambda: self.capture_and_crop(mode="replace"),
        )
        self.replace_capture_button.pack(fill=tk.X, pady=1)

        self.remove_image_button = ttk.Button(
            controls_column,
            text=self.gui_manager.t("remove_image"),
            command=self.remove_image_path,
        )
        self.remove_image_button.pack(fill=tk.X, pady=1)

        ttk.Separator(controls_column, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        self.move_up_button = ttk.Button(
            controls_column,
            text=self.gui_manager.t("move_up"),
            command=lambda: self.move_image(-1),
        )
        self.move_up_button.pack(fill=tk.X, pady=1)

        self.move_down_button = ttk.Button(
            controls_column,
            text=self.gui_manager.t("move_down"),
            command=lambda: self.move_image(1),
        )
        self.move_down_button.pack(fill=tk.X, pady=1)

        # Selected path editor
        editor_frame = ttk.Frame(self.images_frame)
        editor_frame.pack(fill=tk.X, pady=(8, 0))
        self.selected_image_label = ttk.Label(
            editor_frame,
            text=self.gui_manager.t("selected_image") + ":",
            width=18,
        )
        self.selected_image_label.pack(side=tk.LEFT)
        self.path_var = tk.StringVar()
        self.path_entry = ttk.Entry(editor_frame, textvariable=self.path_var)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.apply_path_button = ttk.Button(
            editor_frame,
            text=self.gui_manager.t("apply_path"),
            command=self.apply_path_edit,
        )
        self.apply_path_button.pack(side=tk.LEFT)

        self.image_count_label = ttk.Label(
            self.images_frame,
            text="",
            anchor=tk.W,
        )
        self.image_count_label.pack(fill=tk.X, pady=(4, 0))
        self.refresh_image_listbox()

        # ROI as dropdown (from configured names)
        roi_frame = ttk.Frame(details_frame)
        roi_frame.pack(fill=tk.X, pady=2)
        ttk.Label(roi_frame, text=self.gui_manager.t("roi") + ":", width=15).pack(
            side=tk.LEFT
        )
        self.roi_var = tk.StringVar()
        roi_names = self.gui_manager.get_roi_names()
        self.roi_combo = ttk.Combobox(
            roi_frame, textvariable=self.roi_var, values=roi_names, state="readonly"
        )
        self.roi_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Threshold
        threshold_frame = ttk.Frame(details_frame)
        threshold_frame.pack(fill=tk.X, pady=2)
        ttk.Label(
            threshold_frame, text=self.gui_manager.t("threshold") + ":", width=15
        ).pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=0.85)
        ttk.Spinbox(
            threshold_frame,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.threshold_var,
            width=10,
        ).pack(side=tk.LEFT, padx=5)

        # Use only ROI
        self.use_only_roi_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            details_frame, text="Use Only ROI", variable=self.use_only_roi_var
        ).pack(fill=tk.X, pady=2)

        # Scales
        scales_frame = ttk.Frame(details_frame)
        scales_frame.pack(fill=tk.X, pady=2)
        ttk.Label(scales_frame, text="Scales:", width=15).pack(side=tk.LEFT)
        self.scales_var = tk.StringVar(value="[1.0]")
        ttk.Entry(scales_frame, textvariable=self.scales_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5
        )

        # Save button
        self.save_template_button = ttk.Button(
            details_frame,
            text=self.gui_manager.t("save_changes"),
            command=self.save_template,
        )
        self.save_template_button.pack(pady=10)

        # Preview frame
        self.preview_frame = ttk.LabelFrame(
            right_frame, text=self.gui_manager.t("preview"), padding=10
        )
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.preview_canvas = tk.Canvas(
            self.preview_frame, bg="gray", width=400, height=300
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        # Redraw preview when the canvas is resized
        self.preview_canvas.bind(
            "<Configure>", lambda e: self._redraw_preview_on_resize()
        )

        # Capture/Crop buttons
        capture_frame = ttk.Frame(self.preview_frame)
        capture_frame.pack(fill=tk.X, pady=5)
        self.capture_button = ttk.Button(
            capture_frame,
            text="Capture + Crop",
            command=self.capture_and_crop,
        )
        self.capture_button.pack(side=tk.LEFT, padx=5)

        actions_frame = ttk.Frame(right_frame)
        actions_frame.pack(fill=tk.X, pady=5)
        self.save_config_button = ttk.Button(
            actions_frame,
            text=self.gui_manager.t("save_to_config"),
            command=self.save_templates_to_config_file,
        )
        self.save_config_button.pack(side=tk.RIGHT, padx=5)

    def refresh_texts(self):
        """Update static labels and buttons for current language."""
        try:
            self.templates_label.config(text=self.gui_manager.t("templates"))
            self.add_button.config(text=self.gui_manager.t("add"))
            self.delete_button.config(text=self.gui_manager.t("delete"))
            self.save_template_button.config(text=self.gui_manager.t("save_changes"))
            self.preview_frame.config(text=self.gui_manager.t("preview"))
            self.save_config_button.config(text=self.gui_manager.t("save_to_config"))
            self.images_frame.config(text=self.gui_manager.t("template_images"))
            self.add_image_button.config(text=self.gui_manager.t("add_image"))
            self.add_capture_button.config(text=self.gui_manager.t("add_image_capture"))
            self.replace_image_button.config(text=self.gui_manager.t("replace_image"))
            self.replace_capture_button.config(
                text=self.gui_manager.t("replace_image_capture")
            )
            self.remove_image_button.config(text=self.gui_manager.t("remove_image"))
            self.move_up_button.config(text=self.gui_manager.t("move_up"))
            self.move_down_button.config(text=self.gui_manager.t("move_down"))
            self.selected_image_label.config(
                text=self.gui_manager.t("selected_image") + ":"
            )
            self.apply_path_button.config(text=self.gui_manager.t("apply_path"))
            self.update_image_count_label()
        except Exception:
            pass

    def on_template_select(self, event):
        """Handle template selection"""
        selection = self.template_listbox.curselection()
        if not selection:
            return

        template_name = self.template_listbox.get(selection[0])
        self.current_template = template_name

        if template_name in self.templates:
            template_data = self.templates[template_name]

            self.name_var.set(template_name)

            # Handle path (can be string or list)
            paths = template_data.get("path", [])
            if isinstance(paths, str):
                paths = [paths] if paths else []
            elif isinstance(paths, list):
                paths = [p for p in paths if p][: self.MAX_TEMPLATE_IMAGES]
            else:
                paths = []
            self.current_paths = paths
            self.refresh_image_listbox()
            if self.current_paths:
                self.image_listbox.selection_clear(0, tk.END)
                self.image_listbox.selection_set(0)
                self.image_listbox.activate(0)
                self.on_image_select()
            else:
                self.path_var.set("")
                self._preview_pil = None
                if getattr(self, "preview_canvas", None):
                    self.preview_canvas.delete("all")

            self.roi_var.set(template_data.get("roi", ""))
            self.threshold_var.set(template_data.get("threshold", 0.85))
            self.use_only_roi_var.set(template_data.get("use_only_roi", False))

            # Handle scales
            scales = template_data.get("scales", [1.0])
            self.scales_var.set(str(scales))

    def refresh_image_listbox(self):
        """Refresh the listbox that contains the template image paths."""
        if not hasattr(self, "image_listbox"):
            return
        self.image_listbox.delete(0, tk.END)
        for path in self.current_paths:
            self.image_listbox.insert(tk.END, path)
        self.update_image_count_label()
        self.update_image_buttons_state()

    def update_image_count_label(self):
        """Update label that shows how many images are configured."""
        if not hasattr(self, "image_count_label"):
            return
        try:
            text = self.gui_manager.t("template_images_count").format(
                count=len(self.current_paths), max=self.MAX_TEMPLATE_IMAGES
            )
        except Exception:
            text = f"{len(self.current_paths)}/{self.MAX_TEMPLATE_IMAGES}"
        self.image_count_label.config(text=text)

    def get_selected_image_index(self):
        """Return the currently selected image index or None."""
        if not hasattr(self, "image_listbox"):
            return None
        selection = self.image_listbox.curselection()
        if not selection:
            return None
        return selection[0]

    def update_image_buttons_state(self):
        """Enable or disable buttons based on selection and limits."""
        if not hasattr(self, "image_listbox"):
            return
        template_selected = self.current_template is not None
        selection = self.get_selected_image_index()
        has_selection = template_selected and selection is not None
        add_enabled = (
            tk.NORMAL
            if template_selected and len(self.current_paths) < self.MAX_TEMPLATE_IMAGES
            else tk.DISABLED
        )
        self.add_image_button.config(state=add_enabled)
        self.add_capture_button.config(state=add_enabled)
        self.replace_image_button.config(
            state=tk.NORMAL if has_selection else tk.DISABLED
        )
        self.replace_capture_button.config(
            state=tk.NORMAL if has_selection else tk.DISABLED
        )
        self.remove_image_button.config(
            state=tk.NORMAL if has_selection else tk.DISABLED
        )
        self.move_up_button.config(
            state=tk.NORMAL if has_selection and selection > 0 else tk.DISABLED
        )
        self.move_down_button.config(
            state=tk.NORMAL
            if has_selection and selection < len(self.current_paths) - 1
            else tk.DISABLED
        )
        if not template_selected:
            self.apply_path_button.config(state=tk.DISABLED)
        else:
            if (
                len(self.current_paths) >= self.MAX_TEMPLATE_IMAGES
                and selection is None
            ):
                self.apply_path_button.config(state=tk.DISABLED)
            else:
                self.apply_path_button.config(state=tk.NORMAL)
        if template_selected:
            self.path_entry.config(state=tk.NORMAL)
        else:
            self.path_entry.config(state=tk.DISABLED)
        if hasattr(self, "capture_button"):
            self.capture_button.config(
                state=tk.NORMAL if template_selected else tk.DISABLED
            )

    def on_image_select(self, _event=None):
        """Update entry and preview when a different image is selected."""
        index = self.get_selected_image_index()
        if index is None:
            self.path_var.set("")
            self._preview_pil = None
            if getattr(self, "preview_canvas", None):
                self.preview_canvas.delete("all")
            self.update_image_buttons_state()
            return
        path = self.current_paths[index]
        self.path_var.set(path)
        self.load_preview(path)
        self.update_image_buttons_state()

    def add_image_path_from_dialog(self):
        """Open a file dialog and append the selected image."""
        if self.current_template is None:
            return
        if len(self.current_paths) >= self.MAX_TEMPLATE_IMAGES:
            messagebox.showwarning(
                self.gui_manager.t("template_image_limit_title"),
                self.gui_manager.t("template_image_limit_body").format(
                    max=self.MAX_TEMPLATE_IMAGES
                ),
            )
            return
        filename = filedialog.askopenfilename(
            title="Select Template Image",
            initialdir=str(PROFILE_TEMPLATES_DIR),
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            stored = make_profile_relative(filename)
            self.current_paths.append(stored)
            self.refresh_image_listbox()
            new_index = len(self.current_paths) - 1
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(new_index)
            self.image_listbox.activate(new_index)
            self.on_image_select()
            self._sync_current_template_paths()

    def replace_image_path(self):
        """Replace the selected image path with a new file selection."""
        if self.current_template is None:
            return
        index = self.get_selected_image_index()
        if index is None:
            return
        filename = filedialog.askopenfilename(
            title="Select Template Image",
            initialdir=str(PROFILE_TEMPLATES_DIR),
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.current_paths[index] = make_profile_relative(filename)
            self.refresh_image_listbox()
            self.image_listbox.selection_set(index)
            self.image_listbox.activate(index)
            self.on_image_select()
            self._sync_current_template_paths()

    def remove_image_path(self):
        """Remove the currently selected image path."""
        if self.current_template is None:
            return
        index = self.get_selected_image_index()
        if index is None:
            return
        del self.current_paths[index]
        self.refresh_image_listbox()
        if self.current_paths:
            new_index = min(index, len(self.current_paths) - 1)
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(new_index)
            self.image_listbox.activate(new_index)
            self.on_image_select()
        else:
            self.path_var.set("")
            self._preview_pil = None
            if getattr(self, "preview_canvas", None):
                self.preview_canvas.delete("all")
            self.update_image_buttons_state()
        self._sync_current_template_paths()

    def move_image(self, direction: int):
        """Move the selected image up or down within the list."""
        if self.current_template is None:
            return
        index = self.get_selected_image_index()
        if index is None:
            return
        new_index = index + direction
        if new_index < 0 or new_index >= len(self.current_paths):
            return
        self.current_paths[index], self.current_paths[new_index] = (
            self.current_paths[new_index],
            self.current_paths[index],
        )
        self.refresh_image_listbox()
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(new_index)
        self.image_listbox.activate(new_index)
        self.on_image_select()
        self._sync_current_template_paths()

    def apply_path_edit(self):
        """Apply manual edits from the path entry."""
        if self.current_template is None:
            return
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning(
                self.gui_manager.t("template_image_invalid_title"),
                self.gui_manager.t("template_image_invalid_body"),
            )
            return
        index = self.get_selected_image_index()
        if index is not None:
            stored = make_profile_relative(path)
            self.current_paths[index] = stored
            target_index = index
        else:
            if len(self.current_paths) >= self.MAX_TEMPLATE_IMAGES:
                messagebox.showwarning(
                    self.gui_manager.t("template_image_limit_title"),
                    self.gui_manager.t("template_image_limit_body").format(
                        max=self.MAX_TEMPLATE_IMAGES
                    ),
                )
                return
            stored = make_profile_relative(path)
            self.current_paths.append(stored)
            target_index = len(self.current_paths) - 1
        self.refresh_image_listbox()
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(target_index)
        self.image_listbox.activate(target_index)
        self.on_image_select()
        self._sync_current_template_paths()

    def _sync_current_template_paths(self):
        """Keep the in-memory template dictionary in sync with the UI list."""
        if not self.current_template:
            return
        data = self.templates.get(self.current_template, {}).copy()
        data["path"] = self._format_paths_for_storage()
        self.templates[self.current_template] = data

    def _format_paths_for_storage(self):
        """Return the path field in the same structure as the config."""
        paths = [make_profile_relative(p) for p in self.current_paths if p]
        if not paths:
            return ""
        if len(paths) == 1:
            return paths[0]
        return paths.copy()

    def load_preview(self, image_path):
        """Load and display template preview"""
        try:
            # If preview canvas is not yet initialized, skip gracefully
            if getattr(self, "preview_canvas", None) is None:
                return
            canvas = self.preview_canvas
            cw = max(1, int(canvas.winfo_width() or 400))
            ch = max(1, int(canvas.winfo_height() or 300))
            if not image_path:
                canvas.delete("all")
                canvas.create_text(
                    cw // 2,
                    ch // 2,
                    text=self.gui_manager.t("template_no_image"),
                    fill="white",
                    font=("Arial", 12),
                )
                return
            resolved_path = resolve_template_path(image_path) or image_path
            if resolved_path and os.path.exists(resolved_path):
                image = Image.open(resolved_path)
                # Keep original around for responsive redraws
                self._preview_pil = image.copy()
                self._render_preview_to_canvas(self._preview_pil)
            else:
                canvas.delete("all")
                canvas.create_text(
                    cw // 2,
                    ch // 2,
                    text=self.gui_manager.t("template_image_not_found"),
                    fill="white",
                    font=("Arial", 12),
                )
        except Exception as e:
            self.gui_manager.log_message(f"Error loading preview: {e}", "ERROR")

    def _bgr_to_pil(self, img_bgr):
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        except Exception:
            return None

    def _render_preview_to_canvas(self, img_pil):
        try:
            if getattr(self, "preview_canvas", None) is None:
                return
            cw = max(1, int(self.preview_canvas.winfo_width() or 400))
            ch = max(1, int(self.preview_canvas.winfo_height() or 300))
            img = img_pil.copy()
            img.thumbnail((cw, ch), Image.Resampling.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(img)
            self.preview_canvas.delete("all")
            x = (cw - img.width) // 2
            y = (ch - img.height) // 2
            self.preview_canvas.create_image(
                x, y, anchor=tk.NW, image=self.preview_image
            )
        except Exception:
            pass

    def capture_and_crop(self, mode="auto"):
        if self.current_template is None:
            messagebox.showwarning(
                self.gui_manager.t("template_select_first_title"),
                self.gui_manager.t("template_select_first_body"),
            )
            return
        if mode == "replace" and self.get_selected_image_index() is None:
            messagebox.showwarning(
                self.gui_manager.t("template_capture_no_selection_title"),
                self.gui_manager.t("template_capture_no_selection_body"),
            )
            return
        if mode == "append" and len(self.current_paths) >= self.MAX_TEMPLATE_IMAGES:
            messagebox.showwarning(
                self.gui_manager.t("template_image_limit_title"),
                self.gui_manager.t("template_image_limit_body").format(
                    max=self.MAX_TEMPLATE_IMAGES
                ),
            )
            return
        self._capture_mode = mode
        try:
            img_bgr = screenshot_bgr()
            if img_bgr is None:
                self.gui_manager.log_message("No screenshot available", "ERROR")
                self._capture_mode = "auto"
                return
            img_pil = self._bgr_to_pil(img_bgr)
            if img_pil is None:
                self.gui_manager.log_message("Error converting screenshot", "ERROR")
                self._capture_mode = "auto"
                return
            self.open_crop_tool(img_pil)
        except Exception as e:
            self.gui_manager.log_message(f"Capture failed: {e}", "ERROR")
            self._capture_mode = "auto"

    def _redraw_preview_on_resize(self):
        try:
            if hasattr(self, "_preview_pil") and self._preview_pil is not None:
                self._render_preview_to_canvas(self._preview_pil)
        except Exception:
            pass

    def open_crop_tool(self, img_pil):
        try:
            if self._crop_window and tk.Toplevel.winfo_exists(self._crop_window):
                self._crop_window.destroy()
        except Exception:
            pass
        # Reset crop state
        self._crop_img_pil = img_pil
        self._crop_rect = None
        self._crop_sel = None
        self._crop_start = None
        self._crop_base_scale = 1.0
        self._crop_zoom = 1.0

        # Build window
        self._crop_window = tk.Toplevel(self)
        self._crop_window.title("Crop Template")
        self._crop_window.geometry("1000x700")
        self._crop_window.minsize(400, 300)

        # Canvas for image
        self._crop_canvas = tk.Canvas(
            self._crop_window, bg="black", highlightthickness=0
        )
        self._crop_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Controls
        controls = ttk.Frame(self._crop_window)
        controls.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(controls, text="Save Crop", command=self._save_crop).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(controls, text="Cancel", command=self._crop_window.destroy).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Label(controls, text="Zoom:").pack(side=tk.LEFT, padx=(15, 5))
        self._crop_zoom_var = tk.DoubleVar(value=1.0)
        zoom_scale = ttk.Scale(
            controls,
            from_=0.5,
            to=4.0,
            orient=tk.HORIZONTAL,
            variable=self._crop_zoom_var,
            command=lambda _e=None: self._on_zoom_change(),
        )
        zoom_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(
            controls, text="-", width=3, command=lambda: self._nudge_zoom(-0.1)
        ).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(
            controls, text="+", width=3, command=lambda: self._nudge_zoom(0.1)
        ).pack(side=tk.LEFT, padx=(2, 10))

        # Bind interactions
        self._crop_canvas.bind("<ButtonPress-1>", self._on_crop_press)
        self._crop_canvas.bind("<B1-Motion>", self._on_crop_drag)
        self._crop_canvas.bind("<ButtonRelease-1>", self._on_crop_release)
        self._crop_canvas.bind("<Configure>", self._on_crop_canvas_resize)
        # Ctrl+MouseWheel for zoom
        self._crop_window.bind("<Control-MouseWheel>", self._on_mousewheel_zoom)

        # Initial render
        self._update_base_scale_to_fit()
        self._render_crop_image()

    def _on_crop_press(self, event):
        # Convert event coords to image space and start selection
        scale = self._crop_effective_scale()
        ix = int(max(0, min(self._crop_img_pil.width, event.x / scale)))
        iy = int(max(0, min(self._crop_img_pil.height, event.y / scale)))
        self._crop_start = (ix, iy)
        self._crop_sel = (ix, iy, ix, iy)
        self._redraw_selection()

    def _on_crop_drag(self, event):
        if not self._crop_start:
            return
        scale = self._crop_effective_scale()
        x0, y0 = self._crop_start
        x1 = int(max(0, min(self._crop_img_pil.width, event.x / scale)))
        y1 = int(max(0, min(self._crop_img_pil.height, event.y / scale)))
        self._crop_sel = (x0, y0, x1, y1)
        self._redraw_selection()

    def _on_crop_release(self, event):
        pass

    def _save_crop(self):
        try:
            if not (self._crop_sel and self._crop_img_pil):
                return
            x0, y0, x1, y1 = self._crop_sel
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            ox0 = max(0, min(self._crop_img_pil.width, int(x0)))
            oy0 = max(0, min(self._crop_img_pil.height, int(y0)))
            ox1 = max(0, min(self._crop_img_pil.width, int(x1)))
            oy1 = max(0, min(self._crop_img_pil.height, int(y1)))
            if ox1 <= ox0 or oy1 <= oy0:
                self.gui_manager.log_message("Invalid crop area", "ERROR")
                return
            cropped = self._crop_img_pil.crop((ox0, oy0, ox1, oy1))
            default_dir = str(PROFILE_TEMPLATES_DIR)
            os.makedirs(default_dir, exist_ok=True)
            default_name = (self.name_var.get() or "template") + ".png"
            save_path = filedialog.asksaveasfilename(
                initialdir=default_dir,
                initialfile=default_name,
                title="Save Cropped Template",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("All files", "*.*")],
            )
            if not save_path:
                return
            try:
                cropped.save(save_path)
            except Exception as e:
                self.gui_manager.log_message(f"Error saving crop: {e}", "ERROR")
                return
            mode = getattr(self, "_capture_mode", "auto")
            selection = self.get_selected_image_index()
            target_index = None

            if mode == "replace":
                if selection is None:
                    messagebox.showwarning(
                        self.gui_manager.t("template_capture_no_selection_title"),
                        self.gui_manager.t("template_capture_no_selection_body"),
                    )
                    return
                stored_path = make_profile_relative(save_path)
                self.current_paths[selection] = stored_path
                target_index = selection
            elif mode == "append":
                if len(self.current_paths) >= self.MAX_TEMPLATE_IMAGES:
                    messagebox.showwarning(
                        self.gui_manager.t("template_image_limit_title"),
                        self.gui_manager.t("template_image_limit_body").format(
                            max=self.MAX_TEMPLATE_IMAGES
                        ),
                    )
                    return
                stored_path = make_profile_relative(save_path)
                self.current_paths.append(stored_path)
                target_index = len(self.current_paths) - 1
            else:  # auto
                if selection is not None:
                    stored_path = make_profile_relative(save_path)
                    self.current_paths[selection] = stored_path
                    target_index = selection
                else:
                    if len(self.current_paths) >= self.MAX_TEMPLATE_IMAGES:
                        messagebox.showwarning(
                            self.gui_manager.t("template_image_limit_title"),
                            self.gui_manager.t("template_image_limit_body").format(
                                max=self.MAX_TEMPLATE_IMAGES
                            ),
                        )
                        return
                    stored_path = make_profile_relative(save_path)
                    self.current_paths.append(stored_path)
                    target_index = len(self.current_paths) - 1
            self.refresh_image_listbox()
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(target_index)
            self.image_listbox.activate(target_index)
            self.on_image_select()
            self._sync_current_template_paths()
            self.gui_manager.log_message(f"Saved cropped template: {save_path}", "INFO")
        finally:
            self._capture_mode = "auto"
            try:
                if self._crop_window:
                    self._crop_window.destroy()
            except Exception:
                pass

    # --- Crop window helpers ---
    def _crop_effective_scale(self):
        try:
            return max(0.01, float(self._crop_base_scale) * float(self._crop_zoom))
        except Exception:
            return 1.0

    def _update_base_scale_to_fit(self):
        if not self._crop_canvas or not self._crop_img_pil:
            return
        cw = max(1, int(self._crop_canvas.winfo_width() or 1))
        ch = max(1, int(self._crop_canvas.winfo_height() or 1))
        iw = max(1, int(self._crop_img_pil.width))
        ih = max(1, int(self._crop_img_pil.height))
        self._crop_base_scale = min(cw / iw, ch / ih)

    def _render_crop_image(self):
        if not self._crop_canvas or not self._crop_img_pil:
            return
        scale = self._crop_effective_scale()
        iw, ih = self._crop_img_pil.width, self._crop_img_pil.height
        disp_w = max(1, int(iw * scale))
        disp_h = max(1, int(ih * scale))
        disp_img = self._crop_img_pil.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self._crop_img_tk = ImageTk.PhotoImage(disp_img)

        # Clear and redraw image
        self._crop_canvas.delete("all")
        self._crop_image_id = self._crop_canvas.create_image(
            0, 0, anchor=tk.NW, image=self._crop_img_tk
        )
        # Resize scrollregion to image size
        self._crop_canvas.config(scrollregion=(0, 0, disp_w, disp_h))

        # Redraw selection overlay if exists
        self._redraw_selection()

    def _redraw_selection(self):
        # Draw selection rectangle from image-space selection
        if not self._crop_sel or not self._crop_canvas:
            if self._crop_rect:
                try:
                    self._crop_canvas.delete(self._crop_rect)
                except Exception:
                    pass
                self._crop_rect = None
            return
        x0, y0, x1, y1 = self._crop_sel
        scale = self._crop_effective_scale()
        cx0, cy0 = x0 * scale, y0 * scale
        cx1, cy1 = x1 * scale, y1 * scale
        if self._crop_rect:
            try:
                self._crop_canvas.coords(self._crop_rect, cx0, cy0, cx1, cy1)
            except Exception:
                self._crop_rect = None
        if not self._crop_rect:
            self._crop_rect = self._crop_canvas.create_rectangle(
                cx0, cy0, cx1, cy1, outline="red", width=2
            )

    def _on_crop_canvas_resize(self, event):
        # Recompute base scale to fit canvas and re-render
        self._update_base_scale_to_fit()
        self._render_crop_image()

    def _on_zoom_change(self):
        try:
            self._crop_zoom = float(self._crop_zoom_var.get())
        except Exception:
            self._crop_zoom = 1.0
        self._render_crop_image()

    def _nudge_zoom(self, delta):
        try:
            val = float(self._crop_zoom_var.get()) + delta
        except Exception:
            val = 1.0
        val = min(4.0, max(0.5, val))
        self._crop_zoom_var.set(val)
        self._on_zoom_change()

    def _on_mousewheel_zoom(self, event):
        # Zoom in/out with Ctrl+Wheel
        delta = 0.1 if getattr(event, "delta", 0) > 0 else -0.1
        self._nudge_zoom(delta)

    def browse_template(self):
        """Backward-compatible alias for adding a template image."""
        self.add_image_path_from_dialog()

    def add_template(self):
        """Add new template"""
        new_name = simpledialog.askstring("New Template", "Enter template name:")
        if new_name:
            if new_name not in self.templates:
                self.templates[new_name] = {
                    "path": [],
                    "roi": "",
                    "use_only_roi": False,
                    "threshold": 0.85,
                    "scales": [1.0],
                }
                self.template_listbox.insert(tk.END, new_name)
                self.template_listbox.selection_clear(0, tk.END)
                last_index = self.template_listbox.size() - 1
                if last_index >= 0:
                    self.template_listbox.selection_set(last_index)
                    self.template_listbox.see(last_index)
                    self.on_template_select(None)
                self.gui_manager.log_message(f"Added template: {new_name}", "INFO")
            else:
                messagebox.showwarning("Warning", "Template already exists!")

    def delete_template(self):
        """Delete selected template"""
        selection = self.template_listbox.curselection()
        if not selection:
            return

        template_name = self.template_listbox.get(selection[0])

        if messagebox.askyesno("Confirm Delete", f"Delete template '{template_name}'?"):
            del self.templates[template_name]
            self.template_listbox.delete(selection[0])
            self.clear_form()
            self.gui_manager.log_message(f"Deleted template: {template_name}", "INFO")

    def save_template(self):
        """Save current template"""
        if not self.current_template:
            return

        try:
            # Parse scales
            scales_str = self.scales_var.get()
            if scales_str:
                try:
                    parsed = ast.literal_eval(scales_str)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(f"Invalid scales definition: {exc}") from exc
                if isinstance(parsed, (int, float)):
                    scales = [float(parsed)]
                elif isinstance(parsed, (list, tuple)):
                    try:
                        scales = [float(item) for item in parsed]
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "Scales list must contain only numeric values."
                        ) from exc
                else:
                    raise ValueError("Scales must be a number or list of numbers.")
            else:
                scales = [1.0]

            path_value = self._format_paths_for_storage()

            template_data = {
                "path": path_value,
                "roi": self.roi_var.get(),
                "use_only_roi": self.use_only_roi_var.get(),
                "threshold": self.threshold_var.get(),
                "scales": scales,
            }

            self.templates[self.current_template] = template_data
            # Keep in-memory paths consistent in case formatting changed
            if isinstance(path_value, list):
                self.current_paths = path_value.copy()
            elif isinstance(path_value, str) and path_value:
                self.current_paths = [path_value]
            else:
                self.current_paths = []
            self.refresh_image_listbox()

            # Persist changes to the active profile configuration
            self.save_templates_to_config_file(show_message=False)

            self.gui_manager.log_message(
                f"Saved template: {self.current_template}", "INFO"
            )
            messagebox.showinfo(
                "Success",
                f"Template '{self.current_template}' saved to profile configuration.",
            )
        except Exception as e:
            self.gui_manager.log_message(f"Error saving template: {e}", "ERROR")
            messagebox.showerror("Error", f"Error saving template: {e}")

    def clear_form(self):
        """Clear form fields"""
        self.name_var.set("")
        self.path_var.set("")
        self.roi_var.set("")
        self.threshold_var.set(0.85)
        self.use_only_roi_var.set(False)
        self.scales_var.set("[1.0]")
        self.current_template = None
        self.current_paths = []
        self.refresh_image_listbox()
        self.preview_canvas.delete("all")
        self._preview_pil = None

    def load_templates(self, templates_dict):
        """Load templates from configuration"""
        self.templates = templates_dict.copy()
        self.template_listbox.delete(0, tk.END)

        for name in sorted(self.templates.keys()):
            self.template_listbox.insert(tk.END, name)
        self.update_roi_choices(self.gui_manager.get_roi_names())

    def save_templates_to_config(self):
        """Return templates dict for saving to config"""
        return self.templates.copy()

    def save_templates_to_config_file(self, config_path=None, show_message=True):
        """Write current templates to configuration file."""
        global TEMPLATES_CONFIG
        target_path = config_path or self.gui_manager.active_config_path()
        try:
            if os.path.exists(target_path):
                with open(target_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            else:
                data = {}
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            data["templates"] = self.save_templates_to_config()
            with open(target_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            CFG["templates"] = data.get("templates", {}).copy()
            TEMPLATES_CONFIG = CFG.get("templates", {}).copy()
            try:
                load_templates()
            except Exception:
                pass
            self.gui_manager.log_message(self.gui_manager.t("config_saved"), "INFO")
            if show_message:
                messagebox.showinfo(
                    "Success", f"Templates saved to {os.path.basename(target_path)}"
                )
        except Exception as e:
            self.gui_manager.log_message(
                f"{self.gui_manager.t('error_saving')}: {e}", "ERROR"
            )
            messagebox.showerror("Error", f"Error saving templates: {e}")

    def update_roi_choices(self, roi_names: List[str]):
        """Update ROI selection options."""
        roi_names = sorted(roi_names)
        self.roi_combo.configure(values=roi_names)
        if self.roi_var.get() not in roi_names:
            self.roi_var.set("")


class GUIManager:
    """Main GUI manager class"""

    def __init__(
        self,
        config_path: str,
        profile_name: Optional[str],
        available_profiles: List[str],
    ):
        self.config_path = config_path
        self.profile_name = profile_name
        self.available_profiles = sorted(available_profiles)
        self.root = None
        self.language = str(CFG.get("language", "en")).lower()
        self.available_languages = list(AVAILABLE_LANG_CODES)
        if self.language not in self.available_languages:
            self.language = "en"
        self.window_title = CFG.get("bot_title", translate_key("en", "title"))
        self.command_queue = queue.Queue()
        self.update_queue = queue.Queue()
        self.running = False
        # Log tail state for integrating file logs into GUI
        self.log_tail_path = None
        self.log_tail_pos = 0

        # Tab references
        self.monitor_tab = None
        self.dashboard_tab = None
        self.log_tab = None
        self.settings_tab = None
        self.roi_tab = None
        self.templates_tab = None
        self.tasks_tab = None
        self.current_rois = CFG.get("rois", {}).copy()
        self.language_var = None

    def t(self, key):
        """Translate key to current language"""
        if key == "title":
            return self.window_title
        return translate_key(self.language, key)

    def _build_menubar(self):
        """Build the menubar with language menu (rebuild-safe)."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        lang_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.t("language"), menu=lang_menu)
        if self.language_var is None:
            self.language_var = tk.StringVar(self.root, value=self.language)
        else:
            self.language_var.set(self.language)
        for code in self.available_languages:
            display = get_language_display_name(code)
            lang_menu.add_radiobutton(
                label=display,
                variable=self.language_var,
                value=code,
                command=lambda c=code: self.change_language(c),
            )
        self.menubar = menubar
        self.lang_menu = lang_menu

    def create_gui(self):
        """Create the main GUI"""
        self.root = tk.Tk()
        self.root.title(self.window_title)
        self.root.geometry("1600x920")
        # Track root window handle for foreground detection
        try:
            self.root_hwnd = self.root.winfo_id()
        except Exception:
            self.root_hwnd = None

        # Menu bar
        self._build_menubar()

        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        self.monitor_tab = MonitorTab(self.notebook, self)
        self.notebook.add(self.monitor_tab, text=self.t("monitor"))

        self.dashboard_tab = DashboardTab(self.notebook, self)
        self.notebook.add(self.dashboard_tab, text=self.t("dashboard"))

        self.roi_tab = ROITab(self.notebook, self)
        self.notebook.add(self.roi_tab, text=self.t("rois_tab"))

        self.templates_tab = TemplatesTab(self.notebook, self)
        self.notebook.add(self.templates_tab, text=self.t("templates"))

        self.tasks_tab = TasksTab(self.notebook, self)
        self.notebook.add(self.tasks_tab, text=self.t("tasks_tab"))

        self.settings_tab = SettingsTab(self.notebook, self)
        self.notebook.add(self.settings_tab, text=self.t("settings"))

        self.log_tab = LogTab(self.notebook, self)
        self.notebook.add(self.log_tab, text=self.t("log"))

        # Load initial configuration
        self.settings_tab.load_config(self.config_path)
        self.roi_tab.load_rois(self.config_path)

        # Load templates from config
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
                if config and "templates" in config:
                    self.templates_tab.load_templates(config["templates"])
        except Exception as e:
            self.log_message(f"Error loading templates: {e}", "ERROR")

        # Status bar
        status_text = "Ready"
        if self.profile_name:
            status_text = f"Profile: {self.profile_name} | {status_text}"
        self.status_bar = ttk.Label(
            self.root, text=status_text, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Start update checker
        self.check_updates()

        self.log_message("GUI initialized successfully", "INFO")
        # Start tailing the rotating logfile into the Log tab
        try:
            logs_dir = "logs"
            log_path = os.path.join(logs_dir, "bot.log")
            self.start_log_tail(log_path)
        except Exception:
            pass

    def change_language(self, lang):
        """Change GUI language"""
        lang = (lang or "en").lower()
        if lang not in self.available_languages:
            lang = "en"
        if lang == self.language:
            return
        self.language = lang
        if self.language_var is not None:
            try:
                self.language_var.set(lang)
            except Exception:
                pass
        CFG["language"] = lang
        if lang in TRANSLATION_COMMENTS:
            for key, text in TRANSLATION_COMMENTS[lang].items():
                CFG_COMMENTS.setdefault(key, text)
        persist_language_to_config(self.config_path, lang)
        self.log_message(f"Language changed to: {lang}", "INFO")
        # Update visible texts immediately
        self.refresh_language_texts()

    def refresh_language_texts(self):
        """Refresh static texts for current language without restart."""
        if not self.root:
            return
        # Window title
        self.root.title(self.window_title)
        # Menubar
        self._build_menubar()
        # Tab titles
        try:
            self.notebook.tab(self.monitor_tab, text=self.t("monitor"))
            self.notebook.tab(self.dashboard_tab, text=self.t("dashboard"))
            self.notebook.tab(self.log_tab, text=self.t("log"))
            self.notebook.tab(self.settings_tab, text=self.t("settings"))
            self.notebook.tab(self.roi_tab, text=self.t("rois_tab"))
            self.notebook.tab(self.templates_tab, text=self.t("templates"))
            self.notebook.tab(self.tasks_tab, text=self.t("tasks_tab"))
        except Exception:
            pass
        # Per-tab widgets
        for tab in [
            self.monitor_tab,
            self.dashboard_tab,
            self.log_tab,
            self.settings_tab,
            self.roi_tab,
            self.templates_tab,
            self.tasks_tab,
        ]:
            if hasattr(tab, "refresh_texts"):
                try:
                    tab.refresh_texts()
                except Exception:
                    pass

    def update_rois(self, rois: Dict[str, List[float]]):
        """Update cached ROI definitions and propagate to dependent tabs."""
        try:
            self.current_rois = dict(rois)
        except Exception:
            self.current_rois = {}
        if hasattr(self, "templates_tab") and self.templates_tab:
            try:
                self.templates_tab.update_roi_choices(self.get_roi_names())
            except Exception:
                pass

    def get_roi_names(self) -> List[str]:
        """Return sorted ROI names."""
        return sorted(self.current_rois.keys())

    def get_template_names(self) -> List[str]:
        """Return sorted template names from current configuration."""
        try:
            if hasattr(self, "templates_tab") and self.templates_tab:
                templates = getattr(self.templates_tab, "templates", {})
                if isinstance(templates, dict):
                    return sorted(templates.keys())
        except Exception:
            pass
        try:
            templates = CFG.get("templates", {})
            if isinstance(templates, dict):
                return sorted(templates.keys())
        except Exception:
            pass
        return []

    def active_config_path(self) -> str:
        """Return the configuration file path for the active profile."""
        if self.profile_name:
            return str((PROFILES_DIR / self.profile_name / "config.yaml").resolve())
        return self.config_path

    def apply_profile_selection(self, profile_name: str) -> Tuple[bool, str]:
        """Persist a new profile selection and inform the caller."""
        if not profile_name:
            return False, self.t("profile_invalid_selection")
        if profile_name == self.profile_name:
            return True, self.t("profile_no_change")
        profiles = list_profiles()
        if profile_name not in profiles:
            return False, self.t("profile_not_found").format(name=profile_name)
        try:
            write_active_profile(profile_name)
            self.profile_name = profile_name
            self.available_profiles = sorted(profiles)
            self.log_message(
                f"Profile set to {profile_name}. Restart required to apply.", "INFO"
            )
            return True, self.t("profile_restart_hint")
        except Exception as exc:
            self.log_message(f"Failed to set profile: {exc}", "ERROR")
            return False, f"{self.t('profile_apply_error')}: {exc}"

    def set_paused(self, paused: bool):
        """Synchronize paused state into the GUI."""
        self.update_queue.put(("paused", {"value": bool(paused)}))

    # --- Log tailing ---
    def start_log_tail(self, path: str):
        """Begin tailing the logfile at `path` and stream into the Log tab."""
        try:
            self.log_tail_path = path
            # Start at end of file to avoid dumping entire history
            if os.path.exists(path):
                with open(path, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    self.log_tail_pos = f.tell()
            # Schedule periodic tailing
            if self.root:
                self.root.after(300, self._tail_log_once)
        except Exception:
            pass

    def _tail_log_once(self):
        """Read any new content from the logfile and append to GUI log."""
        try:
            if not self.running or not self.root or not self.log_tail_path:
                return
            path = self.log_tail_path
            if not os.path.exists(path):
                # If file temporarily missing (rotation), retry later
                self.root.after(500, self._tail_log_once)
                return
            # Open and read new content
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                size = os.path.getsize(path)
                # Handle rotation/truncate
                if self.log_tail_pos > size:
                    self.log_tail_pos = 0
                f.seek(self.log_tail_pos)
                new_data = f.read()
                self.log_tail_pos = f.tell()
            if new_data:
                for line in new_data.splitlines():
                    line = line.rstrip("\n")
                    # Parse "YYYY-mm-dd ... - LEVEL - message"
                    m = re.search(r" - (DEBUG|INFO|WARNING|ERROR) - ", line)
                    lvl = m.group(1) if m else "INFO"
                    # Strip timestamp and level prefix for GUI message body
                    msg = line
                    if m:
                        # Split once after level separator
                        parts = line.split(" - ", 2)
                        if len(parts) == 3:
                            msg = parts[2]
                    self.log_tab.add_log(msg, lvl)
        except Exception:
            # Swallow errors; continue tailing
            pass
        finally:
            if self.root and self.running:
                self.root.after(500, self._tail_log_once)

    def run(self):
        """Run the GUI in a separate thread"""
        self.running = True
        gui_thread = threading.Thread(target=self._run_gui, daemon=True)
        gui_thread.start()
        return gui_thread

    def _run_gui(self):
        """Internal method to run GUI main loop"""
        self.create_gui()
        self.root.mainloop()
        self.running = False

    def check_updates(self):
        """Check for updates from the queue"""
        if not self.running or not self.root:
            return

        try:
            while not self.update_queue.empty():
                update_type, data = self.update_queue.get_nowait()

                if update_type == "log":
                    self.log_tab.add_log(
                        data.get("message", ""), data.get("level", "INFO")
                    )
                elif update_type == "stats":
                    self.dashboard_tab.update_stats(data)
                elif update_type == "action":
                    self.dashboard_tab.add_action(
                        data.get("action", ""), data.get("result", "")
                    )
                elif update_type == "image":
                    self.monitor_tab.update_image(data)
                elif update_type == "status":
                    self.status_bar.configure(text=data.get("text", "Ready"))
                elif update_type == "paused":
                    try:
                        self.monitor_tab.set_paused(bool(data.get("value", False)))
                        # Update status bar text accordingly
                        status_text = f"{self.t('status')}: " + (
                            self.t("paused") if data.get("value") else self.t("running")
                        )
                        self.status_bar.configure(text=status_text)
                    except Exception:
                        pass

        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error processing update: {e}")

        # Schedule next check
        if self.root:
            self.root.after(100, self.check_updates)

    def send_command(self, command, data=None):
        """Send command from GUI to bot"""
        self.command_queue.put((command, data))
        self.log_message(f"Command sent: {command}", "DEBUG")

    def log_message(self, message, level="INFO"):
        """Add log message"""
        self.update_queue.put(("log", {"message": message, "level": level}))
        # Also forward GUI logs to the global logfile
        try:
            lvl = str(level).upper()
            if lvl == "DEBUG":
                LOGGER.debug(message)
            elif lvl == "WARNING":
                LOGGER.warning(message)
            elif lvl == "ERROR":
                LOGGER.error(message)
            else:
                LOGGER.info(message)
        except Exception:
            pass

    def update_stats(self, stats):
        """Update statistics"""
        self.update_queue.put(("stats", stats))

    def add_action(self, action, result):
        """Add action to dashboard"""
        self.update_queue.put(("action", {"action": action, "result": result}))

    def update_image(self, image_data):
        """Update monitor image"""
        self.update_queue.put(("image", image_data))

    def update_status(self, status_text):
        """Update status bar"""
        self.update_queue.put(("status", {"text": status_text}))

    def get_command(self):
        """Get command from queue (non-blocking)"""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Stop the GUI"""
        self.running = False
        if self.root:
            self.root.quit()


# --- Monitor GUI mit OpenCV ---

# Global reference to the Tk GUI manager (used by the monitor tab)
gui = None

# Global controller for the OpenCV/Tk hybrid UI
gui_controller = None


# --- GUI controller & heatmap management ---
class GUIController:
    """
    Manage the OpenCV window, interactive elements, and heatmap overlays.
    """

    def __init__(
        self,
        history_size,
        screen_width,
        screen_height,
        initial_width=None,
        initial_height=None,
        backend="tk"
    ):
        self.LIST_PANEL_WIDTH = 250
        self.LIST_ITEM_HEIGHT = 60
        bot_title = CFG.get("bot_title", translate_key("en", "title"))
        self.window_name = f"{bot_title} Overlay"
        # Selected rendering backend: "opencv" uses cv2.imshow window, "tk" renders into the Tk Monitor tab
        self.backend = backend

        self.game_screen_width = screen_width
        self.game_screen_height = screen_height

        self.initial_height = (
            initial_height if initial_height else self.game_screen_height
        )

        calculated_width = (
            int(
                self.initial_height * (self.game_screen_width / self.game_screen_height)
            )
            + self.LIST_PANEL_WIDTH
        )
        self.initial_width = initial_width if initial_width else calculated_width

        self.detection_history = deque(maxlen=history_size)
        self.history_view_index = -1
        self.step_mode_active = False
        self.list_scroll_offset = 0

        # Target size used by Tk backend; updated by MonitorTab on resize
        self._target_width = self.initial_width
        self._target_height = self.initial_height

        # Button definitions: (rectangle, label, color)
        self.buttons = {
            "prev": {"text": "<", "rect": (10, 10, 30, 30), "color": (220, 220, 220)},
            "next": {"text": ">", "rect": (50, 10, 30, 30), "color": (220, 220, 220)},
        }

        # When embedded in Tk we do not create native OpenCV windows
        if self.backend != "opencv":
            self.canvas = None
            # Monkey-patch OpenCV window helpers to keep the Tk pipeline in control
            try:
                cv2.namedWindow = lambda *args, **kwargs: None
                cv2.resizeWindow = lambda *args, **kwargs: None
                cv2.setMouseCallback = lambda *args, **kwargs: None
                cv2.getWindowImageRect = lambda *args, **kwargs: (
                    0,
                    0,
                    int(self._target_width or 0),
                    int(self._target_height or 0),
                )
            except Exception:
                pass
            # Skip further OpenCV window set-up
            return

    # --- Public controls used by the Tk Monitor tab ---
    def set_target_size(self, width: int, height: int):
        """Set target render size when using the Tk backend."""
        try:
            if width and height:
                self._target_width = max(100, int(width))
                self._target_height = max(100, int(height))
        except Exception:
            pass

    def process_click_from_tk(self, x: int, y: int):
        """Process a left-click coming from the Tk canvas coordinates."""
        try:
            window_w = int(self._target_width or 0)
            window_h = int(self._target_height or 0)
            if window_w <= 0 or window_h <= 0:
                return
            game_panel_w = max(0, window_w - self.LIST_PANEL_WIDTH)
            self._handle_click(x, y, game_panel_w, window_h)
        except Exception:
            pass

    def process_wheel_from_tk(self, delta: int, x: int, y: int):
        """Process a mouse wheel event from Tk (delta>0 up, delta<0 down)."""
        try:
            window_w = int(self._target_width or 0)
            window_h = int(self._target_height or 0)
            if window_w <= 0 or window_h <= 0:
                return
            game_panel_w = max(0, window_w - self.LIST_PANEL_WIDTH)
            if x > game_panel_w:
                max_scroll = max(
                    0, len(self.detection_history) - (window_h // self.LIST_ITEM_HEIGHT)
                )
                if delta > 0:
                    self.list_scroll_offset = max(0, self.list_scroll_offset - 1)
                else:
                    self.list_scroll_offset = min(
                        max_scroll, self.list_scroll_offset + 1
                    )
        except Exception:
            pass

        # Create a resizable OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Apply initial window size
        cv2.resizeWindow(self.window_name, self.initial_width, self.initial_height)
        # Register mouse callback for clicks and scrolling
        cv2.setMouseCallback(self.window_name, self._handle_mouse)
        # Canvas is built dynamically in the render method
        self.canvas = None

    def _handle_mouse(self, event, x, y, flags, param):
        """Handle mouse clicks and scrolls, respecting the current window size."""
        # Calculate the current game panel width based on window size
        try:
            if self.backend == "opencv":
                _, _, window_w, current_window_h = cv2.getWindowImageRect(
                    self.window_name
                )
            else:
                window_w, current_window_h = (
                    self._target_width or 0,
                    self._target_height or 0,
                )
                if window_w <= 0 or current_window_h <= 0:
                    return
        except cv2.error:
            return  # Window was closed

        game_panel_w = window_w - self.LIST_PANEL_WIDTH
        if game_panel_w < 100:
            return  # Prevent interactions when the game panel is too small

        # Click events
        if event == cv2.EVENT_LBUTTONDOWN:
            self._handle_click(x, y, game_panel_w, current_window_h)

        # Scroll events limited to the sidebar
        if event == cv2.EVENT_MOUSEWHEEL and x > game_panel_w:
            max_scroll = max(
                0,
                len(self.detection_history)
                - (current_window_h // self.LIST_ITEM_HEIGHT),
            )
            if flags > 0:  # Scroll up
                self.list_scroll_offset = max(0, self.list_scroll_offset - 1)
            else:  # Scroll down
                self.list_scroll_offset = min(max_scroll, self.list_scroll_offset + 1)

    def _handle_click(self, x, y, game_panel_w, window_h):
        global is_paused, bot_is_running

        # Top buttons (rendered on the left panel)
        if self._is_point_in_rect(x, y, self.buttons["prev"]["rect"]):
            # Pause the bot to lock the selection
            if not is_paused:
                _apply_pause_state(True)
                # Also push immediate GUI update (defensive: ensure GUI sees the change)
                try:
                    if GUI_ENABLED and gui:
                        gui.set_paused(True)
                except Exception:
                    pass
                LOGGER.warning("Bot paused via GUI Interaction.")
            self.step_mode_active = True
            self.history_view_index = max(0, self.history_view_index - 1)
        elif self._is_point_in_rect(x, y, self.buttons["next"]["rect"]):
            if not is_paused:
                _apply_pause_state(True)
                try:
                    if GUI_ENABLED and gui:
                        gui.set_paused(True)
                except Exception:
                    pass
                LOGGER.warning("Bot paused via GUI Interaction.")
            self.step_mode_active = True
            self.history_view_index = min(
                len(self.detection_history) - 1, self.history_view_index + 1
            )

        # Handle clicks on the history sidebar
        if x > game_panel_w:
            clicked_item_on_screen = y // self.LIST_ITEM_HEIGHT
            reversed_index = self.list_scroll_offset + clicked_item_on_screen

            history_len = len(self.detection_history)
            if 0 <= reversed_index < history_len:
                # Convert the reversed index back to the original detection_history index
                actual_index = history_len - 1 - reversed_index

                self.history_view_index = actual_index
                self.step_mode_active = True
                # Pause the bot to lock the selection
                if not is_paused:
                    _apply_pause_state(True)
                    try:
                        if GUI_ENABLED and gui:
                            gui.set_paused(True)
                    except Exception:
                        pass
                    LOGGER.warning("Bot paused via history selection.")

                LOGGER.info(
                    f"History selection activated: index {self.history_view_index}"
                )

    def _is_point_in_rect(self, x, y, rect):
        """Return True if point (x, y) lies inside the rectangle (x1, y1, w, h)."""
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def add_history(self, image, detections):
        """Append a new frame and its detections to the history (optionally filtered)."""
        has_success = any(d["found"] for d in detections if not d.get("is_roi"))
        if GUI_HISTORY_ONLY_SUCCESSFUL and not has_success:
            return  # Skip storing unsuccessful frames when configured

        # Create thumbnails for successful detections
        thumbnails = []
        for det in detections:
            if det.get("is_roi"):
                continue  # Skip thumbnails for ROIs
            thumb_data = {"key": det["key"], "score": det["score"], "thumb": None}
            if det["found"]:
                # Thumbnail-Erstellung bei Erfolg
                r = det["rect"]
                thumb = image[r[1] : r[3], r[0] : r[2]]
                if thumb.size > 0:
                    max_h, max_w = (
                        self.LIST_ITEM_HEIGHT - 10,
                        self.LIST_PANEL_WIDTH - 10,
                    )
                    h, w = thumb.shape[:2]
                    scale = min(max_w / w, max_h / h) if w > 0 and h > 0 else 0
                    if scale > 0:
                        new_w, new_h = int(w * scale), int(h * scale)
                        thumb_data["thumb"] = cv2.resize(
                            thumb, (new_w, new_h), interpolation=cv2.INTER_AREA
                        )
            thumbnails.append(thumb_data)

        self.detection_history.append((image.copy(), detections, thumbnails))

    def render(self, current_image, current_detections):
        """Draw the GUI overlay and adapt dynamically to window size."""
        try:
            _, _, window_w, window_h = cv2.getWindowImageRect(self.window_name)
            if window_w <= 0 or window_h <= 0:
                return  # Window minimised or invalid size
        except cv2.error:
            return  # Window was closed

        # Create or resize the canvas
        if (
            self.canvas is None
            or self.canvas.shape[0] != window_h
            or self.canvas.shape[1] != window_w
        ):
            self.canvas = np.zeros((window_h, window_w, 3), dtype=np.uint8)
        else:
            self.canvas.fill(0)  # Clear the canvas for this frame

        # Select the image to display (live or from history)
        img_source, dets_to_draw, _ = (
            self.detection_history[self.history_view_index]
            if self.step_mode_active
            and 0 <= self.history_view_index < len(self.detection_history)
            else (current_image, current_detections, [])
        )

        # Fallback for empty frames
        if img_source is None or img_source.size == 0:
            img_source = np.zeros(
                (self.game_screen_height, self.game_screen_width, 3), dtype=np.uint8
            )

        # Compute game panel width based on aspect ratio
        game_panel_w_calc = window_h * self.game_screen_width // self.game_screen_height
        game_panel_w = min(window_w - self.LIST_PANEL_WIDTH, game_panel_w_calc)

        # Scale the captured frame to the panel size
        display_img = cv2.resize(
            img_source, (game_panel_w, window_h), interpolation=cv2.INTER_AREA
        )

        # Determine the actual size of the scaled frame
        actual_h, actual_w, _ = display_img.shape

        # Overlay for drawing heatmap and buttons
        overlay = display_img.copy()

        # Compute scale factors for rectangles
        scale_x = actual_w / self.game_screen_width
        scale_y = actual_h / self.game_screen_height

        # Erkennungen zeichnen
        for det in dets_to_draw:
            rx1, ry1, rx2, ry2 = det["rect"]
            rect_scaled = (
                int(rx1 * scale_x),
                int(ry1 * scale_y),
                int(rx2 * scale_x),
                int(ry2 * scale_y),
            )

            # Differentiate ROI markers from detections
            if det.get("is_roi"):
                if GUI_SHOW_ROI:
                    color = (0, 255, 255)  # Yellow for ROI
                    cv2.rectangle(
                        overlay,
                        (rect_scaled[0], rect_scaled[1]),
                        (rect_scaled[2], rect_scaled[3]),
                        color,
                        1,
                    )
                    cv2.putText(
                        overlay,
                        det["key"],
                        (rect_scaled[0], rect_scaled[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )
            else:
                color = (0, 255, 0) if det["found"] else (0, 0, 255)
                cv2.rectangle(
                    overlay,
                    (rect_scaled[0], rect_scaled[1]),
                    (rect_scaled[2], rect_scaled[3]),
                    color,
                    2,
                )

                # Show scaling factor when a match was successful
                scale_info = (
                    f" @ {det.get('scale', 1.0):.2f}x"
                    if det.get("scale", 1.0) != 1.0
                    else ""
                )
                label = f"{det['key']}: {det['score']:.2f}{scale_info}"
                cv2.putText(
                    overlay,
                    label,
                    (rect_scaled[0], rect_scaled[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # Blend overlay with base image
        cv2.addWeighted(
            overlay, gui_heatmap_alpha, display_img, 1 - gui_heatmap_alpha, 0, display_img
        )

        # Draw buttons
        for btn in self.buttons.values():
            x, y, w, h = btn["rect"]
            cv2.rectangle(display_img, (x, y), (x + w, y + h), btn["color"], -1)
            cv2.putText(
                display_img,
                btn["text"],
                (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        self.canvas[0:actual_h, 0:actual_w] = display_img

        # Draw right sidebar
        sidebar_w = window_w - actual_w
        if sidebar_w > 0:
            list_panel = self.canvas[:, actual_w:]
            list_panel.fill(40)
            self.LIST_PANEL_WIDTH = sidebar_w

            reversed_history = list(reversed(self.detection_history))
            items_to_show = reversed_history[self.list_scroll_offset :]

            for i, (img_hist, dets_hist, thumbnails) in enumerate(items_to_show):
                item_y = i * self.LIST_ITEM_HEIGHT
                if item_y + self.LIST_ITEM_HEIGHT > window_h:
                    break

                actual_index = len(reversed_history) - 1 - (i + self.list_scroll_offset)
                is_selected = (
                    self.step_mode_active and actual_index == self.history_view_index
                )

                # Find the first real detection to display
                first_real_det = next(
                    (t for t in thumbnails if not t.get("is_roi")), None
                )
                if not first_real_det:
                    # Fallback if the frame contains ROIs only
                    key, score, thumb_img = "ROI-Check", 0.0, None
                else:
                    key = first_real_det["key"]
                    score = first_real_det["score"]
                    thumb_img = first_real_det["thumb"]

                bg_color = (0, 90, 130) if is_selected else (40, 40, 40)
                banner = np.full(
                    (self.LIST_ITEM_HEIGHT, sidebar_w, 3), bg_color, dtype=np.uint8
                )

                if thumb_img is not None:
                    th, tw, _ = thumb_img.shape

                    y_off, x_off = (self.LIST_ITEM_HEIGHT - th) // 2, 5
                    if x_off + tw <= banner.shape[1]:
                        banner[y_off : y_off + th, x_off : x_off + tw] = thumb_img

                    # Provide a subtle background for text
                    text_bg_overlay = banner.copy()

                    # Define text positions
                    text1_pos = (x_off + 5, y_off + 12)
                    text2_pos = (x_off + 5, y_off + 27)
                    text1 = f"#{actual_index}: {key}"
                    text2 = f"Score: {score:.2f}"
                    score_color = (0, 255, 0) if score >= THRESH else (0, 0, 255)

                    # Draw background rectangles
                    cv2.rectangle(
                        text_bg_overlay,
                        (text1_pos[0] - 2, text1_pos[1] - 10),
                        (text1_pos[0] + 100, text1_pos[1] + 5),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.rectangle(
                        text_bg_overlay,
                        (text2_pos[0] - 2, text2_pos[1] - 10),
                        (text2_pos[0] + 100, text2_pos[1] + 5),
                        (0, 0, 0),
                        -1,
                    )

                    # Apply overlay with transparency
                    alpha = 0.5
                    cv2.addWeighted(
                        text_bg_overlay, alpha, banner, 1 - alpha, 0, banner
                    )

                    # Render text over the background
                    cv2.putText(
                        banner,
                        text1,
                        text1_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        banner,
                        text2,
                        text2_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        score_color,
                        1,
                    )

                    list_panel[item_y : item_y + self.LIST_ITEM_HEIGHT, 0:sidebar_w] = (
                        banner
                    )
                else:
                    # Wenn kein Thumbnail, nur Text auf den Hintergrund zeichnen
                    list_panel[item_y : item_y + self.LIST_ITEM_HEIGHT, 0:sidebar_w] = (
                        banner
                    )
                    cv2.putText(
                        list_panel,
                        f"#{actual_index}: {key}",
                        (10, item_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (200, 200, 200),
                        1,
                    )
                    cv2.putText(
                        list_panel,
                        f"Score: {score:.2f}",
                        (10, item_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1,
                    )

                cv2.line(
                    list_panel,
                    (0, item_y + self.LIST_ITEM_HEIGHT - 1),
                    (sidebar_w, item_y + self.LIST_ITEM_HEIGHT - 1),
                    (60, 60, 60),
                    1,
                )

        if getattr(self, "backend", "tk") == "opencv":
            cv2.imshow(self.window_name, self.canvas)
            cv2.waitKey(1)
        else:
            # Push frame to Tk Monitor tab as PIL.Image
            try:
                rgb = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb)
                if gui is not None and getattr(gui, "running", False):
                    gui.update_image(img_pil)
            except Exception:
                pass

def ensure_critical_task_available(task_id: str) -> bool:
    if TASK_MANAGER and TASK_MANAGER.has_task(task_id):
        if task_id in MISSING_CRITICAL_TASKS:
            MISSING_CRITICAL_TASKS.discard(task_id)
        return True
    if not TASK_MANAGER:
        return False
    if task_id not in MISSING_CRITICAL_TASKS:
        LOGGER.error(
            "Critical task '%s' is missing from the active configuration. Restore it to maintain runtime stability.",
            task_id,
        )
        MISSING_CRITICAL_TASKS.add(task_id)
    return False


def application_has_focus() -> bool:
    if os.name != "nt":
        return True
    try:
        fg_hwnd = ctypes.windll.user32.GetForegroundWindow()
    except Exception:
        return True
    if not fg_hwnd:
        return True
    if GUI_ENABLED and gui and getattr(gui, "root_hwnd", None):
        if fg_hwnd == getattr(gui, "root_hwnd", None):
            return True
    try:
        console_hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if console_hwnd and fg_hwnd == console_hwnd:
            return True
    except Exception:
        pass
    try:
        pshell_hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if pshell_hwnd and fg_hwnd == pshell_hwnd:
            return True
    except Exception:
        pass
    try:
        buffer_len = 512
        class_buf = ctypes.create_unicode_buffer(buffer_len)
        if ctypes.windll.user32.GetClassNameW(fg_hwnd, class_buf, buffer_len):
            class_name = class_buf.value.lower()
            if "powershell" in class_name or "consolewindowclass" in class_name:
                return True
        title_buf = ctypes.create_unicode_buffer(buffer_len)
        if ctypes.windll.user32.GetWindowTextW(fg_hwnd, title_buf, buffer_len):
            title = title_buf.value.lower()
            if "powershell" in title or "pwsh" in title:
                return True
    except Exception:
        pass
    if CFG.get("control_mode") == "windows":
        title = CFG.get("windows_title")
        if title and "gw" in globals():
            try:
                active_window = gw.getActiveWindow()
                if active_window and title.lower() in (active_window.title or "").lower():
                    return True
            except Exception:
                pass
    return False


# --- Pause synchronization helper ---
def _apply_pause_state(paused: bool) -> bool:
    """
    Update the global pause flag and synchronize UI/monitor state.

    Returns the previous pause value so callers can detect transitions.
    """
    global is_paused, gui, gui_controller

    previous = is_paused
    is_paused = bool(paused)

    if GUI_ENABLED and gui:
        try:
            gui.set_paused(is_paused)
        except Exception:
            pass

    if not is_paused and gui_controller:
        gui_controller.step_mode_active = False
        gui_controller.history_view_index = -1
        # Force an immediate live render so the Monitor tab stops showing the
        # previously selected history frame when resuming. Also push the frame
        # directly into the Tk Monitor tab (gui.update_image) so the GUI shows
        # the live screen even when the controller uses the OpenCV backend.
        try:
            if GUI_ENABLED and gui_controller:
                try:
                    screen = screenshot_bgr()
                    # Use current detections if available, fallback to empty list
                    gui_controller.render(screen, detections_this_frame)
                except Exception:
                    try:
                        screen = screenshot_bgr()
                        gui_controller.render(screen, [])
                    except Exception:
                        screen = None

                # Also update the Tk Monitor tab directly, if GUI is running
                try:
                    if gui is not None and getattr(gui, "running", False) and screen is not None:
                        rgb = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(rgb)
                        gui.update_image(img_pil)
                except Exception:
                    pass
        except Exception:
            pass

    return previous


# --- Pause toggle (keyboard & GUI) ---
def toggle_pause(e=None):
    """
    Toggle the paused state. Can be triggered via keyboard or GUI.
    `e` is the keyboard library event object.
    """
    global is_paused, gui_controller
    # Ignore key events that are not the 'p' hotkey
    if e is None or e.name == "p":
        new_state = not is_paused
        _apply_pause_state(new_state)

        if new_state:
            LOGGER.warning("Bot paused. Press 'P' or use the GUI button to resume.")
            if gui_controller and len(gui_controller.detection_history) > 0:
                # Keep the current index in step mode; otherwise jump to the latest entry
                if not gui_controller.step_mode_active:
                    gui_controller.history_view_index = (
                        len(gui_controller.detection_history) - 1
                    )
        else:
            LOGGER.info("Bot resumed.")


# --- Statistics & Dashboard ---
class Stats:
    """Track per-task statistics for the automation engine."""

    def __init__(self, max_history: int = 10):
        self.start_ts = time.time()
        self.attempts: Counter[str] = Counter()
        self.successes: Counter[str] = Counter()
        self.history: deque = deque(maxlen=max_history)
        self.counter_defs: Dict[str, Dict[str, Any]] = {}
        self.counter_order: List[str] = []
        self.dynamic_counters: Dict[str, float] = {}
        self.counter_last_update: Dict[str, float] = {}
        self.task_counter_map: Dict[str, List[str]] = defaultdict(list)
        self.series_defs: Dict[str, Dict[str, Any]] = {}
        self.series_order: List[str] = []
        self.series_history: Dict[str, deque] = {}
        self.task_series_map: Dict[str, List[str]] = defaultdict(list)
        self.last_cleanup_ts: float = 0.0

    @property
    def total_successes(self) -> int:
        return sum(self.successes.values())

    @property
    def total_attempts(self) -> int:
        return sum(self.attempts.values())

    @property
    def total_actions(self) -> int:
        return sum(self.successes.values()) + sum(self.attempts.values())

    def configure_tasks(self, tasks: List[TaskDefinition]) -> None:
        """Register metric metadata declared on task definitions."""
        old_counters = dict(self.dynamic_counters)
        old_series = {
            key: list(history) for key, history in self.series_history.items()
        }
        old_counter_ts = dict(self.counter_last_update)

        self.counter_defs.clear()
        self.counter_order.clear()
        self.dynamic_counters.clear()
        self.counter_last_update.clear()
        self.task_counter_map.clear()

        self.series_defs.clear()
        self.series_order.clear()
        self.series_history.clear()
        self.task_series_map.clear()

        for task in tasks:
            stats_meta = getattr(task, "stats", {}) or {}
            self._register_counter_defs(
                task,
                stats_meta.get("counters", {}),
                old_counters,
                old_counter_ts,
            )
            self._register_series_defs(task, stats_meta.get("series", {}), old_series)

    def _register_counter_defs(
        self,
        task: TaskDefinition,
        raw_counters: Any,
        old_values: Dict[str, float],
        old_ts: Dict[str, float],
    ) -> None:
        for entry in self._iterate_metric_entries(raw_counters):
            base_name = entry.get("name") or entry.get("key") or entry.get("id")
            if not base_name:
                continue
            metric_key = str(entry.get("key") or f"{task.id}.{base_name}")
            label = entry.get("label") or base_name.replace("_", " ").title()
            source = entry.get("source") or f"metrics.{base_name}"
            accumulate = (entry.get("accumulate") or "sum").lower()
            unit = entry.get("unit")
            self.counter_defs[metric_key] = {
                "label": label,
                "source": source,
                "accumulate": accumulate,
                "unit": unit,
                "task_id": task.id,
            }
            self.counter_order.append(metric_key)
            self.task_counter_map[task.id].append(metric_key)
            if accumulate == "sum":
                self.dynamic_counters[metric_key] = float(old_values.get(metric_key, 0.0))
            else:
                self.dynamic_counters[metric_key] = old_values.get(metric_key)
            if metric_key in old_ts:
                self.counter_last_update[metric_key] = old_ts[metric_key]

    def _register_series_defs(
        self,
        task: TaskDefinition,
        raw_series: Any,
        old_values: Dict[str, List[Any]],
    ) -> None:
        for entry in self._iterate_metric_entries(raw_series):
            base_name = entry.get("name") or entry.get("key") or entry.get("id")
            if not base_name:
                continue
            metric_key = str(entry.get("key") or f"{task.id}.{base_name}")
            label = entry.get("label") or base_name.replace("_", " ").title()
            source = entry.get("source") or f"metrics.{base_name}"
            max_points = int(entry.get("max_points") or 50)
            unit = entry.get("unit")
            self.series_defs[metric_key] = {
                "label": label,
                "source": source,
                "max_points": max_points,
                "unit": unit,
                "task_id": task.id,
            }
            self.series_order.append(metric_key)
            self.task_series_map[task.id].append(metric_key)
            history = deque(old_values.get(metric_key, []), maxlen=max(1, max_points))
            if not history and STATS_DB:
                try:
                    rows = STATS_DB.execute(
                        "SELECT ts, value FROM series_events WHERE metric_key = ? "
                        "ORDER BY ts DESC LIMIT ?",
                        (metric_key, max_points),
                    ).fetchall()
                    for ts, value in reversed(rows):
                        history.append((ts, float(value)))
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "Failed to load persisted series '%s': %s", metric_key, exc
                    )
            self.series_history[metric_key] = history

    def _iterate_metric_entries(self, raw: Any) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        if isinstance(raw, dict):
            for key, value in raw.items():
                if isinstance(value, dict):
                    entry = dict(value)
                else:
                    entry = {"label": str(value)}
                entry.setdefault("name", key)
                entries.append(entry)
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    entry = dict(item)
                    if "name" not in entry:
                        entry["name"] = entry.get("key") or entry.get("id")
                    entries.append(entry)
        return entries

    def record_task(self, task: TaskDefinition, result: TaskResult):
        name = task.name
        if result.executed:
            self.attempts[name] += 1
            if result.success is True:
                self.successes[name] += 1
            self._record_dynamic_metrics(task, result)

        self.history.appendleft(
            {
                "time": time.strftime("%H:%M:%S"),
                "action": name,
                "ok": result.success,
                "detail": result.detail or "",
            }
        )

    def _record_dynamic_metrics(self, task: TaskDefinition, result: TaskResult) -> None:
        for metric_key in self.task_counter_map.get(task.id, []):
            definition = self.counter_defs.get(metric_key)
            if not definition:
                continue
            value = self._resolve_metric_source(result, definition["source"])
            if value is None:
                continue
            try:
                if definition["accumulate"] == "sum":
                    numeric = float(value)
                    self.dynamic_counters[metric_key] += numeric
                    self.counter_last_update[metric_key] = time.time()
                elif definition["accumulate"] == "set":
                    self.dynamic_counters[metric_key] = value
                    self.counter_last_update[metric_key] = time.time()
            except (TypeError, ValueError):
                continue

        for metric_key in self.task_series_map.get(task.id, []):
            definition = self.series_defs.get(metric_key)
            if not definition:
                continue
            value = self._resolve_metric_source(result, definition["source"])
            if value is None:
                continue
            history = self.series_history.get(metric_key)
            if not history:
                history = deque(maxlen=definition.get("max_points", 50))
                self.series_history[metric_key] = history
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric == 0:
                continue
            now_ts = time.time()
            history.append((now_ts, numeric))
            if STATS_DB:
                try:
                    STATS_DB.execute(
                        "INSERT INTO series_events(metric_key, ts, value) VALUES (?, ?, ?)",
                        (metric_key, now_ts, numeric),
                    )
                    STATS_DB.commit()
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "Failed to store metric '%s': %s", metric_key, exc
                    )
                if now_ts - self.last_cleanup_ts > 3600.0:
                    prune_old_series_data(now_ts - 30 * 86400)
                    self.last_cleanup_ts = now_ts

    def _resolve_metric_source(self, result: TaskResult, source: str) -> Any:
        if not source:
            return None
        source = str(source)
        data: Any
        base_is_context = False
        if source.startswith("result."):
            data = {
                "executed": result.executed,
                "success": result.success,
                "detail": result.detail,
            }
            path = source[len("result.") :]
        elif source.startswith("context."):
            data = result.context.data
            path = source[len("context.") :]
            base_is_context = True
        else:
            data = result.context.data
            path = source
            base_is_context = True
        parts = [p for p in path.split(".") if p]
        if not parts:
            return data
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                if base_is_context and isinstance(result.context.data, dict):
                    direct_key = ".".join(parts)
                    if direct_key in result.context.data:
                        return result.context.data.get(direct_key)
                    direct_key = "".join(parts)
                    if direct_key in result.context.data:
                        return result.context.data.get(direct_key)
                return None
        return current

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_successes": self.total_successes,
            "total_attempts": self.total_attempts,
            "uptime_secs": int(time.time() - self.start_ts),
            "history": list(self.history),
            "success_by_task": dict(self.successes),
            "attempts_by_task": dict(self.attempts),
            "dynamic_counters": self._dynamic_counter_snapshot(),
            "series": self._series_snapshot(),
        }

    def _dynamic_counter_snapshot(self) -> List[Dict[str, Any]]:
        snapshot: List[Dict[str, Any]] = []
        for key in self.counter_order:
            definition = self.counter_defs.get(key)
            if not definition:
                continue
            value = self.dynamic_counters.get(key)
            snapshot.append(
                {
                    "key": key,
                    "label": definition.get("label", key),
                    "value": value,
                    "unit": definition.get("unit"),
                    "last_ts": self.counter_last_update.get(key),
                }
            )
        return snapshot

    def _series_snapshot(self) -> List[Dict[str, Any]]:
        snapshot: List[Dict[str, Any]] = []
        for key in self.series_order:
            definition = self.series_defs.get(key)
            if not definition:
                continue
            history = self.series_history.get(key, deque())
            snapshot.append(
                {
                    "key": key,
                    "label": definition.get("label", key),
                    "points": [
                        {"ts": ts, "value": value} for ts, value in history
                    ],
                    "unit": definition.get("unit"),
                    "max_points": definition.get("max_points"),
                }
            )
        return snapshot

    def dynamic_counters_for_display(self) -> List[Dict[str, Any]]:
        """Return dynamic counters for console dashboard rendering."""
        return self._dynamic_counter_snapshot()


# --- ROI utilities (Region of Interest) ---
def roi_to_rect(img_shape, roi_norm):
    """Convert normalized ROI coordinates into absolute pixel coordinates."""
    H, W = img_shape[:2]
    x1 = int(roi_norm[0] * W)
    y1 = int(roi_norm[1] * H)
    x2 = int(roi_norm[2] * W)
    y2 = int(roi_norm[3] * H)
    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)


def rect_from_center(cx, cy, w, h, img_shape):
    """Return a rectangle (x1, y1, x2, y2) given a center point and dimensions."""
    H, W = img_shape[:2]
    x1 = max(0, cx - w // 2)
    y1 = max(0, cy - h // 2)
    x2 = min(W, cx + w // 2)
    y2 = min(H, cy + h // 2)
    return x1, y1, x2, y2


# --- Template-Matching ---
def load_templates():
    """
    Load all template images defined in the configuration.
    The structure expects an object with 'path', 'roi', etc.
    """
    for key, config in TEMPLATES_CONFIG.items():
        try:
            paths = config.get("path")
            if isinstance(paths, str):
                paths = [paths]  # Convert single path to a list

            if not paths:
                LOGGER.error(f"No path specified for template '{key}'.")
                raise ValueError(
                    f"The key 'path' is missing or empty for template '{key}'."
                )

            loaded_imgs = []
            for raw_path in paths:
                resolved_path = resolve_template_path(raw_path)
                if not resolved_path or not os.path.exists(resolved_path):
                    LOGGER.error(
                        f"Template '{key}' image missing: {raw_path} (resolved: {resolved_path})."
                    )
                    raise FileNotFoundError(
                        f"Template '{key}' image missing: {raw_path}"
                    )
                img = cv2.imread(resolved_path, cv2.IMREAD_COLOR)
                if img is None:
                    LOGGER.error(
                        f"Template '{key}' image could not be decoded: {resolved_path}"
                    )
                    raise FileNotFoundError(
                        f"Template '{key}' image could not be decoded: {resolved_path}"
                    )
                loaded_imgs.append(img)

            LOADED_TEMPLATES[key] = loaded_imgs

        except Exception as e:
            LOGGER.error(f"Failed to load templates for '{key}': {e}")
            sys.exit(1)


# Initialize templates on script start
load_templates()

# Cache for the most recently detected positions
LAST_SEEN = {}

# Global variable storing the detected scale factor
FOUND_SCALE = None

# Globale Liste, um die Erkennungen pro Screenshot zu sammeln
detections_this_frame = []


def _search_in_rect(screen, templates, thresh, key, roi_offset=(0, 0), scales=[1.0]):
    """Internal helper that executes matchTemplate across multiple scales."""
    global detections_this_frame, OPTIMAL_SCALES, DETECTION_COUNTS
    best_match = (None, 0.0, None, 1.0)  # (position, score, template_shape, scale)

    for scale in scales:
        for tmpl in templates:
            if tmpl is None:
                continue

            scaled_tmpl = (
                tmpl
                if scale == 1.0
                else cv2.resize(
                    tmpl,
                    (0, 0),
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4,
                )
            )
            if (
                scaled_tmpl.shape[0] > screen.shape[0]
                or scaled_tmpl.shape[1] > screen.shape[1]
            ):
                continue

            res = cv2.matchTemplate(screen, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            # Wenn der aktuelle Fund besser ist, aktualisiere best_match
            if max_val > best_match[1]:
                h, w = scaled_tmpl.shape[:2]
                center = (
                    max_loc[0] + w // 2 + roi_offset[0],
                    max_loc[1] + h // 2 + roi_offset[1],
                )
                best_match = (center, max_val, (w, h), scale)

    pos, score, shape, final_scale = best_match
    found = score >= thresh

    # Record every search (successful or not) for the GUI
    if pos and shape:
        w, h = shape
        rect = (pos[0] - w // 2, pos[1] - h // 2, pos[0] + w // 2, pos[1] + h // 2)
        detections_this_frame.append(
            {
                "key": key,
                "rect": rect,
                "score": score,
                "found": found,
                "scale": final_scale,
            }
        )

        # Return the detected scale so it can be cached globally
        if found:
            LOGGER.debug(
                f"Scale for '{key}' detected: {final_scale:.2f} (score: {score:.2f})"
            )
            return pos, score, final_scale

    # If nothing was found, return only position None and the score
    return None, score, None


def find_template(screen, key, thresh=None, use_roi=True):
    """
    Search for a template and return the best position and score.
    Records detection results for the heatmap and applies auto-scaling when configured.
    """
    global FOUND_SCALE
    template_config = TEMPLATES_CONFIG.get(key)
    if not template_config:
        raise KeyError(f"No configuration found for template key '{key}'.")

    templates = LOADED_TEMPLATES.get(key)
    if not templates:
        raise KeyError(f"No template images loaded for '{key}'.")

    current_thresh = thresh or THRESH

    # --- Automatische Skalierung ---
    scale_config = CFG.get("template_scale_test", {})
    scale_template_key = scale_config.get("scale_template_key", "alliance_button")
    scales_to_test = [1.0]

    if (
        FOUND_SCALE is None
        and key == scale_template_key
        and scale_config.get("enabled", False)
    ):
        min_s = scale_config.get("scale_min", 0.8)
        max_s = scale_config.get("scale_max", 1.2)
        step_s = scale_config.get("scale_step", 0.05)
        scales_to_test = np.arange(min_s, max_s, step_s).tolist()
        LOGGER.info(f"Performing initial scale search for '{key}'...")
    elif FOUND_SCALE is not None:
        scales_to_test = [FOUND_SCALE]

    # Hole ROI-Informationen aus Konfiguration
    roi_name = template_config.get("roi")
    use_only_roi = template_config.get("use_only_roi", False)
    fallback_full = not use_only_roi

    # Keep track of the best result across all search regions
    best_overall_pos = None
    best_overall_score = 0.0
    best_found_scale = None

    # Define search functions and their regions
    search_areas = []

    # 1. Local search (cache-based when available)
    if key in LAST_SEEN and use_roi:
        cx, cy, _ = LAST_SEEN[key]
        local_rect = rect_from_center(cx, cy, 300, 200, screen.shape)
        search_region = screen[
            local_rect[1] : local_rect[3], local_rect[0] : local_rect[2]
        ]
        search_areas.append(
            ("local search", search_region, (local_rect[0], local_rect[1]))
        )

    # 2. ROI search
    if use_roi and roi_name and roi_name in CFG.get("rois", {}):
        roi_rect = roi_to_rect(screen.shape, CFG["rois"][roi_name])
        search_region = screen[roi_rect[1] : roi_rect[3], roi_rect[0] : roi_rect[2]]

        # Store the ROI for GUI display
        if GUI_SHOW_ROI:
            detections_this_frame.append(
                {
                    "key": f"ROI: {roi_name}",
                    "rect": roi_rect,
                    "score": 1.0,
                    "found": False,
                    "is_roi": True,
                }
            )
        search_areas.append(
            (f"ROI '{roi_name}'", search_region, (roi_rect[0], roi_rect[1]))
        )

    # 3. Full-screen search (fallback)
    if fallback_full:
        search_areas.append(("full-screen search", screen, (0, 0)))

    # Iterate through search areas and stop once a match crosses the threshold
    for area_name, search_img, offset in search_areas:
        pos, score, found_scale = _search_in_rect(
            search_img,
            templates,
            current_thresh,
            key,
            roi_offset=offset,
            scales=scales_to_test,
        )

        # Track the best score encountered so far
        if score > best_overall_score:
            best_overall_score = score

        # Stop iteration as soon as a hit is found
        if pos:
            best_overall_pos = pos
            best_found_scale = found_scale
            LOGGER.info(
                f"Template '{key}' found in {area_name} at {pos} (score: {score:.2f})."
            )
            break  # Match found

    # After the loop: evaluate results
    if best_overall_pos:
        if (
            FOUND_SCALE is None
            and key == scale_template_key
            and best_found_scale is not None
        ):
            FOUND_SCALE = best_found_scale
            LOGGER.warning(f"Global scale locked to {FOUND_SCALE:.2f}.")
        return best_overall_pos, best_overall_score

    # If nothing is found, log the best score and return
    LOGGER.info(
        f"Template '{key}' not found (threshold: {current_thresh}, best score: {best_overall_score:.2f})."
    )

    # Record a 'not found' detection with the achieved score
    detections_this_frame.append(
        {"key": key, "rect": (0, 0, 0, 0), "score": best_overall_score, "found": False}
    )

    return None, best_overall_score


def find_and_tap(key, thresh=None, use_roi=True, duration=None):
    """Search for a template and tap it when found."""
    global detections_this_frame
    detections_this_frame = []  # Reset detections for this frame
    screen = screenshot_bgr()

    pos, score = find_template(screen, key, thresh, use_roi)

    # Append the current frame and detections to the GUI history
    if GUI_ENABLED and gui_controller:
        gui_controller.add_history(screen, detections_this_frame)
        if not is_paused:  # Render only when the bot is active; the pause loop handles rendering otherwise
            gui_controller.render(screen, detections_this_frame)
    if pos:
        LOGGER.debug(f"Position found: ({pos[0]}, {pos[1]})")

        if duration is None:
            tap(*pos, JITTER)
        else:
            longtap(*pos, duration, JITTER)
        LAST_SEEN[key] = (pos[0], pos[1], time.time())
        return True, score, pos
    return False, score, None


def template_exists(key, thresh=None, use_roi=True):
    """Return True if the template can be located on the current screen."""
    screen = screenshot_bgr()
    threshold = thresh
    if isinstance(thresh, str):
        try:
            threshold = float(thresh)
        except ValueError:
            try:
                threshold = float(str(thresh).replace(",", "."))
            except ValueError:
                threshold = None
    pos, _ = find_template(screen, key, threshold, use_roi)
    return pos is not None


def wait_and_tap(key, timeout=6.0, thresh=None, sleep_interval=1.0, duration=None):
    """Wait for a template during a timeout window and tap it once detected."""
    LOGGER.info(f"Waiting up to {timeout}s for '{key}'...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if is_paused:  # Break the wait loop if the bot is paused
            time.sleep(0.5)
            continue
        if find_and_tap(key, thresh=thresh, use_roi=True, duration=duration)[0]:
            LOGGER.info(f"'{key}' found and tapped.")
            return True
        else:
            time.sleep(sleep_interval)
    LOGGER.warning(f"Timed out waiting for '{key}'.")
    return False


# -- Dashboard (Konsole) --
def strip_ansi(s: str) -> str:
    """Entfernt ANSI-Farbcodes aus einem String."""
    return ANSI_RE.sub("", s or "")


def pad_visible(s: str, width: int, align: str = "<") -> str:
    """Pad a string to a visible width (ignoring ANSI escape codes)."""
    pad = max(0, width - len(strip_ansi(s)))
    if align == ">":
        return f"{' ' * pad}{s}"
    if align == "^":
        return f"{' ' * (pad // 2)}{s}{' ' * (pad - pad // 2)}"
    return f"{s}{' ' * pad}"


def _fmt_uptime(s):
    h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def render_dashboard(stats: Stats, paused: bool):
    """Rendert das Status-Dashboard in der Konsole."""
    sys.stdout.write("\x1b[2J\x1b[H")  # Clear the terminal
    sys.stdout.flush()

    uptime = _fmt_uptime(time.time() - stats.start_ts)
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # Pausenstatus-Anzeige
    pause_str = (
        f"{BOLD}{YELLOW}PAUSED (Press 'P' to resume){RESET}"
        if paused
        else "Pause with 'P'"
    )

    print(
        f"{BOLD}{BOT_TITLE} — Status{RESET}   Uptime: {uptime}   {CYAN}{now_str}{RESET}  {pause_str}"
    )

    def _format_metric_value(value: Any, unit: Optional[str]) -> str:
        if isinstance(value, float):
            if abs(value - round(value)) < 0.01:
                value = int(round(value))
            else:
                value = f"{value:.2f}"
        text = str(value)
        if unit:
            text = f"{text} {unit}"
        return text

    rows: List[Tuple[str, str]] = [
        ("Total actions", str(stats.total_actions)),
    ]
    for entry in stats.dynamic_counters_for_display():
        value = entry.get("value")
        if value is None:
            continue
        label = entry.get("label") or entry.get("key")
        unit = entry.get("unit")
        rows.append((str(label), _format_metric_value(value, unit)))

    print("┌───────────────────────────────────────────┬─────────┐")
    for idx, (label, value_text) in enumerate(rows):
        if idx == 1:
            print("├───────────────────────────────────────────┼─────────┤")
        print(f"│ {pad_visible(label, 41)} │ {pad_visible(value_text, 7)} │")
    print("└───────────────────────────────────────────┴─────────┘\n")
    print("Recent actions:")
    print(
        "┌──────────┬────────────────────┬─────────┬────────────────────────────────┐"
    )
    print(
        "│ Zeit     │ Aktion             │ Status  │ Details                        │"
    )
    print(
        "├──────────┼────────────────────┼─────────┼────────────────────────────────┤"
    )
    for item in list(stats.history):
        status = (
            f"{GREEN}OK{RESET}"
            if item["ok"]
            else f"{RED}FAIL{RESET}"
            if item["ok"] is False
            else "Info"
        )
        print(
            f"│ {item['time']:<8} │ {pad_visible(item['action'], 18)} │ {pad_visible(status, 7, '^')} │ {pad_visible(item['detail'][:30], 30)} │"
        )
    for _ in range(10 - len(stats.history)):
        print(
            "│          │                    │         │                                │"
        )
    print(
        "└──────────┴────────────────────┴─────────┴────────────────────────────────┘"
    )
    sys.stdout.flush()


# --- Hauptschleife ---
def main():
    """Die Hauptfunktion, die den Bot-Loop steuert."""
    global gui_controller, detections_this_frame, bot_is_running, is_paused, STATS

    # Initial detection to determine screen size
    try:
        screen_w, screen_h = control.get_screen_size()
        LOGGER.info(f"Target resolution detected: {screen_w}x{screen_h}")
    except Exception as e:
        LOGGER.error(
            f"Could not connect to target and create screenshot: {e}"
        )
        LOGGER.error(
            "Ensure the ADB emulator or Windows game is running and the configuration is correct."
        )
        sys.exit(1)

    if GUI_ENABLED:
        gui_controller = GUIController(
            GUI_HISTORY_SIZE,
            screen_w,
            screen_h,
            GUI_INITIAL_WIDTH,
            GUI_INITIAL_HEIGHT,
            backend="tk"
        )
        # Global Tk GUI instance for delivering frames to the Monitor tab
        global gui
        gui = GUIManager(CONFIG_PATH, ACTIVE_PROFILE_NAME, list_profiles())

        # Start GUI in separate thread
        gui_thread = gui.run()

        # Wait for GUI to initialize
        time.sleep(2)

    # Register the pause hotkey for the 'P' key only when the app is in focus
    def _keyboard_pause_handler(e):
        try:
            if not application_has_focus():
                return
            toggle_pause(e)
        except Exception:
            pass

    keyboard.on_press_key("p", _keyboard_pause_handler)

    stats = Stats(max_history=10)
    STATS = stats

    load_task_definitions(TASKS_PATH)

    # Lade Intervalle aus der Konfigurationsdatei
    PAUSE_MIN = CFG.get("random_pause_min", 1)
    PAUSE_MAX = CFG.get("random_pause_max", 2)

    LOGGER.info("=" * 50 + "\nBot started. Press 'P' to pause. CTRL+C to quit.\n" + "=" * 50)

    try:
        while bot_is_running:
            if GUI_ENABLED and gui_controller and gui.running:
                # Check for commands from GUI
                command = gui.get_command()
                if command:
                    cmd_type, cmd_data = command
                    print(f"GUI Received command: {cmd_type}")

                    if cmd_type == "stop":
                        bot_is_running = False
                        break
                    elif cmd_type == "pause":
                        _apply_pause_state(True)
                        gui.log_message("Bot paused by user", "WARNING")
                    elif cmd_type == "resume":
                        _apply_pause_state(False)
                        gui.log_message("Bot resumed by user", "INFO")

                # Update stats from global Stats instance
                try:
                    snapshot = stats.snapshot()
                    gui.update_stats(snapshot)
                    gui.update_status(
                        "Running - Actions: "
                        + str(
                            snapshot.get(
                                "total_successes", snapshot.get("total_actions", 0)
                            )
                        )
                    )
                except Exception:
                    pass

            # render Console Dashboard
            render_dashboard(stats, is_paused)

            # Pause mode
            if is_paused:
                # While paused, keep rendering the selected frame in the GUI
                if GUI_ENABLED and gui_controller and gui_controller.detection_history:
                    # Ensure a valid index is selected
                    if (
                        0
                        <= gui_controller.history_view_index
                        < len(gui_controller.detection_history)
                    ):
                        img, dets, _ = gui_controller.detection_history[
                            gui_controller.history_view_index
                        ]
                        gui_controller.render(img, dets)
                time.sleep(0.5)
                continue

            # Active Mode
            now = time.time()

            # Render the current screen at beginning of loop
            if GUI_ENABLED and gui_controller and not is_paused:
                try:
                    screen = screenshot_bgr()
                    detections_this_frame = []
                    gui_controller.render(screen, detections_this_frame)
                except Exception as render_exc:
                    LOGGER.debug(f"Monitor refresh failed: {render_exc}")

            preflight_result = None
            if ensure_critical_task_available("preflight_guard") and TASK_MANAGER:
                preflight_result = TASK_MANAGER.run_task_by_id("preflight_guard")
            if preflight_result and preflight_result.success is False:
                time.sleep(random.uniform(1.0, 2.0))
                continue

            try:
                # if tasks have been modified on disk, reload them
                maybe_reload_tasks(TASKS_PATH)

                if TASK_MANAGER and TASK_MANAGER.tasks:
                    task_results: List[TaskResult] = []
                    try:
                        task_results = TASK_MANAGER.run_due_tasks(now)
                    except Exception as task_exc:
                        LOGGER.error(f"Task execution error: {task_exc}")

                    for result in task_results:
                        task_def = result.context.task
                        stats.record_task(task_def, result)
                        detail = (result.detail or "").strip()

                        if not result.executed:
                            outcome = "SKIPPED"
                            log_message = (
                                f"Task '{task_def.name}' skipped. {detail}".strip()
                            )
                            LOGGER.info(log_message)
                        elif result.success is True:
                            outcome = "SUCCESS"
                            log_message = f"Task '{task_def.name}' completed successfully. {detail}".strip()
                            LOGGER.info(log_message)
                        elif result.success is False:
                            outcome = "FAILED"
                            log_message = (
                                f"Task '{task_def.name}' failed. {detail}".strip()
                            )
                            LOGGER.warning(log_message)
                        else:
                            outcome = "INFO"
                            log_message = (
                                f"Task '{task_def.name}' executed. {detail}".strip()
                            )
                            LOGGER.info(log_message)

                        if GUI_ENABLED and gui:
                            try:
                                gui.add_action(
                                    task_def.name, f"{outcome}: {detail}".strip()
                                )
                            except Exception:
                                pass

            except (RuntimeError, FileNotFoundError) as e:
                LOGGER.error(
                    f"A critical error occurred: {e}. Attempting to stabilize state."
                )
                stats.record("Error", False, str(e)[:50])
                if ensure_critical_task_available("stabilize_home") and TASK_MANAGER:
                    TASK_MANAGER.run_task_by_id("stabilize_home")
            except Exception as e:
                LOGGER.error(f"An unexpected error occurred: {e} . Attempting to stabilize state.")
                stats.record("Error", False, "Unexpected error")
                if ensure_critical_task_available("stabilize_home") and TASK_MANAGER:
                    TASK_MANAGER.run_task_by_id("stabilize_home")

            # Render the current screen at end of loop
            if GUI_ENABLED and gui_controller and not is_paused:
                try:
                    screen = screenshot_bgr()
                    gui_controller.render(screen, detections_this_frame)
                except Exception as render_exc:
                    LOGGER.debug(f"Monitor refresh failed: {render_exc}")

            # Wartezeit am Ende der Schleife
            LOGGER.info("Pausing before the next cycle...")
            # Letztes Bild rendern vor der Pause
            # Render the live screen with the collected detections so that when
            # the bot resumes the Monitor shows current detections instead of an
            # empty/old frame.
            if GUI_ENABLED and gui_controller and not is_paused:
                try:
                    gui_controller.render(screenshot_bgr(), detections_this_frame)
                except Exception:
                    # Fallback to empty list if something goes wrong
                    try:
                        gui_controller.render(screenshot_bgr(), [])
                    except Exception:
                        pass
            time.sleep(random.uniform(PAUSE_MIN, PAUSE_MAX))

    except KeyboardInterrupt:
        LOGGER.info("Bot stopped by user (Ctrl+C).")
    finally:
        if GUI_ENABLED and gui_controller and gui.running:
            gui.stop()
            gui_thread.join(timeout=2)
        bot_is_running = False
        keyboard.unhook_all()
        if GUI_ENABLED and gui_controller:
            cv2.destroyAllWindows()
        print(
            f"\n{BOLD}Bot beendet. Gesamtlaufzeit: {_fmt_uptime(time.time() - stats.start_ts)}{RESET}"
        )


if __name__ == "__main__":
    main()
