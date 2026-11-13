"""Windows/desktop GUI bridge to open, inspect, and control ElegooSlicer windows & dialogs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def find_elegoo_slicer_executable() -> Path:
    """Locate the `elegoo-slicer.exe` (or OS binary) to run GUI/CLI operations."""
    candidates: list[Path] = []
    env_bin = os.environ.get("ELEGOO_SLICER_BIN")
    if env_bin:
        candidates.append(Path(env_bin))

    repo_root = Path(__file__).resolve().parent.parent
    if os.name == "nt":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        candidates.extend(
            [
                repo_root / "build" / "src" / "Release" / "elegoo-slicer.exe",
                repo_root / "build_release" / "src" / "Release" / "elegoo-slicer.exe",
                Path(pf) / "ElegooSlicer" / "elegoo-slicer.exe",
            ]
        )
    else:
        candidates.extend(
            [
                repo_root / "build" / "src" / "elegoo-slicer",
                Path("/usr/bin/elegoo-slicer"),
                Path("/usr/local/bin/elegoo-slicer"),
            ]
        )

    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "ElegooSlicer binary not found. Set ELEGOO_SLICER_BIN to the path of elegoo-slicer.exe."
    )


def get_elegoo_conf_path(conf_path: Path | str | None = None) -> Path:
    """Resolve the path to `ElegooSlicer.conf`."""
    if conf_path is not None:
        return Path(conf_path)
    env_conf = os.environ.get("ELEGOO_SLICER_CONF")
    if env_conf:
        return Path(env_conf)
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else (Path.home() / "AppData" / "Roaming")
        return base / "ElegooSlicer" / "ElegooSlicer.conf"
    if os.uname().sysname == "Darwin":
        return Path.home() / "Library" / "Application Support" / "ElegooSlicer" / "ElegooSlicer.conf"
    return Path.home() / ".config" / "ElegooSlicer" / "ElegooSlicer.conf"


def _load_elegoo_conf_json(path: Path) -> dict[str, Any]:
    """Load ElegooSlicer.conf ignoring the trailing `# MD5 checksum ...` footer."""
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(raw.lstrip())
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected root JSON type in {path}")
    return data


def _write_elegoo_conf_json(path: Path, data: dict[str, Any]) -> None:
    """Write ElegooSlicer.conf atomically with ElegooSlicer's uppercase MD5 checksum footer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, indent="\t", ensure_ascii=False)
    digest = hashlib.md5(body.encode("utf-8")).hexdigest().upper()
    content = f"{body}\n# MD5 checksum {digest}\n"
    tmp_path = path.with_name(f"{path.name}.mcp.tmp")
    tmp_path.write_text(content, encoding="utf-8", newline="\r\n" if os.name == "nt" else "\n")
    tmp_path.replace(path)


def get_elegoo_app_config(
    keys: list[str] | None = None,
    conf_path: Path | str | None = None,
) -> dict[str, Any]:
    """Read ElegooSlicer `app` configuration settings from `ElegooSlicer.conf`."""
    target = get_elegoo_conf_path(conf_path)
    data = _load_elegoo_conf_json(target)
    app_cfg = data.get("app", {})
    if not isinstance(app_cfg, dict):
        app_cfg = {}
    if keys:
        selected = {k: app_cfg.get(k) for k in keys}
    else:
        selected = dict(app_cfg)
    return {"conf_path": str(target), "exists": target.exists(), "app": selected}


def set_elegoo_app_config(
    updates: dict[str, Any],
    conf_path: Path | str | None = None,
) -> dict[str, Any]:
    """Update `app` keys in `ElegooSlicer.conf` while keeping the MD5 checksum valid."""
    target = get_elegoo_conf_path(conf_path)
    data = _load_elegoo_conf_json(target)
    app_cfg = data.setdefault("app", {})
    if not isinstance(app_cfg, dict):
        app_cfg = {}
        data["app"] = app_cfg
    for k, v in updates.items():
        app_cfg[k] = v
    _write_elegoo_conf_json(target, data)
    return {"conf_path": str(target), "updated": dict(updates)}


def _cstyle_escape_args(args: list[str]) -> str:
    """Match Slic3r::escape_strings_cstyle (semicolon-delimited) used by InstanceCheck."""
    escaped: list[str] = []
    for item in args:
        needs_quotes = any(ch in item for ch in (" ", "\t", '"', "\\", "\n", "\r", ";")) or (
            len(args) == 1 and item == ""
        )
        out = []
        for ch in item:
            if ch == "\\":
                out.append("\\\\")
            elif ch == '"':
                out.append('\\"')
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(ch)
        s = "".join(out)
        escaped.append(f'"{s}"' if needs_quotes else s)
    return ";".join(escaped)


_PRESET_WARNING_MARKERS = (
    "préréglage personnalisé",
    "prereglage personnalise",
    "customized preset",
    "modified g-code",
    "g-code modifié",
    "g-code modifie",
)

_PROGRESS_DIALOG_MARKERS = (
    "chargement",
    "loading",
    "slicing",
    "tranchage",
    "please wait",
    "veuillez patienter",
)

_DSA_MARKERS = (
    "ne plus afficher",
    "don't show again",
    "do not show again",
)

_BUTTON_ALIASES: dict[str, tuple[str, ...]] = {
    "ok": ("ok", "valider", "confirm", "confirmer", "yes", "oui"),
    "cancel": ("cancel", "annuler", "later", "plus tard"),
    "yes": ("yes", "oui", "ok"),
    "no": ("no", "non", "cancel", "annuler"),
    "close": ("close", "fermer"),
}

_VK_MAP: dict[str, int] = {
    "enter": 0x0D,
    "return": 0x0D,
    "escape": 0x1B,
    "esc": 0x1B,
    "space": 0x20,
    "tab": 0x09,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
}


def _find_elegoo_pids_nt() -> list[int]:
    """Return all running ElegooSlicer process IDs on Windows via Toolhelp32."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    TH32CS_SNAPPROCESS = 0x00000002
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_void_p),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == INVALID_HANDLE_VALUE or not snap:
        return []
    pids: list[int] = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            exe_name = entry.szExeFile.lower()
            if exe_name in {"elegoo-slicer.exe", "elegoo_slicer.exe", "elegooslicer.exe"}:
                pids.append(int(entry.th32ProcessID))
            ok = kernel32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snap)
    return pids


def _win32_text(user32: Any, hwnd: int) -> str:
    import ctypes

    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 2)
    user32.GetWindowTextW(hwnd, buf, length + 2)
    return buf.value.strip()


def _win32_class(user32: Any, hwnd: int) -> str:
    import ctypes

    buf = ctypes.create_unicode_buffer(128)
    user32.GetClassNameW(hwnd, buf, 128)
    return buf.value


def _win32_rect(user32: Any, hwnd: int) -> dict[str, int]:
    from ctypes import byref, wintypes

    rc = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, byref(rc)):
        return {"x": 0, "y": 0, "width": 0, "height": 0}
    return {
        "x": int(rc.left),
        "y": int(rc.top),
        "width": int(rc.right - rc.left),
        "height": int(rc.bottom - rc.top),
    }


def _classify_child_control(user32: Any, child_hwnd: int, parent_title: str) -> dict[str, Any]:
    class_name = _win32_class(user32, child_hwnd)
    text = _win32_text(user32, child_hwnd)
    visible = bool(user32.IsWindowVisible(child_hwnd))
    enabled = bool(user32.IsWindowEnabled(child_hwnd))
    rect = _win32_rect(user32, child_hwnd)
    style = int(user32.GetWindowLongW(child_hwnd, -16))  # GWL_STYLE
    text_low = text.lower()
    is_dsa_label = any(marker in text_low for marker in _DSA_MARKERS)

    role = "other"
    checked: bool | None = None
    if class_name == "Button":
        btn_type = style & 0x0F
        if btn_type in {2, 3, 5, 6} or (
            btn_type == 11 and not text and 10 <= rect["width"] <= 40 and 10 <= rect["height"] <= 40
        ):
            role = "checkbox"
            BM_GETCHECK = 0x00F0
            checked = bool(user32.SendMessageW(child_hwnd, BM_GETCHECK, 0, 0) == 1)
        else:
            role = "button"
    elif class_name == "Static":
        role = "text"
    elif class_name in {"Edit", "RichEdit20W", "RichEditD2DPT"}:
        role = "edit"
    elif class_name == "wxWindowNR":
        # Custom Widgets/Button inherits StaticBox/wxWindow -> wxWindowNR with compact bounds & label
        if (
            text
            and text != parent_title
            and 12 <= rect["height"] <= 80
            and 24 <= rect["width"] <= 380
        ):
            role = "button"
        elif is_dsa_label:
            role = "text"

    info: dict[str, Any] = {
        "hwnd": int(child_hwnd),
        "class_name": class_name,
        "role": role,
        "text": text,
        "visible": visible,
        "enabled": enabled,
        "rect": rect,
    }
    if checked is not None:
        info["checked"] = checked
    if is_dsa_label:
        info["is_dont_show_again"] = True
    return info


def _classify_dialog_kind(title: str, texts: list[str], buttons: list[dict[str, Any]]) -> str:
    combined = " ".join([title, *texts]).lower()
    if any(marker in combined for marker in _PRESET_WARNING_MARKERS):
        return "preset_warning"
    if not buttons and any(marker in combined for marker in _PROGRESS_DIALOG_MARKERS):
        return "progress"
    return "dialog"


def _inspect_top_window_nt(user32: Any, hwnd: int, pid: int, include_children: bool) -> dict[str, Any]:
    import ctypes
    from ctypes import wintypes

    class_name = _win32_class(user32, hwnd)
    title = _win32_text(user32, hwnd)
    visible = bool(user32.IsWindowVisible(hwnd))
    enabled = bool(user32.IsWindowEnabled(hwnd))
    minimized = bool(user32.IsIconic(hwnd))
    maximized = bool(user32.IsZoomed(hwnd))
    owner_hwnd = int(user32.GetWindow(hwnd, 4) or 0)  # GW_OWNER
    foreground_hwnd = int(user32.GetForegroundWindow() or 0)
    style = int(user32.GetWindowLongW(hwnd, -16))  # GWL_STYLE
    rect = _win32_rect(user32, hwnd)

    is_main_window = False
    if class_name == "wxWindowNR":
        has_minor = user32.GetPropW(hwnd, "Instance_Hash_Minor")
        has_major = user32.GetPropW(hwnd, "Instance_Hash_Major")
        is_main_window = bool(has_minor and has_major)

    WS_POPUP = 0x80000000
    is_dialog = (not is_main_window) and (
        owner_hwnd != 0 or class_name == "#32770" or bool(style & WS_POPUP) or bool(title)
    )

    controls: list[dict[str, Any]] = []
    buttons: list[dict[str, Any]] = []
    checkboxes: list[dict[str, Any]] = []
    texts: list[str] = []

    if include_children:
        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def child_cb(child_hwnd: int, _lparam: int) -> bool:
            ctrl = _classify_child_control(user32, child_hwnd, title)
            controls.append(ctrl)
            if ctrl["visible"]:
                if ctrl["role"] == "button" and ctrl["text"]:
                    buttons.append(ctrl)
                elif ctrl["role"] == "checkbox" or ctrl.get("is_dont_show_again"):
                    checkboxes.append(ctrl)
                elif ctrl["role"] == "text" and ctrl["text"]:
                    texts.append(ctrl["text"])
            return True

        user32.EnumChildWindows(hwnd, WNDENUMPROC(child_cb), 0)

    dialog_kind = _classify_dialog_kind(title, texts, buttons) if is_dialog else None

    info: dict[str, Any] = {
        "hwnd": int(hwnd),
        "pid": pid,
        "owner_hwnd": owner_hwnd,
        "title": title,
        "class_name": class_name,
        "visible": visible,
        "enabled": enabled,
        "minimized": minimized,
        "maximized": maximized,
        "foreground": foreground_hwnd == hwnd,
        "is_main_window": is_main_window,
        "is_dialog": is_dialog,
        "dialog_kind": dialog_kind,
        "rect": rect,
        "buttons": buttons,
        "checkboxes": checkboxes,
        "texts": texts,
    }
    if include_children:
        info["controls"] = controls
    return info


def list_elegoo_windows(
    include_invisible: bool = False,
    include_children: bool = True,
) -> dict[str, Any]:
    """List running ElegooSlicer GUI windows, popups, modal dialogs, and child controls."""
    if os.name != "nt":
        return {
            "platform": os.name,
            "running": False,
            "pids": [],
            "windows": [],
            "dialogs": [],
            "note": "Window enumeration is currently implemented for Windows.",
        }

    import ctypes
    from ctypes import wintypes

    pids = _find_elegoo_pids_nt()
    if not pids:
        return {"platform": "nt", "running": False, "pids": [], "windows": [], "dialogs": []}

    pid_set = set(pids)
    user32 = ctypes.windll.user32
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    windows: list[dict[str, Any]] = []

    def enum_cb(hwnd: int, _lparam: int) -> bool:
        proc_id = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(proc_id))
        if int(proc_id.value) not in pid_set:
            return True
        if not include_invisible and not user32.IsWindowVisible(hwnd):
            return True
        win_info = _inspect_top_window_nt(user32, int(hwnd), int(proc_id.value), include_children)
        # Skip toolless tiny helper windows when include_invisible=False unless main/titled
        if (
            not include_invisible
            and not win_info["is_main_window"]
            and not win_info["title"]
            and not win_info["buttons"]
            and (win_info["rect"]["width"] <= 2 or win_info["rect"]["height"] <= 2)
        ):
            return True
        windows.append(win_info)
        return True

    user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
    dialogs = [w for w in windows if w["is_dialog"] and w["visible"]]
    main_windows = [w for w in windows if w["is_main_window"]]
    return {
        "platform": "nt",
        "running": True,
        "pids": pids,
        "main_hwnd": main_windows[0]["hwnd"] if main_windows else None,
        "windows": windows,
        "dialogs": dialogs,
    }


def _select_elegoo_window(
    hwnd: int | None = None,
    title_contains: str | None = None,
    prefer_dialog: bool = False,
    skip_progress: bool = False,
) -> dict[str, Any]:
    state = list_elegoo_windows(include_invisible=False, include_children=True)
    if not state["running"]:
        raise RuntimeError("ElegooSlicer is not currently running.")

    windows: list[dict[str, Any]] = state["windows"]
    if hwnd is not None:
        for w in windows:
            if w["hwnd"] == int(hwnd):
                return w
        raise ValueError(f"No visible ElegooSlicer window found with hwnd={hwnd}.")

    if title_contains:
        needle = title_contains.lower()
        matches = [w for w in windows if needle in w["title"].lower()]
        if not matches:
            matches = [
                w
                for w in windows
                if any(needle in t.lower() for t in w.get("texts", []))
            ]
        if not matches:
            raise ValueError(f"No visible ElegooSlicer window matches title '{title_contains}'.")
        if prefer_dialog:
            dlg_matches = [w for w in matches if w["is_dialog"]]
            if dlg_matches:
                return dlg_matches[0]
        return matches[0]

    if prefer_dialog:
        dialogs = state["dialogs"]
        if skip_progress:
            dialogs = [d for d in dialogs if d.get("dialog_kind") != "progress"]
        preset_dlgs = [d for d in dialogs if d.get("dialog_kind") == "preset_warning"]
        if preset_dlgs:
            return preset_dlgs[0]
        if dialogs:
            return dialogs[0]
        raise ValueError("No visible ElegooSlicer dialog/popup is currently open.")

    if state["dialogs"]:
        non_prog = [d for d in state["dialogs"] if d.get("dialog_kind") != "progress"]
        if non_prog:
            return non_prog[0]
    mains = [w for w in windows if w["is_main_window"]]
    if mains:
        return mains[0]
    if windows:
        return windows[0]
    raise RuntimeError("ElegooSlicer process is running, but no visible GUI windows were found.")


def _focus_hwnd_nt(user32: Any, kernel32: Any, hwnd: int) -> None:
    from ctypes import wintypes

    SW_RESTORE = 9
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    pid = wintypes.DWORD(0)
    target_tid = user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value:
        user32.AllowSetForegroundWindow(pid.value)
    fg = user32.GetForegroundWindow()
    fg_tid = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    cur_tid = kernel32.GetCurrentThreadId()
    attached_fg = bool(fg_tid and fg_tid != cur_tid and user32.AttachThreadInput(cur_tid, fg_tid, True))
    attached_tgt = bool(target_tid and target_tid != cur_tid and user32.AttachThreadInput(cur_tid, target_tid, True))
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        user32.SetFocus(hwnd)
    finally:
        if attached_tgt:
            user32.AttachThreadInput(cur_tid, target_tid, False)
        if attached_fg:
            user32.AttachThreadInput(cur_tid, fg_tid, False)


def _click_hwnd_nt(user32: Any, hwnd: int) -> None:
    from ctypes import byref, wintypes

    rc = wintypes.RECT()
    user32.GetClientRect(hwnd, byref(rc))
    cx = max(1, (int(rc.right) - int(rc.left)) // 2)
    cy = max(1, (int(rc.bottom) - int(rc.top)) // 2)
    lparam = ((cy & 0xFFFF) << 16) | (cx & 0xFFFF)
    WM_LBUTTONDOWN = 0x0201
    WM_LBUTTONUP = 0x0202
    MK_LBUTTON = 0x0001
    user32.PostMessageW(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
    time.sleep(0.03)
    user32.PostMessageW(hwnd, WM_LBUTTONUP, 0, lparam)


def _send_key_nt(user32: Any, hwnd: int, key_name: str) -> None:
    vk = _VK_MAP.get(key_name.strip().lower())
    if vk is None:
        raise ValueError(f"Unsupported key '{key_name}'. Supported keys: {sorted(_VK_MAP)}")
    WM_KEYDOWN = 0x0100
    WM_KEYUP = 0x0101
    user32.PostMessageW(hwnd, WM_KEYDOWN, vk, 1)
    time.sleep(0.02)
    user32.PostMessageW(hwnd, WM_KEYUP, vk, 0xC0000001)


def _match_dialog_button(buttons: list[dict[str, Any]], button: str) -> dict[str, Any] | None:
    if not buttons:
        return None
    target = button.strip().lower()
    if not target:
        return buttons[0]

    for b in buttons:
        if b["text"].strip().lower() == target:
            return b

    aliases = _BUTTON_ALIASES.get(target, (target,))
    for alias in aliases:
        for b in buttons:
            if b["text"].strip().lower() == alias:
                return b

    for b in buttons:
        bt = b["text"].strip().lower()
        if target in bt or bt in target:
            return b
    return None


def control_elegoo_window(
    hwnd: int | None = None,
    action: str = "focus",
    title_contains: str | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    """Perform a window-management action on an open ElegooSlicer window or dialog."""
    if os.name != "nt":
        raise NotImplementedError("Window control is currently supported on Windows only.")

    import ctypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    win = _select_elegoo_window(hwnd=hwnd, title_contains=title_contains, prefer_dialog=False)
    target_hwnd = win["hwnd"]
    act = action.strip().lower()

    if act == "focus":
        _focus_hwnd_nt(user32, kernel32, target_hwnd)
    elif act == "minimize":
        user32.ShowWindow(target_hwnd, 6)  # SW_MINIMIZE
    elif act == "maximize":
        user32.ShowWindow(target_hwnd, 3)  # SW_MAXIMIZE
    elif act == "restore":
        user32.ShowWindow(target_hwnd, 9)  # SW_RESTORE
    elif act == "close":
        user32.PostMessageW(target_hwnd, 0x0010, 0, 0)  # WM_CLOSE
    elif act == "send_key":
        if not key:
            raise ValueError("'key' parameter is required when action='send_key'.")
        _send_key_nt(user32, target_hwnd, key)
    else:
        raise ValueError(
            f"Unknown action '{action}'. Supported: focus, minimize, maximize, restore, close, send_key."
        )

    return {
        "hwnd": target_hwnd,
        "title": win["title"],
        "action": act,
        "key": key,
        "ok": True,
    }


def _toggle_dont_show_again(user32: Any, dlg: dict[str, Any]) -> bool:
    checkboxes = dlg.get("checkboxes", [])
    if not checkboxes:
        return False

    native_cbs = [c for c in checkboxes if c.get("role") == "checkbox"]
    for cb in native_cbs:
        if cb.get("checked") is True:
            return True

    # Clicking m_text_dsa ("Ne plus afficher" / "Don't show again") invokes on_toggle() in MsgDialog.cpp,
    # while clicking the CheckBox child works when there is only a checkbox HWND.
    dsa_labels = [c for c in checkboxes if c.get("is_dont_show_again")]
    target_ctrl = dsa_labels[0] if dsa_labels else native_cbs[0]
    _click_hwnd_nt(user32, target_ctrl["hwnd"])
    time.sleep(0.05)
    return True


def handle_elegoo_dialog(
    hwnd: int | None = None,
    title_contains: str | None = None,
    action: str = "click_button",
    button: str = "OK",
    check_dont_show_again: bool = False,
    persist_no_preset_warning: bool = False,
) -> dict[str, Any]:
    """Interact with a modal ElegooSlicer dialog (e.g. `Préréglage personnalisé` / `Customized Preset`)."""
    if os.name != "nt":
        raise NotImplementedError("Dialog control is currently supported on Windows only.")

    import ctypes

    user32 = ctypes.windll.user32
    dlg = _select_elegoo_window(
        hwnd=hwnd,
        title_contains=title_contains,
        prefer_dialog=True,
        skip_progress=(hwnd is None and title_contains is None),
    )
    dlg_hwnd = dlg["hwnd"]
    act = action.strip().lower()

    dsa_toggled = False
    if check_dont_show_again or act == "check_dont_show_again":
        dsa_toggled = _toggle_dont_show_again(user32, dlg)

    conf_updated = None
    if persist_no_preset_warning or check_dont_show_again:
        try:
            conf_updated = set_elegoo_app_config({"no_warn_when_modified_gcodes": True})
        except Exception as exc:
            conf_updated = {"error": str(exc)}

    clicked_button: dict[str, Any] | None = None
    closed = False

    if act == "check_dont_show_again":
        return {
            "hwnd": dlg_hwnd,
            "title": dlg["title"],
            "dialog_kind": dlg.get("dialog_kind"),
            "action": act,
            "dont_show_again_toggled": dsa_toggled,
            "app_config": conf_updated,
            "closed": False,
        }

    if act == "close":
        user32.PostMessageW(dlg_hwnd, 0x0010, 0, 0)  # WM_CLOSE
        deadline = time.time() + 0.8
        while time.time() < deadline:
            if not user32.IsWindow(dlg_hwnd) or not user32.IsWindowVisible(dlg_hwnd):
                closed = True
                break
            time.sleep(0.05)
        return {
            "hwnd": dlg_hwnd,
            "title": dlg["title"],
            "dialog_kind": dlg.get("dialog_kind"),
            "action": act,
            "dont_show_again_toggled": dsa_toggled,
            "app_config": conf_updated,
            "closed": closed,
        }

    if act not in {"click_button", "dismiss_preset_warning"}:
        raise ValueError(
            f"Unknown dialog action '{action}'. Supported: click_button, dismiss_preset_warning, check_dont_show_again, close."
        )

    matched = _match_dialog_button(dlg.get("buttons", []), button)
    if matched is not None:
        clicked_button = {"hwnd": matched["hwnd"], "text": matched["text"]}
        _click_hwnd_nt(user32, matched["hwnd"])
        deadline = time.time() + 0.6
        while time.time() < deadline:
            if not user32.IsWindow(dlg_hwnd) or not user32.IsWindowVisible(dlg_hwnd):
                closed = True
                break
            time.sleep(0.05)

        if not closed:
            if matched.get("class_name") == "Button":
                user32.PostMessageW(matched["hwnd"], 0x00F5, 0, 0)  # BM_CLICK
            fallback_key = "escape" if button.strip().lower() in {"cancel", "annuler", "no", "non"} else "enter"
            _send_key_nt(user32, matched["hwnd"], fallback_key)
            time.sleep(0.15)
            closed = not user32.IsWindow(dlg_hwnd) or not user32.IsWindowVisible(dlg_hwnd)
    else:
        # Fallback when dialog has no exposed button HWND
        fallback_key = "escape" if button.strip().lower() in {"cancel", "annuler", "no", "non"} else "enter"
        _send_key_nt(user32, dlg_hwnd, fallback_key)
        time.sleep(0.15)
        closed = not user32.IsWindow(dlg_hwnd) or not user32.IsWindowVisible(dlg_hwnd)

    return {
        "hwnd": dlg_hwnd,
        "title": dlg["title"],
        "dialog_kind": dlg.get("dialog_kind"),
        "action": act,
        "button": clicked_button,
        "dont_show_again_toggled": dsa_toggled,
        "app_config": conf_updated,
        "closed": closed,
    }


def dismiss_preset_dialogs(
    dont_show_again: bool = False,
    timeout_seconds: float = 2.5,
    prefer_button: str = "OK",
) -> list[dict[str, Any]]:
    """Wait briefly and confirm any preset/G-code confirmation dialog spawned during `.3mf` load."""
    if os.name != "nt":
        return []
    handled: list[dict[str, Any]] = []
    seen_hwnds: set[int] = set()
    deadline = time.time() + max(0.0, timeout_seconds)
    while True:
        state = list_elegoo_windows(include_invisible=False, include_children=True)
        for dlg in state.get("dialogs", []):
            if dlg.get("dialog_kind") == "preset_warning" and dlg["hwnd"] not in seen_hwnds:
                seen_hwnds.add(dlg["hwnd"])
                try:
                    res = handle_elegoo_dialog(
                        hwnd=dlg["hwnd"],
                        action="click_button",
                        button=prefer_button,
                        check_dont_show_again=dont_show_again,
                        persist_no_preset_warning=dont_show_again,
                    )
                    handled.append(res)
                    deadline = max(deadline, time.time() + 0.35)
                except Exception as exc:
                    handled.append({"hwnd": dlg["hwnd"], "title": dlg.get("title"), "error": str(exc)})
        if time.time() >= deadline:
            break
        time.sleep(0.15)
    return handled


def launch_elegoo_gui(
    project_path: Path | str | None = None,
    exe_path: Path | None = None,
    wait_seconds: float = 10.0,
    auto_confirm_preset_warning: bool = True,
    dont_show_again: bool = False,
) -> dict[str, Any]:
    """Launch ElegooSlicer desktop GUI (empty or with a `.3mf` project) and return window state."""
    slicer_bin = exe_path or find_elegoo_slicer_executable()
    resolved_project = str(Path(project_path).resolve()) if project_path else None

    if resolved_project is not None:
        return send_project_to_running_gui(
            Path(resolved_project),
            exe_path=slicer_bin,
            auto_confirm_preset_warning=auto_confirm_preset_warning,
            dont_show_again=dont_show_again,
        )

    existing = list_elegoo_windows(include_invisible=False, include_children=False)
    if existing["running"] and existing.get("main_hwnd"):
        if os.name == "nt":
            try:
                control_elegoo_window(hwnd=existing["main_hwnd"], action="focus")
            except Exception:
                pass
        return {
            "method": "already_running",
            "pid": existing["pids"][0],
            "main_hwnd": existing["main_hwnd"],
            "windows": existing["windows"],
        }

    cmd = [str(slicer_bin)]
    if os.name == "nt":
        DETACHED_PROCESS = 0x00000008
        proc = subprocess.Popen(cmd, creationflags=DETACHED_PROCESS, close_fds=True)
    else:
        proc = subprocess.Popen(cmd)

    main_hwnd = None
    windows_state: dict[str, Any] = {"windows": [], "dialogs": []}
    if os.name == "nt":
        deadline = time.time() + max(0.5, wait_seconds)
        while time.time() < deadline:
            windows_state = list_elegoo_windows(include_invisible=False, include_children=True)
            if windows_state.get("main_hwnd"):
                main_hwnd = windows_state["main_hwnd"]
                break
            time.sleep(0.25)

    return {
        "method": "launch",
        "pid": proc.pid,
        "main_hwnd": main_hwnd,
        "windows": windows_state.get("windows", []),
        "dialogs": windows_state.get("dialogs", []),
    }


def send_project_to_running_gui(
    project_path: Path,
    exe_path: Path | None = None,
    auto_confirm_preset_warning: bool = True,
    dont_show_again: bool = False,
    popup_wait_seconds: float = 2.5,
) -> dict[str, Any]:
    """Send WM_COPYDATA on Windows (or spawn single-instance open) to load project in live GUI."""
    resolved_project = str(project_path.resolve())
    slicer_bin = exe_path or find_elegoo_slicer_executable()

    if os.name != "nt":
        proc = subprocess.Popen([str(slicer_bin), resolved_project])
        return {"method": "spawn", "pid": proc.pid, "project": resolved_project}

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32

    class COPYDATASTRUCT(ctypes.Structure):
        _fields_ = [
            ("dwData", ctypes.c_size_t),
            ("cbData", wintypes.DWORD),
            ("lpData", ctypes.c_void_p),
        ]

    matched_hwnds: list[int] = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def enum_cb(hwnd: int, _lparam: int) -> bool:
        buf = ctypes.create_unicode_buffer(128)
        user32.GetClassNameW(hwnd, buf, 128)
        if buf.value == "wxWindowNR":
            # InstanceCheck.cpp checks both Instance_Hash_Minor and Instance_Hash_Major != nullptr
            has_minor = user32.GetPropW(hwnd, "Instance_Hash_Minor")
            has_major = user32.GetPropW(hwnd, "Instance_Hash_Major")
            if has_minor and has_major:
                matched_hwnds.append(hwnd)
        return True

    user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

    if matched_hwnds:
        payload_str = _cstyle_escape_args([str(slicer_bin), resolved_project])
        raw_utf16 = (payload_str + "\0").encode("utf-16-le")
        raw_buf = ctypes.create_string_buffer(raw_utf16, len(raw_utf16))
        cds = COPYDATASTRUCT()
        cds.dwData = 1  # InstanceCheck.hpp WM_COPYDATA command tag
        cds.cbData = len(raw_utf16)
        cds.lpData = ctypes.cast(raw_buf, ctypes.c_void_p)

        pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(matched_hwnds[0], ctypes.byref(pid))
        if pid.value:
            user32.AllowSetForegroundWindow(pid.value)

        dialogs_handled: list[dict[str, Any]] = []
        watcher_thread: threading.Thread | None = None
        if auto_confirm_preset_warning and popup_wait_seconds > 0:
            def _watch() -> None:
                dialogs_handled.extend(
                    dismiss_preset_dialogs(
                        dont_show_again=dont_show_again,
                        timeout_seconds=popup_wait_seconds,
                    )
                )

            watcher_thread = threading.Thread(target=_watch, daemon=True)
            watcher_thread.start()

        SMTO_ABORTIFHUNG = 0x0002
        result_dword = ctypes.c_size_t(0)
        ok = user32.SendMessageTimeoutW(
            matched_hwnds[0],
            0x004A,  # WM_COPYDATA
            0,
            ctypes.byref(cds),
            SMTO_ABORTIFHUNG,
            5000,
            ctypes.byref(result_dword),
        )
        if watcher_thread is not None:
            watcher_thread.join(timeout=popup_wait_seconds + 0.5)

        if ok:
            return {
                "method": "WM_COPYDATA",
                "hwnd": matched_hwnds[0],
                "pid": int(pid.value),
                "result": int(result_dword.value),
                "project": resolved_project,
                "dialogs_handled": dialogs_handled,
            }

    # Fallback: launch ElegooSlicer GUI with the project path
    DETACHED_PROCESS = 0x00000008
    proc = subprocess.Popen(
        [str(slicer_bin), resolved_project],
        creationflags=DETACHED_PROCESS,
        close_fds=True,
    )
    time.sleep(0.5)
    dialogs_handled = []
    if auto_confirm_preset_warning and popup_wait_seconds > 0:
        dialogs_handled = dismiss_preset_dialogs(
            dont_show_again=dont_show_again,
            timeout_seconds=max(popup_wait_seconds, 4.0),
        )
    return {
        "method": "launch",
        "pid": proc.pid,
        "project": resolved_project,
        "dialogs_handled": dialogs_handled,
    }
