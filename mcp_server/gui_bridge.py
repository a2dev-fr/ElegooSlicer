"""Windows/desktop GUI bridge to open or refresh projects inside running ElegooSlicer."""

from __future__ import annotations

import os
import subprocess
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


def _cstyle_escape_args(args: list[str]) -> str:
    """Match Slic3r::escape_strings_cstyle (semicolon-delimited) used by InstanceCheck."""
    escaped: list[str] = []
    for item in args:
        needs_quotes = any(ch in item for ch in (' ', '\t', '"', '\\', '\n', '\r', ';')) or (
            len(args) == 1 and item == ""
        )
        out = []
        for ch in item:
            if ch == '\\':
                out.append("\\\\")
            elif ch == '"':
                out.append('\\"')
            elif ch == '\n':
                out.append("\\n")
            elif ch == '\r':
                out.append("\\r")
            elif ch == '\t':
                out.append("\\t")
            else:
                out.append(ch)
        s = "".join(out)
        escaped.append(f'"{s}"' if needs_quotes else s)
    return ";".join(escaped)


def send_project_to_running_gui(project_path: Path, exe_path: Path | None = None) -> dict[str, Any]:
    """Send WM_COPYDATA on Windows (or spawn single-instance open) to load project in live GUI."""
    resolved_project = str(project_path.resolve())
    slicer_bin = exe_path or find_elegoo_slicer_executable()

    if os.name != "nt":
        proc = subprocess.Popen([str(slicer_bin), resolved_project])
        return {"method": "spawn", "pid": proc.pid, "project": resolved_project}

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

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
        if ok:
            return {
                "method": "WM_COPYDATA",
                "hwnd": matched_hwnds[0],
                "pid": int(pid.value),
                "result": int(result_dword.value),
                "project": resolved_project,
            }

    # Fallback: launch ElegooSlicer GUI with the project path
    DETACHED_PROCESS = 0x00000008
    proc = subprocess.Popen(
        [str(slicer_bin), resolved_project],
        creationflags=DETACHED_PROCESS,
        close_fds=True,
    )
    time.sleep(0.5)
    return {"method": "launch", "pid": proc.pid, "project": resolved_project}
