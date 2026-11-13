"""TCP IPC client talking directly to ElegooSlicer's running IPCServer."""

from __future__ import annotations

import json
import os
import socket
import struct
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class IPCError(RuntimeError):
    """Raised when communicating with ElegooSlicer IPC fails."""

    def __init__(self, message: str, code: int = -1, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.payload = payload or {}


@dataclass(frozen=True)
class IPCEndpoint:
    port: int
    auth_token: str
    port_file: Path


def get_default_ipc_port_file() -> Path:
    """Return the platform path to ElegooSlicer's `ipc_port.txt`."""
    override = os.environ.get("ELEGOO_SLICER_IPC_PORT_FILE")
    if override:
        return Path(override)

    if os.name == "nt":
        appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(appdata) / "ElegooSlicer" / "cache" / "ipc_port.txt"

    xdg_data = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(xdg_data) / "ElegooSlicer" / "cache" / "ipc_port.txt"


def discover_ipc_endpoint(port_file: Path | None = None) -> IPCEndpoint:
    """Read ElegooSlicer's `ipc_port.txt` (`<port>\\n<auth_token>`)."""
    target = port_file or get_default_ipc_port_file()
    if not target.exists():
        raise IPCError(
            f"ElegooSlicer IPC port file not found at {target}. Is ElegooSlicer running?"
        )
    lines = [line.strip() for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise IPCError(f"Invalid ipc_port.txt contents in {target}")
    try:
        port = int(lines[0])
    except ValueError as exc:
        raise IPCError(f"Invalid port in {target}: {lines[0]}") from exc
    return IPCEndpoint(port=port, auth_token=lines[1], port_file=target)


class ElegooIPCClient:
    """Length-prefixed JSON-over-TCP client for ElegooSlicer `IPCServer`."""

    def __init__(self, port_file: Path | None = None, timeout: float = 10.0) -> None:
        self.port_file = port_file
        self.timeout = timeout
        self._slicer_ns_cache: dict[tuple[int, str], bool] = {}

    def is_available(self) -> bool:
        """Return True if `ipc_port.txt` exists and IPCServer answers `printer.getPrinterList`."""
        try:
            self.call("printer.getPrinterList", timeout=1.5)
            return True
        except Exception:
            return False

    def call(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        """Send an authenticated IPC request and return its `data` payload."""
        endpoint = discover_ipc_endpoint(self.port_file)
        req_id = f"mcp-{uuid.uuid4().hex[:12]}"
        payload = {
            "id": req_id,
            "method": method,
            "auth": endpoint.auth_token,
            "params": params or {},
        }
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        frame = struct.pack(">I", len(raw)) + raw
        wait = timeout if timeout is not None else self.timeout

        try:
            with socket.create_connection(("127.0.0.1", endpoint.port), timeout=wait) as sock:
                sock.settimeout(wait)
                sock.sendall(frame)
                deadline = time.monotonic() + wait
                while time.monotonic() < deadline:
                    msg = self._recv_frame(sock)
                    msg_type = msg.get("type")
                    if msg_type == "event":
                        # Server pushes async telemetry/RTM events on open connections.
                        continue
                    if msg_type == "response" or msg.get("id") == req_id:
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                        code = msg.get("code", 0)
                        if code != 0:
                            raise IPCError(
                                msg.get("message") or f"IPC call {method} failed with code {code}",
                                code=int(code),
                                payload=msg,
                            )
                        return msg.get("data")
        except IPCError:
            raise
        except OSError as exc:
            raise IPCError(f"IPC connection error calling {method}: {exc}") from exc

        raise IPCError(f"Timed out waiting for response to {method}")

    def supports_slicer_namespace(self) -> bool:
        """Return True when the running ElegooSlicer binary implements `slicer.*` (cached per endpoint)."""
        try:
            endpoint = discover_ipc_endpoint(self.port_file)
        except Exception:
            return False
        cache_key = (endpoint.port, endpoint.auth_token)
        if cache_key in self._slicer_ns_cache:
            return self._slicer_ns_cache[cache_key]
        try:
            res = self.call("slicer.ping", {}, timeout=2.0)
            supported = isinstance(res, dict) and bool(res.get("ok"))
        except IPCError:
            supported = False
        self._slicer_ns_cache[cache_key] = supported
        return supported

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise IPCError("IPC socket closed by ElegooSlicer")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    @classmethod
    def _recv_frame(cls, sock: socket.socket) -> dict[str, Any]:
        header = cls._recv_exact(sock, 4)
        (length,) = struct.unpack(">I", header)
        if length == 0 or length > 64 * 1024 * 1024:
            raise IPCError(f"Invalid IPC frame length: {length}")
        body = cls._recv_exact(sock, length).decode("utf-8", errors="replace")
        return json.loads(body)

    # ---- High-level printer & user convenience methods ----

    def get_printer_list(self) -> list[dict[str, Any]]:
        data = self.call("printer.getPrinterList")
        if isinstance(data, list):
            return data
        return data.get("printers", []) if isinstance(data, dict) else []

    def get_selected_printer(self, printer_model: str = "", printer_id: str = "") -> str:
        """Return selected printerId (or online/first printer without triggering IPC null-param errors)."""
        if printer_model or printer_id:
            try:
                res = self.call(
                    "printer.getSelectedPrinter",
                    {"printerModel": printer_model, "printerId": printer_id},
                )
                if isinstance(res, str):
                    return res
                if isinstance(res, dict) and res.get("printerId"):
                    return str(res.get("printerId", ""))
            except IPCError:
                pass
        try:
            printers = self.get_printer_list()
        except IPCError:
            return ""
        if not printers:
            return ""
        online = [p for p in printers if p.get("connectStatus") == 1]
        chosen = online[0] if online else printers[0]
        return str(chosen.get("printerId", ""))

    def get_printer_network_info(self, printer_id: str) -> dict[str, Any]:
        return self.call("printer.getPrinterNetworkInfo", {"printerId": printer_id}) or {}

    def get_printer_status_raw(self, printer_id: str) -> dict[str, Any]:
        return self.call("printer.getPrinterStatusRaw", {"printerId": printer_id}) or {}

    def refresh_printer_status(self, printer_id: str) -> Any:
        return self.call("printer.refreshPrinterStatus", {"printerId": printer_id})

    def get_printer_mms_info(self, printer_id: str) -> dict[str, Any]:
        return self.call("printer.getPrinterMmsInfo", {"printerId": printer_id}) or {}

    def get_file_list(self, printer_id: str, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        return self.call(
            "printer.getFileList",
            {"printerId": printer_id, "pageNumber": page, "pageSize": page_size},
        ) or {}

    def get_print_task_list(self, printer_id: str, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        return self.call(
            "printer.getPrintTaskList",
            {"printerId": printer_id, "pageNumber": page, "pageSize": page_size},
        ) or {}

    def get_exception_list(self, printer_id: str, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        return self.call(
            "printer.getExceptionList",
            {"printerId": printer_id, "pageNumber": page, "pageSize": page_size},
        ) or {}

    def delete_print_tasks(self, printer_id: str, task_ids: list[str]) -> Any:
        return self.call("printer.deletePrintTasks", {"printerId": printer_id, "taskIds": task_ids})

    def send_rtm_message(self, printer_id: str, message: str | dict[str, Any]) -> Any:
        msg_str = message if isinstance(message, str) else json.dumps(message, ensure_ascii=False)
        return self.call("printer.sendRtmMessage", {"printerId": printer_id, "message": msg_str})

    def send_sdcp_command(
        self,
        printer_id: str,
        cmd: int,
        data: dict[str, Any] | None = None,
        mainboard_id: str = "",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Build & send an ElegooLink SDCP/RTM command (129=Pause, 130=Stop, 131=Resume, 0=Refresh)."""
        mb_id = mainboard_id
        if not mb_id:
            try:
                info = self.get_printer_network_info(printer_id)
                mb_id = str(info.get("mainboardId") or "")
            except IPCError:
                mb_id = ""
        envelope = {
            "Id": mb_id,
            "Topic": f"sdcp/request/{mb_id}" if mb_id else "",
            "Data": {
                "Cmd": int(cmd),
                "Data": data or {},
                "MainboardID": mb_id,
                "RequestID": uuid.uuid4().hex,
                "TimeStamp": int(time.time()),
                "From": 0,
            },
        }
        if dry_run:
            return {"dry_run": True, "printerId": printer_id, "envelope": envelope}
        res = self.send_rtm_message(printer_id, envelope)
        return {"ok": True, "printerId": printer_id, "result": res, "envelope": envelope}

    def upload_and_print(
        self,
        printer_id: str,
        file_path: str,
        file_name: str,
        upload_and_start_print: bool = True,
        bed_type: int = 0,
        timelapse: bool = False,
        heated_bed_leveling: bool = False,
        auto_refill: bool = False,
        has_mms: bool = False,
        filament_mms_mapping: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        params = {
            "printerId": printer_id,
            "filePath": file_path,
            "fileName": file_name,
            "bedType": bed_type,
            "timeLapse": timelapse,
            "heatedBedLeveling": heated_bed_leveling,
            "autoRefill": auto_refill,
            "uploadAndStartPrint": upload_and_start_print,
            "hasMms": has_mms,
            "filamentMmsMappingList": filament_mms_mapping or [],
        }
        return self.call("printer.upload", params, timeout=30.0) or {}

    def get_upload_task(self, task_id: str) -> dict[str, Any]:
        return self.call("printer.getUploadTask", {"taskId": task_id}) or {}

    def cancel_upload_task(self, task_id: str) -> Any:
        return self.call("printer.cancelUploadTask", {"taskId": task_id})

    def get_user_info(self) -> dict[str, Any]:
        return self.call("user.getUserInfo") or {}
