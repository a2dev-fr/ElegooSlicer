"""End-to-end tests for the ElegooSlicer MCP Server."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from unittest import mock

from mcp_server.config_registry import ConfigRegistry
from mcp_server.ipc_client import ElegooIPCClient
from mcp_server.mesh_engine import make_primitive, save_binary_stl
from mcp_server.project_engine import ProjectEngine
from mcp_server.server import ElegooMCPServer


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = sys.executable


class TestElegooSlicerMCP(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="elegoo_mcp_test_")
        self.tmp_dir = Path(self.tmp.name)
        self.engine = ProjectEngine(workspace_dir=self.tmp_dir / "ws")
        self.server = ElegooMCPServer(engine=self.engine)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _call_tool_raw(self, name: str, arguments: dict | None = None) -> dict:
        resp = self.server.handle_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments or {}},
            }
        )
        self.assertIsNotNone(resp)
        result = resp["result"]
        self.assertFalse(result["isError"], f"Tool {name} failed: {result}")
        return result

    def _call_tool(self, name: str, arguments: dict | None = None) -> dict:
        result = self._call_tool_raw(name, arguments)
        return json.loads(result["content"][0]["text"])

    def test_mcp_stdio_handshake_and_tools_list(self) -> None:
        env = dict(os.environ)
        env["ELEGOO_SLICER_MCP_WORKSPACE"] = str(self.tmp_dir / "stdio_ws")
        reqs = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "params_search", "arguments": {"query": "Hauteur de couche", "limit": 5}},
                }
            ),
        ]
        proc = subprocess.run(
            [PYTHON_EXE, "-m", "mcp_server"],
            cwd=str(REPO_ROOT),
            input="\n".join(reqs) + "\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = [json.loads(line) for line in proc.stdout.strip().splitlines() if line.strip()]
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0]["result"]["serverInfo"]["name"], "elegoo-slicer-mcp")
        tool_names = {t["name"] for t in lines[1]["result"]["tools"]}
        for expected in (
            "project_new",
            "object_add",
            "object_list_faces",
            "object_lay_on_face",
            "object_rotate",
            "filament_set_slots",
            "params_set",
            "slice_project",
            "export_output",
            "printer_print",
            "preview_plate",
            "preview_object_faces",
            "object_params_set",
            "object_modifier_add",
            "object_cut",
            "object_layer_ranges_set",
            "plate_create",
            "plate_move_object",
            "gui_sync_status",
            "profile_save_user",
            "printer_tasks",
            "printer_pause",
        ):
            self.assertIn(expected, tool_names)

        search_payload = json.loads(lines[2]["result"]["content"][0]["text"])
        keys = [r["key"] for r in search_payload["results"]]
        self.assertIn("layer_height", keys)

    def test_french_and_english_parameters_and_profiles(self) -> None:
        reg = ConfigRegistry()
        reg.ensure_params_loaded()
        self.assertEqual(reg.resolve_key("Hauteur de couche"), "layer_height")
        self.assertEqual(reg.resolve_key("Hauteur de couche initiale"), "initial_layer_print_height")
        self.assertEqual(reg.resolve_key("Largeur de ligne > Paroi extérieure"), "outer_wall_line_width")
        self.assertEqual(reg.resolve_key("Position de la couture"), "seam_position")
        self.assertEqual(reg.resolve_key("Coutures intérieures décalées"), "staggered_inner_seams")
        self.assertEqual(reg.resolve_key("Écart de couture"), "seam_gap")
        self.assertEqual(reg.resolve_key("Couture en biseau (beta)"), "seam_slope_type")
        self.assertEqual(reg.normalize_value("seam_position", "Alignée"), "aligned")
        self.assertEqual(reg.normalize_value("outer_wall_line_width", "0,42"), "0.42")

        proj = self._call_tool("project_new", {"project_path": str(self.tmp_dir / "params_test.3mf")})
        self.assertTrue(Path(proj["project_path"]).exists())

        updated = self._call_tool(
            "params_set",
            {
                "parameters": {
                    "Hauteur de couche": "0,16",
                    "Largeur de ligne > Paroi extérieure": 0.42,
                    "Position de la couture": "Alignée",
                    "Coutures intérieures décalées": True,
                    "Écart de couture": "10%",
                    "sparse_infill_density": "25%",
                }
            },
        )
        applied = updated["applied"]
        self.assertEqual(applied["layer_height"]["value"], "0.16")
        self.assertEqual(applied["outer_wall_line_width"]["value"], "0.42")
        self.assertEqual(applied["seam_position"]["value"], "aligned")
        self.assertEqual(applied["staggered_inner_seams"]["value"], "1")
        self.assertEqual(applied["seam_gap"]["value"], "10%")
        self.assertEqual(applied["sparse_infill_density"]["value"], "25%")

        profs = self._call_tool("profiles_list", {"kind": "print", "query": "Elegoo"})
        self.assertGreater(len(profs["print"]), 0)

    def test_multi_objects_faces_lay_on_face_rotate_and_filaments(self) -> None:
        self._call_tool("project_new", {"project_path": str(self.tmp_dir / "multi_obj.3mf")})

        # Create a real binary STL mesh externally (non-cubic cuboid 30x20x10)
        stl_path = self.tmp_dir / "block.stl"
        save_binary_stl(make_primitive("box", {"size_x": 30.0, "size_y": 20.0, "size_z": 10.0}), stl_path)

        added = self._call_tool(
            "object_add",
            {
                "items": [
                    {"path": str(stl_path), "name": "TestBlock", "filament_slot": 1},
                    {"primitive": "cylinder", "params": {"radius": 8.0, "height": 25.0}, "name": "Pin", "filament_slot": 2},
                ],
                "auto_arrange": True,
            },
        )
        self.assertEqual(len(added["objects"]), 2)

        # Rotate TestBlock +90 deg around X so height (Z) becomes ~20mm instead of 10mm
        rot = self._call_tool("object_rotate", {"object_index": 0, "axis": "X", "angle_deg": 90.0, "relative": True})
        self.assertAlmostEqual(rot["object"]["instances"][0]["world_size_mm"][2], 20.0, delta=0.2)
        self.assertAlmostEqual(rot["object"]["instances"][0]["world_bbox_min_mm"][2], 0.0, delta=1e-3)

        # Detect printable faces; face 0 should have largest area (30x20 = 600 mm^2)
        faces_res = self._call_tool("object_list_faces", {"object_index": 0})
        self.assertGreaterEqual(len(faces_res["faces"]), 6)
        self.assertAlmostEqual(faces_res["faces"][0]["area_mm2"], 600.0, delta=0.5)

        # Lay back down onto largest printable face (600 mm^2 -> height returns to 10mm, z_min=0)
        laid = self._call_tool("object_lay_on_face", {"object_index": 0, "face_id": 0})
        self.assertAlmostEqual(laid["object"]["instances"][0]["world_size_mm"][2], 10.0, delta=0.2)
        self.assertAlmostEqual(laid["object"]["instances"][0]["world_bbox_min_mm"][2], 0.0, delta=1e-3)

        # Configure two filament slots and assign slot 2 to Pin
        fset = self._call_tool(
            "filament_set_slots",
            {
                "slots": [
                    {"slot": 1, "color": "#2850DF"},
                    {"slot": 2, "color": "#F72221", "overrides": {"nozzle_temperature": "215"}},
                ]
            },
        )
        self.assertEqual(len(fset["slots"]), 2)
        self.assertEqual(fset["slots"][1]["nozzle_temperature"], "215")

        assigned = self._call_tool("filament_assign_to_object", {"object_index": 1, "slot": 2})
        self.assertEqual(assigned["filament_slot"], 2)

    def test_headless_slice_export_and_print_dry_run(self) -> None:
        self._call_tool("project_new", {"project_path": str(self.tmp_dir / "slice_me.3mf")})
        self._call_tool(
            "object_add",
            {
                "items": [{"primitive": "box", "params": {"size_x": 15.0, "size_y": 15.0, "size_z": 6.0}, "name": "CalibCube"}],
            },
        )
        self._call_tool("params_set", {"parameters": {"Hauteur de couche": 0.2}})

        sliced = self._call_tool("slice_project", {"plate": 1})
        self.assertTrue(sliced["ok"], f"Slicing failed: {sliced}")
        self.assertTrue(Path(sliced["gcode_3mf_path"]).exists())
        self.assertIn("estimated_time", sliced["stats"])

        # Test exports
        out_gcode = self.tmp_dir / "exports" / "calib.gcode"
        out_3mf = self.tmp_dir / "exports" / "calib.3mf"
        out_stl = self.tmp_dir / "exports" / "calib.stl"
        self._call_tool("export_output", {"export_format": "gcode", "destination_path": str(out_gcode)})
        self._call_tool("export_output", {"export_format": "3mf", "destination_path": str(out_3mf)})
        self._call_tool("export_output", {"export_format": "stl", "destination_path": str(out_stl)})
        self.assertGreater(out_gcode.stat().st_size, 1000)
        self.assertGreater(out_stl.stat().st_size, 84)
        with zipfile.ZipFile(out_3mf, "r") as zf:
            self.assertIn("Metadata/project_settings.config", zf.namelist())

        # Verify dry-run printer_print (never triggers real upload/print during test)
        dry = self._call_tool(
            "printer_print",
            {
                "printer_id": "test_printer",
                "sliced_file_path": sliced["gcode_3mf_path"],
                "upload_and_start_print": True,
                "dry_run": True,
            },
        )
        self.assertTrue(dry["dry_run"])
        self.assertEqual(dry["payload"]["printerId"], "test_printer")

    def test_live_elegoo_ipc_readonly_if_running(self) -> None:
        ipc = ElegooIPCClient()
        if not ipc.is_available():
            self.skipTest("ElegooSlicer desktop app is not running with IPC")

        plist = self._call_tool("printer_list")
        self.assertIn("printers", plist)
        if plist["printers"]:
            sync = self._call_tool(
                "project_new",
                {"project_path": str(self.tmp_dir / "live_sync.3mf"), "sync_printer_filaments": True},
            )
            self.assertTrue(Path(sync["project_path"]).exists())
            first_pid = plist["printers"][0].get("printer_id") or plist["printers"][0].get("id")
            if first_pid:
                tasks = self._call_tool("printer_tasks", {"printer_id": first_pid, "page": 1, "page_size": 5})
                self.assertIn("tasks", tasks)
                exc = self._call_tool("printer_exceptions", {"printer_id": first_pid, "page": 1, "page_size": 5})
                self.assertIn("exceptions", exc)

    def test_visual_png_previews(self) -> None:
        self._call_tool("project_new", {"project_path": str(self.tmp_dir / "preview_proj.3mf")})
        self._call_tool(
            "object_add",
            {
                "items": [
                    {"primitive": "box", "params": {"size_x": 30.0, "size_y": 20.0, "size_z": 12.0}, "name": "PlateCube"},
                    {"primitive": "cylinder", "params": {"radius": 8.0, "height": 22.0}, "name": "Tower"},
                ],
                "auto_arrange": True,
            },
        )

        plate_out = self.tmp_dir / "plate1.png"
        plate_raw = self._call_tool_raw(
            "preview_plate",
            {"plate_id": 1, "width": 480, "height": 360, "output_path": str(plate_out)},
        )
        self.assertEqual(len(plate_raw["content"]), 2)
        self.assertEqual(plate_raw["content"][1]["type"], "image")
        self.assertEqual(plate_raw["content"][1]["mimeType"], "image/png")
        plate_png = base64.b64decode(plate_raw["content"][1]["data"])
        self.assertTrue(plate_png.startswith(b"\x89PNG\r\n\x1a\n"))
        plate_meta = json.loads(plate_raw["content"][0]["text"])
        self.assertTrue(Path(plate_meta["output_path"]).exists())

        faces_raw = self._call_tool_raw("preview_object_faces", {"object_index": 0, "top_n": 6})
        self.assertEqual(len(faces_raw["content"]), 2)
        self.assertEqual(faces_raw["content"][1]["type"], "image")
        faces_png = base64.b64decode(faces_raw["content"][1]["data"])
        self.assertTrue(faces_png.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_per_object_modifiers_layer_ranges_cut_and_real_slice(self) -> None:
        self._call_tool("project_new", {"project_path": str(self.tmp_dir / "advanced_ops.3mf")})
        self._call_tool(
            "object_add",
            {"items": [{"primitive": "box", "params": {"size_x": 24.0, "size_y": 24.0, "size_z": 20.0}, "name": "BaseBox"}]},
        )

        # Per-object parameters (French + canonical)
        obj_p = self._call_tool(
            "object_params_set",
            {
                "object_index": 0,
                "parameters": {"sparse_infill_density": "40%", "wall_loops": 3},
            },
        )
        self.assertEqual(obj_p["parameters"]["sparse_infill_density"], "40%")
        got_p = self._call_tool("object_params_get", {"object_index": 0})
        self.assertEqual(got_p["parameters"]["wall_loops"], "3")

        # Add modifier volume + negative volume + support enforcer
        self._call_tool(
            "object_modifier_add",
            {
                "object_index": 0,
                "subtype": "modifier_part",
                "primitive": "box",
                "params": {"size_x": 10.0, "size_y": 10.0, "size_z": 8.0},
                "offset_mm": [0.0, 0.0, 0.0],
                "part_parameters": {"sparse_infill_density": "85%"},
                "name": "DenseZone",
            },
        )
        self._call_tool(
            "object_modifier_add",
            {
                "object_index": 0,
                "subtype": "negative_part",
                "primitive": "cylinder",
                "params": {"radius": 3.5, "height": 24.0},
                "offset_mm": [0.0, 0.0, 0.0],
                "name": "CenterHole",
            },
        )
        self._call_tool(
            "object_modifier_add",
            {
                "object_index": 0,
                "subtype": "support_enforcer",
                "primitive": "box",
                "params": {"size_x": 6.0, "size_y": 6.0, "size_z": 4.0},
                "offset_mm": [5.0, 5.0, -5.0],
                "name": "SupEnf",
            },
        )

        parts = self._call_tool("object_parts_list", {"object_index": 0})
        self.assertEqual(len(parts["parts"]), 4)
        self.assertEqual(
            [p["subtype"] for p in parts["parts"]],
            ["normal_part", "modifier_part", "negative_part", "support_enforcer"],
        )

        # Layer height ranges
        lset = self._call_tool(
            "object_layer_ranges_set",
            {
                "object_index": 0,
                "ranges": [
                    {"min_z": 0.0, "max_z": 5.0, "options": {"layer_height": 0.12}},
                    {"min_z": 5.0, "max_z": 12.0, "options": {"layer_height": 0.24, "sparse_infill_density": "30%"}},
                ],
            },
        )
        self.assertEqual(len(lset["layer_ranges"]), 2)
        lget = self._call_tool("object_layer_ranges_get", {"object_index": 0})
        self.assertEqual(len(lget["layer_ranges"]), 2)
        self.assertEqual(lget["layer_ranges"][0]["options"]["layer_height"], "0.12")

        # Real ElegooSlicer CLI slice of project containing object params + modifier + negative hole + layer ranges
        sliced = self._call_tool("slice_project", {"plate": 1})
        self.assertTrue(sliced["ok"], f"Advanced CLI slice failed: {sliced}")
        self.assertTrue(Path(sliced["gcode_3mf_path"]).exists())

        # Planar Z cut on BaseBox at z = 10mm (keeping both halves, flipping top onto cut face)
        cut_res = self._call_tool(
            "object_cut",
            {"object_index": 0, "cut_z_mm": 10.0, "keep": "both", "flip_top": True},
        )
        self.assertEqual(cut_res["created_count"], 2)
        self.assertEqual(len(cut_res["objects"]), 2)
        objs = self._call_tool("object_list")["objects"]
        self.assertEqual(len(objs), 2)
        for obj in objs:
            self.assertAlmostEqual(obj["instances"][0]["world_size_mm"][2], 10.0, delta=0.2)
            self.assertAlmostEqual(obj["instances"][0]["world_bbox_min_mm"][2], 0.0, delta=1e-3)

    def test_multi_plate_management(self) -> None:
        self._call_tool("project_new", {"project_path": str(self.tmp_dir / "plates_test.3mf")})
        self._call_tool(
            "object_add",
            {
                "items": [
                    {"primitive": "box", "params": {"size_x": 12.0, "size_y": 12.0, "size_z": 12.0}, "name": "ObjA"},
                    {"primitive": "box", "params": {"size_x": 14.0, "size_y": 14.0, "size_z": 14.0}, "name": "ObjB"},
                ],
                "auto_arrange": True,
            },
        )

        p2 = self._call_tool("plate_create", {"name": "Second Plate"})
        p2_id = [p for p in p2["plates"] if p["name"] == "Second Plate"][0]["plate_id"]
        self.assertGreaterEqual(p2_id, 2)

        self._call_tool(
            "plate_move_object",
            {"object_index": 1, "target_plate_id": p2_id, "arrange_target_plate": True},
        )
        self._call_tool(
            "plate_set_settings",
            {"plate_id": p2_id, "name": "Detailed Plate", "locked": True, "bed_type": "Textured PEI Plate"},
        )
        plist = self._call_tool("plate_list")
        self.assertEqual(len(plist["plates"]), 2)
        plate2 = [p for p in plist["plates"] if p["plate_id"] == p2_id][0]
        self.assertEqual(plate2["name"], "Detailed Plate")
        self.assertTrue(plate2["locked"])
        self.assertEqual(len(plate2["objects"]), 1)

        # Delete plate 2 -> object moves back to plate 1
        plist_after = self._call_tool("plate_delete", {"plate_id": p2_id, "move_objects_to_plate": 1})
        self.assertEqual(len(plist_after["plates"]), 1)
        self.assertEqual(len(plist_after["plates"][0]["objects"]), 2)

    def test_persistent_user_profiles(self) -> None:
        user_dir = self.tmp_dir / "user_presets"
        with mock.patch.dict(os.environ, {"ELEGOO_SLICER_USER_DIR": str(user_dir)}):
            saved = self._call_tool(
                "profile_save_user",
                {
                    "kind": "process",
                    "name": "MCP Custom Fast 0.24",
                    "overrides": {"Hauteur de couche": 0.24, "sparse_infill_density": "18%"},
                },
            )
            self.assertTrue(Path(saved["json_path"]).exists())
            self.assertTrue(Path(saved["info_path"]).exists())

            listed = self._call_tool("profiles_list", {"kind": "process", "query": "MCP Custom Fast"})
            names = [p["name"] for p in listed["print"]]
            self.assertIn("MCP Custom Fast 0.24", names)

            deleted = self._call_tool("profile_delete_user", {"kind": "process", "name": "MCP Custom Fast 0.24"})
            self.assertTrue(deleted["deleted"])
            self.assertFalse(Path(saved["json_path"]).exists())

    def test_gui_sync_and_printer_sdcp_dry_run(self) -> None:
        sync_stat = self._call_tool("gui_sync_status")
        self.assertIn("ipc_connected", sync_stat)
        self.assertIn("sync_mode", sync_stat)

        for tool_name, expected_cmd in (
            ("printer_pause", 129),
            ("printer_stop", 130),
            ("printer_resume", 131),
        ):
            res = self._call_tool(tool_name, {"printer_id": "cloud_test_01", "dry_run": True})
            self.assertTrue(res["dry_run"])
            self.assertEqual(res["printerId"], "cloud_test_01")
            self.assertEqual(res["envelope"]["Data"]["Cmd"], expected_cmd)

        rtm = self._call_tool(
            "printer_rtm_command",
            {"printer_id": "cloud_test_01", "cmd": 133, "data": {}, "dry_run": True},
        )
        self.assertTrue(rtm["dry_run"])
        self.assertEqual(rtm["envelope"]["Data"]["Cmd"], 133)

    def test_streamable_http_transport_and_auth(self) -> None:
        http_srv, stop_evt = self.server.create_http_server(host="127.0.0.1", port=0, auth_token="secret-tok")
        port = http_srv.server_address[1]
        thread = threading.Thread(target=http_srv.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{port}"
        try:
            # Health endpoint
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                health = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(health["status"], "ok")
                self.assertGreaterEqual(health["tools"], 45)

            # Unauthenticated POST /mcp -> 401
            req_unauth = urllib.request.Request(
                f"{base_url}/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req_unauth, timeout=5)
            self.assertEqual(ctx.exception.code, 401)
            ctx.exception.close()

            # Authenticated initialize -> Mcp-Session-Id
            auth_headers = {"Content-Type": "application/json", "Authorization": "Bearer secret-tok"}
            req_init = urllib.request.Request(
                f"{base_url}/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}).encode("utf-8"),
                headers=auth_headers,
                method="POST",
            )
            with urllib.request.urlopen(req_init, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                session_id = resp.headers.get("Mcp-Session-Id")
                self.assertTrue(session_id)

            # Session tools/list
            req_list = urllib.request.Request(
                f"{base_url}/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).encode("utf-8"),
                headers={**auth_headers, "Mcp-Session-Id": session_id},
                method="POST",
            )
            with urllib.request.urlopen(req_list, timeout=5) as resp:
                listed = json.loads(resp.read().decode("utf-8"))
                self.assertGreaterEqual(len(listed["result"]["tools"]), 45)

            # GET /mcp SSE open event
            req_sse = urllib.request.Request(
                f"{base_url}/mcp",
                headers={"Authorization": "Bearer secret-tok", "Mcp-Session-Id": session_id},
                method="GET",
            )
            with urllib.request.urlopen(req_sse, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                self.assertIn("text/event-stream", resp.headers.get("Content-Type", ""))
                first_chunk = b""
                for _ in range(4):
                    line = resp.readline()
                    first_chunk += line
                    if b"event: open" in first_chunk and line == b"\n":
                        break
                self.assertIn(b"event: open", first_chunk)

            # DELETE /mcp closes session
            req_del = urllib.request.Request(
                f"{base_url}/mcp",
                headers={"Authorization": "Bearer secret-tok", "Mcp-Session-Id": session_id},
                method="DELETE",
            )
            with urllib.request.urlopen(req_del, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                del_res = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(del_res["closed"])

            # Subsequent use of closed session -> 404
            req_dead = urllib.request.Request(
                f"{base_url}/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}).encode("utf-8"),
                headers={**auth_headers, "Mcp-Session-Id": session_id},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx2:
                urllib.request.urlopen(req_dead, timeout=5)
            self.assertEqual(ctx2.exception.code, 404)
            ctx2.exception.close()
        finally:
            stop_evt.set()
            http_srv.shutdown()
            http_srv.server_close()
            thread.join(timeout=3)

    def test_gui_cstyle_escape_and_ipc_safety(self) -> None:
        from mcp_server.gui_bridge import _cstyle_escape_args
        from mcp_server.ipc_client import IPCEndpoint

        # Exact match for Slic3r::escape_strings_cstyle / unescape_strings_cstyle (semicolon delimiter)
        escaped = _cstyle_escape_args([
            r"C:\Program Files\ElegooSlicer\elegoo-slicer.exe",
            r"C:\Users\Alex\OneDrive - a2-dev.fr\STL\10_Patins_TPU.3mf",
        ])
        self.assertEqual(
            escaped,
            r'"C:\\Program Files\\ElegooSlicer\\elegoo-slicer.exe";"C:\\Users\\Alex\\OneDrive - a2-dev.fr\\STL\\10_Patins_TPU.3mf"',
        )
        # Single empty string must be quoted; items with ';' must be quoted too
        self.assertEqual(_cstyle_escape_args([""]), '""')
        self.assertEqual(_cstyle_escape_args(["plain.exe", "a;b.3mf"]), 'plain.exe;"a;b.3mf"')

        # IPC safety: supports_slicer_namespace caches per endpoint; get_file_list sends pagination; get_selected_printer picks online printer without null call
        client = ElegooIPCClient()
        ep = IPCEndpoint(port=12345, auth_token="tok", port_file=self.tmp_dir / "ipc_port.txt")
        calls: list[tuple[str, dict]] = []

        def fake_call(method: str, params: dict | None = None, timeout: float | None = None):
            calls.append((method, params or {}))
            if method == "printer.getPrinterList":
                return [
                    {"printerId": "p1", "connectStatus": 0},
                    {"printerId": "p2", "connectStatus": 1},
                ]
            if method == "printer.getFileList":
                return {"list": []}
            if method == "slicer.ping":
                from mcp_server.ipc_client import IPCError
                raise IPCError("unknown namespace", code=-1)
            return {}

        with mock.patch("mcp_server.ipc_client.discover_ipc_endpoint", return_value=ep), mock.patch.object(
            client, "call", side_effect=fake_call
        ):
            self.assertFalse(client.supports_slicer_namespace())
            self.assertFalse(client.supports_slicer_namespace())
            self.assertEqual(sum(1 for m, _ in calls if m == "slicer.ping"), 1)

            self.assertEqual(client.get_selected_printer(), "p2")
            self.assertFalse(any(m == "printer.getSelectedPrinter" for m, _ in calls))

            client.get_file_list("p2")
            fl_params = [p for m, p in calls if m == "printer.getFileList"][0]
            self.assertEqual(fl_params, {"printerId": "p2", "pageNumber": 1, "pageSize": 50})

    def test_gui_dialog_classification_app_config_and_tools(self) -> None:
        import hashlib
        from mcp_server.gui_bridge import (
            _classify_dialog_kind,
            _match_dialog_button,
            get_elegoo_app_config,
            handle_elegoo_dialog,
            set_elegoo_app_config,
        )

        # Tool registration check
        tools_list = self.server.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tool_names = {t["name"] for t in tools_list["result"]["tools"]}
        for expected in (
            "gui_launch",
            "gui_open_project",
            "gui_list_windows",
            "gui_handle_dialog",
            "gui_control_window",
            "gui_app_config",
        ):
            self.assertIn(expected, tool_names)

        # ElegooSlicer.conf read/write preserves exact MD5 checksum footer
        conf_path = self.tmp_dir / "ElegooSlicer.conf"
        set_elegoo_app_config({"no_warn_when_modified_gcodes": True}, conf_path=conf_path)
        read_back = get_elegoo_app_config(keys=["no_warn_when_modified_gcodes"], conf_path=conf_path)
        self.assertTrue(read_back["exists"])
        self.assertTrue(read_back["app"]["no_warn_when_modified_gcodes"])
        raw = conf_path.read_bytes()
        marker = b"# MD5 checksum "
        idx = raw.rfind(marker)
        self.assertGreater(idx, 0)
        body_lf = raw[:idx].replace(b"\r\n", b"\n").rstrip(b"\n")
        expected_md5 = hashlib.md5(body_lf).hexdigest().upper()
        self.assertEqual(raw[idx + len(marker) :].decode("utf-8").strip(), expected_md5)

        # Dialog classification & button matching (French & English)
        self.assertEqual(
            _classify_dialog_kind(
                "Préréglage personnalisé",
                ["Le 3MF contient les préréglages..."],
                [{"hwnd": 11, "text": "OK"}],
            ),
            "preset_warning",
        )
        self.assertEqual(_classify_dialog_kind("Chargement...", [], []), "progress")
        matched = _match_dialog_button(
            [{"hwnd": 10, "text": "Annuler"}, {"hwnd": 11, "text": "OK"}],
            "ok",
        )
        self.assertIsNotNone(matched)
        self.assertEqual(matched["hwnd"], 11)
        matched_cancel = _match_dialog_button(
            [{"hwnd": 10, "text": "Annuler"}, {"hwnd": 11, "text": "OK"}],
            "cancel",
        )
        self.assertIsNotNone(matched_cancel)
        self.assertEqual(matched_cancel["hwnd"], 10)

        # Dialog selection prefers preset_warning over background progress dialog
        fake_windows = {
            "platform": "nt",
            "running": True,
            "pids": [9999],
            "main_hwnd": 100,
            "windows": [
                {
                    "hwnd": 100,
                    "pid": 9999,
                    "owner_hwnd": 0,
                    "title": "ElegooSlicer",
                    "class_name": "wxWindowNR",
                    "visible": True,
                    "enabled": True,
                    "minimized": False,
                    "maximized": False,
                    "foreground": False,
                    "is_main_window": True,
                    "is_dialog": False,
                    "dialog_kind": None,
                    "rect": {"x": 0, "y": 0, "width": 1200, "height": 800},
                    "buttons": [],
                    "checkboxes": [],
                    "texts": [],
                },
                {
                    "hwnd": 200,
                    "pid": 9999,
                    "owner_hwnd": 100,
                    "title": "Chargement...",
                    "class_name": "wxWindowNR",
                    "visible": True,
                    "enabled": True,
                    "minimized": False,
                    "maximized": False,
                    "foreground": False,
                    "is_main_window": False,
                    "is_dialog": True,
                    "dialog_kind": "progress",
                    "rect": {"x": 300, "y": 200, "width": 300, "height": 100},
                    "buttons": [],
                    "checkboxes": [],
                    "texts": [],
                },
                {
                    "hwnd": 300,
                    "pid": 9999,
                    "owner_hwnd": 100,
                    "title": "Préréglage personnalisé",
                    "class_name": "wxWindowNR",
                    "visible": True,
                    "enabled": True,
                    "minimized": False,
                    "maximized": False,
                    "foreground": True,
                    "is_main_window": False,
                    "is_dialog": True,
                    "dialog_kind": "preset_warning",
                    "rect": {"x": 200, "y": 220, "width": 500, "height": 220},
                    "buttons": [{"hwnd": 301, "class_name": "wxWindowNR", "text": "OK", "enabled": True}],
                    "checkboxes": [
                        {
                            "hwnd": 302,
                            "class_name": "Static",
                            "role": "text",
                            "text": "Ne plus afficher",
                            "is_dont_show_again": True,
                        }
                    ],
                    "texts": ["Le 3MF contient les préréglages..."],
                },
            ],
        }
        fake_windows["dialogs"] = [w for w in fake_windows["windows"] if w["is_dialog"]]
        clicked: list[int] = []
        if os.name == "nt":
            with (
                mock.patch("mcp_server.gui_bridge.list_elegoo_windows", return_value=fake_windows),
                mock.patch("mcp_server.gui_bridge._click_hwnd_nt", side_effect=lambda _u, h: clicked.append(h)),
                mock.patch("mcp_server.gui_bridge.set_elegoo_app_config", return_value={"updated": {"no_warn_when_modified_gcodes": True}}),
                mock.patch("ctypes.windll.user32.IsWindow", return_value=0),
                mock.patch("ctypes.windll.user32.IsWindowVisible", return_value=0),
            ):
                res = handle_elegoo_dialog(check_dont_show_again=True, button="OK")
                self.assertEqual(res["hwnd"], 300)
                self.assertTrue(res["dont_show_again_toggled"])
                self.assertEqual(res["button"]["hwnd"], 301)
                self.assertTrue(res["closed"])
                self.assertEqual(clicked, [302, 301])


if __name__ == "__main__":
    unittest.main()
