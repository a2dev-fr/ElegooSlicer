"""High-level ElegooSlicer project (.3mf), filament, face/rotation, slicing, export & print engine."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .config_registry import ConfigRegistry
from .gui_bridge import (
    control_elegoo_window,
    find_elegoo_slicer_executable,
    get_elegoo_app_config,
    handle_elegoo_dialog,
    launch_elegoo_gui,
    list_elegoo_windows,
    send_project_to_running_gui,
    set_elegoo_app_config,
)
from .ipc_client import ElegooIPCClient, IPCError
from .mesh_engine import (
    TriangleMeshData,
    arrange_bounding_boxes_2d,
    compute_lay_on_face_matrix,
    compose_transform,
    detect_printable_faces,
    euler_deg_to_rot3,
    load_3mf_combined_mesh,
    load_obj,
    load_stl,
    make_primitive,
    matrix_to_3mf_str,
    parse_3mf_model_meshes,
    rot3_to_euler_deg,
    save_binary_stl,
    split_mesh_by_z,
    str_to_matrix,
)
from .render_engine import render_object_faces, render_plate_preview


NS_3MF = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS_PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
NS_BBS = "http://schemas.bambulab.com/package/2021"
NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"

CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="png" ContentType="image/png"/>
 <Default Extension="gcode" ContentType="text/x.gcode"/>
 <Default Extension="config" ContentType="text/plain"/>
 <Default Extension="json" ContentType="application/json"/>
 <Default Extension="xml" ContentType="application/xml"/>
</Types>
"""

ROOT_RELS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""


def _xml_escape(val: str) -> str:
    return (
        val.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _format_parts_model_xml(parts: list[dict[str, Any]]) -> bytes:
    """Serialize one or more sub-part meshes into a `/3D/Objects/*.model` file."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<model unit="millimeter" xml:lang="en-US" xmlns="{NS_3MF}" '
            f'xmlns:BambuStudio="{NS_BBS}" xmlns:p="{NS_PROD}" requiredextensions="p">'
        ),
        ' <metadata name="BambuStudio:3mfVersion">1</metadata>',
        " <resources>",
    ]
    for part in parts:
        pid = str(part["part_id"])
        puuid = str(part.get("uuid") or uuid.uuid4())
        mesh: TriangleMeshData = part["mesh"]
        lines.extend(
            [
                f'  <object id="{pid}" p:UUID="{puuid}" type="model">',
                "   <mesh>",
                "    <vertices>",
            ]
        )
        for x, y, z in mesh.vertices:
            lines.append(f'     <vertex x="{x:.7g}" y="{y:.7g}" z="{z:.7g}"/>')
        lines.append("    </vertices>")
        lines.append("    <triangles>")
        for v1, v2, v3 in mesh.triangles:
            lines.append(f'     <triangle v1="{int(v1)}" v2="{int(v2)}" v3="{int(v3)}"/>')
        lines.extend(["    </triangles>", "   </mesh>", "  </object>"])
    lines.extend([" </resources>", " <build/>", "</model>"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _matrix4x4_rowmajor_str(m: np.ndarray) -> str:
    vals = []
    for r in range(4):
        for c in range(4):
            vals.append(f"{float(m[r, c]):.9g}")
    return " ".join(vals)


def _parse_rowmajor_4x4(s: str) -> np.ndarray:
    parts = [float(x) for x in s.strip().split()] if s else []
    if len(parts) != 16:
        return np.eye(4, dtype=np.float64)
    return np.array(parts, dtype=np.float64).reshape((4, 4))


def _decompose_matrix(m: np.ndarray) -> dict[str, Any]:
    """Extract offset_mm, scale, rotation_deg from a 4x4 affine matrix."""
    A = m[:3, :3]
    sx = float(np.linalg.norm(A[:, 0])) or 1.0
    sy = float(np.linalg.norm(A[:, 1])) or 1.0
    sz = float(np.linalg.norm(A[:, 2])) or 1.0
    if np.linalg.det(A) < 0:
        sx = -sx
    R = np.column_stack([A[:, 0] / sx, A[:, 1] / sy, A[:, 2] / sz])
    return {
        "offset_mm": [round(float(m[0, 3]), 3), round(float(m[1, 3]), 3), round(float(m[2, 3]), 3)],
        "rotation_deg": rot3_to_euler_deg(R),
        "scale": [round(sx, 5), round(sy, 5), round(sz, 5)],
    }


def _recompose_from_parts(offset_mm: list[float], rotation_deg: list[float], scale: list[float]) -> np.ndarray:
    R = euler_deg_to_rot3(float(rotation_deg[0]), float(rotation_deg[1]), float(rotation_deg[2]))
    return compose_transform(scale=scale, rot3=R, translation=offset_mm)


class ProjectEngine:
    """Stateful project & slicer controller backed by ElegooSlicer `.3mf`, CLI, IPC & ConfigRegistry."""

    def __init__(
        self,
        workspace_dir: Path | None = None,
        registry: ConfigRegistry | None = None,
        ipc_client: ElegooIPCClient | None = None,
    ) -> None:
        self.workspace_dir = (workspace_dir or self._default_workspace_dir()).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.workspace_dir / "mcp_state.json"
        self.registry = registry or ConfigRegistry()
        self.ipc = ipc_client or ElegooIPCClient()
        self._active_project: Path | None = self._load_active_project_path()

    @staticmethod
    def _default_workspace_dir() -> Path:
        override = os.environ.get("ELEGOO_SLICER_MCP_WORKSPACE")
        if override:
            return Path(override)
        return Path.home() / ".elegoo_slicer_mcp"

    def _load_active_project_path(self) -> Path | None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                p = data.get("active_project")
                if p and Path(p).exists():
                    return Path(p)
            except Exception:
                pass
        return None

    def _save_active_project_path(self, path: Path) -> None:
        self._active_project = path.resolve()
        self.state_file.write_text(
            json.dumps({"active_project": str(self._active_project)}, indent=2),
            encoding="utf-8",
        )

    def resolve_project_path(self, project_path: str | None = None, must_exist: bool = True) -> Path:
        if project_path:
            p = Path(project_path).expanduser().resolve()
        elif self._active_project is not None:
            p = self._active_project
        else:
            p = (self.workspace_dir / "current_project.3mf").resolve()
            if not p.exists() and not must_exist:
                return p
            if not p.exists() and must_exist:
                raise FileNotFoundError(
                    "No active project. Call project_new or project_open first (or pass project_path)."
                )
        if must_exist and not p.exists():
            raise FileNotFoundError(f"Project file not found: {p}")
        return p

    # ---- Low-level .3mf read/write helpers ----

    def _read_3mf_bundle(self, path: Path) -> dict[str, Any]:
        """Parse an existing ElegooSlicer .3mf archive into structured mutable state."""
        entries: dict[str, bytes] = {}
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                entries[name] = zf.read(name)

        project_settings: dict[str, Any] = {}
        if "Metadata/project_settings.config" in entries:
            try:
                project_settings = json.loads(entries["Metadata/project_settings.config"].decode("utf-8"))
            except Exception:
                project_settings = {}

        # Parse 3D/3dmodel.model objects & build items
        root_model = ET.fromstring(entries["3D/3dmodel.model"])
        raw_meshes_by_file: dict[str, dict[str, TriangleMeshData]] = {}
        for k, raw in entries.items():
            if k.lower().endswith(".model"):
                raw_meshes_by_file[f"/{k}" if not k.startswith("/") else k] = parse_3mf_model_meshes(raw)

        # Parse Metadata/model_settings.config if present
        obj_meta: dict[str, dict[str, Any]] = {}
        plate_info: dict[int, dict[str, Any]] = {}
        if "Metadata/model_settings.config" in entries:
            ms_root = ET.fromstring(entries["Metadata/model_settings.config"])
            for el in ms_root:
                if el.tag == "object":
                    oid = el.attrib.get("id", "")
                    info: dict[str, Any] = {
                        "name": "",
                        "extruder": 1,
                        "source_file": "",
                        "config": {},
                        "parts_meta": {},
                    }
                    for m in el.findall("metadata"):
                        k, v = m.attrib.get("key", ""), m.attrib.get("value", "")
                        if k == "name":
                            info["name"] = v
                        elif k == "extruder":
                            try:
                                info["extruder"] = int(v)
                            except ValueError:
                                pass
                        elif k:
                            info["config"][k] = v
                    for part in el.findall("part"):
                        p_id = part.attrib.get("id", "1")
                        subtype = part.attrib.get("subtype", "normal_part")
                        p_info: dict[str, Any] = {
                            "part_id": p_id,
                            "subtype": subtype,
                            "name": "",
                            "source_file": "",
                            "matrix": np.eye(4, dtype=np.float64),
                            "config": {},
                        }
                        for pm in part.findall("metadata"):
                            pk, pv = pm.attrib.get("key", ""), pm.attrib.get("value", "")
                            if pk == "name":
                                p_info["name"] = pv
                                if not info["name"]:
                                    info["name"] = pv
                            elif pk == "source_file":
                                p_info["source_file"] = pv
                                if not info["source_file"]:
                                    info["source_file"] = pv
                            elif pk == "matrix":
                                p_info["matrix"] = _parse_rowmajor_4x4(pv)
                            elif pk in {"source_object_id", "source_volume_id", "source_offset_x", "source_offset_y", "source_offset_z"}:
                                pass
                            elif pk:
                                p_info["config"][pk] = pv
                        info["parts_meta"][p_id] = p_info
                    obj_meta[oid] = info
                elif el.tag == "plate":
                    pid = 1
                    pmeta: dict[str, Any] = {
                        "plate_id": 1,
                        "name": "",
                        "locked": False,
                        "bed_type": "",
                        "filament_map_mode": "Auto For Flush",
                        "filament_maps": "1",
                        "instances": [],
                    }
                    for m in el.findall("metadata"):
                        mk, mv = m.attrib.get("key", ""), m.attrib.get("value", "")
                        if mk == "plater_id":
                            pid = int(mv or "1")
                            pmeta["plate_id"] = pid
                        elif mk == "plater_name":
                            pmeta["name"] = mv
                        elif mk == "locked":
                            pmeta["locked"] = mv.lower() == "true"
                        elif mk == "filament_map_mode":
                            pmeta["filament_map_mode"] = mv
                        elif mk == "filament_maps":
                            pmeta["filament_maps"] = mv
                    inst_pairs: list[tuple[str, int]] = []
                    for mi in el.findall("model_instance"):
                        mi_oid = ""
                        mi_iid = 0
                        for m in mi.findall("metadata"):
                            if m.attrib.get("key") == "object_id":
                                mi_oid = m.attrib.get("value", "")
                            elif m.attrib.get("key") == "instance_id":
                                mi_iid = int(m.attrib.get("value", "0"))
                        if mi_oid:
                            inst_pairs.append((mi_oid, mi_iid))
                    pmeta["instances"] = inst_pairs
                    plate_json_key = f"Metadata/plate_{pid}.json"
                    if plate_json_key in entries:
                        try:
                            pj = json.loads(entries[plate_json_key].decode("utf-8"))
                            pmeta["bed_type"] = pj.get("bed_type", "")
                        except Exception:
                            pass
                    plate_info[pid] = pmeta

        # Parse Metadata/layer_config_ranges.xml if present (indexed 1-based by object position)
        layer_ranges_by_idx: dict[int, list[dict[str, Any]]] = {}
        if "Metadata/layer_config_ranges.xml" in entries:
            try:
                lr_root = ET.fromstring(entries["Metadata/layer_config_ranges.xml"])
                for obj_el in lr_root.findall("object"):
                    idx_1b = int(obj_el.attrib.get("id", "0"))
                    rlist: list[dict[str, Any]] = []
                    for r_el in obj_el.findall("range"):
                        opts = {
                            opt_el.attrib.get("opt_key", ""): (opt_el.text or "")
                            for opt_el in r_el.findall("option")
                            if opt_el.attrib.get("opt_key")
                        }
                        rlist.append(
                            {
                                "min_z": float(r_el.attrib.get("min_z", "0")),
                                "max_z": float(r_el.attrib.get("max_z", "0")),
                                "options": opts,
                            }
                        )
                    if idx_1b > 0:
                        layer_ranges_by_idx[idx_1b] = rlist
            except Exception:
                pass

        # Map object_id -> parts + primary mesh + build instances
        objects_by_id: dict[str, dict[str, Any]] = {}
        inline_meshes = raw_meshes_by_file.get("/3D/3dmodel.model", {})
        resources_el = root_model.find(f"{{{NS_3MF}}}resources")
        if resources_el is None:
            resources_el = root_model.find("resources")
        if resources_el is not None:
            for obj_el in resources_el:
                if not (obj_el.tag.endswith("}object") or obj_el.tag == "object"):
                    continue
                oid = obj_el.attrib.get("id", "")
                meta = obj_meta.get(oid, {})
                parts_meta = meta.get("parts_meta", {})
                parts_list: list[dict[str, Any]] = []
                comp_path = ""

                if oid in inline_meshes:
                    pm = parts_meta.get("1", {})
                    parts_list.append(
                        {
                            "part_id": "1",
                            "subtype": pm.get("subtype", "normal_part"),
                            "name": pm.get("name") or meta.get("name") or f"Object_{oid}",
                            "source_file": pm.get("source_file") or meta.get("source_file", ""),
                            "matrix": pm.get("matrix", np.eye(4, dtype=np.float64)),
                            "config": dict(pm.get("config", {})),
                            "mesh": inline_meshes[oid],
                        }
                    )
                else:
                    for comp in obj_el.iter():
                        if comp.tag.endswith("}component") or comp.tag == "component":
                            p_attr = None
                            for ak, av in comp.attrib.items():
                                if ak.endswith("path") or ak == "path":
                                    p_attr = av
                                    break
                            sub_id = comp.attrib.get("objectid", "1")
                            if p_attr:
                                norm_p = p_attr if p_attr.startswith("/") else f"/{p_attr}"
                                comp_path = norm_p
                                sub_map = raw_meshes_by_file.get(norm_p, {})
                                sub_mesh = sub_map.get(sub_id) or (next(iter(sub_map.values()), None))
                                if sub_mesh is not None:
                                    pm = parts_meta.get(sub_id, {})
                                    comp_t = comp.attrib.get("transform", "")
                                    pmat = str_to_matrix(comp_t) if comp_t else pm.get("matrix", np.eye(4, dtype=np.float64))
                                    parts_list.append(
                                        {
                                            "part_id": sub_id,
                                            "subtype": pm.get("subtype", "normal_part"),
                                            "name": pm.get("name") or meta.get("name") or f"Part_{sub_id}",
                                            "source_file": pm.get("source_file") or meta.get("source_file", ""),
                                            "matrix": pmat,
                                            "config": dict(pm.get("config", {})),
                                            "mesh": sub_mesh,
                                        }
                                    )
                if not parts_list:
                    continue
                obj_idx_1b = len(objects_by_id) + 1
                objects_by_id[oid] = {
                    "id": oid,
                    "name": obj_el.attrib.get("name") or meta.get("name") or f"Object_{oid}",
                    "extruder": int(meta.get("extruder", 1)),
                    "source_file": meta.get("source_file", ""),
                    "component_path": comp_path,
                    "config": dict(meta.get("config", {})),
                    "layer_ranges": layer_ranges_by_idx.get(obj_idx_1b, []),
                    "parts": parts_list,
                    "mesh": parts_list[0]["mesh"],
                    "instances": [],
                }

        # Populate build items
        build_el = root_model.find(f"{{{NS_3MF}}}build")
        if build_el is None:
            build_el = root_model.find("build")
        if build_el is not None:
            for item_el in build_el:
                if not (item_el.tag.endswith("}item") or item_el.tag == "item"):
                    continue
                oid = item_el.attrib.get("objectid", "")
                if oid not in objects_by_id:
                    continue
                t_str = item_el.attrib.get("transform", "")
                printable = item_el.attrib.get("printable", "1") != "0"
                inst_idx = len(objects_by_id[oid]["instances"])
                assigned_plate = 1
                for pid, pdata in plate_info.items():
                    if (oid, inst_idx) in pdata["instances"]:
                        assigned_plate = pid
                        break
                objects_by_id[oid]["instances"].append(
                    {
                        "matrix": str_to_matrix(t_str),
                        "printable": printable,
                        "plate_id": assigned_plate,
                    }
                )

        if not plate_info:
            plate_info[1] = {
                "plate_id": 1,
                "name": "",
                "locked": False,
                "bed_type": "",
                "filament_map_mode": "Auto For Flush",
                "filament_maps": "1",
                "instances": [],
            }

        objects_list = list(objects_by_id.values())
        return {
            "entries": entries,
            "project_settings": project_settings,
            "objects": objects_list,
            "plates": plate_info,
        }

    def _write_3mf_bundle(self, path: Path, state: dict[str, Any]) -> Path:
        """Serialize structured project state back to a valid ElegooSlicer `.3mf` archive."""
        path.parent.mkdir(parents=True, exist_ok=True)
        entries: dict[str, bytes] = dict(state.get("entries", {}))
        objects: list[dict[str, Any]] = state.get("objects", [])
        project_settings: dict[str, Any] = state.get("project_settings", {})
        plates_meta: dict[int, dict[str, Any]] = state.get("plates") or {
            1: {"plate_id": 1, "name": "", "locked": False, "bed_type": ""}
        }

        # Remove stale generated object models & layer ranges; regenerate from state
        for k in [k for k in entries if k.startswith("3D/Objects/")]:
            del entries[k]
        entries.pop("Metadata/layer_config_ranges.xml", None)

        model_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<model unit="millimeter" xml:lang="en-US" xmlns="{NS_3MF}" '
                f'xmlns:BambuStudio="{NS_BBS}" xmlns:p="{NS_PROD}" requiredextensions="p">'
            ),
            ' <metadata name="Application">ElegooSlicer-1.5.3.5</metadata>',
            ' <metadata name="OrcaSlicer">2.4.2</metadata>',
            ' <metadata name="BambuStudio:3mfVersion">1</metadata>',
            " <resources>",
        ]

        build_lines = [" <build>"]
        ms_lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<config>"]
        plate_instances: dict[int, list[tuple[str, int]]] = {pid: [] for pid in plates_meta}
        rel_targets: list[str] = []
        lr_lines = ['<?xml version="1.0" encoding="utf-8"?>', "<objects>"]
        has_layer_ranges = False

        next_id = 2
        used_ids: set[str] = set()
        for idx, obj in enumerate(objects):
            oid = str(obj.get("id") or next_id)
            while oid in used_ids:
                oid = str(next_id)
                next_id += 1
            used_ids.add(oid)
            obj["id"] = oid
            next_id = max(next_id, (int(oid) + 1) if oid.isdigit() else next_id + 1)

            parts = obj.get("parts")
            if not parts:
                parts = [
                    {
                        "part_id": "1",
                        "subtype": "normal_part",
                        "name": obj.get("name") or f"object_{oid}",
                        "source_file": obj.get("source_file") or f"{obj.get('name', 'object')}.stl",
                        "matrix": np.eye(4, dtype=np.float64),
                        "config": {},
                        "mesh": obj["mesh"],
                    }
                ]
                obj["parts"] = parts
            else:
                # Keep primary mesh synced
                obj["mesh"] = parts[0]["mesh"]

            safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", obj.get("name") or f"object_{oid}")
            comp_rel = f"/3D/Objects/{safe_stem}_{oid}.model"
            comp_zip = comp_rel.lstrip("/")
            rel_targets.append(comp_rel)
            obj_uuid = str(uuid.uuid4())
            entries[comp_zip] = _format_parts_model_xml(parts)

            model_lines.extend(
                [
                    f'  <object id="{oid}" p:UUID="{obj_uuid}" type="model">',
                    "   <components>",
                ]
            )
            for part in parts:
                comp_uuid = str(uuid.uuid4())
                t_comp = matrix_to_3mf_str(part.get("matrix", np.eye(4, dtype=np.float64)))
                model_lines.append(
                    f'    <component p:path="{comp_rel}" objectid="{part["part_id"]}" '
                    f'p:UUID="{comp_uuid}" transform="{t_comp}"/>'
                )
            model_lines.extend(["   </components>", "  </object>"])

            instances = obj.get("instances") or [{"matrix": np.eye(4), "printable": True, "plate_id": 1}]
            for i_idx, inst in enumerate(instances):
                t_str = matrix_to_3mf_str(inst["matrix"])
                printable_attr = "1" if inst.get("printable", True) else "0"
                item_uuid = str(uuid.uuid4())
                build_lines.append(
                    f'  <item objectid="{oid}" p:UUID="{item_uuid}" transform="{t_str}" printable="{printable_attr}"/>'
                )
                pid = int(inst.get("plate_id", 1))
                plate_instances.setdefault(pid, []).append((oid, i_idx))

            extruder = int(obj.get("extruder", 1))
            ms_lines.extend(
                [
                    f'  <object id="{oid}">',
                    f'    <metadata key="name" value="{_xml_escape(obj.get("name", safe_stem))}"/>',
                    f'    <metadata key="extruder" value="{extruder}"/>',
                ]
            )
            for ck, cv in (obj.get("config") or {}).items():
                if ck not in {"name", "extruder"}:
                    ms_lines.append(f'    <metadata key="{_xml_escape(str(ck))}" value="{_xml_escape(str(cv))}"/>')

            for part in parts:
                pmat = part.get("matrix", np.eye(4, dtype=np.float64))
                src_file = part.get("source_file") or obj.get("source_file") or f"{safe_stem}.stl"
                ms_lines.extend(
                    [
                        f'    <part id="{part["part_id"]}" subtype="{_xml_escape(part.get("subtype", "normal_part"))}">',
                        f'      <metadata key="name" value="{_xml_escape(part.get("name") or obj.get("name", safe_stem))}"/>',
                        f'      <metadata key="matrix" value="{_matrix4x4_rowmajor_str(pmat)}"/>',
                        f'      <metadata key="source_file" value="{_xml_escape(src_file)}"/>',
                        '      <metadata key="source_object_id" value="0"/>',
                        '      <metadata key="source_volume_id" value="0"/>',
                    ]
                )
                for pk, pv in (part.get("config") or {}).items():
                    ms_lines.append(f'      <metadata key="{_xml_escape(str(pk))}" value="{_xml_escape(str(pv))}"/>')
                ms_lines.append("    </part>")
            ms_lines.append("  </object>")

            lr = obj.get("layer_ranges") or []
            if lr:
                has_layer_ranges = True
                lr_lines.append(f' <object id="{idx + 1}">')
                for ritem in lr:
                    lr_lines.append(f'  <range min_z="{ritem["min_z"]}" max_z="{ritem["max_z"]}">')
                    for ok, ov in (ritem.get("options") or {}).items():
                        lr_lines.append(f'   <option opt_key="{_xml_escape(str(ok))}">{_xml_escape(str(ov))}</option>')
                    lr_lines.append("  </range>")
                lr_lines.append(" </object>")

        if not plate_instances:
            plate_instances[1] = []

        filament_count = max(1, len(project_settings.get("filament_settings_id", [])) or 1)
        default_fmaps = " ".join(["1"] * filament_count)
        for pid in sorted(plate_instances):
            pm = plates_meta.get(pid, {})
            pname = pm.get("name", "")
            plocked = "true" if pm.get("locked") else "false"
            fmode = pm.get("filament_map_mode") or "Auto For Flush"
            fmaps = pm.get("filament_maps") or default_fmaps
            ms_lines.extend(
                [
                    "  <plate>",
                    f'    <metadata key="plater_id" value="{pid}"/>',
                    f'    <metadata key="plater_name" value="{_xml_escape(str(pname))}"/>',
                    f'    <metadata key="locked" value="{plocked}"/>',
                    f'    <metadata key="filament_map_mode" value="{_xml_escape(str(fmode))}"/>',
                    f'    <metadata key="filament_maps" value="{_xml_escape(str(fmaps))}"/>',
                ]
            )
            for oid, i_idx in plate_instances[pid]:
                ms_lines.extend(
                    [
                        "    <model_instance>",
                        f'      <metadata key="object_id" value="{oid}"/>',
                        f'      <metadata key="instance_id" value="{i_idx}"/>',
                        f'      <metadata key="identify_id" value="{int(oid) * 100 + i_idx if oid.isdigit() else 100 + i_idx}"/>',
                        "    </model_instance>",
                    ]
                )
            ms_lines.append("  </plate>")

        ms_lines.append("</config>")
        build_lines.append(" </build>")
        model_lines.append(" </resources>")
        model_lines.extend(build_lines)
        model_lines.append("</model>")

        if has_layer_ranges:
            lr_lines.append("</objects>")
            entries["Metadata/layer_config_ranges.xml"] = ("\n".join(lr_lines) + "\n").encode("utf-8")

        entries["[Content_Types].xml"] = CONTENT_TYPES_XML.encode("utf-8")
        entries["_rels/.rels"] = ROOT_RELS_XML.encode("utf-8")
        rel_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<Relationships xmlns="{NS_REL}">',
        ]
        for r_idx, tgt in enumerate(rel_targets, start=1):
            rel_lines.append(
                f' <Relationship Target="{_xml_escape(tgt)}" Id="rel-{r_idx}" '
                'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
            )
        rel_lines.append("</Relationships>")
        entries["3D/_rels/3dmodel.model.rels"] = ("\n".join(rel_lines) + "\n").encode("utf-8")
        entries["3D/3dmodel.model"] = ("\n".join(model_lines) + "\n").encode("utf-8")
        entries["Metadata/model_settings.config"] = ("\n".join(ms_lines) + "\n").encode("utf-8")
        normalized_ps = self._normalize_project_settings_types(project_settings)
        entries["Metadata/project_settings.config"] = json.dumps(
            normalized_ps, indent=4, ensure_ascii=False
        ).encode("utf-8")

        tmp_path = path.with_suffix(".3mf.tmp")
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for k, v in entries.items():
                zf.writestr(k, v)
        tmp_path.replace(path)
        self._save_active_project_path(path)
        return path

    # ---- Profile builders ----

    def _build_initial_project_settings(
        self,
        printer_profile: str | None = None,
        print_profile: str | None = None,
        filament_profiles: list[str] | None = None,
    ) -> dict[str, Any]:
        """Construct merged `Metadata/project_settings.config` from selected presets."""
        # Pick sensible Elegoo Centauri Carbon defaults if none specified
        available = self.registry.list_profiles("all", limit=200)

        def pick_default(kind: str, preferred_substrs: list[str]) -> str:
            names = [item["name"] for item in available.get(kind, [])]
            for sub in preferred_substrs:
                for n in names:
                    if sub.lower() in n.lower():
                        return n
            return names[0] if names else ""

        p_printer = printer_profile or pick_default(
            "printer", ["Elegoo Centauri Carbon 2 0.4", "Centauri Carbon 2", "Centauri Carbon 0.4", "Elegoo"]
        )
        p_print = print_profile or pick_default(
            "print", ["0.20mm Standard @Elegoo CC2", "0.20mm Standard @Elegoo", "0.20mm Standard"]
        )
        f_list = filament_profiles or [
            pick_default("filament", ["Elegoo PLA @ECC2", "Elegoo PLA", "Generic PLA"])
        ]

        merged: dict[str, Any] = {}
        for kind, pname in (("printer", p_printer), ("print", p_print)):
            if not pname:
                continue
            try:
                resolved = self.registry.resolve_profile(kind, pname)
                for k, v in resolved.items():
                    if not k.startswith("_") and k not in {"type", "name", "inherits", "instantiation", "from", "setting_id", "compatible_printers"}:
                        merged[k] = v
                if kind == "printer":
                    merged["printer_settings_id"] = resolved["name"]
                else:
                    merged["print_settings_id"] = resolved["name"]
            except KeyError:
                pass

        # Apply filament slots into per-slot vector parameters
        self._apply_filament_slots_to_settings(merged, f_list)
        return self._normalize_project_settings_types(merged)

    def _normalize_project_settings_types(self, ps: dict[str, Any]) -> dict[str, Any]:
        """Ensure every value matches C++ ConfigBase::load_from_json scalar-string vs vector-of-strings types."""
        self.registry.ensure_params_loaded()
        vector_types = {"coFloats", "coInts", "coBools", "coStrings", "coPercents", "coPoints", "coEnums"}
        scalar_only = {"version", "thumbnails", "printer_settings_id", "print_settings_id", "printer_model", "printer_variant"}

        out: dict[str, Any] = {}
        for k, v in ps.items():
            if k in {"filament_id", "description", "instantiation", "compatible_printers_condition"}:
                continue
            pdef = self.registry.params.get(k)
            is_vec = False
            if k in scalar_only:
                is_vec = False
            elif pdef is not None:
                is_vec = pdef.opt_type in vector_types
            elif isinstance(v, list):
                is_vec = True

            if is_vec:
                if isinstance(v, list):
                    out[k] = [str(x) for x in v]
                elif v is None:
                    out[k] = []
                else:
                    out[k] = [str(v)]
            else:
                if isinstance(v, list):
                    out[k] = ",".join(str(x) for x in v) if k == "thumbnails" else (str(v[0]) if v else "")
                elif v is None:
                    out[k] = ""
                else:
                    out[k] = str(v)

        if not out.get("version"):
            out["version"] = "1.5.3.5"
        return out

    def _apply_filament_slots_to_settings(
        self,
        project_settings: dict[str, Any],
        filament_names: list[str],
        slot_colors: list[str] | None = None,
    ) -> None:
        """Resolve each filament preset and populate parallel per-slot arrays inside `project_settings`."""
        resolved_slots: list[dict[str, Any]] = []
        for fname in filament_names:
            try:
                resolved_slots.append(self.registry.resolve_profile("filament", fname))
            except KeyError:
                resolved_slots.append({"name": fname, "filament_type": ["PLA"], "default_filament_colour": ["#FFFFFF"]})

        n = len(resolved_slots)
        all_fil_keys: set[str] = set()
        skip_keys = {
            "type", "name", "inherits", "instantiation", "from", "setting_id",
            "version", "filament_id", "description",
            "compatible_printers", "compatible_printers_condition", "compatible_prints",
        }
        for r in resolved_slots:
            for k in r:
                if not k.startswith("_") and k not in skip_keys:
                    all_fil_keys.add(k)

        for key in all_fil_keys:
            values: list[Any] = []
            for r in resolved_slots:
                val = r.get(key)
                if isinstance(val, list):
                    values.append(val[0] if val else "")
                elif val is not None:
                    values.append(val)
                else:
                    values.append("")
            project_settings[key] = values

        project_settings["filament_settings_id"] = [r.get("name", f"Slot_{i+1}") for i, r in enumerate(resolved_slots)]
        existing_cols = project_settings.get("filament_colour", [])
        colors_out: list[str] = []
        for idx, r in enumerate(resolved_slots):
            c = None
            if slot_colors and idx < len(slot_colors) and slot_colors[idx]:
                c = slot_colors[idx]
            elif idx < len(existing_cols) and existing_cols[idx]:
                c = existing_cols[idx]
            else:
                dc = r.get("default_filament_colour")
                c = dc[0] if isinstance(dc, list) and dc else (dc if isinstance(dc, str) else "#2850DF")
            colors_out.append(c)
        project_settings["filament_colour"] = colors_out
        if "filament_ids" not in project_settings or len(project_settings.get("filament_ids", [])) != n:
            project_settings["filament_ids"] = [str(r.get("filament_id", "GFL99") or "GFL99") for r in resolved_slots]

    def _bed_dimensions(self, project_settings: dict[str, Any]) -> tuple[float, float]:
        area = project_settings.get("printable_area")
        if isinstance(area, list) and area:
            xs = [float(str(pt).split("x")[0]) for pt in area if "x" in str(pt)]
            ys = [float(str(pt).split("x")[1]) for pt in area if "x" in str(pt)]
            if xs and ys:
                return max(xs) - min(xs), max(ys) - min(ys)
        return 256.0, 256.0

    # ---- Public high-level operations exposed to MCP tools ----

    def project_new(
        self,
        project_path: str | None = None,
        printer_profile: str | None = None,
        print_profile: str | None = None,
        filament_profiles: list[str] | None = None,
        sync_printer_filaments: bool = False,
        open_in_gui: bool = False,
    ) -> dict[str, Any]:
        target = Path(project_path).expanduser().resolve() if project_path else (self.workspace_dir / "current_project.3mf").resolve()
        if target.suffix.lower() != ".3mf":
            target = target.with_suffix(".3mf")
        ps = self._build_initial_project_settings(printer_profile, print_profile, filament_profiles)
        state = {"entries": {}, "project_settings": ps, "objects": []}
        self._write_3mf_bundle(target, state)

        synced = None
        if sync_printer_filaments:
            try:
                synced = self.filament_sync_from_printer(project_path=str(target))
            except Exception as exc:
                synced = {"warning": str(exc)}

        gui_res = None
        if open_in_gui:
            gui_res = send_project_to_running_gui(target)

        return {
            "project_path": str(target),
            "printer_profile": ps.get("printer_settings_id"),
            "print_profile": ps.get("print_settings_id"),
            "filament_profiles": ps.get("filament_settings_id", []),
            "synced_filaments": synced,
            "gui": gui_res,
        }

    def project_open(self, project_path: str, open_in_gui: bool = False) -> dict[str, Any]:
        target = Path(project_path).expanduser().resolve()
        if not target.exists():
            raise FileNotFoundError(f"Project not found: {target}")
        state = self._read_3mf_bundle(target)
        self._save_active_project_path(target)
        gui_res = send_project_to_running_gui(target) if open_in_gui else None
        return {
            "project_path": str(target),
            "objects_count": len(state["objects"]),
            "objects": self._summarize_objects(state),
            "printer_profile": state["project_settings"].get("printer_settings_id"),
            "print_profile": state["project_settings"].get("print_settings_id"),
            "filament_profiles": state["project_settings"].get("filament_settings_id", []),
            "gui": gui_res,
        }

    def project_save(self, destination_path: str | None = None, project_path: str | None = None) -> dict[str, Any]:
        src = self.resolve_project_path(project_path, must_exist=True)
        dest = Path(destination_path).expanduser().resolve() if destination_path else src
        if dest != src:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            self._save_active_project_path(dest)
        return {"project_path": str(dest)}

    def project_info(self, project_path: str | None = None) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        ps = state["project_settings"]
        bed_w, bed_d = self._bed_dimensions(ps)
        return {
            "project_path": str(target),
            "bed_size_mm": [bed_w, bed_d],
            "printer_profile": ps.get("printer_settings_id"),
            "print_profile": ps.get("print_settings_id"),
            "filament_profiles": ps.get("filament_settings_id", []),
            "filament_colours": ps.get("filament_colour", []),
            "filament_types": ps.get("filament_type", []),
            "objects": self._summarize_objects(state),
        }

    def open_in_gui(
        self,
        project_path: str | None = None,
        auto_confirm_preset_warning: bool = True,
        dont_show_again: bool = False,
    ) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        return send_project_to_running_gui(
            target,
            auto_confirm_preset_warning=auto_confirm_preset_warning,
            dont_show_again=dont_show_again,
        )

    def gui_launch(
        self,
        project_path: str | None = None,
        wait_seconds: float = 10.0,
        auto_confirm_preset_warning: bool = True,
        dont_show_again: bool = False,
    ) -> dict[str, Any]:
        """Launch ElegooSlicer desktop GUI (with optional project) and wait for its main window."""
        resolved: Path | None = None
        if project_path is not None:
            resolved = self.resolve_project_path(project_path, must_exist=True)
        return launch_elegoo_gui(
            project_path=resolved,
            wait_seconds=wait_seconds,
            auto_confirm_preset_warning=auto_confirm_preset_warning,
            dont_show_again=dont_show_again,
        )

    def gui_list_windows(
        self,
        include_invisible: bool = False,
        include_children: bool = True,
    ) -> dict[str, Any]:
        """List running ElegooSlicer windows, popups, modal dialogs, and child controls."""
        return list_elegoo_windows(
            include_invisible=include_invisible,
            include_children=include_children,
        )

    def gui_handle_dialog(
        self,
        hwnd: int | None = None,
        title_contains: str | None = None,
        action: str = "click_button",
        button: str = "OK",
        check_dont_show_again: bool = False,
        persist_no_preset_warning: bool = False,
    ) -> dict[str, Any]:
        """Interact with a modal ElegooSlicer dialog (e.g. click OK/Cancel, check Don't show again)."""
        return handle_elegoo_dialog(
            hwnd=hwnd,
            title_contains=title_contains,
            action=action,
            button=button,
            check_dont_show_again=check_dont_show_again,
            persist_no_preset_warning=persist_no_preset_warning,
        )

    def gui_control_window(
        self,
        action: str,
        hwnd: int | None = None,
        title_contains: str | None = None,
        key: str | None = None,
    ) -> dict[str, Any]:
        """Focus, minimize, maximize, restore, close, or send keys to an ElegooSlicer window."""
        return control_elegoo_window(
            hwnd=hwnd,
            action=action,
            title_contains=title_contains,
            key=key,
        )

    def gui_app_config(
        self,
        keys: list[str] | None = None,
        updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Read or update ElegooSlicer.conf `app` settings (e.g. `no_warn_when_modified_gcodes`)."""
        if updates:
            return set_elegoo_app_config(updates)
        return get_elegoo_app_config(keys=keys)

    # ---- Object management ----

    def _load_mesh_from_file(self, file_path: Path) -> TriangleMeshData:
        ext = file_path.suffix.lower()
        if ext == ".stl":
            return load_stl(file_path)
        if ext == ".obj":
            return load_obj(file_path)
        if ext == ".3mf":
            return load_3mf_combined_mesh(file_path)
        if ext in {".step", ".stp"}:
            # Convert STEP/STP via ElegooSlicer CLI
            slicer_bin = find_elegoo_slicer_executable()
            with tempfile.TemporaryDirectory(prefix="elegoo_step_") as tmpdir:
                out_dir = Path(tmpdir)
                out_stl = out_dir / f"{file_path.stem}.stl"
                subprocess.run(
                    [str(slicer_bin), "--export-stl", "--output", str(out_stl), str(file_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return load_stl(out_stl)
        raise ValueError(f"Unsupported 3D file extension: {ext}")

    @staticmethod
    def _summarize_objects(state: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx, obj in enumerate(state["objects"]):
            mesh: TriangleMeshData = obj["mesh"]
            raw_min, raw_max = mesh.bounds()
            inst_list: list[dict[str, Any]] = []
            for i_idx, inst in enumerate(obj["instances"]):
                bmin, bmax = mesh.bounds(inst["matrix"])
                dec = _decompose_matrix(inst["matrix"])
                inst_list.append(
                    {
                        "instance_index": i_idx,
                        "plate_id": inst.get("plate_id", 1),
                        "printable": inst.get("printable", True),
                        **dec,
                        "world_bbox_min_mm": [round(float(x), 3) for x in bmin],
                        "world_bbox_max_mm": [round(float(x), 3) for x in bmax],
                        "world_size_mm": [round(float(x), 3) for x in (bmax - bmin)],
                    }
                )
            out.append(
                {
                    "object_index": idx,
                    "object_id": obj["id"],
                    "name": obj["name"],
                    "filament_slot": obj["extruder"],
                    "triangle_count": mesh.triangle_count,
                    "parts_count": len(obj.get("parts") or [1]),
                    "config_overrides": obj.get("config") or {},
                    "layer_ranges_count": len(obj.get("layer_ranges") or []),
                    "raw_size_mm": [round(float(x), 3) for x in (raw_max - raw_min)],
                    "instances": inst_list,
                }
            )
        return out

    def add_objects(
        self,
        items: list[dict[str, Any]],
        project_path: str | None = None,
        auto_arrange: bool = True,
        spacing_mm: float = 6.0,
        plate_id: int = 1,
    ) -> dict[str, Any]:
        """Add one or multiple file or primitive objects, seating them at Z=0 on the build plate."""
        try:
            target = self.resolve_project_path(project_path, must_exist=True)
            state = self._read_3mf_bundle(target)
        except FileNotFoundError:
            created = self.project_new(project_path=project_path)
            target = Path(created["project_path"])
            state = self._read_3mf_bundle(target)

        bed_w, bed_d = self._bed_dimensions(state["project_settings"])
        added_indices: list[int] = []

        for spec in items:
            if "primitive" in spec and spec["primitive"]:
                mesh = make_primitive(spec["primitive"], spec.get("params"))
                name = spec.get("name") or f"{spec['primitive']}_{len(state['objects']) + 1}"
                source_file = f"{name}.stl"
            elif "path" in spec and spec["path"]:
                fpath = Path(spec["path"]).expanduser().resolve()
                if not fpath.exists():
                    raise FileNotFoundError(f"Object file not found: {fpath}")
                mesh = self._load_mesh_from_file(fpath)
                name = spec.get("name") or fpath.stem
                source_file = fpath.name
            else:
                raise ValueError("Each item must provide 'path' or 'primitive'")

            # Center mesh locally on XY origin and seat Z=0 so transforms behave predictably
            bmin, bmax = mesh.bounds()
            center_xy = 0.5 * (bmin[:2] + bmax[:2])
            centered_verts = mesh.vertices.copy()
            centered_verts[:, 0] -= center_xy[0]
            centered_verts[:, 1] -= center_xy[1]
            centered_verts[:, 2] -= bmin[2]
            mesh = TriangleMeshData(centered_verts, mesh.triangles)

            rot_deg = spec.get("rotation_deg") or [0.0, 0.0, 0.0]
            scale = spec.get("scale", 1.0)
            if isinstance(scale, (int, float)):
                scale_vec = [float(scale)] * 3
            else:
                scale_vec = [float(scale[0]), float(scale[1]), float(scale[2])]

            offset_mm = spec.get("offset_mm") or [bed_w * 0.5, bed_d * 0.5, 0.0]
            m = _recompose_from_parts([0.0, 0.0, 0.0], rot_deg, scale_vec)
            # Seat rotated bottom at z=0
            rmin, rmax = mesh.bounds(m)
            m[0, 3] = float(offset_mm[0])
            m[1, 3] = float(offset_mm[1])
            m[2, 3] = float(offset_mm[2] - rmin[2])

            copies = max(1, int(spec.get("copies", 1)))
            instances = []
            for _ in range(copies):
                instances.append(
                    {
                        "matrix": m.copy(),
                        "printable": True,
                        "plate_id": int(spec.get("plate_id", plate_id)),
                    }
                )

            extruder = int(spec.get("filament_slot", 1))
            idx = len(state["objects"])
            state["objects"].append(
                {
                    "id": "",
                    "name": name,
                    "extruder": extruder,
                    "source_file": source_file,
                    "mesh": mesh,
                    "instances": instances,
                }
            )
            added_indices.append(idx)

        if auto_arrange:
            self._arrange_state_in_place(state, plate_id=plate_id, spacing_mm=spacing_mm)

        self._write_3mf_bundle(target, state)
        return {
            "project_path": str(target),
            "added_indices": added_indices,
            "objects": self._summarize_objects(state),
        }

    def remove_object(
        self,
        object_index: int | None = None,
        clear_all: bool = False,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        if clear_all:
            state["objects"] = []
        else:
            if object_index is None or object_index < 0 or object_index >= len(state["objects"]):
                raise IndexError(f"Invalid object_index: {object_index}")
            state["objects"].pop(object_index)
        self._write_3mf_bundle(target, state)
        return {"project_path": str(target), "objects_count": len(state["objects"])}

    def _get_object_and_instance(
        self,
        state: dict[str, Any],
        object_index: int,
        instance_index: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if object_index < 0 or object_index >= len(state["objects"]):
            raise IndexError(f"Invalid object_index {object_index} (count={len(state['objects'])})")
        obj = state["objects"][object_index]
        if instance_index < 0 or instance_index >= len(obj["instances"]):
            raise IndexError(f"Invalid instance_index {instance_index}")
        return obj, obj["instances"][instance_index]

    def object_list_faces(
        self,
        object_index: int = 0,
        instance_index: int = 0,
        min_area_mm2: float = 5.0,
        top_n: int = 20,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        obj, inst = self._get_object_and_instance(state, object_index, instance_index)
        faces = detect_printable_faces(
            obj["mesh"],
            current_matrix=inst["matrix"],
            min_area_mm2=min_area_mm2,
            top_n=top_n,
        )
        return {
            "project_path": str(target),
            "object_index": object_index,
            "object_name": obj["name"],
            "faces": faces,
        }

    def object_lay_on_face(
        self,
        object_index: int = 0,
        instance_index: int = 0,
        face_id: int | None = None,
        normal: list[float] | None = None,
        min_area_mm2: float = 5.0,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        obj, inst = self._get_object_and_instance(state, object_index, instance_index)

        chosen_face = None
        if normal is not None:
            target_normal = normal
        else:
            faces = detect_printable_faces(
                obj["mesh"],
                current_matrix=inst["matrix"],
                min_area_mm2=min_area_mm2,
                top_n=50,
            )
            if not faces:
                raise RuntimeError("No planar printable faces >= min_area_mm2 found on this object.")
            fid = 0 if face_id is None else face_id
            if fid < 0 or fid >= len(faces):
                raise IndexError(f"Invalid face_id {fid} (available 0..{len(faces) - 1})")
            chosen_face = faces[fid]
            target_normal = chosen_face["normal"]

        inst["matrix"] = compute_lay_on_face_matrix(obj["mesh"], inst["matrix"], target_normal, bed_z=0.0)
        self._write_3mf_bundle(target, state)
        return {
            "project_path": str(target),
            "object_index": object_index,
            "applied_face": chosen_face,
            "applied_normal": target_normal,
            "object": self._summarize_objects(state)[object_index],
        }

    def object_auto_orient(
        self,
        object_index: int | None = None,
        min_area_mm2: float = 5.0,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        """Auto-orient object(s) onto the exterior face minimizing overhang and maximizing bed contact."""
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        indices = [object_index] if object_index is not None else list(range(len(state["objects"])))
        results = []
        for idx in indices:
            obj, inst = self._get_object_and_instance(state, idx, 0)
            faces = detect_printable_faces(
                obj["mesh"],
                current_matrix=inst["matrix"],
                min_area_mm2=min_area_mm2,
                top_n=50,
            )
            if not faces:
                continue
            # Rank: exterior hull > low overhang > large contact area
            best = min(
                faces,
                key=lambda f: (
                    0 if f["on_exterior_hull"] else 1,
                    round(f["overhang_ratio"], 2),
                    -f["area_mm2"],
                ),
            )
            inst["matrix"] = compute_lay_on_face_matrix(obj["mesh"], inst["matrix"], best["normal"], bed_z=0.0)
            results.append({"object_index": idx, "chosen_face": best})
        self._write_3mf_bundle(target, state)
        return {
            "project_path": str(target),
            "oriented": results,
            "objects": self._summarize_objects(state),
        }

    def object_rotate(
        self,
        object_index: int = 0,
        instance_index: int = 0,
        rotation_deg: list[float] | None = None,
        axis: str | None = None,
        angle_deg: float | None = None,
        relative: bool = True,
        drop_to_bed: bool = True,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        obj, inst = self._get_object_and_instance(state, object_index, instance_index)

        if rotation_deg is not None:
            rvec = [float(rotation_deg[0]), float(rotation_deg[1]), float(rotation_deg[2])]
        elif axis is not None and angle_deg is not None:
            ax = axis.strip().upper()
            rvec = [
                float(angle_deg) if ax == "X" else 0.0,
                float(angle_deg) if ax == "Y" else 0.0,
                float(angle_deg) if ax == "Z" else 0.0,
            ]
        else:
            raise ValueError("Provide either rotation_deg=[rx, ry, rz] or axis + angle_deg")

        dec = _decompose_matrix(inst["matrix"])
        scale = dec["scale"]
        old_min, old_max = obj["mesh"].bounds(inst["matrix"])
        old_center_xy = 0.5 * (old_min[:2] + old_max[:2])

        R_new = euler_deg_to_rot3(*rvec)
        if relative:
            # Extract current pure rotation
            A = inst["matrix"][:3, :3]
            sx, sy, sz = scale
            R_cur = np.column_stack([A[:, 0] / sx, A[:, 1] / sy, A[:, 2] / sz])
            R_final = R_new @ R_cur
        else:
            R_final = R_new

        m = compose_transform(scale=scale, rot3=R_final, translation=(0.0, 0.0, 0.0))
        new_min, new_max = obj["mesh"].bounds(m)
        new_center_xy = 0.5 * (new_min[:2] + new_max[:2])
        m[0, 3] = float(old_center_xy[0] - new_center_xy[0])
        m[1, 3] = float(old_center_xy[1] - new_center_xy[1])
        m[2, 3] = float(0.0 - new_min[2]) if drop_to_bed else float(inst["matrix"][2, 3])
        inst["matrix"] = m

        self._write_3mf_bundle(target, state)
        return {
            "project_path": str(target),
            "object": self._summarize_objects(state)[object_index],
        }

    def object_transform(
        self,
        object_index: int = 0,
        instance_index: int = 0,
        offset_mm: list[float] | None = None,
        rotation_deg: list[float] | None = None,
        scale: list[float] | float | None = None,
        mirror: list[bool] | None = None,
        drop_to_bed: bool = True,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        obj, inst = self._get_object_and_instance(state, object_index, instance_index)
        dec = _decompose_matrix(inst["matrix"])

        cur_offset = list(dec["offset_mm"])
        cur_rot = list(dec["rotation_deg"])
        cur_scale = list(dec["scale"])

        if rotation_deg is not None:
            cur_rot = [float(x) for x in rotation_deg]
        if scale is not None:
            if isinstance(scale, (int, float)):
                cur_scale = [float(scale)] * 3
            else:
                cur_scale = [float(scale[0]), float(scale[1]), float(scale[2])]
        if mirror is not None:
            for i in range(3):
                if mirror[i]:
                    cur_scale[i] = -abs(cur_scale[i])
                else:
                    cur_scale[i] = abs(cur_scale[i])
        if offset_mm is not None:
            cur_offset = [float(offset_mm[0]), float(offset_mm[1]), float(offset_mm[2])]

        m = _recompose_from_parts([0.0, 0.0, 0.0], cur_rot, cur_scale)
        bmin, _ = obj["mesh"].bounds(m)
        m[0, 3] = cur_offset[0]
        m[1, 3] = cur_offset[1]
        m[2, 3] = float(-bmin[2]) if drop_to_bed else cur_offset[2]
        inst["matrix"] = m

        self._write_3mf_bundle(target, state)
        return {
            "project_path": str(target),
            "object": self._summarize_objects(state)[object_index],
        }

    def _arrange_state_in_place(self, state: dict[str, Any], plate_id: int = 1, spacing_mm: float = 6.0) -> None:
        bed_w, bed_d = self._bed_dimensions(state["project_settings"])
        targets: list[tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]] = []
        sizes: list[tuple[float, float]] = []
        for obj in state["objects"]:
            for inst in obj["instances"]:
                if int(inst.get("plate_id", 1)) != plate_id:
                    continue
                bmin, bmax = obj["mesh"].bounds(inst["matrix"])
                w = max(1.0, float(bmax[0] - bmin[0]))
                d = max(1.0, float(bmax[1] - bmin[1]))
                targets.append((obj, inst, bmin, bmax))
                sizes.append((w, d))
        centers = arrange_bounding_boxes_2d(sizes, bed_width=bed_w, bed_depth=bed_d, spacing=spacing_mm)
        for (obj, inst, bmin, bmax), (cx, cy) in zip(targets, centers):
            cur_cx = 0.5 * (bmin[0] + bmax[0])
            cur_cy = 0.5 * (bmin[1] + bmax[1])
            inst["matrix"][0, 3] += float(cx - cur_cx)
            inst["matrix"][1, 3] += float(cy - cur_cy)
            inst["matrix"][2, 3] -= float(bmin[2])  # Ensure Z=0

    def object_arrange(self, plate_id: int = 1, spacing_mm: float = 6.0, project_path: str | None = None) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        self._arrange_state_in_place(state, plate_id=plate_id, spacing_mm=spacing_mm)
        self._write_3mf_bundle(target, state)
        return {"project_path": str(target), "objects": self._summarize_objects(state)}

    # ---- Filament control ----

    def filament_list_slots(self, project_path: str | None = None) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        ps = state["project_settings"]
        names = ps.get("filament_settings_id", [])
        colors = ps.get("filament_colour", [])
        types = ps.get("filament_type", [])
        nozzle_temps = ps.get("nozzle_temperature", [])
        bed_temps = ps.get("textured_plate_temp", [])
        slots = []
        for i in range(max(1, len(names))):
            slots.append(
                {
                    "slot": i + 1,
                    "preset": names[i] if i < len(names) else "",
                    "color": colors[i] if i < len(colors) else "",
                    "type": types[i] if i < len(types) else "",
                    "nozzle_temperature": nozzle_temps[i] if i < len(nozzle_temps) else "",
                    "textured_plate_temp": bed_temps[i] if i < len(bed_temps) else "",
                }
            )
        return {"project_path": str(target), "slots": slots}

    def filament_set_slots(
        self,
        slots: list[dict[str, Any]],
        project_path: str | None = None,
    ) -> dict[str, Any]:
        """Set presets, colors, types, or overrides on filament slots 1..N."""
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        ps = state["project_settings"]

        cur_presets = list(ps.get("filament_settings_id", ["Elegoo PLA @ECC2"]))
        cur_colors = list(ps.get("filament_colour", ["#2850DF"] * len(cur_presets)))
        max_slot = max([len(cur_presets)] + [int(s.get("slot", idx + 1)) for idx, s in enumerate(slots)])

        while len(cur_presets) < max_slot:
            cur_presets.append(cur_presets[0] if cur_presets else "Elegoo PLA @ECC2")
        while len(cur_colors) < max_slot:
            cur_colors.append("#FFFFFF")

        per_slot_overrides: dict[int, dict[str, Any]] = {}
        for idx, s in enumerate(slots):
            slot_num = int(s.get("slot", idx + 1))
            if slot_num < 1:
                raise ValueError("slot must be >= 1")
            s_idx = slot_num - 1
            if "preset" in s and s["preset"]:
                cur_presets[s_idx] = s["preset"]
            if "color" in s and s["color"]:
                cur_colors[s_idx] = s["color"]
            if "overrides" in s and isinstance(s["overrides"], dict):
                per_slot_overrides[s_idx] = s["overrides"]

        self._apply_filament_slots_to_settings(ps, cur_presets, cur_colors)
        for s_idx, ov in per_slot_overrides.items():
            for raw_k, raw_v in ov.items():
                key = self.registry.resolve_key(raw_k)
                norm_v = self.registry.normalize_value(key, raw_v)
                arr = ps.get(key)
                if isinstance(arr, list):
                    while len(arr) <= s_idx:
                        arr.append(arr[0] if arr else "")
                    arr[s_idx] = str(norm_v)
                else:
                    arr_new = [str(arr) if arr is not None else "" for _ in range(max_slot)]
                    arr_new[s_idx] = str(norm_v)
                    ps[key] = arr_new

        self._write_3mf_bundle(target, state)
        return self.filament_list_slots(project_path=str(target))

    def filament_assign_to_object(
        self,
        object_index: int = 0,
        slot: int = 1,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        obj, _ = self._get_object_and_instance(state, object_index, 0)
        if slot < 1:
            raise ValueError("slot must be >= 1")
        obj["extruder"] = int(slot)
        self._write_3mf_bundle(target, state)
        return {
            "project_path": str(target),
            "object_index": object_index,
            "object_name": obj["name"],
            "filament_slot": int(slot),
        }

    def filament_sync_from_printer(
        self,
        printer_id: str | None = None,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        """Query live printer CANVAS/MMS trays over IPC and synchronize project filament slots."""
        printers = self.ipc.get_printer_list()
        if not printers:
            raise IPCError("No printers registered in running ElegooSlicer.")

        pid = printer_id
        if not pid:
            selected = self.ipc.get_selected_printer()
            if selected:
                pid = selected
            else:
                online = [p for p in printers if p.get("connectStatus") == 1]
                pid = (online[0] if online else printers[0]).get("printerId")

        mms = self.ipc.get_printer_mms_info(pid)
        canvas_list = mms.get("canvasList") or []
        available_presets = [x["name"] for x in self.registry.list_profiles("filament", limit=500).get("filament", [])]

        def match_filament_preset(ftype: str) -> str:
            t_up = (ftype or "PLA").upper()
            preferred = [
                f"Elegoo {t_up} @ECC2",
                f"Elegoo Rapid {t_up} @ECC2",
                f"Generic {t_up}",
                t_up,
            ]
            for cand in preferred:
                for p in available_presets:
                    if cand.lower() in p.lower():
                        return p
            return available_presets[0] if available_presets else "Elegoo PLA @ECC2"

        synced_slots: list[dict[str, Any]] = []
        for canvas in canvas_list:
            for tray in canvas.get("trayList") or []:
                tray_id = int(tray.get("trayId", len(synced_slots)))
                ftype = tray.get("filamentType") or "PLA"
                fcolor = tray.get("filamentColor") or "#FFFFFF"
                synced_slots.append(
                    {
                        "slot": tray_id + 1,
                        "preset": match_filament_preset(ftype),
                        "color": fcolor,
                        "tray_status": tray.get("status"),
                        "brand": tray.get("brand"),
                        "filament_type": ftype,
                    }
                )

        if synced_slots and (project_path or self._active_project):
            target = self.resolve_project_path(project_path, must_exist=True)
            self.filament_set_slots(synced_slots, project_path=str(target))

        return {
            "printer_id": pid,
            "mms_connected": mms.get("mmsStatus", {}).get("connected"),
            "synced_slots": synced_slots,
        }

    # ---- Parameters & Profiles ----

    def params_get(
        self,
        keys: list[str] | None = None,
        category: str | None = None,
        query: str | None = None,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        ps = state["project_settings"]
        self.registry.ensure_params_loaded()

        values: list[dict[str, Any]] = []
        if keys:
            for raw_k in keys:
                key = self.registry.resolve_key(raw_k)
                pdef = self.registry.params.get(key)
                values.append(
                    {
                        "key": key,
                        "label_fr": pdef.label_fr if pdef else "",
                        "label_en": pdef.label_en if pdef else "",
                        "category_fr": pdef.category_fr if pdef else "",
                        "value": ps.get(key, pdef.default_raw if pdef else None),
                        "unit_fr": pdef.sidetext_fr if pdef else "",
                    }
                )
        elif query or category:
            matched = self.registry.search_params(query=query or "", category=category or "", limit=100)
            for item in matched:
                k = item["key"]
                values.append(
                    {
                        "key": k,
                        "label_fr": item["label_fr"],
                        "label_en": item["label_en"],
                        "category_fr": item["category_fr"],
                        "value": ps.get(k, item["default"]),
                        "unit_fr": item["unit_fr"],
                    }
                )
        else:
            for k in sorted(ps.keys()):
                pdef = self.registry.params.get(k)
                values.append(
                    {
                        "key": k,
                        "label_fr": pdef.label_fr if pdef else "",
                        "value": ps[k],
                    }
                )
        return {"project_path": str(target), "count": len(values), "parameters": values}

    def params_set(
        self,
        parameters: dict[str, Any],
        project_path: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        ps = state["project_settings"]
        applied: dict[str, Any] = {}

        for raw_k, raw_v in parameters.items():
            key = self.registry.resolve_key(raw_k)
            val = self.registry.normalize_value(key, raw_v)
            pdef = self.registry.params.get(key)
            # If project already stores list (e.g., per-extruder arrays), update consistently
            if isinstance(ps.get(key), list) and not isinstance(val, list):
                n = max(1, len(ps[key]))
                ps[key] = [str(val)] * n
            else:
                ps[key] = val
            applied[key] = {
                "requested_key": raw_k,
                "label_fr": pdef.label_fr if pdef else "",
                "value": ps[key],
            }

        self._write_3mf_bundle(target, state)
        return {"project_path": str(target), "applied": applied}

    def profile_select(
        self,
        print_profile: str | None = None,
        printer_profile: str | None = None,
        filament_profiles: list[str] | None = None,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        ps = state["project_settings"]

        if printer_profile:
            resolved = self.registry.resolve_profile("printer", printer_profile)
            for k, v in resolved.items():
                if not k.startswith("_") and k not in {"type", "name", "inherits", "instantiation", "from", "setting_id"}:
                    ps[k] = v
            ps["printer_settings_id"] = resolved["name"]

        if print_profile:
            resolved = self.registry.resolve_profile("print", print_profile)
            for k, v in resolved.items():
                if not k.startswith("_") and k not in {"type", "name", "inherits", "instantiation", "from", "setting_id"}:
                    ps[k] = v
            ps["print_settings_id"] = resolved["name"]

        if filament_profiles:
            self._apply_filament_slots_to_settings(ps, filament_profiles)

        self._write_3mf_bundle(target, state)
        return {
            "project_path": str(target),
            "printer_profile": ps.get("printer_settings_id"),
            "print_profile": ps.get("print_settings_id"),
            "filament_profiles": ps.get("filament_settings_id", []),
        }

    # ---- Slicing & Exporting ----

    @staticmethod
    def _parse_gcode_metadata(gcode_text: str) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        patterns = {
            "estimated_time": r";\s*estimated printing time \(normal mode\)\s*=\s*(.+)",
            "filament_used_g": r";\s*total filament used \[g\]\s*=\s*(.+)",
            "filament_used_mm": r";\s*filament used \[mm\]\s*=\s*(.+)",
            "total_layers": r";\s*total layer number:\s*(\d+)",
        }
        for k, pat in patterns.items():
            m = re.search(pat, gcode_text)
            if m:
                stats[k] = m.group(1).strip()
        return stats

    def slice_project(
        self,
        plate: int = 1,
        output_dir: str | None = None,
        export_3mf_name: str | None = None,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        """Run ElegooSlicer CLI `--slice` headless and collect G-code / `.gcode.3mf` metrics."""
        target = self.resolve_project_path(project_path, must_exist=True)
        slicer_bin = find_elegoo_slicer_executable()

        out_path = Path(output_dir).expanduser().resolve() if output_dir else (self.workspace_dir / "slices" / target.stem)
        out_path.mkdir(parents=True, exist_ok=True)
        log_path = out_path / "slice.log"
        gcode_3mf_name = export_3mf_name or f"{target.stem}_plate_{plate}.gcode.3mf"

        cmd = [
            str(slicer_bin),
            "--slice",
            str(plate),
            "--outputdir",
            str(out_path),
            "--export-3mf",
            gcode_3mf_name,
            "--debug",
            "3",
            "--logfile",
            str(log_path),
            str(target),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        gcode_files = sorted(out_path.glob("*.gcode"))
        gcode_3mf_path = out_path / gcode_3mf_name
        stats: dict[str, Any] = {}
        plate_gcode_file = None

        if gcode_files:
            plate_gcode_file = gcode_files[0]
            stats = self._parse_gcode_metadata(
                plate_gcode_file.read_text(encoding="utf-8", errors="replace")[:200000]
            )
            # Some stats are at tail of gcode
            tail = plate_gcode_file.read_bytes()[-65536:].decode("utf-8", errors="replace")
            stats.update(self._parse_gcode_metadata(tail))
        elif gcode_3mf_path.exists():
            with zipfile.ZipFile(gcode_3mf_path, "r") as zf:
                for n in zf.namelist():
                    if n.endswith(".gcode"):
                        raw = zf.read(n)
                        head = raw[:131072].decode("utf-8", errors="replace")
                        tail = raw[-65536:].decode("utf-8", errors="replace")
                        stats.update(self._parse_gcode_metadata(head))
                        stats.update(self._parse_gcode_metadata(tail))
                        break

        result_json_path = out_path / "result.json"
        result_json = None
        if result_json_path.exists():
            try:
                result_json = json.loads(result_json_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        return {
            "ok": proc.returncode == 0 and (gcode_3mf_path.exists() or bool(gcode_files)),
            "return_code": proc.returncode,
            "project_path": str(target),
            "output_dir": str(out_path),
            "gcode_3mf_path": str(gcode_3mf_path) if gcode_3mf_path.exists() else None,
            "gcode_files": [str(g) for g in gcode_files],
            "log_file": str(log_path),
            "stats": stats,
            "result_json": result_json,
        }

    def export_output(
        self,
        export_format: str,
        destination_path: str,
        plate: int = 1,
        object_index: int = 0,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        """Export project or object to `.gcode`, `.gcode.3mf`, `.3mf`, or `.stl`."""
        fmt = export_format.lower().lstrip(".")
        dest = Path(destination_path).expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        target = self.resolve_project_path(project_path, must_exist=True)

        if fmt == "3mf":
            shutil.copy2(target, dest)
            return {"format": "3mf", "output_path": str(dest)}

        if fmt == "stl":
            state = self._read_3mf_bundle(target)
            obj, inst = self._get_object_and_instance(state, object_index, 0)
            save_binary_stl(obj["mesh"], dest, matrix_4x4=inst["matrix"])
            return {"format": "stl", "output_path": str(dest)}

        if fmt in {"gcode", "gcode.3mf"}:
            with tempfile.TemporaryDirectory(prefix="elegoo_export_") as tmpdir:
                res = self.slice_project(
                    plate=plate,
                    output_dir=tmpdir,
                    export_3mf_name=dest.name if fmt == "gcode.3mf" else "export.gcode.3mf",
                    project_path=str(target),
                )
                if not res["ok"]:
                    raise RuntimeError(f"Slicing failed (rc={res['return_code']}). See {res['log_file']}")
                if fmt == "gcode.3mf":
                    shutil.copy2(Path(res["gcode_3mf_path"]), dest)
                else:
                    if res["gcode_files"]:
                        shutil.copy2(Path(res["gcode_files"][0]), dest)
                    else:
                        with zipfile.ZipFile(res["gcode_3mf_path"], "r") as zf:
                            gcodes = [n for n in zf.namelist() if n.endswith(".gcode")]
                            if not gcodes:
                                raise RuntimeError("No .gcode found inside sliced .gcode.3mf")
                            dest.write_bytes(zf.read(gcodes[0]))
                return {"format": fmt, "output_path": str(dest), "stats": res["stats"]}

        raise ValueError(f"Unsupported export format '{export_format}'. Use gcode, gcode.3mf, 3mf, or stl.")

    # ---- Printer communication ----

    def printer_list(self) -> dict[str, Any]:
        printers = self.ipc.get_printer_list()
        selected = self.ipc.get_selected_printer()
        return {"selected_printer_id": selected, "printers": printers}

    def printer_status(self, printer_id: str | None = None) -> dict[str, Any]:
        printers = self.ipc.get_printer_list()
        pid = printer_id or self.ipc.get_selected_printer()
        if not pid and printers:
            online = [p for p in printers if p.get("connectStatus") == 1]
            pid = (online[0] if online else printers[0]).get("printerId", "")
        if not pid:
            raise IPCError("No printer_id provided and no printer found.")
        info = self.ipc.get_printer_network_info(pid)
        raw = self.ipc.get_printer_status_raw(pid)
        mms = None
        try:
            mms = self.ipc.get_printer_mms_info(pid)
        except Exception:
            pass
        return {"printer_id": pid, "network_info": info, "status_raw": raw, "mms": mms}

    def printer_print(
        self,
        printer_id: str | None = None,
        sliced_file_path: str | None = None,
        project_path: str | None = None,
        plate: int = 1,
        bed_type: int = 0,
        timelapse: bool = False,
        heated_bed_leveling: bool = False,
        auto_refill: bool = False,
        upload_and_start_print: bool = True,
        filament_mms_mapping: list[dict[str, Any]] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Slice (if needed) and upload/start print job on a connected Elegoo printer."""
        file_to_send = Path(sliced_file_path).expanduser().resolve() if sliced_file_path else None
        slice_info = None
        if file_to_send is None:
            sliced = self.slice_project(plate=plate, project_path=project_path)
            if not sliced["ok"] or not sliced["gcode_3mf_path"]:
                raise RuntimeError(f"Auto-slicing failed before print: {sliced}")
            file_to_send = Path(sliced["gcode_3mf_path"])
            slice_info = sliced

        if not file_to_send.exists():
            raise FileNotFoundError(f"Sliced print file not found: {file_to_send}")

        pid = printer_id or ""
        if not pid:
            selected = self.ipc.get_selected_printer()
            if selected:
                pid = selected
            else:
                printers = self.ipc.get_printer_list()
                online = [p for p in printers if p.get("connectStatus") == 1]
                pid = (online[0] if online else printers[0]).get("printerId", "") if printers else ""

        payload_preview = {
            "printerId": pid,
            "filePath": str(file_to_send),
            "fileName": file_to_send.name,
            "bedType": bed_type,
            "timeLapse": timelapse,
            "heatedBedLeveling": heated_bed_leveling,
            "autoRefill": auto_refill,
            "uploadAndStartPrint": upload_and_start_print,
            "hasMms": bool(filament_mms_mapping),
            "filamentMmsMappingList": filament_mms_mapping or [],
        }
        if dry_run:
            return {"dry_run": True, "payload": payload_preview, "slice_info": slice_info}

        res = self.ipc.upload_and_print(
            printer_id=pid,
            file_path=str(file_to_send),
            file_name=file_to_send.name,
            upload_and_start_print=upload_and_start_print,
            bed_type=bed_type,
            timelapse=timelapse,
            heated_bed_leveling=heated_bed_leveling,
            auto_refill=auto_refill,
            has_mms=bool(filament_mms_mapping),
            filament_mms_mapping=filament_mms_mapping,
        )
        return {"ok": True, "upload_result": res, "payload": payload_preview, "slice_info": slice_info}

    # ---- 2.1 Visual PNG Previews for AI Inspection ----

    def preview_plate(
        self,
        plate_id: int = 1,
        width: int = 720,
        height: int = 540,
        yaw_deg: float = -40.0,
        pitch_deg: float = 32.0,
        output_path: str | None = None,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        ps = state["project_settings"]
        bed = self._bed_dimensions(ps)
        out_p = Path(output_path).expanduser().resolve() if output_path else None
        rendered = render_plate_preview(
            objects=state["objects"],
            bed_size=bed,
            plate_id=plate_id,
            filament_colours=ps.get("filament_colour", []),
            width=width,
            height=height,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            output_path=out_p,
        )
        return {
            "project_path": str(target),
            "plate_id": plate_id,
            "width": rendered["width"],
            "height": rendered["height"],
            "output_path": rendered["output_path"],
            "objects": self._summarize_objects(state),
            "_mcp_images": [{"base64": rendered["base64"], "mime_type": "image/png"}],
        }

    def preview_object_faces(
        self,
        object_index: int = 0,
        instance_index: int = 0,
        min_area_mm2: float = 5.0,
        top_n: int = 8,
        output_path: str | None = None,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        obj, inst = self._get_object_and_instance(state, object_index, instance_index)
        faces = detect_printable_faces(
            obj["mesh"],
            current_matrix=inst["matrix"],
            min_area_mm2=min_area_mm2,
            top_n=top_n,
        )
        out_p = Path(output_path).expanduser().resolve() if output_path else None
        rendered = render_object_faces(
            mesh=obj["mesh"],
            matrix=inst["matrix"],
            faces=faces,
            output_path=out_p,
        )
        return {
            "project_path": str(target),
            "object_index": object_index,
            "object_name": obj["name"],
            "faces": faces,
            "width": rendered["width"],
            "height": rendered["height"],
            "output_path": rendered["output_path"],
            "_mcp_images": [{"base64": rendered["base64"], "mime_type": "image/png"}],
        }

    # ---- 2.2 Per-Object / Part Config, Modifiers, Cut, Height Ranges ----

    def object_params_get(self, object_index: int = 0, project_path: str | None = None) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        obj, _ = self._get_object_and_instance(state, object_index, 0)
        return {
            "project_path": str(target),
            "object_index": object_index,
            "object_name": obj["name"],
            "extruder": obj["extruder"],
            "parameters": obj.get("config") or {},
        }

    def object_params_set(
        self,
        parameters: dict[str, Any],
        object_index: int = 0,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        obj, _ = self._get_object_and_instance(state, object_index, 0)
        cfg = obj.setdefault("config", {})
        applied: dict[str, Any] = {}
        for raw_k, raw_v in parameters.items():
            key = self.registry.resolve_key(raw_k)
            val = self.registry.normalize_value(key, raw_v)
            cfg[key] = str(val)
            applied[key] = cfg[key]
        self._write_3mf_bundle(target, state)
        return {
            "project_path": str(target),
            "object_index": object_index,
            "applied": applied,
            "parameters": cfg,
        }

    def object_parts_list(self, object_index: int = 0, project_path: str | None = None) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        obj, _ = self._get_object_and_instance(state, object_index, 0)
        parts_out = []
        for idx, part in enumerate(obj.get("parts") or []):
            dec = _decompose_matrix(part.get("matrix", np.eye(4, dtype=np.float64)))
            parts_out.append(
                {
                    "part_index": idx,
                    "part_id": part["part_id"],
                    "subtype": part.get("subtype", "normal_part"),
                    "name": part.get("name", ""),
                    "triangle_count": part["mesh"].triangle_count,
                    "transform": dec,
                    "config": part.get("config") or {},
                }
            )
        return {"project_path": str(target), "object_index": object_index, "parts": parts_out}

    def object_modifier_add(
        self,
        object_index: int = 0,
        subtype: str = "modifier_part",
        primitive: str = "box",
        params: dict[str, Any] | None = None,
        mesh_path: str | None = None,
        name: str | None = None,
        offset_mm: list[float] | None = None,
        rotation_deg: list[float] | None = None,
        scale: list[float] | float = 1.0,
        part_parameters: dict[str, Any] | None = None,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        """Add a modifier_part, negative_part, support_enforcer, support_blocker, or normal_part to an object."""
        allowed = {"normal_part", "modifier_part", "negative_part", "support_enforcer", "support_blocker"}
        if subtype not in allowed:
            raise ValueError(f"Unsupported part subtype '{subtype}'. Expected one of {sorted(allowed)}")

        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        obj, _ = self._get_object_and_instance(state, object_index, 0)

        if mesh_path:
            sub_mesh = self._load_mesh_from_file(Path(mesh_path).expanduser().resolve())
            part_name = name or Path(mesh_path).stem
        else:
            sub_mesh = make_primitive(primitive, params)
            part_name = name or f"{subtype}_{primitive}"

        # Center modifier mesh locally around origin (with bottom at Z=0 or centered)
        bmin, bmax = sub_mesh.bounds()
        c = 0.5 * (bmin + bmax)
        v = sub_mesh.vertices.copy()
        v[:, 0] -= c[0]
        v[:, 1] -= c[1]
        v[:, 2] -= c[2]
        sub_mesh = TriangleMeshData(v, sub_mesh.triangles)

        scale_vec = [float(scale)] * 3 if isinstance(scale, (int, float)) else [float(x) for x in scale]
        p_mat = _recompose_from_parts(offset_mm or [0.0, 0.0, 0.0], rotation_deg or [0.0, 0.0, 0.0], scale_vec)

        norm_cfg: dict[str, str] = {}
        for raw_k, raw_v in (part_parameters or {}).items():
            key = self.registry.resolve_key(raw_k)
            norm_cfg[key] = str(self.registry.normalize_value(key, raw_v))

        parts = obj.setdefault("parts", [])
        used_pids = {int(p["part_id"]) for p in parts if str(p.get("part_id", "")).isdigit()}
        new_pid = str(max(used_pids, default=0) + 1)
        parts.append(
            {
                "part_id": new_pid,
                "subtype": subtype,
                "name": part_name,
                "source_file": f"{part_name}.stl",
                "matrix": p_mat,
                "config": norm_cfg,
                "mesh": sub_mesh,
            }
        )
        self._write_3mf_bundle(target, state)
        return self.object_parts_list(object_index=object_index, project_path=str(target))

    def object_cut(
        self,
        object_index: int = 0,
        cut_z_mm: float = 5.0,
        keep: str = "both",
        flip_top: bool = False,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        """Cut an object horizontally at `cut_z_mm` above bed into lower & upper printable objects."""
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        obj, inst = self._get_object_and_instance(state, object_index, 0)

        low_mesh, up_mesh = split_mesh_by_z(obj["mesh"], inst["matrix"], float(cut_z_mm))
        orig_name = obj["name"]
        extruder = obj["extruder"]
        plate_id = int(inst.get("plate_id", 1))
        bmin, bmax = obj["mesh"].bounds(inst["matrix"])
        cx, cy = float(0.5 * (bmin[0] + bmax[0])), float(0.5 * (bmin[1] + bmax[1]))

        created_items: list[dict[str, Any]] = []
        if keep in {"both", "lower"}:
            m_low = np.eye(4, dtype=np.float64)
            m_low[0, 3], m_low[1, 3], m_low[2, 3] = cx, cy, 0.0
            created_items.append(
                {
                    "id": "",
                    "name": f"{orig_name}_bottom",
                    "extruder": extruder,
                    "source_file": obj.get("source_file", ""),
                    "config": dict(obj.get("config") or {}),
                    "layer_ranges": [],
                    "mesh": low_mesh,
                    "instances": [{"matrix": m_low, "printable": True, "plate_id": plate_id}],
                }
            )
        if keep in {"both", "upper"}:
            m_up = np.eye(4, dtype=np.float64)
            if flip_top:
                m_up = _recompose_from_parts([0.0, 0.0, 0.0], [180.0, 0.0, 0.0], [1.0, 1.0, 1.0])
                ubmin, _ = up_mesh.bounds(m_up)
                m_up[2, 3] = float(-ubmin[2])
            m_up[0, 3] = cx + (bmax[0] - bmin[0] + 10.0 if keep == "both" else 0.0)
            m_up[1, 3] = cy
            created_items.append(
                {
                    "id": "",
                    "name": f"{orig_name}_top",
                    "extruder": extruder,
                    "source_file": obj.get("source_file", ""),
                    "config": dict(obj.get("config") or {}),
                    "layer_ranges": [],
                    "mesh": up_mesh,
                    "instances": [{"matrix": m_up, "printable": True, "plate_id": plate_id}],
                }
            )

        state["objects"].pop(object_index)
        for offset, new_obj in enumerate(created_items):
            state["objects"].insert(object_index + offset, new_obj)

        self._write_3mf_bundle(target, state)
        return {
            "project_path": str(target),
            "cut_z_mm": cut_z_mm,
            "created_count": len(created_items),
            "objects": self._summarize_objects(state),
        }

    def object_layer_ranges_get(self, object_index: int = 0, project_path: str | None = None) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        obj, _ = self._get_object_and_instance(state, object_index, 0)
        return {
            "project_path": str(target),
            "object_index": object_index,
            "layer_ranges": obj.get("layer_ranges") or [],
        }

    def object_layer_ranges_set(
        self,
        ranges: list[dict[str, Any]],
        object_index: int = 0,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        """Set height-range print overrides `[{'min_z': 0, 'max_z': 5, 'options': {...}}]`."""
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        obj, _ = self._get_object_and_instance(state, object_index, 0)

        norm_ranges: list[dict[str, Any]] = []
        for r in ranges:
            opts_out: dict[str, str] = {}
            raw_opts = r.get("options") or r.get("parameters") or {}
            for raw_k, raw_v in raw_opts.items():
                key = self.registry.resolve_key(raw_k)
                opts_out[key] = str(self.registry.normalize_value(key, raw_v))
            norm_ranges.append(
                {
                    "min_z": float(r["min_z"]),
                    "max_z": float(r["max_z"]),
                    "options": opts_out,
                }
            )
        obj["layer_ranges"] = norm_ranges
        self._write_3mf_bundle(target, state)
        return {
            "project_path": str(target),
            "object_index": object_index,
            "layer_ranges": norm_ranges,
        }

    # ---- 2.3 Full Multi-Plate Management ----

    def plate_list(self, project_path: str | None = None) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        plates_meta = state.get("plates") or {}
        by_plate: dict[int, list[dict[str, Any]]] = {pid: [] for pid in plates_meta}
        for idx, obj in enumerate(state["objects"]):
            for i_idx, inst in enumerate(obj["instances"]):
                pid = int(inst.get("plate_id", 1))
                by_plate.setdefault(pid, []).append(
                    {"object_index": idx, "instance_index": i_idx, "name": obj["name"]}
                )
        out = []
        for pid in sorted(by_plate):
            pm = plates_meta.get(pid, {})
            out.append(
                {
                    "plate_id": pid,
                    "name": pm.get("name", ""),
                    "locked": bool(pm.get("locked", False)),
                    "bed_type": pm.get("bed_type", ""),
                    "objects": by_plate[pid],
                }
            )
        return {"project_path": str(target), "plates": out}

    def plate_create(
        self,
        name: str = "",
        plate_id: int | None = None,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        plates = state.setdefault("plates", {})
        new_pid = int(plate_id) if plate_id is not None else (max(plates.keys(), default=0) + 1)
        plates[new_pid] = {"plate_id": new_pid, "name": name, "locked": False, "bed_type": ""}
        self._write_3mf_bundle(target, state)
        return self.plate_list(project_path=str(target))

    def plate_delete(self, plate_id: int, move_objects_to_plate: int = 1, project_path: str | None = None) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        plates = state.get("plates") or {}
        if len(plates) <= 1:
            raise ValueError("Cannot delete the only remaining plate.")
        if plate_id not in plates:
            raise KeyError(f"Plate {plate_id} does not exist.")
        if plate_id == move_objects_to_plate:
            move_objects_to_plate = next(p for p in sorted(plates) if p != plate_id)
        for obj in state["objects"]:
            for inst in obj["instances"]:
                if int(inst.get("plate_id", 1)) == plate_id:
                    inst["plate_id"] = move_objects_to_plate
        del plates[plate_id]
        self._write_3mf_bundle(target, state)
        return self.plate_list(project_path=str(target))

    def plate_move_object(
        self,
        object_index: int,
        target_plate_id: int,
        instance_index: int | None = None,
        arrange_target_plate: bool = True,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        if object_index < 0 or object_index >= len(state["objects"]):
            raise IndexError(f"Invalid object_index: {object_index}")
        obj = state["objects"][object_index]
        plates = state.setdefault("plates", {})
        if target_plate_id not in plates:
            plates[target_plate_id] = {"plate_id": target_plate_id, "name": "", "locked": False, "bed_type": ""}
        for i_idx, inst in enumerate(obj["instances"]):
            if instance_index is None or i_idx == instance_index:
                inst["plate_id"] = int(target_plate_id)
        if arrange_target_plate:
            self._arrange_state_in_place(state, plate_id=int(target_plate_id))
        self._write_3mf_bundle(target, state)
        return self.plate_list(project_path=str(target))

    def plate_set_settings(
        self,
        plate_id: int = 1,
        name: str | None = None,
        locked: bool | None = None,
        bed_type: str | None = None,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_project_path(project_path, must_exist=True)
        state = self._read_3mf_bundle(target)
        plates = state.setdefault("plates", {})
        pm = plates.setdefault(plate_id, {"plate_id": plate_id, "name": "", "locked": False, "bed_type": ""})
        if name is not None:
            pm["name"] = name
        if locked is not None:
            pm["locked"] = bool(locked)
        if bed_type is not None:
            pm["bed_type"] = bed_type
            state["project_settings"]["curr_bed_type"] = bed_type
        self._write_3mf_bundle(target, state)
        return self.plate_list(project_path=str(target))

    # ---- 2.4 GUI Live Sync Status & Bridge ----

    def gui_sync_status(self) -> dict[str, Any]:
        ipc_up = self.ipc.is_available()
        native_slicer_ipc = self.ipc.supports_slicer_namespace() if ipc_up else False
        return {
            "ipc_connected": ipc_up,
            "native_slicer_ipc_available": native_slicer_ipc,
            "active_project_path": str(self._active_project) if self._active_project else None,
            "sync_mode": "native_live_ipc" if native_slicer_ipc else "file_3mf_and_wm_copydata",
        }

    def gui_sync_from_live(self, project_path: str | None = None) -> dict[str, Any]:
        """Pull live scene/config from ElegooSlicer when running a `slicer.*` IPC build."""
        if not self.ipc.supports_slicer_namespace():
            raise IPCError(
                "Live scene sync from ElegooSlicer memory requires a build with SlicerIPC (`slicer.*`). "
                "Your current ElegooSlicer binary uses `.3mf` file workflow + `gui_open_project`."
            )
        live_state = self.ipc.call("slicer.getState")
        live_cfg = self.ipc.call("slicer.getConfig", {"scope": "print"})
        if project_path or self._active_project:
            target = self.resolve_project_path(project_path, must_exist=True)
            state = self._read_3mf_bundle(target)
            for k, v in (live_cfg.get("values") or {}).items():
                state["project_settings"][k] = v
            self._write_3mf_bundle(target, state)
        return {"ok": True, "live_state": live_state, "live_config_count": len(live_cfg.get("values") or {})}

    def gui_sync_to_live(
        self,
        project_path: str | None = None,
        auto_confirm_preset_warning: bool = True,
        dont_show_again: bool = False,
    ) -> dict[str, Any]:
        """Push active project changes to live ElegooSlicer GUI (via SlicerIPC or WM_COPYDATA)."""
        target = self.resolve_project_path(project_path, must_exist=True)
        if self.ipc.supports_slicer_namespace():
            res = self.ipc.call("slicer.openProject", {"path": str(target)})
            dialogs = []
            if auto_confirm_preset_warning:
                from .gui_bridge import dismiss_preset_dialogs

                dialogs = dismiss_preset_dialogs(dont_show_again=dont_show_again, timeout_seconds=1.5)
            return {
                "mode": "native_live_ipc",
                "result": res,
                "project_path": str(target),
                "dialogs_handled": dialogs,
            }
        gui_res = send_project_to_running_gui(
            target,
            auto_confirm_preset_warning=auto_confirm_preset_warning,
            dont_show_again=dont_show_again,
        )
        return {"mode": gui_res["method"], "result": gui_res, "project_path": str(target)}

    # ---- 2.5 Persistent User Profiles ----

    def profile_save_user(
        self,
        kind: str,
        name: str,
        overrides: dict[str, Any] | None = None,
        inherits_from: str | None = None,
        from_active_project: bool = False,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        """Persist a new reusable ElegooSlicer user preset (.json + .info)."""
        kind_norm = {"process": "print", "machine": "printer"}.get(kind.lower(), kind.lower())
        merged_overrides = dict(overrides or {})
        parent = inherits_from or ""
        if from_active_project:
            target = self.resolve_project_path(project_path, must_exist=True)
            state = self._read_3mf_bundle(target)
            ps = state["project_settings"]
            if not parent:
                parent_map = {
                    "print": ps.get("print_settings_id", ""),
                    "printer": ps.get("printer_settings_id", ""),
                    "filament": (ps.get("filament_settings_id") or [""])[0],
                }
                parent = str(parent_map.get(kind_norm, ""))
            if not merged_overrides and kind_norm == "print" and parent:
                # Extract keys that differ from parent preset
                try:
                    base_prof = self.registry.resolve_profile("print", parent)
                    for k, v in ps.items():
                        if k in base_prof and str(base_prof[k]) != str(v):
                            merged_overrides[k] = v
                except KeyError:
                    pass
        return self.registry.save_user_profile(
            kind=kind_norm,
            name=name,
            overrides=merged_overrides,
            inherits_from=parent,
        )

    def profile_delete_user(self, kind: str, name: str) -> dict[str, Any]:
        return self.registry.delete_user_profile(kind=kind, name=name)

    # ---- 2.6 Extended Live Printer & Print Task Control ----

    def _resolve_live_printer_id(self, printer_id: str | None = None) -> str:
        if printer_id:
            return printer_id
        sel = self.ipc.get_selected_printer()
        if sel:
            return sel
        printers = self.ipc.get_printer_list()
        if not printers:
            raise IPCError("No printers registered in running ElegooSlicer.")
        online = [p for p in printers if p.get("connectStatus") == 1]
        return str((online[0] if online else printers[0]).get("printerId", ""))

    def printer_tasks(self, printer_id: str | None = None, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        pid = self._resolve_live_printer_id(printer_id)
        data = self.ipc.get_print_task_list(pid, page=page, page_size=page_size)
        return {"printer_id": pid, **data}

    def printer_tasks_delete(self, task_ids: list[str], printer_id: str | None = None) -> dict[str, Any]:
        pid = self._resolve_live_printer_id(printer_id)
        res = self.ipc.delete_print_tasks(pid, task_ids)
        return {"printer_id": pid, "deleted_task_ids": task_ids, "result": res}

    def printer_exceptions(self, printer_id: str | None = None, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        pid = self._resolve_live_printer_id(printer_id)
        data = self.ipc.get_exception_list(pid, page=page, page_size=page_size)
        return {"printer_id": pid, **data}

    def printer_refresh_status(self, printer_id: str | None = None) -> dict[str, Any]:
        pid = self._resolve_live_printer_id(printer_id)
        res = self.ipc.refresh_printer_status(pid)
        return {"printer_id": pid, "result": res}

    def printer_pause(self, printer_id: str | None = None, dry_run: bool = False) -> dict[str, Any]:
        pid = self._resolve_live_printer_id(printer_id)
        return self.ipc.send_sdcp_command(pid, cmd=129, data={}, dry_run=dry_run)

    def printer_resume(self, printer_id: str | None = None, dry_run: bool = False) -> dict[str, Any]:
        pid = self._resolve_live_printer_id(printer_id)
        return self.ipc.send_sdcp_command(pid, cmd=131, data={}, dry_run=dry_run)

    def printer_stop(self, printer_id: str | None = None, dry_run: bool = False) -> dict[str, Any]:
        pid = self._resolve_live_printer_id(printer_id)
        return self.ipc.send_sdcp_command(pid, cmd=130, data={}, dry_run=dry_run)

    def printer_rtm_command(
        self,
        cmd: int | None = None,
        data: dict[str, Any] | None = None,
        raw_message: str | dict[str, Any] | None = None,
        printer_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        pid = self._resolve_live_printer_id(printer_id)
        if raw_message is not None:
            if dry_run:
                return {"dry_run": True, "printer_id": pid, "message": raw_message}
            res = self.ipc.send_rtm_message(pid, raw_message)
            return {"ok": True, "printer_id": pid, "result": res}
        if cmd is None:
            raise ValueError("Provide either 'cmd' (SDCP integer command) or 'raw_message'.")
        return self.ipc.send_sdcp_command(pid, cmd=cmd, data=data or {}, dry_run=dry_run)
