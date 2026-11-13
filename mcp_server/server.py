"""Model Context Protocol (MCP) JSON-RPC 2.0 server for ElegooSlicer (stdio + HTTP)."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

from . import __version__
from .project_engine import ProjectEngine


PROTOCOL_VERSION = "2024-11-05"


def _tool_def(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }


def build_tool_registry(engine: ProjectEngine) -> tuple[list[dict[str, Any]], dict[str, Callable[[dict[str, Any]], Any]]]:
    """Register all ElegooSlicer MCP tools and their handlers."""
    tools: list[dict[str, Any]] = []
    handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}

    def register(
        name: str,
        description: str,
        properties: dict[str, Any],
        handler: Callable[[dict[str, Any]], Any],
        required: list[str] | None = None,
    ) -> None:
        tools.append(_tool_def(name, description, properties, required))
        handlers[name] = handler

    # ---- Project & GUI ----
    register(
        "project_new",
        "Create a new ElegooSlicer .3mf project with selected printer, print, and filament profiles.",
        {
            "project_path": {"type": "string", "description": "Optional target .3mf path (defaults to workspace active project)."},
            "printer_profile": {"type": "string", "description": "Printer profile name (e.g. 'Elegoo Centauri Carbon 2 0.4 nozzle')."},
            "print_profile": {"type": "string", "description": "Process/print profile name (e.g. '0.20mm Standard @Elegoo CC2')."},
            "filament_profiles": {"type": "array", "items": {"type": "string"}, "description": "List of filament presets for slots 1..N."},
            "sync_printer_filaments": {"type": "boolean", "description": "Read CANVAS/MMS filament slots from the connected printer."},
            "open_in_gui": {"type": "boolean", "description": "Open the created project immediately in running ElegooSlicer GUI."},
        },
        lambda a: engine.project_new(**a),
    )

    register(
        "project_open",
        "Open an existing .3mf project as the active MCP project (and optionally load it into running ElegooSlicer GUI).",
        {
            "project_path": {"type": "string", "description": "Path to existing .3mf file."},
            "open_in_gui": {"type": "boolean", "description": "Send project to running ElegooSlicer GUI window."},
        },
        lambda a: engine.project_open(**a),
        required=["project_path"],
    )

    register(
        "project_save",
        "Save/copy the active .3mf project to a destination path.",
        {
            "destination_path": {"type": "string", "description": "Destination .3mf path."},
            "project_path": {"type": "string", "description": "Optional source project path."},
        },
        lambda a: engine.project_save(**a),
    )

    register(
        "project_info",
        "Inspect the active .3mf project: bed size, active profiles, filament slots, and objects.",
        {"project_path": {"type": "string"}},
        lambda a: engine.project_info(**a),
    )

    register(
        "gui_launch",
        "Launch ElegooSlicer desktop GUI (empty or with an optional .3mf project) and wait for its main window.",
        {
            "project_path": {"type": "string", "description": "Optional .3mf path to open on launch; omit to launch ElegooSlicer empty."},
            "wait_seconds": {"type": "number", "description": "Max seconds to wait for the main window (default 10)."},
            "auto_confirm_preset_warning": {"type": "boolean", "description": "Auto-confirm 'Préréglage personnalisé' / 'Customized Preset' if it pops up on load (default true)."},
            "dont_show_again": {"type": "boolean", "description": "Check 'Ne plus afficher' / persist suppression if a preset warning pops up."},
        },
        lambda a: engine.gui_launch(**a),
    )

    register(
        "gui_open_project",
        "Load/refresh the active (or given) .3mf project inside the live ElegooSlicer desktop GUI.",
        {
            "project_path": {"type": "string"},
            "auto_confirm_preset_warning": {"type": "boolean", "description": "Auto-confirm 'Préréglage personnalisé' / 'Customized Preset' modal popups if shown (default true)."},
            "dont_show_again": {"type": "boolean", "description": "Also check 'Ne plus afficher' and suppress future preset warnings."},
        },
        lambda a: engine.open_in_gui(**a),
    )

    register(
        "gui_list_windows",
        "List all visible ElegooSlicer GUI windows, popups, modal dialogs (e.g. 'Préréglage personnalisé'), buttons, checkboxes, and texts.",
        {
            "include_invisible": {"type": "boolean"},
            "include_children": {"type": "boolean", "description": "Enumerate child buttons/checkboxes/texts (default true)."},
        },
        lambda a: engine.gui_list_windows(**a),
    )

    register(
        "gui_handle_dialog",
        "Control a modal ElegooSlicer popup/dialog (e.g. click OK/Cancel on 'Préréglage personnalisé', check 'Ne plus afficher', or close).",
        {
            "hwnd": {"type": "integer", "description": "Optional dialog HWND from gui_list_windows (auto-selects active dialog if omitted)."},
            "title_contains": {"type": "string", "description": "Optional substring match on dialog title or text (e.g. 'Préréglage')."},
            "action": {"type": "string", "enum": ["click_button", "dismiss_preset_warning", "check_dont_show_again", "close"]},
            "button": {"type": "string", "description": "Button label to click (default 'OK'; supports OK, Cancel/Annuler, Yes/Oui, No/Non, etc.)."},
            "check_dont_show_again": {"type": "boolean", "description": "Check 'Ne plus afficher' / 'Don't show again' and persist setting in ElegooSlicer.conf."},
            "persist_no_preset_warning": {"type": "boolean", "description": "Persist 'no_warn_when_modified_gcodes: true' in ElegooSlicer.conf."},
        },
        lambda a: engine.gui_handle_dialog(**a),
    )

    register(
        "gui_control_window",
        "Control any ElegooSlicer GUI window or dialog: focus, minimize, maximize, restore, close, or send a key (enter/escape/space/tab).",
        {
            "hwnd": {"type": "integer"},
            "title_contains": {"type": "string"},
            "action": {"type": "string", "enum": ["focus", "minimize", "maximize", "restore", "close", "send_key"]},
            "key": {"type": "string", "description": "Key name when action='send_key' (enter, escape, space, tab, up, down, left, right)."},
        },
        lambda a: engine.gui_control_window(**a),
        required=["action"],
    )

    register(
        "gui_live_ipc_call",
        "Invoke any live IPCServer RPC directly (printer.*, user.*, or slicer.* when running an IPC-extended build).",
        {
            "method": {"type": "string", "description": "RPC method name, e.g. 'printer.getPrinterList' or 'slicer.getState'."},
            "params": {"type": "object", "description": "Optional JSON params dictionary."},
        },
        lambda a: engine.ipc.call(a["method"], a.get("params") or {}),
        required=["method"],
    )

    # ---- Objects, Printable Faces, Rotation & Placement ----
    register(
        "object_add",
        "Add one or multiple 3D objects (.stl, .obj, .step/.stp, .3mf files or primitives: cube, cylinder, sphere, disc) to the project.",
        {
            "items": {
                "type": "array",
                "description": "Objects to add. Each item accepts 'path' or 'primitive' (+ optional 'params', 'name', 'copies', 'filament_slot', 'rotation_deg', 'scale', 'offset_mm', 'plate_id').",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "primitive": {"type": "string", "enum": ["cube", "box", "cylinder", "disc", "sphere"]},
                        "params": {"type": "object"},
                        "name": {"type": "string"},
                        "copies": {"type": "integer"},
                        "filament_slot": {"type": "integer"},
                        "rotation_deg": {"type": "array", "items": {"type": "number"}},
                        "scale": {"type": ["number", "array"]},
                        "offset_mm": {"type": "array", "items": {"type": "number"}},
                        "plate_id": {"type": "integer"},
                    },
                },
            },
            "project_path": {"type": "string"},
            "auto_arrange": {"type": "boolean", "description": "Automatically arrange objects cleanly on the bed (default true)."},
            "spacing_mm": {"type": "number", "description": "Spacing between arranged objects in mm (default 6)."},
            "plate_id": {"type": "integer", "description": "Target plate number (default 1)."},
        },
        lambda a: engine.add_objects(**a),
        required=["items"],
    )

    register(
        "object_list",
        "List all objects on the build plate(s) with dimensions, positions, rotations, scales, and filament slots.",
        {"project_path": {"type": "string"}},
        lambda a: {"objects": engine.project_info(**a)["objects"]},
    )

    register(
        "object_remove",
        "Remove an object by index or clear all objects from the project.",
        {
            "object_index": {"type": "integer"},
            "clear_all": {"type": "boolean"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.remove_object(**a),
    )

    register(
        "object_list_faces",
        "Detect candidate planar printable faces ('Lay on face') for an object with area, normal, exterior-hull contact flag, and overhang ratio.",
        {
            "object_index": {"type": "integer"},
            "instance_index": {"type": "integer"},
            "min_area_mm2": {"type": "number", "description": "Minimum planar face area in mm^2 (default 5.0)."},
            "top_n": {"type": "integer", "description": "Max faces to return (default 20)."},
            "project_path": {"type": "string"},
        },
        lambda a: engine.object_list_faces(**a),
    )

    register(
        "object_lay_on_face",
        "Place an object flat on a chosen printable face (by face_id from object_list_faces or custom normal [nx, ny, nz]) and drop it onto Z=0.",
        {
            "object_index": {"type": "integer"},
            "instance_index": {"type": "integer"},
            "face_id": {"type": "integer", "description": "Face ID from object_list_faces."},
            "normal": {"type": "array", "items": {"type": "number"}, "description": "Optional explicit outward face normal [nx, ny, nz]."},
            "min_area_mm2": {"type": "number"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.object_lay_on_face(**a),
    )

    register(
        "object_auto_orient",
        "Automatically orient one or all objects onto the best exterior printable face minimizing overhangs and maximizing bed contact.",
        {
            "object_index": {"type": "integer", "description": "Omit to auto-orient all objects."},
            "min_area_mm2": {"type": "number"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.object_auto_orient(**a),
    )

    register(
        "object_rotate",
        "Rotate an object in degrees (relative or absolute, via [rx, ry, rz] or axis='X'|'Y'|'Z' + angle_deg) and drop to Z=0.",
        {
            "object_index": {"type": "integer"},
            "instance_index": {"type": "integer"},
            "rotation_deg": {"type": "array", "items": {"type": "number"}, "description": "[rx, ry, rz] in degrees."},
            "axis": {"type": "string", "enum": ["X", "Y", "Z"]},
            "angle_deg": {"type": "number"},
            "relative": {"type": "boolean", "description": "Default true (incremental rotation)."},
            "drop_to_bed": {"type": "boolean", "description": "Default true (keep min Z on bed)."},
            "project_path": {"type": "string"},
        },
        lambda a: engine.object_rotate(**a),
    )

    register(
        "object_transform",
        "Set absolute object position (offset_mm), rotation_deg, scale (uniform or [sx, sy, sz]), and/or mirror axes.",
        {
            "object_index": {"type": "integer"},
            "instance_index": {"type": "integer"},
            "offset_mm": {"type": "array", "items": {"type": "number"}},
            "rotation_deg": {"type": "array", "items": {"type": "number"}},
            "scale": {"type": ["number", "array"]},
            "mirror": {"type": "array", "items": {"type": "boolean"}},
            "drop_to_bed": {"type": "boolean"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.object_transform(**a),
    )

    register(
        "object_arrange",
        "Arrange all objects cleanly on the specified plate with collision-free spacing.",
        {
            "plate_id": {"type": "integer"},
            "spacing_mm": {"type": "number"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.object_arrange(**a),
    )

    # ---- Filaments ----
    register(
        "filament_list_profiles",
        "List available system and user filament presets.",
        {
            "query": {"type": "string", "description": "Filter by keyword (e.g. 'PLA', 'TPU', 'PETG', 'ECC2')."},
            "limit": {"type": "integer"},
        },
        lambda a: engine.registry.list_profiles(kind="filament", query=a.get("query", ""), limit=a.get("limit", 100)),
    )

    register(
        "filament_list_slots",
        "List the filament slots (1..N) configured in the active project.",
        {"project_path": {"type": "string"}},
        lambda a: engine.filament_list_slots(**a),
    )

    register(
        "filament_set_slots",
        "Configure filament slots (presets, hex colors, per-slot parameter overrides like temperatures or flow ratio).",
        {
            "slots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "slot": {"type": "integer", "description": "1-based slot index."},
                        "preset": {"type": "string"},
                        "color": {"type": "string", "description": "Hex color e.g. '#FF0000'."},
                        "overrides": {"type": "object", "description": "Per-slot filament parameters e.g. nozzle_temperature."},
                    },
                },
            },
            "project_path": {"type": "string"},
        },
        lambda a: engine.filament_set_slots(**a),
        required=["slots"],
    )

    register(
        "filament_assign_to_object",
        "Assign a filament slot (1-based extruder/CANVAS index) to a specific object.",
        {
            "object_index": {"type": "integer"},
            "slot": {"type": "integer"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.filament_assign_to_object(**a),
        required=["object_index", "slot"],
    )

    register(
        "filament_sync_from_printer",
        "Read live CANVAS/MMS tray types and colors from the connected Elegoo printer and apply them to the project.",
        {
            "printer_id": {"type": "string"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.filament_sync_from_printer(**a),
    )

    # ---- Parameters & Profiles (FR + EN) ----
    register(
        "params_list_categories",
        "List all ElegooSlicer parameter categories/tabs in English and French.",
        {},
        lambda _a: {"categories": engine.registry.list_categories()},
    )

    register(
        "params_search",
        "Search any of ElegooSlicer's ~700+ print/filament/printer parameters by English or French UI label (e.g. 'Hauteur de couche', 'Paroi extérieure', 'seam_position', 'support').",
        {
            "query": {"type": "string"},
            "category": {"type": "string", "description": "e.g. 'Qualité', 'Solidité', 'Vitesse', 'Support', 'Quality', 'Strength'."},
            "limit": {"type": "integer"},
        },
        lambda a: {"results": engine.registry.search_params(**a)},
    )

    register(
        "params_get",
        "Read current parameter values from the active project (by keys, French/English labels, category, or search query).",
        {
            "keys": {"type": "array", "items": {"type": "string"}},
            "category": {"type": "string"},
            "query": {"type": "string"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.params_get(**a),
    )

    register(
        "params_set",
        "Modify any print/filament/printer parameter in the project using canonical keys OR French/English UI labels and values (e.g. {'Hauteur de couche': 0.16, 'Position de la couture': 'Alignée', 'sparse_infill_density': '25%'}).",
        {
            "parameters": {"type": "object", "description": "Mapping of parameter key/label -> new value."},
            "project_path": {"type": "string"},
        },
        lambda a: engine.params_set(**a),
        required=["parameters"],
    )

    register(
        "profiles_list",
        "List available print (process), filament, and printer (machine) presets (system + user presets).",
        {
            "kind": {"type": "string", "enum": ["all", "print", "process", "filament", "printer", "machine"]},
            "query": {"type": "string"},
            "include_hidden": {"type": "boolean"},
            "limit": {"type": "integer"},
        },
        lambda a: engine.registry.list_profiles(**a),
    )

    register(
        "profile_select",
        "Switch the active print, printer, or filament profile(s) in the project cleanly resolving inheritance.",
        {
            "print_profile": {"type": "string"},
            "printer_profile": {"type": "string"},
            "filament_profiles": {"type": "array", "items": {"type": "string"}},
            "project_path": {"type": "string"},
        },
        lambda a: engine.profile_select(**a),
    )

    # ---- Slicing & Exporting ----
    register(
        "slice_project",
        "Slice the project plate headless via ElegooSlicer CLI, producing `.gcode`/`.gcode.3mf` plus estimated time, filament usage, and layer count.",
        {
            "plate": {"type": "integer", "description": "Plate index (1-based, or 0 for all plates)."},
            "output_dir": {"type": "string"},
            "export_3mf_name": {"type": "string"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.slice_project(**a),
    )

    register(
        "export_output",
        "Export the project or object as `.gcode`, `.gcode.3mf`, `.3mf`, or `.stl`.",
        {
            "export_format": {"type": "string", "enum": ["gcode", "gcode.3mf", "3mf", "stl"]},
            "destination_path": {"type": "string"},
            "plate": {"type": "integer"},
            "object_index": {"type": "integer"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.export_output(**a),
        required=["export_format", "destination_path"],
    )

    # ---- Printer & Print Job ----
    register(
        "printer_list",
        "List printers registered in the running ElegooSlicer instance with connection status.",
        {},
        lambda _a: engine.printer_list(),
    )

    register(
        "printer_status",
        "Get detailed live printer status, temperatures, print progress, and CANVAS MMS status.",
        {"printer_id": {"type": "string"}},
        lambda a: engine.printer_status(**a),
    )

    register(
        "printer_mms_info",
        "Get CANVAS / multi-material filament tray info from the printer.",
        {"printer_id": {"type": "string"}},
        lambda a: engine.ipc.get_printer_mms_info(a["printer_id"]) if a.get("printer_id") else engine.printer_status()["mms"],
    )

    register(
        "printer_files",
        "List print files stored on the connected printer.",
        {"printer_id": {"type": "string"}},
        lambda a: engine.ipc.get_file_list(a.get("printer_id") or engine.printer_status()["printer_id"]),
    )

    register(
        "printer_print",
        "Upload a sliced `.gcode` / `.gcode.3mf` (or auto-slice active project first) to the connected Elegoo printer and optionally start printing.",
        {
            "printer_id": {"type": "string", "description": "Omit to use selected/online printer."},
            "sliced_file_path": {"type": "string", "description": "Optional pre-sliced .gcode.3mf / .gcode file. If omitted, slices active project first."},
            "project_path": {"type": "string"},
            "plate": {"type": "integer"},
            "bed_type": {"type": "integer"},
            "timelapse": {"type": "boolean"},
            "heated_bed_leveling": {"type": "boolean"},
            "auto_refill": {"type": "boolean"},
            "upload_and_start_print": {"type": "boolean", "description": "True to upload & immediately start printing; false to upload only."},
            "filament_mms_mapping": {"type": "array", "items": {"type": "object"}},
            "dry_run": {"type": "boolean", "description": "Prepare and return the IPC payload without sending to printer."},
        },
        lambda a: engine.printer_print(**a),
    )

    register(
        "printer_upload_status",
        "Check upload/print task progress returned by printer_print.",
        {"task_id": {"type": "string"}},
        lambda a: engine.ipc.get_upload_task(a["task_id"]),
        required=["task_id"],
    )

    register(
        "printer_upload_cancel",
        "Cancel an active upload task.",
        {"task_id": {"type": "string"}},
        lambda a: engine.ipc.cancel_upload_task(a["task_id"]),
        required=["task_id"],
    )

    # ---- Visual Previews (2.1) ----
    register(
        "preview_plate",
        "Render an isometric PNG preview image of the build plate and all objects (returned as an MCP image block).",
        {
            "plate_id": {"type": "integer", "description": "1-based plate index (default: 1)."},
            "width": {"type": "integer", "description": "Image width in pixels (default: 720)."},
            "height": {"type": "integer", "description": "Image height in pixels (default: 540)."},
            "yaw_deg": {"type": "number", "description": "Camera yaw angle in degrees (default: -40)."},
            "pitch_deg": {"type": "number", "description": "Camera pitch angle in degrees (default: 32)."},
            "output_path": {"type": "string"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.preview_plate(**a),
    )

    register(
        "preview_object_faces",
        "Render a 3-angle PNG contact sheet inspecting candidate planar faces (F0, F1, ...) for orienting/laying an object flat.",
        {
            "object_index": {"type": "integer", "description": "0-based object index (default: 0)."},
            "instance_index": {"type": "integer", "description": "0-based instance index (default: 0)."},
            "min_area_mm2": {"type": "number", "description": "Minimum face area threshold in mm^2."},
            "top_n": {"type": "integer", "description": "Number of candidate faces to annotate (default: 8)."},
            "output_path": {"type": "string"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.preview_object_faces(**a),
    )

    # ---- Per-Object Config, Parts, Modifiers, Cut, Layer Ranges (2.2) ----
    register(
        "object_params_get",
        "Get per-object override parameters (keys/values) for a specific object.",
        {
            "object_index": {"type": "integer", "description": "0-based object index (default: 0)."},
            "project_path": {"type": "string"},
        },
        lambda a: engine.object_params_get(**a),
    )

    register(
        "object_params_set",
        "Set per-object override parameters (supports canonical keys or French/English UI labels).",
        {
            "parameters": {"type": "object", "description": "Mapping of parameter key/label -> value."},
            "object_index": {"type": "integer", "description": "0-based object index (default: 0)."},
            "project_path": {"type": "string"},
        },
        lambda a: engine.object_params_set(**a),
        required=["parameters"],
    )

    register(
        "object_parts_list",
        "List all sub-parts / volumes inside an object (normal parts, modifiers, negative volumes, support enforcers/blockers).",
        {
            "object_index": {"type": "integer", "description": "0-based object index (default: 0)."},
            "project_path": {"type": "string"},
        },
        lambda a: engine.object_parts_list(**a),
    )

    register(
        "object_modifier_add",
        "Add a modifier volume, negative volume, support enforcer, or support blocker (primitive box/cylinder/sphere or external mesh) to an object.",
        {
            "object_index": {"type": "integer", "description": "0-based object index (default: 0)."},
            "subtype": {
                "type": "string",
                "enum": ["modifier_part", "negative_part", "support_enforcer", "support_blocker", "normal_part"],
            },
            "primitive": {"type": "string", "enum": ["box", "cylinder", "sphere"]},
            "params": {"type": "object", "description": "Primitive dimensions (e.g. size_x/y/z or radius/height)."},
            "mesh_path": {"type": "string", "description": "Optional external STL/OBJ/3MF mesh path instead of a primitive."},
            "name": {"type": "string"},
            "offset_mm": {"type": "array", "items": {"type": "number"}, "description": "[x, y, z] offset relative to object."},
            "rotation_deg": {"type": "array", "items": {"type": "number"}, "description": "[rx, ry, rz] degrees."},
            "scale": {"type": "number"},
            "part_parameters": {"type": "object", "description": "Per-part overrides (for modifier_part, supports FR/EN labels)."},
            "project_path": {"type": "string"},
        },
        lambda a: engine.object_modifier_add(**a),
    )

    register(
        "object_cut",
        "Perform a planar Z-cut on an object, splitting it cleanly into lower and/or upper capped printable meshes.",
        {
            "object_index": {"type": "integer"},
            "cut_z_mm": {"type": "number", "description": "World Z plane height in mm to cut through (must be > 0 and within object Z height)."},
            "keep": {"type": "string", "enum": ["both", "lower", "upper"], "description": "Which half(ves) to keep (default: 'both')."},
            "flip_top": {"type": "boolean", "description": "If true, flips the upper half 180 deg onto the cut face."},
            "project_path": {"type": "string"},
        },
        lambda a: engine.object_cut(**a),
    )

    register(
        "object_layer_ranges_get",
        "Get Z height range overrides (layer height modifiers) configured for an object.",
        {
            "object_index": {"type": "integer"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.object_layer_ranges_get(**a),
    )

    register(
        "object_layer_ranges_set",
        "Set Z height range overrides on an object (list of {min_z, max_z, options}).",
        {
            "ranges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "min_z": {"type": "number"},
                        "max_z": {"type": "number"},
                        "options": {"type": "object"},
                        "parameters": {"type": "object"},
                    },
                },
            },
            "object_index": {"type": "integer"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.object_layer_ranges_set(**a),
        required=["ranges"],
    )

    # ---- Multi-Plate Management (2.3) ----
    register(
        "plate_list",
        "List all build plates in the project with their settings and assigned object instances.",
        {"project_path": {"type": "string"}},
        lambda a: engine.plate_list(**a),
    )

    register(
        "plate_create",
        "Create a new build plate in the project.",
        {
            "name": {"type": "string"},
            "plate_id": {"type": "integer"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.plate_create(**a),
    )

    register(
        "plate_delete",
        "Delete a build plate (moving any remaining objects onto move_objects_to_plate).",
        {
            "plate_id": {"type": "integer"},
            "move_objects_to_plate": {"type": "integer"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.plate_delete(**a),
        required=["plate_id"],
    )

    register(
        "plate_move_object",
        "Move an object (or single instance) onto a target plate and optionally re-arrange the target plate.",
        {
            "object_index": {"type": "integer"},
            "target_plate_id": {"type": "integer"},
            "instance_index": {"type": "integer"},
            "arrange_target_plate": {"type": "boolean"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.plate_move_object(**a),
        required=["object_index", "target_plate_id"],
    )

    register(
        "plate_set_settings",
        "Update per-plate metadata (name, locked, bed_type).",
        {
            "plate_id": {"type": "integer"},
            "name": {"type": "string"},
            "locked": {"type": "boolean"},
            "bed_type": {"type": "string"},
            "project_path": {"type": "string"},
        },
        lambda a: engine.plate_set_settings(**a),
    )

    # ---- Bidirectional Live GUI Sync (2.4) ----
    register(
        "gui_sync_status",
        "Inspect ElegooSlicer live GUI IPC state, running status, active project, and native slicer.* extension availability.",
        {},
        lambda _a: engine.gui_sync_status(),
    )

    register(
        "gui_sync_from_live",
        "Pull printer/preset/scene state from the running ElegooSlicer GUI into the MCP workspace.",
        {"project_path": {"type": "string"}},
        lambda a: engine.gui_sync_from_live(**a),
    )

    register(
        "gui_sync_to_live",
        "Push/load the active MCP `.3mf` project into the running ElegooSlicer GUI.",
        {
            "project_path": {"type": "string"},
            "auto_confirm_preset_warning": {"type": "boolean", "description": "Auto-confirm 'Préréglage personnalisé' / 'Customized Preset' if shown (default true)."},
            "dont_show_again": {"type": "boolean", "description": "Check 'Ne plus afficher' and persist warning suppression."},
        },
        lambda a: engine.gui_sync_to_live(**a),
    )

    register(
        "gui_app_config",
        "Read or update ElegooSlicer.conf `app` flags (e.g. `no_warn_when_modified_gcodes`) while preserving ElegooSlicer's MD5 checksum.",
        {
            "keys": {"type": "array", "items": {"type": "string"}, "description": "Optional app config keys to read."},
            "updates": {"type": "object", "description": "Optional key/value updates under `app` in ElegooSlicer.conf."},
        },
        lambda a: engine.gui_app_config(**a),
    )

    # ---- Persistent User Presets (2.5) ----
    register(
        "profile_save_user",
        "Create or update a persistent user preset (.json + .info) in ElegooSlicer's user preset folder.",
        {
            "kind": {"type": "string", "enum": ["process", "print", "filament", "machine", "printer"]},
            "name": {"type": "string"},
            "overrides": {"type": "object", "description": "Parameter overrides (supports FR/EN labels or canonical keys)."},
            "inherits_from": {"type": "string", "description": "Parent preset name to inherit from."},
            "from_active_project": {"type": "boolean", "description": "Copy overrides from the currently open MCP project."},
            "project_path": {"type": "string"},
        },
        lambda a: engine.profile_save_user(**a),
        required=["kind", "name"],
    )

    register(
        "profile_delete_user",
        "Delete a user-created preset from ElegooSlicer's user preset folder (refuses system presets).",
        {
            "kind": {"type": "string", "enum": ["process", "print", "filament", "machine", "printer"]},
            "name": {"type": "string"},
        },
        lambda a: engine.profile_delete_user(**a),
        required=["kind", "name"],
    )

    # ---- Printer Task Queue & SDCP / RTM Control (2.6) ----
    register(
        "printer_tasks",
        "List print task history / queue from the connected Elegoo printer.",
        {
            "printer_id": {"type": "string"},
            "page": {"type": "integer"},
            "page_size": {"type": "integer"},
        },
        lambda a: engine.printer_tasks(**a),
    )

    register(
        "printer_tasks_delete",
        "Delete one or more print task records from the printer history.",
        {
            "task_ids": {"type": "array", "items": {"type": "string"}},
            "printer_id": {"type": "string"},
        },
        lambda a: engine.printer_tasks_delete(**a),
        required=["task_ids"],
    )

    register(
        "printer_exceptions",
        "Fetch printer error/exception log entries from ElegooLink.",
        {
            "printer_id": {"type": "string"},
            "page": {"type": "integer"},
            "page_size": {"type": "integer"},
        },
        lambda a: engine.printer_exceptions(**a),
    )

    register(
        "printer_refresh_status",
        "Send SDCP Cmd 0 to request an immediate live status refresh from the printer.",
        {"printer_id": {"type": "string"}},
        lambda a: engine.printer_refresh_status(**a),
    )

    register(
        "printer_pause",
        "Pause the active print job on the connected Elegoo printer (SDCP Cmd 129). Supports dry_run=true.",
        {
            "printer_id": {"type": "string"},
            "dry_run": {"type": "boolean"},
        },
        lambda a: engine.printer_pause(**a),
    )

    register(
        "printer_resume",
        "Resume a paused print job on the connected Elegoo printer (SDCP Cmd 131). Supports dry_run=true.",
        {
            "printer_id": {"type": "string"},
            "dry_run": {"type": "boolean"},
        },
        lambda a: engine.printer_resume(**a),
    )

    register(
        "printer_stop",
        "Stop/abort the active print job on the connected Elegoo printer (SDCP Cmd 130). Supports dry_run=true.",
        {
            "printer_id": {"type": "string"},
            "dry_run": {"type": "boolean"},
        },
        lambda a: engine.printer_stop(**a),
    )

    register(
        "printer_rtm_command",
        "Send a custom Elegoo SDCP command (Cmd + Data) or raw RTM JSON envelope to the printer. Supports dry_run=true.",
        {
            "cmd": {"type": "integer"},
            "data": {"type": "object"},
            "raw_message": {"type": "object"},
            "printer_id": {"type": "string"},
            "dry_run": {"type": "boolean"},
        },
        lambda a: engine.printer_rtm_command(**a),
    )

    return tools, handlers


class ElegooMCPServer:
    """JSON-RPC 2.0 MCP Server implementation."""

    def __init__(self, engine: ProjectEngine | None = None) -> None:
        self.engine = engine or ProjectEngine()
        self.tools, self.handlers = build_tool_registry(self.engine)

    def handle_jsonrpc(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        msg_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params") or {}

        # Notifications have no id -> no response sent
        is_notification = msg_id is None

        try:
            if method == "initialize":
                client_proto = params.get("protocolVersion") or PROTOCOL_VERSION
                result = {
                    "protocolVersion": client_proto if isinstance(client_proto, str) else PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": "elegoo-slicer-mcp",
                        "version": __version__,
                    },
                }
                return None if is_notification else {"jsonrpc": "2.0", "id": msg_id, "result": result}

            if method in {"notifications/initialized", "initialized"}:
                return None

            if method == "ping":
                return None if is_notification else {"jsonrpc": "2.0", "id": msg_id, "result": {}}

            if method == "tools/list":
                return None if is_notification else {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"tools": self.tools},
                }

            if method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments") or {}
                if tool_name not in self.handlers:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                    }
                try:
                    data = self.handlers[tool_name](arguments)
                    images: list[dict[str, Any]] = []
                    if isinstance(data, dict) and "_mcp_images" in data:
                        data = dict(data)
                        raw_imgs = data.pop("_mcp_images", None)
                        if isinstance(raw_imgs, list):
                            images = [
                                img for img in raw_imgs
                                if isinstance(img, dict) and img.get("base64")
                            ]
                    text = json.dumps(data, indent=2, ensure_ascii=False)
                    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
                    for img in images:
                        content.append(
                            {
                                "type": "image",
                                "data": img["base64"],
                                "mimeType": img.get("mime_type", "image/png"),
                            }
                        )
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": content,
                            "isError": False,
                        },
                    }
                except Exception as exc:
                    err_text = json.dumps(
                        {"error": str(exc), "type": type(exc).__name__},
                        indent=2,
                        ensure_ascii=False,
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [{"type": "text", "text": err_text}],
                            "isError": True,
                        },
                    }

            if is_notification:
                return None
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        except Exception as exc:
            sys.stderr.write(traceback.format_exc() + "\n")
            if is_notification:
                return None
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32603, "message": str(exc)},
            }

    def serve_stdio(self) -> int:
        """Run line-delimited JSON-RPC loop over stdin/stdout (UTF-8)."""
        # Ensure UTF-8 I/O on Windows without CRLF corruption
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()
                continue

            resp = self.handle_jsonrpc(msg)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        return 0

    def create_http_server(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        auth_token: str | None = None,
    ) -> tuple[ThreadingHTTPServer, threading.Event]:
        """Create a Streamable HTTP MCP server (`/mcp` POST/GET SSE/DELETE + `/health`)."""
        mcp = self
        sessions: dict[str, float] = {}
        sessions_lock = threading.Lock()
        stop_event = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send_cors_headers(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
                self.send_header(
                    "Access-Control-Allow-Headers",
                    "Content-Type, Authorization, Mcp-Session-Id, Last-Event-ID",
                )
                self.send_header("Access-Control-Expose-Headers", "Mcp-Session-Id")

            def _check_auth(self) -> bool:
                if not auth_token:
                    return True
                header = self.headers.get("Authorization", "")
                if not header.startswith("Bearer "):
                    self._send_json(
                        401,
                        {"error": "Unauthorized: Bearer token required"},
                        extra_headers={"WWW-Authenticate": "Bearer"},
                    )
                    return False
                supplied = header[7:].strip()
                if not hmac.compare_digest(supplied, auth_token):
                    self._send_json(
                        401,
                        {"error": "Unauthorized: invalid Bearer token"},
                        extra_headers={"WWW-Authenticate": "Bearer"},
                    )
                    return False
                return True

            def _route_path(self) -> str:
                return urlparse(self.path).path.rstrip("/") or "/"

            def do_OPTIONS(self) -> None:
                self.send_response(204)
                self._send_cors_headers()
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:
                path = self._route_path()
                if path == "/health":
                    with sessions_lock:
                        active_count = len(sessions)
                    self._send_json(
                        200,
                        {
                            "status": "ok",
                            "server": "elegoo-slicer-mcp",
                            "version": __version__,
                            "tools": len(mcp.tools),
                            "active_sessions": active_count,
                        },
                    )
                    return

                if path not in {"/mcp", "/"}:
                    self._send_json(404, {"error": f"Not found: {path}"})
                    return

                if not self._check_auth():
                    return

                req_sid = self.headers.get("Mcp-Session-Id")
                with sessions_lock:
                    if req_sid and req_sid not in sessions:
                        self._send_json(404, {"error": "Unknown or closed MCP session"})
                        return
                    sid = req_sid or uuid.uuid4().hex
                    sessions[sid] = time.time()

                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Mcp-Session-Id", sid)
                self.end_headers()

                try:
                    hello = f"retry: 5000\nevent: open\ndata: {json.dumps({'sessionId': sid})}\n\n".encode("utf-8")
                    self.wfile.write(hello)
                    self.wfile.flush()

                    last_ping = time.monotonic()
                    while not stop_event.is_set():
                        if req_sid:
                            with sessions_lock:
                                if req_sid not in sessions:
                                    break
                        stop_event.wait(0.25)
                        if stop_event.is_set():
                            break
                        if time.monotonic() - last_ping >= 15.0:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            last_ping = time.monotonic()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass

            def do_DELETE(self) -> None:
                path = self._route_path()
                if path not in {"/mcp", "/"}:
                    self._send_json(404, {"error": f"Not found: {path}"})
                    return
                if not self._check_auth():
                    return
                req_sid = self.headers.get("Mcp-Session-Id")
                if not req_sid:
                    self._send_json(400, {"error": "Missing Mcp-Session-Id header"})
                    return
                with sessions_lock:
                    existed = sessions.pop(req_sid, None) is not None
                if not existed:
                    self._send_json(404, {"error": "Unknown or closed MCP session"})
                    return
                self._send_json(200, {"closed": True, "sessionId": req_sid})

            def do_POST(self) -> None:
                path = self._route_path()
                if path not in {"/mcp", "/"}:
                    self._send_json(404, {"error": f"Not found: {path}"})
                    return
                if not self._check_auth():
                    return

                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8", errors="replace")
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    self._send_json(
                        400,
                        {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}},
                    )
                    return

                req_sid = self.headers.get("Mcp-Session-Id")
                if req_sid:
                    with sessions_lock:
                        if req_sid not in sessions:
                            self._send_json(404, {"error": "Unknown or closed MCP session"})
                            return
                        sessions[req_sid] = time.time()

                messages = payload if isinstance(payload, list) else [payload]
                assigned_sid = req_sid
                responses: list[dict[str, Any]] = []
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("method") == "initialize" and not assigned_sid:
                        assigned_sid = uuid.uuid4().hex
                        with sessions_lock:
                            sessions[assigned_sid] = time.time()
                    resp = mcp.handle_jsonrpc(msg)
                    if resp is not None:
                        responses.append(resp)

                extra_headers = {"Mcp-Session-Id": assigned_sid} if assigned_sid else None
                if not responses:
                    self.send_response(202)
                    self._send_cors_headers()
                    if assigned_sid:
                        self.send_header("Mcp-Session-Id", assigned_sid)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

                body: Any = responses if isinstance(payload, list) else responses[0]
                self._send_json(200, body, extra_headers=extra_headers)

            def _send_json(
                self,
                status: int,
                body: Any,
                extra_headers: dict[str, str] | None = None,
            ) -> None:
                payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                if extra_headers:
                    for k, v in extra_headers.items():
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        server = ThreadingHTTPServer((host, port), Handler)
        server.daemon_threads = True
        return server, stop_event

    def serve_http(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        auth_token: str | None = None,
    ) -> int:
        """Run Streamable HTTP MCP endpoint (`/mcp`, `/health`) until interrupted."""
        server, stop_event = self.create_http_server(host=host, port=port, auth_token=auth_token)
        actual_host, actual_port = server.server_address[:2]
        sys.stderr.write(f"ElegooSlicer MCP Streamable HTTP listening on http://{actual_host}:{actual_port}/mcp\n")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            server.shutdown()
            server.server_close()
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ElegooSlicer MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("ELEGOO_MCP_AUTH_TOKEN"),
        help="Optional Bearer token required on HTTP `/mcp` requests.",
    )
    parser.add_argument("--workspace", default=None, help="Optional MCP workspace directory")
    args = parser.parse_args(argv)

    from pathlib import Path
    engine = ProjectEngine(workspace_dir=Path(args.workspace) if args.workspace else None)
    srv = ElegooMCPServer(engine=engine)
    if args.transport == "http":
        return srv.serve_http(host=args.host, port=args.port, auth_token=args.auth_token)
    return srv.serve_stdio()
