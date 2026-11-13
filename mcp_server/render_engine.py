"""Pure Python + NumPy Z-buffer 3D preview renderer producing PNG images for MCP visual inspection."""

from __future__ import annotations

import base64
import math
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .mesh_engine import TriangleMeshData


# Compact 5x7 pixel font for crisp canvas annotations
_FONT_5X7: dict[str, tuple[str, ...]] = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00110", "01000", "10000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "#": ("01010", "01010", "11111", "01010", "11111", "01010", "01010"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}

_FACE_PALETTE: list[tuple[int, int, int]] = [
    (46, 204, 113),   # F0 green
    (52, 152, 219),   # F1 blue
    (230, 126, 34),   # F2 orange
    (155, 89, 182),   # F3 purple
    (241, 196, 15),   # F4 yellow
    (231, 76, 60),    # F5 red
    (26, 188, 156),   # F6 teal
    (236, 112, 99),   # F7 coral
]


def _hex_to_rgb(color: str, fallback: tuple[int, int, int] = (70, 140, 225)) -> tuple[int, int, int]:
    c = (color or "").strip().lstrip("#")
    if len(c) >= 6:
        try:
            return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        except ValueError:
            pass
    return fallback


def encode_png_rgb(rgb: np.ndarray) -> bytes:
    """Encode `(H, W, 3)` uint8 numpy array into valid PNG bytes using stdlib zlib."""
    h, w, ch = rgb.shape
    assert ch == 3
    raw_rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    compressed = zlib.compress(raw_rows, level=6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


def _draw_line(
    img: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[int, int, int],
) -> None:
    h, w, _ = img.shape
    steps = int(max(abs(x1 - x0), abs(y1 - y0), 1)) + 1
    xs = np.round(np.linspace(x0, x1, steps)).astype(np.int32)
    ys = np.round(np.linspace(y0, y1, steps)).astype(np.int32)
    mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    img[ys[mask], xs[mask]] = color


def _draw_text_pill(
    img: np.ndarray,
    cx: int,
    cy: int,
    text: str,
    fg: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (20, 24, 30),
    scale: int = 2,
) -> None:
    h, w, _ = img.shape
    txt = text.upper()
    text_w = len(txt) * 6 * scale - scale
    text_h = 7 * scale
    pad = 3 * scale // 2 + 2
    x0 = max(2, min(w - text_w - 2 * pad - 2, cx - (text_w + 2 * pad) // 2))
    y0 = max(2, min(h - text_h - 2 * pad - 2, cy - (text_h + 2 * pad) // 2))
    img[y0 : y0 + text_h + 2 * pad, x0 : x0 + text_w + 2 * pad] = bg

    cur_x = x0 + pad
    cur_y = y0 + pad
    for ch in txt:
        glyph = _FONT_5X7.get(ch, _FONT_5X7[" "])
        for r_idx, row in enumerate(glyph):
            for c_idx, bit in enumerate(row):
                if bit == "1":
                    px = cur_x + c_idx * scale
                    py = cur_y + r_idx * scale
                    img[py : py + scale, px : px + scale] = fg
        cur_x += 6 * scale


def _camera_basis(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Return 3x3 world->view rotation matrix (right, up, forward-toward-camera)."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    right = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float64)
    fwd_xy = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    view_dir = math.cos(pitch) * fwd_xy + math.sin(pitch) * up_world  # points toward camera
    up = np.cross(view_dir, right)
    up /= np.linalg.norm(up)
    return np.vstack([right, up, view_dir])


def _rasterize_triangles(
    img: np.ndarray,
    zbuf: np.ndarray,
    pts_2d: np.ndarray,  # (N, 3): x_px, y_px, depth
    tris: np.ndarray,
    tri_colors: np.ndarray,  # (M, 3) uint8
    normals_cam: np.ndarray,  # (M, 3)
    viewport: tuple[int, int, int, int] = (0, 0, 9999, 9999),
) -> None:
    vx0, vy0, vx1, vy1 = viewport
    h, w, _ = img.shape
    vx0, vy0 = max(0, vx0), max(0, vy0)
    vx1, vy1 = min(w - 1, vx1), min(h - 1, vy1)

    # Sort back-to-front or use zbuffer; use per-triangle bounding-box barycentric test
    v0 = pts_2d[tris[:, 0]]
    v1 = pts_2d[tris[:, 1]]
    v2 = pts_2d[tris[:, 2]]

    light = np.array([0.35, 0.45, 0.82], dtype=np.float64)
    light /= np.linalg.norm(light)

    for idx in range(len(tris)):
        # Cull degenerate or back-facing in camera space
        area2 = (v1[idx, 0] - v0[idx, 0]) * (v2[idx, 1] - v0[idx, 1]) - (v1[idx, 1] - v0[idx, 1]) * (
            v2[idx, 0] - v0[idx, 0]
        )
        if abs(area2) < 1e-4:
            continue

        min_x = max(vx0, int(math.floor(min(v0[idx, 0], v1[idx, 0], v2[idx, 0]))))
        max_x = min(vx1, int(math.ceil(max(v0[idx, 0], v1[idx, 0], v2[idx, 0]))))
        min_y = max(vy0, int(math.floor(min(v0[idx, 1], v1[idx, 1], v2[idx, 1]))))
        max_y = min(vy1, int(math.ceil(max(v0[idx, 1], v1[idx, 1], v2[idx, 1]))))
        if min_x > max_x or min_y > max_y:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float64)[None, :] + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float64)[:, None] + 0.5

        w0 = (v2[idx, 0] - v1[idx, 0]) * (ys - v1[idx, 1]) - (v2[idx, 1] - v1[idx, 1]) * (xs - v1[idx, 0])
        w1 = (v0[idx, 0] - v2[idx, 0]) * (ys - v2[idx, 1]) - (v0[idx, 1] - v2[idx, 1]) * (xs - v2[idx, 0])
        w2 = (v1[idx, 0] - v0[idx, 0]) * (ys - v0[idx, 1]) - (v1[idx, 1] - v0[idx, 1]) * (xs - v0[idx, 0])

        if area2 > 0:
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        else:
            inside = (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
        if not np.any(inside):
            continue

        inv_a = 1.0 / area2
        b0 = w0 * inv_a
        b1 = w1 * inv_a
        b2 = w2 * inv_a
        z_interp = b0 * v0[idx, 2] + b1 * v1[idx, 2] + b2 * v2[idx, 2]

        sub_z = zbuf[min_y : max_y + 1, min_x : max_x + 1]
        hit = inside & (z_interp > sub_z)
        if not np.any(hit):
            continue

        sub_z[hit] = z_interp[hit]
        ndotl = max(0.22, abs(float(np.dot(normals_cam[idx], light))))
        shaded = np.clip(tri_colors[idx].astype(np.float64) * (0.35 + 0.65 * ndotl), 0, 255).astype(np.uint8)
        sub_img = img[min_y : max_y + 1, min_x : max_x + 1]
        sub_img[hit] = shaded


def _project_points(
    pts: np.ndarray,
    cam: np.ndarray,
    scale: float,
    origin_cam: np.ndarray,
    cx_px: float,
    cy_px: float,
) -> np.ndarray:
    cpts = (pts - origin_cam) @ cam.T
    out = np.empty_like(cpts)
    out[:, 0] = cx_px + cpts[:, 0] * scale
    out[:, 1] = cy_px - cpts[:, 1] * scale
    out[:, 2] = cpts[:, 2]
    return out


def render_plate_preview(
    objects: list[dict[str, Any]],
    bed_size: tuple[float, float] = (256.0, 256.0),
    plate_id: int = 1,
    filament_colours: list[str] | None = None,
    width: int = 720,
    height: int = 540,
    yaw_deg: float = -40.0,
    pitch_deg: float = 32.0,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Render an isometric preview PNG of the active build plate & objects."""
    img = np.full((height, width, 3), (26, 29, 36), dtype=np.uint8)
    zbuf = np.full((height, width), -1e12, dtype=np.float64)

    bed_w, bed_d = bed_size
    cam = _camera_basis(yaw_deg, pitch_deg)
    bed_corners = np.array(
        [[0.0, 0.0, 0.0], [bed_w, 0.0, 0.0], [bed_w, bed_d, 0.0], [0.0, bed_d, 0.0]],
        dtype=np.float64,
    )
    origin = np.array([bed_w * 0.5, bed_d * 0.5, 10.0], dtype=np.float64)
    cam_corners = (bed_corners - origin) @ cam.T
    span_x = max(1e-3, float(cam_corners[:, 0].max() - cam_corners[:, 0].min()))
    span_y = max(1e-3, float(cam_corners[:, 1].max() - cam_corners[:, 1].min()))
    scale = min((width * 0.84) / span_x, (height * 0.78) / span_y)
    cx_px, cy_px = width * 0.5, height * 0.55

    # Draw bed grid lines every 32mm
    step = 32.0
    for gx in np.arange(0.0, bed_w + 0.1, step):
        p = _project_points(np.array([[gx, 0.0, 0.0], [gx, bed_d, 0.0]]), cam, scale, origin, cx_px, cy_px)
        col = (95, 105, 122) if gx in (0.0, bed_w) else (48, 54, 65)
        _draw_line(img, p[0, 0], p[0, 1], p[1, 0], p[1, 1], col)
    for gy in np.arange(0.0, bed_d + 0.1, step):
        p = _project_points(np.array([[0.0, gy, 0.0], [bed_w, gy, 0.0]]), cam, scale, origin, cx_px, cy_px)
        col = (95, 105, 122) if gy in (0.0, bed_d) else (48, 54, 65)
        _draw_line(img, p[0, 0], p[0, 1], p[1, 0], p[1, 1], col)

    # Origin axes indicator
    ax_p = _project_points(
        np.array([[0.0, 0.0, 0.0], [35.0, 0.0, 0.0], [0.0, 35.0, 0.0], [0.0, 0.0, 35.0]]),
        cam,
        scale,
        origin,
        cx_px,
        cy_px,
    )
    _draw_line(img, ax_p[0, 0], ax_p[0, 1], ax_p[1, 0], ax_p[1, 1], (235, 75, 75))
    _draw_line(img, ax_p[0, 0], ax_p[0, 1], ax_p[2, 0], ax_p[2, 1], (75, 220, 95))
    _draw_line(img, ax_p[0, 0], ax_p[0, 1], ax_p[3, 0], ax_p[3, 1], (85, 155, 245))

    colours = filament_colours or []
    labels: list[tuple[int, int, str]] = []

    for o_idx, obj in enumerate(objects):
        slot = max(1, int(obj.get("extruder", 1)))
        slot_hex = colours[slot - 1] if slot - 1 < len(colours) else ""
        base_rgb = _hex_to_rgb(slot_hex, fallback=_FACE_PALETTE[o_idx % len(_FACE_PALETTE)])
        mesh: TriangleMeshData = obj["mesh"]
        if mesh.triangle_count == 0:
            continue

        for inst in obj.get("instances", []):
            if int(inst.get("plate_id", 1)) != plate_id:
                continue
            M = inst["matrix"]
            wpts = (mesh.vertices @ M[:3, :3].T) + M[:3, 3]
            pts_2d = _project_points(wpts, cam, scale, origin, cx_px, cy_px)
            v0 = wpts[mesh.triangles[:, 0]]
            v1 = wpts[mesh.triangles[:, 1]]
            v2 = wpts[mesh.triangles[:, 2]]
            wnorm = np.cross(v1 - v0, v2 - v0)
            nlen = np.linalg.norm(wnorm, axis=1, keepdims=True)
            wnorm = wnorm / np.where(nlen > 1e-12, nlen, 1.0)
            cnorm = wnorm @ cam.T
            tri_cols = np.tile(np.array(base_rgb, dtype=np.uint8), (len(mesh.triangles), 1))
            _rasterize_triangles(img, zbuf, pts_2d, mesh.triangles, tri_cols, cnorm)

            bmin, bmax = wpts.min(axis=0), wpts.max(axis=0)
            top_mid = np.array([[0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1]), bmax[2] + 4.0]])
            lp = _project_points(top_mid, cam, scale, origin, cx_px, cy_px)
            labels.append((int(lp[0, 0]), int(lp[0, 1]), f"#{o_idx}"))

    for lx, ly, txt in labels:
        _draw_text_pill(img, lx, ly, txt, fg=(255, 255, 255), bg=(18, 22, 28), scale=2)
    _draw_text_pill(img, 55, 22, f"P{plate_id}", fg=(140, 210, 255), bg=(34, 40, 52), scale=2)

    png_bytes = encode_png_rgb(img)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(png_bytes)
    return {
        "width": width,
        "height": height,
        "png_bytes": png_bytes,
        "base64": base64.b64encode(png_bytes).decode("ascii"),
        "output_path": str(output_path) if output_path else None,
    }


def render_object_faces(
    mesh: TriangleMeshData,
    matrix: np.ndarray,
    faces: list[dict[str, Any]],
    width: int = 760,
    height: int = 440,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Render two opposite isometric views highlighting and labeling candidate faces F0, F1..."""
    img = np.full((height, width, 3), (26, 29, 36), dtype=np.uint8)
    zbuf = np.full((height, width), -1e12, dtype=np.float64)

    wpts = (mesh.vertices @ matrix[:3, :3].T) + matrix[:3, 3]
    bmin, bmax = wpts.min(axis=0), wpts.max(axis=0)
    center = 0.5 * (bmin + bmax)
    size = np.maximum(bmax - bmin, 1.0)
    max_dim = float(np.linalg.norm(size))

    v0 = wpts[mesh.triangles[:, 0]]
    v1 = wpts[mesh.triangles[:, 1]]
    v2 = wpts[mesh.triangles[:, 2]]
    wnorm = np.cross(v1 - v0, v2 - v0)
    nlen = np.linalg.norm(wnorm, axis=1, keepdims=True)
    wnorm = wnorm / np.where(nlen > 1e-12, nlen, 1.0)
    centroids = (v0 + v1 + v2) / 3.0

    # Assign triangle colors matching top 8 candidate faces
    tri_cols = np.full((len(mesh.triangles), 3), (155, 162, 175), dtype=np.uint8)
    top_faces = faces[:8]
    offset_tol = max(0.1, max_dim * 0.005)
    for f in top_faces:
        fn = np.array(f["normal"], dtype=np.float64)
        fc = np.array(f["centroid_mm"], dtype=np.float64)
        dots = wnorm @ fn
        dists = np.abs((centroids - fc) @ fn)
        mask = (dots > 0.994) & (dists <= offset_tol)
        tri_cols[mask] = _FACE_PALETTE[f["face_id"] % len(_FACE_PALETTE)]

    # Render Left (front-top iso) and Right (rear-bottom iso so -Z and back faces are visible!)
    views = [
        (-40.0, 30.0, width * 0.25, height * 0.46, (0, 0, width // 2 - 2, height - 40)),
        (140.0, -28.0, width * 0.75, height * 0.46, (width // 2 + 2, 0, width - 1, height - 40)),
    ]
    scale = (min(width * 0.44, height * 0.72) / max(max_dim, 1e-3))

    for yaw, pitch, cx_px, cy_px, vp in views:
        cam = _camera_basis(yaw, pitch)
        pts_2d = _project_points(wpts, cam, scale, center, cx_px, cy_px)
        cnorm = wnorm @ cam.T
        _rasterize_triangles(img, zbuf, pts_2d, mesh.triangles, tri_cols, cnorm, viewport=vp)

        # Draw label only for faces oriented toward this camera
        for f in top_faces:
            fn = np.array(f["normal"], dtype=np.float64)
            cam_z = float((fn @ cam.T)[2])
            if cam_z > 0.08:
                fc = np.array([f["centroid_mm"]], dtype=np.float64)
                lp = _project_points(fc, cam, scale, center, cx_px, cy_px)
                col = _FACE_PALETTE[f["face_id"] % len(_FACE_PALETTE)]
                _draw_text_pill(
                    img,
                    int(lp[0, 0]),
                    int(lp[0, 1]),
                    f"F{f['face_id']}",
                    fg=(15, 18, 22),
                    bg=col,
                    scale=2,
                )

    # Divider & legend bar at bottom
    _draw_line(img, width // 2, 10, width // 2, height - 42, (55, 62, 74))
    leg_x = 45
    for f in top_faces[:6]:
        col = _FACE_PALETTE[f["face_id"] % len(_FACE_PALETTE)]
        _draw_text_pill(
            img,
            leg_x,
            height - 20,
            f"F{f['face_id']}:{int(round(f['area_mm2']))}",
            fg=(15, 18, 22),
            bg=col,
            scale=2,
        )
        leg_x += 120

    png_bytes = encode_png_rgb(img)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(png_bytes)
    return {
        "width": width,
        "height": height,
        "png_bytes": png_bytes,
        "base64": base64.b64encode(png_bytes).decode("ascii"),
        "output_path": str(output_path) if output_path else None,
    }
