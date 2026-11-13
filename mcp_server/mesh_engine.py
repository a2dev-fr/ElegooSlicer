"""NumPy-based 3D mesh I/O, printable face detection, lay-on-face rotation, transforms, and arranging."""

from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TriangleMeshData:
    vertices: np.ndarray  # (N, 3) float64
    triangles: np.ndarray  # (M, 3) int64

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    def bounds(self, matrix_4x4: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        pts = self.vertices
        if matrix_4x4 is not None:
            pts = (pts @ matrix_4x4[:3, :3].T) + matrix_4x4[:3, 3]
        if len(pts) == 0:
            z = np.zeros(3, dtype=np.float64)
            return z, z
        return pts.min(axis=0), pts.max(axis=0)


def load_stl(path: Path) -> TriangleMeshData:
    """Load binary or ASCII STL into indexed vertices + triangles."""
    data = path.read_bytes()
    if len(data) >= 84:
        tri_count = struct.unpack("<I", data[80:84])[0]
        expected_size = 84 + tri_count * 50
        if expected_size == len(data) and not data[:6].lower().startswith(b"solid "):
            return _parse_binary_stl(data, tri_count)
        if expected_size == len(data):
            # Some binary STLs start with "solid " header
            return _parse_binary_stl(data, tri_count)

    return _parse_ascii_stl(data.decode("utf-8", errors="replace"))


def _parse_binary_stl(data: bytes, tri_count: int) -> TriangleMeshData:
    dt = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("v0", "<f4", (3,)),
            ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)),
            ("attr", "<u2"),
        ]
    )
    arr = np.frombuffer(data, dtype=dt, count=tri_count, offset=84)
    pts = np.stack([arr["v0"], arr["v1"], arr["v2"]], axis=1).reshape(-1, 3).astype(np.float64)
    tris = np.arange(len(pts), dtype=np.int64).reshape(-1, 3)
    return _deduplicate_vertices(pts, tris)


def _parse_ascii_stl(text: str) -> TriangleMeshData:
    verts: list[list[float]] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("vertex "):
            parts = s.split()
            if len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        return TriangleMeshData(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64))
    usable = (len(verts) // 3) * 3
    pts = np.array(verts[:usable], dtype=np.float64)
    tris = np.arange(usable, dtype=np.int64).reshape(-1, 3)
    return _deduplicate_vertices(pts, tris)


def _deduplicate_vertices(pts: np.ndarray, tris: np.ndarray) -> TriangleMeshData:
    if len(pts) == 0:
        return TriangleMeshData(pts, tris)
    rounded = np.round(pts, decimals=6)
    _, unique_idx, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    # Preserve exact unrounded coordinates of first occurrence
    verts = pts[unique_idx]
    new_tris = inverse[tris].astype(np.int64)
    return TriangleMeshData(verts, new_tris)


def save_binary_stl(mesh: TriangleMeshData, path: Path, matrix_4x4: np.ndarray | None = None) -> None:
    """Write a binary STL file, optionally applying a 4x4 transform matrix."""
    pts = mesh.vertices
    if matrix_4x4 is not None:
        pts = (pts @ matrix_4x4[:3, :3].T) + matrix_4x4[:3, 3]
    v0 = pts[mesh.triangles[:, 0]]
    v1 = pts[mesh.triangles[:, 1]]
    v2 = pts[mesh.triangles[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    normals = normals / norms

    path.parent.mkdir(parents=True, exist_ok=True)
    header = b"ElegooSlicer MCP binary STL".ljust(80, b"\0")[:80]
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<I", len(mesh.triangles)))
        dt = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("v0", "<f4", (3,)),
                ("v1", "<f4", (3,)),
                ("v2", "<f4", (3,)),
                ("attr", "<u2"),
            ]
        )
        out = np.zeros(len(mesh.triangles), dtype=dt)
        out["normal"] = normals.astype(np.float32)
        out["v0"] = v0.astype(np.float32)
        out["v1"] = v1.astype(np.float32)
        out["v2"] = v2.astype(np.float32)
        fh.write(out.tobytes())


def load_obj(path: Path) -> TriangleMeshData:
    """Load Wavefront OBJ vertices and triangulate polygon faces."""
    verts: list[list[float]] = []
    tris: list[list[int]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            parts = line.split()[1:]
            idx_list: list[int] = []
            for p in parts:
                vi = int(p.split("/")[0])
                idx_list.append(vi - 1 if vi > 0 else len(verts) + vi)
            for k in range(1, len(idx_list) - 1):
                tris.append([idx_list[0], idx_list[k], idx_list[k + 1]])
    return TriangleMeshData(
        np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3), dtype=np.float64),
        np.array(tris, dtype=np.int64) if tris else np.zeros((0, 3), dtype=np.int64),
    )


def parse_3mf_model_meshes(model_xml_bytes: bytes) -> dict[str, TriangleMeshData]:
    """Parse `<object id="..."><mesh>...` entries from a `.model` XML payload."""
    root = ET.fromstring(model_xml_bytes)
    meshes: dict[str, TriangleMeshData] = {}
    for elem in root.iter():
        if elem.tag.endswith("}object") or elem.tag == "object":
            obj_id = elem.attrib.get("id", "")
            mesh_el = None
            for child in elem:
                if child.tag.endswith("}mesh") or child.tag == "mesh":
                    mesh_el = child
                    break
            if mesh_el is None:
                continue
            verts: list[list[float]] = []
            tris: list[list[int]] = []
            for sub in mesh_el:
                if sub.tag.endswith("}vertices") or sub.tag == "vertices":
                    for v in sub:
                        verts.append(
                            [float(v.attrib["x"]), float(v.attrib["y"]), float(v.attrib["z"])]
                        )
                elif sub.tag.endswith("}triangles") or sub.tag == "triangles":
                    for t in sub:
                        tris.append(
                            [int(t.attrib["v1"]), int(t.attrib["v2"]), int(t.attrib["v3"])]
                        )
            if verts and tris:
                meshes[obj_id] = TriangleMeshData(
                    np.array(verts, dtype=np.float64),
                    np.array(tris, dtype=np.int64),
                )
    return meshes


def load_3mf_combined_mesh(path: Path) -> TriangleMeshData:
    """Load the first object mesh (or combined meshes) from a 3MF archive."""
    all_verts: list[np.ndarray] = []
    all_tris: list[np.ndarray] = []
    vert_offset = 0
    with zipfile.ZipFile(path, "r") as zf:
        for name in sorted(zf.namelist()):
            if name.lower().endswith(".model"):
                meshes = parse_3mf_model_meshes(zf.read(name))
                for m in meshes.values():
                    if m.triangle_count == 0:
                        continue
                    all_verts.append(m.vertices)
                    all_tris.append(m.triangles + vert_offset)
                    vert_offset += m.vertex_count
    if not all_verts:
        raise ValueError(f"No triangle meshes found inside 3MF archive: {path}")
    return TriangleMeshData(np.vstack(all_verts), np.vstack(all_tris))


def make_primitive(kind: str, params: dict[str, Any] | None = None) -> TriangleMeshData:
    """Generate procedural watertight meshes centered on XY origin with min_z=0."""
    p = params or {}
    k = kind.lower().strip()
    if k in {"cube", "box"}:
        sx = float(p.get("size_x", p.get("size", 20.0)))
        sy = float(p.get("size_y", p.get("size", 20.0)))
        sz = float(p.get("size_z", p.get("size", 20.0)))
        hx, hy = sx / 2.0, sy / 2.0
        verts = np.array(
            [
                [-hx, -hy, 0.0],
                [hx, -hy, 0.0],
                [hx, hy, 0.0],
                [-hx, hy, 0.0],
                [-hx, -hy, sz],
                [hx, -hy, sz],
                [hx, hy, sz],
                [-hx, hy, sz],
            ],
            dtype=np.float64,
        )
        tris = np.array(
            [
                [0, 2, 1], [0, 3, 2],  # bottom (-Z)
                [4, 5, 6], [4, 6, 7],  # top (+Z)
                [0, 1, 5], [0, 5, 4],  # front (-Y)
                [2, 3, 7], [2, 7, 6],  # back (+Y)
                [1, 2, 6], [1, 6, 5],  # right (+X)
                [3, 0, 4], [3, 4, 7],  # left (-X)
            ],
            dtype=np.int64,
        )
        return TriangleMeshData(verts, tris)

    if k in {"cylinder", "disc", "disk"}:
        radius = float(p.get("radius", 10.0))
        height = float(p.get("height", 2.0 if k in {"disc", "disk"} else 20.0))
        segs = max(8, int(p.get("segments", 48)))
        angles = np.linspace(0.0, 2.0 * math.pi, segs, endpoint=False)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        verts_list = [[0.0, 0.0, 0.0], [0.0, 0.0, height]]
        for x, y in zip(xs, ys):
            verts_list.append([float(x), float(y), 0.0])
        for x, y in zip(xs, ys):
            verts_list.append([float(x), float(y), height])
        tris_list: list[list[int]] = []
        b0 = 2
        t0 = 2 + segs
        for i in range(segs):
            nxt = (i + 1) % segs
            tris_list.append([0, b0 + nxt, b0 + i])
            tris_list.append([1, t0 + i, t0 + nxt])
            tris_list.append([b0 + i, b0 + nxt, t0 + nxt])
            tris_list.append([b0 + i, t0 + nxt, t0 + i])
        return TriangleMeshData(
            np.array(verts_list, dtype=np.float64),
            np.array(tris_list, dtype=np.int64),
        )

    if k == "sphere":
        radius = float(p.get("radius", 10.0))
        rings = max(8, int(p.get("rings", 24)))
        segs = max(12, int(p.get("segments", 36)))
        verts_list = [[0.0, 0.0, 0.0]]  # bottom pole at z=0
        for r in range(1, rings):
            phi = math.pi * r / rings
            z = radius * (1.0 - math.cos(phi))
            rad_xy = radius * math.sin(phi)
            for s in range(segs):
                theta = 2.0 * math.pi * s / segs
                verts_list.append([rad_xy * math.cos(theta), rad_xy * math.sin(theta), z])
        top_idx = len(verts_list)
        verts_list.append([0.0, 0.0, 2.0 * radius])

        tris_list = []
        for s in range(segs):
            nxt = (s + 1) % segs
            tris_list.append([0, 1 + nxt, 1 + s])
        for r in range(rings - 2):
            row0 = 1 + r * segs
            row1 = 1 + (r + 1) * segs
            for s in range(segs):
                nxt = (s + 1) % segs
                tris_list.append([row0 + s, row0 + nxt, row1 + nxt])
                tris_list.append([row0 + s, row1 + nxt, row1 + s])
        last_row = 1 + (rings - 2) * segs
        for s in range(segs):
            nxt = (s + 1) % segs
            tris_list.append([last_row + s, last_row + nxt, top_idx])
        return TriangleMeshData(
            np.array(verts_list, dtype=np.float64),
            np.array(tris_list, dtype=np.int64),
        )

    raise ValueError(f"Unsupported primitive kind: {kind}")


# ---- Transform matrix utilities (matching bbs_3mf.cpp 12-float column-major 4x3) ----


def matrix_to_3mf_str(m: np.ndarray) -> str:
    """Format 4x4 affine matrix as 3MF 12-float column-by-column string."""
    vals: list[str] = []
    for c in range(4):
        for r in range(3):
            vals.append(f"{float(m[r, c]):.9g}")
    return " ".join(vals)


def str_to_matrix(s: str) -> np.ndarray:
    """Parse 3MF 12-float transform string into a 4x4 float64 matrix."""
    m = np.eye(4, dtype=np.float64)
    if not s or not s.strip():
        return m
    parts = [float(x) for x in s.strip().split()]
    if len(parts) != 12:
        return m
    idx = 0
    for c in range(4):
        for r in range(3):
            m[r, c] = parts[idx]
            idx += 1
    return m


def euler_deg_to_rot3(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Intrinsic XYZ / PrusaSlicer Rz(z)*Ry(y)*Rx(x) rotation matrix."""
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def rot3_to_euler_deg(r: np.ndarray) -> list[float]:
    """Extract Euler angles in degrees matching Rz * Ry * Rx."""
    sy = -float(r[2, 0])
    sy = max(-1.0, min(1.0, sy))
    y = math.asin(sy)
    if abs(abs(sy) - 1.0) > 1e-6:
        x = math.atan2(float(r[2, 1]), float(r[2, 2]))
        z = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        x = math.atan2(-float(r[1, 2]), float(r[1, 1]))
        z = 0.0
    return [round(math.degrees(x), 4), round(math.degrees(y), 4), round(math.degrees(z), 4)]


def rotation_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation matrix rotating unit vector `v_from` onto `v_to`."""
    a = v_from / np.linalg.norm(v_from)
    b = v_to / np.linalg.norm(v_to)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-9:
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ortho)
        axis /= np.linalg.norm(axis)
        # 180 deg rotation around axis
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + K + (K @ K) * ((1.0 - dot) / (s * s))


def compose_transform(
    scale: list[float] | np.ndarray = (1.0, 1.0, 1.0),
    rot3: np.ndarray | None = None,
    translation: list[float] | np.ndarray = (0.0, 0.0, 0.0),
) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    R = rot3 if rot3 is not None else np.eye(3, dtype=np.float64)
    S = np.diag([float(scale[0]), float(scale[1]), float(scale[2])])
    m[:3, :3] = R @ S
    m[:3, 3] = np.array(translation, dtype=np.float64)
    return m


# ---- Printable face detection & lay-on-face ----


def detect_printable_faces(
    mesh: TriangleMeshData,
    current_matrix: np.ndarray | None = None,
    min_area_mm2: float = 5.0,
    angle_tolerance_deg: float = 5.0,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """
    Detect candidate planar printable faces using outward normal + plane-offset clustering,
    checking whether each plane lies on the extreme exterior of the mesh (no geometry protruding past it)
    and computing contact area, overhang fraction, and resulting orientation bbox.
    """
    if mesh.triangle_count == 0:
        return []

    M = current_matrix if current_matrix is not None else np.eye(4, dtype=np.float64)
    pts = (mesh.vertices @ M[:3, :3].T) + M[:3, 3]
    v0 = pts[mesh.triangles[:, 0]]
    v1 = pts[mesh.triangles[:, 1]]
    v2 = pts[mesh.triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    double_areas = np.linalg.norm(cross, axis=1)
    valid = double_areas > 1e-8
    if not np.any(valid):
        return []

    areas = 0.5 * double_areas[valid]
    normals = cross[valid] / double_areas[valid][:, None]
    centroids = (v0[valid] + v1[valid] + v2[valid]) / 3.0

    # Group triangles by quantized normal & plane offset so opposite parallel faces don't merge.
    cos_tol = math.cos(math.radians(angle_tolerance_deg))
    # Sort descending by triangle area so seed normal comes from largest triangle in plane
    order = np.argsort(-areas)
    clusters: list[dict[str, Any]] = []

    bbox_diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    offset_tol = max(0.05, bbox_diag * 1e-3)

    for tri_idx in order:
        n = normals[tri_idx]
        c = centroids[tri_idx]
        a = float(areas[tri_idx])
        placed = False
        for cl in clusters:
            cn = cl["normal_sum"] / cl["area"]
            cn_norm = np.linalg.norm(cn)
            if cn_norm < 1e-9:
                continue
            cn_u = cn / cn_norm
            if float(np.dot(n, cn_u)) >= cos_tol:
                plane_dist = abs(float(np.dot(c - cl["ref_point"], cn_u)))
                if plane_dist <= offset_tol:
                    cl["normal_sum"] += n * a
                    cl["centroid_sum"] += c * a
                    cl["area"] += a
                    cl["count"] += 1
                    placed = True
                    break
        if not placed:
            clusters.append(
                {
                    "normal_sum": n * a,
                    "centroid_sum": c * a,
                    "ref_point": c.copy(),
                    "area": a,
                    "count": 1,
                }
            )
            if len(clusters) > 500:
                break

    # Compute extreme projections across all vertices to tell if the plane sits on exterior boundary
    faces: list[dict[str, Any]] = []
    down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    total_area = float(np.sum(areas))

    for cl in clusters:
        area = cl["area"]
        if area < min_area_mm2:
            continue
        unit_n = cl["normal_sum"] / np.linalg.norm(cl["normal_sum"])
        centroid = cl["centroid_sum"] / area

        # Distance of all vertices ahead of this plane along `unit_n`
        proj_all = pts @ unit_n
        plane_d = float(np.dot(centroid, unit_n))
        protrusion = float(np.max(proj_all) - plane_d)
        on_exterior_hull = protrusion <= max(0.15, offset_tol * 2.0)

        # Evaluate overhang ratio if this face were laid flat onto Z=0
        R_flat = rotation_from_two_vectors(unit_n, down)
        rot_normals = normals @ R_flat.T
        overhang_area = float(np.sum(areas[rot_normals[:, 2] < -0.707]))  # >45 deg overhang downward
        bed_contact_area = float(np.sum(areas[rot_normals[:, 2] < -0.996]))
        rot_pts = pts @ R_flat.T
        bmin, bmax = rot_pts.min(axis=0), rot_pts.max(axis=0)
        size = bmax - bmin

        # Friendly direction label in current coordinates
        dom_idx = int(np.argmax(np.abs(unit_n)))
        axis_names = ["X", "Y", "Z"]
        sign = "+" if unit_n[dom_idx] >= 0 else "-"
        approx_dir = f"{sign}{axis_names[dom_idx]}" if abs(unit_n[dom_idx]) > 0.9 else "angled"

        is_currently_bottom = bool(float(np.dot(unit_n, down)) > 0.996 and abs(float(centroid[2]) - float(pts[:, 2].min())) < 0.25)

        faces.append(
            {
                "face_id": 0,  # filled after sorting
                "area_mm2": round(area, 3),
                "bed_contact_area_mm2": round(bed_contact_area, 3),
                "normal": [round(float(v), 5) for v in unit_n],
                "centroid_mm": [round(float(v), 3) for v in centroid],
                "triangle_count": cl["count"],
                "approx_direction": approx_dir,
                "on_exterior_hull": on_exterior_hull,
                "protrusion_beyond_plane_mm": round(max(0.0, protrusion), 3),
                "overhang_ratio": round(overhang_area / max(total_area, 1e-6), 4),
                "resulting_size_mm": [round(float(size[0]), 2), round(float(size[1]), 2), round(float(size[2]), 2)],
                "is_current_bottom": is_currently_bottom,
            }
        )

    # Sort primarily by exterior hull contact + contact area + low overhang
    faces.sort(
        key=lambda f: (
            not f["on_exterior_hull"],
            -f["area_mm2"],
            f["overhang_ratio"],
        )
    )
    for idx, f in enumerate(faces[:top_n]):
        f["face_id"] = idx
    return faces[:top_n]


def compute_lay_on_face_matrix(
    mesh: TriangleMeshData,
    current_matrix: np.ndarray,
    face_normal: list[float] | np.ndarray,
    bed_z: float = 0.0,
) -> np.ndarray:
    """
    Rotate current instance transform so `face_normal` (in current world space) points to `(0, 0, -1)`
    while keeping X/Y center preserved and dropping min Z cleanly onto `bed_z`.
    """
    n = np.array(face_normal, dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        raise ValueError("Face normal vector cannot be zero")
    n /= norm
    R_align = rotation_from_two_vectors(n, np.array([0.0, 0.0, -1.0], dtype=np.float64))

    new_m = np.eye(4, dtype=np.float64)
    new_m[:3, :3] = R_align @ current_matrix[:3, :3]

    # Preserve XY center, seat Z bottom at bed_z
    old_min, old_max = mesh.bounds(current_matrix)
    old_center_xy = 0.5 * (old_min[:2] + old_max[:2])

    new_min_no_t, new_max_no_t = mesh.bounds(new_m)
    new_center_xy = 0.5 * (new_min_no_t[:2] + new_max_no_t[:2])

    new_m[0, 3] = float(old_center_xy[0] - new_center_xy[0])
    new_m[1, 3] = float(old_center_xy[1] - new_center_xy[1])
    new_m[2, 3] = float(bed_z - new_min_no_t[2])
    return new_m


def arrange_bounding_boxes_2d(
    sizes_xy: list[tuple[float, float]],
    bed_width: float = 256.0,
    bed_depth: float = 256.0,
    spacing: float = 6.0,
) -> list[tuple[float, float]]:
    """Simple deterministic shelf packing centered on Elegoo bed `[0..bed_width, 0..bed_depth]`."""
    positions: list[tuple[float, float]] = []
    cur_x = spacing
    cur_y = spacing
    row_h = 0.0
    placed_boxes: list[tuple[float, float, float, float]] = []  # cx, cy, w, d

    for w, d in sizes_xy:
        if cur_x + w + spacing > bed_width and cur_x > spacing:
            cur_x = spacing
            cur_y += row_h + spacing
            row_h = 0.0
        cx = cur_x + w * 0.5
        cy = cur_y + d * 0.5
        placed_boxes.append((cx, cy, w, d))
        cur_x += w + spacing
        row_h = max(row_h, d)

    if not placed_boxes:
        return []

    min_x = min(cx - w * 0.5 for cx, cy, w, d in placed_boxes)
    max_x = max(cx + w * 0.5 for cx, cy, w, d in placed_boxes)
    min_y = min(cy - d * 0.5 for cx, cy, w, d in placed_boxes)
    max_y = max(cy + d * 0.5 for cx, cy, w, d in placed_boxes)

    shift_x = bed_width * 0.5 - 0.5 * (min_x + max_x)
    shift_y = bed_depth * 0.5 - 0.5 * (min_y + max_y)

    for cx, cy, _w, _d in placed_boxes:
        positions.append((round(cx + shift_x, 3), round(cy + shift_y, 3)))
    return positions


def _cap_z_loops(
    segments: list[tuple[np.ndarray, np.ndarray]],
    cut_z: float,
    upward: bool,
) -> tuple[list[list[float]], list[list[int]]]:
    """Build cap triangles on z=cut_z from intersection segments using centroid fan per closed loop."""
    if not segments:
        return [], []

    def qkey(pt: np.ndarray) -> tuple[int, int]:
        return (int(round(float(pt[0]) * 1e4)), int(round(float(pt[1]) * 1e4)))

    adj: dict[tuple[int, int], list[tuple[tuple[int, int], np.ndarray]]] = {}
    pts_map: dict[tuple[int, int], np.ndarray] = {}
    for a, b in segments:
        ka, kb = qkey(a), qkey(b)
        if ka == kb:
            continue
        pts_map[ka] = a
        pts_map[kb] = b
        adj.setdefault(ka, []).append((kb, b))

    cap_verts: list[list[float]] = []
    cap_tris: list[list[int]] = []
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for start_k in list(adj.keys()):
        for nxt_k, _ in adj[start_k]:
            if (start_k, nxt_k) in visited_edges:
                continue
            loop: list[np.ndarray] = [pts_map[start_k]]
            cur_k = start_k
            edge_nxt = nxt_k
            while True:
                visited_edges.add((cur_k, edge_nxt))
                loop.append(pts_map[edge_nxt])
                cur_k = edge_nxt
                if cur_k == start_k or len(loop) > 20000:
                    break
                cands = [nb for nb, _ in adj.get(cur_k, []) if (cur_k, nb) not in visited_edges]
                if not cands:
                    break
                edge_nxt = cands[0]

            if len(loop) < 4:
                continue
            pts_loop = np.array(loop[:-1], dtype=np.float64)
            centroid = pts_loop.mean(axis=0)
            centroid[2] = cut_z
            c_idx = len(cap_verts)
            cap_verts.append([float(centroid[0]), float(centroid[1]), float(cut_z)])
            ring_start = len(cap_verts)
            for p in pts_loop:
                cap_verts.append([float(p[0]), float(p[1]), float(cut_z)])
            m = len(pts_loop)
            for i in range(m):
                i0 = ring_start + i
                i1 = ring_start + ((i + 1) % m)
                # Ensure consistent winding along +/-Z
                p0 = pts_loop[i] - centroid
                p1 = pts_loop[(i + 1) % m] - centroid
                nz = float(p0[0] * p1[1] - p0[1] * p1[0])
                want_pos = upward
                if (nz >= 0) == want_pos:
                    cap_tris.append([c_idx, i0, i1])
                else:
                    cap_tris.append([c_idx, i1, i0])
    return cap_verts, cap_tris


def split_mesh_by_z(
    mesh: TriangleMeshData,
    matrix: np.ndarray,
    cut_z_mm: float,
) -> tuple[TriangleMeshData, TriangleMeshData]:
    """
    Slice mesh in transformed coordinates at horizontal plane `z = cut_z_mm`
    and return `(lower_mesh, upper_mesh)`, each centered at XY=0, min_z=0.
    """
    wpts = (mesh.vertices @ matrix[:3, :3].T) + matrix[:3, 3]
    eps = 1e-6

    low_verts: list[list[float]] = []
    low_tris: list[list[int]] = []
    up_verts: list[list[float]] = []
    up_tris: list[list[int]] = []
    cut_segments_low: list[tuple[np.ndarray, np.ndarray]] = []
    cut_segments_up: list[tuple[np.ndarray, np.ndarray]] = []

    def add_tri(vlist: list[list[float]], tlist: list[list[int]], a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        base = len(vlist)
        vlist.append([float(a[0]), float(a[1]), float(a[2])])
        vlist.append([float(b[0]), float(b[1]), float(b[2])])
        vlist.append([float(c[0]), float(c[1]), float(c[2])])
        tlist.append([base, base + 1, base + 2])

    def interp_z(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        dz = float(p2[2] - p1[2])
        if abs(dz) < 1e-12:
            p = p1.copy()
            p[2] = cut_z_mm
            return p
        t = (cut_z_mm - float(p1[2])) / dz
        out = p1 + t * (p2 - p1)
        out[2] = cut_z_mm
        return out

    for t in mesh.triangles:
        tri_pts = [wpts[int(t[0])], wpts[int(t[1])], wpts[int(t[2])]]
        above = [float(p[2]) > cut_z_mm + eps for p in tri_pts]
        below = [float(p[2]) < cut_z_mm - eps for p in tri_pts]

        if not any(above):
            add_tri(low_verts, low_tris, tri_pts[0], tri_pts[1], tri_pts[2])
            continue
        if not any(below):
            add_tri(up_verts, up_tris, tri_pts[0], tri_pts[1], tri_pts[2])
            continue

        # Straddling plane: rotate indices so index 0 is solo vertex
        if sum(above) == 1:
            solo = above.index(True)
            p_solo = tri_pts[solo]
            p_b = tri_pts[(solo + 1) % 3]
            p_c = tri_pts[(solo + 2) % 3]
            q1 = interp_z(p_solo, p_b)
            q2 = interp_z(p_solo, p_c)
            add_tri(up_verts, up_tris, p_solo, q1, q2)
            add_tri(low_verts, low_tris, p_b, p_c, q2)
            add_tri(low_verts, low_tris, p_b, q2, q1)
            cut_segments_low.append((q1, q2))
            cut_segments_up.append((q2, q1))
        else:
            solo = below.index(True)
            p_solo = tri_pts[solo]
            p_b = tri_pts[(solo + 1) % 3]
            p_c = tri_pts[(solo + 2) % 3]
            q1 = interp_z(p_solo, p_b)
            q2 = interp_z(p_solo, p_c)
            add_tri(low_verts, low_tris, p_solo, q1, q2)
            add_tri(up_verts, up_tris, p_b, p_c, q2)
            add_tri(up_verts, up_tris, p_b, q2, q1)
            cut_segments_up.append((q1, q2))
            cut_segments_low.append((q2, q1))

    if not low_tris or not up_tris:
        raise ValueError(f"Plane z={cut_z_mm}mm does not split the object into two non-empty parts.")

    cv_l, ct_l = _cap_z_loops(cut_segments_low, cut_z_mm, upward=True)
    if cv_l:
        off = len(low_verts)
        low_verts.extend(cv_l)
        low_tris.extend([[a + off, b + off, c + off] for a, b, c in ct_l])

    cv_u, ct_u = _cap_z_loops(cut_segments_up, cut_z_mm, upward=False)
    if cv_u:
        off = len(up_verts)
        up_verts.extend(cv_u)
        up_tris.extend([[a + off, b + off, c + off] for a, b, c in ct_u])

    def normalize_half(vlist: list[list[float]], tlist: list[list[int]]) -> TriangleMeshData:
        m = _deduplicate_vertices(np.array(vlist, dtype=np.float64), np.array(tlist, dtype=np.int64))
        bmin, bmax = m.bounds()
        cxy = 0.5 * (bmin[:2] + bmax[:2])
        v = m.vertices.copy()
        v[:, 0] -= cxy[0]
        v[:, 1] -= cxy[1]
        v[:, 2] -= bmin[2]
        return TriangleMeshData(v, m.triangles)

    return normalize_half(low_verts, low_tris), normalize_half(up_verts, up_tris)

