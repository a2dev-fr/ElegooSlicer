"""Full parameter & profile registry parsed from PrintConfig.cpp, French .po, and JSON profiles."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_accents(text).lower()).strip()


def _unescape_c_str(raw: str) -> str:
    return (
        raw.replace(r"\"", '"')
        .replace(r"\n", "\n")
        .replace(r"\t", "\t")
        .replace(r"\\", "\\")
    )


@dataclass
class ParamDef:
    key: str
    opt_type: str
    label_en: str = ""
    label_fr: str = ""
    full_label_en: str = ""
    full_label_fr: str = ""
    category_en: str = ""
    category_fr: str = ""
    tooltip_en: str = ""
    tooltip_fr: str = ""
    sidetext_en: str = ""
    sidetext_fr: str = ""
    min_val: float | None = None
    max_val: float | None = None
    cli: str = ""
    enum_values: list[str] = field(default_factory=list)
    enum_labels_en: list[str] = field(default_factory=list)
    enum_labels_fr: list[str] = field(default_factory=list)
    default_raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "type": self.opt_type,
            "label_en": self.label_en,
            "label_fr": self.label_fr,
            "full_label_en": self.full_label_en,
            "full_label_fr": self.full_label_fr,
            "category_en": self.category_en,
            "category_fr": self.category_fr,
            "unit_en": self.sidetext_en,
            "unit_fr": self.sidetext_fr,
            "min": self.min_val,
            "max": self.max_val,
            "cli": self.cli,
            "enum_values": self.enum_values,
            "enum_labels_en": self.enum_labels_en,
            "enum_labels_fr": self.enum_labels_fr,
            "default": self.default_raw,
            "tooltip_fr": self.tooltip_fr,
            "tooltip_en": self.tooltip_en,
        }


def parse_po_file(po_path: Path) -> dict[str, str]:
    """Parse GNU gettext .po file into `{msgid: msgstr}`."""
    translations: dict[str, str] = {}
    if not po_path.exists():
        return translations

    state: str | None = None
    msgid_parts: list[str] = []
    msgstr_parts: list[str] = []
    str_re = re.compile(r'^"(.*)"\s*$')

    def flush() -> None:
        nonlocal msgid_parts, msgstr_parts
        if msgid_parts and msgstr_parts:
            msgid = _unescape_c_str("".join(msgid_parts))
            msgstr = _unescape_c_str("".join(msgstr_parts))
            if msgid and msgstr:
                translations[msgid] = msgstr
        msgid_parts = []
        msgstr_parts = []

    with po_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                if not line:
                    flush()
                    state = None
                continue
            if line.startswith("msgid "):
                flush()
                state = "id"
                m = str_re.match(line[6:].strip())
                if m:
                    msgid_parts.append(m.group(1))
            elif line.startswith("msgstr "):
                state = "str"
                m = str_re.match(line[7:].strip())
                if m:
                    msgstr_parts.append(m.group(1))
            elif line.startswith("msgstr[0] "):
                state = "str"
                m = str_re.match(line[10:].strip())
                if m:
                    msgstr_parts.append(m.group(1))
            elif line.startswith('"'):
                m = str_re.match(line)
                if m:
                    if state == "id":
                        msgid_parts.append(m.group(1))
                    elif state == "str":
                        msgstr_parts.append(m.group(1))
        flush()
    return translations


def _extract_quoted(expr: str) -> str:
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', expr)
    return _unescape_c_str("".join(parts))


class ConfigRegistry:
    """Registry of ElegooSlicer options (with FR/EN labels) and profiles."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or REPO_ROOT
        self.params: dict[str, ParamDef] = {}
        self.fr_map: dict[str, str] = {}
        self._label_index: dict[str, list[str]] = {}
        self._loaded_params = False
        self._profile_dirs = self._discover_profile_dirs()
        self._user_dirs = self._discover_user_dirs()
        self._profiles_cache: dict[str, dict[str, dict[str, Any]]] | None = None

    @staticmethod
    def _discover_profile_dirs() -> list[Path]:
        candidates: list[Path] = []
        env_dir = os.environ.get("ELEGOO_SLICER_PROFILES_DIR")
        if env_dir:
            candidates.append(Path(env_dir))
        if os.name == "nt":
            pf = os.environ.get("ProgramFiles", r"C:\Program Files")
            candidates.append(Path(pf) / "ElegooSlicer" / "resources" / "profiles")
        candidates.append(REPO_ROOT / "resources" / "profiles")
        return [p for p in candidates if p.exists()]

    @staticmethod
    def _discover_user_dirs() -> list[Path]:
        dirs: list[Path] = []
        env_user = os.environ.get("ELEGOO_SLICER_USER_DIR")
        if env_user:
            dirs.append(Path(env_user))
        if os.name == "nt":
            appdata = Path(os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming"))
            user_root = appdata / "ElegooSlicer" / "user"
        else:
            xdg = Path(os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config"))
            user_root = xdg / "ElegooSlicer" / "user"
        if user_root.exists():
            for child in user_root.iterdir():
                if child.is_dir():
                    dirs.append(child)
        return dirs

    def ensure_params_loaded(self) -> None:
        if self._loaded_params:
            return
        self._loaded_params = True

        po_path = self.repo_root / "localization" / "i18n" / "fr" / "ElegooSlicer_fr.po"
        self.fr_map = parse_po_file(po_path)

        cpp_path = self.repo_root / "src" / "libslic3r" / "PrintConfig.cpp"
        if not cpp_path.exists():
            return

        add_re = re.compile(r'def\s*=\s*this->add\(\s*"([^"]+)"\s*,\s*(co[A-Za-z0-9_]+)\s*\)')
        field_re = re.compile(r'def->([a-zA-Z0-9_]+)\s*=\s*(.+?);')
        push_re = re.compile(r'def->(enum_values|enum_labels)\.push_back\((.+?)\)\s*;')
        default_re = re.compile(r'def->set_default_value\(\s*new\s+ConfigOption[A-Za-z0-9_]*\((.+)\)\s*\)\s*;')

        current: ParamDef | None = None

        def tr(en_text: str) -> str:
            if not en_text:
                return ""
            return self.fr_map.get(en_text, en_text)

        def finalize_param(p: ParamDef | None) -> None:
            if p is None or p.key in self.params and not p.label_en:
                return
            if p.key not in self.params or p.label_en:
                p.label_fr = tr(p.label_en)
                p.full_label_fr = tr(p.full_label_en)
                p.category_fr = tr(p.category_en)
                p.tooltip_fr = tr(p.tooltip_en)
                p.sidetext_fr = tr(p.sidetext_en)
                p.enum_labels_fr = [tr(lbl) for lbl in p.enum_labels_en]
                self.params[p.key] = p

                for candidate in (p.key, p.label_en, p.label_fr, p.full_label_en, p.full_label_fr):
                    norm = _norm_text(candidate)
                    if norm:
                        bucket = self._label_index.setdefault(norm, [])
                        if p.key not in bucket:
                            bucket.append(p.key)

        with cpp_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                m_add = add_re.search(stripped)
                if m_add:
                    finalize_param(current)
                    current = ParamDef(key=m_add.group(1), opt_type=m_add.group(2))
                    continue
                if current is None:
                    continue
                m_push = push_re.search(stripped)
                if m_push:
                    val = _extract_quoted(m_push.group(2))
                    if m_push.group(1) == "enum_values":
                        current.enum_values.append(val)
                    else:
                        current.enum_labels_en.append(val)
                    continue
                m_def = default_re.search(stripped)
                if m_def:
                    current.default_raw = m_def.group(1).strip()
                    continue
                m_field = field_re.search(stripped)
                if m_field:
                    fname, fval = m_field.group(1), m_field.group(2).strip()
                    if fname == "label":
                        current.label_en = _extract_quoted(fval)
                    elif fname == "full_label":
                        current.full_label_en = _extract_quoted(fval)
                    elif fname == "category":
                        current.category_en = _extract_quoted(fval)
                    elif fname == "tooltip":
                        current.tooltip_en = _extract_quoted(fval)
                    elif fname == "sidetext":
                        current.sidetext_en = _extract_quoted(fval)
                    elif fname == "cli":
                        current.cli = _extract_quoted(fval)
                    elif fname == "min":
                        try:
                            current.min_val = float(fval.rstrip("f"))
                        except ValueError:
                            pass
                    elif fname == "max":
                        try:
                            current.max_val = float(fval.rstrip("f"))
                        except ValueError:
                            pass
        finalize_param(current)

    def resolve_key(self, name_or_key: str) -> str:
        """Resolve an option key, French label, English label, or 'Group/Category > Label' to canonical PrintConfig key."""
        self.ensure_params_loaded()
        if name_or_key in self.params:
            return name_or_key
        norm = _norm_text(name_or_key)
        # Allow composite names like "Qualité > Paroi extérieure" or "Largeur de ligne > Paroi extérieure"
        if ">" in norm:
            parts = [p.strip() for p in norm.split(">") if p.strip()]
            tail = parts[-1]
            prefix = " ".join(parts[:-1])
            candidates = self._label_index.get(tail, [])
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                prefix_words = set(re.findall(r"[a-z0-9]+", prefix))
                best_key = candidates[0]
                best_score = -1
                for cand in candidates:
                    pdef = self.params[cand]
                    meta_blob = _norm_text(
                        f"{pdef.category_fr} {pdef.category_en} {pdef.full_label_fr} {pdef.full_label_en} {cand.replace('_', ' ')}"
                    )
                    score = sum(2 if w in meta_blob else 0 for w in prefix_words)
                    # Bonus when prefix explicitly names line width / largeur / speed / vitesse
                    if ("largeur" in prefix_words or "width" in prefix_words) and "line_width" in cand:
                        score += 5
                    elif ("vitesse" in prefix_words or "speed" in prefix_words) and cand.endswith("_speed"):
                        score += 5
                    elif ("acceleration" in prefix_words) and "acceleration" in cand:
                        score += 5
                    if score > best_score:
                        best_score = score
                        best_key = cand
                return best_key
        if norm in self._label_index:
            keys = self._label_index[norm]
            return keys[0]
        return name_or_key

    def normalize_value(self, key: str, value: Any) -> Any:
        """Normalize French/English enum labels, comma decimals, booleans, and percentages."""
        self.ensure_params_loaded()
        pdef = self.params.get(key)

        if isinstance(value, bool):
            return "1" if value else "0"

        if isinstance(value, (int, float)):
            if pdef and pdef.opt_type == "coBool":
                return "1" if value else "0"
            return str(value)

        if isinstance(value, list):
            return [self.normalize_value(key, v) for v in value]

        s = str(value).strip()
        if pdef:
            if pdef.opt_type == "coBool":
                low = _norm_text(s)
                if low in {"1", "true", "oui", "yes", "on", "actif", "active"}:
                    return "1"
                if low in {"0", "false", "non", "no", "off", "aucun", "inactif"}:
                    return "0"
            if pdef.enum_values:
                if s in pdef.enum_values:
                    return s
                low = _norm_text(s)
                for idx, ev in enumerate(pdef.enum_values):
                    if _norm_text(ev) == low:
                        return ev
                    if idx < len(pdef.enum_labels_en) and _norm_text(pdef.enum_labels_en[idx]) == low:
                        return ev
                    if idx < len(pdef.enum_labels_fr) and _norm_text(pdef.enum_labels_fr[idx]) == low:
                        return ev

        # Handle French decimal commas like "0,42" -> "0.42"
        if re.fullmatch(r"-?\d+,\d+(%?)", s):
            return s.replace(",", ".")
        return s

    def search_params(
        self,
        query: str = "",
        category: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search all print/filament/printer config options by FR/EN keyword or category."""
        self.ensure_params_loaded()
        q_norm = _norm_text(query)
        c_norm = _norm_text(category)

        scored: list[tuple[int, ParamDef]] = []
        for pdef in self.params.values():
            if c_norm:
                cat_blob = _norm_text(f"{pdef.category_en} {pdef.category_fr}")
                if c_norm not in cat_blob:
                    continue
            if not q_norm:
                scored.append((1, pdef))
                continue

            exact = any(
                _norm_text(x) == q_norm
                for x in (pdef.key, pdef.label_en, pdef.label_fr, pdef.full_label_en, pdef.full_label_fr)
            )
            if exact:
                scored.append((100, pdef))
                continue

            hay_primary = _norm_text(
                f"{pdef.key} {pdef.label_en} {pdef.label_fr} {pdef.full_label_en} {pdef.full_label_fr}"
            )
            if q_norm in hay_primary:
                scored.append((50, pdef))
                continue

            hay_sec = _norm_text(f"{pdef.tooltip_en} {pdef.tooltip_fr} {' '.join(pdef.enum_labels_fr)}")
            if q_norm in hay_sec:
                scored.append((10, pdef))

        scored.sort(key=lambda item: (-item[0], item[1].key))
        return [item[1].to_dict() for item in scored[: max(1, limit)]]

    def list_categories(self) -> list[dict[str, str]]:
        """List all PrintConfig categories with EN and FR names."""
        self.ensure_params_loaded()
        cats: dict[str, str] = {}
        for pdef in self.params.values():
            if pdef.category_en:
                cats[pdef.category_en] = pdef.category_fr or pdef.category_en
        return [{"en": k, "fr": v} for k, v in sorted(cats.items())]

    # ---- Profile loading & inheritance resolution ----

    def _scan_profiles(self) -> dict[str, dict[str, dict[str, Any]]]:
        if self._profiles_cache is not None:
            return self._profiles_cache

        raw_by_kind: dict[str, dict[str, dict[str, Any]]] = {
            "print": {},
            "filament": {},
            "printer": {},
        }

        def ingest_dir(root: Path, kind_hint: str | None = None, source_tag: str = "system") -> None:
            if not root.exists():
                return
            for json_file in root.rglob("*.json"):
                if json_file.name.endswith(".info") or "bundle" in json_file.name.lower():
                    continue
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                ptype = data.get("type") or kind_hint or ""
                if ptype == "process":
                    ptype = "print"
                elif ptype == "machine":
                    ptype = "printer"
                elif ptype == "machine_model":
                    continue
                if ptype not in raw_by_kind:
                    # Infer from folder path
                    parts = {part.lower() for part in json_file.parts}
                    if "process" in parts or "print" in parts:
                        ptype = "print"
                    elif "filament" in parts:
                        ptype = "filament"
                    elif "machine" in parts or "printer" in parts:
                        ptype = "printer"
                    else:
                        continue
                name = data.get("name") or json_file.stem
                entry = dict(data)
                entry["_path"] = str(json_file)
                entry["_source"] = source_tag
                # User profiles override system presets with the same name
                if name not in raw_by_kind[ptype] or source_tag == "user":
                    raw_by_kind[ptype][name] = entry

        for pdir in self._profile_dirs:
            # Prefer Elegoo vendor folder first, then all others
            elegoo_dir = pdir / "Elegoo"
            if elegoo_dir.exists():
                ingest_dir(elegoo_dir, source_tag="system")
            ingest_dir(pdir, source_tag="system")

        for udir in self._user_dirs:
            for sub, k in (("process", "print"), ("print", "print"), ("filament", "filament"), ("machine", "printer")):
                sub_dir = udir / sub
                if sub_dir.exists():
                    ingest_dir(sub_dir, kind_hint=k, source_tag="user")

        self._profiles_cache = raw_by_kind
        return raw_by_kind

    def list_profiles(
        self,
        kind: str = "all",
        query: str = "",
        include_hidden: bool = False,
        limit: int = 100,
    ) -> dict[str, list[dict[str, Any]]]:
        """List available print/process, filament, and printer profiles."""
        scanned = self._scan_profiles()
        kind_map = {
            "process": "print",
            "print": "print",
            "filament": "filament",
            "printer": "printer",
            "machine": "printer",
        }
        kinds = [kind_map[kind]] if kind in kind_map else ["print", "filament", "printer"]
        q_norm = _norm_text(query)
        out: dict[str, list[dict[str, Any]]] = {}

        for k in kinds:
            items: list[dict[str, Any]] = []
            for name, raw in scanned[k].items():
                if not include_hidden and str(raw.get("instantiation", "true")).lower() == "false":
                    continue
                if q_norm and q_norm not in _norm_text(name):
                    continue
                items.append(
                    {
                        "name": name,
                        "kind": k,
                        "source": raw.get("_source", "system"),
                        "inherits": raw.get("inherits", ""),
                        "path": raw.get("_path", ""),
                        "filament_type": raw.get("filament_type"),
                        "default_filament_colour": raw.get("default_filament_colour"),
                    }
                )
            items.sort(key=lambda x: (0 if x["source"] == "user" else 1, x["name"].lower()))
            out[k] = items[: max(1, limit)]
        return out

    def resolve_profile(self, kind: str, name: str) -> dict[str, Any]:
        """Return merged profile dictionary with full `inherits` chain resolved."""
        scanned = self._scan_profiles()
        kind_norm = {"process": "print", "machine": "printer"}.get(kind, kind)
        pool = scanned.get(kind_norm, {})
        if name not in pool:
            # Case-insensitive or partial fallback
            low = _norm_text(name)
            matches = [k for k in pool if _norm_text(k) == low]
            if not matches:
                matches = [k for k in pool if low in _norm_text(k) and str(pool[k].get("instantiation", "true")).lower() != "false"]
            if not matches:
                raise KeyError(f"Profile '{name}' of kind '{kind_norm}' not found")
            name = matches[0]

        merged: dict[str, Any] = {}
        visited: set[str] = set()

        def apply_recursive(cur_name: str) -> None:
            if not cur_name or cur_name in visited:
                return
            visited.add(cur_name)
            cur = pool.get(cur_name)
            if not cur:
                return
            parent = cur.get("inherits", "")
            if parent:
                apply_recursive(parent)
            for k, v in cur.items():
                if not k.startswith("_"):
                    merged[k] = v

        apply_recursive(name)
        merged["name"] = name
        merged["_kind"] = kind_norm
        merged["_path"] = pool[name].get("_path", "")
        return merged

    def export_profile_json(self, kind: str, name: str, dest_path: Path) -> Path:
        """Write a resolved standalone JSON profile file compatible with ElegooSlicer `--load-settings`."""
        resolved = self.resolve_profile(kind, name)
        clean = {k: v for k, v in resolved.items() if not k.startswith("_")}
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
        return dest_path

    def save_user_profile(
        self,
        kind: str,
        name: str,
        overrides: dict[str, Any],
        inherits_from: str = "",
        base_id: str = "",
    ) -> dict[str, Any]:
        """Save a persistent ElegooSlicer user preset (.json + .info) into user/default/<subdir>."""
        import time

        kind_norm = {"process": "print", "machine": "printer"}.get(kind.lower(), kind.lower())
        subdir_map = {"print": "process", "filament": "filament", "printer": "machine"}
        if kind_norm not in subdir_map:
            raise ValueError(f"Unsupported profile kind: {kind}")

        user_default = self._get_writable_user_default_dir()
        target_dir = user_default / subdir_map[kind_norm]
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_name = re.sub(r'[<>:"/\\|?*]+', "_", name).strip()
        if not safe_name:
            raise ValueError("Profile name cannot be empty")

        json_path = target_dir / f"{safe_name}.json"
        info_path = target_dir / f"{safe_name}.info"

        payload: dict[str, Any] = {
            "from": "User",
            "name": name,
            "version": "1.5.3.5",
        }
        if inherits_from:
            payload["inherits"] = inherits_from
        id_field = {"print": "print_settings_id", "filament": "filament_settings_id", "printer": "printer_settings_id"}[kind_norm]
        payload[id_field] = name

        for raw_k, raw_v in overrides.items():
            key = self.resolve_key(raw_k)
            payload[key] = self.normalize_value(key, raw_v)

        json_path.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")
        info_lines = [
            "sync_info = ",
            "user_id = ",
            "setting_id = ",
            f"base_id = {base_id}",
            f"updated_time = {int(time.time())}",
        ]
        info_path.write_text("\n".join(info_lines) + "\n", encoding="utf-8")

        # Refresh discovery cache
        if user_default not in self._user_dirs:
            self._user_dirs.append(user_default)
        self._profiles_cache = None
        return {
            "name": name,
            "kind": kind_norm,
            "json_path": str(json_path),
            "info_path": str(info_path),
            "inherits": inherits_from,
        }

    def delete_user_profile(self, kind: str, name: str) -> dict[str, Any]:
        """Delete a user preset (.json + .info) from the user profile folder."""
        kind_norm = {"process": "print", "machine": "printer"}.get(kind.lower(), kind.lower())
        subdir_map = {"print": "process", "filament": "filament", "printer": "machine"}
        if kind_norm not in subdir_map:
            raise ValueError(f"Unsupported profile kind: {kind}")
        safe_name = re.sub(r'[<>:"/\\|?*]+', "_", name).strip()
        removed: list[str] = []
        for udir in self._user_dirs:
            for ext in (".json", ".info"):
                p = udir / subdir_map[kind_norm] / f"{safe_name}{ext}"
                if p.exists():
                    p.unlink()
                    removed.append(str(p))
        self._profiles_cache = None
        return {"name": name, "kind": kind_norm, "removed_files": removed, "deleted": bool(removed)}

    def _get_writable_user_default_dir(self) -> Path:
        env_user = os.environ.get("ELEGOO_SLICER_USER_DIR")
        if env_user:
            return Path(env_user)
        if os.name == "nt":
            appdata = Path(os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming"))
            return appdata / "ElegooSlicer" / "user" / "default"
        xdg = Path(os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config"))
        return xdg / "ElegooSlicer" / "user" / "default"
