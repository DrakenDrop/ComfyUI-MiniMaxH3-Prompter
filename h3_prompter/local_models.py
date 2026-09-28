"""Find GGUF models (and their mmproj vision files) in ComfyUI/models/LLM."""

from __future__ import annotations

import os
import re

SERVER_DEFAULT = "(llama-server yang sudah jalan)"
MMPROJ_AUTO = "auto"
MMPROJ_NONE = "none (text only)"

_SPLIT_PART = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def llm_dirs(cfg: dict | None = None) -> list[str]:
    """ComfyUI/models/LLM (+ any registered 'LLM' paths + extra dirs from config.json)."""
    dirs: list[str] = []
    try:
        import folder_paths  # type: ignore

        dirs.append(os.path.join(folder_paths.models_dir, "LLM"))
        for key in ("LLM", "llm"):
            try:
                dirs.extend(folder_paths.get_folder_paths(key))
            except Exception:  # noqa: BLE001
                pass
    except ImportError:
        pass
    for d in (cfg or {}).get("extra_model_dirs", []) or []:
        dirs.append(os.path.expanduser(str(d)))
    out, seen = [], set()
    for d in dirs:
        key = os.path.normcase(os.path.abspath(d))
        if key not in seen and os.path.isdir(d):
            seen.add(key)
            out.append(os.path.abspath(d))
    return out


def _is_mmproj(name: str) -> bool:
    return "mmproj" in name.lower()


def _scan(cfg: dict | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """Returns ({label: path} for models, {label: path} for mmproj files)."""
    models: dict[str, str] = {}
    mmprojs: dict[str, str] = {}
    for root in llm_dirs(cfg):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                if not fn.lower().endswith(".gguf"):
                    continue
                m = _SPLIT_PART.search(fn)
                if m and m.group(1) != "00001":
                    continue  # llama.cpp loads the other parts from part 1
                full = os.path.join(dirpath, fn)
                label = os.path.relpath(full, root).replace("\\", "/")
                target = mmprojs if _is_mmproj(fn) else models
                if label in target and os.path.normcase(target[label]) != os.path.normcase(full):
                    label = f"{label}  [{os.path.basename(root)}]"
                target[label] = full
    return dict(sorted(models.items(), key=lambda kv: kv[0].lower())), \
        dict(sorted(mmprojs.items(), key=lambda kv: kv[0].lower()))


def model_choices(cfg: dict | None = None) -> list[str]:
    models, _ = _scan(cfg)
    return [SERVER_DEFAULT, *models.keys()]


def mmproj_choices(cfg: dict | None = None) -> list[str]:
    _, mm = _scan(cfg)
    return [MMPROJ_AUTO, MMPROJ_NONE, *mm.keys()]


def _norm(s: str) -> str:
    s = s.lower().replace(".gguf", "")
    s = re.sub(r"[-_.](ud|i?q\d[\w]*|f16|bf16|f32|q\d_k_\w+)$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def _prec(n: str) -> int:
    n = n.lower()
    return 2 if ("f16" in n and "bf16" not in n) else (1 if "bf16" in n else 0)


def _name_score(mm_file: str, stem: str) -> int:
    """>0 when the mmproj name belongs to the model (common prefix length), else 0."""
    core = _norm(os.path.basename(mm_file).lower().replace("mmproj", ""))
    if not core or len(core) < 4:
        return 0
    common = len(os.path.commonprefix([core, stem]))
    if stem.startswith(core) or core.startswith(stem) or common >= max(6, int(0.8 * len(core))):
        return common
    return 0


def _family(stem: str) -> str:
    """'qwen3827babliterated' -> 'qwen3827b' (name up to and including the size, e.g. 27b / 8b / 0.6b)."""
    m = re.match(r"^(.*?\d+b)", stem)
    return m.group(1) if m else stem


def auto_mmproj(model_path: str, cfg: dict | None = None) -> str | None:
    """Pick the mmproj that belongs to this model.

    1. an mmproj whose name matches the model (e.g. mmproj-Qwen3.8-27B-F16 for Qwen3.8-27B-ABLITERATED-Q4_K_M),
       first in the model's folder, then anywhere in models/LLM; F16 preferred over BF16 over others;
    2. the only mmproj in the model's folder (generic names like mmproj-F16.gguf, as in Unsloth repos), unless
       that folder holds models of different families;
    3. the only mmproj in all of models/LLM;
    otherwise None.
    """
    folder = os.path.dirname(model_path)
    stem = _norm(_SPLIT_PART.sub(".gguf", os.path.basename(model_path)))
    try:
        files = [f for f in os.listdir(folder) if f.lower().endswith(".gguf")]
    except OSError:
        files = []
    local_mm = [os.path.join(folder, f) for f in files if _is_mmproj(f)]
    _, all_mm_map = _scan(cfg)
    all_mm = list(all_mm_map.values())

    for pool in (local_mm, all_mm):
        scored = [(_name_score(f, stem), _prec(f), f) for f in pool]
        scored = [x for x in scored if x[0] > 0]
        if scored:
            return sorted(scored, reverse=True)[0][2]

    if local_mm:
        models = [f for f in files if not _is_mmproj(f)
                  and not (_SPLIT_PART.search(f) and _SPLIT_PART.search(f).group(1) != "00001")]
        families = {_family(_norm(_SPLIT_PART.sub(".gguf", f))) for f in models}
        if len(families) <= 1:
            return sorted(local_mm, key=_prec, reverse=True)[0]
    if len(all_mm) == 1:
        return all_mm[0]
    return None


def resolve(model_label: str, mmproj_label: str, cfg: dict | None = None) -> tuple[str | None, str | None]:
    """(model_path, mmproj_path) or (None, None) for the already-running server."""
    if not model_label or model_label == SERVER_DEFAULT:
        return None, None
    models, mmprojs = _scan(cfg)
    model_path = models.get(model_label)
    if model_path is None:
        raise FileNotFoundError(f"model '{model_label}' tidak ditemukan di {', '.join(llm_dirs(cfg)) or 'models/LLM'}")
    if mmproj_label == MMPROJ_NONE:
        return model_path, None
    if mmproj_label and mmproj_label != MMPROJ_AUTO:
        mm = mmprojs.get(mmproj_label)
        if mm is None:
            raise FileNotFoundError(f"mmproj '{mmproj_label}' tidak ditemukan")
        return model_path, mm
    return model_path, auto_mmproj(model_path, cfg)
