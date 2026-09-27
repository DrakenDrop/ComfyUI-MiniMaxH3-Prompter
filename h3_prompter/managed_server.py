"""llama-server that the node starts itself for a model picked from ComfyUI/models/LLM.

* runs on its own port (default 8090) so it never collides with a server you start by hand
* stays alive (and the model stays in VRAM) across queue runs and ComfyUI restarts
* restarted automatically only when you pick a different model / mmproj / context size
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time

from . import llama_client as lc

_PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATE_PATH = os.path.join(_PACK_DIR, ".managed_server.json")
_LOG_PATH = os.path.join(_PACK_DIR, "llama-server.log")

_proc: subprocess.Popen | None = None

DEFAULT_MANAGED_ARGS = ["--jinja", "-ngl", "999", "-np", "1"]
_help_cache: dict[str, str] = {}


def _exe_name() -> str:
    return "llama-server.exe" if os.name == "nt" else "llama-server"


def find_llama_server(cfg: dict) -> str:
    p = (cfg.get("llama_server_path") or "").strip().strip('"')
    if p:
        if os.path.isdir(p):
            p = os.path.join(p, _exe_name())
        if os.path.isfile(p):
            return p
        raise lc.ServerError(f"llama_server_path di config.json tidak ditemukan: {p}")
    found = shutil.which("llama-server")
    if found:
        return found
    home = os.path.expanduser("~")
    guesses = [
        os.path.join(_PACK_DIR, "llama.cpp"),
        os.path.join(_PACK_DIR, "llama.cpp", "bin"),
        r"C:\llama.cpp",
        os.path.join(home, "llama.cpp"),
        os.path.join(home, "llama.cpp", "build", "bin"),
    ]
    for d in guesses:
        f = os.path.join(d, _exe_name())
        if os.path.isfile(f):
            return f
    raise lc.ServerError(
        "llama-server tidak ditemukan. Isi 'llama_server_path' di config.json (mis. C:\\llama.cpp\\llama-server.exe), "
        "atau taruh folder llama.cpp di C:\\llama.cpp."
    )


def _help_text(exe: str) -> str:
    if exe not in _help_cache:
        try:
            out = subprocess.run([exe, "--help"], capture_output=True, timeout=30, stdin=subprocess.DEVNULL)
            _help_cache[exe] = (out.stdout + out.stderr).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            _help_cache[exe] = ""
    return _help_cache[exe]


def _drop_unsupported(exe: str, args: list[str]) -> list[str]:
    """Remove options this llama-server build does not know (flags get renamed/removed between releases,
    e.g. --no-mmap was replaced by --load-mode). Keeps everything when --help can't be read."""
    help_text = _help_text(exe)
    if "--port" not in help_text or "--ctx-size" not in help_text:
        return args  # not a recognizable llama-server help page -> don't touch the args
    known = set(re.findall(r"(?<![\w-])(-{1,2}[a-zA-Z][\w-]*)", help_text))
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("-") and not re.match(r"^-\d", a) and a.split("=")[0] not in known:
            lc.log(f"llama-server build does not support '{a}', skipping it.")
            if "=" not in a and i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 1  # skip its value too
            i += 1
            continue
        out.append(a)
        i += 1
    return out


# ------------------------------------------------------------------ state

def _read_state() -> dict:
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def _write_state(state: dict) -> None:
    try:
        with open(_STATE_PATH, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1)
    except OSError as exc:
        lc.log(f"could not write {_STATE_PATH}: {exc}")


def _norm_path(p: str | None) -> str:
    return os.path.normcase(os.path.abspath(p)) if p else ""


def _signature(model: str, mmproj: str | None, ctx: int) -> dict:
    return {"model": _norm_path(model), "mmproj": _norm_path(mmproj), "ctx": int(ctx)}


def _props_model(url: str) -> str | None:
    try:
        _, body = lc._get(url.rstrip("/") + "/props", timeout=3.0)
        data = json.loads(body.decode("utf-8", "replace"))
        mp = data.get("model_path") or (data.get("default_generation_settings") or {}).get("model")
        return _norm_path(mp) if mp else None
    except Exception:  # noqa: BLE001
        return None


def _pid_is_llama(pid: int) -> bool:
    try:
        import psutil  # type: ignore

        p = psutil.Process(pid)
        return "llama-server" in (p.name() or "").lower()
    except Exception:  # noqa: BLE001
        return False


def _kill(pid: int) -> None:
    try:
        import psutil  # type: ignore

        p = psutil.Process(pid)
        p.terminate()
        try:
            p.wait(timeout=15)
        except psutil.TimeoutExpired:
            p.kill()
    except Exception as exc:  # noqa: BLE001
        lc.log(f"could not stop llama-server pid {pid}: {exc}")


# ------------------------------------------------------------------ public

def managed_url(cfg: dict) -> str:
    return f"http://127.0.0.1:{int(cfg.get('managed_port', 8090))}"


def stop(cfg: dict) -> bool:
    """Stop the llama-server this pack started (frees VRAM)."""
    global _proc
    stopped = False
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _proc.kill()
        stopped = True
    _proc = None
    state = _read_state()
    pid = state.get("pid")
    if pid and _pid_is_llama(int(pid)):
        _kill(int(pid))
        stopped = True
    if state:
        _write_state({})
    url = managed_url(cfg)
    for _ in range(30):
        if not lc.server_ready(url):
            break
        time.sleep(0.5)
    return stopped


def ensure(model_path: str, mmproj_path: str | None, ctx: int, cfg: dict) -> str:
    """Make sure the managed server runs `model_path` (+ mmproj). Returns its URL."""
    global _proc
    url = managed_url(cfg)
    want = _signature(model_path, mmproj_path, ctx)
    state = _read_state()

    # a server we started earlier may still be loading (e.g. ComfyUI restarted meanwhile)
    pid = state.get("pid")
    if state.get("sig") == want and pid and not lc.server_ready(url) and _pid_is_llama(int(pid)):
        lc.log("managed llama-server is still loading this model, waiting ...")
        return _wait_ready(url, model_path, cfg, proc=None)

    if lc.server_ready(url):
        running_model = _props_model(url)
        if state.get("sig") == want and running_model in (None, want["model"]):
            return url
        if not state and running_model == want["model"]:
            return url
        lc.log("switching model: stopping the running managed llama-server ...")
        if not stop(cfg) and lc.server_ready(url):
            raise lc.ServerError(
                f"port {url} dipakai llama-server lain yang tidak dijalankan oleh node ini. "
                "Tutup server itu atau ganti 'managed_port' di config.json."
            )

    exe = find_llama_server(cfg)
    port = str(int(cfg.get("managed_port", 8090)))
    args = _drop_unsupported(exe, [str(a) for a in (cfg.get("managed_server_args") or DEFAULT_MANAGED_ARGS)])
    cmd = [exe, "-m", model_path]
    if mmproj_path:
        cmd += ["--mmproj", mmproj_path]
    cmd += ["-c", str(int(ctx)), "--host", "127.0.0.1", "--port", port,
            "--alias", os.path.splitext(os.path.basename(model_path))[0], *[str(a) for a in args]]
    lc.log("starting llama-server: " + " ".join(f'"{c}"' if " " in c else c for c in cmd))

    # every platform logs to llama-server.log, so a failed start can be diagnosed from the error itself
    log_fh = open(_LOG_PATH, "wb")  # noqa: SIM115  (fresh log per start)
    log_fh.write(("$ " + " ".join(cmd) + "\n").encode("utf-8", "replace"))
    log_fh.flush()
    kwargs: dict = {"stdout": log_fh, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        # no console window; detached so it survives ComfyUI restarts (model stays in VRAM)
        kwargs["creationflags"] = (subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
                                   | subprocess.CREATE_NEW_PROCESS_GROUP)  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    _proc = subprocess.Popen(cmd, **kwargs)
    _write_state({"pid": _proc.pid, "sig": want, "port": int(port), "cmd": cmd})

    return _wait_ready(url, model_path, cfg, proc=_proc)


_HINTS = [
    (("unknown model architecture", "unknown architecture"),
     "build llama.cpp terlalu lama untuk arsitektur model ini -> update llama.cpp ke release terbaru."),
    (("couldn't bind", "address already in use", "bind:"),
     "port sudah dipakai -> matikan server lain di port itu atau ganti 'managed_port' di config.json."),
    (("out of memory", "cudamalloc failed", "failed to allocate", "unable to allocate"),
     "VRAM tidak cukup -> kecilkan context_size, pakai quant lebih kecil, atau bebaskan VRAM."),
    (("failed to load mmproj", "clip_init: failed", "failed to load multimodal", "mtmd_init_from_file: error",
      "failed to load vision"),
     "mmproj tidak cocok / tidak didukung -> pilih mmproj lain atau 'none (text only)'."),
    (("error while handling argument", "invalid argument", "unknown argument", "error: invalid"),
     "argumen tidak dikenal oleh build llama.cpp ini -> update llama.cpp atau ubah 'managed_server_args' di config.json."),
    (("failed to load model", "error loading model", "gguf_init", "invalid magic"),
     "file model tidak bisa dibaca (rusak / belum selesai download / format tidak didukung build ini)."),
    (("no such file", "cannot open shared object", "libcuda", "libcudart"),
     "library CUDA atau file tidak ditemukan -> pakai build llama.cpp yang cocok dengan CUDA di mesin ini."),
]


def _log_tail(n: int = 25) -> str:
    try:
        with open(_LOG_PATH, "rb") as fh:
            lines = fh.read().decode("utf-8", "replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return ""


def _explain_failure(code) -> str:
    tail = _log_tail()
    low = tail.lower()
    hint = next((h for keys, h in _HINTS if any(k in low for k in keys)), None)
    msg = f"llama-server berhenti (exit code {code}) saat start."
    if hint:
        msg += f" Kemungkinan: {hint}"
    msg += f"\nLog lengkap: {_LOG_PATH}\n--- akhir log ---\n{tail or '(kosong)'}"
    return msg


def _wait_ready(url: str, model_path: str, cfg: dict, proc) -> str:
    deadline = time.time() + float(cfg.get("startup_wait_seconds", 900))
    t0 = time.time()
    last_note = 0.0
    while time.time() < deadline:
        if lc.server_ready(url):
            lc.log(f"llama-server ready in {time.time() - t0:.0f}s ({os.path.basename(model_path)}).")
            return url
        if proc is not None and proc.poll() is not None:
            _write_state({})
            raise lc.ServerError(_explain_failure(proc.returncode))
        if lc._interrupted():
            lc._raise_if_interrupted()
        if time.time() - last_note > 15:
            lc.log(f"loading model ... {time.time() - t0:.0f}s")
            last_note = time.time()
        time.sleep(1.0)
    raise lc.ServerError("llama-server belum siap setelah batas waktu (startup_wait_seconds).")
