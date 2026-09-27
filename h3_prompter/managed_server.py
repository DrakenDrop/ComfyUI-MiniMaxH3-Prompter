"""llama-server that the node starts itself for a model picked from ComfyUI/models/LLM.

* runs on its own port (default 8090) so it never collides with a server you start by hand
* stays alive (and the model stays in VRAM) across queue runs and ComfyUI restarts
* restarted automatically only when you pick a different model / mmproj / context size
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

from . import llama_client as lc

_PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATE_PATH = os.path.join(_PACK_DIR, ".managed_server.json")
_LOG_PATH = os.path.join(_PACK_DIR, "llama-server.log")

_proc: subprocess.Popen | None = None

DEFAULT_MANAGED_ARGS = ["--jinja", "-ngl", "999", "-np", "1", "--no-mmap"]


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
    args = cfg.get("managed_server_args") or DEFAULT_MANAGED_ARGS
    cmd = [exe, "-m", model_path]
    if mmproj_path:
        cmd += ["--mmproj", mmproj_path]
    cmd += ["-c", str(int(ctx)), "--host", "127.0.0.1", "--port", port,
            "--alias", os.path.splitext(os.path.basename(model_path))[0], *[str(a) for a in args]]
    lc.log("starting llama-server: " + " ".join(f'"{c}"' if " " in c else c for c in cmd))

    kwargs: dict = {}
    if os.name == "nt":
        # own console window: shows llama.cpp logs and survives ComfyUI restarts (model stays in VRAM)
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
        log_fh = open(_LOG_PATH, "ab")  # noqa: SIM115
        kwargs["stdout"] = log_fh
        kwargs["stderr"] = subprocess.STDOUT
    _proc = subprocess.Popen(cmd, **kwargs)
    _write_state({"pid": _proc.pid, "sig": want, "port": int(port), "cmd": cmd})

    return _wait_ready(url, model_path, cfg, proc=_proc)


def _wait_ready(url: str, model_path: str, cfg: dict, proc) -> str:
    deadline = time.time() + float(cfg.get("startup_wait_seconds", 900))
    t0 = time.time()
    last_note = 0.0
    while time.time() < deadline:
        if lc.server_ready(url):
            lc.log(f"llama-server ready in {time.time() - t0:.0f}s ({os.path.basename(model_path)}).")
            return url
        if proc is not None and proc.poll() is not None:
            where = "jendela console llama-server" if os.name == "nt" else _LOG_PATH
            _write_state({})
            raise lc.ServerError(
                f"llama-server berhenti (exit code {proc.returncode}) saat memuat model. Lihat {where}. "
                "Penyebab umum: file mmproj tidak cocok dengan model, build llama.cpp terlalu lama untuk arsitektur "
                "model ini, atau VRAM penuh."
            )
        if lc._interrupted():
            lc._raise_if_interrupted()
        if time.time() - last_note > 15:
            lc.log(f"loading model ... {time.time() - t0:.0f}s")
            last_note = time.time()
        time.sleep(1.0)
    raise lc.ServerError("llama-server belum siap setelah batas waktu (startup_wait_seconds).")
