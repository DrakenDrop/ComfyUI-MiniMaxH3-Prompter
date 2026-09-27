"""Minimal, dependency-free client for llama.cpp's llama-server (OpenAI-compatible API).

Design goals:
  * zero extra pip dependencies (urllib only)
  * streaming, so ComfyUI's Cancel button stops generation immediately
  * thinking OFF really means OFF (template kwargs + assistant prefill + abort guard)
  * optional auto-start of llama-server so the model stays resident in VRAM
"""

from __future__ import annotations

import http.client
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

_PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_PACK_DIR, "config.json")

DEFAULT_CONFIG = {
    "server_url": "http://127.0.0.1:8080",
    "model_alias": "qwen3.8-27b",
    "request_timeout_seconds": 600,
    "autostart": False,
    "llama_server_path": "",
    "llama_server_args": [
        "-hf", "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL",
        "--jinja", "-ngl", "999", "-c", "32768", "-np", "1",
        "--host", "127.0.0.1", "--port", "8080",
        "--alias", "qwen3.8-27b",
    ],
    "startup_wait_seconds": 900,
}

_server_process = None


class ReasoningDetected(Exception):
    """Raised when the model starts reasoning although thinking is off."""


class ServerError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[H3 Prompter] {msg}", flush=True)


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if os.path.isfile(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
                cfg.update(json.load(fh))
        except Exception as exc:  # noqa: BLE001
            log(f"config.json could not be read ({exc}); using defaults.")
    return cfg


# --------------------------------------------------------------------------- server

def _get(url: str, timeout: float = 3.0):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def server_ready(server_url: str) -> bool:
    try:
        status, _ = _get(server_url.rstrip("/") + "/health", timeout=2.0)
        return status == 200
    except Exception:  # noqa: BLE001  (503 while loading, connection refused, ...)
        return False


def _start_server(cfg: dict) -> None:
    global _server_process
    exe = (cfg.get("llama_server_path") or "").strip()
    if not exe:
        raise ServerError("autostart is on but 'llama_server_path' is empty in config.json.")
    args = cfg.get("llama_server_args") or []
    if isinstance(args, str):
        args = shlex.split(args, posix=(os.name != "nt"))
    cmd = [exe, *[str(a) for a in args]]
    log("starting llama-server: " + " ".join(cmd))
    kwargs = {}
    if os.name == "nt":
        # own console window, survives ComfyUI restarts -> model stays in VRAM
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    _server_process = subprocess.Popen(cmd, **kwargs)


def ensure_server(server_url: str, cfg: dict | None = None) -> None:
    """Make sure llama-server answers /health. Optionally start it."""
    if server_ready(server_url):
        return
    cfg = cfg or load_config()
    if not cfg.get("autostart"):
        # maybe it's still loading: wait briefly before failing
        for _ in range(10):
            time.sleep(1.0)
            if server_ready(server_url):
                return
        raise ServerError(
            f"llama-server tidak bisa dihubungi di {server_url}. Jalankan start_llama_server.bat / .sh "
            "terlebih dulu, atau aktifkan 'autostart' di config.json."
        )
    if _server_process is None or _server_process.poll() is not None:
        _start_server(cfg)
    deadline = time.time() + float(cfg.get("startup_wait_seconds", 900))
    while time.time() < deadline:
        if server_ready(server_url):
            log("llama-server ready.")
            return
        if _server_process is not None and _server_process.poll() is not None:
            raise ServerError(f"llama-server exited with code {_server_process.returncode}.")
        time.sleep(2.0)
    raise ServerError("llama-server did not become ready in time (model still downloading/loading?).")


# --------------------------------------------------------------------------- chat

def _interrupted() -> bool:
    try:
        import comfy.model_management as mm  # type: ignore

        return bool(mm.processing_interrupted())
    except Exception:  # noqa: BLE001
        return False


def _raise_if_interrupted() -> None:
    try:
        import comfy.model_management as mm  # type: ignore

        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def stream_chat(
    server_url: str,
    payload: dict,
    *,
    timeout: float = 600,
    abort_on_reasoning: bool = False,
    print_tokens: bool = False,
    on_token=None,
) -> tuple[str, str, dict]:
    """POST /v1/chat/completions with stream=True.

    Returns (content, reasoning, timings). Inline <think>...</think> in `content` is split out.
    """
    body = dict(payload)
    body["stream"] = True
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        server_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    timings: dict = {}
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:800]
        raise ServerError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ServerError(f"cannot reach llama-server at {server_url}: {exc.reason}") from exc
    except (OSError, http.client.HTTPException) as exc:  # reset / disconnected / timeout
        raise ServerError(f"connection to llama-server failed: {exc}") from exc

    with resp:
        for raw in resp:
            if _interrupted():
                resp.close()  # closing the socket makes llama-server stop generating
                _raise_if_interrupted()
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if "error" in obj:
                raise ServerError(str(obj["error"]))
            if obj.get("timings"):
                timings = obj["timings"]
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                r = delta.get("reasoning_content")
                c = delta.get("content")
                if r:
                    if abort_on_reasoning:
                        resp.close()
                        raise ReasoningDetected()
                    reasoning_parts.append(r)
                    if print_tokens:
                        sys.stdout.write(r)
                        sys.stdout.flush()
                if c:
                    content_parts.append(c)
                    if abort_on_reasoning:
                        head = "".join(content_parts).lstrip()
                        if head.startswith(_THINK_OPEN) and len(head) > len(_THINK_OPEN) + 12:
                            after = head[len(_THINK_OPEN):].lstrip()
                            if not after.startswith(_THINK_CLOSE):
                                resp.close()
                                raise ReasoningDetected()
                    if print_tokens:
                        sys.stdout.write(c)
                        sys.stdout.flush()
                    if on_token is not None:
                        on_token(c)
    if print_tokens:
        sys.stdout.write("\n")
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    # split inline think blocks (server started without reasoning extraction)
    if _THINK_CLOSE in content:
        head, tail = content.split(_THINK_CLOSE, 1)
        reasoning = (reasoning + "\n" + head.replace(_THINK_OPEN, "")).strip()
        content = tail
    elif content.lstrip().startswith(_THINK_OPEN):
        # unfinished reasoning -> no usable answer
        reasoning = (reasoning + "\n" + content.replace(_THINK_OPEN, "")).strip()
        content = ""
    return content.strip(), reasoning.strip(), timings
