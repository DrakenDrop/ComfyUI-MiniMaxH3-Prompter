"""ComfyUI-MiniMaxH3-Prompter: fast MiniMax H3 R2V prompt writer on llama.cpp (llama-server)."""

import threading

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from . import nodes_v2v as _v2v
from . import nodes_i2v as _i2v
from . import nodes_h3qwen as _h3qwen

NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_v2v.NODE_CLASS_MAPPINGS, **_i2v.NODE_CLASS_MAPPINGS,
                       **_h3qwen.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_v2v.NODE_DISPLAY_NAME_MAPPINGS,
                              **_i2v.NODE_DISPLAY_NAME_MAPPINGS, **_h3qwen.NODE_DISPLAY_NAME_MAPPINGS}
from .h3_prompter import llama_client as _lc


def _warm_start():
    cfg = _lc.load_config()
    if not cfg.get("autostart"):
        return
    try:
        _lc.ensure_server(cfg["server_url"], cfg)
    except Exception as exc:  # noqa: BLE001
        _lc.log(f"autostart failed: {exc}")


# start llama-server in the background when ComfyUI boots, so the model is already in VRAM
threading.Thread(target=_warm_start, name="h3-llama-autostart", daemon=True).start()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
