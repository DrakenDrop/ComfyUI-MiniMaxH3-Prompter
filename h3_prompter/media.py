"""Media helpers: ComfyUI IMAGE/VIDEO/AUDIO -> what the LLM needs."""

from __future__ import annotations

import base64
import io
import math

import numpy as np
from PIL import Image

H3_FPS = 24


def h3_frames(seconds: float) -> tuple[int, float]:
    """MiniMax H3 uses a 17k+5 frame grid at 24 fps. Returns (frames, effective_seconds)."""
    frames = max(5, int(round(float(seconds) * H3_FPS)))
    k = math.ceil((frames - 5) / 17)
    frames = 17 * k + 5
    return frames, frames / H3_FPS


def h3_frames_floor(n_frames: int) -> tuple[int, float]:
    """Largest valid 17k+5 frame count <= n_frames (how the H3 node trims a reference video)."""
    n = max(5, int(n_frames))
    n = 5 + 17 * ((n - 5) // 17)
    return n, n / H3_FPS


def _to_numpy(t):
    if hasattr(t, "detach"):
        t = t.detach()
    if hasattr(t, "cpu"):
        t = t.cpu()
    if hasattr(t, "float") and not isinstance(t, np.ndarray):
        t = t.float()
    if hasattr(t, "numpy"):
        t = t.numpy()
    return np.asarray(t)


def image_batch_to_pil(images) -> list[Image.Image]:
    """ComfyUI IMAGE [B,H,W,C] float 0..1 -> list of PIL images."""
    arr = _to_numpy(images)
    if arr.ndim == 3:
        arr = arr[None]
    out = []
    for frame in arr:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        out.append(Image.fromarray(frame))
    return out


def pil_to_data_url(img: Image.Image, max_side: int = 768, quality: int = 90) -> str:
    img = img.convert("RGB")
    w, h = img.size
    scale = max_side / float(max(w, h))
    if scale < 1.0:
        img = img.resize((max(32, int(w * scale)), max(32, int(h * scale))), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def sample_indices(n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def _video_images(video):
    """ComfyUI VIDEO object (get_components) or IMAGE batch -> image tensor [N,H,W,C]."""
    if hasattr(video, "get_components"):
        return video.get_components().images
    return video


def video_len(video) -> int:
    imgs = _video_images(video)
    try:
        return int(imgs.shape[0]) if len(imgs.shape) == 4 else 1
    except Exception:  # noqa: BLE001
        return 0


def video_sample(video, sample_fps: float, limit: int | None = None) -> list[tuple[int, Image.Image]]:
    """Sample frames at `sample_fps` (source assumed 24 fps, like the H3 node), only within the
    first `limit` frames (the part the H3 node actually uses). Only sampled frames are converted."""
    imgs = _video_images(video)
    n = video_len(video)
    if limit:
        n = min(n, int(limit))
    step = max(1, int(round(H3_FPS / max(0.1, float(sample_fps)))))
    idx = list(range(0, n, step))
    out = []
    for i in idx:
        out.append((i, image_batch_to_pil(imgs[i:i + 1])[0]))
    return out


def audio_duration(audio) -> float | None:
    try:
        wf = audio["waveform"]
        sr = float(audio["sample_rate"])
        return float(wf.shape[-1]) / sr
    except Exception:  # noqa: BLE001
        return None
