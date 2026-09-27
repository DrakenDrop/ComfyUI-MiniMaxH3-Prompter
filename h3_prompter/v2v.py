"""Video-to-video helpers (from Minimax-H3-V2V, same author, MIT) + LLM edit presets.

H3 works on a fixed grid: 24 fps, clip lengths of 17k + 5 frames, a canvas whose sides are multiples of 32
and whose area is capped at short_edge^2 * 1344/768. Everything that must line up with the generated video
(source frames, control videos, the edited first frame, the frames the LLM sees) goes through these helpers.
"""

from __future__ import annotations

import logging
import math

log = logging.getLogger("MiniMaxH3-Prompter")

H3_FPS = 24
CANVAS_MULTIPLE = 32
MIN_FRAMES = 5
MAX_FRAMES = 362  # 17*21+5, ~15.1 s, top of the trained range
TRAINED_MIN_FRAMES = 124

RESOLUTIONS = {
    "768p (native)": 768,
    "704p": 704,
    "640p": 640,
    "576p": 576,
    "512p": 512,
    "480p": 480,
}


def align_down(n: int) -> int:
    if n < MIN_FRAMES:
        return 0
    return ((n - 5) // 17) * 17 + 5


def canvas_for(width: int, height: int, short_edge: int = 768) -> tuple[int, int]:
    """Same rule as core adapt_canvas(): aspect ratio of the input, short edge `short_edge`, area cap, x32."""
    ratio = width / height
    max_pixels = short_edge * short_edge * 1344 / 768
    if ratio >= 1.0:
        nom_w, nom_h = short_edge * ratio, short_edge
    else:
        nom_w, nom_h = short_edge, short_edge / ratio
    if nom_w * nom_h > max_pixels:
        s = math.sqrt(max_pixels / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def resize_frames(frames, width: int, height: int, method: str = "bilinear"):
    """[N,H,W,C] -> [N,height,width,3], aspect kept by center crop."""
    import torch
    import comfy.utils  # type: ignore

    frames = frames[..., :3]
    if frames.shape[1] == height and frames.shape[2] == width:
        return frames
    out = []
    for chunk in torch.split(frames, 32, dim=0):  # chunked: sane peak RAM on long 1080p inputs
        s = chunk.movedim(-1, 1)
        s = comfy.utils.common_upscale(s, width, height, method, "center")
        out.append(s.movedim(1, -1))
    return torch.cat(out, dim=0).clamp(0.0, 1.0)


class Timeline:
    """Maps a source clip (any fps) onto H3's 24 fps / 17k+5 grid."""

    def __init__(self, src_frames: int, src_fps: float, start_seconds: float = 0.0, max_seconds: float = 15.0):
        import torch

        if src_frames < 1:
            raise ValueError("source video has no frames")
        if src_fps <= 0:
            raise ValueError("source fps must be > 0")
        self.src_frames = int(src_frames)
        self.src_fps = float(src_fps)
        self.start_seconds = max(0.0, float(start_seconds))
        src_duration = self.src_frames / self.src_fps
        avail = src_duration - self.start_seconds
        if avail <= 0:
            raise ValueError(f"start_seconds ({self.start_seconds:.2f}s) is past the end of the source "
                             f"video ({src_duration:.2f}s)")
        want = min(avail, max(0.2, float(max_seconds)))
        n = min(int(math.floor(want * H3_FPS + 1e-6)), MAX_FRAMES)
        self.frame_count = align_down(n)
        if self.frame_count < MIN_FRAMES:
            raise ValueError("source clip is too short for MiniMax H3 (needs at least 5 frames at 24 fps)")
        if self.frame_count < TRAINED_MIN_FRAMES:
            log.warning("MiniMax H3 V2V: %d frames (%.2fs) is below the trained range (~124-362 frames); "
                        "results may be weaker.", self.frame_count, self.frame_count / H3_FPS)
        start_idx = self.start_seconds * self.src_fps
        idx = [min(self.src_frames - 1, int(round(start_idx + j * self.src_fps / H3_FPS)))
               for j in range(self.frame_count)]
        self.indices = torch.tensor(idx, dtype=torch.long)

    @property
    def duration(self) -> float:
        return self.frame_count / H3_FPS

    def take(self, frames, name: str = "video"):
        """raw source length -> same fps map; already frame_count long -> as is; 1 frame -> repeated;
        anything else -> stretched in time (warned)."""
        import torch

        n = frames.shape[0]
        if n == self.src_frames:
            return frames[self.indices.clamp(max=n - 1)]
        if n == self.frame_count:
            return frames
        if n == 1:
            return frames.expand(self.frame_count, *frames.shape[1:])
        log.warning("MiniMax H3 V2V: %s has %d frames, source has %d (conformed %d); stretching it in time. "
                    "Build control videos from the same (or the conformed) clip for exact motion sync.",
                    name, n, self.src_frames, self.frame_count)
        idx = torch.linspace(0, n - 1, self.frame_count).round().long()
        return frames[idx]

    def take_audio(self, audio):
        if audio is None:
            return None
        wf = audio["waveform"]
        sr = int(audio["sample_rate"])
        a = int(round(self.start_seconds * sr))
        b = a + int(round(self.duration * sr))
        wf = wf[..., a:b]
        if wf.shape[-1] < 1:
            return None
        return {"waveform": wf.contiguous(), "sample_rate": sr}


# ----------------------------------------------------------------------------- Fun ControlNet strengths
# (pose, depth, edge, structure_end_percent). Pose keeps motion without dictating shapes; depth/edge also pin
# silhouettes and background, so they are weaker and released earlier where the edit changes shapes.
# The Fun ControlNet skips of chained controls add up: keep the sum around 1.
AUTO_STRENGTH = {
    "custom":            (0.90, 0.30, 0.00, 0.50),
    "change_outfit":     (0.85, 0.30, 0.00, 0.40),
    "replace_person":    (1.00, 0.00, 0.00, 0.40),
    "add_object":        (0.80, 0.20, 0.00, 0.30),
    "add_subject":       (0.80, 0.00, 0.00, 0.30),
    "remove_object":     (0.80, 0.20, 0.00, 0.30),
    "change_background": (0.95, 0.00, 0.00, 0.30),
    "restyle":           (0.60, 0.50, 0.30, 0.70),
}
SOLO_FALLBACK = 0.8
EDIT_MODES = list(AUTO_STRENGTH.keys())


def resolve_strengths(edit_mode, motion_lock, connected, pose_strength, depth_strength, edge_strength,
                      structure_end_percent):
    """connected: {name: bool}. Returns {name: (strength, end_percent)} for connected controls."""
    auto_pose, auto_depth, auto_edge, auto_end = AUTO_STRENGTH[edit_mode]
    auto = {"pose": auto_pose, "depth": auto_depth, "edge": auto_edge}
    manual = {"pose": pose_strength, "depth": depth_strength, "edge": edge_strength}
    end_struct = auto_end if structure_end_percent < 0 else structure_end_percent
    active = [n for n, c in connected.items() if c]
    out = {}
    for name in active:
        if manual[name] >= 0:
            s = manual[name]
        else:
            s = auto[name]
            others = [auto[o] for o in active if o != name]
            if s == 0 and not any(v > 0 for v in others):
                s = SOLO_FALLBACK
            s *= motion_lock
        end = 1.0 if name == "pose" else end_struct
        if name != "pose" and len(active) == 1:  # a lone structure control carries the motion
            end = 1.0 if structure_end_percent < 0 else structure_end_percent
        out[name] = (float(s), float(end))
    return out


# ----------------------------------------------------------------------------- LLM edit presets
# Sent in the (per-run) user message, so the static system prompt stays cached.
PRESET_RULES = {
    "custom": "",
    "change_outfit": (
        "EDIT TYPE - OUTFIT SWAP: keep the person's identity, face, hair, body, pose, motion and position; only the "
        "clothing changes. <Subject 1> (the person in <Video 1>) is partially_preserved. If pictures show the new "
        "garments, add a retention line '<Picture N>: attribute_transfer - the garments are worn by <Subject 1>' "
        "and match their cut, color, material, pattern and details exactly. Describe how the new clothes fold and "
        "move with every movement."
    ),
    "replace_person": (
        "EDIT TYPE - PERSON REPLACEMENT: <Subject 1> is the person in <Video 1>; <Subject 2> is the new person from "
        "the pictures (or described in the request). Retention: '<Subject 1>: attribute_transfer - body motion, pose, "
        "gestures, position and timing transfer to <Subject 2>; identity, face and appearance are replaced' and "
        "'<Subject 2>: fully_preserved - identity, face, hair, body type and look'. Describe <Subject 2> performing "
        "each action of the source at the same moments, in the same place and framing."
    ),
    "add_object": (
        "EDIT TYPE - ADD OBJECT: everything in <Video 1> stays. The new object (from the pictures or the request) is "
        "added where the request says, with consistent appearance and scale for the whole video and correct "
        "perspective, lighting, shadows, reflections and occlusion. Describe where it is and how it behaves in each "
        "moment, and how people in the scene interact with it if the request says so."
    ),
    "add_subject": (
        "EDIT TYPE - ADD SUBJECT: everything in <Video 1> stays. A new person/animal/character (from the pictures or "
        "the request) is added where the request says, with a consistent identity, correct scale, lighting and "
        "shadows, interacting naturally with the existing scene. Describe what the new subject does moment by moment."
    ),
    "remove_object": (
        "EDIT TYPE - REMOVE: the element named in the request is absent for the whole video; the area it covered "
        "shows the plausible background that continues the scene. Everything else in <Video 1> stays. Do not "
        "mention the removed element in detailed_description except once as absent."
    ),
    "change_background": (
        "EDIT TYPE - BACKGROUND CHANGE: keep the people (identity, clothing, pose, motion), the framing and the camera "
        "movement of <Video 1>; replace the environment with the requested one (or the one in the pictures) and "
        "describe it concretely. Lighting on the people adapts subtly to the new environment."
    ),
    "restyle": (
        "EDIT TYPE - RESTYLE: content, motion, timing, poses, camera and composition of <Video 1> stay; only the "
        "rendering style changes. <Video 1> is partially_preserved; style pictures are weak_reference (style only). "
        "Describe the style concretely: line work, shading, color palette, textures, lighting treatment."
    ),
}

DEFAULT_INSTRUCTIONS = {
    "change_outfit": "Put the outfit from the reference picture(s) on the person.",
    "replace_person": "Replace the person with the person from the reference picture(s).",
    "add_object": "Add the object from the reference picture(s) to the scene in a natural place.",
    "add_subject": "Add the subject from the reference picture(s) to the scene, interacting naturally.",
    "restyle": "Restyle the whole video in the style of the reference picture(s).",
    "change_background": "Replace the background with the environment shown in the reference picture(s).",
}


def tensor_sig(t) -> tuple:
    """Cheap fingerprint of a tensor for prompt caching."""
    if t is None:
        return ()
    try:
        flat = t.reshape(-1)
        n = flat.shape[0]
        step = max(1, n // 4096)
        return (tuple(t.shape), round(float(flat[::step].float().sum()), 4))
    except Exception:  # noqa: BLE001
        return (id(t),)
