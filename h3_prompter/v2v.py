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
MAX_PAD_FRAMES = 8  # ~0.33 s of held last frame, only when that is the nearer grid length

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


def snap_frames(n: int) -> int:
    """Nearest valid H3 length (17k+5, 5..362); ties go up."""
    n = max(MIN_FRAMES, min(int(n), MAX_FRAMES))
    down = align_down(n)
    up = min(down + 17, MAX_FRAMES)
    return up if (up - n) <= (n - down) else down


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

    def __init__(self, src_frames: int, src_fps: float, start_seconds: float = 0.0, max_seconds: float = 15.0,
                 frame_count: int = 0):
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
        down = align_down(n)
        up = min(down + 17, MAX_FRAMES) if n > down else down
        # Snap to the NEAREST 17k+5 length (5 s = 120 frames -> 124, not 107). When that is longer than the
        # source, the last source frame is held for the few missing frames (at most MAX_PAD_FRAMES).
        self.padded_frames = 0
        avail_frames = int(math.floor(avail * H3_FPS + 1e-6))
        if frame_count and frame_count > 0:  # explicit length wins over max_seconds
            self.frame_count = snap_frames(frame_count)
            if self.frame_count != int(frame_count):
                log.warning("MiniMax H3 V2V: frame_count %d is not on the 17k+5 grid -> using %d.",
                            int(frame_count), self.frame_count)
            self.padded_frames = max(0, self.frame_count - avail_frames)
            if self.padded_frames:
                log.warning("MiniMax H3 V2V: the source has only %d frames (@24fps) after start_seconds; the last "
                            "frame is held for %d frames.", avail_frames, self.padded_frames)
        elif up > n and up - n <= MAX_PAD_FRAMES and (up - n) <= (n - down):
            self.frame_count = up
            self.padded_frames = max(0, up - avail_frames)
        else:
            self.frame_count = down
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
        want = b - a
        if wf.shape[-1] < want:  # held frames at the end -> matching silence
            import torch
            pad = torch.zeros(*wf.shape[:-1], want - wf.shape[-1], dtype=wf.dtype, device=wf.device)
            wf = torch.cat([wf, pad], dim=-1)
        return {"waveform": wf.contiguous(), "sample_rate": sr}


# ----------------------------------------------------------------------------- Fun ControlNet strengths
# (pose, depth, edge, structure_end_percent). Pose keeps motion without dictating shapes; depth/edge also pin
# silhouettes and background, so they are weaker and released earlier where the edit changes shapes.
# The Fun ControlNet skips of chained controls add up: keep the sum around 1.
AUTO_STRENGTH = {
    "video_edit":        (0.90, 0.30, 0.00, 0.50),  # plain prompter [video editing] prompt, no preset rules
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
    "video_edit": "",
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


def _srgb_to_lab(x):
    """x: [...,3] in 0..1 -> Lab (D65)."""
    import torch

    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    m = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=x.dtype, device=x.device)
    xyz = lin @ m.T
    xyz = xyz / torch.tensor([0.95047, 1.0, 1.08883], dtype=x.dtype, device=x.device)
    eps = 216 / 24389
    kappa = 24389 / 27
    f = torch.where(xyz > eps, xyz.clamp(min=1e-8) ** (1 / 3), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


def _lab_to_srgb(lab):
    import torch

    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    eps = 216 / 24389
    kappa = 24389 / 27
    fx3, fz3 = fx ** 3, fz ** 3
    xr = torch.where(fx3 > eps, fx3, (116 * fx - 16) / kappa)
    yr = torch.where(L > kappa * eps, fy ** 3, L / kappa)
    zr = torch.where(fz3 > eps, fz3, (116 * fz - 16) / kappa)
    xyz = torch.stack([xr * 0.95047, yr, zr * 1.08883], dim=-1)
    m = torch.tensor([[3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660, 1.8760108, 0.0415560],
                      [0.0556434, -0.2040259, 1.0572252]], dtype=lab.dtype, device=lab.device)
    lin = (xyz @ m.T).clamp(0, 1)
    return torch.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin.clamp(min=1e-8) ** (1 / 2.4) - 0.055).clamp(0, 1)


def _ref_chunk(reference, n: int, h: int, w: int, i: int, j: int, device):
    """Frames i..j of `reference`, time-mapped to n frames and resized to h x w, on `device`."""
    import torch
    import torch.nn.functional as F

    m = reference.shape[0]
    idx = torch.arange(i, j)
    if m != n:
        idx = (idx.float() * (m - 1) / max(n - 1, 1)).round().long() if m > 1 else torch.zeros(j - i).long()
    r = reference[idx, ..., :3].to(device, non_blocking=True).float()
    if r.shape[1] != h or r.shape[2] != w:
        r = F.interpolate(r.movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False).movedim(1, -1)
    return r


def _smooth_stats(v, radius: int, weight=None):
    """Moving average over time of [N,C] stats (optionally weighted)."""
    import torch.nn.functional as F

    k = 2 * radius + 1
    if weight is None:
        pad = F.pad(v.T.unsqueeze(0), (radius, radius), mode="replicate")
        return F.avg_pool1d(pad, kernel_size=k, stride=1)[0].T
    num = F.avg_pool1d(F.pad((v * weight[:, None]).T.unsqueeze(0), (radius, radius), mode="replicate"), k, 1)[0].T
    den = F.avg_pool1d(F.pad(weight[None, None], (radius, radius), mode="replicate"), k, 1)[0, 0]
    return num / den.clamp(min=1e-6)[:, None]


def match_color(images, reference, strength: float = 1.0, smooth_frames: int = 4, chunk: int = 16, device=None):
    """Per-frame Lab mean/std transfer so `images` get the lighting/exposure/white balance of `reference`.
    Runs chunk by chunk on `device` (GPU); the result goes back to the device of `images`."""
    import torch

    dev = device or images.device
    n, h, w, _ = images.shape
    mus_x, sds_x, mus_r, sds_r = [], [], [], []
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        x = _srgb_to_lab(images[i:j, ..., :3].to(dev, non_blocking=True).float())
        r = _srgb_to_lab(_ref_chunk(reference, n, h, w, i, j, dev))
        mus_x.append(x.mean(dim=(1, 2))); sds_x.append(x.std(dim=(1, 2)).clamp(min=1e-3))
        mus_r.append(r.mean(dim=(1, 2))); sds_r.append(r.std(dim=(1, 2)).clamp(min=1e-3))
    mu_x, sd_x, mu_r, sd_r = (torch.cat(v) for v in (mus_x, sds_x, mus_r, sds_r))
    if smooth_frames > 0 and n > 1:  # moving average -> no flicker from per-frame stats
        mu_x, sd_x, mu_r, sd_r = (_smooth_stats(v, smooth_frames) for v in (mu_x, sd_x, mu_r, sd_r))

    out = []
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        lab = _srgb_to_lab(images[i:j, ..., :3].to(dev, non_blocking=True).float())
        new = (lab - mu_x[i:j, None, None]) / sd_x[i:j, None, None] * sd_r[i:j, None, None] + mu_r[i:j, None, None]
        new = lab + (new - lab) * float(strength)
        out.append(_lab_to_srgb(new).to(images.device, images.dtype))
    return torch.cat(out, dim=0)


# ----------------------------------------------------------------------------- skin tone match (no mask needed)
def skin_prob(x):
    """Soft skin likelihood [N,H,W] from RGB 0..1 (YCrCb chroma ellipse; hair, black/white cloth, walls -> ~0)."""
    import torch

    r, g, b = x[..., 0] * 255, x[..., 1] * 255, x[..., 2] * 255
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128
    cb = (b - y) * 0.564 + 128
    d = ((cr - 153.0) / 9.0) ** 2 + ((cb - 108.0) / 10.0) ** 2
    p = torch.exp(-0.5 * d)
    p = p * ((y - 35.0) / 25.0).clamp(0, 1) * ((250.0 - y) / 20.0).clamp(0, 1)  # not near black / blown white
    return p


def match_skin_tone(images, reference, strength: float = 1.0, smooth_frames: int = 6, chunk: int = 16,
                    feather: int = 7, device=None):
    """Give the skin in `images` (H3 result) the skin tone of `reference` (source frames, same motion).

    No mask: skin is found automatically. Statistics use only pixels that are skin in BOTH videos at the same
    place (so a beige dress in the source or a new outfit in the result does not count), then the colour shift is
    applied to the skin of the result only, feathered, with per-frame stats smoothed over time (no flicker).
    Runs chunk by chunk on `device` (GPU); the result goes back to the device of `images`."""
    import torch
    import torch.nn.functional as F

    dev = device or images.device
    n, h, w, _ = images.shape

    def wstats(lab, wgt):
        tot = wgt.sum(dim=(1, 2))
        t = tot.clamp(min=1e-6)[:, None]
        mu = (lab * wgt[..., None]).sum(dim=(1, 2)) / t
        sd = (((lab - mu[:, None, None]) ** 2 * wgt[..., None]).sum(dim=(1, 2)) / t).sqrt()
        return mu, sd.clamp(min=1e-3), tot

    mo, so, mr, sr, tw = [], [], [], [], []
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        x = images[i:j, ..., :3].to(dev, non_blocking=True).float()
        rf = _ref_chunk(reference, n, h, w, i, j, dev)
        wgt = skin_prob(x) * skin_prob(rf)
        a, b, t = wstats(_srgb_to_lab(x), wgt)
        c, d, _ = wstats(_srgb_to_lab(rf), wgt)
        mo.append(a); so.append(b); mr.append(c); sr.append(d); tw.append(t)
    mu_o, sd_o, mu_r, sd_r, tot = (torch.cat(v) for v in (mo, so, mr, sr, tw))

    # frames with (almost) no shared skin borrow the stats of the others through the weighted smoothing
    wt = (tot / (h * w) / 0.002).clamp(0, 1)  # full weight from 0.2 % of the frame
    if float(wt.sum()) <= 0:
        return images
    if smooth_frames > 0 and n > 1:
        mu_o, sd_o, mu_r, sd_r = (_smooth_stats(v, smooth_frames, wt) for v in (mu_o, sd_o, mu_r, sd_r))
        wt = torch.ones_like(wt)
    ratio = (sd_r / sd_o).clamp(0.8, 1.25)  # mild contrast match only

    out = []
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        x = images[i:j, ..., :3].to(dev, non_blocking=True).float()
        lab = _srgb_to_lab(x)
        new = (lab - mu_o[i:j, None, None]) * ratio[i:j, None, None] + mu_r[i:j, None, None]
        alpha = skin_prob(x)
        if feather > 0:
            alpha = F.avg_pool2d(alpha[:, None], 2 * feather + 1, 1, feather, count_include_pad=False)[:, 0]
        alpha = (alpha * 1.6).clamp(0, 1) * float(strength) * wt[i:j, None, None]
        res = _lab_to_srgb(lab + (new - lab) * alpha[..., None])
        if images.shape[-1] == 4:
            res = torch.cat([res, images[i:j, ..., 3:].to(dev).float()], dim=-1)
        out.append(res.to(images.device, images.dtype))
    return torch.cat(out, dim=0)
