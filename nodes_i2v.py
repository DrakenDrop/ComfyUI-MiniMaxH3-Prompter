"""MiniMax H3 Image(+audio) to Video + LLM, and an aspect-ratio / resolution helper.

image (+ .mp3) -> canvas from resolution + aspect ratio (or the image's own aspect)
  -> the LLM sees the image exactly as it will be framed and writes the H3 prompt
  -> MiniMaxH3ReferenceToVideo (<Picture 1> = image, <Audio 1> = audio)
  -> MiniMaxH3AddGuide at frame 0 (image = first frame, audio = exact soundtrack)
"""

from __future__ import annotations

import math

from . import nodes as prompter_nodes
from .h3_prompter import llama_client as lc
from .h3_prompter import local_models
from .h3_prompter import managed_server
from .h3_prompter import media
from .h3_prompter import v2v

try:
    from comfy_extras import nodes_minimax_h3 as h3  # type: ignore
    _H3_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    h3 = None
    _H3_IMPORT_ERROR = e

CATEGORY = "MiniMax H3/Image to Video"
_CFG = prompter_nodes._CFG

SAME_AS_IMAGE = "same as image"
ASPECTS = [SAME_AS_IMAGE, "9:16", "16:9", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "21:9", "9:21"]
RESOLUTIONS = ["768p (native)", "480p", "576p", "640p", "704p", "512p"]


def canvas(resolution: str, aspect: str, image=None) -> tuple[int, int]:
    """(width, height) on the H3 grid: short edge from `resolution`, area cap short^2*1344/768, multiples of 32."""
    short = v2v.RESOLUTIONS[resolution]
    if aspect == SAME_AS_IMAGE:
        if image is None:
            raise ValueError("aspect_ratio 'same as image' needs an image")
        rw, rh = int(image.shape[2]), int(image.shape[1])
    else:
        a, b = aspect.split(":")
        rw, rh = float(a), float(b)
    return v2v.canvas_for(rw, rh, short)


def _args(node_output):
    return node_output.args if hasattr(node_output, "args") else node_output


def _blank(w=64, h=64):
    import torch

    return torch.zeros((1, h, w, 3), dtype=torch.float32)


def fit_audio(audio, seconds: float):
    """Trim / pad (silence) an AUDIO dict to exactly `seconds`."""
    if audio is None:
        return None
    import torch

    wf = audio["waveform"]
    sr = int(audio["sample_rate"])
    want = int(round(seconds * sr))
    wf = wf[..., :want]
    if wf.shape[-1] < want:
        wf = torch.cat([wf, torch.zeros(*wf.shape[:-1], want - wf.shape[-1], dtype=wf.dtype, device=wf.device)], -1)
    return {"waveform": wf.contiguous(), "sample_rate": sr}


class MiniMaxH3AspectRatio:
    """Resolution + aspect ratio -> width / height for the H3 nodes (optionally the image cropped to it)."""

    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("INT", "INT", "IMAGE")
    RETURN_NAMES = ("width", "height", "image")
    DESCRIPTION = ("H3 canvas size: short edge 768 (native) or 480 etc., the chosen aspect ratio (or the image's own), "
                   "multiples of 32, area capped like the H3 nodes. The optional image comes out center-cropped to it.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (RESOLUTIONS, {"default": "768p (native)", "tooltip": "Short edge of the video."}),
                "aspect_ratio": (ASPECTS, {"default": SAME_AS_IMAGE}),
            },
            "optional": {"image": ("IMAGE",)},
        }

    def run(self, resolution, aspect_ratio, image=None):
        if image is None and aspect_ratio == SAME_AS_IMAGE:
            aspect_ratio = "16:9"
        w, h = canvas(resolution, aspect_ratio, image)
        out = v2v.resize_frames(image, w, h) if image is not None else _blank(w, h)
        return (int(w), int(h), out)


class MiniMaxH3I2VLLM:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "IMAGE", "AUDIO", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("positive", "latent", "prompt", "first_frame", "audio", "width", "height", "frame_count", "fps")
    DESCRIPTION = (
        "Image (+ audio) to video for MiniMax H3 with the prompt written by the local LLM, which sees the image "
        "exactly as it is framed. Canvas = resolution + aspect ratio (or the image's own aspect). The image is the "
        "first frame (center-cropped to the aspect) or only a reference; the audio is the exact soundtrack (anchored "
        "at frame 0, lip-sync) or only a voice/music reference. Length follows the audio unless set. "
        "Wire positive to BasicGuider, latent to SamplerCustomAdvanced; audio output to Create Video."
    )

    _cache: dict = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "MiniMax H3 text encoder (CLIPLoader type 'minimax')."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "image": ("IMAGE", {"tooltip": "The image (first image of a batch is used)."}),
                "instruction": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "What happens in the video (any language). Say what the audio "
                                                      "is, e.g. 'she sings the song in Audio 1', 'he talks'."}),
                "llm_model": (local_models.model_choices(_CFG), {}),
                "mmproj": (local_models.mmproj_choices(_CFG), {"default": local_models.MMPROJ_AUTO}),
                "thinking": (prompter_nodes.THINKING, {"default": "off"}),
                "length": (["compact", "standard", "detailed"], {"default": "standard"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "resolution": (RESOLUTIONS, {"default": "768p (native)",
                                             "tooltip": "Short edge: 768p (native, max 768x1344) or 480p (max 480x832)."}),
                "aspect_ratio": (ASPECTS, {"default": SAME_AS_IMAGE,
                                           "tooltip": "'same as image' keeps the image's aspect. Other ratios crop the "
                                                      "image (center) when it is the first frame."}),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE (needed when audio is connected)."}),
                "audio": ("AUDIO", {"tooltip": "e.g. an .mp3 from Load Audio."}),
                "audio_is_soundtrack": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On = the audio IS the video's sound, used exactly from 0 s (lip-sync / rhythm). "
                               "Off = only a reference (voice timbre / music style); H3 makes new sound.",
                }),
                "image_as_first_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On = the video starts exactly on the image (cropped to the aspect). Off = the image is "
                               "only a reference (person/outfit/style); H3 frames a new shot in the chosen aspect.",
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 15.08, "step": 0.01,
                    "tooltip": "0 = length of the audio (or 5 s without audio). Rounded up to 17k+5 frames @24fps.",
                }),
                "frame_count": ("INT", {"default": 0, "min": 0, "max": 362, "step": 1,
                                        "tooltip": "> 0 overrides duration: 124, 141, ... 362 (nearest grid value)."}),
                "shots": (["auto", "1", "2", "3", "4", "5", "6"], {"default": "auto"}),
                "allow_invented_dialogue": ("BOOLEAN", {"default": False}),
                "asset_notes": ("STRING", {"multiline": True, "default": ""}),
                "prompt_override": ("STRING", {"multiline": True, "default": "",
                                               "tooltip": "If not empty, used as the prompt and the LLM is skipped."}),
                "prompt_style": (["full (official H3)", "simple"], {"default": "full (official H3)"}),
                "ref_image_size": (["match", "max"], {"default": "match",
                                                      "tooltip": "max = sharper identity from the image, slower."}),
                "describe_refs": ("BOOLEAN", {"default": True}),
                "unload_llm_after_prompt": ("BOOLEAN", {"default": False}),
                "context_size": ("INT", {"default": int(_CFG.get("context_size", 32768)), "min": 4096,
                                         "max": 262144, "step": 1024}),
                "max_tokens": ("INT", {"default": 2048, "min": 256, "max": 32768, "step": 64}),
            },
        }

    @staticmethod
    def _frames(duration_seconds, frame_count, audio):
        if frame_count and frame_count > 0:
            return v2v.snap_frames(frame_count), "frame_count"
        if duration_seconds and duration_seconds > 0:
            return min(media.h3_frames(duration_seconds)[0], v2v.MAX_FRAMES), "duration_seconds"
        d = media.audio_duration(audio) if audio is not None else None
        if d:
            n = media.h3_frames(d)[0]
            if n > v2v.MAX_FRAMES:
                lc.log(f"I2V: audio is {d:.2f}s; H3 max is {v2v.MAX_FRAMES / 24:.2f}s -> the audio is cut there.")
                n = v2v.MAX_FRAMES
            if n < v2v.TRAINED_MIN_FRAMES:
                lc.log(f"I2V: audio is only {d:.2f}s -> {n} frames, below H3's trained range (124+); results may be "
                       "weaker. Set duration_seconds to make the video longer (the rest is silent).")
            return n, "audio"
        return 124, "default 5 s"

    def run(self, clip, vae, image, instruction, llm_model, mmproj, thinking, length, seed, resolution, aspect_ratio,
            audio_vae=None, audio=None, audio_is_soundtrack=True, image_as_first_frame=True, duration_seconds=0.0,
            frame_count=0, shots="auto", allow_invented_dialogue=False, asset_notes="", prompt_override="",
            prompt_style="full (official H3)", ref_image_size="match", describe_refs=True,
            unload_llm_after_prompt=False, context_size=32768, max_tokens=2048):
        if h3 is None:
            raise RuntimeError(f"needs a ComfyUI with native MiniMax H3 nodes. Import error: {_H3_IMPORT_ERROR}")

        width, height = canvas(resolution, aspect_ratio, image)
        frames, why = self._frames(duration_seconds, frame_count, audio)
        seconds = frames / v2v.H3_FPS
        img = image[:1, ..., :3]
        framed = v2v.resize_frames(img, width, height)  # exactly what frame 0 will look like
        aud = fit_audio(audio, seconds)
        soundtrack = aud is not None and audio_is_soundtrack
        if aud is not None and audio_vae is None:
            if soundtrack:
                raise ValueError("audio_is_soundtrack needs the audio_vae input (MiniMax H3 audio VAE).")
            lc.log("I2V: no audio_vae -> the audio only conditions the text encoder.")
        lc.log(f"I2V: {width}x{height} ({resolution}, {aspect_ratio}), {frames} frames = {seconds:.2f}s (from {why})"
               + (", image = first frame" if image_as_first_frame else ", image = reference only")
               + ("" if aud is None else (", audio = exact soundtrack" if soundtrack else ", audio = reference")))

        # ---- prompt ----------------------------------------------------------------------------
        prompt = (prompt_override or "").strip()
        if not prompt:
            prompt = self._write_prompt(
                instruction=instruction, llm_model=llm_model, mmproj=mmproj, thinking=thinking, length=length,
                seed=seed, pic=framed if image_as_first_frame else img, audio=aud, soundtrack=soundtrack,
                first=image_as_first_frame, frames=frames, shots=shots, allow_invented_dialogue=allow_invented_dialogue,
                asset_notes=asset_notes, prompt_style=prompt_style, describe_refs=describe_refs,
                context_size=context_size, max_tokens=max_tokens, aspect=f"{width}x{height}")
            if unload_llm_after_prompt:
                managed_server.stop(_CFG)

        # ---- H3 conditioning -------------------------------------------------------------------
        positive, latent = _args(h3.MiniMaxH3ReferenceToVideo.execute(
            clip=clip, prompt=prompt, width=width, height=height, length=frames, ref_image_size=ref_image_size,
            vae=vae, audio_vae=audio_vae, ref_images={"ref_image_0": framed if image_as_first_frame else img},
            ref_videos=None, ref_video_audios=None,
            ref_audios={"ref_audio_0": aud} if aud is not None else None))[:2]
        guide = {}
        if image_as_first_frame:
            guide["image"] = framed
        if soundtrack:
            guide["audio"] = aud
        if guide:
            positive = _args(h3.MiniMaxH3AddGuide.execute(
                positive=positive, latent=latent, frame_idx=0, vae=vae, audio_vae=audio_vae, **guide))[0]

        return (positive, latent, prompt, framed, aud, int(width), int(height), int(frames), float(v2v.H3_FPS))

    def _write_prompt(self, *, instruction, llm_model, mmproj, thinking, length, seed, pic, audio, soundtrack, first,
                      frames, shots, allow_invented_dialogue, asset_notes, prompt_style, describe_refs, context_size,
                      max_tokens, aspect):
        key = (instruction, llm_model, mmproj, thinking, length, seed, v2v.tensor_sig(pic),
               v2v.tensor_sig(audio["waveform"]) if audio is not None else (), soundtrack, first, frames, shots,
               allow_invented_dialogue, asset_notes, prompt_style, describe_refs)
        if key in self._cache:
            lc.log("I2V: inputs for the LLM unchanged -> reusing the cached prompt.")
            return self._cache[key]

        rules = []
        if first:
            rules.append(f"The target video is {aspect} and starts exactly on <Picture 1> (its framing, subject, "
                         "layout and lighting); everything that happens grows naturally out of that frame.")
        else:
            rules.append("<Picture 1> is a reference only (identity, outfit, style), not a frame: compose a new shot "
                         f"for a {aspect} video.")
        if audio is not None and soundtrack:
            rules.append(
                "AUDIO: <Audio 1> IS the soundtrack of the target video, used exactly from 0 s (it is anchored). Mark "
                "it fully_copy and add 'audio reuse' to the task prefix. You cannot hear it: never invent its words, "
                "lyrics or sounds. Follow the request: speech or singing -> <Subject 1> lip-syncs to <Audio 1> "
                "(mouth, breathing, expressions and gestures follow it); music -> the motion follows its rhythm. "
                "overall_soundscape: one short sentence saying the sound comes from <Audio 1>; non_diegetic_music: "
                "N/A unless the request says <Audio 1> is music.")
        elif audio is not None:
            rules.append("AUDIO: <Audio 1> is only a reference (voice timbre or music style, as the request says): "
                         "marker reference, task prefix gets 'audio reference'; H3 creates the actual sound.")
        simple_opener = simple_closer = None
        if prompt_style.startswith("simple"):
            tags = ["keyframe completion" if first else "reference generation"]
            if audio is not None:
                tags.append("audio reuse" if soundtrack else "audio reference")
            simple_opener = (f"[{' + '.join(tags)}] "
                             + ("The shot begins from <Picture 1>: " if first else "<Subject 1> from <Picture 1>: "))
            simple_closer = ("The sound is <Audio 1>, reused exactly; lips and motion follow it." if soundtrack else
                             ("The voice/music follows the style of <Audio 1>." if audio is not None else ""))
        kw = {"image_1": pic}
        if audio is not None:
            kw["audio_1"] = audio
        out = prompter_nodes.MiniMaxH3R2VPrompter().generate(
            instruction=instruction, task="keyframe completion" if first else "reference generation",
            frame_anchor="reference 1 = first frame" if first else "none", duration_seconds=0.0,
            thinking=thinking, length=length, allow_invented_dialogue=allow_invented_dialogue,
            max_tokens=max_tokens, seed=seed, model=llm_model, mmproj=mmproj, asset_notes=asset_notes,
            extra_rules="\n".join(rules), image_max_side=768, context_size=context_size, print_to_console=True,
            frame_count=frames, shots=shots, prompt_style=prompt_style, describe_refs=describe_refs,
            simple_opener=simple_opener, simple_closer=simple_closer, **kw)
        prompt = out[0]
        if len(self._cache) > 16:
            self._cache.clear()
        self._cache[key] = prompt
        return prompt


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3I2VLLM": MiniMaxH3I2VLLM,
    "MiniMaxH3AspectRatio": MiniMaxH3AspectRatio,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3I2VLLM": "MiniMax H3 Image(+Audio) to Video + LLM",
    "MiniMaxH3AspectRatio": "MiniMax H3 Resolution / Aspect Ratio",
}
