"""MiniMax H3 V2V Edit + LLM: video-to-video editing with the prompt written by the local LLM.

One node does everything:
  source clip -> 24 fps / 17k+5 timeline + H3 canvas
  -> the LLM sees exactly those frames (+ refs + edited first frame) and writes the H3 [video editing] prompt
     for the chosen preset (outfit swap, replace person, add object/subject, remove, background, restyle, custom)
  -> MiniMaxH3ReferenceToVideo (<Video 1> = source, <Picture i> = refs, last <Picture> = first frame)
  -> optional MiniMaxH3AddGuide (edited first frame pinned at frame 0)
  -> MiniMaxH3FunControlNetApply for pose / depth / edge (motion lock, auto strengths per preset)
Outputs go to BasicGuider / BasicScheduler / SamplerCustomAdvanced like the Minimax-H3-V2V workflow.
"""

from __future__ import annotations

import logging

from . import nodes as prompter_nodes
from .h3_prompter import llama_client as lc
from .h3_prompter import local_models
from .h3_prompter import managed_server
from .h3_prompter import v2v

log = logging.getLogger("MiniMaxH3-Prompter")

try:
    from comfy_extras import nodes_minimax_h3 as h3  # type: ignore
    _H3_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001  (ComfyUI too old / outside ComfyUI)
    h3 = None
    _H3_IMPORT_ERROR = e

CATEGORY = "MiniMax H3/V2V Edit"
MAX_REFS = 8  # + edited first frame = 9 pictures (H3 limit)
_CFG = prompter_nodes._CFG


def _require_h3():
    if h3 is None or not hasattr(h3, "MiniMaxH3FunControlNetApply"):
        raise RuntimeError(
            "MiniMax H3 V2V Edit + LLM needs a ComfyUI with native MiniMax H3 + Fun ControlNet "
            f"(comfy_extras/nodes_minimax_h3.py). Update ComfyUI. Import error: {_H3_IMPORT_ERROR}")


def _args(node_output):
    return node_output.args if hasattr(node_output, "args") else node_output


class MiniMaxH3V2VEditLLM:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "STRING", "IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("model", "positive", "latent", "prompt", "source_frames", "frame_count", "fps")
    DESCRIPTION = (
        "Video-to-video edit for MiniMax H3 (ref2va model) with the prompt written by a local LLM (llama.cpp) that "
        "sees the actual frames. Presets: outfit swap, replace person, add object/subject, remove object, change "
        "background, restyle, custom. Pose/depth/edge videos drive the Fun ControlNet-Union so motion matches the "
        "input. Wire model+positive to BasicGuider, model to BasicScheduler, latent to SamplerCustomAdvanced."
    )

    _cache: dict = {}

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": ("MODEL", {"tooltip": "MiniMax H3 ref2va diffusion model (optionally with the turbo LoRA)."}),
            "clip": ("CLIP", {"tooltip": "Qwen3-VL MiniMax H3 text encoder (CLIPLoader type 'minimax')."}),
            "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
            "source_video": ("IMAGE", {"tooltip": "Source frames (raw, or from a conform node)."}),
            "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                                     "tooltip": "FPS of source_video (e.g. from Get Video Components)."}),
            "edit_mode": (v2v.EDIT_MODES, {
                "default": "change_outfit",
                "tooltip": "Preset: what the LLM writes and how strong pose/depth/edge lock the motion. "
                           "custom = only your instruction.",
            }),
            "instruction": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": "What to change (any language). Empty = default for the preset using the reference "
                           "picture(s), e.g. 'put the outfit from the reference on the person'.",
            }),
            "llm_model": (local_models.model_choices(_CFG), {
                "tooltip": "GGUF from ComfyUI/models/LLM, served by llama-server (stays in VRAM).",
            }),
            "mmproj": (local_models.mmproj_choices(_CFG), {
                "default": local_models.MMPROJ_AUTO,
                "tooltip": "Vision file for the LLM - required so it can see the video and references.",
            }),
            "thinking": (prompter_nodes.THINKING, {"default": "off"}),
            "length": (["compact", "standard", "detailed"], {"default": "standard"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                             "tooltip": "LLM seed (prompt variations). The sampler has its own noise seed."}),
        }
        optional = {
            "model_patch": ("MODEL_PATCH", {"tooltip": "MiniMax H3 Fun ControlNet-Union (ModelPatchLoader)."}),
            "control_pose": ("IMAGE", {"tooltip": "Pose video of the SAME clip."}),
            "control_depth": ("IMAGE", {"tooltip": "Depth video of the same clip."}),
            "control_edge": ("IMAGE", {"tooltip": "Canny / HED / MLSD video of the same clip (strong)."}),
        }
        for i in range(1, MAX_REFS + 1):
            optional[f"ref_image_{i}"] = ("IMAGE", {
                "tooltip": f"Reference {i} (new person / outfit / object / style) -> <Picture {i}> "
                           "(connected refs are numbered in order).",
            })
        optional.update({
            "first_frame": ("IMAGE", {
                "tooltip": "Optional EDITED frame 0: pinned at frame 0 and given as the last <Picture>. "
                           "Biggest consistency boost.",
            }),
            "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE (only for reuse_audio)."}),
            "source_audio": ("AUDIO", {"tooltip": "Soundtrack of the source (for reuse_audio)."}),
            "reuse_audio": ("BOOLEAN", {
                "default": False,
                "tooltip": "Give the soundtrack to H3 as <Audio 1> and keep it. Off = mux the original audio in "
                           "Create Video instead (the prompt then keeps the sound sections minimal).",
            }),
            "asset_notes": ("STRING", {"multiline": True, "default": "",
                                       "tooltip": "Optional roles, e.g. 'Picture 1 = only the dress, ignore the face'."}),
            "prompt_override": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "If not empty, used as the prompt and the LLM is skipped."}),
            "motion_lock": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                      "tooltip": "Multiplier on the auto control strengths (0.8 freer, 1.2 tighter)."}),
            "pose_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 3.0, "step": 0.01}),
            "depth_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 3.0, "step": 0.01}),
            "edge_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 3.0, "step": 0.01}),
            "structure_end_percent": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            "use_source_as_reference": ("BOOLEAN", {"default": True}),
            "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01}),
            "max_seconds": ("FLOAT", {"default": 15.0, "min": 0.25, "max": 15.1, "step": 0.01}),
            "resolution": (list(v2v.RESOLUTIONS.keys()), {
                "default": "768p (native)",
                "tooltip": "Short edge; the aspect ratio follows the source video.",
            }),
            "ref_image_size": (["match", "max"], {"default": "max"}),
            "video_sample_fps": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 6.0, "step": 0.5}),
            "unload_llm_after_prompt": ("BOOLEAN", {
                "default": False,
                "tooltip": "Stop llama-server right after the prompt is written (frees its VRAM for sampling).",
            }),
            "context_size": ("INT", {"default": int(_CFG.get("context_size", 32768)), "min": 4096,
                                     "max": 262144, "step": 1024}),
            "max_tokens": ("INT", {"default": 2048, "min": 256, "max": 32768, "step": 64}),
        })
        return {"required": required, "optional": optional}

    # ------------------------------------------------------------------ prompt
    def _write_prompt(self, *, edit_mode, instruction, llm_model, mmproj, thinking, length, seed, src, refs, first,
                      audio, asset_notes, video_sample_fps, context_size, max_tokens, duration, use_src):
        key = (edit_mode, instruction, llm_model, mmproj, thinking, length, seed, asset_notes, video_sample_fps,
               bool(audio), use_src, v2v.tensor_sig(src), tuple(v2v.tensor_sig(r) for r in refs),
               v2v.tensor_sig(first))
        if key in self._cache:
            lc.log("V2V: inputs for the LLM unchanged -> reusing the cached prompt.")
            return self._cache[key]

        text = instruction.strip() or v2v.DEFAULT_INSTRUCTIONS.get(edit_mode, "")
        if not refs and not text:
            raise ValueError("Write an instruction (or connect a reference image) so the LLM knows what to change.")
        kw = {f"image_{i}": r for i, r in enumerate(refs, start=1)}
        if first is not None:
            kw["first_frame"] = first
        if use_src:
            kw["video_1"] = src
            if audio is not None:
                kw["video_1_audio"] = audio
        rules = v2v.PRESET_RULES.get(edit_mode, "")
        if not use_src:
            rules += ("\nThe source clip is NOT supplied as <Video 1>: never mention <Video 1>; describe the whole "
                      "resulting scene explicitly (the motion comes from control videos).")
        out = prompter_nodes.MiniMaxH3R2VPrompter().generate(
            instruction=text, task="video editing" if use_src else "reference generation", frame_anchor="none",
            duration_seconds=duration, thinking=thinking, length=length, allow_invented_dialogue=False,
            max_tokens=max_tokens, seed=seed, model=llm_model, mmproj=mmproj,
            asset_notes=asset_notes, extra_rules=rules, video_sample_fps=video_sample_fps,
            video_max_side=512, image_max_side=768, context_size=context_size, print_to_console=True, **kw)
        prompt = out[0]
        if len(self._cache) > 16:
            self._cache.clear()
        self._cache[key] = prompt
        return prompt

    # ------------------------------------------------------------------ main
    def run(self, model, clip, vae, source_video, source_fps, edit_mode, instruction, llm_model, mmproj, thinking,
            length, seed, model_patch=None, control_pose=None, control_depth=None, control_edge=None,
            first_frame=None, audio_vae=None, source_audio=None, reuse_audio=False, asset_notes="",
            prompt_override="", motion_lock=1.0, pose_strength=-1.0, depth_strength=-1.0, edge_strength=-1.0,
            structure_end_percent=-1.0, use_source_as_reference=True, start_seconds=0.0, max_seconds=15.0,
            resolution="768p (native)", ref_image_size="max", video_sample_fps=2.0, unload_llm_after_prompt=False,
            context_size=32768, max_tokens=2048, **refs_kw):
        _require_h3()

        # ---- timeline + canvas (the LLM sees exactly these frames) ------------------------------
        tl = v2v.Timeline(source_video.shape[0], source_fps, start_seconds, max_seconds)
        src = tl.take(source_video, "source_video")
        width, height = v2v.canvas_for(src.shape[2], src.shape[1], v2v.RESOLUTIONS[resolution])
        src = v2v.resize_frames(src, width, height)
        lc.log(f"V2V: {tl.frame_count} frames ({tl.duration:.2f}s @24fps) at {width}x{height}, mode={edit_mode}")

        refs = [refs_kw[f"ref_image_{i}"][:1] for i in range(1, MAX_REFS + 1)
                if refs_kw.get(f"ref_image_{i}") is not None]
        first = v2v.resize_frames(first_frame[:1], width, height) if first_frame is not None else None
        audio = tl.take_audio(source_audio) if (reuse_audio and use_source_as_reference) else None
        if reuse_audio and audio is not None and audio_vae is None:
            log.warning("MiniMax H3 V2V: reuse_audio needs audio_vae; the soundtrack only reaches the text encoder.")

        # ---- prompt -------------------------------------------------------------------------------
        prompt = (prompt_override or "").strip()
        if not prompt:
            prompt = self._write_prompt(
                edit_mode=edit_mode, instruction=instruction, llm_model=llm_model, mmproj=mmproj,
                thinking=thinking, length=length, seed=seed, src=src, refs=refs, first=first, audio=audio,
                asset_notes=asset_notes, video_sample_fps=video_sample_fps, context_size=context_size,
                max_tokens=max_tokens, duration=tl.duration, use_src=use_source_as_reference)
            if unload_llm_after_prompt:
                managed_server.stop(_CFG)

        # ---- H3 reference conditioning (same label order as the prompt) ---------------------------
        core_refs = {f"ref_image_{i}": r for i, r in enumerate(refs)}
        if first is not None:
            core_refs[f"ref_image_{len(refs)}"] = first
        if len(core_refs) > 9:
            raise ValueError(f"MiniMax H3 accepts at most 9 reference images (got {len(core_refs)})")
        ref_videos = {"ref_video_0": src} if use_source_as_reference else None
        ref_video_audios = {"ref_video_audio_0": audio} if audio is not None else None

        positive, latent = _args(h3.MiniMaxH3ReferenceToVideo.execute(
            clip=clip, prompt=prompt, width=width, height=height, length=tl.frame_count,
            ref_image_size=ref_image_size, vae=vae, audio_vae=audio_vae,
            ref_images=core_refs or None, ref_videos=ref_videos,
            ref_video_audios=ref_video_audios, ref_audios=None))[:2]

        if first is not None:
            positive = _args(h3.MiniMaxH3AddGuide.execute(
                positive=positive, latent=latent, frame_idx=0, vae=vae, image=first))[0]

        # ---- motion lock: Fun ControlNet-Union ----------------------------------------------------
        controls = {"pose": control_pose, "depth": control_depth, "edge": control_edge}
        connected = {k: v is not None for k, v in controls.items()}
        if any(connected.values()) and model_patch is None:
            raise ValueError("control videos are connected but model_patch (MiniMax H3 Fun ControlNet-Union, "
                             "loaded with ModelPatchLoader) is missing")
        out_model = model
        plan = v2v.resolve_strengths(edit_mode, motion_lock, connected, pose_strength, depth_strength,
                                     edge_strength, structure_end_percent)
        for name, (strength, end) in plan.items():
            if strength <= 0:
                continue
            cv = v2v.resize_frames(tl.take(controls[name], "control_" + name), width, height)
            lc.log(f"V2V: control {name} strength={strength:.2f} end={end:.2f}")
            out_model = _args(h3.MiniMaxH3FunControlNetApply.execute(
                model=out_model, model_patch=model_patch, vae=vae, strength=strength,
                start_percent=0.0, end_percent=end, control_video=cv))[0]
        if not any(connected.values()):
            log.warning("MiniMax H3 V2V: no control video connected - motion only follows <Video 1> loosely. "
                        "Connect at least control_pose for motion that matches the input.")

        return (out_model, positive, latent, prompt, src, int(tl.frame_count), float(v2v.H3_FPS))


class MiniMaxH3ConformVideo:
    """Source clip -> 24 fps, 17k+5 frames, H3 canvas. Put it before pose/depth/canny preprocessors."""

    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "INT", "INT", "AUDIO")
    RETURN_NAMES = ("images", "frame_count", "fps", "width", "height", "audio")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01}),
                "max_seconds": ("FLOAT", {"default": 15.0, "min": 0.25, "max": 15.1, "step": 0.01}),
                "resolution": (list(v2v.RESOLUTIONS.keys()), {"default": "768p (native)"}),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    def run(self, images, fps, start_seconds, max_seconds, resolution, audio=None):
        tl = v2v.Timeline(images.shape[0], fps, start_seconds, max_seconds)
        frames = tl.take(images)
        w, h = v2v.canvas_for(frames.shape[2], frames.shape[1], v2v.RESOLUTIONS[resolution])
        frames = v2v.resize_frames(frames, w, h)
        return (frames, int(tl.frame_count), float(v2v.H3_FPS), int(w), int(h), tl.take_audio(audio))


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3V2VEditLLM": MiniMaxH3V2VEditLLM,
    "MiniMaxH3PrompterConformVideo": MiniMaxH3ConformVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3V2VEditLLM": "MiniMax H3 V2V Edit + LLM (Fun ControlNet)",
    "MiniMaxH3PrompterConformVideo": "MiniMax H3 Conform Video (24 fps)",
}
