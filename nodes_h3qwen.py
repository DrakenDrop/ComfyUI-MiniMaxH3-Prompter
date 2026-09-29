"""MiniMax H3 + Qwen-Image 2.1 keyframe video edit (new, independent of the V2V Edit node).

source clip -> H3 timeline (24 fps, 17k+5) + canvas
  -> keyframes every N seconds (+ last frame)
  -> Qwen-Image 2.1 edits keyframe 0 from the instruction (+ reference images)
  -> Qwen-Image 2.1 edits every other keyframe WITH the edited keyframe 0 as reference (same look everywhere)
  -> H3 prompt (LLM or template)
  -> MiniMaxH3ReferenceToVideo (<Video 1> = source, refs, edited keyframe 0 = last <Picture>)
  -> MiniMaxH3AddGuide at every keyframe index (the edited keyframes are pinned in time)
  -> Fun ControlNet pose / depth / edge (motion from the source)
H3 fills the frames between the keyframes, so the edit has Qwen's quality and the video stays smooth.
"""

from __future__ import annotations

import hashlib

from . import nodes as prompter_nodes
from .h3_prompter import llama_client as lc
from .h3_prompter import local_models
from .h3_prompter import managed_server
from .h3_prompter import v2v

try:
    from comfy_extras import nodes_minimax_h3 as h3  # type: ignore
    _H3_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    h3 = None
    _H3_IMPORT_ERROR = e

CATEGORY = "MiniMax H3/H3 + Qwen Image"
_CFG = prompter_nodes._CFG
MAX_REFS = 4
PROMPT_MODES = ["LLM simple", "LLM full (official H3)", "template (no LLM)"]


def _args(o):
    return o.args if hasattr(o, "args") else o


def _sampler_lists():
    try:
        import comfy.samplers  # type: ignore

        return list(comfy.samplers.KSampler.SAMPLERS), list(comfy.samplers.KSampler.SCHEDULERS)
    except Exception:  # noqa: BLE001
        return ["euler"], ["simple"]


def keyframe_indices(frame_count: int, every_seconds: float, include_last: bool) -> list[int]:
    step = max(1, int(round(every_seconds * v2v.H3_FPS)))
    idx = list(range(0, frame_count, step))
    last = frame_count - 1
    if include_last and last not in idx:
        if last - idx[-1] < step // 2 and len(idx) > 1:
            idx[-1] = last  # a keyframe very close to the end -> move it to the end instead of adding one
        else:
            idx.append(last)
    return idx


def _sig(t) -> str:
    return hashlib.md5(repr(v2v.tensor_sig(t)).encode()).hexdigest()


class MiniMaxH3QwenKeyframeEdit:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "STRING", "IMAGE", "IMAGE", "IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("model", "positive", "latent", "prompt", "source_frames", "edited_keyframes", "source_keyframes",
                    "keyframe_indices", "frame_count", "fps")
    DESCRIPTION = (
        "Video edit: Qwen-Image 2.1 edits a few keyframes (every N seconds + the last frame; keyframe 0 is the "
        "reference for the others so the look stays the same), then MiniMax H3 generates the whole video through "
        "those keyframes with the source clip as <Video 1> and pose/depth/edge control. Wire model+positive to "
        "BasicGuider, model to BasicScheduler, latent to SamplerCustomAdvanced (denoise 1.0). Audio: take it from the "
        "source video."
    )

    _qwen_cache: dict = {}
    _prompt_cache: dict = {}

    @classmethod
    def INPUT_TYPES(cls):
        samplers, schedulers = _sampler_lists()
        optional = {
            "model_patch": ("MODEL_PATCH", {"tooltip": "MiniMax H3 Fun ControlNet-Union (ModelPatchLoader)."}),
            "control_pose": ("IMAGE", {"tooltip": "Pose video of the SAME clip (e.g. from Conform Video -> DWPose)."}),
            "control_depth": ("IMAGE", {"tooltip": "Depth video of the same clip."}),
            "control_edge": ("IMAGE", {"tooltip": "Canny / HED video of the same clip."}),
        }
        for i in range(1, MAX_REFS + 1):
            optional[f"ref_image_{i}"] = ("IMAGE", {
                "tooltip": f"Reference {i} for the edit (e.g. the new dress). Qwen sees it for every keyframe and H3 "
                           f"gets it as <Picture {i}>."})
        optional.update({
            "include_last_frame": ("BOOLEAN", {"default": True,
                                               "tooltip": "Also edit and pin the last frame (keeps the end on-model)."}),
            "qwen_prompt": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": "Optional English edit instruction for Qwen (else `instruction` is used). Write the change "
                           "only, e.g. 'change her dress to a burgundy satin mini dress'.",
            }),
            "qwen_steps": ("INT", {"default": 25, "min": 1, "max": 100}),
            "qwen_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            "qwen_sampler": (samplers, {"default": "euler" if "euler" in samplers else samplers[0]}),
            "qwen_scheduler": (schedulers, {"default": "simple" if "simple" in schedulers else schedulers[0]}),
            "qwen_resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32,
                                        "tooltip": "Qwen works at about this x this pixels (frame aspect kept); the "
                                                   "result is resized to the H3 canvas."}),
            "h3_prompt_mode": (PROMPT_MODES, {"default": "LLM simple"}),
            "llm_model": (local_models.model_choices(_CFG), {}),
            "mmproj": (local_models.mmproj_choices(_CFG), {"default": local_models.MMPROJ_AUTO}),
            "thinking": (prompter_nodes.THINKING, {"default": "off"}),
            "prompt_override": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "If not empty, used as the H3 prompt (LLM and template skipped)."}),
            "pose_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 3.0, "step": 0.01}),
            "depth_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.01,
                                         "tooltip": "0 for outfit/shape changes (depth pins the old silhouette)."}),
            "edge_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            "control_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "use_source_as_reference": ("BOOLEAN", {"default": True,
                                                    "tooltip": "Give the source clip to H3 as <Video 1> (identity, "
                                                               "background, lip sync)."}),
            "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01}),
            "max_seconds": ("FLOAT", {"default": 15.0, "min": 0.25, "max": 15.1, "step": 0.01}),
            "frame_count": ("INT", {"default": 0, "min": 0, "max": 362, "step": 1,
                                    "tooltip": "0 = from the source (+ max_seconds); else nearest 17k+5 value."}),
            "resolution": (list(v2v.RESOLUTIONS.keys()), {"default": "768p (native)"}),
            "ref_image_size": (["match", "max"], {"default": "match"}),
            "video_sample_fps": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 6.0, "step": 0.5}),
            "unload_llm_after_prompt": ("BOOLEAN", {"default": True}),
            "context_size": ("INT", {"default": int(_CFG.get("context_size", 32768)), "min": 4096,
                                     "max": 262144, "step": 1024}),
        })
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MiniMax H3 ref2va model."}),
                "clip": ("CLIP", {"tooltip": "MiniMax H3 text encoder."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "qwen_model": ("MODEL", {"tooltip": "Qwen-Image 2.1 diffusion model."}),
                "qwen_clip": ("CLIP", {"tooltip": "Qwen-Image 2.1 text encoder."}),
                "qwen_vae": ("VAE", {"tooltip": "Qwen-Image 2.1 VAE."}),
                "source_video": ("IMAGE",),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "instruction": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "What to change, e.g. 'ganti dress-nya jadi dress satin merah "
                                                      "seperti referensi 1'."}),
                "keyframe_every_seconds": ("FLOAT", {
                    "default": 2.5, "min": 0.5, "max": 15.0, "step": 0.25,
                    "tooltip": "Distance between edited keyframes. 10 s video: 2.5 -> 5 Qwen edits, 5 -> 3.",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "tooltip": "Qwen seed (the same for every keyframe) and LLM seed."}),
            },
            "optional": optional,
        }

    # ------------------------------------------------------------------ Qwen-Image 2.1
    @staticmethod
    def _qwen_edit(qwen_model, qwen_clip, qwen_vae, prompt, images, seed, steps, cfg, sampler, scheduler, resolution):
        import nodes as comfy_nodes  # type: ignore
        from comfy_extras import nodes_qwen  # type: ignore

        if not hasattr(nodes_qwen, "TextEncodeQwenImage21"):
            raise RuntimeError("This ComfyUI has no Qwen-Image 2.1 support (TextEncodeQwenImage21). Update ComfyUI "
                               "to 0.37.0 or newer.")
        positive, negative, latent = _args(nodes_qwen.TextEncodeQwenImage21.execute(
            clip=qwen_clip, prompt=prompt, negative_prompt="", vae=qwen_vae, resolution=int(resolution),
            images={f"image_{i}": im for i, im in enumerate(images, start=1)}))[:3]
        out = comfy_nodes.common_ksampler(qwen_model, int(seed), int(steps), float(cfg), sampler, scheduler,
                                          positive, negative, latent, denoise=1.0)[0]
        img = qwen_vae.decode(out["samples"])
        if img.ndim == 5:
            img = img.reshape(-1, *img.shape[-3:])
        return img[:1, ..., :3].float().clamp(0, 1)

    def _edit_keyframes(self, src_kf, refs, qwen_model, qwen_clip, qwen_vae, text, seed, steps, cfg, sampler,
                        scheduler, resolution, width, height):
        """Keyframe 0 from the instruction (+refs); the others with the edited keyframe 0 as the look reference."""
        import torch

        keep = ("Keep everything else in <image1> exactly the same: the person's identity, face, hair, skin, body, "
                "pose, the background, lighting, framing and composition.")
        n_refs = len(refs)
        edited = []
        for k in range(src_kf.shape[0]):
            frame = src_kf[k:k + 1]
            if k == 0:
                imgs = [frame, *refs]
                ref_txt = (" Reference image(s): " + ", ".join(f"<image{i}>" for i in range(2, n_refs + 2)) + "."
                           if n_refs else "")
                prompt = f"Edit <image1>: {text}.{ref_txt} {keep}"
            else:
                imgs = [frame, edited[0], *refs]
                prompt = (f"Edit <image1>: {text}. The edited element must look exactly as it does in <image2> "
                          "(same design, color, material, pattern and details), adapted to the pose in <image1>. "
                          + keep)
            key = (_sig(frame), tuple(_sig(i) for i in imgs[1:]), prompt, int(seed), int(steps), float(cfg), sampler,
                   scheduler, int(resolution), id(qwen_model), id(qwen_clip), id(qwen_vae))
            res = self._qwen_cache.get(key)
            if res is None:
                lc.log(f"H3+Qwen: Qwen-Image 2.1 edit {k + 1}/{src_kf.shape[0]} ...")
                res = self._qwen_edit(qwen_model, qwen_clip, qwen_vae, prompt, imgs, seed, steps, cfg, sampler,
                                      scheduler, resolution)
                res = v2v.resize_frames(res, width, height).cpu()
                if len(self._qwen_cache) > 64:
                    self._qwen_cache.clear()
                self._qwen_cache[key] = res
            else:
                lc.log(f"H3+Qwen: keyframe {k + 1}/{src_kf.shape[0]} unchanged -> cached Qwen edit.")
            edited.append(res)
        return torch.cat(edited, dim=0)

    # ------------------------------------------------------------------ H3 prompt
    def _h3_prompt(self, *, mode, instruction, llm_model, mmproj, thinking, seed, src, refs, first, frames, duration,
                   use_src, video_sample_fps, context_size, n_kf, every):
        ff_label = f"<Picture {len(refs) + 1}>"
        if mode.startswith("template"):
            if use_src:
                return (f"[video editing + keyframe completion] The target video is an edited version of <Video 1> and "
                        f"begins from {ff_label}, the edited first frame: {instruction.strip()}. The edited look stays "
                        "exactly the same in every frame; the motion, timing, camera, framing, face and background "
                        "follow <Video 1>.")
            return (f"[keyframe completion + reference generation] The video begins from {ff_label}: "
                    f"{instruction.strip()}. The look stays exactly the same in every frame.")
        key = (mode, instruction, llm_model, mmproj, thinking, seed, v2v.tensor_sig(src), v2v.tensor_sig(first),
               tuple(v2v.tensor_sig(r) for r in refs), frames, use_src, video_sample_fps)
        if key in self._prompt_cache:
            lc.log("H3+Qwen: LLM inputs unchanged -> cached prompt.")
            return self._prompt_cache[key]
        rules = (f"KEYFRAMES: the same edit is pinned at {n_kf} keyframes (about every {every:g} s) edited like "
                 f"{ff_label}, so the edited element looks identical for the whole video.")
        if not use_src:
            rules += ("\nThe source clip is NOT supplied as <Video 1>: never mention <Video 1>; describe the whole "
                      "resulting scene explicitly (the motion comes from control videos).")
        kw = {f"image_{i}": r for i, r in enumerate(refs, start=1)}
        kw["first_frame"] = first
        if use_src:
            kw["video_1"] = src
        out = prompter_nodes.MiniMaxH3R2VPrompter().generate(
            instruction=instruction, task="video editing" if use_src else "reference generation", frame_anchor="none",
            duration_seconds=duration, thinking=thinking, length="standard", allow_invented_dialogue=False,
            max_tokens=2048, seed=int(seed) & 0xFFFFFFFF, model=llm_model, mmproj=mmproj, extra_rules=rules,
            video_sample_fps=video_sample_fps, video_max_side=512, image_max_side=768, context_size=context_size,
            print_to_console=True, frame_count=frames, describe_refs=True, edit_mode="custom",
            prompt_style="simple" if mode == "LLM simple" else "full (official H3)", **kw)
        prompt = out[0]
        if len(self._prompt_cache) > 16:
            self._prompt_cache.clear()
        self._prompt_cache[key] = prompt
        return prompt

    # ------------------------------------------------------------------ main
    def run(self, model, clip, vae, qwen_model, qwen_clip, qwen_vae, source_video, source_fps, instruction,
            keyframe_every_seconds, seed, model_patch=None, control_pose=None, control_depth=None, control_edge=None,
            include_last_frame=True, qwen_prompt="", qwen_steps=25, qwen_cfg=1.0, qwen_sampler="euler",
            qwen_scheduler="simple", qwen_resolution=1024, h3_prompt_mode="LLM simple",
            llm_model=local_models.SERVER_DEFAULT, mmproj=local_models.MMPROJ_AUTO, thinking="off", prompt_override="",
            pose_strength=0.9, depth_strength=0.0, edge_strength=0.0, control_end_percent=1.0,
            use_source_as_reference=True, start_seconds=0.0, max_seconds=15.0, frame_count=0,
            resolution="768p (native)", ref_image_size="match", video_sample_fps=2.0, unload_llm_after_prompt=True,
            context_size=32768, **refs_kw):
        if h3 is None or not hasattr(h3, "MiniMaxH3FunControlNetApply"):
            raise RuntimeError(f"needs a ComfyUI with native MiniMax H3 nodes. Import error: {_H3_IMPORT_ERROR}")

        # ---- timeline, canvas, keyframes ------------------------------------------------------
        tl = v2v.Timeline(source_video.shape[0], source_fps, start_seconds, max_seconds, frame_count=frame_count)
        src = tl.take(source_video, "source_video")
        width, height = v2v.canvas_for(src.shape[2], src.shape[1], v2v.RESOLUTIONS[resolution])
        src = v2v.resize_frames(src, width, height)
        idx = keyframe_indices(tl.frame_count, keyframe_every_seconds, include_last_frame)
        src_kf = src[idx]
        refs = [refs_kw[f"ref_image_{i}"][:1, ..., :3] for i in range(1, MAX_REFS + 1)
                if refs_kw.get(f"ref_image_{i}") is not None]
        lc.log(f"H3+Qwen: {tl.frame_count} frames ({tl.duration:.2f}s) at {width}x{height}; {len(idx)} keyframes "
               f"at frames {idx} -> {len(idx)} Qwen edits")
        text = (qwen_prompt or instruction).strip().rstrip(".")
        if not text:
            raise ValueError("Write an instruction (what to change).")

        # ---- Qwen-Image 2.1: edit the keyframes -------------------------------------------------
        edited = self._edit_keyframes(src_kf, refs, qwen_model, qwen_clip, qwen_vae, text, seed, qwen_steps, qwen_cfg,
                                      qwen_sampler, qwen_scheduler, qwen_resolution, width, height)
        first = edited[:1]

        # ---- H3 prompt ------------------------------------------------------------------------
        prompt = (prompt_override or "").strip()
        if not prompt:
            prompt = self._h3_prompt(
                mode=h3_prompt_mode, instruction=instruction.strip() or text, llm_model=llm_model, mmproj=mmproj,
                thinking=thinking, seed=seed, src=src, refs=refs, first=first, frames=tl.frame_count,
                duration=tl.duration, use_src=use_source_as_reference, video_sample_fps=video_sample_fps,
                context_size=context_size, n_kf=len(idx), every=keyframe_every_seconds)
            if unload_llm_after_prompt and not h3_prompt_mode.startswith("template"):
                managed_server.stop(_CFG)

        # ---- H3 conditioning: refs, <Video 1>, keyframes pinned in time --------------------------
        core_refs = {f"ref_image_{i}": r for i, r in enumerate([*refs, first])}
        positive, latent = _args(h3.MiniMaxH3ReferenceToVideo.execute(
            clip=clip, prompt=prompt, width=width, height=height, length=tl.frame_count,
            ref_image_size=ref_image_size, vae=vae, audio_vae=None, ref_images=core_refs,
            ref_videos={"ref_video_0": src} if use_source_as_reference else None,
            ref_video_audios=None, ref_audios=None))[:2]
        for k, fi in enumerate(idx):
            positive = _args(h3.MiniMaxH3AddGuide.execute(
                positive=positive, latent=latent, frame_idx=int(fi), vae=vae, image=edited[k:k + 1]))[0]

        # ---- motion: Fun ControlNet-Union ------------------------------------------------------
        out_model = model
        controls = {"pose": (control_pose, pose_strength), "depth": (control_depth, depth_strength),
                    "edge": (control_edge, edge_strength)}
        if any(v is not None and s > 0 for v, s in controls.values()) and model_patch is None:
            raise ValueError("control videos are connected but model_patch (MiniMax H3 Fun ControlNet-Union) is missing")
        for name, (video, strength) in controls.items():
            if video is None or strength <= 0:
                continue
            cv = v2v.resize_frames(tl.take(video, "control_" + name), width, height)
            lc.log(f"H3+Qwen: control {name} strength={strength:.2f} end={control_end_percent:.2f}")
            out_model = _args(h3.MiniMaxH3FunControlNetApply.execute(
                model=out_model, model_patch=model_patch, vae=vae, strength=float(strength), start_percent=0.0,
                end_percent=float(control_end_percent), control_video=cv))[0]

        return (out_model, positive, latent, prompt, src, edited, src_kf.cpu(), ",".join(str(i) for i in idx),
                int(tl.frame_count), float(v2v.H3_FPS))


NODE_CLASS_MAPPINGS = {"MiniMaxH3QwenKeyframeEdit": MiniMaxH3QwenKeyframeEdit}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3QwenKeyframeEdit": "MiniMax H3 + Qwen-Image 2.1 Keyframe Video Edit"}
