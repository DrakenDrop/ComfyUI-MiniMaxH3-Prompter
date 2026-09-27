"""ComfyUI nodes: MiniMax H3 R2V Prompter (llama.cpp / llama-server backend).

Label numbering mirrors ComfyUI's core `MiniMax H3 Reference to Video` node:
  * <Picture i>  : connected images in slot order (ref_image_1, ref_image_2, ...)
  * <Video k>    : connected videos (IMAGE frames @24 fps) in slot order
  * <Audio j>    : first the soundtrack of each video that has one (in video order),
                   then the standalone audios in slot order
"""

from __future__ import annotations

import time

from .h3_prompter import llama_client as lc
from .h3_prompter import media
from .h3_prompter import prompts

_CFG = lc.load_config()
MAX_PICTURES = 9   # core H3 R2V node accepts up to 9 reference images
MAX_VIDEOS = 3
MAX_AUDIOS = 3
H3_MAX_TRAINED_FRAMES = 362  # ~15.08 s @24fps, upper end of H3 trained range (17*21+5)

THINKING = ["off", "low", "medium", "xhigh"]
SAMPLING = {
    # Qwen3.8 recommendations (Unsloth docs)
    "off": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5),
    "on": dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0),
}
FIRST_FIELD = "subject_definitions:"
AUDIO_USES = list(prompts.AUDIO_USE.keys())


class MiniMaxH3R2VPrompter:
    CATEGORY = "MiniMax H3/Prompt"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "INT", "FLOAT", "IMAGE", "STRING")
    RETURN_NAMES = ("prompt", "length", "duration_seconds", "first_frame", "reasoning")
    OUTPUT_TOOLTIPS = (
        "Prompt H3 Full-Reference -> sambungkan ke 'prompt' di MiniMax H3 Reference to Video.",
        "Jumlah frame (grid 17k+5 @24fps) -> sambungkan ke 'length' di node H3.",
        "Durasi efektif dalam detik.",
        "image_1 (untuk frame_anchor 'first frame'): sambungkan ke 'Add Guide for MiniMax H3' dengan frame_idx 0.",
        "Teks reasoning (kosong kalau thinking = off).",
    )
    DESCRIPTION = (
        "Writes a MiniMax H3 Full-Reference (R2V) prompt with a local Qwen3.8 GGUF served by llama-server. "
        "Connect images/videos/audio to the H3 Reference to Video node in the SAME slot order as here."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "instruction": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Apa yang kamu mau (boleh Bahasa Indonesia). Tulis dialog dalam tanda kutip; dialog dipertahankan kata per kata.",
            }),
            "task": (list(prompts.TASKS.keys()), {
                "default": "auto",
                "tooltip": "reference generation = gambar jadi referensi; image edit = ubah image_1 lalu animasikan; "
                           "video editing = edit video_1; video continuation = lanjutkan video_1.",
            }),
            "frame_anchor": (prompts.FRAME_ANCHORS, {
                "default": "none",
                "tooltip": "'reference 1 = first frame' -> image_1 jadi frame pertama persis (keyframe completion). "
                           "Sambungkan output first_frame ke 'Add Guide for MiniMax H3' (frame_idx 0).",
            }),
            "duration_seconds": ("FLOAT", {
                "default": 5.0, "min": 0.0, "max": 15.1, "step": 0.1,
                "tooltip": "Durasi target (maks H3 ~15 s = 362 frame), dibulatkan ke grid H3 17k+5 frame @24fps. 0 = ikuti panjang video_1 "
                           "(untuk video editing), selain itu 5 detik.",
            }),
            "thinking": (THINKING, {
                "default": "off",
                "tooltip": "off = tercepat, thinking dijamin mati. low/medium/xhigh = Qwen3.8 reasoning_effort.",
            }),
            "length": (["compact", "standard", "detailed"], {
                "default": "standard",
                "tooltip": "Panjang detailed_description. compact ~150-250 kata (paling cepat), "
                           "standard 350-500 kata (pedoman resmi), detailed 500-700.",
            }),
            "music": (["auto", "none", "include"], {"default": "auto"}),
            "allow_invented_dialogue": ("BOOLEAN", {"default": False}),
            "max_tokens": ("INT", {"default": 2048, "min": 256, "max": 32768, "step": 64}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
        }
        optional = {}
        for i in range(1, MAX_PICTURES + 1):
            optional[f"image_{i}"] = ("IMAGE", {
                "tooltip": "Satu gambar per input (tanpa batch). Gambar yang tersambung dinomori berurutan <Picture 1>, "
                           "<Picture 2>, ... Sambungkan ke ref_image_N di node H3 dengan urutan yang sama.",
            })
        for i in range(1, MAX_VIDEOS + 1):
            optional[f"video_{i}"] = ("IMAGE", {
                "tooltip": "Frame video @24fps (sama dengan ref_video_N di node H3).",
            })
            optional[f"video_{i}_audio"] = ("AUDIO", {
                "tooltip": "Soundtrack video ini (sama dengan ref_video_audio_N di node H3).",
            })
            optional[f"video_{i}_audio_use"] = (AUDIO_USES, {"default": "exact reuse / lip-sync (fully_copy)"})
        for i in range(1, MAX_AUDIOS + 1):
            optional[f"audio_{i}"] = ("AUDIO", {"tooltip": "Audio mandiri (sama dengan ref_audio_N di node H3)."})
            optional[f"audio_{i}_use"] = (AUDIO_USES, {"default": "voice timbre only (reference)"})
        optional.update({
            "asset_notes": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": "Opsional: peran tiap referensi, mis. 'Picture 1 = wajah wanita, Picture 2 = bajunya, Audio 1 = suaranya'.",
            }),
            "extra_rules": ("STRING", {"multiline": True, "default": ""}),
            "video_sample_fps": ("FLOAT", {
                "default": 2.0, "min": 0.5, "max": 6.0, "step": 0.5,
                "tooltip": "Berapa frame per detik video yang dilihat LLM (2 = sama dengan Qwen di dalam H3). "
                           "Hanya bagian video yang benar-benar dipakai H3 yang dikirim. 1 = lebih cepat.",
            }),
            "video_max_side": ("INT", {
                "default": 512, "min": 256, "max": 1536, "step": 64,
                "tooltip": "Resolusi frame video untuk LLM. 512 cukup untuk memahami aksi dan tetap cepat.",
            }),
            "image_max_side": ("INT", {
                "default": 768, "min": 256, "max": 2048, "step": 64,
                "tooltip": "Gambar diperkecil sebelum dikirim ke LLM (tidak mempengaruhi H3). Lebih kecil = lebih cepat.",
            }),
            "sampling": (["qwen recommended", "custom"], {"default": "qwen recommended"}),
            "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
            "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            "top_k": ("INT", {"default": 20, "min": 0, "max": 200}),
            "presence_penalty": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 2.0, "step": 0.05}),
            "server_url": ("STRING", {"default": _CFG.get("server_url", "http://127.0.0.1:8080")}),
            "print_to_console": ("BOOLEAN", {"default": True}),
        })
        return {"required": required, "optional": optional}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _audio_line(label: str, audio, use: str, soundtrack_of: str | None = None) -> str:
        marker = prompts.AUDIO_USE.get(use, "reference")
        d = media.audio_duration(audio)
        dur = f"{d:.2f} s " if d else ""
        if soundtrack_of:
            role = f"the synchronized audio track of {soundtrack_of}"
            role += ", reused in the target video ([audio reuse])" if marker in ("fully_copy", "partially_copy") \
                else ", used only as a timbre/style reference ([audio reference])"
        elif use.startswith("music style"):
            role = "a music-style reference ([audio reference])"
        elif marker in ("fully_copy", "partially_copy"):
            role = "audio reused in the target video ([audio reuse]); its role is described in the request"
        else:
            role = "a voice-timbre / style reference ([audio reference]); its role is described in the request"
        return f"{label}: {dur}audio clip (not shown) - {role}. Retention marker: {marker}."

    def _collect_assets(self, kw, target_frames, video_fps, video_side, max_side):
        parts: list[dict] = []
        pic_desc, vid_desc, aud_desc = [], [], []
        first_image = None

        pic_n = 0
        for slot in range(1, MAX_PICTURES + 1):
            img = kw.get(f"image_{slot}")
            if img is None:
                continue
            batch = media.image_batch_to_pil(img)
            if not batch:
                continue
            if len(batch) > 1:
                lc.log(f"image_{slot} is a batch of {len(batch)}; only the first image is used (like the H3 node).")
            if first_image is None:
                first_image = img[:1] if hasattr(img, "__getitem__") else img
            pil = batch[0]
            pic_n += 1
            w, h = pil.size
            pic_desc.append(f"<Picture {pic_n}>: still image ({w}x{h}) from input image_{slot}, shown below.")
            parts.append({"type": "text", "text": f"<Picture {pic_n}>:"})
            parts.append({"type": "image_url", "image_url": {"url": media.pil_to_data_url(pil, max_side)}})

        vid_n = 0
        aud_n = 0
        for slot in range(1, MAX_VIDEOS + 1):
            vid = kw.get(f"video_{slot}")
            if vid is None or media.video_len(vid) == 0:
                continue
            vid_n += 1
            n_src = media.video_len(vid)
            # same trimming as the H3 node: first `target_frames`, then down to 17k+5
            used, used_s = media.h3_frames_floor(min(n_src, target_frames))
            soundtrack = kw.get(f"video_{slot}_audio")
            if soundtrack is not None:
                # core node emits the soundtrack's <Audio j> right before its <Video k>
                aud_n += 1
                aud_desc.append(self._audio_line(
                    f"<Audio {aud_n}>", soundtrack,
                    kw.get(f"video_{slot}_audio_use", "exact reuse / lip-sync (fully_copy)"),
                    soundtrack_of=f"<Video {vid_n}>",
                ))
            samples = media.video_sample(vid, video_fps, limit=used)
            trim = f" (source has {n_src} frames; only the first {used} are used)" if n_src > used else ""
            vid_desc.append(
                f"<Video {vid_n}>: video of {used_s:.2f} s ({used} frames @24fps){trim} from input video_{slot}; "
                f"{len(samples)} frames shown below at {video_fps:g} fps with timestamps."
            )
            for i, pil in samples:
                parts.append({"type": "text", "text": f"<Video {vid_n}> t={i / media.H3_FPS:.1f}s:"})
                parts.append({"type": "image_url", "image_url": {"url": media.pil_to_data_url(pil, video_side)}})

        for slot in range(1, MAX_AUDIOS + 1):
            aud = kw.get(f"audio_{slot}")
            if aud is None:
                continue
            aud_n += 1
            aud_desc.append(self._audio_line(
                f"<Audio {aud_n}>", aud, kw.get(f"audio_{slot}_use", "voice timbre only (reference)")))

        return parts, pic_desc, vid_desc, aud_desc, first_image

    # ------------------------------------------------------------------ main
    def generate(self, instruction, task, frame_anchor, duration_seconds, thinking, length, music,
                 allow_invented_dialogue, max_tokens, seed, **kw):
        server_url = (kw.get("server_url") or _CFG["server_url"]).strip()
        print_tokens = kw.get("print_to_console", True)
        lc.ensure_server(server_url, _CFG)

        vframes = [media.video_len(kw[f"video_{i}"]) for i in range(1, MAX_VIDEOS + 1)
                   if kw.get(f"video_{i}") is not None]
        vframes = [n for n in vframes if n > 0]
        if duration_seconds <= 0:
            if vframes and task in ("video editing", "auto"):
                n_src = vframes[0]
                if n_src > H3_MAX_TRAINED_FRAMES:
                    lc.log(f"first video has {n_src} frames; auto length capped to {H3_MAX_TRAINED_FRAMES} "
                           "(H3 max ~15 s). The H3 node only uses the first part of the video.")
                    n_src = H3_MAX_TRAINED_FRAMES
                frames, eff = media.h3_frames_floor(n_src)
            else:
                frames, eff = media.h3_frames(5.0)
        else:
            frames, eff = media.h3_frames(duration_seconds)
        if frames > H3_MAX_TRAINED_FRAMES:
            lc.log(f"length {frames} frames is above the H3 trained range (max {H3_MAX_TRAINED_FRAMES} = ~15 s).")

        parts, pics, vids, auds, first_image = self._collect_assets(
            kw, frames, kw.get("video_sample_fps", 2.0), kw.get("video_max_side", 512),
            kw.get("image_max_side", 768))

        user_text = prompts.build_user_text(
            instruction=instruction, task=task, frame_anchor=frame_anchor, duration_s=eff, frames=frames,
            pictures=pics, videos=vids, audios=auds, length=length,
            allow_invented_dialogue=allow_invented_dialogue, music=music,
            asset_notes=kw.get("asset_notes", ""), extra_rules=kw.get("extra_rules", ""),
        )
        user_content = [{"type": "text", "text": user_text}, *parts] if parts else user_text

        think_on = thinking != "off"
        if kw.get("sampling", "qwen recommended") == "custom":
            samp = dict(temperature=kw.get("temperature", 0.7), top_p=kw.get("top_p", 0.8),
                        top_k=kw.get("top_k", 20), min_p=0.0, presence_penalty=kw.get("presence_penalty", 1.5))
        else:
            samp = dict(SAMPLING["on" if think_on else "off"])

        base = {
            "model": _CFG.get("model_alias", "qwen3.8-27b"),
            "max_tokens": int(max_tokens),
            "seed": int(seed),
            "cache_prompt": True,
            **samp,
        }
        system = {"role": "system", "content": prompts.SYSTEM_PROMPT_R2V}
        user = {"role": "user", "content": user_content}

        progress = None
        try:
            import comfy.utils  # type: ignore

            progress = comfy.utils.ProgressBar(int(max_tokens))
        except Exception:  # noqa: BLE001
            pass

        def on_token(_tok):
            if progress is not None:
                progress.update(1)

        timeout = float(_CFG.get("request_timeout_seconds", 600))
        t0 = time.time()
        content, reasoning, timings = self._run(
            server_url, base, system, user, thinking, timeout, print_tokens, on_token)

        prompt = prompts.clean_output(content)
        dt = time.time() - t0
        tps = timings.get("predicted_per_second")
        lc.log(
            f"done in {dt:.1f}s"
            + (f" | {tps:.1f} tok/s" if tps else "")
            + (f" | prompt {timings.get('prompt_n')} tok (cached {timings.get('cache_n', 0)})" if timings else "")
            + f" | length {frames} frames = {eff:.2f}s"
        )
        if first_image is None:
            first_image = _blank_image()
        return (prompt, int(frames), float(eff), first_image, reasoning)

    @staticmethod
    def _run(server_url, base, system, user, thinking, timeout, print_tokens, on_token):
        if thinking == "off":
            attempts = [
                # 1) template switch + prefill of the first field -> the model starts writing the answer at once
                ({"enable_thinking": False}, FIRST_FIELD + "\n"),
                # 2) server refused the prefill -> template switch only (reasoning guard still active)
                ({"enable_thinking": False}, None),
                # 3) template ignored the switch -> hard prefill of an empty think block
                ({"enable_thinking": False}, "<think>\n\n</think>\n\n" + FIRST_FIELD + "\n"),
                # 4) last resort: let it run, strip any reasoning afterwards
                (None, None),
            ]
            last_err = None
            for kwargs, prefill in attempts:
                msgs = [system, user]
                if prefill:
                    msgs.append({"role": "assistant", "content": prefill})
                payload = dict(base, messages=msgs)
                if kwargs:
                    payload["chat_template_kwargs"] = kwargs
                try:
                    content, _, timings = lc.stream_chat(
                        server_url, payload, timeout=timeout, abort_on_reasoning=kwargs is not None,
                        print_tokens=print_tokens, on_token=on_token,
                    )
                except lc.ReasoningDetected:
                    lc.log("model started thinking although thinking=off -> aborted, retrying with hard no-think.")
                    last_err = "reasoning detected"
                    continue
                except lc.ServerError as exc:
                    lc.log(f"attempt failed: {exc}")
                    last_err = exc
                    continue
                if prefill and not content.lstrip().startswith(FIRST_FIELD):
                    content = FIRST_FIELD + "\n" + content.lstrip()
                if content.strip():
                    return content, "", timings
            raise RuntimeError(f"[H3 Prompter] no prompt generated: {last_err}")

        content, reasoning, timings = "", "", {}
        for kwargs in ({"enable_thinking": True, "reasoning_effort": thinking}, {"enable_thinking": True}):
            payload = dict(base, messages=[system, user], chat_template_kwargs=kwargs)
            try:
                content, reasoning, timings = lc.stream_chat(
                    server_url, payload, timeout=timeout, print_tokens=print_tokens, on_token=on_token)
                break
            except lc.ServerError as exc:
                lc.log(f"thinking request failed ({exc}); retrying without reasoning_effort.")
        if not content.strip():
            lc.log("thinking used up max_tokens without a final answer -> fast retry with thinking off.")
            payload = dict(base, **SAMPLING["off"],
                           messages=[system, user, {"role": "assistant", "content": FIRST_FIELD + "\n"}],
                           chat_template_kwargs={"enable_thinking": False})
            content, _, timings = lc.stream_chat(server_url, payload, timeout=timeout, print_tokens=print_tokens)
            if not content.lstrip().startswith(FIRST_FIELD):
                content = FIRST_FIELD + "\n" + content.lstrip()
        return content, reasoning, timings


def _blank_image():
    try:
        import torch  # type: ignore

        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    except ImportError:  # tests without torch
        import numpy as np

        return np.zeros((1, 64, 64, 3), dtype=np.float32)


class MiniMaxH3Frames:
    """Duration -> H3 frame count (17k+5 @ 24 fps), to feed the video node with the same value."""

    CATEGORY = "MiniMax H3/Prompt"
    FUNCTION = "calc"
    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("length", "duration_seconds")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seconds": ("FLOAT", {"default": 5.0, "min": 0.2, "max": 30.0, "step": 0.1})}}

    def calc(self, seconds):
        frames, eff = media.h3_frames(seconds)
        return (frames, float(eff))


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3R2VPrompter": MiniMaxH3R2VPrompter,
    "MiniMaxH3Frames": MiniMaxH3Frames,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3R2VPrompter": "MiniMax H3 R2V Prompter (llama.cpp)",
    "MiniMaxH3Frames": "MiniMax H3 Duration → Frames",
}
