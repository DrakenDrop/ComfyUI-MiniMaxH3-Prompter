"""ComfyUI nodes: MiniMax H3 R2V Prompter (llama.cpp / llama-server backend).

Label numbering mirrors ComfyUI's core `MiniMax H3 Reference to Video` node:
  * <Picture i>  : connected images in slot order (ref_image_1, ref_image_2, ...)
  * <Video k>    : connected videos (IMAGE frames @24 fps) in slot order
  * <Audio j>    : first the soundtrack of each video that has one (in video order),
                   then the standalone audios in slot order
"""

from __future__ import annotations

import os
import time

from .h3_prompter import llama_client as lc
from .h3_prompter import local_models
from .h3_prompter import managed_server
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
# factual description of a reference picture: low temperature, no creativity
CAPTION_SAMPLING = dict(temperature=0.2, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=0.0)


class MiniMaxH3R2VPrompter:
    CATEGORY = "MiniMax H3/Prompt"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "INT", "FLOAT", "IMAGE", "STRING", "IMAGE", "INT")
    RETURN_NAMES = ("prompt", "length", "duration_seconds", "first_frame", "reasoning", "keyframe_image",
                    "keyframe_frame_idx")
    OUTPUT_TOOLTIPS = (
        "Prompt H3 Full-Reference -> sambungkan ke 'prompt' di MiniMax H3 Reference to Video.",
        "Jumlah frame (grid 17k+5 @24fps) -> sambungkan ke 'length' di node H3.",
        "Durasi efektif dalam detik.",
        "Input first_frame (atau image_1 untuk frame_anchor 'first frame'): sambungkan ke 'Add Guide for MiniMax H3' "
        "dengan frame_idx 0. Tidak perlu kalau memakai MiniMax H3 V2V Edit (node itu sudah melakukannya).",
        "Teks reasoning (kosong kalau thinking = off).",
        "Gambar keyframe (keyframe_picture) -> 'Add Guide for MiniMax H3'.image.",
        "Frame index keyframe -> 'Add Guide for MiniMax H3'.frame_idx.",
    )
    DESCRIPTION = (
        "Writes a MiniMax H3 Full-Reference (R2V) prompt with a local Qwen3.8 GGUF served by llama-server. "
        "Connect images/videos/audio to the H3 Reference to Video node in the SAME slot order as here."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": (local_models.model_choices(_CFG), {
                "tooltip": "GGUF dari ComfyUI/models/LLM (dicari otomatis, termasuk subfolder). Node menjalankan "
                           "llama-server sendiri dan model tetap di VRAM sampai kamu memilih model lain. "
                           "'(llama-server yang sudah jalan)' = pakai server dari start_llama_server.bat / server_url.",
            }),
            "mmproj": (local_models.mmproj_choices(_CFG), {
                "default": local_models.MMPROJ_AUTO,
                "tooltip": "File vision untuk model. auto = cari mmproj di folder yang sama dengan model. "
                           "none = text only (tidak bisa melihat image/video).",
            }),
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
        optional["first_frame"] = ("IMAGE", {
            "tooltip": "Frame 0 yang sudah diedit (sama dengan input first_frame di MiniMax H3 V2V Edit). "
                       "Diberi label <Picture> TERAKHIR (setelah image_1..n), persis seperti node V2V.",
        })
        for i in range(1, MAX_VIDEOS + 1):
            optional[f"video_{i}"] = ("IMAGE", {
                "tooltip": "Frame video @24fps (sama dengan ref_video_N di node H3).",
            })
            optional[f"video_{i}_audio"] = ("AUDIO", {
                "tooltip": "Soundtrack video ini (sama dengan ref_video_audio_N di node H3).",
            })
        for i in range(1, MAX_AUDIOS + 1):
            optional[f"audio_{i}"] = ("AUDIO", {"tooltip": "Audio mandiri (sama dengan ref_audio_N di node H3)."})
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
            "context_size": ("INT", {
                "default": int(_CFG.get("context_size", 32768)), "min": 4096, "max": 262144, "step": 1024,
                "tooltip": "Context llama-server yang dijalankan node (mengganti nilai ini me-restart server).",
            }),
            "server_url": ("STRING", {
                "default": _CFG.get("server_url", "http://127.0.0.1:8080"),
                "tooltip": "Hanya dipakai kalau model = '(llama-server yang sudah jalan)'.",
            }),
            "print_to_console": ("BOOLEAN", {"default": True}),
            # --- added later: keep NEW widgets at the END so saved workflows keep their widget values ---
            "keyframe_picture": ("INT", {
                "default": 0, "min": 0, "max": 9,
                "tooltip": "Nomor <Picture N> yang dijadikan keyframe di tengah video (0 = off). Sambungkan output "
                           "keyframe_image + keyframe_frame_idx ke 'Add Guide for MiniMax H3'.",
            }),
            "keyframe_seconds": ("FLOAT", {
                "default": 2.5, "min": 0.0, "max": 15.1, "step": 0.05,
                "tooltip": "Detik ke berapa keyframe_picture harus muncul persis (dibulatkan ke frame @24fps).",
            }),
            "shots": (["auto", "1", "2", "3", "4", "5", "6"], {
                "default": "auto",
                "tooltip": "Jumlah shot. auto = LLM membuat storyboard sendiri dari prompt sederhana (potongan shot + "
                           "waktunya). Tidak dipakai untuk video editing (shot mengikuti video asli).",
            }),
            "timed_beats": ("BOOLEAN", {
                "default": False,
                "tooltip": "LLM juga menulis waktu aksi di dalam shot (mis. 'At 1.5 s she turns'). Eksperimental - bukan "
                           "format resmi H3. Diabaikan untuk video editing.",
            }),
            "frame_count": ("INT", {
                "default": 0, "min": 0, "max": 362, "step": 1,
                "tooltip": "Panjang video dalam frame (@24fps): 124, 141, ... 345, 362. Kalau > 0, menggantikan "
                           "duration_seconds. Nilai di luar grid 17k+5 dibulatkan ke yang terdekat.",
            }),
            "prompt_style": (["full (official H3)", "simple"], {
                "default": "full (official H3)",
                "tooltip": "simple = 1-3 kalimat yang hanya menjelaskan perubahannya (tanpa 6 bagian, tanpa [Shot]). "
                           "Untuk video editing: '[video editing] The target video is an edited version of <Video 1>: "
                           "<perubahan>. Everything else stays exactly as in <Video 1>.' Lebih cepat.",
            }),
            "describe_refs": ("BOOLEAN", {
                "default": True,
                "tooltip": "Sebelum menulis prompt, LLM melihat tiap gambar ref SENDIRIAN (tanpa frame video) dan "
                           "mendeskripsikannya (+~1-2 s per gambar, di-cache). Deskripsi itu dipakai sebagai fakta, jadi "
                           "detail baju/objek tidak tertukar dengan isi video. Hasilnya dicetak di console.",
            }),
        })
        return {"required": required, "optional": optional}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _audio_line(label: str, audio, soundtrack_of: str | None = None) -> str:
        d = media.audio_duration(audio)
        dur = f"{d:.2f} s " if d else ""
        if soundtrack_of:
            return (f"{label}: {dur}synchronized audio track of {soundtrack_of} (not shown). "
                    "Choose its use and marker from the request.")
        return f"{label}: {dur}standalone audio clip (not shown). Choose its role, use and marker from the request."

    def _collect_assets(self, kw, target_frames, video_fps, video_side, max_side):
        parts: list[dict] = []
        pic_desc, vid_desc, aud_desc = [], [], []
        first_image = None

        pic_parts: list[dict] = []  # pictures go AFTER the video frames (see below)
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
            kw.setdefault("_pic_tensors", {})[pic_n] = img[:1] if hasattr(img, "__getitem__") else img
            w, h = pil.size
            pic_desc.append(f"<Picture {pic_n}>: reference still image ({w}x{h}) from reference input {slot}, shown "
                            "below - a separate image, NOT a frame of any video.")
            url = media.pil_to_data_url(pil, max_side)
            kw.setdefault("_pic_urls", []).append((pic_n, url, "reference image"))
            pic_parts.append({"type": "text", "text": f"<Picture {pic_n}> (reference image):"})
            pic_parts.append({"type": "image_url", "image_url": {"url": url}})

        ff = kw.get("first_frame")
        ff_label = None
        if ff is not None:
            batch = media.image_batch_to_pil(ff)
            if batch:
                pil = batch[0]
                pic_n += 1
                ff_label = f"<Picture {pic_n}>"
                w, h = pil.size
                pic_desc.append(f"{ff_label}: EDITED FIRST FRAME of the target video ({w}x{h}), from input first_frame, "
                                "shown below. It is pinned at frame 0.")
                url = media.pil_to_data_url(pil, max_side)
                kw.setdefault("_pic_urls", []).append((pic_n, url, "edited first frame"))
                pic_parts.append({"type": "text", "text": f"{ff_label} (edited first frame):"})
                pic_parts.append({"type": "image_url", "image_url": {"url": url}})
                first_image = ff[:1] if hasattr(ff, "__getitem__") else ff
        kw["_ff_label"] = ff_label

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
                aud_desc.append(self._audio_line(f"<Audio {aud_n}>", soundtrack, soundtrack_of=f"<Video {vid_n}>"))
            samples = media.video_sample(vid, video_fps, limit=used)
            trim = f" (source has {n_src} frames; only the first {used} are used)" if n_src > used else ""
            vid_desc.append(
                f"<Video {vid_n}>: video of {used_s:.2f} s ({used} frames @24fps){trim} from input video_{slot}; "
                f"{len(samples)} frames shown below at {video_fps:g} fps with timestamps."
            )
            for i, pil in samples:
                parts.append({"type": "text", "text": f"<Video {vid_n}> t={i / media.H3_FPS:.1f}s:"})
                parts.append({"type": "image_url", "image_url": {"url": media.pil_to_data_url(pil, video_side)}})

        if pic_parts:
            # pictures last: after dozens of video frames a single picture shown first is easily overlooked
            if parts:
                parts.append({"type": "text", "text": "REFERENCE PICTURES (separate still images, NOT frames of the "
                                                      "video above - look at each one closely):"})
            parts.extend(pic_parts)

        for slot in range(1, MAX_AUDIOS + 1):
            aud = kw.get(f"audio_{slot}")
            if aud is None:
                continue
            aud_n += 1
            aud_desc.append(self._audio_line(f"<Audio {aud_n}>", aud))

        return parts, pic_desc, vid_desc, aud_desc, first_image

    # ------------------------------------------------------------------ main
    def generate(self, instruction, task, frame_anchor, duration_seconds, thinking, length,
                 allow_invented_dialogue, max_tokens, seed,
                 model=local_models.SERVER_DEFAULT, mmproj=local_models.MMPROJ_AUTO, **kw):
        print_tokens = kw.get("print_to_console", True)
        model_path, mmproj_path = local_models.resolve(model, mmproj, _CFG)
        has_media = any(kw.get(k) is not None for k in kw
                        if (k.startswith("image_") or k.startswith("video_")) and k[6:].isdigit()) \
            or kw.get("first_frame") is not None
        if model_path:
            if mmproj_path is None and mmproj != local_models.MMPROJ_NONE:
                found = list(local_models.mmproj_choices(_CFG))[2:]
                if has_media:
                    raise RuntimeError(
                        f"[H3 Prompter] mmproj 'auto' tidak menemukan file vision untuk {model}, jadi LLM tidak bisa "
                        "MELIHAT gambar/video yang tersambung (prompt-nya akan dikarang). Pilih mmproj secara manual. "
                        + ("File mmproj yang ada: " + ", ".join(found) if found else
                           "Tidak ada file mmproj*.gguf di models/LLM - download mmproj model ini (mis. mmproj-F16.gguf "
                           "dari repo Unsloth yang sama) ke folder modelnya.")
                        + " Kalau memang ingin tanpa vision, pilih 'none (text only)'.")
                lc.log(f"no matching mmproj found for {model}: running text-only.")
            elif mmproj_path and mmproj == local_models.MMPROJ_AUTO:
                lc.log(f"mmproj auto -> {os.path.basename(mmproj_path)}")
            server_url = managed_server.ensure(model_path, mmproj_path, kw.get("context_size", 32768), _CFG)
            model_alias = os.path.splitext(os.path.basename(model_path))[0]
            vision_ok = mmproj_path is not None
        else:
            server_url = (kw.get("server_url") or _CFG["server_url"]).strip()
            lc.ensure_server(server_url, _CFG)
            model_alias = _CFG.get("model_alias", "qwen3.8-27b")
            vision_ok = True  # unknown: assume the hand-started server has its mmproj
        if not vision_ok:
            for k in list(kw):
                if (k.startswith("image_") or k.startswith("video_")) and k[6:].isdigit():
                    if kw[k] is not None:
                        lc.log("WARNING: text-only model (no mmproj) - the LLM cannot SEE the pictures/videos, so it "
                               "will describe them generically. Pick the model's mmproj for good edit prompts.")
                        kw["_text_only"] = True
                        break

        vframes = [media.video_len(kw[f"video_{i}"]) for i in range(1, MAX_VIDEOS + 1)
                   if kw.get(f"video_{i}") is not None]
        vframes = [n for n in vframes if n > 0]
        fc_in = int(kw.get("frame_count", 0) or 0)
        if fc_in > 0:
            n = max(5, min(fc_in, H3_MAX_TRAINED_FRAMES))
            down = 5 + 17 * ((n - 5) // 17)
            up = min(down + 17, H3_MAX_TRAINED_FRAMES)
            frames = up if (up - n) <= (n - down) else down
            if frames != fc_in:
                lc.log(f"frame_count {fc_in} is not on the 17k+5 grid -> using {frames}.")
            eff = frames / media.H3_FPS
        elif duration_seconds <= 0:
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

        if kw.get("describe_refs", True) and kw.get("_pic_urls") and not kw.get("_text_only"):
            captions = self._describe_pictures(server_url, model_alias, kw["_pic_urls"], instruction,
                                               int(seed), float(_CFG.get("request_timeout_seconds", 600)))
            for n, cap in captions.items():
                for i, d in enumerate(pics):
                    if d.startswith(f"<Picture {n}>"):
                        pics[i] = d + f" VERIFIED CONTENT (checked on the picture alone): {cap}"

        keyframe = None
        kf_pic = int(kw.get("keyframe_picture", 0) or 0)
        kf_image, kf_idx = None, 0
        if kf_pic > 0:
            tensors = kw.get("_pic_tensors", {})
            if kf_pic not in tensors:
                lc.log(f"keyframe_picture = {kf_pic} but only {len(tensors)} image(s) are connected -> keyframe ignored.")
            elif kf_pic == 1 and frame_anchor.startswith("reference 1 = first frame"):
                lc.log("keyframe_picture = 1 is already the first frame (frame_anchor) -> keyframe ignored.")
            else:
                kf_idx = max(0, min(frames - 1, int(round(float(kw.get("keyframe_seconds", 2.5)) * media.H3_FPS))))
                kf_image = tensors[kf_pic]
                keyframe = (f"<Picture {kf_pic}>", kf_idx / media.H3_FPS, kf_idx)

        user_text = prompts.build_user_text(
            instruction=instruction, task=task, keyframe=keyframe,
            shots=str(kw.get("shots", "auto")), timed_beats=bool(kw.get("timed_beats", False)),
            frame_anchor=("none" if kw.get("_ff_label") else frame_anchor), first_frame_label=kw.get("_ff_label"),
            duration_s=eff, frames=frames,
            pictures=pics, videos=vids, audios=auds, length=length,
            allow_invented_dialogue=allow_invented_dialogue,
            asset_notes=kw.get("asset_notes", ""), extra_rules=kw.get("extra_rules", ""),
        )
        simple = str(kw.get("prompt_style", "full")).startswith("simple")
        prefill = FIRST_FIELD + "\n"
        system_text = prompts.SYSTEM_PROMPT_R2V
        opener = None
        if simple:
            user_text = prompts.build_simple_text(
                instruction=instruction, task=task, pictures=pics, videos=vids, audios=auds,
                asset_notes=kw.get("asset_notes", ""), extra_rules=kw.get("extra_rules", ""),
                first_frame_label=kw.get("_ff_label"))
            system_text = prompts.SYSTEM_PROMPT_SIMPLE
            parts = _drop_video_parts(parts)  # the change is described from the request/pictures only -> faster
            opener = prompts.simple_opener(task, bool(vids), any("synchronized audio track" in a for a in auds))
            prefill = opener
            max_tokens = min(int(max_tokens), 400)
        if kw.get("_text_only"):
            parts = []  # a text-only server rejects image parts
        n_img = sum(1 for p in parts if p.get("type") == "image_url")
        n_vid = sum(1 for p in parts if p.get("type") == "text" and p.get("text", "").startswith("<Video"))
        lc.log(f"LLM input: {n_img - n_vid} picture(s) + {n_vid} video frame(s)"
               + (" (video frames not sent in simple style)" if simple and vids else ""))
        user_content = [{"type": "text", "text": user_text}, *parts] if parts else user_text

        think_on = thinking != "off"
        # fixed sampling (seed gives variations): Qwen's recommendation for Qwen models, neutral otherwise
        if "qwen" in model_alias.lower() or not model_path:
            samp = dict(SAMPLING["on" if think_on else "off"])
        else:
            samp = dict(temperature=0.8 if think_on else 0.7, top_p=0.95, top_k=40, min_p=0.05, presence_penalty=0.0)

        base = {
            "model": model_alias,
            "max_tokens": int(max_tokens),
            "seed": int(seed),
            "cache_prompt": True,
            **samp,
        }
        system = {"role": "system", "content": system_text}
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
            server_url, base, system, user, thinking, timeout, print_tokens, on_token, prefill=prefill)

        if simple:
            prompt = prompts.clean_simple(content)
            if opener:
                body = prompt[len(opener.strip()):].strip() if prompt.startswith(opener.strip()) else prompt
                body = body.rstrip()
                if body and body[-1] not in ".!?":
                    body += "."
                prompt = opener + body + " " + prompts.simple_closer(kw.get("edit_mode", ""))
        else:
            prompt = prompts.clean_output(
                content, keep_timed_beats=bool(kw.get("timed_beats", False)) and task != "video editing")
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
        if kf_image is None:
            kf_image = _blank_image()
        return (prompt, int(frames), float(eff), first_image, reasoning, kf_image, int(kf_idx))

    _caption_cache: dict = {}

    def _describe_pictures(self, server_url, model_alias, pic_urls, instruction, seed, timeout) -> dict:
        """One short, picture-only request per image -> {pic_n: description}. Nothing else is in the context, so
        the description cannot be mixed up with the video frames."""
        import hashlib

        out: dict = {}
        for n, url, kind in pic_urls:
            key = (server_url, model_alias, hashlib.md5(url.encode("ascii")).hexdigest(), kind, instruction.strip())
            cap = self._caption_cache.get(key)
            if cap is None:
                user = {"role": "user", "content": [
                    {"type": "text", "text": prompts.build_caption_text(instruction, n, kind)},
                    {"type": "image_url", "image_url": {"url": url}},
                ]}
                base = {"model": model_alias, "max_tokens": 220, "seed": seed, "cache_prompt": True,
                        **CAPTION_SAMPLING}
                pre = f"<Picture {n}>: "
                t0 = time.time()
                try:
                    content, _, _ = self._run(server_url, base, {"role": "system", "content": prompts.SYSTEM_PROMPT_CAPTION},
                                              user, "off", timeout, False, None, prefill=pre)
                except Exception as exc:  # noqa: BLE001  (the main prompt still works without it)
                    lc.log(f"describe_refs: <Picture {n}> could not be described ({exc}); continuing without it.")
                    continue
                cap = prompts.clean_caption(content, n)
                if not cap:
                    continue
                if len(self._caption_cache) > 64:
                    self._caption_cache.clear()
                self._caption_cache[key] = cap
                lc.log(f"<Picture {n}> seen as ({time.time() - t0:.1f}s): {cap}")
            else:
                lc.log(f"<Picture {n}> seen as (cached): {cap}")
            out[n] = cap
        return out

    @staticmethod
    def _run(server_url, base, system, user, thinking, timeout, print_tokens, on_token, prefill=FIRST_FIELD + "\n"):
        """prefill = visible start of the answer (prefilled so the model writes the answer at once); None = none."""
        def _fix(content, used):
            vis = (used or "").replace("<think>\n\n</think>\n\n", "")
            if vis and not content.lstrip().startswith(vis.strip()):
                content = vis + content.lstrip()
            return content

        if thinking == "off":
            think_block = "<think>\n\n</think>\n\n"
            attempts = []
            if prefill:
                # 1) template switch + prefill -> the model starts writing the answer at once
                attempts.append(({"enable_thinking": False}, prefill))
            # 2) no prefill (server refused it) - the reasoning guard is still active
            attempts.append(({"enable_thinking": False}, None))
            # 3) template ignored the switch -> hard prefill of an empty think block
            attempts.append(({"enable_thinking": False}, think_block + (prefill or "")))
            # 4) last resort: let it run, strip any reasoning afterwards
            attempts.append((None, None))
            last_err = None
            for kwargs, pre in attempts:
                msgs = [system, user]
                if pre:
                    msgs.append({"role": "assistant", "content": pre})
                payload = dict(base, messages=msgs)
                if kwargs:
                    payload["chat_template_kwargs"] = kwargs
                    payload["reasoning_effort"] = "none"  # newer llama-server: OpenAI-style switch, also disables thinking
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
                content = _fix(content, pre)
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
            msgs = [system, user] + ([{"role": "assistant", "content": prefill}] if prefill else [])
            payload = dict(base, **SAMPLING["off"], messages=msgs, chat_template_kwargs={"enable_thinking": False})
            content, _, timings = lc.stream_chat(server_url, payload, timeout=timeout, print_tokens=print_tokens)
            content = _fix(content, prefill)
        return content, reasoning, timings


def _drop_video_parts(parts: list[dict]) -> list[dict]:
    """Remove '<Video N> t=..' label + frame pairs from the multimodal parts (keep pictures)."""
    out, skip_next = [], False
    for p in parts:
        if skip_next:
            skip_next = False
            continue
        if p.get("type") == "text" and p.get("text", "").startswith("<Video"):
            skip_next = True
            continue
        if p.get("type") == "text" and p.get("text", "").startswith("REFERENCE PICTURES"):
            continue  # header only makes sense after video frames
        out.append(p)
    return out


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


class _AnyType(str):
    def __ne__(self, other):  # accepts any link type
        return False


class MiniMaxH3UnloadLLM:
    """Stop the llama-server started by the prompter node (frees its VRAM)."""

    CATEGORY = "MiniMax H3/Prompt"
    FUNCTION = "unload"
    OUTPUT_NODE = True
    RETURN_TYPES = (_AnyType("*"),)
    RETURN_NAMES = ("passthrough",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unload": ("BOOLEAN", {"default": True})},
                "optional": {"trigger": (_AnyType("*"), {"tooltip": "Sambungkan output apa saja supaya unload terjadi setelah node itu selesai."})}}

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")  # always run

    def unload(self, unload, trigger=None):
        if unload:
            stopped = managed_server.stop(_CFG)
            lc.log("managed llama-server stopped, VRAM freed." if stopped else "no managed llama-server was running.")
        return (trigger,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3R2VPrompter": MiniMaxH3R2VPrompter,
    "MiniMaxH3Frames": MiniMaxH3Frames,
    "MiniMaxH3UnloadLLM": MiniMaxH3UnloadLLM,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3R2VPrompter": "MiniMax H3 R2V Prompter (llama.cpp)",
    "MiniMaxH3Frames": "MiniMax H3 Duration → Frames",
    "MiniMaxH3UnloadLLM": "MiniMax H3 Unload LLM (free VRAM)",
}
