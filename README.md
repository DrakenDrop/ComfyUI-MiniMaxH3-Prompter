# ComfyUI-MiniMaxH3-Prompter

Node ComfyUI untuk menulis prompt **MiniMax H3 Full-Reference (R2V)** secara cepat, memakai **Qwen3.8-27B GGUF dari Unsloth** lewat **llama.cpp (`llama-server`)**. Model tetap tinggal di VRAM, jadi setiap run langsung jalan tanpa loading ulang.

Format output mengikuti [official Full-Reference prompt guide MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md):
`subject_definitions` → `summary` → `retention_analysis` → `detailed_description` → `overall_soundscape` → `non_diegetic_music`.

## Fitur

- **Thinking benar-benar OFF** (default). Tiga lapis pengaman:
  1. `chat_template_kwargs: {"enable_thinking": false}` di setiap request;
  2. *prefill*: jawaban model langsung dimulai dari `subject_definitions:`, jadi model tidak sempat mulai berpikir;
  3. penjaga: kalau token reasoning tetap muncul, stream langsung diputus dan diulang dengan blok `<think></think>` kosong. Output `reasoning` selalu kosong kalau thinking = off.
- **Multi input image**: `image_1` … `image_9` terpisah (tanpa batch), sama seperti `ref_image_1..9` di node H3.
- **Set reference 1 as first frame**: `frame_anchor = reference 1 = first frame` (juga ada opsi last frame, atau first + last).
- **Edit video**: `task = video editing` + `video_1` (+ `video_1_audio` kalau suara asli mau dipakai lagi).
- **Edit image**: `task = image edit`. image_1 diubah sesuai instruksi lalu dianimasikan.
- Video continuation, reference generation, audio sebagai timbre suara atau dipakai ulang persis (lip-sync).
- Durasi otomatis dibulatkan ke grid H3 **17k+5 frame @24fps**; output `length` bisa langsung disambung ke node H3.
- Streaming: tombol **Cancel** di ComfyUI langsung menghentikan generasi; progress bar dan kecepatan (tok/s) tampil di console.
- System prompt statis → llama-server meng-cache prefix-nya, jadi run kedua dan seterusnya lebih cepat.
- Tanpa dependency pip tambahan (hanya `numpy` + `Pillow`, yang sudah ada di ComfyUI).

## 1. Instal llama.cpp (llama-server)

**Windows:** download build **CUDA untuk Windows** terbaru dari <https://github.com/ggml-org/llama.cpp/releases>. Cari file yang namanya mengandung `win` dan `cuda`; kalau ada zip `cudart` yang cocok, download juga. Untuk GPU Blackwell (RTX 50xx / RTX PRO 6000), pilih versi CUDA paling baru. Extract semuanya ke satu folder, misalnya `C:\llama.cpp`.

**Linux:** pakai build release, atau compile dengan `-DGGML_CUDA=ON`.

## 2. Jalankan server (model menetap di VRAM)

Edit `LLAMA_DIR` di `start_llama_server.bat`, lalu double-click. Saat pertama kali dijalankan, model otomatis di-download dari Hugging Face:

- `Qwen3.8-27B-UD-Q4_K_XL.gguf` (~17.6 GB)
- `mmproj` (vision, ~0.9 GB), dibutuhkan supaya LLM bisa melihat gambar dan video.

Parameter yang dipakai: `-ngl 999` (semua layer di GPU), `--no-mmap` (dimuat penuh), `-c 32768`, `-np 1`. Selama jendela server terbuka, model tetap ada di VRAM, termasuk saat ComfyUI di-restart.

> VRAM 96 GB: Q4_K_XL + KV cache butuh sekitar 20 GB, jadi masih banyak sisa untuk H3. Kalau mau kualitas lebih tinggi, ganti ke `UD-Q6_K_XL` (25 GB) atau `Q8_0` (29 GB). Kecepatan turun sedikit karena bobot yang dibaca per token lebih besar.

**Opsional, autostart:** salin `config.example.json` menjadi `config.json`, isi `llama_server_path`, lalu set `"autostart": true`. Server akan dinyalakan otomatis saat ComfyUI boot. Kalau server sudah jalan, server yang ada dipakai ulang.

## 3. Instal node

Salin folder `ComfyUI-MiniMaxH3-Prompter` ke `ComfyUI/custom_nodes/`, lalu restart ComfyUI.
Node ada di **MiniMax H3/Prompt → MiniMax H3 R2V Prompter (llama.cpp)**.

## 4. Menyambung ke node H3 (penting: urutannya harus sama)

Penomoran label sama persis dengan node core **MiniMax H3 Reference to Video**:

| Prompter | Node H3 Reference to Video | Label di prompt |
|---|---|---|
| `image_1`, `image_2`, … (yang tersambung saja, berurutan) | `ref_image_1`, `ref_image_2`, … | `<Picture 1>`, `<Picture 2>`, … |
| `video_N` (frame IMAGE @24fps) | `ref_video_N` | `<Video N>` |
| `video_N_audio` | `ref_video_audio_N` | `<Audio j>`, dinomori **sebelum** audio mandiri |
| `audio_N` | `ref_audio_N` | `<Audio j>` berikutnya |

Output node:

- `prompt` → `prompt` di H3 Reference to Video
- `length` → `length` di H3 Reference to Video (dan Empty Latent kalau dipakai)
- `first_frame` → untuk **frame_anchor first frame**: sambungkan ke **Add Guide for MiniMax H3** dengan `frame_idx = 0` (positive + latent dari node H3). Gambar yang sama tetap juga disambung ke `ref_image_1`.
  Untuk *last frame*, sambungkan gambar terakhir ke Add Guide kedua dengan `frame_idx = -1`.
- `reasoning` → teks reasoning (hanya terisi kalau thinking ≠ off)

Pakai node **Preview Any** untuk melihat prompt yang dihasilkan.

## Contoh pemakaian

**Edit video:** `task = video editing`, `duration_seconds = 0` (ikut panjang video, dipotong ke 17k+5 seperti node H3), `video_1` = frame video, `video_1_audio` = audionya dengan `exact reuse / lip-sync (fully_copy)`.
Instruksi: `ganti jaket pria jadi kulit hitam, latar jadi malam hujan`.

**Reference 1 sebagai frame pertama:** `frame_anchor = reference 1 = first frame`, `image_1` = frame awal, `image_2` = wajah karakter, `image_3` = baju.
`asset_notes`: `Picture 2 = wajah wanita, Picture 3 = gaun merah yang harus dipakai`.

**Edit gambar:** `task = image edit`, `image_1` = foto. Instruksi: `ubah rambutnya jadi pirang, dia tersenyum lalu melambai`.

## Tips kecepatan

- `thinking = off` (default) adalah mode tercepat.
- `length = compact` menghasilkan output kira-kira setengah dari standard, jadi sekitar 2× lebih cepat.
- `video_sample_fps` (default 2, sama dengan cara Qwen di dalam H3 melihat video) dan `video_max_side` (default 512) mengatur berapa banyak frame video yang dilihat LLM. Klip 15 detik di 2 fps = 31 frame. Pakai 1 fps kalau adegannya tenang. Hanya bagian video yang benar-benar dipakai H3 yang dikirim ke LLM.
- `image_max_side` 512–768 mengurangi token vision (tidak mempengaruhi kualitas video H3).
- Batas H3: **24 fps**, maksimal **362 frame (~15 detik)**. Video yang lebih panjang dipotong dari awal. Load video dengan `force_rate = 24`.
- Seed yang sama dengan input yang sama tidak akan dijalankan ulang (cache ComfyUI).

## Sampling

`qwen recommended` = rekomendasi Qwen3.8 untuk mode non-thinking (temp 0.7, top_p 0.8, top_k 20, presence_penalty 1.5), atau mode thinking (1.0 / 0.95 / 20 / 0). Pilih `custom` kalau mau mengatur sendiri.
