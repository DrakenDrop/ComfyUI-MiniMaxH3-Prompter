# ComfyUI-MiniMaxH3-Prompter

Node ComfyUI untuk menulis prompt **MiniMax H3 Full-Reference (R2V)** secara cepat, memakai **Qwen3.8-27B GGUF dari Unsloth** lewat **llama.cpp (`llama-server`)**. Model tetap tinggal di VRAM, jadi setiap run langsung jalan tanpa loading ulang.

Format output mengikuti [official Full-Reference prompt guide MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md):
`subject_definitions` → `summary` → `retention_analysis` → `detailed_description` → `overall_soundscape` → `non_diegetic_music`.

## Fitur

- **Deteksi model otomatis**: dropdown berisi semua GGUF di `ComfyUI/models/LLM`, lengkap dengan mmproj yang cocok. llama-server dijalankan otomatis, dan model tetap di VRAM sampai kamu memilih model lain.
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

## 2. Pilih model (otomatis dari `ComfyUI/models/LLM`)

Node memindai **`ComfyUI/models/LLM`** (termasuk subfolder, misalnya `LLM/GGUF/...` dari ThinkingLLM) dan menampilkan semua file `.gguf` di dropdown **model**. Tidak perlu download ulang.

- **model**: pilih GGUF-nya. Node menjalankan `llama-server` sendiri di port 8090 (`-ngl 999`, semua layer di GPU), dan model **tetap di VRAM** antar-run maupun saat ComfyUI di-restart. Server baru di-restart kalau kamu memilih model, mmproj, atau `context_size` yang lain.
- **mmproj** (vision): `auto` memilih mmproj di folder yang sama. Yang diutamakan adalah nama yang cocok dengan model (mis. `mmproj-Qwen3.8-27B-ABLITERATED-F16`), atau nama generik seperti `mmproj-F16.gguf` kalau foldernya hanya berisi satu model. Kalau ada keraguan, node memakai mode text-only dan menulis peringatan di console. Pilih mmproj secara manual dari dropdown dalam kasus itu. mmproj yang salah pasangan membuat llama-server crash.
- Syarat satu-satunya: node harus tahu lokasi `llama-server.exe`. Taruh llama.cpp di `C:\llama.cpp`, atau salin `config.example.json` menjadi `config.json` lalu isi `llama_server_path`. Model tambahan di luar `models/LLM` bisa ditambahkan lewat `extra_model_dirs`.
- File model yang baru ditambahkan akan muncul setelah browser di-refresh.
- Node **MiniMax H3 Unload LLM** mematikan server itu untuk membebaskan VRAM. Sambungkan `trigger` ke output prompter kalau kamu mau VRAM langsung dilepas setelah prompt jadi.

### Alternatif: jalankan server sendiri

Pilih model `(llama-server yang sudah jalan)` dan jalankan `start_llama_server.bat`. Edit `LLAMA_DIR`, lalu double-click. Model default di-download otomatis dari Hugging Face (`unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL` + mmproj). Bisa juga diganti ke file lokal dengan `-m ... --mmproj ...`. Server ini memakai port 8080 dan `server_url`.

> VRAM 96 GB: Q4_K_XL + KV cache butuh sekitar 20 GB, jadi masih banyak sisa untuk H3. Untuk kualitas lebih tinggi, pakai `UD-Q6_K_XL` (25 GB) atau `Q8_0` (29 GB). Jangan menjalankan dua server sekaligus (bat + dropdown), karena modelnya akan terpakai dua kali di VRAM.

## 3. Instal node

Salin folder `ComfyUI-MiniMaxH3-Prompter` ke `ComfyUI/custom_nodes/`, lalu restart ComfyUI. Atau:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DrakenDrop/ComfyUI-MiniMaxH3-Prompter.git
pip install -r ComfyUI-MiniMaxH3-Prompter/requirements.txt
```

Dependency-nya hanya `numpy`, `Pillow` dan `psutil`, yang semuanya sudah terpasang bersama ComfyUI. llama-server tidak diinstal lewat pip (lihat langkah 1).
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

## Dipakai bersama MiniMax H3 V2V Edit (Minimax-H3-V2V)

Prompter ini bisa menggantikan pembuat prompt bawaan [Minimax-H3-V2V](https://github.com/DrakenDrop/Minimax-H3-V2V). Labelnya sudah sama: `ref_image_*` → `<Picture 1..n>`, `first_frame` → `<Picture>` terakhir, video sumber → `<Video 1>`.

```
Load Video → MiniMax H3 V2V Conform Video ──images──┬──> V2V Edit.source_video (source_fps 24, start 0)
                                                    ├──> Prompter.video_1
                                                    └──> pose / depth / canny preprocessors → control_*
Foto referensi ─────────────────────────────────────┬──> V2V Edit.ref_image_0
                                                    └──> Prompter.image_1
Frame 0 yang sudah diedit (opsional) ───────────────┬──> V2V Edit.first_frame
                                                    └──> Prompter.first_frame
Prompter.prompt ──────────────────────────────────────> V2V Edit.prompt_override
```

- Prompter: `task = video editing`, `duration_seconds = 0`, dan pilih mmproj (wajib supaya LLM melihat videonya).
- V2V Edit: `use_source_as_reference = true`, `start_seconds = 0` (pemotongan dilakukan di Conform Video), dan `source_fps = 24`.
- Karena kedua node menerima frame yang sama, durasi dan timestamp di prompt sama persis dengan yang dipakai H3.

## Contoh pemakaian

**Edit video:** `task = video editing`, `duration_seconds = 0` (ikut panjang video, dipotong ke 17k+5 seperti node H3), `video_1` = frame video, `video_1_audio` = audionya. Suara asli otomatis dipakai ulang (fully_copy) kecuali instruksi bilang lain.
Audio hasil edit video diambil dari video input, jadi prompt-nya otomatis dibuat hemat di bagian suara: soundscape cukup satu kalimat, musik N/A, dan tidak ada dialog baru. Kalau orangnya berbicara, sambungkan juga `video_1_audio` (dan `ref_video_audio_1` di node H3) supaya gerak bibir mengikuti audio aslinya.
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

## Instruction vs system prompt

- **instruction** = user prompt: apa yang kamu mau, dalam bahasa apa saja. Peran audio dan musik juga ditulis di sini, misalnya "pakai suara di Audio 1 persis, lip-sync", "suaranya seperti Audio 1", "tanpa musik", atau "musik piano pelan".
- **System prompt** sudah tertanam di node (aturan resmi MiniMax H3) dan tidak perlu kamu tulis.
- Kalau instruksi tidak menyebut audio: soundtrack video yang diedit dipakai ulang persis, dan audio mandiri dipakai sebagai referensi timbre suara. Kalau tidak menyebut musik: musik hanya ditambahkan kalau cocok dengan adegannya.
- Sampling diatur otomatis: rekomendasi Qwen untuk model Qwen, dan setelan netral untuk model lain. Ganti `seed` untuk mendapat variasi.
