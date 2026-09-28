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

## MiniMax H3 V2V Edit + LLM (satu node untuk edit video)

Node **MiniMax H3 V2V Edit + LLM (Fun ControlNet)** menggabungkan V2V Edit ([Minimax-H3-V2V](https://github.com/DrakenDrop/Minimax-H3-V2V)) dengan prompter ini, jadi prompt-nya ditulis oleh LLM lokal yang melihat frame videonya sendiri.

**Preset `edit_mode`** (masing-masing punya aturan prompt dan kekuatan pose/depth/edge sendiri):

| preset | untuk | pose / depth / edge (release) |
|---|---|---|
| `video_edit` (default) | prompt sama persis dengan prompter biasa (`task = video editing`), tanpa aturan preset tambahan | 0.90 / 0.30 / 0 (0.50) |
| `change_outfit` | ganti baju (outfit swap) | 0.85 / 0.30 / 0 (0.40) |
| `replace_person` | ganti orang | 1.00 / 0 / 0 (0.40) |
| `add_object` | tambah objek | 0.80 / 0.20 / 0 (0.30) |
| `add_subject` | tambah orang/hewan/karakter | 0.80 / 0 / 0 (0.30) |
| `remove_object` | hapus objek (baru) | 0.80 / 0.20 / 0 (0.30) |
| `change_background` | ganti latar (baru) | 0.95 / 0 / 0 (0.30) |
| `restyle` | ubah gaya visual | 0.60 / 0.50 / 0.30 (0.70) |
| `custom` | edit apa saja lewat instruction | 0.90 / 0.30 / 0 (0.50) |

- **instruction** boleh kosong kalau ada gambar referensi. Tiap preset punya instruksi default, misalnya "put the outfit from the reference on the person".
- Frame yang dilihat LLM sama persis dengan yang dipakai H3 (sudah 24 fps, dipotong, dan di-resize di dalam node), jadi timestamp prompt selalu cocok.
- Prompt di-cache: kalau hanya `motion_lock`, strength, atau setting sampler yang diubah, LLM tidak dijalankan ulang.
- `prompt_override` diisi → LLM dilewati.
- `unload_llm_after_prompt` → llama-server dimatikan setelah prompt jadi, supaya VRAM-nya bebas untuk sampling.
- Resolusi: aspect ratio mengikuti video sumber (sisi pendek 768, maks 768×1344, kelipatan 32).
- Preset `remove_object` dan `change_background` masih baru: kekuatan ControlNet-nya belum teruji, jadi atur `motion_lock` / strength manual kalau hasilnya kurang pas.

### Mask dengan prompt teks (SAM 3.1)

Dengan mask, **hanya area yang di-mask yang dibuat ulang**. Piksel lain, termasuk wajah kalau tidak ikut di-mask, disalin langsung dari video asli lewat inpainting Fun ControlNet-Union.

1. Download `sam3.1_multiplex_fp16.safetensors` dari [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) ke `models/checkpoints/`.
2. `CheckpointLoaderSimple` (sam3.1) → `MODEL` ke **sam3_model**, `CLIP` ke **sam3_clip**.
3. Isi **mask_prompt** dalam bahasa Inggris, boleh beberapa dipisah koma, misalnya `shirt, pants`, `person`, `red car`, `dog`.

| preset | mask default kalau `mask_prompt` kosong |
|---|---|
| `change_outfit` | `clothes` (baju saja; wajah & rambut tetap asli) |
| `replace_person` | `person` |
| `change_background` | `person` **dibalik** (semua kecuali orang dibuat ulang) |
| preset lain | tanpa mask, kecuali `mask_prompt` diisi |

- **`mask_strength`** (default 1.0) dan **`mask_patch`** (`separate` / `combined with first control`): inpainting Fun ControlNet mengisi area mask dengan **hitam** di sinyal kontrolnya (begitu model ini dilatih). Kalau kekuatannya kurang, warna hitam itu tembus ke hasil (baju jadi hitam atau ada outline gelap). Karena itu inpaint sekarang punya patch sendiri dengan kekuatannya sendiri; naikkan ke 1.2–1.3 kalau masih ada tepi gelap. `mask_grow` default diturunkan ke 6 px.
- **`mask_mode`**: `inpaint` (default) = hanya area mask yang digambar ulang. `reference only` = area mask (misalnya baju lama) hanya **disembunyikan** dari `<Video 1>`, lalu seluruh frame digambar ulang mengikuti **pose saja** (depth/edge otomatis 0). Baju baru bebas bentuk dan panjangnya, dan H3 tidak bisa menyalin baju lama.
- **`hide_masked_in_reference`** (default on): area mask di referensi `<Video 1>` dibuat abu-abu, supaya H3 tidak menyalin isi lama (misalnya dress hitam) kembali ke area yang diedit.
- **`use_mask`** (on/off): matikan untuk mengabaikan semua mask (input `mask`, SAM3, dan default preset) tanpa melepas kabel. Seluruh frame dibuat ulang, dan gerakan tetap mengikuti pose/depth/edge.
- **Beberapa area sekaligus**: pisahkan dengan koma, misalnya `hat, shoes`. Semua area digabung jadi satu mask, dan tiap area diedit sesuai instruction ("topinya jadi merah, sepatunya jadi sneakers putih"). Batas jumlah objek yang dilacak diatur lewat `mask_max_objects` (default 8; sepasang sepatu = 2 objek).
- `mask_invert`: membuat ulang semua **kecuali** objek yang di-prompt.
- `mask_grow` (default 12 px): memperbesar mask, supaya ada ruang untuk lengan yang lebih panjang, rambut, atau bayangan. Naikkan kalau baju baru lebih besar dari baju lama.
- `mask_threshold`: turunkan kalau SAM3 tidak menemukan objeknya.
- `mask`: bisa juga memakai mask sendiri (MASK) sebagai pengganti SAM3.
- Output **mask** bisa di-preview (Convert Mask to Image → Preview) untuk mengecek area yang akan diedit.
- Mask di-cache: mengubah `motion_lock` atau setting sampler tidak menjalankan SAM3 lagi.
- Butuh `model_patch` (Fun ControlNet-Union), karena inpainting-nya berjalan lewat model itu.

**Sambungan:**

```
Load Video → Get Video Components ──images──┬──> V2V Edit + LLM.source_video   (fps → source_fps)
                                            └──> MiniMax H3 Conform Video (24 fps) → pose/depth/canny → control_*
UNETLoader (ref2va) → LoRA turbo ────────────> V2V Edit + LLM.model
CLIPLoader (minimax) / VAELoader ────────────> clip / vae
ModelPatchLoader (Fun ControlNet-Union) ─────> model_patch
Foto referensi ──────────────────────────────> ref_image_1 (… ref_image_8)

V2V Edit + LLM.model ───┬──> BasicGuider.model ──┐
                        └──> BasicScheduler.model │
V2V Edit + LLM.positive ───> BasicGuider.conditioning
RandomNoise + KSamplerSelect (res_multistep) + BasicScheduler (simple, 4 step turbo / 20 tanpa LoRA)
V2V Edit + LLM.latent ─────> SamplerCustomAdvanced → VAEDecode (H3 video VAE)
                            → CreateVideo (24 fps, audio = Conform.audio) → SaveVideo
```

Kalau kamu tetap memakai node V2V Edit yang lama, sambungkan `prompt` dari prompter ke `prompt_override`. Labelnya sama (`ref_image` → `<Picture 1..n>`, `first_frame` → `<Picture>` terakhir, sumber → `<Video 1>`). Lewatkan videonya dulu ke Conform Video, supaya kedua node menerima frame yang sama.

## Storyboard otomatis dari prompt sederhana (shots + timed_beats)

Cukup tulis prompt singkat, misalnya `kucing melompat ke meja lalu tidur`. Prompter yang menyusun storyboard-nya:

- **`shots`** = `auto` (default): LLM menentukan sendiri jumlah shot (±1 shot per 2,5–5 detik), framing dan gerak kamera tiap shot, serta **waktu setiap potongan** (`[Shot 2] At 00:03.200, …`) supaya ceritanya pas dengan durasi.
- `shots` = `1` … `6`: jumlah shot dikunci, tapi LLM tetap memilih kapan potongannya. `1` berarti satu shot panjang tanpa potongan.
- **`timed_beats`** = on: di dalam tiap shot, aksi utamanya juga diberi detik (mis. "At 1.5 s it jumps; at 3.0 s it lands"). Ini eksperimental karena bukan format resmi H3; coba bandingkan hasilnya.
- Untuk **video editing**, kedua opsi ini diabaikan, karena shot dan gerakan mengikuti video asli.

## Keyframe di tengah video (keyframe_picture + keyframe_seconds)

Selain frame pertama/terakhir (`frame_anchor`), gambar mana pun bisa dikunci di detik tertentu:

- `keyframe_picture` = nomor `<Picture N>` (0 = off), `keyframe_seconds` = detiknya (dibulatkan ke frame @24fps).
- LLM menulis `<Picture N> is the keyframe of [Shot N] at the S.SS-second mark …` dan "the shot's keyframe corresponds to `<Picture N>`", lalu menambahkan tag `[keyframe completion]`.
- Output `keyframe_image` dan `keyframe_frame_idx` → **Add Guide for MiniMax H3** (`image` dan `frame_idx`), dirangkai setelah Add Guide untuk first frame kalau ada.

```
H3 Reference to Video.positive/latent → Add Guide (first_frame, frame_idx 0) → Add Guide (keyframe_image, keyframe_frame_idx) → BasicGuider
```

Gambar yang sama tetap disambung ke `ref_image_N` di node H3 supaya labelnya ada.

## Prompt simpel (prompt_style = simple)

Di prompter dan V2V Edit + LLM ada pilihan **`prompt_style`**:

- **`full (official H3)`** (default): format resmi 6 bagian.
- **`simple`**: hanya 1–3 kalimat yang menjelaskan **perubahannya saja**, tanpa bagian lain dan tanpa `[Shot]`. Contoh untuk video editing dengan instruction `change her outfit to black dress`:

  > [video editing] The target video is an edited version of `<Video 1>`: she now wears a knee-length black satin slip dress with thin straps. Everything else - the person's identity, face, hair, body, motion, timing, camera, framing, background and lighting - stays exactly as in `<Video 1>`.

  Awal dan akhir kalimatnya tetap (ditulis oleh node), jadi LLM hanya menulis bagian tengah. Frame video tidak dikirim ke LLM, sehingga prosesnya hanya beberapa detik.

## Lighting berubah setelah edit? (Match Color to Source)

Tanpa mask, H3 menggambar ulang seluruh frame, jadi exposure, white balance, atau pencahayaan bisa sedikit bergeser. Ada dua perbaikan:

1. **Prompt**: untuk video editing, LLM tidak lagi mendeskripsikan ulang lighting (kata seperti "warm key light" atau "cinematic" membuat H3 menata ulang cahaya). Cukup ditulis "same lighting, exposure and color grade as `<Video 1>`".
2. **Node MiniMax H3 Match Color to Source** (setelah VAE Decode):
   ```
   VAEDecode ─> images
   V2V Edit + LLM.source_frames ─> source_frames
   V2V Edit + LLM.mask ─> mask (opsional)
   → CreateVideo
   ```
   Warna dan kecerahan tiap frame dicocokkan lagi ke video asli (Lab, dihaluskan antar-frame supaya tidak flicker). Statistiknya hanya diukur di luar mask, jadi elemen yang diedit (misalnya dress merah) tetap merah.

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
