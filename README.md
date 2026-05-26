<div align="center">

# 🎵 MOD2GLSL v1.666

$${\Large\color{orange}\textsf{GLSL MOD Player v1.666 for ShaderToy and standalone HTML embedded page}}$$
$${\small\color{lightgray}\textsf{© 2026 Orblivius — All rights reserved}}$$

![Version](https://img.shields.io/badge/version-1.37-orange?style=flat-square)
![Python](https://img.shields.io/badge/python-3.x-blue?style=flat-square&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/platform-ShaderToy-7e57c2?style=flat-square)
![License](https://img.shields.io/badge/license-non--commercial-green?style=flat-square)
![Format](https://img.shields.io/badge/MOD-ProTracker-ff69b4?style=flat-square)

</div>
<div align="center">

### 🎬 Live Demo

https://github.com/user-attachments/assets/4c7fe125-086c-410b-bcba-808fc7648984

*BEYOND_MUSIC.MOD rendered through GLSL MOD Player v1.666*

</div>

Convert Amiga ProTracker mod files (MOD, S3M, XM, and IT) into a ShaderToy shader, complete with visualization and a note/effect overlay.

Examples: <br>
&nbsp; % python3 mod_player.py beyond.mod --viz 6   --resampler lanczos3<br>
&nbsp; % python3 mod_player.py Firestorm.it --viz 3 --no-rvq2  --bitrate ultra<br>
    
---

## $${\color{limegreen}\textsf{✨ What's new in v1.666}}$$

$\color{limegreen}\textsf{+}$ &nbsp; **Impulse Tracker (IT) w/ NNA support (reference precision)**  <br>
$\color{limegreen}\textsf{+}$ &nbsp; **Fast Tracker II (XM) support is working, but not solid!**  <br>
$\color{limegreen}\textsf{+}$ &nbsp; **Scream Tracker 3.xx (S3M) support is solid!**<br>
$\color{limegreen}\textsf{+}$ &nbsp; **Amiga ProTracker (MOD) support is solid**<br>
$\color{limegreen}\textsf{+}$ &nbsp; **Rewritten from scratch with individual format loaders to avoid a big mess**<br>
$\color{limegreen}\textsf{+}$ &nbsp; **Avanced micro-click removal**  <br>
$\color{limegreen}\textsf{+}$ &nbsp; **3D Surround, Velvet Reverb, Phat Bass, Soft Limiter, FAT4X**<br>
$\color{limegreen}\textsf{+}$ &nbsp; **PNG loader is wokring, and tested! (--png) loads from JSON too!**<br>
$\color{limegreen}\textsf{+}$ &nbsp; **RVQ sample compression** — 27.7 dB<br>
$\color{limegreen}\textsf{+}$ &nbsp; **Mouse control to horizontal scroll tracks**  <br>
$\color{limegreen}\textsf{+}$ &nbsp; **Downsampling integrated with RVQ (--downsample 1, 2 or 4)**<br>
$\color{limegreen}\textsf{+}$ &nbsp; **Firefox ShaderToy Unofficial plugin now imports .json tab structure**  <br>
$\color{limegreen}\textsf{+}$ &nbsp; **Linear, B-Spline, and Lanczos3 resamplers**<br>
$\color{limegreen}\textsf{+}$ &nbsp; **Up to 32 tracks supported**  <br>
$\color{limegreen}\textsf{+}$ &nbsp; **Data packing optimizations**  <br>
$\color{limegreen}\textsf{+}$ &nbsp; **Loader optimizations**  <br>
$\color{limegreen}\textsf{+}$ &nbsp; **8 Visualizers backdrop for ShaderToy builds** <br>
$\color{limegreen}\textsf{+}$ &nbsp; Added `--help`   <br>
$\color{limegreen}\textsf{+}$ &nbsp; Added `--use-png` — roll MOD into a PNG for a much faster load<br>
$\color{limegreen}\textsf{+}$ &nbsp; Added `--viz 0` (no backdrop) plus visualizers `1`–`8`  <br>
$\color{limegreen}\textsf{+}$ &nbsp; Added `--split` / `--no-split` — `--split` for faster compile  <br>
$\color{limegreen}\textsf{+}$ &nbsp; Added `--vec-dim 8` — better results than `--downsample 2`  <br>
$\color{limegreen}\textsf{+}$ &nbsp; Added `--no-rvq2` — faster compile, ~4 dB SNR cost<br>
$\color{limegreen}\textsf{+}$ &nbsp; **Fixed most of the pattern playback bugs in S3M and MOD**  <br>
---

## $${\color{deepskyblue}\textsf{🎬 Live demos}}$$
<p align="center">
  <img width="60%" alt="MOD2GLSL screenshot" src="https://github.com/user-attachments/assets/b10632a5-c7a6-47e7-a28e-832251b19e6c" />
</p>
<p align="center">
<samp>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>BEYOND.MOD</b> • <a href="https://www.shadertoy.com/view/s3l3R8">shadertoy.com/view/s3l3R8</a></samp><br>
<samp>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>GUITAROUS.MOD</b> • <a href="https://www.shadertoy.com/view/N3s3WN">shadertoy.com/view/N3s3WN</a></samp><br>
<samp>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>SPCDEBRIS.MOD</b> • <a href="https://www.shadertoy.com/view/fXf3D4">shadertoy.com/view/fXf3D4</a></samp><br>
<samp>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>GADGET.IT</b> • <a href="https://www.shadertoy.com/view/s3sGWM">shadertoy.com/view/s3sGWM</a></samp><br>
<samp>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>MV-FLUTES.XM</b> • <a href="https://www.shadertoy.com/view/7Xs3WM">shadertoy.com/view/7Xs3WM</a></samp><br>
<samp>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>HYBRID.XM</b> • <a href="https://www.shadertoy.com/view/7Xs3WM">shadertoy.com/view/7Xs3WM</a></samp><br>
<samp>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>FIRESTORM.IT</b> • <a href="https://www.shadertoy.com/view/N3lGzj">shadertoy.com/view/N3lGzj</a></samp><br>
</p>

---

> [!WARNING]
> **Don't release large MOD shaders publicly on ShaderToy.com.**
> I recently got banned for posting all my fancy MODs — proud that I'd squeezed them into the 64K stack — but the admin didn't take it lightly. If you want to use this:
> - Clone your shader as **Private** or **Unlisted**, **or**
> - Build your own with small MOD files.
>
> Large MODs take forever to load in embedded mode and crash Linux 3D graphics drivers. Users complain → you get banned. You've been warned.

> [!IMPORTANT]
> **Help us get custom textures on ShaderToy** 🙏
>
> There *is* a way to embed a MOD into a PNG and load it instantly — but ShaderToy.com doesn't allow custom-texture uploads. I've proposed a simple fix: let established contributors upload textures behind an automated content check.
>
> If you'd like to support the proposal, please [📧 email the ShaderToy admin](mailto:info@shadertoy.com?subject=PNG%20texture%20upload%20feature%20request&body=Hi%2C%0A%0AI%27d%20like%20to%20support%20the%20proposal%20to%20allow%20custom%20texture%20uploads.) and copy-paste the paragraph above. The more voices, the better.

<details>
<summary>🔧 <b>Why does it crash, technically?</b></summary>

The 64K stack-space limit. As shader local-variable usage approaches that ceiling, some Linux 3D drivers spill internal variables into that space when they shouldn't, and the driver crashes. The WebGL/ANGLE standard requires the 64K local-variable stack to be respected — apparently Linux 3D driver developers don't know this, or ignore it.

</details>
---
### Command-line reference

```text
MOD2GLSL % python mod_player.py --help
usage: mod_player.py [-h] [--downsample DOWNSAMPLE] [--bitrate {lo,med,hi,ultra}] [--vec-dim {2,4,8}]
                     [--resampler {linear,bspline,lanczos3}] [--aa] [--no-json] [--viz [0..19]]
                     [--samples] [--samples-dir SAMPLES_DIR] [--solo CH] [--positions A-B]
                     [--intro-silence SEC] [--mp3] [--mp3-secs SEC] [--xfade SAMPLES] [--start POS]
                     [--no-rvq2] [--preserve PRESERVE] [--raw-perc] [--no-raw-perc]
                     [--raw-perc-budget BYTES] [--png] [--reverb-size {full,small}]
                     [--surround | --no-surround] [--phatbass | --no-phatbass]
                     [--phatbass-mode {auto,sample,mix}] [--fat4x | --no-fat4x] [--no-dsp]
                     [--fft-n {64,128,256,512,1024,2048}] [--max-compat]
                     modfile

MOD/S3M/IT → HTML player + ShaderToy GLSL (+ optional PNG samples).

positional arguments:
  modfile               MOD/S3M/IT file to play

options:
  -h, --help            show this help message and exit
  --downsample DOWNSAMPLE
                        Sample decimation factor: 1=full-rate, 2=22kHz, 4=11kHz. (DEFAULT 2.) HF percussion (cymbals/rides) gets max(1,DS//2) to keep shimmer. (default: 2)
  --bitrate {lo,med,hi,ultra}
                        RVQ codebook size (mp3-style quality knob). lo=K(128,64) 13b/pair smallest+grainy, med=K(256,128) 15b/pair balanced, hi=K(512,256) 17b/pair sharper, ultra=K(1024,512) 19b/pair near-transparent. (default: hi)
  --vec-dim {2,4,8}     RVQ vector dimensionality. 8=smallest (~2.1 bits/sample), 4=medium (4.25 bits/sample), 2=highest fidelity (8.5 bits/sample). (default: 8)
  --resampler {linear,bspline,lanczos3}
                        Sample resampler. linear=2-tap (cheapest, ProTracker-style), bspline=4-tap cubic (smooth, slightly softer HF), lanczos3=6-tap sinc (sharpest/brightest — DEFAULT). Use --resampler bspline for a softer sound or to save GPU headroom. Default: lanczos3. (default: None)
  --aa                  Enable the gated ratio anti-aliasing (stateless box integrator around getSampleF: averages K sub-taps across the per-output-sample step, K tracks the resample ratio; notes not pitched up are bit-identical / zero cost). Suppresses alias whine on high/pitched-up notes — most audible on full-rate --raw-perc drums. NOTE: this is a deliberate "cleaner than the oracle" divergence — real Impulse Tracker / MikIT use plain 2-tap linear with NO anti-aliasing, so --aa is NOT more 1:1, just nicer-sounding. Off by default; emits #define AA_RESAMPLE 1 when set. (default: False)
  --no-json             Skip generating the {base}_shadertoy.json import file (JSON is generated by default). (default: True)
  --viz [0..19]         Image-tab visualizer (choose from [0..19]):
                          0 = Sun Rays          (cabbibo-style — warm/cool god-ray corona)
                          1 = Reactive 001     (PAEz fork — SDF circles + cosmic web)  ← default
                          2 = Fluxline Surfer  (mrange — DR2 dodecahedron + glowtracer)
                          3 = Zuvuya           (city/stars + audio-reactive curtain)
                          4 = Maya             (raymarched fractal tunnel-warp)
                          5 = Dodecahedron     (Philip Bertani — DR2 IFS fractal raymarcher)
                          6 = Disco Combined   (smoke spotlights + lasers/clouds, time-driven)
                          7 = Sparkly 4D       (Philip Bertani — 4D IFS volumetric raymarcher)
                          8 = Skywalker        (orblivius — flying-curve terrain + sync stars)
                          9 = Music in the DNA (jaszunio15/enbe fork — DNA helix + parallax dunes)
                         10 = LED Band Spectro (Orblivious — pentagon LED tunnel + spectrum history)
                         11 = Telekenesis v1.1 (Orblivious — IFS fractal raymarcher + waveform strip)
                         12 = Laser Patterns   (0rblivius — Newton/IFS laser-grid fractal marcher)
                         13 = Prismatic Frac.  (Smull fork — IFS box raymarch + audio glow)
                         14 = Interdiml. Fold  (PAEz fork — spatial fold + sine warp march)
                         15 = Fractal Torus    (ytt fork — periodic torus march + Menger IFS fold)
                         16 = Unfound          (diatribes fork — orbit-trap fold + sinusoidal orb march)
                         17 = Evrthing Temp.   (diatribes/FabriceNeyret2 — noise terrain + orb march)
                         18 = sm0g             (diatribes/Shane — tri-planar SDF box corridors + bump)
                         19 = Spectro 3D Mist  (Orblivius — 3D SDF bar spectrum + volumetric fog + reflective ground) (default: 6)
  --samples             Extract each sample (instrument) from the module as a separate WAV file (named like 1-samplename.wav, 2-anothername.wav). Skips GLSL/HTML generation. Saves to current directory unless --samples-dir is also given. Useful for diagnosing per-sample playback issues (which sample is wrong/buzzy/missing). (default: False)
  --samples-dir SAMPLES_DIR
                        Output directory for --samples WAV files (default: current dir). (default: None)
  --solo CH             Solo a single channel: mute every other channel by clearing their cells before encoding. CH is 1-based (so --solo 1 keeps channel 1, mutes 2..N). Useful for diagnosing per-channel issues (which channel has the wrong sample, missing notes, wrong panning, etc.). Pipeline runs normally — sample selection, effects, panning, FX all apply — only the soloed channel produces output. (default: None)
  --positions A-B       Render only order positions A..B (0-based, as shown on the Image tab), e.g. --positions 35-36 or --positions 35. The build plays JUST that slice, but the running speed/tempo are first computed by walking positions 0..A-1 break-aware, so the slice plays at the SAME speed it would inside the full song — never the file-header default. Use for isolating/auditioning a section (e.g. the inst-25 porta lead) without hand-trimming a .S3M. (default: None)
  --intro-silence SEC   Seconds of silence before the song starts (Sound tab holds, then plays from row 0; the Image/BufferA visualizer is offset to stay in sync). Default 0.0 = music starts immediately. Use e.g. --intro-silence 10 to let an Image-tab loading splash render before the audio kicks in. (default: 0.0)
  --mp3                 After generating the GLSL, render an .mp3 of the actual ShaderToy Sound tab on CPU (glslang -> spirv-cross -> clang -> WAV -> mp3 via sound_exec.py). This is the SAME audio ShaderToy plays — handy for quick listening/sharing without opening the site. Needs the glslang+spirv-cross+clang toolchain and ffmpeg/lame; if missing, prints how to install it instead of failing. (default: False)
  --mp3-secs SEC        Duration to render for --mp3 (default 180 = the full ShaderToy cap). CPU render is ~real-time-ish, so lower this (e.g. 30) for a quick preview. (default: 180.0)
  --xfade SAMPLES       Retrigger/note-on declick crossfade length in samples (default 64 = 1.45ms). On a same-channel sample restart the OLD voice ramps down and the NEW ramps up over this window. If you still hear clicks on busy leads, raise it (e.g. 256 = 5.8ms, 512 = 11.6ms) and re-check by ear (--mp3). Costs no extra GPU private-vars, just a longer blend region. (default: 64)
  --start POS           Force a manual 2-way split at order position POS instead of the automatic split: emit exactly TWO bundles — SONG_shadertoy_part1_* = positions 0..POS-1, and SONG_shadertoy_part2_* = positions POS..end. Lets you pick the split point at a musical boundary. Carries speed/tempo over so each part plays at its true in-song speed. POS is an order position (the Image-tab numbering), 1..(numPositions-1). (default: None)
  --no-rvq2             Skip RVQ stage 2 (residual quantization).  Drops ~40% of sample-data const arrays from Sound tab → faster compile. Quality cost: ~4 dB SNR (sounds noisier but pitch is unchanged). IMPORTANT: when re-pasting into ShaderToy, paste BOTH the new Common AND new Sound — otherwise mismatched RVQ_BITS produces high-pitch garbage from a stale Common reading 15-bit-packed codes that were actually written at 8 bits. (default: False)
  --preserve PRESERVE   Comma-separated 1-based instrument numbers stored UNCOMPRESSED (raw int8, no VQ quantization) for perfect quality — e.g. --preserve 28,25 keeps the lead/voice samples pristine while the rest stay VQ-compressed small. getSample() intercepts those instruments' index ranges and reads the raw array instead of VQ-decoding (resampled to the same rate as the VQ stream). (default: )
  --raw-perc            Auto-store percussion (kick/snare/hat/clap/cymbal — samples the waveform classifier tags NOISE) UNCOMPRESSED, exactly like --preserve but auto-detected. Percussion transients/noise are the worst-hit by RVQ, so this keeps drums crisp and matching the HTML player. Percussion samples are short → small size cost. Default ON. (default: True)
  --no-raw-perc         Disable --raw-perc (let percussion be VQ-compressed too). (default: True)
  --raw-perc-budget BYTES
                        Max total raw (un-VQ) percussion bytes kept by --raw-perc (shortest-first; the rest fall back to VQ). Default 28672. The raw PCM becomes a big const array (_presvPCM) in the Sound tab — on a tight GPU that const-register/private-var load can push the shader over ANGLE's limit (e.g. jeff.it: full raw-perc = +42KB Sound = does not fit). LOWER this to keep only the smallest kick/hat pristine and still fit (e.g. 8192 ≈ one short drum); 0 ≈ effectively --no-raw-perc. Quality-vs-fit dial when full raw-perc overflows. (default: 28672)
  --png                 Use the PNG-loaded data path (samples/patterns read via texelFetch from iChannel0 = a 1024×1024 RGBA PNG = 4 MB) instead of VQ-encoded const arrays, AND write SONG_player_data.png. DEFAULT OFF: normally NO PNG is written and the build is embedded. Because one PNG holds the WHOLE song, a --png build is a SINGLE bundle (one set of .glsl + one .png) — it is NOT auto-split into parts and the song is NOT trimmed to 180s. Smaller Common source = faster compile, but raw 8-bit samples (no RVQ) so quality differs. ShaderToy setup: Image/Common iChannel0 = SONG_player_data.png via the Unofficial Plugin "Custom Textures". (default: False)
  --reverb-size {full,small}
                        Reverb dimensions. full = 4 combs × 3 iters (default), small = 2 combs × 2 iters (--max-compat default). Reduces compile cost and stereo width. (default: None)
  --surround, --no-surround
                        3D surround widening on outer LRRL pair. Default: ON (or OFF if --max-compat without override). (default: None)
  --phatbass, --no-phatbass
                        PhatBass Hilbert allpass enhancement on bass instruments. Default: ON (or OFF if --max-compat without override). (default: None)
  --phatbass-mode {auto,sample,mix}
                        PhatBass routing. 'sample' (default) forces per-sample via isBass[] flags — cleanest, leaves leads/pads alone. 'auto' uses per-sample when bass instruments were detected, else mix-wide. 'mix' forces mix-wide (applies Hilbert cross-pan to the entire mixdown — wider stereo + bass enhancement on everything, can smear mid/high transients slightly). (default: sample)
  --fat4x, --no-fat4x   FAT4X harmonic exciter on master output. Default: ON (kept ON even under --max-compat — it's cheap). (default: None)
  --no-dsp              MASTER SWITCH: disable ALL DSP effect processing in the output shaders (3D surround, FAT4X exciter, PhatBass; velvet/comb reverb are already off in v1.63). Forces ENABLE_3D/FAT/PHATBASS/VELVETREVERB/COMBREVERB = 0 and WINS over any individual --surround/--phatbass/--fat4x passed alongside it. This is the lightest Sound-tab path (no DSP private-vars) → best chance of fitting ANGLE's per-GPU private-variable ceiling. Note: AA is a resampler option, NOT part of the DSP chain — control it with --aa (default off). (default: False)
  --fft-n {64,128,256,512,1024,2048}
                        FFT size for Buffer A spectrum. Larger = more frequency resolution but slower compile. Default: 1024 (or 128 if --max-compat without override). (default: None)
  --max-compat          [NO-OP — max-compat is now the DEFAULT in v1.40+ (current: v1.63)] This flag previously enabled compatibility mode for problematic GPUs/drivers (Windows + Firefox + NVIDIA, etc.). The compat preset (--resampler lanczos3, --reverb-size small, --no-surround, --phatbass, --fft-n 512, FAT4X on, extra HLSL pragmas) is now applied by default since most consumer setups need it and the quality difference is small. To opt OUT of any compat setting, pass the inverse individual flag — e.g. --reverb-size full, --surround. The flag is kept for backward compatibility with old command lines but does nothing. (default: False)

Examples:
  # Standard ShaderToy build — embedded, no DSP, fits most GPUs:
  python3 mod_player.py SONG.S3M --no-dsp --downsample 2

  # Full-rate, highest quality (may exceed a tight GPU's limit):
  python3 mod_player.py SONG.S3M --no-dsp --bitrate hi

  # Audition only order positions 35-36 (speed carried over from earlier):
  python3 mod_player.py SONG.S3M --positions 35-36 --no-dsp

Input formats: .mod  .s3m  .it   (.xm not yet implemented)
Outputs:       SONG_player.html, SONG_shadertoy_{common,sound,bufferA,image}.glsl,
               SONG_shadertoy.json (one-click ShaderToy import)
```

---

## $${\color{plum}\textsf{👋 About the author}}$$

> [!TIP]
> ☕ Like the project? **Donate via PayPal** to <subband@protonmail.com>

I've been an audio guy my whole life. At 6 years old I was picked from the crowd and told *"you — must do music,"* so from age 6 to 16 I learned piano and music theory. I write small tunes — nothing too amazing, but at least I can call them my own. You can find them on my SoundCloud:

🎧 **https://soundcloud.com/analogintelligence**

---

## $${\color{cyan}\textsf{📖 Overview}}$$

`mod_player.py` converts MOD files of size up to roughly **150–200 KB** into a ShaderToy shader.

### Features

<img src="https://github.githubassets.com/images/icons/emoji/unicode/1f538.png" width="20" /> **2-stage Residual VQ** — K-means sample compression, ~14.7 dB SNR at ~2.1 bits/sample.  
<img src="https://github.githubassets.com/images/icons/emoji/unicode/1f538.png" width="20" /> **Per-sample FFT bandwidth analysis** — auto-decimates low-bandwidth samples, preserves HF shimmer.  
<img src="https://github.githubassets.com/images/icons/emoji/unicode/1f538.png" width="20" /> **Loop-seam smoothing** — patches post-loop guard so VQ doesn't break loop boundaries.  
<img src="https://github.githubassets.com/images/icons/emoji/unicode/1f538.png" width="20" /> **Configurable resampler** — linear, B-spline, or Lanczos-3.  
<img src="https://github.githubassets.com/images/icons/emoji/unicode/1f538.png" width="20" /> **3D Surround, PhatBass, Comb Reverb, FAT** — Hilbert bass enhance + channel-pair widening.  
<img src="https://github.githubassets.com/images/icons/emoji/unicode/1f538.png" width="20" /> **RLE pattern compression** — bitmap + dictionary + O(1) row seek.  
<img src="https://github.githubassets.com/images/icons/emoji/unicode/1f538.png" width="20" /> **`ivec4` chunked data loader** — 4 bytes per int32, beats GLSL array limits.  
<img src="https://github.githubassets.com/images/icons/emoji/unicode/1f538.png" width="20" /> **6 built-in visualizers** — `--viz 0..5`.  
<img src="https://github.githubassets.com/images/icons/emoji/unicode/1f538.png" width="20" /> **Tracker-like GUI** — pattern grid, oscilloscope/FFT toggle, BPM/Speed/Position.  
<img src="https://github.githubassets.com/images/icons/emoji/unicode/1f538.png" width="20" /> **Full output bundle** — HTML player + 4 ShaderToy tabs + PNG + paste instructions.  
<img src="https://github.githubassets.com/images/icons/emoji/unicode/1f538.png" width="20" /> **`--use-png`** — fast-compile alternative, raw 8-bit samples via texelFetch.  
<img src="https://github.githubassets.com/images/icons/emoji/unicode/1f538.png" width="20" /> **`--no-rvq2`** — single-stage RVQ, ~33% smaller Sound, ~4 dB SNR cost.

### Requirements

```bash
pip install numpy pillow
```

---

## $${\color{gold}\textsf{🚀 Usage}}$$

### Basic conversion

```bash
python mod_player.py yourfavorite.mod
```

> [!NOTE]
> Pick MODs under **150 KB** — otherwise you'll need external storage in a PNG. I'll write up an explanation of how to play back any-size MOD inside ShaderToy using a special Firefox plugin I wrote that lets you drop a custom texture and resets the `iTime` clock so the engine picks it up.

### With downsampling

```bash
python mod_player.py yourfavorite.mod --downsample 2
```

> [!NOTE]
> Valid values are `2`, `4`, and `8`. Beyond that, sample degradation is unbearable.

### Generated files

For an input named `ars.mod`, the script generates:

| File | Purpose |
|---|---|
| 🌐 `ars_player.html` | HTML page that loads the GLSL outside of ShaderToy |
| 🖼️ `ars_player_data.png` | Pattern data texture (RGBA) |
| 🟦 `ars_shadertoy_common.glsl` | Common tab |
| 🟧 `ars_shadertoy_image.glsl` | Image tab |
| 🟨 `ars_shadertoy_bufferA.glsl` | Buffer A tab |
| 🟩 `ars_shadertoy_sound.glsl` | Sound tab |
| 📄 `ars_shadertoy_instructions.txt` | Setup instructions |

### Setup checklist

1. Insert an alphabet texture into `iChannel0` on the Image tab.
2. Add Buffer A to `iChannel1` on the Image tab.
3. Feed Buffer A back onto itself by setting `iChannel0` of the Buffer A tab to itself.

---

## $${\color{mediumseagreen}\textsf{⚙️ How it works}}$$

### 1. Sample packing

All samples are converted to floating-point and packed into an RGBA texture:

- Each pixel stores 4 sample values (R, G, B, A)
- 8-bit samples are mapped to `[0, 255]`
- Samples are stored sequentially with an index table

### 2. Sample index table

The GLSL code contains a sample table with **6 values per sample**:

| Field | Description |
|---|---|
| `start` | Start position in texture |
| `length` | Length in samples |
| `loop_start` | Loop start position |
| `loop_length` | Loop length |
| `volume` | Volume (0–64) |
| `finetune` | Finetune value |

### 3. Pattern data

Pattern information is encoded as:

| Bits | Field |
|---|---|
| 5 | Sample number |
| 12 | Period (determines pitch) |
| 4 | Effect type |
| 8 | Effect parameter |

For large MODs, patterns are packed into a second texture.

### 4. GLSL playback

The shader:

- Reads sample data from the texture via UV coordinates
- Handles sample looping
- Converts ProTracker periods to playback rates
- Mixes 4 channels
- Has a framework for processing effects

---

## $${\color{tomato}\textsf{🔬 Technical details}}$$

### MOD format

ProTracker MOD files contain:

- 31 sample slots (8-bit signed PCM)
- 4-channel pattern data
- Up to 128 pattern positions
- 64 rows per pattern
- Effects: arpeggio, portamento, vibrato, etc.

### Sample-rate conversion

Amiga uses period-based timing:

$$\text{Playback Rate} = \frac{7\,093\,789.2}{\text{period} \times 2}$$

Middle C (period 428) ≈ **8287 Hz**.

### Texture format

**Sample texture**

- Width: 1024 px (configurable)
- Height: auto-calculated from total samples
- Format: `RGBA8`
- Data: `[-1.0, 1.0]` mapped to `[0, 255]`

**Pattern texture**

- Width: 1024 px
- Format: `RGBA8`
- Each note: 4 bytes — `(sample, period_hi, period_lo|effect, param)`

### Limitations

> [!CAUTION]
> - ShaderToy has a shader size limit (~64 KB), but you gain headroom from RLE pattern compression and `ivec4` packing for **8× byte storage** — effectively ~64 KB × 8 or more
> - Large MODs may need pattern data in a texture rather than inline
> - Effects need manual implementation
> - No support for extended MOD formats (8-channel, etc.)

---

## $${\color{orchid}\textsf{💻 Example: playing your MOD}}$$

```glsl
void mainSound(out vec2 sound, int sampleIndex, float time) {
    // Calculate song position
    float samplesPerRow = SAMPLE_RATE * 2.5 / BPM * SPEED;
    int row = int(float(sampleIndex) / samplesPerRow);

    // Get pattern and row
    int patternIdx = row / 64;
    int patternRow = row % 64;

    // Read notes from pattern data
    // Mix 4 channels
    // Apply effects
    // Output audio
}
```

---

## $${\color{coral}\textsf{🎛️ MOD effects reference}}$$

| Code  | Effect |
|-------|---|
| `0xy` | Arpeggio |
| `1xx` | Portamento up |
| `2xx` | Portamento down |
| `3xx` | Tone portamento |
| `4xy` | Vibrato |
| `5xy` | Tone portamento + volume slide |
| `6xy` | Vibrato + volume slide |
| `9xx` | Set sample offset |
| `Axy` | Volume slide |
| `Bxx` | Position jump |
| `Cxx` | Set volume |
| `Dxx` | Pattern break |
| `Fxx` | Set speed/tempo |

---

## $${\color{dodgerblue}\textsf{🛠️ Advanced usage}}$$

### Custom texture size

```python
texture, info, total = pack_samples_to_texture(mod, texture_width=2048)
```

### Sample rate

```python
texture, info, total = pack_samples_to_texture(mod, target_rate=44100)
```

### Testing

Try with the classic *12th Warrior* MOD:

```bash
python mod_to_shadertoy_complete.py 12th_warrior.mod
```

---

## $${\color{salmon}\textsf{🚨 Troubleshooting}}$$

| Problem | Solution |
|---|---|
| 🟥 Texture too large | Increase texture width, or split samples |
| 🟧 Shader won't compile | Pattern data may be too big — use texture encoding |
| 🟨 Audio sounds wrong | Check sample-rate conversion and period calculations |
| 🟦 No sound output | Verify `iChannel0` is set to the sample texture |

---

## $${\color{mediumpurple}\textsf{🗺️ Roadmap}}$$

- [ ] Full effect implementation
- [ ] Multi-pattern texture optimization
- [ ] Support for 8-channel MODs
- [ ] XM/IT format support
- [ ] Real-time pattern editor
- [ ] Visualization support

---

## $${\color{lightgray}\textsf{📜 Credits \\& License}}$$

Created for converting Amiga ProTracker MOD files to ShaderToy format. Based on the ProTracker specification and the ShaderToy audio API.

**License:** Free for non-commercial use. Contact the author for any other use.

### Contact

| | |
|---|---|
| 👤 **Orblivius** | <subband@gmail.com> |
| 💸 **Donate (PayPal)** | <subband@protonmail.com> |

### Links

- 🎨 [ShaderToy](https://www.shadertoy.com/)
- 📦 [The Mod Archive](https://modarchive.org/)
- 📘 [ProTracker Spec](http://16-bits.org/mod/)

<div align="center">

---

$${\small\color{gray}\textsf{Made with ♥ and questionable amounts of caffeine}}$$

</div>
