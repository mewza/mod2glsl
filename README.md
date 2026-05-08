<div align="center">

# 🎵 MOD2GLSL v1.50

$${\Large\color{orange}\textsf{GLSL MOD Player v1.50 for ShaderToy}}$$
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

*BEYOND_MUSIC.MOD rendered through GLSL MOD Player v1.50*

</div>

Convert Amiga ProTracker MOD files (and soon S3M, XM, and IT) into a ShaderToy shader, complete with visualization and a note/effect overlay.
Example: % python mod_player.py beyond.mod --max-compat --viz 6   --resampler lanczos3<br>

---

## $${\color{limegreen}\textsf{✨ What's new in v1.50}}$$

$\color{limegreen}\textsf{+}$ &nbsp; **Impulse Tracker (IT) w/ NNA support**  <br>
$\color{limegreen}\textsf{+}$ &nbsp; **Fast Tracker II (XM) support**  <br>
$\color{limegreen}\textsf{+}$ &nbsp; **ScreamTracker 3.xx (S3M) support**  <br>
$\color{limegreen}\textsf{+}$ &nbsp; **TRU 3D Surround** (new improved AllPass 3D Surround technique)<br>
$\color{limegreen}\textsf{+}$ &nbsp; **Phat Bass — Hilbert applied to bass tracks or a mix**<br>
$\color{limegreen}\textsf{+}$ &nbsp; **Velvet Reverb** <br>
$\color{limegreen}\textsf{+}$ &nbsp; **FAT 4X** (Fat curve compressor)<br>
$\color{limegreen}\textsf{+}$ &nbsp; **W1 (Low latency) Limiter** <br>
$\color{limegreen}\textsf{+}$ &nbsp; **RVQ sample compression** — 27.7 dB<br>
$\color{limegreen}\textsf{+}$ &nbsp; **Mouse control to horizontal scroll between tracks**  <br>
$\color{limegreen}\textsf{+}$ &nbsp; **Downsampling integrated with RVQ (--downsample 1, 2 or 4)**<br>
$\color{limegreen}\textsf{+}$ &nbsp; **Linear, B-Spline, and Lanczos3 resamplers**<br>
$\color{limegreen}\textsf{+}$ &nbsp; **Up to 32 tracks supported**  <br>
$\color{limegreen}\textsf{+}$ &nbsp; **Data packing optimizations**  <br>
$\color{limegreen}\textsf{+}$ &nbsp; **Loader optimizations**  <br>
$\color{limegreen}\textsf{+}$ &nbsp; **NEW visualizers** <br>
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
$ python mod_player.py --help
usage: mod_player.py [-h] [--downsample DOWNSAMPLE] [--bitrate {lo,med,hi,ultra}]
                     [--vec-dim {2,4,8}] [--resampler {linear,bspline,lanczos3}]
                     [--no-split] [--split] [--viz {0,1,2,3,4,5}] [--no-rvq2]
                     [--use-png]
                     modfile

MOD/S3M Player — Generates HTML player + ShaderToy GLSL + PNG samples

positional arguments:
  modfile               MOD or S3M file to play

options:
  -h, --help            show this help message and exit

  --downsample DOWNSAMPLE
                        Sample decimation factor: 1=full-rate, 2=22kHz, 4=11kHz.
                        HF percussion (cymbals/rides) gets max(1, DS//2) to keep
                        shimmer. (default: 1)

  --bitrate {lo,med,hi,ultra}
                        RVQ codebook size (mp3-style quality knob).
                          lo    = K(128, 64)   13b/pair  smallest + grainy
                          med   = K(256, 128)  15b/pair  balanced
                          hi    = K(512, 256)  17b/pair  sharper
                          ultra = K(1024, 512) 19b/pair  near-transparent
                        (default: med)

  --vec-dim {2,4,8}     RVQ vector dimensionality.
                          8 = smallest         (~2.1 bits/sample)
                          4 = medium           (4.25 bits/sample)
                          2 = highest fidelity (8.5 bits/sample)
                        (default: 8)

  --resampler {linear,bspline,lanczos3}
                        Sample resampler.
                          linear   = 2-tap (cheapest, ProTracker-style)
                          bspline  = 4-tap cubic (smooth/soft)
                          lanczos3 = 6-tap sinc (sharpest, ~50% more cost)
                        (default: lanczos3)

  --no-split            Keep VQ arrays + decoders in the Common tab. Required
                        for oscilloscope/spectrum/Buffer A visualizers to
                        decode actual audio via getChannelOutput. Default ON.
                        (default: True)

  --split               Split VQ arrays into the Sound tab — fast Common
                        compile, but breaks audio-driven visualizers
                        (no getChannelOutput in Image/BufferA). (default: True)

  --viz {0,1,2,3,4,5}   Image-tab visualizer:
                          0 = None             (black backdrop, fastest compile)
                          1 = Reactive 001     (PAEz fork — SDF circles + cosmic web)  ← default
                          2 = Fluxline Surfer  (mrange — DR2 dodecahedron + glowtracer)
                          3 = Zuvuya           (city/stars + audio-reactive curtain)
                          4 = Maya             (raymarched fractal tunnel-warp)
                          5 = Dodecahedron     (Philip Bertani — DR2 IFS fractal raymarcher)
                        (default: 1)

  --no-rvq2             Skip RVQ stage 2 (residual quantization). Drops ~40% of
                        sample-data const arrays from Sound tab → faster compile.
                        Quality cost: ~4 dB SNR (noisier but pitch is unchanged).
                        IMPORTANT: when re-pasting into ShaderToy, paste BOTH
                        the new Common AND the new Sound — otherwise mismatched
                        RVQ_BITS produces high-pitch garbage from a stale Common
                        reading 15-bit-packed codes that were actually written
                        at 8 bits. (default: False)

  --use-png             Use legacy PNG-loaded Common (samples read via
                        texelFetch from iChannel0=PNG) instead of VQ-encoded
                        const arrays. Smaller Common source = faster compile,
                        but raw 8-bit samples (no RVQ) so quality differs.
                        ShaderToy setup: Image/Common iChannel0 =
                        GSLINGER_player_data.png via Unofficial Plugin
                        "Custom Textures". Implies --no-split. (default: False)
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
