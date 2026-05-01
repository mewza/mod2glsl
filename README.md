# MOD2GLSL v1.37<br><br>$${\color{orange}GLSL\ MOD\ Player\ v1.37\ for\ ShaderToy}$$<br>$${\color{litegray}©2026\ Orblivius.\ All\ rights\ reserved.}$$
<br>
Convert Amiga ProTracker MOD(and soon S3M/XM and IT) files 
to ShaderToy shader with visualization and note/effect overlay.<br>
<br>
WARNING: Do not release your MOD shaders on ShaderToy.com if the MOD 
is large and takes a long time to load - I recently got banned because
I was posting all my fancy MODs all proud that I made them load into 64k 
stack, but the admin didn't take it lightly. If you want to use this,
either clone it as Private or Unlisted, otherwise just build your own
and do not use large MOD files. The reason is because large MOD files 
take forever to load in embedded mode and crash Linux 3D Gfx drivers, 
so the users will complain and you will get banned. You've been warned!<br>
<br>
There is however an option to embed MOD file into PNG and load it from there
very quickly, however, the ShaderToy.com site does not let you upload
custom textures. I did propose to them a simple solution, for established
contributors to the site offer that possibility and still run it through a
simple automated approval process that checks image for nafarious content.
Well, if you feel like writing an email to support this proposal please
email to: <info@shadertoy.com> and just copy paste the paragraph. That would
be the only possibility if enough people ask them for this feature, then,
embedding MOD into PNG and uploading it to the site with the local reference
would make things a LOT simplier, make loading instant and not crash
the poorly written Linux 3d gfx drivers, it is happening because of
64K stack space limit, when it reaches near that level, some drivers
store some internal variables in that space which they shouldn't and
it causes gfx driver to crash. By the standdard imposed by WebGL and
ANGLE the stack of 64k of allocation of local variables and arrays
must be respected, but apparently Linux 3D gfx driver developers don't
know this, or ignore it.<br>
<br>
<h4><code>NEW in v1.37</code></h4>
       • CombFilter Reverb<br>
       • RVQ advanced sample compression (27.7 dB)<br>
       • Downsample feature now fuse with RVQ ensures smooth samples <br>(--downsample 1 (default), 2. 4, 8)<br>
       • 3D Surround<br>
       • Linear, B-Spline, Lanczcos3 resamplers<br>
       • PHAT Bass (Hilbert, improved bass track detection)<br>
       • FAT<br>
       • Added --help<br>
       • Added --viz 0 (no backrop viz, and 1 to 5 different ones)<br>
       • Added options for --split and --no-split (--split for faster loading time)<br>
       • Added --vec-dim 8 (instead of doing --downsample 2 you can just go --vec-dim 8 for better results)<br>
       • Added --no-rvq2 mode (feature to reduce load time but shaves 4 dB)<br>
       • Bug fixes (fixed --downsample 1 which produced white noise, but --downsample 2, 4 should also work now)<br>
       <br>
       
       bash-3.2$ python mod_player.py --help              
       usage: mod_player.py [-h] [--downsample DOWNSAMPLE] [--bitrate {lo,med,hi,ultra}]
                            [--vec-dim {2,4,8}] [--resampler {linear,bspline,lanczos3}]
                            [--no-split] [--split] [--viz {0,1,2,3,4,5}] [--no-rvq2]
                            modfile
       
       MOD/S3M Player - Generates HTML player + ShaderToy GLSL + PNG samples
       
       positional arguments:
         modfile               MOD or S3M file to play
       
       options:
         -h, --help            show this help message and exit
         --downsample DOWNSAMPLE
                               Sample decimation factor: 1=full-rate, 2=22kHz, 4=11kHz. HF
                               percussion (cymbals/rides) gets max(1,DS//2) to keep shimmer.
                               (default: 1)
         --bitrate {lo,med,hi,ultra}
                               RVQ codebook size (mp3-style quality knob). lo=K(128,64) 13b/pair
                               smallest+grainy, med=K(256,128) 15b/pair balanced, hi=K(512,256)
                               17b/pair sharper, ultra=K(1024,512) 19b/pair near-transparent.
                               (default: med)
         --vec-dim {2,4,8}     RVQ vector dimensionality. 8=smallest (~2.1 bits/sample), 4=medium
                               (4.25 bits/sample), 2=highest fidelity (8.5 bits/sample). (default:
                               8)
         --resampler {linear,bspline,lanczos3}
                               Sample resampler. linear=2-tap (cheapest, ProTracker-style),
                               bspline=4-tap cubic (smooth/soft), lanczos3=6-tap sinc
                               (sharpest/brightest, ~50% more cost). (default: lanczos3)
         --no-split            Keep VQ arrays + decoders in Common tab. Required for
                               oscilloscope/spectrum/Buffer A visualizers to decode actual audio
                               via getChannelOutput. Default ON. (default: True)
         --split               Split VQ arrays into Sound tab — fast Common compile, but breaks
                               audio-driven visualizers (no getChannelOutput in Image/BufferA).
                               (default: True)
         --viz {0,1,2,3,4,5}   Image-tab visualizer: 0=None (black backdrop, fastest compile),
                               1=Reactive 001 (PAEz fork — SDF circles + cosmic web, default),
                               2=Fluxline Surfer (mrange — DR2 dodecahedron + glowtracer), 3=Zuvuya
                               (city/stars + audio-reactive curtain), 4=Maya (raymarched fractal
                               tunnel-warp), 5=Dodecahedron (Philip Bertani — DR2 IFS fractal
                               raymarcher). (default: 1)
         --no-rvq2             Skip RVQ stage 2 (residual quantization). Drops ~40% of sample-data
                               const arrays from Sound tab → faster compile. Quality cost: ~4 dB
                               SNR (sounds noisier but pitch is unchanged). IMPORTANT: when re-
                               pasting into ShaderToy, paste BOTH the new Common AND new Sound —
                               otherwise mismatched RVQ_BITS produces high-pitch garbage from a
                               stale Common reading 15-bit-packed codes that were actually written
                               at 8 bits. (default: False)
                        
`Live Demos` <br>
<br>
       • https://www.shadertoy.com/view/7XlGRr<br>
<br>

<img width="50%" height="50%" alt="image" src="https://github.com/user-attachments/assets/b10632a5-c7a6-47e7-a28e-832251b19e6c" />
<br>

If you feel like it donate PayPal to <subband@protonmail.com><br>

I am a life long committed audio guy from the start, at 6 years
old they picked me from the crowd and said, you - must do music,
so since 6 to 16 I was learning to play piano and learning music
theory. I do write some small putty tunes, nothing too amazing,
but at least I can call my own creation, you can find them on 
my SoundCloud page at:

 https://soundcloud.com/analogintelligence

## Overview

This python script mod_player.py allows you to convert MOD
of size up to 150k-200k into a ShaderToy shader. 

## Features

✅ RLE compression for patterns
✅ Optimized ivec4 chunked data loader
✅ Creates GLSL infrastructure for you
✅ Has a fancy tracker-like GUI with note, fx, displayed
✅ Handles looping samples

## Requirements

```bash
pip install numpy pillow
```

## Usage

### Basic conversion:
```bash
python mod_player.py yourfavorite.mod
```
(pick MODs under 150k otherwise you would have to 
deal with external storage into PNG. I will write 
explanation how you can playback any size MOD inside
of ShaderToy using a special plugin I wrote for FireFox
which lets you drop a custom texture and at same time
it resets iTime clock for engine to pick it up)

### Conversion using downsample argument:
```bash
python mod_player.py yourfavorite.mod --downsample 2
```
(number can be 2, 4, and 8, beyond that sample degradation is too 
unbearable)

ars_player.html
ars_player_data.png
ars_shadertoy_bufferA.glsl
ars_shadertoy_common.glsl
ars_shadertoy_image.glsl
ars_shadertoy_instructions.txt
ars_shadertoy_sound.glsl

For a MOD file named ars.mod, then the script would generate:
- `ars_player.html` - HTML page that loads GLSL outside of ShaderToy.com
- `ars_player_data.png` - Pattern data texture (RGBA)  
- `ars_shadertoy_bufferA.glsl` - Shader code that goes into Buffer A tab (+)
- `ars_shadertoy_common.glsl` - Common tab on ShaderToy (+)
- `ars_shadertoy_image.glsl` - Image tab on ShaderToy (+)
- `ars_shadertoy_sound.glsl` - Sound tab on ShaderToy (+)

In addition to that you need to insert alphabet texture on Image tab
into iChannel0, and add Buffer A to iChannel1 on the same Image tab,
and you need to feedback Buffer A onto itself by setting iChannel0 
of the Buffer A tab with itself.

## How It Works

### 1. Sample Packing

All samples are converted to floating-point format and packed into an RGBA texture:
- Each pixel stores 4 sample values (R, G, B, A channels)
- 8-bit samples converted to range [0, 255]
- Samples stored sequentially with index table

### 2. Sample Index Table

The GLSL code contains a sample table with 6 values per sample:
- **Start position** in texture
- **Length** in samples
- **Loop start** position
- **Loop length**
- **Volume** (0-64)
- **Finetune** value

### 3. Pattern Data

Pattern information is encoded as:
- **Sample number** (5 bits)
- **Period** (12 bits) - determines pitch
- **Effect type** (4 bits)
- **Effect parameter** (8 bits)

For large MODs, patterns are packed into a second texture.

### 4. GLSL Playback

The shader:
- Reads sample data from texture via UV coordinates
- Handles sample looping
- Converts ProTracker periods to playback rates
- Mixes 4 channels
- (Framework for) processes effects

## Technical Details

### MOD Format

ProTracker MOD files contain:
- 31 sample slots (8-bit signed PCM)
- 4-channel pattern data
- Up to 128 pattern positions
- 64 rows per pattern
- Effects: arpeggio, portamento, vibrato, etc.

### Sample Rate Conversion

Amiga uses period-based timing:
```
Playback Rate = 7093789.2 / (period × 2)
```

Middle C (period 428) ≈ 8287 Hz

### Texture Format

**Sample Texture:**
- Width: 1024 pixels (configurable)
- Height: Auto-calculated based on total samples
- Format: RGBA8
- Data: [-1.0, 1.0] mapped to [0, 255]

**Pattern Texture:**
- Width: 1024 pixels
- Format: RGBA8
- Each note = 4 bytes (sample, period_hi, period_lo|effect, param)

### Limitations

- ShaderToy has shader size limits (~64KB but you win some with RLE
  pattern compression, and packing data into an ivec4 you get 8x byte storage,
  so multiply that 64k by 8 potentially or more)
- Large MODs may need pattern data in texture, not code
- Effects need manual implementation
- No support for extended MOD formats (8-channel, etc.)

## Example: Playing Your MOD

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

## MOD Effects Reference

Common ProTracker effects:
- `0xy` - Arpeggio
- `1xx` - Portamento up
- `2xx` - Portamento down  
- `3xx` - Tone portamento
- `4xy` - Vibrato
- `5xy` - Tone portamento + volume slide
- `6xy` - Vibrato + volume slide
- `9xx` - Set sample offset
- `Axy` - Volume slide
- `Bxx` - Position jump
- `Cxx` - Set volume
- `Dxx` - Pattern break
- `Fxx` - Set speed/tempo

## Advanced Usage

### Custom Texture Size

Modify the `texture_width` parameter:
```python
texture, info, total = pack_samples_to_texture(mod, texture_width=2048)
```

### Sample Rate

Adjust target sample rate for resampling (if needed):
```python
texture, info, total = pack_samples_to_texture(mod, target_rate=44100)
```

## Testing

Try with the classic "12th Warrior" MOD:
```bash
python mod_to_shadertoy_complete.py 12th_warrior.mod
```

## Troubleshooting

**Problem:** Texture too large
- **Solution:** Increase texture width or split samples

**Problem:** Shader won't compile
- **Solution:** Pattern data might be too big - use texture encoding

**Problem:** Audio sounds wrong
- **Solution:** Check sample rate conversion and period calculations

**Problem:** No sound output
- **Solution:** Verify iChannel0 is set to sample texture

## TODO / Future Enhancements

- [ ] Full effect implementation
- [ ] Multi-pattern texture optimization  
- [ ] Support for 8-channel MODs
- [ ] XM/IT format support
- [ ] Real-time pattern editor
- [ ] Visualization support

## Credits

Created for converting Amiga ProTracker MOD files to ShaderToy format.

Based on ProTracker specification and ShaderToy audio API.

## License

Free for non-commercial use. Ask author for more information

Contact:<br>
   Orblivius : <subband@gmail.com><br>
<br>
   Donate PayPal : <subband@protonmail.com><br>
   
## Links

- [ShaderToy](https://www.shadertoy.com/)
- [The Mod Archive](https://modarchive.org/)
- [ProTracker Spec](http://16-bits.org/mod/)


---

