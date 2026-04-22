## MOD2GLSL - Amiga MOD Player v1.1 for ShaderToy/GLSL<br>(c) 2026 Orblivius. All rights reserved.

## NEW: Real-time Surround Sound + FAT DSP processing

Convert Amiga ProTracker MOD(and soon S3M/XM and IT) files 
and export them as ShaderToy shaders. 

Please please please donate $ to support this project, 
I am (still) not rich, and just like you trying to make
ends meet, and to be honest, I never received any $ from 
anybody for open source  projects, I don't mind, but 
maybe this time it will be different, and how
much time I donated to support developers on #macdev, 
and released source code for all to leran from from my
35 years software engineering career, with a break for 15 years
admittedly, due to serious health issues. It'd probably make 
me happier and more optimistic about youth and the future.

Donate PayPal to <subband@protonmail.com><br>

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
   Dmitry Boldyrev : <subband@gmail.com><br>
   <a href="https://t.me/hrooster">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/82/Telegram_logo.svg" width="30">
</a>
<br>
   Donate PayPal : <subband@protonmail.com><br>
   
## Links

- [ShaderToy](https://www.shadertoy.com/)
- [The Mod Archive](https://modarchive.org/)
- [ProTracker Spec](http://16-bits.org/mod/)


---

