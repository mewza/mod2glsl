# MOD to ShaderToy Converter

Convert Amiga ProTracker MOD files into texture-based ShaderToy players!

## Overview

This tool converts MOD files into:
1. **PNG texture** containing all sample data
2. **GLSL shader code** for playback in ShaderToy
3. **Pattern data texture** (optional) for complete playback

## Features

✅ Extracts and packs all samples into a single texture
✅ Generates sample index table with loop points
✅ Creates GLSL player framework
✅ Encodes pattern data for full playback
✅ Handles looping samples
✅ Support for 31 samples

## Requirements

```bash
pip install numpy pillow
```

## Usage

### Basic conversion:
```bash
python mod_to_shadertoy.py yourfile.mod
```

### Complete conversion with pattern data:
```bash
python mod_to_shadertoy_complete.py yourfile.mod
```

This generates:
- `yourfile_samples.png` - Sample data texture (RGBA)
- `yourfile_patterns.png` - Pattern data texture (RGBA)  
- `yourfile_player.glsl` - GLSL shader code
- `yourfile_info.json` - MOD metadata

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

## ShaderToy Setup

1. Create a new **Sound** shader on ShaderToy
2. Upload `*_samples.png` as **iChannel0**
3. Upload `*_patterns.png` as **iChannel1** (if available)
4. Paste the GLSL code from `*_player.glsl`
5. Click play!

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

- ShaderToy has shader size limits (~64KB)
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

Public domain / MIT - Use freely!

## Links

- [ShaderToy](https://www.shadertoy.com/)
- [The Mod Archive](https://modarchive.org/)
- [ProTracker Spec](http://16-bits.org/mod/)

---

**Example Output:**

```
Loading: 12th_warrior.mod

Title: 12th warrior
Patterns: 13
Length: 12 positions

Saved: 12th_warrior_samples.png (1024x143)
Saved: 12th_warrior_patterns.png (1024x13)
Saved: 12th_warrior_player.glsl
Saved: 12th_warrior_info.json

SHADERTOY SETUP:
1. Upload 12th_warrior_samples.png as iChannel0
2. Upload 12th_warrior_patterns.png as iChannel1
3. Copy shader code
4. Click Play!
```
