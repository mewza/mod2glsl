# Sample Buffer Memory Map - Complete Guide

## Overview

When converting a MOD file to ShaderToy format, all samples are packed into a single RGBA texture. This document explains exactly where each sample lives in that texture.

## Buffer Structure

### Texture Format
- **Format:** RGBA8 (4 channels, 8 bits each)
- **Width:** 1024 pixels (configurable)
- **Height:** Auto-calculated based on total sample data
- **Packing:** 4 samples per pixel (one per channel: R, G, B, A)

### Data Layout

```
Pixel 0: [R G B A] <- Samples 0, 1, 2, 3
Pixel 1: [R G B A] <- Samples 4, 5, 6, 7
Pixel 2: [R G B A] <- Samples 8, 9, 10, 11
...
```

## Sample Index Table

Each MOD sample has 6 values stored in the `SAMPLE_TABLE` constant:

```glsl
const int SAMPLE_TABLE[186] = int[186](
    // Sample 1: [start, length, loop_start, loop_length, volume, finetune]
    0,     5234,  0,     0,     64,  0,
    
    // Sample 2:
    5234,  3120,  0,     1024,  64,  0,
    
    // Sample 3:
    8354,  2048,  512,   512,   64,  0,
    
    // ... (31 total samples)
);
```

### Table Format

For sample N (1-based index):
```glsl
int base = (N - 1) * 6;

int start       = SAMPLE_TABLE[base + 0];  // Absolute start position in buffer
int length      = SAMPLE_TABLE[base + 1];  // Sample length in samples
int loop_start  = SAMPLE_TABLE[base + 2];  // Loop start offset (relative to sample start)
int loop_length = SAMPLE_TABLE[base + 3];  // Loop length (0 = no loop)
int volume      = SAMPLE_TABLE[base + 4];  // Default volume (0-64)
int finetune    = SAMPLE_TABLE[base + 5];  // Finetune value (-8 to +7)
```

## Reading Samples from Texture

### Step-by-Step Process

#### 1. Get Sample Info
```glsl
void getSampleInfo(int sampleIdx, 
                   out int start, out int length,
                   out int loopStart, out int loopLength,
                   out int volume, out int finetune) {
    int base = sampleIdx * 6;
    start = SAMPLE_TABLE[base + 0];
    length = SAMPLE_TABLE[base + 1];
    loopStart = SAMPLE_TABLE[base + 2];
    loopLength = SAMPLE_TABLE[base + 3];
    volume = SAMPLE_TABLE[base + 4];
    finetune = SAMPLE_TABLE[base + 5];
}
```

#### 2. Calculate Buffer Position
```glsl
// Example: Read sample 1 at position 100
int sampleNumber = 0;  // 0-based (sample 1 = index 0)
int samplePosition = 100;

int start, length, loopStart, loopLength, volume, finetune;
getSampleInfo(sampleNumber, start, length, loopStart, loopLength, volume, finetune);

int absolutePosition = start + samplePosition;
```

#### 3. Convert to Texture Coordinates
```glsl
// Calculate which pixel and channel
int pixelIndex = absolutePosition / 4;
int channel = absolutePosition % 4;  // 0=R, 1=G, 2=B, 3=A

// Calculate pixel coordinates
int x = pixelIndex % TEXTURE_WIDTH;
int y = pixelIndex / TEXTURE_WIDTH;

// Calculate UV coordinates (0.0 to 1.0)
vec2 uv = (vec2(x, y) + 0.5) / vec2(TEXTURE_WIDTH, TEXTURE_HEIGHT);
```

#### 4. Read Sample Value
```glsl
// Read pixel from texture
vec4 pixel = texture(iChannel0, uv);

// Extract the specific channel
float sampleValue = pixel[channel];

// Convert from [0, 1] to [-1, 1]
sampleValue = (sampleValue - 0.5) * 2.0;
```

### Complete Read Function
```glsl
float readSample(sampler2D tex, int position) {
    int pixelIdx = position / 4;
    int channel = position % 4;
    int x = pixelIdx % TEXTURE_WIDTH;
    int y = pixelIdx / TEXTURE_WIDTH;
    
    vec2 uv = (vec2(x, y) + 0.5) / vec2(TEXTURE_WIDTH, TEXTURE_HEIGHT);
    vec4 pixel = texture(tex, uv);
    
    return (pixel[channel] - 0.5) * 2.0;
}
```

## Loop Handling

### One-Shot Samples
```glsl
if (loopLength <= 2) {
    // No loop - stop at end
    if (samplePos >= length) {
        return 0.0;  // Silence
    }
}
```

### Looping Samples
```glsl
if (loopLength > 2) {
    // Has a loop
    if (samplePos >= loopStart + loopLength) {
        // Wrap around within loop region
        int offset = (samplePos - loopStart) % loopLength;
        samplePos = loopStart + offset;
    }
}
```

## Example Memory Map

For a MOD with 3 samples:

```
Sample 1: "BassDrum"
  Byte Range:  0 - 4095 (length: 4096)
  Start: Pixel 0 @ (0, 0).R
  End:   Pixel 1023 @ (1023, 0).A
  Loop: None (one-shot)

Sample 2: "Strings"  
  Byte Range:  4096 - 8191 (length: 4096)
  Start: Pixel 1024 @ (0, 1).R
  End:   Pixel 2047 @ (1023, 1).A  
  Loop: 1024 - 3072 (length: 2048)

Sample 3: "Piano"
  Byte Range:  8192 - 10239 (length: 2048)
  Start: Pixel 2048 @ (0, 2).R
  End:   Pixel 2559 @ (511, 2).A
  Loop: None (one-shot)
```

## Visual Layout

```
Texture (1024x3):

Row 0: [Sample 1...................................................]
       ^ Pixel 0                                      Pixel 1023 ^
       
Row 1: [Sample 2...................................................]
       ^ Pixel 1024                                   Pixel 2047 ^
       
Row 2: [Sample 3...................][Empty........................]
       ^ Pixel 2048    Pixel 2559 ^
```

## Address Calculation Examples

### Example 1: Read Sample 1, Position 500

```
1. Sample info: start=0, length=4096
2. Absolute position: 0 + 500 = 500
3. Pixel index: 500 / 4 = 125
4. Channel: 500 % 4 = 0 (R channel)
5. X coordinate: 125 % 1024 = 125
6. Y coordinate: 125 / 1024 = 0
7. UV: (125.5/1024, 0.5/3) = (0.1225, 0.1667)
```

### Example 2: Read Sample 2, Position 1500 (in loop)

```
1. Sample info: start=4096, loop_start=1024, loop_length=2048
2. Position 1500 >= loop region (1024-3072)
3. Wrap: offset = (1500 - 1024) % 2048 = 476
4. Wrapped position: 1024 + 476 = 1500 (same, still in loop)
5. Absolute position: 4096 + 1500 = 5596
6. Pixel index: 5596 / 4 = 1399
7. Channel: 5596 % 4 = 0 (R channel)
8. X coordinate: 1399 % 1024 = 375
9. Y coordinate: 1399 / 1024 = 1
10. UV: (375.5/1024, 1.5/3) = (0.3667, 0.5)
```

## Buffer Map Files

The converter generates these map files:

### 1. `*_buffer_map.json`
- Machine-readable buffer layout
- Sample positions and metadata
- GLSL code snippets

### 2. `*_buffer_map.txt` (from console output)
- Human-readable memory map
- ASCII visualization
- Detailed address calculations

### 3. `*_buffer_map.html` (optional)
- Interactive visual browser tool
- Hover pixels to see details
- Color-coded samples

## Tools

### Generate Buffer Map
```bash
python mod_buffer_map.py yourfile.mod
```

Output:
- Detailed console output with memory map
- JSON file with programmatic access
- Visual representations

### View in Browser
```bash
# The HTML visualizer is included in buffer_visualizer.py
# Integrated into the main conversion tools
```

## Performance Tips

### Cache Texture Reads
```glsl
// DON'T: Read same pixel multiple times
float s1 = readSample(tex, pos);
float s2 = readSample(tex, pos + 1);  // Might re-read same pixel!

// DO: Read pixel once, extract channels
int pixIdx = pos / 4;
vec4 pixel = readPixel(tex, pixIdx);
float s1 = pixel[pos % 4];
float s2 = pixel[(pos + 1) % 4];
```

### Batch Lookups
```glsl
// When reading sequential samples, batch pixel reads
for (int i = 0; i < 16; i += 4) {
    vec4 pixel = readPixelAtPosition(start + i);
    // Process all 4 samples from this pixel
}
```

## Debugging Tips

### Verify Sample Location
```glsl
// Test: Read first sample value
int testPos = SAMPLE_TABLE[0];  // Start of sample 1
float value = readSample(iChannel0, testPos);
// Should be valid audio data [-1, 1]
```

### Check Texture Coordinates
```glsl
// Visualize UV coordinates
vec2 testUV = calculateUV(somePosition);
// testUV should be in range [0, 1]
```

### Validate Loop Region
```glsl
// Ensure loop wrap works correctly
int looped = applyLoop(position, loopStart, loopLength);
// looped should always be in range [loopStart, loopStart + loopLength)
```

## Common Issues

### Issue: Clicking/Popping in Audio
- **Cause:** Reading wrong pixel or channel
- **Fix:** Verify address calculations

### Issue: Sample Plays at Wrong Pitch  
- **Cause:** Period-to-rate conversion error
- **Fix:** Check Amiga clock rate (7093789.2 Hz)

### Issue: Loop Glitches
- **Cause:** Loop boundaries incorrect
- **Fix:** Verify loop_start and loop_length values

### Issue: Silence When Should Play
- **Cause:** Reading outside buffer bounds
- **Fix:** Add bounds checking, verify sample table

## Summary

The buffer map provides:
- ✅ Exact byte positions for each sample
- ✅ Texture coordinates for GLSL access
- ✅ Loop information for seamless playback
- ✅ Visual tools for debugging
- ✅ Performance optimization guidance

All samples → One texture → Fast GPU access → Real-time MOD playback!
