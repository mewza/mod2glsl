# ShaderToy Sound Shader Guide for MOD Players

## ShaderToy Sound Shader Format

### Correct Function Signature

```glsl
vec2 mainSound(float time)
```

**NOT:**
```glsl
void mainSound(out vec2 sound, int sampleIndex, float time)  // WRONG!
```

### Parameters

- **Input:** `float time` - Current playback time in seconds
- **Output:** `vec2` - Stereo audio (left, right channels)
- **Range:** Audio samples should be in range [-1.0, 1.0]

### Available Uniforms

```glsl
uniform float iSampleRate;  // Sample rate (typically 44100 Hz)
uniform float iTime;         // Shader playback time
// ... other standard ShaderToy uniforms
```

### How It Works

ShaderToy calls `mainSound(time)` for each audio sample that needs to be generated:

```
Sample 0: mainSound(0.0000000) → vec2(-0.1, -0.1)
Sample 1: mainSound(0.0000227) → vec2(0.05, 0.05)  // 1/44100
Sample 2: mainSound(0.0000454) → vec2(0.12, 0.12)  // 2/44100
...
```

The shader generates audio in real-time as the user plays it.

## The Stateless Challenge

### Problem: GLSL Shaders Are Stateless

Traditional MOD players maintain state:
```c
// Traditional approach (C/C++)
Channel channels[4];  // Persistent state

void processNextSample() {
    for (int i = 0; i < 4; i++) {
        channels[i].samplePos += channels[i].rate;  // Update state
        output += getSample(channels[i]);
    }
}
```

**This doesn't work in GLSL!** Each `mainSound()` call is independent - you can't store state between calls.

### Solution: Time-Based State Reconstruction

Instead of maintaining state, **recalculate everything from the current time**:

```glsl
vec2 mainSound(float time) {
    // 1. Calculate song position from time
    int currentRow = int(time * BPM / 2.5);
    int pattern = currentRow / 64;
    int row = currentRow % 64;
    
    // 2. For each channel, find what should be playing NOW
    vec2 output = vec2(0.0);
    for (int ch = 0; ch < 4; ch++) {
        // Scan backwards to find last note trigger
        Note lastNote = findLastNoteForChannel(pattern, row, ch);
        
        // Calculate how long this note has been playing
        float noteStartTime = calculateNoteStartTime(pattern, row, ch);
        float timeSinceStart = time - noteStartTime;
        
        // Calculate sample position
        float samplePos = timeSinceStart * sampleRate;
        
        // Read and mix sample
        output += readSampleAtPosition(lastNote.sample, samplePos);
    }
    
    return output;
}
```

## MOD Player Architecture for ShaderToy

### Step 1: Encode All Pattern Data

Store pattern data in a texture (iChannel1):

```
Byte 0-3:   Pattern 0, Row 0, Channel 0 (sample, period, effect, param)
Byte 4-7:   Pattern 0, Row 0, Channel 1
Byte 8-11:  Pattern 0, Row 0, Channel 2
Byte 12-15: Pattern 0, Row 0, Channel 3
Byte 16-19: Pattern 0, Row 1, Channel 0
...
```

### Step 2: Song Position Calculation

```glsl
// MOD timing
const float TICKS_PER_ROW = 6.0;  // Default speed
const float ROWS_PER_PATTERN = 64.0;

float getRowFromTime(float time, float bpm) {
    // Each row takes (60 / BPM) * (ticks / 6) seconds
    float secondsPerRow = (60.0 / bpm) * (TICKS_PER_ROW / 6.0);
    return time / secondsPerRow;
}
```

### Step 3: Pattern Reading

```glsl
Note getNote(int pattern, int row, int channel) {
    // Calculate byte offset in pattern texture
    int offset = (pattern * 64 * 4 + row * 4 + channel) * 4;
    
    // Read 4 bytes from texture
    vec4 data = readPatternData(offset);
    
    Note n;
    n.sample = int(data.r * 255.0);
    n.period = int(data.g * 255.0) * 16 + int(data.b * 255.0) / 16;
    n.effect = int(mod(data.b * 255.0, 16.0));
    n.param = int(data.a * 255.0);
    
    return n;
}
```

### Step 4: Note Tracking Per Channel

```glsl
// Find what note is currently playing on a channel
Note getCurrentNoteForChannel(float time, int channel) {
    float currentRow = getRowFromTime(time, DEFAULT_BPM);
    int pattern = int(currentRow / 64.0);
    int row = int(mod(currentRow, 64.0));
    
    // Scan backwards from current row to find last note trigger
    for (int r = row; r >= 0; r--) {
        Note n = getNote(pattern, r, channel);
        if (n.sample > 0) {
            return n;  // Found the note!
        }
    }
    
    // If no note in current pattern, check previous patterns...
    // (simplified here)
    
    return Note(0, 0, 0, 0);  // No note
}
```

### Step 5: Sample Position Calculation

```glsl
float getSamplePosition(float time, Note note, int channel) {
    // Find when this note was triggered
    float noteStartTime = findNoteTriggerTime(note, channel);
    float timeSinceStart = time - noteStartTime;
    
    // Calculate playback rate from period
    float rate = 7093789.2 / (float(note.period) * 2.0);
    
    // Sample position = time * rate
    return timeSinceStart * rate;
}
```

### Step 6: Complete mainSound Implementation

```glsl
vec2 mainSound(float time) {
    vec2 output = vec2(0.0);
    
    // Process all 4 channels
    for (int ch = 0; ch < 4; ch++) {
        // Get current note for this channel
        Note note = getCurrentNoteForChannel(time, ch);
        
        if (note.sample > 0) {
            // Calculate sample position
            float samplePos = getSamplePosition(time, note, ch);
            
            // Read sample from texture
            float sample = readSampleFromTexture(note.sample, samplePos);
            
            // Mix into output
            output += vec2(sample) * 0.25;  // Divide by 4 for 4 channels
        }
    }
    
    return output;
}
```

## Performance Optimization Tips

### 1. Precompute Pattern Offsets

Instead of scanning backwards every time, encode "last note" info:

```
For each row, store: (sample, period, timeSinceNoteStart)
```

### 2. Limit Lookback Range

```glsl
// Don't scan too far back
for (int r = row; r >= max(0, row - 64); r--) {
    // ...
}
```

### 3. Use Texture Lookups Efficiently

```glsl
// Cache texture reads
vec4 patternData[256];  // One pattern worth
// Read once, use many times
```

## Effect Implementation

Effects need to modify parameters based on time within the row:

```glsl
float applyVibratoEffect(float period, int param, float timeInRow) {
    int speed = param >> 4;
    int depth = param & 0x0F;
    
    float vibrato = sin(timeInRow * float(speed)) * float(depth);
    return period + vibrato;
}
```

## Testing Your MOD Player

### Simple Test: Single Note

```glsl
vec2 mainSound(float time) {
    // Play middle C at 440 Hz for 1 second
    if (time < 1.0) {
        return vec2(sin(time * 440.0 * 6.28318));
    }
    return vec2(0.0);
}
```

### Test: Read Sample from Texture

```glsl
vec2 mainSound(float time) {
    // Play sample 0 at original rate
    float samplePos = time * 8287.0;  // 8287 Hz = middle C period
    float sample = readSampleFromTexture(0, samplePos);
    return vec2(sample);
}
```

### Test: Pattern Sequencing

```glsl
vec2 mainSound(float time) {
    // Play note C, then E, then G, each for 0.5 seconds
    int noteIndex = int(time / 0.5);
    int periods[3] = int[](428, 340, 285);  // C, E, G
    
    if (noteIndex < 3) {
        float rate = 7093789.2 / (float(periods[noteIndex]) * 2.0);
        float samplePos = fract(time / 0.5) * rate;
        float sample = readSampleFromTexture(0, samplePos);
        return vec2(sample);
    }
    
    return vec2(0.0);
}
```

## Common Pitfalls

❌ **Don't:** Try to maintain state between calls
```glsl
float samplePos = 0.0;  // Global won't persist!
vec2 mainSound(float time) {
    samplePos += rate;  // WRONG - resets each call
}
```

✅ **Do:** Calculate everything from time
```glsl
vec2 mainSound(float time) {
    float samplePos = time * rate;  // Correct!
}
```

❌ **Don't:** Use loops that depend on previous iterations
```glsl
for (int i = 0; i < 1000; i++) {
    state.update();  // Can't preserve state
}
```

✅ **Do:** Use time-based calculations
```glsl
float position = time * rate;
```

## Performance Considerations

- Texture reads are expensive - minimize them
- Complex pattern scanning can be slow
- Consider encoding "helper data" to avoid runtime calculation
- Test with simple MODs first

## Recommended Approach

1. **Start simple:** Get one sample playing at one pitch
2. **Add sample reading:** Load samples from texture successfully  
3. **Add pattern reading:** Read notes from pattern texture
4. **Add timing:** Calculate which row should be playing
5. **Add multi-channel:** Mix 4 channels
6. **Add effects:** Implement common effects one by one

## Summary

The key difference from traditional MOD players:

**Traditional:** Maintain state, update incrementally
**ShaderToy:** Stateless, recalculate everything from time

Think of it like rendering a video frame - you don't know what the previous frame was, you only know "draw frame at time T" and must figure out what should be visible based solely on T.

For audio: You don't know what the previous sample was, you only know "generate audio at time T" and must figure out what should be audible based solely on T.
