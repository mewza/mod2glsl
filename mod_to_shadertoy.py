#!/usr/bin/env python3
"""
MOD to ShaderToy Converter
Converts Amiga ProTracker MOD files into a texture-based ShaderToy player.
"""

import struct
import numpy as np
from PIL import Image
import sys
import os

class MODFile:
    def __init__(self, filename):
        self.filename = filename
        self.samples = []
        self.patterns = []
        self.song_positions = []
        self.num_patterns = 0
        self.bpm = 125
        self.speed = 6
        
        self.parse_mod(filename)
    
    def parse_mod(self, filename):
        with open(filename, 'rb') as f:
            # Read song title (20 bytes)
            self.title = f.read(20).decode('ascii', errors='ignore').strip('\x00')
            print(f"Song title: {self.title}")
            
            # Read 31 sample headers
            for i in range(31):
                sample_name = f.read(22).decode('ascii', errors='ignore').strip('\x00')
                sample_length = struct.unpack('>H', f.read(2))[0] * 2  # Words to bytes
                finetune = struct.unpack('B', f.read(1))[0] & 0x0F
                if finetune > 7:
                    finetune = finetune - 16
                volume = struct.unpack('B', f.read(1))[0]
                repeat_point = struct.unpack('>H', f.read(2))[0] * 2
                repeat_length = struct.unpack('>H', f.read(2))[0] * 2
                
                if repeat_length <= 2:
                    repeat_length = 0
                
                self.samples.append({
                    'name': sample_name,
                    'length': sample_length,
                    'finetune': finetune,
                    'volume': volume,
                    'repeat_point': repeat_point,
                    'repeat_length': repeat_length,
                    'data': None
                })
                
                if sample_length > 0:
                    print(f"Sample {i+1:2d}: {sample_name[:20]:20s} len={sample_length:6d} vol={volume:3d} loop={repeat_point:6d}/{repeat_length:6d}")
            
            # Read song length and positions
            song_length = struct.unpack('B', f.read(1))[0]
            restart_pos = struct.unpack('B', f.read(1))[0]  # Not used in most MODs
            
            # Read pattern table (128 positions)
            self.song_positions = list(f.read(128))[:song_length]
            self.num_patterns = max(self.song_positions) + 1
            
            print(f"Song length: {song_length} positions")
            print(f"Number of patterns: {self.num_patterns}")
            print(f"Pattern order: {self.song_positions[:20]}...")
            
            # Read MOD signature (M.K., M!K!, FLT4, etc.)
            signature = f.read(4).decode('ascii', errors='ignore')
            print(f"MOD signature: {signature}")
            
            # Read pattern data (64 rows * 4 channels * 4 bytes per note)
            for p in range(self.num_patterns):
                pattern = []
                for row in range(64):
                    channels = []
                    for ch in range(4):
                        data = struct.unpack('>I', f.read(4))[0]
                        
                        # Decode note data
                        sample_num = (data & 0xF0000000) >> 24 | (data & 0x0000F000) >> 12
                        period = (data & 0x0FFF0000) >> 16
                        effect = (data & 0x00000FFF)
                        effect_type = (effect & 0xF00) >> 8
                        effect_param = effect & 0xFF
                        
                        channels.append({
                            'sample': sample_num,
                            'period': period,
                            'effect': effect_type,
                            'param': effect_param
                        })
                    pattern.append(channels)
                self.patterns.append(pattern)
            
            # Read sample data
            for i, sample in enumerate(self.samples):
                if sample['length'] > 0:
                    sample_data = f.read(sample['length'])
                    # Convert to signed 8-bit
                    sample['data'] = np.frombuffer(sample_data, dtype=np.int8)
                else:
                    sample['data'] = np.array([], dtype=np.int8)

def resample_audio(data, src_rate, dst_rate):
    """Simple linear resampling"""
    if len(data) == 0:
        return data
    
    src_len = len(data)
    dst_len = int(src_len * dst_rate / src_rate)
    
    if dst_len == 0:
        return np.array([], dtype=np.float32)
    
    # Linear interpolation
    src_indices = np.linspace(0, src_len - 1, dst_len)
    return np.interp(src_indices, np.arange(src_len), data.astype(np.float32))

def pack_samples_to_texture(mod, target_rate=44100, texture_width=1024):
    """Pack all samples into a texture with uniform sample rate"""
    
    print("\n=== Packing samples to texture ===")
    
    # Resample all samples to target rate
    resampled_samples = []
    sample_info = []
    
    total_samples = 0
    
    for i, sample in enumerate(mod.samples):
        if sample['length'] > 0:
            # ProTracker periods correspond to different playback rates
            # Middle C (period 428) = 8287 Hz at PAL timing
            # We'll use a base rate and adjust
            base_rate = 8363  # Amiga base rate for middle C
            
            # Convert 8-bit signed to float [-1, 1]
            data_float = sample['data'].astype(np.float32) / 128.0
            
            # Resample to target rate (we'll handle pitch shifting in shader)
            # For simplicity, we store at a fixed rate and handle rate conversion in shader
            resampled = data_float  # Keep original for now, shader will handle playback rate
            
            start_pos = total_samples
            length = len(resampled)
            
            sample_info.append({
                'index': i,
                'start': start_pos,
                'length': length,
                'loop_start': sample['repeat_point'],
                'loop_length': sample['repeat_length'],
                'volume': sample['volume'],
                'finetune': sample['finetune']
            })
            
            resampled_samples.append(resampled)
            total_samples += length
            
            print(f"Sample {i+1:2d}: {len(resampled):8d} samples, pos {start_pos:8d}")
        else:
            sample_info.append({
                'index': i,
                'start': 0,
                'length': 0,
                'loop_start': 0,
                'loop_length': 0,
                'volume': 0,
                'finetune': 0
            })
    
    # Concatenate all samples
    all_samples = np.concatenate(resampled_samples) if resampled_samples else np.array([])
    
    print(f"Total samples: {total_samples}")
    
    # Pack into texture (RGBA, each channel is a sample value)
    # We'll use R and G channels to store 16-bit precision per sample
    num_pixels = (len(all_samples) + 3) // 4  # 4 samples per pixel (RGBA)
    
    # Pad to fit
    padded_length = num_pixels * 4
    all_samples_padded = np.zeros(padded_length, dtype=np.float32)
    all_samples_padded[:len(all_samples)] = all_samples
    
    # Reshape to RGBA
    rgba_data = all_samples_padded.reshape(-1, 4)
    
    # Convert to 8-bit unsigned [0, 255]
    rgba_uint8 = ((rgba_data + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    
    # Calculate texture dimensions
    texture_height = (num_pixels + texture_width - 1) // texture_width
    
    # Pad to texture dimensions
    total_pixels = texture_width * texture_height
    if num_pixels < total_pixels:
        padding = np.zeros((total_pixels - num_pixels, 4), dtype=np.uint8)
        rgba_uint8 = np.vstack([rgba_uint8, padding])
    
    # Reshape to texture
    texture = rgba_uint8.reshape(texture_height, texture_width, 4)
    
    print(f"Texture size: {texture_width}x{texture_height} (RGBA)")
    
    return texture, sample_info, total_samples

def generate_glsl_code(mod, sample_info, texture_width, texture_height, total_samples):
    """Generate GLSL shader code for ShaderToy"""
    
    # Build sample info table as GLSL constants
    sample_table = "// Sample info: start, length, loop_start, loop_length, volume, finetune\n"
    sample_table += "const int SAMPLE_TABLE[186] = int[186](\n"  # 31 samples * 6 values
    
    entries = []
    for info in sample_info[:31]:
        entries.append(f"    {info['start']}, {info['length']}, {info['loop_start']}, {info['loop_length']}, {info['volume']}, {info['finetune']}")
    
    sample_table += ",\n".join(entries) + "\n);\n"
    
    # Period to frequency table (ProTracker period table)
    period_table = """
// ProTracker period table for notes
const int PERIOD_TABLE[37] = int[37](
    856, 808, 762, 720, 678, 640, 604, 570, 538, 508, 480, 453,  // C-1 to B-1
    428, 404, 381, 360, 339, 320, 302, 285, 269, 254, 240, 226,  // C-2 to B-2
    214, 202, 190, 180, 170, 160, 151, 143, 135, 127, 120, 113,  // C-3 to B-3
    0  // Empty
);
"""
    
    # Build pattern data (simplified - just note/sample info)
    # For full implementation, we'd encode all patterns, but this is a starting point
    
    glsl_code = f"""// MOD Player for ShaderToy
// Generated from: {mod.title}
// Texture size: {texture_width}x{texture_height}
// Total samples: {total_samples}

{sample_table}

{period_table}

const int NUM_SAMPLES = {len([s for s in sample_info if s['length'] > 0])};
const int TEXTURE_WIDTH = {texture_width};
const int TEXTURE_HEIGHT = {texture_height};
const float AMIGA_CLOCK = 7093789.2; // PAL
const float BASE_RATE = 8363.0;
// Note: iSampleRate uniform is provided by ShaderToy (typically 44100)

// Get sample info (6 values per sample)
void getSampleInfo(int sampleIdx, out int start, out int length, out int loopStart, out int loopLength, out int volume, out int finetune) {{
    int base = sampleIdx * 6;
    start = SAMPLE_TABLE[base + 0];
    length = SAMPLE_TABLE[base + 1];
    loopStart = SAMPLE_TABLE[base + 2];
    loopLength = SAMPLE_TABLE[base + 3];
    volume = SAMPLE_TABLE[base + 4];
    finetune = SAMPLE_TABLE[base + 5];
}}

// Read sample value from texture
float readSample(sampler2D sampleTex, int position, int channel) {{
    // 4 samples per pixel (RGBA)
    int pixelIdx = position / 4;
    int channelIdx = position % 4;
    
    int x = pixelIdx % TEXTURE_WIDTH;
    int y = pixelIdx / TEXTURE_WIDTH;
    
    vec2 uv = (vec2(x, y) + 0.5) / vec2(TEXTURE_WIDTH, TEXTURE_HEIGHT);
    vec4 pixel = texture(sampleTex, uv);
    
    float value = pixel[channelIdx];
    // Convert from [0,1] to [-1,1]
    return (value - 0.5) * 2.0;
}}

// Convert period to playback rate
float periodToRate(int period) {{
    if (period == 0) return 0.0;
    // Amiga formula: rate = AMIGA_CLOCK / (period * 2)
    return AMIGA_CLOCK / (float(period) * 2.0);
}}

// Play a sample at given position and rate
float playSample(sampler2D sampleTex, int sampleIdx, float position, float rate) {{
    int start, length, loopStart, loopLength, volume, finetune;
    getSampleInfo(sampleIdx, start, length, loopStart, loopLength, volume, finetune);
    
    if (length == 0) return 0.0;
    
    // Handle looping
    int samplePos = int(position);
    
    if (loopLength > 2) {{
        // Looping sample
        if (samplePos >= loopStart + loopLength) {{
            samplePos = loopStart + ((samplePos - loopStart) % loopLength);
        }}
    }}
    
    if (samplePos >= length) return 0.0;
    
    float sample = readSample(sampleTex, start + samplePos, 0);
    
    // Apply volume (0-64)
    sample *= float(volume) / 64.0;
    
    return sample;
}}

vec2 mainSound(float time) {{
    // ShaderToy Sound shader: returns stereo vec2, time in seconds
    // iSampleRate uniform provides sample rate (typically 44100 Hz)
    
    // This is a basic framework - you'll need to add:
    // 1. Pattern playback logic
    // 2. Effect processing (vibrato, portamento, etc.)
    // 3. Proper timing and tempo
    
    // Example: Play first sample as a test
    // In a real player, you'd track which notes are playing on which channels
    
    // Simple test: play sample 1 continuously
    int testSample = 0;  // First non-empty sample
    float playbackRate = BASE_RATE;
    float position = fract(time * playbackRate / iSampleRate) * 10000.0; // Example position
    
    float output = playSample(iChannel0, testSample, position, playbackRate);
    
    return vec2(output * 0.5);  // Return stereo (left, right)
}}

// Note: To fully implement the MOD player, you need to:
// 1. Encode pattern data (notes, effects, samples per row)
// 2. Implement a sequencer that steps through patterns
// 3. Track 4 channels with independent note playback
// 4. Implement MOD effects (0-F hex effects)
// 5. Handle tempo changes (BPM, speed)

// Pattern data would look like:
// const int SONG_LENGTH = {len(mod.song_positions)};
// const int PATTERN_ORDER[{len(mod.song_positions)}] = int[]({', '.join(map(str, mod.song_positions))});
// Then encode each pattern's 64 rows x 4 channels...
"""
    
    return glsl_code

def main():
    if len(sys.argv) < 2:
        print("Usage: python mod_to_shadertoy.py <modfile.mod>")
        print("\nConverts an Amiga ProTracker MOD file to ShaderToy format:")
        print("  - Generates a PNG texture with packed sample data")
        print("  - Generates GLSL code for playback")
        sys.exit(1)
    
    mod_filename = sys.argv[1]
    
    if not os.path.exists(mod_filename):
        print(f"Error: File '{mod_filename}' not found")
        sys.exit(1)
    
    print(f"Loading MOD file: {mod_filename}\n")
    
    # Parse MOD file
    mod = MODFile(mod_filename)
    
    # Pack samples to texture
    texture, sample_info, total_samples = pack_samples_to_texture(mod, texture_width=1024)
    
    # Save texture as PNG
    output_basename = os.path.splitext(os.path.basename(mod_filename))[0]
    texture_filename = f"{output_basename}_samples.png"
    
    img = Image.fromarray(texture, mode='RGBA')
    img.save(texture_filename)
    print(f"\nSaved texture: {texture_filename}")
    
    # Generate GLSL code
    glsl_code = generate_glsl_code(mod, sample_info, texture.shape[1], texture.shape[0], total_samples)
    
    glsl_filename = f"{output_basename}_player.glsl"
    with open(glsl_filename, 'w') as f:
        f.write(glsl_code)
    
    print(f"Saved GLSL code: {glsl_filename}")
    
    print("\n=== Instructions ===")
    print("1. Create a new ShaderToy shader")
    print("2. Add the PNG texture as iChannel0")
    print("3. Copy the GLSL code into the shader")
    print("4. Set shader to 'Sound' mode (bottom left)")
    print("5. The basic framework is there - you'll need to implement:")
    print("   - Pattern sequencing")
    print("   - Multi-channel playback (4 channels)")
    print("   - MOD effects")
    print("\nThis generates the foundation - a complete player requires")
    print("encoding pattern data and implementing the full playback engine.")

if __name__ == '__main__':
    main()
