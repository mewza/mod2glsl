#!/usr/bin/env python3
"""
MOD Player with Correct ProTracker Timing + Efficient int32 Storage + Microclick Removal
OPTIMIZATIONS:
  1. Pack 4 bytes per int32 (75% storage reduction)
  2. Volume crossfade microclick removal (C++-style dying channels)
"""

import struct
import numpy as np
import sys
import os
import json
import argparse

# Adaptive sample compression (BW analysis + anti-alias decimation)
try:
    from scipy.signal import resample_poly as _resample_poly
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ============================================================================
# OPTIMIZATION 1: Efficient int32 Packing
# ============================================================================

def pack_bytes_to_int32(byte_array):
    """
    Pack 4 bytes into int32 values for efficient storage
    
    Storage reduction: 75% smaller arrays!
    Before: const int data[13862] = int[](12, 34, 56, 78, ...)
    After:  const int data[3466] = int[](0x0C22384Eu, ...)
    
    Extraction in GLSL:
        int val = data[i];
        int b0 = int((val >> 24) & 0xFF);
        int b1 = int((val >> 16) & 0xFF);
        int b2 = int((val >> 8) & 0xFF);
        int b3 = int(val & 0xFF);
    """
    packed = []
    for i in range(0, len(byte_array), 4):
        b0 = byte_array[i] if i < len(byte_array) else 0
        b1 = byte_array[i+1] if i+1 < len(byte_array) else 0
        b2 = byte_array[i+2] if i+2 < len(byte_array) else 0
        b3 = byte_array[i+3] if i+3 < len(byte_array) else 0
        # Pack 4 bytes: (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
        packed_value = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
        packed.append(packed_value)
    return packed

def pack_bytes_to_int32(byte_array):
    """Pack 4 bytes per int32 — exact, no float precision issues."""
    packed = []
    for i in range(0, len(byte_array), 4):
        b0 = byte_array[i]     if i     < len(byte_array) else 0
        b1 = byte_array[i + 1] if i + 1 < len(byte_array) else 0
        b2 = byte_array[i + 2] if i + 2 < len(byte_array) else 0
        b3 = byte_array[i + 3] if i + 3 < len(byte_array) else 0
        packed.append((b0 << 24) | (b1 << 16) | (b2 << 8) | b3)
    return packed

def format_int32_chunk_glsl(ints, name, index):
    """Pack 4 int32s per ivec4 — signed decimal, same as original int[] but 4x smaller array."""
    def s32(v):
        v = v & 0xFFFFFFFF
        return v if v < 0x80000000 else v - 0x100000000
    ints = list(ints)
    while len(ints) % 4:
        ints.append(0)
    ivec4s = []
    for j in range(0, len(ints), 4):
        a,b,c,d = s32(ints[j]),s32(ints[j+1]),s32(ints[j+2]),s32(ints[j+3])
        ivec4s.append(f"ivec4({a},{b},{c},{d})")
    lines = []
    for j in range(0, len(ivec4s), 2):
        lines.append(", ".join(ivec4s[j:j+2]))
    body = ",\n    ".join(lines)
    return f"const ivec4 {name}{index}[{len(ivec4s)}] = ivec4[](\n    {body}\n);\n"

# ============================================================================
# CONFIGURATION
# ============================================================================

# Sample array chunk size for ShaderToy compatibility
# Adjust this based on your target platform's limits
# Typical values: 2000-5000 (GLSL has strict limits!)
CHUNK_SIZE = 5000  # Elements per array chunk

# Conservative estimate for total JSON size limit
# Adjust based on your target platform
SHADERTOY_LIMIT = 200000  # ~200KB (conservative estimate)

# ============================================================================

def compress_patterns_rle(pattern_data):
    """RLE compress pattern data"""
    compressed = []
    i = 0
    while i < len(pattern_data):
        run_length = 1
        current = pattern_data[i]
        
        # Count consecutive identical values
        while i + run_length < len(pattern_data) and pattern_data[i + run_length] == current and run_length < 255:
            run_length += 1
        
        # Use RLE if run >= 4
        if run_length >= 4:
            compressed.extend([255, run_length, current])  # 255 = RLE marker
            i += run_length
        else:
            # Raw bytes (escape 255)
            for j in range(run_length):
                if pattern_data[i] == 255:
                    compressed.extend([255, 1, 255])
                else:
                    compressed.append(pattern_data[i])
                i += 1
    
    return compressed

def create_sample_pngs(samples, mod_name, output_dir):
    """Create PNG files containing sample data"""
    from PIL import Image
    import os
    
    png_files = []
    
    for i, sample_data in enumerate(samples):
        if len(sample_data) == 0:
            continue
        
        # Convert samples to unsigned bytes (0-255)
        pixels = [(s + 128) % 256 for s in sample_data]
        
        # Create image: 1 row, width = sample length, grayscale
        width = len(pixels)
        img = Image.new('L', (width, 1))
        img.putdata(pixels)
        
        # Save PNG file
        png_filename = f"{mod_name}_sample_{i:02d}.png"
        png_path = os.path.join(output_dir, png_filename)
        img.save(png_path, format='PNG', optimize=True)
        png_files.append(png_filename)
    
    return png_files

def bw_compress_sample(data, sr=44100):
    """
    Bandwidth-adaptive sample compression.
    Analyzes frequency content via FFT, finds highest significant frequency,
    computes minimum required sample rate (Nyquist + 20% headroom),
    then applies anti-alias filter and decimates by the best power-of-2 factor.
    Returns (bw_factor, compressed_int8_array).
    """
    d = data.astype(np.float32)
    n = len(d)
    if n < 32 or not _HAS_SCIPY:
        return 1, d.astype(np.int8)
    
    # FFT with Hann window to find highest significant frequency
    fft_mag = np.abs(np.fft.rfft(d * np.hanning(n)))
    freqs   = np.fft.rfftfreq(n, 1.0/sr)
    peak    = np.max(fft_mag)
    if peak == 0:
        return 1, d.astype(np.int8)
    
    # Find highest bin with > 0.5% of peak energy
    sig_bins = np.where(fft_mag > peak * 0.005)[0]
    max_freq = freqs[sig_bins[-1]] if len(sig_bins) else 22050.0
    
    # Choose best power-of-2 downsample factor (Nyquist + 20% headroom)
    best_factor = 1
    for f in [2, 4, 8, 16]:
        if sr / f >= max_freq * 2.4:
            best_factor = f
    
    if best_factor == 1:
        return 1, d.astype(np.int8)
    
    # High-quality polyphase anti-alias filter + decimate
    compressed = _resample_poly(d, 1, best_factor)
    compressed = np.clip(np.round(compressed).astype(np.int32), -128, 127).astype(np.int8)
    return best_factor, compressed


class MODFile:
    def __init__(self, filename):
        self.filename = filename
        self.samples = []
        self.patterns = []
        self.song_positions = []
        self.num_patterns = 0
        self.title = ""
        self.parse_mod(filename)
    
    def parse_mod(self, filename):
        with open(filename, 'rb') as f:
            self.title = f.read(20).decode('ascii', errors='ignore').strip('\x00')
            
            for i in range(31):
                sample_name = f.read(22).decode('ascii', errors='ignore').strip('\x00')
                sample_length = struct.unpack('>H', f.read(2))[0] * 2
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
            
            song_length = struct.unpack('B', f.read(1))[0]
            restart_pos = struct.unpack('B', f.read(1))[0]
            self.song_positions = list(f.read(128))[:song_length]
            
            if len(self.song_positions) == 0:
                raise ValueError(f"Invalid MOD file '{filename}': song_length={song_length}, which results in 0 patterns. "
                               f"File may be corrupted or not a valid 31-instrument MOD format.")
            
            self.num_patterns = max(self.song_positions) + 1
            
            signature = f.read(4)
            
            for p in range(self.num_patterns):
                pattern = []
                for row in range(64):
                    channels = []
                    for ch in range(4):
                        data = struct.unpack('>I', f.read(4))[0]
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
            
            for i, sample in enumerate(self.samples):
                if sample['length'] > 0:
                    sample_data = f.read(sample['length'])
                    sample['data'] = np.frombuffer(sample_data, dtype=np.int8)
                else:
                    sample['data'] = np.array([], dtype=np.int8)
        
        # Scan first few rows of the song for effect F to find actual initial speed/tempo.
        # Many MODs set speed/BPM in row 0 of pattern 0 — using the hardcoded default
        # of speed=6 will play at the wrong speed if the MOD uses e.g. F03.
        self.initial_speed = 6
        self.initial_tempo = 125
        first_pattern = self.patterns[self.song_positions[0]]
        for row in first_pattern[:8]:       # scan first 8 rows max
            for ch in row:
                if ch['effect'] == 0xF and ch['param'] > 0:
                    if ch['param'] < 0x20:  # < 32 = speed
                        self.initial_speed = ch['param']
                    else:                   # >= 32 = BPM
                        self.initial_tempo = ch['param']

class S3MFile:
    """ScreamTracker 3 Module file parser"""
    def __init__(self, filename):
        self.filename = filename
        self.title = ""
        self.num_orders = 0
        self.num_instruments = 0
        self.num_patterns = 0
        self.initial_speed = 6
        self.initial_tempo = 125
        self.global_volume = 64
        self.master_volume = 48
        self.channel_settings = []
        self.orders = []
        self.instruments = []
        self.patterns = []
        self.num_channels = 0
        self.parse_s3m(filename)
    
    def parse_s3m(self, filename):
        """Parse S3M file format"""
        with open(filename, 'rb') as f:
            # Read header (96 bytes)
            header = f.read(96)
            
            # Song title (28 bytes)
            self.title = header[0:28].decode('ascii', errors='ignore').rstrip('\x00')
            
            # Check signature
            if header[28] != 0x1A:
                raise ValueError(f"Invalid S3M file: missing 0x1A marker")
            
            # Parse header fields
            self.num_orders = struct.unpack('<H', header[32:34])[0]
            self.num_instruments = struct.unpack('<H', header[34:36])[0]
            self.num_patterns = struct.unpack('<H', header[36:38])[0]
            
            # Check SCRM signature
            if header[44:48] != b'SCRM':
                raise ValueError(f"Invalid S3M file: missing SCRM signature")
            
            # Global settings
            self.global_volume = header[48]
            self.initial_speed = header[49]
            self.initial_tempo = header[50]
            self.master_volume = header[51] & 0x7F
            
            # Channel settings (32 bytes at offset 64)
            self.channel_settings = list(header[64:96])
            
            # Count active channels
            self.num_channels = 0
            for ch_setting in self.channel_settings:
                if ch_setting < 16:  # Channel is enabled
                    self.num_channels += 1
            
            # Read orders
            self.orders = list(f.read(self.num_orders))
            # Filter out marker values (254=skip, 255=end)
            self.orders = [o for o in self.orders if o < 254]
            
            # Read parapointers (16-bit offsets in paragraphs of 16 bytes)
            instrument_paras = struct.unpack(f'<{self.num_instruments}H', f.read(self.num_instruments * 2))
            pattern_paras = struct.unpack(f'<{self.num_patterns}H', f.read(self.num_patterns * 2))
            
            # Read instruments
            self.instruments = [None] * self.num_instruments
            for i, para in enumerate(instrument_paras):
                if para == 0:
                    continue
                offset = para * 16
                f.seek(offset)
                inst_data = f.read(80)
                
                inst_type = inst_data[0]
                if inst_type != 1:  # Not a sample
                    continue
                
                # Parse instrument header
                memseg_hi = inst_data[13]
                memseg = struct.unpack('<H', inst_data[14:16])[0]
                sample_offset = (memseg_hi << 16) | memseg
                length = struct.unpack('<I', inst_data[16:20])[0]
                loop_begin = struct.unpack('<I', inst_data[20:24])[0]
                loop_end = struct.unpack('<I', inst_data[24:28])[0]
                volume = inst_data[28]
                flags = inst_data[31]
                c2spd = struct.unpack('<I', inst_data[32:36])[0]
                name = inst_data[48:76].decode('ascii', errors='ignore').rstrip('\x00')
                
                # Read sample data
                sample_data = None
                if length > 0 and sample_offset > 0:
                    f.seek(sample_offset * 16)
                    sample_bytes = f.read(length)
                    # Convert to signed 8-bit
                    sample_data = np.frombuffer(sample_bytes, dtype=np.int8)
                
                loop_length = loop_end - loop_begin if (flags & 1) else 0
                
                self.instruments[i] = {
                    'name': name,
                    'length': length,
                    'repeat_point': loop_begin,
                    'repeat_length': loop_length,
                    'volume': volume,
                    'c2spd': c2spd,
                    'data': sample_data,
                    # Add MOD-compatible fields
                    'finetune': 0,  # S3M doesn't use finetune
                }
            
            # Read patterns
            self.patterns = [None] * self.num_patterns
            for i, para in enumerate(pattern_paras):
                if para == 0:
                    self.patterns[i] = self.create_empty_pattern()
                    continue
                
                offset = para * 16
                f.seek(offset)
                packed_length = struct.unpack('<H', f.read(2))[0]
                packed_data = f.read(packed_length)
                
                # Unpack pattern
                self.patterns[i] = self.unpack_pattern(packed_data)
    
    def create_empty_pattern(self, rows=64):
        """Create an empty pattern"""
        pattern = []
        for row in range(rows):
            pattern.append([{
                'note': 255,  # No note
                'instrument': 0,
                'volume': 255,  # No volume
                'command': 0,
                'info': 0
            } for _ in range(32)])  # 32 channels max
        return pattern
    
    def unpack_pattern(self, packed_data):
        """Unpack S3M packed pattern format"""
        pattern = self.create_empty_pattern()
        pos = 0
        row = 0
        
        while pos < len(packed_data) and row < 64:
            if packed_data[pos] == 0:
                # End of row
                row += 1
                pos += 1
                continue
            
            what = packed_data[pos]
            channel = what & 31
            pos += 1
            
            if what & 32:  # Note and instrument follow
                note = packed_data[pos]
                instrument = packed_data[pos + 1]
                pattern[row][channel]['note'] = note
                pattern[row][channel]['instrument'] = instrument
                pos += 2
            
            if what & 64:  # Volume follows
                volume = packed_data[pos]
                pattern[row][channel]['volume'] = volume
                pos += 1
            
            if what & 128:  # Command and info follow
                command = packed_data[pos]
                info = packed_data[pos + 1]
                pattern[row][channel]['command'] = command
                pattern[row][channel]['info'] = info
                pos += 2
        
        return pattern

def detect_module_format(filename):
    """Detect if file is MOD or S3M"""
    with open(filename, 'rb') as f:
        # Check for S3M signature at offset 44
        f.seek(44)
        sig = f.read(4)
        if sig == b'SCRM':
            return 'S3M'
        
        # Check for MOD signature at offset 1080
        f.seek(1080)
        sig = f.read(4)
        if sig in [b'M.K.', b'M!K!', b'FLT4', b'FLT8', b'4CHN', b'6CHN', b'8CHN']:
            return 'MOD'
        
        return 'UNKNOWN'

def create_fixed_player_html(mod, output_file, downsample=1, compress=False):
    """Create HTML with PROPERLY TIMED MOD player"""
    
    # Pack samples with bandwidth-adaptive compression
    sample_map = []
    all_samples = []
    individual_samples = []
    current_pos = 0
    
    for i, sample in enumerate(mod.samples):
        if sample['length'] > 0:
            # Bandwidth-adaptive compression: decimate by power-of-2 factor
            bf, compressed = bw_compress_sample(sample['data'])
            data_float = np.concatenate([
                compressed.astype(np.float32) / 128.0,
                np.zeros(32, dtype=np.float32)   # zero-padding for cubic interpolation
            ])
            
            sample_map.append({
                'index':       i,
                'start':       current_pos,
                'length':      len(compressed),  # compressed length (not including padding)
                'loop_start':  sample['repeat_point'] // bf,
                'loop_length': sample['repeat_length'] // bf if sample['repeat_length'] > 2 else sample['repeat_length'],
                'bw_factor':   bf,
                'volume':      sample['volume'],
                'finetune':    sample['finetune'],
                'name':        sample['name']
            })
            
            all_samples.extend(data_float.tolist())
            individual_samples.append(compressed.tolist())
            current_pos += len(data_float)
        else:
            sample_map.append({
                'index': i, 'start': 0, 'length': 0,
                'loop_start': 0, 'loop_length': 0, 'bw_factor': 1,
                'volume': 0, 'finetune': 0, 'name': ''
            })
            individual_samples.append([])
    
    # Encode patterns
    # S3M note to MOD period conversion (same as in main())
    s3m_note_to_period = [
        1712, 1616, 1525, 1440, 1357, 1281, 1209, 1141, 1077, 1017, 961, 907,
        856, 808, 762, 720, 678, 640, 604, 570, 538, 508, 480, 453,
        428, 404, 381, 360, 339, 320, 302, 285, 269, 254, 240, 226,
        214, 202, 190, 180, 170, 160, 151, 143, 135, 127, 120, 113,
        107, 101, 95, 90, 85, 80, 76, 71, 67, 64, 60, 57,
    ]
    
    def s3m_note_to_mod_period(note):
        if note == 255 or note == 254:
            return 0
        if note >= len(s3m_note_to_period):
            return 0
        return s3m_note_to_period[note]
    
    pattern_data = []
    is_s3m = hasattr(mod, 'is_s3m') and mod.is_s3m
    num_channels = mod.num_channels if hasattr(mod, 'num_channels') else 4
    
    for pattern in mod.patterns:
        for row in pattern:
            for ch_idx in range(num_channels):
                ch = row[ch_idx] if ch_idx < len(row) else {}
                
                # Convert S3M to MOD format if needed
                if is_s3m:
                    sample = ch.get('instrument', 0)
                    period = s3m_note_to_mod_period(ch.get('note', 255))
                    effect = ch.get('command', 0)
                    param = ch.get('info', 0)
                else:
                    sample = ch.get('sample', 0)
                    period = ch.get('period', 0)
                    effect = ch.get('effect', 0)
                    param = ch.get('param', 0)
                
                pattern_data.append({
                    'sample': sample,
                    'period': period,
                    'effect': effect,
                    'param': param
                })
    
    # Optional compression
    if compress:
        print("   Compressing patterns...")
        # Flatten to bytes for RLE
        pattern_bytes = []
        for note in pattern_data:
            pattern_bytes.extend([
                note['sample'],
                note['period'] >> 8,
                note['period'] & 0xFF,
                (note['effect'] << 4) | (note['param'] >> 4),
                note['param'] & 0x0F
            ])
        
        compressed_patterns = compress_patterns_rle(pattern_bytes)
        print(f"      {len(pattern_bytes)} → {len(compressed_patterns)} bytes ({100*len(compressed_patterns)/len(pattern_bytes):.1f}%)")
        
        # Samples stay uncompressed (fast loading!)
        compressed_samples = None
    else:
        compressed_patterns = None
        compressed_samples = None
    
    # Build decompression code if needed
    if compress:
        decompress_code = """
// Decompress RLE patterns
function decompressPatterns(compressed) {
    const decompressed = [];
    let i = 0;
    while (i < compressed.length) {
        if (compressed[i] === 255) {
            const runLength = compressed[i + 1];
            const value = compressed[i + 2];
            for (let j = 0; j < runLength; j++) {
                decompressed.push(value);
            }
            i += 3;
        } else {
            decompressed.push(compressed[i]);
            i++;
        }
    }
    
    console.log('Decompressed bytes:', decompressed.length, 'expected:', modData.numPatterns * 64 * 4 * 5);
    
    // Reconstruct pattern objects
    const patterns = [];
    let offset = 0;
    const totalNotes = modData.numPatterns * 64 * 4;
    
    for (let n = 0; n < totalNotes && offset + 4 < decompressed.length; n++) {
        patterns.push({
            sample: decompressed[offset++] || 0,
            period: ((decompressed[offset++] || 0) << 8) | (decompressed[offset++] || 0),
            effect: (decompressed[offset] || 0) >> 4,
            param: (((decompressed[offset++] || 0) & 0x0F) << 4) | (decompressed[offset++] || 0)
        });
    }
    
    console.log('Reconstructed patterns:', patterns.length, 'expected:', totalNotes);
    return patterns;
}

// Apply decompression if needed
if (modData.compressed) {
    console.log('🗜️ Decompressing patterns...');
    modData.patterns = decompressPatterns(modData.compressedPatterns);
    delete modData.compressedPatterns;
    console.log('✅ Decompression complete!');
}
console.log('MOD Player Ready!');
"""
    else:
        decompress_code = ""
    
    # Build modData fields conditionally
    # HTML player always embeds samples (no size limit)
    # ShaderToy limits only apply to .glsl output
    
    # Split samples into chunks for clean code organization
    sample_chunks = []
    for i in range(0, len(all_samples), CHUNK_SIZE):
        sample_chunks.append(all_samples[i:i+CHUNK_SIZE])
    
    # Generate chunk declarations
    chunk_decls = "\n".join([
        f"const sampleChunk{i} = {json.dumps(chunk)};"
        for i, chunk in enumerate(sample_chunks)
    ])
    
    # Generate concatenation
    chunk_concat = "[..." + ", ...".join([f"sampleChunk{i}" for i in range(len(sample_chunks))]) + "]"
    
    print(f"   📦 Sample data split into {len(sample_chunks)} chunks ({len(all_samples)} total samples)")
    
    if compress:
        data_fields = f"""compressedPatterns: {json.dumps(compressed_patterns)},
    sampleMap: {json.dumps(sample_map)},
    compressed: true"""
    else:
        data_fields = f"""patterns: {json.dumps(pattern_data)},
    sampleMap: {json.dumps(sample_map)},
    compressed: false"""
    
    num_channels = mod.num_channels if hasattr(mod, 'num_channels') else 4
    _ch_panels = '\n'.join(
        f'  <div class="ch-panel ch{i}">\n'
        f'    <div class="ch-header">Track #{i+1}</div>\n'
        f'    <div class="ch-note" id="chNote{i}">---</div>\n'
        f'    <div class="ch-bar-wrap"><div class="ch-bar" id="chBar{i}"></div></div>\n'
        f'  </div>'
        for i in range(num_channels)
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{mod.title} — GLSL MOD Player</title>
<style>
:root {{
  --bg0:#0d0d12;--bg1:#13131a;--bg2:#1c1c26;--bg3:#252533;
  --accent:#3d8ef0;--accent2:#5af0c8;--text:#c8ccd8;--dim:#555870;--border:#2a2a3c;
  --font:'Courier New',monospace;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg0);color:var(--text);font-family:var(--font);display:flex;flex-direction:column;align-items:center;min-height:100vh}}
#topbar{{width:100%;background:var(--bg1);border-bottom:1px solid var(--border);display:flex;align-items:center;gap:16px;padding:10px 20px}}
.logo{{color:var(--accent);font-size:13px;letter-spacing:2px;font-weight:bold;white-space:nowrap}}
.ttl{{color:var(--accent2);font-size:14px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.spc{{flex:1}}
.meta{{color:var(--dim);font-size:11px;white-space:nowrap}}
#canvasWrap{{width:100%;background:var(--bg0);border-bottom:1px solid var(--border)}}
#oscCanvas{{width:100%;display:block;height:120px}}
#progressWrap{{width:100%;height:5px;background:var(--bg3);cursor:pointer}}
#progressBar{{height:100%;background:var(--accent);width:0%;position:relative}}
#progressBar::after{{content:'';position:absolute;right:-5px;top:-3px;width:10px;height:10px;border-radius:50%;background:var(--accent)}}
#controls{{width:100%;background:var(--bg1);border-bottom:1px solid var(--border);display:flex;align-items:center;gap:6px;padding:8px 16px;flex-wrap:wrap}}
.ctrl-btn{{background:var(--bg2);border:1px solid var(--border);color:var(--text);font-family:var(--font);font-size:13px;padding:6px 14px;cursor:pointer;border-radius:3px;transition:all .15s;letter-spacing:1px}}
.ctrl-btn:hover:not(:disabled){{background:var(--accent);color:#fff;border-color:var(--accent)}}
.ctrl-btn:disabled{{opacity:.3;cursor:not-allowed}}
.sep{{width:1px;height:28px;background:var(--border);margin:0 4px}}
#volWrap{{display:flex;align-items:center;gap:8px;color:var(--dim);font-size:12px}}
#volSlider{{-webkit-appearance:none;width:80px;height:4px;border-radius:2px;background:var(--bg3);outline:none;cursor:pointer}}
#volSlider::-webkit-slider-thumb{{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:var(--accent);cursor:pointer}}
#timeDisplay{{margin-left:auto;color:var(--dim);font-size:12px;letter-spacing:1px;white-space:nowrap}}
#infoGrid{{width:100%;display:grid;grid-template-columns:repeat(4,1fr);border-bottom:1px solid var(--border)}}
.info-cell{{padding:10px 16px;border-right:1px solid var(--border);background:var(--bg1)}}
.info-cell:last-child{{border-right:none}}
.info-label{{font-size:10px;color:var(--dim);letter-spacing:2px;text-transform:uppercase;margin-bottom:4px}}
.info-value{{font-size:22px;color:var(--accent2);letter-spacing:1px;font-weight:bold}}
.info-value .sub{{color:var(--dim);font-size:13px;font-weight:normal}}
#channels{{width:100%;display:grid;grid-template-columns:repeat({num_channels},1fr);gap:1px;background:var(--border);border-bottom:1px solid var(--border)}}
.ch-panel{{background:var(--bg1);padding:10px 14px}}
.ch-header{{font-size:10px;letter-spacing:2px;color:var(--dim);margin-bottom:6px;text-transform:uppercase}}
.ch-note{{font-size:16px;font-weight:bold;margin-bottom:6px;letter-spacing:1px;height:22px}}
.ch-bar-wrap{{height:4px;background:var(--bg3);border-radius:2px;overflow:hidden}}
.ch-bar{{height:100%;width:0%;border-radius:2px;transition:width .08s ease-out}}
.ch0 .ch-note{{color:#3df0f0}}.ch0 .ch-bar{{background:#3df0f0}}
.ch1 .ch-note{{color:#f0d040}}.ch1 .ch-bar{{background:#f0d040}}
.ch2 .ch-note{{color:#f040d0}}.ch2 .ch-bar{{background:#f040d0}}
.ch3 .ch-note{{color:#f08030}}.ch3 .ch-bar{{background:#f08030}}
#footer{{width:100%;padding:10px 20px;color:var(--dim);font-size:11px;display:flex;justify-content:space-between;border-top:1px solid var(--border);margin-top:auto}}

#tracker {{
  width:100%; background:var(--bg0); border-bottom:1px solid var(--border);
  overflow:hidden; font-size:13px;
}}
.trk-header {{
  display:flex; background:var(--bg2); border-bottom:1px solid var(--border); padding:3px 0;
}}
.trk-col-hdr {{
  font-size:10px; letter-spacing:2px; color:var(--dim); text-transform:uppercase;
  padding:2px 8px; flex:1;
}}
.trk-col-hdr:first-child {{ flex:0 0 44px; }}
.trk-row {{ display:flex; border-bottom:1px solid #0f0f18; }}
.trk-row.current {{
  background:rgba(255,255,255,0.07) !important; border-left:3px solid var(--accent);
}}
.trk-row:nth-child(even) {{ background:#0d0d14; }}
.trk-row:nth-child(odd)  {{ background:#101018; }}
.trk-rownum {{
  flex:0 0 44px; color:var(--dim); font-size:11px;
  padding:3px 8px; align-self:center;
}}
.trk-row.current .trk-rownum {{ color:var(--accent); font-weight:bold; }}
.trk-cell {{ flex:1; padding:3px 8px; font-family:var(--font); white-space:nowrap; }}
.trk-empty {{ color:#252535; }}
.ch0-color {{ color:#3df0f0; }}
.ch1-color {{ color:#f0d040; }}
.ch2-color {{ color:#f040d0; }}
.ch3-color {{ color:#f08030; }}
.trk-samp {{ color:#6677aa; font-size:11px; }}
.trk-eff  {{ color:#557799; font-size:11px; }}
</style>
</head>
<body>
<div id="topbar">
  <span class="logo">## GLSL MOD PLAYER ##</span>
  <span class="ttl">{mod.title}</span><span class="spc"></span>
  <span class="meta">BPM {mod.initial_tempo} &middot; {mod.num_channels}ch &middot; {len(mod.song_positions)} patterns</span>
</div>
<div id="canvasWrap"><canvas id="oscCanvas"></canvas></div>
<div id="progressWrap"><div id="progressBar"></div></div>
<div id="controls">
  <button class="ctrl-btn" id="playBtn">&#9654; PLAY</button>
  <button class="ctrl-btn" id="pauseBtn" disabled>&#9646;&#9646; PAUSE</button>
  <button class="ctrl-btn" id="stopBtn" disabled>&#9632; STOP</button>
  <div class="sep"></div>
  <div id="volWrap"><span>VOL</span><input type="range" id="volSlider" min="0" max="1" step="0.01" value="0.8"></div>
  <div id="volWrap"><span>3D</span><input type="range" id="surroundSlider" min="0" max="2" step="0.05" value="0.5" title="Only3D surround depth"></div>
  <div id="volWrap"><button id="surroundModeBtn" title="Toggle: total mix vs outer channels only" style="font-size:11px;padding:2px 6px">3D:MIX</button></div>
  <div id="volWrap"><span>PHAT</span><input type="range" id="phatSlider" min="0" max="1.5" step="0.05" value="0.5" title="Phat bass (Hilbert allpass)"></div>
  <div id="timeDisplay">00:00 / 00:00</div>
</div>
<div id="infoGrid">
  <div class="info-cell"><div class="info-label">Pattern</div>
    <div class="info-value"><span id="patternInfo">00</span><span class="sub"> / {mod.num_patterns-1:02d}</span></div></div>
  <div class="info-cell"><div class="info-label">Row</div>
    <div class="info-value"><span id="rowInfo">00</span><span class="sub"> / 63</span></div></div>
  <div class="info-cell"><div class="info-label">BPM</div>
    <div class="info-value" id="bpmInfo">{mod.initial_tempo}</div></div>
  <div class="info-cell"><div class="info-label">Speed</div>
    <div class="info-value" id="speedInfo">{mod.initial_speed}</div></div>
</div>
<div id="channels">{_ch_panels}</div>
<div id="tracker">
  <div class="trk-header" id="trkHeader"></div>
  <div id="trkBody"></div>
</div>
<div id="footer">
  <span>&#169; 2026 Orblivius &middot; subband@gmail.com &middot; github.com/mewza</span>
  <span id="statusMsg">Ready</span>
</div>


    <script>
// Sample chunks (ShaderToy-compatible splitting)
{chunk_decls}

const modData = {{
    title: {json.dumps(mod.title)},
    songLength: {len(mod.song_positions)},
    songPositions: {json.dumps(mod.song_positions)},
    numPatterns: {mod.num_patterns},
    initialBPM: {mod.initial_tempo},
    initialSpeed: {mod.initial_speed},
    bassSamples: {json.dumps([i+1 for i,s in enumerate(mod.samples) if s['length']>0 and 'bass' in s['name'].lower()])},
    {data_fields},
    samples: {chunk_concat},
    downsample: {downsample}
}};

{decompress_code}

// ── PhatBass — Hilbert allpass pair for bass enhancement ───────────────────────
// Direct port of PHASESHIFT0 / PHASESHIFT90 from mss (c) Dmitry Boldyrev
// Usage: PHASESHIFT90 → L,  PHASESHIFT0 → R  (separate state arrays, same input)
// The 3-stage allpass shifts bass ~82° at 100Hz.  Adding the phase-rotated
// signal to the dry mix gives +4 to +6 dB bass enhancement below 100Hz
// and natural mid-roll-off — creating the "phat" bass sensation.
// PHASESHIFT90 returns the PREVIOUS output (one-sample sync delay).
class PhatBass {{
    constructor() {{
        this.coL = new Float64Array(11);  // state for L (PHASESHIFT90)
        this.coR = new Float64Array(11);  // state for R (PHASESHIFT0)
        this.depth = 0.5;
    }}
    reset() {{ this.coL.fill(0); this.coR.fill(0); }}
    // PHASESHIFT0: current allpass output
    _ps0(inp, co) {{
        const s0 = (co[6]-inp)  * 0.232829 + co[4];
        const s1 = (co[8]-s0)   * 0.843573 + co[6];
        const s2 = (co[10]-s1)  * 0.980351 + co[8];
        co[10]=co[9]; co[9]=s2; co[8]=co[7]; co[7]=s1;
        co[6]=co[5];  co[5]=s0; co[4]=co[3]; co[3]=inp;
        return s2;
    }}
    // PHASESHIFT90: previous allpass output (sync-delayed 1 sample)
    _ps90(inp, co) {{
        const s0 = (co[6]-inp)  * 0.232829 + co[4];
        const s1 = (co[8]-s0)   * 0.843573 + co[6];
        const s2 = (co[10]-s1)  * 0.980351 + co[8];
        const out = co[9];                         // previous s2
        co[10]=co[9]; co[9]=s2; co[8]=co[7]; co[7]=s1;
        co[6]=co[5];  co[5]=s0; co[4]=co[3]; co[3]=inp;
        return out;
    }}
    // Process a bass-channel sample: add phase-shifted version to L and R outputs
    // bassIn = mono bass signal, outL/outR = current mix outputs
    // Returns [new_outL, new_outR]
    process(bassIn, outL, outR) {{
        const d = this.depth;
        if (d <= 0) return [outL, outR];
        const ps90 = this._ps90(bassIn, this.coL);  // PHASESHIFT90 → L
        const ps0  = this._ps0 (bassIn, this.coR);  // PHASESHIFT0  → R
        return [outL + ps90 * d, outR + ps0 * d];
    }}
}}

// ── Only3D — stereo surround widener ──────────────────────────────────────────
// Ported from Only3D.h (c) Dmitry Boldyrev / mss
// Algorithm: extract stereo difference → 6th-order IIR lowpass (flt6_44) →
//            two first-order allpass filters at different frequencies →
//            soft-saturate → cross-blend into L/R for 3D depth
class Only3D {{
    constructor(sampleRate) {{
        this.sr = sampleRate;
        // Allpass coefficients — two frequencies for genuine dd1 ≠ dd2
        // Original uses 500Hz at 4× oversampled rate; we use 1× equivalents
        const ap = (f) => {{
            const d = Math.tan(f * Math.PI / sampleRate);
            const s = Math.sin(d), c = Math.cos(d), sc = s + c;
            return {{ p0: s/sc, p1: s/sc, p2: (c-s)/sc }};
        }};
        const a1 = ap(500), a2 = ap(2500);
        this.p0_1=a1.p0; this.p1_1=a1.p1; this.p2_1=a1.p2;
        this.p0_2=a2.p0; this.p1_2=a2.p1; this.p2_2=a2.p2;
        // Allpass state
        this.xx1=0; this.yy1=0;
        this.xx2=0; this.yy2=0;
        // flt6_44 ring buffers (xv=feedforward, yv=feedback)
        this.xv = new Float64Array(7);
        this.yv = new Float64Array(7);
        this.fi = 0;
        // DC blocker (fltefx equivalent)
        this.dcx=0; this.dcy=0;
        this.depth = 1.0;
    }}
    // 6th-order Butterworth lowpass — direct port of flt6_44 from Filter_Original
    flt6(inp) {{
        const i = this.fi = (this.fi + 1) % 7;
        const xv=this.xv, yv=this.yv;
        const x = n => xv[(i-n+70)%7];
        const y = n => yv[(i-n+70)%7];
        xv[i] = inp * (1.0/3526.975418);
        yv[i] = (x(0)+x(6)) + 6*(x(1)+x(5)) + 15*(x(2)+x(4)) + 20*x(3)
              + (-0.0916957868*y(6)) + (0.7643814944*y(5))
              + (-2.7105761157*y(4)) + (5.2526413293*y(3))
              + (-5.8968166830*y(2)) + (3.6639199024*y(1));
        return yv[i];
    }}
    // DC blocker — simplified fltefx_44 (removes DC drift from difference signal)
    dcBlock(x) {{
        const y = x - this.dcx + 0.9977*this.dcy;
        this.dcx=x; this.dcy=y;
        return y;
    }}
    // Soft saturation: x/√(1 + x²·0.5)  — direct port from Only3D (saturation=0.5)
    sat(x) {{ return x / Math.sqrt(1.0 + x*x*0.5); }}
    
    process(L, R) {{
        if (this.depth <= 0) return [L, R];
        // Extract stereo difference (side signal)
        const diff = (L - R) * 0.5;
        // Filter: DC block then 6th-order lowpass (focuses on mid-range content)
        const w = this.flt6(this.dcBlock(diff));
        // Allpass 1 at 500Hz → dd1
        let dd1 = w*this.p0_1 + this.p1_1*this.xx1 + this.p2_1*this.yy1;
        this.xx1 = w;
        this.yy1 = dd1 = this.sat(dd1);
        // Allpass 2 at 2500Hz → dd2  (different phase response → dd1 ≠ dd2)
        let dd2 = w*this.p0_2 + this.p1_2*this.xx2 + this.p2_2*this.yy2;
        this.xx2 = w;
        this.yy2 = dd2 = this.sat(dd2);
        // Cross-blend: (dd1-dd2) is a bandpass-shaped surround signal
        const surround = (dd1 - dd2) * this.depth;
        return [L + surround, R - surround];
    }}
}}

class MODPlayer {{
    constructor() {{
        this.audioCtx = null;
        this.isPlaying = false;
        this.bpm   = Math.max(32, modData.initialBPM   || 125);  // mikIT: bpm min=32
        this.speed = Math.min(32, modData.initialSpeed || 6);    // mikIT: speed max=32
        this.sampleRate = 44100;
        
        // CRITICAL: ProTracker timing
        // CIA tempo: ticks_per_second = (BPM * 2) / 5
        this.updateTiming();
        
        // Channel state (persistent across ticks)
        this.numChannels = {num_channels};
        this._volume = 0.8;
        this._only3dDepth = 0.5;
        this._only3d = null;
        this._only3dMode = 0;  // 0=total output, 1=outer channels only (ch0+ch3)
        this._phatBassDepth = 0.5;
        this._phatBass = new PhatBass();
        this._phatBass.depth = this._phatBassDepth;
        this.channels = [];
        this.dyingChannels = [];  // OPTIMIZATION 2: Dying channels for microclick removal
        this.crossfadeSamples = 0;  // Disabled: bass/percussive samples start at zero, no click risk
        this.volumeRampSamples = 0;  // DISABLED: Testing if this causes sand sound
        
        for (let i = 0; i < this.numChannels; i++) {{
            this.channels.push({{
                sample: 0, period: 0, basePeriod: 0, samplePos: 0.0, 
                volume: 64, active: false,
                effect: 0, effectParam: 0, targetPeriod: 0, vibratoPos: 0, vibratoSpeed: 1, vibratoDepth: 0, arpeggioCounter: 0,
                volumeFade: 1.0, volumeFadeInc: 0, targetVolume: 1.0,
                currentVolume: 64, targetVolume2: 64, volumeRampInc: 0, volumeRamping: false,
                mixVol: 0, volInc: 0,
                loopbackPoint: 0, loopCount: 0, _delayedNote: null
            }});
            
            // Dying channel for crossfade (matches C++ dying[] array)
            this.dyingChannels.push({{
                sample: 0, period: 0, samplePos: 0.0, volume: 64, active: false,
                volumeFade: 0, volumeFadeInc: 0, samplesLeft: 0
            }});
        }}
        
        // Song position
        this.currentPattern = 0;
        this.currentRow = 0;
        this.currentTick = 0;
        this._patBreakPending = false; this._patBreakRow = 0;
        this._jumpPending = false;    this._jumpTarget = 0;
        this._patLoopPending = false; this._patLoopRow = 0;
        this._patDelayTicks = 0;      this._patDelayActive = false;
        this.sampleCounter = 0;
    }}
    
    updateTiming() {{
        // CORRECT ProTracker timing: CIA timer = (BPM * 2) / 5
        this.ticksPerSecond = (this.bpm * 2.0) / 5.0;
        this.samplesPerTick = Math.floor(this.sampleRate / this.ticksPerSecond);
        this.samplesPerRow = this.samplesPerTick * this.speed;
        
        this.log(`Timing: ${{this.ticksPerSecond.toFixed(2)}} Hz (BPM ${{this.bpm}}), ${{this.samplesPerTick}} samples/tick`);
    }}
    
    log(msg) {{ console.log('[MOD]', msg); }}
    
    init() {{
        this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        this.sampleRate = this.audioCtx.sampleRate;
        this._only3d = new Only3D(this.sampleRate);
        this._only3d.depth = this._only3dDepth;
        this.nextPlayTime = this.audioCtx.currentTime; // Track next buffer start time
        this.updateTiming();
        this.log('Audio initialized: ' + this.sampleRate + ' Hz');
    }}
    
    // Finetune table from mikIT (C4 playback speed for each of 16 finetune values)
    // finetune nibble 0-7 = positive (higher pitch), 8-15 = negative (lower pitch)
    periodToFreq(period, finetune) {{
        if (period === 0) return 0;
        const c4speeds = [8363,8413,8463,8529,8581,8651,8723,8757,
                          7895,7941,7985,8046,8107,8169,8232,8280];
        const c4 = c4speeds[(finetune || 0) & 0xF];
        return (c4 * 428) / period;  // 428 = period for middle C (C-3) at standard tuning
    }}
    
    getSampleData(sampleIdx, position) {{
        const info = modData.sampleMap[sampleIdx];
        if (!info || info.length === 0) return 0;
        
        // Map original sample position → compressed position via bw_factor
        const bf = info.bw_factor || 1;
        let pos = position / bf;
        
        // Loop wrapping (loop points already divided by bw_factor at generation time)
        if (info.loop_length > 2) {{
            if (pos >= info.loop_start + info.loop_length) {{
                pos = info.loop_start + (pos - info.loop_start) % info.loop_length;
            }}
        }} else if (pos >= info.length) {{
            return 0;
        }}
        
        // 4-point cubic Hermite (Catmull-Rom) interpolation.
        const pos0 = pos | 0;
        const frac = pos - pos0;
        const base = info.start;
        
        const looping = info.loop_length > 2;
        const wrap = (i) => {{
            if (looping && i < info.loop_start) return info.loop_start + info.loop_length - 1;
            return Math.max(0, i);
        }};
        const p0 = modData.samples[base + wrap(pos0 - 1)];
        const p1 = modData.samples[base + pos0];
        const p2 = modData.samples[base + pos0 + 1];
        const p3 = modData.samples[base + pos0 + 2];
        
        const a = -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3;
        const b =      p0 - 2.5*p1 + 2.0*p2 - 0.5*p3;
        const c = -0.5*p0           + 0.5*p2;
        return ((a * frac + b) * frac + c) * frac + p1;
    }}
    
    getNote(pattern, row, channel) {{
        const idx = (pattern * 64 * 4) + (row * 4) + channel;
        return modData.patterns[idx] || {{ sample: 0, period: 0, effect: 0, param: 0 }};
    }}
    
    processTick() {{
        const patternIdx = modData.songPositions[this.currentPattern];
        
        // On tick 0, trigger new notes
        if (this.currentTick === 0) {{
            for (let ch = 0; ch < 4; ch++) {{
                const note = this.getNote(patternIdx, this.currentRow, ch);
                
                // Handle new sample trigger
                if (note.sample > 0) {{
                    const state = this.channels[ch];
                    this.dyingChannels[ch].active = false;
                    
                    // EDx note delay: stash and trigger x ticks later
                    if (note.effect === 0xE && ((note.param >> 4) & 0xF) === 0xD) {{
                        state._delayedNote = note;
                        // Don't trigger sample now
                    }} else {{
                        const sampleInfo = modData.sampleMap[note.sample - 1];
                        state.sample = note.sample - 1;
                        state.volume = sampleInfo.volume;
                        state.currentVolume = sampleInfo.volume;
                        state.active = true;
                        state.volumeRamping = false;
                        
                        if (note.effect === 0x3) {{
                            // Tone portamento: keep current position, don't retrigger
                        }} else {{
                            state.samplePos = 0.0;
                            state.mixVol = 0;
                            state.volumeFade = 1.0;
                            state.volumeFadeInc = 0;
                            state.targetVolume = 1.0;
                            if (note.effect === 0x9) state.samplePos = note.param * 256;
                        }}
                    }}
                }}
                
                // Handle new period (pitch)
                if (note.period > 0) {{
                    if (note.effect === 0x3) {{
                        // Tone portamento: period = TARGET to slide toward, NOT immediate pitch change
                        this.channels[ch].targetPeriod = note.period;
                        // Don't touch state.period or basePeriod
                    }} else {{
                        this.channels[ch].period = note.period;
                        this.channels[ch].basePeriod = note.period;
                        this.channels[ch].targetPeriod = note.period;
                        this.channels[ch].vibratoPos = 0;
                        this.channels[ch].arpeggioCounter = 0;
                        
                        // Period-only (no new sample): retrigger sample from start
                        if (note.sample === 0 && this.channels[ch].active) {{
                            const state = this.channels[ch];
                            state.samplePos = 0.0;
                            state.mixVol = 0;
                        }}
                    }}
                }}
                
                // Store effect for processing on all ticks
                this.channels[ch].effect = note.effect;
                this.channels[ch].effectParam = note.param;
                
                // Process tick-0 effects
                this.processEffect(ch, note.effect, note.param, true);
            }}
        }} else {{
            // Process effects on non-zero ticks
            for (let ch = 0; ch < 4; ch++) {{
                this.processEffect(ch, this.channels[ch].effect, this.channels[ch].effectParam, false);
            }}
        }}
        
        // Advance tick — honour Effect B (jump), D (pattern break), E6x (loop), EEx (delay)
        this.currentTick++;
        if (this._patDelayActive && this._patDelayTicks > 0) {{
            this._patDelayTicks--;
            if (this._patDelayTicks <= 0) this._patDelayActive = false;
        }} else if (this.currentTick >= this.speed) {{
            this.currentTick = 0;
            if (this._patLoopPending) {{
                this._patLoopPending = false;
                this.currentRow = this._patLoopRow;
            }} else if (this._patBreakPending) {{
                this._patBreakPending = false;
                this.currentRow = this._patBreakRow || 0;
                this.currentPattern++;
                if (this.currentPattern >= modData.songLength) {{ this.stop(); }}
            }} else if (this._jumpPending) {{
                this._jumpPending = false;
                this.currentRow = 0;
                this.currentPattern = this._jumpTarget;
                if (this.currentPattern >= modData.songLength) {{ this.stop(); }}
            }} else {{
                this.currentRow++;
                if (this.currentRow >= 64) {{
                    this.currentRow = 0;
                    this.currentPattern++;
                    if (this.currentPattern >= modData.songLength) {{ this.stop(); }}
                }}
            }}
        }}
        
        // mikIT volinc: smooth volume transitions over ~64 samples to prevent clicks
        // Ramp mixVol toward currentVolume once per tick (not once per note)
        const FADE = (this.sampleRate / 689) | 0;  // ≈ 64 samples
        for (let ch = 0; ch < this.numChannels; ch++) {{
            const s = this.channels[ch];
            s.volInc = (s.currentVolume - s.mixVol) / FADE;
        }}
    }}
    
    processEffect(ch, effect, param, tick0) {{
        const state = this.channels[ch];
        
        switch(effect) {{
            case 0x0: // Arpeggio
                if (param !== 0) {{
                    // Just increment counter - period modification happens during playback
                    state.arpeggioCounter++;
                }}
                break;
                
            case 0x1: // Portamento up
                if (!tick0 && param > 0) {{
                    state.period = Math.max(113, state.period - param);
                    state.basePeriod = state.period;
                }}
                break;
                
            case 0x2: // Portamento down
                if (!tick0 && param > 0) {{
                    state.period = Math.min(856, state.period + param);
                    state.basePeriod = state.period;
                }}
                break;
                
            case 0x3: // Tone portamento
                if (!tick0 && param > 0 && state.targetPeriod > 0) {{
                    if (state.period < state.targetPeriod) {{
                        state.period = Math.min(state.targetPeriod, state.period + param);
                    }} else if (state.period > state.targetPeriod) {{
                        state.period = Math.max(state.targetPeriod, state.period - param);
                    }}
                    state.basePeriod = state.period;
                }}
                break;
                
            case 0x4: // Vibrato
            case 0x6: // VolSlide + Vibrato
                if (tick0) {{
                    // Persist speed & depth — only overwrite when non-zero (param=0 = "continue")
                    const vSpeedNew = (param >> 4) & 0x0F;
                    const vDepthNew = param & 0x0F;
                    if (vSpeedNew > 0) state.vibratoSpeed = vSpeedNew;
                    if (vDepthNew > 0) state.vibratoDepth = vDepthNew;
                }} else {{
                    // Advance vibrato LFO on ticks 1+
                    const vspeed = state.vibratoSpeed || 1;
                    state.vibratoPos = (state.vibratoPos + vspeed) % 64;
                    if (effect === 0x6) {{
                        // Vol slide part — use current row's param directly
                        const vup   = (param >> 4) & 0x0F;
                        const vdown = param & 0x0F;
                        if (vup > 0) {{
                            state.volume = Math.min(64, state.volume + vup);
                        }} else if (vdown > 0) {{
                            state.volume = Math.max(0, state.volume - vdown);
                        }}
                        state.currentVolume = state.volume;
                    }}
                }}
                break;
                
            case 0xA: // Volume slide — instant update (no ramp)
                if (!tick0) {{
                    const up = (param >> 4) & 0x0F;
                    const down = param & 0x0F;
                    if (up > 0) {{
                        const newVol = Math.min(64, state.volume + up);
                        state.volume = newVol;
                        state.currentVolume = newVol;
                        state.targetVolume2 = newVol;
                        state.volumeRamping = false;
                    }} else if (down > 0) {{
                        const newVol = Math.max(0, state.volume - down);
                        state.volume = newVol;
                        state.currentVolume = newVol;
                        state.targetVolume2 = newVol;
                        state.volumeRamping = false;
                    }}
                }}
                break;
                
            case 0xC: // Set volume — instant update
                if (tick0) {{
                    const newVol = Math.min(64, param);
                    state.volume = newVol;
                    state.currentVolume = newVol;
                    state.targetVolume2 = newVol;
                    state.volumeRamping = false;
                }}
                break;
                
            case 0xB: // Position Jump
                if (tick0) {{
                    this._jumpPending = true;
                    this._jumpTarget = Math.min(param, modData.songLength - 1);
                }}
                break;

            case 0xD: // Pattern Break — jump to row xx of next pattern (param is BCD)
                if (tick0) {{
                    this._patBreakPending = true;
                    this._patBreakRow = ((param >> 4) & 0xF) * 10 + (param & 0xF);
                }}
                break;

            case 0xF: // Set speed/tempo
                if (tick0) {{
                    if (param < 0x20) {{
                        this.speed = param;
                        this.updateTiming();
                    }} else {{
                        this.bpm = param;
                        this.updateTiming();
                    }}
                }}
                break;

            case 0xE: // Extended effects (Exy — high nibble=sub-command, low nibble=value)
                {{
                    const sub = (param >> 4) & 0xF;
                    const val = param & 0xF;
                    switch (sub) {{
                        case 0x6: // E6x — Pattern Loop
                            if (tick0) {{
                                if (val === 0) {{
                                    // Set loop start
                                    state.loopbackPoint = this.currentRow;
                                }} else {{
                                    // Loop val times
                                    if (!state.loopCount) state.loopCount = val;
                                    if (state.loopCount > 0) {{
                                        state.loopCount--;
                                        if (state.loopCount > 0) {{
                                            this._patLoopRow = state.loopbackPoint || 0;
                                            this._patLoopPending = true;
                                        }} else {{
                                            state.loopCount = 0;
                                        }}
                                    }}
                                }}
                            }}
                            break;
                        case 0x9: // E9x — Retrigger note every x ticks
                            if (!tick0 && val > 0 && (this.currentTick % val) === 0) {{
                                state.samplePos = 0.0;
                                state.mixVol = 0;
                            }}
                            break;
                        case 0xC: // ECx — Note cut after x ticks
                            if (!tick0 && this.currentTick === val) {{
                                state.currentVolume = 0;
                                state.volume = 0;
                            }}
                            break;
                        case 0xD: // EDx — Note delay: trigger note x ticks late
                            if (!tick0 && this.currentTick === val && state._delayedNote) {{
                                const dn = state._delayedNote;
                                state._delayedNote = null;
                                const info = modData.sampleMap[dn.sample - 1];
                                state.sample = dn.sample - 1;
                                state.volume = info.volume;
                                state.currentVolume = info.volume;
                                state.active = true;
                                state.samplePos = 0.0;
                                state.mixVol = 0;
                                if (dn.period > 0) {{
                                    state.period = dn.period;
                                    state.basePeriod = dn.period;
                                }}
                            }}
                            break;
                        case 0xE: // EEx — Pattern delay: extra ticks per row
                            if (tick0 && !this._patDelayActive) {{
                                this._patDelayTicks = val * this.speed;
                                this._patDelayActive = true;
                            }}
                            break;
                        default:
                            break;
                    }}
                }}
                break;
        }}
    }}
    
    getArpeggioPeriod(basePeriod, semitones) {{
        // Arpeggio shifts pitch up by N semitones → period = basePeriod / 2^(N/12)
        // Finetune cancels in the ratio, so it's not needed here
        return Math.round(basePeriod / Math.pow(2, semitones / 12));
    }}
    
    getEffectivePeriod(ch) {{
        // ProTracker-authentic 32-entry sine table for vibrato
        // amplitude = (VibratoTable[pos & 31] * depth) >> 7  (mikIT formula)
        const VibTab = [0,24,49,74,97,120,141,161,180,197,212,224,235,244,250,253,
                        255,253,250,244,235,224,212,197,180,161,141,120,97,74,49,24];
        const state = this.channels[ch];
        let effectivePeriod = state.period;
        
        // Apply arpeggio effect (0xy)
        if (state.effect === 0x0 && state.effectParam !== 0) {{
            const note1 = (state.effectParam >> 4) & 0x0F;
            const note2 = state.effectParam & 0x0F;
            const arpStep = state.arpeggioCounter % 3;
            
            if (arpStep === 1) {{
                effectivePeriod = this.getArpeggioPeriod(state.basePeriod, note1);
            }} else if (arpStep === 2) {{
                effectivePeriod = this.getArpeggioPeriod(state.basePeriod, note2);
            }} else {{
                effectivePeriod = state.basePeriod;
            }}
        }}
        
        // Apply vibrato effect (4xy or 6xy — VolSlide+Vibrato)
        if (state.effect === 0x4 || state.effect === 0x6) {{
            const depth = state.vibratoDepth || 0;
            if (depth > 0) {{
                // ProTracker formula: (VibTab[pos & 31] * depth) >> 7
                // Sign: pos 0-31 = positive, 32-63 = negative  (like a full sine cycle)
                const pos = state.vibratoPos & 63;
                const tabVal = VibTab[pos & 31];
                const vibDelta = (tabVal * depth) >> 7;
                effectivePeriod += (pos < 32) ? vibDelta : -vibDelta;
            }}
        }}
        
        return Math.max(113, Math.min(856, effectivePeriod));
    }}
    
    generateSamples(buffer, offset, count) {{
        const leftChannel = buffer.getChannelData(0);
        const rightChannel = buffer.getChannelData(1);
        
        for (let i = 0; i < count; i++) {{
            // Process tick boundary
            if (this.sampleCounter === 0) {{
                this.processTick();
                this.updateUI();
            }}
            
            // Mix channels — split into surround bus (ch0,ch3 = outer L pair)
            // and center bus (ch1,ch2 = inner R pair).
            // Only3D is applied to surround bus only → ch1&4 get 3D, ch2&3 stay dry center.
            let mixL = 0, mixR = 0;     // center bus  (ch1, ch2)
            let surrL = 0, surrR = 0;  // surround bus (ch0, ch3)
            
            // Amiga stereo: 75% separation — L channels 87.5%L/12.5%R, R channels inverse
            const SEP = 0.75;
            const panL_left  = 0.5 + SEP * 0.5;   // 0.875
            const panL_right = 0.5 - SEP * 0.5;   // 0.125
            const chPanLeft  = [panL_left, panL_right, panL_right, panL_left];
            const chPanRight = [panL_right, panL_left, panL_left, panL_right];
            
            for (let ch = 0; ch < this.numChannels; ch++) {{
                const state = this.channels[ch];
            // Surround channel pair (1-indexed, matches GLSL surr_channels).
            // [1,4] = outer LEFT pair (ch0,ch3); [2,3] = inner RIGHT pair (ch1,ch2)
            const surroundPair = [1, 4];
            const isSurrCh = (surroundPair.includes((ch % 4) + 1));
                
                if (state.active && state.period > 0) {{
                    const sample = this.getSampleData(state.sample, state.samplePos);
                    state.mixVol += state.volInc;
                    if (state.volInc > 0 && state.mixVol > state.currentVolume) {{
                        state.mixVol = state.currentVolume; state.volInc = 0;
                    }} else if (state.volInc < 0 && state.mixVol < state.currentVolume) {{
                        state.mixVol = state.currentVolume; state.volInc = 0;
                    }}
                    const cv = state.mixVol / 64.0;
                    const s  = sample * cv;
                    
                    if (isSurrCh) {{
                        surrL += s * chPanLeft[ch % 4];
                        surrR += s * chPanRight[ch % 4];
                    }} else {{
                        mixL += s * chPanLeft[ch % 4];
                        mixR += s * chPanRight[ch % 4];
                    }}
                    
                    // PhatBass: bass-named instruments get Hilbert allpass enhancement
                    const isBass = modData.bassSamples.length > 0
                        ? modData.bassSamples.includes(state.sample + 1)
                        : state.period >= 300;
                    if (this._phatBass && this._phatBassDepth > 0 && isBass) {{
                        // Add to whichever bus this channel feeds
                        if (isSurrCh) [surrL, surrR] = this._phatBass.process(s, surrL, surrR);
                        else          [mixL, mixR]   = this._phatBass.process(s, mixL, mixR);
                    }}
                    
                    const effectivePeriod = this.getEffectivePeriod(ch);
                    const smpFt = (modData.sampleMap[state.sample] || {{}}).finetune || 0;
                    const freq = this.periodToFreq(effectivePeriod, smpFt) / modData.downsample;
                    state.samplePos += freq / this.sampleRate;
                }}
            }}
            
            const normFactor = 2.0 / this.numChannels;
            const vol = this._volume !== undefined ? this._volume : 0.8;
            
            // Apply Only3D to surround bus (ch0 + ch3 = outer LEFT pair = "Surround L/R")
            let sL = surrL * normFactor * vol;
            let sR = surrR * normFactor * vol;
            if (this._only3d && this._only3dDepth > 0) {{
                [sL, sR] = this._only3d.process(sL, sR);
            }}
            
            // Center bus (ch1 + ch2 = inner RIGHT pair) passes through dry
            const cL = mixL * normFactor * vol;
            const cR = mixR * normFactor * vol;
            
            let outL = sL + cL;
            let outR = sR + cR;
            
            leftChannel[offset + i]  = outL;
            rightChannel[offset + i] = outR;
            
            // Advance sample counter
            this.sampleCounter++;
            if (this.sampleCounter >= this.samplesPerTick) {{
                this.sampleCounter = 0;
            }}
        }}
    }}
    
    scheduleAudio() {{
        if (!this.isPlaying) return;
        
        const bufferSize = 4096;
        const bufferDuration = bufferSize / this.sampleRate;
        const lookahead = 0.5; // Schedule 500ms ahead
        
        // Schedule multiple buffers to stay ahead
        while (this.nextPlayTime < this.audioCtx.currentTime + lookahead) {{
            const buffer = this.audioCtx.createBuffer(2, bufferSize, this.sampleRate);
            this.generateSamples(buffer, 0, bufferSize);
            
            const source = this.audioCtx.createBufferSource();
            source.buffer = buffer;
            source.connect(this._gainNode || this.audioCtx.destination);
            source.start(this.nextPlayTime);
            
            this.nextPlayTime += bufferDuration;
        }}
        
        // Check again in 50ms
        setTimeout(() => this.scheduleAudio(), 50);
    }}
    
    play() {{
        if (!this.audioCtx) this.init();
        this.isPlaying = true;
        this.nextPlayTime = this.audioCtx.currentTime; // Reset scheduling time
        this.scheduleAudio();
        
        document.getElementById('playBtn').disabled = true;
        document.getElementById('pauseBtn').disabled = false;
        document.getElementById('stopBtn').disabled = false;
    }}
    
    pause() {{
        this.isPlaying = false;
        document.getElementById('playBtn').disabled = false;
        document.getElementById('pauseBtn').disabled = true;
    }}
    
    stop() {{
        this.isPlaying = false;
        this.currentPattern = 0;
        this.currentRow = 0;
        this.currentTick = 0;
        this.sampleCounter = 0;
        
        for (let ch = 0; ch < 4; ch++) {{
            this.channels[ch].active = false;
            this.dyingChannels[ch].active = false;  // OPTIMIZATION 2: Clear dying channels
        }}
        
        document.getElementById('playBtn').disabled = false;
        document.getElementById('pauseBtn').disabled = true;
        document.getElementById('stopBtn').disabled = true;
        
        this.updateUI();
    }}
    
    updateUI() {{
        document.getElementById('patternInfo').textContent = 
            `${{this.currentPattern}} / ${{modData.songLength}}`;
        document.getElementById('rowInfo').textContent = 
            `${{this.currentRow}} / 64`;
    }}
}}

const player = new MODPlayer();

document.getElementById('playBtn').addEventListener('click', () => player.play());
document.getElementById('pauseBtn').addEventListener('click', () => player.pause());
document.getElementById('stopBtn').addEventListener('click', () => player.stop());

// ── oscilloscope & UI ──────────────────────────────────────────
const canvas = document.getElementById('oscCanvas');
const ctx2d  = canvas.getContext('2d');
let animFrame=null, gainNode=null, analyser=null, startTime=0, totalDuration=0;

function resizeCanvas(){{canvas.width=canvas.offsetWidth;canvas.height=canvas.offsetHeight;}}
window.addEventListener('resize',resizeCanvas); resizeCanvas();

const PERIODS=[856,808,762,720,678,640,604,570,538,508,480,453,
               428,404,381,360,339,320,302,285,269,254,240,226,
               214,202,190,180,170,160,151,143,135,127,120,113];
const NNAMES=['C-','C#','D-','D#','E-','F-','F#','G-','G#','A-','A#','B-'];
function pToName(p){{
  if(!p)return'---';
  let best=0,bd=1e9;
  for(let i=0;i<PERIODS.length;i++){{const d=Math.abs(PERIODS[i]-p);if(d<bd){{bd=d;best=i;}}}}
  return NNAMES[best%12]+(Math.floor(best/12)+1);
}}
function fmtT(s){{const m=Math.floor(s/60);return String(m).padStart(2,'0')+':'+String(Math.floor(s%60)).padStart(2,'0');}}

function drawFrame(){{
  animFrame=requestAnimationFrame(drawFrame);
  const W=canvas.width,H=canvas.height;
  ctx2d.fillStyle='#0d0d12';ctx2d.fillRect(0,0,W,H);
  ctx2d.strokeStyle='#1c1c2a';ctx2d.lineWidth=1;
  ctx2d.beginPath();ctx2d.moveTo(0,H/2);ctx2d.lineTo(W,H/2);ctx2d.stroke();
  if(analyser){{
    const buf=new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(buf);
    const step=buf.length/W;
    ctx2d.beginPath();ctx2d.strokeStyle='#3d8ef0';ctx2d.lineWidth=1.8;
    ctx2d.shadowBlur=8;ctx2d.shadowColor='#3d8ef0';
    for(let x=0;x<W;x++){{
      const y=H/2-Math.max(-0.95,Math.min(0.95,buf[Math.floor(x*step)]*8.0))*H*0.45;
      x===0?ctx2d.moveTo(x,y):ctx2d.lineTo(x,y);
    }}
    ctx2d.stroke();ctx2d.shadowBlur=0;
  }}
  if(player.isPlaying&&player.audioCtx){{
    const el=player.audioCtx.currentTime-startTime;
    if(totalDuration>0)document.getElementById('progressBar').style.width=Math.min(el/totalDuration*100,100)+'%';
    document.getElementById('timeDisplay').textContent=fmtT(el)+' / '+fmtT(totalDuration);
    document.getElementById('patternInfo').textContent=String(player.currentPattern).padStart(2,'0');
    document.getElementById('rowInfo').textContent=String(player.currentRow).padStart(2,'0');
    document.getElementById('bpmInfo').textContent=player.bpm;
    document.getElementById('speedInfo').textContent=player.speed;
    for(let ch=0;ch<player.numChannels;ch++){{
      const st=player.channels[ch];
      document.getElementById('chNote'+ch).textContent=(st&&st.active)?pToName(st.period):'---';
      document.getElementById('chBar' +ch).style.width=(st&&st.active)?(st.volume/64*100)+'%':'0%';
    }}
    if(window._trackerUpdate) window._trackerUpdate();
  }}
}}

const _origPlay=player.play.bind(player);
player.play=function(){{
  // Init audio context early so we can wire analyser before first buffer
  if(!this.audioCtx) this.init();
  if(!gainNode){{
    gainNode=this.audioCtx.createGain();
    analyser=this.audioCtx.createAnalyser();
    analyser.fftSize=2048;
    gainNode.gain.value=parseFloat(document.getElementById('volSlider').value);
    gainNode.connect(analyser);analyser.connect(this.audioCtx.destination);
    this._gainNode=gainNode;  // scheduleAudio routes through this
  }}
  _origPlay();  // now start scheduling — buffers connect via _gainNode
  startTime=this.audioCtx.currentTime;
  const rowTime=this.speed/((this.bpm*2)/5);
  totalDuration=modData.songLength*64*rowTime;
  document.getElementById('statusMsg').textContent='Playing';
  if(!animFrame)drawFrame();
}};
const _origStop=player.stop.bind(player);
player.stop=function(){{
  _origStop();
  document.getElementById('progressBar').style.width='0%';
  document.getElementById('timeDisplay').textContent='00:00 / '+fmtT(totalDuration);
  document.getElementById('statusMsg').textContent='Stopped';
  for(let ch=0;ch<this.numChannels;ch++){{
    document.getElementById('chNote'+ch).textContent='---';
    document.getElementById('chBar' +ch).style.width='0%';
  }}
}};
document.getElementById('volSlider').addEventListener('input',function(){{
  player._volume = parseFloat(this.value);
}});
document.getElementById('surroundSlider').addEventListener('input',function(){{
  const d = parseFloat(this.value);
  player._only3dDepth = d;
  if (player._only3d) player._only3d.depth = d;
}});
document.getElementById('surroundModeBtn').addEventListener('click',function(){{
  player._only3dMode = player._only3dMode === 0 ? 1 : 0;
  this.textContent = player._only3dMode === 0 ? '3D:MIX' : '3D:CH';
  this.title = player._only3dMode === 0
    ? 'Total mix mode — Only3D applied to full stereo output'
    : 'Channel mode — Only3D applied to outer channels (ch0+ch3) only';
}});
document.getElementById('phatSlider').addEventListener('input',function(){{
  player._phatBassDepth = parseFloat(this.value);
  if (player._phatBass) player._phatBass.depth = parseFloat(this.value);
}});


// ── tracker pattern display ─────────────────────────────────────
(function(){{
  const ROWS_ABOVE = 4, ROWS_BELOW = 4, TOTAL = ROWS_ABOVE+1+ROWS_BELOW;
  const CH_COLORS = ['ch0-color','ch1-color','ch2-color','ch3-color'];
  const PT=[856,808,762,720,678,640,604,570,538,508,480,453,
            428,404,381,360,339,320,302,285,269,254,240,226,
            214,202,190,180,170,160,151,143,135,127,120,113];
  const NN=['C-','C#','D-','D#','E-','F-','F#','G-','G#','A-','A#','B-'];
  function p2n(p){{
    if(!p) return '---';
    let b=0,bd=1e9;
    for(let i=0;i<PT.length;i++){{const d=Math.abs(PT[i]-p);if(d<bd){{bd=d;b=i;}}}}
    return NN[b%12]+(Math.floor(b/12)+1);
  }}
  function hex2(n){{ return n.toString(16).toUpperCase().padStart(2,'0'); }}

  // Build header
  const hdr = document.getElementById('trkHeader');
  const nc = modData.numChannels||4;
  hdr.innerHTML = '<div class="trk-col-hdr">ROW</div>' +
    Array.from({{length:nc}},(_,i)=>`<div class="trk-col-hdr">TRACK #${{i+1}}</div>`).join('');

  // Pre-build row elements
  const body = document.getElementById('trkBody');
  const rowEls = [];
  for(let r=0;r<TOTAL;r++){{
    const row = document.createElement('div');
    row.className = 'trk-row';
    const rn = document.createElement('div');
    rn.className = 'trk-rownum';
    row.appendChild(rn);
    const cells = [];
    for(let c=0;c<nc;c++){{
      const cell = document.createElement('div');
      cell.className = 'trk-cell';
      row.appendChild(cell);
      cells.push(cell);
    }}
    body.appendChild(row);
    rowEls.push({{row, rn, cells}});
  }}

  function updateTracker(){{
    if(!player.isPlaying) return;
    const pat = modData.songPositions[player.currentPattern]||0;
    if(!modData.patterns||!modData.patterns.length) return;
    const curRow = player.currentRow;

    for(let ri=0;ri<TOTAL;ri++){{
      const rowIdx = curRow - ROWS_ABOVE + ri;
      const {{row, rn, cells}} = rowEls[ri];
      const isCur = (ri === ROWS_ABOVE);
      row.className = 'trk-row' + (isCur ? ' current' : '');

      // Clamp row to 0-63
      const r = ((rowIdx % 64) + 64) % 64;
      rn.textContent = r.toString().padStart(2,'0');

      for(let c=0;c<nc;c++){{
        // flat index: pattern * 64 * nc  +  row * nc  +  channel
        const idx = pat * 64 * nc + r * nc + c;
        const ch = modData.patterns[idx]||{{}};
        const period  = ch.period||0;
        const sample  = ch.sample||0;
        const effect  = ch.effect||0;
        const param   = ch.param||0;
        const cell = cells[c];
        const cc = CH_COLORS[c%4];
        if(!period && !sample && !effect && !param){{
          cell.innerHTML = `<span class="trk-empty">--- -- ---</span>`;
        }} else {{
          const note = p2n(period);
          const smp  = sample ? hex2(sample) : '--';
          const eff  = (effect||param) ? hex2(effect)+hex2(param) : '---';
          cell.innerHTML =
            `<span class="${{cc}}">${{note}}</span> ` +
            `<span class="trk-samp">${{smp}}</span> ` +
            `<span class="trk-eff">${{eff}}</span>`;
        }}
      }}
    }}
  }}

  // Hook into the animation frame
  const _df = drawFrame;
  window.drawFrame = function(){{
    _df();
    updateTracker();
  }};
  // Also override the requestAnimationFrame ref
  window._trackerUpdate = updateTracker;
}})();
console.log('MOD Player Ready!');
    </script>
</body>
</html>
"""


    
    with open(output_file, 'w') as f:
        f.write(html)

def to_glsl_font_chars(text, max_len=24):
    """Convert text to kishimisu font-framework char macro sequence. _ = space."""
    MAP = {' ':'_', '!':'_EX','"':'_DBQ','#':'_NUM','$':'_DOL','%':'_PER',
           '&':'_AMP',"'":'_QT','(':'_LPR',')':'_RPR','+':'_ADD',
           ',':'_COM','-':'_SUB','.':'_DOT','/':'_DIV',':':'_COL',';':'_SEM',
           '<':'_LES','=':'_EQ','>':'_GE','?':'_QUE','@':'_AT','[':'_LBR',
           '\\':'_ANTI',']':'_RBR','_':'_UN'}
    out = []
    for c in text.upper()[:max_len]:
        if c in MAP:      out.append(MAP[c])
        elif 'A'<=c<='Z': out.append(f'_{c}')
        elif '0'<=c<='9': out.append(f'_{c}')
        else:              out.append('_')
    return ' '.join(out)

def create_shadertoy_glsl(mod, output_file, downsample=1, compress=True, compressed_pattern_size=None, 
                          pattern_bytes_data=None, sample_bytes_data=None, seek_table=None):
    """Generate ShaderToy GLSL code with texture-based OR embedded data"""
    
    # Configuration
    MAX_TOTAL_SIZE = 160000  # 60KB max for embedded data (GLSL source limit)
    MAX_CHUNK_SIZE = 8192   # 4KB per chunk = 1024 int32s — fast GLSL compile
    
    # Derive PNG filename
    base_name = output_file.replace('_shadertoy.glsl', '')
    png_file = base_name + "_player_data.png"
    
    # Calculate pattern data size (use compressed size if provided)
    if compressed_pattern_size is not None:
        pattern_size = compressed_pattern_size
    else:
        pattern_size = mod.num_patterns * 64 * 4 * 5  # Uncompressed fallback
    
    # Decide: Embed data or use PNG?
    use_embedded = False
    if pattern_bytes_data is not None and sample_bytes_data is not None:
        total_size = len(pattern_bytes_data) + len(sample_bytes_data)
        if total_size <= MAX_TOTAL_SIZE:
            use_embedded = True
            print(f"   💾 Embedding {total_size} bytes directly in GLSL (no PNG needed!)")
        else:
            print(f"   🖼️  Using PNG texture ({total_size} bytes > {MAX_TOTAL_SIZE} limit)")
    
    # Prepare sample map — each sample gets 32-byte zero-padding for cubic interpolation
    all_samples = []
    sample_map = []
    
    for sample in mod.samples:
        start_idx = len(all_samples)
        raw_len = 0
        bw_factor = 1
        if sample['data'] is not None and len(sample['data']) > 0:
            bw_factor, compressed = bw_compress_sample(sample['data'])
            all_samples.extend(compressed.astype(np.float64) / 128.0)
            raw_len = len(compressed)
            all_samples.extend([0.0] * 32)  # zero-padding: pos+1 and pos+2 always safe
        sample_map.append({
            'start':          start_idx,
            'length':         raw_len,
            'repeat_point':   sample['repeat_point'] // bw_factor,
            'repeat_length':  sample['repeat_length'] // bw_factor if sample['repeat_length'] > 2 else sample['repeat_length'],
            'bw_factor':      bw_factor,
        })
    
    # Remove old pattern processing - everything goes in PNG now
    
    # Generate embedded data chunks if needed
    def generate_data_chunks(data, chunk_size=4096):
        """Split data into chunks for GLSL arrays"""
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            chunks.append(chunk)
        return chunks
    
    embedded_data_code = ""
    if use_embedded:
        # Chunk pattern data
        pattern_chunks = generate_data_chunks(pattern_bytes_data, MAX_CHUNK_SIZE)
        
        # Chunk sample data
        sample_chunks = generate_data_chunks(sample_bytes_data, MAX_CHUNK_SIZE)
        
        # VEC4 OPTIMIZATION: Pack bytes as normalized floats.
        embedded_data_code = "// Embedded pattern data (4 bytes per int32, hex literals)\n"
        for i, chunk in enumerate(pattern_chunks):
            embedded_data_code += format_int32_chunk_glsl(pack_bytes_to_int32(chunk), "patternData", i)

        embedded_data_code += "\n// Embedded sample data (4 bytes per int32, hex literals)\n"
        for i, chunk in enumerate(sample_chunks):
            embedded_data_code += format_int32_chunk_glsl(pack_bytes_to_int32(chunk), "sampleData", i)

        # Each float holds 3 bytes
        # Use actual chunk sizes (not padded max) — smaller arrays = faster GLSL compile
        pat_chunk_sizes  = [(len(c) + 3) // 4 for c in pattern_chunks]
        smp_chunk_sizes  = [(len(c) + 3) // 4 for c in sample_chunks]
        max_pat_chunk    = max(pat_chunk_sizes)
        max_smp_chunk    = max(smp_chunk_sizes)
        embedded_data_code += f"\n#define NUM_PATTERN_CHUNKS {len(pattern_chunks)}\n"
        embedded_data_code += f"#define NUM_SAMPLE_CHUNKS {len(sample_chunks)}\n"
        embedded_data_code += f"#define PATTERN_CHUNK_SIZE {max_pat_chunk}\n"
        embedded_data_code += f"#define PAT_INTS_PER_CHUNK {max_pat_chunk}\n"
        embedded_data_code += f"#define SAMPLE_CHUNK_SIZE {max_smp_chunk}\n"
        embedded_data_code += f"#define SMP_INTS_PER_CHUNK {max_smp_chunk}\n"
        embedded_data_code += f"#define PATTERN_BYTE_SIZE {len(pattern_bytes_data)}\n"
        embedded_data_code += f"#define SAMPLE_BYTE_SIZE {len(sample_bytes_data)}\n"
    
    # ========== DETECT EFFECT B LOOP POINT (must be before f-string below) ==========
    loop_target_songpos = 0  # default: loop from start
    num_channels_tmp = mod.num_channels if hasattr(mod, 'num_channels') else 4
    for si, pi in enumerate(mod.song_positions):
        for row in range(64):
            for ch in range(num_channels_tmp):
                try:
                    note = mod.patterns[pi][row][ch]
                    if note.get('effect', 0) == 0xB:
                        param = note.get('param', 0)
                        if 0 < param < len(mod.song_positions):
                            loop_target_songpos = param
                except: pass
    if loop_target_songpos > 0:
        print(f"   🔁  Effect B: song loops back to position {loop_target_songpos}")

    # ========== PRE-COMPUTE ROW OFFSETS (Effect D/B timeline) ==========
    _num_ch = mod.num_channels if hasattr(mod, 'num_channels') else 4
    _pat_row_offsets = []   # cumulative rows BEFORE each song position
    _pat_start_rows  = []   # which pattern row each song position starts at
    _cumul = 0
    _next_start = 0
    for _si, _pi in enumerate(mod.song_positions):
        _pat_row_offsets.append(_cumul)
        _pat_start_rows.append(_next_start)
        _d_row = None; _d_param = 0
        for _ri in range(_next_start, 64):
            for _ch in range(_num_ch):
                try:
                    _n = mod.patterns[_pi][_ri][_ch]
                    if _n.get('effect',0) == 0xD and _d_row is None:
                        _d_row = _ri
                        _d_param = ((_n['param']>>4)&0xF)*10 + (_n['param']&0xF)
                except: pass
        if _d_row is not None:
            _cumul += _d_row - _next_start + 1
            _next_start = _d_param
            if _si == 0:
                print(f"   ⏩  Effect D: pattern {_pi} breaks at row {_d_row} → songPos 1 starts at row {_d_param}")
        else:
            _cumul += 64 - _next_start
            _next_start = 0
    _pat_row_offsets.append(_cumul)   # sentinel (total rows)
    _total_song_rows = _cumul
    _has_d_effects = any(_pat_start_rows[i]!=0 or _pat_row_offsets[i+1]-_pat_row_offsets[i]!=64
                         for i in range(len(mod.song_positions)))
    # ---- build GLSL arrays ----
    _row_off_str  = ', '.join(map(str, _pat_row_offsets))
    _start_row_str = ', '.join(map(str, _pat_start_rows))
    _intro_rows = _pat_row_offsets[loop_target_songpos] if loop_target_songpos else 0
    _loop_rows  = _total_song_rows - _intro_rows

    # ========== COMMON TAB ==========
    data_source_comment = "Embedded data (no PNG required)" if use_embedded else f"All data in 1024×1024 RGBA PNG: {png_file}"
    common_glsl = f"""/* ============================================================================
   GLSL MOD Player v1.3 (c) 2026 Orblivius
   RVQ sample compression, 3D Surround, FAT Bass, cubic resampling
   COMMON TAB
   Contact: subband@gmail.com or
            subband@protonmail.com
   GIT:     https://github.com/mewza/mod2glsl
  ============================================================================ */
// Generated from: {mod.title}
// {data_source_comment}

#define USE_EMBEDDED_DATA {1 if use_embedded else 0}
#define TEX_WIDTH 1024
#define TEX_HEIGHT 1024
#define PATTERN_DATA_SIZE {pattern_size}
#define NUM_PATTERNS {mod.num_patterns}
#define SONG_LENGTH {len(mod.song_positions)}
#define SONG_LOOP_POS {loop_target_songpos}
#define NUM_CHANNELS {mod.num_channels}
#define BYTES_PER_ROW {mod.num_channels * 5}
#define SEEK_TABLE_SIZE {len(seek_table) if seek_table else mod.num_patterns * 64}
#define BPM {float(mod.initial_tempo)}
#define SPEED {float(mod.initial_speed)}

// ── Audio effects — toggle here in Common tab ─────────────────────────────────
// enable3D : Only3D surround widening on the surr_channels pair
// enableFAT: PhatBass Hilbert allpass enhancement on bass instruments
// surr_channels: 1-indexed channel pair that gets Only3D (the other two = dry center)
//   ivec2(1,4) = outer LEFT pair (ch0,ch3) — default Amiga layout
//   ivec2(2,3) = inner RIGHT pair (ch1,ch2) — swap surround and center
const bool  enable3D     = true;
const bool  enableFAT    = true;
const ivec2 surr_channels = ivec2(1, 4);  // 1-indexed; change to ivec2(2,3) to flip

// Channel panning (0=left, 0.5=center, 1.0=right)
const float channelPan[32] = float[]({', '.join([
    f'{[0.0,1.0,1.0,0.0][i%4]:.1f}' if i < mod.num_channels else '0.5'
    for i in range(32)
])});

// Song positions
const int songPositions[{len(mod.song_positions)}] = int[]({', '.join(map(str, mod.song_positions))});

// Row offsets — accounts for Effect D (pattern break) early exits
// patRowOffset[i] = cumulative rows before song position i starts
// patStartRow[i]  = which pattern row song position i begins at (non-zero after D with row > 0)
const int patRowOffset[{len(mod.song_positions)+1}] = int[]({_row_off_str});
const int patStartRow[{len(mod.song_positions)}]    = int[]({_start_row_str});
#define TOTAL_SONG_ROWS {_total_song_rows}

// RLE seek table — one entry per row, value = compressed stream offset at row start.
// Rows never straddle RLE run boundaries, so this is an exact O(1) jump.
// Stored as raw floats (small integers, all exactly representable in float32).
const float patternSeek[SEEK_TABLE_SIZE] = float[]({
    ', '.join(f'{float(v):.1f}' for v in (seek_table if seek_table else [0] * (mod.num_patterns * 64)))
});

{embedded_data_code if use_embedded else ""}

// Multiply instead of divide — GPU multiply is much faster than divide
const float MUL1 = 255.0;   // 1-byte texel decode (pixel.r * MUL1)

#if USE_EMBEDDED_DATA

#pragma optimize(off)  // ivec4 data arrays: skip optimizer = fast compile
int fetchPatternInt(int chunkIdx, int i) {{
    ivec4 v = ivec4(0);
    if (chunkIdx == 0) v = patternData0[i>>2];
#if NUM_PATTERN_CHUNKS > 1
    else if (chunkIdx == 1) v = patternData1[i>>2];
#endif
#if NUM_PATTERN_CHUNKS > 2
    else if (chunkIdx == 2) v = patternData2[i>>2];
#endif
#if NUM_PATTERN_CHUNKS > 3
    else if (chunkIdx == 3) v = patternData3[i>>2];
#endif
#if NUM_PATTERN_CHUNKS > 4
    else if (chunkIdx == 4) v = patternData4[i>>2];
#endif
#if NUM_PATTERN_CHUNKS > 5
    else if (chunkIdx == 5) v = patternData5[i>>2];
#endif
    int ci = i & 3;
    return ci==0 ? v.x : ci==1 ? v.y : ci==2 ? v.z : v.w;
}}

int fetchSampleInt(int chunkIdx, int i) {{
    ivec4 v = ivec4(0);
    if      (chunkIdx == 0) v = sampleData0[i>>2];
#if NUM_SAMPLE_CHUNKS > 1
    else if (chunkIdx == 1) v = sampleData1[i>>2];
#endif
#if NUM_SAMPLE_CHUNKS > 2
    else if (chunkIdx == 2) v = sampleData2[i>>2];
#endif
#if NUM_SAMPLE_CHUNKS > 3
    else if (chunkIdx == 3) v = sampleData3[i>>2];
#endif
#if NUM_SAMPLE_CHUNKS > 4
    else if (chunkIdx == 4) v = sampleData4[i>>2];
#endif
#if NUM_SAMPLE_CHUNKS > 5
    else if (chunkIdx == 5) v = sampleData5[i>>2];
#endif
#if NUM_SAMPLE_CHUNKS > 6
    else if (chunkIdx == 6) v = sampleData6[i>>2];
#endif
#if NUM_SAMPLE_CHUNKS > 7
    else if (chunkIdx == 7) v = sampleData7[i>>2];
#endif
    int ci = i & 3;
    return ci==0 ? v.x : ci==1 ? v.y : ci==2 ? v.z : v.w;
}}
#pragma optimize(on)   // re-enable full optimization for all shader logic

// Extract single byte from packed int32 pattern data
int getPackedByte(int chunkIdx, int localByteIdx) {{
    int packed = fetchPatternInt(chunkIdx, localByteIdx / 4);
    int shift  = 24 - (localByteIdx % 4) * 8;
    return (packed >> shift) & 0xFF;
}}

// Extract 4 contiguous bytes — 1 or 2 int fetches
ivec4 getPackedBytes4(int chunkIdx, int localByteIdx) {{
    int i0  = localByteIdx / 4;
    int off = localByteIdx % 4;
    int p0  = fetchPatternInt(chunkIdx, i0);
    if (off == 0) return ivec4((p0>>24)&0xFF, (p0>>16)&0xFF, (p0>>8)&0xFF, p0&0xFF);
    int p1 = fetchPatternInt(chunkIdx, i0 + 1);
    if (off == 1) return ivec4((p0>>16)&0xFF, (p0>>8)&0xFF, p0&0xFF, (p1>>24)&0xFF);
    if (off == 2) return ivec4((p0>>8)&0xFF,  p0&0xFF, (p1>>24)&0xFF, (p1>>16)&0xFF);
    return             ivec4(p0&0xFF, (p1>>24)&0xFF, (p1>>16)&0xFF, (p1>>8)&0xFF);
}}

// Extract single byte from packed int32 sample data
int getPackedSampleByte(int chunkIdx, int localByteIdx) {{
    int packed = fetchSampleInt(chunkIdx, localByteIdx / 4);
    int shift  = 24 - (localByteIdx % 4) * 8;
    return (packed >> shift) & 0xFF;
}}

#endif

// Sample info
struct SampleInfo {{
    int start, length, loopStart, loopLen, volume, bwFactor;
}};
const SampleInfo samples[31] = SampleInfo[](
"""
    
    for i, s in enumerate(sample_map[:31]):
        comma = "," if i < 30 else ""
        vol = mod.samples[i]['volume'] if i < len(mod.samples) else 64
        common_glsl += f"    SampleInfo({s['start']}, {s['length']}, {s['repeat_point']}, {s['repeat_length']}, {vol}, {s.get('bw_factor',1)}){comma}\n"
    
    common_glsl += f""");

"""
    
    common_glsl += f"""
// ProTracker period table (C-1 to B-3)
const int periodTable[37] = int[](
    856,808,762,720,678,640,604,570,538,508,480,453,
    428,404,381,360,339,320,302,285,269,254,240,226,
    214,202,190,180,170,160,151,143,135,127,120,113,0
);

float periodToFreq(int period) {{
    // Amiga PAL clock / (period * 2 * downsampleFactor)
    return period > 0 ? 7093789.2 / (float(period) * {2.0 * downsample}) : 0.0;
}}

// Note structure  
struct Note {{
    int instrument, period, effect, param;
}};

// Calculate playback position from time
struct Position {{
    int songPos, pattern, row;
    float tick, rowTime;
}};

Position getPosition(float time) {{
    Position pos;
    float ticksPerSec = BPM * 2.0 / 5.0;
    float rowTime = SPEED / ticksPerSec;

    // Song duration using pre-computed row offsets (accounts for Effect D breaks)
    float songDuration = float(TOTAL_SONG_ROWS) * rowTime;
    float loopedTime;
    if (SONG_LOOP_POS == 0) {{
        loopedTime = mod(time, songDuration);
    }} else {{
        float introDur = float({_intro_rows}) * rowTime;
        float loopDur  = float({_loop_rows})  * rowTime;
        if (time < songDuration) {{
            loopedTime = time;
        }} else {{
            loopedTime = introDur + mod(time - songDuration, loopDur);
        }}
    }}

    float totalRows = loopedTime / rowTime;

    // Binary search through patRowOffset to find song position
    int sp = SONG_LENGTH - 1;
    for (int i = 0; i < SONG_LENGTH; i++) {{
        if (float(patRowOffset[i + 1]) > totalRows) {{ sp = i; break; }}
    }}
    pos.songPos = sp;
    pos.pattern = songPositions[sp];
    float rowsIntoPos = totalRows - float(patRowOffset[sp]);
    pos.row     = patStartRow[sp] + int(rowsIntoPos);
    pos.row     = min(pos.row, 63);
    pos.tick    = fract(rowsIntoPos) * float(SPEED);
    pos.rowTime = rowTime;

    return pos;
}}

// ============================================================================
// Data access — shared by Sound and Image tabs
// ============================================================================

// Read byte - supports both embedded arrays and PNG texture
int getByte(int byteIndex) {{
#if USE_EMBEDDED_DATA
    int bytesPerChunk = PAT_INTS_PER_CHUNK * 4;
    int chunkIdx     = byteIndex / bytesPerChunk;
    int localByteIdx = byteIndex % bytesPerChunk;
    int packed = fetchPatternInt(chunkIdx, localByteIdx / 4);
    int shift  = 24 - (localByteIdx % 4) * 8;
    return (packed >> shift) & 0xFF;
#else
    int actualIndex = byteIndex + 4;
    int pixelIdx = actualIndex >> 2;
    int channel = actualIndex & 3;
    int x = pixelIdx & 1023;
    int y = pixelIdx >> 10;
    vec4 pixel = texelFetch(iChannel0, ivec2(x, y), 0);
    if (channel == 0) return int(pixel.r * MUL1 + 0.5);
    if (channel == 1) return int(pixel.g * MUL1 + 0.5);
    if (channel == 2) return int(pixel.b * MUL1 + 0.5);
    return int(pixel.a * MUL1 + 0.5);
#endif
}}

// RLE decompressor — O(1) via seek table, bounded inner scan
int getPatternByte(int targetIndex) {{
    int rowIdx   = targetIndex / BYTES_PER_ROW;
    int posInRow = targetIndex % BYTES_PER_ROW;
    int rlePos   = int(patternSeek[rowIdx]);
    int pos = 0;
    // Bounded loop — prevents GPU hang if data is corrupted (count==0 guard)
    for (int iter = 0; iter < BYTES_PER_ROW + 4; iter++) {{
        int count = getByte(rlePos);
        if (count == 0) return 0;  // safety: corrupted data guard
        int val = getByte(rlePos + 1);
        if (pos + count > posInRow) return val;
        pos    += count;
        rlePos += 2;
        if (pos > posInRow) return 0;
    }}
    return 0;
}}

float getSample(int index) {{
#if USE_EMBEDDED_DATA
    int bytesPerChunk = SMP_INTS_PER_CHUNK * 4;
    int chunkIdx     = index / bytesPerChunk;
    int localByteIdx = index % bytesPerChunk;
    int packed  = fetchSampleInt(chunkIdx, localByteIdx / 4);
    int shift   = 24 - (localByteIdx % 4) * 8;
    int byteVal = (packed >> shift) & 0xFF;
    return (float(byteVal) - 128.0) / MUL1;
#else
    int byteVal = getByte(PATTERN_DATA_SIZE + index);
    return (float(byteVal) - 128.0) / 128.0;
#endif
}}

// 4-point cubic Hermite (Catmull-Rom) interpolation.
// getSample() reads one integer-indexed byte; getSampleF() interpolates between 4.
float getSampleF(int base, float fpos, int smpLen, int loopStart, int loopLen) {{
    int i  = int(fpos);
    float t = fpos - float(i);
    // p0: if at loop start, wrap back to loop end for seamless cubic interpolation
    int i0 = i - 1;
    if (loopLen > 2 && i0 < loopStart) i0 = loopStart + loopLen - 1;
    else i0 = max(0, i0);
    float p0 = getSample(base + i0);
    float p1 = getSample(base + i);
    float p2 = getSample(base + min(i + 1, smpLen + 31));
    float p3 = getSample(base + min(i + 2, smpLen + 31));
    float a = -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3;
    float b =      p0 - 2.5*p1 + 2.0*p2 - 0.5*p3;
    float c = -0.5*p0           + 0.5*p2;
    return ((a * t + b) * t + c) * t + p1;
}}

Note getNote(int songPos, int row, int channel) {{
    int pattern = songPositions[songPos];
    int baseIdx = ((pattern * 64) + row) * NUM_CHANNELS * 5 + channel * 5;
    int b0 = getPatternByte(baseIdx);
    int b1 = getPatternByte(baseIdx + 1);
    int b2 = getPatternByte(baseIdx + 2);
    int b3 = getPatternByte(baseIdx + 3);
    Note n;
    n.instrument = (b0 & 0xF0) | ((b2 >> 4) & 0x0F);
    n.period     = ((b0 & 0x0F) << 8) | b1;
    n.effect     = b2 & 0x0F;
    n.param      = b3;
    return n;
}}

// ── getChannelOutput: stateless per-channel audio (shared by Sound + Image) ──
float getChannelOutput(int ch, float time, Position pos, float rowTime) {{

    // Step 1: find most-recently-triggered note on this channel
    Note trigNote = getNote(pos.songPos, pos.row, ch);
    int  trigRow  = pos.row;
    int  trigPat  = pos.songPos;
    if (trigNote.instrument <= 0 || trigNote.period <= 0) {{
        int scanRow = pos.row;
        int scanPat = pos.songPos;
        for (int lb = 1; lb < 64; lb++) {{
            scanRow--;
            if (scanRow < 0) {{
                if (scanPat > 0) {{
                    scanPat--;
                    // last row played in previous song position = startRow + rowCount - 1
                    int _rowCount = patRowOffset[scanPat+1] - patRowOffset[scanPat];
                    scanRow = patStartRow[scanPat] + _rowCount - 1;
                }} else {{ break; }}
            }}
            Note prev = getNote(scanPat, scanRow, ch);
            if (prev.instrument > 0 || prev.period > 0) {{
                trigNote = prev; trigRow = scanRow; trigPat = scanPat;
                break;
            }}
        }}
    }}

    if (trigNote.instrument <= 0 || trigNote.instrument > 31 || trigNote.period <= 0)
        return 0.0;

    SampleInfo smp = samples[trigNote.instrument - 1];
    if (smp.length == 0) return 0.0;

    // Use patRowOffset so timing is correct even when patterns end early (Effect D)
    float triggerTime = (float(patRowOffset[trigPat]) + float(trigRow - patStartRow[trigPat])) * rowTime;
    float currentTime = (float(patRowOffset[pos.songPos]) + float(pos.row - patStartRow[pos.songPos])) * rowTime + mod(time, rowTime);
    float elapsed = currentTime - triggerTime;
    if (elapsed < 0.0) return 0.0;

    // ── Vibrato period modulation (Effect 4 or 6) ────────────────────────────
    // Phase is derived from elapsed ticks since trigger — stateless approximation.
    // For bare vibrato rows after the trigger, we use the trigger note's speed/depth.
    float effectivePeriod = float(trigNote.period);
    {{
        int _vibSpeed = 0, _vibDepth = 0;
        if (trigNote.effect == 0x4 || trigNote.effect == 0x6) {{
            _vibSpeed = (trigNote.param >> 4) & 0xF;
            _vibDepth = trigNote.param & 0xF;
        }}
        // Also check for bare vibrato rows between trigger and current
        if (trigPat == pos.songPos || _vibDepth == 0) {{
            // Quick scan for any bare 4xx between trigRow+1 and pos.row
            for (int _vi=1; _vi<=16; _vi++) {{
                int _vr = trigRow + _vi;
                if (_vr >= 64 || _vr >= pos.row) break;
                Note _vn = getNote(trigPat, _vr, ch);
                if (_vn.instrument>0 || _vn.period>0) break;  // new note
                if ((_vn.effect==0x4 || _vn.effect==0x6) && _vn.param!=0) {{
                    _vibSpeed = (_vn.param >> 4) & 0xF;
                    _vibDepth = _vn.param & 0xF;
                }}
            }}
        }}
        if (_vibDepth > 0) {{
            // Total ticks elapsed since trigger (vibrato resets on note trigger)
            float _vibTicks = elapsed * float(SPEED) / rowTime;
            int   _vibPos   = int(_vibTicks) * _vibSpeed & 63;
            // ProTracker VibratoTable: 32-entry sine, amplitude=(tab[pos&31]*depth)>>7
            float _vibTab[32]; 
            _vibTab[ 0]=  0.0; _vibTab[ 1]= 24.0; _vibTab[ 2]= 49.0; _vibTab[ 3]= 74.0;
            _vibTab[ 4]= 97.0; _vibTab[ 5]=120.0; _vibTab[ 6]=141.0; _vibTab[ 7]=161.0;
            _vibTab[ 8]=180.0; _vibTab[ 9]=197.0; _vibTab[10]=212.0; _vibTab[11]=224.0;
            _vibTab[12]=235.0; _vibTab[13]=244.0; _vibTab[14]=250.0; _vibTab[15]=253.0;
            _vibTab[16]=255.0; _vibTab[17]=253.0; _vibTab[18]=250.0; _vibTab[19]=244.0;
            _vibTab[20]=235.0; _vibTab[21]=224.0; _vibTab[22]=212.0; _vibTab[23]=197.0;
            _vibTab[24]=180.0; _vibTab[25]=161.0; _vibTab[26]=141.0; _vibTab[27]=120.0;
            _vibTab[28]= 97.0; _vibTab[29]= 74.0; _vibTab[30]= 49.0; _vibTab[31]= 24.0;
            float _vibDelta = (_vibTab[_vibPos & 31] * float(_vibDepth)) / 128.0;
            effectivePeriod += (_vibPos < 32) ? _vibDelta : -_vibDelta;
        }}
    }}

    float freq       = periodToFreq(max(1, int(effectivePeriod)));
    float fSamplePos = elapsed * freq / float(smp.bwFactor);  // map to compressed sample space

    if (smp.loopLen > 2) {{
        if (fSamplePos >= float(smp.loopStart + smp.loopLen))
            fSamplePos = float(smp.loopStart) + mod(fSamplePos - float(smp.loopStart), float(smp.loopLen));
    }} else if (fSamplePos >= float(smp.length)) {{
        return 0.0;
    }}
    if (fSamplePos < 0.0) return 0.0;

    float s = getSampleF(smp.start, fSamplePos, smp.length, smp.loopStart, smp.loopLen);

    // ── Volume: forward scan trigger→current to honour Cxx cuts & Axx slides ─
    // ProTracker volume slide (Effect A/6) SKIPS tick 0 → applies (SPEED-1) ticks per row.
    int volume = smp.volume;
    // Trigger-row effect
    if (trigNote.effect == 0xC) {{
        volume = min(trigNote.param, 64);
    }} else if (trigNote.effect == 0xA || trigNote.effect == 0x6) {{
        int _su=(trigNote.param>>4)&0xF, _sd=trigNote.param&0xF;
        volume = clamp(volume + (_su>0?_su:-_sd)*(int(SPEED)-1), 0, 64);
    }}
    // Forward scan through non-note rows from trigRow+1 to pos.row-1
    // Effect D can shorten patterns — skip phantom rows by using patRowOffset boundaries
    if (trigPat != pos.songPos || trigRow != pos.row) {{
        int _fp=trigPat, _fr=trigRow+1;
        // If _fr is past the actual end of this song position (Effect D), jump to next
        if (_fr >= patRowOffset[_fp+1] - patRowOffset[_fp] + patStartRow[_fp])
            {{ _fr=patStartRow[_fp+1]; _fp++; }}
        else if (_fr >= 64) {{ _fr=0; _fp++; }}
        for (int _fi=0; _fi<64; _fi++) {{
            if (_fp > pos.songPos || (_fp==pos.songPos && _fr>=pos.row)) break;
            if (_fp >= SONG_LENGTH) break;
            // Skip phantom rows: rows >= patStartRow + rowCount for this position
            int _posRows = patRowOffset[_fp+1] - patRowOffset[_fp];
            if (_fr >= patStartRow[_fp] + _posRows) {{
                _fp++; _fr = (_fp < SONG_LENGTH) ? patStartRow[_fp] : 0;
                continue;
            }}
            Note _fn = getNote(_fp, _fr, ch);
            if (_fn.instrument>0 || _fn.period>0) break; // new note = different trigger
            if (_fn.effect==0xC)
                volume = min(_fn.param, 64);
            else if (_fn.effect==0xA || _fn.effect==0x6) {{
                int _su=(_fn.param>>4)&0xF, _sd=_fn.param&0xF;
                volume = clamp(volume+(_su>0?_su:-_sd)*(int(SPEED)-1), 0, 64);
            }}
            _fr++;
            if (_fr >= 64) {{ _fr=0; _fp++; }}
        }}
        // Current row: Cxx fully, Axx for elapsed ticks (tick 0 skipped)
        Note _cr = getNote(pos.songPos, pos.row, ch);
        if (_cr.instrument<=0 && _cr.period<=0) {{
            if (_cr.effect==0xC)
                volume = min(_cr.param, 64);
            else if (_cr.effect==0xA || _cr.effect==0x6) {{
                int _su=(_cr.param>>4)&0xF, _sd=_cr.param&0xF;
                int _ticks = max(0, int(pos.tick) - 1);  // tick 0 skipped
                volume = clamp(volume+(_su>0?_su:-_sd)*_ticks, 0, 64);
            }}
        }}
    }}
    return s * float(volume) / 64.0;
}}
"""
    
    # ========== SOUND TAB ==========
    # ========== SOUND TAB ==========
    # Build bass sample boolean array for GLSL (1-indexed samples named "bass")
    bass_sample_flags = [
        'true' if (i < len(mod.samples) and mod.samples[i]['length'] > 0 and 
                   'bass' in mod.samples[i]['name'].lower()) else 'false'
        for i in range(31)
    ]
    bass_flags_str = ', '.join(bass_sample_flags)
    
    sound_glsl = f"""/* ============================================================================
   GLSL MOD Player v1.3 (c) 2026 Orblivius
   RVQ sample compression, 3D Surround, FAT Bass, cubic resampling
   SOUND TAB
   Contact: subband@gmail.com or
            subband@protonmail.com
   GIT:     https://github.com/mewza/mod2glsl
  ============================================================================ */
// getByte / getPatternByte / getSample / getNote / getChannelOutput are in Common.

// Bass sample flags (true = instrument name contains "bass") — for PhatBass
const bool isBass[31] = bool[]({bass_flags_str});

vec2 mainSound(int samp, float time) {{
#if !USE_EMBEDDED_DATA
    // PNG mode - check magic signature
    vec4 magic = texelFetch(iChannel0, ivec2(0, 0), 0);
    int magicR = int(magic.r * MUL1 + 0.5);
    int magicG = int(magic.g * MUL1 + 0.5);
    int magicB = int(magic.b * MUL1 + 0.5);
    int loopMode = int(magic.a * MUL1 + 0.5);
    
    bool hasSignature = (magicR == 77 && magicG == 79 && magicB == 68);
    
    if (!hasSignature) {{
        float t = mod(time, 3.0);
        float debugFreq = (t < 1.0) ? 300.0 + float(magicR) :
                          (t < 2.0) ? 300.0 + float(magicG) :
                                      300.0 + float(magicB);
        float localT = mod(time, 1.0);
        float ping = sin(6.2831 * debugFreq * localT) * exp(-3.0 * localT) * 0.5;
        return vec2(ping);
    }}
    
    float playbackTime = time;
    if (loopMode == 255) {{
        float songDuration = float(TOTAL_SONG_ROWS) * SPEED / (BPM * 2.0 / 5.0);
        playbackTime = mod(time, min(10.0, songDuration));
    }}
#else
    float playbackTime = time;
#endif
    
    Position pos = getPosition(playbackTime);
    float ticksPerSec = BPM * 2.0 / 5.0;
    float rowTime = SPEED / ticksPerSec;
    
    // 75% Amiga stereo separation
    const float SEP = 0.50;  // 50% separation: all 4 channels audible in both ears
    float chL[4], chR[4];
    chL[0]=0.5+SEP*0.5; chL[1]=0.5-SEP*0.5; chL[2]=0.5-SEP*0.5; chL[3]=0.5+SEP*0.5;
    chR[0]=0.5-SEP*0.5; chR[1]=0.5+SEP*0.5; chR[2]=0.5+SEP*0.5; chR[3]=0.5-SEP*0.5;
    
    // Split into surround bus (ch0,ch3 = outer LEFT pair → Surround L/R)
    // and center bus (ch1,ch2 = inner RIGHT pair → dry center)
    float surrL = 0.0, surrR = 0.0;
    float centL = 0.0, centR = 0.0;
    
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
        float s = getChannelOutput(ch, playbackTime, pos, rowTime);
        int cm = ch % 4;
        int ch1 = cm + 1;  // 1-indexed
        bool isSurr = (ch1 == surr_channels.x || ch1 == surr_channels.y);
        if (isSurr) {{ surrL += s * chL[cm]; surrR += s * chR[cm]; }}
        else        {{ centL += s * chL[cm]; centR += s * chR[cm]; }}
    }}
    
    float normFactor = 2.0 / float(NUM_CHANNELS);
    surrL *= normFactor; surrR *= normFactor;
    centL *= normFactor; centR *= normFactor;
    
    // ── Only3D — surround bus only (ch0+ch3 = outer LEFT pair, ch1+ch2 = dry center) ─
    const float ONLY3D_DELAY = 0.000431;  // 19 samples @ 44100Hz
    const float ONLY3D_DEPTH = 0.25;  // reduced: SEP already gives stereo width
    if (enable3D) {{
        float tW = playbackTime - ONLY3D_DELAY;
        Position posW = getPosition(tW);
        float wL = 0.0, wR = 0.0;
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
            int cm = ch % 4;
            int ch1 = cm + 1;
            if (ch1 == surr_channels.x || ch1 == surr_channels.y) {{
                float sw = getChannelOutput(ch, tW, posW, rowTime);
                wL += sw * chL[cm]; wR += sw * chR[cm];
            }}
        }}
        wL *= normFactor; wR *= normFactor;
        float diff = (wL - wR) * ONLY3D_DEPTH;
        surrL += diff;
        surrR -= diff;
    }}
    
    // ── PhatBass — bass-named instruments on center bus ────────────────────────
    const float PHAT_DELAY = 0.001814;  // 80 samples @ 44100Hz
    const float PHAT_DEPTH = 0.5;
    if (enableFAT) {{
        float tP = playbackTime - PHAT_DELAY;
        Position posP = getPosition(tP);
        float pbL = 0.0, pbR = 0.0;
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
            Note n = getNote(pos.songPos, pos.row, ch);
            int inst = n.instrument;
            bool bass = (inst >= 1 && inst <= 31) ? isBass[inst - 1] : false;
            if (bass) {{
                float sp = getChannelOutput(ch, tP, posP, rowTime);
                int cm = ch % 4;
                pbL += sp * chR[cm];  // cross-pan: PHASESHIFT90→L, PHASESHIFT0→R
                pbR += sp * chL[cm];
            }}
        }}
        centL += pbL * normFactor * PHAT_DEPTH;
        centR += pbR * normFactor * PHAT_DEPTH;
    }}
    
    float outL = surrL + centL;
    float outR = surrR + centR;
    return vec2(clamp(outL, -1.0, 1.0), clamp(outR, -1.0, 1.0));
}}
"""
    
    
    # ========== IMAGE TAB ==========
    raw_title   = mod.title.strip() or "UNTITLED"
    title_text  = raw_title[:20]
    title_chars = to_glsl_font_chars(title_text)
    title_len   = len(title_text)
    bpm_val_chars = to_glsl_font_chars(str(int(mod.initial_tempo)))
    bpm_val_len   = len(str(int(mod.initial_tempo)))
    spd_val_chars = to_glsl_font_chars(str(int(mod.initial_speed)))
    spd_val_len   = len(str(int(mod.initial_speed)))

    image_glsl = f"""/* ============================================================================
   GLSL MOD Player v1.3 (c) 2026 Orblivius
   RVQ sample compression, 3D Surround, FAT Bass, cubic resampling
   IMAGE TAB — iChannel0: alphabet texture (shadertoy.com/view/4sf3RB)
   Contact: subband@gmail.com or
            subband@protonmail.com
   GIT:     https://github.com/mewza/mod2glsl
  ============================================================================ */


// ============================================================
// FONT FRAMEWORK  (kishimisu "Better Text in Shaders")
// ============================================================
#define FONT_TEXTURE iChannel0
#define SPACING 1.4
#define _    32,
#define _EX   33,
#define _DBQ  34,
#define _NUM  35,
#define _DOL  36,
#define _PER  37,
#define _AMP  38,
#define _QT   39,
#define _LPR  40,
#define _RPR  41,
#define _ADD  43,
#define _COM  44,
#define _SUB  45,
#define _DOT  46,
#define _DIV  47,
#define _COL  58,
#define _SEM  59,
#define _LES  60,
#define _EQ   61,
#define _GE   62,
#define _QUE  63,
#define _AT   64,
#define _COPY 169,  // © copyright symbol
#define _LBR  91,
#define _ANTI 92,
#define _RBR  93,
#define _UN   95,
#define _0 48,
#define _1 49,
#define _2 50,
#define _3 51,
#define _4 52,
#define _5 53,
#define _6 54,
#define _7 55,
#define _8 56,
#define _9 57,
#define _A 65,
#define _B 66,
#define _C 67,
#define _D 68,
#define _E 69,
#define _F 70,
#define _G 71,
#define _H 72,
#define _I 73,
#define _J 74,
#define _K 75,
#define _L 76,
#define _M 77,
#define _N 78,
#define _O 79,
#define _P 80,
#define _Q 81,
#define _R 82,
#define _S 83,
#define _T 84,
#define _U 85,
#define _V 86,
#define _W 87,
#define _X 88,
#define _Y 89,
#define _Z 90,

#define print_char(i) \\
    texture(FONT_TEXTURE, u + vec2(float(i)-float(x)/SPACING+SPACING/8.,15-(i)/16)/16.).r

#define makeStr(fn) \\
    float fn(vec2 u) {{ \\
        if (u.x<0.||abs(u.y-.03)>.03) return 0.; \\
        const int[] str = int[](

#define _end 0); \\
    int x=int(u.x*16.*SPACING); \\
    if (x>=str.length()-1) return 0.; \\
    return print_char(str[x]); \\
}}

// ---- Python-generated strings ----
makeStr(printTitle)  {title_chars} _end
makeStr(printBPMVal) {bpm_val_chars} _end
makeStr(printSpdVal) {spd_val_chars} _end

// ---- Static label strings ----
makeStr(printHdr)   _NUM _NUM _NUM _ _G _L _S _L _ _M _O _D _ _P _L _A _Y _E _R _ _V _1 _DOT _3 _ _NUM _NUM _NUM _end
makeStr(printCredit) _COPY _ _2 _0 _2 _6 _ _O _R _B _L _I _V _I _U _S _end
makeStr(printLoad)   _L _O _A _D _I _N _G _DOT _DOT _DOT _end
makeStr(printSpec)   _S _P _E _C _T _R _U _M _ _LBR _H _O _L _D _ _M _O _U _S _E _RBR _end
makeStr(printOsci)   _O _S _C _I _L _L _O _S _C _O _P _E _end
makeStr(printNoSnd)  _S _E _T _ _I _C _H _A _N _1 _ _EQ _ _S _O _U _N _D _ _O _U _T _P _U _T _end
makeStr(printPatt)  _P _A _T _T _E _R _N _COL _end
makeStr(printRow)   _R _O _W _COL _end
makeStr(printBPM)   _B _P _M _COL _end
makeStr(printSpd)   _S _P _E _E _D _COL _end
makeStr(printTrk1)  _T _R _A _C _K _ _NUM _1 _end
makeStr(printTrk2)  _T _R _A _C _K _ _NUM _2 _end
makeStr(printTrk3)  _T _R _A _C _K _ _NUM _3 _end
makeStr(printTrk4)  _T _R _A _C _K _ _NUM _4 _end

// ============================================================
// pUV — pixel coords (y from TOP) → makeStr font UV
// sx: x of LEFT edge,  sy: y of TOP edge,  ch: char height px
// ============================================================
vec2 pUV(vec2 fp, float sx, float sy, float ch) {{
    // sy = top of character on screen (fp.y increases downward).
    // makeStr needs u.y=0 at glyph bottom = bottom of char on screen,
    // so flip: u.y = (sy + ch - fp.y) * s
    float s = 0.06 / ch;
    return vec2((fp.x - sx) * s, (sy + ch - fp.y) * s);
}}

// ---- Dynamic char renderer (same texture as makeStr) ----
float drawCh(int c, vec2 fp, float x, float y, float cw, float ch) {{
    // p = normalised position within character cell [0,1]×[0,1]
    vec2 p = (fp - vec2(x,y)) / vec2(cw,ch);
    p.y = 1.0 - p.y;                     // flip: screen y↓  →  font UV y↑
    vec2 gp = p / 16.0;                  // font-cell-space coord (used for gradients)
    // Compute gradients BEFORE the bounds clip so the 2×2 quad stays consistent
    vec2 dx = dFdx(gp), dy = dFdy(gp);
    if (p.x<0.||p.x>1.||p.y<0.||p.y>1.) return 0.0;
    // Base UV of this character in the 16×16 atlas
    vec2 uv = gp + fract(vec2(float(c), float(15 - c/16)) / 16.0);
    // Horizontal-only dilation: wider glyph, thinner stroke
    float texel = 0.4/256.0;
    float r = textureGrad(iChannel0, uv,             dx, dy).r;
    r = max(r, textureGrad(iChannel0, uv+vec2( texel,0), dx, dy).r);
    r = max(r, textureGrad(iChannel0, uv+vec2(-texel,0), dx, dy).r);
    return min(r * 1.6, 1.0);
}}

// ---- drawNum: render decimal integer, rightmost digit at (x,y) ----
// digits: how many digit places to show (e.g. 2 = "00".."99")
float drawNum(int val, int digits, float x, float y,
              float cw, float ch, vec2 fp) {{
    float r = 0.;
    for (int i = 0; i < digits; i++) {{
        int d = val % 10; val /= 10;
        r = max(r, drawCh(48+d, fp, x - float(i)*cw, y, cw, ch));
    }}
    return r;
}}

// ---- Period → note index ----
int pToNi(int p) {{
    if(p<=0) return -1;
    const int T[36]=int[](856,808,762,720,678,640,604,570,538,508,480,453,
                           428,404,381,360,339,320,302,285,269,254,240,226,
                           214,202,190,180,170,160,151,143,135,127,120,113);
    int b=0,bd=9999;
    for(int i=0;i<36;i++){{int d=abs(T[i]-p);if(d<bd){{bd=d;b=i;}}}}
    return b;
}}

// ---- Hex char ----
int hx(int v){{v&=15;return v<10?48+v:55+v;}}

// ---- ASCII at position ci in 8-char note cell "NNS1S2E1E2" ----
// NN = note+sharp+octave  S1S2 = sample hex  E1E2E3 = effect+param
// ---- 10-char note cell: "NNN SS EXX"  (note space sample space effect+param) ----
// 9-char note cell: "NNN SSEXXX" — note(3) space(1) sample(2) effect(1) param(2)
int nCell(int period, int smp, int eff, int prm, int ci) {{
    bool empty=(period==0&&smp==0&&eff==0&&prm==0);
    if (ci<3) {{
        if(empty||period==0) return 45; // '-'
        int ni=pToNi(period); if(ni<0) return 45;
        int st=ni%12, oc=ni/12+1;
        const int NL[12]=int[](67,67,68,68,69,70,70,71,71,65,65,66);
        const int SH[12]=int[](0,1,0,1,0,0,1,0,1,0,1,0);
        if(ci==0) return NL[st];
        if(ci==1) return SH[st]!=0?35:45;
        return 48+oc;
    }}
    if(ci==3) return 32; // space after note
    if(ci==4) return hx((smp>>4)&15);
    if(ci==5) return hx(smp&15);
    if(ci==6) return hx(eff);
    if(ci==7) return hx((prm>>4)&15);
    return hx(prm&15);  // ci==8
}}

// ---- Line helpers ----
float hline(vec2 fp,float y,float x0,float x1){{
    return abs(fp.y-y)<.6&&fp.x>x0&&fp.x<x1?1.:0.;
}}
float vline(vec2 fp,float x,float y0,float y1){{
    return abs(fp.x-x)<.6&&fp.y>y0&&fp.y<y1?1.:0.;
}}

// === Zuvuya Visualizer (c) 2026 Orblivius ===
#define FFT_N 256
float _vrand(vec2 s){{return fract(sin(dot(s,vec2(12.9898,78.233)))*43758.5453);}}
float _vpow(float s,float x){{return s-(s-s*s)*(-x);}}
float _vstar(vec2 uv,float fl){{float d=length(uv),m=.05/d;m+=m*fl;m*=smoothstep(.85,.2,d);return m;}}
float _vsl(vec2 uv){{vec2 gv=fract(uv)-.5,id=floor(uv);float c=0.;
    for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){{vec2 o=vec2(x,y);float n=_vrand(id+o),sz=fract(n*345.32);
    c+=_vstar(gv-o-vec2(n,fract(n*34.))+.5,smoothstep(.9,1.,sz)*6.)*(sin(iTime*2.+n*13.256)*.5+.5)*sz;}}return c;}}
float _vboxinf(vec2 p){{vec2 q=abs(p);return max(q.x+.5,q.y)-1.;}}
float _vbox(vec3 p,vec3 b){{vec3 q=abs(p)-b;return length(max(q,0.))+min(max(q.y,max(q.x,q.z)),0.);}}
float _vmap(vec3 p){{vec3 q=p;q.xz=mod(q.xz,1.)-.5;float h=abs(_vrand(floor(p.xz)+_vrand(floor(p.xz))));
    float id=floor(_vpow(abs(p.x*.1),-1.));return min(_vboxinf(p.xy),_vbox(q,vec3(.15,1.7*h+id,.15)));}}
vec3 _vhsv(vec3 c){{vec4 K=vec4(1.,2./3.,1./3.,3.);vec3 p=abs(fract(c.xxx+K.xyz)*6.-K.www);
    return c.z*mix(K.xxx,clamp(p-K.xxx,0.,1.),c.y);}}
vec3 _visBG(vec2 fc){{
    vec2 uv=(fc*2.-iResolution.xy)/min(iResolution.x,iResolution.y);
    vec3 ray=vec3(0.,1.5,1.-iTime),dir=normalize(cross(vec3(0.,0.,-1.),vec3(0.,1.,0.))*uv.x+
             vec3(0.,1.,0.)*uv.y+vec3(0.,0.,-1.)*1.8);
    int march=0;float rLen=0.,tot=0.;
    for(int i=0;i<38;++i){{float d=_vmap(ray);march=i;tot+=d;if(d<.001||tot>60.)break;
        rLen+=min(min(min((step(0.,dir.x)-fract(ray.x))/dir.x,(step(0.,dir.y)-fract(ray.y))/dir.y)+.01,
                      (step(0.,dir.z)-fract(ray.z))/dir.z)+.01,d);ray=vec3(0.,1.5,1.-iTime)+dir*rLen;}}
    float fog=float(march)/108.;vec3 fog2=vec3(tot*.01);
    vec3 city=vec3(.05,.5,2.)*fog+fog2*vec3(0.,.5,.1),stars=vec3(0.);
    for(float i=0.;i<=1.;i+=.25){{float depth=fract(i+iTime*.2),sc=mix(20.,.5,depth);
        float fade=depth*smoothstep(1.,.9,depth);
        stars+=vec3(_vsl(uv*sc+i*432.)*fade);}}
    stars*=abs(uv.y*.5)*vec3(.5,.5,1.);
    return mix(city,stars,clamp(fog2*1.5,0.,1.))*fog;}}
vec3 _visCurtain(vec2 s,float a0,float a1,float a2,float a3){{
    float chA[4];chA[0]=a0;chA[1]=a1;chA[2]=a2;chA[3]=a3;
    float per=2./max(abs(s.y),.12);vec3 col=vec3(0.);
    for(float z=0.;z<1.;z+=.1){{
        int ch=int(z*4.)%NUM_CHANNELS;
        float wm=chA[ch]*abs(sin(z*9.42+iTime*2.1+float(ch)*1.57));
        vec2 p=vec2(s.x*(1.+z),s.y+(1.+z))*per;p.y+=2.8*iTime;
        p.x+=cos(z/.06)+wm*sin(p.y*3.14159*3.)*z*2.;p.y+=wm*2.*z;
        float w=p.x,l=sin(p.y*.5+z/.08+3.4*iTime);
        float heat=clamp(wm*2.,0.,1.);
        float intensity=exp(min(l,-l/mix(.3,.05,heat)/(1.+4.*w*w)));
        vec3 tint=_vhsv(vec3(float(ch)*.25+.05,mix(.6,1.,heat),mix(.5,1.4,heat)));
        tint+=vec3(.15,0.,.25)*smoothstep(.3,.7,sin(z*30.+iTime*.7))*(1.-heat);
        col+=intensity*tint/(abs(w)+.01*per)*per;}}
    return tanh(col*col/2e3);}}

void mainImage(out vec4 O, vec2 C) {{
    vec2 fp = vec2(C.x, iResolution.y - C.y);

    const vec3 BG     = vec3(0.00,0.00,0.08);
    const vec3 CYAN   = vec3(0.20,0.90,1.00);
    const vec3 YELLOW = vec3(1.00,0.90,0.10);
    const vec3 RED    = vec3(1.00,0.28,0.05);
    const vec3 WHITE  = vec3(0.88,0.88,0.96);
    const vec3 BLUE   = vec3(0.30,0.55,1.00);
    const vec3 GREEN  = vec3(0.10,1.00,0.22);
    const vec3 DIM    = vec3(0.22,0.22,0.42);
    const vec3 TC0    = vec3(0.10,1.00,0.90);
    const vec3 TC1    = vec3(1.00,0.90,0.10);
    const vec3 TC2    = vec3(1.00,0.45,0.90);
    const vec3 TC3    = vec3(1.00,0.55,0.10);

    const float CH=28., CW=25., ML=10.;
    const int LOADING_FRAMES = 16;

    // ── Per-channel amplitudes → Zuvuya curtain ───────────────────────────
    float _tps=float(BPM)*2./5., _rt=float(SPEED)/_tps;
    Position _pos=getPosition(iTime);
    float _a0=abs(getChannelOutput(0,iTime,_pos,_rt));
    float _a1=abs(getChannelOutput(1,iTime,_pos,_rt));
    float _a2=abs(getChannelOutput(2,iTime,_pos,_rt));
    float _a3=abs(getChannelOutput(3,iTime,_pos,_rt));
    vec2 _uv=(C*2.-iResolution.xy)/iResolution.y;
    vec3 col = _visCurtain(vec2(_uv.y,abs(_uv.x)),_a0,_a1,_a2,_a3) + _visBG(C);

    if (iFrame < LOADING_FRAMES) {{
        vec2 res = iResolution.xy;
        float prog = float(iFrame) / float(LOADING_FRAMES - 1);

        // (data arrays paged in by compiler — no runtime loop needed)

        // Header
        col += CYAN   * printHdr   (pUV(fp, ML, 6., CH));
        col += WHITE  * printCredit(pUV(fp, res.x - ML - 250., 6., CH*0.75));
        col += YELLOW * printTitle (pUV(fp, ML, CH+9., CH));

        // "LOADING..." label
        col += WHITE * printLoad(pUV(fp, res.x*0.5 - 4.*CW, res.y*0.5 - CH*2., CH));

        // Progress bar
        float barX0 = res.x * 0.1, barX1 = res.x * 0.9;
        float barY  = res.y * 0.5, barH  = CH * 0.5;
        if (fp.y > barY - barH && fp.y < barY + barH &&
            fp.x > barX0 && fp.x < barX1) {{
            float fill    = barX0 + (barX1 - barX0) * prog;
            vec3 fillCol  = mix(CYAN, GREEN, prog);
            col = fp.x < fill ? fillCol : DIM * 0.5;
            float edge = abs(fp.x - fill);
            if (edge < 3.0) col = mix(WHITE, fillCol, edge / 3.0);
        }}

        col += DIM * drawNum(iFrame, 2, res.x*0.5 + 5.*CW, res.y*0.5 + CH*1.5, CW, CH, fp);
        O = vec4(col, 1.0);
        return;
    }}
    // ── End loading screen ─────────────────────────────────────────────────

    // getPosition() loops via mod(time, songDuration) — GUI matches audio loop
    Position pos = getPosition(iTime);
    // Accumulate tracker UI into separate buffer so it overlays pure (no curtain tint)
    vec3 trk = vec3(0.0);
    trk += CYAN   * printHdr   (pUV(fp, ML,  6., CH));
    trk += WHITE  * printCredit(pUV(fp, iResolution.x - ML - 250., 6., CH*0.75));
    trk += YELLOW * printTitle (pUV(fp, ML, CH+9., CH));
    trk += DIM    * hline(fp, CH*2.+13., ML, iResolution.x-ML);

    // ============ INFO BAR ============
    float iy  = CH*2.+20.;
    float iy2 = iy + CH + 4.;
    float rx  = iResolution.x*0.52;

    trk += BLUE  * printPatt(pUV(fp, ML, iy, CH));
    trk += WHITE * drawNum(pos.songPos, 2, ML+10.*CW, iy, CW,CH,fp);
    trk += DIM   * drawCh(47,fp, ML+11.*CW, iy, CW,CH);
    trk += DIM   * drawNum({mod.num_patterns-1}, 2, ML+13.*CW, iy, CW,CH,fp);

    trk += BLUE  * printRow(pUV(fp, rx, iy, CH));
    trk += WHITE * drawNum(pos.row, 2, rx+5.*CW, iy, CW,CH,fp);
    trk += DIM   * drawCh(47,fp, rx+6.*CW, iy, CW,CH);
    trk += DIM   * drawCh(54,fp, rx+7.*CW, iy, CW,CH);
    trk += DIM   * drawCh(52,fp, rx+8.*CW, iy, CW,CH);

    trk += BLUE   * printBPM(pUV(fp, ML,  iy2, CH));
    trk += YELLOW * printBPMVal(pUV(fp, ML+5.*CW, iy2, CH));
    trk += BLUE   * printSpd(pUV(fp, rx,  iy2, CH));
    trk += YELLOW * printSpdVal(pUV(fp, rx+7.*CW, iy2, CH));

    trk += DIM * hline(fp, iy2+CH+4., ML, iResolution.x-ML);

    // ============ TRACKER ============
    float ty   = iy2+CH+10.;
    float TW   = 9.*CW+6.;    // 9 chars per cell + 6px gap
    float rNW  = 2.*CW;
    float txOff= ML+rNW+8.;
    const int HVR = 4;  // 9 visible rows → more room for oscilloscope

    // Track headers (each in its own color)
    trk += TC0 * printTrk1(pUV(fp, txOff+0.*TW, ty, CH));
    trk += TC1 * printTrk2(pUV(fp, txOff+1.*TW, ty, CH));
    trk += TC2 * printTrk3(pUV(fp, txOff+2.*TW, ty, CH));
    trk += TC3 * printTrk4(pUV(fp, txOff+3.*TW, ty, CH));

    // Vertical separators
    for(int tc=1;tc<NUM_CHANNELS;tc++)
        trk += DIM*0.7*vline(fp, txOff+float(tc)*TW-4., ty, ty+float(2*HVR+3)*CH);

    float tTop = ty+CH+3.;
    float tBot = tTop+float(2*HVR+1)*CH;

    // Page mode: show a fixed page of rows, frame moves line by line within it
    // Page flips when frame hits bottom → jumps to top of next page
    int pageSize = 2*HVR+1;                             // e.g. 9 rows visible
    int pageStart = (pos.row / pageSize) * pageSize;    // first row of current page
    int frameRow  = pos.row - pageStart;                // frame position within page (0..pageSize-1)

    // ── Tracker background: highlight row = solid blue (inset 2px), others = zebra ─
    float _inset = 2.0;
    float _hX0 = ML - _inset;
    float _hX1 = iResolution.x - ML + _inset;
    if(fp.y>=tTop && fp.y<tBot) {{
        int ri_z = int((fp.y-tTop)/CH);
        float _rowY0 = tTop + float(ri_z)*CH + _inset;
        float _rowY1 = tTop + float(ri_z+1)*CH - _inset;
        if(ri_z == frameRow && fp.y>=_rowY0 && fp.y<_rowY1 && fp.x>=_hX0 && fp.x<_hX1) {{
            col = vec3(0.12, 0.38, 0.72);            // solid blue inset highlight
        }} else if((ri_z & 1) == 0) {{
            col = mix(col, vec3(0.0), 0.38);         // zebra darker rows
        }}
    }}

    if(fp.y>=tTop && fp.y<tBot) {{
        int ri_abs=int((fp.y-tTop)/CH);
        int ri=ri_abs - frameRow;

        // Row number — digits tightened together (CW*0.75 spacing)
        {{
            int rn = pageStart + ri_abs;
            if(rn<0)   rn+=64;
            if(rn>=64) rn-=64;
            bool on4 = (rn%4)==0;
            vec3 rnc = ri_abs==frameRow ? WHITE : (on4 ? YELLOW*0.7 : RED*0.7);
            trk += rnc * drawNum(rn, 2, ML+rNW-CW*1.2, tTop+float(ri_abs)*CH, CW*0.75,CH,fp);
        }}

        // Per-track note data
        float xInT=fp.x-txOff;
        if(xInT>=0.&&xInT<float(NUM_CHANNELS)*TW) {{
            int tc =int(xInT/TW);
            int ci =int((xInT-float(tc)*TW)/CW);
            if(ci<9) {{
                int rn = pageStart + ri_abs;
                int sp = pos.songPos;
                if(rn<0)  {{ rn+=64; sp=max(0,sp-1); }}
                if(rn>=64){{ rn-=64; sp=min(SONG_LENGTH-1,sp+1); }}
                Note n=getNote(sp,rn,tc);
                int c=nCell(n.period,n.instrument,n.effect,n.param,ci);
                float g=drawCh(c, fp,
                    txOff+float(tc)*TW+float(ci)*CW,
                    tTop+float(ri_abs)*CH, CW,CH);
                bool isEmpty=(n.period==0&&n.instrument==0&&n.effect==0);
                float fade=max(0.25,1.0-float(abs(ri))*0.1);
                // Per-track color, dimmed by distance from current row
                const vec3 TCols[4]=vec3[](TC0,TC1,TC2,TC3);
                vec3 nc;
                if(isEmpty) nc = DIM*fade;
                else        nc = (ri==0 ? TCols[tc] : TCols[tc]*fade);
                trk += nc*g;
            }}
        }}
    }}
    trk += DIM*hline(fp, tBot+3., ML, iResolution.x-ML);

    // ── Composite: text always on top at full color ────────────────────────
    col += trk;

    // ============ OSCILLOSCOPE ============
    // ============ OSCILLOSCOPE / SPECTRUM (mouse click toggles) ============
    // iChannel1 = Sound tab output → gives waveform (y=0.25) and FFT (y=0.75)
    // Hold/click mouse anywhere to switch to spectrum view
    float oy=tBot+8.;
    float oh=max(0., iResolution.y-oy-10.);
    // specMode persisted in Buffer A row 2, px 0 (click to toggle)
    bool specMode = (iChannelResolution[1].x > 1.0)
                    ? texelFetch(iChannel1, ivec2(0, 2), 0).r > 0.5
                    : (iMouse.z > 0.0);  // fallback if Buffer A not connected

    if(oh>20.&&fp.y>oy&&fp.y<oy+oh) {{
        float sy = fp.y - oy;

        if (!specMode) {{
            // ── Oscilloscope ────────────────────────────────────────────
            float window = 0.030;
            float ticksPerSecO = BPM * 2.0 / 5.0;
            float rowTimeO = SPEED / ticksPerSecO;
            float oscT = iTime + (C.x/iResolution.x - 0.5) * window;
            Position oscPos = getPosition(oscT);
            float mono = 0.0;
            for (int ch = 0; ch < NUM_CHANNELS; ch++)
                mono += getChannelOutput(ch, oscT, oscPos, rowTimeO);
            mono /= float(NUM_CHANNELS);

            float amp   = clamp(mono * 4.0, -0.9, 0.9);
            float waveY = (amp * 0.46 + 0.5) * oh;
            float slope = abs(dFdx(waveY));
            float thick = max(1.5, slope * 0.8);
            col = mix(col, DIM*0.18, step(abs(sy - oh*0.5), 0.4));
            float dist = abs(sy - waveY);
            if (dist < thick) {{
                float t = dist / thick;
                col = mix(CYAN, col, t * t * 0.6);
            }}
        }} else {{
            // ── FFT Spectrum from Buffer A (iChannel1) ───────────────────
            // Buffer A must be set up: Image iChannel1 = Buffer A output
            if (iChannelResolution[1].x > 1.0) {{
                // Buffer A connected — read DFT magnitudes
                float xf   = C.x / iResolution.x;
                float logX = pow(xf, 0.5);  // perceptual log-scale
                int   bin  = clamp(int(logX * float(FFT_N / 2)), 0, FFT_N/2 - 1);
                float mag  = texelFetch(iChannel1, ivec2(bin, 1), 0).r;
                float barH = clamp(mag * 3.0, 0.0, 1.0) * oh;
                float barY = oh - barH;
                if (sy > barY) {{
                    float t2 = (sy - barY) / max(barH, 1.0);
                    vec3 specCol = mix(CYAN, BLUE, sqrt(t2));
                    specCol = mix(specCol, mix(GREEN, YELLOW, xf), xf * 0.35);
                    col = specCol;
                }} else {{
                    col = mix(col, BG, 0.4);
                }}
                float gridX = step(0.96, fract(C.x * 8.0 / iResolution.x));
                col = mix(col, DIM * 0.4, gridX * 0.4);
                col += DIM * 0.6 * printSpec(pUV(fp, ML, oy+oh-CH-4., CH*0.8));
                O = vec4(col, 1.0); return;
            }}
            // Buffer A not connected — fall through to note-freq display
            {{ // Note-frequency spectrum: draw a bar for each active note ──
            // Map pixel X → log-frequency [27Hz..4186Hz] (A0..C8 piano range)
            float logMin = log2(27.5), logMax = log2(4186.0);
            float logFreq = logMin + (C.x / iResolution.x) * (logMax - logMin);
            float pixFreq = pow(2.0, logFreq);
            const vec3 TCols[4] = vec3[](TC0, TC1, TC2, TC3);
            float ticksPerSecV = BPM * 2.0 / 5.0;
            float rowTimeV = SPEED / ticksPerSecV;
            float totalBar = 0.0;
            vec3  barColor = BG;
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
                // Find active note period → frequency
                Note tn = getNote(pos.songPos, pos.row, ch);
                int trow = pos.row, tpat = pos.songPos;
                if (tn.instrument <= 0 || tn.period <= 0) {{
                    int sr = pos.row, sp2 = pos.songPos;
                    for (int lb = 1; lb < 48; lb++) {{
                        sr--;
                        if (sr < 0) {{ if (sp2>0) {{sp2--; sr=63;}} else break; }}
                        Note pn = getNote(sp2, sr, ch);
                        if (pn.instrument > 0 && pn.period > 0) {{
                            tn = pn; trow = sr; tpat = sp2; break;
                        }}
                    }}
                }}
                if (tn.period <= 0) continue;
                float noteFreq = periodToFreq(tn.period);
                // Get volume for this channel
                SampleInfo si = samples[tn.instrument - 1];
                int vol = si.volume;
                Note cr = getNote(pos.songPos, pos.row, ch);
                if (cr.effect == 0xC) vol = min(cr.param, 64);
                else if (tn.effect == 0xC) vol = min(tn.param, 64);
                float amp = float(vol) / 64.0;
                // Bar width = 1 semitone = log2(noteFreq) ± 0.5/12
                float semitone = 0.5 / 12.0;
                float dist = abs(log2(pixFreq) - log2(noteFreq));
                if (dist < semitone * 1.5) {{
                    float edge = 1.0 - dist / (semitone * 1.5);
                    totalBar = max(totalBar, amp * edge);
                    barColor = TCols[ch];
                }}
            }}
            // Draw bar
            float barH = totalBar * oh * 0.92;
            float barY = oh - barH;
            if (sy > barY && barH > 2.0) {{
                float t = (sy - barY) / max(barH, 1.0);
                col = mix(WHITE, barColor, t * 0.8);
            }} else {{
                col = mix(col, BG, 0.3);
            }}
            // Faint piano-key grid
            float semPos = logFreq * 12.0;
            float isBlack = step(0.5, fract(semPos));
            col = mix(col, col * 0.85, isBlack * 0.15);
            // Octave lines
            float octLine = step(0.97, fract(logFreq));
            col = mix(col, DIM * 0.6, octLine * 0.5);
        }} // end note-freq inner block
        }}  // end spectrum else

        // Label: show mode name
        if (fp.y > oy + oh - CH - 4. && fp.y < oy + oh - 4.) {{
            if (specMode)
                col += GREEN  * 0.8 * printSpec(pUV(fp, ML, oy+oh-CH-4., CH*0.8));
            else
                col += CYAN   * 0.6 * printOsci(pUV(fp, ML, oy+oh-CH-4., CH*0.8));
        }}
    }}


    // Outer border
    float bx0=ML-2.,bx1=iResolution.x-ML+2.,by0=2.,by1=oy+oh+6.;
    if(fp.x>bx0&&fp.x<bx1&&fp.y>by0&&fp.y<by1) {{
        float e=step(abs(fp.x-bx0),1.)+step(abs(fp.x-bx1),1.)+
                step(abs(fp.y-by0),1.)+step(abs(fp.y-by1),1.);
        col=mix(col,BLUE,clamp(e,0.,1.));
    }}

    O = vec4(col, 1.0);
}}
"""


    # ========== BUFFER A TAB — FFT Spectrum Analyzer ==========
    # Row 0 (y=0): FFT_N audio samples collected via getChannelOutput
    # Row 1 (y=1): FFT_N/2 DFT magnitude bins (phasor-rotation, reads row 0 prev frame)
    # Setup: Buffer A iChannel0 = Buffer A (self-ref)
    #        Image   iChannel1 = Buffer A output
    buffer_a_glsl = f"""/* ============================================================================
   GLSL MOD Player v1.3 (c) 2026 Orblivius
   RVQ sample compression, 3D Surround, FAT Bass, cubic resampling
   Contact: subband@gmail.com or
            subband@protonmail.com
   GIT:     https://github.com/mewza/mod2glsl
   BUFFER A TAB
   Row 0      : FFT_N mixed audio samples      (getChannelOutput sum, per px)
   Row 1      : FFT_N/2 DFT magnitudes         (phasor-rotation DFT)
   Row 2      : UI state px0=specMode px1=prevMouse
   Rows 3-66  : Per-channel oscilloscope history  (4 px wide, one per channel)
                Row 3 = newest frame, rows scroll downward each frame.
                waveMem for Zuvuya curtain reads from here via Image iChannel1.
   ShaderToy setup:
     Buffer A -> iChannel0 = Buffer A  (self-reference)
     Buffer A -> iChannel1 = Sound tab output  (audio waveform for Zuvuya)
     Image    -> iChannel1 = Buffer A
   ============================================================================ */

#define FFT_N     256
#define FFT_SR    8192.0
#define HIST_ROWS 64      // rows 3..(3+HIST_ROWS-1)
#define HIST_BASE 3
#define WAVE_BASE 70      // rows 70..(70+WAVE_ROWS-1) — Zuvuya waveform scroll memory
#define WAVE_ROWS 64      // 64 rows of history, full-width x

void mainImage(out vec4 O, vec2 C) {{
    int px = int(C.x), py = int(C.y);
    O = vec4(0.0);

    if (py == 0 && px < FFT_N) {{
        // ── Row 0: mixed audio sample at time-offset px ────────────────────
        float dt  = 1.0 / FFT_SR;
        float t   = iTime - float(FFT_N - px - 1) * dt;
        float ticksPerSec = float(BPM) * 2.0 / 5.0;
        float rowTime = float(SPEED) / ticksPerSec;
        Position pos = getPosition(t);
        float s = 0.0;
        for (int ch = 0; ch < NUM_CHANNELS; ch++)
            s += getChannelOutput(ch, t, pos, rowTime);
        s /= float(NUM_CHANNELS);
        O = vec4(s, 0.0, 0.0, 1.0);

    }} else if (py == 1 && px < FFT_N / 2) {{
        // ── Row 1: DFT magnitude at bin px ─────────────────────────────────
        const float TWO_PI = 6.28318530718;
        int   k  = px;
        float dk = TWO_PI * float(k) / float(FFT_N);
        float delta_re = cos(-dk), delta_im = sin(-dk);
        float phase_re = 1.0,     phase_im = 0.0;
        float re = 0.0, im = 0.0;
        for (int n = 0; n < FFT_N; n++) {{
            float s = texelFetch(iChannel0, ivec2(n, 0), 0).r;
            float w = 0.5 * (1.0 - cos(TWO_PI * float(n) / float(FFT_N)));
            re += s * w * phase_re;
            im += s * w * phase_im;
            float nr = phase_re * delta_re - phase_im * delta_im;
            float ni = phase_re * delta_im + phase_im * delta_re;
            phase_re = nr; phase_im = ni;
        }}
        float mag = 15.0 * sqrt(re*re + im*im) / float(FFT_N);
        O = vec4(mag, 0.0, 0.0, 1.0);

    }} else if (py == 2) {{
        // ── Row 2: UI state ─────────────────────────────────────────────────
        float prevMode  = texelFetch(iChannel0, ivec2(0, 2), 0).r;
        float prevMouse = texelFetch(iChannel0, ivec2(1, 2), 0).r;
        float currMouse = iMouse.z > 0.0 ? 1.0 : 0.0;
        bool  newClick  = (currMouse > 0.5 && prevMouse < 0.5);
        if (px == 0)      O = vec4(newClick ? 1.0 - prevMode : prevMode, 0., 0., 1.);
        else if (px == 1) O = vec4(currMouse, 0., 0., 1.);

    }} else if (py >= HIST_BASE && py < HIST_BASE + HIST_ROWS && px < NUM_CHANNELS) {{
        // ── Rows 3–66: per-channel oscilloscope history ─────────────────────
        if (py == HIST_BASE) {{
            float ticksPerSec = float(BPM) * 2.0 / 5.0;
            float rowTime = float(SPEED) / ticksPerSec;
            Position pos = getPosition(iTime);
            float s = getChannelOutput(px, iTime, pos, rowTime);
            O = vec4(0., 0., 0., abs(s) * 0.5 + 0.5);
        }} else {{
            O = texelFetch(iChannel0, ivec2(px, py - 1), 0);
        }}

    }} else if (py == WAVE_BASE) {{
        // ── Row 70: newest row — RAW BIPOLAR audio, no abs, no envelope ──────
        // Different x-columns sample different audio phases via mu*0.5 time offset.
        // Positive phase → waveMem near 1.0 (hot), negative → 0.0 (cold blue).
        // This is what makes z-bands independent: each band samples a different
        // audio phase, giving genuinely different waveMem → different displacement.
        float u  = float(px) / float(iResolution.x);
        float mu = 1.0 - abs(u * 2.0 - 1.0);
        float t  = iTime - mu * 0.5;
        float ticksPerSec = float(BPM) * 2.0 / 5.0;
        float rowTime = float(SPEED) / ticksPerSec;
        Position pos = getPosition(t);
        float s = 0.0;
        for (int ch = 0; ch < NUM_CHANNELS; ch++)
            s += getChannelOutput(ch, t, pos, rowTime);
        s /= float(NUM_CHANNELS);
        O = vec4(0.0, 0.0, 0.0, clamp(s * 1.5 + 0.5, 0.0, 1.0));

    }} else if (py > WAVE_BASE && py < WAVE_BASE + WAVE_ROWS) {{
        // ── Rows 71-133: scroll — copy row above (one frame younger) ────────
        O = texelFetch(iChannel0, ivec2(px, py - 1), 0);
    }}
}}
"""
    # Write Buffer A file
    bufA_file = output_file.replace('_shadertoy.glsl', '_shadertoy_bufferA.glsl') \
                           .replace('_shadertoy_common.glsl', '_shadertoy_bufferA.glsl')
    if '_shadertoy' not in bufA_file:
        bufA_file = output_file.replace('.glsl', '_shadertoy_bufferA.glsl')
    with open(bufA_file, 'w') as f:
        f.write(buffer_a_glsl)

    # Write 3 ShaderToy tabs: Common, Sound, Image
    with open(output_file.replace('.glsl', '_common.glsl'), 'w') as f:
        f.write(common_glsl)

    with open(output_file.replace('.glsl', '_sound.glsl'), 'w') as f:
        f.write(sound_glsl)
    
    with open(output_file.replace('.glsl', '_image.glsl'), 'w') as f:
        f.write(image_glsl)
    
    print(f"   📁 Created ShaderToy tabs:")
    print(f"      Common:   {output_file.replace('.glsl', '_common.glsl')}")
    print(f"      Sound:    {output_file.replace('.glsl', '_sound.glsl')}")
    print(f"      Image:    {output_file.replace('.glsl', '_image.glsl')}")
    print(f"      Buffer A: {bufA_file}  (FFT spectrum + click toggle state)")
    print()
    print(f"   🔗 ShaderToy channel setup:")
    print(f"      Image    → iChannel0 = Alphabet texture (shadertoy.com/view/4sf3RB)")
    print(f"      Image    → iChannel1 = Buffer A")
    print(f"      Buffer A → iChannel0 = Buffer A  (self-reference)")
    print(f"      Buffer A → iChannel1 = Sound tab  (audio waveform for Zuvuya)")
    print(f"      Sound    → (no channels needed)")
    print()
    print(f"   🖱️  Click anywhere to toggle oscilloscope ↔ spectrum view")


# Embedded VQ encoder (base64-encoded to avoid string escaping issues).
# Contains vq_encoder_v2 + getchanneloutput.glsl bundled — no external files needed.
_VQ_ENCODER_B64 = (
    "IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiIKTU9EIOKGkiBTaGFkZXJUb3kgQ29tbW9uIHRhYiBlbmNv"
    "ZGVyIHdpdGg6CiAgLSBQYXR0ZXJuIGNydW5jaDogYml0bWFwICsgZGljdGlvbmFyeSArIG5pYmJsZS1w"
    "YWNrZWQgcm93IHNlZWsKICAtIFNhbXBsZSBjcnVuY2g6IDMtYml0IGxpbmVhciBwYWNrZWQgKHVuaWZv"
    "cm0gbm9pc2UgZmxvb3Ig4oCUIHN0YWJsZSBhY3Jvc3MgYWxsIHBsYXliYWNrIHBpdGNoZXMpClRhcmdl"
    "dDog4omkIDY0IEtCIHRvdGFsIHByaXZhdGUgY29uc3QgZGF0YSAoTWFjIEFOR0xFL01ldGFsIHNhZmUg"
    "em9uZSkKIiIiCmltcG9ydCBzdHJ1Y3QsIHN5cywgb3MKCmNsYXNzIE1PREZpbGU6CiAgICBkZWYgX19p"
    "bml0X18oc2VsZiwgcGF0aCk6CiAgICAgICAgd2l0aCBvcGVuKHBhdGgsICdyYicpIGFzIGY6CiAgICAg"
    "ICAgICAgIHNlbGYuZGF0YSA9IGYucmVhZCgpCiAgICAgICAgc2VsZi5wYXJzZSgpCgogICAgZGVmIHBh"
    "cnNlKHNlbGYpOgogICAgICAgIGQgPSBzZWxmLmRhdGEKICAgICAgICBzZWxmLnRpdGxlID0gZFswOjIw"
    "XS5yc3RyaXAoYidceDAwJykuZGVjb2RlKCdsYXRpbjEnLCAncmVwbGFjZScpCiAgICAgICAgc2VsZi5z"
    "YW1wbGVzX2luZm8gPSBbXQogICAgICAgIGZvciBpIGluIHJhbmdlKDMxKToKICAgICAgICAgICAgYmFz"
    "ZSA9IDIwICsgaSozMAogICAgICAgICAgICBuYW1lID0gZFtiYXNlOmJhc2UrMjJdLnJzdHJpcChiJ1x4"
    "MDAnKS5kZWNvZGUoJ2xhdGluMScsICdyZXBsYWNlJykKICAgICAgICAgICAgbGVuZ3RoX3cgICAgID0g"
    "c3RydWN0LnVucGFjaygnPkgnLCBkW2Jhc2UrMjI6YmFzZSsyNF0pWzBdCiAgICAgICAgICAgIGZpbmV0"
    "dW5lICAgICA9IGRbYmFzZSsyNF0gJiAweDBGCiAgICAgICAgICAgIHZvbHVtZSAgICAgICA9IGRbYmFz"
    "ZSsyNV0KICAgICAgICAgICAgbG9vcF9zdGFydF93ID0gc3RydWN0LnVucGFjaygnPkgnLCBkW2Jhc2Ur"
    "MjY6YmFzZSsyOF0pWzBdCiAgICAgICAgICAgIGxvb3BfbGVuX3cgICA9IHN0cnVjdC51bnBhY2soJz5I"
    "JywgZFtiYXNlKzI4OmJhc2UrMzBdKVswXQogICAgICAgICAgICBzZWxmLnNhbXBsZXNfaW5mby5hcHBl"
    "bmQoZGljdCgKICAgICAgICAgICAgICAgIG5hbWU9bmFtZSwgbGVuZ3RoPWxlbmd0aF93KjIsIGZpbmV0"
    "dW5lPWZpbmV0dW5lLAogICAgICAgICAgICAgICAgdm9sdW1lPXZvbHVtZSwgbG9vcF9zdGFydD1sb29w"
    "X3N0YXJ0X3cqMiwgbG9vcF9sZW49bG9vcF9sZW5fdyoyKSkKICAgICAgICBzZWxmLnNvbmdfbGVuZ3Ro"
    "ID0gZFs5NTBdCiAgICAgICAgc2VsZi5wYXR0ZXJuX29yZGVyID0gbGlzdChkWzk1Mjo5NTIrMTI4XSkK"
    "ICAgICAgICBzZWxmLm1hZ2ljID0gZFsxMDgwOjEwODRdCiAgICAgICAgc2VsZi5udW1fcGF0dGVybnMg"
    "PSBtYXgoc2VsZi5wYXR0ZXJuX29yZGVyWzpzZWxmLnNvbmdfbGVuZ3RoXSkgKyAxCiAgICAgICAgIyBQ"
    "YXR0ZXJucwogICAgICAgIHNlbGYucGF0dGVybnMgPSBbXQogICAgICAgIG9mZiA9IDEwODQKICAgICAg"
    "ICBmb3IgcCBpbiByYW5nZShzZWxmLm51bV9wYXR0ZXJucyk6CiAgICAgICAgICAgIHNlbGYucGF0dGVy"
    "bnMuYXBwZW5kKGRbb2ZmOm9mZisxMDI0XSkKICAgICAgICAgICAgb2ZmICs9IDEwMjQKICAgICAgICAj"
    "IFNhbXBsZXMgKHJhdyBzaWduZWQgOC1iaXQgYnl0ZXMpCiAgICAgICAgc2VsZi5zYW1wbGVfYnl0ZXMg"
    "PSBbXQogICAgICAgIGZvciBzIGluIHNlbGYuc2FtcGxlc19pbmZvOgogICAgICAgICAgICBzZWxmLnNh"
    "bXBsZV9ieXRlcy5hcHBlbmQoZFtvZmY6b2ZmK3NbJ2xlbmd0aCddXSkKICAgICAgICAgICAgb2ZmICs9"
    "IHNbJ2xlbmd0aCddCgojIOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkAojIFBBVFRF"
    "Uk4gQ1JVTkNIOiBiaXRtYXAgKyBkaWN0ICsgbmliYmxlLXNlZWsKIyDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZAKCkVNUFRZX05PVEUgPSBiJ1x4MDBceDAwXHgwMFx4MDAnCgpkZWYgZW5j"
    "b2RlX3BhdHRlcm5zKG1vZCk6CiAgICAiIiJSZXR1cm5zIGRpY3Qgb2YgYWxsIHBhdHRlcm4gZGF0YSBz"
    "dHJ1Y3R1cmVzLiIiIgogICAgIyBCdWlsZCBmbGF0IGxpc3Qgb2YgNC1ieXRlIG5vdGVzIGluIG9yZGVy"
    "OiBwYXQgMC4uTi0xLCByb3cgMC4uNjMsIGNoIDAuLjMKICAgIG5vdGVzID0gW10KICAgIGZvciBwYXQg"
    "aW4gcmFuZ2UobW9kLm51bV9wYXR0ZXJucyk6CiAgICAgICAgcGRhdGEgPSBtb2QucGF0dGVybnNbcGF0"
    "XQogICAgICAgIGZvciByb3cgaW4gcmFuZ2UoNjQpOgogICAgICAgICAgICBmb3IgY2ggaW4gcmFuZ2Uo"
    "NCk6CiAgICAgICAgICAgICAgICBiYXNlID0gcm93KjE2ICsgY2gqNAogICAgICAgICAgICAgICAgbm90"
    "ZXMuYXBwZW5kKHBkYXRhW2Jhc2U6YmFzZSs0XSkKICAgIHRvdGFsX25vdGVzID0gbGVuKG5vdGVzKQog"
    "ICAgbnVtX3Jvd3MgICAgPSBtb2QubnVtX3BhdHRlcm5zICogNjQKCiAgICAjIFVuaXF1ZSBub24tZW1w"
    "dHkgbm90ZXMg4oaSIGRpY3Rpb25hcnkKICAgIHVuaXEgPSBzb3J0ZWQoc2V0KG4gZm9yIG4gaW4gbm90"
    "ZXMgaWYgbiAhPSBFTVBUWV9OT1RFKSkKICAgIGlkeF9ieXRlcyA9IDEgaWYgbGVuKHVuaXEpIDw9IDI1"
    "NiBlbHNlIDIKICAgIGFzc2VydCBsZW4odW5pcSkgPD0gNjU1MzYsIGYidG9vIG1hbnkgdW5pcXVlIG5v"
    "dGVzOiB7bGVuKHVuaXEpfSIKICAgIG5vdGVfdG9faWR4ID0ge246aSBmb3IgaSxuIGluIGVudW1lcmF0"
    "ZSh1bmlxKX0KCiAgICAjIEJpdG1hcCAoMSBiaXQgcGVyIG5vdGUsIExTQi1maXJzdCB3aXRoaW4gZWFj"
    "aCBieXRlKQogICAgYml0bWFwID0gYnl0ZWFycmF5KCh0b3RhbF9ub3RlcyArIDcpIC8vIDgpCiAgICBm"
    "b3IgaSwgbiBpbiBlbnVtZXJhdGUobm90ZXMpOgogICAgICAgIGlmIG4gIT0gRU1QVFlfTk9URToKICAg"
    "ICAgICAgICAgYml0bWFwW2kgPj4gM10gfD0gMSA8PCAoaSAmIDcpCgogICAgIyBJbmRleCBzdHJlYW0g"
    "KDEgb3IgMiBieXRlcyBwZXIgbm9uLWVtcHR5IG5vdGUsIGxpdHRsZS1lbmRpYW4gaWYgMkIpCiAgICBp"
    "ZHhfc3RyZWFtID0gYnl0ZWFycmF5KCkKICAgIGZvciBuIGluIG5vdGVzOgogICAgICAgIGlmIG4gIT0g"
    "RU1QVFlfTk9URToKICAgICAgICAgICAgaSA9IG5vdGVfdG9faWR4W25dCiAgICAgICAgICAgIGlmIGlk"
    "eF9ieXRlcyA9PSAxOgogICAgICAgICAgICAgICAgaWR4X3N0cmVhbS5hcHBlbmQoaSkKICAgICAgICAg"
    "ICAgZWxzZToKICAgICAgICAgICAgICAgIGlkeF9zdHJlYW0uYXBwZW5kKGkgJiAweEZGKQogICAgICAg"
    "ICAgICAgICAgaWR4X3N0cmVhbS5hcHBlbmQoKGkgPj4gOCkgJiAweEZGKQoKICAgICMgUGVyLXJvdyBj"
    "b3VudDogY291bnQgb2Ygbm9uLWVtcHR5IG5vdGVzIElOIHRoaXMgcm93ICgwLi40KS4KICAgIHBlcl9y"
    "b3dfY291bnQgPSBbXQogICAgZm9yIHJvdyBpbiByYW5nZShudW1fcm93cyk6CiAgICAgICAgY291bnQg"
    "PSBzdW0oMSBmb3IgY2ggaW4gcmFuZ2UoNCkgaWYgbm90ZXNbcm93KjQgKyBjaF0gIT0gRU1QVFlfTk9U"
    "RSkKICAgICAgICBwZXJfcm93X2NvdW50LmFwcGVuZChjb3VudCkKCiAgICAjIFByZWZpeCBzdW06IHBy"
    "ZWZpeFtyb3ddID0gbm9uLWVtcHR5IGNvdW50IGluIHJvd3MgWzAsIHJvdykgPSByYW5rIGF0IHN0YXJ0"
    "IG9mIHJvdy4KICAgICMgU3RvcmVkIGFzIDE2LWJpdCBMRSB3b3JkcyBzbyBkZWNvZGVyIGlzIE8oMSku"
    "CiAgICAjIFJhbmdlOiAwIHRvIH50b3RhbF9ub25fZW1wdHkgKOKJpCA1ODg4IGZvciAyMy1wYXQgTU9E"
    "KSDihpIgZml0cyBlYXNpbHkgaW4gMTYgYml0cy4KICAgIHByZWZpeCA9IFswXSAqIG51bV9yb3dzCiAg"
    "ICBydW5uaW5nID0gMAogICAgZm9yIHJvdyBpbiByYW5nZShudW1fcm93cyk6CiAgICAgICAgcHJlZml4"
    "W3Jvd10gPSBydW5uaW5nCiAgICAgICAgcnVubmluZyArPSBwZXJfcm93X2NvdW50W3Jvd10KCiAgICBy"
    "b3dfc2Vla19ieXRlcyA9IGJ5dGVhcnJheSgpCiAgICBmb3IgdiBpbiBwcmVmaXg6CiAgICAgICAgYXNz"
    "ZXJ0IDAgPD0gdiA8IDY1NTM2LCBmInByZWZpeCB7dn0gb3ZlcmZsb3dzIDE2IGJpdHMiCiAgICAgICAg"
    "cm93X3NlZWtfYnl0ZXMuYXBwZW5kKHYgJiAweEZGKQogICAgICAgIHJvd19zZWVrX2J5dGVzLmFwcGVu"
    "ZCgodiA+PiA4KSAmIDB4RkYpCgogICAgcmV0dXJuIGRpY3QoCiAgICAgICAgdG90YWxfbm90ZXM9dG90"
    "YWxfbm90ZXMsIG51bV9yb3dzPW51bV9yb3dzLAogICAgICAgIHVuaXE9dW5pcSwgbm90ZV90b19pZHg9"
    "bm90ZV90b19pZHgsIGlkeF9ieXRlcz1pZHhfYnl0ZXMsCiAgICAgICAgYml0bWFwPWJpdG1hcCwgaWR4"
    "X3N0cmVhbT1pZHhfc3RyZWFtLAogICAgICAgIHJvd19zZWVrX2J5dGVzPXJvd19zZWVrX2J5dGVzLCBw"
    "cmVmaXg9cHJlZml4LAogICAgKQoKIyDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZAK"
    "IyAzLUJJVCBMSU5FQVIgU0FNUExFIENSVU5DSAojIOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkAoKZGVmIGVuY29kZV9zYW1wbGVzX3BhY2tlZChtb2QsIGJpdHM9Myk6CiAgICAiIiJDb25j"
    "YXRlbmF0ZSBhbGwgc2FtcGxlcywgZW5jb2RlIGVhY2ggdG8gYGJpdHNgIGJpdHMgKHJvdW5kZWQpLgog"
    "ICAgU3VwcG9ydHMgMy1iaXQgYW5kIDQtYml0IGxpbmVhciBxdWFudGl6YXRpb24uCiAgICAgIDMtYml0"
    "OiBjb2RlIDAuLjcsIGxldmVscyAoY29kZSozMiAtIDExMiksIHN0ZXAgMzIvMjU2ID0gMTIuNSUKICAg"
    "ICAgNC1iaXQ6IGNvZGUgMC4uMTUsIGxldmVscyAoY29kZSoxNiAtIDEyMCksIHN0ZXAgMTYvMjU2ID0g"
    "Ni4yNSUgKCs2IGRCIFNOUikKICAgIFJldHVybnMgcGFja2VkIGJ5dGVzICsgcGVyLXNhbXBsZSBzdGFy"
    "dCBpbmRpY2VzIChsb2dpY2FsIHNhbXBsZSB1bml0cykuIiIiCiAgICBpZiBiaXRzIG5vdCBpbiAoMywg"
    "NCk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmImJpdHMgbXVzdCBiZSAzIG9yIDQsIGdvdCB7Yml0"
    "c30iKQoKICAgIGNvbmNhdF9zaWduZWQgPSBbXQogICAgc3RhcnRzID0gW10KICAgIGZvciBzIGluIG1v"
    "ZC5zYW1wbGVfYnl0ZXM6CiAgICAgICAgc3RhcnRzLmFwcGVuZChsZW4oY29uY2F0X3NpZ25lZCkpCiAg"
    "ICAgICAgZm9yIGIgaW4gczoKICAgICAgICAgICAgY29uY2F0X3NpZ25lZC5hcHBlbmQoYiAtIDI1NiBp"
    "ZiBiID49IDEyOCBlbHNlIGIpCiAgICAgICAgY29uY2F0X3NpZ25lZC5leHRlbmQoWzBdICogMTYpCgog"
    "ICAgdG90YWxfc2FtcGxlcyA9IGxlbihjb25jYXRfc2lnbmVkKQogICAgY29kZXMgPSBieXRlYXJyYXko"
    "KQogICAgbWF4X2NvZGUgPSAoMSA8PCBiaXRzKSAtIDEKICAgIHNoaWZ0ID0gOCAtIGJpdHMKICAgIGZv"
    "ciBzdiBpbiBjb25jYXRfc2lnbmVkOgogICAgICAgIHVuc2lnbmVkX29mZnNldCA9IHN2ICsgMTI4ICAj"
    "IFswLCAyNTVdCiAgICAgICAgY29kZSA9IHVuc2lnbmVkX29mZnNldCA+PiBzaGlmdAogICAgICAgIGlm"
    "IGNvZGUgPiBtYXhfY29kZTogY29kZSA9IG1heF9jb2RlCiAgICAgICAgY29kZXMuYXBwZW5kKGNvZGUp"
    "CgogICAgdG90YWxfYml0cyA9IHRvdGFsX3NhbXBsZXMgKiBiaXRzCiAgICB0b3RhbF9ieXRlcyA9ICh0"
    "b3RhbF9iaXRzICsgNykgLy8gOAogICAgcGFja2VkID0gYnl0ZWFycmF5KHRvdGFsX2J5dGVzKQoKICAg"
    "IGlmIGJpdHMgPT0gNDoKICAgICAgICAjIE5pYmJsZSBwYWNraW5nOiAyIGNvZGVzIHBlciBieXRlLCBs"
    "b3cgbmliYmxlIGZpcnN0CiAgICAgICAgZm9yIGksIGMgaW4gZW51bWVyYXRlKGNvZGVzKToKICAgICAg"
    "ICAgICAgYnl0ZV9wb3MgPSBpID4+IDEKICAgICAgICAgICAgaWYgaSAmIDE6CiAgICAgICAgICAgICAg"
    "ICBwYWNrZWRbYnl0ZV9wb3NdIHw9IChjICYgMHhGKSA8PCA0CiAgICAgICAgICAgIGVsc2U6CiAgICAg"
    "ICAgICAgICAgICBwYWNrZWRbYnl0ZV9wb3NdIHw9IGMgJiAweEYKICAgIGVsc2U6ICAjIGJpdHMgPT0g"
    "MwogICAgICAgIGZvciBpLCBjIGluIGVudW1lcmF0ZShjb2Rlcyk6CiAgICAgICAgICAgIGJpdF9wb3Mg"
    "ICA9IGkgKiAzCiAgICAgICAgICAgIGJ5dGVfcG9zICA9IGJpdF9wb3MgPj4gMwogICAgICAgICAgICBi"
    "aXRfc2hpZnQgPSBiaXRfcG9zICYgNwogICAgICAgICAgICB2YWwgPSAoYyAmIDcpIDw8IGJpdF9zaGlm"
    "dAogICAgICAgICAgICBwYWNrZWRbYnl0ZV9wb3NdIHw9IHZhbCAmIDB4RkYKICAgICAgICAgICAgaWYg"
    "Yml0X3NoaWZ0ID4gNSBhbmQgYnl0ZV9wb3MgKyAxIDwgdG90YWxfYnl0ZXM6CiAgICAgICAgICAgICAg"
    "ICBwYWNrZWRbYnl0ZV9wb3MgKyAxXSB8PSAodmFsID4+IDgpICYgMHhGRgoKICAgIHJldHVybiBwYWNr"
    "ZWQsIHN0YXJ0cywgdG90YWxfc2FtcGxlcwoKIyBCYWNrd2FyZC1jb21wYXQgYWxpYXMKZGVmIGVuY29k"
    "ZV9zYW1wbGVzXzNiaXQobW9kKToKICAgIHJldHVybiBlbmNvZGVfc2FtcGxlc19wYWNrZWQobW9kLCBi"
    "aXRzPTMpCgoKZGVmIGNvbXB1dGVfcm93X3NwZWVkX3RhYmxlKG1vZCk6CiAgICAiIiJTaW11bGF0ZSB0"
    "aGUgc29uZyB0byBmaW5kIHBlci1yb3cgU1BFRUQgKGhvbm91cmluZyBGeHgvRHh4L0J4eCBlZmZlY3Rz"
    "KS4KICAgIFJldHVybnMgcm93U3BlZWRbbnVtX3Nvbmdfcm93c10gYW5kIHJvd1N0YXJ0VGlja1tudW1f"
    "c29uZ19yb3dzKzFdLgogICAgQ29ycmVjdGx5IGhhbmRsZXMgRHh4IChwYXR0ZXJuIGJyZWFrKSBhbmQg"
    "Qnh4IChwb3NpdGlvbiBqdW1wKSB3aGljaAogICAgc2hvcnRlbiBhIHBhdHRlcm4ncyBlZmZlY3RpdmUg"
    "cm93IGNvdW50LiIiIgogICAgc3BlZWQgPSA2ICAjIFByb1RyYWNrZXIgZGVmYXVsdAogICAgYnBtICAg"
    "PSAxMjUKICAgIHJvd1NwZWVkID0gW10KICAgIGJwbV9jaGFuZ2VzID0gRmFsc2UKICAgIGZvciBwb3Mg"
    "aW4gcmFuZ2UobW9kLnNvbmdfbGVuZ3RoKToKICAgICAgICBwYXRfaWR4ID0gbW9kLnBhdHRlcm5fb3Jk"
    "ZXJbcG9zXQogICAgICAgIHBkYXRhID0gbW9kLnBhdHRlcm5zW3BhdF9pZHhdCiAgICAgICAgYnJva2Ug"
    "PSBGYWxzZQogICAgICAgIGZvciByb3cgaW4gcmFuZ2UoNjQpOgogICAgICAgICAgICAjIFNjYW4gYWxs"
    "IDQgY2hhbm5lbHMgZm9yIEZ4eCAvIER4eCAvIEJ4eCBvbiB0aGlzIHJvdwogICAgICAgICAgICBmb3Ig"
    "Y2ggaW4gcmFuZ2UoNCk6CiAgICAgICAgICAgICAgICBiYXNlID0gcm93ICogMTYgKyBjaCAqIDQKICAg"
    "ICAgICAgICAgICAgIGIwLCBiMSwgYjIsIGIzID0gcGRhdGFbYmFzZTpiYXNlKzRdCiAgICAgICAgICAg"
    "ICAgICBlZmZlY3QgPSBiMiAmIDB4MEYKICAgICAgICAgICAgICAgIHBhcmFtICA9IGIzCiAgICAgICAg"
    "ICAgICAgICBpZiBlZmZlY3QgPT0gMHhGIGFuZCBwYXJhbSA+IDA6CiAgICAgICAgICAgICAgICAgICAg"
    "aWYgcGFyYW0gPCAweDIwOgogICAgICAgICAgICAgICAgICAgICAgICBzcGVlZCA9IHBhcmFtCiAgICAg"
    "ICAgICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICAgICAgaWYgYnBtICE9IHBhcmFt"
    "OgogICAgICAgICAgICAgICAgICAgICAgICAgICAgYnBtX2NoYW5nZXMgPSBUcnVlCiAgICAgICAgICAg"
    "ICAgICAgICAgICAgIGJwbSA9IHBhcmFtCiAgICAgICAgICAgICAgICBlbGlmIGVmZmVjdCA9PSAweEQg"
    "b3IgZWZmZWN0ID09IDB4QjoKICAgICAgICAgICAgICAgICAgICBicm9rZSA9IFRydWUgICAjIHBhdHRl"
    "cm4gYnJlYWsgLyBwb3NpdGlvbiBqdW1wCiAgICAgICAgICAgIHJvd1NwZWVkLmFwcGVuZChzcGVlZCkK"
    "ICAgICAgICAgICAgaWYgYnJva2U6CiAgICAgICAgICAgICAgICBicmVhayAgICMgc3RvcCBhZGRpbmcg"
    "cm93cyBmb3IgdGhpcyBzb25nIHBvc2l0aW9uCiAgICByb3dTdGFydFRpY2sgPSBbMF0KICAgIGZvciBz"
    "IGluIHJvd1NwZWVkOgogICAgICAgIHJvd1N0YXJ0VGljay5hcHBlbmQocm93U3RhcnRUaWNrWy0xXSAr"
    "IHMpCiAgICByZXR1cm4gcm93U3BlZWQsIHJvd1N0YXJ0VGljaywgYnBtX2NoYW5nZXMKCgpkZWYgZW5j"
    "b2RlX3NhbXBsZXNfdnEyZChtb2QsIEs9MjU2LCB3ZWlnaHRlZD1UcnVlLCBkb3duc2FtcGxlPTIpOgog"
    "ICAgIiIiMi1zdGFnZSBSZXNpZHVhbCBWUSB3aXRoIGludGVncmF0ZWQgYW50aS1hbGlhc2VkIGRvd25z"
    "YW1wbGluZy4KICAgIERvd25zYW1wbGVzIGJ5IGBkb3duc2FtcGxlYCB1c2luZyBwb2x5cGhhc2UgYW50"
    "aS1hbGlhcyBmaWx0ZXIgKHNjaXB5KSwKICAgIHRoZW4gdHJhaW5zIFJWUSBLMT01MTIsIEsyPTI1NiBv"
    "biB0aGUgc3BhcnNlciB2ZWN0b3JzLgogICAgR0xTTCBwZXJpb2RUb0ZyZXEgZGl2aWRlcyBieSBkb3du"
    "c2FtcGxlIHNvIHBsYXliYWNrIHBpdGNoIGlzIHVuY2hhbmdlZC4KICAgIFNOUjogfjQwIGRCIEAgNjMg"
    "S0IgICgrMTIuNyBkQiB2cyBmdWxsLXJlcyBSVlEsIHdpdGhpbiBvcmlnaW5hbCBWUSBidWRnZXQpCiAg"
    "ICAiIiIKICAgIGltcG9ydCBudW1weSBhcyBucAogICAgZnJvbSBza2xlYXJuLmNsdXN0ZXIgaW1wb3J0"
    "IE1pbmlCYXRjaEtNZWFucwogICAgdHJ5OgogICAgICAgIGZyb20gc2NpcHkuc2lnbmFsIGltcG9ydCBy"
    "ZXNhbXBsZV9wb2x5CiAgICBleGNlcHQgSW1wb3J0RXJyb3I6CiAgICAgICAgcmFpc2UgSW1wb3J0RXJy"
    "b3IoInNjaXB5IHJlcXVpcmVkIGZvciBhbnRpLWFsaWFzZWQgZG93bnNhbXBsaW5nIGluIFJWUSIpCgog"
    "ICAgSzEgICA9IDUxMiAgICMgc3RhZ2UgMQogICAgSzIgICA9IDI1NiAgICMgc3RhZ2UgMgogICAgQklU"
    "UzEgPSA5ICAgICMgY2VpbChsb2cyKDUxMikpCiAgICBCSVRTMiA9IDggICAgIyBjZWlsKGxvZzIoMjU2"
    "KSkKICAgIEJJVFNfVE9UQUwgPSBCSVRTMSArIEJJVFMyICAjIDE3IGJpdHMgcGVyIHZlY3RvcgoKICAg"
    "IGNvbmNhdF9kcyA9IFtdCiAgICBzdGFydHMgICAgPSBbXQogICAgdG90YWxfc2FtcGxlc19mdWxsID0g"
    "MCAgIyB0cmFjayBvcmlnaW5hbCBzYW1wbGUgY291bnQgZm9yIEdMU0wgbWV0YWRhdGEKCiAgICBmb3Ig"
    "cywgcmF3X2J5dGVzIGluIHppcChtb2Quc2FtcGxlc19pbmZvLCBtb2Quc2FtcGxlX2J5dGVzKToKICAg"
    "ICAgICBzdGFydHMuYXBwZW5kKGxlbihjb25jYXRfZHMpKQogICAgICAgIGlmIHNbJ2xlbmd0aCddID4g"
    "MDoKICAgICAgICAgICAgcmF3ID0gbnAuZnJvbWJ1ZmZlcihyYXdfYnl0ZXMsIGR0eXBlPW5wLmludDgp"
    "LmFzdHlwZShucC5mbG9hdDMyKSAvIDEyOC4wCiAgICAgICAgICAgIHRvdGFsX3NhbXBsZXNfZnVsbCAr"
    "PSBsZW4ocmF3KQogICAgICAgICAgICAjIEFudGktYWxpYXNlZCBkb3duc2FtcGxpbmcKICAgICAgICAg"
    "ICAgaWYgZG93bnNhbXBsZSA+IDE6CiAgICAgICAgICAgICAgICBkcyA9IHJlc2FtcGxlX3BvbHkocmF3"
    "LCAxLCBkb3duc2FtcGxlKS5hc3R5cGUobnAuZmxvYXQzMikKICAgICAgICAgICAgZWxzZToKICAgICAg"
    "ICAgICAgICAgIGRzID0gcmF3LmNvcHkoKQogICAgICAgICAgICBjb25jYXRfZHMuZXh0ZW5kKGRzLnRv"
    "bGlzdCgpKQogICAgICAgICMgcGFkIHRvIGV2ZW4gbGVuZ3RoICsgOC1ieXRlIHplcm8gZ3VhcmQKICAg"
    "ICAgICB3aGlsZSBsZW4oY29uY2F0X2RzKSAlIDI6IGNvbmNhdF9kcy5hcHBlbmQoMCkKICAgICAgICBj"
    "b25jYXRfZHMuZXh0ZW5kKFswXSAqIDgpCgogICAgd2hpbGUgbGVuKGNvbmNhdF9kcykgJSAyOiBjb25j"
    "YXRfZHMuYXBwZW5kKDApCiAgICB0b3RhbF9zYW1wbGVzID0gbGVuKGNvbmNhdF9kcykgICMgZG93bnNh"
    "bXBsZWQgY291bnQgKHVzZWQgaW4gR0xTTCBUT1RBTF9TQU1QTEVTKQoKICAgIHZlY3RvcnMgPSBucC5h"
    "cnJheShjb25jYXRfZHMsIGR0eXBlPW5wLmZsb2F0MzIpLnJlc2hhcGUoLTEsIDIpCgogICAgIyBTdGFn"
    "ZSAxIOKAlCByaW5nLXdlaWdodGVkCiAgICB3ZWlnaHRzID0gTm9uZQogICAgaWYgd2VpZ2h0ZWQ6CiAg"
    "ICAgICAgc2xvcGVzICA9IG5wLmFicyh2ZWN0b3JzWzosIDFdIC0gdmVjdG9yc1s6LCAwXSkKICAgICAg"
    "ICB3ZWlnaHRzID0gKHNsb3BlcyArIDEuMCkKICAgICAgICB3ZWlnaHRzIC89IHdlaWdodHMubWVhbigp"
    "CgogICAgcHJpbnQoZiIgIFJWUSDDl3tkb3duc2FtcGxlfSBTdGFnZSAxOiBLPXtLMX0gb24ge2xlbih2"
    "ZWN0b3JzKX0gMi12ZWN0b3JzLi4uIiwgZmx1c2g9VHJ1ZSkKICAgIGttMSA9IE1pbmlCYXRjaEtNZWFu"
    "cyhuX2NsdXN0ZXJzPUsxLCBuX2luaXQ9NSwgbWF4X2l0ZXI9NjAsIGJhdGNoX3NpemU9ODE5MiwKICAg"
    "ICAgICAgICAgICAgICAgICAgICAgICByYW5kb21fc3RhdGU9MCwgcmVhc3NpZ25tZW50X3JhdGlvPTAu"
    "MDEpCiAgICBrbTEuZml0KHZlY3RvcnMsIHNhbXBsZV93ZWlnaHQ9d2VpZ2h0cykKICAgIGNvZGVzMSAg"
    "ID0ga20xLnByZWRpY3QodmVjdG9ycykuYXN0eXBlKG5wLmludDMyKQogICAgIyBDZW50cm9pZHMgYXJl"
    "IGluIFstMSwxXSBmbG9hdCByYW5nZSDigJQgc2NhbGUgYmFjayB0byBbLTEyOCwxMjddIGludCByYW5n"
    "ZSBmb3Igc3RvcmFnZQogICAgY2IxICAgICAgPSBucC5jbGlwKG5wLnJvdW5kKGttMS5jbHVzdGVyX2Nl"
    "bnRlcnNfICogMTI4KSwgLTEyOCwgMTI3KS5hc3R5cGUobnAuaW50MzIpCiAgICByZXNpZHVhbCA9IHZl"
    "Y3RvcnMgLSBrbTEuY2x1c3Rlcl9jZW50ZXJzX1tjb2RlczFdCgogICAgc25yMSA9IDEwKm5wLmxvZzEw"
    "KG5wLm1lYW4odmVjdG9ycyoqMikgLyAobnAubWVhbihyZXNpZHVhbCoqMikgKyAxZS05KSkKICAgIHBy"
    "aW50KGYiICBTdGFnZSAxIFNOUjoge3NucjE6LjJmfSBkQiIsIGZsdXNoPVRydWUpCgogICAgIyBTdGFn"
    "ZSAyCiAgICBwcmludChmIiAgUlZRIFN0YWdlIDI6IEs9e0syfSBvbiByZXNpZHVhbC4uLiIsIGZsdXNo"
    "PVRydWUpCiAgICBrbTIgPSBNaW5pQmF0Y2hLTWVhbnMobl9jbHVzdGVycz1LMiwgbl9pbml0PTUsIG1h"
    "eF9pdGVyPTYwLCBiYXRjaF9zaXplPTgxOTIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgcmFuZG9t"
    "X3N0YXRlPTEsIHJlYXNzaWdubWVudF9yYXRpbz0wLjAxKQogICAga20yLmZpdChyZXNpZHVhbCkKICAg"
    "IGNvZGVzMiAgICAgICAgID0ga20yLnByZWRpY3QocmVzaWR1YWwpLmFzdHlwZShucC5pbnQzMikKICAg"
    "IGNiMiAgICAgICAgICAgID0gbnAuY2xpcChucC5yb3VuZChrbTIuY2x1c3Rlcl9jZW50ZXJzXyAqIDEy"
    "OCksIC0xMjgsIDEyNykuYXN0eXBlKG5wLmludDMyKQogICAgZmluYWxfcmVzaWR1YWwgPSByZXNpZHVh"
    "bCAtIGttMi5jbHVzdGVyX2NlbnRlcnNfW2NvZGVzMl0KCiAgICBzbnIyID0gMTAqbnAubG9nMTAobnAu"
    "bWVhbih2ZWN0b3JzKioyKSAvIChucC5tZWFuKGZpbmFsX3Jlc2lkdWFsKioyKSArIDFlLTkpKQogICAg"
    "cHJpbnQoZiIgIFJWUSB0b3RhbCBTTlI6IHtzbnIyOi4yZn0gZEIgKCt7c25yMi1zbnIxOi4yZn0gZEIg"
    "ZnJvbSBzdGFnZSAyKSIsIGZsdXNoPVRydWUpCgogICAgIyBQYWNrIEJJVFMxK0JJVFMyIGJpdHMgcGVy"
    "IHZlY3RvciBMU0ItZmlyc3QKICAgIG5fdmVjcyAgICAgID0gbGVuKHZlY3RvcnMpCiAgICB0b3RhbF9i"
    "aXRzICA9IG5fdmVjcyAqIEJJVFNfVE9UQUwKICAgIHRvdGFsX2J5dGVzID0gKHRvdGFsX2JpdHMgKyA3"
    "KSAvLyA4CiAgICBjb2Rlc19ieXRlcyA9IGJ5dGVhcnJheSh0b3RhbF9ieXRlcykKICAgIG1hc2sxID0g"
    "KDEgPDwgQklUUzEpIC0gMQogICAgbWFzazIgPSAoMSA8PCBCSVRTMikgLSAxCiAgICBmb3IgaSBpbiBy"
    "YW5nZShuX3ZlY3MpOgogICAgICAgIGNvbWJpbmVkICA9IChpbnQoY29kZXMxW2ldKSAmIG1hc2sxKSB8"
    "ICgoaW50KGNvZGVzMltpXSkgJiBtYXNrMikgPDwgQklUUzEpCiAgICAgICAgYml0X3BvcyAgID0gaSAq"
    "IEJJVFNfVE9UQUwKICAgICAgICBieXRlX3BvcyAgPSBiaXRfcG9zID4+IDMKICAgICAgICBiaXRfc2hp"
    "ZnQgPSBiaXRfcG9zICYgNwogICAgICAgIHZhbCA9IGNvbWJpbmVkIDw8IGJpdF9zaGlmdAogICAgICAg"
    "IGNvZGVzX2J5dGVzW2J5dGVfcG9zXSAgICAgfD0gdmFsICAgICAgICAmIDB4RkYKICAgICAgICBpZiBi"
    "eXRlX3BvcysxIDwgdG90YWxfYnl0ZXM6IGNvZGVzX2J5dGVzW2J5dGVfcG9zKzFdIHw9ICh2YWwgPj4g"
    "OCkgICYgMHhGRgogICAgICAgIGlmIGJ5dGVfcG9zKzIgPCB0b3RhbF9ieXRlczogY29kZXNfYnl0ZXNb"
    "Ynl0ZV9wb3MrMl0gfD0gKHZhbCA+PiAxNikgJiAweEZGCiAgICAgICAgaWYgYnl0ZV9wb3MrMyA8IHRv"
    "dGFsX2J5dGVzOiBjb2Rlc19ieXRlc1tieXRlX3BvcyszXSB8PSAodmFsID4+IDI0KSAmIDB4RkYKCiAg"
    "ICAjIENvZGVib29rIGJ5dGVzOiBbSzHDlzIgYnl0ZXNdW0syw5cyIGJ5dGVzXSBzdG9yZWQgdW5zaWdu"
    "ZWQgKCsxMjgpCiAgICBjYl9ieXRlcyA9IGJ5dGVhcnJheSgpCiAgICBmb3IgZW50cnkgaW4gY2IxOgog"
    "ICAgICAgIGZvciB2IGluIGVudHJ5OiBjYl9ieXRlcy5hcHBlbmQoKGludCh2KSsyNTYpICYgMHhGRikK"
    "ICAgIGZvciBlbnRyeSBpbiBjYjI6CiAgICAgICAgZm9yIHYgaW4gZW50cnk6IGNiX2J5dGVzLmFwcGVu"
    "ZCgoaW50KHYpKzI1NikgJiAweEZGKQoKICAgIHJldHVybiBjb2Rlc19ieXRlcywgY2JfYnl0ZXMsIHN0"
    "YXJ0cywgdG90YWxfc2FtcGxlcywgQklUU19UT1RBTAoKCiMg4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQCiMgR0xTTCBFTUlUVEVSUwojIOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkAoKZGVmIGJ5dGVzX3RvX2ludDMyX2JlX2FycmF5KGRhdGEsIGNodW5rX2l2ZWM0PTUxMik6"
    "CiAgICAiIiJQYWNrIGJ5dGVzIGludG8gaXZlYzQgYXJyYXlzIChiaWctZW5kaWFuOiBieXRlIDAgPSBN"
    "U0Igb2YgaW50LngpLiIiIgogICAgIyBQYWQgdG8gbXVsdGlwbGUgb2YgMTYgKHNpbmNlIGVhY2ggaXZl"
    "YzQgaG9sZHMgMTYgYnl0ZXMpCiAgICBwYWRkZWQgPSBieXRlcyhkYXRhKSArIGInXHgwMCcgKiAoKDE2"
    "IC0gbGVuKGRhdGEpICUgMTYpICUgMTYpCiAgICBpbnRzID0gW10KICAgIGZvciBpIGluIHJhbmdlKDAs"
    "IGxlbihwYWRkZWQpLCA0KToKICAgICAgICB2ID0gc3RydWN0LnVucGFjaygnPkknLCBwYWRkZWRbaTpp"
    "KzRdKVswXQogICAgICAgICMgY29udmVydCB0byBzaWduZWQgaW50MzIgZm9yIEdMU0wgKGhhbmRsZXMg"
    "dmFsdWVzID49IDJeMzEpCiAgICAgICAgaWYgdiA+PSAoMSA8PCAzMSk6CiAgICAgICAgICAgIHYgLT0g"
    "KDEgPDwgMzIpCiAgICAgICAgaW50cy5hcHBlbmQodikKICAgICMgU3BsaXQgaW50byBpdmVjNCBhcnJh"
    "eSBjaHVua3Mgb2YgY2h1bmtfaXZlYzQgaXZlYzQgZW50cmllcyBlYWNoCiAgICBjaHVua3MgPSBbXQog"
    "ICAgY3VyID0gW10KICAgIGZvciBpIGluIHJhbmdlKDAsIGxlbihpbnRzKSwgNCk6CiAgICAgICAgY3Vy"
    "LmFwcGVuZCh0dXBsZShpbnRzW2k6aSs0XSkpCiAgICAgICAgaWYgbGVuKGN1cikgPT0gY2h1bmtfaXZl"
    "YzQ6CiAgICAgICAgICAgIGNodW5rcy5hcHBlbmQoY3VyKQogICAgICAgICAgICBjdXIgPSBbXQogICAg"
    "aWYgY3VyOgogICAgICAgIGNodW5rcy5hcHBlbmQoY3VyKQogICAgcmV0dXJuIGNodW5rcwoKZGVmIGVt"
    "aXRfaXZlYzRfYXJyYXkobmFtZSwgY2h1bmtzX29yX3NpbmdsZSwgaXRlbXNfcGVyX2xpbmU9Mik6CiAg"
    "ICAiIiJFbWl0IG9uZSBvciBtb3JlIGNvbnN0IGl2ZWM0IGFycmF5cy4gYGNodW5rc19vcl9zaW5nbGVg"
    "IGlzIGEgbGlzdCBvZiBjaHVua3MuIiIiCiAgICBvdXQgPSBbXQogICAgZm9yIGNpLCBjaHVuayBpbiBl"
    "bnVtZXJhdGUoY2h1bmtzX29yX3NpbmdsZSk6CiAgICAgICAgYXJyX25hbWUgPSBmIntuYW1lfXtjaX0i"
    "IGlmIGxlbihjaHVua3Nfb3Jfc2luZ2xlKSA+IDEgZWxzZSBmIntuYW1lfTAiCiAgICAgICAgb3V0LmFw"
    "cGVuZChmImNvbnN0IGl2ZWM0IHthcnJfbmFtZX1be2xlbihjaHVuayl9XSA9IGl2ZWM0W10oIikKICAg"
    "ICAgICBsaW5lcyA9IFtdCiAgICAgICAgZm9yIHJvd19zdGFydCBpbiByYW5nZSgwLCBsZW4oY2h1bmsp"
    "LCBpdGVtc19wZXJfbGluZSk6CiAgICAgICAgICAgIHJvdyA9IGNodW5rW3Jvd19zdGFydDpyb3dfc3Rh"
    "cnQgKyBpdGVtc19wZXJfbGluZV0KICAgICAgICAgICAgcGFydHMgPSBbIml2ZWM0KHt9LHt9LHt9LHt9"
    "KSIuZm9ybWF0KCp0KSBmb3IgdCBpbiByb3ddCiAgICAgICAgICAgIGxpbmVzLmFwcGVuZCgiICAgICIg"
    "KyAiLCAiLmpvaW4ocGFydHMpKQogICAgICAgIG91dC5hcHBlbmQoIixcbiIuam9pbihsaW5lcykpCiAg"
    "ICAgICAgb3V0LmFwcGVuZCgiKTtcbiIpCiAgICByZXR1cm4gIlxuIi5qb2luKG91dCkKCgojIOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkAojIE1BSU4gQlVJTEQKIyDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZAKCmRlZiBtYWluKG1vZF9wYXRoLCBvdXRfcGF0aCwgSz0yNTYs"
    "IHdlaWdodGVkPVRydWUsIGRvd25zYW1wbGU9Mik6CiAgICAiIiJHZW5lcmF0ZSBTaGFkZXJUb3kgQ29t"
    "bW9uIEdMU0wgZm9yIGEgTU9EIGZpbGUuCiAgICBkb3duc2FtcGxlOiBhbnRpLWFsaWFzIGRvd25zYW1w"
    "bGUgZmFjdG9yIGZvciBzYW1wbGUgZW5jb2RpbmcgKDE9b2ZmLCAyPXJlY29tbWVuZGVkKS4KICAgICAg"
    "ICAgICAgICAgIEhpZ2hlciBkb3duc2FtcGxlIOKGkiBzbWFsbGVyIGRhdGEsIGxhcmdlciBjb2RlYm9v"
    "a3MsIGJldHRlciBTTlIuCiAgICAgICAgICAgICAgICAgIMOXMTogSzE9NjQsICBLMj0zMiAgKH43OSBL"
    "QiwgIDI3LjcgZEIpCiAgICAgICAgICAgICAgICAgIMOXMjogSzE9NTEyLCBLMj0yNTYgKH43NyBLQiwg"
    "IDM4LjQgZEIpICDihpAgcmVjb21tZW5kZWQKICAgICAgICAgICAgICAgICAgw5c0OiBLMT01MTIsIEsy"
    "PTI1NiAofjM5IEtCLCAgMzcuMSBkQikKICAgICIiIgogICAgbW9kID0gTU9ERmlsZShtb2RfcGF0aCkK"
    "ICAgIHByaW50KGYi8J+TpiBMb2FkZWQ6IHttb2QudGl0bGV9IikKICAgIHByaW50KGYiICAgUGF0dGVy"
    "bnM6IHttb2QubnVtX3BhdHRlcm5zfSwgU29uZyBsZW5ndGg6IHttb2Quc29uZ19sZW5ndGh9IikKCiAg"
    "ICAjIFBhdHRlcm4gY3J1bmNoCiAgICBwID0gZW5jb2RlX3BhdHRlcm5zKG1vZCkKICAgIHByaW50KGYi"
    "XG7wn5ec77iPICBQQVRURVJOIENSVU5DSCIpCiAgICBwcmludChmIiAgIFRvdGFsIG5vdGVzOiAgICAg"
    "ICB7cFsndG90YWxfbm90ZXMnXX0iKQogICAgcHJpbnQoZiIgICBVbmlxdWUgbm9uLWVtcHR5OiAge2xl"
    "bihwWyd1bmlxJ10pfSIpCiAgICBwcmludChmIiAgIERpY3Rpb25hcnk6ICAgICAgICB7bGVuKHBbJ3Vu"
    "aXEnXSkqNH0gYnl0ZXMiKQogICAgcHJpbnQoZiIgICBCaXRtYXA6ICAgICAgICAgICAge2xlbihwWydi"
    "aXRtYXAnXSl9IGJ5dGVzIikKICAgIHByaW50KGYiICAgSW5kZXggc3RyZWFtOiAgICAgIHtsZW4ocFsn"
    "aWR4X3N0cmVhbSddKX0gYnl0ZXMiKQogICAgcHJpbnQoZiIgICBSb3cgc2VlayAoMTYtYml0IHByZWZp"
    "eCk6IHtsZW4ocFsncm93X3NlZWtfYnl0ZXMnXSl9IGJ5dGVzIikKICAgIHBhdHRlcm5fdG90YWwgPSBs"
    "ZW4ocFsndW5pcSddKSo0ICsgbGVuKHBbJ2JpdG1hcCddKSArIGxlbihwWydpZHhfc3RyZWFtJ10pICsg"
    "bGVuKHBbJ3Jvd19zZWVrX2J5dGVzJ10pCiAgICBwcmludChmIiAgIOKGkiBQYXR0ZXJuIHRvdGFsOiAg"
    "IHtwYXR0ZXJuX3RvdGFsOix9IGJ5dGVzIikKCiAgICAjIFNwZWVkL3RpY2sgdGFibGUgZnJvbSBGeHgg"
    "ZWZmZWN0cwogICAgcm93U3BlZWQsIHJvd1N0YXJ0VGljaywgYnBtX2NoYW5nZXMgPSBjb21wdXRlX3Jv"
    "d19zcGVlZF90YWJsZShtb2QpCiAgICBwcmludChmIlxu4o+x77iPICBTUEVFRCBUQUJMRSIpCiAgICBw"
    "cmludChmIiAgIFNvbmcgcm93czoge2xlbihyb3dTcGVlZCl9LCB0b3RhbCB0aWNrczoge3Jvd1N0YXJ0"
    "VGlja1stMV19IikKICAgIHByaW50KGYiICAgVW5pcXVlIHNwZWVkczoge3NvcnRlZChzZXQocm93U3Bl"
    "ZWQpKX0iKQogICAgc3BlZWRfdGFibGVfYnl0ZXMgPSBsZW4ocm93U3RhcnRUaWNrKSAqIDIKICAgIHBy"
    "aW50KGYiICAgcm93U3RhcnRUaWNrOiB7c3BlZWRfdGFibGVfYnl0ZXN9IGJ5dGVzICgxNi1iaXQgcGFj"
    "a2VkKSIpCiAgICBpZiBicG1fY2hhbmdlczoKICAgICAgICBwcmludChmIiAgIOKaoO+4jyAgQlBNIGNo"
    "YW5nZXMgZGV0ZWN0ZWQgKDEyVEguTU9EIGhhcyBub25lLCBidXQgb3RoZXIgTU9EcyBtaWdodCkiKQoK"
    "ICAgICMgQXV0by1zZWxlY3QgZG93bnNhbXBsZSBpZiBub3QgZXhwbGljaXRseSBvdmVycmlkZGVuIChk"
    "b3duc2FtcGxlPTIgaXMgZGVmYXVsdCkKICAgICMgQnVkZ2V0IGVzdGltYXRlOiB0b3RhbF9yYXdfYnl0"
    "ZXMgLyBkb3duc2FtcGxlICogMTcvMTYgKDE3LWJpdCBjb2RlcywgMiBieXRlcy9zYW1wbGUpCiAgICAj"
    "IFNoYWRlclRveSBzYWZlIHpvbmU6IOKJpCA4MCBLQiBzYW1wbGUgY29kZXMgKyBwYXR0ZXJuIGRhdGEK"
    "ICAgIGltcG9ydCBudW1weSBhcyBucAogICAgdG90YWxfcmF3ID0gc3VtKHNbJ2xlbmd0aCddIGZvciBz"
    "IGluIG1vZC5zYW1wbGVzX2luZm8pCiAgICBlc3RpbWF0ZWRfYnVkZ2V0X2RzMiA9ICh0b3RhbF9yYXcg"
    "Ly8gMikgKiAxNyAvLyAxNiArIDE2MDAwICAjIHJvdWdoIGVzdGltYXRlIGluYy4gcGF0dGVybnMKICAg"
    "IGlmIGRvd25zYW1wbGUgPT0gMiBhbmQgZXN0aW1hdGVkX2J1ZGdldF9kczIgPiA4MCAqIDEwMjQ6CiAg"
    "ICAgICAgb2xkX2RzID0gZG93bnNhbXBsZQogICAgICAgIGRvd25zYW1wbGUgPSA0CiAgICAgICAgcHJp"
    "bnQoZiIgICDimqDvuI8gIEF1dG8tc3dpdGNoaW5nIHRvIC0tZG93bnNhbXBsZSA0IChlc3RpbWF0ZWQg"
    "YnVkZ2V0IGF0IMOXMiB3YXMgIgogICAgICAgICAgICAgIGYie2VzdGltYXRlZF9idWRnZXRfZHMyLy8x"
    "MDI0fSBLQiwgZXhjZWVkcyA4MCBLQiBzYWZlIHpvbmUpIikKCiAgICAjIFJWUSBjb2RlYm9vayBzaXpl"
    "cyBzY2FsZSB3aXRoIGRvd25zYW1wbGUKICAgIGlmIGRvd25zYW1wbGUgPj0gMjoKICAgICAgICBLMSwg"
    "SzIgPSA1MTIsIDI1NgogICAgZWxzZTogICMgbm8gZG93bnNhbXBsaW5nCiAgICAgICAgSzEsIEsyID0g"
    "NjQsIDMyCiAgICBCSVRTMSA9IGludChucC5jZWlsKG5wLmxvZzIoSzEpKSkKICAgIEJJVFMyID0gaW50"
    "KG5wLmNlaWwobnAubG9nMihLMikpKQogICAgQklUU19UT1RBTCA9IEJJVFMxICsgQklUUzIKCiAgICAj"
    "IFNhbXBsZSBlbmNvZGluZzogUlZRIHdpdGggaW50ZWdyYXRlZCBhbnRpLWFsaWFzZWQgZG93bnNhbXBs"
    "aW5nCiAgICBkc19sYWJlbCA9IGYiw5d7ZG93bnNhbXBsZX0iIGlmIGRvd25zYW1wbGUgPiAxIGVsc2Ug"
    "ImZ1bGwtcmVzIgogICAgcHJpbnQoZiJcbvCfl5zvuI8gIFNBTVBMRSBDUlVOQ0ggKFJWUSB7ZHNfbGFi"
    "ZWx9IEsxPXtLMX0gSzI9e0syfSwgcmluZy13ZWlnaHRlZCkiKQogICAgY29kZXNfYnl0ZXMsIGNiX2J5"
    "dGVzLCBzdGFydHMsIHRvdGFsX3NhbXBsZXMsIGJpdHNfcGVyX2NvZGUgPSBlbmNvZGVfc2FtcGxlc192"
    "cTJkKAogICAgICAgIG1vZCwgSywgd2VpZ2h0ZWQsIGRvd25zYW1wbGU9ZG93bnNhbXBsZSkKICAgIHBy"
    "aW50KGYiICAgTG9naWNhbCBzYW1wbGVzOiAgIHt0b3RhbF9zYW1wbGVzOix9ICAoe2RzX2xhYmVsfSki"
    "KQogICAgcHJpbnQoZiIgICBDb2RlcyBwYWNrZWQ6ICAgICAge2xlbihjb2Rlc19ieXRlcyk6LH0gYnl0"
    "ZXMgICh7Yml0c19wZXJfY29kZX0gYml0cy92ZWN0b3Igw5cge3RvdGFsX3NhbXBsZXMvLzJ9IHZlY3Rv"
    "cnMpIikKICAgIHByaW50KGYiICAgQ29kZWJvb2tzOiAgICAgICAgIHtsZW4oY2JfYnl0ZXMpOix9IGJ5"
    "dGVzICAoe0sxfcOXMiArIHtLMn3DlzIgYnl0ZXMpIikKCiAgICB0b3RhbF9idWRnZXQgPSBwYXR0ZXJu"
    "X3RvdGFsICsgbGVuKGNvZGVzX2J5dGVzKSArIGxlbihjYl9ieXRlcykgKyAzMSoyNCArIHNwZWVkX3Rh"
    "YmxlX2J5dGVzCiAgICBwcmludChmIlxu8J+TiiBUT1RBTCBjb25zdCBkYXRhIGJ1ZGdldDogfnt0b3Rh"
    "bF9idWRnZXQ6LH0gYnl0ZXMgICh7dG90YWxfYnVkZ2V0LzEwMjQ6LjFmfSBLQikiKQoKICAgICMgQ2h1"
    "bmsgZm9yIEdMU0wKICAgIGRpY3RfYnl0ZXMgPSBiJycuam9pbihwWyd1bmlxJ10pCiAgICBkaWN0X2No"
    "dW5rcyAgICA9IGJ5dGVzX3RvX2ludDMyX2JlX2FycmF5KGRpY3RfYnl0ZXMpCiAgICBiaXRtYXBfY2h1"
    "bmtzICA9IGJ5dGVzX3RvX2ludDMyX2JlX2FycmF5KGJ5dGVzKHBbJ2JpdG1hcCddKSkKICAgIGlkeF9j"
    "aHVua3MgICAgID0gYnl0ZXNfdG9faW50MzJfYmVfYXJyYXkoYnl0ZXMocFsnaWR4X3N0cmVhbSddKSkK"
    "ICAgIHJvd3NlZWtfY2h1bmtzID0gYnl0ZXNfdG9faW50MzJfYmVfYXJyYXkoYnl0ZXMocFsncm93X3Nl"
    "ZWtfYnl0ZXMnXSkpCiAgICBjb2Rlc19jaHVua3MgICA9IGJ5dGVzX3RvX2ludDMyX2JlX2FycmF5KGJ5"
    "dGVzKGNvZGVzX2J5dGVzKSkKICAgIGNiX2NodW5rcyAgICAgID0gYnl0ZXNfdG9faW50MzJfYmVfYXJy"
    "YXkoYnl0ZXMoY2JfYnl0ZXMpKQoKICAgICMgUGFjayByb3dTdGFydFRpY2sgYXMgMTYtYml0IExFIGJ5"
    "dGVzIOKGkiBpdmVjNCBjaHVua3MKICAgIHRpY2tfYnl0ZXMgPSBieXRlYXJyYXkoKQogICAgZm9yIHQg"
    "aW4gcm93U3RhcnRUaWNrOgogICAgICAgIHRpY2tfYnl0ZXMuYXBwZW5kKHQgJiAweEZGKQogICAgICAg"
    "IHRpY2tfYnl0ZXMuYXBwZW5kKCh0ID4+IDgpICYgMHhGRikKICAgIHRpY2tfY2h1bmtzID0gYnl0ZXNf"
    "dG9faW50MzJfYmVfYXJyYXkoYnl0ZXModGlja19ieXRlcykpCgogICAgc2FtcGxlc19pbmZvX25ldyA9"
    "IFtdCiAgICBmb3IgaSwgKHMsIHN0KSBpbiBlbnVtZXJhdGUoemlwKG1vZC5zYW1wbGVzX2luZm8sIHN0"
    "YXJ0cykpOgogICAgICAgICMgTGVuZ3RocyBhbmQgbG9vcCBwb2ludHMgYXJlIGluIHRoZSBkb3duc2Ft"
    "cGxlZCBkb21haW4KICAgICAgICBzYW1wbGVzX2luZm9fbmV3LmFwcGVuZChkaWN0KAogICAgICAgICAg"
    "ICBzdGFydD1zdCwKICAgICAgICAgICAgbGVuZ3RoPXNbJ2xlbmd0aCddIC8vIGRvd25zYW1wbGUsCiAg"
    "ICAgICAgICAgIGxvb3BTdGFydD1zWydsb29wX3N0YXJ0J10gLy8gZG93bnNhbXBsZSwKICAgICAgICAg"
    "ICAgbG9vcExlbj1zWydsb29wX2xlbiddIC8vIGRvd25zYW1wbGUsCiAgICAgICAgICAgIHZvbHVtZT1z"
    "Wyd2b2x1bWUnXSwgZmluZXR1bmU9c1snZmluZXR1bmUnXSwKICAgICAgICApKQoKICAgIGdsc2wgPSBi"
    "dWlsZF9nbHNsKG1vZCwgcCwgY29kZXNfYnl0ZXMsIHN0YXJ0cywgdG90YWxfc2FtcGxlcywKICAgICAg"
    "ICAgICAgICAgICAgICAgZGljdF9jaHVua3MsIGJpdG1hcF9jaHVua3MsIGlkeF9jaHVua3MsIHJvd3Nl"
    "ZWtfY2h1bmtzLAogICAgICAgICAgICAgICAgICAgICBjb2Rlc19jaHVua3MsIGNiX2NodW5rcywgc2Ft"
    "cGxlc19pbmZvX25ldywgSywgYml0c19wZXJfY29kZSwKICAgICAgICAgICAgICAgICAgICAgdGlja19j"
    "aHVua3MsIHJvd1N0YXJ0VGljaywKICAgICAgICAgICAgICAgICAgICAgSzE9SzEsIEsyPUsyLCBCSVRT"
    "MT1CSVRTMSwgQklUUzI9QklUUzIsIEJJVFNfVE9UQUw9QklUU19UT1RBTCwKICAgICAgICAgICAgICAg"
    "ICAgICAgZG93bnNhbXBsZT1kb3duc2FtcGxlKQogICAgd2l0aCBvcGVuKG91dF9wYXRoLCAndycpIGFz"
    "IGY6CiAgICAgICAgZi53cml0ZShnbHNsKQogICAgcHJpbnQoZiJcbuKchSBXcm90ZToge291dF9wYXRo"
    "fSAgKHtsZW4oZ2xzbC5lbmNvZGUoJ3V0Zi04JykpOix9IGJ5dGVzKSIpCgoKZGVmIGJ1aWxkX2dsc2wo"
    "bW9kLCBwLCBwYWNrZWQsIHN0YXJ0cywgdG90YWxfc2FtcGxlcywKICAgICAgICAgICAgICAgZGljdF9j"
    "aHVua3MsIGJpdG1hcF9jaHVua3MsIGlkeF9jaHVua3MsIHJvd3NlZWtfY2h1bmtzLAogICAgICAgICAg"
    "ICAgICBjb2Rlc19jaHVua3MsIGNiX2NodW5rcywgc2FtcGxlc19pbmZvX25ldywgSywgYml0c19wZXJf"
    "Y29kZSwKICAgICAgICAgICAgICAgdGlja19jaHVua3MsIHJvd1N0YXJ0VGljaywgSzE9NTEyLCBLMj0y"
    "NTYsIEJJVFMxPTksIEJJVFMyPTgsIEJJVFNfVE9UQUw9MTcsIGRvd25zYW1wbGU9Mik6CgogICAgIyDi"
    "lIDilIAgU29uZyBtZXRhZGF0YQogICAgc29uZ19wb3NpdGlvbnMgPSBtb2QucGF0dGVybl9vcmRlcls6"
    "bW9kLnNvbmdfbGVuZ3RoXQoKICAgICMgQ29tcHV0ZSBhY3R1YWwgcm93cyBwZXIgc29uZyBwb3NpdGlv"
    "biDigJQgUHJvVHJhY2tlciBEeHggKHBhdHRlcm4gYnJlYWspCiAgICAjIGFuZCBCeHggKHBvc2l0aW9u"
    "IGp1bXApIHNob3J0ZW4gdGhlIGVmZmVjdGl2ZSBwYXR0ZXJuIGxlbmd0aC4KICAgIGRlZiBhY3R1YWxf"
    "cGF0dGVybl9yb3dzKHNwKToKICAgICAgICBwYXQgPSBtb2QucGF0dGVybl9vcmRlcltzcF0KICAgICAg"
    "ICBmb3Igcm93IGluIHJhbmdlKDY0KToKICAgICAgICAgICAgZm9yIGNoIGluIHJhbmdlKDQpOgogICAg"
    "ICAgICAgICAgICAgYmFzZSA9IDEwODQgKyBwYXQqMTAyNCArIHJvdyoxNiArIGNoKjQKICAgICAgICAg"
    "ICAgICAgIG5iID0gbW9kLmRhdGFbYmFzZTpiYXNlKzRdCiAgICAgICAgICAgICAgICBlZmYgPSBuYlsy"
    "XSAmIDB4RgogICAgICAgICAgICAgICAgaWYgZWZmID09IDB4RCBvciBlZmYgPT0gMHhCOiAgICMgcGF0"
    "dGVybiBicmVhayBvciBwb3NpdGlvbiBqdW1wCiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHJvdyAr"
    "IDEKICAgICAgICByZXR1cm4gNjQKCiAgICBwYXRfcm93cyA9IFthY3R1YWxfcGF0dGVybl9yb3dzKHNw"
    "KSBmb3Igc3AgaW4gcmFuZ2UobW9kLnNvbmdfbGVuZ3RoKV0KICAgIHBhdF9yb3dfb2Zmc2V0ID0gWzBd"
    "CiAgICBmb3IgciBpbiBwYXRfcm93czoKICAgICAgICBwYXRfcm93X29mZnNldC5hcHBlbmQocGF0X3Jv"
    "d19vZmZzZXRbLTFdICsgcikKICAgIHBhdF9zdGFydF9yb3cgID0gWzBdKm1vZC5zb25nX2xlbmd0aAoK"
    "ICAgICMgcGF0VGlja09mZnNldFtzcF0gPSBpbmRleCBpbnRvIHJvd1N0YXJ0VGljayBmb3Igcm93IDAg"
    "b2Ygc29uZyBwb3NpdGlvbiBzcAogICAgIyBTYW1lIGFzIHBhdF9yb3dfb2Zmc2V0IHNpbmNlIHRpY2sg"
    "dGFibGUgcm93cyA9PSBzb25nIHJvd3MgYWZ0ZXIgRDAwIGZpeC4KICAgIHBhdF90aWNrX29mZnNldCA9"
    "IHBhdF9yb3dfb2Zmc2V0WzpdCgogICAgdG90YWxfc29uZ19yb3dzID0gbW9kLnNvbmdfbGVuZ3RoICog"
    "NjQKICAgIG51bV9wYXR0ZXJucyA9IG1vZC5udW1fcGF0dGVybnMKCiAgICAjIOKUgOKUgCBTYW1wbGVJ"
    "bmZvIGVtaXNzaW9uICh1c2UgbmV3IGBzdGFydGAgPSBzYW1wbGUgaW5kZXggaW4gdGhlIGNvbmNhdGVu"
    "YXRlZCBzdHJlYW0pCiAgICBkZWYgZm10X3NhbXBsZWluZm8ocyk6CiAgICAgICAgcmV0dXJuIGYiU2Ft"
    "cGxlSW5mbyh7c1snc3RhcnQnXX0sIHtzWydsZW5ndGgnXX0sIHtzWydsb29wU3RhcnQnXX0sIHtzWyds"
    "b29wTGVuJ119LCB7c1sndm9sdW1lJ119LCAxKSIKICAgIHNpX2xpbmVzID0gW10KICAgIGZvciBpLCBz"
    "IGluIGVudW1lcmF0ZShzYW1wbGVzX2luZm9fbmV3KToKICAgICAgICBzaV9saW5lcy5hcHBlbmQoZiIg"
    "ICAge2ZtdF9zYW1wbGVpbmZvKHMpfXsnLCcgaWYgaTwzMCBlbHNlICcnfSIpCiAgICBzYW1wbGVzX2lu"
    "Zm9fZ2xzbCA9ICJjb25zdCBTYW1wbGVJbmZvIHNhbXBsZXNbMzFdID0gU2FtcGxlSW5mb1tdKFxuIiAr"
    "ICJcbiIuam9pbihzaV9saW5lcykgKyAiXG4pOyIKCiAgICAjIOKUgOKUgCBjaGFubmVsUGFuIChzYW1l"
    "IGFzIGV4aXN0aW5nOiBBbWlnYSBMUlJMIHdpdGggcmVzdCBjZW50ZXJlZCkKICAgIGNoYW5fcGFuID0g"
    "WzAuMCwgMS4wLCAxLjAsIDAuMF0gKyBbMC41XSoyOAoKICAgICMg4pSA4pSAIENodW5rIGFycmF5IGRl"
    "Y2xhcmF0aW9ucwogICAgZGljdF9sZW4gICAgPSBzdW0obGVuKGMpIGZvciBjIGluIGRpY3RfY2h1bmtz"
    "KQogICAgYml0bWFwX2xlbiAgPSBzdW0obGVuKGMpIGZvciBjIGluIGJpdG1hcF9jaHVua3MpCiAgICBp"
    "ZHhfbGVuICAgICA9IHN1bShsZW4oYykgZm9yIGMgaW4gaWR4X2NodW5rcykKICAgIHJvd3NlZWtfbGVu"
    "ID0gc3VtKGxlbihjKSBmb3IgYyBpbiByb3dzZWVrX2NodW5rcykKICAgIGNvZGVzX2xlbiAgID0gc3Vt"
    "KGxlbihjKSBmb3IgYyBpbiBjb2Rlc19jaHVua3MpCiAgICBjYl9sZW4gICAgICA9IHN1bShsZW4oYykg"
    "Zm9yIGMgaW4gY2JfY2h1bmtzKQoKICAgIGhlYWRlciA9IGYiIiIvKiA9PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CiAg"
    "IEdMU0wgTU9EIFBsYXllciB2MS4yIChjKSAyMDI2IE9yYmxpdml1cwogICBOZXc6IFZRIHNhbXBsZSBj"
    "b21wcmVzc2lvbiwgM0QgU3Vycm91bmQsIEZBVCBCYXNzLCBhbmQgY3ViaWMgcmVzYW1wbGluZwogICBD"
    "b250YWN0OiBzdWJiYW5kQGdtYWlsLmNvbSBvcgogICAgICAgICAgICBzdWJiYW5kQHByb3Rvbm1haWwu"
    "Y29tCiAgIEdJVDogICAgIGh0dHBzOi8vZ2l0aHViLmNvbS9tZXd6YS9tb2QyZ2xzbAogICBDT01NT04g"
    "VEFCCiAgIEdlbmVyYXRlZCBmcm9tOiB7bW9kLnRpdGxlfQogICAKICAgQ29tcHJlc3Npb246CiAgICAg"
    "4oCiIFBhdHRlcm5zOiBiaXRtYXAgKyBkaWN0aW9uYXJ5ICsgMTYtYml0IHByZWZpeC1zdW0gcm93IHNl"
    "ZWsgKE8oMSkpCiAgICAg4oCiIFNhbXBsZXM6ICAyLXN0YWdlIFJWUSDDl3tkb3duc2FtcGxlfSBBQS1k"
    "b3duc2FtcGxlZCAoSzE9e0sxfSwgSzI9e0syfSksIHtCSVRTX1RPVEFMfSBiaXRzL3BhaXIKICAgICAg"
    "ICAgICAgICAgICByaW5nLXdlaWdodGVkIGstbWVhbnMgdHJhaW5lZCBvbiB0aGlzIE1PRCdzIGNvbnRl"
    "bnQKICAgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PSAqLwoKI2RlZmluZSBVU0VfRU1CRURERURfREFUQSAxCiNkZWZp"
    "bmUgTlVNX1BBVFRFUk5TICAgICAge251bV9wYXR0ZXJuc30KI2RlZmluZSBTT05HX0xFTkdUSCAgICAg"
    "ICB7bW9kLnNvbmdfbGVuZ3RofQojZGVmaW5lIFNPTkdfTE9PUF9QT1MgICAgIDAKI2RlZmluZSBOVU1f"
    "Q0hBTk5FTFMgICAgICA0CiNkZWZpbmUgQlBNICAgICAgICAgICAgICAgMTI1LjAKI2RlZmluZSBTUEVF"
    "RCAgICAgICAgICAgICA2LjAKI2RlZmluZSBUT1RBTF9TT05HX1JPV1MgICB7dG90YWxfc29uZ19yb3dz"
    "fQoKLy8g4pSA4pSAIFBhdHRlcm4gY3J1bmNoIGNvbnN0YW50cyDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKI2Rl"
    "ZmluZSBUT1RBTF9OT1RFUyAgICAgICB7cFsndG90YWxfbm90ZXMnXX0KI2RlZmluZSBUT1RBTF9ST1dT"
    "ICAgICAgICB7cFsnbnVtX3Jvd3MnXX0KI2RlZmluZSBESUNUX05PVEVTICAgICAgICB7bGVuKHBbJ3Vu"
    "aXEnXSl9CiNkZWZpbmUgSURYX0JZVEVTX1BFUiAgICAge3BbJ2lkeF9ieXRlcyddfQojZGVmaW5lIERJ"
    "Q1RfSU5UUyAgICAgICAgIHtkaWN0X2xlbn0KI2RlZmluZSBCSVRNQVBfSU5UUyAgICAgICB7Yml0bWFw"
    "X2xlbn0KI2RlZmluZSBJRFhfSU5UUyAgICAgICAgICB7aWR4X2xlbn0KI2RlZmluZSBST1dTRUVLX0lO"
    "VFMgICAgICB7cm93c2Vla19sZW59CgovLyDilIDilIAgUlZRIHNhbXBsZSBjb25zdGFudHMg4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACi8vIFNhbXBsZXMgYXJlIHtkb3duc2FtcGxlfcOXIGFu"
    "dGktYWxpYXMgZG93bnNhbXBsZWQgYmVmb3JlIGVuY29kaW5nLgovLyBwZXJpb2RUb0ZyZXEgZGl2aWRl"
    "cyBieSB7Mipkb3duc2FtcGxlfS4wICg9IDIuMCDDlyB7ZG93bnNhbXBsZX0pIHNvIHBpdGNoIGlzIHVu"
    "Y2hhbmdlZC4KI2RlZmluZSBSVlFfQ09ERVNfQllURVMgICB7bGVuKHBhY2tlZCl9CiNkZWZpbmUgUlZR"
    "X0NCX0JZVEVTICAgICAge0sxKjIgKyBLMioyfQojZGVmaW5lIFRPVEFMX1NBTVBMRVMgICAgIHt0b3Rh"
    "bF9zYW1wbGVzfQoKI2RlZmluZSBCSVRNQVBfQllURVMgICAgICB7bGVuKHBbJ2JpdG1hcCddKX0KI2Rl"
    "ZmluZSBJRFhfQllURVMgICAgICAgICB7bGVuKHBbJ2lkeF9zdHJlYW0nXSl9CiNkZWZpbmUgUk9XU0VF"
    "S19CWVRFUyAgICAge2xlbihwWydyb3dfc2Vla19ieXRlcyddKX0KCi8vIOKUgOKUgCBGeHgtYXdhcmUg"
    "dGltaW5nIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAojZGVmaW5lIFRP"
    "VEFMX1RJQ0tTICAgICAgIHtyb3dTdGFydFRpY2tbLTFdfQojZGVmaW5lIE5VTV9TT05HX1JPV1MgICAg"
    "IHtsZW4ocm93U3RhcnRUaWNrKS0xfQojZGVmaW5lIFRJQ0tTX1BFUl9TRUMgICAgIDUwLjAgICAvLyBC"
    "UE09MTI1IGNvbnN0YW50IGZvciAxMlRILk1PRAoKLy8g4pSA4pSAIEF1ZGlvIGVmZmVjdHMg4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACmNvbnN0IGJvb2wg"
    "IGVuYWJsZTNEICAgICAgPSB0cnVlOwpjb25zdCBib29sICBlbmFibGVGQVQgICAgID0gdHJ1ZTsKY29u"
    "c3QgaXZlYzIgc3Vycl9jaGFubmVscyA9IGl2ZWMyKDEsIDQpOwoiIiIKCiAgICAjIOKUgOKUgCBzb25n"
    "IG1ldGFkYXRhIGFycmF5cwogICAgY2hhbl9wYW5fc3RyICAgPSAiLCAiLmpvaW4oZiJ7djouMWZ9IiBm"
    "b3IgdiBpbiBjaGFuX3BhbikKICAgIHNvbmdwb3Nfc3RyICAgID0gIiwgIi5qb2luKHN0cih4KSBmb3Ig"
    "eCBpbiBzb25nX3Bvc2l0aW9ucykKICAgIHJvd29mZl9zdHIgICAgID0gIiwgIi5qb2luKHN0cih4KSBm"
    "b3IgeCBpbiBwYXRfcm93X29mZnNldCkKICAgIHN0YXJ0cm93X3N0ciAgID0gIiwgIi5qb2luKHN0cih4"
    "KSBmb3IgeCBpbiBwYXRfc3RhcnRfcm93KQogICAgdGlja29mZl9zdHIgICAgPSAiLCAiLmpvaW4oc3Ry"
    "KHgpIGZvciB4IGluIHBhdF90aWNrX29mZnNldFs6LTFdKSAgIyBsZW5ndGggPSBzb25nX2xlbmd0aAoK"
    "ICAgIG1ldGEgPSBmIiIiCmNvbnN0IGZsb2F0IGNoYW5uZWxQYW5bMzJdID0gZmxvYXRbXSh7Y2hhbl9w"
    "YW5fc3RyfSk7CmNvbnN0IGludCAgIHNvbmdQb3NpdGlvbnNbe21vZC5zb25nX2xlbmd0aH1dICAgPSBp"
    "bnRbXSh7c29uZ3Bvc19zdHJ9KTsKY29uc3QgaW50ICAgcGF0Um93T2Zmc2V0W3ttb2Quc29uZ19sZW5n"
    "dGgrMX1dICAgID0gaW50W10oe3Jvd29mZl9zdHJ9KTsKY29uc3QgaW50ICAgcGF0U3RhcnRSb3dbe21v"
    "ZC5zb25nX2xlbmd0aH1dICAgICA9IGludFtdKHtzdGFydHJvd19zdHJ9KTsKY29uc3QgaW50ICAgcGF0"
    "VGlja09mZnNldFt7bW9kLnNvbmdfbGVuZ3RofV0gICA9IGludFtdKHt0aWNrb2ZmX3N0cn0pOwoiIiIK"
    "CiAgICAjIOKUgOKUgCBEYXRhIGFycmF5cyAoaXZlYzQgY2h1bmtzKQogICAgZGF0YV9hcnJheXMgPSBb"
    "IlxuLy8g4pSA4pSAIFBhdHRlcm4gZGljdGlvbmFyeSAodW5pcXVlIDQtYnl0ZSBub3RlcywgTVNCLWZp"
    "cnN0IHBlciBpbnQpIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgFxuIl0KICAgIGRhdGFf"
    "YXJyYXlzLmFwcGVuZChlbWl0X2l2ZWM0X2FycmF5KCJwYXREaWN0IiwgZGljdF9jaHVua3MpKQogICAg"
    "ZGF0YV9hcnJheXMuYXBwZW5kKCJcbi8vIOKUgOKUgCBQYXR0ZXJuIGJpdG1hcCAoMSBiaXQvbm90ZSwg"
    "TFNCLWZpcnN0IHdpdGhpbiBieXRlKSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIBcbiIpCiAgICBkYXRhX2FycmF5cy5hcHBlbmQoZW1pdF9pdmVj"
    "NF9hcnJheSgicGF0Qml0bWFwIiwgYml0bWFwX2NodW5rcykpCiAgICBkYXRhX2FycmF5cy5hcHBlbmQo"
    "IlxuLy8g4pSA4pSAIEluZGV4IHN0cmVhbSAoJXMgYnl0ZXMgcGVyIG5vbi1lbXB0eSBub3RlKSDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIBcbiIgJSBwWydpZHhfYnl0ZXMnXSkKICAgIGRhdGFfYXJyYXlzLmFwcGVuZChl"
    "bWl0X2l2ZWM0X2FycmF5KCJwYXRJZHgiLCBpZHhfY2h1bmtzKSkKICAgIGRhdGFfYXJyYXlzLmFwcGVu"
    "ZCgiXG4vLyDilIDilIAgUm93IHNlZWsgdGFibGUgKDE2LWJpdCBMRSBwcmVmaXggc3VtcywgTygxKSBs"
    "b29rdXApIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gFxuIikKICAgIGRhdGFfYXJyYXlzLmFwcGVuZChlbWl0X2l2ZWM0X2FycmF5KCJwYXRSb3dTZWVrIiwg"
    "cm93c2Vla19jaHVua3MpKQogICAgZGF0YV9hcnJheXMuYXBwZW5kKCJcbi8vIOKUgOKUgCBWUSBjb2Rl"
    "cyAocGFja2VkIGJpdCBzdHJlYW0pIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgFxuIikKICAgIGRhdGFfYXJyYXlzLmFwcGVuZChlbWl0X2l2"
    "ZWM0X2FycmF5KCJ2cUNvZGVzIiwgY29kZXNfY2h1bmtzKSkKICAgIGRhdGFfYXJyYXlzLmFwcGVuZChm"
    "IlxuLy8g4pSA4pSAIFZRIGNvZGVib29rICh7S30gZW50cmllcyDDlyAyIHNhbXBsZXMsIHNpZ25lZCA4"
    "LWJpdCBhcyB1bnNpZ25lZCkg4pSA4pSAXG4iKQogICAgZGF0YV9hcnJheXMuYXBwZW5kKGVtaXRfaXZl"
    "YzRfYXJyYXkoInZxQ29kZWJvb2siLCBjYl9jaHVua3MpKQogICAgZGF0YV9hcnJheXMuYXBwZW5kKCJc"
    "bi8vIOKUgOKUgCBQZXItcm93IGN1bXVsYXRpdmUgdGljayB0YWJsZSAoMTYtYml0IExFLCBGeHgtYXdh"
    "cmUpIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgFxuIikKICAg"
    "IGRhdGFfYXJyYXlzLmFwcGVuZChlbWl0X2l2ZWM0X2FycmF5KCJyb3dTdGFydFRpY2siLCB0aWNrX2No"
    "dW5rcykpCgogICAgIyDilIDilIAgU2FtcGxlSW5mbyAmIHBlcmlvZFRhYmxlCiAgICB0YWJsZXMgPSBm"
    "IiIiCi8vIOKUgOKUgCBTYW1wbGUgbWV0YWRhdGEgKHN0YXJ0ID0gc2FtcGxlIGluZGV4IGluIHBhY2tl"
    "ZCAzLWJpdCBzdHJlYW0pIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgApzdHJ1Y3QgU2FtcGxlSW5m"
    "byB7ewogICAgaW50IHN0YXJ0LCBsZW5ndGgsIGxvb3BTdGFydCwgbG9vcExlbiwgdm9sdW1lLCBid0Zh"
    "Y3RvcjsKfX07CntzYW1wbGVzX2luZm9fZ2xzbH0KCi8vIFByb1RyYWNrZXIgcGVyaW9kIHRhYmxlIChD"
    "LTEgdG8gQi0zKQpjb25zdCBpbnQgcGVyaW9kVGFibGVbMzddID0gaW50W10oCiAgICA4NTYsODA4LDc2"
    "Miw3MjAsNjc4LDY0MCw2MDQsNTcwLDUzOCw1MDgsNDgwLDQ1MywKICAgIDQyOCw0MDQsMzgxLDM2MCwz"
    "MzksMzIwLDMwMiwyODUsMjY5LDI1NCwyNDAsMjI2LAogICAgMjE0LDIwMiwxOTAsMTgwLDE3MCwxNjAs"
    "MTUxLDE0MywxMzUsMTI3LDEyMCwxMTMsMAopOwoKLy8gUHJvVHJhY2tlciAzMi1lbnRyeSBzaW5lIHRh"
    "YmxlIGZvciB2aWJyYXRvIChMVVQsIGtlcHQgZ2xvYmFsIHNvIGl0IGRvZXNuJ3QKLy8gY29uc3VtZSBw"
    "ZXItY2FsbCBwcml2YXRlL3N0YWNrIHN0b3JhZ2UgaW4gZ2V0Q2hhbm5lbE91dHB1dCkuCmNvbnN0IGZs"
    "b2F0IHZpYlRhYlszMl0gPSBmbG9hdFtdKAogICAgICAwLjAsICAyNC4wLCAgNDkuMCwgIDc0LjAsICA5"
    "Ny4wLCAxMjAuMCwgMTQxLjAsIDE2MS4wLAogICAgMTgwLjAsIDE5Ny4wLCAyMTIuMCwgMjI0LjAsIDIz"
    "NS4wLCAyNDQuMCwgMjUwLjAsIDI1My4wLAogICAgMjU1LjAsIDI1My4wLCAyNTAuMCwgMjQ0LjAsIDIz"
    "NS4wLCAyMjQuMCwgMjEyLjAsIDE5Ny4wLAogICAgMTgwLjAsIDE2MS4wLCAxNDEuMCwgMTIwLjAsICA5"
    "Ny4wLCAgNzQuMCwgIDQ5LjAsICAyNC4wCik7CgpmbG9hdCBwZXJpb2RUb0ZyZXEoaW50IHBlcmlvZCkg"
    "e3sKICAgIC8vIEFtaWdhIFBBTDogNzA5Mzc4OS4yIC8gKHBlcmlvZCDDlyAyKS4gRXh0cmEgw5d7ZG93"
    "bnNhbXBsZX0gZm9yIEFBLWRvd25zYW1wbGVkIHNhbXBsZXMuCiAgICByZXR1cm4gcGVyaW9kID4gMCA/"
    "IDcwOTM3ODkuMiAvIChmbG9hdChwZXJpb2QpICogezIqZG93bnNhbXBsZX0uMCkgOiAwLjA7Cn19CiIi"
    "IgoKICAgICMg4pSA4pSAIEZldGNoIGhlbHBlcnMgKGNodW5rIGRpc3BhdGNoZXJzIGZvciBlYWNoIGFy"
    "cmF5KQogICAgZGVmIGNodW5rX2Rpc3BhdGNoKG5hbWUsIG51bV9jaHVua3MsIHZhcj0naScpOgogICAg"
    "ICAgIGlmIG51bV9jaHVua3MgPT0gMToKICAgICAgICAgICAgcmV0dXJuIGYiICAgIHJldHVybiB7bmFt"
    "ZX0wW3t2YXJ9Pj4yXTsiCiAgICAgICAgbGluZXMgPSBbZiIgICAgaXZlYzQgdiA9IGl2ZWM0KDApOyJd"
    "CiAgICAgICAgbGluZXMuYXBwZW5kKGYiICAgIGlmIChjaHVua0lkeCA9PSAwKSB2ID0ge25hbWV9MFt7"
    "dmFyfT4+Ml07IikKICAgICAgICBmb3IgayBpbiByYW5nZSgxLCBudW1fY2h1bmtzKToKICAgICAgICAg"
    "ICAgbGluZXMuYXBwZW5kKGYiICAgIGVsc2UgaWYgKGNodW5rSWR4ID09IHtrfSkgdiA9IHtuYW1lfXtr"
    "fVt7dmFyfT4+Ml07IikKICAgICAgICBsaW5lcy5hcHBlbmQoZiIgICAgcmV0dXJuIHY7IikKICAgICAg"
    "ICByZXR1cm4gIlxuIi5qb2luKGxpbmVzKQoKICAgIGRlZiBpdmVjNF9zZWxlY3QodmFyPSdpJyk6CiAg"
    "ICAgICAgcmV0dXJuIGYiIiIgICAgaW50IGNpID0ge3Zhcn0gJiAzOwogICAgcmV0dXJuIGNpPT0wID8g"
    "di54IDogY2k9PTEgPyB2LnkgOiBjaT09MiA/IHYueiA6IHYudzsiIiIKCiAgICBmZXRjaGVycyA9IGYi"
    "IiIKLy8g4pWQ4pWQ4pWQIENodW5rZWQgaXZlYzQgZmV0Y2hlcnMg4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQCgovLyBGZXRjaCBhIGJ5dGUgZnJvbSBhbnkgY2h1bmtlZCBieXRlIGFycmF5IChNU0ItZmlyc3Qg"
    "d2l0aGluIGVhY2ggaW50MzIpLgovLyBFYWNoIGl2ZWM0IGhvbGRzIDE2IGJ5dGVzOiAueCA9IGJ5dGVz"
    "IDAtMywgLnkgPSA0LTcsIC56ID0gOC0xMSwgLncgPSAxMi0xNQovLyBXaXRoaW4gZWFjaCBpbnQ6IGJ5"
    "dGUgMCA9IE1TQiwgYnl0ZSAzID0gTFNCLgoKaW50IF9leHRyYWN0Qnl0ZShpdmVjNCB2LCBpbnQgYnl0"
    "ZUluSXZlYzQpIHt7CiAgICBpbnQgaW50SWR4ID0gYnl0ZUluSXZlYzQgPj4gMjsKICAgIGludCBieXRl"
    "SW5JbnQgPSBieXRlSW5JdmVjNCAmIDM7CiAgICBpbnQgcGFja2VkID0gaW50SWR4PT0wID8gdi54IDog"
    "aW50SWR4PT0xID8gdi55IDogaW50SWR4PT0yID8gdi56IDogdi53OwogICAgaW50IHNoaWZ0ID0gMjQg"
    "LSBieXRlSW5JbnQgKiA4OwogICAgcmV0dXJuIChwYWNrZWQgPj4gc2hpZnQpICYgMHhGRjsKfX0KCi8v"
    "IOKUgOKUgCBEaWN0aW9uYXJ5IGJ5dGUgZmV0Y2ggKGJ5dGVJZHggaW4gWzAsIERJQ1RfTk9URVMqNCkp"
    "IOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAppbnQg"
    "ZmV0Y2hEaWN0Qnl0ZShpbnQgYnl0ZUlkeCkge3sKICAgIGludCBpdmVjNElkeCA9IGJ5dGVJZHggPj4g"
    "NDsKICAgIGludCBieXRlSW5JdmVjNCA9IGJ5dGVJZHggJiAxNTsKICAgIGl2ZWM0IHYgPSBwYXREaWN0"
    "MFtpdmVjNElkeF07CiAgICByZXR1cm4gX2V4dHJhY3RCeXRlKHYsIGJ5dGVJbkl2ZWM0KTsKfX0KCi8v"
    "IOKUgOKUgCBCaXRtYXAgYnl0ZSBmZXRjaCDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIAKaW50IGZldGNoQml0bWFwQnl0ZShpbnQgYnl0ZUlkeCkge3sKICAgIGludCBpdmVjNElkeCA9"
    "IGJ5dGVJZHggPj4gNDsKICAgIGludCBieXRlSW5JdmVjNCA9IGJ5dGVJZHggJiAxNTsKICAgIGl2ZWM0"
    "IHYgPSBwYXRCaXRtYXAwW2l2ZWM0SWR4XTsKICAgIHJldHVybiBfZXh0cmFjdEJ5dGUodiwgYnl0ZUlu"
    "SXZlYzQpOwp9fQoKLy8g4pSA4pSAIEluZGV4IHN0cmVhbSBieXRlIGZldGNoIChjaHVua2VkIGlmIG5l"
    "ZWRlZCkg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACmludCBmZXRjaElkeEJ5dGUoaW50IGJ5dGVJZHgpIHt7CiAg"
    "ICBpbnQgaXZlYzRJZHggPSBieXRlSWR4ID4+IDQ7CiAgICBpbnQgYnl0ZUluSXZlYzQgPSBieXRlSWR4"
    "ICYgMTU7CiAgICBpbnQgY2h1bmtJZHggPSBpdmVjNElkeCAvIDUxMjsKICAgIGludCBsb2NhbEl2ZWM0"
    "ID0gaXZlYzRJZHggJSA1MTI7CiAgICBpdmVjNCB2ID0gaXZlYzQoMCk7CntjaHIoMTApLmpvaW4oZicg"
    "ICAgeyJpZiIgaWYgaz09MCBlbHNlICJlbHNlIGlmIn0gKGNodW5rSWR4ID09IHtrfSkgdiA9IHBhdElk"
    "eHtrfVtsb2NhbEl2ZWM0XTsnIGZvciBrIGluIHJhbmdlKGxlbihpZHhfY2h1bmtzKSkpfQogICAgcmV0"
    "dXJuIF9leHRyYWN0Qnl0ZSh2LCBieXRlSW5JdmVjNCk7Cn19CgovLyDilIDilIAgUm93LXNlZWsgbmli"
    "YmxlIGJ5dGUgZmV0Y2gg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACmludCBmZXRjaFJvd1NlZWtCeXRlKGludCBieXRlSWR4KSB7"
    "ewogICAgaW50IGl2ZWM0SWR4ID0gYnl0ZUlkeCA+PiA0OwogICAgaW50IGJ5dGVJbkl2ZWM0ID0gYnl0"
    "ZUlkeCAmIDE1OwogICAgaXZlYzQgdiA9IHBhdFJvd1NlZWswW2l2ZWM0SWR4XTsKICAgIHJldHVybiBf"
    "ZXh0cmFjdEJ5dGUodiwgYnl0ZUluSXZlYzQpOwp9fQoKLy8g4pSA4pSAIFZRIGNvZGUgc3RyZWFtIGJ5"
    "dGUgZmV0Y2ggKGNodW5rZWQpIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gAppbnQgZmV0Y2hDb2Rlc0J5dGUoaW50IGJ5dGVJZHgpIHt7CiAgICBpbnQgaXZlYzRJZHggPSBieXRl"
    "SWR4ID4+IDQ7CiAgICBpbnQgYnl0ZUluSXZlYzQgPSBieXRlSWR4ICYgMTU7CiAgICBpbnQgY2h1bmtJ"
    "ZHggPSBpdmVjNElkeCAvIDUxMjsKICAgIGludCBsb2NhbEl2ZWM0ID0gaXZlYzRJZHggJSA1MTI7CiAg"
    "ICBpdmVjNCB2ID0gaXZlYzQoMCk7CntjaHIoMTApLmpvaW4oZicgICAgeyJpZiIgaWYgaz09MCBlbHNl"
    "ICJlbHNlIGlmIn0gKGNodW5rSWR4ID09IHtrfSkgdiA9IHZxQ29kZXN7a31bbG9jYWxJdmVjNF07JyBm"
    "b3IgayBpbiByYW5nZShsZW4oY29kZXNfY2h1bmtzKSkpfQogICAgcmV0dXJuIF9leHRyYWN0Qnl0ZSh2"
    "LCBieXRlSW5JdmVjNCk7Cn19CgovLyDilIDilIAgVlEgY29kZWJvb2sgYnl0ZSBmZXRjaCAoc21hbGws"
    "IGZpdHMgaW4gMSBjaHVuayB1c3VhbGx5KSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIAKaW50IGZldGNoQ29kZWJvb2tCeXRlKGludCBieXRlSWR4KSB7ewogICAgaW50IGl2ZWM0"
    "SWR4ID0gYnl0ZUlkeCA+PiA0OwogICAgaW50IGJ5dGVJbkl2ZWM0ID0gYnl0ZUlkeCAmIDE1OwogICAg"
    "aW50IGNodW5rSWR4ID0gaXZlYzRJZHggLyA1MTI7CiAgICBpbnQgbG9jYWxJdmVjNCA9IGl2ZWM0SWR4"
    "ICUgNTEyOwogICAgaXZlYzQgdiA9IGl2ZWM0KDApOwp7Y2hyKDEwKS5qb2luKGYnICAgIHsiaWYiIGlm"
    "IGs9PTAgZWxzZSAiZWxzZSBpZiJ9IChjaHVua0lkeCA9PSB7a30pIHYgPSB2cUNvZGVib29re2t9W2xv"
    "Y2FsSXZlYzRdOycgZm9yIGsgaW4gcmFuZ2UobGVuKGNiX2NodW5rcykpKX0KICAgIHJldHVybiBfZXh0"
    "cmFjdEJ5dGUodiwgYnl0ZUluSXZlYzQpOwp9fQoiIiIKCiAgICAjIOKUgOKUgCBwb3Bjb3VudCBoZWxw"
    "ZXIgKDQtYml0IG5pYmJsZSkKICAgICMg4pSA4pSAIGdldE5vdGU6IGJpdG1hcCArIGRpY3QgbG9va3Vw"
    "IHdpdGggTygxKSByb3cgc2VlayArIHByZWZpeCBwb3Bjb3VudAogICAgZGVjb2RlcnMgPSAiIiIKLy8g"
    "4pWQ4pWQ4pWQIFBhdHRlcm4gZGVjb2RlcjogYml0bWFwICsgZGljdGlvbmFyeSArIHJvdyBzZWVrIOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkAoKc3RydWN0IE5vdGUgeyBpbnQgaW5zdHJ1bWVudCwgcGVyaW9kLCBlZmZlY3QsIHBhcmFt"
    "OyB9OwoKLy8gUG9wY291bnQgZm9yIDQtYml0IG5pYmJsZSAoMC4uMTUg4oaSIDAuLjQpCmludCBwb3Bj"
    "b3VudDQoaW50IHgpIHsKICAgIHggPSAoeCAmIDB4NSkgKyAoKHggPj4gMSkgJiAweDUpOwogICAgcmV0"
    "dXJuICh4ICYgMHgzKSArICgoeCA+PiAyKSAmIDB4Myk7Cn0KCi8vIFJlY29uc3RydWN0IGN1bXVsYXRp"
    "dmUgbm9uLWVtcHR5IGNvdW50IHVwIHRvIHN0YXJ0IG9mIGByb3dgIOKAlCBPKDEpLgovLyBSb3cgc2Vl"
    "ayB0YWJsZSBob2xkcyAxNi1iaXQgTEUgcHJlZml4IHN1bXM6IDIgYnl0ZXMgcGVyIHJvdy4KaW50IHJv"
    "d1NlZWtDdW0oaW50IHRhcmdldFJvdykgewogICAgaW50IGJ5dGVJZHggPSB0YXJnZXRSb3cgKiAyOwog"
    "ICAgaW50IGxvID0gZmV0Y2hSb3dTZWVrQnl0ZShieXRlSWR4KTsKICAgIGludCBoaSA9IGZldGNoUm93"
    "U2Vla0J5dGUoYnl0ZUlkeCArIDEpOwogICAgcmV0dXJuIGxvIHwgKGhpIDw8IDgpOwp9CgpOb3RlIGVt"
    "cHR5Tm90ZSgpIHsgTm90ZSBuOyBuLmluc3RydW1lbnQ9MDsgbi5wZXJpb2Q9MDsgbi5lZmZlY3Q9MDsg"
    "bi5wYXJhbT0wOyByZXR1cm4gbjsgfQoKTm90ZSBnZXROb3RlKGludCBzb25nUG9zLCBpbnQgcm93LCBp"
    "bnQgY2hhbm5lbCkgewogICAgaW50IHBhdCA9IHNvbmdQb3NpdGlvbnNbc29uZ1Bvc107CiAgICBpbnQg"
    "cm93R2xvYmFsID0gcGF0ICogNjQgKyByb3c7CiAgICBpbnQgbm90ZUlkeCAgID0gcm93R2xvYmFsICog"
    "NCArIGNoYW5uZWw7CgogICAgLy8gMSkgQml0bWFwIGNoZWNrCiAgICBpbnQgYm1CeXRlID0gZmV0Y2hC"
    "aXRtYXBCeXRlKG5vdGVJZHggPj4gMyk7CiAgICBpbnQgYml0ID0gKGJtQnl0ZSA+PiAobm90ZUlkeCAm"
    "IDcpKSAmIDE7CiAgICBpZiAoYml0ID09IDApIHJldHVybiBlbXB0eU5vdGUoKTsKCiAgICAvLyAyKSBD"
    "b3VudCBub24tZW1wdHkgbm90ZXMgYmVmb3JlIHRoaXMgcG9zaXRpb24KICAgIC8vICAgID0gY3VtdWxh"
    "dGl2ZSB1cCB0byByb3dHbG9iYWwgKyBwb3Bjb3VudCBvZiBiaXRtYXAgbmliYmxlIHdpdGhpbiB0aGlz"
    "IHJvdwogICAgaW50IHJhbmsgPSByb3dTZWVrQ3VtKHJvd0dsb2JhbCk7CiAgICAvLyBUaGlzIHJvdydz"
    "IDQgYml0cyBzcGFuIGNoYW5uZWxzIDAuLjMg4oaSIHRha2UgY2hhbm5lbHMgWzAuLmNoYW5uZWwtMV0K"
    "ICAgIGludCByb3dCaXRtYXBTdGFydCA9IHJvd0dsb2JhbCAqIDQ7CiAgICAvLyBUaGUgNCBiaXRzIG9m"
    "IHRoaXMgcm93IG1heSBzcGFuIDEgYnl0ZSAoaWYgYWxpZ25lZCkgb3IgMi4KICAgIGludCBieXRlMElk"
    "eCA9IHJvd0JpdG1hcFN0YXJ0ID4+IDM7CiAgICBpbnQgc2hpZnQgICAgPSByb3dCaXRtYXBTdGFydCAm"
    "IDc7CiAgICBpbnQgYnl0ZTAgPSBmZXRjaEJpdG1hcEJ5dGUoYnl0ZTBJZHgpOwogICAgaW50IGJ5dGUx"
    "ID0gZmV0Y2hCaXRtYXBCeXRlKGJ5dGUwSWR4ICsgMSk7CiAgICBpbnQgcm93Qml0cyA9ICgoYnl0ZTAg"
    "Pj4gc2hpZnQpIHwgKGJ5dGUxIDw8ICg4IC0gc2hpZnQpKSkgJiAweEY7CiAgICBpbnQgbWFzayA9ICgx"
    "IDw8IGNoYW5uZWwpIC0gMTsKICAgIHJhbmsgKz0gcG9wY291bnQ0KHJvd0JpdHMgJiBtYXNrKTsKCiAg"
    "ICAvLyAzKSBMb29rIHVwIGluZGV4IGFuZCBmZXRjaCBub3RlIGZyb20gZGljdGlvbmFyeQogICAgaW50"
    "IGRpY3RJZHg7CiNpZiBJRFhfQllURVNfUEVSID09IDEKICAgIGRpY3RJZHggPSBmZXRjaElkeEJ5dGUo"
    "cmFuayk7CiNlbHNlCiAgICBpbnQgbG8gPSBmZXRjaElkeEJ5dGUocmFuayAqIDIpOwogICAgaW50IGhp"
    "ID0gZmV0Y2hJZHhCeXRlKHJhbmsgKiAyICsgMSk7CiAgICBkaWN0SWR4ID0gbG8gfCAoaGkgPDwgOCk7"
    "CiNlbmRpZgogICAgaW50IGIwID0gZmV0Y2hEaWN0Qnl0ZShkaWN0SWR4ICogNCArIDApOwogICAgaW50"
    "IGIxID0gZmV0Y2hEaWN0Qnl0ZShkaWN0SWR4ICogNCArIDEpOwogICAgaW50IGIyID0gZmV0Y2hEaWN0"
    "Qnl0ZShkaWN0SWR4ICogNCArIDIpOwogICAgaW50IGIzID0gZmV0Y2hEaWN0Qnl0ZShkaWN0SWR4ICog"
    "NCArIDMpOwoKICAgIE5vdGUgbjsKICAgIG4uaW5zdHJ1bWVudCA9IChiMCAmIDB4RjApIHwgKChiMiA+"
    "PiA0KSAmIDB4MEYpOwogICAgbi5wZXJpb2QgICAgID0gKChiMCAmIDB4MEYpIDw8IDgpIHwgYjE7CiAg"
    "ICBuLmVmZmVjdCAgICAgPSBiMiAmIDB4MEY7CiAgICBuLnBhcmFtICAgICAgPSBiMzsKICAgIHJldHVy"
    "biBuOwp9CgoiIiIKCiAgICAjIFNhbXBsZSBkZWNvZGVyOiBmLXN0cmluZyBmb3IgI2RlZmluZXMgKG5l"
    "ZWQgUHl0aG9uIHZhcnMpLCBwbGFpbiBzdHJpbmcgZm9yIGZ1bmN0aW9uIGJvZGllcwogICAgZGVjb2Rl"
    "cnMgKz0gKAogICAgICAgIGYiLy8g4pWQ4pWQ4pWQIFNhbXBsZSBkZWNvZGVyOiAyLXN0YWdlIFJWUSDD"
    "l3tkb3duc2FtcGxlfSBBQS1kb3duc2FtcGxlZCDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZBcbiIKICAgICAgICBmIi8vIHtCSVRTX1RPVEFMfS1iaXQgY29kZXMgcGFja2Vk"
    "IExTQi1maXJzdDogW3tCSVRTMX0tYml0IGNvZGUxXVt7QklUUzJ9LWJpdCBjb2RlMl1cbiIKICAgICAg"
    "ICBmIi8vIHBlcmlvZFRvRnJlcSB1c2VzIMOXezIqZG93bnNhbXBsZX0g4oCUIHNhbXBsZXMgc3RvcmVk"
    "IGF0IDEve2Rvd25zYW1wbGV9IHJhdGVcbiIKICAgICAgICBmIiNkZWZpbmUgUlZRX0JJVFMgICAgIHtC"
    "SVRTX1RPVEFMfVxuIgogICAgICAgIGYiI2RlZmluZSBSVlFfQklUU18xICAge0JJVFMxfVxuIgogICAg"
    "ICAgIGYiI2RlZmluZSBSVlFfSzEgICAgICAge0sxfVxuIgogICAgICAgIGYiI2RlZmluZSBSVlFfSzIg"
    "ICAgICAge0syfVxuIgogICAgICAgIGYiI2RlZmluZSBSVlFfQ0IyX0JZVEUgKHtLMX0gKiAyKVxuIgog"
    "ICAgICAgIGYiI2RlZmluZSBSVlFfTUFTSzEgICAgeygxPDxCSVRTMSktMX1cbiIKICAgICAgICBmIiNk"
    "ZWZpbmUgUlZRX01BU0syICAgIHsoMTw8QklUUzIpLTF9XG4iCiAgICApCiAgICBkZWNvZGVycyArPSAi"
    "IiIKdm9pZCBfZ2V0UlZRQ29kZXMoaW50IHZlY0lkeCwgb3V0IGludCBjb2RlMSwgb3V0IGludCBjb2Rl"
    "MikgewogICAgaW50IGJpdFBvcyAgPSB2ZWNJZHggKiBSVlFfQklUUzsKICAgIGludCBieXRlUG9zID0g"
    "Yml0UG9zID4+IDM7CiAgICBpbnQgc2hpZnQgICA9IGJpdFBvcyAmIDc7CiAgICBpbnQgYjAgPSBmZXRj"
    "aENvZGVzQnl0ZShieXRlUG9zKTsKICAgIGludCBiMSA9IGZldGNoQ29kZXNCeXRlKGJ5dGVQb3MgKyAx"
    "KTsKICAgIGludCBiMiA9IGZldGNoQ29kZXNCeXRlKGJ5dGVQb3MgKyAyKTsKICAgIGludCBiMyA9IGZl"
    "dGNoQ29kZXNCeXRlKGJ5dGVQb3MgKyAzKTsKICAgIGludCBjb21iaW5lZCA9IGIwIHwgKGIxIDw8IDgp"
    "IHwgKGIyIDw8IDE2KSB8IChiMyA8PCAyNCk7CiAgICBpbnQgcmF3ID0gKGNvbWJpbmVkID4+IHNoaWZ0"
    "KSAmICgoMSA8PCBSVlFfQklUUykgLSAxKTsKICAgIGNvZGUxID0gcmF3ICYgUlZRX01BU0sxOwogICAg"
    "Y29kZTIgPSAocmF3ID4+IFJWUV9CSVRTXzEpICYgUlZRX01BU0syOwp9CgpmbG9hdCBnZXRTYW1wbGUo"
    "aW50IHNhbXBsZUlkeCkgewogICAgaWYgKHNhbXBsZUlkeCA8IDAgfHwgc2FtcGxlSWR4ID49IFRPVEFM"
    "X1NBTVBMRVMpIHJldHVybiAwLjA7CiAgICBpbnQgdmVjSWR4ID0gc2FtcGxlSWR4ID4+IDE7CiAgICBp"
    "bnQgbGFuZSAgID0gc2FtcGxlSWR4ICYgMTsKICAgIC8vIElubGluZSBSVlEgZGVjb2RlIChhdm9pZHMg"
    "b3V0LXBhcmFtZXRlciBzdGFjayBhbGxvY2F0aW9uKQogICAgaW50IF9icCA9IHZlY0lkeCAqIFJWUV9C"
    "SVRTLCBfYnkgPSBfYnAgPj4gMywgX3NoID0gX2JwICYgNzsKICAgIGludCBfcmF3ID0gKGZldGNoQ29k"
    "ZXNCeXRlKF9ieSkgfCAoZmV0Y2hDb2Rlc0J5dGUoX2J5KzEpPDw4KSB8CiAgICAgICAgICAgICAgICAo"
    "ZmV0Y2hDb2Rlc0J5dGUoX2J5KzIpPDwxNikgfCAoZmV0Y2hDb2Rlc0J5dGUoX2J5KzMpPDwyNCkpOwog"
    "ICAgX3JhdyA9IChfcmF3ID4+IF9zaCkgJiAoKDEgPDwgUlZRX0JJVFMpIC0gMSk7CiAgICBpbnQgY29k"
    "ZTEgPSBfcmF3ICYgUlZRX01BU0sxOwogICAgaW50IGNvZGUyID0gKF9yYXcgPj4gUlZRX0JJVFNfMSkg"
    "JiBSVlFfTUFTSzI7CiAgICBpbnQgdWIxID0gZmV0Y2hDb2RlYm9va0J5dGUoY29kZTEgKiAyICsgbGFu"
    "ZSk7CiAgICBpbnQgczEgID0gdWIxIDwgMTI4ID8gdWIxIDogdWIxIC0gMjU2OwogICAgaW50IHViMiA9"
    "IGZldGNoQ29kZWJvb2tCeXRlKFJWUV9DQjJfQllURSArIGNvZGUyICogMiArIGxhbmUpOwogICAgaW50"
    "IHMyICA9IHViMiA8IDEyOCA/IHViMiA6IHViMiAtIDI1NjsKICAgIHJldHVybiBmbG9hdChzMSArIHMy"
    "KSAvIDEyOC4wOwp9CgovLyDilIDilIAgUG9zaXRpb24gY2FsY3VsYXRpb24gKEZ4eC1hd2FyZSB2aWEg"
    "cm93U3RhcnRUaWNrKSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIAKc3RydWN0IFBvc2l0aW9uIHsgaW50IHNvbmdQb3MsIHBhdHRlcm4sIHJvdzsg"
    "ZmxvYXQgdGljaywgcm93VGltZTsgfTsKCi8vIEZldGNoIDE2LWJpdCBMRSB2YWx1ZSBhdCByb3cgaW5k"
    "ZXggaW50byByb3dTdGFydFRpY2sKaW50IGZldGNoVGljayhpbnQgcm93SWR4KSB7CiAgICBpbnQgYnl0"
    "ZUlkeCA9IHJvd0lkeCAqIDI7CiAgICBpbnQgY2h1bmtJZHggID0gYnl0ZUlkeCA+PiA2OwogICAgaW50"
    "IGJ5dGVJbjE2ICA9IGJ5dGVJZHggJiA2MzsKICAgIGludCBsbyA9IF9leHRyYWN0Qnl0ZShyb3dTdGFy"
    "dFRpY2swWyhjaHVua0lkeDw8MikrKGJ5dGVJbjE2Pj40KV0sIGJ5dGVJbjE2ICYgMTUpOwogICAgLy8g"
    "bmV4dCBieXRlCiAgICBpbnQgYnl0ZUlkeDIgPSBieXRlSWR4ICsgMTsKICAgIGludCBjaHVua0lkeDIg"
    "PSBieXRlSWR4MiA+PiA2OwogICAgaW50IGJ5dGVJbjE2XzIgPSBieXRlSWR4MiAmIDYzOwogICAgaW50"
    "IGhpID0gX2V4dHJhY3RCeXRlKHJvd1N0YXJ0VGljazBbKGNodW5rSWR4Mjw8MikrKGJ5dGVJbjE2XzI+"
    "PjQpXSwgYnl0ZUluMTZfMiAmIDE1KTsKICAgIHJldHVybiBsbyB8IChoaSA8PCA4KTsKfQoKUG9zaXRp"
    "b24gZ2V0UG9zaXRpb24oZmxvYXQgdGltZSkgewogICAgUG9zaXRpb24gcG9zOwogICAgZmxvYXQgc29u"
    "Z0R1cmF0aW9uID0gZmxvYXQoVE9UQUxfVElDS1MpIC8gVElDS1NfUEVSX1NFQzsKICAgIGZsb2F0IGxv"
    "b3BlZFRpbWUgPSBtb2QodGltZSwgc29uZ0R1cmF0aW9uKTsKICAgIGZsb2F0IHRvdGFsVGlja0YgPSBs"
    "b29wZWRUaW1lICogVElDS1NfUEVSX1NFQzsKCiAgICAvLyBCaW5hcnkgc2VhcmNoIHJvd1N0YXJ0VGlj"
    "ayBmb3IgdGhlIGN1cnJlbnQgcm93CiAgICBpbnQgbG8gPSAwLCBoaSA9IE5VTV9TT05HX1JPV1M7CiAg"
    "ICBmb3IgKGludCBfYnMgPSAwOyBfYnMgPCAxMjsgX2JzKyspIHsgIC8vIGxvZzIoMTkyMCspIOKJiCAx"
    "MQogICAgICAgIGlmIChsbyA+PSBoaSAtIDEpIGJyZWFrOwogICAgICAgIGludCBtaWQgPSAobG8gKyBo"
    "aSkgPj4gMTsKICAgICAgICBpZiAoZmxvYXQoZmV0Y2hUaWNrKG1pZCkpIDw9IHRvdGFsVGlja0YpIGxv"
    "ID0gbWlkOwogICAgICAgIGVsc2UgaGkgPSBtaWQ7CiAgICB9CiAgICBpbnQgZ2xvYmFsUm93ID0gbG87"
    "CiAgICBpZiAoZ2xvYmFsUm93ID49IE5VTV9TT05HX1JPV1MpIGdsb2JhbFJvdyA9IE5VTV9TT05HX1JP"
    "V1MgLSAxOwoKICAgIC8vIEZpbmQgc29uZ1BvcyB2aWEgbGluZWFyIHNlYXJjaCBvdmVyIHBhdFRpY2tP"
    "ZmZzZXQgKFNPTkdfTEVOR1RIIOKJpCAxMjgsIGZhc3QgZW5vdWdoKQogICAgaW50IHNwID0gU09OR19M"
    "RU5HVEggLSAxOwogICAgZm9yIChpbnQgX2kgPSAwOyBfaSA8IFNPTkdfTEVOR1RIIC0gMTsgX2krKykg"
    "ewogICAgICAgIGlmIChwYXRUaWNrT2Zmc2V0W19pICsgMV0gPiBnbG9iYWxSb3cpIHsgc3AgPSBfaTsg"
    "YnJlYWs7IH0KICAgIH0KICAgIHBvcy5zb25nUG9zID0gc3A7CiAgICBwb3MucGF0dGVybiA9IHNvbmdQ"
    "b3NpdGlvbnNbc3BdOwogICAgcG9zLnJvdyAgICAgPSBnbG9iYWxSb3cgLSBwYXRUaWNrT2Zmc2V0W3Nw"
    "XTsKCiAgICBpbnQgcm93VGljayAgICA9IGZldGNoVGljayhnbG9iYWxSb3cpOwogICAgaW50IG5leHRU"
    "aWNrICAgPSBmZXRjaFRpY2soZ2xvYmFsUm93ICsgMSk7CiAgICBpbnQgcm93U3BlZWQgICA9IG5leHRU"
    "aWNrIC0gcm93VGljazsKICAgIHBvcy50aWNrICAgICAgID0gdG90YWxUaWNrRiAtIGZsb2F0KHJvd1Rp"
    "Y2spOwogICAgcG9zLnJvd1RpbWUgICAgPSBmbG9hdChyb3dTcGVlZCkgLyBUSUNLU19QRVJfU0VDOwog"
    "ICAgcmV0dXJuIHBvczsKfQoKLy8gNC1wb2ludCBjdWJpYyBCLXNwbGluZSBpbnRlcnBvbGF0aW9uLgov"
    "LyBCLXNwbGluZSBpcyBBUFBST1hJTUFUSU5HIChzbW9vdGhzIHRocm91Z2ggc2FtcGxlIHBvaW50cykg"
    "cmF0aGVyIHRoYW4KLy8gSU5URVJQT0xBVElORyAocGFzc2luZyBleGFjdGx5IHRocm91Z2ggdGhlbSks"
    "IGdpdmluZyBpbmhlcmVudCBsb3ctcGFzcwovLyBjaGFyYWN0ZXIuIEF0IDMtYml0IHF1YW50aXphdGlv"
    "biB0aGlzIHJlZHVjZXMgSEYgbm9pc2UgYnkgfjI1JSB2cwovLyBDYXRtdWxsLVJvbS9IZXJtaXRlIGF0"
    "IGlkZW50aWNhbCBjb3N0IChzYW1lIDQgdGFwcywgZGlmZmVyZW50IHdlaWdodHMpLgpmbG9hdCBnZXRT"
    "YW1wbGVGKGludCBiYXNlLCBmbG9hdCBmcG9zLCBpbnQgc21wTGVuLCBpbnQgbG9vcFN0YXJ0LCBpbnQg"
    "bG9vcExlbikgewogICAgaW50IGkgID0gaW50KGZwb3MpOwogICAgZmxvYXQgdCA9IGZwb3MgLSBmbG9h"
    "dChpKTsKICAgIGludCBpMCA9IGkgLSAxOwogICAgaWYgKGxvb3BMZW4gPiAyICYmIGkwIDwgbG9vcFN0"
    "YXJ0KSBpMCA9IGxvb3BTdGFydCArIGxvb3BMZW4gLSAxOwogICAgZWxzZSBpMCA9IG1heCgwLCBpMCk7"
    "CiAgICBmbG9hdCBwMCA9IGdldFNhbXBsZShiYXNlICsgaTApOwogICAgZmxvYXQgcDEgPSBnZXRTYW1w"
    "bGUoYmFzZSArIGkpOwogICAgZmxvYXQgcDIgPSBnZXRTYW1wbGUoYmFzZSArIG1pbihpICsgMSwgc21w"
    "TGVuICsgMTUpKTsKICAgIGZsb2F0IHAzID0gZ2V0U2FtcGxlKGJhc2UgKyBtaW4oaSArIDIsIHNtcExl"
    "biArIDE1KSk7CiAgICBmbG9hdCB0MiA9IHQgKiB0OwogICAgZmxvYXQgdDMgPSB0MiAqIHQ7CiAgICBm"
    "bG9hdCB3MCA9ICgxLjAgLSB0KSAqICgxLjAgLSB0KSAqICgxLjAgLSB0KSAvIDYuMDsKICAgIGZsb2F0"
    "IHcxID0gKDMuMCAqIHQzIC0gNi4wICogdDIgKyA0LjApIC8gNi4wOwogICAgZmxvYXQgdzIgPSAoLTMu"
    "MCAqIHQzICsgMy4wICogdDIgKyAzLjAgKiB0ICsgMS4wKSAvIDYuMDsKICAgIGZsb2F0IHczID0gdDMg"
    "LyA2LjA7CiAgICByZXR1cm4gdzAgKiBwMCArIHcxICogcDEgKyB3MiAqIHAyICsgdzMgKiBwMzsKfQoK"
    "IiIiCgogICAgaW1wb3J0IGJhc2U2NCBhcyBfYjY0ZQogICAgZ2V0X2NoYW5uZWxfb3V0cHV0ID0gX2I2"
    "NGUuYjY0ZGVjb2RlKCdMeThnZG1saVZHRmlJR2x6SUdSbFkyeGhjbVZrSUdGeklHRWdaMnh2WW1Gc0lH"
    "TnZibk4wSUdac2IyRjBXek15WFNCdVpXRnlJSFJvWlNCMGIzQWdiMllnUTI5dGJXOXVDaTh2SUNoeWFX"
    "ZG9kQ0JoWm5SbGNpQndaWEpwYjJSVVlXSnNaU2t1SUVSdmJpZDBJSEpsWkdWamJHRnlaU0JwZENCb1pY"
    "SmxMZ29LWm14dllYUWdaMlYwUTJoaGJtNWxiRTkxZEhCMWRDaHBiblFnWTJnc0lHWnNiMkYwSUhScGJX"
    "VXNJRkJ2YzJsMGFXOXVJSEJ2Y3l3Z1pteHZZWFFnY205M1ZHbHRaU2tnZXdvS0lDQWdJQzh2SUZOMFpY"
    "QWdNVG9nWm1sdVpDQnRiM04wTFhKbFkyVnVkR3g1TFhSeWFXZG5aWEpsWkNCdWIzUmxJRzl1SUhSb2FY"
    "TWdZMmhoYm01bGJBb2dJQ0FnVG05MFpTQjBjbWxuVG05MFpTQTlJR2RsZEU1dmRHVW9jRzl6TG5OdmJt"
    "ZFFiM01zSUhCdmN5NXliM2NzSUdOb0tUc0tJQ0FnSUdsdWRDQWdkSEpwWjFKdmR5QWdQU0J3YjNNdWNt"
    "OTNPd29nSUNBZ2FXNTBJQ0IwY21sblVHRjBJQ0E5SUhCdmN5NXpiMjVuVUc5ek93b2dJQ0FnYVdZZ0tI"
    "UnlhV2RPYjNSbExtbHVjM1J5ZFcxbGJuUWdQRDBnTUNCOGZDQjBjbWxuVG05MFpTNXdaWEpwYjJRZ1BE"
    "MGdNQ2tnZXdvZ0lDQWdJQ0FnSUdsdWRDQnpVaUE5SUhCdmN5NXliM2NzSUhOUUlEMGdjRzl6TG5OdmJt"
    "ZFFiM003Q2lBZ0lDQWdJQ0FnWm05eUlDaHBiblFnYkdJZ1BTQXhPeUJzWWlBOElEWTBPeUJzWWlzcktT"
    "QjdDaUFnSUNBZ0lDQWdJQ0FnSUhOU0xTMDdDaUFnSUNBZ0lDQWdJQ0FnSUdsbUlDaHpVaUE4SURBcElI"
    "c0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2h6VUNBK0lEQXBJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lD"
    "QWdJQ0FnSUNCelVDMHRPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUhOU0lEMGdjR0YwVTNSaGNu"
    "UlNiM2RiYzFCZElDc2dLSEJoZEZKdmQwOW1abk5sZEZ0elVDc3hYU0F0SUhCaGRGSnZkMDltWm5ObGRG"
    "dHpVRjBwSUMwZ01Uc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lIMGdaV3h6WlNCN0lHSnlaV0ZyT3lCOUNp"
    "QWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ1RtOTBaU0J3Y21WMklEMGdaMlYwVG05MFpT"
    "aHpVQ3dnYzFJc0lHTm9LVHNLSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLSEJ5WlhZdWFXNXpkSEoxYldWdWRD"
    "QStJREFnZkh3Z2NISmxkaTV3WlhKcGIyUWdQaUF3S1NCN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCMGNt"
    "bG5UbTkwWlNBOUlIQnlaWFk3SUhSeWFXZFNiM2NnUFNCelVqc2dkSEpwWjFCaGRDQTlJSE5RT3dvZ0lD"
    "QWdJQ0FnSUNBZ0lDQWdJQ0FnWW5KbFlXczdDaUFnSUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNCOUNp"
    "QWdJQ0I5Q2dvZ0lDQWdhV1lnS0hSeWFXZE9iM1JsTG1sdWMzUnlkVzFsYm5RZ1BEMGdNQ0I4ZkNCMGNt"
    "bG5UbTkwWlM1cGJuTjBjblZ0Wlc1MElENGdNekVnZkh3Z2RISnBaMDV2ZEdVdWNHVnlhVzlrSUR3OUlE"
    "QXBDaUFnSUNBZ0lDQWdjbVYwZFhKdUlEQXVNRHNLQ2lBZ0lDQlRZVzF3YkdWSmJtWnZJSE50Y0NBOUlI"
    "TmhiWEJzWlhOYmRISnBaMDV2ZEdVdWFXNXpkSEoxYldWdWRDQXRJREZkT3dvZ0lDQWdhV1lnS0hOdGND"
    "NXNaVzVuZEdnZ1BUMGdNQ2tnY21WMGRYSnVJREF1TURzS0NpQWdJQ0F2THlCVWFXTnJMV0poYzJWa0lH"
    "VnNZWEJ6WldRNklHbHViR2x1WlNCSFVpQmpiMjF3ZFhSaGRHbHZiaXdnYzJ0cGNDQnVZVzFsWkNCcGJu"
    "UmxjbTFsWkdsaGRHVnpDaUFnSUNCbWJHOWhkQ0JsYkdGd2MyVmtJRDBnS0dac2IyRjBLR1psZEdOb1ZH"
    "bGpheWh3WVhSVWFXTnJUMlptYzJWMFczQnZjeTV6YjI1blVHOXpYU3NvY0c5ekxuSnZkeTF3WVhSVGRH"
    "RnlkRkp2ZDF0d2IzTXVjMjl1WjFCdmMxMHBLU2tLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0t5"
    "QndiM011ZEdsamF3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0F0SUdac2IyRjBLR1psZEdOb1ZH"
    "bGpheWh3WVhSVWFXTnJUMlptYzJWMFczUnlhV2RRWVhSZEt5aDBjbWxuVW05M0xYQmhkRk4wWVhKMFVt"
    "OTNXM1J5YVdkUVlYUmRLU2twS1FvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDOGdWRWxEUzFOZlVF"
    "VlNYMU5GUXpzS0lDQWdJR2xtSUNobGJHRndjMlZrSUR3Z01DNHdLU0J5WlhSMWNtNGdNQzR3T3dvS0lD"
    "QWdJR2x1ZENCZmNHTjBJRDBnYVc1MEtIQnZjeTUwYVdOcktUc0tJQ0FnSUU1dmRHVWdYM0JqY2lBOUlH"
    "ZGxkRTV2ZEdVb2NHOXpMbk52Ym1kUWIzTXNJSEJ2Y3k1eWIzY3NJR05vS1RzS0NpQWdJQ0F2THlEaWxJ"
    "RGlsSUFnUTI5dFltbHVaV1FnWm05eWQyRnlaQ0J6WTJGdU9pQnlaV0oxYVd4a0lIQnBkR05vSUVGT1JD"
    "QjJiMngxYldVZ1puSnZiU0IwY21sbloyVnlJSFJ2SUdOMWNuSmxiblFnNHBTQTRwU0FDaUFnSUNCbWJH"
    "OWhkQ0JsWm1abFkzUnBkbVZRWlhKcGIyUWdQU0JtYkc5aGRDaDBjbWxuVG05MFpTNXdaWEpwYjJRcE93"
    "b2dJQ0FnWm14dllYUWdkR0Z5WjJWMFVHVnlhVzlrSUNBZ0lEMGdabXh2WVhRb2RISnBaMDV2ZEdVdWNH"
    "VnlhVzlrS1RzS0lDQWdJR2x1ZENBZ0lIWnZiSFZ0WlNBZ0lDQWdJQ0FnSUNBOUlITnRjQzUyYjJ4MWJX"
    "VTdDZ29nSUNBZ0x5OGdRWEJ3YkhrZ2RISnBaMmRsY2kxeWIzY2daV1ptWldOMGN6b2dRM2g0SUNoelpY"
    "UWdkbTlzS1N3Z1FYaDRMelo0ZUNBb2RtOXNJSE5zYVdSbElIQmhjblJwWVd3dlpuVnNiQ2tLSUNBZ0lH"
    "bG1JQ2gwY21sblRtOTBaUzVsWm1abFkzUWdQVDBnTUhoREtTQjdDaUFnSUNBZ0lDQWdkbTlzZFcxbElE"
    "MGdiV2x1S0hSeWFXZE9iM1JsTG5CaGNtRnRMQ0EyTkNrN0NpQWdJQ0I5SUdWc2MyVWdhV1lnS0hSeWFX"
    "ZE9iM1JsTG1WbVptVmpkQ0E5UFNBd2VFRWdmSHdnZEhKcFowNXZkR1V1WldabVpXTjBJRDA5SURCNE5p"
    "a2dld29nSUNBZ0lDQWdJR2x1ZENCZmMzVWdQU0FvZEhKcFowNXZkR1V1Y0dGeVlXMCtQalFwSmpCNFJp"
    "d2dYM05rSUQwZ2RISnBaMDV2ZEdVdWNHRnlZVzBtTUhoR093b2dJQ0FnSUNBZ0lHbHVkQ0JmYzNSbGND"
    "QTlJQ2hmYzNVK01Da2dQeUJmYzNVZ09pQXRYM05rT3dvZ0lDQWdJQ0FnSUdsbUlDaDBjbWxuVUdGMElE"
    "MDlJSEJ2Y3k1emIyNW5VRzl6SUNZbUlIUnlhV2RTYjNjZ1BUMGdjRzl6TG5KdmR5a2dld29nSUNBZ0lD"
    "QWdJQ0FnSUNCMmIyeDFiV1VnUFNCamJHRnRjQ2gyYjJ4MWJXVWdLeUJmYzNSbGNDQXFJRjl3WTNRc0lE"
    "QXNJRFkwS1RzS0lDQWdJQ0FnSUNCOUlHVnNjMlVnZXdvZ0lDQWdJQ0FnSUNBZ0lDQXZMeUJRWVhOMElI"
    "UnlhV2RuWlhJZ2NtOTNJT0tBbENCMWMyVWdkR2hoZENCeWIzY25jeUJoWTNSMVlXd2djM0JsWldRS0lD"
    "QWdJQ0FnSUNBZ0lDQWdhVzUwSUY5MGN5QTlJR1psZEdOb1ZHbGpheWh3WVhSVWFXTnJUMlptYzJWMFcz"
    "UnlhV2RRWVhSZEt5aDBjbWxuVW05M0xYQmhkRk4wWVhKMFVtOTNXM1J5YVdkUVlYUmRLU3N4S1FvZ0lD"
    "QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDMGdabVYwWTJoVWFXTnJLSEJoZEZScFkydFBabVp6WlhSYmRI"
    "SnBaMUJoZEYwcktIUnlhV2RTYjNjdGNHRjBVM1JoY25SU2IzZGJkSEpwWjFCaGRGMHBLVHNLSUNBZ0lD"
    "QWdJQ0FnSUNBZ2RtOXNkVzFsSUQwZ1kyeGhiWEFvZG05c2RXMWxJQ3NnWDNOMFpYQWdLaUFvWDNSekxU"
    "RXBMQ0F3TENBMk5DazdDaUFnSUNBZ0lDQWdmUW9nSUNBZ2ZRb0tJQ0FnSUM4dklFWnZjbmRoY21RZ2My"
    "Tmhiam9nY205M2N5QlRWRkpKUTFSTVdTQmlaWFIzWldWdUlIUnlhV2RuWlhJZ1lXNWtJR04xY25KbGJu"
    "UUtJQ0FnSUdsbUlDaDBjbWxuVUdGMElDRTlJSEJ2Y3k1emIyNW5VRzl6SUh4OElIUnlhV2RTYjNjZ0lU"
    "MGdjRzl6TG5KdmR5a2dld29nSUNBZ0lDQWdJR2x1ZENCZlpuQWdQU0IwY21sblVHRjBMQ0JmWm5JZ1BT"
    "QjBjbWxuVW05M0lDc2dNVHNLSUNBZ0lDQWdJQ0JwWmlBb1gyWnlJRDQ5SUhCaGRGTjBZWEowVW05M1cx"
    "OW1jRjBnS3lBb2NHRjBVbTkzVDJabWMyVjBXMTltY0NzeFhTQXRJSEJoZEZKdmQwOW1abk5sZEZ0Zlpu"
    "QmRLU2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQmZabkFyS3pzZ1gyWnlJRDBnS0Y5bWNDQThJRk5QVGtkZlRF"
    "Vk9SMVJJS1NBL0lIQmhkRk4wWVhKMFVtOTNXMTltY0YwZ09pQXdPd29nSUNBZ0lDQWdJSDBLSUNBZ0lD"
    "QWdJQ0JtYjNJZ0tHbHVkQ0JmWm1rZ1BTQXdPeUJmWm1rZ1BDQTJORHNnWDJacEt5c3BJSHNLSUNBZ0lD"
    "QWdJQ0FnSUNBZ2FXWWdLRjltY0NBK0lIQnZjeTV6YjI1blVHOXpJSHg4SUNoZlpuQWdQVDBnY0c5ekxu"
    "TnZibWRRYjNNZ0ppWWdYMlp5SUQ0OUlIQnZjeTV5YjNjcEtTQmljbVZoYXpzS0lDQWdJQ0FnSUNBZ0lD"
    "QWdhV1lnS0Y5bWNDQStQU0JUVDA1SFgweEZUa2RVU0NrZ1luSmxZV3M3Q2lBZ0lDQWdJQ0FnSUNBZ0lH"
    "bG1JQ2hmWm5JZ1BqMGdjR0YwVTNSaGNuUlNiM2RiWDJad1hTQXJJQ2h3WVhSU2IzZFBabVp6WlhSYlgy"
    "WndLekZkSUMwZ2NHRjBVbTkzVDJabWMyVjBXMTltY0YwcEtTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lD"
    "QmZabkFyS3pzZ1gyWnlJRDBnS0Y5bWNDQThJRk5QVGtkZlRFVk9SMVJJS1NBL0lIQmhkRk4wWVhKMFVt"
    "OTNXMTltY0YwZ09pQXdPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdZMjl1ZEdsdWRXVTdDaUFnSUNBZ0lD"
    "QWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lDQWdUbTkwWlNCZlptNGdQU0JuWlhST2IzUmxLRjltY0N3Z1gy"
    "WnlMQ0JqYUNrN0NpQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZlptNHVhVzV6ZEhKMWJXVnVkQ0ErSURBZ2ZI"
    "d2dYMlp1TG5CbGNtbHZaQ0ErSURBcElHSnlaV0ZyT3lBZ0x5OGdibVYzSUhSeWFXZG5aWElnWlc1a2N5"
    "QjBhR1VnYzJOaGJnb0tJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOXpaM0lnSUNBOUlIQmhkRlJwWTJ0UFpt"
    "WnpaWFJiWDJad1hTQXJJQ2hmWm5JZ0xTQndZWFJUZEdGeWRGSnZkMXRmWm5CZEtUc0tJQ0FnSUNBZ0lD"
    "QWdJQ0FnYVc1MElGOW1kV3hzSUNBOUlHWmxkR05vVkdsamF5aGZjMmR5SUNzZ01Ta2dMU0JtWlhSamFG"
    "UnBZMnNvWDNObmNpa2dMU0F4T3lBZ0x5OGdkR2xqYTNNZ01TNHVjM0JsWldRdE1Rb0tJQ0FnSUNBZ0lD"
    "QWdJQ0FnTHk4Z1VHbDBZMmdnWldabVpXTjBjd29nSUNBZ0lDQWdJQ0FnSUNCcFppQW9YMlp1TG1WbVpt"
    "VmpkQ0E5UFNBd2VERXBDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmxabVpsWTNScGRtVlFaWEpwYjJRZ1BT"
    "QnRZWGdvTVRFekxqQXNJR1ZtWm1WamRHbDJaVkJsY21sdlpDQXRJR1pzYjJGMEtGOW1iaTV3WVhKaGJT"
    "QXFJRjltZFd4c0tTazdDaUFnSUNBZ0lDQWdJQ0FnSUdWc2MyVWdhV1lnS0Y5bWJpNWxabVpsWTNRZ1BU"
    "MGdNSGd5S1FvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnWldabVpXTjBhWFpsVUdWeWFXOWtJRDBnYldsdUtE"
    "ZzFOaTR3TENCbFptWmxZM1JwZG1WUVpYSnBiMlFnS3lCbWJHOWhkQ2hmWm00dWNHRnlZVzBnS2lCZlpu"
    "VnNiQ2twT3dvZ0lDQWdJQ0FnSUNBZ0lDQmxiSE5sSUdsbUlDaGZabTR1WldabVpXTjBJRDA5SURCNE15"
    "a2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0dWbVptVmpkR2wyWlZCbGNtbHZaQ0E4SUhSaGNt"
    "ZGxkRkJsY21sdlpDa0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JsWm1abFkzUnBkbVZRWlhKcGIy"
    "UWdQU0J0YVc0b2RHRnlaMlYwVUdWeWFXOWtMQ0JsWm1abFkzUnBkbVZRWlhKcGIyUWdLeUJtYkc5aGRD"
    "aGZabTR1Y0dGeVlXMGdLaUJmWm5Wc2JDa3BPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdaV3h6WlNCcFpp"
    "QW9aV1ptWldOMGFYWmxVR1Z5YVc5a0lENGdkR0Z5WjJWMFVHVnlhVzlrS1FvZ0lDQWdJQ0FnSUNBZ0lD"
    "QWdJQ0FnSUNBZ0lHVm1abVZqZEdsMlpWQmxjbWx2WkNBOUlHMWhlQ2gwWVhKblpYUlFaWEpwYjJRc0lH"
    "Vm1abVZqZEdsMlpWQmxjbWx2WkNBdElHWnNiMkYwS0Y5bWJpNXdZWEpoYlNBcUlGOW1kV3hzS1NrN0Np"
    "QWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ0x5OGdWbTlzZFcxbElHVm1abVZqZEhNS0lD"
    "QWdJQ0FnSUNBZ0lDQWdaV3h6WlNCcFppQW9YMlp1TG1WbVptVmpkQ0E5UFNBd2VFTXBDaUFnSUNBZ0lD"
    "QWdJQ0FnSUNBZ0lDQjJiMngxYldVZ1BTQnRhVzRvWDJadUxuQmhjbUZ0TENBMk5DazdDaUFnSUNBZ0lD"
    "QWdJQ0FnSUdWc2MyVWdhV1lnS0Y5bWJpNWxabVpsWTNRZ1BUMGdNSGhCSUh4OElGOW1iaTVsWm1abFkz"
    "UWdQVDBnTUhnMktTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBiblFnWDNaMUlEMGdLRjltYmk1d1lY"
    "SmhiVDQrTkNrbU1IaEdMQ0JmZG1RZ1BTQmZabTR1Y0dGeVlXMG1NSGhHT3dvZ0lDQWdJQ0FnSUNBZ0lD"
    "QWdJQ0FnZG05c2RXMWxJRDBnWTJ4aGJYQW9kbTlzZFcxbElDc2dLRjkyZFQ0d1AxOTJkVG90WDNaa0tT"
    "QXFJRjltZFd4c0xDQXdMQ0EyTkNrN0NpQWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ1gy"
    "WnlLeXM3Q2lBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmWm5JZ1BqMGdOalFwSUhzZ1gyWnlJRDBnTURzZ1gy"
    "WndLeXM3SUgwS0lDQWdJQ0FnSUNCOUNnb2dJQ0FnSUNBZ0lDOHZJRU4xY25KbGJuUWdjbTkzSUhCaGNu"
    "UnBZV3dnS0c1dmJpMTBjbWxuWjJWeUlISnZkeUJ2Ym14NUlPS0FsQ0IwY21sbloyVnlJR2hoYm1Sc1pX"
    "UWdZV0p2ZG1VcENpQWdJQ0FnSUNBZ2FXWWdLRjl3WTNJdWFXNXpkSEoxYldWdWRDQThQU0F3SUNZbUlG"
    "OXdZM0l1Y0dWeWFXOWtJRHc5SURBcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOXdZM0l1WldabVpX"
    "TjBJRDA5SURCNFF5a0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lIWnZiSFZ0WlNBOUlHMXBiaWhmY0dOeUxu"
    "QmhjbUZ0TENBMk5DazdDaUFnSUNBZ0lDQWdJQ0FnSUdWc2MyVWdhV1lnS0Y5d1kzSXVaV1ptWldOMElE"
    "MDlJREI0UVNCOGZDQmZjR055TG1WbVptVmpkQ0E5UFNBd2VEWXBJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lD"
    "QWdJR2x1ZENCZmRuVWdQU0FvWDNCamNpNXdZWEpoYlQ0K05Da21NSGhHTENCZmRtUWdQU0JmY0dOeUxu"
    "QmhjbUZ0SmpCNFJqc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lIWnZiSFZ0WlNBOUlHTnNZVzF3S0hadmJI"
    "VnRaU0FySUNoZmRuVStNRDlmZG5VNkxWOTJaQ2tnS2lCZmNHTjBMQ0F3TENBMk5DazdDaUFnSUNBZ0lD"
    "QWdJQ0FnSUgwS0lDQWdJQ0FnSUNCOUNpQWdJQ0I5Q2dvZ0lDQWdMeThnUTNWeWNtVnVkQ0J5YjNjZ2NH"
    "RnlkR2xoYkNCd2FYUmphQ0JsWm1abFkzUWdLR0Z3Y0d4cFpYTWdaWFpsYmlCdmJpQjBjbWxuWjJWeUlI"
    "SnZkeWtLSUNBZ0lHbG1JQ2hmY0dOeUxtVm1abVZqZENBOVBTQXdlREVwQ2lBZ0lDQWdJQ0FnWldabVpX"
    "TjBhWFpsVUdWeWFXOWtJRDBnYldGNEtERXhNeTR3TENCbFptWmxZM1JwZG1WUVpYSnBiMlFnTFNCbWJH"
    "OWhkQ2hmY0dOeUxuQmhjbUZ0SUNvZ1gzQmpkQ2twT3dvZ0lDQWdaV3h6WlNCcFppQW9YM0JqY2k1bFpt"
    "WmxZM1FnUFQwZ01IZ3lLUW9nSUNBZ0lDQWdJR1ZtWm1WamRHbDJaVkJsY21sdlpDQTlJRzFwYmlnNE5U"
    "WXVNQ3dnWldabVpXTjBhWFpsVUdWeWFXOWtJQ3NnWm14dllYUW9YM0JqY2k1d1lYSmhiU0FxSUY5d1kz"
    "UXBLVHNLSUNBZ0lHVnNjMlVnYVdZZ0tGOXdZM0l1WldabVpXTjBJRDA5SURCNE15QW1KaUJmY0dOeUxt"
    "bHVjM1J5ZFcxbGJuUWdQVDBnTUNBbUppQmZjR055TG5CbGNtbHZaQ0E5UFNBd0tTQjdDaUFnSUNBZ0lD"
    "QWdhV1lnS0dWbVptVmpkR2wyWlZCbGNtbHZaQ0E4SUhSaGNtZGxkRkJsY21sdlpDa0tJQ0FnSUNBZ0lD"
    "QWdJQ0FnWldabVpXTjBhWFpsVUdWeWFXOWtJRDBnYldsdUtIUmhjbWRsZEZCbGNtbHZaQ3dnWldabVpX"
    "TjBhWFpsVUdWeWFXOWtJQ3NnWm14dllYUW9YM0JqY2k1d1lYSmhiU0FxSUY5d1kzUXBLVHNLSUNBZ0lD"
    "QWdJQ0JsYkhObElHbG1JQ2hsWm1abFkzUnBkbVZRWlhKcGIyUWdQaUIwWVhKblpYUlFaWEpwYjJRcENp"
    "QWdJQ0FnSUNBZ0lDQWdJR1ZtWm1WamRHbDJaVkJsY21sdlpDQTlJRzFoZUNoMFlYSm5aWFJRWlhKcGIy"
    "UXNJR1ZtWm1WamRHbDJaVkJsY21sdlpDQXRJR1pzYjJGMEtGOXdZM0l1Y0dGeVlXMGdLaUJmY0dOMEtT"
    "azdDaUFnSUNCOUNnb2dJQ0FnTHk4Z1FYSndaV2RuYVc4Z0tFVm1abVZqZENBd2VIa3BJT0tBbENCd2VX"
    "MXZaQ2R6SUc5eVpHVnlJR2x6SUdKaGMyWGlocEpZS0docFoyZ3A0b2FTV1Noc2IzY3BDaUFnSUNCcFpp"
    "QW9YM0JqY2k1bFptWmxZM1FnUFQwZ01IZ3dJQ1ltSUY5d1kzSXVjR0Z5WVcwZ0lUMGdNQ2tnZXdvZ0lD"
    "QWdJQ0FnSUdsdWRDQmZZWEp3VTNSbGNDQTlJR2x1ZENod2IzTXVkR2xqYXlrZ0xTQnBiblFvY0c5ekxu"
    "UnBZMnNnTHlBekxqQXBJQ29nTXpzS0lDQWdJQ0FnSUNBdkx5QmxabVpsWTNScGRtVlFaWEpwYjJRZ1NW"
    "TWdZbUZ6WlZCbGNtbHZaQ0JvWlhKbElDaHVieUJtZFhKMGFHVnlJRzF2WkdsbWFXTmhkR2x2YmlCaVpX"
    "WnZjbVVnWVhKd0tRb2dJQ0FnSUNBZ0lHbG1JQ2hmWVhKd1UzUmxjQ0E5UFNBeEtRb2dJQ0FnSUNBZ0lD"
    "QWdJQ0JsWm1abFkzUnBkbVZRWlhKcGIyUWdQU0JsWm1abFkzUnBkbVZRWlhKcGIyUWdLaUJ3YjNjb01p"
    "NHdMQ0F0Wm14dllYUW9LRjl3WTNJdWNHRnlZVzBnUGo0Z05Da2dKaUF3ZUVZcElDOGdNVEl1TUNrN0Np"
    "QWdJQ0FnSUNBZ1pXeHpaU0JwWmlBb1gyRnljRk4wWlhBZ1BUMGdNaWtLSUNBZ0lDQWdJQ0FnSUNBZ1pX"
    "Wm1aV04wYVhabFVHVnlhVzlrSUQwZ1pXWm1aV04wYVhabFVHVnlhVzlrSUNvZ2NHOTNLREl1TUN3Z0xX"
    "WnNiMkYwS0Y5d1kzSXVjR0Z5WVcwZ0ppQXdlRVlwSUM4Z01USXVNQ2s3Q2lBZ0lDQjlDZ29nSUNBZ0x5"
    "OGdWbWxpY21GMGJ5QW9SV1ptWldOMElEUXZOaWtnNG9DVUlIVnpaWE1nWjJ4dlltRnNJSFpwWWxSaFln"
    "b2dJQ0FnZXdvZ0lDQWdJQ0FnSUdsdWRDQmZkbE1nUFNBd0xDQmZka1FnUFNBd093b2dJQ0FnSUNBZ0lH"
    "bG1JQ2gwY21sblRtOTBaUzVsWm1abFkzUWdQVDBnTUhnMElIeDhJSFJ5YVdkT2IzUmxMbVZtWm1WamRD"
    "QTlQU0F3ZURZcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnWDNaVElEMGdLSFJ5YVdkT2IzUmxMbkJoY21GdElE"
    "NCtJRFFwSUNZZ01IaEdPd29nSUNBZ0lDQWdJQ0FnSUNCZmRrUWdQU0FnZEhKcFowNXZkR1V1Y0dGeVlX"
    "MGdKaUF3ZUVZN0NpQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lHbG1JQ2gwY21sblVHRjBJRDA5SUhCdmN5"
    "NXpiMjVuVUc5eklIeDhJRjkyUkNBOVBTQXdLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lHWnZjaUFvYVc1MElG"
    "OTJhU0E5SURFN0lGOTJhU0E4UFNBeE5qc2dYM1pwS3lzcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lH"
    "bHVkQ0JmZG5JZ1BTQjBjbWxuVW05M0lDc2dYM1pwT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tG"
    "OTJjaUErUFNBMk5DQjhmQ0JmZG5JZ1BqMGdjRzl6TG5KdmR5a2dZbkpsWVdzN0NpQWdJQ0FnSUNBZ0lD"
    "QWdJQ0FnSUNCT2IzUmxJRjkyYmlBOUlHZGxkRTV2ZEdVb2RISnBaMUJoZEN3Z1gzWnlMQ0JqYUNrN0Np"
    "QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9YM1p1TG1sdWMzUnlkVzFsYm5RZ1BpQXdJSHg4SUY5MmJp"
    "NXdaWEpwYjJRZ1BpQXdLU0JpY21WaGF6c0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2dvWDNadUxt"
    "Vm1abVZqZENBOVBTQXdlRFFnZkh3Z1gzWnVMbVZtWm1WamRDQTlQU0F3ZURZcElDWW1JRjkyYmk1d1lY"
    "SmhiU0FoUFNBd0tTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnWDNaVElEMGdLRjkyYmk1d1lY"
    "SmhiU0ErUGlBMEtTQW1JREI0UmpzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmZka1FnUFNBZ1gz"
    "WnVMbkJoY21GdElDWWdNSGhHT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNBZ0lD"
    "QjlDaUFnSUNBZ0lDQWdmUW9nSUNBZ0lDQWdJR2xtSUNoZmRrUWdQaUF3S1NCN0NpQWdJQ0FnSUNBZ0lD"
    "QWdJR2x1ZENCZmRsQWdQU0JwYm5Rb1pXeGhjSE5sWkNBcUlGUkpRMHRUWDFCRlVsOVRSVU1wSUNvZ1gz"
    "WlRJQ1lnTmpNN0NpQWdJQ0FnSUNBZ0lDQWdJR1pzYjJGMElGOTJSR1ZzZEdFZ1BTQW9kbWxpVkdGaVcx"
    "OTJVQ0FtSURNeFhTQXFJR1pzYjJGMEtGOTJSQ2twSUM4Z01USTRMakE3Q2lBZ0lDQWdJQ0FnSUNBZ0lH"
    "Vm1abVZqZEdsMlpWQmxjbWx2WkNBclBTQW9YM1pRSUR3Z016SXBJRDhnWDNaRVpXeDBZU0E2SUMxZmRr"
    "UmxiSFJoT3dvZ0lDQWdJQ0FnSUgwS0lDQWdJSDBLQ2lBZ0lDQXZMeUJTWlc1a1pYSWdjMkZ0Y0d4bENp"
    "QWdJQ0JtYkc5aGRDQm1jbVZ4SUNBZ0lDQWdJRDBnY0dWeWFXOWtWRzlHY21WeEtHMWhlQ2d4TENCcGJu"
    "UW9aV1ptWldOMGFYWmxVR1Z5YVc5a0tTa3BPd29nSUNBZ1pteHZZWFFnWmxOaGJYQnNaVkJ2Y3lBOUlH"
    "VnNZWEJ6WldRZ0tpQm1jbVZ4SUM4Z1pteHZZWFFvYzIxd0xtSjNSbUZqZEc5eUtUc0tDaUFnSUNCcFpp"
    "QW9jMjF3TG14dmIzQk1aVzRnUGlBeUtTQjdDaUFnSUNBZ0lDQWdhV1lnS0daVFlXMXdiR1ZRYjNNZ1Bq"
    "MGdabXh2WVhRb2MyMXdMbXh2YjNCVGRHRnlkQ0FySUhOdGNDNXNiMjl3VEdWdUtTa0tJQ0FnSUNBZ0lD"
    "QWdJQ0FnWmxOaGJYQnNaVkJ2Y3lBOUlHWnNiMkYwS0hOdGNDNXNiMjl3VTNSaGNuUXBJQ3NnYlc5a0tH"
    "WlRZVzF3YkdWUWIzTWdMU0JtYkc5aGRDaHpiWEF1Ykc5dmNGTjBZWEowS1N3Z1pteHZZWFFvYzIxd0xt"
    "eHZiM0JNWlc0cEtUc0tJQ0FnSUgwZ1pXeHpaU0JwWmlBb1psTmhiWEJzWlZCdmN5QStQU0JtYkc5aGRD"
    "aHpiWEF1YkdWdVozUm9LU2tnZXdvZ0lDQWdJQ0FnSUhKbGRIVnliaUF3TGpBN0NpQWdJQ0I5Q2lBZ0lD"
    "QnBaaUFvWmxOaGJYQnNaVkJ2Y3lBOElEQXVNQ2tnY21WMGRYSnVJREF1TURzS0NpQWdJQ0JtYkc5aGRD"
    "QnpJRDBnWjJWMFUyRnRjR3hsUmloemJYQXVjM1JoY25Rc0lHWlRZVzF3YkdWUWIzTXNJSE50Y0M1c1pX"
    "NW5kR2dzSUhOdGNDNXNiMjl3VTNSaGNuUXNJSE50Y0M1c2IyOXdUR1Z1S1RzS0lDQWdJSEpsZEhWeWJp"
    "QnpJQ29nWm14dllYUW9kbTlzZFcxbEtTQXZJRFkwTGpBN0NuMEsnKS5kZWNvZGUoJ3V0Zi04JykKCiAg"
    "ICAjIEFzc2VtYmxlCiAgICByZXR1cm4gaGVhZGVyICsgbWV0YSArICIiLmpvaW4oZGF0YV9hcnJheXMp"
    "ICsgIlxuIiArIHRhYmxlcyArIGZldGNoZXJzICsgZGVjb2RlcnMgKyBnZXRfY2hhbm5lbF9vdXRwdXQK"
    "CgppZiBfX25hbWVfXyA9PSAnX19tYWluX18nOgogICAgbW9kX3BhdGggPSBzeXMuYXJndlsxXSBpZiBs"
    "ZW4oc3lzLmFyZ3YpID4gMSBlbHNlICcvbW50L3VzZXItZGF0YS91cGxvYWRzLzEyVEguTU9EJwogICAg"
    "b3V0X3BhdGggPSBzeXMuYXJndlsyXSBpZiBsZW4oc3lzLmFyZ3YpID4gMiBlbHNlICcvaG9tZS9jbGF1"
    "ZGUvbW9kX2NydW5jaC8xMlRIX2NydW5jaF9jb21tb24uZ2xzbCcKICAgIG1haW4obW9kX3BhdGgsIG91"
    "dF9wYXRoKQo="
)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='MOD/S3M Player - Generates HTML player + ShaderToy GLSL + PNG samples')
    parser.add_argument('modfile', help='MOD or S3M file to play')
    parser.add_argument('--downsample', type=int, default=2,
                        help='RVQ sample downsample factor: 1=full-res K=(64,32) ~80KB 27dB, '
                             '2=recommended K=(512,256) ~77KB 38dB, 4=smallest K=(512,256) ~40KB 37dB')
    args = parser.parse_args()
    
    # Detect file format
    fmt = detect_module_format(args.modfile)
    print(f"📻 Detected format: {fmt}")
    
    if fmt == 'S3M':
        s3m = S3MFile(args.modfile)
        print(f"🎵 {s3m.title}")
        print(f"   Instruments: {s3m.num_instruments}, Patterns: {s3m.num_patterns}, Channels: {s3m.num_channels}")
        print(f"   Speed: {s3m.initial_speed}, Tempo: {s3m.initial_tempo}")
        
        # Convert S3M to MOD-compatible structure for now
        # (We'll improve this later)
        mod = type('obj', (object,), {
            'title': s3m.title,
            'samples': s3m.instruments,
            'num_patterns': len(s3m.orders),
            'song_length': len(s3m.orders),
            'song_positions': s3m.orders,
            'orders': s3m.orders,
            'patterns': [s3m.patterns[i] for i in s3m.orders if i < len(s3m.patterns)],
            'num_channels': s3m.num_channels,
            'initial_speed': s3m.initial_speed,
            'initial_tempo': s3m.initial_tempo,
            'is_s3m': True
        })()
    elif fmt == 'MOD':
        mod = MODFile(args.modfile)
        mod.is_s3m = False
        mod.num_channels = 4
    else:
        raise ValueError(f"Unknown module format: {args.modfile}")
    
    # Note: sample downsampling is now handled inside vq_encoder_v2 (anti-aliased RVQ).
    # The --downsample flag controls the RVQ downsampling factor — no manual decimation needed.
    
    base_name = os.path.splitext(os.path.basename(args.modfile))[0]
    
    # Collect all samples with bandwidth-adaptive compression + zero-padding
    all_samples = []
    png_sample_bw = {}  # track bw_factor per sample index for sample_map
    for idx, sample in enumerate(mod.samples):
        if sample['data'] is not None and len(sample['data']) > 0:
            bf, compressed = bw_compress_sample(sample['data'])
            png_sample_bw[idx] = bf
            all_samples.extend(compressed.tolist())
            all_samples.extend([0] * 32)
        else:
            png_sample_bw[idx] = 1
    
    # RLE compression helper
    def rle_compress_with_row_breaks(data, row_size):
        """
        RLE compress [count, value, ...] but never let a run cross a row boundary.
        Returns (compressed_stream, seek_table) where seek_table[i] is the
        compressed stream offset at the start of decompressed row i.
        Guarantees every row starts on a clean [count, value] pair boundary.
        """
        if not data:
            return [], []
        compressed = []
        seek_table = []
        i = 0
        while i < len(data):
            if i % row_size == 0:
                seek_table.append(len(compressed))
            # Never cross the next row boundary
            next_break = ((i // row_size) + 1) * row_size
            max_run = min(255, next_break - i)
            count = 1
            value = data[i]
            while i + count < len(data) and data[i + count] == value and count < max_run:
                count += 1
            compressed.append(count)
            compressed.append(value)
            i += count
        return compressed, seek_table
    
    # S3M note to MOD period conversion table
    s3m_note_to_period = [
        1712, 1616, 1525, 1440, 1357, 1281, 1209, 1141, 1077, 1017, 961, 907,  # C-2 to B-2
        856, 808, 762, 720, 678, 640, 604, 570, 538, 508, 480, 453,           # C-3 to B-3
        428, 404, 381, 360, 339, 320, 302, 285, 269, 254, 240, 226,           # C-4 to B-4
        214, 202, 190, 180, 170, 160, 151, 143, 135, 127, 120, 113,           # C-5 to B-5
        107, 101, 95, 90, 85, 80, 76, 71, 67, 64, 60, 57,                     # C-6 to B-6
    ]
    
    def s3m_note_to_mod_period(note):
        """Convert S3M note number to MOD period"""
        if note == 255 or note == 254:  # No note or note cut
            return 0
        if note >= len(s3m_note_to_period):
            return 0
        return s3m_note_to_period[note]
    
    # ── Only embed patterns actually referenced in song_positions ──────────────
    # Many MODs allocate 64 pattern slots but use only 15-20.
    # We remap indices to a dense array and rewrite songPositions accordingly.
    num_channels = mod.num_channels if hasattr(mod, 'num_channels') else 4

    used_pat_indices = sorted(set(mod.song_positions))          # e.g. [0,1,3,5,7]
    remap = {old: new for new, old in enumerate(used_pat_indices)}  # {0:0,1:1,3:2,...}
    remapped_song_positions = [remap[p] for p in mod.song_positions]

    skipped = mod.num_patterns - len(used_pat_indices)
    if skipped > 0:
        print(f"   ✂️   Skipping {skipped} unused patterns "
              f"({mod.num_patterns} total → {len(used_pat_indices)} used)")

    # Update mod so GLSL defines use the remapped counts
    original_num_patterns   = mod.num_patterns
    mod.num_patterns        = len(used_pat_indices)
    mod.song_positions      = remapped_song_positions          # rewrite for GLSL embed

    pattern_bytes = []
    for pat_idx in used_pat_indices:
        pattern = mod.patterns[pat_idx]
        for row in pattern:
            for ch_idx in range(num_channels):
                ch = row[ch_idx] if ch_idx < len(row) else {}
                if mod.is_s3m:
                    sample = ch.get('instrument', 0)
                    period = s3m_note_to_mod_period(ch.get('note', 255))
                    effect = ch.get('command', 0)
                    param  = ch.get('info', 0)
                else:
                    sample = ch.get('sample', 0)
                    period = ch.get('period', 0)
                    effect = ch.get('effect', 0)
                    param  = ch.get('param', 0)
                sample_hi = (sample & 0xF0)
                sample_lo = (sample & 0x0F) << 4
                period_hi = (period >> 8) & 0x0F
                period_lo = period & 0xFF
                pattern_bytes.append(sample_hi | period_hi)
                pattern_bytes.append(period_lo)
                pattern_bytes.append(sample_lo | (effect & 0x0F))
                pattern_bytes.append(param)
                pattern_bytes.append(0)  # 5th byte

    bytes_per_row = num_channels * 5
    pattern_bytes_uncompressed = len(pattern_bytes)
    pattern_bytes, seek_table = rle_compress_with_row_breaks(pattern_bytes, bytes_per_row)
    pattern_bytes_compressed = len(pattern_bytes)
    print(f"   🗜️  Pattern RLE: {pattern_bytes_uncompressed} → {pattern_bytes_compressed} bytes ({pattern_bytes_compressed * 100 // pattern_bytes_uncompressed}%)")
    
    # Convert samples to bytes (signed → unsigned)
    sample_bytes = [int(int(s) + 128) % 256 for s in all_samples]
    
    # Magic signature: 'M', 'O', 'D', loopMode (PNG texture mode only)
    # NOTE: This is ONLY for external PNG files, not embedded data
    # loopMode: 255 = testing (10 sec loop), 0 = normal (full song)
    magic_bytes = [77, 79, 68, 0]  # 'MOD' + normal mode (change to 255 for testing)
    
    # Combine: [magic][compressed_pattern_bytes][sample_bytes]
    all_bytes = magic_bytes + pattern_bytes + sample_bytes
    pattern_size = len(pattern_bytes)  # Compressed size
    
    # Pack into 1024x1024 RGBA PNG (4 bytes per pixel)
    png_file = base_name + "_player_data.png"
    if len(all_bytes) > 0:
        from PIL import Image
        
        TEX_SIZE = 1024
        total_pixels = TEX_SIZE * TEX_SIZE
        total_capacity = total_pixels * 4  # 4 bytes per pixel (RGBA)
        
        # Create RGBA pixel data
        rgba_pixels = []
        for i in range(total_pixels):
            idx = i * 4
            r = all_bytes[idx] if idx < len(all_bytes) else 0
            g = all_bytes[idx+1] if idx+1 < len(all_bytes) else 0
            b = all_bytes[idx+2] if idx+2 < len(all_bytes) else 0
            a = all_bytes[idx+3] if idx+3 < len(all_bytes) else 0
            rgba_pixels.append((r, g, b, a))
        
        img = Image.new('RGBA', (TEX_SIZE, TEX_SIZE))
        img.putdata(rgba_pixels)
        img.save(png_file, format='PNG', optimize=True)
        png_size = os.path.getsize(png_file)
        
        print(f"   📦 Magic signature: 4 bytes ('MOD' + loopMode={magic_bytes[3]})")
        print(f"   📦 Pattern bytes (RLE compressed): {pattern_size}")
        print(f"   📦 Sample bytes: {len(sample_bytes)}")
        print(f"   📦 Total: {len(all_bytes)} bytes packed into {TEX_SIZE}×{TEX_SIZE} RGBA")
        print(f"   📦 Capacity: {total_capacity} bytes ({len(all_bytes) * 100 // total_capacity}% used)")

    # Generate instruction file
    instructions_file = base_name + "_shadertoy_instructions.txt"
    with open(instructions_file, 'w') as f:
        f.write(f"""ShaderToy Setup Instructions for {mod.title}
{'=' * 60}

METHOD 1: ShaderToy Unofficial Plugin (RECOMMENDED)
----------------------------------------------------
1. Install: https://github.com/patuwwy/ShaderToy-Unofficial-Plugin
2. Load your shader in ShaderToy
3. Plugin settings → Custom Textures → Add "{png_file}"
4. In Common tab: iChannel0 → Custom → Select "{png_file}"
5. Both Common and Sound read data via texelFetch

Data Info:
- Magic signature: 4 bytes at pixel 0 ('M','O','D',loopMode)
- Pattern bytes: {pattern_size}
- Sample bytes: {len(sample_bytes)}
- PNG: 1024×1024 RGBA ({png_size} bytes)
- Format: 4 bytes per pixel (texelFetch)
- Downsample: {args.downsample}x

Loop Modes (edit PNG pixel 0, alpha channel):
- Alpha = 0   : Normal mode (full song loop)
- Alpha = 255 : Testing mode (10 second loop)
- No signature: Waiting mode (silence until valid PNG)

To enable testing mode:
1. Open PNG in image editor
2. Set pixel (0,0) alpha to 255
3. Reload in ShaderToy → 10 sec loops!

METHOD 2: Embedded Data
--------------------------------------------
Not used - all data in texture for maximum compatibility!

File Overview:
--------------
{base_name}_player.html              - Standalone HTML player (works offline!)
{base_name}_shadertoy_common.glsl    - MOD data + helper functions
{base_name}_shadertoy_sound.glsl     - Complete ProTracker engine
{base_name}_shadertoy_image.glsl     - Visualization (Zuvuya demo style)
{base_name}_shadertoy_bufferA.glsl   - FFT spectrum analyzer + UI state
{png_file}              - Sample data as PNG texture

How to Use in ShaderToy:
-------------------------
1. Create new shader at shadertoy.com
2. Add "Common" tab  → paste {base_name}_shadertoy_common.glsl
3. Add "Sound"  tab  → paste {base_name}_shadertoy_sound.glsl
4. Add "Image"  tab  → paste {base_name}_shadertoy_image.glsl
5. Add "Buffer A" tab → paste {base_name}_shadertoy_bufferA.glsl

Channel setup (REQUIRED for spectrum + click-toggle):
  Image    iChannel0 = Alphabet texture  (shadertoy.com/view/4sf3RB)
  Image    iChannel1 = Buffer A
  Buffer A iChannel0 = Buffer A   ← self-reference (feedback loop)
  Sound    → no channels needed

6. Press PLAY! 🎵
   Click anywhere to toggle oscilloscope ↔ spectrum view

Effects Implemented:
--------------------
✅ 0xy - Arpeggio
✅ 1xx - Portamento Up
✅ 2xx - Portamento Down  
✅ 3xx - Tone Portamento
✅ 4xy - Vibrato
✅ 5xy - Tone Portamento + Volume Slide
✅ 6xy - Vibrato + Volume Slide
✅ Axy - Volume Slide
✅ Cxx - Set Volume

Architecture:
-------------
Common: Shared by Sound & Image (MOD data, helpers)
Sound: Generates audio independently from iTime
Image: Reads Sound output + calculates position from iTime
Both stay in sync via same time calculation!

Troubleshooting:
----------------
- No sound? Check Sound tab has no syntax errors
- Wrong notes? Verify getSample() reads from iChannel0 correctly
- Timing off? BPM/SPEED constants in Common tab

Generated by MOD2GLSL
{'=' * 60}
""")
    
    # Generate HTML player (now works for both MOD and S3M)
    html_file = base_name + "_player.html"
    create_fixed_player_html(mod, html_file, args.downsample, compress=True)

    # ShaderToy Common tab: VQ-encoded via embedded vq_encoder_v2
    glsl_common_file = base_name + "_shadertoy_common.glsl"
    try:
        import types as _types, base64 as _b64
        _vqmod = _types.ModuleType('vq_encoder_v2')
        _vqmod.__file__ = __file__
        exec(compile(_b64.b64decode(_VQ_ENCODER_B64).decode('utf-8'), 'vq_encoder_v2', 'exec'), _vqmod.__dict__)
        print(f"\n\U0001f3b5 Generating VQ-encoded Common tab...")
        _vqmod.main(args.modfile, glsl_common_file, K=256, weighted=True, downsample=args.downsample)
    except Exception as _e:
        print(f"   WARNING: VQ encoder failed ({_e}), falling back to built-in")
        _fb_glsl = base_name + "_shadertoy.glsl"
        create_shadertoy_glsl(mod, _fb_glsl, args.downsample, compress=True,
                             compressed_pattern_size=pattern_size,
                             pattern_bytes_data=pattern_bytes,
                             sample_bytes_data=sample_bytes,
                             seek_table=seek_table)
        glsl_common_file = _fb_glsl.replace('.glsl', '_common.glsl')

    # Sound / Image / Buffer A tabs from built-in emitter
    # Use a stub name that has NO overlap with _shadertoy.glsl patterns
    _glsl_stub = base_name + "_tmp_tabs_shadertoy.glsl"
    create_shadertoy_glsl(mod, _glsl_stub, args.downsample, compress=True,
                         compressed_pattern_size=pattern_size,
                         pattern_bytes_data=pattern_bytes,
                         sample_bytes_data=sample_bytes,
                         seek_table=seek_table)
    import os as _os2
    # create_shadertoy_glsl writes: _tmp_tabs_shadertoy_common/sound/image/bufferA.glsl
    for _ext in ('_sound.glsl', '_image.glsl', '_bufferA.glsl'):
        _src = _glsl_stub.replace('.glsl', _ext)
        _dst = base_name + "_shadertoy" + _ext
        if _os2.path.exists(_src): _os2.replace(_src, _dst)
    for _del in (_glsl_stub.replace('.glsl', '_common.glsl'), _glsl_stub):
        if _os2.path.exists(_del): _os2.remove(_del)

    # Summary
    bufA_file_short = base_name + "_shadertoy_bufferA.glsl"

    print(f"\n✅ Generated:")
    print(f"   🌐 HTML Player:    {html_file}")
    print(f"   📁 ShaderToy tabs: {glsl_common_file}  (VQ-encoded)")
    print(f"                      {base_name}_shadertoy_sound.glsl")
    print(f"                      {base_name}_shadertoy_image.glsl")
    print(f"                      {bufA_file_short}  ← Buffer A (FFT + state)")
    print(f"   🖼️  Sample PNG:     {png_file} ({png_size} bytes)")
    print(f"   📄 Instructions:   {instructions_file}")
    print(f"   🗜️  Compression:    RLE (patterns ~50%)")
    print(f"")
    print(f"   🔗 ShaderToy channel setup:")
    print(f"      Image    → iChannel0 = Alphabet texture  (shadertoy.com/view/4sf3RB)")
    print(f"      Image    → iChannel1 = Buffer A")
    print(f"      Buffer A → iChannel0 = Buffer A          (self-reference)")
    print(f"      Sound    → (no channels needed)")
    print(f"   🖱️  Click anywhere to toggle oscilloscope ↔ spectrum")
    if args.downsample > 1:
        print(f"   ⬇️  Downsampled:    {args.downsample}x")
    print(f"\n💡 ProTracker timing:")
    print(f"   - Tick-based playback")
    print(f"   - Notes trigger on tick 0 only")
    print(f"   - Persistent channel state")




if __name__ == '__main__':
    main()
