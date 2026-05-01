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
    then decimates by the best power-of-2 factor via raw stride (no LPF).
    Returns (bw_factor, compressed_int8_array).
    """
    d = data.astype(np.float32)
    n = len(d)
    if n < 32:
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
    
    # Pre-emphasis: y[n] = x[n] - alpha*x[n-1]  (first-order high-shelf boost)
    # Compensates the anti-alias LPF roll-off baked in by resample_poly.
    # No de-emphasis on playback → brighter, more present sound.
    # alpha=0.85 ≈ +6 dB shelf above ~3 kHz; raise to 0.93 for more air.
    _alpha = 0.85
    d_pre = np.empty_like(d)
    d_pre[0] = d[0]
    for i in range(1, len(d)):
        d_pre[i] = d[i] - _alpha * d[i-1]
    d = d_pre

    # Raw decimation — no LPF, full HF content preserved
    compressed = d[::best_factor]
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
            
            # Determine channel count from signature
            if signature in (b'M.K.', b'M!K!', b'M&K!', b'N.T.', b'FLT4', b'4CHN'):
                self.num_channels = 4
            elif signature == b'FLT8' or signature in (b'OCTA', b'CD81', b'OKTA'):
                self.num_channels = 8
            elif len(signature) == 4 and signature[1:4] == b'CHN' and signature[0:1].isdigit():
                self.num_channels = int(signature[0:1])
            elif len(signature) == 4 and signature[2:4] == b'CH' and signature[0:1].isdigit() and signature[1:2].isdigit():
                self.num_channels = int(signature[0:2])
            elif len(signature) == 4 and signature[:3] == b'TDZ' and signature[3:4].isdigit():
                self.num_channels = int(signature[3:4])
            else:
                self.num_channels = 4  # fallback
            
            for p in range(self.num_patterns):
                pattern = []
                for row in range(64):
                    channels = []
                    for ch in range(self.num_channels):
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
        # Standard 4-channel: M.K., M!K!, FLT4 (Startrekker), 4CHN (FT)
        # Extended N-channel: NCHN (1-9 channels), NNCH (10-99 channels), FLT8
        # All TakeTracker/FastTracker/OctaMED variants are valid MODs.
        if sig in [b'M.K.', b'M!K!', b'M&K!', b'N.T.', b'FLT4', b'FLT8',
                   b'OCTA', b'CD81', b'OKTA',
                   b'1CHN', b'2CHN', b'3CHN', b'4CHN', b'5CHN', b'6CHN',
                   b'7CHN', b'8CHN', b'9CHN']:
            return 'MOD'
        # NNCH form: "10CH"…"99CH" — first two chars are decimal digits
        if (len(sig) == 4 and sig[2:4] == b'CH'
                and sig[0:1].isdigit() and sig[1:2].isdigit()):
            return 'MOD'
        # TDZN form: "TDZ1"…"TDZ9" (TakeTracker dynamic Z-N channels)
        if len(sig) == 4 and sig[:3] == b'TDZ' and sig[3:4].isdigit():
            return 'MOD'
        
        return 'UNKNOWN'

def create_fixed_player_html(mod, output_file, downsample=1, compress=False, vec_dim=2):
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
                np.zeros(32, dtype=np.float32)   # zero-padding guard for loop boundary
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
    # Spectral bass detection (same logic as Sound tab)
    def _html_is_bass(s):
        if s['length'] < 256: return False
        name = s['name'].lower()
        for kw in ('bass', ' bs ', 'sub', '808', 'kick', 'bassdr'):
            if kw == 'bass' and 'brass' in name and 'bass' not in name.replace('brass', ''):
                continue
            if kw in name: return True
        try:
            data = np.frombuffer(s['data'].tobytes() if hasattr(s['data'], 'tobytes') else bytes(s['data']),
                                 dtype=np.int8).astype(np.float32) / 128.0
            if len(data) < 256: return False
            data = data[:min(len(data), 4096)]
            win = np.hanning(len(data))
            mag = np.abs(np.fft.rfft(data * win))
            if len(mag) < 4: return False
            freqs = np.fft.rfftfreq(len(data), 1.0 / 8363.0)
            mag[0] = 0
            total_energy = float(np.sum(mag**2)) + 1e-12
            centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12))
            sub_cut  = max(1, int(np.searchsorted(freqs, 100.0)))
            bass_cut = max(1, int(np.searchsorted(freqs, 250.0)))
            sub_ratio  = float(np.sum(mag[:sub_cut]**2))  / total_energy
            bass_ratio = float(np.sum(mag[:bass_cut]**2)) / total_energy
            peak_freq  = float(freqs[int(np.argmax(mag))])
            if centroid < 600.0 and peak_freq < 300.0 and (bass_ratio > 0.60 or sub_ratio > 0.40):
                return True
        except Exception:
            pass
        return False
    _html_bass_idx = [i+1 for i, s in enumerate(mod.samples)
                      if s['length'] > 0 and _html_is_bass(s)]
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
    bassSamples: {json.dumps(_html_bass_idx)},
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
                
                // Store effect for processing on all ticks.
                // ProTracker param-memory rules (corrected):
                //   1xx/2xx/3xx — param=0 → use last param for that effect
                //   4xx/7xx     — param=0 → use last 4xx/7xx (whole-byte memory)
                //   5xx/6xx     — param byte is VOL-SLIDE only (pitch part
                //                 inherits last 3xx/4xx automatically); 500
                //                 and 600 mean "no vol slide", do NOT rewrite
                //   Axx         — A00 = no-op in PT, NOT memory; do NOT rewrite
                this.channels[ch].effect = note.effect;
                let _newParam = note.param;
                if (_newParam === 0) {{
                    const _e = note.effect;
                    if (_e === 0x1 && this.channels[ch].lastPortaUpParam !== undefined)
                        _newParam = this.channels[ch].lastPortaUpParam;
                    else if (_e === 0x2 && this.channels[ch].lastPortaDownParam !== undefined)
                        _newParam = this.channels[ch].lastPortaDownParam;
                    else if (_e === 0x3 && this.channels[ch].lastTonePortaParam !== undefined)
                        _newParam = this.channels[ch].lastTonePortaParam;
                    else if (_e === 0x4 && this.channels[ch].lastVibratoParam !== undefined)
                        _newParam = this.channels[ch].lastVibratoParam;
                    else if (_e === 0x7 && this.channels[ch].lastTremoloParam !== undefined)
                        _newParam = this.channels[ch].lastTremoloParam;
                    // 5xx, 6xx, Axx: NO memory — param=0 is a valid command
                }} else {{
                    // Save non-zero param for this effect's memory
                    const _e = note.effect;
                    if (_e === 0x1) this.channels[ch].lastPortaUpParam = _newParam;
                    else if (_e === 0x2) this.channels[ch].lastPortaDownParam = _newParam;
                    else if (_e === 0x3) this.channels[ch].lastTonePortaParam = _newParam;
                    else if (_e === 0x4) this.channels[ch].lastVibratoParam = _newParam;
                    else if (_e === 0x7) this.channels[ch].lastTremoloParam = _newParam;
                }}
                this.channels[ch].effectParam = _newParam;
                
                // Process tick-0 effects
                this.processEffect(ch, note.effect, _newParam, true);
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
                
            case 0x4: // Vibrato — param is speed (high nibble) + depth (low nibble)
                if (tick0) {{
                    // Persist speed & depth — only overwrite when non-zero (nibble-level memory)
                    const vSpeedNew = (param >> 4) & 0x0F;
                    const vDepthNew = param & 0x0F;
                    if (vSpeedNew > 0) state.vibratoSpeed = vSpeedNew;
                    if (vDepthNew > 0) state.vibratoDepth = vDepthNew;
                }} else {{
                    // Advance vibrato LFO on ticks 1+
                    const vspeed = state.vibratoSpeed || 1;
                    state.vibratoPos = (state.vibratoPos + vspeed) % 64;
                }}
                break;

            case 0x6: // Continue Vibrato + Volume Slide — param is VOL-SLIDE ONLY
                // Vibrato part: continues with prior 4xx speed/depth (do NOT
                // touch state.vibratoSpeed/vibratoDepth — param is not for them)
                if (!tick0) {{
                    const vspeed = state.vibratoSpeed || 1;
                    state.vibratoPos = (state.vibratoPos + vspeed) % 64;
                    // Vol slide part — high nibble = up, low nibble = down
                    const vup   = (param >> 4) & 0x0F;
                    const vdown = param & 0x0F;
                    if (vup > 0) {{
                        state.volume = Math.min(64, state.volume + vup);
                    }} else if (vdown > 0) {{
                        state.volume = Math.max(0, state.volume - vdown);
                    }}
                    state.currentVolume = state.volume;
                }}
                break;
                
            case 0x5: // Continue Tone Porta + Volume Slide — param is VOL-SLIDE ONLY
                // Pitch slide continues with prior 3xx slide rate (state.lastTonePortaParam)
                if (!tick0 && state.targetPeriod > 0) {{
                    const tpRate = state.lastTonePortaParam || 0;
                    if (tpRate > 0) {{
                        if (state.period < state.targetPeriod) {{
                            state.period = Math.min(state.targetPeriod, state.period + tpRate);
                        }} else if (state.period > state.targetPeriod) {{
                            state.period = Math.max(state.targetPeriod, state.period - tpRate);
                        }}
                        state.basePeriod = state.period;
                    }}
                    // Vol slide part
                    const vup   = (param >> 4) & 0x0F;
                    const vdown = param & 0x0F;
                    if (vup > 0) {{
                        state.volume = Math.min(64, state.volume + vup);
                    }} else if (vdown > 0) {{
                        state.volume = Math.max(0, state.volume - vdown);
                    }}
                    state.currentVolume = state.volume;
                }}
                break;

            case 0x7: // Tremolo — param is speed (high nib) + depth (low nib), modulates volume
                if (tick0) {{
                    const tSpeedNew = (param >> 4) & 0x0F;
                    const tDepthNew = param & 0x0F;
                    if (tSpeedNew > 0) state.tremoloSpeed = tSpeedNew;
                    if (tDepthNew > 0) state.tremoloDepth = tDepthNew;
                }} else {{
                    // Advance tremolo LFO on ticks 1+
                    const tspeed = state.tremoloSpeed || 1;
                    state.tremoloPos = ((state.tremoloPos || 0) + tspeed) % 64;
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
                    
                    // PhatBass: per-sample when bass samples were detected, otherwise mix-wide.
                    // bassSamples.length == 0 → no detection → apply to ALL channels at reduced depth.
                    const noBassDetected = modData.bassSamples.length === 0;
                    const isBass = noBassDetected
                        ? true   // mix-wide mode
                        : modData.bassSamples.includes(state.sample + 1);
                    const phatScale = noBassDetected ? 0.25 : 1.0;
                    if (this._phatBass && this._phatBassDepth > 0 && isBass) {{
                        // Add to whichever bus this channel feeds
                        const sScaled = s * phatScale;
                        if (isSurrCh) [surrL, surrR] = this._phatBass.process(sScaled, surrL, surrR);
                        else          [mixL, mixR]   = this._phatBass.process(sScaled, mixL, mixR);
                    }}
                    
                    const effectivePeriod = this.getEffectivePeriod(ch);
                    const smpFt = (modData.sampleMap[state.sample] || {{}}).finetune || 0;
                    // samplePos is in original (uncompressed) sample space.
                    // getSampleData maps it to compressed space via pos/bw_factor — so
                    // freq must NOT be divided by bw_factor here (would double-divide).
                    const freq = this.periodToFreq(effectivePeriod, smpFt);
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
            
            // ── FAT4X harmonic exciter ─────────────────────────────
            const FAT_AMOUNT = 1.0;  // matches FAT4X (FIR weights sum to 1.0)
            const fat_cs1 = (x) => {{
                const x2=x*x,x4=x2*x2,x6=x4*x2,x8=x4*x4,x10=x4*x6,x12=x6*x6;
                return 0.4375 - 0.3228759765625*x2 + 0.1123046875*x4
                     - 0.50537109375*x6 + 0.1993408203125*x8
                     + 0.634521484375*x10 - 0.6513671875*x12;
            }};
            outL = outL * (1.0 + fat_cs1(outL) * FAT_AMOUNT);
            outR = outR * (1.0 + fat_cs1(outR) * FAT_AMOUNT);
            
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
                          pattern_bytes_data=None, sample_bytes_data=None, seek_table=None, vec_dim=2, viz=1):
    """Generate ShaderToy GLSL code with texture-based OR embedded data.
    viz: 0=None, 1=Reactive 001 (default), 2=Fluxline Surfer, 3=Zuvuya,
         4=Maya tunnel-warp, 5=Dodecahedron (Philip Bertani)"""

    # Human-readable visualizer name (stamped into every tab header).
    _VIZ_NAMES = {
        0: "None (black backdrop)",
        1: "Reactive 001 (PAEz fork — SDF circles + cosmic web)",
        2: "Fluxline Surfer (mrange — DR2 dodecahedron + glowtracer)",
        3: "Zuvuya (city/stars + audio-reactive curtain)",
        4: "Maya (raymarched fractal tunnel-warp)",
        5: "Dodecahedron (Philip Bertani — DR2 IFS fractal raymarcher)",
    }
    viz_name = _VIZ_NAMES.get(viz, f"viz{viz}")

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
   GLSL MOD Player v1.37 (c)2026 Orblivius
   3D Surround, PhatBass, Comb Reverb, FAT, RVQ sample compression, configurable resampler
   COMMON TAB
   Visualizer: {viz_name}
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

// Linear interpolation with loop-aware next-sample clamping.
// AA is provided by the RVQ anti-aliased downsampling of the source sample.
float getSampleF(int base, float fpos, int smpLen, int loopStart, int loopLen) {{
    int i = int(fpos);
    float t = fpos - float(i);
    int i1 = i + 1;
    if (loopLen > 2 && i1 >= loopStart + loopLen)
        i1 = loopStart;
    else
        i1 = min(i1, smpLen - 1);
    return mix(getSample(base + i), getSample(base + i1), t);
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
    # Build bass sample boolean array for GLSL.
    # Detection strategy (any one is sufficient):
    #   1. Sample name contains "bass", "sub", "kick", "808", "low", or "bs"
    #   2. Spectral analysis: dominant frequency below 250 Hz when played at C-2
    #      (period 428, the "natural" pitch defined by Amiga sample rate 8363 Hz).
    # The spectral check catches numerically-named bass samples that name-match misses.
    def _is_bass_sample(s):
        if s['length'] < 256:  # too short to analyze
            return False
        name = s['name'].lower()
        # Name keywords (avoid false positives like "brass" containing "bass")
        for kw in ('bass', ' bs ', 'sub', '808', 'kick', 'bassdr'):
            # Tolerate kw at any position EXCEPT for "bass" inside "brass"
            if kw == 'bass' and 'brass' in name and 'bass' not in name.replace('brass', ''):
                continue
            if kw in name:
                return True
        # Spectral check
        try:
            data = np.frombuffer(s['data'].tobytes() if hasattr(s['data'], 'tobytes') else bytes(s['data']),
                                 dtype=np.int8).astype(np.float32) / 128.0
            if len(data) < 256: return False
            data = data[:min(len(data), 4096)]
            win = np.hanning(len(data))
            mag = np.abs(np.fft.rfft(data * win))
            if len(mag) < 4: return False
            sr_native = 8363.0
            freqs = np.fft.rfftfreq(len(data), 1.0 / sr_native)
            mag[0] = 0  # ignore DC
            total_energy = float(np.sum(mag**2)) + 1e-12
            # Spectral centroid (energy-weighted mean frequency).  Real bass has
            # most of its mass concentrated in the low band even after harmonics
            # are counted — centroid is typically below 500 Hz.  Instruments
            # with a low *fundamental* but rich highs (guitar, snare, piano)
            # have centroid > 800 Hz and should NOT be tagged.
            centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12))
            # Energy under 100 Hz (sub-bass) and 250 Hz (bass)
            sub_cut = max(1, int(np.searchsorted(freqs, 100.0)))
            bass_cut = max(1, int(np.searchsorted(freqs, 250.0)))
            sub_ratio  = float(np.sum(mag[:sub_cut]**2))  / total_energy
            bass_ratio = float(np.sum(mag[:bass_cut]**2)) / total_energy
            peak_freq  = float(freqs[int(np.argmax(mag))])
            # Bass criteria — ALL must hold:
            #  - spectral centroid below 600 Hz (rules out bright instruments
            #    with a low fundamental)
            #  - peak below 300 Hz
            #  - at least 60% of energy below 250 Hz OR 40% below 100 Hz
            if centroid < 600.0 and peak_freq < 300.0 and (bass_ratio > 0.60 or sub_ratio > 0.40):
                return True
        except Exception:
            pass
        return False
    
    bass_sample_flags = [
        'true' if (i < len(mod.samples) and mod.samples[i]['length'] > 0
                   and _is_bass_sample(mod.samples[i])) else 'false'
        for i in range(31)
    ]
    # Diagnostic: print which samples were tagged
    _bass_idx = [i+1 for i, f in enumerate(bass_sample_flags) if f == 'true' and i < len(mod.samples)]
    if _bass_idx:
        _names = ', '.join(f"#{i} '{mod.samples[i-1]['name'].strip()}'" for i in _bass_idx)
        print(f"   🔊 Bass samples (PhatBass targets): {_names}")
    else:
        print(f"   🔊 No bass samples detected — PhatBass effect inactive")
    bass_flags_str = ', '.join(bass_sample_flags)
    # Auto mix-wide fallback when no bass samples detected
    phatbass_mix_mode = 1 if not _bass_idx else 0
    
    sound_glsl = f"""/* ============================================================================
   GLSL MOD Player v1.37 (c)2026 Orblivius
   3D Surround, PhatBass, Comb Reverb, FAT, RVQ sample compression, configurable resampler
   SOUND TAB
   Visualizer: {viz_name}
   Contact: subband@gmail.com or
            subband@protonmail.com
   GIT:     https://github.com/mewza/mod2glsl
  ============================================================================ */
// getByte / getPatternByte / getSample / getNote / getChannelOutput are in Common.

// Bass sample flags (true = instrument detected as bass) — for PhatBass
const bool isBass[31] = bool[]({bass_flags_str});

// PhatBass routing: 0 = per-sample (uses isBass[]), 1 = mix-wide (no detection)
#define PHATBASS_MIX_MODE {phatbass_mix_mode}

// ── FAT4X harmonic exciter helper ────────────────────────────────────────
// cs1 polynomial waveshaper (even harmonics only) from FAT4X by Orblivius.
// Even-power series → zero at x=0, adds warm even harmonics, soft clip near ±1.
float fat_cs1(float x) {{
    float x2=x*x, x4=x2*x2, x6=x4*x2, x8=x4*x4, x10=x4*x6, x12=x6*x6;
    return 0.4375 - 0.3228759765625*x2 + 0.1123046875*x4
         - 0.50537109375*x6 + 0.1993408203125*x8
         + 0.634521484375*x10 - 0.6513671875*x12;
}}

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
    
    // ── PhatBass — bass enhancement (cross-panned allpass) ─────────────────
    // Two modes:
    //  1. Per-sample: process only channels playing bass-detected samples.
    //     Cleaner — leaves leads/pads alone — but requires reliable detection.
    //  2. Mix-wide: process a low-passed copy of the full mix.  Works on any
    //     song without bass detection, but slightly colors mid-bass content.
    // We pick mix-wide automatically when no bass samples were detected at
    // encode time (PHATBASS_MIX_MODE = 1).
    const float PHAT_DELAY = 0.001814;  // 80 samples @ 44100Hz
    const float PHAT_DEPTH = 0.25;       // Reduced from 0.5 — was overpowering bass on dense MODs
    if (enableFAT) {{
        float tP = playbackTime - PHAT_DELAY;
        Position posP = getPosition(tP);
        float pbL = 0.0, pbR = 0.0;
#if PHATBASS_MIX_MODE
        // Mix-wide: take the same per-channel sum but with NO bass filter,
        // then let the listener treat the allpass as a global bass shaper.
        // The allpass only meaningfully shifts <100 Hz, so mids pass through
        // largely unaffected — net effect is sub-bass widening.
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
            float sp = getChannelOutput(ch, tP, posP, rowTime);
            int cm = ch % 4;
            pbL += sp * chR[cm];
            pbR += sp * chL[cm];
        }}
        // Quarter strength — applied to entire mix, must not overwhelm
        centL += pbL * normFactor * PHAT_DEPTH * 0.25;
        centR += pbR * normFactor * PHAT_DEPTH * 0.25;
#else
        // Per-sample: only bass-detected instruments
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
            Note n = getNote(pos.songPos, pos.row, ch);
            int inst = n.instrument;
            bool bass = (inst >= 1 && inst <= 31) ? isBass[inst - 1] : false;
            if (bass) {{
                float sp = getChannelOutput(ch, tP, posP, rowTime);
                int cm = ch % 4;
                pbL += sp * chR[cm];
                pbR += sp * chL[cm];
            }}
        }}
        centL += pbL * normFactor * PHAT_DEPTH;
        centR += pbR * normFactor * PHAT_DEPTH;
#endif
    }}
    
    float outL = surrL + centL;
    float outR = surrR + centR;

    // ── FAT4X harmonic exciter (stateless) ─────────────────────────────────
    // Ported from FAT4X: x*(1 + cs1(x)*FAT_AMOUNT).
    // cs1 produces even harmonics → adds warmth/presence, soft-limits peaks.
    // In original FAT4X this uses a 3-sample delay line + envelope follower;
    // stateless approximation replaces delay with current sample (trivially
    // same in our system since getPosition has ~20ms tick resolution).
    // FAT_AMOUNT: 0.0=off  0.5=half  1.0=full FAT4X-equivalent  >1.0=heavy
    const float FAT_AMOUNT = 1.0;  // matches FAT4X (FIR weights sum to 1.0)
    outL = outL * (1.0 + fat_cs1(outL) * FAT_AMOUNT);
    outR = outR * (1.0 + fat_cs1(outR) * FAT_AMOUNT);

    // ── Freeverb-inspired parallel comb reverb (stateless) ─────────────────
    // 6 parallel combs with Freeverb delays, each unrolled k steps:
    //   y(t) = Σ g^k · source(t - k·D)
    // Separate L/R panning per comb creates stereo width without doubling cost.
    //
    // N_ITER = unroll depth. Full 80 iterations as in the original would give
    // perfect RT60, but k=12 gives ~-14 dB at tail end (decent room feel).
    // Reduce N_ITER to 8 if audio drops out.
    //
    const float RV_WET  = 0.15;
    const int   N_ITER  = 5;   // iterations per comb
    const float RT60    = 2.4;
    const float _decay  = 8.9078 / RT60;  // ln(1000)/RT60

    // Freeverb-inspired comb delays (seconds), mutually prime in samples
    const float _D[6] = float[](0.0253, 0.0269, 0.0290, 0.0307, 0.0322, 0.0338);
    // Per-comb L/R pan — alternating for stereo spread, no extra source calls
    const float _pL[6] = float[](0.85, 0.40, 0.90, 0.35, 0.80, 0.45);
    const float _pR[6] = float[](0.40, 0.85, 0.35, 0.90, 0.45, 0.80);

    vec2 _wet = vec2(0.0);
    for (int _c = 0; _c < 6; _c++) {{
        float _d  = _D[_c];
        float _g  = exp(-_decay * _d);
        float _gk = 1.0, _tk = 0.0;
        for (int _k = 0; _k < N_ITER; _k++) {{
            _gk *= _g;
            _tk += _d;
            float _tw = playbackTime - _tk;
            Position _rp = getPosition(_tw);
            float _m = 0.0;
            for (int ch = 0; ch < NUM_CHANNELS; ch++)
                _m += getChannelOutput(ch, _tw, _rp, rowTime);
            _m *= normFactor * _gk;
            _wet.x += _pL[_c] * _m;
            _wet.y += _pR[_c] * _m;
        }}
    }}
    _wet /= 6.0;
    outL += _wet.x * RV_WET;
    outR += _wet.y * RV_WET;

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

    # Image-tab visualizer dispatch — sets up `vec3 col` for the image pipeline.
    if viz == 0:
        viz_setup_block = (
            "    vec2 _uv=(C*2.-iResolution.xy)/iResolution.y;\n"
            "    vec3 col = vec3(0.0);  // --viz 0: no visualizer"
        )
    elif viz == 3:
        viz_setup_block = (
            "    vec2 _uv=(C*2.-iResolution.xy)/iResolution.y;\n"
            "    float _tps_v=float(BPM)*2./5., _rt_v=float(SPEED)/_tps_v;\n"
            "    Position _pos_v=getPosition(iTime);\n"
            "    float _va0=abs(getChannelOutput(0,iTime,_pos_v,_rt_v));\n"
            "    float _va1=abs(getChannelOutput(1,iTime,_pos_v,_rt_v));\n"
            "    float _va2=abs(getChannelOutput(2,iTime,_pos_v,_rt_v));\n"
            "    float _va3=abs(getChannelOutput(3,iTime,_pos_v,_rt_v));\n"
            "    vec3 col = _visCurtain(vec2(_uv.y, abs(_uv.x)), _va0, _va1, _va2, _va3) + _visBG(C);"
        )
    else:  # 1, 2, 4, 5 all use _VizScene
        viz_setup_block = (
            "    vec2 _uv=(C*2.-iResolution.xy)/iResolution.y;\n"
            "    vec3 col = _VizScene(C);"
        )

    # ── Viz scene functions: only the chosen viz's GLSL is emitted ──────────
    if viz == 0:
        # No visualizer — empty scene block.  vec3 col = vec3(0.0) is set in setup.
        viz_scene_block = "\n// === VIZ 0: None — black backdrop ===\n"
    elif viz == 1:
        # Reactive 001 (PAEz fork) — SDF circles + cosmic web folding
        viz_scene_block = r"""
// === VIZ 1: Reactive 001 (PAEz fork of nayk's mosaic fractal circles) ===
// https://shadertoy.com/view/NfSXDc
mat2 _viz_rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }
#define _viz_RT(X) mat2(cos(X), sin(X), -sin(X), cos(X))
float _viz_sdf(vec2 uv, float t) {
    float sdf1 = 0.0;
    for (int i = 0; i < 3; i++) {
        uv = abs(uv) - vec2(0.1, 0.1);
        uv *= _viz_RT(float(i) * 0.3);
        sdf1 += max(abs(uv.x) - 0.3, abs(uv.y));
    }
    for (int i = 0; i < 3; i++) {
        uv = abs(uv) - vec2(0.2, 0.02);
        uv *= _viz_RT(float(i) * 2.0 + t * 0.1);
        sdf1 = max(sdf1, max(abs(uv.x) - 0.3, abs(uv.y)));
    }
    return sdf1;
}
float _viz_box(vec3 sp, vec3 d) { sp = abs(sp) - d; return max(max(sp.x, sp.y), sp.z); }
void _viz_inner(out vec4 out_color, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    uv *= 0.5;
    float t = mod(iTime, acos(-1.0) * 30.0);
    vec2 uv2 = uv;
    vec3 clr = vec3(0);
    float sdf1 = _viz_sdf(uv, t);
    sdf1 = max(sdf1, _viz_sdf(uv * 0.5, t + 5.0));
    uv *= _viz_RT(acos(-1.0) * 0.5 + t);
    sdf1 = max(sdf1, _viz_sdf(uv * 0.25, t + 10.0));
    sdf1 = max(sdf1, -_viz_sdf(uv * 0.1, t * 0.8 + 100.0));
    clr += pow((0.018 / sdf1) * 8.0 - 0.2, 12.0);
    vec3 ro = vec3(0, 1.0, -55);
    vec3  w = normalize(vec3(0) - ro);
    vec3  u = normalize(cross(w, vec3(0, 1, 0)));
    vec3  v = normalize(cross(u, w));
    vec3 rd = normalize(mat3(u, v, w) * vec3(uv2, 0.5));
    vec3 sp = ro;
    float d0 = 0.0;
    float it = 0.0;
    for (it; it < 100.0; it++) {
        sp.xz *= _viz_RT(t * 0.5);
        sp.zy *= _viz_RT(t * 0.25);
        sp.zy *= _viz_RT(sp.x * 0.05);
        for (int i = 0; i < 3; i++) {
            sp = abs(sp) - vec3(6, 2, 2);
            sp.xy *= _viz_RT(t * 0.25);
            sp.xz *= _viz_RT(0.5);
            sp.zy *= _viz_RT(5.0);
        }
        float ds = _viz_box(sp, vec3(1.0, 4.2, 1.0));
        if (abs(ds) < 0.001 || d0 > 90.0) break;
        d0 += ds;
        sp = ro + rd * d0;
    }
    clr += pow(it / 100.0, 4.0);
    out_color = vec4(pow(clr, vec3(0.4545)), 1.0);
}
vec3 _VizScene(vec2 F) {
    float T = iTime * 0.25;
    float a = 0.0;
    float r = 0.0;
    float t = iTime * 0.30;
    vec4 oo;
    _viz_inner(oo, F);
    vec4 O = vec4(0);
    for (int iter = 0; iter < 70; iter++) {
        vec3 i = iResolution;
        vec3 p = r * normalize(vec3((F + F - i.xy) / i.y, 0.5));
        p.z -= 7.5;
        p.xz *= _viz_rot(t);
        p.yz *= _viz_rot(t * 0.5);
        for (int j = 0; j < 7; j++) {
            p = abs(p) - 0.7;
            p.xy *= _viz_rot(0.785 + T * 0.3);
            p.xz *= _viz_rot(3.5 + T * 0.25);
        }
        p.xy *= _viz_rot(round(atan(p.x, p.y) * 8.0) / 3.0);
        float dist = length(p);
        // iChannel1 = Buffer A; row 0 has FFT_N audio samples in pixels 0..255.
        // Use texelFetch into the row-0 audio strip.
        int aBin = int(mod(1.1 - dist, 1.0) * float(FFT_N));
        aBin = clamp(aBin, 0, FFT_N - 1);
        float audio = texelFetch(iChannel1, ivec2(aBin, 0), 0).r;
        float glow = 0.3 + audio;
        glow = 0.00 + pow(glow, 4.0) * 0.5;
        vec4 pal = 0.5 + 0.5 * cos(a * 0.05 + t + vec4(4.0, 2.0, 1.0, 0.0));
        O += (0.35 * glow) * smoothstep(0.0, 1.0, dist) * pal / (length(p.xy) + 0.01);
        p += normalize(p) * sin(length(p) * 20.0 - T * 3.0) * 0.01;
        float d1 = length(p.xy) - 0.1;
        float d2 = length(p.yz) - 0.1;
        r += min(d1, d2) * 0.5 + 0.06;
        a += 1.0;
    }
    return oo.rgb * oo.rgb + O.rgb;
}
"""
    elif viz == 2:
        # Fluxline Surfer (mrange) — DR2 dodecahedron + glowtracer
        viz_scene_block = r"""
// === VIZ 2: Fluxline Surfer (CC0, mrange) ===
// https://twigl.app?ol=true&ss=-Or86HW1JKxI_HohuZew
#define _viz2_rot(x) mat2(cos(x+vec4(0,11,33,0)))
vec2 _viz2_csD, _viz2_csD2;
float _viz2_sc=1., _viz2_shell;
const float _viz2_pi = 3.14159;
float _viz2_totdist = 0.;
vec2 _viz2_Rot2D(vec2 q, float a) {
    vec2 cs = sin(a + vec2(0.5 * _viz2_pi, 0.));
    return vec2(dot(q, vec2(cs.x, -cs.y)), dot(q.yx, cs));
}
vec2 _viz2_Rot2Cs(vec2 q, vec2 cs) {
    return vec2(dot(q, vec2(cs.x, -cs.y)), dot(q.yx, cs));
}
vec3 _viz2_DodecSym(vec3 p) {
    float a, w = 2. * _viz2_pi / 5.;
    p.xz = _viz2_Rot2Cs(vec2(p.x, abs(p.z)), vec2(_viz2_csD.x, -_viz2_csD.y));
    p.xy = _viz2_Rot2D(p.xy, -0.25 * w);
    p.x = -abs(p.x);
    for (int k = 0; k < 3; k++) {
        if (dot(p.yz, _viz2_csD) > 0.) p.zy = _viz2_Rot2Cs(p.zy, _viz2_csD2) * vec2(1., -1.);
        p.xy = _viz2_Rot2D(p.xy, -w);
    }
    if (dot(p.yz, _viz2_csD) > 0.) p.zy = _viz2_Rot2Cs(p.zy, _viz2_csD2) * vec2(1., -1.);
    a = mod(atan(p.x, p.y) + 0.5 * w, w) - 0.5 * w;
    p.yx = vec2(cos(a), sin(a)) * length(p.xy);
    p.xz = -vec2(abs(p.x), p.z);
    return p;
}
void _viz2_inner(inout vec4 O, vec2 U) {
    float dihedDodec = 0.5 * atan(2.);
    _viz2_csD  = vec2(cos(dihedDodec), -sin(dihedDodec));
    _viz2_csD2 = vec2(cos(2. * dihedDodec), -sin(2. * dihedDodec));
    O = vec4(0);
    vec3 c = vec3(0);
    float sc, dotp, tt = iTime;
    _viz2_totdist = 0.;
    U = (U + U - iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0, 0, -.65);
    vec3 rd = normalize(vec3(U, .7));
    for (float i = 0.; i < 30.; i++) {
        vec3 p = vec3(rd * _viz2_totdist) * 7.;
        _viz2_shell = length(p) - .15;
        p.xyz += ro;
        sc = 1.;
        p.xz *= _viz2_rot(1.015 + tt / 3.);
        p.yz *= _viz2_rot(1.57  + tt / 5.);
        vec3 b = vec3(max(1.1, 1.14 - .01 * mod(tt, 15.)), .2, .2);
#define _viz2_ref(q) (q.x>q.y) ? q.xy : q.yx
        for (float n = 0.; n < 7.; n++) {
            p = _viz2_DodecSym(p * .98);
            p.xy = _viz2_ref(p.xy);
            p.xz = _viz2_ref(p.xz);
            p.yz = _viz2_ref(p.yz);
            p.xyz = 2. * p.xyz - b;
            float dotp2 = .97 / clamp(pow(length(p), 1.2), .8, 1.97);
            sc *= dotp2;
            p  *= dotp2;
        }
        float dist = max((length(p.xz) - 1.) / 50. / sc, -_viz2_shell);
        float stepsize = dist / 16.;
        _viz2_totdist += stepsize;
        if (dist < .0001) break;
        if (i > 1.)
            c += .05 * cos(log(sc) * 5. - 2. + 2. * vec3(1, 2, 3))
                     * exp(-(_viz2_totdist * _viz2_totdist) * 6e1);
    }
    c = 1. - exp(-c * c);
    O = vec4(c, 0);
}
vec3 _VizScene(vec2 C) {
    vec4 oo;
    _viz2_inner(oo, C);
    float i, d, z = 0.;
    vec4 o = vec4(0), O = vec4(0), p, r = iResolution.yxxx;
    mat2 R;
    for (i = 0.; ++i < 77.;) {
        p = vec4(z * normalize(vec3(C - .5 * r.yx, r.x)), .3);
        p.z += iTime;
        p.yw *= R = mat2(cos(cos(.4 * p.z) + vec4(0, 11, 33, 0)));
        p.wx *= R;
        p.xy *= R;
        o = 1. + sin(.3 * p.z + 4. * p.x + vec4(6, 3, 1, p.z));
        p.xy = abs(p.xy) - .5;
        d = .6 * abs(length(p.xy) + 5e-3 * (sin(z + 17. * sqrt(abs(dot(p.yzwx, p)))) - 89.)) + 2e-3;
        z += d;
        O += o.w / d * o;
    }
    vec4 fin = mix(oo * O * O * 1e-8, oo, 1.1 - _viz2_totdist);
    return fin.rgb;
}
"""
    elif viz == 3:
        # Zuvuya — city/stars backdrop + audio-reactive curtain
        viz_scene_block = r"""
// === VIZ 3: Zuvuya Visualizer (c) 2026 Orblivius ===
float _vrand(vec2 s){return fract(sin(dot(s,vec2(12.9898,78.233)))*43758.5453);}
float _vpow(float s,float x){return s-(s-s*s)*(-x);}
float _vstar(vec2 uv,float fl){float d=length(uv),m=.05/d;m+=m*fl;m*=smoothstep(.85,.2,d);return m;}
float _vsl(vec2 uv){vec2 gv=fract(uv)-.5,id=floor(uv);float c=0.;
    for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){vec2 o=vec2(x,y);float n=_vrand(id+o),sz=fract(n*345.32);
    c+=_vstar(gv-o-vec2(n,fract(n*34.))+.5,smoothstep(.9,1.,sz)*6.)*(sin(iTime*2.+n*13.256)*.5+.5)*sz;}return c;}
float _vboxinf(vec2 p){vec2 q=abs(p);return max(q.x+.5,q.y)-1.;}
float _vbox(vec3 p,vec3 b){vec3 q=abs(p)-b;return length(max(q,0.))+min(max(q.y,max(q.x,q.z)),0.);}
float _vmap(vec3 p){vec3 q=p;q.xz=mod(q.xz,1.)-.5;float h=abs(_vrand(floor(p.xz)+_vrand(floor(p.xz))));
    float id=floor(_vpow(abs(p.x*.1),-1.));return min(_vboxinf(p.xy),_vbox(q,vec3(.15,1.7*h+id,.15)));}
vec3 _vhsv(vec3 c){vec4 K=vec4(1.,2./3.,1./3.,3.);vec3 p=abs(fract(c.xxx+K.xyz)*6.-K.www);
    return c.z*mix(K.xxx,clamp(p-K.xxx,0.,1.),c.y);}
vec3 _visBG(vec2 fc){
    vec2 uv=(fc*2.-iResolution.xy)/min(iResolution.x,iResolution.y);
    vec3 ray=vec3(0.,1.5,1.-iTime),dir=normalize(cross(vec3(0.,0.,-1.),vec3(0.,1.,0.))*uv.x+
             vec3(0.,1.,0.)*uv.y+vec3(0.,0.,-1.)*1.8);
    int march=0;float rLen=0.,tot=0.;
    for(int i=0;i<38;++i){float d=_vmap(ray);march=i;tot+=d;if(d<.001||tot>60.)break;
        rLen+=min(min(min((step(0.,dir.x)-fract(ray.x))/dir.x,(step(0.,dir.y)-fract(ray.y))/dir.y)+.01,
                      (step(0.,dir.z)-fract(ray.z))/dir.z)+.01,d);ray=vec3(0.,1.5,1.-iTime)+dir*rLen;}
    float fog=float(march)/108.;vec3 fog2=vec3(tot*.01);
    vec3 city=vec3(.05,.5,2.)*fog+fog2*vec3(0.,.5,.1),stars=vec3(0.);
    for(float i=0.;i<=1.;i+=.25){float depth=fract(i+iTime*.2),sc=mix(20.,.5,depth);
        float fade=depth*smoothstep(1.,.9,depth);
        stars+=vec3(_vsl(uv*sc+i*432.)*fade);}
    stars*=abs(uv.y*.5)*vec3(.5,.5,1.);
    return mix(city,stars,clamp(fog2*1.5,0.,1.))*fog;}
vec3 _visCurtain(vec2 s,float a0,float a1,float a2,float a3){
    float chA[4];chA[0]=a0;chA[1]=a1;chA[2]=a2;chA[3]=a3;
    float per=2./max(abs(s.y),.12);vec3 col=vec3(0.);
    for(float z=0.;z<1.;z+=.1){
        int ch=int(z*4.)%NUM_CHANNELS;
        float wm=chA[ch]*abs(sin(z*9.42+iTime*2.1+float(ch)*1.57));
        vec2 p=vec2(s.x*(1.+z),s.y+(1.+z))*per;p.y+=2.8*iTime;
        p.x+=cos(z/.06)+wm*sin(p.y*3.14159*3.)*z*2.;p.y+=wm*2.*z;
        float w=p.x,l=sin(p.y*.5+z/.08+3.4*iTime);
        float heat=clamp(wm*2.,0.,1.);
        float intensity=exp(min(l,-l/mix(.3,.05,heat)/(1.+4.*w*w)));
        vec3 tint=_vhsv(vec3(float(ch)*.25+.05,mix(.6,1.,heat),mix(.5,1.4,heat)));
        tint+=vec3(.15,0.,.25)*smoothstep(.3,.7,sin(z*30.+iTime*.7))*(1.-heat);
        col+=intensity*tint/(abs(w)+.01*per)*per;}
    return tanh(col*col/2e3);}
"""
    elif viz == 4:
        # Maya — raymarched fractal tunnel-warp
        viz_scene_block = r"""
// === VIZ 4: Maya — raymarched fractal tunnel-warp ===
// 60-iteration raymarcher with mirror/voxel/sine warp.
vec3 _VizScene(vec2 C) {
    vec4 O = vec4(0.0);
    vec3 p, r = normalize(vec3(C+C, 0.0) - iResolution.xyy);
    float i = 0.0, t = 0.0, v = 0.0, e = iTime * 0.7, z = 0.0;
    for ( ; i++ < 60.0; t += 0.2 * v) {
        p = t * normalize(vec3(r.xy, r.z / exp(t * 0.1)));
        p = abs(p);
        p.xy *= mat2(cos( floor((e - t * 0.5) / 3.0)
                        + min(3.0 * fract((e - t * 0.5) / 3.0), 1.0)
                        + vec4(0.0, 11.0, 33.0, 0.0) ));
        z = p.z -= e;
        p = ceil(p * 20.0) / 20.0;
        p += sin(p.zxy);
        p = abs(mod(p, 4.0) - 2.0);
        v = abs(min(min(max(p.x, p.y), max(p.x, p.z)), max(p.y, p.z)) - 0.2) + 0.01;
        O.rgb += exp(cos(i * 0.2 + vec3(0.0, 2.0, 4.0)))
               / v
               / (abs(sin(z - e * 2.0 - vec3(0.0, 0.2, 0.4))) + 0.1);
    }
    O = tanh(O / 3e3);
    return O.rgb * O.rgb;
}
"""
    else:  # viz == 5
        # Dodecahedron (Philip Bertani) — DR2 dodecahedral symmetry + IFS fractal raymarcher
        viz_scene_block = r"""
// === VIZ 5: Dodecahedron Fractal Visualizer (Philip Bertani / Orblivius adapt) ===
// Original: Philip Bertani — DR2 dodecahedral symmetry + iterative IFS fractal
// Single-sample render (no 5-tap smoothing).
#define _v5_drot(x) mat2(cos((x) + vec4(0.,11.,33.,0.)))
vec2 _v5_csD, _v5_csD2;
vec2 _v5_Rot2D(vec2 q, float a) {
    vec2 cs = sin(a + vec2(0.5*3.14159, 0.));
    return vec2(dot(q, vec2(cs.x, -cs.y)), dot(q.yx, cs));
}
vec2 _v5_Rot2Cs(vec2 q, vec2 cs) {
    return vec2(dot(q, vec2(cs.x, -cs.y)), dot(q.yx, cs));
}
vec3 _v5_DodecSym(vec3 p) {
    float a, w = 2.*3.14159/5.;
    p.xz = _v5_Rot2Cs(vec2(p.x, abs(p.z)), vec2(_v5_csD.x, -_v5_csD.y));
    p.xy = _v5_Rot2D(p.xy, -0.25*w);
    p.x = -abs(p.x);
    for (int k = 0; k < 3; k++) {
        if (dot(p.yz, _v5_csD) > 0.) p.zy = _v5_Rot2Cs(p.zy, _v5_csD2) * vec2(1., -1.);
        p.xy = _v5_Rot2D(p.xy, -w);
    }
    if (dot(p.yz, _v5_csD) > 0.) p.zy = _v5_Rot2Cs(p.zy, _v5_csD2) * vec2(1., -1.);
    a = mod(atan(p.x, p.y) + 0.5*w, w) - 0.5*w;
    p.yx = vec2(cos(a), sin(a)) * length(p.xy);
    p.xz = -vec2(abs(p.x), p.z);
    return p;
}
// Single-sample dodec render at pixel position U (fragCoord, full resolution)
vec3 _v5_DodecOne(vec2 U) {
    float dihedDodec = 0.5 * atan(2.);
    _v5_csD  = vec2(cos(dihedDodec),     -sin(dihedDodec));
    _v5_csD2 = vec2(cos(2.*dihedDodec), -sin(2.*dihedDodec));
    vec3 c = vec3(0.);
    float sc, totdist = 0., tt = iTime*2., shell;
    U = (U+U - iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0., 0., max(-1.5, -0.9*mod(iTime, 20.)));
    vec3 rd = normalize(vec3(U, 0.7));
    for (float i = 0.; i < 50.; i++) {
        vec3 p = vec3(rd*totdist) * 10.;
        shell = length(p) - 0.1;
        p.xyz += ro;
        sc = 1.;
        p.xz *= _v5_drot(1.015 + tt/3.);
        p.yz *= _v5_drot(1.57  + tt/5.);
        p = _v5_DodecSym(p);
        vec3 b = vec3(2.35, 0.9, 0.5);
        for (float n = 0.; n < 7.; n++) {
            p = abs(p);
            p.xz *= _v5_drot(0.3);
            p.xy = (p.x > p.y) ? p.xy : p.yx;
            p.xz = (p.x > p.z) ? p.xz : p.zx;
            p.yz = (p.y > p.z) ? p.yz : p.zy;
            p.xyz = 2. * p.xyz - b;
            float dotp = 1.9 / clamp(pow(length(p.xy), 3.), 0.1, 2.2);
            sc *= dotp;
            p  *= dotp;
        }
        float dist = max((length(p.yz) - 0.75) / 50. / sc, -shell);
        totdist += dist / 30.;
        if (dist < 0.0001) break;
        if (i > 5.) c += 0.05 * cos(log(sc)*1.2 + 4.*vec3(1, 2, 3))
                          * exp(-(totdist*totdist)*1e1);
    }
    return 1. - exp(-c*c);
}
// Single-sample (no 5-tap smoothing per --viz 5 spec)
vec3 _VizScene(vec2 C) {
    return _v5_DodecOne(C);
}
"""

    image_glsl = f"""/* ============================================================================
   GLSL MOD Player v1.37 (c)2026 Orblivius
   3D Surround, PhatBass, Comb Reverb, FAT, RVQ sample compression, configurable resampler
   IMAGE TAB — iChannel0: alphabet texture (shadertoy.com/view/4sf3RB)
   Visualizer: {viz_name}
   Contact: subband@gmail.com or
            subband@protonmail.com
   GIT:     https://github.com/mewza/mod2glsl
  ============================================================================ */

// FFT_N must match Buffer A — used by spectrum view to index Buffer A row 1.
#define FFT_N 256


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
makeStr(printHdr)   _NUM _NUM _NUM _ _G _L _S _L _ _M _O _D _ _P _L _A _Y _E _R _ _V _1 _DOT _3 _7 _ _NUM _NUM _NUM _end
makeStr(printCredit) _COPY _2 _0 _2 _6 _ _O _R _B _L _I _V _I _U _S _end
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

{viz_scene_block}


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

    // ── Per-channel amplitudes computed inside viz_setup_block when needed
    float _tps=float(BPM)*2./5., _rt=float(SPEED)/_tps;
    Position _pos=getPosition(iTime);
{viz_setup_block}

    if (iFrame < LOADING_FRAMES) {{
        vec2 res = iResolution.xy;
        float prog = float(iFrame) / float(LOADING_FRAMES - 1);

        // (data arrays paged in by compiler — no runtime loop needed)

        // Header
        col += CYAN   * printHdr   (pUV(fp, ML, 6., CH));
        col += WHITE  * printCredit(pUV(fp, res.x - ML - 234., 6., CH*0.75));
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
    trk += WHITE  * printCredit(pUV(fp, iResolution.x - ML - 234., 6., CH*0.75));
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

    // ============ OSCILLOSCOPE / SPECTRUM (mouse click toggles) ============
    // iChannel1 = Buffer A → row 0 = audio, row 1 = FFT mags, row 2 = toggle state
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
#ifdef VQ_IN_COMMON
            for (int ch = 0; ch < NUM_CHANNELS; ch++)
                mono += getChannelOutput(ch, oscT, oscPos, rowTimeO);
            mono /= float(NUM_CHANNELS);
#else
            // --split: synthesize from notes (no getChannelOutput here)
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
                Note tn = getNote(oscPos.songPos, oscPos.row, ch);
                int trow = oscPos.row, tpat = oscPos.songPos;
                if (tn.instrument <= 0 || tn.period <= 0) {{
                    int sr = oscPos.row, sp2 = oscPos.songPos;
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
                float f = periodToFreq(tn.period);
                SampleInfo si = samples[tn.instrument - 1];
                int vol = si.volume;
                Note cr = getNote(oscPos.songPos, oscPos.row, ch);
                if (cr.effect == 0xC) vol = min(cr.param, 64);
                else if (tn.effect == 0xC) vol = min(tn.param, 64);
                float amp = float(vol) / 64.0;
                int trigSgr = patTickOffset[tpat] + (trow - patStartRow[tpat]);
                float trigT = float(fetchTick(trigSgr)) / TICKS_PER_SEC;
                float age   = max(0.0, oscT - trigT);
                float env   = exp(-age * 1.5);
                mono += amp * env * sin(6.2831853 * f * oscT) / float(NUM_CHANNELS);
            }}
#endif

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
            // ── FFT Spectrum from Buffer A (iChannel1 row 1) ─────────────
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
            float logMin = log2(27.5), logMax = log2(4186.0);
            float logFreq = logMin + (C.x / iResolution.x) * (logMax - logMin);
            float pixFreq = pow(2.0, logFreq);
            const vec3 TCols[4] = vec3[](TC0, TC1, TC2, TC3);
            float ticksPerSecV = BPM * 2.0 / 5.0;
            float rowTimeV = SPEED / ticksPerSecV;
            float totalBar = 0.0;
            vec3  barColor = BG;
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
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
                SampleInfo si = samples[tn.instrument - 1];
                int vol = si.volume;
                Note cr = getNote(pos.songPos, pos.row, ch);
                if (cr.effect == 0xC) vol = min(cr.param, 64);
                else if (tn.effect == 0xC) vol = min(tn.param, 64);
                float amp = float(vol) / 64.0;
                float semitone = 0.5 / 12.0;
                float dist = abs(log2(pixFreq) - log2(noteFreq));
                if (dist < semitone * 1.5) {{
                    float edge = 1.0 - dist / (semitone * 1.5);
                    totalBar = max(totalBar, amp * edge);
                    barColor = TCols[ch];
                }}
            }}
            float barH = totalBar * oh * 0.92;
            float barY = oh - barH;
            if (sy > barY && barH > 2.0) {{
                float t = (sy - barY) / max(barH, 1.0);
                col = mix(WHITE, barColor, t * 0.8);
            }} else {{
                col = mix(col, BG, 0.3);
            }}
            float semPos = logFreq * 12.0;
            float isBlack = step(0.5, fract(semPos));
            col = mix(col, col * 0.85, isBlack * 0.15);
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
   GLSL MOD Player v1.37 (c)2026 Orblivius
   3D Surround, PhatBass, Comb Reverb, FAT, RVQ sample compression, configurable resampler
   Contact: subband@gmail.com or
            subband@protonmail.com
   GIT:     https://github.com/mewza/mod2glsl
   BUFFER A TAB
   Visualizer: {viz_name}
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
        // VQ_IN_COMMON path: real getChannelOutput audio (default --no-split).
        // Otherwise: synthesize from note pattern data (when VQ moved to Sound).
        float dt  = 1.0 / FFT_SR;
        float t   = iTime - float(FFT_N - px - 1) * dt;
        float ticksPerSec = float(BPM) * 2.0 / 5.0;
        float rowTime = float(SPEED) / ticksPerSec;
        Position pos = getPosition(t);
        float s = 0.0;
#ifdef VQ_IN_COMMON
        for (int ch = 0; ch < NUM_CHANNELS; ch++)
            s += getChannelOutput(ch, t, pos, rowTime);
        s /= float(NUM_CHANNELS);
#else
        // --split: synthesize from notes (no VQ access in this tab)
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
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
            float f = periodToFreq(tn.period);
            SampleInfo si = samples[tn.instrument - 1];
            int vol = si.volume;
            Note cr = getNote(pos.songPos, pos.row, ch);
            if (cr.effect == 0xC) vol = min(cr.param, 64);
            else if (tn.effect == 0xC) vol = min(tn.param, 64);
            float amp = float(vol) / 64.0;
            int trigSgr = patTickOffset[tpat] + (trow - patStartRow[tpat]);
            float trigT = float(fetchTick(trigSgr)) / TICKS_PER_SEC;
            float age   = max(0.0, t - trigT);
            float env   = exp(-age * 1.5);
            s += amp * env * sin(6.2831853 * f * t);
        }}
        s /= float(NUM_CHANNELS);
#endif
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
            float s = 0.0;
#ifdef VQ_IN_COMMON
            s = getChannelOutput(px, iTime, pos, rowTime);
#else
            // --split: note-synth fallback for single channel `px`
            int ch = px;
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
            if (tn.period > 0) {{
                float f = periodToFreq(tn.period);
                SampleInfo si = samples[tn.instrument - 1];
                int vol = si.volume;
                Note cr = getNote(pos.songPos, pos.row, ch);
                if (cr.effect == 0xC) vol = min(cr.param, 64);
                else if (tn.effect == 0xC) vol = min(tn.param, 64);
                float amp = float(vol) / 64.0;
                int trigSgr = patTickOffset[tpat] + (trow - patStartRow[tpat]);
                float trigT = float(fetchTick(trigSgr)) / TICKS_PER_SEC;
                float age   = max(0.0, iTime - trigT);
                float env   = exp(-age * 1.5);
                s = amp * env * sin(6.2831853 * f * iTime);
            }}
#endif
            O = vec4(0., 0., 0., abs(s) * 0.5 + 0.5);
        }} else {{
            O = texelFetch(iChannel0, ivec2(px, py - 1), 0);
        }}

    }} else if (py == WAVE_BASE) {{
        // ── Row 70: newest row — RAW BIPOLAR audio, no abs, no envelope ──────
        float u  = float(px) / float(iResolution.x);
        float mu = 1.0 - abs(u * 2.0 - 1.0);
        float t  = iTime - mu * 0.5;
        float ticksPerSec = float(BPM) * 2.0 / 5.0;
        float rowTime = float(SPEED) / ticksPerSec;
        Position pos = getPosition(t);
        float s = 0.0;
#ifdef VQ_IN_COMMON
        for (int ch = 0; ch < NUM_CHANNELS; ch++)
            s += getChannelOutput(ch, t, pos, rowTime);
        s /= float(NUM_CHANNELS);
#else
        // --split: note-synth fallback (no VQ access in this tab)
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
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
            float f = periodToFreq(tn.period);
            SampleInfo si = samples[tn.instrument - 1];
            int vol = si.volume;
            Note cr = getNote(pos.songPos, pos.row, ch);
            if (cr.effect == 0xC) vol = min(cr.param, 64);
            else if (tn.effect == 0xC) vol = min(tn.param, 64);
            float amp = float(vol) / 64.0;
            int trigSgr = patTickOffset[tpat] + (trow - patStartRow[tpat]);
            float trigT = float(fetchTick(trigSgr)) / TICKS_PER_SEC;
            float age   = max(0.0, t - trigT);
            float env   = exp(-age * 1.5);
            s += amp * env * sin(6.2831853 * f * t);
        }}
        s /= float(NUM_CHANNELS);
#endif
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
    "ICAgICAgICBzZWxmLm1hZ2ljID0gZFsxMDgwOjEwODRdCiAgICAgICAgIyBEZXRlY3QgY2hhbm5lbCBj"
    "b3VudCBmcm9tIHNpZ25hdHVyZQogICAgICAgIHNpZyA9IHNlbGYubWFnaWMKICAgICAgICBpZiBzaWcg"
    "aW4gKGInTS5LLicsIGInTSFLIScsIGInTSZLIScsIGInTi5ULicsIGInRkxUNCcsIGInNENITicpOgog"
    "ICAgICAgICAgICBzZWxmLm51bV9jaGFubmVscyA9IDQKICAgICAgICBlbGlmIHNpZyA9PSBiJ0ZMVDgn"
    "IG9yIHNpZyBpbiAoYidPQ1RBJywgYidDRDgxJywgYidPS1RBJyk6CiAgICAgICAgICAgIHNlbGYubnVt"
    "X2NoYW5uZWxzID0gOAogICAgICAgIGVsaWYgbGVuKHNpZykgPT0gNCBhbmQgc2lnWzE6NF0gPT0gYidD"
    "SE4nIGFuZCBzaWdbMDoxXS5pc2RpZ2l0KCk6CiAgICAgICAgICAgIHNlbGYubnVtX2NoYW5uZWxzID0g"
    "aW50KHNpZ1swOjFdKQogICAgICAgIGVsaWYgbGVuKHNpZykgPT0gNCBhbmQgc2lnWzI6NF0gPT0gYidD"
    "SCcgYW5kIHNpZ1swOjFdLmlzZGlnaXQoKSBhbmQgc2lnWzE6Ml0uaXNkaWdpdCgpOgogICAgICAgICAg"
    "ICBzZWxmLm51bV9jaGFubmVscyA9IGludChzaWdbMDoyXSkKICAgICAgICBlbGlmIGxlbihzaWcpID09"
    "IDQgYW5kIHNpZ1s6M10gPT0gYidURFonIGFuZCBzaWdbMzo0XS5pc2RpZ2l0KCk6CiAgICAgICAgICAg"
    "IHNlbGYubnVtX2NoYW5uZWxzID0gaW50KHNpZ1szOjRdKQogICAgICAgIGVsc2U6CiAgICAgICAgICAg"
    "IHNlbGYubnVtX2NoYW5uZWxzID0gNAogICAgICAgIHNlbGYubnVtX3BhdHRlcm5zID0gbWF4KHNlbGYu"
    "cGF0dGVybl9vcmRlcls6c2VsZi5zb25nX2xlbmd0aF0pICsgMQogICAgICAgICMgRWFjaCBwYXR0ZXJu"
    "IHJvdyA9IG51bV9jaGFubmVscyDDlyA0IGJ5dGVzOyA2NCByb3dzL3BhdHRlcm4KICAgICAgICBwYXRf"
    "c2l6ZSA9IDY0ICogc2VsZi5udW1fY2hhbm5lbHMgKiA0CiAgICAgICAgc2VsZi5wYXR0ZXJucyA9IFtd"
    "CiAgICAgICAgb2ZmID0gMTA4NAogICAgICAgIGZvciBwIGluIHJhbmdlKHNlbGYubnVtX3BhdHRlcm5z"
    "KToKICAgICAgICAgICAgc2VsZi5wYXR0ZXJucy5hcHBlbmQoZFtvZmY6b2ZmK3BhdF9zaXplXSkKICAg"
    "ICAgICAgICAgb2ZmICs9IHBhdF9zaXplCiAgICAgICAgIyBTYW1wbGVzIChyYXcgc2lnbmVkIDgtYml0"
    "IGJ5dGVzKQogICAgICAgIHNlbGYuc2FtcGxlX2J5dGVzID0gW10KICAgICAgICBmb3IgcyBpbiBzZWxm"
    "LnNhbXBsZXNfaW5mbzoKICAgICAgICAgICAgc2VsZi5zYW1wbGVfYnl0ZXMuYXBwZW5kKGRbb2ZmOm9m"
    "ZitzWydsZW5ndGgnXV0pCiAgICAgICAgICAgIG9mZiArPSBzWydsZW5ndGgnXQoKIyDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZAKIyBQQVRURVJOIENSVU5DSDogYml0bWFwICsgZGljdCAr"
    "IG5pYmJsZS1zZWVrCiMg4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQCgpFTVBUWV9O"
    "T1RFID0gYidceDAwXHgwMFx4MDBceDAwJwoKZGVmIGVuY29kZV9wYXR0ZXJucyhtb2QpOgogICAgIiIi"
    "UmV0dXJucyBkaWN0IG9mIGFsbCBwYXR0ZXJuIGRhdGEgc3RydWN0dXJlcy4iIiIKICAgICMgQnVpbGQg"
    "ZmxhdCBsaXN0IG9mIDQtYnl0ZSBub3RlcyBpbiBvcmRlcjogcGF0IDAuLk4tMSwgcm93IDAuLjYzLCBj"
    "aCAwLi4zLgogICAgIyBGaXJzdCBhcHBseSBQcm9UcmFja2VyIHBhcmFtLW1lbW9yeSByZXdyaXRpbmcg"
    "aW4gc29uZy1wb3NpdGlvbiBwbGF5YmFjawogICAgIyBvcmRlcjogd2hlbiBhIG5vdGUgaGFzIGVmZmVj"
    "dCAxLzIvMy80LzUvNi9BIGFuZCBwYXJhbT09MCwgc3Vic3RpdHV0ZSB0aGUKICAgICMgbGFzdCBub24t"
    "emVybyBwYXJhbSBzZWVuIGZvciB0aGF0IGVmZmVjdCBvbiB0aGlzIGNoYW5uZWwuICBUaGlzIG1ha2Vz"
    "CiAgICAjIHRvbmUtcG9ydGEgcnVucyBsaWtlICIzMDAgMzAwIDMwMCIgY29udGludWUgd2l0aCB0aGUg"
    "cHJldmlvdXMgc2xpZGUgcmF0ZQogICAgIyDigJQgcmVxdWlyZWQgZm9yIG1hbnkgTU9EcyAoaW5jbC4g"
    "R1NMSU5HRVIgcGF0dGVybiAzKS4KICAgIE5DID0gbW9kLm51bV9jaGFubmVscwogICAgcm93X3N0cmlk"
    "ZSA9IE5DICogNAoKICAgICMgV2FsayBzb25nIHBvc2l0aW9ucyB0byBmaW5kIHBhcmFtLW1lbW9yeSBj"
    "aGFpbnMgcGVyIGNoYW5uZWwuCiAgICAjIEVmZmVjdCBncm91cHMgdGhhdCBzaGFyZSBtZW1vcnk6CiAg"
    "ICAjICAgMHgxIChwb3J0YSB1cCksIDB4MiAocG9ydGEgZG93biksIDB4MyAodG9uZSBwb3J0YSksIDB4"
    "NSAodG9uZSt2b2wpLAogICAgIyAgIDB4NCAodmlicmF0byksIDB4NiAodmliK3ZvbCksIDB4QSAodm9s"
    "IHNsaWRlKQogICAgIyBXZSByZXdyaXRlIHRoZSBpbi1tZW1vcnkgcGF0dGVybiBieXRlcyAoYSBjb3B5"
    "KSBzbyBlbmNvZGluZyBzZWVzIHRoZQogICAgIyBjb3JyZWN0ZWQgcGFyYW1zLiAgQnVpbGQgYSBmcmVz"
    "aCBwZXItcGF0dGVybiBub3RlIGxpc3Qgd2l0aCByZXdyaXRlcy4KICAgICMgVXNlIG1vZC5wYXR0ZXJu"
    "X29yZGVyIChlbmNvZGVyIE1PREZpbGUgZXF1aXZhbGVudCBvZiBzb25nX3Bvc2l0aW9ucykuCiAgICBf"
    "c29uZ19vcmRlciA9IGdldGF0dHIobW9kLCAncGF0dGVybl9vcmRlcicsIE5vbmUpIG9yIGdldGF0dHIo"
    "bW9kLCAnc29uZ19wb3NpdGlvbnMnLCBbXSkKICAgIHBhdF9jb3BpZXMgPSB7fQogICAgbGFzdF9wYXJh"
    "bSA9IFt7fSBmb3IgXyBpbiByYW5nZShOQyldICAjIGxhc3RfcGFyYW1bY2hdW2VmZmVjdF0gPSBsYXN0"
    "IG5vbnplcm8gcGFyYW0KICAgIHJld3JpdHRlbl9ub3Rlc19jb3VudCA9IDAKICAgIGZvciBzcCBpbiBf"
    "c29uZ19vcmRlcls6Z2V0YXR0cihtb2QsICdzb25nX2xlbmd0aCcsIGxlbihfc29uZ19vcmRlcikpXToK"
    "ICAgICAgICBpZiBzcCBub3QgaW4gcGF0X2NvcGllczoKICAgICAgICAgICAgcGF0X2NvcGllc1tzcF0g"
    "PSBieXRlYXJyYXkobW9kLnBhdHRlcm5zW3NwXSkKICAgICAgICAjIE5vdGU6IGEgcGF0dGVybiBtYXkg"
    "YXBwZWFyIGF0IG11bHRpcGxlIHNvbmcgcG9zaXRpb25zOyB3ZSBhcHBseQogICAgICAgICMgcmV3cml0"
    "ZXMgaW4gcGxheWJhY2sgb3JkZXIgc28gbWVtb3J5IHN0YXRlIHByb3BhZ2F0ZXMgYWNyb3NzIHRoZW0u"
    "CiAgICAgICAgIyBSZXdyaXRpbmcgYSBwYXR0ZXJuIHRoYXQgaXMgcmV1c2VkIGxhdGVyIG1lYW5zIHRo"
    "ZSBzZWNvbmQgdmlzaXQKICAgICAgICAjIHVzZXMgdGhlIGFscmVhZHktcmV3cml0dGVuIHBhcmFtcywg"
    "d2hpY2ggaXMgYWNjZXB0YWJsZSBzaW5jZSB0aGUKICAgICAgICAjIGxhc3RfcGFyYW0gc3RhdGUgYXQg"
    "c2Vjb25kIHZpc2l0IHdvdWxkIG5hdHVyYWxseSBhbHNvIGhhdmUgdGhvc2UuCiAgICAjIFJlc2V0IGZv"
    "ciBhY3R1YWwgcmV3cml0ZSB3YWxrCiAgICBsYXN0X3BhcmFtID0gW3t9IGZvciBfIGluIHJhbmdlKE5D"
    "KV0KICAgIHZpc2l0ZWRfa2V5cyA9IHNldCgpCiAgICBmb3Igc3AgaW4gX3Nvbmdfb3JkZXJbOmdldGF0"
    "dHIobW9kLCAnc29uZ19sZW5ndGgnLCBsZW4oX3Nvbmdfb3JkZXIpKV06CiAgICAgICAgcGF0X2NvcHkg"
    "PSBwYXRfY29waWVzW3NwXQogICAgICAgIGZvciByb3cgaW4gcmFuZ2UoNjQpOgogICAgICAgICAgICBm"
    "b3IgY2ggaW4gcmFuZ2UoTkMpOgogICAgICAgICAgICAgICAgYmFzZSA9IHJvdypyb3dfc3RyaWRlICsg"
    "Y2gqNAogICAgICAgICAgICAgICAga2V5ID0gKHNwLCByb3csIGNoKQogICAgICAgICAgICAgICAgaWYg"
    "a2V5IGluIHZpc2l0ZWRfa2V5czogY29udGludWUgICMgYXZvaWQgZG91YmxlLXJld3JpdGUgb24gcGF0"
    "dGVybiByZXVzZQogICAgICAgICAgICAgICAgdmlzaXRlZF9rZXlzLmFkZChrZXkpCiAgICAgICAgICAg"
    "ICAgICAjIERlY29kZSBub3RlOiBieXRlcyBbcGVyaW9kX2hpLCBwZXJpb2RfbG8sIHNhbXBsZV9sb3xl"
    "ZmZlY3QsIHBhcmFtXQogICAgICAgICAgICAgICAgIyBNT0QgbGF5b3V0OiBieXRlMCA9IHNhbXBsZV9o"
    "aSg0KSB8IHBlcmlvZF9oaSg0KQogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICBieXRlMSA9IHBl"
    "cmlvZF9sbyg4KQogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICBieXRlMiA9IHNhbXBsZV9sbyg0"
    "KSB8IGVmZmVjdCg0KQogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICBieXRlMyA9IHBhcmFtCiAg"
    "ICAgICAgICAgICAgICBiMCwgYjEsIGIyLCBiMyA9IHBhdF9jb3B5W2Jhc2VdLCBwYXRfY29weVtiYXNl"
    "KzFdLCBwYXRfY29weVtiYXNlKzJdLCBwYXRfY29weVtiYXNlKzNdCiAgICAgICAgICAgICAgICBlZmZl"
    "Y3QgPSBiMiAmIDB4MEYKICAgICAgICAgICAgICAgIHBhcmFtICA9IGIzCiAgICAgICAgICAgICAgICAj"
    "IFByb1RyYWNrZXIgcGFyYW0tbWVtb3J5IHJ1bGVzOgogICAgICAgICAgICAgICAgIwogICAgICAgICAg"
    "ICAgICAgIyAgIDF4eCAocG9ydGEgdXApICAgICAgICDigJQgcGFyYW09MCDihpIgdXNlIGxhc3QgMXh4"
    "CiAgICAgICAgICAgICAgICAjICAgMnh4IChwb3J0YSBkb3duKSAgICAgIOKAlCBwYXJhbT0wIOKGkiB1"
    "c2UgbGFzdCAyeHgKICAgICAgICAgICAgICAgICMgICAzeHggKHRvbmUgcG9ydGEpICAgICAg4oCUIHBh"
    "cmFtPTAg4oaSIHVzZSBsYXN0IDN4eAogICAgICAgICAgICAgICAgIwogICAgICAgICAgICAgICAgIyAg"
    "IDR4eCAodmlicmF0bykgICAgICAgICDigJQgTklCQkxFIG1lbW9yeTogaGlnaCBuaWI9MCBrZWVwcwog"
    "ICAgICAgICAgICAgICAgIyAgICAgICAgICAgICAgICAgICAgICAgICAgICBwcmlvciBzcGVlZCwgbG93"
    "IG5pYj0wIGtlZXBzCiAgICAgICAgICAgICAgICAjICAgICAgICAgICAgICAgICAgICAgICAgICAgIHBy"
    "aW9yIGRlcHRoLiAgV2UgZG9uJ3QgcmV3cml0ZQogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICAg"
    "ICAgICAgICAgICAgICBoZXJlIOKAlCBHTFNML0hUTUwgZG8gbmliYmxlLWxldmVsCiAgICAgICAgICAg"
    "ICAgICAjICAgICAgICAgICAgICAgICAgICAgICAgICAgIGhhbmRsaW5nLiAgT25seSByZXdyaXRlIHBh"
    "cmFtPTAKICAgICAgICAgICAgICAgICMgICAgICAgICAgICAgICAgICAgICAgICAgICAgKHdob2xlIGJ5"
    "dGUgemVybykg4oaSIHVzZSBsYXN0IDR4eC4KICAgICAgICAgICAgICAgICMgICA3eHggKHRyZW1vbG8p"
    "ICAgICAgICAg4oCUIE5JQkJMRSBtZW1vcnkgbGlrZSA0eHguCiAgICAgICAgICAgICAgICAjCiAgICAg"
    "ICAgICAgICAgICAjICAgNXh4IChjb250aW51ZSB0b25lIHBvcnRhICsgdm9sIHNsaWRlKSDigJQgcGFy"
    "YW0gYnl0ZSBpcwogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICAgICAgICAgICAgICAgICBWT0wt"
    "U0xJREUgT05MWS4gIDUwMCA9IGNvbnRpbnVlCiAgICAgICAgICAgICAgICAjICAgICAgICAgICAgICAg"
    "ICAgICAgICAgICAgIHNsaWRlIHdpdGggTk8gdm9sIGNoYW5nZTsgdmFsaWQKICAgICAgICAgICAgICAg"
    "ICMgICAgICAgICAgICAgICAgICAgICAgICAgICAgY29tbWFuZCwgZG8gTk9UIHJld3JpdGUuCiAgICAg"
    "ICAgICAgICAgICAjICAgNnh4IChjb250aW51ZSB2aWJyYXRvICsgdm9sIHNsaWRlKSDigJQgc2FtZSBh"
    "cyA1eHg7CiAgICAgICAgICAgICAgICAjICAgICAgICAgICAgICAgICAgICAgICAgICAgIHBhcmFtIGJ5"
    "dGUgaXMgdm9sLXNsaWRlIG9ubHkuCiAgICAgICAgICAgICAgICAjICAgICAgICAgICAgICAgICAgICAg"
    "ICAgICAgIDYwMCA9IGNvbnRpbnVlIHZpYnJhdG8sIG5vIHZvbAogICAgICAgICAgICAgICAgIyAgICAg"
    "ICAgICAgICAgICAgICAgICAgICAgICBzbGlkZTsgdmFsaWQgY29tbWFuZC4KICAgICAgICAgICAgICAg"
    "ICMgICBBeHggKHZvbCBzbGlkZSkgICAgICAg4oCUIEEwMCA9IG5vLW9wIGluIFBUIChOT1QgbWVtb3J5"
    "KS4KICAgICAgICAgICAgICAgICMgICAgICAgICAgICAgICAgICAgICAgICAgICAgRG8gTk9UIHJld3Jp"
    "dGUuCiAgICAgICAgICAgICAgICBpZiBlZmZlY3QgaW4gKDB4MSwgMHgyLCAweDMpOgogICAgICAgICAg"
    "ICAgICAgICAgIGlmIHBhcmFtID09IDAgYW5kIGVmZmVjdCBpbiBsYXN0X3BhcmFtW2NoXToKICAgICAg"
    "ICAgICAgICAgICAgICAgICAgbmV3X3BhcmFtID0gbGFzdF9wYXJhbVtjaF1bZWZmZWN0XQogICAgICAg"
    "ICAgICAgICAgICAgICAgICBwYXRfY29weVtiYXNlKzNdID0gbmV3X3BhcmFtCiAgICAgICAgICAgICAg"
    "ICAgICAgICAgIHJld3JpdHRlbl9ub3Rlc19jb3VudCArPSAxCiAgICAgICAgICAgICAgICAgICAgZWxp"
    "ZiBwYXJhbSAhPSAwOgogICAgICAgICAgICAgICAgICAgICAgICBsYXN0X3BhcmFtW2NoXVtlZmZlY3Rd"
    "ID0gcGFyYW0KICAgICAgICAgICAgICAgIGVsaWYgZWZmZWN0IGluICgweDQsIDB4Nyk6CiAgICAgICAg"
    "ICAgICAgICAgICAgIyBXaG9sZS1ieXRlPTAg4oaSIHVzZSBsYXN0IHdob2xlLWJ5dGUgbWVtb3J5Lgog"
    "ICAgICAgICAgICAgICAgICAgICMgTm9uLXplcm86IGFsc28gc3RvcmUgYXMgbGFzdC1ieXRlIG1lbW9y"
    "eSAobmliYmxlLWxldmVsCiAgICAgICAgICAgICAgICAgICAgIyBoYW5kbGluZyBpcyBkb25lIGJ5IEhU"
    "TUwvR0xTTCBkdXJpbmcgcGxheWJhY2spLgogICAgICAgICAgICAgICAgICAgIGlmIHBhcmFtID09IDAg"
    "YW5kIGVmZmVjdCBpbiBsYXN0X3BhcmFtW2NoXToKICAgICAgICAgICAgICAgICAgICAgICAgbmV3X3Bh"
    "cmFtID0gbGFzdF9wYXJhbVtjaF1bZWZmZWN0XQogICAgICAgICAgICAgICAgICAgICAgICBwYXRfY29w"
    "eVtiYXNlKzNdID0gbmV3X3BhcmFtCiAgICAgICAgICAgICAgICAgICAgICAgIHJld3JpdHRlbl9ub3Rl"
    "c19jb3VudCArPSAxCiAgICAgICAgICAgICAgICAgICAgZWxpZiBwYXJhbSAhPSAwOgogICAgICAgICAg"
    "ICAgICAgICAgICAgICBsYXN0X3BhcmFtW2NoXVtlZmZlY3RdID0gcGFyYW0KICAgICAgICAgICAgICAg"
    "ICMgNS82L0E6IG5vIHBhcmFtLW1lbW9yeSByZXdyaXRpbmcgKHRoZWlyIHBhcmFtPTAgaXMgbWVhbmlu"
    "Z2Z1bCkKCiAgICBub3RlcyA9IFtdCiAgICBmb3IgcGF0IGluIHJhbmdlKG1vZC5udW1fcGF0dGVybnMp"
    "OgogICAgICAgIGlmIHBhdCBpbiBwYXRfY29waWVzOgogICAgICAgICAgICBwZGF0YSA9IHBhdF9jb3Bp"
    "ZXNbcGF0XQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHBkYXRhID0gbW9kLnBhdHRlcm5zW3BhdF0K"
    "ICAgICAgICBmb3Igcm93IGluIHJhbmdlKDY0KToKICAgICAgICAgICAgZm9yIGNoIGluIHJhbmdlKE5D"
    "KToKICAgICAgICAgICAgICAgIGJhc2UgPSByb3cqcm93X3N0cmlkZSArIGNoKjQKICAgICAgICAgICAg"
    "ICAgIG5vdGVzLmFwcGVuZChieXRlcyhwZGF0YVtiYXNlOmJhc2UrNF0pKQogICAgaWYgcmV3cml0dGVu"
    "X25vdGVzX2NvdW50ID4gMDoKICAgICAgICBwcmludChmIiAgIOKame+4jyAgUGFyYW0tbWVtb3J5OiB7"
    "cmV3cml0dGVuX25vdGVzX2NvdW50fSBwYXJhbT0wIGVmZmVjdHMgcmV3cml0dGVuIHdpdGggcHJldmlv"
    "dXMgdmFsdWVzIikKICAgIHRvdGFsX25vdGVzID0gbGVuKG5vdGVzKQogICAgbnVtX3Jvd3MgICAgPSBt"
    "b2QubnVtX3BhdHRlcm5zICogNjQKCiAgICAjIFVuaXF1ZSBub24tZW1wdHkgbm90ZXMg4oaSIGRpY3Rp"
    "b25hcnkKICAgIHVuaXEgPSBzb3J0ZWQoc2V0KG4gZm9yIG4gaW4gbm90ZXMgaWYgbiAhPSBFTVBUWV9O"
    "T1RFKSkKICAgIGlkeF9ieXRlcyA9IDEgaWYgbGVuKHVuaXEpIDw9IDI1NiBlbHNlIDIKICAgIGFzc2Vy"
    "dCBsZW4odW5pcSkgPD0gNjU1MzYsIGYidG9vIG1hbnkgdW5pcXVlIG5vdGVzOiB7bGVuKHVuaXEpfSIK"
    "ICAgIG5vdGVfdG9faWR4ID0ge246aSBmb3IgaSxuIGluIGVudW1lcmF0ZSh1bmlxKX0KCiAgICAjIEJp"
    "dG1hcCAoMSBiaXQgcGVyIG5vdGUsIExTQi1maXJzdCB3aXRoaW4gZWFjaCBieXRlKQogICAgYml0bWFw"
    "ID0gYnl0ZWFycmF5KCh0b3RhbF9ub3RlcyArIDcpIC8vIDgpCiAgICBmb3IgaSwgbiBpbiBlbnVtZXJh"
    "dGUobm90ZXMpOgogICAgICAgIGlmIG4gIT0gRU1QVFlfTk9URToKICAgICAgICAgICAgYml0bWFwW2kg"
    "Pj4gM10gfD0gMSA8PCAoaSAmIDcpCgogICAgIyBJbmRleCBzdHJlYW0gKDEgb3IgMiBieXRlcyBwZXIg"
    "bm9uLWVtcHR5IG5vdGUsIGxpdHRsZS1lbmRpYW4gaWYgMkIpCiAgICBpZHhfc3RyZWFtID0gYnl0ZWFy"
    "cmF5KCkKICAgIGZvciBuIGluIG5vdGVzOgogICAgICAgIGlmIG4gIT0gRU1QVFlfTk9URToKICAgICAg"
    "ICAgICAgaSA9IG5vdGVfdG9faWR4W25dCiAgICAgICAgICAgIGlmIGlkeF9ieXRlcyA9PSAxOgogICAg"
    "ICAgICAgICAgICAgaWR4X3N0cmVhbS5hcHBlbmQoaSkKICAgICAgICAgICAgZWxzZToKICAgICAgICAg"
    "ICAgICAgIGlkeF9zdHJlYW0uYXBwZW5kKGkgJiAweEZGKQogICAgICAgICAgICAgICAgaWR4X3N0cmVh"
    "bS5hcHBlbmQoKGkgPj4gOCkgJiAweEZGKQoKICAgICMgUGVyLXJvdyBjb3VudDogY291bnQgb2Ygbm9u"
    "LWVtcHR5IG5vdGVzIElOIHRoaXMgcm93ICgwLi40KS4KICAgIHBlcl9yb3dfY291bnQgPSBbXQogICAg"
    "Zm9yIHJvdyBpbiByYW5nZShudW1fcm93cyk6CiAgICAgICAgY291bnQgPSBzdW0oMSBmb3IgY2ggaW4g"
    "cmFuZ2UoTkMpIGlmIG5vdGVzW3JvdypOQyArIGNoXSAhPSBFTVBUWV9OT1RFKQogICAgICAgIHBlcl9y"
    "b3dfY291bnQuYXBwZW5kKGNvdW50KQoKICAgICMgUHJlZml4IHN1bTogcHJlZml4W3Jvd10gPSBub24t"
    "ZW1wdHkgY291bnQgaW4gcm93cyBbMCwgcm93KSA9IHJhbmsgYXQgc3RhcnQgb2Ygcm93LgogICAgIyBT"
    "dG9yZWQgYXMgMTYtYml0IExFIHdvcmRzIHNvIGRlY29kZXIgaXMgTygxKS4KICAgICMgUmFuZ2U6IDAg"
    "dG8gfnRvdGFsX25vbl9lbXB0eSAo4omkIDU4ODggZm9yIDIzLXBhdCBNT0QpIOKGkiBmaXRzIGVhc2ls"
    "eSBpbiAxNiBiaXRzLgogICAgcHJlZml4ID0gWzBdICogbnVtX3Jvd3MKICAgIHJ1bm5pbmcgPSAwCiAg"
    "ICBmb3Igcm93IGluIHJhbmdlKG51bV9yb3dzKToKICAgICAgICBwcmVmaXhbcm93XSA9IHJ1bm5pbmcK"
    "ICAgICAgICBydW5uaW5nICs9IHBlcl9yb3dfY291bnRbcm93XQoKICAgIHJvd19zZWVrX2J5dGVzID0g"
    "Ynl0ZWFycmF5KCkKICAgIGZvciB2IGluIHByZWZpeDoKICAgICAgICBhc3NlcnQgMCA8PSB2IDwgNjU1"
    "MzYsIGYicHJlZml4IHt2fSBvdmVyZmxvd3MgMTYgYml0cyIKICAgICAgICByb3dfc2Vla19ieXRlcy5h"
    "cHBlbmQodiAmIDB4RkYpCiAgICAgICAgcm93X3NlZWtfYnl0ZXMuYXBwZW5kKCh2ID4+IDgpICYgMHhG"
    "RikKCiAgICByZXR1cm4gZGljdCgKICAgICAgICB0b3RhbF9ub3Rlcz10b3RhbF9ub3RlcywgbnVtX3Jv"
    "d3M9bnVtX3Jvd3MsCiAgICAgICAgdW5pcT11bmlxLCBub3RlX3RvX2lkeD1ub3RlX3RvX2lkeCwgaWR4"
    "X2J5dGVzPWlkeF9ieXRlcywKICAgICAgICBiaXRtYXA9Yml0bWFwLCBpZHhfc3RyZWFtPWlkeF9zdHJl"
    "YW0sCiAgICAgICAgcm93X3NlZWtfYnl0ZXM9cm93X3NlZWtfYnl0ZXMsIHByZWZpeD1wcmVmaXgsCiAg"
    "ICApCgojIOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkAojIDMtQklUIExJTkVBUiBT"
    "QU1QTEUgQ1JVTkNICiMg4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQCgpkZWYgZW5j"
    "b2RlX3NhbXBsZXNfcGFja2VkKG1vZCwgYml0cz0zKToKICAgICIiIkNvbmNhdGVuYXRlIGFsbCBzYW1w"
    "bGVzLCBlbmNvZGUgZWFjaCB0byBgYml0c2AgYml0cyAocm91bmRlZCkuCiAgICBTdXBwb3J0cyAzLWJp"
    "dCBhbmQgNC1iaXQgbGluZWFyIHF1YW50aXphdGlvbi4KICAgICAgMy1iaXQ6IGNvZGUgMC4uNywgbGV2"
    "ZWxzIChjb2RlKjMyIC0gMTEyKSwgc3RlcCAzMi8yNTYgPSAxMi41JQogICAgICA0LWJpdDogY29kZSAw"
    "Li4xNSwgbGV2ZWxzIChjb2RlKjE2IC0gMTIwKSwgc3RlcCAxNi8yNTYgPSA2LjI1JSAoKzYgZEIgU05S"
    "KQogICAgUmV0dXJucyBwYWNrZWQgYnl0ZXMgKyBwZXItc2FtcGxlIHN0YXJ0IGluZGljZXMgKGxvZ2lj"
    "YWwgc2FtcGxlIHVuaXRzKS4iIiIKICAgIGlmIGJpdHMgbm90IGluICgzLCA0KToKICAgICAgICByYWlz"
    "ZSBWYWx1ZUVycm9yKGYiYml0cyBtdXN0IGJlIDMgb3IgNCwgZ290IHtiaXRzfSIpCgogICAgY29uY2F0"
    "X3NpZ25lZCA9IFtdCiAgICBzdGFydHMgPSBbXQogICAgZm9yIHMgaW4gbW9kLnNhbXBsZV9ieXRlczoK"
    "ICAgICAgICBzdGFydHMuYXBwZW5kKGxlbihjb25jYXRfc2lnbmVkKSkKICAgICAgICBmb3IgYiBpbiBz"
    "OgogICAgICAgICAgICBjb25jYXRfc2lnbmVkLmFwcGVuZChiIC0gMjU2IGlmIGIgPj0gMTI4IGVsc2Ug"
    "YikKICAgICAgICBjb25jYXRfc2lnbmVkLmV4dGVuZChbMF0gKiAxNikKCiAgICB0b3RhbF9zYW1wbGVz"
    "ID0gbGVuKGNvbmNhdF9zaWduZWQpCiAgICBjb2RlcyA9IGJ5dGVhcnJheSgpCiAgICBtYXhfY29kZSA9"
    "ICgxIDw8IGJpdHMpIC0gMQogICAgc2hpZnQgPSA4IC0gYml0cwogICAgZm9yIHN2IGluIGNvbmNhdF9z"
    "aWduZWQ6CiAgICAgICAgdW5zaWduZWRfb2Zmc2V0ID0gc3YgKyAxMjggICMgWzAsIDI1NV0KICAgICAg"
    "ICBjb2RlID0gdW5zaWduZWRfb2Zmc2V0ID4+IHNoaWZ0CiAgICAgICAgaWYgY29kZSA+IG1heF9jb2Rl"
    "OiBjb2RlID0gbWF4X2NvZGUKICAgICAgICBjb2Rlcy5hcHBlbmQoY29kZSkKCiAgICB0b3RhbF9iaXRz"
    "ID0gdG90YWxfc2FtcGxlcyAqIGJpdHMKICAgIHRvdGFsX2J5dGVzID0gKHRvdGFsX2JpdHMgKyA3KSAv"
    "LyA4CiAgICBwYWNrZWQgPSBieXRlYXJyYXkodG90YWxfYnl0ZXMpCgogICAgaWYgYml0cyA9PSA0Ogog"
    "ICAgICAgICMgTmliYmxlIHBhY2tpbmc6IDIgY29kZXMgcGVyIGJ5dGUsIGxvdyBuaWJibGUgZmlyc3QK"
    "ICAgICAgICBmb3IgaSwgYyBpbiBlbnVtZXJhdGUoY29kZXMpOgogICAgICAgICAgICBieXRlX3BvcyA9"
    "IGkgPj4gMQogICAgICAgICAgICBpZiBpICYgMToKICAgICAgICAgICAgICAgIHBhY2tlZFtieXRlX3Bv"
    "c10gfD0gKGMgJiAweEYpIDw8IDQKICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgIHBhY2tl"
    "ZFtieXRlX3Bvc10gfD0gYyAmIDB4RgogICAgZWxzZTogICMgYml0cyA9PSAzCiAgICAgICAgZm9yIGks"
    "IGMgaW4gZW51bWVyYXRlKGNvZGVzKToKICAgICAgICAgICAgYml0X3BvcyAgID0gaSAqIDMKICAgICAg"
    "ICAgICAgYnl0ZV9wb3MgID0gYml0X3BvcyA+PiAzCiAgICAgICAgICAgIGJpdF9zaGlmdCA9IGJpdF9w"
    "b3MgJiA3CiAgICAgICAgICAgIHZhbCA9IChjICYgNykgPDwgYml0X3NoaWZ0CiAgICAgICAgICAgIHBh"
    "Y2tlZFtieXRlX3Bvc10gfD0gdmFsICYgMHhGRgogICAgICAgICAgICBpZiBiaXRfc2hpZnQgPiA1IGFu"
    "ZCBieXRlX3BvcyArIDEgPCB0b3RhbF9ieXRlczoKICAgICAgICAgICAgICAgIHBhY2tlZFtieXRlX3Bv"
    "cyArIDFdIHw9ICh2YWwgPj4gOCkgJiAweEZGCgogICAgcmV0dXJuIHBhY2tlZCwgc3RhcnRzLCB0b3Rh"
    "bF9zYW1wbGVzCgojIEJhY2t3YXJkLWNvbXBhdCBhbGlhcwpkZWYgZW5jb2RlX3NhbXBsZXNfM2JpdCht"
    "b2QpOgogICAgcmV0dXJuIGVuY29kZV9zYW1wbGVzX3BhY2tlZChtb2QsIGJpdHM9MykKCgpkZWYgY29t"
    "cHV0ZV9yb3dfc3BlZWRfdGFibGUobW9kKToKICAgICIiIlNpbXVsYXRlIHRoZSBzb25nIHRvIGZpbmQg"
    "cGVyLXJvdyBTUEVFRCAoaG9ub3VyaW5nIEZ4eC9EeHgvQnh4IGVmZmVjdHMpLgogICAgUmV0dXJucyBy"
    "b3dTcGVlZFtudW1fc29uZ19yb3dzXSBhbmQgcm93U3RhcnRUaWNrW251bV9zb25nX3Jvd3MrMV0uCiAg"
    "ICBDb3JyZWN0bHkgaGFuZGxlcyBEeHggKHBhdHRlcm4gYnJlYWspIGFuZCBCeHggKHBvc2l0aW9uIGp1"
    "bXApIHdoaWNoCiAgICBzaG9ydGVuIGEgcGF0dGVybidzIGVmZmVjdGl2ZSByb3cgY291bnQuIiIiCiAg"
    "ICBzcGVlZCA9IDYgICMgUHJvVHJhY2tlciBkZWZhdWx0CiAgICBicG0gICA9IDEyNQogICAgcm93U3Bl"
    "ZWQgPSBbXQogICAgYnBtX2NoYW5nZXMgPSBGYWxzZQogICAgZm9yIHBvcyBpbiByYW5nZShtb2Quc29u"
    "Z19sZW5ndGgpOgogICAgICAgIHBhdF9pZHggPSBtb2QucGF0dGVybl9vcmRlcltwb3NdCiAgICAgICAg"
    "cGRhdGEgPSBtb2QucGF0dGVybnNbcGF0X2lkeF0KICAgICAgICBicm9rZSA9IEZhbHNlCiAgICAgICAg"
    "Zm9yIHJvdyBpbiByYW5nZSg2NCk6CiAgICAgICAgICAgICMgU2NhbiBhbGwgNCBjaGFubmVscyBmb3Ig"
    "Rnh4IC8gRHh4IC8gQnh4IG9uIHRoaXMgcm93CiAgICAgICAgICAgIGZvciBjaCBpbiByYW5nZShtb2Qu"
    "bnVtX2NoYW5uZWxzKToKICAgICAgICAgICAgICAgIGJhc2UgPSByb3cgKiBtb2QubnVtX2NoYW5uZWxz"
    "ICogNCArIGNoICogNAogICAgICAgICAgICAgICAgYjAsIGIxLCBiMiwgYjMgPSBwZGF0YVtiYXNlOmJh"
    "c2UrNF0KICAgICAgICAgICAgICAgIGVmZmVjdCA9IGIyICYgMHgwRgogICAgICAgICAgICAgICAgcGFy"
    "YW0gID0gYjMKICAgICAgICAgICAgICAgIGlmIGVmZmVjdCA9PSAweEYgYW5kIHBhcmFtID4gMDoKICAg"
    "ICAgICAgICAgICAgICAgICBpZiBwYXJhbSA8IDB4MjA6CiAgICAgICAgICAgICAgICAgICAgICAgIHNw"
    "ZWVkID0gcGFyYW0KICAgICAgICAgICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgICAgICAg"
    "ICBpZiBicG0gIT0gcGFyYW06CiAgICAgICAgICAgICAgICAgICAgICAgICAgICBicG1fY2hhbmdlcyA9"
    "IFRydWUKICAgICAgICAgICAgICAgICAgICAgICAgYnBtID0gcGFyYW0KICAgICAgICAgICAgICAgIGVs"
    "aWYgZWZmZWN0ID09IDB4RCBvciBlZmZlY3QgPT0gMHhCOgogICAgICAgICAgICAgICAgICAgIGJyb2tl"
    "ID0gVHJ1ZSAgICMgcGF0dGVybiBicmVhayAvIHBvc2l0aW9uIGp1bXAKICAgICAgICAgICAgcm93U3Bl"
    "ZWQuYXBwZW5kKHNwZWVkKQogICAgICAgICAgICBpZiBicm9rZToKICAgICAgICAgICAgICAgIGJyZWFr"
    "ICAgIyBzdG9wIGFkZGluZyByb3dzIGZvciB0aGlzIHNvbmcgcG9zaXRpb24KICAgIHJvd1N0YXJ0VGlj"
    "ayA9IFswXQogICAgZm9yIHMgaW4gcm93U3BlZWQ6CiAgICAgICAgcm93U3RhcnRUaWNrLmFwcGVuZChy"
    "b3dTdGFydFRpY2tbLTFdICsgcykKICAgIHJldHVybiByb3dTcGVlZCwgcm93U3RhcnRUaWNrLCBicG1f"
    "Y2hhbmdlcwoKCmRlZiBlbmNvZGVfc2FtcGxlc192cTJkKG1vZCwgSz0yNTYsIHdlaWdodGVkPVRydWUs"
    "IGRvd25zYW1wbGU9MiwgYml0cmF0ZT0nbWVkJywgdmVjX2RpbT0yLCBub19ydnEyPUZhbHNlKToKICAg"
    "ICIiIjItc3RhZ2UgUmVzaWR1YWwgVlEgd2l0aCBGRlQtZ3VpZGVkIHBlci1zYW1wbGUgZGVjaW1hdGlv"
    "bi4KICAgIFBlci1zYW1wbGUgRFMgdmlhIEZGVCBiYW5kd2lkdGggYW5hbHlzaXMg4oCUIERTPTEgZm9y"
    "IGZ1bGwtYmFuZHdpZHRoIHNhbXBsZXMKICAgIChwcmVzZXJ2ZXMgYWxsIEhGKSwgb25seSBkb3duc2Ft"
    "cGxlIGlmIGNvbnRlbnQgaXMgZ2VudWluZWx5IGxvdy1iYW5kd2lkdGguCiAgICBSYXcgc3RyaWRlIGRl"
    "Y2ltYXRpb24gKG5vIExQRikuIGJ3RmFjdG9yIHBlciBzYW1wbGUgPSBhY3R1YWwgRFMgdXNlZC4KICAg"
    "ICIiIgogICAgaW1wb3J0IG51bXB5IGFzIG5wCiAgICBmcm9tIHNrbGVhcm4uY2x1c3RlciBpbXBvcnQg"
    "TWluaUJhdGNoS01lYW5zCgogICAgIyBCaXRyYXRlIOKGkiBjb2RlYm9vayBzaXplIChtcDMtc3R5bGUg"
    "cXVhbGl0eSBrbm9iKQogICAgX2JpdHJhdGVfdGFibGUgPSB7CiAgICAgICAgJ2xvJzogICAgKDEyOCwg"
    "IDY0KSwgICAjIDEzIGJpdHMvcGFpciwgc21hbGxlc3QrZ3JhaW55CiAgICAgICAgJ21lZCc6ICAgKDI1"
    "NiwgMTI4KSwgICAjIDE1IGJpdHMvcGFpciwgYmFsYW5jZWQKICAgICAgICAnaGknOiAgICAoNTEyLCAy"
    "NTYpLCAgICMgMTcgYml0cy9wYWlyLCBkZWZhdWx0CiAgICAgICAgJ3VsdHJhJzooMTAyNCwgNTEyKSwg"
    "ICAjIDE5IGJpdHMvcGFpciwgbmVhci10cmFuc3BhcmVudAogICAgfQogICAgSzEsIEsyID0gX2JpdHJh"
    "dGVfdGFibGUuZ2V0KGJpdHJhdGUsIF9iaXRyYXRlX3RhYmxlWydoaSddKQogICAgaWYgbm9fcnZxMjoK"
    "ICAgICAgICBLMiA9IDAgICMgc2lnbmFsOiBza2lwIHN0YWdlIDIKICAgIEJJVFMxID0gaW50KG5wLmNl"
    "aWwobnAubG9nMihLMSkpKQogICAgQklUUzIgPSBpbnQobnAuY2VpbChucC5sb2cyKEsyKSkpIGlmIEsy"
    "ID4gMCBlbHNlIDAKICAgIEJJVFNfVE9UQUwgPSBCSVRTMSArIEJJVFMyICAjIGlmIEsyPT0wLCBCSVRT"
    "Mj09MCwgc28ganVzdCBCSVRTMQoKICAgIGRlZiBoZl9yYXRpbyhyYXdfYnl0ZXMsIGxlbmd0aCwgbnlx"
    "dWlzdF9oej0yMjA1MCk6CiAgICAgICAgIiIiRnJhY3Rpb24gb2YgZW5lcmd5IGFib3ZlIDhrSHog4oCU"
    "IGhpZ2ggPSBwZXJjdXNzaW9uL2N5bWJhbC4iIiIKICAgICAgICBpZiBsZW5ndGggPCAzMjogcmV0dXJu"
    "IDAuMAogICAgICAgIGRhdGEgPSBucC5mcm9tYnVmZmVyKHJhd19ieXRlc1s6bGVuZ3RoXSwgZHR5cGU9"
    "bnAuaW50OCkuYXN0eXBlKG5wLmZsb2F0MzIpCiAgICAgICAgZmZ0ICA9IG5wLmFicyhucC5mZnQucmZm"
    "dChkYXRhWzptaW4obGVuZ3RoLCA0MDk2KV0pKQogICAgICAgIGUgICAgPSBmbG9hdChucC5zdW0oZmZ0"
    "KioyKSkgKyAxZS0xMAogICAgICAgIGN1dCAgPSBtYXgoMSwgaW50KGxlbihmZnQpICogODAwMCAvIG55"
    "cXVpc3RfaHopKQogICAgICAgIHJldHVybiBmbG9hdChucC5zdW0oZmZ0W2N1dDpdKioyKSkgLyBlCgog"
    "ICAgY29uY2F0X2RzID0gW10KICAgIHN0YXJ0cyAgICA9IFtdCiAgICBzYW1wbGVfZHMgPSBbXSAgIyBw"
    "ZXItc2FtcGxlIGFjdHVhbCBEUyB1c2VkCiAgICB0b3RhbF9zYW1wbGVzX2Z1bGwgPSAwCgogICAgZm9y"
    "IHMsIHJhd19ieXRlcyBpbiB6aXAobW9kLnNhbXBsZXNfaW5mbywgbW9kLnNhbXBsZV9ieXRlcyk6CiAg"
    "ICAgICAgc3RhcnRzLmFwcGVuZChsZW4oY29uY2F0X2RzKSkKICAgICAgICBpZiBzWydsZW5ndGgnXSA+"
    "IDA6CiAgICAgICAgICAgIHJhdyA9IG5wLmZyb21idWZmZXIocmF3X2J5dGVzLCBkdHlwZT1ucC5pbnQ4"
    "KS5hc3R5cGUobnAuZmxvYXQzMikgLyAxMjguMAogICAgICAgICAgICB0b3RhbF9zYW1wbGVzX2Z1bGwg"
    "Kz0gbGVuKHJhdykKICAgICAgICAgICAgIyBQZXItc2FtcGxlIERTIHZpYSBGRlQgYmFuZHdpZHRoIGFu"
    "YWx5c2lzIChtaXJyb3JzIEhUTUwgcGxheWVyJ3MKICAgICAgICAgICAgIyBid19jb21wcmVzc19zYW1w"
    "bGUpLiAgLS1kb3duc2FtcGxlIGlzIGEgQ0FQLCBub3QgYSBmbG9vcjogZnVsbC0KICAgICAgICAgICAg"
    "IyBiYW5kd2lkdGggc2FtcGxlcyAoZ3VpdGFycywgdm9jYWxzKSBzdGF5IGF0IERTPTEsIG5hcnJvdy1i"
    "YW5kCiAgICAgICAgICAgICMgc2FtcGxlcyAobG93IGJhc3MsIG11dGVkIGluc3RydW1lbnRzKSBkcm9w"
    "IHRvIERTPTIvNC84LgogICAgICAgICAgICAjCiAgICAgICAgICAgICMgV2l0aG91dCB0aGlzIHRoZSBH"
    "TFNMIHdhcyBmb3JjZS1kZWNpbWF0aW5nIGV2ZXJ5IHNhbXBsZSB0bwogICAgICAgICAgICAjIGRvd25z"
    "YW1wbGUsIHByb2R1Y2luZyB0aGUgIjgga0h6IGxvLWZpIiBhcnRpZmFjdHMgdGhlIEhUTUwKICAgICAg"
    "ICAgICAgIyBwbGF5ZXIgYXZvaWRlZC4KICAgICAgICAgICAgc3IgPSA0NDEwMC4wCiAgICAgICAgICAg"
    "IG5fZmZ0ID0gbWluKGxlbihyYXcpLCA4MTkyKQogICAgICAgICAgICBmZnRfbWFnID0gbnAuYWJzKG5w"
    "LmZmdC5yZmZ0KHJhd1s6bl9mZnRdICogbnAuaGFubmluZyhuX2ZmdCkpKQogICAgICAgICAgICBmcmVx"
    "cyAgID0gbnAuZmZ0LnJmZnRmcmVxKG5fZmZ0LCAxLjAgLyBzcilbOmxlbihmZnRfbWFnKV0KICAgICAg"
    "ICAgICAgcGVhayAgICA9IGZsb2F0KG5wLm1heChmZnRfbWFnKSkgKyAxZS0xMgogICAgICAgICAgICBz"
    "aWdfYmlucyA9IG5wLndoZXJlKGZmdF9tYWcgPiBwZWFrICogMC4wMDUpWzBdCiAgICAgICAgICAgIG1h"
    "eF9mcmVxID0gZmxvYXQoZnJlcXNbc2lnX2JpbnNbLTFdXSkgaWYgbGVuKHNpZ19iaW5zKSBlbHNlIDIy"
    "MDUwLjAKICAgICAgICAgICAgIyBVc2VyJ3MgLS1kb3duc2FtcGxlIGlzIHRoZSBGTE9PUiAoYWx3YXlz"
    "IGF0IGxlYXN0IHRoaXMgbXVjaCkuCiAgICAgICAgICAgICMgQmFuZHdpZHRoIGFuYWx5c2lzIGNhbiBj"
    "aG9vc2UgdG8gZGVjaW1hdGUgTU9SRSBmb3IgZ2VudWluZWx5CiAgICAgICAgICAgICMgbG93LWJhbmR3"
    "aWR0aCBjb250ZW50IChlLmcuIHN1Yi1iYXNzIGF0IERTPTQgZXZlbiB3aGVuIHVzZXIKICAgICAgICAg"
    "ICAgIyByZXF1ZXN0ZWQgRFM9MikuICBOZXZlciBsZXNzIHRoYW4gdXNlcidzIHJlcXVlc3QuCiAgICAg"
    "ICAgICAgIGFjdHVhbF9kcyA9IGRvd25zYW1wbGUKICAgICAgICAgICAgaWYgZG93bnNhbXBsZSA8IDE2"
    "OgogICAgICAgICAgICAgICAgIyBUcnkgaGlnaGVyIGZhY3RvcnMgb25seSDigJQgbmV2ZXIgbGVzcyB0"
    "aGFuIHVzZXIncyByZXF1ZXN0LgogICAgICAgICAgICAgICAgZm9yIGYgaW4gW2Rvd25zYW1wbGUgKiAy"
    "LCBkb3duc2FtcGxlICogNF06CiAgICAgICAgICAgICAgICAgICAgaWYgZiA+IDE2OiBicmVhawogICAg"
    "ICAgICAgICAgICAgICAgIGlmIHNyIC8gZiA+PSBtYXhfZnJlcSAqIDIuNDoKICAgICAgICAgICAgICAg"
    "ICAgICAgICAgYWN0dWFsX2RzID0gZgogICAgICAgICAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAg"
    "ICAgICAgICAgICAgIGJyZWFrICAjIGlmIDJ4IGRvZXNuJ3Qgc2F0aXNmeSBOeXF1aXN0LCA0eCB3b24n"
    "dCBlaXRoZXIKICAgICAgICAgICAgIyBSYXcgc3RyaWRlIGRlY2ltYXRpb24g4oCUIG5vIExQRiwgcHJl"
    "c2VydmVzIEhGIGNvbnRlbnQKICAgICAgICAgICAgaWYgYWN0dWFsX2RzID4gMToKICAgICAgICAgICAg"
    "ICAgIGRzID0gcmF3Wzo6YWN0dWFsX2RzXS5jb3B5KCkKICAgICAgICAgICAgZWxzZToKICAgICAgICAg"
    "ICAgICAgIGRzID0gcmF3LmNvcHkoKQogICAgICAgICAgICBzYW1wbGVfZHMuYXBwZW5kKGFjdHVhbF9k"
    "cykKICAgICAgICAgICAgY29uY2F0X2RzLmV4dGVuZChkcy50b2xpc3QoKSkKICAgICAgICAgICAgIyBM"
    "b29wLXNlYW0gc21vb3RoaW5nOiBmb3IgbG9vcGluZyBzYW1wbGVzLCByZXBsYWNlIHRoZSBwb3N0LWxv"
    "b3AKICAgICAgICAgICAgIyBndWFyZCByZWdpb24gd2l0aCB0aGUgRklSU1QgZmV3IHNhbXBsZXMgZnJv"
    "bSBsb29wX3N0YXJ0LgogICAgICAgICAgICAjIFRoaXMgbWFrZXMgdmVjdG9ycyBuZWFyIGxvb3BfZW5k"
    "IGluY2x1ZGUgcHJvcGVyIHdyYXAgY29udGV4dCBzbwogICAgICAgICAgICAjIFZRIHF1YW50aXphdGlv"
    "biBkb2Vzbid0IGludHJvZHVjZSBhIHN0ZXAgZGlzY29udGludWl0eSBhdCB0aGUKICAgICAgICAgICAg"
    "IyBsb29wIGJvdW5kYXJ5LiAgV2l0aG91dCB0aGlzLCB2ZWNfZGltPTggcHJvZHVjZXMgYW4gYXVkaWJs"
    "ZSBidXp6CiAgICAgICAgICAgICMgYXQgdGhlIGxvb3AgcmF0ZSAoc2FtcGxlW2xvb3BFbmQtMV0gYW5k"
    "IHNhbXBsZVtsb29wU3RhcnRdIGdldAogICAgICAgICAgICAjIHF1YW50aXplZCB0byBpbmNvbXBhdGli"
    "bGUgY29kZWJvb2sgcHJvdG90eXBlcykuCiAgICAgICAgICAgICMKICAgICAgICAgICAgIyBFbmNvZGVy"
    "IE1PREZpbGUgc3RvcmVzIGxvb3Bfc3RhcnQvbG9vcF9sZW4gaW4gUkFXIGJ5dGUgdW5pdHMKICAgICAg"
    "ICAgICAgIyAoYWxyZWFkeSBwcmUtbXVsdGlwbGllZCBieSAyIGluIHNhbXBsZXNfaW5mbykuCiAgICAg"
    "ICAgICAgIGxvb3BfbGVuX3JhdyA9IGludChzLmdldCgnbG9vcF9sZW4nLCAwKSBvciAwKQogICAgICAg"
    "ICAgICBpZiBsb29wX2xlbl9yYXcgPiA0OgogICAgICAgICAgICAgICAgbG9vcF9zdGFydF9yYXcgPSBp"
    "bnQocy5nZXQoJ2xvb3Bfc3RhcnQnLCAwKSBvciAwKQogICAgICAgICAgICAgICAgIyBDb252ZXJ0IHRv"
    "IGRlY2ltYXRlZC1zdHJlYW0gaW5kZXggKG1hdGNoZXMgYGRzYCBhcnJheSBpbmRleGluZykKICAgICAg"
    "ICAgICAgICAgIGxvb3Bfc3RhcnRfZHMgPSBsb29wX3N0YXJ0X3JhdyAvLyBhY3R1YWxfZHMKICAgICAg"
    "ICAgICAgICAgICMgQ29tcHV0ZSB0b3RhbCBwYWRkaW5nIG5lZWRlZDogYWxpZ24tdG8tdmVjX2RpbSAr"
    "IGV4dHJhIGd1YXJkCiAgICAgICAgICAgICAgICBwYWRfY291bnQgPSAodmVjX2RpbSAtIGxlbihjb25j"
    "YXRfZHMpICUgdmVjX2RpbSkgJSB2ZWNfZGltICsgOAogICAgICAgICAgICAgICAgIyBUYWtlIHBhZF9j"
    "b3VudCBzYW1wbGVzIHN0YXJ0aW5nIGZyb20gbG9vcF9zdGFydCBpbiB0aGUKICAgICAgICAgICAgICAg"
    "ICMgZGVjaW1hdGVkIGRhdGEg4oCUIHRoaXMgaXMgd2hhdCBwbGF5YmFjayB3cmFwcyB0by4KICAgICAg"
    "ICAgICAgICAgIHdyYXBfZGF0YSA9IFtdCiAgICAgICAgICAgICAgICBpZiBsb29wX3N0YXJ0X2RzIDwg"
    "bGVuKGRzKToKICAgICAgICAgICAgICAgICAgICB0YWtlID0gbWluKHBhZF9jb3VudCwgbGVuKGRzKSAt"
    "IGxvb3Bfc3RhcnRfZHMpCiAgICAgICAgICAgICAgICAgICAgd3JhcF9kYXRhLmV4dGVuZChkcy50b2xp"
    "c3QoKVtsb29wX3N0YXJ0X2RzOmxvb3Bfc3RhcnRfZHMrdGFrZV0pCiAgICAgICAgICAgICAgICB3aGls"
    "ZSBsZW4od3JhcF9kYXRhKSA8IHBhZF9jb3VudDoKICAgICAgICAgICAgICAgICAgICB3cmFwX2RhdGEu"
    "YXBwZW5kKDApCiAgICAgICAgICAgICAgICBjb25jYXRfZHMuZXh0ZW5kKHdyYXBfZGF0YSkKICAgICAg"
    "ICAgICAgZWxzZToKICAgICAgICAgICAgICAgICMgTm9uLWxvb3Bpbmc6IHBhZCB3aXRoIHplcm9zIChv"
    "cmlnaW5hbCBiZWhhdmlvcikKICAgICAgICAgICAgICAgIHdoaWxlIGxlbihjb25jYXRfZHMpICUgdmVj"
    "X2RpbTogY29uY2F0X2RzLmFwcGVuZCgwKQogICAgICAgICAgICAgICAgY29uY2F0X2RzLmV4dGVuZChb"
    "MF0gKiA4KQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHNhbXBsZV9kcy5hcHBlbmQoZG93bnNhbXBs"
    "ZSkKICAgICAgICAgICAgIyBFbXB0eSBzYW1wbGU6IGp1c3QgcGFkCiAgICAgICAgICAgIHdoaWxlIGxl"
    "bihjb25jYXRfZHMpICUgdmVjX2RpbTogY29uY2F0X2RzLmFwcGVuZCgwKQogICAgICAgICAgICBjb25j"
    "YXRfZHMuZXh0ZW5kKFswXSAqIDgpCgogICAgd2hpbGUgbGVuKGNvbmNhdF9kcykgJSB2ZWNfZGltOiBj"
    "b25jYXRfZHMuYXBwZW5kKDApCiAgICB0b3RhbF9zYW1wbGVzID0gbGVuKGNvbmNhdF9kcykKCiAgICB2"
    "ZWN0b3JzID0gbnAuYXJyYXkoY29uY2F0X2RzLCBkdHlwZT1ucC5mbG9hdDMyKS5yZXNoYXBlKC0xLCB2"
    "ZWNfZGltKQoKICAgICMgU3RhZ2UgMSDigJQgcmluZy13ZWlnaHRlZAogICAgd2VpZ2h0cyA9IE5vbmUK"
    "ICAgIGlmIHdlaWdodGVkOgogICAgICAgIHNsb3BlcyAgPSBucC5hYnModmVjdG9yc1s6LCAtMV0gLSB2"
    "ZWN0b3JzWzosIDBdKQogICAgICAgIHdlaWdodHMgPSAoc2xvcGVzICsgMS4wKQogICAgICAgIHdlaWdo"
    "dHMgLz0gd2VpZ2h0cy5tZWFuKCkKCiAgICBwcmludChmIiAgUlZRIMOXe2Rvd25zYW1wbGV9IFN0YWdl"
    "IDE6IEs9e0sxfSBvbiB7bGVuKHZlY3RvcnMpfSB7dmVjX2RpbX0tdmVjdG9ycy4uLiIsIGZsdXNoPVRy"
    "dWUpCiAgICBrbTEgPSBNaW5pQmF0Y2hLTWVhbnMobl9jbHVzdGVycz1LMSwgbl9pbml0PTUsIG1heF9p"
    "dGVyPTYwLCBiYXRjaF9zaXplPTgxOTIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgcmFuZG9tX3N0"
    "YXRlPTAsIHJlYXNzaWdubWVudF9yYXRpbz0wLjAxKQogICAga20xLmZpdCh2ZWN0b3JzLCBzYW1wbGVf"
    "d2VpZ2h0PXdlaWdodHMpCiAgICBjb2RlczEgICA9IGttMS5wcmVkaWN0KHZlY3RvcnMpLmFzdHlwZShu"
    "cC5pbnQzMikKICAgICMgQ2VudHJvaWRzIGFyZSBpbiBbLTEsMV0gZmxvYXQgcmFuZ2Ug4oCUIHNjYWxl"
    "IGJhY2sgdG8gWy0xMjgsMTI3XSBpbnQgcmFuZ2UgZm9yIHN0b3JhZ2UKICAgIGNiMSAgICAgID0gbnAu"
    "Y2xpcChucC5yb3VuZChrbTEuY2x1c3Rlcl9jZW50ZXJzXyAqIDEyOCksIC0xMjgsIDEyNykuYXN0eXBl"
    "KG5wLmludDMyKQogICAgcmVzaWR1YWwgPSB2ZWN0b3JzIC0ga20xLmNsdXN0ZXJfY2VudGVyc19bY29k"
    "ZXMxXQoKICAgIHNucjEgPSAxMCpucC5sb2cxMChucC5tZWFuKHZlY3RvcnMqKjIpIC8gKG5wLm1lYW4o"
    "cmVzaWR1YWwqKjIpICsgMWUtOSkpCiAgICBwcmludChmIiAgU3RhZ2UgMSBTTlI6IHtzbnIxOi4yZn0g"
    "ZEIiLCBmbHVzaD1UcnVlKQoKICAgICMgU3RhZ2UgMiAoc2tpcHBlZCB3aGVuIG5vX3J2cTIg4oaSIEsy"
    "PT0wKQogICAgaWYgSzIgPiAwOgogICAgICAgIHByaW50KGYiICBSVlEgU3RhZ2UgMjogSz17SzJ9IG9u"
    "IHJlc2lkdWFsLi4uIiwgZmx1c2g9VHJ1ZSkKICAgICAgICBrbTIgPSBNaW5pQmF0Y2hLTWVhbnMobl9j"
    "bHVzdGVycz1LMiwgbl9pbml0PTUsIG1heF9pdGVyPTYwLCBiYXRjaF9zaXplPTgxOTIsCiAgICAgICAg"
    "ICAgICAgICAgICAgICAgICAgICAgIHJhbmRvbV9zdGF0ZT0xLCByZWFzc2lnbm1lbnRfcmF0aW89MC4w"
    "MSkKICAgICAgICBrbTIuZml0KHJlc2lkdWFsKQogICAgICAgIGNvZGVzMiAgICAgICAgID0ga20yLnBy"
    "ZWRpY3QocmVzaWR1YWwpLmFzdHlwZShucC5pbnQzMikKICAgICAgICBjYjIgICAgICAgICAgICA9IG5w"
    "LmNsaXAobnAucm91bmQoa20yLmNsdXN0ZXJfY2VudGVyc18gKiAxMjgpLCAtMTI4LCAxMjcpLmFzdHlw"
    "ZShucC5pbnQzMikKICAgICAgICBmaW5hbF9yZXNpZHVhbCA9IHJlc2lkdWFsIC0ga20yLmNsdXN0ZXJf"
    "Y2VudGVyc19bY29kZXMyXQogICAgICAgIHNucjIgPSAxMCpucC5sb2cxMChucC5tZWFuKHZlY3RvcnMq"
    "KjIpIC8gKG5wLm1lYW4oZmluYWxfcmVzaWR1YWwqKjIpICsgMWUtOSkpCiAgICAgICAgcHJpbnQoZiIg"
    "IFJWUSB0b3RhbCBTTlI6IHtzbnIyOi4yZn0gZEIgKCt7c25yMi1zbnIxOi4yZn0gZEIgZnJvbSBzdGFn"
    "ZSAyKSIsIGZsdXNoPVRydWUpCiAgICBlbHNlOgogICAgICAgIHByaW50KGYiICDimqEgU3RhZ2UgMiBz"
    "a2lwcGVkICgtLW5vLXJ2cTIpOiBTTlIgPSB7c25yMTouMmZ9IGRCIiwgZmx1c2g9VHJ1ZSkKICAgICAg"
    "ICBjb2RlczIgPSBucC56ZXJvc19saWtlKGNvZGVzMSkgICMgcGxhY2Vob2xkZXIKICAgICAgICBjYjIg"
    "ICAgPSBucC56ZXJvcygoMCwgdmVjX2RpbSksIGR0eXBlPW5wLmludDMyKSAgIyBlbXB0eSBLMiBjb2Rl"
    "Ym9vawoKICAgICMgUGFjayBCSVRTMStCSVRTMiBiaXRzIHBlciB2ZWN0b3IgTFNCLWZpcnN0CiAgICAj"
    "IFdoZW4gbm9fcnZxMiAoSzI9PTApLCBCSVRTMj09MCBzbyBjb21iaW5lZCBjb2xsYXBzZXMgdG8ganVz"
    "dCBjb2RlczEgYml0cy4KICAgIG5fdmVjcyAgICAgID0gbGVuKHZlY3RvcnMpCiAgICB0b3RhbF9iaXRz"
    "ICA9IG5fdmVjcyAqIEJJVFNfVE9UQUwKICAgIHRvdGFsX2J5dGVzID0gKHRvdGFsX2JpdHMgKyA3KSAv"
    "LyA4CiAgICBjb2Rlc19ieXRlcyA9IGJ5dGVhcnJheSh0b3RhbF9ieXRlcykKICAgIG1hc2sxID0gKDEg"
    "PDwgQklUUzEpIC0gMQogICAgbWFzazIgPSAoMSA8PCBCSVRTMikgLSAxIGlmIEJJVFMyID4gMCBlbHNl"
    "IDAKICAgIGZvciBpIGluIHJhbmdlKG5fdmVjcyk6CiAgICAgICAgaWYgSzIgPiAwOgogICAgICAgICAg"
    "ICBjb21iaW5lZCAgPSAoaW50KGNvZGVzMVtpXSkgJiBtYXNrMSkgfCAoKGludChjb2RlczJbaV0pICYg"
    "bWFzazIpIDw8IEJJVFMxKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGNvbWJpbmVkICA9IGludChj"
    "b2RlczFbaV0pICYgbWFzazEKICAgICAgICBiaXRfcG9zICAgPSBpICogQklUU19UT1RBTAogICAgICAg"
    "IGJ5dGVfcG9zICA9IGJpdF9wb3MgPj4gMwogICAgICAgIGJpdF9zaGlmdCA9IGJpdF9wb3MgJiA3CiAg"
    "ICAgICAgdmFsID0gY29tYmluZWQgPDwgYml0X3NoaWZ0CiAgICAgICAgY29kZXNfYnl0ZXNbYnl0ZV9w"
    "b3NdICAgICB8PSB2YWwgICAgICAgICYgMHhGRgogICAgICAgIGlmIGJ5dGVfcG9zKzEgPCB0b3RhbF9i"
    "eXRlczogY29kZXNfYnl0ZXNbYnl0ZV9wb3MrMV0gfD0gKHZhbCA+PiA4KSAgJiAweEZGCiAgICAgICAg"
    "aWYgYnl0ZV9wb3MrMiA8IHRvdGFsX2J5dGVzOiBjb2Rlc19ieXRlc1tieXRlX3BvcysyXSB8PSAodmFs"
    "ID4+IDE2KSAmIDB4RkYKICAgICAgICBpZiBieXRlX3BvcyszIDwgdG90YWxfYnl0ZXM6IGNvZGVzX2J5"
    "dGVzW2J5dGVfcG9zKzNdIHw9ICh2YWwgPj4gMjQpICYgMHhGRgoKICAgICMgQ29kZWJvb2sgYnl0ZXM6"
    "IFtLMcOXMiBieXRlc11bSzLDlzIgYnl0ZXNdIHN0b3JlZCB1bnNpZ25lZCAoKzEyOCkKICAgIGNiX2J5"
    "dGVzID0gYnl0ZWFycmF5KCkKICAgIGZvciBlbnRyeSBpbiBjYjE6CiAgICAgICAgZm9yIHYgaW4gZW50"
    "cnk6IGNiX2J5dGVzLmFwcGVuZCgoaW50KHYpKzI1NikgJiAweEZGKQogICAgaWYgSzIgPiAwOgogICAg"
    "ICAgIGZvciBlbnRyeSBpbiBjYjI6CiAgICAgICAgICAgIGZvciB2IGluIGVudHJ5OiBjYl9ieXRlcy5h"
    "cHBlbmQoKGludCh2KSsyNTYpICYgMHhGRikKCiAgICByZXR1cm4gY29kZXNfYnl0ZXMsIGNiX2J5dGVz"
    "LCBzdGFydHMsIHRvdGFsX3NhbXBsZXMsIEJJVFNfVE9UQUwsIHNhbXBsZV9kcywgSzEsIEsyCgoKIyDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi"
    "lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZAKIyBHTFNMIEVNSVRURVJTCiMg4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQCgpkZWYgYnl0ZXNfdG9faW50MzJfYmVfYXJyYXko"
    "ZGF0YSwgY2h1bmtfaXZlYzQ9NTEyKToKICAgICIiIlBhY2sgYnl0ZXMgaW50byBpdmVjNCBhcnJheXMg"
    "KGJpZy1lbmRpYW46IGJ5dGUgMCA9IE1TQiBvZiBpbnQueCkuIiIiCiAgICAjIFBhZCB0byBtdWx0aXBs"
    "ZSBvZiAxNiAoc2luY2UgZWFjaCBpdmVjNCBob2xkcyAxNiBieXRlcykKICAgIHBhZGRlZCA9IGJ5dGVz"
    "KGRhdGEpICsgYidceDAwJyAqICgoMTYgLSBsZW4oZGF0YSkgJSAxNikgJSAxNikKICAgIGludHMgPSBb"
    "XQogICAgZm9yIGkgaW4gcmFuZ2UoMCwgbGVuKHBhZGRlZCksIDQpOgogICAgICAgIHYgPSBzdHJ1Y3Qu"
    "dW5wYWNrKCc+SScsIHBhZGRlZFtpOmkrNF0pWzBdCiAgICAgICAgIyBjb252ZXJ0IHRvIHNpZ25lZCBp"
    "bnQzMiBmb3IgR0xTTCAoaGFuZGxlcyB2YWx1ZXMgPj0gMl4zMSkKICAgICAgICBpZiB2ID49ICgxIDw8"
    "IDMxKToKICAgICAgICAgICAgdiAtPSAoMSA8PCAzMikKICAgICAgICBpbnRzLmFwcGVuZCh2KQogICAg"
    "IyBTcGxpdCBpbnRvIGl2ZWM0IGFycmF5IGNodW5rcyBvZiBjaHVua19pdmVjNCBpdmVjNCBlbnRyaWVz"
    "IGVhY2gKICAgIGNodW5rcyA9IFtdCiAgICBjdXIgPSBbXQogICAgZm9yIGkgaW4gcmFuZ2UoMCwgbGVu"
    "KGludHMpLCA0KToKICAgICAgICBjdXIuYXBwZW5kKHR1cGxlKGludHNbaTppKzRdKSkKICAgICAgICBp"
    "ZiBsZW4oY3VyKSA9PSBjaHVua19pdmVjNDoKICAgICAgICAgICAgY2h1bmtzLmFwcGVuZChjdXIpCiAg"
    "ICAgICAgICAgIGN1ciA9IFtdCiAgICBpZiBjdXI6CiAgICAgICAgY2h1bmtzLmFwcGVuZChjdXIpCiAg"
    "ICByZXR1cm4gY2h1bmtzCgpkZWYgZW1pdF9pdmVjNF9hcnJheShuYW1lLCBjaHVua3Nfb3Jfc2luZ2xl"
    "LCBpdGVtc19wZXJfbGluZT0yKToKICAgICIiIkVtaXQgb25lIG9yIG1vcmUgY29uc3QgaXZlYzQgYXJy"
    "YXlzLiBgY2h1bmtzX29yX3NpbmdsZWAgaXMgYSBsaXN0IG9mIGNodW5rcy4iIiIKICAgIG91dCA9IFtd"
    "CiAgICBmb3IgY2ksIGNodW5rIGluIGVudW1lcmF0ZShjaHVua3Nfb3Jfc2luZ2xlKToKICAgICAgICBh"
    "cnJfbmFtZSA9IGYie25hbWV9e2NpfSIgaWYgbGVuKGNodW5rc19vcl9zaW5nbGUpID4gMSBlbHNlIGYi"
    "e25hbWV9MCIKICAgICAgICBvdXQuYXBwZW5kKGYiY29uc3QgaXZlYzQge2Fycl9uYW1lfVt7bGVuKGNo"
    "dW5rKX1dID0gaXZlYzRbXSgiKQogICAgICAgIGxpbmVzID0gW10KICAgICAgICBmb3Igcm93X3N0YXJ0"
    "IGluIHJhbmdlKDAsIGxlbihjaHVuayksIGl0ZW1zX3Blcl9saW5lKToKICAgICAgICAgICAgcm93ID0g"
    "Y2h1bmtbcm93X3N0YXJ0OnJvd19zdGFydCArIGl0ZW1zX3Blcl9saW5lXQogICAgICAgICAgICBwYXJ0"
    "cyA9IFsiaXZlYzQoe30se30se30se30pIi5mb3JtYXQoKnQpIGZvciB0IGluIHJvd10KICAgICAgICAg"
    "ICAgbGluZXMuYXBwZW5kKCIgICAgIiArICIsICIuam9pbihwYXJ0cykpCiAgICAgICAgb3V0LmFwcGVu"
    "ZCgiLFxuIi5qb2luKGxpbmVzKSkKICAgICAgICBvdXQuYXBwZW5kKCIpO1xuIikKICAgIHJldHVybiAi"
    "XG4iLmpvaW4ob3V0KQoKCiMg4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQCiMgTUFJ"
    "TiBCVUlMRAojIOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkAoKZGVmIG1haW4obW9k"
    "X3BhdGgsIG91dF9wYXRoLCBLPTI1Niwgd2VpZ2h0ZWQ9VHJ1ZSwgZG93bnNhbXBsZT0yLCBiaXRyYXRl"
    "PSdoaScsIHZlY19kaW09MiwgcmVzYW1wbGVyPSdic3BsaW5lJywgbm9fcnZxMj1GYWxzZSk6CiAgICAi"
    "IiJHZW5lcmF0ZSBTaGFkZXJUb3kgQ29tbW9uIEdMU0wgZm9yIGEgTU9EIGZpbGUuCiAgICBkb3duc2Ft"
    "cGxlOiBhbnRpLWFsaWFzIGRvd25zYW1wbGUgZmFjdG9yIGZvciBzYW1wbGUgZW5jb2RpbmcgKDE9b2Zm"
    "LCAyPXJlY29tbWVuZGVkKS4KICAgICAgICAgICAgICAgIEhpZ2hlciBkb3duc2FtcGxlIOKGkiBzbWFs"
    "bGVyIGRhdGEsIGxhcmdlciBjb2RlYm9va3MsIGJldHRlciBTTlIuCiAgICAgICAgICAgICAgICAgIMOX"
    "MTogSzE9NjQsICBLMj0zMiAgKH43OSBLQiwgIDI3LjcgZEIpCiAgICAgICAgICAgICAgICAgIMOXMjog"
    "SzE9NTEyLCBLMj0yNTYgKH43NyBLQiwgIDM4LjQgZEIpICDihpAgcmVjb21tZW5kZWQKICAgICAgICAg"
    "ICAgICAgICAgw5c0OiBLMT01MTIsIEsyPTI1NiAofjM5IEtCLCAgMzcuMSBkQikKICAgICIiIgogICAg"
    "bW9kID0gTU9ERmlsZShtb2RfcGF0aCkKICAgIHByaW50KGYi8J+TpiBMb2FkZWQ6IHttb2QudGl0bGV9"
    "IikKICAgIHByaW50KGYiICAgUGF0dGVybnM6IHttb2QubnVtX3BhdHRlcm5zfSwgU29uZyBsZW5ndGg6"
    "IHttb2Quc29uZ19sZW5ndGh9IikKCiAgICAjIFBhdHRlcm4gY3J1bmNoCiAgICBwID0gZW5jb2RlX3Bh"
    "dHRlcm5zKG1vZCkKICAgIHByaW50KGYiXG7wn5ec77iPICBQQVRURVJOIENSVU5DSCIpCiAgICBwcmlu"
    "dChmIiAgIFRvdGFsIG5vdGVzOiAgICAgICB7cFsndG90YWxfbm90ZXMnXX0iKQogICAgcHJpbnQoZiIg"
    "ICBVbmlxdWUgbm9uLWVtcHR5OiAge2xlbihwWyd1bmlxJ10pfSIpCiAgICBwcmludChmIiAgIERpY3Rp"
    "b25hcnk6ICAgICAgICB7bGVuKHBbJ3VuaXEnXSkqNH0gYnl0ZXMiKQogICAgcHJpbnQoZiIgICBCaXRt"
    "YXA6ICAgICAgICAgICAge2xlbihwWydiaXRtYXAnXSl9IGJ5dGVzIikKICAgIHByaW50KGYiICAgSW5k"
    "ZXggc3RyZWFtOiAgICAgIHtsZW4ocFsnaWR4X3N0cmVhbSddKX0gYnl0ZXMiKQogICAgcHJpbnQoZiIg"
    "ICBSb3cgc2VlayAoMTYtYml0IHByZWZpeCk6IHtsZW4ocFsncm93X3NlZWtfYnl0ZXMnXSl9IGJ5dGVz"
    "IikKICAgIHBhdHRlcm5fdG90YWwgPSBsZW4ocFsndW5pcSddKSo0ICsgbGVuKHBbJ2JpdG1hcCddKSAr"
    "IGxlbihwWydpZHhfc3RyZWFtJ10pICsgbGVuKHBbJ3Jvd19zZWVrX2J5dGVzJ10pCiAgICBwcmludChm"
    "IiAgIOKGkiBQYXR0ZXJuIHRvdGFsOiAgIHtwYXR0ZXJuX3RvdGFsOix9IGJ5dGVzIikKCiAgICAjIFNw"
    "ZWVkL3RpY2sgdGFibGUgZnJvbSBGeHggZWZmZWN0cwogICAgcm93U3BlZWQsIHJvd1N0YXJ0VGljaywg"
    "YnBtX2NoYW5nZXMgPSBjb21wdXRlX3Jvd19zcGVlZF90YWJsZShtb2QpCiAgICBwcmludChmIlxu4o+x"
    "77iPICBTUEVFRCBUQUJMRSIpCiAgICBwcmludChmIiAgIFNvbmcgcm93czoge2xlbihyb3dTcGVlZCl9"
    "LCB0b3RhbCB0aWNrczoge3Jvd1N0YXJ0VGlja1stMV19IikKICAgIHByaW50KGYiICAgVW5pcXVlIHNw"
    "ZWVkczoge3NvcnRlZChzZXQocm93U3BlZWQpKX0iKQogICAgc3BlZWRfdGFibGVfYnl0ZXMgPSBsZW4o"
    "cm93U3RhcnRUaWNrKSAqIDIKICAgIHByaW50KGYiICAgcm93U3RhcnRUaWNrOiB7c3BlZWRfdGFibGVf"
    "Ynl0ZXN9IGJ5dGVzICgxNi1iaXQgcGFja2VkKSIpCiAgICBpZiBicG1fY2hhbmdlczoKICAgICAgICBw"
    "cmludChmIiAgIOKaoO+4jyAgQlBNIGNoYW5nZXMgZGV0ZWN0ZWQgKDEyVEguTU9EIGhhcyBub25lLCBi"
    "dXQgb3RoZXIgTU9EcyBtaWdodCkiKQoKICAgICMgQXV0by1zZWxlY3QgZG93bnNhbXBsZSBpZiBub3Qg"
    "ZXhwbGljaXRseSBvdmVycmlkZGVuIChkb3duc2FtcGxlPTIgaXMgZGVmYXVsdCkKICAgICMgQnVkZ2V0"
    "IGVzdGltYXRlOiB0b3RhbF9yYXdfYnl0ZXMgLyBkb3duc2FtcGxlICogMTcvMTYgKDE3LWJpdCBjb2Rl"
    "cywgMiBieXRlcy9zYW1wbGUpCiAgICAjIFNoYWRlclRveSBzYWZlIHpvbmU6IOKJpCA4MCBLQiBzYW1w"
    "bGUgY29kZXMgKyBwYXR0ZXJuIGRhdGEKICAgIGltcG9ydCBudW1weSBhcyBucAogICAgdG90YWxfcmF3"
    "ID0gc3VtKHNbJ2xlbmd0aCddIGZvciBzIGluIG1vZC5zYW1wbGVzX2luZm8pCiAgICAjIE5PVEU6IHVz"
    "ZXItcmVxdWVzdGVkIGRvd25zYW1wbGUgaXMgcmVzcGVjdGVkIGFzIGEgSEFSRCBDQVAg4oCUIG5vIGF1"
    "dG8tYnVtcC4KICAgICMgVXNlIC0tdmVjLWRpbSA0IG9yIC0tYml0cmF0ZSBsbyBpZiB5b3UgbmVlZCBt"
    "b3JlIHNpemUgcmVkdWN0aW9uLgogICAgZXN0aW1hdGVkX2J1ZGdldF9kczIgPSAodG90YWxfcmF3IC8v"
    "IGRvd25zYW1wbGUpICogMTcgLy8gMTYgKyAxNjAwMCAgIyBmb3IgbG9nIG9ubHkKCiAgICAjIFNhbXBs"
    "ZSBlbmNvZGluZzogUlZRIHdpdGggYml0cmF0ZS1jb250cm9sbGVkIGNvZGVib29rICsgcGVyLXNhbXBs"
    "ZSBEUwogICAgZHNfbGFiZWwgPSBmIsOXe2Rvd25zYW1wbGV9IiBpZiBkb3duc2FtcGxlID4gMSBlbHNl"
    "ICJmdWxsLXJlcyIKICAgIHByaW50KGYiXG7wn5ec77iPICBTQU1QTEUgQ1JVTkNIIChSVlEge2RzX2xh"
    "YmVsfSBiaXRyYXRlPXtiaXRyYXRlfSwgcmluZy13ZWlnaHRlZCkiKQogICAgY29kZXNfYnl0ZXMsIGNi"
    "X2J5dGVzLCBzdGFydHMsIHRvdGFsX3NhbXBsZXMsIGJpdHNfcGVyX2NvZGUsIHNhbXBsZV9kcywgSzEs"
    "IEsyID0gZW5jb2RlX3NhbXBsZXNfdnEyZCgKICAgICAgICBtb2QsIEssIHdlaWdodGVkLCBkb3duc2Ft"
    "cGxlPWRvd25zYW1wbGUsIGJpdHJhdGU9Yml0cmF0ZSwgdmVjX2RpbT12ZWNfZGltLCBub19ydnEyPW5v"
    "X3J2cTIpCiAgICBCSVRTMSA9IGludChucC5jZWlsKG5wLmxvZzIoSzEpKSkKICAgIEJJVFMyID0gaW50"
    "KG5wLmNlaWwobnAubG9nMihLMikpKSBpZiBLMiA+IDAgZWxzZSAwCiAgICBCSVRTX1RPVEFMID0gYml0"
    "c19wZXJfY29kZQogICAgcHJpbnQoZiIgICBMb2dpY2FsIHNhbXBsZXM6ICAge3RvdGFsX3NhbXBsZXM6"
    "LH0gICh7ZHNfbGFiZWx9KSIpCiAgICBwcmludChmIiAgIENvZGVzIHBhY2tlZDogICAgICB7bGVuKGNv"
    "ZGVzX2J5dGVzKTosfSBieXRlcyAgKHtiaXRzX3Blcl9jb2RlfSBiaXRzL3ZlY3RvciDDlyB7dG90YWxf"
    "c2FtcGxlcy8vMn0gdmVjdG9ycykiKQogICAgcHJpbnQoZiIgICBDb2RlYm9va3M6ICAgICAgICAge2xl"
    "bihjYl9ieXRlcyk6LH0gYnl0ZXMgICh7SzF9w5cyICsge0syfcOXMiBieXRlcykiKQoKICAgIHRvdGFs"
    "X2J1ZGdldCA9IHBhdHRlcm5fdG90YWwgKyBsZW4oY29kZXNfYnl0ZXMpICsgbGVuKGNiX2J5dGVzKSAr"
    "IDMxKjI0ICsgc3BlZWRfdGFibGVfYnl0ZXMKICAgIHByaW50KGYiXG7wn5OKIFRPVEFMIGNvbnN0IGRh"
    "dGEgYnVkZ2V0OiB+e3RvdGFsX2J1ZGdldDosfSBieXRlcyAgKHt0b3RhbF9idWRnZXQvMTAyNDouMWZ9"
    "IEtCKSIpCgogICAgIyBDaHVuayBmb3IgR0xTTAogICAgZGljdF9ieXRlcyA9IGInJy5qb2luKHBbJ3Vu"
    "aXEnXSkKICAgIGRpY3RfY2h1bmtzICAgID0gYnl0ZXNfdG9faW50MzJfYmVfYXJyYXkoZGljdF9ieXRl"
    "cykKICAgIGJpdG1hcF9jaHVua3MgID0gYnl0ZXNfdG9faW50MzJfYmVfYXJyYXkoYnl0ZXMocFsnYml0"
    "bWFwJ10pKQogICAgaWR4X2NodW5rcyAgICAgPSBieXRlc190b19pbnQzMl9iZV9hcnJheShieXRlcyhw"
    "WydpZHhfc3RyZWFtJ10pKQogICAgcm93c2Vla19jaHVua3MgPSBieXRlc190b19pbnQzMl9iZV9hcnJh"
    "eShieXRlcyhwWydyb3dfc2Vla19ieXRlcyddKSkKICAgIGNvZGVzX2NodW5rcyAgID0gYnl0ZXNfdG9f"
    "aW50MzJfYmVfYXJyYXkoYnl0ZXMoY29kZXNfYnl0ZXMpKQogICAgY2JfY2h1bmtzICAgICAgPSBieXRl"
    "c190b19pbnQzMl9iZV9hcnJheShieXRlcyhjYl9ieXRlcykpCgogICAgIyBQYWNrIHJvd1N0YXJ0VGlj"
    "ayBhcyAxNi1iaXQgTEUgYnl0ZXMg4oaSIGl2ZWM0IGNodW5rcwogICAgdGlja19ieXRlcyA9IGJ5dGVh"
    "cnJheSgpCiAgICBmb3IgdCBpbiByb3dTdGFydFRpY2s6CiAgICAgICAgdGlja19ieXRlcy5hcHBlbmQo"
    "dCAmIDB4RkYpCiAgICAgICAgdGlja19ieXRlcy5hcHBlbmQoKHQgPj4gOCkgJiAweEZGKQogICAgdGlj"
    "a19jaHVua3MgPSBieXRlc190b19pbnQzMl9iZV9hcnJheShieXRlcyh0aWNrX2J5dGVzKSkKCiAgICBz"
    "YW1wbGVzX2luZm9fbmV3ID0gW10KICAgIGZvciBpLCAocywgc3QpIGluIGVudW1lcmF0ZSh6aXAobW9k"
    "LnNhbXBsZXNfaW5mbywgc3RhcnRzKSk6CiAgICAgICAgIyBVc2UgcGVyLXNhbXBsZSBhY3R1YWwgRFMg"
    "Zm9yIGxlbmd0aC9sb29wIHNjYWxpbmcKICAgICAgICBzZHMgPSBzYW1wbGVfZHNbaV0gaWYgaSA8IGxl"
    "bihzYW1wbGVfZHMpIGVsc2UgZG93bnNhbXBsZQogICAgICAgIHNhbXBsZXNfaW5mb19uZXcuYXBwZW5k"
    "KGRpY3QoCiAgICAgICAgICAgIHN0YXJ0PXN0LAogICAgICAgICAgICBsZW5ndGg9c1snbGVuZ3RoJ10g"
    "Ly8gc2RzLAogICAgICAgICAgICBsb29wU3RhcnQ9c1snbG9vcF9zdGFydCddIC8vIHNkcywKICAgICAg"
    "ICAgICAgbG9vcExlbj1zWydsb29wX2xlbiddIC8vIHNkcywKICAgICAgICAgICAgdm9sdW1lPXNbJ3Zv"
    "bHVtZSddLCBmaW5ldHVuZT1zWydmaW5ldHVuZSddLAogICAgICAgICAgICBid0ZhY3Rvcj1zZHMsICAg"
    "IyBhY3R1YWwgRFMg4oCUIHVzZWQgYnkgR0xTTCBhcyBmcmVxIGRpdmlzb3IKICAgICAgICApKQoKICAg"
    "IGdsc2wgPSBidWlsZF9nbHNsKG1vZCwgcCwgY29kZXNfYnl0ZXMsIHN0YXJ0cywgdG90YWxfc2FtcGxl"
    "cywKICAgICAgICAgICAgICAgICAgICAgZGljdF9jaHVua3MsIGJpdG1hcF9jaHVua3MsIGlkeF9jaHVu"
    "a3MsIHJvd3NlZWtfY2h1bmtzLAogICAgICAgICAgICAgICAgICAgICBjb2Rlc19jaHVua3MsIGNiX2No"
    "dW5rcywgc2FtcGxlc19pbmZvX25ldywgSywgYml0c19wZXJfY29kZSwKICAgICAgICAgICAgICAgICAg"
    "ICAgdGlja19jaHVua3MsIHJvd1N0YXJ0VGljaywKICAgICAgICAgICAgICAgICAgICAgSzE9SzEsIEsy"
    "PUsyLCBCSVRTMT1CSVRTMSwgQklUUzI9QklUUzIsIEJJVFNfVE9UQUw9QklUU19UT1RBTCwKICAgICAg"
    "ICAgICAgICAgICAgICAgZG93bnNhbXBsZT1kb3duc2FtcGxlLCB2ZWNfZGltPXZlY19kaW0sIHJlc2Ft"
    "cGxlcj1yZXNhbXBsZXIsIG5vX3J2cTI9bm9fcnZxMikKICAgIHdpdGggb3BlbihvdXRfcGF0aCwgJ3cn"
    "KSBhcyBmOgogICAgICAgIGYud3JpdGUoZ2xzbCkKICAgIHByaW50KGYiXG7inIUgV3JvdGU6IHtvdXRf"
    "cGF0aH0gICh7bGVuKGdsc2wuZW5jb2RlKCd1dGYtOCcpKTosfSBieXRlcykiKQoKCmRlZiBidWlsZF9n"
    "bHNsKG1vZCwgcCwgcGFja2VkLCBzdGFydHMsIHRvdGFsX3NhbXBsZXMsCiAgICAgICAgICAgICAgIGRp"
    "Y3RfY2h1bmtzLCBiaXRtYXBfY2h1bmtzLCBpZHhfY2h1bmtzLCByb3dzZWVrX2NodW5rcywKICAgICAg"
    "ICAgICAgICAgY29kZXNfY2h1bmtzLCBjYl9jaHVua3MsIHNhbXBsZXNfaW5mb19uZXcsIEssIGJpdHNf"
    "cGVyX2NvZGUsCiAgICAgICAgICAgICAgIHRpY2tfY2h1bmtzLCByb3dTdGFydFRpY2ssIEsxPTUxMiwg"
    "SzI9MjU2LCBCSVRTMT05LCBCSVRTMj04LCBCSVRTX1RPVEFMPTE3LCBkb3duc2FtcGxlPTIsIHZlY19k"
    "aW09MiwgcmVzYW1wbGVyPSdic3BsaW5lJywgbm9fcnZxMj1GYWxzZSk6CgogICAgIyDilIDilIAgU29u"
    "ZyBtZXRhZGF0YQogICAgc29uZ19wb3NpdGlvbnMgPSBtb2QucGF0dGVybl9vcmRlcls6bW9kLnNvbmdf"
    "bGVuZ3RoXQoKICAgICMgQ29tcHV0ZSBhY3R1YWwgcm93cyBwZXIgc29uZyBwb3NpdGlvbiDigJQgUHJv"
    "VHJhY2tlciBEeHggKHBhdHRlcm4gYnJlYWspCiAgICAjIGFuZCBCeHggKHBvc2l0aW9uIGp1bXApIHNo"
    "b3J0ZW4gdGhlIGVmZmVjdGl2ZSBwYXR0ZXJuIGxlbmd0aC4KICAgIGRlZiBhY3R1YWxfcGF0dGVybl9y"
    "b3dzKHNwKToKICAgICAgICBwYXQgPSBtb2QucGF0dGVybl9vcmRlcltzcF0KICAgICAgICBOQ19sb2Nh"
    "bCA9IG1vZC5udW1fY2hhbm5lbHMKICAgICAgICBwYXRfc2l6ZSA9IDY0ICogTkNfbG9jYWwgKiA0CiAg"
    "ICAgICAgZm9yIHJvdyBpbiByYW5nZSg2NCk6CiAgICAgICAgICAgIGZvciBjaCBpbiByYW5nZShOQ19s"
    "b2NhbCk6CiAgICAgICAgICAgICAgICBiYXNlID0gMTA4NCArIHBhdCpwYXRfc2l6ZSArIHJvdypOQ19s"
    "b2NhbCo0ICsgY2gqNAogICAgICAgICAgICAgICAgbmIgPSBtb2QuZGF0YVtiYXNlOmJhc2UrNF0KICAg"
    "ICAgICAgICAgICAgIGVmZiA9IG5iWzJdICYgMHhGCiAgICAgICAgICAgICAgICBpZiBlZmYgPT0gMHhE"
    "IG9yIGVmZiA9PSAweEI6ICAgIyBwYXR0ZXJuIGJyZWFrIG9yIHBvc2l0aW9uIGp1bXAKICAgICAgICAg"
    "ICAgICAgICAgICByZXR1cm4gcm93ICsgMQogICAgICAgIHJldHVybiA2NAoKICAgIHBhdF9yb3dzID0g"
    "W2FjdHVhbF9wYXR0ZXJuX3Jvd3Moc3ApIGZvciBzcCBpbiByYW5nZShtb2Quc29uZ19sZW5ndGgpXQog"
    "ICAgcGF0X3Jvd19vZmZzZXQgPSBbMF0KICAgIGZvciByIGluIHBhdF9yb3dzOgogICAgICAgIHBhdF9y"
    "b3dfb2Zmc2V0LmFwcGVuZChwYXRfcm93X29mZnNldFstMV0gKyByKQogICAgcGF0X3N0YXJ0X3JvdyAg"
    "PSBbMF0qbW9kLnNvbmdfbGVuZ3RoCgogICAgIyBwYXRUaWNrT2Zmc2V0W3NwXSA9IGluZGV4IGludG8g"
    "cm93U3RhcnRUaWNrIGZvciByb3cgMCBvZiBzb25nIHBvc2l0aW9uIHNwCiAgICAjIFNhbWUgYXMgcGF0"
    "X3Jvd19vZmZzZXQgc2luY2UgdGljayB0YWJsZSByb3dzID09IHNvbmcgcm93cyBhZnRlciBEMDAgZml4"
    "LgogICAgcGF0X3RpY2tfb2Zmc2V0ID0gcGF0X3Jvd19vZmZzZXRbOl0KCiAgICB0b3RhbF9zb25nX3Jv"
    "d3MgPSBtb2Quc29uZ19sZW5ndGggKiA2NAogICAgbnVtX3BhdHRlcm5zID0gbW9kLm51bV9wYXR0ZXJu"
    "cwoKICAgICMg4pSA4pSAIFNhbXBsZUluZm8gZW1pc3Npb24gKHVzZSBuZXcgYHN0YXJ0YCA9IHNhbXBs"
    "ZSBpbmRleCBpbiB0aGUgY29uY2F0ZW5hdGVkIHN0cmVhbSkKICAgIGRlZiBmbXRfc2FtcGxlaW5mbyhz"
    "KToKICAgICAgICByZXR1cm4gZiJTYW1wbGVJbmZvKHtzWydzdGFydCddfSwge3NbJ2xlbmd0aCddfSwg"
    "e3NbJ2xvb3BTdGFydCddfSwge3NbJ2xvb3BMZW4nXX0sIHtzWyd2b2x1bWUnXX0sIHtzLmdldCgnYndG"
    "YWN0b3InLDEpfSwge3MuZ2V0KCdmaW5ldHVuZScsMCl9KSIKICAgIHNpX2xpbmVzID0gW10KICAgIGZv"
    "ciBpLCBzIGluIGVudW1lcmF0ZShzYW1wbGVzX2luZm9fbmV3KToKICAgICAgICBzaV9saW5lcy5hcHBl"
    "bmQoZiIgICAge2ZtdF9zYW1wbGVpbmZvKHMpfXsnLCcgaWYgaTwzMCBlbHNlICcnfSIpCiAgICBzYW1w"
    "bGVzX2luZm9fZ2xzbCA9ICJjb25zdCBTYW1wbGVJbmZvIHNhbXBsZXNbMzFdID0gU2FtcGxlSW5mb1td"
    "KFxuIiArICJcbiIuam9pbihzaV9saW5lcykgKyAiXG4pOyIKCiAgICAjIOKUgOKUgCBjaGFubmVsUGFu"
    "IChzYW1lIGFzIGV4aXN0aW5nOiBBbWlnYSBMUlJMIHdpdGggcmVzdCBjZW50ZXJlZCkKICAgIGNoYW5f"
    "cGFuID0gWzAuMCwgMS4wLCAxLjAsIDAuMF0gKyBbMC41XSoyOAoKICAgICMg4pSA4pSAIENodW5rIGFy"
    "cmF5IGRlY2xhcmF0aW9ucwogICAgZGljdF9sZW4gICAgPSBzdW0obGVuKGMpIGZvciBjIGluIGRpY3Rf"
    "Y2h1bmtzKQogICAgYml0bWFwX2xlbiAgPSBzdW0obGVuKGMpIGZvciBjIGluIGJpdG1hcF9jaHVua3Mp"
    "CiAgICBpZHhfbGVuICAgICA9IHN1bShsZW4oYykgZm9yIGMgaW4gaWR4X2NodW5rcykKICAgIHJvd3Nl"
    "ZWtfbGVuID0gc3VtKGxlbihjKSBmb3IgYyBpbiByb3dzZWVrX2NodW5rcykKICAgIGNvZGVzX2xlbiAg"
    "ID0gc3VtKGxlbihjKSBmb3IgYyBpbiBjb2Rlc19jaHVua3MpCiAgICBjYl9sZW4gICAgICA9IHN1bShs"
    "ZW4oYykgZm9yIGMgaW4gY2JfY2h1bmtzKQoKICAgIGhlYWRlciA9IGYiIiIvKiA9PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09CiAgIEdMU0wgTU9EIFBsYXllciB2MS4zNSAoYykgMjAyNiBPcmJsaXZpdXMKICAgM0QgU3Vycm91"
    "bmQsIFBoYXRCYXNzLCBDb21iIFJldmVyYiwgRkFULCBSVlEgc2FtcGxlIGNvbXByZXNzaW9uLCBjb25m"
    "aWd1cmFibGUgcmVzYW1wbGVyCiAgIENvbnRhY3Q6IHN1YmJhbmRAZ21haWwuY29tIG9yCiAgICAgICAg"
    "ICAgIHN1YmJhbmRAcHJvdG9ubWFpbC5jb20KICAgR0lUOiAgICAgaHR0cHM6Ly9naXRodWIuY29tL21l"
    "d3phL21vZDJnbHNsCiAgIENPTU1PTiBUQUIKICAgR2VuZXJhdGVkIGZyb206IHttb2QudGl0bGV9CiAg"
    "IAogICBDb21wcmVzc2lvbjoKICAgICDigKIgUGF0dGVybnM6IGJpdG1hcCArIGRpY3Rpb25hcnkgKyAx"
    "Ni1iaXQgcHJlZml4LXN1bSByb3cgc2VlayAoTygxKSkKICAgICDigKIgU2FtcGxlczogIDItc3RhZ2Ug"
    "UlZRIMOXe2Rvd25zYW1wbGV9IEFBLWRvd25zYW1wbGVkIChLMT17SzF9LCBLMj17SzJ9KSwge0JJVFNf"
    "VE9UQUx9IGJpdHMvcGFpcgogICAgICAgICAgICAgICAgIHJpbmctd2VpZ2h0ZWQgay1tZWFucyB0cmFp"
    "bmVkIG9uIHRoaXMgTU9EJ3MgY29udGVudAogICA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09ICovCgojZGVmaW5lIFVT"
    "RV9FTUJFRERFRF9EQVRBIDEKI2RlZmluZSBOVU1fUEFUVEVSTlMgICAgICB7bnVtX3BhdHRlcm5zfQoj"
    "ZGVmaW5lIFNPTkdfTEVOR1RIICAgICAgIHttb2Quc29uZ19sZW5ndGh9CiNkZWZpbmUgU09OR19MT09Q"
    "X1BPUyAgICAgMAojZGVmaW5lIE5VTV9DSEFOTkVMUyAgICAgIHttb2QubnVtX2NoYW5uZWxzfQojZGVm"
    "aW5lIEJQTSAgICAgICAgICAgICAgIDEyNS4wCiNkZWZpbmUgU1BFRUQgICAgICAgICAgICAgNi4wCiNk"
    "ZWZpbmUgVE9UQUxfU09OR19ST1dTICAge3RvdGFsX3Nvbmdfcm93c30KCi8vIOKUgOKUgCBQYXR0ZXJu"
    "IGNydW5jaCBjb25zdGFudHMg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACiNkZWZpbmUgVE9UQUxfTk9URVMgICAg"
    "ICAge3BbJ3RvdGFsX25vdGVzJ119CiNkZWZpbmUgVE9UQUxfUk9XUyAgICAgICAge3BbJ251bV9yb3dz"
    "J119CiNkZWZpbmUgRElDVF9OT1RFUyAgICAgICAge2xlbihwWyd1bmlxJ10pfQojZGVmaW5lIElEWF9C"
    "WVRFU19QRVIgICAgIHtwWydpZHhfYnl0ZXMnXX0KI2RlZmluZSBESUNUX0lOVFMgICAgICAgICB7ZGlj"
    "dF9sZW59CiNkZWZpbmUgQklUTUFQX0lOVFMgICAgICAge2JpdG1hcF9sZW59CiNkZWZpbmUgSURYX0lO"
    "VFMgICAgICAgICAge2lkeF9sZW59CiNkZWZpbmUgUk9XU0VFS19JTlRTICAgICAge3Jvd3NlZWtfbGVu"
    "fQoKLy8g4pSA4pSAIFJWUSBzYW1wbGUgY29uc3RhbnRzIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgAovLyBTYW1wbGVzIGFyZSBhbnRpLWFsaWFzIGRvd25zYW1wbGVkIHBlci1zYW1wbGUgKERT"
    "PTEgZm9yIEhGIHBlcmN1c3Npb24sCi8vIERTPXtkb3duc2FtcGxlfSBmb3IgbWVsb2RpYykuIFBlci1z"
    "YW1wbGUgRFMgaXMgc3RvcmVkIGluIFNhbXBsZUluZm8uYndGYWN0b3IuCi8vIHBlcmlvZFRvRnJlcSA9"
    "IDcwOTM3ODkuMi8ocGVyaW9kKjIpIOKAlCBid0ZhY3RvciBoYW5kbGVzIHBlci1zYW1wbGUgcGl0Y2gu"
    "CiNkZWZpbmUgUlZRX0NPREVTX0JZVEVTICAge2xlbihwYWNrZWQpfQojZGVmaW5lIFJWUV9DQl9CWVRF"
    "UyAgICAgIHtLMSoyICsgSzIqMn0KI2RlZmluZSBUT1RBTF9TQU1QTEVTICAgICB7dG90YWxfc2FtcGxl"
    "c30KCiNkZWZpbmUgQklUTUFQX0JZVEVTICAgICAge2xlbihwWydiaXRtYXAnXSl9CiNkZWZpbmUgSURY"
    "X0JZVEVTICAgICAgICAge2xlbihwWydpZHhfc3RyZWFtJ10pfQojZGVmaW5lIFJPV1NFRUtfQllURVMg"
    "ICAgIHtsZW4ocFsncm93X3NlZWtfYnl0ZXMnXSl9CgovLyDilIDilIAgRnh4LWF3YXJlIHRpbWluZyDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKI2RlZmluZSBUT1RBTF9USUNL"
    "UyAgICAgICB7cm93U3RhcnRUaWNrWy0xXX0KI2RlZmluZSBOVU1fU09OR19ST1dTICAgICB7bGVuKHJv"
    "d1N0YXJ0VGljayktMX0KI2RlZmluZSBUSUNLU19QRVJfU0VDICAgICA1MC4wICAgLy8gQlBNPTEyNSBj"
    "b25zdGFudCBmb3IgMTJUSC5NT0QKCi8vIOKUgOKUgCBBdWRpbyBlZmZlY3RzIOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgApjb25zdCBib29sICBlbmFibGUz"
    "RCAgICAgID0gdHJ1ZTsKY29uc3QgYm9vbCAgZW5hYmxlRkFUICAgICA9IHRydWU7CmNvbnN0IGl2ZWMy"
    "IHN1cnJfY2hhbm5lbHMgPSBpdmVjMigxLCA0KTsKIiIiCgogICAgIyDilIDilIAgc29uZyBtZXRhZGF0"
    "YSBhcnJheXMKICAgIGNoYW5fcGFuX3N0ciAgID0gIiwgIi5qb2luKGYie3Y6LjFmfSIgZm9yIHYgaW4g"
    "Y2hhbl9wYW4pCiAgICBzb25ncG9zX3N0ciAgICA9ICIsICIuam9pbihzdHIoeCkgZm9yIHggaW4gc29u"
    "Z19wb3NpdGlvbnMpCiAgICByb3dvZmZfc3RyICAgICA9ICIsICIuam9pbihzdHIoeCkgZm9yIHggaW4g"
    "cGF0X3Jvd19vZmZzZXQpCiAgICBzdGFydHJvd19zdHIgICA9ICIsICIuam9pbihzdHIoeCkgZm9yIHgg"
    "aW4gcGF0X3N0YXJ0X3JvdykKICAgIHRpY2tvZmZfc3RyICAgID0gIiwgIi5qb2luKHN0cih4KSBmb3Ig"
    "eCBpbiBwYXRfdGlja19vZmZzZXRbOi0xXSkgICMgbGVuZ3RoID0gc29uZ19sZW5ndGgKCiAgICBtZXRh"
    "ID0gZiIiIgpjb25zdCBmbG9hdCBjaGFubmVsUGFuWzMyXSA9IGZsb2F0W10oe2NoYW5fcGFuX3N0cn0p"
    "Owpjb25zdCBpbnQgICBzb25nUG9zaXRpb25zW3ttb2Quc29uZ19sZW5ndGh9XSAgID0gaW50W10oe3Nv"
    "bmdwb3Nfc3RyfSk7CmNvbnN0IGludCAgIHBhdFJvd09mZnNldFt7bW9kLnNvbmdfbGVuZ3RoKzF9XSAg"
    "ICA9IGludFtdKHtyb3dvZmZfc3RyfSk7CmNvbnN0IGludCAgIHBhdFN0YXJ0Um93W3ttb2Quc29uZ19s"
    "ZW5ndGh9XSAgICAgPSBpbnRbXSh7c3RhcnRyb3dfc3RyfSk7CmNvbnN0IGludCAgIHBhdFRpY2tPZmZz"
    "ZXRbe21vZC5zb25nX2xlbmd0aH1dICAgPSBpbnRbXSh7dGlja29mZl9zdHJ9KTsKIiIiCgogICAgIyDi"
    "lIDilIAgRGF0YSBhcnJheXMgKGl2ZWM0IGNodW5rcykKICAgIGRhdGFfYXJyYXlzID0gWyJcbi8vIOKU"
    "gOKUgCBQYXR0ZXJuIGRpY3Rpb25hcnkgKHVuaXF1ZSA0LWJ5dGUgbm90ZXMsIE1TQi1maXJzdCBwZXIg"
    "aW50KSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbiJdCiAgICBkYXRhX2FycmF5cy5h"
    "cHBlbmQoZW1pdF9pdmVjNF9hcnJheSgicGF0RGljdCIsIGRpY3RfY2h1bmtzKSkKICAgIGRhdGFfYXJy"
    "YXlzLmFwcGVuZCgiXG4vLyDilIDilIAgUGF0dGVybiBiaXRtYXAgKDEgYml0L25vdGUsIExTQi1maXJz"
    "dCB3aXRoaW4gYnl0ZSkg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSAXG4iKQogICAgZGF0YV9hcnJheXMuYXBwZW5kKGVtaXRfaXZlYzRfYXJyYXko"
    "InBhdEJpdG1hcCIsIGJpdG1hcF9jaHVua3MpKQogICAgZGF0YV9hcnJheXMuYXBwZW5kKCJcbi8vIOKU"
    "gOKUgCBJbmRleCBzdHJlYW0gKCVzIGJ5dGVzIHBlciBub24tZW1wdHkgbm90ZSkg4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSAXG4iICUgcFsnaWR4X2J5dGVzJ10pCiAgICBkYXRhX2FycmF5cy5hcHBlbmQoZW1pdF9pdmVj"
    "NF9hcnJheSgicGF0SWR4IiwgaWR4X2NodW5rcykpCiAgICBkYXRhX2FycmF5cy5hcHBlbmQoIlxuLy8g"
    "4pSA4pSAIFJvdyBzZWVrIHRhYmxlICgxNi1iaXQgTEUgcHJlZml4IHN1bXMsIE8oMSkgbG9va3VwKSDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbiIpCiAg"
    "ICBkYXRhX2FycmF5cy5hcHBlbmQoZW1pdF9pdmVjNF9hcnJheSgicGF0Um93U2VlayIsIHJvd3NlZWtf"
    "Y2h1bmtzKSkKICAgIGRhdGFfYXJyYXlzLmFwcGVuZCgiXG4vLyDilIDilIAgVlEgY29kZXMgKHBhY2tl"
    "ZCBiaXQgc3RyZWFtKSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIBcbiIpCiAgICBkYXRhX2FycmF5cy5hcHBlbmQoZW1pdF9pdmVjNF9hcnJh"
    "eSgidnFDb2RlcyIsIGNvZGVzX2NodW5rcykpCiAgICBkYXRhX2FycmF5cy5hcHBlbmQoZiJcbi8vIOKU"
    "gOKUgCBWUSBjb2RlYm9vayAoe0t9IGVudHJpZXMgw5cgMiBzYW1wbGVzLCBzaWduZWQgOC1iaXQgYXMg"
    "dW5zaWduZWQpIOKUgOKUgFxuIikKICAgIGRhdGFfYXJyYXlzLmFwcGVuZChlbWl0X2l2ZWM0X2FycmF5"
    "KCJ2cUNvZGVib29rIiwgY2JfY2h1bmtzKSkKICAgIGRhdGFfYXJyYXlzLmFwcGVuZCgiXG4vLyDilIDi"
    "lIAgUGVyLXJvdyBjdW11bGF0aXZlIHRpY2sgdGFibGUgKDE2LWJpdCBMRSwgRnh4LWF3YXJlKSDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbiIpCiAgICBkYXRhX2Fy"
    "cmF5cy5hcHBlbmQoZW1pdF9pdmVjNF9hcnJheSgicm93U3RhcnRUaWNrIiwgdGlja19jaHVua3MpKQoK"
    "ICAgICMg4pSA4pSAIFNhbXBsZUluZm8gJiBwZXJpb2RUYWJsZQogICAgdGFibGVzID0gZiIiIgovLyDi"
    "lIDilIAgU2FtcGxlIG1ldGFkYXRhIChzdGFydCA9IHNhbXBsZSBpbmRleCBpbiBwYWNrZWQgMy1iaXQg"
    "c3RyZWFtKSDilIDilIDilIDilIDilIDilIDilIDilIDilIAKc3RydWN0IFNhbXBsZUluZm8ge3sKICAg"
    "IGludCBzdGFydCwgbGVuZ3RoLCBsb29wU3RhcnQsIGxvb3BMZW4sIHZvbHVtZSwgYndGYWN0b3IsIGZp"
    "bmV0dW5lOwp9fTsKe3NhbXBsZXNfaW5mb19nbHNsfQoKLy8gUHJvVHJhY2tlciBwZXJpb2QgdGFibGUg"
    "KEMtMSB0byBCLTMpCmNvbnN0IGludCBwZXJpb2RUYWJsZVszN10gPSBpbnRbXSgKICAgIDg1Niw4MDgs"
    "NzYyLDcyMCw2NzgsNjQwLDYwNCw1NzAsNTM4LDUwOCw0ODAsNDUzLAogICAgNDI4LDQwNCwzODEsMzYw"
    "LDMzOSwzMjAsMzAyLDI4NSwyNjksMjU0LDI0MCwyMjYsCiAgICAyMTQsMjAyLDE5MCwxODAsMTcwLDE2"
    "MCwxNTEsMTQzLDEzNSwxMjcsMTIwLDExMywwCik7CgovLyBQcm9UcmFja2VyIDMyLWVudHJ5IHNpbmUg"
    "dGFibGUgZm9yIHZpYnJhdG8gKExVVCwga2VwdCBnbG9iYWwgc28gaXQgZG9lc24ndAovLyBjb25zdW1l"
    "IHBlci1jYWxsIHByaXZhdGUvc3RhY2sgc3RvcmFnZSBpbiBnZXRDaGFubmVsT3V0cHV0KS4KY29uc3Qg"
    "ZmxvYXQgdmliVGFiWzMyXSA9IGZsb2F0W10oCiAgICAgIDAuMCwgIDI0LjAsICA0OS4wLCAgNzQuMCwg"
    "IDk3LjAsIDEyMC4wLCAxNDEuMCwgMTYxLjAsCiAgICAxODAuMCwgMTk3LjAsIDIxMi4wLCAyMjQuMCwg"
    "MjM1LjAsIDI0NC4wLCAyNTAuMCwgMjUzLjAsCiAgICAyNTUuMCwgMjUzLjAsIDI1MC4wLCAyNDQuMCwg"
    "MjM1LjAsIDIyNC4wLCAyMTIuMCwgMTk3LjAsCiAgICAxODAuMCwgMTYxLjAsIDE0MS4wLCAxMjAuMCwg"
    "IDk3LjAsICA3NC4wLCAgNDkuMCwgIDI0LjAKKTsKCi8vIEM0IHNwZWVkcyBmb3IgZWFjaCBmaW5ldHVu"
    "ZSB2YWx1ZSAobWlrSVQvUFQgc3BlYykuICBJbmRleCAwLi43ID0gcG9zaXRpdmUKLy8gZmluZXR1bmUg"
    "KHNsaWdodGx5IGhpZ2hlciBwaXRjaCksIGluZGV4IDguLjE1ID0gbmVnYXRpdmUgZmluZXR1bmUgKGxv"
    "d2VyKS4KLy8gSW4gc2FtcGxlIGRhdGEgd2Ugc3RvcmUgZmluZXR1bmUgYXMgYSBTSUdORUQgLTguLjcg"
    "aW50IOKAlCBjb252ZXJ0IHZpYSAmMHhGLgpjb25zdCBmbG9hdCBjNHNwZWVkc1sxNl0gPSBmbG9hdFtd"
    "KAogICAgODM2My4wLCA4NDEzLjAsIDg0NjMuMCwgODUyOS4wLCA4NTgxLjAsIDg2NTEuMCwgODcyMy4w"
    "LCA4NzU3LjAsCiAgICA3ODk1LjAsIDc5NDEuMCwgNzk4NS4wLCA4MDQ2LjAsIDgxMDcuMCwgODE2OS4w"
    "LCA4MjMyLjAsIDgyODAuMAopOwpmbG9hdCBwZXJpb2RUb0ZyZXEoaW50IHBlcmlvZCkge3sKICAgIC8v"
    "IERlZmF1bHQgKGZpbmV0dW5lPTApOiA3MDkzNzg5LjIgLyAocGVyaW9kIMOXIDIpIOKJiCAzNTQ2ODk0"
    "LjYvcGVyaW9kLiAgVXNlCiAgICAvLyBwZXJpb2RUb0ZyZXFGdCBiZWxvdyB3aGVuIGZpbmV0dW5lIG1h"
    "dHRlcnMuCiAgICByZXR1cm4gcGVyaW9kID4gMCA/IDcwOTM3ODkuMiAvIChmbG9hdChwZXJpb2QpICog"
    "Mi4wKSA6IDAuMDsKfX0KZmxvYXQgcGVyaW9kVG9GcmVxRnQoaW50IHBlcmlvZCwgaW50IGZpbmV0dW5l"
    "KSB7ewogICAgLy8gKGM0ICogNDI4KSAvIHBlcmlvZCDigJQgbWF0Y2hlcyBIVE1MJ3MgcGl0Y2ggdGFi"
    "bGUgZXhhY3RseS4KICAgIGlmIChwZXJpb2QgPD0gMCkgcmV0dXJuIDAuMDsKICAgIGludCBpZHggPSBm"
    "aW5ldHVuZSAmIDB4RjsgIC8vIC0xIChzaWduZWQpIOKGkiAweEYsIGV0Yy4KICAgIHJldHVybiAoYzRz"
    "cGVlZHNbaWR4XSAqIDQyOC4wKSAvIGZsb2F0KHBlcmlvZCk7Cn19CiIiIgoKICAgICMg4pSA4pSAIEZl"
    "dGNoIGhlbHBlcnMgKGNodW5rIGRpc3BhdGNoZXJzIGZvciBlYWNoIGFycmF5KQogICAgZGVmIGNodW5r"
    "X2Rpc3BhdGNoKG5hbWUsIG51bV9jaHVua3MsIHZhcj0naScpOgogICAgICAgIGlmIG51bV9jaHVua3Mg"
    "PT0gMToKICAgICAgICAgICAgcmV0dXJuIGYiICAgIHJldHVybiB7bmFtZX0wW3t2YXJ9Pj4yXTsiCiAg"
    "ICAgICAgbGluZXMgPSBbZiIgICAgaXZlYzQgdiA9IGl2ZWM0KDApOyJdCiAgICAgICAgbGluZXMuYXBw"
    "ZW5kKGYiICAgIGlmIChjaHVua0lkeCA9PSAwKSB2ID0ge25hbWV9MFt7dmFyfT4+Ml07IikKICAgICAg"
    "ICBmb3IgayBpbiByYW5nZSgxLCBudW1fY2h1bmtzKToKICAgICAgICAgICAgbGluZXMuYXBwZW5kKGYi"
    "ICAgIGVsc2UgaWYgKGNodW5rSWR4ID09IHtrfSkgdiA9IHtuYW1lfXtrfVt7dmFyfT4+Ml07IikKICAg"
    "ICAgICBsaW5lcy5hcHBlbmQoZiIgICAgcmV0dXJuIHY7IikKICAgICAgICByZXR1cm4gIlxuIi5qb2lu"
    "KGxpbmVzKQoKICAgIGRlZiBpdmVjNF9zZWxlY3QodmFyPSdpJyk6CiAgICAgICAgcmV0dXJuIGYiIiIg"
    "ICAgaW50IGNpID0ge3Zhcn0gJiAzOwogICAgcmV0dXJuIGNpPT0wID8gdi54IDogY2k9PTEgPyB2Lnkg"
    "OiBjaT09MiA/IHYueiA6IHYudzsiIiIKCiAgICBmZXRjaGVycyA9IGYiIiIKLy8g4pWQ4pWQ4pWQIENo"
    "dW5rZWQgaXZlYzQgZmV0Y2hlcnMg4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ"
    "4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQCgovLyBGZXRjaCBhIGJ5"
    "dGUgZnJvbSBhbnkgY2h1bmtlZCBieXRlIGFycmF5IChNU0ItZmlyc3Qgd2l0aGluIGVhY2ggaW50MzIp"
    "LgovLyBFYWNoIGl2ZWM0IGhvbGRzIDE2IGJ5dGVzOiAueCA9IGJ5dGVzIDAtMywgLnkgPSA0LTcsIC56"
    "ID0gOC0xMSwgLncgPSAxMi0xNQovLyBXaXRoaW4gZWFjaCBpbnQ6IGJ5dGUgMCA9IE1TQiwgYnl0ZSAz"
    "ID0gTFNCLgoKaW50IF9leHRyYWN0Qnl0ZShpdmVjNCB2LCBpbnQgYnl0ZUluSXZlYzQpIHt7CiAgICBp"
    "bnQgaW50SWR4ID0gYnl0ZUluSXZlYzQgPj4gMjsKICAgIGludCBieXRlSW5JbnQgPSBieXRlSW5JdmVj"
    "NCAmIDM7CiAgICBpbnQgcGFja2VkID0gaW50SWR4PT0wID8gdi54IDogaW50SWR4PT0xID8gdi55IDog"
    "aW50SWR4PT0yID8gdi56IDogdi53OwogICAgaW50IHNoaWZ0ID0gMjQgLSBieXRlSW5JbnQgKiA4Owog"
    "ICAgcmV0dXJuIChwYWNrZWQgPj4gc2hpZnQpICYgMHhGRjsKfX0KCi8vIOKUgOKUgCBEaWN0aW9uYXJ5"
    "IGJ5dGUgZmV0Y2ggKGJ5dGVJZHggaW4gWzAsIERJQ1RfTk9URVMqNCkpIOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAppbnQgZmV0Y2hEaWN0Qnl0ZShpbnQg"
    "Ynl0ZUlkeCkge3sKICAgIGludCBpdmVjNElkeCA9IGJ5dGVJZHggPj4gNDsKICAgIGludCBieXRlSW5J"
    "dmVjNCA9IGJ5dGVJZHggJiAxNTsKICAgIGl2ZWM0IHYgPSBwYXREaWN0MFtpdmVjNElkeF07CiAgICBy"
    "ZXR1cm4gX2V4dHJhY3RCeXRlKHYsIGJ5dGVJbkl2ZWM0KTsKfX0KCi8vIOKUgOKUgCBCaXRtYXAgYnl0"
    "ZSBmZXRjaCDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKaW50IGZldGNoQml0"
    "bWFwQnl0ZShpbnQgYnl0ZUlkeCkge3sKICAgIGludCBpdmVjNElkeCA9IGJ5dGVJZHggPj4gNDsKICAg"
    "IGludCBieXRlSW5JdmVjNCA9IGJ5dGVJZHggJiAxNTsKICAgIGl2ZWM0IHYgPSBwYXRCaXRtYXAwW2l2"
    "ZWM0SWR4XTsKICAgIHJldHVybiBfZXh0cmFjdEJ5dGUodiwgYnl0ZUluSXZlYzQpOwp9fQoKLy8g4pSA"
    "4pSAIEluZGV4IHN0cmVhbSBieXRlIGZldGNoIChjaHVua2VkIGlmIG5lZWRlZCkg4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSACmludCBmZXRjaElkeEJ5dGUoaW50IGJ5dGVJZHgpIHt7CiAgICBpbnQgaXZlYzRJZHggPSBi"
    "eXRlSWR4ID4+IDQ7CiAgICBpbnQgYnl0ZUluSXZlYzQgPSBieXRlSWR4ICYgMTU7CiAgICBpbnQgY2h1"
    "bmtJZHggPSBpdmVjNElkeCAvIDUxMjsKICAgIGludCBsb2NhbEl2ZWM0ID0gaXZlYzRJZHggJSA1MTI7"
    "CiAgICBpdmVjNCB2ID0gaXZlYzQoMCk7CntjaHIoMTApLmpvaW4oZicgICAgeyJpZiIgaWYgaz09MCBl"
    "bHNlICJlbHNlIGlmIn0gKGNodW5rSWR4ID09IHtrfSkgdiA9IHBhdElkeHtrfVtsb2NhbEl2ZWM0XTsn"
    "IGZvciBrIGluIHJhbmdlKGxlbihpZHhfY2h1bmtzKSkpfQogICAgcmV0dXJuIF9leHRyYWN0Qnl0ZSh2"
    "LCBieXRlSW5JdmVjNCk7Cn19CgovLyDilIDilIAgUm93LXNlZWsgbmliYmxlIGJ5dGUgZmV0Y2gg4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA"
    "4pSA4pSACmludCBmZXRjaFJvd1NlZWtCeXRlKGludCBieXRlSWR4KSB7ewogICAgaW50IGl2ZWM0SWR4"
    "ID0gYnl0ZUlkeCA+PiA0OwogICAgaW50IGJ5dGVJbkl2ZWM0ID0gYnl0ZUlkeCAmIDE1OwogICAgaXZl"
    "YzQgdiA9IHBhdFJvd1NlZWswW2l2ZWM0SWR4XTsKICAgIHJldHVybiBfZXh0cmFjdEJ5dGUodiwgYnl0"
    "ZUluSXZlYzQpOwp9fQoKLy8g4pSA4pSAIFZRIGNvZGUgc3RyZWFtIGJ5dGUgZmV0Y2ggKGNodW5rZWQp"
    "IOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU"
    "gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAppbnQgZmV0Y2hDb2Rlc0J5"
    "dGUoaW50IGJ5dGVJZHgpIHt7CiAgICBpbnQgaXZlYzRJZHggPSBieXRlSWR4ID4+IDQ7CiAgICBpbnQg"
    "Ynl0ZUluSXZlYzQgPSBieXRlSWR4ICYgMTU7CiAgICBpbnQgY2h1bmtJZHggPSBpdmVjNElkeCAvIDUx"
    "MjsKICAgIGludCBsb2NhbEl2ZWM0ID0gaXZlYzRJZHggJSA1MTI7CiAgICBpdmVjNCB2ID0gaXZlYzQo"
    "MCk7CntjaHIoMTApLmpvaW4oZicgICAgeyJpZiIgaWYgaz09MCBlbHNlICJlbHNlIGlmIn0gKGNodW5r"
    "SWR4ID09IHtrfSkgdiA9IHZxQ29kZXN7a31bbG9jYWxJdmVjNF07JyBmb3IgayBpbiByYW5nZShsZW4o"
    "Y29kZXNfY2h1bmtzKSkpfQogICAgcmV0dXJuIF9leHRyYWN0Qnl0ZSh2LCBieXRlSW5JdmVjNCk7Cn19"
    "CgovLyDilIDilIAgVlEgY29kZWJvb2sgYnl0ZSBmZXRjaCAoc21hbGwsIGZpdHMgaW4gMSBjaHVuayB1"
    "c3VhbGx5KSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKaW50IGZldGNo"
    "Q29kZWJvb2tCeXRlKGludCBieXRlSWR4KSB7ewogICAgaW50IGl2ZWM0SWR4ID0gYnl0ZUlkeCA+PiA0"
    "OwogICAgaW50IGJ5dGVJbkl2ZWM0ID0gYnl0ZUlkeCAmIDE1OwogICAgaW50IGNodW5rSWR4ID0gaXZl"
    "YzRJZHggLyA1MTI7CiAgICBpbnQgbG9jYWxJdmVjNCA9IGl2ZWM0SWR4ICUgNTEyOwogICAgaXZlYzQg"
    "diA9IGl2ZWM0KDApOwp7Y2hyKDEwKS5qb2luKGYnICAgIHsiaWYiIGlmIGs9PTAgZWxzZSAiZWxzZSBp"
    "ZiJ9IChjaHVua0lkeCA9PSB7a30pIHYgPSB2cUNvZGVib29re2t9W2xvY2FsSXZlYzRdOycgZm9yIGsg"
    "aW4gcmFuZ2UobGVuKGNiX2NodW5rcykpKX0KICAgIHJldHVybiBfZXh0cmFjdEJ5dGUodiwgYnl0ZUlu"
    "SXZlYzQpOwp9fQoiIiIKCiAgICAjIOKUgOKUgCBwb3Bjb3VudCBoZWxwZXIgKDQtYml0IG5pYmJsZSkK"
    "ICAgICMg4pSA4pSAIGdldE5vdGU6IGJpdG1hcCArIGRpY3QgbG9va3VwIHdpdGggTygxKSByb3cgc2Vl"
    "ayArIHByZWZpeCBwb3Bjb3VudAogICAgZGVjb2RlcnMgPSAiIiIKLy8g4pWQ4pWQ4pWQIFBhdHRlcm4g"
    "ZGVjb2RlcjogYml0bWFwICsgZGljdGlvbmFyeSArIHJvdyBzZWVrIOKVkOKVkOKVkOKVkOKVkOKVkOKV"
    "kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkAoKc3RydWN0IE5v"
    "dGUgeyBpbnQgaW5zdHJ1bWVudCwgcGVyaW9kLCBlZmZlY3QsIHBhcmFtOyB9OwoKLy8gUG9wY291bnQg"
    "Zm9yIDQtYml0IG5pYmJsZSAoMC4uMTUg4oaSIDAuLjQpCmludCBwb3Bjb3VudDQoaW50IHgpIHsKICAg"
    "IHggPSAoeCAmIDB4NSkgKyAoKHggPj4gMSkgJiAweDUpOwogICAgcmV0dXJuICh4ICYgMHgzKSArICgo"
    "eCA+PiAyKSAmIDB4Myk7Cn0KCi8vIFJlY29uc3RydWN0IGN1bXVsYXRpdmUgbm9uLWVtcHR5IGNvdW50"
    "IHVwIHRvIHN0YXJ0IG9mIGByb3dgIOKAlCBPKDEpLgovLyBSb3cgc2VlayB0YWJsZSBob2xkcyAxNi1i"
    "aXQgTEUgcHJlZml4IHN1bXM6IDIgYnl0ZXMgcGVyIHJvdy4KaW50IHJvd1NlZWtDdW0oaW50IHRhcmdl"
    "dFJvdykgewogICAgaW50IGJ5dGVJZHggPSB0YXJnZXRSb3cgKiAyOwogICAgaW50IGxvID0gZmV0Y2hS"
    "b3dTZWVrQnl0ZShieXRlSWR4KTsKICAgIGludCBoaSA9IGZldGNoUm93U2Vla0J5dGUoYnl0ZUlkeCAr"
    "IDEpOwogICAgcmV0dXJuIGxvIHwgKGhpIDw8IDgpOwp9CgpOb3RlIGVtcHR5Tm90ZSgpIHsgTm90ZSBu"
    "OyBuLmluc3RydW1lbnQ9MDsgbi5wZXJpb2Q9MDsgbi5lZmZlY3Q9MDsgbi5wYXJhbT0wOyByZXR1cm4g"
    "bjsgfQoKTm90ZSBnZXROb3RlKGludCBzb25nUG9zLCBpbnQgcm93LCBpbnQgY2hhbm5lbCkgewogICAg"
    "aW50IHBhdCA9IHNvbmdQb3NpdGlvbnNbc29uZ1Bvc107CiAgICBpbnQgcm93R2xvYmFsID0gcGF0ICog"
    "NjQgKyByb3c7CiAgICBpbnQgbm90ZUlkeCAgID0gcm93R2xvYmFsICogNCArIGNoYW5uZWw7CgogICAg"
    "Ly8gMSkgQml0bWFwIGNoZWNrCiAgICBpbnQgYm1CeXRlID0gZmV0Y2hCaXRtYXBCeXRlKG5vdGVJZHgg"
    "Pj4gMyk7CiAgICBpbnQgYml0ID0gKGJtQnl0ZSA+PiAobm90ZUlkeCAmIDcpKSAmIDE7CiAgICBpZiAo"
    "Yml0ID09IDApIHJldHVybiBlbXB0eU5vdGUoKTsKCiAgICAvLyAyKSBDb3VudCBub24tZW1wdHkgbm90"
    "ZXMgYmVmb3JlIHRoaXMgcG9zaXRpb24KICAgIC8vICAgID0gY3VtdWxhdGl2ZSB1cCB0byByb3dHbG9i"
    "YWwgKyBwb3Bjb3VudCBvZiBiaXRtYXAgbmliYmxlIHdpdGhpbiB0aGlzIHJvdwogICAgaW50IHJhbmsg"
    "PSByb3dTZWVrQ3VtKHJvd0dsb2JhbCk7CiAgICAvLyBUaGlzIHJvdydzIDQgYml0cyBzcGFuIGNoYW5u"
    "ZWxzIDAuLjMg4oaSIHRha2UgY2hhbm5lbHMgWzAuLmNoYW5uZWwtMV0KICAgIGludCByb3dCaXRtYXBT"
    "dGFydCA9IHJvd0dsb2JhbCAqIDQ7CiAgICAvLyBUaGUgNCBiaXRzIG9mIHRoaXMgcm93IG1heSBzcGFu"
    "IDEgYnl0ZSAoaWYgYWxpZ25lZCkgb3IgMi4KICAgIGludCBieXRlMElkeCA9IHJvd0JpdG1hcFN0YXJ0"
    "ID4+IDM7CiAgICBpbnQgc2hpZnQgICAgPSByb3dCaXRtYXBTdGFydCAmIDc7CiAgICBpbnQgYnl0ZTAg"
    "PSBmZXRjaEJpdG1hcEJ5dGUoYnl0ZTBJZHgpOwogICAgaW50IGJ5dGUxID0gZmV0Y2hCaXRtYXBCeXRl"
    "KGJ5dGUwSWR4ICsgMSk7CiAgICBpbnQgcm93Qml0cyA9ICgoYnl0ZTAgPj4gc2hpZnQpIHwgKGJ5dGUx"
    "IDw8ICg4IC0gc2hpZnQpKSkgJiAweEY7CiAgICBpbnQgbWFzayA9ICgxIDw8IGNoYW5uZWwpIC0gMTsK"
    "ICAgIHJhbmsgKz0gcG9wY291bnQ0KHJvd0JpdHMgJiBtYXNrKTsKCiAgICAvLyAzKSBMb29rIHVwIGlu"
    "ZGV4IGFuZCBmZXRjaCBub3RlIGZyb20gZGljdGlvbmFyeQogICAgaW50IGRpY3RJZHg7CiNpZiBJRFhf"
    "QllURVNfUEVSID09IDEKICAgIGRpY3RJZHggPSBmZXRjaElkeEJ5dGUocmFuayk7CiNlbHNlCiAgICBp"
    "bnQgbG8gPSBmZXRjaElkeEJ5dGUocmFuayAqIDIpOwogICAgaW50IGhpID0gZmV0Y2hJZHhCeXRlKHJh"
    "bmsgKiAyICsgMSk7CiAgICBkaWN0SWR4ID0gbG8gfCAoaGkgPDwgOCk7CiNlbmRpZgogICAgaW50IGIw"
    "ID0gZmV0Y2hEaWN0Qnl0ZShkaWN0SWR4ICogNCArIDApOwogICAgaW50IGIxID0gZmV0Y2hEaWN0Qnl0"
    "ZShkaWN0SWR4ICogNCArIDEpOwogICAgaW50IGIyID0gZmV0Y2hEaWN0Qnl0ZShkaWN0SWR4ICogNCAr"
    "IDIpOwogICAgaW50IGIzID0gZmV0Y2hEaWN0Qnl0ZShkaWN0SWR4ICogNCArIDMpOwoKICAgIE5vdGUg"
    "bjsKICAgIG4uaW5zdHJ1bWVudCA9IChiMCAmIDB4RjApIHwgKChiMiA+PiA0KSAmIDB4MEYpOwogICAg"
    "bi5wZXJpb2QgICAgID0gKChiMCAmIDB4MEYpIDw8IDgpIHwgYjE7CiAgICBuLmVmZmVjdCAgICAgPSBi"
    "MiAmIDB4MEY7CiAgICBuLnBhcmFtICAgICAgPSBiMzsKICAgIHJldHVybiBuOwp9CgoiIiIKCiAgICAj"
    "IFNhbXBsZSBkZWNvZGVyOiBmLXN0cmluZyBmb3IgI2RlZmluZXMgKG5lZWQgUHl0aG9uIHZhcnMpLCBw"
    "bGFpbiBzdHJpbmcgZm9yIGZ1bmN0aW9uIGJvZGllcwogICAgX3N0YWdlX2xhYmVsID0gIjEtc3RhZ2Ug"
    "UlZRIChubyBzdGFnZSAyKSIgaWYgbm9fcnZxMiBlbHNlICIyLXN0YWdlIFJWUSIKICAgIF9wYWNrZm10"
    "ICAgICA9IGYie0JJVFMxfS1iaXQgY29kZTEgb25seSIgaWYgbm9fcnZxMiBlbHNlIGYiW3tCSVRTMX0t"
    "Yml0IGNvZGUxXVt7QklUUzJ9LWJpdCBjb2RlMl0iCiAgICBkZWNvZGVycyArPSAoCiAgICAgICAgZiIv"
    "LyDilZDilZDilZAgU2FtcGxlIGRlY29kZXI6IHtfc3RhZ2VfbGFiZWx9IMOXe2Rvd25zYW1wbGV9IEFB"
    "LWRvd25zYW1wbGVkIChwZXItc2FtcGxlIERTKSDilZDilZBcbiIKICAgICAgICBmIi8vIHtCSVRTX1RP"
    "VEFMfS1iaXQgY29kZXMgcGFja2VkIExTQi1maXJzdDoge19wYWNrZm10fVxuIgogICAgICAgIGYiLy8g"
    "cGVyaW9kVG9GcmVxID0gNzA5Mzc4OS4yLyhwZXJpb2QqMikg4oCUIHBlci1zYW1wbGUgRFMgdmlhIFNh"
    "bXBsZUluZm8uYndGYWN0b3JcbiIKICAgICAgICBmIiNkZWZpbmUgUlZRX0JJVFMgICAgIHtCSVRTX1RP"
    "VEFMfVxuIgogICAgICAgIGYiI2RlZmluZSBSVlFfQklUU18xICAge0JJVFMxfVxuIgogICAgICAgIGYi"
    "I2RlZmluZSBSVlFfSzEgICAgICAge0sxfVxuIgogICAgICAgIGYiI2RlZmluZSBSVlFfSzIgICAgICAg"
    "e0syfVxuIgogICAgICAgIGYiI2RlZmluZSBSVlFfVkVDX0RJTSAge3ZlY19kaW19XG4iCiAgICAgICAg"
    "ZiIjZGVmaW5lIFJWUV9DQjJfQllURSAoe0sxfSAqIHt2ZWNfZGltfSlcbiIKICAgICAgICBmIiNkZWZp"
    "bmUgUlZRX01BU0sxICAgIHsoMTw8QklUUzEpLTF9XG4iCiAgICAgICAgZiIjZGVmaW5lIFJWUV9NQVNL"
    "MiAgICB7KDE8PEJJVFMyKS0xIGlmIEJJVFMyPjAgZWxzZSAwfVxuIgogICAgICAgICsgKGYiI2RlZmlu"
    "ZSBSVlFfTk9fU1RBR0UyIDFcbiIgaWYgbm9fcnZxMiBlbHNlICIiKQogICAgKQogICAgZGVjb2RlcnMg"
    "Kz0gIiIiCnZvaWQgX2dldFJWUUNvZGVzKGludCB2ZWNJZHgsIG91dCBpbnQgY29kZTEsIG91dCBpbnQg"
    "Y29kZTIpIHsKICAgIGludCBiaXRQb3MgID0gdmVjSWR4ICogUlZRX0JJVFM7CiAgICBpbnQgYnl0ZVBv"
    "cyA9IGJpdFBvcyA+PiAzOwogICAgaW50IHNoaWZ0ICAgPSBiaXRQb3MgJiA3OwogICAgaW50IGIwID0g"
    "ZmV0Y2hDb2Rlc0J5dGUoYnl0ZVBvcyk7CiAgICBpbnQgYjEgPSBmZXRjaENvZGVzQnl0ZShieXRlUG9z"
    "ICsgMSk7CiAgICBpbnQgYjIgPSBmZXRjaENvZGVzQnl0ZShieXRlUG9zICsgMik7CiAgICBpbnQgYjMg"
    "PSBmZXRjaENvZGVzQnl0ZShieXRlUG9zICsgMyk7CiAgICBpbnQgY29tYmluZWQgPSBiMCB8IChiMSA8"
    "PCA4KSB8IChiMiA8PCAxNikgfCAoYjMgPDwgMjQpOwogICAgaW50IHJhdyA9IChjb21iaW5lZCA+PiBz"
    "aGlmdCkgJiAoKDEgPDwgUlZRX0JJVFMpIC0gMSk7CiAgICBjb2RlMSA9IHJhdyAmIFJWUV9NQVNLMTsK"
    "I2lmZGVmIFJWUV9OT19TVEFHRTIKICAgIGNvZGUyID0gMDsKI2Vsc2UKICAgIGNvZGUyID0gKHJhdyA+"
    "PiBSVlFfQklUU18xKSAmIFJWUV9NQVNLMjsKI2VuZGlmCn0KCmZsb2F0IGdldFNhbXBsZShpbnQgc2Ft"
    "cGxlSWR4KSB7CiAgICBpZiAoc2FtcGxlSWR4IDwgMCB8fCBzYW1wbGVJZHggPj0gVE9UQUxfU0FNUExF"
    "UykgcmV0dXJuIDAuMDsKICAgIGludCB2ZWNJZHggPSBzYW1wbGVJZHggLyBSVlFfVkVDX0RJTTsKICAg"
    "IGludCBsYW5lICAgPSBzYW1wbGVJZHggLSB2ZWNJZHggKiBSVlFfVkVDX0RJTTsKICAgIC8vIElubGlu"
    "ZSBSVlEgZGVjb2RlIChhdm9pZHMgb3V0LXBhcmFtZXRlciBzdGFjayBhbGxvY2F0aW9uKQogICAgaW50"
    "IF9icCA9IHZlY0lkeCAqIFJWUV9CSVRTLCBfYnkgPSBfYnAgPj4gMywgX3NoID0gX2JwICYgNzsKICAg"
    "IGludCBfcmF3ID0gKGZldGNoQ29kZXNCeXRlKF9ieSkgfCAoZmV0Y2hDb2Rlc0J5dGUoX2J5KzEpPDw4"
    "KSB8CiAgICAgICAgICAgICAgICAoZmV0Y2hDb2Rlc0J5dGUoX2J5KzIpPDwxNikgfCAoZmV0Y2hDb2Rl"
    "c0J5dGUoX2J5KzMpPDwyNCkpOwogICAgX3JhdyA9IChfcmF3ID4+IF9zaCkgJiAoKDEgPDwgUlZRX0JJ"
    "VFMpIC0gMSk7CiAgICBpbnQgY29kZTEgPSBfcmF3ICYgUlZRX01BU0sxOwogICAgaW50IHViMSA9IGZl"
    "dGNoQ29kZWJvb2tCeXRlKGNvZGUxICogUlZRX1ZFQ19ESU0gKyBsYW5lKTsKICAgIGludCBzMSAgPSB1"
    "YjEgPCAxMjggPyB1YjEgOiB1YjEgLSAyNTY7CiNpZmRlZiBSVlFfTk9fU1RBR0UyCiAgICByZXR1cm4g"
    "ZmxvYXQoczEpIC8gMTI4LjA7CiNlbHNlCiAgICBpbnQgY29kZTIgPSAoX3JhdyA+PiBSVlFfQklUU18x"
    "KSAmIFJWUV9NQVNLMjsKICAgIGludCB1YjIgPSBmZXRjaENvZGVib29rQnl0ZShSVlFfQ0IyX0JZVEUg"
    "KyBjb2RlMiAqIFJWUV9WRUNfRElNICsgbGFuZSk7CiAgICBpbnQgczIgID0gdWIyIDwgMTI4ID8gdWIy"
    "IDogdWIyIC0gMjU2OwogICAgcmV0dXJuIGZsb2F0KHMxICsgczIpIC8gMTI4LjA7CiNlbmRpZgp9Cgov"
    "LyDilIDilIAgUG9zaXRpb24gY2FsY3VsYXRpb24gKEZ4eC1hd2FyZSB2aWEgcm93U3RhcnRUaWNrKSDi"
    "lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAK"
    "c3RydWN0IFBvc2l0aW9uIHsgaW50IHNvbmdQb3MsIHBhdHRlcm4sIHJvdzsgZmxvYXQgdGljaywgcm93"
    "VGltZTsgfTsKCi8vIEZldGNoIDE2LWJpdCBMRSB2YWx1ZSBhdCByb3cgaW5kZXggaW50byByb3dTdGFy"
    "dFRpY2sKaW50IGZldGNoVGljayhpbnQgcm93SWR4KSB7CiAgICBpbnQgYnl0ZUlkeCA9IHJvd0lkeCAq"
    "IDI7CiAgICBpbnQgY2h1bmtJZHggID0gYnl0ZUlkeCA+PiA2OwogICAgaW50IGJ5dGVJbjE2ICA9IGJ5"
    "dGVJZHggJiA2MzsKICAgIGludCBsbyA9IF9leHRyYWN0Qnl0ZShyb3dTdGFydFRpY2swWyhjaHVua0lk"
    "eDw8MikrKGJ5dGVJbjE2Pj40KV0sIGJ5dGVJbjE2ICYgMTUpOwogICAgLy8gbmV4dCBieXRlCiAgICBp"
    "bnQgYnl0ZUlkeDIgPSBieXRlSWR4ICsgMTsKICAgIGludCBjaHVua0lkeDIgPSBieXRlSWR4MiA+PiA2"
    "OwogICAgaW50IGJ5dGVJbjE2XzIgPSBieXRlSWR4MiAmIDYzOwogICAgaW50IGhpID0gX2V4dHJhY3RC"
    "eXRlKHJvd1N0YXJ0VGljazBbKGNodW5rSWR4Mjw8MikrKGJ5dGVJbjE2XzI+PjQpXSwgYnl0ZUluMTZf"
    "MiAmIDE1KTsKICAgIHJldHVybiBsbyB8IChoaSA8PCA4KTsKfQoKUG9zaXRpb24gZ2V0UG9zaXRpb24o"
    "ZmxvYXQgdGltZSkgewogICAgUG9zaXRpb24gcG9zOwogICAgZmxvYXQgc29uZ0R1cmF0aW9uID0gZmxv"
    "YXQoVE9UQUxfVElDS1MpIC8gVElDS1NfUEVSX1NFQzsKICAgIGZsb2F0IGxvb3BlZFRpbWUgPSBtb2Qo"
    "dGltZSwgc29uZ0R1cmF0aW9uKTsKICAgIGZsb2F0IHRvdGFsVGlja0YgPSBsb29wZWRUaW1lICogVElD"
    "S1NfUEVSX1NFQzsKCiAgICAvLyBCaW5hcnkgc2VhcmNoIHJvd1N0YXJ0VGljayBmb3IgdGhlIGN1cnJl"
    "bnQgcm93CiAgICBpbnQgbG8gPSAwLCBoaSA9IE5VTV9TT05HX1JPV1M7CiAgICBmb3IgKGludCBfYnMg"
    "PSAwOyBfYnMgPCAxMjsgX2JzKyspIHsgIC8vIGxvZzIoMTkyMCspIOKJiCAxMQogICAgICAgIGlmIChs"
    "byA+PSBoaSAtIDEpIGJyZWFrOwogICAgICAgIGludCBtaWQgPSAobG8gKyBoaSkgPj4gMTsKICAgICAg"
    "ICBpZiAoZmxvYXQoZmV0Y2hUaWNrKG1pZCkpIDw9IHRvdGFsVGlja0YpIGxvID0gbWlkOwogICAgICAg"
    "IGVsc2UgaGkgPSBtaWQ7CiAgICB9CiAgICBpbnQgZ2xvYmFsUm93ID0gbG87CiAgICBpZiAoZ2xvYmFs"
    "Um93ID49IE5VTV9TT05HX1JPV1MpIGdsb2JhbFJvdyA9IE5VTV9TT05HX1JPV1MgLSAxOwoKICAgIC8v"
    "IEZpbmQgc29uZ1BvcyB2aWEgbGluZWFyIHNlYXJjaCBvdmVyIHBhdFRpY2tPZmZzZXQgKFNPTkdfTEVO"
    "R1RIIOKJpCAxMjgsIGZhc3QgZW5vdWdoKQogICAgaW50IHNwID0gU09OR19MRU5HVEggLSAxOwogICAg"
    "Zm9yIChpbnQgX2kgPSAwOyBfaSA8IFNPTkdfTEVOR1RIIC0gMTsgX2krKykgewogICAgICAgIGlmIChw"
    "YXRUaWNrT2Zmc2V0W19pICsgMV0gPiBnbG9iYWxSb3cpIHsgc3AgPSBfaTsgYnJlYWs7IH0KICAgIH0K"
    "ICAgIHBvcy5zb25nUG9zID0gc3A7CiAgICBwb3MucGF0dGVybiA9IHNvbmdQb3NpdGlvbnNbc3BdOwog"
    "ICAgcG9zLnJvdyAgICAgPSBnbG9iYWxSb3cgLSBwYXRUaWNrT2Zmc2V0W3NwXTsKCiAgICBpbnQgcm93"
    "VGljayAgICA9IGZldGNoVGljayhnbG9iYWxSb3cpOwogICAgaW50IG5leHRUaWNrICAgPSBmZXRjaFRp"
    "Y2soZ2xvYmFsUm93ICsgMSk7CiAgICBpbnQgcm93U3BlZWQgICA9IG5leHRUaWNrIC0gcm93VGljazsK"
    "ICAgIHBvcy50aWNrICAgICAgID0gdG90YWxUaWNrRiAtIGZsb2F0KHJvd1RpY2spOwogICAgcG9zLnJv"
    "d1RpbWUgICAgPSBmbG9hdChyb3dTcGVlZCkgLyBUSUNLU19QRVJfU0VDOwogICAgcmV0dXJuIHBvczsK"
    "fQoKLy8gNC1wb2ludCBjdWJpYyBCLXNwbGluZSBpbnRlcnBvbGF0aW9uLgovLyBCLXNwbGluZSBpcyBB"
    "UFBST1hJTUFUSU5HIChzbW9vdGhzIHRocm91Z2ggc2FtcGxlIHBvaW50cykgcmF0aGVyIHRoYW4KLy8g"
    "SU5URVJQT0xBVElORyAocGFzc2luZyBleGFjdGx5IHRocm91Z2ggdGhlbSksIGdpdmluZyBpbmhlcmVu"
    "dCBsb3ctcGFzcwovLyBjaGFyYWN0ZXIgdGhhdCByZWR1Y2VzIGhpZ2gtZnJlcXVlbmN5IHF1YW50aXph"
    "dGlvbiBub2lzZS4KIiIiICsgKAogICAgICAgICAgICAjIOKUgOKUgCBMaW5lYXI6IDIgdGFwcywgUHJv"
    "VHJhY2tlci1hdXRoZW50aWMsIGNoZWFwZXN0IOKUgOKUgAogICAgICAgICAgICAnJydmbG9hdCBnZXRT"
    "YW1wbGVGKGludCBiYXNlLCBmbG9hdCBmcG9zLCBpbnQgc21wTGVuLCBpbnQgbG9vcFN0YXJ0LCBpbnQg"
    "bG9vcExlbikgewogICAgaW50IGkgPSBpbnQoZnBvcyk7CiAgICBmbG9hdCB0ID0gZnBvcyAtIGZsb2F0"
    "KGkpOwogICAgZmxvYXQgcDEgPSBnZXRTYW1wbGUoYmFzZSArIGkpOwogICAgZmxvYXQgcDIgPSBnZXRT"
    "YW1wbGUoYmFzZSArIG1pbihpICsgMSwgc21wTGVuICsgMTUpKTsKICAgIHJldHVybiBtaXgocDEsIHAy"
    "LCB0KTsKfScnJyBpZiByZXNhbXBsZXIgPT0gJ2xpbmVhcicgZWxzZQogICAgICAgICAgICAjIOKUgOKU"
    "gCBMYW5jem9zLTM6IDYgdGFwcywgc2hhcnBlc3QsIGJyaWdodGVzdCDilIDilIAKICAgICAgICAgICAg"
    "JycnLy8gTGFuY3pvcy0zIHdpbmRvd2VkIHNpbmM6IHcoeCkgPSBzaW5jKM+AeCkgKiBzaW5jKM+AeC8z"
    "KSBmb3IgfHh8PDMKZmxvYXQgX2xhbmN6b3MzKGZsb2F0IHgpIHsKICAgIGlmICh4IDwgMWUtNikgcmV0"
    "dXJuIDEuMDsKICAgIGZsb2F0IHBpeCA9IDMuMTQxNTkyNjUgKiB4OwogICAgZmxvYXQgcGl4MyA9IHBp"
    "eCAvIDMuMDsKICAgIHJldHVybiAoc2luKHBpeCkgKiBzaW4ocGl4MykpIC8gKHBpeCAqIHBpeDMpOwp9"
    "CmZsb2F0IGdldFNhbXBsZUYoaW50IGJhc2UsIGZsb2F0IGZwb3MsIGludCBzbXBMZW4sIGludCBsb29w"
    "U3RhcnQsIGludCBsb29wTGVuKSB7CiAgICBpbnQgaSAgPSBpbnQoZnBvcyk7CiAgICBmbG9hdCB0ID0g"
    "ZnBvcyAtIGZsb2F0KGkpOwogICAgaW50IGltMiA9IGkgLSAyLCBpbTEgPSBpIC0gMSwgaXAxID0gaSAr"
    "IDEsIGlwMiA9IGkgKyAyLCBpcDMgPSBpICsgMzsKICAgIGlmIChsb29wTGVuID4gMiAmJiBpbTIgPCBs"
    "b29wU3RhcnQpIGltMiA9IGxvb3BTdGFydCArIGxvb3BMZW4gKyAoaW0yIC0gbG9vcFN0YXJ0KTsKICAg"
    "IGlmIChsb29wTGVuID4gMiAmJiBpbTEgPCBsb29wU3RhcnQpIGltMSA9IGxvb3BTdGFydCArIGxvb3BM"
    "ZW4gKyAoaW0xIC0gbG9vcFN0YXJ0KTsKICAgIGltMiA9IG1heCgwLCBpbTIpOyBpbTEgPSBtYXgoMCwg"
    "aW0xKTsKICAgIGlwMSA9IG1pbihpcDEsIHNtcExlbiArIDE1KTsKICAgIGlwMiA9IG1pbihpcDIsIHNt"
    "cExlbiArIDE1KTsKICAgIGlwMyA9IG1pbihpcDMsIHNtcExlbiArIDE1KTsKICAgIGZsb2F0IHcwID0g"
    "X2xhbmN6b3MzKGFicyh0ICsgMi4wKSk7CiAgICBmbG9hdCB3MSA9IF9sYW5jem9zMyhhYnModCArIDEu"
    "MCkpOwogICAgZmxvYXQgdzIgPSBfbGFuY3pvczMoYWJzKHQgICAgICApKTsKICAgIGZsb2F0IHczID0g"
    "X2xhbmN6b3MzKGFicyh0IC0gMS4wKSk7CiAgICBmbG9hdCB3NCA9IF9sYW5jem9zMyhhYnModCAtIDIu"
    "MCkpOwogICAgZmxvYXQgdzUgPSBfbGFuY3pvczMoYWJzKHQgLSAzLjApKTsKICAgIGZsb2F0IHdzdW0g"
    "PSB3MCt3MSt3Mit3Myt3NCt3NTsKICAgIHJldHVybiAodzAqZ2V0U2FtcGxlKGJhc2UraW0yKSArIHcx"
    "KmdldFNhbXBsZShiYXNlK2ltMSkgKwogICAgICAgICAgICB3MipnZXRTYW1wbGUoYmFzZStpICApICsg"
    "dzMqZ2V0U2FtcGxlKGJhc2UraXAxKSArCiAgICAgICAgICAgIHc0KmdldFNhbXBsZShiYXNlK2lwMikg"
    "KyB3NSpnZXRTYW1wbGUoYmFzZStpcDMpKSAvIHdzdW07Cn0nJycgaWYgcmVzYW1wbGVyID09ICdsYW5j"
    "em9zMycgZWxzZQogICAgICAgICAgICAjIOKUgOKUgCBCLXNwbGluZSAoZGVmYXVsdCk6IDQgdGFwcywg"
    "c21vb3RoLCBnZW50bGUgTFBGIOKUgOKUgAogICAgICAgICAgICAnJydmbG9hdCBnZXRTYW1wbGVGKGlu"
    "dCBiYXNlLCBmbG9hdCBmcG9zLCBpbnQgc21wTGVuLCBpbnQgbG9vcFN0YXJ0LCBpbnQgbG9vcExlbikg"
    "ewogICAgaW50IGkgID0gaW50KGZwb3MpOwogICAgZmxvYXQgdCA9IGZwb3MgLSBmbG9hdChpKTsKICAg"
    "IGludCBpMCA9IGkgLSAxOwogICAgaWYgKGxvb3BMZW4gPiAyICYmIGkwIDwgbG9vcFN0YXJ0KSBpMCA9"
    "IGxvb3BTdGFydCArIGxvb3BMZW4gLSAxOwogICAgZWxzZSBpMCA9IG1heCgwLCBpMCk7CiAgICBmbG9h"
    "dCBwMCA9IGdldFNhbXBsZShiYXNlICsgaTApOwogICAgZmxvYXQgcDEgPSBnZXRTYW1wbGUoYmFzZSAr"
    "IGkpOwogICAgZmxvYXQgcDIgPSBnZXRTYW1wbGUoYmFzZSArIG1pbihpICsgMSwgc21wTGVuICsgMTUp"
    "KTsKICAgIGZsb2F0IHAzID0gZ2V0U2FtcGxlKGJhc2UgKyBtaW4oaSArIDIsIHNtcExlbiArIDE1KSk7"
    "CiAgICBmbG9hdCB0MiA9IHQgKiB0OwogICAgZmxvYXQgdDMgPSB0MiAqIHQ7CiAgICBmbG9hdCB3MCA9"
    "ICgxLjAgLSB0KSAqICgxLjAgLSB0KSAqICgxLjAgLSB0KSAvIDYuMDsKICAgIGZsb2F0IHcxID0gKDMu"
    "MCAqIHQzIC0gNi4wICogdDIgKyA0LjApIC8gNi4wOwogICAgZmxvYXQgdzIgPSAoLTMuMCAqIHQzICsg"
    "My4wICogdDIgKyAzLjAgKiB0ICsgMS4wKSAvIDYuMDsKICAgIGZsb2F0IHczID0gdDMgLyA2LjA7CiAg"
    "ICByZXR1cm4gdzAgKiBwMCArIHcxICogcDEgKyB3MiAqIHAyICsgdzMgKiBwMzsKfScnJwogICAgICAg"
    "ICkgKyAiIiIKCiIiIgoKICAgIGltcG9ydCBiYXNlNjQgYXMgX2I2NGUKICAgIGdldF9jaGFubmVsX291"
    "dHB1dCA9IF9iNjRlLmI2NGRlY29kZSgnTHk4Z2RtbGlWR0ZpSUdseklHUmxZMnhoY21Wa0lHRnpJR0Vn"
    "WjJ4dlltRnNJR052Ym5OMElHWnNiMkYwV3pNeVhTQnVaV0Z5SUhSb1pTQjBiM0FnYjJZZ1EyOXRiVzl1"
    "Q2k4dklDaHlhV2RvZENCaFpuUmxjaUJ3WlhKcGIyUlVZV0pzWlNrdUlFUnZiaWQwSUhKbFpHVmpiR0Z5"
    "WlNCcGRDQm9aWEpsTGdvS1pteHZZWFFnWjJWMFEyaGhibTVsYkU5MWRIQjFkQ2hwYm5RZ1kyZ3NJR1pz"
    "YjJGMElIUnBiV1VzSUZCdmMybDBhVzl1SUhCdmN5d2dabXh2WVhRZ2NtOTNWR2x0WlNrZ2V3b0tJQ0Fn"
    "SUM4dklGTjBaWEFnTVRvZ1ptbHVaQ0J0YjNOMExYSmxZMlZ1ZEd4NUxYUnlhV2RuWlhKbFpDQnViM1Js"
    "SUc5dUlIUm9hWE1nWTJoaGJtNWxiQzRLSUNBZ0lDOHZJRkJVSUhObGJXRnVkR2xqY3lEaWdKUWdZU0Fp"
    "ZEhKcFoyZGxjaUlnYVhNZ2MyOXRaWFJvYVc1bklIUm9ZWFFnYzNSaGNuUnpJSFJvWlNCellXMXdiR1Vn"
    "WVhRZ2NHOXpJREE2Q2lBZ0lDQXZMeUFnSU9LQW9pQkdkV3hzSUhKdmR5QW9hVzV6ZEhKMWJXVnVkQ0Fy"
    "SUhCbGNtbHZaQ2tnSUNBZ0lDQWdJQ0FnSUNBZ0lPS0FsQ0J5WlhSeWFXZG5aWElLSUNBZ0lDOHZJQ0Fn"
    "NG9DaUlGQmxjbWx2WkMxdmJteDVJSEp2ZHlBb2JtOGdhVzV6ZEN3Z2JtOGdaV1ptWldOMElETXZOU2tn"
    "SUNBZzRvQ1VJSEpsZEhKcFoyZGxjaXdnYVc1b1pYSnBkQ0JwYm5OMGNuVnRaVzUwQ2lBZ0lDQXZMeUFn"
    "SU9LQW9pQlFaWEpwYjJRdGIyNXNlU0IzYVhSb0lHVm1abVZqZENBekx6VWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lPS0FsQ0J6Ykdsa1pTQjBZWEpuWlhRZ2IyNXNlU3dnYm04Z2NtVjBjbWxuWjJWeUNpQWdJQ0F2"
    "THlBZ0lPS0FvaUJHZFd4c0lISnZkeUIzYVhSb0lHVm1abVZqZENBekx6VWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJT0tBbENCemJHbGtaU0IwWVhKblpYUWdiMjVzZVN3Z2JtOGdjbVYwY21sbloyVnlDaUFn"
    "SUNBdkx5QWdJT0tBb2lCRmJYQjBlU0F2SUdsdWMzUnlkVzFsYm5RdGIyNXNlU0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0FnSU9LQWxDQmpiMjUwYVc1MVpTQndjbWx2Y2lCdWIzUmxDaUFnSUNCT2IzUmxJRjlq"
    "ZFhKU2IzY2dQU0JuWlhST2IzUmxLSEJ2Y3k1emIyNW5VRzl6TENCd2IzTXVjbTkzTENCamFDazdDaUFn"
    "SUNCT2IzUmxJSFJ5YVdkT2IzUmxJRDBnWDJOMWNsSnZkenNLSUNBZ0lHbHVkQ0FnZEhKcFoxSnZkeUFn"
    "UFNCd2IzTXVjbTkzT3dvZ0lDQWdhVzUwSUNCMGNtbG5VR0YwSUNBOUlIQnZjeTV6YjI1blVHOXpPd29n"
    "SUNBZ2FXNTBJQ0IwYjI1bFUyeHBaR1ZVWVhKblpYUWdQU0F3T3lBZ0x5OGdkMmhsYmlCelpYUXNJSFJv"
    "YVhNZ2NtOTNJR05oY25KcFpYTWdZU0F6ZUhndk5YaDRJSE5zYVdSbElIUmhjbWRsZEFvZ0lDQWdZbTl2"
    "YkNCZlkzVnlTWE5VYjI1bFVHOXlkR0VnUFNBb0tGOWpkWEpTYjNjdVpXWm1aV04wSUQwOUlEQjRNeUI4"
    "ZkNCZlkzVnlVbTkzTG1WbVptVmpkQ0E5UFNBd2VEVXBJQ1ltQ2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0FnSUNCZlkzVnlVbTkzTG5CbGNtbHZaQ0ErSURBcE93b2dJQ0FnWW05dmJDQmZZM1Z5"
    "U1hOU1pYUnlhV2NnSUNBZ1BTQW9YMk4xY2xKdmR5NXdaWEpwYjJRZ1BpQXdJQ1ltSUNGZlkzVnlTWE5V"
    "YjI1bFVHOXlkR0VwT3lBZ0x5OGdZVzU1SUhCbGNtbHZaQ0IzYVhSb2IzVjBJRE12TlNCeVpYUnlhV2Ru"
    "WlhKekNpQWdJQ0JpYjI5c0lGOWpkWEpJWVhOSmJuTjBJQ0FnSUNBOUlDaGZZM1Z5VW05M0xtbHVjM1J5"
    "ZFcxbGJuUWdQaUF3S1RzS0NpQWdJQ0JwWmlBb1gyTjFja2x6Vkc5dVpWQnZjblJoS1NCN0NpQWdJQ0Fn"
    "SUNBZ0x5OGdVMnhwWkdVZ2RHRnlaMlYwSU9LQWxDQm1hVzVrSUhCeWFXOXlJRkpGUVV3Z2RISnBaMmRs"
    "Y2lCbWIzSWdjMkZ0Y0d4bEwzQmxjbWx2WkNCamIyNTBaWGgwQ2lBZ0lDQWdJQ0FnZEc5dVpWTnNhV1Js"
    "VkdGeVoyVjBJRDBnWDJOMWNsSnZkeTV3WlhKcGIyUTdDaUFnSUNBZ0lDQWdhVzUwSUhOU0lEMGdjRzl6"
    "TG5KdmR5d2djMUFnUFNCd2IzTXVjMjl1WjFCdmN6c0tJQ0FnSUNBZ0lDQm1iM0lnS0dsdWRDQnNZaUE5"
    "SURFN0lHeGlJRHdnTVRJNE95QnNZaXNyS1NCN0NpQWdJQ0FnSUNBZ0lDQWdJSE5TTFMwN0NpQWdJQ0Fn"
    "SUNBZ0lDQWdJR2xtSUNoelVpQThJREFwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsbUlDaHpVQ0Er"
    "SURBcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0J6VUMwdE93b2dJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJSE5TSUQwZ2NHRjBVM1JoY25SU2IzZGJjMUJkSUNzZ0tIQmhkRkp2ZDA5bVpuTmxkRnR6"
    "VUNzeFhTQXRJSEJoZEZKdmQwOW1abk5sZEZ0elVGMHBJQzBnTVRzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUgwZ1pXeHpaU0I3SUdKeVpXRnJPeUI5Q2lBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0Fn"
    "VG05MFpTQndjbVYySUQwZ1oyVjBUbTkwWlNoelVDd2djMUlzSUdOb0tUc0tJQ0FnSUNBZ0lDQWdJQ0Fn"
    "WW05dmJDQndjbVYyU1hOVWIyNWxWSEpwWnlBOUlDZ29jSEpsZGk1bFptWmxZM1FnUFQwZ01IZ3pJSHg4"
    "SUhCeVpYWXVaV1ptWldOMElEMDlJREI0TlNrZ0ppWWdjSEpsZGk1d1pYSnBiMlFnUGlBd0tUc0tJQ0Fn"
    "SUNBZ0lDQWdJQ0FnTHk4Z1VtVmhiQ0IwY21sbloyVnlPaUJvWVhNZ2NHVnlhVzlrSUVGT1JDQnViM1Fn"
    "WVNCMGIyNWxMWEJ2Y25SaElIUmhjbWRsZENCeWIzY0tJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tIQnlaWFl1"
    "Y0dWeWFXOWtJRDRnTUNBbUppQWhjSEpsZGtselZHOXVaVlJ5YVdjcElIc0tJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDOHZJRVJsZEdWeWJXbHVaU0JwYm5OMGNuVnRaVzUwT2lCd2NtVm1aWElnY0hKbGRpNXBibk4w"
    "Y25WdFpXNTBMQ0JsYkhObElITmpZVzRnWm5WeWRHaGxjaUJtYjNJZ1kyOXVkR1Y0ZEFvZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0FnYVdZZ0tIQnlaWFl1YVc1emRISjFiV1Z1ZENBK0lEQXBJSHNLSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0FnSUNCMGNtbG5UbTkwWlNBOUlIQnlaWFk3SUhSeWFXZFNiM2NnUFNCelVqc2dkSEpw"
    "WjFCaGRDQTlJSE5RT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnZlNCbGJITmxJSHNLSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0FnSUNBdkx5QlFaWEpwYjJRdGIyNXNlU0J5YjNjZzRvQ1VJR1pwYm1RZ2FXNXpkSEox"
    "YldWdWRDQmpiMjUwWlhoMENpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdUbTkwWlNCeVpXRnNJRDBn"
    "Y0hKbGRqc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwYm5RZ2MxSXlJRDBnYzFJc0lITlFNaUE5"
    "SUhOUU93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR1p2Y2lBb2FXNTBJR3hpTWlBOUlERTdJR3hp"
    "TWlBOElERXlPRHNnYkdJeUt5c3BJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdjMUl5"
    "TFMwN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsbUlDaHpVaklnUENBd0tTQjdDaUFn"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwWmlBb2MxQXlJRDRnTUNrZ2V5QnpVREl0"
    "TFRzZ2MxSXlJRDBnY0dGMFUzUmhjblJTYjNkYmMxQXlYU0FySUNod1lYUlNiM2RQWm1aelpYUmJjMUF5"
    "S3pGZElDMGdjR0YwVW05M1QyWm1jMlYwVzNOUU1sMHBJQzBnTVRzZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdaV3h6WlNCaWNtVmhhenNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdmUW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQk9iM1JsSUhBeUlEMGdaMlYw"
    "VG05MFpTaHpVRElzSUhOU01pd2dZMmdwT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Jw"
    "WmlBb2NESXVhVzV6ZEhKMWJXVnVkQ0ErSURBcElIc2djbVZoYkM1cGJuTjBjblZ0Wlc1MElEMGdjREl1"
    "YVc1emRISjFiV1Z1ZERzZ1luSmxZV3M3SUgwS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQjlDaUFn"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnZEhKcFowNXZkR1VnUFNCeVpXRnNPeUIwY21sblVtOTNJRDBn"
    "YzFJN0lIUnlhV2RRWVhRZ1BTQnpVRHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJR0p5WldGck93b2dJQ0FnSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0FnZlFvZ0lDQWdmU0Js"
    "YkhObElHbG1JQ2doWDJOMWNrbHpVbVYwY21sbktTQjdDaUFnSUNBZ0lDQWdMeThnVG04Z2NHVnlhVzlr"
    "SU9LQWxDQmpiMjUwYVc1MVpTQndjbWx2Y2lCdWIzUmxJQ2h2Y2lCdWJ5QmhkV1JwYnlCcFppQnViM1Jv"
    "YVc1bklIQnlhVzl5S1M0S0lDQWdJQ0FnSUNBdkx5QmZZM1Z5U0dGelNXNXpkQ0IzYVhSb0lHNXZJSEJs"
    "Y21sdlpDQnBjeUJoSUc1dkxXOXdJR1p2Y2lCMGNtbG5aMlZ5SUhCMWNuQnZjMlZ6SUNoUVZDQnhkV2x5"
    "YXlrdUNpQWdJQ0FnSUNBZ2FXNTBJSE5TSUQwZ2NHOXpMbkp2ZHl3Z2MxQWdQU0J3YjNNdWMyOXVaMUJ2"
    "Y3pzS0lDQWdJQ0FnSUNCbWIzSWdLR2x1ZENCc1lpQTlJREU3SUd4aUlEd2dNVEk0T3lCc1lpc3JLU0I3"
    "Q2lBZ0lDQWdJQ0FnSUNBZ0lITlNMUzA3Q2lBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2h6VWlBOElEQXBJSHNL"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoelVDQStJREFwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQnpVQzB0T3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lITlNJRDBnY0dGMFUzUmhjblJT"
    "YjNkYmMxQmRJQ3NnS0hCaGRGSnZkMDltWm5ObGRGdHpVQ3N4WFNBdElIQmhkRkp2ZDA5bVpuTmxkRnR6"
    "VUYwcElDMGdNVHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJSDBnWld4elpTQjdJR0p5WldGck95QjlDaUFn"
    "SUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lDQWdUbTkwWlNCd2NtVjJJRDBnWjJWMFRtOTBaU2h6"
    "VUN3Z2MxSXNJR05vS1RzS0lDQWdJQ0FnSUNBZ0lDQWdZbTl2YkNCd2NtVjJTWE5VYjI1bFZISnBaeUE5"
    "SUNnb2NISmxkaTVsWm1abFkzUWdQVDBnTUhneklIeDhJSEJ5WlhZdVpXWm1aV04wSUQwOUlEQjROU2tn"
    "SmlZZ2NISmxkaTV3WlhKcGIyUWdQaUF3S1RzS0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0hCeVpYWXVjR1Z5"
    "YVc5a0lENGdNQ0FtSmlBaGNISmxka2x6Vkc5dVpWUnlhV2NwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUdsbUlDaHdjbVYyTG1sdWMzUnlkVzFsYm5RZ1BpQXdLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ2RISnBaMDV2ZEdVZ1BTQndjbVYyT3lCMGNtbG5VbTkzSUQwZ2MxSTdJSFJ5YVdkUVlYUWdQU0J6"
    "VURzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUgwZ1pXeHpaU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ1RtOTBaU0J5WldGc0lEMGdjSEpsZGpzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBiblFn"
    "YzFJeUlEMGdjMUlzSUhOUU1pQTlJSE5RT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHWnZjaUFv"
    "YVc1MElHeGlNaUE5SURFN0lHeGlNaUE4SURFeU9Ec2diR0l5S3lzcElIc0tJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0FnSUNBZ2MxSXlMUzA3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2xt"
    "SUNoelVqSWdQQ0F3S1NCN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFv"
    "YzFBeUlENGdNQ2tnZXlCelVESXRMVHNnYzFJeUlEMGdjR0YwVTNSaGNuUlNiM2RiYzFBeVhTQXJJQ2h3"
    "WVhSU2IzZFBabVp6WlhSYmMxQXlLekZkSUMwZ2NHRjBVbTkzVDJabWMyVjBXM05RTWwwcElDMGdNVHNn"
    "ZlFvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1pXeHpaU0JpY21WaGF6c0tJQ0Fn"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNCT2IzUmxJSEF5SUQwZ1oyVjBUbTkwWlNoelVESXNJSE5TTWl3Z1kyZ3BPd29nSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvY0RJdWFXNXpkSEoxYldWdWRDQStJREFwSUhzZ2NtVmhiQzVw"
    "Ym5OMGNuVnRaVzUwSUQwZ2NESXVhVzV6ZEhKMWJXVnVkRHNnWW5KbFlXczdJSDBLSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdkSEpwWjA1dmRHVWdQU0J5"
    "WldGc095QjBjbWxuVW05M0lEMGdjMUk3SUhSeWFXZFFZWFFnUFNCelVEc0tJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHSnlaV0ZyT3dvZ0lDQWdJQ0FnSUNBZ0lDQjlDaUFn"
    "SUNBZ0lDQWdmUW9nSUNBZ2ZTQmxiSE5sSUdsbUlDaGZZM1Z5U1hOU1pYUnlhV2NnSmlZZ0lWOWpkWEpJ"
    "WVhOSmJuTjBLU0I3Q2lBZ0lDQWdJQ0FnTHk4Z1VHVnlhVzlrTFc5dWJIa2djbVYwY21sbloyVnlJT0tB"
    "bENCbWFXNWtJR2x1YzNSeWRXMWxiblFnWTI5dWRHVjRkQ0FvYzJGdGNHeGxJR2x1YUdWeWFYUmxaQ0Jt"
    "Y205dElIQnlhVzl5SUhSeWFXZG5aWElwQ2lBZ0lDQWdJQ0FnYVc1MElITlNJRDBnY0c5ekxuSnZkeXdn"
    "YzFBZ1BTQndiM011YzI5dVoxQnZjenNLSUNBZ0lDQWdJQ0JtYjNJZ0tHbHVkQ0JzWWlBOUlERTdJR3hp"
    "SUR3Z01USTRPeUJzWWlzcktTQjdDaUFnSUNBZ0lDQWdJQ0FnSUhOU0xTMDdDaUFnSUNBZ0lDQWdJQ0Fn"
    "SUdsbUlDaHpVaUE4SURBcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2h6VUNBK0lEQXBJSHNL"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCelVDMHRPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUhOU0lEMGdjR0YwVTNSaGNuUlNiM2RiYzFCZElDc2dLSEJoZEZKdmQwOW1abk5sZEZ0elVDc3hYU0F0"
    "SUhCaGRGSnZkMDltWm5ObGRGdHpVRjBwSUMwZ01Uc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lIMGdaV3h6"
    "WlNCN0lHSnlaV0ZyT3lCOUNpQWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ1RtOTBaU0J3"
    "Y21WMklEMGdaMlYwVG05MFpTaHpVQ3dnYzFJc0lHTm9LVHNLSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLSEJ5"
    "WlhZdWFXNXpkSEoxYldWdWRDQStJREFwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUhSeWFXZE9iM1Js"
    "TG1sdWMzUnlkVzFsYm5RZ1BTQndjbVYyTG1sdWMzUnlkVzFsYm5RN0NpQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNCaWNtVmhhenNLSUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQXZMeUIw"
    "Y21sblVHRjBMM1J5YVdkU2IzY2djM1JoZVNCaGRDQmpkWEp5Wlc1MElISnZkeURpZ0pRZ2RHaHBjeUJK"
    "VXlCaElISmxkSEpwWjJkbGNnb2dJQ0FnZlFvZ0lDQWdMeThnWld4elpUb2dablZzYkNCMGNtbG5aMlZ5"
    "SUNod1pYSnBiMlFnS3lCcGJuTjBjblZ0Wlc1MExDQnVieUF6THpVcElPS0FsQ0IwY21sblRtOTBaU0Jo"
    "YkhKbFlXUjVJR052Y25KbFkzUUtDaUFnSUNCcFppQW9kSEpwWjA1dmRHVXVhVzV6ZEhKMWJXVnVkQ0E4"
    "UFNBd0lIeDhJSFJ5YVdkT2IzUmxMbWx1YzNSeWRXMWxiblFnUGlBek1TQjhmQ0IwY21sblRtOTBaUzV3"
    "WlhKcGIyUWdQRDBnTUNrS0lDQWdJQ0FnSUNCeVpYUjFjbTRnTUM0d093b0tJQ0FnSUZOaGJYQnNaVWx1"
    "Wm04Z2MyMXdJRDBnYzJGdGNHeGxjMXQwY21sblRtOTBaUzVwYm5OMGNuVnRaVzUwSUMwZ01WMDdDaUFn"
    "SUNCcFppQW9jMjF3TG14bGJtZDBhQ0E5UFNBd0tTQnlaWFIxY200Z01DNHdPd29LSUNBZ0lDOHZJRlJw"
    "WTJzdFltRnpaV1FnWld4aGNITmxaRG9nYVc1c2FXNWxJRWRTSUdOdmJYQjFkR0YwYVc5dUxDQnphMmx3"
    "SUc1aGJXVmtJR2x1ZEdWeWJXVmthV0YwWlhNS0lDQWdJR1pzYjJGMElHVnNZWEJ6WldRZ1BTQW9abXh2"
    "WVhRb1ptVjBZMmhVYVdOcktIQmhkRlJwWTJ0UFptWnpaWFJiY0c5ekxuTnZibWRRYjNOZEt5aHdiM011"
    "Y205M0xYQmhkRk4wWVhKMFVtOTNXM0J2Y3k1emIyNW5VRzl6WFNrcEtRb2dJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0FySUhCdmN5NTBhV05yQ2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDMGdabXh2"
    "WVhRb1ptVjBZMmhVYVdOcktIQmhkRlJwWTJ0UFptWnpaWFJiZEhKcFoxQmhkRjByS0hSeWFXZFNiM2N0"
    "Y0dGMFUzUmhjblJTYjNkYmRISnBaMUJoZEYwcEtTa3BDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "THlCVVNVTkxVMTlRUlZKZlUwVkRPd29nSUNBZ2FXWWdLR1ZzWVhCelpXUWdQQ0F3TGpBcElISmxkSFZ5"
    "YmlBd0xqQTdDZ29nSUNBZ2FXNTBJRjl3WTNRZ1BTQnBiblFvY0c5ekxuUnBZMnNwT3dvZ0lDQWdUbTkw"
    "WlNCZmNHTnlJRDBnWjJWMFRtOTBaU2h3YjNNdWMyOXVaMUJ2Y3l3Z2NHOXpMbkp2ZHl3Z1kyZ3BPd29L"
    "SUNBZ0lDOHZJT0tVZ09LVWdDQkRiMjFpYVc1bFpDQm1iM0ozWVhKa0lITmpZVzQ2SUhKbFluVnBiR1Fn"
    "Y0dsMFkyZ2dRVTVFSUhadmJIVnRaU0JtY205dElIUnlhV2RuWlhJZ2RHOGdZM1Z5Y21WdWRDRGlsSURp"
    "bElBS0lDQWdJR1pzYjJGMElHVm1abVZqZEdsMlpWQmxjbWx2WkNBOUlHWnNiMkYwS0hSeWFXZE9iM1Js"
    "TG5CbGNtbHZaQ2s3Q2lBZ0lDQm1iRzloZENCMFlYSm5aWFJRWlhKcGIyUWdJQ0FnUFNCbWJHOWhkQ2gw"
    "Y21sblRtOTBaUzV3WlhKcGIyUXBPd29nSUNBZ2FXNTBJQ0FnZG05c2RXMWxJQ0FnSUNBZ0lDQWdJRDBn"
    "YzIxd0xuWnZiSFZ0WlRzS0NpQWdJQ0F2THlCVWIyNWxMWEJ2Y25SaElIUnlhV2RuWlhJZ2NtOTNPaUIw"
    "YUdseklISnZkeUJqWVhKeWFXVnpJR0VnTTNoNEx6VjRlQ0J6Ykdsa1pTQjBZWEpuWlhRdUNpQWdJQ0F2"
    "THlCbFptWmxZM1JwZG1WUVpYSnBiMlFnYzNSaGVYTWdZWFFnZEdobElIQnlaWFpwYjNWeklIUnlhV2Ru"
    "WlhJbmN5QndaWEpwYjJRZ0tHRnNjbVZoWkhrZ2MyVjBJR0ZpYjNabENpQWdJQ0F2THlCbWNtOXRJSFJ5"
    "YVdkT2IzUmxMbkJsY21sdlpDazdJSE5zYVdSbElHRmpZM1Z0ZFd4aGRHVnpJSFJ2ZDJGeVpDQjBiMjVs"
    "VTJ4cFpHVlVZWEpuWlhRZ2IzWmxjaUJ5YjNkekxnb2dJQ0FnYVdZZ0tIUnZibVZUYkdsa1pWUmhjbWRs"
    "ZENBK0lEQXBJSHNLSUNBZ0lDQWdJQ0IwWVhKblpYUlFaWEpwYjJRZ1BTQm1iRzloZENoMGIyNWxVMnhw"
    "WkdWVVlYSm5aWFFwT3dvZ0lDQWdmUW9LSUNBZ0lDOHZJRUZ3Y0d4NUlIUnlhV2RuWlhJdGNtOTNJR1Zt"
    "Wm1WamRITTZJRU40ZUNBb2MyVjBJSFp2YkNrc0lFRjRlQzgyZUhnZ0tIWnZiQ0J6Ykdsa1pTQndZWEow"
    "YVdGc0wyWjFiR3dwTEFvZ0lDQWdMeThnUlVGNElDaG1hVzVsSUhadmJDQjFjQ0RpZ0pRZ2FXNXpkR0Z1"
    "ZENrc0lFVkNlQ0FvWm1sdVpTQjJiMndnWkc5M2JpRGlnSlFnYVc1emRHRnVkQ2tzSURWNGVDQW9kRzl1"
    "WlN0MmIyd2djMnhwWkdVcExnb2dJQ0FnYVdZZ0tIUnlhV2RPYjNSbExtVm1abVZqZENBOVBTQXdlRU1w"
    "SUhzS0lDQWdJQ0FnSUNCMmIyeDFiV1VnUFNCdGFXNG9kSEpwWjA1dmRHVXVjR0Z5WVcwc0lEWTBLVHNL"
    "SUNBZ0lIMGdaV3h6WlNCcFppQW9kSEpwWjA1dmRHVXVaV1ptWldOMElEMDlJREI0UlNrZ2V3b2dJQ0Fn"
    "SUNBZ0lDOHZJRVY0ZEdWdVpHVmtJR1ZtWm1WamRITTZJRVZCZUNCbWFXNWxJSFp2YkNCMWNDd2dSVUo0"
    "SUdacGJtVWdkbTlzSUdSdmQyNGdLR2x1YzNSaGJuUWdiMjRnZEdsamF5QXdLUW9nSUNBZ0lDQWdJR2x1"
    "ZENCZlpYTWdQU0FvZEhKcFowNXZkR1V1Y0dGeVlXMGdQajRnTkNrZ0ppQXdlRVk3Q2lBZ0lDQWdJQ0Fn"
    "YVc1MElGOWxkaUE5SUNCMGNtbG5UbTkwWlM1d1lYSmhiU0FnSUNBZ0lDQW1JREI0UmpzS0lDQWdJQ0Fn"
    "SUNCcFppQW9YMlZ6SUQwOUlEQjRRU2tnSUNBZ0lDQjJiMngxYldVZ1BTQmpiR0Z0Y0NoMmIyeDFiV1Vn"
    "S3lCZlpYWXNJREFzSURZMEtUc0tJQ0FnSUNBZ0lDQmxiSE5sSUdsbUlDaGZaWE1nUFQwZ01IaENLU0Iy"
    "YjJ4MWJXVWdQU0JqYkdGdGNDaDJiMngxYldVZ0xTQmZaWFlzSURBc0lEWTBLVHNLSUNBZ0lIMGdaV3h6"
    "WlNCcFppQW9kSEpwWjA1dmRHVXVaV1ptWldOMElEMDlJREI0UVNCOGZDQjBjbWxuVG05MFpTNWxabVps"
    "WTNRZ1BUMGdNSGcySUh4OElIUnlhV2RPYjNSbExtVm1abVZqZENBOVBTQXdlRFVwSUhzS0lDQWdJQ0Fn"
    "SUNBdkx5QXdlRFVnUFNCMGIyNWxLM1p2YkNCemJHbGtaVG9nY0dsMFkyZ2dhR0Z1Wkd4bFpDQmllU0F3"
    "ZURNdFpYRjFhWFpoYkdWdWRDQmliRzlqYXl3Z2RtOXNJSEJoY21GdElITmhiV1VnWVhNZ01IaEJDaUFn"
    "SUNBZ0lDQWdhVzUwSUY5emRTQTlJQ2gwY21sblRtOTBaUzV3WVhKaGJUNCtOQ2ttTUhoR0xDQmZjMlFn"
    "UFNCMGNtbG5UbTkwWlM1d1lYSmhiU1l3ZUVZN0NpQWdJQ0FnSUNBZ2FXNTBJRjl6ZEdWd0lEMGdLRjl6"
    "ZFQ0d0tTQS9JRjl6ZFNBNklDMWZjMlE3Q2lBZ0lDQWdJQ0FnYVdZZ0tIUnlhV2RRWVhRZ1BUMGdjRzl6"
    "TG5OdmJtZFFiM01nSmlZZ2RISnBaMUp2ZHlBOVBTQndiM011Y205M0tTQjdDaUFnSUNBZ0lDQWdJQ0Fn"
    "SUhadmJIVnRaU0E5SUdOc1lXMXdLSFp2YkhWdFpTQXJJRjl6ZEdWd0lDb2dYM0JqZEN3Z01Dd2dOalFw"
    "T3dvZ0lDQWdJQ0FnSUgwZ1pXeHpaU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0JmZEhNZ1BTQm1aWFJq"
    "YUZScFkyc29jR0YwVkdsamEwOW1abk5sZEZ0MGNtbG5VR0YwWFNzb2RISnBaMUp2ZHkxd1lYUlRkR0Z5"
    "ZEZKdmQxdDBjbWxuVUdGMFhTa3JNU2tLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBdElHWmxkR05v"
    "VkdsamF5aHdZWFJVYVdOclQyWm1jMlYwVzNSeWFXZFFZWFJkS3loMGNtbG5VbTkzTFhCaGRGTjBZWEow"
    "VW05M1czUnlhV2RRWVhSZEtTazdDaUFnSUNBZ0lDQWdJQ0FnSUhadmJIVnRaU0E5SUdOc1lXMXdLSFp2"
    "YkhWdFpTQXJJRjl6ZEdWd0lDb2dLRjkwY3kweEtTd2dNQ3dnTmpRcE93b2dJQ0FnSUNBZ0lIMEtJQ0Fn"
    "SUgwS0NpQWdJQ0F2THlEaWxJRGlsSUFnTUhoRlJDQnViM1JsSUdSbGJHRjVPaUJ6YTJsd0lHOTFkSEIx"
    "ZENCbWIzSWdZSEJoY21GdFlDQjBhV05yY3lCaFpuUmxjaUIwY21sbloyVnlJSEp2ZHlEaWxJRGlsSUFL"
    "SUNBZ0lDOHZJRlJvWlNCdWIzUmxJR1J2WlhOdUozUWdZV04wZFdGc2JIa2djM1JoY25RZ2RXNTBhV3dn"
    "ZEdsamF5QmdjR0Z5WVcxZ0lHOW1JSFJvWlNCMGNtbG5aMlZ5SUhKdmR5NEtJQ0FnSUM4dklFVmhjbXhw"
    "WlhJZ2RHaGhiaUIwYUdGMElPS0draUJ5WlhSMWNtNGdjMmxzWlc1alpUc2daV1ptWldOMGFYWmxJR1Zz"
    "WVhCelpXUWdhWE1nY21Wa2RXTmxaQzRLSUNBZ0lHbHVkQ0JmYm05MFpVUmxiR0Y1VkdsamEzTWdQU0F3"
    "T3dvZ0lDQWdhV1lnS0hSeWFXZE9iM1JsTG1WbVptVmpkQ0E5UFNBd2VFVWdKaVlnS0NoMGNtbG5UbTkw"
    "WlM1d1lYSmhiU0ErUGlBMEtTQW1JREI0UmlrZ1BUMGdNSGhFS1NCN0NpQWdJQ0FnSUNBZ1gyNXZkR1ZF"
    "Wld4aGVWUnBZMnR6SUQwZ2RISnBaMDV2ZEdVdWNHRnlZVzBnSmlBd2VFWTdDaUFnSUNBZ0lDQWdhV1ln"
    "S0hSeWFXZFFZWFFnUFQwZ2NHOXpMbk52Ym1kUWIzTWdKaVlnZEhKcFoxSnZkeUE5UFNCd2IzTXVjbTkz"
    "SUNZbUlGOXdZM1FnUENCZmJtOTBaVVJsYkdGNVZHbGphM01wQ2lBZ0lDQWdJQ0FnSUNBZ0lISmxkSFZ5"
    "YmlBd0xqQTdJQ0F2THlCaVpXWnZjbVVnWkdWc1lYbGxaQ0IwY21sbloyVnlDaUFnSUNBZ0lDQWdMeThn"
    "UVdaMFpYSWdaR1ZzWVhrNklITjFZblJ5WVdOMElHUmxiR0Y1SUhScFkydHpJR1p5YjIwZ1pXeGhjSE5s"
    "WkNCemJ5QjBhR1VnYm05MFpTQnpkR0Z5ZEhNZ1lYUWdabkpsYzJnZ2REMHdDaUFnSUNBZ0lDQWdaV3ho"
    "Y0hObFpDQTlJRzFoZUNnd0xqQXNJR1ZzWVhCelpXUWdMU0JtYkc5aGRDaGZibTkwWlVSbGJHRjVWR2xq"
    "YTNNcElDOGdWRWxEUzFOZlVFVlNYMU5GUXlrN0NpQWdJQ0I5Q2dvZ0lDQWdMeThnNHBTQTRwU0FJREI0"
    "T1hoNElITmhiWEJzWlNCdlptWnpaWFFnS0hSeWFXZG5aWElnY205M0lHOXViSGtwT2lCemRHRnlkQ0Jo"
    "ZENCd1lYSmhiU0FxSURJMU5pQnBiaUJ6WVcxd2JHVWdaR0YwWVNEaWxJRGlsSUFLSUNBZ0lHbHVkQ0Jm"
    "YzJGdGNHeGxUMlptYzJWMElEMGdNRHNLSUNBZ0lHbG1JQ2gwY21sblRtOTBaUzVsWm1abFkzUWdQVDBn"
    "TUhnNUlDWW1JSFJ5YVdkT2IzUmxMbkJoY21GdElENGdNQ2tnZXdvZ0lDQWdJQ0FnSUY5ellXMXdiR1ZQ"
    "Wm1aelpYUWdQU0IwY21sblRtOTBaUzV3WVhKaGJTQXFJREkxTmpzS0lDQWdJSDBLQ2lBZ0lDQXZMeURp"
    "bElEaWxJQWdWSEpwWjJkbGNpQnliM2NuY3lCd2FYUmphQ0J6Ykdsa1pTQmxabVpsWTNSeklDZ3hlSGd2"
    "TW5oNEtTRGlsSURpbElBS0lDQWdJQzh2SUVsbUlIUm9aU0IwY21sbloyVnlJSEp2ZHlCallYSnlhV1Zr"
    "SURGNGVDQW9jRzl5ZEdFZ2RYQXBJRzl5SURKNGVDQW9jRzl5ZEdFZ1pHOTNiaWtzSUhSb2IzTmxDaUFn"
    "SUNBdkx5QnpiR2xrWlhNZ2FHRndjR1Z1SUc5dUlIUnBZMnR6SURFdUxpaHpjR1ZsWkMweEtTQnZaaUIw"
    "YUdVZ2RISnBaMmRsY2lCeWIzY3VJQ0JYYUdWdUlIQnZjeUJwY3dvZ0lDQWdMeThnVUVGVFZDQjBhR1Vn"
    "ZEhKcFoyZGxjaUJ5YjNjc0lHRnNiQ0FvYzNCbFpXUXRNU2tnYjJZZ2RHaHZjMlVnZEdsamEzTWdhR0Yy"
    "WlNCamIyMXdiR1YwWldRdUNpQWdJQ0F2THlCWGFHVnVJSEJ2Y3lCcGN5QnZiaUIwYUdVZ2RISnBaMmRs"
    "Y2lCeWIzY2dhWFJ6Wld4bUxDQjBhR1VnSWtOMWNuSmxiblFnY205M0lIQmhjblJwWVd3Z2NHbDBZMmdL"
    "SUNBZ0lDOHZJR1ZtWm1WamRDSWdZbXh2WTJzZ1ltVnNiM2NnYUdGdVpHeGxjeUJwZENEaWdKUWdaRzl1"
    "SjNRZ1pHOTFZbXhsTFdGd2NHeDVJR2hsY21VdUNpQWdJQ0JwWmlBb0tIUnlhV2RRWVhRZ0lUMGdjRzl6"
    "TG5OdmJtZFFiM01nZkh3Z2RISnBaMUp2ZHlBaFBTQndiM011Y205M0tTQW1KZ29nSUNBZ0lDQWdJQ2gw"
    "Y21sblRtOTBaUzVsWm1abFkzUWdQVDBnTUhneElIeDhJSFJ5YVdkT2IzUmxMbVZtWm1WamRDQTlQU0F3"
    "ZURJcElDWW1JSFJ5YVdkT2IzUmxMbkJoY21GdElENGdNQ2tnZXdvZ0lDQWdJQ0FnSUdsdWRDQmZkSEpU"
    "WjNJZ1BTQndZWFJVYVdOclQyWm1jMlYwVzNSeWFXZFFZWFJkSUNzZ0tIUnlhV2RTYjNjZ0xTQndZWFJU"
    "ZEdGeWRGSnZkMXQwY21sblVHRjBYU2s3Q2lBZ0lDQWdJQ0FnYVc1MElGOTBjbE53WkNBOUlHWmxkR05v"
    "VkdsamF5aGZkSEpUWjNJZ0t5QXhLU0F0SUdabGRHTm9WR2xqYXloZmRISlRaM0lwT3dvZ0lDQWdJQ0Fn"
    "SUdsdWRDQmZkSEpVYVdOcmN5QTlJRjkwY2xOd1pDQXRJREU3SUNBdkx5QmhiR3dnY0c5emRDMTBhV05y"
    "TFRBZ2RHbGphM01nYjJZZ2RISnBaMmRsY2lCeWIzY0tJQ0FnSUNBZ0lDQnBaaUFvZEhKcFowNXZkR1V1"
    "WldabVpXTjBJRDA5SURCNE1Ta0tJQ0FnSUNBZ0lDQWdJQ0FnWldabVpXTjBhWFpsVUdWeWFXOWtJRDBn"
    "YldGNEtERXhNeTR3TENCbFptWmxZM1JwZG1WUVpYSnBiMlFnTFNCbWJHOWhkQ2gwY21sblRtOTBaUzV3"
    "WVhKaGJTQXFJRjkwY2xScFkydHpLU2s3Q2lBZ0lDQWdJQ0FnWld4elpTQWdMeThnTUhneUNpQWdJQ0Fn"
    "SUNBZ0lDQWdJR1ZtWm1WamRHbDJaVkJsY21sdlpDQTlJRzFwYmlnNE5UWXVNQ3dnWldabVpXTjBhWFps"
    "VUdWeWFXOWtJQ3NnWm14dllYUW9kSEpwWjA1dmRHVXVjR0Z5WVcwZ0tpQmZkSEpVYVdOcmN5a3BPd29n"
    "SUNBZ2ZRb0tJQ0FnSUM4dklGUnlZV05ySUd4aGMzUWdkRzl1WlMxd2IzSjBZU0J5WVhSbElDaG1iM0ln"
    "WldabVpXTjBJRFVnZEc4Z2FXNW9aWEpwZENrdUlDQkpibWwwYVdGc2FYcGxaQ0JtY205dENpQWdJQ0F2"
    "THlCMGNtbG5aMlZ5SUhKdmR5ZHpJRE40ZUNCd1lYSmhiVHNnZFhCa1lYUmxaQ0JpZVNCbWIzSjNZWEpr"
    "SUhOallXNGdZWE1nYVhRZ2QyRnNhM01nY0dGemRDQXplSGdnY205M2N5NEtJQ0FnSUdsdWRDQmZiR0Z6"
    "ZEZSUVVtRjBaU0E5SURBN0NpQWdJQ0JwWmlBb2RISnBaMDV2ZEdVdVpXWm1aV04wSUQwOUlEQjRNeUFt"
    "SmlCMGNtbG5UbTkwWlM1d1lYSmhiU0ErSURBcElGOXNZWE4wVkZCU1lYUmxJRDBnZEhKcFowNXZkR1V1"
    "Y0dGeVlXMDdDZ29nSUNBZ0x5OGdSbTl5ZDJGeVpDQnpZMkZ1T2lCeWIzZHpJRk5VVWtsRFZFeFpJR0ps"
    "ZEhkbFpXNGdkSEpwWjJkbGNpQmhibVFnWTNWeWNtVnVkQW9nSUNBZ2FXWWdLSFJ5YVdkUVlYUWdJVDBn"
    "Y0c5ekxuTnZibWRRYjNNZ2ZId2dkSEpwWjFKdmR5QWhQU0J3YjNNdWNtOTNLU0I3Q2lBZ0lDQWdJQ0Fn"
    "YVc1MElGOW1jQ0E5SUhSeWFXZFFZWFFzSUY5bWNpQTlJSFJ5YVdkU2IzY2dLeUF4T3dvZ0lDQWdJQ0Fn"
    "SUdsbUlDaGZabklnUGowZ2NHRjBVM1JoY25SU2IzZGJYMlp3WFNBcklDaHdZWFJTYjNkUFptWnpaWFJi"
    "WDJad0t6RmRJQzBnY0dGMFVtOTNUMlptYzJWMFcxOW1jRjBwS1NCN0NpQWdJQ0FnSUNBZ0lDQWdJRjlt"
    "Y0Nzck95QmZabklnUFNBb1gyWndJRHdnVTA5T1IxOU1SVTVIVkVncElEOGdjR0YwVTNSaGNuUlNiM2Ri"
    "WDJad1hTQTZJREE3Q2lBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUdadmNpQW9hVzUwSUY5bWFTQTlJREE3"
    "SUY5bWFTQThJREV5T0RzZ1gyWnBLeXNwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5bWNDQStJSEJ2"
    "Y3k1emIyNW5VRzl6SUh4OElDaGZabkFnUFQwZ2NHOXpMbk52Ym1kUWIzTWdKaVlnWDJaeUlENDlJSEJ2"
    "Y3k1eWIzY3BLU0JpY21WaGF6c0tJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOW1jQ0ErUFNCVFQwNUhYMHhG"
    "VGtkVVNDa2dZbkpsWVdzN0NpQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZlpuSWdQajBnY0dGMFUzUmhjblJT"
    "YjNkYlgyWndYU0FySUNod1lYUlNiM2RQWm1aelpYUmJYMlp3S3pGZElDMGdjR0YwVW05M1QyWm1jMlYw"
    "VzE5bWNGMHBLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JmWm5Bckt6c2dYMlp5SUQwZ0tGOW1jQ0E4"
    "SUZOUFRrZGZURVZPUjFSSUtTQS9JSEJoZEZOMFlYSjBVbTkzVzE5bWNGMGdPaUF3T3dvZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0FnWTI5dWRHbHVkV1U3Q2lBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0Fn"
    "VG05MFpTQmZabTRnUFNCblpYUk9iM1JsS0Y5bWNDd2dYMlp5TENCamFDazdDaUFnSUNBZ0lDQWdJQ0Fn"
    "SUM4dklFRk9XU0J5YjNjZ2QybDBhQ0J3WlhKcGIyUWdQaUF3SUdGdVpDQmxabVpsWTNRZ2JtOTBJRE12"
    "TlNCcGN5QmhJSEpsWVd3Z1VrVlVVa2xIUjBWU0NpQWdJQ0FnSUNBZ0lDQWdJQzh2SUhSb1lYUWdaVzVr"
    "Y3lCMGFHVWdabTl5ZDJGeVpDQnpZMkZ1SUNodVpYaDBJR2RsZEVOb1lXNXVaV3hQZFhSd2RYUWdZMkZz"
    "YkNCb1lXNWtiR1Z6SUdsMEtTNEtJQ0FnSUNBZ0lDQWdJQ0FnWW05dmJDQmZabTVKYzFSdmJtVlVjbWxu"
    "SUQwZ0tDaGZabTR1WldabVpXTjBJRDA5SURCNE15QjhmQ0JmWm00dVpXWm1aV04wSUQwOUlEQjROU2tn"
    "SmlZZ1gyWnVMbkJsY21sdlpDQStJREFwT3dvZ0lDQWdJQ0FnSUNBZ0lDQmliMjlzSUY5bWJrbHpVbVYw"
    "Y21sbklDQWdQU0FvWDJadUxuQmxjbWx2WkNBK0lEQWdKaVlnSVY5bWJrbHpWRzl1WlZSeWFXY3BPd29n"
    "SUNBZ0lDQWdJQ0FnSUNCcFppQW9YMlp1U1hOU1pYUnlhV2NwSUdKeVpXRnJPd29nSUNBZ0lDQWdJQ0Fn"
    "SUNBdkx5QlViMjVsTFhCdmNuUmhJSFJoY21kbGREb2djR2wwWTJnZ2MyeHBaR1Z6SUhSdmQyRnlaQ0Jm"
    "Wm00dWNHVnlhVzlrSUc5MlpYSWdjbVZ0WVdsdWFXNW5JSEp2ZDNNdUNpQWdJQ0FnSUNBZ0lDQWdJR2xt"
    "SUNoZlptNUpjMVJ2Ym1WVWNtbG5LU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0IwWVhKblpYUlFaWEpw"
    "YjJRZ1BTQm1iRzloZENoZlptNHVjR1Z5YVc5a0tUc0tJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0Fn"
    "SUNBZ0lDQXZMeUJVY21GamF5QnNZWE4wSURONGVDQnlZWFJsSUdadmNpQmxabVpsWTNRZ05TQjBieUJw"
    "Ym1obGNtbDBDaUFnSUNBZ0lDQWdJQ0FnSUdsbUlDaGZabTR1WldabVpXTjBJRDA5SURCNE15QW1KaUJm"
    "Wm00dWNHRnlZVzBnUGlBd0tTQmZiR0Z6ZEZSUVVtRjBaU0E5SUY5bWJpNXdZWEpoYlRzS0NpQWdJQ0Fn"
    "SUNBZ0lDQWdJR2x1ZENCZmMyZHlJQ0FnUFNCd1lYUlVhV05yVDJabWMyVjBXMTltY0YwZ0t5QW9YMlp5"
    "SUMwZ2NHRjBVM1JoY25SU2IzZGJYMlp3WFNrN0NpQWdJQ0FnSUNBZ0lDQWdJR2x1ZENCZlpuVnNiQ0Fn"
    "UFNCbVpYUmphRlJwWTJzb1gzTm5jaUFySURFcElDMGdabVYwWTJoVWFXTnJLRjl6WjNJcElDMGdNVHNn"
    "SUM4dklIUnBZMnR6SURFdUxuTndaV1ZrTFRFS0NpQWdJQ0FnSUNBZ0lDQWdJQzh2SUZCcGRHTm9JR1Zt"
    "Wm1WamRITUtJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOW1iaTVsWm1abFkzUWdQVDBnTUhneEtRb2dJQ0Fn"
    "SUNBZ0lDQWdJQ0FnSUNBZ1pXWm1aV04wYVhabFVHVnlhVzlrSUQwZ2JXRjRLREV4TXk0d0xDQmxabVps"
    "WTNScGRtVlFaWEpwYjJRZ0xTQm1iRzloZENoZlptNHVjR0Z5WVcwZ0tpQmZablZzYkNrcE93b2dJQ0Fn"
    "SUNBZ0lDQWdJQ0JsYkhObElHbG1JQ2hmWm00dVpXWm1aV04wSUQwOUlEQjRNaWtLSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJR1ZtWm1WamRHbDJaVkJsY21sdlpDQTlJRzFwYmlnNE5UWXVNQ3dnWldabVpXTjBhWFps"
    "VUdWeWFXOWtJQ3NnWm14dllYUW9YMlp1TG5CaGNtRnRJQ29nWDJaMWJHd3BLVHNLSUNBZ0lDQWdJQ0Fn"
    "SUNBZ1pXeHpaU0JwWmlBb1gyWnVMbVZtWm1WamRDQTlQU0F3ZURNcElIc0tJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDOHZJRlJ2Ym1VZ2NHOXlkR0VnNG9DVUlIVnpaWE1nYVhSeklHOTNiaUJ3WVhKaGJTQmhjeUJ6"
    "Ykdsa1pTQnlZWFJsQ2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1pXWm1aV04wYVhabFVHVnlhVzlr"
    "SUR3Z2RHRnlaMlYwVUdWeWFXOWtLUW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdWbVptVmpkR2wy"
    "WlZCbGNtbHZaQ0E5SUcxcGJpaDBZWEpuWlhSUVpYSnBiMlFzSUdWbVptVmpkR2wyWlZCbGNtbHZaQ0Fy"
    "SUdac2IyRjBLRjltYmk1d1lYSmhiU0FxSUY5bWRXeHNLU2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Js"
    "YkhObElHbG1JQ2hsWm1abFkzUnBkbVZRWlhKcGIyUWdQaUIwWVhKblpYUlFaWEpwYjJRcENpQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdaV1ptWldOMGFYWmxVR1Z5YVc5a0lEMGdiV0Y0S0hSaGNtZGxkRkJs"
    "Y21sdlpDd2daV1ptWldOMGFYWmxVR1Z5YVc5a0lDMGdabXh2WVhRb1gyWnVMbkJoY21GdElDb2dYMlox"
    "Ykd3cEtUc0tJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNBZ0lDQmxiSE5sSUdsbUlDaGZabTR1"
    "WldabVpXTjBJRDA5SURCNE5Ta2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdMeThnUTI5dWRHbHVkV1Vn"
    "ZEc5dVpTQndiM0owWVNEaWdKUWdkWE5sY3lCTVFWTlVJRE40ZUNCeVlYUmxJQ2h1YjNRZ1gyWnVMbkJo"
    "Y21GdElTa0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmYkdGemRGUlFVbUYwWlNBK0lEQXBJSHNL"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9aV1ptWldOMGFYWmxVR1Z5YVc5a0lEd2dkR0Z5"
    "WjJWMFVHVnlhVzlrS1FvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JsWm1abFkzUnBkbVZR"
    "WlhKcGIyUWdQU0J0YVc0b2RHRnlaMlYwVUdWeWFXOWtMQ0JsWm1abFkzUnBkbVZRWlhKcGIyUWdLeUJt"
    "Ykc5aGRDaGZiR0Z6ZEZSUVVtRjBaU0FxSUY5bWRXeHNLU2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ1pXeHpaU0JwWmlBb1pXWm1aV04wYVhabFVHVnlhVzlrSUQ0Z2RHRnlaMlYwVUdWeWFXOWtLUW9n"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmxabVpsWTNScGRtVlFaWEpwYjJRZ1BTQnRZWGdv"
    "ZEdGeVoyVjBVR1Z5YVc5a0xDQmxabVpsWTNScGRtVlFaWEpwYjJRZ0xTQm1iRzloZENoZmJHRnpkRlJR"
    "VW1GMFpTQXFJRjltZFd4c0tTazdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdJQ0Fn"
    "SUgwS0lDQWdJQ0FnSUNBZ0lDQWdMeThnVm05c2RXMWxJR1ZtWm1WamRITUtJQ0FnSUNBZ0lDQWdJQ0Fn"
    "Wld4elpTQnBaaUFvWDJadUxtVm1abVZqZENBOVBTQXdlRU1wQ2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Iy"
    "YjJ4MWJXVWdQU0J0YVc0b1gyWnVMbkJoY21GdExDQTJOQ2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lHVnNjMlVn"
    "YVdZZ0tGOW1iaTVsWm1abFkzUWdQVDBnTUhoQklIeDhJRjltYmk1bFptWmxZM1FnUFQwZ01IZzJLU0I3"
    "Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwYm5RZ1gzWjFJRDBnS0Y5bWJpNXdZWEpoYlQ0K05Da21NSGhH"
    "TENCZmRtUWdQU0JmWm00dWNHRnlZVzBtTUhoR093b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2RtOXNkVzFs"
    "SUQwZ1kyeGhiWEFvZG05c2RXMWxJQ3NnS0Y5MmRUNHdQMTkyZFRvdFgzWmtLU0FxSUY5bWRXeHNMQ0F3"
    "TENBMk5DazdDaUFnSUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lDQWdaV3h6WlNCcFppQW9YMlp1"
    "TG1WbVptVmpkQ0E5UFNBd2VFVXBJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQzh2SUVWQmVDQm1hVzVs"
    "SUhadmJDQjFjQ3dnUlVKNElHWnBibVVnZG05c0lHUnZkMjRnS0dsdWMzUmhiblFnY0dWeUlISnZkeWtL"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2x1ZENCZlpYTWdQU0FvWDJadUxuQmhjbUZ0SUQ0K0lEUXBJQ1ln"
    "TUhoR093b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjlsZGlBOUlDQmZabTR1Y0dGeVlXMGdJQ0Fn"
    "SUNBZ0ppQXdlRVk3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1gyVnpJRDA5SURCNFFTa2dJQ0Fn"
    "SUNCMmIyeDFiV1VnUFNCamJHRnRjQ2gyYjJ4MWJXVWdLeUJmWlhZc0lEQXNJRFkwS1RzS0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0FnSUdWc2MyVWdhV1lnS0Y5bGN5QTlQU0F3ZUVJcElIWnZiSFZ0WlNBOUlHTnNZVzF3"
    "S0hadmJIVnRaU0F0SUY5bGRpd2dNQ3dnTmpRcE93b2dJQ0FnSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0Fn"
    "SUNBZ0lDOHZJREI0TlNCaGJITnZJR0Z3Y0d4cFpYTWdkR2hsSUhadmJIVnRaU0J6Ykdsa1pTQndiM0ow"
    "YVc5dUlDaG9hV2RvSUc1cFltSnNaU0E5SUhWd0xDQnNiM2NnUFNCa2IzZHVLUW9nSUNBZ0lDQWdJQ0Fn"
    "SUNCcFppQW9YMlp1TG1WbVptVmpkQ0E5UFNBd2VEVXBJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2x1"
    "ZENCZmRuVWdQU0FvWDJadUxuQmhjbUZ0UGo0MEtTWXdlRVlzSUY5MlpDQTlJRjltYmk1d1lYSmhiU1l3"
    "ZUVZN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCMmIyeDFiV1VnUFNCamJHRnRjQ2gyYjJ4MWJXVWdLeUFv"
    "WDNaMVBqQS9YM1oxT2kxZmRtUXBJQ29nWDJaMWJHd3NJREFzSURZMEtUc0tJQ0FnSUNBZ0lDQWdJQ0Fn"
    "ZlFvZ0lDQWdJQ0FnSUNBZ0lDQmZabklyS3pzS0lDQWdJQ0FnSUNBZ0lDQWdMeThnUVdSMllXNWpaU0Iw"
    "YnlCdVpYaDBJSE52Ym1jZ2NHOXphWFJwYjI0Z2QyaGxiaUIzWlNkMlpTQmxlR2hoZFhOMFpXUWdkR2hw"
    "Y3lCd1lYUjBaWEp1SjNNZ2NtOTNjd29nSUNBZ0lDQWdJQ0FnSUNCcFppQW9YMlp5SUQ0OUlIQmhkRk4w"
    "WVhKMFVtOTNXMTltY0YwZ0t5QW9jR0YwVW05M1QyWm1jMlYwVzE5bWNDc3hYU0F0SUhCaGRGSnZkMDlt"
    "Wm5ObGRGdGZabkJkS1NrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1gyWndLeXM3Q2lBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0JmWm5JZ1BTQW9YMlp3SUR3Z1UwOU9SMTlNUlU1SFZFZ3BJRDhnY0dGMFUzUmhjblJT"
    "YjNkYlgyWndYU0E2SURBN0NpQWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0I5Q2dvZ0lDQWdJQ0Fn"
    "SUM4dklFTjFjbkpsYm5RZ2NtOTNJSEJoY25ScFlXd2dLRzV2YmkxMGNtbG5aMlZ5SUhKdmR5QnZibXg1"
    "SU9LQWxDQjBjbWxuWjJWeUlHaGhibVJzWldRZ1lXSnZkbVVwQ2lBZ0lDQWdJQ0FnYVdZZ0tGOXdZM0l1"
    "YVc1emRISjFiV1Z1ZENBOFBTQXdJQ1ltSUY5d1kzSXVjR1Z5YVc5a0lEdzlJREFwSUhzS0lDQWdJQ0Fn"
    "SUNBZ0lDQWdhV1lnS0Y5d1kzSXVaV1ptWldOMElEMDlJREI0UXlrS0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUhadmJIVnRaU0E5SUcxcGJpaGZjR055TG5CaGNtRnRMQ0EyTkNrN0NpQWdJQ0FnSUNBZ0lDQWdJR1Zz"
    "YzJVZ2FXWWdLRjl3WTNJdVpXWm1aV04wSUQwOUlEQjRRU0I4ZkNCZmNHTnlMbVZtWm1WamRDQTlQU0F3"
    "ZURZcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0JmZG5VZ1BTQW9YM0JqY2k1d1lYSmhiVDQr"
    "TkNrbU1IaEdMQ0JmZG1RZ1BTQmZjR055TG5CaGNtRnRKakI0UmpzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUhadmJIVnRaU0E5SUdOc1lXMXdLSFp2YkhWdFpTQXJJQ2hmZG5VK01EOWZkblU2TFY5MlpDa2dLaUJm"
    "Y0dOMExDQXdMQ0EyTkNrN0NpQWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ1pXeHpaU0Jw"
    "WmlBb1gzQmpjaTVsWm1abFkzUWdQVDBnTUhoRktTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBiblFn"
    "WDJWeklEMGdLRjl3WTNJdWNHRnlZVzBnUGo0Z05Da2dKaUF3ZUVZN0NpQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNCcGJuUWdYMlYySUQwZ0lGOXdZM0l1Y0dGeVlXMGdJQ0FnSUNBZ0ppQXdlRVk3Q2lBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0JwWmlBb1gyVnpJRDA5SURCNFFTa2dJQ0FnSUNCMmIyeDFiV1VnUFNCamJHRnRjQ2gy"
    "YjJ4MWJXVWdLeUJmWlhZc0lEQXNJRFkwS1RzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdWc2MyVWdhV1ln"
    "S0Y5bGN5QTlQU0F3ZUVJcElIWnZiSFZ0WlNBOUlHTnNZVzF3S0hadmJIVnRaU0F0SUY5bGRpd2dNQ3dn"
    "TmpRcE93b2dJQ0FnSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmY0dOeUxtVm1abVZq"
    "ZENBOVBTQXdlRFVwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUM4dklEQjROU0IyYjJ3dGMyeHBaR1Vn"
    "Y0c5eWRHbHZiaUJ2YmlCamRYSnlaVzUwSUhKdmR3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjky"
    "ZFNBOUlDaGZjR055TG5CaGNtRnRQajQwS1NZd2VFWXNJRjkyWkNBOUlGOXdZM0l1Y0dGeVlXMG1NSGhH"
    "T3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnZG05c2RXMWxJRDBnWTJ4aGJYQW9kbTlzZFcxbElDc2dLRjky"
    "ZFQ0d1AxOTJkVG90WDNaa0tTQXFJRjl3WTNRc0lEQXNJRFkwS1RzS0lDQWdJQ0FnSUNBZ0lDQWdmUW9n"
    "SUNBZ0lDQWdJSDBLSUNBZ0lIMEtDaUFnSUNBdkx5QkRkWEp5Wlc1MElISnZkeUJ3WVhKMGFXRnNJSEJw"
    "ZEdOb0lHVm1abVZqZENBb1lYQndiR2xsY3lCbGRtVnVJRzl1SUhSeWFXZG5aWElnY205M0tTNEtJQ0Fn"
    "SUM4dklGVnpaU0JqYjI1MGFXNTFiM1Z6SUhCdmN5NTBhV05ySU9LQWxDQmlkWFFnWTJGd0lHbDBJR0Yw"
    "SUNoemNHVmxaQzB4S1NCemJ5QjBhR1VnWTI5dWRISnBZblYwYVc5dUNpQWdJQ0F2THlCaGRDQjBhR1Vn"
    "YkdGemRDQnpZVzF3YkdVZ2IyWWdkR2hwY3lCeWIzY2daWGhoWTNSc2VTQnRZWFJqYUdWeklIZG9ZWFFn"
    "ZEdobElHWnZjbmRoY21RZ2MyTmhiZ29nSUNBZ0x5OGdkMmxzYkNCMWMyVWdabTl5SUhSb2FYTWdjbTkz"
    "SUc5dVkyVWdhWFFnWW1WamIyMWxjeUJoSUNKamIyMXdiR1YwWldRaUlISnZkeTRnSUZkcGRHaHZkWFFn"
    "ZEdobENpQWdJQ0F2THlCallYQXNJSEJ2Y3k1MGFXTnJJR0Z3Y0hKdllXTm9aWE1nWUhOd1pXVmtZQ0Jo"
    "ZENCMGFHVWdjbTkzSUdKdmRXNWtZWEo1SUhkb2FXeGxJSFJvWlNCbWIzSjNZWEprQ2lBZ0lDQXZMeUJ6"
    "WTJGdUlIVnpaWE1nWUhOd1pXVmtMVEZnTENCd2NtOWtkV05wYm1jZ1lTQitNUzEwYVdOcklHSmhZMnQz"
    "WVhKa0lIQmxjbWx2WkNCcWRXMXdJRDBnWTJ4cFkyc3VDaUFnSUNBdkx5QlBibXg1SUhCaGVTQjBhR1Vn"
    "Wm1WMFkyaFVhV05ySUdOdmMzUWdkMmhsYmlCaElIQnBkR05vSUdWbVptVmpkQ0JwY3lCaFkzUjFZV3hz"
    "ZVNCd2NtVnpaVzUwTGdvZ0lDQWdMeThnUTNWeWNtVnVkQ0J5YjNjZ2NHRnlkR2xoYkNCd2FYUmphQ0Js"
    "Wm1abFkzUWdLR0Z3Y0d4cFpYTWdiMjRnZEhKcFoyZGxjaUJ5YjNjZ1QxSWdZMjl1ZEdsdWRXRjBhVzl1"
    "SUhKdmR5a3VDaUFnSUNBdkx5QkdiM0lnTVhoNEx6SjRlRG9nWVd4M1lYbHpJR0Z3Y0d4cFpYTWdiMjRn"
    "WTNWeWNtVnVkQ0J5YjNjdUNpQWdJQ0F2THlCR2IzSWdNM2g0THpWNGVEb2dZWEJ3YkdsbGN5QjNhR1Z1"
    "WlhabGNpQmpkWEp5Wlc1MElISnZkeUJqWVhKeWFXVnpJSFJvWlNCbFptWmxZM1FnNG9DVUlHSnZkR2dL"
    "SUNBZ0lDOHZJQ0FnWTI5dWRHbHVkV0YwYVc5dUlISnZkM01nS0hCbGNtbHZaRDA5TUNrZ1FVNUVJSFJ2"
    "Ym1VdGNHOXlkR0VnZEdGeVoyVjBJSEp2ZDNNZ0tIQmxjbWx2WkQ0d0tTNEtJQ0FnSUM4dklDQWdWR2hs"
    "SUhSaGNtZGxkRkJsY21sdlpDQjNZWE1nWVd4eVpXRmtlU0J6WlhRZ1lXSnZkbVVnS0dWcGRHaGxjaUJt"
    "Y205dElIUnlhV2RPYjNSbElHOXlDaUFnSUNBdkx5QWdJSFJ2Ym1WVGJHbGtaVlJoY21kbGRDazdJSFJv"
    "YVhNZ1lteHZZMnNnWkc5bGN5QjBhR1VnY0dWeUxYUnBZMnNnWVdOamRXMTFiR0YwYVc5dUlIUnZkMkZ5"
    "WkNCcGRDNEtJQ0FnSUdsbUlDaGZjR055TG1WbVptVmpkQ0E5UFNBd2VERWdmSHdnWDNCamNpNWxabVps"
    "WTNRZ1BUMGdNSGd5SUh4OENpQWdJQ0FnSUNBZ1gzQmpjaTVsWm1abFkzUWdQVDBnTUhneklIeDhJRjl3"
    "WTNJdVpXWm1aV04wSUQwOUlEQjROU2tnZXdvZ0lDQWdJQ0FnSUdsdWRDQmZjMmR5WDJOMWNpQTlJSEJo"
    "ZEZScFkydFBabVp6WlhSYmNHOXpMbk52Ym1kUWIzTmRJQ3NnS0hCdmN5NXliM2NnTFNCd1lYUlRkR0Z5"
    "ZEZKdmQxdHdiM011YzI5dVoxQnZjMTBwT3dvZ0lDQWdJQ0FnSUdac2IyRjBJRjl3ZEdZZ1BTQnRhVzRv"
    "Y0c5ekxuUnBZMnNzQ2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JtYkc5aGRDaG1aWFJq"
    "YUZScFkyc29YM05uY2w5amRYSWdLeUF4S1NBdElHWmxkR05vVkdsamF5aGZjMmR5WDJOMWNpa2dMU0F4"
    "S1NrN0NpQWdJQ0FnSUNBZ2FXWWdLRjl3WTNJdVpXWm1aV04wSUQwOUlEQjRNU2tLSUNBZ0lDQWdJQ0Fn"
    "SUNBZ1pXWm1aV04wYVhabFVHVnlhVzlrSUQwZ2JXRjRLREV4TXk0d0xDQmxabVpsWTNScGRtVlFaWEpw"
    "YjJRZ0xTQm1iRzloZENoZmNHTnlMbkJoY21GdEtTQXFJRjl3ZEdZcE93b2dJQ0FnSUNBZ0lHVnNjMlVn"
    "YVdZZ0tGOXdZM0l1WldabVpXTjBJRDA5SURCNE1pa0tJQ0FnSUNBZ0lDQWdJQ0FnWldabVpXTjBhWFps"
    "VUdWeWFXOWtJRDBnYldsdUtEZzFOaTR3TENCbFptWmxZM1JwZG1WUVpYSnBiMlFnS3lCbWJHOWhkQ2hm"
    "Y0dOeUxuQmhjbUZ0S1NBcUlGOXdkR1lwT3dvZ0lDQWdJQ0FnSUdWc2MyVWdhV1lnS0Y5d1kzSXVaV1pt"
    "WldOMElEMDlJREI0TXlrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0F2THlCVWIyNWxJSEJ2Y25SaElPS0FsQ0Ix"
    "YzJWeklHbDBjeUJ2ZDI0Z2NHRnlZVzBnWVhNZ2MyeHBaR1VnY21GMFpRb2dJQ0FnSUNBZ0lDQWdJQ0Jw"
    "WmlBb1pXWm1aV04wYVhabFVHVnlhVzlrSUR3Z2RHRnlaMlYwVUdWeWFXOWtLUW9nSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdaV1ptWldOMGFYWmxVR1Z5YVc5a0lEMGdiV2x1S0hSaGNtZGxkRkJsY21sdlpDd2daV1pt"
    "WldOMGFYWmxVR1Z5YVc5a0lDc2dabXh2WVhRb1gzQmpjaTV3WVhKaGJTa2dLaUJmY0hSbUtUc0tJQ0Fn"
    "SUNBZ0lDQWdJQ0FnWld4elpTQnBaaUFvWldabVpXTjBhWFpsVUdWeWFXOWtJRDRnZEdGeVoyVjBVR1Z5"
    "YVc5a0tRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1pXWm1aV04wYVhabFVHVnlhVzlrSUQwZ2JXRjRLSFJo"
    "Y21kbGRGQmxjbWx2WkN3Z1pXWm1aV04wYVhabFVHVnlhVzlrSUMwZ1pteHZZWFFvWDNCamNpNXdZWEpo"
    "YlNrZ0tpQmZjSFJtS1RzS0lDQWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ1pXeHpaU0I3SUNBdkx5QXdlRFVn"
    "NG9DVUlHTnZiblJwYm5WbElIUnZibVVnY0c5eWRHRWdkWE5wYm1jZ2JHRnpkQ0F6ZUhnZ2NtRjBaU0Fv"
    "Y0dGeVlXMGdhWE1nZG05c0xYTnNhV1JsSUc5dWJIa3BDaUFnSUNBZ0lDQWdJQ0FnSUdsbUlDaGZiR0Z6"
    "ZEZSUVVtRjBaU0ErSURBcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hsWm1abFkzUnBkbVZR"
    "WlhKcGIyUWdQQ0IwWVhKblpYUlFaWEpwYjJRcENpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdaV1pt"
    "WldOMGFYWmxVR1Z5YVc5a0lEMGdiV2x1S0hSaGNtZGxkRkJsY21sdlpDd2daV1ptWldOMGFYWmxVR1Z5"
    "YVc5a0lDc2dabXh2WVhRb1gyeGhjM1JVVUZKaGRHVXBJQ29nWDNCMFppazdDaUFnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQmxiSE5sSUdsbUlDaGxabVpsWTNScGRtVlFaWEpwYjJRZ1BpQjBZWEpuWlhSUVpYSnBiMlFw"
    "Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1pXWm1aV04wYVhabFVHVnlhVzlrSUQwZ2JXRjRLSFJo"
    "Y21kbGRGQmxjbWx2WkN3Z1pXWm1aV04wYVhabFVHVnlhVzlrSUMwZ1pteHZZWFFvWDJ4aGMzUlVVRkpo"
    "ZEdVcElDb2dYM0IwWmlrN0NpQWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0I5Q2lBZ0lDQjlDZ29n"
    "SUNBZ0x5OGdWSEpsYlc5c2J5QW9SV1ptWldOMElEQjROeWtnNG9DVUlITmhiV1VnZDJGMlpXWnZjbTBn"
    "WVhNZ2RtbGljbUYwYnlCaWRYUWdiVzlrZFd4aGRHVnpJRlpQVEZWTlJTNEtJQ0FnSUM4dklGVnpaWE1n"
    "YzJGdFpTQnliM2N0WW5rdGNtOTNJR2hwYzNSdmNtbGpZV3dnZEZNdmRFUWdkSEpoWTJ0cGJtY2dZWE1n"
    "ZG1saWNtRjBieTRLSUNBZ0lIc0tJQ0FnSUNBZ0lDQnBiblFnWDNSVElEMGdNQ3dnWDNSRUlEMGdNRHNL"
    "SUNBZ0lDQWdJQ0JwYm5RZ1gzUnlaVkJ2Y3lBOUlEQTdDaUFnSUNBZ0lDQWdhV1lnS0hSeWFXZE9iM1Js"
    "TG1WbVptVmpkQ0E5UFNBd2VEY3BJSHNLSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjl1Y3lBOUlDaDBjbWxu"
    "VG05MFpTNXdZWEpoYlNBK1BpQTBLU0FtSURCNFJqc0tJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOXVaQ0E5"
    "SUNCMGNtbG5UbTkwWlM1d1lYSmhiU0FnSUNBZ0lDQW1JREI0UmpzS0lDQWdJQ0FnSUNBZ0lDQWdhV1ln"
    "S0Y5dWN5QStJREFwSUY5MFV5QTlJRjl1Y3pzS0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5dVpDQStJREFw"
    "SUY5MFJDQTlJRjl1WkRzS0lDQWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ2FXWWdLSFJ5YVdkUVlYUWdQVDBn"
    "Y0c5ekxuTnZibWRRYjNNZ0ppWWdkSEpwWjFKdmR5QTlQU0J3YjNNdWNtOTNLU0I3Q2lBZ0lDQWdJQ0Fn"
    "SUNBZ0lGOTBjbVZRYjNNZ1BTQnBiblFvY0c5ekxuUnBZMnNwSUNvZ1gzUlRPd29nSUNBZ0lDQWdJSDBn"
    "Wld4elpTQjdDaUFnSUNBZ0lDQWdJQ0FnSUdsdWRDQmZkSEpUWjNJZ1BTQndZWFJVYVdOclQyWm1jMlYw"
    "VzNSeWFXZFFZWFJkSUNzZ0tIUnlhV2RTYjNjZ0xTQndZWFJUZEdGeWRGSnZkMXQwY21sblVHRjBYU2s3"
    "Q2lBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0JmZEhKVGNHUWdQU0JtWlhSamFGUnBZMnNvWDNSeVUyZHlJQ3Nn"
    "TVNrZ0xTQm1aWFJqYUZScFkyc29YM1J5VTJkeUtUc0tJQ0FnSUNBZ0lDQWdJQ0FnWDNSeVpWQnZjeUE5"
    "SUNoZmRISlRjR1FnTFNBeEtTQXFJRjkwVXpzS0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5M2NDQTlJSFJ5"
    "YVdkUVlYUXNJRjkzY2lBOUlIUnlhV2RTYjNjZ0t5QXhPd29nSUNBZ0lDQWdJQ0FnSUNCcFppQW9YM2R5"
    "SUQ0OUlIQmhkRk4wWVhKMFVtOTNXMTkzY0YwZ0t5QW9jR0YwVW05M1QyWm1jMlYwVzE5M2NDc3hYU0F0"
    "SUhCaGRGSnZkMDltWm5ObGRGdGZkM0JkS1NrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1gzZHdLeXM3"
    "SUY5M2NpQTlJQ2hmZDNBZ1BDQlRUMDVIWDB4RlRrZFVTQ2tnUHlCd1lYUlRkR0Z5ZEZKdmQxdGZkM0Jk"
    "SURvZ01Ec0tJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNBZ0lDQm1iM0lnS0dsdWRDQmZkMmtn"
    "UFNBd095QmZkMmtnUENBeE1qZzdJRjkzYVNzcktTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFv"
    "WDNkd0lENGdjRzl6TG5OdmJtZFFiM01nZkh3Z0tGOTNjQ0E5UFNCd2IzTXVjMjl1WjFCdmN5QW1KaUJm"
    "ZDNJZ1BqMGdjRzl6TG5KdmR5a3BJR0p5WldGck93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjkz"
    "Y0NBK1BTQlRUMDVIWDB4RlRrZFVTQ2tnWW5KbFlXczdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFv"
    "WDNkeUlENDlJSEJoZEZOMFlYSjBVbTkzVzE5M2NGMGdLeUFvY0dGMFVtOTNUMlptYzJWMFcxOTNjQ3N4"
    "WFNBdElIQmhkRkp2ZDA5bVpuTmxkRnRmZDNCZEtTa2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUY5M2NDc3JPeUJmZDNJZ1BTQW9YM2R3SUR3Z1UwOU9SMTlNUlU1SFZFZ3BJRDhnY0dGMFUzUmhjblJT"
    "YjNkYlgzZHdYU0E2SURBN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdZMjl1ZEdsdWRXVTdDaUFn"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQk9iM1JsSUY5MGJpQTlJR2Rs"
    "ZEU1dmRHVW9YM2R3TENCZmQzSXNJR05vS1RzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdKdmIyd2dYM1J1"
    "U1hOVWIyNWxJRDBnS0NoZmRHNHVaV1ptWldOMElEMDlJREI0TXlCOGZDQmZkRzR1WldabVpXTjBJRDA5"
    "SURCNE5Ta2dKaVlnWDNSdUxuQmxjbWx2WkNBK0lEQXBPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhV1ln"
    "S0Y5MGJpNXdaWEpwYjJRZ1BpQXdJQ1ltSUNGZmRHNUpjMVJ2Ym1VcElHSnlaV0ZyT3dvZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0FnYVdZZ0tGOTBiaTVsWm1abFkzUWdQVDBnTUhnM0lDWW1JRjkwYmk1d1lYSmhiU0Fo"
    "UFNBd0tTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOXVjeUE5SUNoZmRHNHVjR0Z5"
    "WVcwZ1BqNGdOQ2tnSmlBd2VFWTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOXVaQ0E5"
    "SUNCZmRHNHVjR0Z5WVcwZ0lDQWdJQ0FnSmlBd2VFWTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "YVdZZ0tGOXVjeUErSURBcElGOTBVeUE5SUY5dWN6c0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Jw"
    "WmlBb1gyNWtJRDRnTUNrZ1gzUkVJRDBnWDI1a093b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0Fn"
    "SUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjl6WjNJZ1BTQndZWFJVYVdOclQyWm1jMlYwVzE5M2NGMGdLeUFv"
    "WDNkeUlDMGdjR0YwVTNSaGNuUlNiM2RiWDNkd1hTazdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBiblFn"
    "WDNOd1pDQTlJR1psZEdOb1ZHbGpheWhmYzJkeUlDc2dNU2tnTFNCbVpYUmphRlJwWTJzb1gzTm5jaWs3"
    "Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JmZEhKbFVHOXpJQ3M5SUNoZmMzQmtJQzBnTVNrZ0tpQmZkRk03"
    "Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JmZDNJckt6c0tJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0Fn"
    "SUNBZ0lDQXZMeUJWY0dSaGRHVWdabkp2YlNCamRYSnlaVzUwSUhKdmR5QnBaaUJwZENCallYSnlhV1Z6"
    "SUhSeVpXMXZiRzhLSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjl3WTNJdVpXWm1aV04wSUQwOUlEQjROeUFt"
    "SmlCZmNHTnlMbkJoY21GdElDRTlJREFwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsdWRDQmZibk1n"
    "UFNBb1gzQmpjaTV3WVhKaGJTQStQaUEwS1NBbUlEQjRSanNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2x1"
    "ZENCZmJtUWdQU0FnWDNCamNpNXdZWEpoYlNBZ0lDQWdJQ0FtSURCNFJqc0tJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lHbG1JQ2hmYm5NZ1BpQXdLU0JmZEZNZ1BTQmZibk03Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Jw"
    "WmlBb1gyNWtJRDRnTUNrZ1gzUkVJRDBnWDI1a093b2dJQ0FnSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0Fn"
    "SUNBZ0lGOTBjbVZRYjNNZ0t6MGdhVzUwS0hCdmN5NTBhV05yS1NBcUlGOTBVenNLSUNBZ0lDQWdJQ0I5"
    "Q2lBZ0lDQWdJQ0FnYVdZZ0tGOTBSQ0ErSURBZ0ppWWdYM1JUSUQ0Z01Da2dld29nSUNBZ0lDQWdJQ0Fn"
    "SUNCcGJuUWdYM1JRSUQwZ1gzUnlaVkJ2Y3lBbUlEWXpPd29nSUNBZ0lDQWdJQ0FnSUNCbWJHOWhkQ0Jm"
    "ZEVSbGJIUmhJRDBnS0hacFlsUmhZbHRmZEZBZ0ppQXpNVjBnS2lCbWJHOWhkQ2hmZEVRcEtTQXZJRFkw"
    "TGpBN0NpQWdJQ0FnSUNBZ0lDQWdJSFp2YkhWdFpTQTlJR05zWVcxd0tHbHVkQ2htYkc5aGRDaDJiMngx"
    "YldVcElDc2dLQ2hmZEZBZ1BDQXpNaWtnUHlCZmRFUmxiSFJoSURvZ0xWOTBSR1ZzZEdFcEtTd2dNQ3dn"
    "TmpRcE93b2dJQ0FnSUNBZ0lIMEtJQ0FnSUgwS0NpQWdJQ0F2THlCQmNuQmxaMmRwYnlBb1JXWm1aV04w"
    "SURCNGVTa2c0b0NVSUhCNWJXOWtKM01nYjNKa1pYSWdhWE1nWW1GelplS0drbGdvYUdsbmFDbmlocEpa"
    "S0d4dmR5a0tJQ0FnSUdsbUlDaGZjR055TG1WbVptVmpkQ0E5UFNBd2VEQWdKaVlnWDNCamNpNXdZWEpo"
    "YlNBaFBTQXdLU0I3Q2lBZ0lDQWdJQ0FnYVc1MElGOWhjbkJUZEdWd0lEMGdhVzUwS0hCdmN5NTBhV05y"
    "S1NBdElHbHVkQ2h3YjNNdWRHbGpheUF2SURNdU1Da2dLaUF6T3dvZ0lDQWdJQ0FnSUM4dklHVm1abVZq"
    "ZEdsMlpWQmxjbWx2WkNCSlV5QmlZWE5sVUdWeWFXOWtJR2hsY21VZ0tHNXZJR1oxY25Sb1pYSWdiVzlr"
    "YVdacFkyRjBhVzl1SUdKbFptOXlaU0JoY25BcENpQWdJQ0FnSUNBZ2FXWWdLRjloY25CVGRHVndJRDA5"
    "SURFcENpQWdJQ0FnSUNBZ0lDQWdJR1ZtWm1WamRHbDJaVkJsY21sdlpDQTlJR1ZtWm1WamRHbDJaVkJs"
    "Y21sdlpDQXFJSEJ2ZHlneUxqQXNJQzFtYkc5aGRDZ29YM0JqY2k1d1lYSmhiU0ErUGlBMEtTQW1JREI0"
    "UmlrZ0x5QXhNaTR3S1RzS0lDQWdJQ0FnSUNCbGJITmxJR2xtSUNoZllYSndVM1JsY0NBOVBTQXlLUW9n"
    "SUNBZ0lDQWdJQ0FnSUNCbFptWmxZM1JwZG1WUVpYSnBiMlFnUFNCbFptWmxZM1JwZG1WUVpYSnBiMlFn"
    "S2lCd2IzY29NaTR3TENBdFpteHZZWFFvWDNCamNpNXdZWEpoYlNBbUlEQjRSaWtnTHlBeE1pNHdLVHNL"
    "SUNBZ0lIMEtDaUFnSUNBdkx5QldhV0p5WVhSdklDaEZabVpsWTNRZ05Da2c0b0NVSUhWelpYTWdaMnh2"
    "WW1Gc0lIWnBZbFJoWWk0S0lDQWdJQzh2SUVWbVptVmpkQ0EwZUhnNklIQmhjbUZ0SUQwZ0tITndaV1Zr"
    "SUR3OElEUXBJSHdnWkdWd2RHZ3VJQ0JUWlhSeklIWlRMQ0IyUkM0S0lDQWdJQzh2SUVWbVptVmpkQ0Ey"
    "ZUhnNklHTnZiblJwYm5WbElIWnBZbkpoZEc4Z0tIVnpaWE1nY0hKcGIzSWdOSGg0SjNNZ2RsTXZka1E3"
    "SUdsMGN5QnZkMjRnY0dGeVlXMGdhWE1LSUNBZ0lDOHZJQ0FnSUNBZ0lDQWdJQ0FnSUhadmJDMXpiR2xr"
    "WlNCdmJteDVJT0tBbENCb1lXNWtiR1ZrSUhObGNHRnlZWFJsYkhrZ2FXNGdkbTlzZFcxbElHTnZaR1Vn"
    "Y0dGMGFDa3VDaUFnSUNBdkx3b2dJQ0FnTHk4Z2RtbGljbUYwYjFCdmN5QnBibU55WlcxbGJuUnpJR0o1"
    "SUhaVElHOXVJR1ZoWTJnZ1RrOU9MWFJwWTJzdE1DQW9hUzVsTGl3Z0tITndaV1ZrTFRFcElIQmxjaUJ5"
    "YjNjcExnb2dJQ0FnTHk4Z1YyRnNheUJtY205dElIUnlhV2RuWlhJZ2RHOGdZM1Z5Y21WdWRDd2dkWEJr"
    "WVhScGJtY2dkbE12ZGtRZ1QwNU1XU0J2YmlBMGVIZ2djbTkzY3l3Z1lXNWtDaUFnSUNBdkx5QmhZMk4x"
    "YlhWc1lYUnBibWNnS0hOd1pXVmtMVEVwS25aVElIQmxjaUJqYjIxd2JHVjBaV1FnY205M0lIVnphVzVu"
    "SUdocGMzUnZjbWxqWVd3Z2RsTXVDaUFnSUNCN0NpQWdJQ0FnSUNBZ2FXNTBJRjkyVXlBOUlEQXNJRjky"
    "UkNBOUlEQTdDaUFnSUNBZ0lDQWdhVzUwSUY5MmFXSlFiM01nUFNBd093b0tJQ0FnSUNBZ0lDQXZMeUJK"
    "Ym1sMGFXRnNhWHBsSUhaVEwzWkVJRTlPVEZrZ1puSnZiU0IwY21sbloyVnlJSEp2ZHlkeklEQjROQ0Fv"
    "VGs5VUlEQjROaURpZ0pRZ2FYUnpJSEJoY21GdElHbHpJSFp2YkMxemJHbGtaU2tLSUNBZ0lDQWdJQ0Jw"
    "WmlBb2RISnBaMDV2ZEdVdVpXWm1aV04wSUQwOUlEQjROQ2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQnBiblFn"
    "WDI1eklEMGdLSFJ5YVdkT2IzUmxMbkJoY21GdElENCtJRFFwSUNZZ01IaEdPd29nSUNBZ0lDQWdJQ0Fn"
    "SUNCcGJuUWdYMjVrSUQwZ0lIUnlhV2RPYjNSbExuQmhjbUZ0SUNBZ0lDQWdJQ1lnTUhoR093b2dJQ0Fn"
    "SUNBZ0lDQWdJQ0JwWmlBb1gyNXpJRDRnTUNrZ1gzWlRJRDBnWDI1ek93b2dJQ0FnSUNBZ0lDQWdJQ0Jw"
    "WmlBb1gyNWtJRDRnTUNrZ1gzWkVJRDBnWDI1a093b2dJQ0FnSUNBZ0lIMEtDaUFnSUNBZ0lDQWdhV1ln"
    "S0hSeWFXZFFZWFFnUFQwZ2NHOXpMbk52Ym1kUWIzTWdKaVlnZEhKcFoxSnZkeUE5UFNCd2IzTXVjbTkz"
    "S1NCN0NpQWdJQ0FnSUNBZ0lDQWdJQzh2SUU5dUlIUnlhV2RuWlhJZ2NtOTNPaUIyYVdKeVlYUnZJR2ho"
    "Y3lCdmJteDVJR2hoWkNCd2IzTXVkR2xqYXlCcGJtTnlaVzFsYm5SekNpQWdJQ0FnSUNBZ0lDQWdJRjky"
    "YVdKUWIzTWdQU0JwYm5Rb2NHOXpMblJwWTJzcElDb2dYM1pUT3dvZ0lDQWdJQ0FnSUgwZ1pXeHpaU0I3"
    "Q2lBZ0lDQWdJQ0FnSUNBZ0lDOHZJRlJ5YVdkblpYSWdjbTkzSUdOdmJuUnlhV0oxZEdWeklDaHpjR1Zs"
    "WkMweEtTQnBibU55WlcxbGJuUnpJR0YwSUhSeWFXZG5aWEl0Y205M0lIWlRDaUFnSUNBZ0lDQWdJQ0Fn"
    "SUdsdWRDQmZkSEpUWjNJZ1BTQndZWFJVYVdOclQyWm1jMlYwVzNSeWFXZFFZWFJkSUNzZ0tIUnlhV2RT"
    "YjNjZ0xTQndZWFJUZEdGeWRGSnZkMXQwY21sblVHRjBYU2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0Jm"
    "ZEhKVGNHUWdQU0JtWlhSamFGUnBZMnNvWDNSeVUyZHlJQ3NnTVNrZ0xTQm1aWFJqYUZScFkyc29YM1J5"
    "VTJkeUtUc0tJQ0FnSUNBZ0lDQWdJQ0FnTHk4Z1ZtbGljbUYwYnlCclpXVndjeUJ5ZFc1dWFXNW5JRzl1"
    "SURCNE5pQnliM2R6SUhSdmJ5RGlnSlFnWVdOamRXMTFiR0YwWlNBb2MzQmxaV1F0TVNrcWRsTUtJQ0Fn"
    "SUNBZ0lDQWdJQ0FnTHk4Z1pYWmxiaUIzYUdWdUlIUm9hWE1nY205M0lIZGhjeUF3ZURZc0lIVnphVzVu"
    "SUhSb1pTQnBibWhsY21sMFpXUWdkbE11Q2lBZ0lDQWdJQ0FnSUNBZ0lHSnZiMndnWDNSeWFXZEpjMVpw"
    "WWtGamRHbDJaU0E5SUNoMGNtbG5UbTkwWlM1bFptWmxZM1FnUFQwZ01IZzBJSHg4SUhSeWFXZE9iM1Js"
    "TG1WbVptVmpkQ0E5UFNBd2VEWXBPd29nSUNBZ0lDQWdJQ0FnSUNCZmRtbGlVRzl6SUQwZ1gzUnlhV2RK"
    "YzFacFlrRmpkR2wyWlNBL0lDaGZkSEpUY0dRZ0xTQXhLU0FxSUY5MlV5QTZJREE3Q2dvZ0lDQWdJQ0Fn"
    "SUNBZ0lDQXZMeUJYWVd4cklISnZkeTFpZVMxeWIzY2dabkp2YlNCMGNtbG5aMlZ5S3pFZ2RHOGdZM1Z5"
    "Y21WdWRDMHhMQ0IxY0dSaGRHbHVaeUIyVXk5MlJBb2dJQ0FnSUNBZ0lDQWdJQ0F2THlCdmJpQXdlRFFn"
    "Y205M2N5d2dZVzVrSUdGalkzVnRkV3hoZEdsdVp5QndaWEl0Y205M0lIVnphVzVuSUdocGMzUnZjbWxq"
    "WVd3Z2RsTXVDaUFnSUNBZ0lDQWdJQ0FnSUdsdWRDQmZkM0FnUFNCMGNtbG5VR0YwTENCZmQzSWdQU0Iw"
    "Y21sblVtOTNJQ3NnTVRzS0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5M2NpQStQU0J3WVhSVGRHRnlkRkp2"
    "ZDF0ZmQzQmRJQ3NnS0hCaGRGSnZkMDltWm5ObGRGdGZkM0FyTVYwZ0xTQndZWFJTYjNkUFptWnpaWFJi"
    "WDNkd1hTa3BJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJRjkzY0Nzck95QmZkM0lnUFNBb1gzZHdJRHdn"
    "VTA5T1IxOU1SVTVIVkVncElEOGdjR0YwVTNSaGNuUlNiM2RiWDNkd1hTQTZJREE3Q2lBZ0lDQWdJQ0Fn"
    "SUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnWm05eUlDaHBiblFnWDNkcElEMGdNRHNnWDNkcElEd2dNVEk0"
    "T3lCZmQya3JLeWtnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOTNjQ0ErSUhCdmN5NXpiMjVu"
    "VUc5eklIeDhJQ2hmZDNBZ1BUMGdjRzl6TG5OdmJtZFFiM01nSmlZZ1gzZHlJRDQ5SUhCdmN5NXliM2Nw"
    "S1NCaWNtVmhhenNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZmQzQWdQajBnVTA5T1IxOU1SVTVI"
    "VkVncElHSnlaV0ZyT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOTNjaUErUFNCd1lYUlRkR0Z5"
    "ZEZKdmQxdGZkM0JkSUNzZ0tIQmhkRkp2ZDA5bVpuTmxkRnRmZDNBck1WMGdMU0J3WVhSU2IzZFBabVp6"
    "WlhSYlgzZHdYU2twSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmZkM0FyS3pzZ1gzZHlJRDBn"
    "S0Y5M2NDQThJRk5QVGtkZlRFVk9SMVJJS1NBL0lIQmhkRk4wWVhKMFVtOTNXMTkzY0YwZ09pQXdPd29n"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdOdmJuUnBiblZsT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "ZlFvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnVG05MFpTQmZkbTRnUFNCblpYUk9iM1JsS0Y5M2NDd2dYM2R5"
    "TENCamFDazdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQXZMeUJUZEc5d0lHOXVJSEpsZEhKcFoyZGxjZ29n"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdZbTl2YkNCZmRtNUpjMVJ2Ym1VZ1BTQW9LRjkyYmk1bFptWmxZM1Fn"
    "UFQwZ01IZ3pJSHg4SUY5MmJpNWxabVpsWTNRZ1BUMGdNSGcxS1NBbUppQmZkbTR1Y0dWeWFXOWtJRDRn"
    "TUNrN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9YM1p1TG5CbGNtbHZaQ0ErSURBZ0ppWWdJVjky"
    "YmtselZHOXVaU2tnWW5KbFlXczdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQXZMeUJWY0dSaGRHVWdkbE12"
    "ZGtRZ1QwNU1XU0J2YmlBd2VEUWdjbTkzY3lBb01IZzJJR2hoY3lCMmIyd3RjMnhwWkdVZ2NHRnlZVzBz"
    "SUc1dmRDQjJhV0p5WVhSdktRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjkyYmk1bFptWmxZM1Fn"
    "UFQwZ01IZzBJQ1ltSUY5MmJpNXdZWEpoYlNBaFBTQXdLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ2FXNTBJRjl1Y3lBOUlDaGZkbTR1Y0dGeVlXMGdQajRnTkNrZ0ppQXdlRVk3Q2lBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjl1WkNBOUlDQmZkbTR1Y0dGeVlXMGdJQ0FnSUNBZ0ppQXdlRVk3"
    "Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjl1Y3lBK0lEQXBJRjkyVXlBOUlGOXVjenNL"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9YMjVrSUQ0Z01Da2dYM1pFSUQwZ1gyNWtPd29n"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdmUW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdMeThnUVdOamRXMTFiR0Yw"
    "WlNCMmFXSnlZWFJ2SUhCdmN5QjNhR1Z1SUhKdmR5QnBjeUF3ZURRZ1QxSWdNSGcySUNoMmFXSnlZWFJ2"
    "SUhKMWJuTWdiMjRnWW05MGFDa0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmZG00dVpXWm1aV04w"
    "SUQwOUlEQjROQ0I4ZkNCZmRtNHVaV1ptWldOMElEMDlJREI0TmlrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJR2x1ZENCZmMyZHlJRDBnY0dGMFZHbGphMDltWm5ObGRGdGZkM0JkSUNzZ0tGOTNjaUF0"
    "SUhCaGRGTjBZWEowVW05M1cxOTNjRjBwT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0Jm"
    "YzNCa0lEMGdabVYwWTJoVWFXTnJLRjl6WjNJZ0t5QXhLU0F0SUdabGRHTm9WR2xqYXloZmMyZHlLVHNL"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCZmRtbGlVRzl6SUNzOUlDaGZjM0JrSUMwZ01Ta2dLaUJm"
    "ZGxNN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCZmQzSXJLenNL"
    "SUNBZ0lDQWdJQ0FnSUNBZ2ZRb0tJQ0FnSUNBZ0lDQWdJQ0FnTHk4Z1ZYQmtZWFJsSUhaVEwzWkVJR1p5"
    "YjIwZ1kzVnljbVZ1ZENCeWIzY2dUMDVNV1NCcFppQXdlRFFLSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjl3"
    "WTNJdVpXWm1aV04wSUQwOUlEQjROQ0FtSmlCZmNHTnlMbkJoY21GdElDRTlJREFwSUhzS0lDQWdJQ0Fn"
    "SUNBZ0lDQWdJQ0FnSUdsdWRDQmZibk1nUFNBb1gzQmpjaTV3WVhKaGJTQStQaUEwS1NBbUlEQjRSanNL"
    "SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2x1ZENCZmJtUWdQU0FnWDNCamNpNXdZWEpoYlNBZ0lDQWdJQ0Ft"
    "SURCNFJqc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmYm5NZ1BpQXdLU0JmZGxNZ1BTQmZibk03"
    "Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1gyNWtJRDRnTUNrZ1gzWkVJRDBnWDI1a093b2dJQ0Fn"
    "SUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0FnSUNBZ0lDOHZJRU4xY25KbGJuUWdjbTkzSUhCaGNuUnBZV3c2"
    "SUhCdmN5NTBhV05ySUdsdVkzSmxiV1Z1ZEhNZ1lYUWdZM1Z5Y21WdWRDMXliM2NnZGxNS0lDQWdJQ0Fn"
    "SUNBZ0lDQWdMeThnVDA1TVdTQjNhR1Z1SUdOMWNuSmxiblFnY205M0lHaGhjeUIyYVdKeVlYUnZJR0Zq"
    "ZEdsMlpTQW9NSGcwSUc5eUlEQjROaWtLSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjl3WTNJdVpXWm1aV04w"
    "SUQwOUlEQjROQ0I4ZkNCZmNHTnlMbVZtWm1WamRDQTlQU0F3ZURZcElIc0tJQ0FnSUNBZ0lDQWdJQ0Fn"
    "SUNBZ0lGOTJhV0pRYjNNZ0t6MGdhVzUwS0hCdmN5NTBhV05yS1NBcUlGOTJVenNLSUNBZ0lDQWdJQ0Fn"
    "SUNBZ2ZRb2dJQ0FnSUNBZ0lIMEtDaUFnSUNBZ0lDQWdhV1lnS0Y5MlJDQStJREFnSmlZZ1gzWlRJRDRn"
    "TUNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0JwYm5RZ1gzWlFJRDBnWDNacFlsQnZjeUFtSURZek93b2dJQ0Fn"
    "SUNBZ0lDQWdJQ0JtYkc5aGRDQmZka1JsYkhSaElEMGdLSFpwWWxSaFlsdGZkbEFnSmlBek1WMGdLaUJt"
    "Ykc5aGRDaGZka1FwS1NBdklERXlPQzR3T3dvZ0lDQWdJQ0FnSUNBZ0lDQmxabVpsWTNScGRtVlFaWEpw"
    "YjJRZ0t6MGdLRjkyVUNBOElETXlLU0EvSUY5MlJHVnNkR0VnT2lBdFgzWkVaV3gwWVRzS0lDQWdJQ0Fn"
    "SUNCOUNpQWdJQ0I5Q2dvZ0lDQWdMeThnVW1WdVpHVnlJSE5oYlhCc1pRb2dJQ0FnTHk4Z1ltRnpaVkJs"
    "Y21sdlpDQTlJR1ZtWm1WamRHbDJaVkJsY21sdlpDQlhTVlJJVDFWVUlIWnBZbkpoZEc4dmRISmxiVzlz"
    "YnlCdGIyUjFiR0YwYVc5dUxpQWdWWE5wYm1jZ2RHaGxDaUFnSUNBdkx5QnRiMlIxYkdGMFpXUWdkbUZz"
    "ZFdVZ1ptOXlJR1pUWVcxd2JHVlFiM01nYVc1MFpXZHlZWFJwYjI0Z2QyOTFiR1FnYlhWc2RHbHdiSGtn"
    "ZEdobElHMXZaSFZzWVhScGIyNEtJQ0FnSUM4dklHRnRjR3hwZEhWa1pTQmllU0JnWld4aGNITmxaR0Fz"
    "SUhCeWIyUjFZMmx1WnlCaElITjFZbk4wWVc1MGFXRnNJR0oxZW5vZ1lYUWdkbWxpY21GMGJ5QnlZWFJs"
    "Q2lBZ0lDQXZMeUFvWlM1bkxpd2dabXgxZEdVZ2IyNGdjR0YwSURFZ1kyZ3dJR0YwSUhabFkxOWthVzA5"
    "T0NrdUlDQlVhR1VnVkZKVlJTQndiM05wZEdsdmJpMWtiMjFoYVc0Z1pXWm1aV04wQ2lBZ0lDQXZMeUJ2"
    "WmlCMmFXSnlZWFJ2SUdseklIUm9aU0JwYm5SbFozSmhiQ0J2WmlCMGFHVWdabkpsY1NCdGIyUjFiR0Yw"
    "YVc5dUxDQjNhR2xqYUNCcGN5QmhJSFJwYm5rZ1BERXRDaUFnSUNBdkx5QnpZVzF3YkdVZ2IzTmphV3hz"
    "WVhScGIyNGc0b0NVSUhOaFptVnNlU0J1Wldkc2FXZHBZbXhsTGdvZ0lDQWdabXh2WVhRZ1ltRnpaVkJs"
    "Y21sdlpDQTlJR1ZtWm1WamRHbDJaVkJsY21sdlpEc0tJQ0FnSUdsbUlDaDBjbWxuVG05MFpTNWxabVps"
    "WTNRZ1BUMGdNSGcwSUh4OElIUnlhV2RPYjNSbExtVm1abVZqZENBOVBTQXdlRFlnZkh3Z2RISnBaMDV2"
    "ZEdVdVpXWm1aV04wSUQwOUlEQjROeUI4ZkFvZ0lDQWdJQ0FnSUY5d1kzSXVaV1ptWldOMElEMDlJREI0"
    "TkNCOGZDQmZjR055TG1WbVptVmpkQ0E5UFNBd2VEWWdmSHdnWDNCamNpNWxabVpsWTNRZ1BUMGdNSGcz"
    "S1NCN0NpQWdJQ0FnSUNBZ1ltRnpaVkJsY21sdlpDQTlJQ2gwWVhKblpYUlFaWEpwYjJRZ1BpQXdMakFw"
    "SUQ4Z2RHRnlaMlYwVUdWeWFXOWtJRG9nWm14dllYUW9kSEpwWjA1dmRHVXVjR1Z5YVc5a0tUc0tJQ0Fn"
    "SUgwS0lDQWdJR1pzYjJGMElHWnlaWEVnUFNCd1pYSnBiMlJVYjBaeVpYRkdkQ2h0WVhnb01Td2dhVzUw"
    "S0dKaGMyVlFaWEpwYjJRcEtTd2djMjF3TG1acGJtVjBkVzVsS1RzS0lDQWdJQzh2SUZOaGJYQnNaU0J3"
    "YjNOcGRHbHZiam9LSUNBZ0lDOHZJQ0FnTFNCSlppQmpkWEp5Wlc1MElISnZkeUJvWVhNZ1lXTjBhWFps"
    "SUhCcGRHTm9JSE5zYVdSbElDZ3hlSGd2TW5oNEx6TjRlQ2tzSUhWelpTQnNiMmNnYVc1MFpXZHlZV3dL"
    "SUNBZ0lDOHZJQ0FnSUNEaWlLdERMMUFvZENsa2RDRGlpWWdnUThPWFZDL09sRkFndzVjZ2JHNG9VREV2"
    "VURBcElDQW9ZWE56ZFcxbGN5QnNhVzVsWVhJZ2NtRnRjRHNnWTJ4dmMyVWdaVzV2ZFdkb0tRb2dJQ0Fn"
    "THk4Z0lDQXRJRTkwYUdWeWQybHpaU0J3WlhKcGIyUWdhWE1nYzNSaFlteGxJSEJoYzNRZ2RISnBaMmRs"
    "Y2pzZ2MybHRjR3hsSUdWc1lYQnpaV1REbDJaeVpYRWdhWE1nWlhoaFkzUXVDaUFnSUNCaWIyOXNJRjl6"
    "Ykdsa1pVRmpkR2wyWlNBOUlDaGZjR055TG1WbVptVmpkQ0E5UFNBd2VERWdmSHdnWDNCamNpNWxabVps"
    "WTNRZ1BUMGdNSGd5SUh4OENpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCZmNHTnlMbVZt"
    "Wm1WamRDQTlQU0F3ZURNZ2ZId2dYM0JqY2k1bFptWmxZM1FnUFQwZ01IZzFLVHNLSUNBZ0lHWnNiMkYw"
    "SUdaVFlXMXdiR1ZRYjNNN0NpQWdJQ0JwWmlBb1gzTnNhV1JsUVdOMGFYWmxLU0I3Q2lBZ0lDQWdJQ0Fn"
    "Wm14dllYUWdVREJtSUQwZ1pteHZZWFFvZEhKcFowNXZkR1V1Y0dWeWFXOWtLVHNLSUNBZ0lDQWdJQ0Jt"
    "Ykc5aGRDQlFNV1lnUFNCaVlYTmxVR1Z5YVc5a093b2dJQ0FnSUNBZ0lHWnNiMkYwSUdSUVppQTlJRkF4"
    "WmlBdElGQXdaanNLSUNBZ0lDQWdJQ0JwWmlBb1lXSnpLR1JRWmlrZ1BpQXdMalVnSmlZZ1pXeGhjSE5s"
    "WkNBK0lERmxMVFlwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdMeThnVEc5bklHbHVkR1ZuY21Gc0lIZHBkR2dn"
    "Wm1sdVpYUjFibVU2SUhOMVluTjBhWFIxZEdVZ1l6UXFOREk0THpJZ1ptOXlJSFJvWlNCamIyNXpkR0Z1"
    "ZENCRExnb2dJQ0FnSUNBZ0lDQWdJQ0JtYkc5aGRDQm1kRU1nUFNCak5ITndaV1ZrYzF0emJYQXVabWx1"
    "WlhSMWJtVWdKaUF3ZUVaZElDb2dOREk0TGpBZ0x5Qm1iRzloZENoemJYQXVZbmRHWVdOMGIzSXBPd29n"
    "SUNBZ0lDQWdJQ0FnSUNCbVUyRnRjR3hsVUc5eklEMGdablJESUNvZ1pXeGhjSE5sWkNBdklHUlFaaUFx"
    "SUd4dlp5aFFNV1lnTHlCUU1HWXBPd29nSUNBZ0lDQWdJSDBnWld4elpTQjdDaUFnSUNBZ0lDQWdJQ0Fn"
    "SUdaVFlXMXdiR1ZRYjNNZ1BTQmxiR0Z3YzJWa0lDb2dabkpsY1NBdklHWnNiMkYwS0hOdGNDNWlkMFpo"
    "WTNSdmNpazdDaUFnSUNBZ0lDQWdmUW9nSUNBZ2ZTQmxiSE5sSUhzS0lDQWdJQ0FnSUNCbVUyRnRjR3hs"
    "VUc5eklEMGdaV3hoY0hObFpDQXFJR1p5WlhFZ0x5Qm1iRzloZENoemJYQXVZbmRHWVdOMGIzSXBPd29n"
    "SUNBZ2ZRb2dJQ0FnTHk4Z01IZzVlSGdnYzJGdGNHeGxJRzltWm5ObGREb2djMmhwWm5RZ2MzUmhjblJw"
    "Ym1jZ2NHOXphWFJwYjI0Z0tHbHVJR052YlhCeVpYTnpaV1F0Wkc5dFlXbHVJSE5oYlhCc1pYTXBDaUFn"
    "SUNCcFppQW9YM05oYlhCc1pVOW1abk5sZENBK0lEQXBJSHNLSUNBZ0lDQWdJQ0JtVTJGdGNHeGxVRzl6"
    "SUNzOUlHWnNiMkYwS0Y5ellXMXdiR1ZQWm1aelpYUWdMeUJ0WVhnb01Td2djMjF3TG1KM1JtRmpkRzl5"
    "S1NrN0NpQWdJQ0I5Q2dvZ0lDQWdhV1lnS0hOdGNDNXNiMjl3VEdWdUlENGdNaWtnZXdvZ0lDQWdJQ0Fn"
    "SUdsbUlDaG1VMkZ0Y0d4bFVHOXpJRDQ5SUdac2IyRjBLSE50Y0M1c2IyOXdVM1JoY25RZ0t5QnpiWEF1"
    "Ykc5dmNFeGxiaWtwQ2lBZ0lDQWdJQ0FnSUNBZ0lHWlRZVzF3YkdWUWIzTWdQU0JtYkc5aGRDaHpiWEF1"
    "Ykc5dmNGTjBZWEowS1NBcklHMXZaQ2htVTJGdGNHeGxVRzl6SUMwZ1pteHZZWFFvYzIxd0xteHZiM0JU"
    "ZEdGeWRDa3NJR1pzYjJGMEtITnRjQzVzYjI5d1RHVnVLU2s3Q2lBZ0lDQjlJR1ZzYzJVZ2FXWWdLR1pU"
    "WVcxd2JHVlFiM01nUGowZ1pteHZZWFFvYzIxd0xteGxibWQwYUNrcElIc0tJQ0FnSUNBZ0lDQnlaWFIx"
    "Y200Z01DNHdPd29nSUNBZ2ZRb2dJQ0FnYVdZZ0tHWlRZVzF3YkdWUWIzTWdQQ0F3TGpBcElISmxkSFZ5"
    "YmlBd0xqQTdDZ29nSUNBZ0x5OGdVMkZ0Y0d4bElIWmhiSFZsSUhkcGRHZ2djSEp2Y0dWeUlHVnVaQzFt"
    "WVdSbElDaHpZVzF3YkdVZ2RHVnliV2x1WVhScGIyNGdjMmh2ZFd4a0lHNXZkQ0J6Ym1Gd0lIUnZJREFw"
    "Q2lBZ0lDQm1iRzloZENCek93b2dJQ0FnYVdZZ0tITnRjQzVzYjI5d1RHVnVJRHc5SURJZ0ppWWdabE5o"
    "YlhCc1pWQnZjeUErUFNCbWJHOWhkQ2h6YlhBdWJHVnVaM1JvS1NBdElERXVNQ2tnZXdvZ0lDQWdJQ0Fn"
    "SUM4dklFNWxZWElnWlc1a0lHOW1JRzV2Ymkxc2IyOXdhVzVuSUhOaGJYQnNaVG9nWm1Ga1pTQnZkWFFn"
    "YjNabGNpQnNZWE4wSUhOaGJYQnNaU0IwYnlCaGRtOXBaQ0JqYkdsamF3b2dJQ0FnSUNBZ0lITWdQU0F3"
    "TGpBN0NpQWdJQ0I5SUdWc2MyVWdld29nSUNBZ0lDQWdJSE1nUFNCblpYUlRZVzF3YkdWR0tITnRjQzV6"
    "ZEdGeWRDd2dabE5oYlhCc1pWQnZjeXdnYzIxd0xteGxibWQwYUN3Z2MyMXdMbXh2YjNCVGRHRnlkQ3dn"
    "YzIxd0xteHZiM0JNWlc0cE93b2dJQ0FnZlFvS0lDQWdJQzh2SU9LVWdPS1VnQ0JCYm5ScExXTnNhV05y"
    "SUhKaGJYQnpJT0tVZ09LVWdBb2dJQ0FnTHk4Z01TNGdWSEpwWjJkbGNpQnlZVzF3T2lBMk5DMXpZVzF3"
    "YkdVZ1ptRmtaUzFwYmlCdmJpQmxkbVZ5ZVNCdVpYY2dibTkwWlNBb2JXbHJTVlFnWm1Ga1pXTnZkVzUw"
    "S1M0S0lDQWdJQzh2SUNBZ0lGWnZiSFZ0WlNCbmIyVnpJRERpaHBJeElHOTJaWElnWm1seWMzUWdOalFn"
    "YjNWMGNIVjBJSE5oYlhCc1pYTWdZV1owWlhJZ2RISnBaMmRsY2k0S0lDQWdJR1pzYjJGMElHUmxZMnhw"
    "WTJzZ1BTQmpiR0Z0Y0NobGJHRndjMlZrSUNvZ0tEUTBNVEF3TGpBZ0x5QTJOQzR3S1N3Z01DNHdMQ0F4"
    "TGpBcE93b0tJQ0FnSUM4dklESXVJRVZ1WkMxdlppMXpZVzF3YkdVZ1ptRmtaUzF2ZFhRNklEWTBMWE5o"
    "YlhCc1pTQm1ZV1JsTFc5MWRDQmhjeUJtVTJGdGNHeGxVRzl6SUdGd2NISnZZV05vWlhNS0lDQWdJQzh2"
    "SUNBZ0lITmhiWEJzWlNCbGJtUWdLRzl1YkhrZ1ptOXlJRzV2Ymkxc2IyOXdhVzVuSUhOaGJYQnNaWE1w"
    "TGlBZ1VISmxkbVZ1ZEhNZ2MzVmtaR1Z1SUhOcGJHVnVZMlV1Q2lBZ0lDQm1iRzloZENCbGJtUkdZV1Js"
    "SUQwZ01TNHdPd29nSUNBZ2FXWWdLSE50Y0M1c2IyOXdUR1Z1SUR3OUlESXBJSHNLSUNBZ0lDQWdJQ0Jt"
    "Ykc5aGRDQnlaVzFoYVc1cGJtY2dQU0JtYkc5aGRDaHpiWEF1YkdWdVozUm9LU0F0SUdaVFlXMXdiR1ZR"
    "YjNNN0NpQWdJQ0FnSUNBZ2FXWWdLSEpsYldGcGJtbHVaeUE4SURZMExqQXBJR1Z1WkVaaFpHVWdQU0J0"
    "WVhnb01DNHdMQ0J5WlcxaGFXNXBibWNnTHlBMk5DNHdLVHNLSUNBZ0lIMEtDaUFnSUNBdkx5QXpMaUJN"
    "YjI5d0lHTnliM056Wm1Ga1pUb2djMjF2YjNSb2N5QmhibmtnY21WemFXUjFZV3dnYkc5dmNFVnVaT0tH"
    "a214dmIzQlRkR0Z5ZENCa2FYTmpiMjUwYVc1MWFYUjVMZ29nSUNBZ0x5OGdJQ0FnVkdobElHVnVZMjlr"
    "WlhJZ2JtOTNJR1Z0WW1Wa2N5QnNiMjl3SUhkeVlYQWdZMjl1ZEdWNGRDQnVaWGgwSUhSdklHeHZiM0JG"
    "Ym1RZ2MyOGdWbEVLSUNBZ0lDOHZJQ0FnSUhGMVlXNTBhWHBoZEdsdmJpQnJaV1Z3Y3lCMGFHVWdjMlZo"
    "YlNCamIyNTBhVzUxYjNWekxDQmlkWFFnWVNBeE5pMXpZVzF3YkdVZ1kzSnZjM05tWVdSbENpQWdJQ0F2"
    "THlBZ0lDQmpZWFJqYUdWeklHRnVlU0J5WlcxaGFXNXBibWNnYldsemJXRjBZMmd1Q2lBZ0lDQnBaaUFv"
    "YzIxd0xteHZiM0JNWlc0Z1BpQXlLU0I3Q2lBZ0lDQWdJQ0FnWTI5dWMzUWdabXh2WVhRZ1ExSlBVMU5H"
    "UVVSRlgweEZUaUE5SURFMkxqQTdDaUFnSUNBZ0lDQWdabXh2WVhRZ2JHOXZjRVZ1WkNBOUlHWnNiMkYw"
    "S0hOdGNDNXNiMjl3VTNSaGNuUWdLeUJ6YlhBdWJHOXZjRXhsYmlrN0NpQWdJQ0FnSUNBZ1pteHZZWFFn"
    "WkdsemRFWnliMjFGYm1RZ1BTQnNiMjl3Ulc1a0lDMGdabE5oYlhCc1pWQnZjenNLSUNBZ0lDQWdJQ0Jw"
    "WmlBb1pHbHpkRVp5YjIxRmJtUWdQaUF3TGpBZ0ppWWdaR2x6ZEVaeWIyMUZibVFnUENCRFVrOVRVMFpC"
    "UkVWZlRFVk9LU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lHWnNiMkYwSUhkeVlYQlFiM01nUFNCbWJHOWhkQ2h6"
    "YlhBdWJHOXZjRk4wWVhKMEtTQXJJQ2hEVWs5VFUwWkJSRVZmVEVWT0lDMGdaR2x6ZEVaeWIyMUZibVFw"
    "T3dvZ0lDQWdJQ0FnSUNBZ0lDQm1iRzloZENCM2NtRndVMkZ0Y0d4bElEMGdaMlYwVTJGdGNHeGxSaWh6"
    "YlhBdWMzUmhjblFzSUhkeVlYQlFiM01zSUhOdGNDNXNaVzVuZEdnc0lITnRjQzVzYjI5d1UzUmhjblFz"
    "SUhOdGNDNXNiMjl3VEdWdUtUc0tJQ0FnSUNBZ0lDQWdJQ0FnWm14dllYUWdZbXhsYm1RZ1BTQW9RMUpQ"
    "VTFOR1FVUkZYMHhGVGlBdElHUnBjM1JHY205dFJXNWtLU0F2SUVOU1QxTlRSa0ZFUlY5TVJVNDdDaUFn"
    "SUNBZ0lDQWdJQ0FnSUM4dklFVnhkV0ZzTFhCdmQyVnlJR055YjNOelptRmtaUW9nSUNBZ0lDQWdJQ0Fn"
    "SUNCbWJHOWhkQ0IzTVNBOUlHTnZjeWhpYkdWdVpDQXFJREV1TlRjd056azJNeWs3Q2lBZ0lDQWdJQ0Fn"
    "SUNBZ0lHWnNiMkYwSUhjeUlEMGdjMmx1S0dKc1pXNWtJQ29nTVM0MU56QTNPVFl6S1RzS0lDQWdJQ0Fn"
    "SUNBZ0lDQWdjeUE5SUhNZ0tpQjNNU0FySUhkeVlYQlRZVzF3YkdVZ0tpQjNNanNLSUNBZ0lDQWdJQ0I5"
    "Q2lBZ0lDQjlDZ29nSUNBZ2NtVjBkWEp1SUhNZ0tpQW9abXh2WVhRb2RtOXNkVzFsS1NBdklEWTBMakFw"
    "SUNvZ1pHVmpiR2xqYXlBcUlHVnVaRVpoWkdVN0NuMEsnKS5kZWNvZGUoJ3V0Zi04JykKCiAgICAjIEFz"
    "c2VtYmxlCiAgICByZXR1cm4gaGVhZGVyICsgbWV0YSArICIiLmpvaW4oZGF0YV9hcnJheXMpICsgIlxu"
    "IiArIHRhYmxlcyArIGZldGNoZXJzICsgZGVjb2RlcnMgKyBnZXRfY2hhbm5lbF9vdXRwdXQKCgppZiBf"
    "X25hbWVfXyA9PSAnX19tYWluX18nOgogICAgbW9kX3BhdGggPSBzeXMuYXJndlsxXSBpZiBsZW4oc3lz"
    "LmFyZ3YpID4gMSBlbHNlICcvbW50L3VzZXItZGF0YS91cGxvYWRzLzEyVEguTU9EJwogICAgb3V0X3Bh"
    "dGggPSBzeXMuYXJndlsyXSBpZiBsZW4oc3lzLmFyZ3YpID4gMiBlbHNlICcvaG9tZS9jbGF1ZGUvbW9k"
    "X2NydW5jaC8xMlRIX2NydW5jaF9jb21tb24uZ2xzbCcKICAgIG1haW4obW9kX3BhdGgsIG91dF9wYXRo"
    "KQo="
)

def main():
    import argparse
    class _ArgFmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        """Combined formatter: preserves newlines AND shows '(default: …)'."""
        pass
    parser = argparse.ArgumentParser(
        description='MOD/S3M Player - Generates HTML player + ShaderToy GLSL + PNG samples',
        formatter_class=_ArgFmt)
    parser.add_argument('modfile', help='MOD or S3M file to play')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Sample decimation factor: 1=full-rate, 2=22kHz, 4=11kHz. '
                             'HF percussion (cymbals/rides) gets max(1,DS//2) to keep shimmer.')
    parser.add_argument('--bitrate', choices=['lo','med','hi','ultra'], default='med',
                        help='RVQ codebook size (mp3-style quality knob). '
                             'lo=K(128,64) 13b/pair smallest+grainy, med=K(256,128) 15b/pair balanced, '
                             'hi=K(512,256) 17b/pair sharper, ultra=K(1024,512) 19b/pair near-transparent.')
    parser.add_argument('--vec-dim', type=int, default=8, choices=[2, 4, 8],
                        help='RVQ vector dimensionality. 8=smallest (~2.1 bits/sample), '
                             '4=medium (4.25 bits/sample), 2=highest fidelity (8.5 bits/sample).')
    parser.add_argument('--resampler', choices=['linear','bspline','lanczos3'],
                        default='lanczos3',
                        help='Sample resampler. linear=2-tap (cheapest, ProTracker-style), '
                             'bspline=4-tap cubic (smooth/soft), '
                             'lanczos3=6-tap sinc (sharpest/brightest, ~50%% more cost).')
    parser.add_argument('--no-split', action='store_true', default=True,
                        help='Keep VQ arrays + decoders in Common tab.  Required for '
                             'oscilloscope/spectrum/Buffer A visualizers to decode actual '
                             'audio via getChannelOutput.  Default ON.')
    parser.add_argument('--split', dest='no_split', action='store_false',
                        help='Split VQ arrays into Sound tab — fast Common compile, but '
                             'breaks audio-driven visualizers (no getChannelOutput in Image/BufferA).')
    parser.add_argument('--viz', type=int, choices=[0, 1, 2, 3, 4, 5], default=1,
                        help='Image-tab visualizer:\n'
                             '  0 = None             (black backdrop, fastest compile)\n'
                             '  1 = Reactive 001     (PAEz fork — SDF circles + cosmic web)  ← default\n'
                             '  2 = Fluxline Surfer  (mrange — DR2 dodecahedron + glowtracer)\n'
                             '  3 = Zuvuya           (city/stars + audio-reactive curtain)\n'
                             '  4 = Maya             (raymarched fractal tunnel-warp)\n'
                             '  5 = Dodecahedron     (Philip Bertani — DR2 IFS fractal raymarcher)')
    parser.add_argument('--no-rvq2', dest='no_rvq2', action='store_true', default=False,
                        help='Skip RVQ stage 2 (residual quantization).  Drops ~40%% of '
                             'sample-data const arrays from Sound tab → faster compile. '
                             'Quality cost: ~4 dB SNR (sounds noisier but pitch is unchanged). '
                             'IMPORTANT: when re-pasting into ShaderToy, paste BOTH the new '
                             'Common AND new Sound — otherwise mismatched RVQ_BITS produces '
                             'high-pitch garbage from a stale Common reading 15-bit-packed '
                             'codes that were actually written at 8 bits.')
    parser.add_argument('--use-png', action='store_true', default=False,
                        help='Use legacy PNG-loaded Common (samples read via texelFetch from '
                             'iChannel0=PNG) instead of VQ-encoded const arrays. Smaller Common '
                             'source = faster compile, but raw 8-bit samples (no RVQ) so quality '
                             'differs.  ShaderToy setup: Image/Common iChannel0 = '
                             'GSLINGER_player_data.png via Unofficial Plugin "Custom Textures". '
                             'Implies --no-split.')
    args = parser.parse_args()

    # --use-png implies --no-split: the legacy PNG-loaded Common keeps
    # getChannelOutput inline (no VQ arrays to move into Sound).
    if args.use_png:
        args.no_split = True

    # ── Print active settings as a single command-line — copy/pasteable.
    import sys as _sys
    _passed = set()
    for _a in _sys.argv[1:]:
        if _a.startswith('--'):
            _passed.add(_a.split('=')[0].lstrip('-').replace('-', '_'))
    _all_default = not (_passed - {'modfile'})
    _prefix = '[default] ' if _all_default else ''
    _flags = []
    _flags.append(f'--viz {args.viz}')
    _flags.append('--split' if not args.no_split else '--no-split')
    if args.no_rvq2: _flags.append('--no-rvq2')
    if args.use_png: _flags.append('--use-png')
    _flags.append(f'--vec-dim {args.vec_dim}')
    _flags.append(f'--downsample {args.downsample}')
    _flags.append(f'--resampler {args.resampler}')
    _flags.append(f'--bitrate {args.bitrate}')
    print(_prefix + ' '.join(_flags))
    
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
    create_fixed_player_html(mod, html_file, args.downsample, compress=True, vec_dim=args.vec_dim)

    # ShaderToy Common tab: VQ-encoded via embedded vq_encoder_v2 (default),
    # or legacy PNG-loaded Common via create_shadertoy_glsl when --use-png.
    glsl_common_file = base_name + "_shadertoy_common.glsl"
    if args.use_png:
        print(f"\n\U0001f5bc\ufe0f  --use-png: skipping VQ encoder, generating PNG-loaded Common")
        # Legacy Common is generated below by create_shadertoy_glsl into the
        # _tmp_tabs_*_common.glsl file. We'll RENAME it (instead of deleting)
        # to become the final glsl_common_file.
    else:
        try:
            import types as _types, base64 as _b64
            _vqmod = _types.ModuleType('vq_encoder_v2')
            _vqmod.__file__ = __file__
            exec(compile(_b64.b64decode(_VQ_ENCODER_B64).decode('utf-8'), 'vq_encoder_v2', 'exec'), _vqmod.__dict__)
            print(f"\n\U0001f3b5 Generating VQ-encoded Common tab...")
            _vqmod.main(args.modfile, glsl_common_file, K=256, weighted=True, downsample=args.downsample, bitrate=args.bitrate, vec_dim=args.vec_dim, resampler=args.resampler, no_rvq2=args.no_rvq2)
            # Stamp visualizer name into the b64-emitted Common header so it
            # matches the other 3 tabs.
            _viz_names = {
                0: "None (black backdrop)",
                1: "Reactive 001 (PAEz fork — SDF circles + cosmic web)",
                2: "Fluxline Surfer (mrange — DR2 dodecahedron + glowtracer)",
                3: "Zuvuya (city/stars + audio-reactive curtain)",
                4: "Maya (raymarched fractal tunnel-warp)",
                5: "Dodecahedron (Philip Bertani — DR2 IFS fractal raymarcher)",
            }
            _vname = _viz_names.get(args.viz, f"viz{args.viz}")
            try:
                with open(glsl_common_file) as _cf: _ct = _cf.read()
                _ct = _ct.replace(
                    "GLSL MOD Player v1.35 (c) 2026 Orblivius",
                    "GLSL MOD Player v1.37 (c)2026 Orblivius", 1)
                _ct = _ct.replace("   COMMON TAB\n", f"   COMMON TAB\n   Visualizer: {_vname}\n", 1)
                with open(glsl_common_file, 'w') as _cf: _cf.write(_ct)
            except Exception:
                pass
        except Exception as _e:
            print(f"   WARNING: VQ encoder failed ({_e}), falling back to built-in")
            _fb_glsl = base_name + "_shadertoy.glsl"
            create_shadertoy_glsl(mod, _fb_glsl, args.downsample, compress=True,
                                 compressed_pattern_size=pattern_size,
                                 pattern_bytes_data=pattern_bytes,
                                 sample_bytes_data=sample_bytes,
                                 seek_table=seek_table, vec_dim=args.vec_dim,
                                 viz=args.viz)
            glsl_common_file = _fb_glsl.replace('.glsl', '_common.glsl')

    # Sound / Image / Buffer A tabs from built-in emitter
    # Use a stub name that has NO overlap with _shadertoy.glsl patterns
    _glsl_stub = base_name + "_tmp_tabs_shadertoy.glsl"
    create_shadertoy_glsl(mod, _glsl_stub, args.downsample, compress=True,
                         compressed_pattern_size=pattern_size,
                         pattern_bytes_data=pattern_bytes,
                         sample_bytes_data=sample_bytes,
                         seek_table=seek_table, vec_dim=args.vec_dim,
                         viz=args.viz)
    import os as _os2
    # create_shadertoy_glsl writes: _tmp_tabs_shadertoy_common/sound/image/bufferA.glsl
    for _ext in ('_sound.glsl', '_image.glsl', '_bufferA.glsl'):
        _src = _glsl_stub.replace('.glsl', _ext)
        _dst = base_name + "_shadertoy" + _ext
        if _os2.path.exists(_src): _os2.replace(_src, _dst)
    # Common: --use-png → keep the legacy Common (rename to final). Otherwise → delete it.
    _legacy_common = _glsl_stub.replace('.glsl', '_common.glsl')
    if args.use_png:
        if _os2.path.exists(_legacy_common):
            _os2.replace(_legacy_common, glsl_common_file)
    else:
        if _os2.path.exists(_legacy_common): _os2.remove(_legacy_common)
    if _os2.path.exists(_glsl_stub): _os2.remove(_glsl_stub)



    # ── POST-PROCESS: split sample-decode plumbing out of Common into Sound ──
    # vqCodes/vqCodebook arrays + fetchCodesByte/fetchCodebookByte + getSample/getSampleF
    # only need to live where getChannelOutput is called (Sound tab).  Common doesn't
    # use them (Image/BufferA do their own note synthesis), so we move them.
    # Skip this when --no-split is set (user wants VQ in Common for visualizers).
    if args.no_split:
        if args.use_png:
            print("   📌 --use-png: legacy PNG-loaded Common (getChannelOutput inline, samples via texelFetch)")
        else:
            print("   📌 --no-split: keeping VQ arrays in Common (visualizers can access actual sample audio)")
        # Inject #define VQ_IN_COMMON 1 at top of Common so BufferA/Image
        # gates light up.  Must be visible before BufferA's #ifdef checks.
        try:
            with open(glsl_common_file) as _f: _common_src = _f.read()
            if 'VQ_IN_COMMON' not in _common_src:
                # Insert after first /* ... */ header block, or at file start.
                _hdr_end = _common_src.find('*/')
                if _hdr_end > 0:
                    _hdr_end = _common_src.find('\n', _hdr_end) + 1
                else:
                    _hdr_end = 0
                _common_src = (_common_src[:_hdr_end]
                               + '\n#define VQ_IN_COMMON 1  // --no-split: real audio for visualizers\n'
                               + _common_src[_hdr_end:])
                with open(glsl_common_file, 'w') as _f: _f.write(_common_src)
        except Exception as _vq_def_err:
            print(f"   WARNING: failed to inject VQ_IN_COMMON ({_vq_def_err})")
    else:
        try:
            _glsl_sound = base_name + "_shadertoy_sound.glsl"
            if _os2.path.exists(glsl_common_file) and _os2.path.exists(_glsl_sound):
                with open(glsl_common_file) as _f: _common_src = _f.read()
                with open(_glsl_sound)       as _f: _sound_src  = _f.read()
    
                import re as _re3
                # Capture all vqCodes/vqCodebook array decls (multi-line: const ivec4 vqCodes0[…] = …;)
                _arr_pat = _re3.compile(
                    r'(?:^//[^\n]*\n)?const\s+ivec4\s+(?:vqCodes|vqCodebook)\d+\[\d+\]\s*=\s*ivec4\[\]\([^;]+\);',
                    _re3.MULTILINE | _re3.DOTALL)
                _arrays = _arr_pat.findall(_common_src)
                _common_src = _arr_pat.sub('', _common_src)
    
                # Capture the two fetch functions
                _fn_pat = _re3.compile(
                    r'(?://[^\n]*\n)?int\s+fetch(?:Codes|Codebook)Byte\s*\([^)]*\)\s*\{[^}]*\}',
                    _re3.MULTILINE | _re3.DOTALL)
                _fetch_fns = _fn_pat.findall(_common_src)
                _common_src = _fn_pat.sub('', _common_src)
    
                # Capture #define RVQ_* directives — they describe the VQ layout and
                # MUST live with the decoder (Sound prelude).  If Common keeps stale
                # values from an old run while Sound is freshly regenerated, getSample
                # computes wrong vecIdx/lane → audible pitch shift (e.g. vec_dim=4 data
                # decoded with stale vec_dim=2 defines plays 2× too fast).
                _define_pat = _re3.compile(
                    r'^#define\s+RVQ_(?:BITS|BITS_1|K1|K2|VEC_DIM|CB2_BYTE|MASK1|MASK2)\b[^\n]*\n',
                    _re3.MULTILINE)
                _rvq_defines = _define_pat.findall(_common_src)
                _common_src = _define_pat.sub('', _common_src)
    
                # Capture _getRVQCodes helper (uses fetchCodesByte)
                _rvq_pat = _re3.compile(
                    r'(?://[^\n]*\n)*void\s+_getRVQCodes\s*\([^)]*\)\s*\{[^}]*\}',
                    _re3.MULTILINE | _re3.DOTALL)
                _rvq_fns = _rvq_pat.findall(_common_src)
                _common_src = _rvq_pat.sub('', _common_src)
    
                # Capture getSample (4-tap b-spline / lanczos / linear getSampleF too)
                _gs_pat = _re3.compile(
                    r'(?://[^\n]*\n)*float\s+getSample\s*\([^)]*\)\s*\{(?:[^{}]|\{[^}]*\})*\}',
                    _re3.MULTILINE | _re3.DOTALL)
                _gs_fns = _gs_pat.findall(_common_src)
                _common_src = _gs_pat.sub('', _common_src)
    
                _gsf_pat = _re3.compile(
                    r'(?://[^\n]*\n)*float\s+getSampleF\s*\([^)]*\)\s*\{(?:[^{}]|\{[^}]*\})*\}',
                    _re3.MULTILINE | _re3.DOTALL)
                _gsf_fns = _gsf_pat.findall(_common_src)
                _common_src = _gsf_pat.sub('', _common_src)
    
                # Capture getChannelOutput — large function, balanced-brace match
                # It calls getSampleF, so must move to Sound where the decoders live.
                _gco_match = _re3.search(
                    r'(?://[^\n]*\n)*float\s+getChannelOutput\s*\([^)]*\)\s*\{', _common_src)
                _gco_fn = ''
                if _gco_match:
                    _start = _gco_match.start()
                    # Walk braces from the opening { to find matching close
                    _bi = _common_src.index('{', _gco_match.end() - 1)
                    _depth = 1
                    _i = _bi + 1
                    while _i < len(_common_src) and _depth > 0:
                        _c = _common_src[_i]
                        if _c == '{': _depth += 1
                        elif _c == '}': _depth -= 1
                        _i += 1
                    _gco_fn = _common_src[_start:_i]
                    _common_src = _common_src[:_start] + _common_src[_i:]
    
                # Also capture _lanczos3 helper if present
                _lz_pat = _re3.compile(
                    r'(?://[^\n]*\n)*float\s+_lanczos3\s*\([^)]*\)\s*\{[^}]*\}',
                    _re3.MULTILINE | _re3.DOTALL)
                _lz_fns = _lz_pat.findall(_common_src)
                _common_src = _lz_pat.sub('', _common_src)
    
                # Only inject prelude if we actually extracted something; otherwise
                # Common is from the fallback writer which has a different layout
                # (no vqCodes/decoders) and Sound already has what it needs.
                _extracted_anything = bool(_arrays or _fetch_fns or _gs_fns or _gsf_fns or _gco_fn)
                if not _extracted_anything:
                    raise RuntimeError("nothing to extract — Common has no VQ data; using fallback layout")
    
                # Build sound prelude block, prepend to Sound tab
                _prelude_parts = ["// ═══ Sample decoders (moved from Common to fit ANGLE source budget) ═══\n"]
                # Defines first — getSample reads RVQ_VEC_DIM/RVQ_BITS/RVQ_CB2_BYTE etc.
                _prelude_parts.append("// ── RVQ layout defines (must match the data packed below) ──\n")
                _prelude_parts.extend(_rvq_defines)
                _prelude_parts.append("\n")
                _prelude_parts.extend(a + '\n' for a in _arrays)
                _prelude_parts.extend(f + '\n' for f in _fetch_fns)
                _prelude_parts.extend(f + '\n' for f in _rvq_fns)
                _prelude_parts.extend(l + '\n' for l in _lz_fns)
                _prelude_parts.extend(g + '\n' for g in _gs_fns)
                _prelude_parts.extend(g + '\n' for g in _gsf_fns)
                if _gco_fn: _prelude_parts.append(_gco_fn + '\n')
                _prelude_parts.append("// ═══ end sample decoders ═══════════════════════════════════════════════\n\n")
                _prelude = '\n'.join(_prelude_parts)
    
                # Replace any "are in Common" comment with "are above"
                _sound_src = _sound_src.replace(
                    '// getByte / getPatternByte / getSample / getNote / getChannelOutput are in Common.',
                    '// Sample decoders defined above; pattern/getNote/getChannelOutput in Common.')
    
                # Find insertion point: after the file header /* ... */ and any leading #defines
                _hdr_end = _sound_src.find('*/')
                if _hdr_end > 0:
                    _hdr_end = _sound_src.find('\n', _hdr_end) + 1
                else:
                    _hdr_end = 0
                _sound_src = _sound_src[:_hdr_end] + '\n' + _prelude + _sound_src[_hdr_end:]
    
                # Collapse runs of blank lines in common
                _common_src = _re3.sub(r'\n{3,}', '\n\n', _common_src)
    
                with open(glsl_common_file, 'w') as _f: _f.write(_common_src)
                with open(_glsl_sound,       'w') as _f: _f.write(_sound_src)
                _moved_kb = sum(len(a) for a in _arrays) // 1024
                print(f"   ✂️  Moved {len(_arrays)} arrays + {len(_fetch_fns)+len(_gs_fns)+len(_gsf_fns)+len(_lz_fns)} fns "
                      f"({_moved_kb} KB) from Common → Sound prelude")
        except Exception as _splerr:
            print(f"   WARNING: Common→Sound split failed ({_splerr}); leaving Common intact")



    # Summary
    bufA_file_short = base_name + "_shadertoy_bufferA.glsl"

    print(f"\n✅ Generated:")
    print(f"   🌐 HTML Player:    {html_file}")
    _enc_label = "PNG-loaded" if args.use_png else "VQ-encoded"
    print(f"   📁 ShaderToy tabs: {glsl_common_file}  ({_enc_label})")
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
