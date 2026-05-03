#!/usr/bin/env python3
"""
MOD Player with Correct ProTracker Timing + Efficient int32 Storage + Microclick Removal
OPTIMIZATIONS:
  1. Pack 4 bytes per int32 (75% storage reduction)
  2. Volume crossfade microclick removal (C++-style dying channels)

GLSL OUTPUT OPTIMIZATIONS (stable across v1.39+):
  A. fetchPatternInt / fetchSampleInt: ternary chain → vector indexing v[i & 3]
                                       (1 instr vs 4 selects on most drivers)
  C. getPosition: actual binary search through patRowOffset
                  (was a linear scan despite the existing comment)
  D. Default --resampler changed from lanczos3 to bspline:
                  4-tap cubic instead of 6-tap sinc — eliminates 12 sin()
                  calls per sample fetch with no audible loss on 8-bit
                  MOD source (encoder already AA-filters below Nyquist).
                  Pass --resampler lanczos3 explicitly to restore old behavior.
  G. Forward scan in getChannelOutput: rowsInPattern (_posRows) hoisted out,
                                       refreshed only on pattern transitions.
  Sound:  reverb cut from 6 combs × 5 iterations to 4 combs × 3 iterations
          (60% reduction on the hottest path; spectral extremes preserved).

NEW IN v1.39: Windows + ANGLE + NVIDIA crash fixes.
  v1.38 shipped a stack of unroll-defeat hacks (`< N + min(0, x)` bounds,
  forward-declared `uniform int iFrame` in Sound, `#pragma optimize(off)`
  wrapping `getChannelOutput` and `getSampleF`) that targeted GLSL→HLSL
  compile-time blowup. They didn't help, and the `min(0, x)` pattern was
  actively breaking compilation on mobile GLES drivers that need constant
  loop bounds. All of those are removed in v1.39.

  The actual root cause was data location, not loop unrolling: with the
  v1.38 layout, the VQ codebook (~38 KB packed, ~128 KB source as
  `ivec4(...)` literals) lived in Common, which Shadertoy concatenates
  into every pass. So Image, Buffer A, and Sound each separately compiled
  those literals — three independent OOM-prone initializer-list passes
  through fxc. The v1.39 fix:

  VQ data now lives in Sound only; Common stays slim. Buffer A and the
  Image-tab oscilloscope/spectrum can no longer call getChannelOutput,
  so they synthesize an audio-shaped waveform from note pattern data
  instead. Per-instrument waveform is inferred at generation time from
  the sample name:

       drums (kick/snare/hat/cymbal/perc/...)  → noise
       bass / sub / 808                         → sine
       lead / saw / synth / acid                → saw
       square                                   → square
       pluck / pad / string / brass / organ /
              guitar / pizz / harp / choir      → triangle
       everything unmatched                     → sine (safe default)

  Sine is the unmatched fallback: undershooting (sine where saw was right)
  just looks plain on the spectrum; overshooting adds harmonics the real
  audio doesn't have, which reads as wrong. The synth lives in a tiny
  `_synthWave(wt, freq, t)` helper in Common.

  v1.40: --split / --no-split removed entirely. Splitting is now
  unconditional — there is no longer a code path that puts VQ in Common.
  --use-png remains as the alternative data-storage scheme (samples in
  a PNG texture rather than VQ const arrays); it skips the splitter
  because there's nothing VQ-shaped to move.

  --max-compat is preserved but now mostly redundant; with splitting
  unconditional, Common is small enough that the reverb/feature trims
  it bundles aren't usually necessary. Use it only if the Sound tab
  itself is over fxc's threshold (rare unless the MOD has very large
  samples).
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
<title>{mod.title} — GLSL (The Last) MOD Player</title>
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

# ─── Sample-name → waveform classifier (module scope) ───────────────────────
# Used by the visualizer's note-synth path (Buffer A row 0 + Image-tab
# oscilloscope). Both create_shadertoy_glsl AND main()'s VQ-Common post-
# processing need this — that's why it's here at module scope rather than
# inside create_shadertoy_glsl. The encoder-emitted Common doesn't know
# anything about the visualizer, so we inject waveType[] and _synthWave
# into it after the fact.
#
# Returns one of: 0=SINE, 1=SAW, 2=SQUARE, 3=TRIANGLE, 4=NOISE.
# Order matters: more-specific keywords first ('bassdrum' before 'bass',
# 'subbass' before 'sub', 'crash' before 'cr').
_MOD_WAVE_RULES = [
    # (substring, wave_type)  — checked in order, first match wins
    ('bassdrum', 4), ('bass dr', 4), ('bd ',     4), ('bdrum',   4),
    ('kickdrum', 4), ('kick',    4),
    ('snaredr',  4), ('snare',   4), ('clap',    4), ('rim',     4),
    ('hihat',    4), ('hh ',     4), ('hh.',     4), ('hat',     4),
    ('crash',    4), ('cymbal',  4), ('ride',    4), ('cym',     4),
    ('tom',      4), ('shaker',  4), ('perc',    4),
    ('subbass',  0), ('sub bs',  0), ('sub-bass',0), ('sub.bass',0),
    ('808',      0),
    ('bass',     0), (' bs ',    0), ('bs.',     0),
    ('lead',     1), ('saw',     1), ('synth',   1), ('acid',    1), ('square', 2),
    ('pluck',    3), ('guitar',  3), ('pizz',    3), ('harp',    3),
    ('pad',      3), ('string',  3), ('strings', 3), ('brass',   3),
    ('organ',    3), ('choir',   3), ('flute',   3), ('horn',    3),
    ('bell',     0), ('chime',   0), ('glock',   0),
]
_MOD_WAVE_NAMES = ['SINE', 'SAW', 'SQUARE', 'TRIANGLE', 'NOISE']

def _classify_mod_sample_waveform(name):
    """Map a single sample name to a waveType int (0=SINE..4=NOISE)."""
    if not name:
        return 0  # SINE default
    nm = ' ' + name.lower().strip() + ' '
    for kw, wt in _MOD_WAVE_RULES:
        if kw in nm:
            return wt
    return 0  # SINE fallback for unknown

def _classify_mod_waveforms_for(mod, slots=31, verbose=True):
    """Classify all samples in a MOD and optionally print diagnostics.
    Returns a list of `slots` ints (one waveType per instrument slot)."""
    wave_types = [
        _classify_mod_sample_waveform(mod.samples[i]['name']) if i < len(mod.samples) else 0
        for i in range(slots)
    ]
    if verbose:
        diag = []
        for i in range(min(slots, len(mod.samples))):
            nm = mod.samples[i]['name'].strip()
            if nm and mod.samples[i]['length'] > 0:
                diag.append(f"#{i+1} '{nm}' → {_MOD_WAVE_NAMES[wave_types[i]]}")
        if diag:
            print(f"   🎹 Visualizer waveforms: {len(diag)} samples classified")
            for line in diag[:8]:
                print(f"      {line}")
            if len(diag) > 8:
                print(f"      ... and {len(diag) - 8} more")
    return wave_types

def _emit_visualizer_synth_glsl(wave_types):
    """GLSL block defining waveType[31] + _synthWave(). Injected into Common
    so Image and Buffer A can synthesize audio-shaped waveforms from note
    pattern data (the VQ codebook lives only in Sound)."""
    wt_str = ', '.join(str(w) for w in wave_types)
    return (
        "\n// ─── Waveform classifier table (visualizer note-synth path) ──────────\n"
        "//   0 = SINE   (default for unknown / clean tones / bells)\n"
        "//   1 = SAW    (leads, synths, acid)\n"
        "//   2 = SQUARE (square, pulse-style instruments)\n"
        "//   3 = TRIANGLE (plucks, pads, strings, brass)\n"
        "//   4 = NOISE  (kick, snare, hat, cymbal, perc)\n"
        f"const int waveType[31] = int[]({wt_str});\n"
        "\n"
        "// Synthesizer used by Buffer A row 0 and the Image-tab oscilloscope.\n"
        "// VQ codebook lives in Sound only, so visualizers approximate audio\n"
        "// from note pattern data via this 5-way dispatch (~10 lines of math).\n"
        "float _synthWave(int wt, float freq, float t) {\n"
        "    float ph2pi = 6.2831853 * freq * t;\n"
        "    if (wt == 1) return fract(freq * t) * 2.0 - 1.0;                  // SAW\n"
        "    if (wt == 2) return fract(freq * t) < 0.5 ? 0.7 : -0.7;           // SQUARE\n"
        "    if (wt == 3) {                                                    // TRIANGLE\n"
        "        float p = fract(freq * t);\n"
        "        return abs(p * 4.0 - 2.0) - 1.0;\n"
        "    }\n"
        "    if (wt == 4) {                                                    // NOISE\n"
        "        return fract(sin(t * 12345.6789) * 43758.5453) * 2.0 - 1.0;\n"
        "    }\n"
        "    return sin(ph2pi);                                                // SINE (0 + fallback)\n"
        "}\n\n"
    )

def create_shadertoy_glsl(mod, output_file, downsample=1, compress=True, compressed_pattern_size=None, 
                          pattern_bytes_data=None, sample_bytes_data=None, seek_table=None, vec_dim=2, viz=1,
                          compat=None):
    """Generate ShaderToy GLSL code with texture-based OR embedded data.
    viz: 0=None, 1=Reactive 001 (default), 2=Fluxline Surfer, 3=Zuvuya,
         4=Maya tunnel-warp, 5=Dodecahedron (Philip Bertani),
         6=Disco Combined (orblivius/finalman — smoke spotlights + lasers/clouds),
         7=Sparkly 4D (Philip Bertani — 4D IFS fractal raymarcher)
    compat: optional dict of compatibility overrides from --max-compat. Keys:
            no_surround, no_fat, reverb_2x2, fft_n, extra_pragmas. Missing
            keys default to permissive values (full-quality mode)."""

    # Compat defaults — used when the caller didn't pass a compat dict, or
    # when it passed one missing some keys. These match v1.37 default behavior.
    _compat = {
        'no_surround':    False,
        'no_fat':         False,
        'reverb_2x2':     False,
        'fft_n':          512,
        'extra_pragmas':  False,
        'phatbass_mode':  'sample',  # 'auto' | 'sample' | 'mix'
    }
    if compat:
        _compat.update(compat)

    # Human-readable visualizer name (stamped into every tab header).
    _VIZ_NAMES = {
        0: "None (black backdrop)",
        1: "Reactive 001 (PAEz fork — SDF circles + cosmic web)",
        2: "Fluxline Surfer (mrange — DR2 dodecahedron + glowtracer)",
        3: "Zuvuya (city/stars + audio-reactive curtain)",
        4: "Maya (raymarched fractal tunnel-warp)",
        5: "Dodecahedron (Philip Bertani — DR2 IFS fractal raymarcher)",
        6: "Disco Combined (orblivius/finalman — smoke spotlights + lasers/clouds)",
        7: "Sparkly 4D (Philip Bertani — 4D IFS volumetric raymarcher)",
    }
    viz_name = _VIZ_NAMES.get(viz, f"viz{viz}")

    # Per-instrument waveform classification (sine/saw/square/triangle/noise)
    # — used by the visualizer's note-synth path. See _MOD_WAVE_RULES /
    # _classify_mod_sample_waveform at module scope. The same data is also
    # needed by main()'s VQ-Common post-processing step (to inject waveType[]
    # and _synthWave into the encoder-emitted Common), which is why these
    # live at module scope rather than as inner-functions here.
    wave_types = _classify_mod_waveforms_for(mod)
    wave_types_str = ', '.join(str(w) for w in wave_types)

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

    # ── Estimate song duration vs Shadertoy's audio buffer cap (~180s) ──
    # The Sound shader is rendered ONCE into a pre-allocated buffer at
    # compile time, not streamed. After the buffer is exhausted, audio
    # stops while visuals continue. We can't extend playback past this
    # cap from inside GLSL — only Shadertoy's host JS controls the buffer
    # length. What we CAN do is warn the user up front so they know what
    # to expect, and (in mainSound) wrap the time so short songs loop to
    # fill the buffer instead of going silent partway through.
    _bpm_init   = getattr(mod, 'initial_tempo', 125)
    _speed_init = getattr(mod, 'initial_speed', 6)
    _ticks_per_sec = _bpm_init * 2.0 / 5.0
    _row_time      = _speed_init / _ticks_per_sec if _ticks_per_sec > 0 else 0.0
    _song_seconds  = _total_song_rows * _row_time
    _SHADERTOY_AUDIO_CAP_SEC = 180.0  # Shadertoy buffer length (per IQ via ttg)
    _mins, _secs = divmod(_song_seconds, 60)
    print(f"   ⏱️  Song duration: {int(_mins)}m {_secs:.1f}s ({_total_song_rows} rows @ {_bpm_init} BPM, speed {_speed_init})")
    if _song_seconds > _SHADERTOY_AUDIO_CAP_SEC:
        _trunc_pct = (_SHADERTOY_AUDIO_CAP_SEC / _song_seconds) * 100.0
        print(f"   ⚠️  Song longer than Shadertoy's ~180s audio buffer cap.")
        print(f"      Audio will play the first {_SHADERTOY_AUDIO_CAP_SEC:.0f}s ({_trunc_pct:.0f}% of song) then stop.")
        print(f"      Visuals continue running. This is a Shadertoy host limit, not a bug.")
        print(f"      Workarounds: trim the song, raise tempo, or lower speed (=fewer rows/beat).")
    elif _song_seconds < _SHADERTOY_AUDIO_CAP_SEC * 0.5:
        # Song fits comfortably and will loop to fill the buffer
        _loops = _SHADERTOY_AUDIO_CAP_SEC / max(_song_seconds, 0.001)
        print(f"      Fits within Shadertoy's ~180s buffer; loops ~{_loops:.1f}× to fill.")

    # ---- build GLSL arrays ----
    _row_off_str  = ', '.join(map(str, _pat_row_offsets))
    _start_row_str = ', '.join(map(str, _pat_start_rows))
    _intro_rows = _pat_row_offsets[loop_target_songpos] if loop_target_songpos else 0
    _loop_rows  = _total_song_rows - _intro_rows

    # ========== COMMON TAB ==========
    data_source_comment = "Embedded data (no PNG required)" if use_embedded else f"All data in 1024×1024 RGBA PNG: {png_file}"
    common_glsl = f"""/* ============================================================================
   GLSL (The Last) MOD Player v1.40 (c) 2026 Orblivius
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
const bool  enable3D     = {str(not _compat["no_surround"]).lower()};
const bool  enableFAT    = {str(not _compat["no_fat"]).lower()};
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

// Audio playback timing constants — used by both Sound (for playbackTime
// computation) and Image/BufferA (for GUI/visualizer sync with audio).
//
// SONG_DURATION_S: total song length in seconds. Sound wraps playback to
//   this duration (mod) so short songs loop within the audio buffer.
// INTRO_SILENCE_S: silence at start of audio output, while Image renders
//   its loading splash. Sound subtracts this from `time` before wrapping
//   so audio starts at row 0 of the song when audio actually begins, NOT
//   1.5s into the song.
// AUDIO_BUFFER_S: Shadertoy's audio buffer length. After this many seconds
//   of audio output, mainSound stops being called and audio dies. We use
//   this to clamp iTime in the visualizer so the GUI freezes when audio
//   stops (matches what the user actually hears). Bumped to 300 since
//   modern Shadertoy/WebGL2 sometimes delivers more than the legacy 180s.
//   If audio stops earlier than 300, the clamp here won't be reached but
//   the visualizer won't go further than the audio either.
#define SONG_DURATION_S  (float(TOTAL_SONG_ROWS) * float(SPEED) / (float(BPM) * 0.4))
#define INTRO_SILENCE_S  1.5
#define AUDIO_BUFFER_S   180.0

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
    // OPT A: dynamic vector indexing — single MOV vs ternary chain's 4 selects.
    return v[i & 3];
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
    // OPT A: dynamic vector indexing — single MOV vs ternary chain's 4 selects.
    return v[i & 3];
}}

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
    
    common_glsl += ");\n\n"
    common_glsl += _emit_visualizer_synth_glsl(wave_types)
    common_glsl += f"""
// ProTracker period table (C-1 to B-3)
const int periodTable[37] = int[](
    856,808,762,720,678,640,604,570,538,508,480,453,
    428,404,381,360,339,320,302,285,269,254,240,226,
    214,202,190,180,170,160,151,143,135,127,120,113,0
);

// OPT: pre-fold the (2.0 * downsample) divisor into the numerator.
// Original was `7093789.2 / (period * 2.0)` → 1 mul + 1 div.
// Folded form is `PERIOD_TO_FREQ_NUM / period` → 1 div.
// Python computes the constant; emitted as a literal so GLSL doesn't
// re-do the math at compile time on every shader load.
const float PERIOD_TO_FREQ_NUM = {7093789.2 / (2.0 * downsample)};  // 7093789.2 / (2 * downsample)
float periodToFreq(int period) {{
    // Amiga PAL clock / (period * 2 * downsampleFactor)  — pre-folded
    return period > 0 ? PERIOD_TO_FREQ_NUM / float(period) : 0.0;
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
    // OPT: const-qualified — derived purely from #define constants (BPM/SPEED).
    // Compiler should fold these anyway, but explicit `const` is bulletproof.
    const float ticksPerSec  = BPM * 2.0 / 5.0;
    const float rowTime      = SPEED / ticksPerSec;
    const float songDuration = float(TOTAL_SONG_ROWS) * rowTime;

    // OPT: SONG_LOOP_POS is a #define — use #if so the unused branch is
    // never compiled. Saves a runtime comparison + dead code elimination.
    float loopedTime;
#if SONG_LOOP_POS == 0
    loopedTime = mod(time, songDuration);
#else
    const float introDur = float({_intro_rows}) * rowTime;
    const float loopDur  = float({_loop_rows})  * rowTime;
    if (time < songDuration) {{
        loopedTime = time;
    }} else {{
        loopedTime = introDur + mod(time - songDuration, loopDur);
    }}
#endif

    float totalRows = loopedTime / rowTime;

    // OPT C: actual binary search through patRowOffset (was linear despite
    // the comment). Uniform cost regardless of song length — irrelevant for
    // SONG_LENGTH=24 here, but matters for longer modules where this gets
    // re-emitted by the same script.
    int sp_lo = 0, sp_hi = SONG_LENGTH;
    for (int _bi = 0; _bi < 8; _bi++) {{  // ceil(log2(128)) = 7
        if (sp_lo >= sp_hi - 1) break;
        int sp_mid = (sp_lo + sp_hi) >> 1;
        if (float(patRowOffset[sp_mid]) <= totalRows) sp_lo = sp_mid;
        else sp_hi = sp_mid;
    }}
    int sp = sp_lo;
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
// Called ~60× per audio sample on the full audio path. Inlining cost is real,
// but #pragma optimize is non-standard GLSL and ignored or mishandled by every
// backend that matters. The compiler is going to inline this regardless of
// what we ask — if call-site code size becomes a problem, address it by
// shrinking the function body (e.g. fewer reverb taps), not by pragma hints.
float getChannelOutput(int ch, float time, Position pos, float rowTime) {{

    // Step 1: find most-recently-triggered note on this channel
    Note trigNote = getNote(pos.songPos, pos.row, ch);
    int  trigRow  = pos.row;
    int  trigPat  = pos.songPos;
    if (trigNote.instrument <= 0 || trigNote.period <= 0) {{
        int scanRow = pos.row;
        int scanPat = pos.songPos;
        // The early `break` makes this data-dependent — modern shader compilers
        // won't fully unroll a 64-iter loop with a runtime exit. Plain bound.
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
        // OPT G: rows-in-this-pattern hoisted out of the forward scan loop;
        // only refreshed on pattern transitions inside the loop body.
        int _posRows = (_fp < SONG_LENGTH) ? patRowOffset[_fp+1] - patRowOffset[_fp] : 0;
        // Forward scan also has data-dependent breaks; plain bound.
        for (int _fi=0; _fi<64; _fi++) {{
            if (_fp > pos.songPos || (_fp==pos.songPos && _fr>=pos.row)) break;
            if (_fp >= SONG_LENGTH) break;
            // OPT G: _posRows hoisted — only refreshed on pattern transition
            if (_fr >= patStartRow[_fp] + _posRows) {{
                _fp++; _fr = (_fp < SONG_LENGTH) ? patStartRow[_fp] : 0;
                if (_fp < SONG_LENGTH) _posRows = patRowOffset[_fp+1] - patRowOffset[_fp];
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
    # Resolve PhatBass routing mode:
    #   'auto'   → mix-wide if no bass detected, else per-sample (legacy)
    #   'sample' → force per-sample (uses isBass[] flags as encoded)
    #   'mix'    → force mix-wide (Hilbert cross-pan on entire mixdown)
    _pb_mode = _compat['phatbass_mode']
    if _pb_mode == 'sample':
        phatbass_mix_mode = 0
        print(f"   🎚️  PhatBass routing: per-sample (forced via --phatbass-mode sample)")
    elif _pb_mode == 'mix':
        phatbass_mix_mode = 1
        print(f"   🎚️  PhatBass routing: mix-wide (forced via --phatbass-mode mix)")
    else:  # auto
        phatbass_mix_mode = 1 if not _bass_idx else 0
        _routing = 'mix-wide' if phatbass_mix_mode == 1 else 'per-sample'
        print(f"   🎚️  PhatBass routing: {_routing} (auto)")
    
    sound_glsl = f"""/* ============================================================================
   GLSL (The Last) MOD Player v1.40 (c) 2026 Orblivius
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
    // ── Full-precision time from sample index ──────────────────────────
    // The `time` float that Shadertoy passes is `samp/iSampleRate`. By
    // ~150s in, float32 mantissa step exceeds 1/44100 — consecutive
    // samples can land on the SAME float value, freezing audio
    // progression. (This was the "audio quits around pattern 20"
    // symptom: at 153s, float32 step is ~1.5e-5 < sample period
    // 2.27e-5, so playbackTime stops advancing per-sample.)
    //
    // Fix: do all time math in INTEGER samp space first, only convert
    // to float ONCE the integer is small (after subtracting intro
    // silence and any base offset). Float32 has 24-bit mantissa
    // (~16M = 363s @ 44.1kHz), so as long as the integer stays under
    // 16M, the float division gives full precision.
    //
    // Reference: ttg's full-precision timing technique
    // (https://www.shadertoy.com/view/fdBcRV).
    const int SAMP_PER_SEC = 44100;
    const int INTRO_SAMP   = int(INTRO_SILENCE_S * float(SAMP_PER_SEC));

    if (samp < INTRO_SAMP) return vec2(0.0, 0.0);

    // Integer-domain sample index relative to start-of-music.
    int play_samp = samp - INTRO_SAMP;
    // Wrap to song duration in INTEGER samples (avoids float mod precision).
    // SONG_DURATION_S is computed at compile time; convert to integer samps
    // using Common's BPM/SPEED — Common-scope #defines are visible here.
    const int SONG_DURATION_SAMPS = int(SONG_DURATION_S * float(SAMP_PER_SEC));
    play_samp = play_samp - (play_samp / SONG_DURATION_SAMPS) * SONG_DURATION_SAMPS;
    // Now convert to float — value is small (always < song-duration ≤ 363s).
    time = float(play_samp) / float(SAMP_PER_SEC);

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
#endif

    // ── Wrap playbackTime to song duration ─────────────────────────────────
    // Shadertoy renders the Sound pass ONCE into a pre-allocated audio
    // buffer (~180 seconds at 44.1kHz, baked at compile time) — it does not
    // stream audio. After that buffer is exhausted, audio stops while
    // visuals continue (since iTime is independent and live). There is
    // nothing the shader can do to extend playback past the buffer cap.
    //
    // Wrapping playbackTime to songDuration means short songs (<180s) get
    // looped to fill the buffer instead of going silent partway through.
    // Longer songs (>180s) still cut off at the buffer cap regardless —
    // the user only hears the first 180s. Generation-time warning printed
    // in main() when songDuration would exceed the cap.
    //
    // INTRO_SILENCE offset: subtract it from `time` BEFORE wrapping so
    // playback starts at row 0 of the song when audio actually begins,
    // not 1.5s into the song.
    //
    // playbackTime is `time` directly — already corrected for intro silence
    // and wrapped to song duration in the integer-domain block at the top
    // of this function. The previous `mod(time - INTRO_SILENCE_S, ...)`
    // approach lost precision at long times because time as a float was
    // already large; doing the wrap+offset in int-samp space avoids that.
    const float ticksPerSec_ms = BPM * 2.0 / 5.0;
    const float rowTime_ms     = SPEED / ticksPerSec_ms;
    const float songDuration   = float(TOTAL_SONG_ROWS) * rowTime_ms;
    float playbackTime = time;

#if !USE_EMBEDDED_DATA
    // PNG testing-mode override: 10-second loop for fast iteration when
    // editing PNG signature byte. We re-wrap here in float (loopMode is
    // a runtime check so we can't constant-fold). Precision is fine since
    // `time` is already small (<= songDuration after upfront wrap).
    if (loopMode == 255) {{
        playbackTime = mod(time, min(10.0, songDuration));
    }}
#endif
    
    Position pos = getPosition(playbackTime);
    // OPT: const-qualified — same #define-derived constants as getPosition().
    const float ticksPerSec = BPM * 2.0 / 5.0;
    const float rowTime     = SPEED / ticksPerSec;

    // 75% Amiga stereo separation
    // OPT: chL/chR are compile-time constants (SEP = 0.50 hardcoded).
    // Old code rebuilt 8 floats per sample with arithmetic on every call.
    // New code uses literal float[] arrays — zero runtime cost, identical values.
    //   SEP=0.5  →  0.5 + 0.5*0.5 = 0.75,  0.5 - 0.5*0.5 = 0.25
    const float chL[4] = float[](0.75, 0.25, 0.25, 0.75);
    const float chR[4] = float[](0.25, 0.75, 0.75, 0.25);
    
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
    
    // OPT: const-qualified — NUM_CHANNELS is a #define.
    const float normFactor = 2.0 / float(NUM_CHANNELS);
    surrL *= normFactor; surrR *= normFactor;
    centL *= normFactor; centR *= normFactor;
    
    // ── Only3D — surround bus only (ch0+ch3 = outer LEFT pair, ch1+ch2 = dry center) ─
    const float ONLY3D_DELAY = 0.000431;  // 19 samples @ 44100Hz
    const float ONLY3D_DEPTH = 0.12;  // halved — was smearing native LRRL pan
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
    const float PHAT_DEPTH = 1.5;        // Cranked again — user wants OBVIOUS bass
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
        // Mix-wide PhatBass — applied to entire mix at PHAT_DEPTH strength.
        // Was previously attenuated 0.25× extra ("must not overwhelm") but
        // user feedback was "I don't hear it at all" — so the cap is gone.
        // PHAT_DEPTH itself controls overall intensity now.
        centL += pbL * normFactor * PHAT_DEPTH;
        centR += pbR * normFactor * PHAT_DEPTH;
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
    // Gated by enableFAT (also controls PhatBass) — --max-compat sets to false.
    if (enableFAT) {{
        const float FAT_AMOUNT = 1.0;  // matches FAT4X (FIR weights sum to 1.0)
        outL = outL * (1.0 + fat_cs1(outL) * FAT_AMOUNT);
        outR = outR * (1.0 + fat_cs1(outR) * FAT_AMOUNT);
    }}

    // ── Freeverb-inspired parallel comb reverb (stateless) ─────────────────
    // OPTIMIZED: 6 combs × 5 iters → 4 combs × 3 iters
    //   per-sample cost:    120 getChannelOutput + 30 getPosition calls
    //                    →   48 getChannelOutput + 12 getPosition calls
    //   = 60% reduction on the hottest path (typically 60–80% of frame time).
    //
    // Comb selection: kept indices {{0,1,4,5}} of the original 6 to preserve
    // spectral extremes (shortest pair + longest pair); middle delays drop.
    // L/R pans still alternate L/R/L/R for stereo spread.
    //
    // ── Reverb dimensions: --max-compat reduces from 4×3 to 2×2 ──
    //   default (4 combs × 3 iters): full Freeverb-style spread.
    //   --max-compat (2 combs × 2 iters): half the work, narrower stereo,
    //   shorter tail. Still musical — just less roomy.
    {(
        '''const int   N_ITER  = 2;   // (--max-compat: was 3)
    const float RT60    = 2.4;
    const float _decay  = 8.9078 / RT60;  // ln(1000)/RT60
    const float _D[2]  = float[](0.0253, 0.0338);   // shortest + longest only
    const float _pL[2] = float[](0.85, 0.45);
    const float _pR[2] = float[](0.40, 0.80);
    const int   N_COMB  = 2;
    const float COMB_DIV = 2.0;'''
        if _compat["reverb_2x2"] else
        '''const int   N_ITER  = 3;   // iterations per comb (was 5)
    const float RT60    = 2.4;
    const float _decay  = 8.9078 / RT60;  // ln(1000)/RT60

    // Freeverb-inspired comb delays (seconds), mutually prime in samples.
    // Kept original indices {0,1,4,5} → shortest pair + longest pair.
    const float _D[4]  = float[](0.0253, 0.0269, 0.0322, 0.0338);
    const float _pL[4] = float[](0.85, 0.40, 0.80, 0.45);
    const float _pR[4] = float[](0.40, 0.85, 0.45, 0.80);
    const int   N_COMB  = 4;
    const float COMB_DIV = 4.0;'''
    )}
    const float RV_WET  = 0.15;

    vec2 _wet = vec2(0.0);
    // Comb-reverb outer loop: small constant N_COMB. Plain bound.
    for (int _c = 0; _c < N_COMB; _c++) {{
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
    _wet /= COMB_DIV;       // RMS match
    outL += _wet.x * RV_WET;
    outR += _wet.y * RV_WET;

    // ── Buffer-end fade-out ─────────────────────────────────────────────
    // Shadertoy's audio buffer is ~180 seconds (precomputed at compile time,
    // not streamed). When `time` reaches that limit, audio just stops — if
    // the last sample is mid-waveform at high amplitude, the cut is audible
    // as a "halt" or click. Fade the last 0.4s smoothly to zero so the
    // ending sounds intentional rather than truncated. 0.4s ≈ 17640 samples
    // at 44.1kHz — long enough to be smooth, short enough that user only
    // loses a fraction of a row at the very end.
    const float BUFFER_CAP   = 180.0;
    const float FADE_LEN     = 0.4;
    float _fadeT = clamp((BUFFER_CAP - time) / FADE_LEN, 0.0, 1.0);
    // Cosine ease-out: 1.0 at start of fade window, 0.0 at the end
    float _bufFade = 0.5 - 0.5 * cos(_fadeT * 3.14159265);
    outL *= _bufFade;
    outR *= _bufFade;

    return vec2(clamp(outL, -1.0, 1.0), clamp(outR, -1.0, 1.0));
}}
"""
    
    
    # ========== IMAGE TAB ==========
    raw_title   = mod.title.strip() or "UNTITLED"
    title_text  = raw_title[:20]
    title_chars = to_glsl_font_chars(title_text)
    title_len   = len(title_text)
    # Format suffix (" (MOD)" or " (S3M)") rendered SEPARATELY in WHITE
    # right after the title (in YELLOW). Two prints, two colors.
    fmt_text    = " (S3M)" if getattr(mod, 'is_s3m', False) else " (MOD)"
    fmt_chars   = to_glsl_font_chars(fmt_text)
    fmt_len     = len(fmt_text)
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
    else:  # 1, 2, 4, 5, 6, 7 all use _VizScene
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
    elif viz == 5:
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
    elif viz == 6:
        # Disco Combined — orblivius/finalman fork with infinity swirl integrated.
        # Volumetric smoke spotlights (trace()) + audio-reactive infinity swirl
        # layer (was mainImage2 in source, never called there) + iChannel1
        # waveform-driven tunnel distortion.
        #
        # Adaptations vs. standalone:
        #   - iChannel0 noise texture lookup → procedural 3D value noise
        #   - iChannel1 waveform texture sample → synthesized from the four
        #     channel amplitudes (low x → ch0, high x → ch3), boosted to match
        #     Shadertoy waveform amplitude range.
        #   - Final ccol (iChannel1 background overlay) dropped — no equivalent.
        #   - renderScene() left in code but disabled in composite (was the
        #     "wide-shot 3D club view" the user explicitly didn't want).
        #
        # All identifiers prefixed _v6_ to avoid colliding with the rest of
        # the framework.
        viz_scene_block = r"""
// === VIZ 6: Disco Combined (orblivius/finalman + infinity swirl) ===

#define _v6_PI  3.1415926535897932
#define _v6_TAU (2.*_v6_PI)

// ─── Procedural 3D value noise (replaces iChannel0 lookup) ────────────
float _v6_hash3(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float _v6_noiseZ(in vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0 - 2.0*f);
    return mix(mix(mix(_v6_hash3(p+vec3(0,0,0)), _v6_hash3(p+vec3(1,0,0)), f.x),
                   mix(_v6_hash3(p+vec3(0,1,0)), _v6_hash3(p+vec3(1,1,0)), f.x), f.y),
               mix(mix(_v6_hash3(p+vec3(0,0,1)), _v6_hash3(p+vec3(1,0,1)), f.x),
                   mix(_v6_hash3(p+vec3(0,1,1)), _v6_hash3(p+vec3(1,1,1)), f.x), f.y), f.z) - 0.5;
}

// ─── Synthesized waveform sample (replaces iChannel1 texture lookup) ──
float _v6_wave_a0 = 0.0, _v6_wave_a1 = 0.0, _v6_wave_a2 = 0.0, _v6_wave_a3 = 0.0;
float _v6_waveSample(float u01) {
    float u = clamp(u01, 0., 1.) * 3.0;
    float v;
    if (u <= 1.0)      v = mix(_v6_wave_a0, _v6_wave_a1, u);
    else if (u <= 2.0) v = mix(_v6_wave_a1, _v6_wave_a2, u - 1.0);
    else               v = mix(_v6_wave_a2, _v6_wave_a3, u - 2.0);
    return max(v * 3.0, 0.25);   // boost to match Shadertoy waveform range
}

// ─── Constants & state ─────────────────────────────────────────────────
const float _v6_BIG     = 1e30;
const float _v6_EPS     = 1e-10;
const int   _v6_NUM_LIGHTS = 3;
const int   _v6_SPOTS   = 3;
const int   _v6_MAX_SPOTS = 9;
const float _v6_FAR     = 1.0;
const int   _v6_MAX_STEPS = 100;
const float _v6_MIN_STEP = 0.0082;
const float _v6_FLOOR_Y = -0.13;
const float _v6_LIGHT_BASE_W = 0.19;
const float _v6_CONE_W = 0.2;
const float _v6_LIGHT_POW = 3.0;
const float _v6_LIGHT_INTENS = 0.4;
const float _v6_OMNI_LIGHT = 0.1;
const float _v6_SMOKE_CONE_1 = 1.0;
const float _v6_SMOKE_CONE_2 = 2.0;
const float _v6_SMOKE_CONE_3 = 3.0;
const float _v6_NOTHING = -0.1;

struct _v6_Ray { vec3 o; vec3 d; };
struct _v6_Light { vec3 d; vec3 c; float a; };

_v6_Light _v6_lights[3];
vec3      _v6_SPOT_POS[9];
vec4      _v6_SPOT_COL[9];
mat3      _v6_SPOT_ROT[9];

mat4 _v6_rotX4(float a){ float c=cos(a),s=sin(a);
    return mat4(1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1); }
mat4 _v6_rotY4(float a){ float c=cos(a),s=sin(a);
    return mat4(c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1); }
mat4 _v6_rotZ4(float a){ float c=cos(a),s=sin(a);
    return mat4(c,s,0,0, -s,c,0,0, 0,0,1,0, 0,0,0,1); }
mat3 _v6_rotx(float a){ float c=cos(a),s=sin(a); return mat3(1,0,0, 0,c,-s, 0,s,c); }
mat3 _v6_rotz(float a){ float c=cos(a),s=sin(a); return mat3(c,-s,0, s,c,0, 0,0,1); }

vec2 _v6_rotateUV(vec2 uv, float angle) {
    angle *= _v6_TAU;
    return mat2(cos(angle), -sin(angle), sin(angle), cos(angle)) * uv;
}

float _v6_sdCappedCyl(vec3 p, vec2 h) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// ─── 2D helpers for laser/clouds layer ─────────────────────────────────
float _v6_rand2(vec2 p) {
    p *= 2000.0;
    vec3 p3 = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float _v6_noise2(vec2 p) {
    vec2 f = smoothstep(0.0, 1.0, fract(p));
    vec2 i = floor(p);
    float a = _v6_rand2(i);
    float b = _v6_rand2(i + vec2(1, 0));
    float c = _v6_rand2(i + vec2(0, 1));
    float d = _v6_rand2(i + vec2(1, 1));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float _v6_fbm2(vec2 p) {
    float a = 0.5, r = 0.0;
    for (int i = 0; i < 8; i++) { r += a * _v6_noise2(p); a *= 0.5; p *= 2.8; }
    return r;
}
// Lasers radiating from a central point (the "spinning fixture" effect)
float _v6_laser(vec2 p, int num) {
    float r = atan(p.x, p.y);
    float sn = sin(r * float(num) + iTime);
    float lzr  = pow(0.5 + 0.5 * sn, 500.);
    float glow = pow(clamp(sn, 0.0, 1.0), 10.0);
    return lzr + glow;
}
// Mix of fractal noises to simulate moving fog/clouds
float _v6_clouds(vec2 uv) {
    vec2 t = vec2(0, iTime);
    float c1 = _v6_fbm2(_v6_fbm2(uv*3.0)*0.75 + uv*3.0 + t/3.0);
    float c2 = _v6_fbm2(_v6_fbm2(uv*2.0)*0.5  + uv*7.0 + t/3.0);
    float c3 = _v6_fbm2(_v6_fbm2(uv*10.0-t)*0.75 + uv*5.0 + t/6.0);
    float r = mix(c1, c2, c3*c3);
    return r*r;
}

// ─── Cone & plane intersections (for renderScene/volumetric — kept but
//     not used in the visible composite by default) ────────────────────
float _v6_insideCone(vec3 dir, float ang, vec3 o) {
    float oz = dot(o, dir);
    vec3  oxy = o - dir*oz;
    float c = dot(oxy,oxy)/(ang*ang) - oz*oz;
    return smoothstep(20.0, -50.0, c);
}

void _v6_coneRange(vec3 dir, float ang, _v6_Ray r, out float s, out float e) {
    s = _v6_BIG; e = -_v6_BIG;
    float dz = dot(r.d, dir), oz = dot(r.o, dir);
    vec3 dxy = r.d - dir*dz, oxy = r.o - dir*oz;
    float a = dot(dxy,dxy) - dz*dz*ang*ang;
    float b = dot(dxy,oxy) - dz*oz*ang*ang;
    float c = dot(oxy,oxy) - oz*oz*ang*ang;
    float p = 2.*b/a, q = c/a, rr = p*p*0.25 - q;
    if (rr < 0.) return;
    float m = -p*0.5, sr = sqrt(rr);
    if (c < 0.0) {
        if      (m + sr < 0.0) { s = 0.0; e = _v6_BIG; }
        else if (m - sr < 0.0) { s = 0.0; e = m + sr; }
        else                   { s = 0.0; e = m - sr; }
    } else {
        if (m + sr < 0.0) return;
        if (m - sr < 0.0) { s = m + sr; e = _v6_BIG; }
        else              { s = m - sr; e = m + sr; }
    }
}

// ─── Smoke cone field (volumetric noise inside three rotating spotlights) ──
// hit_ids uses BITMASK encoding so multi-cone overlaps decode unambiguously:
//   cone 0 alone = 1, cone 1 alone = 2, cone 2 alone = 4
//   pair overlaps: 0+1=3, 0+2=5, 1+2=6
//   triple: 0+1+2=7
// (The original Doc 12 used additive 1+2+3 which has 3 ambiguous between
//  "cone 2 alone" vs "cones 0+1 overlap" — we fix that here.)
vec2 _v6_maplight(vec3 orp) {
    float t = iTime * 0.025;
    float minm = 1e4, mm = 1e4, hit_ids = 0.0;
    for (int i = 0; i < _v6_SPOTS; ++i) {
        vec3 _rp = orp;
        vec3 rp  = _rp + _v6_SPOT_POS[i];
        rp *= _v6_SPOT_ROT[i];
        float m = _v6_sdCappedCyl(rp, vec2(_v6_CONE_W, 2.0));
        m -= -_v6_LIGHT_BASE_W + length(rp) * 0.2;
        float d = dot(rp, vec3(0., -1., 0.));
        if (m < 0.0 && d >= 0.0) {
            float n  = _v6_noiseZ(_rp*10.0  + vec3(t,     0,0)) - 0.5;
                  n += _v6_noiseZ(_rp*22.5  + vec3(t*1.2, 0,0)) * 0.5;
                  n += _v6_noiseZ(_rp*52.5  + vec3(t*2.0, 0,0)) * 0.5;
                  n += _v6_noiseZ(_rp*152.5 + vec3(t*2.8, 0,0)) * 0.25;
            mm = min(mm, min(n, m));
            mm = min(mm, -0.2);
            hit_ids += float(i + 1);     // Doc 12 additive: 1, 2, 3
        }
        minm = min(abs(m), minm);
    }
    if (hit_ids > 0.0) return vec2(mm, hit_ids);
    return vec2(minm, _v6_NOTHING);
}

void _v6_colorize(vec4 fgc, vec3 pos, vec4 spotcol, float music, inout vec4 color) {
    float flf = pow(inversesqrt(max(length(pos), 1e-3)), _v6_LIGHT_POW) * _v6_LIGHT_INTENS;
    color += fgc * flf * spotcol * music;
}

bool _v6_trace(vec3 ro, vec3 rd, inout vec4 color) {
    color = vec4(0.0);
    vec3 rp = ro;
    float h = 0.0;
    // Music color from the synthesized waveform (replacing the original's
    // pow(texture(iChannel1, vec2(.25, .25)).r * 2., 2.) * 0.5 + 0.15)
    float music = pow(_v6_wave_a0 + _v6_wave_a1 + _v6_wave_a2 + _v6_wave_a3, 2.) * 0.5 + 0.15;
    float sg  = (sin(iTime)        + 1.0) * 0.25;
    float sg2 = (sin(iTime * 0.5)  + 1.0) * 0.25;
    float sg3 = (sin(iTime * 0.25) + 1.0) * 0.25;
    vec4 spcol1 = _v6_SPOT_COL[0] + vec4(0.0, 0.0, sg,  0.0);
    vec4 spcol2 = _v6_SPOT_COL[1] + vec4(0.0, sg2, 0.0, 0.0);
    vec4 spcol3 = _v6_SPOT_COL[2] + vec4(0.0, 0.0, sg3, 0.0);
    for (int i = 0; i < _v6_MAX_STEPS; ++i) {
        rp += rd * max(_v6_MIN_STEP, h * 0.5);
        vec2 hp = _v6_maplight(rp);
        h = hp.x;
        if (rp.z > _v6_FAR) return false;
        if (h < 0.0) {
            vec4 fgc = vec4(abs(h * 0.05));

            // ── Smoke colorize: Doc 12 verbatim ─────────────────────────
            if      (hp.y == _v6_SMOKE_CONE_1) _v6_colorize(fgc, (-_v6_SPOT_POS[0]-rp), spcol1, music, color);
            else if (hp.y == _v6_SMOKE_CONE_2) _v6_colorize(fgc, (-_v6_SPOT_POS[1]-rp), spcol2, music, color);
            else if (hp.y == _v6_SMOKE_CONE_3) _v6_colorize(fgc, (-_v6_SPOT_POS[2]-rp), spcol3, music, color);
            else if (hp.y == _v6_SMOKE_CONE_3 + _v6_SMOKE_CONE_2 + _v6_SMOKE_CONE_1) {
                _v6_colorize(fgc, (-_v6_SPOT_POS[0]-rp), spcol1, music, color);
                _v6_colorize(fgc, (-_v6_SPOT_POS[1]-rp), spcol2, music, color);
                _v6_colorize(fgc, (-_v6_SPOT_POS[2]-rp), spcol3, music, color);
            }

            // ── Floor band: Doc 12 verbatim ─────────────────────────────
            //   collo = vec4(-normalize(pos).y * floorTexture(pos), 1.)
            //   for(i=0..SPOTS+1) if(hp.y == float(i))
            //     color += collo * SPOT_COL[i] + color * vec4(volumetric, 1.)
            // (renderVolumetric is heavy and not present in viz 6 → vec3(0))
            if (rp.y < _v6_FLOOR_Y && rp.y > _v6_FLOOR_Y - 0.0017) {
                // Doc 12 plane intersection: y=-18 plane for distant pos
                float tPlane = -18.0 / (rd.y - 1e-6);
                vec3 pos = ro + rd * tPlane;

                // floorTexture(pos): diagonal-stripe pattern
                vec3 fp = pos; fp.z += fp.x * 0.25;
                vec3 floorC = fract(fp.x * 0.1) > fract(fp.z * 0.1)
                            ? vec3(1.0) : vec3(0.7);

                vec4 collo = vec4(-normalize(pos).y * floorC, 1.0);
                vec3 volumetric = vec3(0.0);  // renderVolumetric stub

                // Doc 12 floor for-loop — preserves the i-as-id indexing
                for (int i = 0; i < _v6_SPOTS + 1; i++) {
                    if (hp.y == float(i)) {
                        color += collo * _v6_SPOT_COL[i]
                               + color * vec4(volumetric, 1.0);
                    }
                }
                return true;
            }
            if (rp.y < _v6_FLOOR_Y) {
                color = vec4(0.0);
                return true;
            }
        }
    }
    return false;
}

vec3 _v6_hueShift(vec3 col, float shift) {
    vec3 m = vec3(cos(shift), -sin(shift) * .57735, 0);
    m = vec3(m.xy, -m.y) + (1. - m.x) * .33333;
    return mat3(m, m.zxy, m.yzx) * col;
}

// ─── Entry point ───────────────────────────────────────────────────────
vec3 _VizScene(vec2 fragCoord) {
    // Audio inputs are intentionally NOT used by this viz.  Globals
    // _v6_wave_a0..a3 stay at their default 0.0 — _v6_waveSample() will
    // therefore return its 0.25 floor, giving a constant "silence" value
    // that just establishes a stable spatial distortion in the swirl.
    // No getChannelOutput calls; viz 6 is purely time-driven.

    vec2 uv = fragCoord.xy / iResolution.xy;
    float aspect = iResolution.x / iResolution.y;

    // Spotlight setup — three brand-color cones sweeping with time + beat
    vec3 spotpos = vec3(0.35, -0.25, 0.0);
    _v6_SPOT_POS[0] = spotpos;
    _v6_SPOT_POS[1] = vec3(-spotpos.x, spotpos.y, spotpos.z);
    _v6_SPOT_POS[2] = vec3( spotpos.y * 0.5, spotpos.y, spotpos.z);

    _v6_SPOT_COL[0] = vec4(0.076, 0.443, 0.392, 0.0);  // teal
    _v6_SPOT_COL[1] = vec4(0.753, 0.584, 0.220, 0.0);  // amber
    _v6_SPOT_COL[2] = vec4(0.569, 0.235, 0.294, 0.0);  // magenta

    // Beat-driven sweep rotations (use mod_player's BPM, not hardcoded 128)
    float beatPhase = iTime * float(BPM) / 60.0;
    float iBeat_v = beatPhase * 0.1;       // damped beat phase
    float iBeatNrg_v = 0.7;
    float iBeatDet_v = 1.0;
    float iBeatAvg_v = 0.5;
    float rotSpeed = 1.0;

    float xrot = -.3 + sin(iTime*.2)*cos(iBeat_v) + iBeatNrg_v*iBeatDet_v*.1
                 + -1.0 + sin(iBeatNrg_v*iBeatDet_v + iBeat_v*rotSpeed - 0.75)*0.25;
    float yrot =  0.5 + .1*iBeatNrg_v*iBeatDet_v*cos(iTime) + sin(iBeat_v*rotSpeed)*0.35;

    _v6_SPOT_ROT[0] = _v6_rotx(xrot)        * _v6_rotz(-yrot);
    _v6_SPOT_ROT[1] = _v6_rotx(xrot)        * _v6_rotz( yrot);
    _v6_SPOT_ROT[2] = _v6_rotx(-1.0 - xrot) * _v6_rotz( yrot);

    // Foreground smoke trace
    vec3 rd = vec3(uv - vec2(0.5), 1.0);
    rd.y /= aspect;
    rd = normalize(rd);
    vec4 col = vec4(0.0);
    float b = _v6_trace(vec3(0.0, 0.0, -1.1), rd, col) ? 0.0 : 1.0;

    // Lasers + clouds overlay (the iconic green diagonal beams + fog)
    float l = (1.+_v6_noise2(vec2(20.0-iTime)))
            * _v6_laser(vec2(uv.x+0.05,
                             uv.y*(0.2 + 20.0*_v6_noise2(vec2(iTime*.20))) + 0.01),
                        25);
    float c = _v6_clouds(uv);
    vec4 col2 = vec4(0, 1, 0, 1) * (uv.y*l + l*uv.y*uv.y) * c;
    col2.rgb = pow(col2.rgb, vec3(5));

    // ── Base floor (extends to horizon where trace didn't hit) ─────────
    // Doc 12 used `b * ccol.rgb` for this — ccol was the iChannel1
    // background texture filling everywhere the smoke-trace missed.
    // Viz 6 has no equivalent texture, so we generate the same diagonal-
    // stripe floor pattern used inside the cones.  Same `pos` (y=-18
    // plane intersection) keeps the pattern continuous between the
    // brightly-lit cone areas and the dim distant floor.
    vec3 baseFloor = vec3(0.0);
    if (rd.y < -0.001) {                  // ray heading downward
        float tPlane = -18.0 / (rd.y - 1e-6);
        vec3 pos = ro + rd * tPlane;
        vec3 fp = pos; fp.z += fp.x * 0.25;
        vec3 floorC = fract(fp.x * 0.1) > fract(fp.z * 0.1)
                    ? vec3(1.0) : vec3(0.7);
        float ny = max(-normalize(pos).y, 0.0);
        baseFloor = ny * floorC * 0.18;   // dim — no cone illumination here
    }

    // Composite: smoke spotlights + lasers/clouds + base floor where trace missed
    vec3 final = b * baseFloor + col.rgb + col2.rgb;

    // NaN/Inf guard (length(max(v=0, v/snd)) etc. can leak NaNs on bad input)
    final = mix(vec3(0.0), final, vec3(equal(final, final)));
    final = clamp(final, 0.0, 4.0);

    return final;
}
"""


    else:  # viz == 7
        # Sparkly 4D — Philip Bertani's 4D IFS volumetric raymarcher.
        # Source: https://shadertoy.com/view/MXyXzz "sparkly 4d gr" by pb.
        # Pure time-based animation; no audio dependency, so this falls
        # through to the standard `_VizScene(C)` dispatch like viz 1/2/4/5.
        viz_scene_block = r"""
// === VIZ 7: Sparkly 4D Fractal (Philip Bertani) ===
// 4D iterated-function-system raymarched as a volumetric cloud, projected
// down via Rodrigues axis-angle rotation.  All identifiers prefixed _v7_.

#define _v7_rot(x)        mat2(cos((x) + vec4(0., 11., 33., 0.)))
// Rodrigues axis-angle rotation: rotates p around `axis` by angle t.
// NOTE: cross order is (p, axis), NOT (axis, p) — flipping them inverts the rotation.
#define _v7_ROT(p,axis,t) (mix((axis)*dot((p),(axis)), (p), cos(t)) + sin(t)*cross((p),(axis)))
// Color formula — second arg from original was unused, dropped here.
#define _v7_H(h)          (cos((h) + vec3(70., 10. + 5.*sin(iTime), 3.))*.7 + .5)
// Scale-factor → log mapping for color modulation
#define _v7_M(c)          (2.*log(1. + (c)))

vec3 _VizScene(vec2 U) {
    vec3 c = vec3(0.);

    // ── Audio-reactive lighting: smoothed bands from Buffer A row 2 ─────
    // Reading TIME-SMOOTHED band values that BufferA computes once per
    // frame and IIR-smooths with asymmetric attack/release (fast rise,
    // slow fall — see BufferA row 2 logic). This is critical: raw FFT
    // bins flicker at frame rate, which would make per-beam color and
    // brightness modulation strobe horribly. The pre-smoothed values
    // give us a stable "loudness in this freq range" reading we can
    // safely use to modulate visuals without flickering.
    //
    // Use cases:
    //   _v7_audioPhase  → additive shift to color cosine phase per channel
    //                     (red shifts on bass, green on mid, blue on highs)
    //   _v7_audioRGB    → direct color tint added to the per-iteration hue
    //   _v7_audioBright → overall loudness, multiplies the per-beam glow
    vec3 _v7_audioPhase = vec3(0.);
    vec3 _v7_audioRGB   = vec3(0.);
    float _v7_audioBright = 0.0;
    if (iChannelResolution[1].x > 1.0) {
        // Read pre-smoothed bands written by Buffer A row 2 px 2-4.
        // These are the OUTPUTS of an IIR low-pass filter (200ms attack,
        // 700ms release) — they cannot flicker because the smoothing is
        // applied at the source.
        float lo = texelFetch(iChannel1, ivec2(2, 2), 0).r;
        float mi = texelFetch(iChannel1, ivec2(3, 2), 0).r;
        float hi = texelFetch(iChannel1, ivec2(4, 2), 0).r;
        // We pass the bands DIRECTLY into the iteration loop so each
        // beam can pick its own band — see per-iteration mapping below.
        _v7_audioPhase  = vec3(lo, mi, hi) * 8.0;
    }

    // 4D ray direction — y treated as z, z as w (the "extra" dimension)
    vec4 rd = normalize(vec4(U - 0.5*iResolution.xy,
                             iResolution.y,
                             iResolution.y * 2.0)) * 80.0;

    float sc, dotp, totdist = 0.0;
    float tt = iTime;

    // ── Per-iteration FFT bin sampling — INDIVIDUAL flicker per beam ─────
    // The IFS has 200 iterations. Each iteration now samples a UNIQUE FFT
    // bin and modulates its own brightness/color independently. The result:
    // 200 separate beams, each pulsing to its own narrow frequency band.
    // When a kick hits a low bin lights up. When a hi-hat hits, its bin
    // lights up. The fixture is no longer "globally pulsing to the music"
    // — it's a 200-channel spectrum analyzer arranged as a rotating 4D
    // light sculpture.
    //
    // Bin mapping: iteration i  →  FFT bin = floor(i * (FFT_N/2) / 200).
    // For FFT_N=256 → 128 bins, so bins ~0..127 are sampled across the 200
    // iterations (with some duplication — many iterations near the same
    // bin, smoothing out spatial reads). For FFT_N=512 → 256 bins, more
    // bin diversity per beam.
    //
    // Color tint: still segmented R/G/B based on iteration position
    // (early=red bass region, mid=green mid region, late=blue treble).
    //
    // Baseline glow stays at 0.018; modulation factor cranked to 50× for
    // strong flicker — quiet bins ≈ no contribution, loud bins explode.
    // Baseline glow toned down 0.018 → 0.013 so the fixture is dimmer
    // when audio is quiet — audio peaks pop harder against a darker base.
    const float _v7_BASE_GLOW = 0.013;

    for (float i = 0.0; i < 200.0; i++) {
        vec4 p = vec4(rd * totdist);

        // Mix 3D subspaces — the source of the 4D character
        p.yzw = _v7_ROT(p.xyz + vec3(0., 0., -1.5),
                        normalize(vec3(sin(17.1/2.), sin(17.1), cos(17.5/3.))),
                        3.83);

        sc = 1.0;                                  // accumulated scale factor
        p.xz = cos(p.xz / 5.0);                    // radial blur (smear in line with observer)
        p.yz *= _v7_rot(tt + cos(tt));             // time-driven rotation

        // Inner IFS — 8 fold/scale iterations
        for (float j = 0.0; j < 8.0; j++) {
            p = abs(p) * 0.89;
            dotp = max(1.0 / dot(p, p), 0.05);
            sc  *= dotp;
            p = abs(p) * dotp - 0.33;
        }

        // "Funky" distance estimate — empirical, not a true SDF
        float dist     = abs(length(p) - 0.05) / sc;
        float stepsize = dist / 20.0;
        totdist += stepsize;

        // ── Light-emission-point binning via grid quantization ───────────
        // The IFS folds rays toward fractal fixed-points — each cluster of
        // converged rays is a "lamp" (visible bright emission point in the
        // fixture). To assign each lamp its own frequency bin, we quantize
        // the post-IFS position to a coarse 3D grid: rays converging to
        // nearby positions land in the SAME grid cell → hash to the SAME
        // bin → respond to the SAME spectral value → same color/brightness.
        //
        // Cell size 0.40 was tuned experimentally — small enough that the
        // fixture has many distinct lamps (50+ visible at any time) but
        // large enough that individual ray noise doesn't fragment a single
        // lamp into multiple flickering bins.
        //
        // The hash is the standard sin-dot pattern; deterministic per cell.
        // Stable across rotation because we hash AFTER all rotations are
        // applied — the lamp's identity tracks its physical fold-point.
        // Coarse 2.5-unit cells (was 1.25) — fewer unique cells in view
        // means much larger contiguous "lamp" regions, which reads as
        // smooth glow blobs instead of speckled noise. Combined with the
        // halved modulation factor below, the granular look is gone.
        vec3  _cell    = floor(p.xyz * 0.4);
        float _cellKey = abs(fract(sin(dot(_cell, vec3(127.1, 311.7, 74.7))) * 43758.5453));
        int   _binIdx  = clamp(int(_cellKey * float(FFT_N/2)), 0, FFT_N/2 - 1);
        float _binVal  = max(0.0, texelFetch(iChannel1, ivec2(_binIdx, 1), 0).r);

        // ── Frequency-mapped FULL RAINBOW palette ─────────────────────────
        // Each bin's frequency position (0=bass, 1=treble) maps to a hue
        // angle around the color wheel. Three offset cosines for R/G/B
        // give a smooth rainbow:
        //   0.00 (bass)     → red
        //   0.17            → orange/yellow
        //   0.33 (low-mid)  → green
        //   0.50 (mid)      → cyan
        //   0.67 (high-mid) → blue
        //   0.83            → violet/magenta
        //   1.00 (treble)   → red (cycle)
        // So when a bass kick hits → its mapped beam glows RED.
        // When a hi-hat hits → its beam glows BLUE/VIOLET.
        // When a mid-range melody plays → beams glow GREEN/CYAN.
        // The fixture becomes a literal frequency rainbow.
        float _freqPos   = float(_binIdx) / float(FFT_N/2 - 1);
        float _hueAngle  = _freqPos * 6.28318;
        vec3  _freqColor = 0.5 + 0.5 * vec3(
            cos(_hueAngle),                // red branch
            cos(_hueAngle - 2.094),        // green branch (120° shift)
            cos(_hueAngle - 4.189)         // blue branch (240° shift)
        );

        // Per-iteration brightness driven by THIS beam's bin.
        // Modulator stays at 50× — quiet bins basically vanish,
        // loud bins blaze in their frequency's signature color.
        // Modulation lowered (22 → 12) — combined with the much coarser
        // cells above, the granular noise drops to essentially zero. The
        // trade-off is less aggressive flicker per beat, but the lamps
        // still pulse visibly with the audio.
        float _local_glow = _v7_BASE_GLOW * (1.0 + _binVal * 12.0);

        // ── Color blending ───────────────────────────────────────────────
        // Structural cosine hue from the IFS shape, mixed toward the
        // frequency-rainbow color by bin loudness. Strong tint factor
        // (15× clamped at 0.92) so even modest bin values pull the beam
        // strongly toward its frequency color.
        vec3 _structHue = cos(_v7_M(sc) + vec3(70., 10. + 5.*sin(iTime), 3.) + _v7_audioPhase) * 0.7 + 0.5;
        vec3 _hue = mix(_structHue, _freqColor, clamp(_binVal * 15.0, 0.0, 0.92));

        c += mix(vec3(1.), _hue, 0.9) * _local_glow * exp(-i*i*stepsize*stepsize * 1e3);
    }

    return 1.0 - exp(-c);                          // tone-map
}
"""


    image_glsl = f"""/* ============================================================================
   GLSL (The Last) MOD Player v1.40 (c) 2026 Orblivius
   3D Surround, PhatBass, Comb Reverb, FAT, RVQ sample compression, configurable resampler
   IMAGE TAB — iChannel0: alphabet texture (shadertoy.com/view/4sf3RB)
   Visualizer: {viz_name}
   Contact: subband@gmail.com or
            subband@protonmail.com
   GIT:     https://github.com/mewza/mod2glsl
  ============================================================================ */

// FFT_N and FFT_SR must match Buffer A — used by spectrum view to index
// Buffer A row 1 and map screen-x to frequency. If you change FFT_SR
// here you MUST change it in Buffer A too (see #define FFT_SR there).
#define FFT_N  {_compat["fft_n"]}
#define FFT_SR 8192.0


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
makeStr(printFormat) {fmt_chars} _end
makeStr(printBPMVal) {bpm_val_chars} _end
makeStr(printSpdVal) {spd_val_chars} _end

// ---- Static label strings ----
makeStr(printHdr)   _NUM _NUM _NUM _ _G _L _S _L _ _M _O _D _ _P _L _A _Y _E _R _ _V _1 _DOT _4 _0 _ _NUM _NUM _NUM _end
makeStr(printCredit) _COPY _2 _0 _2 _6 _ _O _R _B _L _I _V _I _U _S _end
makeStr(printLoad)   _L _O _A _D _I _N _G _DOT _DOT _DOT _end
makeStr(printSpec)   _S _P _E _C _T _R _U _M _end
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

// ─── GUI/audio sync via iTime macro (file scope) ───────────────────────
// Re-route every `iTime` reference (in helper functions and in mainImage)
// through the same time transformation Sound's mainSound applies. This
// makes the visualizer's iTime track the AUDIO PLAYBACK time, not the
// shader's wall-clock runtime.
//
// Three corrections applied:
//   1. Subtract INTRO_SILENCE_S so visualizer is at row 0 when audio is
//      at row 0 (during the silence, viz is at row 0; without this it
//      would be 12+ rows ahead of audio).
//   2. Clamp at AUDIO_BUFFER_S so visualizer freezes when Shadertoy's
//      audio buffer is exhausted (audio dies — viz should too).
//   3. mod by SONG_DURATION_S so short songs that loop in the audio
//      buffer also loop in the visualizer.
//
// Why no infinite recursion: GLSL's preprocessor never expands a macro
// inside its own replacement text. The inner `iTime` is the real uniform.
//
// Must be at file scope (BEFORE viz_scene_block) so helper functions
// also get the substitution.
#define iTime mod(clamp(iTime - INTRO_SILENCE_S, 0.0, AUDIO_BUFFER_S - INTRO_SILENCE_S), SONG_DURATION_S)

{viz_scene_block}


void mainImage(out vec4 O, vec2 C) {{
    vec2 fp = vec2(C.x, iResolution.y - C.y);

    // (iTime is clamped at file scope via #define above — no local
    // override needed here; every iTime reference in this function
    // and in the visualizer helpers transparently uses min(iTime, 60).)

    const vec3 BG     = vec3(0.00,0.00,0.08);
    const vec3 CYAN   = vec3(0.15,0.75,1.00);   // less green, more blue feel
    const vec3 YELLOW = vec3(1.00,0.90,0.10);
    const vec3 RED    = vec3(1.00,0.28,0.05);
    const vec3 WHITE  = vec3(1.00,1.00,1.00);   // pure white — was 0.88,0.88,0.96 (slightly bluish, looked gray)
    const vec3 BLUE   = vec3(0.30,0.55,1.00);
    const vec3 GREEN  = vec3(0.10,1.00,0.22);
    const vec3 DIM    = vec3(0.22,0.22,0.42);
    const vec3 TC0    = vec3(0.50,1.00,0.30);   // lime green — back per user request
    const vec3 TC1    = vec3(1.00,0.90,0.10);
    const vec3 TC2    = vec3(1.00,0.45,0.90);
    const vec3 TC3    = vec3(1.00,0.55,0.10);

    const float CH=28., CW=25., ML=10.;
    // 90 frames @ 60Hz = 1.5s — long enough that the loading dialog is
    // actually readable. Anything under ~30 frames flashes by too fast to
    // register; over ~150 starts to feel like the page is broken.
    const int LOADING_FRAMES = 90;

    // ── Per-channel amplitudes computed inside viz_setup_block when needed
    float _tps=float(BPM)*2./5., _rt=float(SPEED)/_tps;
    Position _pos=getPosition(iTime);
{viz_setup_block}

    if (iFrame < LOADING_FRAMES) {{
        vec2 res = iResolution.xy;
        float prog = float(iFrame) / float(LOADING_FRAMES - 1);

        // (data arrays paged in by compiler — no runtime loop needed)

        // Header
        col += vec3(0.45, 0.70, 1.20) * printHdr   (pUV(fp, ML, 6., CH));
        col += WHITE  * printCredit(pUV(fp, res.x - ML - 234., 6., CH*0.75));
        col += YELLOW * printTitle (pUV(fp, ML, CH+9., CH));
        col += WHITE  * printFormat(pUV(fp, ML + float({title_len}) * CW, CH+9., CH));

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
    trk += vec3(0.45, 0.70, 1.20) * printHdr   (pUV(fp, ML,  6., CH));
    trk += WHITE  * printCredit(pUV(fp, iResolution.x - ML - 234., 6., CH*0.75));
    trk += YELLOW * printTitle (pUV(fp, ML, CH+9., CH));
    trk += WHITE  * printFormat(pUV(fp, ML + float({title_len}) * CW, CH+9., CH));
    // hlines extended to frame edges (3px inset from canvas) so they
    // visually CONNECT with the left and right vertical frame borders.
    // Was [ML, iResolution.x-ML] which left ~7px gaps on each side.
    trk += BLUE * 0.55 * hline(fp, CH*2.+13., 1., iResolution.x-1.);

    // ============ INFO BAR ============
    float iy  = CH*2.+20.;
    float iy2 = iy + CH + 4.;
    float rx  = iResolution.x*0.52;

    trk += BLUE   * printPatt(pUV(fp, ML, iy, CH));
    trk += WHITE  * drawNum(pos.songPos, 2, ML+10.*CW, iy, CW,CH,fp);
    trk += BLUE   * drawCh(47,fp, ML+11.*CW, iy, CW,CH);
    trk += YELLOW * drawNum({mod.num_patterns-1}, 2, ML+13.*CW, iy, CW,CH,fp);

    trk += BLUE   * printRow(pUV(fp, rx, iy, CH));
    trk += WHITE  * drawNum(pos.row, 2, rx+5.*CW, iy, CW,CH,fp);
    trk += BLUE   * drawCh(47,fp, rx+6.*CW, iy, CW,CH);
    trk += YELLOW * drawCh(54,fp, rx+7.*CW, iy, CW,CH);
    trk += YELLOW * drawCh(52,fp, rx+8.*CW, iy, CW,CH);

    trk += BLUE   * printBPM(pUV(fp, ML,  iy2, CH));
    trk += YELLOW * printBPMVal(pUV(fp, ML+5.*CW, iy2, CH));
    trk += BLUE   * printSpd(pUV(fp, rx,  iy2, CH));
    trk += YELLOW * printSpdVal(pUV(fp, rx+7.*CW, iy2, CH));

    trk += BLUE * 0.55 * hline(fp, iy2+CH+4., 1., iResolution.x-1.);

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

    // Vertical separators — bounded to end at the bottom hline (tBot+3),
    // not extending below into the spectrum/oscilloscope area.
    // Math: tBot = ty + CH+3 + (2*HVR+1)*CH = ty + (2*HVR+2)*CH + 3
    //       tBot + 3 = ty + (2*HVR+2)*CH + 6
    for(int tc=1;tc<NUM_CHANNELS;tc++)
        // Verticals stop 1px short of the bottom hline (tBot+3) so the
        // T-junctions don't have a double-bright dot from additive trk
        // accumulation overlapping vline + hline at the same pixel.
        // Vline starts at ty+CH+1 (just BELOW the "TRACK #N" header so the
        // line doesn't cut through the title text — that was creating the
        // "boxes around text" artifacts the user noticed) and ends at
        // tBot+4 (1px short of the bottom hline at tBot+5 to avoid bright-
        // dot doubling at the T-junction).
        trk += BLUE*0.55*vline(fp, txOff+float(tc)*TW-4., ty+CH+1., ty+float(2*HVR+2)*CH+7.);

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
        // V7 stays at full intensity across the tracker — only the zebra
        // rows do the darkening (alternating "even" rows get mix(col,
        // black, 0.38)). The previous col*=0.45 was too aggressive and
        // made the area look cut off from the rest of the player.
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
            vec3 rnc = ri_abs==frameRow ? WHITE : (on4 ? YELLOW*0.7 : TC3*0.7);
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
                if(isEmpty) nc = DIM*fade*1.6;     // empty "0 0 0 0" — boosted 60% so rows are readable
                else        nc = (ri==0 ? TCols[tc] : TCols[tc]*fade);
                trk += nc*g;
            }}
        }}
    }}
    trk += BLUE * 0.55 * hline(fp, tBot+5., 1., iResolution.x-1.);

    // ── Composite: PREMULTIPLIED ALPHA-OVER ────────────────────────────────
    // The previous `col = mix(col, trk, trkA)` was squaring the glyph value:
    // for a YELLOW glyph at AA opacity 0.85, trk was YELLOW*0.85 (already
    // premultiplied), and mix() then re-multiplied by 0.85 — so the
    // effective contribution was YELLOW*0.85*0.85 ≈ YELLOW*0.72. That's
    // why the title and (MOD) suffix looked dimmer than they should.
    // Standard alpha-over for premultiplied source is just:
    //   col = col*(1-α) + trk
    // With trk already containing the premultiplied color, this puts text
    // at full intensity wherever it's fully drawn.
    float trkA = clamp(max(max(trk.r, trk.g), trk.b), 0.0, 1.0);
    col = col * (1.0 - trkA) + trk;

    // ============ OSCILLOSCOPE / SPECTRUM (mouse click toggles) ============
    // iChannel1 = Buffer A → row 0 = audio, row 1 = FFT mags, row 2 = toggle state
    // Strip y-bounds: top sits 1px below the bottom-of-tracker hline
    // (was +8, now +4 — bars now reach close to that hline at peak),
    // bottom extends ALL the way to the frame's bottom border so bars
    // are flush with it. The frame's 2px-thick band overrides the last
    // 2px of bar at by1, giving a clean visual where bars touch frame.
    float oy=tBot+4.;
    float by1 = iResolution.y - 4.;
    float oh  = max(0., by1 - oy);
    // specMode persisted in Buffer A row 2, px 0 (click to toggle)
    bool specMode = (iChannelResolution[1].x > 1.0)
                    ? texelFetch(iChannel1, ivec2(0, 2), 0).r > 0.5
                    : (iMouse.z > 0.0);  // fallback if Buffer A not connected

    // Strip rendering — bounded to inside the outer frame so content
    // doesn't draw outside the visible window or cross the border line.
    // Frame goes flush to screen edges (bx0=1, bx1=iResolution.x-1) so
    // the 2px content margin keeps the spectrum/oscilloscope inside.
    // ── Pre-compute rounded-rect SDF early so the strip render can
    // gate itself against the curve at the bottom corners.
    const float CR_pre = 10.0;
    vec2 _rectCenter_pre = vec2(iResolution.x * 0.5, by1 * 0.5);
    vec2 _rectHalf_pre   = vec2(iResolution.x * 0.5, by1 * 0.5);
    vec2 _q_pre = abs(fp - _rectCenter_pre) - _rectHalf_pre + vec2(CR_pre);
    float _sdf_pre = min(max(_q_pre.x, _q_pre.y), 0.0) + length(max(_q_pre, vec2(0.0))) - CR_pre;

    if(oh>20.&&fp.y>oy&&fp.y<by1 && fp.x>2.0 && fp.x<iResolution.x-2.0 && _sdf_pre < -1.5) {{
        float sy = fp.y - oy;
        // Dim V7 under the strip — was 0.35; now 0.50 so the visualizer
        // reads brighter as ambient context behind the spectrum/oscillo
        // content (per user feedback: "brighten the backdrop").
        col *= 0.50;

        if (!specMode) {{
            // ── Oscilloscope (real audio from Buffer A row 0) ────────────
            // The synthesis approach (sum of sines from active notes) was
            // the source of all the chaos: 4 channels at different freqs
            // sum into a multi-tone signal that LOOKS like noise no matter
            // the time window. The fix: read directly from Buffer A row 0,
            // which already has the actual mixed mono audio waveform that
            // feeds the FFT. That's REAL audio sampled at FFT_SR (8192Hz),
            // FFT_N samples wide. A real oscilloscope, not a re-synthesis.
            //
            // Window: FFT_N samples / 8192Hz = 15.6ms (256 samples / 8192).
            // Across screen width that's plenty of detail for one or two
            // bass cycles, with content above ~2kHz visible as ripple.
            //
            // Linear filtering via texture() smooths the 128-sample data
            // across ~1000 screen pixels — no stair-step artifacts.
            if (iChannelResolution[1].x > 1.0) {{
                // ── Min/Max scan: continuous, antialiased, no dot artifacts
                // The previous "compute waveY at this pixel only" approach
                // dropped pixels when slope was steep (line jumped 50px but
                // each pixel only filled 3px around its waveY). The fix:
                // for each pixel, scan its horizontal extent (±0.5 pixel)
                // collecting the MIN and MAX wave height seen in that window.
                // Then fill from min to max — guarantees continuous coverage
                // because adjacent pixels' windows TILE: pixel x's right edge
                // = pixel x+1's left edge, so the wave values overlap.
                //
                // For a slow wave (min ≈ max) → thin line.
                // For a fast wave → naturally thicker filled band — exactly
                // what real oscilloscopes show on high-frequency content.
                //
                // GPU bilinear filtering via texture() handles sub-sample
                // smoothness — no cusps between sample points.
                int   bufW   = int(iChannelResolution[1].x);
                int   maxIdx = min(FFT_N - 1, bufW - 1);
                float bufY   = 0.5 / iChannelResolution[1].y;

                // 9 samples spanning ±0.9 pixels gives proper horizontal AA:
                // adjacent pixel columns now SHARE samples (overlap), so
                // wMin/wMax transition smoothly instead of stair-stepping.
                // The previous N_SAMP=5 over ±0.5 only sampled within one
                // pixel — adjacent columns had completely independent ranges,
                // creating visible step artifacts on the wave.
                const int N_SAMP = 9;
                float wMin = 1e10;
                float wMax = -1e10;
                for (int i = 0; i < N_SAMP; i++) {{
                    float xo = C.x - 0.9 + float(i) * (1.8 / float(N_SAMP - 1));
                    float xn = xo / iResolution.x;
                    // Texture coord with +0.5 offset for pixel centers
                    float texX = (xn * float(maxIdx) + 0.5) / float(bufW);
                    float audioVal = texture(iChannel1, vec2(texX, bufY)).r;
                    float amp   = clamp(audioVal * 3.0, -0.9, 0.9);
                    float wy    = (amp * 0.40 + 0.5) * oh;
                    wMin = min(wMin, wy);
                    wMax = max(wMax, wy);
                }}

                // Mid-line baseline first (drawn underneath the trace).
                col = mix(col, DIM*0.18, step(abs(sy - oh*0.5), 0.4));

                // Tighter antialiasing — sharper core, dimmer glow halo.
                float distInside = max(wMin - sy, sy - wMax);
                float core = 1.0 - smoothstep(-0.3, 1.0, distInside);
                float glow = exp(-max(distInside, 0.0) * 1.0);
                float aa   = max(core, glow * 0.30);
                if (aa > 0.0) {{
                    // Center-symmetric vertical gradient: trace is brightest
                    // at the strip's vertical center (sy = oh/2) and fades
                    // to a darker edge color symmetrically toward both top
                    // and bottom edges. dist_from_center is 0 at center,
                    // 1 at edges — used to mix bright center color toward
                    // the edge color.
                    float dist_from_center = abs(sy - oh*0.5) / (oh*0.5);
                    vec3 traceCenter = vec3(0.50, 0.90, 1.15);   // bright baby-blue
                    vec3 traceEdge   = vec3(0.04, 0.15, 0.75);   // saturated dark blue
                    vec3 traceCol = mix(traceCenter, traceEdge, dist_from_center);
                    col = mix(col, traceCol, aa);
                }}
            }} else {{
                col = mix(col, DIM*0.18, step(abs(sy - oh*0.5), 0.4));
            }}
        }} else {{
            // ── FFT Spectrum from Buffer A (iChannel1 row 1) ─────────────
            if (iChannelResolution[1].x > 1.0) {{
                // Buffer A connected — read DFT magnitudes with proper
                // oversampling for smooth visual bars.
                //
                // Why naive texelFetch is blocky: FFT_N=256 → 128 bins.
                // After pow(0.5) log-warping, mid-screen has ~10 screen
                // pixels per bin. Within each bin, the height is
                // determined by a single magnitude value and adjacent
                // bins often have similar values — so the bar stays flat
                // across those 10 pixels and you see clear stair steps.
                //
                // The fix is to sample MULTIPLE bins per screen pixel,
                // Lorentzian-peak spectrum. Each FFT bin contributes a
                // Lorentzian (Cauchy) lineshape:
                //
                //   I(x) = h_i / (1 + ((x - x_i) / γ)^2)
                //
                // where x_i is the bin's screen position, h_i is the bin's
                // magnitude, and γ controls peak width. Small γ = narrow
                // sharp spikes (γ=0.4 here, in bin units). The total
                // intensity at a pixel is the sum of contributions from
                // all nearby bins. Far-away bins contribute essentially
                // zero so we only sum over a small window around the
                // current pixel's bin position.

                // ── LINEAR mapping with horizontal AA oversampling ────────
                // Each bin gets the same fixed pixel width. Linear (not log).
                // START_BIN=2 trims DC and ~16Hz (sub-bass mud).
                //
                // To antialias the polyline: sample 5 points across each
                // pixel's horizontal width and average the magnitudes. This
                // smooths the step transitions between adjacent bins (where
                // each bin spans ~4 pixels of screen, the edges between bins
                // would otherwise be visible as kinks). The TOP edge of the
                // filled spectrum gets a 1px smoothstep AA on top of that.
                // Heights/gains: NOT touched; only smoothness improved.
                const int START_BIN = 2;
                const int N_SAA     = 5;
                float xf            = C.x / iResolution.x;
                float magSum        = 0.0;
                for (int i = 0; i < N_SAA; i++) {{
                    float xo  = (C.x - 0.5 + float(i) / float(N_SAA - 1)) / iResolution.x;
                    float bF  = float(START_BIN) + xo * float(FFT_N/2 - 1 - START_BIN);
                    int   b0  = clamp(int(floor(bF)), START_BIN, FFT_N/2 - 1);
                    int   b1  = min(b0 + 1, FFT_N/2 - 1);
                    float tt  = bF - float(b0);
                    float h0  = max(0.0, texelFetch(iChannel1, ivec2(b0, 1), 0).r);
                    float h1  = max(0.0, texelFetch(iChannel1, ivec2(b1, 1), 0).r);
                    float bn  = float(b0 - START_BIN) / float(FFT_N/2 - 1 - START_BIN);
                    float fg  = 1.0 + bn * 4.0;
                    magSum   += mix(h0, h1, tt) * fg;
                }}
                float mag      = magSum / float(N_SAA);
                // Bars reach to 95% of strip height at peak — leaves a small
                // 5% margin so peaks don't clip flat against the very top
                // line. User wanted bars closer to the top.
                float barH     = clamp(mag * 1.4, 0.0, 1.0) * oh * 0.95;
                float barY = oh - barH;
                // Tighter 1px smoothstep on top edge — combined with the
                // horizontal oversampling, gives crisp but smooth bar tops.
                float edge = smoothstep(barY - 1.0, barY + 1.0, sy);
                if (edge > 0.0) {{
                    // ── VU-meter style coloring ─────────────────────────
                    // (fp uses inverted Y so sy=0 is TOP of strip.)
                    //   strip_t < 0.15 → RED                 peak/clipping
                    //   0.15..0.28     → RED → YELLOW        hot transition
                    //   0.28..0.40     → YELLOW → CYAN       back to base
                    //   0.40..1.00     → babyblue → blue-green darker
                    // The "base" color at strip_t=0.40 is now a bright
                    // baby-blue/sky-blue (more blue than cyan, brighter),
                    // and the bottom is a darker, more green-tinted teal.
                    // Top is "blueish" with extra punch, bottom is "blue-
                    // green" — strong hue + brightness gradient down.
                    float strip_t = sy / oh;
                    vec3 topBlue   = vec3(0.45, 0.85, 1.10);   // bright baby-blue
                    vec3 bottomBG  = vec3(0.04, 0.15, 0.75);   // saturated dark BLUE (different hue from teal)
                    vec3 specCol;
                    if (strip_t < 0.15) {{
                        specCol = RED;
                    }} else if (strip_t < 0.28) {{
                        specCol = mix(RED, YELLOW, (strip_t - 0.15) / 0.13);
                    }} else if (strip_t < 0.40) {{
                        specCol = mix(YELLOW, topBlue, (strip_t - 0.28) / 0.12);
                    }} else {{
                        specCol = mix(topBlue, bottomBG, (strip_t - 0.40) / 0.60);
                    }}
                    col = mix(col, specCol, edge);
                }}
                // Mode label at TOP-RIGHT of the strip, slightly bigger
                // (CH*1.0 = 28px tall, was CH*0.8). SPECTRUM = 8 chars at
                // base CW=25 = 200px wide, so x = iResolution.x - 200 - 8.
                col += DIM * 0.6 * printSpec(pUV(fp, iResolution.x - 8.*CW - 2., oy+4., CH));
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

        // Label: show mode name (DIM in both modes — was CYAN/GREEN
        // bright which was distracting; now both labels read as a
        // subtle "you're in this mode" rather than competing with the
        // visualizer).
        // Label: show mode name at TOP-RIGHT of strip, slightly bigger.
        // Both modes use DIM*0.6 — subtle, doesn't compete with content.
        // SPECTRUM (8 chars), OSCILLOSCOPE (12 chars) — text width =
        // n*CW at full CH size; right-align with 8px margin to frame.
        if (fp.y > oy + 4. && fp.y < oy + 4. + CH) {{
            if (specMode)
                col += DIM * 0.6 * printSpec(pUV(fp, iResolution.x - 8.*CW - 2., oy+4., CH));
            else
                col += DIM * 0.6 * printOsci(pUV(fp, iResolution.x - 12.*CW - 2., oy+4., CH));
        }}
    }}


    // ── Rounded-corner frame + content clipping mask (anti-aliased) ───────
    // Reuses the SDF computed earlier (_sdf_pre) — no need to re-derive.
    vec3 frameCol = BLUE * 0.55;
    float _outside = smoothstep(-0.5, 0.5, _sdf_pre);
    float _frame   = smoothstep(-2.5, -1.5, _sdf_pre) * (1.0 - _outside);
    col = mix(col, frameCol, _frame);
    col = mix(col, vec3(0.0), _outside);

    O = vec4(col, 1.0);
}}
"""


    # ========== BUFFER A TAB — FFT Spectrum Analyzer ==========
    # Row 0 (y=0): FFT_N audio samples collected via getChannelOutput
    # Row 1 (y=1): FFT_N/2 DFT magnitude bins (phasor-rotation, reads row 0 prev frame)
    # Setup: Buffer A iChannel0 = Buffer A (self-ref)
    #        Image   iChannel1 = Buffer A output
    buffer_a_glsl = f"""/* ============================================================================
   GLSL (The Last) MOD Player v1.40 (c) 2026 Orblivius
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

#define FFT_N     {_compat["fft_n"]}
#define FFT_SR    8192.0
#define HIST_ROWS 64      // rows 3..(3+HIST_ROWS-1)
#define HIST_BASE 3
#define WAVE_BASE 70      // rows 70..(70+WAVE_ROWS-1) — Zuvuya waveform scroll memory
#define WAVE_ROWS 64      // 64 rows of history, full-width x

// File-scope iTime clamp — see Image tab for explanation.
#define iTime mod(clamp(iTime - INTRO_SILENCE_S, 0.0, AUDIO_BUFFER_S - INTRO_SILENCE_S), SONG_DURATION_S)

void mainImage(out vec4 O, vec2 C) {{
    int px = int(C.x), py = int(C.y);
    O = vec4(0.0);

    // (iTime is clamped at file scope via #define above.)

    // ── Loading splash: skip all per-pixel work for the first 16 frames. ───
    // Image tab is showing its LOADING progress bar during this window, so
    // BufferA's output isn't being read anyway — no point burning GPU on
    // FFT, audio synth, or oscilloscope history while the GPU is still
    // warming caches and JITing native ISA. Must match Image's threshold.
    if (iFrame < 16) return;

    if (py == 0 && px < FFT_N) {{
        // ── Row 0: mixed audio sample at time-offset px ────────────────────
        // Synthesizes from note pattern data via _synthWave (waveType[]
        // dispatch on instrument). Real audio (getChannelOutput) lives in
        // Sound only — the VQ codebook isn't accessible here by design.
        float dt  = 1.0 / FFT_SR;
        float t   = iTime - float(FFT_N - px - 1) * dt;
        float ticksPerSec = float(BPM) * 2.0 / 5.0;
        float rowTime = float(SPEED) / ticksPerSec;
        Position pos = getPosition(t);
        float s = 0.0;
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
            float env   = exp(-age * 3.5);
            int wt = waveType[tn.instrument - 1];
            s += amp * env * _synthWave(wt, f, t);
        }}
        s /= float(NUM_CHANNELS);
        O = vec4(s, 0.0, 0.0, 1.0);

    }} else if (py == 1 && px < FFT_N / 2) {{
        // ── Row 1: DFT magnitude at LINEAR-spaced frequency bin px ──────────
        // 128 bins evenly spaced from 0 to FFT_SR/2 = 4096 Hz (32 Hz per bin).
        // Image draws this with a Catmull-Rom spline between adjacent bins,
        // so even though there are only 128 sample points the rendered curve
        // is smooth across all screen pixels.
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
        // ── Per-bin temporal IIR smoothing ───────────────────────────────
        // Apply a one-pole low-pass to each bin's magnitude over time:
        //   smoothed += (raw - smoothed) * alpha
        // Equivalent to mix(smoothed, raw, alpha). Asymmetric attack/release:
        //   attack alpha = 0.5  → bin can rise ~70% in 1 frame (snappy peaks)
        //   release alpha = 0.05 → bin decays slowly (smooth fall, no jitter)
        // This kills frame-to-frame noise that made the spectrum look "busy"
        // without killing reactivity — peaks still pop instantly when notes
        // hit but the small high-frequency jitter between hits is averaged out.
        // State persists across frames via iChannel0 self-feedback.
        float mag_raw  = 15.0 * sqrt(re*re + im*im) / float(FFT_N);
        float mag_prev = texelFetch(iChannel0, ivec2(px, 1), 0).r;
        // ── Per-bin frequency-dependent dim rate ─────────────────────────
        // Different frequencies decay at different rates per user request:
        // bass slow (held lights, like sustained sub-bass), treble fast
        // (snappy hi-hat snaps). Logarithmic-style IIR formula:
        //   smoothed += (raw - smoothed) * alpha_per_bin
        // where alpha = base + bin_norm * range. Lower bins → smaller alpha
        // (slow attack/release), higher bins → larger alpha (snappy).
        //
        // Attack alpha bumped overall (0.75 → 0.82) for slightly faster
        // light response. Release range bass→treble: 0.10..0.55.
        float bin_norm     = float(px) / float(FFT_N/2 - 1);
        float attack_a     = 0.92;                        // was 0.82 — even snappier peaks
        float release_a    = 0.20 + bin_norm * 0.55;      // bass=0.20, treble=0.75 — faster overall
        float mag_alpha    = (mag_raw > mag_prev) ? attack_a : release_a;
        float mag = mag_prev + (mag_raw - mag_prev) * mag_alpha;
        O = vec4(mag, 0.0, 0.0, 1.0);

    }} else if (py == 2) {{
        // ── Row 2: UI state + smoothed audio bands ──────────────────────────
        // px 0 = specMode toggle, px 1 = prevMouse for click-edge detection,
        // px 2-4 = time-smoothed spectrum bands (low/mid/high) for visualizers
        // that need stable, non-flickering audio-reactive coloring. Raw FFT
        // bins jitter too fast to drive visual elements directly without
        // strobing — the asymmetric IIR below smooths them with a fast
        // attack (visible beat onsets) and slow release (no flicker between
        // beats).
        if (px == 0) {{
            float prevMode  = texelFetch(iChannel0, ivec2(0, 2), 0).r;
            float prevMouse = texelFetch(iChannel0, ivec2(1, 2), 0).r;
            float currMouse = iMouse.z > 0.0 ? 1.0 : 0.0;
            bool  newClick  = (currMouse > 0.5 && prevMouse < 0.5);
            O = vec4(newClick ? 1.0 - prevMode : prevMode, 0., 0., 1.);
        }} else if (px == 1) {{
            float currMouse = iMouse.z > 0.0 ? 1.0 : 0.0;
            O = vec4(currMouse, 0., 0., 1.);
        }} else if (px == 2 || px == 3 || px == 4) {{
            // ── Smoothed audio bands (px 2=low, 3=mid, 4=high) ───────────
            // Average bins within a band, then asymmetric IIR-smooth:
            //   alpha = 0.08 when rising → ~200ms attack (gentle pop on
            //                              beats, not flickery)
            //   alpha = 0.025 when falling → ~700ms release (slow fade out
            //                                between beats — visually calm)
            // Both values deliberately slow. Faster attack would let the
            // lighting strobe at frame rate during noise-heavy passages.
            // Slow release means the lighting "holds" between beats so
            // the eye sees a smooth glow envelope, not flicker.
            //
            // Band ranges are FRACTIONS of available FFT bins (FFT_N/2)
            // rather than hard-coded integers. Previous code used 60-109
            // which only worked at FFT_N≥256; at FFT_N=128 (max-compat)
            // the high band was reading mostly out-of-bounds zeros and
            // appeared dead. Now it adapts: at 128 → bins 35-63, at 256
            // → bins 70-127, etc.
            int  _nyquist = FFT_N / 2;
            int  _bLoMin  = 2;
            int  _bLoMax  = max(_bLoMin + 1, _nyquist * 15 / 100);
            int  _bMidMin = _bLoMax;
            int  _bMidMax = max(_bMidMin + 1, _nyquist * 55 / 100);
            int  _bHiMin  = _bMidMax;
            int  _bHiMax  = _nyquist;
            float current = 0.0;
            if (px == 2) {{
                for (int b = _bLoMin; b < _bLoMax; b++) current += texelFetch(iChannel0, ivec2(b, 1), 0).r;
                current /= float(max(1, _bLoMax - _bLoMin));
            }} else if (px == 3) {{
                for (int b = _bMidMin; b < _bMidMax; b++) current += texelFetch(iChannel0, ivec2(b, 1), 0).r;
                current /= float(max(1, _bMidMax - _bMidMin));
            }} else {{  // px == 4
                for (int b = _bHiMin; b < _bHiMax; b++) current += texelFetch(iChannel0, ivec2(b, 1), 0).r;
                current /= float(max(1, _bHiMax - _bHiMin));
            }}
            float prev = texelFetch(iChannel0, ivec2(px, 2), 0).r;
            float alpha = (current > prev) ? 0.08 : 0.025;
            O = vec4(mix(prev, current, alpha), 0., 0., 1.);
        }}

    }} else if (py >= HIST_BASE && py < HIST_BASE + HIST_ROWS && px < NUM_CHANNELS) {{
        // ── Rows 3–66: per-channel oscilloscope history ─────────────────────
        if (py == HIST_BASE) {{
            float ticksPerSec = float(BPM) * 2.0 / 5.0;
            float rowTime = float(SPEED) / ticksPerSec;
            Position pos = getPosition(iTime);
            float s = 0.0;
            // Note-synth fallback for single channel `px` — see _synthWave.
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
                float env   = exp(-age * 3.5);
                int wt = waveType[tn.instrument - 1];
                s = amp * env * _synthWave(wt, f, iTime);
            }}
            // Per-channel oscilloscope history. tanh-compressed amplitude
            // and reduced alpha range — the new waveforms (saw/square/etc.)
            // hit ±1 frequently which made these history rows draw as solid
            // black streaks behind the main oscilloscope. abs(tanh(...))
            // gives a smoother dynamic and the 0.25 floor + 0.55 range
            // keeps the rows visible without dominating the composition.
            O = vec4(0., 0., 0., 0.25 + abs(tanh(s * 1.5)) * 0.55);
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
        // Note-synth fallback (no VQ access in this tab) — see _synthWave.
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
            float env   = exp(-age * 3.5);
            int wt = waveType[tn.instrument - 1];
            s += amp * env * _synthWave(wt, f, t);
        }}
        s /= float(NUM_CHANNELS);
        // Raw bipolar waveform row. tanh soft-clip prevents the alpha from
        // pinning to 0 or 1 when multi-channel saw/square stacks would have
        // pushed clamp(s*1.5+0.5) to its limits. Bias 0.5 = mid-gray when
        // silent; tanh range maps ±1.5 amplitude into ~±0.4 around mid.
        O = vec4(0.0, 0.0, 0.0, 0.5 + tanh(s * 1.5) * 0.4);

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
    'IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiIKTU9EIOKGkiBTaGFkZXJUb3kgQ29tbW9uIHRhYiBl'
    'bmNvZGVyIHdpdGg6CiAgLSBQYXR0ZXJuIGNydW5jaDogYml0bWFwICsgZGljdGlvbmFyeSArIG5p'
    'YmJsZS1wYWNrZWQgcm93IHNlZWsKICAtIFNhbXBsZSBjcnVuY2g6IDMtYml0IGxpbmVhciBwYWNr'
    'ZWQgKHVuaWZvcm0gbm9pc2UgZmxvb3Ig4oCUIHN0YWJsZSBhY3Jvc3MgYWxsIHBsYXliYWNrIHBp'
    'dGNoZXMpClRhcmdldDog4omkIDY0IEtCIHRvdGFsIHByaXZhdGUgY29uc3QgZGF0YSAoTWFjIEFO'
    'R0xFL01ldGFsIHNhZmUgem9uZSkKIiIiCmltcG9ydCBzdHJ1Y3QsIHN5cywgb3MKCmNsYXNzIE1P'
    'REZpbGU6CiAgICBkZWYgX19pbml0X18oc2VsZiwgcGF0aCk6CiAgICAgICAgd2l0aCBvcGVuKHBh'
    'dGgsICdyYicpIGFzIGY6CiAgICAgICAgICAgIHNlbGYuZGF0YSA9IGYucmVhZCgpCiAgICAgICAg'
    'c2VsZi5wYXJzZSgpCgogICAgZGVmIHBhcnNlKHNlbGYpOgogICAgICAgIGQgPSBzZWxmLmRhdGEK'
    'ICAgICAgICBzZWxmLnRpdGxlID0gZFswOjIwXS5yc3RyaXAoYidceDAwJykuZGVjb2RlKCdsYXRp'
    'bjEnLCAncmVwbGFjZScpCiAgICAgICAgc2VsZi5zYW1wbGVzX2luZm8gPSBbXQogICAgICAgIGZv'
    'ciBpIGluIHJhbmdlKDMxKToKICAgICAgICAgICAgYmFzZSA9IDIwICsgaSozMAogICAgICAgICAg'
    'ICBuYW1lID0gZFtiYXNlOmJhc2UrMjJdLnJzdHJpcChiJ1x4MDAnKS5kZWNvZGUoJ2xhdGluMScs'
    'ICdyZXBsYWNlJykKICAgICAgICAgICAgbGVuZ3RoX3cgICAgID0gc3RydWN0LnVucGFjaygnPkgn'
    'LCBkW2Jhc2UrMjI6YmFzZSsyNF0pWzBdCiAgICAgICAgICAgIGZpbmV0dW5lICAgICA9IGRbYmFz'
    'ZSsyNF0gJiAweDBGCiAgICAgICAgICAgIHZvbHVtZSAgICAgICA9IGRbYmFzZSsyNV0KICAgICAg'
    'ICAgICAgbG9vcF9zdGFydF93ID0gc3RydWN0LnVucGFjaygnPkgnLCBkW2Jhc2UrMjY6YmFzZSsy'
    'OF0pWzBdCiAgICAgICAgICAgIGxvb3BfbGVuX3cgICA9IHN0cnVjdC51bnBhY2soJz5IJywgZFti'
    'YXNlKzI4OmJhc2UrMzBdKVswXQogICAgICAgICAgICBzZWxmLnNhbXBsZXNfaW5mby5hcHBlbmQo'
    'ZGljdCgKICAgICAgICAgICAgICAgIG5hbWU9bmFtZSwgbGVuZ3RoPWxlbmd0aF93KjIsIGZpbmV0'
    'dW5lPWZpbmV0dW5lLAogICAgICAgICAgICAgICAgdm9sdW1lPXZvbHVtZSwgbG9vcF9zdGFydD1s'
    'b29wX3N0YXJ0X3cqMiwgbG9vcF9sZW49bG9vcF9sZW5fdyoyKSkKICAgICAgICBzZWxmLnNvbmdf'
    'bGVuZ3RoID0gZFs5NTBdCiAgICAgICAgc2VsZi5wYXR0ZXJuX29yZGVyID0gbGlzdChkWzk1Mjo5'
    'NTIrMTI4XSkKICAgICAgICBzZWxmLm1hZ2ljID0gZFsxMDgwOjEwODRdCiAgICAgICAgIyBEZXRl'
    'Y3QgY2hhbm5lbCBjb3VudCBmcm9tIHNpZ25hdHVyZQogICAgICAgIHNpZyA9IHNlbGYubWFnaWMK'
    'ICAgICAgICBpZiBzaWcgaW4gKGInTS5LLicsIGInTSFLIScsIGInTSZLIScsIGInTi5ULicsIGIn'
    'RkxUNCcsIGInNENITicpOgogICAgICAgICAgICBzZWxmLm51bV9jaGFubmVscyA9IDQKICAgICAg'
    'ICBlbGlmIHNpZyA9PSBiJ0ZMVDgnIG9yIHNpZyBpbiAoYidPQ1RBJywgYidDRDgxJywgYidPS1RB'
    'Jyk6CiAgICAgICAgICAgIHNlbGYubnVtX2NoYW5uZWxzID0gOAogICAgICAgIGVsaWYgbGVuKHNp'
    'ZykgPT0gNCBhbmQgc2lnWzE6NF0gPT0gYidDSE4nIGFuZCBzaWdbMDoxXS5pc2RpZ2l0KCk6CiAg'
    'ICAgICAgICAgIHNlbGYubnVtX2NoYW5uZWxzID0gaW50KHNpZ1swOjFdKQogICAgICAgIGVsaWYg'
    'bGVuKHNpZykgPT0gNCBhbmQgc2lnWzI6NF0gPT0gYidDSCcgYW5kIHNpZ1swOjFdLmlzZGlnaXQo'
    'KSBhbmQgc2lnWzE6Ml0uaXNkaWdpdCgpOgogICAgICAgICAgICBzZWxmLm51bV9jaGFubmVscyA9'
    'IGludChzaWdbMDoyXSkKICAgICAgICBlbGlmIGxlbihzaWcpID09IDQgYW5kIHNpZ1s6M10gPT0g'
    'YidURFonIGFuZCBzaWdbMzo0XS5pc2RpZ2l0KCk6CiAgICAgICAgICAgIHNlbGYubnVtX2NoYW5u'
    'ZWxzID0gaW50KHNpZ1szOjRdKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHNlbGYubnVtX2No'
    'YW5uZWxzID0gNAogICAgICAgIHNlbGYubnVtX3BhdHRlcm5zID0gbWF4KHNlbGYucGF0dGVybl9v'
    'cmRlcls6c2VsZi5zb25nX2xlbmd0aF0pICsgMQogICAgICAgICMgRWFjaCBwYXR0ZXJuIHJvdyA9'
    'IG51bV9jaGFubmVscyDDlyA0IGJ5dGVzOyA2NCByb3dzL3BhdHRlcm4KICAgICAgICBwYXRfc2l6'
    'ZSA9IDY0ICogc2VsZi5udW1fY2hhbm5lbHMgKiA0CiAgICAgICAgc2VsZi5wYXR0ZXJucyA9IFtd'
    'CiAgICAgICAgb2ZmID0gMTA4NAogICAgICAgIGZvciBwIGluIHJhbmdlKHNlbGYubnVtX3BhdHRl'
    'cm5zKToKICAgICAgICAgICAgc2VsZi5wYXR0ZXJucy5hcHBlbmQoZFtvZmY6b2ZmK3BhdF9zaXpl'
    'XSkKICAgICAgICAgICAgb2ZmICs9IHBhdF9zaXplCiAgICAgICAgIyBTYW1wbGVzIChyYXcgc2ln'
    'bmVkIDgtYml0IGJ5dGVzKQogICAgICAgIHNlbGYuc2FtcGxlX2J5dGVzID0gW10KICAgICAgICBm'
    'b3IgcyBpbiBzZWxmLnNhbXBsZXNfaW5mbzoKICAgICAgICAgICAgc2VsZi5zYW1wbGVfYnl0ZXMu'
    'YXBwZW5kKGRbb2ZmOm9mZitzWydsZW5ndGgnXV0pCiAgICAgICAgICAgIG9mZiArPSBzWydsZW5n'
    'dGgnXQoKIyDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZAKIyBQ'
    'QVRURVJOIENSVU5DSDogYml0bWFwICsgZGljdCArIG5pYmJsZS1zZWVrCiMg4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQCgpFTVBUWV9OT1RFID0gYidceDAwXHgw'
    'MFx4MDBceDAwJwoKZGVmIGVuY29kZV9wYXR0ZXJucyhtb2QpOgogICAgIiIiUmV0dXJucyBkaWN0'
    'IG9mIGFsbCBwYXR0ZXJuIGRhdGEgc3RydWN0dXJlcy4iIiIKICAgICMgQnVpbGQgZmxhdCBsaXN0'
    'IG9mIDQtYnl0ZSBub3RlcyBpbiBvcmRlcjogcGF0IDAuLk4tMSwgcm93IDAuLjYzLCBjaCAwLi4z'
    'LgogICAgIyBGaXJzdCBhcHBseSBQcm9UcmFja2VyIHBhcmFtLW1lbW9yeSByZXdyaXRpbmcgaW4g'
    'c29uZy1wb3NpdGlvbiBwbGF5YmFjawogICAgIyBvcmRlcjogd2hlbiBhIG5vdGUgaGFzIGVmZmVj'
    'dCAxLzIvMy80LzUvNi9BIGFuZCBwYXJhbT09MCwgc3Vic3RpdHV0ZSB0aGUKICAgICMgbGFzdCBu'
    'b24temVybyBwYXJhbSBzZWVuIGZvciB0aGF0IGVmZmVjdCBvbiB0aGlzIGNoYW5uZWwuICBUaGlz'
    'IG1ha2VzCiAgICAjIHRvbmUtcG9ydGEgcnVucyBsaWtlICIzMDAgMzAwIDMwMCIgY29udGludWUg'
    'd2l0aCB0aGUgcHJldmlvdXMgc2xpZGUgcmF0ZQogICAgIyDigJQgcmVxdWlyZWQgZm9yIG1hbnkg'
    'TU9EcyAoaW5jbC4gR1NMSU5HRVIgcGF0dGVybiAzKS4KICAgIE5DID0gbW9kLm51bV9jaGFubmVs'
    'cwogICAgcm93X3N0cmlkZSA9IE5DICogNAoKICAgICMgV2FsayBzb25nIHBvc2l0aW9ucyB0byBm'
    'aW5kIHBhcmFtLW1lbW9yeSBjaGFpbnMgcGVyIGNoYW5uZWwuCiAgICAjIEVmZmVjdCBncm91cHMg'
    'dGhhdCBzaGFyZSBtZW1vcnk6CiAgICAjICAgMHgxIChwb3J0YSB1cCksIDB4MiAocG9ydGEgZG93'
    'biksIDB4MyAodG9uZSBwb3J0YSksIDB4NSAodG9uZSt2b2wpLAogICAgIyAgIDB4NCAodmlicmF0'
    'byksIDB4NiAodmliK3ZvbCksIDB4QSAodm9sIHNsaWRlKQogICAgIyBXZSByZXdyaXRlIHRoZSBp'
    'bi1tZW1vcnkgcGF0dGVybiBieXRlcyAoYSBjb3B5KSBzbyBlbmNvZGluZyBzZWVzIHRoZQogICAg'
    'IyBjb3JyZWN0ZWQgcGFyYW1zLiAgQnVpbGQgYSBmcmVzaCBwZXItcGF0dGVybiBub3RlIGxpc3Qg'
    'd2l0aCByZXdyaXRlcy4KICAgICMgVXNlIG1vZC5wYXR0ZXJuX29yZGVyIChlbmNvZGVyIE1PREZp'
    'bGUgZXF1aXZhbGVudCBvZiBzb25nX3Bvc2l0aW9ucykuCiAgICBfc29uZ19vcmRlciA9IGdldGF0'
    'dHIobW9kLCAncGF0dGVybl9vcmRlcicsIE5vbmUpIG9yIGdldGF0dHIobW9kLCAnc29uZ19wb3Np'
    'dGlvbnMnLCBbXSkKICAgIHBhdF9jb3BpZXMgPSB7fQogICAgbGFzdF9wYXJhbSA9IFt7fSBmb3Ig'
    'XyBpbiByYW5nZShOQyldICAjIGxhc3RfcGFyYW1bY2hdW2VmZmVjdF0gPSBsYXN0IG5vbnplcm8g'
    'cGFyYW0KICAgIHJld3JpdHRlbl9ub3Rlc19jb3VudCA9IDAKICAgIGZvciBzcCBpbiBfc29uZ19v'
    'cmRlcls6Z2V0YXR0cihtb2QsICdzb25nX2xlbmd0aCcsIGxlbihfc29uZ19vcmRlcikpXToKICAg'
    'ICAgICBpZiBzcCBub3QgaW4gcGF0X2NvcGllczoKICAgICAgICAgICAgcGF0X2NvcGllc1tzcF0g'
    'PSBieXRlYXJyYXkobW9kLnBhdHRlcm5zW3NwXSkKICAgICAgICAjIE5vdGU6IGEgcGF0dGVybiBt'
    'YXkgYXBwZWFyIGF0IG11bHRpcGxlIHNvbmcgcG9zaXRpb25zOyB3ZSBhcHBseQogICAgICAgICMg'
    'cmV3cml0ZXMgaW4gcGxheWJhY2sgb3JkZXIgc28gbWVtb3J5IHN0YXRlIHByb3BhZ2F0ZXMgYWNy'
    'b3NzIHRoZW0uCiAgICAgICAgIyBSZXdyaXRpbmcgYSBwYXR0ZXJuIHRoYXQgaXMgcmV1c2VkIGxh'
    'dGVyIG1lYW5zIHRoZSBzZWNvbmQgdmlzaXQKICAgICAgICAjIHVzZXMgdGhlIGFscmVhZHktcmV3'
    'cml0dGVuIHBhcmFtcywgd2hpY2ggaXMgYWNjZXB0YWJsZSBzaW5jZSB0aGUKICAgICAgICAjIGxh'
    'c3RfcGFyYW0gc3RhdGUgYXQgc2Vjb25kIHZpc2l0IHdvdWxkIG5hdHVyYWxseSBhbHNvIGhhdmUg'
    'dGhvc2UuCiAgICAjIFJlc2V0IGZvciBhY3R1YWwgcmV3cml0ZSB3YWxrCiAgICBsYXN0X3BhcmFt'
    'ID0gW3t9IGZvciBfIGluIHJhbmdlKE5DKV0KICAgIHZpc2l0ZWRfa2V5cyA9IHNldCgpCiAgICBm'
    'b3Igc3AgaW4gX3Nvbmdfb3JkZXJbOmdldGF0dHIobW9kLCAnc29uZ19sZW5ndGgnLCBsZW4oX3Nv'
    'bmdfb3JkZXIpKV06CiAgICAgICAgcGF0X2NvcHkgPSBwYXRfY29waWVzW3NwXQogICAgICAgIGZv'
    'ciByb3cgaW4gcmFuZ2UoNjQpOgogICAgICAgICAgICBmb3IgY2ggaW4gcmFuZ2UoTkMpOgogICAg'
    'ICAgICAgICAgICAgYmFzZSA9IHJvdypyb3dfc3RyaWRlICsgY2gqNAogICAgICAgICAgICAgICAg'
    'a2V5ID0gKHNwLCByb3csIGNoKQogICAgICAgICAgICAgICAgaWYga2V5IGluIHZpc2l0ZWRfa2V5'
    'czogY29udGludWUgICMgYXZvaWQgZG91YmxlLXJld3JpdGUgb24gcGF0dGVybiByZXVzZQogICAg'
    'ICAgICAgICAgICAgdmlzaXRlZF9rZXlzLmFkZChrZXkpCiAgICAgICAgICAgICAgICAjIERlY29k'
    'ZSBub3RlOiBieXRlcyBbcGVyaW9kX2hpLCBwZXJpb2RfbG8sIHNhbXBsZV9sb3xlZmZlY3QsIHBh'
    'cmFtXQogICAgICAgICAgICAgICAgIyBNT0QgbGF5b3V0OiBieXRlMCA9IHNhbXBsZV9oaSg0KSB8'
    'IHBlcmlvZF9oaSg0KQogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICBieXRlMSA9IHBlcmlv'
    'ZF9sbyg4KQogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICBieXRlMiA9IHNhbXBsZV9sbyg0'
    'KSB8IGVmZmVjdCg0KQogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICBieXRlMyA9IHBhcmFt'
    'CiAgICAgICAgICAgICAgICBiMCwgYjEsIGIyLCBiMyA9IHBhdF9jb3B5W2Jhc2VdLCBwYXRfY29w'
    'eVtiYXNlKzFdLCBwYXRfY29weVtiYXNlKzJdLCBwYXRfY29weVtiYXNlKzNdCiAgICAgICAgICAg'
    'ICAgICBlZmZlY3QgPSBiMiAmIDB4MEYKICAgICAgICAgICAgICAgIHBhcmFtICA9IGIzCiAgICAg'
    'ICAgICAgICAgICAjIFByb1RyYWNrZXIgcGFyYW0tbWVtb3J5IHJ1bGVzOgogICAgICAgICAgICAg'
    'ICAgIwogICAgICAgICAgICAgICAgIyAgIDF4eCAocG9ydGEgdXApICAgICAgICDigJQgcGFyYW09'
    'MCDihpIgdXNlIGxhc3QgMXh4CiAgICAgICAgICAgICAgICAjICAgMnh4IChwb3J0YSBkb3duKSAg'
    'ICAgIOKAlCBwYXJhbT0wIOKGkiB1c2UgbGFzdCAyeHgKICAgICAgICAgICAgICAgICMgICAzeHgg'
    'KHRvbmUgcG9ydGEpICAgICAg4oCUIHBhcmFtPTAg4oaSIHVzZSBsYXN0IDN4eAogICAgICAgICAg'
    'ICAgICAgIwogICAgICAgICAgICAgICAgIyAgIDR4eCAodmlicmF0bykgICAgICAgICDigJQgTklC'
    'QkxFIG1lbW9yeTogaGlnaCBuaWI9MCBrZWVwcwogICAgICAgICAgICAgICAgIyAgICAgICAgICAg'
    'ICAgICAgICAgICAgICAgICBwcmlvciBzcGVlZCwgbG93IG5pYj0wIGtlZXBzCiAgICAgICAgICAg'
    'ICAgICAjICAgICAgICAgICAgICAgICAgICAgICAgICAgIHByaW9yIGRlcHRoLiAgV2UgZG9uJ3Qg'
    'cmV3cml0ZQogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICAgICAgICAgICAgICAgICBoZXJl'
    'IOKAlCBHTFNML0hUTUwgZG8gbmliYmxlLWxldmVsCiAgICAgICAgICAgICAgICAjICAgICAgICAg'
    'ICAgICAgICAgICAgICAgICAgIGhhbmRsaW5nLiAgT25seSByZXdyaXRlIHBhcmFtPTAKICAgICAg'
    'ICAgICAgICAgICMgICAgICAgICAgICAgICAgICAgICAgICAgICAgKHdob2xlIGJ5dGUgemVybykg'
    '4oaSIHVzZSBsYXN0IDR4eC4KICAgICAgICAgICAgICAgICMgICA3eHggKHRyZW1vbG8pICAgICAg'
    'ICAg4oCUIE5JQkJMRSBtZW1vcnkgbGlrZSA0eHguCiAgICAgICAgICAgICAgICAjCiAgICAgICAg'
    'ICAgICAgICAjICAgNXh4IChjb250aW51ZSB0b25lIHBvcnRhICsgdm9sIHNsaWRlKSDigJQgcGFy'
    'YW0gYnl0ZSBpcwogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICAgICAgICAgICAgICAgICBW'
    'T0wtU0xJREUgT05MWS4gIDUwMCA9IGNvbnRpbnVlCiAgICAgICAgICAgICAgICAjICAgICAgICAg'
    'ICAgICAgICAgICAgICAgICAgIHNsaWRlIHdpdGggTk8gdm9sIGNoYW5nZTsgdmFsaWQKICAgICAg'
    'ICAgICAgICAgICMgICAgICAgICAgICAgICAgICAgICAgICAgICAgY29tbWFuZCwgZG8gTk9UIHJl'
    'd3JpdGUuCiAgICAgICAgICAgICAgICAjICAgNnh4IChjb250aW51ZSB2aWJyYXRvICsgdm9sIHNs'
    'aWRlKSDigJQgc2FtZSBhcyA1eHg7CiAgICAgICAgICAgICAgICAjICAgICAgICAgICAgICAgICAg'
    'ICAgICAgICAgIHBhcmFtIGJ5dGUgaXMgdm9sLXNsaWRlIG9ubHkuCiAgICAgICAgICAgICAgICAj'
    'ICAgICAgICAgICAgICAgICAgICAgICAgICAgIDYwMCA9IGNvbnRpbnVlIHZpYnJhdG8sIG5vIHZv'
    'bAogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICAgICAgICAgICAgICAgICBzbGlkZTsgdmFs'
    'aWQgY29tbWFuZC4KICAgICAgICAgICAgICAgICMgICBBeHggKHZvbCBzbGlkZSkgICAgICAg4oCU'
    'IEEwMCA9IG5vLW9wIGluIFBUIChOT1QgbWVtb3J5KS4KICAgICAgICAgICAgICAgICMgICAgICAg'
    'ICAgICAgICAgICAgICAgICAgICAgRG8gTk9UIHJld3JpdGUuCiAgICAgICAgICAgICAgICBpZiBl'
    'ZmZlY3QgaW4gKDB4MSwgMHgyLCAweDMpOgogICAgICAgICAgICAgICAgICAgIGlmIHBhcmFtID09'
    'IDAgYW5kIGVmZmVjdCBpbiBsYXN0X3BhcmFtW2NoXToKICAgICAgICAgICAgICAgICAgICAgICAg'
    'bmV3X3BhcmFtID0gbGFzdF9wYXJhbVtjaF1bZWZmZWN0XQogICAgICAgICAgICAgICAgICAgICAg'
    'ICBwYXRfY29weVtiYXNlKzNdID0gbmV3X3BhcmFtCiAgICAgICAgICAgICAgICAgICAgICAgIHJl'
    'd3JpdHRlbl9ub3Rlc19jb3VudCArPSAxCiAgICAgICAgICAgICAgICAgICAgZWxpZiBwYXJhbSAh'
    'PSAwOgogICAgICAgICAgICAgICAgICAgICAgICBsYXN0X3BhcmFtW2NoXVtlZmZlY3RdID0gcGFy'
    'YW0KICAgICAgICAgICAgICAgIGVsaWYgZWZmZWN0IGluICgweDQsIDB4Nyk6CiAgICAgICAgICAg'
    'ICAgICAgICAgIyBXaG9sZS1ieXRlPTAg4oaSIHVzZSBsYXN0IHdob2xlLWJ5dGUgbWVtb3J5Lgog'
    'ICAgICAgICAgICAgICAgICAgICMgTm9uLXplcm86IGFsc28gc3RvcmUgYXMgbGFzdC1ieXRlIG1l'
    'bW9yeSAobmliYmxlLWxldmVsCiAgICAgICAgICAgICAgICAgICAgIyBoYW5kbGluZyBpcyBkb25l'
    'IGJ5IEhUTUwvR0xTTCBkdXJpbmcgcGxheWJhY2spLgogICAgICAgICAgICAgICAgICAgIGlmIHBh'
    'cmFtID09IDAgYW5kIGVmZmVjdCBpbiBsYXN0X3BhcmFtW2NoXToKICAgICAgICAgICAgICAgICAg'
    'ICAgICAgbmV3X3BhcmFtID0gbGFzdF9wYXJhbVtjaF1bZWZmZWN0XQogICAgICAgICAgICAgICAg'
    'ICAgICAgICBwYXRfY29weVtiYXNlKzNdID0gbmV3X3BhcmFtCiAgICAgICAgICAgICAgICAgICAg'
    'ICAgIHJld3JpdHRlbl9ub3Rlc19jb3VudCArPSAxCiAgICAgICAgICAgICAgICAgICAgZWxpZiBw'
    'YXJhbSAhPSAwOgogICAgICAgICAgICAgICAgICAgICAgICBsYXN0X3BhcmFtW2NoXVtlZmZlY3Rd'
    'ID0gcGFyYW0KICAgICAgICAgICAgICAgICMgNS82L0E6IG5vIHBhcmFtLW1lbW9yeSByZXdyaXRp'
    'bmcgKHRoZWlyIHBhcmFtPTAgaXMgbWVhbmluZ2Z1bCkKCiAgICBub3RlcyA9IFtdCiAgICBmb3Ig'
    'cGF0IGluIHJhbmdlKG1vZC5udW1fcGF0dGVybnMpOgogICAgICAgIGlmIHBhdCBpbiBwYXRfY29w'
    'aWVzOgogICAgICAgICAgICBwZGF0YSA9IHBhdF9jb3BpZXNbcGF0XQogICAgICAgIGVsc2U6CiAg'
    'ICAgICAgICAgIHBkYXRhID0gbW9kLnBhdHRlcm5zW3BhdF0KICAgICAgICBmb3Igcm93IGluIHJh'
    'bmdlKDY0KToKICAgICAgICAgICAgZm9yIGNoIGluIHJhbmdlKE5DKToKICAgICAgICAgICAgICAg'
    'IGJhc2UgPSByb3cqcm93X3N0cmlkZSArIGNoKjQKICAgICAgICAgICAgICAgIG5vdGVzLmFwcGVu'
    'ZChieXRlcyhwZGF0YVtiYXNlOmJhc2UrNF0pKQogICAgaWYgcmV3cml0dGVuX25vdGVzX2NvdW50'
    'ID4gMDoKICAgICAgICBwcmludChmIiAgIOKame+4jyAgUGFyYW0tbWVtb3J5OiB7cmV3cml0dGVu'
    'X25vdGVzX2NvdW50fSBwYXJhbT0wIGVmZmVjdHMgcmV3cml0dGVuIHdpdGggcHJldmlvdXMgdmFs'
    'dWVzIikKICAgIHRvdGFsX25vdGVzID0gbGVuKG5vdGVzKQogICAgbnVtX3Jvd3MgICAgPSBtb2Qu'
    'bnVtX3BhdHRlcm5zICogNjQKCiAgICAjIFVuaXF1ZSBub24tZW1wdHkgbm90ZXMg4oaSIGRpY3Rp'
    'b25hcnkKICAgIHVuaXEgPSBzb3J0ZWQoc2V0KG4gZm9yIG4gaW4gbm90ZXMgaWYgbiAhPSBFTVBU'
    'WV9OT1RFKSkKICAgIGlkeF9ieXRlcyA9IDEgaWYgbGVuKHVuaXEpIDw9IDI1NiBlbHNlIDIKICAg'
    'IGFzc2VydCBsZW4odW5pcSkgPD0gNjU1MzYsIGYidG9vIG1hbnkgdW5pcXVlIG5vdGVzOiB7bGVu'
    'KHVuaXEpfSIKICAgIG5vdGVfdG9faWR4ID0ge246aSBmb3IgaSxuIGluIGVudW1lcmF0ZSh1bmlx'
    'KX0KCiAgICAjIEJpdG1hcCAoMSBiaXQgcGVyIG5vdGUsIExTQi1maXJzdCB3aXRoaW4gZWFjaCBi'
    'eXRlKQogICAgYml0bWFwID0gYnl0ZWFycmF5KCh0b3RhbF9ub3RlcyArIDcpIC8vIDgpCiAgICBm'
    'b3IgaSwgbiBpbiBlbnVtZXJhdGUobm90ZXMpOgogICAgICAgIGlmIG4gIT0gRU1QVFlfTk9URToK'
    'ICAgICAgICAgICAgYml0bWFwW2kgPj4gM10gfD0gMSA8PCAoaSAmIDcpCgogICAgIyBJbmRleCBz'
    'dHJlYW0gKDEgb3IgMiBieXRlcyBwZXIgbm9uLWVtcHR5IG5vdGUsIGxpdHRsZS1lbmRpYW4gaWYg'
    'MkIpCiAgICBpZHhfc3RyZWFtID0gYnl0ZWFycmF5KCkKICAgIGZvciBuIGluIG5vdGVzOgogICAg'
    'ICAgIGlmIG4gIT0gRU1QVFlfTk9URToKICAgICAgICAgICAgaSA9IG5vdGVfdG9faWR4W25dCiAg'
    'ICAgICAgICAgIGlmIGlkeF9ieXRlcyA9PSAxOgogICAgICAgICAgICAgICAgaWR4X3N0cmVhbS5h'
    'cHBlbmQoaSkKICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgIGlkeF9zdHJlYW0uYXBw'
    'ZW5kKGkgJiAweEZGKQogICAgICAgICAgICAgICAgaWR4X3N0cmVhbS5hcHBlbmQoKGkgPj4gOCkg'
    'JiAweEZGKQoKICAgICMgUGVyLXJvdyBjb3VudDogY291bnQgb2Ygbm9uLWVtcHR5IG5vdGVzIElO'
    'IHRoaXMgcm93ICgwLi40KS4KICAgIHBlcl9yb3dfY291bnQgPSBbXQogICAgZm9yIHJvdyBpbiBy'
    'YW5nZShudW1fcm93cyk6CiAgICAgICAgY291bnQgPSBzdW0oMSBmb3IgY2ggaW4gcmFuZ2UoTkMp'
    'IGlmIG5vdGVzW3JvdypOQyArIGNoXSAhPSBFTVBUWV9OT1RFKQogICAgICAgIHBlcl9yb3dfY291'
    'bnQuYXBwZW5kKGNvdW50KQoKICAgICMgUHJlZml4IHN1bTogcHJlZml4W3Jvd10gPSBub24tZW1w'
    'dHkgY291bnQgaW4gcm93cyBbMCwgcm93KSA9IHJhbmsgYXQgc3RhcnQgb2Ygcm93LgogICAgIyBT'
    'dG9yZWQgYXMgMTYtYml0IExFIHdvcmRzIHNvIGRlY29kZXIgaXMgTygxKS4KICAgICMgUmFuZ2U6'
    'IDAgdG8gfnRvdGFsX25vbl9lbXB0eSAo4omkIDU4ODggZm9yIDIzLXBhdCBNT0QpIOKGkiBmaXRz'
    'IGVhc2lseSBpbiAxNiBiaXRzLgogICAgcHJlZml4ID0gWzBdICogbnVtX3Jvd3MKICAgIHJ1bm5p'
    'bmcgPSAwCiAgICBmb3Igcm93IGluIHJhbmdlKG51bV9yb3dzKToKICAgICAgICBwcmVmaXhbcm93'
    'XSA9IHJ1bm5pbmcKICAgICAgICBydW5uaW5nICs9IHBlcl9yb3dfY291bnRbcm93XQoKICAgIHJv'
    'd19zZWVrX2J5dGVzID0gYnl0ZWFycmF5KCkKICAgIGZvciB2IGluIHByZWZpeDoKICAgICAgICBh'
    'c3NlcnQgMCA8PSB2IDwgNjU1MzYsIGYicHJlZml4IHt2fSBvdmVyZmxvd3MgMTYgYml0cyIKICAg'
    'ICAgICByb3dfc2Vla19ieXRlcy5hcHBlbmQodiAmIDB4RkYpCiAgICAgICAgcm93X3NlZWtfYnl0'
    'ZXMuYXBwZW5kKCh2ID4+IDgpICYgMHhGRikKCiAgICByZXR1cm4gZGljdCgKICAgICAgICB0b3Rh'
    'bF9ub3Rlcz10b3RhbF9ub3RlcywgbnVtX3Jvd3M9bnVtX3Jvd3MsCiAgICAgICAgdW5pcT11bmlx'
    'LCBub3RlX3RvX2lkeD1ub3RlX3RvX2lkeCwgaWR4X2J5dGVzPWlkeF9ieXRlcywKICAgICAgICBi'
    'aXRtYXA9Yml0bWFwLCBpZHhfc3RyZWFtPWlkeF9zdHJlYW0sCiAgICAgICAgcm93X3NlZWtfYnl0'
    'ZXM9cm93X3NlZWtfYnl0ZXMsIHByZWZpeD1wcmVmaXgsCiAgICApCgojIOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkAojIDMtQklUIExJTkVBUiBTQU1QTEUgQ1JV'
    'TkNICiMg4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQCgpkZWYg'
    'ZW5jb2RlX3NhbXBsZXNfcGFja2VkKG1vZCwgYml0cz0zKToKICAgICIiIkNvbmNhdGVuYXRlIGFs'
    'bCBzYW1wbGVzLCBlbmNvZGUgZWFjaCB0byBgYml0c2AgYml0cyAocm91bmRlZCkuCiAgICBTdXBw'
    'b3J0cyAzLWJpdCBhbmQgNC1iaXQgbGluZWFyIHF1YW50aXphdGlvbi4KICAgICAgMy1iaXQ6IGNv'
    'ZGUgMC4uNywgbGV2ZWxzIChjb2RlKjMyIC0gMTEyKSwgc3RlcCAzMi8yNTYgPSAxMi41JQogICAg'
    'ICA0LWJpdDogY29kZSAwLi4xNSwgbGV2ZWxzIChjb2RlKjE2IC0gMTIwKSwgc3RlcCAxNi8yNTYg'
    'PSA2LjI1JSAoKzYgZEIgU05SKQogICAgUmV0dXJucyBwYWNrZWQgYnl0ZXMgKyBwZXItc2FtcGxl'
    'IHN0YXJ0IGluZGljZXMgKGxvZ2ljYWwgc2FtcGxlIHVuaXRzKS4iIiIKICAgIGlmIGJpdHMgbm90'
    'IGluICgzLCA0KToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYiYml0cyBtdXN0IGJlIDMgb3Ig'
    'NCwgZ290IHtiaXRzfSIpCgogICAgY29uY2F0X3NpZ25lZCA9IFtdCiAgICBzdGFydHMgPSBbXQog'
    'ICAgZm9yIHMgaW4gbW9kLnNhbXBsZV9ieXRlczoKICAgICAgICBzdGFydHMuYXBwZW5kKGxlbihj'
    'b25jYXRfc2lnbmVkKSkKICAgICAgICBmb3IgYiBpbiBzOgogICAgICAgICAgICBjb25jYXRfc2ln'
    'bmVkLmFwcGVuZChiIC0gMjU2IGlmIGIgPj0gMTI4IGVsc2UgYikKICAgICAgICBjb25jYXRfc2ln'
    'bmVkLmV4dGVuZChbMF0gKiAxNikKCiAgICB0b3RhbF9zYW1wbGVzID0gbGVuKGNvbmNhdF9zaWdu'
    'ZWQpCiAgICBjb2RlcyA9IGJ5dGVhcnJheSgpCiAgICBtYXhfY29kZSA9ICgxIDw8IGJpdHMpIC0g'
    'MQogICAgc2hpZnQgPSA4IC0gYml0cwogICAgZm9yIHN2IGluIGNvbmNhdF9zaWduZWQ6CiAgICAg'
    'ICAgdW5zaWduZWRfb2Zmc2V0ID0gc3YgKyAxMjggICMgWzAsIDI1NV0KICAgICAgICBjb2RlID0g'
    'dW5zaWduZWRfb2Zmc2V0ID4+IHNoaWZ0CiAgICAgICAgaWYgY29kZSA+IG1heF9jb2RlOiBjb2Rl'
    'ID0gbWF4X2NvZGUKICAgICAgICBjb2Rlcy5hcHBlbmQoY29kZSkKCiAgICB0b3RhbF9iaXRzID0g'
    'dG90YWxfc2FtcGxlcyAqIGJpdHMKICAgIHRvdGFsX2J5dGVzID0gKHRvdGFsX2JpdHMgKyA3KSAv'
    'LyA4CiAgICBwYWNrZWQgPSBieXRlYXJyYXkodG90YWxfYnl0ZXMpCgogICAgaWYgYml0cyA9PSA0'
    'OgogICAgICAgICMgTmliYmxlIHBhY2tpbmc6IDIgY29kZXMgcGVyIGJ5dGUsIGxvdyBuaWJibGUg'
    'Zmlyc3QKICAgICAgICBmb3IgaSwgYyBpbiBlbnVtZXJhdGUoY29kZXMpOgogICAgICAgICAgICBi'
    'eXRlX3BvcyA9IGkgPj4gMQogICAgICAgICAgICBpZiBpICYgMToKICAgICAgICAgICAgICAgIHBh'
    'Y2tlZFtieXRlX3Bvc10gfD0gKGMgJiAweEYpIDw8IDQKICAgICAgICAgICAgZWxzZToKICAgICAg'
    'ICAgICAgICAgIHBhY2tlZFtieXRlX3Bvc10gfD0gYyAmIDB4RgogICAgZWxzZTogICMgYml0cyA9'
    'PSAzCiAgICAgICAgZm9yIGksIGMgaW4gZW51bWVyYXRlKGNvZGVzKToKICAgICAgICAgICAgYml0'
    'X3BvcyAgID0gaSAqIDMKICAgICAgICAgICAgYnl0ZV9wb3MgID0gYml0X3BvcyA+PiAzCiAgICAg'
    'ICAgICAgIGJpdF9zaGlmdCA9IGJpdF9wb3MgJiA3CiAgICAgICAgICAgIHZhbCA9IChjICYgNykg'
    'PDwgYml0X3NoaWZ0CiAgICAgICAgICAgIHBhY2tlZFtieXRlX3Bvc10gfD0gdmFsICYgMHhGRgog'
    'ICAgICAgICAgICBpZiBiaXRfc2hpZnQgPiA1IGFuZCBieXRlX3BvcyArIDEgPCB0b3RhbF9ieXRl'
    'czoKICAgICAgICAgICAgICAgIHBhY2tlZFtieXRlX3BvcyArIDFdIHw9ICh2YWwgPj4gOCkgJiAw'
    'eEZGCgogICAgcmV0dXJuIHBhY2tlZCwgc3RhcnRzLCB0b3RhbF9zYW1wbGVzCgojIEJhY2t3YXJk'
    'LWNvbXBhdCBhbGlhcwpkZWYgZW5jb2RlX3NhbXBsZXNfM2JpdChtb2QpOgogICAgcmV0dXJuIGVu'
    'Y29kZV9zYW1wbGVzX3BhY2tlZChtb2QsIGJpdHM9MykKCgpkZWYgY29tcHV0ZV9yb3dfc3BlZWRf'
    'dGFibGUobW9kKToKICAgICIiIlNpbXVsYXRlIHRoZSBzb25nIHRvIGZpbmQgcGVyLXJvdyBTUEVF'
    'RCAoaG9ub3VyaW5nIEZ4eC9EeHgvQnh4IGVmZmVjdHMpLgogICAgUmV0dXJucyByb3dTcGVlZFtu'
    'dW1fc29uZ19yb3dzXSBhbmQgcm93U3RhcnRUaWNrW251bV9zb25nX3Jvd3MrMV0uCiAgICBDb3Jy'
    'ZWN0bHkgaGFuZGxlcyBEeHggKHBhdHRlcm4gYnJlYWspIGFuZCBCeHggKHBvc2l0aW9uIGp1bXAp'
    'IHdoaWNoCiAgICBzaG9ydGVuIGEgcGF0dGVybidzIGVmZmVjdGl2ZSByb3cgY291bnQuIiIiCiAg'
    'ICBzcGVlZCA9IDYgICMgUHJvVHJhY2tlciBkZWZhdWx0CiAgICBicG0gICA9IDEyNQogICAgcm93'
    'U3BlZWQgPSBbXQogICAgYnBtX2NoYW5nZXMgPSBGYWxzZQogICAgZm9yIHBvcyBpbiByYW5nZSht'
    'b2Quc29uZ19sZW5ndGgpOgogICAgICAgIHBhdF9pZHggPSBtb2QucGF0dGVybl9vcmRlcltwb3Nd'
    'CiAgICAgICAgcGRhdGEgPSBtb2QucGF0dGVybnNbcGF0X2lkeF0KICAgICAgICBicm9rZSA9IEZh'
    'bHNlCiAgICAgICAgZm9yIHJvdyBpbiByYW5nZSg2NCk6CiAgICAgICAgICAgICMgU2NhbiBhbGwg'
    'NCBjaGFubmVscyBmb3IgRnh4IC8gRHh4IC8gQnh4IG9uIHRoaXMgcm93CiAgICAgICAgICAgIGZv'
    'ciBjaCBpbiByYW5nZShtb2QubnVtX2NoYW5uZWxzKToKICAgICAgICAgICAgICAgIGJhc2UgPSBy'
    'b3cgKiBtb2QubnVtX2NoYW5uZWxzICogNCArIGNoICogNAogICAgICAgICAgICAgICAgYjAsIGIx'
    'LCBiMiwgYjMgPSBwZGF0YVtiYXNlOmJhc2UrNF0KICAgICAgICAgICAgICAgIGVmZmVjdCA9IGIy'
    'ICYgMHgwRgogICAgICAgICAgICAgICAgcGFyYW0gID0gYjMKICAgICAgICAgICAgICAgIGlmIGVm'
    'ZmVjdCA9PSAweEYgYW5kIHBhcmFtID4gMDoKICAgICAgICAgICAgICAgICAgICBpZiBwYXJhbSA8'
    'IDB4MjA6CiAgICAgICAgICAgICAgICAgICAgICAgIHNwZWVkID0gcGFyYW0KICAgICAgICAgICAg'
    'ICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgICAgICAgICBpZiBicG0gIT0gcGFyYW06CiAg'
    'ICAgICAgICAgICAgICAgICAgICAgICAgICBicG1fY2hhbmdlcyA9IFRydWUKICAgICAgICAgICAg'
    'ICAgICAgICAgICAgYnBtID0gcGFyYW0KICAgICAgICAgICAgICAgIGVsaWYgZWZmZWN0ID09IDB4'
    'RCBvciBlZmZlY3QgPT0gMHhCOgogICAgICAgICAgICAgICAgICAgIGJyb2tlID0gVHJ1ZSAgICMg'
    'cGF0dGVybiBicmVhayAvIHBvc2l0aW9uIGp1bXAKICAgICAgICAgICAgcm93U3BlZWQuYXBwZW5k'
    'KHNwZWVkKQogICAgICAgICAgICBpZiBicm9rZToKICAgICAgICAgICAgICAgIGJyZWFrICAgIyBz'
    'dG9wIGFkZGluZyByb3dzIGZvciB0aGlzIHNvbmcgcG9zaXRpb24KICAgIHJvd1N0YXJ0VGljayA9'
    'IFswXQogICAgZm9yIHMgaW4gcm93U3BlZWQ6CiAgICAgICAgcm93U3RhcnRUaWNrLmFwcGVuZChy'
    'b3dTdGFydFRpY2tbLTFdICsgcykKICAgIHJldHVybiByb3dTcGVlZCwgcm93U3RhcnRUaWNrLCBi'
    'cG1fY2hhbmdlcwoKCmRlZiBlbmNvZGVfc2FtcGxlc192cTJkKG1vZCwgSz0yNTYsIHdlaWdodGVk'
    'PVRydWUsIGRvd25zYW1wbGU9MiwgYml0cmF0ZT0nbWVkJywgdmVjX2RpbT0yLCBub19ydnEyPUZh'
    'bHNlKToKICAgICIiIjItc3RhZ2UgUmVzaWR1YWwgVlEgd2l0aCBGRlQtZ3VpZGVkIHBlci1zYW1w'
    'bGUgZGVjaW1hdGlvbi4KICAgIFBlci1zYW1wbGUgRFMgdmlhIEZGVCBiYW5kd2lkdGggYW5hbHlz'
    'aXMg4oCUIERTPTEgZm9yIGZ1bGwtYmFuZHdpZHRoIHNhbXBsZXMKICAgIChwcmVzZXJ2ZXMgYWxs'
    'IEhGKSwgb25seSBkb3duc2FtcGxlIGlmIGNvbnRlbnQgaXMgZ2VudWluZWx5IGxvdy1iYW5kd2lk'
    'dGguCiAgICBSYXcgc3RyaWRlIGRlY2ltYXRpb24gKG5vIExQRikuIGJ3RmFjdG9yIHBlciBzYW1w'
    'bGUgPSBhY3R1YWwgRFMgdXNlZC4KICAgICIiIgogICAgaW1wb3J0IG51bXB5IGFzIG5wCiAgICBm'
    'cm9tIHNrbGVhcm4uY2x1c3RlciBpbXBvcnQgTWluaUJhdGNoS01lYW5zCgogICAgIyBCaXRyYXRl'
    'IOKGkiBjb2RlYm9vayBzaXplIChtcDMtc3R5bGUgcXVhbGl0eSBrbm9iKQogICAgX2JpdHJhdGVf'
    'dGFibGUgPSB7CiAgICAgICAgJ2xvJzogICAgKDEyOCwgIDY0KSwgICAjIDEzIGJpdHMvcGFpciwg'
    'c21hbGxlc3QrZ3JhaW55CiAgICAgICAgJ21lZCc6ICAgKDI1NiwgMTI4KSwgICAjIDE1IGJpdHMv'
    'cGFpciwgYmFsYW5jZWQKICAgICAgICAnaGknOiAgICAoNTEyLCAyNTYpLCAgICMgMTcgYml0cy9w'
    'YWlyLCBkZWZhdWx0CiAgICAgICAgJ3VsdHJhJzooMTAyNCwgNTEyKSwgICAjIDE5IGJpdHMvcGFp'
    'ciwgbmVhci10cmFuc3BhcmVudAogICAgfQogICAgSzEsIEsyID0gX2JpdHJhdGVfdGFibGUuZ2V0'
    'KGJpdHJhdGUsIF9iaXRyYXRlX3RhYmxlWydoaSddKQogICAgaWYgbm9fcnZxMjoKICAgICAgICBL'
    'MiA9IDAgICMgc2lnbmFsOiBza2lwIHN0YWdlIDIKICAgIEJJVFMxID0gaW50KG5wLmNlaWwobnAu'
    'bG9nMihLMSkpKQogICAgQklUUzIgPSBpbnQobnAuY2VpbChucC5sb2cyKEsyKSkpIGlmIEsyID4g'
    'MCBlbHNlIDAKICAgIEJJVFNfVE9UQUwgPSBCSVRTMSArIEJJVFMyICAjIGlmIEsyPT0wLCBCSVRT'
    'Mj09MCwgc28ganVzdCBCSVRTMQoKICAgIGRlZiBoZl9yYXRpbyhyYXdfYnl0ZXMsIGxlbmd0aCwg'
    'bnlxdWlzdF9oej0yMjA1MCk6CiAgICAgICAgIiIiRnJhY3Rpb24gb2YgZW5lcmd5IGFib3ZlIDhr'
    'SHog4oCUIGhpZ2ggPSBwZXJjdXNzaW9uL2N5bWJhbC4iIiIKICAgICAgICBpZiBsZW5ndGggPCAz'
    'MjogcmV0dXJuIDAuMAogICAgICAgIGRhdGEgPSBucC5mcm9tYnVmZmVyKHJhd19ieXRlc1s6bGVu'
    'Z3RoXSwgZHR5cGU9bnAuaW50OCkuYXN0eXBlKG5wLmZsb2F0MzIpCiAgICAgICAgZmZ0ICA9IG5w'
    'LmFicyhucC5mZnQucmZmdChkYXRhWzptaW4obGVuZ3RoLCA0MDk2KV0pKQogICAgICAgIGUgICAg'
    'PSBmbG9hdChucC5zdW0oZmZ0KioyKSkgKyAxZS0xMAogICAgICAgIGN1dCAgPSBtYXgoMSwgaW50'
    'KGxlbihmZnQpICogODAwMCAvIG55cXVpc3RfaHopKQogICAgICAgIHJldHVybiBmbG9hdChucC5z'
    'dW0oZmZ0W2N1dDpdKioyKSkgLyBlCgogICAgY29uY2F0X2RzID0gW10KICAgIHN0YXJ0cyAgICA9'
    'IFtdCiAgICBzYW1wbGVfZHMgPSBbXSAgIyBwZXItc2FtcGxlIGFjdHVhbCBEUyB1c2VkCiAgICB0'
    'b3RhbF9zYW1wbGVzX2Z1bGwgPSAwCgogICAgZm9yIHMsIHJhd19ieXRlcyBpbiB6aXAobW9kLnNh'
    'bXBsZXNfaW5mbywgbW9kLnNhbXBsZV9ieXRlcyk6CiAgICAgICAgc3RhcnRzLmFwcGVuZChsZW4o'
    'Y29uY2F0X2RzKSkKICAgICAgICBpZiBzWydsZW5ndGgnXSA+IDA6CiAgICAgICAgICAgIHJhdyA9'
    'IG5wLmZyb21idWZmZXIocmF3X2J5dGVzLCBkdHlwZT1ucC5pbnQ4KS5hc3R5cGUobnAuZmxvYXQz'
    'MikgLyAxMjguMAogICAgICAgICAgICB0b3RhbF9zYW1wbGVzX2Z1bGwgKz0gbGVuKHJhdykKICAg'
    'ICAgICAgICAgIyBQZXItc2FtcGxlIERTIHZpYSBGRlQgYmFuZHdpZHRoIGFuYWx5c2lzIChtaXJy'
    'b3JzIEhUTUwgcGxheWVyJ3MKICAgICAgICAgICAgIyBid19jb21wcmVzc19zYW1wbGUpLiAgLS1k'
    'b3duc2FtcGxlIGlzIGEgQ0FQLCBub3QgYSBmbG9vcjogZnVsbC0KICAgICAgICAgICAgIyBiYW5k'
    'd2lkdGggc2FtcGxlcyAoZ3VpdGFycywgdm9jYWxzKSBzdGF5IGF0IERTPTEsIG5hcnJvdy1iYW5k'
    'CiAgICAgICAgICAgICMgc2FtcGxlcyAobG93IGJhc3MsIG11dGVkIGluc3RydW1lbnRzKSBkcm9w'
    'IHRvIERTPTIvNC84LgogICAgICAgICAgICAjCiAgICAgICAgICAgICMgV2l0aG91dCB0aGlzIHRo'
    'ZSBHTFNMIHdhcyBmb3JjZS1kZWNpbWF0aW5nIGV2ZXJ5IHNhbXBsZSB0bwogICAgICAgICAgICAj'
    'IGRvd25zYW1wbGUsIHByb2R1Y2luZyB0aGUgIjgga0h6IGxvLWZpIiBhcnRpZmFjdHMgdGhlIEhU'
    'TUwKICAgICAgICAgICAgIyBwbGF5ZXIgYXZvaWRlZC4KICAgICAgICAgICAgc3IgPSA0NDEwMC4w'
    'CiAgICAgICAgICAgIG5fZmZ0ID0gbWluKGxlbihyYXcpLCA4MTkyKQogICAgICAgICAgICBmZnRf'
    'bWFnID0gbnAuYWJzKG5wLmZmdC5yZmZ0KHJhd1s6bl9mZnRdICogbnAuaGFubmluZyhuX2ZmdCkp'
    'KQogICAgICAgICAgICBmcmVxcyAgID0gbnAuZmZ0LnJmZnRmcmVxKG5fZmZ0LCAxLjAgLyBzcilb'
    'OmxlbihmZnRfbWFnKV0KICAgICAgICAgICAgcGVhayAgICA9IGZsb2F0KG5wLm1heChmZnRfbWFn'
    'KSkgKyAxZS0xMgogICAgICAgICAgICBzaWdfYmlucyA9IG5wLndoZXJlKGZmdF9tYWcgPiBwZWFr'
    'ICogMC4wMDUpWzBdCiAgICAgICAgICAgIG1heF9mcmVxID0gZmxvYXQoZnJlcXNbc2lnX2JpbnNb'
    'LTFdXSkgaWYgbGVuKHNpZ19iaW5zKSBlbHNlIDIyMDUwLjAKICAgICAgICAgICAgIyBVc2VyJ3Mg'
    'LS1kb3duc2FtcGxlIGlzIHRoZSBGTE9PUiAoYWx3YXlzIGF0IGxlYXN0IHRoaXMgbXVjaCkuCiAg'
    'ICAgICAgICAgICMgQmFuZHdpZHRoIGFuYWx5c2lzIGNhbiBjaG9vc2UgdG8gZGVjaW1hdGUgTU9S'
    'RSBmb3IgZ2VudWluZWx5CiAgICAgICAgICAgICMgbG93LWJhbmR3aWR0aCBjb250ZW50IChlLmcu'
    'IHN1Yi1iYXNzIGF0IERTPTQgZXZlbiB3aGVuIHVzZXIKICAgICAgICAgICAgIyByZXF1ZXN0ZWQg'
    'RFM9MikuICBOZXZlciBsZXNzIHRoYW4gdXNlcidzIHJlcXVlc3QuCiAgICAgICAgICAgIGFjdHVh'
    'bF9kcyA9IGRvd25zYW1wbGUKICAgICAgICAgICAgaWYgZG93bnNhbXBsZSA8IDE2OgogICAgICAg'
    'ICAgICAgICAgIyBUcnkgaGlnaGVyIGZhY3RvcnMgb25seSDigJQgbmV2ZXIgbGVzcyB0aGFuIHVz'
    'ZXIncyByZXF1ZXN0LgogICAgICAgICAgICAgICAgZm9yIGYgaW4gW2Rvd25zYW1wbGUgKiAyLCBk'
    'b3duc2FtcGxlICogNF06CiAgICAgICAgICAgICAgICAgICAgaWYgZiA+IDE2OiBicmVhawogICAg'
    'ICAgICAgICAgICAgICAgIGlmIHNyIC8gZiA+PSBtYXhfZnJlcSAqIDIuNDoKICAgICAgICAgICAg'
    'ICAgICAgICAgICAgYWN0dWFsX2RzID0gZgogICAgICAgICAgICAgICAgICAgIGVsc2U6CiAgICAg'
    'ICAgICAgICAgICAgICAgICAgIGJyZWFrICAjIGlmIDJ4IGRvZXNuJ3Qgc2F0aXNmeSBOeXF1aXN0'
    'LCA0eCB3b24ndCBlaXRoZXIKICAgICAgICAgICAgIyBSYXcgc3RyaWRlIGRlY2ltYXRpb24g4oCU'
    'IG5vIExQRiwgcHJlc2VydmVzIEhGIGNvbnRlbnQKICAgICAgICAgICAgaWYgYWN0dWFsX2RzID4g'
    'MToKICAgICAgICAgICAgICAgIGRzID0gcmF3Wzo6YWN0dWFsX2RzXS5jb3B5KCkKICAgICAgICAg'
    'ICAgZWxzZToKICAgICAgICAgICAgICAgIGRzID0gcmF3LmNvcHkoKQogICAgICAgICAgICBzYW1w'
    'bGVfZHMuYXBwZW5kKGFjdHVhbF9kcykKICAgICAgICAgICAgY29uY2F0X2RzLmV4dGVuZChkcy50'
    'b2xpc3QoKSkKICAgICAgICAgICAgIyBMb29wLXNlYW0gc21vb3RoaW5nOiBmb3IgbG9vcGluZyBz'
    'YW1wbGVzLCByZXBsYWNlIHRoZSBwb3N0LWxvb3AKICAgICAgICAgICAgIyBndWFyZCByZWdpb24g'
    'd2l0aCB0aGUgRklSU1QgZmV3IHNhbXBsZXMgZnJvbSBsb29wX3N0YXJ0LgogICAgICAgICAgICAj'
    'IFRoaXMgbWFrZXMgdmVjdG9ycyBuZWFyIGxvb3BfZW5kIGluY2x1ZGUgcHJvcGVyIHdyYXAgY29u'
    'dGV4dCBzbwogICAgICAgICAgICAjIFZRIHF1YW50aXphdGlvbiBkb2Vzbid0IGludHJvZHVjZSBh'
    'IHN0ZXAgZGlzY29udGludWl0eSBhdCB0aGUKICAgICAgICAgICAgIyBsb29wIGJvdW5kYXJ5LiAg'
    'V2l0aG91dCB0aGlzLCB2ZWNfZGltPTggcHJvZHVjZXMgYW4gYXVkaWJsZSBidXp6CiAgICAgICAg'
    'ICAgICMgYXQgdGhlIGxvb3AgcmF0ZSAoc2FtcGxlW2xvb3BFbmQtMV0gYW5kIHNhbXBsZVtsb29w'
    'U3RhcnRdIGdldAogICAgICAgICAgICAjIHF1YW50aXplZCB0byBpbmNvbXBhdGlibGUgY29kZWJv'
    'b2sgcHJvdG90eXBlcykuCiAgICAgICAgICAgICMKICAgICAgICAgICAgIyBFbmNvZGVyIE1PREZp'
    'bGUgc3RvcmVzIGxvb3Bfc3RhcnQvbG9vcF9sZW4gaW4gUkFXIGJ5dGUgdW5pdHMKICAgICAgICAg'
    'ICAgIyAoYWxyZWFkeSBwcmUtbXVsdGlwbGllZCBieSAyIGluIHNhbXBsZXNfaW5mbykuCiAgICAg'
    'ICAgICAgIGxvb3BfbGVuX3JhdyA9IGludChzLmdldCgnbG9vcF9sZW4nLCAwKSBvciAwKQogICAg'
    'ICAgICAgICBpZiBsb29wX2xlbl9yYXcgPiA0OgogICAgICAgICAgICAgICAgbG9vcF9zdGFydF9y'
    'YXcgPSBpbnQocy5nZXQoJ2xvb3Bfc3RhcnQnLCAwKSBvciAwKQogICAgICAgICAgICAgICAgIyBD'
    'b252ZXJ0IHRvIGRlY2ltYXRlZC1zdHJlYW0gaW5kZXggKG1hdGNoZXMgYGRzYCBhcnJheSBpbmRl'
    'eGluZykKICAgICAgICAgICAgICAgIGxvb3Bfc3RhcnRfZHMgPSBsb29wX3N0YXJ0X3JhdyAvLyBh'
    'Y3R1YWxfZHMKICAgICAgICAgICAgICAgICMgQ29tcHV0ZSB0b3RhbCBwYWRkaW5nIG5lZWRlZDog'
    'YWxpZ24tdG8tdmVjX2RpbSArIGV4dHJhIGd1YXJkCiAgICAgICAgICAgICAgICBwYWRfY291bnQg'
    'PSAodmVjX2RpbSAtIGxlbihjb25jYXRfZHMpICUgdmVjX2RpbSkgJSB2ZWNfZGltICsgOAogICAg'
    'ICAgICAgICAgICAgIyBUYWtlIHBhZF9jb3VudCBzYW1wbGVzIHN0YXJ0aW5nIGZyb20gbG9vcF9z'
    'dGFydCBpbiB0aGUKICAgICAgICAgICAgICAgICMgZGVjaW1hdGVkIGRhdGEg4oCUIHRoaXMgaXMg'
    'd2hhdCBwbGF5YmFjayB3cmFwcyB0by4KICAgICAgICAgICAgICAgIHdyYXBfZGF0YSA9IFtdCiAg'
    'ICAgICAgICAgICAgICBpZiBsb29wX3N0YXJ0X2RzIDwgbGVuKGRzKToKICAgICAgICAgICAgICAg'
    'ICAgICB0YWtlID0gbWluKHBhZF9jb3VudCwgbGVuKGRzKSAtIGxvb3Bfc3RhcnRfZHMpCiAgICAg'
    'ICAgICAgICAgICAgICAgd3JhcF9kYXRhLmV4dGVuZChkcy50b2xpc3QoKVtsb29wX3N0YXJ0X2Rz'
    'Omxvb3Bfc3RhcnRfZHMrdGFrZV0pCiAgICAgICAgICAgICAgICB3aGlsZSBsZW4od3JhcF9kYXRh'
    'KSA8IHBhZF9jb3VudDoKICAgICAgICAgICAgICAgICAgICB3cmFwX2RhdGEuYXBwZW5kKDApCiAg'
    'ICAgICAgICAgICAgICBjb25jYXRfZHMuZXh0ZW5kKHdyYXBfZGF0YSkKICAgICAgICAgICAgZWxz'
    'ZToKICAgICAgICAgICAgICAgICMgTm9uLWxvb3Bpbmc6IHBhZCB3aXRoIHplcm9zIChvcmlnaW5h'
    'bCBiZWhhdmlvcikKICAgICAgICAgICAgICAgIHdoaWxlIGxlbihjb25jYXRfZHMpICUgdmVjX2Rp'
    'bTogY29uY2F0X2RzLmFwcGVuZCgwKQogICAgICAgICAgICAgICAgY29uY2F0X2RzLmV4dGVuZChb'
    'MF0gKiA4KQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHNhbXBsZV9kcy5hcHBlbmQoZG93bnNh'
    'bXBsZSkKICAgICAgICAgICAgIyBFbXB0eSBzYW1wbGU6IGp1c3QgcGFkCiAgICAgICAgICAgIHdo'
    'aWxlIGxlbihjb25jYXRfZHMpICUgdmVjX2RpbTogY29uY2F0X2RzLmFwcGVuZCgwKQogICAgICAg'
    'ICAgICBjb25jYXRfZHMuZXh0ZW5kKFswXSAqIDgpCgogICAgd2hpbGUgbGVuKGNvbmNhdF9kcykg'
    'JSB2ZWNfZGltOiBjb25jYXRfZHMuYXBwZW5kKDApCiAgICB0b3RhbF9zYW1wbGVzID0gbGVuKGNv'
    'bmNhdF9kcykKCiAgICB2ZWN0b3JzID0gbnAuYXJyYXkoY29uY2F0X2RzLCBkdHlwZT1ucC5mbG9h'
    'dDMyKS5yZXNoYXBlKC0xLCB2ZWNfZGltKQoKICAgICMgU3RhZ2UgMSDigJQgcmluZy13ZWlnaHRl'
    'ZAogICAgd2VpZ2h0cyA9IE5vbmUKICAgIGlmIHdlaWdodGVkOgogICAgICAgIHNsb3BlcyAgPSBu'
    'cC5hYnModmVjdG9yc1s6LCAtMV0gLSB2ZWN0b3JzWzosIDBdKQogICAgICAgIHdlaWdodHMgPSAo'
    'c2xvcGVzICsgMS4wKQogICAgICAgIHdlaWdodHMgLz0gd2VpZ2h0cy5tZWFuKCkKCiAgICBwcmlu'
    'dChmIiAgUlZRIMOXe2Rvd25zYW1wbGV9IFN0YWdlIDE6IEs9e0sxfSBvbiB7bGVuKHZlY3RvcnMp'
    'fSB7dmVjX2RpbX0tdmVjdG9ycy4uLiIsIGZsdXNoPVRydWUpCiAgICBrbTEgPSBNaW5pQmF0Y2hL'
    'TWVhbnMobl9jbHVzdGVycz1LMSwgbl9pbml0PTUsIG1heF9pdGVyPTYwLCBiYXRjaF9zaXplPTgx'
    'OTIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgcmFuZG9tX3N0YXRlPTAsIHJlYXNzaWdubWVu'
    'dF9yYXRpbz0wLjAxKQogICAga20xLmZpdCh2ZWN0b3JzLCBzYW1wbGVfd2VpZ2h0PXdlaWdodHMp'
    'CiAgICBjb2RlczEgICA9IGttMS5wcmVkaWN0KHZlY3RvcnMpLmFzdHlwZShucC5pbnQzMikKICAg'
    'ICMgQ2VudHJvaWRzIGFyZSBpbiBbLTEsMV0gZmxvYXQgcmFuZ2Ug4oCUIHNjYWxlIGJhY2sgdG8g'
    'Wy0xMjgsMTI3XSBpbnQgcmFuZ2UgZm9yIHN0b3JhZ2UKICAgIGNiMSAgICAgID0gbnAuY2xpcChu'
    'cC5yb3VuZChrbTEuY2x1c3Rlcl9jZW50ZXJzXyAqIDEyOCksIC0xMjgsIDEyNykuYXN0eXBlKG5w'
    'LmludDMyKQogICAgcmVzaWR1YWwgPSB2ZWN0b3JzIC0ga20xLmNsdXN0ZXJfY2VudGVyc19bY29k'
    'ZXMxXQoKICAgIHNucjEgPSAxMCpucC5sb2cxMChucC5tZWFuKHZlY3RvcnMqKjIpIC8gKG5wLm1l'
    'YW4ocmVzaWR1YWwqKjIpICsgMWUtOSkpCiAgICBwcmludChmIiAgU3RhZ2UgMSBTTlI6IHtzbnIx'
    'Oi4yZn0gZEIiLCBmbHVzaD1UcnVlKQoKICAgICMgU3RhZ2UgMiAoc2tpcHBlZCB3aGVuIG5vX3J2'
    'cTIg4oaSIEsyPT0wKQogICAgaWYgSzIgPiAwOgogICAgICAgIHByaW50KGYiICBSVlEgU3RhZ2Ug'
    'MjogSz17SzJ9IG9uIHJlc2lkdWFsLi4uIiwgZmx1c2g9VHJ1ZSkKICAgICAgICBrbTIgPSBNaW5p'
    'QmF0Y2hLTWVhbnMobl9jbHVzdGVycz1LMiwgbl9pbml0PTUsIG1heF9pdGVyPTYwLCBiYXRjaF9z'
    'aXplPTgxOTIsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHJhbmRvbV9zdGF0ZT0xLCBy'
    'ZWFzc2lnbm1lbnRfcmF0aW89MC4wMSkKICAgICAgICBrbTIuZml0KHJlc2lkdWFsKQogICAgICAg'
    'IGNvZGVzMiAgICAgICAgID0ga20yLnByZWRpY3QocmVzaWR1YWwpLmFzdHlwZShucC5pbnQzMikK'
    'ICAgICAgICBjYjIgICAgICAgICAgICA9IG5wLmNsaXAobnAucm91bmQoa20yLmNsdXN0ZXJfY2Vu'
    'dGVyc18gKiAxMjgpLCAtMTI4LCAxMjcpLmFzdHlwZShucC5pbnQzMikKICAgICAgICBmaW5hbF9y'
    'ZXNpZHVhbCA9IHJlc2lkdWFsIC0ga20yLmNsdXN0ZXJfY2VudGVyc19bY29kZXMyXQogICAgICAg'
    'IHNucjIgPSAxMCpucC5sb2cxMChucC5tZWFuKHZlY3RvcnMqKjIpIC8gKG5wLm1lYW4oZmluYWxf'
    'cmVzaWR1YWwqKjIpICsgMWUtOSkpCiAgICAgICAgcHJpbnQoZiIgIFJWUSB0b3RhbCBTTlI6IHtz'
    'bnIyOi4yZn0gZEIgKCt7c25yMi1zbnIxOi4yZn0gZEIgZnJvbSBzdGFnZSAyKSIsIGZsdXNoPVRy'
    'dWUpCiAgICBlbHNlOgogICAgICAgIHByaW50KGYiICDimqEgU3RhZ2UgMiBza2lwcGVkICgtLW5v'
    'LXJ2cTIpOiBTTlIgPSB7c25yMTouMmZ9IGRCIiwgZmx1c2g9VHJ1ZSkKICAgICAgICBjb2RlczIg'
    'PSBucC56ZXJvc19saWtlKGNvZGVzMSkgICMgcGxhY2Vob2xkZXIKICAgICAgICBjYjIgICAgPSBu'
    'cC56ZXJvcygoMCwgdmVjX2RpbSksIGR0eXBlPW5wLmludDMyKSAgIyBlbXB0eSBLMiBjb2RlYm9v'
    'awoKICAgICMgUGFjayBCSVRTMStCSVRTMiBiaXRzIHBlciB2ZWN0b3IgTFNCLWZpcnN0CiAgICAj'
    'IFdoZW4gbm9fcnZxMiAoSzI9PTApLCBCSVRTMj09MCBzbyBjb21iaW5lZCBjb2xsYXBzZXMgdG8g'
    'anVzdCBjb2RlczEgYml0cy4KICAgIG5fdmVjcyAgICAgID0gbGVuKHZlY3RvcnMpCiAgICB0b3Rh'
    'bF9iaXRzICA9IG5fdmVjcyAqIEJJVFNfVE9UQUwKICAgIHRvdGFsX2J5dGVzID0gKHRvdGFsX2Jp'
    'dHMgKyA3KSAvLyA4CiAgICBjb2Rlc19ieXRlcyA9IGJ5dGVhcnJheSh0b3RhbF9ieXRlcykKICAg'
    'IG1hc2sxID0gKDEgPDwgQklUUzEpIC0gMQogICAgbWFzazIgPSAoMSA8PCBCSVRTMikgLSAxIGlm'
    'IEJJVFMyID4gMCBlbHNlIDAKICAgIGZvciBpIGluIHJhbmdlKG5fdmVjcyk6CiAgICAgICAgaWYg'
    'SzIgPiAwOgogICAgICAgICAgICBjb21iaW5lZCAgPSAoaW50KGNvZGVzMVtpXSkgJiBtYXNrMSkg'
    'fCAoKGludChjb2RlczJbaV0pICYgbWFzazIpIDw8IEJJVFMxKQogICAgICAgIGVsc2U6CiAgICAg'
    'ICAgICAgIGNvbWJpbmVkICA9IGludChjb2RlczFbaV0pICYgbWFzazEKICAgICAgICBiaXRfcG9z'
    'ICAgPSBpICogQklUU19UT1RBTAogICAgICAgIGJ5dGVfcG9zICA9IGJpdF9wb3MgPj4gMwogICAg'
    'ICAgIGJpdF9zaGlmdCA9IGJpdF9wb3MgJiA3CiAgICAgICAgdmFsID0gY29tYmluZWQgPDwgYml0'
    'X3NoaWZ0CiAgICAgICAgY29kZXNfYnl0ZXNbYnl0ZV9wb3NdICAgICB8PSB2YWwgICAgICAgICYg'
    'MHhGRgogICAgICAgIGlmIGJ5dGVfcG9zKzEgPCB0b3RhbF9ieXRlczogY29kZXNfYnl0ZXNbYnl0'
    'ZV9wb3MrMV0gfD0gKHZhbCA+PiA4KSAgJiAweEZGCiAgICAgICAgaWYgYnl0ZV9wb3MrMiA8IHRv'
    'dGFsX2J5dGVzOiBjb2Rlc19ieXRlc1tieXRlX3BvcysyXSB8PSAodmFsID4+IDE2KSAmIDB4RkYK'
    'ICAgICAgICBpZiBieXRlX3BvcyszIDwgdG90YWxfYnl0ZXM6IGNvZGVzX2J5dGVzW2J5dGVfcG9z'
    'KzNdIHw9ICh2YWwgPj4gMjQpICYgMHhGRgoKICAgICMgQ29kZWJvb2sgYnl0ZXM6IFtLMcOXMiBi'
    'eXRlc11bSzLDlzIgYnl0ZXNdIHN0b3JlZCB1bnNpZ25lZCAoKzEyOCkKICAgIGNiX2J5dGVzID0g'
    'Ynl0ZWFycmF5KCkKICAgIGZvciBlbnRyeSBpbiBjYjE6CiAgICAgICAgZm9yIHYgaW4gZW50cnk6'
    'IGNiX2J5dGVzLmFwcGVuZCgoaW50KHYpKzI1NikgJiAweEZGKQogICAgaWYgSzIgPiAwOgogICAg'
    'ICAgIGZvciBlbnRyeSBpbiBjYjI6CiAgICAgICAgICAgIGZvciB2IGluIGVudHJ5OiBjYl9ieXRl'
    'cy5hcHBlbmQoKGludCh2KSsyNTYpICYgMHhGRikKCiAgICByZXR1cm4gY29kZXNfYnl0ZXMsIGNi'
    'X2J5dGVzLCBzdGFydHMsIHRvdGFsX3NhbXBsZXMsIEJJVFNfVE9UQUwsIHNhbXBsZV9kcywgSzEs'
    'IEsyCgoKIyDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZAKIyBH'
    'TFNMIEVNSVRURVJTCiMg4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQCgpkZWYgYnl0ZXNfdG9faW50MzJfYmVfYXJyYXkoZGF0YSwgY2h1bmtfaXZlYzQ9NTEyKToK'
    'ICAgICIiIlBhY2sgYnl0ZXMgaW50byBpdmVjNCBhcnJheXMgKGJpZy1lbmRpYW46IGJ5dGUgMCA9'
    'IE1TQiBvZiBpbnQueCkuIiIiCiAgICAjIFBhZCB0byBtdWx0aXBsZSBvZiAxNiAoc2luY2UgZWFj'
    'aCBpdmVjNCBob2xkcyAxNiBieXRlcykKICAgIHBhZGRlZCA9IGJ5dGVzKGRhdGEpICsgYidceDAw'
    'JyAqICgoMTYgLSBsZW4oZGF0YSkgJSAxNikgJSAxNikKICAgIGludHMgPSBbXQogICAgZm9yIGkg'
    'aW4gcmFuZ2UoMCwgbGVuKHBhZGRlZCksIDQpOgogICAgICAgIHYgPSBzdHJ1Y3QudW5wYWNrKCc+'
    'SScsIHBhZGRlZFtpOmkrNF0pWzBdCiAgICAgICAgIyBjb252ZXJ0IHRvIHNpZ25lZCBpbnQzMiBm'
    'b3IgR0xTTCAoaGFuZGxlcyB2YWx1ZXMgPj0gMl4zMSkKICAgICAgICBpZiB2ID49ICgxIDw8IDMx'
    'KToKICAgICAgICAgICAgdiAtPSAoMSA8PCAzMikKICAgICAgICBpbnRzLmFwcGVuZCh2KQogICAg'
    'IyBTcGxpdCBpbnRvIGl2ZWM0IGFycmF5IGNodW5rcyBvZiBjaHVua19pdmVjNCBpdmVjNCBlbnRy'
    'aWVzIGVhY2gKICAgIGNodW5rcyA9IFtdCiAgICBjdXIgPSBbXQogICAgZm9yIGkgaW4gcmFuZ2Uo'
    'MCwgbGVuKGludHMpLCA0KToKICAgICAgICBjdXIuYXBwZW5kKHR1cGxlKGludHNbaTppKzRdKSkK'
    'ICAgICAgICBpZiBsZW4oY3VyKSA9PSBjaHVua19pdmVjNDoKICAgICAgICAgICAgY2h1bmtzLmFw'
    'cGVuZChjdXIpCiAgICAgICAgICAgIGN1ciA9IFtdCiAgICBpZiBjdXI6CiAgICAgICAgY2h1bmtz'
    'LmFwcGVuZChjdXIpCiAgICByZXR1cm4gY2h1bmtzCgpkZWYgZW1pdF9pdmVjNF9hcnJheShuYW1l'
    'LCBjaHVua3Nfb3Jfc2luZ2xlLCBpdGVtc19wZXJfbGluZT0yKToKICAgICIiIkVtaXQgb25lIG9y'
    'IG1vcmUgY29uc3QgaXZlYzQgYXJyYXlzLiBgY2h1bmtzX29yX3NpbmdsZWAgaXMgYSBsaXN0IG9m'
    'IGNodW5rcy4iIiIKICAgIG91dCA9IFtdCiAgICBmb3IgY2ksIGNodW5rIGluIGVudW1lcmF0ZShj'
    'aHVua3Nfb3Jfc2luZ2xlKToKICAgICAgICBhcnJfbmFtZSA9IGYie25hbWV9e2NpfSIgaWYgbGVu'
    'KGNodW5rc19vcl9zaW5nbGUpID4gMSBlbHNlIGYie25hbWV9MCIKICAgICAgICBvdXQuYXBwZW5k'
    'KGYiY29uc3QgaXZlYzQge2Fycl9uYW1lfVt7bGVuKGNodW5rKX1dID0gaXZlYzRbXSgiKQogICAg'
    'ICAgIGxpbmVzID0gW10KICAgICAgICBmb3Igcm93X3N0YXJ0IGluIHJhbmdlKDAsIGxlbihjaHVu'
    'ayksIGl0ZW1zX3Blcl9saW5lKToKICAgICAgICAgICAgcm93ID0gY2h1bmtbcm93X3N0YXJ0OnJv'
    'd19zdGFydCArIGl0ZW1zX3Blcl9saW5lXQogICAgICAgICAgICBwYXJ0cyA9IFsiaXZlYzQoe30s'
    'e30se30se30pIi5mb3JtYXQoKnQpIGZvciB0IGluIHJvd10KICAgICAgICAgICAgbGluZXMuYXBw'
    'ZW5kKCIgICAgIiArICIsICIuam9pbihwYXJ0cykpCiAgICAgICAgb3V0LmFwcGVuZCgiLFxuIi5q'
    'b2luKGxpbmVzKSkKICAgICAgICBvdXQuYXBwZW5kKCIpO1xuIikKICAgIHJldHVybiAiXG4iLmpv'
    'aW4ob3V0KQoKCiMg4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    'CiMgTUFJTiBCVUlMRAojIOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkAoKZGVmIG1haW4obW9kX3BhdGgsIG91dF9wYXRoLCBLPTI1Niwgd2VpZ2h0ZWQ9VHJ1ZSwg'
    'ZG93bnNhbXBsZT0yLCBiaXRyYXRlPSdoaScsIHZlY19kaW09MiwgcmVzYW1wbGVyPSdic3BsaW5l'
    'Jywgbm9fcnZxMj1GYWxzZSk6CiAgICAiIiJHZW5lcmF0ZSBTaGFkZXJUb3kgQ29tbW9uIEdMU0wg'
    'Zm9yIGEgTU9EIGZpbGUuCiAgICBkb3duc2FtcGxlOiBhbnRpLWFsaWFzIGRvd25zYW1wbGUgZmFj'
    'dG9yIGZvciBzYW1wbGUgZW5jb2RpbmcgKDE9b2ZmLCAyPXJlY29tbWVuZGVkKS4KICAgICAgICAg'
    'ICAgICAgIEhpZ2hlciBkb3duc2FtcGxlIOKGkiBzbWFsbGVyIGRhdGEsIGxhcmdlciBjb2RlYm9v'
    'a3MsIGJldHRlciBTTlIuCiAgICAgICAgICAgICAgICAgIMOXMTogSzE9NjQsICBLMj0zMiAgKH43'
    'OSBLQiwgIDI3LjcgZEIpCiAgICAgICAgICAgICAgICAgIMOXMjogSzE9NTEyLCBLMj0yNTYgKH43'
    'NyBLQiwgIDM4LjQgZEIpICDihpAgcmVjb21tZW5kZWQKICAgICAgICAgICAgICAgICAgw5c0OiBL'
    'MT01MTIsIEsyPTI1NiAofjM5IEtCLCAgMzcuMSBkQikKICAgICIiIgogICAgbW9kID0gTU9ERmls'
    'ZShtb2RfcGF0aCkKICAgIHByaW50KGYi8J+TpiBMb2FkZWQ6IHttb2QudGl0bGV9IikKICAgIHBy'
    'aW50KGYiICAgUGF0dGVybnM6IHttb2QubnVtX3BhdHRlcm5zfSwgU29uZyBsZW5ndGg6IHttb2Qu'
    'c29uZ19sZW5ndGh9IikKCiAgICAjIFBhdHRlcm4gY3J1bmNoCiAgICBwID0gZW5jb2RlX3BhdHRl'
    'cm5zKG1vZCkKICAgIHByaW50KGYiXG7wn5ec77iPICBQQVRURVJOIENSVU5DSCIpCiAgICBwcmlu'
    'dChmIiAgIFRvdGFsIG5vdGVzOiAgICAgICB7cFsndG90YWxfbm90ZXMnXX0iKQogICAgcHJpbnQo'
    'ZiIgICBVbmlxdWUgbm9uLWVtcHR5OiAge2xlbihwWyd1bmlxJ10pfSIpCiAgICBwcmludChmIiAg'
    'IERpY3Rpb25hcnk6ICAgICAgICB7bGVuKHBbJ3VuaXEnXSkqNH0gYnl0ZXMiKQogICAgcHJpbnQo'
    'ZiIgICBCaXRtYXA6ICAgICAgICAgICAge2xlbihwWydiaXRtYXAnXSl9IGJ5dGVzIikKICAgIHBy'
    'aW50KGYiICAgSW5kZXggc3RyZWFtOiAgICAgIHtsZW4ocFsnaWR4X3N0cmVhbSddKX0gYnl0ZXMi'
    'KQogICAgcHJpbnQoZiIgICBSb3cgc2VlayAoMTYtYml0IHByZWZpeCk6IHtsZW4ocFsncm93X3Nl'
    'ZWtfYnl0ZXMnXSl9IGJ5dGVzIikKICAgIHBhdHRlcm5fdG90YWwgPSBsZW4ocFsndW5pcSddKSo0'
    'ICsgbGVuKHBbJ2JpdG1hcCddKSArIGxlbihwWydpZHhfc3RyZWFtJ10pICsgbGVuKHBbJ3Jvd19z'
    'ZWVrX2J5dGVzJ10pCiAgICBwcmludChmIiAgIOKGkiBQYXR0ZXJuIHRvdGFsOiAgIHtwYXR0ZXJu'
    'X3RvdGFsOix9IGJ5dGVzIikKCiAgICAjIFNwZWVkL3RpY2sgdGFibGUgZnJvbSBGeHggZWZmZWN0'
    'cwogICAgcm93U3BlZWQsIHJvd1N0YXJ0VGljaywgYnBtX2NoYW5nZXMgPSBjb21wdXRlX3Jvd19z'
    'cGVlZF90YWJsZShtb2QpCiAgICBwcmludChmIlxu4o+x77iPICBTUEVFRCBUQUJMRSIpCiAgICBw'
    'cmludChmIiAgIFNvbmcgcm93czoge2xlbihyb3dTcGVlZCl9LCB0b3RhbCB0aWNrczoge3Jvd1N0'
    'YXJ0VGlja1stMV19IikKICAgIHByaW50KGYiICAgVW5pcXVlIHNwZWVkczoge3NvcnRlZChzZXQo'
    'cm93U3BlZWQpKX0iKQogICAgc3BlZWRfdGFibGVfYnl0ZXMgPSBsZW4ocm93U3RhcnRUaWNrKSAq'
    'IDIKICAgIHByaW50KGYiICAgcm93U3RhcnRUaWNrOiB7c3BlZWRfdGFibGVfYnl0ZXN9IGJ5dGVz'
    'ICgxNi1iaXQgcGFja2VkKSIpCiAgICBpZiBicG1fY2hhbmdlczoKICAgICAgICBwcmludChmIiAg'
    'IOKaoO+4jyAgQlBNIGNoYW5nZXMgZGV0ZWN0ZWQgKDEyVEguTU9EIGhhcyBub25lLCBidXQgb3Ro'
    'ZXIgTU9EcyBtaWdodCkiKQoKICAgICMgQXV0by1zZWxlY3QgZG93bnNhbXBsZSBpZiBub3QgZXhw'
    'bGljaXRseSBvdmVycmlkZGVuIChkb3duc2FtcGxlPTIgaXMgZGVmYXVsdCkKICAgICMgQnVkZ2V0'
    'IGVzdGltYXRlOiB0b3RhbF9yYXdfYnl0ZXMgLyBkb3duc2FtcGxlICogMTcvMTYgKDE3LWJpdCBj'
    'b2RlcywgMiBieXRlcy9zYW1wbGUpCiAgICAjIFNoYWRlclRveSBzYWZlIHpvbmU6IOKJpCA4MCBL'
    'QiBzYW1wbGUgY29kZXMgKyBwYXR0ZXJuIGRhdGEKICAgIGltcG9ydCBudW1weSBhcyBucAogICAg'
    'dG90YWxfcmF3ID0gc3VtKHNbJ2xlbmd0aCddIGZvciBzIGluIG1vZC5zYW1wbGVzX2luZm8pCiAg'
    'ICAjIE5PVEU6IHVzZXItcmVxdWVzdGVkIGRvd25zYW1wbGUgaXMgcmVzcGVjdGVkIGFzIGEgSEFS'
    'RCBDQVAg4oCUIG5vIGF1dG8tYnVtcC4KICAgICMgVXNlIC0tdmVjLWRpbSA0IG9yIC0tYml0cmF0'
    'ZSBsbyBpZiB5b3UgbmVlZCBtb3JlIHNpemUgcmVkdWN0aW9uLgogICAgZXN0aW1hdGVkX2J1ZGdl'
    'dF9kczIgPSAodG90YWxfcmF3IC8vIGRvd25zYW1wbGUpICogMTcgLy8gMTYgKyAxNjAwMCAgIyBm'
    'b3IgbG9nIG9ubHkKCiAgICAjIFNhbXBsZSBlbmNvZGluZzogUlZRIHdpdGggYml0cmF0ZS1jb250'
    'cm9sbGVkIGNvZGVib29rICsgcGVyLXNhbXBsZSBEUwogICAgZHNfbGFiZWwgPSBmIsOXe2Rvd25z'
    'YW1wbGV9IiBpZiBkb3duc2FtcGxlID4gMSBlbHNlICJmdWxsLXJlcyIKICAgIHByaW50KGYiXG7w'
    'n5ec77iPICBTQU1QTEUgQ1JVTkNIIChSVlEge2RzX2xhYmVsfSBiaXRyYXRlPXtiaXRyYXRlfSwg'
    'cmluZy13ZWlnaHRlZCkiKQogICAgY29kZXNfYnl0ZXMsIGNiX2J5dGVzLCBzdGFydHMsIHRvdGFs'
    'X3NhbXBsZXMsIGJpdHNfcGVyX2NvZGUsIHNhbXBsZV9kcywgSzEsIEsyID0gZW5jb2RlX3NhbXBs'
    'ZXNfdnEyZCgKICAgICAgICBtb2QsIEssIHdlaWdodGVkLCBkb3duc2FtcGxlPWRvd25zYW1wbGUs'
    'IGJpdHJhdGU9Yml0cmF0ZSwgdmVjX2RpbT12ZWNfZGltLCBub19ydnEyPW5vX3J2cTIpCiAgICBC'
    'SVRTMSA9IGludChucC5jZWlsKG5wLmxvZzIoSzEpKSkKICAgIEJJVFMyID0gaW50KG5wLmNlaWwo'
    'bnAubG9nMihLMikpKSBpZiBLMiA+IDAgZWxzZSAwCiAgICBCSVRTX1RPVEFMID0gYml0c19wZXJf'
    'Y29kZQogICAgcHJpbnQoZiIgICBMb2dpY2FsIHNhbXBsZXM6ICAge3RvdGFsX3NhbXBsZXM6LH0g'
    'ICh7ZHNfbGFiZWx9KSIpCiAgICBwcmludChmIiAgIENvZGVzIHBhY2tlZDogICAgICB7bGVuKGNv'
    'ZGVzX2J5dGVzKTosfSBieXRlcyAgKHtiaXRzX3Blcl9jb2RlfSBiaXRzL3ZlY3RvciDDlyB7dG90'
    'YWxfc2FtcGxlcy8vMn0gdmVjdG9ycykiKQogICAgcHJpbnQoZiIgICBDb2RlYm9va3M6ICAgICAg'
    'ICAge2xlbihjYl9ieXRlcyk6LH0gYnl0ZXMgICh7SzF9w5cyICsge0syfcOXMiBieXRlcykiKQoK'
    'ICAgIHRvdGFsX2J1ZGdldCA9IHBhdHRlcm5fdG90YWwgKyBsZW4oY29kZXNfYnl0ZXMpICsgbGVu'
    'KGNiX2J5dGVzKSArIDMxKjI0ICsgc3BlZWRfdGFibGVfYnl0ZXMKICAgIHByaW50KGYiXG7wn5OK'
    'IFRPVEFMIGNvbnN0IGRhdGEgYnVkZ2V0OiB+e3RvdGFsX2J1ZGdldDosfSBieXRlcyAgKHt0b3Rh'
    'bF9idWRnZXQvMTAyNDouMWZ9IEtCKSIpCgogICAgIyBDaHVuayBmb3IgR0xTTAogICAgZGljdF9i'
    'eXRlcyA9IGInJy5qb2luKHBbJ3VuaXEnXSkKICAgIGRpY3RfY2h1bmtzICAgID0gYnl0ZXNfdG9f'
    'aW50MzJfYmVfYXJyYXkoZGljdF9ieXRlcykKICAgIGJpdG1hcF9jaHVua3MgID0gYnl0ZXNfdG9f'
    'aW50MzJfYmVfYXJyYXkoYnl0ZXMocFsnYml0bWFwJ10pKQogICAgaWR4X2NodW5rcyAgICAgPSBi'
    'eXRlc190b19pbnQzMl9iZV9hcnJheShieXRlcyhwWydpZHhfc3RyZWFtJ10pKQogICAgcm93c2Vl'
    'a19jaHVua3MgPSBieXRlc190b19pbnQzMl9iZV9hcnJheShieXRlcyhwWydyb3dfc2Vla19ieXRl'
    'cyddKSkKICAgIGNvZGVzX2NodW5rcyAgID0gYnl0ZXNfdG9faW50MzJfYmVfYXJyYXkoYnl0ZXMo'
    'Y29kZXNfYnl0ZXMpKQogICAgY2JfY2h1bmtzICAgICAgPSBieXRlc190b19pbnQzMl9iZV9hcnJh'
    'eShieXRlcyhjYl9ieXRlcykpCgogICAgIyBQYWNrIHJvd1N0YXJ0VGljayBhcyAxNi1iaXQgTEUg'
    'Ynl0ZXMg4oaSIGl2ZWM0IGNodW5rcwogICAgdGlja19ieXRlcyA9IGJ5dGVhcnJheSgpCiAgICBm'
    'b3IgdCBpbiByb3dTdGFydFRpY2s6CiAgICAgICAgdGlja19ieXRlcy5hcHBlbmQodCAmIDB4RkYp'
    'CiAgICAgICAgdGlja19ieXRlcy5hcHBlbmQoKHQgPj4gOCkgJiAweEZGKQogICAgdGlja19jaHVu'
    'a3MgPSBieXRlc190b19pbnQzMl9iZV9hcnJheShieXRlcyh0aWNrX2J5dGVzKSkKCiAgICBzYW1w'
    'bGVzX2luZm9fbmV3ID0gW10KICAgIGZvciBpLCAocywgc3QpIGluIGVudW1lcmF0ZSh6aXAobW9k'
    'LnNhbXBsZXNfaW5mbywgc3RhcnRzKSk6CiAgICAgICAgIyBVc2UgcGVyLXNhbXBsZSBhY3R1YWwg'
    'RFMgZm9yIGxlbmd0aC9sb29wIHNjYWxpbmcKICAgICAgICBzZHMgPSBzYW1wbGVfZHNbaV0gaWYg'
    'aSA8IGxlbihzYW1wbGVfZHMpIGVsc2UgZG93bnNhbXBsZQogICAgICAgIHNhbXBsZXNfaW5mb19u'
    'ZXcuYXBwZW5kKGRpY3QoCiAgICAgICAgICAgIHN0YXJ0PXN0LAogICAgICAgICAgICBsZW5ndGg9'
    'c1snbGVuZ3RoJ10gLy8gc2RzLAogICAgICAgICAgICBsb29wU3RhcnQ9c1snbG9vcF9zdGFydCdd'
    'IC8vIHNkcywKICAgICAgICAgICAgbG9vcExlbj1zWydsb29wX2xlbiddIC8vIHNkcywKICAgICAg'
    'ICAgICAgdm9sdW1lPXNbJ3ZvbHVtZSddLCBmaW5ldHVuZT1zWydmaW5ldHVuZSddLAogICAgICAg'
    'ICAgICBid0ZhY3Rvcj1zZHMsICAgIyBhY3R1YWwgRFMg4oCUIHVzZWQgYnkgR0xTTCBhcyBmcmVx'
    'IGRpdmlzb3IKICAgICAgICApKQoKICAgIGdsc2wgPSBidWlsZF9nbHNsKG1vZCwgcCwgY29kZXNf'
    'Ynl0ZXMsIHN0YXJ0cywgdG90YWxfc2FtcGxlcywKICAgICAgICAgICAgICAgICAgICAgZGljdF9j'
    'aHVua3MsIGJpdG1hcF9jaHVua3MsIGlkeF9jaHVua3MsIHJvd3NlZWtfY2h1bmtzLAogICAgICAg'
    'ICAgICAgICAgICAgICBjb2Rlc19jaHVua3MsIGNiX2NodW5rcywgc2FtcGxlc19pbmZvX25ldywg'
    'SywgYml0c19wZXJfY29kZSwKICAgICAgICAgICAgICAgICAgICAgdGlja19jaHVua3MsIHJvd1N0'
    'YXJ0VGljaywKICAgICAgICAgICAgICAgICAgICAgSzE9SzEsIEsyPUsyLCBCSVRTMT1CSVRTMSwg'
    'QklUUzI9QklUUzIsIEJJVFNfVE9UQUw9QklUU19UT1RBTCwKICAgICAgICAgICAgICAgICAgICAg'
    'ZG93bnNhbXBsZT1kb3duc2FtcGxlLCB2ZWNfZGltPXZlY19kaW0sIHJlc2FtcGxlcj1yZXNhbXBs'
    'ZXIsIG5vX3J2cTI9bm9fcnZxMikKICAgIHdpdGggb3BlbihvdXRfcGF0aCwgJ3cnKSBhcyBmOgog'
    'ICAgICAgIGYud3JpdGUoZ2xzbCkKICAgIHByaW50KGYiXG7inIUgV3JvdGU6IHtvdXRfcGF0aH0g'
    'ICh7bGVuKGdsc2wuZW5jb2RlKCd1dGYtOCcpKTosfSBieXRlcykiKQoKCmRlZiBidWlsZF9nbHNs'
    'KG1vZCwgcCwgcGFja2VkLCBzdGFydHMsIHRvdGFsX3NhbXBsZXMsCiAgICAgICAgICAgICAgIGRp'
    'Y3RfY2h1bmtzLCBiaXRtYXBfY2h1bmtzLCBpZHhfY2h1bmtzLCByb3dzZWVrX2NodW5rcywKICAg'
    'ICAgICAgICAgICAgY29kZXNfY2h1bmtzLCBjYl9jaHVua3MsIHNhbXBsZXNfaW5mb19uZXcsIEss'
    'IGJpdHNfcGVyX2NvZGUsCiAgICAgICAgICAgICAgIHRpY2tfY2h1bmtzLCByb3dTdGFydFRpY2ss'
    'IEsxPTUxMiwgSzI9MjU2LCBCSVRTMT05LCBCSVRTMj04LCBCSVRTX1RPVEFMPTE3LCBkb3duc2Ft'
    'cGxlPTIsIHZlY19kaW09MiwgcmVzYW1wbGVyPSdic3BsaW5lJywgbm9fcnZxMj1GYWxzZSk6Cgog'
    'ICAgIyDilIDilIAgU29uZyBtZXRhZGF0YQogICAgc29uZ19wb3NpdGlvbnMgPSBtb2QucGF0dGVy'
    'bl9vcmRlcls6bW9kLnNvbmdfbGVuZ3RoXQoKICAgICMgQ29tcHV0ZSBhY3R1YWwgcm93cyBwZXIg'
    'c29uZyBwb3NpdGlvbiDigJQgUHJvVHJhY2tlciBEeHggKHBhdHRlcm4gYnJlYWspCiAgICAjIGFu'
    'ZCBCeHggKHBvc2l0aW9uIGp1bXApIHNob3J0ZW4gdGhlIGVmZmVjdGl2ZSBwYXR0ZXJuIGxlbmd0'
    'aC4KICAgIGRlZiBhY3R1YWxfcGF0dGVybl9yb3dzKHNwKToKICAgICAgICBwYXQgPSBtb2QucGF0'
    'dGVybl9vcmRlcltzcF0KICAgICAgICBOQ19sb2NhbCA9IG1vZC5udW1fY2hhbm5lbHMKICAgICAg'
    'ICBwYXRfc2l6ZSA9IDY0ICogTkNfbG9jYWwgKiA0CiAgICAgICAgZm9yIHJvdyBpbiByYW5nZSg2'
    'NCk6CiAgICAgICAgICAgIGZvciBjaCBpbiByYW5nZShOQ19sb2NhbCk6CiAgICAgICAgICAgICAg'
    'ICBiYXNlID0gMTA4NCArIHBhdCpwYXRfc2l6ZSArIHJvdypOQ19sb2NhbCo0ICsgY2gqNAogICAg'
    'ICAgICAgICAgICAgbmIgPSBtb2QuZGF0YVtiYXNlOmJhc2UrNF0KICAgICAgICAgICAgICAgIGVm'
    'ZiA9IG5iWzJdICYgMHhGCiAgICAgICAgICAgICAgICBpZiBlZmYgPT0gMHhEIG9yIGVmZiA9PSAw'
    'eEI6ICAgIyBwYXR0ZXJuIGJyZWFrIG9yIHBvc2l0aW9uIGp1bXAKICAgICAgICAgICAgICAgICAg'
    'ICByZXR1cm4gcm93ICsgMQogICAgICAgIHJldHVybiA2NAoKICAgIHBhdF9yb3dzID0gW2FjdHVh'
    'bF9wYXR0ZXJuX3Jvd3Moc3ApIGZvciBzcCBpbiByYW5nZShtb2Quc29uZ19sZW5ndGgpXQogICAg'
    'cGF0X3Jvd19vZmZzZXQgPSBbMF0KICAgIGZvciByIGluIHBhdF9yb3dzOgogICAgICAgIHBhdF9y'
    'b3dfb2Zmc2V0LmFwcGVuZChwYXRfcm93X29mZnNldFstMV0gKyByKQogICAgcGF0X3N0YXJ0X3Jv'
    'dyAgPSBbMF0qbW9kLnNvbmdfbGVuZ3RoCgogICAgIyBwYXRUaWNrT2Zmc2V0W3NwXSA9IGluZGV4'
    'IGludG8gcm93U3RhcnRUaWNrIGZvciByb3cgMCBvZiBzb25nIHBvc2l0aW9uIHNwCiAgICAjIFNh'
    'bWUgYXMgcGF0X3Jvd19vZmZzZXQgc2luY2UgdGljayB0YWJsZSByb3dzID09IHNvbmcgcm93cyBh'
    'ZnRlciBEMDAgZml4LgogICAgcGF0X3RpY2tfb2Zmc2V0ID0gcGF0X3Jvd19vZmZzZXRbOl0KCiAg'
    'ICB0b3RhbF9zb25nX3Jvd3MgPSBtb2Quc29uZ19sZW5ndGggKiA2NAogICAgbnVtX3BhdHRlcm5z'
    'ID0gbW9kLm51bV9wYXR0ZXJucwoKICAgICMg4pSA4pSAIFNhbXBsZUluZm8gZW1pc3Npb24gKHVz'
    'ZSBuZXcgYHN0YXJ0YCA9IHNhbXBsZSBpbmRleCBpbiB0aGUgY29uY2F0ZW5hdGVkIHN0cmVhbSkK'
    'ICAgIGRlZiBmbXRfc2FtcGxlaW5mbyhzKToKICAgICAgICByZXR1cm4gZiJTYW1wbGVJbmZvKHtz'
    'WydzdGFydCddfSwge3NbJ2xlbmd0aCddfSwge3NbJ2xvb3BTdGFydCddfSwge3NbJ2xvb3BMZW4n'
    'XX0sIHtzWyd2b2x1bWUnXX0sIHtzLmdldCgnYndGYWN0b3InLDEpfSwge3MuZ2V0KCdmaW5ldHVu'
    'ZScsMCl9KSIKICAgIHNpX2xpbmVzID0gW10KICAgIGZvciBpLCBzIGluIGVudW1lcmF0ZShzYW1w'
    'bGVzX2luZm9fbmV3KToKICAgICAgICBzaV9saW5lcy5hcHBlbmQoZiIgICAge2ZtdF9zYW1wbGVp'
    'bmZvKHMpfXsnLCcgaWYgaTwzMCBlbHNlICcnfSIpCiAgICBzYW1wbGVzX2luZm9fZ2xzbCA9ICJj'
    'b25zdCBTYW1wbGVJbmZvIHNhbXBsZXNbMzFdID0gU2FtcGxlSW5mb1tdKFxuIiArICJcbiIuam9p'
    'bihzaV9saW5lcykgKyAiXG4pOyIKCiAgICAjIOKUgOKUgCBjaGFubmVsUGFuIChzYW1lIGFzIGV4'
    'aXN0aW5nOiBBbWlnYSBMUlJMIHdpdGggcmVzdCBjZW50ZXJlZCkKICAgIGNoYW5fcGFuID0gWzAu'
    'MCwgMS4wLCAxLjAsIDAuMF0gKyBbMC41XSoyOAoKICAgICMg4pSA4pSAIENodW5rIGFycmF5IGRl'
    'Y2xhcmF0aW9ucwogICAgZGljdF9sZW4gICAgPSBzdW0obGVuKGMpIGZvciBjIGluIGRpY3RfY2h1'
    'bmtzKQogICAgYml0bWFwX2xlbiAgPSBzdW0obGVuKGMpIGZvciBjIGluIGJpdG1hcF9jaHVua3Mp'
    'CiAgICBpZHhfbGVuICAgICA9IHN1bShsZW4oYykgZm9yIGMgaW4gaWR4X2NodW5rcykKICAgIHJv'
    'd3NlZWtfbGVuID0gc3VtKGxlbihjKSBmb3IgYyBpbiByb3dzZWVrX2NodW5rcykKICAgIGNvZGVz'
    'X2xlbiAgID0gc3VtKGxlbihjKSBmb3IgYyBpbiBjb2Rlc19jaHVua3MpCiAgICBjYl9sZW4gICAg'
    'ICA9IHN1bShsZW4oYykgZm9yIGMgaW4gY2JfY2h1bmtzKQoKICAgIGhlYWRlciA9IGYiIiIvKiA9'
    'PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09'
    'PT09PT09PT09PT09PT09PT09CiAgIEdMU0wgTU9EIFBsYXllciB2MS4zNSAoYykgMjAyNiBPcmJs'
    'aXZpdXMKICAgM0QgU3Vycm91bmQsIFBoYXRCYXNzLCBDb21iIFJldmVyYiwgRkFULCBSVlEgc2Ft'
    'cGxlIGNvbXByZXNzaW9uLCBjb25maWd1cmFibGUgcmVzYW1wbGVyCiAgIENvbnRhY3Q6IHN1YmJh'
    'bmRAZ21haWwuY29tIG9yCiAgICAgICAgICAgIHN1YmJhbmRAcHJvdG9ubWFpbC5jb20KICAgR0lU'
    'OiAgICAgaHR0cHM6Ly9naXRodWIuY29tL21ld3phL21vZDJnbHNsCiAgIENPTU1PTiBUQUIKICAg'
    'R2VuZXJhdGVkIGZyb206IHttb2QudGl0bGV9CiAgIAogICBDb21wcmVzc2lvbjoKICAgICDigKIg'
    'UGF0dGVybnM6IGJpdG1hcCArIGRpY3Rpb25hcnkgKyAxNi1iaXQgcHJlZml4LXN1bSByb3cgc2Vl'
    'ayAoTygxKSkKICAgICDigKIgU2FtcGxlczogIDItc3RhZ2UgUlZRIMOXe2Rvd25zYW1wbGV9IEFB'
    'LWRvd25zYW1wbGVkIChLMT17SzF9LCBLMj17SzJ9KSwge0JJVFNfVE9UQUx9IGJpdHMvcGFpcgog'
    'ICAgICAgICAgICAgICAgIHJpbmctd2VpZ2h0ZWQgay1tZWFucyB0cmFpbmVkIG9uIHRoaXMgTU9E'
    'J3MgY29udGVudAogICA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09'
    'PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09ICovCgojZGVmaW5lIFVTRV9FTUJFRERF'
    'RF9EQVRBIDEKI2RlZmluZSBOVU1fUEFUVEVSTlMgICAgICB7bnVtX3BhdHRlcm5zfQojZGVmaW5l'
    'IFNPTkdfTEVOR1RIICAgICAgIHttb2Quc29uZ19sZW5ndGh9CiNkZWZpbmUgU09OR19MT09QX1BP'
    'UyAgICAgMAojZGVmaW5lIE5VTV9DSEFOTkVMUyAgICAgIHttb2QubnVtX2NoYW5uZWxzfQojZGVm'
    'aW5lIEJQTSAgICAgICAgICAgICAgIDEyNS4wCiNkZWZpbmUgU1BFRUQgICAgICAgICAgICAgNi4w'
    'CiNkZWZpbmUgVE9UQUxfU09OR19ST1dTICAge3RvdGFsX3Nvbmdfcm93c30KCi8vIOKUgOKUgCBQ'
    'YXR0ZXJuIGNydW5jaCBjb25zdGFudHMg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACiNkZWZpbmUg'
    'VE9UQUxfTk9URVMgICAgICAge3BbJ3RvdGFsX25vdGVzJ119CiNkZWZpbmUgVE9UQUxfUk9XUyAg'
    'ICAgICAge3BbJ251bV9yb3dzJ119CiNkZWZpbmUgRElDVF9OT1RFUyAgICAgICAge2xlbihwWyd1'
    'bmlxJ10pfQojZGVmaW5lIElEWF9CWVRFU19QRVIgICAgIHtwWydpZHhfYnl0ZXMnXX0KI2RlZmlu'
    'ZSBESUNUX0lOVFMgICAgICAgICB7ZGljdF9sZW59CiNkZWZpbmUgQklUTUFQX0lOVFMgICAgICAg'
    'e2JpdG1hcF9sZW59CiNkZWZpbmUgSURYX0lOVFMgICAgICAgICAge2lkeF9sZW59CiNkZWZpbmUg'
    'Uk9XU0VFS19JTlRTICAgICAge3Jvd3NlZWtfbGVufQoKLy8g4pSA4pSAIFJWUSBzYW1wbGUgY29u'
    'c3RhbnRzIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAovLyBTYW1wbGVz'
    'IGFyZSBhbnRpLWFsaWFzIGRvd25zYW1wbGVkIHBlci1zYW1wbGUgKERTPTEgZm9yIEhGIHBlcmN1'
    'c3Npb24sCi8vIERTPXtkb3duc2FtcGxlfSBmb3IgbWVsb2RpYykuIFBlci1zYW1wbGUgRFMgaXMg'
    'c3RvcmVkIGluIFNhbXBsZUluZm8uYndGYWN0b3IuCi8vIHBlcmlvZFRvRnJlcSA9IDcwOTM3ODku'
    'Mi8ocGVyaW9kKjIpIOKAlCBid0ZhY3RvciBoYW5kbGVzIHBlci1zYW1wbGUgcGl0Y2guCiNkZWZp'
    'bmUgUlZRX0NPREVTX0JZVEVTICAge2xlbihwYWNrZWQpfQojZGVmaW5lIFJWUV9DQl9CWVRFUyAg'
    'ICAgIHtLMSoyICsgSzIqMn0KI2RlZmluZSBUT1RBTF9TQU1QTEVTICAgICB7dG90YWxfc2FtcGxl'
    'c30KCiNkZWZpbmUgQklUTUFQX0JZVEVTICAgICAge2xlbihwWydiaXRtYXAnXSl9CiNkZWZpbmUg'
    'SURYX0JZVEVTICAgICAgICAge2xlbihwWydpZHhfc3RyZWFtJ10pfQojZGVmaW5lIFJPV1NFRUtf'
    'QllURVMgICAgIHtsZW4ocFsncm93X3NlZWtfYnl0ZXMnXSl9CgovLyDilIDilIAgRnh4LWF3YXJl'
    'IHRpbWluZyDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAK'
    'I2RlZmluZSBUT1RBTF9USUNLUyAgICAgICB7cm93U3RhcnRUaWNrWy0xXX0KI2RlZmluZSBOVU1f'
    'U09OR19ST1dTICAgICB7bGVuKHJvd1N0YXJ0VGljayktMX0KI2RlZmluZSBUSUNLU19QRVJfU0VD'
    'ICAgICA1MC4wICAgLy8gQlBNPTEyNSBjb25zdGFudCBmb3IgMTJUSC5NT0QKCi8vIOKUgOKUgCBB'
    'dWRpbyBlZmZlY3RzIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgApjb25zdCBib29sICBlbmFibGUzRCAgICAgID0gdHJ1ZTsKY29uc3Qg'
    'Ym9vbCAgZW5hYmxlRkFUICAgICA9IHRydWU7CmNvbnN0IGl2ZWMyIHN1cnJfY2hhbm5lbHMgPSBp'
    'dmVjMigxLCA0KTsKIiIiCgogICAgIyDilIDilIAgc29uZyBtZXRhZGF0YSBhcnJheXMKICAgIGNo'
    'YW5fcGFuX3N0ciAgID0gIiwgIi5qb2luKGYie3Y6LjFmfSIgZm9yIHYgaW4gY2hhbl9wYW4pCiAg'
    'ICBzb25ncG9zX3N0ciAgICA9ICIsICIuam9pbihzdHIoeCkgZm9yIHggaW4gc29uZ19wb3NpdGlv'
    'bnMpCiAgICByb3dvZmZfc3RyICAgICA9ICIsICIuam9pbihzdHIoeCkgZm9yIHggaW4gcGF0X3Jv'
    'd19vZmZzZXQpCiAgICBzdGFydHJvd19zdHIgICA9ICIsICIuam9pbihzdHIoeCkgZm9yIHggaW4g'
    'cGF0X3N0YXJ0X3JvdykKICAgIHRpY2tvZmZfc3RyICAgID0gIiwgIi5qb2luKHN0cih4KSBmb3Ig'
    'eCBpbiBwYXRfdGlja19vZmZzZXRbOi0xXSkgICMgbGVuZ3RoID0gc29uZ19sZW5ndGgKCiAgICBt'
    'ZXRhID0gZiIiIgpjb25zdCBmbG9hdCBjaGFubmVsUGFuWzMyXSA9IGZsb2F0W10oe2NoYW5fcGFu'
    'X3N0cn0pOwpjb25zdCBpbnQgICBzb25nUG9zaXRpb25zW3ttb2Quc29uZ19sZW5ndGh9XSAgID0g'
    'aW50W10oe3Nvbmdwb3Nfc3RyfSk7CmNvbnN0IGludCAgIHBhdFJvd09mZnNldFt7bW9kLnNvbmdf'
    'bGVuZ3RoKzF9XSAgICA9IGludFtdKHtyb3dvZmZfc3RyfSk7CmNvbnN0IGludCAgIHBhdFN0YXJ0'
    'Um93W3ttb2Quc29uZ19sZW5ndGh9XSAgICAgPSBpbnRbXSh7c3RhcnRyb3dfc3RyfSk7CmNvbnN0'
    'IGludCAgIHBhdFRpY2tPZmZzZXRbe21vZC5zb25nX2xlbmd0aH1dICAgPSBpbnRbXSh7dGlja29m'
    'Zl9zdHJ9KTsKIiIiCgogICAgIyDilIDilIAgRGF0YSBhcnJheXMgKGl2ZWM0IGNodW5rcykKICAg'
    'IGRhdGFfYXJyYXlzID0gWyJcbi8vIOKUgOKUgCBQYXR0ZXJuIGRpY3Rpb25hcnkgKHVuaXF1ZSA0'
    'LWJ5dGUgbm90ZXMsIE1TQi1maXJzdCBwZXIgaW50KSDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIBcbiJdCiAgICBkYXRhX2FycmF5cy5hcHBlbmQoZW1pdF9pdmVjNF9hcnJheSgicGF0'
    'RGljdCIsIGRpY3RfY2h1bmtzKSkKICAgIGRhdGFfYXJyYXlzLmFwcGVuZCgiXG4vLyDilIDilIAg'
    'UGF0dGVybiBiaXRtYXAgKDEgYml0L25vdGUsIExTQi1maXJzdCB3aXRoaW4gYnl0ZSkg4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAXG4i'
    'KQogICAgZGF0YV9hcnJheXMuYXBwZW5kKGVtaXRfaXZlYzRfYXJyYXkoInBhdEJpdG1hcCIsIGJp'
    'dG1hcF9jaHVua3MpKQogICAgZGF0YV9hcnJheXMuYXBwZW5kKCJcbi8vIOKUgOKUgCBJbmRleCBz'
    'dHJlYW0gKCVzIGJ5dGVzIHBlciBub24tZW1wdHkgbm90ZSkg4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    'XG4iICUgcFsnaWR4X2J5dGVzJ10pCiAgICBkYXRhX2FycmF5cy5hcHBlbmQoZW1pdF9pdmVjNF9h'
    'cnJheSgicGF0SWR4IiwgaWR4X2NodW5rcykpCiAgICBkYXRhX2FycmF5cy5hcHBlbmQoIlxuLy8g'
    '4pSA4pSAIFJvdyBzZWVrIHRhYmxlICgxNi1iaXQgTEUgcHJlZml4IHN1bXMsIE8oMSkgbG9va3Vw'
    'KSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBc'
    'biIpCiAgICBkYXRhX2FycmF5cy5hcHBlbmQoZW1pdF9pdmVjNF9hcnJheSgicGF0Um93U2VlayIs'
    'IHJvd3NlZWtfY2h1bmtzKSkKICAgIGRhdGFfYXJyYXlzLmFwcGVuZCgiXG4vLyDilIDilIAgVlEg'
    'Y29kZXMgKHBhY2tlZCBiaXQgc3RyZWFtKSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIBcbiIpCiAgICBkYXRhX2FycmF5cy5h'
    'cHBlbmQoZW1pdF9pdmVjNF9hcnJheSgidnFDb2RlcyIsIGNvZGVzX2NodW5rcykpCiAgICBkYXRh'
    'X2FycmF5cy5hcHBlbmQoZiJcbi8vIOKUgOKUgCBWUSBjb2RlYm9vayAoe0t9IGVudHJpZXMgw5cg'
    'MiBzYW1wbGVzLCBzaWduZWQgOC1iaXQgYXMgdW5zaWduZWQpIOKUgOKUgFxuIikKICAgIGRhdGFf'
    'YXJyYXlzLmFwcGVuZChlbWl0X2l2ZWM0X2FycmF5KCJ2cUNvZGVib29rIiwgY2JfY2h1bmtzKSkK'
    'ICAgIGRhdGFfYXJyYXlzLmFwcGVuZCgiXG4vLyDilIDilIAgUGVyLXJvdyBjdW11bGF0aXZlIHRp'
    'Y2sgdGFibGUgKDE2LWJpdCBMRSwgRnh4LWF3YXJlKSDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIBcbiIpCiAgICBkYXRhX2FycmF5cy5hcHBlbmQoZW1pdF9p'
    'dmVjNF9hcnJheSgicm93U3RhcnRUaWNrIiwgdGlja19jaHVua3MpKQoKICAgICMg4pSA4pSAIFNh'
    'bXBsZUluZm8gJiBwZXJpb2RUYWJsZQogICAgdGFibGVzID0gZiIiIgovLyDilIDilIAgU2FtcGxl'
    'IG1ldGFkYXRhIChzdGFydCA9IHNhbXBsZSBpbmRleCBpbiBwYWNrZWQgMy1iaXQgc3RyZWFtKSDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIAKc3RydWN0IFNhbXBsZUluZm8ge3sKICAgIGludCBz'
    'dGFydCwgbGVuZ3RoLCBsb29wU3RhcnQsIGxvb3BMZW4sIHZvbHVtZSwgYndGYWN0b3IsIGZpbmV0'
    'dW5lOwp9fTsKe3NhbXBsZXNfaW5mb19nbHNsfQoKLy8gUHJvVHJhY2tlciBwZXJpb2QgdGFibGUg'
    'KEMtMSB0byBCLTMpCmNvbnN0IGludCBwZXJpb2RUYWJsZVszN10gPSBpbnRbXSgKICAgIDg1Niw4'
    'MDgsNzYyLDcyMCw2NzgsNjQwLDYwNCw1NzAsNTM4LDUwOCw0ODAsNDUzLAogICAgNDI4LDQwNCwz'
    'ODEsMzYwLDMzOSwzMjAsMzAyLDI4NSwyNjksMjU0LDI0MCwyMjYsCiAgICAyMTQsMjAyLDE5MCwx'
    'ODAsMTcwLDE2MCwxNTEsMTQzLDEzNSwxMjcsMTIwLDExMywwCik7CgovLyBQcm9UcmFja2VyIDMy'
    'LWVudHJ5IHNpbmUgdGFibGUgZm9yIHZpYnJhdG8gKExVVCwga2VwdCBnbG9iYWwgc28gaXQgZG9l'
    'c24ndAovLyBjb25zdW1lIHBlci1jYWxsIHByaXZhdGUvc3RhY2sgc3RvcmFnZSBpbiBnZXRDaGFu'
    'bmVsT3V0cHV0KS4KY29uc3QgZmxvYXQgdmliVGFiWzMyXSA9IGZsb2F0W10oCiAgICAgIDAuMCwg'
    'IDI0LjAsICA0OS4wLCAgNzQuMCwgIDk3LjAsIDEyMC4wLCAxNDEuMCwgMTYxLjAsCiAgICAxODAu'
    'MCwgMTk3LjAsIDIxMi4wLCAyMjQuMCwgMjM1LjAsIDI0NC4wLCAyNTAuMCwgMjUzLjAsCiAgICAy'
    'NTUuMCwgMjUzLjAsIDI1MC4wLCAyNDQuMCwgMjM1LjAsIDIyNC4wLCAyMTIuMCwgMTk3LjAsCiAg'
    'ICAxODAuMCwgMTYxLjAsIDE0MS4wLCAxMjAuMCwgIDk3LjAsICA3NC4wLCAgNDkuMCwgIDI0LjAK'
    'KTsKCi8vIEM0IHNwZWVkcyBmb3IgZWFjaCBmaW5ldHVuZSB2YWx1ZSAobWlrSVQvUFQgc3BlYyku'
    'ICBJbmRleCAwLi43ID0gcG9zaXRpdmUKLy8gZmluZXR1bmUgKHNsaWdodGx5IGhpZ2hlciBwaXRj'
    'aCksIGluZGV4IDguLjE1ID0gbmVnYXRpdmUgZmluZXR1bmUgKGxvd2VyKS4KLy8gSW4gc2FtcGxl'
    'IGRhdGEgd2Ugc3RvcmUgZmluZXR1bmUgYXMgYSBTSUdORUQgLTguLjcgaW50IOKAlCBjb252ZXJ0'
    'IHZpYSAmMHhGLgpjb25zdCBmbG9hdCBjNHNwZWVkc1sxNl0gPSBmbG9hdFtdKAogICAgODM2My4w'
    'LCA4NDEzLjAsIDg0NjMuMCwgODUyOS4wLCA4NTgxLjAsIDg2NTEuMCwgODcyMy4wLCA4NzU3LjAs'
    'CiAgICA3ODk1LjAsIDc5NDEuMCwgNzk4NS4wLCA4MDQ2LjAsIDgxMDcuMCwgODE2OS4wLCA4MjMy'
    'LjAsIDgyODAuMAopOwpmbG9hdCBwZXJpb2RUb0ZyZXEoaW50IHBlcmlvZCkge3sKICAgIC8vIERl'
    'ZmF1bHQgKGZpbmV0dW5lPTApOiA3MDkzNzg5LjIgLyAocGVyaW9kIMOXIDIpIOKJiCAzNTQ2ODk0'
    'LjYvcGVyaW9kLiAgVXNlCiAgICAvLyBwZXJpb2RUb0ZyZXFGdCBiZWxvdyB3aGVuIGZpbmV0dW5l'
    'IG1hdHRlcnMuCiAgICByZXR1cm4gcGVyaW9kID4gMCA/IDcwOTM3ODkuMiAvIChmbG9hdChwZXJp'
    'b2QpICogMi4wKSA6IDAuMDsKfX0KZmxvYXQgcGVyaW9kVG9GcmVxRnQoaW50IHBlcmlvZCwgaW50'
    'IGZpbmV0dW5lKSB7ewogICAgLy8gKGM0ICogNDI4KSAvIHBlcmlvZCDigJQgbWF0Y2hlcyBIVE1M'
    'J3MgcGl0Y2ggdGFibGUgZXhhY3RseS4KICAgIGlmIChwZXJpb2QgPD0gMCkgcmV0dXJuIDAuMDsK'
    'ICAgIGludCBpZHggPSBmaW5ldHVuZSAmIDB4RjsgIC8vIC0xIChzaWduZWQpIOKGkiAweEYsIGV0'
    'Yy4KICAgIHJldHVybiAoYzRzcGVlZHNbaWR4XSAqIDQyOC4wKSAvIGZsb2F0KHBlcmlvZCk7Cn19'
    'CiIiIgoKICAgICMg4pSA4pSAIEZldGNoIGhlbHBlcnMgKGNodW5rIGRpc3BhdGNoZXJzIGZvciBl'
    'YWNoIGFycmF5KQogICAgZGVmIGNodW5rX2Rpc3BhdGNoKG5hbWUsIG51bV9jaHVua3MsIHZhcj0n'
    'aScpOgogICAgICAgIGlmIG51bV9jaHVua3MgPT0gMToKICAgICAgICAgICAgcmV0dXJuIGYiICAg'
    'IHJldHVybiB7bmFtZX0wW3t2YXJ9Pj4yXTsiCiAgICAgICAgbGluZXMgPSBbZiIgICAgaXZlYzQg'
    'diA9IGl2ZWM0KDApOyJdCiAgICAgICAgbGluZXMuYXBwZW5kKGYiICAgIGlmIChjaHVua0lkeCA9'
    'PSAwKSB2ID0ge25hbWV9MFt7dmFyfT4+Ml07IikKICAgICAgICBmb3IgayBpbiByYW5nZSgxLCBu'
    'dW1fY2h1bmtzKToKICAgICAgICAgICAgbGluZXMuYXBwZW5kKGYiICAgIGVsc2UgaWYgKGNodW5r'
    'SWR4ID09IHtrfSkgdiA9IHtuYW1lfXtrfVt7dmFyfT4+Ml07IikKICAgICAgICBsaW5lcy5hcHBl'
    'bmQoZiIgICAgcmV0dXJuIHY7IikKICAgICAgICByZXR1cm4gIlxuIi5qb2luKGxpbmVzKQoKICAg'
    'IGRlZiBpdmVjNF9zZWxlY3QodmFyPSdpJyk6CiAgICAgICAgcmV0dXJuIGYiIiIgICAgaW50IGNp'
    'ID0ge3Zhcn0gJiAzOwogICAgcmV0dXJuIGNpPT0wID8gdi54IDogY2k9PTEgPyB2LnkgOiBjaT09'
    'MiA/IHYueiA6IHYudzsiIiIKCiAgICBmZXRjaGVycyA9IGYiIiIKLy8g4pWQ4pWQ4pWQIENodW5r'
    'ZWQgaXZlYzQgZmV0Y2hlcnMg4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQCgovLyBGZXRj'
    'aCBhIGJ5dGUgZnJvbSBhbnkgY2h1bmtlZCBieXRlIGFycmF5IChNU0ItZmlyc3Qgd2l0aGluIGVh'
    'Y2ggaW50MzIpLgovLyBFYWNoIGl2ZWM0IGhvbGRzIDE2IGJ5dGVzOiAueCA9IGJ5dGVzIDAtMywg'
    'LnkgPSA0LTcsIC56ID0gOC0xMSwgLncgPSAxMi0xNQovLyBXaXRoaW4gZWFjaCBpbnQ6IGJ5dGUg'
    'MCA9IE1TQiwgYnl0ZSAzID0gTFNCLgoKaW50IF9leHRyYWN0Qnl0ZShpdmVjNCB2LCBpbnQgYnl0'
    'ZUluSXZlYzQpIHt7CiAgICBpbnQgaW50SWR4ID0gYnl0ZUluSXZlYzQgPj4gMjsKICAgIGludCBi'
    'eXRlSW5JbnQgPSBieXRlSW5JdmVjNCAmIDM7CiAgICBpbnQgcGFja2VkID0gaW50SWR4PT0wID8g'
    'di54IDogaW50SWR4PT0xID8gdi55IDogaW50SWR4PT0yID8gdi56IDogdi53OwogICAgaW50IHNo'
    'aWZ0ID0gMjQgLSBieXRlSW5JbnQgKiA4OwogICAgcmV0dXJuIChwYWNrZWQgPj4gc2hpZnQpICYg'
    'MHhGRjsKfX0KCi8vIOKUgOKUgCBEaWN0aW9uYXJ5IGJ5dGUgZmV0Y2ggKGJ5dGVJZHggaW4gWzAs'
    'IERJQ1RfTk9URVMqNCkpIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgAppbnQgZmV0Y2hEaWN0Qnl0ZShpbnQgYnl0ZUlkeCkge3sKICAgIGludCBp'
    'dmVjNElkeCA9IGJ5dGVJZHggPj4gNDsKICAgIGludCBieXRlSW5JdmVjNCA9IGJ5dGVJZHggJiAx'
    'NTsKICAgIGl2ZWM0IHYgPSBwYXREaWN0MFtpdmVjNElkeF07CiAgICByZXR1cm4gX2V4dHJhY3RC'
    'eXRlKHYsIGJ5dGVJbkl2ZWM0KTsKfX0KCi8vIOKUgOKUgCBCaXRtYXAgYnl0ZSBmZXRjaCDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKaW50IGZldGNoQml0bWFw'
    'Qnl0ZShpbnQgYnl0ZUlkeCkge3sKICAgIGludCBpdmVjNElkeCA9IGJ5dGVJZHggPj4gNDsKICAg'
    'IGludCBieXRlSW5JdmVjNCA9IGJ5dGVJZHggJiAxNTsKICAgIGl2ZWM0IHYgPSBwYXRCaXRtYXAw'
    'W2l2ZWM0SWR4XTsKICAgIHJldHVybiBfZXh0cmFjdEJ5dGUodiwgYnl0ZUluSXZlYzQpOwp9fQoK'
    'Ly8g4pSA4pSAIEluZGV4IHN0cmVhbSBieXRlIGZldGNoIChjaHVua2VkIGlmIG5lZWRlZCkg4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSACmludCBmZXRjaElkeEJ5dGUoaW50IGJ5dGVJZHgpIHt7CiAgICBp'
    'bnQgaXZlYzRJZHggPSBieXRlSWR4ID4+IDQ7CiAgICBpbnQgYnl0ZUluSXZlYzQgPSBieXRlSWR4'
    'ICYgMTU7CiAgICBpbnQgY2h1bmtJZHggPSBpdmVjNElkeCAvIDUxMjsKICAgIGludCBsb2NhbEl2'
    'ZWM0ID0gaXZlYzRJZHggJSA1MTI7CiAgICBpdmVjNCB2ID0gaXZlYzQoMCk7CntjaHIoMTApLmpv'
    'aW4oZicgICAgeyJpZiIgaWYgaz09MCBlbHNlICJlbHNlIGlmIn0gKGNodW5rSWR4ID09IHtrfSkg'
    'diA9IHBhdElkeHtrfVtsb2NhbEl2ZWM0XTsnIGZvciBrIGluIHJhbmdlKGxlbihpZHhfY2h1bmtz'
    'KSkpfQogICAgcmV0dXJuIF9leHRyYWN0Qnl0ZSh2LCBieXRlSW5JdmVjNCk7Cn19CgovLyDilIDi'
    'lIAgUm93LXNlZWsgbmliYmxlIGJ5dGUgZmV0Y2gg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACmludCBmZXRjaFJv'
    'd1NlZWtCeXRlKGludCBieXRlSWR4KSB7ewogICAgaW50IGl2ZWM0SWR4ID0gYnl0ZUlkeCA+PiA0'
    'OwogICAgaW50IGJ5dGVJbkl2ZWM0ID0gYnl0ZUlkeCAmIDE1OwogICAgaXZlYzQgdiA9IHBhdFJv'
    'd1NlZWswW2l2ZWM0SWR4XTsKICAgIHJldHVybiBfZXh0cmFjdEJ5dGUodiwgYnl0ZUluSXZlYzQp'
    'Owp9fQoKLy8g4pSA4pSAIFZRIGNvZGUgc3RyZWFtIGJ5dGUgZmV0Y2ggKGNodW5rZWQpIOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAppbnQgZmV0Y2hDb2Rlc0J5'
    'dGUoaW50IGJ5dGVJZHgpIHt7CiAgICBpbnQgaXZlYzRJZHggPSBieXRlSWR4ID4+IDQ7CiAgICBp'
    'bnQgYnl0ZUluSXZlYzQgPSBieXRlSWR4ICYgMTU7CiAgICBpbnQgY2h1bmtJZHggPSBpdmVjNElk'
    'eCAvIDUxMjsKICAgIGludCBsb2NhbEl2ZWM0ID0gaXZlYzRJZHggJSA1MTI7CiAgICBpdmVjNCB2'
    'ID0gaXZlYzQoMCk7CntjaHIoMTApLmpvaW4oZicgICAgeyJpZiIgaWYgaz09MCBlbHNlICJlbHNl'
    'IGlmIn0gKGNodW5rSWR4ID09IHtrfSkgdiA9IHZxQ29kZXN7a31bbG9jYWxJdmVjNF07JyBmb3Ig'
    'ayBpbiByYW5nZShsZW4oY29kZXNfY2h1bmtzKSkpfQogICAgcmV0dXJuIF9leHRyYWN0Qnl0ZSh2'
    'LCBieXRlSW5JdmVjNCk7Cn19CgovLyDilIDilIAgVlEgY29kZWJvb2sgYnl0ZSBmZXRjaCAoc21h'
    'bGwsIGZpdHMgaW4gMSBjaHVuayB1c3VhbGx5KSDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIAKaW50IGZldGNoQ29kZWJvb2tCeXRlKGludCBieXRlSWR4KSB7ewogICAg'
    'aW50IGl2ZWM0SWR4ID0gYnl0ZUlkeCA+PiA0OwogICAgaW50IGJ5dGVJbkl2ZWM0ID0gYnl0ZUlk'
    'eCAmIDE1OwogICAgaW50IGNodW5rSWR4ID0gaXZlYzRJZHggLyA1MTI7CiAgICBpbnQgbG9jYWxJ'
    'dmVjNCA9IGl2ZWM0SWR4ICUgNTEyOwogICAgaXZlYzQgdiA9IGl2ZWM0KDApOwp7Y2hyKDEwKS5q'
    'b2luKGYnICAgIHsiaWYiIGlmIGs9PTAgZWxzZSAiZWxzZSBpZiJ9IChjaHVua0lkeCA9PSB7a30p'
    'IHYgPSB2cUNvZGVib29re2t9W2xvY2FsSXZlYzRdOycgZm9yIGsgaW4gcmFuZ2UobGVuKGNiX2No'
    'dW5rcykpKX0KICAgIHJldHVybiBfZXh0cmFjdEJ5dGUodiwgYnl0ZUluSXZlYzQpOwp9fQoiIiIK'
    'CiAgICAjIOKUgOKUgCBwb3Bjb3VudCBoZWxwZXIgKDQtYml0IG5pYmJsZSkKICAgICMg4pSA4pSA'
    'IGdldE5vdGU6IGJpdG1hcCArIGRpY3QgbG9va3VwIHdpdGggTygxKSByb3cgc2VlayArIHByZWZp'
    'eCBwb3Bjb3VudAogICAgZGVjb2RlcnMgPSAiIiIKLy8g4pWQ4pWQ4pWQIFBhdHRlcm4gZGVjb2Rl'
    'cjogYml0bWFwICsgZGljdGlvbmFyeSArIHJvdyBzZWVrIOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkAoKc3RydWN0IE5v'
    'dGUgeyBpbnQgaW5zdHJ1bWVudCwgcGVyaW9kLCBlZmZlY3QsIHBhcmFtOyB9OwoKLy8gUG9wY291'
    'bnQgZm9yIDQtYml0IG5pYmJsZSAoMC4uMTUg4oaSIDAuLjQpCmludCBwb3Bjb3VudDQoaW50IHgp'
    'IHsKICAgIHggPSAoeCAmIDB4NSkgKyAoKHggPj4gMSkgJiAweDUpOwogICAgcmV0dXJuICh4ICYg'
    'MHgzKSArICgoeCA+PiAyKSAmIDB4Myk7Cn0KCi8vIFJlY29uc3RydWN0IGN1bXVsYXRpdmUgbm9u'
    'LWVtcHR5IGNvdW50IHVwIHRvIHN0YXJ0IG9mIGByb3dgIOKAlCBPKDEpLgovLyBSb3cgc2VlayB0'
    'YWJsZSBob2xkcyAxNi1iaXQgTEUgcHJlZml4IHN1bXM6IDIgYnl0ZXMgcGVyIHJvdy4KaW50IHJv'
    'd1NlZWtDdW0oaW50IHRhcmdldFJvdykgewogICAgaW50IGJ5dGVJZHggPSB0YXJnZXRSb3cgKiAy'
    'OwogICAgaW50IGxvID0gZmV0Y2hSb3dTZWVrQnl0ZShieXRlSWR4KTsKICAgIGludCBoaSA9IGZl'
    'dGNoUm93U2Vla0J5dGUoYnl0ZUlkeCArIDEpOwogICAgcmV0dXJuIGxvIHwgKGhpIDw8IDgpOwp9'
    'CgpOb3RlIGVtcHR5Tm90ZSgpIHsgTm90ZSBuOyBuLmluc3RydW1lbnQ9MDsgbi5wZXJpb2Q9MDsg'
    'bi5lZmZlY3Q9MDsgbi5wYXJhbT0wOyByZXR1cm4gbjsgfQoKTm90ZSBnZXROb3RlKGludCBzb25n'
    'UG9zLCBpbnQgcm93LCBpbnQgY2hhbm5lbCkgewogICAgaW50IHBhdCA9IHNvbmdQb3NpdGlvbnNb'
    'c29uZ1Bvc107CiAgICBpbnQgcm93R2xvYmFsID0gcGF0ICogNjQgKyByb3c7CiAgICBpbnQgbm90'
    'ZUlkeCAgID0gcm93R2xvYmFsICogNCArIGNoYW5uZWw7CgogICAgLy8gMSkgQml0bWFwIGNoZWNr'
    'CiAgICBpbnQgYm1CeXRlID0gZmV0Y2hCaXRtYXBCeXRlKG5vdGVJZHggPj4gMyk7CiAgICBpbnQg'
    'Yml0ID0gKGJtQnl0ZSA+PiAobm90ZUlkeCAmIDcpKSAmIDE7CiAgICBpZiAoYml0ID09IDApIHJl'
    'dHVybiBlbXB0eU5vdGUoKTsKCiAgICAvLyAyKSBDb3VudCBub24tZW1wdHkgbm90ZXMgYmVmb3Jl'
    'IHRoaXMgcG9zaXRpb24KICAgIC8vICAgID0gY3VtdWxhdGl2ZSB1cCB0byByb3dHbG9iYWwgKyBw'
    'b3Bjb3VudCBvZiBiaXRtYXAgbmliYmxlIHdpdGhpbiB0aGlzIHJvdwogICAgaW50IHJhbmsgPSBy'
    'b3dTZWVrQ3VtKHJvd0dsb2JhbCk7CiAgICAvLyBUaGlzIHJvdydzIDQgYml0cyBzcGFuIGNoYW5u'
    'ZWxzIDAuLjMg4oaSIHRha2UgY2hhbm5lbHMgWzAuLmNoYW5uZWwtMV0KICAgIGludCByb3dCaXRt'
    'YXBTdGFydCA9IHJvd0dsb2JhbCAqIDQ7CiAgICAvLyBUaGUgNCBiaXRzIG9mIHRoaXMgcm93IG1h'
    'eSBzcGFuIDEgYnl0ZSAoaWYgYWxpZ25lZCkgb3IgMi4KICAgIGludCBieXRlMElkeCA9IHJvd0Jp'
    'dG1hcFN0YXJ0ID4+IDM7CiAgICBpbnQgc2hpZnQgICAgPSByb3dCaXRtYXBTdGFydCAmIDc7CiAg'
    'ICBpbnQgYnl0ZTAgPSBmZXRjaEJpdG1hcEJ5dGUoYnl0ZTBJZHgpOwogICAgaW50IGJ5dGUxID0g'
    'ZmV0Y2hCaXRtYXBCeXRlKGJ5dGUwSWR4ICsgMSk7CiAgICBpbnQgcm93Qml0cyA9ICgoYnl0ZTAg'
    'Pj4gc2hpZnQpIHwgKGJ5dGUxIDw8ICg4IC0gc2hpZnQpKSkgJiAweEY7CiAgICBpbnQgbWFzayA9'
    'ICgxIDw8IGNoYW5uZWwpIC0gMTsKICAgIHJhbmsgKz0gcG9wY291bnQ0KHJvd0JpdHMgJiBtYXNr'
    'KTsKCiAgICAvLyAzKSBMb29rIHVwIGluZGV4IGFuZCBmZXRjaCBub3RlIGZyb20gZGljdGlvbmFy'
    'eQogICAgaW50IGRpY3RJZHg7CiNpZiBJRFhfQllURVNfUEVSID09IDEKICAgIGRpY3RJZHggPSBm'
    'ZXRjaElkeEJ5dGUocmFuayk7CiNlbHNlCiAgICBpbnQgbG8gPSBmZXRjaElkeEJ5dGUocmFuayAq'
    'IDIpOwogICAgaW50IGhpID0gZmV0Y2hJZHhCeXRlKHJhbmsgKiAyICsgMSk7CiAgICBkaWN0SWR4'
    'ID0gbG8gfCAoaGkgPDwgOCk7CiNlbmRpZgogICAgaW50IGIwID0gZmV0Y2hEaWN0Qnl0ZShkaWN0'
    'SWR4ICogNCArIDApOwogICAgaW50IGIxID0gZmV0Y2hEaWN0Qnl0ZShkaWN0SWR4ICogNCArIDEp'
    'OwogICAgaW50IGIyID0gZmV0Y2hEaWN0Qnl0ZShkaWN0SWR4ICogNCArIDIpOwogICAgaW50IGIz'
    'ID0gZmV0Y2hEaWN0Qnl0ZShkaWN0SWR4ICogNCArIDMpOwoKICAgIE5vdGUgbjsKICAgIG4uaW5z'
    'dHJ1bWVudCA9IChiMCAmIDB4RjApIHwgKChiMiA+PiA0KSAmIDB4MEYpOwogICAgbi5wZXJpb2Qg'
    'ICAgID0gKChiMCAmIDB4MEYpIDw8IDgpIHwgYjE7CiAgICBuLmVmZmVjdCAgICAgPSBiMiAmIDB4'
    'MEY7CiAgICBuLnBhcmFtICAgICAgPSBiMzsKICAgIHJldHVybiBuOwp9CgoiIiIKCiAgICAjIFNh'
    'bXBsZSBkZWNvZGVyOiBmLXN0cmluZyBmb3IgI2RlZmluZXMgKG5lZWQgUHl0aG9uIHZhcnMpLCBw'
    'bGFpbiBzdHJpbmcgZm9yIGZ1bmN0aW9uIGJvZGllcwogICAgX3N0YWdlX2xhYmVsID0gIjEtc3Rh'
    'Z2UgUlZRIChubyBzdGFnZSAyKSIgaWYgbm9fcnZxMiBlbHNlICIyLXN0YWdlIFJWUSIKICAgIF9w'
    'YWNrZm10ICAgICA9IGYie0JJVFMxfS1iaXQgY29kZTEgb25seSIgaWYgbm9fcnZxMiBlbHNlIGYi'
    'W3tCSVRTMX0tYml0IGNvZGUxXVt7QklUUzJ9LWJpdCBjb2RlMl0iCiAgICBkZWNvZGVycyArPSAo'
    'CiAgICAgICAgZiIvLyDilZDilZDilZAgU2FtcGxlIGRlY29kZXI6IHtfc3RhZ2VfbGFiZWx9IMOX'
    'e2Rvd25zYW1wbGV9IEFBLWRvd25zYW1wbGVkIChwZXItc2FtcGxlIERTKSDilZDilZBcbiIKICAg'
    'ICAgICBmIi8vIHtCSVRTX1RPVEFMfS1iaXQgY29kZXMgcGFja2VkIExTQi1maXJzdDoge19wYWNr'
    'Zm10fVxuIgogICAgICAgIGYiLy8gcGVyaW9kVG9GcmVxID0gNzA5Mzc4OS4yLyhwZXJpb2QqMikg'
    '4oCUIHBlci1zYW1wbGUgRFMgdmlhIFNhbXBsZUluZm8uYndGYWN0b3JcbiIKICAgICAgICBmIiNk'
    'ZWZpbmUgUlZRX0JJVFMgICAgIHtCSVRTX1RPVEFMfVxuIgogICAgICAgIGYiI2RlZmluZSBSVlFf'
    'QklUU18xICAge0JJVFMxfVxuIgogICAgICAgIGYiI2RlZmluZSBSVlFfSzEgICAgICAge0sxfVxu'
    'IgogICAgICAgIGYiI2RlZmluZSBSVlFfSzIgICAgICAge0syfVxuIgogICAgICAgIGYiI2RlZmlu'
    'ZSBSVlFfVkVDX0RJTSAge3ZlY19kaW19XG4iCiAgICAgICAgZiIjZGVmaW5lIFJWUV9DQjJfQllU'
    'RSAoe0sxfSAqIHt2ZWNfZGltfSlcbiIKICAgICAgICBmIiNkZWZpbmUgUlZRX01BU0sxICAgIHso'
    'MTw8QklUUzEpLTF9XG4iCiAgICAgICAgZiIjZGVmaW5lIFJWUV9NQVNLMiAgICB7KDE8PEJJVFMy'
    'KS0xIGlmIEJJVFMyPjAgZWxzZSAwfVxuIgogICAgICAgICsgKGYiI2RlZmluZSBSVlFfTk9fU1RB'
    'R0UyIDFcbiIgaWYgbm9fcnZxMiBlbHNlICIiKQogICAgKQogICAgZGVjb2RlcnMgKz0gIiIiCnZv'
    'aWQgX2dldFJWUUNvZGVzKGludCB2ZWNJZHgsIG91dCBpbnQgY29kZTEsIG91dCBpbnQgY29kZTIp'
    'IHsKICAgIGludCBiaXRQb3MgID0gdmVjSWR4ICogUlZRX0JJVFM7CiAgICBpbnQgYnl0ZVBvcyA9'
    'IGJpdFBvcyA+PiAzOwogICAgaW50IHNoaWZ0ICAgPSBiaXRQb3MgJiA3OwogICAgaW50IGIwID0g'
    'ZmV0Y2hDb2Rlc0J5dGUoYnl0ZVBvcyk7CiAgICBpbnQgYjEgPSBmZXRjaENvZGVzQnl0ZShieXRl'
    'UG9zICsgMSk7CiAgICBpbnQgYjIgPSBmZXRjaENvZGVzQnl0ZShieXRlUG9zICsgMik7CiAgICBp'
    'bnQgYjMgPSBmZXRjaENvZGVzQnl0ZShieXRlUG9zICsgMyk7CiAgICBpbnQgY29tYmluZWQgPSBi'
    'MCB8IChiMSA8PCA4KSB8IChiMiA8PCAxNikgfCAoYjMgPDwgMjQpOwogICAgaW50IHJhdyA9IChj'
    'b21iaW5lZCA+PiBzaGlmdCkgJiAoKDEgPDwgUlZRX0JJVFMpIC0gMSk7CiAgICBjb2RlMSA9IHJh'
    'dyAmIFJWUV9NQVNLMTsKI2lmZGVmIFJWUV9OT19TVEFHRTIKICAgIGNvZGUyID0gMDsKI2Vsc2UK'
    'ICAgIGNvZGUyID0gKHJhdyA+PiBSVlFfQklUU18xKSAmIFJWUV9NQVNLMjsKI2VuZGlmCn0KCmZs'
    'b2F0IGdldFNhbXBsZShpbnQgc2FtcGxlSWR4KSB7CiAgICBpZiAoc2FtcGxlSWR4IDwgMCB8fCBz'
    'YW1wbGVJZHggPj0gVE9UQUxfU0FNUExFUykgcmV0dXJuIDAuMDsKICAgIGludCB2ZWNJZHggPSBz'
    'YW1wbGVJZHggLyBSVlFfVkVDX0RJTTsKICAgIGludCBsYW5lICAgPSBzYW1wbGVJZHggLSB2ZWNJ'
    'ZHggKiBSVlFfVkVDX0RJTTsKICAgIC8vIElubGluZSBSVlEgZGVjb2RlIChhdm9pZHMgb3V0LXBh'
    'cmFtZXRlciBzdGFjayBhbGxvY2F0aW9uKQogICAgaW50IF9icCA9IHZlY0lkeCAqIFJWUV9CSVRT'
    'LCBfYnkgPSBfYnAgPj4gMywgX3NoID0gX2JwICYgNzsKICAgIGludCBfcmF3ID0gKGZldGNoQ29k'
    'ZXNCeXRlKF9ieSkgfCAoZmV0Y2hDb2Rlc0J5dGUoX2J5KzEpPDw4KSB8CiAgICAgICAgICAgICAg'
    'ICAoZmV0Y2hDb2Rlc0J5dGUoX2J5KzIpPDwxNikgfCAoZmV0Y2hDb2Rlc0J5dGUoX2J5KzMpPDwy'
    'NCkpOwogICAgX3JhdyA9IChfcmF3ID4+IF9zaCkgJiAoKDEgPDwgUlZRX0JJVFMpIC0gMSk7CiAg'
    'ICBpbnQgY29kZTEgPSBfcmF3ICYgUlZRX01BU0sxOwogICAgaW50IHViMSA9IGZldGNoQ29kZWJv'
    'b2tCeXRlKGNvZGUxICogUlZRX1ZFQ19ESU0gKyBsYW5lKTsKICAgIGludCBzMSAgPSB1YjEgPCAx'
    'MjggPyB1YjEgOiB1YjEgLSAyNTY7CiNpZmRlZiBSVlFfTk9fU1RBR0UyCiAgICByZXR1cm4gZmxv'
    'YXQoczEpIC8gMTI4LjA7CiNlbHNlCiAgICBpbnQgY29kZTIgPSAoX3JhdyA+PiBSVlFfQklUU18x'
    'KSAmIFJWUV9NQVNLMjsKICAgIGludCB1YjIgPSBmZXRjaENvZGVib29rQnl0ZShSVlFfQ0IyX0JZ'
    'VEUgKyBjb2RlMiAqIFJWUV9WRUNfRElNICsgbGFuZSk7CiAgICBpbnQgczIgID0gdWIyIDwgMTI4'
    'ID8gdWIyIDogdWIyIC0gMjU2OwogICAgcmV0dXJuIGZsb2F0KHMxICsgczIpIC8gMTI4LjA7CiNl'
    'bmRpZgp9CgovLyDilIDilIAgUG9zaXRpb24gY2FsY3VsYXRpb24gKEZ4eC1hd2FyZSB2aWEgcm93'
    'U3RhcnRUaWNrKSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIAKc3RydWN0IFBvc2l0aW9uIHsgaW50IHNvbmdQb3MsIHBhdHRlcm4sIHJv'
    'dzsgZmxvYXQgdGljaywgcm93VGltZTsgfTsKCi8vIEZldGNoIDE2LWJpdCBMRSB2YWx1ZSBhdCBy'
    'b3cgaW5kZXggaW50byByb3dTdGFydFRpY2sKaW50IGZldGNoVGljayhpbnQgcm93SWR4KSB7CiAg'
    'ICBpbnQgYnl0ZUlkeCA9IHJvd0lkeCAqIDI7CiAgICBpbnQgY2h1bmtJZHggID0gYnl0ZUlkeCA+'
    'PiA2OwogICAgaW50IGJ5dGVJbjE2ICA9IGJ5dGVJZHggJiA2MzsKICAgIGludCBsbyA9IF9leHRy'
    'YWN0Qnl0ZShyb3dTdGFydFRpY2swWyhjaHVua0lkeDw8MikrKGJ5dGVJbjE2Pj40KV0sIGJ5dGVJ'
    'bjE2ICYgMTUpOwogICAgLy8gbmV4dCBieXRlCiAgICBpbnQgYnl0ZUlkeDIgPSBieXRlSWR4ICsg'
    'MTsKICAgIGludCBjaHVua0lkeDIgPSBieXRlSWR4MiA+PiA2OwogICAgaW50IGJ5dGVJbjE2XzIg'
    'PSBieXRlSWR4MiAmIDYzOwogICAgaW50IGhpID0gX2V4dHJhY3RCeXRlKHJvd1N0YXJ0VGljazBb'
    'KGNodW5rSWR4Mjw8MikrKGJ5dGVJbjE2XzI+PjQpXSwgYnl0ZUluMTZfMiAmIDE1KTsKICAgIHJl'
    'dHVybiBsbyB8IChoaSA8PCA4KTsKfQoKUG9zaXRpb24gZ2V0UG9zaXRpb24oZmxvYXQgdGltZSkg'
    'ewogICAgUG9zaXRpb24gcG9zOwogICAgZmxvYXQgc29uZ0R1cmF0aW9uID0gZmxvYXQoVE9UQUxf'
    'VElDS1MpIC8gVElDS1NfUEVSX1NFQzsKICAgIGZsb2F0IGxvb3BlZFRpbWUgPSBtb2QodGltZSwg'
    'c29uZ0R1cmF0aW9uKTsKICAgIGZsb2F0IHRvdGFsVGlja0YgPSBsb29wZWRUaW1lICogVElDS1Nf'
    'UEVSX1NFQzsKCiAgICAvLyBCaW5hcnkgc2VhcmNoIHJvd1N0YXJ0VGljayBmb3IgdGhlIGN1cnJl'
    'bnQgcm93CiAgICBpbnQgbG8gPSAwLCBoaSA9IE5VTV9TT05HX1JPV1M7CiAgICBmb3IgKGludCBf'
    'YnMgPSAwOyBfYnMgPCAxMjsgX2JzKyspIHsgIC8vIGxvZzIoMTkyMCspIOKJiCAxMQogICAgICAg'
    'IGlmIChsbyA+PSBoaSAtIDEpIGJyZWFrOwogICAgICAgIGludCBtaWQgPSAobG8gKyBoaSkgPj4g'
    'MTsKICAgICAgICBpZiAoZmxvYXQoZmV0Y2hUaWNrKG1pZCkpIDw9IHRvdGFsVGlja0YpIGxvID0g'
    'bWlkOwogICAgICAgIGVsc2UgaGkgPSBtaWQ7CiAgICB9CiAgICBpbnQgZ2xvYmFsUm93ID0gbG87'
    'CiAgICBpZiAoZ2xvYmFsUm93ID49IE5VTV9TT05HX1JPV1MpIGdsb2JhbFJvdyA9IE5VTV9TT05H'
    'X1JPV1MgLSAxOwoKICAgIC8vIEZpbmQgc29uZ1BvcyB2aWEgbGluZWFyIHNlYXJjaCBvdmVyIHBh'
    'dFRpY2tPZmZzZXQgKFNPTkdfTEVOR1RIIOKJpCAxMjgsIGZhc3QgZW5vdWdoKQogICAgaW50IHNw'
    'ID0gU09OR19MRU5HVEggLSAxOwogICAgZm9yIChpbnQgX2kgPSAwOyBfaSA8IFNPTkdfTEVOR1RI'
    'IC0gMTsgX2krKykgewogICAgICAgIGlmIChwYXRUaWNrT2Zmc2V0W19pICsgMV0gPiBnbG9iYWxS'
    'b3cpIHsgc3AgPSBfaTsgYnJlYWs7IH0KICAgIH0KICAgIHBvcy5zb25nUG9zID0gc3A7CiAgICBw'
    'b3MucGF0dGVybiA9IHNvbmdQb3NpdGlvbnNbc3BdOwogICAgcG9zLnJvdyAgICAgPSBnbG9iYWxS'
    'b3cgLSBwYXRUaWNrT2Zmc2V0W3NwXTsKCiAgICBpbnQgcm93VGljayAgICA9IGZldGNoVGljayhn'
    'bG9iYWxSb3cpOwogICAgaW50IG5leHRUaWNrICAgPSBmZXRjaFRpY2soZ2xvYmFsUm93ICsgMSk7'
    'CiAgICBpbnQgcm93U3BlZWQgICA9IG5leHRUaWNrIC0gcm93VGljazsKICAgIHBvcy50aWNrICAg'
    'ICAgID0gdG90YWxUaWNrRiAtIGZsb2F0KHJvd1RpY2spOwogICAgcG9zLnJvd1RpbWUgICAgPSBm'
    'bG9hdChyb3dTcGVlZCkgLyBUSUNLU19QRVJfU0VDOwogICAgcmV0dXJuIHBvczsKfQoKLy8gNC1w'
    'b2ludCBjdWJpYyBCLXNwbGluZSBpbnRlcnBvbGF0aW9uLgovLyBCLXNwbGluZSBpcyBBUFBST1hJ'
    'TUFUSU5HIChzbW9vdGhzIHRocm91Z2ggc2FtcGxlIHBvaW50cykgcmF0aGVyIHRoYW4KLy8gSU5U'
    'RVJQT0xBVElORyAocGFzc2luZyBleGFjdGx5IHRocm91Z2ggdGhlbSksIGdpdmluZyBpbmhlcmVu'
    'dCBsb3ctcGFzcwovLyBjaGFyYWN0ZXIgdGhhdCByZWR1Y2VzIGhpZ2gtZnJlcXVlbmN5IHF1YW50'
    'aXphdGlvbiBub2lzZS4KIiIiICsgKAogICAgICAgICAgICAjIOKUgOKUgCBMaW5lYXI6IDIgdGFw'
    'cywgUHJvVHJhY2tlci1hdXRoZW50aWMsIGNoZWFwZXN0IOKUgOKUgAogICAgICAgICAgICAnJydm'
    'bG9hdCBnZXRTYW1wbGVGKGludCBiYXNlLCBmbG9hdCBmcG9zLCBpbnQgc21wTGVuLCBpbnQgbG9v'
    'cFN0YXJ0LCBpbnQgbG9vcExlbikgewogICAgaW50IGkgPSBpbnQoZnBvcyk7CiAgICBmbG9hdCB0'
    'ID0gZnBvcyAtIGZsb2F0KGkpOwogICAgZmxvYXQgcDEgPSBnZXRTYW1wbGUoYmFzZSArIGkpOwog'
    'ICAgZmxvYXQgcDIgPSBnZXRTYW1wbGUoYmFzZSArIG1pbihpICsgMSwgc21wTGVuICsgMTUpKTsK'
    'ICAgIHJldHVybiBtaXgocDEsIHAyLCB0KTsKfScnJyBpZiByZXNhbXBsZXIgPT0gJ2xpbmVhcicg'
    'ZWxzZQogICAgICAgICAgICAjIOKUgOKUgCBMYW5jem9zLTM6IDYgdGFwcywgc2hhcnBlc3QsIGJy'
    'aWdodGVzdCDilIDilIAKICAgICAgICAgICAgJycnLy8gTGFuY3pvcy0zIHdpbmRvd2VkIHNpbmM6'
    'IHcoeCkgPSBzaW5jKM+AeCkgKiBzaW5jKM+AeC8zKSBmb3IgfHh8PDMKZmxvYXQgX2xhbmN6b3Mz'
    'KGZsb2F0IHgpIHsKICAgIGlmICh4IDwgMWUtNikgcmV0dXJuIDEuMDsKICAgIGZsb2F0IHBpeCA9'
    'IDMuMTQxNTkyNjUgKiB4OwogICAgZmxvYXQgcGl4MyA9IHBpeCAvIDMuMDsKICAgIHJldHVybiAo'
    'c2luKHBpeCkgKiBzaW4ocGl4MykpIC8gKHBpeCAqIHBpeDMpOwp9CmZsb2F0IGdldFNhbXBsZUYo'
    'aW50IGJhc2UsIGZsb2F0IGZwb3MsIGludCBzbXBMZW4sIGludCBsb29wU3RhcnQsIGludCBsb29w'
    'TGVuKSB7CiAgICBpbnQgaSAgPSBpbnQoZnBvcyk7CiAgICBmbG9hdCB0ID0gZnBvcyAtIGZsb2F0'
    'KGkpOwogICAgaW50IGltMiA9IGkgLSAyLCBpbTEgPSBpIC0gMSwgaXAxID0gaSArIDEsIGlwMiA9'
    'IGkgKyAyLCBpcDMgPSBpICsgMzsKICAgIGlmIChsb29wTGVuID4gMiAmJiBpbTIgPCBsb29wU3Rh'
    'cnQpIGltMiA9IGxvb3BTdGFydCArIGxvb3BMZW4gKyAoaW0yIC0gbG9vcFN0YXJ0KTsKICAgIGlm'
    'IChsb29wTGVuID4gMiAmJiBpbTEgPCBsb29wU3RhcnQpIGltMSA9IGxvb3BTdGFydCArIGxvb3BM'
    'ZW4gKyAoaW0xIC0gbG9vcFN0YXJ0KTsKICAgIGltMiA9IG1heCgwLCBpbTIpOyBpbTEgPSBtYXgo'
    'MCwgaW0xKTsKICAgIGlwMSA9IG1pbihpcDEsIHNtcExlbiArIDE1KTsKICAgIGlwMiA9IG1pbihp'
    'cDIsIHNtcExlbiArIDE1KTsKICAgIGlwMyA9IG1pbihpcDMsIHNtcExlbiArIDE1KTsKICAgIGZs'
    'b2F0IHcwID0gX2xhbmN6b3MzKGFicyh0ICsgMi4wKSk7CiAgICBmbG9hdCB3MSA9IF9sYW5jem9z'
    'MyhhYnModCArIDEuMCkpOwogICAgZmxvYXQgdzIgPSBfbGFuY3pvczMoYWJzKHQgICAgICApKTsK'
    'ICAgIGZsb2F0IHczID0gX2xhbmN6b3MzKGFicyh0IC0gMS4wKSk7CiAgICBmbG9hdCB3NCA9IF9s'
    'YW5jem9zMyhhYnModCAtIDIuMCkpOwogICAgZmxvYXQgdzUgPSBfbGFuY3pvczMoYWJzKHQgLSAz'
    'LjApKTsKICAgIGZsb2F0IHdzdW0gPSB3MCt3MSt3Mit3Myt3NCt3NTsKICAgIHJldHVybiAodzAq'
    'Z2V0U2FtcGxlKGJhc2UraW0yKSArIHcxKmdldFNhbXBsZShiYXNlK2ltMSkgKwogICAgICAgICAg'
    'ICB3MipnZXRTYW1wbGUoYmFzZStpICApICsgdzMqZ2V0U2FtcGxlKGJhc2UraXAxKSArCiAgICAg'
    'ICAgICAgIHc0KmdldFNhbXBsZShiYXNlK2lwMikgKyB3NSpnZXRTYW1wbGUoYmFzZStpcDMpKSAv'
    'IHdzdW07Cn0nJycgaWYgcmVzYW1wbGVyID09ICdsYW5jem9zMycgZWxzZQogICAgICAgICAgICAj'
    'IOKUgOKUgCBCLXNwbGluZSAoZGVmYXVsdCk6IDQgdGFwcywgc21vb3RoLCBnZW50bGUgTFBGIOKU'
    'gOKUgAogICAgICAgICAgICAnJydmbG9hdCBnZXRTYW1wbGVGKGludCBiYXNlLCBmbG9hdCBmcG9z'
    'LCBpbnQgc21wTGVuLCBpbnQgbG9vcFN0YXJ0LCBpbnQgbG9vcExlbikgewogICAgaW50IGkgID0g'
    'aW50KGZwb3MpOwogICAgZmxvYXQgdCA9IGZwb3MgLSBmbG9hdChpKTsKICAgIGludCBpMCA9IGkg'
    'LSAxOwogICAgaWYgKGxvb3BMZW4gPiAyICYmIGkwIDwgbG9vcFN0YXJ0KSBpMCA9IGxvb3BTdGFy'
    'dCArIGxvb3BMZW4gLSAxOwogICAgZWxzZSBpMCA9IG1heCgwLCBpMCk7CiAgICBmbG9hdCBwMCA9'
    'IGdldFNhbXBsZShiYXNlICsgaTApOwogICAgZmxvYXQgcDEgPSBnZXRTYW1wbGUoYmFzZSArIGkp'
    'OwogICAgZmxvYXQgcDIgPSBnZXRTYW1wbGUoYmFzZSArIG1pbihpICsgMSwgc21wTGVuICsgMTUp'
    'KTsKICAgIGZsb2F0IHAzID0gZ2V0U2FtcGxlKGJhc2UgKyBtaW4oaSArIDIsIHNtcExlbiArIDE1'
    'KSk7CiAgICBmbG9hdCB0MiA9IHQgKiB0OwogICAgZmxvYXQgdDMgPSB0MiAqIHQ7CiAgICBmbG9h'
    'dCB3MCA9ICgxLjAgLSB0KSAqICgxLjAgLSB0KSAqICgxLjAgLSB0KSAvIDYuMDsKICAgIGZsb2F0'
    'IHcxID0gKDMuMCAqIHQzIC0gNi4wICogdDIgKyA0LjApIC8gNi4wOwogICAgZmxvYXQgdzIgPSAo'
    'LTMuMCAqIHQzICsgMy4wICogdDIgKyAzLjAgKiB0ICsgMS4wKSAvIDYuMDsKICAgIGZsb2F0IHcz'
    'ID0gdDMgLyA2LjA7CiAgICByZXR1cm4gdzAgKiBwMCArIHcxICogcDEgKyB3MiAqIHAyICsgdzMg'
    'KiBwMzsKfScnJwogICAgICAgICkgKyAiIiIKCiIiIgoKICAgIGltcG9ydCBiYXNlNjQgYXMgX2I2'
    'NGUKICAgIGdldF9jaGFubmVsX291dHB1dCA9IF9iNjRlLmI2NGRlY29kZSgnTHk4Z2RtbGlWR0Zp'
    'SUdseklHUmxZMnhoY21Wa0lHRnpJR0VnWjJ4dlltRnNJR052Ym5OMElHWnNiMkYwV3pNeVhTQnVa'
    'V0Z5SUhSb1pTQjBiM0FnYjJZZ1EyOXRiVzl1Q2k4dklDaHlhV2RvZENCaFpuUmxjaUJ3WlhKcGIy'
    'UlVZV0pzWlNrdUlFUnZiaWQwSUhKbFpHVmpiR0Z5WlNCcGRDQm9aWEpsTGdvS0x5OGc0cFNBNHBT'
    'QTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNB'
    'NHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0'
    'cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRw'
    'U0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBT'
    'QTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBQ2k4dklGOW5ZMjlD'
    'YjJSNUlPS0FsQ0J5Wlc1a1pYSWdiMjVsSUhOaGJYQnNaU0J2WmlCaElHTm9ZVzV1Wld3Z1oybDJa'
    'VzRnUzA1UFYwNGdkSEpwWjJkbGNpQnBibVp2TGdvdkx3b3ZMeUJGZUhSeVlXTjBaV1FnWm5KdmJT'
    'Qm5aWFJEYUdGdWJtVnNUM1YwY0hWMEozTWdZbTlrZVNCMGJ5QnpkWEJ3YjNKMElIQnlaWFpwYjNW'
    'ekxXNXZkR1VnWTNKdmMzTm1ZV1JsTGdvdkx5QlVhR1VnYjNKcFoybHVZV3dnYjNWMFpYSWdablZ1'
    'WTNScGIyNGdaR2xrSUNKbWFXNWtJSFJ5YVdkblpYSWc0b2FTSUdOdmJYQjFkR1VnYjNWMGNIVjBJ'
    'R0Z6SUc5dVpRb3ZMeUJ6WVcxd2JHVXVJaUJHYjNJZ1kzSnZjM05tWVdSbElIZGxJRzVsWldRZ2RH'
    'OGdZMjl0Y0hWMFpTQjBhR1VnYjNWMGNIVjBJRlJYU1VORklPS0FsQ0J2Ym1ObElHWnZjZ292THlC'
    'MGFHVWdZM1Z5Y21WdWRDQjBjbWxuWjJWeUxDQnZibU5sSUdadmNpQjBhR1VnZEhKcFoyZGxjaUJD'
    'UlVaUFVrVWdhWFFnNG9DVUlHRnVaQ0JpYkdWdVpDQnZkbVZ5Q2k4dklIUm9aU0JtYVhKemRDQTJO'
    'Q0J6WVcxd2JHVnpJR0ZtZEdWeUlHRWdjbVYwY21sbloyVnlMZ292THdvdkx5QlFZWEpoYldWMFpY'
    'SnpJSFJ5YVdkUVlYUXZkSEpwWjFKdmR5OTBjbWxuVG05MFpTOTBiMjVsVTJ4cFpHVlVZWEpuWlhR'
    'Z1lYSmxJSGRvWVhRZ2RHaGxJRzkxZEdWeUNpOHZJR1oxYm1OMGFXOXVKM01nZEhKcFoyZGxjaUJ6'
    'WldGeVkyZ2dkMjkxYkdRZ2FHRjJaU0JqYjIxd2RYUmxaRHNnZEdobElHSnZaSGtnZFhObGN5QjBh'
    'R1Z0Q2k4dklHbGtaVzUwYVdOaGJHeDVJQ2gzWVhNZ1lYTWdZR3h2WTJGc0lIWmhjaUE5SUhKbGMz'
    'VnNkQ0J2WmlCelpXRnlZMmhnTENCdWIzY2dZWE1nY0dGeVlXMWxkR1Z5Y3lrdUNpOHZJT0tVZ09L'
    'VWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tV'
    'Z09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1Vn'
    'T0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdP'
    'S1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09L'
    'VWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ0FwbWJHOWhkQ0Jm'
    'WjJOdlFtOWtlU2hwYm5RZ1kyZ3NJRkJ2YzJsMGFXOXVJSEJ2Y3l3Z1pteHZZWFFnZEdsdFpTd2da'
    'bXh2WVhRZ2NtOTNWR2x0WlN3S0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElIUnlhV2RRWVhRc0lH'
    'bHVkQ0IwY21sblVtOTNMQ0JPYjNSbElIUnlhV2RPYjNSbExDQnBiblFnZEc5dVpWTnNhV1JsVkdG'
    'eVoyVjBLU0I3Q2lBZ0lDQnBaaUFvZEhKcFowNXZkR1V1YVc1emRISjFiV1Z1ZENBOFBTQXdJSHg4'
    'SUhSeWFXZE9iM1JsTG1sdWMzUnlkVzFsYm5RZ1BpQXpNU0I4ZkNCMGNtbG5UbTkwWlM1d1pYSnBi'
    'MlFnUEQwZ01Da0tJQ0FnSUNBZ0lDQnlaWFIxY200Z01DNHdPd29LSUNBZ0lGTmhiWEJzWlVsdVpt'
    'OGdjMjF3SUQwZ2MyRnRjR3hsYzF0MGNtbG5UbTkwWlM1cGJuTjBjblZ0Wlc1MElDMGdNVjA3Q2lB'
    'Z0lDQnBaaUFvYzIxd0xteGxibWQwYUNBOVBTQXdLU0J5WlhSMWNtNGdNQzR3T3dvS0lDQWdJQzh2'
    'SUZScFkyc3RZbUZ6WldRZ1pXeGhjSE5sWkRvZ2FXNXNhVzVsSUVkU0lHTnZiWEIxZEdGMGFXOXVM'
    'Q0J6YTJsd0lHNWhiV1ZrSUdsdWRHVnliV1ZrYVdGMFpYTUtJQ0FnSUdac2IyRjBJR1ZzWVhCelpX'
    'UWdQU0FvWm14dllYUW9abVYwWTJoVWFXTnJLSEJoZEZScFkydFBabVp6WlhSYmNHOXpMbk52Ym1k'
    'UWIzTmRLeWh3YjNNdWNtOTNMWEJoZEZOMFlYSjBVbTkzVzNCdmN5NXpiMjVuVUc5elhTa3BLUW9n'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBcklIQnZjeTUwYVdOckNpQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQzBnWm14dllYUW9abVYwWTJoVWFXTnJLSEJoZEZScFkydFBabVp6WlhSYmRI'
    'SnBaMUJoZEYwcktIUnlhV2RTYjNjdGNHRjBVM1JoY25SU2IzZGJkSEpwWjFCaGRGMHBLU2twQ2lB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0x5QlVTVU5MVTE5UVJWSmZVMFZET3dvZ0lDQWdhV1ln'
    'S0dWc1lYQnpaV1FnUENBd0xqQXBJSEpsZEhWeWJpQXdMakE3Q2dvZ0lDQWdMeThnNHBTQTRwU0E0'
    'cFNBSUdaVFlXMXdiR1ZRYjNNZ1lXTmpkVzExYkdGMGIzSWdLR1Y0WVdOMElIQmxjaTF5YjNjZ2JH'
    'bHVaV0Z5TFhKaGJYQWdhVzUwWldkeVlYUnBiMjRwSU9LVWdPS1VnT0tVZ0FvZ0lDQWdMeThnVW1W'
    'aGJDQlFWQ0J3WlhKcGIyUWdaWFp2YkhWMGFXOXVJR2x6SUhCcFpXTmxkMmx6WlMxc2FXNWxZWEln'
    'ZDJsMGFDQmljbVZoYTNCdmFXNTBjeUJoZENCeWIzY0tJQ0FnSUM4dklHSnZkVzVrWVhKcFpYTWdL'
    'R0Z1WkNCamJHRnRjSE1nZDJobGJpQXplSGdnY21WaFkyaGxjeUIwWVhKblpYUXBMaUJVYUdVZ2My'
    'bHVaMnhsTFdsdWRHVm5jbUZzQ2lBZ0lDQXZMeUJtYjNKdGRXeGhJR0JES2xRdlpGQWdLaUJzYmlo'
    'UU1TOVFNQ2xnSUdaeWIyMGdkSEpwWjJkbGNpQjBieUJqZFhKeVpXNTBJR2x6SUhkeWIyNW5JR0ps'
    'WTJGMWMyVUtJQ0FnSUM4dklHbDBJR0Z6YzNWdFpYTWdZU0JUU1U1SFRFVWdiR2x1WldGeUlISmhi'
    'WEFnNG9DVUlHRmpkSFZoYkNCUUtIUXBJR2x6SUcxMWJIUnBMWE5sWjIxbGJuUXVJRVpwZURvS0lD'
    'QWdJQzh2SUdGalkzVnRkV3hoZEdVZ1pYaGhZM1FnY0dWeUxYSnZkeUJqYjI1MGNtbGlkWFJwYjI1'
    'eklHUjFjbWx1WnlCMGFHVWdabTl5ZDJGeVpDQnpZMkZ1SUhWemFXNW5DaUFnSUNBdkx5RGlpS3Nv'
    'UXk5UUtIUXBLV1IwSUQwZ1F5cFVYM0p2ZHk4b1VGOWxibVF0VUY5emRHRnlkQ2tnS2lCc2JpaFFY'
    'MlZ1WkM5UVgzTjBZWEowS1NCbWIzSWdaV0ZqYUFvZ0lDQWdMeThnYzJWbmJXVnVkQ3dnY0d4MWN5'
    'QmhJSEJoY25ScFlXd3RjbTkzSUhSaGFXd3ZhR1ZoWkNCbWIzSWdkSEpwWjJkbGNpQmhibVFnWTNW'
    'eWNtVnVkQ0J5YjNkekxnb2dJQ0FnTHk4Z1EyOXpkRG9nZmpFd0lHVjRkSEpoSUc5d2N5QndaWEln'
    'Wm05eWQyRnlaQzF6WTJGdUlISnZkeXdnYm04Z1pYaDBjbUVnZEdWNGRIVnlaU0JtWlhSamFHVnpM'
    'Z29nSUNBZ1pteHZZWFFnWDJaVFlXMXdiR1ZRYjNOQlkyTWdQU0F3TGpBN0Nnb2dJQ0FnYVc1MElG'
    'OXdZM1FnUFNCcGJuUW9jRzl6TG5ScFkyc3BPd29nSUNBZ1RtOTBaU0JmY0dOeUlEMGdaMlYwVG05'
    'MFpTaHdiM011YzI5dVoxQnZjeXdnY0c5ekxuSnZkeXdnWTJncE93b0tJQ0FnSUM4dklPS1VnT0tV'
    'Z0NCRGIyMWlhVzVsWkNCbWIzSjNZWEprSUhOallXNDZJSEpsWW5WcGJHUWdjR2wwWTJnZ1FVNUVJ'
    'SFp2YkhWdFpTQm1jbTl0SUhSeWFXZG5aWElnZEc4Z1kzVnljbVZ1ZENEaWxJRGlsSUFLSUNBZ0lH'
    'WnNiMkYwSUdWbVptVmpkR2wyWlZCbGNtbHZaQ0E5SUdac2IyRjBLSFJ5YVdkT2IzUmxMbkJsY21s'
    'dlpDazdDaUFnSUNCbWJHOWhkQ0IwWVhKblpYUlFaWEpwYjJRZ0lDQWdQU0JtYkc5aGRDaDBjbWxu'
    'VG05MFpTNXdaWEpwYjJRcE93b0tJQ0FnSUM4dklPS1VnT0tVZ0NCV2IyeDFiV1VnYVc1cGRHbGhi'
    'R2w2WVhScGIyNGdLRkJVSUhCbGNtbHZaQzF2Ym14NUxYSmxkSEpwWjJkbGNpQnhkV2x5YXlrZzRw'
    'U0E0cFNBQ2lBZ0lDQXZMeUJRVkNCelpXMWhiblJwWTNNNklHRWdjbTkzSUhkcGRHZ2djR1Z5YVc5'
    'a0lENGdNQ0JpZFhRZ1RrOGdhVzV6ZEhKMWJXVnVkQ0J1ZFcxaVpYSWdhWE1nWVFvZ0lDQWdMeThn'
    'Y21WMGNtbG5aMlZ5SUhSb1lYUWdVa1ZUVkVGU1ZGTWdkR2hsSUhOaGJYQnNaU0JoZENCdlptWnpa'
    'WFFnTUNCQ1ZWUWdTMFZGVUZNZ2RHaGxJSEJ5YVc5eUNpQWdJQ0F2THlCMmIyeDFiV1VnNG9DVUlH'
    'bDBKM01nZEdobElITmhiV1VnYVc1emRISjFiV1Z1ZENCaVpXbHVaeUJ5WlMxd2JHRjVaV1FzSUc1'
    'dmRDQmhJR1p5WlhOb0lHeGhkR05vTGdvZ0lDQWdMeThLSUNBZ0lDOHZJRlJvWlNCaWRXNWtiR1Zr'
    'SUhSeWFXZG5aWEl0Wm1sdVpHVnlJR0poWTJ0MGNtRmphM01nWUhSeWFXZE9iM1JsTG1sdWMzUnlk'
    'VzFsYm5SZ0lHWnliMjBnWVFvZ0lDQWdMeThnY0hKcGIzSWdjbTkzSUdadmNpQnpZVzF3YkdVZ2JH'
    'OXZhM1Z3TENCaWRYUWdkR2hsSUU5U1NVZEpUa0ZNSUhSeWFXZG5aWElnWTJWc2JDQjBaV3hzY3lC'
    'MWN3b2dJQ0FnTHk4Z2QyaGxkR2hsY2lCUVZDQjNiM1ZzWkNCa2J5QmhJSFp2YkNCeVpYTmxkQzRn'
    'U1dZZ2RHaGxJRzl5YVdkcGJtRnNJR05sYkd3Z2FHRmtJRzV2SUdsdWMzUXNDaUFnSUNBdkx5QjNa'
    'U0J1WldWa0lIUnZJSEpsWTI5dWMzUnlkV04wSUhSb1pTQjJiMngxYldVZ2RHaGhkQ0IzWVhNZ2FX'
    'NGdaV1ptWldOMElHcDFjM1FnWW1WbWIzSmxDaUFnSUNBdkx5QjBhR2x6SUhCbGNtbHZaQzF2Ym14'
    'NUlISmxkSEpwWjJkbGNpQmllU0IzWVd4cmFXNW5JR0poWTJzZ2RHOGdkR2hsSUd4aGMzUWdhVzV6'
    'ZEMxaVpXRnlhVzVuQ2lBZ0lDQXZMeUJ5YjNjZ1lXNWtJR1p2Y25kaGNtUXRjMk5oYm01cGJtY2da'
    'V1ptWldOMGN5NEtJQ0FnSUM4dkNpQWdJQ0F2THlCVWFHbHpJSGRoY3lCbWIzVnVaQ0JpZVNCd1pY'
    'SXRjbTkzSUdWdVpYSm5lU0JqYjIxd1lYSmxJR0ZuWVdsdWMzUWdlRzF3SUc5dUlITnZibWN0Y0c5'
    'eklERTBDaUFnSUNBdkx5QW9jR0YwZEdWeWJpQXpOQ2tnWTJnek9pQndaWEpwYjJRdGIyNXNlU0J5'
    'WlhSeWFXZG5aWEp6SUdKbGRIZGxaVzRnUTNoNExYTmxkQ0J5YjNkeklIZGxjbVVLSUNBZ0lDOHZJ'
    'SEJzWVhscGJtY2dZWFFnWm5Wc2JDQjJiMngxYldVZ0tINHdMakE1SUZKTlV5a2dhVzV6ZEdWaFpD'
    'QnZaaUIwYUdVZ1pHbHRiV1ZrSUdWamFHOGdiR1YyWld3S0lDQWdJQzh2SUNoK01DNHdNeUJTVFZN'
    'cExpQldiMnd0Y0hKbGMyVnlkbVVnWTNWMGN5QjBiM1JoYkNCamFETWdVazFUSUdWeWNtOXlJSFp6'
    'SUhodGNDQmllU0F5TU1PWExnb2dJQ0FnTHk4ZzRwU0E0cFNBSUZadmJIVnRaU0J6Ylc5dmRHaHBi'
    'bWNnYzNsemRHVnRJT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdP'
    'S1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09L'
    'VWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tV'
    'Z09LVWdPS1VnQW9nSUNBZ0x5OGdSMlZ1WlhKaGJHbDZaV1FnTmpRdGMyRnRjR3hsSUhKaGJYQWdi'
    'MjRnUlZaRlVsa2dkbTlzZFcxbElHTm9ZVzVuWlNBb1EzaDRMQ0JGUVhnc0lFVkNlQ3dnUVhoNENp'
    'QWdJQ0F2THlCMGFXTnJJR0p2ZFc1a1lYSnBaWE1zSUhCc2RYTWdhVzVvWlhKcGRHVmtJSFpoYkhW'
    'bGN5QmhZM0p2YzNNZ1ptOXlkMkZ5WkMxelkyRnVJSEp2ZDNNcExnb2dJQ0FnTHk4Z1VtVndiR0Zq'
    'WlhNZ01qZ2dhVzVzYVc1bElHQjJiMngxYldVZ1BTQllZQ0J0ZFhSaGRHbHZibk1nZDJsMGFDQmhJ'
    'SFZ1YVdadmNtMGdjR0YwZEdWeWJpQjBhR0YwQ2lBZ0lDQXZMeUJ3Y21WelpYSjJaWE1nZEdobElH'
    'MXZjM1FnY21WalpXNTBJR05vWVc1blpTQmhibVFnY21GdGNITWdZWFFnYjNWMGNIVjBJSFJwYldV'
    'dUlGZHBkR2h2ZFhRZ2RHaHBjd29nSUNBZ0x5OGdjbUZ0Y0N3Z1pXRmphQ0IyYjJ4MWJXVWdaV1pt'
    'WldOMElIQnliMlIxWTJWeklHRWdjMmx1WjJ4bExYTmhiWEJzWlNCemRHVndJR2x1SUdGdGNHeHBk'
    'SFZrWlN3S0lDQWdJQzh2SUdGdVpDQjViM1VnYUdWaGNpQnBkQ0JoY3lCaElITm9ZWEp3SUdOc2FX'
    'TnJJT0tBbENCd1lYSjBhV04xYkdGeWJIa2dZbUZrSUc5dUlFTjRlQzFvWldGMmVRb2dJQ0FnTHk4'
    'Z2NHRjBkR1Z5Ym5NZ0tISnZkM01nTlMwek1DQnZaaUJ3WVhSMFpYSnVJREFzSUdGc2JDQnZaaUJ3'
    'WVhSMFpYSnVJREUzTENCbGRHTXVLUzRLSUNBZ0lDOHZDaUFnSUNBdkx5QlVkMjhnYzNSaGRHVWdk'
    'bUZzZFdWeklIQnNkWE1nWVNCMGFXTnJJSE4wWVcxd09nb2dJQ0FnTHk4Z0lDQmZkbTlzVUhKbGRp'
    'QTlJSFJvWlNCMllXeDFaU0JDUlVaUFVrVWdkR2hsSUcxdmMzUWdjbVZqWlc1MElHTm9ZVzVuWlFv'
    'Z0lDQWdMeThnSUNCZmRtOXNRM1Z5Y2lBOUlIUm9aU0IyWVd4MVpTQkJSbFJGVWlBb1kzVnljbVZ1'
    'ZENCbmNtOTFibVFnZEhKMWRHZ3BDaUFnSUNBdkx5QWdJRjkyYjJ4RGFHRnVaMlZCZEZScFkydEdJ'
    'RDBnWjJ4dlltRnNMWFJwWTJzdFpteHZZWFFnWVhRZ2QyaHBZMmdnZEdobElHTm9ZVzVuWlNCb1lY'
    'QndaVzVsWkFvZ0lDQWdMeThLSUNBZ0lDOHZJRUYwSUc5MWRIQjFkQ0IwYVcxbE9nb2dJQ0FnTHk4'
    'Z0lDQjJVbUZ0Y0NBZ1BTQmpiR0Z0Y0Nnb2NHOXpMblJwWTJ0R0lDMGdYM1p2YkVOb1lXNW5aVUYw'
    'VkdsamEwWXBJQ29nVTBGTlVGOVFSVkpmVkVsRFN5QXZJRFkwTENBd0xDQXhLUW9nSUNBZ0x5OGdJ'
    'Q0JsWm1aV2Iyd2dQU0J0YVhnb1gzWnZiRkJ5WlhZc0lGOTJiMnhEZFhKeUxDQjJVbUZ0Y0NrZ0t5'
    'QjBjbVZ0YjJ4dlJHVnNkR0VLSUNBZ0lDOHZDaUFnSUNBdkx5QlVkMjhnYUdWc2NHVnlJRzFoWTNK'
    'dmN6b0tJQ0FnSUM4dklDQWdWazlNWDBsT1NWUW9WaWtnSUNBZzRvQ1VJSE5sZENCaWIzUm9JSEJ5'
    'WlhZZ1lXNWtJR04xY25JZ2RHOGdWaXdnYm04Z2RISmhibk5wZEdsdmJnb2dJQ0FnTHk4Z0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDaDFjMlZrSUdGMElIUnlhV2RuWlhJZ2FXNXBkRHNnWlhocGMz'
    'UnBibWNnTmpRdGMyRnRjR3hsSUdCa1pXTnNhV05yWUFvZ0lDQWdMeThnSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUdaaFkzUnZjaUJvWVc1a2JHVnpJSFJvWlNCMGNtbG5aMlZ5SUdaaFpHVXRhVzRw'
    'TGdvZ0lDQWdMeThnSUNCV1QweGZVMFZVS0ZZc0lGUXBJQ0RpZ0pRZ2NISnZiVzkwWlNCamRYSnk0'
    'b2FTY0hKbGRpd2djMlYwSUdOMWNuSWdkRzhnVml3Z2MzUmhiWEFnZEdsamF5QlVDaUFnSUNBdkx5'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdLSFZ6WldRZ1lYUWdaWFpsY25rZ2FXNHRjR3hoZVNC'
    'MmIyeDFiV1VnWTJoaGJtZGxLUzRLSUNBZ0lDOHZDaUFnSUNBdkx5QlVhR1VnYldGamNtOXpJSFZ6'
    'WlNCaElHTnZiVzFoTFdWNGNISmxjM05wYjI0Z1ltOWtlU0J6YnlCMGFHVjVJR1Y0Y0dGdVpDQmpi'
    'R1ZoYm14NUlHbHVjMmxrWlFvZ0lDQWdMeThnWVc1NUlFZE1VMHdnYzNSaGRHVnRaVzUwSUdOdmJu'
    'UmxlSFFnS0dsbUwyVnNjMlVnZDJsMGFHOTFkQ0JpY21GalpYTXNJR1YwWXk0cExnb2dJQ0FnYVc1'
    'MElDQWdYM1p2YkZCeVpYWWdQU0F3T3dvZ0lDQWdhVzUwSUNBZ1gzWnZiRU4xY25JZ1BTQXdPd29n'
    'SUNBZ1pteHZZWFFnWDNadmJFTm9ZVzVuWlVGMFZHbGphMFlnUFNBdE1XVTVPeUFnTHk4Z1ptRnlM'
    'WEJoYzNRZ2MyVnVkR2x1Wld3NklISmhiWEFnWm5Wc2JIa2dZMjl0Y0d4bGRHVUtJQ0FnSUdac2Iy'
    'RjBJRjkwY21WdGIyeHZSR1ZzZEdFZ0lDQWdJRDBnTUM0d095QWdJQzh2SUhSeVpXMXZiRzhnWVhC'
    'd2JHbGxjeUJoZENCdmRYUndkWFFzSUc1dmRDQjJhV0VnVms5TVgxTkZWQW9LSUNBZ0lDTmtaV1pw'
    'Ym1VZ1ZrOU1YMGxPU1ZRb1Zpa2dJQ0FnS0Y5MmIyeFFjbVYySUQwZ0tGWXBMQ0JmZG05c1EzVnlj'
    'aUE5SUY5MmIyeFFjbVYyS1FvZ0lDQWdJMlJsWm1sdVpTQldUMHhmVTBWVUtGWXNJRlFwSUNBb1gz'
    'WnZiRkJ5WlhZZ1BTQmZkbTlzUTNWeWNpd2dYM1p2YkVOMWNuSWdQU0FvVmlrc0lGOTJiMnhEYUdG'
    'dVoyVkJkRlJwWTJ0R0lEMGdLRlFwS1FvS0lDQWdJQzh2SUZCeVpTMWpiMjF3ZFhSbFpDQjBhV05y'
    'SUc5bUlIUm9aU0IwY21sbloyVnlJSEp2ZHlkeklHWnBjbk4wSUhScFkyc3VJRlZ6WldRZ1lYTWdk'
    'R2hsSUdOb1lXNW5aUW9nSUNBZ0x5OGdjM1JoYlhBZ1ptOXlJR0ZzYkNCMGNtbG5aMlZ5TFhKdmR5'
    'QjJiMndnWldabVpXTjBjeTRLSUNBZ0lHWnNiMkYwSUY5MGNtbG5aMlZ5VkdsamEwWWdQU0JtYkc5'
    'aGRDaG1aWFJqYUZScFkyc29jR0YwVkdsamEwOW1abk5sZEZ0MGNtbG5VR0YwWFNBcklDaDBjbWxu'
    'VW05M0lDMGdjR0YwVTNSaGNuUlNiM2RiZEhKcFoxQmhkRjBwS1NrN0Nnb2dJQ0FnVG05MFpTQmZk'
    'SEpwWjBObGJHeFBjbWxuSUQwZ1oyVjBUbTkwWlNoMGNtbG5VR0YwTENCMGNtbG5VbTkzTENCamFD'
    'azdDaUFnSUNCcFppQW9YM1J5YVdkRFpXeHNUM0pwWnk1cGJuTjBjblZ0Wlc1MElENGdNQ2tnZXdv'
    'Z0lDQWdJQ0FnSUZaUFRGOUpUa2xVS0hOdGNDNTJiMngxYldVcE95QWdMeThnVW1WaGJDQnBibk4w'
    'Y25WdFpXNTBJR3hoZEdOb0lPS0FsQ0JrWldOc2FXTnJJR2hoYm1Sc1pYTWdabUZrWlFvZ0lDQWdm'
    'U0JsYkhObElIc0tJQ0FnSUNBZ0lDQXZMeUJRWlhKcGIyUXRiMjVzZVNCeVpYUnlhV2RuWlhJZzRv'
    'Q1VJR1pwYm1RZ2RHaGxJR3hoYzNRZ2FXNXpkQzFpWldGeWFXNW5JSEp2ZHl3Z2FXNXBkQ0JtY205'
    'dENpQWdJQ0FnSUNBZ0x5OGdhWFJ6SUdsdWMzUnlkVzFsYm5RbmN5QmtaV1poZFd4MElIWnZiSFZ0'
    'WlN3Z2RHaGxiaUJtYjNKM1lYSmtMWE5qWVc0Z1pXWm1aV04wY3lCMWNDQjBid29nSUNBZ0lDQWdJ'
    'Qzh2SUNoaWRYUWdibTkwSUdsdVkyeDFaR2x1WnlrZ2RISnBaMUp2ZHk0Z1FtOTFibVJsWkNCelky'
    'RnVPaUF6TWlCeWIzZHpJR0poWTJ0M1lYSmtDaUFnSUNBZ0lDQWdMeThnWTI5MlpYSnpJSFJvWlNC'
    'MllYTjBJRzFoYW05eWFYUjVJRzltSUcxMWMybGpZV3dnWTJGelpYTXVDaUFnSUNBZ0lDQWdhVzUw'
    'SUY5cGJuTjBVbTkzSUQwZ2RISnBaMUp2ZHl3Z1gybHVjM1JRWVhRZ1BTQjBjbWxuVUdGME93b2dJ'
    'Q0FnSUNBZ0lHSnZiMndnWDJadmRXNWtTVzV6ZEV4aGRHTm9JRDBnWm1Gc2MyVTdDaUFnSUNBZ0lD'
    'QWdld29nSUNBZ0lDQWdJQ0FnSUNCcGJuUWdjMUlnUFNCMGNtbG5VbTkzTENCelVDQTlJSFJ5YVdk'
    'UVlYUTdDaUFnSUNBZ0lDQWdJQ0FnSUdadmNpQW9hVzUwSUd4aUlEMGdNVHNnYkdJZ1BDQXpNanNn'
    'YkdJckt5a2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdjMUl0TFRzS0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUdsbUlDaHpVaUE4SURBcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwWmlBb2Mx'
    'QWdQaUF3S1NCN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUhOUUxTMDdDaUFnSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lITlNJRDBnY0dGMFUzUmhjblJTYjNkYmMxQmRJQ3Nn'
    'S0hCaGRGSnZkMDltWm5ObGRGdHpVQ3N4WFNBdElIQmhkRkp2ZDA5bVpuTmxkRnR6VUYwcElDMGdN'
    'VHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCOUlHVnNjMlVnZXlCaWNtVmhhenNnZlFvZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnVG05MFpTQndiaUE5SUdk'
    'bGRFNXZkR1VvYzFBc0lITlNMQ0JqYUNrN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9jRzR1'
    'YVc1emRISjFiV1Z1ZENBK0lEQXBJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCZmFXNXpk'
    'Rkp2ZHlBOUlITlNPeUJmYVc1emRGQmhkQ0E5SUhOUU93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJRjltYjNWdVpFbHVjM1JNWVhSamFDQTlJSFJ5ZFdVN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdZbkpsWVdzN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ0lDQWdJSDBL'
    'SUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0FnYVdZZ0tDRmZabTkxYm1SSmJuTjBUR0YwWTJncElIc0tJ'
    'Q0FnSUNBZ0lDQWdJQ0FnVms5TVgwbE9TVlFvYzIxd0xuWnZiSFZ0WlNrN0lDQXZMeUJPYnlCcGJu'
    'TjBJR3hoZEdOb0lPS0FsQ0JrWldOc2FXTnJJR2hoYm1Sc1pYTWdabUZrWlFvZ0lDQWdJQ0FnSUgw'
    'Z1pXeHpaU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lFNXZkR1VnWDJ4aGRHTm9UbTkwWlNBOUlHZGxkRTV2'
    'ZEdVb1gybHVjM1JRWVhRc0lGOXBibk4wVW05M0xDQmphQ2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lGTmhi'
    'WEJzWlVsdVptOGdYMnhoZEdOb1UyMXdJRDBnYzJGdGNHeGxjMXRmYkdGMFkyaE9iM1JsTG1sdWMz'
    'UnlkVzFsYm5RZ0xTQXhYVHNLSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjlzWVhSamFGTm5jaUE5SUhC'
    'aGRGUnBZMnRQWm1aelpYUmJYMmx1YzNSUVlYUmRJQ3NnS0Y5cGJuTjBVbTkzSUMwZ2NHRjBVM1Jo'
    'Y25SU2IzZGJYMmx1YzNSUVlYUmRLVHNLSUNBZ0lDQWdJQ0FnSUNBZ1pteHZZWFFnWDJ4aGRHTm9W'
    'R2xqYTBZZ1BTQm1iRzloZENobVpYUmphRlJwWTJzb1gyeGhkR05vVTJkeUtTazdDaUFnSUNBZ0lD'
    'QWdJQ0FnSUZaUFRGOUpUa2xVS0Y5c1lYUmphRk50Y0M1MmIyeDFiV1VwT3lBZ0x5OGdVbVZqYjI1'
    'emRISjFZM1JwYjI0Z1ltRnpaV3hwYm1VZzRvQ1VJRzV2SUhSeVlXNXphWFJwYjI0S0lDQWdJQ0Fn'
    'SUNBZ0lDQWdMeThnUVhCd2JIa2diR0YwWTJnZ2NtOTNKM01nZEdsamF5MHdJSFp2YkNCbFptWmxZ'
    'M1J6SUNoMGNtRnVjMmwwYVc5dWN5QnpkR0Z0Y0dWa0lHRjBJSEp2ZHlCMGFXTnJLUzRLSUNBZ0lD'
    'QWdJQ0FnSUNBZ2FXWWdLRjlzWVhSamFFNXZkR1V1WldabVpXTjBJRDA5SURCNFF5a2dWazlNWDFO'
    'RlZDaHRhVzRvWDJ4aGRHTm9UbTkwWlM1d1lYSmhiU3dnTmpRcExDQmZiR0YwWTJoVWFXTnJSaWs3'
    'Q2lBZ0lDQWdJQ0FnSUNBZ0lHVnNjMlVnYVdZZ0tGOXNZWFJqYUU1dmRHVXVaV1ptWldOMElEMDlJ'
    'REI0UlNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjlsY3lBOUlDaGZiR0YwWTJoT2Iz'
    'UmxMbkJoY21GdElENCtJRFFwSUNZZ01IaEdPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5'
    'bGRpQTlJQ0JmYkdGMFkyaE9iM1JsTG5CaGNtRnRJQ0FnSUNBZ0lDWWdNSGhHT3dvZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnYVdZZ0tGOWxjeUE5UFNBd2VFRXBJQ0FnSUNBZ1ZrOU1YMU5GVkNoamJHRnRj'
    'Q2hmZG05c1EzVnljaUFySUY5bGRpd2dNQ3dnTmpRcExDQmZiR0YwWTJoVWFXTnJSaWs3Q2lBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0JsYkhObElHbG1JQ2hmWlhNZ1BUMGdNSGhDS1NCV1QweGZVMFZVS0dO'
    'c1lXMXdLRjkyYjJ4RGRYSnlJQzBnWDJWMkxDQXdMQ0EyTkNrc0lGOXNZWFJqYUZScFkydEdLVHNL'
    'SUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0F2THlCTVlYUmphQ0J5YjNjbmN5Qnda'
    'WEl0ZEdsamF5QnpiR2xrWlNBb1FYaDRMQ0EyZUhnc0lEVjRlQ0IyYjJ4MWJXVWdjR0Z5ZENrZ2Iz'
    'WmxjaUJtZFd4c0lISnZkeTRLSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjlzWVhSamFFNXZkR1V1Wlda'
    'bVpXTjBJRDA5SURCNFFTQjhmQ0JmYkdGMFkyaE9iM1JsTG1WbVptVmpkQ0E5UFNBd2VEWWdmSHdL'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJRjlzWVhSamFFNXZkR1V1WldabVpXTjBJRDA5SURCNE5Ta2dl'
    'd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5MmRTQTlJQ2hmYkdGMFkyaE9iM1JsTG5CaGNt'
    'RnRQajQwS1NZd2VFWTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBiblFnWDNaa0lEMGdJRjlzWVhS'
    'amFFNXZkR1V1Y0dGeVlXMGdJQ0FnSmpCNFJqc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0Jm'
    'YzNSbGNDQTlJQ2hmZG5VK01Da2dQeUJmZG5VZ09pQXRYM1prT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnYVc1MElGOW1kRXdnSUQwZ1ptVjBZMmhVYVdOcktGOXNZWFJqYUZObmNpQXJJREVwSUMwZ1pt'
    'VjBZMmhVYVdOcktGOXNZWFJqYUZObmNpa2dMU0F4T3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnVms5'
    'TVgxTkZWQ2hqYkdGdGNDaGZkbTlzUTNWeWNpQXJJRjl6ZEdWd0lDb2dYMlowVEN3Z01Dd2dOalFw'
    'TENCZmJHRjBZMmhVYVdOclJpazdDaUFnSUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lDQWdM'
    'eThnUm05eWQyRnlaQzF6WTJGdUlHVm1abVZqZEhNZ2IyNGdjbTkzY3lCZmFXNXpkRkp2ZHlzeElD'
    'NHVMaUIwY21sblVtOTNMVEVzSUhkaGJHdHBibWNLSUNBZ0lDQWdJQ0FnSUNBZ0x5OGdkR2h5YjNW'
    'bmFDQmhibmtnYVc1MFpYSnRaV1JwWVhSbElIQmxjbWx2WkMxdmJteDVJSEpsZEhKcFoyZGxjbk1n'
    'ZDJsMGFHOTFkQ0J5WlhObGRIUnBibWN1Q2lBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0JmZG1ad0lEMGdY'
    'Mmx1YzNSUVlYUXNJRjkyWm5JZ1BTQmZhVzV6ZEZKdmR5QXJJREU3Q2lBZ0lDQWdJQ0FnSUNBZ0lH'
    'bG1JQ2hmZG1aeUlENDlJSEJoZEZOMFlYSjBVbTkzVzE5MlpuQmRJQ3NnS0hCaGRGSnZkMDltWm5O'
    'bGRGdGZkbVp3S3pGZElDMGdjR0YwVW05M1QyWm1jMlYwVzE5MlpuQmRLU2tnZXdvZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnWDNabWNDc3JPeUJmZG1aeUlEMGdLRjkyWm5BZ1BDQlRUMDVIWDB4RlRrZFVT'
    'Q2tnUHlCd1lYUlRkR0Z5ZEZKdmQxdGZkbVp3WFNBNklEQTdDaUFnSUNBZ0lDQWdJQ0FnSUgwS0lD'
    'QWdJQ0FnSUNBZ0lDQWdabTl5SUNocGJuUWdYM1pwSUQwZ01Ec2dYM1pwSUR3Z05qUTdJRjkyYVNz'
    'cktTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvWDNabWNDQStJSFJ5YVdkUVlYUWdmSHdn'
    'S0Y5MlpuQWdQVDBnZEhKcFoxQmhkQ0FtSmlCZmRtWnlJRDQ5SUhSeWFXZFNiM2NwS1NCaWNtVmhh'
    'enNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZmRtWndJRDQ5SUZOUFRrZGZURVZPUjFSSUtT'
    'QmljbVZoYXpzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsbUlDaGZkbVp5SUQ0OUlIQmhkRk4wWVhK'
    'MFVtOTNXMTkyWm5CZElDc2dLSEJoZEZKdmQwOW1abk5sZEZ0ZmRtWndLekZkSUMwZ2NHRjBVbTkz'
    'VDJabWMyVjBXMTkyWm5CZEtTa2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUY5MlpuQXJL'
    'enNnWDNabWNpQTlJQ2hmZG1ad0lEd2dVMDlPUjE5TVJVNUhWRWdwSUQ4Z2NHRjBVM1JoY25SU2Iz'
    'ZGJYM1ptY0YwZ09pQXdPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdOdmJuUnBiblZsT3dv'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnVG05MFpTQmZkbTRn'
    'UFNCblpYUk9iM1JsS0Y5MlpuQXNJRjkyWm5Jc0lHTm9LVHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'R2x1ZENCZmMyZHlWaUE5SUhCaGRGUnBZMnRQWm1aelpYUmJYM1ptY0YwZ0t5QW9YM1ptY2lBdElI'
    'QmhkRk4wWVhKMFVtOTNXMTkyWm5CZEtUc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0JmWm5S'
    'V0lDQTlJR1psZEdOb1ZHbGpheWhmYzJkeVZpQXJJREVwSUMwZ1ptVjBZMmhVYVdOcktGOXpaM0pX'
    'S1NBdElERTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQm1iRzloZENCZmRsUnBZMnRHSUQwZ1pteHZZ'
    'WFFvWm1WMFkyaFVhV05yS0Y5elozSldLU2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1gz'
    'WnVMbVZtWm1WamRDQTlQU0F3ZUVNcElGWlBURjlUUlZRb2JXbHVLRjkyYmk1d1lYSmhiU3dnTmpR'
    'cExDQmZkbFJwWTJ0R0tUc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHVnNjMlVnYVdZZ0tGOTJiaTVs'
    'Wm1abFkzUWdQVDBnTUhoQklIeDhJRjkyYmk1bFptWmxZM1FnUFQwZ01IZzJLU0I3Q2lBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjkyZFNBOUlDaGZkbTR1Y0dGeVlXMCtQalFwSmpCNFJp'
    'd2dYM1prSUQwZ1gzWnVMbkJoY21GdEpqQjRSanNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNC'
    'V1QweGZVMFZVS0dOc1lXMXdLRjkyYjJ4RGRYSnlJQ3NnS0Y5MmRUNHdQMTkyZFRvdFgzWmtLU0Fx'
    'SUY5bWRGWXNJREFzSURZMEtTd2dYM1pVYVdOclJpazdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQjlD'
    'aUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmxiSE5sSUdsbUlDaGZkbTR1WldabVpXTjBJRDA5SURCNFJT'
    'a2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsdWRDQmZaWE1nUFNBb1gzWnVMbkJoY21G'
    'dElENCtJRFFwSUNZZ01IaEdPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsdWRDQmZaWFln'
    'UFNBZ1gzWnVMbkJoY21GdElDQWdJQ0FnSUNZZ01IaEdPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUdsbUlDaGZaWE1nUFQwZ01IaEJLU0FnSUNBZ0lGWlBURjlUUlZRb1kyeGhiWEFvWDNadmJF'
    'TjFjbklnS3lCZlpYWXNJREFzSURZMEtTd2dYM1pVYVdOclJpazdDaUFnSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0FnWld4elpTQnBaaUFvWDJWeklEMDlJREI0UWlrZ1ZrOU1YMU5GVkNoamJHRnRjQ2hm'
    'ZG05c1EzVnljaUF0SUY5bGRpd2dNQ3dnTmpRcExDQmZkbFJwWTJ0R0tUc0tJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmZG00dVpXWm1aV04wSUQwOUlE'
    'QjROU2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0JmZG5VZ1BTQW9YM1p1TG5C'
    'aGNtRnRQajQwS1NZd2VFWXNJRjkyWkNBOUlGOTJiaTV3WVhKaGJTWXdlRVk3Q2lBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ1ZrOU1YMU5GVkNoamJHRnRjQ2hmZG05c1EzVnljaUFySUNoZmRuVStN'
    'RDlmZG5VNkxWOTJaQ2tnS2lCZlpuUldMQ0F3TENBMk5Da3NJRjkyVkdsamEwWXBPd29nSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdmUW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdYM1ptY2lzck93b2dJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ2FXWWdLRjkyWm5JZ1BqMGdjR0YwVTNSaGNuUlNiM2RiWDNabWNGMGdLeUFv'
    'Y0dGMFVtOTNUMlptYzJWMFcxOTJabkFyTVYwZ0xTQndZWFJTYjNkUFptWnpaWFJiWDNabWNGMHBL'
    'U0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1gzWm1jQ3NyT3lCZmRtWnlJRDBnS0Y5Mlpu'
    'QWdQQ0JUVDA1SFgweEZUa2RVU0NrZ1B5QndZWFJUZEdGeWRGSnZkMXRmZG1ad1hTQTZJREE3Q2lB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQjlDaUFn'
    'SUNCOUNnb2dJQ0FnTHk4Z1ZHOXVaUzF3YjNKMFlTQjBjbWxuWjJWeUlISnZkem9nZEdocGN5Qnli'
    'M2NnWTJGeWNtbGxjeUJoSURONGVDODFlSGdnYzJ4cFpHVWdkR0Z5WjJWMExnb2dJQ0FnTHk4Z1pX'
    'Wm1aV04wYVhabFVHVnlhVzlrSUhOMFlYbHpJR0YwSUhSb1pTQndjbVYyYVc5MWN5QjBjbWxuWjJW'
    'eUozTWdjR1Z5YVc5a0lDaGhiSEpsWVdSNUlITmxkQ0JoWW05MlpRb2dJQ0FnTHk4Z1puSnZiU0Iw'
    'Y21sblRtOTBaUzV3WlhKcGIyUXBPeUJ6Ykdsa1pTQmhZMk4xYlhWc1lYUmxjeUIwYjNkaGNtUWdk'
    'Rzl1WlZOc2FXUmxWR0Z5WjJWMElHOTJaWElnY205M2N5NEtJQ0FnSUdsbUlDaDBiMjVsVTJ4cFpH'
    'VlVZWEpuWlhRZ1BpQXdLU0I3Q2lBZ0lDQWdJQ0FnZEdGeVoyVjBVR1Z5YVc5a0lEMGdabXh2WVhR'
    'b2RHOXVaVk5zYVdSbFZHRnlaMlYwS1RzS0lDQWdJSDBLQ2lBZ0lDQXZMeUJCY0hCc2VTQjBjbWxu'
    'WjJWeUxYSnZkeUJsWm1abFkzUnpPaUJEZUhnZ0tITmxkQ0IyYjJ3cExDQkJlSGd2Tm5oNElDaDJi'
    'MndnYzJ4cFpHVWdjR0Z5ZEdsaGJDOW1kV3hzS1N3S0lDQWdJQzh2SUVWQmVDQW9abWx1WlNCMmIy'
    'd2dkWEFnNG9DVUlHbHVjM1JoYm5RcExDQkZRbmdnS0dacGJtVWdkbTlzSUdSdmQyNGc0b0NVSUds'
    'dWMzUmhiblFwTENBMWVIZ2dLSFJ2Ym1VcmRtOXNJSE5zYVdSbEtTNEtJQ0FnSUM4dklFRnNiQ0Iw'
    'Y21sbloyVnlMWEp2ZHlCMmIyd2dZMmhoYm1kbGN5QnpkR0Z0Y0NCMGFHVWdZMmhoYm1kbElIUnBZ'
    'MnNnWVhNZ1gzUnlhV2RuWlhKVWFXTnJSaUJ6YnlCMGFHVUtJQ0FnSUM4dklEWTBMWE5oYlhCc1pT'
    'QnlZVzF3SUdOdmJYQnNaWFJsY3lCM2FYUm9hVzRnZEdobElHWnBjbk4wSUg0eExqVnRjeUJ2WmlC'
    'MGFHVWdkSEpwWjJkbGNpQnliM2N1Q2lBZ0lDQnBaaUFvZEhKcFowNXZkR1V1WldabVpXTjBJRDA5'
    'SURCNFF5a2dld29nSUNBZ0lDQWdJRlpQVEY5VFJWUW9iV2x1S0hSeWFXZE9iM1JsTG5CaGNtRnRM'
    'Q0EyTkNrc0lGOTBjbWxuWjJWeVZHbGphMFlwT3dvZ0lDQWdmU0JsYkhObElHbG1JQ2gwY21sblRt'
    'OTBaUzVsWm1abFkzUWdQVDBnTUhoRktTQjdDaUFnSUNBZ0lDQWdMeThnUlhoMFpXNWtaV1FnWlda'
    'bVpXTjBjem9nUlVGNElHWnBibVVnZG05c0lIVndMQ0JGUW5nZ1ptbHVaU0IyYjJ3Z1pHOTNiaUFv'
    'YVc1emRHRnVkQ0J2YmlCMGFXTnJJREFwQ2lBZ0lDQWdJQ0FnYVc1MElGOWxjeUE5SUNoMGNtbG5U'
    'bTkwWlM1d1lYSmhiU0ErUGlBMEtTQW1JREI0UmpzS0lDQWdJQ0FnSUNCcGJuUWdYMlYySUQwZ0lI'
    'UnlhV2RPYjNSbExuQmhjbUZ0SUNBZ0lDQWdJQ1lnTUhoR093b2dJQ0FnSUNBZ0lHbG1JQ2hmWlhN'
    'Z1BUMGdNSGhCS1NBZ0lDQWdJRlpQVEY5VFJWUW9ZMnhoYlhBb1gzWnZiRU4xY25JZ0t5QmZaWFlz'
    'SURBc0lEWTBLU3dnWDNSeWFXZG5aWEpVYVdOclJpazdDaUFnSUNBZ0lDQWdaV3h6WlNCcFppQW9Y'
    'MlZ6SUQwOUlEQjRRaWtnVms5TVgxTkZWQ2hqYkdGdGNDaGZkbTlzUTNWeWNpQXRJRjlsZGl3Z01D'
    'd2dOalFwTENCZmRISnBaMmRsY2xScFkydEdLVHNLSUNBZ0lIMGdaV3h6WlNCcFppQW9kSEpwWjA1'
    'dmRHVXVaV1ptWldOMElEMDlJREI0UVNCOGZDQjBjbWxuVG05MFpTNWxabVpsWTNRZ1BUMGdNSGcy'
    'SUh4OElIUnlhV2RPYjNSbExtVm1abVZqZENBOVBTQXdlRFVwSUhzS0lDQWdJQ0FnSUNBdkx5QXdl'
    'RFVnUFNCMGIyNWxLM1p2YkNCemJHbGtaVG9nY0dsMFkyZ2dhR0Z1Wkd4bFpDQmllU0F3ZURNdFpY'
    'RjFhWFpoYkdWdWRDQmliRzlqYXl3Z2RtOXNJSEJoY21GdElITmhiV1VnWVhNZ01IaEJDaUFnSUNB'
    'Z0lDQWdhVzUwSUY5emRTQTlJQ2gwY21sblRtOTBaUzV3WVhKaGJUNCtOQ2ttTUhoR0xDQmZjMlFn'
    'UFNCMGNtbG5UbTkwWlM1d1lYSmhiU1l3ZUVZN0NpQWdJQ0FnSUNBZ2FXNTBJRjl6ZEdWd0lEMGdL'
    'Rjl6ZFQ0d0tTQS9JRjl6ZFNBNklDMWZjMlE3Q2lBZ0lDQWdJQ0FnYVdZZ0tIUnlhV2RRWVhRZ1BU'
    'MGdjRzl6TG5OdmJtZFFiM01nSmlZZ2RISnBaMUp2ZHlBOVBTQndiM011Y205M0tTQjdDaUFnSUNB'
    'Z0lDQWdJQ0FnSUZaUFRGOVRSVlFvWTJ4aGJYQW9YM1p2YkVOMWNuSWdLeUJmYzNSbGNDQXFJRjl3'
    'WTNRc0lEQXNJRFkwS1N3Z1gzUnlhV2RuWlhKVWFXTnJSaWs3Q2lBZ0lDQWdJQ0FnZlNCbGJITmxJ'
    'SHNLSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjkwY3lBOUlHWmxkR05vVkdsamF5aHdZWFJVYVdOclQy'
    'Wm1jMlYwVzNSeWFXZFFZWFJkS3loMGNtbG5VbTkzTFhCaGRGTjBZWEowVW05M1czUnlhV2RRWVhS'
    'ZEtTc3hLUW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUMwZ1ptVjBZMmhVYVdOcktIQmhkRlJw'
    'WTJ0UFptWnpaWFJiZEhKcFoxQmhkRjByS0hSeWFXZFNiM2N0Y0dGMFUzUmhjblJTYjNkYmRISnBa'
    'MUJoZEYwcEtUc0tJQ0FnSUNBZ0lDQWdJQ0FnVms5TVgxTkZWQ2hqYkdGdGNDaGZkbTlzUTNWeWNp'
    'QXJJRjl6ZEdWd0lDb2dLRjkwY3kweEtTd2dNQ3dnTmpRcExDQmZkSEpwWjJkbGNsUnBZMnRHS1Rz'
    'S0lDQWdJQ0FnSUNCOUNpQWdJQ0I5Q2dvZ0lDQWdMeThnNHBTQTRwU0FJREI0UlVRZ2JtOTBaU0Jr'
    'Wld4aGVUb2djMnRwY0NCdmRYUndkWFFnWm05eUlHQndZWEpoYldBZ2RHbGphM01nWVdaMFpYSWdk'
    'SEpwWjJkbGNpQnliM2NnNHBTQTRwU0FDaUFnSUNBdkx5QlVhR1VnYm05MFpTQmtiMlZ6YmlkMElH'
    'RmpkSFZoYkd4NUlITjBZWEowSUhWdWRHbHNJSFJwWTJzZ1lIQmhjbUZ0WUNCdlppQjBhR1VnZEhK'
    'cFoyZGxjaUJ5YjNjdUNpQWdJQ0F2THlCRllYSnNhV1Z5SUhSb1lXNGdkR2hoZENEaWhwSWdjbVYw'
    'ZFhKdUlITnBiR1Z1WTJVN0lHVm1abVZqZEdsMlpTQmxiR0Z3YzJWa0lHbHpJSEpsWkhWalpXUXVD'
    'aUFnSUNCcGJuUWdYMjV2ZEdWRVpXeGhlVlJwWTJ0eklEMGdNRHNLSUNBZ0lHbG1JQ2gwY21sblRt'
    'OTBaUzVsWm1abFkzUWdQVDBnTUhoRklDWW1JQ2dvZEhKcFowNXZkR1V1Y0dGeVlXMGdQajRnTkNr'
    'Z0ppQXdlRVlwSUQwOUlEQjRSQ2tnZXdvZ0lDQWdJQ0FnSUY5dWIzUmxSR1ZzWVhsVWFXTnJjeUE5'
    'SUhSeWFXZE9iM1JsTG5CaGNtRnRJQ1lnTUhoR093b2dJQ0FnSUNBZ0lHbG1JQ2gwY21sblVHRjBJ'
    'RDA5SUhCdmN5NXpiMjVuVUc5eklDWW1JSFJ5YVdkU2IzY2dQVDBnY0c5ekxuSnZkeUFtSmlCZmNH'
    'TjBJRHdnWDI1dmRHVkVaV3hoZVZScFkydHpLUW9nSUNBZ0lDQWdJQ0FnSUNCeVpYUjFjbTRnTUM0'
    'd095QWdMeThnWW1WbWIzSmxJR1JsYkdGNVpXUWdkSEpwWjJkbGNnb2dJQ0FnSUNBZ0lDOHZJRUZt'
    'ZEdWeUlHUmxiR0Y1T2lCemRXSjBjbUZqZENCa1pXeGhlU0IwYVdOcmN5Qm1jbTl0SUdWc1lYQnpa'
    'V1FnYzI4Z2RHaGxJRzV2ZEdVZ2MzUmhjblJ6SUdGMElHWnlaWE5vSUhROU1Bb2dJQ0FnSUNBZ0lH'
    'VnNZWEJ6WldRZ1BTQnRZWGdvTUM0d0xDQmxiR0Z3YzJWa0lDMGdabXh2WVhRb1gyNXZkR1ZFWld4'
    'aGVWUnBZMnR6S1NBdklGUkpRMHRUWDFCRlVsOVRSVU1wT3dvZ0lDQWdmUW9LSUNBZ0lDOHZJT0tV'
    'Z09LVWdDQXdlRGw0ZUNCellXMXdiR1VnYjJabWMyVjBJQ2gwY21sbloyVnlJSEp2ZHlCdmJteDVL'
    'VG9nYzNSaGNuUWdZWFFnY0dGeVlXMGdLaUF5TlRZZ2FXNGdjMkZ0Y0d4bElHUmhkR0VnNHBTQTRw'
    'U0FDaUFnSUNCcGJuUWdYM05oYlhCc1pVOW1abk5sZENBOUlEQTdDaUFnSUNCcFppQW9kSEpwWjA1'
    'dmRHVXVaV1ptWldOMElEMDlJREI0T1NBbUppQjBjbWxuVG05MFpTNXdZWEpoYlNBK0lEQXBJSHNL'
    'SUNBZ0lDQWdJQ0JmYzJGdGNHeGxUMlptYzJWMElEMGdkSEpwWjA1dmRHVXVjR0Z5WVcwZ0tpQXlO'
    'VFk3Q2lBZ0lDQjlDZ29nSUNBZ0x5OGc0cFNBNHBTQUlGUnlhV2RuWlhJZ2NtOTNKM01nY0dsMFky'
    'Z2djMnhwWkdVZ1pXWm1aV04wY3lBb01YaDRMeko0ZUNrZzRwU0E0cFNBQ2lBZ0lDQXZMeUJKWmlC'
    'MGFHVWdkSEpwWjJkbGNpQnliM2NnWTJGeWNtbGxaQ0F4ZUhnZ0tIQnZjblJoSUhWd0tTQnZjaUF5'
    'ZUhnZ0tIQnZjblJoSUdSdmQyNHBMQ0IwYUc5elpRb2dJQ0FnTHk4Z2MyeHBaR1Z6SUdoaGNIQmxi'
    'aUJ2YmlCMGFXTnJjeUF4TGk0b2MzQmxaV1F0TVNrZ2IyWWdkR2hsSUhSeWFXZG5aWElnY205M0xp'
    'QWdWMmhsYmlCd2IzTWdhWE1LSUNBZ0lDOHZJRkJCVTFRZ2RHaGxJSFJ5YVdkblpYSWdjbTkzTENC'
    'aGJHd2dLSE53WldWa0xURXBJRzltSUhSb2IzTmxJSFJwWTJ0eklHaGhkbVVnWTI5dGNHeGxkR1Zr'
    'TGdvZ0lDQWdMeThnVjJobGJpQndiM01nYVhNZ2IyNGdkR2hsSUhSeWFXZG5aWElnY205M0lHbDBj'
    'MlZzWml3Z2RHaGxJQ0pEZFhKeVpXNTBJSEp2ZHlCd1lYSjBhV0ZzSUhCcGRHTm9DaUFnSUNBdkx5'
    'QmxabVpsWTNRaUlHSnNiMk5ySUdKbGJHOTNJR2hoYm1Sc1pYTWdhWFFnNG9DVUlHUnZiaWQwSUdS'
    'dmRXSnNaUzFoY0hCc2VTQm9aWEpsTGdvZ0lDQWdhV1lnS0NoMGNtbG5VR0YwSUNFOUlIQnZjeTV6'
    'YjI1blVHOXpJSHg4SUhSeWFXZFNiM2NnSVQwZ2NHOXpMbkp2ZHlrZ0ppWUtJQ0FnSUNBZ0lDQW9k'
    'SEpwWjA1dmRHVXVaV1ptWldOMElEMDlJREI0TVNCOGZDQjBjbWxuVG05MFpTNWxabVpsWTNRZ1BU'
    'MGdNSGd5S1NBbUppQjBjbWxuVG05MFpTNXdZWEpoYlNBK0lEQXBJSHNLSUNBZ0lDQWdJQ0JwYm5R'
    'Z1gzUnlVMmR5SUQwZ2NHRjBWR2xqYTA5bVpuTmxkRnQwY21sblVHRjBYU0FySUNoMGNtbG5VbTkz'
    'SUMwZ2NHRjBVM1JoY25SU2IzZGJkSEpwWjFCaGRGMHBPd29nSUNBZ0lDQWdJR2x1ZENCZmRISlRj'
    'R1FnUFNCbVpYUmphRlJwWTJzb1gzUnlVMmR5SUNzZ01Ta2dMU0JtWlhSamFGUnBZMnNvWDNSeVUy'
    'ZHlLVHNLSUNBZ0lDQWdJQ0JwYm5RZ1gzUnlWR2xqYTNNZ1BTQmZkSEpUY0dRZ0xTQXhPeUFnTHk4'
    'Z1lXeHNJSEJ2YzNRdGRHbGpheTB3SUhScFkydHpJRzltSUhSeWFXZG5aWElnY205M0NpQWdJQ0Fn'
    'SUNBZ2FXWWdLSFJ5YVdkT2IzUmxMbVZtWm1WamRDQTlQU0F3ZURFcENpQWdJQ0FnSUNBZ0lDQWdJ'
    'R1ZtWm1WamRHbDJaVkJsY21sdlpDQTlJRzFoZUNneE1UTXVNQ3dnWldabVpXTjBhWFpsVUdWeWFX'
    'OWtJQzBnWm14dllYUW9kSEpwWjA1dmRHVXVjR0Z5WVcwZ0tpQmZkSEpVYVdOcmN5a3BPd29nSUNB'
    'Z0lDQWdJR1ZzYzJVZ0lDOHZJREI0TWdvZ0lDQWdJQ0FnSUNBZ0lDQmxabVpsWTNScGRtVlFaWEpw'
    'YjJRZ1BTQnRhVzRvT0RVMkxqQXNJR1ZtWm1WamRHbDJaVkJsY21sdlpDQXJJR1pzYjJGMEtIUnlh'
    'V2RPYjNSbExuQmhjbUZ0SUNvZ1gzUnlWR2xqYTNNcEtUc0tJQ0FnSUgwS0NpQWdJQ0F2THlCVWNt'
    'RmpheUJzWVhOMElIUnZibVV0Y0c5eWRHRWdjbUYwWlNBb1ptOXlJR1ZtWm1WamRDQTFJSFJ2SUds'
    'dWFHVnlhWFFwTGlBZ1NXNXBkR2xoYkdsNlpXUWdabkp2YlFvZ0lDQWdMeThnZEhKcFoyZGxjaUJ5'
    'YjNjbmN5QXplSGdnY0dGeVlXMDdJSFZ3WkdGMFpXUWdZbmtnWm05eWQyRnlaQ0J6WTJGdUlHRnpJ'
    'R2wwSUhkaGJHdHpJSEJoYzNRZ00zaDRJSEp2ZDNNdUNpQWdJQ0JwYm5RZ1gyeGhjM1JVVUZKaGRH'
    'VWdQU0F3T3dvZ0lDQWdhV1lnS0hSeWFXZE9iM1JsTG1WbVptVmpkQ0E5UFNBd2VETWdKaVlnZEhK'
    'cFowNXZkR1V1Y0dGeVlXMGdQaUF3S1NCZmJHRnpkRlJRVW1GMFpTQTlJSFJ5YVdkT2IzUmxMbkJo'
    'Y21GdE93b0tJQ0FnSUM4dklGUnlhV2RuWlhJZ2NtOTNKM01nZEdGcGJDQmpiMjUwY21saWRYUnBi'
    'MjRnZEc4Z1psTmhiWEJzWlZCdmN5QjJhV0VnY0dWeUxYUnBZMnNnYVc1MFpXZHlZWFJwYjI0dUNp'
    'QWdJQ0F2THlCUVZDQmtiMlZ6SUdScGMyTnlaWFJsSUhCbGNpMTBhV05ySUhOc2FXUmxJSFZ3WkdG'
    'MFpYTXNJRzV2ZENCamIyNTBhVzUxYjNWeklISmhiWEJ6SU9LQWxBb2dJQ0FnTHk4Z1kyOXVkR2x1'
    'ZFc5MWN5MXlZVzF3SUdsdWRHVm5jbUZzY3lCa2FYWmxjbWRsSUdaeWIyMGdkSEoxZEdnZ1ptOXlJ'
    'R1poYzNRZ2MyeHBaR1Z6TGlCWFpTQnNiMjl3Q2lBZ0lDQXZMeUJ2ZG1WeUlHVmhZMmdnZEdsamF5'
    'QnZaaUIwYUdVZ2RISnBaMmRsY2lCeWIzY2dZVzVrSUdGa1pDQkR3NWRrZEM5d1pYSnBiMlJmWVhS'
    'ZmRHbGpheTRLSUNBZ0lDOHZJRlJwWTJzZ01DQnpaV1Z6SUhSeWFXZE9iM1JsTG5CbGNtbHZaQzRn'
    'VkdsamEzTWdNUzR1S0hOd1pXVmtMVEVwSUhObFpTQnBibU55WlcxbGJuUmhiR3g1Q2lBZ0lDQXZM'
    'eUIxY0dSaGRHVmtJSEJsY21sdlpITWdhV1lnTVhoNEx6SjRlQ0JwY3lCd2NtVnpaVzUwTGdvZ0lD'
    'QWdhV1lnS0hSeWFXZFFZWFFnSVQwZ2NHOXpMbk52Ym1kUWIzTWdmSHdnZEhKcFoxSnZkeUFoUFNC'
    'd2IzTXVjbTkzS1NCN0NpQWdJQ0FnSUNBZ2FXNTBJRjl6WjNKVWNtbG5JQ0E5SUhCaGRGUnBZMnRQ'
    'Wm1aelpYUmJkSEpwWjFCaGRGMGdLeUFvZEhKcFoxSnZkeUF0SUhCaGRGTjBZWEowVW05M1czUnlh'
    'V2RRWVhSZEtUc0tJQ0FnSUNBZ0lDQnBiblFnWDNSeWFXZEdkV3hzSUQwZ1ptVjBZMmhVYVdOcktG'
    'OXpaM0pVY21sbklDc2dNU2tnTFNCbVpYUmphRlJwWTJzb1gzTm5jbFJ5YVdjcE95QWdMeThnUFNC'
    'emNHVmxaQW9nSUNBZ0lDQWdJR1pzYjJGMElGOURabDkwY21sbklEMGdZelJ6Y0dWbFpITmJjMjF3'
    'TG1acGJtVjBkVzVsSUNZZ01IaEdYU0FxSURReU9DNHdPd29nSUNBZ0lDQWdJR1pzYjJGMElGOWtk'
    'Q0E5SURFdU1DQXZJRlJKUTB0VFgxQkZVbDlUUlVNN0NpQWdJQ0FnSUNBZ1pteHZZWFFnWDFCMElE'
    'MGdabXh2WVhRb2RISnBaMDV2ZEdVdWNHVnlhVzlrS1RzS0lDQWdJQ0FnSUNBdkx5QlFaWEl0ZEds'
    'amF5QnpiR2xrWlNCemRHVndJR0Z0YjNWdWRDQW9jMmxuYm1Wa0tRb2dJQ0FnSUNBZ0lHbHVkQ0Jm'
    'YzNSbGNDQTlJREE3Q2lBZ0lDQWdJQ0FnYVdZZ0tIUnlhV2RPYjNSbExtVm1abVZqZENBOVBTQXdl'
    'REVwSUY5emRHVndJRDBnTFhSeWFXZE9iM1JsTG5CaGNtRnRPd29nSUNBZ0lDQWdJR1ZzYzJVZ2FX'
    'WWdLSFJ5YVdkT2IzUmxMbVZtWm1WamRDQTlQU0F3ZURJcElGOXpkR1Z3SUQwZ2RISnBaMDV2ZEdV'
    'dWNHRnlZVzA3Q2lBZ0lDQWdJQ0FnTHk4Z1FXTmpkVzExYkdGMFpTQndaWEl0ZEdsamF6b2dkR2xq'
    'YXlBd0lDc2dkR2xqYTNNZ01TNHVLSE53WldWa0xURXBDaUFnSUNBZ0lDQWdMeThnUW05MWJtUWdk'
    'RzhnTXpJZ1ptOXlJR052YlhCcGJHVnlJSE5oWm1WMGVUc2djM0JsWldRZ2FYTWc0b21rSURNeElH'
    'bHVJSEJ5WVdOMGFXTmxMZ29nSUNBZ0lDQWdJR1p2Y2lBb2FXNTBJRjkwSUQwZ01Ec2dYM1FnUENB'
    'ek1qc2dYM1FyS3lrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1gzUWdQajBnWDNSeWFXZEdkV3hz'
    'S1NCaWNtVmhhenNLSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjlRZENBK0lEQXVNQ2tnWDJaVFlXMXdi'
    'R1ZRYjNOQlkyTWdLejBnWDBObVgzUnlhV2NnS2lCZlpIUWdMeUJmVUhRN0NpQWdJQ0FnSUNBZ0lD'
    'QWdJQzh2SUZWd1pHRjBaU0J3WlhKcGIyUWdabTl5SUc1bGVIUWdkR2xqYXlBb1kyeGhiWEJ6SUcx'
    'aGRHTm9JRkJVSUhKaGJtZGxLUW9nSUNBZ0lDQWdJQ0FnSUNCcFppQW9YM04wWlhBZ0lUMGdNQ2tn'
    'WDFCMElEMGdZMnhoYlhBb1gxQjBJQ3NnWm14dllYUW9YM04wWlhBcExDQXhNVE11TUN3Z09EVTJM'
    'akFwT3dvZ0lDQWdJQ0FnSUgwS0lDQWdJSDBLQ2lBZ0lDQXZMeUJHYjNKM1lYSmtJSE5qWVc0NklI'
    'SnZkM01nVTFSU1NVTlVURmtnWW1WMGQyVmxiaUIwY21sbloyVnlJR0Z1WkNCamRYSnlaVzUwQ2lB'
    'Z0lDQnBaaUFvZEhKcFoxQmhkQ0FoUFNCd2IzTXVjMjl1WjFCdmN5QjhmQ0IwY21sblVtOTNJQ0U5'
    'SUhCdmN5NXliM2NwSUhzS0lDQWdJQ0FnSUNCcGJuUWdYMlp3SUQwZ2RISnBaMUJoZEN3Z1gyWnlJ'
    'RDBnZEhKcFoxSnZkeUFySURFN0NpQWdJQ0FnSUNBZ2FXWWdLRjltY2lBK1BTQndZWFJUZEdGeWRG'
    'SnZkMXRmWm5CZElDc2dLSEJoZEZKdmQwOW1abk5sZEZ0ZlpuQXJNVjBnTFNCd1lYUlNiM2RQWm1a'
    'elpYUmJYMlp3WFNrcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnWDJad0t5czdJRjltY2lBOUlDaGZabkFn'
    'UENCVFQwNUhYMHhGVGtkVVNDa2dQeUJ3WVhSVGRHRnlkRkp2ZDF0ZlpuQmRJRG9nTURzS0lDQWdJ'
    'Q0FnSUNCOUNpQWdJQ0FnSUNBZ1ptOXlJQ2hwYm5RZ1gyWnBJRDBnTURzZ1gyWnBJRHdnTVRJNE95'
    'QmZabWtyS3lrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1gyWndJRDRnY0c5ekxuTnZibWRRYjNN'
    'Z2ZId2dLRjltY0NBOVBTQndiM011YzI5dVoxQnZjeUFtSmlCZlpuSWdQajBnY0c5ekxuSnZkeWtw'
    'SUdKeVpXRnJPd29nSUNBZ0lDQWdJQ0FnSUNCcFppQW9YMlp3SUQ0OUlGTlBUa2RmVEVWT1IxUklL'
    'U0JpY21WaGF6c0tJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOW1jaUErUFNCd1lYUlRkR0Z5ZEZKdmQx'
    'dGZabkJkSUNzZ0tIQmhkRkp2ZDA5bVpuTmxkRnRmWm5Bck1WMGdMU0J3WVhSU2IzZFBabVp6WlhS'
    'YlgyWndYU2twSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUY5bWNDc3JPeUJmWm5JZ1BTQW9YMlp3'
    'SUR3Z1UwOU9SMTlNUlU1SFZFZ3BJRDhnY0dGMFUzUmhjblJTYjNkYlgyWndYU0E2SURBN0NpQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNCamIyNTBhVzUxWlRzS0lDQWdJQ0FnSUNBZ0lDQWdmUW9nSUNBZ0lD'
    'QWdJQ0FnSUNCT2IzUmxJRjltYmlBOUlHZGxkRTV2ZEdVb1gyWndMQ0JmWm5Jc0lHTm9LVHNLSUNB'
    'Z0lDQWdJQ0FnSUNBZ0x5OGdRVTVaSUhKdmR5QjNhWFJvSUhCbGNtbHZaQ0ErSURBZ1lXNWtJR1Zt'
    'Wm1WamRDQnViM1FnTXk4MUlHbHpJR0VnY21WaGJDQlNSVlJTU1VkSFJWSUtJQ0FnSUNBZ0lDQWdJ'
    'Q0FnTHk4Z2RHaGhkQ0JsYm1SeklIUm9aU0JtYjNKM1lYSmtJSE5qWVc0Z0tHNWxlSFFnWjJWMFEy'
    'aGhibTVsYkU5MWRIQjFkQ0JqWVd4c0lHaGhibVJzWlhNZ2FYUXBMZ29nSUNBZ0lDQWdJQ0FnSUNC'
    'aWIyOXNJRjltYmtselZHOXVaVlJ5YVdjZ1BTQW9LRjltYmk1bFptWmxZM1FnUFQwZ01IZ3pJSHg4'
    'SUY5bWJpNWxabVpsWTNRZ1BUMGdNSGcxS1NBbUppQmZabTR1Y0dWeWFXOWtJRDRnTUNrN0NpQWdJ'
    'Q0FnSUNBZ0lDQWdJR0p2YjJ3Z1gyWnVTWE5TWlhSeWFXY2dJQ0E5SUNoZlptNHVjR1Z5YVc5a0lE'
    'NGdNQ0FtSmlBaFgyWnVTWE5VYjI1bFZISnBaeWs3Q2lBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmWm01'
    'SmMxSmxkSEpwWnlrZ1luSmxZV3M3Q2lBZ0lDQWdJQ0FnSUNBZ0lDOHZJRlJ2Ym1VdGNHOXlkR0Vn'
    'ZEdGeVoyVjBPaUJ3YVhSamFDQnpiR2xrWlhNZ2RHOTNZWEprSUY5bWJpNXdaWEpwYjJRZ2IzWmxj'
    'aUJ5WlcxaGFXNXBibWNnY205M2N5NEtJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOW1ia2x6Vkc5dVpW'
    'UnlhV2NwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUhSaGNtZGxkRkJsY21sdlpDQTlJR1pzYjJG'
    'MEtGOW1iaTV3WlhKcGIyUXBPd29nSUNBZ0lDQWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ0lDQWdJQzh2'
    'SUZSeVlXTnJJR3hoYzNRZ00zaDRJSEpoZEdVZ1ptOXlJR1ZtWm1WamRDQTFJSFJ2SUdsdWFHVnlh'
    'WFFLSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjltYmk1bFptWmxZM1FnUFQwZ01IZ3pJQ1ltSUY5bWJp'
    'NXdZWEpoYlNBK0lEQXBJRjlzWVhOMFZGQlNZWFJsSUQwZ1gyWnVMbkJoY21GdE93b0tJQ0FnSUNB'
    'Z0lDQWdJQ0FnYVc1MElGOXpaM0lnSUNBOUlIQmhkRlJwWTJ0UFptWnpaWFJiWDJad1hTQXJJQ2hm'
    'Wm5JZ0xTQndZWFJUZEdGeWRGSnZkMXRmWm5CZEtUc0tJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOW1k'
    'V3hzSUNBOUlHWmxkR05vVkdsamF5aGZjMmR5SUNzZ01Ta2dMU0JtWlhSamFGUnBZMnNvWDNObmNp'
    'a2dMU0F4T3lBZ0x5OGdkR2xqYTNNZ01TNHVjM0JsWldRdE1Rb2dJQ0FnSUNBZ0lDQWdJQ0JtYkc5'
    'aGRDQmZjRk4wWVhKMFVtOTNJRDBnWldabVpXTjBhWFpsVUdWeWFXOWtPeUFnTHk4Z1ptOXlJR1pU'
    'WVcxd2JHVlFiM01nYVc1MFpXZHlZWFJwYjI0S0NpQWdJQ0FnSUNBZ0lDQWdJQzh2SUZCcGRHTm9J'
    'R1ZtWm1WamRITUtJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOW1iaTVsWm1abFkzUWdQVDBnTUhneEtR'
    'b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1pXWm1aV04wYVhabFVHVnlhVzlrSUQwZ2JXRjRLREV4TXk0'
    'd0xDQmxabVpsWTNScGRtVlFaWEpwYjJRZ0xTQm1iRzloZENoZlptNHVjR0Z5WVcwZ0tpQmZablZz'
    'YkNrcE93b2dJQ0FnSUNBZ0lDQWdJQ0JsYkhObElHbG1JQ2hmWm00dVpXWm1aV04wSUQwOUlEQjRN'
    'aWtLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR1ZtWm1WamRHbDJaVkJsY21sdlpDQTlJRzFwYmlnNE5U'
    'WXVNQ3dnWldabVpXTjBhWFpsVUdWeWFXOWtJQ3NnWm14dllYUW9YMlp1TG5CaGNtRnRJQ29nWDJa'
    'MWJHd3BLVHNLSUNBZ0lDQWdJQ0FnSUNBZ1pXeHpaU0JwWmlBb1gyWnVMbVZtWm1WamRDQTlQU0F3'
    'ZURNcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDOHZJRlJ2Ym1VZ2NHOXlkR0VnNG9DVUlIVnpa'
    'WE1nYVhSeklHOTNiaUJ3WVhKaGJTQmhjeUJ6Ykdsa1pTQnlZWFJsQ2lBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0JwWmlBb1pXWm1aV04wYVhabFVHVnlhVzlrSUR3Z2RHRnlaMlYwVUdWeWFXOWtLUW9nSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdWbVptVmpkR2wyWlZCbGNtbHZaQ0E5SUcxcGJpaDBZWEpu'
    'WlhSUVpYSnBiMlFzSUdWbVptVmpkR2wyWlZCbGNtbHZaQ0FySUdac2IyRjBLRjltYmk1d1lYSmhi'
    'U0FxSUY5bWRXeHNLU2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JsYkhObElHbG1JQ2hsWm1abFkz'
    'UnBkbVZRWlhKcGIyUWdQaUIwWVhKblpYUlFaWEpwYjJRcENpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdaV1ptWldOMGFYWmxVR1Z5YVc5a0lEMGdiV0Y0S0hSaGNtZGxkRkJsY21sdlpDd2daV1pt'
    'WldOMGFYWmxVR1Z5YVc5a0lDMGdabXh2WVhRb1gyWnVMbkJoY21GdElDb2dYMloxYkd3cEtUc0tJ'
    'Q0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNBZ0lDQmxiSE5sSUdsbUlDaGZabTR1WldabVpX'
    'TjBJRDA5SURCNE5Ta2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdMeThnUTI5dWRHbHVkV1VnZEc5'
    'dVpTQndiM0owWVNEaWdKUWdkWE5sY3lCTVFWTlVJRE40ZUNCeVlYUmxJQ2h1YjNRZ1gyWnVMbkJo'
    'Y21GdElTa0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmYkdGemRGUlFVbUYwWlNBK0lEQXBJ'
    'SHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9aV1ptWldOMGFYWmxVR1Z5YVc5a0lE'
    'd2dkR0Z5WjJWMFVHVnlhVzlrS1FvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JsWm1a'
    'bFkzUnBkbVZRWlhKcGIyUWdQU0J0YVc0b2RHRnlaMlYwVUdWeWFXOWtMQ0JsWm1abFkzUnBkbVZR'
    'WlhKcGIyUWdLeUJtYkc5aGRDaGZiR0Z6ZEZSUVVtRjBaU0FxSUY5bWRXeHNLU2s3Q2lBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBZ1pXeHpaU0JwWmlBb1pXWm1aV04wYVhabFVHVnlhVzlrSUQ0Z2RH'
    'RnlaMlYwVUdWeWFXOWtLUW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmxabVpsWTNS'
    'cGRtVlFaWEpwYjJRZ1BTQnRZWGdvZEdGeVoyVjBVR1Z5YVc5a0xDQmxabVpsWTNScGRtVlFaWEpw'
    'YjJRZ0xTQm1iRzloZENoZmJHRnpkRlJRVW1GMFpTQXFJRjltZFd4c0tTazdDaUFnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lDQWdMeThnVm05c2RX'
    'MWxJR1ZtWm1WamRITWdLSFJ5WVc1emFYUnBiMjRnYzNSaGJYQmxaQ0JoZENCMGFHbHpJSEp2ZHlk'
    'eklITjBZWEowSUhScFkyc3BDaUFnSUNBZ0lDQWdJQ0FnSUdac2IyRjBJRjltYmxScFkydEdJRDBn'
    'Wm14dllYUW9abVYwWTJoVWFXTnJLRjl6WjNJcEtUc0tJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOW1i'
    'aTVsWm1abFkzUWdQVDBnTUhoREtRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1ZrOU1YMU5GVkNodGFX'
    'NG9YMlp1TG5CaGNtRnRMQ0EyTkNrc0lGOW1ibFJwWTJ0R0tUc0tJQ0FnSUNBZ0lDQWdJQ0FnWld4'
    'elpTQnBaaUFvWDJadUxtVm1abVZqZENBOVBTQXdlRUVnZkh3Z1gyWnVMbVZtWm1WamRDQTlQU0F3'
    'ZURZcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0JmZG5VZ1BTQW9YMlp1TG5CaGNtRnRQ'
    'ajQwS1NZd2VFWXNJRjkyWkNBOUlGOW1iaTV3WVhKaGJTWXdlRVk3Q2lBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0JXVDB4ZlUwVlVLR05zWVcxd0tGOTJiMnhEZFhKeUlDc2dLRjkyZFQ0d1AxOTJkVG90WDNa'
    'a0tTQXFJRjltZFd4c0xDQXdMQ0EyTkNrc0lGOW1ibFJwWTJ0R0tUc0tJQ0FnSUNBZ0lDQWdJQ0Fn'
    'ZlFvZ0lDQWdJQ0FnSUNBZ0lDQmxiSE5sSUdsbUlDaGZabTR1WldabVpXTjBJRDA5SURCNFJTa2dl'
    'd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdMeThnUlVGNElHWnBibVVnZG05c0lIVndMQ0JGUW5nZ1pt'
    'bHVaU0IyYjJ3Z1pHOTNiaUFvYVc1emRHRnVkQ0J3WlhJZ2NtOTNLUW9nSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdhVzUwSUY5bGN5QTlJQ2hmWm00dWNHRnlZVzBnUGo0Z05Da2dKaUF3ZUVZN0NpQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNCcGJuUWdYMlYySUQwZ0lGOW1iaTV3WVhKaGJTQWdJQ0FnSUNBbUlEQjRS'
    'anNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZlpYTWdQVDBnTUhoQktTQWdJQ0FnSUZaUFRG'
    'OVRSVlFvWTJ4aGJYQW9YM1p2YkVOMWNuSWdLeUJmWlhZc0lEQXNJRFkwS1N3Z1gyWnVWR2xqYTBZ'
    'cE93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1pXeHpaU0JwWmlBb1gyVnpJRDA5SURCNFFpa2dWazlN'
    'WDFORlZDaGpiR0Z0Y0NoZmRtOXNRM1Z5Y2lBdElGOWxkaXdnTUN3Z05qUXBMQ0JmWm01VWFXTnJS'
    'aWs3Q2lBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnTHk4Z01IZzFJR0ZzYzI4Z1lY'
    'QndiR2xsY3lCMGFHVWdkbTlzZFcxbElITnNhV1JsSUhCdmNuUnBiMjRnS0docFoyZ2dibWxpWW14'
    'bElEMGdkWEFzSUd4dmR5QTlJR1J2ZDI0cENpQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZlptNHVaV1pt'
    'WldOMElEMDlJREI0TlNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjkyZFNBOUlDaGZa'
    'bTR1Y0dGeVlXMCtQalFwSmpCNFJpd2dYM1prSUQwZ1gyWnVMbkJoY21GdEpqQjRSanNLSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJRlpQVEY5VFJWUW9ZMnhoYlhBb1gzWnZiRU4xY25JZ0t5QW9YM1oxUGpB'
    'L1gzWjFPaTFmZG1RcElDb2dYMloxYkd3c0lEQXNJRFkwS1N3Z1gyWnVWR2xqYTBZcE93b2dJQ0Fn'
    'SUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0FnSUNBZ0lDOHZJRkJsY2kxMGFXTnJJR1pUWVcxd2JHVlFi'
    'M01nYVc1MFpXZHlZWFJwYjI0Z1ptOXlJSFJvYVhNZ2NtOTNJQ2h0WVhSamFHVnpJRkJVSUhObGJX'
    'RnVkR2xqY3dvZ0lDQWdJQ0FnSUNBZ0lDQXZMeUJsZUdGamRHeDVJT0tBbENCa2FYTmpjbVYwWlNC'
    'MGFXTnJMV0o1TFhScFkyc3NJRzV2ZENCc2FXNWxZWEl0Y21GdGNDQmpiMjUwYVc1MWIzVnpLUzRL'
    'SUNBZ0lDQWdJQ0FnSUNBZ0x5OGdVR1Z5YVc5a0lHRjBJSFJwWTJzZ01DQnZaaUIwYUdseklISnZk'
    'eUE5SUY5d1UzUmhjblJTYjNjdUlGUnBZMnR6SURFdUxtWjFiR3dnWVhCd2JIa0tJQ0FnSUNBZ0lD'
    'QWdJQ0FnTHk4Z2RHaGxJSE5zYVdSbElITjBaWEF1SUZkbElISmxMV1JsY21sMlpTQjBhR1VnY0dW'
    'eUxYUnBZMnNnYzNSbGNDQm1jbTl0SUhSb1pTQnliM2NuY3dvZ0lDQWdJQ0FnSUNBZ0lDQXZMeUJs'
    'Wm1abFkzUWdjbUYwYUdWeUlIUm9ZVzRnWkdsMmFXUnBibWNnS0dWbVptVmpkR2wyWlZCbGNtbHZa'
    'Q0F0SUY5d1UzUmhjblJTYjNjcEwxOW1kV3hzQ2lBZ0lDQWdJQ0FnSUNBZ0lDOHZJSE5wYm1ObElE'
    'TjRlQzgxZUhnZ1kyeGhiWEJwYm1jZ1lYUWdkR0Z5WjJWMElHMWhhMlZ6SUhSb1lYUWdZWFpsY21G'
    'blpTQnRhWE5zWldGa2FXNW5MZ29nSUNBZ0lDQWdJQ0FnSUNCN0NpQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNCbWJHOWhkQ0JmUTJZZ0lEMGdZelJ6Y0dWbFpITmJjMjF3TG1acGJtVjBkVzVsSUNZZ01IaEdY'
    'U0FxSURReU9DNHdPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdabXh2WVhRZ1gyUjBJQ0E5SURFdU1D'
    'QXZJRlJKUTB0VFgxQkZVbDlUUlVNN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCbWJHOWhkQ0JmVUhR'
    'Z0lEMGdYM0JUZEdGeWRGSnZkenNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQzh2SUVSbGRHVnliV2x1'
    'WlNCd1pYSXRkR2xqYXlCemRHVndJQ2h6YVdkdVpXUXBJR0Z1WkNCMFlYSm5aWFFnWm05eUlHTnNZ'
    'VzF3YVc1bkxnb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjl6ZEdWd0lEMGdNRHNLSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJR0p2YjJ3Z1gyTnNZVzF3Vkc5VVozUWdQU0JtWVd4elpUc0tJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmWm00dVpXWm1aV04wSUQwOUlEQjRNU2tnWDNOMFpYQWdQU0F0'
    'WDJadUxuQmhjbUZ0T3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnWld4elpTQnBaaUFvWDJadUxtVm1a'
    'bVZqZENBOVBTQXdlRElwSUY5emRHVndJRDBnWDJadUxuQmhjbUZ0T3dvZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnWld4elpTQnBaaUFvWDJadUxtVm1abVZqZENBOVBTQXdlRE1wSUhzS0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lDQmZZMnhoYlhCVWIxUm5kQ0E5SUhSeWRXVTdDaUFnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnYVdZZ0tGOXdVM1JoY25SU2IzY2dQQ0IwWVhKblpYUlFaWEpwYjJRcElDQWdJ'
    'Q0FnWDNOMFpYQWdQU0JmWm00dWNHRnlZVzA3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1pX'
    'eHpaU0JwWmlBb1gzQlRkR0Z5ZEZKdmR5QStJSFJoY21kbGRGQmxjbWx2WkNrZ1gzTjBaWEFnUFNB'
    'dFgyWnVMbkJoY21GdE93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ1pXeHpaU0JwWmlBb1gyWnVMbVZtWm1WamRDQTlQU0F3ZURVcElIc0tJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0JmWTJ4aGJYQlViMVJuZENBOUlIUnlkV1U3Q2lBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ2FXWWdLRjl3VTNSaGNuUlNiM2NnUENCMFlYSm5aWFJRWlhKcGIyUXBJQ0FnSUNB'
    'Z1gzTjBaWEFnUFNCZmJHRnpkRlJRVW1GMFpUc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Js'
    'YkhObElHbG1JQ2hmY0ZOMFlYSjBVbTkzSUQ0Z2RHRnlaMlYwVUdWeWFXOWtLU0JmYzNSbGNDQTlJ'
    'QzFmYkdGemRGUlFVbUYwWlRzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUM4dklGOW1kV3hzSUNzZ01TQTlJSFJ2ZEdGc0lIUnBZMnR6SUdsdUlIUm9hWE1nY205'
    'M0lDaHpjR1ZsWkNrS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsdWRDQmZkR2xqYTNOZmFXNWZjbTkz'
    'SUQwZ1gyWjFiR3dnS3lBeE93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1ptOXlJQ2hwYm5RZ1gzUWdQ'
    'U0F3T3lCZmRDQThJRE15T3lCZmRDc3JLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FX'
    'WWdLRjkwSUQ0OUlGOTBhV05yYzE5cGJsOXliM2NwSUdKeVpXRnJPd29nSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUdsbUlDaGZVSFFnUGlBd0xqQXBJRjltVTJGdGNHeGxVRzl6UVdOaklDczlJRjlE'
    'WmlBcUlGOWtkQ0F2SUY5UWREc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0F2THlCVmNHUmhk'
    'R1VnY0dWeWFXOWtJR1p2Y2lCdVpYaDBJSFJwWTJzZ0tHOXViSGtnYVdZZ2RDQThJRjltZFd4c0xD'
    'QnBMbVV1SUhSb1pYSmxJR0Z5WlFvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDOHZJRzF2Y21V'
    'Z2RHbGphM01nZEc4Z2MyeHBaR1VnYVc0Z2RHaHBjeUJ5YjNjcENpQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdhV1lnS0Y5emRHVndJQ0U5SURBZ0ppWWdYM1FnUENCZlpuVnNiQ2tnZXdvZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JtYkc5aGRDQmZVRzRnUFNCZlVIUWdLeUJtYkc5aGRD'
    'aGZjM1JsY0NrN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsbUlDaGZZMnhoYlhC'
    'VWIxUm5kQ2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjl6'
    'ZEdWd0lENGdNQ2tnSUNBZ0lDQmZVRzRnUFNCdGFXNG9YMUJ1TENCMFlYSm5aWFJRWlhKcGIyUXBP'
    'd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnWld4elpTQnBaaUFvWDNOMFpY'
    'QWdQQ0F3S1NCZlVHNGdQU0J0WVhnb1gxQnVMQ0IwWVhKblpYUlFaWEpwYjJRcE93b2dJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCOUlHVnNjMlVnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ1gxQnVJRDBnWTJ4aGJYQW9YMUJ1TENBeE1UTXVNQ3dnT0RVMkxqQXBP'
    'd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lGOVFkQ0E5SUY5UWJqc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0I5Q2lB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0Fn'
    'WDJaeUt5czdDaUFnSUNBZ0lDQWdJQ0FnSUM4dklFRmtkbUZ1WTJVZ2RHOGdibVY0ZENCemIyNW5J'
    'SEJ2YzJsMGFXOXVJSGRvWlc0Z2QyVW5kbVVnWlhob1lYVnpkR1ZrSUhSb2FYTWdjR0YwZEdWeWJp'
    'ZHpJSEp2ZDNNS0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5bWNpQStQU0J3WVhSVGRHRnlkRkp2ZDF0'
    'ZlpuQmRJQ3NnS0hCaGRGSnZkMDltWm5ObGRGdGZabkFyTVYwZ0xTQndZWFJTYjNkUFptWnpaWFJi'
    'WDJad1hTa3BJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJRjltY0Nzck93b2dJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ1gyWnlJRDBnS0Y5bWNDQThJRk5QVGtkZlRFVk9SMVJJS1NBL0lIQmhkRk4wWVhKMFVt'
    'OTNXMTltY0YwZ09pQXdPd29nSUNBZ0lDQWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ2ZRb0tJQ0FnSUNB'
    'Z0lDQXZMeUJEZFhKeVpXNTBJSEp2ZHlCd1lYSjBhV0ZzSUNodWIyNHRkSEpwWjJkbGNpQnliM2Nn'
    'YjI1c2VTRGlnSlFnZEhKcFoyZGxjaUJvWVc1a2JHVmtJR0ZpYjNabEtRb2dJQ0FnSUNBZ0lHbG1J'
    'Q2hmY0dOeUxtbHVjM1J5ZFcxbGJuUWdQRDBnTUNBbUppQmZjR055TG5CbGNtbHZaQ0E4UFNBd0tT'
    'QjdDaUFnSUNBZ0lDQWdJQ0FnSUM4dklGUnBZMnNnYzNSaGJYQWdabTl5SUdOMWNuSmxiblF0Y205'
    'M0lIQmhjblJwWVd3Z2RtOXNJR1ZtWm1WamRITTZJSFJvWlNCamRYSnlaVzUwSUhKdmR5ZHpDaUFn'
    'SUNBZ0lDQWdJQ0FnSUM4dklITjBZWEowSUhScFkyc3VJRlJvWlNBMk5DMXpZVzF3YkdVZ2NtRnRj'
    'Q0JqYjIxd2JHVjBaWE1nZDJWc2JDQjNhWFJvYVc0Z2RHaGxJR1pwY25OMENpQWdJQ0FnSUNBZ0lD'
    'QWdJQzh2SUhScFkyc3NJSE52SUdGdWVTQjJiMndnWTJoaGJtZGxJQ0pvWVhCd1pXNXBibWNnWVhR'
    'Z2RHaHBjeUJ5YjNjaUlISmxZV1J6SUdGeklHaGhkbWx1WndvZ0lDQWdJQ0FnSUNBZ0lDQXZMeUJ5'
    'WVcxd1pXUWdkRzhnYVhSeklHWnBibUZzSUhaaGJIVmxJR0ZzYlc5emRDQnBiVzFsWkdsaGRHVnNl'
    'UzRLSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjlqZFhKVFozSWdQU0J3WVhSVWFXTnJUMlptYzJWMFcz'
    'QnZjeTV6YjI1blVHOXpYU0FySUNod2IzTXVjbTkzSUMwZ2NHRjBVM1JoY25SU2IzZGJjRzl6TG5O'
    'dmJtZFFiM05kS1RzS0lDQWdJQ0FnSUNBZ0lDQWdabXh2WVhRZ1gyTjFjbFJwWTJ0R0lEMGdabXh2'
    'WVhRb1ptVjBZMmhVYVdOcktGOWpkWEpUWjNJcEtUc0tJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOXdZ'
    'M0l1WldabVpXTjBJRDA5SURCNFF5a0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lGWlBURjlUUlZRb2JX'
    'bHVLRjl3WTNJdWNHRnlZVzBzSURZMEtTd2dYMk4xY2xScFkydEdLVHNLSUNBZ0lDQWdJQ0FnSUNB'
    'Z1pXeHpaU0JwWmlBb1gzQmpjaTVsWm1abFkzUWdQVDBnTUhoQklIeDhJRjl3WTNJdVpXWm1aV04w'
    'SUQwOUlEQjROaWtnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOTJkU0E5SUNoZmNHTnlM'
    'bkJoY21GdFBqNDBLU1l3ZUVZc0lGOTJaQ0E5SUY5d1kzSXVjR0Z5WVcwbU1IaEdPd29nSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdWazlNWDFORlZDaGpiR0Z0Y0NoZmRtOXNRM1Z5Y2lBcklDaGZkblUrTUQ5'
    'ZmRuVTZMVjkyWkNrZ0tpQmZjR04wTENBd0xDQTJOQ2tzSUY5amRYSlVhV05yUmlrN0NpQWdJQ0Fn'
    'SUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ1pXeHpaU0JwWmlBb1gzQmpjaTVsWm1abFkzUWdQ'
    'VDBnTUhoRktTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBiblFnWDJWeklEMGdLRjl3WTNJdWNH'
    'RnlZVzBnUGo0Z05Da2dKaUF3ZUVZN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcGJuUWdYMlYySUQw'
    'Z0lGOXdZM0l1Y0dGeVlXMGdJQ0FnSUNBZ0ppQXdlRVk3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Jw'
    'WmlBb1gyVnpJRDA5SURCNFFTa2dJQ0FnSUNCV1QweGZVMFZVS0dOc1lXMXdLRjkyYjJ4RGRYSnlJ'
    'Q3NnWDJWMkxDQXdMQ0EyTkNrc0lGOWpkWEpVYVdOclJpazdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QmxiSE5sSUdsbUlDaGZaWE1nUFQwZ01IaENLU0JXVDB4ZlUwVlVLR05zWVcxd0tGOTJiMnhEZFhK'
    'eUlDMGdYMlYyTENBd0xDQTJOQ2tzSUY5amRYSlVhV05yUmlrN0NpQWdJQ0FnSUNBZ0lDQWdJSDBL'
    'SUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjl3WTNJdVpXWm1aV04wSUQwOUlEQjROU2tnZXdvZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnTHk4Z01IZzFJSFp2YkMxemJHbGtaU0J3YjNKMGFXOXVJRzl1SUdOMWNu'
    'SmxiblFnY205M0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcGJuUWdYM1oxSUQwZ0tGOXdZM0l1Y0dG'
    'eVlXMCtQalFwSmpCNFJpd2dYM1prSUQwZ1gzQmpjaTV3WVhKaGJTWXdlRVk3Q2lBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0JXVDB4ZlUwVlVLR05zWVcxd0tGOTJiMnhEZFhKeUlDc2dLRjkyZFQ0d1AxOTJk'
    'VG90WDNaa0tTQXFJRjl3WTNRc0lEQXNJRFkwS1N3Z1gyTjFjbFJwWTJ0R0tUc0tJQ0FnSUNBZ0lD'
    'QWdJQ0FnZlFvZ0lDQWdJQ0FnSUgwS0lDQWdJSDBLQ2lBZ0lDQXZMeUJEZFhKeVpXNTBJSEp2ZHlC'
    'd1lYSjBhV0ZzSUhCcGRHTm9JR1ZtWm1WamRDQW9ZWEJ3YkdsbGN5QmxkbVZ1SUc5dUlIUnlhV2Ru'
    'WlhJZ2NtOTNLUzRLSUNBZ0lDOHZJRlZ6WlNCamIyNTBhVzUxYjNWeklIQnZjeTUwYVdOcklPS0Fs'
    'Q0JpZFhRZ1kyRndJR2wwSUdGMElDaHpjR1ZsWkMweEtTQnpieUIwYUdVZ1kyOXVkSEpwWW5WMGFX'
    'OXVDaUFnSUNBdkx5QmhkQ0IwYUdVZ2JHRnpkQ0J6WVcxd2JHVWdiMllnZEdocGN5QnliM2NnWlho'
    'aFkzUnNlU0J0WVhSamFHVnpJSGRvWVhRZ2RHaGxJR1p2Y25kaGNtUWdjMk5oYmdvZ0lDQWdMeThn'
    'ZDJsc2JDQjFjMlVnWm05eUlIUm9hWE1nY205M0lHOXVZMlVnYVhRZ1ltVmpiMjFsY3lCaElDSmpi'
    'MjF3YkdWMFpXUWlJSEp2ZHk0Z0lGZHBkR2h2ZFhRZ2RHaGxDaUFnSUNBdkx5QmpZWEFzSUhCdmN5'
    'NTBhV05ySUdGd2NISnZZV05vWlhNZ1lITndaV1ZrWUNCaGRDQjBhR1VnY205M0lHSnZkVzVrWVhK'
    'NUlIZG9hV3hsSUhSb1pTQm1iM0ozWVhKa0NpQWdJQ0F2THlCelkyRnVJSFZ6WlhNZ1lITndaV1Zr'
    'TFRGZ0xDQndjbTlrZFdOcGJtY2dZU0IrTVMxMGFXTnJJR0poWTJ0M1lYSmtJSEJsY21sdlpDQnFk'
    'VzF3SUQwZ1kyeHBZMnN1Q2lBZ0lDQXZMeUJQYm14NUlIQmhlU0IwYUdVZ1ptVjBZMmhVYVdOcklH'
    'TnZjM1FnZDJobGJpQmhJSEJwZEdOb0lHVm1abVZqZENCcGN5QmhZM1IxWVd4c2VTQndjbVZ6Wlc1'
    'MExnb2dJQ0FnTHk4Z1UyRjJaU0JsWm1abFkzUnBkbVZRWlhKcGIyUWdZWFFnYzNSaGNuUWdiMlln'
    'WTNWeWNtVnVkQ0J5YjNjc0lFSkZSazlTUlNCd1lYSjBhV0ZzSUhCcGRHTm9DaUFnSUNBdkx5Qmxa'
    'bVpsWTNRZ1lYQndiR2xqWVhScGIyNGc0b0NVSUc1bFpXUmxaQ0JtYjNJZ2RHaGxJR04xY25KbGJu'
    'UXRjbTkzSUdobFlXUWdZMjl1ZEhKcFluVjBhVzl1SUhSdkNpQWdJQ0F2THlCZlpsTmhiWEJzWlZC'
    'dmMwRmpZeUJpWld4dmR5NEtJQ0FnSUdac2IyRjBJRjl3VTNSaGNuUkRkWElnUFNCbFptWmxZM1Jw'
    'ZG1WUVpYSnBiMlE3Q2dvZ0lDQWdMeThnUTNWeWNtVnVkQ0J5YjNjZ2NHRnlkR2xoYkNCd2FYUmph'
    'Q0JsWm1abFkzUWdLR0Z3Y0d4cFpYTWdiMjRnZEhKcFoyZGxjaUJ5YjNjZ1QxSWdZMjl1ZEdsdWRX'
    'RjBhVzl1SUhKdmR5a3VDaUFnSUNBdkx5QkdiM0lnTVhoNEx6SjRlRG9nWVd4M1lYbHpJR0Z3Y0d4'
    'cFpYTWdiMjRnWTNWeWNtVnVkQ0J5YjNjdUNpQWdJQ0F2THlCR2IzSWdNM2g0THpWNGVEb2dZWEJ3'
    'YkdsbGN5QjNhR1Z1WlhabGNpQmpkWEp5Wlc1MElISnZkeUJqWVhKeWFXVnpJSFJvWlNCbFptWmxZ'
    'M1FnNG9DVUlHSnZkR2dLSUNBZ0lDOHZJQ0FnWTI5dWRHbHVkV0YwYVc5dUlISnZkM01nS0hCbGNt'
    'bHZaRDA5TUNrZ1FVNUVJSFJ2Ym1VdGNHOXlkR0VnZEdGeVoyVjBJSEp2ZDNNZ0tIQmxjbWx2WkQ0'
    'd0tTNEtJQ0FnSUM4dklDQWdWR2hsSUhSaGNtZGxkRkJsY21sdlpDQjNZWE1nWVd4eVpXRmtlU0J6'
    'WlhRZ1lXSnZkbVVnS0dWcGRHaGxjaUJtY205dElIUnlhV2RPYjNSbElHOXlDaUFnSUNBdkx5QWdJ'
    'SFJ2Ym1WVGJHbGtaVlJoY21kbGRDazdJSFJvYVhNZ1lteHZZMnNnWkc5bGN5QjBhR1VnY0dWeUxY'
    'UnBZMnNnWVdOamRXMTFiR0YwYVc5dUlIUnZkMkZ5WkNCcGRDNEtJQ0FnSUdsbUlDaGZjR055TG1W'
    'bVptVmpkQ0E5UFNBd2VERWdmSHdnWDNCamNpNWxabVpsWTNRZ1BUMGdNSGd5SUh4OENpQWdJQ0Fn'
    'SUNBZ1gzQmpjaTVsWm1abFkzUWdQVDBnTUhneklIeDhJRjl3WTNJdVpXWm1aV04wSUQwOUlEQjRO'
    'U2tnZXdvZ0lDQWdJQ0FnSUdsdWRDQmZjMmR5WDJOMWNpQTlJSEJoZEZScFkydFBabVp6WlhSYmNH'
    'OXpMbk52Ym1kUWIzTmRJQ3NnS0hCdmN5NXliM2NnTFNCd1lYUlRkR0Z5ZEZKdmQxdHdiM011YzI5'
    'dVoxQnZjMTBwT3dvZ0lDQWdJQ0FnSUdac2IyRjBJRjl3ZEdZZ1BTQnRhVzRvY0c5ekxuUnBZMnNz'
    'Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JtYkc5aGRDaG1aWFJqYUZScFkyc29Y'
    'M05uY2w5amRYSWdLeUF4S1NBdElHWmxkR05vVkdsamF5aGZjMmR5WDJOMWNpa2dMU0F4S1NrN0Np'
    'QWdJQ0FnSUNBZ2FXWWdLRjl3WTNJdVpXWm1aV04wSUQwOUlEQjRNU2tLSUNBZ0lDQWdJQ0FnSUNB'
    'Z1pXWm1aV04wYVhabFVHVnlhVzlrSUQwZ2JXRjRLREV4TXk0d0xDQmxabVpsWTNScGRtVlFaWEpw'
    'YjJRZ0xTQm1iRzloZENoZmNHTnlMbkJoY21GdEtTQXFJRjl3ZEdZcE93b2dJQ0FnSUNBZ0lHVnNj'
    'MlVnYVdZZ0tGOXdZM0l1WldabVpXTjBJRDA5SURCNE1pa0tJQ0FnSUNBZ0lDQWdJQ0FnWldabVpX'
    'TjBhWFpsVUdWeWFXOWtJRDBnYldsdUtEZzFOaTR3TENCbFptWmxZM1JwZG1WUVpYSnBiMlFnS3lC'
    'bWJHOWhkQ2hmY0dOeUxuQmhjbUZ0S1NBcUlGOXdkR1lwT3dvZ0lDQWdJQ0FnSUdWc2MyVWdhV1ln'
    'S0Y5d1kzSXVaV1ptWldOMElEMDlJREI0TXlrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0F2THlCVWIyNWxJ'
    'SEJ2Y25SaElPS0FsQ0IxYzJWeklHbDBjeUJ2ZDI0Z2NHRnlZVzBnWVhNZ2MyeHBaR1VnY21GMFpR'
    'b2dJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1pXWm1aV04wYVhabFVHVnlhVzlrSUR3Z2RHRnlaMlYwVUdW'
    'eWFXOWtLUW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdaV1ptWldOMGFYWmxVR1Z5YVc5a0lEMGdiV2x1'
    'S0hSaGNtZGxkRkJsY21sdlpDd2daV1ptWldOMGFYWmxVR1Z5YVc5a0lDc2dabXh2WVhRb1gzQmpj'
    'aTV3WVhKaGJTa2dLaUJmY0hSbUtUc0tJQ0FnSUNBZ0lDQWdJQ0FnWld4elpTQnBaaUFvWldabVpX'
    'TjBhWFpsVUdWeWFXOWtJRDRnZEdGeVoyVjBVR1Z5YVc5a0tRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z1pXWm1aV04wYVhabFVHVnlhVzlrSUQwZ2JXRjRLSFJoY21kbGRGQmxjbWx2WkN3Z1pXWm1aV04w'
    'YVhabFVHVnlhVzlrSUMwZ1pteHZZWFFvWDNCamNpNXdZWEpoYlNrZ0tpQmZjSFJtS1RzS0lDQWdJ'
    'Q0FnSUNCOUNpQWdJQ0FnSUNBZ1pXeHpaU0I3SUNBdkx5QXdlRFVnNG9DVUlHTnZiblJwYm5WbElI'
    'UnZibVVnY0c5eWRHRWdkWE5wYm1jZ2JHRnpkQ0F6ZUhnZ2NtRjBaU0FvY0dGeVlXMGdhWE1nZG05'
    'c0xYTnNhV1JsSUc5dWJIa3BDaUFnSUNBZ0lDQWdJQ0FnSUdsbUlDaGZiR0Z6ZEZSUVVtRjBaU0Er'
    'SURBcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hsWm1abFkzUnBkbVZRWlhKcGIyUWdQ'
    'Q0IwWVhKblpYUlFaWEpwYjJRcENpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdaV1ptWldOMGFY'
    'WmxVR1Z5YVc5a0lEMGdiV2x1S0hSaGNtZGxkRkJsY21sdlpDd2daV1ptWldOMGFYWmxVR1Z5YVc5'
    'a0lDc2dabXh2WVhRb1gyeGhjM1JVVUZKaGRHVXBJQ29nWDNCMFppazdDaUFnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQmxiSE5sSUdsbUlDaGxabVpsWTNScGRtVlFaWEpwYjJRZ1BpQjBZWEpuWlhSUVpYSnBi'
    'MlFwQ2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1pXWm1aV04wYVhabFVHVnlhVzlrSUQwZ2JX'
    'RjRLSFJoY21kbGRGQmxjbWx2WkN3Z1pXWm1aV04wYVhabFVHVnlhVzlrSUMwZ1pteHZZWFFvWDJ4'
    'aGMzUlVVRkpoZEdVcElDb2dYM0IwWmlrN0NpQWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0I5'
    'Q2lBZ0lDQjlDZ29nSUNBZ0x5OGc0cFNBNHBTQTRwU0FJRU4xY25KbGJuUXRjbTkzSUdobFlXUWdZ'
    'Mjl1ZEhKcFluVjBhVzl1SUhacFlTQndaWEl0ZEdsamF5QnBiblJsWjNKaGRHbHZiaURpbElEaWxJ'
    'RGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJQUtJQ0FnSUM4dklGTmhiV1VnY0dW'
    'eUxYUnBZMnNnYlc5a1pXd2dZWE1nWm05eWQyRnlaQ0J6WTJGdUxpQlhaU2R5WlNCaGRDQndiM011'
    'ZEdsamF5QjNhWFJvYVc0Z2RHaGxDaUFnSUNBdkx5QmpkWEp5Wlc1MElISnZkeTRnVUdWeWFXOWtJ'
    'R0YwSUhScFkyc2dNQ0E5SUY5d1UzUmhjblJEZFhJdUlGZGxJR2x1ZEdWbmNtRjBaU0IwYVdOcmN3'
    'b2dJQ0FnTHk4Z1d6QXNJSEJ2Y3k1MGFXTnJLU0RpZ0pRZ2FTNWxMaXdnZDJVbmNtVWdJbUpsWm05'
    'eVpTSWdkR2hsSUdKdmRXNWtZWEo1SUdGMElHVnVaQ0J2WmlCMGFXTnJJR1pzYjI5eUtIQnZjeTUw'
    'YVdOcktTNEtJQ0FnSUM4dklFWnZjaUIwYUdVZ1puSmhZM1JwYjI1aGJDQnpkV0l0ZEdsamF5QW9Z'
    'bVYwZDJWbGJpQnBiblJsWjJWeUlIUnBZMnNnWldSblpYTXBMQ0JoWkdRS0lDQWdJQzh2SUhCaGNu'
    'UnBZV3d0ZEdsamF5QmpiMjUwY21saWRYUnBiMjRnWVhRZ2RHaGxJSEJsY21sdlpDQmpkWEp5Wlc1'
    'MGJIa2dhVzRnWldabVpXTjBMZ29nSUNBZ2V3b2dJQ0FnSUNBZ0lHWnNiMkYwSUY5RFpsOW9JRDBn'
    'WXpSemNHVmxaSE5iYzIxd0xtWnBibVYwZFc1bElDWWdNSGhHWFNBcUlEUXlPQzR3T3dvZ0lDQWdJ'
    'Q0FnSUdac2IyRjBJRjlrZENBZ0lEMGdNUzR3SUM4Z1ZFbERTMU5mVUVWU1gxTkZRenNLSUNBZ0lD'
    'QWdJQ0JtYkc5aGRDQmZVSFFnSUNBOUlGOXdVM1JoY25SRGRYSTdDaUFnSUNBZ0lDQWdhVzUwSUY5'
    'emRHVndJRDBnTURzS0lDQWdJQ0FnSUNCaWIyOXNJRjlqYkdGdGNGUnZWR2QwSUQwZ1ptRnNjMlU3'
    'Q2lBZ0lDQWdJQ0FnYVdZZ0tGOXdZM0l1WldabVpXTjBJRDA5SURCNE1Ta2dYM04wWlhBZ1BTQXRY'
    'M0JqY2k1d1lYSmhiVHNLSUNBZ0lDQWdJQ0JsYkhObElHbG1JQ2hmY0dOeUxtVm1abVZqZENBOVBT'
    'QXdlRElwSUY5emRHVndJRDBnWDNCamNpNXdZWEpoYlRzS0lDQWdJQ0FnSUNCbGJITmxJR2xtSUNo'
    'ZmNHTnlMbVZtWm1WamRDQTlQU0F3ZURNcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnWDJOc1lXMXdWRzlV'
    'WjNRZ1BTQjBjblZsT3dvZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvWDNCVGRHRnlkRU4xY2lBOElIUmhj'
    'bWRsZEZCbGNtbHZaQ2tnSUNBZ0lDQmZjM1JsY0NBOUlGOXdZM0l1Y0dGeVlXMDdDaUFnSUNBZ0lD'
    'QWdJQ0FnSUdWc2MyVWdhV1lnS0Y5d1UzUmhjblJEZFhJZ1BpQjBZWEpuWlhSUVpYSnBiMlFwSUY5'
    'emRHVndJRDBnTFY5d1kzSXVjR0Z5WVcwN0NpQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lHVnNjMlVn'
    'YVdZZ0tGOXdZM0l1WldabVpXTjBJRDA5SURCNE5Ta2dld29nSUNBZ0lDQWdJQ0FnSUNCZlkyeGhi'
    'WEJVYjFSbmRDQTlJSFJ5ZFdVN0NpQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZmNGTjBZWEowUTNWeUlE'
    'd2dkR0Z5WjJWMFVHVnlhVzlrS1NBZ0lDQWdJRjl6ZEdWd0lEMGdYMnhoYzNSVVVGSmhkR1U3Q2lB'
    'Z0lDQWdJQ0FnSUNBZ0lHVnNjMlVnYVdZZ0tGOXdVM1JoY25SRGRYSWdQaUIwWVhKblpYUlFaWEpw'
    'YjJRcElGOXpkR1Z3SUQwZ0xWOXNZWE4wVkZCU1lYUmxPd29nSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJ'
    'Q0JwYm5RZ1gyWjFiR3hmZEdsamEzTWdQU0JwYm5Rb2NHOXpMblJwWTJzcE93b2dJQ0FnSUNBZ0lH'
    'WnNiMkYwSUY5bWNtRmpJQ0FnSUNBOUlIQnZjeTUwYVdOcklDMGdabXh2WVhRb1gyWjFiR3hmZEds'
    'amEzTXBPd29nSUNBZ0lDQWdJR1p2Y2lBb2FXNTBJRjkwSUQwZ01Ec2dYM1FnUENBek1qc2dYM1Fy'
    'S3lrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1gzUWdQajBnWDJaMWJHeGZkR2xqYTNNcElHSnla'
    'V0ZyT3dvZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvWDFCMElENGdNQzR3S1NCZlpsTmhiWEJzWlZCdmMw'
    'RmpZeUFyUFNCZlEyWmZhQ0FxSUY5a2RDQXZJRjlRZERzS0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5'
    'emRHVndJQ0U5SURBcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHWnNiMkYwSUY5UWJpQTlJRjlR'
    'ZENBcklHWnNiMkYwS0Y5emRHVndLVHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZlkyeGhi'
    'WEJVYjFSbmRDa2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsbUlDaGZjM1JsY0NBK0lE'
    'QXBJQ0FnSUNBZ1gxQnVJRDBnYldsdUtGOVFiaXdnZEdGeVoyVjBVR1Z5YVc5a0tUc0tJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0JsYkhObElHbG1JQ2hmYzNSbGNDQThJREFwSUY5UWJpQTlJRzFo'
    'ZUNoZlVHNHNJSFJoY21kbGRGQmxjbWx2WkNrN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCOUlHVnNj'
    'MlVnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lGOVFiaUE5SUdOc1lXMXdLRjlRYml3Z01U'
    'RXpMakFzSURnMU5pNHdLVHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJRjlRZENBOUlGOVFianNLSUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lIMEtJQ0Fn'
    'SUNBZ0lDQXZMeUJUZFdJdGRHbGpheUJtY21GamRHbHZibUZzSUdOdmJuUnlhV0oxZEdsdmJpQmhk'
    'Q0JqZFhKeVpXNTBJSEJsY21sdlpBb2dJQ0FnSUNBZ0lHbG1JQ2hmWm5KaFl5QStJREF1TUNBbUpp'
    'QmZVSFFnUGlBd0xqQXBDaUFnSUNBZ0lDQWdJQ0FnSUY5bVUyRnRjR3hsVUc5elFXTmpJQ3M5SUY5'
    'RFpsOW9JQ29nWDJSMElDb2dYMlp5WVdNZ0x5QmZVSFE3Q2lBZ0lDQjlDZ29nSUNBZ0x5OGdVM0Js'
    'WTJsaGJDQmpZWE5sT2lCMGNtbG5aMlZ5SUhKdmR5QkpVeUJqZFhKeVpXNTBJSEp2ZHlBb2JtOGdj'
    'MlZ3WVhKaGRHVWdkSEpwWjJkbGNpQjBZV2xzSUM4Z2MyTmhiaWt1Q2lBZ0lDQXZMeUJVYUdVZ1lX'
    'TmpkVzExYkdGMGIzSWdZV0p2ZG1VZ2FHRnVaR3hsWkNCb1pXRmtJR1p5YjIwZ2RHbGpheUF3SUhS'
    'dklIQnZjeTUwYVdOcklHRjBJR052Ym5OMFlXNTBDaUFnSUNBdkx5QjBjbWxuVG05MFpTNXdaWEpw'
    'YjJRZ0tITnBibU5sSUY5d1UzUmhjblJEZFhJZ2QyRnpJSE5sZENCMGJ5QmxabVpsWTNScGRtVlFa'
    'WEpwYjJRZ2QyaHBZMmdnYjI0S0lDQWdJQzh2SUhSb2FYTWdZMjlrWlNCd1lYUm9JR1Z4ZFdGc2N5'
    'QjBjbWxuVG05MFpTNXdaWEpwYjJRZzRvQ1VJSFJvWlNCd1lYSjBhV0ZzTFhCcGRHTm9JR0pzYjJO'
    'cklHUnBaRzRuZEFvZ0lDQWdMeThnY25WdUlHbG1JRzV2SUhOc2FXUmxJR1ZtWm1WamRDd2diM0ln'
    'YVhRZ2NtRnVJR1p5YjIwZ2RISnBaMDV2ZEdVdWNHVnlhVzlrSUdGeklITjBZWEowYVc1bklIQnZh'
    'VzUwS1M0S0lDQWdJQzh2SUU1dklHRmtaR2wwYVc5dVlXd2dZMjlrWlNCdVpXVmtaV1E2SUdGalkz'
    'VnRkV3hoZEc5eUlHbHpJR052Y25KbFkzUXVDZ29nSUNBZ0x5OGdWSEpsYlc5c2J5QW9SV1ptWldO'
    'MElEQjROeWtnNG9DVUlITmhiV1VnZDJGMlpXWnZjbTBnWVhNZ2RtbGljbUYwYnlCaWRYUWdiVzlr'
    'ZFd4aGRHVnpJRlpQVEZWTlJTNEtJQ0FnSUM4dklGVnpaWE1nYzJGdFpTQnliM2N0WW5rdGNtOTNJ'
    'R2hwYzNSdmNtbGpZV3dnZEZNdmRFUWdkSEpoWTJ0cGJtY2dZWE1nZG1saWNtRjBieTRLSUNBZ0lI'
    'c0tJQ0FnSUNBZ0lDQnBiblFnWDNSVElEMGdNQ3dnWDNSRUlEMGdNRHNLSUNBZ0lDQWdJQ0JwYm5R'
    'Z1gzUnlaVkJ2Y3lBOUlEQTdDaUFnSUNBZ0lDQWdhV1lnS0hSeWFXZE9iM1JsTG1WbVptVmpkQ0E5'
    'UFNBd2VEY3BJSHNLSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjl1Y3lBOUlDaDBjbWxuVG05MFpTNXdZ'
    'WEpoYlNBK1BpQTBLU0FtSURCNFJqc0tJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOXVaQ0E5SUNCMGNt'
    'bG5UbTkwWlM1d1lYSmhiU0FnSUNBZ0lDQW1JREI0UmpzS0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5'
    'dWN5QStJREFwSUY5MFV5QTlJRjl1Y3pzS0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5dVpDQStJREFw'
    'SUY5MFJDQTlJRjl1WkRzS0lDQWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ2FXWWdLSFJ5YVdkUVlYUWdQ'
    'VDBnY0c5ekxuTnZibWRRYjNNZ0ppWWdkSEpwWjFKdmR5QTlQU0J3YjNNdWNtOTNLU0I3Q2lBZ0lD'
    'QWdJQ0FnSUNBZ0lGOTBjbVZRYjNNZ1BTQnBiblFvY0c5ekxuUnBZMnNwSUNvZ1gzUlRPd29nSUNB'
    'Z0lDQWdJSDBnWld4elpTQjdDaUFnSUNBZ0lDQWdJQ0FnSUdsdWRDQmZkSEpUWjNJZ1BTQndZWFJV'
    'YVdOclQyWm1jMlYwVzNSeWFXZFFZWFJkSUNzZ0tIUnlhV2RTYjNjZ0xTQndZWFJUZEdGeWRGSnZk'
    'MXQwY21sblVHRjBYU2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0JmZEhKVGNHUWdQU0JtWlhSamFG'
    'UnBZMnNvWDNSeVUyZHlJQ3NnTVNrZ0xTQm1aWFJqYUZScFkyc29YM1J5VTJkeUtUc0tJQ0FnSUNB'
    'Z0lDQWdJQ0FnWDNSeVpWQnZjeUE5SUNoZmRISlRjR1FnTFNBeEtTQXFJRjkwVXpzS0lDQWdJQ0Fn'
    'SUNBZ0lDQWdhVzUwSUY5M2NDQTlJSFJ5YVdkUVlYUXNJRjkzY2lBOUlIUnlhV2RTYjNjZ0t5QXhP'
    'd29nSUNBZ0lDQWdJQ0FnSUNCcFppQW9YM2R5SUQ0OUlIQmhkRk4wWVhKMFVtOTNXMTkzY0YwZ0t5'
    'QW9jR0YwVW05M1QyWm1jMlYwVzE5M2NDc3hYU0F0SUhCaGRGSnZkMDltWm5ObGRGdGZkM0JkS1Nr'
    'Z2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1gzZHdLeXM3SUY5M2NpQTlJQ2hmZDNBZ1BDQlRUMDVI'
    'WDB4RlRrZFVTQ2tnUHlCd1lYUlRkR0Z5ZEZKdmQxdGZkM0JkSURvZ01Ec0tJQ0FnSUNBZ0lDQWdJ'
    'Q0FnZlFvZ0lDQWdJQ0FnSUNBZ0lDQm1iM0lnS0dsdWRDQmZkMmtnUFNBd095QmZkMmtnUENBeE1q'
    'ZzdJRjkzYVNzcktTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvWDNkd0lENGdjRzl6TG5O'
    'dmJtZFFiM01nZkh3Z0tGOTNjQ0E5UFNCd2IzTXVjMjl1WjFCdmN5QW1KaUJmZDNJZ1BqMGdjRzl6'
    'TG5KdmR5a3BJR0p5WldGck93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjkzY0NBK1BTQlRU'
    'MDVIWDB4RlRrZFVTQ2tnWW5KbFlXczdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvWDNkeUlE'
    'NDlJSEJoZEZOMFlYSjBVbTkzVzE5M2NGMGdLeUFvY0dGMFVtOTNUMlptYzJWMFcxOTNjQ3N4WFNB'
    'dElIQmhkRkp2ZDA5bVpuTmxkRnRmZDNCZEtTa2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUY5M2NDc3JPeUJmZDNJZ1BTQW9YM2R3SUR3Z1UwOU9SMTlNUlU1SFZFZ3BJRDhnY0dGMFUzUmhj'
    'blJTYjNkYlgzZHdYU0E2SURBN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdZMjl1ZEdsdWRX'
    'VTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQk9iM1JsSUY5'
    'MGJpQTlJR2RsZEU1dmRHVW9YM2R3TENCZmQzSXNJR05vS1RzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUdKdmIyd2dYM1J1U1hOVWIyNWxJRDBnS0NoZmRHNHVaV1ptWldOMElEMDlJREI0TXlCOGZDQmZk'
    'RzR1WldabVpXTjBJRDA5SURCNE5Ta2dKaVlnWDNSdUxuQmxjbWx2WkNBK0lEQXBPd29nSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5MGJpNXdaWEpwYjJRZ1BpQXdJQ1ltSUNGZmRHNUpjMVJ2Ym1V'
    'cElHSnlaV0ZyT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOTBiaTVsWm1abFkzUWdQVDBn'
    'TUhnM0lDWW1JRjkwYmk1d1lYSmhiU0FoUFNBd0tTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnYVc1MElGOXVjeUE5SUNoZmRHNHVjR0Z5WVcwZ1BqNGdOQ2tnSmlBd2VFWTdDaUFnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOXVaQ0E5SUNCZmRHNHVjR0Z5WVcwZ0lDQWdJQ0FnSmlB'
    'd2VFWTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOXVjeUErSURBcElGOTBVeUE5'
    'SUY5dWN6c0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1gyNWtJRDRnTUNrZ1gzUkVJ'
    'RDBnWDI1a093b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FX'
    'NTBJRjl6WjNJZ1BTQndZWFJVYVdOclQyWm1jMlYwVzE5M2NGMGdLeUFvWDNkeUlDMGdjR0YwVTNS'
    'aGNuUlNiM2RiWDNkd1hTazdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBiblFnWDNOd1pDQTlJR1ps'
    'ZEdOb1ZHbGpheWhmYzJkeUlDc2dNU2tnTFNCbVpYUmphRlJwWTJzb1gzTm5jaWs3Q2lBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0JmZEhKbFVHOXpJQ3M5SUNoZmMzQmtJQzBnTVNrZ0tpQmZkRk03Q2lBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0JmZDNJckt6c0tJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNB'
    'Z0lDQXZMeUJWY0dSaGRHVWdabkp2YlNCamRYSnlaVzUwSUhKdmR5QnBaaUJwZENCallYSnlhV1Z6'
    'SUhSeVpXMXZiRzhLSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjl3WTNJdVpXWm1aV04wSUQwOUlEQjRO'
    'eUFtSmlCZmNHTnlMbkJoY21GdElDRTlJREFwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsdWRD'
    'QmZibk1nUFNBb1gzQmpjaTV3WVhKaGJTQStQaUEwS1NBbUlEQjRSanNLSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJR2x1ZENCZmJtUWdQU0FnWDNCamNpNXdZWEpoYlNBZ0lDQWdJQ0FtSURCNFJqc0tJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmYm5NZ1BpQXdLU0JmZEZNZ1BTQmZibk03Q2lBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0JwWmlBb1gyNWtJRDRnTUNrZ1gzUkVJRDBnWDI1a093b2dJQ0FnSUNBZ0lD'
    'QWdJQ0I5Q2lBZ0lDQWdJQ0FnSUNBZ0lGOTBjbVZRYjNNZ0t6MGdhVzUwS0hCdmN5NTBhV05yS1NB'
    'cUlGOTBVenNLSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0FnYVdZZ0tGOTBSQ0ErSURBZ0ppWWdYM1JU'
    'SUQ0Z01Da2dld29nSUNBZ0lDQWdJQ0FnSUNCcGJuUWdYM1JRSUQwZ1gzUnlaVkJ2Y3lBbUlEWXpP'
    'd29nSUNBZ0lDQWdJQ0FnSUNCbWJHOWhkQ0JmZEVSbGJIUmhJRDBnS0hacFlsUmhZbHRmZEZBZ0pp'
    'QXpNVjBnS2lCbWJHOWhkQ2hmZEVRcEtTQXZJRFkwTGpBN0NpQWdJQ0FnSUNBZ0lDQWdJQzh2SUZS'
    'eVpXMXZiRzhnYlc5a2FXWnBaWE1nZEdobElFOVZWRkJWVkNCaGJYQnNhWFIxWkdVZ2NHVnlMWE5o'
    'YlhCc1pUc2dhWFFnWkc5bGMyNG5kQW9nSUNBZ0lDQWdJQ0FnSUNBdkx5Qm5aWFFnYzIxdmIzUm9a'
    'V1FnZG1saElGWlBURjlUUlZRZ1ltVmpZWFZ6WlNCcGRDZHpJR0ZzY21WaFpIa2dZU0JtWVhOMElH'
    'OXpZMmxzYkdGMGFXOXVDaUFnSUNBZ0lDQWdJQ0FnSUM4dklDaHpiVzl2ZEdocGJtY2dkMjkxYkdR'
    'Z2MzVndjSEpsYzNNZ2FYUXBMaUJUZEc5eVpXUWdZWE1nWVNCa1pXeDBZU0JoYm1RZ1lXUmtaV1Fn'
    'ZEc4S0lDQWdJQ0FnSUNBZ0lDQWdMeThnZEdobElITnRiMjkwYUdWa0lIWnZiSFZ0WlNCaGRDQnZk'
    'WFJ3ZFhRZ2RHbHRaUzRLSUNBZ0lDQWdJQ0FnSUNBZ1gzUnlaVzF2Ykc5RVpXeDBZU0E5SUNoZmRG'
    'QWdQQ0F6TWlrZ1B5QmZkRVJsYkhSaElEb2dMVjkwUkdWc2RHRTdDaUFnSUNBZ0lDQWdmUW9nSUNB'
    'Z2ZRb0tJQ0FnSUM4dklFRnljR1ZuWjJsdklDaEZabVpsWTNRZ01IaDVLU0RpZ0pRZ2NIbHRiMlFu'
    'Y3lCdmNtUmxjaUJwY3lCaVlYTmw0b2FTV0Nob2FXZG9LZUtHa2xrb2JHOTNLUW9nSUNBZ2FXWWdL'
    'Rjl3WTNJdVpXWm1aV04wSUQwOUlEQjRNQ0FtSmlCZmNHTnlMbkJoY21GdElDRTlJREFwSUhzS0lD'
    'QWdJQ0FnSUNCcGJuUWdYMkZ5Y0ZOMFpYQWdQU0JwYm5Rb2NHOXpMblJwWTJzcElDMGdhVzUwS0hC'
    'dmN5NTBhV05ySUM4Z015NHdLU0FxSURNN0NpQWdJQ0FnSUNBZ0x5OGdaV1ptWldOMGFYWmxVR1Z5'
    'YVc5a0lFbFRJR0poYzJWUVpYSnBiMlFnYUdWeVpTQW9ibThnWm5WeWRHaGxjaUJ0YjJScFptbGpZ'
    'WFJwYjI0Z1ltVm1iM0psSUdGeWNDa0tJQ0FnSUNBZ0lDQnBaaUFvWDJGeWNGTjBaWEFnUFQwZ01T'
    'a0tJQ0FnSUNBZ0lDQWdJQ0FnWldabVpXTjBhWFpsVUdWeWFXOWtJRDBnWldabVpXTjBhWFpsVUdW'
    'eWFXOWtJQ29nY0c5M0tESXVNQ3dnTFdac2IyRjBLQ2hmY0dOeUxuQmhjbUZ0SUQ0K0lEUXBJQ1ln'
    'TUhoR0tTQXZJREV5TGpBcE93b2dJQ0FnSUNBZ0lHVnNjMlVnYVdZZ0tGOWhjbkJUZEdWd0lEMDlJ'
    'RElwQ2lBZ0lDQWdJQ0FnSUNBZ0lHVm1abVZqZEdsMlpWQmxjbWx2WkNBOUlHVm1abVZqZEdsMlpW'
    'QmxjbWx2WkNBcUlIQnZkeWd5TGpBc0lDMW1iRzloZENoZmNHTnlMbkJoY21GdElDWWdNSGhHS1NB'
    'dklERXlMakFwT3dvZ0lDQWdmUW9LSUNBZ0lDOHZJRlpwWW5KaGRHOGdLRVZtWm1WamRDQTBLU0Rp'
    'Z0pRZ2RYTmxjeUJuYkc5aVlXd2dkbWxpVkdGaUxnb2dJQ0FnTHk4Z1JXWm1aV04wSURSNGVEb2dj'
    'R0Z5WVcwZ1BTQW9jM0JsWldRZ1BEd2dOQ2tnZkNCa1pYQjBhQzRnSUZObGRITWdkbE1zSUhaRUxn'
    'b2dJQ0FnTHk4Z1JXWm1aV04wSURaNGVEb2dZMjl1ZEdsdWRXVWdkbWxpY21GMGJ5QW9kWE5sY3lC'
    'd2NtbHZjaUEwZUhnbmN5QjJVeTkyUkRzZ2FYUnpJRzkzYmlCd1lYSmhiU0JwY3dvZ0lDQWdMeThn'
    'SUNBZ0lDQWdJQ0FnSUNBZ2RtOXNMWE5zYVdSbElHOXViSGtnNG9DVUlHaGhibVJzWldRZ2MyVndZ'
    'WEpoZEdWc2VTQnBiaUIyYjJ4MWJXVWdZMjlrWlNCd1lYUm9LUzRLSUNBZ0lDOHZDaUFnSUNBdkx5'
    'QjJhV0p5WVhSdlVHOXpJR2x1WTNKbGJXVnVkSE1nWW5rZ2RsTWdiMjRnWldGamFDQk9UMDR0ZEds'
    'amF5MHdJQ2hwTG1VdUxDQW9jM0JsWldRdE1Ta2djR1Z5SUhKdmR5a3VDaUFnSUNBdkx5QlhZV3hy'
    'SUdaeWIyMGdkSEpwWjJkbGNpQjBieUJqZFhKeVpXNTBMQ0IxY0dSaGRHbHVaeUIyVXk5MlJDQlBU'
    'a3haSUc5dUlEUjRlQ0J5YjNkekxDQmhibVFLSUNBZ0lDOHZJR0ZqWTNWdGRXeGhkR2x1WnlBb2Mz'
    'QmxaV1F0TVNrcWRsTWdjR1Z5SUdOdmJYQnNaWFJsWkNCeWIzY2dkWE5wYm1jZ2FHbHpkRzl5YVdO'
    'aGJDQjJVeTRLSUNBZ0lIc0tJQ0FnSUNBZ0lDQnBiblFnWDNaVElEMGdNQ3dnWDNaRUlEMGdNRHNL'
    'SUNBZ0lDQWdJQ0JwYm5RZ1gzWnBZbEJ2Y3lBOUlEQTdDZ29nSUNBZ0lDQWdJQzh2SUVsdWFYUnBZ'
    'V3hwZW1VZ2RsTXZka1FnVDA1TVdTQm1jbTl0SUhSeWFXZG5aWElnY205M0ozTWdNSGcwSUNoT1Qx'
    'UWdNSGcySU9LQWxDQnBkSE1nY0dGeVlXMGdhWE1nZG05c0xYTnNhV1JsS1FvZ0lDQWdJQ0FnSUds'
    'bUlDaDBjbWxuVG05MFpTNWxabVpsWTNRZ1BUMGdNSGcwS1NCN0NpQWdJQ0FnSUNBZ0lDQWdJR2x1'
    'ZENCZmJuTWdQU0FvZEhKcFowNXZkR1V1Y0dGeVlXMGdQajRnTkNrZ0ppQXdlRVk3Q2lBZ0lDQWdJ'
    'Q0FnSUNBZ0lHbHVkQ0JmYm1RZ1BTQWdkSEpwWjA1dmRHVXVjR0Z5WVcwZ0lDQWdJQ0FnSmlBd2VF'
    'WTdDaUFnSUNBZ0lDQWdJQ0FnSUdsbUlDaGZibk1nUGlBd0tTQmZkbE1nUFNCZmJuTTdDaUFnSUNB'
    'Z0lDQWdJQ0FnSUdsbUlDaGZibVFnUGlBd0tTQmZka1FnUFNCZmJtUTdDaUFnSUNBZ0lDQWdmUW9L'
    'SUNBZ0lDQWdJQ0JwWmlBb2RISnBaMUJoZENBOVBTQndiM011YzI5dVoxQnZjeUFtSmlCMGNtbG5V'
    'bTkzSUQwOUlIQnZjeTV5YjNjcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnTHk4Z1QyNGdkSEpwWjJkbGNp'
    'QnliM2M2SUhacFluSmhkRzhnYUdGeklHOXViSGtnYUdGa0lIQnZjeTUwYVdOcklHbHVZM0psYldW'
    'dWRITUtJQ0FnSUNBZ0lDQWdJQ0FnWDNacFlsQnZjeUE5SUdsdWRDaHdiM011ZEdsamF5a2dLaUJm'
    'ZGxNN0NpQWdJQ0FnSUNBZ2ZTQmxiSE5sSUhzS0lDQWdJQ0FnSUNBZ0lDQWdMeThnVkhKcFoyZGxj'
    'aUJ5YjNjZ1kyOXVkSEpwWW5WMFpYTWdLSE53WldWa0xURXBJR2x1WTNKbGJXVnVkSE1nWVhRZ2RI'
    'SnBaMmRsY2kxeWIzY2dkbE1LSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjkwY2xObmNpQTlJSEJoZEZS'
    'cFkydFBabVp6WlhSYmRISnBaMUJoZEYwZ0t5QW9kSEpwWjFKdmR5QXRJSEJoZEZOMFlYSjBVbTkz'
    'VzNSeWFXZFFZWFJkS1RzS0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5MGNsTndaQ0E5SUdabGRHTm9W'
    'R2xqYXloZmRISlRaM0lnS3lBeEtTQXRJR1psZEdOb1ZHbGpheWhmZEhKVFozSXBPd29nSUNBZ0lD'
    'QWdJQ0FnSUNBdkx5QldhV0p5WVhSdklHdGxaWEJ6SUhKMWJtNXBibWNnYjI0Z01IZzJJSEp2ZDNN'
    'Z2RHOXZJT0tBbENCaFkyTjFiWFZzWVhSbElDaHpjR1ZsWkMweEtTcDJVd29nSUNBZ0lDQWdJQ0Fn'
    'SUNBdkx5QmxkbVZ1SUhkb1pXNGdkR2hwY3lCeWIzY2dkMkZ6SURCNE5pd2dkWE5wYm1jZ2RHaGxJ'
    'R2x1YUdWeWFYUmxaQ0IyVXk0S0lDQWdJQ0FnSUNBZ0lDQWdZbTl2YkNCZmRISnBaMGx6Vm1saVFX'
    'TjBhWFpsSUQwZ0tIUnlhV2RPYjNSbExtVm1abVZqZENBOVBTQXdlRFFnZkh3Z2RISnBaMDV2ZEdV'
    'dVpXWm1aV04wSUQwOUlEQjROaWs3Q2lBZ0lDQWdJQ0FnSUNBZ0lGOTJhV0pRYjNNZ1BTQmZkSEpw'
    'WjBselZtbGlRV04wYVhabElEOGdLRjkwY2xOd1pDQXRJREVwSUNvZ1gzWlRJRG9nTURzS0NpQWdJ'
    'Q0FnSUNBZ0lDQWdJQzh2SUZkaGJHc2djbTkzTFdKNUxYSnZkeUJtY205dElIUnlhV2RuWlhJck1T'
    'QjBieUJqZFhKeVpXNTBMVEVzSUhWd1pHRjBhVzVuSUhaVEwzWkVDaUFnSUNBZ0lDQWdJQ0FnSUM4'
    'dklHOXVJREI0TkNCeWIzZHpMQ0JoYm1RZ1lXTmpkVzExYkdGMGFXNW5JSEJsY2kxeWIzY2dkWE5w'
    'Ym1jZ2FHbHpkRzl5YVdOaGJDQjJVeTRLSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjkzY0NBOUlIUnlh'
    'V2RRWVhRc0lGOTNjaUE5SUhSeWFXZFNiM2NnS3lBeE93b2dJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1gz'
    'ZHlJRDQ5SUhCaGRGTjBZWEowVW05M1cxOTNjRjBnS3lBb2NHRjBVbTkzVDJabWMyVjBXMTkzY0Nz'
    'eFhTQXRJSEJoZEZKdmQwOW1abk5sZEZ0ZmQzQmRLU2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'WDNkd0t5czdJRjkzY2lBOUlDaGZkM0FnUENCVFQwNUhYMHhGVGtkVVNDa2dQeUJ3WVhSVGRHRnlk'
    'Rkp2ZDF0ZmQzQmRJRG9nTURzS0lDQWdJQ0FnSUNBZ0lDQWdmUW9nSUNBZ0lDQWdJQ0FnSUNCbWIz'
    'SWdLR2x1ZENCZmQya2dQU0F3T3lCZmQya2dQQ0F4TWpnN0lGOTNhU3NyS1NCN0NpQWdJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNCcFppQW9YM2R3SUQ0Z2NHOXpMbk52Ym1kUWIzTWdmSHdnS0Y5M2NDQTlQU0J3'
    'YjNNdWMyOXVaMUJ2Y3lBbUppQmZkM0lnUGowZ2NHOXpMbkp2ZHlrcElHSnlaV0ZyT3dvZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOTNjQ0ErUFNCVFQwNUhYMHhGVGtkVVNDa2dZbkpsWVdzN0Np'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9YM2R5SUQ0OUlIQmhkRk4wWVhKMFVtOTNXMTkzY0Yw'
    'Z0t5QW9jR0YwVW05M1QyWm1jMlYwVzE5M2NDc3hYU0F0SUhCaGRGSnZkMDltWm5ObGRGdGZkM0Jk'
    'S1NrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJRjkzY0Nzck95QmZkM0lnUFNBb1gzZHdJ'
    'RHdnVTA5T1IxOU1SVTVIVkVncElEOGdjR0YwVTNSaGNuUlNiM2RiWDNkd1hTQTZJREE3Q2lBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1kyOXVkR2x1ZFdVN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNC'
    'OUNpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCT2IzUmxJRjkyYmlBOUlHZGxkRTV2ZEdVb1gzZHdMQ0Jm'
    'ZDNJc0lHTm9LVHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQzh2SUZOMGIzQWdiMjRnY21WMGNtbG5a'
    'MlZ5Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JpYjI5c0lGOTJia2x6Vkc5dVpTQTlJQ2dvWDNadUxt'
    'Vm1abVZqZENBOVBTQXdlRE1nZkh3Z1gzWnVMbVZtWm1WamRDQTlQU0F3ZURVcElDWW1JRjkyYmk1'
    'd1pYSnBiMlFnUGlBd0tUc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmZG00dWNHVnlhVzlr'
    'SUQ0Z01DQW1KaUFoWDNadVNYTlViMjVsS1NCaWNtVmhhenNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Qzh2SUZWd1pHRjBaU0IyVXk5MlJDQlBUa3haSUc5dUlEQjROQ0J5YjNkeklDZ3dlRFlnYUdGeklI'
    'WnZiQzF6Ykdsa1pTQndZWEpoYlN3Z2JtOTBJSFpwWW5KaGRHOHBDaUFnSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQnBaaUFvWDNadUxtVm1abVZqZENBOVBTQXdlRFFnSmlZZ1gzWnVMbkJoY21GdElDRTlJREFw'
    'SUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBiblFnWDI1eklEMGdLRjkyYmk1d1lYSmhi'
    'U0ErUGlBMEtTQW1JREI0UmpzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBiblFnWDI1a0lE'
    'MGdJRjkyYmk1d1lYSmhiU0FnSUNBZ0lDQW1JREI0UmpzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQnBaaUFvWDI1eklENGdNQ2tnWDNaVElEMGdYMjV6T3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lHbG1JQ2hmYm1RZ1BpQXdLU0JmZGtRZ1BTQmZibVE3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0I5Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0F2THlCQlkyTjFiWFZzWVhSbElIWnBZbkpoZEc4Z2NH'
    'OXpJSGRvWlc0Z2NtOTNJR2x6SURCNE5DQlBVaUF3ZURZZ0tIWnBZbkpoZEc4Z2NuVnVjeUJ2YmlC'
    'aWIzUm9LUW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5MmJpNWxabVpsWTNRZ1BUMGdNSGcw'
    'SUh4OElGOTJiaTVsWm1abFkzUWdQVDBnTUhnMktTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnYVc1MElGOXpaM0lnUFNCd1lYUlVhV05yVDJabWMyVjBXMTkzY0YwZ0t5QW9YM2R5SUMwZ2NH'
    'RjBVM1JoY25SU2IzZGJYM2R3WFNrN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5'
    'emNHUWdQU0JtWlhSamFGUnBZMnNvWDNObmNpQXJJREVwSUMwZ1ptVjBZMmhVYVdOcktGOXpaM0lw'
    'T3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lGOTJhV0pRYjNNZ0t6MGdLRjl6Y0dRZ0xTQXhL'
    'U0FxSUY5MlV6c0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lG'
    'OTNjaXNyT3dvZ0lDQWdJQ0FnSUNBZ0lDQjlDZ29nSUNBZ0lDQWdJQ0FnSUNBdkx5QlZjR1JoZEdV'
    'Z2RsTXZka1FnWm5KdmJTQmpkWEp5Wlc1MElISnZkeUJQVGt4WklHbG1JREI0TkFvZ0lDQWdJQ0Fn'
    'SUNBZ0lDQnBaaUFvWDNCamNpNWxabVpsWTNRZ1BUMGdNSGcwSUNZbUlGOXdZM0l1Y0dGeVlXMGdJ'
    'VDBnTUNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjl1Y3lBOUlDaGZjR055TG5CaGNt'
    'RnRJRDQrSURRcElDWWdNSGhHT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOXVaQ0E5SUNC'
    'ZmNHTnlMbkJoY21GdElDQWdJQ0FnSUNZZ01IaEdPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhV1ln'
    'S0Y5dWN5QStJREFwSUY5MlV5QTlJRjl1Y3pzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsbUlDaGZi'
    'bVFnUGlBd0tTQmZka1FnUFNCZmJtUTdDaUFnSUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lD'
    'QWdMeThnUTNWeWNtVnVkQ0J5YjNjZ2NHRnlkR2xoYkRvZ2NHOXpMblJwWTJzZ2FXNWpjbVZ0Wlc1'
    'MGN5QmhkQ0JqZFhKeVpXNTBMWEp2ZHlCMlV3b2dJQ0FnSUNBZ0lDQWdJQ0F2THlCUFRreFpJSGRv'
    'Wlc0Z1kzVnljbVZ1ZENCeWIzY2dhR0Z6SUhacFluSmhkRzhnWVdOMGFYWmxJQ2d3ZURRZ2IzSWdN'
    'SGcyS1FvZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvWDNCamNpNWxabVpsWTNRZ1BUMGdNSGcwSUh4OElG'
    'OXdZM0l1WldabVpXTjBJRDA5SURCNE5pa2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdYM1pwWWxC'
    'dmN5QXJQU0JwYm5Rb2NHOXpMblJwWTJzcElDb2dYM1pUT3dvZ0lDQWdJQ0FnSUNBZ0lDQjlDaUFn'
    'SUNBZ0lDQWdmUW9LSUNBZ0lDQWdJQ0JwWmlBb1gzWkVJRDRnTUNBbUppQmZkbE1nUGlBd0tTQjdD'
    'aUFnSUNBZ0lDQWdJQ0FnSUdsdWRDQmZkbEFnUFNCZmRtbGlVRzl6SUNZZ05qTTdDaUFnSUNBZ0lD'
    'QWdJQ0FnSUdac2IyRjBJRjkyUkdWc2RHRWdQU0FvZG1saVZHRmlXMTkyVUNBbUlETXhYU0FxSUda'
    'c2IyRjBLRjkyUkNrcElDOGdNVEk0TGpBN0NpQWdJQ0FnSUNBZ0lDQWdJR1ZtWm1WamRHbDJaVkJs'
    'Y21sdlpDQXJQU0FvWDNaUUlEd2dNeklwSUQ4Z1gzWkVaV3gwWVNBNklDMWZka1JsYkhSaE93b2dJ'
    'Q0FnSUNBZ0lIMEtJQ0FnSUgwS0NpQWdJQ0F2THlCU1pXNWtaWElnYzJGdGNHeGxDaUFnSUNBdkx5'
    'QmlZWE5sVUdWeWFXOWtJRDBnWldabVpXTjBhWFpsVUdWeWFXOWtJRmRKVkVoUFZWUWdkbWxpY21G'
    'MGJ5OTBjbVZ0YjJ4dklHMXZaSFZzWVhScGIyNHVJQ0JWYzJsdVp5QjBhR1VLSUNBZ0lDOHZJRzF2'
    'WkhWc1lYUmxaQ0IyWVd4MVpTQm1iM0lnWmxOaGJYQnNaVkJ2Y3lCcGJuUmxaM0poZEdsdmJpQjNi'
    'M1ZzWkNCdGRXeDBhWEJzZVNCMGFHVWdiVzlrZFd4aGRHbHZiZ29nSUNBZ0x5OGdZVzF3YkdsMGRX'
    'UmxJR0o1SUdCbGJHRndjMlZrWUN3Z2NISnZaSFZqYVc1bklHRWdjM1ZpYzNSaGJuUnBZV3dnWW5W'
    'NmVpQmhkQ0IyYVdKeVlYUnZJSEpoZEdVS0lDQWdJQzh2SUNobExtY3VMQ0JtYkhWMFpTQnZiaUJ3'
    'WVhRZ01TQmphREFnWVhRZ2RtVmpYMlJwYlQwNEtTNGdJRlJvWlNCVVVsVkZJSEJ2YzJsMGFXOXVM'
    'V1J2YldGcGJpQmxabVpsWTNRS0lDQWdJQzh2SUc5bUlIWnBZbkpoZEc4Z2FYTWdkR2hsSUdsdWRH'
    'Vm5jbUZzSUc5bUlIUm9aU0JtY21WeElHMXZaSFZzWVhScGIyNHNJSGRvYVdOb0lHbHpJR0VnZEds'
    'dWVTQThNUzBLSUNBZ0lDOHZJSE5oYlhCc1pTQnZjMk5wYkd4aGRHbHZiaURpZ0pRZ2MyRm1aV3g1'
    'SUc1bFoyeHBaMmxpYkdVdUNpQWdJQ0JtYkc5aGRDQmlZWE5sVUdWeWFXOWtJRDBnWldabVpXTjBh'
    'WFpsVUdWeWFXOWtPd29nSUNBZ2FXWWdLSFJ5YVdkT2IzUmxMbVZtWm1WamRDQTlQU0F3ZURRZ2ZI'
    'd2dkSEpwWjA1dmRHVXVaV1ptWldOMElEMDlJREI0TmlCOGZDQjBjbWxuVG05MFpTNWxabVpsWTNR'
    'Z1BUMGdNSGczSUh4OENpQWdJQ0FnSUNBZ1gzQmpjaTVsWm1abFkzUWdQVDBnTUhnMElIeDhJRjl3'
    'WTNJdVpXWm1aV04wSUQwOUlEQjROaUI4ZkNCZmNHTnlMbVZtWm1WamRDQTlQU0F3ZURjcElIc0tJ'
    'Q0FnSUNBZ0lDQmlZWE5sVUdWeWFXOWtJRDBnS0hSaGNtZGxkRkJsY21sdlpDQStJREF1TUNrZ1B5'
    'QjBZWEpuWlhSUVpYSnBiMlFnT2lCbWJHOWhkQ2gwY21sblRtOTBaUzV3WlhKcGIyUXBPd29nSUNB'
    'Z2ZRb2dJQ0FnWm14dllYUWdabkpsY1NBOUlIQmxjbWx2WkZSdlJuSmxjVVowS0cxaGVDZ3hMQ0Jw'
    'Ym5Rb1ltRnpaVkJsY21sdlpDa3BMQ0J6YlhBdVptbHVaWFIxYm1VcE93b2dJQ0FnTHk4Z1UyRnRj'
    'R3hsSUhCdmMybDBhVzl1T2dvZ0lDQWdMeThnSUNBdElFbG1JR04xY25KbGJuUWdjbTkzSUdoaGN5'
    'QmhZM1JwZG1VZ2NHbDBZMmdnYzJ4cFpHVWdLREY0ZUM4eWVIZ3ZNM2g0S1N3Z2RYTmxJR3h2WnlC'
    'cGJuUmxaM0poYkFvZ0lDQWdMeThnSUNBZ0lPS0lxME12VUNoMEtXUjBJT0tKaUNCRHc1ZFVMODZV'
    'VUNERGx5QnNiaWhRTVM5UU1Da2dJQ2hoYzNOMWJXVnpJR3hwYm1WaGNpQnlZVzF3T3lCamJHOXpa'
    'U0JsYm05MVoyZ3BDaUFnSUNBdkx5QWdJQzBnVDNSb1pYSjNhWE5sSUhCbGNtbHZaQ0JwY3lCemRH'
    'RmliR1VnY0dGemRDQjBjbWxuWjJWeU95QnphVzF3YkdVZ1pXeGhjSE5sWk1PWFpuSmxjU0JwY3lC'
    'bGVHRmpkQzRLSUNBZ0lDOHZJRk5oYlhCc1pTQndiM05wZEdsdmJpQm1jbTl0SUhSb1pTQndaWEl0'
    'Y205M0lHVjRZV04wSUdsdWRHVm5jbUYwYjNJZ0tISmxjR3hoWTJWeklIUm9aU0J2YkdRS0lDQWdJ'
    'Qzh2SUhOcGJtZHNaUzF6WldkdFpXNTBJR1p2Y20xMWJHRWdkMmhwWTJnZ2QyRnpJSGR5YjI1bklH'
    'WnZjaUJ0ZFd4MGFTMXliM2NnYzJ4cFpHVnpJT0tBbENCelpXVUtJQ0FnSUM4dklGOW1VMkZ0Y0d4'
    'bFVHOXpRV05qSUdOdmJuTjBjblZqZEdsdmJpQmhZbTkyWlNrdUlGSmxjM1ZzZENCcGN5QnBiaUJ6'
    'YjNWeVkyVXRjbUYwWlNCellXMXdiR1Z6T3dvZ0lDQWdMeThnWkdsMmFXUmxJR0o1SUdKM1JtRmpk'
    'Rzl5SUhSdklHTnZiblpsY25RZ2RHOGdZMjl0Y0hKbGMzTmxaQzFrYjIxaGFXNGdjMkZ0Y0d4bGN5'
    'QnNhV3RsSUhSb1pRb2dJQ0FnTHk4Z2JHVm5ZV041SUdOdlpHVWdaR2xrTGdvZ0lDQWdabXh2WVhR'
    'Z1psTmhiWEJzWlZCdmN5QTlJRjltVTJGdGNHeGxVRzl6UVdOaklDOGdabXh2WVhRb2MyMXdMbUoz'
    'Um1GamRHOXlLVHNLSUNBZ0lDOHZJREI0T1hoNElITmhiWEJzWlNCdlptWnpaWFE2SUhOb2FXWjBJ'
    'SE4wWVhKMGFXNW5JSEJ2YzJsMGFXOXVJQ2hwYmlCamIyMXdjbVZ6YzJWa0xXUnZiV0ZwYmlCellX'
    'MXdiR1Z6S1FvZ0lDQWdhV1lnS0Y5ellXMXdiR1ZQWm1aelpYUWdQaUF3S1NCN0NpQWdJQ0FnSUNB'
    'Z1psTmhiWEJzWlZCdmN5QXJQU0JtYkc5aGRDaGZjMkZ0Y0d4bFQyWm1jMlYwSUM4Z2JXRjRLREVz'
    'SUhOdGNDNWlkMFpoWTNSdmNpa3BPd29nSUNBZ2ZRb0tJQ0FnSUdsbUlDaHpiWEF1Ykc5dmNFeGxi'
    'aUErSURJcElIc0tJQ0FnSUNBZ0lDQnBaaUFvWmxOaGJYQnNaVkJ2Y3lBK1BTQm1iRzloZENoemJY'
    'QXViRzl2Y0ZOMFlYSjBJQ3NnYzIxd0xteHZiM0JNWlc0cEtRb2dJQ0FnSUNBZ0lDQWdJQ0JtVTJG'
    'dGNHeGxVRzl6SUQwZ1pteHZZWFFvYzIxd0xteHZiM0JUZEdGeWRDa2dLeUJ0YjJRb1psTmhiWEJz'
    'WlZCdmN5QXRJR1pzYjJGMEtITnRjQzVzYjI5d1UzUmhjblFwTENCbWJHOWhkQ2h6YlhBdWJHOXZj'
    'RXhsYmlrcE93b2dJQ0FnZlNCbGJITmxJR2xtSUNobVUyRnRjR3hsVUc5eklENDlJR1pzYjJGMEtI'
    'TnRjQzVzWlc1bmRHZ3BLU0I3Q2lBZ0lDQWdJQ0FnY21WMGRYSnVJREF1TURzS0lDQWdJSDBLSUNB'
    'Z0lHbG1JQ2htVTJGdGNHeGxVRzl6SUR3Z01DNHdLU0J5WlhSMWNtNGdNQzR3T3dvS0lDQWdJQzh2'
    'SUZOaGJYQnNaU0IyWVd4MVpTQjNhWFJvSUhCeWIzQmxjaUJsYm1RdFptRmtaU0FvYzJGdGNHeGxJ'
    'SFJsY20xcGJtRjBhVzl1SUhOb2IzVnNaQ0J1YjNRZ2MyNWhjQ0IwYnlBd0tRb2dJQ0FnWm14dllY'
    'UWdjenNLSUNBZ0lHbG1JQ2h6YlhBdWJHOXZjRXhsYmlBOFBTQXlJQ1ltSUdaVFlXMXdiR1ZRYjNN'
    'Z1BqMGdabXh2WVhRb2MyMXdMbXhsYm1kMGFDa2dMU0F4TGpBcElIc0tJQ0FnSUNBZ0lDQXZMeUJP'
    'WldGeUlHVnVaQ0J2WmlCdWIyNHRiRzl2Y0dsdVp5QnpZVzF3YkdVNklHWmhaR1VnYjNWMElHOTJa'
    'WElnYkdGemRDQnpZVzF3YkdVZ2RHOGdZWFp2YVdRZ1kyeHBZMnNLSUNBZ0lDQWdJQ0J6SUQwZ01D'
    'NHdPd29nSUNBZ2ZTQmxiSE5sSUhzS0lDQWdJQ0FnSUNCeklEMGdaMlYwVTJGdGNHeGxSaWh6YlhB'
    'dWMzUmhjblFzSUdaVFlXMXdiR1ZRYjNNc0lITnRjQzVzWlc1bmRHZ3NJSE50Y0M1c2IyOXdVM1Jo'
    'Y25Rc0lITnRjQzVzYjI5d1RHVnVLVHNLSUNBZ0lIMEtDaUFnSUNBdkx5RGlsSURpbElBZ1FXNTBh'
    'UzFqYkdsamF5QnlZVzF3Y3lEaWxJRGlsSUFLSUNBZ0lDOHZJREV1SUZSeWFXZG5aWElnY21GdGNE'
    'b2dRVVJCVUZSSlZrVWdabUZrWlMxcGJpNEtJQ0FnSUM4dklDQWdJRVJsWm1GMWJIUTZJRFkwTFhO'
    'aGJYQnNaU0JzYVc1bFlYSWdLRzFwYTBsVUlHWmhaR1ZqYjNWdWRDa2c0b0NVSUhOb1lYSndJR1J5'
    'ZFcwZ1lYUjBZV05yTGdvZ0lDQWdMeThnSUNBZ1UyRnRjR3hsTFc5bVpuTmxkQ0J5WlhSeWFXZG5a'
    'WEp6SUNnNWVIZ3BPaUF4T1RJdGMyRnRjR3hsSUhOdGIyOTBhSE4wWlhBZzRvQ1VJRzFoYzJ0eklI'
    'Um9aUW9nSUNBZ0x5OGdJQ0FnYldsa0xYZGhkbVZtYjNKdElHUnBjMk52Ym5ScGJuVnBkSGtnZEdo'
    'aGRDQmpZWFZ6WlhNZ1kyeHBZMnR6SUc5dUlHUnlkVzB0WTJodmNIQnBibWNLSUNBZ0lDOHZJQ0Fn'
    'SUhCaGRIUmxjbTV6TGlCRWNuVnRjeUJqYUc5d2NHVmtJSFpwWVNBNWVIZ2djM1JoY25RZ1lYUWdi'
    'bTl1TFhwbGNtOGdZVzF3YkdsMGRXUmxJR2x1YzJsa1pRb2dJQ0FnTHk4Z0lDQWdkR2hsSUhOaGJY'
    'QnNaU3dnWVc1a0lIUm9aU0J3Y21WMmFXOTFjeUJ1YjNSbEozTWdkR0ZwYkNCcWRYTjBJSE4wYjNC'
    'ekxDQnpieUIzYVhSb2IzVjBDaUFnSUNBdkx5QWdJQ0JoSUd4dmJtZGxjaUJ5WVcxd0lHVjJaWEo1'
    'SUhKbGRISnBaMmRsY2lCd2IzQnpJR0YxWkdsaWJIa3VDaUFnSUNCbWJHOWhkQ0JrWldOc2FXTnJP'
    'd29nSUNBZ2FXWWdLRjl6WVcxd2JHVlBabVp6WlhRZ1BpQXdLU0I3Q2lBZ0lDQWdJQ0FnTHk4Z1Uy'
    'MXZiM1JvYzNSbGNDQnZkbVZ5SURFNU1pQnpZVzF3YkdWeklDaCtOQzQwYlhNZ1FDQTBOQzR4YTBo'
    'NktTNGdVMjF2YjNSb2MzUmxjQ0JvWVhNS0lDQWdJQ0FnSUNBdkx5QjZaWEp2SUdSbGNtbDJZWFJw'
    'ZG1VZ1lYUWdZbTkwYUNCbGJtUndiMmx1ZEhNZzRvQ1VJRzV2SUdGMVpHbGliR1VnYTJsdWF5Qmhk'
    'Q0IwYUdVZ2MzUmhjblFLSUNBZ0lDQWdJQ0F2THlCdmNpQmxibVFnYjJZZ2RHaGxJR1poWkdVc0lH'
    'cDFjM1FnWVNCemJXOXZkR2dnYzNkbGJHd3VJRXh2Ym1jZ1pXNXZkV2RvSUhSdklHMWhjMnNLSUNB'
    'Z0lDQWdJQ0F2THlCMGFHVWdiV2xrTFhkaGRtVm1iM0p0SUhOMFlYSjBJR1JwYzJOdmJuUnBiblZw'
    'ZEhrc0lITm9iM0owSUdWdWIzVm5hQ0IwYnlCd2NtVnpaWEoyWlFvZ0lDQWdJQ0FnSUM4dklIQmxj'
    'bU5sYVhabFpDQmhkSFJoWTJzZ2IyNGdjMnh2ZHlCa2NuVnRJR2hwZEhNdUNpQWdJQ0FnSUNBZ1pt'
    'eHZZWFFnZENBOUlHTnNZVzF3S0dWc1lYQnpaV1FnS2lBb05EUXhNREF1TUNBdklERTVNaTR3S1N3'
    'Z01DNHdMQ0F4TGpBcE93b2dJQ0FnSUNBZ0lHUmxZMnhwWTJzZ1BTQjBJQ29nZENBcUlDZ3pMakFn'
    'TFNBeUxqQWdLaUIwS1RzS0lDQWdJSDBnWld4elpTQjdDaUFnSUNBZ0lDQWdMeThnVTJoaGNuQWdO'
    'alF0YzJGdGNHeGxJRzFwYTBsVUlHUmxabUYxYkhRZ1ptOXlJRzV2Y20xaGJDQjBjbWxuWjJWeUxX'
    'WnliMjB0YzJGdGNHeGxMVEF1Q2lBZ0lDQWdJQ0FnWkdWamJHbGpheUE5SUdOc1lXMXdLR1ZzWVhC'
    'elpXUWdLaUFvTkRReE1EQXVNQ0F2SURZMExqQXBMQ0F3TGpBc0lERXVNQ2s3Q2lBZ0lDQjlDZ29n'
    'SUNBZ0x5OGdNaTRnUlc1a0xXOW1MWE5oYlhCc1pTQm1ZV1JsTFc5MWREb2dOalF0YzJGdGNHeGxJ'
    'R1poWkdVdGIzVjBJR0Z6SUdaVFlXMXdiR1ZRYjNNZ1lYQndjbTloWTJobGN3b2dJQ0FnTHk4Z0lD'
    'QWdjMkZ0Y0d4bElHVnVaQ0FvYjI1c2VTQm1iM0lnYm05dUxXeHZiM0JwYm1jZ2MyRnRjR3hsY3lr'
    'dUlDQlFjbVYyWlc1MGN5QnpkV1JrWlc0Z2MybHNaVzVqWlM0S0lDQWdJR1pzYjJGMElHVnVaRVpo'
    'WkdVZ1BTQXhMakE3Q2lBZ0lDQnBaaUFvYzIxd0xteHZiM0JNWlc0Z1BEMGdNaWtnZXdvZ0lDQWdJ'
    'Q0FnSUdac2IyRjBJSEpsYldGcGJtbHVaeUE5SUdac2IyRjBLSE50Y0M1c1pXNW5kR2dwSUMwZ1ps'
    'TmhiWEJzWlZCdmN6c0tJQ0FnSUNBZ0lDQnBaaUFvY21WdFlXbHVhVzVuSUR3Z05qUXVNQ2tnWlc1'
    'a1JtRmtaU0E5SUcxaGVDZ3dMakFzSUhKbGJXRnBibWx1WnlBdklEWTBMakFwT3dvZ0lDQWdmUW9L'
    'SUNBZ0lDOHZJRE11SUV4dmIzQWdZM0p2YzNObVlXUmxPaUJ6Ylc5dmRHaHpJR0Z1ZVNCeVpYTnBa'
    'SFZoYkNCc2IyOXdSVzVrNG9hU2JHOXZjRk4wWVhKMElHUnBjMk52Ym5ScGJuVnBkSGt1Q2lBZ0lD'
    'QXZMeUFnSUNCVWFHVWdaVzVqYjJSbGNpQnViM2NnWlcxaVpXUnpJR3h2YjNBZ2QzSmhjQ0JqYjI1'
    'MFpYaDBJRzVsZUhRZ2RHOGdiRzl2Y0VWdVpDQnpieUJXVVFvZ0lDQWdMeThnSUNBZ2NYVmhiblJw'
    'ZW1GMGFXOXVJR3RsWlhCeklIUm9aU0J6WldGdElHTnZiblJwYm5WdmRYTXNJR0oxZENCaElERTJM'
    'WE5oYlhCc1pTQmpjbTl6YzJaaFpHVUtJQ0FnSUM4dklDQWdJR05oZEdOb1pYTWdZVzU1SUhKbGJX'
    'RnBibWx1WnlCdGFYTnRZWFJqYUM0S0lDQWdJR2xtSUNoemJYQXViRzl2Y0V4bGJpQStJRElwSUhz'
    'S0lDQWdJQ0FnSUNCamIyNXpkQ0JtYkc5aGRDQkRVazlUVTBaQlJFVmZURVZPSUQwZ01UWXVNRHNL'
    'SUNBZ0lDQWdJQ0JtYkc5aGRDQnNiMjl3Ulc1a0lEMGdabXh2WVhRb2MyMXdMbXh2YjNCVGRHRnlk'
    'Q0FySUhOdGNDNXNiMjl3VEdWdUtUc0tJQ0FnSUNBZ0lDQm1iRzloZENCa2FYTjBSbkp2YlVWdVpD'
    'QTlJR3h2YjNCRmJtUWdMU0JtVTJGdGNHeGxVRzl6T3dvZ0lDQWdJQ0FnSUdsbUlDaGthWE4wUm5K'
    'dmJVVnVaQ0ErSURBdU1DQW1KaUJrYVhOMFJuSnZiVVZ1WkNBOElFTlNUMU5UUmtGRVJWOU1SVTRw'
    'SUhzS0lDQWdJQ0FnSUNBZ0lDQWdabXh2WVhRZ2QzSmhjRkJ2Y3lBOUlHWnNiMkYwS0hOdGNDNXNi'
    'Mjl3VTNSaGNuUXBJQ3NnS0VOU1QxTlRSa0ZFUlY5TVJVNGdMU0JrYVhOMFJuSnZiVVZ1WkNrN0Np'
    'QWdJQ0FnSUNBZ0lDQWdJR1pzYjJGMElIZHlZWEJUWVcxd2JHVWdQU0JuWlhSVFlXMXdiR1ZHS0hO'
    'dGNDNXpkR0Z5ZEN3Z2QzSmhjRkJ2Y3l3Z2MyMXdMbXhsYm1kMGFDd2djMjF3TG14dmIzQlRkR0Z5'
    'ZEN3Z2MyMXdMbXh2YjNCTVpXNHBPd29nSUNBZ0lDQWdJQ0FnSUNCbWJHOWhkQ0JpYkdWdVpDQTlJ'
    'Q2hEVWs5VFUwWkJSRVZmVEVWT0lDMGdaR2x6ZEVaeWIyMUZibVFwSUM4Z1ExSlBVMU5HUVVSRlgw'
    'eEZUanNLSUNBZ0lDQWdJQ0FnSUNBZ0x5OGdSWEYxWVd3dGNHOTNaWElnWTNKdmMzTm1ZV1JsQ2lB'
    'Z0lDQWdJQ0FnSUNBZ0lHWnNiMkYwSUhjeElEMGdZMjl6S0dKc1pXNWtJQ29nTVM0MU56QTNPVFl6'
    'S1RzS0lDQWdJQ0FnSUNBZ0lDQWdabXh2WVhRZ2R6SWdQU0J6YVc0b1lteGxibVFnS2lBeExqVTNN'
    'RGM1TmpNcE93b2dJQ0FnSUNBZ0lDQWdJQ0J6SUQwZ2N5QXFJSGN4SUNzZ2QzSmhjRk5oYlhCc1pT'
    'QXFJSGN5T3dvZ0lDQWdJQ0FnSUgwS0lDQWdJSDBLQ2lBZ0lDQXZMeURpbElEaWxJQWdVMjF2YjNS'
    'b1pXUWdkbTlzZFcxbElHRndjR3hwWTJGMGFXOXVJT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tV'
    'Z09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1Vn'
    'T0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdP'
    'S1VnQW9nSUNBZ0x5OGdRMjl0Y0hWMFpTQjBhR1VnY21GdGNDQm1ZV04wYjNJZ1ltRnpaV1FnYjI0'
    'Z2FHOTNJRzFoYm5rZ2MyRnRjR3hsY3lCb1lYWmxJR1ZzWVhCelpXUWdjMmx1WTJVS0lDQWdJQzh2'
    'SUhSb1pTQnRiM04wSUhKbFkyVnVkQ0IyYjJ4MWJXVWdZMmhoYm1kbExpQlRRVTFRVEVWVFgxQkZV'
    'bDlVU1VOTElHbHpJRFEwTVRBd0wxUkpRMHRUWDFCRlVsOVRSVU1LSUNBZ0lDOHZJQ2gwZVhCcFky'
    'RnNiSGtnT0RneUlHRjBJRlJKUTB0VFgxQkZVbDlUUlVNOU5UQXBMaUJTWVcxd0lHTnZiWEJzWlhS'
    'bGN5QnZkbVZ5SURZMElITmhiWEJzWlhNS0lDQWdJQzh2SU9LSmlDQXhMalExYlhNZzRvQ1VJR1po'
    'YzNRZ1pXNXZkV2RvSUhSdklHSmxJR2x0Y0dWeVkyVndkR2xpYkdVZ1lYTWdZU0JtWVdSbExXbHVJ'
    'R0oxZENCemJHOTNJR1Z1YjNWbmFBb2dJQ0FnTHk4Z2RHOGdhR2xrWlNCMGFHVWdjR1Z5TFhOaGJY'
    'QnNaU0J6ZEdWd0lIUm9ZWFFnY0hKdlpIVmpaWE1nZEdobElHTnNhV05yTGdvZ0lDQWdMeThLSUNB'
    'Z0lDOHZJSEJ2Y3k1MGFXTnJSaUE5SUhSb1pTQm5iRzlpWVd3Z2RHbGpheUJ2WmlCMGFHVWdZM1Z5'
    'Y21WdWRDQnpZVzF3YkdVc0lHWnlZV04wYVc5dVlXd3VDaUFnSUNCcGJuUWdYMk4xY2xKdmQxTm5j'
    'aUE5SUhCaGRGUnBZMnRQWm1aelpYUmJjRzl6TG5OdmJtZFFiM05kSUNzZ0tIQnZjeTV5YjNjZ0xT'
    'QndZWFJUZEdGeWRGSnZkMXR3YjNNdWMyOXVaMUJ2YzEwcE93b2dJQ0FnWm14dllYUWdYM0J2YzFS'
    'cFkydEdJRDBnWm14dllYUW9abVYwWTJoVWFXTnJLRjlqZFhKU2IzZFRaM0lwS1NBcklIQnZjeTUw'
    'YVdOck93b2dJQ0FnWm14dllYUWdYMU5CVFZCTVJWTmZVRVZTWDFSSlEwc2dQU0EwTkRFd01DNHdJ'
    'QzhnVkVsRFMxTmZVRVZTWDFORlF6c0tJQ0FnSUdac2IyRjBJRjkyVW1GdGNDQTlJR05zWVcxd0tD'
    'aGZjRzl6VkdsamEwWWdMU0JmZG05c1EyaGhibWRsUVhSVWFXTnJSaWtnS2lCZlUwRk5VRXhGVTE5'
    'UVJWSmZWRWxEU3lBdklEWTBMakFzSURBdU1Dd2dNUzR3S1RzS0lDQWdJR1pzYjJGMElGOWxabVpX'
    'YjJ3Z1BTQnRhWGdvWm14dllYUW9YM1p2YkZCeVpYWXBMQ0JtYkc5aGRDaGZkbTlzUTNWeWNpa3NJ'
    'RjkyVW1GdGNDazdDaUFnSUNBdkx5QlVjbVZ0YjJ4dklHRndjR3hwWlhNZ2IyNGdkRzl3SUc5bUlI'
    'Um9aU0J6Ylc5dmRHaGxaQ0IyYjJ4MWJXVXNJSFJvWlc0Z1kyeGhiWEFnZEc4Z01DNHVOalF1Q2lB'
    'Z0lDQmZaV1ptVm05c0lEMGdZMnhoYlhBb1gyVm1abFp2YkNBcklGOTBjbVZ0YjJ4dlJHVnNkR0Vz'
    'SURBdU1Dd2dOalF1TUNrN0Nnb2dJQ0FnY21WMGRYSnVJSE1nS2lBb1gyVm1abFp2YkNBdklEWTBM'
    'akFwSUNvZ1pHVmpiR2xqYXlBcUlHVnVaRVpoWkdVN0NuMEtDZ292THlEaWxJRGlsSURpbElEaWxJ'
    'RGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElE'
    'aWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURp'
    'bElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGls'
    'SURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJ'
    'RGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJQUtMeThnWjJWMFEyaGhibTVsYkU5'
    'MWRIQjFkQ0RpZ0pRZ2NIVmliR2xqSUdWdWRISjVMaUJTZFc1eklIUnlhV2RuWlhJZ2MyVmhjbU5v'
    'TENCMGFHVnVJSFJvWlNCaWIyUjVMZ292THlCR2IzSWdkR2hsSUdacGNuTjBJRFkwSUhOaGJYQnNa'
    'WE1nWVdaMFpYSWdZU0J5WlhSeWFXZG5aWElzSUVGTVUwOGdjbVZ1WkdWeUlIZHBkR2dnZEdobENp'
    'OHZJSEJ5WlhacGIzVnpJSFJ5YVdkblpYSWdZVzVrSUdKc1pXNWtJT0tBbENCMGFHbHpJR2x6SUhS'
    'b1pTQndjbVYyYVc5MWN5MXViM1JsSUdOeWIzTnpabUZrWlFvdkx5QjBhR0YwSUdWc2FXMXBibUYw'
    'WlhNZ2FXNTBaWEl0Ym05MFpTQmpiR2xqYTNNZ0tHMWhkR05vWlhNZ2RHaGxJR1I1YVc1blczUmRJ'
    'Q3NnWTJoaGJtNWxiRnQwWFFvdkx5QmpjbTl6YzJaaFpHVWdhVzRnVFdsclRXOWtKM01nVFVSU1Zs'
    'OU5TVmd1UTFCUUtTNEtMeThnNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBT'
    'QTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNB'
    'NHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0'
    'cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRw'
    'U0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBT'
    'QTRwU0E0cFNBNHBTQUNtWnNiMkYwSUdkbGRFTm9ZVzV1Wld4UGRYUndkWFFvYVc1MElHTm9MQ0Jt'
    'Ykc5aGRDQjBhVzFsTENCUWIzTnBkR2x2YmlCd2IzTXNJR1pzYjJGMElISnZkMVJwYldVcElIc0tD'
    'aUFnSUNBdkx5QlRkR1Z3SURFNklHWnBibVFnYlc5emRDMXlaV05sYm5Sc2VTMTBjbWxuWjJWeVpX'
    'UWdibTkwWlNCdmJpQjBhR2x6SUdOb1lXNXVaV3d1Q2lBZ0lDQXZMeUJRVkNCelpXMWhiblJwWTNN'
    'ZzRvQ1VJR0VnSW5SeWFXZG5aWElpSUdseklITnZiV1YwYUdsdVp5QjBhR0YwSUhOMFlYSjBjeUIw'
    'YUdVZ2MyRnRjR3hsSUdGMElIQnZjeUF3T2dvZ0lDQWdMeThnSUNEaWdLSWdSblZzYkNCeWIzY2dL'
    'R2x1YzNSeWRXMWxiblFnS3lCd1pYSnBiMlFwSUNBZ0lDQWdJQ0FnSUNBZ0lDRGlnSlFnY21WMGNt'
    'bG5aMlZ5Q2lBZ0lDQXZMeUFnSU9LQW9pQlFaWEpwYjJRdGIyNXNlU0J5YjNjZ0tHNXZJR2x1YzNR'
    'c0lHNXZJR1ZtWm1WamRDQXpMelVwSUNBZ0lPS0FsQ0J5WlhSeWFXZG5aWElzSUdsdWFHVnlhWFFn'
    'YVc1emRISjFiV1Z1ZEFvZ0lDQWdMeThnSUNEaWdLSWdVR1Z5YVc5a0xXOXViSGtnZDJsMGFDQmxa'
    'bVpsWTNRZ015ODFJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDRGlnSlFnYzJ4cFpHVWdkR0Z5WjJWMElH'
    'OXViSGtzSUc1dklISmxkSEpwWjJkbGNnb2dJQ0FnTHk4Z0lDRGlnS0lnUm5Wc2JDQnliM2NnZDJs'
    'MGFDQmxabVpsWTNRZ015ODFJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0RpZ0pRZ2MyeHBaR1Vn'
    'ZEdGeVoyVjBJRzl1Ykhrc0lHNXZJSEpsZEhKcFoyZGxjZ29nSUNBZ0x5OGdJQ0RpZ0tJZ1JXMXdk'
    'SGtnTHlCcGJuTjBjblZ0Wlc1MExXOXViSGtnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNEaWdK'
    'UWdZMjl1ZEdsdWRXVWdjSEpwYjNJZ2JtOTBaUW9nSUNBZ1RtOTBaU0JmWTNWeVVtOTNJRDBnWjJW'
    'MFRtOTBaU2h3YjNNdWMyOXVaMUJ2Y3l3Z2NHOXpMbkp2ZHl3Z1kyZ3BPd29nSUNBZ1RtOTBaU0Iw'
    'Y21sblRtOTBaU0E5SUY5amRYSlNiM2M3Q2lBZ0lDQnBiblFnSUhSeWFXZFNiM2NnSUQwZ2NHOXpM'
    'bkp2ZHpzS0lDQWdJR2x1ZENBZ2RISnBaMUJoZENBZ1BTQndiM011YzI5dVoxQnZjenNLSUNBZ0lH'
    'bHVkQ0FnZEc5dVpWTnNhV1JsVkdGeVoyVjBJRDBnTURzZ0lDOHZJSGRvWlc0Z2MyVjBMQ0IwYUds'
    'eklISnZkeUJqWVhKeWFXVnpJR0VnTTNoNEx6VjRlQ0J6Ykdsa1pTQjBZWEpuWlhRS0lDQWdJR0p2'
    'YjJ3Z1gyTjFja2x6Vkc5dVpWQnZjblJoSUQwZ0tDaGZZM1Z5VW05M0xtVm1abVZqZENBOVBTQXdl'
    'RE1nZkh3Z1gyTjFjbEp2ZHk1bFptWmxZM1FnUFQwZ01IZzFLU0FtSmdvZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1gyTjFjbEp2ZHk1d1pYSnBiMlFnUGlBd0tUc0tJQ0FnSUdK'
    'dmIyd2dYMk4xY2tselVtVjBjbWxuSUNBZ0lEMGdLRjlqZFhKU2IzY3VjR1Z5YVc5a0lENGdNQ0Ft'
    'SmlBaFgyTjFja2x6Vkc5dVpWQnZjblJoS1RzZ0lDOHZJR0Z1ZVNCd1pYSnBiMlFnZDJsMGFHOTFk'
    'Q0F6THpVZ2NtVjBjbWxuWjJWeWN3b2dJQ0FnWW05dmJDQmZZM1Z5U0dGelNXNXpkQ0FnSUNBZ1BT'
    'QW9YMk4xY2xKdmR5NXBibk4wY25WdFpXNTBJRDRnTUNrN0Nnb2dJQ0FnYVdZZ0tGOWpkWEpKYzFS'
    'dmJtVlFiM0owWVNrZ2V3b2dJQ0FnSUNBZ0lDOHZJRk5zYVdSbElIUmhjbWRsZENEaWdKUWdabWx1'
    'WkNCd2NtbHZjaUJTUlVGTUlIUnlhV2RuWlhJZ1ptOXlJSE5oYlhCc1pTOXdaWEpwYjJRZ1kyOXVk'
    'R1Y0ZEFvZ0lDQWdJQ0FnSUhSdmJtVlRiR2xrWlZSaGNtZGxkQ0E5SUY5amRYSlNiM2N1Y0dWeWFX'
    'OWtPd29nSUNBZ0lDQWdJR2x1ZENCelVpQTlJSEJ2Y3k1eWIzY3NJSE5RSUQwZ2NHOXpMbk52Ym1k'
    'UWIzTTdDaUFnSUNBZ0lDQWdabTl5SUNocGJuUWdiR0lnUFNBeE95QnNZaUE4SURFeU9Ec2diR0ly'
    'S3lrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0J6VWkwdE93b2dJQ0FnSUNBZ0lDQWdJQ0JwWmlBb2MxSWdQ'
    'Q0F3S1NCN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9jMUFnUGlBd0tTQjdDaUFnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnYzFBdExUc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0J6VWlB'
    'OUlIQmhkRk4wWVhKMFVtOTNXM05RWFNBcklDaHdZWFJTYjNkUFptWnpaWFJiYzFBck1WMGdMU0J3'
    'WVhSU2IzZFBabVp6WlhSYmMxQmRLU0F0SURFN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCOUlHVnNj'
    'MlVnZXlCaWNtVmhhenNnZlFvZ0lDQWdJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdJQ0FnSUU1dmRH'
    'VWdjSEpsZGlBOUlHZGxkRTV2ZEdVb2MxQXNJSE5TTENCamFDazdDaUFnSUNBZ0lDQWdJQ0FnSUdK'
    'dmIyd2djSEpsZGtselZHOXVaVlJ5YVdjZ1BTQW9LSEJ5WlhZdVpXWm1aV04wSUQwOUlEQjRNeUI4'
    'ZkNCd2NtVjJMbVZtWm1WamRDQTlQU0F3ZURVcElDWW1JSEJ5WlhZdWNHVnlhVzlrSUQ0Z01DazdD'
    'aUFnSUNBZ0lDQWdJQ0FnSUM4dklGSmxZV3dnZEhKcFoyZGxjam9nYUdGeklIQmxjbWx2WkNCQlRr'
    'UWdibTkwSUdFZ2RHOXVaUzF3YjNKMFlTQjBZWEpuWlhRZ2NtOTNDaUFnSUNBZ0lDQWdJQ0FnSUds'
    'bUlDaHdjbVYyTG5CbGNtbHZaQ0ErSURBZ0ppWWdJWEJ5WlhaSmMxUnZibVZVY21sbktTQjdDaUFn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQXZMeUJFWlhSbGNtMXBibVVnYVc1emRISjFiV1Z1ZERvZ2NISmxa'
    'bVZ5SUhCeVpYWXVhVzV6ZEhKMWJXVnVkQ3dnWld4elpTQnpZMkZ1SUdaMWNuUm9aWElnWm05eUlH'
    'TnZiblJsZUhRS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsbUlDaHdjbVYyTG1sdWMzUnlkVzFsYm5R'
    'Z1BpQXdLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2RISnBaMDV2ZEdVZ1BTQndjbVYy'
    'T3lCMGNtbG5VbTkzSUQwZ2MxSTdJSFJ5YVdkUVlYUWdQU0J6VURzS0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUgwZ1pXeHpaU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0x5OGdVR1Z5YVc5a0xX'
    'OXViSGtnY205M0lPS0FsQ0JtYVc1a0lHbHVjM1J5ZFcxbGJuUWdZMjl1ZEdWNGRBb2dJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJRTV2ZEdVZ2NtVmhiQ0E5SUhCeVpYWTdDaUFnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnYVc1MElITlNNaUE5SUhOU0xDQnpVRElnUFNCelVEc0tJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0JtYjNJZ0tHbHVkQ0JzWWpJZ1BTQXhPeUJzWWpJZ1BDQXhNamc3SUd4aU1p'
    'c3JLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJSE5TTWkwdE93b2dJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9jMUl5SUR3Z01Da2dld29nSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tITlFNaUErSURBcElIc2djMUF5TFMwN0lITlNN'
    'aUE5SUhCaGRGTjBZWEowVW05M1czTlFNbDBnS3lBb2NHRjBVbTkzVDJabWMyVjBXM05RTWlzeFhT'
    'QXRJSEJoZEZKdmQwOW1abk5sZEZ0elVESmRLU0F0SURFN0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJR1ZzYzJVZ1luSmxZV3M3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdUbTkwWlNCd01pQTlJ'
    'R2RsZEU1dmRHVW9jMUF5TENCelVqSXNJR05vS1RzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnYVdZZ0tIQXlMbWx1YzNSeWRXMWxiblFnUGlBd0tTQjdJSEpsWVd3dWFXNXpkSEoxYldW'
    'dWRDQTlJSEF5TG1sdWMzUnlkVzFsYm5RN0lHSnlaV0ZyT3lCOUNpQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdmUW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUhSeWFXZE9iM1JsSUQwZ2NtVmhi'
    'RHNnZEhKcFoxSnZkeUE5SUhOU095QjBjbWxuVUdGMElEMGdjMUE3Q2lBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0I5Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JpY21WaGF6c0tJQ0FnSUNBZ0lDQWdJQ0FnZlFv'
    'Z0lDQWdJQ0FnSUgwS0lDQWdJSDBnWld4elpTQnBaaUFvSVY5amRYSkpjMUpsZEhKcFp5a2dld29n'
    'SUNBZ0lDQWdJQzh2SUU1dklIQmxjbWx2WkNEaWdKUWdZMjl1ZEdsdWRXVWdjSEpwYjNJZ2JtOTBa'
    'U0FvYjNJZ2JtOGdZWFZrYVc4Z2FXWWdibTkwYUdsdVp5QndjbWx2Y2lrdUNpQWdJQ0FnSUNBZ0x5'
    'OGdYMk4xY2toaGMwbHVjM1FnZDJsMGFDQnVieUJ3WlhKcGIyUWdhWE1nWVNCdWJ5MXZjQ0JtYjNJ'
    'Z2RISnBaMmRsY2lCd2RYSndiM05sY3lBb1VGUWdjWFZwY21zcExnb2dJQ0FnSUNBZ0lHbHVkQ0J6'
    'VWlBOUlIQnZjeTV5YjNjc0lITlFJRDBnY0c5ekxuTnZibWRRYjNNN0NpQWdJQ0FnSUNBZ1ptOXlJ'
    'Q2hwYm5RZ2JHSWdQU0F4T3lCc1lpQThJREV5T0RzZ2JHSXJLeWtnZXdvZ0lDQWdJQ0FnSUNBZ0lD'
    'QnpVaTB0T3dvZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvYzFJZ1BDQXdLU0I3Q2lBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0JwWmlBb2MxQWdQaUF3S1NCN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdjMUF0'
    'TFRzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnpVaUE5SUhCaGRGTjBZWEowVW05M1czTlFY'
    'U0FySUNod1lYUlNiM2RQWm1aelpYUmJjMUFyTVYwZ0xTQndZWFJTYjNkUFptWnpaWFJiYzFCZEtT'
    'QXRJREU3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0I5SUdWc2MyVWdleUJpY21WaGF6c2dmUW9nSUNB'
    'Z0lDQWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ0lDQWdJRTV2ZEdVZ2NISmxkaUE5SUdkbGRFNXZkR1Vv'
    'YzFBc0lITlNMQ0JqYUNrN0NpQWdJQ0FnSUNBZ0lDQWdJR0p2YjJ3Z2NISmxka2x6Vkc5dVpWUnlh'
    'V2NnUFNBb0tIQnlaWFl1WldabVpXTjBJRDA5SURCNE15QjhmQ0J3Y21WMkxtVm1abVZqZENBOVBT'
    'QXdlRFVwSUNZbUlIQnlaWFl1Y0dWeWFXOWtJRDRnTUNrN0NpQWdJQ0FnSUNBZ0lDQWdJR2xtSUNo'
    'd2NtVjJMbkJsY21sdlpDQStJREFnSmlZZ0lYQnlaWFpKYzFSdmJtVlVjbWxuS1NCN0NpQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNCcFppQW9jSEpsZGk1cGJuTjBjblZ0Wlc1MElENGdNQ2tnZXdvZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBZ0lIUnlhV2RPYjNSbElEMGdjSEpsZGpzZ2RISnBaMUp2ZHlBOUlI'
    'TlNPeUIwY21sblVHRjBJRDBnYzFBN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCOUlHVnNjMlVnZXdv'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lFNXZkR1VnY21WaGJDQTlJSEJ5WlhZN0NpQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUhOU01pQTlJSE5TTENCelVESWdQU0J6VURzS0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQm1iM0lnS0dsdWRDQnNZaklnUFNBeE95QnNZaklnUENBeE1q'
    'ZzdJR3hpTWlzcktTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lITlNNaTB0T3dv'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwWmlBb2MxSXlJRHdnTUNrZ2V3b2dJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0hOUU1pQStJREFwSUhzZ2MxQXlM'
    'UzA3SUhOU01pQTlJSEJoZEZOMFlYSjBVbTkzVzNOUU1sMGdLeUFvY0dGMFVtOTNUMlptYzJWMFcz'
    'TlFNaXN4WFNBdElIQmhkRkp2ZDA5bVpuTmxkRnR6VURKZEtTQXRJREU3SUgwS0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHVnNjMlVnWW5KbFlXczdDaUFnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1RtOTBa'
    'U0J3TWlBOUlHZGxkRTV2ZEdVb2MxQXlMQ0J6VWpJc0lHTm9LVHNLSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdhV1lnS0hBeUxtbHVjM1J5ZFcxbGJuUWdQaUF3S1NCN0lISmxZV3d1YVc1'
    'emRISjFiV1Z1ZENBOUlIQXlMbWx1YzNSeWRXMWxiblE3SUdKeVpXRnJPeUI5Q2lBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJSFJ5YVdkT2IzUmxJ'
    'RDBnY21WaGJEc2dkSEpwWjFKdmR5QTlJSE5TT3lCMGNtbG5VR0YwSUQwZ2MxQTdDaUFnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmljbVZoYXpzS0lDQWdJQ0FnSUNB'
    'Z0lDQWdmUW9nSUNBZ0lDQWdJSDBLSUNBZ0lIMGdaV3h6WlNCcFppQW9YMk4xY2tselVtVjBjbWxu'
    'SUNZbUlDRmZZM1Z5U0dGelNXNXpkQ2tnZXdvZ0lDQWdJQ0FnSUM4dklGQmxjbWx2WkMxdmJteDVJ'
    'SEpsZEhKcFoyZGxjaURpZ0pRZ1ptbHVaQ0JwYm5OMGNuVnRaVzUwSUdOdmJuUmxlSFFnS0hOaGJY'
    'QnNaU0JwYm1obGNtbDBaV1FnWm5KdmJTQndjbWx2Y2lCMGNtbG5aMlZ5S1FvZ0lDQWdJQ0FnSUds'
    'dWRDQnpVaUE5SUhCdmN5NXliM2NzSUhOUUlEMGdjRzl6TG5OdmJtZFFiM003Q2lBZ0lDQWdJQ0Fn'
    'Wm05eUlDaHBiblFnYkdJZ1BTQXhPeUJzWWlBOElERXlPRHNnYkdJckt5a2dld29nSUNBZ0lDQWdJ'
    'Q0FnSUNCelVpMHRPd29nSUNBZ0lDQWdJQ0FnSUNCcFppQW9jMUlnUENBd0tTQjdDaUFnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQnBaaUFvYzFBZ1BpQXdLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z2MxQXRMVHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCelVpQTlJSEJoZEZOMFlYSjBVbTkz'
    'VzNOUVhTQXJJQ2h3WVhSU2IzZFBabVp6WlhSYmMxQXJNVjBnTFNCd1lYUlNiM2RQWm1aelpYUmJj'
    'MUJkS1NBdElERTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQjlJR1ZzYzJVZ2V5QmljbVZoYXpzZ2ZR'
    'b2dJQ0FnSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0FnSUNBZ0lFNXZkR1VnY0hKbGRpQTlJR2RsZEU1'
    'dmRHVW9jMUFzSUhOU0xDQmphQ2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2h3Y21WMkxtbHVjM1J5'
    'ZFcxbGJuUWdQaUF3S1NCN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCMGNtbG5UbTkwWlM1cGJuTjBj'
    'blZ0Wlc1MElEMGdjSEpsZGk1cGJuTjBjblZ0Wlc1ME93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1lu'
    'SmxZV3M3Q2lBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdMeThnZEhK'
    'cFoxQmhkQzkwY21sblVtOTNJSE4wWVhrZ1lYUWdZM1Z5Y21WdWRDQnliM2NnNG9DVUlIUm9hWE1n'
    'U1ZNZ1lTQnlaWFJ5YVdkblpYSUtJQ0FnSUgwS0lDQWdJQzh2SUdWc2MyVTZJR1oxYkd3Z2RISnBa'
    'MmRsY2lBb2NHVnlhVzlrSUNzZ2FXNXpkSEoxYldWdWRDd2dibThnTXk4MUtTRGlnSlFnZEhKcFow'
    'NXZkR1VnWVd4eVpXRmtlU0JqYjNKeVpXTjBDZ29nSUNBZ0x5OGc0cFNBNHBTQUlGSmxibVJsY2lC'
    'M2FYUm9JR04xY25KbGJuUWdkSEpwWjJkbGNpRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURp'
    'bElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGls'
    'SURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJ'
    'RGlsSUFLSUNBZ0lHWnNiMkYwSUhOZlkzVnljaUE5SUY5blkyOUNiMlI1S0dOb0xDQndiM01zSUhS'
    'cGJXVXNJSEp2ZDFScGJXVXNJSFJ5YVdkUVlYUXNJSFJ5YVdkU2IzY3NJSFJ5YVdkT2IzUmxMQ0Iw'
    'YjI1bFUyeHBaR1ZVWVhKblpYUXBPd29LSUNBZ0lDOHZJT0tVZ09LVWdDQkRjbTl6YzJaaFpHVWdk'
    'Mmx1Wkc5M0lHTm9aV05ySU9LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdP'
    'S1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09L'
    'VWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tV'
    'Z09LVWdPS1VnT0tVZ0FvZ0lDQWdMeThnUTI5dGNIVjBaU0J6WVcxd2JHVnpJR1ZzWVhCelpXUWdj'
    'Mmx1WTJVZ2RHaGxJR04xY25KbGJuUWdkSEpwWjJkbGNpQm1hWEpsWkM0Z1QyNXNlU0JwYm5OcFpH'
    'VUtJQ0FnSUM4dklIUm9aU0JtYVhKemRDQTJOQ0J6WVcxd2JHVnpJR2x6SUhSb1pTQmpjbTl6YzJa'
    'aFpHVWdiV1ZoYm1sdVoyWjFiQ0RpZ0pRZ1ltVjViMjVrSUhSb1lYUXNJSFJvWlFvZ0lDQWdMeThn'
    'Y0hKbGRtbHZkWE1nYm05MFpTQm9ZWE1nYkc5dVp5Qm1ZV1JsWkNCdmRYUXVDaUFnSUNCbWJHOWhk'
    'Q0JqZFhKVWNtbG5WR2x0WlVZZ1BTQm1iRzloZENobVpYUmphRlJwWTJzb2NHRjBWR2xqYTA5bVpu'
    'TmxkRnQwY21sblVHRjBYU0FySUNoMGNtbG5VbTkzSUMwZ2NHRjBVM1JoY25SU2IzZGJkSEpwWjFC'
    'aGRGMHBLU2tLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQXZJRlJKUTB0VFgxQkZVbDlU'
    'UlVNN0NpQWdJQ0JtYkc5aGRDQmhaMlZUWVcxd2JHVnpJRDBnS0hScGJXVWdMU0JqZFhKVWNtbG5W'
    'R2x0WlVZcElDb2dORFF4TURBdU1Ec0tDaUFnSUNCcFppQW9ZV2RsVTJGdGNHeGxjeUE4SURZMExq'
    'QWdKaVlnWVdkbFUyRnRjR3hsY3lBK1BTQXdMakFwSUhzS0lDQWdJQ0FnSUNBdkx5RGlsSURpbElB'
    'Z1UyVmhjbU5vSUdadmNpQjBhR1VnVUZKRlZrbFBWVk1nZEhKcFoyZGxjaUFvYjI1bElISnZkeUJp'
    'WldadmNtVWdZM1Z5Y21WdWRDa2c0cFNBNHBTQTRwU0E0cFNBQ2lBZ0lDQWdJQ0FnTHk4Z1UyRnRa'
    'U0JoYkdkdmNtbDBhRzBnWVhNZ2RHaGxJRzFoYVc0Z2RISnBaMmRsY2lCelpXRnlZMmdnWW5WMElI'
    'TjBZWEowYVc1bklHOXVaU0J5YjNjS0lDQWdJQ0FnSUNBdkx5QmxZWEpzYVdWeUxpQlRaWFJ6SUhC'
    'VWNtbG5VR0YwTDNCVWNtbG5VbTkzTDNCVWNtbG5UbTkwWlM0S0lDQWdJQ0FnSUNCcGJuUWdJSEJV'
    'Y21sblVHRjBJRDBnTFRFc0lIQlVjbWxuVW05M0lEMGdMVEU3Q2lBZ0lDQWdJQ0FnVG05MFpTQndW'
    'SEpwWjA1dmRHVTdDaUFnSUNBZ0lDQWdld29nSUNBZ0lDQWdJQ0FnSUNCcGJuUWdjMUlnUFNCMGNt'
    'bG5VbTkzTENCelVDQTlJSFJ5YVdkUVlYUTdDaUFnSUNBZ0lDQWdJQ0FnSUdadmNpQW9hVzUwSUd4'
    'aUlEMGdNVHNnYkdJZ1BDQXhNamc3SUd4aUt5c3BJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJSE5T'
    'TFMwN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9jMUlnUENBd0tTQjdDaUFnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnYVdZZ0tITlFJRDRnTUNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNCelVDMHRPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnpVaUE5SUhC'
    'aGRGTjBZWEowVW05M1czTlFYU0FySUNod1lYUlNiM2RQWm1aelpYUmJjMUFyTVYwZ0xTQndZWFJT'
    'YjNkUFptWnpaWFJiYzFCZEtTQXRJREU3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2ZTQmxi'
    'SE5sSUhzZ1luSmxZV3M3SUgwS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUU1dmRHVWdjSEpsZGlBOUlHZGxkRTV2ZEdVb2MxQXNJSE5TTENCamFDazdDaUFnSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lDQmliMjlzSUhCeVpYWkpjMVJ2Ym1WVWNtbG5JRDBnS0Nod2NtVjJMbVZt'
    'Wm1WamRDQTlQU0F3ZURNZ2ZId2djSEpsZGk1bFptWmxZM1FnUFQwZ01IZzFLU0FtSmlCd2NtVjJM'
    'bkJsY21sdlpDQStJREFwT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tIQnlaWFl1Y0dWeWFX'
    'OWtJRDRnTUNBbUppQWhjSEpsZGtselZHOXVaVlJ5YVdjcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0JwWmlBb2NISmxkaTVwYm5OMGNuVnRaVzUwSUQ0Z01Da2dld29nSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQndWSEpwWjA1dmRHVWdQU0J3Y21WMk95QndWSEpwWjFKdmR5QTlJ'
    'SE5TT3lCd1ZISnBaMUJoZENBOUlITlFPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUgwZ1pX'
    'eHpaU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQzh2SUZCbGNtbHZaQzF2Ym14'
    'NUlPS0FsQ0JtYVc1a0lHbHVjM1J5ZFcxbGJuUWdZMjl1ZEdWNGRBb2dJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNCT2IzUmxJSEpsWVd3Z1BTQndjbVYyT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0JwYm5RZ2MxSXlJRDBnYzFJc0lITlFNaUE5SUhOUU93b2dJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCbWIzSWdLR2x1ZENCc1lqSWdQU0F4T3lCc1lqSWdQQ0F4TWpn'
    'N0lHeGlNaXNyS1NCN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnpVakl0'
    'TFRzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2h6VWpJZ1BDQXdL'
    'U0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0hOUU1p'
    'QStJREFwSUhzZ2MxQXlMUzA3SUhOU01pQTlJSEJoZEZOMFlYSjBVbTkzVzNOUU1sMGdLeUFvY0dG'
    'MFVtOTNUMlptYzJWMFczTlFNaXN4WFNBdElIQmhkRkp2ZDA5bVpuTmxkRnR6VURKZEtTQXRJREU3'
    'SUgwS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JsYkhObElHSnla'
    'V0ZyT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdUbTkwWlNCd01pQTlJR2RsZEU1dmRHVW9jMUF5TENC'
    'elVqSXNJR05vS1RzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2h3'
    'TWk1cGJuTjBjblZ0Wlc1MElENGdNQ2tnZXlCeVpXRnNMbWx1YzNSeWRXMWxiblFnUFNCd01pNXBi'
    'bk4wY25WdFpXNTBPeUJpY21WaGF6c2dmUW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QjlDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lIQlVjbWxuVG05MFpTQTlJSEpsWVd3'
    'N0lIQlVjbWxuVW05M0lEMGdjMUk3SUhCVWNtbG5VR0YwSUQwZ2MxQTdDaUFnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHSnlaV0ZyT3dvZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdmUW9LSUNBZ0lD'
    'QWdJQ0F2THlEaWxJRGlsSUFnVW1WdVpHVnlJSGRwZEdnZ2NISmxkbWx2ZFhNZ2RISnBaMmRsY2lC'
    'aGJtUWdZbXhsYm1RZzRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNB'
    'NHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0FDaUFnSUNBZ0lDQWdM'
    'eThnVkdobElIQnlaWFpwYjNWeklIUnlhV2RuWlhJZ1oyVjBjeUIwYjI1bFUyeHBaR1ZVWVhKblpY'
    'UTlNQ0FvZDJVZ1pHOXVKM1FnZEhKaFkyc2dhWFJ6Q2lBZ0lDQWdJQ0FnTHk4Z2MyeHBaR1VnWTJo'
    'aGFXNGc0b0NVSUdadmNpQjBhR1VnTmpRdGMyRnRjR3hsSUdOeWIzTnpabUZrWlNCM2FXNWtiM2Nn'
    'ZEdobElHUnBabVpsY21WdVkyVUtJQ0FnSUNBZ0lDQXZMeUJwY3lCcGJtRjFaR2xpYkdVcExpQk1h'
    'VzVsWVhJZ1kzSnZjM05tWVdSbE9pQjBQVEFnYVhNZ1lXeHNMWEJ5WlhacGIzVnpMQ0IwUFRFZ2FY'
    'TUtJQ0FnSUNBZ0lDQXZMeUJoYkd3dFkzVnljbVZ1ZEM0Z1UyMXZiM1JvYzNSbGNDQjNiM1ZzWkNC'
    'M2IzSnJJR0oxZENCc2FXNWxZWElnYldGMFkyaGxjeUJOYVd0TmIyUW5jd29nSUNBZ0lDQWdJQzh2'
    'SUVNckt5Qk5hWGdxVTNSbGNtVnZUbTlqYkdsamF5QjJiMngxYldVZ2NtRnRjR2x1WnlCbGVHRmpk'
    'R3g1TGdvZ0lDQWdJQ0FnSUdsbUlDaHdWSEpwWjFCaGRDQStQU0F3S1NCN0NpQWdJQ0FnSUNBZ0lD'
    'QWdJR1pzYjJGMElITmZjSEpsZGlBOUlGOW5ZMjlDYjJSNUtHTm9MQ0J3YjNNc0lIUnBiV1VzSUhK'
    'dmQxUnBiV1VzSUhCVWNtbG5VR0YwTENCd1ZISnBaMUp2ZHl3Z2NGUnlhV2RPYjNSbExDQXdLVHNL'
    'SUNBZ0lDQWdJQ0FnSUNBZ1pteHZZWFFnZENBOUlHRm5aVk5oYlhCc1pYTWdMeUEyTkM0d093b2dJ'
    'Q0FnSUNBZ0lDQWdJQ0J5WlhSMWNtNGdjMTl3Y21WMklDb2dLREV1TUNBdElIUXBJQ3NnYzE5amRY'
    'SnlJQ29nZERzS0lDQWdJQ0FnSUNCOUNpQWdJQ0I5Q2dvZ0lDQWdjbVYwZFhKdUlITmZZM1Z5Y2pz'
    'S2ZRbz0nKS5kZWNvZGUoJ3V0Zi04JykKCiAgICAjIEFzc2VtYmxlCiAgICByZXR1cm4gaGVhZGVy'
    'ICsgbWV0YSArICIiLmpvaW4oZGF0YV9hcnJheXMpICsgIlxuIiArIHRhYmxlcyArIGZldGNoZXJz'
    'ICsgZGVjb2RlcnMgKyBnZXRfY2hhbm5lbF9vdXRwdXQKCgppZiBfX25hbWVfXyA9PSAnX19tYWlu'
    'X18nOgogICAgbW9kX3BhdGggPSBzeXMuYXJndlsxXSBpZiBsZW4oc3lzLmFyZ3YpID4gMSBlbHNl'
    'ICcvbW50L3VzZXItZGF0YS91cGxvYWRzLzEyVEguTU9EJwogICAgb3V0X3BhdGggPSBzeXMuYXJn'
    'dlsyXSBpZiBsZW4oc3lzLmFyZ3YpID4gMiBlbHNlICcvaG9tZS9jbGF1ZGUvbW9kX2NydW5jaC8x'
    'MlRIX2NydW5jaF9jb21tb24uZ2xzbCcKICAgIG1haW4obW9kX3BhdGgsIG91dF9wYXRoKQo='
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
                        default=None,
                        help='Sample resampler. linear=2-tap (cheapest, ProTracker-style), '
                             'bspline=4-tap cubic (smooth/soft, RECOMMENDED — quality '
                             'indistinguishable from lanczos3 on 8-bit MOD source since '
                             'the encoder pre-AA-filters; saves 12 sin() calls per sample), '
                             'lanczos3=6-tap sinc (sharpest/brightest, ~50%% more cost — '
                             'use only if you can hear the difference and have headroom). '
                             'Default: bspline (or linear if --max-compat without override).')
    parser.add_argument('--viz', type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7], default=1,
                        help='Image-tab visualizer:\n'
                             '  0 = None             (black backdrop, fastest compile)\n'
                             '  1 = Reactive 001     (PAEz fork — SDF circles + cosmic web)  ← default\n'
                             '  2 = Fluxline Surfer  (mrange — DR2 dodecahedron + glowtracer)\n'
                             '  3 = Zuvuya           (city/stars + audio-reactive curtain)\n'
                             '  4 = Maya             (raymarched fractal tunnel-warp)\n'
                             '  5 = Dodecahedron     (Philip Bertani — DR2 IFS fractal raymarcher)\n'
                             '  6 = Disco Combined   (smoke spotlights + lasers/clouds, time-driven)\n'
                             '  7 = Sparkly 4D       (Philip Bertani — 4D IFS volumetric raymarcher)')
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
                             'Skips the VQ→Sound splitter (no VQ arrays exist to move).')
    # ── Individual compat-component knobs (overridable when used with --max-compat) ──
    # Each defaults to None (sentinel "user did not set"), so we can do layered
    # resolution: --max-compat fills None values with compat defaults, then any
    # remaining None gets the normal default. Result: explicit user flags ALWAYS
    # win, regardless of CLI argument order. Pass any of these alongside
    # --max-compat to selectively keep one feature at full quality.
    parser.add_argument('--reverb-size', choices=['full','small'], default=None,
                        help="Reverb dimensions. full = 4 combs × 3 iters (default), "
                             "small = 2 combs × 2 iters (--max-compat default). "
                             "Reduces compile cost and stereo width.")
    parser.add_argument('--surround', dest='surround',
                        action=argparse.BooleanOptionalAction, default=None,
                        help="3D surround widening on outer LRRL pair. "
                             "Default: ON (or OFF if --max-compat without override).")
    parser.add_argument('--phatbass', dest='phatbass',
                        action=argparse.BooleanOptionalAction, default=None,
                        help="PhatBass Hilbert allpass enhancement on bass instruments. "
                             "Default: ON (or OFF if --max-compat without override).")
    parser.add_argument('--phatbass-mode', dest='phatbass_mode',
                        choices=['auto', 'sample', 'mix'], default='sample',
                        help="PhatBass routing. 'sample' (default) forces per-sample "
                             "via isBass[] flags — cleanest, leaves leads/pads alone. "
                             "'auto' uses per-sample when bass instruments were detected, "
                             "else mix-wide. 'mix' forces mix-wide (applies Hilbert "
                             "cross-pan to the entire mixdown — wider stereo + bass "
                             "enhancement on everything, can smear mid/high transients "
                             "slightly).")
    parser.add_argument('--fat4x', dest='fat4x',
                        action=argparse.BooleanOptionalAction, default=None,
                        help="FAT4X harmonic exciter on master output. "
                             "Default: ON (kept ON even under --max-compat — it's cheap).")
    parser.add_argument('--fft-n', dest='fft_n',
                        type=int, choices=[64,128,256,512,1024,2048], default=None,
                        help="FFT size for Buffer A spectrum. Larger = more frequency "
                             "resolution but slower compile. Default: 1024 (or 128 if "
                             "--max-compat without override).")
    parser.add_argument('--max-compat', action='store_true', default=False,
                        help='[NO-OP — max-compat is now the DEFAULT in v1.40+] '
                             'This flag previously enabled compatibility mode '
                             'for problematic GPUs/drivers (Windows + Firefox + '
                             'NVIDIA, etc.). The compat preset (--resampler '
                             'lanczos3, --reverb-size small, --no-surround, '
                             '--phatbass, --fft-n 512, FAT4X on, extra HLSL '
                             'pragmas) is now applied by default since most '
                             'consumer setups need it and the quality '
                             'difference is small. To opt OUT of any compat '
                             'setting, pass the inverse individual flag — '
                             'e.g. --reverb-size full, --surround. The flag is '
                             'kept for backward compatibility with old '
                             'command lines but does nothing.')
    args = parser.parse_args()

    # ── Two-pass argv-order config resolution ────────────────────────────
    # Architecture: argv is scanned in TWO passes, in argv order.
    #   Pass 1: aggregate flags (e.g. --max-compat). Each aggregate carries a
    #           bundle of preset values; processing in argv order lets a later
    #           aggregate override an earlier one cleanly.
    #   Pass 2: individual flags (e.g. --resampler, --phatbass). These override
    #           anything aggregates set, regardless of position. Argparse
    #           already takes the LAST occurrence if a flag appears multiple
    #           times, so within-individual order is handled automatically.
    # Anything still unset after both passes falls back to NORMAL_DEFAULTS.
    # Result: explicit individual flags always win; aggregates are baselines.
    #
    # Adding a new aggregate: register a CLI flag (action='store_true'), then
    # add an entry to AGGREGATE_PRESETS keyed by the flag string.
    # Adding a new individual: register a CLI flag with default=None, then
    # add the dest name to INDIVIDUAL_KNOBS plus normal/aggregate values.

    AGGREGATE_PRESETS = {
        '--max-compat': {
            'resampler':              'lanczos3', # User wants lanczos3 even under max-compat
            'reverb_size':            'small',    # 2 combs × 2 iters
            'surround':               False,      # disable 3D widening
            'phatbass':               True,       # User wants phatbass even under max-compat
            'fat4x':                  True,       # KEEP — cheap, audibly worth it
            'fft_n':                  512,        # user wants 512 by default
            '_compat_extra_pragmas':  True,       # HLSL [unroll/loop] pragmas
        },
        # Add future aggregates here. e.g. '--quality-max': {...}
    }

    INDIVIDUAL_KNOBS = ['resampler', 'reverb_size', 'surround',
                        'phatbass', 'fat4x', 'fft_n']

    # NORMAL_DEFAULTS — these are now equivalent to --max-compat. The user
    # wanted max-compat to be the default since most consumer GPUs / browsers
    # need the compat path anyway, and the audible quality difference vs
    # full-quality is small. Anyone wanting full-quality (large reverb,
    # surround, no extra HLSL pragmas) can pass the relevant individual
    # flags explicitly: --reverb-size full, --surround.
    NORMAL_DEFAULTS = {
        'resampler':              'lanczos3',
        'reverb_size':            'small',     # 2 combs × 2 iters
        'surround':               False,       # 3D widening off
        'phatbass':               True,
        'fat4x':                  True,
        'fft_n':                  512,
        '_compat_extra_pragmas':  True,        # HLSL [unroll/loop] pragmas
    }

    # ── Pass 1: aggregates in argv order ──────────────────────────────────
    state = {}
    aggregates_seen = []
    for tok in sys.argv[1:]:
        # Strip any '=value' suffix to match a flag name like '--max-compat'.
        flag = tok.split('=', 1)[0]
        if flag in AGGREGATE_PRESETS:
            aggregates_seen.append(flag)
            for k, v in AGGREGATE_PRESETS[flag].items():
                state[k] = v
    if aggregates_seen:
        for agg in aggregates_seen:
            print(f"⚙️  {agg}: applying aggregate preset")
        for k, v in state.items():
            print(f"     {k:23s}→ {v!r}")

    # ── Pass 2: individual flags override aggregates ──────────────────────
    overrides_applied = []
    for k in INDIVIDUAL_KNOBS:
        user_v = getattr(args, k)
        if user_v is not None:
            prior = state.get(k)
            state[k] = user_v
            if prior is not None and prior != user_v:
                overrides_applied.append((k, prior, user_v))
            elif aggregates_seen and k not in NORMAL_DEFAULTS:
                # Aggregate-only setting that user re-affirmed
                pass
    if overrides_applied:
        print("🎛️  Individual overrides applied on top of aggregates:")
        for k, was, now in overrides_applied:
            print(f"     {k:23s}→ {now!r}  (overrides {was!r})")

    # ── Pass 3: normal defaults fill anything still unset ─────────────────
    for k, default_v in NORMAL_DEFAULTS.items():
        if k not in state:
            state[k] = default_v

    # ── Pass 4: copy resolved state back into args namespace ──────────────
    for k, v in state.items():
        setattr(args, k, v)

    # ── Pass 5: map user-facing knobs to internal _compat_* attributes the
    # rest of the codebase reads. The _compat_no_X naming with inverted sense
    # ("X is disabled") is preserved so emit-time guards don't need touching.
    args._compat_reverb_2x2  = (args.reverb_size == 'small')
    args._compat_no_phatbass = not args.phatbass
    args._compat_no_fat      = not args.fat4x
    args._compat_no_surround = not args.surround
    args._compat_fft_n       = args.fft_n
    # _compat_extra_pragmas is set in state directly (no user-facing knob).

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
                    "GLSL (The Last) MOD Player v1.40 (c) 2026 Orblivius",
                    "GLSL (The Last) MOD Player v1.40 (c) 2026 Orblivius", 1)
                _ct = _ct.replace("   COMMON TAB\n", f"   COMMON TAB\n   Visualizer: {_vname}\n", 1)

                # Inject visualizer note-synth helpers (waveType[] + _synthWave).
                # The VQ encoder doesn't know about Image/Buffer A's synth path,
                # so its emitted Common is missing these. Without them the Image
                # tab fails to compile with "'waveType': undeclared identifier".
                # Splice point: just before SampleInfo's `samples[]` array, which
                # is roughly where create_shadertoy_glsl emits these in its own
                # Common template — keeps file structure consistent.
                if 'waveType[31]' not in _ct and 'float _synthWave' not in _ct:
                    _wave_types = _classify_mod_waveforms_for(mod, verbose=False)
                    _synth_block = _emit_visualizer_synth_glsl(_wave_types)
                    # Try inserting before the samples[] declaration first;
                    # fall back to before periodTable; last resort = before
                    # the first non-comment GLSL declaration.
                    _splice_idx = -1
                    for _anchor in ('const SampleInfo samples[',
                                    'const int periodTable[',
                                    'struct SampleInfo'):
                        _splice_idx = _ct.find(_anchor)
                        if _splice_idx >= 0:
                            # Back up to the start of the line so the insertion
                            # lands cleanly above the anchor declaration.
                            _line_start = _ct.rfind('\n', 0, _splice_idx) + 1
                            _splice_idx = _line_start
                            break
                    if _splice_idx >= 0:
                        _ct = _ct[:_splice_idx] + _synth_block + _ct[_splice_idx:]
                    else:
                        # No anchor found — append at end. Image will still find
                        # the symbols since GLSL allows forward use within Common.
                        _ct = _ct + '\n' + _synth_block

                # Inject audio playback timing defines used by Sound (for
                # playbackTime computation) and Image/BufferA (for visualizer
                # iTime macro). The VQ encoder's Common template doesn't emit
                # these — splice them right after TOTAL_SONG_ROWS so they're
                # visible to the rest of Common and to any tab that #includes
                # via the Common-tab compile.
                if 'SONG_DURATION_S' not in _ct:
                    _timing_block = (
                        "\n// Audio playback timing constants — used by both Sound (for\n"
                        "// playbackTime computation) and Image/BufferA (for GUI/visualizer\n"
                        "// sync with audio). See create_shadertoy_glsl for full explanation.\n"
                        "#define SONG_DURATION_S  (float(TOTAL_SONG_ROWS) * float(SPEED) / (float(BPM) * 0.4))\n"
                        "#define INTRO_SILENCE_S  1.5\n"
                        "#define AUDIO_BUFFER_S   180.0\n"
                    )
                    _tsr_idx = _ct.find('#define TOTAL_SONG_ROWS')
                    if _tsr_idx >= 0:
                        _eol = _ct.find('\n', _tsr_idx)
                        if _eol >= 0:
                            _ct = _ct[:_eol+1] + _timing_block + _ct[_eol+1:]
                    else:
                        # No TOTAL_SONG_ROWS to anchor against — prepend before
                        # the first GLSL line. Should never happen in practice.
                        _ct = _timing_block + _ct
                with open(glsl_common_file, 'w') as _cf: _cf.write(_ct)
            except Exception as _vqcommon_err:
                print(f"   WARNING: VQ-Common post-processing failed ({_vqcommon_err})")
        except Exception as _e:
            print(f"   WARNING: VQ encoder failed ({_e}), falling back to built-in")
            _fb_glsl = base_name + "_shadertoy.glsl"
            create_shadertoy_glsl(mod, _fb_glsl, args.downsample, compress=True,
                                 compressed_pattern_size=pattern_size,
                                 pattern_bytes_data=pattern_bytes,
                                 sample_bytes_data=sample_bytes,
                                 seek_table=seek_table, vec_dim=args.vec_dim,
                                 viz=args.viz,
                                 compat={
                                     'no_surround':   getattr(args, '_compat_no_surround', False),
                                     'no_fat':        getattr(args, '_compat_no_fat', False),
                                     'reverb_2x2':    getattr(args, '_compat_reverb_2x2', False),
                                     'fft_n':         getattr(args, '_compat_fft_n', 256),
                                     'extra_pragmas': getattr(args, '_compat_extra_pragmas', False),
                                     'phatbass_mode': args.phatbass_mode,
                                 })
            glsl_common_file = _fb_glsl.replace('.glsl', '_common.glsl')

    # Sound / Image / Buffer A tabs from built-in emitter
    # Use a stub name that has NO overlap with _shadertoy.glsl patterns
    _glsl_stub = base_name + "_tmp_tabs_shadertoy.glsl"
    create_shadertoy_glsl(mod, _glsl_stub, args.downsample, compress=True,
                         compressed_pattern_size=pattern_size,
                         pattern_bytes_data=pattern_bytes,
                         sample_bytes_data=sample_bytes,
                         seek_table=seek_table, vec_dim=args.vec_dim,
                         viz=args.viz,
                         compat={
                             'no_surround':   getattr(args, '_compat_no_surround', False),
                             'no_fat':        getattr(args, '_compat_no_fat', False),
                             'reverb_2x2':    getattr(args, '_compat_reverb_2x2', False),
                             'fft_n':         getattr(args, '_compat_fft_n', 256),
                             'extra_pragmas': getattr(args, '_compat_extra_pragmas', False),
                             'phatbass_mode': args.phatbass_mode,
                         })
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
    # use them — Image and Buffer A synthesize approximate audio from note pattern
    # data via the name-classified _synthWave helper instead. This split is the
    # core fix for the Windows + ANGLE + NVIDIA OOM-on-large-const-arrays crash.
    # Skipped only for --use-png, which uses a totally different data path
    # (samples in a PNG texture, no VQ arrays exist to move).
    if args.use_png:
        print("   📌 --use-png: legacy PNG-loaded Common (getChannelOutput inline, samples via texelFetch)")
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

                # Capture _gcoBody — extracted body of getChannelOutput, used
                # by the previous-note crossfade. Same constraint: calls
                # getSampleF, must live in Sound. Must be defined BEFORE
                # getChannelOutput in Sound (which calls it), so prepend.
                _gcobody_match = _re3.search(
                    r'(?://[^\n]*\n)*float\s+_gcoBody\s*\([^)]*\)\s*\{', _common_src)
                _gcobody_fn = ''
                if _gcobody_match:
                    _start = _gcobody_match.start()
                    _bi = _common_src.index('{', _gcobody_match.end() - 1)
                    _depth = 1
                    _i = _bi + 1
                    while _i < len(_common_src) and _depth > 0:
                        _c = _common_src[_i]
                        if _c == '{': _depth += 1
                        elif _c == '}': _depth -= 1
                        _i += 1
                    _gcobody_fn = _common_src[_start:_i]
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
                _extracted_anything = bool(_arrays or _fetch_fns or _gs_fns or _gsf_fns or _gco_fn or _gcobody_fn)
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
                # _gcoBody must come BEFORE getChannelOutput (which calls it).
                if _gcobody_fn: _prelude_parts.append(_gcobody_fn + '\n')
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
                _moved_fn_count = (len(_fetch_fns) + len(_gs_fns) + len(_gsf_fns) + len(_lz_fns)
                                   + (1 if _gco_fn else 0) + (1 if _gcobody_fn else 0))
                print(f"   ✂️  Moved {len(_arrays)} arrays + {_moved_fn_count} fns "
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
