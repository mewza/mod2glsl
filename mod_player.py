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
  D. Default --resampler is lanczos3 (sharp 6-tap kernel, ProTracker-clear).
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
            elif len(signature) == 4 and signature[2:4] in (b'CH', b'CN') and signature[0:1].isdigit() and signature[1:2].isdigit():
                # xxCH (e.g. '10CH') is the common form; xxCN (e.g. '10CN') is the
                # variant Schism Tracker writes when exporting S3M→MOD. Same semantics:
                # two ASCII digits = channel count.
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

            # File Format Information byte (offset 0x2A in header).
            # 1 = signed sample data (unusual)
            # 2 = unsigned sample data (the ScreamTracker-3 default — every
            #     ST3-saved S3M in the wild uses this)
            # We default to 2 if absent or out-of-spec, since that matches
            # what ST3 wrote and what >99% of S3Ms in the wild contain.
            ffi = header[0x2A]
            self.samples_unsigned = (ffi != 1)
            
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
            
            # Read instruments. Pre-fill every slot with an empty-sample
            # placeholder dict so the rest of the pipeline never sees None
            # — match MODFile's convention (length=0, data=empty np.int8 array)
            # so the downstream `sample['data'] is not None and len(...)` test
            # works uniformly across MOD/S3M.
            def _empty_inst():
                return {
                    'name': '', 'length': 0,
                    'repeat_point': 0, 'repeat_length': 0,
                    'volume': 0, 'c2spd': 8363,
                    'data': np.array([], dtype=np.int8),
                    'finetune': 0,
                }
            self.instruments = [_empty_inst() for _ in range(self.num_instruments)]
            for i, para in enumerate(instrument_paras):
                if para == 0:
                    continue                  # leave placeholder
                offset = para * 16
                f.seek(offset)
                inst_data = f.read(80)

                inst_type = inst_data[0]
                if inst_type != 1:            # AdLib (2..7) or empty — skip sample read
                    # Still capture the name so the visualizer can display it
                    try:
                        nm = inst_data[48:76].decode('ascii', errors='ignore').rstrip('\x00')
                        self.instruments[i]['name'] = nm
                    except Exception:
                        pass
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

                # Read sample data — guarantee we always populate `data` with
                # a numpy array (possibly empty), never None.
                # ScreamTracker 3 stores 8-bit PCM as UNSIGNED (0..255, silence=128)
                # while MOD format and the rest of this player expect SIGNED int8
                # (-128..+127, silence=0). Without conversion every sample plays
                # with a +128 DC offset → loud click on each note-on, polarity
                # inversion, and harsh distortion (the "samples encoded wrong"
                # symptom). Honour the file's FFI byte: subtract 128 only when
                # the file claims unsigned storage.
                sample_data = np.array([], dtype=np.int8)
                if length > 0 and sample_offset > 0:
                    f.seek(sample_offset * 16)
                    sample_bytes = f.read(length)
                    if self.samples_unsigned:
                        # uint8 → int16 (room for the subtraction) → int8
                        sample_data = (np.frombuffer(sample_bytes, dtype=np.uint8)
                                         .astype(np.int16) - 128).astype(np.int8)
                    else:
                        sample_data = np.frombuffer(sample_bytes, dtype=np.int8).copy()
                    # If the file was truncated, zero-pad rather than crashing
                    # downstream when length doesn't match what was read.
                    if len(sample_data) < length:
                        pad = np.zeros(length - len(sample_data), dtype=np.int8)
                        sample_data = np.concatenate([sample_data, pad])

                loop_length = loop_end - loop_begin if (flags & 1) else 0

                self.instruments[i] = {
                    'name': name,
                    'length': length,
                    'repeat_point': loop_begin,
                    'repeat_length': loop_length,
                    'volume': volume,
                    'c2spd': c2spd,
                    'data': sample_data,
                    'finetune': 0,            # S3M uses c2spd, not finetune
                }
            
            # Read patterns
            # Pre-fill every pattern slot with an empty 64-row pattern so any
            # slot whose parapointer is missing/zero/out-of-range still yields
            # a valid pattern (downstream code does mod.patterns[idx] without
            # None-checks).
            self.patterns = [self.create_empty_pattern() for _ in range(self.num_patterns)]
            for i, para in enumerate(pattern_paras):
                if para == 0:
                    continue                      # leave the empty placeholder
                
                offset = para * 16
                f.seek(offset)
                packed_length_raw = f.read(2)
                if len(packed_length_raw) < 2:
                    continue                      # truncated → keep empty
                packed_length = struct.unpack('<H', packed_length_raw)[0]
                packed_data = f.read(packed_length)
                
                # Unpack pattern
                try:
                    self.patterns[i] = self.unpack_pattern(packed_data)
                except Exception:
                    pass                          # malformed → keep empty
    
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
    """Detect module format. Tries signature first, then file extension as
    fallback. Returns 'MOD', 'S3M', 'XM', 'IT', 'STM', 'MTM', or 'UNKNOWN'.
    Extension fallback handles cases where the signature check is fooled by
    slightly-mangled or non-standard headers."""
    import os
    # Surface missing-file errors immediately rather than swallowing them and
    # falling through to the (misleading) "signature check failed" path that
    # used to make a typo'd filename look like a header-corruption issue.
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file: {filename!r}")
    sig_at_44 = sig_at_1080 = b''
    sig_at_0 = b''
    try:
        with open(filename, 'rb') as f:
            sig_at_0 = f.read(17)            # 'Extended Module: ' for XM
            f.seek(44);   sig_at_44   = f.read(4)
            f.seek(1080); sig_at_1080 = f.read(4)
    except Exception:
        pass

    # ── Signature pass ──────────────────────────────────────────────────────
    if sig_at_44 == b'SCRM':
        return 'S3M'
    if sig_at_0 == b'Extended Module: ':
        return 'XM'
    if sig_at_0[:4] == b'IMPM':
        return 'IT'
    # MOD: 4-byte tag at offset 1080
    if sig_at_1080 in (b'M.K.', b'M!K!', b'M&K!', b'N.T.', b'FLT4', b'FLT8',
                       b'OCTA', b'CD81', b'OKTA',
                       b'1CHN', b'2CHN', b'3CHN', b'4CHN', b'5CHN', b'6CHN',
                       b'7CHN', b'8CHN', b'9CHN'):
        return 'MOD'
    if (len(sig_at_1080) == 4 and sig_at_1080[2:4] in (b'CH', b'CN')
            and sig_at_1080[0:1].isdigit() and sig_at_1080[1:2].isdigit()):
        return 'MOD'
    if len(sig_at_1080) == 4 and sig_at_1080[:3] == b'TDZ' and sig_at_1080[3:4].isdigit():
        return 'MOD'
    # MTM: 'MTM' at offset 0
    if sig_at_0[:3] == b'MTM':
        return 'MTM'
    # STM: 'STM' at offset 20 — hard to detect at offset 0, skip; fall through to ext.

    # ── Extension fallback ─────────────────────────────────────────────────
    # Signature failed (mangled header, unknown variant, etc.). Trust the
    # filename extension as a hint — better to attempt parsing and surface
    # a meaningful parse error than to bail with "Unknown format" when the
    # user clearly meant .s3m / .xm / .it.
    ext = os.path.splitext(filename)[1].lower().lstrip('.')
    ext_map = {
        'mod': 'MOD', 'm15': 'MOD', 'nst': 'MOD', 'wow': 'MOD',
        's3m': 'S3M',
        'xm':  'XM',
        'it':  'IT',
        'stm': 'STM',
        'mtm': 'MTM',
    }
    if ext in ext_map:
        print(f"⚠️  Signature check failed for {filename!r}, falling back to .{ext} extension → {ext_map[ext]}")
        return ext_map[ext]

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
    # S3M note to MOD period conversion (same as in main()).
    # Anchored on S3M C-5 = period 214 (per user ear test).
    # See main()'s copy for the bug history; both must stay in sync.
    s3m_note_to_period = [
        6848, 6464, 6101, 5758, 5435, 5130, 4842, 4570, 4314, 4072, 3843, 3628,        # oct 0
        3424, 3232, 3050, 2879, 2718, 2565, 2421, 2285, 2157, 2036, 1922, 1814,        # oct 1
        1712, 1616, 1525, 1440, 1359, 1283, 1211, 1143, 1078, 1018,  961,  907,        # oct 2
         856,  808,  763,  720,  679,  641,  605,  571,  539,  509,  480,  453,        # oct 3
         428,  404,  381,  360,  340,  321,  303,  286,  270,  254,  240,  227,        # oct 4
         214,  202,  191,  180,  170,  160,  151,  143,  135,  127,  120,  113,        # oct 5
         107,  101,   95,   90,   85,   80,   76,   71,   67,   64,   60,   57,        # oct 6
          54,   50,   48,   45,   42,   40,   38,   36,   34,   32,   30,   28,        # oct 7
    ]

    def s3m_note_to_mod_period(note):
        if note == 255 or note == 254:
            return 0
        octave   = (note >> 4) & 0x0F
        semitone = note & 0x0F
        if semitone >= 12:
            return 0
        idx = octave * 12 + semitone
        if idx >= len(s3m_note_to_period):
            return 0
        return s3m_note_to_period[idx]
    
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

/* Tracks area click-to-scroll: left half scrolls back, right half scrolls
   forward. Hidden when nc <= TRACKS_PER_VIEW (nothing to scroll). */
#tracker.scrollable {{ cursor:ew-resize; user-select:none; }}
#tracker.scrollable .trk-row:hover {{ background:#181828 !important; }}
.trk-scroll-hint {{
  display:none; padding:4px 12px; background:var(--bg2);
  color:var(--accent2); font-size:10px; letter-spacing:2px;
  border-top:1px solid var(--border); border-bottom:1px solid var(--border);
  text-align:center; white-space:nowrap; user-select:none;
}}
#tracker.scrollable .trk-scroll-hint {{ display:block; }}
.trk-scroll-arrow {{ color:var(--accent); margin:0 8px; font-weight:bold; }}

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
  <div class="trk-scroll-hint" id="trkScrollHint">
    <span class="trk-scroll-arrow">&#9664;</span>
    <span id="trkScrollRange">1-4 of 4</span>
    <span class="trk-scroll-arrow">&#9654;</span>
    &nbsp;&nbsp;click left or right side to scroll tracks
  </div>
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
  // How many tracks fit comfortably across the page width without crushing
  // the cells. With 8 channels the per-cell width was ~12% of viewport which
  // truncated the eff/sample fields. 4 visible at a time keeps each cell
  // legible at any reasonable window size.
  const TRACKS_PER_VIEW = 4;
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

  const nc = modData.numChannels||4;
  // Visible window: tracks [firstTrack .. firstTrack+visibleCount). When nc is
  // smaller than TRACKS_PER_VIEW (e.g. 4-channel MOD), we just show all of
  // them and hide the slider entirely.
  const visibleCount = Math.min(TRACKS_PER_VIEW, nc);
  const maxFirstTrack = Math.max(0, nc - visibleCount);
  let firstTrack = 0;

  // Header column elements — text gets re-set when window scrolls
  const hdr = document.getElementById('trkHeader');
  hdr.innerHTML = '<div class="trk-col-hdr">ROW</div>' +
    Array.from({{length:visibleCount}},()=>`<div class="trk-col-hdr"></div>`).join('');
  const hdrCols = Array.from(hdr.querySelectorAll('.trk-col-hdr')).slice(1);

  // Pre-build row elements with `visibleCount` cells each (not nc) — only the
  // visible window is rendered, so small/cheap regardless of nc.
  const body = document.getElementById('trkBody');
  const rowEls = [];
  for(let r=0;r<TOTAL;r++){{
    const row = document.createElement('div');
    row.className = 'trk-row';
    const rn = document.createElement('div');
    rn.className = 'trk-rownum';
    row.appendChild(rn);
    const cells = [];
    for(let c=0;c<visibleCount;c++){{
      const cell = document.createElement('div');
      cell.className = 'trk-cell';
      row.appendChild(cell);
      cells.push(cell);
    }}
    body.appendChild(row);
    rowEls.push({{row, rn, cells}});
  }}

  // Click-to-scroll setup — only enabled when there are more tracks than
  // fit in the view. Click on the LEFT half of the tracker shifts the
  // visible window one track earlier; right half shifts one track later.
  // Clamps at the edges (no wrap-around — easier to know when you've
  // reached the first/last track).
  const trackerEl = document.getElementById('tracker');
  const rangeEl   = document.getElementById('trkScrollRange');
  if(maxFirstTrack > 0){{
    trackerEl.classList.add('scrollable');
    trackerEl.addEventListener('click', function(ev){{
      // Where in the tracker did they click? Decide direction by x position
      // relative to the tracker's bounding rect.
      const rect = trackerEl.getBoundingClientRect();
      const xFrac = (ev.clientX - rect.left) / rect.width;
      const dir = (xFrac < 0.5) ? -1 : 1;
      const next = firstTrack + dir;
      if(next < 0 || next > maxFirstTrack) return;   // clamp at edges
      firstTrack = next;
      updateHeaderAndRange();
      updateTracker();
    }});
  }}

  function updateHeaderAndRange(){{
    for(let i=0;i<visibleCount;i++){{
      hdrCols[i].textContent = `TRACK #${{firstTrack + i + 1}}`;
    }}
    rangeEl.textContent = `${{firstTrack + 1}}-${{firstTrack + visibleCount}} of ${{nc}}`;
  }}
  updateHeaderAndRange();

  function updateTracker(){{
    if(!player.isPlaying && firstTrack === 0) {{
      // Even when stopped, allow the slider to move and show pattern data
      // for the current row (curRow=0 default).
    }}
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

      for(let i=0;i<visibleCount;i++){{
        const c = firstTrack + i;     // actual channel index in the pattern
        // flat index: pattern * 64 * nc  +  row * nc  +  channel
        const idx = pat * 64 * nc + r * nc + c;
        const ch = modData.patterns[idx]||{{}};
        const period  = ch.period||0;
        const sample  = ch.sample||0;
        const effect  = ch.effect||0;
        const param   = ch.param||0;
        const cell = cells[i];
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
        6: "Disco Inferno + UFOff Dancer (orblivius/finalman/Lallis — dance-floor scene with poi)",
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
    
    for smp in mod.samples:
        start_idx = len(all_samples)
        raw_len = 0
        bw_factor = 1
        if smp['data'] is not None and len(smp['data']) > 0:
            bw_factor, compressed = bw_compress_sample(smp['data'])
            all_samples.extend(compressed.astype(np.float64) / 128.0)
            raw_len = len(compressed)
            all_samples.extend([0.0] * 32)  # zero-padding: pos+1 and pos+2 always safe
        sample_map.append({
            'start':          start_idx,
            'length':         raw_len,
            'repeat_point':   smp['repeat_point'] // bw_factor,
            'repeat_length':  smp['repeat_length'] // bw_factor if smp['repeat_length'] > 2 else smp['repeat_length'],
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

    # ── Per-row absolute-start-time table (variable speed/tempo support) ────
    # Original code used a single SPEED/BPM constant for the whole song. That
    # breaks any module that issues Fxx (set speed/tempo) effects mid-song —
    # the rows after the F effect played at the wrong rate, causing audible
    # silence after sample triggers in songs like hippy.mod which alternate
    # F01 / F02 stutter patterns.
    #
    # Fix: pre-compute the actual time at which each global row STARTS, by
    # walking the song in playback order applying every F effect in turn.
    # _row_start_times[r] = absolute seconds at which global row r begins.
    # _row_speeds[r]      = speed (ticks/row) active at row r — needed so
    #                       pos.tick is computed against the correct speed
    #                       and tick-based effects (volume slide, vibrato,
    #                       note delay/cut) align with the player.
    _row_start_times = [0.0]
    _row_speeds      = []
    _cur_speed       = getattr(mod, 'initial_speed', 6)
    _cur_tempo       = getattr(mod, 'initial_tempo', 125)
    _row_idx         = 0
    for _si, _pi in enumerate(mod.song_positions):
        _start_r = _pat_start_rows[_si]
        # Number of rows actually played from this song position (handles D effects)
        _rows_here = _pat_row_offsets[_si+1] - _pat_row_offsets[_si]
        for _local_r in range(_rows_here):
            _ri = _start_r + _local_r
            # Apply F effects FIRST (ProTracker processes them on tick 0, so
            # they affect the duration of the row they appear in)
            try:
                _row = mod.patterns[_pi][_ri]
            except Exception:
                _row = []
            for _ch in range(_num_ch):
                try:
                    _n = _row[_ch]
                    if _n.get('effect', 0) == 0xF and _n.get('param', 0) > 0:
                        _p = _n['param']
                        if _p < 0x20: _cur_speed = _p
                        else:         _cur_tempo = _p
                except Exception:
                    pass
            _tps = _cur_tempo * 2.0 / 5.0
            _row_dur = _cur_speed / _tps if _tps > 0 else 0.0
            _row_speeds.append(_cur_speed)
            _row_start_times.append(_row_start_times[-1] + _row_dur)
            _row_idx += 1
    assert len(_row_start_times) == _total_song_rows + 1, (
        f"row time table length mismatch: {len(_row_start_times)} vs {_total_song_rows+1}")
    _actual_song_seconds = _row_start_times[-1]
    _f_effect_count = sum(1 for _si, _pi in enumerate(mod.song_positions)
                          for _ri in range(64)
                          for _ch in range(_num_ch)
                          if (lambda n: n is not None and n.get('effect',0)==0xF and n.get('param',0)>0)
                             ((mod.patterns[_pi][_ri][_ch] if _ri < len(mod.patterns[_pi]) and _ch < len(mod.patterns[_pi][_ri]) else None) if _pi < len(mod.patterns) else None))
    if _f_effect_count > 2:   # >2 since initial-speed-set typically uses 1-2 Fxx
        print(f"   🎚️  {_f_effect_count} mid-song F effects detected → using per-row time table")


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
    # _actual_song_seconds is the variable-speed total from the row-time walk;
    # use it whenever F effects shifted the duration away from the naive
    # initial-speed estimate.
    _song_seconds  = _actual_song_seconds if _actual_song_seconds > 0 else _total_song_rows * _row_time
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
    # Per-row time/speed tables (variable-speed support).  Format with enough
    # precision for Shadertoy's ~180s buffer (~6 decimal digits).
    _row_time_str  = ', '.join(f'{t:.6f}' for t in _row_start_times)
    _row_speed_str = ', '.join(map(str, _row_speeds))

    # ========== COMMON TAB ==========
    data_source_comment = "Embedded data (no PNG required)" if use_embedded else f"All data in 1024×1024 RGBA PNG: {png_file}"
    common_glsl = f"""/* ============================================================================
   GLSL (The Last) MOD Player v1.42 (c) 2026 Orblivius
   4+ Tracks support, S3M/MOD loader, 3D Surround, PhatBass, Comb Reverb, FAT, RVQ sample compression, configurable resampler
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
// For S3M, use the channel_settings field from the file header (per-channel
// L/R pan as authored by the composer). For MOD or other formats without
// per-channel pan, fall back to Amiga LRRL convention.
//
// SATELL.S3M for example uses LRLR — both lead voices in pat 1 row 0 are
// on ch 0 (L) and ch 3 (R). Hardcoding LRRL would put both on L and produce
// a left-heavy mix that sounds clipped/distorted on the loud lead side.
const float channelPan[32] = float[]({', '.join([
    (
        # S3M: use file-specified pan when available (channel_settings list
        # exists and contains valid entries: <8=L, 8..15=R, else fall back).
        ('0.0' if (getattr(mod, 'channel_settings', None) and
                   i < len(mod.channel_settings) and
                   (mod.channel_settings[i] & 0x7F) < 8)
         else '1.0' if (getattr(mod, 'channel_settings', None) and
                        i < len(mod.channel_settings) and
                        8 <= (mod.channel_settings[i] & 0x7F) < 16)
         else f'{[0.0,1.0,1.0,0.0][i%4]:.1f}')
        if i < mod.num_channels else '0.5'
    )
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

// ── Per-row absolute-start-time + speed tables ──────────────────────────────
// rowStartTime[r] = seconds at which global row r begins (variable speed/tempo)
// rowSpeed[r]     = ticks-per-row at row r, for tick-based effect math
// rowStartTime has length TOTAL_SONG_ROWS+1 (last entry = total song duration).
// Without these, mid-song Fxx effects (set speed/tempo) are silently ignored
// — every row plays at the initial speed/tempo, which can leave audible gaps
// between sample triggers in stutter-timed modules (cf. hippy.mod's F01/F02).
const float rowStartTime[{_total_song_rows + 1}] = float[]({_row_time_str});
const int   rowSpeed[{_total_song_rows}]     = int[]({_row_speed_str});

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
    int speed;          // ← row-local speed (ticks/row) at the current row
}};

Position getPosition(float time) {{
    Position pos;
    // Total song duration is the last entry of the row-time table — already
    // accounts for every Fxx speed/tempo change applied along the way.
    const float songDuration = rowStartTime[TOTAL_SONG_ROWS];

    float loopedTime;
#if SONG_LOOP_POS == 0
    loopedTime = mod(time, songDuration);
#else
    const float introDur = rowStartTime[{_intro_rows}];
    const float loopDur  = songDuration - introDur;
    if (time < songDuration) {{
        loopedTime = time;
    }} else {{
        loopedTime = introDur + mod(time - songDuration, loopDur);
    }}
#endif

    // ── Binary search rowStartTime[] to find the current GLOBAL row ─────────
    // Was: totalRows = loopedTime / rowTime  (uniform-speed assumption)
    // Now: walks the precomputed time table so mid-song Fxx changes apply.
    int r_lo = 0, r_hi = TOTAL_SONG_ROWS;
    for (int _bi = 0; _bi < 16; _bi++) {{   // ceil(log2(65536)) = 16, plenty
        if (r_lo >= r_hi - 1) break;
        int r_mid = (r_lo + r_hi) >> 1;
        if (rowStartTime[r_mid] <= loopedTime) r_lo = r_mid;
        else r_hi = r_mid;
    }}
    int globalRow = r_lo;
    int curSpeed  = rowSpeed[globalRow];
    float thisRowStart = rowStartTime[globalRow];
    float thisRowEnd   = rowStartTime[globalRow + 1];
    float thisRowDur   = max(thisRowEnd - thisRowStart, 1e-9);

    // Map global row → song position via patRowOffset[] binary search
    int sp_lo = 0, sp_hi = SONG_LENGTH;
    for (int _bi = 0; _bi < 8; _bi++) {{
        if (sp_lo >= sp_hi - 1) break;
        int sp_mid = (sp_lo + sp_hi) >> 1;
        if (patRowOffset[sp_mid] <= globalRow) sp_lo = sp_mid;
        else sp_hi = sp_mid;
    }}
    int sp = sp_lo;
    pos.songPos = sp;
    pos.pattern = songPositions[sp];
    int rowsIntoPos = globalRow - patRowOffset[sp];
    pos.row     = patStartRow[sp] + rowsIntoPos;
    pos.row     = min(pos.row, 63);
    // Tick within the current row, scaled by THIS row's speed (not initial)
    pos.tick    = ((loopedTime - thisRowStart) / thisRowDur) * float(curSpeed);
    pos.rowTime = thisRowDur;
    pos.speed   = curSpeed;

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

    // Use rowStartTime[] for absolute timing — variable speed/tempo aware.
    // Was: (patRowOffset[p] + rowsIntoP) * rowTime  (uniform-speed assumption)
    int   trigGlobalRow = patRowOffset[trigPat] + (trigRow - patStartRow[trigPat]);
    int   curGlobalRow  = patRowOffset[pos.songPos] + (pos.row - patStartRow[pos.songPos]);
    int   trigSpeed     = rowSpeed[trigGlobalRow];
    float triggerTime   = rowStartTime[trigGlobalRow];
    float currentTime   = rowStartTime[curGlobalRow]
                        + (pos.tick / float(pos.speed)) * pos.rowTime;
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
            // Total ticks elapsed since trigger.  Was: elapsed*SPEED/rowTime
            // = elapsed*ticksPerSec — broke under variable BPM. Use the
            // current row's tempo (pos.speed/pos.rowTime = ticksPerSec[r]).
            float _vibTicks = elapsed * float(pos.speed) / pos.rowTime;
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
        // (speed-1) ticks because tick 0 is skipped — use trigger row's
        // actual speed from the per-row table, not the initial SPEED constant
        volume = clamp(volume + (_su>0?_su:-_sd)*(trigSpeed-1), 0, 64);
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
                // Use the speed at THIS scanned row, not the initial SPEED.
                int _fGlobalRow = patRowOffset[_fp] + (_fr - patStartRow[_fp]);
                int _fSpd = rowSpeed[_fGlobalRow];
                volume = clamp(volume+(_su>0?_su:-_sd)*(_fSpd-1), 0, 64);
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
   GLSL (The Last) MOD Player v1.42 (c) 2026 Orblivius
   4+ Tracks support, S3M/MOD loader, 3D Surround, PhatBass, Comb Reverb, FAT, RVQ sample compression, configurable resampler
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

    // 75% Amiga stereo separation, applied to whatever pan the FILE
    // specifies for each channel (channelPan[ch] in Common). For S3M
    // files this is read from header.channel_settings; for MOD files
    // we still default to LRRL via channelPan[].
    //
    // SEP=0.5 maps a "fully panned" channel (0.0 or 1.0) into 0.25..0.75
    // — i.e. 25% bleed into the opposite speaker. SATELL.S3M used to
    // sound L-heavy / R-thin because the old hardcoded chL/chR arrays
    // assumed LRRL by index, which puts ch3 (the R lead) onto L for
    // 8-channel S3Ms whose actual layout is LRLR.
    //
    //   panR_out = 0.25 + 0.5 * channelPan[ch]
    //   panL_out = 1.0 - panR_out  (= 0.75 - 0.5 * channelPan[ch])
    
    // Split into surround bus (ch0,ch3 = outer LEFT pair → Surround L/R)
    // and center bus (ch1,ch2 = inner RIGHT pair → dry center)
    float surrL = 0.0, surrR = 0.0;
    float centL = 0.0, centR = 0.0;
    
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
        float s = getChannelOutput(ch, playbackTime, pos, rowTime);
        float panR = 0.25 + 0.5 * channelPan[ch];   // 0.25..0.75
        float panL = 1.0 - panR;                     // 0.75..0.25
        int cm = ch % 4;
        int ch1 = cm + 1;  // 1-indexed
        bool isSurr = (ch1 == surr_channels.x || ch1 == surr_channels.y);
        if (isSurr) {{ surrL += s * panL; surrR += s * panR; }}
        else        {{ centL += s * panL; centR += s * panR; }}
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
        // Skip Only3D entirely when the delay tap reaches before song-start.
        // Without this guard, the 19-sample delay tap at t=0 sees tW<0 and
        // getPosition() wraps it to song-end via mod() — which means a tiny
        // amount of song-tail audio leaks into the very first samples,
        // audible as a faint click on every voice's attack.
        if (tW >= 0.0) {{
        Position posW = getPosition(tW);
        float wL = 0.0, wR = 0.0;
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
            int cm = ch % 4;
            int ch1 = cm + 1;
            if (ch1 == surr_channels.x || ch1 == surr_channels.y) {{
                float sw = getChannelOutput(ch, tW, posW, rowTime);
                float panR = 0.25 + 0.5 * channelPan[ch];
                float panL = 1.0 - panR;
                wL += sw * panL; wR += sw * panR;
            }}
        }}
        wL *= normFactor; wR *= normFactor;
        float diff = (wL - wR) * ONLY3D_DEPTH;
        surrL += diff;
        surrR -= diff;
        }}  // end if(tW >= 0)
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
        // Skip PhatBass entirely when the delay tap reaches before song-start.
        // 80-sample delay tap means tP<0 for the first 80 output samples;
        // without this guard, getPosition() wraps to song-end and PhatBass
        // (which has cross-pan + 1.5× depth) ends up injecting song-tail
        // audio into voice-attack regions, audible as a click on the lead.
        if (tP < 0.0) {{
            // Skip PhatBass for these initial samples; centL/centR remain
            // unchanged.
        }} else {{
        Position posP = getPosition(tP);
        float pbL = 0.0, pbR = 0.0;
#if PHATBASS_MIX_MODE
        // Mix-wide: take the same per-channel sum but with NO bass filter,
        // then let the listener treat the allpass as a global bass shaper.
        // The allpass only meaningfully shifts <100 Hz, so mids pass through
        // largely unaffected — net effect is sub-bass widening.
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
            float sp = getChannelOutput(ch, tP, posP, rowTime);
            // PhatBass cross-panning: send L's contribution to pbR and R's
            // to pbL (intentional Hilbert-allpass widening). Compute panR
            // from the file's pan, then swap.
            float panR = 0.25 + 0.5 * channelPan[ch];
            float panL = 1.0 - panR;
            pbL += sp * panR;   // <-- swapped (was sp * chR[cm])
            pbR += sp * panL;   // <-- swapped (was sp * chL[cm])
        }}
        // Mix-wide PhatBass — applied to entire mix at PHAT_DEPTH strength.
        // Was previously attenuated 0.25× extra ("must not overwhelm") but
        // user feedback was "I don't hear it at all" — so the cap is gone.
        // PHAT_DEPTH itself controls overall intensity now.
        centL += pbL * normFactor * PHAT_DEPTH;
        centR += pbR * normFactor * PHAT_DEPTH;
#else
        // Per-sample: only bass-detected instruments.
        // Walk back from posP to find the most recently TRIGGERED instrument
        // on this channel. The current row's cell is often empty (inst=0) on
        // continuation rows — using that directly would flip bass→non-bass at
        // every row boundary even though the channel is still sustaining the
        // bass note. The flip created 1-sample steps that FAT amplified into
        // audible clicks. Walking back keeps the classification stable across
        // a sustained note.
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
            int inst = 0;
            int sR = posP.row, sP = posP.songPos;
            for (int lb = 0; lb < 64; lb++) {{
                Note n2 = getNote(sP, sR, ch);
                if (n2.instrument > 0) {{ inst = n2.instrument; break; }}
                sR--;
                if (sR < 0) {{
                    if (sP > 0) {{
                        sP--;
                        sR = patStartRow[sP] + (patRowOffset[sP+1] - patRowOffset[sP]) - 1;
                    }} else {{ break; }}
                }}
            }}
            bool bass = (inst >= 1 && inst <= 31) ? isBass[inst - 1] : false;
            if (bass) {{
                float sp = getChannelOutput(ch, tP, posP, rowTime);
                float panR = 0.25 + 0.5 * channelPan[ch];
                float panL = 1.0 - panR;
                pbL += sp * panR;   // cross-panned (was chR[cm])
                pbR += sp * panL;   // cross-panned (was chL[cm])
            }}
        }}
        centL += pbL * normFactor * PHAT_DEPTH;
        centR += pbR * normFactor * PHAT_DEPTH;
#endif
        }}  // end else (tP >= 0)
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
            // Gate reverb contribution when the delay-line tap reaches
            // BEFORE song-start. Without this, getPosition's mod() wrap
            // makes _tw refer to the END of the song, so the reverb at
            // song t≈0 plays back the song's tail — audible as a click /
            // distortion across the lead voice's attack for the first
            // ~30ms (longest comb delay). The visible symptom was a +0.04
            // bias on samples 0..6 where the source samples are zero.
            if (_tw < 0.0) continue;
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
        # Disco Inferno + UFOff Dancer + Laser
        # Same volumetric cone-light system as viz 6 (analytic-floor fix from
        # Lallis), with a poi-spinning humanoid dancer SDF added in front.
        # Audio hooks wired into:
        #   - dancer body energy (gAudioLevel ← smoothed bands lo+mi+hi)
        #   - beat bounce (iBeatSharp ← BPM-quantized sin pulse + bass boost)
        #   - cone-smoke music color (← mid band)
        # See per-block comments for hook details.
        viz_scene_block = r"""
// === VIZ 6: Disco Inferno + UFOff Dancer + Laser ============================
// All identifiers prefixed _v6_ to coexist cleanly with sibling viz blocks.
// Adapted from Lallis' working "disco_inferno_fixed.glsl" + UFOff Dancer.
// ----------------------------------------------------------------------------
// Audio hooks (read at the top of _VizScene):
//   _v6_audioBass  ← BufferA row 2 px 2  (200ms attack / 700ms release IIR)
//   _v6_audioMid   ← BufferA row 2 px 3
//   _v6_audioHigh  ← BufferA row 2 px 4
//   _v6_audioLevel ← weighted mix → drives dancer pose energy / leg swing /
//                    head wobble / beat bounce amplitude
// Beat-quantized:
//   _v6_iBeatSharp ← pow(|sin(t·BPM/60·π)|, 3) · (1 + bass·1.5)
//                    bass kicks PUNCH the sine pulse so the dancer's bounce
//                    actually fires on the kick instead of just the metronome.
// ----------------------------------------------------------------------------

#define _v6_PI  3.14159265359
#define _v6_TAU 6.28318530718

// ── DISCO INFERNO CONSTANTS (from Lallis' working version) ───────────────────
const float _v6_LIGHT_POW    = 3.;
const float _v6_LIGHT_INTENS = 0.8;
const float _v6_FLOOR_Y      = -0.13;
const float _v6_BIG          = 1e30;
const float _v6_EPSILON      = 1e-10;
const int   _v6_SPOTS        = 3;
const int   _v6_MAX_SPOTS    = 9;
const float _v6_NOTHING      = -0.1;
const float _v6_LIGHT_BASE_W = 0.19;
const float _v6_CONE_W       = 0.2;
const int   _v6_MAX_STEPS    = 100;
const float _v6_MIN_STEP     = 0.0082;
const float _v6_FAR          = 1.;

// ── DANCER CONSTANTS (UFOff dancer, scaled to disco coordinate system) ──────
// Dancer's local SDF lives in a 4-unit space; scaling to 0.06 maps it to
// 0.24 world units. Y offset puts feet exactly on FLOOR_Y.
#define _v6_DANCER_SCALE       0.06
#define _v6_DANCER_BASE_Y      (_v6_FLOOR_Y + 1.5*_v6_DANCER_SCALE)
// ── Dancer animation rate — TEMPO-DRIVEN ────────────────────────────────────
// Constants below were tuned at BPM=125; we now scale them by the song's
// actual BPM so the dance stays in sync regardless of tempo. At 125 BPM
// the legs swing roughly once per beat and the body sways once per bar
// (4 beats), and that musical feel persists at any BPM the song uses.
#define _v6_BPM_SCALE          (float(BPM) / 125.0)
// Slow drift rotation so the dancer faces slightly off-axis over time.
// Drift rate is also BPM-scaled — 0.3 rad/sec at 125 BPM, scaling with tempo.
#define _v6_DANCER_ROTATE      (_v6_PI + 0.4*sin(iTime * 0.3 * _v6_BPM_SCALE))
#define _v6_DANCER_CLOCK       (iTime * 32. * _v6_BPM_SCALE)
#define _v6_DANCER_HEAD_CLOCK  (iTime * 16. * _v6_BPM_SCALE)
#define _v6_LEG_PHASE_OFFSET   _v6_PI
#define _v6_TORSO_SWING        0.2

#define _v6_SMILE_DOTS    25
#define _v6_SMILE_BEND    (0.01 * 2.*abs(sin(iTime*.2)))
#define _v6_EYE_SIZE      0.03
#define _v6_BODY_LOWER    0.3
#define _v6_BODY_UPPER    0.35
#define _v6_HEAD_SIZE     0.40
#define _v6_NECK_LENGTH   0.5
#define _v6_NECK_SIZE     0.15
#define _v6_ARM_UPPER     0.20
#define _v6_ARM_LOWER     0.10
#define _v6_ARM_HAND      0.13
#define _v6_LEG_UPPER     0.13
#define _v6_LEG_LOWER     0.18
#define _v6_FOOT_SIZE     0.15
#define _v6_WINK_INTERVAL (0.5 + _v6_hash(floor(iTime*0.5))*3.5)
#define _v6_EYE_COLOR     vec3(2.05,2.05,0.05)
#define _v6_SMILE_COLOR   vec3(2.05,2.05,0.05)

#define _v6_POI_OFF       abs(sin(iTime*.02))
#define _v6_POI_STRING    2.0
#define _v6_POI_CX        0.74
#define _v6_POI_CZ        0.28
#define _v6_POI_STICK_LEN 0.13
#define _v6_POI_STICK_RAD 0.028
#define _v6_POI_STR_RAD   0.018
// Poi spin rate — also BPM-tied. 7 rev/s at 125 BPM = ~3.4 spins per beat.
// Same _v6_BPM_SCALE so faster songs spin poi proportionally faster.
#define _v6_POI_RPM       (7. * _v6_BPM_SCALE)
#define _v6_POI_CLOCK     (iTime*_v6_POI_RPM)
#define _v6_POI_COL_R     vec3(0.15,4.0,0.4)
#define _v6_POI_COL_L     vec3(3.2,0.15,2.2)

// ── GLOBALS ──────────────────────────────────────────────────────────────────
vec3 _v6_SPOT_POS[_v6_MAX_SPOTS];
vec4 _v6_SPOT_COL[_v6_MAX_SPOTS];
mat3 _v6_SPOT_ROT[_v6_MAX_SPOTS];

float _v6_v3 = 0.;          // debug remnant (kept for fidelity to source)

// Audio-driven globals — set at the top of _VizScene from BufferA bands
float _v6_gAudioLevel = 1.0;   // 0..~2, drives dancer energy
float _v6_audioBass   = 0.0;
float _v6_audioMid    = 0.0;
float _v6_audioHigh   = 0.0;
vec2  _v6_gDancerXZ   = vec2(0., 0.30);

struct _v6_Ray  { vec3 o; vec3 d; };
struct _v6_Isct { float dist; vec3 normal; };
struct _v6_Res  { _v6_Isct start; _v6_Isct end; };

// ── ROTATIONS ────────────────────────────────────────────────────────────────
mat3 _v6_rotx(float a){
    mat3 r;
    r[0]=vec3(1.,0.,0.);
    r[1]=vec3(0.,cos(a),-sin(a));
    r[2]=vec3(0.,sin(a), cos(a));
    return r;
}
mat3 _v6_rotz(float a){
    mat3 r;
    r[0]=vec3( cos(a),-sin(a),0.);
    r[1]=vec3( sin(a), cos(a),0.);
    r[2]=vec3(0.,0.,1.);
    return r;
}

// ── SHARED UTILITIES ─────────────────────────────────────────────────────────
float _v6_hash(float n){ return fract(sin(n)*43758.5453); }
vec3  _v6_dRX(vec3 p,float a){vec2 c=vec2(cos(a),sin(a));return vec3(p.x,c.x*p.y-c.y*p.z,c.y*p.y+c.x*p.z);}
vec3  _v6_dRY(vec3 p,float a){vec2 c=vec2(cos(a),sin(a));return vec3(c.x*p.x+c.y*p.z,p.y,c.x*p.z-c.y*p.x);}
vec3  _v6_dRZ(vec3 p,float a){vec2 c=vec2(cos(a),sin(a));return vec3(c.x*p.x+c.y*p.y,c.x*p.y-c.y*p.x,p.z);}
float _v6_dSeg(vec3 p,vec3 a,vec3 b){vec3 pa=p-a,ba=b-a;float h=clamp(dot(pa,ba)/dot(ba,ba),0.,1.);return length(pa-ba*h);}
float _v6_dUnion(float a,float b,inout float m,float nm){m=(b<a)?nm:m;return min(a,b);}

// ── 3D value noise from RGBA Noise Small on iChannel2 ───────────────────────
// Same texture-based noise the standalone uses on iChannel1. In mod_player
// iChannel1 is BufferA, so the smoke noise sampler moves to iChannel2.
// Bind "RGBA Noise Small" (or any tiling 8-bit noise) to iChannel2 in
// Shadertoy's channel inputs. Without this binding the smoke will be flat.
float _v6_noiseZ(in vec3 x){
    vec3 p = floor(x), f = fract(x);
    f = f*f*(3.-2.*f);
    vec2 uv = (p.xy + vec2(37.,17.)*p.z) + f.xy;
    vec2 rg = 1.5 * vec2(texture(iChannel2, (uv+0.5)/128., 0.).r);
    return mix(rg.x, rg.y, f.z) - 0.5;
}

float _v6_sdCappedCylinder(vec3 p,vec2 h){
    vec2 d=abs(vec2(length(p.xz),p.y))-h;
    return min(max(d.x,d.y),0.)+length(max(d,0.));
}

// ── Smoke cone SDF — bit-flag IDs (1,2,4) ────────────────────────────────────
vec2 _v6_maplight(vec3 orp){
    float t_=iTime*.025, minm=1e4, mm=1e4, hit_ids=0.;
    for(int i=0;i<_v6_SPOTS;++i){
        vec3 rp=orp, _rp=rp;
        rp+=_v6_SPOT_POS[i]; rp*=_v6_SPOT_ROT[i];
        float m_=_v6_sdCappedCylinder(rp,vec2(_v6_CONE_W,1.))-(-_v6_LIGHT_BASE_W+length(rp)*.2);
        float d_=dot(rp,vec3(0.,-1.,0.));
        if(m_<0.&&d_>=0.){
            vec3 uv=_rp+vec3(t_,0.,0.);
            float n=_v6_noiseZ(uv*10.)-.5;
            uv=_rp+vec3(t_*1.2,0.,0.); n+=_v6_noiseZ(uv*22.5)*.5;
            uv=_rp+vec3(t_*2.,0.,0.);  n+=_v6_noiseZ(uv*52.5)*.5;
            uv=_rp+vec3(t_*2.8,0.,0.); n+=_v6_noiseZ(uv*152.5)*.25;
            mm=min(n,m_); mm=min(mm,-.2);
            hit_ids+=exp2(float(i));
        }
        _v6_v3=m_; minm=min(abs(m_),minm);
    }
    if(hit_ids>0.) return vec2(mm,hit_ids);
    return vec2(minm,_v6_NOTHING);
}

void _v6_colorize(in vec4 fgc,in vec3 pos,in vec4 spotcol,float musiccolor,inout vec4 color){
    float flf=inversesqrt(length(pos));
    flf=pow(flf,_v6_LIGHT_POW)*_v6_LIGHT_INTENS;
    color+=fgc*flf*spotcol*musiccolor;
}

vec3 _v6_floorTexture(vec3 pos){
    pos.z+=pos.x*.25;
    float diff=fract(pos.x*.1)-fract(pos.z*.1);
    float fw=fwidth(diff)*1.5;
    return mix(vec3(.7),vec3(1.),smoothstep(-fw,fw,diff));
}

_v6_Res _v6_planeCut(vec3 pos,vec3 normal,_v6_Ray ray){
    ray.o-=pos;
    float rdn=dot(ray.d,normal),ron=dot(ray.o,normal);
    _v6_Res result; result.start.normal=normal; result.end.normal=normal;
    if(ron>0.){
        result.start.dist=_v6_BIG; result.end.dist=-_v6_BIG;
        if(abs(rdn)>_v6_EPSILON){
            float d=-ron/rdn;
            if(d>0.){result.start.dist=d;result.end.dist=_v6_BIG;}
            else{result.start.dist=-_v6_BIG;result.end.dist=d;}
        }
    }else{
        result.start.dist=-_v6_BIG; result.end.dist=_v6_BIG;
        if(abs(rdn)>_v6_EPSILON){
            float d=-ron/rdn;
            if(d>0.){result.start.dist=-_v6_BIG;result.end.dist=d;}
            else{result.start.dist=d;result.end.dist=_v6_BIG;}
        }
    }
    return result;
}

// ── Smoke marcher + analytic floor renderer (working analytic-floor pattern) ─
// musiccolor parameter passed in from _VizScene (was a texture(iChannel0)
// lookup in standalone — replaced with mid-band audio so the cone smoke
// pulses with music instead of black/static).
bool _v6_trace(in vec3 ro, in vec3 rd, float musiccolor, inout vec4 color){
    float tFloor=_v6_BIG;
    if(rd.y<0.) tFloor=(_v6_FLOOR_Y-ro.y)/rd.y;
    vec3 rp=ro; float h=0.;
    float sg =(sin(iTime)        +1.)*.25;
    float sg2=(sin(iTime*.5)     +1.)*.25;
    float sg3=(sin(iTime*.25)    +1.)*.25;
    vec4 spcol1=_v6_SPOT_COL[0]+vec4(0,0,sg,0);
    vec4 spcol2=_v6_SPOT_COL[1]+vec4(0,sg2,0,0);
    vec4 spcol3=_v6_SPOT_COL[2]+vec4(0,0,sg3,0);
    for(int i=0;i<_v6_MAX_STEPS;++i){
        rp+=rd*max(_v6_MIN_STEP,h*.5);
        vec2 hp=_v6_maplight(rp); h=hp.x;
        if(rp.z>_v6_FAR||rp.y<_v6_FLOOR_Y) break;
        if(h<0.){
            vec4 fgc=vec4(abs(h*.05));
            int id=int(hp.y+.5);
            if((id&1)!=0) _v6_colorize(fgc,-_v6_SPOT_POS[0]-rp,spcol1,musiccolor,color);
            if((id&2)!=0) _v6_colorize(fgc,-_v6_SPOT_POS[1]-rp,spcol2,musiccolor,color);
            if((id&4)!=0) _v6_colorize(fgc,-_v6_SPOT_POS[2]-rp,spcol3,musiccolor,color);
        }
    }
    if(tFloor<_v6_BIG){
        vec3 floorMarchPos=ro+rd*tFloor;
        _v6_Ray fr; fr.o=ro; fr.d=rd;
        _v6_Res r=_v6_planeCut(vec3(0.,-18.,0.),vec3(0.,1.,0.),fr);
        vec3 pos=ro+rd*r.start.dist;
        vec4 collo=vec4(-normalize(pos).y*_v6_floorTexture(pos),1.);
        vec4 spotMix=vec4(0.); float wTotal=0.;
        for(int j=0;j<_v6_SPOTS;j++){
            vec3 cp=floorMarchPos+_v6_SPOT_POS[j]; cp*=_v6_SPOT_ROT[j];
            float cm=_v6_sdCappedCylinder(cp,vec2(_v6_CONE_W,1.))-(-_v6_LIGHT_BASE_W+length(cp)*.2);
            float w=1.-smoothstep(-.04,.02,cm);
            spotMix+=_v6_SPOT_COL[j]*w; wTotal+=w;
        }
        if(wTotal>0.) color.rgb=collo.rgb*spotMix.rgb+color.rgb;
        else           color.rgb+=collo.rgb*.05;
        return true;
    }
    return false;
}

// ── IES beam profile (from doc7 / UFOff Dancer) ──────────────────────────────
// Physically shaped cone beam: bright spike at axis, gentle shoulder halo,
// then a specular ring at the edge. theta = angle from beam axis in radians.
float _v6_iesProfile(float theta){
    float b = exp(-theta*theta*40.);
    float h = exp(-theta*theta*8.) * 0.18;
    float s = exp(-theta*theta*0.65) * 0.008;
    float r = exp(-theta*theta*20.) * abs(sin(theta*16.+0.3)) * 0.08;
    return (b + h + s + r) * cos(theta);
}

// ── LASER SYSTEM ─────────────────────────────────────────────────────────────
float _v6_rand2(vec2 p){
    p*=2000.;
    vec3 p3=fract(vec3(p.xyx)*.1031);
    p3+=dot(p3,p3.yzx+33.33);
    return fract((p3.x+p3.y)*p3.z);
}
float _v6_noise2(vec2 p){
    vec2 f=smoothstep(0.,1.,fract(p));
    vec2 i=floor(p);
    float a=_v6_rand2(i),b=_v6_rand2(i+vec2(1,0)),c_=_v6_rand2(i+vec2(0,1)),d_=_v6_rand2(i+vec2(1,1));
    return mix(mix(a,b,f.x),mix(c_,d_,f.x),f.y);
}
float _v6_fbm(vec2 p){
    float a=.5,r_=0.;
    for(int i=0;i<8;i++){r_+=a*_v6_noise2(p);a*=.5;p*=2.8;}
    return r_;
}
float _v6_laser(vec2 p,int num){
    float r_=atan(p.x,p.y);
    float sn=sin(r_*float(num)+iTime*5.);
    float lzr=pow(.5+.5*sn,500.);
    float glow=pow(clamp(sn,0.,1.),20.);
    return lzr+glow;
}
float _v6_clouds(vec2 uv){
    vec2 tv=vec2(0,iTime);
    float c1=_v6_fbm(_v6_fbm(uv*3.)*.75+uv*3.+tv/3.);
    float c2=_v6_fbm(_v6_fbm(uv*2.)*.5 +uv*7.+tv/3.);
    float c3=_v6_fbm(_v6_fbm(uv*10.-tv)*.75+uv*5.+tv/6.);
    float r_=mix(c1,c2,c3*c3); return r_*r_;
}
vec4 _v6_laserLayer(vec2 uv){
    float yScale = 0.2 + 10.*_v6_noise2(vec2(iTime/5.));
    vec2  luv    = vec2(uv.x+0.0001, uv.y*yScale+0.01);
    float l      = (2.+_v6_noise2(vec2(10.-iTime)))*_v6_laser(luv,35);
    float c_     = _v6_clouds(uv);
    vec4  col_   = vec4(0,1,0,1)*(uv.y*l+uv.y*uv.y)*c_;
    return col_*col_;
}

// ── DANCER SDF ───────────────────────────────────────────────────────────────
float _v6_dBein(vec3 p,float pose,float swing,inout float m){
    float d; m=1.;
    d=_v6_dSeg(p,vec3(-0.5,pose,0.),vec3(-(0.625-pose*0.5),0.,0.25))-_v6_FOOT_SIZE;
    d=_v6_dUnion(d,_v6_dSeg(p,vec3(-0.5,pose,0.),vec3(-0.4,1.+pose,swing))-_v6_LEG_UPPER,m,2.);
    d=_v6_dUnion(d,_v6_dSeg(p,vec3(-0.4,1.+pose,swing),vec3(-0.3,2.+pose,0.))-_v6_LEG_LOWER,m,3.);
    return d;
}
float _v6_dOberkoerper(vec3 p,float b,inout float m){
    float d; m=3.;
    d=_v6_dSeg(p,vec3(0.,2.+b,0.),vec3(0.,2.25+b,-b*0.1))-_v6_BODY_LOWER;
    d=_v6_dUnion(d,_v6_dSeg(p,vec3(0.,2.25+b,-b*0.1),vec3(0.,3.25+b,0.))-_v6_BODY_UPPER,m,4.);
    return d;
}
vec3 _v6_poiWrist(float t, float b, float side){
    float roll = sin(2.0*t) * 0.09;
    float lean = sin(t)     * 0.06;
    return vec3(side * (0.27 - lean), 3.00 + b + roll, 0.50);
}
void _v6_poiHeadDir(float t, out vec3 dir, out vec3 tang){
    float snt = sin(t), ct = cos(t);
    float st  = sqrt(snt*snt + 0.08);
    vec3  raw = vec3(-2.0*_v6_POI_CX*snt*st,
                      2.0*_v6_POI_CX*st*ct,
                      snt*_v6_POI_CZ + 0.06);
    dir = normalize(raw);
    float ist  = 1.0 / st;
    float dxdt = -2.0*_v6_POI_CX * ct  * (2.0*snt*snt + 0.08) * ist;
    float dydt =  2.0*_v6_POI_CX * snt * (ct*ct - snt*snt - 0.08) * ist;
    float dzdt =  ct * _v6_POI_CZ;
    tang = normalize(vec3(dxdt, dydt, dzdt) + vec3(0,0,1e-5));
}
float _v6_dPoiAll(vec3 p, float b, inout float m){
    float tR = _v6_POI_CLOCK;
    float tL = _v6_POI_CLOCK + _v6_PI*abs(sin(iTime*.5))*_v6_POI_OFF + _v6_PI*abs(cos(iTime*.5))*(1.-_v6_POI_OFF);
    vec3 wR = _v6_poiWrist(tR, b, +1.0);
    vec3 wL = _v6_poiWrist(tL, b, -1.0);
    vec3 sR = vec3(+0.375, 3.25+b, 0.0);
    vec3 sL = vec3(-0.375, 3.25+b, 0.0);
    vec3 eR = (sR+wR)*0.5 + vec3(+0.05, -0.09, 0.13);
    vec3 eL = (sL+wL)*0.5 + vec3(-0.05, -0.09, 0.13);
    vec3 tipR = wR + normalize(wR - eR) * 0.14;
    vec3 tipL = wL + normalize(wL - eL) * 0.14;
    float d = _v6_dSeg(p, sR, eR)   - _v6_ARM_UPPER;  m = 4.;
    d = _v6_dUnion(d, _v6_dSeg(p, eR, wR)   - _v6_ARM_LOWER, m, 2.);
    d = _v6_dUnion(d, _v6_dSeg(p, wR, tipR) - _v6_ARM_HAND,  m, 2.);
    d = _v6_dUnion(d, _v6_dSeg(p, sL, eL)   - _v6_ARM_UPPER, m, 4.);
    d = _v6_dUnion(d, _v6_dSeg(p, eL, wL)   - _v6_ARM_LOWER, m, 2.);
    d = _v6_dUnion(d, _v6_dSeg(p, wL, tipL) - _v6_ARM_HAND,  m, 2.);
    vec3 dirR, tanR, dirL, tanL;
    _v6_poiHeadDir(tR, dirR, tanR);
    _v6_poiHeadDir(tL, dirL, tanL);
    vec3 hdR = wR + dirR * _v6_POI_STRING;
    vec3 hdL = wL + dirL * _v6_POI_STRING;
    d = _v6_dUnion(d, _v6_dSeg(p, wR, hdR) - _v6_POI_STR_RAD, m, 11.);
    d = _v6_dUnion(d, _v6_dSeg(p, wL, hdL) - _v6_POI_STR_RAD, m, 11.);
    d = _v6_dUnion(d, _v6_dSeg(p, hdR-tanR*(_v6_POI_STICK_LEN*.5), hdR+tanR*(_v6_POI_STICK_LEN*.5)) - _v6_POI_STICK_RAD, m,  9.);
    d = _v6_dUnion(d, _v6_dSeg(p, hdL-tanL*(_v6_POI_STICK_LEN*.5), hdL+tanL*(_v6_POI_STICK_LEN*.5)) - _v6_POI_STICK_RAD, m, 10.);
    return d;
}
float _v6_dKopf(vec3 p,float b,inout float m){
    float d;
    float tcz=_v6_DANCER_HEAD_CLOCK;
    float al = _v6_gAudioLevel;
    vec3 bv=vec3(0.,3.25+b+_v6_NECK_LENGTH,0.); m=1.;
    d=_v6_dSeg(p,vec3(0.,3.25+b,0.),bv)-_v6_NECK_SIZE;
    p-=bv;
    p=_v6_dRY(p,(b*0.25)+(sin(tcz/32.*_v6_PI)*_v6_PI*0.1*al));
    p=_v6_dRZ(p,(b*0.125)+(sin(tcz/64.*_v6_PI)*_v6_PI*0.1*al));
    p=_v6_dRX(p,(b*0.75)+(sin(tcz/8.*_v6_PI)*_v6_PI*0.1*al));
    p+=bv;
    float hf=_v6_HEAD_SIZE;
    vec3 hc=vec3(0.,3.25+b+_v6_NECK_LENGTH+hf*0.5,0.);
    d=_v6_dUnion(d,length(p-hc)-hf,m,2.);
    float winkT=mod(iTime,_v6_WINK_INTERVAL);
    float blink=smoothstep(0.0,0.12,winkT)*(1.0-smoothstep(0.12,0.35,winkT));
    float eyeR=mix(_v6_EYE_SIZE,0.005,blink);
    d=_v6_dUnion(d,length(p-vec3(hc.x-0.38*hf,hc.y+0.26*hf,hc.z+0.88*hf))-eyeR,m,6.);
    d=_v6_dUnion(d,length(p-vec3(hc.x+0.38*hf,hc.y+0.26*hf,hc.z+0.88*hf))-eyeR,m,6.);
    for(int i=0;i<_v6_SMILE_DOTS;i++){
        float t=float(i)/float(_v6_SMILE_DOTS-1)-0.5;
        d=_v6_dUnion(d,length(p-vec3(hc.x+t*0.44*hf, hc.y-0.20*hf+t*t*_v6_SMILE_BEND, hc.z+hf*0.95))-max(0.03*hf,0.018),m,7.);
    }
    float hatBrimY = hc.y + hf * 0.6;
    float brimR    = hf * 1.18;
    float brimH    = hf * 0.05;
    float crownR   = hf * 0.70;
    float crownH   = hf * 0.35;
    vec3 pb = p - vec3(hc.x, hatBrimY, hc.z);
    vec2 dbr = abs(vec2(length(pb.xz), pb.y)) - vec2(brimR, brimH);
    float dHat = min(max(dbr.x, dbr.y), 0.0) + length(max(dbr, 0.0));
    vec3 pc = p - vec3(hc.x, hatBrimY + brimH + crownH, hc.z);
    vec2 dcr = abs(vec2(length(pc.xz), pc.y)) - vec2(crownR, crownH);
    float dCrown = min(max(dcr.x, dcr.y), 0.0) + length(max(dcr, 0.0));
    dHat = min(dHat, dCrown);
    vec3 pDent = p - vec3(hc.x, hatBrimY + brimH + crownH * 1.82, hc.z);
    float dDent = length(pDent) - crownR * 0.60;
    dHat = max(dHat, -dDent);
    d = _v6_dUnion(d, dHat, m, 8.);
    return d;
}

vec2 _v6_dancerSDF(vec3 p){
    float tcz = _v6_DANCER_CLOCK;
    float al  = _v6_gAudioLevel;
    // Beat bounce: BPM-driven sine pulse, bass-kick boosted. The pulse
    // shape pow(|sin|,3) gives a sharp attack and exponential decay
    // — that's what makes the dancer "land" on each beat instead of
    // smoothly oscillating. Bass boost (1 + bass*1.5) means real kicks
    // PUNCH the bounce above the metronome floor.
    float beatSharp = pow(abs(sin(iTime*(float(BPM)/60.)*_v6_PI)), 3.0)
                    * (1.0 + _v6_audioBass * 1.5);
    beatSharp = clamp(beatSharp, 0.0, 1.5);

    float pose   = ((sin((tcz/32.)*_v6_PI)*0.5)+0.5)*0.25 * al;
    float swingR = sin((tcz/8.)*_v6_PI)*0.25 * al;
    float swingL = sin((tcz/8.)*_v6_PI+_v6_LEG_PHASE_OFFSET)*0.25 * al;

    float m=1., m2=0.;
    p+=vec3(0., 1.5 + beatSharp*0.18*al, 0.);
    float dR=_v6_dBein(vec3(-abs(p.x),p.yz),pose,swingR,m);
    float dL=_v6_dBein(vec3(-abs(p.x),p.yz),pose,swingL,m2);
    float d = (p.x>0.0)?dR:dL;
    float d2;
    m = m2;
    d2=_v6_dOberkoerper(p,swingR*_v6_TORSO_SWING,m2); d=_v6_dUnion(d,d2,m,m2);
    d2=_v6_dKopf(p,pose,m2);   d=_v6_dUnion(d,d2,m,m2);
    d2=_v6_dPoiAll(p,pose,m2); d=_v6_dUnion(d,d2,m,m2);
    return vec2(d,m);
}

vec3 _v6_toLocal(vec3 wp){
    vec3 p=wp-vec3(_v6_gDancerXZ.x,_v6_DANCER_BASE_Y,_v6_gDancerXZ.y);
    float c=cos(-_v6_DANCER_ROTATE),s=sin(-_v6_DANCER_ROTATE);
    p=vec3(c*p.x+s*p.z,p.y,-s*p.x+c*p.z);
    return p/_v6_DANCER_SCALE;
}
vec3 _v6_toWorld(vec3 lp){
    lp*=_v6_DANCER_SCALE;
    float c=cos(_v6_DANCER_ROTATE),s=sin(_v6_DANCER_ROTATE);
    lp=vec3(c*lp.x+s*lp.z,lp.y,-s*lp.x+c*lp.z);
    return lp+vec3(_v6_gDancerXZ.x,_v6_DANCER_BASE_Y,_v6_gDancerXZ.y);
}
vec3 _v6_dancerNormal(vec3 wp){
    float e=1e-4/_v6_DANCER_SCALE; vec3 lp=_v6_toLocal(wp);
    float d=_v6_dancerSDF(lp).x;
    return normalize(vec3(_v6_dancerSDF(lp+vec3(e,0,0)).x-d,
                          _v6_dancerSDF(lp+vec3(0,e,0)).x-d,
                          _v6_dancerSDF(lp+vec3(0,0,e)).x-d));
}
float _v6_marchDancer(vec3 ro,vec3 rd,out float matId){
    float tLimit=length(ro-vec3(_v6_gDancerXZ.x,_v6_DANCER_BASE_Y,_v6_gDancerXZ.y))+4.0;
    float t=0.01;
    const float HIT = 4e-4;
    for(int i=0;i<55;i++){
        vec2 res=_v6_dancerSDF(_v6_toLocal(ro+rd*t));
        float d=res.x*_v6_DANCER_SCALE;
        if(d<HIT){matId=res.y;return t;}
        if(t>tLimit) break;
        t+=max(d*0.75,HIT);
    }
    matId=-1.; return -1.;
}
vec3 _v6_dancerMatColor(float m){
    if(m<1.5) return vec3(0.25);
    if(m<2.5) return vec3(0.125);
    if(m<3.5) return vec3(0.75,0.6,0.06);
    if(m<4.5) return vec3(0.12,0.12,0.75);
    if(m<5.5) return vec3(1.,1.,1.);
    if(m<6.5) return _v6_EYE_COLOR;
    if(m<7.5) return _v6_SMILE_COLOR;
    if(m<8.5) return vec3(1.,0.25,0.12);
    if(m<9.5) return _v6_POI_COL_R * 0.25;
    if(m<10.5) return _v6_POI_COL_L * 0.25;
    return vec3(0.14, 0.14, 0.18);
}

// ── Main scene assembly ──────────────────────────────────────────────────────
vec3 _VizScene(vec2 fragCoord){
    vec2  uv     = fragCoord.xy / iResolution.xy;
    float aspect = iResolution.x / iResolution.y;

    // ── Audio hooks: read once per frame, populate globals ───────────────────
    // BufferA row 2 px 2-4 are the IIR-smoothed loudness bands (200ms attack /
    // 700ms release). Reading them here means the dancer's energy follows the
    // audio envelope smoothly without flickering at frame rate.
    // px 2 = bass (~0..200Hz), px 3 = mid (~200..2kHz), px 4 = high (~2..8kHz).
    if (iChannelResolution[1].x > 1.0) {
        _v6_audioBass = texelFetch(iChannel1, ivec2(2, 2), 0).r;
        _v6_audioMid  = texelFetch(iChannel1, ivec2(3, 2), 0).r;
        _v6_audioHigh = texelFetch(iChannel1, ivec2(4, 2), 0).r;
    }
    // Weighted sum → dancer energy. Bass dominant (the kick drives body),
    // mids contribute, highs less so. 0.6 baseline keeps him moving even in
    // silence; cap at ~2.0 so loud passages don't break the IK.
    float loudness = _v6_audioBass*1.0 + _v6_audioMid*0.6 + _v6_audioHigh*0.3;
    _v6_gAudioLevel = clamp(0.6 + loudness * 1.4, 0.5, 2.0);

    // musiccolor for cone smoke: pulses with the mid band so the volumetric
    // cone glow brightens on melody/snare/synth content. Floor 0.15 keeps
    // smoke visible during silence; mid-band scaling is generous (×1.4)
    // because pre-smoothed bands rarely exceed 1.0.
    float musiccolor = _v6_audioMid * 1.4 + 0.15;

    // ── Spotlight setup (from Lallis' working version) ───────────────────────
    vec3 spotpos     = vec3(0.35, -0.25, 0.15);
    _v6_SPOT_POS[0]  = spotpos;
    _v6_SPOT_POS[1]  = vec3(spotpos.x*-1., spotpos.y, spotpos.z);
    _v6_SPOT_POS[2]  = vec3(spotpos.x*-1.5+.5, spotpos.y, spotpos.z);
    _v6_SPOT_COL[0]  = vec4(0.076, 0.443, 0.392, 0.);  // teal
    _v6_SPOT_COL[1]  = vec4(0.753, 0.584, 0.220, 0.);  // amber
    _v6_SPOT_COL[2]  = vec4(0.569, 0.235, 0.294, 0.);  // magenta

    float xrot = -.5 + cos(iTime - .75) * .25;
    float yrot =  .5 + sin(iTime)        * .35;
    _v6_SPOT_ROT[0] = _v6_rotx(xrot)*_v6_rotz(-yrot);
    _v6_SPOT_ROT[1] = _v6_rotx(xrot)*_v6_rotz( yrot);
    _v6_SPOT_ROT[2] = _v6_rotx(xrot)*_v6_rotz(-yrot*2.+_v6_PI/4.);

    // ── Cone-trace ray ───────────────────────────────────────────────────────
    vec3 rd = vec3(uv - vec2(.5), 1.);
    rd.y /= aspect;
    rd = normalize(rd);
    vec3 ro = vec3(0., 0., -1.);

    vec4 col = vec4(0.0);
    _v6_trace(ro, rd, musiccolor, col);

    // Dancer placement.  z=0.0 puts him right between camera (z=-1) and the
    // spot world-origins (z=-0.15), with the cones converging on him from
    // above — matches Lallis' updated dancer GLSL exactly.
    // X-drift rate is BPM-scaled so the side-to-side groove tracks the song.
    // bassSway adds a tiny extra step on bass kicks so he visibly "feels" them.
    float bassSway = _v6_audioBass * 0.04;
    _v6_gDancerXZ = vec2(sin(iTime * 0.18 * _v6_BPM_SCALE) * 0.10 + bassSway, 0.0);

    float dMat;
    float tDancer = _v6_marchDancer(ro, rd, dMat);
    float dancerMask = 0.0;
    if (tDancer > 0.0) {
        float tFloor = (rd.y < -1e-4) ? (_v6_FLOOR_Y - ro.y) / rd.y : 1e9;
        if (tDancer < tFloor) {
            vec3 hp = ro + rd * tDancer;
            // dancerNormal returns gradient in dancer-LOCAL space.
            // toLocal rotates world by -DANCER_ROTATE around Y, so the
            // inverse is +DANCER_ROTATE — apply that to get world-space normal.
            vec3 dnL = _v6_dancerNormal(hp);
            float cDR = cos(_v6_DANCER_ROTATE), sDR = sin(_v6_DANCER_ROTATE);
            vec3 dn = vec3(cDR*dnL.x - sDR*dnL.z,
                           dnL.y,
                           sDR*dnL.x + cDR*dnL.z);
            vec3 dc;
            if (dMat > 8.5 && dMat < 9.5) {
                // Right poi head — emissive, brightens on bass
                dc = _v6_POI_COL_R * (1.8 + _v6_audioBass * 0.8);
            } else if (dMat > 9.5 && dMat < 10.5) {
                // Left poi head — emissive, brightens on highs
                dc = _v6_POI_COL_L * (1.8 + _v6_audioHigh * 0.8);
            } else {
                vec3 matCol = _v6_dancerMatColor(dMat);
                dc = matCol * 0.04;     // ambient — same as doc7 shadeDancer
                // ── Spot lighting — exact doc7 evalLightRaw + shadeDancer math ──
                // lpos  = spot world origin = -SPOT_POS[j]   (verified from maplight)
                // ldir  = beam direction in world space
                //       = local -Y transformed to world
                //       = -(row 1 of SPOT_ROTATION[j])
                //       = -vec3(M[0][1], M[1][1], M[2][1])    (column-major indexing)
                // lc    = SPOT_COL · brightness · iesProfile / (0.2 + dist²·0.14)
                // diff  = max(0, dot(world_normal, surface→light))
                // spec  = Phong specular toward camera (doc7 line-for-line)
                for (int j = 0; j < _v6_SPOTS; j++) {
                    vec3 lpos = -_v6_SPOT_POS[j];
                    vec3 toP  = hp - lpos;            // light → surface
                    float dist = length(toP);
                    if (dist < 0.001) continue;
                    // World-space beam direction (local -Y in world)
                    vec3 ldir = -vec3(_v6_SPOT_ROT[j][0][1],
                                      _v6_SPOT_ROT[j][1][1],
                                      _v6_SPOT_ROT[j][2][1]);
                    float cosT = dot(ldir, toP / dist);
                    if (cosT <= 0.0) continue;        // outside beam cone
                    // evalLightRaw equivalent
                    vec3 lc = _v6_SPOT_COL[j].rgb * 30.0
                              * _v6_iesProfile(acos(clamp(cosT, 0.0, 1.0)))
                              / (0.2 + dist*dist*0.14);
                    // shadeDancer diffuse + specular (doc7 verbatim)
                    vec3 toL = normalize(lpos - hp);
                    float diff = max(0.0, dot(dn, toL));
                    float spec = pow(max(0.0, dot(reflect(-toL, dn), normalize(ro - hp))), 32.0);
                    dc += matCol * lc * diff / _v6_PI + lc * spec * 0.15;
                }
            }
            col.rgb = mix(col.rgb, dc, .95);
            dancerMask = 1.0;
        }
    }

    // ── Poi-head halos (additive bloom along the spinning trajectories) ──────
    // Three glow points per string: head (bright), mid, wrist root.
    {
        // Mirror the SDF pose formula (sin((DANCER_CLOCK/32)*PI)) so this
        // halo math stays in sync if DANCER_CLOCK is retuned.
        float _pose = ((sin((_v6_DANCER_CLOCK/32.)*_v6_PI)*0.5)+0.5)*0.25 * _v6_gAudioLevel;
        float _bs   = pow(abs(sin(iTime*(float(BPM)/60.)*_v6_PI)), 3.0)
                    * (1.0 + _v6_audioBass * 1.5);
        _bs = clamp(_bs, 0.0, 1.5);
        float _yShift = 1.5 + _bs * 0.18 * _v6_gAudioLevel;

        float tR = _v6_POI_CLOCK;
        float tL = _v6_POI_CLOCK + _v6_PI*abs(sin(iTime*.5))*_v6_POI_OFF + _v6_PI*abs(cos(iTime*.5))*(1.-_v6_POI_OFF);

        vec3 wR = _v6_poiWrist(tR, _pose, +1.0);
        vec3 wL = _v6_poiWrist(tL, _pose, -1.0);
        vec3 dirR, tanR, dirL, tanL;
        _v6_poiHeadDir(tR, dirR, tanR);
        _v6_poiHeadDir(tL, dirL, tanL);
        vec3 rhd = wR + dirR * _v6_POI_STRING;
        vec3 lhd = wL + dirL * _v6_POI_STRING;
        vec3 rH = wR, lH = wL;

        rhd.y -= _yShift;
        lhd.y -= _yShift;
        vec3 rW = _v6_toWorld(rhd);
        vec3 lW = _v6_toWorld(lhd);

        // Halo brightness scales with audio so the orbs flare to the music.
        float haloBoost = 1.0 + _v6_audioBass * 0.7 + _v6_audioHigh * 0.4;

        float dR_ = length(cross(rd, rW - ro));
        float dL_ = length(cross(rd, lW - ro));
        float fR_ = step(0.0, dot(rW - ro, rd));
        float fL_ = step(0.0, dot(lW - ro, rd));
        col.rgb += _v6_POI_COL_R * 0.22 * haloBoost / (0.007 + dR_*dR_*3800.) * fR_;
        col.rgb += _v6_POI_COL_L * 0.22 * haloBoost / (0.007 + dL_*dL_*3800.) * fL_;

        vec3 rMW = _v6_toWorld(vec3((rH.x+rhd.x)*0.5, (rH.y+rhd.y)*0.5 - _yShift, (rH.z+rhd.z)*0.5));
        vec3 lMW = _v6_toWorld(vec3((lH.x+lhd.x)*0.5, (lH.y+lhd.y)*0.5 - _yShift, (lH.z+lhd.z)*0.5));
        float dmR = length(cross(rd, rMW - ro));
        float dmL = length(cross(rd, lMW - ro));
        float fmR = step(0.0, dot(rMW - ro, rd));
        float fmL = step(0.0, dot(lMW - ro, rd));
        col.rgb += _v6_POI_COL_R * 0.08 * haloBoost / (0.006 + dmR*dmR*3800.) * fmR;
        col.rgb += _v6_POI_COL_L * 0.08 * haloBoost / (0.006 + dmL*dmL*3800.) * fmL;

        vec3 rHW = _v6_toWorld(vec3(rH.x, rH.y - _yShift, rH.z));
        vec3 lHW = _v6_toWorld(vec3(lH.x, lH.y - _yShift, lH.z));
        float dhR = length(cross(rd, rHW - ro));
        float dhL = length(cross(rd, lHW - ro));
        float fhR = step(0.0, dot(rHW - ro, rd));
        float fhL = step(0.0, dot(lHW - ro, rd));
        col.rgb += _v6_POI_COL_R * 0.036 * haloBoost / (0.005 + dhR*dhR*3800.) * fhR;
        col.rgb += _v6_POI_COL_L * 0.036 * haloBoost / (0.005 + dhL*dhL*3800.) * fhL;
    }

    // ── Laser layer (background, masked by dancer silhouette) ────────────────
    vec4 lzr = _v6_laserLayer(uv);
    col.rgb += lzr.rgb * (1.0 - dancerMask);

    // NaN guard
    col.rgb = mix(vec3(0.0), col.rgb, vec3(equal(col.rgb, col.rgb)));
    col.rgb = clamp(col.rgb, 0.0, 4.0);
    return col.rgb;
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

    // 4D ray direction — y treated as z, z as w (the "extra" dimension)
    vec4 rd = normalize(vec4(U - 0.5*iResolution.xy,
                             iResolution.y,
                             iResolution.y * 2.0)) * 80.0;

    float sc, dotp, totdist = 0.0;
    float tt = iTime;

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

        // Accumulate colour, fading with both distance AND iteration count.
        // exp(-i*i*step*step*1e3) makes early-bailout cheap when the ray
        // wandered into dense regions (large step → strong falloff).
        c += mix(vec3(1.), _v7_H(_v7_M(sc)), 0.9)
             * 0.03 * exp(-i*i*stepsize*stepsize * 1e3);
    }

    return 1.0 - exp(-c);                          // tone-map
}
"""


    image_glsl = f"""/* ============================================================================
   GLSL (The Last) MOD Player v1.42 (c) 2026 Orblivius
   4+ Tracks support, S3M/MOD loader, 3D Surround, PhatBass, Comb Reverb, FAT, RVQ sample compression, configurable resampler
   IMAGE TAB — iChannel0: alphabet texture (shadertoy.com/view/4sf3RB)
                iChannel1: Buffer A (audio + FFT + smoothed bands)
                iChannel2: RGBA Noise Small  ← required for viz 6 smoke turbulence
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
makeStr(printHdr)   _NUM _NUM _NUM _ _G _L _S _L _ _M _O _D _ _P _L _A _Y _E _R _ _V _1 _DOT _4 _2 _ _NUM _NUM _NUM _end
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
    const vec3 TC0    = vec3(0.30,1.00,0.90);   // mint-cyan — leans cyan but greener than spectrum CYAN so it's distinct
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
   GLSL (The Last) MOD Player v1.42 (c) 2026 Orblivius
   4+ Tracks support, S3M/MOD loader, 3D Surround, PhatBass, Comb Reverb, FAT, RVQ sample compression, configurable resampler
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
    'IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiIKTU9EIOKGkiBTaGFkZXJUb3kg'
    'Q29tbW9uIHRhYiBlbmNvZGVyIHdpdGg6CiAgLSBQYXR0ZXJuIGNydW5jaDog'
    'Yml0bWFwICsgZGljdGlvbmFyeSArIG5pYmJsZS1wYWNrZWQgcm93IHNlZWsK'
    'ICAtIFNhbXBsZSBjcnVuY2g6IDMtYml0IGxpbmVhciBwYWNrZWQgKHVuaWZv'
    'cm0gbm9pc2UgZmxvb3Ig4oCUIHN0YWJsZSBhY3Jvc3MgYWxsIHBsYXliYWNr'
    'IHBpdGNoZXMpClRhcmdldDog4omkIDY0IEtCIHRvdGFsIHByaXZhdGUgY29u'
    'c3QgZGF0YSAoTWFjIEFOR0xFL01ldGFsIHNhZmUgem9uZSkKIiIiCmltcG9y'
    'dCBzdHJ1Y3QsIHN5cywgb3MKCmNsYXNzIE1PREZpbGU6CiAgICBkZWYgX19p'
    'bml0X18oc2VsZiwgcGF0aCk6CiAgICAgICAgd2l0aCBvcGVuKHBhdGgsICdy'
    'YicpIGFzIGY6CiAgICAgICAgICAgIHNlbGYuZGF0YSA9IGYucmVhZCgpCiAg'
    'ICAgICAgc2VsZi5wYXJzZSgpCgogICAgZGVmIHBhcnNlKHNlbGYpOgogICAg'
    'ICAgIGQgPSBzZWxmLmRhdGEKICAgICAgICBzZWxmLnRpdGxlID0gZFswOjIw'
    'XS5yc3RyaXAoYidceDAwJykuZGVjb2RlKCdsYXRpbjEnLCAncmVwbGFjZScp'
    'CiAgICAgICAgc2VsZi5zYW1wbGVzX2luZm8gPSBbXQogICAgICAgIGZvciBp'
    'IGluIHJhbmdlKDMxKToKICAgICAgICAgICAgYmFzZSA9IDIwICsgaSozMAog'
    'ICAgICAgICAgICBuYW1lID0gZFtiYXNlOmJhc2UrMjJdLnJzdHJpcChiJ1x4'
    'MDAnKS5kZWNvZGUoJ2xhdGluMScsICdyZXBsYWNlJykKICAgICAgICAgICAg'
    'bGVuZ3RoX3cgICAgID0gc3RydWN0LnVucGFjaygnPkgnLCBkW2Jhc2UrMjI6'
    'YmFzZSsyNF0pWzBdCiAgICAgICAgICAgIGZpbmV0dW5lICAgICA9IGRbYmFz'
    'ZSsyNF0gJiAweDBGCiAgICAgICAgICAgIHZvbHVtZSAgICAgICA9IGRbYmFz'
    'ZSsyNV0KICAgICAgICAgICAgbG9vcF9zdGFydF93ID0gc3RydWN0LnVucGFj'
    'aygnPkgnLCBkW2Jhc2UrMjY6YmFzZSsyOF0pWzBdCiAgICAgICAgICAgIGxv'
    'b3BfbGVuX3cgICA9IHN0cnVjdC51bnBhY2soJz5IJywgZFtiYXNlKzI4OmJh'
    'c2UrMzBdKVswXQogICAgICAgICAgICBzZWxmLnNhbXBsZXNfaW5mby5hcHBl'
    'bmQoZGljdCgKICAgICAgICAgICAgICAgIG5hbWU9bmFtZSwgbGVuZ3RoPWxl'
    'bmd0aF93KjIsIGZpbmV0dW5lPWZpbmV0dW5lLAogICAgICAgICAgICAgICAg'
    'dm9sdW1lPXZvbHVtZSwgbG9vcF9zdGFydD1sb29wX3N0YXJ0X3cqMiwgbG9v'
    'cF9sZW49bG9vcF9sZW5fdyoyKSkKICAgICAgICBzZWxmLnNvbmdfbGVuZ3Ro'
    'ID0gZFs5NTBdCiAgICAgICAgc2VsZi5wYXR0ZXJuX29yZGVyID0gbGlzdChk'
    'Wzk1Mjo5NTIrMTI4XSkKICAgICAgICBzZWxmLm1hZ2ljID0gZFsxMDgwOjEw'
    'ODRdCiAgICAgICAgIyBEZXRlY3QgY2hhbm5lbCBjb3VudCBmcm9tIHNpZ25h'
    'dHVyZQogICAgICAgIHNpZyA9IHNlbGYubWFnaWMKICAgICAgICBpZiBzaWcg'
    'aW4gKGInTS5LLicsIGInTSFLIScsIGInTSZLIScsIGInTi5ULicsIGInRkxU'
    'NCcsIGInNENITicpOgogICAgICAgICAgICBzZWxmLm51bV9jaGFubmVscyA9'
    'IDQKICAgICAgICBlbGlmIHNpZyA9PSBiJ0ZMVDgnIG9yIHNpZyBpbiAoYidP'
    'Q1RBJywgYidDRDgxJywgYidPS1RBJyk6CiAgICAgICAgICAgIHNlbGYubnVt'
    'X2NoYW5uZWxzID0gOAogICAgICAgIGVsaWYgbGVuKHNpZykgPT0gNCBhbmQg'
    'c2lnWzE6NF0gPT0gYidDSE4nIGFuZCBzaWdbMDoxXS5pc2RpZ2l0KCk6CiAg'
    'ICAgICAgICAgIHNlbGYubnVtX2NoYW5uZWxzID0gaW50KHNpZ1swOjFdKQog'
    'ICAgICAgIGVsaWYgbGVuKHNpZykgPT0gNCBhbmQgc2lnWzI6NF0gPT0gYidD'
    'SCcgYW5kIHNpZ1swOjFdLmlzZGlnaXQoKSBhbmQgc2lnWzE6Ml0uaXNkaWdp'
    'dCgpOgogICAgICAgICAgICBzZWxmLm51bV9jaGFubmVscyA9IGludChzaWdb'
    'MDoyXSkKICAgICAgICBlbGlmIGxlbihzaWcpID09IDQgYW5kIHNpZ1s6M10g'
    'PT0gYidURFonIGFuZCBzaWdbMzo0XS5pc2RpZ2l0KCk6CiAgICAgICAgICAg'
    'IHNlbGYubnVtX2NoYW5uZWxzID0gaW50KHNpZ1szOjRdKQogICAgICAgIGVs'
    'c2U6CiAgICAgICAgICAgIHNlbGYubnVtX2NoYW5uZWxzID0gNAogICAgICAg'
    'IHNlbGYubnVtX3BhdHRlcm5zID0gbWF4KHNlbGYucGF0dGVybl9vcmRlcls6'
    'c2VsZi5zb25nX2xlbmd0aF0pICsgMQogICAgICAgICMgRWFjaCBwYXR0ZXJu'
    'IHJvdyA9IG51bV9jaGFubmVscyDDlyA0IGJ5dGVzOyA2NCByb3dzL3BhdHRl'
    'cm4KICAgICAgICBwYXRfc2l6ZSA9IDY0ICogc2VsZi5udW1fY2hhbm5lbHMg'
    'KiA0CiAgICAgICAgc2VsZi5wYXR0ZXJucyA9IFtdCiAgICAgICAgb2ZmID0g'
    'MTA4NAogICAgICAgIGZvciBwIGluIHJhbmdlKHNlbGYubnVtX3BhdHRlcm5z'
    'KToKICAgICAgICAgICAgc2VsZi5wYXR0ZXJucy5hcHBlbmQoZFtvZmY6b2Zm'
    'K3BhdF9zaXplXSkKICAgICAgICAgICAgb2ZmICs9IHBhdF9zaXplCiAgICAg'
    'ICAgIyBTYW1wbGVzIChyYXcgc2lnbmVkIDgtYml0IGJ5dGVzKQogICAgICAg'
    'IHNlbGYuc2FtcGxlX2J5dGVzID0gW10KICAgICAgICBmb3IgcyBpbiBzZWxm'
    'LnNhbXBsZXNfaW5mbzoKICAgICAgICAgICAgc2VsZi5zYW1wbGVfYnl0ZXMu'
    'YXBwZW5kKGRbb2ZmOm9mZitzWydsZW5ndGgnXV0pCiAgICAgICAgICAgIG9m'
    'ZiArPSBzWydsZW5ndGgnXQoKIyDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZAKIyBQQVRURVJOIENSVU5DSDogYml0bWFwICsgZGljdCAr'
    'IG5pYmJsZS1zZWVrCiMg4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQCgpFTVBUWV9OT1RFID0gYidceDAwXHgwMFx4MDBceDAwJwoKZGVm'
    'IGVuY29kZV9wYXR0ZXJucyhtb2QpOgogICAgIiIiUmV0dXJucyBkaWN0IG9m'
    'IGFsbCBwYXR0ZXJuIGRhdGEgc3RydWN0dXJlcy4iIiIKICAgICMgQnVpbGQg'
    'ZmxhdCBsaXN0IG9mIDQtYnl0ZSBub3RlcyBpbiBvcmRlcjogcGF0IDAuLk4t'
    'MSwgcm93IDAuLjYzLCBjaCAwLi4zLgogICAgIyBGaXJzdCBhcHBseSBQcm9U'
    'cmFja2VyIHBhcmFtLW1lbW9yeSByZXdyaXRpbmcgaW4gc29uZy1wb3NpdGlv'
    'biBwbGF5YmFjawogICAgIyBvcmRlcjogd2hlbiBhIG5vdGUgaGFzIGVmZmVj'
    'dCAxLzIvMy80LzUvNi9BIGFuZCBwYXJhbT09MCwgc3Vic3RpdHV0ZSB0aGUK'
    'ICAgICMgbGFzdCBub24temVybyBwYXJhbSBzZWVuIGZvciB0aGF0IGVmZmVj'
    'dCBvbiB0aGlzIGNoYW5uZWwuICBUaGlzIG1ha2VzCiAgICAjIHRvbmUtcG9y'
    'dGEgcnVucyBsaWtlICIzMDAgMzAwIDMwMCIgY29udGludWUgd2l0aCB0aGUg'
    'cHJldmlvdXMgc2xpZGUgcmF0ZQogICAgIyDigJQgcmVxdWlyZWQgZm9yIG1h'
    'bnkgTU9EcyAoaW5jbC4gR1NMSU5HRVIgcGF0dGVybiAzKS4KICAgIE5DID0g'
    'bW9kLm51bV9jaGFubmVscwogICAgcm93X3N0cmlkZSA9IE5DICogNAoKICAg'
    'ICMgV2FsayBzb25nIHBvc2l0aW9ucyB0byBmaW5kIHBhcmFtLW1lbW9yeSBj'
    'aGFpbnMgcGVyIGNoYW5uZWwuCiAgICAjIEVmZmVjdCBncm91cHMgdGhhdCBz'
    'aGFyZSBtZW1vcnk6CiAgICAjICAgMHgxIChwb3J0YSB1cCksIDB4MiAocG9y'
    'dGEgZG93biksIDB4MyAodG9uZSBwb3J0YSksIDB4NSAodG9uZSt2b2wpLAog'
    'ICAgIyAgIDB4NCAodmlicmF0byksIDB4NiAodmliK3ZvbCksIDB4QSAodm9s'
    'IHNsaWRlKQogICAgIyBXZSByZXdyaXRlIHRoZSBpbi1tZW1vcnkgcGF0dGVy'
    'biBieXRlcyAoYSBjb3B5KSBzbyBlbmNvZGluZyBzZWVzIHRoZQogICAgIyBj'
    'b3JyZWN0ZWQgcGFyYW1zLiAgQnVpbGQgYSBmcmVzaCBwZXItcGF0dGVybiBu'
    'b3RlIGxpc3Qgd2l0aCByZXdyaXRlcy4KICAgICMgVXNlIG1vZC5wYXR0ZXJu'
    'X29yZGVyIChlbmNvZGVyIE1PREZpbGUgZXF1aXZhbGVudCBvZiBzb25nX3Bv'
    'c2l0aW9ucykuCiAgICBfc29uZ19vcmRlciA9IGdldGF0dHIobW9kLCAncGF0'
    'dGVybl9vcmRlcicsIE5vbmUpIG9yIGdldGF0dHIobW9kLCAnc29uZ19wb3Np'
    'dGlvbnMnLCBbXSkKICAgIHBhdF9jb3BpZXMgPSB7fQogICAgbGFzdF9wYXJh'
    'bSA9IFt7fSBmb3IgXyBpbiByYW5nZShOQyldICAjIGxhc3RfcGFyYW1bY2hd'
    'W2VmZmVjdF0gPSBsYXN0IG5vbnplcm8gcGFyYW0KICAgIHJld3JpdHRlbl9u'
    'b3Rlc19jb3VudCA9IDAKICAgIGZvciBzcCBpbiBfc29uZ19vcmRlcls6Z2V0'
    'YXR0cihtb2QsICdzb25nX2xlbmd0aCcsIGxlbihfc29uZ19vcmRlcikpXToK'
    'ICAgICAgICBpZiBzcCBub3QgaW4gcGF0X2NvcGllczoKICAgICAgICAgICAg'
    'cGF0X2NvcGllc1tzcF0gPSBieXRlYXJyYXkobW9kLnBhdHRlcm5zW3NwXSkK'
    'ICAgICAgICAjIE5vdGU6IGEgcGF0dGVybiBtYXkgYXBwZWFyIGF0IG11bHRp'
    'cGxlIHNvbmcgcG9zaXRpb25zOyB3ZSBhcHBseQogICAgICAgICMgcmV3cml0'
    'ZXMgaW4gcGxheWJhY2sgb3JkZXIgc28gbWVtb3J5IHN0YXRlIHByb3BhZ2F0'
    'ZXMgYWNyb3NzIHRoZW0uCiAgICAgICAgIyBSZXdyaXRpbmcgYSBwYXR0ZXJu'
    'IHRoYXQgaXMgcmV1c2VkIGxhdGVyIG1lYW5zIHRoZSBzZWNvbmQgdmlzaXQK'
    'ICAgICAgICAjIHVzZXMgdGhlIGFscmVhZHktcmV3cml0dGVuIHBhcmFtcywg'
    'd2hpY2ggaXMgYWNjZXB0YWJsZSBzaW5jZSB0aGUKICAgICAgICAjIGxhc3Rf'
    'cGFyYW0gc3RhdGUgYXQgc2Vjb25kIHZpc2l0IHdvdWxkIG5hdHVyYWxseSBh'
    'bHNvIGhhdmUgdGhvc2UuCiAgICAjIFJlc2V0IGZvciBhY3R1YWwgcmV3cml0'
    'ZSB3YWxrCiAgICBsYXN0X3BhcmFtID0gW3t9IGZvciBfIGluIHJhbmdlKE5D'
    'KV0KICAgIHZpc2l0ZWRfa2V5cyA9IHNldCgpCiAgICBmb3Igc3AgaW4gX3Nv'
    'bmdfb3JkZXJbOmdldGF0dHIobW9kLCAnc29uZ19sZW5ndGgnLCBsZW4oX3Nv'
    'bmdfb3JkZXIpKV06CiAgICAgICAgcGF0X2NvcHkgPSBwYXRfY29waWVzW3Nw'
    'XQogICAgICAgIGZvciByb3cgaW4gcmFuZ2UoNjQpOgogICAgICAgICAgICBm'
    'b3IgY2ggaW4gcmFuZ2UoTkMpOgogICAgICAgICAgICAgICAgYmFzZSA9IHJv'
    'dypyb3dfc3RyaWRlICsgY2gqNAogICAgICAgICAgICAgICAga2V5ID0gKHNw'
    'LCByb3csIGNoKQogICAgICAgICAgICAgICAgaWYga2V5IGluIHZpc2l0ZWRf'
    'a2V5czogY29udGludWUgICMgYXZvaWQgZG91YmxlLXJld3JpdGUgb24gcGF0'
    'dGVybiByZXVzZQogICAgICAgICAgICAgICAgdmlzaXRlZF9rZXlzLmFkZChr'
    'ZXkpCiAgICAgICAgICAgICAgICAjIERlY29kZSBub3RlOiBieXRlcyBbcGVy'
    'aW9kX2hpLCBwZXJpb2RfbG8sIHNhbXBsZV9sb3xlZmZlY3QsIHBhcmFtXQog'
    'ICAgICAgICAgICAgICAgIyBNT0QgbGF5b3V0OiBieXRlMCA9IHNhbXBsZV9o'
    'aSg0KSB8IHBlcmlvZF9oaSg0KQogICAgICAgICAgICAgICAgIyAgICAgICAg'
    'ICAgICBieXRlMSA9IHBlcmlvZF9sbyg4KQogICAgICAgICAgICAgICAgIyAg'
    'ICAgICAgICAgICBieXRlMiA9IHNhbXBsZV9sbyg0KSB8IGVmZmVjdCg0KQog'
    'ICAgICAgICAgICAgICAgIyAgICAgICAgICAgICBieXRlMyA9IHBhcmFtCiAg'
    'ICAgICAgICAgICAgICBiMCwgYjEsIGIyLCBiMyA9IHBhdF9jb3B5W2Jhc2Vd'
    'LCBwYXRfY29weVtiYXNlKzFdLCBwYXRfY29weVtiYXNlKzJdLCBwYXRfY29w'
    'eVtiYXNlKzNdCiAgICAgICAgICAgICAgICBlZmZlY3QgPSBiMiAmIDB4MEYK'
    'ICAgICAgICAgICAgICAgIHBhcmFtICA9IGIzCiAgICAgICAgICAgICAgICAj'
    'IFByb1RyYWNrZXIgcGFyYW0tbWVtb3J5IHJ1bGVzOgogICAgICAgICAgICAg'
    'ICAgIwogICAgICAgICAgICAgICAgIyAgIDF4eCAocG9ydGEgdXApICAgICAg'
    'ICDigJQgcGFyYW09MCDihpIgdXNlIGxhc3QgMXh4CiAgICAgICAgICAgICAg'
    'ICAjICAgMnh4IChwb3J0YSBkb3duKSAgICAgIOKAlCBwYXJhbT0wIOKGkiB1'
    'c2UgbGFzdCAyeHgKICAgICAgICAgICAgICAgICMgICAzeHggKHRvbmUgcG9y'
    'dGEpICAgICAg4oCUIHBhcmFtPTAg4oaSIHVzZSBsYXN0IDN4eAogICAgICAg'
    'ICAgICAgICAgIwogICAgICAgICAgICAgICAgIyAgIDR4eCAodmlicmF0bykg'
    'ICAgICAgICDigJQgTklCQkxFIG1lbW9yeTogaGlnaCBuaWI9MCBrZWVwcwog'
    'ICAgICAgICAgICAgICAgIyAgICAgICAgICAgICAgICAgICAgICAgICAgICBw'
    'cmlvciBzcGVlZCwgbG93IG5pYj0wIGtlZXBzCiAgICAgICAgICAgICAgICAj'
    'ICAgICAgICAgICAgICAgICAgICAgICAgICAgIHByaW9yIGRlcHRoLiAgV2Ug'
    'ZG9uJ3QgcmV3cml0ZQogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICAg'
    'ICAgICAgICAgICAgICBoZXJlIOKAlCBHTFNML0hUTUwgZG8gbmliYmxlLWxl'
    'dmVsCiAgICAgICAgICAgICAgICAjICAgICAgICAgICAgICAgICAgICAgICAg'
    'ICAgIGhhbmRsaW5nLiAgT25seSByZXdyaXRlIHBhcmFtPTAKICAgICAgICAg'
    'ICAgICAgICMgICAgICAgICAgICAgICAgICAgICAgICAgICAgKHdob2xlIGJ5'
    'dGUgemVybykg4oaSIHVzZSBsYXN0IDR4eC4KICAgICAgICAgICAgICAgICMg'
    'ICA3eHggKHRyZW1vbG8pICAgICAgICAg4oCUIE5JQkJMRSBtZW1vcnkgbGlr'
    'ZSA0eHguCiAgICAgICAgICAgICAgICAjCiAgICAgICAgICAgICAgICAjICAg'
    'NXh4IChjb250aW51ZSB0b25lIHBvcnRhICsgdm9sIHNsaWRlKSDigJQgcGFy'
    'YW0gYnl0ZSBpcwogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICAgICAg'
    'ICAgICAgICAgICBWT0wtU0xJREUgT05MWS4gIDUwMCA9IGNvbnRpbnVlCiAg'
    'ICAgICAgICAgICAgICAjICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNs'
    'aWRlIHdpdGggTk8gdm9sIGNoYW5nZTsgdmFsaWQKICAgICAgICAgICAgICAg'
    'ICMgICAgICAgICAgICAgICAgICAgICAgICAgICAgY29tbWFuZCwgZG8gTk9U'
    'IHJld3JpdGUuCiAgICAgICAgICAgICAgICAjICAgNnh4IChjb250aW51ZSB2'
    'aWJyYXRvICsgdm9sIHNsaWRlKSDigJQgc2FtZSBhcyA1eHg7CiAgICAgICAg'
    'ICAgICAgICAjICAgICAgICAgICAgICAgICAgICAgICAgICAgIHBhcmFtIGJ5'
    'dGUgaXMgdm9sLXNsaWRlIG9ubHkuCiAgICAgICAgICAgICAgICAjICAgICAg'
    'ICAgICAgICAgICAgICAgICAgICAgIDYwMCA9IGNvbnRpbnVlIHZpYnJhdG8s'
    'IG5vIHZvbAogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICAgICAgICAg'
    'ICAgICAgICBzbGlkZTsgdmFsaWQgY29tbWFuZC4KICAgICAgICAgICAgICAg'
    'ICMgICBBeHggKHZvbCBzbGlkZSkgICAgICAg4oCUIEEwMCA9IG5vLW9wIGlu'
    'IFBUIChOT1QgbWVtb3J5KS4KICAgICAgICAgICAgICAgICMgICAgICAgICAg'
    'ICAgICAgICAgICAgICAgICAgRG8gTk9UIHJld3JpdGUuCiAgICAgICAgICAg'
    'ICAgICBpZiBlZmZlY3QgaW4gKDB4MSwgMHgyLCAweDMpOgogICAgICAgICAg'
    'ICAgICAgICAgIGlmIHBhcmFtID09IDAgYW5kIGVmZmVjdCBpbiBsYXN0X3Bh'
    'cmFtW2NoXToKICAgICAgICAgICAgICAgICAgICAgICAgbmV3X3BhcmFtID0g'
    'bGFzdF9wYXJhbVtjaF1bZWZmZWN0XQogICAgICAgICAgICAgICAgICAgICAg'
    'ICBwYXRfY29weVtiYXNlKzNdID0gbmV3X3BhcmFtCiAgICAgICAgICAgICAg'
    'ICAgICAgICAgIHJld3JpdHRlbl9ub3Rlc19jb3VudCArPSAxCiAgICAgICAg'
    'ICAgICAgICAgICAgZWxpZiBwYXJhbSAhPSAwOgogICAgICAgICAgICAgICAg'
    'ICAgICAgICBsYXN0X3BhcmFtW2NoXVtlZmZlY3RdID0gcGFyYW0KICAgICAg'
    'ICAgICAgICAgIGVsaWYgZWZmZWN0IGluICgweDQsIDB4Nyk6CiAgICAgICAg'
    'ICAgICAgICAgICAgIyBXaG9sZS1ieXRlPTAg4oaSIHVzZSBsYXN0IHdob2xl'
    'LWJ5dGUgbWVtb3J5LgogICAgICAgICAgICAgICAgICAgICMgTm9uLXplcm86'
    'IGFsc28gc3RvcmUgYXMgbGFzdC1ieXRlIG1lbW9yeSAobmliYmxlLWxldmVs'
    'CiAgICAgICAgICAgICAgICAgICAgIyBoYW5kbGluZyBpcyBkb25lIGJ5IEhU'
    'TUwvR0xTTCBkdXJpbmcgcGxheWJhY2spLgogICAgICAgICAgICAgICAgICAg'
    'IGlmIHBhcmFtID09IDAgYW5kIGVmZmVjdCBpbiBsYXN0X3BhcmFtW2NoXToK'
    'ICAgICAgICAgICAgICAgICAgICAgICAgbmV3X3BhcmFtID0gbGFzdF9wYXJh'
    'bVtjaF1bZWZmZWN0XQogICAgICAgICAgICAgICAgICAgICAgICBwYXRfY29w'
    'eVtiYXNlKzNdID0gbmV3X3BhcmFtCiAgICAgICAgICAgICAgICAgICAgICAg'
    'IHJld3JpdHRlbl9ub3Rlc19jb3VudCArPSAxCiAgICAgICAgICAgICAgICAg'
    'ICAgZWxpZiBwYXJhbSAhPSAwOgogICAgICAgICAgICAgICAgICAgICAgICBs'
    'YXN0X3BhcmFtW2NoXVtlZmZlY3RdID0gcGFyYW0KICAgICAgICAgICAgICAg'
    'ICMgNS82L0E6IG5vIHBhcmFtLW1lbW9yeSByZXdyaXRpbmcgKHRoZWlyIHBh'
    'cmFtPTAgaXMgbWVhbmluZ2Z1bCkKCiAgICBub3RlcyA9IFtdCiAgICBmb3Ig'
    'cGF0IGluIHJhbmdlKG1vZC5udW1fcGF0dGVybnMpOgogICAgICAgIGlmIHBh'
    'dCBpbiBwYXRfY29waWVzOgogICAgICAgICAgICBwZGF0YSA9IHBhdF9jb3Bp'
    'ZXNbcGF0XQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHBkYXRhID0gbW9k'
    'LnBhdHRlcm5zW3BhdF0KICAgICAgICBmb3Igcm93IGluIHJhbmdlKDY0KToK'
    'ICAgICAgICAgICAgZm9yIGNoIGluIHJhbmdlKE5DKToKICAgICAgICAgICAg'
    'ICAgIGJhc2UgPSByb3cqcm93X3N0cmlkZSArIGNoKjQKICAgICAgICAgICAg'
    'ICAgIG5vdGVzLmFwcGVuZChieXRlcyhwZGF0YVtiYXNlOmJhc2UrNF0pKQog'
    'ICAgaWYgcmV3cml0dGVuX25vdGVzX2NvdW50ID4gMDoKICAgICAgICBwcmlu'
    'dChmIiAgIOKame+4jyAgUGFyYW0tbWVtb3J5OiB7cmV3cml0dGVuX25vdGVz'
    'X2NvdW50fSBwYXJhbT0wIGVmZmVjdHMgcmV3cml0dGVuIHdpdGggcHJldmlv'
    'dXMgdmFsdWVzIikKICAgIHRvdGFsX25vdGVzID0gbGVuKG5vdGVzKQogICAg'
    'bnVtX3Jvd3MgICAgPSBtb2QubnVtX3BhdHRlcm5zICogNjQKCiAgICAjIFVu'
    'aXF1ZSBub24tZW1wdHkgbm90ZXMg4oaSIGRpY3Rpb25hcnkKICAgIHVuaXEg'
    'PSBzb3J0ZWQoc2V0KG4gZm9yIG4gaW4gbm90ZXMgaWYgbiAhPSBFTVBUWV9O'
    'T1RFKSkKICAgIGlkeF9ieXRlcyA9IDEgaWYgbGVuKHVuaXEpIDw9IDI1NiBl'
    'bHNlIDIKICAgIGFzc2VydCBsZW4odW5pcSkgPD0gNjU1MzYsIGYidG9vIG1h'
    'bnkgdW5pcXVlIG5vdGVzOiB7bGVuKHVuaXEpfSIKICAgIG5vdGVfdG9faWR4'
    'ID0ge246aSBmb3IgaSxuIGluIGVudW1lcmF0ZSh1bmlxKX0KCiAgICAjIEJp'
    'dG1hcCAoMSBiaXQgcGVyIG5vdGUsIExTQi1maXJzdCB3aXRoaW4gZWFjaCBi'
    'eXRlKQogICAgYml0bWFwID0gYnl0ZWFycmF5KCh0b3RhbF9ub3RlcyArIDcp'
    'IC8vIDgpCiAgICBmb3IgaSwgbiBpbiBlbnVtZXJhdGUobm90ZXMpOgogICAg'
    'ICAgIGlmIG4gIT0gRU1QVFlfTk9URToKICAgICAgICAgICAgYml0bWFwW2kg'
    'Pj4gM10gfD0gMSA8PCAoaSAmIDcpCgogICAgIyBJbmRleCBzdHJlYW0gKDEg'
    'b3IgMiBieXRlcyBwZXIgbm9uLWVtcHR5IG5vdGUsIGxpdHRsZS1lbmRpYW4g'
    'aWYgMkIpCiAgICBpZHhfc3RyZWFtID0gYnl0ZWFycmF5KCkKICAgIGZvciBu'
    'IGluIG5vdGVzOgogICAgICAgIGlmIG4gIT0gRU1QVFlfTk9URToKICAgICAg'
    'ICAgICAgaSA9IG5vdGVfdG9faWR4W25dCiAgICAgICAgICAgIGlmIGlkeF9i'
    'eXRlcyA9PSAxOgogICAgICAgICAgICAgICAgaWR4X3N0cmVhbS5hcHBlbmQo'
    'aSkKICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgIGlkeF9zdHJl'
    'YW0uYXBwZW5kKGkgJiAweEZGKQogICAgICAgICAgICAgICAgaWR4X3N0cmVh'
    'bS5hcHBlbmQoKGkgPj4gOCkgJiAweEZGKQoKICAgICMgUGVyLXJvdyBjb3Vu'
    'dDogY291bnQgb2Ygbm9uLWVtcHR5IG5vdGVzIElOIHRoaXMgcm93ICgwLi40'
    'KS4KICAgIHBlcl9yb3dfY291bnQgPSBbXQogICAgZm9yIHJvdyBpbiByYW5n'
    'ZShudW1fcm93cyk6CiAgICAgICAgY291bnQgPSBzdW0oMSBmb3IgY2ggaW4g'
    'cmFuZ2UoTkMpIGlmIG5vdGVzW3JvdypOQyArIGNoXSAhPSBFTVBUWV9OT1RF'
    'KQogICAgICAgIHBlcl9yb3dfY291bnQuYXBwZW5kKGNvdW50KQoKICAgICMg'
    'UHJlZml4IHN1bTogcHJlZml4W3Jvd10gPSBub24tZW1wdHkgY291bnQgaW4g'
    'cm93cyBbMCwgcm93KSA9IHJhbmsgYXQgc3RhcnQgb2Ygcm93LgogICAgIyBT'
    'dG9yZWQgYXMgMTYtYml0IExFIHdvcmRzIHNvIGRlY29kZXIgaXMgTygxKS4K'
    'ICAgICMgUmFuZ2U6IDAgdG8gfnRvdGFsX25vbl9lbXB0eSAo4omkIDU4ODgg'
    'Zm9yIDIzLXBhdCBNT0QpIOKGkiBmaXRzIGVhc2lseSBpbiAxNiBiaXRzLgog'
    'ICAgcHJlZml4ID0gWzBdICogbnVtX3Jvd3MKICAgIHJ1bm5pbmcgPSAwCiAg'
    'ICBmb3Igcm93IGluIHJhbmdlKG51bV9yb3dzKToKICAgICAgICBwcmVmaXhb'
    'cm93XSA9IHJ1bm5pbmcKICAgICAgICBydW5uaW5nICs9IHBlcl9yb3dfY291'
    'bnRbcm93XQoKICAgIHJvd19zZWVrX2J5dGVzID0gYnl0ZWFycmF5KCkKICAg'
    'IGZvciB2IGluIHByZWZpeDoKICAgICAgICBhc3NlcnQgMCA8PSB2IDwgNjU1'
    'MzYsIGYicHJlZml4IHt2fSBvdmVyZmxvd3MgMTYgYml0cyIKICAgICAgICBy'
    'b3dfc2Vla19ieXRlcy5hcHBlbmQodiAmIDB4RkYpCiAgICAgICAgcm93X3Nl'
    'ZWtfYnl0ZXMuYXBwZW5kKCh2ID4+IDgpICYgMHhGRikKCiAgICByZXR1cm4g'
    'ZGljdCgKICAgICAgICB0b3RhbF9ub3Rlcz10b3RhbF9ub3RlcywgbnVtX3Jv'
    'd3M9bnVtX3Jvd3MsCiAgICAgICAgdW5pcT11bmlxLCBub3RlX3RvX2lkeD1u'
    'b3RlX3RvX2lkeCwgaWR4X2J5dGVzPWlkeF9ieXRlcywKICAgICAgICBiaXRt'
    'YXA9Yml0bWFwLCBpZHhfc3RyZWFtPWlkeF9zdHJlYW0sCiAgICAgICAgcm93'
    'X3NlZWtfYnl0ZXM9cm93X3NlZWtfYnl0ZXMsIHByZWZpeD1wcmVmaXgsCiAg'
    'ICApCgojIOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkAoj'
    'IDMtQklUIExJTkVBUiBTQU1QTEUgQ1JVTkNICiMg4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQCgpkZWYgZW5jb2RlX3NhbXBsZXNfcGFj'
    'a2VkKG1vZCwgYml0cz0zKToKICAgICIiIkNvbmNhdGVuYXRlIGFsbCBzYW1w'
    'bGVzLCBlbmNvZGUgZWFjaCB0byBgYml0c2AgYml0cyAocm91bmRlZCkuCiAg'
    'ICBTdXBwb3J0cyAzLWJpdCBhbmQgNC1iaXQgbGluZWFyIHF1YW50aXphdGlv'
    'bi4KICAgICAgMy1iaXQ6IGNvZGUgMC4uNywgbGV2ZWxzIChjb2RlKjMyIC0g'
    'MTEyKSwgc3RlcCAzMi8yNTYgPSAxMi41JQogICAgICA0LWJpdDogY29kZSAw'
    'Li4xNSwgbGV2ZWxzIChjb2RlKjE2IC0gMTIwKSwgc3RlcCAxNi8yNTYgPSA2'
    'LjI1JSAoKzYgZEIgU05SKQogICAgUmV0dXJucyBwYWNrZWQgYnl0ZXMgKyBw'
    'ZXItc2FtcGxlIHN0YXJ0IGluZGljZXMgKGxvZ2ljYWwgc2FtcGxlIHVuaXRz'
    'KS4iIiIKICAgIGlmIGJpdHMgbm90IGluICgzLCA0KToKICAgICAgICByYWlz'
    'ZSBWYWx1ZUVycm9yKGYiYml0cyBtdXN0IGJlIDMgb3IgNCwgZ290IHtiaXRz'
    'fSIpCgogICAgY29uY2F0X3NpZ25lZCA9IFtdCiAgICBzdGFydHMgPSBbXQog'
    'ICAgZm9yIHMgaW4gbW9kLnNhbXBsZV9ieXRlczoKICAgICAgICBzdGFydHMu'
    'YXBwZW5kKGxlbihjb25jYXRfc2lnbmVkKSkKICAgICAgICBmb3IgYiBpbiBz'
    'OgogICAgICAgICAgICBjb25jYXRfc2lnbmVkLmFwcGVuZChiIC0gMjU2IGlm'
    'IGIgPj0gMTI4IGVsc2UgYikKICAgICAgICBjb25jYXRfc2lnbmVkLmV4dGVu'
    'ZChbMF0gKiAxNikKCiAgICB0b3RhbF9zYW1wbGVzID0gbGVuKGNvbmNhdF9z'
    'aWduZWQpCiAgICBjb2RlcyA9IGJ5dGVhcnJheSgpCiAgICBtYXhfY29kZSA9'
    'ICgxIDw8IGJpdHMpIC0gMQogICAgc2hpZnQgPSA4IC0gYml0cwogICAgZm9y'
    'IHN2IGluIGNvbmNhdF9zaWduZWQ6CiAgICAgICAgdW5zaWduZWRfb2Zmc2V0'
    'ID0gc3YgKyAxMjggICMgWzAsIDI1NV0KICAgICAgICBjb2RlID0gdW5zaWdu'
    'ZWRfb2Zmc2V0ID4+IHNoaWZ0CiAgICAgICAgaWYgY29kZSA+IG1heF9jb2Rl'
    'OiBjb2RlID0gbWF4X2NvZGUKICAgICAgICBjb2Rlcy5hcHBlbmQoY29kZSkK'
    'CiAgICB0b3RhbF9iaXRzID0gdG90YWxfc2FtcGxlcyAqIGJpdHMKICAgIHRv'
    'dGFsX2J5dGVzID0gKHRvdGFsX2JpdHMgKyA3KSAvLyA4CiAgICBwYWNrZWQg'
    'PSBieXRlYXJyYXkodG90YWxfYnl0ZXMpCgogICAgaWYgYml0cyA9PSA0Ogog'
    'ICAgICAgICMgTmliYmxlIHBhY2tpbmc6IDIgY29kZXMgcGVyIGJ5dGUsIGxv'
    'dyBuaWJibGUgZmlyc3QKICAgICAgICBmb3IgaSwgYyBpbiBlbnVtZXJhdGUo'
    'Y29kZXMpOgogICAgICAgICAgICBieXRlX3BvcyA9IGkgPj4gMQogICAgICAg'
    'ICAgICBpZiBpICYgMToKICAgICAgICAgICAgICAgIHBhY2tlZFtieXRlX3Bv'
    'c10gfD0gKGMgJiAweEYpIDw8IDQKICAgICAgICAgICAgZWxzZToKICAgICAg'
    'ICAgICAgICAgIHBhY2tlZFtieXRlX3Bvc10gfD0gYyAmIDB4RgogICAgZWxz'
    'ZTogICMgYml0cyA9PSAzCiAgICAgICAgZm9yIGksIGMgaW4gZW51bWVyYXRl'
    'KGNvZGVzKToKICAgICAgICAgICAgYml0X3BvcyAgID0gaSAqIDMKICAgICAg'
    'ICAgICAgYnl0ZV9wb3MgID0gYml0X3BvcyA+PiAzCiAgICAgICAgICAgIGJp'
    'dF9zaGlmdCA9IGJpdF9wb3MgJiA3CiAgICAgICAgICAgIHZhbCA9IChjICYg'
    'NykgPDwgYml0X3NoaWZ0CiAgICAgICAgICAgIHBhY2tlZFtieXRlX3Bvc10g'
    'fD0gdmFsICYgMHhGRgogICAgICAgICAgICBpZiBiaXRfc2hpZnQgPiA1IGFu'
    'ZCBieXRlX3BvcyArIDEgPCB0b3RhbF9ieXRlczoKICAgICAgICAgICAgICAg'
    'IHBhY2tlZFtieXRlX3BvcyArIDFdIHw9ICh2YWwgPj4gOCkgJiAweEZGCgog'
    'ICAgcmV0dXJuIHBhY2tlZCwgc3RhcnRzLCB0b3RhbF9zYW1wbGVzCgojIEJh'
    'Y2t3YXJkLWNvbXBhdCBhbGlhcwpkZWYgZW5jb2RlX3NhbXBsZXNfM2JpdCht'
    'b2QpOgogICAgcmV0dXJuIGVuY29kZV9zYW1wbGVzX3BhY2tlZChtb2QsIGJp'
    'dHM9MykKCgpkZWYgY29tcHV0ZV9yb3dfc3BlZWRfdGFibGUobW9kKToKICAg'
    'ICIiIlNpbXVsYXRlIHRoZSBzb25nIHRvIGZpbmQgcGVyLXJvdyBTUEVFRCAo'
    'aG9ub3VyaW5nIEZ4eC9EeHgvQnh4IGVmZmVjdHMpLgogICAgUmV0dXJucyBy'
    'b3dTcGVlZFtudW1fc29uZ19yb3dzXSBhbmQgcm93U3RhcnRUaWNrW251bV9z'
    'b25nX3Jvd3MrMV0uCiAgICBDb3JyZWN0bHkgaGFuZGxlcyBEeHggKHBhdHRl'
    'cm4gYnJlYWspIGFuZCBCeHggKHBvc2l0aW9uIGp1bXApIHdoaWNoCiAgICBz'
    'aG9ydGVuIGEgcGF0dGVybidzIGVmZmVjdGl2ZSByb3cgY291bnQuIiIiCiAg'
    'ICBzcGVlZCA9IDYgICMgUHJvVHJhY2tlciBkZWZhdWx0CiAgICBicG0gICA9'
    'IDEyNQogICAgcm93U3BlZWQgPSBbXQogICAgYnBtX2NoYW5nZXMgPSBGYWxz'
    'ZQogICAgZm9yIHBvcyBpbiByYW5nZShtb2Quc29uZ19sZW5ndGgpOgogICAg'
    'ICAgIHBhdF9pZHggPSBtb2QucGF0dGVybl9vcmRlcltwb3NdCiAgICAgICAg'
    'cGRhdGEgPSBtb2QucGF0dGVybnNbcGF0X2lkeF0KICAgICAgICBicm9rZSA9'
    'IEZhbHNlCiAgICAgICAgZm9yIHJvdyBpbiByYW5nZSg2NCk6CiAgICAgICAg'
    'ICAgICMgU2NhbiBhbGwgNCBjaGFubmVscyBmb3IgRnh4IC8gRHh4IC8gQnh4'
    'IG9uIHRoaXMgcm93CiAgICAgICAgICAgIGZvciBjaCBpbiByYW5nZShtb2Qu'
    'bnVtX2NoYW5uZWxzKToKICAgICAgICAgICAgICAgIGJhc2UgPSByb3cgKiBt'
    'b2QubnVtX2NoYW5uZWxzICogNCArIGNoICogNAogICAgICAgICAgICAgICAg'
    'YjAsIGIxLCBiMiwgYjMgPSBwZGF0YVtiYXNlOmJhc2UrNF0KICAgICAgICAg'
    'ICAgICAgIGVmZmVjdCA9IGIyICYgMHgwRgogICAgICAgICAgICAgICAgcGFy'
    'YW0gID0gYjMKICAgICAgICAgICAgICAgIGlmIGVmZmVjdCA9PSAweEYgYW5k'
    'IHBhcmFtID4gMDoKICAgICAgICAgICAgICAgICAgICBpZiBwYXJhbSA8IDB4'
    'MjA6CiAgICAgICAgICAgICAgICAgICAgICAgIHNwZWVkID0gcGFyYW0KICAg'
    'ICAgICAgICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgICAgICAg'
    'ICBpZiBicG0gIT0gcGFyYW06CiAgICAgICAgICAgICAgICAgICAgICAgICAg'
    'ICBicG1fY2hhbmdlcyA9IFRydWUKICAgICAgICAgICAgICAgICAgICAgICAg'
    'YnBtID0gcGFyYW0KICAgICAgICAgICAgICAgIGVsaWYgZWZmZWN0ID09IDB4'
    'RCBvciBlZmZlY3QgPT0gMHhCOgogICAgICAgICAgICAgICAgICAgIGJyb2tl'
    'ID0gVHJ1ZSAgICMgcGF0dGVybiBicmVhayAvIHBvc2l0aW9uIGp1bXAKICAg'
    'ICAgICAgICAgcm93U3BlZWQuYXBwZW5kKHNwZWVkKQogICAgICAgICAgICBp'
    'ZiBicm9rZToKICAgICAgICAgICAgICAgIGJyZWFrICAgIyBzdG9wIGFkZGlu'
    'ZyByb3dzIGZvciB0aGlzIHNvbmcgcG9zaXRpb24KICAgIHJvd1N0YXJ0VGlj'
    'ayA9IFswXQogICAgZm9yIHMgaW4gcm93U3BlZWQ6CiAgICAgICAgcm93U3Rh'
    'cnRUaWNrLmFwcGVuZChyb3dTdGFydFRpY2tbLTFdICsgcykKICAgIHJldHVy'
    'biByb3dTcGVlZCwgcm93U3RhcnRUaWNrLCBicG1fY2hhbmdlcwoKCmRlZiBl'
    'bmNvZGVfc2FtcGxlc192cTJkKG1vZCwgSz0yNTYsIHdlaWdodGVkPVRydWUs'
    'IGRvd25zYW1wbGU9MiwgYml0cmF0ZT0nbWVkJywgdmVjX2RpbT0yLCBub19y'
    'dnEyPUZhbHNlKToKICAgICIiIjItc3RhZ2UgUmVzaWR1YWwgVlEgd2l0aCBG'
    'RlQtZ3VpZGVkIHBlci1zYW1wbGUgZGVjaW1hdGlvbi4KICAgIFBlci1zYW1w'
    'bGUgRFMgdmlhIEZGVCBiYW5kd2lkdGggYW5hbHlzaXMg4oCUIERTPTEgZm9y'
    'IGZ1bGwtYmFuZHdpZHRoIHNhbXBsZXMKICAgIChwcmVzZXJ2ZXMgYWxsIEhG'
    'KSwgb25seSBkb3duc2FtcGxlIGlmIGNvbnRlbnQgaXMgZ2VudWluZWx5IGxv'
    'dy1iYW5kd2lkdGguCiAgICBSYXcgc3RyaWRlIGRlY2ltYXRpb24gKG5vIExQ'
    'RikuIGJ3RmFjdG9yIHBlciBzYW1wbGUgPSBhY3R1YWwgRFMgdXNlZC4KICAg'
    'ICIiIgogICAgaW1wb3J0IG51bXB5IGFzIG5wCiAgICBmcm9tIHNrbGVhcm4u'
    'Y2x1c3RlciBpbXBvcnQgTWluaUJhdGNoS01lYW5zCgogICAgIyBCaXRyYXRl'
    'IOKGkiBjb2RlYm9vayBzaXplIChtcDMtc3R5bGUgcXVhbGl0eSBrbm9iKQog'
    'ICAgX2JpdHJhdGVfdGFibGUgPSB7CiAgICAgICAgJ2xvJzogICAgKDEyOCwg'
    'IDY0KSwgICAjIDEzIGJpdHMvcGFpciwgc21hbGxlc3QrZ3JhaW55CiAgICAg'
    'ICAgJ21lZCc6ICAgKDI1NiwgMTI4KSwgICAjIDE1IGJpdHMvcGFpciwgYmFs'
    'YW5jZWQKICAgICAgICAnaGknOiAgICAoNTEyLCAyNTYpLCAgICMgMTcgYml0'
    'cy9wYWlyLCBkZWZhdWx0CiAgICAgICAgJ3VsdHJhJzooMTAyNCwgNTEyKSwg'
    'ICAjIDE5IGJpdHMvcGFpciwgbmVhci10cmFuc3BhcmVudAogICAgfQogICAg'
    'SzEsIEsyID0gX2JpdHJhdGVfdGFibGUuZ2V0KGJpdHJhdGUsIF9iaXRyYXRl'
    'X3RhYmxlWydoaSddKQogICAgaWYgbm9fcnZxMjoKICAgICAgICBLMiA9IDAg'
    'ICMgc2lnbmFsOiBza2lwIHN0YWdlIDIKICAgIEJJVFMxID0gaW50KG5wLmNl'
    'aWwobnAubG9nMihLMSkpKQogICAgQklUUzIgPSBpbnQobnAuY2VpbChucC5s'
    'b2cyKEsyKSkpIGlmIEsyID4gMCBlbHNlIDAKICAgIEJJVFNfVE9UQUwgPSBC'
    'SVRTMSArIEJJVFMyICAjIGlmIEsyPT0wLCBCSVRTMj09MCwgc28ganVzdCBC'
    'SVRTMQoKICAgIGRlZiBoZl9yYXRpbyhyYXdfYnl0ZXMsIGxlbmd0aCwgbnlx'
    'dWlzdF9oej0yMjA1MCk6CiAgICAgICAgIiIiRnJhY3Rpb24gb2YgZW5lcmd5'
    'IGFib3ZlIDhrSHog4oCUIGhpZ2ggPSBwZXJjdXNzaW9uL2N5bWJhbC4iIiIK'
    'ICAgICAgICBpZiBsZW5ndGggPCAzMjogcmV0dXJuIDAuMAogICAgICAgIGRh'
    'dGEgPSBucC5mcm9tYnVmZmVyKHJhd19ieXRlc1s6bGVuZ3RoXSwgZHR5cGU9'
    'bnAuaW50OCkuYXN0eXBlKG5wLmZsb2F0MzIpCiAgICAgICAgZmZ0ICA9IG5w'
    'LmFicyhucC5mZnQucmZmdChkYXRhWzptaW4obGVuZ3RoLCA0MDk2KV0pKQog'
    'ICAgICAgIGUgICAgPSBmbG9hdChucC5zdW0oZmZ0KioyKSkgKyAxZS0xMAog'
    'ICAgICAgIGN1dCAgPSBtYXgoMSwgaW50KGxlbihmZnQpICogODAwMCAvIG55'
    'cXVpc3RfaHopKQogICAgICAgIHJldHVybiBmbG9hdChucC5zdW0oZmZ0W2N1'
    'dDpdKioyKSkgLyBlCgogICAgY29uY2F0X2RzID0gW10KICAgIHN0YXJ0cyAg'
    'ICA9IFtdCiAgICBzYW1wbGVfZHMgPSBbXSAgIyBwZXItc2FtcGxlIGFjdHVh'
    'bCBEUyB1c2VkCiAgICB0b3RhbF9zYW1wbGVzX2Z1bGwgPSAwCgogICAgZm9y'
    'IHMsIHJhd19ieXRlcyBpbiB6aXAobW9kLnNhbXBsZXNfaW5mbywgbW9kLnNh'
    'bXBsZV9ieXRlcyk6CiAgICAgICAgc3RhcnRzLmFwcGVuZChsZW4oY29uY2F0'
    'X2RzKSkKICAgICAgICBpZiBzWydsZW5ndGgnXSA+IDA6CiAgICAgICAgICAg'
    'IHJhdyA9IG5wLmZyb21idWZmZXIocmF3X2J5dGVzLCBkdHlwZT1ucC5pbnQ4'
    'KS5hc3R5cGUobnAuZmxvYXQzMikgLyAxMjguMAogICAgICAgICAgICB0b3Rh'
    'bF9zYW1wbGVzX2Z1bGwgKz0gbGVuKHJhdykKICAgICAgICAgICAgIyBQZXIt'
    'c2FtcGxlIERTIHZpYSBGRlQgYmFuZHdpZHRoIGFuYWx5c2lzIChtaXJyb3Jz'
    'IEhUTUwgcGxheWVyJ3MKICAgICAgICAgICAgIyBid19jb21wcmVzc19zYW1w'
    'bGUpLiAgLS1kb3duc2FtcGxlIGlzIGEgQ0FQLCBub3QgYSBmbG9vcjogZnVs'
    'bC0KICAgICAgICAgICAgIyBiYW5kd2lkdGggc2FtcGxlcyAoZ3VpdGFycywg'
    'dm9jYWxzKSBzdGF5IGF0IERTPTEsIG5hcnJvdy1iYW5kCiAgICAgICAgICAg'
    'ICMgc2FtcGxlcyAobG93IGJhc3MsIG11dGVkIGluc3RydW1lbnRzKSBkcm9w'
    'IHRvIERTPTIvNC84LgogICAgICAgICAgICAjCiAgICAgICAgICAgICMgV2l0'
    'aG91dCB0aGlzIHRoZSBHTFNMIHdhcyBmb3JjZS1kZWNpbWF0aW5nIGV2ZXJ5'
    'IHNhbXBsZSB0bwogICAgICAgICAgICAjIGRvd25zYW1wbGUsIHByb2R1Y2lu'
    'ZyB0aGUgIjgga0h6IGxvLWZpIiBhcnRpZmFjdHMgdGhlIEhUTUwKICAgICAg'
    'ICAgICAgIyBwbGF5ZXIgYXZvaWRlZC4KICAgICAgICAgICAgc3IgPSA0NDEw'
    'MC4wCiAgICAgICAgICAgIG5fZmZ0ID0gbWluKGxlbihyYXcpLCA4MTkyKQog'
    'ICAgICAgICAgICBmZnRfbWFnID0gbnAuYWJzKG5wLmZmdC5yZmZ0KHJhd1s6'
    'bl9mZnRdICogbnAuaGFubmluZyhuX2ZmdCkpKQogICAgICAgICAgICBmcmVx'
    'cyAgID0gbnAuZmZ0LnJmZnRmcmVxKG5fZmZ0LCAxLjAgLyBzcilbOmxlbihm'
    'ZnRfbWFnKV0KICAgICAgICAgICAgcGVhayAgICA9IGZsb2F0KG5wLm1heChm'
    'ZnRfbWFnKSkgKyAxZS0xMgogICAgICAgICAgICBzaWdfYmlucyA9IG5wLndo'
    'ZXJlKGZmdF9tYWcgPiBwZWFrICogMC4wMDUpWzBdCiAgICAgICAgICAgIG1h'
    'eF9mcmVxID0gZmxvYXQoZnJlcXNbc2lnX2JpbnNbLTFdXSkgaWYgbGVuKHNp'
    'Z19iaW5zKSBlbHNlIDIyMDUwLjAKICAgICAgICAgICAgIyBVc2VyJ3MgLS1k'
    'b3duc2FtcGxlIGlzIHRoZSBGTE9PUiAoYWx3YXlzIGF0IGxlYXN0IHRoaXMg'
    'bXVjaCkuCiAgICAgICAgICAgICMgQmFuZHdpZHRoIGFuYWx5c2lzIGNhbiBj'
    'aG9vc2UgdG8gZGVjaW1hdGUgTU9SRSBmb3IgZ2VudWluZWx5CiAgICAgICAg'
    'ICAgICMgbG93LWJhbmR3aWR0aCBjb250ZW50IChlLmcuIHN1Yi1iYXNzIGF0'
    'IERTPTQgZXZlbiB3aGVuIHVzZXIKICAgICAgICAgICAgIyByZXF1ZXN0ZWQg'
    'RFM9MikuICBOZXZlciBsZXNzIHRoYW4gdXNlcidzIHJlcXVlc3QuCiAgICAg'
    'ICAgICAgIGFjdHVhbF9kcyA9IGRvd25zYW1wbGUKICAgICAgICAgICAgaWYg'
    'RmFsc2U6ICAjIGF1dG8tRFMtYnVtcCBwZXJtYW5lbnRseSBkaXNhYmxlZCAo'
    'Zm9ybWVyIG5vLWRzYnVtcCBiZWhhdmlvciwgbm93IGRlZmF1bHQpCiAgICAg'
    'ICAgICAgICAgICAjIFRyeSBoaWdoZXIgZmFjdG9ycyBvbmx5IOKAlCBuZXZl'
    'ciBsZXNzIHRoYW4gdXNlcidzIHJlcXVlc3QuCiAgICAgICAgICAgICAgICBm'
    'b3IgZiBpbiBbZG93bnNhbXBsZSAqIDIsIGRvd25zYW1wbGUgKiA0XToKICAg'
    'ICAgICAgICAgICAgICAgICBpZiBmID4gMTY6IGJyZWFrCiAgICAgICAgICAg'
    'ICAgICAgICAgaWYgc3IgLyBmID49IG1heF9mcmVxICogMi40OgogICAgICAg'
    'ICAgICAgICAgICAgICAgICBhY3R1YWxfZHMgPSBmCiAgICAgICAgICAgICAg'
    'ICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICAgICAgYnJlYWsgICMg'
    'aWYgMnggZG9lc24ndCBzYXRpc2Z5IE55cXVpc3QsIDR4IHdvbid0IGVpdGhl'
    'cgogICAgICAgICAgICAjIFJhdyBzdHJpZGUgZGVjaW1hdGlvbiDigJQgbm8g'
    'TFBGLCBwcmVzZXJ2ZXMgSEYgY29udGVudAogICAgICAgICAgICBpZiBhY3R1'
    'YWxfZHMgPiAxOgogICAgICAgICAgICAgICAgZHMgPSByYXdbOjphY3R1YWxf'
    'ZHNdLmNvcHkoKQogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAg'
    'ZHMgPSByYXcuY29weSgpCiAgICAgICAgICAgIHNhbXBsZV9kcy5hcHBlbmQo'
    'YWN0dWFsX2RzKQogICAgICAgICAgICBjb25jYXRfZHMuZXh0ZW5kKGRzLnRv'
    'bGlzdCgpKQogICAgICAgICAgICAjIExvb3Atc2VhbSBzbW9vdGhpbmc6IGZv'
    'ciBsb29waW5nIHNhbXBsZXMsIHJlcGxhY2UgdGhlIHBvc3QtbG9vcAogICAg'
    'ICAgICAgICAjIGd1YXJkIHJlZ2lvbiB3aXRoIHRoZSBGSVJTVCBmZXcgc2Ft'
    'cGxlcyBmcm9tIGxvb3Bfc3RhcnQuCiAgICAgICAgICAgICMgVGhpcyBtYWtl'
    'cyB2ZWN0b3JzIG5lYXIgbG9vcF9lbmQgaW5jbHVkZSBwcm9wZXIgd3JhcCBj'
    'b250ZXh0IHNvCiAgICAgICAgICAgICMgVlEgcXVhbnRpemF0aW9uIGRvZXNu'
    'J3QgaW50cm9kdWNlIGEgc3RlcCBkaXNjb250aW51aXR5IGF0IHRoZQogICAg'
    'ICAgICAgICAjIGxvb3AgYm91bmRhcnkuICBXaXRob3V0IHRoaXMsIHZlY19k'
    'aW09OCBwcm9kdWNlcyBhbiBhdWRpYmxlIGJ1enoKICAgICAgICAgICAgIyBh'
    'dCB0aGUgbG9vcCByYXRlIChzYW1wbGVbbG9vcEVuZC0xXSBhbmQgc2FtcGxl'
    'W2xvb3BTdGFydF0gZ2V0CiAgICAgICAgICAgICMgcXVhbnRpemVkIHRvIGlu'
    'Y29tcGF0aWJsZSBjb2RlYm9vayBwcm90b3R5cGVzKS4KICAgICAgICAgICAg'
    'IwogICAgICAgICAgICAjIEVuY29kZXIgTU9ERmlsZSBzdG9yZXMgbG9vcF9z'
    'dGFydC9sb29wX2xlbiBpbiBSQVcgYnl0ZSB1bml0cwogICAgICAgICAgICAj'
    'IChhbHJlYWR5IHByZS1tdWx0aXBsaWVkIGJ5IDIgaW4gc2FtcGxlc19pbmZv'
    'KS4KICAgICAgICAgICAgbG9vcF9sZW5fcmF3ID0gaW50KHMuZ2V0KCdsb29w'
    'X2xlbicsIDApIG9yIDApCiAgICAgICAgICAgIGlmIGxvb3BfbGVuX3JhdyA+'
    'IDQ6CiAgICAgICAgICAgICAgICBsb29wX3N0YXJ0X3JhdyA9IGludChzLmdl'
    'dCgnbG9vcF9zdGFydCcsIDApIG9yIDApCiAgICAgICAgICAgICAgICAjIENv'
    'bnZlcnQgdG8gZGVjaW1hdGVkLXN0cmVhbSBpbmRleCAobWF0Y2hlcyBgZHNg'
    'IGFycmF5IGluZGV4aW5nKQogICAgICAgICAgICAgICAgbG9vcF9zdGFydF9k'
    'cyA9IGxvb3Bfc3RhcnRfcmF3IC8vIGFjdHVhbF9kcwogICAgICAgICAgICAg'
    'ICAgIyBDb21wdXRlIHRvdGFsIHBhZGRpbmcgbmVlZGVkOiBhbGlnbi10by12'
    'ZWNfZGltICsgZXh0cmEgZ3VhcmQKICAgICAgICAgICAgICAgIHBhZF9jb3Vu'
    'dCA9ICh2ZWNfZGltIC0gbGVuKGNvbmNhdF9kcykgJSB2ZWNfZGltKSAlIHZl'
    'Y19kaW0gKyA4CiAgICAgICAgICAgICAgICAjIFRha2UgcGFkX2NvdW50IHNh'
    'bXBsZXMgc3RhcnRpbmcgZnJvbSBsb29wX3N0YXJ0IGluIHRoZQogICAgICAg'
    'ICAgICAgICAgIyBkZWNpbWF0ZWQgZGF0YSDigJQgdGhpcyBpcyB3aGF0IHBs'
    'YXliYWNrIHdyYXBzIHRvLgogICAgICAgICAgICAgICAgd3JhcF9kYXRhID0g'
    'W10KICAgICAgICAgICAgICAgIGlmIGxvb3Bfc3RhcnRfZHMgPCBsZW4oZHMp'
    'OgogICAgICAgICAgICAgICAgICAgIHRha2UgPSBtaW4ocGFkX2NvdW50LCBs'
    'ZW4oZHMpIC0gbG9vcF9zdGFydF9kcykKICAgICAgICAgICAgICAgICAgICB3'
    'cmFwX2RhdGEuZXh0ZW5kKGRzLnRvbGlzdCgpW2xvb3Bfc3RhcnRfZHM6bG9v'
    'cF9zdGFydF9kcyt0YWtlXSkKICAgICAgICAgICAgICAgIHdoaWxlIGxlbih3'
    'cmFwX2RhdGEpIDwgcGFkX2NvdW50OgogICAgICAgICAgICAgICAgICAgIHdy'
    'YXBfZGF0YS5hcHBlbmQoMCkKICAgICAgICAgICAgICAgIGNvbmNhdF9kcy5l'
    'eHRlbmQod3JhcF9kYXRhKQogICAgICAgICAgICBlbHNlOgogICAgICAgICAg'
    'ICAgICAgIyBOb24tbG9vcGluZzogcGFkIHdpdGggemVyb3MgKG9yaWdpbmFs'
    'IGJlaGF2aW9yKQogICAgICAgICAgICAgICAgd2hpbGUgbGVuKGNvbmNhdF9k'
    'cykgJSB2ZWNfZGltOiBjb25jYXRfZHMuYXBwZW5kKDApCiAgICAgICAgICAg'
    'ICAgICBjb25jYXRfZHMuZXh0ZW5kKFswXSAqIDgpCiAgICAgICAgZWxzZToK'
    'ICAgICAgICAgICAgc2FtcGxlX2RzLmFwcGVuZChkb3duc2FtcGxlKQogICAg'
    'ICAgICAgICAjIEVtcHR5IHNhbXBsZToganVzdCBwYWQKICAgICAgICAgICAg'
    'd2hpbGUgbGVuKGNvbmNhdF9kcykgJSB2ZWNfZGltOiBjb25jYXRfZHMuYXBw'
    'ZW5kKDApCiAgICAgICAgICAgIGNvbmNhdF9kcy5leHRlbmQoWzBdICogOCkK'
    'CiAgICB3aGlsZSBsZW4oY29uY2F0X2RzKSAlIHZlY19kaW06IGNvbmNhdF9k'
    'cy5hcHBlbmQoMCkKICAgIHRvdGFsX3NhbXBsZXMgPSBsZW4oY29uY2F0X2Rz'
    'KQoKICAgIHZlY3RvcnMgPSBucC5hcnJheShjb25jYXRfZHMsIGR0eXBlPW5w'
    'LmZsb2F0MzIpLnJlc2hhcGUoLTEsIHZlY19kaW0pCgogICAgIyBTdGFnZSAx'
    'IOKAlCByaW5nLXdlaWdodGVkCiAgICB3ZWlnaHRzID0gTm9uZQogICAgaWYg'
    'd2VpZ2h0ZWQ6CiAgICAgICAgc2xvcGVzICA9IG5wLmFicyh2ZWN0b3JzWzos'
    'IC0xXSAtIHZlY3RvcnNbOiwgMF0pCiAgICAgICAgd2VpZ2h0cyA9IChzbG9w'
    'ZXMgKyAxLjApCiAgICAgICAgd2VpZ2h0cyAvPSB3ZWlnaHRzLm1lYW4oKQoK'
    'ICAgIHByaW50KGYiICBSVlEgw5d7ZG93bnNhbXBsZX0gU3RhZ2UgMTogSz17'
    'SzF9IG9uIHtsZW4odmVjdG9ycyl9IHt2ZWNfZGltfS12ZWN0b3JzLi4uIiwg'
    'Zmx1c2g9VHJ1ZSkKICAgIGttMSA9IE1pbmlCYXRjaEtNZWFucyhuX2NsdXN0'
    'ZXJzPUsxLCBuX2luaXQ9NSwgbWF4X2l0ZXI9NjAsIGJhdGNoX3NpemU9ODE5'
    'MiwKICAgICAgICAgICAgICAgICAgICAgICAgICByYW5kb21fc3RhdGU9MCwg'
    'cmVhc3NpZ25tZW50X3JhdGlvPTAuMDEpCiAgICBrbTEuZml0KHZlY3RvcnMs'
    'IHNhbXBsZV93ZWlnaHQ9d2VpZ2h0cykKICAgIGNvZGVzMSAgID0ga20xLnBy'
    'ZWRpY3QodmVjdG9ycykuYXN0eXBlKG5wLmludDMyKQogICAgIyBDZW50cm9p'
    'ZHMgYXJlIGluIFstMSwxXSBmbG9hdCByYW5nZSDigJQgc2NhbGUgYmFjayB0'
    'byBbLTEyOCwxMjddIGludCByYW5nZSBmb3Igc3RvcmFnZQogICAgY2IxICAg'
    'ICAgPSBucC5jbGlwKG5wLnJvdW5kKGttMS5jbHVzdGVyX2NlbnRlcnNfICog'
    'MTI4KSwgLTEyOCwgMTI3KS5hc3R5cGUobnAuaW50MzIpCiAgICByZXNpZHVh'
    'bCA9IHZlY3RvcnMgLSBrbTEuY2x1c3Rlcl9jZW50ZXJzX1tjb2RlczFdCgog'
    'ICAgc25yMSA9IDEwKm5wLmxvZzEwKG5wLm1lYW4odmVjdG9ycyoqMikgLyAo'
    'bnAubWVhbihyZXNpZHVhbCoqMikgKyAxZS05KSkKICAgIHByaW50KGYiICBT'
    'dGFnZSAxIFNOUjoge3NucjE6LjJmfSBkQiIsIGZsdXNoPVRydWUpCgogICAg'
    'IyBTdGFnZSAyIChza2lwcGVkIHdoZW4gbm9fcnZxMiDihpIgSzI9PTApCiAg'
    'ICBpZiBLMiA+IDA6CiAgICAgICAgcHJpbnQoZiIgIFJWUSBTdGFnZSAyOiBL'
    'PXtLMn0gb24gcmVzaWR1YWwuLi4iLCBmbHVzaD1UcnVlKQogICAgICAgIGtt'
    'MiA9IE1pbmlCYXRjaEtNZWFucyhuX2NsdXN0ZXJzPUsyLCBuX2luaXQ9NSwg'
    'bWF4X2l0ZXI9NjAsIGJhdGNoX3NpemU9ODE5MiwKICAgICAgICAgICAgICAg'
    'ICAgICAgICAgICAgICAgcmFuZG9tX3N0YXRlPTEsIHJlYXNzaWdubWVudF9y'
    'YXRpbz0wLjAxKQogICAgICAgIGttMi5maXQocmVzaWR1YWwpCiAgICAgICAg'
    'Y29kZXMyICAgICAgICAgPSBrbTIucHJlZGljdChyZXNpZHVhbCkuYXN0eXBl'
    'KG5wLmludDMyKQogICAgICAgIGNiMiAgICAgICAgICAgID0gbnAuY2xpcChu'
    'cC5yb3VuZChrbTIuY2x1c3Rlcl9jZW50ZXJzXyAqIDEyOCksIC0xMjgsIDEy'
    'NykuYXN0eXBlKG5wLmludDMyKQogICAgICAgIGZpbmFsX3Jlc2lkdWFsID0g'
    'cmVzaWR1YWwgLSBrbTIuY2x1c3Rlcl9jZW50ZXJzX1tjb2RlczJdCiAgICAg'
    'ICAgc25yMiA9IDEwKm5wLmxvZzEwKG5wLm1lYW4odmVjdG9ycyoqMikgLyAo'
    'bnAubWVhbihmaW5hbF9yZXNpZHVhbCoqMikgKyAxZS05KSkKICAgICAgICBw'
    'cmludChmIiAgUlZRIHRvdGFsIFNOUjoge3NucjI6LjJmfSBkQiAoK3tzbnIy'
    'LXNucjE6LjJmfSBkQiBmcm9tIHN0YWdlIDIpIiwgZmx1c2g9VHJ1ZSkKICAg'
    'IGVsc2U6CiAgICAgICAgcHJpbnQoZiIgIOKaoSBTdGFnZSAyIHNraXBwZWQg'
    'KC0tbm8tcnZxMik6IFNOUiA9IHtzbnIxOi4yZn0gZEIiLCBmbHVzaD1UcnVl'
    'KQogICAgICAgIGNvZGVzMiA9IG5wLnplcm9zX2xpa2UoY29kZXMxKSAgIyBw'
    'bGFjZWhvbGRlcgogICAgICAgIGNiMiAgICA9IG5wLnplcm9zKCgwLCB2ZWNf'
    'ZGltKSwgZHR5cGU9bnAuaW50MzIpICAjIGVtcHR5IEsyIGNvZGVib29rCgog'
    'ICAgIyBQYWNrIEJJVFMxK0JJVFMyIGJpdHMgcGVyIHZlY3RvciBMU0ItZmly'
    'c3QKICAgICMgV2hlbiBub19ydnEyIChLMj09MCksIEJJVFMyPT0wIHNvIGNv'
    'bWJpbmVkIGNvbGxhcHNlcyB0byBqdXN0IGNvZGVzMSBiaXRzLgogICAgbl92'
    'ZWNzICAgICAgPSBsZW4odmVjdG9ycykKICAgIHRvdGFsX2JpdHMgID0gbl92'
    'ZWNzICogQklUU19UT1RBTAogICAgdG90YWxfYnl0ZXMgPSAodG90YWxfYml0'
    'cyArIDcpIC8vIDgKICAgIGNvZGVzX2J5dGVzID0gYnl0ZWFycmF5KHRvdGFs'
    'X2J5dGVzKQogICAgbWFzazEgPSAoMSA8PCBCSVRTMSkgLSAxCiAgICBtYXNr'
    'MiA9ICgxIDw8IEJJVFMyKSAtIDEgaWYgQklUUzIgPiAwIGVsc2UgMAogICAg'
    'Zm9yIGkgaW4gcmFuZ2Uobl92ZWNzKToKICAgICAgICBpZiBLMiA+IDA6CiAg'
    'ICAgICAgICAgIGNvbWJpbmVkICA9IChpbnQoY29kZXMxW2ldKSAmIG1hc2sx'
    'KSB8ICgoaW50KGNvZGVzMltpXSkgJiBtYXNrMikgPDwgQklUUzEpCiAgICAg'
    'ICAgZWxzZToKICAgICAgICAgICAgY29tYmluZWQgID0gaW50KGNvZGVzMVtp'
    'XSkgJiBtYXNrMQogICAgICAgIGJpdF9wb3MgICA9IGkgKiBCSVRTX1RPVEFM'
    'CiAgICAgICAgYnl0ZV9wb3MgID0gYml0X3BvcyA+PiAzCiAgICAgICAgYml0'
    'X3NoaWZ0ID0gYml0X3BvcyAmIDcKICAgICAgICB2YWwgPSBjb21iaW5lZCA8'
    'PCBiaXRfc2hpZnQKICAgICAgICBjb2Rlc19ieXRlc1tieXRlX3Bvc10gICAg'
    'IHw9IHZhbCAgICAgICAgJiAweEZGCiAgICAgICAgaWYgYnl0ZV9wb3MrMSA8'
    'IHRvdGFsX2J5dGVzOiBjb2Rlc19ieXRlc1tieXRlX3BvcysxXSB8PSAodmFs'
    'ID4+IDgpICAmIDB4RkYKICAgICAgICBpZiBieXRlX3BvcysyIDwgdG90YWxf'
    'Ynl0ZXM6IGNvZGVzX2J5dGVzW2J5dGVfcG9zKzJdIHw9ICh2YWwgPj4gMTYp'
    'ICYgMHhGRgogICAgICAgIGlmIGJ5dGVfcG9zKzMgPCB0b3RhbF9ieXRlczog'
    'Y29kZXNfYnl0ZXNbYnl0ZV9wb3MrM10gfD0gKHZhbCA+PiAyNCkgJiAweEZG'
    'CgogICAgIyBDb2RlYm9vayBieXRlczogW0sxw5cyIGJ5dGVzXVtLMsOXMiBi'
    'eXRlc10gc3RvcmVkIHVuc2lnbmVkICgrMTI4KQogICAgY2JfYnl0ZXMgPSBi'
    'eXRlYXJyYXkoKQogICAgZm9yIGVudHJ5IGluIGNiMToKICAgICAgICBmb3Ig'
    'diBpbiBlbnRyeTogY2JfYnl0ZXMuYXBwZW5kKChpbnQodikrMjU2KSAmIDB4'
    'RkYpCiAgICBpZiBLMiA+IDA6CiAgICAgICAgZm9yIGVudHJ5IGluIGNiMjoK'
    'ICAgICAgICAgICAgZm9yIHYgaW4gZW50cnk6IGNiX2J5dGVzLmFwcGVuZCgo'
    'aW50KHYpKzI1NikgJiAweEZGKQoKICAgIHJldHVybiBjb2Rlc19ieXRlcywg'
    'Y2JfYnl0ZXMsIHN0YXJ0cywgdG90YWxfc2FtcGxlcywgQklUU19UT1RBTCwg'
    'c2FtcGxlX2RzLCBLMSwgSzIKCgojIOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkAojIEdMU0wgRU1JVFRFUlMKIyDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZAKCmRlZiBieXRlc190b19pbnQzMl9i'
    'ZV9hcnJheShkYXRhLCBjaHVua19pdmVjND01MTIpOgogICAgIiIiUGFjayBi'
    'eXRlcyBpbnRvIGl2ZWM0IGFycmF5cyAoYmlnLWVuZGlhbjogYnl0ZSAwID0g'
    'TVNCIG9mIGludC54KS4iIiIKICAgICMgUGFkIHRvIG11bHRpcGxlIG9mIDE2'
    'IChzaW5jZSBlYWNoIGl2ZWM0IGhvbGRzIDE2IGJ5dGVzKQogICAgcGFkZGVk'
    'ID0gYnl0ZXMoZGF0YSkgKyBiJ1x4MDAnICogKCgxNiAtIGxlbihkYXRhKSAl'
    'IDE2KSAlIDE2KQogICAgaW50cyA9IFtdCiAgICBmb3IgaSBpbiByYW5nZSgw'
    'LCBsZW4ocGFkZGVkKSwgNCk6CiAgICAgICAgdiA9IHN0cnVjdC51bnBhY2so'
    'Jz5JJywgcGFkZGVkW2k6aSs0XSlbMF0KICAgICAgICAjIGNvbnZlcnQgdG8g'
    'c2lnbmVkIGludDMyIGZvciBHTFNMIChoYW5kbGVzIHZhbHVlcyA+PSAyXjMx'
    'KQogICAgICAgIGlmIHYgPj0gKDEgPDwgMzEpOgogICAgICAgICAgICB2IC09'
    'ICgxIDw8IDMyKQogICAgICAgIGludHMuYXBwZW5kKHYpCiAgICAjIFNwbGl0'
    'IGludG8gaXZlYzQgYXJyYXkgY2h1bmtzIG9mIGNodW5rX2l2ZWM0IGl2ZWM0'
    'IGVudHJpZXMgZWFjaAogICAgY2h1bmtzID0gW10KICAgIGN1ciA9IFtdCiAg'
    'ICBmb3IgaSBpbiByYW5nZSgwLCBsZW4oaW50cyksIDQpOgogICAgICAgIGN1'
    'ci5hcHBlbmQodHVwbGUoaW50c1tpOmkrNF0pKQogICAgICAgIGlmIGxlbihj'
    'dXIpID09IGNodW5rX2l2ZWM0OgogICAgICAgICAgICBjaHVua3MuYXBwZW5k'
    'KGN1cikKICAgICAgICAgICAgY3VyID0gW10KICAgIGlmIGN1cjoKICAgICAg'
    'ICBjaHVua3MuYXBwZW5kKGN1cikKICAgIHJldHVybiBjaHVua3MKCmRlZiBl'
    'bWl0X2l2ZWM0X2FycmF5KG5hbWUsIGNodW5rc19vcl9zaW5nbGUsIGl0ZW1z'
    'X3Blcl9saW5lPTIpOgogICAgIiIiRW1pdCBvbmUgb3IgbW9yZSBjb25zdCBp'
    'dmVjNCBhcnJheXMuIGBjaHVua3Nfb3Jfc2luZ2xlYCBpcyBhIGxpc3Qgb2Yg'
    'Y2h1bmtzLiIiIgogICAgb3V0ID0gW10KICAgIGZvciBjaSwgY2h1bmsgaW4g'
    'ZW51bWVyYXRlKGNodW5rc19vcl9zaW5nbGUpOgogICAgICAgIGFycl9uYW1l'
    'ID0gZiJ7bmFtZX17Y2l9IiBpZiBsZW4oY2h1bmtzX29yX3NpbmdsZSkgPiAx'
    'IGVsc2UgZiJ7bmFtZX0wIgogICAgICAgIG91dC5hcHBlbmQoZiJjb25zdCBp'
    'dmVjNCB7YXJyX25hbWV9W3tsZW4oY2h1bmspfV0gPSBpdmVjNFtdKCIpCiAg'
    'ICAgICAgbGluZXMgPSBbXQogICAgICAgIGZvciByb3dfc3RhcnQgaW4gcmFu'
    'Z2UoMCwgbGVuKGNodW5rKSwgaXRlbXNfcGVyX2xpbmUpOgogICAgICAgICAg'
    'ICByb3cgPSBjaHVua1tyb3dfc3RhcnQ6cm93X3N0YXJ0ICsgaXRlbXNfcGVy'
    'X2xpbmVdCiAgICAgICAgICAgIHBhcnRzID0gWyJpdmVjNCh7fSx7fSx7fSx7'
    'fSkiLmZvcm1hdCgqdCkgZm9yIHQgaW4gcm93XQogICAgICAgICAgICBsaW5l'
    'cy5hcHBlbmQoIiAgICAiICsgIiwgIi5qb2luKHBhcnRzKSkKICAgICAgICBv'
    'dXQuYXBwZW5kKCIsXG4iLmpvaW4obGluZXMpKQogICAgICAgIG91dC5hcHBl'
    'bmQoIik7XG4iKQogICAgcmV0dXJuICJcbiIuam9pbihvdXQpCgoKIyDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZAKIyBNQUlOIEJVSUxE'
    'CiMg4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQCgpkZWYg'
    'bWFpbihtb2RfcGF0aCwgb3V0X3BhdGgsIEs9MjU2LCB3ZWlnaHRlZD1UcnVl'
    'LCBkb3duc2FtcGxlPTIsIGJpdHJhdGU9J2hpJywgdmVjX2RpbT0yLCByZXNh'
    'bXBsZXI9J2JzcGxpbmUnLCBub19ydnEyPUZhbHNlKToKICAgICIiIkdlbmVy'
    'YXRlIFNoYWRlclRveSBDb21tb24gR0xTTCBmb3IgYSBNT0QgZmlsZS4KICAg'
    'IGRvd25zYW1wbGU6IGFudGktYWxpYXMgZG93bnNhbXBsZSBmYWN0b3IgZm9y'
    'IHNhbXBsZSBlbmNvZGluZyAoMT1vZmYsIDI9cmVjb21tZW5kZWQpLgogICAg'
    'ICAgICAgICAgICAgSGlnaGVyIGRvd25zYW1wbGUg4oaSIHNtYWxsZXIgZGF0'
    'YSwgbGFyZ2VyIGNvZGVib29rcywgYmV0dGVyIFNOUi4KICAgICAgICAgICAg'
    'ICAgICAgw5cxOiBLMT02NCwgIEsyPTMyICAofjc5IEtCLCAgMjcuNyBkQikK'
    'ICAgICAgICAgICAgICAgICAgw5cyOiBLMT01MTIsIEsyPTI1NiAofjc3IEtC'
    'LCAgMzguNCBkQikgIOKGkCByZWNvbW1lbmRlZAogICAgICAgICAgICAgICAg'
    'ICDDlzQ6IEsxPTUxMiwgSzI9MjU2ICh+MzkgS0IsICAzNy4xIGRCKQogICAg'
    'IiIiCiAgICBtb2QgPSBNT0RGaWxlKG1vZF9wYXRoKQogICAgcHJpbnQoZiLw'
    'n5OmIExvYWRlZDoge21vZC50aXRsZX0iKQogICAgcHJpbnQoZiIgICBQYXR0'
    'ZXJuczoge21vZC5udW1fcGF0dGVybnN9LCBTb25nIGxlbmd0aDoge21vZC5z'
    'b25nX2xlbmd0aH0iKQoKICAgICMgUGF0dGVybiBjcnVuY2gKICAgIHAgPSBl'
    'bmNvZGVfcGF0dGVybnMobW9kKQogICAgcHJpbnQoZiJcbvCfl5zvuI8gIFBB'
    'VFRFUk4gQ1JVTkNIIikKICAgIHByaW50KGYiICAgVG90YWwgbm90ZXM6ICAg'
    'ICAgIHtwWyd0b3RhbF9ub3RlcyddfSIpCiAgICBwcmludChmIiAgIFVuaXF1'
    'ZSBub24tZW1wdHk6ICB7bGVuKHBbJ3VuaXEnXSl9IikKICAgIHByaW50KGYi'
    'ICAgRGljdGlvbmFyeTogICAgICAgIHtsZW4ocFsndW5pcSddKSo0fSBieXRl'
    'cyIpCiAgICBwcmludChmIiAgIEJpdG1hcDogICAgICAgICAgICB7bGVuKHBb'
    'J2JpdG1hcCddKX0gYnl0ZXMiKQogICAgcHJpbnQoZiIgICBJbmRleCBzdHJl'
    'YW06ICAgICAge2xlbihwWydpZHhfc3RyZWFtJ10pfSBieXRlcyIpCiAgICBw'
    'cmludChmIiAgIFJvdyBzZWVrICgxNi1iaXQgcHJlZml4KToge2xlbihwWydy'
    'b3dfc2Vla19ieXRlcyddKX0gYnl0ZXMiKQogICAgcGF0dGVybl90b3RhbCA9'
    'IGxlbihwWyd1bmlxJ10pKjQgKyBsZW4ocFsnYml0bWFwJ10pICsgbGVuKHBb'
    'J2lkeF9zdHJlYW0nXSkgKyBsZW4ocFsncm93X3NlZWtfYnl0ZXMnXSkKICAg'
    'IHByaW50KGYiICAg4oaSIFBhdHRlcm4gdG90YWw6ICAge3BhdHRlcm5fdG90'
    'YWw6LH0gYnl0ZXMiKQoKICAgICMgU3BlZWQvdGljayB0YWJsZSBmcm9tIEZ4'
    'eCBlZmZlY3RzCiAgICByb3dTcGVlZCwgcm93U3RhcnRUaWNrLCBicG1fY2hh'
    'bmdlcyA9IGNvbXB1dGVfcm93X3NwZWVkX3RhYmxlKG1vZCkKICAgIHByaW50'
    'KGYiXG7ij7HvuI8gIFNQRUVEIFRBQkxFIikKICAgIHByaW50KGYiICAgU29u'
    'ZyByb3dzOiB7bGVuKHJvd1NwZWVkKX0sIHRvdGFsIHRpY2tzOiB7cm93U3Rh'
    'cnRUaWNrWy0xXX0iKQogICAgcHJpbnQoZiIgICBVbmlxdWUgc3BlZWRzOiB7'
    'c29ydGVkKHNldChyb3dTcGVlZCkpfSIpCiAgICBzcGVlZF90YWJsZV9ieXRl'
    'cyA9IGxlbihyb3dTdGFydFRpY2spICogMgogICAgcHJpbnQoZiIgICByb3dT'
    'dGFydFRpY2s6IHtzcGVlZF90YWJsZV9ieXRlc30gYnl0ZXMgKDE2LWJpdCBw'
    'YWNrZWQpIikKICAgIGlmIGJwbV9jaGFuZ2VzOgogICAgICAgIHByaW50KGYi'
    'ICAg4pqg77iPICBCUE0gY2hhbmdlcyBkZXRlY3RlZCAoMTJUSC5NT0QgaGFz'
    'IG5vbmUsIGJ1dCBvdGhlciBNT0RzIG1pZ2h0KSIpCgogICAgIyBBdXRvLXNl'
    'bGVjdCBkb3duc2FtcGxlIGlmIG5vdCBleHBsaWNpdGx5IG92ZXJyaWRkZW4g'
    'KGRvd25zYW1wbGU9MiBpcyBkZWZhdWx0KQogICAgIyBCdWRnZXQgZXN0aW1h'
    'dGU6IHRvdGFsX3Jhd19ieXRlcyAvIGRvd25zYW1wbGUgKiAxNy8xNiAoMTct'
    'Yml0IGNvZGVzLCAyIGJ5dGVzL3NhbXBsZSkKICAgICMgU2hhZGVyVG95IHNh'
    'ZmUgem9uZTog4omkIDgwIEtCIHNhbXBsZSBjb2RlcyArIHBhdHRlcm4gZGF0'
    'YQogICAgaW1wb3J0IG51bXB5IGFzIG5wCiAgICB0b3RhbF9yYXcgPSBzdW0o'
    'c1snbGVuZ3RoJ10gZm9yIHMgaW4gbW9kLnNhbXBsZXNfaW5mbykKICAgICMg'
    'Tk9URTogdXNlci1yZXF1ZXN0ZWQgZG93bnNhbXBsZSBpcyByZXNwZWN0ZWQg'
    'YXMgYSBIQVJEIENBUCDigJQgbm8gYXV0by1idW1wLgogICAgIyBVc2UgLS12'
    'ZWMtZGltIDQgb3IgLS1iaXRyYXRlIGxvIGlmIHlvdSBuZWVkIG1vcmUgc2l6'
    'ZSByZWR1Y3Rpb24uCiAgICBlc3RpbWF0ZWRfYnVkZ2V0X2RzMiA9ICh0b3Rh'
    'bF9yYXcgLy8gZG93bnNhbXBsZSkgKiAxNyAvLyAxNiArIDE2MDAwICAjIGZv'
    'ciBsb2cgb25seQoKICAgICMgU2FtcGxlIGVuY29kaW5nOiBSVlEgd2l0aCBi'
    'aXRyYXRlLWNvbnRyb2xsZWQgY29kZWJvb2sgKyBwZXItc2FtcGxlIERTCiAg'
    'ICBkc19sYWJlbCA9IGYiw5d7ZG93bnNhbXBsZX0iIGlmIGRvd25zYW1wbGUg'
    'PiAxIGVsc2UgImZ1bGwtcmVzIgogICAgcHJpbnQoZiJcbvCfl5zvuI8gIFNB'
    'TVBMRSBDUlVOQ0ggKFJWUSB7ZHNfbGFiZWx9IGJpdHJhdGU9e2JpdHJhdGV9'
    'LCByaW5nLXdlaWdodGVkKSIpCiAgICBjb2Rlc19ieXRlcywgY2JfYnl0ZXMs'
    'IHN0YXJ0cywgdG90YWxfc2FtcGxlcywgYml0c19wZXJfY29kZSwgc2FtcGxl'
    'X2RzLCBLMSwgSzIgPSBlbmNvZGVfc2FtcGxlc192cTJkKAogICAgICAgIG1v'
    'ZCwgSywgd2VpZ2h0ZWQsIGRvd25zYW1wbGU9ZG93bnNhbXBsZSwgYml0cmF0'
    'ZT1iaXRyYXRlLCB2ZWNfZGltPXZlY19kaW0sIG5vX3J2cTI9bm9fcnZxMikK'
    'ICAgIEJJVFMxID0gaW50KG5wLmNlaWwobnAubG9nMihLMSkpKQogICAgQklU'
    'UzIgPSBpbnQobnAuY2VpbChucC5sb2cyKEsyKSkpIGlmIEsyID4gMCBlbHNl'
    'IDAKICAgIEJJVFNfVE9UQUwgPSBiaXRzX3Blcl9jb2RlCiAgICBwcmludChm'
    'IiAgIExvZ2ljYWwgc2FtcGxlczogICB7dG90YWxfc2FtcGxlczosfSAgKHtk'
    'c19sYWJlbH0pIikKICAgIHByaW50KGYiICAgQ29kZXMgcGFja2VkOiAgICAg'
    'IHtsZW4oY29kZXNfYnl0ZXMpOix9IGJ5dGVzICAoe2JpdHNfcGVyX2NvZGV9'
    'IGJpdHMvdmVjdG9yIMOXIHt0b3RhbF9zYW1wbGVzLy8yfSB2ZWN0b3JzKSIp'
    'CiAgICBwcmludChmIiAgIENvZGVib29rczogICAgICAgICB7bGVuKGNiX2J5'
    'dGVzKTosfSBieXRlcyAgKHtLMX3DlzIgKyB7SzJ9w5cyIGJ5dGVzKSIpCgog'
    'ICAgdG90YWxfYnVkZ2V0ID0gcGF0dGVybl90b3RhbCArIGxlbihjb2Rlc19i'
    'eXRlcykgKyBsZW4oY2JfYnl0ZXMpICsgMzEqMjQgKyBzcGVlZF90YWJsZV9i'
    'eXRlcwogICAgcHJpbnQoZiJcbvCfk4ogVE9UQUwgY29uc3QgZGF0YSBidWRn'
    'ZXQ6IH57dG90YWxfYnVkZ2V0Oix9IGJ5dGVzICAoe3RvdGFsX2J1ZGdldC8x'
    'MDI0Oi4xZn0gS0IpIikKCiAgICAjIENodW5rIGZvciBHTFNMCiAgICBkaWN0'
    'X2J5dGVzID0gYicnLmpvaW4ocFsndW5pcSddKQogICAgZGljdF9jaHVua3Mg'
    'ICAgPSBieXRlc190b19pbnQzMl9iZV9hcnJheShkaWN0X2J5dGVzKQogICAg'
    'Yml0bWFwX2NodW5rcyAgPSBieXRlc190b19pbnQzMl9iZV9hcnJheShieXRl'
    'cyhwWydiaXRtYXAnXSkpCiAgICBpZHhfY2h1bmtzICAgICA9IGJ5dGVzX3Rv'
    'X2ludDMyX2JlX2FycmF5KGJ5dGVzKHBbJ2lkeF9zdHJlYW0nXSkpCiAgICBy'
    'b3dzZWVrX2NodW5rcyA9IGJ5dGVzX3RvX2ludDMyX2JlX2FycmF5KGJ5dGVz'
    'KHBbJ3Jvd19zZWVrX2J5dGVzJ10pKQogICAgY29kZXNfY2h1bmtzICAgPSBi'
    'eXRlc190b19pbnQzMl9iZV9hcnJheShieXRlcyhjb2Rlc19ieXRlcykpCiAg'
    'ICBjYl9jaHVua3MgICAgICA9IGJ5dGVzX3RvX2ludDMyX2JlX2FycmF5KGJ5'
    'dGVzKGNiX2J5dGVzKSkKCiAgICAjIFBhY2sgcm93U3RhcnRUaWNrIGFzIDE2'
    'LWJpdCBMRSBieXRlcyDihpIgaXZlYzQgY2h1bmtzCiAgICB0aWNrX2J5dGVz'
    'ID0gYnl0ZWFycmF5KCkKICAgIGZvciB0IGluIHJvd1N0YXJ0VGljazoKICAg'
    'ICAgICB0aWNrX2J5dGVzLmFwcGVuZCh0ICYgMHhGRikKICAgICAgICB0aWNr'
    'X2J5dGVzLmFwcGVuZCgodCA+PiA4KSAmIDB4RkYpCiAgICB0aWNrX2NodW5r'
    'cyA9IGJ5dGVzX3RvX2ludDMyX2JlX2FycmF5KGJ5dGVzKHRpY2tfYnl0ZXMp'
    'KQoKICAgIHNhbXBsZXNfaW5mb19uZXcgPSBbXQogICAgZm9yIGksIChzLCBz'
    'dCkgaW4gZW51bWVyYXRlKHppcChtb2Quc2FtcGxlc19pbmZvLCBzdGFydHMp'
    'KToKICAgICAgICAjIFVzZSBwZXItc2FtcGxlIGFjdHVhbCBEUyBmb3IgbGVu'
    'Z3RoL2xvb3Agc2NhbGluZwogICAgICAgIHNkcyA9IHNhbXBsZV9kc1tpXSBp'
    'ZiBpIDwgbGVuKHNhbXBsZV9kcykgZWxzZSBkb3duc2FtcGxlCiAgICAgICAg'
    'c2FtcGxlc19pbmZvX25ldy5hcHBlbmQoZGljdCgKICAgICAgICAgICAgc3Rh'
    'cnQ9c3QsCiAgICAgICAgICAgIGxlbmd0aD1zWydsZW5ndGgnXSAvLyBzZHMs'
    'CiAgICAgICAgICAgIGxvb3BTdGFydD1zWydsb29wX3N0YXJ0J10gLy8gc2Rz'
    'LAogICAgICAgICAgICBsb29wTGVuPXNbJ2xvb3BfbGVuJ10gLy8gc2RzLAog'
    'ICAgICAgICAgICB2b2x1bWU9c1sndm9sdW1lJ10sIGZpbmV0dW5lPXNbJ2Zp'
    'bmV0dW5lJ10sCiAgICAgICAgICAgIGJ3RmFjdG9yPXNkcywgICAjIGFjdHVh'
    'bCBEUyDigJQgdXNlZCBieSBHTFNMIGFzIGZyZXEgZGl2aXNvcgogICAgICAg'
    'ICkpCgogICAgZ2xzbCA9IGJ1aWxkX2dsc2wobW9kLCBwLCBjb2Rlc19ieXRl'
    'cywgc3RhcnRzLCB0b3RhbF9zYW1wbGVzLAogICAgICAgICAgICAgICAgICAg'
    'ICBkaWN0X2NodW5rcywgYml0bWFwX2NodW5rcywgaWR4X2NodW5rcywgcm93'
    'c2Vla19jaHVua3MsCiAgICAgICAgICAgICAgICAgICAgIGNvZGVzX2NodW5r'
    'cywgY2JfY2h1bmtzLCBzYW1wbGVzX2luZm9fbmV3LCBLLCBiaXRzX3Blcl9j'
    'b2RlLAogICAgICAgICAgICAgICAgICAgICB0aWNrX2NodW5rcywgcm93U3Rh'
    'cnRUaWNrLAogICAgICAgICAgICAgICAgICAgICBLMT1LMSwgSzI9SzIsIEJJ'
    'VFMxPUJJVFMxLCBCSVRTMj1CSVRTMiwgQklUU19UT1RBTD1CSVRTX1RPVEFM'
    'LAogICAgICAgICAgICAgICAgICAgICBkb3duc2FtcGxlPWRvd25zYW1wbGUs'
    'IHZlY19kaW09dmVjX2RpbSwgcmVzYW1wbGVyPXJlc2FtcGxlciwgbm9fcnZx'
    'Mj1ub19ydnEyKQogICAgd2l0aCBvcGVuKG91dF9wYXRoLCAndycpIGFzIGY6'
    'CiAgICAgICAgZi53cml0ZShnbHNsKQogICAgcHJpbnQoZiJcbuKchSBXcm90'
    'ZToge291dF9wYXRofSAgKHtsZW4oZ2xzbC5lbmNvZGUoJ3V0Zi04JykpOix9'
    'IGJ5dGVzKSIpCgoKZGVmIGJ1aWxkX2dsc2wobW9kLCBwLCBwYWNrZWQsIHN0'
    'YXJ0cywgdG90YWxfc2FtcGxlcywKICAgICAgICAgICAgICAgZGljdF9jaHVu'
    'a3MsIGJpdG1hcF9jaHVua3MsIGlkeF9jaHVua3MsIHJvd3NlZWtfY2h1bmtz'
    'LAogICAgICAgICAgICAgICBjb2Rlc19jaHVua3MsIGNiX2NodW5rcywgc2Ft'
    'cGxlc19pbmZvX25ldywgSywgYml0c19wZXJfY29kZSwKICAgICAgICAgICAg'
    'ICAgdGlja19jaHVua3MsIHJvd1N0YXJ0VGljaywgSzE9NTEyLCBLMj0yNTYs'
    'IEJJVFMxPTksIEJJVFMyPTgsIEJJVFNfVE9UQUw9MTcsIGRvd25zYW1wbGU9'
    'MiwgdmVjX2RpbT0yLCByZXNhbXBsZXI9J2JzcGxpbmUnLCBub19ydnEyPUZh'
    'bHNlKToKCiAgICAjIOKUgOKUgCBTb25nIG1ldGFkYXRhCiAgICBzb25nX3Bv'
    'c2l0aW9ucyA9IG1vZC5wYXR0ZXJuX29yZGVyWzptb2Quc29uZ19sZW5ndGhd'
    'CgogICAgIyBDb21wdXRlIGFjdHVhbCByb3dzIHBlciBzb25nIHBvc2l0aW9u'
    'IOKAlCBQcm9UcmFja2VyIER4eCAocGF0dGVybiBicmVhaykKICAgICMgYW5k'
    'IEJ4eCAocG9zaXRpb24ganVtcCkgc2hvcnRlbiB0aGUgZWZmZWN0aXZlIHBh'
    'dHRlcm4gbGVuZ3RoLgogICAgZGVmIGFjdHVhbF9wYXR0ZXJuX3Jvd3Moc3Ap'
    'OgogICAgICAgIHBhdCA9IG1vZC5wYXR0ZXJuX29yZGVyW3NwXQogICAgICAg'
    'IE5DX2xvY2FsID0gbW9kLm51bV9jaGFubmVscwogICAgICAgIHBhdF9zaXpl'
    'ID0gNjQgKiBOQ19sb2NhbCAqIDQKICAgICAgICBmb3Igcm93IGluIHJhbmdl'
    'KDY0KToKICAgICAgICAgICAgZm9yIGNoIGluIHJhbmdlKE5DX2xvY2FsKToK'
    'ICAgICAgICAgICAgICAgIGJhc2UgPSAxMDg0ICsgcGF0KnBhdF9zaXplICsg'
    'cm93Kk5DX2xvY2FsKjQgKyBjaCo0CiAgICAgICAgICAgICAgICBuYiA9IG1v'
    'ZC5kYXRhW2Jhc2U6YmFzZSs0XQogICAgICAgICAgICAgICAgZWZmID0gbmJb'
    'Ml0gJiAweEYKICAgICAgICAgICAgICAgIGlmIGVmZiA9PSAweEQgb3IgZWZm'
    'ID09IDB4QjogICAjIHBhdHRlcm4gYnJlYWsgb3IgcG9zaXRpb24ganVtcAog'
    'ICAgICAgICAgICAgICAgICAgIHJldHVybiByb3cgKyAxCiAgICAgICAgcmV0'
    'dXJuIDY0CgogICAgcGF0X3Jvd3MgPSBbYWN0dWFsX3BhdHRlcm5fcm93cyhz'
    'cCkgZm9yIHNwIGluIHJhbmdlKG1vZC5zb25nX2xlbmd0aCldCiAgICBwYXRf'
    'cm93X29mZnNldCA9IFswXQogICAgZm9yIHIgaW4gcGF0X3Jvd3M6CiAgICAg'
    'ICAgcGF0X3Jvd19vZmZzZXQuYXBwZW5kKHBhdF9yb3dfb2Zmc2V0Wy0xXSAr'
    'IHIpCiAgICBwYXRfc3RhcnRfcm93ICA9IFswXSptb2Quc29uZ19sZW5ndGgK'
    'CiAgICAjIHBhdFRpY2tPZmZzZXRbc3BdID0gaW5kZXggaW50byByb3dTdGFy'
    'dFRpY2sgZm9yIHJvdyAwIG9mIHNvbmcgcG9zaXRpb24gc3AKICAgICMgU2Ft'
    'ZSBhcyBwYXRfcm93X29mZnNldCBzaW5jZSB0aWNrIHRhYmxlIHJvd3MgPT0g'
    'c29uZyByb3dzIGFmdGVyIEQwMCBmaXguCiAgICBwYXRfdGlja19vZmZzZXQg'
    'PSBwYXRfcm93X29mZnNldFs6XQoKICAgIHRvdGFsX3Nvbmdfcm93cyA9IG1v'
    'ZC5zb25nX2xlbmd0aCAqIDY0CiAgICBudW1fcGF0dGVybnMgPSBtb2QubnVt'
    'X3BhdHRlcm5zCgogICAgIyDilIDilIAgU2FtcGxlSW5mbyBlbWlzc2lvbiAo'
    'dXNlIG5ldyBgc3RhcnRgID0gc2FtcGxlIGluZGV4IGluIHRoZSBjb25jYXRl'
    'bmF0ZWQgc3RyZWFtKQogICAgZGVmIGZtdF9zYW1wbGVpbmZvKHMpOgogICAg'
    'ICAgIHJldHVybiBmIlNhbXBsZUluZm8oe3NbJ3N0YXJ0J119LCB7c1snbGVu'
    'Z3RoJ119LCB7c1snbG9vcFN0YXJ0J119LCB7c1snbG9vcExlbiddfSwge3Nb'
    'J3ZvbHVtZSddfSwge3MuZ2V0KCdid0ZhY3RvcicsMSl9LCB7cy5nZXQoJ2Zp'
    'bmV0dW5lJywwKX0pIgogICAgc2lfbGluZXMgPSBbXQogICAgZm9yIGksIHMg'
    'aW4gZW51bWVyYXRlKHNhbXBsZXNfaW5mb19uZXcpOgogICAgICAgIHNpX2xp'
    'bmVzLmFwcGVuZChmIiAgICB7Zm10X3NhbXBsZWluZm8ocyl9eycsJyBpZiBp'
    'PDMwIGVsc2UgJyd9IikKICAgIHNhbXBsZXNfaW5mb19nbHNsID0gImNvbnN0'
    'IFNhbXBsZUluZm8gc2FtcGxlc1szMV0gPSBTYW1wbGVJbmZvW10oXG4iICsg'
    'IlxuIi5qb2luKHNpX2xpbmVzKSArICJcbik7IgoKICAgICMg4pSA4pSAIGNo'
    'YW5uZWxQYW4gKHNhbWUgYXMgZXhpc3Rpbmc6IEFtaWdhIExSUkwgd2l0aCBy'
    'ZXN0IGNlbnRlcmVkKQogICAgIyBCdWlsZCBwZXItY2hhbm5lbCBwYW4gZnJv'
    'bSB0aGUgc291cmNlIGZpbGUncyBwYW4gaW5mbyBpZiBhdmFpbGFibGUuCiAg'
    'ICAjIFMzTSBmaWxlcyBzdG9yZSBwZXItY2hhbm5lbCBwYW4gaW4gYGNoYW5u'
    'ZWxfc2V0dGluZ3NgICgzMiBieXRlcyBhdAogICAgIyBoZWFkZXIgb2Zmc2V0'
    'IDB4NDApOiBsb3cgNyBiaXRzID0gcG9zaXRpb24sIHdoZXJlIDAuLjcgPSBs'
    'ZWZ0IFBDTQogICAgIyBjaGFubmVsIGFuZCA4Li4xNSA9IHJpZ2h0IFBDTSBj'
    'aGFubmVsLiBTQVRFTEwuUzNNIGZvciBleGFtcGxlIHVzZXMKICAgICMgYW4g'
    'YWx0ZXJuYXRpbmcgTFJMUiBsYXlvdXQg4oCUIGhhcmRjb2RpbmcgTFJSTCB3'
    'b3VsZCBkdW1wIGJvdGggbGVhZAogICAgIyB2b2ljZXMgKGNoIDAgKyBjaCAz'
    'KSBvbnRvIHRoZSBMRUZUIGJ1cywgcHJvZHVjaW5nIGEgbGVmdC1oZWF2eSBt'
    'aXgKICAgICMgdGhhdCBzb3VuZHMgY2xpcHBlZC9kaXN0b3J0ZWQgb24gdGhl'
    'IGxvdWQgc2lkZSBhbmQgdGhpbiBvbiB0aGUgcmlnaHQuCiAgICAjIEZhbGwg'
    'YmFjayB0byBBbWlnYSBMUlJMIHdoZW4gdGhlIGZpbGUgZG9lc24ndCBzcGVj'
    'aWZ5IHBhbnMgKE1PRCkuCiAgICBfY3MgPSBnZXRhdHRyKG1vZCwgJ2NoYW5u'
    'ZWxfc2V0dGluZ3MnLCBOb25lKSBvciBbXQogICAgY2hhbl9wYW4gPSBbXQog'
    'ICAgZm9yIF9jaCBpbiByYW5nZSgzMik6CiAgICAgICAgaWYgX2NzIGFuZCBf'
    'Y2ggPCBsZW4oX2NzKToKICAgICAgICAgICAgX3BvcyA9IF9jc1tfY2hdICYg'
    'MHg3RgogICAgICAgICAgICBpZiBfcG9zIDwgODoKICAgICAgICAgICAgICAg'
    'IGNoYW5fcGFuLmFwcGVuZCgwLjApCiAgICAgICAgICAgICAgICBjb250aW51'
    'ZQogICAgICAgICAgICBlbGlmIDggPD0gX3BvcyA8IDE2OgogICAgICAgICAg'
    'ICAgICAgY2hhbl9wYW4uYXBwZW5kKDEuMCkKICAgICAgICAgICAgICAgIGNv'
    'bnRpbnVlCiAgICAgICAgIyBGYWxsYmFjazogTFJSTCBBbWlnYSBjb252ZW50'
    'aW9uIGZvciBmaXJzdCA0LCBjZW50ZXIgZm9yIGNoID49IDQKICAgICAgICBj'
    'aGFuX3Bhbi5hcHBlbmQoWzAuMCwgMS4wLCAxLjAsIDAuMF1bX2NoICUgNF0g'
    'aWYgX2NoIDwgNCBlbHNlIDAuNSkKCiAgICAjIOKUgOKUgCBDaHVuayBhcnJh'
    'eSBkZWNsYXJhdGlvbnMKICAgIGRpY3RfbGVuICAgID0gc3VtKGxlbihjKSBm'
    'b3IgYyBpbiBkaWN0X2NodW5rcykKICAgIGJpdG1hcF9sZW4gID0gc3VtKGxl'
    'bihjKSBmb3IgYyBpbiBiaXRtYXBfY2h1bmtzKQogICAgaWR4X2xlbiAgICAg'
    'PSBzdW0obGVuKGMpIGZvciBjIGluIGlkeF9jaHVua3MpCiAgICByb3dzZWVr'
    'X2xlbiA9IHN1bShsZW4oYykgZm9yIGMgaW4gcm93c2Vla19jaHVua3MpCiAg'
    'ICBjb2Rlc19sZW4gICA9IHN1bShsZW4oYykgZm9yIGMgaW4gY29kZXNfY2h1'
    'bmtzKQogICAgY2JfbGVuICAgICAgPSBzdW0obGVuKGMpIGZvciBjIGluIGNi'
    'X2NodW5rcykKCiAgICBoZWFkZXIgPSBmIiIiLyogPT09PT09PT09PT09PT09'
    'PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09'
    'PT09PT09PT09PT09PT09PQogICBHTFNMIChUaGUgTGFzdCkgTU9EIFBsYXll'
    'ciB2MS40MiAoYykgMjAyNiBPcmJsaXZpdXMKICAgNCsgVHJhY2tzIHN1cHBv'
    'cnQsIFMzTS9NT0QgbG9hZGVyLCAzRCBTdXJyb3VuZCwgUGhhdEJhc3MsIENv'
    'bWIgUmV2ZXJiLCBGQVQsIFJWUSBzYW1wbGUgY29tcHJlc3Npb24sIGNvbmZp'
    'Z3VyYWJsZSByZXNhbXBsZXIKICAgQ29udGFjdDogc3ViYmFuZEBnbWFpbC5j'
    'b20gb3IKICAgICAgICAgICAgc3ViYmFuZEBwcm90b25tYWlsLmNvbQogICBH'
    'SVQ6ICAgICBodHRwczovL2dpdGh1Yi5jb20vbWV3emEvbW9kMmdsc2wKICAg'
    'Q09NTU9OIFRBQgogICBHZW5lcmF0ZWQgZnJvbToge21vZC50aXRsZX0KICAg'
    'CiAgIENvbXByZXNzaW9uOgogICAgIOKAoiBQYXR0ZXJuczogYml0bWFwICsg'
    'ZGljdGlvbmFyeSArIDE2LWJpdCBwcmVmaXgtc3VtIHJvdyBzZWVrIChPKDEp'
    'KQogICAgIOKAoiBTYW1wbGVzOiAgMi1zdGFnZSBSVlEgw5d7ZG93bnNhbXBs'
    'ZX0gQUEtZG93bnNhbXBsZWQgKEsxPXtLMX0sIEsyPXtLMn0pLCB7QklUU19U'
    'T1RBTH0gYml0cy9wYWlyCiAgICAgICAgICAgICAgICAgcmluZy13ZWlnaHRl'
    'ZCBrLW1lYW5zIHRyYWluZWQgb24gdGhpcyBNT0QncyBjb250ZW50CiAgID09'
    'PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09'
    'PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0gKi8KCiNkZWZpbmUgVVNF'
    'X0VNQkVEREVEX0RBVEEgMQojZGVmaW5lIE5VTV9QQVRURVJOUyAgICAgIHtu'
    'dW1fcGF0dGVybnN9CiNkZWZpbmUgU09OR19MRU5HVEggICAgICAge21vZC5z'
    'b25nX2xlbmd0aH0KI2RlZmluZSBTT05HX0xPT1BfUE9TICAgICAwCiNkZWZp'
    'bmUgTlVNX0NIQU5ORUxTICAgICAge21vZC5udW1fY2hhbm5lbHN9CiNkZWZp'
    'bmUgQlBNICAgICAgICAgICAgICAgMTI1LjAKI2RlZmluZSBTUEVFRCAgICAg'
    'ICAgICAgICA2LjAKI2RlZmluZSBUT1RBTF9TT05HX1JPV1MgICB7dG90YWxf'
    'c29uZ19yb3dzfQoKLy8g4pSA4pSAIFBhdHRlcm4gY3J1bmNoIGNvbnN0YW50'
    'cyDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIAKI2RlZmluZSBUT1RBTF9OT1RFUyAgICAgICB7cFsndG90YWxfbm90'
    'ZXMnXX0KI2RlZmluZSBUT1RBTF9ST1dTICAgICAgICB7cFsnbnVtX3Jvd3Mn'
    'XX0KI2RlZmluZSBESUNUX05PVEVTICAgICAgICB7bGVuKHBbJ3VuaXEnXSl9'
    'CiNkZWZpbmUgSURYX0JZVEVTX1BFUiAgICAge3BbJ2lkeF9ieXRlcyddfQoj'
    'ZGVmaW5lIERJQ1RfSU5UUyAgICAgICAgIHtkaWN0X2xlbn0KI2RlZmluZSBC'
    'SVRNQVBfSU5UUyAgICAgICB7Yml0bWFwX2xlbn0KI2RlZmluZSBJRFhfSU5U'
    'UyAgICAgICAgICB7aWR4X2xlbn0KI2RlZmluZSBST1dTRUVLX0lOVFMgICAg'
    'ICB7cm93c2Vla19sZW59CgovLyDilIDilIAgUlZRIHNhbXBsZSBjb25zdGFu'
    'dHMg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSACi8vIFNhbXBsZXMgYXJlIGFudGktYWxp'
    'YXMgZG93bnNhbXBsZWQgcGVyLXNhbXBsZSAoRFM9MSBmb3IgSEYgcGVyY3Vz'
    'c2lvbiwKLy8gRFM9e2Rvd25zYW1wbGV9IGZvciBtZWxvZGljKS4gUGVyLXNh'
    'bXBsZSBEUyBpcyBzdG9yZWQgaW4gU2FtcGxlSW5mby5id0ZhY3Rvci4KLy8g'
    'cGVyaW9kVG9GcmVxID0gNzA5Mzc4OS4yLyhwZXJpb2QqMikg4oCUIGJ3RmFj'
    'dG9yIGhhbmRsZXMgcGVyLXNhbXBsZSBwaXRjaC4KI2RlZmluZSBSVlFfQ09E'
    'RVNfQllURVMgICB7bGVuKHBhY2tlZCl9CiNkZWZpbmUgUlZRX0NCX0JZVEVT'
    'ICAgICAge0sxKjIgKyBLMioyfQojZGVmaW5lIFRPVEFMX1NBTVBMRVMgICAg'
    'IHt0b3RhbF9zYW1wbGVzfQoKI2RlZmluZSBCSVRNQVBfQllURVMgICAgICB7'
    'bGVuKHBbJ2JpdG1hcCddKX0KI2RlZmluZSBJRFhfQllURVMgICAgICAgICB7'
    'bGVuKHBbJ2lkeF9zdHJlYW0nXSl9CiNkZWZpbmUgUk9XU0VFS19CWVRFUyAg'
    'ICAge2xlbihwWydyb3dfc2Vla19ieXRlcyddKX0KCi8vIOKUgOKUgCBGeHgt'
    'YXdhcmUgdGltaW5nIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAojZGVm'
    'aW5lIFRPVEFMX1RJQ0tTICAgICAgIHtyb3dTdGFydFRpY2tbLTFdfQojZGVm'
    'aW5lIE5VTV9TT05HX1JPV1MgICAgIHtsZW4ocm93U3RhcnRUaWNrKS0xfQoj'
    'ZGVmaW5lIFRJQ0tTX1BFUl9TRUMgICAgIDUwLjAgICAvLyBCUE09MTI1IGNv'
    'bnN0YW50IGZvciAxMlRILk1PRAoKLy8g4pSA4pSAIEF1ZGlvIGVmZmVjdHMg'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACmNvbnN0'
    'IGJvb2wgIGVuYWJsZTNEICAgICAgPSB0cnVlOwpjb25zdCBib29sICBlbmFi'
    'bGVGQVQgICAgID0gdHJ1ZTsKY29uc3QgaXZlYzIgc3Vycl9jaGFubmVscyA9'
    'IGl2ZWMyKDEsIDQpOwoiIiIKCiAgICAjIOKUgOKUgCBzb25nIG1ldGFkYXRh'
    'IGFycmF5cwogICAgY2hhbl9wYW5fc3RyICAgPSAiLCAiLmpvaW4oZiJ7djou'
    'MWZ9IiBmb3IgdiBpbiBjaGFuX3BhbikKICAgIHNvbmdwb3Nfc3RyICAgID0g'
    'IiwgIi5qb2luKHN0cih4KSBmb3IgeCBpbiBzb25nX3Bvc2l0aW9ucykKICAg'
    'IHJvd29mZl9zdHIgICAgID0gIiwgIi5qb2luKHN0cih4KSBmb3IgeCBpbiBw'
    'YXRfcm93X29mZnNldCkKICAgIHN0YXJ0cm93X3N0ciAgID0gIiwgIi5qb2lu'
    'KHN0cih4KSBmb3IgeCBpbiBwYXRfc3RhcnRfcm93KQogICAgdGlja29mZl9z'
    'dHIgICAgPSAiLCAiLmpvaW4oc3RyKHgpIGZvciB4IGluIHBhdF90aWNrX29m'
    'ZnNldFs6LTFdKSAgIyBsZW5ndGggPSBzb25nX2xlbmd0aAoKICAgIG1ldGEg'
    'PSBmIiIiCmNvbnN0IGZsb2F0IGNoYW5uZWxQYW5bMzJdID0gZmxvYXRbXSh7'
    'Y2hhbl9wYW5fc3RyfSk7CmNvbnN0IGludCAgIHNvbmdQb3NpdGlvbnNbe21v'
    'ZC5zb25nX2xlbmd0aH1dICAgPSBpbnRbXSh7c29uZ3Bvc19zdHJ9KTsKY29u'
    'c3QgaW50ICAgcGF0Um93T2Zmc2V0W3ttb2Quc29uZ19sZW5ndGgrMX1dICAg'
    'ID0gaW50W10oe3Jvd29mZl9zdHJ9KTsKY29uc3QgaW50ICAgcGF0U3RhcnRS'
    'b3dbe21vZC5zb25nX2xlbmd0aH1dICAgICA9IGludFtdKHtzdGFydHJvd19z'
    'dHJ9KTsKY29uc3QgaW50ICAgcGF0VGlja09mZnNldFt7bW9kLnNvbmdfbGVu'
    'Z3RofV0gICA9IGludFtdKHt0aWNrb2ZmX3N0cn0pOwoiIiIKCiAgICAjIOKU'
    'gOKUgCBEYXRhIGFycmF5cyAoaXZlYzQgY2h1bmtzKQogICAgZGF0YV9hcnJh'
    'eXMgPSBbIlxuLy8g4pSA4pSAIFBhdHRlcm4gZGljdGlvbmFyeSAodW5pcXVl'
    'IDQtYnl0ZSBub3RlcywgTVNCLWZpcnN0IHBlciBpbnQpIOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgFxuIl0KICAgIGRhdGFfYXJyYXlzLmFw'
    'cGVuZChlbWl0X2l2ZWM0X2FycmF5KCJwYXREaWN0IiwgZGljdF9jaHVua3Mp'
    'KQogICAgZGF0YV9hcnJheXMuYXBwZW5kKCJcbi8vIOKUgOKUgCBQYXR0ZXJu'
    'IGJpdG1hcCAoMSBiaXQvbm90ZSwgTFNCLWZpcnN0IHdpdGhpbiBieXRlKSDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIBcbiIpCiAgICBkYXRhX2FycmF5cy5hcHBlbmQoZW1p'
    'dF9pdmVjNF9hcnJheSgicGF0Qml0bWFwIiwgYml0bWFwX2NodW5rcykpCiAg'
    'ICBkYXRhX2FycmF5cy5hcHBlbmQoIlxuLy8g4pSA4pSAIEluZGV4IHN0cmVh'
    'bSAoJXMgYnl0ZXMgcGVyIG5vbi1lbXB0eSBub3RlKSDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIBcbiIgJSBwWydpZHhfYnl0ZXMnXSkKICAg'
    'IGRhdGFfYXJyYXlzLmFwcGVuZChlbWl0X2l2ZWM0X2FycmF5KCJwYXRJZHgi'
    'LCBpZHhfY2h1bmtzKSkKICAgIGRhdGFfYXJyYXlzLmFwcGVuZCgiXG4vLyDi'
    'lIDilIAgUm93IHNlZWsgdGFibGUgKDE2LWJpdCBMRSBwcmVmaXggc3Vtcywg'
    'TygxKSBsb29rdXApIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgFxuIikKICAgIGRhdGFfYXJyYXlzLmFw'
    'cGVuZChlbWl0X2l2ZWM0X2FycmF5KCJwYXRSb3dTZWVrIiwgcm93c2Vla19j'
    'aHVua3MpKQogICAgZGF0YV9hcnJheXMuYXBwZW5kKCJcbi8vIOKUgOKUgCBW'
    'USBjb2RlcyAocGFja2VkIGJpdCBzdHJlYW0pIOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgFxuIikKICAgIGRhdGFfYXJyYXlzLmFwcGVuZChl'
    'bWl0X2l2ZWM0X2FycmF5KCJ2cUNvZGVzIiwgY29kZXNfY2h1bmtzKSkKICAg'
    'IGRhdGFfYXJyYXlzLmFwcGVuZChmIlxuLy8g4pSA4pSAIFZRIGNvZGVib29r'
    'ICh7S30gZW50cmllcyDDlyAyIHNhbXBsZXMsIHNpZ25lZCA4LWJpdCBhcyB1'
    'bnNpZ25lZCkg4pSA4pSAXG4iKQogICAgZGF0YV9hcnJheXMuYXBwZW5kKGVt'
    'aXRfaXZlYzRfYXJyYXkoInZxQ29kZWJvb2siLCBjYl9jaHVua3MpKQogICAg'
    'ZGF0YV9hcnJheXMuYXBwZW5kKCJcbi8vIOKUgOKUgCBQZXItcm93IGN1bXVs'
    'YXRpdmUgdGljayB0YWJsZSAoMTYtYml0IExFLCBGeHgtYXdhcmUpIOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgFxu'
    'IikKICAgIGRhdGFfYXJyYXlzLmFwcGVuZChlbWl0X2l2ZWM0X2FycmF5KCJy'
    'b3dTdGFydFRpY2siLCB0aWNrX2NodW5rcykpCgogICAgIyDilIDilIAgU2Ft'
    'cGxlSW5mbyAmIHBlcmlvZFRhYmxlCiAgICB0YWJsZXMgPSBmIiIiCi8vIOKU'
    'gOKUgCBTYW1wbGUgbWV0YWRhdGEgKHN0YXJ0ID0gc2FtcGxlIGluZGV4IGlu'
    'IHBhY2tlZCAzLWJpdCBzdHJlYW0pIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgApzdHJ1Y3QgU2FtcGxlSW5mbyB7ewogICAgaW50IHN0YXJ0LCBsZW5n'
    'dGgsIGxvb3BTdGFydCwgbG9vcExlbiwgdm9sdW1lLCBid0ZhY3RvciwgZmlu'
    'ZXR1bmU7Cn19Owp7c2FtcGxlc19pbmZvX2dsc2x9CgovLyBQcm9UcmFja2Vy'
    'IHBlcmlvZCB0YWJsZSAoQy0xIHRvIEItMykKY29uc3QgaW50IHBlcmlvZFRh'
    'YmxlWzM3XSA9IGludFtdKAogICAgODU2LDgwOCw3NjIsNzIwLDY3OCw2NDAs'
    'NjA0LDU3MCw1MzgsNTA4LDQ4MCw0NTMsCiAgICA0MjgsNDA0LDM4MSwzNjAs'
    'MzM5LDMyMCwzMDIsMjg1LDI2OSwyNTQsMjQwLDIyNiwKICAgIDIxNCwyMDIs'
    'MTkwLDE4MCwxNzAsMTYwLDE1MSwxNDMsMTM1LDEyNywxMjAsMTEzLDAKKTsK'
    'Ci8vIFByb1RyYWNrZXIgMzItZW50cnkgc2luZSB0YWJsZSBmb3IgdmlicmF0'
    'byAoTFVULCBrZXB0IGdsb2JhbCBzbyBpdCBkb2Vzbid0Ci8vIGNvbnN1bWUg'
    'cGVyLWNhbGwgcHJpdmF0ZS9zdGFjayBzdG9yYWdlIGluIGdldENoYW5uZWxP'
    'dXRwdXQpLgpjb25zdCBmbG9hdCB2aWJUYWJbMzJdID0gZmxvYXRbXSgKICAg'
    'ICAgMC4wLCAgMjQuMCwgIDQ5LjAsICA3NC4wLCAgOTcuMCwgMTIwLjAsIDE0'
    'MS4wLCAxNjEuMCwKICAgIDE4MC4wLCAxOTcuMCwgMjEyLjAsIDIyNC4wLCAy'
    'MzUuMCwgMjQ0LjAsIDI1MC4wLCAyNTMuMCwKICAgIDI1NS4wLCAyNTMuMCwg'
    'MjUwLjAsIDI0NC4wLCAyMzUuMCwgMjI0LjAsIDIxMi4wLCAxOTcuMCwKICAg'
    'IDE4MC4wLCAxNjEuMCwgMTQxLjAsIDEyMC4wLCAgOTcuMCwgIDc0LjAsICA0'
    'OS4wLCAgMjQuMAopOwoKLy8gQzQgc3BlZWRzIGZvciBlYWNoIGZpbmV0dW5l'
    'IHZhbHVlIChtaWtJVC9QVCBzcGVjKS4gIEluZGV4IDAuLjcgPSBwb3NpdGl2'
    'ZQovLyBmaW5ldHVuZSAoc2xpZ2h0bHkgaGlnaGVyIHBpdGNoKSwgaW5kZXgg'
    'OC4uMTUgPSBuZWdhdGl2ZSBmaW5ldHVuZSAobG93ZXIpLgovLyBJbiBzYW1w'
    'bGUgZGF0YSB3ZSBzdG9yZSBmaW5ldHVuZSBhcyBhIFNJR05FRCAtOC4uNyBp'
    'bnQg4oCUIGNvbnZlcnQgdmlhICYweEYuCmNvbnN0IGZsb2F0IGM0c3BlZWRz'
    'WzE2XSA9IGZsb2F0W10oCiAgICA4MzYzLjAsIDg0MTMuMCwgODQ2My4wLCA4'
    'NTI5LjAsIDg1ODEuMCwgODY1MS4wLCA4NzIzLjAsIDg3NTcuMCwKICAgIDc4'
    'OTUuMCwgNzk0MS4wLCA3OTg1LjAsIDgwNDYuMCwgODEwNy4wLCA4MTY5LjAs'
    'IDgyMzIuMCwgODI4MC4wCik7CmZsb2F0IHBlcmlvZFRvRnJlcShpbnQgcGVy'
    'aW9kKSB7ewogICAgLy8gRGVmYXVsdCAoZmluZXR1bmU9MCk6IDcwOTM3ODku'
    'MiAvIChwZXJpb2Qgw5cgMikg4omIIDM1NDY4OTQuNi9wZXJpb2QuICBVc2UK'
    'ICAgIC8vIHBlcmlvZFRvRnJlcUZ0IGJlbG93IHdoZW4gZmluZXR1bmUgbWF0'
    'dGVycy4KICAgIHJldHVybiBwZXJpb2QgPiAwID8gNzA5Mzc4OS4yIC8gKGZs'
    'b2F0KHBlcmlvZCkgKiAyLjApIDogMC4wOwp9fQpmbG9hdCBwZXJpb2RUb0Zy'
    'ZXFGdChpbnQgcGVyaW9kLCBpbnQgZmluZXR1bmUpIHt7CiAgICAvLyAoYzQg'
    'KiA0MjgpIC8gcGVyaW9kIOKAlCBtYXRjaGVzIEhUTUwncyBwaXRjaCB0YWJs'
    'ZSBleGFjdGx5LgogICAgaWYgKHBlcmlvZCA8PSAwKSByZXR1cm4gMC4wOwog'
    'ICAgaW50IGlkeCA9IGZpbmV0dW5lICYgMHhGOyAgLy8gLTEgKHNpZ25lZCkg'
    '4oaSIDB4RiwgZXRjLgogICAgcmV0dXJuIChjNHNwZWVkc1tpZHhdICogNDI4'
    'LjApIC8gZmxvYXQocGVyaW9kKTsKfX0KIiIiCgogICAgIyDilIDilIAgRmV0'
    'Y2ggaGVscGVycyAoY2h1bmsgZGlzcGF0Y2hlcnMgZm9yIGVhY2ggYXJyYXkp'
    'CiAgICBkZWYgY2h1bmtfZGlzcGF0Y2gobmFtZSwgbnVtX2NodW5rcywgdmFy'
    'PSdpJyk6CiAgICAgICAgaWYgbnVtX2NodW5rcyA9PSAxOgogICAgICAgICAg'
    'ICByZXR1cm4gZiIgICAgcmV0dXJuIHtuYW1lfTBbe3Zhcn0+PjJdOyIKICAg'
    'ICAgICBsaW5lcyA9IFtmIiAgICBpdmVjNCB2ID0gaXZlYzQoMCk7Il0KICAg'
    'ICAgICBsaW5lcy5hcHBlbmQoZiIgICAgaWYgKGNodW5rSWR4ID09IDApIHYg'
    'PSB7bmFtZX0wW3t2YXJ9Pj4yXTsiKQogICAgICAgIGZvciBrIGluIHJhbmdl'
    'KDEsIG51bV9jaHVua3MpOgogICAgICAgICAgICBsaW5lcy5hcHBlbmQoZiIg'
    'ICAgZWxzZSBpZiAoY2h1bmtJZHggPT0ge2t9KSB2ID0ge25hbWV9e2t9W3t2'
    'YXJ9Pj4yXTsiKQogICAgICAgIGxpbmVzLmFwcGVuZChmIiAgICByZXR1cm4g'
    'djsiKQogICAgICAgIHJldHVybiAiXG4iLmpvaW4obGluZXMpCgogICAgZGVm'
    'IGl2ZWM0X3NlbGVjdCh2YXI9J2knKToKICAgICAgICByZXR1cm4gZiIiIiAg'
    'ICBpbnQgY2kgPSB7dmFyfSAmIDM7CiAgICByZXR1cm4gY2k9PTAgPyB2Lngg'
    'OiBjaT09MSA/IHYueSA6IGNpPT0yID8gdi56IDogdi53OyIiIgoKICAgIGZl'
    'dGNoZXJzID0gZiIiIgovLyDilZDilZDilZAgQ2h1bmtlZCBpdmVjNCBmZXRj'
    'aGVycyDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZAKCi8vIEZldGNoIGEgYnl0ZSBmcm9tIGFueSBjaHVu'
    'a2VkIGJ5dGUgYXJyYXkgKE1TQi1maXJzdCB3aXRoaW4gZWFjaCBpbnQzMiku'
    'Ci8vIEVhY2ggaXZlYzQgaG9sZHMgMTYgYnl0ZXM6IC54ID0gYnl0ZXMgMC0z'
    'LCAueSA9IDQtNywgLnogPSA4LTExLCAudyA9IDEyLTE1Ci8vIFdpdGhpbiBl'
    'YWNoIGludDogYnl0ZSAwID0gTVNCLCBieXRlIDMgPSBMU0IuCgppbnQgX2V4'
    'dHJhY3RCeXRlKGl2ZWM0IHYsIGludCBieXRlSW5JdmVjNCkge3sKICAgIGlu'
    'dCBpbnRJZHggPSBieXRlSW5JdmVjNCA+PiAyOwogICAgaW50IGJ5dGVJbklu'
    'dCA9IGJ5dGVJbkl2ZWM0ICYgMzsKICAgIGludCBwYWNrZWQgPSBpbnRJZHg9'
    'PTAgPyB2LnggOiBpbnRJZHg9PTEgPyB2LnkgOiBpbnRJZHg9PTIgPyB2Lnog'
    'OiB2Lnc7CiAgICBpbnQgc2hpZnQgPSAyNCAtIGJ5dGVJbkludCAqIDg7CiAg'
    'ICByZXR1cm4gKHBhY2tlZCA+PiBzaGlmdCkgJiAweEZGOwp9fQoKLy8g4pSA'
    '4pSAIERpY3Rpb25hcnkgYnl0ZSBmZXRjaCAoYnl0ZUlkeCBpbiBbMCwgRElD'
    'VF9OT1RFUyo0KSkg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSACmludCBmZXRjaERpY3RCeXRlKGludCBi'
    'eXRlSWR4KSB7ewogICAgaW50IGl2ZWM0SWR4ID0gYnl0ZUlkeCA+PiA0Owog'
    'ICAgaW50IGJ5dGVJbkl2ZWM0ID0gYnl0ZUlkeCAmIDE1OwogICAgaXZlYzQg'
    'diA9IHBhdERpY3QwW2l2ZWM0SWR4XTsKICAgIHJldHVybiBfZXh0cmFjdEJ5'
    'dGUodiwgYnl0ZUluSXZlYzQpOwp9fQoKLy8g4pSA4pSAIEJpdG1hcCBieXRl'
    'IGZldGNoIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAppbnQgZmV0Y2hCaXRt'
    'YXBCeXRlKGludCBieXRlSWR4KSB7ewogICAgaW50IGl2ZWM0SWR4ID0gYnl0'
    'ZUlkeCA+PiA0OwogICAgaW50IGJ5dGVJbkl2ZWM0ID0gYnl0ZUlkeCAmIDE1'
    'OwogICAgaXZlYzQgdiA9IHBhdEJpdG1hcDBbaXZlYzRJZHhdOwogICAgcmV0'
    'dXJuIF9leHRyYWN0Qnl0ZSh2LCBieXRlSW5JdmVjNCk7Cn19CgovLyDilIDi'
    'lIAgSW5kZXggc3RyZWFtIGJ5dGUgZmV0Y2ggKGNodW5rZWQgaWYgbmVlZGVk'
    'KSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKaW50IGZldGNo'
    'SWR4Qnl0ZShpbnQgYnl0ZUlkeCkge3sKICAgIGludCBpdmVjNElkeCA9IGJ5'
    'dGVJZHggPj4gNDsKICAgIGludCBieXRlSW5JdmVjNCA9IGJ5dGVJZHggJiAx'
    'NTsKICAgIGludCBjaHVua0lkeCA9IGl2ZWM0SWR4IC8gNTEyOwogICAgaW50'
    'IGxvY2FsSXZlYzQgPSBpdmVjNElkeCAlIDUxMjsKICAgIGl2ZWM0IHYgPSBp'
    'dmVjNCgwKTsKe2NocigxMCkuam9pbihmJyAgICB7ImlmIiBpZiBrPT0wIGVs'
    'c2UgImVsc2UgaWYifSAoY2h1bmtJZHggPT0ge2t9KSB2ID0gcGF0SWR4e2t9'
    'W2xvY2FsSXZlYzRdOycgZm9yIGsgaW4gcmFuZ2UobGVuKGlkeF9jaHVua3Mp'
    'KSl9CiAgICByZXR1cm4gX2V4dHJhY3RCeXRlKHYsIGJ5dGVJbkl2ZWM0KTsK'
    'fX0KCi8vIOKUgOKUgCBSb3ctc2VlayBuaWJibGUgYnl0ZSBmZXRjaCDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKaW50IGZldGNo'
    'Um93U2Vla0J5dGUoaW50IGJ5dGVJZHgpIHt7CiAgICBpbnQgaXZlYzRJZHgg'
    'PSBieXRlSWR4ID4+IDQ7CiAgICBpbnQgYnl0ZUluSXZlYzQgPSBieXRlSWR4'
    'ICYgMTU7CiAgICBpdmVjNCB2ID0gcGF0Um93U2VlazBbaXZlYzRJZHhdOwog'
    'ICAgcmV0dXJuIF9leHRyYWN0Qnl0ZSh2LCBieXRlSW5JdmVjNCk7Cn19Cgov'
    'LyDilIDilIAgVlEgY29kZSBzdHJlYW0gYnl0ZSBmZXRjaCAoY2h1bmtlZCkg'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSACmludCBmZXRjaENvZGVzQnl0ZShpbnQgYnl0ZUlkeCkg'
    'e3sKICAgIGludCBpdmVjNElkeCA9IGJ5dGVJZHggPj4gNDsKICAgIGludCBi'
    'eXRlSW5JdmVjNCA9IGJ5dGVJZHggJiAxNTsKICAgIGludCBjaHVua0lkeCA9'
    'IGl2ZWM0SWR4IC8gNTEyOwogICAgaW50IGxvY2FsSXZlYzQgPSBpdmVjNElk'
    'eCAlIDUxMjsKICAgIGl2ZWM0IHYgPSBpdmVjNCgwKTsKe2NocigxMCkuam9p'
    'bihmJyAgICB7ImlmIiBpZiBrPT0wIGVsc2UgImVsc2UgaWYifSAoY2h1bmtJ'
    'ZHggPT0ge2t9KSB2ID0gdnFDb2Rlc3trfVtsb2NhbEl2ZWM0XTsnIGZvciBr'
    'IGluIHJhbmdlKGxlbihjb2Rlc19jaHVua3MpKSl9CiAgICByZXR1cm4gX2V4'
    'dHJhY3RCeXRlKHYsIGJ5dGVJbkl2ZWM0KTsKfX0KCi8vIOKUgOKUgCBWUSBj'
    'b2RlYm9vayBieXRlIGZldGNoIChzbWFsbCwgZml0cyBpbiAxIGNodW5rIHVz'
    'dWFsbHkpIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgAppbnQgZmV0Y2hDb2RlYm9va0J5dGUoaW50IGJ5dGVJZHgpIHt7CiAg'
    'ICBpbnQgaXZlYzRJZHggPSBieXRlSWR4ID4+IDQ7CiAgICBpbnQgYnl0ZUlu'
    'SXZlYzQgPSBieXRlSWR4ICYgMTU7CiAgICBpbnQgY2h1bmtJZHggPSBpdmVj'
    'NElkeCAvIDUxMjsKICAgIGludCBsb2NhbEl2ZWM0ID0gaXZlYzRJZHggJSA1'
    'MTI7CiAgICBpdmVjNCB2ID0gaXZlYzQoMCk7CntjaHIoMTApLmpvaW4oZicg'
    'ICAgeyJpZiIgaWYgaz09MCBlbHNlICJlbHNlIGlmIn0gKGNodW5rSWR4ID09'
    'IHtrfSkgdiA9IHZxQ29kZWJvb2t7a31bbG9jYWxJdmVjNF07JyBmb3IgayBp'
    'biByYW5nZShsZW4oY2JfY2h1bmtzKSkpfQogICAgcmV0dXJuIF9leHRyYWN0'
    'Qnl0ZSh2LCBieXRlSW5JdmVjNCk7Cn19CiIiIgoKICAgICMg4pSA4pSAIHBv'
    'cGNvdW50IGhlbHBlciAoNC1iaXQgbmliYmxlKQogICAgIyDilIDilIAgZ2V0'
    'Tm90ZTogYml0bWFwICsgZGljdCBsb29rdXAgd2l0aCBPKDEpIHJvdyBzZWVr'
    'ICsgcHJlZml4IHBvcGNvdW50CiAgICBkZWNvZGVycyA9ICIiIgovLyDilZDi'
    'lZDilZAgUGF0dGVybiBkZWNvZGVyOiBiaXRtYXAgKyBkaWN0aW9uYXJ5ICsg'
    'cm93IHNlZWsg4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQCgpzdHJ1Y3QgTm90'
    'ZSB7IGludCBpbnN0cnVtZW50LCBwZXJpb2QsIGVmZmVjdCwgcGFyYW07IH07'
    'CgovLyBQb3Bjb3VudCBmb3IgNC1iaXQgbmliYmxlICgwLi4xNSDihpIgMC4u'
    'NCkg4oCUIGtlcHQgZm9yIGJhY2stY29tcGF0LgppbnQgcG9wY291bnQ0KGlu'
    'dCB4KSB7CiAgICB4ID0gKHggJiAweDUpICsgKCh4ID4+IDEpICYgMHg1KTsK'
    'ICAgIHJldHVybiAoeCAmIDB4MykgKyAoKHggPj4gMikgJiAweDMpOwp9Cgov'
    'LyBQb3Bjb3VudCBmb3IgdXAgdG8gMTYgYml0cyAoaGFuZGxlcyBOVU1fQ0hB'
    'Tk5FTFMgPiA0IGluIFMzTSBldGMuKS4KaW50IHBvcGNvdW50MTYoaW50IHgp'
    'IHsKICAgIHggPSAoeCAmIDB4NTU1NSkgKyAoKHggPj4gMSkgJiAweDU1NTUp'
    'OwogICAgeCA9ICh4ICYgMHgzMzMzKSArICgoeCA+PiAyKSAmIDB4MzMzMyk7'
    'CiAgICB4ID0gKHggJiAweDBGMEYpICsgKCh4ID4+IDQpICYgMHgwRjBGKTsK'
    'ICAgIHJldHVybiAoeCAmIDB4RkYpICsgKCh4ID4+IDgpICYgMHhGRik7Cn0K'
    'Ci8vIFJlY29uc3RydWN0IGN1bXVsYXRpdmUgbm9uLWVtcHR5IGNvdW50IHVw'
    'IHRvIHN0YXJ0IG9mIGByb3dgIOKAlCBPKDEpLgovLyBSb3cgc2VlayB0YWJs'
    'ZSBob2xkcyAxNi1iaXQgTEUgcHJlZml4IHN1bXM6IDIgYnl0ZXMgcGVyIHJv'
    'dy4KaW50IHJvd1NlZWtDdW0oaW50IHRhcmdldFJvdykgewogICAgaW50IGJ5'
    'dGVJZHggPSB0YXJnZXRSb3cgKiAyOwogICAgaW50IGxvID0gZmV0Y2hSb3dT'
    'ZWVrQnl0ZShieXRlSWR4KTsKICAgIGludCBoaSA9IGZldGNoUm93U2Vla0J5'
    'dGUoYnl0ZUlkeCArIDEpOwogICAgcmV0dXJuIGxvIHwgKGhpIDw8IDgpOwp9'
    'CgpOb3RlIGVtcHR5Tm90ZSgpIHsgTm90ZSBuOyBuLmluc3RydW1lbnQ9MDsg'
    'bi5wZXJpb2Q9MDsgbi5lZmZlY3Q9MDsgbi5wYXJhbT0wOyByZXR1cm4gbjsg'
    'fQoKTm90ZSBnZXROb3RlKGludCBzb25nUG9zLCBpbnQgcm93LCBpbnQgY2hh'
    'bm5lbCkgewogICAgaW50IHBhdCA9IHNvbmdQb3NpdGlvbnNbc29uZ1Bvc107'
    'CiAgICBpbnQgcm93R2xvYmFsID0gcGF0ICogNjQgKyByb3c7CiAgICAvLyBG'
    'SVhFRDogd2FzIGhhcmRjb2RlZCAiKiA0IiBhc3N1bWluZyA0LWNoYW5uZWwg'
    'TU9ELiBGb3IgUzNNIHdpdGggdXAgdG8KICAgIC8vIDE2IGNoYW5uZWxzIHBl'
    'ciByb3csIG11c3QgdXNlIE5VTV9DSEFOTkVMUy4gVGhlIGVuY29kZXIgcGFj'
    'a3Mgbm90ZXMgYXMKICAgIC8vIFtwYXQqNjQgKyByb3ddKk5VTV9DSEFOTkVM'
    'UyArIGNoIGludG8gdGhlIGJpdG1hcDsgcHJldmlvdXNseSB0aGUgR0xTTAog'
    'ICAgLy8gcmVhZCAiKiA0IiBzbyBmb3Igc29uZ3Mgd2l0aCA+NCBjaGFubmVs'
    'cyBldmVyeSBub3RlIHBhc3Qgcm93IDAgb2YgcGF0IDAKICAgIC8vIHdhcyBk'
    'ZWNvZGVkIGZyb20gdGhlIHdyb25nIGJpdG1hcCBwb3NpdGlvbiDihpIgZ2Fy'
    'YmFnZSBjZWxscy4KICAgIGludCBub3RlSWR4ICAgPSByb3dHbG9iYWwgKiBO'
    'VU1fQ0hBTk5FTFMgKyBjaGFubmVsOwoKICAgIC8vIDEpIEJpdG1hcCBjaGVj'
    'awogICAgaW50IGJtQnl0ZSA9IGZldGNoQml0bWFwQnl0ZShub3RlSWR4ID4+'
    'IDMpOwogICAgaW50IGJpdCA9IChibUJ5dGUgPj4gKG5vdGVJZHggJiA3KSkg'
    'JiAxOwogICAgaWYgKGJpdCA9PSAwKSByZXR1cm4gZW1wdHlOb3RlKCk7Cgog'
    'ICAgLy8gMikgQ291bnQgbm9uLWVtcHR5IG5vdGVzIGJlZm9yZSB0aGlzIHBv'
    'c2l0aW9uCiAgICAvLyAgICA9IGN1bXVsYXRpdmUgdXAgdG8gcm93R2xvYmFs'
    'ICsgcG9wY291bnQgb2YgYml0bWFwIGJpdHMgd2l0aGluIHRoaXMgcm93CiAg'
    'ICAvLyAgICB1cCB0byAoYnV0IG5vdCBpbmNsdWRpbmcpIHRoZSByZXF1ZXN0'
    'ZWQgY2hhbm5lbC4KICAgIGludCByYW5rID0gcm93U2Vla0N1bShyb3dHbG9i'
    'YWwpOwogICAgaW50IHJvd0JpdG1hcFN0YXJ0ID0gcm93R2xvYmFsICogTlVN'
    'X0NIQU5ORUxTOwogICAgLy8gRm9yIE5VTV9DSEFOTkVMUyB1cCB0byAxNiAr'
    'IHdvcnN0LWNhc2UgYml0IHNoaWZ0LCBzcGFuIHVwIHRvIDI0IGJpdHMgPSAz'
    'IGJ5dGVzLgogICAgaW50IGJ5dGUwSWR4ID0gcm93Qml0bWFwU3RhcnQgPj4g'
    'MzsKICAgIGludCBzaGlmdCAgICA9IHJvd0JpdG1hcFN0YXJ0ICYgNzsKICAg'
    'IGludCBieXRlMCA9IGZldGNoQml0bWFwQnl0ZShieXRlMElkeCk7CiAgICBp'
    'bnQgYnl0ZTEgPSBmZXRjaEJpdG1hcEJ5dGUoYnl0ZTBJZHggKyAxKTsKICAg'
    'IGludCBieXRlMiA9IGZldGNoQml0bWFwQnl0ZShieXRlMElkeCArIDIpOwog'
    'ICAgaW50IGNvbWJpbmVkID0gYnl0ZTAgfCAoYnl0ZTEgPDwgOCkgfCAoYnl0'
    'ZTIgPDwgMTYpOwogICAgaW50IHJvd0JpdHMgPSAoY29tYmluZWQgPj4gc2hp'
    'ZnQpICYgKCgxIDw8IE5VTV9DSEFOTkVMUykgLSAxKTsKICAgIGludCBtYXNr'
    'ID0gKDEgPDwgY2hhbm5lbCkgLSAxOwogICAgcmFuayArPSBwb3Bjb3VudDE2'
    'KHJvd0JpdHMgJiBtYXNrKTsKCiAgICAvLyAzKSBMb29rIHVwIGluZGV4IGFu'
    'ZCBmZXRjaCBub3RlIGZyb20gZGljdGlvbmFyeQogICAgaW50IGRpY3RJZHg7'
    'CiNpZiBJRFhfQllURVNfUEVSID09IDEKICAgIGRpY3RJZHggPSBmZXRjaElk'
    'eEJ5dGUocmFuayk7CiNlbHNlCiAgICBpbnQgbG8gPSBmZXRjaElkeEJ5dGUo'
    'cmFuayAqIDIpOwogICAgaW50IGhpID0gZmV0Y2hJZHhCeXRlKHJhbmsgKiAy'
    'ICsgMSk7CiAgICBkaWN0SWR4ID0gbG8gfCAoaGkgPDwgOCk7CiNlbmRpZgog'
    'ICAgaW50IGIwID0gZmV0Y2hEaWN0Qnl0ZShkaWN0SWR4ICogNCArIDApOwog'
    'ICAgaW50IGIxID0gZmV0Y2hEaWN0Qnl0ZShkaWN0SWR4ICogNCArIDEpOwog'
    'ICAgaW50IGIyID0gZmV0Y2hEaWN0Qnl0ZShkaWN0SWR4ICogNCArIDIpOwog'
    'ICAgaW50IGIzID0gZmV0Y2hEaWN0Qnl0ZShkaWN0SWR4ICogNCArIDMpOwoK'
    'ICAgIE5vdGUgbjsKICAgIG4uaW5zdHJ1bWVudCA9IChiMCAmIDB4RjApIHwg'
    'KChiMiA+PiA0KSAmIDB4MEYpOwogICAgbi5wZXJpb2QgICAgID0gKChiMCAm'
    'IDB4MEYpIDw8IDgpIHwgYjE7CiAgICBuLmVmZmVjdCAgICAgPSBiMiAmIDB4'
    'MEY7CiAgICBuLnBhcmFtICAgICAgPSBiMzsKICAgIHJldHVybiBuOwp9Cgoi'
    'IiIKCiAgICAjIFNhbXBsZSBkZWNvZGVyOiBmLXN0cmluZyBmb3IgI2RlZmlu'
    'ZXMgKG5lZWQgUHl0aG9uIHZhcnMpLCBwbGFpbiBzdHJpbmcgZm9yIGZ1bmN0'
    'aW9uIGJvZGllcwogICAgX3N0YWdlX2xhYmVsID0gIjEtc3RhZ2UgUlZRIChu'
    'byBzdGFnZSAyKSIgaWYgbm9fcnZxMiBlbHNlICIyLXN0YWdlIFJWUSIKICAg'
    'IF9wYWNrZm10ICAgICA9IGYie0JJVFMxfS1iaXQgY29kZTEgb25seSIgaWYg'
    'bm9fcnZxMiBlbHNlIGYiW3tCSVRTMX0tYml0IGNvZGUxXVt7QklUUzJ9LWJp'
    'dCBjb2RlMl0iCiAgICBkZWNvZGVycyArPSAoCiAgICAgICAgZiIvLyDilZDi'
    'lZDilZAgU2FtcGxlIGRlY29kZXI6IHtfc3RhZ2VfbGFiZWx9IMOXe2Rvd25z'
    'YW1wbGV9IEFBLWRvd25zYW1wbGVkIChwZXItc2FtcGxlIERTKSDilZDilZBc'
    'biIKICAgICAgICBmIi8vIHtCSVRTX1RPVEFMfS1iaXQgY29kZXMgcGFja2Vk'
    'IExTQi1maXJzdDoge19wYWNrZm10fVxuIgogICAgICAgIGYiLy8gcGVyaW9k'
    'VG9GcmVxID0gNzA5Mzc4OS4yLyhwZXJpb2QqMikg4oCUIHBlci1zYW1wbGUg'
    'RFMgdmlhIFNhbXBsZUluZm8uYndGYWN0b3JcbiIKICAgICAgICBmIiNkZWZp'
    'bmUgUlZRX0JJVFMgICAgIHtCSVRTX1RPVEFMfVxuIgogICAgICAgIGYiI2Rl'
    'ZmluZSBSVlFfQklUU18xICAge0JJVFMxfVxuIgogICAgICAgIGYiI2RlZmlu'
    'ZSBSVlFfSzEgICAgICAge0sxfVxuIgogICAgICAgIGYiI2RlZmluZSBSVlFf'
    'SzIgICAgICAge0syfVxuIgogICAgICAgIGYiI2RlZmluZSBSVlFfVkVDX0RJ'
    'TSAge3ZlY19kaW19XG4iCiAgICAgICAgZiIjZGVmaW5lIFJWUV9DQjJfQllU'
    'RSAoe0sxfSAqIHt2ZWNfZGltfSlcbiIKICAgICAgICBmIiNkZWZpbmUgUlZR'
    'X01BU0sxICAgIHsoMTw8QklUUzEpLTF9XG4iCiAgICAgICAgZiIjZGVmaW5l'
    'IFJWUV9NQVNLMiAgICB7KDE8PEJJVFMyKS0xIGlmIEJJVFMyPjAgZWxzZSAw'
    'fVxuIgogICAgICAgICsgKGYiI2RlZmluZSBSVlFfTk9fU1RBR0UyIDFcbiIg'
    'aWYgbm9fcnZxMiBlbHNlICIiKQogICAgKQogICAgZGVjb2RlcnMgKz0gIiIi'
    'CnZvaWQgX2dldFJWUUNvZGVzKGludCB2ZWNJZHgsIG91dCBpbnQgY29kZTEs'
    'IG91dCBpbnQgY29kZTIpIHsKICAgIGludCBiaXRQb3MgID0gdmVjSWR4ICog'
    'UlZRX0JJVFM7CiAgICBpbnQgYnl0ZVBvcyA9IGJpdFBvcyA+PiAzOwogICAg'
    'aW50IHNoaWZ0ICAgPSBiaXRQb3MgJiA3OwogICAgaW50IGIwID0gZmV0Y2hD'
    'b2Rlc0J5dGUoYnl0ZVBvcyk7CiAgICBpbnQgYjEgPSBmZXRjaENvZGVzQnl0'
    'ZShieXRlUG9zICsgMSk7CiAgICBpbnQgYjIgPSBmZXRjaENvZGVzQnl0ZShi'
    'eXRlUG9zICsgMik7CiAgICBpbnQgYjMgPSBmZXRjaENvZGVzQnl0ZShieXRl'
    'UG9zICsgMyk7CiAgICBpbnQgY29tYmluZWQgPSBiMCB8IChiMSA8PCA4KSB8'
    'IChiMiA8PCAxNikgfCAoYjMgPDwgMjQpOwogICAgaW50IHJhdyA9IChjb21i'
    'aW5lZCA+PiBzaGlmdCkgJiAoKDEgPDwgUlZRX0JJVFMpIC0gMSk7CiAgICBj'
    'b2RlMSA9IHJhdyAmIFJWUV9NQVNLMTsKI2lmZGVmIFJWUV9OT19TVEFHRTIK'
    'ICAgIGNvZGUyID0gMDsKI2Vsc2UKICAgIGNvZGUyID0gKHJhdyA+PiBSVlFf'
    'QklUU18xKSAmIFJWUV9NQVNLMjsKI2VuZGlmCn0KCmZsb2F0IGdldFNhbXBs'
    'ZShpbnQgc2FtcGxlSWR4KSB7CiAgICBpZiAoc2FtcGxlSWR4IDwgMCB8fCBz'
    'YW1wbGVJZHggPj0gVE9UQUxfU0FNUExFUykgcmV0dXJuIDAuMDsKICAgIGlu'
    'dCB2ZWNJZHggPSBzYW1wbGVJZHggLyBSVlFfVkVDX0RJTTsKICAgIGludCBs'
    'YW5lICAgPSBzYW1wbGVJZHggLSB2ZWNJZHggKiBSVlFfVkVDX0RJTTsKICAg'
    'IC8vIElubGluZSBSVlEgZGVjb2RlIChhdm9pZHMgb3V0LXBhcmFtZXRlciBz'
    'dGFjayBhbGxvY2F0aW9uKQogICAgaW50IF9icCA9IHZlY0lkeCAqIFJWUV9C'
    'SVRTLCBfYnkgPSBfYnAgPj4gMywgX3NoID0gX2JwICYgNzsKICAgIGludCBf'
    'cmF3ID0gKGZldGNoQ29kZXNCeXRlKF9ieSkgfCAoZmV0Y2hDb2Rlc0J5dGUo'
    'X2J5KzEpPDw4KSB8CiAgICAgICAgICAgICAgICAoZmV0Y2hDb2Rlc0J5dGUo'
    'X2J5KzIpPDwxNikgfCAoZmV0Y2hDb2Rlc0J5dGUoX2J5KzMpPDwyNCkpOwog'
    'ICAgX3JhdyA9IChfcmF3ID4+IF9zaCkgJiAoKDEgPDwgUlZRX0JJVFMpIC0g'
    'MSk7CiAgICBpbnQgY29kZTEgPSBfcmF3ICYgUlZRX01BU0sxOwogICAgaW50'
    'IHViMSA9IGZldGNoQ29kZWJvb2tCeXRlKGNvZGUxICogUlZRX1ZFQ19ESU0g'
    'KyBsYW5lKTsKICAgIGludCBzMSAgPSB1YjEgPCAxMjggPyB1YjEgOiB1YjEg'
    'LSAyNTY7CiNpZmRlZiBSVlFfTk9fU1RBR0UyCiAgICByZXR1cm4gZmxvYXQo'
    'czEpIC8gMTI4LjA7CiNlbHNlCiAgICBpbnQgY29kZTIgPSAoX3JhdyA+PiBS'
    'VlFfQklUU18xKSAmIFJWUV9NQVNLMjsKICAgIGludCB1YjIgPSBmZXRjaENv'
    'ZGVib29rQnl0ZShSVlFfQ0IyX0JZVEUgKyBjb2RlMiAqIFJWUV9WRUNfRElN'
    'ICsgbGFuZSk7CiAgICBpbnQgczIgID0gdWIyIDwgMTI4ID8gdWIyIDogdWIy'
    'IC0gMjU2OwogICAgcmV0dXJuIGZsb2F0KHMxICsgczIpIC8gMTI4LjA7CiNl'
    'bmRpZgp9CgovLyDilIDilIAgUG9zaXRpb24gY2FsY3VsYXRpb24gKEZ4eC1h'
    'd2FyZSB2aWEgcm93U3RhcnRUaWNrKSDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKc3RydWN0'
    'IFBvc2l0aW9uIHsgaW50IHNvbmdQb3MsIHBhdHRlcm4sIHJvdzsgZmxvYXQg'
    'dGljaywgcm93VGltZTsgfTsKCi8vIEZldGNoIDE2LWJpdCBMRSB2YWx1ZSBh'
    'dCByb3cgaW5kZXggaW50byByb3dTdGFydFRpY2sKaW50IGZldGNoVGljayhp'
    'bnQgcm93SWR4KSB7CiAgICBpbnQgYnl0ZUlkeCA9IHJvd0lkeCAqIDI7CiAg'
    'ICBpbnQgY2h1bmtJZHggID0gYnl0ZUlkeCA+PiA2OwogICAgaW50IGJ5dGVJ'
    'bjE2ICA9IGJ5dGVJZHggJiA2MzsKICAgIGludCBsbyA9IF9leHRyYWN0Qnl0'
    'ZShyb3dTdGFydFRpY2swWyhjaHVua0lkeDw8MikrKGJ5dGVJbjE2Pj40KV0s'
    'IGJ5dGVJbjE2ICYgMTUpOwogICAgLy8gbmV4dCBieXRlCiAgICBpbnQgYnl0'
    'ZUlkeDIgPSBieXRlSWR4ICsgMTsKICAgIGludCBjaHVua0lkeDIgPSBieXRl'
    'SWR4MiA+PiA2OwogICAgaW50IGJ5dGVJbjE2XzIgPSBieXRlSWR4MiAmIDYz'
    'OwogICAgaW50IGhpID0gX2V4dHJhY3RCeXRlKHJvd1N0YXJ0VGljazBbKGNo'
    'dW5rSWR4Mjw8MikrKGJ5dGVJbjE2XzI+PjQpXSwgYnl0ZUluMTZfMiAmIDE1'
    'KTsKICAgIHJldHVybiBsbyB8IChoaSA8PCA4KTsKfQoKUG9zaXRpb24gZ2V0'
    'UG9zaXRpb24oZmxvYXQgdGltZSkgewogICAgUG9zaXRpb24gcG9zOwogICAg'
    'ZmxvYXQgc29uZ0R1cmF0aW9uID0gZmxvYXQoVE9UQUxfVElDS1MpIC8gVElD'
    'S1NfUEVSX1NFQzsKICAgIGZsb2F0IGxvb3BlZFRpbWUgPSBtb2QodGltZSwg'
    'c29uZ0R1cmF0aW9uKTsKICAgIGZsb2F0IHRvdGFsVGlja0YgPSBsb29wZWRU'
    'aW1lICogVElDS1NfUEVSX1NFQzsKCiAgICAvLyBCaW5hcnkgc2VhcmNoIHJv'
    'd1N0YXJ0VGljayBmb3IgdGhlIGN1cnJlbnQgcm93CiAgICBpbnQgbG8gPSAw'
    'LCBoaSA9IE5VTV9TT05HX1JPV1M7CiAgICBmb3IgKGludCBfYnMgPSAwOyBf'
    'YnMgPCAxMjsgX2JzKyspIHsgIC8vIGxvZzIoMTkyMCspIOKJiCAxMQogICAg'
    'ICAgIGlmIChsbyA+PSBoaSAtIDEpIGJyZWFrOwogICAgICAgIGludCBtaWQg'
    'PSAobG8gKyBoaSkgPj4gMTsKICAgICAgICBpZiAoZmxvYXQoZmV0Y2hUaWNr'
    'KG1pZCkpIDw9IHRvdGFsVGlja0YpIGxvID0gbWlkOwogICAgICAgIGVsc2Ug'
    'aGkgPSBtaWQ7CiAgICB9CiAgICBpbnQgZ2xvYmFsUm93ID0gbG87CiAgICBp'
    'ZiAoZ2xvYmFsUm93ID49IE5VTV9TT05HX1JPV1MpIGdsb2JhbFJvdyA9IE5V'
    'TV9TT05HX1JPV1MgLSAxOwoKICAgIC8vIEZpbmQgc29uZ1BvcyB2aWEgbGlu'
    'ZWFyIHNlYXJjaCBvdmVyIHBhdFRpY2tPZmZzZXQgKFNPTkdfTEVOR1RIIOKJ'
    'pCAxMjgsIGZhc3QgZW5vdWdoKQogICAgaW50IHNwID0gU09OR19MRU5HVEgg'
    'LSAxOwogICAgZm9yIChpbnQgX2kgPSAwOyBfaSA8IFNPTkdfTEVOR1RIIC0g'
    'MTsgX2krKykgewogICAgICAgIGlmIChwYXRUaWNrT2Zmc2V0W19pICsgMV0g'
    'PiBnbG9iYWxSb3cpIHsgc3AgPSBfaTsgYnJlYWs7IH0KICAgIH0KICAgIHBv'
    'cy5zb25nUG9zID0gc3A7CiAgICBwb3MucGF0dGVybiA9IHNvbmdQb3NpdGlv'
    'bnNbc3BdOwogICAgcG9zLnJvdyAgICAgPSBnbG9iYWxSb3cgLSBwYXRUaWNr'
    'T2Zmc2V0W3NwXTsKCiAgICBpbnQgcm93VGljayAgICA9IGZldGNoVGljayhn'
    'bG9iYWxSb3cpOwogICAgaW50IG5leHRUaWNrICAgPSBmZXRjaFRpY2soZ2xv'
    'YmFsUm93ICsgMSk7CiAgICBpbnQgcm93U3BlZWQgICA9IG5leHRUaWNrIC0g'
    'cm93VGljazsKICAgIHBvcy50aWNrICAgICAgID0gdG90YWxUaWNrRiAtIGZs'
    'b2F0KHJvd1RpY2spOwogICAgcG9zLnJvd1RpbWUgICAgPSBmbG9hdChyb3dT'
    'cGVlZCkgLyBUSUNLU19QRVJfU0VDOwogICAgcmV0dXJuIHBvczsKfQoKLy8g'
    'NC1wb2ludCBjdWJpYyBCLXNwbGluZSBpbnRlcnBvbGF0aW9uLgovLyBCLXNw'
    'bGluZSBpcyBBUFBST1hJTUFUSU5HIChzbW9vdGhzIHRocm91Z2ggc2FtcGxl'
    'IHBvaW50cykgcmF0aGVyIHRoYW4KLy8gSU5URVJQT0xBVElORyAocGFzc2lu'
    'ZyBleGFjdGx5IHRocm91Z2ggdGhlbSksIGdpdmluZyBpbmhlcmVudCBsb3ct'
    'cGFzcwovLyBjaGFyYWN0ZXIgdGhhdCByZWR1Y2VzIGhpZ2gtZnJlcXVlbmN5'
    'IHF1YW50aXphdGlvbiBub2lzZS4KIiIiICsgKAogICAgICAgICAgICAjIOKU'
    'gOKUgCBMaW5lYXI6IDIgdGFwcywgUHJvVHJhY2tlci1hdXRoZW50aWMsIGNo'
    'ZWFwZXN0IOKUgOKUgAogICAgICAgICAgICAnJydmbG9hdCBnZXRTYW1wbGVG'
    'KGludCBiYXNlLCBmbG9hdCBmcG9zLCBpbnQgc21wTGVuLCBpbnQgbG9vcFN0'
    'YXJ0LCBpbnQgbG9vcExlbikgewogICAgaW50IGkgPSBpbnQoZnBvcyk7CiAg'
    'ICBmbG9hdCB0ID0gZnBvcyAtIGZsb2F0KGkpOwogICAgZmxvYXQgcDEgPSBn'
    'ZXRTYW1wbGUoYmFzZSArIGkpOwogICAgZmxvYXQgcDIgPSBnZXRTYW1wbGUo'
    'YmFzZSArIG1pbihpICsgMSwgc21wTGVuICsgMTUpKTsKICAgIHJldHVybiBt'
    'aXgocDEsIHAyLCB0KTsKfScnJyBpZiByZXNhbXBsZXIgPT0gJ2xpbmVhcicg'
    'ZWxzZQogICAgICAgICAgICAjIOKUgOKUgCBMYW5jem9zLTM6IDYgdGFwcywg'
    'c2hhcnBlc3QsIGJyaWdodGVzdCDilIDilIAKICAgICAgICAgICAgJycnLy8g'
    'TGFuY3pvcy0zIHdpbmRvd2VkIHNpbmM6IHcoeCkgPSBzaW5jKM+AeCkgKiBz'
    'aW5jKM+AeC8zKSBmb3IgfHh8PDMKZmxvYXQgX2xhbmN6b3MzKGZsb2F0IHgp'
    'IHsKICAgIGlmICh4IDwgMWUtNikgcmV0dXJuIDEuMDsKICAgIGZsb2F0IHBp'
    'eCA9IDMuMTQxNTkyNjUgKiB4OwogICAgZmxvYXQgcGl4MyA9IHBpeCAvIDMu'
    'MDsKICAgIHJldHVybiAoc2luKHBpeCkgKiBzaW4ocGl4MykpIC8gKHBpeCAq'
    'IHBpeDMpOwp9CmZsb2F0IGdldFNhbXBsZUYoaW50IGJhc2UsIGZsb2F0IGZw'
    'b3MsIGludCBzbXBMZW4sIGludCBsb29wU3RhcnQsIGludCBsb29wTGVuKSB7'
    'CiAgICBpbnQgaSAgPSBpbnQoZnBvcyk7CiAgICBmbG9hdCB0ID0gZnBvcyAt'
    'IGZsb2F0KGkpOwogICAgaW50IGltMiA9IGkgLSAyLCBpbTEgPSBpIC0gMSwg'
    'aXAxID0gaSArIDEsIGlwMiA9IGkgKyAyLCBpcDMgPSBpICsgMzsKICAgIGlu'
    'dCBsb29wRW5kID0gbG9vcFN0YXJ0ICsgbG9vcExlbjsgIC8vIG9uZS1wYXN0'
    'IGxhc3QgbG9vcCBzYW1wbGUKICAgIC8vIExvb3Agd3JhcGFyb3VuZCBmb3Ig'
    'QUxMIGtlcm5lbCB0YXBzLiBUaGUgTGFuY3pvcyBrZXJuZWwgcmVhY2hlcyAy'
    'CiAgICAvLyBzYW1wbGVzIGJhY2sgYW5kIDMgZm9yd2FyZDsgd2hlbmV2ZXIg'
    'YW55IG9mIHRob3NlIGZhbGxzIG91dHNpZGUKICAgIC8vIFtsb29wU3RhcnQs'
    'IGxvb3BFbmQpIHdoaWxlIHdlJ3JlIHBsYXlpbmcgSU4gdGhlIGxvb3AsIGl0'
    'IG11c3Qgd3JhcC4KICAgIC8vCiAgICAvLyBUaGUgcHJldmlvdXMgdmVyc2lv'
    'biB3cmFwcGVkIG9ubHkgdGhlIEJBQ0tXQVJEIHRhcHMgKGltMiwgaW0xKSBh'
    'bmQKICAgIC8vIENMQU1QRUQgdGhlIGZvcndhcmQgdGFwcyB0byBzbXBMZW4r'
    'MTUuIEF0IGhpZ2ggcGl0Y2ggdGhlIHNvdXJjZS1zdGVwCiAgICAvLyBwZXIg'
    'b3V0cHV0IHNhbXBsZSBpcyBsYXJnZSAoZS5nLiA1LjfDlyksIHNvIGV2ZXJ5'
    'IGxvb3AgaXRlcmF0aW9uIGNyb3NzZXMKICAgIC8vIHRoZSBsb29wIGJvdW5k'
    'YXJ5OyBmb3J3YXJkLWNsYW1wIG1lYW50IGlwMS9pcDIvaXAzIHJlYWQgcGFz'
    'dCBzbXBMZW4KICAgIC8vIGludG8gYWRqYWNlbnQgZ2FyYmFnZSBldmVyeSBs'
    'b29wIGN5Y2xlLCBjcmVhdGluZyBidXp6IHRoYXQgc2NhbGVkIHdpdGgKICAg'
    'IC8vIHBpdGNoIOKAlCBleGFjdGx5IHRoZSBmYWlsdXJlIG1vZGUgd2Ugb2Jz'
    'ZXJ2ZWQ6IGNsZWFuIG5vdGUgYm9kaWVzLCBidXQKICAgIC8vIDMtOSUgSEYg'
    'ZW5lcmd5IG9uIGV2ZXJ5IGxvb3AtdGFpbCByZXBlYXQuCiAgICAvLwogICAg'
    'Ly8gUHJlLWxvb3AgKGkgPCBsb29wU3RhcnQpOiBiYWNrd2FyZCB0YXBzIGNs'
    'YW1wIHRvIDAgKHNpbGVudCBwcmVmaXgpLAogICAgLy8gZm9yd2FyZCB0YXBz'
    'IGNsYW1wIHRvIHNtcExlbisxNSAocG9zdC1zYW1wbGUgcGFkZGluZyB6ZXJv'
    'cykuIFN0YW5kYXJkCiAgICAvLyBhdHRhY2stcmVnaW9uIGJlaGF2aW9yLgog'
    'ICAgaWYgKGxvb3BMZW4gPiAyICYmIGkgPj0gbG9vcFN0YXJ0KSB7CiAgICAg'
    'ICAgaWYgKGltMiA8IGxvb3BTdGFydCkgaW0yID0gbG9vcEVuZCArIChpbTIg'
    'LSBsb29wU3RhcnQpOwogICAgICAgIGlmIChpbTEgPCBsb29wU3RhcnQpIGlt'
    'MSA9IGxvb3BFbmQgKyAoaW0xIC0gbG9vcFN0YXJ0KTsKICAgICAgICBpZiAo'
    'aXAxID49IGxvb3BFbmQpICBpcDEgPSBsb29wU3RhcnQgKyAoaXAxIC0gbG9v'
    'cEVuZCk7CiAgICAgICAgaWYgKGlwMiA+PSBsb29wRW5kKSAgaXAyID0gbG9v'
    'cFN0YXJ0ICsgKGlwMiAtIGxvb3BFbmQpOwogICAgICAgIGlmIChpcDMgPj0g'
    'bG9vcEVuZCkgIGlwMyA9IGxvb3BTdGFydCArIChpcDMgLSBsb29wRW5kKTsK'
    'ICAgIH0gZWxzZSB7CiAgICAgICAgaXAxID0gbWluKGlwMSwgc21wTGVuICsg'
    'MTUpOwogICAgICAgIGlwMiA9IG1pbihpcDIsIHNtcExlbiArIDE1KTsKICAg'
    'ICAgICBpcDMgPSBtaW4oaXAzLCBzbXBMZW4gKyAxNSk7CiAgICB9CiAgICBp'
    'bTIgPSBtYXgoMCwgaW0yKTsgaW0xID0gbWF4KDAsIGltMSk7CiAgICBmbG9h'
    'dCB3MCA9IF9sYW5jem9zMyhhYnModCArIDIuMCkpOwogICAgZmxvYXQgdzEg'
    'PSBfbGFuY3pvczMoYWJzKHQgKyAxLjApKTsKICAgIGZsb2F0IHcyID0gX2xh'
    'bmN6b3MzKGFicyh0ICAgICAgKSk7CiAgICBmbG9hdCB3MyA9IF9sYW5jem9z'
    'MyhhYnModCAtIDEuMCkpOwogICAgZmxvYXQgdzQgPSBfbGFuY3pvczMoYWJz'
    'KHQgLSAyLjApKTsKICAgIGZsb2F0IHc1ID0gX2xhbmN6b3MzKGFicyh0IC0g'
    'My4wKSk7CiAgICBmbG9hdCB3c3VtID0gdzArdzErdzIrdzMrdzQrdzU7CiAg'
    'ICByZXR1cm4gKHcwKmdldFNhbXBsZShiYXNlK2ltMikgKyB3MSpnZXRTYW1w'
    'bGUoYmFzZStpbTEpICsKICAgICAgICAgICAgdzIqZ2V0U2FtcGxlKGJhc2Ur'
    'aSAgKSArIHczKmdldFNhbXBsZShiYXNlK2lwMSkgKwogICAgICAgICAgICB3'
    'NCpnZXRTYW1wbGUoYmFzZStpcDIpICsgdzUqZ2V0U2FtcGxlKGJhc2UraXAz'
    'KSkgLyB3c3VtOwp9JycnIGlmIHJlc2FtcGxlciA9PSAnbGFuY3pvczMnIGVs'
    'c2UKICAgICAgICAgICAgIyDilIDilIAgQi1zcGxpbmUgKGRlZmF1bHQpOiA0'
    'IHRhcHMsIHNtb290aCwgZ2VudGxlIExQRiDilIDilIAKICAgICAgICAgICAg'
    'JycnZmxvYXQgZ2V0U2FtcGxlRihpbnQgYmFzZSwgZmxvYXQgZnBvcywgaW50'
    'IHNtcExlbiwgaW50IGxvb3BTdGFydCwgaW50IGxvb3BMZW4pIHsKICAgIGlu'
    'dCBpICA9IGludChmcG9zKTsKICAgIGZsb2F0IHQgPSBmcG9zIC0gZmxvYXQo'
    'aSk7CiAgICBpbnQgaTAgPSBpIC0gMTsKICAgIGlmIChsb29wTGVuID4gMiAm'
    'JiBpMCA8IGxvb3BTdGFydCkgaTAgPSBsb29wU3RhcnQgKyBsb29wTGVuIC0g'
    'MTsKICAgIGVsc2UgaTAgPSBtYXgoMCwgaTApOwogICAgZmxvYXQgcDAgPSBn'
    'ZXRTYW1wbGUoYmFzZSArIGkwKTsKICAgIGZsb2F0IHAxID0gZ2V0U2FtcGxl'
    'KGJhc2UgKyBpKTsKICAgIGZsb2F0IHAyID0gZ2V0U2FtcGxlKGJhc2UgKyBt'
    'aW4oaSArIDEsIHNtcExlbiArIDE1KSk7CiAgICBmbG9hdCBwMyA9IGdldFNh'
    'bXBsZShiYXNlICsgbWluKGkgKyAyLCBzbXBMZW4gKyAxNSkpOwogICAgZmxv'
    'YXQgdDIgPSB0ICogdDsKICAgIGZsb2F0IHQzID0gdDIgKiB0OwogICAgZmxv'
    'YXQgdzAgPSAoMS4wIC0gdCkgKiAoMS4wIC0gdCkgKiAoMS4wIC0gdCkgLyA2'
    'LjA7CiAgICBmbG9hdCB3MSA9ICgzLjAgKiB0MyAtIDYuMCAqIHQyICsgNC4w'
    'KSAvIDYuMDsKICAgIGZsb2F0IHcyID0gKC0zLjAgKiB0MyArIDMuMCAqIHQy'
    'ICsgMy4wICogdCArIDEuMCkgLyA2LjA7CiAgICBmbG9hdCB3MyA9IHQzIC8g'
    'Ni4wOwogICAgcmV0dXJuIHcwICogcDAgKyB3MSAqIHAxICsgdzIgKiBwMiAr'
    'IHczICogcDM7Cn0nJycKICAgICAgICApICsgIiIiCgoiIiIKCiAgICBpbXBv'
    'cnQgYmFzZTY0IGFzIF9iNjRlCiAgICBnZXRfY2hhbm5lbF9vdXRwdXQgPSBf'
    'YjY0ZS5iNjRkZWNvZGUoJ0x5OGdkbWxpVkdGaUlHbHpJR1JsWTJ4aGNtVmtJ'
    'R0Z6SUdFZ1oyeHZZbUZzSUdOdmJuTjBJR1pzYjJGMFd6TXlYU0J1WldGeUlI'
    'Um9aU0IwYjNBZ2IyWWdRMjl0Ylc5dUNpOHZJQ2h5YVdkb2RDQmhablJsY2lC'
    'd1pYSnBiMlJVWVdKc1pTa3VJRVJ2YmlkMElISmxaR1ZqYkdGeVpTQnBkQ0Jv'
    'WlhKbExnb0tMeThnNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0'
    'cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRw'
    'U0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBT'
    'QTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNB'
    'NHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0'
    'cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRw'
    'U0E0cFNBNHBTQTRwU0E0cFNBNHBTQUNpOHZJRjluWTI5Q2IyUjVJT0tBbENC'
    'eVpXNWtaWElnYjI1bElITmhiWEJzWlNCdlppQmhJR05vWVc1dVpXd2daMmwy'
    'Wlc0Z1MwNVBWMDRnZEhKcFoyZGxjaUJwYm1adkxnb3ZMd292THlCRmVIUnlZ'
    'V04wWldRZ1puSnZiU0JuWlhSRGFHRnVibVZzVDNWMGNIVjBKM01nWW05a2VT'
    'QjBieUJ6ZFhCd2IzSjBJSEJ5WlhacGIzVnpMVzV2ZEdVZ1kzSnZjM05tWVdS'
    'bExnb3ZMeUJVYUdVZ2IzSnBaMmx1WVd3Z2IzVjBaWElnWm5WdVkzUnBiMjRn'
    'Wkdsa0lDSm1hVzVrSUhSeWFXZG5aWElnNG9hU0lHTnZiWEIxZEdVZ2IzVjBj'
    'SFYwSUdGeklHOXVaUW92THlCellXMXdiR1V1SWlCR2IzSWdZM0p2YzNObVlX'
    'UmxJSGRsSUc1bFpXUWdkRzhnWTI5dGNIVjBaU0IwYUdVZ2IzVjBjSFYwSUZS'
    'WFNVTkZJT0tBbENCdmJtTmxJR1p2Y2dvdkx5QjBhR1VnWTNWeWNtVnVkQ0Iw'
    'Y21sbloyVnlMQ0J2Ym1ObElHWnZjaUIwYUdVZ2RISnBaMmRsY2lCQ1JVWlBV'
    'a1VnYVhRZzRvQ1VJR0Z1WkNCaWJHVnVaQ0J2ZG1WeUNpOHZJSFJvWlNCbWFY'
    'SnpkQ0EyTkNCellXMXdiR1Z6SUdGbWRHVnlJR0VnY21WMGNtbG5aMlZ5TGdv'
    'dkx3b3ZMeUJRWVhKaGJXVjBaWEp6SUhSeWFXZFFZWFF2ZEhKcFoxSnZkeTkw'
    'Y21sblRtOTBaUzkwYjI1bFUyeHBaR1ZVWVhKblpYUWdZWEpsSUhkb1lYUWdk'
    'R2hsSUc5MWRHVnlDaTh2SUdaMWJtTjBhVzl1SjNNZ2RISnBaMmRsY2lCelpX'
    'RnlZMmdnZDI5MWJHUWdhR0YyWlNCamIyMXdkWFJsWkRzZ2RHaGxJR0p2Wkhr'
    'Z2RYTmxjeUIwYUdWdENpOHZJR2xrWlc1MGFXTmhiR3g1SUNoM1lYTWdZWE1n'
    'WUd4dlkyRnNJSFpoY2lBOUlISmxjM1ZzZENCdlppQnpaV0Z5WTJoZ0xDQnVi'
    'M2NnWVhNZ2NHRnlZVzFsZEdWeWN5a3VDaTh2SU9LVWdPS1VnT0tVZ09LVWdP'
    'S1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09L'
    'VWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tV'
    'Z09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1Vn'
    'T0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdP'
    'S1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09L'
    'VWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdBcG1iRzlo'
    'ZENCZloyTnZRbTlrZVNocGJuUWdZMmdzSUZCdmMybDBhVzl1SUhCdmN5d2da'
    'bXh2WVhRZ2RHbHRaU3dnWm14dllYUWdjbTkzVkdsdFpTd0tJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ2FXNTBJSFJ5YVdkUVlYUXNJR2x1ZENCMGNtbG5VbTkzTENC'
    'T2IzUmxJSFJ5YVdkT2IzUmxMQ0JwYm5RZ2RHOXVaVk5zYVdSbFZHRnlaMlYw'
    'S1NCN0NpQWdJQ0JwWmlBb2RISnBaMDV2ZEdVdWFXNXpkSEoxYldWdWRDQThQ'
    'U0F3SUh4OElIUnlhV2RPYjNSbExtbHVjM1J5ZFcxbGJuUWdQaUF6TVNCOGZD'
    'QjBjbWxuVG05MFpTNXdaWEpwYjJRZ1BEMGdNQ2tLSUNBZ0lDQWdJQ0J5WlhS'
    'MWNtNGdNQzR3T3dvS0lDQWdJRk5oYlhCc1pVbHVabThnYzIxd0lEMGdjMkZ0'
    'Y0d4bGMxdDBjbWxuVG05MFpTNXBibk4wY25WdFpXNTBJQzBnTVYwN0NpQWdJ'
    'Q0JwWmlBb2MyMXdMbXhsYm1kMGFDQTlQU0F3S1NCeVpYUjFjbTRnTUM0d093'
    'b0tJQ0FnSUM4dklGUnBZMnN0WW1GelpXUWdaV3hoY0hObFpEb2dhVzVzYVc1'
    'bElFZFNJR052YlhCMWRHRjBhVzl1TENCemEybHdJRzVoYldWa0lHbHVkR1Z5'
    'YldWa2FXRjBaWE1LSUNBZ0lHWnNiMkYwSUdWc1lYQnpaV1FnUFNBb1pteHZZ'
    'WFFvWm1WMFkyaFVhV05yS0hCaGRGUnBZMnRQWm1aelpYUmJjRzl6TG5OdmJt'
    'ZFFiM05kS3lod2IzTXVjbTkzTFhCaGRGTjBZWEowVW05M1czQnZjeTV6YjI1'
    'blVHOXpYU2twS1FvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQXJJSEJ2'
    'Y3k1MGFXTnJDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUMwZ1pteHZZ'
    'WFFvWm1WMFkyaFVhV05yS0hCaGRGUnBZMnRQWm1aelpYUmJkSEpwWjFCaGRG'
    'MHJLSFJ5YVdkU2IzY3RjR0YwVTNSaGNuUlNiM2RiZEhKcFoxQmhkRjBwS1Nr'
    'cENpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdMeUJVU1VOTFUxOVFSVkpm'
    'VTBWRE93b2dJQ0FnYVdZZ0tHVnNZWEJ6WldRZ1BDQXdMakFwSUhKbGRIVnli'
    'aUF3TGpBN0Nnb2dJQ0FnTHk4ZzRwU0E0cFNBNHBTQUlHWlRZVzF3YkdWUWIz'
    'TWdZV05qZFcxMWJHRjBiM0lnS0dWNFlXTjBJSEJsY2kxeWIzY2diR2x1WldG'
    'eUxYSmhiWEFnYVc1MFpXZHlZWFJwYjI0cElPS1VnT0tVZ09LVWdBb2dJQ0Fn'
    'THk4Z1VtVmhiQ0JRVkNCd1pYSnBiMlFnWlhadmJIVjBhVzl1SUdseklIQnBa'
    'V05sZDJselpTMXNhVzVsWVhJZ2QybDBhQ0JpY21WaGEzQnZhVzUwY3lCaGRD'
    'QnliM2NLSUNBZ0lDOHZJR0p2ZFc1a1lYSnBaWE1nS0dGdVpDQmpiR0Z0Y0hN'
    'Z2QyaGxiaUF6ZUhnZ2NtVmhZMmhsY3lCMFlYSm5aWFFwTGlCVWFHVWdjMmx1'
    'WjJ4bExXbHVkR1ZuY21Gc0NpQWdJQ0F2THlCbWIzSnRkV3hoSUdCREtsUXZa'
    'RkFnS2lCc2JpaFFNUzlRTUNsZ0lHWnliMjBnZEhKcFoyZGxjaUIwYnlCamRY'
    'SnlaVzUwSUdseklIZHliMjVuSUdKbFkyRjFjMlVLSUNBZ0lDOHZJR2wwSUdG'
    'emMzVnRaWE1nWVNCVFNVNUhURVVnYkdsdVpXRnlJSEpoYlhBZzRvQ1VJR0Zq'
    'ZEhWaGJDQlFLSFFwSUdseklHMTFiSFJwTFhObFoyMWxiblF1SUVacGVEb0tJ'
    'Q0FnSUM4dklHRmpZM1Z0ZFd4aGRHVWdaWGhoWTNRZ2NHVnlMWEp2ZHlCamIy'
    'NTBjbWxpZFhScGIyNXpJR1IxY21sdVp5QjBhR1VnWm05eWQyRnlaQ0J6WTJG'
    'dUlIVnphVzVuQ2lBZ0lDQXZMeURpaUtzb1F5OVFLSFFwS1dSMElEMGdReXBV'
    'WDNKdmR5OG9VRjlsYm1RdFVGOXpkR0Z5ZENrZ0tpQnNiaWhRWDJWdVpDOVFY'
    'M04wWVhKMEtTQm1iM0lnWldGamFBb2dJQ0FnTHk4Z2MyVm5iV1Z1ZEN3Z2NH'
    'eDFjeUJoSUhCaGNuUnBZV3d0Y205M0lIUmhhV3d2YUdWaFpDQm1iM0lnZEhK'
    'cFoyZGxjaUJoYm1RZ1kzVnljbVZ1ZENCeWIzZHpMZ29nSUNBZ0x5OGdRMjl6'
    'ZERvZ2ZqRXdJR1Y0ZEhKaElHOXdjeUJ3WlhJZ1ptOXlkMkZ5WkMxelkyRnVJ'
    'SEp2ZHl3Z2JtOGdaWGgwY21FZ2RHVjRkSFZ5WlNCbVpYUmphR1Z6TGdvZ0lD'
    'QWdabXh2WVhRZ1gyWlRZVzF3YkdWUWIzTkJZMk1nUFNBd0xqQTdDZ29nSUNB'
    'Z2FXNTBJRjl3WTNRZ1BTQnBiblFvY0c5ekxuUnBZMnNwT3dvZ0lDQWdUbTkw'
    'WlNCZmNHTnlJRDBnWjJWMFRtOTBaU2h3YjNNdWMyOXVaMUJ2Y3l3Z2NHOXpM'
    'bkp2ZHl3Z1kyZ3BPd29LSUNBZ0lDOHZJT0tVZ09LVWdDQkRiMjFpYVc1bFpD'
    'Qm1iM0ozWVhKa0lITmpZVzQ2SUhKbFluVnBiR1FnY0dsMFkyZ2dRVTVFSUha'
    'dmJIVnRaU0JtY205dElIUnlhV2RuWlhJZ2RHOGdZM1Z5Y21WdWRDRGlsSURp'
    'bElBS0lDQWdJR1pzYjJGMElHVm1abVZqZEdsMlpWQmxjbWx2WkNBOUlHWnNi'
    'MkYwS0hSeWFXZE9iM1JsTG5CbGNtbHZaQ2s3Q2lBZ0lDQm1iRzloZENCMFlY'
    'Sm5aWFJRWlhKcGIyUWdJQ0FnUFNCbWJHOWhkQ2gwY21sblRtOTBaUzV3WlhK'
    'cGIyUXBPd29LSUNBZ0lDOHZJT0tVZ09LVWdDQldiMngxYldVZ2FXNXBkR2xo'
    'YkdsNllYUnBiMjRnS0ZCVUlIQmxjbWx2WkMxdmJteDVMWEpsZEhKcFoyZGxj'
    'aUJ4ZFdseWF5a2c0cFNBNHBTQUNpQWdJQ0F2THlCUVZDQnpaVzFoYm5ScFkz'
    'TTZJR0VnY205M0lIZHBkR2dnY0dWeWFXOWtJRDRnTUNCaWRYUWdUazhnYVc1'
    'emRISjFiV1Z1ZENCdWRXMWlaWElnYVhNZ1lRb2dJQ0FnTHk4Z2NtVjBjbWxu'
    'WjJWeUlIUm9ZWFFnVWtWVFZFRlNWRk1nZEdobElITmhiWEJzWlNCaGRDQnZa'
    'bVp6WlhRZ01DQkNWVlFnUzBWRlVGTWdkR2hsSUhCeWFXOXlDaUFnSUNBdkx5'
    'QjJiMngxYldVZzRvQ1VJR2wwSjNNZ2RHaGxJSE5oYldVZ2FXNXpkSEoxYldW'
    'dWRDQmlaV2x1WnlCeVpTMXdiR0Y1WldRc0lHNXZkQ0JoSUdaeVpYTm9JR3ho'
    'ZEdOb0xnb2dJQ0FnTHk4S0lDQWdJQzh2SUZSb1pTQmlkVzVrYkdWa0lIUnlh'
    'V2RuWlhJdFptbHVaR1Z5SUdKaFkydDBjbUZqYTNNZ1lIUnlhV2RPYjNSbExt'
    'bHVjM1J5ZFcxbGJuUmdJR1p5YjIwZ1lRb2dJQ0FnTHk4Z2NISnBiM0lnY205'
    'M0lHWnZjaUJ6WVcxd2JHVWdiRzl2YTNWd0xDQmlkWFFnZEdobElFOVNTVWRK'
    'VGtGTUlIUnlhV2RuWlhJZ1kyVnNiQ0IwWld4c2N5QjFjd29nSUNBZ0x5OGdk'
    'MmhsZEdobGNpQlFWQ0IzYjNWc1pDQmtieUJoSUhadmJDQnlaWE5sZEM0Z1NX'
    'WWdkR2hsSUc5eWFXZHBibUZzSUdObGJHd2dhR0ZrSUc1dklHbHVjM1FzQ2lB'
    'Z0lDQXZMeUIzWlNCdVpXVmtJSFJ2SUhKbFkyOXVjM1J5ZFdOMElIUm9aU0Iy'
    'YjJ4MWJXVWdkR2hoZENCM1lYTWdhVzRnWldabVpXTjBJR3AxYzNRZ1ltVm1i'
    'M0psQ2lBZ0lDQXZMeUIwYUdseklIQmxjbWx2WkMxdmJteDVJSEpsZEhKcFoy'
    'ZGxjaUJpZVNCM1lXeHJhVzVuSUdKaFkyc2dkRzhnZEdobElHeGhjM1FnYVc1'
    'emRDMWlaV0Z5YVc1bkNpQWdJQ0F2THlCeWIzY2dZVzVrSUdadmNuZGhjbVF0'
    'YzJOaGJtNXBibWNnWldabVpXTjBjeTRLSUNBZ0lDOHZDaUFnSUNBdkx5QlVh'
    'R2x6SUhkaGN5Qm1iM1Z1WkNCaWVTQndaWEl0Y205M0lHVnVaWEpuZVNCamIy'
    'MXdZWEpsSUdGbllXbHVjM1FnZUcxd0lHOXVJSE52Ym1jdGNHOXpJREUwQ2lB'
    'Z0lDQXZMeUFvY0dGMGRHVnliaUF6TkNrZ1kyZ3pPaUJ3WlhKcGIyUXRiMjVz'
    'ZVNCeVpYUnlhV2RuWlhKeklHSmxkSGRsWlc0Z1EzaDRMWE5sZENCeWIzZHpJ'
    'SGRsY21VS0lDQWdJQzh2SUhCc1lYbHBibWNnWVhRZ1puVnNiQ0IyYjJ4MWJX'
    'VWdLSDR3TGpBNUlGSk5VeWtnYVc1emRHVmhaQ0J2WmlCMGFHVWdaR2x0YldW'
    'a0lHVmphRzhnYkdWMlpXd0tJQ0FnSUM4dklDaCtNQzR3TXlCU1RWTXBMaUJX'
    'YjJ3dGNISmxjMlZ5ZG1VZ1kzVjBjeUIwYjNSaGJDQmphRE1nVWsxVElHVnlj'
    'bTl5SUhaeklIaHRjQ0JpZVNBeU1NT1hMZ29nSUNBZ0x5OGc0cFNBNHBTQUlG'
    'WnZiSFZ0WlNCemJXOXZkR2hwYm1jZ2MzbHpkR1Z0SU9LVWdPS1VnT0tVZ09L'
    'VWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tV'
    'Z09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1Vn'
    'T0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdP'
    'S1VnT0tVZ09LVWdPS1VnT0tVZ0FvZ0lDQWdMeThnUjJWdVpYSmhiR2w2WldR'
    'Z05qUXRjMkZ0Y0d4bElISmhiWEFnYjI0Z1JWWkZVbGtnZG05c2RXMWxJR05v'
    'WVc1blpTQW9RM2g0TENCRlFYZ3NJRVZDZUN3Z1FYaDRDaUFnSUNBdkx5QjBh'
    'V05ySUdKdmRXNWtZWEpwWlhNc0lIQnNkWE1nYVc1b1pYSnBkR1ZrSUhaaGJI'
    'VmxjeUJoWTNKdmMzTWdabTl5ZDJGeVpDMXpZMkZ1SUhKdmQzTXBMZ29nSUNB'
    'Z0x5OGdVbVZ3YkdGalpYTWdNamdnYVc1c2FXNWxJR0IyYjJ4MWJXVWdQU0JZ'
    'WUNCdGRYUmhkR2x2Ym5NZ2QybDBhQ0JoSUhWdWFXWnZjbTBnY0dGMGRHVnli'
    'aUIwYUdGMENpQWdJQ0F2THlCd2NtVnpaWEoyWlhNZ2RHaGxJRzF2YzNRZ2Nt'
    'VmpaVzUwSUdOb1lXNW5aU0JoYm1RZ2NtRnRjSE1nWVhRZ2IzVjBjSFYwSUhS'
    'cGJXVXVJRmRwZEdodmRYUWdkR2hwY3dvZ0lDQWdMeThnY21GdGNDd2daV0Zq'
    'YUNCMmIyeDFiV1VnWldabVpXTjBJSEJ5YjJSMVkyVnpJR0VnYzJsdVoyeGxM'
    'WE5oYlhCc1pTQnpkR1Z3SUdsdUlHRnRjR3hwZEhWa1pTd0tJQ0FnSUM4dklH'
    'RnVaQ0I1YjNVZ2FHVmhjaUJwZENCaGN5QmhJSE5vWVhKd0lHTnNhV05ySU9L'
    'QWxDQndZWEowYVdOMWJHRnliSGtnWW1Ga0lHOXVJRU40ZUMxb1pXRjJlUW9n'
    'SUNBZ0x5OGdjR0YwZEdWeWJuTWdLSEp2ZDNNZ05TMHpNQ0J2WmlCd1lYUjBa'
    'WEp1SURBc0lHRnNiQ0J2WmlCd1lYUjBaWEp1SURFM0xDQmxkR011S1M0S0lD'
    'QWdJQzh2Q2lBZ0lDQXZMeUJVZDI4Z2MzUmhkR1VnZG1Gc2RXVnpJSEJzZFhN'
    'Z1lTQjBhV05ySUhOMFlXMXdPZ29nSUNBZ0x5OGdJQ0JmZG05c1VISmxkaUE5'
    'SUhSb1pTQjJZV3gxWlNCQ1JVWlBVa1VnZEdobElHMXZjM1FnY21WalpXNTBJ'
    'R05vWVc1blpRb2dJQ0FnTHk4Z0lDQmZkbTlzUTNWeWNpQTlJSFJvWlNCMllX'
    'eDFaU0JCUmxSRlVpQW9ZM1Z5Y21WdWRDQm5jbTkxYm1RZ2RISjFkR2dwQ2lB'
    'Z0lDQXZMeUFnSUY5MmIyeERhR0Z1WjJWQmRGUnBZMnRHSUQwZ1oyeHZZbUZz'
    'TFhScFkyc3RabXh2WVhRZ1lYUWdkMmhwWTJnZ2RHaGxJR05vWVc1blpTQm9Z'
    'WEJ3Wlc1bFpBb2dJQ0FnTHk4S0lDQWdJQzh2SUVGMElHOTFkSEIxZENCMGFX'
    'MWxPZ29nSUNBZ0x5OGdJQ0IyVW1GdGNDQWdQU0JqYkdGdGNDZ29jRzl6TG5S'
    'cFkydEdJQzBnWDNadmJFTm9ZVzVuWlVGMFZHbGphMFlwSUNvZ1UwRk5VRjlR'
    'UlZKZlZFbERTeUF2SURZMExDQXdMQ0F4S1FvZ0lDQWdMeThnSUNCbFptWldi'
    'MndnUFNCdGFYZ29YM1p2YkZCeVpYWXNJRjkyYjJ4RGRYSnlMQ0IyVW1GdGND'
    'a2dLeUIwY21WdGIyeHZSR1ZzZEdFS0lDQWdJQzh2Q2lBZ0lDQXZMeUJVZDI4'
    'Z2FHVnNjR1Z5SUcxaFkzSnZjem9LSUNBZ0lDOHZJQ0FnVms5TVgwbE9TVlFv'
    'VmlrZ0lDQWc0b0NVSUhObGRDQmliM1JvSUhCeVpYWWdZVzVrSUdOMWNuSWdk'
    'RzhnVml3Z2JtOGdkSEpoYm5OcGRHbHZiZ29nSUNBZ0x5OGdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ2gxYzJWa0lHRjBJSFJ5YVdkblpYSWdhVzVwZERz'
    'Z1pYaHBjM1JwYm1jZ05qUXRjMkZ0Y0d4bElHQmtaV05zYVdOcllBb2dJQ0Fn'
    'THk4Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHWmhZM1J2Y2lCb1lXNWti'
    'R1Z6SUhSb1pTQjBjbWxuWjJWeUlHWmhaR1V0YVc0cExnb2dJQ0FnTHk4Z0lD'
    'QldUMHhmVTBWVUtGWXNJRlFwSUNEaWdKUWdjSEp2Ylc5MFpTQmpkWEp5NG9h'
    'U2NISmxkaXdnYzJWMElHTjFjbklnZEc4Z1Zpd2djM1JoYlhBZ2RHbGpheUJV'
    'Q2lBZ0lDQXZMeUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnS0hWelpXUWdZ'
    'WFFnWlhabGNua2dhVzR0Y0d4aGVTQjJiMngxYldVZ1kyaGhibWRsS1M0S0lD'
    'QWdJQzh2Q2lBZ0lDQXZMeUJVYUdVZ2JXRmpjbTl6SUhWelpTQmhJR052Ylcx'
    'aExXVjRjSEpsYzNOcGIyNGdZbTlrZVNCemJ5QjBhR1Y1SUdWNGNHRnVaQ0Jq'
    'YkdWaGJteDVJR2x1YzJsa1pRb2dJQ0FnTHk4Z1lXNTVJRWRNVTB3Z2MzUmhk'
    'R1Z0Wlc1MElHTnZiblJsZUhRZ0tHbG1MMlZzYzJVZ2QybDBhRzkxZENCaWNt'
    'RmpaWE1zSUdWMFl5NHBMZ29nSUNBZ2FXNTBJQ0FnWDNadmJGQnlaWFlnUFNB'
    'd093b2dJQ0FnYVc1MElDQWdYM1p2YkVOMWNuSWdQU0F3T3dvZ0lDQWdabXh2'
    'WVhRZ1gzWnZiRU5vWVc1blpVRjBWR2xqYTBZZ1BTQXRNV1U1T3lBZ0x5OGda'
    'bUZ5TFhCaGMzUWdjMlZ1ZEdsdVpXdzZJSEpoYlhBZ1puVnNiSGtnWTI5dGNH'
    'eGxkR1VLSUNBZ0lHWnNiMkYwSUY5MGNtVnRiMnh2UkdWc2RHRWdJQ0FnSUQw'
    'Z01DNHdPeUFnSUM4dklIUnlaVzF2Ykc4Z1lYQndiR2xsY3lCaGRDQnZkWFJ3'
    'ZFhRc0lHNXZkQ0IyYVdFZ1ZrOU1YMU5GVkFvS0lDQWdJQ05rWldacGJtVWdW'
    'azlNWDBsT1NWUW9WaWtnSUNBZ0tGOTJiMnhRY21WMklEMGdLRllwTENCZmRt'
    'OXNRM1Z5Y2lBOUlGOTJiMnhRY21WMktRb2dJQ0FnSTJSbFptbHVaU0JXVDB4'
    'ZlUwVlVLRllzSUZRcElDQW9YM1p2YkZCeVpYWWdQU0JmZG05c1EzVnljaXdn'
    'WDNadmJFTjFjbklnUFNBb1Zpa3NJRjkyYjJ4RGFHRnVaMlZCZEZScFkydEdJ'
    'RDBnS0ZRcEtRb0tJQ0FnSUM4dklGQnlaUzFqYjIxd2RYUmxaQ0IwYVdOcklH'
    'OW1JSFJvWlNCMGNtbG5aMlZ5SUhKdmR5ZHpJR1pwY25OMElIUnBZMnN1SUZW'
    'elpXUWdZWE1nZEdobElHTm9ZVzVuWlFvZ0lDQWdMeThnYzNSaGJYQWdabTl5'
    'SUdGc2JDQjBjbWxuWjJWeUxYSnZkeUIyYjJ3Z1pXWm1aV04wY3k0S0lDQWdJ'
    'R1pzYjJGMElGOTBjbWxuWjJWeVZHbGphMFlnUFNCbWJHOWhkQ2htWlhSamFG'
    'UnBZMnNvY0dGMFZHbGphMDltWm5ObGRGdDBjbWxuVUdGMFhTQXJJQ2gwY21s'
    'blVtOTNJQzBnY0dGMFUzUmhjblJTYjNkYmRISnBaMUJoZEYwcEtTazdDZ29n'
    'SUNBZ1RtOTBaU0JmZEhKcFowTmxiR3hQY21sbklEMGdaMlYwVG05MFpTaDBj'
    'bWxuVUdGMExDQjBjbWxuVW05M0xDQmphQ2s3Q2lBZ0lDQnBaaUFvWDNSeWFX'
    'ZERaV3hzVDNKcFp5NXBibk4wY25WdFpXNTBJRDRnTUNrZ2V3b2dJQ0FnSUNB'
    'Z0lGWlBURjlKVGtsVUtITnRjQzUyYjJ4MWJXVXBPeUFnTHk4Z1VtVmhiQ0Jw'
    'Ym5OMGNuVnRaVzUwSUd4aGRHTm9JT0tBbENCa1pXTnNhV05ySUdoaGJtUnNa'
    'WE1nWm1Ga1pRb2dJQ0FnZlNCbGJITmxJSHNLSUNBZ0lDQWdJQ0F2THlCUVpY'
    'SnBiMlF0YjI1c2VTQnlaWFJ5YVdkblpYSWc0b0NVSUdacGJtUWdkR2hsSUd4'
    'aGMzUWdhVzV6ZEMxaVpXRnlhVzVuSUhKdmR5d2dhVzVwZENCbWNtOXRDaUFn'
    'SUNBZ0lDQWdMeThnYVhSeklHbHVjM1J5ZFcxbGJuUW5jeUJrWldaaGRXeDBJ'
    'SFp2YkhWdFpTd2dkR2hsYmlCbWIzSjNZWEprTFhOallXNGdaV1ptWldOMGN5'
    'QjFjQ0IwYndvZ0lDQWdJQ0FnSUM4dklDaGlkWFFnYm05MElHbHVZMngxWkds'
    'dVp5a2dkSEpwWjFKdmR5NGdRbTkxYm1SbFpDQnpZMkZ1T2lBek1pQnliM2R6'
    'SUdKaFkydDNZWEprQ2lBZ0lDQWdJQ0FnTHk4Z1kyOTJaWEp6SUhSb1pTQjJZ'
    'WE4wSUcxaGFtOXlhWFI1SUc5bUlHMTFjMmxqWVd3Z1kyRnpaWE11Q2lBZ0lD'
    'QWdJQ0FnYVc1MElGOXBibk4wVW05M0lEMGdkSEpwWjFKdmR5d2dYMmx1YzNS'
    'UVlYUWdQU0IwY21sblVHRjBPd29nSUNBZ0lDQWdJR0p2YjJ3Z1gyWnZkVzVr'
    'U1c1emRFeGhkR05vSUQwZ1ptRnNjMlU3Q2lBZ0lDQWdJQ0FnZXdvZ0lDQWdJ'
    'Q0FnSUNBZ0lDQnBiblFnYzFJZ1BTQjBjbWxuVW05M0xDQnpVQ0E5SUhSeWFX'
    'ZFFZWFE3Q2lBZ0lDQWdJQ0FnSUNBZ0lHWnZjaUFvYVc1MElHeGlJRDBnTVRz'
    'Z2JHSWdQQ0F6TWpzZ2JHSXJLeWtnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'YzFJdExUc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2h6VWlBOElEQXBJ'
    'SHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9jMUFnUGlBd0tT'
    'QjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lITlFMUzA3Q2lB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJSE5TSUQwZ2NHRjBVM1Jo'
    'Y25SU2IzZGJjMUJkSUNzZ0tIQmhkRkp2ZDA5bVpuTmxkRnR6VUNzeFhTQXRJ'
    'SEJoZEZKdmQwOW1abk5sZEZ0elVGMHBJQzBnTVRzS0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQjlJR1ZzYzJVZ2V5QmljbVZoYXpzZ2ZRb2dJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1RtOTBaU0J3'
    'YmlBOUlHZGxkRTV2ZEdVb2MxQXNJSE5TTENCamFDazdDaUFnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQnBaaUFvY0c0dWFXNXpkSEoxYldWdWRDQStJREFwSUhzS0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmZhVzV6ZEZKdmR5QTlJSE5TT3lC'
    'ZmFXNXpkRkJoZENBOUlITlFPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUY5bWIzVnVaRWx1YzNSTVlYUmphQ0E5SUhSeWRXVTdDaUFnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnWW5KbFlXczdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QjlDaUFnSUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNCOUNpQWdJQ0FnSUNB'
    'Z2FXWWdLQ0ZmWm05MWJtUkpibk4wVEdGMFkyZ3BJSHNLSUNBZ0lDQWdJQ0Fn'
    'SUNBZ1ZrOU1YMGxPU1ZRb2MyMXdMblp2YkhWdFpTazdJQ0F2THlCT2J5QnBi'
    'bk4wSUd4aGRHTm9JT0tBbENCa1pXTnNhV05ySUdoaGJtUnNaWE1nWm1Ga1pR'
    'b2dJQ0FnSUNBZ0lIMGdaV3h6WlNCN0NpQWdJQ0FnSUNBZ0lDQWdJRTV2ZEdV'
    'Z1gyeGhkR05vVG05MFpTQTlJR2RsZEU1dmRHVW9YMmx1YzNSUVlYUXNJRjlw'
    'Ym5OMFVtOTNMQ0JqYUNrN0NpQWdJQ0FnSUNBZ0lDQWdJRk5oYlhCc1pVbHVa'
    'bThnWDJ4aGRHTm9VMjF3SUQwZ2MyRnRjR3hsYzF0ZmJHRjBZMmhPYjNSbExt'
    'bHVjM1J5ZFcxbGJuUWdMU0F4WFRzS0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5'
    'c1lYUmphRk5uY2lBOUlIQmhkRlJwWTJ0UFptWnpaWFJiWDJsdWMzUlFZWFJk'
    'SUNzZ0tGOXBibk4wVW05M0lDMGdjR0YwVTNSaGNuUlNiM2RiWDJsdWMzUlFZ'
    'WFJkS1RzS0lDQWdJQ0FnSUNBZ0lDQWdabXh2WVhRZ1gyeGhkR05vVkdsamEw'
    'WWdQU0JtYkc5aGRDaG1aWFJqYUZScFkyc29YMnhoZEdOb1UyZHlLU2s3Q2lB'
    'Z0lDQWdJQ0FnSUNBZ0lGWlBURjlKVGtsVUtGOXNZWFJqYUZOdGNDNTJiMngx'
    'YldVcE95QWdMeThnVW1WamIyNXpkSEoxWTNScGIyNGdZbUZ6Wld4cGJtVWc0'
    'b0NVSUc1dklIUnlZVzV6YVhScGIyNEtJQ0FnSUNBZ0lDQWdJQ0FnTHk4Z1FY'
    'QndiSGtnYkdGMFkyZ2djbTkzSjNNZ2RHbGpheTB3SUhadmJDQmxabVpsWTNS'
    'eklDaDBjbUZ1YzJsMGFXOXVjeUJ6ZEdGdGNHVmtJR0YwSUhKdmR5QjBhV05y'
    'S1M0S0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5c1lYUmphRTV2ZEdVdVpXWm1a'
    'V04wSUQwOUlEQjRReWtnVms5TVgxTkZWQ2h0YVc0b1gyeGhkR05vVG05MFpT'
    'NXdZWEpoYlN3Z05qUXBMQ0JmYkdGMFkyaFVhV05yUmlrN0NpQWdJQ0FnSUNB'
    'Z0lDQWdJR1ZzYzJVZ2FXWWdLRjlzWVhSamFFNXZkR1V1WldabVpXTjBJRDA5'
    'SURCNFJTa2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5bGN5QTlJ'
    'Q2hmYkdGMFkyaE9iM1JsTG5CaGNtRnRJRDQrSURRcElDWWdNSGhHT3dvZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOWxkaUE5SUNCZmJHRjBZMmhPYjNS'
    'bExuQmhjbUZ0SUNBZ0lDQWdJQ1lnTUhoR093b2dJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ2FXWWdLRjlsY3lBOVBTQXdlRUVwSUNBZ0lDQWdWazlNWDFORlZDaGpi'
    'R0Z0Y0NoZmRtOXNRM1Z5Y2lBcklGOWxkaXdnTUN3Z05qUXBMQ0JmYkdGMFky'
    'aFVhV05yUmlrN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCbGJITmxJR2xtSUNo'
    'ZlpYTWdQVDBnTUhoQ0tTQldUMHhmVTBWVUtHTnNZVzF3S0Y5MmIyeERkWEp5'
    'SUMwZ1gyVjJMQ0F3TENBMk5Da3NJRjlzWVhSamFGUnBZMnRHS1RzS0lDQWdJ'
    'Q0FnSUNBZ0lDQWdmUW9nSUNBZ0lDQWdJQ0FnSUNBdkx5Qk1ZWFJqYUNCeWIz'
    'Y25jeUJ3WlhJdGRHbGpheUJ6Ykdsa1pTQW9RWGg0TENBMmVIZ3NJRFY0ZUNC'
    'MmIyeDFiV1VnY0dGeWRDa2diM1psY2lCbWRXeHNJSEp2ZHk0S0lDQWdJQ0Fn'
    'SUNBZ0lDQWdhV1lnS0Y5c1lYUmphRTV2ZEdVdVpXWm1aV04wSUQwOUlEQjRR'
    'U0I4ZkNCZmJHRjBZMmhPYjNSbExtVm1abVZqZENBOVBTQXdlRFlnZkh3S0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUY5c1lYUmphRTV2ZEdVdVpXWm1aV04wSUQw'
    'OUlEQjROU2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOTJkU0E5'
    'SUNoZmJHRjBZMmhPYjNSbExuQmhjbUZ0UGo0MEtTWXdlRVk3Q2lBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0JwYm5RZ1gzWmtJRDBnSUY5c1lYUmphRTV2ZEdVdWNH'
    'RnlZVzBnSUNBZ0pqQjRSanNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2x1ZENC'
    'ZmMzUmxjQ0E5SUNoZmRuVStNQ2tnUHlCZmRuVWdPaUF0WDNaa093b2dJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjltZEV3Z0lEMGdabVYwWTJoVWFXTnJL'
    'RjlzWVhSamFGTm5jaUFySURFcElDMGdabVYwWTJoVWFXTnJLRjlzWVhSamFG'
    'Tm5jaWtnTFNBeE93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1ZrOU1YMU5GVkNo'
    'amJHRnRjQ2hmZG05c1EzVnljaUFySUY5emRHVndJQ29nWDJaMFRDd2dNQ3dn'
    'TmpRcExDQmZiR0YwWTJoVWFXTnJSaWs3Q2lBZ0lDQWdJQ0FnSUNBZ0lIMEtJ'
    'Q0FnSUNBZ0lDQWdJQ0FnTHk4Z1JtOXlkMkZ5WkMxelkyRnVJR1ZtWm1WamRI'
    'TWdiMjRnY205M2N5QmZhVzV6ZEZKdmR5c3hJQzR1TGlCMGNtbG5VbTkzTFRF'
    'c0lIZGhiR3RwYm1jS0lDQWdJQ0FnSUNBZ0lDQWdMeThnZEdoeWIzVm5hQ0Jo'
    'Ym5rZ2FXNTBaWEp0WldScFlYUmxJSEJsY21sdlpDMXZibXg1SUhKbGRISnBa'
    'MmRsY25NZ2QybDBhRzkxZENCeVpYTmxkSFJwYm1jdUNpQWdJQ0FnSUNBZ0lD'
    'QWdJR2x1ZENCZmRtWndJRDBnWDJsdWMzUlFZWFFzSUY5MlpuSWdQU0JmYVc1'
    'emRGSnZkeUFySURFN0NpQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZmRtWnlJRDQ5'
    'SUhCaGRGTjBZWEowVW05M1cxOTJabkJkSUNzZ0tIQmhkRkp2ZDA5bVpuTmxk'
    'RnRmZG1ad0t6RmRJQzBnY0dGMFVtOTNUMlptYzJWMFcxOTJabkJkS1NrZ2V3'
    'b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1gzWm1jQ3NyT3lCZmRtWnlJRDBnS0Y5'
    'MlpuQWdQQ0JUVDA1SFgweEZUa2RVU0NrZ1B5QndZWFJUZEdGeWRGSnZkMXRm'
    'ZG1ad1hTQTZJREE3Q2lBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJ'
    'Q0FnWm05eUlDaHBiblFnWDNacElEMGdNRHNnWDNacElEd2dOalE3SUY5MmFT'
    'c3JLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1gzWm1jQ0ErSUhS'
    'eWFXZFFZWFFnZkh3Z0tGOTJabkFnUFQwZ2RISnBaMUJoZENBbUppQmZkbVp5'
    'SUQ0OUlIUnlhV2RTYjNjcEtTQmljbVZoYXpzS0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUdsbUlDaGZkbVp3SUQ0OUlGTlBUa2RmVEVWT1IxUklLU0JpY21WaGF6'
    'c0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmZG1aeUlENDlJSEJoZEZO'
    'MFlYSjBVbTkzVzE5MlpuQmRJQ3NnS0hCaGRGSnZkMDltWm5ObGRGdGZkbVp3'
    'S3pGZElDMGdjR0YwVW05M1QyWm1jMlYwVzE5MlpuQmRLU2tnZXdvZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBZ0lGOTJabkFyS3pzZ1gzWm1jaUE5SUNoZmRt'
    'WndJRHdnVTA5T1IxOU1SVTVIVkVncElEOGdjR0YwVTNSaGNuUlNiM2RiWDNa'
    'bWNGMGdPaUF3T3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHTnZiblJw'
    'Ym5WbE93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ1RtOTBaU0JmZG00Z1BTQm5aWFJPYjNSbEtGOTJabkFzSUY5Mlpu'
    'SXNJR05vS1RzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsdWRDQmZjMmR5VmlB'
    'OUlIQmhkRlJwWTJ0UFptWnpaWFJiWDNabWNGMGdLeUFvWDNabWNpQXRJSEJo'
    'ZEZOMFlYSjBVbTkzVzE5MlpuQmRLVHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'R2x1ZENCZlpuUldJQ0E5SUdabGRHTm9WR2xqYXloZmMyZHlWaUFySURFcElD'
    'MGdabVYwWTJoVWFXTnJLRjl6WjNKV0tTQXRJREU3Q2lBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0JtYkc5aGRDQmZkbFJwWTJ0R0lEMGdabXh2WVhRb1ptVjBZMmhV'
    'YVdOcktGOXpaM0pXS1NrN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9Y'
    'M1p1TG1WbVptVmpkQ0E5UFNBd2VFTXBJRlpQVEY5VFJWUW9iV2x1S0Y5MmJp'
    'NXdZWEpoYlN3Z05qUXBMQ0JmZGxScFkydEdLVHNLSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJR1ZzYzJVZ2FXWWdLRjkyYmk1bFptWmxZM1FnUFQwZ01IaEJJSHg4'
    'SUY5MmJpNWxabVpsWTNRZ1BUMGdNSGcyS1NCN0NpQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdhVzUwSUY5MmRTQTlJQ2hmZG00dWNHRnlZVzArUGpRcEpq'
    'QjRSaXdnWDNaa0lEMGdYM1p1TG5CaGNtRnRKakI0UmpzS0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lDQldUMHhmVTBWVUtHTnNZVzF3S0Y5MmIyeERkWEp5'
    'SUNzZ0tGOTJkVDR3UDE5MmRUb3RYM1prS1NBcUlGOW1kRllzSURBc0lEWTBL'
    'U3dnWDNaVWFXTnJSaWs3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0I5Q2lBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0JsYkhObElHbG1JQ2hmZG00dVpXWm1aV04wSUQw'
    'OUlEQjRSU2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0Jm'
    'WlhNZ1BTQW9YM1p1TG5CaGNtRnRJRDQrSURRcElDWWdNSGhHT3dvZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0JmWlhZZ1BTQWdYM1p1TG5CaGNt'
    'RnRJQ0FnSUNBZ0lDWWdNSGhHT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lHbG1JQ2hmWlhNZ1BUMGdNSGhCS1NBZ0lDQWdJRlpQVEY5VFJWUW9ZMnho'
    'YlhBb1gzWnZiRU4xY25JZ0t5QmZaWFlzSURBc0lEWTBLU3dnWDNaVWFXTnJS'
    'aWs3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1pXeHpaU0JwWmlBb1gy'
    'VnpJRDA5SURCNFFpa2dWazlNWDFORlZDaGpiR0Z0Y0NoZmRtOXNRM1Z5Y2lB'
    'dElGOWxkaXdnTUN3Z05qUXBMQ0JmZGxScFkydEdLVHNLSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZmRtNHVa'
    'V1ptWldOMElEMDlJREI0TlNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJR2x1ZENCZmRuVWdQU0FvWDNadUxuQmhjbUZ0UGo0MEtTWXdlRVlzSUY5'
    'MlpDQTlJRjkyYmk1d1lYSmhiU1l3ZUVZN0NpQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdWazlNWDFORlZDaGpiR0Z0Y0NoZmRtOXNRM1Z5Y2lBcklDaGZk'
    'blUrTUQ5ZmRuVTZMVjkyWkNrZ0tpQmZablJXTENBd0xDQTJOQ2tzSUY5MlZH'
    'bGphMFlwT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0FnWDNabWNpc3JPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhV1ln'
    'S0Y5MlpuSWdQajBnY0dGMFUzUmhjblJTYjNkYlgzWm1jRjBnS3lBb2NHRjBV'
    'bTkzVDJabWMyVjBXMTkyWm5Bck1WMGdMU0J3WVhSU2IzZFBabVp6WlhSYlgz'
    'Wm1jRjBwS1NCN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdYM1ptY0Nz'
    'ck95QmZkbVp5SUQwZ0tGOTJabkFnUENCVFQwNUhYMHhGVGtkVVNDa2dQeUJ3'
    'WVhSVGRHRnlkRkp2ZDF0ZmRtWndYU0E2SURBN0NpQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNCOUNpQWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0I5Q2lBZ0lD'
    'QjlDZ29nSUNBZ0x5OGdWRzl1WlMxd2IzSjBZU0IwY21sbloyVnlJSEp2ZHpv'
    'Z2RHaHBjeUJ5YjNjZ1kyRnljbWxsY3lCaElETjRlQzgxZUhnZ2MyeHBaR1Vn'
    'ZEdGeVoyVjBMZ29nSUNBZ0x5OGdaV1ptWldOMGFYWmxVR1Z5YVc5a0lITjBZ'
    'WGx6SUdGMElIUm9aU0J3Y21WMmFXOTFjeUIwY21sbloyVnlKM01nY0dWeWFX'
    'OWtJQ2hoYkhKbFlXUjVJSE5sZENCaFltOTJaUW9nSUNBZ0x5OGdabkp2YlNC'
    'MGNtbG5UbTkwWlM1d1pYSnBiMlFwT3lCemJHbGtaU0JoWTJOMWJYVnNZWFJs'
    'Y3lCMGIzZGhjbVFnZEc5dVpWTnNhV1JsVkdGeVoyVjBJRzkyWlhJZ2NtOTNj'
    'eTRLSUNBZ0lHbG1JQ2gwYjI1bFUyeHBaR1ZVWVhKblpYUWdQaUF3S1NCN0Np'
    'QWdJQ0FnSUNBZ2RHRnlaMlYwVUdWeWFXOWtJRDBnWm14dllYUW9kRzl1WlZO'
    'c2FXUmxWR0Z5WjJWMEtUc0tJQ0FnSUgwS0NpQWdJQ0F2THlCQmNIQnNlU0Iw'
    'Y21sbloyVnlMWEp2ZHlCbFptWmxZM1J6T2lCRGVIZ2dLSE5sZENCMmIyd3BM'
    'Q0JCZUhndk5uaDRJQ2gyYjJ3Z2MyeHBaR1VnY0dGeWRHbGhiQzltZFd4c0tT'
    'd0tJQ0FnSUM4dklFVkJlQ0FvWm1sdVpTQjJiMndnZFhBZzRvQ1VJR2x1YzNS'
    'aGJuUXBMQ0JGUW5nZ0tHWnBibVVnZG05c0lHUnZkMjRnNG9DVUlHbHVjM1Jo'
    'Ym5RcExDQTFlSGdnS0hSdmJtVXJkbTlzSUhOc2FXUmxLUzRLSUNBZ0lDOHZJ'
    'RUZzYkNCMGNtbG5aMlZ5TFhKdmR5QjJiMndnWTJoaGJtZGxjeUJ6ZEdGdGND'
    'QjBhR1VnWTJoaGJtZGxJSFJwWTJzZ1lYTWdYM1J5YVdkblpYSlVhV05yUmlC'
    'emJ5QjBhR1VLSUNBZ0lDOHZJRFkwTFhOaGJYQnNaU0J5WVcxd0lHTnZiWEJz'
    'WlhSbGN5QjNhWFJvYVc0Z2RHaGxJR1pwY25OMElINHhMalZ0Y3lCdlppQjBh'
    'R1VnZEhKcFoyZGxjaUJ5YjNjdUNpQWdJQ0JwWmlBb2RISnBaMDV2ZEdVdVpX'
    'Wm1aV04wSUQwOUlEQjRReWtnZXdvZ0lDQWdJQ0FnSUZaUFRGOVRSVlFvYlds'
    'dUtIUnlhV2RPYjNSbExuQmhjbUZ0TENBMk5Da3NJRjkwY21sbloyVnlWR2xq'
    'YTBZcE93b2dJQ0FnZlNCbGJITmxJR2xtSUNoMGNtbG5UbTkwWlM1bFptWmxZ'
    'M1FnUFQwZ01IaEZLU0I3Q2lBZ0lDQWdJQ0FnTHk4Z1JYaDBaVzVrWldRZ1pX'
    'Wm1aV04wY3pvZ1JVRjRJR1pwYm1VZ2RtOXNJSFZ3TENCRlFuZ2dabWx1WlNC'
    'MmIyd2daRzkzYmlBb2FXNXpkR0Z1ZENCdmJpQjBhV05ySURBcENpQWdJQ0Fn'
    'SUNBZ2FXNTBJRjlsY3lBOUlDaDBjbWxuVG05MFpTNXdZWEpoYlNBK1BpQTBL'
    'U0FtSURCNFJqc0tJQ0FnSUNBZ0lDQnBiblFnWDJWMklEMGdJSFJ5YVdkT2Iz'
    'UmxMbkJoY21GdElDQWdJQ0FnSUNZZ01IaEdPd29nSUNBZ0lDQWdJR2xtSUNo'
    'ZlpYTWdQVDBnTUhoQktTQWdJQ0FnSUZaUFRGOVRSVlFvWTJ4aGJYQW9YM1p2'
    'YkVOMWNuSWdLeUJmWlhZc0lEQXNJRFkwS1N3Z1gzUnlhV2RuWlhKVWFXTnJS'
    'aWs3Q2lBZ0lDQWdJQ0FnWld4elpTQnBaaUFvWDJWeklEMDlJREI0UWlrZ1Zr'
    'OU1YMU5GVkNoamJHRnRjQ2hmZG05c1EzVnljaUF0SUY5bGRpd2dNQ3dnTmpR'
    'cExDQmZkSEpwWjJkbGNsUnBZMnRHS1RzS0lDQWdJSDBnWld4elpTQnBaaUFv'
    'ZEhKcFowNXZkR1V1WldabVpXTjBJRDA5SURCNFFTQjhmQ0IwY21sblRtOTBa'
    'UzVsWm1abFkzUWdQVDBnTUhnMklIeDhJSFJ5YVdkT2IzUmxMbVZtWm1WamRD'
    'QTlQU0F3ZURVcElIc0tJQ0FnSUNBZ0lDQXZMeUF3ZURVZ1BTQjBiMjVsSzNa'
    'dmJDQnpiR2xrWlRvZ2NHbDBZMmdnYUdGdVpHeGxaQ0JpZVNBd2VETXRaWEYx'
    'YVhaaGJHVnVkQ0JpYkc5amF5d2dkbTlzSUhCaGNtRnRJSE5oYldVZ1lYTWdN'
    'SGhCQ2lBZ0lDQWdJQ0FnYVc1MElGOXpkU0E5SUNoMGNtbG5UbTkwWlM1d1lY'
    'SmhiVDQrTkNrbU1IaEdMQ0JmYzJRZ1BTQjBjbWxuVG05MFpTNXdZWEpoYlNZ'
    'd2VFWTdDaUFnSUNBZ0lDQWdhVzUwSUY5emRHVndJRDBnS0Y5emRUNHdLU0Ev'
    'SUY5emRTQTZJQzFmYzJRN0NpQWdJQ0FnSUNBZ2FXWWdLSFJ5YVdkUVlYUWdQ'
    'VDBnY0c5ekxuTnZibWRRYjNNZ0ppWWdkSEpwWjFKdmR5QTlQU0J3YjNNdWNt'
    'OTNLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lGWlBURjlUUlZRb1kyeGhiWEFvWDNa'
    'dmJFTjFjbklnS3lCZmMzUmxjQ0FxSUY5d1kzUXNJREFzSURZMEtTd2dYM1J5'
    'YVdkblpYSlVhV05yUmlrN0NpQWdJQ0FnSUNBZ2ZTQmxiSE5sSUhzS0lDQWdJ'
    'Q0FnSUNBZ0lDQWdhVzUwSUY5MGN5QTlJR1psZEdOb1ZHbGpheWh3WVhSVWFX'
    'TnJUMlptYzJWMFczUnlhV2RRWVhSZEt5aDBjbWxuVW05M0xYQmhkRk4wWVhK'
    'MFVtOTNXM1J5YVdkUVlYUmRLU3N4S1FvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDMGdabVYwWTJoVWFXTnJLSEJoZEZScFkydFBabVp6WlhSYmRISnBa'
    'MUJoZEYwcktIUnlhV2RTYjNjdGNHRjBVM1JoY25SU2IzZGJkSEpwWjFCaGRG'
    'MHBLVHNLSUNBZ0lDQWdJQ0FnSUNBZ1ZrOU1YMU5GVkNoamJHRnRjQ2hmZG05'
    'c1EzVnljaUFySUY5emRHVndJQ29nS0Y5MGN5MHhLU3dnTUN3Z05qUXBMQ0Jm'
    'ZEhKcFoyZGxjbFJwWTJ0R0tUc0tJQ0FnSUNBZ0lDQjlDaUFnSUNCOUNnb2dJ'
    'Q0FnTHk4ZzRwU0E0cFNBSURCNFJVUWdibTkwWlNCa1pXeGhlVG9nYzJ0cGND'
    'QnZkWFJ3ZFhRZ1ptOXlJR0J3WVhKaGJXQWdkR2xqYTNNZ1lXWjBaWElnZEhK'
    'cFoyZGxjaUJ5YjNjZzRwU0E0cFNBQ2lBZ0lDQXZMeUJVYUdVZ2JtOTBaU0Jr'
    'YjJWemJpZDBJR0ZqZEhWaGJHeDVJSE4wWVhKMElIVnVkR2xzSUhScFkyc2dZ'
    'SEJoY21GdFlDQnZaaUIwYUdVZ2RISnBaMmRsY2lCeWIzY3VDaUFnSUNBdkx5'
    'QkZZWEpzYVdWeUlIUm9ZVzRnZEdoaGRDRGlocElnY21WMGRYSnVJSE5wYkdW'
    'dVkyVTdJR1ZtWm1WamRHbDJaU0JsYkdGd2MyVmtJR2x6SUhKbFpIVmpaV1F1'
    'Q2lBZ0lDQnBiblFnWDI1dmRHVkVaV3hoZVZScFkydHpJRDBnTURzS0lDQWdJ'
    'R2xtSUNoMGNtbG5UbTkwWlM1bFptWmxZM1FnUFQwZ01IaEZJQ1ltSUNnb2RI'
    'SnBaMDV2ZEdVdWNHRnlZVzBnUGo0Z05Da2dKaUF3ZUVZcElEMDlJREI0UkNr'
    'Z2V3b2dJQ0FnSUNBZ0lGOXViM1JsUkdWc1lYbFVhV05yY3lBOUlIUnlhV2RP'
    'YjNSbExuQmhjbUZ0SUNZZ01IaEdPd29nSUNBZ0lDQWdJR2xtSUNoMGNtbG5V'
    'R0YwSUQwOUlIQnZjeTV6YjI1blVHOXpJQ1ltSUhSeWFXZFNiM2NnUFQwZ2NH'
    'OXpMbkp2ZHlBbUppQmZjR04wSUR3Z1gyNXZkR1ZFWld4aGVWUnBZMnR6S1Fv'
    'Z0lDQWdJQ0FnSUNBZ0lDQnlaWFIxY200Z01DNHdPeUFnTHk4Z1ltVm1iM0ps'
    'SUdSbGJHRjVaV1FnZEhKcFoyZGxjZ29nSUNBZ0lDQWdJQzh2SUVGbWRHVnlJ'
    'R1JsYkdGNU9pQnpkV0owY21GamRDQmtaV3hoZVNCMGFXTnJjeUJtY205dElH'
    'VnNZWEJ6WldRZ2MyOGdkR2hsSUc1dmRHVWdjM1JoY25SeklHRjBJR1p5WlhO'
    'b0lIUTlNQW9nSUNBZ0lDQWdJR1ZzWVhCelpXUWdQU0J0WVhnb01DNHdMQ0Js'
    'YkdGd2MyVmtJQzBnWm14dllYUW9YMjV2ZEdWRVpXeGhlVlJwWTJ0ektTQXZJ'
    'RlJKUTB0VFgxQkZVbDlUUlVNcE93b2dJQ0FnZlFvS0lDQWdJQzh2SU9LVWdP'
    'S1VnQ0F3ZURsNGVDQnpZVzF3YkdVZ2IyWm1jMlYwSUNoMGNtbG5aMlZ5SUhK'
    'dmR5QnZibXg1S1RvZ2MzUmhjblFnWVhRZ2NHRnlZVzBnS2lBeU5UWWdhVzRn'
    'YzJGdGNHeGxJR1JoZEdFZzRwU0E0cFNBQ2lBZ0lDQnBiblFnWDNOaGJYQnNa'
    'VTltWm5ObGRDQTlJREE3Q2lBZ0lDQnBaaUFvZEhKcFowNXZkR1V1WldabVpX'
    'TjBJRDA5SURCNE9TQW1KaUIwY21sblRtOTBaUzV3WVhKaGJTQStJREFwSUhz'
    'S0lDQWdJQ0FnSUNCZmMyRnRjR3hsVDJabWMyVjBJRDBnZEhKcFowNXZkR1V1'
    'Y0dGeVlXMGdLaUF5TlRZN0NpQWdJQ0I5Q2dvZ0lDQWdMeThnNHBTQTRwU0FJ'
    'RlJ5YVdkblpYSWdjbTkzSjNNZ2NHbDBZMmdnYzJ4cFpHVWdaV1ptWldOMGN5'
    'QW9NWGg0THpKNGVDa2c0cFNBNHBTQUNpQWdJQ0F2THlCSlppQjBhR1VnZEhK'
    'cFoyZGxjaUJ5YjNjZ1kyRnljbWxsWkNBeGVIZ2dLSEJ2Y25SaElIVndLU0J2'
    'Y2lBeWVIZ2dLSEJ2Y25SaElHUnZkMjRwTENCMGFHOXpaUW9nSUNBZ0x5OGdj'
    'MnhwWkdWeklHaGhjSEJsYmlCdmJpQjBhV05yY3lBeExpNG9jM0JsWldRdE1T'
    'a2diMllnZEdobElIUnlhV2RuWlhJZ2NtOTNMaUFnVjJobGJpQndiM01nYVhN'
    'S0lDQWdJQzh2SUZCQlUxUWdkR2hsSUhSeWFXZG5aWElnY205M0xDQmhiR3dn'
    'S0hOd1pXVmtMVEVwSUc5bUlIUm9iM05sSUhScFkydHpJR2hoZG1VZ1kyOXRj'
    'R3hsZEdWa0xnb2dJQ0FnTHk4Z1YyaGxiaUJ3YjNNZ2FYTWdiMjRnZEdobElI'
    'UnlhV2RuWlhJZ2NtOTNJR2wwYzJWc1ppd2dkR2hsSUNKRGRYSnlaVzUwSUhK'
    'dmR5QndZWEowYVdGc0lIQnBkR05vQ2lBZ0lDQXZMeUJsWm1abFkzUWlJR0pz'
    'YjJOcklHSmxiRzkzSUdoaGJtUnNaWE1nYVhRZzRvQ1VJR1J2YmlkMElHUnZk'
    'V0pzWlMxaGNIQnNlU0JvWlhKbExnb2dJQ0FnYVdZZ0tDaDBjbWxuVUdGMElD'
    'RTlJSEJ2Y3k1emIyNW5VRzl6SUh4OElIUnlhV2RTYjNjZ0lUMGdjRzl6TG5K'
    'dmR5a2dKaVlLSUNBZ0lDQWdJQ0FvZEhKcFowNXZkR1V1WldabVpXTjBJRDA5'
    'SURCNE1TQjhmQ0IwY21sblRtOTBaUzVsWm1abFkzUWdQVDBnTUhneUtTQW1K'
    'aUIwY21sblRtOTBaUzV3WVhKaGJTQStJREFwSUhzS0lDQWdJQ0FnSUNCcGJu'
    'UWdYM1J5VTJkeUlEMGdjR0YwVkdsamEwOW1abk5sZEZ0MGNtbG5VR0YwWFNB'
    'cklDaDBjbWxuVW05M0lDMGdjR0YwVTNSaGNuUlNiM2RiZEhKcFoxQmhkRjBw'
    'T3dvZ0lDQWdJQ0FnSUdsdWRDQmZkSEpUY0dRZ1BTQm1aWFJqYUZScFkyc29Y'
    'M1J5VTJkeUlDc2dNU2tnTFNCbVpYUmphRlJwWTJzb1gzUnlVMmR5S1RzS0lD'
    'QWdJQ0FnSUNCcGJuUWdYM1J5VkdsamEzTWdQU0JmZEhKVGNHUWdMU0F4T3lB'
    'Z0x5OGdZV3hzSUhCdmMzUXRkR2xqYXkwd0lIUnBZMnR6SUc5bUlIUnlhV2Ru'
    'WlhJZ2NtOTNDaUFnSUNBZ0lDQWdhV1lnS0hSeWFXZE9iM1JsTG1WbVptVmpk'
    'Q0E5UFNBd2VERXBDaUFnSUNBZ0lDQWdJQ0FnSUdWbVptVmpkR2wyWlZCbGNt'
    'bHZaQ0E5SUcxaGVDZ3hNVE11TUN3Z1pXWm1aV04wYVhabFVHVnlhVzlrSUMw'
    'Z1pteHZZWFFvZEhKcFowNXZkR1V1Y0dGeVlXMGdLaUJmZEhKVWFXTnJjeWtw'
    'T3dvZ0lDQWdJQ0FnSUdWc2MyVWdJQzh2SURCNE1nb2dJQ0FnSUNBZ0lDQWdJ'
    'Q0JsWm1abFkzUnBkbVZRWlhKcGIyUWdQU0J0YVc0b09EVTJMakFzSUdWbVpt'
    'VmpkR2wyWlZCbGNtbHZaQ0FySUdac2IyRjBLSFJ5YVdkT2IzUmxMbkJoY21G'
    'dElDb2dYM1J5VkdsamEzTXBLVHNLSUNBZ0lIMEtDaUFnSUNBdkx5QlVjbUZq'
    'YXlCc1lYTjBJSFJ2Ym1VdGNHOXlkR0VnY21GMFpTQW9abTl5SUdWbVptVmpk'
    'Q0ExSUhSdklHbHVhR1Z5YVhRcExpQWdTVzVwZEdsaGJHbDZaV1FnWm5KdmJR'
    'b2dJQ0FnTHk4Z2RISnBaMmRsY2lCeWIzY25jeUF6ZUhnZ2NHRnlZVzA3SUhW'
    'd1pHRjBaV1FnWW5rZ1ptOXlkMkZ5WkNCelkyRnVJR0Z6SUdsMElIZGhiR3R6'
    'SUhCaGMzUWdNM2g0SUhKdmQzTXVDaUFnSUNCcGJuUWdYMnhoYzNSVVVGSmhk'
    'R1VnUFNBd093b2dJQ0FnYVdZZ0tIUnlhV2RPYjNSbExtVm1abVZqZENBOVBT'
    'QXdlRE1nSmlZZ2RISnBaMDV2ZEdVdWNHRnlZVzBnUGlBd0tTQmZiR0Z6ZEZS'
    'UVVtRjBaU0E5SUhSeWFXZE9iM1JsTG5CaGNtRnRPd29LSUNBZ0lDOHZJRlJ5'
    'YVdkblpYSWdjbTkzSjNNZ2RHRnBiQ0JqYjI1MGNtbGlkWFJwYjI0Z2RHOGda'
    'bE5oYlhCc1pWQnZjeUIyYVdFZ2NHVnlMWFJwWTJzZ2FXNTBaV2R5WVhScGIy'
    'NHVDaUFnSUNBdkx5QlFWQ0JrYjJWeklHUnBjMk55WlhSbElIQmxjaTEwYVdO'
    'cklITnNhV1JsSUhWd1pHRjBaWE1zSUc1dmRDQmpiMjUwYVc1MWIzVnpJSEpo'
    'YlhCeklPS0FsQW9nSUNBZ0x5OGdZMjl1ZEdsdWRXOTFjeTF5WVcxd0lHbHVk'
    'R1ZuY21Gc2N5QmthWFpsY21kbElHWnliMjBnZEhKMWRHZ2dabTl5SUdaaGMz'
    'UWdjMnhwWkdWekxpQlhaU0JzYjI5d0NpQWdJQ0F2THlCdmRtVnlJR1ZoWTJn'
    'Z2RHbGpheUJ2WmlCMGFHVWdkSEpwWjJkbGNpQnliM2NnWVc1a0lHRmtaQ0JE'
    'dzVka2RDOXdaWEpwYjJSZllYUmZkR2xqYXk0S0lDQWdJQzh2SUZScFkyc2dN'
    'Q0J6WldWeklIUnlhV2RPYjNSbExuQmxjbWx2WkM0Z1ZHbGphM01nTVM0dUtI'
    'TndaV1ZrTFRFcElITmxaU0JwYm1OeVpXMWxiblJoYkd4NUNpQWdJQ0F2THlC'
    'MWNHUmhkR1ZrSUhCbGNtbHZaSE1nYVdZZ01YaDRMeko0ZUNCcGN5QndjbVZ6'
    'Wlc1MExnb2dJQ0FnYVdZZ0tIUnlhV2RRWVhRZ0lUMGdjRzl6TG5OdmJtZFFi'
    'M01nZkh3Z2RISnBaMUp2ZHlBaFBTQndiM011Y205M0tTQjdDaUFnSUNBZ0lD'
    'QWdhVzUwSUY5elozSlVjbWxuSUNBOUlIQmhkRlJwWTJ0UFptWnpaWFJiZEhK'
    'cFoxQmhkRjBnS3lBb2RISnBaMUp2ZHlBdElIQmhkRk4wWVhKMFVtOTNXM1J5'
    'YVdkUVlYUmRLVHNLSUNBZ0lDQWdJQ0JwYm5RZ1gzUnlhV2RHZFd4c0lEMGda'
    'bVYwWTJoVWFXTnJLRjl6WjNKVWNtbG5JQ3NnTVNrZ0xTQm1aWFJqYUZScFky'
    'c29YM05uY2xSeWFXY3BPeUFnTHk4Z1BTQnpjR1ZsWkFvZ0lDQWdJQ0FnSUda'
    'c2IyRjBJRjlEWmw5MGNtbG5JRDBnWXpSemNHVmxaSE5iYzIxd0xtWnBibVYw'
    'ZFc1bElDWWdNSGhHWFNBcUlEUXlPQzR3T3dvZ0lDQWdJQ0FnSUdac2IyRjBJ'
    'RjlrZENBOUlERXVNQ0F2SUZSSlEwdFRYMUJGVWw5VFJVTTdDaUFnSUNBZ0lD'
    'QWdabXh2WVhRZ1gxQjBJRDBnWm14dllYUW9kSEpwWjA1dmRHVXVjR1Z5YVc5'
    'a0tUc0tJQ0FnSUNBZ0lDQXZMeUJRWlhJdGRHbGpheUJ6Ykdsa1pTQnpkR1Z3'
    'SUdGdGIzVnVkQ0FvYzJsbmJtVmtLUW9nSUNBZ0lDQWdJR2x1ZENCZmMzUmxj'
    'Q0E5SURBN0NpQWdJQ0FnSUNBZ2FXWWdLSFJ5YVdkT2IzUmxMbVZtWm1WamRD'
    'QTlQU0F3ZURFcElGOXpkR1Z3SUQwZ0xYUnlhV2RPYjNSbExuQmhjbUZ0T3dv'
    'Z0lDQWdJQ0FnSUdWc2MyVWdhV1lnS0hSeWFXZE9iM1JsTG1WbVptVmpkQ0E5'
    'UFNBd2VESXBJRjl6ZEdWd0lEMGdkSEpwWjA1dmRHVXVjR0Z5WVcwN0NpQWdJ'
    'Q0FnSUNBZ0x5OGdRV05qZFcxMWJHRjBaU0J3WlhJdGRHbGphem9nZEdsamF5'
    'QXdJQ3NnZEdsamEzTWdNUzR1S0hOd1pXVmtMVEVwQ2lBZ0lDQWdJQ0FnTHk4'
    'Z1FtOTFibVFnZEc4Z016SWdabTl5SUdOdmJYQnBiR1Z5SUhOaFptVjBlVHNn'
    'YzNCbFpXUWdhWE1nNG9ta0lETXhJR2x1SUhCeVlXTjBhV05sTGdvZ0lDQWdJ'
    'Q0FnSUdadmNpQW9hVzUwSUY5MElEMGdNRHNnWDNRZ1BDQXpNanNnWDNRckt5'
    'a2dld29nSUNBZ0lDQWdJQ0FnSUNCcFppQW9YM1FnUGowZ1gzUnlhV2RHZFd4'
    'c0tTQmljbVZoYXpzS0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5UWRDQStJREF1'
    'TUNrZ1gyWlRZVzF3YkdWUWIzTkJZMk1nS3owZ1gwTm1YM1J5YVdjZ0tpQmZa'
    'SFFnTHlCZlVIUTdDaUFnSUNBZ0lDQWdJQ0FnSUM4dklGVndaR0YwWlNCd1pY'
    'SnBiMlFnWm05eUlHNWxlSFFnZEdsamF5QW9ZMnhoYlhCeklHMWhkR05vSUZC'
    'VUlISmhibWRsS1FvZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvWDNOMFpYQWdJVDBn'
    'TUNrZ1gxQjBJRDBnWTJ4aGJYQW9YMUIwSUNzZ1pteHZZWFFvWDNOMFpYQXBM'
    'Q0F4TVRNdU1Dd2dPRFUyTGpBcE93b2dJQ0FnSUNBZ0lIMEtJQ0FnSUgwS0Np'
    'QWdJQ0F2THlCR2IzSjNZWEprSUhOallXNDZJSEp2ZDNNZ1UxUlNTVU5VVEZr'
    'Z1ltVjBkMlZsYmlCMGNtbG5aMlZ5SUdGdVpDQmpkWEp5Wlc1MENpQWdJQ0Jw'
    'WmlBb2RISnBaMUJoZENBaFBTQndiM011YzI5dVoxQnZjeUI4ZkNCMGNtbG5V'
    'bTkzSUNFOUlIQnZjeTV5YjNjcElIc0tJQ0FnSUNBZ0lDQnBiblFnWDJad0lE'
    'MGdkSEpwWjFCaGRDd2dYMlp5SUQwZ2RISnBaMUp2ZHlBcklERTdDaUFnSUNB'
    'Z0lDQWdhV1lnS0Y5bWNpQStQU0J3WVhSVGRHRnlkRkp2ZDF0ZlpuQmRJQ3Nn'
    'S0hCaGRGSnZkMDltWm5ObGRGdGZabkFyTVYwZ0xTQndZWFJTYjNkUFptWnpa'
    'WFJiWDJad1hTa3BJSHNLSUNBZ0lDQWdJQ0FnSUNBZ1gyWndLeXM3SUY5bWNp'
    'QTlJQ2hmWm5BZ1BDQlRUMDVIWDB4RlRrZFVTQ2tnUHlCd1lYUlRkR0Z5ZEZK'
    'dmQxdGZabkJkSURvZ01Ec0tJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdabTl5'
    'SUNocGJuUWdYMlpwSUQwZ01Ec2dYMlpwSUR3Z01USTRPeUJmWm1rckt5a2dl'
    'd29nSUNBZ0lDQWdJQ0FnSUNCcFppQW9YMlp3SUQ0Z2NHOXpMbk52Ym1kUWIz'
    'TWdmSHdnS0Y5bWNDQTlQU0J3YjNNdWMyOXVaMUJ2Y3lBbUppQmZabklnUGow'
    'Z2NHOXpMbkp2ZHlrcElHSnlaV0ZyT3dvZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFv'
    'WDJad0lENDlJRk5QVGtkZlRFVk9SMVJJS1NCaWNtVmhhenNLSUNBZ0lDQWdJ'
    'Q0FnSUNBZ2FXWWdLRjltY2lBK1BTQndZWFJUZEdGeWRGSnZkMXRmWm5CZElD'
    'c2dLSEJoZEZKdmQwOW1abk5sZEZ0ZlpuQXJNVjBnTFNCd1lYUlNiM2RQWm1a'
    'elpYUmJYMlp3WFNrcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lGOW1jQ3Ny'
    'T3lCZlpuSWdQU0FvWDJad0lEd2dVMDlPUjE5TVJVNUhWRWdwSUQ4Z2NHRjBV'
    'M1JoY25SU2IzZGJYMlp3WFNBNklEQTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QmpiMjUwYVc1MVpUc0tJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNB'
    'Z0lDQk9iM1JsSUY5bWJpQTlJR2RsZEU1dmRHVW9YMlp3TENCZlpuSXNJR05v'
    'S1RzS0lDQWdJQ0FnSUNBZ0lDQWdMeThnUVU1WklISnZkeUIzYVhSb0lIQmxj'
    'bWx2WkNBK0lEQWdZVzVrSUdWbVptVmpkQ0J1YjNRZ015ODFJR2x6SUdFZ2Nt'
    'VmhiQ0JTUlZSU1NVZEhSVklLSUNBZ0lDQWdJQ0FnSUNBZ0x5OGdkR2hoZENC'
    'bGJtUnpJSFJvWlNCbWIzSjNZWEprSUhOallXNGdLRzVsZUhRZ1oyVjBRMmho'
    'Ym01bGJFOTFkSEIxZENCallXeHNJR2hoYm1Sc1pYTWdhWFFwTGdvZ0lDQWdJ'
    'Q0FnSUNBZ0lDQmliMjlzSUY5bWJrbHpWRzl1WlZSeWFXY2dQU0FvS0Y5bWJp'
    'NWxabVpsWTNRZ1BUMGdNSGd6SUh4OElGOW1iaTVsWm1abFkzUWdQVDBnTUhn'
    'MUtTQW1KaUJmWm00dWNHVnlhVzlrSUQ0Z01DazdDaUFnSUNBZ0lDQWdJQ0Fn'
    'SUdKdmIyd2dYMlp1U1hOU1pYUnlhV2NnSUNBOUlDaGZabTR1Y0dWeWFXOWtJ'
    'RDRnTUNBbUppQWhYMlp1U1hOVWIyNWxWSEpwWnlrN0NpQWdJQ0FnSUNBZ0lD'
    'QWdJR2xtSUNoZlptNUpjMUpsZEhKcFp5a2dZbkpsWVdzN0NpQWdJQ0FnSUNB'
    'Z0lDQWdJQzh2SUZSdmJtVXRjRzl5ZEdFZ2RHRnlaMlYwT2lCd2FYUmphQ0J6'
    'Ykdsa1pYTWdkRzkzWVhKa0lGOW1iaTV3WlhKcGIyUWdiM1psY2lCeVpXMWhh'
    'VzVwYm1jZ2NtOTNjeTRLSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjltYmtselZH'
    'OXVaVlJ5YVdjcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lIUmhjbWRsZEZC'
    'bGNtbHZaQ0E5SUdac2IyRjBLRjltYmk1d1pYSnBiMlFwT3dvZ0lDQWdJQ0Fn'
    'SUNBZ0lDQjlDaUFnSUNBZ0lDQWdJQ0FnSUM4dklGUnlZV05ySUd4aGMzUWdN'
    'M2g0SUhKaGRHVWdabTl5SUdWbVptVmpkQ0ExSUhSdklHbHVhR1Z5YVhRS0lD'
    'QWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5bWJpNWxabVpsWTNRZ1BUMGdNSGd6SUNZ'
    'bUlGOW1iaTV3WVhKaGJTQStJREFwSUY5c1lYTjBWRkJTWVhSbElEMGdYMlp1'
    'TG5CaGNtRnRPd29LSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjl6WjNJZ0lDQTlJ'
    'SEJoZEZScFkydFBabVp6WlhSYlgyWndYU0FySUNoZlpuSWdMU0J3WVhSVGRH'
    'RnlkRkp2ZDF0ZlpuQmRLVHNLSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjltZFd4'
    'c0lDQTlJR1psZEdOb1ZHbGpheWhmYzJkeUlDc2dNU2tnTFNCbVpYUmphRlJw'
    'WTJzb1gzTm5jaWtnTFNBeE95QWdMeThnZEdsamEzTWdNUzR1YzNCbFpXUXRN'
    'UW9nSUNBZ0lDQWdJQ0FnSUNCbWJHOWhkQ0JmY0ZOMFlYSjBVbTkzSUQwZ1pX'
    'Wm1aV04wYVhabFVHVnlhVzlrT3lBZ0x5OGdabTl5SUdaVFlXMXdiR1ZRYjNN'
    'Z2FXNTBaV2R5WVhScGIyNEtDaUFnSUNBZ0lDQWdJQ0FnSUM4dklGQnBkR05v'
    'SUdWbVptVmpkSE1LSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjltYmk1bFptWmxZ'
    'M1FnUFQwZ01IZ3hLUW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdaV1ptWldOMGFY'
    'WmxVR1Z5YVc5a0lEMGdiV0Y0S0RFeE15NHdMQ0JsWm1abFkzUnBkbVZRWlhK'
    'cGIyUWdMU0JtYkc5aGRDaGZabTR1Y0dGeVlXMGdLaUJmWm5Wc2JDa3BPd29n'
    'SUNBZ0lDQWdJQ0FnSUNCbGJITmxJR2xtSUNoZlptNHVaV1ptWldOMElEMDlJ'
    'REI0TWlrS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdWbVptVmpkR2wyWlZCbGNt'
    'bHZaQ0E5SUcxcGJpZzROVFl1TUN3Z1pXWm1aV04wYVhabFVHVnlhVzlrSUNz'
    'Z1pteHZZWFFvWDJadUxuQmhjbUZ0SUNvZ1gyWjFiR3dwS1RzS0lDQWdJQ0Fn'
    'SUNBZ0lDQWdaV3h6WlNCcFppQW9YMlp1TG1WbVptVmpkQ0E5UFNBd2VETXBJ'
    'SHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQzh2SUZSdmJtVWdjRzl5ZEdFZzRv'
    'Q1VJSFZ6WlhNZ2FYUnpJRzkzYmlCd1lYSmhiU0JoY3lCemJHbGtaU0J5WVhS'
    'bENpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9aV1ptWldOMGFYWmxVR1Z5'
    'YVc5a0lEd2dkR0Z5WjJWMFVHVnlhVzlrS1FvZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lHVm1abVZqZEdsMlpWQmxjbWx2WkNBOUlHMXBiaWgwWVhKblpY'
    'UlFaWEpwYjJRc0lHVm1abVZqZEdsMlpWQmxjbWx2WkNBcklHWnNiMkYwS0Y5'
    'bWJpNXdZWEpoYlNBcUlGOW1kV3hzS1NrN0NpQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNCbGJITmxJR2xtSUNobFptWmxZM1JwZG1WUVpYSnBiMlFnUGlCMFlYSm5a'
    'WFJRWlhKcGIyUXBDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnWldabVpX'
    'TjBhWFpsVUdWeWFXOWtJRDBnYldGNEtIUmhjbWRsZEZCbGNtbHZaQ3dnWlda'
    'bVpXTjBhWFpsVUdWeWFXOWtJQzBnWm14dllYUW9YMlp1TG5CaGNtRnRJQ29n'
    'WDJaMWJHd3BLVHNLSUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lDQWdJ'
    'Q0JsYkhObElHbG1JQ2hmWm00dVpXWm1aV04wSUQwOUlEQjROU2tnZXdvZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnTHk4Z1EyOXVkR2x1ZFdVZ2RHOXVaU0J3YjNK'
    'MFlTRGlnSlFnZFhObGN5Qk1RVk5VSURONGVDQnlZWFJsSUNodWIzUWdYMlp1'
    'TG5CaGNtRnRJU2tLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZmJHRnpk'
    'RlJRVW1GMFpTQStJREFwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QnBaaUFvWldabVpXTjBhWFpsVUdWeWFXOWtJRHdnZEdGeVoyVjBVR1Z5YVc5'
    'a0tRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCbFptWmxZM1Jw'
    'ZG1WUVpYSnBiMlFnUFNCdGFXNG9kR0Z5WjJWMFVHVnlhVzlrTENCbFptWmxZ'
    'M1JwZG1WUVpYSnBiMlFnS3lCbWJHOWhkQ2hmYkdGemRGUlFVbUYwWlNBcUlG'
    'OW1kV3hzS1NrN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdaV3h6WlNC'
    'cFppQW9aV1ptWldOMGFYWmxVR1Z5YVc5a0lENGdkR0Z5WjJWMFVHVnlhVzlr'
    'S1FvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JsWm1abFkzUnBk'
    'bVZRWlhKcGIyUWdQU0J0WVhnb2RHRnlaMlYwVUdWeWFXOWtMQ0JsWm1abFkz'
    'UnBkbVZRWlhKcGIyUWdMU0JtYkc5aGRDaGZiR0Z6ZEZSUVVtRjBaU0FxSUY5'
    'bWRXeHNLU2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0Fn'
    'SUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnTHk4Z1ZtOXNkVzFsSUdWbVptVmpk'
    'SE1nS0hSeVlXNXphWFJwYjI0Z2MzUmhiWEJsWkNCaGRDQjBhR2x6SUhKdmR5'
    'ZHpJSE4wWVhKMElIUnBZMnNwQ2lBZ0lDQWdJQ0FnSUNBZ0lHWnNiMkYwSUY5'
    'bWJsUnBZMnRHSUQwZ1pteHZZWFFvWm1WMFkyaFVhV05yS0Y5elozSXBLVHNL'
    'SUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjltYmk1bFptWmxZM1FnUFQwZ01IaERL'
    'UW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdWazlNWDFORlZDaHRhVzRvWDJadUxu'
    'QmhjbUZ0TENBMk5Da3NJRjltYmxScFkydEdLVHNLSUNBZ0lDQWdJQ0FnSUNB'
    'Z1pXeHpaU0JwWmlBb1gyWnVMbVZtWm1WamRDQTlQU0F3ZUVFZ2ZId2dYMlp1'
    'TG1WbVptVmpkQ0E5UFNBd2VEWXBJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'R2x1ZENCZmRuVWdQU0FvWDJadUxuQmhjbUZ0UGo0MEtTWXdlRVlzSUY5MlpD'
    'QTlJRjltYmk1d1lYSmhiU1l3ZUVZN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNC'
    'V1QweGZVMFZVS0dOc1lXMXdLRjkyYjJ4RGRYSnlJQ3NnS0Y5MmRUNHdQMTky'
    'ZFRvdFgzWmtLU0FxSUY5bWRXeHNMQ0F3TENBMk5Da3NJRjltYmxScFkydEdL'
    'VHNLSUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0JsYkhObElH'
    'bG1JQ2hmWm00dVpXWm1aV04wSUQwOUlEQjRSU2tnZXdvZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0FnTHk4Z1JVRjRJR1pwYm1VZ2RtOXNJSFZ3TENCRlFuZ2dabWx1'
    'WlNCMmIyd2daRzkzYmlBb2FXNXpkR0Z1ZENCd1pYSWdjbTkzS1FvZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnYVc1MElGOWxjeUE5SUNoZlptNHVjR0Z5WVcwZ1Bq'
    'NGdOQ2tnSmlBd2VFWTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBiblFnWDJW'
    'MklEMGdJRjltYmk1d1lYSmhiU0FnSUNBZ0lDQW1JREI0UmpzS0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUdsbUlDaGZaWE1nUFQwZ01IaEJLU0FnSUNBZ0lGWlBU'
    'RjlUUlZRb1kyeGhiWEFvWDNadmJFTjFjbklnS3lCZlpYWXNJREFzSURZMEtT'
    'd2dYMlp1VkdsamEwWXBPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdaV3h6WlNC'
    'cFppQW9YMlZ6SUQwOUlEQjRRaWtnVms5TVgxTkZWQ2hqYkdGdGNDaGZkbTlz'
    'UTNWeWNpQXRJRjlsZGl3Z01Dd2dOalFwTENCZlptNVVhV05yUmlrN0NpQWdJ'
    'Q0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ0x5OGdNSGcxSUdGc2My'
    'OGdZWEJ3YkdsbGN5QjBhR1VnZG05c2RXMWxJSE5zYVdSbElIQnZjblJwYjI0'
    'Z0tHaHBaMmdnYm1saVlteGxJRDBnZFhBc0lHeHZkeUE5SUdSdmQyNHBDaUFn'
    'SUNBZ0lDQWdJQ0FnSUdsbUlDaGZabTR1WldabVpXTjBJRDA5SURCNE5Ta2dl'
    'd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5MmRTQTlJQ2hmWm00dWNH'
    'RnlZVzArUGpRcEpqQjRSaXdnWDNaa0lEMGdYMlp1TG5CaGNtRnRKakI0Umpz'
    'S0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUZaUFRGOVRSVlFvWTJ4aGJYQW9YM1p2'
    'YkVOMWNuSWdLeUFvWDNaMVBqQS9YM1oxT2kxZmRtUXBJQ29nWDJaMWJHd3NJ'
    'REFzSURZMEtTd2dYMlp1VkdsamEwWXBPd29nSUNBZ0lDQWdJQ0FnSUNCOUNp'
    'QWdJQ0FnSUNBZ0lDQWdJQzh2SUZCbGNpMTBhV05ySUdaVFlXMXdiR1ZRYjNN'
    'Z2FXNTBaV2R5WVhScGIyNGdabTl5SUhSb2FYTWdjbTkzSUNodFlYUmphR1Z6'
    'SUZCVUlITmxiV0Z1ZEdsamN3b2dJQ0FnSUNBZ0lDQWdJQ0F2THlCbGVHRmpk'
    'R3g1SU9LQWxDQmthWE5qY21WMFpTQjBhV05yTFdKNUxYUnBZMnNzSUc1dmRD'
    'QnNhVzVsWVhJdGNtRnRjQ0JqYjI1MGFXNTFiM1Z6S1M0S0lDQWdJQ0FnSUNB'
    'Z0lDQWdMeThnVUdWeWFXOWtJR0YwSUhScFkyc2dNQ0J2WmlCMGFHbHpJSEp2'
    'ZHlBOUlGOXdVM1JoY25SU2IzY3VJRlJwWTJ0eklERXVMbVoxYkd3Z1lYQndi'
    'SGtLSUNBZ0lDQWdJQ0FnSUNBZ0x5OGdkR2hsSUhOc2FXUmxJSE4wWlhBdUlG'
    'ZGxJSEpsTFdSbGNtbDJaU0IwYUdVZ2NHVnlMWFJwWTJzZ2MzUmxjQ0JtY205'
    'dElIUm9aU0J5YjNjbmN3b2dJQ0FnSUNBZ0lDQWdJQ0F2THlCbFptWmxZM1Fn'
    'Y21GMGFHVnlJSFJvWVc0Z1pHbDJhV1JwYm1jZ0tHVm1abVZqZEdsMlpWQmxj'
    'bWx2WkNBdElGOXdVM1JoY25SU2IzY3BMMTltZFd4c0NpQWdJQ0FnSUNBZ0lD'
    'QWdJQzh2SUhOcGJtTmxJRE40ZUM4MWVIZ2dZMnhoYlhCcGJtY2dZWFFnZEdG'
    'eVoyVjBJRzFoYTJWeklIUm9ZWFFnWVhabGNtRm5aU0J0YVhOc1pXRmthVzVu'
    'TGdvZ0lDQWdJQ0FnSUNBZ0lDQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQm1i'
    'RzloZENCZlEyWWdJRDBnWXpSemNHVmxaSE5iYzIxd0xtWnBibVYwZFc1bElD'
    'WWdNSGhHWFNBcUlEUXlPQzR3T3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnWm14'
    'dllYUWdYMlIwSUNBOUlERXVNQ0F2SUZSSlEwdFRYMUJGVWw5VFJVTTdDaUFn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQm1iRzloZENCZlVIUWdJRDBnWDNCVGRHRnlk'
    'Rkp2ZHpzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUM4dklFUmxkR1Z5YldsdVpT'
    'QndaWEl0ZEdsamF5QnpkR1Z3SUNoemFXZHVaV1FwSUdGdVpDQjBZWEpuWlhR'
    'Z1ptOXlJR05zWVcxd2FXNW5MZ29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhVzUw'
    'SUY5emRHVndJRDBnTURzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdKdmIyd2dY'
    'Mk5zWVcxd1ZHOVVaM1FnUFNCbVlXeHpaVHNLSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJR2xtSUNoZlptNHVaV1ptWldOMElEMDlJREI0TVNrZ1gzTjBaWEFnUFNB'
    'dFgyWnVMbkJoY21GdE93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1pXeHpaU0Jw'
    'WmlBb1gyWnVMbVZtWm1WamRDQTlQU0F3ZURJcElGOXpkR1Z3SUQwZ1gyWnVM'
    'bkJoY21GdE93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1pXeHpaU0JwWmlBb1gy'
    'WnVMbVZtWm1WamRDQTlQU0F3ZURNcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0JmWTJ4aGJYQlViMVJuZENBOUlIUnlkV1U3Q2lBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjl3VTNSaGNuUlNiM2NnUENCMFlYSm5a'
    'WFJRWlhKcGIyUXBJQ0FnSUNBZ1gzTjBaWEFnUFNCZlptNHVjR0Z5WVcwN0Np'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdaV3h6WlNCcFppQW9YM0JUZEdG'
    'eWRGSnZkeUErSUhSaGNtZGxkRkJsY21sdlpDa2dYM04wWlhBZ1BTQXRYMlp1'
    'TG5CaGNtRnRPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdmUW9nSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdaV3h6WlNCcFppQW9YMlp1TG1WbVptVmpkQ0E5UFNBd2VE'
    'VXBJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCZlkyeGhiWEJVYjFS'
    'bmRDQTlJSFJ5ZFdVN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhV1ln'
    'S0Y5d1UzUmhjblJTYjNjZ1BDQjBZWEpuWlhSUVpYSnBiMlFwSUNBZ0lDQWdY'
    'M04wWlhBZ1BTQmZiR0Z6ZEZSUVVtRjBaVHNLSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNCbGJITmxJR2xtSUNoZmNGTjBZWEowVW05M0lENGdkR0Z5WjJW'
    'MFVHVnlhVzlrS1NCZmMzUmxjQ0E5SUMxZmJHRnpkRlJRVW1GMFpUc0tJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDOHZJ'
    'RjltZFd4c0lDc2dNU0E5SUhSdmRHRnNJSFJwWTJ0eklHbHVJSFJvYVhNZ2Nt'
    'OTNJQ2h6Y0dWbFpDa0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0JmZEds'
    'amEzTmZhVzVmY205M0lEMGdYMloxYkd3Z0t5QXhPd29nSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdabTl5SUNocGJuUWdYM1FnUFNBd095QmZkQ0E4SURNeU95QmZk'
    'Q3NyS1NCN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5MElE'
    'NDlJRjkwYVdOcmMxOXBibDl5YjNjcElHSnlaV0ZyT3dvZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmVUhRZ1BpQXdMakFwSUY5bVUyRnRjR3hs'
    'VUc5elFXTmpJQ3M5SUY5RFppQXFJRjlrZENBdklGOVFkRHNLSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBdkx5QlZjR1JoZEdVZ2NHVnlhVzlrSUdadmNp'
    'QnVaWGgwSUhScFkyc2dLRzl1YkhrZ2FXWWdkQ0E4SUY5bWRXeHNMQ0JwTG1V'
    'dUlIUm9aWEpsSUdGeVpRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQzh2'
    'SUcxdmNtVWdkR2xqYTNNZ2RHOGdjMnhwWkdVZ2FXNGdkR2hwY3lCeWIzY3BD'
    'aUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOXpkR1Z3SUNFOUlE'
    'QWdKaVlnWDNRZ1BDQmZablZzYkNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNCbWJHOWhkQ0JmVUc0Z1BTQmZVSFFnS3lCbWJHOWhkQ2hm'
    'YzNSbGNDazdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1J'
    'Q2hmWTJ4aGJYQlViMVJuZENrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5emRHVndJRDRnTUNrZ0lDQWdJQ0JmVUc0'
    'Z1BTQnRhVzRvWDFCdUxDQjBZWEpuWlhSUVpYSnBiMlFwT3dvZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1pXeHpaU0JwWmlBb1gzTjBa'
    'WEFnUENBd0tTQmZVRzRnUFNCdFlYZ29YMUJ1TENCMFlYSm5aWFJRWlhKcGIy'
    'UXBPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQjlJR1ZzYzJV'
    'Z2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdYMUJ1'
    'SUQwZ1kyeGhiWEFvWDFCdUxDQXhNVE11TUN3Z09EVTJMakFwT3dvZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJRjlRZENBOUlGOVFianNLSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCOUNpQWdJQ0Fn'
    'SUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ1gyWnlLeXM3Q2lBZ0lDQWdJ'
    'Q0FnSUNBZ0lDOHZJRUZrZG1GdVkyVWdkRzhnYm1WNGRDQnpiMjVuSUhCdmMy'
    'bDBhVzl1SUhkb1pXNGdkMlVuZG1VZ1pYaG9ZWFZ6ZEdWa0lIUm9hWE1nY0dG'
    'MGRHVnliaWR6SUhKdmQzTUtJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOW1jaUEr'
    'UFNCd1lYUlRkR0Z5ZEZKdmQxdGZabkJkSUNzZ0tIQmhkRkp2ZDA5bVpuTmxk'
    'RnRmWm5Bck1WMGdMU0J3WVhSU2IzZFBabVp6WlhSYlgyWndYU2twSUhzS0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUY5bWNDc3JPd29nSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdYMlp5SUQwZ0tGOW1jQ0E4SUZOUFRrZGZURVZPUjFSSUtTQS9JSEJo'
    'ZEZOMFlYSjBVbTkzVzE5bWNGMGdPaUF3T3dvZ0lDQWdJQ0FnSUNBZ0lDQjlD'
    'aUFnSUNBZ0lDQWdmUW9LSUNBZ0lDQWdJQ0F2THlCRGRYSnlaVzUwSUhKdmR5'
    'QndZWEowYVdGc0lDaHViMjR0ZEhKcFoyZGxjaUJ5YjNjZ2IyNXNlU0RpZ0pR'
    'Z2RISnBaMmRsY2lCb1lXNWtiR1ZrSUdGaWIzWmxLUW9nSUNBZ0lDQWdJR2xt'
    'SUNoZmNHTnlMbWx1YzNSeWRXMWxiblFnUEQwZ01DQW1KaUJmY0dOeUxuQmxj'
    'bWx2WkNBOFBTQXdLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDOHZJRlJwWTJzZ2Mz'
    'UmhiWEFnWm05eUlHTjFjbkpsYm5RdGNtOTNJSEJoY25ScFlXd2dkbTlzSUdW'
    'bVptVmpkSE02SUhSb1pTQmpkWEp5Wlc1MElISnZkeWR6Q2lBZ0lDQWdJQ0Fn'
    'SUNBZ0lDOHZJSE4wWVhKMElIUnBZMnN1SUZSb1pTQTJOQzF6WVcxd2JHVWdj'
    'bUZ0Y0NCamIyMXdiR1YwWlhNZ2QyVnNiQ0IzYVhSb2FXNGdkR2hsSUdacGNu'
    'TjBDaUFnSUNBZ0lDQWdJQ0FnSUM4dklIUnBZMnNzSUhOdklHRnVlU0IyYjJ3'
    'Z1kyaGhibWRsSUNKb1lYQndaVzVwYm1jZ1lYUWdkR2hwY3lCeWIzY2lJSEps'
    'WVdSeklHRnpJR2hoZG1sdVp3b2dJQ0FnSUNBZ0lDQWdJQ0F2THlCeVlXMXda'
    'V1FnZEc4Z2FYUnpJR1pwYm1Gc0lIWmhiSFZsSUdGc2JXOXpkQ0JwYlcxbFpH'
    'bGhkR1ZzZVM0S0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5amRYSlRaM0lnUFNC'
    'd1lYUlVhV05yVDJabWMyVjBXM0J2Y3k1emIyNW5VRzl6WFNBcklDaHdiM011'
    'Y205M0lDMGdjR0YwVTNSaGNuUlNiM2RiY0c5ekxuTnZibWRRYjNOZEtUc0tJ'
    'Q0FnSUNBZ0lDQWdJQ0FnWm14dllYUWdYMk4xY2xScFkydEdJRDBnWm14dllY'
    'UW9abVYwWTJoVWFXTnJLRjlqZFhKVFozSXBLVHNLSUNBZ0lDQWdJQ0FnSUNB'
    'Z2FXWWdLRjl3WTNJdVpXWm1aV04wSUQwOUlEQjRReWtLSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJRlpQVEY5VFJWUW9iV2x1S0Y5d1kzSXVjR0Z5WVcwc0lEWTBL'
    'U3dnWDJOMWNsUnBZMnRHS1RzS0lDQWdJQ0FnSUNBZ0lDQWdaV3h6WlNCcFpp'
    'QW9YM0JqY2k1bFptWmxZM1FnUFQwZ01IaEJJSHg4SUY5d1kzSXVaV1ptWldO'
    'MElEMDlJREI0TmlrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjky'
    'ZFNBOUlDaGZjR055TG5CaGNtRnRQajQwS1NZd2VFWXNJRjkyWkNBOUlGOXdZ'
    'M0l1Y0dGeVlXMG1NSGhHT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnVms5TVgx'
    'TkZWQ2hqYkdGdGNDaGZkbTlzUTNWeWNpQXJJQ2hmZG5VK01EOWZkblU2TFY5'
    'MlpDa2dLaUJmY0dOMExDQXdMQ0EyTkNrc0lGOWpkWEpVYVdOclJpazdDaUFn'
    'SUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lDQWdaV3h6WlNCcFppQW9Y'
    'M0JqY2k1bFptWmxZM1FnUFQwZ01IaEZLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0JwYm5RZ1gyVnpJRDBnS0Y5d1kzSXVjR0Z5WVcwZ1BqNGdOQ2tnSmlB'
    'd2VFWTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBiblFnWDJWMklEMGdJRjl3'
    'WTNJdWNHRnlZVzBnSUNBZ0lDQWdKaUF3ZUVZN0NpQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNCcFppQW9YMlZ6SUQwOUlEQjRRU2tnSUNBZ0lDQldUMHhmVTBWVUtH'
    'TnNZVzF3S0Y5MmIyeERkWEp5SUNzZ1gyVjJMQ0F3TENBMk5Da3NJRjlqZFhK'
    'VWFXTnJSaWs3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JsYkhObElHbG1JQ2hm'
    'WlhNZ1BUMGdNSGhDS1NCV1QweGZVMFZVS0dOc1lXMXdLRjkyYjJ4RGRYSnlJ'
    'QzBnWDJWMkxDQXdMQ0EyTkNrc0lGOWpkWEpVYVdOclJpazdDaUFnSUNBZ0lD'
    'QWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5d1kzSXVaV1ptWldO'
    'MElEMDlJREI0TlNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0x5OGdNSGcx'
    'SUhadmJDMXpiR2xrWlNCd2IzSjBhVzl1SUc5dUlHTjFjbkpsYm5RZ2NtOTND'
    'aUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBiblFnWDNaMUlEMGdLRjl3WTNJdWNH'
    'RnlZVzArUGpRcEpqQjRSaXdnWDNaa0lEMGdYM0JqY2k1d1lYSmhiU1l3ZUVZ'
    'N0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCV1QweGZVMFZVS0dOc1lXMXdLRjky'
    'YjJ4RGRYSnlJQ3NnS0Y5MmRUNHdQMTkyZFRvdFgzWmtLU0FxSUY5d1kzUXNJ'
    'REFzSURZMEtTd2dYMk4xY2xScFkydEdLVHNLSUNBZ0lDQWdJQ0FnSUNBZ2ZR'
    'b2dJQ0FnSUNBZ0lIMEtJQ0FnSUgwS0NpQWdJQ0F2THlCRGRYSnlaVzUwSUhK'
    'dmR5QndZWEowYVdGc0lIQnBkR05vSUdWbVptVmpkQ0FvWVhCd2JHbGxjeUJs'
    'ZG1WdUlHOXVJSFJ5YVdkblpYSWdjbTkzS1M0S0lDQWdJQzh2SUZWelpTQmpi'
    'MjUwYVc1MWIzVnpJSEJ2Y3k1MGFXTnJJT0tBbENCaWRYUWdZMkZ3SUdsMElH'
    'RjBJQ2h6Y0dWbFpDMHhLU0J6YnlCMGFHVWdZMjl1ZEhKcFluVjBhVzl1Q2lB'
    'Z0lDQXZMeUJoZENCMGFHVWdiR0Z6ZENCellXMXdiR1VnYjJZZ2RHaHBjeUJ5'
    'YjNjZ1pYaGhZM1JzZVNCdFlYUmphR1Z6SUhkb1lYUWdkR2hsSUdadmNuZGhj'
    'bVFnYzJOaGJnb2dJQ0FnTHk4Z2QybHNiQ0IxYzJVZ1ptOXlJSFJvYVhNZ2Nt'
    'OTNJRzl1WTJVZ2FYUWdZbVZqYjIxbGN5QmhJQ0pqYjIxd2JHVjBaV1FpSUhK'
    'dmR5NGdJRmRwZEdodmRYUWdkR2hsQ2lBZ0lDQXZMeUJqWVhBc0lIQnZjeTUw'
    'YVdOcklHRndjSEp2WVdOb1pYTWdZSE53WldWa1lDQmhkQ0IwYUdVZ2NtOTNJ'
    'R0p2ZFc1a1lYSjVJSGRvYVd4bElIUm9aU0JtYjNKM1lYSmtDaUFnSUNBdkx5'
    'QnpZMkZ1SUhWelpYTWdZSE53WldWa0xURmdMQ0J3Y205a2RXTnBibWNnWVNC'
    'K01TMTBhV05ySUdKaFkydDNZWEprSUhCbGNtbHZaQ0JxZFcxd0lEMGdZMnhw'
    'WTJzdUNpQWdJQ0F2THlCUGJteDVJSEJoZVNCMGFHVWdabVYwWTJoVWFXTnJJ'
    'R052YzNRZ2QyaGxiaUJoSUhCcGRHTm9JR1ZtWm1WamRDQnBjeUJoWTNSMVlX'
    'eHNlU0J3Y21WelpXNTBMZ29nSUNBZ0x5OGdVMkYyWlNCbFptWmxZM1JwZG1W'
    'UVpYSnBiMlFnWVhRZ2MzUmhjblFnYjJZZ1kzVnljbVZ1ZENCeWIzY3NJRUpG'
    'Ums5U1JTQndZWEowYVdGc0lIQnBkR05vQ2lBZ0lDQXZMeUJsWm1abFkzUWdZ'
    'WEJ3YkdsallYUnBiMjRnNG9DVUlHNWxaV1JsWkNCbWIzSWdkR2hsSUdOMWNu'
    'SmxiblF0Y205M0lHaGxZV1FnWTI5dWRISnBZblYwYVc5dUlIUnZDaUFnSUNB'
    'dkx5QmZabE5oYlhCc1pWQnZjMEZqWXlCaVpXeHZkeTRLSUNBZ0lHWnNiMkYw'
    'SUY5d1UzUmhjblJEZFhJZ1BTQmxabVpsWTNScGRtVlFaWEpwYjJRN0Nnb2dJ'
    'Q0FnTHk4Z1EzVnljbVZ1ZENCeWIzY2djR0Z5ZEdsaGJDQndhWFJqYUNCbFpt'
    'WmxZM1FnS0dGd2NHeHBaWE1nYjI0Z2RISnBaMmRsY2lCeWIzY2dUMUlnWTI5'
    'dWRHbHVkV0YwYVc5dUlISnZkeWt1Q2lBZ0lDQXZMeUJHYjNJZ01YaDRMeko0'
    'ZURvZ1lXeDNZWGx6SUdGd2NHeHBaWE1nYjI0Z1kzVnljbVZ1ZENCeWIzY3VD'
    'aUFnSUNBdkx5QkdiM0lnTTNoNEx6VjRlRG9nWVhCd2JHbGxjeUIzYUdWdVpY'
    'WmxjaUJqZFhKeVpXNTBJSEp2ZHlCallYSnlhV1Z6SUhSb1pTQmxabVpsWTNR'
    'ZzRvQ1VJR0p2ZEdnS0lDQWdJQzh2SUNBZ1kyOXVkR2x1ZFdGMGFXOXVJSEp2'
    'ZDNNZ0tIQmxjbWx2WkQwOU1Da2dRVTVFSUhSdmJtVXRjRzl5ZEdFZ2RHRnla'
    'MlYwSUhKdmQzTWdLSEJsY21sdlpENHdLUzRLSUNBZ0lDOHZJQ0FnVkdobElI'
    'UmhjbWRsZEZCbGNtbHZaQ0IzWVhNZ1lXeHlaV0ZrZVNCelpYUWdZV0p2ZG1V'
    'Z0tHVnBkR2hsY2lCbWNtOXRJSFJ5YVdkT2IzUmxJRzl5Q2lBZ0lDQXZMeUFn'
    'SUhSdmJtVlRiR2xrWlZSaGNtZGxkQ2s3SUhSb2FYTWdZbXh2WTJzZ1pHOWxj'
    'eUIwYUdVZ2NHVnlMWFJwWTJzZ1lXTmpkVzExYkdGMGFXOXVJSFJ2ZDJGeVpD'
    'QnBkQzRLSUNBZ0lHbG1JQ2hmY0dOeUxtVm1abVZqZENBOVBTQXdlREVnZkh3'
    'Z1gzQmpjaTVsWm1abFkzUWdQVDBnTUhneUlIeDhDaUFnSUNBZ0lDQWdYM0Jq'
    'Y2k1bFptWmxZM1FnUFQwZ01IZ3pJSHg4SUY5d1kzSXVaV1ptWldOMElEMDlJ'
    'REI0TlNrZ2V3b2dJQ0FnSUNBZ0lHbHVkQ0JmYzJkeVgyTjFjaUE5SUhCaGRG'
    'UnBZMnRQWm1aelpYUmJjRzl6TG5OdmJtZFFiM05kSUNzZ0tIQnZjeTV5YjNj'
    'Z0xTQndZWFJUZEdGeWRGSnZkMXR3YjNNdWMyOXVaMUJ2YzEwcE93b2dJQ0Fn'
    'SUNBZ0lHWnNiMkYwSUY5d2RHWWdQU0J0YVc0b2NHOXpMblJwWTJzc0NpQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCbWJHOWhkQ2htWlhSamFG'
    'UnBZMnNvWDNObmNsOWpkWElnS3lBeEtTQXRJR1psZEdOb1ZHbGpheWhmYzJk'
    'eVgyTjFjaWtnTFNBeEtTazdDaUFnSUNBZ0lDQWdhV1lnS0Y5d1kzSXVaV1pt'
    'WldOMElEMDlJREI0TVNrS0lDQWdJQ0FnSUNBZ0lDQWdaV1ptWldOMGFYWmxV'
    'R1Z5YVc5a0lEMGdiV0Y0S0RFeE15NHdMQ0JsWm1abFkzUnBkbVZRWlhKcGIy'
    'UWdMU0JtYkc5aGRDaGZjR055TG5CaGNtRnRLU0FxSUY5d2RHWXBPd29nSUNB'
    'Z0lDQWdJR1ZzYzJVZ2FXWWdLRjl3WTNJdVpXWm1aV04wSUQwOUlEQjRNaWtL'
    'SUNBZ0lDQWdJQ0FnSUNBZ1pXWm1aV04wYVhabFVHVnlhVzlrSUQwZ2JXbHVL'
    'RGcxTmk0d0xDQmxabVpsWTNScGRtVlFaWEpwYjJRZ0t5Qm1iRzloZENoZmNH'
    'TnlMbkJoY21GdEtTQXFJRjl3ZEdZcE93b2dJQ0FnSUNBZ0lHVnNjMlVnYVdZ'
    'Z0tGOXdZM0l1WldabVpXTjBJRDA5SURCNE15a2dld29nSUNBZ0lDQWdJQ0Fn'
    'SUNBdkx5QlViMjVsSUhCdmNuUmhJT0tBbENCMWMyVnpJR2wwY3lCdmQyNGdj'
    'R0Z5WVcwZ1lYTWdjMnhwWkdVZ2NtRjBaUW9nSUNBZ0lDQWdJQ0FnSUNCcFpp'
    'QW9aV1ptWldOMGFYWmxVR1Z5YVc5a0lEd2dkR0Z5WjJWMFVHVnlhVzlrS1Fv'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnWldabVpXTjBhWFpsVUdWeWFXOWtJRDBn'
    'YldsdUtIUmhjbWRsZEZCbGNtbHZaQ3dnWldabVpXTjBhWFpsVUdWeWFXOWtJ'
    'Q3NnWm14dllYUW9YM0JqY2k1d1lYSmhiU2tnS2lCZmNIUm1LVHNLSUNBZ0lD'
    'QWdJQ0FnSUNBZ1pXeHpaU0JwWmlBb1pXWm1aV04wYVhabFVHVnlhVzlrSUQ0'
    'Z2RHRnlaMlYwVUdWeWFXOWtLUW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdaV1pt'
    'WldOMGFYWmxVR1Z5YVc5a0lEMGdiV0Y0S0hSaGNtZGxkRkJsY21sdlpDd2da'
    'V1ptWldOMGFYWmxVR1Z5YVc5a0lDMGdabXh2WVhRb1gzQmpjaTV3WVhKaGJT'
    'a2dLaUJmY0hSbUtUc0tJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdaV3h6WlNC'
    'N0lDQXZMeUF3ZURVZzRvQ1VJR052Ym5ScGJuVmxJSFJ2Ym1VZ2NHOXlkR0Vn'
    'ZFhOcGJtY2diR0Z6ZENBemVIZ2djbUYwWlNBb2NHRnlZVzBnYVhNZ2RtOXNM'
    'WE5zYVdSbElHOXViSGtwQ2lBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmYkdGemRG'
    'UlFVbUYwWlNBK0lEQXBJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2xtSUNo'
    'bFptWmxZM1JwZG1WUVpYSnBiMlFnUENCMFlYSm5aWFJRWlhKcGIyUXBDaUFn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnWldabVpXTjBhWFpsVUdWeWFXOWtJ'
    'RDBnYldsdUtIUmhjbWRsZEZCbGNtbHZaQ3dnWldabVpXTjBhWFpsVUdWeWFX'
    'OWtJQ3NnWm14dllYUW9YMnhoYzNSVVVGSmhkR1VwSUNvZ1gzQjBaaWs3Q2lB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0JsYkhObElHbG1JQ2hsWm1abFkzUnBkbVZR'
    'WlhKcGIyUWdQaUIwWVhKblpYUlFaWEpwYjJRcENpQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdaV1ptWldOMGFYWmxVR1Z5YVc5a0lEMGdiV0Y0S0hSaGNt'
    'ZGxkRkJsY21sdlpDd2daV1ptWldOMGFYWmxVR1Z5YVc5a0lDMGdabXh2WVhR'
    'b1gyeGhjM1JVVUZKaGRHVXBJQ29nWDNCMFppazdDaUFnSUNBZ0lDQWdJQ0Fn'
    'SUgwS0lDQWdJQ0FnSUNCOUNpQWdJQ0I5Q2dvZ0lDQWdMeThnNHBTQTRwU0E0'
    'cFNBSUVOMWNuSmxiblF0Y205M0lHaGxZV1FnWTI5dWRISnBZblYwYVc5dUlI'
    'WnBZU0J3WlhJdGRHbGpheUJwYm5SbFozSmhkR2x2YmlEaWxJRGlsSURpbElE'
    'aWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSUFLSUNBZ0lDOHZJRk5o'
    'YldVZ2NHVnlMWFJwWTJzZ2JXOWtaV3dnWVhNZ1ptOXlkMkZ5WkNCelkyRnVM'
    'aUJYWlNkeVpTQmhkQ0J3YjNNdWRHbGpheUIzYVhSb2FXNGdkR2hsQ2lBZ0lD'
    'QXZMeUJqZFhKeVpXNTBJSEp2ZHk0Z1VHVnlhVzlrSUdGMElIUnBZMnNnTUNB'
    'OUlGOXdVM1JoY25SRGRYSXVJRmRsSUdsdWRHVm5jbUYwWlNCMGFXTnJjd29n'
    'SUNBZ0x5OGdXekFzSUhCdmN5NTBhV05yS1NEaWdKUWdhUzVsTGl3Z2QyVW5j'
    'bVVnSW1KbFptOXlaU0lnZEdobElHSnZkVzVrWVhKNUlHRjBJR1Z1WkNCdlpp'
    'QjBhV05ySUdac2IyOXlLSEJ2Y3k1MGFXTnJLUzRLSUNBZ0lDOHZJRVp2Y2lC'
    'MGFHVWdabkpoWTNScGIyNWhiQ0J6ZFdJdGRHbGpheUFvWW1WMGQyVmxiaUJw'
    'Ym5SbFoyVnlJSFJwWTJzZ1pXUm5aWE1wTENCaFpHUUtJQ0FnSUM4dklIQmhj'
    'blJwWVd3dGRHbGpheUJqYjI1MGNtbGlkWFJwYjI0Z1lYUWdkR2hsSUhCbGNt'
    'bHZaQ0JqZFhKeVpXNTBiSGtnYVc0Z1pXWm1aV04wTGdvZ0lDQWdld29nSUNB'
    'Z0lDQWdJR1pzYjJGMElGOURabDlvSUQwZ1l6UnpjR1ZsWkhOYmMyMXdMbVpw'
    'Ym1WMGRXNWxJQ1lnTUhoR1hTQXFJRFF5T0M0d093b2dJQ0FnSUNBZ0lHWnNi'
    'MkYwSUY5a2RDQWdJRDBnTVM0d0lDOGdWRWxEUzFOZlVFVlNYMU5GUXpzS0lD'
    'QWdJQ0FnSUNCbWJHOWhkQ0JmVUhRZ0lDQTlJRjl3VTNSaGNuUkRkWEk3Q2lB'
    'Z0lDQWdJQ0FnYVc1MElGOXpkR1Z3SUQwZ01Ec0tJQ0FnSUNBZ0lDQmliMjlz'
    'SUY5amJHRnRjRlJ2VkdkMElEMGdabUZzYzJVN0NpQWdJQ0FnSUNBZ2FXWWdL'
    'Rjl3WTNJdVpXWm1aV04wSUQwOUlEQjRNU2tnWDNOMFpYQWdQU0F0WDNCamNp'
    'NXdZWEpoYlRzS0lDQWdJQ0FnSUNCbGJITmxJR2xtSUNoZmNHTnlMbVZtWm1W'
    'amRDQTlQU0F3ZURJcElGOXpkR1Z3SUQwZ1gzQmpjaTV3WVhKaGJUc0tJQ0Fn'
    'SUNBZ0lDQmxiSE5sSUdsbUlDaGZjR055TG1WbVptVmpkQ0E5UFNBd2VETXBJ'
    'SHNLSUNBZ0lDQWdJQ0FnSUNBZ1gyTnNZVzF3Vkc5VVozUWdQU0IwY25WbE93'
    'b2dJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1gzQlRkR0Z5ZEVOMWNpQThJSFJoY21k'
    'bGRGQmxjbWx2WkNrZ0lDQWdJQ0JmYzNSbGNDQTlJRjl3WTNJdWNHRnlZVzA3'
    'Q2lBZ0lDQWdJQ0FnSUNBZ0lHVnNjMlVnYVdZZ0tGOXdVM1JoY25SRGRYSWdQ'
    'aUIwWVhKblpYUlFaWEpwYjJRcElGOXpkR1Z3SUQwZ0xWOXdZM0l1Y0dGeVlX'
    'MDdDaUFnSUNBZ0lDQWdmUW9nSUNBZ0lDQWdJR1ZzYzJVZ2FXWWdLRjl3WTNJ'
    'dVpXWm1aV04wSUQwOUlEQjROU2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQmZZMnho'
    'YlhCVWIxUm5kQ0E5SUhSeWRXVTdDaUFnSUNBZ0lDQWdJQ0FnSUdsbUlDaGZj'
    'Rk4wWVhKMFEzVnlJRHdnZEdGeVoyVjBVR1Z5YVc5a0tTQWdJQ0FnSUY5emRH'
    'VndJRDBnWDJ4aGMzUlVVRkpoZEdVN0NpQWdJQ0FnSUNBZ0lDQWdJR1ZzYzJV'
    'Z2FXWWdLRjl3VTNSaGNuUkRkWElnUGlCMFlYSm5aWFJRWlhKcGIyUXBJRjl6'
    'ZEdWd0lEMGdMVjlzWVhOMFZGQlNZWFJsT3dvZ0lDQWdJQ0FnSUgwS0lDQWdJ'
    'Q0FnSUNCcGJuUWdYMloxYkd4ZmRHbGphM01nUFNCcGJuUW9jRzl6TG5ScFky'
    'c3BPd29nSUNBZ0lDQWdJR1pzYjJGMElGOW1jbUZqSUNBZ0lDQTlJSEJ2Y3k1'
    'MGFXTnJJQzBnWm14dllYUW9YMloxYkd4ZmRHbGphM01wT3dvZ0lDQWdJQ0Fn'
    'SUdadmNpQW9hVzUwSUY5MElEMGdNRHNnWDNRZ1BDQXpNanNnWDNRckt5a2dl'
    'd29nSUNBZ0lDQWdJQ0FnSUNCcFppQW9YM1FnUGowZ1gyWjFiR3hmZEdsamEz'
    'TXBJR0p5WldGck93b2dJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1gxQjBJRDRnTUM0'
    'd0tTQmZabE5oYlhCc1pWQnZjMEZqWXlBclBTQmZRMlpmYUNBcUlGOWtkQ0F2'
    'SUY5UWREc0tJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOXpkR1Z3SUNFOUlEQXBJ'
    'SHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR1pzYjJGMElGOVFiaUE5SUY5UWRD'
    'QXJJR1pzYjJGMEtGOXpkR1Z3S1RzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUds'
    'bUlDaGZZMnhoYlhCVWIxUm5kQ2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lHbG1JQ2hmYzNSbGNDQStJREFwSUNBZ0lDQWdYMUJ1SUQwZ2JXbHVL'
    'RjlRYml3Z2RHRnlaMlYwVUdWeWFXOWtLVHNLSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNCbGJITmxJR2xtSUNoZmMzUmxjQ0E4SURBcElGOVFiaUE5SUcx'
    'aGVDaGZVRzRzSUhSaGNtZGxkRkJsY21sdlpDazdDaUFnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQjlJR1ZzYzJVZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'RjlRYmlBOUlHTnNZVzF3S0Y5UWJpd2dNVEV6TGpBc0lEZzFOaTR3S1RzS0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUY5'
    'UWRDQTlJRjlRYmpzS0lDQWdJQ0FnSUNBZ0lDQWdmUW9nSUNBZ0lDQWdJSDBL'
    'SUNBZ0lDQWdJQ0F2THlCVGRXSXRkR2xqYXlCbWNtRmpkR2x2Ym1Gc0lHTnZi'
    'blJ5YVdKMWRHbHZiaUJoZENCamRYSnlaVzUwSUhCbGNtbHZaQW9nSUNBZ0lD'
    'QWdJR2xtSUNoZlpuSmhZeUErSURBdU1DQW1KaUJmVUhRZ1BpQXdMakFwQ2lB'
    'Z0lDQWdJQ0FnSUNBZ0lGOW1VMkZ0Y0d4bFVHOXpRV05qSUNzOUlGOURabDlv'
    'SUNvZ1gyUjBJQ29nWDJaeVlXTWdMeUJmVUhRN0NpQWdJQ0I5Q2dvZ0lDQWdM'
    'eThnVTNCbFkybGhiQ0JqWVhObE9pQjBjbWxuWjJWeUlISnZkeUJKVXlCamRY'
    'SnlaVzUwSUhKdmR5QW9ibThnYzJWd1lYSmhkR1VnZEhKcFoyZGxjaUIwWVds'
    'c0lDOGdjMk5oYmlrdUNpQWdJQ0F2THlCVWFHVWdZV05qZFcxMWJHRjBiM0ln'
    'WVdKdmRtVWdhR0Z1Wkd4bFpDQm9aV0ZrSUdaeWIyMGdkR2xqYXlBd0lIUnZJ'
    'SEJ2Y3k1MGFXTnJJR0YwSUdOdmJuTjBZVzUwQ2lBZ0lDQXZMeUIwY21sblRt'
    'OTBaUzV3WlhKcGIyUWdLSE5wYm1ObElGOXdVM1JoY25SRGRYSWdkMkZ6SUhO'
    'bGRDQjBieUJsWm1abFkzUnBkbVZRWlhKcGIyUWdkMmhwWTJnZ2IyNEtJQ0Fn'
    'SUM4dklIUm9hWE1nWTI5a1pTQndZWFJvSUdWeGRXRnNjeUIwY21sblRtOTBa'
    'UzV3WlhKcGIyUWc0b0NVSUhSb1pTQndZWEowYVdGc0xYQnBkR05vSUdKc2Iy'
    'TnJJR1JwWkc0bmRBb2dJQ0FnTHk4Z2NuVnVJR2xtSUc1dklITnNhV1JsSUdW'
    'bVptVmpkQ3dnYjNJZ2FYUWdjbUZ1SUdaeWIyMGdkSEpwWjA1dmRHVXVjR1Z5'
    'YVc5a0lHRnpJSE4wWVhKMGFXNW5JSEJ2YVc1MEtTNEtJQ0FnSUM4dklFNXZJ'
    'R0ZrWkdsMGFXOXVZV3dnWTI5a1pTQnVaV1ZrWldRNklHRmpZM1Z0ZFd4aGRH'
    'OXlJR2x6SUdOdmNuSmxZM1F1Q2dvZ0lDQWdMeThnVkhKbGJXOXNieUFvUlda'
    'bVpXTjBJREI0TnlrZzRvQ1VJSE5oYldVZ2QyRjJaV1p2Y20wZ1lYTWdkbWxp'
    'Y21GMGJ5QmlkWFFnYlc5a2RXeGhkR1Z6SUZaUFRGVk5SUzRLSUNBZ0lDOHZJ'
    'RlZ6WlhNZ2MyRnRaU0J5YjNjdFlua3RjbTkzSUdocGMzUnZjbWxqWVd3Z2RG'
    'TXZkRVFnZEhKaFkydHBibWNnWVhNZ2RtbGljbUYwYnk0S0lDQWdJSHNLSUNB'
    'Z0lDQWdJQ0JwYm5RZ1gzUlRJRDBnTUN3Z1gzUkVJRDBnTURzS0lDQWdJQ0Fn'
    'SUNCcGJuUWdYM1J5WlZCdmN5QTlJREE3Q2lBZ0lDQWdJQ0FnYVdZZ0tIUnlh'
    'V2RPYjNSbExtVm1abVZqZENBOVBTQXdlRGNwSUhzS0lDQWdJQ0FnSUNBZ0lD'
    'QWdhVzUwSUY5dWN5QTlJQ2gwY21sblRtOTBaUzV3WVhKaGJTQStQaUEwS1NB'
    'bUlEQjRSanNLSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjl1WkNBOUlDQjBjbWxu'
    'VG05MFpTNXdZWEpoYlNBZ0lDQWdJQ0FtSURCNFJqc0tJQ0FnSUNBZ0lDQWdJ'
    'Q0FnYVdZZ0tGOXVjeUErSURBcElGOTBVeUE5SUY5dWN6c0tJQ0FnSUNBZ0lD'
    'QWdJQ0FnYVdZZ0tGOXVaQ0ErSURBcElGOTBSQ0E5SUY5dVpEc0tJQ0FnSUNB'
    'Z0lDQjlDaUFnSUNBZ0lDQWdhV1lnS0hSeWFXZFFZWFFnUFQwZ2NHOXpMbk52'
    'Ym1kUWIzTWdKaVlnZEhKcFoxSnZkeUE5UFNCd2IzTXVjbTkzS1NCN0NpQWdJ'
    'Q0FnSUNBZ0lDQWdJRjkwY21WUWIzTWdQU0JwYm5Rb2NHOXpMblJwWTJzcElD'
    'b2dYM1JUT3dvZ0lDQWdJQ0FnSUgwZ1pXeHpaU0I3Q2lBZ0lDQWdJQ0FnSUNB'
    'Z0lHbHVkQ0JmZEhKVFozSWdQU0J3WVhSVWFXTnJUMlptYzJWMFczUnlhV2RR'
    'WVhSZElDc2dLSFJ5YVdkU2IzY2dMU0J3WVhSVGRHRnlkRkp2ZDF0MGNtbG5V'
    'R0YwWFNrN0NpQWdJQ0FnSUNBZ0lDQWdJR2x1ZENCZmRISlRjR1FnUFNCbVpY'
    'UmphRlJwWTJzb1gzUnlVMmR5SUNzZ01Ta2dMU0JtWlhSamFGUnBZMnNvWDNS'
    'eVUyZHlLVHNLSUNBZ0lDQWdJQ0FnSUNBZ1gzUnlaVkJ2Y3lBOUlDaGZkSEpU'
    'Y0dRZ0xTQXhLU0FxSUY5MFV6c0tJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOTNj'
    'Q0E5SUhSeWFXZFFZWFFzSUY5M2NpQTlJSFJ5YVdkU2IzY2dLeUF4T3dvZ0lD'
    'QWdJQ0FnSUNBZ0lDQnBaaUFvWDNkeUlENDlJSEJoZEZOMFlYSjBVbTkzVzE5'
    'M2NGMGdLeUFvY0dGMFVtOTNUMlptYzJWMFcxOTNjQ3N4WFNBdElIQmhkRkp2'
    'ZDA5bVpuTmxkRnRmZDNCZEtTa2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdY'
    'M2R3S3lzN0lGOTNjaUE5SUNoZmQzQWdQQ0JUVDA1SFgweEZUa2RVU0NrZ1B5'
    'QndZWFJUZEdGeWRGSnZkMXRmZDNCZElEb2dNRHNLSUNBZ0lDQWdJQ0FnSUNB'
    'Z2ZRb2dJQ0FnSUNBZ0lDQWdJQ0JtYjNJZ0tHbHVkQ0JmZDJrZ1BTQXdPeUJm'
    'ZDJrZ1BDQXhNamc3SUY5M2FTc3JLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0JwWmlBb1gzZHdJRDRnY0c5ekxuTnZibWRRYjNNZ2ZId2dLRjkzY0NBOVBT'
    'QndiM011YzI5dVoxQnZjeUFtSmlCZmQzSWdQajBnY0c5ekxuSnZkeWtwSUdK'
    'eVpXRnJPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5M2NDQStQU0JU'
    'VDA1SFgweEZUa2RVU0NrZ1luSmxZV3M3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0JwWmlBb1gzZHlJRDQ5SUhCaGRGTjBZWEowVW05M1cxOTNjRjBnS3lBb2NH'
    'RjBVbTkzVDJabWMyVjBXMTkzY0NzeFhTQXRJSEJoZEZKdmQwOW1abk5sZEZ0'
    'ZmQzQmRLU2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lGOTNjQ3Ny'
    'T3lCZmQzSWdQU0FvWDNkd0lEd2dVMDlPUjE5TVJVNUhWRWdwSUQ4Z2NHRjBV'
    'M1JoY25SU2IzZGJYM2R3WFNBNklEQTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnWTI5dWRHbHVkV1U3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0I5Q2lB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0JPYjNSbElGOTBiaUE5SUdkbGRFNXZkR1Vv'
    'WDNkd0xDQmZkM0lzSUdOb0tUc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHSnZi'
    'MndnWDNSdVNYTlViMjVsSUQwZ0tDaGZkRzR1WldabVpXTjBJRDA5SURCNE15'
    'QjhmQ0JmZEc0dVpXWm1aV04wSUQwOUlEQjROU2tnSmlZZ1gzUnVMbkJsY21s'
    'dlpDQStJREFwT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOTBiaTV3'
    'WlhKcGIyUWdQaUF3SUNZbUlDRmZkRzVKYzFSdmJtVXBJR0p5WldGck93b2dJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjkwYmk1bFptWmxZM1FnUFQwZ01I'
    'ZzNJQ1ltSUY5MGJpNXdZWEpoYlNBaFBTQXdLU0I3Q2lBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ2FXNTBJRjl1Y3lBOUlDaGZkRzR1Y0dGeVlXMGdQajRn'
    'TkNrZ0ppQXdlRVk3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJ'
    'Rjl1WkNBOUlDQmZkRzR1Y0dGeVlXMGdJQ0FnSUNBZ0ppQXdlRVk3Q2lBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjl1Y3lBK0lEQXBJRjkwVXlB'
    'OUlGOXVjenNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9YMjVr'
    'SUQ0Z01Da2dYM1JFSUQwZ1gyNWtPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdm'
    'UW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5elozSWdQU0J3WVhSVWFX'
    'TnJUMlptYzJWMFcxOTNjRjBnS3lBb1gzZHlJQzBnY0dGMFUzUmhjblJTYjNk'
    'YlgzZHdYU2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwYm5RZ1gzTndaQ0E5'
    'SUdabGRHTm9WR2xqYXloZmMyZHlJQ3NnTVNrZ0xTQm1aWFJqYUZScFkyc29Y'
    'M05uY2lrN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCZmRISmxVRzl6SUNzOUlD'
    'aGZjM0JrSUMwZ01Ta2dLaUJmZEZNN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNC'
    'ZmQzSXJLenNLSUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0F2'
    'THlCVmNHUmhkR1VnWm5KdmJTQmpkWEp5Wlc1MElISnZkeUJwWmlCcGRDQmpZ'
    'WEp5YVdWeklIUnlaVzF2Ykc4S0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5d1kz'
    'SXVaV1ptWldOMElEMDlJREI0TnlBbUppQmZjR055TG5CaGNtRnRJQ0U5SURB'
    'cElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0JmYm5NZ1BTQW9YM0Jq'
    'Y2k1d1lYSmhiU0ErUGlBMEtTQW1JREI0UmpzS0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUdsdWRDQmZibVFnUFNBZ1gzQmpjaTV3WVhKaGJTQWdJQ0FnSUNBbUlE'
    'QjRSanNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZmJuTWdQaUF3S1NC'
    'ZmRGTWdQU0JmYm5NN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9YMjVr'
    'SUQ0Z01Da2dYM1JFSUQwZ1gyNWtPd29nSUNBZ0lDQWdJQ0FnSUNCOUNpQWdJ'
    'Q0FnSUNBZ0lDQWdJRjkwY21WUWIzTWdLejBnYVc1MEtIQnZjeTUwYVdOcktT'
    'QXFJRjkwVXpzS0lDQWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ2FXWWdLRjkwUkNB'
    'K0lEQWdKaVlnWDNSVElENGdNQ2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQnBiblFn'
    'WDNSUUlEMGdYM1J5WlZCdmN5QW1JRFl6T3dvZ0lDQWdJQ0FnSUNBZ0lDQm1i'
    'RzloZENCZmRFUmxiSFJoSUQwZ0tIWnBZbFJoWWx0ZmRGQWdKaUF6TVYwZ0tp'
    'Qm1iRzloZENoZmRFUXBLU0F2SURZMExqQTdDaUFnSUNBZ0lDQWdJQ0FnSUM4'
    'dklGUnlaVzF2Ykc4Z2JXOWthV1pwWlhNZ2RHaGxJRTlWVkZCVlZDQmhiWEJz'
    'YVhSMVpHVWdjR1Z5TFhOaGJYQnNaVHNnYVhRZ1pHOWxjMjRuZEFvZ0lDQWdJ'
    'Q0FnSUNBZ0lDQXZMeUJuWlhRZ2MyMXZiM1JvWldRZ2RtbGhJRlpQVEY5VFJW'
    'UWdZbVZqWVhWelpTQnBkQ2R6SUdGc2NtVmhaSGtnWVNCbVlYTjBJRzl6WTJs'
    'c2JHRjBhVzl1Q2lBZ0lDQWdJQ0FnSUNBZ0lDOHZJQ2h6Ylc5dmRHaHBibWNn'
    'ZDI5MWJHUWdjM1Z3Y0hKbGMzTWdhWFFwTGlCVGRHOXlaV1FnWVhNZ1lTQmta'
    'V3gwWVNCaGJtUWdZV1JrWldRZ2RHOEtJQ0FnSUNBZ0lDQWdJQ0FnTHk4Z2RH'
    'aGxJSE50YjI5MGFHVmtJSFp2YkhWdFpTQmhkQ0J2ZFhSd2RYUWdkR2x0WlM0'
    'S0lDQWdJQ0FnSUNBZ0lDQWdYM1J5WlcxdmJHOUVaV3gwWVNBOUlDaGZkRkFn'
    'UENBek1pa2dQeUJmZEVSbGJIUmhJRG9nTFY5MFJHVnNkR0U3Q2lBZ0lDQWdJ'
    'Q0FnZlFvZ0lDQWdmUW9LSUNBZ0lDOHZJRUZ5Y0dWbloybHZJQ2hGWm1abFkz'
    'UWdNSGg1S1NEaWdKUWdjSGx0YjJRbmN5QnZjbVJsY2lCcGN5QmlZWE5sNG9h'
    'U1dDaG9hV2RvS2VLR2tsa29iRzkzS1FvZ0lDQWdhV1lnS0Y5d1kzSXVaV1pt'
    'WldOMElEMDlJREI0TUNBbUppQmZjR055TG5CaGNtRnRJQ0U5SURBcElIc0tJ'
    'Q0FnSUNBZ0lDQnBiblFnWDJGeWNGTjBaWEFnUFNCcGJuUW9jRzl6TG5ScFky'
    'c3BJQzBnYVc1MEtIQnZjeTUwYVdOcklDOGdNeTR3S1NBcUlETTdDaUFnSUNB'
    'Z0lDQWdMeThnWldabVpXTjBhWFpsVUdWeWFXOWtJRWxUSUdKaGMyVlFaWEpw'
    'YjJRZ2FHVnlaU0FvYm04Z1puVnlkR2hsY2lCdGIyUnBabWxqWVhScGIyNGdZ'
    'bVZtYjNKbElHRnljQ2tLSUNBZ0lDQWdJQ0JwWmlBb1gyRnljRk4wWlhBZ1BU'
    'MGdNU2tLSUNBZ0lDQWdJQ0FnSUNBZ1pXWm1aV04wYVhabFVHVnlhVzlrSUQw'
    'Z1pXWm1aV04wYVhabFVHVnlhVzlrSUNvZ2NHOTNLREl1TUN3Z0xXWnNiMkYw'
    'S0NoZmNHTnlMbkJoY21GdElENCtJRFFwSUNZZ01IaEdLU0F2SURFeUxqQXBP'
    'd29nSUNBZ0lDQWdJR1ZzYzJVZ2FXWWdLRjloY25CVGRHVndJRDA5SURJcENp'
    'QWdJQ0FnSUNBZ0lDQWdJR1ZtWm1WamRHbDJaVkJsY21sdlpDQTlJR1ZtWm1W'
    'amRHbDJaVkJsY21sdlpDQXFJSEJ2ZHlneUxqQXNJQzFtYkc5aGRDaGZjR055'
    'TG5CaGNtRnRJQ1lnTUhoR0tTQXZJREV5TGpBcE93b2dJQ0FnZlFvS0lDQWdJ'
    'Qzh2SUZacFluSmhkRzhnS0VWbVptVmpkQ0EwS1NEaWdKUWdkWE5sY3lCbmJH'
    'OWlZV3dnZG1saVZHRmlMZ29nSUNBZ0x5OGdSV1ptWldOMElEUjRlRG9nY0dG'
    'eVlXMGdQU0FvYzNCbFpXUWdQRHdnTkNrZ2ZDQmtaWEIwYUM0Z0lGTmxkSE1n'
    'ZGxNc0lIWkVMZ29nSUNBZ0x5OGdSV1ptWldOMElEWjRlRG9nWTI5dWRHbHVk'
    'V1VnZG1saWNtRjBieUFvZFhObGN5QndjbWx2Y2lBMGVIZ25jeUIyVXk5MlJE'
    'c2dhWFJ6SUc5M2JpQndZWEpoYlNCcGN3b2dJQ0FnTHk4Z0lDQWdJQ0FnSUNB'
    'Z0lDQWdkbTlzTFhOc2FXUmxJRzl1YkhrZzRvQ1VJR2hoYm1Sc1pXUWdjMlZ3'
    'WVhKaGRHVnNlU0JwYmlCMmIyeDFiV1VnWTI5a1pTQndZWFJvS1M0S0lDQWdJ'
    'Qzh2Q2lBZ0lDQXZMeUIyYVdKeVlYUnZVRzl6SUdsdVkzSmxiV1Z1ZEhNZ1lu'
    'a2dkbE1nYjI0Z1pXRmphQ0JPVDA0dGRHbGpheTB3SUNocExtVXVMQ0FvYzNC'
    'bFpXUXRNU2tnY0dWeUlISnZkeWt1Q2lBZ0lDQXZMeUJYWVd4cklHWnliMjBn'
    'ZEhKcFoyZGxjaUIwYnlCamRYSnlaVzUwTENCMWNHUmhkR2x1WnlCMlV5OTJS'
    'Q0JQVGt4WklHOXVJRFI0ZUNCeWIzZHpMQ0JoYm1RS0lDQWdJQzh2SUdGalkz'
    'VnRkV3hoZEdsdVp5QW9jM0JsWldRdE1Ta3FkbE1nY0dWeUlHTnZiWEJzWlhS'
    'bFpDQnliM2NnZFhOcGJtY2dhR2x6ZEc5eWFXTmhiQ0IyVXk0S0lDQWdJSHNL'
    'SUNBZ0lDQWdJQ0JwYm5RZ1gzWlRJRDBnTUN3Z1gzWkVJRDBnTURzS0lDQWdJ'
    'Q0FnSUNCcGJuUWdYM1pwWWxCdmN5QTlJREE3Q2dvZ0lDQWdJQ0FnSUM4dklF'
    'bHVhWFJwWVd4cGVtVWdkbE12ZGtRZ1QwNU1XU0JtY205dElIUnlhV2RuWlhJ'
    'Z2NtOTNKM01nTUhnMElDaE9UMVFnTUhnMklPS0FsQ0JwZEhNZ2NHRnlZVzBn'
    'YVhNZ2RtOXNMWE5zYVdSbEtRb2dJQ0FnSUNBZ0lHbG1JQ2gwY21sblRtOTBa'
    'UzVsWm1abFkzUWdQVDBnTUhnMEtTQjdDaUFnSUNBZ0lDQWdJQ0FnSUdsdWRD'
    'QmZibk1nUFNBb2RISnBaMDV2ZEdVdWNHRnlZVzBnUGo0Z05Da2dKaUF3ZUVZ'
    'N0NpQWdJQ0FnSUNBZ0lDQWdJR2x1ZENCZmJtUWdQU0FnZEhKcFowNXZkR1V1'
    'Y0dGeVlXMGdJQ0FnSUNBZ0ppQXdlRVk3Q2lBZ0lDQWdJQ0FnSUNBZ0lHbG1J'
    'Q2hmYm5NZ1BpQXdLU0JmZGxNZ1BTQmZibk03Q2lBZ0lDQWdJQ0FnSUNBZ0lH'
    'bG1JQ2hmYm1RZ1BpQXdLU0JmZGtRZ1BTQmZibVE3Q2lBZ0lDQWdJQ0FnZlFv'
    'S0lDQWdJQ0FnSUNCcFppQW9kSEpwWjFCaGRDQTlQU0J3YjNNdWMyOXVaMUJ2'
    'Y3lBbUppQjBjbWxuVW05M0lEMDlJSEJ2Y3k1eWIzY3BJSHNLSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0x5OGdUMjRnZEhKcFoyZGxjaUJ5YjNjNklIWnBZbkpoZEc4Z2FH'
    'RnpJRzl1YkhrZ2FHRmtJSEJ2Y3k1MGFXTnJJR2x1WTNKbGJXVnVkSE1LSUNB'
    'Z0lDQWdJQ0FnSUNBZ1gzWnBZbEJ2Y3lBOUlHbHVkQ2h3YjNNdWRHbGpheWtn'
    'S2lCZmRsTTdDaUFnSUNBZ0lDQWdmU0JsYkhObElIc0tJQ0FnSUNBZ0lDQWdJ'
    'Q0FnTHk4Z1ZISnBaMmRsY2lCeWIzY2dZMjl1ZEhKcFluVjBaWE1nS0hOd1pX'
    'VmtMVEVwSUdsdVkzSmxiV1Z1ZEhNZ1lYUWdkSEpwWjJkbGNpMXliM2NnZGxN'
    'S0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5MGNsTm5jaUE5SUhCaGRGUnBZMnRQ'
    'Wm1aelpYUmJkSEpwWjFCaGRGMGdLeUFvZEhKcFoxSnZkeUF0SUhCaGRGTjBZ'
    'WEowVW05M1czUnlhV2RRWVhSZEtUc0tJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElG'
    'OTBjbE53WkNBOUlHWmxkR05vVkdsamF5aGZkSEpUWjNJZ0t5QXhLU0F0SUda'
    'bGRHTm9WR2xqYXloZmRISlRaM0lwT3dvZ0lDQWdJQ0FnSUNBZ0lDQXZMeUJX'
    'YVdKeVlYUnZJR3RsWlhCeklISjFibTVwYm1jZ2IyNGdNSGcySUhKdmQzTWdk'
    'Rzl2SU9LQWxDQmhZMk4xYlhWc1lYUmxJQ2h6Y0dWbFpDMHhLU3AyVXdvZ0lD'
    'QWdJQ0FnSUNBZ0lDQXZMeUJsZG1WdUlIZG9aVzRnZEdocGN5QnliM2NnZDJG'
    'eklEQjROaXdnZFhOcGJtY2dkR2hsSUdsdWFHVnlhWFJsWkNCMlV5NEtJQ0Fn'
    'SUNBZ0lDQWdJQ0FnWW05dmJDQmZkSEpwWjBselZtbGlRV04wYVhabElEMGdL'
    'SFJ5YVdkT2IzUmxMbVZtWm1WamRDQTlQU0F3ZURRZ2ZId2dkSEpwWjA1dmRH'
    'VXVaV1ptWldOMElEMDlJREI0TmlrN0NpQWdJQ0FnSUNBZ0lDQWdJRjkyYVdK'
    'UWIzTWdQU0JmZEhKcFowbHpWbWxpUVdOMGFYWmxJRDhnS0Y5MGNsTndaQ0F0'
    'SURFcElDb2dYM1pUSURvZ01Ec0tDaUFnSUNBZ0lDQWdJQ0FnSUM4dklGZGhi'
    'R3NnY205M0xXSjVMWEp2ZHlCbWNtOXRJSFJ5YVdkblpYSXJNU0IwYnlCamRY'
    'SnlaVzUwTFRFc0lIVndaR0YwYVc1bklIWlRMM1pFQ2lBZ0lDQWdJQ0FnSUNB'
    'Z0lDOHZJRzl1SURCNE5DQnliM2R6TENCaGJtUWdZV05qZFcxMWJHRjBhVzVu'
    'SUhCbGNpMXliM2NnZFhOcGJtY2dhR2x6ZEc5eWFXTmhiQ0IyVXk0S0lDQWdJ'
    'Q0FnSUNBZ0lDQWdhVzUwSUY5M2NDQTlJSFJ5YVdkUVlYUXNJRjkzY2lBOUlI'
    'UnlhV2RTYjNjZ0t5QXhPd29nSUNBZ0lDQWdJQ0FnSUNCcFppQW9YM2R5SUQ0'
    'OUlIQmhkRk4wWVhKMFVtOTNXMTkzY0YwZ0t5QW9jR0YwVW05M1QyWm1jMlYw'
    'VzE5M2NDc3hYU0F0SUhCaGRGSnZkMDltWm5ObGRGdGZkM0JkS1NrZ2V3b2dJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBZ1gzZHdLeXM3SUY5M2NpQTlJQ2hmZDNBZ1BD'
    'QlRUMDVIWDB4RlRrZFVTQ2tnUHlCd1lYUlRkR0Z5ZEZKdmQxdGZkM0JkSURv'
    'Z01Ec0tJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNBZ0lDQm1iM0ln'
    'S0dsdWRDQmZkMmtnUFNBd095QmZkMmtnUENBeE1qZzdJRjkzYVNzcktTQjdD'
    'aUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvWDNkd0lENGdjRzl6TG5OdmJt'
    'ZFFiM01nZkh3Z0tGOTNjQ0E5UFNCd2IzTXVjMjl1WjFCdmN5QW1KaUJmZDNJ'
    'Z1BqMGdjRzl6TG5KdmR5a3BJR0p5WldGck93b2dJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ2FXWWdLRjkzY0NBK1BTQlRUMDVIWDB4RlRrZFVTQ2tnWW5KbFlXczdD'
    'aUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvWDNkeUlENDlJSEJoZEZOMFlY'
    'SjBVbTkzVzE5M2NGMGdLeUFvY0dGMFVtOTNUMlptYzJWMFcxOTNjQ3N4WFNB'
    'dElIQmhkRkp2ZDA5bVpuTmxkRnRmZDNCZEtTa2dld29nSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUY5M2NDc3JPeUJmZDNJZ1BTQW9YM2R3SUR3Z1UwOU9S'
    'MTlNUlU1SFZFZ3BJRDhnY0dGMFUzUmhjblJTYjNkYlgzZHdYU0E2SURBN0Np'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdZMjl1ZEdsdWRXVTdDaUFnSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQk9iM1Js'
    'SUY5MmJpQTlJR2RsZEU1dmRHVW9YM2R3TENCZmQzSXNJR05vS1RzS0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUM4dklGTjBiM0FnYjI0Z2NtVjBjbWxuWjJWeUNp'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCaWIyOXNJRjkyYmtselZHOXVaU0E5SUNn'
    'b1gzWnVMbVZtWm1WamRDQTlQU0F3ZURNZ2ZId2dYM1p1TG1WbVptVmpkQ0E5'
    'UFNBd2VEVXBJQ1ltSUY5MmJpNXdaWEpwYjJRZ1BpQXdLVHNLSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJR2xtSUNoZmRtNHVjR1Z5YVc5a0lENGdNQ0FtSmlBaFgz'
    'WnVTWE5VYjI1bEtTQmljbVZoYXpzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUM4'
    'dklGVndaR0YwWlNCMlV5OTJSQ0JQVGt4WklHOXVJREI0TkNCeWIzZHpJQ2d3'
    'ZURZZ2FHRnpJSFp2YkMxemJHbGtaU0J3WVhKaGJTd2dibTkwSUhacFluSmhk'
    'RzhwQ2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1gzWnVMbVZtWm1WamRD'
    'QTlQU0F3ZURRZ0ppWWdYM1p1TG5CaGNtRnRJQ0U5SURBcElIc0tJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwYm5RZ1gyNXpJRDBnS0Y5MmJpNXdZWEpo'
    'YlNBK1BpQTBLU0FtSURCNFJqc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0JwYm5RZ1gyNWtJRDBnSUY5MmJpNXdZWEpoYlNBZ0lDQWdJQ0FtSURCNFJq'
    'c0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1gyNXpJRDRnTUNr'
    'Z1gzWlRJRDBnWDI1ek93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2xt'
    'SUNoZmJtUWdQaUF3S1NCZmRrUWdQU0JmYm1RN0NpQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNCOUNpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBdkx5QkJZMk4xYlhWc1lY'
    'UmxJSFpwWW5KaGRHOGdjRzl6SUhkb1pXNGdjbTkzSUdseklEQjROQ0JQVWlB'
    'd2VEWWdLSFpwWW5KaGRHOGdjblZ1Y3lCdmJpQmliM1JvS1FvZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnYVdZZ0tGOTJiaTVsWm1abFkzUWdQVDBnTUhnMElIeDhJ'
    'RjkyYmk1bFptWmxZM1FnUFQwZ01IZzJLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ2FXNTBJRjl6WjNJZ1BTQndZWFJVYVdOclQyWm1jMlYwVzE5'
    'M2NGMGdLeUFvWDNkeUlDMGdjR0YwVTNSaGNuUlNiM2RiWDNkd1hTazdDaUFn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOXpjR1FnUFNCbVpYUmph'
    'RlJwWTJzb1gzTm5jaUFySURFcElDMGdabVYwWTJoVWFXTnJLRjl6WjNJcE93'
    'b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJRjkyYVdKUWIzTWdLejBnS0Y5'
    'emNHUWdMU0F4S1NBcUlGOTJVenNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJSDBL'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJRjkzY2lzck93b2dJQ0FnSUNBZ0lDQWdJ'
    'Q0I5Q2dvZ0lDQWdJQ0FnSUNBZ0lDQXZMeUJWY0dSaGRHVWdkbE12ZGtRZ1pu'
    'SnZiU0JqZFhKeVpXNTBJSEp2ZHlCUFRreFpJR2xtSURCNE5Bb2dJQ0FnSUNB'
    'Z0lDQWdJQ0JwWmlBb1gzQmpjaTVsWm1abFkzUWdQVDBnTUhnMElDWW1JRjl3'
    'WTNJdWNHRnlZVzBnSVQwZ01Da2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdh'
    'VzUwSUY5dWN5QTlJQ2hmY0dOeUxuQmhjbUZ0SUQ0K0lEUXBJQ1lnTUhoR093'
    'b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjl1WkNBOUlDQmZjR055TG5C'
    'aGNtRnRJQ0FnSUNBZ0lDWWdNSGhHT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'YVdZZ0tGOXVjeUErSURBcElGOTJVeUE5SUY5dWN6c0tJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lHbG1JQ2hmYm1RZ1BpQXdLU0JmZGtRZ1BTQmZibVE3Q2lBZ0lD'
    'QWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnTHk4Z1EzVnljbVZ1ZENC'
    'eWIzY2djR0Z5ZEdsaGJEb2djRzl6TG5ScFkyc2dhVzVqY21WdFpXNTBjeUJo'
    'ZENCamRYSnlaVzUwTFhKdmR5QjJVd29nSUNBZ0lDQWdJQ0FnSUNBdkx5QlBU'
    'a3haSUhkb1pXNGdZM1Z5Y21WdWRDQnliM2NnYUdGeklIWnBZbkpoZEc4Z1lX'
    'TjBhWFpsSUNnd2VEUWdiM0lnTUhnMktRb2dJQ0FnSUNBZ0lDQWdJQ0JwWmlB'
    'b1gzQmpjaTVsWm1abFkzUWdQVDBnTUhnMElIeDhJRjl3WTNJdVpXWm1aV04w'
    'SUQwOUlEQjROaWtnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnWDNacFlsQnZj'
    'eUFyUFNCcGJuUW9jRzl6TG5ScFkyc3BJQ29nWDNaVE93b2dJQ0FnSUNBZ0lD'
    'QWdJQ0I5Q2lBZ0lDQWdJQ0FnZlFvS0lDQWdJQ0FnSUNCcFppQW9YM1pFSUQ0'
    'Z01DQW1KaUJmZGxNZ1BpQXdLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0Jm'
    'ZGxBZ1BTQmZkbWxpVUc5eklDWWdOak03Q2lBZ0lDQWdJQ0FnSUNBZ0lHWnNi'
    'MkYwSUY5MlJHVnNkR0VnUFNBb2RtbGlWR0ZpVzE5MlVDQW1JRE14WFNBcUlH'
    'WnNiMkYwS0Y5MlJDa3BJQzhnTVRJNExqQTdDaUFnSUNBZ0lDQWdJQ0FnSUdW'
    'bVptVmpkR2wyWlZCbGNtbHZaQ0FyUFNBb1gzWlFJRHdnTXpJcElEOGdYM1pF'
    'Wld4MFlTQTZJQzFmZGtSbGJIUmhPd29nSUNBZ0lDQWdJSDBLSUNBZ0lIMEtD'
    'aUFnSUNBdkx5QlNaVzVrWlhJZ2MyRnRjR3hsQ2lBZ0lDQXZMeUJpWVhObFVH'
    'VnlhVzlrSUQwZ1pXWm1aV04wYVhabFVHVnlhVzlrSUZkSlZFaFBWVlFnZG1s'
    'aWNtRjBieTkwY21WdGIyeHZJRzF2WkhWc1lYUnBiMjR1SUNCVmMybHVaeUIw'
    'YUdVS0lDQWdJQzh2SUcxdlpIVnNZWFJsWkNCMllXeDFaU0JtYjNJZ1psTmhi'
    'WEJzWlZCdmN5QnBiblJsWjNKaGRHbHZiaUIzYjNWc1pDQnRkV3gwYVhCc2VT'
    'QjBhR1VnYlc5a2RXeGhkR2x2YmdvZ0lDQWdMeThnWVcxd2JHbDBkV1JsSUdK'
    'NUlHQmxiR0Z3YzJWa1lDd2djSEp2WkhWamFXNW5JR0VnYzNWaWMzUmhiblJw'
    'WVd3Z1luVjZlaUJoZENCMmFXSnlZWFJ2SUhKaGRHVUtJQ0FnSUM4dklDaGxM'
    'bWN1TENCbWJIVjBaU0J2YmlCd1lYUWdNU0JqYURBZ1lYUWdkbVZqWDJScGJU'
    'MDRLUzRnSUZSb1pTQlVVbFZGSUhCdmMybDBhVzl1TFdSdmJXRnBiaUJsWm1a'
    'bFkzUUtJQ0FnSUM4dklHOW1JSFpwWW5KaGRHOGdhWE1nZEdobElHbHVkR1Zu'
    'Y21Gc0lHOW1JSFJvWlNCbWNtVnhJRzF2WkhWc1lYUnBiMjRzSUhkb2FXTm9J'
    'R2x6SUdFZ2RHbHVlU0E4TVMwS0lDQWdJQzh2SUhOaGJYQnNaU0J2YzJOcGJH'
    'eGhkR2x2YmlEaWdKUWdjMkZtWld4NUlHNWxaMnhwWjJsaWJHVXVDaUFnSUNC'
    'bWJHOWhkQ0JpWVhObFVHVnlhVzlrSUQwZ1pXWm1aV04wYVhabFVHVnlhVzlr'
    'T3dvZ0lDQWdhV1lnS0hSeWFXZE9iM1JsTG1WbVptVmpkQ0E5UFNBd2VEUWdm'
    'SHdnZEhKcFowNXZkR1V1WldabVpXTjBJRDA5SURCNE5pQjhmQ0IwY21sblRt'
    'OTBaUzVsWm1abFkzUWdQVDBnTUhnM0lIeDhDaUFnSUNBZ0lDQWdYM0JqY2k1'
    'bFptWmxZM1FnUFQwZ01IZzBJSHg4SUY5d1kzSXVaV1ptWldOMElEMDlJREI0'
    'TmlCOGZDQmZjR055TG1WbVptVmpkQ0E5UFNBd2VEY3BJSHNLSUNBZ0lDQWdJ'
    'Q0JpWVhObFVHVnlhVzlrSUQwZ0tIUmhjbWRsZEZCbGNtbHZaQ0ErSURBdU1D'
    'a2dQeUIwWVhKblpYUlFaWEpwYjJRZ09pQm1iRzloZENoMGNtbG5UbTkwWlM1'
    'd1pYSnBiMlFwT3dvZ0lDQWdmUW9nSUNBZ1pteHZZWFFnWm5KbGNTQTlJSEJs'
    'Y21sdlpGUnZSbkpsY1VaMEtHMWhlQ2d4TENCcGJuUW9ZbUZ6WlZCbGNtbHZa'
    'Q2twTENCemJYQXVabWx1WlhSMWJtVXBPd29nSUNBZ0x5OGdVMkZ0Y0d4bElI'
    'QnZjMmwwYVc5dU9nb2dJQ0FnTHk4Z0lDQXRJRWxtSUdOMWNuSmxiblFnY205'
    'M0lHaGhjeUJoWTNScGRtVWdjR2wwWTJnZ2MyeHBaR1VnS0RGNGVDOHllSGd2'
    'TTNoNEtTd2dkWE5sSUd4dlp5QnBiblJsWjNKaGJBb2dJQ0FnTHk4Z0lDQWdJ'
    'T0tJcTBNdlVDaDBLV1IwSU9LSmlDQkR3NWRVTDg2VVVDRERseUJzYmloUU1T'
    'OVFNQ2tnSUNoaGMzTjFiV1Z6SUd4cGJtVmhjaUJ5WVcxd095QmpiRzl6WlNC'
    'bGJtOTFaMmdwQ2lBZ0lDQXZMeUFnSUMwZ1QzUm9aWEozYVhObElIQmxjbWx2'
    'WkNCcGN5QnpkR0ZpYkdVZ2NHRnpkQ0IwY21sbloyVnlPeUJ6YVcxd2JHVWda'
    'V3hoY0hObFpNT1habkpsY1NCcGN5QmxlR0ZqZEM0S0lDQWdJQzh2SUZOaGJY'
    'QnNaU0J3YjNOcGRHbHZiaUJtY205dElIUm9aU0J3WlhJdGNtOTNJR1Y0WVdO'
    'MElHbHVkR1ZuY21GMGIzSWdLSEpsY0d4aFkyVnpJSFJvWlNCdmJHUUtJQ0Fn'
    'SUM4dklITnBibWRzWlMxelpXZHRaVzUwSUdadmNtMTFiR0VnZDJocFkyZ2dk'
    'MkZ6SUhkeWIyNW5JR1p2Y2lCdGRXeDBhUzF5YjNjZ2MyeHBaR1Z6SU9LQWxD'
    'QnpaV1VLSUNBZ0lDOHZJRjltVTJGdGNHeGxVRzl6UVdOaklHTnZibk4wY25W'
    'amRHbHZiaUJoWW05MlpTa3VJRkpsYzNWc2RDQnBjeUJwYmlCemIzVnlZMlV0'
    'Y21GMFpTQnpZVzF3YkdWek93b2dJQ0FnTHk4Z1pHbDJhV1JsSUdKNUlHSjNS'
    'bUZqZEc5eUlIUnZJR052Ym5abGNuUWdkRzhnWTI5dGNISmxjM05sWkMxa2Iy'
    'MWhhVzRnYzJGdGNHeGxjeUJzYVd0bElIUm9aUW9nSUNBZ0x5OGdiR1ZuWVdO'
    'NUlHTnZaR1VnWkdsa0xnb2dJQ0FnWm14dllYUWdabE5oYlhCc1pWQnZjeUE5'
    'SUY5bVUyRnRjR3hsVUc5elFXTmpJQzhnWm14dllYUW9jMjF3TG1KM1JtRmpk'
    'Rzl5S1RzS0lDQWdJQzh2SURCNE9YaDRJSE5oYlhCc1pTQnZabVp6WlhRNklI'
    'Tm9hV1owSUhOMFlYSjBhVzVuSUhCdmMybDBhVzl1SUNocGJpQmpiMjF3Y21W'
    'emMyVmtMV1J2YldGcGJpQnpZVzF3YkdWektRb2dJQ0FnYVdZZ0tGOXpZVzF3'
    'YkdWUFptWnpaWFFnUGlBd0tTQjdDaUFnSUNBZ0lDQWdabE5oYlhCc1pWQnZj'
    'eUFyUFNCbWJHOWhkQ2hmYzJGdGNHeGxUMlptYzJWMElDOGdiV0Y0S0RFc0lI'
    'TnRjQzVpZDBaaFkzUnZjaWtwT3dvZ0lDQWdmUW9LSUNBZ0lHbG1JQ2h6YlhB'
    'dWJHOXZjRXhsYmlBK0lESXBJSHNLSUNBZ0lDQWdJQ0JwWmlBb1psTmhiWEJz'
    'WlZCdmN5QStQU0JtYkc5aGRDaHpiWEF1Ykc5dmNGTjBZWEowSUNzZ2MyMXdM'
    'bXh2YjNCTVpXNHBLUW9nSUNBZ0lDQWdJQ0FnSUNCbVUyRnRjR3hsVUc5eklE'
    'MGdabXh2WVhRb2MyMXdMbXh2YjNCVGRHRnlkQ2tnS3lCdGIyUW9abE5oYlhC'
    'c1pWQnZjeUF0SUdac2IyRjBLSE50Y0M1c2IyOXdVM1JoY25RcExDQm1iRzlo'
    'ZENoemJYQXViRzl2Y0V4bGJpa3BPd29nSUNBZ2ZTQmxiSE5sSUdsbUlDaG1V'
    'MkZ0Y0d4bFVHOXpJRDQ5SUdac2IyRjBLSE50Y0M1c1pXNW5kR2dwS1NCN0Np'
    'QWdJQ0FnSUNBZ2NtVjBkWEp1SURBdU1Ec0tJQ0FnSUgwS0lDQWdJR2xtSUNo'
    'bVUyRnRjR3hsVUc5eklEd2dNQzR3S1NCeVpYUjFjbTRnTUM0d093b0tJQ0Fn'
    'SUM4dklGTmhiWEJzWlNCMllXeDFaU0IzYVhSb0lIQnliM0JsY2lCbGJtUXRa'
    'bUZrWlNBb2MyRnRjR3hsSUhSbGNtMXBibUYwYVc5dUlITm9iM1ZzWkNCdWIz'
    'UWdjMjVoY0NCMGJ5QXdLUW9nSUNBZ1pteHZZWFFnY3pzS0lDQWdJR2xtSUNo'
    'emJYQXViRzl2Y0V4bGJpQThQU0F5SUNZbUlHWlRZVzF3YkdWUWIzTWdQajBn'
    'Wm14dllYUW9jMjF3TG14bGJtZDBhQ2tnTFNBeExqQXBJSHNLSUNBZ0lDQWdJ'
    'Q0F2THlCT1pXRnlJR1Z1WkNCdlppQnViMjR0Ykc5dmNHbHVaeUJ6WVcxd2JH'
    'VTZJR1poWkdVZ2IzVjBJRzkyWlhJZ2JHRnpkQ0J6WVcxd2JHVWdkRzhnWVha'
    'dmFXUWdZMnhwWTJzS0lDQWdJQ0FnSUNCeklEMGdNQzR3T3dvZ0lDQWdmU0Js'
    'YkhObElIc0tJQ0FnSUNBZ0lDQnpJRDBnWjJWMFUyRnRjR3hsUmloemJYQXVj'
    'M1JoY25Rc0lHWlRZVzF3YkdWUWIzTXNJSE50Y0M1c1pXNW5kR2dzSUhOdGND'
    'NXNiMjl3VTNSaGNuUXNJSE50Y0M1c2IyOXdUR1Z1S1RzS0lDQWdJSDBLQ2lB'
    'Z0lDQXZMeURpbElEaWxJQWdRVzUwYVMxamJHbGpheUJ5WVcxd2N5RGlsSURp'
    'bElBS0lDQWdJQzh2SURFdUlGUnlhV2RuWlhJZ2NtRnRjRG9nUVVSQlVGUkpW'
    'a1VnWm1Ga1pTMXBiaTRLSUNBZ0lDOHZJQ0FnSUVSbFptRjFiSFE2SURZMExY'
    'TmhiWEJzWlNCc2FXNWxZWElnS0cxcGEwbFVJR1poWkdWamIzVnVkQ2tnNG9D'
    'VUlITm9ZWEp3SUdSeWRXMGdZWFIwWVdOckxnb2dJQ0FnTHk4Z0lDQWdVMkZ0'
    'Y0d4bExXOW1abk5sZENCeVpYUnlhV2RuWlhKeklDZzVlSGdwT2lBeE9USXRj'
    'MkZ0Y0d4bElITnRiMjkwYUhOMFpYQWc0b0NVSUcxaGMydHpJSFJvWlFvZ0lD'
    'QWdMeThnSUNBZ2JXbGtMWGRoZG1WbWIzSnRJR1JwYzJOdmJuUnBiblZwZEhr'
    'Z2RHaGhkQ0JqWVhWelpYTWdZMnhwWTJ0eklHOXVJR1J5ZFcwdFkyaHZjSEJw'
    'Ym1jS0lDQWdJQzh2SUNBZ0lIQmhkSFJsY201ekxpQkVjblZ0Y3lCamFHOXdj'
    'R1ZrSUhacFlTQTVlSGdnYzNSaGNuUWdZWFFnYm05dUxYcGxjbThnWVcxd2JH'
    'bDBkV1JsSUdsdWMybGtaUW9nSUNBZ0x5OGdJQ0FnZEdobElITmhiWEJzWlN3'
    'Z1lXNWtJSFJvWlNCd2NtVjJhVzkxY3lCdWIzUmxKM01nZEdGcGJDQnFkWE4w'
    'SUhOMGIzQnpMQ0J6YnlCM2FYUm9iM1YwQ2lBZ0lDQXZMeUFnSUNCaElHeHZi'
    'bWRsY2lCeVlXMXdJR1YyWlhKNUlISmxkSEpwWjJkbGNpQndiM0J6SUdGMVpH'
    'bGliSGt1Q2lBZ0lDQm1iRzloZENCa1pXTnNhV05yT3dvZ0lDQWdhV1lnS0Y5'
    'ellXMXdiR1ZQWm1aelpYUWdQaUF3S1NCN0NpQWdJQ0FnSUNBZ0x5OGdVMjF2'
    'YjNSb2MzUmxjQ0J2ZG1WeUlERTVNaUJ6WVcxd2JHVnpJQ2grTkM0MGJYTWdR'
    'Q0EwTkM0eGEwaDZLUzRnVTIxdmIzUm9jM1JsY0NCb1lYTUtJQ0FnSUNBZ0lD'
    'QXZMeUI2WlhKdklHUmxjbWwyWVhScGRtVWdZWFFnWW05MGFDQmxibVJ3YjJs'
    'dWRITWc0b0NVSUc1dklHRjFaR2xpYkdVZ2EybHVheUJoZENCMGFHVWdjM1Jo'
    'Y25RS0lDQWdJQ0FnSUNBdkx5QnZjaUJsYm1RZ2IyWWdkR2hsSUdaaFpHVXNJ'
    'R3AxYzNRZ1lTQnpiVzl2ZEdnZ2MzZGxiR3d1SUV4dmJtY2daVzV2ZFdkb0lI'
    'UnZJRzFoYzJzS0lDQWdJQ0FnSUNBdkx5QjBhR1VnYldsa0xYZGhkbVZtYjNK'
    'dElITjBZWEowSUdScGMyTnZiblJwYm5WcGRIa3NJSE5vYjNKMElHVnViM1Zu'
    'YUNCMGJ5QndjbVZ6WlhKMlpRb2dJQ0FnSUNBZ0lDOHZJSEJsY21ObGFYWmxa'
    'Q0JoZEhSaFkyc2diMjRnYzJ4dmR5QmtjblZ0SUdocGRITXVDaUFnSUNBZ0lD'
    'QWdabXh2WVhRZ2RDQTlJR05zWVcxd0tHVnNZWEJ6WldRZ0tpQW9ORFF4TURB'
    'dU1DQXZJREU1TWk0d0tTd2dNQzR3TENBeExqQXBPd29nSUNBZ0lDQWdJR1Js'
    'WTJ4cFkyc2dQU0IwSUNvZ2RDQXFJQ2d6TGpBZ0xTQXlMakFnS2lCMEtUc0tJ'
    'Q0FnSUgwZ1pXeHpaU0I3Q2lBZ0lDQWdJQ0FnTHk4Z1UyaGhjbkFnTmpRdGMy'
    'RnRjR3hsSUcxcGEwbFVJR1JsWm1GMWJIUWdabTl5SUc1dmNtMWhiQ0IwY21s'
    'bloyVnlMV1p5YjIwdGMyRnRjR3hsTFRBdUNpQWdJQ0FnSUNBZ1pHVmpiR2xq'
    'YXlBOUlHTnNZVzF3S0dWc1lYQnpaV1FnS2lBb05EUXhNREF1TUNBdklEWTBM'
    'akFwTENBd0xqQXNJREV1TUNrN0NpQWdJQ0I5Q2dvZ0lDQWdMeThnTWk0Z1JX'
    'NWtMVzltTFhOaGJYQnNaU0JtWVdSbExXOTFkRG9nTmpRdGMyRnRjR3hsSUda'
    'aFpHVXRiM1YwSUdGeklHWlRZVzF3YkdWUWIzTWdZWEJ3Y205aFkyaGxjd29n'
    'SUNBZ0x5OGdJQ0FnYzJGdGNHeGxJR1Z1WkNBb2IyNXNlU0JtYjNJZ2JtOXVM'
    'V3h2YjNCcGJtY2djMkZ0Y0d4bGN5a3VJQ0JRY21WMlpXNTBjeUJ6ZFdSa1pX'
    'NGdjMmxzWlc1alpTNEtJQ0FnSUdac2IyRjBJR1Z1WkVaaFpHVWdQU0F4TGpB'
    'N0NpQWdJQ0JwWmlBb2MyMXdMbXh2YjNCTVpXNGdQRDBnTWlrZ2V3b2dJQ0Fn'
    'SUNBZ0lHWnNiMkYwSUhKbGJXRnBibWx1WnlBOUlHWnNiMkYwS0hOdGNDNXNa'
    'VzVuZEdncElDMGdabE5oYlhCc1pWQnZjenNLSUNBZ0lDQWdJQ0JwWmlBb2Nt'
    'VnRZV2x1YVc1bklEd2dOalF1TUNrZ1pXNWtSbUZrWlNBOUlHMWhlQ2d3TGpB'
    'c0lISmxiV0ZwYm1sdVp5QXZJRFkwTGpBcE93b2dJQ0FnZlFvS0lDQWdJQzh2'
    'SURNdUlFeHZiM0FnWTNKdmMzTm1ZV1JsT2lCemJXOXZkR2h6SUdGdWVTQnla'
    'WE5wWkhWaGJDQnNiMjl3Ulc1azRvYVNiRzl2Y0ZOMFlYSjBJR1JwYzJOdmJu'
    'UnBiblZwZEhrdUNpQWdJQ0F2THlBZ0lDQlVhR1VnWlc1amIyUmxjaUJ1YjNj'
    'Z1pXMWlaV1J6SUd4dmIzQWdkM0poY0NCamIyNTBaWGgwSUc1bGVIUWdkRzhn'
    'Ykc5dmNFVnVaQ0J6YnlCV1VRb2dJQ0FnTHk4Z0lDQWdjWFZoYm5ScGVtRjBh'
    'Vzl1SUd0bFpYQnpJSFJvWlNCelpXRnRJR052Ym5ScGJuVnZkWE1zSUdKMWRD'
    'QmhJREUyTFhOaGJYQnNaU0JqY205emMyWmhaR1VLSUNBZ0lDOHZJQ0FnSUdO'
    'aGRHTm9aWE1nWVc1NUlISmxiV0ZwYm1sdVp5QnRhWE50WVhSamFDNEtJQ0Fn'
    'SUM4dklPS1VnT0tVZ0NCTWIyOXdJR055YjNOelptRmtaU0JFU1ZOQlFreEZS'
    'Q0RpbElEaWxJQUtJQ0FnSUM4dklGUm9aU0J3Y21WMmFXOTFjeUF4TmkxellX'
    'MXdiR1VnWlhGMVlXd3RjRzkzWlhJZ1kzSnZjM05tWVdSbElIZGhjeUJoSUU1'
    'RlZDQklRVkpOTENCdWIzUWdZU0JtYVhndUNpQWdJQ0F2THlCSmRDQnlaV0Zr'
    'SUdFZ2QzSmhjQzF3YjNNZ2MyRnRjR3hsSUdGMElHeHZiM0JUZEdGeWRDQXJJ'
    'Q2hEVWs5VFUwWkJSRVZmVEVWT0lDMGdaR2x6ZEVaeWIyMUZibVFwQ2lBZ0lD'
    'QXZMeUJoYm1RZ1lteGxibVJsWkNCcGRDQkpUbFJQSUhSb1pTQndiR0Y1WW1G'
    'amF5QmhjeUIzWlNCaGNIQnliMkZqYUdWa0lHeHZiM0JGYm1Rc0lIUm9aVzRn'
    'WkhKdmNIQmxaQW9nSUNBZ0x5OGdkR2hsSUdKc1pXNWtJSGRvWlc0Z1psTmhi'
    'WEJzWlZCdmN5QjNjbUZ3Y0dWa0lIUnZJR3h2YjNCVGRHRnlkQzRnVW1WemRX'
    'eDBPaUJoSUdkMVlYSmhiblJsWldRS0lDQWdJQzh2SUhOcGJtZHNaUzF6WVcx'
    'd2JHVWdaR2x6WTI5dWRHbHVkV2wwZVNCbGRtVnllU0JzYjI5d0lHbDBaWEpo'
    'ZEdsdmJpRGlnSlFnYldGbmJtbDBkV1JsSUhOallXeHBibWNLSUNBZ0lDOHZJ'
    'SGRwZEdnZ2RHaGxJR1JwWm1abGNtVnVZMlVnWW1WMGQyVmxiaUFvYkc5dmND'
    'MWxibVF0Y21WbmFXOXVJSEpsWVdRcElHRnVaQ0FvZDNKaGNGQnZjeUJ5WldG'
    'a0tTd0tJQ0FnSUM4dklIZG9hV05vSUdadmNpQjBlWEJwWTJGc0lITjFjM1Jo'
    'YVc0Z2JHOXZjSE1nYVhNZ1lTQnpkV0p6ZEdGdWRHbGhiQ0JtY21GamRHbHZi'
    'aUJ2WmlCMGFHVUtJQ0FnSUM4dklITnBaMjVoYkM0Z1ZHaHBjeUJqY21WaGRH'
    'VmtJSE4wY205dVp5Qm9ZWEp0YjI1cFkzTWdZWFFnZEdobElHeHZiM0FnWm5W'
    'dVpHRnRaVzUwWVd3Z2NtRjBaUW9nSUNBZ0x5OGdLR1V1Wnk0Z016Y3hJRWg2'
    'SUdGMElFWWpOQ0IzYVhSb0lITmhiWEJzWlNBMElDOGdNekl0YzJGdGNHeGxJ'
    'R3h2YjNBcExDQmhkV1JwWW14bElHRnpJR0oxZW5vdUNpQWdJQ0F2THdvZ0lD'
    'QWdMeThnVEdGdVkzcHZjeTB6SUdadmNuZGhjbVF0ZEdGd0lHeHZiM0FnZDNK'
    'aGNDQW9hVzRnWjJWMFUyRnRjR3hsUmlrZ2FHRnVaR3hsY3lCMGFHVWdjMlZo'
    'YlNCcGRITmxiR1l1Q2dvZ0lDQWdMeThnNHBTQTRwU0FJRk50YjI5MGFHVmtJ'
    'SFp2YkhWdFpTQmhjSEJzYVdOaGRHbHZiaURpbElEaWxJRGlsSURpbElEaWxJ'
    'RGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElE'
    'aWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURp'
    'bElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSUFLSUNBZ0lDOHZJ'
    'RU52YlhCMWRHVWdkR2hsSUhKaGJYQWdabUZqZEc5eUlHSmhjMlZrSUc5dUlH'
    'aHZkeUJ0WVc1NUlITmhiWEJzWlhNZ2FHRjJaU0JsYkdGd2MyVmtJSE5wYm1O'
    'bENpQWdJQ0F2THlCMGFHVWdiVzl6ZENCeVpXTmxiblFnZG05c2RXMWxJR05v'
    'WVc1blpTNGdVMEZOVUV4RlUxOVFSVkpmVkVsRFN5QnBjeUEwTkRFd01DOVVT'
    'VU5MVTE5UVJWSmZVMFZEQ2lBZ0lDQXZMeUFvZEhsd2FXTmhiR3g1SURnNE1p'
    'QmhkQ0JVU1VOTFUxOVFSVkpmVTBWRFBUVXdLUzRnVW1GdGNDQmpiMjF3YkdW'
    'MFpYTWdiM1psY2lBMk5DQnpZVzF3YkdWekNpQWdJQ0F2THlEaWlZZ2dNUzQw'
    'TlcxeklPS0FsQ0JtWVhOMElHVnViM1ZuYUNCMGJ5QmlaU0JwYlhCbGNtTmxj'
    'SFJwWW14bElHRnpJR0VnWm1Ga1pTMXBiaUJpZFhRZ2MyeHZkeUJsYm05MVoy'
    'Z0tJQ0FnSUM4dklIUnZJR2hwWkdVZ2RHaGxJSEJsY2kxellXMXdiR1VnYzNS'
    'bGNDQjBhR0YwSUhCeWIyUjFZMlZ6SUhSb1pTQmpiR2xqYXk0S0lDQWdJQzh2'
    'Q2lBZ0lDQXZMeUJ3YjNNdWRHbGphMFlnUFNCMGFHVWdaMnh2WW1Gc0lIUnBZ'
    'MnNnYjJZZ2RHaGxJR04xY25KbGJuUWdjMkZ0Y0d4bExDQm1jbUZqZEdsdmJt'
    'RnNMZ29nSUNBZ2FXNTBJRjlqZFhKU2IzZFRaM0lnUFNCd1lYUlVhV05yVDJa'
    'bWMyVjBXM0J2Y3k1emIyNW5VRzl6WFNBcklDaHdiM011Y205M0lDMGdjR0Yw'
    'VTNSaGNuUlNiM2RiY0c5ekxuTnZibWRRYjNOZEtUc0tJQ0FnSUdac2IyRjBJ'
    'Rjl3YjNOVWFXTnJSaUE5SUdac2IyRjBLR1psZEdOb1ZHbGpheWhmWTNWeVVt'
    'OTNVMmR5S1NrZ0t5QndiM011ZEdsamF6c0tJQ0FnSUdac2IyRjBJRjlUUVUx'
    'UVRFVlRYMUJGVWw5VVNVTkxJRDBnTkRReE1EQXVNQ0F2SUZSSlEwdFRYMUJG'
    'VWw5VFJVTTdDaUFnSUNCbWJHOWhkQ0JmZGxKaGJYQWdQU0JqYkdGdGNDZ29Y'
    'M0J2YzFScFkydEdJQzBnWDNadmJFTm9ZVzVuWlVGMFZHbGphMFlwSUNvZ1gx'
    'TkJUVkJNUlZOZlVFVlNYMVJKUTBzZ0x5QTJOQzR3TENBd0xqQXNJREV1TUNr'
    'N0NpQWdJQ0JtYkc5aGRDQmZaV1ptVm05c0lEMGdiV2w0S0dac2IyRjBLRjky'
    'YjJ4UWNtVjJLU3dnWm14dllYUW9YM1p2YkVOMWNuSXBMQ0JmZGxKaGJYQXBP'
    'd29nSUNBZ0x5OGdWSEpsYlc5c2J5QmhjSEJzYVdWeklHOXVJSFJ2Y0NCdlpp'
    'QjBhR1VnYzIxdmIzUm9aV1FnZG05c2RXMWxMQ0IwYUdWdUlHTnNZVzF3SUhS'
    'dklEQXVMalkwTGdvZ0lDQWdYMlZtWmxadmJDQTlJR05zWVcxd0tGOWxabVpX'
    'YjJ3Z0t5QmZkSEpsYlc5c2IwUmxiSFJoTENBd0xqQXNJRFkwTGpBcE93b0tJ'
    'Q0FnSUhKbGRIVnliaUJ6SUNvZ0tGOWxabVpXYjJ3Z0x5QTJOQzR3S1NBcUlH'
    'UmxZMnhwWTJzZ0tpQmxibVJHWVdSbE93cDlDZ29LTHk4ZzRwU0E0cFNBNHBT'
    'QTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNB'
    'NHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0'
    'cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRw'
    'U0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBT'
    'QTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNB'
    'NHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0FD'
    'aTh2SUdkbGRFTm9ZVzV1Wld4UGRYUndkWFFnNG9DVUlIQjFZbXhwWXlCbGJu'
    'UnllUzRnVW5WdWN5QjBjbWxuWjJWeUlITmxZWEpqYUN3Z2RHaGxiaUIwYUdV'
    'Z1ltOWtlUzRLTHk4Z1JtOXlJSFJvWlNCbWFYSnpkQ0EyTkNCellXMXdiR1Z6'
    'SUdGbWRHVnlJR0VnY21WMGNtbG5aMlZ5TENCQlRGTlBJSEpsYm1SbGNpQjNh'
    'WFJvSUhSb1pRb3ZMeUJ3Y21WMmFXOTFjeUIwY21sbloyVnlJR0Z1WkNCaWJH'
    'VnVaQ0RpZ0pRZ2RHaHBjeUJwY3lCMGFHVWdjSEpsZG1sdmRYTXRibTkwWlNC'
    'amNtOXpjMlpoWkdVS0x5OGdkR2hoZENCbGJHbHRhVzVoZEdWeklHbHVkR1Z5'
    'TFc1dmRHVWdZMnhwWTJ0eklDaHRZWFJqYUdWeklIUm9aU0JrZVdsdVoxdDBY'
    'U0FySUdOb1lXNXVaV3hiZEYwS0x5OGdZM0p2YzNObVlXUmxJR2x1SUUxcGEw'
    'MXZaQ2R6SUUxRVVsWmZUVWxZTGtOUVVDa3VDaTh2SU9LVWdPS1VnT0tVZ09L'
    'VWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tV'
    'Z09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1Vn'
    'T0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdP'
    'S1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09L'
    'VWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tV'
    'Z09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdBcG1i'
    'RzloZENCblpYUkRhR0Z1Ym1Wc1QzVjBjSFYwS0dsdWRDQmphQ3dnWm14dllY'
    'UWdkR2x0WlN3Z1VHOXphWFJwYjI0Z2NHOXpMQ0JtYkc5aGRDQnliM2RVYVcx'
    'bEtTQjdDZ29nSUNBZ0x5OGdVM1JsY0NBeE9pQm1hVzVrSUcxdmMzUXRjbVZq'
    'Wlc1MGJIa3RkSEpwWjJkbGNtVmtJRzV2ZEdVZ2IyNGdkR2hwY3lCamFHRnVi'
    'bVZzTGdvZ0lDQWdMeThnVUZRZ2MyVnRZVzUwYVdOeklPS0FsQ0JoSUNKMGNt'
    'bG5aMlZ5SWlCcGN5QnpiMjFsZEdocGJtY2dkR2hoZENCemRHRnlkSE1nZEdo'
    'bElITmhiWEJzWlNCaGRDQndiM01nTURvS0lDQWdJQzh2SUNBZzRvQ2lJRVox'
    'Ykd3Z2NtOTNJQ2hwYm5OMGNuVnRaVzUwSUNzZ2NHVnlhVzlrS1NBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWc0b0NVSUhKbGRISnBaMmRsY2dvZ0lDQWdMeThnSUNEaWdL'
    'SWdVR1Z5YVc5a0xXOXViSGtnY205M0lDaHVieUJwYm5OMExDQnVieUJsWm1a'
    'bFkzUWdNeTgxS1NBZ0lDRGlnSlFnY21WMGNtbG5aMlZ5TENCcGJtaGxjbWww'
    'SUdsdWMzUnlkVzFsYm5RS0lDQWdJQzh2SUNBZzRvQ2lJRkJsY21sdlpDMXZi'
    'bXg1SUhkcGRHZ2daV1ptWldOMElETXZOU0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWc0b0NVSUhOc2FXUmxJSFJoY21kbGRDQnZibXg1TENCdWJ5QnlaWFJ5YVdk'
    'blpYSUtJQ0FnSUM4dklDQWc0b0NpSUVaMWJHd2djbTkzSUhkcGRHZ2daV1pt'
    'WldOMElETXZOU0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnNG9DVUlITnNh'
    'V1JsSUhSaGNtZGxkQ0J2Ym14NUxDQnVieUJ5WlhSeWFXZG5aWElLSUNBZ0lD'
    'OHZJQ0FnNG9DaUlFVnRjSFI1SUM4Z2FXNXpkSEoxYldWdWRDMXZibXg1SUNB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZzRvQ1VJR052Ym5ScGJuVmxJSEJ5'
    'YVc5eUlHNXZkR1VLSUNBZ0lFNXZkR1VnWDJOMWNsSnZkeUE5SUdkbGRFNXZk'
    'R1VvY0c5ekxuTnZibWRRYjNNc0lIQnZjeTV5YjNjc0lHTm9LVHNLSUNBZ0lF'
    'NXZkR1VnZEhKcFowNXZkR1VnUFNCZlkzVnlVbTkzT3dvZ0lDQWdhVzUwSUNC'
    'MGNtbG5VbTkzSUNBOUlIQnZjeTV5YjNjN0NpQWdJQ0JwYm5RZ0lIUnlhV2RR'
    'WVhRZ0lEMGdjRzl6TG5OdmJtZFFiM003Q2lBZ0lDQnBiblFnSUhSdmJtVlRi'
    'R2xrWlZSaGNtZGxkQ0E5SURBN0lDQXZMeUIzYUdWdUlITmxkQ3dnZEdocGN5'
    'QnliM2NnWTJGeWNtbGxjeUJoSURONGVDODFlSGdnYzJ4cFpHVWdkR0Z5WjJW'
    'MENpQWdJQ0JpYjI5c0lGOWpkWEpKYzFSdmJtVlFiM0owWVNBOUlDZ29YMk4x'
    'Y2xKdmR5NWxabVpsWTNRZ1BUMGdNSGd6SUh4OElGOWpkWEpTYjNjdVpXWm1a'
    'V04wSUQwOUlEQjROU2tnSmlZS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lGOWpkWEpTYjNjdWNHVnlhVzlrSUQ0Z01DazdDaUFnSUNC'
    'aWIyOXNJRjlqZFhKSmMxSmxkSEpwWnlBZ0lDQTlJQ2hmWTNWeVVtOTNMbkJs'
    'Y21sdlpDQStJREFnSmlZZ0lWOWpkWEpKYzFSdmJtVlFiM0owWVNrN0lDQXZM'
    'eUJoYm5rZ2NHVnlhVzlrSUhkcGRHaHZkWFFnTXk4MUlISmxkSEpwWjJkbGNu'
    'TUtJQ0FnSUdKdmIyd2dYMk4xY2toaGMwbHVjM1FnSUNBZ0lEMGdLRjlqZFhK'
    'U2IzY3VhVzV6ZEhKMWJXVnVkQ0ErSURBcE93b0tJQ0FnSUdsbUlDaGZZM1Z5'
    'U1hOVWIyNWxVRzl5ZEdFcElIc0tJQ0FnSUNBZ0lDQXZMeUJUYkdsa1pTQjBZ'
    'WEpuWlhRZzRvQ1VJR1pwYm1RZ2NISnBiM0lnVWtWQlRDQjBjbWxuWjJWeUlH'
    'WnZjaUJ6WVcxd2JHVXZjR1Z5YVc5a0lHTnZiblJsZUhRS0lDQWdJQ0FnSUNC'
    'MGIyNWxVMnhwWkdWVVlYSm5aWFFnUFNCZlkzVnlVbTkzTG5CbGNtbHZaRHNL'
    'SUNBZ0lDQWdJQ0JwYm5RZ2MxSWdQU0J3YjNNdWNtOTNMQ0J6VUNBOUlIQnZj'
    'eTV6YjI1blVHOXpPd29nSUNBZ0lDQWdJR1p2Y2lBb2FXNTBJR3hpSUQwZ01U'
    'c2diR0lnUENBeE1qZzdJR3hpS3lzcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnYzFJ'
    'dExUc0tJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tITlNJRHdnTUNrZ2V3b2dJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLSE5RSUQ0Z01Da2dld29nSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUhOUUxTMDdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnYzFJZ1BTQndZWFJUZEdGeWRGSnZkMXR6VUYwZ0t5QW9jR0YwVW05'
    'M1QyWm1jMlYwVzNOUUt6RmRJQzBnY0dGMFVtOTNUMlptYzJWMFczTlFYU2tn'
    'TFNBeE93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2ZTQmxiSE5sSUhzZ1luSmxZ'
    'V3M3SUgwS0lDQWdJQ0FnSUNBZ0lDQWdmUW9nSUNBZ0lDQWdJQ0FnSUNCT2Iz'
    'UmxJSEJ5WlhZZ1BTQm5aWFJPYjNSbEtITlFMQ0J6VWl3Z1kyZ3BPd29nSUNB'
    'Z0lDQWdJQ0FnSUNCaWIyOXNJSEJ5WlhaSmMxUnZibVZVY21sbklEMGdLQ2h3'
    'Y21WMkxtVm1abVZqZENBOVBTQXdlRE1nZkh3Z2NISmxkaTVsWm1abFkzUWdQ'
    'VDBnTUhnMUtTQW1KaUJ3Y21WMkxuQmxjbWx2WkNBK0lEQXBPd29nSUNBZ0lD'
    'QWdJQ0FnSUNBdkx5QlNaV0ZzSUhSeWFXZG5aWEk2SUdoaGN5QndaWEpwYjJR'
    'Z1FVNUVJRzV2ZENCaElIUnZibVV0Y0c5eWRHRWdkR0Z5WjJWMElISnZkd29n'
    'SUNBZ0lDQWdJQ0FnSUNCcFppQW9jSEpsZGk1d1pYSnBiMlFnUGlBd0lDWW1J'
    'Q0Z3Y21WMlNYTlViMjVsVkhKcFp5a2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdMeThnUkdWMFpYSnRhVzVsSUdsdWMzUnlkVzFsYm5RNklIQnlaV1psY2lC'
    'd2NtVjJMbWx1YzNSeWRXMWxiblFzSUdWc2MyVWdjMk5oYmlCbWRYSjBhR1Z5'
    'SUdadmNpQmpiMjUwWlhoMENpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9j'
    'SEpsZGk1cGJuTjBjblZ0Wlc1MElENGdNQ2tnZXdvZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lIUnlhV2RPYjNSbElEMGdjSEpsZGpzZ2RISnBaMUp2ZHlB'
    'OUlITlNPeUIwY21sblVHRjBJRDBnYzFBN0NpQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNCOUlHVnNjMlVnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDOHZJ'
    'RkJsY21sdlpDMXZibXg1SUhKdmR5RGlnSlFnWm1sdVpDQnBibk4wY25WdFpX'
    'NTBJR052Ym5SbGVIUUtJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JPYjNS'
    'bElISmxZV3dnUFNCd2NtVjJPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUdsdWRDQnpVaklnUFNCelVpd2djMUF5SUQwZ2MxQTdDaUFnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnWm05eUlDaHBiblFnYkdJeUlEMGdNVHNnYkdJeUlE'
    'd2dNVEk0T3lCc1lqSXJLeWtnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0J6VWpJdExUc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ2FXWWdLSE5TTWlBOElEQXBJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUdsbUlDaHpVRElnUGlBd0tTQjdJSE5RTWkwdE95'
    'QnpVaklnUFNCd1lYUlRkR0Z5ZEZKdmQxdHpVREpkSUNzZ0tIQmhkRkp2ZDA5'
    'bVpuTmxkRnR6VURJck1WMGdMU0J3WVhSU2IzZFBabVp6WlhSYmMxQXlYU2tn'
    'TFNBeE95QjlDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0JsYkhObElHSnlaV0ZyT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0I5Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJRTV2ZEdV'
    'Z2NESWdQU0JuWlhST2IzUmxLSE5RTWl3Z2MxSXlMQ0JqYUNrN0NpQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsbUlDaHdNaTVwYm5OMGNuVnRa'
    'VzUwSUQ0Z01Da2dleUJ5WldGc0xtbHVjM1J5ZFcxbGJuUWdQU0J3TWk1cGJu'
    'TjBjblZ0Wlc1ME95QmljbVZoYXpzZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCMGNtbG5UbTkw'
    'WlNBOUlISmxZV3c3SUhSeWFXZFNiM2NnUFNCelVqc2dkSEpwWjFCaGRDQTlJ'
    'SE5RT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnWW5KbFlXczdDaUFnSUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNC'
    'OUNpQWdJQ0I5SUdWc2MyVWdhV1lnS0NGZlkzVnlTWE5TWlhSeWFXY3BJSHNL'
    'SUNBZ0lDQWdJQ0F2THlCT2J5QndaWEpwYjJRZzRvQ1VJR052Ym5ScGJuVmxJ'
    'SEJ5YVc5eUlHNXZkR1VnS0c5eUlHNXZJR0YxWkdsdklHbG1JRzV2ZEdocGJt'
    'Y2djSEpwYjNJcExnb2dJQ0FnSUNBZ0lDOHZJRjlqZFhKSVlYTkpibk4wSUhk'
    'cGRHZ2dibThnY0dWeWFXOWtJR2x6SUdFZ2JtOHRiM0FnWm05eUlIUnlhV2Ru'
    'WlhJZ2NIVnljRzl6WlhNZ0tGQlVJSEYxYVhKcktTNEtJQ0FnSUNBZ0lDQnBi'
    'blFnYzFJZ1BTQndiM011Y205M0xDQnpVQ0E5SUhCdmN5NXpiMjVuVUc5ek93'
    'b2dJQ0FnSUNBZ0lHWnZjaUFvYVc1MElHeGlJRDBnTVRzZ2JHSWdQQ0F4TWpn'
    'N0lHeGlLeXNwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdjMUl0TFRzS0lDQWdJQ0Fn'
    'SUNBZ0lDQWdhV1lnS0hOU0lEd2dNQ2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnYVdZZ0tITlFJRDRnTUNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJSE5RTFMwN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdjMUlnUFNC'
    'd1lYUlRkR0Z5ZEZKdmQxdHpVRjBnS3lBb2NHRjBVbTkzVDJabWMyVjBXM05R'
    'S3pGZElDMGdjR0YwVW05M1QyWm1jMlYwVzNOUVhTa2dMU0F4T3dvZ0lDQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnZlNCbGJITmxJSHNnWW5KbFlXczdJSDBLSUNBZ0lD'
    'QWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0JPYjNSbElIQnlaWFlnUFNC'
    'blpYUk9iM1JsS0hOUUxDQnpVaXdnWTJncE93b2dJQ0FnSUNBZ0lDQWdJQ0Jp'
    'YjI5c0lIQnlaWFpKYzFSdmJtVlVjbWxuSUQwZ0tDaHdjbVYyTG1WbVptVmpk'
    'Q0E5UFNBd2VETWdmSHdnY0hKbGRpNWxabVpsWTNRZ1BUMGdNSGcxS1NBbUpp'
    'QndjbVYyTG5CbGNtbHZaQ0ErSURBcE93b2dJQ0FnSUNBZ0lDQWdJQ0JwWmlB'
    'b2NISmxkaTV3WlhKcGIyUWdQaUF3SUNZbUlDRndjbVYyU1hOVWIyNWxWSEpw'
    'WnlrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLSEJ5WlhZdWFXNXpk'
    'SEoxYldWdWRDQStJREFwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QjBjbWxuVG05MFpTQTlJSEJ5WlhZN0lIUnlhV2RTYjNjZ1BTQnpVanNnZEhK'
    'cFoxQmhkQ0E5SUhOUU93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2ZTQmxiSE5s'
    'SUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQk9iM1JsSUhKbFlXd2dQ'
    'U0J3Y21WMk93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2x1ZENCelVq'
    'SWdQU0J6VWl3Z2MxQXlJRDBnYzFBN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdabTl5SUNocGJuUWdiR0l5SUQwZ01Uc2diR0l5SUR3Z01USTRPeUJz'
    'WWpJckt5a2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnpV'
    'akl0TFRzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tI'
    'TlNNaUE4SURBcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJR2xtSUNoelVESWdQaUF3S1NCN0lITlFNaTB0T3lCelVqSWdQU0J3'
    'WVhSVGRHRnlkRkp2ZDF0elVESmRJQ3NnS0hCaGRGSnZkMDltWm5ObGRGdHpV'
    'RElyTVYwZ0xTQndZWFJTYjNkUFptWnpaWFJiYzFBeVhTa2dMU0F4T3lCOUNp'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmxiSE5sSUdK'
    'eVpXRnJPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQjlDaUFn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lFNXZkR1VnY0RJZ1BTQm5a'
    'WFJPYjNSbEtITlFNaXdnYzFJeUxDQmphQ2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJR2xtSUNod01pNXBibk4wY25WdFpXNTBJRDRnTUNr'
    'Z2V5QnlaV0ZzTG1sdWMzUnlkVzFsYm5RZ1BTQndNaTVwYm5OMGNuVnRaVzUw'
    'T3lCaWNtVmhhenNnZlFvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lIMEtJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0IwY21sblRtOTBaU0E5SUhKbFlX'
    'dzdJSFJ5YVdkU2IzY2dQU0J6VWpzZ2RISnBaMUJoZENBOUlITlFPd29nSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdmUW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdZbkps'
    'WVdzN0NpQWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0I5Q2lBZ0lDQjlJ'
    'R1ZzYzJVZ2FXWWdLRjlqZFhKSmMxSmxkSEpwWnlBbUppQWhYMk4xY2toaGMw'
    'bHVjM1FwSUhzS0lDQWdJQ0FnSUNBdkx5QlFaWEpwYjJRdGIyNXNlU0J5WlhS'
    'eWFXZG5aWElnNG9DVUlHWnBibVFnYVc1emRISjFiV1Z1ZENCamIyNTBaWGgw'
    'SUNoellXMXdiR1VnYVc1b1pYSnBkR1ZrSUdaeWIyMGdjSEpwYjNJZ2RISnBa'
    'MmRsY2lrS0lDQWdJQ0FnSUNCcGJuUWdjMUlnUFNCd2IzTXVjbTkzTENCelVD'
    'QTlJSEJ2Y3k1emIyNW5VRzl6T3dvZ0lDQWdJQ0FnSUdadmNpQW9hVzUwSUd4'
    'aUlEMGdNVHNnYkdJZ1BDQXhNamc3SUd4aUt5c3BJSHNLSUNBZ0lDQWdJQ0Fn'
    'SUNBZ2MxSXRMVHNLSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLSE5TSUR3Z01Da2dl'
    'd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0hOUUlENGdNQ2tnZXdvZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lITlFMUzA3Q2lBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ2MxSWdQU0J3WVhSVGRHRnlkRkp2ZDF0elVGMGdLeUFv'
    'Y0dGMFVtOTNUMlptYzJWMFczTlFLekZkSUMwZ2NHRjBVbTkzVDJabWMyVjBX'
    'M05RWFNrZ0xTQXhPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdmU0JsYkhObElI'
    'c2dZbkpsWVdzN0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNB'
    'Z0lDQk9iM1JsSUhCeVpYWWdQU0JuWlhST2IzUmxLSE5RTENCelVpd2dZMmdw'
    'T3dvZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvY0hKbGRpNXBibk4wY25WdFpXNTBJ'
    'RDRnTUNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2RISnBaMDV2ZEdVdWFX'
    'NXpkSEoxYldWdWRDQTlJSEJ5WlhZdWFXNXpkSEoxYldWdWREc0tJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lHSnlaV0ZyT3dvZ0lDQWdJQ0FnSUNBZ0lDQjlDaUFn'
    'SUNBZ0lDQWdmUW9nSUNBZ0lDQWdJQzh2SUhSeWFXZFFZWFF2ZEhKcFoxSnZk'
    'eUJ6ZEdGNUlHRjBJR04xY25KbGJuUWdjbTkzSU9LQWxDQjBhR2x6SUVsVElH'
    'RWdjbVYwY21sbloyVnlDaUFnSUNCOUNpQWdJQ0F2THlCbGJITmxPaUJtZFd4'
    'c0lIUnlhV2RuWlhJZ0tIQmxjbWx2WkNBcklHbHVjM1J5ZFcxbGJuUXNJRzV2'
    'SURNdk5Ta2c0b0NVSUhSeWFXZE9iM1JsSUdGc2NtVmhaSGtnWTI5eWNtVmpk'
    'QW9LSUNBZ0lDOHZJT0tVZ09LVWdDQlNaVzVrWlhJZ2QybDBhQ0JqZFhKeVpX'
    'NTBJSFJ5YVdkblpYSWc0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBT'
    'QTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNB'
    'NHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0'
    'cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBQ2lBZ0lDQm1iRzloZENCelgy'
    'TjFjbklnUFNCZloyTnZRbTlrZVNoamFDd2djRzl6TENCMGFXMWxMQ0J5YjNk'
    'VWFXMWxMQ0IwY21sblVHRjBMQ0IwY21sblVtOTNMQ0IwY21sblRtOTBaU3dn'
    'ZEc5dVpWTnNhV1JsVkdGeVoyVjBLVHNLQ2lBZ0lDQXZMeURpbElEaWxJQWdR'
    'M0p2YzNObVlXUmxJSGRwYm1SdmR5QmphR1ZqYXlEaWxJRGlsSURpbElEaWxJ'
    'RGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElE'
    'aWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURp'
    'bElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGls'
    'SURpbElEaWxJRGlsSURpbElBS0lDQWdJQzh2SUVOdmJYQjFkR1VnYzJGdGNH'
    'eGxjeUJsYkdGd2MyVmtJSE5wYm1ObElIUm9aU0JqZFhKeVpXNTBJSFJ5YVdk'
    'blpYSWdabWx5WldRdUlFOXViSGtnYVc1emFXUmxDaUFnSUNBdkx5QjBhR1Vn'
    'Wm1seWMzUWdOalFnYzJGdGNHeGxjeUJwY3lCMGFHVWdZM0p2YzNObVlXUmxJ'
    'RzFsWVc1cGJtZG1kV3dnNG9DVUlHSmxlVzl1WkNCMGFHRjBMQ0IwYUdVS0lD'
    'QWdJQzh2SUhCeVpYWnBiM1Z6SUc1dmRHVWdhR0Z6SUd4dmJtY2dabUZrWldR'
    'Z2IzVjBMZ29nSUNBZ1pteHZZWFFnWTNWeVZISnBaMVJwYldWR0lEMGdabXh2'
    'WVhRb1ptVjBZMmhVYVdOcktIQmhkRlJwWTJ0UFptWnpaWFJiZEhKcFoxQmhk'
    'RjBnS3lBb2RISnBaMUp2ZHlBdElIQmhkRk4wWVhKMFVtOTNXM1J5YVdkUVlY'
    'UmRLU2twQ2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdMeUJVU1VO'
    'TFUxOVFSVkpmVTBWRE93b2dJQ0FnWm14dllYUWdZV2RsVTJGdGNHeGxjeUE5'
    'SUNoMGFXMWxJQzBnWTNWeVZISnBaMVJwYldWR0tTQXFJRFEwTVRBd0xqQTdD'
    'Z29nSUNBZ2FXWWdLR0ZuWlZOaGJYQnNaWE1nUENBMk5DNHdJQ1ltSUdGblpW'
    'TmhiWEJzWlhNZ1BqMGdNQzR3S1NCN0NpQWdJQ0FnSUNBZ0x5OGc0cFNBNHBT'
    'QUlGTmxZWEpqYUNCbWIzSWdkR2hsSUZCU1JWWkpUMVZUSUhSeWFXZG5aWEln'
    'S0c5dVpTQnliM2NnWW1WbWIzSmxJR04xY25KbGJuUXBJT0tVZ09LVWdPS1Vn'
    'T0tVZ0FvZ0lDQWdJQ0FnSUM4dklGTmhiV1VnWVd4bmIzSnBkR2h0SUdGeklI'
    'Um9aU0J0WVdsdUlIUnlhV2RuWlhJZ2MyVmhjbU5vSUdKMWRDQnpkR0Z5ZEds'
    'dVp5QnZibVVnY205M0NpQWdJQ0FnSUNBZ0x5OGdaV0Z5YkdsbGNpNGdVMlYw'
    'Y3lCd1ZISnBaMUJoZEM5d1ZISnBaMUp2ZHk5d1ZISnBaMDV2ZEdVdUNpQWdJ'
    'Q0FnSUNBZ2FXNTBJQ0J3VkhKcFoxQmhkQ0E5SUMweExDQndWSEpwWjFKdmR5'
    'QTlJQzB4T3dvZ0lDQWdJQ0FnSUU1dmRHVWdjRlJ5YVdkT2IzUmxPd29nSUNB'
    'Z0lDQWdJSHNLSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJSE5TSUQwZ2RISnBaMUp2'
    'ZHl3Z2MxQWdQU0IwY21sblVHRjBPd29nSUNBZ0lDQWdJQ0FnSUNCbWIzSWdL'
    'R2x1ZENCc1lpQTlJREU3SUd4aUlEd2dNVEk0T3lCc1lpc3JLU0I3Q2lBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0J6VWkwdE93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z2FXWWdLSE5TSUR3Z01Da2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUdsbUlDaHpVQ0ErSURBcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ2MxQXRMVHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdjMUlnUFNCd1lYUlRkR0Z5ZEZKdmQxdHpVRjBnS3lBb2NHRjBVbTkzVDJa'
    'bWMyVjBXM05RS3pGZElDMGdjR0YwVW05M1QyWm1jMlYwVzNOUVhTa2dMU0F4'
    'T3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lIMGdaV3h6WlNCN0lHSnla'
    'V0ZyT3lCOUNpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNCT2IzUmxJSEJ5WlhZZ1BTQm5aWFJPYjNSbEtITlFMQ0J6VWl3'
    'Z1kyZ3BPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdZbTl2YkNCd2NtVjJTWE5V'
    'YjI1bFZISnBaeUE5SUNnb2NISmxkaTVsWm1abFkzUWdQVDBnTUhneklIeDhJ'
    'SEJ5WlhZdVpXWm1aV04wSUQwOUlEQjROU2tnSmlZZ2NISmxkaTV3WlhKcGIy'
    'UWdQaUF3S1RzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsbUlDaHdjbVYyTG5C'
    'bGNtbHZaQ0ErSURBZ0ppWWdJWEJ5WlhaSmMxUnZibVZVY21sbktTQjdDaUFn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tIQnlaWFl1YVc1emRISjFi'
    'V1Z1ZENBK0lEQXBJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdjRlJ5YVdkT2IzUmxJRDBnY0hKbGRqc2djRlJ5YVdkU2IzY2dQU0J6VWpz'
    'Z2NGUnlhV2RRWVhRZ1BTQnpVRHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNCOUlHVnNjMlVnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0F2THlCUVpYSnBiMlF0YjI1c2VTRGlnSlFnWm1sdVpDQnBibk4wY25WdFpX'
    'NTBJR052Ym5SbGVIUUtJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z1RtOTBaU0J5WldGc0lEMGdjSEpsZGpzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnYVc1MElITlNNaUE5SUhOU0xDQnpVRElnUFNCelVEc0tJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1ptOXlJQ2hwYm5RZ2JH'
    'SXlJRDBnTVRzZ2JHSXlJRHdnTVRJNE95QnNZaklyS3lrZ2V3b2dJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdjMUl5TFMwN0NpQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvYzFJeUlEd2dN'
    'Q2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJR2xtSUNoelVESWdQaUF3S1NCN0lITlFNaTB0T3lCelVqSWdQU0J3WVhS'
    'VGRHRnlkRkp2ZDF0elVESmRJQ3NnS0hCaGRGSnZkMDltWm5ObGRGdHpVREly'
    'TVYwZ0xTQndZWFJTYjNkUFptWnpaWFJiYzFBeVhTa2dMU0F4T3lCOUNpQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnWld4elpT'
    'QmljbVZoYXpzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJRTV2'
    'ZEdVZ2NESWdQU0JuWlhST2IzUmxLSE5RTWl3Z2MxSXlMQ0JqYUNrN0NpQWdJ'
    'Q0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvY0RJdWFX'
    'NXpkSEoxYldWdWRDQStJREFwSUhzZ2NtVmhiQzVwYm5OMGNuVnRaVzUwSUQw'
    'Z2NESXVhVzV6ZEhKMWJXVnVkRHNnWW5KbFlXczdJSDBLSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdmUW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJ'
    'Q0FnSUNBZ0lDQndWSEpwWjA1dmRHVWdQU0J5WldGc095QndWSEpwWjFKdmR5'
    'QTlJSE5TT3lCd1ZISnBaMUJoZENBOUlITlFPd29nSUNBZ0lDQWdJQ0FnSUNB'
    'Z0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmljbVZo'
    'YXpzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lDQWdm'
    'UW9nSUNBZ0lDQWdJSDBLQ2lBZ0lDQWdJQ0FnTHk4ZzRwU0E0cFNBSUZKbGJt'
    'UmxjaUIzYVhSb0lIQnlaWFpwYjNWeklIUnlhV2RuWlhJZ1lXNWtJR0pzWlc1'
    'a0lPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tV'
    'Z09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1Vn'
    'QW9nSUNBZ0lDQWdJQzh2SUZSb1pTQndjbVYyYVc5MWN5QjBjbWxuWjJWeUlH'
    'ZGxkSE1nZEc5dVpWTnNhV1JsVkdGeVoyVjBQVEFnS0hkbElHUnZiaWQwSUhS'
    'eVlXTnJJR2wwY3dvZ0lDQWdJQ0FnSUM4dklITnNhV1JsSUdOb1lXbHVJT0tB'
    'bENCbWIzSWdkR2hsSURZMExYTmhiWEJzWlNCamNtOXpjMlpoWkdVZ2QybHVa'
    'RzkzSUhSb1pTQmthV1ptWlhKbGJtTmxDaUFnSUNBZ0lDQWdMeThnYVhNZ2FX'
    'NWhkV1JwWW14bEtTNGdUR2x1WldGeUlHTnliM056Wm1Ga1pUb2dkRDB3SUds'
    'eklHRnNiQzF3Y21WMmFXOTFjeXdnZEQweElHbHpDaUFnSUNBZ0lDQWdMeThn'
    'WVd4c0xXTjFjbkpsYm5RdUlGTnRiMjkwYUhOMFpYQWdkMjkxYkdRZ2QyOXlh'
    'eUJpZFhRZ2JHbHVaV0Z5SUcxaGRHTm9aWE1nVFdsclRXOWtKM01LSUNBZ0lD'
    'QWdJQ0F2THlCREt5c2dUV2w0S2xOMFpYSmxiMDV2WTJ4cFkyc2dkbTlzZFcx'
    'bElISmhiWEJwYm1jZ1pYaGhZM1JzZVM0S0lDQWdJQ0FnSUNCcFppQW9jRlJ5'
    'YVdkUVlYUWdQajBnTUNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0JtYkc5aGRDQnpY'
    'M0J5WlhZZ1BTQmZaMk52UW05a2VTaGphQ3dnY0c5ekxDQjBhVzFsTENCeWIz'
    'ZFVhVzFsTENCd1ZISnBaMUJoZEN3Z2NGUnlhV2RTYjNjc0lIQlVjbWxuVG05'
    'MFpTd2dNQ2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lHWnNiMkYwSUhRZ1BTQmhaMlZU'
    'WVcxd2JHVnpJQzhnTmpRdU1Ec0tJQ0FnSUNBZ0lDQWdJQ0FnY21WMGRYSnVJ'
    'SE5mY0hKbGRpQXFJQ2d4TGpBZ0xTQjBLU0FySUhOZlkzVnljaUFxSUhRN0Np'
    'QWdJQ0FnSUNBZ2ZRb2dJQ0FnZlFvS0lDQWdJSEpsZEhWeWJpQnpYMk4xY25J'
    'N0NuMEsnKS5kZWNvZGUoJ3V0Zi04JykKCiAgICAjIEFzc2VtYmxlCiAgICBy'
    'ZXR1cm4gaGVhZGVyICsgbWV0YSArICIiLmpvaW4oZGF0YV9hcnJheXMpICsg'
    'IlxuIiArIHRhYmxlcyArIGZldGNoZXJzICsgZGVjb2RlcnMgKyBnZXRfY2hh'
    'bm5lbF9vdXRwdXQKCgppZiBfX25hbWVfXyA9PSAnX19tYWluX18nOgogICAg'
    'bW9kX3BhdGggPSBzeXMuYXJndlsxXSBpZiBsZW4oc3lzLmFyZ3YpID4gMSBl'
    'bHNlICcvbW50L3VzZXItZGF0YS91cGxvYWRzLzEyVEguTU9EJwogICAgb3V0'
    'X3BhdGggPSBzeXMuYXJndlsyXSBpZiBsZW4oc3lzLmFyZ3YpID4gMiBlbHNl'
    'ICcvaG9tZS9jbGF1ZGUvbW9kX2NydW5jaC8xMlRIX2NydW5jaF9jb21tb24u'
    'Z2xzbCcKICAgIG1haW4obW9kX3BhdGgsIG91dF9wYXRoKQo='
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
    parser.add_argument('--samples', action='store_true', default=False,
                        help='Extract each sample (instrument) from the module as a separate '
                             'WAV file (named like 1-samplename.wav, 2-anothername.wav). '
                             'Skips GLSL/HTML generation. Saves to current directory unless '
                             '--samples-dir is also given. Useful for diagnosing per-sample '
                             'playback issues (which sample is wrong/buzzy/missing).')
    parser.add_argument('--samples-dir', type=str, default=None,
                        help='Output directory for --samples WAV files (default: current dir).')
    parser.add_argument('--solo', type=int, default=None, metavar='CH',
                        help='Solo a single channel: mute every other channel by clearing '
                             'their cells before encoding. CH is 1-based (so --solo 1 keeps '
                             'channel 1, mutes 2..N). Useful for diagnosing per-channel '
                             'issues (which channel has the wrong sample, missing notes, '
                             'wrong panning, etc.). Pipeline runs normally — sample '
                             'selection, effects, panning, FX all apply — only the soloed '
                             'channel produces output.')

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
                        help='[NO-OP — max-compat is now the DEFAULT in v1.40+ (current: v1.42)] '
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
            # bspline matches openmpt's reference more closely than lanczos3,
            # particularly for samples with in-loop discontinuities (e.g.
            # SATELL.S3M sample 4). Lanczos's sharp kernel preserves and
            # amplifies single-sample spikes as audible per-loop-cycle buzz.
            'resampler':              'lanczos3',
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
        # bspline is now the default. Lanczos-3's 6-tap sinc kernel is sharp
        # enough to expose 1-sample anomalies in tracker samples (e.g. SATELL.S3M
        # sample 4 has a +12288-unit spike at index 693 inside its loop region)
        # as audible buzz on every loop wrap. B-spline's 4-tap cubic kernel
        # smooths these gracefully and produces output that closely tracks
        # openmpt's reference (HF spectrum within 0.5% of reference vs ~3% for
        # lanczos3, peak amplitude within 10% of reference). Pass --resampler
        # lanczos3 explicitly to opt back in.
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
        print("   ⚠️  S3M support is partial: file loads but effect commands are not yet")
        print("      remapped from S3M (A,B,C,…) to MOD (1,2,3,…) numbering, c2spd is not")
        print("      applied, and the volume column is dropped. Many S3Ms will play with")
        print("      wrong effects/pitches until that work lands.")
        
        # Convert S3M to MOD-compatible structure for now (effect-letter and
        # c2spd remapping is still TODO — see warning above).
        # IMPORTANT: keep `patterns` indexable by the ORIGINAL pattern number,
        # NOT reordered into song-play order. `song_positions` carries the
        # original pattern indices (0..num_patterns-1), and `mod.patterns[i]`
        # must give the i-th pattern, exactly like MODFile does it. The old
        # shim built a play-order-reordered list which then crashed when
        # downstream code did `mod.patterns[mod.song_positions[k]]`.
        mod = type('obj', (object,), {
            'title': s3m.title,
            'samples': s3m.instruments,
            'num_patterns': s3m.num_patterns,
            'song_length': len(s3m.orders),
            'song_positions': s3m.orders,
            'orders': s3m.orders,
            'patterns': s3m.patterns,                  # full pattern table (no reorder)
            'num_channels': s3m.num_channels,
            'initial_speed': s3m.initial_speed,
            'initial_tempo': s3m.initial_tempo,
            # Forward S3M's per-channel pan settings so downstream code
            # (notably the VQ encoder's chan_pan emission) can produce
            # the file-specified pan layout. Without this the encoder
            # falls back to LRRL — wrong for SATELL.S3M (LRLR) and other
            # files that don't follow the Amiga convention. Result:
            # both lead voices would land on the same speaker, audible
            # as harshness/clipping on one side and hollowness on the
            # other.
            'channel_settings': list(getattr(s3m, 'channel_settings', []) or []),
            'is_s3m': True
        })()
    elif fmt == 'MOD':
        mod = MODFile(args.modfile)
        mod.is_s3m = False
        mod.num_channels = 4
    elif fmt in ('XM', 'IT', 'STM', 'MTM'):
        raise ValueError(
            f"{fmt} format is not yet implemented in this player. "
            f"Currently supported: MOD (full), S3M (partial — effects/c2spd not remapped). "
            f"Patches welcome."
        )
    else:
        raise ValueError(
            f"Unknown module format for {args.modfile!r}: signature check failed and "
            f"file extension is not in the recognized set "
            f"(.mod / .s3m / .xm / .it / .stm / .mtm / .m15 / .nst / .wow). "
            f"If this really is a tracker module, rename it with the correct extension "
            f"or report the file so we can add a signature for it."
        )

    # ── --samples: extract every instrument as a separate WAV and exit. ──
    # Useful for diagnosing per-sample issues (which sample sounds wrong, has
    # noise, plays at wrong pitch, etc.) without going through the full GLSL
    # pipeline. Naming: <idx>-<sanitized_name>.wav, 1-based, with name taken
    # from the sample's own header. Empty/AdLib/zero-length slots are skipped
    # but reported.
    if getattr(args, 'samples', False):
        import wave as _wave_, re as _re_, numpy as _np_
        out_dir = args.samples_dir or '.'
        os.makedirs(out_dir, exist_ok=True)

        def _sanitize_name(s):
            # Strip nulls, control chars, replace anything not [A-Za-z0-9_-]
            # with underscore. Collapse runs and trim. Empty → 'unnamed'.
            if isinstance(s, bytes):
                s = s.decode('latin-1', errors='replace')
            s = (s or '').strip().rstrip('\x00').strip()
            s = _re_.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_')
            return s or 'unnamed'

        # Sample rate to write the WAVs at: use each sample's c2spd if present
        # (S3M), else the standard Amiga base rate 8363 Hz (MOD). This way
        # the WAV plays back at the sample's "natural" pitch.
        wrote = 0; skipped = 0
        for idx, smp in enumerate(mod.samples):
            n = idx + 1  # 1-based
            if not isinstance(smp, dict):
                print(f"   [{n:2d}] (empty slot — skipped)")
                skipped += 1
                continue
            data = smp.get('data', None)
            length = smp.get('length', 0) if smp.get('length', 0) else (
                len(data) if data is not None else 0)
            name = smp.get('name', '') or smp.get('title', '') or ''
            if data is None or length == 0:
                print(f"   [{n:2d}] '{_sanitize_name(name)[:30]}' — empty (length=0), skipped")
                skipped += 1
                continue

            # Sample rate: c2spd for S3M, 8363 for MOD (Amiga base).
            sr = smp.get('c2spd', 0) or 8363
            sr = int(sr) if sr > 0 else 8363

            # Convert sample.data (np.int8 array, signed -128..127) to int16
            # PCM. MOD/S3M loaded data is already signed int8.
            arr = _np_.asarray(data, dtype=_np_.int8)
            # Trim to declared length in case array is over-allocated
            if len(arr) > length:
                arr = arr[:length]
            pcm16 = (arr.astype(_np_.int16) * 256)  # int8 → int16 (sign-extended)

            wav_name = f"{n}-{_sanitize_name(name)}.wav"
            wav_path = os.path.join(out_dir, wav_name)
            with _wave_.open(wav_path, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm16.tobytes())

            # Loop info / extra context
            ls = smp.get('loop_start', smp.get('repeat_point', 0)) or 0
            ll = smp.get('loop_len',   smp.get('repeat_length', 0)) or 0
            vol = smp.get('volume', 64)
            loop_str = f" loop {ls}..{ls+ll}" if ll > 2 else " (no loop)"
            print(f"   [{n:2d}] {wav_name}  ({length} bytes @ {sr} Hz, vol {vol}{loop_str})")
            wrote += 1

        print(f"\n✅ Wrote {wrote} sample WAV(s) to {out_dir!r}; skipped {skipped} empty/invalid slot(s).")
        return

    # ── --solo: mute every channel except the chosen one. ──────────────────
    # Implementation: walk every cell of every pattern, and for any channel
    # other than the soloed one, replace its cell with an empty cell. The
    # whole encoding pipeline downstream (effect remap, sample selection,
    # panning, FX, GLSL emission) runs unchanged — but only the soloed
    # channel produces sound. Lets you pinpoint which channel has the
    # broken sample / wrong note / wrong pan / etc.
    #
    # CH is 1-based per the user-facing convention (channel 1 is index 0).
    if getattr(args, 'solo', None) is not None:
        solo_ch_1based = int(args.solo)
        solo_ch = solo_ch_1based - 1   # 0-based internally
        if solo_ch < 0 or solo_ch >= mod.num_channels:
            raise ValueError(
                f"--solo {solo_ch_1based}: out of range for this module "
                f"(has {mod.num_channels} channels — pick 1..{mod.num_channels})."
            )
        is_s3m = bool(getattr(mod, 'is_s3m', False))
        if is_s3m:
            # S3M cell format: dict with note/instrument/volume/command/info.
            # note=255 means "no note", instrument=0 means "no instrument
            # change", volume=255 means "no volume change". This is a true
            # no-op cell that won't trigger or affect the channel state.
            empty = {'note': 255, 'instrument': 0, 'volume': 255,
                     'command': 0, 'info': 0}
        else:
            # MOD cell format: dict with sample/period/effect/param.
            empty = {'sample': 0, 'period': 0, 'effect': 0, 'param': 0}

        # Mute only the first num_channels columns. S3M rows can have 32 slots
        # in memory (one per possible channel hardware) but only the first
        # num_channels carry song data; the rest are already empty/disabled.
        # We don't touch those — saves work and avoids any chance of confusing
        # the encoder if it ever cares about the extra columns.
        muted_cells = 0
        for pat_idx, pat in enumerate(mod.patterns):
            if pat is None:
                continue
            for row in pat:
                upper = min(len(row), mod.num_channels)
                for ch in range(upper):
                    if ch != solo_ch:
                        # Replace in-place. Use copies of `empty` so all rows
                        # don't alias the same dict (safer if anything later
                        # mutates a cell).
                        row[ch] = dict(empty)
                        muted_cells += 1

        print(f"🎚️  --solo {solo_ch_1based}: kept channel {solo_ch_1based}, "
              f"muted {mod.num_channels - 1} other channel(s) "
              f"({muted_cells} cells cleared across {len(mod.patterns)} pattern(s)).")

    # Note: sample downsampling is now handled inside vq_encoder_v2 (anti-aliased RVQ).
    # The --downsample flag controls the RVQ downsampling factor — no manual decimation needed.
    
    base_name = os.path.splitext(os.path.basename(args.modfile))[0]
    
    # Collect all samples with bandwidth-adaptive compression + zero-padding
    all_samples = []
    png_sample_bw = {}  # track bw_factor per sample index for sample_map
    for idx, smp in enumerate(mod.samples):
        # S3M parser used to leave None for empty/AdLib slots — defensively
        # treat any non-dict entry (or one missing 'data') as an empty sample.
        if not isinstance(smp, dict):
            png_sample_bw[idx] = 1
            continue
        smp_data = smp.get('data', None)
        if smp_data is not None and len(smp_data) > 0:
            bf, compressed = bw_compress_sample(smp_data)
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
    
    # S3M note to MOD period conversion table.
    # Indexed by FLAT semitone (octave*12 + semitone), 0..95 covering 8 octaves.
    # ANCHORED on user-validated playback: S3M C-5 (note byte 0x50, idx 60) maps
    # to period 214. Pitch was validated by ear against libopenmpt's reference
    # rendering of the same S3M source.
    #
    # History: prior anchor was idx 60 → 428, which played 1 octave too low by
    # ear comparison. Empirically halving the periods (shifting up 1 octave) by
    # the user's ear test put pitch in the right register. Note that this means
    # the convention used here for S3M is essentially "C-5 = ProTracker C-4"
    # (period 214), one octave higher than the documented "C-5 plays at c2spd"
    # interpretation — likely an artifact of how libopenmpt's S3M C-5 maps to a
    # different reference frequency than naive ST3 spec reading would suggest.
    s3m_note_to_period = [
        # octave 0
        6848, 6464, 6101, 5758, 5435, 5130, 4842, 4570, 4314, 4072, 3843, 3628,
        # octave 1
        3424, 3232, 3050, 2879, 2718, 2565, 2421, 2285, 2157, 2036, 1922, 1814,
        # octave 2
        1712, 1616, 1525, 1440, 1359, 1283, 1211, 1143, 1078, 1018,  961,  907,
        # octave 3
         856,  808,  763,  720,  679,  641,  605,  571,  539,  509,  480,  453,
        # octave 4
         428,  404,  381,  360,  340,  321,  303,  286,  270,  254,  240,  227,
        # octave 5  (S3M "middle" — anchor)
         214,  202,  191,  180,  170,  160,  151,  143,  135,  127,  120,  113,
        # octave 6
         107,  101,   95,   90,   85,   80,   76,   71,   67,   64,   60,   57,
        # octave 7
          54,   50,   48,   45,   42,   40,   38,   36,   34,   32,   30,   28,
    ]

    def s3m_note_to_mod_period(note):
        """Convert S3M note byte to MOD period.
        S3M encodes a note as (octave<<4)|semitone with semitone in 0..11.
        Special values: 0xFF = no note, 0xFE = note-off."""
        if note == 255 or note == 254:
            return 0
        octave   = (note >> 4) & 0x0F
        semitone = note & 0x0F
        if semitone >= 12:                 # malformed cell
            return 0
        idx = octave * 12 + semitone
        if idx >= len(s3m_note_to_period):
            return 0                       # beyond table — caller treats as no-note
        return s3m_note_to_period[idx]
    
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
    # CRITICAL: also reorder mod.patterns so dense-index lookups land on the
    # correct content. Without this, the VQ encoder reads mod.patterns[N]
    # expecting "the Nth pattern in the dense sequence" but gets the original
    # pattern index N — for any song that doesn't use patterns 0..N-1 contiguously
    # (introspect.s3m skips 15, 27..29 and reorders 8,16-20), this loads the
    # wrong notes, wrong effects, and wrong rows from order 8 onward. Visible
    # symptoms: tempo changes, pitch jumps, missed pattern-break commands.
    mod.patterns            = [mod.patterns[i] for i in used_pat_indices]

    pattern_bytes = []
    # mod.patterns is now in dense order (matches used_pat_indices) — iterate
    # by dense index, not original index.
    for dense_idx in range(len(used_pat_indices)):
        pattern = mod.patterns[dense_idx]
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

            # ── Patch: BPM-aware row tick scaling ─────────────────────────────
            # The encoder's compute_row_speed_table tracked BPM but never
            # used it — rowStartTick was built from physical speeds only,
            # then GLSL hard-coded TICKS_PER_SEC=50.0 (BPM=125 equivalent).
            # Songs with non-default BPM (e.g. hippy.mod uses FA7 → BPM=167)
            # ran at the wrong rate: rows arrived ~25% late, leaving audible
            # silence between sample triggers when other players had already
            # moved on. Fix: scale each row's tick contribution by 125/bpm[r]
            # so the GLSL formula `time = rowStartTick / 50.0` stays valid
            # under variable BPM.
            def _bpm_aware_compute_row_speed_table(_mod):
                speed = 6
                bpm   = 125
                rowSpeed = []
                cumF = 0.0
                rowStartTickF = [0.0]
                bpm_changes = False
                for _pos in range(_mod.song_length):
                    pat_idx = _mod.pattern_order[_pos]
                    pdata = _mod.patterns[pat_idx]
                    broke = False
                    for row in range(64):
                        for ch in range(_mod.num_channels):
                            base = row * _mod.num_channels * 4 + ch * 4
                            b0, b1, b2, b3 = pdata[base:base+4]
                            effect = b2 & 0x0F
                            param  = b3
                            if effect == 0xF and param > 0:
                                if param < 0x20:
                                    speed = param
                                else:
                                    if bpm != param: bpm_changes = True
                                    bpm = param
                            elif effect == 0xD or effect == 0xB:
                                broke = True
                        rowSpeed.append(speed)
                        # Scale: ticks-equivalent at BPM=125 so time = ticks/50
                        cumF += speed * 125.0 / max(bpm, 1)
                        rowStartTickF.append(cumF)
                        if broke:
                            break
                # Round cumulative (not per-row) → max drift 0.5 tick = 10ms
                rowStartTick = [int(round(t)) for t in rowStartTickF]
                return rowSpeed, rowStartTick, bpm_changes
            _vqmod.compute_row_speed_table = _bpm_aware_compute_row_speed_table

            # ── Patch: xxCN signature variant (Schism S3M→MOD export) ────────
            # Schism Tracker writes '10CN'/'12CN' etc. for extended-channel
            # MODs instead of the more common '10CH' form. The encoder's
            # internal MODFile.parse() only checks for xxCH and silently
            # falls back to 4 channels on xxCN — every pattern row then gets
            # sliced at the wrong stride, producing garbled playback.
            # Fix at the lowest level: rewrite the two signature bytes in
            # the in-memory file buffer before the original parse() runs,
            # so all downstream code (channel detect, pattern slicing,
            # sample offsets) sees the canonical xxCH form.
            _orig_parse = _vqmod.MODFile.parse
            def _parse_with_xxCN(self):
                if (len(self.data) >= 1084
                        and self.data[1082:1084] == b'CN'
                        and self.data[1080:1081].isdigit()
                        and self.data[1081:1082].isdigit()):
                    self.data = self.data[:1082] + b'CH' + self.data[1084:]
                _orig_parse(self)
            _vqmod.MODFile.parse = _parse_with_xxCN

            print(f"\n\U0001f3b5 Generating VQ-encoded Common tab...")

            # ── S3M → VQ encoder bridge ───────────────────────────────────
            # The encoder's internal MODFile reparses the file from disk as
            # MOD-format bytes — that crashes on .s3m, falling back to the
            # legacy create_shadertoy_glsl path which decides USE_EMBEDDED_DATA
            # by file size and often emits =0 (PNG mode) for big S3Ms. To
            # keep S3Ms on the VQ success path (always USE_EMBEDDED_DATA=1
            # plus the proper 4-tab structure), monkey-patch _vqmod.MODFile
            # for the duration of this call so it returns an adapter built
            # from the S3M data we already parsed in this main().
            _saved_modfile = _vqmod.MODFile
            if getattr(mod, 'is_s3m', False):
                _outer_mod = mod
                _outer_s3m_to_period = s3m_note_to_mod_period

                # ── S3M → MOD effect remap ──────────────────────────────────
                # The naive `cmd & 0x0F` mask we used before silently turned
                # S3M command letters into completely different MOD effects:
                # most disastrously, S3M Oxx (sample offset, cmd=15) masked to
                # MOD Fxx (set speed/tempo), so every Oxx in the song crashed
                # the playhead into 1-tick-per-row mode (≈6× faster). 279
                # instances in introspect.s3m alone — the first one fired at
                # order 1 row 48 and made everything past pattern 0 sound 10×
                # too fast.
                #
                # Proper remap below. Returns (mod_cmd_nibble, mod_param_byte).
                # Returns (0, 0) for S3M-only commands with no MOD equivalent
                # (I tremor, M channel-vol, N channel-volslide, P pan-slide,
                # V global-vol) — better to drop these silently than fire the
                # wrong MOD effect.
                #
                # NOTE: This is best-effort. S3M Axx and Txx both map to MOD
                # Fxx but the param-range split is different (S3M has separate
                # commands; MOD picks based on param value). MOD Fxx with
                # param < 0x20 = speed, ≥ 0x20 = BPM. We pass S3M params
                # through unchanged — works for canonical usage but will
                # mis-fire if a song uses A with param ≥ 0x20 (rare).
                def _s3m_cmd_to_mod(cmd, param):
                    """Map an S3M (command, param) to (mod_effect_nibble, param)."""
                    if cmd == 0:
                        return (0, param)
                    table = {
                        1:  0xF,  # A → Fxx (set speed)
                        2:  0xB,  # B → Bxx (jump to order)
                        3:  0xD,  # C → Dxx (pattern break)
                        4:  0xA,  # D → Axx (volume slide)
                        5:  0x2,  # E → 2xx (porta down)
                        6:  0x1,  # F → 1xx (porta up)
                        7:  0x3,  # G → 3xx (tone porta)
                        8:  0x4,  # H → 4xx (vibrato)
                        # 9 (I tremor) → no MOD equivalent
                        10: 0x0,  # J → 0xx (arpeggio)
                        11: 0x6,  # K → 6xx (vibrato + volslide)
                        12: 0x5,  # L → 5xx (tone porta + volslide)
                        # 13 (M channel vol) → no MOD equivalent
                        # 14 (N channel volslide) → no MOD equivalent
                        15: 0x9,  # O → 9xx (sample offset)
                        # 16 (P pan slide) → no MOD equivalent
                        17: 0xE,  # Q → E9x (retrigger) — needs param remap
                        18: 0x7,  # R → 7xx (tremolo)
                        19: 0xE,  # S → Exx (extended)  passthrough param hi nibble
                        20: 0xF,  # T → Fxx (set tempo if ≥0x20)
                        21: 0x4,  # U → 4xx (fine vibrato; approximated as vibrato)
                        # 22 (V global vol) → no MOD equivalent
                    }
                    if cmd not in table:
                        return (0, 0)
                    eff = table[cmd]
                    # Param adjustments for special cases:
                    if cmd == 17:  # Q retrig — MOD E9x: nibble 9 + retrig speed
                        return (eff, 0x90 | (param & 0x0F))
                    return (eff, param)

                class _S3MtoVQAdapter:
                    """Duck-typed mod object that exposes the attribute set
                    the VQ encoder reads (title, samples_info, sample_bytes,
                    song_length, pattern_order, num_channels, num_patterns,
                    patterns) — built from already-parsed S3M data instead
                    of re-reading raw MOD-format bytes from disk."""
                    def __init__(self, _path_unused):
                        m = _outer_mod
                        self.title = m.title
                        self.song_length  = len(m.song_positions)
                        # MOD format stores pattern_order as a fixed 128-byte
                        # table; pad with zeros to match.
                        po = list(m.song_positions[:128])
                        while len(po) < 128:
                            po.append(0)
                        self.pattern_order = po
                        self.num_channels  = m.num_channels
                        self.num_patterns  = m.num_patterns
                        # S3M stores per-channel pan in channel_settings
                        # (32 bytes from header offset 0x40). Forward this
                        # so the VQ encoder can emit accurate channelPan
                        # values rather than defaulting to LRRL — a wrong
                        # default for files like SATELL.S3M which use LRLR
                        # (alternating). Without this, both lead voices
                        # land on the same speaker → distortion / clipping
                        # on the loud side, hollow on the quiet side.
                        self.channel_settings = list(getattr(m, 'channel_settings', []) or [])
                        self.magic         = b'M.K.'   # placeholder — encoder
                                                       # only uses it for ch
                                                       # count, which we set
                                                       # explicitly above.
                        # samples_info: encoder expects 31 entries with keys
                        # name, length, finetune, volume, loop_start, loop_len.
                        # S3M instruments use 'repeat_point'/'repeat_length' so
                        # remap field names. Pad with empty entries up to 31.
                        self.samples_info = []
                        self.sample_bytes = []
                        for i in range(31):
                            si = m.samples[i] if (i < len(m.samples) and isinstance(m.samples[i], dict)) else None
                            if si is not None:
                                arr = si.get('data', None)
                                raw = (arr.tobytes() if (arr is not None and getattr(arr, 'size', 0) > 0) else b'')
                                self.samples_info.append(dict(
                                    name       = si.get('name', ''),
                                    length     = si.get('length', 0),
                                    finetune   = si.get('finetune', 0),
                                    volume     = si.get('volume', 0),
                                    loop_start = si.get('repeat_point', 0),
                                    loop_len   = si.get('repeat_length', 0),
                                ))
                                self.sample_bytes.append(raw)
                            else:
                                self.samples_info.append(dict(
                                    name='', length=0, finetune=0, volume=0,
                                    loop_start=0, loop_len=0))
                                self.sample_bytes.append(b'')
                        # Patterns: encoder consumes packed MOD bytes
                        # (64 rows × num_channels × 4 bytes per cell). Convert
                        # each S3M cell using the same period table the rest
                        # of main() uses.
                        self.patterns = []
                        for pat_idx in range(self.num_patterns):
                            pat = m.patterns[pat_idx] if pat_idx < len(m.patterns) else None
                            buf = bytearray(64 * self.num_channels * 4)
                            if pat is not None:
                                for row_i in range(min(64, len(pat))):
                                    row = pat[row_i]
                                    for ch_i in range(self.num_channels):
                                        cell = row[ch_i] if ch_i < len(row) else None
                                        if cell is None:
                                            continue
                                        inst   = cell.get('instrument', 0)
                                        note   = cell.get('note', 255)
                                        cmd    = cell.get('command', 0)
                                        inf    = cell.get('info', 0)
                                        period = _outer_s3m_to_period(note)
                                        # c2spd compensation: S3M instruments
                                        # carry a per-sample c2spd (samples/sec
                                        # at C-5). Standard reference is 8363
                                        # Hz. Samples with non-standard c2spd
                                        # need their playback rate scaled by
                                        # c2spd/8363 — equivalently, the period
                                        # we encode must be divided by that
                                        # factor so the GLSL's
                                        # `freq = 7093789.2 / (period * 2)`
                                        # produces the right rate.
                                        # For introspect.s3m this fixes the
                                        # BChord1-4 (c2spd=16726, 2× standard,
                                        # so periods halved → 1 octave up) and
                                        # LeadGuit (c2spd=13140) which were
                                        # otherwise playing nearly 1 octave low
                                        # vs the libopenmpt reference.
                                        if period > 0 and 1 <= inst <= len(_outer_mod.samples):
                                            _si = _outer_mod.samples[inst-1]
                                            if isinstance(_si, dict):
                                                _c2 = _si.get('c2spd', 8363) or 8363
                                                if _c2 != 8363:
                                                    period = max(1, int(round(period * 8363.0 / _c2)))
                                                    if period > 4095:
                                                        period = 4095   # 12-bit field cap
                                        # Remap S3M command letter to MOD effect
                                        # nibble (see _s3m_cmd_to_mod for the
                                        # rationale and table).
                                        mod_eff, mod_param = _s3m_cmd_to_mod(cmd, inf)
                                        # Volume column: S3M cells carry an
                                        # optional per-cell volume (0..64) in
                                        # cell['volume']. Value 255 (or out of
                                        # range) means "no override". We
                                        # propagate it to MOD as effect Cxx
                                        # (set volume), but ONLY when the cell
                                        # doesn't already have an effect — MOD
                                        # has only one effect slot per cell.
                                        # Without this, samples like BChord
                                        # whose first trigger has cell vol=0
                                        # play at full instrument volume
                                        # (= much louder than libopenmpt).
                                        # ~2500 cells in introspect have both
                                        # effect AND volume; for those the
                                        # volume column is dropped (the effect
                                        # wins, since it's likely a slide).
                                        _cv = cell.get('volume', 255)
                                        if mod_eff == 0 and 0 <= _cv <= 64:
                                            mod_eff   = 0xC
                                            mod_param = _cv
                                        # Special case: when cell has BOTH a
                                        # volume slide (MOD Axx, from S3M D)
                                        # AND a volume column value, the song
                                        # is using the volume column to
                                        # express the per-row volume
                                        # explicitly while D drives within-row
                                        # behavior. We can't encode both, so
                                        # prefer the volume column — that
                                        # carries the actual dynamics curve.
                                        # Without this, every row of a vol-
                                        # slide passage in S3M (like SATELL's
                                        # pat 1 rows 44-63 ramping from 10 to
                                        # 64) gets the slide effect with NO
                                        # volume reset, so the volume sticks
                                        # at whatever the prior tick state
                                        # was — typically 0 after the first
                                        # downward slide tick.
                                        elif mod_eff == 0xA and 0 <= _cv <= 64:
                                            mod_eff   = 0xC
                                            mod_param = _cv
                                        # Volume column override: S3M cells
                                        # carry an optional per-cell volume
                                        # (0..64). 255 = "no volume column"
                                        # (use sample header default). When a
                                        # cell has an explicit volume but no
                                        # other effect, encode it as MOD `Cxx`
                                        # (set channel volume). This fixes
                                        # BChord at pat 1 row 0 in introspect:
                                        # the cell has vol=0 (silent trigger)
                                        # which libopenmpt honors but our
                                        # earlier code ignored, so BChord
                                        # blasted in at full sample-default
                                        # volume.
                                        # Caveat: cells with BOTH a volume
                                        # column AND another effect lose the
                                        # volume column (effect wins) since
                                        # the 4-byte cell layout has no room
                                        # for both. ~50% of introspect's
                                        # interesting cells fall in this
                                        # category — a future refactor could
                                        # add a separate per-cell volume table
                                        # alongside the pattern data.
                                        cell_vol = cell.get('volume', 255)
                                        if mod_eff == 0 and cell_vol != 255 and cell_vol <= 64:
                                            mod_eff = 0xC          # MOD Cxx = set channel volume
                                            mod_param = cell_vol
                                        o = (row_i * self.num_channels + ch_i) * 4
                                        buf[o]   = (inst & 0xF0) | ((period >> 8) & 0x0F)
                                        buf[o+1] = period & 0xFF
                                        buf[o+2] = ((inst & 0x0F) << 4) | (mod_eff & 0x0F)
                                        buf[o+3] = mod_param & 0xFF
                            self.patterns.append(bytes(buf))

                        # Encoder reads `mod.data[1084 + pat*pat_size + row*NC*4 + ch*4 : +4]`
                        # in actual_pattern_rows() to detect Dxx/Bxx pattern-break/jump
                        # effects. Synthesize a MOD-format blob: 1084-byte header (zeros)
                        # + concatenated pattern bytes so that offset math resolves to
                        # the same cell bytes we already wrote into self.patterns.
                        self.data = b'\x00' * 1084 + b''.join(self.patterns)

                _vqmod.MODFile = _S3MtoVQAdapter

            try:
                _vqmod.main(args.modfile, glsl_common_file, K=256, weighted=True, downsample=args.downsample, bitrate=args.bitrate, vec_dim=args.vec_dim, resampler=args.resampler, no_rvq2=args.no_rvq2)
            finally:
                _vqmod.MODFile = _saved_modfile
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
                    "GLSL (The Last) MOD Player v1.42 (c) 2026 Orblivius",
                    "GLSL (The Last) MOD Player v1.42 (c) 2026 Orblivius", 1)
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
                        "#define SONG_DURATION_S  (float(TOTAL_TICKS) / TICKS_PER_SEC)\n"
                        "#define INTRO_SILENCE_S  1.5\n"
                        "#define AUDIO_BUFFER_S   180.0\n"
                    )
                    _tsr_idx = _ct.find('#define TOTAL_SONG_ROWS')
                    if _tsr_idx >= 0:
                        _eol = _ct.find('\n', _tsr_idx)
                        if _eol >= 0:
                            _ct = _ct[:_eol+1] + _timing_block + _ct[_eol+1:]
                    else:
                        _ct = _timing_block + _ct
                # Fix misleading "BPM=125 constant for 12TH.MOD" comment that
                # was hardcoded in the encoder's GLSL emitter — TICKS_PER_SEC
                # is now BPM-aware via the row-tick scaling patch above.
                _ct = _ct.replace(
                    "#define TICKS_PER_SEC     50.0   // BPM=125 constant for 12TH.MOD",
                    "#define TICKS_PER_SEC     50.0   // rowStartTick is BPM-scaled (see compute_row_speed_table patch)"
                )
                # Replace the encoder's hardcoded `#define BPM 125.0` /
                # `#define SPEED 6.0` with the song's actual initial values.
                # This isn't used for SOUND timing (that goes through the
                # BPM-aware rowStartTick table), but it IS used by the
                # visualizer for tempo-synced animations (iBeatSharp,
                # _v6_BPM_SCALE for the dancer's clock multipliers, etc.).
                # Without this, viz 6's dancer at BPM=167 still moved at
                # BPM=125 rate — visibly out of sync with the song.
                _bpm_actual   = float(getattr(mod, 'initial_tempo', 125))
                _speed_actual = float(getattr(mod, 'initial_speed', 6))
                _ct = _ct.replace(
                    "#define BPM               125.0",
                    f"#define BPM               {_bpm_actual}",
                    1
                )
                _ct = _ct.replace(
                    "#define SPEED             6.0",
                    f"#define SPEED             {_speed_actual}",
                    1
                )
                # Override FX flags hardcoded by the VQ encoder.
                # The encoder bakes `enableFAT = true` and `enable3D = true`
                # into its emitted Common file regardless of CLI flags. The
                # FX it gates (PhatBass at 1.5× depth, Only3D widening) push
                # the mix 1.5–3× louder than openmpt's reference rendering on
                # bass-heavy passages — audible as harshness/clipping on lead
                # voices and bass. Honor --no-fat4x / --no-surround properly
                # by rewriting these constants here.
                _no_fat = bool(getattr(args, '_compat_no_fat', False))
                _no_surround = bool(getattr(args, '_compat_no_surround', False))
                if _no_fat:
                    _ct = _ct.replace(
                        'const bool  enableFAT     = true;',
                        'const bool  enableFAT     = false;', 1
                    )
                if _no_surround:
                    _ct = _ct.replace(
                        'const bool  enable3D      = true;',
                        'const bool  enable3D      = false;', 1
                    )
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
