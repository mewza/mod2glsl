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
  --png remains as the alternative data-storage scheme (samples in
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

# ── Shadertoy plays the Sound shader from a single pre-rendered buffer of
#    this length; audio past it never plays. Pattern/sample data for the
#    unplayed tail is dead weight in the embedded GLSL, so we trim the song
#    order to this cap (keeps the build small enough to stay embedded).
SHADERTOY_AUDIO_CAP_SEC = 180.0

# NOTE: this only affects the legacy create_shadertoy_glsl / HTML
# bw_compress_sample path.  The VQ-encoded ShaderToy build decimates inside
# the base64 vq_encoder_v2 (fed `--downsample`), NOT here, so toggling this
# does nothing for the ShaderToy tabs.  Exact loop bounds for the VQ build
# (the inst5/25/29 periodic-wow fix) come from `--preserve <inst,...>` which
# stores those samples raw/un-decimated.  Left False (no-op) by default.
LOOP_NO_DOWNSAMPLE = False

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
                'c2spd':       sample.get('c2spd', 8363),  # S3M per-sample tuning (JS periodToFreq scales by c2spd/8363)
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
    
    console.log('Decompressed bytes:', decompressed.length, 'expected:', modData.numPatterns * 64 * _NCH * 5);

    // Reconstruct pattern objects
    const patterns = [];
    let offset = 0;
    const totalNotes = modData.numPatterns * 64 * _NCH;
    
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
    # Period clamp bounds — UNIVERSAL max range across MOD/S3M/XM/IT (user-confirmed).
    # Each format's widest Amiga-style note span (8 octaves):
    #   MOD ProTracker  : ~113..856 standard, ~107..907 with finetune
    #   S3M             : 28..6848  (oct7 B .. oct0 C, the s3m_note_to_period table)
    #   XM (Amiga mode) : ~28..6848 ; XM linear period domain tops out ~7680
    #   IT              : 8 octaves, same ~28..6848 envelope
    # Union + finetune/vibrato/portamento headroom → [13, 7680]. Applied to ALL
    # formats: real MOD notes live in [113,856] so this only relaxes extreme
    # porta/vibrato edges (never alters normal MOD playback), while S3M/XM/IT high
    # leads (period<113) and deep bass (period>856) finally play at correct pitch.
    # Floor 13 keeps period safely >0 (no divide-by-~0 in periodToFreq).
    _pmin, _pmax = 13, 7680
    _ch_panels = '\n'.join(
        f'  <div class="ch-panel ch{i}" id="chPanel{i}">\n'
        f'    <div class="ch-header">Track #{i+1}'
        f'<span class="ch-sm"><span class="smbtn smS" id="chS{i}" title="Solo this track">S</span>'
        f'<span class="smbtn smM" id="chM{i}" title="Mute this track">M</span></span></div>\n'
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
#infoGrid{{width:100%;display:grid;grid-template-columns:repeat(5,1fr);border-bottom:1px solid var(--border)}}
.info-cell{{padding:10px 16px;border-right:1px solid var(--border);background:var(--bg1)}}
.info-cell:last-child{{border-right:none}}
.info-label{{font-size:10px;color:var(--dim);letter-spacing:2px;text-transform:uppercase;margin-bottom:4px}}
.info-value{{font-size:22px;color:var(--accent2);letter-spacing:1px;font-weight:bold}}
.info-value .sub{{color:var(--dim);font-size:13px;font-weight:normal}}
#channels{{width:100%;display:grid;grid-template-columns:repeat({num_channels},1fr);gap:1px;background:var(--border);border-bottom:1px solid var(--border)}}
.ch-panel{{background:var(--bg1);padding:10px 14px}}
.ch-header{{font-size:10px;letter-spacing:2px;color:var(--dim);margin-bottom:6px;text-transform:uppercase}}
.ch-sm{{float:right;letter-spacing:0}}
.smbtn{{display:inline-block;width:15px;height:15px;line-height:15px;text-align:center;font-size:9px;font-weight:bold;border-radius:3px;cursor:pointer;margin-left:3px;background:var(--bg2);color:var(--dim);user-select:none}}
.smbtn:hover{{filter:brightness(1.3)}}
.smbtn.on-s{{background:#2e9e6b;color:#fff}}
.smbtn.on-m{{background:#b04545;color:#fff}}
.ch-panel.muted{{opacity:0.4}}
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
  <div class="info-cell"><div class="info-label">Tracks</div>
    <div class="info-value" id="tracksInfo">{mod.num_channels}</div></div>
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
    numChannels: {num_channels},  // FIX: JS pattern decode/getNote MUST use real channel count (was hardcoded 4 → 8-ch S3M scrambled)
    initialBPM: {mod.initial_tempo},
    initialSpeed: {mod.initial_speed},
    bassSamples: {json.dumps(_html_bass_idx)},
    {data_fields},
    samples: {chunk_concat},
    downsample: {downsample}
}};

// Global channel/track count for the loaded mod — single source of truth.
// Pattern decode, getNote indexing, and the mix loop all read this.
// (Was hardcoded 4 in two places → 8-channel S3M played scrambled cells.)
const _NCH = modData.numChannels;

// ── Per-voice mute/solo state ──────────────────────────────────────────────
// channelMuted[ch] is the FINAL gate read by the mixer (see generateSamples).
// It's derived from the user's solo toggles + explicit mutes: if ANY track is
// soloed, every non-soloed track is muted; otherwise only explicitly-muted
// tracks are silenced. Muting gates OUTPUT only — effect/pitch state still
// advances so a track resumes correctly when unmuted.
const channelMuted    = new Array(_NCH).fill(false);
const channelSoloed   = new Array(_NCH).fill(false);
const channelExplicitMuted = new Array(_NCH).fill(false);
function _recomputeMutes() {{
    const anySolo = channelSoloed.some(v => v);
    for (let i = 0; i < _NCH; i++) {{
        channelMuted[i] = anySolo ? !channelSoloed[i] : channelExplicitMuted[i];
        const p = document.getElementById('chPanel' + i);
        if (p) p.classList.toggle('muted', channelMuted[i]);
    }}
}}

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
        this.numChannels = _NCH;  // single source of truth (global, = mod's track count)
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
    periodToFreq(period, finetune, c2spd) {{
        if (period === 0) return 0;
        const c4speeds = [8363,8413,8463,8529,8581,8651,8723,8757,
                          7895,7941,7985,8046,8107,8169,8232,8280];
        const c4 = c4speeds[(finetune || 0) & 0xF];
        // Scale by c2spd/8363 to match the GLSL Sound tab. S3M samples carry
        // per-sample tuning in c2spd (finetune=0); without this factor every
        // sample with c2spd != 8363 (e.g. inst-25 = 14650) plays at the wrong
        // pitch → "right samples, wrong notes". MOD/XM have c2spd=8363 (no-op).
        const tune = (c2spd || 8363) / 8363.0;
        return (c4 * 428 * tune) / period;  // 428 = period for middle C at standard tuning
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
        const idx = (pattern * 64 * _NCH) + (row * _NCH) + channel;
        return modData.patterns[idx] || {{ sample: 0, period: 0, effect: 0, param: 0 }};
    }}
    
    processTick() {{
        const patternIdx = modData.songPositions[this.currentPattern];
        
        // On tick 0, trigger new notes
        if (this.currentTick === 0) {{
            for (let ch = 0; ch < this.numChannels; ch++) {{
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
                        }} else if (note.period > 0) {{
                            // New note + sample: retrigger from start
                            state.samplePos = 0.0;
                            state.mixVol = 0;
                            state.volumeFade = 1.0;
                            state.volumeFadeInc = 0;
                            state.targetVolume = 1.0;
                            if (note.effect === 0x9) state.samplePos = note.param * 256;
                        }}
                        // note.period === 0 (sample-number-only, no note): ProTracker only
                        // resets volume (done above) — does NOT retrigger/restart the sample.
                        // Restarting here would prevent long pre-loop bodies from reaching
                        // their sustain loop (e.g. ENIGMA.MOD inst 14: loop at byte 9364/9490).
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
            for (let ch = 0; ch < this.numChannels; ch++) {{
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
                    state.period = Math.max({_pmin}, state.period - param);
                    state.basePeriod = state.period;
                }}
                break;
                
            case 0x2: // Portamento down
                if (!tick0 && param > 0) {{
                    state.period = Math.min({_pmax}, state.period + param);
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
        
        return Math.max({_pmin}, Math.min({_pmax}, effectivePeriod));
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
                if (channelMuted[ch]) continue;   // mute/solo gate (output only; effect state still advances elsewhere)
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
                    const _smpInfo = modData.sampleMap[state.sample] || {{}};
                    const smpFt = _smpInfo.finetune || 0;
                    const smpC2 = _smpInfo.c2spd || 8363;
                    // samplePos is in original (uncompressed) sample space.
                    // getSampleData maps it to compressed space via pos/bw_factor — so
                    // freq must NOT be divided by bw_factor here (would double-divide).
                    const freq = this.periodToFreq(effectivePeriod, smpFt, smpC2);
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

// Per-voice Solo (S) / Mute (M) button wiring.
for (let i = 0; i < _NCH; i++) {{
    const sB = document.getElementById('chS' + i);
    const mB = document.getElementById('chM' + i);
    if (sB) sB.addEventListener('click', () => {{
        channelSoloed[i] = !channelSoloed[i];
        sB.classList.toggle('on-s', channelSoloed[i]);
        _recomputeMutes();
    }});
    if (mB) mB.addEventListener('click', () => {{
        channelExplicitMuted[i] = !channelExplicitMuted[i];
        mB.classList.toggle('on-m', channelExplicitMuted[i]);
        _recomputeMutes();
    }});
}}

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
    // Auto-gain: scale so the loudest sample this frame hits ~80% of the
    // visible range.  Clamp gain between 1× (already loud) and 8× (very quiet).
    let _peak=0; for(let i=0;i<buf.length;i++) _peak=Math.max(_peak,Math.abs(buf[i]));
    const _gain=_peak>0.01?Math.min(8.0,0.8/_peak):8.0;
    ctx2d.beginPath();ctx2d.strokeStyle='#3d8ef0';ctx2d.lineWidth=1.8;
    ctx2d.shadowBlur=8;ctx2d.shadowColor='#3d8ef0';
    for(let x=0;x<W;x++){{
      const y=H/2-Math.max(-0.95,Math.min(0.95,buf[Math.floor(x*step)]*_gain))*H*0.45;
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

  const nc = _NCH;
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
    // Horizontal mouse/trackpad scroll to preview tracks
    let _wheelAccum = 0;
    trackerEl.addEventListener('wheel', function(ev){{
      ev.preventDefault();
      // Use deltaX for trackpad horizontal swipe; fall back to deltaY
      const delta = Math.abs(ev.deltaX) > Math.abs(ev.deltaY) ? ev.deltaX : ev.deltaY;
      _wheelAccum += delta;
      const THRESHOLD = 60;
      if(Math.abs(_wheelAccum) >= THRESHOLD){{
        const dir = _wheelAccum > 0 ? 1 : -1;
        _wheelAccum = 0;
        const next = Math.max(0, Math.min(maxFirstTrack, firstTrack + dir));
        if(next !== firstTrack){{
          firstTrack = next;
          updateHeaderAndRange();
          updateTracker();
        }}
      }}
    }}, {{passive: false}});
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
            print(f"   🎹 Visualizer: {len(diag)} sample waveforms classified")
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
            no_surround, no_fat, no_phatbass, reverb_2x2, fft_n, extra_pragmas. Missing
            keys default to permissive values (full-quality mode)."""

    # Compat defaults — used when the caller didn't pass a compat dict, or
    # when it passed one missing some keys. These match v1.37 default behavior.
    _compat = {
        'no_surround':    False,
        'no_fat':         False,
        'no_phatbass':    False,
        'reverb_2x2':     False,
        'fft_n':          512,
        'extra_pragmas':  False,
        'phatbass_mode':  'sample',  # 'auto' | 'sample' | 'mix'
    }
    if compat:
        _compat.update(compat)

    # Warn early when high channel count + 3D surround may overflow ANGLE private-var budget.
    # ANGLE's Metal backend inlines _gcoBody into the channel loop; with NUM_CHANNELS > 8
    # and ENABLE_3D=1, the inlined channel loops triple (main + two 3D taps), causing
    # "Total size of declared private variables exceeds implementation-defined limit"
    # on iOS/mobile WebGL. --no-surround removes the two extra channel loops.
    _nch_warn = getattr(mod, 'num_channels', 4)
    if _nch_warn > 8 and not _compat.get('no_surround', False):
        print(f"   ⚠️  {_nch_warn} channels + ENABLE_3D=1: the 3D surround adds 2 extra channel-loop inlines")
        print(f"      ({_nch_warn}ch × 3 passes). May exceed ANGLE private-var limit on iOS/mobile →")
        print(f"      WebGL 'private variables' error. Rebuild with --no-surround to fix.")
        print(f"      (--no-phatbass also helps if the error persists.)")

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
        8: "Skywalker (orblivius — synchronized flying-curve terrain + star field + cloud)",
        9: "Music is in the DNA (jaszunio15/enbe fork — DNA helix + parallax dunes)",
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
    MAX_TOTAL_SIZE = 160000  # embedded-data threshold (GLSL source size).
                             # NOT a PNG cap — a PNG holds 1024*1024*4 = 4MB.
                             # Above this the data goes in a PNG instead of
                             # inline GLSL. (This user needs embedded — keep
                             # the real ShaderToy tabs VQ-encoded, not PNG.)
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
            print(f"   🖼️  HTML player: raw sample data {total_size:,} B > {MAX_TOTAL_SIZE:,} B "
                  f"→ using external data PNG (HTML only; ShaderToy GLSL is always VQ-embedded, unaffected)")
    
    # Prepare sample map — each sample gets 32-byte zero-padding for cubic interpolation
    all_samples = []
    sample_map = []
    
    _loop_fr = []   # (sample idx, adaptive factor that was overridden → 1)
    for _smp_i, smp in enumerate(mod.samples):
        start_idx = len(all_samples)
        raw_len = 0
        bw_factor = 1
        if smp['data'] is not None and len(smp['data']) > 0:
            bw_factor, compressed = bw_compress_sample(smp['data'])
            # Looped samples must NOT be decimated (floor-divided loop bounds
            # → per-iteration position error → audible periodic wow on
            # sustained notes).  Restore full-rate (exact int loop bounds);
            # mirrors bw_compress_sample's factor-1 return exactly.
            if (LOOP_NO_DOWNSAMPLE and bw_factor != 1
                    and smp.get('repeat_length', 0) > 2):
                _loop_fr.append((_smp_i, bw_factor))
                bw_factor = 1
                compressed = smp['data'].astype(np.float32).astype(np.int8)
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
    if _loop_fr:
        _extra = sum(len(mod.samples[_i]['data']) -
                     (len(mod.samples[_i]['data']) // _f)
                     for _i, _f in _loop_fr)
        print(f"   ✓ loop-fullrate: {len(_loop_fr)} looped sample(s) kept "
              f"un-decimated for exact loop bounds "
              f"(was ×{sorted({_f for _,_f in _loop_fr})}; "
              f"+{_extra} raw samples → larger Sound, watch GPU budget). "
              f"Set LOOP_NO_DOWNSAMPLE=False to revert.")

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
        # default=1 — a module can legitimately have no raw sample bytes
        # here (VQ-embedded builds keep samples in the codebook, not in
        # sample_bytes_data). max([]) used to hard-crash the whole render;
        # a 1-int placeholder array keeps the GLSL valid (NUM_SAMPLE_CHUNKS
        # is 0 so it's never indexed).
        max_pat_chunk    = max(pat_chunk_sizes, default=1)
        max_smp_chunk    = max(smp_chunk_sizes, default=1)
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
    # to expect, and (in mainSound) silence after song ends so it plays once
    # fill the buffer instead of going silent partway through.
    _bpm_init   = getattr(mod, 'initial_tempo', 125)
    _speed_init = getattr(mod, 'initial_speed', 6)
    _ticks_per_sec = _bpm_init * 2.0 / 5.0
    _row_time      = _speed_init / _ticks_per_sec if _ticks_per_sec > 0 else 0.0
    # _actual_song_seconds is the variable-speed total from the row-time walk;
    # use it whenever F effects shifted the duration away from the naive
    # initial-speed estimate.
    _song_seconds  = _actual_song_seconds if _actual_song_seconds > 0 else _total_song_rows * _row_time
    _SHADERTOY_AUDIO_CAP_SEC = SHADERTOY_AUDIO_CAP_SEC  # module constant (top of file)
    _mins, _secs = divmod(_song_seconds, 60)
    print(f"   ⏱️  Song duration: {int(_mins)}m {_secs:.1f}s ({_total_song_rows} rows @ {_bpm_init} BPM, speed {_speed_init})")
    if _song_seconds > _SHADERTOY_AUDIO_CAP_SEC:
        _over = _song_seconds - _SHADERTOY_AUDIO_CAP_SEC
        _trunc_pct = (_SHADERTOY_AUDIO_CAP_SEC / _song_seconds) * 100.0
        print(f"   ⚠️  {_over:.1f}s OVER the {_SHADERTOY_AUDIO_CAP_SEC:.0f}s Shadertoy audio cap "
              f"— plays first {_SHADERTOY_AUDIO_CAP_SEC:.0f}s ({_trunc_pct:.0f}%), "
              f"last {100-_trunc_pct:.0f}% cut. Trimming pattern/sample data to "
              f"{_SHADERTOY_AUDIO_CAP_SEC:.0f}s would also shrink the build.")
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
    # Instrument count: S3M/IT can exceed 31 (2ND_PM=54). Hardcoded 31
    # silenced instruments 32+ (voice/vocal samples) — feedback_s3m_
    # instrument_cap, ported onto v1.61. Used by _gcoBody guard, isBass[],
    # samples[] across Common+Sound; must be defined before common_glsl.
    # DYNAMIC instrument/sample count — sized to the actual file, never a
    # hardcoded cap. MOD naturally yields 31 (its fixed header), S3M/XM/IT
    # yield their real count (2ND_PM=54). `mod.samples` for S3M can be the
    # short MOD-default list, so take the max of every real source; floor
    # of 1 only to avoid a zero-length GLSL array. (feedback_s3m_instrument_cap)
    _NSMP = max(1, len(mod.samples),
                int(getattr(mod, 'num_instruments', 0) or 0),
                int(getattr(mod, 'num_samples', 0) or 0),
                len(getattr(mod, 'instruments', []) or []),
                len(getattr(mod, 'samples', []) or []))
    data_source_comment = "Embedded data (no PNG required)" if use_embedded else f"All data in 1024×1024 RGBA PNG: {png_file}"
    common_glsl = f"""/* ============================================================================
   GLSL (The Last) MOD Player v1.61 (c) 2026 Orblivius
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
// enable3D          : Only3D surround widening (2-tap precomputed allpass)
// enablePhatBass    : PhatBass low-shelf + Haas cross-pan on bass instruments
// enableFAT         : FAT4X harmonic exciter (cs1 polynomial) on master
// enableVelvetReverb: sparse 6-tap velvet-noise reverb (opt-in; default off)
// surr_channels: 1-indexed channel pair that gets Only3D (the other two = dry center)
//   ivec2(1,4) = outer LEFT pair (ch0,ch3) — default Amiga layout
//   ivec2(2,3) = inner RIGHT pair (ch1,ch2) — swap surround and center
// Each `const bool` below is converted to a `#define ENABLE_X 0|1` + `#if`
// by the _flagdef post-process so a disabled feature's code is PHYSICALLY
// removed by the GLSL preprocessor (keeps the private-var budget down on
// GPUs with a tight declared-variable limit).
const bool  enable3D           = {str(not _compat["no_surround"]).lower()};
const bool  enablePhatBass     = {str(not _compat.get("no_phatbass", _compat["no_fat"])).lower()};
const bool  enableFAT          = {str(not _compat["no_fat"]).lower()};
const bool  enableVelvetReverb = false;   // reverb OFF by default (opt-in)
const bool  enableCombReverb   = false;   // reverb OFF by default (opt-in)
const ivec2 surr_channels = ivec2(1, 4);  // 1-indexed; change to ivec2(2,3) to flip
// Master volume (IT global·mix/128²). MOD/S3M/XM = 1.0; <1.0 for hot IT files.
const float MASTER_GAIN = 1.0;

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
//   this duration — after the song ends, silence fills the buffer.
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
#define INTRO_SILENCE_S  {getattr(mod, '_intro_silence_s', 0.0):.3f}
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
    int start, smpLen, loopStart, loopLen, volume, bwFactor;
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
    int vol_col;   // IT vol-column note volume override (0=absent, 1-64=explicit note vol)
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
    int b4 = getPatternByte(baseIdx + 4);
    Note n;
    n.instrument = (b0 & 0xF0) | ((b2 >> 4) & 0x0F);
    n.period     = ((b0 & 0x0F) << 8) | b1;
    n.effect     = b2 & 0x0F;
    n.param      = b3;
    n.vol_col    = b4;    // IT note-volume override (0 = absent)
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

    if (trigNote.instrument <= 0 || trigNote.instrument > {_NSMP} || trigNote.period <= 0)
        return 0.0;

    SampleInfo smp = samples[trigNote.instrument - 1];
    if (smp.smpLen == 0) return 0.0;

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
    }} else if (fSamplePos >= float(smp.smpLen)) {{
        return 0.0;
    }}
    if (fSamplePos < 0.0) return 0.0;

    float s = getSampleF(smp.start, fSamplePos, smp.smpLen, smp.loopStart, smp.loopLen);

    // ── Volume: forward scan trigger→current to honour Cxx cuts & Axx slides ─
    // ProTracker volume slide (Effect A/6) SKIPS tick 0 → applies (SPEED-1) ticks per row.
    int volume = smp.volume;
    // Trigger-row effect — Cxx takes priority, then IT vol-col, then vol-slide
    if (trigNote.effect == 0xC) {{
        volume = min(trigNote.param, 64);
    }} else if (trigNote.vol_col > 0) {{
        // IT volume-column note volume (for cells that have both a vol-col and
        // a non-Cxx effect — encoded in byte 4 of the pattern cell)
        volume = min(trigNote.vol_col, 64);
    }} else if (trigNote.effect == 0xA || trigNote.effect == 0x6) {{
        int _su=(trigNote.param>>4)&0xF, _sd=trigNote.param&0xF;
        int _delta;
        if (_su==0xF && _sd>0)       _delta = -_sd;
        else if (_sd==0xF && _su>0)  _delta = _su;
        else {{
            // Bug fix: when we ARE still in the trigger row use per-tick elapsed
            // count (matches ProTracker tick-by-tick vol-slide); for PAST trigger
            // rows (forward-scan callers) use full (trigSpeed-1) ticks as before.
            int _tTicks = (trigRow==pos.row && trigPat==pos.songPos)
                          ? max(0, int(pos.tick) - 1) : trigSpeed - 1;
            _delta = (_su>0?_su:-_sd) * _tTicks;
        }}
        volume = clamp(volume + _delta, 0, 64);
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
            if (_fn.period>0) break; // new pitch = new trigger; stop scan
            // Instrument-only row (period=0, instrument>0): ProTracker resets
            // channel volume to the new instrument's default without retriggering.
            // Do NOT break — this is still under the original trigger's frequency.
            if (_fn.instrument>0 && _fn.period==0 && _fn.instrument<={_NSMP})
                volume = samples[_fn.instrument-1].volume;
            if (_fn.effect==0xC)
                volume = min(_fn.param, 64);
            else if (_fn.effect==0xA || _fn.effect==0x6) {{
                int _su=(_fn.param>>4)&0xF, _sd=_fn.param&0xF;
                int _delta;
                if (_su==0xF && _sd>0)       _delta = -_sd;
                else if (_sd==0xF && _su>0)  _delta = _su;
                else {{ int _fGlobalRow=patRowOffset[_fp]+(_fr-patStartRow[_fp]); int _fSpd=rowSpeed[_fGlobalRow]; _delta=(_su>0?_su:-_sd)*(_fSpd-1); }}
                volume = clamp(volume + _delta, 0, 64);
            }}
            _fr++;
            if (_fr >= 64) {{ _fr=0; _fp++; }}
        }}
        // Current row: Cxx fully, Axx for elapsed ticks (tick 0 skipped).
        // Also handle instrument-only rows (period=0, instrument>0) which
        // in ProTracker reset the channel volume to that instrument's default.
        Note _cr = getNote(pos.songPos, pos.row, ch);
        // Vol reset for instrument-only row (--- sXX ...) — applies before effects
        if (_cr.instrument>0 && _cr.period==0 && _cr.instrument<={_NSMP})
            volume = samples[_cr.instrument-1].volume;
        if (_cr.period<=0) {{  // no new pitch → apply vol effects for this row
            if (_cr.effect==0xC)
                volume = min(_cr.param, 64);
            else if (_cr.effect==0xA || _cr.effect==0x6) {{
                int _su=(_cr.param>>4)&0xF, _sd=_cr.param&0xF;
                int _delta;
                if (_su==0xF && _sd>0)       _delta = -_sd;
                else if (_sd==0xF && _su>0)  _delta = _su;
                else {{ int _ticks = max(0, int(pos.tick) - 1); _delta = (_su>0?_su:-_sd)*_ticks; }}
                volume = clamp(volume + _delta, 0, 64);
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
    
    # _NSMP defined above (before common_glsl) — instrument count, not 31.
    bass_sample_flags = [
        'true' if (i < len(mod.samples) and mod.samples[i]['length'] > 0
                   and _is_bass_sample(mod.samples[i])) else 'false'
        for i in range(_NSMP)
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

    # ── Only3D allpass coefficients (precomputed; ported from mod_player.py) ──
    # All coeffs are pure functions of freq+SR → bake to literal const vec2 so
    # the shader does zero per-sample trig. d=tan(f·π/SR); p0=sin(d)/(cos+sin);
    # p1+p2=cos(d)/(cos+sin); delay=0.5/f.
    _ONLY3D_FREQ1 = 700.0
    _ONLY3D_FREQ2 = 2500.0
    _ONLY3D_DEPTH = 0.2
    _ONLY3D_SAT   = 1.0
    _ONLY3D_SR    = 44100.0
    def _only3d_coeffs(freq, sr=_ONLY3D_SR):
        d     = np.tan(freq * np.pi / sr)
        sin_d = np.sin(d); cos_d = np.cos(d)
        denom = cos_d + sin_d
        return float(sin_d / denom), float(cos_d / denom), float(0.5 / freq)
    _ap_p0_1, _ap_p1p2_1, _ap_delay_1 = _only3d_coeffs(_ONLY3D_FREQ1)
    _ap_p0_2, _ap_p1p2_2, _ap_delay_2 = _only3d_coeffs(_ONLY3D_FREQ2)

    sound_glsl = f"""/* ============================================================================
   GLSL (The Last) MOD Player v1.61 (c) 2026 Orblivius
   4+ Tracks support, S3M/MOD loader, 3D Surround, PhatBass, Comb Reverb, FAT, RVQ sample compression, configurable resampler
   SOUND TAB
   Visualizer: {viz_name}
   Contact: subband@gmail.com or
            subband@protonmail.com
   GIT:     https://github.com/mewza/mod2glsl
  ============================================================================ */
// getByte / getPatternByte / getSample / getNote / getChannelOutput are in Common.

// Bass sample flags (true = instrument detected as bass) — for PhatBass
const bool isBass[{_NSMP}] = bool[]({bass_flags_str});

// PhatBass routing: 0 = per-sample (uses isBass[]), 1 = mix-wide (no detection)
#define PHATBASS_MIX_MODE {phatbass_mix_mode}

// ── FAT4X harmonic exciter helper ────────────────────────────────────────
// cs1 polynomial waveshaper (even harmonics only) from FAT4X by Orblivius.
// Even-power series → zero at x=0, adds warm even harmonics, soft clip near ±1.
// Soft limiter (tanh) — smooth saturation, no hard-clip clicks. Used to
// (a) bound fat_cs1's input so its even-power polynomial can't blow up on
// transients (that was the audible crackle), and (b) gently limit the
// final mix instead of a hard clamp.
float _softlim(float x) {{ return tanh(x); }}

float fat_cs1(float x) {{
    float x2=x*x, x4=x2*x2, x6=x4*x2, x8=x4*x4, x10=x4*x6, x12=x6*x6;
    return 0.4375 - 0.3228759765625*x2 + 0.1123046875*x4
         - 0.50537109375*x6 + 0.1993408203125*x8
         + 0.634521484375*x10 - 0.6513671875*x12;
}}
// vec2 overload (ported from mod_player.py) — both channels in one call.
vec2 fat_cs1(vec2 x) {{
    vec2 x2=x*x, x4=x2*x2, x6=x4*x2, x8=x4*x4, x10=x4*x6, x12=x6*x6;
    return vec2(0.4375) - 0.3228759765625*x2 + 0.1123046875*x4
         - 0.50537109375*x6 + 0.1993408203125*x8
         + 0.634521484375*x10 - 0.6513671875*x12;
}}
// ── Master limiter — rational soft-knee, stateless (ported from mod_player.py)
// over=max(|x|-T,0); reduced=HEAD·over/(over+HEAD); y=sign·(min(|x|,T)+reduced)
// T=0.85 knee, CEIL=1.0 → keeps fat_cs1 strictly in [-1,1] (its even-power
// series goes non-monotonic past 1.0). Below T = bit-perfect pass-through.
vec2 softLimit(vec2 x) {{
    const float T    = 0.85;
    const float CEIL = 1.0;
    vec2 ax = abs(x);
    if (T >= CEIL) {{ return sign(x) * min(ax, vec2(CEIL)); }}
    const float HEAD = CEIL - T;
    vec2 over    = max(ax - T, vec2(0.0));
    vec2 reduced = (HEAD * over) / (over + HEAD);
    return sign(x) * (min(ax, vec2(T)) + reduced);
}}
// Mono mix at a time offset (for velvet-reverb taps). Mirrors mod_player.py.
float getMixedMono(float time_offset, Position pos, float rowTime) {{
    float mix = 0.0;
    for (int ch = 0; ch < NUM_CHANNELS; ch++)
        mix += getChannelOutput(ch, time_offset, pos, rowTime);
    const float normFactor = 2.0 / float(NUM_CHANNELS);
    return mix * normFactor;
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
    // Clamp to song duration — return silence after song ends.
    // SONG_DURATION_S is computed at compile time; convert to integer samps
    // using Common's BPM/SPEED — Common-scope #defines are visible here.
    const int SONG_DURATION_SAMPS = int(SONG_DURATION_S * float(SAMP_PER_SEC));
    if (play_samp >= SONG_DURATION_SAMPS) return vec2(0.0);
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

    // ── Silence after song ends ────────────────────────────────────────────
    // Shadertoy renders the Sound pass ONCE into a pre-allocated audio
    // buffer (~180 seconds at 44.1kHz). After the song finishes
    // (play_samp >= SONG_DURATION_SAMPS) we return silence so the song
    // plays once and stops cleanly — no looping back to position 0.
    // Longer songs (>180s) still cut off at the buffer cap regardless.
    //
    // INTRO_SILENCE offset: subtracted from `samp` BEFORE the duration
    // check so the song starts at row 0 when audio actually begins.
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
    
#if USE_TIMELINE_DSP
    // IT NNA path: mod_player.py's refined 64-voice ITPlayer baked
    // NNA / DCT / DCA / fadeout into the VoiceSegment timeline; tlGetOutput
    // sums the active segments. The entire pattern-player dry/3D/PhatBass
    // path below is preprocessor-removed for IT (no double-mix, no wasted
    // per-sample pattern walk). v1.61's downstream limiter/FAT/comb still
    // apply to _out.
    vec2 _out = tlGetOutput(playbackTime);
#else
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
    
    // ── Only3D — 2-tap precomputed allpass (ported from mod_player.py) ──────
    // Direct port of Only3D.h (Dmitry Boldyrev / mss): two 1st-order allpass
    // filters at different freqs make phase-shifted copies of the stereo
    // difference, cross-mixed to widen. Coeffs are pure functions of freq+SR
    // → baked to literal const vec2 in Python (zero per-sample trig).
    //   d=tan(f·π/SR); p0=sin(d)/(cos+sin); p1+p2=cos(d)/(cos+sin); delay=0.5/f
    const float ONLY3D_DEPTH = {_ONLY3D_DEPTH};
    const float SATURATION   = {_ONLY3D_SAT};
    const vec2 AP_P0         = vec2({_ap_p0_1:.6f}, {_ap_p0_2:.6f});
    const vec2 AP_P1_PLUS_P2 = vec2({_ap_p1p2_1:.6f}, {_ap_p1p2_2:.6f});
    const vec2 AP_DELAY      = vec2({_ap_delay_1:.7f}, {_ap_delay_2:.7f});
    if (enable3D) {{
        // 2-tap parallel allpass (v1.45 path) — wider, smoother.
        vec2 t = vec2(playbackTime) - AP_DELAY;
        if (t.x >= 0.0 && t.y >= 0.0) {{
            Position pos1 = getPosition(t.x);
            Position pos2 = getPosition(t.y);
            float diffNow = surrL - surrR;
            vec2 wL = vec2(0.0);
            vec2 wR = vec2(0.0);
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
                int  inst1   = getNote(pos1.songPos, pos1.row, ch).instrument;
                bool isBass1 = (inst1 >= 1 && inst1 <= {_NSMP}) ? isBass[inst1 - 1] : false;
                if (isBass1) continue;
                float panR = 0.125 + 0.75 * channelPan[ch];
                vec2  pan  = vec2(1.0 - panR, panR);
                vec2 sxy = vec2(getChannelOutput(ch, t.x, pos1, rowTime),
                                getChannelOutput(ch, t.y, pos2, rowTime));
                wL += sxy * pan.x;
                wR += sxy * pan.y;
            }}
            wL *= normFactor;
            wR *= normFactor;
            vec2 diffDelayed = wL - wR;
            vec2 ap = diffNow * AP_P0 + diffDelayed * AP_P1_PLUS_P2;
            // Soft saturation x/sqrt(1+x²·SAT) — inversesqrt is one HW op.
            vec2 dd = ap * inversesqrt(1.0 + ap * ap * SATURATION);
            float shuffle = (dd.x - dd.y) * ONLY3D_DEPTH;
            surrL += shuffle;
            surrR -= shuffle;
        }}
    }}
    
    // ── PhatBass — LIGHT low-shelf bass boost (mix-wide, 1 tap) ─────────────────
    // The old version called getChannelOutput THREE times per channel (now/0.5ms/
    // 8ms) → 3 inlined copies of the heavy mixer in the Sound tab = very slow
    // ANGLE compile ("loading forever"). This rebuilds it light, the way it used
    // to be: reuse the ALREADY-computed dry mix for the "now" tap (free —
    // surrL+surrR+centL+centR == the normalized mono mix), plus ONE getMixedMono
    // tap at 0.5 ms for the 2-tap boxcar low-pass (first null ~1 kHz → passes
    // bass, kills mids). One mixer inline instead of three → ~3× faster load.
    // Bass DOUBLED (mono boost added to both channels) for a pronounced low end;
    // bass is centered so mono is fine, and the downstream soft-limiter catches
    // peaks. Haas widening dropped (it was the 3rd, costliest tap). Ear-verify.
    const float PHAT_SHELF_T     = 0.0005;  // 0.5 ms (≈22-sample) LPF spacing
    const float PHAT_SHELF_DEPTH = 0.7;     // mono-to-both ≈ 2× the old panned-split bass
    vec2 _phatPB = vec2(0.0);
    if (enablePhatBass) {{
        float tA = playbackTime - PHAT_SHELF_T;
        if (tA >= 0.0) {{
            float dryMono = surrL + surrR + centL + centR;   // == getMixedMono(playbackTime), free
            float lpMono  = 0.5 * (dryMono + getMixedMono(tA, getPosition(tA), rowTime));
            _phatPB = vec2(lpMono * PHAT_SHELF_DEPTH);       // low-shelf bass → both channels
        }}
    }}

    vec2 _out = vec2(surrL + centL, surrR + centR);
    _out += _phatPB;   // chain: 3D → PhatBass → ...
#endif // USE_TIMELINE_DSP

    // ── Velvet-noise reverb — sparse 6-tap, ±sign, exp decay (opt-in) ───────
    if (enableVelvetReverb) {{
        const int   _VELV_N    = 6;
        const float _VELV_T[6] = float[6](0.011, 0.019, 0.027, 0.038, 0.052, 0.071);
        const float _VELV_S[6] = float[6](+1.0, -1.0, +1.0, +1.0, -1.0, +1.0);
        const float _VELV_RT60 = 0.060;
        const float _VELV_WET  = 0.18;
        // Reverb on the FULL mixed-down signal (no bass distinction).
        vec2 _vw = vec2(0.0);
        for (int _vi = 0; _vi < _VELV_N; _vi++) {{
            float _vtt = playbackTime - _VELV_T[_vi];
            if (_vtt < 0.0) continue;
            Position _vp = getPosition(_vtt);
            float _vdry = getMixedMono(_vtt, _vp, rowTime);
            float _vamp = exp(-_VELV_T[_vi] / _VELV_RT60) * _VELV_S[_vi];
            _vw.x += _vdry * _vamp;
            _vw.y -= _vdry * _vamp;     // L/R polarity flip → width
        }}
        _out += _vw * _VELV_WET;
    }}

    // Rational soft-knee BEFORE FAT4X (keeps fat_cs1 arg strictly in [-1,1]).
    _out = softLimit(_out);

    // ── FAT4X harmonic exciter (stateless, vec2) — ported from mod_player.py ─
    // cs1 even-harmonic waveshaper. FAT_AMOUNT 0.5 (+the inner ×0.5) tuned so
    // bass-heavy mixes don't stack gain past the ceiling.
    if (enableFAT) {{
        const float FAT_AMOUNT = 0.5;
        _out = _out * (1.0 + 0.5 * fat_cs1(_out) * FAT_AMOUNT);
    }}

    // ── End-of-chain soft-limit (UNCONDITIONAL) — guarantees |_out| ≤ 1 ────
    {{
        vec2 _ax = abs(_out);
        const float _T = 0.85;
        const float _C = 1.0;
        const float _H = _C - _T;
        vec2 _over    = max(_ax - _T, vec2(0.0));
        vec2 _reduced = (_H * _over) / (_over + _H);
        _out = sign(_out) * (min(_ax, vec2(_T)) + _reduced);
    }}

    // Hand off to v1.61's comb reverb + buffer-fade (kept active; mod_player.py
    // has these disabled but removing v1.61's working tail is not requested).
    float outL = _out.x;
    float outR = _out.y;

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
    // Gated OFF by default (enableCombReverb=false) — user uses velvet
    // reverb instead. The #if removal also reclaims the hottest path
    // (48 getChannelOutput + 12 getPosition/sample) from the GPU budget.
    if (enableCombReverb) {{
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
    }}  // end if(enableCombReverb)

    // ── Buffer-end fade-out ─────────────────────────────────────────────
    // Shadertoy's audio buffer is ~180 seconds (precomputed at compile time,
    // not streamed). When the OUTPUT reaches that limit, audio just stops — if
    // the last sample is mid-waveform at high amplitude, the cut is audible
    // as a "halt" or click. Fade the last 0.4s smoothly to zero so the
    // ending sounds intentional rather than truncated. 0.4s ≈ 17640 samples
    // at 44.1kHz — long enough to be smooth, short enough that user only
    // loses a fraction of a row at the very end.
    //
    // Drive this by the OUTPUT-buffer position float(samp)/SR (the monotonic
    // time actually written into the 180s buffer), NOT the song-position
    // `time`. They're equal on a normal first playthrough, but diverge when a
    // short song LOOPS to fill the buffer (there `time` wraps and never
    // reaches 180 → a hard cut with no fade). samp/SR is correct in every
    // case: the fade fires only as the real audio buffer runs out.
    const float BUFFER_CAP   = 180.0;
    const float FADE_LEN     = 0.4;
    float _outBufT = float(samp) / float(SAMP_PER_SEC);
    float _fadeT = clamp((BUFFER_CAP - _outBufT) / FADE_LEN, 0.0, 1.0);
    // Cosine ease-out: 1.0 at start of fade window, 0.0 at the end
    float _bufFade = 0.5 - 0.5 * cos(_fadeT * 3.14159265);
    outL *= _bufFade;
    outR *= _bufFade;

    // Master gain (IT global·mix; 1.0 for S3M/MOD) then rational soft-knee
    // — the comb reverb above adds post end-soft-knee, so a final limiter
    // guarantees no hard clip. softLimit (ported) replaces the old tanh.
    return softLimit(vec2(outL, outR) * MASTER_GAIN);
}}
"""
    
    
    # ========== IMAGE TAB ==========
    raw_title   = mod.title.strip() or "UNTITLED"
    title_text  = raw_title[:20]
    title_chars = to_glsl_font_chars(title_text)
    title_len   = len(title_text)
    # Format suffix (" (MOD)" or " (S3M)") rendered SEPARATELY in WHITE
    # right after the title (in YELLOW). Two prints, two colors.
    fmt_text    = (" (S3M)" if getattr(mod, 'is_s3m', False)
                   else " (XM)"  if getattr(mod, 'is_xm',  False)
                   else " (IT)"  if getattr(mod, 'is_it',  False)
                   else " (MOD)")
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
            "    float _scrollX=texelFetch(iChannel1,ivec2(5,2),0).r;\n"
            "    vec3 col = vec3(0.0);  // --viz 0: no visualizer"
        )
    elif viz == 3:
        viz_setup_block = (
            "    vec2 _uv=(C*2.-iResolution.xy)/iResolution.y;\n"
            "    float _scrollX=texelFetch(iChannel1,ivec2(5,2),0).r;\n"
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
            "    float _scrollX=texelFetch(iChannel1,ivec2(5,2),0).r;\n"
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
        p.yz *= _viz2_rot(1.6  + tt / 5.);
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
        float wm=chA[ch]*abs(sin(z*9.42+iTime*2.1+float(ch)*1.6));
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
        p.yz *= _v5_drot(1.6  + tt/5.);
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
const int   _v6_MAX_STEPS    = 72;   // perf: 100→72 (~28% fewer worst-case volumetric smoke-march iters; adaptive h*.5 step + scene-exit break keep far-smoke detail acceptable)
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
#define _v6_POI_COL_R     vec3(0.2,0.8,4.0)
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
    // Perf: the dancer is a small object; most screen rays never come near
    // it.  Cull rays whose infinite line misses a generous (R=2.0) bounding
    // sphere around the dancer BEFORE the 55-step SDF march (exact — the
    // dancer + poi reach is well inside R; h<0 means the line never enters
    // the sphere so it cannot hit the dancer; forward-only rays unaffected).
    {
        vec3 _v6_oc = ro - vec3(_v6_gDancerXZ.x,_v6_DANCER_BASE_Y,_v6_gDancerXZ.y);
        float _v6_bq = dot(_v6_oc, rd);
        if (_v6_bq*_v6_bq - (dot(_v6_oc,_v6_oc) - 4.0) < 0.0) {
            matId = -1.; return -1.;
        }
    }
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

    // ── Poi PLASMA trail — sub-sample fire (replaces point halos) ───────────────
    {
        const int   _v6_FIRE_N  = 14;
        const float _v6_FIRE_DT = 0.012;
        const int   _v6_FIRE_S  = 3;

        vec3  _pRW = vec3(0.0), _pLW = vec3(0.0);
        float _pcrR = 0.0, _parR = 0.0;
        float _pcrL = 0.0, _parL = 0.0;

        for (int i = 0; i < _v6_FIRE_N; i++) {
            float pastTime = iTime - float(i) * _v6_FIRE_DT;

            float ptR = pastTime * _v6_POI_RPM;
            float ptL = pastTime * _v6_POI_RPM
                      + _v6_PI*abs(sin(pastTime*.5))*_v6_POI_OFF
                      + _v6_PI*abs(cos(pastTime*.5))*(1.-_v6_POI_OFF);
            float ppose   = ((sin(pastTime * _v6_BPM_SCALE * _v6_PI)*0.5)+0.5)*0.25 * _v6_gAudioLevel;
            float pbs     = pow(abs(sin(pastTime*(float(BPM)/60.)*_v6_PI)), 3.0);
            float pyShift = 1.5 + pbs * 0.18 * _v6_gAudioLevel;

            vec3 pwR = _v6_poiWrist(ptR, ppose, +1.0);
            vec3 pwL = _v6_poiWrist(ptL, ppose, -1.0);
            vec3 pdR, ptanR, pdL, ptanL;
            _v6_poiHeadDir(ptR, pdR, ptanR);
            _v6_poiHeadDir(ptL, pdL, ptanL);

            vec3 phR = pwR + pdR * _v6_POI_STRING;
            vec3 phL = pwL + pdL * _v6_POI_STRING;
            phR.y -= pyShift;
            phL.y -= pyShift;

            float baseAge = float(i) / float(_v6_FIRE_N);
            phR.y += baseAge * 0.014;
            phL.y += baseAge * 0.014;

            vec3 phRW = _v6_toWorld(phR);
            vec3 phLW = _v6_toWorld(phL);

            vec3 nposR = phRW * 80. + vec3(iTime*4., -iTime*8., iTime*2.);
            vec3 nposL = phLW * 80. + vec3(iTime*4., -iTime*8., iTime*2.);
            float n1R = 0.5 + 0.5*_v6_noiseZ(nposR);
            float n2R = 0.5 + 0.5*_v6_noiseZ(nposR*2.5 + vec3(50.));
            float n3R = 0.5 + 0.5*_v6_noiseZ(nposR*7.0 + vec3(100.));
            float n1L = 0.5 + 0.5*_v6_noiseZ(nposL);
            float n2L = 0.5 + 0.5*_v6_noiseZ(nposL*2.5 + vec3(50.));
            float n3L = 0.5 + 0.5*_v6_noiseZ(nposL*7.0 + vec3(100.));
            float crR_n = n1R*0.55 + n2R*0.30 + n3R*0.15;
            float crL_n = n1L*0.55 + n2L*0.30 + n3L*0.15;
            float arR_n = pow(n2R, 6.0) * pow(n3R, 4.0) * 12.0;
            float arL_n = pow(n2L, 6.0) * pow(n3L, 4.0) * 12.0;

            if (i == 0) { _pRW=phRW; _pLW=phLW; _pcrR=crR_n; _parR=arR_n; _pcrL=crL_n; _parL=arL_n; }

            float sw = 1.0 / float(_v6_FIRE_S);
            for (int k = 0; k < _v6_FIRE_S; k++) {
                float kt  = float(k) / float(_v6_FIRE_S);
                vec3  sR  = mix(_pRW, phRW, kt);
                vec3  sL  = mix(_pLW, phLW, kt);
                float age = clamp((float(i) - 1.0 + kt) / float(_v6_FIRE_N), 0.0, 1.0);
                float crR = mix(_pcrR, crR_n, kt);
                float crL = mix(_pcrL, crL_n, kt);
                float arR = mix(_parR, arR_n, kt);
                float arL = mix(_parL, arL_n, kt);

                float dR = length(cross(rd, sR - ro));
                float dL = length(cross(rd, sL - ro));
                float fR = step(0.0, dot(sR - ro, rd));
                float fL = step(0.0, dot(sL - ro, rd));

                float sparkR=1./(0.002+dR*dR*8000.), coreR=1./(0.010+dR*dR*8500.);
                float bodyR =1./(0.006+dR*dR*1800.), haloR=1./(0.012+dR*dR* 600.);
                float sparkL=1./(0.002+dL*dL*8000.), coreL=1./(0.010+dL*dL*8500.);
                float bodyL =1./(0.006+dL*dL*1800.), haloL=1./(0.012+dL*dL* 600.);

                vec3 cCR=vec3(2.8), cBR=_v6_POI_COL_R*1.4+vec3(0.2);
                vec3 cHR=_v6_POI_COL_R*0.5, cAR=_v6_POI_COL_R*1.8+vec3(0.5);
                vec3 cCL=vec3(2.8), cBL=_v6_POI_COL_L*1.4+vec3(0.2);
                vec3 cHL=_v6_POI_COL_L*0.5, cAL=_v6_POI_COL_L*1.8+vec3(0.5);

                float inten = pow(1.0-age, 1.8) * (0.7 + 0.6*_v6_gAudioLevel);
                vec3 colo = vec3(0);
                colo += cCR*sparkR*inten*0.008*sw*fR + cCR*coreR*inten*0.020*sw*fR;
                colo += cBR*bodyR*inten*crR*0.022*sw*fR + cHR*haloR*inten*0.022*sw*fR;
                colo += cAR*sparkR*inten*arR*0.004*sw*fR;
                colo += cCL*sparkL*inten*0.008*sw*fL + cCL*coreL*inten*0.020*sw*fL;
                colo += cBL*bodyL*inten*crL*0.022*sw*fL + cHL*haloL*inten*0.022*sw*fL;
                colo += cAL*sparkL*inten*arL*0.004*sw*fL;
                col.rgb += .5 * tanh(colo);
            }

            _pRW=phRW; _pLW=phLW; _pcrR=crR_n; _parR=arR_n; _pcrL=crL_n; _parL=arL_n;
        }
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

    elif viz == 7:
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

    elif viz == 8:
        # Skywalker — orblivius's flying-curve terrain + synchronized star field
        # + cloud overlay. Returns full screen-space scene; drops the original
        # mainImage buffer-write logic (we can't persist state from Image tab,
        # so the kval[] star list always uses the deterministic time-based
        # fallback). Identifiers prefixed `_v8_`.
        viz_scene_block = r"""
// === VIZ 8: Skywalker (orblivius) ===========================================
// Flying-curve terrain raymarched per-row, synchronized with a star field
// (deterministic timestamp-based) and a cloud overlay sampled from iChannel2.
// iChannel2 is RGBA Noise / nebula texture in this Image-tab setup; iChannel1
// is Buffer A (audio + FFT). iChannel0 (alphabet) is untouched by this viz.
// ---------------------------------------------------------------------------

#define _v8_PI 3.14159265
#define _v8_MAX_STARS 8
#define _v8_ID 0.12321
#define _v8_NUM_SAMPLES 3

#define _v8_H(P)    fract(sin(dot(P,vec2(127.1,311.7)))*43758.545)
#define _v8_pR(a)   mat2(cos(a),sin(a),-sin(a),cos(a))
// Buffer A layout in this player: row 0 px 0..FFT_N-1 = raw audio samples
// (signed, summed across NUM_CHANNELS). The original Skywalker macros
// sampled a 2-row audio texture where y picked waveform-vs-FFT; we
// collapse both to row 0 (the waveform) and fold the y argument into the
// x-phase so each terrain-depth iteration still reads a different sample.
#define _v8_FUNC(x,y)    (texelFetch(iChannel1, ivec2(int(fract((x) + (y) * 0.31) * float(FFT_N)), 0), 0).r * 0.5)
#define _v8_CAPS2(xx,yy) (texelFetch(iChannel1, ivec2(int(fract((xx) + (yy) * 0.17) * float(FFT_N)), 0), 0).r * 0.35)

#define _v8_CURVE_SPEED   0.35
#define _v8_STAR_SPEED    0.5
#define _v8_CAMERA_SPEED  0.3
#define _v8_SYNC_BASE_TIME (iTime * 0.4)

vec3 _v8_hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float _v8_happy_star(vec2 uv, float anim) {
    uv = abs(uv);
    vec2 pos = min(uv.xy/uv.yx, anim);
    float p = (2.0 - pos.x - pos.y);
    return (2.0+p*(p*p-1.5)) / (uv.x+uv.y);
}

// _v8_iChannel0 — emulates the original Skywalker shader's iChannel0, which
// was a Buffer A pass that pre-rendered a 6-octave fractal-noise nebula from
// a noise source (iChannel2 in that pass). Our Image-tab iChannel0 is pinned
// to the alphabet font, so we recreate the Buffer-A bake INLINE here as a
// function. Anywhere the original shader read `texture(iChannel0, uv)` we
// instead call `_v8_iChannel0(uv)` to get the same nebula sample.
vec4 _v8_iChannel0(vec2 uv) {
    vec4 cloud =
        texture(iChannel2, uv * 0.25)  * 0.5      +
        texture(iChannel2, uv * 0.5 )  * 0.25     +
        texture(iChannel2, uv       )  * 0.125    +
        texture(iChannel2, uv * 2.0 )  * 0.0625   +
        texture(iChannel2, uv * 4.0 )  * 0.03125  +
        texture(iChannel2, uv * 8.0 )  * 0.015625;
    return pow(max(vec4(0.0), (1.0 - length(uv - 0.5) * 2.0) * cloud), vec4(4.0));
}

// ─── Voyage-style volumetric nebula ─────────────────────────────────────────
// Replaces the previous 6-octave fractal-noise bake background with a proper
// raymarched cosmic gas adapted from sebastien durand's "Voyage to the Stars"
// (CC BY-NC-SA 3.0): https://www.shadertoy.com/view/4dlGW2 — specifically the
// renderIntergalacticClouds path (mapIntergalacticCloud + SpiralNoiseC +
// HSV-cell point lights). The original used iChannel1 as a 256x256 noise
// texture for value noise; we route to iChannel2 (mod_player's noise slot).
// Step budget cut 100 -> 36 for in-viz framerate; outer loop preserves the
// original td/sum.a early-out behavior.
//
// NOTE: _v8_iChannel0 + _v8_f (cloud sparkle overlay) below are UNTOUCHED —
// they're a separate moving-foreground layer that depends on cheap close-UV
// texture samples for its motion-blur trick. Only the *background* nebula
// has been swapped.
// ----------------------------------------------------------------------------

float _v8_neb_hash11(float p) {
    vec3 p3 = fract(vec3(p) * .1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

float _v8_neb_pn(vec3 x) {
    vec3 p = floor(x), f = fract(x);
    f *= f*(3.-f-f);
    vec2 uv = (p.xy + vec2(37., 17.)*p.z) + f.xy;
    vec2 rg = textureLod(iChannel2, (uv+.5)/256., -100.).yx;
    return 2.4*mix(rg.x, rg.y, f.z) - 1.;
}

// otaviogood's spiral noise — successively adds/rotates sin waves with
// increasing frequency. Cheap, aperiodic, no hash table.
const float _v8_neb_nudge      = 20.0;
const float _v8_neb_normalizer = 1.0 / 20.024984394500787; // = 1/sqrt(1+20^2)

float _v8_neb_spiralNoise(vec3 p, vec4 id) {
    float iter = 2., n = 2. - id.x;
    for (int i = 0; i < 6; i++) {
        n += -abs(sin(p.y*iter) + cos(p.x*iter)) / iter;
        p.xy += vec2(p.y, -p.x) * _v8_neb_nudge;
        p.xy *= _v8_neb_normalizer;
        p.xz += vec2(p.z, -p.x) * _v8_neb_nudge;
        p.xz *= _v8_neb_normalizer;
        iter *= id.y + .733733;
    }
    return n;
}

float _v8_neb_map(vec3 p, vec4 id) {
    float k = 2.*id.w + .1;
    return k*(.5 + _v8_neb_spiralNoise(p.zxy*.4132 + 333., id)*3.
                 + _v8_neb_pn(p*8.5)*.12);
}

vec3 _v8_neb_hsv(float x, float y, float z) {
    return z + z*y*(clamp(abs(mod(x*6. + vec3(0,4,2), 6.) - 3.) - 1., 0., 1.) - 1.);
}

// Volumetric march. id = (lightRadiusRef, spiralFreqStep, densityBias, densityGain)
vec4 _v8_neb_render(vec3 ro, vec3 rd, float tmax, vec4 id) {
    float max_dist = min(tmax, 22.0);
    float td = 0., d, t, noi, lDist, a;
    const float sp = 9.;          // periodic light-cell spacing
    float rRef = 2.*id.x;
    float h    = .05 + .25*id.z;  // density edge threshold
    vec3  pos, lightColor;
    vec4  sum = vec4(0);

    // Small per-pixel start jitter — breaks step banding without showing as
    // visible static. Was .1 which produced speckle behind the tracker UI;
    // .02 is enough to dither the step boundary without per-pixel noise.
    t = .02 * _v8_neb_hash11(rd.x + rd.y*17.0 + rd.z*113.0);

    // Iteration count: 40 -> 24. Each step does a 6-iter spiral noise plus
    // a texture sample plus an HSV mix, so per-step cost is high. Stacked on
    // top of viz 8's existing 100-flare loop and 40-step terrain raymarch,
    // 40 nebula steps was tipping the GPU into frame-deadline misses, which
    // glitched the audio in Shadertoy. 24 steps + early-out keeps the look
    // while reclaiming ~40% of the nebula's cost.
    for (int i = 0; i < 24; i++) {
        if (td > .9 || sum.a > .99 || t > max_dist) break;
        a   = smoothstep(max_dist, 0., t);
        pos = ro + t*rd;
        d   = abs(_v8_neb_map(pos, id)) + .07;

        // Periodic point-light grid colors the gas pockets. Saturation .7 /
        // value .85 — saturated enough to read as nebula color, but not so
        // neon that it looks like a fluid sim.
        lDist = max(length(mod(pos + sp*.5, sp) - sp*.5), .001);
        noi   = _v8_neb_pn(.05*pos);
        lightColor = mix(_v8_neb_hsv(noi,     .7, .85),
                         _v8_neb_hsv(noi+.3,  .7, .85),
                         smoothstep(rRef*.5, rRef*2., lDist));
        // Light divisor /18 — midway between Voyage's /30 (too dim) and
        // /12 (blew out into solid white blobs).
        sum.rgb += a * lightColor / exp(lDist*lDist*lDist*.08) / 18.;

        if (d < h) {
            td += (1.-td)*(h-d) + .005;
            sum.rgb += sum.a * sum.rgb * .30 / lDist;   // emission
            sum    += (1. - sum.a) * .03 * td * a;       // density alpha
        }
        td += .015;
        t  += max(d * .08 * max(min(lDist, d), 2.), .01);
    }

    // NOTE: Voyage's `sum.xyz *= sum.xyz*(3.-sum.xyz-sum.xyz)` curve was
    // dropped. That formula peaks at x=0.75 with value 1.125, so it pushes
    // anything above ~0.2 alpha toward full white — fine when the nebula is
    // a thin atmosphere over a star field, but as a *background* it produced
    // glossy fluid-blob artifacts. Plain clamp gives proper gas falloff.
    return clamp(sum, 0., 1.);
}

// Nebula background — independent forward drift through the noise field so
// the gas evolves even when viz 8's main scene camera is spinning elsewhere.
// The "starts small, grows as approached" effect comes naturally from the
// fast forward camera motion: distant formations appear small, then enlarge
// on screen as the camera flies toward them. No cycle pulsing — earlier
// attempt with a fade-in/fade-out cycle made the nebula invisible 25% of
// the time and reset visible extent back to small, breaking continuity.
vec3 _v8_dtcmain(vec2 fragCoord) {
    vec2 uv = (fragCoord.xy - 0.5*iResolution.xy) / iResolution.y;

    // Fast forward drift — gives the "flying toward the nebula" punch.
    // Speed 0.15 means the camera covers ~1 nebula-feature unit every ~7
    // seconds, so formations visibly approach and pass.
    float ct = iTime * 0.15;
    vec3  ro = vec3(0.0, 0.0, ct);

    // Camera ray + very slow pan (no continuous yaw). Focal 1.9 (was 1.4)
    // pulls the gas back so formations read as smaller, more distant clouds.
    vec3  rd = normalize(vec3(uv, 1.9));
    rd.xz *= _v8_pR(0.18 * sin(ct*0.3));   // gentle horizontal pan
    rd.yz *= _v8_pR(0.10 * sin(ct*0.2));   // gentler vertical drift

    // (lightRef, freqStep, densityBias, densityGain). id.z slightly above
    // Voyage's .16 to thicken the gas a touch; id.w stays at default.
    vec4 id  = vec4(0.50, 0.40, 0.22, 0.75);
    vec4 neb = _v8_neb_render(ro, rd, 22.0, id);

    // Static center-weighted mask — matches the original _v8_iChannel0 bake
    // which used (1 - length(uv-0.5)*2)^4. Concentrates the gas as a defined
    // central formation; the forward drift naturally enlarges it as we
    // approach, no time-based cycle needed.
    float r   = length(uv);
    float vig = max(1.0 - r, 0.0);
    vig       = pow(vig, 3.0);

    // Linear base + squared highlight punch — squaring brightens bright cells
    // more than midtones, so the HSV pockets pop without flooding the gas.
    return (neb.rgb * 1.8 + neb.rgb * neb.rgb * 1.5) * vig;
}

vec4 _v8_f(vec2 uv, float t, float i) {
    float life = fract(i / float(_v8_NUM_SAMPLES) + t * .13);
    float fade = pow(life, 1.75) * (1.0 - pow(life, 16.0));
    vec4 color = 8.0 * fade * vec4(fract(i*0.33), fract(0.33+i*0.33), fract(0.67+i*0.33), 1.0);
    float s = 1.5 - 1.5 * life;
    float r = i * 2.73 + t * 0.05 * 0.5;
    vec2 cs = cos(r + vec2(0.0, -0.5) * 3.14);
    mat2 m = mat2(cs.x, -cs.y, cs.y, cs.x);
    // All 4 cloud taps sample the nebula function (same as the original
    // shader's `texture(iChannel0, …)` reads when iChannel0 was the Buffer-A
    // nebula bake). _v8_iChannel0 returns vec4 — take .r to match the
    // original's single-channel read.
    float samp1 = _v8_iChannel0((0.5 + m * uv * s)).r;
    float samp2 = _v8_iChannel0((0.5 + m * (uv - vec2(0.0, 0.05))   * s)).r;
    float samp3 = _v8_iChannel0((0.5 + m * (uv - vec2(0.0, 0.025))  * s)).r;
    float samp4 = _v8_iChannel0((0.5 + m * (uv - vec2(0.0, 0.0125)) * s)).r;
    return samp1
         * (1.0 + (samp2 - samp1) * 8.0)
         * (1.0 + (samp3 - samp1) * 7.0)
         * (1.0 + (samp4 - samp1) * 6.0)
         * color;
}

vec3 _VizScene(vec2 u) {
    float kval[_v8_MAX_STARS];
    vec2 uvTrue = u.xy / iResolution.xy;

    float syncTime   = _v8_SYNC_BASE_TIME * _v8_CURVE_SPEED;
    float starTime   = _v8_SYNC_BASE_TIME * _v8_STAR_SPEED;
    float cameraTime = _v8_SYNC_BASE_TIME * _v8_CAMERA_SPEED;

    vec2 curve = vec2(sin(syncTime) * 2.0 * 0.003,
                      sin(syncTime) * 2.0 * 0.001);

    vec4 o = vec4(0.0, 0.0, 0.0, 1.0);
    vec2 fc = u; fc.y = iResolution.y - fc.y;
    o = vec4(_v8_dtcmain(fc), 1.);

    // ── Terrain (synchronized curve) ──
    vec3 col11 = vec3(0);
    float d21 = 0.0;
    for (float i = 0.0; i <= 20.0; i += .5) {
        float Y1 = _v8_CAPS2(uvTrue.x, i/50.0);
        vec3 p = d21 * normalize(vec3(u + u, 0) - iResolution.xyx);
        p.xy += d21 * d21 * curve;
        if (d21 > 150.0) break;
        p.z += starTime * 20.0 + i * 0.25;
        p.y += 1.5 + _v8_FUNC(mod(p.x, p.y), i/20.0);
        float rotationAngle = syncTime * 0.0001;
        mat2 rotationMatrix = mat2(cos(rotationAngle), -sin(rotationAngle),
                                   sin(rotationAngle),  cos(rotationAngle));
        vec2 rotatedPos = rotationMatrix * p.xy;
        float s = 1.0 + max(rotatedPos.x, rotatedPos.y + i/2.0)
                - fract(p.z * 0.2 + i * 0.2) * 0.4
                - fract(p.z * 0.2 + i * 0.2) * 0.4;
        d21 += s;
        vec4 tc = (1.0 + cos(p.z + 1.0 * d21 + vec4(4,2,1,0))) / s;
        vec2 dd = vec2(1.0, 0.5 * Y1);
        float zIntensity = 1. + sin(p.z * 0.0002);
        col11 += 50.0 / exp(-2.0 * cos(i*.2 + sqrt(i) + vec3(0,1,3)) + 2.0)
               / max(i * sqrt(i), 0.01) * 0.15
               / (length(rotatedPos - clamp(dot(rotatedPos, dd) / dot(dd,dd), -3.0, 3.0) * dd) / i + i/1e9)
               * zIntensity;
    }
    col11 = tanh(col11 / 300.0);

    // ── Star ID generation (deterministic time-based) ──
    float timeStamp = floor(iTime / 4.0);
    int numActiveStars = int(1.0 + fract(sin(timeStamp * 45.123) * 43758.5453) * float(_v8_MAX_STARS));
    for (int i = 0; i < _v8_MAX_STARS; i++) {
        kval[i] = (i < numActiveStars)
            ? fract(sin(timeStamp * (12.9898 + float(i+1) * 78.233)) * 43758.5453) * 50.0
            : 0.0;
    }

    // ── Camera setup ──
    vec2 uv = (u - 0.5 * iResolution.xy - 0.5) / iResolution.y;
    uv *= 2.0;
    vec3 vuv = vec3(2.0 * sin(cameraTime), 1.0, sin(cameraTime));
    vec3 ro  = vec3(0.0, 0.0, 134.0);
    vec3 vrp = vec3(5.0, sin(cameraTime) * 60.0, 20.0);
    vrp.xz *= _v8_pR(cameraTime);
    vrp.yz *= _v8_pR(cameraTime * 0.2);
    vec3 vpn = normalize(vrp - ro);
    vec3 uu  = normalize(cross(vuv, vpn));
    vec3 rd  = normalize(vpn + uv.x * uu + uv.y * cross(vpn, uu));

    vec3 sceneColor      = vec3(0.0, 0.0, 0.3);
    vec3 flareCol        = vec3(0.0);
    vec3 flareIntensity  = vec3(0.0);

    for (float k = 0.0; k < 100.0; k++) {
        float r = _v8_H(vec2(k)) * 2.0 - 1.0;
        vec3 flarePos = vec3(
            _v8_H(vec2(k) * r) * 20.0 - 10.0,
            r * 10.0,
            mod(sin(k / 100.0 * _v8_PI * 4.0) * 15.0 - starTime * 13.0 * k * 0.007, 10.0)
        );
        float starDistance = length(flarePos.xy);
        flarePos.xy += starDistance * starDistance * curve;
        flarePos = normalize(flarePos);
        float v = abs(dot(flarePos, rd));
        flareIntensity += pow(v, 30000.0) * 2.0;
        flareIntensity += pow(v, 1e3) * 0.2;
        flareIntensity *= 1.0 - flarePos.z / 2.0;

        bool showStar = false;
        float bestZ = 10.0;
        for (int i = 0; i < _v8_MAX_STARS; i++) {
            if (kval[i] > 0.0 && k >= kval[i] && k < kval[i] + 1.0) {
                float starZ = mod(sin(kval[i] / 100.0 * _v8_PI * 4.0) * 15.0
                                  - starTime * 13.0 * kval[i] * 0.007, 10.0);
                if (starZ < bestZ) { bestZ = starZ; showStar = true; }
            }
        }
        if (showStar) {
            float starX = dot(flarePos, uu);
            float starY = dot(flarePos, cross(vpn, uu));
            vec2 origUV = (u - 0.5 * iResolution.xy) / iResolution.y;
            float closeness = 1.0 - flarePos.z;
            float starScale = 4.0 + closeness * 12.0;
            vec2 starUV = (origUV - vec2(starX, starY) / 2.4) * starScale;
            float starShape = _v8_happy_star(starUV, 1.0 + 0.5 * sin(iTime * 0.3));
            vec3 starCol = starShape * _v8_hsv2rgb(vec3(mod(iTime * 0.1, 1.0), 0.8, 1.0));
            starCol = min(0.2 * starCol, 6.0);
            flareIntensity += starCol;
        }
        flareCol += flareIntensity * vec3(sin(r * 3.12 - k), r, cos(k) * 2.0) * 0.3;
    }
    sceneColor += abs(flareCol);
    sceneColor = mix(sceneColor, sceneColor.rrr * 1.4, length(uv) / 2.0);

    // ── Cloud overlay ──
    vec4 sum = vec4(0.0);
    float rotationAngle = syncTime * 0.0001;
    mat2 rotationMatrix = mat2(cos(rotationAngle), -sin(rotationAngle),
                               sin(rotationAngle),  cos(rotationAngle));
    uv = rotationMatrix * uv;
    for (int i = 0; i < _v8_NUM_SAMPLES; ++i)
        sum += _v8_f(uv, iTime * .2, float(i));
    vec4 cloud = sum.bgra;

    // ── Swirl HSV overlay ──
    vec2 uv5 = u / iResolution.xy;
    uv5 -= 0.5;
    uv5 /= vec2(iResolution.y / iResolution.x, 1.0);
    uv5.y += 0.4;
    uv5 = rotationMatrix * uv5;
    float angle = atan(uv5.y, uv5.x) + starTime;
    float radius = length(uv5);
    float swirl = sin(radius * 10.0 - iTime * 3.0) * 0.5 + 0.5;
    float hue = fract(angle / (2.0 * _v8_PI) + swirl);
    float saturation = 1.0 - radius;
    float value = swirl;
    vec3 color3 = _v8_hsv2rgb(vec3(hue, 0.5 * saturation, value * 0.75));

    // ── Final composite ──
    // The original shader assigned `o = vec4(sceneColor + tanh(...), 1.0)`
    // here, which silently discarded the dtcmain nebula bake.  We use the
    // nebula as the actual background and additively layer the scene
    // (terrain + stars + swirl) on top, with a tiny dose of the cloud
    // overlay as fine atmospheric texture.  Cloud weight history:
    //   0.50 — original; flooded frame with bright cyan
    //   0.15 — earlier reduction; still produced particle spray over UI
    //   0.04 — current; cloud reads as faint texture, tracker stays clean
    vec3 nebulaBG = o.rgb;                                        // _v8_dtcmain output, has its own vignette + gamma
    // Terrain (col11) math is left at viz 8's original tuning — it's an
    // inherently sparkly inverse-distance accumulation across 40 rays.
    // Scaling it to 0.4 in the composite dims those sparkles so they stop
    // competing with the tracker UI, without changing what the terrain is
    // actually drawing.
    vec3 scene    = sceneColor + tanh(col11 * 0.4 + color3 * color3 * 0.35);
    scene        *= sqrt(max(scene, vec3(0.0)));                  // matches original `o *= sqrt(o)` curve
    o = vec4(nebulaBG + scene + cloud.rgb * 0.04, 1.0);
    return o.rgb;
}
"""

    else:  # viz == 9 — Music is in the DNA (jaszunio15 / enbe fork)
        # Fork of enbe's fork of jaszunio15's "Music is in the DNA": a DNA
        # double-helix over a parallax dune background. ShaderToy's
        # iChannel0(music FFT)/iChannel1(mic) audio reads are collapsed to
        # MOD2GLSL's Buffer-A row-0 waveform — this player exposes no real
        # FFT to the Image tab, so |sample| is used as a per-band amplitude
        # proxy (same convention as viz 3 / viz 8; visually equivalent).
        # Full mainImage()+mainImage2() folded into vec3 _VizScene();
        # identifiers prefixed `_v9_`.
        viz_scene_block = r"""
// === VIZ 9: Music is in the DNA (CC BY 3.0) =================================
// Fork of "Music is in the DNA" forked" by enbe. https://shadertoy.com/view/sXlGzj
// Author: Jan Mróz (jaszunio15). Parallax dunes + yellow palette + anti-clip.
// iChannel0/iChannel1 audio remapped to MOD2GLSL Buffer-A row-0 waveform.
#define _v9_TIME (iTime)
#define _v9_SIN_DENSITY 0.4
#define _v9_COLOR_DIFFERENCE 0.8
// ShaderToy texture(iChannelN, vec2(x,y)) → MOD2GLSL row-0 waveform sample.
// y folds into the x-phase so distinct (x,y) lookups hit distinct samples;
// |.| gives an amplitude proxy (no real FFT in the Image tab).
#define _v9_aud(x,y) (abs(texelFetch(iChannel1, ivec2(int(fract((x)+(y)*0.31)*float(FFT_N)),0),0).r))
#define _v9_BLEND(dest, srcCol, srcDepth, srcAlpha) dest = mix(dest, (srcCol) * (srcDepth), (srcAlpha))

float _v9_getDuneLayer(vec2 uv, float speed, float scale, float heightOffset) {
    float x = uv.x * scale + _v9_TIME * speed;
    float y = sin(x) * 0.15;
    y += sin(x * 0.43 + 2.0) * 0.25;
    y += sin(x * 1.7 + 1.0) * 0.05;
    y += heightOffset;
    float edge = 2.0 / iResolution.y;
    return smoothstep(y + edge, y - edge, uv.y);
}
float _v9_linearstep(float a, float b, float x) {
    return clamp((b - x) / (b - a), 0.0, 1.0);
}
vec2 _v9_circle(vec2 uv, float pixelSize, float sinDna, float cosDna, float _sign, float audioReact) {
    float height = _sign * sinDna;
    float depth = abs((_sign * 0.5 + 0.5) - (cosDna * 0.25 + 0.5));
    float rawSize = 0.15 + depth * 0.1 + (audioReact * 0.40);
    float size = min(rawSize, 0.48);
    float alpha = 1.0 - smoothstep(size - pixelSize, size + pixelSize,
                                   distance(uv, vec2(0.5, height)));
    return vec2(alpha, depth * _v9_COLOR_DIFFERENCE + (1.0 - _v9_COLOR_DIFFERENCE));
}
vec4 _v9_dna(vec2 i) {
    vec4 o = vec4(0.0);
    vec3 r = vec3(1.0, 0.0, 0.0);
    vec3 y = vec3(1.3, 0.8, 0.0);
    vec3 g = vec3(0.0, 1.0, 0.0);
    vec3 b = vec3(0.0, 0.0, 1.0);
    vec3 ca[19];
    ca[1]=b*2./6.+r*4./9.;  ca[2]=b*1./9.+r*5./3.;  ca[3]=r;
    ca[4]=r*22./6.+y/6.;    ca[5]=r*2./9.+y*2./6.;  ca[6]=r*3./9.+y*3./3.;
    ca[7]=r*7./6.+y*4./6.;  ca[8]=r*1./9.+y*5./6.;  ca[9]=y;
    ca[10]=y*2./3.+g*1./3.; ca[11]=y*1./3.+g*2./3.; ca[12]=g;
    ca[13]=g*2./5.+b*1./6.; ca[14]=g*1./5.+b*2./3.; ca[15]=b;
    ca[16]=b*5./6.+r*1./4.; ca[17]=b*4./6.+r*2./6.; ca[18]=b*3./6.+r*3./5.;
    vec2 uv = (i - 0.5 * iResolution.xy) / iResolution.y;
    float time = iTime;
    int c = 0;
    for (float z = 0.0; z < 20.0; z += 0.2) {
        c += 1;
        vec4 color = vec4(ca[(c % 18) + 1], 0.0);
        float depth = mod(z - time * 4.0, 20.0) + 0.1;
        float twist = depth * 1.2 + time * 0.8;
        float radius = 1.0;
        vec3 strand1 = vec3(cos(twist) * radius, sin(twist) * radius, depth);
        vec3 strand2 = vec3(cos(twist + 3.14159) * radius, sin(twist + 3.14159) * radius, depth);
        vec2 proj1 = strand1.xy / strand1.z;
        vec2 proj2 = strand2.xy / strand2.z;
        float audio = _v9_aud(depth * 0.02, 0.0) * 0.5 + 0.5;
        float fade = smoothstep(20.0, 2.0, depth);
        float pointSize = 0.008 * audio * fade / depth;
        o += color * pointSize / length(uv - proj1);
        o += color * pointSize / length(uv - proj2);
        if (mod(z, 1.0) < 0.2) {
            for(float f = 0.25; f <= 0.75; f += 0.25) {
                vec2 rungPos = mix(proj1, proj2, f);
                o += color * (pointSize * 0.5) / length(uv - rungPos);
            }
        }
    }
    o += vec4(0.0, 0.0, 0.05, 1.0);
    return o;
}
vec3 _VizScene(vec2 fragCoord) {
    vec2 screenUV = fragCoord.xy / iResolution.xy;
    vec2 bgUV = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec2 dnaUV = bgUV * 2.0;
    vec3 skyTop = vec3(0.8, 0.3, 0.2);
    vec3 skyBottom = vec3(1.0, 0.6, 0.3);
    vec3 bgFinal = mix(skyBottom, skyTop, screenUV.y);
    vec4 dnaTunnel = _v9_dna(fragCoord);
    float mask1 = _v9_getDuneLayer(bgUV, 0.1, 1.5, -0.1);
    bgFinal = mix(bgFinal, vec3(0.9, 0.45, 0.2), mask1);
    float mask2 = _v9_getDuneLayer(bgUV, 0.25, 2.2, -0.25);
    bgFinal = mix(bgFinal, vec3(0.75, 0.25, 0.15), mask2);
    float mask3 = _v9_getDuneLayer(bgUV, 0.45, 1.8, -0.4);
    bgFinal = mix(bgFinal, vec3(0.45, 0.1, 0.1), mask3);
    bgFinal = tanh(bgFinal * dnaTunnel.rgb * 2. * sqrt(max(dnaTunnel.rgb, vec3(0.0))));
    dnaUV *= 5.0;
    float angle = 0.3;
    dnaUV *= mat2(cos(angle + vec4(0,11,33,0)));
    dnaUV.x -= _v9_TIME * 0.5;
    float pixelSize = 10.0 / iResolution.y;
    vec2 baseUV = dnaUV;
    dnaUV.x = fract(dnaUV.x);
    float lineIndex = floor(baseUV.x);
    float dnaTimeIndex = lineIndex * _v9_SIN_DENSITY + _v9_TIME;
    float bandType = mod(abs(lineIndex), 4.0);
    float freq = 0.0;
    vec3 targetColor = vec3(1.0);
    if (bandType < 0.5)      { freq = 0.05; targetColor = vec3(0.85, 0.65, 0.12); }
    else if (bandType < 1.5) { freq = 0.30; targetColor = vec3(1.00, 0.84, 0.00); }
    else if (bandType < 2.5) { freq = 0.60; targetColor = vec3(0.98, 0.84, 0.48); }
    else                     { freq = 0.85; targetColor = vec3(0.95, 0.90, 0.67); }
    float rawAudio = _v9_aud(freq, 0.25) * 2.0;
    float audioReact = pow(smoothstep(0.1, 0.85, rawAudio), 2.0);
    if (rawAudio <= 0.01) {
        audioReact = pow(sin(_v9_TIME * 3.0 + lineIndex) * 0.5 + 0.5, 6.0) * 0.3;
    }
    vec3 elementColor = mix(vec3(1.0), targetColor, clamp(audioReact * 2.0, 0.0, 1.0));
    float amplitude = 2. + audioReact * 0.75;
    float sinDna = sin(dnaTimeIndex) * amplitude;
    float cosDna = cos(dnaTimeIndex) * amplitude;
    float lineSDF = abs(dnaUV.x - 0.5);
    float line = smoothstep(pixelSize * 2.0, 0.0, lineSDF);
    float sinCutLineUp = abs(sinDna);
    float sinCutMaskUp = smoothstep(sinCutLineUp + pixelSize, sinCutLineUp - pixelSize, dnaUV.y);
    float sinCutLineDown = -abs(sinDna);
    float sinCutMaskDown = smoothstep(sinCutLineDown - pixelSize, sinCutLineDown + pixelSize, dnaUV.y);
    vec2 circle1 = _v9_circle(dnaUV, pixelSize, sinDna, cosDna,  1.0, audioReact);
    vec2 circle2 = _v9_circle(dnaUV, pixelSize, sinDna, cosDna, -1.0, audioReact);
    float lineGradient = _v9_linearstep(sinCutLineUp, sinCutLineDown, dnaUV.y);
    if (sin(lineIndex * _v9_SIN_DENSITY + _v9_TIME) > 0.0) lineGradient = 1.0 - lineGradient;
    lineGradient = mix(circle1.y, circle2.y, lineGradient);
    float lineAlpha = line * sinCutMaskUp * sinCutMaskDown;
    vec3 finalColor = bgFinal;
    if (circle1.y < circle2.y) {
        _v9_BLEND(finalColor, elementColor, circle1.y, circle1.x);
        _v9_BLEND(finalColor, elementColor, lineGradient, lineAlpha);
        _v9_BLEND(finalColor, elementColor, circle2.y, circle2.x);
    } else {
        _v9_BLEND(finalColor, elementColor, circle2.y, circle2.x);
        _v9_BLEND(finalColor, elementColor, lineGradient, lineAlpha);
        _v9_BLEND(finalColor, elementColor, circle1.y, circle1.x);
    }
    return finalColor;
}
"""


    image_glsl = f"""/* ============================================================================
   GLSL (The Last) MOD Player v1.61 (c) 2026 Orblivius
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
makeStr(printHdr)   _NUM _NUM _NUM _ _G _L _S _L _ _M _O _D _ _P _L _A _Y _E _R _ _V _1 _DOT _6 _1 _ _NUM _NUM _NUM _end
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
makeStr(printTracks) _T _R _X _COL _end
makeStr(printTrack)  _T _R _A _C _K _ _NUM _end

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
    float cell_u = fract(float(c) / 16.0);          // left edge of this glyph's atlas column
    vec2 uv = gp + vec2(cell_u, fract(float(15 - c/16) / 16.0));
    // Horizontal-only dilation: wider glyph, thinner stroke.
    // Clamp dilated U to [cell_u, cell_u+1/16] so we never bleed into an adjacent glyph's
    // atlas column, which would produce a faint vertical line artifact at cell boundaries.
    float texel  = 0.4/256.0;
    float u_lo   = cell_u + 0.5/256.0;              // half-texel inset from left boundary
    float u_hi   = cell_u + 1.0/16.0 - 0.5/256.0;  // half-texel inset from right boundary
    float r = textureGrad(iChannel0, uv, dx, dy).r;
    r = max(r, textureGrad(iChannel0, vec2(min(uv.x + texel, u_hi), uv.y), dx, dy).r);
    r = max(r, textureGrad(iChannel0, vec2(max(uv.x - texel, u_lo), uv.y), dx, dy).r);
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
//   2. Clamp at SONG_DURATION_S so the visualizer freezes on the last
//      frame when the song ends — no looping back to position 0.
//
// Why no infinite recursion: GLSL's preprocessor never expands a macro
// inside its own replacement text. The inner `iTime` is the real uniform.
//
// Must be at file scope (BEFORE viz_scene_block) so helper functions
// also get the substitution.
#define iTime clamp(iTime - INTRO_SILENCE_S, 0.0, SONG_DURATION_S)

{viz_scene_block}


// Unsigned distance from p to a quadratic Bézier (A=start, B=control,
// C=end). Closed-form cubic solve (Inigo Quilez,
// https://iquilezles.org/articles/distfunctions2d). Used by the
// oscilloscope to draw a smooth curve through the audio samples.
float sdBezier(vec2 p, vec2 A, vec2 B, vec2 C) {{
    vec2 a = B - A;
    vec2 b = A - 2.0*B + C;
    vec2 c = a * 2.0;
    vec2 d = A - p;
    float bb = dot(b,b);
    if (bb < 1e-6) {{
        // Degenerate (collinear control point) → straight segment.
        vec2 ba = C - A;
        float hh = clamp(dot(p-A,ba) / max(dot(ba,ba),1e-6), 0.0, 1.0);
        return length(p - A - ba*hh);
    }}
    float kk = 1.0 / bb;
    float kx = kk * dot(a,b);
    float ky = kk * (2.0*dot(a,a)+dot(d,b)) / 3.0;
    float kz = kk * dot(d,a);
    float res;
    float pq = ky - kx*kx;
    float pq3 = pq*pq*pq;
    float q = kx*(2.0*kx*kx - 3.0*ky) + kz;
    float h = q*q + 4.0*pq3;
    if (h >= 0.0) {{
        h = sqrt(h);
        vec2 x = (vec2(h,-h) - q) * 0.5;
        vec2 uv = sign(x) * pow(abs(x), vec2(1.0/3.0));
        float t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
        vec2 qd = d + (c + b*t)*t;
        res = dot(qd, qd);
    }} else {{
        float z = sqrt(-pq);
        float v = acos(q / (pq*z*2.0)) / 3.0;
        float m = cos(v), n = sin(v)*1.732050808;
        vec3 t = clamp(vec3(m+m, -n-m, n-m)*z - kx, 0.0, 1.0);
        vec2 q0 = d + (c + b*t.x)*t.x;
        vec2 q1 = d + (c + b*t.y)*t.y;
        res = min(dot(q0,q0), dot(q1,q1));
    }}
    return sqrt(res);
}}

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

    // getPosition() clamps time to SONG_DURATION_S — visualizer freezes at end
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
    // TRX: N — channel count, between BPM value and SPEED label (fits at 800px+)
    trk += BLUE   * printTracks(pUV(fp, ML+9.*CW, iy2, CH));
    trk += YELLOW * drawNum(NUM_CHANNELS, 2, ML+14.*CW, iy2, CW, CH, fp);
    trk += BLUE   * printSpd(pUV(fp, rx,  iy2, CH));
    trk += YELLOW * printSpdVal(pUV(fp, rx+7.*CW, iy2, CH));

    trk += BLUE * 0.55 * hline(fp, iy2+CH+4., 1., iResolution.x-1.);

    // ============ TRACKER ============
    float ty   = iy2+CH+10.;
    float TW   = 9.*CW+6.;    // 9 chars per cell + 6px gap
    float rNW  = 2.*CW;
    float txOff= ML+rNW+8. + _scrollX;  // scroll offset applied to all tracks
    const int HVR = 4;  // 9 visible rows → more room for oscilloscope

    // Track headers — dynamic loop over all channels, colored by tc%4,
    // scrolled with _scrollX so wide songs scroll horizontally.
    {{
        vec3 trackColors[4];
        trackColors[0]=TC0; trackColors[1]=TC1; trackColors[2]=TC2; trackColors[3]=TC3;
        for(int tc=0; tc<NUM_CHANNELS; tc++) {{
            vec3 tCol = trackColors[tc % 4];
            float tx = txOff + float(tc)*TW;
            // Only render headers inside the visible area (skip off-screen left/right)
            if(tx > ML+rNW+8.-TW && tx < iResolution.x - ML) {{
                int digits = (tc+1 >= 10) ? 2 : 1;
                float numCW = CW * 0.65;
                float rightmostX = 6.6*CW + float(digits-1)*numCW;
                float textW = rightmostX + numCW;
                float xCenter = tx + (TW - textW) * 0.5;
                trk += tCol * printTrack(pUV(fp, xCenter, ty, CH));
                trk += tCol * drawNum(tc+1, digits, xCenter + rightmostX, ty, numCW, CH, fp);
            }}
        }}
    }}

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

        // BLACK MASK: hide track content that scrolled under the row-number gutter
        if(fp.x < ML+rNW+8.) col = vec3(0.0);

        // Per-track note data
        float xInT=fp.x-txOff;
        if(xInT>=0.&&xInT<float(NUM_CHANNELS)*TW && fp.x >= ML+rNW+8.) {{
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

                // ── Smooth quadratic-Bézier waveform (ported mod_player.py) ─
                // The trace is a smooth curve threaded through the audio
                // sample points (classic midpoint scheme): each Bézier
                // segment runs between the midpoints of consecutive samples,
                // using the sample point itself as the control (bend) point,
                // so the curve passes through every sample with C1 continuity.
                // Replaces the straight min/max fill — much smoother trace.
                float _sampF = C.x / iResolution.x * float(maxIdx);
                int   _i0 = clamp(int(floor(_sampF)), 0, maxIdx);
                int   _iA = max(0, _i0 - 1);
                int   _iB = _i0;
                int   _iC = min(maxIdx, _i0 + 1);
                int   _iD = min(maxIdx, _i0 + 2);
                vec2 _PA = vec2(float(_iA) / float(maxIdx) * iResolution.x,
                    (clamp(texelFetch(iChannel1, ivec2(_iA, 0), 0).r * 3.0, -0.9, 0.9) * 0.40 + 0.5) * oh);
                vec2 _PB = vec2(float(_iB) / float(maxIdx) * iResolution.x,
                    (clamp(texelFetch(iChannel1, ivec2(_iB, 0), 0).r * 3.0, -0.9, 0.9) * 0.40 + 0.5) * oh);
                vec2 _PC = vec2(float(_iC) / float(maxIdx) * iResolution.x,
                    (clamp(texelFetch(iChannel1, ivec2(_iC, 0), 0).r * 3.0, -0.9, 0.9) * 0.40 + 0.5) * oh);
                vec2 _PD = vec2(float(_iD) / float(maxIdx) * iResolution.x,
                    (clamp(texelFetch(iChannel1, ivec2(_iD, 0), 0).r * 3.0, -0.9, 0.9) * 0.40 + 0.5) * oh);
                vec2 _M0 = 0.5 * (_PA + _PB);
                vec2 _M1 = 0.5 * (_PB + _PC);
                vec2 _M2 = 0.5 * (_PC + _PD);
                vec2 _pp = vec2(C.x, sy);
                // Perf: the trace is a thin curve in a tall strip — the
                // quadratic Béziers are bounded by the y-hull of _PA.._PD,
                // and the AA cutoff is _d>=1.75.  Reject pixels >2px outside
                // that hull BEFORE the 2 cubic-solve sdBezier calls (exact —
                // same final pixel, skips the expensive solves for the vast
                // majority of strip pixels far from the trace).
                float _loY = min(min(_PA.y,_PB.y), min(_PC.y,_PD.y));
                float _hiY = max(max(_PA.y,_PB.y), max(_PC.y,_PD.y));
                float _d = (sy < _loY - 2.0 || sy > _hiY + 2.0) ? 1e9
                         : min(sdBezier(_pp, _M0, _PB, _M1),
                               sdBezier(_pp, _M1, _PC, _M2));

                // Mid-line baseline first (drawn underneath the trace).
                col = mix(col, DIM*0.18, step(abs(sy - oh*0.5), 0.4));

                // AA: ~1px core + ~0.75px soft edge each side (smooth Bézier).
                float aa = 1.0 - smoothstep(0.25, 1.75, _d);
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
   GLSL (The Last) MOD Player v1.61 (c) 2026 Orblivius
   4+ Tracks support, S3M/MOD loader, 3D Surround, PhatBass, Comb Reverb, FAT, RVQ sample compression, configurable resampler
   Contact: subband@gmail.com or
            subband@protonmail.com
   GIT:     https://github.com/mewza/mod2glsl
   BUFFER A TAB
   Visualizer: {viz_name}
   Row 0      : FFT_N mixed audio samples      (getChannelOutput sum, per px)
   Row 1      : FFT_N/2 DFT magnitudes         (phasor-rotation DFT)
   Row 2      : UI state px0=specMode px1=prevMouse px2-4=audio bands px5=scrollOff px6=scrollAnchor px7=prevPressed px8=dragDead
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
#define iTime clamp(iTime - INTRO_SILENCE_S, 0.0, SONG_DURATION_S)

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
            // Toggle ONLY when the click STARTED inside the oscilloscope/spectrum
            // strip (fp.y > ~430px from top). iMouse.w = Y where button went down.
            // Prevents dragging the track scroll bar from toggling the spectrum mode.
            bool inOscArea = iMouse.z > 0.0 && (iResolution.y - iMouse.w) > 430.0;
            float currMouse = inOscArea ? 1.0 : 0.0;
            bool  newClick  = (currMouse > 0.5 && prevMouse < 0.5);
            O = vec4(newClick ? 1.0 - prevMode : prevMode, 0., 0., 1.);
        }} else if (px == 1) {{
            // Track the same filtered press state used by px=0 so edge detection works.
            bool inOscArea = iMouse.z > 0.0 && (iResolution.y - iMouse.w) > 430.0;
            float currMouse = inOscArea ? 1.0 : 0.0;
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
        }} else if (px == 5) {{
            // ── Horizontal track scroll offset (drag in tracker area) ──────────
            // Stored as a negative pixel offset applied to txOff in Image tab.
            // Drag 1:1 with mouse X while pressed inside tracker (Y 0.18-0.82).
            // Out-of-bounds press kills drag; release leaves offset where it was.
            float scrollOffset = texelFetch(iChannel0, ivec2(5, 2), 0).r;
            float scrollAnchor = texelFetch(iChannel0, ivec2(6, 2), 0).r;
            float prevPressed  = texelFetch(iChannel0, ivec2(7, 2), 0).r;
            float currPressed  = iMouse.z > 0.0 ? 1.0 : 0.0;
            float mouseY = iMouse.y / iResolution.y;
            bool inBounds = mouseY > 0.18 && mouseY < 0.82;
            // Track width in pixels (9 chars * CW=25 + 6 gap = 231px)
            float TW_PX = 9.0 * 25.0 + 6.0;
            float visWidth = iResolution.x - 68.0;
            float totalWidth = float(NUM_CHANNELS) * TW_PX;
            float MAX_SCROLL = max(0.0, ceil((totalWidth - visWidth) / TW_PX) * TW_PX);
            float dragDead = texelFetch(iChannel0, ivec2(8, 2), 0).r;
            if (currPressed > 0.5 && prevPressed < 0.5) {{
                dragDead = 0.0;
                scrollAnchor = scrollOffset;
            }}
            if (currPressed > 0.5 && !inBounds) dragDead = 1.0;
            if (currPressed > 0.5 && inBounds && dragDead < 0.5)
                scrollOffset = scrollAnchor + (iMouse.x - abs(iMouse.z));
            scrollOffset = clamp(scrollOffset, -MAX_SCROLL, 0.0);
            O = vec4(scrollOffset, 0., 0., 1.);
        }} else if (px == 6) {{
            // ── Scroll anchor — saved at the moment of a new click ────────────
            float scrollAnchor = texelFetch(iChannel0, ivec2(6, 2), 0).r;
            float scrollOffset = texelFetch(iChannel0, ivec2(5, 2), 0).r;
            float prevPressed  = texelFetch(iChannel0, ivec2(7, 2), 0).r;
            float currPressed  = iMouse.z > 0.0 ? 1.0 : 0.0;
            if (currPressed > 0.5 && prevPressed < 0.5) scrollAnchor = scrollOffset;
            O = vec4(scrollAnchor, 0., 0., 1.);
        }} else if (px == 7) {{
            // ── Previous mouse pressed state ──────────────────────────────────
            O = vec4(iMouse.z > 0.0 ? 1.0 : 0.0, 0., 0., 1.);
        }} else if (px == 8) {{
            // ── Drag-dead flag — kills drag when mouse leaves tracker area ─────
            float dragDead    = texelFetch(iChannel0, ivec2(8, 2), 0).r;
            float prevPressed = texelFetch(iChannel0, ivec2(7, 2), 0).r;
            float currPressed = iMouse.z > 0.0 ? 1.0 : 0.0;
            float mouseY      = iMouse.y / iResolution.y;
            bool inBounds     = mouseY > 0.18 && mouseY < 0.82;
            if (currPressed > 0.5 && prevPressed < 0.5) dragDead = 0.0;
            if (currPressed > 0.5 && !inBounds) dragDead = 1.0;
            if (currPressed < 0.5) dragDead = 0.0;
            O = vec4(dragDead, 0., 0., 1.);
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
    
    # ── 180s-cap dialog overlay (viz-agnostic) ──────────────────────────────
    # When ShaderToy's pre-rendered audio buffer is exhausted, the visualizer's
    # iTime saturates at AUDIO_BUFFER_S (it's clamped). Wrap mainImage so a
    # dimmed-backdrop + framed message box draws on top once that happens, so
    # it's obvious the platform (not our shader) ended playback. Done as a
    # rename+wrapper post-pass so it works regardless of which --viz is active.
    _dlg_anchor = 'void mainImage(out vec4 O, vec2 C)'
    if image_glsl.count(_dlg_anchor) == 1:
        image_glsl = image_glsl.replace(_dlg_anchor, 'void _vizMainImage(out vec4 O, vec2 C)', 1)
        image_glsl += r'''

// ── 180s-cap "limit reached" dialog (drawn on top of whatever viz is active) ──
// drawCh()/iResolution/AUDIO_BUFFER_S/INTRO_SILENCE_S are all defined above.
void _drawLimitDialog(inout vec3 col, vec2 fp) {
    // fp is in the same flipped space as the visualizer (y=0 at top).
    // iTime here is the audio-synced clamped time; it saturates at the cap.
    if (iTime < AUDIO_BUFFER_S - INTRO_SILENCE_S - 0.30) return;
    vec2 res = iResolution.xy;
    col = mix(col, vec3(0.0), 0.18);               // subtle dim — just a hint darker
    vec2 ctr = res * 0.5;
    vec2 hw  = vec2(min(res.x * 0.42, 320.0), min(res.y * 0.20, 92.0));
    vec2 d   = abs(fp - ctr);
    if (d.x > hw.x || d.y > hw.y) return;          // outside the dialog box
    if (d.x > hw.x - 2.5 || d.y > hw.y - 2.5) { col = vec3(0.82, 0.84, 0.94); return; } // border
    col = mix(col, vec3(0.05, 0.06, 0.09), 0.88);  // panel fill (slightly see-through)
    float CW = clamp(hw.x * 2.0 / 22.0, 6.0, 16.0), CH = CW * 1.55;
    // ASCII (uppercase). 32=space 46='.' 48-57=digits 65-90=A-Z
    int L1[23] = int[](77,65,88,32,76,73,77,73,84,32,49,56,48,46,48,32,83,69,67,79,78,68,83); // MAX LIMIT 180.0 SECONDS
    int L2[20] = int[](73,77,80,79,83,69,68,32,66,89,32,83,72,65,68,69,82,84,79,89);           // IMPOSED BY SHADERTOY
    int L3[16] = int[](87,69,66,83,73,84,69,32,82,69,65,67,72,69,68,32);                       // WEBSITE REACHED
    // Each line centred independently: lxN = ctr.x - (len/2) * CW
    float lx1 = ctr.x - 11.5 * CW;  // 23 chars
    float lx2 = ctr.x - 10.0 * CW;  // 20 chars
    float lx3 = ctr.x -  8.0 * CW;  // 16 chars
    // ty in flipped space (y=0 at top): subtract to go above centre, lines step downward (+)
    float t = 0.0, ty = ctr.y - CH * 1.1;
    for (int i = 0; i < 23; i++) if (L1[i] != 32) t = max(t, drawCh(L1[i], fp, lx1 + float(i)*CW, ty,         CW, CH));
    for (int i = 0; i < 20; i++) if (L2[i] != 32) t = max(t, drawCh(L2[i], fp, lx2 + float(i)*CW, ty + CH*1.6, CW, CH));
    for (int i = 0; i < 16; i++) if (L3[i] != 32) t = max(t, drawCh(L3[i], fp, lx3 + float(i)*CW, ty + CH*3.2, CW, CH));
    col = mix(col, vec3(1.0, 0.93, 0.65), clamp(t, 0.0, 1.0));   // amber text
}

void mainImage(out vec4 O, vec2 C) {
    _vizMainImage(O, C);
    vec3 _c = O.rgb;
    // Pass the same y-flipped coordinate the visualizer uses so drawCh renders glyphs upright.
    _drawLimitDialog(_c, vec2(C.x, iResolution.y - C.y));
    O = vec4(_c, 1.0);
}
'''
        print("   ✓ 180s-cap dialog overlay injected (Image tab; shows when buffer exhausted)")
    else:
        print(f"   ⚠ dialog overlay skipped (mainImage anchor count="
              f"{image_glsl.count(_dlg_anchor)}, expected 1)")

    with open(output_file.replace('.glsl', '_image.glsl'), 'w') as f:
        f.write(image_glsl)

    # When called with the _tmp_tabs stub (VQ path) these names are
    # intermediate and get renamed — printing them is misleading. The
    # real final tab names + channel setup are printed once at the end.
    if "_tmp_tabs" not in output_file:
        print(f"   📁 Created ShaderToy tabs:")
        print(f"      Common:   {output_file.replace('.glsl', '_common.glsl')}")
        print(f"      Sound:    {output_file.replace('.glsl', '_sound.glsl')}")
        print(f"      Image:    {output_file.replace('.glsl', '_image.glsl')}")
        print(f"      Buffer A: {bufA_file}")
    print()
    print(f"   🖱️  Click anywhere to toggle oscilloscope ↔ spectrum view")


# Embedded VQ encoder (base64-encoded to avoid string escaping issues).
# Contains vq_encoder_v2 + getchanneloutput.glsl bundled — no external files needed.
_VQ_ENCODER_B64 = (
    'IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiIKTU9EIOKGkiBTaGFkZXJUb3kgQ29tbW9uIHRh'
    'YiBlbmNvZGVyIHdpdGg6CiAgLSBQYXR0ZXJuIGNydW5jaDogYml0bWFwICsgZGljdGlvbmFy'
    'eSArIG5pYmJsZS1wYWNrZWQgcm93IHNlZWsKICAtIFNhbXBsZSBjcnVuY2g6IDMtYml0IGxp'
    'bmVhciBwYWNrZWQgKHVuaWZvcm0gbm9pc2UgZmxvb3Ig4oCUIHN0YWJsZSBhY3Jvc3MgYWxs'
    'IHBsYXliYWNrIHBpdGNoZXMpClRhcmdldDog4omkIDY0IEtCIHRvdGFsIHByaXZhdGUgY29u'
    'c3QgZGF0YSAoTWFjIEFOR0xFL01ldGFsIHNhZmUgem9uZSkKIiIiCmltcG9ydCBzdHJ1Y3Qs'
    'IHN5cywgb3MKCmNsYXNzIE1PREZpbGU6CiAgICBkZWYgX19pbml0X18oc2VsZiwgcGF0aCk6'
    'CiAgICAgICAgd2l0aCBvcGVuKHBhdGgsICdyYicpIGFzIGY6CiAgICAgICAgICAgIHNlbGYu'
    'ZGF0YSA9IGYucmVhZCgpCiAgICAgICAgc2VsZi5wYXJzZSgpCgogICAgZGVmIHBhcnNlKHNl'
    'bGYpOgogICAgICAgIGQgPSBzZWxmLmRhdGEKICAgICAgICBzZWxmLnRpdGxlID0gZFswOjIw'
    'XS5yc3RyaXAoYidceDAwJykuZGVjb2RlKCdsYXRpbjEnLCAncmVwbGFjZScpCiAgICAgICAg'
    'c2VsZi5zYW1wbGVzX2luZm8gPSBbXQogICAgICAgIGZvciBpIGluIHJhbmdlKDMxKToKICAg'
    'ICAgICAgICAgYmFzZSA9IDIwICsgaSozMAogICAgICAgICAgICBuYW1lID0gZFtiYXNlOmJh'
    'c2UrMjJdLnJzdHJpcChiJ1x4MDAnKS5kZWNvZGUoJ2xhdGluMScsICdyZXBsYWNlJykKICAg'
    'ICAgICAgICAgbGVuZ3RoX3cgICAgID0gc3RydWN0LnVucGFjaygnPkgnLCBkW2Jhc2UrMjI6'
    'YmFzZSsyNF0pWzBdCiAgICAgICAgICAgIGZpbmV0dW5lICAgICA9IGRbYmFzZSsyNF0gJiAw'
    'eDBGCiAgICAgICAgICAgIHZvbHVtZSAgICAgICA9IGRbYmFzZSsyNV0KICAgICAgICAgICAg'
    'bG9vcF9zdGFydF93ID0gc3RydWN0LnVucGFjaygnPkgnLCBkW2Jhc2UrMjY6YmFzZSsyOF0p'
    'WzBdCiAgICAgICAgICAgIGxvb3BfbGVuX3cgICA9IHN0cnVjdC51bnBhY2soJz5IJywgZFti'
    'YXNlKzI4OmJhc2UrMzBdKVswXQogICAgICAgICAgICBzZWxmLnNhbXBsZXNfaW5mby5hcHBl'
    'bmQoZGljdCgKICAgICAgICAgICAgICAgIG5hbWU9bmFtZSwgbGVuZ3RoPWxlbmd0aF93KjIs'
    'IGZpbmV0dW5lPWZpbmV0dW5lLAogICAgICAgICAgICAgICAgdm9sdW1lPXZvbHVtZSwgbG9v'
    'cF9zdGFydD1sb29wX3N0YXJ0X3cqMiwgbG9vcF9sZW49bG9vcF9sZW5fdyoyKSkKICAgICAg'
    'ICBzZWxmLnNvbmdfbGVuZ3RoID0gZFs5NTBdCiAgICAgICAgc2VsZi5wYXR0ZXJuX29yZGVy'
    'ID0gbGlzdChkWzk1Mjo5NTIrMTI4XSkKICAgICAgICBzZWxmLm1hZ2ljID0gZFsxMDgwOjEw'
    'ODRdCiAgICAgICAgIyBEZXRlY3QgY2hhbm5lbCBjb3VudCBmcm9tIHNpZ25hdHVyZQogICAg'
    'ICAgIHNpZyA9IHNlbGYubWFnaWMKICAgICAgICBpZiBzaWcgaW4gKGInTS5LLicsIGInTSFL'
    'IScsIGInTSZLIScsIGInTi5ULicsIGInRkxUNCcsIGInNENITicpOgogICAgICAgICAgICBz'
    'ZWxmLm51bV9jaGFubmVscyA9IDQKICAgICAgICBlbGlmIHNpZyA9PSBiJ0ZMVDgnIG9yIHNp'
    'ZyBpbiAoYidPQ1RBJywgYidDRDgxJywgYidPS1RBJyk6CiAgICAgICAgICAgIHNlbGYubnVt'
    'X2NoYW5uZWxzID0gOAogICAgICAgIGVsaWYgbGVuKHNpZykgPT0gNCBhbmQgc2lnWzE6NF0g'
    'PT0gYidDSE4nIGFuZCBzaWdbMDoxXS5pc2RpZ2l0KCk6CiAgICAgICAgICAgIHNlbGYubnVt'
    'X2NoYW5uZWxzID0gaW50KHNpZ1swOjFdKQogICAgICAgIGVsaWYgbGVuKHNpZykgPT0gNCBh'
    'bmQgc2lnWzI6NF0gPT0gYidDSCcgYW5kIHNpZ1swOjFdLmlzZGlnaXQoKSBhbmQgc2lnWzE6'
    'Ml0uaXNkaWdpdCgpOgogICAgICAgICAgICBzZWxmLm51bV9jaGFubmVscyA9IGludChzaWdb'
    'MDoyXSkKICAgICAgICBlbGlmIGxlbihzaWcpID09IDQgYW5kIHNpZ1s6M10gPT0gYidURFon'
    'IGFuZCBzaWdbMzo0XS5pc2RpZ2l0KCk6CiAgICAgICAgICAgIHNlbGYubnVtX2NoYW5uZWxz'
    'ID0gaW50KHNpZ1szOjRdKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHNlbGYubnVtX2No'
    'YW5uZWxzID0gNAogICAgICAgIHNlbGYubnVtX3BhdHRlcm5zID0gbWF4KHNlbGYucGF0dGVy'
    'bl9vcmRlcls6c2VsZi5zb25nX2xlbmd0aF0pICsgMQogICAgICAgICMgRWFjaCBwYXR0ZXJu'
    'IHJvdyA9IG51bV9jaGFubmVscyDDlyA0IGJ5dGVzOyA2NCByb3dzL3BhdHRlcm4KICAgICAg'
    'ICBwYXRfc2l6ZSA9IDY0ICogc2VsZi5udW1fY2hhbm5lbHMgKiA0CiAgICAgICAgc2VsZi5w'
    'YXR0ZXJucyA9IFtdCiAgICAgICAgb2ZmID0gMTA4NAogICAgICAgIGZvciBwIGluIHJhbmdl'
    'KHNlbGYubnVtX3BhdHRlcm5zKToKICAgICAgICAgICAgc2VsZi5wYXR0ZXJucy5hcHBlbmQo'
    'ZFtvZmY6b2ZmK3BhdF9zaXplXSkKICAgICAgICAgICAgb2ZmICs9IHBhdF9zaXplCiAgICAg'
    'ICAgIyBTYW1wbGVzIChyYXcgc2lnbmVkIDgtYml0IGJ5dGVzKQogICAgICAgIHNlbGYuc2Ft'
    'cGxlX2J5dGVzID0gW10KICAgICAgICBmb3IgcyBpbiBzZWxmLnNhbXBsZXNfaW5mbzoKICAg'
    'ICAgICAgICAgc2VsZi5zYW1wbGVfYnl0ZXMuYXBwZW5kKGRbb2ZmOm9mZitzWydsZW5ndGgn'
    'XV0pCiAgICAgICAgICAgIG9mZiArPSBzWydsZW5ndGgnXQoKIyDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZAKIyBQQVRURVJOIENSVU5D'
    'SDogYml0bWFwICsgZGljdCArIG5pYmJsZS1zZWVrCiMg4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQCgpFTVBUWV9OT1RFID0gYidceDAw'
    'XHgwMFx4MDBceDAwJwoKZGVmIGVuY29kZV9wYXR0ZXJucyhtb2QpOgogICAgIiIiUmV0dXJu'
    'cyBkaWN0IG9mIGFsbCBwYXR0ZXJuIGRhdGEgc3RydWN0dXJlcy4iIiIKICAgIENFTExfU0la'
    'RSAgPSA1IGlmIGdldGF0dHIobW9kLCAnaXNfaXQnLCBGYWxzZSkgZWxzZSA0CiAgICBFTVBU'
    'WV9OT1RFID0gYnl0ZXMoQ0VMTF9TSVpFKSAgIyBzaGFkb3dzIG1vZHVsZS1sZXZlbCBjb25z'
    'dGFudAogICAgIyBCdWlsZCBmbGF0IGxpc3Qgb2YgNC1ieXRlIG5vdGVzIGluIG9yZGVyOiBw'
    'YXQgMC4uTi0xLCByb3cgMC4uNjMsIGNoIDAuLjMuCiAgICAjIEZpcnN0IGFwcGx5IFByb1Ry'
    'YWNrZXIgcGFyYW0tbWVtb3J5IHJld3JpdGluZyBpbiBzb25nLXBvc2l0aW9uIHBsYXliYWNr'
    'CiAgICAjIG9yZGVyOiB3aGVuIGEgbm90ZSBoYXMgZWZmZWN0IDEvMi8zLzQvNS82L0EgYW5k'
    'IHBhcmFtPT0wLCBzdWJzdGl0dXRlIHRoZQogICAgIyBsYXN0IG5vbi16ZXJvIHBhcmFtIHNl'
    'ZW4gZm9yIHRoYXQgZWZmZWN0IG9uIHRoaXMgY2hhbm5lbC4gIFRoaXMgbWFrZXMKICAgICMg'
    'dG9uZS1wb3J0YSBydW5zIGxpa2UgIjMwMCAzMDAgMzAwIiBjb250aW51ZSB3aXRoIHRoZSBw'
    'cmV2aW91cyBzbGlkZSByYXRlCiAgICAjIOKAlCByZXF1aXJlZCBmb3IgbWFueSBNT0RzIChp'
    'bmNsLiBHU0xJTkdFUiBwYXR0ZXJuIDMpLgogICAgTkMgPSBtb2QubnVtX2NoYW5uZWxzCiAg'
    'ICByb3dfc3RyaWRlID0gTkMgKiBDRUxMX1NJWkUKCiAgICAjIFdhbGsgc29uZyBwb3NpdGlv'
    'bnMgdG8gZmluZCBwYXJhbS1tZW1vcnkgY2hhaW5zIHBlciBjaGFubmVsLgogICAgIyBFZmZl'
    'Y3QgZ3JvdXBzIHRoYXQgc2hhcmUgbWVtb3J5OgogICAgIyAgIDB4MSAocG9ydGEgdXApLCAw'
    'eDIgKHBvcnRhIGRvd24pLCAweDMgKHRvbmUgcG9ydGEpLCAweDUgKHRvbmUrdm9sKSwKICAg'
    'ICMgICAweDQgKHZpYnJhdG8pLCAweDYgKHZpYit2b2wpLCAweEEgKHZvbCBzbGlkZSkKICAg'
    'ICMgV2UgcmV3cml0ZSB0aGUgaW4tbWVtb3J5IHBhdHRlcm4gYnl0ZXMgKGEgY29weSkgc28g'
    'ZW5jb2Rpbmcgc2VlcyB0aGUKICAgICMgY29ycmVjdGVkIHBhcmFtcy4gIEJ1aWxkIGEgZnJl'
    'c2ggcGVyLXBhdHRlcm4gbm90ZSBsaXN0IHdpdGggcmV3cml0ZXMuCiAgICAjIFVzZSBtb2Qu'
    'cGF0dGVybl9vcmRlciAoZW5jb2RlciBNT0RGaWxlIGVxdWl2YWxlbnQgb2Ygc29uZ19wb3Np'
    'dGlvbnMpLgogICAgX3Nvbmdfb3JkZXIgPSBnZXRhdHRyKG1vZCwgJ3BhdHRlcm5fb3JkZXIn'
    'LCBOb25lKSBvciBnZXRhdHRyKG1vZCwgJ3NvbmdfcG9zaXRpb25zJywgW10pCiAgICBwYXRf'
    'Y29waWVzID0ge30KICAgIGxhc3RfcGFyYW0gPSBbe30gZm9yIF8gaW4gcmFuZ2UoTkMpXSAg'
    'IyBsYXN0X3BhcmFtW2NoXVtlZmZlY3RdID0gbGFzdCBub256ZXJvIHBhcmFtCiAgICByZXdy'
    'aXR0ZW5fbm90ZXNfY291bnQgPSAwCiAgICBmb3Igc3AgaW4gX3Nvbmdfb3JkZXJbOmdldGF0'
    'dHIobW9kLCAnc29uZ19sZW5ndGgnLCBsZW4oX3Nvbmdfb3JkZXIpKV06CiAgICAgICAgaWYg'
    'c3Agbm90IGluIHBhdF9jb3BpZXM6CiAgICAgICAgICAgIHBhdF9jb3BpZXNbc3BdID0gYnl0'
    'ZWFycmF5KG1vZC5wYXR0ZXJuc1tzcF0pCiAgICAgICAgIyBOb3RlOiBhIHBhdHRlcm4gbWF5'
    'IGFwcGVhciBhdCBtdWx0aXBsZSBzb25nIHBvc2l0aW9uczsgd2UgYXBwbHkKICAgICAgICAj'
    'IHJld3JpdGVzIGluIHBsYXliYWNrIG9yZGVyIHNvIG1lbW9yeSBzdGF0ZSBwcm9wYWdhdGVz'
    'IGFjcm9zcyB0aGVtLgogICAgICAgICMgUmV3cml0aW5nIGEgcGF0dGVybiB0aGF0IGlzIHJl'
    'dXNlZCBsYXRlciBtZWFucyB0aGUgc2Vjb25kIHZpc2l0CiAgICAgICAgIyB1c2VzIHRoZSBh'
    'bHJlYWR5LXJld3JpdHRlbiBwYXJhbXMsIHdoaWNoIGlzIGFjY2VwdGFibGUgc2luY2UgdGhl'
    'CiAgICAgICAgIyBsYXN0X3BhcmFtIHN0YXRlIGF0IHNlY29uZCB2aXNpdCB3b3VsZCBuYXR1'
    'cmFsbHkgYWxzbyBoYXZlIHRob3NlLgogICAgIyBSZXNldCBmb3IgYWN0dWFsIHJld3JpdGUg'
    'd2FsawogICAgbGFzdF9wYXJhbSA9IFt7fSBmb3IgXyBpbiByYW5nZShOQyldCiAgICB2aXNp'
    'dGVkX2tleXMgPSBzZXQoKQogICAgZm9yIHNwIGluIF9zb25nX29yZGVyWzpnZXRhdHRyKG1v'
    'ZCwgJ3NvbmdfbGVuZ3RoJywgbGVuKF9zb25nX29yZGVyKSldOgogICAgICAgIHBhdF9jb3B5'
    'ID0gcGF0X2NvcGllc1tzcF0KICAgICAgICBmb3Igcm93IGluIHJhbmdlKDY0KToKICAgICAg'
    'ICAgICAgZm9yIGNoIGluIHJhbmdlKE5DKToKICAgICAgICAgICAgICAgIGJhc2UgPSByb3cq'
    'cm93X3N0cmlkZSArIGNoKkNFTExfU0laRQogICAgICAgICAgICAgICAga2V5ID0gKHNwLCBy'
    'b3csIGNoKQogICAgICAgICAgICAgICAgaWYga2V5IGluIHZpc2l0ZWRfa2V5czogY29udGlu'
    'dWUgICMgYXZvaWQgZG91YmxlLXJld3JpdGUgb24gcGF0dGVybiByZXVzZQogICAgICAgICAg'
    'ICAgICAgdmlzaXRlZF9rZXlzLmFkZChrZXkpCiAgICAgICAgICAgICAgICAjIERlY29kZSBu'
    'b3RlOiBieXRlcyBbcGVyaW9kX2hpLCBwZXJpb2RfbG8sIHNhbXBsZV9sb3xlZmZlY3QsIHBh'
    'cmFtXQogICAgICAgICAgICAgICAgIyBNT0QgbGF5b3V0OiBieXRlMCA9IHNhbXBsZV9oaSg0'
    'KSB8IHBlcmlvZF9oaSg0KQogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICBieXRlMSA9'
    'IHBlcmlvZF9sbyg4KQogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICBieXRlMiA9IHNh'
    'bXBsZV9sbyg0KSB8IGVmZmVjdCg0KQogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICBi'
    'eXRlMyA9IHBhcmFtCiAgICAgICAgICAgICAgICBiMCwgYjEsIGIyLCBiMyA9IHBhdF9jb3B5'
    'W2Jhc2VdLCBwYXRfY29weVtiYXNlKzFdLCBwYXRfY29weVtiYXNlKzJdLCBwYXRfY29weVti'
    'YXNlKzNdCiAgICAgICAgICAgICAgICBlZmZlY3QgPSBiMiAmIDB4MEYKICAgICAgICAgICAg'
    'ICAgIHBhcmFtICA9IGIzCiAgICAgICAgICAgICAgICAjIFByb1RyYWNrZXIgcGFyYW0tbWVt'
    'b3J5IHJ1bGVzOgogICAgICAgICAgICAgICAgIwogICAgICAgICAgICAgICAgIyAgIDF4eCAo'
    'cG9ydGEgdXApICAgICAgICDigJQgcGFyYW09MCDihpIgdXNlIGxhc3QgMXh4CiAgICAgICAg'
    'ICAgICAgICAjICAgMnh4IChwb3J0YSBkb3duKSAgICAgIOKAlCBwYXJhbT0wIOKGkiB1c2Ug'
    'bGFzdCAyeHgKICAgICAgICAgICAgICAgICMgICAzeHggKHRvbmUgcG9ydGEpICAgICAg4oCU'
    'IHBhcmFtPTAg4oaSIHVzZSBsYXN0IDN4eAogICAgICAgICAgICAgICAgIwogICAgICAgICAg'
    'ICAgICAgIyAgIDR4eCAodmlicmF0bykgICAgICAgICDigJQgTklCQkxFIG1lbW9yeTogaGln'
    'aCBuaWI9MCBrZWVwcwogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICAgICAgICAgICAg'
    'ICAgICBwcmlvciBzcGVlZCwgbG93IG5pYj0wIGtlZXBzCiAgICAgICAgICAgICAgICAjICAg'
    'ICAgICAgICAgICAgICAgICAgICAgICAgIHByaW9yIGRlcHRoLiAgV2UgZG9uJ3QgcmV3cml0'
    'ZQogICAgICAgICAgICAgICAgIyAgICAgICAgICAgICAgICAgICAgICAgICAgICBoZXJlIOKA'
    'lCBHTFNML0hUTUwgZG8gbmliYmxlLWxldmVsCiAgICAgICAgICAgICAgICAjICAgICAgICAg'
    'ICAgICAgICAgICAgICAgICAgIGhhbmRsaW5nLiAgT25seSByZXdyaXRlIHBhcmFtPTAKICAg'
    'ICAgICAgICAgICAgICMgICAgICAgICAgICAgICAgICAgICAgICAgICAgKHdob2xlIGJ5dGUg'
    'emVybykg4oaSIHVzZSBsYXN0IDR4eC4KICAgICAgICAgICAgICAgICMgICA3eHggKHRyZW1v'
    'bG8pICAgICAgICAg4oCUIE5JQkJMRSBtZW1vcnkgbGlrZSA0eHguCiAgICAgICAgICAgICAg'
    'ICAjCiAgICAgICAgICAgICAgICAjICAgNXh4IChjb250aW51ZSB0b25lIHBvcnRhICsgdm9s'
    'IHNsaWRlKSDigJQgcGFyYW0gYnl0ZSBpcwogICAgICAgICAgICAgICAgIyAgICAgICAgICAg'
    'ICAgICAgICAgICAgICAgICBWT0wtU0xJREUgT05MWS4gIDUwMCA9IGNvbnRpbnVlCiAgICAg'
    'ICAgICAgICAgICAjICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNsaWRlIHdpdGggTk8g'
    'dm9sIGNoYW5nZTsgdmFsaWQKICAgICAgICAgICAgICAgICMgICAgICAgICAgICAgICAgICAg'
    'ICAgICAgICAgY29tbWFuZCwgZG8gTk9UIHJld3JpdGUuCiAgICAgICAgICAgICAgICAjICAg'
    'Nnh4IChjb250aW51ZSB2aWJyYXRvICsgdm9sIHNsaWRlKSDigJQgc2FtZSBhcyA1eHg7CiAg'
    'ICAgICAgICAgICAgICAjICAgICAgICAgICAgICAgICAgICAgICAgICAgIHBhcmFtIGJ5dGUg'
    'aXMgdm9sLXNsaWRlIG9ubHkuCiAgICAgICAgICAgICAgICAjICAgICAgICAgICAgICAgICAg'
    'ICAgICAgICAgIDYwMCA9IGNvbnRpbnVlIHZpYnJhdG8sIG5vIHZvbAogICAgICAgICAgICAg'
    'ICAgIyAgICAgICAgICAgICAgICAgICAgICAgICAgICBzbGlkZTsgdmFsaWQgY29tbWFuZC4K'
    'ICAgICAgICAgICAgICAgICMgICBBeHggKHZvbCBzbGlkZSkgICAgICAg4oCUIEEwMCA9IG5v'
    'LW9wIGluIFBUIChOT1QgbWVtb3J5KS4KICAgICAgICAgICAgICAgICMgICAgICAgICAgICAg'
    'ICAgICAgICAgICAgICAgRG8gTk9UIHJld3JpdGUuCiAgICAgICAgICAgICAgICBpZiBlZmZl'
    'Y3QgaW4gKDB4MSwgMHgyLCAweDMpOgogICAgICAgICAgICAgICAgICAgIGlmIHBhcmFtID09'
    'IDAgYW5kIGVmZmVjdCBpbiBsYXN0X3BhcmFtW2NoXToKICAgICAgICAgICAgICAgICAgICAg'
    'ICAgbmV3X3BhcmFtID0gbGFzdF9wYXJhbVtjaF1bZWZmZWN0XQogICAgICAgICAgICAgICAg'
    'ICAgICAgICBwYXRfY29weVtiYXNlKzNdID0gbmV3X3BhcmFtCiAgICAgICAgICAgICAgICAg'
    'ICAgICAgIHJld3JpdHRlbl9ub3Rlc19jb3VudCArPSAxCiAgICAgICAgICAgICAgICAgICAg'
    'ZWxpZiBwYXJhbSAhPSAwOgogICAgICAgICAgICAgICAgICAgICAgICBsYXN0X3BhcmFtW2No'
    'XVtlZmZlY3RdID0gcGFyYW0KICAgICAgICAgICAgICAgIGVsaWYgZWZmZWN0IGluICgweDQs'
    'IDB4Nyk6CiAgICAgICAgICAgICAgICAgICAgIyBXaG9sZS1ieXRlPTAg4oaSIHVzZSBsYXN0'
    'IHdob2xlLWJ5dGUgbWVtb3J5LgogICAgICAgICAgICAgICAgICAgICMgTm9uLXplcm86IGFs'
    'c28gc3RvcmUgYXMgbGFzdC1ieXRlIG1lbW9yeSAobmliYmxlLWxldmVsCiAgICAgICAgICAg'
    'ICAgICAgICAgIyBoYW5kbGluZyBpcyBkb25lIGJ5IEhUTUwvR0xTTCBkdXJpbmcgcGxheWJh'
    'Y2spLgogICAgICAgICAgICAgICAgICAgIGlmIHBhcmFtID09IDAgYW5kIGVmZmVjdCBpbiBs'
    'YXN0X3BhcmFtW2NoXToKICAgICAgICAgICAgICAgICAgICAgICAgbmV3X3BhcmFtID0gbGFz'
    'dF9wYXJhbVtjaF1bZWZmZWN0XQogICAgICAgICAgICAgICAgICAgICAgICBwYXRfY29weVti'
    'YXNlKzNdID0gbmV3X3BhcmFtCiAgICAgICAgICAgICAgICAgICAgICAgIHJld3JpdHRlbl9u'
    'b3Rlc19jb3VudCArPSAxCiAgICAgICAgICAgICAgICAgICAgZWxpZiBwYXJhbSAhPSAwOgog'
    'ICAgICAgICAgICAgICAgICAgICAgICBsYXN0X3BhcmFtW2NoXVtlZmZlY3RdID0gcGFyYW0K'
    'ICAgICAgICAgICAgICAgICMgNS82L0E6IG5vIHBhcmFtLW1lbW9yeSByZXdyaXRpbmcgKHRo'
    'ZWlyIHBhcmFtPTAgaXMgbWVhbmluZ2Z1bCkKCiAgICBub3RlcyA9IFtdCiAgICBmb3IgcGF0'
    'IGluIHJhbmdlKG1vZC5udW1fcGF0dGVybnMpOgogICAgICAgIGlmIHBhdCBpbiBwYXRfY29w'
    'aWVzOgogICAgICAgICAgICBwZGF0YSA9IHBhdF9jb3BpZXNbcGF0XQogICAgICAgIGVsc2U6'
    'CiAgICAgICAgICAgIHBkYXRhID0gbW9kLnBhdHRlcm5zW3BhdF0KICAgICAgICBmb3Igcm93'
    'IGluIHJhbmdlKDY0KToKICAgICAgICAgICAgZm9yIGNoIGluIHJhbmdlKE5DKToKICAgICAg'
    'ICAgICAgICAgIGJhc2UgPSByb3cqcm93X3N0cmlkZSArIGNoKkNFTExfU0laRQogICAgICAg'
    'ICAgICAgICAgbm90ZXMuYXBwZW5kKGJ5dGVzKHBkYXRhW2Jhc2U6YmFzZStDRUxMX1NJWkVd'
    'KSkKICAgIGlmIHJld3JpdHRlbl9ub3Rlc19jb3VudCA+IDA6CiAgICAgICAgcHJpbnQoZiIg'
    'ICDimpnvuI8gIFBhcmFtLW1lbW9yeToge3Jld3JpdHRlbl9ub3Rlc19jb3VudH0gcGFyYW09'
    'MCBlZmZlY3RzIHJld3JpdHRlbiB3aXRoIHByZXZpb3VzIHZhbHVlcyIpCiAgICB0b3RhbF9u'
    'b3RlcyA9IGxlbihub3RlcykKICAgIG51bV9yb3dzICAgID0gbW9kLm51bV9wYXR0ZXJucyAq'
    'IDY0CgogICAgIyBVbmlxdWUgbm9uLWVtcHR5IG5vdGVzIOKGkiBkaWN0aW9uYXJ5CiAgICB1'
    'bmlxID0gc29ydGVkKHNldChuIGZvciBuIGluIG5vdGVzIGlmIG4gIT0gRU1QVFlfTk9URSkp'
    'CiAgICBpZHhfYnl0ZXMgPSAxIGlmIGxlbih1bmlxKSA8PSAyNTYgZWxzZSAyCiAgICBhc3Nl'
    'cnQgbGVuKHVuaXEpIDw9IDY1NTM2LCBmInRvbyBtYW55IHVuaXF1ZSBub3Rlczoge2xlbih1'
    'bmlxKX0iCiAgICBub3RlX3RvX2lkeCA9IHtuOmkgZm9yIGksbiBpbiBlbnVtZXJhdGUodW5p'
    'cSl9CgogICAgIyBCaXRtYXAgKDEgYml0IHBlciBub3RlLCBMU0ItZmlyc3Qgd2l0aGluIGVh'
    'Y2ggYnl0ZSkKICAgIGJpdG1hcCA9IGJ5dGVhcnJheSgodG90YWxfbm90ZXMgKyA3KSAvLyA4'
    'KQogICAgZm9yIGksIG4gaW4gZW51bWVyYXRlKG5vdGVzKToKICAgICAgICBpZiBuICE9IEVN'
    'UFRZX05PVEU6CiAgICAgICAgICAgIGJpdG1hcFtpID4+IDNdIHw9IDEgPDwgKGkgJiA3KQoK'
    'ICAgICMgSW5kZXggc3RyZWFtICgxIG9yIDIgYnl0ZXMgcGVyIG5vbi1lbXB0eSBub3RlLCBs'
    'aXR0bGUtZW5kaWFuIGlmIDJCKQogICAgaWR4X3N0cmVhbSA9IGJ5dGVhcnJheSgpCiAgICBm'
    'b3IgbiBpbiBub3RlczoKICAgICAgICBpZiBuICE9IEVNUFRZX05PVEU6CiAgICAgICAgICAg'
    'IGkgPSBub3RlX3RvX2lkeFtuXQogICAgICAgICAgICBpZiBpZHhfYnl0ZXMgPT0gMToKICAg'
    'ICAgICAgICAgICAgIGlkeF9zdHJlYW0uYXBwZW5kKGkpCiAgICAgICAgICAgIGVsc2U6CiAg'
    'ICAgICAgICAgICAgICBpZHhfc3RyZWFtLmFwcGVuZChpICYgMHhGRikKICAgICAgICAgICAg'
    'ICAgIGlkeF9zdHJlYW0uYXBwZW5kKChpID4+IDgpICYgMHhGRikKCiAgICAjIFBlci1yb3cg'
    'Y291bnQ6IGNvdW50IG9mIG5vbi1lbXB0eSBub3RlcyBJTiB0aGlzIHJvdyAoMC4uNCkuCiAg'
    'ICBwZXJfcm93X2NvdW50ID0gW10KICAgIGZvciByb3cgaW4gcmFuZ2UobnVtX3Jvd3MpOgog'
    'ICAgICAgIGNvdW50ID0gc3VtKDEgZm9yIGNoIGluIHJhbmdlKE5DKSBpZiBub3Rlc1tyb3cq'
    'TkMgKyBjaF0gIT0gRU1QVFlfTk9URSkKICAgICAgICBwZXJfcm93X2NvdW50LmFwcGVuZChj'
    'b3VudCkKCiAgICAjIFByZWZpeCBzdW06IHByZWZpeFtyb3ddID0gbm9uLWVtcHR5IGNvdW50'
    'IGluIHJvd3MgWzAsIHJvdykgPSByYW5rIGF0IHN0YXJ0IG9mIHJvdy4KICAgICMgU3RvcmVk'
    'IGFzIDE2LWJpdCBMRSB3b3JkcyBzbyBkZWNvZGVyIGlzIE8oMSkuCiAgICAjIFJhbmdlOiAw'
    'IHRvIH50b3RhbF9ub25fZW1wdHkgKOKJpCA1ODg4IGZvciAyMy1wYXQgTU9EKSDihpIgZml0'
    'cyBlYXNpbHkgaW4gMTYgYml0cy4KICAgIHByZWZpeCA9IFswXSAqIG51bV9yb3dzCiAgICBy'
    'dW5uaW5nID0gMAogICAgZm9yIHJvdyBpbiByYW5nZShudW1fcm93cyk6CiAgICAgICAgcHJl'
    'Zml4W3Jvd10gPSBydW5uaW5nCiAgICAgICAgcnVubmluZyArPSBwZXJfcm93X2NvdW50W3Jv'
    'd10KCiAgICByb3dfc2Vla19ieXRlcyA9IGJ5dGVhcnJheSgpCiAgICBmb3IgdiBpbiBwcmVm'
    'aXg6CiAgICAgICAgYXNzZXJ0IDAgPD0gdiA8IDY1NTM2LCBmInByZWZpeCB7dn0gb3ZlcmZs'
    'b3dzIDE2IGJpdHMiCiAgICAgICAgcm93X3NlZWtfYnl0ZXMuYXBwZW5kKHYgJiAweEZGKQog'
    'ICAgICAgIHJvd19zZWVrX2J5dGVzLmFwcGVuZCgodiA+PiA4KSAmIDB4RkYpCgogICAgcmV0'
    'dXJuIGRpY3QoCiAgICAgICAgdG90YWxfbm90ZXM9dG90YWxfbm90ZXMsIG51bV9yb3dzPW51'
    'bV9yb3dzLAogICAgICAgIHVuaXE9dW5pcSwgbm90ZV90b19pZHg9bm90ZV90b19pZHgsIGlk'
    'eF9ieXRlcz1pZHhfYnl0ZXMsCiAgICAgICAgYml0bWFwPWJpdG1hcCwgaWR4X3N0cmVhbT1p'
    'ZHhfc3RyZWFtLAogICAgICAgIHJvd19zZWVrX2J5dGVzPXJvd19zZWVrX2J5dGVzLCBwcmVm'
    'aXg9cHJlZml4LAogICAgICAgIGNlbGxfc2l6ZT1DRUxMX1NJWkUsCiAgICApCgojIOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkAojIDMt'
    'QklUIExJTkVBUiBTQU1QTEUgQ1JVTkNICiMg4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQCgpkZWYgZW5jb2RlX3NhbXBsZXNfcGFja2Vk'
    'KG1vZCwgYml0cz0zKToKICAgICIiIkNvbmNhdGVuYXRlIGFsbCBzYW1wbGVzLCBlbmNvZGUg'
    'ZWFjaCB0byBgYml0c2AgYml0cyAocm91bmRlZCkuCiAgICBTdXBwb3J0cyAzLWJpdCBhbmQg'
    'NC1iaXQgbGluZWFyIHF1YW50aXphdGlvbi4KICAgICAgMy1iaXQ6IGNvZGUgMC4uNywgbGV2'
    'ZWxzIChjb2RlKjMyIC0gMTEyKSwgc3RlcCAzMi8yNTYgPSAxMi41JQogICAgICA0LWJpdDog'
    'Y29kZSAwLi4xNSwgbGV2ZWxzIChjb2RlKjE2IC0gMTIwKSwgc3RlcCAxNi8yNTYgPSA2LjI1'
    'JSAoKzYgZEIgU05SKQogICAgUmV0dXJucyBwYWNrZWQgYnl0ZXMgKyBwZXItc2FtcGxlIHN0'
    'YXJ0IGluZGljZXMgKGxvZ2ljYWwgc2FtcGxlIHVuaXRzKS4iIiIKICAgIGlmIGJpdHMgbm90'
    'IGluICgzLCA0KToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYiYml0cyBtdXN0IGJlIDMg'
    'b3IgNCwgZ290IHtiaXRzfSIpCgogICAgY29uY2F0X3NpZ25lZCA9IFtdCiAgICBzdGFydHMg'
    'PSBbXQogICAgZm9yIHMgaW4gbW9kLnNhbXBsZV9ieXRlczoKICAgICAgICBzdGFydHMuYXBw'
    'ZW5kKGxlbihjb25jYXRfc2lnbmVkKSkKICAgICAgICBmb3IgYiBpbiBzOgogICAgICAgICAg'
    'ICBjb25jYXRfc2lnbmVkLmFwcGVuZChiIC0gMjU2IGlmIGIgPj0gMTI4IGVsc2UgYikKICAg'
    'ICAgICBjb25jYXRfc2lnbmVkLmV4dGVuZChbMF0gKiAxNikKCiAgICB0b3RhbF9zYW1wbGVz'
    'ID0gbGVuKGNvbmNhdF9zaWduZWQpCiAgICBjb2RlcyA9IGJ5dGVhcnJheSgpCiAgICBtYXhf'
    'Y29kZSA9ICgxIDw8IGJpdHMpIC0gMQogICAgc2hpZnQgPSA4IC0gYml0cwogICAgZm9yIHN2'
    'IGluIGNvbmNhdF9zaWduZWQ6CiAgICAgICAgdW5zaWduZWRfb2Zmc2V0ID0gc3YgKyAxMjgg'
    'ICMgWzAsIDI1NV0KICAgICAgICBjb2RlID0gdW5zaWduZWRfb2Zmc2V0ID4+IHNoaWZ0CiAg'
    'ICAgICAgaWYgY29kZSA+IG1heF9jb2RlOiBjb2RlID0gbWF4X2NvZGUKICAgICAgICBjb2Rl'
    'cy5hcHBlbmQoY29kZSkKCiAgICB0b3RhbF9iaXRzID0gdG90YWxfc2FtcGxlcyAqIGJpdHMK'
    'ICAgIHRvdGFsX2J5dGVzID0gKHRvdGFsX2JpdHMgKyA3KSAvLyA4CiAgICBwYWNrZWQgPSBi'
    'eXRlYXJyYXkodG90YWxfYnl0ZXMpCgogICAgaWYgYml0cyA9PSA0OgogICAgICAgICMgTmli'
    'YmxlIHBhY2tpbmc6IDIgY29kZXMgcGVyIGJ5dGUsIGxvdyBuaWJibGUgZmlyc3QKICAgICAg'
    'ICBmb3IgaSwgYyBpbiBlbnVtZXJhdGUoY29kZXMpOgogICAgICAgICAgICBieXRlX3BvcyA9'
    'IGkgPj4gMQogICAgICAgICAgICBpZiBpICYgMToKICAgICAgICAgICAgICAgIHBhY2tlZFti'
    'eXRlX3Bvc10gfD0gKGMgJiAweEYpIDw8IDQKICAgICAgICAgICAgZWxzZToKICAgICAgICAg'
    'ICAgICAgIHBhY2tlZFtieXRlX3Bvc10gfD0gYyAmIDB4RgogICAgZWxzZTogICMgYml0cyA9'
    'PSAzCiAgICAgICAgZm9yIGksIGMgaW4gZW51bWVyYXRlKGNvZGVzKToKICAgICAgICAgICAg'
    'Yml0X3BvcyAgID0gaSAqIDMKICAgICAgICAgICAgYnl0ZV9wb3MgID0gYml0X3BvcyA+PiAz'
    'CiAgICAgICAgICAgIGJpdF9zaGlmdCA9IGJpdF9wb3MgJiA3CiAgICAgICAgICAgIHZhbCA9'
    'IChjICYgNykgPDwgYml0X3NoaWZ0CiAgICAgICAgICAgIHBhY2tlZFtieXRlX3Bvc10gfD0g'
    'dmFsICYgMHhGRgogICAgICAgICAgICBpZiBiaXRfc2hpZnQgPiA1IGFuZCBieXRlX3BvcyAr'
    'IDEgPCB0b3RhbF9ieXRlczoKICAgICAgICAgICAgICAgIHBhY2tlZFtieXRlX3BvcyArIDFd'
    'IHw9ICh2YWwgPj4gOCkgJiAweEZGCgogICAgcmV0dXJuIHBhY2tlZCwgc3RhcnRzLCB0b3Rh'
    'bF9zYW1wbGVzCgojIEJhY2t3YXJkLWNvbXBhdCBhbGlhcwpkZWYgZW5jb2RlX3NhbXBsZXNf'
    'M2JpdChtb2QpOgogICAgcmV0dXJuIGVuY29kZV9zYW1wbGVzX3BhY2tlZChtb2QsIGJpdHM9'
    'MykKCgpkZWYgY29tcHV0ZV9yb3dfc3BlZWRfdGFibGUobW9kKToKICAgICIiIlNpbXVsYXRl'
    'IHRoZSBzb25nIHRvIGZpbmQgcGVyLXJvdyBTUEVFRCAoaG9ub3VyaW5nIEZ4eC9EeHgvQnh4'
    'IGVmZmVjdHMpLgogICAgUmV0dXJucyByb3dTcGVlZFtudW1fc29uZ19yb3dzXSBhbmQgcm93'
    'U3RhcnRUaWNrW251bV9zb25nX3Jvd3MrMV0uCiAgICBDb3JyZWN0bHkgaGFuZGxlcyBEeHgg'
    'KHBhdHRlcm4gYnJlYWspIGFuZCBCeHggKHBvc2l0aW9uIGp1bXApIHdoaWNoCiAgICBzaG9y'
    'dGVuIGEgcGF0dGVybidzIGVmZmVjdGl2ZSByb3cgY291bnQuIiIiCiAgICBzcGVlZCA9IDYg'
    'ICMgUHJvVHJhY2tlciBkZWZhdWx0CiAgICBicG0gICA9IDEyNQogICAgcm93U3BlZWQgPSBb'
    'XQogICAgYnBtX2NoYW5nZXMgPSBGYWxzZQogICAgZm9yIHBvcyBpbiByYW5nZShtb2Quc29u'
    'Z19sZW5ndGgpOgogICAgICAgIHBhdF9pZHggPSBtb2QucGF0dGVybl9vcmRlcltwb3NdCiAg'
    'ICAgICAgcGRhdGEgPSBtb2QucGF0dGVybnNbcGF0X2lkeF0KICAgICAgICBicm9rZSA9IEZh'
    'bHNlCiAgICAgICAgZm9yIHJvdyBpbiByYW5nZSg2NCk6CiAgICAgICAgICAgICMgU2NhbiBh'
    'bGwgNCBjaGFubmVscyBmb3IgRnh4IC8gRHh4IC8gQnh4IG9uIHRoaXMgcm93CiAgICAgICAg'
    'ICAgIGZvciBjaCBpbiByYW5nZShtb2QubnVtX2NoYW5uZWxzKToKICAgICAgICAgICAgICAg'
    'IGJhc2UgPSByb3cgKiBtb2QubnVtX2NoYW5uZWxzICogNCArIGNoICogNAogICAgICAgICAg'
    'ICAgICAgYjAsIGIxLCBiMiwgYjMgPSBwZGF0YVtiYXNlOmJhc2UrNF0KICAgICAgICAgICAg'
    'ICAgIGVmZmVjdCA9IGIyICYgMHgwRgogICAgICAgICAgICAgICAgcGFyYW0gID0gYjMKICAg'
    'ICAgICAgICAgICAgIGlmIGVmZmVjdCA9PSAweEYgYW5kIHBhcmFtID4gMDoKICAgICAgICAg'
    'ICAgICAgICAgICBpZiBwYXJhbSA8IDB4MjA6CiAgICAgICAgICAgICAgICAgICAgICAgIHNw'
    'ZWVkID0gcGFyYW0KICAgICAgICAgICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAg'
    'ICAgICAgICBpZiBicG0gIT0gcGFyYW06CiAgICAgICAgICAgICAgICAgICAgICAgICAgICBi'
    'cG1fY2hhbmdlcyA9IFRydWUKICAgICAgICAgICAgICAgICAgICAgICAgYnBtID0gcGFyYW0K'
    'ICAgICAgICAgICAgICAgIGVsaWYgZWZmZWN0ID09IDB4RCBvciBlZmZlY3QgPT0gMHhCOgog'
    'ICAgICAgICAgICAgICAgICAgIGJyb2tlID0gVHJ1ZSAgICMgcGF0dGVybiBicmVhayAvIHBv'
    'c2l0aW9uIGp1bXAKICAgICAgICAgICAgcm93U3BlZWQuYXBwZW5kKHNwZWVkKQogICAgICAg'
    'ICAgICBpZiBicm9rZToKICAgICAgICAgICAgICAgIGJyZWFrICAgIyBzdG9wIGFkZGluZyBy'
    'b3dzIGZvciB0aGlzIHNvbmcgcG9zaXRpb24KICAgIHJvd1N0YXJ0VGljayA9IFswXQogICAg'
    'Zm9yIHMgaW4gcm93U3BlZWQ6CiAgICAgICAgcm93U3RhcnRUaWNrLmFwcGVuZChyb3dTdGFy'
    'dFRpY2tbLTFdICsgcykKICAgIHJldHVybiByb3dTcGVlZCwgcm93U3RhcnRUaWNrLCBicG1f'
    'Y2hhbmdlcwoKCmRlZiBlbmNvZGVfc2FtcGxlc192cTJkKG1vZCwgSz0yNTYsIHdlaWdodGVk'
    'PVRydWUsIGRvd25zYW1wbGU9MiwgYml0cmF0ZT0nbWVkJywgdmVjX2RpbT0yLCBub19ydnEy'
    'PUZhbHNlKToKICAgICIiIjItc3RhZ2UgUmVzaWR1YWwgVlEgd2l0aCBGRlQtZ3VpZGVkIHBl'
    'ci1zYW1wbGUgZGVjaW1hdGlvbi4KICAgIFBlci1zYW1wbGUgRFMgdmlhIEZGVCBiYW5kd2lk'
    'dGggYW5hbHlzaXMg4oCUIERTPTEgZm9yIGZ1bGwtYmFuZHdpZHRoIHNhbXBsZXMKICAgIChw'
    'cmVzZXJ2ZXMgYWxsIEhGKSwgb25seSBkb3duc2FtcGxlIGlmIGNvbnRlbnQgaXMgZ2VudWlu'
    'ZWx5IGxvdy1iYW5kd2lkdGguCiAgICBSYXcgc3RyaWRlIGRlY2ltYXRpb24gKG5vIExQRiku'
    'IGJ3RmFjdG9yIHBlciBzYW1wbGUgPSBhY3R1YWwgRFMgdXNlZC4KICAgICIiIgogICAgaW1w'
    'b3J0IG51bXB5IGFzIG5wCiAgICBmcm9tIHNrbGVhcm4uY2x1c3RlciBpbXBvcnQgTWluaUJh'
    'dGNoS01lYW5zCgogICAgIyBCaXRyYXRlIOKGkiBjb2RlYm9vayBzaXplIChtcDMtc3R5bGUg'
    'cXVhbGl0eSBrbm9iKQogICAgX2JpdHJhdGVfdGFibGUgPSB7CiAgICAgICAgJ2xvJzogICAg'
    'KDEyOCwgIDY0KSwgICAjIDEzIGJpdHMvcGFpciwgc21hbGxlc3QrZ3JhaW55CiAgICAgICAg'
    'J21lZCc6ICAgKDI1NiwgMTI4KSwgICAjIDE1IGJpdHMvcGFpciwgYmFsYW5jZWQKICAgICAg'
    'ICAnaGknOiAgICAoNTEyLCAyNTYpLCAgICMgMTcgYml0cy9wYWlyLCBkZWZhdWx0CiAgICAg'
    'ICAgJ3VsdHJhJzooMTAyNCwgNTEyKSwgICAjIDE5IGJpdHMvcGFpciwgbmVhci10cmFuc3Bh'
    'cmVudAogICAgfQogICAgSzEsIEsyID0gX2JpdHJhdGVfdGFibGUuZ2V0KGJpdHJhdGUsIF9i'
    'aXRyYXRlX3RhYmxlWydoaSddKQogICAgaWYgbm9fcnZxMjoKICAgICAgICBLMiA9IDAgICMg'
    'c2lnbmFsOiBza2lwIHN0YWdlIDIKICAgIEJJVFMxID0gaW50KG5wLmNlaWwobnAubG9nMihL'
    'MSkpKQogICAgQklUUzIgPSBpbnQobnAuY2VpbChucC5sb2cyKEsyKSkpIGlmIEsyID4gMCBl'
    'bHNlIDAKICAgIEJJVFNfVE9UQUwgPSBCSVRTMSArIEJJVFMyICAjIGlmIEsyPT0wLCBCSVRT'
    'Mj09MCwgc28ganVzdCBCSVRTMQoKICAgIGRlZiBoZl9yYXRpbyhyYXdfYnl0ZXMsIGxlbmd0'
    'aCwgbnlxdWlzdF9oej0yMjA1MCk6CiAgICAgICAgIiIiRnJhY3Rpb24gb2YgZW5lcmd5IGFi'
    'b3ZlIDhrSHog4oCUIGhpZ2ggPSBwZXJjdXNzaW9uL2N5bWJhbC4iIiIKICAgICAgICBpZiBs'
    'ZW5ndGggPCAzMjogcmV0dXJuIDAuMAogICAgICAgIGRhdGEgPSBucC5mcm9tYnVmZmVyKHJh'
    'd19ieXRlc1s6bGVuZ3RoXSwgZHR5cGU9bnAuaW50OCkuYXN0eXBlKG5wLmZsb2F0MzIpCiAg'
    'ICAgICAgZmZ0ICA9IG5wLmFicyhucC5mZnQucmZmdChkYXRhWzptaW4obGVuZ3RoLCA0MDk2'
    'KV0pKQogICAgICAgIGUgICAgPSBmbG9hdChucC5zdW0oZmZ0KioyKSkgKyAxZS0xMAogICAg'
    'ICAgIGN1dCAgPSBtYXgoMSwgaW50KGxlbihmZnQpICogODAwMCAvIG55cXVpc3RfaHopKQog'
    'ICAgICAgIHJldHVybiBmbG9hdChucC5zdW0oZmZ0W2N1dDpdKioyKSkgLyBlCgogICAgY29u'
    'Y2F0X2RzID0gW10KICAgIHN0YXJ0cyAgICA9IFtdCiAgICBzYW1wbGVfZHMgPSBbXSAgIyBw'
    'ZXItc2FtcGxlIGFjdHVhbCBEUyB1c2VkCiAgICB0b3RhbF9zYW1wbGVzX2Z1bGwgPSAwCgog'
    'ICAgZm9yIHMsIHJhd19ieXRlcyBpbiB6aXAobW9kLnNhbXBsZXNfaW5mbywgbW9kLnNhbXBs'
    'ZV9ieXRlcyk6CiAgICAgICAgc3RhcnRzLmFwcGVuZChsZW4oY29uY2F0X2RzKSkKICAgICAg'
    'ICBpZiBzWydsZW5ndGgnXSA+IDA6CiAgICAgICAgICAgIHJhdyA9IG5wLmZyb21idWZmZXIo'
    'cmF3X2J5dGVzLCBkdHlwZT1ucC5pbnQ4KS5hc3R5cGUobnAuZmxvYXQzMikgLyAxMjguMAog'
    'ICAgICAgICAgICB0b3RhbF9zYW1wbGVzX2Z1bGwgKz0gbGVuKHJhdykKICAgICAgICAgICAg'
    'IyBQZXItc2FtcGxlIERTIHZpYSBGRlQgYmFuZHdpZHRoIGFuYWx5c2lzIChtaXJyb3JzIEhU'
    'TUwgcGxheWVyJ3MKICAgICAgICAgICAgIyBid19jb21wcmVzc19zYW1wbGUpLiAgLS1kb3du'
    'c2FtcGxlIGlzIGEgQ0FQLCBub3QgYSBmbG9vcjogZnVsbC0KICAgICAgICAgICAgIyBiYW5k'
    'd2lkdGggc2FtcGxlcyAoZ3VpdGFycywgdm9jYWxzKSBzdGF5IGF0IERTPTEsIG5hcnJvdy1i'
    'YW5kCiAgICAgICAgICAgICMgc2FtcGxlcyAobG93IGJhc3MsIG11dGVkIGluc3RydW1lbnRz'
    'KSBkcm9wIHRvIERTPTIvNC84LgogICAgICAgICAgICAjCiAgICAgICAgICAgICMgV2l0aG91'
    'dCB0aGlzIHRoZSBHTFNMIHdhcyBmb3JjZS1kZWNpbWF0aW5nIGV2ZXJ5IHNhbXBsZSB0bwog'
    'ICAgICAgICAgICAjIGRvd25zYW1wbGUsIHByb2R1Y2luZyB0aGUgIjgga0h6IGxvLWZpIiBh'
    'cnRpZmFjdHMgdGhlIEhUTUwKICAgICAgICAgICAgIyBwbGF5ZXIgYXZvaWRlZC4KICAgICAg'
    'ICAgICAgc3IgPSA0NDEwMC4wCiAgICAgICAgICAgIG5fZmZ0ID0gbWluKGxlbihyYXcpLCA4'
    'MTkyKQogICAgICAgICAgICBmZnRfbWFnID0gbnAuYWJzKG5wLmZmdC5yZmZ0KHJhd1s6bl9m'
    'ZnRdICogbnAuaGFubmluZyhuX2ZmdCkpKQogICAgICAgICAgICBmcmVxcyAgID0gbnAuZmZ0'
    'LnJmZnRmcmVxKG5fZmZ0LCAxLjAgLyBzcilbOmxlbihmZnRfbWFnKV0KICAgICAgICAgICAg'
    'cGVhayAgICA9IGZsb2F0KG5wLm1heChmZnRfbWFnKSkgKyAxZS0xMgogICAgICAgICAgICBz'
    'aWdfYmlucyA9IG5wLndoZXJlKGZmdF9tYWcgPiBwZWFrICogMC4wMDUpWzBdCiAgICAgICAg'
    'ICAgIG1heF9mcmVxID0gZmxvYXQoZnJlcXNbc2lnX2JpbnNbLTFdXSkgaWYgbGVuKHNpZ19i'
    'aW5zKSBlbHNlIDIyMDUwLjAKICAgICAgICAgICAgIyBVc2VyJ3MgLS1kb3duc2FtcGxlIGlz'
    'IHRoZSBGTE9PUiAoYWx3YXlzIGF0IGxlYXN0IHRoaXMgbXVjaCkuCiAgICAgICAgICAgICMg'
    'QmFuZHdpZHRoIGFuYWx5c2lzIGNhbiBjaG9vc2UgdG8gZGVjaW1hdGUgTU9SRSBmb3IgZ2Vu'
    'dWluZWx5CiAgICAgICAgICAgICMgbG93LWJhbmR3aWR0aCBjb250ZW50IChlLmcuIHN1Yi1i'
    'YXNzIGF0IERTPTQgZXZlbiB3aGVuIHVzZXIKICAgICAgICAgICAgIyByZXF1ZXN0ZWQgRFM9'
    'MikuICBOZXZlciBsZXNzIHRoYW4gdXNlcidzIHJlcXVlc3QuCiAgICAgICAgICAgIGFjdHVh'
    'bF9kcyA9IGRvd25zYW1wbGUKICAgICAgICAgICAgaWYgRmFsc2U6ICAjIGF1dG8tRFMtYnVt'
    'cCBwZXJtYW5lbnRseSBkaXNhYmxlZCAoZm9ybWVyIG5vLWRzYnVtcCBiZWhhdmlvciwgbm93'
    'IGRlZmF1bHQpCiAgICAgICAgICAgICAgICAjIFRyeSBoaWdoZXIgZmFjdG9ycyBvbmx5IOKA'
    'lCBuZXZlciBsZXNzIHRoYW4gdXNlcidzIHJlcXVlc3QuCiAgICAgICAgICAgICAgICBmb3Ig'
    'ZiBpbiBbZG93bnNhbXBsZSAqIDIsIGRvd25zYW1wbGUgKiA0XToKICAgICAgICAgICAgICAg'
    'ICAgICBpZiBmID4gMTY6IGJyZWFrCiAgICAgICAgICAgICAgICAgICAgaWYgc3IgLyBmID49'
    'IG1heF9mcmVxICogMi40OgogICAgICAgICAgICAgICAgICAgICAgICBhY3R1YWxfZHMgPSBm'
    'CiAgICAgICAgICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICAgICAgYnJl'
    'YWsgICMgaWYgMnggZG9lc24ndCBzYXRpc2Z5IE55cXVpc3QsIDR4IHdvbid0IGVpdGhlcgog'
    'ICAgICAgICAgICAjIFJhdyBzdHJpZGUgZGVjaW1hdGlvbiDigJQgbm8gTFBGLCBwcmVzZXJ2'
    'ZXMgSEYgY29udGVudAogICAgICAgICAgICBpZiBhY3R1YWxfZHMgPiAxOgogICAgICAgICAg'
    'ICAgICAgZHMgPSByYXdbOjphY3R1YWxfZHNdLmNvcHkoKQogICAgICAgICAgICBlbHNlOgog'
    'ICAgICAgICAgICAgICAgZHMgPSByYXcuY29weSgpCiAgICAgICAgICAgIHNhbXBsZV9kcy5h'
    'cHBlbmQoYWN0dWFsX2RzKQogICAgICAgICAgICBjb25jYXRfZHMuZXh0ZW5kKGRzLnRvbGlz'
    'dCgpKQogICAgICAgICAgICAjIExvb3Atc2VhbSBzbW9vdGhpbmc6IGZvciBsb29waW5nIHNh'
    'bXBsZXMsIHJlcGxhY2UgdGhlIHBvc3QtbG9vcAogICAgICAgICAgICAjIGd1YXJkIHJlZ2lv'
    'biB3aXRoIHRoZSBGSVJTVCBmZXcgc2FtcGxlcyBmcm9tIGxvb3Bfc3RhcnQuCiAgICAgICAg'
    'ICAgICMgVGhpcyBtYWtlcyB2ZWN0b3JzIG5lYXIgbG9vcF9lbmQgaW5jbHVkZSBwcm9wZXIg'
    'd3JhcCBjb250ZXh0IHNvCiAgICAgICAgICAgICMgVlEgcXVhbnRpemF0aW9uIGRvZXNuJ3Qg'
    'aW50cm9kdWNlIGEgc3RlcCBkaXNjb250aW51aXR5IGF0IHRoZQogICAgICAgICAgICAjIGxv'
    'b3AgYm91bmRhcnkuICBXaXRob3V0IHRoaXMsIHZlY19kaW09OCBwcm9kdWNlcyBhbiBhdWRp'
    'YmxlIGJ1enoKICAgICAgICAgICAgIyBhdCB0aGUgbG9vcCByYXRlIChzYW1wbGVbbG9vcEVu'
    'ZC0xXSBhbmQgc2FtcGxlW2xvb3BTdGFydF0gZ2V0CiAgICAgICAgICAgICMgcXVhbnRpemVk'
    'IHRvIGluY29tcGF0aWJsZSBjb2RlYm9vayBwcm90b3R5cGVzKS4KICAgICAgICAgICAgIwog'
    'ICAgICAgICAgICAjIEVuY29kZXIgTU9ERmlsZSBzdG9yZXMgbG9vcF9zdGFydC9sb29wX2xl'
    'biBpbiBSQVcgYnl0ZSB1bml0cwogICAgICAgICAgICAjIChhbHJlYWR5IHByZS1tdWx0aXBs'
    'aWVkIGJ5IDIgaW4gc2FtcGxlc19pbmZvKS4KICAgICAgICAgICAgbG9vcF9sZW5fcmF3ID0g'
    'aW50KHMuZ2V0KCdsb29wX2xlbicsIDApIG9yIDApCiAgICAgICAgICAgIGlmIGxvb3BfbGVu'
    'X3JhdyA+IDQ6CiAgICAgICAgICAgICAgICBsb29wX3N0YXJ0X3JhdyA9IGludChzLmdldCgn'
    'bG9vcF9zdGFydCcsIDApIG9yIDApCiAgICAgICAgICAgICAgICAjIENvbnZlcnQgdG8gZGVj'
    'aW1hdGVkLXN0cmVhbSBpbmRleCAobWF0Y2hlcyBgZHNgIGFycmF5IGluZGV4aW5nKQogICAg'
    'ICAgICAgICAgICAgbG9vcF9zdGFydF9kcyA9IGxvb3Bfc3RhcnRfcmF3IC8vIGFjdHVhbF9k'
    'cwogICAgICAgICAgICAgICAgIyBDb21wdXRlIHRvdGFsIHBhZGRpbmcgbmVlZGVkOiBhbGln'
    'bi10by12ZWNfZGltICsgZXh0cmEgZ3VhcmQKICAgICAgICAgICAgICAgIHBhZF9jb3VudCA9'
    'ICh2ZWNfZGltIC0gbGVuKGNvbmNhdF9kcykgJSB2ZWNfZGltKSAlIHZlY19kaW0gKyA4CiAg'
    'ICAgICAgICAgICAgICAjIFRha2UgcGFkX2NvdW50IHNhbXBsZXMgc3RhcnRpbmcgZnJvbSBs'
    'b29wX3N0YXJ0IGluIHRoZQogICAgICAgICAgICAgICAgIyBkZWNpbWF0ZWQgZGF0YSDigJQg'
    'dGhpcyBpcyB3aGF0IHBsYXliYWNrIHdyYXBzIHRvLgogICAgICAgICAgICAgICAgd3JhcF9k'
    'YXRhID0gW10KICAgICAgICAgICAgICAgIGlmIGxvb3Bfc3RhcnRfZHMgPCBsZW4oZHMpOgog'
    'ICAgICAgICAgICAgICAgICAgIHRha2UgPSBtaW4ocGFkX2NvdW50LCBsZW4oZHMpIC0gbG9v'
    'cF9zdGFydF9kcykKICAgICAgICAgICAgICAgICAgICB3cmFwX2RhdGEuZXh0ZW5kKGRzLnRv'
    'bGlzdCgpW2xvb3Bfc3RhcnRfZHM6bG9vcF9zdGFydF9kcyt0YWtlXSkKICAgICAgICAgICAg'
    'ICAgIHdoaWxlIGxlbih3cmFwX2RhdGEpIDwgcGFkX2NvdW50OgogICAgICAgICAgICAgICAg'
    'ICAgIHdyYXBfZGF0YS5hcHBlbmQoMCkKICAgICAgICAgICAgICAgIGNvbmNhdF9kcy5leHRl'
    'bmQod3JhcF9kYXRhKQogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgIyBOb24t'
    'bG9vcGluZzogcGFkIHdpdGggemVyb3MgKG9yaWdpbmFsIGJlaGF2aW9yKQogICAgICAgICAg'
    'ICAgICAgd2hpbGUgbGVuKGNvbmNhdF9kcykgJSB2ZWNfZGltOiBjb25jYXRfZHMuYXBwZW5k'
    'KDApCiAgICAgICAgICAgICAgICBjb25jYXRfZHMuZXh0ZW5kKFswXSAqIDgpCiAgICAgICAg'
    'ZWxzZToKICAgICAgICAgICAgc2FtcGxlX2RzLmFwcGVuZChkb3duc2FtcGxlKQogICAgICAg'
    'ICAgICAjIEVtcHR5IHNhbXBsZToganVzdCBwYWQKICAgICAgICAgICAgd2hpbGUgbGVuKGNv'
    'bmNhdF9kcykgJSB2ZWNfZGltOiBjb25jYXRfZHMuYXBwZW5kKDApCiAgICAgICAgICAgIGNv'
    'bmNhdF9kcy5leHRlbmQoWzBdICogOCkKCiAgICB3aGlsZSBsZW4oY29uY2F0X2RzKSAlIHZl'
    'Y19kaW06IGNvbmNhdF9kcy5hcHBlbmQoMCkKICAgIHRvdGFsX3NhbXBsZXMgPSBsZW4oY29u'
    'Y2F0X2RzKQoKICAgIHZlY3RvcnMgPSBucC5hcnJheShjb25jYXRfZHMsIGR0eXBlPW5wLmZs'
    'b2F0MzIpLnJlc2hhcGUoLTEsIHZlY19kaW0pCgogICAgIyBTdGFnZSAxIOKAlCByaW5nLXdl'
    'aWdodGVkCiAgICB3ZWlnaHRzID0gTm9uZQogICAgaWYgd2VpZ2h0ZWQ6CiAgICAgICAgc2xv'
    'cGVzICA9IG5wLmFicyh2ZWN0b3JzWzosIC0xXSAtIHZlY3RvcnNbOiwgMF0pCiAgICAgICAg'
    'd2VpZ2h0cyA9IChzbG9wZXMgKyAxLjApCiAgICAgICAgd2VpZ2h0cyAvPSB3ZWlnaHRzLm1l'
    'YW4oKQoKICAgIHByaW50KGYiICBSVlEgw5d7ZG93bnNhbXBsZX0gU3RhZ2UgMTogSz17SzF9'
    'IG9uIHtsZW4odmVjdG9ycyl9IHt2ZWNfZGltfS12ZWN0b3JzLi4uIiwgZmx1c2g9VHJ1ZSkK'
    'ICAgIGttMSA9IE1pbmlCYXRjaEtNZWFucyhuX2NsdXN0ZXJzPUsxLCBuX2luaXQ9NSwgbWF4'
    'X2l0ZXI9NjAsIGJhdGNoX3NpemU9ODE5MiwKICAgICAgICAgICAgICAgICAgICAgICAgICBy'
    'YW5kb21fc3RhdGU9MCwgcmVhc3NpZ25tZW50X3JhdGlvPTAuMDEpCiAgICBrbTEuZml0KHZl'
    'Y3RvcnMsIHNhbXBsZV93ZWlnaHQ9d2VpZ2h0cykKICAgIGNvZGVzMSAgID0ga20xLnByZWRp'
    'Y3QodmVjdG9ycykuYXN0eXBlKG5wLmludDMyKQogICAgIyBDZW50cm9pZHMgYXJlIGluIFst'
    'MSwxXSBmbG9hdCByYW5nZSDigJQgc2NhbGUgYmFjayB0byBbLTEyOCwxMjddIGludCByYW5n'
    'ZSBmb3Igc3RvcmFnZQogICAgY2IxICAgICAgPSBucC5jbGlwKG5wLnJvdW5kKGttMS5jbHVz'
    'dGVyX2NlbnRlcnNfICogMTI4KSwgLTEyOCwgMTI3KS5hc3R5cGUobnAuaW50MzIpCiAgICBy'
    'ZXNpZHVhbCA9IHZlY3RvcnMgLSBrbTEuY2x1c3Rlcl9jZW50ZXJzX1tjb2RlczFdCgogICAg'
    'c25yMSA9IDEwKm5wLmxvZzEwKG5wLm1lYW4odmVjdG9ycyoqMikgLyAobnAubWVhbihyZXNp'
    'ZHVhbCoqMikgKyAxZS05KSkKICAgIHByaW50KGYiICBTdGFnZSAxIFNOUjoge3NucjE6LjJm'
    'fSBkQiIsIGZsdXNoPVRydWUpCgogICAgIyBTdGFnZSAyIChza2lwcGVkIHdoZW4gbm9fcnZx'
    'MiDihpIgSzI9PTApCiAgICBpZiBLMiA+IDA6CiAgICAgICAgcHJpbnQoZiIgIFJWUSBTdGFn'
    'ZSAyOiBLPXtLMn0gb24gcmVzaWR1YWwuLi4iLCBmbHVzaD1UcnVlKQogICAgICAgIGttMiA9'
    'IE1pbmlCYXRjaEtNZWFucyhuX2NsdXN0ZXJzPUsyLCBuX2luaXQ9NSwgbWF4X2l0ZXI9NjAs'
    'IGJhdGNoX3NpemU9ODE5MiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmFuZG9t'
    'X3N0YXRlPTEsIHJlYXNzaWdubWVudF9yYXRpbz0wLjAxKQogICAgICAgIGttMi5maXQocmVz'
    'aWR1YWwpCiAgICAgICAgY29kZXMyICAgICAgICAgPSBrbTIucHJlZGljdChyZXNpZHVhbCku'
    'YXN0eXBlKG5wLmludDMyKQogICAgICAgIGNiMiAgICAgICAgICAgID0gbnAuY2xpcChucC5y'
    'b3VuZChrbTIuY2x1c3Rlcl9jZW50ZXJzXyAqIDEyOCksIC0xMjgsIDEyNykuYXN0eXBlKG5w'
    'LmludDMyKQogICAgICAgIGZpbmFsX3Jlc2lkdWFsID0gcmVzaWR1YWwgLSBrbTIuY2x1c3Rl'
    'cl9jZW50ZXJzX1tjb2RlczJdCiAgICAgICAgc25yMiA9IDEwKm5wLmxvZzEwKG5wLm1lYW4o'
    'dmVjdG9ycyoqMikgLyAobnAubWVhbihmaW5hbF9yZXNpZHVhbCoqMikgKyAxZS05KSkKICAg'
    'ICAgICBwcmludChmIiAgUlZRIHRvdGFsIFNOUjoge3NucjI6LjJmfSBkQiAoK3tzbnIyLXNu'
    'cjE6LjJmfSBkQiBmcm9tIHN0YWdlIDIpIiwgZmx1c2g9VHJ1ZSkKICAgIGVsc2U6CiAgICAg'
    'ICAgcHJpbnQoZiIgIOKaoSBTdGFnZSAyIHNraXBwZWQgKC0tbm8tcnZxMik6IFNOUiA9IHtz'
    'bnIxOi4yZn0gZEIiLCBmbHVzaD1UcnVlKQogICAgICAgIGNvZGVzMiA9IG5wLnplcm9zX2xp'
    'a2UoY29kZXMxKSAgIyBwbGFjZWhvbGRlcgogICAgICAgIGNiMiAgICA9IG5wLnplcm9zKCgw'
    'LCB2ZWNfZGltKSwgZHR5cGU9bnAuaW50MzIpICAjIGVtcHR5IEsyIGNvZGVib29rCgogICAg'
    'IyBQYWNrIEJJVFMxK0JJVFMyIGJpdHMgcGVyIHZlY3RvciBMU0ItZmlyc3QKICAgICMgV2hl'
    'biBub19ydnEyIChLMj09MCksIEJJVFMyPT0wIHNvIGNvbWJpbmVkIGNvbGxhcHNlcyB0byBq'
    'dXN0IGNvZGVzMSBiaXRzLgogICAgbl92ZWNzICAgICAgPSBsZW4odmVjdG9ycykKICAgIHRv'
    'dGFsX2JpdHMgID0gbl92ZWNzICogQklUU19UT1RBTAogICAgdG90YWxfYnl0ZXMgPSAodG90'
    'YWxfYml0cyArIDcpIC8vIDgKICAgIGNvZGVzX2J5dGVzID0gYnl0ZWFycmF5KHRvdGFsX2J5'
    'dGVzKQogICAgbWFzazEgPSAoMSA8PCBCSVRTMSkgLSAxCiAgICBtYXNrMiA9ICgxIDw8IEJJ'
    'VFMyKSAtIDEgaWYgQklUUzIgPiAwIGVsc2UgMAogICAgZm9yIGkgaW4gcmFuZ2Uobl92ZWNz'
    'KToKICAgICAgICBpZiBLMiA+IDA6CiAgICAgICAgICAgIGNvbWJpbmVkICA9IChpbnQoY29k'
    'ZXMxW2ldKSAmIG1hc2sxKSB8ICgoaW50KGNvZGVzMltpXSkgJiBtYXNrMikgPDwgQklUUzEp'
    'CiAgICAgICAgZWxzZToKICAgICAgICAgICAgY29tYmluZWQgID0gaW50KGNvZGVzMVtpXSkg'
    'JiBtYXNrMQogICAgICAgIGJpdF9wb3MgICA9IGkgKiBCSVRTX1RPVEFMCiAgICAgICAgYnl0'
    'ZV9wb3MgID0gYml0X3BvcyA+PiAzCiAgICAgICAgYml0X3NoaWZ0ID0gYml0X3BvcyAmIDcK'
    'ICAgICAgICB2YWwgPSBjb21iaW5lZCA8PCBiaXRfc2hpZnQKICAgICAgICBjb2Rlc19ieXRl'
    'c1tieXRlX3Bvc10gICAgIHw9IHZhbCAgICAgICAgJiAweEZGCiAgICAgICAgaWYgYnl0ZV9w'
    'b3MrMSA8IHRvdGFsX2J5dGVzOiBjb2Rlc19ieXRlc1tieXRlX3BvcysxXSB8PSAodmFsID4+'
    'IDgpICAmIDB4RkYKICAgICAgICBpZiBieXRlX3BvcysyIDwgdG90YWxfYnl0ZXM6IGNvZGVz'
    'X2J5dGVzW2J5dGVfcG9zKzJdIHw9ICh2YWwgPj4gMTYpICYgMHhGRgogICAgICAgIGlmIGJ5'
    'dGVfcG9zKzMgPCB0b3RhbF9ieXRlczogY29kZXNfYnl0ZXNbYnl0ZV9wb3MrM10gfD0gKHZh'
    'bCA+PiAyNCkgJiAweEZGCgogICAgIyBDb2RlYm9vayBieXRlczogW0sxw5cyIGJ5dGVzXVtL'
    'MsOXMiBieXRlc10gc3RvcmVkIHVuc2lnbmVkICgrMTI4KQogICAgY2JfYnl0ZXMgPSBieXRl'
    'YXJyYXkoKQogICAgZm9yIGVudHJ5IGluIGNiMToKICAgICAgICBmb3IgdiBpbiBlbnRyeTog'
    'Y2JfYnl0ZXMuYXBwZW5kKChpbnQodikrMjU2KSAmIDB4RkYpCiAgICBpZiBLMiA+IDA6CiAg'
    'ICAgICAgZm9yIGVudHJ5IGluIGNiMjoKICAgICAgICAgICAgZm9yIHYgaW4gZW50cnk6IGNi'
    'X2J5dGVzLmFwcGVuZCgoaW50KHYpKzI1NikgJiAweEZGKQoKICAgIHJldHVybiBjb2Rlc19i'
    'eXRlcywgY2JfYnl0ZXMsIHN0YXJ0cywgdG90YWxfc2FtcGxlcywgQklUU19UT1RBTCwgc2Ft'
    'cGxlX2RzLCBLMSwgSzIKCgojIOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKVkOKV'
    'kOKVkOKVkOKVkOKVkOKVkOKVkAojIEdMU0wgRU1JVFRFUlMKIyDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZAKCmRlZiBieXRlc190b19p'
    'bnQzMl9iZV9hcnJheShkYXRhLCBjaHVua19pdmVjND01MTIpOgogICAgIiIiUGFjayBieXRl'
    'cyBpbnRvIGl2ZWM0IGFycmF5cyAoYmlnLWVuZGlhbjogYnl0ZSAwID0gTVNCIG9mIGludC54'
    'KS4iIiIKICAgICMgUGFkIHRvIG11bHRpcGxlIG9mIDE2IChzaW5jZSBlYWNoIGl2ZWM0IGhv'
    'bGRzIDE2IGJ5dGVzKQogICAgcGFkZGVkID0gYnl0ZXMoZGF0YSkgKyBiJ1x4MDAnICogKCgx'
    'NiAtIGxlbihkYXRhKSAlIDE2KSAlIDE2KQogICAgaW50cyA9IFtdCiAgICBmb3IgaSBpbiBy'
    'YW5nZSgwLCBsZW4ocGFkZGVkKSwgNCk6CiAgICAgICAgdiA9IHN0cnVjdC51bnBhY2soJz5J'
    'JywgcGFkZGVkW2k6aSs0XSlbMF0KICAgICAgICAjIGNvbnZlcnQgdG8gc2lnbmVkIGludDMy'
    'IGZvciBHTFNMIChoYW5kbGVzIHZhbHVlcyA+PSAyXjMxKQogICAgICAgIGlmIHYgPj0gKDEg'
    'PDwgMzEpOgogICAgICAgICAgICB2IC09ICgxIDw8IDMyKQogICAgICAgIGludHMuYXBwZW5k'
    'KHYpCiAgICAjIFNwbGl0IGludG8gaXZlYzQgYXJyYXkgY2h1bmtzIG9mIGNodW5rX2l2ZWM0'
    'IGl2ZWM0IGVudHJpZXMgZWFjaAogICAgY2h1bmtzID0gW10KICAgIGN1ciA9IFtdCiAgICBm'
    'b3IgaSBpbiByYW5nZSgwLCBsZW4oaW50cyksIDQpOgogICAgICAgIGN1ci5hcHBlbmQodHVw'
    'bGUoaW50c1tpOmkrNF0pKQogICAgICAgIGlmIGxlbihjdXIpID09IGNodW5rX2l2ZWM0Ogog'
    'ICAgICAgICAgICBjaHVua3MuYXBwZW5kKGN1cikKICAgICAgICAgICAgY3VyID0gW10KICAg'
    'IGlmIGN1cjoKICAgICAgICBjaHVua3MuYXBwZW5kKGN1cikKICAgIHJldHVybiBjaHVua3MK'
    'CmRlZiBlbWl0X2l2ZWM0X2FycmF5KG5hbWUsIGNodW5rc19vcl9zaW5nbGUsIGl0ZW1zX3Bl'
    'cl9saW5lPTIpOgogICAgIiIiRW1pdCBvbmUgb3IgbW9yZSBjb25zdCBpdmVjNCBhcnJheXMu'
    'IGBjaHVua3Nfb3Jfc2luZ2xlYCBpcyBhIGxpc3Qgb2YgY2h1bmtzLiIiIgogICAgb3V0ID0g'
    'W10KICAgIGZvciBjaSwgY2h1bmsgaW4gZW51bWVyYXRlKGNodW5rc19vcl9zaW5nbGUpOgog'
    'ICAgICAgIGFycl9uYW1lID0gZiJ7bmFtZX17Y2l9IiBpZiBsZW4oY2h1bmtzX29yX3Npbmds'
    'ZSkgPiAxIGVsc2UgZiJ7bmFtZX0wIgogICAgICAgIG91dC5hcHBlbmQoZiJjb25zdCBpdmVj'
    'NCB7YXJyX25hbWV9W3tsZW4oY2h1bmspfV0gPSBpdmVjNFtdKCIpCiAgICAgICAgbGluZXMg'
    'PSBbXQogICAgICAgIGZvciByb3dfc3RhcnQgaW4gcmFuZ2UoMCwgbGVuKGNodW5rKSwgaXRl'
    'bXNfcGVyX2xpbmUpOgogICAgICAgICAgICByb3cgPSBjaHVua1tyb3dfc3RhcnQ6cm93X3N0'
    'YXJ0ICsgaXRlbXNfcGVyX2xpbmVdCiAgICAgICAgICAgIHBhcnRzID0gWyJpdmVjNCh7fSx7'
    'fSx7fSx7fSkiLmZvcm1hdCgqdCkgZm9yIHQgaW4gcm93XQogICAgICAgICAgICBsaW5lcy5h'
    'cHBlbmQoIiAgICAiICsgIiwgIi5qb2luKHBhcnRzKSkKICAgICAgICBvdXQuYXBwZW5kKCIs'
    'XG4iLmpvaW4obGluZXMpKQogICAgICAgIG91dC5hcHBlbmQoIik7XG4iKQogICAgcmV0dXJu'
    'ICJcbiIuam9pbihvdXQpCgoKIyDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZAKIyBNQUlOIEJVSUxECiMg4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQCgpkZWYgbWFpbihtb2RfcGF0'
    'aCwgb3V0X3BhdGgsIEs9MjU2LCB3ZWlnaHRlZD1UcnVlLCBkb3duc2FtcGxlPTIsIGJpdHJh'
    'dGU9J2hpJywgdmVjX2RpbT0yLCByZXNhbXBsZXI9J2JzcGxpbmUnLCBub19ydnEyPUZhbHNl'
    'KToKICAgICIiIkdlbmVyYXRlIFNoYWRlclRveSBDb21tb24gR0xTTCBmb3IgYSBNT0QgZmls'
    'ZS4KICAgIGRvd25zYW1wbGU6IGFudGktYWxpYXMgZG93bnNhbXBsZSBmYWN0b3IgZm9yIHNh'
    'bXBsZSBlbmNvZGluZyAoMT1vZmYsIDI9cmVjb21tZW5kZWQpLgogICAgICAgICAgICAgICAg'
    'SGlnaGVyIGRvd25zYW1wbGUg4oaSIHNtYWxsZXIgZGF0YSwgbGFyZ2VyIGNvZGVib29rcywg'
    'YmV0dGVyIFNOUi4KICAgICAgICAgICAgICAgICAgw5cxOiBLMT02NCwgIEsyPTMyICAofjc5'
    'IEtCLCAgMjcuNyBkQikKICAgICAgICAgICAgICAgICAgw5cyOiBLMT01MTIsIEsyPTI1NiAo'
    'fjc3IEtCLCAgMzguNCBkQikgIOKGkCByZWNvbW1lbmRlZAogICAgICAgICAgICAgICAgICDD'
    'lzQ6IEsxPTUxMiwgSzI9MjU2ICh+MzkgS0IsICAzNy4xIGRCKQogICAgIiIiCiAgICBtb2Qg'
    'PSBNT0RGaWxlKG1vZF9wYXRoKQogICAgcHJpbnQoZiLwn5OmIExvYWRlZDoge21vZC50aXRs'
    'ZX0iKQogICAgcHJpbnQoZiIgICBQYXR0ZXJuczoge21vZC5udW1fcGF0dGVybnN9LCBTb25n'
    'IGxlbmd0aDoge21vZC5zb25nX2xlbmd0aH0iKQoKICAgICMgUGF0dGVybiBjcnVuY2gKICAg'
    'IHAgPSBlbmNvZGVfcGF0dGVybnMobW9kKQogICAgcHJpbnQoZiJcbvCfl5zvuI8gIFBBVFRF'
    'Uk4gQ1JVTkNIIikKICAgIHByaW50KGYiICAgVG90YWwgbm90ZXM6ICAgICAgIHtwWyd0b3Rh'
    'bF9ub3RlcyddfSIpCiAgICBwcmludChmIiAgIFVuaXF1ZSBub24tZW1wdHk6ICB7bGVuKHBb'
    'J3VuaXEnXSl9IikKICAgIHByaW50KGYiICAgRGljdGlvbmFyeTogICAgICAgIHtsZW4ocFsn'
    'dW5pcSddKSo0fSBieXRlcyIpCiAgICBwcmludChmIiAgIEJpdG1hcDogICAgICAgICAgICB7'
    'bGVuKHBbJ2JpdG1hcCddKX0gYnl0ZXMiKQogICAgcHJpbnQoZiIgICBJbmRleCBzdHJlYW06'
    'ICAgICAge2xlbihwWydpZHhfc3RyZWFtJ10pfSBieXRlcyIpCiAgICBwcmludChmIiAgIFJv'
    'dyBzZWVrICgxNi1iaXQgcHJlZml4KToge2xlbihwWydyb3dfc2Vla19ieXRlcyddKX0gYnl0'
    'ZXMiKQogICAgcGF0dGVybl90b3RhbCA9IGxlbihwWyd1bmlxJ10pKjQgKyBsZW4ocFsnYml0'
    'bWFwJ10pICsgbGVuKHBbJ2lkeF9zdHJlYW0nXSkgKyBsZW4ocFsncm93X3NlZWtfYnl0ZXMn'
    'XSkKICAgIHByaW50KGYiICAg4oaSIFBhdHRlcm4gdG90YWw6ICAge3BhdHRlcm5fdG90YWw6'
    'LH0gYnl0ZXMiKQoKICAgICMgU3BlZWQvdGljayB0YWJsZSBmcm9tIEZ4eCBlZmZlY3RzCiAg'
    'ICByb3dTcGVlZCwgcm93U3RhcnRUaWNrLCBicG1fY2hhbmdlcyA9IGNvbXB1dGVfcm93X3Nw'
    'ZWVkX3RhYmxlKG1vZCkKICAgIHByaW50KGYiXG7ij7HvuI8gIFNQRUVEIFRBQkxFIikKICAg'
    'IHByaW50KGYiICAgU29uZyByb3dzOiB7bGVuKHJvd1NwZWVkKX0sIHRvdGFsIHRpY2tzOiB7'
    'cm93U3RhcnRUaWNrWy0xXX0iKQogICAgcHJpbnQoZiIgICBVbmlxdWUgc3BlZWRzOiB7c29y'
    'dGVkKHNldChyb3dTcGVlZCkpfSIpCiAgICBzcGVlZF90YWJsZV9ieXRlcyA9IGxlbihyb3dT'
    'dGFydFRpY2spICogMgogICAgcHJpbnQoZiIgICByb3dTdGFydFRpY2s6IHtzcGVlZF90YWJs'
    'ZV9ieXRlc30gYnl0ZXMgKDE2LWJpdCBwYWNrZWQpIikKICAgIGlmIGJwbV9jaGFuZ2VzOgog'
    'ICAgICAgIHByaW50KGYiICAg4pqg77iPICBCUE0gY2hhbmdlcyBkZXRlY3RlZCAoMTJUSC5N'
    'T0QgaGFzIG5vbmUsIGJ1dCBvdGhlciBNT0RzIG1pZ2h0KSIpCgogICAgIyBBdXRvLXNlbGVj'
    'dCBkb3duc2FtcGxlIGlmIG5vdCBleHBsaWNpdGx5IG92ZXJyaWRkZW4gKGRvd25zYW1wbGU9'
    'MiBpcyBkZWZhdWx0KQogICAgIyBCdWRnZXQgZXN0aW1hdGU6IHRvdGFsX3Jhd19ieXRlcyAv'
    'IGRvd25zYW1wbGUgKiAxNy8xNiAoMTctYml0IGNvZGVzLCAyIGJ5dGVzL3NhbXBsZSkKICAg'
    'ICMgU2hhZGVyVG95IHNhZmUgem9uZTog4omkIDgwIEtCIHNhbXBsZSBjb2RlcyArIHBhdHRl'
    'cm4gZGF0YQogICAgaW1wb3J0IG51bXB5IGFzIG5wCiAgICB0b3RhbF9yYXcgPSBzdW0oc1sn'
    'bGVuZ3RoJ10gZm9yIHMgaW4gbW9kLnNhbXBsZXNfaW5mbykKICAgICMgTk9URTogdXNlci1y'
    'ZXF1ZXN0ZWQgZG93bnNhbXBsZSBpcyByZXNwZWN0ZWQgYXMgYSBIQVJEIENBUCDigJQgbm8g'
    'YXV0by1idW1wLgogICAgIyBVc2UgLS12ZWMtZGltIDQgb3IgLS1iaXRyYXRlIGxvIGlmIHlv'
    'dSBuZWVkIG1vcmUgc2l6ZSByZWR1Y3Rpb24uCiAgICBlc3RpbWF0ZWRfYnVkZ2V0X2RzMiA9'
    'ICh0b3RhbF9yYXcgLy8gZG93bnNhbXBsZSkgKiAxNyAvLyAxNiArIDE2MDAwICAjIGZvciBs'
    'b2cgb25seQoKICAgICMgU2FtcGxlIGVuY29kaW5nOiBSVlEgd2l0aCBiaXRyYXRlLWNvbnRy'
    'b2xsZWQgY29kZWJvb2sgKyBwZXItc2FtcGxlIERTCiAgICBkc19sYWJlbCA9IGYiw5d7ZG93'
    'bnNhbXBsZX0iIGlmIGRvd25zYW1wbGUgPiAxIGVsc2UgImZ1bGwtcmVzIgogICAgcHJpbnQo'
    'ZiJcbvCfl5zvuI8gIFNBTVBMRSBDUlVOQ0ggKFJWUSB7ZHNfbGFiZWx9IGJpdHJhdGU9e2Jp'
    'dHJhdGV9LCByaW5nLXdlaWdodGVkKSIpCiAgICBjb2Rlc19ieXRlcywgY2JfYnl0ZXMsIHN0'
    'YXJ0cywgdG90YWxfc2FtcGxlcywgYml0c19wZXJfY29kZSwgc2FtcGxlX2RzLCBLMSwgSzIg'
    'PSBlbmNvZGVfc2FtcGxlc192cTJkKAogICAgICAgIG1vZCwgSywgd2VpZ2h0ZWQsIGRvd25z'
    'YW1wbGU9ZG93bnNhbXBsZSwgYml0cmF0ZT1iaXRyYXRlLCB2ZWNfZGltPXZlY19kaW0sIG5v'
    'X3J2cTI9bm9fcnZxMikKICAgIEJJVFMxID0gaW50KG5wLmNlaWwobnAubG9nMihLMSkpKQog'
    'ICAgQklUUzIgPSBpbnQobnAuY2VpbChucC5sb2cyKEsyKSkpIGlmIEsyID4gMCBlbHNlIDAK'
    'ICAgIEJJVFNfVE9UQUwgPSBiaXRzX3Blcl9jb2RlCiAgICBwcmludChmIiAgIExvZ2ljYWwg'
    'c2FtcGxlczogICB7dG90YWxfc2FtcGxlczosfSAgKHtkc19sYWJlbH0pIikKICAgIHByaW50'
    'KGYiICAgQ29kZXMgcGFja2VkOiAgICAgIHtsZW4oY29kZXNfYnl0ZXMpOix9IGJ5dGVzICAo'
    'e2JpdHNfcGVyX2NvZGV9IGJpdHMvdmVjdG9yIMOXIHt0b3RhbF9zYW1wbGVzLy8yfSB2ZWN0'
    'b3JzKSIpCiAgICBwcmludChmIiAgIENvZGVib29rczogICAgICAgICB7bGVuKGNiX2J5dGVz'
    'KTosfSBieXRlcyAgKHtLMX3DlzIgKyB7SzJ9w5cyIGJ5dGVzKSIpCgogICAgdG90YWxfYnVk'
    'Z2V0ID0gcGF0dGVybl90b3RhbCArIGxlbihjb2Rlc19ieXRlcykgKyBsZW4oY2JfYnl0ZXMp'
    'ICsgMzEqMjQgKyBzcGVlZF90YWJsZV9ieXRlcwogICAgcHJpbnQoZiJcbvCfk4ogVE9UQUwg'
    'Y29uc3QgZGF0YSBidWRnZXQ6IH57dG90YWxfYnVkZ2V0Oix9IGJ5dGVzICAoe3RvdGFsX2J1'
    'ZGdldC8xMDI0Oi4xZn0gS0IpIikKCiAgICAjIENodW5rIGZvciBHTFNMCiAgICBkaWN0X2J5'
    'dGVzID0gYicnLmpvaW4ocFsndW5pcSddKQogICAgZGljdF9jaHVua3MgICAgPSBieXRlc190'
    'b19pbnQzMl9iZV9hcnJheShkaWN0X2J5dGVzKQogICAgYml0bWFwX2NodW5rcyAgPSBieXRl'
    'c190b19pbnQzMl9iZV9hcnJheShieXRlcyhwWydiaXRtYXAnXSkpCiAgICBpZHhfY2h1bmtz'
    'ICAgICA9IGJ5dGVzX3RvX2ludDMyX2JlX2FycmF5KGJ5dGVzKHBbJ2lkeF9zdHJlYW0nXSkp'
    'CiAgICByb3dzZWVrX2NodW5rcyA9IGJ5dGVzX3RvX2ludDMyX2JlX2FycmF5KGJ5dGVzKHBb'
    'J3Jvd19zZWVrX2J5dGVzJ10pKQogICAgY29kZXNfY2h1bmtzICAgPSBieXRlc190b19pbnQz'
    'Ml9iZV9hcnJheShieXRlcyhjb2Rlc19ieXRlcykpCiAgICBjYl9jaHVua3MgICAgICA9IGJ5'
    'dGVzX3RvX2ludDMyX2JlX2FycmF5KGJ5dGVzKGNiX2J5dGVzKSkKCiAgICAjIFBhY2sgcm93'
    'U3RhcnRUaWNrIGFzIDE2LWJpdCBMRSBieXRlcyDihpIgaXZlYzQgY2h1bmtzCiAgICB0aWNr'
    'X2J5dGVzID0gYnl0ZWFycmF5KCkKICAgIGZvciB0IGluIHJvd1N0YXJ0VGljazoKICAgICAg'
    'ICB0aWNrX2J5dGVzLmFwcGVuZCh0ICYgMHhGRikKICAgICAgICB0aWNrX2J5dGVzLmFwcGVu'
    'ZCgodCA+PiA4KSAmIDB4RkYpCiAgICB0aWNrX2NodW5rcyA9IGJ5dGVzX3RvX2ludDMyX2Jl'
    'X2FycmF5KGJ5dGVzKHRpY2tfYnl0ZXMpKQoKICAgIHNhbXBsZXNfaW5mb19uZXcgPSBbXQog'
    'ICAgZm9yIGksIChzLCBzdCkgaW4gZW51bWVyYXRlKHppcChtb2Quc2FtcGxlc19pbmZvLCBz'
    'dGFydHMpKToKICAgICAgICAjIFVzZSBwZXItc2FtcGxlIGFjdHVhbCBEUyBmb3IgbGVuZ3Ro'
    'L2xvb3Agc2NhbGluZwogICAgICAgIHNkcyA9IHNhbXBsZV9kc1tpXSBpZiBpIDwgbGVuKHNh'
    'bXBsZV9kcykgZWxzZSBkb3duc2FtcGxlCiAgICAgICAgc2FtcGxlc19pbmZvX25ldy5hcHBl'
    'bmQoZGljdCgKICAgICAgICAgICAgc3RhcnQ9c3QsCiAgICAgICAgICAgIGxlbmd0aD1zWyds'
    'ZW5ndGgnXSAvLyBzZHMsCiAgICAgICAgICAgIGxvb3BTdGFydD1zWydsb29wX3N0YXJ0J10g'
    'Ly8gc2RzLAogICAgICAgICAgICBsb29wTGVuPXNbJ2xvb3BfbGVuJ10gLy8gc2RzLAogICAg'
    'ICAgICAgICB2b2x1bWU9c1sndm9sdW1lJ10sIGZpbmV0dW5lPXNbJ2ZpbmV0dW5lJ10sCiAg'
    'ICAgICAgICAgIGJ3RmFjdG9yPXNkcywgICAjIGFjdHVhbCBEUyDigJQgdXNlZCBieSBHTFNM'
    'IGFzIGZyZXEgZGl2aXNvcgogICAgICAgICkpCgogICAgZ2xzbCA9IGJ1aWxkX2dsc2wobW9k'
    'LCBwLCBjb2Rlc19ieXRlcywgc3RhcnRzLCB0b3RhbF9zYW1wbGVzLAogICAgICAgICAgICAg'
    'ICAgICAgICBkaWN0X2NodW5rcywgYml0bWFwX2NodW5rcywgaWR4X2NodW5rcywgcm93c2Vl'
    'a19jaHVua3MsCiAgICAgICAgICAgICAgICAgICAgIGNvZGVzX2NodW5rcywgY2JfY2h1bmtz'
    'LCBzYW1wbGVzX2luZm9fbmV3LCBLLCBiaXRzX3Blcl9jb2RlLAogICAgICAgICAgICAgICAg'
    'ICAgICB0aWNrX2NodW5rcywgcm93U3RhcnRUaWNrLAogICAgICAgICAgICAgICAgICAgICBL'
    'MT1LMSwgSzI9SzIsIEJJVFMxPUJJVFMxLCBCSVRTMj1CSVRTMiwgQklUU19UT1RBTD1CSVRT'
    'X1RPVEFMLAogICAgICAgICAgICAgICAgICAgICBkb3duc2FtcGxlPWRvd25zYW1wbGUsIHZl'
    'Y19kaW09dmVjX2RpbSwgcmVzYW1wbGVyPXJlc2FtcGxlciwgbm9fcnZxMj1ub19ydnEyKQog'
    'ICAgd2l0aCBvcGVuKG91dF9wYXRoLCAndycpIGFzIGY6CiAgICAgICAgZi53cml0ZShnbHNs'
    'KQogICAgcHJpbnQoZiJcbuKchSBXcm90ZToge291dF9wYXRofSAgKHtsZW4oZ2xzbC5lbmNv'
    'ZGUoJ3V0Zi04JykpOix9IGJ5dGVzKSIpCgoKZGVmIGJ1aWxkX2dsc2wobW9kLCBwLCBwYWNr'
    'ZWQsIHN0YXJ0cywgdG90YWxfc2FtcGxlcywKICAgICAgICAgICAgICAgZGljdF9jaHVua3Ms'
    'IGJpdG1hcF9jaHVua3MsIGlkeF9jaHVua3MsIHJvd3NlZWtfY2h1bmtzLAogICAgICAgICAg'
    'ICAgICBjb2Rlc19jaHVua3MsIGNiX2NodW5rcywgc2FtcGxlc19pbmZvX25ldywgSywgYml0'
    'c19wZXJfY29kZSwKICAgICAgICAgICAgICAgdGlja19jaHVua3MsIHJvd1N0YXJ0VGljaywg'
    'SzE9NTEyLCBLMj0yNTYsIEJJVFMxPTksIEJJVFMyPTgsIEJJVFNfVE9UQUw9MTcsIGRvd25z'
    'YW1wbGU9MiwgdmVjX2RpbT0yLCByZXNhbXBsZXI9J2JzcGxpbmUnLCBub19ydnEyPUZhbHNl'
    'KToKCiAgICAjIOKUgOKUgCBTb25nIG1ldGFkYXRhCiAgICBzb25nX3Bvc2l0aW9ucyA9IG1v'
    'ZC5wYXR0ZXJuX29yZGVyWzptb2Quc29uZ19sZW5ndGhdCgogICAgIyBDb21wdXRlIGFjdHVh'
    'bCByb3dzIHBlciBzb25nIHBvc2l0aW9uIOKAlCBQcm9UcmFja2VyIER4eCAocGF0dGVybiBi'
    'cmVhaykKICAgICMgYW5kIEJ4eCAocG9zaXRpb24ganVtcCkgc2hvcnRlbiB0aGUgZWZmZWN0'
    'aXZlIHBhdHRlcm4gbGVuZ3RoLgogICAgZGVmIGFjdHVhbF9wYXR0ZXJuX3Jvd3Moc3ApOgog'
    'ICAgICAgIHBhdCA9IG1vZC5wYXR0ZXJuX29yZGVyW3NwXQogICAgICAgIE5DX2xvY2FsID0g'
    'bW9kLm51bV9jaGFubmVscwogICAgICAgIHBhdF9zaXplID0gNjQgKiBOQ19sb2NhbCAqIDQK'
    'ICAgICAgICBmb3Igcm93IGluIHJhbmdlKDY0KToKICAgICAgICAgICAgZm9yIGNoIGluIHJh'
    'bmdlKE5DX2xvY2FsKToKICAgICAgICAgICAgICAgIGJhc2UgPSAxMDg0ICsgcGF0KnBhdF9z'
    'aXplICsgcm93Kk5DX2xvY2FsKjQgKyBjaCo0CiAgICAgICAgICAgICAgICBuYiA9IG1vZC5k'
    'YXRhW2Jhc2U6YmFzZSs0XQogICAgICAgICAgICAgICAgZWZmID0gbmJbMl0gJiAweEYKICAg'
    'ICAgICAgICAgICAgIGlmIGVmZiA9PSAweEQgb3IgZWZmID09IDB4QjogICAjIHBhdHRlcm4g'
    'YnJlYWsgb3IgcG9zaXRpb24ganVtcAogICAgICAgICAgICAgICAgICAgIHJldHVybiByb3cg'
    'KyAxCiAgICAgICAgcmV0dXJuIDY0CgogICAgcGF0X3Jvd3MgPSBbYWN0dWFsX3BhdHRlcm5f'
    'cm93cyhzcCkgZm9yIHNwIGluIHJhbmdlKG1vZC5zb25nX2xlbmd0aCldCiAgICBwYXRfcm93'
    'X29mZnNldCA9IFswXQogICAgZm9yIHIgaW4gcGF0X3Jvd3M6CiAgICAgICAgcGF0X3Jvd19v'
    'ZmZzZXQuYXBwZW5kKHBhdF9yb3dfb2Zmc2V0Wy0xXSArIHIpCiAgICBwYXRfc3RhcnRfcm93'
    'ICA9IFswXSptb2Quc29uZ19sZW5ndGgKCiAgICAjIHBhdFRpY2tPZmZzZXRbc3BdID0gaW5k'
    'ZXggaW50byByb3dTdGFydFRpY2sgZm9yIHJvdyAwIG9mIHNvbmcgcG9zaXRpb24gc3AKICAg'
    'ICMgU2FtZSBhcyBwYXRfcm93X29mZnNldCBzaW5jZSB0aWNrIHRhYmxlIHJvd3MgPT0gc29u'
    'ZyByb3dzIGFmdGVyIEQwMCBmaXguCiAgICBwYXRfdGlja19vZmZzZXQgPSBwYXRfcm93X29m'
    'ZnNldFs6XQoKICAgIHRvdGFsX3Nvbmdfcm93cyA9IG1vZC5zb25nX2xlbmd0aCAqIDY0CiAg'
    'ICBudW1fcGF0dGVybnMgPSBtb2QubnVtX3BhdHRlcm5zCgogICAgIyDilIDilIAgU2FtcGxl'
    'SW5mbyBlbWlzc2lvbiAodXNlIG5ldyBgc3RhcnRgID0gc2FtcGxlIGluZGV4IGluIHRoZSBj'
    'b25jYXRlbmF0ZWQgc3RyZWFtKQogICAgZGVmIGZtdF9zYW1wbGVpbmZvKHMpOgogICAgICAg'
    'IHJldHVybiBmIlNhbXBsZUluZm8oe3NbJ3N0YXJ0J119LCB7c1snbGVuZ3RoJ119LCB7c1sn'
    'bG9vcFN0YXJ0J119LCB7c1snbG9vcExlbiddfSwge3NbJ3ZvbHVtZSddfSwge3MuZ2V0KCdi'
    'd0ZhY3RvcicsMSl9LCB7cy5nZXQoJ2ZpbmV0dW5lJywwKX0pIgogICAgc2lfbGluZXMgPSBb'
    'XQogICAgZm9yIGksIHMgaW4gZW51bWVyYXRlKHNhbXBsZXNfaW5mb19uZXcpOgogICAgICAg'
    'IHNpX2xpbmVzLmFwcGVuZChmIiAgICB7Zm10X3NhbXBsZWluZm8ocyl9eycsJyBpZiBpPDMw'
    'IGVsc2UgJyd9IikKICAgIHNhbXBsZXNfaW5mb19nbHNsID0gImNvbnN0IFNhbXBsZUluZm8g'
    'c2FtcGxlc1szMV0gPSBTYW1wbGVJbmZvW10oXG4iICsgIlxuIi5qb2luKHNpX2xpbmVzKSAr'
    'ICJcbik7IgoKICAgICMg4pSA4pSAIGNoYW5uZWxQYW4gKHNhbWUgYXMgZXhpc3Rpbmc6IEFt'
    'aWdhIExSUkwgd2l0aCByZXN0IGNlbnRlcmVkKQogICAgIyBCdWlsZCBwZXItY2hhbm5lbCBw'
    'YW4gZnJvbSB0aGUgc291cmNlIGZpbGUncyBwYW4gaW5mbyBpZiBhdmFpbGFibGUuCiAgICAj'
    'IFMzTSBmaWxlcyBzdG9yZSBwZXItY2hhbm5lbCBwYW4gaW4gYGNoYW5uZWxfc2V0dGluZ3Ng'
    'ICgzMiBieXRlcyBhdAogICAgIyBoZWFkZXIgb2Zmc2V0IDB4NDApOiBsb3cgNyBiaXRzID0g'
    'cG9zaXRpb24sIHdoZXJlIDAuLjcgPSBsZWZ0IFBDTQogICAgIyBjaGFubmVsIGFuZCA4Li4x'
    'NSA9IHJpZ2h0IFBDTSBjaGFubmVsLiBTQVRFTEwuUzNNIGZvciBleGFtcGxlIHVzZXMKICAg'
    'ICMgYW4gYWx0ZXJuYXRpbmcgTFJMUiBsYXlvdXQg4oCUIGhhcmRjb2RpbmcgTFJSTCB3b3Vs'
    'ZCBkdW1wIGJvdGggbGVhZAogICAgIyB2b2ljZXMgKGNoIDAgKyBjaCAzKSBvbnRvIHRoZSBM'
    'RUZUIGJ1cywgcHJvZHVjaW5nIGEgbGVmdC1oZWF2eSBtaXgKICAgICMgdGhhdCBzb3VuZHMg'
    'Y2xpcHBlZC9kaXN0b3J0ZWQgb24gdGhlIGxvdWQgc2lkZSBhbmQgdGhpbiBvbiB0aGUgcmln'
    'aHQuCiAgICAjIEZhbGwgYmFjayB0byBBbWlnYSBMUlJMIHdoZW4gdGhlIGZpbGUgZG9lc24n'
    'dCBzcGVjaWZ5IHBhbnMgKE1PRCkuCiAgICBfY3MgPSBnZXRhdHRyKG1vZCwgJ2NoYW5uZWxf'
    'c2V0dGluZ3MnLCBOb25lKSBvciBbXQogICAgY2hhbl9wYW4gPSBbXQogICAgZm9yIF9jaCBp'
    'biByYW5nZSgzMik6CiAgICAgICAgaWYgX2NzIGFuZCBfY2ggPCBsZW4oX2NzKToKICAgICAg'
    'ICAgICAgX3BvcyA9IF9jc1tfY2hdICYgMHg3RgogICAgICAgICAgICBpZiBfcG9zIDwgODoK'
    'ICAgICAgICAgICAgICAgIGNoYW5fcGFuLmFwcGVuZCgwLjApCiAgICAgICAgICAgICAgICBj'
    'b250aW51ZQogICAgICAgICAgICBlbGlmIDggPD0gX3BvcyA8IDE2OgogICAgICAgICAgICAg'
    'ICAgY2hhbl9wYW4uYXBwZW5kKDEuMCkKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAg'
    'ICAgIyBGYWxsYmFjazogTFJSTCBBbWlnYSBjb252ZW50aW9uIGZvciBmaXJzdCA0LCBjZW50'
    'ZXIgZm9yIGNoID49IDQKICAgICAgICBjaGFuX3Bhbi5hcHBlbmQoWzAuMCwgMS4wLCAxLjAs'
    'IDAuMF1bX2NoICUgNF0gaWYgX2NoIDwgNCBlbHNlIDAuNSkKCiAgICAjIOKUgOKUgCBDaHVu'
    'ayBhcnJheSBkZWNsYXJhdGlvbnMKICAgIGRpY3RfbGVuICAgID0gc3VtKGxlbihjKSBmb3Ig'
    'YyBpbiBkaWN0X2NodW5rcykKICAgIGJpdG1hcF9sZW4gID0gc3VtKGxlbihjKSBmb3IgYyBp'
    'biBiaXRtYXBfY2h1bmtzKQogICAgaWR4X2xlbiAgICAgPSBzdW0obGVuKGMpIGZvciBjIGlu'
    'IGlkeF9jaHVua3MpCiAgICByb3dzZWVrX2xlbiA9IHN1bShsZW4oYykgZm9yIGMgaW4gcm93'
    'c2Vla19jaHVua3MpCiAgICBjb2Rlc19sZW4gICA9IHN1bShsZW4oYykgZm9yIGMgaW4gY29k'
    'ZXNfY2h1bmtzKQogICAgY2JfbGVuICAgICAgPSBzdW0obGVuKGMpIGZvciBjIGluIGNiX2No'
    'dW5rcykKCiAgICBoZWFkZXIgPSBmIiIiLyogPT09PT09PT09PT09PT09PT09PT09PT09PT09'
    'PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQogICBH'
    'TFNMIChUaGUgTGFzdCkgTU9EIFBsYXllciB2MS40MiAoYykgMjAyNiBPcmJsaXZpdXMKICAg'
    'NCsgVHJhY2tzIHN1cHBvcnQsIFMzTS9NT0QgbG9hZGVyLCAzRCBTdXJyb3VuZCwgUGhhdEJh'
    'c3MsIENvbWIgUmV2ZXJiLCBGQVQsIFJWUSBzYW1wbGUgY29tcHJlc3Npb24sIGNvbmZpZ3Vy'
    'YWJsZSByZXNhbXBsZXIKICAgQ29udGFjdDogc3ViYmFuZEBnbWFpbC5jb20gb3IKICAgICAg'
    'ICAgICAgc3ViYmFuZEBwcm90b25tYWlsLmNvbQogICBHSVQ6ICAgICBodHRwczovL2dpdGh1'
    'Yi5jb20vbWV3emEvbW9kMmdsc2wKICAgQ09NTU9OIFRBQgogICBHZW5lcmF0ZWQgZnJvbTog'
    'e21vZC50aXRsZX0KICAgCiAgIENvbXByZXNzaW9uOgogICAgIOKAoiBQYXR0ZXJuczogYml0'
    'bWFwICsgZGljdGlvbmFyeSArIDE2LWJpdCBwcmVmaXgtc3VtIHJvdyBzZWVrIChPKDEpKQog'
    'ICAgIOKAoiBTYW1wbGVzOiAgMi1zdGFnZSBSVlEgw5d7ZG93bnNhbXBsZX0gQUEtZG93bnNh'
    'bXBsZWQgKEsxPXtLMX0sIEsyPXtLMn0pLCB7QklUU19UT1RBTH0gYml0cy9wYWlyCiAgICAg'
    'ICAgICAgICAgICAgcmluZy13ZWlnaHRlZCBrLW1lYW5zIHRyYWluZWQgb24gdGhpcyBNT0Qn'
    'cyBjb250ZW50CiAgID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09'
    'PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0gKi8KCiNkZWZpbmUgVVNFX0VN'
    'QkVEREVEX0RBVEEgMQojZGVmaW5lIE5VTV9QQVRURVJOUyAgICAgIHtudW1fcGF0dGVybnN9'
    'CiNkZWZpbmUgU09OR19MRU5HVEggICAgICAge21vZC5zb25nX2xlbmd0aH0KI2RlZmluZSBT'
    'T05HX0xPT1BfUE9TICAgICAwCiNkZWZpbmUgTlVNX0NIQU5ORUxTICAgICAge21vZC5udW1f'
    'Y2hhbm5lbHN9CiNkZWZpbmUgQlBNICAgICAgICAgICAgICAgMTI1LjAKI2RlZmluZSBTUEVF'
    'RCAgICAgICAgICAgICA2LjAKI2RlZmluZSBUT1RBTF9TT05HX1JPV1MgICB7dG90YWxfc29u'
    'Z19yb3dzfQoKLy8g4pSA4pSAIFBhdHRlcm4gY3J1bmNoIGNvbnN0YW50cyDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIAKI2RlZmluZSBUT1RBTF9OT1RFUyAgICAgICB7cFsndG90'
    'YWxfbm90ZXMnXX0KI2RlZmluZSBUT1RBTF9ST1dTICAgICAgICB7cFsnbnVtX3Jvd3MnXX0K'
    'I2RlZmluZSBESUNUX05PVEVTICAgICAgICB7bGVuKHBbJ3VuaXEnXSl9CiNkZWZpbmUgSURY'
    'X0JZVEVTX1BFUiAgICAge3BbJ2lkeF9ieXRlcyddfQojZGVmaW5lIERJQ1RfSU5UUyAgICAg'
    'ICAgIHtkaWN0X2xlbn0KI2RlZmluZSBCSVRNQVBfSU5UUyAgICAgICB7Yml0bWFwX2xlbn0K'
    'I2RlZmluZSBJRFhfSU5UUyAgICAgICAgICB7aWR4X2xlbn0KI2RlZmluZSBST1dTRUVLX0lO'
    'VFMgICAgICB7cm93c2Vla19sZW59CgovLyDilIDilIAgUlZRIHNhbXBsZSBjb25zdGFudHMg'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACi8vIFNhbXBs'
    'ZXMgYXJlIGFudGktYWxpYXMgZG93bnNhbXBsZWQgcGVyLXNhbXBsZSAoRFM9MSBmb3IgSEYg'
    'cGVyY3Vzc2lvbiwKLy8gRFM9e2Rvd25zYW1wbGV9IGZvciBtZWxvZGljKS4gUGVyLXNhbXBs'
    'ZSBEUyBpcyBzdG9yZWQgaW4gU2FtcGxlSW5mby5id0ZhY3Rvci4KLy8gcGVyaW9kVG9GcmVx'
    'ID0gNzA5Mzc4OS4yLyhwZXJpb2QqMikg4oCUIGJ3RmFjdG9yIGhhbmRsZXMgcGVyLXNhbXBs'
    'ZSBwaXRjaC4KI2RlZmluZSBSVlFfQ09ERVNfQllURVMgICB7bGVuKHBhY2tlZCl9CiNkZWZp'
    'bmUgUlZRX0NCX0JZVEVTICAgICAge0sxKjIgKyBLMioyfQojZGVmaW5lIFRPVEFMX1NBTVBM'
    'RVMgICAgIHt0b3RhbF9zYW1wbGVzfQoKI2RlZmluZSBCSVRNQVBfQllURVMgICAgICB7bGVu'
    'KHBbJ2JpdG1hcCddKX0KI2RlZmluZSBJRFhfQllURVMgICAgICAgICB7bGVuKHBbJ2lkeF9z'
    'dHJlYW0nXSl9CiNkZWZpbmUgUk9XU0VFS19CWVRFUyAgICAge2xlbihwWydyb3dfc2Vla19i'
    'eXRlcyddKX0KCi8vIOKUgOKUgCBGeHgtYXdhcmUgdGltaW5nIOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAojZGVmaW5lIFRPVEFMX1RJ'
    'Q0tTICAgICAgIHtyb3dTdGFydFRpY2tbLTFdfQojZGVmaW5lIE5VTV9TT05HX1JPV1MgICAg'
    'IHtsZW4ocm93U3RhcnRUaWNrKS0xfQojZGVmaW5lIFRJQ0tTX1BFUl9TRUMgICAgIDUwLjAg'
    'ICAvLyBCUE09MTI1IGNvbnN0YW50IGZvciAxMlRILk1PRAoKLy8g4pSA4pSAIEF1ZGlvIGVm'
    'ZmVjdHMg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSACmNvbnN0IGJvb2wgIGVuYWJsZTNEICAgICAgPSB0cnVlOwpj'
    'b25zdCBib29sICBlbmFibGVGQVQgICAgID0gdHJ1ZTsKY29uc3QgaXZlYzIgc3Vycl9jaGFu'
    'bmVscyA9IGl2ZWMyKDEsIDQpOwoiIiIKCiAgICAjIOKUgOKUgCBzb25nIG1ldGFkYXRhIGFy'
    'cmF5cwogICAgY2hhbl9wYW5fc3RyICAgPSAiLCAiLmpvaW4oZiJ7djouMWZ9IiBmb3IgdiBp'
    'biBjaGFuX3BhbikKICAgIHNvbmdwb3Nfc3RyICAgID0gIiwgIi5qb2luKHN0cih4KSBmb3Ig'
    'eCBpbiBzb25nX3Bvc2l0aW9ucykKICAgIHJvd29mZl9zdHIgICAgID0gIiwgIi5qb2luKHN0'
    'cih4KSBmb3IgeCBpbiBwYXRfcm93X29mZnNldCkKICAgIHN0YXJ0cm93X3N0ciAgID0gIiwg'
    'Ii5qb2luKHN0cih4KSBmb3IgeCBpbiBwYXRfc3RhcnRfcm93KQogICAgdGlja29mZl9zdHIg'
    'ICAgPSAiLCAiLmpvaW4oc3RyKHgpIGZvciB4IGluIHBhdF90aWNrX29mZnNldFs6LTFdKSAg'
    'IyBsZW5ndGggPSBzb25nX2xlbmd0aAoKICAgIG1ldGEgPSBmIiIiCmNvbnN0IGZsb2F0IGNo'
    'YW5uZWxQYW5bMzJdID0gZmxvYXRbXSh7Y2hhbl9wYW5fc3RyfSk7CmNvbnN0IGludCAgIHNv'
    'bmdQb3NpdGlvbnNbe21vZC5zb25nX2xlbmd0aH1dICAgPSBpbnRbXSh7c29uZ3Bvc19zdHJ9'
    'KTsKY29uc3QgaW50ICAgcGF0Um93T2Zmc2V0W3ttb2Quc29uZ19sZW5ndGgrMX1dICAgID0g'
    'aW50W10oe3Jvd29mZl9zdHJ9KTsKY29uc3QgaW50ICAgcGF0U3RhcnRSb3dbe21vZC5zb25n'
    'X2xlbmd0aH1dICAgICA9IGludFtdKHtzdGFydHJvd19zdHJ9KTsKY29uc3QgaW50ICAgcGF0'
    'VGlja09mZnNldFt7bW9kLnNvbmdfbGVuZ3RofV0gICA9IGludFtdKHt0aWNrb2ZmX3N0cn0p'
    'OwoiIiIKCiAgICAjIOKUgOKUgCBEYXRhIGFycmF5cyAoaXZlYzQgY2h1bmtzKQogICAgZGF0'
    'YV9hcnJheXMgPSBbIlxuLy8g4pSA4pSAIFBhdHRlcm4gZGljdGlvbmFyeSAodW5pcXVlIDQt'
    'Ynl0ZSBub3RlcywgTVNCLWZpcnN0IHBlciBpbnQpIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgFxuIl0KICAgIGRhdGFfYXJyYXlzLmFwcGVuZChlbWl0X2l2ZWM0X2FycmF5'
    'KCJwYXREaWN0IiwgZGljdF9jaHVua3MpKQogICAgZGF0YV9hcnJheXMuYXBwZW5kKCJcbi8v'
    'IOKUgOKUgCBQYXR0ZXJuIGJpdG1hcCAoMSBiaXQvbm90ZSwgTFNCLWZpcnN0IHdpdGhpbiBi'
    'eXRlKSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIBcbiIpCiAgICBkYXRhX2FycmF5cy5hcHBlbmQoZW1pdF9pdmVjNF9hcnJh'
    'eSgicGF0Qml0bWFwIiwgYml0bWFwX2NodW5rcykpCiAgICBkYXRhX2FycmF5cy5hcHBlbmQo'
    'IlxuLy8g4pSA4pSAIEluZGV4IHN0cmVhbSAoJXMgYnl0ZXMgcGVyIG5vbi1lbXB0eSBub3Rl'
    'KSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIBcbiIgJSBwWydpZHhfYnl0ZXMnXSkKICAgIGRh'
    'dGFfYXJyYXlzLmFwcGVuZChlbWl0X2l2ZWM0X2FycmF5KCJwYXRJZHgiLCBpZHhfY2h1bmtz'
    'KSkKICAgIGRhdGFfYXJyYXlzLmFwcGVuZCgiXG4vLyDilIDilIAgUm93IHNlZWsgdGFibGUg'
    'KDE2LWJpdCBMRSBwcmVmaXggc3VtcywgTygxKSBsb29rdXApIOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgFxuIikKICAgIGRhdGFfYXJy'
    'YXlzLmFwcGVuZChlbWl0X2l2ZWM0X2FycmF5KCJwYXRSb3dTZWVrIiwgcm93c2Vla19jaHVu'
    'a3MpKQogICAgZGF0YV9hcnJheXMuYXBwZW5kKCJcbi8vIOKUgOKUgCBWUSBjb2RlcyAocGFj'
    'a2VkIGJpdCBzdHJlYW0pIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgFxuIikKICAgIGRhdGFfYXJyYXlzLmFw'
    'cGVuZChlbWl0X2l2ZWM0X2FycmF5KCJ2cUNvZGVzIiwgY29kZXNfY2h1bmtzKSkKICAgIGRh'
    'dGFfYXJyYXlzLmFwcGVuZChmIlxuLy8g4pSA4pSAIFZRIGNvZGVib29rICh7S30gZW50cmll'
    'cyDDlyAyIHNhbXBsZXMsIHNpZ25lZCA4LWJpdCBhcyB1bnNpZ25lZCkg4pSA4pSAXG4iKQog'
    'ICAgZGF0YV9hcnJheXMuYXBwZW5kKGVtaXRfaXZlYzRfYXJyYXkoInZxQ29kZWJvb2siLCBj'
    'Yl9jaHVua3MpKQogICAgZGF0YV9hcnJheXMuYXBwZW5kKCJcbi8vIOKUgOKUgCBQZXItcm93'
    'IGN1bXVsYXRpdmUgdGljayB0YWJsZSAoMTYtYml0IExFLCBGeHgtYXdhcmUpIOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgFxuIikKICAgIGRhdGFf'
    'YXJyYXlzLmFwcGVuZChlbWl0X2l2ZWM0X2FycmF5KCJyb3dTdGFydFRpY2siLCB0aWNrX2No'
    'dW5rcykpCgogICAgIyDilIDilIAgU2FtcGxlSW5mbyAmIHBlcmlvZFRhYmxlCiAgICB0YWJs'
    'ZXMgPSBmIiIiCi8vIOKUgOKUgCBTYW1wbGUgbWV0YWRhdGEgKHN0YXJ0ID0gc2FtcGxlIGlu'
    'ZGV4IGluIHBhY2tlZCAzLWJpdCBzdHJlYW0pIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gApzdHJ1Y3QgU2FtcGxlSW5mbyB7ewogICAgaW50IHN0YXJ0LCBsZW5ndGgsIGxvb3BTdGFy'
    'dCwgbG9vcExlbiwgdm9sdW1lLCBid0ZhY3RvciwgZmluZXR1bmU7Cn19Owp7c2FtcGxlc19p'
    'bmZvX2dsc2x9CgovLyBQcm9UcmFja2VyIHBlcmlvZCB0YWJsZSAoQy0xIHRvIEItMykKY29u'
    'c3QgaW50IHBlcmlvZFRhYmxlWzM3XSA9IGludFtdKAogICAgODU2LDgwOCw3NjIsNzIwLDY3'
    'OCw2NDAsNjA0LDU3MCw1MzgsNTA4LDQ4MCw0NTMsCiAgICA0MjgsNDA0LDM4MSwzNjAsMzM5'
    'LDMyMCwzMDIsMjg1LDI2OSwyNTQsMjQwLDIyNiwKICAgIDIxNCwyMDIsMTkwLDE4MCwxNzAs'
    'MTYwLDE1MSwxNDMsMTM1LDEyNywxMjAsMTEzLDAKKTsKCi8vIFByb1RyYWNrZXIgMzItZW50'
    'cnkgc2luZSB0YWJsZSBmb3IgdmlicmF0byAoTFVULCBrZXB0IGdsb2JhbCBzbyBpdCBkb2Vz'
    'bid0Ci8vIGNvbnN1bWUgcGVyLWNhbGwgcHJpdmF0ZS9zdGFjayBzdG9yYWdlIGluIGdldENo'
    'YW5uZWxPdXRwdXQpLgpjb25zdCBmbG9hdCB2aWJUYWJbMzJdID0gZmxvYXRbXSgKICAgICAg'
    'MC4wLCAgMjQuMCwgIDQ5LjAsICA3NC4wLCAgOTcuMCwgMTIwLjAsIDE0MS4wLCAxNjEuMCwK'
    'ICAgIDE4MC4wLCAxOTcuMCwgMjEyLjAsIDIyNC4wLCAyMzUuMCwgMjQ0LjAsIDI1MC4wLCAy'
    'NTMuMCwKICAgIDI1NS4wLCAyNTMuMCwgMjUwLjAsIDI0NC4wLCAyMzUuMCwgMjI0LjAsIDIx'
    'Mi4wLCAxOTcuMCwKICAgIDE4MC4wLCAxNjEuMCwgMTQxLjAsIDEyMC4wLCAgOTcuMCwgIDc0'
    'LjAsICA0OS4wLCAgMjQuMAopOwoKLy8gQzQgc3BlZWRzIGZvciBlYWNoIGZpbmV0dW5lIHZh'
    'bHVlIChtaWtJVC9QVCBzcGVjKS4gIEluZGV4IDAuLjcgPSBwb3NpdGl2ZQovLyBmaW5ldHVu'
    'ZSAoc2xpZ2h0bHkgaGlnaGVyIHBpdGNoKSwgaW5kZXggOC4uMTUgPSBuZWdhdGl2ZSBmaW5l'
    'dHVuZSAobG93ZXIpLgovLyBJbiBzYW1wbGUgZGF0YSB3ZSBzdG9yZSBmaW5ldHVuZSBhcyBh'
    'IFNJR05FRCAtOC4uNyBpbnQg4oCUIGNvbnZlcnQgdmlhICYweEYuCmNvbnN0IGZsb2F0IGM0'
    'c3BlZWRzWzE2XSA9IGZsb2F0W10oCiAgICA4MzYzLjAsIDg0MTMuMCwgODQ2My4wLCA4NTI5'
    'LjAsIDg1ODEuMCwgODY1MS4wLCA4NzIzLjAsIDg3NTcuMCwKICAgIDc4OTUuMCwgNzk0MS4w'
    'LCA3OTg1LjAsIDgwNDYuMCwgODEwNy4wLCA4MTY5LjAsIDgyMzIuMCwgODI4MC4wCik7CmZs'
    'b2F0IHBlcmlvZFRvRnJlcShpbnQgcGVyaW9kKSB7ewogICAgLy8gRGVmYXVsdCAoZmluZXR1'
    'bmU9MCk6IDcwOTM3ODkuMiAvIChwZXJpb2Qgw5cgMikg4omIIDM1NDY4OTQuNi9wZXJpb2Qu'
    'ICBVc2UKICAgIC8vIHBlcmlvZFRvRnJlcUZ0IGJlbG93IHdoZW4gZmluZXR1bmUgbWF0dGVy'
    'cy4KICAgIHJldHVybiBwZXJpb2QgPiAwID8gNzA5Mzc4OS4yIC8gKGZsb2F0KHBlcmlvZCkg'
    'KiAyLjApIDogMC4wOwp9fQpmbG9hdCBwZXJpb2RUb0ZyZXFGdChpbnQgcGVyaW9kLCBpbnQg'
    'ZmluZXR1bmUpIHt7CiAgICAvLyAoYzQgKiA0MjgpIC8gcGVyaW9kIOKAlCBtYXRjaGVzIEhU'
    'TUwncyBwaXRjaCB0YWJsZSBleGFjdGx5LgogICAgaWYgKHBlcmlvZCA8PSAwKSByZXR1cm4g'
    'MC4wOwogICAgaW50IGlkeCA9IGZpbmV0dW5lICYgMHhGOyAgLy8gLTEgKHNpZ25lZCkg4oaS'
    'IDB4RiwgZXRjLgogICAgcmV0dXJuIChjNHNwZWVkc1tpZHhdICogNDI4LjApIC8gZmxvYXQo'
    'cGVyaW9kKTsKfX0KIiIiCgogICAgIyDilIDilIAgRmV0Y2ggaGVscGVycyAoY2h1bmsgZGlz'
    'cGF0Y2hlcnMgZm9yIGVhY2ggYXJyYXkpCiAgICBkZWYgY2h1bmtfZGlzcGF0Y2gobmFtZSwg'
    'bnVtX2NodW5rcywgdmFyPSdpJyk6CiAgICAgICAgaWYgbnVtX2NodW5rcyA9PSAxOgogICAg'
    'ICAgICAgICByZXR1cm4gZiIgICAgcmV0dXJuIHtuYW1lfTBbe3Zhcn0+PjJdOyIKICAgICAg'
    'ICBsaW5lcyA9IFtmIiAgICBpdmVjNCB2ID0gaXZlYzQoMCk7Il0KICAgICAgICBsaW5lcy5h'
    'cHBlbmQoZiIgICAgaWYgKGNodW5rSWR4ID09IDApIHYgPSB7bmFtZX0wW3t2YXJ9Pj4yXTsi'
    'KQogICAgICAgIGZvciBrIGluIHJhbmdlKDEsIG51bV9jaHVua3MpOgogICAgICAgICAgICBs'
    'aW5lcy5hcHBlbmQoZiIgICAgZWxzZSBpZiAoY2h1bmtJZHggPT0ge2t9KSB2ID0ge25hbWV9'
    'e2t9W3t2YXJ9Pj4yXTsiKQogICAgICAgIGxpbmVzLmFwcGVuZChmIiAgICByZXR1cm4gdjsi'
    'KQogICAgICAgIHJldHVybiAiXG4iLmpvaW4obGluZXMpCgogICAgZGVmIGl2ZWM0X3NlbGVj'
    'dCh2YXI9J2knKToKICAgICAgICByZXR1cm4gZiIiIiAgICBpbnQgY2kgPSB7dmFyfSAmIDM7'
    'CiAgICByZXR1cm4gY2k9PTAgPyB2LnggOiBjaT09MSA/IHYueSA6IGNpPT0yID8gdi56IDog'
    'di53OyIiIgoKICAgIGZldGNoZXJzID0gZiIiIgovLyDilZDilZDilZAgQ2h1bmtlZCBpdmVj'
    'NCBmZXRjaGVycyDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDi'
    'lZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZAKCi8vIEZl'
    'dGNoIGEgYnl0ZSBmcm9tIGFueSBjaHVua2VkIGJ5dGUgYXJyYXkgKE1TQi1maXJzdCB3aXRo'
    'aW4gZWFjaCBpbnQzMikuCi8vIEVhY2ggaXZlYzQgaG9sZHMgMTYgYnl0ZXM6IC54ID0gYnl0'
    'ZXMgMC0zLCAueSA9IDQtNywgLnogPSA4LTExLCAudyA9IDEyLTE1Ci8vIFdpdGhpbiBlYWNo'
    'IGludDogYnl0ZSAwID0gTVNCLCBieXRlIDMgPSBMU0IuCgppbnQgX2V4dHJhY3RCeXRlKGl2'
    'ZWM0IHYsIGludCBieXRlSW5JdmVjNCkge3sKICAgIGludCBpbnRJZHggPSBieXRlSW5JdmVj'
    'NCA+PiAyOwogICAgaW50IGJ5dGVJbkludCA9IGJ5dGVJbkl2ZWM0ICYgMzsKICAgIGludCBw'
    'YWNrZWQgPSBpbnRJZHg9PTAgPyB2LnggOiBpbnRJZHg9PTEgPyB2LnkgOiBpbnRJZHg9PTIg'
    'PyB2LnogOiB2Lnc7CiAgICBpbnQgc2hpZnQgPSAyNCAtIGJ5dGVJbkludCAqIDg7CiAgICBy'
    'ZXR1cm4gKHBhY2tlZCA+PiBzaGlmdCkgJiAweEZGOwp9fQoKLy8g4pSA4pSAIERpY3Rpb25h'
    'cnkgYnl0ZSBmZXRjaCAoYnl0ZUlkeCBpbiBbMCwgRElDVF9OT1RFUyo0KSkg4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACmludCBmZXRj'
    'aERpY3RCeXRlKGludCBieXRlSWR4KSB7ewogICAgaW50IGl2ZWM0SWR4ID0gYnl0ZUlkeCA+'
    'PiA0OwogICAgaW50IGJ5dGVJbkl2ZWM0ID0gYnl0ZUlkeCAmIDE1OwogICAgaXZlYzQgdiA9'
    'IHBhdERpY3QwW2l2ZWM0SWR4XTsKICAgIHJldHVybiBfZXh0cmFjdEJ5dGUodiwgYnl0ZUlu'
    'SXZlYzQpOwp9fQoKLy8g4pSA4pSAIEJpdG1hcCBieXRlIGZldGNoIOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU'
    'gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAppbnQgZmV0Y2hCaXRtYXBC'
    'eXRlKGludCBieXRlSWR4KSB7ewogICAgaW50IGl2ZWM0SWR4ID0gYnl0ZUlkeCA+PiA0Owog'
    'ICAgaW50IGJ5dGVJbkl2ZWM0ID0gYnl0ZUlkeCAmIDE1OwogICAgaXZlYzQgdiA9IHBhdEJp'
    'dG1hcDBbaXZlYzRJZHhdOwogICAgcmV0dXJuIF9leHRyYWN0Qnl0ZSh2LCBieXRlSW5JdmVj'
    'NCk7Cn19CgovLyDilIDilIAgSW5kZXggc3RyZWFtIGJ5dGUgZmV0Y2ggKGNodW5rZWQgaWYg'
    'bmVlZGVkKSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKaW50IGZldGNoSWR4Qnl0ZShpbnQg'
    'Ynl0ZUlkeCkge3sKICAgIGludCBpdmVjNElkeCA9IGJ5dGVJZHggPj4gNDsKICAgIGludCBi'
    'eXRlSW5JdmVjNCA9IGJ5dGVJZHggJiAxNTsKICAgIGludCBjaHVua0lkeCA9IGl2ZWM0SWR4'
    'IC8gNTEyOwogICAgaW50IGxvY2FsSXZlYzQgPSBpdmVjNElkeCAlIDUxMjsKICAgIGl2ZWM0'
    'IHYgPSBpdmVjNCgwKTsKe2NocigxMCkuam9pbihmJyAgICB7ImlmIiBpZiBrPT0wIGVsc2Ug'
    'ImVsc2UgaWYifSAoY2h1bmtJZHggPT0ge2t9KSB2ID0gcGF0SWR4e2t9W2xvY2FsSXZlYzRd'
    'OycgZm9yIGsgaW4gcmFuZ2UobGVuKGlkeF9jaHVua3MpKSl9CiAgICByZXR1cm4gX2V4dHJh'
    'Y3RCeXRlKHYsIGJ5dGVJbkl2ZWM0KTsKfX0KCi8vIOKUgOKUgCBSb3ctc2VlayBuaWJibGUg'
    'Ynl0ZSBmZXRjaCDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKaW50IGZldGNoUm93U2Vla0J5dGUo'
    'aW50IGJ5dGVJZHgpIHt7CiAgICBpbnQgaXZlYzRJZHggPSBieXRlSWR4ID4+IDQ7CiAgICBp'
    'bnQgYnl0ZUluSXZlYzQgPSBieXRlSWR4ICYgMTU7CiAgICBpdmVjNCB2ID0gcGF0Um93U2Vl'
    'azBbaXZlYzRJZHhdOwogICAgcmV0dXJuIF9leHRyYWN0Qnl0ZSh2LCBieXRlSW5JdmVjNCk7'
    'Cn19CgovLyDilIDilIAgVlEgY29kZSBzdHJlYW0gYnl0ZSBmZXRjaCAoY2h1bmtlZCkg4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA'
    '4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACmludCBmZXRj'
    'aENvZGVzQnl0ZShpbnQgYnl0ZUlkeCkge3sKICAgIGludCBpdmVjNElkeCA9IGJ5dGVJZHgg'
    'Pj4gNDsKICAgIGludCBieXRlSW5JdmVjNCA9IGJ5dGVJZHggJiAxNTsKICAgIGludCBjaHVu'
    'a0lkeCA9IGl2ZWM0SWR4IC8gNTEyOwogICAgaW50IGxvY2FsSXZlYzQgPSBpdmVjNElkeCAl'
    'IDUxMjsKICAgIGl2ZWM0IHYgPSBpdmVjNCgwKTsKe2NocigxMCkuam9pbihmJyAgICB7Imlm'
    'IiBpZiBrPT0wIGVsc2UgImVsc2UgaWYifSAoY2h1bmtJZHggPT0ge2t9KSB2ID0gdnFDb2Rl'
    'c3trfVtsb2NhbEl2ZWM0XTsnIGZvciBrIGluIHJhbmdlKGxlbihjb2Rlc19jaHVua3MpKSl9'
    'CiAgICByZXR1cm4gX2V4dHJhY3RCeXRlKHYsIGJ5dGVJbkl2ZWM0KTsKfX0KCi8vIOKUgOKU'
    'gCBWUSBjb2RlYm9vayBieXRlIGZldGNoIChzbWFsbCwgZml0cyBpbiAxIGNodW5rIHVzdWFs'
    'bHkpIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAppbnQgZmV0'
    'Y2hDb2RlYm9va0J5dGUoaW50IGJ5dGVJZHgpIHt7CiAgICBpbnQgaXZlYzRJZHggPSBieXRl'
    'SWR4ID4+IDQ7CiAgICBpbnQgYnl0ZUluSXZlYzQgPSBieXRlSWR4ICYgMTU7CiAgICBpbnQg'
    'Y2h1bmtJZHggPSBpdmVjNElkeCAvIDUxMjsKICAgIGludCBsb2NhbEl2ZWM0ID0gaXZlYzRJ'
    'ZHggJSA1MTI7CiAgICBpdmVjNCB2ID0gaXZlYzQoMCk7CntjaHIoMTApLmpvaW4oZicgICAg'
    'eyJpZiIgaWYgaz09MCBlbHNlICJlbHNlIGlmIn0gKGNodW5rSWR4ID09IHtrfSkgdiA9IHZx'
    'Q29kZWJvb2t7a31bbG9jYWxJdmVjNF07JyBmb3IgayBpbiByYW5nZShsZW4oY2JfY2h1bmtz'
    'KSkpfQogICAgcmV0dXJuIF9leHRyYWN0Qnl0ZSh2LCBieXRlSW5JdmVjNCk7Cn19CiIiIgoK'
    'ICAgICMg4pSA4pSAIHBvcGNvdW50IGhlbHBlciAoNC1iaXQgbmliYmxlKQogICAgIyDilIDi'
    'lIAgZ2V0Tm90ZTogYml0bWFwICsgZGljdCBsb29rdXAgd2l0aCBPKDEpIHJvdyBzZWVrICsg'
    'cHJlZml4IHBvcGNvdW50CiAgICBkZWNvZGVycyA9ICIiIgovLyDilZDilZDilZAgUGF0dGVy'
    'biBkZWNvZGVyOiBiaXRtYXAgKyBkaWN0aW9uYXJ5ICsgcm93IHNlZWsg4pWQ4pWQ4pWQ4pWQ'
    '4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ'
    '4pWQCgpzdHJ1Y3QgTm90ZSB7IGludCBpbnN0cnVtZW50LCBwZXJpb2QsIGVmZmVjdCwgcGFy'
    'YW07IGludCB2b2xfY29sOyB9OwoKLy8gUG9wY291bnQgZm9yIDQtYml0IG5pYmJsZSAoMC4u'
    'MTUg4oaSIDAuLjQpIOKAlCBrZXB0IGZvciBiYWNrLWNvbXBhdC4KaW50IHBvcGNvdW50NChp'
    'bnQgeCkgewogICAgeCA9ICh4ICYgMHg1KSArICgoeCA+PiAxKSAmIDB4NSk7CiAgICByZXR1'
    'cm4gKHggJiAweDMpICsgKCh4ID4+IDIpICYgMHgzKTsKfQoKLy8gUG9wY291bnQgZm9yIHVw'
    'IHRvIDE2IGJpdHMgKGhhbmRsZXMgTlVNX0NIQU5ORUxTID4gNCBpbiBTM00gZXRjLikuCmlu'
    'dCBwb3Bjb3VudDE2KGludCB4KSB7CiAgICB4ID0gKHggJiAweDU1NTUpICsgKCh4ID4+IDEp'
    'ICYgMHg1NTU1KTsKICAgIHggPSAoeCAmIDB4MzMzMykgKyAoKHggPj4gMikgJiAweDMzMzMp'
    'OwogICAgeCA9ICh4ICYgMHgwRjBGKSArICgoeCA+PiA0KSAmIDB4MEYwRik7CiAgICByZXR1'
    'cm4gKHggJiAweEZGKSArICgoeCA+PiA4KSAmIDB4RkYpOwp9CgovLyBSZWNvbnN0cnVjdCBj'
    'dW11bGF0aXZlIG5vbi1lbXB0eSBjb3VudCB1cCB0byBzdGFydCBvZiBgcm93YCDigJQgTygx'
    'KS4KLy8gUm93IHNlZWsgdGFibGUgaG9sZHMgMTYtYml0IExFIHByZWZpeCBzdW1zOiAyIGJ5'
    'dGVzIHBlciByb3cuCmludCByb3dTZWVrQ3VtKGludCB0YXJnZXRSb3cpIHsKICAgIGludCBi'
    'eXRlSWR4ID0gdGFyZ2V0Um93ICogMjsKICAgIGludCBsbyA9IGZldGNoUm93U2Vla0J5dGUo'
    'Ynl0ZUlkeCk7CiAgICBpbnQgaGkgPSBmZXRjaFJvd1NlZWtCeXRlKGJ5dGVJZHggKyAxKTsK'
    'ICAgIHJldHVybiBsbyB8IChoaSA8PCA4KTsKfQoKTm90ZSBlbXB0eU5vdGUoKSB7IE5vdGUg'
    'bjsgbi5pbnN0cnVtZW50PTA7IG4ucGVyaW9kPTA7IG4uZWZmZWN0PTA7IG4ucGFyYW09MDsg'
    'bi52b2xfY29sPTA7IHJldHVybiBuOyB9CgpOb3RlIGdldE5vdGUoaW50IHNvbmdQb3MsIGlu'
    'dCByb3csIGludCBjaGFubmVsKSB7CiAgICBpbnQgcGF0ID0gc29uZ1Bvc2l0aW9uc1tzb25n'
    'UG9zXTsKICAgIGludCByb3dHbG9iYWwgPSBwYXQgKiA2NCArIHJvdzsKICAgIC8vIEZJWEVE'
    'OiB3YXMgaGFyZGNvZGVkICIqIDQiIGFzc3VtaW5nIDQtY2hhbm5lbCBNT0QuIEZvciBTM00g'
    'd2l0aCB1cCB0bwogICAgLy8gMTYgY2hhbm5lbHMgcGVyIHJvdywgbXVzdCB1c2UgTlVNX0NI'
    'QU5ORUxTLiBUaGUgZW5jb2RlciBwYWNrcyBub3RlcyBhcwogICAgLy8gW3BhdCo2NCArIHJv'
    'd10qTlVNX0NIQU5ORUxTICsgY2ggaW50byB0aGUgYml0bWFwOyBwcmV2aW91c2x5IHRoZSBH'
    'TFNMCiAgICAvLyByZWFkICIqIDQiIHNvIGZvciBzb25ncyB3aXRoID40IGNoYW5uZWxzIGV2'
    'ZXJ5IG5vdGUgcGFzdCByb3cgMCBvZiBwYXQgMAogICAgLy8gd2FzIGRlY29kZWQgZnJvbSB0'
    'aGUgd3JvbmcgYml0bWFwIHBvc2l0aW9uIOKGkiBnYXJiYWdlIGNlbGxzLgogICAgaW50IG5v'
    'dGVJZHggICA9IHJvd0dsb2JhbCAqIE5VTV9DSEFOTkVMUyArIGNoYW5uZWw7CgogICAgLy8g'
    'MSkgQml0bWFwIGNoZWNrCiAgICBpbnQgYm1CeXRlID0gZmV0Y2hCaXRtYXBCeXRlKG5vdGVJ'
    'ZHggPj4gMyk7CiAgICBpbnQgYml0ID0gKGJtQnl0ZSA+PiAobm90ZUlkeCAmIDcpKSAmIDE7'
    'CiAgICBpZiAoYml0ID09IDApIHJldHVybiBlbXB0eU5vdGUoKTsKCiAgICAvLyAyKSBDb3Vu'
    'dCBub24tZW1wdHkgbm90ZXMgYmVmb3JlIHRoaXMgcG9zaXRpb24KICAgIC8vICAgID0gY3Vt'
    'dWxhdGl2ZSB1cCB0byByb3dHbG9iYWwgKyBwb3Bjb3VudCBvZiBiaXRtYXAgYml0cyB3aXRo'
    'aW4gdGhpcyByb3cKICAgIC8vICAgIHVwIHRvIChidXQgbm90IGluY2x1ZGluZykgdGhlIHJl'
    'cXVlc3RlZCBjaGFubmVsLgogICAgaW50IHJhbmsgPSByb3dTZWVrQ3VtKHJvd0dsb2JhbCk7'
    'CiAgICBpbnQgcm93Qml0bWFwU3RhcnQgPSByb3dHbG9iYWwgKiBOVU1fQ0hBTk5FTFM7CiAg'
    'ICAvLyBGb3IgTlVNX0NIQU5ORUxTIHVwIHRvIDE2ICsgd29yc3QtY2FzZSBiaXQgc2hpZnQs'
    'IHNwYW4gdXAgdG8gMjQgYml0cyA9IDMgYnl0ZXMuCiAgICBpbnQgYnl0ZTBJZHggPSByb3dC'
    'aXRtYXBTdGFydCA+PiAzOwogICAgaW50IHNoaWZ0ICAgID0gcm93Qml0bWFwU3RhcnQgJiA3'
    'OwogICAgaW50IGJ5dGUwID0gZmV0Y2hCaXRtYXBCeXRlKGJ5dGUwSWR4KTsKICAgIGludCBi'
    'eXRlMSA9IGZldGNoQml0bWFwQnl0ZShieXRlMElkeCArIDEpOwogICAgaW50IGJ5dGUyID0g'
    'ZmV0Y2hCaXRtYXBCeXRlKGJ5dGUwSWR4ICsgMik7CiAgICBpbnQgY29tYmluZWQgPSBieXRl'
    'MCB8IChieXRlMSA8PCA4KSB8IChieXRlMiA8PCAxNik7CiAgICBpbnQgcm93Qml0cyA9IChj'
    'b21iaW5lZCA+PiBzaGlmdCkgJiAoKDEgPDwgTlVNX0NIQU5ORUxTKSAtIDEpOwogICAgaW50'
    'IG1hc2sgPSAoMSA8PCBjaGFubmVsKSAtIDE7CiAgICByYW5rICs9IHBvcGNvdW50MTYocm93'
    'Qml0cyAmIG1hc2spOwoKICAgIC8vIDMpIExvb2sgdXAgaW5kZXggYW5kIGZldGNoIG5vdGUg'
    'ZnJvbSBkaWN0aW9uYXJ5CiAgICBpbnQgZGljdElkeDsKI2lmIElEWF9CWVRFU19QRVIgPT0g'
    'MQogICAgZGljdElkeCA9IGZldGNoSWR4Qnl0ZShyYW5rKTsKI2Vsc2UKICAgIGludCBsbyA9'
    'IGZldGNoSWR4Qnl0ZShyYW5rICogMik7CiAgICBpbnQgaGkgPSBmZXRjaElkeEJ5dGUocmFu'
    'ayAqIDIgKyAxKTsKICAgIGRpY3RJZHggPSBsbyB8IChoaSA8PCA4KTsKI2VuZGlmCiAgICBp'
    'bnQgYjAgPSBmZXRjaERpY3RCeXRlKGRpY3RJZHggKiA0ICsgMCk7CiAgICBpbnQgYjEgPSBm'
    'ZXRjaERpY3RCeXRlKGRpY3RJZHggKiA0ICsgMSk7CiAgICBpbnQgYjIgPSBmZXRjaERpY3RC'
    'eXRlKGRpY3RJZHggKiA0ICsgMik7CiAgICBpbnQgYjMgPSBmZXRjaERpY3RCeXRlKGRpY3RJ'
    'ZHggKiA0ICsgMyk7CgogICAgTm90ZSBuOwogICAgbi5pbnN0cnVtZW50ID0gKGIwICYgMHhG'
    'MCkgfCAoKGIyID4+IDQpICYgMHgwRik7CiAgICBuLnBlcmlvZCAgICAgPSAoKGIwICYgMHgw'
    'RikgPDwgOCkgfCBiMTsKICAgIG4uZWZmZWN0ICAgICA9IGIyICYgMHgwRjsKICAgIG4ucGFy'
    'YW0gICAgICA9IGIzOwogICAgcmV0dXJuIG47Cn0KCiIiIgoKICAgICMg4pSA4pSAIHZvbF9j'
    'b2wgcG9zdC1wYXRjaDogZml4IGRpY3RJZHggc3RyaWRlIGFuZCBhZGQgYjQgZmV0Y2ggZm9y'
    'IElUIOKUgOKUgAogICAgX2NzID0gcC5nZXQoJ2NlbGxfc2l6ZScsIDQpCiAgICBkZWNvZGVy'
    'cyA9IGRlY29kZXJzLnJlcGxhY2UoCiAgICAgICAgJyAgICBpbnQgYjAgPSBmZXRjaERpY3RC'
    'eXRlKGRpY3RJZHggKiA0ICsgMCk7XG4gICAgaW50IGIxID0gZmV0Y2hEaWN0Qnl0ZShkaWN0'
    'SWR4ICogNCArIDEpO1xuICAgIGludCBiMiA9IGZldGNoRGljdEJ5dGUoZGljdElkeCAqIDQg'
    'KyAyKTtcbiAgICBpbnQgYjMgPSBmZXRjaERpY3RCeXRlKGRpY3RJZHggKiA0ICsgMyk7JywK'
    'ICAgICAgICAnICAgIGludCBiMCA9IGZldGNoRGljdEJ5dGUoZGljdElkeCAqICcgKyBzdHIo'
    'X2NzKSArICcgKyAwKTtcbiAgICBpbnQgYjEgPSBmZXRjaERpY3RCeXRlKGRpY3RJZHggKiAn'
    'ICsgc3RyKF9jcykgKyAnICsgMSk7XG4gICAgaW50IGIyID0gZmV0Y2hEaWN0Qnl0ZShkaWN0'
    'SWR4ICogJyArIHN0cihfY3MpICsgJyArIDIpO1xuICAgIGludCBiMyA9IGZldGNoRGljdEJ5'
    'dGUoZGljdElkeCAqICcgKyBzdHIoX2NzKSArICcgKyAzKTsnCiAgICApCiAgICBpZiBfY3Mg'
    'PT0gNToKICAgICAgICBkZWNvZGVycyA9IGRlY29kZXJzLnJlcGxhY2UoCiAgICAgICAgICAg'
    'ICcgICAgbi5wYXJhbSAgICAgID0gYjM7XG4gICAgcmV0dXJuIG47XG59JywKICAgICAgICAg'
    'ICAgJyAgICBuLnBhcmFtICAgICAgPSBiMztcbiAgICBpbnQgYjQgPSBmZXRjaERpY3RCeXRl'
    'KGRpY3RJZHggKiAnICsgc3RyKF9jcykgKyAnICsgNCk7XG4gICAgbi52b2xfY29sICAgID0g'
    'YjQ7XG4gICAgcmV0dXJuIG47XG59JwogICAgICAgICkKCiAgICAjIFNhbXBsZSBkZWNvZGVy'
    'OiBmLXN0cmluZyBmb3IgI2RlZmluZXMgKG5lZWQgUHl0aG9uIHZhcnMpLCBwbGFpbiBzdHJp'
    'bmcgZm9yIGZ1bmN0aW9uIGJvZGllcwogICAgX3N0YWdlX2xhYmVsID0gIjEtc3RhZ2UgUlZR'
    'IChubyBzdGFnZSAyKSIgaWYgbm9fcnZxMiBlbHNlICIyLXN0YWdlIFJWUSIKICAgIF9wYWNr'
    'Zm10ICAgICA9IGYie0JJVFMxfS1iaXQgY29kZTEgb25seSIgaWYgbm9fcnZxMiBlbHNlIGYi'
    'W3tCSVRTMX0tYml0IGNvZGUxXVt7QklUUzJ9LWJpdCBjb2RlMl0iCiAgICBkZWNvZGVycyAr'
    'PSAoCiAgICAgICAgZiIvLyDilZDilZDilZAgU2FtcGxlIGRlY29kZXI6IHtfc3RhZ2VfbGFi'
    'ZWx9IMOXe2Rvd25zYW1wbGV9IEFBLWRvd25zYW1wbGVkIChwZXItc2FtcGxlIERTKSDilZDi'
    'lZBcbiIKICAgICAgICBmIi8vIHtCSVRTX1RPVEFMfS1iaXQgY29kZXMgcGFja2VkIExTQi1m'
    'aXJzdDoge19wYWNrZm10fVxuIgogICAgICAgIGYiLy8gcGVyaW9kVG9GcmVxID0gNzA5Mzc4'
    'OS4yLyhwZXJpb2QqMikg4oCUIHBlci1zYW1wbGUgRFMgdmlhIFNhbXBsZUluZm8uYndGYWN0'
    'b3JcbiIKICAgICAgICBmIiNkZWZpbmUgUlZRX0JJVFMgICAgIHtCSVRTX1RPVEFMfVxuIgog'
    'ICAgICAgIGYiI2RlZmluZSBSVlFfQklUU18xICAge0JJVFMxfVxuIgogICAgICAgIGYiI2Rl'
    'ZmluZSBSVlFfSzEgICAgICAge0sxfVxuIgogICAgICAgIGYiI2RlZmluZSBSVlFfSzIgICAg'
    'ICAge0syfVxuIgogICAgICAgIGYiI2RlZmluZSBSVlFfVkVDX0RJTSAge3ZlY19kaW19XG4i'
    'CiAgICAgICAgZiIjZGVmaW5lIFJWUV9DQjJfQllURSAoe0sxfSAqIHt2ZWNfZGltfSlcbiIK'
    'ICAgICAgICBmIiNkZWZpbmUgUlZRX01BU0sxICAgIHsoMTw8QklUUzEpLTF9XG4iCiAgICAg'
    'ICAgZiIjZGVmaW5lIFJWUV9NQVNLMiAgICB7KDE8PEJJVFMyKS0xIGlmIEJJVFMyPjAgZWxz'
    'ZSAwfVxuIgogICAgICAgICsgKGYiI2RlZmluZSBSVlFfTk9fU1RBR0UyIDFcbiIgaWYgbm9f'
    'cnZxMiBlbHNlICIiKQogICAgKQogICAgZGVjb2RlcnMgKz0gIiIiCnZvaWQgX2dldFJWUUNv'
    'ZGVzKGludCB2ZWNJZHgsIG91dCBpbnQgY29kZTEsIG91dCBpbnQgY29kZTIpIHsKICAgIGlu'
    'dCBiaXRQb3MgID0gdmVjSWR4ICogUlZRX0JJVFM7CiAgICBpbnQgYnl0ZVBvcyA9IGJpdFBv'
    'cyA+PiAzOwogICAgaW50IHNoaWZ0ICAgPSBiaXRQb3MgJiA3OwogICAgaW50IGIwID0gZmV0'
    'Y2hDb2Rlc0J5dGUoYnl0ZVBvcyk7CiAgICBpbnQgYjEgPSBmZXRjaENvZGVzQnl0ZShieXRl'
    'UG9zICsgMSk7CiAgICBpbnQgYjIgPSBmZXRjaENvZGVzQnl0ZShieXRlUG9zICsgMik7CiAg'
    'ICBpbnQgYjMgPSBmZXRjaENvZGVzQnl0ZShieXRlUG9zICsgMyk7CiAgICBpbnQgY29tYmlu'
    'ZWQgPSBiMCB8IChiMSA8PCA4KSB8IChiMiA8PCAxNikgfCAoYjMgPDwgMjQpOwogICAgaW50'
    'IHJhdyA9IChjb21iaW5lZCA+PiBzaGlmdCkgJiAoKDEgPDwgUlZRX0JJVFMpIC0gMSk7CiAg'
    'ICBjb2RlMSA9IHJhdyAmIFJWUV9NQVNLMTsKI2lmZGVmIFJWUV9OT19TVEFHRTIKICAgIGNv'
    'ZGUyID0gMDsKI2Vsc2UKICAgIGNvZGUyID0gKHJhdyA+PiBSVlFfQklUU18xKSAmIFJWUV9N'
    'QVNLMjsKI2VuZGlmCn0KCmZsb2F0IGdldFNhbXBsZShpbnQgc2FtcGxlSWR4KSB7CiAgICBp'
    'ZiAoc2FtcGxlSWR4IDwgMCB8fCBzYW1wbGVJZHggPj0gVE9UQUxfU0FNUExFUykgcmV0dXJu'
    'IDAuMDsKICAgIGludCB2ZWNJZHggPSBzYW1wbGVJZHggLyBSVlFfVkVDX0RJTTsKICAgIGlu'
    'dCBsYW5lICAgPSBzYW1wbGVJZHggLSB2ZWNJZHggKiBSVlFfVkVDX0RJTTsKICAgIC8vIElu'
    'bGluZSBSVlEgZGVjb2RlIChhdm9pZHMgb3V0LXBhcmFtZXRlciBzdGFjayBhbGxvY2F0aW9u'
    'KQogICAgaW50IF9icCA9IHZlY0lkeCAqIFJWUV9CSVRTLCBfYnkgPSBfYnAgPj4gMywgX3No'
    'ID0gX2JwICYgNzsKICAgIGludCBfcmF3ID0gKGZldGNoQ29kZXNCeXRlKF9ieSkgfCAoZmV0'
    'Y2hDb2Rlc0J5dGUoX2J5KzEpPDw4KSB8CiAgICAgICAgICAgICAgICAoZmV0Y2hDb2Rlc0J5'
    'dGUoX2J5KzIpPDwxNikgfCAoZmV0Y2hDb2Rlc0J5dGUoX2J5KzMpPDwyNCkpOwogICAgX3Jh'
    'dyA9IChfcmF3ID4+IF9zaCkgJiAoKDEgPDwgUlZRX0JJVFMpIC0gMSk7CiAgICBpbnQgY29k'
    'ZTEgPSBfcmF3ICYgUlZRX01BU0sxOwogICAgaW50IHViMSA9IGZldGNoQ29kZWJvb2tCeXRl'
    'KGNvZGUxICogUlZRX1ZFQ19ESU0gKyBsYW5lKTsKICAgIGludCBzMSAgPSB1YjEgPCAxMjgg'
    'PyB1YjEgOiB1YjEgLSAyNTY7CiNpZmRlZiBSVlFfTk9fU1RBR0UyCiAgICByZXR1cm4gZmxv'
    'YXQoczEpIC8gMTI4LjA7CiNlbHNlCiAgICBpbnQgY29kZTIgPSAoX3JhdyA+PiBSVlFfQklU'
    'U18xKSAmIFJWUV9NQVNLMjsKICAgIGludCB1YjIgPSBmZXRjaENvZGVib29rQnl0ZShSVlFf'
    'Q0IyX0JZVEUgKyBjb2RlMiAqIFJWUV9WRUNfRElNICsgbGFuZSk7CiAgICBpbnQgczIgID0g'
    'dWIyIDwgMTI4ID8gdWIyIDogdWIyIC0gMjU2OwogICAgcmV0dXJuIGZsb2F0KHMxICsgczIp'
    'IC8gMTI4LjA7CiNlbmRpZgp9CgovLyDilIDilIAgUG9zaXRpb24gY2FsY3VsYXRpb24gKEZ4'
    'eC1hd2FyZSB2aWEgcm93U3RhcnRUaWNrKSDilIDilIDilIDilIDilIDilIDilIDilIDilIDi'
    'lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKc3RydWN0IFBvc2l0aW9uIHsgaW50'
    'IHNvbmdQb3MsIHBhdHRlcm4sIHJvdzsgZmxvYXQgdGljaywgcm93VGltZTsgfTsKCi8vIEZl'
    'dGNoIDE2LWJpdCBMRSB2YWx1ZSBhdCByb3cgaW5kZXggaW50byByb3dTdGFydFRpY2sKaW50'
    'IGZldGNoVGljayhpbnQgcm93SWR4KSB7CiAgICBpbnQgYnl0ZUlkeCA9IHJvd0lkeCAqIDI7'
    'CiAgICBpbnQgY2h1bmtJZHggID0gYnl0ZUlkeCA+PiA2OwogICAgaW50IGJ5dGVJbjE2ICA9'
    'IGJ5dGVJZHggJiA2MzsKICAgIGludCBsbyA9IF9leHRyYWN0Qnl0ZShyb3dTdGFydFRpY2sw'
    'WyhjaHVua0lkeDw8MikrKGJ5dGVJbjE2Pj40KV0sIGJ5dGVJbjE2ICYgMTUpOwogICAgLy8g'
    'bmV4dCBieXRlCiAgICBpbnQgYnl0ZUlkeDIgPSBieXRlSWR4ICsgMTsKICAgIGludCBjaHVu'
    'a0lkeDIgPSBieXRlSWR4MiA+PiA2OwogICAgaW50IGJ5dGVJbjE2XzIgPSBieXRlSWR4MiAm'
    'IDYzOwogICAgaW50IGhpID0gX2V4dHJhY3RCeXRlKHJvd1N0YXJ0VGljazBbKGNodW5rSWR4'
    'Mjw8MikrKGJ5dGVJbjE2XzI+PjQpXSwgYnl0ZUluMTZfMiAmIDE1KTsKICAgIHJldHVybiBs'
    'byB8IChoaSA8PCA4KTsKfQoKUG9zaXRpb24gZ2V0UG9zaXRpb24oZmxvYXQgdGltZSkgewog'
    'ICAgUG9zaXRpb24gcG9zOwogICAgZmxvYXQgc29uZ0R1cmF0aW9uID0gZmxvYXQoVE9UQUxf'
    'VElDS1MpIC8gVElDS1NfUEVSX1NFQzsKICAgIGZsb2F0IGxvb3BlZFRpbWUgPSBtb2QodGlt'
    'ZSwgc29uZ0R1cmF0aW9uKTsKICAgIGZsb2F0IHRvdGFsVGlja0YgPSBsb29wZWRUaW1lICog'
    'VElDS1NfUEVSX1NFQzsKCiAgICAvLyBCaW5hcnkgc2VhcmNoIHJvd1N0YXJ0VGljayBmb3Ig'
    'dGhlIGN1cnJlbnQgcm93CiAgICBpbnQgbG8gPSAwLCBoaSA9IE5VTV9TT05HX1JPV1M7CiAg'
    'ICBmb3IgKGludCBfYnMgPSAwOyBfYnMgPCAxMjsgX2JzKyspIHsgIC8vIGxvZzIoMTkyMCsp'
    'IOKJiCAxMQogICAgICAgIGlmIChsbyA+PSBoaSAtIDEpIGJyZWFrOwogICAgICAgIGludCBt'
    'aWQgPSAobG8gKyBoaSkgPj4gMTsKICAgICAgICBpZiAoZmxvYXQoZmV0Y2hUaWNrKG1pZCkp'
    'IDw9IHRvdGFsVGlja0YpIGxvID0gbWlkOwogICAgICAgIGVsc2UgaGkgPSBtaWQ7CiAgICB9'
    'CiAgICBpbnQgZ2xvYmFsUm93ID0gbG87CiAgICBpZiAoZ2xvYmFsUm93ID49IE5VTV9TT05H'
    'X1JPV1MpIGdsb2JhbFJvdyA9IE5VTV9TT05HX1JPV1MgLSAxOwoKICAgIC8vIEZpbmQgc29u'
    'Z1BvcyB2aWEgbGluZWFyIHNlYXJjaCBvdmVyIHBhdFRpY2tPZmZzZXQgKFNPTkdfTEVOR1RI'
    'IOKJpCAxMjgsIGZhc3QgZW5vdWdoKQogICAgaW50IHNwID0gU09OR19MRU5HVEggLSAxOwog'
    'ICAgZm9yIChpbnQgX2kgPSAwOyBfaSA8IFNPTkdfTEVOR1RIIC0gMTsgX2krKykgewogICAg'
    'ICAgIGlmIChwYXRUaWNrT2Zmc2V0W19pICsgMV0gPiBnbG9iYWxSb3cpIHsgc3AgPSBfaTsg'
    'YnJlYWs7IH0KICAgIH0KICAgIHBvcy5zb25nUG9zID0gc3A7CiAgICBwb3MucGF0dGVybiA9'
    'IHNvbmdQb3NpdGlvbnNbc3BdOwogICAgcG9zLnJvdyAgICAgPSBnbG9iYWxSb3cgLSBwYXRU'
    'aWNrT2Zmc2V0W3NwXTsKCiAgICBpbnQgcm93VGljayAgICA9IGZldGNoVGljayhnbG9iYWxS'
    'b3cpOwogICAgaW50IG5leHRUaWNrICAgPSBmZXRjaFRpY2soZ2xvYmFsUm93ICsgMSk7CiAg'
    'ICBpbnQgcm93U3BlZWQgICA9IG5leHRUaWNrIC0gcm93VGljazsKICAgIHBvcy50aWNrICAg'
    'ICAgID0gdG90YWxUaWNrRiAtIGZsb2F0KHJvd1RpY2spOwogICAgcG9zLnJvd1RpbWUgICAg'
    'PSBmbG9hdChyb3dTcGVlZCkgLyBUSUNLU19QRVJfU0VDOwogICAgcmV0dXJuIHBvczsKfQoK'
    'Ly8gNC1wb2ludCBjdWJpYyBCLXNwbGluZSBpbnRlcnBvbGF0aW9uLgovLyBCLXNwbGluZSBp'
    'cyBBUFBST1hJTUFUSU5HIChzbW9vdGhzIHRocm91Z2ggc2FtcGxlIHBvaW50cykgcmF0aGVy'
    'IHRoYW4KLy8gSU5URVJQT0xBVElORyAocGFzc2luZyBleGFjdGx5IHRocm91Z2ggdGhlbSks'
    'IGdpdmluZyBpbmhlcmVudCBsb3ctcGFzcwovLyBjaGFyYWN0ZXIgdGhhdCByZWR1Y2VzIGhp'
    'Z2gtZnJlcXVlbmN5IHF1YW50aXphdGlvbiBub2lzZS4KIiIiICsgKAogICAgICAgICAgICAj'
    'IOKUgOKUgCBMaW5lYXI6IDIgdGFwcywgUHJvVHJhY2tlci1hdXRoZW50aWMsIGNoZWFwZXN0'
    'IOKUgOKUgAogICAgICAgICAgICAnJydmbG9hdCBnZXRTYW1wbGVGKGludCBiYXNlLCBmbG9h'
    'dCBmcG9zLCBpbnQgc21wTGVuLCBpbnQgbG9vcFN0YXJ0LCBpbnQgbG9vcExlbikgewogICAg'
    'aW50IGkgPSBpbnQoZnBvcyk7CiAgICBmbG9hdCB0ID0gZnBvcyAtIGZsb2F0KGkpOwogICAg'
    'ZmxvYXQgcDEgPSBnZXRTYW1wbGUoYmFzZSArIGkpOwogICAgZmxvYXQgcDIgPSBnZXRTYW1w'
    'bGUoYmFzZSArIG1pbihpICsgMSwgc21wTGVuICsgMTUpKTsKICAgIHJldHVybiBtaXgocDEs'
    'IHAyLCB0KTsKfScnJyBpZiByZXNhbXBsZXIgPT0gJ2xpbmVhcicgZWxzZQogICAgICAgICAg'
    'ICAjIOKUgOKUgCBMYW5jem9zLTM6IDYgdGFwcywgc2hhcnBlc3QsIGJyaWdodGVzdCDilIDi'
    'lIAKICAgICAgICAgICAgJycnLy8gTGFuY3pvcy0zIHdpbmRvd2VkIHNpbmM6IHcoeCkgPSBz'
    'aW5jKM+AeCkgKiBzaW5jKM+AeC8zKSBmb3IgfHh8PDMKZmxvYXQgX2xhbmN6b3MzKGZsb2F0'
    'IHgpIHsKICAgIGlmICh4IDwgMWUtNikgcmV0dXJuIDEuMDsKICAgIGZsb2F0IHBpeCA9IDMu'
    'MTQxNTkyNjUgKiB4OwogICAgZmxvYXQgcGl4MyA9IHBpeCAvIDMuMDsKICAgIHJldHVybiAo'
    'c2luKHBpeCkgKiBzaW4ocGl4MykpIC8gKHBpeCAqIHBpeDMpOwp9CmZsb2F0IGdldFNhbXBs'
    'ZUYoaW50IGJhc2UsIGZsb2F0IGZwb3MsIGludCBzbXBMZW4sIGludCBsb29wU3RhcnQsIGlu'
    'dCBsb29wTGVuKSB7CiAgICBpbnQgaSAgPSBpbnQoZnBvcyk7CiAgICBmbG9hdCB0ID0gZnBv'
    'cyAtIGZsb2F0KGkpOwogICAgaW50IGltMiA9IGkgLSAyLCBpbTEgPSBpIC0gMSwgaXAxID0g'
    'aSArIDEsIGlwMiA9IGkgKyAyLCBpcDMgPSBpICsgMzsKICAgIGludCBsb29wRW5kID0gbG9v'
    'cFN0YXJ0ICsgbG9vcExlbjsgIC8vIG9uZS1wYXN0IGxhc3QgbG9vcCBzYW1wbGUKICAgIC8v'
    'IExvb3Agd3JhcGFyb3VuZCBmb3IgQUxMIGtlcm5lbCB0YXBzLiBUaGUgTGFuY3pvcyBrZXJu'
    'ZWwgcmVhY2hlcyAyCiAgICAvLyBzYW1wbGVzIGJhY2sgYW5kIDMgZm9yd2FyZDsgd2hlbmV2'
    'ZXIgYW55IG9mIHRob3NlIGZhbGxzIG91dHNpZGUKICAgIC8vIFtsb29wU3RhcnQsIGxvb3BF'
    'bmQpIHdoaWxlIHdlJ3JlIHBsYXlpbmcgSU4gdGhlIGxvb3AsIGl0IG11c3Qgd3JhcC4KICAg'
    'IC8vCiAgICAvLyBUaGUgcHJldmlvdXMgdmVyc2lvbiB3cmFwcGVkIG9ubHkgdGhlIEJBQ0tX'
    'QVJEIHRhcHMgKGltMiwgaW0xKSBhbmQKICAgIC8vIENMQU1QRUQgdGhlIGZvcndhcmQgdGFw'
    'cyB0byBzbXBMZW4rMTUuIEF0IGhpZ2ggcGl0Y2ggdGhlIHNvdXJjZS1zdGVwCiAgICAvLyBw'
    'ZXIgb3V0cHV0IHNhbXBsZSBpcyBsYXJnZSAoZS5nLiA1LjfDlyksIHNvIGV2ZXJ5IGxvb3Ag'
    'aXRlcmF0aW9uIGNyb3NzZXMKICAgIC8vIHRoZSBsb29wIGJvdW5kYXJ5OyBmb3J3YXJkLWNs'
    'YW1wIG1lYW50IGlwMS9pcDIvaXAzIHJlYWQgcGFzdCBzbXBMZW4KICAgIC8vIGludG8gYWRq'
    'YWNlbnQgZ2FyYmFnZSBldmVyeSBsb29wIGN5Y2xlLCBjcmVhdGluZyBidXp6IHRoYXQgc2Nh'
    'bGVkIHdpdGgKICAgIC8vIHBpdGNoIOKAlCBleGFjdGx5IHRoZSBmYWlsdXJlIG1vZGUgd2Ug'
    'b2JzZXJ2ZWQ6IGNsZWFuIG5vdGUgYm9kaWVzLCBidXQKICAgIC8vIDMtOSUgSEYgZW5lcmd5'
    'IG9uIGV2ZXJ5IGxvb3AtdGFpbCByZXBlYXQuCiAgICAvLwogICAgLy8gUHJlLWxvb3AgKGkg'
    'PCBsb29wU3RhcnQpOiBiYWNrd2FyZCB0YXBzIGNsYW1wIHRvIDAgKHNpbGVudCBwcmVmaXgp'
    'LAogICAgLy8gZm9yd2FyZCB0YXBzIGNsYW1wIHRvIHNtcExlbisxNSAocG9zdC1zYW1wbGUg'
    'cGFkZGluZyB6ZXJvcykuIFN0YW5kYXJkCiAgICAvLyBhdHRhY2stcmVnaW9uIGJlaGF2aW9y'
    'LgogICAgaWYgKGxvb3BMZW4gPiAyICYmIGkgPj0gbG9vcFN0YXJ0KSB7CiAgICAgICAgaWYg'
    'KGltMiA8IGxvb3BTdGFydCkgaW0yID0gbG9vcEVuZCArIChpbTIgLSBsb29wU3RhcnQpOwog'
    'ICAgICAgIGlmIChpbTEgPCBsb29wU3RhcnQpIGltMSA9IGxvb3BFbmQgKyAoaW0xIC0gbG9v'
    'cFN0YXJ0KTsKICAgICAgICBpZiAoaXAxID49IGxvb3BFbmQpICBpcDEgPSBsb29wU3RhcnQg'
    'KyAoaXAxIC0gbG9vcEVuZCk7CiAgICAgICAgaWYgKGlwMiA+PSBsb29wRW5kKSAgaXAyID0g'
    'bG9vcFN0YXJ0ICsgKGlwMiAtIGxvb3BFbmQpOwogICAgICAgIGlmIChpcDMgPj0gbG9vcEVu'
    'ZCkgIGlwMyA9IGxvb3BTdGFydCArIChpcDMgLSBsb29wRW5kKTsKICAgIH0gZWxzZSB7CiAg'
    'ICAgICAgaXAxID0gbWluKGlwMSwgc21wTGVuICsgMTUpOwogICAgICAgIGlwMiA9IG1pbihp'
    'cDIsIHNtcExlbiArIDE1KTsKICAgICAgICBpcDMgPSBtaW4oaXAzLCBzbXBMZW4gKyAxNSk7'
    'CiAgICB9CiAgICBpbTIgPSBtYXgoMCwgaW0yKTsgaW0xID0gbWF4KDAsIGltMSk7CiAgICBm'
    'bG9hdCB3MCA9IF9sYW5jem9zMyhhYnModCArIDIuMCkpOwogICAgZmxvYXQgdzEgPSBfbGFu'
    'Y3pvczMoYWJzKHQgKyAxLjApKTsKICAgIGZsb2F0IHcyID0gX2xhbmN6b3MzKGFicyh0ICAg'
    'ICAgKSk7CiAgICBmbG9hdCB3MyA9IF9sYW5jem9zMyhhYnModCAtIDEuMCkpOwogICAgZmxv'
    'YXQgdzQgPSBfbGFuY3pvczMoYWJzKHQgLSAyLjApKTsKICAgIGZsb2F0IHc1ID0gX2xhbmN6'
    'b3MzKGFicyh0IC0gMy4wKSk7CiAgICBmbG9hdCB3c3VtID0gdzArdzErdzIrdzMrdzQrdzU7'
    'CiAgICByZXR1cm4gKHcwKmdldFNhbXBsZShiYXNlK2ltMikgKyB3MSpnZXRTYW1wbGUoYmFz'
    'ZStpbTEpICsKICAgICAgICAgICAgdzIqZ2V0U2FtcGxlKGJhc2UraSAgKSArIHczKmdldFNh'
    'bXBsZShiYXNlK2lwMSkgKwogICAgICAgICAgICB3NCpnZXRTYW1wbGUoYmFzZStpcDIpICsg'
    'dzUqZ2V0U2FtcGxlKGJhc2UraXAzKSkgLyB3c3VtOwp9JycnIGlmIHJlc2FtcGxlciA9PSAn'
    'bGFuY3pvczMnIGVsc2UKICAgICAgICAgICAgIyDilIDilIAgQi1zcGxpbmUgKGRlZmF1bHQp'
    'OiA0IHRhcHMsIHNtb290aCwgZ2VudGxlIExQRiDilIDilIAKICAgICAgICAgICAgJycnZmxv'
    'YXQgZ2V0U2FtcGxlRihpbnQgYmFzZSwgZmxvYXQgZnBvcywgaW50IHNtcExlbiwgaW50IGxv'
    'b3BTdGFydCwgaW50IGxvb3BMZW4pIHsKICAgIGludCBpICA9IGludChmcG9zKTsKICAgIGZs'
    'b2F0IHQgPSBmcG9zIC0gZmxvYXQoaSk7CiAgICBpbnQgaTAgPSBpIC0gMTsKICAgIGlmIChs'
    'b29wTGVuID4gMiAmJiBpMCA8IGxvb3BTdGFydCkgaTAgPSBsb29wU3RhcnQgKyBsb29wTGVu'
    'IC0gMTsKICAgIGVsc2UgaTAgPSBtYXgoMCwgaTApOwogICAgZmxvYXQgcDAgPSBnZXRTYW1w'
    'bGUoYmFzZSArIGkwKTsKICAgIGZsb2F0IHAxID0gZ2V0U2FtcGxlKGJhc2UgKyBpKTsKICAg'
    'IGZsb2F0IHAyID0gZ2V0U2FtcGxlKGJhc2UgKyBtaW4oaSArIDEsIHNtcExlbiArIDE1KSk7'
    'CiAgICBmbG9hdCBwMyA9IGdldFNhbXBsZShiYXNlICsgbWluKGkgKyAyLCBzbXBMZW4gKyAx'
    'NSkpOwogICAgZmxvYXQgdDIgPSB0ICogdDsKICAgIGZsb2F0IHQzID0gdDIgKiB0OwogICAg'
    'ZmxvYXQgdzAgPSAoMS4wIC0gdCkgKiAoMS4wIC0gdCkgKiAoMS4wIC0gdCkgLyA2LjA7CiAg'
    'ICBmbG9hdCB3MSA9ICgzLjAgKiB0MyAtIDYuMCAqIHQyICsgNC4wKSAvIDYuMDsKICAgIGZs'
    'b2F0IHcyID0gKC0zLjAgKiB0MyArIDMuMCAqIHQyICsgMy4wICogdCArIDEuMCkgLyA2LjA7'
    'CiAgICBmbG9hdCB3MyA9IHQzIC8gNi4wOwogICAgcmV0dXJuIHcwICogcDAgKyB3MSAqIHAx'
    'ICsgdzIgKiBwMiArIHczICogcDM7Cn0nJycKICAgICAgICApICsgIiIiCgoiIiIKCiAgICBp'
    'bXBvcnQgYmFzZTY0IGFzIF9iNjRlCiAgICBnZXRfY2hhbm5lbF9vdXRwdXQgPSBfYjY0ZS5i'
    'NjRkZWNvZGUoJ0x5OGdkbWxpVkdGaUlHbHpJR1JsWTJ4aGNtVmtJR0Z6SUdFZ1oyeHZZbUZz'
    'SUdOdmJuTjBJR1pzYjJGMFd6TXlYU0J1WldGeUlIUm9aU0IwYjNBZ2IyWWdRMjl0Ylc5dUNp'
    'OHZJQ2h5YVdkb2RDQmhablJsY2lCd1pYSnBiMlJVWVdKc1pTa3VJRVJ2YmlkMElISmxaR1Zq'
    'YkdGeVpTQnBkQ0JvWlhKbExnb0tMeThnNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRw'
    'U0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNB'
    'NHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRw'
    'U0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNB'
    'NHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRw'
    'U0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQUNpOHZJRjluWTI5Q2IyUjVJT0tB'
    'bENCeVpXNWtaWElnYjI1bElITmhiWEJzWlNCdlppQmhJR05vWVc1dVpXd2daMmwyWlc0Z1Mw'
    'NVBWMDRnZEhKcFoyZGxjaUJwYm1adkxnb3ZMd292THlCRmVIUnlZV04wWldRZ1puSnZiU0Ju'
    'WlhSRGFHRnVibVZzVDNWMGNIVjBKM01nWW05a2VTQjBieUJ6ZFhCd2IzSjBJSEJ5WlhacGIz'
    'VnpMVzV2ZEdVZ1kzSnZjM05tWVdSbExnb3ZMeUJVYUdVZ2IzSnBaMmx1WVd3Z2IzVjBaWEln'
    'Wm5WdVkzUnBiMjRnWkdsa0lDSm1hVzVrSUhSeWFXZG5aWElnNG9hU0lHTnZiWEIxZEdVZ2Iz'
    'VjBjSFYwSUdGeklHOXVaUW92THlCellXMXdiR1V1SWlCR2IzSWdZM0p2YzNObVlXUmxJSGRs'
    'SUc1bFpXUWdkRzhnWTI5dGNIVjBaU0IwYUdVZ2IzVjBjSFYwSUZSWFNVTkZJT0tBbENCdmJt'
    'TmxJR1p2Y2dvdkx5QjBhR1VnWTNWeWNtVnVkQ0IwY21sbloyVnlMQ0J2Ym1ObElHWnZjaUIw'
    'YUdVZ2RISnBaMmRsY2lCQ1JVWlBVa1VnYVhRZzRvQ1VJR0Z1WkNCaWJHVnVaQ0J2ZG1WeUNp'
    'OHZJSFJvWlNCbWFYSnpkQ0EyTkNCellXMXdiR1Z6SUdGbWRHVnlJR0VnY21WMGNtbG5aMlZ5'
    'TGdvdkx3b3ZMeUJRWVhKaGJXVjBaWEp6SUhSeWFXZFFZWFF2ZEhKcFoxSnZkeTkwY21sblRt'
    'OTBaUzkwYjI1bFUyeHBaR1ZVWVhKblpYUWdZWEpsSUhkb1lYUWdkR2hsSUc5MWRHVnlDaTh2'
    'SUdaMWJtTjBhVzl1SjNNZ2RISnBaMmRsY2lCelpXRnlZMmdnZDI5MWJHUWdhR0YyWlNCamIy'
    'MXdkWFJsWkRzZ2RHaGxJR0p2WkhrZ2RYTmxjeUIwYUdWdENpOHZJR2xrWlc1MGFXTmhiR3g1'
    'SUNoM1lYTWdZWE1nWUd4dlkyRnNJSFpoY2lBOUlISmxjM1ZzZENCdlppQnpaV0Z5WTJoZ0xD'
    'QnViM2NnWVhNZ2NHRnlZVzFsZEdWeWN5a3VDaTh2SU9LVWdPS1VnT0tVZ09LVWdPS1VnT0tV'
    'Z09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdP'
    'S1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tV'
    'Z09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdP'
    'S1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tV'
    'Z09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdBcG1iRzloZENCZloy'
    'TnZRbTlrZVNocGJuUWdZMmdzSUZCdmMybDBhVzl1SUhCdmN5d2dabXh2WVhRZ2RHbHRaU3dn'
    'Wm14dllYUWdjbTkzVkdsdFpTd0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJSFJ5YVdkUVlY'
    'UXNJR2x1ZENCMGNtbG5VbTkzTENCT2IzUmxJSFJ5YVdkT2IzUmxMQ0JwYm5RZ2RHOXVaVk5z'
    'YVdSbFZHRnlaMlYwS1NCN0NpQWdJQ0JwWmlBb2RISnBaMDV2ZEdVdWFXNXpkSEoxYldWdWRD'
    'QThQU0F3SUh4OElIUnlhV2RPYjNSbExtbHVjM1J5ZFcxbGJuUWdQaUF6TVNCOGZDQjBjbWxu'
    'VG05MFpTNXdaWEpwYjJRZ1BEMGdNQ2tLSUNBZ0lDQWdJQ0J5WlhSMWNtNGdNQzR3T3dvS0lD'
    'QWdJRk5oYlhCc1pVbHVabThnYzIxd0lEMGdjMkZ0Y0d4bGMxdDBjbWxuVG05MFpTNXBibk4w'
    'Y25WdFpXNTBJQzBnTVYwN0NpQWdJQ0JwWmlBb2MyMXdMbXhsYm1kMGFDQTlQU0F3S1NCeVpY'
    'UjFjbTRnTUM0d093b0tJQ0FnSUM4dklGUnBZMnN0WW1GelpXUWdaV3hoY0hObFpEb2dhVzVz'
    'YVc1bElFZFNJR052YlhCMWRHRjBhVzl1TENCemEybHdJRzVoYldWa0lHbHVkR1Z5YldWa2FX'
    'RjBaWE1LSUNBZ0lHWnNiMkYwSUdWc1lYQnpaV1FnUFNBb1pteHZZWFFvWm1WMFkyaFVhV05y'
    'S0hCaGRGUnBZMnRQWm1aelpYUmJjRzl6TG5OdmJtZFFiM05kS3lod2IzTXVjbTkzTFhCaGRG'
    'TjBZWEowVW05M1czQnZjeTV6YjI1blVHOXpYU2twS1FvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQXJJSEJ2Y3k1MGFXTnJDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUMwZ1pt'
    'eHZZWFFvWm1WMFkyaFVhV05yS0hCaGRGUnBZMnRQWm1aelpYUmJkSEpwWjFCaGRGMHJLSFJ5'
    'YVdkU2IzY3RjR0YwVTNSaGNuUlNiM2RiZEhKcFoxQmhkRjBwS1NrcENpQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdMeUJVU1VOTFUxOVFSVkpmVTBWRE93b2dJQ0FnYVdZZ0tHVnNZWEJ6'
    'WldRZ1BDQXdMakFwSUhKbGRIVnliaUF3TGpBN0Nnb2dJQ0FnTHk4ZzRwU0E0cFNBNHBTQUlH'
    'WlRZVzF3YkdWUWIzTWdZV05qZFcxMWJHRjBiM0lnS0dWNFlXTjBJSEJsY2kxeWIzY2diR2x1'
    'WldGeUxYSmhiWEFnYVc1MFpXZHlZWFJwYjI0cElPS1VnT0tVZ09LVWdBb2dJQ0FnTHk4Z1Vt'
    'VmhiQ0JRVkNCd1pYSnBiMlFnWlhadmJIVjBhVzl1SUdseklIQnBaV05sZDJselpTMXNhVzVs'
    'WVhJZ2QybDBhQ0JpY21WaGEzQnZhVzUwY3lCaGRDQnliM2NLSUNBZ0lDOHZJR0p2ZFc1a1lY'
    'SnBaWE1nS0dGdVpDQmpiR0Z0Y0hNZ2QyaGxiaUF6ZUhnZ2NtVmhZMmhsY3lCMFlYSm5aWFFw'
    'TGlCVWFHVWdjMmx1WjJ4bExXbHVkR1ZuY21Gc0NpQWdJQ0F2THlCbWIzSnRkV3hoSUdCREts'
    'UXZaRkFnS2lCc2JpaFFNUzlRTUNsZ0lHWnliMjBnZEhKcFoyZGxjaUIwYnlCamRYSnlaVzUw'
    'SUdseklIZHliMjVuSUdKbFkyRjFjMlVLSUNBZ0lDOHZJR2wwSUdGemMzVnRaWE1nWVNCVFNV'
    'NUhURVVnYkdsdVpXRnlJSEpoYlhBZzRvQ1VJR0ZqZEhWaGJDQlFLSFFwSUdseklHMTFiSFJw'
    'TFhObFoyMWxiblF1SUVacGVEb0tJQ0FnSUM4dklHRmpZM1Z0ZFd4aGRHVWdaWGhoWTNRZ2NH'
    'VnlMWEp2ZHlCamIyNTBjbWxpZFhScGIyNXpJR1IxY21sdVp5QjBhR1VnWm05eWQyRnlaQ0J6'
    'WTJGdUlIVnphVzVuQ2lBZ0lDQXZMeURpaUtzb1F5OVFLSFFwS1dSMElEMGdReXBVWDNKdmR5'
    'OG9VRjlsYm1RdFVGOXpkR0Z5ZENrZ0tpQnNiaWhRWDJWdVpDOVFYM04wWVhKMEtTQm1iM0ln'
    'WldGamFBb2dJQ0FnTHk4Z2MyVm5iV1Z1ZEN3Z2NHeDFjeUJoSUhCaGNuUnBZV3d0Y205M0lI'
    'UmhhV3d2YUdWaFpDQm1iM0lnZEhKcFoyZGxjaUJoYm1RZ1kzVnljbVZ1ZENCeWIzZHpMZ29n'
    'SUNBZ0x5OGdRMjl6ZERvZ2ZqRXdJR1Y0ZEhKaElHOXdjeUJ3WlhJZ1ptOXlkMkZ5WkMxelky'
    'RnVJSEp2ZHl3Z2JtOGdaWGgwY21FZ2RHVjRkSFZ5WlNCbVpYUmphR1Z6TGdvZ0lDQWdabXh2'
    'WVhRZ1gyWlRZVzF3YkdWUWIzTkJZMk1nUFNBd0xqQTdDZ29nSUNBZ2FXNTBJRjl3WTNRZ1BT'
    'QnBiblFvY0c5ekxuUnBZMnNwT3dvZ0lDQWdUbTkwWlNCZmNHTnlJRDBnWjJWMFRtOTBaU2h3'
    'YjNNdWMyOXVaMUJ2Y3l3Z2NHOXpMbkp2ZHl3Z1kyZ3BPd29LSUNBZ0lDOHZJT0tVZ09LVWdD'
    'QkRiMjFpYVc1bFpDQm1iM0ozWVhKa0lITmpZVzQ2SUhKbFluVnBiR1FnY0dsMFkyZ2dRVTVF'
    'SUhadmJIVnRaU0JtY205dElIUnlhV2RuWlhJZ2RHOGdZM1Z5Y21WdWRDRGlsSURpbElBS0lD'
    'QWdJR1pzYjJGMElHVm1abVZqZEdsMlpWQmxjbWx2WkNBOUlHWnNiMkYwS0hSeWFXZE9iM1Js'
    'TG5CbGNtbHZaQ2s3Q2lBZ0lDQm1iRzloZENCMFlYSm5aWFJRWlhKcGIyUWdJQ0FnUFNCbWJH'
    'OWhkQ2gwY21sblRtOTBaUzV3WlhKcGIyUXBPd29LSUNBZ0lDOHZJT0tVZ09LVWdDQldiMngx'
    'YldVZ2FXNXBkR2xoYkdsNllYUnBiMjRnS0ZCVUlIQmxjbWx2WkMxdmJteDVMWEpsZEhKcFoy'
    'ZGxjaUJ4ZFdseWF5a2c0cFNBNHBTQUNpQWdJQ0F2THlCUVZDQnpaVzFoYm5ScFkzTTZJR0Vn'
    'Y205M0lIZHBkR2dnY0dWeWFXOWtJRDRnTUNCaWRYUWdUazhnYVc1emRISjFiV1Z1ZENCdWRX'
    'MWlaWElnYVhNZ1lRb2dJQ0FnTHk4Z2NtVjBjbWxuWjJWeUlIUm9ZWFFnVWtWVFZFRlNWRk1n'
    'ZEdobElITmhiWEJzWlNCaGRDQnZabVp6WlhRZ01DQkNWVlFnUzBWRlVGTWdkR2hsSUhCeWFX'
    'OXlDaUFnSUNBdkx5QjJiMngxYldVZzRvQ1VJR2wwSjNNZ2RHaGxJSE5oYldVZ2FXNXpkSEox'
    'YldWdWRDQmlaV2x1WnlCeVpTMXdiR0Y1WldRc0lHNXZkQ0JoSUdaeVpYTm9JR3hoZEdOb0xn'
    'b2dJQ0FnTHk4S0lDQWdJQzh2SUZSb1pTQmlkVzVrYkdWa0lIUnlhV2RuWlhJdFptbHVaR1Z5'
    'SUdKaFkydDBjbUZqYTNNZ1lIUnlhV2RPYjNSbExtbHVjM1J5ZFcxbGJuUmdJR1p5YjIwZ1lR'
    'b2dJQ0FnTHk4Z2NISnBiM0lnY205M0lHWnZjaUJ6WVcxd2JHVWdiRzl2YTNWd0xDQmlkWFFn'
    'ZEdobElFOVNTVWRKVGtGTUlIUnlhV2RuWlhJZ1kyVnNiQ0IwWld4c2N5QjFjd29nSUNBZ0x5'
    'OGdkMmhsZEdobGNpQlFWQ0IzYjNWc1pDQmtieUJoSUhadmJDQnlaWE5sZEM0Z1NXWWdkR2hs'
    'SUc5eWFXZHBibUZzSUdObGJHd2dhR0ZrSUc1dklHbHVjM1FzQ2lBZ0lDQXZMeUIzWlNCdVpX'
    'VmtJSFJ2SUhKbFkyOXVjM1J5ZFdOMElIUm9aU0IyYjJ4MWJXVWdkR2hoZENCM1lYTWdhVzRn'
    'WldabVpXTjBJR3AxYzNRZ1ltVm1iM0psQ2lBZ0lDQXZMeUIwYUdseklIQmxjbWx2WkMxdmJt'
    'eDVJSEpsZEhKcFoyZGxjaUJpZVNCM1lXeHJhVzVuSUdKaFkyc2dkRzhnZEdobElHeGhjM1Fn'
    'YVc1emRDMWlaV0Z5YVc1bkNpQWdJQ0F2THlCeWIzY2dZVzVrSUdadmNuZGhjbVF0YzJOaGJt'
    'NXBibWNnWldabVpXTjBjeTRLSUNBZ0lDOHZDaUFnSUNBdkx5QlVhR2x6SUhkaGN5Qm1iM1Z1'
    'WkNCaWVTQndaWEl0Y205M0lHVnVaWEpuZVNCamIyMXdZWEpsSUdGbllXbHVjM1FnZUcxd0lH'
    'OXVJSE52Ym1jdGNHOXpJREUwQ2lBZ0lDQXZMeUFvY0dGMGRHVnliaUF6TkNrZ1kyZ3pPaUJ3'
    'WlhKcGIyUXRiMjVzZVNCeVpYUnlhV2RuWlhKeklHSmxkSGRsWlc0Z1EzaDRMWE5sZENCeWIz'
    'ZHpJSGRsY21VS0lDQWdJQzh2SUhCc1lYbHBibWNnWVhRZ1puVnNiQ0IyYjJ4MWJXVWdLSDR3'
    'TGpBNUlGSk5VeWtnYVc1emRHVmhaQ0J2WmlCMGFHVWdaR2x0YldWa0lHVmphRzhnYkdWMlpX'
    'd0tJQ0FnSUM4dklDaCtNQzR3TXlCU1RWTXBMaUJXYjJ3dGNISmxjMlZ5ZG1VZ1kzVjBjeUIw'
    'YjNSaGJDQmphRE1nVWsxVElHVnljbTl5SUhaeklIaHRjQ0JpZVNBeU1NT1hMZ29nSUNBZ0x5'
    'OGc0cFNBNHBTQUlGWnZiSFZ0WlNCemJXOXZkR2hwYm1jZ2MzbHpkR1Z0SU9LVWdPS1VnT0tV'
    'Z09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdP'
    'S1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tV'
    'Z09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ0FvZ0lD'
    'QWdMeThnUjJWdVpYSmhiR2w2WldRZ05qUXRjMkZ0Y0d4bElISmhiWEFnYjI0Z1JWWkZVbGtn'
    'ZG05c2RXMWxJR05vWVc1blpTQW9RM2g0TENCRlFYZ3NJRVZDZUN3Z1FYaDRDaUFnSUNBdkx5'
    'QjBhV05ySUdKdmRXNWtZWEpwWlhNc0lIQnNkWE1nYVc1b1pYSnBkR1ZrSUhaaGJIVmxjeUJo'
    'WTNKdmMzTWdabTl5ZDJGeVpDMXpZMkZ1SUhKdmQzTXBMZ29nSUNBZ0x5OGdVbVZ3YkdGalpY'
    'TWdNamdnYVc1c2FXNWxJR0IyYjJ4MWJXVWdQU0JZWUNCdGRYUmhkR2x2Ym5NZ2QybDBhQ0Jo'
    'SUhWdWFXWnZjbTBnY0dGMGRHVnliaUIwYUdGMENpQWdJQ0F2THlCd2NtVnpaWEoyWlhNZ2RH'
    'aGxJRzF2YzNRZ2NtVmpaVzUwSUdOb1lXNW5aU0JoYm1RZ2NtRnRjSE1nWVhRZ2IzVjBjSFYw'
    'SUhScGJXVXVJRmRwZEdodmRYUWdkR2hwY3dvZ0lDQWdMeThnY21GdGNDd2daV0ZqYUNCMmIy'
    'eDFiV1VnWldabVpXTjBJSEJ5YjJSMVkyVnpJR0VnYzJsdVoyeGxMWE5oYlhCc1pTQnpkR1Z3'
    'SUdsdUlHRnRjR3hwZEhWa1pTd0tJQ0FnSUM4dklHRnVaQ0I1YjNVZ2FHVmhjaUJwZENCaGN5'
    'QmhJSE5vWVhKd0lHTnNhV05ySU9LQWxDQndZWEowYVdOMWJHRnliSGtnWW1Ga0lHOXVJRU40'
    'ZUMxb1pXRjJlUW9nSUNBZ0x5OGdjR0YwZEdWeWJuTWdLSEp2ZDNNZ05TMHpNQ0J2WmlCd1lY'
    'UjBaWEp1SURBc0lHRnNiQ0J2WmlCd1lYUjBaWEp1SURFM0xDQmxkR011S1M0S0lDQWdJQzh2'
    'Q2lBZ0lDQXZMeUJVZDI4Z2MzUmhkR1VnZG1Gc2RXVnpJSEJzZFhNZ1lTQjBhV05ySUhOMFlX'
    'MXdPZ29nSUNBZ0x5OGdJQ0JmZG05c1VISmxkaUE5SUhSb1pTQjJZV3gxWlNCQ1JVWlBVa1Vn'
    'ZEdobElHMXZjM1FnY21WalpXNTBJR05vWVc1blpRb2dJQ0FnTHk4Z0lDQmZkbTlzUTNWeWNp'
    'QTlJSFJvWlNCMllXeDFaU0JCUmxSRlVpQW9ZM1Z5Y21WdWRDQm5jbTkxYm1RZ2RISjFkR2dw'
    'Q2lBZ0lDQXZMeUFnSUY5MmIyeERhR0Z1WjJWQmRGUnBZMnRHSUQwZ1oyeHZZbUZzTFhScFky'
    'c3RabXh2WVhRZ1lYUWdkMmhwWTJnZ2RHaGxJR05vWVc1blpTQm9ZWEJ3Wlc1bFpBb2dJQ0Fn'
    'THk4S0lDQWdJQzh2SUVGMElHOTFkSEIxZENCMGFXMWxPZ29nSUNBZ0x5OGdJQ0IyVW1GdGND'
    'QWdQU0JqYkdGdGNDZ29jRzl6TG5ScFkydEdJQzBnWDNadmJFTm9ZVzVuWlVGMFZHbGphMFlw'
    'SUNvZ1UwRk5VRjlRUlZKZlZFbERTeUF2SURZMExDQXdMQ0F4S1FvZ0lDQWdMeThnSUNCbFpt'
    'WldiMndnUFNCdGFYZ29YM1p2YkZCeVpYWXNJRjkyYjJ4RGRYSnlMQ0IyVW1GdGNDa2dLeUIw'
    'Y21WdGIyeHZSR1ZzZEdFS0lDQWdJQzh2Q2lBZ0lDQXZMeUJVZDI4Z2FHVnNjR1Z5SUcxaFkz'
    'SnZjem9LSUNBZ0lDOHZJQ0FnVms5TVgwbE9TVlFvVmlrZ0lDQWc0b0NVSUhObGRDQmliM1Jv'
    'SUhCeVpYWWdZVzVrSUdOMWNuSWdkRzhnVml3Z2JtOGdkSEpoYm5OcGRHbHZiZ29nSUNBZ0x5'
    'OGdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ2gxYzJWa0lHRjBJSFJ5YVdkblpYSWdhVzVw'
    'ZERzZ1pYaHBjM1JwYm1jZ05qUXRjMkZ0Y0d4bElHQmtaV05zYVdOcllBb2dJQ0FnTHk4Z0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHWmhZM1J2Y2lCb1lXNWtiR1Z6SUhSb1pTQjBjbWxu'
    'WjJWeUlHWmhaR1V0YVc0cExnb2dJQ0FnTHk4Z0lDQldUMHhmVTBWVUtGWXNJRlFwSUNEaWdK'
    'UWdjSEp2Ylc5MFpTQmpkWEp5NG9hU2NISmxkaXdnYzJWMElHTjFjbklnZEc4Z1Zpd2djM1Jo'
    'YlhBZ2RHbGpheUJVQ2lBZ0lDQXZMeUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnS0hWelpX'
    'UWdZWFFnWlhabGNua2dhVzR0Y0d4aGVTQjJiMngxYldVZ1kyaGhibWRsS1M0S0lDQWdJQzh2'
    'Q2lBZ0lDQXZMeUJVYUdVZ2JXRmpjbTl6SUhWelpTQmhJR052YlcxaExXVjRjSEpsYzNOcGIy'
    'NGdZbTlrZVNCemJ5QjBhR1Y1SUdWNGNHRnVaQ0JqYkdWaGJteDVJR2x1YzJsa1pRb2dJQ0Fn'
    'THk4Z1lXNTVJRWRNVTB3Z2MzUmhkR1Z0Wlc1MElHTnZiblJsZUhRZ0tHbG1MMlZzYzJVZ2Qy'
    'bDBhRzkxZENCaWNtRmpaWE1zSUdWMFl5NHBMZ29nSUNBZ2FXNTBJQ0FnWDNadmJGQnlaWFln'
    'UFNBd093b2dJQ0FnYVc1MElDQWdYM1p2YkVOMWNuSWdQU0F3T3dvZ0lDQWdabXh2WVhRZ1gz'
    'WnZiRU5vWVc1blpVRjBWR2xqYTBZZ1BTQXRNV1U1T3lBZ0x5OGdabUZ5TFhCaGMzUWdjMlZ1'
    'ZEdsdVpXdzZJSEpoYlhBZ1puVnNiSGtnWTI5dGNHeGxkR1VLSUNBZ0lHWnNiMkYwSUY5MGNt'
    'VnRiMnh2UkdWc2RHRWdJQ0FnSUQwZ01DNHdPeUFnSUM4dklIUnlaVzF2Ykc4Z1lYQndiR2xs'
    'Y3lCaGRDQnZkWFJ3ZFhRc0lHNXZkQ0IyYVdFZ1ZrOU1YMU5GVkFvS0lDQWdJQ05rWldacGJt'
    'VWdWazlNWDBsT1NWUW9WaWtnSUNBZ0tGOTJiMnhRY21WMklEMGdLRllwTENCZmRtOXNRM1Z5'
    'Y2lBOUlGOTJiMnhRY21WMktRb2dJQ0FnSTJSbFptbHVaU0JXVDB4ZlUwVlVLRllzSUZRcElD'
    'QW9YM1p2YkZCeVpYWWdQU0JmZG05c1EzVnljaXdnWDNadmJFTjFjbklnUFNBb1Zpa3NJRjky'
    'YjJ4RGFHRnVaMlZCZEZScFkydEdJRDBnS0ZRcEtRb0tJQ0FnSUM4dklGQnlaUzFqYjIxd2RY'
    'UmxaQ0IwYVdOcklHOW1JSFJvWlNCMGNtbG5aMlZ5SUhKdmR5ZHpJR1pwY25OMElIUnBZMnN1'
    'SUZWelpXUWdZWE1nZEdobElHTm9ZVzVuWlFvZ0lDQWdMeThnYzNSaGJYQWdabTl5SUdGc2JD'
    'QjBjbWxuWjJWeUxYSnZkeUIyYjJ3Z1pXWm1aV04wY3k0S0lDQWdJR1pzYjJGMElGOTBjbWxu'
    'WjJWeVZHbGphMFlnUFNCbWJHOWhkQ2htWlhSamFGUnBZMnNvY0dGMFZHbGphMDltWm5ObGRG'
    'dDBjbWxuVUdGMFhTQXJJQ2gwY21sblVtOTNJQzBnY0dGMFUzUmhjblJTYjNkYmRISnBaMUJo'
    'ZEYwcEtTazdDZ29nSUNBZ1RtOTBaU0JmZEhKcFowTmxiR3hQY21sbklEMGdaMlYwVG05MFpT'
    'aDBjbWxuVUdGMExDQjBjbWxuVW05M0xDQmphQ2s3Q2lBZ0lDQnBaaUFvWDNSeWFXZERaV3hz'
    'VDNKcFp5NXBibk4wY25WdFpXNTBJRDRnTUNrZ2V3b2dJQ0FnSUNBZ0lGWlBURjlKVGtsVUtI'
    'TnRjQzUyYjJ4MWJXVXBPeUFnTHk4Z1VtVmhiQ0JwYm5OMGNuVnRaVzUwSUd4aGRHTm9JT0tB'
    'bENCa1pXTnNhV05ySUdoaGJtUnNaWE1nWm1Ga1pRb2dJQ0FnZlNCbGJITmxJSHNLSUNBZ0lD'
    'QWdJQ0F2THlCUVpYSnBiMlF0YjI1c2VTQnlaWFJ5YVdkblpYSWc0b0NVSUdacGJtUWdkR2hs'
    'SUd4aGMzUWdhVzV6ZEMxaVpXRnlhVzVuSUhKdmR5d2dhVzVwZENCbWNtOXRDaUFnSUNBZ0lD'
    'QWdMeThnYVhSeklHbHVjM1J5ZFcxbGJuUW5jeUJrWldaaGRXeDBJSFp2YkhWdFpTd2dkR2hs'
    'YmlCbWIzSjNZWEprTFhOallXNGdaV1ptWldOMGN5QjFjQ0IwYndvZ0lDQWdJQ0FnSUM4dklD'
    'aGlkWFFnYm05MElHbHVZMngxWkdsdVp5a2dkSEpwWjFKdmR5NGdRbTkxYm1SbFpDQnpZMkZ1'
    'T2lBek1pQnliM2R6SUdKaFkydDNZWEprQ2lBZ0lDQWdJQ0FnTHk4Z1kyOTJaWEp6SUhSb1pT'
    'QjJZWE4wSUcxaGFtOXlhWFI1SUc5bUlHMTFjMmxqWVd3Z1kyRnpaWE11Q2lBZ0lDQWdJQ0Fn'
    'YVc1MElGOXBibk4wVW05M0lEMGdkSEpwWjFKdmR5d2dYMmx1YzNSUVlYUWdQU0IwY21sblVH'
    'RjBPd29nSUNBZ0lDQWdJR0p2YjJ3Z1gyWnZkVzVrU1c1emRFeGhkR05vSUQwZ1ptRnNjMlU3'
    'Q2lBZ0lDQWdJQ0FnZXdvZ0lDQWdJQ0FnSUNBZ0lDQnBiblFnYzFJZ1BTQjBjbWxuVW05M0xD'
    'QnpVQ0E5SUhSeWFXZFFZWFE3Q2lBZ0lDQWdJQ0FnSUNBZ0lHWnZjaUFvYVc1MElHeGlJRDBn'
    'TVRzZ2JHSWdQQ0F6TWpzZ2JHSXJLeWtnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYzFJdExU'
    'c0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2h6VWlBOElEQXBJSHNLSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNCcFppQW9jMUFnUGlBd0tTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lITlFMUzA3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJSE5T'
    'SUQwZ2NHRjBVM1JoY25SU2IzZGJjMUJkSUNzZ0tIQmhkRkp2ZDA5bVpuTmxkRnR6VUNzeFhT'
    'QXRJSEJoZEZKdmQwOW1abk5sZEZ0elVGMHBJQzBnTVRzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQjlJR1ZzYzJVZ2V5QmljbVZoYXpzZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2ZR'
    'b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1RtOTBaU0J3YmlBOUlHZGxkRTV2ZEdVb2MxQXNJSE5T'
    'TENCamFDazdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvY0c0dWFXNXpkSEoxYldWdWRD'
    'QStJREFwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmZhVzV6ZEZKdmR5QTlJSE5T'
    'T3lCZmFXNXpkRkJoZENBOUlITlFPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUY5bWIz'
    'VnVaRWx1YzNSTVlYUmphQ0E5SUhSeWRXVTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'WW5KbFlXczdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdJQ0FnSUgwS0lD'
    'QWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ2FXWWdLQ0ZmWm05MWJtUkpibk4wVEdGMFkyZ3BJSHNL'
    'SUNBZ0lDQWdJQ0FnSUNBZ1ZrOU1YMGxPU1ZRb2MyMXdMblp2YkhWdFpTazdJQ0F2THlCT2J5'
    'QnBibk4wSUd4aGRHTm9JT0tBbENCa1pXTnNhV05ySUdoaGJtUnNaWE1nWm1Ga1pRb2dJQ0Fn'
    'SUNBZ0lIMGdaV3h6WlNCN0NpQWdJQ0FnSUNBZ0lDQWdJRTV2ZEdVZ1gyeGhkR05vVG05MFpT'
    'QTlJR2RsZEU1dmRHVW9YMmx1YzNSUVlYUXNJRjlwYm5OMFVtOTNMQ0JqYUNrN0NpQWdJQ0Fn'
    'SUNBZ0lDQWdJRk5oYlhCc1pVbHVabThnWDJ4aGRHTm9VMjF3SUQwZ2MyRnRjR3hsYzF0ZmJH'
    'RjBZMmhPYjNSbExtbHVjM1J5ZFcxbGJuUWdMU0F4WFRzS0lDQWdJQ0FnSUNBZ0lDQWdhVzUw'
    'SUY5c1lYUmphRk5uY2lBOUlIQmhkRlJwWTJ0UFptWnpaWFJiWDJsdWMzUlFZWFJkSUNzZ0tG'
    'OXBibk4wVW05M0lDMGdjR0YwVTNSaGNuUlNiM2RiWDJsdWMzUlFZWFJkS1RzS0lDQWdJQ0Fn'
    'SUNBZ0lDQWdabXh2WVhRZ1gyeGhkR05vVkdsamEwWWdQU0JtYkc5aGRDaG1aWFJqYUZScFky'
    'c29YMnhoZEdOb1UyZHlLU2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lGWlBURjlKVGtsVUtGOXNZWFJq'
    'YUZOdGNDNTJiMngxYldVcE95QWdMeThnVW1WamIyNXpkSEoxWTNScGIyNGdZbUZ6Wld4cGJt'
    'VWc0b0NVSUc1dklIUnlZVzV6YVhScGIyNEtJQ0FnSUNBZ0lDQWdJQ0FnTHk4Z1FYQndiSGtn'
    'YkdGMFkyZ2djbTkzSjNNZ2RHbGpheTB3SUhadmJDQmxabVpsWTNSeklDaDBjbUZ1YzJsMGFX'
    'OXVjeUJ6ZEdGdGNHVmtJR0YwSUhKdmR5QjBhV05yS1M0S0lDQWdJQ0FnSUNBZ0lDQWdhV1ln'
    'S0Y5c1lYUmphRTV2ZEdVdVpXWm1aV04wSUQwOUlEQjRReWtnVms5TVgxTkZWQ2h0YVc0b1gy'
    'eGhkR05vVG05MFpTNXdZWEpoYlN3Z05qUXBMQ0JmYkdGMFkyaFVhV05yUmlrN0NpQWdJQ0Fn'
    'SUNBZ0lDQWdJR1ZzYzJVZ2FXWWdLRjlzWVhSamFFNXZkR1V1WldabVpXTjBJRDA5SURCNFJT'
    'a2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5bGN5QTlJQ2hmYkdGMFkyaE9iM1Js'
    'TG5CaGNtRnRJRDQrSURRcElDWWdNSGhHT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElG'
    'OWxkaUE5SUNCZmJHRjBZMmhPYjNSbExuQmhjbUZ0SUNBZ0lDQWdJQ1lnTUhoR093b2dJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjlsY3lBOVBTQXdlRUVwSUNBZ0lDQWdWazlNWDFORlZD'
    'aGpiR0Z0Y0NoZmRtOXNRM1Z5Y2lBcklGOWxkaXdnTUN3Z05qUXBMQ0JmYkdGMFkyaFVhV05y'
    'UmlrN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCbGJITmxJR2xtSUNoZlpYTWdQVDBnTUhoQ0tT'
    'QldUMHhmVTBWVUtHTnNZVzF3S0Y5MmIyeERkWEp5SUMwZ1gyVjJMQ0F3TENBMk5Da3NJRjlz'
    'WVhSamFGUnBZMnRHS1RzS0lDQWdJQ0FnSUNBZ0lDQWdmUW9nSUNBZ0lDQWdJQ0FnSUNBdkx5'
    'Qk1ZWFJqYUNCeWIzY25jeUJ3WlhJdGRHbGpheUJ6Ykdsa1pTQW9RWGg0TENBMmVIZ3NJRFY0'
    'ZUNCMmIyeDFiV1VnY0dGeWRDa2diM1psY2lCbWRXeHNJSEp2ZHk0S0lDQWdJQ0FnSUNBZ0lD'
    'QWdhV1lnS0Y5c1lYUmphRTV2ZEdVdVpXWm1aV04wSUQwOUlEQjRRU0I4ZkNCZmJHRjBZMmhP'
    'YjNSbExtVm1abVZqZENBOVBTQXdlRFlnZkh3S0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUY5c1lY'
    'UmphRTV2ZEdVdVpXWm1aV04wSUQwOUlEQjROU2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'YVc1MElGOTJkU0E5SUNoZmJHRjBZMmhPYjNSbExuQmhjbUZ0UGo0MEtTWXdlRVk3Q2lBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0JwYm5RZ1gzWmtJRDBnSUY5c1lYUmphRTV2ZEdVdWNHRnlZVzBn'
    'SUNBZ0pqQjRSanNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2x1ZENCZmMzUmxjQ0E5SUNoZmRu'
    'VStNQ2tnUHlCZmRuVWdPaUF0WDNaa093b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjlt'
    'ZEV3Z0lEMGdabVYwWTJoVWFXTnJLRjlzWVhSamFGTm5jaUFySURFcElDMGdabVYwWTJoVWFX'
    'TnJLRjlzWVhSamFGTm5jaWtnTFNBeE93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1ZrOU1YMU5G'
    'VkNoamJHRnRjQ2hmZG05c1EzVnljaUFySUY5emRHVndJQ29nWDJaMFRDd2dNQ3dnTmpRcExD'
    'QmZiR0YwWTJoVWFXTnJSaWs3Q2lBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0Fn'
    'THk4Z1JtOXlkMkZ5WkMxelkyRnVJR1ZtWm1WamRITWdiMjRnY205M2N5QmZhVzV6ZEZKdmR5'
    'c3hJQzR1TGlCMGNtbG5VbTkzTFRFc0lIZGhiR3RwYm1jS0lDQWdJQ0FnSUNBZ0lDQWdMeThn'
    'ZEdoeWIzVm5hQ0JoYm5rZ2FXNTBaWEp0WldScFlYUmxJSEJsY21sdlpDMXZibXg1SUhKbGRI'
    'SnBaMmRsY25NZ2QybDBhRzkxZENCeVpYTmxkSFJwYm1jdUNpQWdJQ0FnSUNBZ0lDQWdJR2x1'
    'ZENCZmRtWndJRDBnWDJsdWMzUlFZWFFzSUY5MlpuSWdQU0JmYVc1emRGSnZkeUFySURFN0Np'
    'QWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZmRtWnlJRDQ5SUhCaGRGTjBZWEowVW05M1cxOTJabkJk'
    'SUNzZ0tIQmhkRkp2ZDA5bVpuTmxkRnRmZG1ad0t6RmRJQzBnY0dGMFVtOTNUMlptYzJWMFcx'
    'OTJabkJkS1NrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1gzWm1jQ3NyT3lCZmRtWnlJRDBn'
    'S0Y5MlpuQWdQQ0JUVDA1SFgweEZUa2RVU0NrZ1B5QndZWFJUZEdGeWRGSnZkMXRmZG1ad1hT'
    'QTZJREE3Q2lBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnWm05eUlDaHBiblFn'
    'WDNacElEMGdNRHNnWDNacElEd2dOalE3SUY5MmFTc3JLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0JwWmlBb1gzWm1jQ0ErSUhSeWFXZFFZWFFnZkh3Z0tGOTJabkFnUFQwZ2RISnBaMUJo'
    'ZENBbUppQmZkbVp5SUQ0OUlIUnlhV2RTYjNjcEtTQmljbVZoYXpzS0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUdsbUlDaGZkbVp3SUQ0OUlGTlBUa2RmVEVWT1IxUklLU0JpY21WaGF6c0tJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmZG1aeUlENDlJSEJoZEZOMFlYSjBVbTkzVzE5Mlpu'
    'QmRJQ3NnS0hCaGRGSnZkMDltWm5ObGRGdGZkbVp3S3pGZElDMGdjR0YwVW05M1QyWm1jMlYw'
    'VzE5MlpuQmRLU2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lGOTJabkFyS3pzZ1gz'
    'Wm1jaUE5SUNoZmRtWndJRHdnVTA5T1IxOU1SVTVIVkVncElEOGdjR0YwVTNSaGNuUlNiM2Ri'
    'WDNabWNGMGdPaUF3T3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHTnZiblJwYm5WbE93'
    'b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1RtOTBaU0Jm'
    'ZG00Z1BTQm5aWFJPYjNSbEtGOTJabkFzSUY5MlpuSXNJR05vS1RzS0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUdsdWRDQmZjMmR5VmlBOUlIQmhkRlJwWTJ0UFptWnpaWFJiWDNabWNGMGdLeUFv'
    'WDNabWNpQXRJSEJoZEZOMFlYSjBVbTkzVzE5MlpuQmRLVHNLSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJR2x1ZENCZlpuUldJQ0E5SUdabGRHTm9WR2xqYXloZmMyZHlWaUFySURFcElDMGdabVYw'
    'WTJoVWFXTnJLRjl6WjNKV0tTQXRJREU3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JtYkc5aGRD'
    'QmZkbFJwWTJ0R0lEMGdabXh2WVhRb1ptVjBZMmhVYVdOcktGOXpaM0pXS1NrN0NpQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNCcFppQW9YM1p1TG1WbVptVmpkQ0E5UFNBd2VFTXBJRlpQVEY5VFJW'
    'UW9iV2x1S0Y5MmJpNXdZWEpoYlN3Z05qUXBMQ0JmZGxScFkydEdLVHNLSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJR1ZzYzJVZ2FXWWdLRjkyYmk1bFptWmxZM1FnUFQwZ01IaEJJSHg4SUY5MmJp'
    'NWxabVpsWTNRZ1BUMGdNSGcyS1NCN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhVzUw'
    'SUY5MmRTQTlJQ2hmZG00dWNHRnlZVzArUGpRcEpqQjRSaXdnWDNaa0lEMGdYM1p1TG5CaGNt'
    'RnRKakI0UmpzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQldUMHhmVTBWVUtHTnNZVzF3'
    'S0Y5MmIyeERkWEp5SUNzZ0tGOTJkVDR3UDE5MmRUb3RYM1prS1NBcUlGOW1kRllzSURBc0lE'
    'WTBLU3dnWDNaVWFXTnJSaWs3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0JsYkhObElHbG1JQ2hmZG00dVpXWm1aV04wSUQwOUlEQjRSU2tnZXdvZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0JmWlhNZ1BTQW9YM1p1TG5CaGNtRnRJRDQr'
    'SURRcElDWWdNSGhHT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0JmWlhZZ1BT'
    'QWdYM1p1TG5CaGNtRnRJQ0FnSUNBZ0lDWWdNSGhHT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lHbG1JQ2hmWlhNZ1BUMGdNSGhCS1NBZ0lDQWdJRlpQVEY5VFJWUW9ZMnhoYlhBb1gz'
    'WnZiRU4xY25JZ0t5QmZaWFlzSURBc0lEWTBLU3dnWDNaVWFXTnJSaWs3Q2lBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ1pXeHpaU0JwWmlBb1gyVnpJRDA5SURCNFFpa2dWazlNWDFORlZD'
    'aGpiR0Z0Y0NoZmRtOXNRM1Z5Y2lBdElGOWxkaXdnTUN3Z05qUXBMQ0JmZGxScFkydEdLVHNL'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZmRt'
    'NHVaV1ptWldOMElEMDlJREI0TlNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2x1'
    'ZENCZmRuVWdQU0FvWDNadUxuQmhjbUZ0UGo0MEtTWXdlRVlzSUY5MlpDQTlJRjkyYmk1d1lY'
    'SmhiU1l3ZUVZN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdWazlNWDFORlZDaGpiR0Z0'
    'Y0NoZmRtOXNRM1Z5Y2lBcklDaGZkblUrTUQ5ZmRuVTZMVjkyWkNrZ0tpQmZablJXTENBd0xD'
    'QTJOQ2tzSUY5MlZHbGphMFlwT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnWDNabWNpc3JPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5Mlpu'
    'SWdQajBnY0dGMFUzUmhjblJTYjNkYlgzWm1jRjBnS3lBb2NHRjBVbTkzVDJabWMyVjBXMTky'
    'Wm5Bck1WMGdMU0J3WVhSU2IzZFBabVp6WlhSYlgzWm1jRjBwS1NCN0NpQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdYM1ptY0Nzck95QmZkbVp5SUQwZ0tGOTJabkFnUENCVFQwNUhYMHhG'
    'VGtkVVNDa2dQeUJ3WVhSVGRHRnlkRkp2ZDF0ZmRtWndYU0E2SURBN0NpQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0I5Q2lBZ0lDQjlDZ29n'
    'SUNBZ0x5OGdWRzl1WlMxd2IzSjBZU0IwY21sbloyVnlJSEp2ZHpvZ2RHaHBjeUJ5YjNjZ1ky'
    'RnljbWxsY3lCaElETjRlQzgxZUhnZ2MyeHBaR1VnZEdGeVoyVjBMZ29nSUNBZ0x5OGdaV1pt'
    'WldOMGFYWmxVR1Z5YVc5a0lITjBZWGx6SUdGMElIUm9aU0J3Y21WMmFXOTFjeUIwY21sbloy'
    'VnlKM01nY0dWeWFXOWtJQ2hoYkhKbFlXUjVJSE5sZENCaFltOTJaUW9nSUNBZ0x5OGdabkp2'
    'YlNCMGNtbG5UbTkwWlM1d1pYSnBiMlFwT3lCemJHbGtaU0JoWTJOMWJYVnNZWFJsY3lCMGIz'
    'ZGhjbVFnZEc5dVpWTnNhV1JsVkdGeVoyVjBJRzkyWlhJZ2NtOTNjeTRLSUNBZ0lHbG1JQ2gw'
    'YjI1bFUyeHBaR1ZVWVhKblpYUWdQaUF3S1NCN0NpQWdJQ0FnSUNBZ2RHRnlaMlYwVUdWeWFX'
    'OWtJRDBnWm14dllYUW9kRzl1WlZOc2FXUmxWR0Z5WjJWMEtUc0tJQ0FnSUgwS0NpQWdJQ0F2'
    'THlCQmNIQnNlU0IwY21sbloyVnlMWEp2ZHlCbFptWmxZM1J6T2lCRGVIZ2dLSE5sZENCMmIy'
    'd3BMQ0JCZUhndk5uaDRJQ2gyYjJ3Z2MyeHBaR1VnY0dGeWRHbGhiQzltZFd4c0tTd0tJQ0Fn'
    'SUM4dklFVkJlQ0FvWm1sdVpTQjJiMndnZFhBZzRvQ1VJR2x1YzNSaGJuUXBMQ0JGUW5nZ0tH'
    'WnBibVVnZG05c0lHUnZkMjRnNG9DVUlHbHVjM1JoYm5RcExDQTFlSGdnS0hSdmJtVXJkbTlz'
    'SUhOc2FXUmxLUzRLSUNBZ0lDOHZJRUZzYkNCMGNtbG5aMlZ5TFhKdmR5QjJiMndnWTJoaGJt'
    'ZGxjeUJ6ZEdGdGNDQjBhR1VnWTJoaGJtZGxJSFJwWTJzZ1lYTWdYM1J5YVdkblpYSlVhV05y'
    'UmlCemJ5QjBhR1VLSUNBZ0lDOHZJRFkwTFhOaGJYQnNaU0J5WVcxd0lHTnZiWEJzWlhSbGN5'
    'QjNhWFJvYVc0Z2RHaGxJR1pwY25OMElINHhMalZ0Y3lCdlppQjBhR1VnZEhKcFoyZGxjaUJ5'
    'YjNjdUNpQWdJQ0JwWmlBb2RISnBaMDV2ZEdVdVpXWm1aV04wSUQwOUlEQjRReWtnZXdvZ0lD'
    'QWdJQ0FnSUZaUFRGOVRSVlFvYldsdUtIUnlhV2RPYjNSbExuQmhjbUZ0TENBMk5Da3NJRjkw'
    'Y21sbloyVnlWR2xqYTBZcE93b2dJQ0FnZlNCbGJITmxJR2xtSUNoMGNtbG5UbTkwWlM1MmIy'
    'eGZZMjlzSUQ0Z01Da2dld29nSUNBZ0lDQWdJQzh2SUVsVUlIWnZiQzFqYjJ4MWJXNGdiM1ps'
    'Y25KcFpHVTZJRzV2ZEdVdGRtOXNkVzFsSUhObGRDQnBiaUJ3WVhSMFpYSnVJSFp2YkhWdFpT'
    'QmpiMngxYlc0S0lDQWdJQ0FnSUNCV1QweGZVMFZVS0cxcGJpaDBjbWxuVG05MFpTNTJiMnhm'
    'WTI5c0xDQTJOQ2tzSUY5MGNtbG5aMlZ5VkdsamEwWXBPd29nSUNBZ2ZTQmxiSE5sSUdsbUlD'
    'aDBjbWxuVG05MFpTNWxabVpsWTNRZ1BUMGdNSGhGS1NCN0NpQWdJQ0FnSUNBZ0x5OGdSWGgw'
    'Wlc1a1pXUWdaV1ptWldOMGN6b2dSVUY0SUdacGJtVWdkbTlzSUhWd0xDQkZRbmdnWm1sdVpT'
    'QjJiMndnWkc5M2JpQW9hVzV6ZEdGdWRDQnZiaUIwYVdOcklEQXBDaUFnSUNBZ0lDQWdhVzUw'
    'SUY5bGN5QTlJQ2gwY21sblRtOTBaUzV3WVhKaGJTQStQaUEwS1NBbUlEQjRSanNLSUNBZ0lD'
    'QWdJQ0JwYm5RZ1gyVjJJRDBnSUhSeWFXZE9iM1JsTG5CaGNtRnRJQ0FnSUNBZ0lDWWdNSGhH'
    'T3dvZ0lDQWdJQ0FnSUdsbUlDaGZaWE1nUFQwZ01IaEJLU0FnSUNBZ0lGWlBURjlUUlZRb1ky'
    'eGhiWEFvWDNadmJFTjFjbklnS3lCZlpYWXNJREFzSURZMEtTd2dYM1J5YVdkblpYSlVhV05y'
    'UmlrN0NpQWdJQ0FnSUNBZ1pXeHpaU0JwWmlBb1gyVnpJRDA5SURCNFFpa2dWazlNWDFORlZD'
    'aGpiR0Z0Y0NoZmRtOXNRM1Z5Y2lBdElGOWxkaXdnTUN3Z05qUXBMQ0JmZEhKcFoyZGxjbFJw'
    'WTJ0R0tUc0tJQ0FnSUgwZ1pXeHpaU0JwWmlBb2RISnBaMDV2ZEdVdVpXWm1aV04wSUQwOUlE'
    'QjRRU0I4ZkNCMGNtbG5UbTkwWlM1bFptWmxZM1FnUFQwZ01IZzJJSHg4SUhSeWFXZE9iM1Js'
    'TG1WbVptVmpkQ0E5UFNBd2VEVXBJSHNLSUNBZ0lDQWdJQ0F2THlBd2VEVWdQU0IwYjI1bEsz'
    'WnZiQ0J6Ykdsa1pUb2djR2wwWTJnZ2FHRnVaR3hsWkNCaWVTQXdlRE10WlhGMWFYWmhiR1Z1'
    'ZENCaWJHOWpheXdnZG05c0lIQmhjbUZ0SUhOaGJXVWdZWE1nTUhoQkNpQWdJQ0FnSUNBZ2FX'
    'NTBJRjl6ZFNBOUlDaDBjbWxuVG05MFpTNXdZWEpoYlQ0K05Da21NSGhHTENCZmMyUWdQU0Iw'
    'Y21sblRtOTBaUzV3WVhKaGJTWXdlRVk3Q2lBZ0lDQWdJQ0FnYVc1MElGOXpkR1Z3SUQwZ0tG'
    'OXpkVDR3S1NBL0lGOXpkU0E2SUMxZmMyUTdDaUFnSUNBZ0lDQWdhV1lnS0hSeWFXZFFZWFFn'
    'UFQwZ2NHOXpMbk52Ym1kUWIzTWdKaVlnZEhKcFoxSnZkeUE5UFNCd2IzTXVjbTkzS1NCN0Np'
    'QWdJQ0FnSUNBZ0lDQWdJRlpQVEY5VFJWUW9ZMnhoYlhBb1gzWnZiRU4xY25JZ0t5QmZjM1Js'
    'Y0NBcUlGOXdZM1FzSURBc0lEWTBLU3dnWDNSeWFXZG5aWEpVYVdOclJpazdDaUFnSUNBZ0lD'
    'QWdmU0JsYkhObElIc0tJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOTBjeUE5SUdabGRHTm9WR2xq'
    'YXlod1lYUlVhV05yVDJabWMyVjBXM1J5YVdkUVlYUmRLeWgwY21sblVtOTNMWEJoZEZOMFlY'
    'SjBVbTkzVzNSeWFXZFFZWFJkS1NzeEtRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQzBn'
    'Wm1WMFkyaFVhV05yS0hCaGRGUnBZMnRQWm1aelpYUmJkSEpwWjFCaGRGMHJLSFJ5YVdkU2Iz'
    'Y3RjR0YwVTNSaGNuUlNiM2RiZEhKcFoxQmhkRjBwS1RzS0lDQWdJQ0FnSUNBZ0lDQWdWazlN'
    'WDFORlZDaGpiR0Z0Y0NoZmRtOXNRM1Z5Y2lBcklGOXpkR1Z3SUNvZ0tGOTBjeTB4S1N3Z01D'
    'd2dOalFwTENCZmRISnBaMmRsY2xScFkydEdLVHNLSUNBZ0lDQWdJQ0I5Q2lBZ0lDQjlDZ29n'
    'SUNBZ0x5OGc0cFNBNHBTQUlEQjRSVVFnYm05MFpTQmtaV3hoZVRvZ2MydHBjQ0J2ZFhSd2RY'
    'UWdabTl5SUdCd1lYSmhiV0FnZEdsamEzTWdZV1owWlhJZ2RISnBaMmRsY2lCeWIzY2c0cFNB'
    'NHBTQUNpQWdJQ0F2THlCVWFHVWdibTkwWlNCa2IyVnpiaWQwSUdGamRIVmhiR3g1SUhOMFlY'
    'SjBJSFZ1ZEdsc0lIUnBZMnNnWUhCaGNtRnRZQ0J2WmlCMGFHVWdkSEpwWjJkbGNpQnliM2N1'
    'Q2lBZ0lDQXZMeUJGWVhKc2FXVnlJSFJvWVc0Z2RHaGhkQ0RpaHBJZ2NtVjBkWEp1SUhOcGJH'
    'VnVZMlU3SUdWbVptVmpkR2wyWlNCbGJHRndjMlZrSUdseklISmxaSFZqWldRdUNpQWdJQ0Jw'
    'Ym5RZ1gyNXZkR1ZFWld4aGVWUnBZMnR6SUQwZ01Ec0tJQ0FnSUdsbUlDaDBjbWxuVG05MFpT'
    'NWxabVpsWTNRZ1BUMGdNSGhGSUNZbUlDZ29kSEpwWjA1dmRHVXVjR0Z5WVcwZ1BqNGdOQ2tn'
    'SmlBd2VFWXBJRDA5SURCNFJDa2dld29nSUNBZ0lDQWdJRjl1YjNSbFJHVnNZWGxVYVdOcmN5'
    'QTlJSFJ5YVdkT2IzUmxMbkJoY21GdElDWWdNSGhHT3dvZ0lDQWdJQ0FnSUdsbUlDaDBjbWxu'
    'VUdGMElEMDlJSEJ2Y3k1emIyNW5VRzl6SUNZbUlIUnlhV2RTYjNjZ1BUMGdjRzl6TG5KdmR5'
    'QW1KaUJmY0dOMElEd2dYMjV2ZEdWRVpXeGhlVlJwWTJ0ektRb2dJQ0FnSUNBZ0lDQWdJQ0J5'
    'WlhSMWNtNGdNQzR3T3lBZ0x5OGdZbVZtYjNKbElHUmxiR0Y1WldRZ2RISnBaMmRsY2dvZ0lD'
    'QWdJQ0FnSUM4dklFRm1kR1Z5SUdSbGJHRjVPaUJ6ZFdKMGNtRmpkQ0JrWld4aGVTQjBhV05y'
    'Y3lCbWNtOXRJR1ZzWVhCelpXUWdjMjhnZEdobElHNXZkR1VnYzNSaGNuUnpJR0YwSUdaeVpY'
    'Tm9JSFE5TUFvZ0lDQWdJQ0FnSUdWc1lYQnpaV1FnUFNCdFlYZ29NQzR3TENCbGJHRndjMlZr'
    'SUMwZ1pteHZZWFFvWDI1dmRHVkVaV3hoZVZScFkydHpLU0F2SUZSSlEwdFRYMUJGVWw5VFJV'
    'TXBPd29nSUNBZ2ZRb0tJQ0FnSUM4dklPS1VnT0tVZ0NBd2VEbDRlQ0J6WVcxd2JHVWdiMlpt'
    'YzJWMElDaDBjbWxuWjJWeUlISnZkeUJ2Ym14NUtUb2djM1JoY25RZ1lYUWdjR0Z5WVcwZ0tp'
    'QXlOVFlnYVc0Z2MyRnRjR3hsSUdSaGRHRWc0cFNBNHBTQUNpQWdJQ0JwYm5RZ1gzTmhiWEJz'
    'WlU5bVpuTmxkQ0E5SURBN0NpQWdJQ0JwWmlBb2RISnBaMDV2ZEdVdVpXWm1aV04wSUQwOUlE'
    'QjRPU0FtSmlCMGNtbG5UbTkwWlM1d1lYSmhiU0ErSURBcElIc0tJQ0FnSUNBZ0lDQmZjMkZ0'
    'Y0d4bFQyWm1jMlYwSUQwZ2RISnBaMDV2ZEdVdWNHRnlZVzBnS2lBeU5UWTdDaUFnSUNCOUNn'
    'b2dJQ0FnTHk4ZzRwU0E0cFNBSUZSeWFXZG5aWElnY205M0ozTWdjR2wwWTJnZ2MyeHBaR1Vn'
    'WldabVpXTjBjeUFvTVhoNEx6SjRlQ2tnNHBTQTRwU0FDaUFnSUNBdkx5QkpaaUIwYUdVZ2RI'
    'SnBaMmRsY2lCeWIzY2dZMkZ5Y21sbFpDQXhlSGdnS0hCdmNuUmhJSFZ3S1NCdmNpQXllSGdn'
    'S0hCdmNuUmhJR1J2ZDI0cExDQjBhRzl6WlFvZ0lDQWdMeThnYzJ4cFpHVnpJR2hoY0hCbGJp'
    'QnZiaUIwYVdOcmN5QXhMaTRvYzNCbFpXUXRNU2tnYjJZZ2RHaGxJSFJ5YVdkblpYSWdjbTkz'
    'TGlBZ1YyaGxiaUJ3YjNNZ2FYTUtJQ0FnSUM4dklGQkJVMVFnZEdobElIUnlhV2RuWlhJZ2Nt'
    'OTNMQ0JoYkd3Z0tITndaV1ZrTFRFcElHOW1JSFJvYjNObElIUnBZMnR6SUdoaGRtVWdZMjl0'
    'Y0d4bGRHVmtMZ29nSUNBZ0x5OGdWMmhsYmlCd2IzTWdhWE1nYjI0Z2RHaGxJSFJ5YVdkblpY'
    'SWdjbTkzSUdsMGMyVnNaaXdnZEdobElDSkRkWEp5Wlc1MElISnZkeUJ3WVhKMGFXRnNJSEJw'
    'ZEdOb0NpQWdJQ0F2THlCbFptWmxZM1FpSUdKc2IyTnJJR0psYkc5M0lHaGhibVJzWlhNZ2FY'
    'UWc0b0NVSUdSdmJpZDBJR1J2ZFdKc1pTMWhjSEJzZVNCb1pYSmxMZ29nSUNBZ2FXWWdLQ2gw'
    'Y21sblVHRjBJQ0U5SUhCdmN5NXpiMjVuVUc5eklIeDhJSFJ5YVdkU2IzY2dJVDBnY0c5ekxu'
    'SnZkeWtnSmlZS0lDQWdJQ0FnSUNBb2RISnBaMDV2ZEdVdVpXWm1aV04wSUQwOUlEQjRNU0I4'
    'ZkNCMGNtbG5UbTkwWlM1bFptWmxZM1FnUFQwZ01IZ3lLU0FtSmlCMGNtbG5UbTkwWlM1d1lY'
    'SmhiU0ErSURBcElIc0tJQ0FnSUNBZ0lDQnBiblFnWDNSeVUyZHlJRDBnY0dGMFZHbGphMDlt'
    'Wm5ObGRGdDBjbWxuVUdGMFhTQXJJQ2gwY21sblVtOTNJQzBnY0dGMFUzUmhjblJTYjNkYmRI'
    'SnBaMUJoZEYwcE93b2dJQ0FnSUNBZ0lHbHVkQ0JmZEhKVGNHUWdQU0JtWlhSamFGUnBZMnNv'
    'WDNSeVUyZHlJQ3NnTVNrZ0xTQm1aWFJqYUZScFkyc29YM1J5VTJkeUtUc0tJQ0FnSUNBZ0lD'
    'QnBiblFnWDNSeVZHbGphM01nUFNCZmRISlRjR1FnTFNBeE95QWdMeThnWVd4c0lIQnZjM1F0'
    'ZEdsamF5MHdJSFJwWTJ0eklHOW1JSFJ5YVdkblpYSWdjbTkzQ2lBZ0lDQWdJQ0FnYVdZZ0tI'
    'UnlhV2RPYjNSbExtVm1abVZqZENBOVBTQXdlREVwQ2lBZ0lDQWdJQ0FnSUNBZ0lHVm1abVZq'
    'ZEdsMlpWQmxjbWx2WkNBOUlHMWhlQ2d4TVRNdU1Dd2daV1ptWldOMGFYWmxVR1Z5YVc5a0lD'
    'MGdabXh2WVhRb2RISnBaMDV2ZEdVdWNHRnlZVzBnS2lCZmRISlVhV05yY3lrcE93b2dJQ0Fn'
    'SUNBZ0lHVnNjMlVnSUM4dklEQjRNZ29nSUNBZ0lDQWdJQ0FnSUNCbFptWmxZM1JwZG1WUVpY'
    'SnBiMlFnUFNCdGFXNG9PRFUyTGpBc0lHVm1abVZqZEdsMlpWQmxjbWx2WkNBcklHWnNiMkYw'
    'S0hSeWFXZE9iM1JsTG5CaGNtRnRJQ29nWDNSeVZHbGphM01wS1RzS0lDQWdJSDBLQ2lBZ0lD'
    'QXZMeUJVY21GamF5QnNZWE4wSUhSdmJtVXRjRzl5ZEdFZ2NtRjBaU0FvWm05eUlHVm1abVZq'
    'ZENBMUlIUnZJR2x1YUdWeWFYUXBMaUFnU1c1cGRHbGhiR2w2WldRZ1puSnZiUW9nSUNBZ0x5'
    'OGdkSEpwWjJkbGNpQnliM2NuY3lBemVIZ2djR0Z5WVcwN0lIVndaR0YwWldRZ1lua2dabTl5'
    'ZDJGeVpDQnpZMkZ1SUdGeklHbDBJSGRoYkd0eklIQmhjM1FnTTNoNElISnZkM011Q2lBZ0lD'
    'QnBiblFnWDJ4aGMzUlVVRkpoZEdVZ1BTQXdPd29nSUNBZ2FXWWdLSFJ5YVdkT2IzUmxMbVZt'
    'Wm1WamRDQTlQU0F3ZURNZ0ppWWdkSEpwWjA1dmRHVXVjR0Z5WVcwZ1BpQXdLU0JmYkdGemRG'
    'UlFVbUYwWlNBOUlIUnlhV2RPYjNSbExuQmhjbUZ0T3dvS0lDQWdJQzh2SUZSeWFXZG5aWEln'
    'Y205M0ozTWdkR0ZwYkNCamIyNTBjbWxpZFhScGIyNGdkRzhnWmxOaGJYQnNaVkJ2Y3lCMmFX'
    'RWdjR1Z5TFhScFkyc2dhVzUwWldkeVlYUnBiMjR1Q2lBZ0lDQXZMeUJRVkNCa2IyVnpJR1Jw'
    'YzJOeVpYUmxJSEJsY2kxMGFXTnJJSE5zYVdSbElIVndaR0YwWlhNc0lHNXZkQ0JqYjI1MGFX'
    'NTFiM1Z6SUhKaGJYQnpJT0tBbEFvZ0lDQWdMeThnWTI5dWRHbHVkVzkxY3kxeVlXMXdJR2x1'
    'ZEdWbmNtRnNjeUJrYVhabGNtZGxJR1p5YjIwZ2RISjFkR2dnWm05eUlHWmhjM1FnYzJ4cFpH'
    'VnpMaUJYWlNCc2IyOXdDaUFnSUNBdkx5QnZkbVZ5SUdWaFkyZ2dkR2xqYXlCdlppQjBhR1Vn'
    'ZEhKcFoyZGxjaUJ5YjNjZ1lXNWtJR0ZrWkNCRHc1ZGtkQzl3WlhKcGIyUmZZWFJmZEdsamF5'
    'NEtJQ0FnSUM4dklGUnBZMnNnTUNCelpXVnpJSFJ5YVdkT2IzUmxMbkJsY21sdlpDNGdWR2xq'
    'YTNNZ01TNHVLSE53WldWa0xURXBJSE5sWlNCcGJtTnlaVzFsYm5SaGJHeDVDaUFnSUNBdkx5'
    'QjFjR1JoZEdWa0lIQmxjbWx2WkhNZ2FXWWdNWGg0THpKNGVDQnBjeUJ3Y21WelpXNTBMZ29n'
    'SUNBZ2FXWWdLSFJ5YVdkUVlYUWdJVDBnY0c5ekxuTnZibWRRYjNNZ2ZId2dkSEpwWjFKdmR5'
    'QWhQU0J3YjNNdWNtOTNLU0I3Q2lBZ0lDQWdJQ0FnYVc1MElGOXpaM0pVY21sbklDQTlJSEJo'
    'ZEZScFkydFBabVp6WlhSYmRISnBaMUJoZEYwZ0t5QW9kSEpwWjFKdmR5QXRJSEJoZEZOMFlY'
    'SjBVbTkzVzNSeWFXZFFZWFJkS1RzS0lDQWdJQ0FnSUNCcGJuUWdYM1J5YVdkR2RXeHNJRDBn'
    'Wm1WMFkyaFVhV05yS0Y5elozSlVjbWxuSUNzZ01Ta2dMU0JtWlhSamFGUnBZMnNvWDNObmNs'
    'UnlhV2NwT3lBZ0x5OGdQU0J6Y0dWbFpBb2dJQ0FnSUNBZ0lHWnNiMkYwSUY5RFpsOTBjbWxu'
    'SUQwZ1l6UnpjR1ZsWkhOYmMyMXdMbVpwYm1WMGRXNWxJQ1lnTUhoR1hTQXFJRFF5T0M0d093'
    'b2dJQ0FnSUNBZ0lHWnNiMkYwSUY5a2RDQTlJREV1TUNBdklGUkpRMHRUWDFCRlVsOVRSVU03'
    'Q2lBZ0lDQWdJQ0FnWm14dllYUWdYMUIwSUQwZ1pteHZZWFFvZEhKcFowNXZkR1V1Y0dWeWFX'
    'OWtLVHNLSUNBZ0lDQWdJQ0F2THlCUVpYSXRkR2xqYXlCemJHbGtaU0J6ZEdWd0lHRnRiM1Z1'
    'ZENBb2MybG5ibVZrS1FvZ0lDQWdJQ0FnSUdsdWRDQmZjM1JsY0NBOUlEQTdDaUFnSUNBZ0lD'
    'QWdhV1lnS0hSeWFXZE9iM1JsTG1WbVptVmpkQ0E5UFNBd2VERXBJRjl6ZEdWd0lEMGdMWFJ5'
    'YVdkT2IzUmxMbkJoY21GdE93b2dJQ0FnSUNBZ0lHVnNjMlVnYVdZZ0tIUnlhV2RPYjNSbExt'
    'Vm1abVZqZENBOVBTQXdlRElwSUY5emRHVndJRDBnZEhKcFowNXZkR1V1Y0dGeVlXMDdDaUFn'
    'SUNBZ0lDQWdMeThnUVdOamRXMTFiR0YwWlNCd1pYSXRkR2xqYXpvZ2RHbGpheUF3SUNzZ2RH'
    'bGphM01nTVM0dUtITndaV1ZrTFRFcENpQWdJQ0FnSUNBZ0x5OGdRbTkxYm1RZ2RHOGdNekln'
    'Wm05eUlHTnZiWEJwYkdWeUlITmhabVYwZVRzZ2MzQmxaV1FnYVhNZzRvbWtJRE14SUdsdUlI'
    'QnlZV04wYVdObExnb2dJQ0FnSUNBZ0lHWnZjaUFvYVc1MElGOTBJRDBnTURzZ1gzUWdQQ0F6'
    'TWpzZ1gzUXJLeWtnZXdvZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvWDNRZ1BqMGdYM1J5YVdkR2RX'
    'eHNLU0JpY21WaGF6c0tJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOVFkQ0ErSURBdU1Da2dYMlpU'
    'WVcxd2JHVlFiM05CWTJNZ0t6MGdYME5tWDNSeWFXY2dLaUJmWkhRZ0x5QmZVSFE3Q2lBZ0lD'
    'QWdJQ0FnSUNBZ0lDOHZJRlZ3WkdGMFpTQndaWEpwYjJRZ1ptOXlJRzVsZUhRZ2RHbGpheUFv'
    'WTJ4aGJYQnpJRzFoZEdOb0lGQlVJSEpoYm1kbEtRb2dJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1gz'
    'TjBaWEFnSVQwZ01Da2dYMUIwSUQwZ1kyeGhiWEFvWDFCMElDc2dabXh2WVhRb1gzTjBaWEFw'
    'TENBeE1UTXVNQ3dnT0RVMkxqQXBPd29nSUNBZ0lDQWdJSDBLSUNBZ0lIMEtDaUFnSUNBdkx5'
    'QkdiM0ozWVhKa0lITmpZVzQ2SUhKdmQzTWdVMVJTU1VOVVRGa2dZbVYwZDJWbGJpQjBjbWxu'
    'WjJWeUlHRnVaQ0JqZFhKeVpXNTBDaUFnSUNCcFppQW9kSEpwWjFCaGRDQWhQU0J3YjNNdWMy'
    'OXVaMUJ2Y3lCOGZDQjBjbWxuVW05M0lDRTlJSEJ2Y3k1eWIzY3BJSHNLSUNBZ0lDQWdJQ0Jw'
    'Ym5RZ1gyWndJRDBnZEhKcFoxQmhkQ3dnWDJaeUlEMGdkSEpwWjFKdmR5QXJJREU3Q2lBZ0lD'
    'QWdJQ0FnYVdZZ0tGOW1jaUErUFNCd1lYUlRkR0Z5ZEZKdmQxdGZabkJkSUNzZ0tIQmhkRkp2'
    'ZDA5bVpuTmxkRnRmWm5Bck1WMGdMU0J3WVhSU2IzZFBabVp6WlhSYlgyWndYU2twSUhzS0lD'
    'QWdJQ0FnSUNBZ0lDQWdYMlp3S3lzN0lGOW1jaUE5SUNoZlpuQWdQQ0JUVDA1SFgweEZUa2RV'
    'U0NrZ1B5QndZWFJUZEdGeWRGSnZkMXRmWm5CZElEb2dNRHNLSUNBZ0lDQWdJQ0I5Q2lBZ0lD'
    'QWdJQ0FnWm05eUlDaHBiblFnWDJacElEMGdNRHNnWDJacElEd2dNVEk0T3lCZlpta3JLeWtn'
    'ZXdvZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvWDJad0lENGdjRzl6TG5OdmJtZFFiM01nZkh3Z0tG'
    'OW1jQ0E5UFNCd2IzTXVjMjl1WjFCdmN5QW1KaUJmWm5JZ1BqMGdjRzl6TG5KdmR5a3BJR0p5'
    'WldGck93b2dJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1gyWndJRDQ5SUZOUFRrZGZURVZPUjFSSUtT'
    'QmljbVZoYXpzS0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5bWNpQStQU0J3WVhSVGRHRnlkRkp2'
    'ZDF0ZlpuQmRJQ3NnS0hCaGRGSnZkMDltWm5ObGRGdGZabkFyTVYwZ0xTQndZWFJTYjNkUFpt'
    'WnpaWFJiWDJad1hTa3BJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJRjltY0Nzck95QmZabkln'
    'UFNBb1gyWndJRHdnVTA5T1IxOU1SVTVIVkVncElEOGdjR0YwVTNSaGNuUlNiM2RiWDJad1hT'
    'QTZJREE3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JqYjI1MGFXNTFaVHNLSUNBZ0lDQWdJQ0Fn'
    'SUNBZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0JPYjNSbElGOW1iaUE5SUdkbGRFNXZkR1VvWDJad0xD'
    'QmZabklzSUdOb0tUc0tJQ0FnSUNBZ0lDQWdJQ0FnTHk4Z1FVNVpJSEp2ZHlCM2FYUm9JSEJs'
    'Y21sdlpDQStJREFnWVc1a0lHVm1abVZqZENCdWIzUWdNeTgxSUdseklHRWdjbVZoYkNCU1JW'
    'UlNTVWRIUlZJS0lDQWdJQ0FnSUNBZ0lDQWdMeThnZEdoaGRDQmxibVJ6SUhSb1pTQm1iM0oz'
    'WVhKa0lITmpZVzRnS0c1bGVIUWdaMlYwUTJoaGJtNWxiRTkxZEhCMWRDQmpZV3hzSUdoaGJt'
    'UnNaWE1nYVhRcExnb2dJQ0FnSUNBZ0lDQWdJQ0JpYjI5c0lGOW1ia2x6Vkc5dVpWUnlhV2Nn'
    'UFNBb0tGOW1iaTVsWm1abFkzUWdQVDBnTUhneklIeDhJRjltYmk1bFptWmxZM1FnUFQwZ01I'
    'ZzFLU0FtSmlCZlptNHVjR1Z5YVc5a0lENGdNQ2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lHSnZiMndn'
    'WDJadVNYTlNaWFJ5YVdjZ0lDQTlJQ2hmWm00dWNHVnlhVzlrSUQ0Z01DQW1KaUFoWDJadVNY'
    'TlViMjVsVkhKcFp5azdDaUFnSUNBZ0lDQWdJQ0FnSUdsbUlDaGZabTVKYzFKbGRISnBaeWtn'
    'WW5KbFlXczdDaUFnSUNBZ0lDQWdJQ0FnSUM4dklGUnZibVV0Y0c5eWRHRWdkR0Z5WjJWME9p'
    'QndhWFJqYUNCemJHbGtaWE1nZEc5M1lYSmtJRjltYmk1d1pYSnBiMlFnYjNabGNpQnlaVzFo'
    'YVc1cGJtY2djbTkzY3k0S0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5bWJrbHpWRzl1WlZSeWFX'
    'Y3BJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJSFJoY21kbGRGQmxjbWx2WkNBOUlHWnNiMkYw'
    'S0Y5bWJpNXdaWEpwYjJRcE93b2dJQ0FnSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0FnSUNBZ0lD'
    'OHZJRlJ5WVdOcklHeGhjM1FnTTNoNElISmhkR1VnWm05eUlHVm1abVZqZENBMUlIUnZJR2x1'
    'YUdWeWFYUUtJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOW1iaTVsWm1abFkzUWdQVDBnTUhneklD'
    'WW1JRjltYmk1d1lYSmhiU0ErSURBcElGOXNZWE4wVkZCU1lYUmxJRDBnWDJadUxuQmhjbUZ0'
    'T3dvS0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5elozSWdJQ0E5SUhCaGRGUnBZMnRQWm1aelpY'
    'UmJYMlp3WFNBcklDaGZabklnTFNCd1lYUlRkR0Z5ZEZKdmQxdGZabkJkS1RzS0lDQWdJQ0Fn'
    'SUNBZ0lDQWdhVzUwSUY5bWRXeHNJQ0E5SUdabGRHTm9WR2xqYXloZmMyZHlJQ3NnTVNrZ0xT'
    'Qm1aWFJqYUZScFkyc29YM05uY2lrZ0xTQXhPeUFnTHk4Z2RHbGphM01nTVM0dWMzQmxaV1F0'
    'TVFvZ0lDQWdJQ0FnSUNBZ0lDQm1iRzloZENCZmNGTjBZWEowVW05M0lEMGdaV1ptWldOMGFY'
    'WmxVR1Z5YVc5a095QWdMeThnWm05eUlHWlRZVzF3YkdWUWIzTWdhVzUwWldkeVlYUnBiMjRL'
    'Q2lBZ0lDQWdJQ0FnSUNBZ0lDOHZJRkJwZEdOb0lHVm1abVZqZEhNS0lDQWdJQ0FnSUNBZ0lD'
    'QWdhV1lnS0Y5bWJpNWxabVpsWTNRZ1BUMGdNSGd4S1FvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'WldabVpXTjBhWFpsVUdWeWFXOWtJRDBnYldGNEtERXhNeTR3TENCbFptWmxZM1JwZG1WUVpY'
    'SnBiMlFnTFNCbWJHOWhkQ2hmWm00dWNHRnlZVzBnS2lCZlpuVnNiQ2twT3dvZ0lDQWdJQ0Fn'
    'SUNBZ0lDQmxiSE5sSUdsbUlDaGZabTR1WldabVpXTjBJRDA5SURCNE1pa0tJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lHVm1abVZqZEdsMlpWQmxjbWx2WkNBOUlHMXBiaWc0TlRZdU1Dd2daV1pt'
    'WldOMGFYWmxVR1Z5YVc5a0lDc2dabXh2WVhRb1gyWnVMbkJoY21GdElDb2dYMloxYkd3cEtU'
    'c0tJQ0FnSUNBZ0lDQWdJQ0FnWld4elpTQnBaaUFvWDJadUxtVm1abVZqZENBOVBTQXdlRE1w'
    'SUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUM4dklGUnZibVVnY0c5eWRHRWc0b0NVSUhWelpY'
    'TWdhWFJ6SUc5M2JpQndZWEpoYlNCaGN5QnpiR2xrWlNCeVlYUmxDaUFnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQnBaaUFvWldabVpXTjBhWFpsVUdWeWFXOWtJRHdnZEdGeVoyVjBVR1Z5YVc5a0tR'
    'b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR1ZtWm1WamRHbDJaVkJsY21sdlpDQTlJRzFw'
    'YmloMFlYSm5aWFJRWlhKcGIyUXNJR1ZtWm1WamRHbDJaVkJsY21sdlpDQXJJR1pzYjJGMEtG'
    'OW1iaTV3WVhKaGJTQXFJRjltZFd4c0tTazdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmxiSE5s'
    'SUdsbUlDaGxabVpsWTNScGRtVlFaWEpwYjJRZ1BpQjBZWEpuWlhSUVpYSnBiMlFwQ2lBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1pXWm1aV04wYVhabFVHVnlhVzlrSUQwZ2JXRjRLSFJo'
    'Y21kbGRGQmxjbWx2WkN3Z1pXWm1aV04wYVhabFVHVnlhVzlrSUMwZ1pteHZZWFFvWDJadUxu'
    'QmhjbUZ0SUNvZ1gyWjFiR3dwS1RzS0lDQWdJQ0FnSUNBZ0lDQWdmUW9nSUNBZ0lDQWdJQ0Fn'
    'SUNCbGJITmxJR2xtSUNoZlptNHVaV1ptWldOMElEMDlJREI0TlNrZ2V3b2dJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0x5OGdRMjl1ZEdsdWRXVWdkRzl1WlNCd2IzSjBZU0RpZ0pRZ2RYTmxjeUJN'
    'UVZOVUlETjRlQ0J5WVhSbElDaHViM1FnWDJadUxuQmhjbUZ0SVNrS0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUdsbUlDaGZiR0Z6ZEZSUVVtRjBaU0ErSURBcElIc0tJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0JwWmlBb1pXWm1aV04wYVhabFVHVnlhVzlrSUR3Z2RHRnlaMlYwVUdWeWFX'
    'OWtLUW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmxabVpsWTNScGRtVlFaWEpw'
    'YjJRZ1BTQnRhVzRvZEdGeVoyVjBVR1Z5YVc5a0xDQmxabVpsWTNScGRtVlFaWEpwYjJRZ0t5'
    'Qm1iRzloZENoZmJHRnpkRlJRVW1GMFpTQXFJRjltZFd4c0tTazdDaUFnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnWld4elpTQnBaaUFvWldabVpXTjBhWFpsVUdWeWFXOWtJRDRnZEdGeVoy'
    'VjBVR1Z5YVc5a0tRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCbFptWmxZM1Jw'
    'ZG1WUVpYSnBiMlFnUFNCdFlYZ29kR0Z5WjJWMFVHVnlhVzlrTENCbFptWmxZM1JwZG1WUVpY'
    'SnBiMlFnTFNCbWJHOWhkQ2hmYkdGemRGUlFVbUYwWlNBcUlGOW1kV3hzS1NrN0NpQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ0x5'
    'OGdWbTlzZFcxbElHVm1abVZqZEhNZ0tIUnlZVzV6YVhScGIyNGdjM1JoYlhCbFpDQmhkQ0Iw'
    'YUdseklISnZkeWR6SUhOMFlYSjBJSFJwWTJzcENpQWdJQ0FnSUNBZ0lDQWdJR1pzYjJGMElG'
    'OW1ibFJwWTJ0R0lEMGdabXh2WVhRb1ptVjBZMmhVYVdOcktGOXpaM0lwS1RzS0lDQWdJQ0Fn'
    'SUNBZ0lDQWdhV1lnS0Y5bWJpNWxabVpsWTNRZ1BUMGdNSGhES1FvZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnVms5TVgxTkZWQ2h0YVc0b1gyWnVMbkJoY21GdExDQTJOQ2tzSUY5bWJsUnBZMnRH'
    'S1RzS0lDQWdJQ0FnSUNBZ0lDQWdaV3h6WlNCcFppQW9YMlp1TG1WbVptVmpkQ0E5UFNBd2VF'
    'RWdmSHdnWDJadUxtVm1abVZqZENBOVBTQXdlRFlwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUdsdWRDQmZkblVnUFNBb1gyWnVMbkJoY21GdFBqNDBLU1l3ZUVZc0lGOTJaQ0E5SUY5bWJp'
    'NXdZWEpoYlNZd2VFWTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQldUMHhmVTBWVUtHTnNZVzF3'
    'S0Y5MmIyeERkWEp5SUNzZ0tGOTJkVDR3UDE5MmRUb3RYM1prS1NBcUlGOW1kV3hzTENBd0xD'
    'QTJOQ2tzSUY5bWJsUnBZMnRHS1RzS0lDQWdJQ0FnSUNBZ0lDQWdmUW9nSUNBZ0lDQWdJQ0Fn'
    'SUNCbGJITmxJR2xtSUNoZlptNHVaV1ptWldOMElEMDlJREI0UlNrZ2V3b2dJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0x5OGdSVUY0SUdacGJtVWdkbTlzSUhWd0xDQkZRbmdnWm1sdVpTQjJiMndn'
    'Wkc5M2JpQW9hVzV6ZEdGdWRDQndaWElnY205M0tRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FX'
    'NTBJRjlsY3lBOUlDaGZabTR1Y0dGeVlXMGdQajRnTkNrZ0ppQXdlRVk3Q2lBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0JwYm5RZ1gyVjJJRDBnSUY5bWJpNXdZWEpoYlNBZ0lDQWdJQ0FtSURCNFJq'
    'c0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmWlhNZ1BUMGdNSGhCS1NBZ0lDQWdJRlpQ'
    'VEY5VFJWUW9ZMnhoYlhBb1gzWnZiRU4xY25JZ0t5QmZaWFlzSURBc0lEWTBLU3dnWDJadVZH'
    'bGphMFlwT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnWld4elpTQnBaaUFvWDJWeklEMDlJREI0'
    'UWlrZ1ZrOU1YMU5GVkNoamJHRnRjQ2hmZG05c1EzVnljaUF0SUY5bGRpd2dNQ3dnTmpRcExD'
    'QmZabTVVYVdOclJpazdDaUFnSUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lDQWdMeThn'
    'TUhnMUlHRnNjMjhnWVhCd2JHbGxjeUIwYUdVZ2RtOXNkVzFsSUhOc2FXUmxJSEJ2Y25ScGIy'
    'NGdLR2hwWjJnZ2JtbGlZbXhsSUQwZ2RYQXNJR3h2ZHlBOUlHUnZkMjRwQ2lBZ0lDQWdJQ0Fn'
    'SUNBZ0lHbG1JQ2hmWm00dVpXWm1aV04wSUQwOUlEQjROU2tnZXdvZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnYVc1MElGOTJkU0E5SUNoZlptNHVjR0Z5WVcwK1BqUXBKakI0Uml3Z1gzWmtJRDBn'
    'WDJadUxuQmhjbUZ0SmpCNFJqc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lGWlBURjlUUlZRb1ky'
    'eGhiWEFvWDNadmJFTjFjbklnS3lBb1gzWjFQakEvWDNaMU9pMWZkbVFwSUNvZ1gyWjFiR3dz'
    'SURBc0lEWTBLU3dnWDJadVZHbGphMFlwT3dvZ0lDQWdJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lD'
    'QWdJQ0FnSUM4dklGQmxjaTEwYVdOcklHWlRZVzF3YkdWUWIzTWdhVzUwWldkeVlYUnBiMjRn'
    'Wm05eUlIUm9hWE1nY205M0lDaHRZWFJqYUdWeklGQlVJSE5sYldGdWRHbGpjd29nSUNBZ0lD'
    'QWdJQ0FnSUNBdkx5QmxlR0ZqZEd4NUlPS0FsQ0JrYVhOamNtVjBaU0IwYVdOckxXSjVMWFJw'
    'WTJzc0lHNXZkQ0JzYVc1bFlYSXRjbUZ0Y0NCamIyNTBhVzUxYjNWektTNEtJQ0FnSUNBZ0lD'
    'QWdJQ0FnTHk4Z1VHVnlhVzlrSUdGMElIUnBZMnNnTUNCdlppQjBhR2x6SUhKdmR5QTlJRjl3'
    'VTNSaGNuUlNiM2N1SUZScFkydHpJREV1TG1aMWJHd2dZWEJ3YkhrS0lDQWdJQ0FnSUNBZ0lD'
    'QWdMeThnZEdobElITnNhV1JsSUhOMFpYQXVJRmRsSUhKbExXUmxjbWwyWlNCMGFHVWdjR1Z5'
    'TFhScFkyc2djM1JsY0NCbWNtOXRJSFJvWlNCeWIzY25jd29nSUNBZ0lDQWdJQ0FnSUNBdkx5'
    'QmxabVpsWTNRZ2NtRjBhR1Z5SUhSb1lXNGdaR2wyYVdScGJtY2dLR1ZtWm1WamRHbDJaVkJs'
    'Y21sdlpDQXRJRjl3VTNSaGNuUlNiM2NwTDE5bWRXeHNDaUFnSUNBZ0lDQWdJQ0FnSUM4dklI'
    'TnBibU5sSURONGVDODFlSGdnWTJ4aGJYQnBibWNnWVhRZ2RHRnlaMlYwSUcxaGEyVnpJSFJv'
    'WVhRZ1lYWmxjbUZuWlNCdGFYTnNaV0ZrYVc1bkxnb2dJQ0FnSUNBZ0lDQWdJQ0I3Q2lBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0JtYkc5aGRDQmZRMllnSUQwZ1l6UnpjR1ZsWkhOYmMyMXdMbVpw'
    'Ym1WMGRXNWxJQ1lnTUhoR1hTQXFJRFF5T0M0d093b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1pt'
    'eHZZWFFnWDJSMElDQTlJREV1TUNBdklGUkpRMHRUWDFCRlVsOVRSVU03Q2lBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0JtYkc5aGRDQmZVSFFnSUQwZ1gzQlRkR0Z5ZEZKdmR6c0tJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDOHZJRVJsZEdWeWJXbHVaU0J3WlhJdGRHbGpheUJ6ZEdWd0lDaHphV2R1'
    'WldRcElHRnVaQ0IwWVhKblpYUWdabTl5SUdOc1lXMXdhVzVuTGdvZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnYVc1MElGOXpkR1Z3SUQwZ01Ec0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHSnZiMndn'
    'WDJOc1lXMXdWRzlVWjNRZ1BTQm1ZV3h6WlRzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsbUlD'
    'aGZabTR1WldabVpXTjBJRDA5SURCNE1Ta2dYM04wWlhBZ1BTQXRYMlp1TG5CaGNtRnRPd29n'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdaV3h6WlNCcFppQW9YMlp1TG1WbVptVmpkQ0E5UFNBd2VE'
    'SXBJRjl6ZEdWd0lEMGdYMlp1TG5CaGNtRnRPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdaV3h6'
    'WlNCcFppQW9YMlp1TG1WbVptVmpkQ0E5UFNBd2VETXBJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNCZlkyeGhiWEJVYjFSbmRDQTlJSFJ5ZFdVN0NpQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdhV1lnS0Y5d1UzUmhjblJTYjNjZ1BDQjBZWEpuWlhSUVpYSnBiMlFwSUNBZ0lD'
    'QWdYM04wWlhBZ1BTQmZabTR1Y0dGeVlXMDdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'Wld4elpTQnBaaUFvWDNCVGRHRnlkRkp2ZHlBK0lIUmhjbWRsZEZCbGNtbHZaQ2tnWDNOMFpY'
    'QWdQU0F0WDJadUxuQmhjbUZ0T3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnWld4elpTQnBaaUFvWDJadUxtVm1abVZqZENBOVBTQXdlRFVwSUhzS0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmZZMnhoYlhCVWIxUm5kQ0E5SUhSeWRXVTdDaUFn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOXdVM1JoY25SU2IzY2dQQ0IwWVhKblpY'
    'UlFaWEpwYjJRcElDQWdJQ0FnWDNOMFpYQWdQU0JmYkdGemRGUlFVbUYwWlRzS0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQmxiSE5sSUdsbUlDaGZjRk4wWVhKMFVtOTNJRDRnZEdGeVoy'
    'VjBVR1Z5YVc5a0tTQmZjM1JsY0NBOUlDMWZiR0Z6ZEZSUVVtRjBaVHNLSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQzh2SUY5bWRXeHNJQ3NnTVNBOUlI'
    'UnZkR0ZzSUhScFkydHpJR2x1SUhSb2FYTWdjbTkzSUNoemNHVmxaQ2tLSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJR2x1ZENCZmRHbGphM05mYVc1ZmNtOTNJRDBnWDJaMWJHd2dLeUF4T3dvZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnWm05eUlDaHBiblFnWDNRZ1BTQXdPeUJmZENBOElETXlPeUJm'
    'ZENzcktTQjdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tGOTBJRDQ5SUY5MGFX'
    'TnJjMTlwYmw5eWIzY3BJR0p5WldGck93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2xt'
    'SUNoZlVIUWdQaUF3TGpBcElGOW1VMkZ0Y0d4bFVHOXpRV05qSUNzOUlGOURaaUFxSUY5a2RD'
    'QXZJRjlRZERzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQXZMeUJWY0dSaGRHVWdjR1Z5'
    'YVc5a0lHWnZjaUJ1WlhoMElIUnBZMnNnS0c5dWJIa2dhV1lnZENBOElGOW1kV3hzTENCcExt'
    'VXVJSFJvWlhKbElHRnlaUW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUM4dklHMXZjbVVn'
    'ZEdsamEzTWdkRzhnYzJ4cFpHVWdhVzRnZEdocGN5QnliM2NwQ2lBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ2FXWWdLRjl6ZEdWd0lDRTlJREFnSmlZZ1gzUWdQQ0JmWm5Wc2JDa2dld29n'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQm1iRzloZENCZlVHNGdQU0JmVUhRZ0t5'
    'Qm1iRzloZENoZmMzUmxjQ2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2xt'
    'SUNoZlkyeGhiWEJVYjFSbmRDa2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnYVdZZ0tGOXpkR1Z3SUQ0Z01Da2dJQ0FnSUNCZlVHNGdQU0J0YVc0b1gxQnVMQ0Iw'
    'WVhKblpYUlFaWEpwYjJRcE93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdaV3h6WlNCcFppQW9YM04wWlhBZ1BDQXdLU0JmVUc0Z1BTQnRZWGdvWDFCdUxDQjBZWEpu'
    'WlhSUVpYSnBiMlFwT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0I5SUdWc2My'
    'VWdld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnWDFCdUlEMGdZMnho'
    'YlhBb1gxQnVMQ0F4TVRNdU1Dd2dPRFUyTGpBcE93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUY5UWRDQTlJRjlR'
    'YmpzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QjlDaUFnSUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lDQWdYMlp5S3lzN0NpQWdJQ0Fn'
    'SUNBZ0lDQWdJQzh2SUVGa2RtRnVZMlVnZEc4Z2JtVjRkQ0J6YjI1bklIQnZjMmwwYVc5dUlI'
    'ZG9aVzRnZDJVbmRtVWdaWGhvWVhWemRHVmtJSFJvYVhNZ2NHRjBkR1Z5YmlkeklISnZkM01L'
    'SUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjltY2lBK1BTQndZWFJUZEdGeWRGSnZkMXRmWm5CZElD'
    'c2dLSEJoZEZKdmQwOW1abk5sZEZ0ZlpuQXJNVjBnTFNCd1lYUlNiM2RQWm1aelpYUmJYMlp3'
    'WFNrcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lGOW1jQ3NyT3dvZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnWDJaeUlEMGdLRjltY0NBOElGTlBUa2RmVEVWT1IxUklLU0EvSUhCaGRGTjBZWEow'
    'VW05M1cxOW1jRjBnT2lBd093b2dJQ0FnSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0FnZlFvS0lD'
    'QWdJQ0FnSUNBdkx5QkRkWEp5Wlc1MElISnZkeUJ3WVhKMGFXRnNJQ2h1YjI0dGRISnBaMmRs'
    'Y2lCeWIzY2diMjVzZVNEaWdKUWdkSEpwWjJkbGNpQm9ZVzVrYkdWa0lHRmliM1psS1FvZ0lD'
    'QWdJQ0FnSUdsbUlDaGZjR055TG1sdWMzUnlkVzFsYm5RZ1BEMGdNQ0FtSmlCZmNHTnlMbkJs'
    'Y21sdlpDQThQU0F3S1NCN0NpQWdJQ0FnSUNBZ0lDQWdJQzh2SUZScFkyc2djM1JoYlhBZ1pt'
    'OXlJR04xY25KbGJuUXRjbTkzSUhCaGNuUnBZV3dnZG05c0lHVm1abVZqZEhNNklIUm9aU0Jq'
    'ZFhKeVpXNTBJSEp2ZHlkekNpQWdJQ0FnSUNBZ0lDQWdJQzh2SUhOMFlYSjBJSFJwWTJzdUlG'
    'Um9aU0EyTkMxellXMXdiR1VnY21GdGNDQmpiMjF3YkdWMFpYTWdkMlZzYkNCM2FYUm9hVzRn'
    'ZEdobElHWnBjbk4wQ2lBZ0lDQWdJQ0FnSUNBZ0lDOHZJSFJwWTJzc0lITnZJR0Z1ZVNCMmIy'
    'd2dZMmhoYm1kbElDSm9ZWEJ3Wlc1cGJtY2dZWFFnZEdocGN5QnliM2NpSUhKbFlXUnpJR0Z6'
    'SUdoaGRtbHVad29nSUNBZ0lDQWdJQ0FnSUNBdkx5QnlZVzF3WldRZ2RHOGdhWFJ6SUdacGJt'
    'RnNJSFpoYkhWbElHRnNiVzl6ZENCcGJXMWxaR2xoZEdWc2VTNEtJQ0FnSUNBZ0lDQWdJQ0Fn'
    'YVc1MElGOWpkWEpUWjNJZ1BTQndZWFJVYVdOclQyWm1jMlYwVzNCdmN5NXpiMjVuVUc5elhT'
    'QXJJQ2h3YjNNdWNtOTNJQzBnY0dGMFUzUmhjblJTYjNkYmNHOXpMbk52Ym1kUWIzTmRLVHNL'
    'SUNBZ0lDQWdJQ0FnSUNBZ1pteHZZWFFnWDJOMWNsUnBZMnRHSUQwZ1pteHZZWFFvWm1WMFky'
    'aFVhV05yS0Y5amRYSlRaM0lwS1RzS0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5d1kzSXVaV1pt'
    'WldOMElEMDlJREI0UXlrS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUZaUFRGOVRSVlFvYldsdUtG'
    'OXdZM0l1Y0dGeVlXMHNJRFkwS1N3Z1gyTjFjbFJwWTJ0R0tUc0tJQ0FnSUNBZ0lDQWdJQ0Fn'
    'Wld4elpTQnBaaUFvWDNCamNpNWxabVpsWTNRZ1BUMGdNSGhCSUh4OElGOXdZM0l1WldabVpX'
    'TjBJRDA5SURCNE5pa2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5MmRTQTlJQ2hm'
    'Y0dOeUxuQmhjbUZ0UGo0MEtTWXdlRVlzSUY5MlpDQTlJRjl3WTNJdWNHRnlZVzBtTUhoR093'
    'b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1ZrOU1YMU5GVkNoamJHRnRjQ2hmZG05c1EzVnljaUFy'
    'SUNoZmRuVStNRDlmZG5VNkxWOTJaQ2tnS2lCZmNHTjBMQ0F3TENBMk5Da3NJRjlqZFhKVWFX'
    'TnJSaWs3Q2lBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnWld4elpTQnBaaUFv'
    'WDNCamNpNWxabVpsWTNRZ1BUMGdNSGhGS1NCN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcGJu'
    'UWdYMlZ6SUQwZ0tGOXdZM0l1Y0dGeVlXMGdQajRnTkNrZ0ppQXdlRVk3Q2lBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0JwYm5RZ1gyVjJJRDBnSUY5d1kzSXVjR0Z5WVcwZ0lDQWdJQ0FnSmlBd2VF'
    'WTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvWDJWeklEMDlJREI0UVNrZ0lDQWdJQ0JX'
    'VDB4ZlUwVlVLR05zWVcxd0tGOTJiMnhEZFhKeUlDc2dYMlYyTENBd0xDQTJOQ2tzSUY5amRY'
    'SlVhV05yUmlrN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCbGJITmxJR2xtSUNoZlpYTWdQVDBn'
    'TUhoQ0tTQldUMHhmVTBWVUtHTnNZVzF3S0Y5MmIyeERkWEp5SUMwZ1gyVjJMQ0F3TENBMk5D'
    'a3NJRjlqZFhKVWFXTnJSaWs3Q2lBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0Fn'
    'YVdZZ0tGOXdZM0l1WldabVpXTjBJRDA5SURCNE5Ta2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdMeThnTUhnMUlIWnZiQzF6Ykdsa1pTQndiM0owYVc5dUlHOXVJR04xY25KbGJuUWdjbTkz'
    'Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwYm5RZ1gzWjFJRDBnS0Y5d1kzSXVjR0Z5WVcwK1Bq'
    'UXBKakI0Uml3Z1gzWmtJRDBnWDNCamNpNXdZWEpoYlNZd2VFWTdDaUFnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQldUMHhmVTBWVUtHTnNZVzF3S0Y5MmIyeERkWEp5SUNzZ0tGOTJkVDR3UDE5MmRU'
    'b3RYM1prS1NBcUlGOXdZM1FzSURBc0lEWTBLU3dnWDJOMWNsUnBZMnRHS1RzS0lDQWdJQ0Fn'
    'SUNBZ0lDQWdmUW9nSUNBZ0lDQWdJSDBLSUNBZ0lIMEtDaUFnSUNBdkx5QkRkWEp5Wlc1MElI'
    'SnZkeUJ3WVhKMGFXRnNJSEJwZEdOb0lHVm1abVZqZENBb1lYQndiR2xsY3lCbGRtVnVJRzl1'
    'SUhSeWFXZG5aWElnY205M0tTNEtJQ0FnSUM4dklGVnpaU0JqYjI1MGFXNTFiM1Z6SUhCdmN5'
    'NTBhV05ySU9LQWxDQmlkWFFnWTJGd0lHbDBJR0YwSUNoemNHVmxaQzB4S1NCemJ5QjBhR1Vn'
    'WTI5dWRISnBZblYwYVc5dUNpQWdJQ0F2THlCaGRDQjBhR1VnYkdGemRDQnpZVzF3YkdVZ2Iy'
    'WWdkR2hwY3lCeWIzY2daWGhoWTNSc2VTQnRZWFJqYUdWeklIZG9ZWFFnZEdobElHWnZjbmRo'
    'Y21RZ2MyTmhiZ29nSUNBZ0x5OGdkMmxzYkNCMWMyVWdabTl5SUhSb2FYTWdjbTkzSUc5dVky'
    'VWdhWFFnWW1WamIyMWxjeUJoSUNKamIyMXdiR1YwWldRaUlISnZkeTRnSUZkcGRHaHZkWFFn'
    'ZEdobENpQWdJQ0F2THlCallYQXNJSEJ2Y3k1MGFXTnJJR0Z3Y0hKdllXTm9aWE1nWUhOd1pX'
    'VmtZQ0JoZENCMGFHVWdjbTkzSUdKdmRXNWtZWEo1SUhkb2FXeGxJSFJvWlNCbWIzSjNZWEpr'
    'Q2lBZ0lDQXZMeUJ6WTJGdUlIVnpaWE1nWUhOd1pXVmtMVEZnTENCd2NtOWtkV05wYm1jZ1lT'
    'QitNUzEwYVdOcklHSmhZMnQzWVhKa0lIQmxjbWx2WkNCcWRXMXdJRDBnWTJ4cFkyc3VDaUFn'
    'SUNBdkx5QlBibXg1SUhCaGVTQjBhR1VnWm1WMFkyaFVhV05ySUdOdmMzUWdkMmhsYmlCaElI'
    'QnBkR05vSUdWbVptVmpkQ0JwY3lCaFkzUjFZV3hzZVNCd2NtVnpaVzUwTGdvZ0lDQWdMeThn'
    'VTJGMlpTQmxabVpsWTNScGRtVlFaWEpwYjJRZ1lYUWdjM1JoY25RZ2IyWWdZM1Z5Y21WdWRD'
    'QnliM2NzSUVKRlJrOVNSU0J3WVhKMGFXRnNJSEJwZEdOb0NpQWdJQ0F2THlCbFptWmxZM1Fn'
    'WVhCd2JHbGpZWFJwYjI0ZzRvQ1VJRzVsWldSbFpDQm1iM0lnZEdobElHTjFjbkpsYm5RdGNt'
    'OTNJR2hsWVdRZ1kyOXVkSEpwWW5WMGFXOXVJSFJ2Q2lBZ0lDQXZMeUJmWmxOaGJYQnNaVkJ2'
    'YzBGall5QmlaV3h2ZHk0S0lDQWdJR1pzYjJGMElGOXdVM1JoY25SRGRYSWdQU0JsWm1abFkz'
    'UnBkbVZRWlhKcGIyUTdDZ29nSUNBZ0x5OGdRM1Z5Y21WdWRDQnliM2NnY0dGeWRHbGhiQ0J3'
    'YVhSamFDQmxabVpsWTNRZ0tHRndjR3hwWlhNZ2IyNGdkSEpwWjJkbGNpQnliM2NnVDFJZ1ky'
    'OXVkR2x1ZFdGMGFXOXVJSEp2ZHlrdUNpQWdJQ0F2THlCR2IzSWdNWGg0THpKNGVEb2dZV3gz'
    'WVhseklHRndjR3hwWlhNZ2IyNGdZM1Z5Y21WdWRDQnliM2N1Q2lBZ0lDQXZMeUJHYjNJZ00z'
    'aDRMelY0ZURvZ1lYQndiR2xsY3lCM2FHVnVaWFpsY2lCamRYSnlaVzUwSUhKdmR5QmpZWEp5'
    'YVdWeklIUm9aU0JsWm1abFkzUWc0b0NVSUdKdmRHZ0tJQ0FnSUM4dklDQWdZMjl1ZEdsdWRX'
    'RjBhVzl1SUhKdmQzTWdLSEJsY21sdlpEMDlNQ2tnUVU1RUlIUnZibVV0Y0c5eWRHRWdkR0Z5'
    'WjJWMElISnZkM01nS0hCbGNtbHZaRDR3S1M0S0lDQWdJQzh2SUNBZ1ZHaGxJSFJoY21kbGRG'
    'QmxjbWx2WkNCM1lYTWdZV3h5WldGa2VTQnpaWFFnWVdKdmRtVWdLR1ZwZEdobGNpQm1jbTl0'
    'SUhSeWFXZE9iM1JsSUc5eUNpQWdJQ0F2THlBZ0lIUnZibVZUYkdsa1pWUmhjbWRsZENrN0lI'
    'Um9hWE1nWW14dlkyc2daRzlsY3lCMGFHVWdjR1Z5TFhScFkyc2dZV05qZFcxMWJHRjBhVzl1'
    'SUhSdmQyRnlaQ0JwZEM0S0lDQWdJR2xtSUNoZmNHTnlMbVZtWm1WamRDQTlQU0F3ZURFZ2ZI'
    'd2dYM0JqY2k1bFptWmxZM1FnUFQwZ01IZ3lJSHg4Q2lBZ0lDQWdJQ0FnWDNCamNpNWxabVps'
    'WTNRZ1BUMGdNSGd6SUh4OElGOXdZM0l1WldabVpXTjBJRDA5SURCNE5Ta2dld29nSUNBZ0lD'
    'QWdJR2x1ZENCZmMyZHlYMk4xY2lBOUlIQmhkRlJwWTJ0UFptWnpaWFJiY0c5ekxuTnZibWRR'
    'YjNOZElDc2dLSEJ2Y3k1eWIzY2dMU0J3WVhSVGRHRnlkRkp2ZDF0d2IzTXVjMjl1WjFCdmMx'
    'MHBPd29nSUNBZ0lDQWdJR1pzYjJGMElGOXdkR1lnUFNCdGFXNG9jRzl6TG5ScFkyc3NDaUFn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQm1iRzloZENobVpYUmphRlJwWTJzb1gz'
    'Tm5jbDlqZFhJZ0t5QXhLU0F0SUdabGRHTm9WR2xqYXloZmMyZHlYMk4xY2lrZ0xTQXhLU2s3'
    'Q2lBZ0lDQWdJQ0FnYVdZZ0tGOXdZM0l1WldabVpXTjBJRDA5SURCNE1Ta0tJQ0FnSUNBZ0lD'
    'QWdJQ0FnWldabVpXTjBhWFpsVUdWeWFXOWtJRDBnYldGNEtERXhNeTR3TENCbFptWmxZM1Jw'
    'ZG1WUVpYSnBiMlFnTFNCbWJHOWhkQ2hmY0dOeUxuQmhjbUZ0S1NBcUlGOXdkR1lwT3dvZ0lD'
    'QWdJQ0FnSUdWc2MyVWdhV1lnS0Y5d1kzSXVaV1ptWldOMElEMDlJREI0TWlrS0lDQWdJQ0Fn'
    'SUNBZ0lDQWdaV1ptWldOMGFYWmxVR1Z5YVc5a0lEMGdiV2x1S0RnMU5pNHdMQ0JsWm1abFkz'
    'UnBkbVZRWlhKcGIyUWdLeUJtYkc5aGRDaGZjR055TG5CaGNtRnRLU0FxSUY5d2RHWXBPd29n'
    'SUNBZ0lDQWdJR1ZzYzJVZ2FXWWdLRjl3WTNJdVpXWm1aV04wSUQwOUlEQjRNeWtnZXdvZ0lD'
    'QWdJQ0FnSUNBZ0lDQXZMeUJVYjI1bElIQnZjblJoSU9LQWxDQjFjMlZ6SUdsMGN5QnZkMjRn'
    'Y0dGeVlXMGdZWE1nYzJ4cFpHVWdjbUYwWlFvZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvWldabVpX'
    'TjBhWFpsVUdWeWFXOWtJRHdnZEdGeVoyVjBVR1Z5YVc5a0tRb2dJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ1pXWm1aV04wYVhabFVHVnlhVzlrSUQwZ2JXbHVLSFJoY21kbGRGQmxjbWx2WkN3Z1pX'
    'Wm1aV04wYVhabFVHVnlhVzlrSUNzZ1pteHZZWFFvWDNCamNpNXdZWEpoYlNrZ0tpQmZjSFJt'
    'S1RzS0lDQWdJQ0FnSUNBZ0lDQWdaV3h6WlNCcFppQW9aV1ptWldOMGFYWmxVR1Z5YVc5a0lE'
    'NGdkR0Z5WjJWMFVHVnlhVzlrS1FvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnWldabVpXTjBhWFps'
    'VUdWeWFXOWtJRDBnYldGNEtIUmhjbWRsZEZCbGNtbHZaQ3dnWldabVpXTjBhWFpsVUdWeWFX'
    'OWtJQzBnWm14dllYUW9YM0JqY2k1d1lYSmhiU2tnS2lCZmNIUm1LVHNLSUNBZ0lDQWdJQ0I5'
    'Q2lBZ0lDQWdJQ0FnWld4elpTQjdJQ0F2THlBd2VEVWc0b0NVSUdOdmJuUnBiblZsSUhSdmJt'
    'VWdjRzl5ZEdFZ2RYTnBibWNnYkdGemRDQXplSGdnY21GMFpTQW9jR0Z5WVcwZ2FYTWdkbTlz'
    'TFhOc2FXUmxJRzl1YkhrcENpQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZmJHRnpkRlJRVW1GMFpT'
    'QStJREFwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsbUlDaGxabVpsWTNScGRtVlFaWEpw'
    'YjJRZ1BDQjBZWEpuWlhSUVpYSnBiMlFwQ2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1pX'
    'Wm1aV04wYVhabFVHVnlhVzlrSUQwZ2JXbHVLSFJoY21kbGRGQmxjbWx2WkN3Z1pXWm1aV04w'
    'YVhabFVHVnlhVzlrSUNzZ1pteHZZWFFvWDJ4aGMzUlVVRkpoZEdVcElDb2dYM0IwWmlrN0Np'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCbGJITmxJR2xtSUNobFptWmxZM1JwZG1WUVpYSnBiMlFn'
    'UGlCMFlYSm5aWFJRWlhKcGIyUXBDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnWldabVpX'
    'TjBhWFpsVUdWeWFXOWtJRDBnYldGNEtIUmhjbWRsZEZCbGNtbHZaQ3dnWldabVpXTjBhWFps'
    'VUdWeWFXOWtJQzBnWm14dllYUW9YMnhoYzNSVVVGSmhkR1VwSUNvZ1gzQjBaaWs3Q2lBZ0lD'
    'QWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQjlDaUFnSUNCOUNnb2dJQ0FnTHk4ZzRwU0E0cFNB'
    'NHBTQUlFTjFjbkpsYm5RdGNtOTNJR2hsWVdRZ1kyOXVkSEpwWW5WMGFXOXVJSFpwWVNCd1pY'
    'SXRkR2xqYXlCcGJuUmxaM0poZEdsdmJpRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURp'
    'bElEaWxJRGlsSURpbElBS0lDQWdJQzh2SUZOaGJXVWdjR1Z5TFhScFkyc2diVzlrWld3Z1lY'
    'TWdabTl5ZDJGeVpDQnpZMkZ1TGlCWFpTZHlaU0JoZENCd2IzTXVkR2xqYXlCM2FYUm9hVzRn'
    'ZEdobENpQWdJQ0F2THlCamRYSnlaVzUwSUhKdmR5NGdVR1Z5YVc5a0lHRjBJSFJwWTJzZ01D'
    'QTlJRjl3VTNSaGNuUkRkWEl1SUZkbElHbHVkR1ZuY21GMFpTQjBhV05yY3dvZ0lDQWdMeThn'
    'V3pBc0lIQnZjeTUwYVdOcktTRGlnSlFnYVM1bExpd2dkMlVuY21VZ0ltSmxabTl5WlNJZ2RH'
    'aGxJR0p2ZFc1a1lYSjVJR0YwSUdWdVpDQnZaaUIwYVdOcklHWnNiMjl5S0hCdmN5NTBhV05y'
    'S1M0S0lDQWdJQzh2SUVadmNpQjBhR1VnWm5KaFkzUnBiMjVoYkNCemRXSXRkR2xqYXlBb1lt'
    'VjBkMlZsYmlCcGJuUmxaMlZ5SUhScFkyc2daV1JuWlhNcExDQmhaR1FLSUNBZ0lDOHZJSEJo'
    'Y25ScFlXd3RkR2xqYXlCamIyNTBjbWxpZFhScGIyNGdZWFFnZEdobElIQmxjbWx2WkNCamRY'
    'SnlaVzUwYkhrZ2FXNGdaV1ptWldOMExnb2dJQ0FnZXdvZ0lDQWdJQ0FnSUdac2IyRjBJRjlE'
    'Wmw5b0lEMGdZelJ6Y0dWbFpITmJjMjF3TG1acGJtVjBkVzVsSUNZZ01IaEdYU0FxSURReU9D'
    'NHdPd29nSUNBZ0lDQWdJR1pzYjJGMElGOWtkQ0FnSUQwZ01TNHdJQzhnVkVsRFMxTmZVRVZT'
    'WDFORlF6c0tJQ0FnSUNBZ0lDQm1iRzloZENCZlVIUWdJQ0E5SUY5d1UzUmhjblJEZFhJN0Np'
    'QWdJQ0FnSUNBZ2FXNTBJRjl6ZEdWd0lEMGdNRHNLSUNBZ0lDQWdJQ0JpYjI5c0lGOWpiR0Z0'
    'Y0ZSdlZHZDBJRDBnWm1Gc2MyVTdDaUFnSUNBZ0lDQWdhV1lnS0Y5d1kzSXVaV1ptWldOMElE'
    'MDlJREI0TVNrZ1gzTjBaWEFnUFNBdFgzQmpjaTV3WVhKaGJUc0tJQ0FnSUNBZ0lDQmxiSE5s'
    'SUdsbUlDaGZjR055TG1WbVptVmpkQ0E5UFNBd2VESXBJRjl6ZEdWd0lEMGdYM0JqY2k1d1lY'
    'SmhiVHNLSUNBZ0lDQWdJQ0JsYkhObElHbG1JQ2hmY0dOeUxtVm1abVZqZENBOVBTQXdlRE1w'
    'SUhzS0lDQWdJQ0FnSUNBZ0lDQWdYMk5zWVcxd1ZHOVVaM1FnUFNCMGNuVmxPd29nSUNBZ0lD'
    'QWdJQ0FnSUNCcFppQW9YM0JUZEdGeWRFTjFjaUE4SUhSaGNtZGxkRkJsY21sdlpDa2dJQ0Fn'
    'SUNCZmMzUmxjQ0E5SUY5d1kzSXVjR0Z5WVcwN0NpQWdJQ0FnSUNBZ0lDQWdJR1ZzYzJVZ2FX'
    'WWdLRjl3VTNSaGNuUkRkWElnUGlCMFlYSm5aWFJRWlhKcGIyUXBJRjl6ZEdWd0lEMGdMVjl3'
    'WTNJdWNHRnlZVzA3Q2lBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUdWc2MyVWdhV1lnS0Y5d1kz'
    'SXVaV1ptWldOMElEMDlJREI0TlNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0JmWTJ4aGJYQlViMVJu'
    'ZENBOUlIUnlkV1U3Q2lBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmY0ZOMFlYSjBRM1Z5SUR3Z2RH'
    'RnlaMlYwVUdWeWFXOWtLU0FnSUNBZ0lGOXpkR1Z3SUQwZ1gyeGhjM1JVVUZKaGRHVTdDaUFn'
    'SUNBZ0lDQWdJQ0FnSUdWc2MyVWdhV1lnS0Y5d1UzUmhjblJEZFhJZ1BpQjBZWEpuWlhSUVpY'
    'SnBiMlFwSUY5emRHVndJRDBnTFY5c1lYTjBWRkJTWVhSbE93b2dJQ0FnSUNBZ0lIMEtJQ0Fn'
    'SUNBZ0lDQnBiblFnWDJaMWJHeGZkR2xqYTNNZ1BTQnBiblFvY0c5ekxuUnBZMnNwT3dvZ0lD'
    'QWdJQ0FnSUdac2IyRjBJRjltY21GaklDQWdJQ0E5SUhCdmN5NTBhV05ySUMwZ1pteHZZWFFv'
    'WDJaMWJHeGZkR2xqYTNNcE93b2dJQ0FnSUNBZ0lHWnZjaUFvYVc1MElGOTBJRDBnTURzZ1gz'
    'UWdQQ0F6TWpzZ1gzUXJLeWtnZXdvZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvWDNRZ1BqMGdYMlox'
    'Ykd4ZmRHbGphM01wSUdKeVpXRnJPd29nSUNBZ0lDQWdJQ0FnSUNCcFppQW9YMUIwSUQ0Z01D'
    'NHdLU0JmWmxOaGJYQnNaVkJ2YzBGall5QXJQU0JmUTJaZmFDQXFJRjlrZENBdklGOVFkRHNL'
    'SUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjl6ZEdWd0lDRTlJREFwSUhzS0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUdac2IyRjBJRjlRYmlBOUlGOVFkQ0FySUdac2IyRjBLRjl6ZEdWd0tUc0tJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2hmWTJ4aGJYQlViMVJuZENrZ2V3b2dJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZmMzUmxjQ0ErSURBcElDQWdJQ0FnWDFCdUlEMGdiV2x1'
    'S0Y5UWJpd2dkR0Z5WjJWMFVHVnlhVzlrS1RzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QmxiSE5sSUdsbUlDaGZjM1JsY0NBOElEQXBJRjlRYmlBOUlHMWhlQ2hmVUc0c0lIUmhjbWRs'
    'ZEZCbGNtbHZaQ2s3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0I5SUdWc2MyVWdld29nSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUY5UWJpQTlJR05zWVcxd0tGOVFiaXdnTVRFekxqQXNJRGcx'
    'Tmk0d0tUc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lG'
    'OVFkQ0E5SUY5UWJqc0tJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUgwS0lDQWdJQ0Fn'
    'SUNBdkx5QlRkV0l0ZEdsamF5Qm1jbUZqZEdsdmJtRnNJR052Ym5SeWFXSjFkR2x2YmlCaGRD'
    'QmpkWEp5Wlc1MElIQmxjbWx2WkFvZ0lDQWdJQ0FnSUdsbUlDaGZabkpoWXlBK0lEQXVNQ0Ft'
    'SmlCZlVIUWdQaUF3TGpBcENpQWdJQ0FnSUNBZ0lDQWdJRjltVTJGdGNHeGxVRzl6UVdOaklD'
    'czlJRjlEWmw5b0lDb2dYMlIwSUNvZ1gyWnlZV01nTHlCZlVIUTdDaUFnSUNCOUNnb2dJQ0Fn'
    'THk4Z1UzQmxZMmxoYkNCallYTmxPaUIwY21sbloyVnlJSEp2ZHlCSlV5QmpkWEp5Wlc1MElI'
    'SnZkeUFvYm04Z2MyVndZWEpoZEdVZ2RISnBaMmRsY2lCMFlXbHNJQzhnYzJOaGJpa3VDaUFn'
    'SUNBdkx5QlVhR1VnWVdOamRXMTFiR0YwYjNJZ1lXSnZkbVVnYUdGdVpHeGxaQ0JvWldGa0lH'
    'WnliMjBnZEdsamF5QXdJSFJ2SUhCdmN5NTBhV05ySUdGMElHTnZibk4wWVc1MENpQWdJQ0F2'
    'THlCMGNtbG5UbTkwWlM1d1pYSnBiMlFnS0hOcGJtTmxJRjl3VTNSaGNuUkRkWElnZDJGeklI'
    'TmxkQ0IwYnlCbFptWmxZM1JwZG1WUVpYSnBiMlFnZDJocFkyZ2diMjRLSUNBZ0lDOHZJSFJv'
    'YVhNZ1kyOWtaU0J3WVhSb0lHVnhkV0ZzY3lCMGNtbG5UbTkwWlM1d1pYSnBiMlFnNG9DVUlI'
    'Um9aU0J3WVhKMGFXRnNMWEJwZEdOb0lHSnNiMk5ySUdScFpHNG5kQW9nSUNBZ0x5OGdjblZ1'
    'SUdsbUlHNXZJSE5zYVdSbElHVm1abVZqZEN3Z2IzSWdhWFFnY21GdUlHWnliMjBnZEhKcFow'
    'NXZkR1V1Y0dWeWFXOWtJR0Z6SUhOMFlYSjBhVzVuSUhCdmFXNTBLUzRLSUNBZ0lDOHZJRTV2'
    'SUdGa1pHbDBhVzl1WVd3Z1kyOWtaU0J1WldWa1pXUTZJR0ZqWTNWdGRXeGhkRzl5SUdseklH'
    'TnZjbkpsWTNRdUNnb2dJQ0FnTHk4Z1ZISmxiVzlzYnlBb1JXWm1aV04wSURCNE55a2c0b0NV'
    'SUhOaGJXVWdkMkYyWldadmNtMGdZWE1nZG1saWNtRjBieUJpZFhRZ2JXOWtkV3hoZEdWeklG'
    'WlBURlZOUlM0S0lDQWdJQzh2SUZWelpYTWdjMkZ0WlNCeWIzY3RZbmt0Y205M0lHaHBjM1J2'
    'Y21sallXd2dkRk12ZEVRZ2RISmhZMnRwYm1jZ1lYTWdkbWxpY21GMGJ5NEtJQ0FnSUhzS0lD'
    'QWdJQ0FnSUNCcGJuUWdYM1JUSUQwZ01Dd2dYM1JFSUQwZ01Ec0tJQ0FnSUNBZ0lDQnBiblFn'
    'WDNSeVpWQnZjeUE5SURBN0NpQWdJQ0FnSUNBZ2FXWWdLSFJ5YVdkT2IzUmxMbVZtWm1WamRD'
    'QTlQU0F3ZURjcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOXVjeUE5SUNoMGNtbG5UbTkw'
    'WlM1d1lYSmhiU0ErUGlBMEtTQW1JREI0UmpzS0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5dVpD'
    'QTlJQ0IwY21sblRtOTBaUzV3WVhKaGJTQWdJQ0FnSUNBbUlEQjRSanNLSUNBZ0lDQWdJQ0Fn'
    'SUNBZ2FXWWdLRjl1Y3lBK0lEQXBJRjkwVXlBOUlGOXVjenNLSUNBZ0lDQWdJQ0FnSUNBZ2FX'
    'WWdLRjl1WkNBK0lEQXBJRjkwUkNBOUlGOXVaRHNLSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0Fn'
    'YVdZZ0tIUnlhV2RRWVhRZ1BUMGdjRzl6TG5OdmJtZFFiM01nSmlZZ2RISnBaMUp2ZHlBOVBT'
    'QndiM011Y205M0tTQjdDaUFnSUNBZ0lDQWdJQ0FnSUY5MGNtVlFiM01nUFNCcGJuUW9jRzl6'
    'TG5ScFkyc3BJQ29nWDNSVE93b2dJQ0FnSUNBZ0lIMGdaV3h6WlNCN0NpQWdJQ0FnSUNBZ0lD'
    'QWdJR2x1ZENCZmRISlRaM0lnUFNCd1lYUlVhV05yVDJabWMyVjBXM1J5YVdkUVlYUmRJQ3Nn'
    'S0hSeWFXZFNiM2NnTFNCd1lYUlRkR0Z5ZEZKdmQxdDBjbWxuVUdGMFhTazdDaUFnSUNBZ0lD'
    'QWdJQ0FnSUdsdWRDQmZkSEpUY0dRZ1BTQm1aWFJqYUZScFkyc29YM1J5VTJkeUlDc2dNU2tn'
    'TFNCbVpYUmphRlJwWTJzb1gzUnlVMmR5S1RzS0lDQWdJQ0FnSUNBZ0lDQWdYM1J5WlZCdmN5'
    'QTlJQ2hmZEhKVGNHUWdMU0F4S1NBcUlGOTBVenNLSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjkz'
    'Y0NBOUlIUnlhV2RRWVhRc0lGOTNjaUE5SUhSeWFXZFNiM2NnS3lBeE93b2dJQ0FnSUNBZ0lD'
    'QWdJQ0JwWmlBb1gzZHlJRDQ5SUhCaGRGTjBZWEowVW05M1cxOTNjRjBnS3lBb2NHRjBVbTkz'
    'VDJabWMyVjBXMTkzY0NzeFhTQXRJSEJoZEZKdmQwOW1abk5sZEZ0ZmQzQmRLU2tnZXdvZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnWDNkd0t5czdJRjkzY2lBOUlDaGZkM0FnUENCVFQwNUhYMHhG'
    'VGtkVVNDa2dQeUJ3WVhSVGRHRnlkRkp2ZDF0ZmQzQmRJRG9nTURzS0lDQWdJQ0FnSUNBZ0lD'
    'QWdmUW9nSUNBZ0lDQWdJQ0FnSUNCbWIzSWdLR2x1ZENCZmQya2dQU0F3T3lCZmQya2dQQ0F4'
    'TWpnN0lGOTNhU3NyS1NCN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9YM2R3SUQ0Z2NH'
    'OXpMbk52Ym1kUWIzTWdmSHdnS0Y5M2NDQTlQU0J3YjNNdWMyOXVaMUJ2Y3lBbUppQmZkM0ln'
    'UGowZ2NHOXpMbkp2ZHlrcElHSnlaV0ZyT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tG'
    'OTNjQ0ErUFNCVFQwNUhYMHhGVGtkVVNDa2dZbkpsWVdzN0NpQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNCcFppQW9YM2R5SUQ0OUlIQmhkRk4wWVhKMFVtOTNXMTkzY0YwZ0t5QW9jR0YwVW05M1Qy'
    'Wm1jMlYwVzE5M2NDc3hYU0F0SUhCaGRGSnZkMDltWm5ObGRGdGZkM0JkS1NrZ2V3b2dJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJRjkzY0Nzck95QmZkM0lnUFNBb1gzZHdJRHdnVTA5T1Ix'
    'OU1SVTVIVkVncElEOGdjR0YwVTNSaGNuUlNiM2RiWDNkd1hTQTZJREE3Q2lBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ1kyOXVkR2x1ZFdVN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCOUNp'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCT2IzUmxJRjkwYmlBOUlHZGxkRTV2ZEdVb1gzZHdMQ0Jm'
    'ZDNJc0lHTm9LVHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR0p2YjJ3Z1gzUnVTWE5VYjI1bElE'
    'MGdLQ2hmZEc0dVpXWm1aV04wSUQwOUlEQjRNeUI4ZkNCZmRHNHVaV1ptWldOMElEMDlJREI0'
    'TlNrZ0ppWWdYM1J1TG5CbGNtbHZaQ0ErSURBcE93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FX'
    'WWdLRjkwYmk1d1pYSnBiMlFnUGlBd0lDWW1JQ0ZmZEc1SmMxUnZibVVwSUdKeVpXRnJPd29n'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5MGJpNWxabVpsWTNRZ1BUMGdNSGczSUNZbUlG'
    'OTBiaTV3WVhKaGJTQWhQU0F3S1NCN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhVzUw'
    'SUY5dWN5QTlJQ2hmZEc0dWNHRnlZVzBnUGo0Z05Da2dKaUF3ZUVZN0NpQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdhVzUwSUY5dVpDQTlJQ0JmZEc0dWNHRnlZVzBnSUNBZ0lDQWdKaUF3'
    'ZUVZN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5dWN5QStJREFwSUY5MFV5'
    'QTlJRjl1Y3pzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvWDI1a0lENGdNQ2tn'
    'WDNSRUlEMGdYMjVrT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnYVc1MElGOXpaM0lnUFNCd1lYUlVhV05yVDJabWMyVjBXMTkzY0YwZ0t5QW9YM2R5'
    'SUMwZ2NHRjBVM1JoY25SU2IzZGJYM2R3WFNrN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcGJu'
    'UWdYM053WkNBOUlHWmxkR05vVkdsamF5aGZjMmR5SUNzZ01Ta2dMU0JtWlhSamFGUnBZMnNv'
    'WDNObmNpazdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmZkSEpsVUc5eklDczlJQ2hmYzNCa0lD'
    'MGdNU2tnS2lCZmRGTTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmZkM0lyS3pzS0lDQWdJQ0Fn'
    'SUNBZ0lDQWdmUW9nSUNBZ0lDQWdJQ0FnSUNBdkx5QlZjR1JoZEdVZ1puSnZiU0JqZFhKeVpX'
    'NTBJSEp2ZHlCcFppQnBkQ0JqWVhKeWFXVnpJSFJ5WlcxdmJHOEtJQ0FnSUNBZ0lDQWdJQ0Fn'
    'YVdZZ0tGOXdZM0l1WldabVpXTjBJRDA5SURCNE55QW1KaUJmY0dOeUxuQmhjbUZ0SUNFOUlE'
    'QXBJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2x1ZENCZmJuTWdQU0FvWDNCamNpNXdZWEpo'
    'YlNBK1BpQTBLU0FtSURCNFJqc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0JmYm1RZ1BT'
    'QWdYM0JqY2k1d1lYSmhiU0FnSUNBZ0lDQW1JREI0UmpzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUdsbUlDaGZibk1nUGlBd0tTQmZkRk1nUFNCZmJuTTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QnBaaUFvWDI1a0lENGdNQ2tnWDNSRUlEMGdYMjVrT3dvZ0lDQWdJQ0FnSUNBZ0lDQjlDaUFn'
    'SUNBZ0lDQWdJQ0FnSUY5MGNtVlFiM01nS3owZ2FXNTBLSEJ2Y3k1MGFXTnJLU0FxSUY5MFV6'
    'c0tJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdhV1lnS0Y5MFJDQStJREFnSmlZZ1gzUlRJRDRn'
    'TUNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0JwYm5RZ1gzUlFJRDBnWDNSeVpWQnZjeUFtSURZek93'
    'b2dJQ0FnSUNBZ0lDQWdJQ0JtYkc5aGRDQmZkRVJsYkhSaElEMGdLSFpwWWxSaFlsdGZkRkFn'
    'SmlBek1WMGdLaUJtYkc5aGRDaGZkRVFwS1NBdklEWTBMakE3Q2lBZ0lDQWdJQ0FnSUNBZ0lD'
    'OHZJRlJ5WlcxdmJHOGdiVzlrYVdacFpYTWdkR2hsSUU5VlZGQlZWQ0JoYlhCc2FYUjFaR1Vn'
    'Y0dWeUxYTmhiWEJzWlRzZ2FYUWdaRzlsYzI0bmRBb2dJQ0FnSUNBZ0lDQWdJQ0F2THlCblpY'
    'UWdjMjF2YjNSb1pXUWdkbWxoSUZaUFRGOVRSVlFnWW1WallYVnpaU0JwZENkeklHRnNjbVZo'
    'WkhrZ1lTQm1ZWE4wSUc5elkybHNiR0YwYVc5dUNpQWdJQ0FnSUNBZ0lDQWdJQzh2SUNoemJX'
    'OXZkR2hwYm1jZ2QyOTFiR1FnYzNWd2NISmxjM01nYVhRcExpQlRkRzl5WldRZ1lYTWdZU0Jr'
    'Wld4MFlTQmhibVFnWVdSa1pXUWdkRzhLSUNBZ0lDQWdJQ0FnSUNBZ0x5OGdkR2hsSUhOdGIy'
    'OTBhR1ZrSUhadmJIVnRaU0JoZENCdmRYUndkWFFnZEdsdFpTNEtJQ0FnSUNBZ0lDQWdJQ0Fn'
    'WDNSeVpXMXZiRzlFWld4MFlTQTlJQ2hmZEZBZ1BDQXpNaWtnUHlCZmRFUmxiSFJoSURvZ0xW'
    'OTBSR1ZzZEdFN0NpQWdJQ0FnSUNBZ2ZRb2dJQ0FnZlFvS0lDQWdJQzh2SUVGeWNHVm5aMmx2'
    'SUNoRlptWmxZM1FnTUhoNUtTRGlnSlFnY0hsdGIyUW5jeUJ2Y21SbGNpQnBjeUJpWVhObDRv'
    'YVNXQ2hvYVdkb0tlS0drbGtvYkc5M0tRb2dJQ0FnYVdZZ0tGOXdZM0l1WldabVpXTjBJRDA5'
    'SURCNE1DQW1KaUJmY0dOeUxuQmhjbUZ0SUNFOUlEQXBJSHNLSUNBZ0lDQWdJQ0JwYm5RZ1gy'
    'RnljRk4wWlhBZ1BTQnBiblFvY0c5ekxuUnBZMnNwSUMwZ2FXNTBLSEJ2Y3k1MGFXTnJJQzhn'
    'TXk0d0tTQXFJRE03Q2lBZ0lDQWdJQ0FnTHk4Z1pXWm1aV04wYVhabFVHVnlhVzlrSUVsVElH'
    'SmhjMlZRWlhKcGIyUWdhR1Z5WlNBb2JtOGdablZ5ZEdobGNpQnRiMlJwWm1sallYUnBiMjRn'
    'WW1WbWIzSmxJR0Z5Y0NrS0lDQWdJQ0FnSUNCcFppQW9YMkZ5Y0ZOMFpYQWdQVDBnTVNrS0lD'
    'QWdJQ0FnSUNBZ0lDQWdaV1ptWldOMGFYWmxVR1Z5YVc5a0lEMGdaV1ptWldOMGFYWmxVR1Z5'
    'YVc5a0lDb2djRzkzS0RJdU1Dd2dMV1pzYjJGMEtDaGZjR055TG5CaGNtRnRJRDQrSURRcElD'
    'WWdNSGhHS1NBdklERXlMakFwT3dvZ0lDQWdJQ0FnSUdWc2MyVWdhV1lnS0Y5aGNuQlRkR1Z3'
    'SUQwOUlESXBDaUFnSUNBZ0lDQWdJQ0FnSUdWbVptVmpkR2wyWlZCbGNtbHZaQ0E5SUdWbVpt'
    'VmpkR2wyWlZCbGNtbHZaQ0FxSUhCdmR5Z3lMakFzSUMxbWJHOWhkQ2hmY0dOeUxuQmhjbUZ0'
    'SUNZZ01IaEdLU0F2SURFeUxqQXBPd29nSUNBZ2ZRb0tJQ0FnSUM4dklGWnBZbkpoZEc4Z0tF'
    'Vm1abVZqZENBMEtTRGlnSlFnZFhObGN5Qm5iRzlpWVd3Z2RtbGlWR0ZpTGdvZ0lDQWdMeThn'
    'UldabVpXTjBJRFI0ZURvZ2NHRnlZVzBnUFNBb2MzQmxaV1FnUER3Z05Da2dmQ0JrWlhCMGFD'
    'NGdJRk5sZEhNZ2RsTXNJSFpFTGdvZ0lDQWdMeThnUldabVpXTjBJRFo0ZURvZ1kyOXVkR2x1'
    'ZFdVZ2RtbGljbUYwYnlBb2RYTmxjeUJ3Y21sdmNpQTBlSGduY3lCMlV5OTJSRHNnYVhSeklH'
    'OTNiaUJ3WVhKaGJTQnBjd29nSUNBZ0x5OGdJQ0FnSUNBZ0lDQWdJQ0FnZG05c0xYTnNhV1Js'
    'SUc5dWJIa2c0b0NVSUdoaGJtUnNaV1FnYzJWd1lYSmhkR1ZzZVNCcGJpQjJiMngxYldVZ1ky'
    'OWtaU0J3WVhSb0tTNEtJQ0FnSUM4dkNpQWdJQ0F2THlCMmFXSnlZWFJ2VUc5eklHbHVZM0ps'
    'YldWdWRITWdZbmtnZGxNZ2IyNGdaV0ZqYUNCT1QwNHRkR2xqYXkwd0lDaHBMbVV1TENBb2Mz'
    'QmxaV1F0TVNrZ2NHVnlJSEp2ZHlrdUNpQWdJQ0F2THlCWFlXeHJJR1p5YjIwZ2RISnBaMmRs'
    'Y2lCMGJ5QmpkWEp5Wlc1MExDQjFjR1JoZEdsdVp5QjJVeTkyUkNCUFRreFpJRzl1SURSNGVD'
    'QnliM2R6TENCaGJtUUtJQ0FnSUM4dklHRmpZM1Z0ZFd4aGRHbHVaeUFvYzNCbFpXUXRNU2tx'
    'ZGxNZ2NHVnlJR052YlhCc1pYUmxaQ0J5YjNjZ2RYTnBibWNnYUdsemRHOXlhV05oYkNCMlV5'
    'NEtJQ0FnSUhzS0lDQWdJQ0FnSUNCcGJuUWdYM1pUSUQwZ01Dd2dYM1pFSUQwZ01Ec0tJQ0Fn'
    'SUNBZ0lDQnBiblFnWDNacFlsQnZjeUE5SURBN0Nnb2dJQ0FnSUNBZ0lDOHZJRWx1YVhScFlX'
    'eHBlbVVnZGxNdmRrUWdUMDVNV1NCbWNtOXRJSFJ5YVdkblpYSWdjbTkzSjNNZ01IZzBJQ2hP'
    'VDFRZ01IZzJJT0tBbENCcGRITWdjR0Z5WVcwZ2FYTWdkbTlzTFhOc2FXUmxLUW9nSUNBZ0lD'
    'QWdJR2xtSUNoMGNtbG5UbTkwWlM1bFptWmxZM1FnUFQwZ01IZzBLU0I3Q2lBZ0lDQWdJQ0Fn'
    'SUNBZ0lHbHVkQ0JmYm5NZ1BTQW9kSEpwWjA1dmRHVXVjR0Z5WVcwZ1BqNGdOQ2tnSmlBd2VF'
    'WTdDaUFnSUNBZ0lDQWdJQ0FnSUdsdWRDQmZibVFnUFNBZ2RISnBaMDV2ZEdVdWNHRnlZVzBn'
    'SUNBZ0lDQWdKaUF3ZUVZN0NpQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZmJuTWdQaUF3S1NCZmRs'
    'TWdQU0JmYm5NN0NpQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZmJtUWdQaUF3S1NCZmRrUWdQU0Jm'
    'Ym1RN0NpQWdJQ0FnSUNBZ2ZRb0tJQ0FnSUNBZ0lDQnBaaUFvZEhKcFoxQmhkQ0E5UFNCd2Iz'
    'TXVjMjl1WjFCdmN5QW1KaUIwY21sblVtOTNJRDA5SUhCdmN5NXliM2NwSUhzS0lDQWdJQ0Fn'
    'SUNBZ0lDQWdMeThnVDI0Z2RISnBaMmRsY2lCeWIzYzZJSFpwWW5KaGRHOGdhR0Z6SUc5dWJI'
    'a2dhR0ZrSUhCdmN5NTBhV05ySUdsdVkzSmxiV1Z1ZEhNS0lDQWdJQ0FnSUNBZ0lDQWdYM1pw'
    'WWxCdmN5QTlJR2x1ZENod2IzTXVkR2xqYXlrZ0tpQmZkbE03Q2lBZ0lDQWdJQ0FnZlNCbGJI'
    'TmxJSHNLSUNBZ0lDQWdJQ0FnSUNBZ0x5OGdWSEpwWjJkbGNpQnliM2NnWTI5dWRISnBZblYw'
    'WlhNZ0tITndaV1ZrTFRFcElHbHVZM0psYldWdWRITWdZWFFnZEhKcFoyZGxjaTF5YjNjZ2Rs'
    'TUtJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOTBjbE5uY2lBOUlIQmhkRlJwWTJ0UFptWnpaWFJi'
    'ZEhKcFoxQmhkRjBnS3lBb2RISnBaMUp2ZHlBdElIQmhkRk4wWVhKMFVtOTNXM1J5YVdkUVlY'
    'UmRLVHNLSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjkwY2xOd1pDQTlJR1psZEdOb1ZHbGpheWhm'
    'ZEhKVFozSWdLeUF4S1NBdElHWmxkR05vVkdsamF5aGZkSEpUWjNJcE93b2dJQ0FnSUNBZ0lD'
    'QWdJQ0F2THlCV2FXSnlZWFJ2SUd0bFpYQnpJSEoxYm01cGJtY2diMjRnTUhnMklISnZkM01n'
    'ZEc5dklPS0FsQ0JoWTJOMWJYVnNZWFJsSUNoemNHVmxaQzB4S1NwMlV3b2dJQ0FnSUNBZ0lD'
    'QWdJQ0F2THlCbGRtVnVJSGRvWlc0Z2RHaHBjeUJ5YjNjZ2QyRnpJREI0Tml3Z2RYTnBibWNn'
    'ZEdobElHbHVhR1Z5YVhSbFpDQjJVeTRLSUNBZ0lDQWdJQ0FnSUNBZ1ltOXZiQ0JmZEhKcFow'
    'bHpWbWxpUVdOMGFYWmxJRDBnS0hSeWFXZE9iM1JsTG1WbVptVmpkQ0E5UFNBd2VEUWdmSHdn'
    'ZEhKcFowNXZkR1V1WldabVpXTjBJRDA5SURCNE5pazdDaUFnSUNBZ0lDQWdJQ0FnSUY5MmFX'
    'SlFiM01nUFNCZmRISnBaMGx6Vm1saVFXTjBhWFpsSUQ4Z0tGOTBjbE53WkNBdElERXBJQ29n'
    'WDNaVElEb2dNRHNLQ2lBZ0lDQWdJQ0FnSUNBZ0lDOHZJRmRoYkdzZ2NtOTNMV0o1TFhKdmR5'
    'Qm1jbTl0SUhSeWFXZG5aWElyTVNCMGJ5QmpkWEp5Wlc1MExURXNJSFZ3WkdGMGFXNW5JSFpU'
    'TDNaRUNpQWdJQ0FnSUNBZ0lDQWdJQzh2SUc5dUlEQjROQ0J5YjNkekxDQmhibVFnWVdOamRX'
    'MTFiR0YwYVc1bklIQmxjaTF5YjNjZ2RYTnBibWNnYUdsemRHOXlhV05oYkNCMlV5NEtJQ0Fn'
    'SUNBZ0lDQWdJQ0FnYVc1MElGOTNjQ0E5SUhSeWFXZFFZWFFzSUY5M2NpQTlJSFJ5YVdkU2Iz'
    'Y2dLeUF4T3dvZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFvWDNkeUlENDlJSEJoZEZOMFlYSjBVbTkz'
    'VzE5M2NGMGdLeUFvY0dGMFVtOTNUMlptYzJWMFcxOTNjQ3N4WFNBdElIQmhkRkp2ZDA5bVpu'
    'TmxkRnRmZDNCZEtTa2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdYM2R3S3lzN0lGOTNjaUE5'
    'SUNoZmQzQWdQQ0JUVDA1SFgweEZUa2RVU0NrZ1B5QndZWFJUZEdGeWRGSnZkMXRmZDNCZElE'
    'b2dNRHNLSUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0JtYjNJZ0tHbHVkQ0Jm'
    'ZDJrZ1BTQXdPeUJmZDJrZ1BDQXhNamc3SUY5M2FTc3JLU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0JwWmlBb1gzZHdJRDRnY0c5ekxuTnZibWRRYjNNZ2ZId2dLRjkzY0NBOVBTQndiM011'
    'YzI5dVoxQnZjeUFtSmlCZmQzSWdQajBnY0c5ekxuSnZkeWtwSUdKeVpXRnJPd29nSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdhV1lnS0Y5M2NDQStQU0JUVDA1SFgweEZUa2RVU0NrZ1luSmxZV3M3'
    'Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwWmlBb1gzZHlJRDQ5SUhCaGRGTjBZWEowVW05M1cx'
    'OTNjRjBnS3lBb2NHRjBVbTkzVDJabWMyVjBXMTkzY0NzeFhTQXRJSEJoZEZKdmQwOW1abk5s'
    'ZEZ0ZmQzQmRLU2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lGOTNjQ3NyT3lCZmQz'
    'SWdQU0FvWDNkd0lEd2dVMDlPUjE5TVJVNUhWRWdwSUQ4Z2NHRjBVM1JoY25SU2IzZGJYM2R3'
    'WFNBNklEQTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnWTI5dWRHbHVkV1U3Q2lBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JPYjNSbElGOTJiaUE5'
    'SUdkbGRFNXZkR1VvWDNkd0xDQmZkM0lzSUdOb0tUc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'OHZJRk4wYjNBZ2IyNGdjbVYwY21sbloyVnlDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQmliMjlz'
    'SUY5MmJrbHpWRzl1WlNBOUlDZ29YM1p1TG1WbVptVmpkQ0E5UFNBd2VETWdmSHdnWDNadUxt'
    'Vm1abVZqZENBOVBTQXdlRFVwSUNZbUlGOTJiaTV3WlhKcGIyUWdQaUF3S1RzS0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUdsbUlDaGZkbTR1Y0dWeWFXOWtJRDRnTUNBbUppQWhYM1p1U1hOVWIy'
    'NWxLU0JpY21WaGF6c0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDOHZJRlZ3WkdGMFpTQjJVeTky'
    'UkNCUFRreFpJRzl1SURCNE5DQnliM2R6SUNnd2VEWWdhR0Z6SUhadmJDMXpiR2xrWlNCd1lY'
    'SmhiU3dnYm05MElIWnBZbkpoZEc4cENpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcFppQW9YM1p1'
    'TG1WbVptVmpkQ0E5UFNBd2VEUWdKaVlnWDNadUxuQmhjbUZ0SUNFOUlEQXBJSHNLSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcGJuUWdYMjV6SUQwZ0tGOTJiaTV3WVhKaGJTQStQaUEw'
    'S1NBbUlEQjRSanNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCcGJuUWdYMjVrSUQwZ0lG'
    'OTJiaTV3WVhKaGJTQWdJQ0FnSUNBbUlEQjRSanNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNCcFppQW9YMjV6SUQ0Z01Da2dYM1pUSUQwZ1gyNXpPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUdsbUlDaGZibVFnUGlBd0tTQmZka1FnUFNCZmJtUTdDaUFnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQjlDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQXZMeUJCWTJOMWJYVnNZWFJsSUhacFlu'
    'SmhkRzhnY0c5eklIZG9aVzRnY205M0lHbHpJREI0TkNCUFVpQXdlRFlnS0hacFluSmhkRzhn'
    'Y25WdWN5QnZiaUJpYjNSb0tRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjkyYmk1bFpt'
    'WmxZM1FnUFQwZ01IZzBJSHg4SUY5MmJpNWxabVpsWTNRZ1BUMGdNSGcyS1NCN0NpQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5elozSWdQU0J3WVhSVWFXTnJUMlptYzJWMFcx'
    'OTNjRjBnS3lBb1gzZHlJQzBnY0dGMFUzUmhjblJTYjNkYlgzZHdYU2s3Q2lBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJRjl6Y0dRZ1BTQm1aWFJqYUZScFkyc29YM05uY2lBcklE'
    'RXBJQzBnWm1WMFkyaFVhV05yS0Y5elozSXBPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUY5MmFXSlFiM01nS3owZ0tGOXpjR1FnTFNBeEtTQXFJRjkyVXpzS0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUY5M2Npc3JPd29nSUNBZ0lDQWdJQ0Fn'
    'SUNCOUNnb2dJQ0FnSUNBZ0lDQWdJQ0F2THlCVmNHUmhkR1VnZGxNdmRrUWdabkp2YlNCamRY'
    'SnlaVzUwSUhKdmR5QlBUa3haSUdsbUlEQjROQW9nSUNBZ0lDQWdJQ0FnSUNCcFppQW9YM0Jq'
    'Y2k1bFptWmxZM1FnUFQwZ01IZzBJQ1ltSUY5d1kzSXVjR0Z5WVcwZ0lUMGdNQ2tnZXdvZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnYVc1MElGOXVjeUE5SUNoZmNHTnlMbkJoY21GdElENCtJRFFw'
    'SUNZZ01IaEdPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUY5dVpDQTlJQ0JmY0dOeUxu'
    'QmhjbUZ0SUNBZ0lDQWdJQ1lnTUhoR093b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLRjl1'
    'Y3lBK0lEQXBJRjkyVXlBOUlGOXVjenNLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJR2xtSUNoZmJt'
    'UWdQaUF3S1NCZmRrUWdQU0JmYm1RN0NpQWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0x5OGdRM1Z5Y21WdWRDQnliM2NnY0dGeWRHbGhiRG9nY0c5ekxuUnBZMnNnYVc1amNt'
    'VnRaVzUwY3lCaGRDQmpkWEp5Wlc1MExYSnZkeUIyVXdvZ0lDQWdJQ0FnSUNBZ0lDQXZMeUJQ'
    'VGt4WklIZG9aVzRnWTNWeWNtVnVkQ0J5YjNjZ2FHRnpJSFpwWW5KaGRHOGdZV04wYVhabElD'
    'Z3dlRFFnYjNJZ01IZzJLUW9nSUNBZ0lDQWdJQ0FnSUNCcFppQW9YM0JqY2k1bFptWmxZM1Fn'
    'UFQwZ01IZzBJSHg4SUY5d1kzSXVaV1ptWldOMElEMDlJREI0TmlrZ2V3b2dJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ1gzWnBZbEJ2Y3lBclBTQnBiblFvY0c5ekxuUnBZMnNwSUNvZ1gzWlRPd29n'
    'SUNBZ0lDQWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ2ZRb0tJQ0FnSUNBZ0lDQnBaaUFvWDNaRUlE'
    'NGdNQ0FtSmlCZmRsTWdQaUF3S1NCN0NpQWdJQ0FnSUNBZ0lDQWdJR2x1ZENCZmRsQWdQU0Jm'
    'ZG1saVVHOXpJQ1lnTmpNN0NpQWdJQ0FnSUNBZ0lDQWdJR1pzYjJGMElGOTJSR1ZzZEdFZ1BT'
    'QW9kbWxpVkdGaVcxOTJVQ0FtSURNeFhTQXFJR1pzYjJGMEtGOTJSQ2twSUM4Z01USTRMakE3'
    'Q2lBZ0lDQWdJQ0FnSUNBZ0lHVm1abVZqZEdsMlpWQmxjbWx2WkNBclBTQW9YM1pRSUR3Z016'
    'SXBJRDhnWDNaRVpXeDBZU0E2SUMxZmRrUmxiSFJoT3dvZ0lDQWdJQ0FnSUgwS0lDQWdJSDBL'
    'Q2lBZ0lDQXZMeUJTWlc1a1pYSWdjMkZ0Y0d4bENpQWdJQ0F2THlCaVlYTmxVR1Z5YVc5a0lE'
    'MGdaV1ptWldOMGFYWmxVR1Z5YVc5a0lGZEpWRWhQVlZRZ2RtbGljbUYwYnk5MGNtVnRiMnh2'
    'SUcxdlpIVnNZWFJwYjI0dUlDQlZjMmx1WnlCMGFHVUtJQ0FnSUM4dklHMXZaSFZzWVhSbFpD'
    'QjJZV3gxWlNCbWIzSWdabE5oYlhCc1pWQnZjeUJwYm5SbFozSmhkR2x2YmlCM2IzVnNaQ0J0'
    'ZFd4MGFYQnNlU0IwYUdVZ2JXOWtkV3hoZEdsdmJnb2dJQ0FnTHk4Z1lXMXdiR2wwZFdSbElH'
    'SjVJR0JsYkdGd2MyVmtZQ3dnY0hKdlpIVmphVzVuSUdFZ2MzVmljM1JoYm5ScFlXd2dZblY2'
    'ZWlCaGRDQjJhV0p5WVhSdklISmhkR1VLSUNBZ0lDOHZJQ2hsTG1jdUxDQm1iSFYwWlNCdmJp'
    'QndZWFFnTVNCamFEQWdZWFFnZG1WalgyUnBiVDA0S1M0Z0lGUm9aU0JVVWxWRklIQnZjMmww'
    'YVc5dUxXUnZiV0ZwYmlCbFptWmxZM1FLSUNBZ0lDOHZJRzltSUhacFluSmhkRzhnYVhNZ2RH'
    'aGxJR2x1ZEdWbmNtRnNJRzltSUhSb1pTQm1jbVZ4SUcxdlpIVnNZWFJwYjI0c0lIZG9hV05v'
    'SUdseklHRWdkR2x1ZVNBOE1TMEtJQ0FnSUM4dklITmhiWEJzWlNCdmMyTnBiR3hoZEdsdmJp'
    'RGlnSlFnYzJGbVpXeDVJRzVsWjJ4cFoybGliR1V1Q2lBZ0lDQm1iRzloZENCaVlYTmxVR1Z5'
    'YVc5a0lEMGdaV1ptWldOMGFYWmxVR1Z5YVc5a093b2dJQ0FnYVdZZ0tIUnlhV2RPYjNSbExt'
    'Vm1abVZqZENBOVBTQXdlRFFnZkh3Z2RISnBaMDV2ZEdVdVpXWm1aV04wSUQwOUlEQjROaUI4'
    'ZkNCMGNtbG5UbTkwWlM1bFptWmxZM1FnUFQwZ01IZzNJSHg4Q2lBZ0lDQWdJQ0FnWDNCamNp'
    'NWxabVpsWTNRZ1BUMGdNSGcwSUh4OElGOXdZM0l1WldabVpXTjBJRDA5SURCNE5pQjhmQ0Jm'
    'Y0dOeUxtVm1abVZqZENBOVBTQXdlRGNwSUhzS0lDQWdJQ0FnSUNCaVlYTmxVR1Z5YVc5a0lE'
    'MGdLSFJoY21kbGRGQmxjbWx2WkNBK0lEQXVNQ2tnUHlCMFlYSm5aWFJRWlhKcGIyUWdPaUJt'
    'Ykc5aGRDaDBjbWxuVG05MFpTNXdaWEpwYjJRcE93b2dJQ0FnZlFvZ0lDQWdabXh2WVhRZ1pu'
    'SmxjU0E5SUhCbGNtbHZaRlJ2Um5KbGNVWjBLRzFoZUNneExDQnBiblFvWW1GelpWQmxjbWx2'
    'WkNrcExDQnpiWEF1Wm1sdVpYUjFibVVwT3dvZ0lDQWdMeThnVTJGdGNHeGxJSEJ2YzJsMGFX'
    'OXVPZ29nSUNBZ0x5OGdJQ0F0SUVsbUlHTjFjbkpsYm5RZ2NtOTNJR2hoY3lCaFkzUnBkbVVn'
    'Y0dsMFkyZ2djMnhwWkdVZ0tERjRlQzh5ZUhndk0zaDRLU3dnZFhObElHeHZaeUJwYm5SbFoz'
    'SmhiQW9nSUNBZ0x5OGdJQ0FnSU9LSXEwTXZVQ2gwS1dSMElPS0ppQ0JEdzVkVUw4NlVVQ0RE'
    'bHlCc2JpaFFNUzlRTUNrZ0lDaGhjM04xYldWeklHeHBibVZoY2lCeVlXMXdPeUJqYkc5elpT'
    'QmxibTkxWjJncENpQWdJQ0F2THlBZ0lDMGdUM1JvWlhKM2FYTmxJSEJsY21sdlpDQnBjeUJ6'
    'ZEdGaWJHVWdjR0Z6ZENCMGNtbG5aMlZ5T3lCemFXMXdiR1VnWld4aGNITmxaTU9YWm5KbGNT'
    'QnBjeUJsZUdGamRDNEtJQ0FnSUM4dklGTmhiWEJzWlNCd2IzTnBkR2x2YmlCbWNtOXRJSFJv'
    'WlNCd1pYSXRjbTkzSUdWNFlXTjBJR2x1ZEdWbmNtRjBiM0lnS0hKbGNHeGhZMlZ6SUhSb1pT'
    'QnZiR1FLSUNBZ0lDOHZJSE5wYm1kc1pTMXpaV2R0Wlc1MElHWnZjbTExYkdFZ2QyaHBZMmdn'
    'ZDJGeklIZHliMjVuSUdadmNpQnRkV3gwYVMxeWIzY2djMnhwWkdWeklPS0FsQ0J6WldVS0lD'
    'QWdJQzh2SUY5bVUyRnRjR3hsVUc5elFXTmpJR052Ym5OMGNuVmpkR2x2YmlCaFltOTJaU2t1'
    'SUZKbGMzVnNkQ0JwY3lCcGJpQnpiM1Z5WTJVdGNtRjBaU0J6WVcxd2JHVnpPd29nSUNBZ0x5'
    'OGdaR2wyYVdSbElHSjVJR0ozUm1GamRHOXlJSFJ2SUdOdmJuWmxjblFnZEc4Z1kyOXRjSEps'
    'YzNObFpDMWtiMjFoYVc0Z2MyRnRjR3hsY3lCc2FXdGxJSFJvWlFvZ0lDQWdMeThnYkdWbllX'
    'TjVJR052WkdVZ1pHbGtMZ29nSUNBZ1pteHZZWFFnWmxOaGJYQnNaVkJ2Y3lBOUlGOW1VMkZ0'
    'Y0d4bFVHOXpRV05qSUM4Z1pteHZZWFFvYzIxd0xtSjNSbUZqZEc5eUtUc0tJQ0FnSUM4dklE'
    'QjRPWGg0SUhOaGJYQnNaU0J2Wm1aelpYUTZJSE5vYVdaMElITjBZWEowYVc1bklIQnZjMmww'
    'YVc5dUlDaHBiaUJqYjIxd2NtVnpjMlZrTFdSdmJXRnBiaUJ6WVcxd2JHVnpLUW9nSUNBZ2FX'
    'WWdLRjl6WVcxd2JHVlBabVp6WlhRZ1BpQXdLU0I3Q2lBZ0lDQWdJQ0FnWmxOaGJYQnNaVkJ2'
    'Y3lBclBTQm1iRzloZENoZmMyRnRjR3hsVDJabWMyVjBJQzhnYldGNEtERXNJSE50Y0M1aWQw'
    'WmhZM1J2Y2lrcE93b2dJQ0FnZlFvS0lDQWdJR2xtSUNoemJYQXViRzl2Y0V4bGJpQStJRElw'
    'SUhzS0lDQWdJQ0FnSUNCcFppQW9abE5oYlhCc1pWQnZjeUErUFNCbWJHOWhkQ2h6YlhBdWJH'
    'OXZjRk4wWVhKMElDc2djMjF3TG14dmIzQk1aVzRwS1FvZ0lDQWdJQ0FnSUNBZ0lDQm1VMkZ0'
    'Y0d4bFVHOXpJRDBnWm14dllYUW9jMjF3TG14dmIzQlRkR0Z5ZENrZ0t5QnRiMlFvWmxOaGJY'
    'QnNaVkJ2Y3lBdElHWnNiMkYwS0hOdGNDNXNiMjl3VTNSaGNuUXBMQ0JtYkc5aGRDaHpiWEF1'
    'Ykc5dmNFeGxiaWtwT3dvZ0lDQWdmU0JsYkhObElHbG1JQ2htVTJGdGNHeGxVRzl6SUQ0OUlH'
    'WnNiMkYwS0hOdGNDNXNaVzVuZEdncEtTQjdDaUFnSUNBZ0lDQWdjbVYwZFhKdUlEQXVNRHNL'
    'SUNBZ0lIMEtJQ0FnSUdsbUlDaG1VMkZ0Y0d4bFVHOXpJRHdnTUM0d0tTQnlaWFIxY200Z01D'
    'NHdPd29LSUNBZ0lDOHZJRk5oYlhCc1pTQjJZV3gxWlNCM2FYUm9JSEJ5YjNCbGNpQmxibVF0'
    'Wm1Ga1pTQW9jMkZ0Y0d4bElIUmxjbTFwYm1GMGFXOXVJSE5vYjNWc1pDQnViM1FnYzI1aGND'
    'QjBieUF3S1FvZ0lDQWdabXh2WVhRZ2N6c0tJQ0FnSUdsbUlDaHpiWEF1Ykc5dmNFeGxiaUE4'
    'UFNBeUlDWW1JR1pUWVcxd2JHVlFiM01nUGowZ1pteHZZWFFvYzIxd0xteGxibWQwYUNrZ0xT'
    'QXhMakFwSUhzS0lDQWdJQ0FnSUNBdkx5Qk9aV0Z5SUdWdVpDQnZaaUJ1YjI0dGJHOXZjR2x1'
    'WnlCellXMXdiR1U2SUdaaFpHVWdiM1YwSUc5MlpYSWdiR0Z6ZENCellXMXdiR1VnZEc4Z1lY'
    'WnZhV1FnWTJ4cFkyc0tJQ0FnSUNBZ0lDQnpJRDBnTUM0d093b2dJQ0FnZlNCbGJITmxJSHNL'
    'SUNBZ0lDQWdJQ0J6SUQwZ1oyVjBVMkZ0Y0d4bFJpaHpiWEF1YzNSaGNuUXNJR1pUWVcxd2JH'
    'VlFiM01zSUhOdGNDNXNaVzVuZEdnc0lITnRjQzVzYjI5d1UzUmhjblFzSUhOdGNDNXNiMjl3'
    'VEdWdUtUc0tJQ0FnSUgwS0NpQWdJQ0F2THlEaWxJRGlsSUFnUVc1MGFTMWpiR2xqYXlCeVlX'
    'MXdjeURpbElEaWxJQUtJQ0FnSUM4dklERXVJRlJ5YVdkblpYSWdjbUZ0Y0RvZ1FVUkJVRlJK'
    'VmtVZ1ptRmtaUzFwYmk0S0lDQWdJQzh2SUNBZ0lFUmxabUYxYkhRNklEWTBMWE5oYlhCc1pT'
    'QnNhVzVsWVhJZ0tHMXBhMGxVSUdaaFpHVmpiM1Z1ZENrZzRvQ1VJSE5vWVhKd0lHUnlkVzBn'
    'WVhSMFlXTnJMZ29nSUNBZ0x5OGdJQ0FnVTJGdGNHeGxMVzltWm5ObGRDQnlaWFJ5YVdkblpY'
    'SnpJQ2c1ZUhncE9pQXhPVEl0YzJGdGNHeGxJSE50YjI5MGFITjBaWEFnNG9DVUlHMWhjMnR6'
    'SUhSb1pRb2dJQ0FnTHk4Z0lDQWdiV2xrTFhkaGRtVm1iM0p0SUdScGMyTnZiblJwYm5WcGRI'
    'a2dkR2hoZENCallYVnpaWE1nWTJ4cFkydHpJRzl1SUdSeWRXMHRZMmh2Y0hCcGJtY0tJQ0Fn'
    'SUM4dklDQWdJSEJoZEhSbGNtNXpMaUJFY25WdGN5QmphRzl3Y0dWa0lIWnBZU0E1ZUhnZ2Mz'
    'UmhjblFnWVhRZ2JtOXVMWHBsY204Z1lXMXdiR2wwZFdSbElHbHVjMmxrWlFvZ0lDQWdMeThn'
    'SUNBZ2RHaGxJSE5oYlhCc1pTd2dZVzVrSUhSb1pTQndjbVYyYVc5MWN5QnViM1JsSjNNZ2RH'
    'RnBiQ0JxZFhOMElITjBiM0J6TENCemJ5QjNhWFJvYjNWMENpQWdJQ0F2THlBZ0lDQmhJR3h2'
    'Ym1kbGNpQnlZVzF3SUdWMlpYSjVJSEpsZEhKcFoyZGxjaUJ3YjNCeklHRjFaR2xpYkhrdUNp'
    'QWdJQ0JtYkc5aGRDQmtaV05zYVdOck93b2dJQ0FnYVdZZ0tGOXpZVzF3YkdWUFptWnpaWFFn'
    'UGlBd0tTQjdDaUFnSUNBZ0lDQWdMeThnVTIxdmIzUm9jM1JsY0NCdmRtVnlJREU1TWlCellX'
    'MXdiR1Z6SUNoK05DNDBiWE1nUUNBME5DNHhhMGg2S1M0Z1UyMXZiM1JvYzNSbGNDQm9ZWE1L'
    'SUNBZ0lDQWdJQ0F2THlCNlpYSnZJR1JsY21sMllYUnBkbVVnWVhRZ1ltOTBhQ0JsYm1Sd2Iy'
    'bHVkSE1nNG9DVUlHNXZJR0YxWkdsaWJHVWdhMmx1YXlCaGRDQjBhR1VnYzNSaGNuUUtJQ0Fn'
    'SUNBZ0lDQXZMeUJ2Y2lCbGJtUWdiMllnZEdobElHWmhaR1VzSUdwMWMzUWdZU0J6Ylc5dmRH'
    'Z2djM2RsYkd3dUlFeHZibWNnWlc1dmRXZG9JSFJ2SUcxaGMyc0tJQ0FnSUNBZ0lDQXZMeUIw'
    'YUdVZ2JXbGtMWGRoZG1WbWIzSnRJSE4wWVhKMElHUnBjMk52Ym5ScGJuVnBkSGtzSUhOb2Iz'
    'SjBJR1Z1YjNWbmFDQjBieUJ3Y21WelpYSjJaUW9nSUNBZ0lDQWdJQzh2SUhCbGNtTmxhWFps'
    'WkNCaGRIUmhZMnNnYjI0Z2MyeHZkeUJrY25WdElHaHBkSE11Q2lBZ0lDQWdJQ0FnWm14dllY'
    'UWdkQ0E5SUdOc1lXMXdLR1ZzWVhCelpXUWdLaUFvTkRReE1EQXVNQ0F2SURFNU1pNHdLU3dn'
    'TUM0d0xDQXhMakFwT3dvZ0lDQWdJQ0FnSUdSbFkyeHBZMnNnUFNCMElDb2dkQ0FxSUNnekxq'
    'QWdMU0F5TGpBZ0tpQjBLVHNLSUNBZ0lIMGdaV3h6WlNCN0NpQWdJQ0FnSUNBZ0x5OGdVMmho'
    'Y25BZ05qUXRjMkZ0Y0d4bElHMXBhMGxVSUdSbFptRjFiSFFnWm05eUlHNXZjbTFoYkNCMGNt'
    'bG5aMlZ5TFdaeWIyMHRjMkZ0Y0d4bExUQXVDaUFnSUNBZ0lDQWdaR1ZqYkdsamF5QTlJR05z'
    'WVcxd0tHVnNZWEJ6WldRZ0tpQW9ORFF4TURBdU1DQXZJRFkwTGpBcExDQXdMakFzSURFdU1D'
    'azdDaUFnSUNCOUNnb2dJQ0FnTHk4Z01pNGdSVzVrTFc5bUxYTmhiWEJzWlNCbVlXUmxMVzkx'
    'ZERvZ05qUXRjMkZ0Y0d4bElHWmhaR1V0YjNWMElHRnpJR1pUWVcxd2JHVlFiM01nWVhCd2Nt'
    'OWhZMmhsY3dvZ0lDQWdMeThnSUNBZ2MyRnRjR3hsSUdWdVpDQW9iMjVzZVNCbWIzSWdibTl1'
    'TFd4dmIzQnBibWNnYzJGdGNHeGxjeWt1SUNCUWNtVjJaVzUwY3lCemRXUmtaVzRnYzJsc1pX'
    'NWpaUzRLSUNBZ0lHWnNiMkYwSUdWdVpFWmhaR1VnUFNBeExqQTdDaUFnSUNCcFppQW9jMjF3'
    'TG14dmIzQk1aVzRnUEQwZ01pa2dld29nSUNBZ0lDQWdJR1pzYjJGMElISmxiV0ZwYm1sdVp5'
    'QTlJR1pzYjJGMEtITnRjQzVzWlc1bmRHZ3BJQzBnWmxOaGJYQnNaVkJ2Y3pzS0lDQWdJQ0Fn'
    'SUNCcFppQW9jbVZ0WVdsdWFXNW5JRHdnTmpRdU1Da2daVzVrUm1Ga1pTQTlJRzFoZUNnd0xq'
    'QXNJSEpsYldGcGJtbHVaeUF2SURZMExqQXBPd29nSUNBZ2ZRb0tJQ0FnSUM4dklETXVJRXh2'
    'YjNBZ1kzSnZjM05tWVdSbE9pQnpiVzl2ZEdoeklHRnVlU0J5WlhOcFpIVmhiQ0JzYjI5d1JX'
    'NWs0b2FTYkc5dmNGTjBZWEowSUdScGMyTnZiblJwYm5WcGRIa3VDaUFnSUNBdkx5QWdJQ0JV'
    'YUdVZ1pXNWpiMlJsY2lCdWIzY2daVzFpWldSeklHeHZiM0FnZDNKaGNDQmpiMjUwWlhoMElH'
    'NWxlSFFnZEc4Z2JHOXZjRVZ1WkNCemJ5QldVUW9nSUNBZ0x5OGdJQ0FnY1hWaGJuUnBlbUYw'
    'YVc5dUlHdGxaWEJ6SUhSb1pTQnpaV0Z0SUdOdmJuUnBiblZ2ZFhNc0lHSjFkQ0JoSURFMkxY'
    'TmhiWEJzWlNCamNtOXpjMlpoWkdVS0lDQWdJQzh2SUNBZ0lHTmhkR05vWlhNZ1lXNTVJSEps'
    'YldGcGJtbHVaeUJ0YVhOdFlYUmphQzRLSUNBZ0lDOHZJT0tVZ09LVWdDQk1iMjl3SUdOeWIz'
    'TnpabUZrWlNCRVNWTkJRa3hGUkNEaWxJRGlsSUFLSUNBZ0lDOHZJRlJvWlNCd2NtVjJhVzkx'
    'Y3lBeE5pMXpZVzF3YkdVZ1pYRjFZV3d0Y0c5M1pYSWdZM0p2YzNObVlXUmxJSGRoY3lCaElF'
    'NUZWQ0JJUVZKTkxDQnViM1FnWVNCbWFYZ3VDaUFnSUNBdkx5QkpkQ0J5WldGa0lHRWdkM0po'
    'Y0Mxd2IzTWdjMkZ0Y0d4bElHRjBJR3h2YjNCVGRHRnlkQ0FySUNoRFVrOVRVMFpCUkVWZlRF'
    'Vk9JQzBnWkdsemRFWnliMjFGYm1RcENpQWdJQ0F2THlCaGJtUWdZbXhsYm1SbFpDQnBkQ0JK'
    'VGxSUElIUm9aU0J3YkdGNVltRmpheUJoY3lCM1pTQmhjSEJ5YjJGamFHVmtJR3h2YjNCRmJt'
    'UXNJSFJvWlc0Z1pISnZjSEJsWkFvZ0lDQWdMeThnZEdobElHSnNaVzVrSUhkb1pXNGdabE5o'
    'YlhCc1pWQnZjeUIzY21Gd2NHVmtJSFJ2SUd4dmIzQlRkR0Z5ZEM0Z1VtVnpkV3gwT2lCaElH'
    'ZDFZWEpoYm5SbFpXUUtJQ0FnSUM4dklITnBibWRzWlMxellXMXdiR1VnWkdselkyOXVkR2x1'
    'ZFdsMGVTQmxkbVZ5ZVNCc2IyOXdJR2wwWlhKaGRHbHZiaURpZ0pRZ2JXRm5ibWwwZFdSbElI'
    'TmpZV3hwYm1jS0lDQWdJQzh2SUhkcGRHZ2dkR2hsSUdScFptWmxjbVZ1WTJVZ1ltVjBkMlZs'
    'YmlBb2JHOXZjQzFsYm1RdGNtVm5hVzl1SUhKbFlXUXBJR0Z1WkNBb2QzSmhjRkJ2Y3lCeVpX'
    'RmtLU3dLSUNBZ0lDOHZJSGRvYVdOb0lHWnZjaUIwZVhCcFkyRnNJSE4xYzNSaGFXNGdiRzl2'
    'Y0hNZ2FYTWdZU0J6ZFdKemRHRnVkR2xoYkNCbWNtRmpkR2x2YmlCdlppQjBhR1VLSUNBZ0lD'
    'OHZJSE5wWjI1aGJDNGdWR2hwY3lCamNtVmhkR1ZrSUhOMGNtOXVaeUJvWVhKdGIyNXBZM01n'
    'WVhRZ2RHaGxJR3h2YjNBZ1puVnVaR0Z0Wlc1MFlXd2djbUYwWlFvZ0lDQWdMeThnS0dVdVp5'
    'NGdNemN4SUVoNklHRjBJRVlqTkNCM2FYUm9JSE5oYlhCc1pTQTBJQzhnTXpJdGMyRnRjR3hs'
    'SUd4dmIzQXBMQ0JoZFdScFlteGxJR0Z6SUdKMWVub3VDaUFnSUNBdkx3b2dJQ0FnTHk4Z1RH'
    'RnVZM3B2Y3kweklHWnZjbmRoY21RdGRHRndJR3h2YjNBZ2QzSmhjQ0FvYVc0Z1oyVjBVMkZ0'
    'Y0d4bFJpa2dhR0Z1Wkd4bGN5QjBhR1VnYzJWaGJTQnBkSE5sYkdZdUNnb2dJQ0FnTHk4ZzRw'
    'U0E0cFNBSUZOdGIyOTBhR1ZrSUhadmJIVnRaU0JoY0hCc2FXTmhkR2x2YmlEaWxJRGlsSURp'
    'bElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJ'
    'RGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURp'
    'bElEaWxJRGlsSURpbElEaWxJRGlsSURpbElBS0lDQWdJQzh2SUVOdmJYQjFkR1VnZEdobElI'
    'SmhiWEFnWm1GamRHOXlJR0poYzJWa0lHOXVJR2h2ZHlCdFlXNTVJSE5oYlhCc1pYTWdhR0Yy'
    'WlNCbGJHRndjMlZrSUhOcGJtTmxDaUFnSUNBdkx5QjBhR1VnYlc5emRDQnlaV05sYm5RZ2Rt'
    'OXNkVzFsSUdOb1lXNW5aUzRnVTBGTlVFeEZVMTlRUlZKZlZFbERTeUJwY3lBME5ERXdNQzlV'
    'U1VOTFUxOVFSVkpmVTBWRENpQWdJQ0F2THlBb2RIbHdhV05oYkd4NUlEZzRNaUJoZENCVVNV'
    'TkxVMTlRUlZKZlUwVkRQVFV3S1M0Z1VtRnRjQ0JqYjIxd2JHVjBaWE1nYjNabGNpQTJOQ0J6'
    'WVcxd2JHVnpDaUFnSUNBdkx5RGlpWWdnTVM0ME5XMXpJT0tBbENCbVlYTjBJR1Z1YjNWbmFD'
    'QjBieUJpWlNCcGJYQmxjbU5sY0hScFlteGxJR0Z6SUdFZ1ptRmtaUzFwYmlCaWRYUWdjMnh2'
    'ZHlCbGJtOTFaMmdLSUNBZ0lDOHZJSFJ2SUdocFpHVWdkR2hsSUhCbGNpMXpZVzF3YkdVZ2Mz'
    'UmxjQ0IwYUdGMElIQnliMlIxWTJWeklIUm9aU0JqYkdsamF5NEtJQ0FnSUM4dkNpQWdJQ0F2'
    'THlCd2IzTXVkR2xqYTBZZ1BTQjBhR1VnWjJ4dlltRnNJSFJwWTJzZ2IyWWdkR2hsSUdOMWNu'
    'SmxiblFnYzJGdGNHeGxMQ0JtY21GamRHbHZibUZzTGdvZ0lDQWdhVzUwSUY5amRYSlNiM2RU'
    'WjNJZ1BTQndZWFJVYVdOclQyWm1jMlYwVzNCdmN5NXpiMjVuVUc5elhTQXJJQ2h3YjNNdWNt'
    'OTNJQzBnY0dGMFUzUmhjblJTYjNkYmNHOXpMbk52Ym1kUWIzTmRLVHNLSUNBZ0lHWnNiMkYw'
    'SUY5d2IzTlVhV05yUmlBOUlHWnNiMkYwS0dabGRHTm9WR2xqYXloZlkzVnlVbTkzVTJkeUtT'
    'a2dLeUJ3YjNNdWRHbGphenNLSUNBZ0lHWnNiMkYwSUY5VFFVMVFURVZUWDFCRlVsOVVTVU5M'
    'SUQwZ05EUXhNREF1TUNBdklGUkpRMHRUWDFCRlVsOVRSVU03Q2lBZ0lDQm1iRzloZENCZmRs'
    'SmhiWEFnUFNCamJHRnRjQ2dvWDNCdmMxUnBZMnRHSUMwZ1gzWnZiRU5vWVc1blpVRjBWR2xq'
    'YTBZcElDb2dYMU5CVFZCTVJWTmZVRVZTWDFSSlEwc2dMeUEyTkM0d0xDQXdMakFzSURFdU1D'
    'azdDaUFnSUNCbWJHOWhkQ0JmWldabVZtOXNJRDBnYldsNEtHWnNiMkYwS0Y5MmIyeFFjbVYy'
    'S1N3Z1pteHZZWFFvWDNadmJFTjFjbklwTENCZmRsSmhiWEFwT3dvZ0lDQWdMeThnVkhKbGJX'
    'OXNieUJoY0hCc2FXVnpJRzl1SUhSdmNDQnZaaUIwYUdVZ2MyMXZiM1JvWldRZ2RtOXNkVzFs'
    'TENCMGFHVnVJR05zWVcxd0lIUnZJREF1TGpZMExnb2dJQ0FnWDJWbVpsWnZiQ0E5SUdOc1lX'
    'MXdLRjlsWm1aV2Iyd2dLeUJmZEhKbGJXOXNiMFJsYkhSaExDQXdMakFzSURZMExqQXBPd29L'
    'SUNBZ0lISmxkSFZ5YmlCeklDb2dLRjlsWm1aV2Iyd2dMeUEyTkM0d0tTQXFJR1JsWTJ4cFky'
    'c2dLaUJsYm1SR1lXUmxPd3A5Q2dvS0x5OGc0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNB'
    'NHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRw'
    'U0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNB'
    'NHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRw'
    'U0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNB'
    'NHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBQ2k4dklHZGxkRU5vWVc1dVpX'
    'eFBkWFJ3ZFhRZzRvQ1VJSEIxWW14cFl5QmxiblJ5ZVM0Z1VuVnVjeUIwY21sbloyVnlJSE5s'
    'WVhKamFDd2dkR2hsYmlCMGFHVWdZbTlrZVM0S0x5OGdSbTl5SUhSb1pTQm1hWEp6ZENBMk5D'
    'QnpZVzF3YkdWeklHRm1kR1Z5SUdFZ2NtVjBjbWxuWjJWeUxDQkJURk5QSUhKbGJtUmxjaUIz'
    'YVhSb0lIUm9aUW92THlCd2NtVjJhVzkxY3lCMGNtbG5aMlZ5SUdGdVpDQmliR1Z1WkNEaWdK'
    'UWdkR2hwY3lCcGN5QjBhR1VnY0hKbGRtbHZkWE10Ym05MFpTQmpjbTl6YzJaaFpHVUtMeThn'
    'ZEdoaGRDQmxiR2x0YVc1aGRHVnpJR2x1ZEdWeUxXNXZkR1VnWTJ4cFkydHpJQ2h0WVhSamFH'
    'VnpJSFJvWlNCa2VXbHVaMXQwWFNBcklHTm9ZVzV1Wld4YmRGMEtMeThnWTNKdmMzTm1ZV1Js'
    'SUdsdUlFMXBhMDF2WkNkeklFMUVVbFpmVFVsWUxrTlFVQ2t1Q2k4dklPS1VnT0tVZ09LVWdP'
    'S1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tV'
    'Z09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdP'
    'S1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tV'
    'Z09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdP'
    'S1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnQXBt'
    'Ykc5aGRDQm5aWFJEYUdGdWJtVnNUM1YwY0hWMEtHbHVkQ0JqYUN3Z1pteHZZWFFnZEdsdFpT'
    'd2dVRzl6YVhScGIyNGdjRzl6TENCbWJHOWhkQ0J5YjNkVWFXMWxLU0I3Q2dvZ0lDQWdMeThn'
    'VTNSbGNDQXhPaUJtYVc1a0lHMXZjM1F0Y21WalpXNTBiSGt0ZEhKcFoyZGxjbVZrSUc1dmRH'
    'VWdiMjRnZEdocGN5QmphR0Z1Ym1Wc0xnb2dJQ0FnTHk4Z1VGUWdjMlZ0WVc1MGFXTnpJT0tB'
    'bENCaElDSjBjbWxuWjJWeUlpQnBjeUJ6YjIxbGRHaHBibWNnZEdoaGRDQnpkR0Z5ZEhNZ2RH'
    'aGxJSE5oYlhCc1pTQmhkQ0J3YjNNZ01Eb0tJQ0FnSUM4dklDQWc0b0NpSUVaMWJHd2djbTkz'
    'SUNocGJuTjBjblZ0Wlc1MElDc2djR1Z5YVc5a0tTQWdJQ0FnSUNBZ0lDQWdJQ0FnNG9DVUlI'
    'SmxkSEpwWjJkbGNnb2dJQ0FnTHk4Z0lDRGlnS0lnVUdWeWFXOWtMVzl1YkhrZ2NtOTNJQ2h1'
    'YnlCcGJuTjBMQ0J1YnlCbFptWmxZM1FnTXk4MUtTQWdJQ0RpZ0pRZ2NtVjBjbWxuWjJWeUxD'
    'QnBibWhsY21sMElHbHVjM1J5ZFcxbGJuUUtJQ0FnSUM4dklDQWc0b0NpSUZCbGNtbHZaQzF2'
    'Ym14NUlIZHBkR2dnWldabVpXTjBJRE12TlNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnNG9DVUlI'
    'TnNhV1JsSUhSaGNtZGxkQ0J2Ym14NUxDQnVieUJ5WlhSeWFXZG5aWElLSUNBZ0lDOHZJQ0Fn'
    'NG9DaUlFWjFiR3dnY205M0lIZHBkR2dnWldabVpXTjBJRE12TlNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZzRvQ1VJSE5zYVdSbElIUmhjbWRsZENCdmJteDVMQ0J1YnlCeVpYUnlhV2Ru'
    'WlhJS0lDQWdJQzh2SUNBZzRvQ2lJRVZ0Y0hSNUlDOGdhVzV6ZEhKMWJXVnVkQzF2Ym14NUlD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWc0b0NVSUdOdmJuUnBiblZsSUhCeWFXOXlJRzV2'
    'ZEdVS0lDQWdJRTV2ZEdVZ1gyTjFjbEp2ZHlBOUlHZGxkRTV2ZEdVb2NHOXpMbk52Ym1kUWIz'
    'TXNJSEJ2Y3k1eWIzY3NJR05vS1RzS0lDQWdJRTV2ZEdVZ2RISnBaMDV2ZEdVZ1BTQmZZM1Z5'
    'VW05M093b2dJQ0FnYVc1MElDQjBjbWxuVW05M0lDQTlJSEJ2Y3k1eWIzYzdDaUFnSUNCcGJu'
    'UWdJSFJ5YVdkUVlYUWdJRDBnY0c5ekxuTnZibWRRYjNNN0NpQWdJQ0JwYm5RZ0lIUnZibVZU'
    'Ykdsa1pWUmhjbWRsZENBOUlEQTdJQ0F2THlCM2FHVnVJSE5sZEN3Z2RHaHBjeUJ5YjNjZ1ky'
    'RnljbWxsY3lCaElETjRlQzgxZUhnZ2MyeHBaR1VnZEdGeVoyVjBDaUFnSUNCaWIyOXNJRjlq'
    'ZFhKSmMxUnZibVZRYjNKMFlTQTlJQ2dvWDJOMWNsSnZkeTVsWm1abFkzUWdQVDBnTUhneklI'
    'eDhJRjlqZFhKU2IzY3VaV1ptWldOMElEMDlJREI0TlNrZ0ppWUtJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJRjlqZFhKU2IzY3VjR1Z5YVc5a0lENGdNQ2s3Q2lBZ0lD'
    'QmliMjlzSUY5amRYSkpjMUpsZEhKcFp5QWdJQ0E5SUNoZlkzVnlVbTkzTG5CbGNtbHZaQ0Er'
    'SURBZ0ppWWdJVjlqZFhKSmMxUnZibVZRYjNKMFlTazdJQ0F2THlCaGJua2djR1Z5YVc5a0lI'
    'ZHBkR2h2ZFhRZ015ODFJSEpsZEhKcFoyZGxjbk1LSUNBZ0lHSnZiMndnWDJOMWNraGhjMGx1'
    'YzNRZ0lDQWdJRDBnS0Y5amRYSlNiM2N1YVc1emRISjFiV1Z1ZENBK0lEQXBPd29LSUNBZ0lH'
    'bG1JQ2hmWTNWeVNYTlViMjVsVUc5eWRHRXBJSHNLSUNBZ0lDQWdJQ0F2THlCVGJHbGtaU0Iw'
    'WVhKblpYUWc0b0NVSUdacGJtUWdjSEpwYjNJZ1VrVkJUQ0IwY21sbloyVnlJR1p2Y2lCellX'
    'MXdiR1V2Y0dWeWFXOWtJR052Ym5SbGVIUUtJQ0FnSUNBZ0lDQjBiMjVsVTJ4cFpHVlVZWEpu'
    'WlhRZ1BTQmZZM1Z5VW05M0xuQmxjbWx2WkRzS0lDQWdJQ0FnSUNCcGJuUWdjMUlnUFNCd2Iz'
    'TXVjbTkzTENCelVDQTlJSEJ2Y3k1emIyNW5VRzl6T3dvZ0lDQWdJQ0FnSUdadmNpQW9hVzUw'
    'SUd4aUlEMGdNVHNnYkdJZ1BDQXhNamc3SUd4aUt5c3BJSHNLSUNBZ0lDQWdJQ0FnSUNBZ2Mx'
    'SXRMVHNLSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLSE5TSUR3Z01Da2dld29nSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdhV1lnS0hOUUlENGdNQ2tnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lI'
    'TlFMUzA3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2MxSWdQU0J3WVhSVGRHRnlkRkp2'
    'ZDF0elVGMGdLeUFvY0dGMFVtOTNUMlptYzJWMFczTlFLekZkSUMwZ2NHRjBVbTkzVDJabWMy'
    'VjBXM05RWFNrZ0xTQXhPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdmU0JsYkhObElIc2dZbkps'
    'WVdzN0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNBZ0lDQk9iM1JsSUhCeVpY'
    'WWdQU0JuWlhST2IzUmxLSE5RTENCelVpd2dZMmdwT3dvZ0lDQWdJQ0FnSUNBZ0lDQmliMjlz'
    'SUhCeVpYWkpjMVJ2Ym1WVWNtbG5JRDBnS0Nod2NtVjJMbVZtWm1WamRDQTlQU0F3ZURNZ2ZI'
    'd2djSEpsZGk1bFptWmxZM1FnUFQwZ01IZzFLU0FtSmlCd2NtVjJMbkJsY21sdlpDQStJREFw'
    'T3dvZ0lDQWdJQ0FnSUNBZ0lDQXZMeUJTWldGc0lIUnlhV2RuWlhJNklHaGhjeUJ3WlhKcGIy'
    'UWdRVTVFSUc1dmRDQmhJSFJ2Ym1VdGNHOXlkR0VnZEdGeVoyVjBJSEp2ZHdvZ0lDQWdJQ0Fn'
    'SUNBZ0lDQnBaaUFvY0hKbGRpNXdaWEpwYjJRZ1BpQXdJQ1ltSUNGd2NtVjJTWE5VYjI1bFZI'
    'SnBaeWtnZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnTHk4Z1JHVjBaWEp0YVc1bElHbHVjM1J5'
    'ZFcxbGJuUTZJSEJ5WldabGNpQndjbVYyTG1sdWMzUnlkVzFsYm5Rc0lHVnNjMlVnYzJOaGJp'
    'Qm1kWEowYUdWeUlHWnZjaUJqYjI1MFpYaDBDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnBaaUFv'
    'Y0hKbGRpNXBibk4wY25WdFpXNTBJRDRnTUNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJSFJ5YVdkT2IzUmxJRDBnY0hKbGRqc2dkSEpwWjFKdmR5QTlJSE5TT3lCMGNtbG5VR0Yw'
    'SUQwZ2MxQTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQjlJR1ZzYzJVZ2V3b2dJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQzh2SUZCbGNtbHZaQzF2Ym14NUlISnZkeURpZ0pRZ1ptbHVaQ0Jw'
    'Ym5OMGNuVnRaVzUwSUdOdmJuUmxlSFFLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCT2Iz'
    'UmxJSEpsWVd3Z1BTQndjbVYyT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbHVkQ0J6'
    'VWpJZ1BTQnpVaXdnYzFBeUlEMGdjMUE3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1pt'
    'OXlJQ2hwYm5RZ2JHSXlJRDBnTVRzZ2JHSXlJRHdnTVRJNE95QnNZaklyS3lrZ2V3b2dJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCelVqSXRMVHNLSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdhV1lnS0hOU01pQThJREFwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2h6VURJZ1BpQXdLU0I3SUhOUU1pMHRPeUJ6VWpJZ1BT'
    'QndZWFJUZEdGeWRGSnZkMXR6VURKZElDc2dLSEJoZEZKdmQwOW1abk5sZEZ0elVESXJNVjBn'
    'TFNCd1lYUlNiM2RQWm1aelpYUmJjMUF5WFNrZ0xTQXhPeUI5Q2lBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNCbGJITmxJR0p5WldGck93b2dJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNCOUNpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUU1dmRH'
    'VWdjRElnUFNCblpYUk9iM1JsS0hOUU1pd2djMUl5TENCamFDazdDaUFnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2h3TWk1cGJuTjBjblZ0Wlc1MElENGdNQ2tnZXlCeVpX'
    'RnNMbWx1YzNSeWRXMWxiblFnUFNCd01pNXBibk4wY25WdFpXNTBPeUJpY21WaGF6c2dmUW9n'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QjBjbWxuVG05MFpTQTlJSEpsWVd3N0lIUnlhV2RTYjNjZ1BTQnpVanNnZEhKcFoxQmhkQ0E5'
    'SUhOUU93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1lu'
    'SmxZV3M3Q2lBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQjlDaUFnSUNCOUlHVnNjMlVn'
    'YVdZZ0tDRmZZM1Z5U1hOU1pYUnlhV2NwSUhzS0lDQWdJQ0FnSUNBdkx5Qk9ieUJ3WlhKcGIy'
    'UWc0b0NVSUdOdmJuUnBiblZsSUhCeWFXOXlJRzV2ZEdVZ0tHOXlJRzV2SUdGMVpHbHZJR2xt'
    'SUc1dmRHaHBibWNnY0hKcGIzSXBMZ29nSUNBZ0lDQWdJQzh2SUY5amRYSklZWE5KYm5OMElI'
    'ZHBkR2dnYm04Z2NHVnlhVzlrSUdseklHRWdibTh0YjNBZ1ptOXlJSFJ5YVdkblpYSWdjSFZ5'
    'Y0c5elpYTWdLRkJVSUhGMWFYSnJLUzRLSUNBZ0lDQWdJQ0JwYm5RZ2MxSWdQU0J3YjNNdWNt'
    'OTNMQ0J6VUNBOUlIQnZjeTV6YjI1blVHOXpPd29nSUNBZ0lDQWdJR1p2Y2lBb2FXNTBJR3hp'
    'SUQwZ01Uc2diR0lnUENBeE1qZzdJR3hpS3lzcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnYzFJdExU'
    'c0tJQ0FnSUNBZ0lDQWdJQ0FnYVdZZ0tITlNJRHdnTUNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ2FXWWdLSE5RSUQ0Z01Da2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUhOUUxT'
    'MDdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnYzFJZ1BTQndZWFJUZEdGeWRGSnZkMXR6'
    'VUYwZ0t5QW9jR0YwVW05M1QyWm1jMlYwVzNOUUt6RmRJQzBnY0dGMFVtOTNUMlptYzJWMFcz'
    'TlFYU2tnTFNBeE93b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2ZTQmxiSE5sSUhzZ1luSmxZV3M3'
    'SUgwS0lDQWdJQ0FnSUNBZ0lDQWdmUW9nSUNBZ0lDQWdJQ0FnSUNCT2IzUmxJSEJ5WlhZZ1BT'
    'Qm5aWFJPYjNSbEtITlFMQ0J6VWl3Z1kyZ3BPd29nSUNBZ0lDQWdJQ0FnSUNCaWIyOXNJSEJ5'
    'WlhaSmMxUnZibVZVY21sbklEMGdLQ2h3Y21WMkxtVm1abVZqZENBOVBTQXdlRE1nZkh3Z2NI'
    'SmxkaTVsWm1abFkzUWdQVDBnTUhnMUtTQW1KaUJ3Y21WMkxuQmxjbWx2WkNBK0lEQXBPd29n'
    'SUNBZ0lDQWdJQ0FnSUNCcFppQW9jSEpsZGk1d1pYSnBiMlFnUGlBd0lDWW1JQ0Z3Y21WMlNY'
    'TlViMjVsVkhKcFp5a2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0hCeVpYWXVhVzV6'
    'ZEhKMWJXVnVkQ0ErSURBcElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0IwY21sblRt'
    'OTBaU0E5SUhCeVpYWTdJSFJ5YVdkU2IzY2dQU0J6VWpzZ2RISnBaMUJoZENBOUlITlFPd29n'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdmU0JsYkhObElIc0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0JPYjNSbElISmxZV3dnUFNCd2NtVjJPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUdsdWRDQnpVaklnUFNCelVpd2djMUF5SUQwZ2MxQTdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnWm05eUlDaHBiblFnYkdJeUlEMGdNVHNnYkdJeUlEd2dNVEk0T3lCc1lqSXJLeWtn'
    'ZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0J6VWpJdExUc0tJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXWWdLSE5TTWlBOElEQXBJSHNLSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsbUlDaHpVRElnUGlBd0tTQjdJSE5RTWkwdE95'
    'QnpVaklnUFNCd1lYUlRkR0Z5ZEZKdmQxdHpVREpkSUNzZ0tIQmhkRkp2ZDA5bVpuTmxkRnR6'
    'VURJck1WMGdMU0J3WVhSU2IzZFBabVp6WlhSYmMxQXlYU2tnTFNBeE95QjlDaUFnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JsYkhObElHSnlaV0ZyT3dvZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJRTV2ZEdVZ2NESWdQU0JuWlhST2IzUmxLSE5RTWl3Z2MxSXlMQ0JqYUNrN0NpQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdsbUlDaHdNaTVwYm5OMGNuVnRaVzUwSUQ0Z01D'
    'a2dleUJ5WldGc0xtbHVjM1J5ZFcxbGJuUWdQU0J3TWk1cGJuTjBjblZ0Wlc1ME95QmljbVZo'
    'YXpzZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNCMGNtbG5UbTkwWlNBOUlISmxZV3c3SUhSeWFXZFNiM2NnUFNCelVqc2dkSEpw'
    'WjFCaGRDQTlJSE5RT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnWW5KbFlXczdDaUFnSUNBZ0lDQWdJQ0FnSUgwS0lDQWdJQ0FnSUNCOUNpQWdJQ0I5'
    'SUdWc2MyVWdhV1lnS0Y5amRYSkpjMUpsZEhKcFp5QW1KaUFoWDJOMWNraGhjMGx1YzNRcElI'
    'c0tJQ0FnSUNBZ0lDQXZMeUJRWlhKcGIyUXRiMjVzZVNCeVpYUnlhV2RuWlhJZzRvQ1VJR1pw'
    'Ym1RZ2FXNXpkSEoxYldWdWRDQmpiMjUwWlhoMElDaHpZVzF3YkdVZ2FXNW9aWEpwZEdWa0lH'
    'WnliMjBnY0hKcGIzSWdkSEpwWjJkbGNpa0tJQ0FnSUNBZ0lDQnBiblFnYzFJZ1BTQndiM011'
    'Y205M0xDQnpVQ0E5SUhCdmN5NXpiMjVuVUc5ek93b2dJQ0FnSUNBZ0lHWnZjaUFvYVc1MElH'
    'eGlJRDBnTVRzZ2JHSWdQQ0F4TWpnN0lHeGlLeXNwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdjMUl0'
    'TFRzS0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0hOU0lEd2dNQ2tnZXdvZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnYVdZZ0tITlFJRDRnTUNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJSE5R'
    'TFMwN0NpQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdjMUlnUFNCd1lYUlRkR0Z5ZEZKdmQx'
    'dHpVRjBnS3lBb2NHRjBVbTkzVDJabWMyVjBXM05RS3pGZElDMGdjR0YwVW05M1QyWm1jMlYw'
    'VzNOUVhTa2dMU0F4T3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnZlNCbGJITmxJSHNnWW5KbFlX'
    'czdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ2ZRb2dJQ0FnSUNBZ0lDQWdJQ0JPYjNSbElIQnlaWFln'
    'UFNCblpYUk9iM1JsS0hOUUxDQnpVaXdnWTJncE93b2dJQ0FnSUNBZ0lDQWdJQ0JwWmlBb2NI'
    'SmxkaTVwYm5OMGNuVnRaVzUwSUQ0Z01Da2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdkSEpw'
    'WjA1dmRHVXVhVzV6ZEhKMWJXVnVkQ0E5SUhCeVpYWXVhVzV6ZEhKMWJXVnVkRHNLSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJR0p5WldGck93b2dJQ0FnSUNBZ0lDQWdJQ0I5Q2lBZ0lDQWdJQ0Fn'
    'ZlFvZ0lDQWdJQ0FnSUM4dklIUnlhV2RRWVhRdmRISnBaMUp2ZHlCemRHRjVJR0YwSUdOMWNu'
    'SmxiblFnY205M0lPS0FsQ0IwYUdseklFbFRJR0VnY21WMGNtbG5aMlZ5Q2lBZ0lDQjlDaUFn'
    'SUNBdkx5QmxiSE5sT2lCbWRXeHNJSFJ5YVdkblpYSWdLSEJsY21sdlpDQXJJR2x1YzNSeWRX'
    'MWxiblFzSUc1dklETXZOU2tnNG9DVUlIUnlhV2RPYjNSbElHRnNjbVZoWkhrZ1kyOXljbVZq'
    'ZEFvS0lDQWdJQzh2SU9LVWdPS1VnQ0JTWlc1a1pYSWdkMmwwYUNCamRYSnlaVzUwSUhSeWFX'
    'ZG5aWElnNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNB'
    'NHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRw'
    'U0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQTRwU0E0cFNBNHBTQUNpQWdJQ0Jt'
    'Ykc5aGRDQnpYMk4xY25JZ1BTQmZaMk52UW05a2VTaGphQ3dnY0c5ekxDQjBhVzFsTENCeWIz'
    'ZFVhVzFsTENCMGNtbG5VR0YwTENCMGNtbG5VbTkzTENCMGNtbG5UbTkwWlN3Z2RHOXVaVk5z'
    'YVdSbFZHRnlaMlYwS1RzS0NpQWdJQ0F2THlEaWxJRGlsSUFnUTNKdmMzTm1ZV1JsSUhkcGJt'
    'UnZkeUJqYUdWamF5RGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURp'
    'bElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJ'
    'RGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURpbElEaWxJRGlsSURp'
    'bElEaWxJRGlsSURpbElEaWxJQUtJQ0FnSUM4dklFTnZiWEIxZEdVZ2MyRnRjR3hsY3lCbGJH'
    'RndjMlZrSUhOcGJtTmxJSFJvWlNCamRYSnlaVzUwSUhSeWFXZG5aWElnWm1seVpXUXVJRTl1'
    'YkhrZ2FXNXphV1JsQ2lBZ0lDQXZMeUIwYUdVZ1ptbHljM1FnTmpRZ2MyRnRjR3hsY3lCcGN5'
    'QjBhR1VnWTNKdmMzTm1ZV1JsSUcxbFlXNXBibWRtZFd3ZzRvQ1VJR0psZVc5dVpDQjBhR0Yw'
    'TENCMGFHVUtJQ0FnSUM4dklIQnlaWFpwYjNWeklHNXZkR1VnYUdGeklHeHZibWNnWm1Ga1pX'
    'UWdiM1YwTGdvZ0lDQWdabXh2WVhRZ1kzVnlWSEpwWjFScGJXVkdJRDBnWm14dllYUW9abVYw'
    'WTJoVWFXTnJLSEJoZEZScFkydFBabVp6WlhSYmRISnBaMUJoZEYwZ0t5QW9kSEpwWjFKdmR5'
    'QXRJSEJoZEZOMFlYSjBVbTkzVzNSeWFXZFFZWFJkS1NrcENpQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnTHlCVVNVTkxVMTlRUlZKZlUwVkRPd29nSUNBZ1pteHZZWFFnWVdkbFUy'
    'RnRjR3hsY3lBOUlDaDBhVzFsSUMwZ1kzVnlWSEpwWjFScGJXVkdLU0FxSURRME1UQXdMakE3'
    'Q2dvZ0lDQWdhV1lnS0dGblpWTmhiWEJzWlhNZ1BDQTJOQzR3SUNZbUlHRm5aVk5oYlhCc1pY'
    'TWdQajBnTUM0d0tTQjdDaUFnSUNBZ0lDQWdMeThnNHBTQTRwU0FJRk5sWVhKamFDQm1iM0ln'
    'ZEdobElGQlNSVlpKVDFWVElIUnlhV2RuWlhJZ0tHOXVaU0J5YjNjZ1ltVm1iM0psSUdOMWNu'
    'SmxiblFwSU9LVWdPS1VnT0tVZ09LVWdBb2dJQ0FnSUNBZ0lDOHZJRk5oYldVZ1lXeG5iM0pw'
    'ZEdodElHRnpJSFJvWlNCdFlXbHVJSFJ5YVdkblpYSWdjMlZoY21Ob0lHSjFkQ0J6ZEdGeWRH'
    'bHVaeUJ2Ym1VZ2NtOTNDaUFnSUNBZ0lDQWdMeThnWldGeWJHbGxjaTRnVTJWMGN5QndWSEpw'
    'WjFCaGRDOXdWSEpwWjFKdmR5OXdWSEpwWjA1dmRHVXVDaUFnSUNBZ0lDQWdhVzUwSUNCd1ZI'
    'SnBaMUJoZENBOUlDMHhMQ0J3VkhKcFoxSnZkeUE5SUMweE93b2dJQ0FnSUNBZ0lFNXZkR1Vn'
    'Y0ZSeWFXZE9iM1JsT3dvZ0lDQWdJQ0FnSUhzS0lDQWdJQ0FnSUNBZ0lDQWdhVzUwSUhOU0lE'
    'MGdkSEpwWjFKdmR5d2djMUFnUFNCMGNtbG5VR0YwT3dvZ0lDQWdJQ0FnSUNBZ0lDQm1iM0ln'
    'S0dsdWRDQnNZaUE5SURFN0lHeGlJRHdnTVRJNE95QnNZaXNyS1NCN0NpQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNCelVpMHRPd29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdhV1lnS0hOU0lEd2dNQ2tn'
    'ZXdvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2h6VUNBK0lEQXBJSHNLSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdjMUF0TFRzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnYzFJZ1BTQndZWFJUZEdGeWRGSnZkMXR6VUYwZ0t5QW9jR0YwVW05M1Qy'
    'Wm1jMlYwVzNOUUt6RmRJQzBnY0dGMFVtOTNUMlptYzJWMFczTlFYU2tnTFNBeE93b2dJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJSDBnWld4elpTQjdJR0p5WldGck95QjlDaUFnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQjlDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQk9iM1JsSUhCeVpYWWdQU0Ju'
    'WlhST2IzUmxLSE5RTENCelVpd2dZMmdwT3dvZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnWW05dmJD'
    'QndjbVYyU1hOVWIyNWxWSEpwWnlBOUlDZ29jSEpsZGk1bFptWmxZM1FnUFQwZ01IZ3pJSHg4'
    'SUhCeVpYWXVaV1ptWldOMElEMDlJREI0TlNrZ0ppWWdjSEpsZGk1d1pYSnBiMlFnUGlBd0tU'
    'c0tJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lHbG1JQ2h3Y21WMkxuQmxjbWx2WkNBK0lEQWdKaVln'
    'SVhCeVpYWkpjMVJ2Ym1WVWNtbG5LU0I3Q2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FX'
    'WWdLSEJ5WlhZdWFXNXpkSEoxYldWdWRDQStJREFwSUhzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnY0ZSeWFXZE9iM1JsSUQwZ2NISmxkanNnY0ZSeWFXZFNiM2NnUFNCelVq'
    'c2djRlJ5YVdkUVlYUWdQU0J6VURzS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQjlJR1Zz'
    'YzJVZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBdkx5QlFaWEpwYjJRdGIy'
    'NXNlU0RpZ0pRZ1ptbHVaQ0JwYm5OMGNuVnRaVzUwSUdOdmJuUmxlSFFLSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdUbTkwWlNCeVpXRnNJRDBnY0hKbGRqc0tJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ2FXNTBJSE5TTWlBOUlITlNMQ0J6VURJZ1BTQnpVRHNL'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdabTl5SUNocGJuUWdiR0l5SUQwZ01U'
    'c2diR0l5SUR3Z01USTRPeUJzWWpJckt5a2dld29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnYzFJeUxTMDdDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0JwWmlBb2MxSXlJRHdnTUNrZ2V3b2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUdsbUlDaHpVRElnUGlBd0tTQjdJSE5RTWkwdE95QnpVaklnUFNCd1lY'
    'UlRkR0Z5ZEZKdmQxdHpVREpkSUNzZ0tIQmhkRkp2ZDA5bVpuTmxkRnR6VURJck1WMGdMU0J3'
    'WVhSU2IzZFBabVp6WlhSYmMxQXlYU2tnTFNBeE95QjlDaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1pXeHpaU0JpY21WaGF6c0tJQ0FnSUNBZ0lDQWdJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJSDBLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUU1dmRHVWdjRElnUFNCblpYUk9iM1JsS0hOUU1pd2djMUl5TENCamFDazdDaUFn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JwWmlBb2NESXVhVzV6ZEhKMWJX'
    'VnVkQ0ErSURBcElIc2djbVZoYkM1cGJuTjBjblZ0Wlc1MElEMGdjREl1YVc1emRISjFiV1Z1'
    'ZERzZ1luSmxZV3M3SUgwS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lD'
    'QWdJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0J3VkhKcFowNXZkR1VnUFNCeVpXRnNPeUJ3'
    'VkhKcFoxSnZkeUE5SUhOU095QndWSEpwWjFCaGRDQTlJSE5RT3dvZ0lDQWdJQ0FnSUNBZ0lD'
    'QWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0JpY21WaGF6c0tJQ0Fn'
    'SUNBZ0lDQWdJQ0FnSUNBZ0lIMEtJQ0FnSUNBZ0lDQWdJQ0FnZlFvZ0lDQWdJQ0FnSUgwS0Np'
    'QWdJQ0FnSUNBZ0x5OGc0cFNBNHBTQUlGSmxibVJsY2lCM2FYUm9JSEJ5WlhacGIzVnpJSFJ5'
    'YVdkblpYSWdZVzVrSUdKc1pXNWtJT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdP'
    'S1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tVZ09LVWdPS1VnT0tV'
    'Z0FvZ0lDQWdJQ0FnSUM4dklGUm9aU0J3Y21WMmFXOTFjeUIwY21sbloyVnlJR2RsZEhNZ2RH'
    'OXVaVk5zYVdSbFZHRnlaMlYwUFRBZ0tIZGxJR1J2YmlkMElIUnlZV05ySUdsMGN3b2dJQ0Fn'
    'SUNBZ0lDOHZJSE5zYVdSbElHTm9ZV2x1SU9LQWxDQm1iM0lnZEdobElEWTBMWE5oYlhCc1pT'
    'QmpjbTl6YzJaaFpHVWdkMmx1Wkc5M0lIUm9aU0JrYVdabVpYSmxibU5sQ2lBZ0lDQWdJQ0Fn'
    'THk4Z2FYTWdhVzVoZFdScFlteGxLUzRnVEdsdVpXRnlJR055YjNOelptRmtaVG9nZEQwd0lH'
    'bHpJR0ZzYkMxd2NtVjJhVzkxY3l3Z2REMHhJR2x6Q2lBZ0lDQWdJQ0FnTHk4Z1lXeHNMV04x'
    'Y25KbGJuUXVJRk50YjI5MGFITjBaWEFnZDI5MWJHUWdkMjl5YXlCaWRYUWdiR2x1WldGeUlH'
    'MWhkR05vWlhNZ1RXbHJUVzlrSjNNS0lDQWdJQ0FnSUNBdkx5QkRLeXNnVFdsNEtsTjBaWEps'
    'YjA1dlkyeHBZMnNnZG05c2RXMWxJSEpoYlhCcGJtY2daWGhoWTNSc2VTNEtJQ0FnSUNBZ0lD'
    'QnBaaUFvY0ZSeWFXZFFZWFFnUGowZ01Da2dld29nSUNBZ0lDQWdJQ0FnSUNCbWJHOWhkQ0J6'
    'WDNCeVpYWWdQU0JmWjJOdlFtOWtlU2hqYUN3Z2NHOXpMQ0IwYVcxbExDQnliM2RVYVcxbExD'
    'QndWSEpwWjFCaGRDd2djRlJ5YVdkU2IzY3NJSEJVY21sblRtOTBaU3dnTUNrN0NpQWdJQ0Fn'
    'SUNBZ0lDQWdJR1pzYjJGMElIUWdQU0JoWjJWVFlXMXdiR1Z6SUM4Z05qUXVNRHNLSUNBZ0lD'
    'QWdJQ0FnSUNBZ2NtVjBkWEp1SUhOZmNISmxkaUFxSUNneExqQWdMU0IwS1NBcklITmZZM1Z5'
    'Y2lBcUlIUTdDaUFnSUNBZ0lDQWdmUW9nSUNBZ2ZRb0tJQ0FnSUhKbGRIVnliaUJ6WDJOMWNu'
    'STdDbjBLJykuZGVjb2RlKCd1dGYtOCcpCgogICAgIyBBc3NlbWJsZQogICAgcmV0dXJuIGhl'
    'YWRlciArIG1ldGEgKyAiIi5qb2luKGRhdGFfYXJyYXlzKSArICJcbiIgKyB0YWJsZXMgKyBm'
    'ZXRjaGVycyArIGRlY29kZXJzICsgZ2V0X2NoYW5uZWxfb3V0cHV0CgoKaWYgX19uYW1lX18g'
    'PT0gJ19fbWFpbl9fJzoKICAgIG1vZF9wYXRoID0gc3lzLmFyZ3ZbMV0gaWYgbGVuKHN5cy5h'
    'cmd2KSA+IDEgZWxzZSAnL21udC91c2VyLWRhdGEvdXBsb2Fkcy8xMlRILk1PRCcKICAgIG91'
    'dF9wYXRoID0gc3lzLmFyZ3ZbMl0gaWYgbGVuKHN5cy5hcmd2KSA+IDIgZWxzZSAnL2hvbWUv'
    'Y2xhdWRlL21vZF9jcnVuY2gvMTJUSF9jcnVuY2hfY29tbW9uLmdsc2wnCiAgICBtYWluKG1v'
    'ZF9wYXRoLCBvdXRfcGF0aCkK'
)

def _trim_song_to_audio_cap(mod, cap_sec=SHADERTOY_AUDIO_CAP_SEC, strict=False):
    """Trim mod.song_positions to the first ~cap_sec of playback.

    Audio past Shadertoy's pre-rendered buffer never plays, so pattern
    (and any tail-only sample) data for it is dead weight in the embedded
    GLSL. Dropping those order entries lets the encoder skip the unused
    tail patterns. Conservative (strict=False): the whole song-position
    straddling the cap is kept, so nothing audible before cap_sec is lost
    (the build is then slightly > cap_sec; Shadertoy plays the first cap_sec).

    strict=True: drop the straddling position too, so the kept range ends
    at ≤ cap_sec EXACTLY. Used by --parts so each part fits the 180s buffer
    with no overflow — the dropped straddling position becomes the first
    position of the next part (gap-free, no overlap)."""
    sp = list(getattr(mod, 'song_positions', []) or [])
    if not sp:
        return
    speed = getattr(mod, 'initial_speed', 6) or 6
    tempo = getattr(mod, 'initial_tempo', 125) or 125
    nch   = getattr(mod, 'num_channels', 4) or 4
    t = 0.0
    keep = len(sp)
    # Native MOD patterns always carry 64 rows on disk, but the VQ encoder
    # honours Effect D (pattern break, Dxx): it encodes only up to the Dxx
    # row, and the NEXT pattern starts at the row specified in the Dxx param.
    # S3M/IT/XM already store the correct per-position row count in len(rows).
    _is_native_mod = not (getattr(mod,'is_s3m',False) or
                          getattr(mod,'is_it', False) or
                          getattr(mod,'is_xm', False))
    _trim_next_start = 0   # row within the pattern where playback starts (Dxx carryover)
    for si, pat in enumerate(sp):
        try:
            rows  = mod.patterns[pat]
            nrows = len(rows) if rows else 64
        except Exception:
            rows, nrows = None, 64
        # For native MOD: scan forward from _trim_next_start for the first
        # Dxx effect — that row is the last played row in this pattern.
        # (S3M/IT/XM: len(rows) is already the correct count; no scan needed.)
        _row_start = _trim_next_start       # first row played in this pattern
        _next_start_out = 0                 # start row for the NEXT pattern
        if _is_native_mod and rows:
            _d_row = None
            for _ri in range(_row_start, nrows):
                for _ch in range(nch):
                    try:
                        _n = rows[_ri][_ch]
                        if _n.get('effect', 0) == 0xD and _d_row is None:
                            _p = _n.get('param', 0)
                            _d_row = _ri
                            _next_start_out = ((_p >> 4) & 0xF) * 10 + (_p & 0xF)
                    except Exception:
                        pass
                if _d_row is not None:
                    break
            if _d_row is not None:
                nrows = _d_row - _row_start + 1
            else:
                nrows = nrows - _row_start
        _trim_next_start = _next_start_out
        sp_start = t
        if sp_start >= cap_sec:        # this position begins past the cap
            keep = si
            break
        # Iterate the actual played rows (_row_start .. _row_start+nrows-1).
        # For non-MOD the original code set _row_start=0 implicitly, so
        # range(_row_start, _row_start+nrows) == range(0, nrows) — no change.
        for ri in range(_row_start, _row_start + nrows):
            if rows is not None and ri < len(rows):
                _row = rows[ri]
                for ch in range(nch):
                    try:
                        n = _row[ch]
                        # Detect speed/tempo in BOTH representations: raw S3M
                        # (command A=1 set-speed, T=20 set-tempo) AND MOD-style
                        # effect 0xF. This trim can run BEFORE the S3M→0xF
                        # conversion, so checking only 0xF missed mid-song speed
                        # changes → wrong trim point → parts > 180s.
                        _cmd = n.get('command'); _inf = n.get('info', 0)
                        if   _cmd == 1  and _inf > 0: speed = _inf
                        elif _cmd == 20 and _inf > 0: tempo = _inf
                        elif n.get('effect', 0) == 0xF and n.get('param', 0) > 0:
                            p = n['param']
                            if p < 0x20: speed = p
                            else:        tempo = p
                    except Exception:
                        pass
            tps = tempo * 2.0 / 5.0
            t  += (speed / tps) if tps > 0 else 0.0
        if strict and t > cap_sec:     # this position ENDS past the cap → drop it
            keep = max(1, si)          # (keep ≥1 so a single >cap position still builds)
            break
    if keep < len(sp):
        mod.song_positions = sp[:keep]
        if hasattr(mod, 'song_length'):
            try: mod.song_length = len(mod.song_positions)
            except Exception: pass
        print(f"   ✂️  Trimmed song order {len(sp)}→{keep} positions "
              f"(tail past {cap_sec:.0f}s never plays on Shadertoy — shrinks build)")

        # Drop sample data for instruments that are triggered ONLY in the
        # now-removed tail patterns — they can never sound in the played
        # window, so their audio is pure dead weight in the embedded GLSL.
        # Conservative: an instrument is dropped only if its number never
        # appears in ANY kept pattern's instrument column.
        try:
            used = set()
            for pat in mod.song_positions:
                rows = mod.patterns[pat] if pat < len(mod.patterns) else None
                if not rows:
                    continue
                for row in rows:
                    for cell in row:
                        try:
                            # S3M/MOD cells key the instrument as
                            # 'instrument'; IT (ITFile) cells use 'sample'.
                            # Without the 'sample' fallback, IT's `used` set
                            # stayed EMPTY → every sample judged "tail-only"
                            # and its audio nuked → VQ got 0 vectors → build
                            # failed/silent (jeff.it, any IT song >180s).
                            inst = (cell.get('instrument', 0)
                                    or cell.get('sample', 0))
                        except Exception:
                            inst = 0
                        if inst:
                            used.add(inst)
            smps = getattr(mod, 'samples', None)
            if smps:
                dropped = 0; dbytes = 0
                for i, s in enumerate(smps):
                    if not isinstance(s, dict):
                        continue
                    if (i + 1) in used:
                        continue
                    _ln = s.get('length', 0) or 0
                    if _ln <= 0:
                        continue
                    dbytes += _ln
                    s['data'] = None
                    s['length'] = 0
                    s['loop_start'] = 0
                    s['loop_len'] = 0
                    dropped += 1
                if dropped:
                    print(f"   ✂️  Dropped {dropped} tail-only instruments' "
                          f"sample audio ({dbytes:,} bytes — never play in "
                          f"first {cap_sec:.0f}s)")
        except Exception as _e:
            print(f"   (tail-sample drop skipped: {_e})")


def _render_mp3_via_toolchain(base_name, secs):
    """Render {base_name}_shadertoy_{common,sound}.glsl to {base_name}.mp3 by
    running the SAME glslang->spirv-cross->clang pipeline sound_exec.py uses
    (so it's the exact audio ShaderToy plays), then encoding with ffmpeg/lame.
    If the toolchain isn't installed, print how to get it and return (no crash)."""
    import os as _os, shutil as _sh, subprocess as _sp
    _root    = _os.path.dirname(_os.path.abspath(__file__))
    _glslang = f"{_root}/glslang/src/StandAlone/glslangValidator"
    _spirv   = f"{_root}/spirv-cross/src/spirv-cross"
    _se      = f"{_root}/sound_exec.py"
    missing = []
    if not _os.path.exists(_glslang): missing.append("glslang (build at ./glslang/src/StandAlone/glslangValidator)")
    if not _os.path.exists(_spirv):   missing.append("spirv-cross (build at ./spirv-cross/src/spirv-cross)")
    if _sh.which("clang") is None and _sh.which("clang++") is None: missing.append("clang / clang++")
    _enc = "ffmpeg" if _sh.which("ffmpeg") else ("lame" if _sh.which("lame") else None)
    if _enc is None:                  missing.append("ffmpeg or lame (mp3 encoder)")
    if not _os.path.exists(_se):      missing.append("sound_exec.py (ships beside this script)")
    if missing:
        print("\n⚠️  --mp3 needs the GLSL→audio toolchain, which isn't fully installed here:")
        for _m in missing:
            print(f"      • {_m}")
        print("   Build glslang + spirv-cross into ./glslang and ./spirv-cross (matching")
        print("   sound_exec.py's paths) and install an mp3 encoder, then re-run with --mp3.")
        return
    _base = base_name + "_shadertoy"
    _wav  = f"{_base}_exec.wav"
    _mp3  = base_name + ".mp3"
    print(f"\n🎧 --mp3: rendering {secs:.0f}s of the Sound tab (glslang→spirv-cross→clang)… "
          f"(CPU, be patient)")
    _r = _sp.run(["python3", _se, _base, str(secs)], capture_output=True, text=True)
    if _r.returncode != 0 or not _os.path.exists(_wav):
        print("   ✗ sound_exec render failed — last output:")
        print((_r.stdout or "")[-1200:]); print((_r.stderr or "")[-600:])
        return
    if _enc == "ffmpeg":
        _sp.run(["ffmpeg", "-y", "-loglevel", "error", "-i", _wav,
                 "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                 "-codec:a", "libmp3lame", "-q:a", "2", _mp3])
    else:
        _sp.run(["lame", "--quiet", "-q", "2", _wav, _mp3])
    if _os.path.exists(_mp3):
        print(f"   ✓ wrote {_mp3}  ({secs:.0f}s, {_os.path.getsize(_mp3):,} B) — "
              f"this is exactly what the ShaderToy Sound tab plays")
    else:
        print("   ✗ mp3 encode failed (WAV is at " + _wav + ")")


def _pack_build_into_png(common_path, sound_path, png_path):
    """Repackage a finished (embedded) VQ build's const-array data into ONE PNG.

    The PNG is purely a data-packaging vehicle — the decoder logic is unchanged.
    Every `const ivec4 NAME[..] = ivec4[](...)` array (patterns, VQ codes/codebook,
    seek tables, …) has its bytes (big-endian per int, x→y→z→w, matching the
    GLSL `_extractByte`) concatenated into one blob; each array's byte offset is
    emitted as `NAME_PNG_OFF`. Every fetcher's body is rewritten from
    `_extractByte(NAME[byteIdx>>4], byteIdx&15)` to `getByte(NAME_PNG_OFF + byteIdx)`,
    where getByte texelFetches the PNG (a pixel = RGBA = 4 bytes). Const arrays are
    stripped and USE_EMBEDDED_DATA flipped to 0. Returns (total_data_bytes, offsets).
    """
    import re as _re, struct as _struct
    from PIL import Image as _Image
    common = open(common_path).read()
    sound  = open(sound_path).read()

    def _logical(nm):
        m = _re.match(r'(.*?)(\d+)$', nm); return m.group(1) if m else nm

    # 1. Parse every const ivec4 array (both tabs), grouped by logical name.
    arr_chunks = {}                      # logical -> {chunk_idx -> [ints]}
    _decl_re = _re.compile(
        r'(?:^//[^\n]*\n)?[ \t]*const ivec4 ([A-Za-z_]+\d*)\[\d+\][ \t]*=[ \t]*ivec4\[\]\((.*?)\);\n?',
        _re.S | _re.M)
    for src in (common, sound):
        for m in _decl_re.finditer(src):
            nm = m.group(1)
            ints = [int(x) for x in _re.findall(r'-?\d+', m.group(2))]
            cm = _re.match(r'.*?(\d+)$', nm)
            ch = int(cm.group(1)) if cm else 0
            arr_chunks.setdefault(_logical(nm), {})[ch] = ints

    # Only the fetcher-backed DATA arrays get moved to the PNG. Any other
    # `const ivec4` array (indexed inline) is left untouched.
    _KNOWN = {'patDict', 'patBitmap', 'patIdx', 'patRowSeek', 'rowStartTick',
              'vqCodes', 'vqCodebook'}

    def _stream(lg):
        out = bytearray()
        for ch in sorted(arr_chunks[lg]):
            for v in arr_chunks[lg][ch]:
                out += _struct.pack('>i', v if v < 0 else v & 0xFFFFFFFF)
        return out

    order = [lg for lg in sorted(arr_chunks) if lg in _KNOWN]
    blob = bytearray(); off = {}
    for lg in order:
        off[lg] = len(blob); blob += _stream(lg)

    # 2. Write the PNG: [magic 'MOD',0][blob] into a 1024x1024 RGBA texture.
    TEX = 1024; cap = TEX * TEX * 4
    data = bytes([77, 79, 68, 0]) + bytes(blob)   # getByte() skips the 4-byte magic
    if len(data) > cap:
        raise ValueError(f"--png: packed data {len(data)} B exceeds PNG capacity {cap} B")
    buf = bytearray(cap); buf[:len(data)] = data
    _Image.frombytes('RGBA', (TEX, TEX), bytes(buf)).save(png_path, format='PNG', optimize=True)

    # 3. Offset #defines (uppercased logical name + _PNG_OFF) for Common.
    def _defname(lg): return lg.upper() + '_PNG_OFF'
    defines = "".join(f"#define {_defname(lg):24s} {off[lg]}\n" for lg in order)

    # 4. PNG data fetch primitives. fetchPixel() = ONE texelFetch = ONE RGBA
    #    pixel = 4 data bytes (big-endian r=MSB..a=LSB, matching _extractByte's
    #    `shift = 24 - byteInInt*8`). getByte() extracts one byte from a pixel;
    #    getU32() reads 4 consecutive bytes (little-endian, the form the VQ code
    #    bitstream wants) in AT MOST 2 texelFetches — so the hot per-sample
    #    4-byte code read is 1–2 fetches instead of 4.
    getbyte = (
        "#ifndef MUL1\n#define MUL1 255.0\n#endif\n"
        "// One texelFetch = one RGBA pixel = 4 data bytes (r=MSB .. a=LSB).\n"
        "int fetchPixel(int pi) {\n"
        "    vec4 p = texelFetch(iChannel0, ivec2(pi & 1023, pi >> 10), 0);\n"
        "    return (int(p.r*MUL1+0.5)<<24)|(int(p.g*MUL1+0.5)<<16)|(int(p.b*MUL1+0.5)<<8)|int(p.a*MUL1+0.5);\n"
        "}\n"
        "// One data byte (the +4 skips the 4-byte magic). One texelFetch.\n"
        "int getByte(int byteIndex) {\n"
        "    int ai = byteIndex + 4;\n"
        "    return (fetchPixel(ai >> 2) >> (24 - (ai & 3) * 8)) & 0xFF;\n"
        "}\n"
        "// Four consecutive data bytes, little-endian, in <=2 texelFetches.\n"
        "int getU32(int b) {\n"
        "    int ai = b + 4; int pi = ai >> 2; int pos = ai & 3;\n"
        "    int w0 = fetchPixel(pi);\n"
        "    if (pos == 0) return ((w0>>24)&0xFF)|(((w0>>16)&0xFF)<<8)|(((w0>>8)&0xFF)<<16)|((w0&0xFF)<<24);\n"
        "    int w1 = fetchPixel(pi + 1);\n"
        "    int q[8];\n"
        "    q[0]=(w0>>24)&0xFF; q[1]=(w0>>16)&0xFF; q[2]=(w0>>8)&0xFF; q[3]=w0&0xFF;\n"
        "    q[4]=(w1>>24)&0xFF; q[5]=(w1>>16)&0xFF; q[6]=(w1>>8)&0xFF; q[7]=w1&0xFF;\n"
        "    return q[pos] | (q[pos+1]<<8) | (q[pos+2]<<16) | (q[pos+3]<<24);\n"
        "}\n")

    # 5. Map each byte-fetcher to its logical array, and rewrite its body.
    fmap = {'fetchDictByte': 'patDict', 'fetchBitmapByte': 'patBitmap',
            'fetchIdxByte': 'patIdx', 'fetchRowSeekByte': 'patRowSeek',
            'fetchCodesByte': 'vqCodes', 'fetchCodebookByte': 'vqCodebook'}

    def _rewrite_fetchers(src):
        # Collapse the per-sample 4-byte VQ-code reads (4× fetchCodesByte → 4
        # texelFetches) into ONE getU32 (1–2 texelFetches). Two emitted forms;
        # if neither matches (encoder changed) the build stays correct, just
        # uses the per-byte path. Done before the per-byte fetcher rewrites.
        if 'vqCodes' in off:
            _co = _defname('vqCodes')
            src = _re.sub(
                r'fetchCodesByte\((\w+)\)\s*\|\s*\(fetchCodesByte\(\1\s*\+?\s*1\)\s*<<\s*8\)\s*\|\s*'
                r'\(fetchCodesByte\(\1\s*\+?\s*2\)\s*<<\s*16\)\s*\|\s*\(fetchCodesByte\(\1\s*\+?\s*3\)\s*<<\s*24\)',
                lambda mm: f'getU32({_co} + {mm.group(1)})', src, flags=_re.S)
            src = _re.sub(
                r'int b0 = fetchCodesByte\((\w+)\);\s*int b1 = fetchCodesByte\(\1\s*\+\s*1\);\s*'
                r'int b2 = fetchCodesByte\(\1\s*\+\s*2\);\s*int b3 = fetchCodesByte\(\1\s*\+\s*3\);\s*'
                r'int combined = b0 \| \(b1 << 8\) \| \(b2 << 16\) \| \(b3 << 24\);',
                lambda mm: f'int combined = getU32({_co} + {mm.group(1)});', src, flags=_re.S)
        for fn, lg in fmap.items():
            if lg not in off:
                continue
            src = _re.sub(
                rf'int {fn}\(int (\w+)\)\s*\{{.*?\n\}}',
                lambda mm, fn=fn, lg=lg: f'int {fn}(int {mm.group(1)}) {{ return getByte({_defname(lg)} + {mm.group(1)}); }}',
                src, count=1, flags=_re.S)
        # fetchTick: 16-bit little-endian read from rowStartTick.
        if 'rowStartTick' in off:
            src = _re.sub(
                r'int fetchTick\(int (\w+)\)\s*\{.*?\n\}',
                lambda mm: (f'int fetchTick(int {mm.group(1)}) {{ int _b = {mm.group(1)} * 2; '
                            f'return getByte(ROWSTARTTICK_PNG_OFF + _b) | '
                            f'(getByte(ROWSTARTTICK_PNG_OFF + _b + 1) << 8); }}'),
                src, count=1, flags=_re.S)
        return src

    common = _rewrite_fetchers(common)
    sound  = _rewrite_fetchers(sound)

    # 6. Strip the now-dead const arrays from both tabs (ONLY the known data ones).
    def _strip_known(src):
        return _decl_re.sub(lambda m: '' if _logical(m.group(1)) in _KNOWN else m.group(0), src)
    common = _strip_known(common)
    sound  = _strip_known(sound)

    # 7. Inject getByte + offset defines into Common (once), flip USE_EMBEDDED_DATA.
    if 'int getByte(' not in common:
        common = common.replace('int _extractByte(',
                                defines + getbyte + '\nint _extractByte(', 1)
    common = _re.sub(r'#define\s+USE_EMBEDDED_DATA\s+\d+', '#define USE_EMBEDDED_DATA 0', common)
    sound  = _re.sub(r'#define\s+USE_EMBEDDED_DATA\s+\d+', '#define USE_EMBEDDED_DATA 0', sound)

    open(common_path, 'w').write(common)
    open(sound_path, 'w').write(sound)
    return len(data), off


def _emit_html_player(mod, html_file, downsample, vec_dim):
    """Generate the standalone HTML player + inject autoplay.

    Single source of truth used by BOTH the normal single-build path and the
    auto-split parent (which emits ONE full-song HTML — the HTML player has no
    180s limit, so one file covers the whole song; only the ShaderToy GLSL is
    split into ≤180s parts). The player itself is RESTORED from
    mod_player_archived.py: the user confirmed it "played the tracks correctly"
    and it has horizontal click-to-scroll tracks, a "tracks N of M" indicator,
    and the stronger dying-voice crossfade declick that the stripped-down
    in-module create_fixed_player_html lost. Compatible because both consume the
    same mod object (samples / patterns / song_positions / num_channels)."""
    import mod_player_archived as _mp_html
    _mp_html.create_fixed_player_html(mod, html_file, downsample, compress=True, vec_dim=vec_dim)
    # Inject autoplay (browsers block autoplay-with-sound before a user gesture,
    # so try on load AND start on the first click / key / touch anywhere).
    try:
        with open(html_file, 'r') as _hf:
            _html_src = _hf.read()
        _autoplay_js = (
            "\n// ── Auto-start playback (injected by mod_player.py) ──\n"
            "(function(){\n"
            "  function _as(){ try{ if(typeof player!=='undefined' && player && !player.isPlaying) player.play(); }catch(e){} }\n"
            "  function _kick(){ _as(); if(typeof player!=='undefined' && player && player.audioCtx && player.audioCtx.state==='running'){\n"
            "    window.removeEventListener('click',_kick,true); window.removeEventListener('keydown',_kick,true); window.removeEventListener('touchstart',_kick,true); } }\n"
            "  if(document.readyState!=='loading') _as(); else window.addEventListener('DOMContentLoaded',_as);\n"
            "  window.addEventListener('load',_as);\n"
            "  window.addEventListener('click',_kick,true); window.addEventListener('keydown',_kick,true); window.addEventListener('touchstart',_kick,true);\n"
            "})();\n"
        )
        if 'Auto-start playback (injected by mod_player.py)' not in _html_src \
                and '_tryPlay' not in _html_src:
            _isp = _html_src.rfind('</script>')
            if _isp != -1:
                _html_src = _html_src[:_isp] + _autoplay_js + _html_src[_isp:]
                with open(html_file, 'w') as _hf:
                    _hf.write(_html_src)
                print("   ✓ autoplay injected (starts on load; falls back to first interaction)")
    except Exception as _e:
        print(f"   ✗ autoplay injection skipped: {_e}")


def main():
    import argparse
    class _ArgFmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        """Combined formatter: preserves newlines AND shows '(default: …)'."""
        pass
    parser = argparse.ArgumentParser(
        prog='mod_player.py',
        description='MOD/S3M/IT → HTML player + ShaderToy GLSL (+ optional PNG samples).',
        formatter_class=_ArgFmt,
        epilog=(
            "Examples:\n"
            "  # Standard ShaderToy build — embedded, no DSP, fits most GPUs:\n"
            "  python3 mod_player.py SONG.S3M --no-dsp --downsample 2\n\n"
            "  # Full-rate, highest quality (may exceed a tight GPU's limit):\n"
            "  python3 mod_player.py SONG.S3M --no-dsp --bitrate hi\n\n"
            "  # Audition only order positions 35-36 (speed carried over from earlier):\n"
            "  python3 mod_player.py SONG.S3M --positions 35-36 --no-dsp\n\n"
            "Input formats: .mod  .s3m  .it   (.xm not yet implemented)\n"
            "Outputs:       SONG_player.html, SONG_shadertoy_{common,sound,bufferA,image}.glsl,\n"
            "               SONG_shadertoy.json (one-click ShaderToy import)\n"))
    # Friendlier argparse errors: on a bad/missing argument, print a short
    # message + usage + a copy-paste example instead of just the terse default.
    def _friendly_argerror(message, _p=parser):
        sys.stderr.write(f"\n❌  {message}\n\n")
        _p.print_usage(sys.stderr)
        sys.stderr.write(
            "\n  Example:   python3 mod_player.py SONG.S3M --no-dsp --downsample 2\n"
            "  Full help: python3 mod_player.py --help\n\n")
        sys.exit(2)
    parser.error = _friendly_argerror
    parser.add_argument('modfile', help='MOD/S3M/IT file to play')
    parser.add_argument('--downsample', type=int, default=2,
                        help='Sample decimation factor: 1=full-rate, 2=22kHz, 4=11kHz. '
                             '(DEFAULT 2.) '
                             'HF percussion (cymbals/rides) gets max(1,DS//2) to keep shimmer.')
    parser.add_argument('--bitrate', choices=['lo','med','hi','ultra'], default='hi',
                        help='RVQ codebook size (mp3-style quality knob). '
                             'lo=K(128,64) 13b/pair smallest+grainy, med=K(256,128) 15b/pair balanced, '
                             'hi=K(512,256) 17b/pair sharper, ultra=K(1024,512) 19b/pair near-transparent.')
    parser.add_argument('--vec-dim', type=int, default=8, choices=[2, 4, 8],
                        help='RVQ vector dimensionality. 8=smallest (~2.1 bits/sample), '
                             '4=medium (4.25 bits/sample), 2=highest fidelity (8.5 bits/sample).')
    parser.add_argument('--resampler', choices=['linear','bspline','lanczos3'],
                        default=None,   # None → NORMAL_DEFAULTS picks lanczos3
                        help='Sample resampler. linear=2-tap (cheapest, ProTracker-style), '
                             'bspline=4-tap cubic (smooth, slightly softer HF), '
                             'lanczos3=6-tap sinc (sharpest/brightest — DEFAULT). '
                             'Use --resampler bspline for a softer sound or to save GPU headroom. '
                             'Default: lanczos3.')
    parser.add_argument('--aa', action='store_true', default=False,
                        help='Enable the gated ratio anti-aliasing (stateless '
                             'box integrator around getSampleF: averages K sub-'
                             'taps across the per-output-sample step, K tracks '
                             'the resample ratio; notes not pitched up are '
                             'bit-identical / zero cost). Suppresses alias '
                             'whine on high/pitched-up notes — most audible on '
                             'full-rate --raw-perc drums. NOTE: this is a '
                             'deliberate "cleaner than the oracle" divergence — '
                             'real Impulse Tracker / MikIT use plain 2-tap '
                             'linear with NO anti-aliasing, so --aa is NOT more '
                             '1:1, just nicer-sounding. Off by default; emits '
                             '#define AA_RESAMPLE 1 when set.')
    parser.add_argument('--no-json', dest='emit_json', action='store_false',
                        default=True,
                        help='Skip generating the {base}_shadertoy.json import file '
                             '(JSON is generated by default).')
    def _viz_arg(v):
        try:
            iv = int(v)
        except (TypeError, ValueError):
            iv = None
        if iv is None or not (0 <= iv <= 9):
            raise argparse.ArgumentTypeError(
                "invalid choice: %r (choose from [0..9], 0=no backdrop viz)" % v)
        return iv
    parser.add_argument('--viz', type=_viz_arg, default=6, metavar='[0..9]',
                        help='Image-tab visualizer (choose from [0..9]):\n'
                             '  0 = no backdrop viz  (black backdrop, fastest compile)\n'
                             '  1 = Reactive 001     (PAEz fork — SDF circles + cosmic web)  ← default\n'
                             '  2 = Fluxline Surfer  (mrange — DR2 dodecahedron + glowtracer)\n'
                             '  3 = Zuvuya           (city/stars + audio-reactive curtain)\n'
                             '  4 = Maya             (raymarched fractal tunnel-warp)\n'
                             '  5 = Dodecahedron     (Philip Bertani — DR2 IFS fractal raymarcher)\n'
                             '  6 = Disco Combined   (smoke spotlights + lasers/clouds, time-driven)\n'
                             '  7 = Sparkly 4D       (Philip Bertani — 4D IFS volumetric raymarcher)\n'
                             '  8 = Skywalker        (orblivius — flying-curve terrain + sync stars)\n'
                             '  9 = Music in the DNA  (jaszunio15/enbe fork — DNA helix + parallax dunes)')
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

    parser.add_argument('--positions', dest='positions', type=str, default=None, metavar='A-B',
                        help='Render only order positions A..B (0-based, as shown on the Image '
                             'tab), e.g. --positions 35-36 or --positions 35. The build plays '
                             'JUST that slice, but the running speed/tempo are first computed by '
                             'walking positions 0..A-1 break-aware, so the '
                             'slice plays at the SAME speed it would inside the full song — '
                             'never the file-header default. Use for isolating/auditioning a '
                             'section (e.g. the inst-25 porta lead) without hand-trimming a .S3M.')

    parser.add_argument('--intro-silence', dest='intro_silence', type=float, default=0.0, metavar='SEC',
                        help='Seconds of silence before the song starts (Sound tab holds, then plays '
                             'from row 0; the Image/BufferA visualizer is offset to stay in sync). '
                             'Default 0.0 = music starts immediately. Use e.g. --intro-silence 10 to '
                             'let an Image-tab loading splash render before the audio kicks in.')

    parser.add_argument('--mp3', dest='mp3', action='store_true', default=False,
                        help='After generating the GLSL, render an .mp3 of the actual ShaderToy '
                             'Sound tab on CPU (glslang -> spirv-cross -> clang -> WAV -> mp3 via '
                             'sound_exec.py). This is the SAME audio ShaderToy plays — handy for '
                             'quick listening/sharing without opening the site. Needs the '
                             'glslang+spirv-cross+clang toolchain and ffmpeg/lame; if missing, '
                             'prints how to install it instead of failing.')
    parser.add_argument('--mp3-secs', dest='mp3_secs', type=float, default=180.0, metavar='SEC',
                        help='Duration to render for --mp3 (default 180 = the full ShaderToy cap). '
                             'CPU render is ~real-time-ish, so lower this (e.g. 30) for a quick preview.')
    parser.add_argument('--xfade', dest='xfade', type=int, default=64, metavar='SAMPLES',
                        help='Retrigger/note-on declick crossfade length in samples (default 64 = '
                             '1.45ms). On a same-channel sample restart the OLD voice ramps down and '
                             'the NEW ramps up over this window. If you still hear clicks on busy '
                             'leads, raise it (e.g. 256 = 5.8ms, 512 = 11.6ms) and re-check by ear '
                             '(--mp3). Costs no extra GPU private-vars, just a longer blend region.')

    # NOTE: long songs are AUTO-SPLIT by default — no flag needed. A song longer
    # than ShaderToy's ~180s audio cap is emitted as several self-contained
    # bundles SONG_shadertoy_part1_*, SONG_shadertoy_part2_*, … (each a contiguous
    # ≤180s order-position slice with speed/tempo carried over, so it plays at its
    # true in-song speed). Import each part's .json into ShaderToy separately.
    # (Single-shader runtime part-switching is impossible — the Sound tab is
    # precomputed once and can't read any buffer/texture/mouse/keyboard — so
    # separate bundles are the only way.)
    parser.add_argument('--start', dest='start', type=int, default=None, metavar='POS',
                        help='Force a manual 2-way split at order position POS instead of the automatic '
                             'split: emit exactly TWO bundles — SONG_shadertoy_part1_* = positions 0..POS-1, '
                             'and SONG_shadertoy_part2_* = positions POS..end. Lets you pick the split point '
                             'at a musical boundary. Carries speed/tempo over so each part plays at its true '
                             'in-song speed. POS is an order position (the Image-tab numbering), 1..(numPositions-1).')
    # Internal: override the output base name (the auto-split / --start driver
    # passes {base}_partK so the child writes an isolated {base}_partK_* set,
    # which the driver then renames to {base}_shadertoy_partK_*). Not for normal use.
    parser.add_argument('--output-base', dest='output_base', type=str, default=None,
                        help=argparse.SUPPRESS)
    # Internal: strict 180s trim (drop the straddling position so the slice ends
    # ≤180s exactly). Set by the --parts driver so each part fits the buffer with
    # no overflow and the next part picks up exactly where this one ended.
    parser.add_argument('--cap-strict', dest='cap_strict', action='store_true',
                        default=False, help=argparse.SUPPRESS)
    # Internal: skip HTML-player generation. Set by the auto-split driver on the
    # per-part GLSL children so ONLY the parent emits a single full-song HTML
    # player (the HTML player has no 180s limit, so one file covers the whole
    # song — no per-part HTMLs).
    parser.add_argument('--no-html', dest='no_html', action='store_true',
                        default=False, help=argparse.SUPPRESS)

    parser.add_argument('--no-rvq2', dest='no_rvq2', action='store_true', default=False,
                        help='Skip RVQ stage 2 (residual quantization).  Drops ~40%% of '
                             'sample-data const arrays from Sound tab → faster compile. '
                             'Quality cost: ~4 dB SNR (sounds noisier but pitch is unchanged). '
                             'IMPORTANT: when re-pasting into ShaderToy, paste BOTH the new '
                             'Common AND new Sound — otherwise mismatched RVQ_BITS produces '
                             'high-pitch garbage from a stale Common reading 15-bit-packed '
                             'codes that were actually written at 8 bits.')
    parser.add_argument('--preserve', dest='preserve', type=str, default='',
                        help='Comma-separated 1-based instrument numbers stored UNCOMPRESSED '
                             '(raw int8, no VQ quantization) for perfect quality — e.g. '
                             '--preserve 28,25 keeps the lead/voice samples pristine while '
                             'the rest stay VQ-compressed small. getSample() intercepts those '
                             'instruments\' index ranges and reads the raw array instead of '
                             'VQ-decoding (resampled to the same rate as the VQ stream).')
    parser.add_argument('--raw-perc', dest='raw_perc', action='store_true', default=True,
                        help='Auto-store percussion (kick/snare/hat/clap/cymbal — samples '
                             'the waveform classifier tags NOISE) UNCOMPRESSED, exactly like '
                             '--preserve but auto-detected. Percussion transients/noise are '
                             'the worst-hit by RVQ, so this keeps drums crisp and matching '
                             'the HTML player. Percussion samples are short → small size '
                             'cost. Default ON.')
    parser.add_argument('--no-raw-perc', dest='raw_perc', action='store_false',
                        help='Disable --raw-perc (let percussion be VQ-compressed too).')
    parser.add_argument('--raw-perc-budget', dest='raw_perc_budget',
                        type=int, default=28672, metavar='BYTES',
                        help='Max total raw (un-VQ) percussion bytes kept by '
                             '--raw-perc (shortest-first; the rest fall back '
                             'to VQ). Default 28672. The raw PCM becomes a big '
                             'const array (_presvPCM) in the Sound tab — on a '
                             'tight GPU that const-register/private-var load '
                             'can push the shader over ANGLE\'s limit (e.g. '
                             'jeff.it: full raw-perc = +42KB Sound = does not '
                             'fit). LOWER this to keep only the smallest '
                             'kick/hat pristine and still fit (e.g. 8192 ≈ one '
                             'short drum); 0 ≈ effectively --no-raw-perc. '
                             'Quality-vs-fit dial when full raw-perc overflows.')
    parser.add_argument('--png', dest='use_png', action='store_true', default=False,
                        help='Use the PNG-loaded data path (samples/patterns read via texelFetch from '
                             'iChannel0 = a 1024×1024 RGBA PNG = 4 MB) instead of VQ-encoded const arrays, '
                             'AND write SONG_player_data.png. DEFAULT OFF: normally NO PNG is written and '
                             'the build is embedded. Because one PNG holds the WHOLE song, a --png build is '
                             'a SINGLE bundle (one set of .glsl + one .png) — it is NOT auto-split into '
                             'parts and the song is NOT trimmed to 180s. Smaller Common source = faster '
                             'compile, but raw 8-bit samples (no RVQ) so quality differs. ShaderToy setup: '
                             'Image/Common iChannel0 = SONG_player_data.png via the Unofficial Plugin '
                             '"Custom Textures".')
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
    parser.add_argument('--no-dsp', dest='no_dsp', action='store_true', default=False,
                        help="MASTER SWITCH: disable ALL DSP effect processing in the output "
                             "shaders (3D surround, FAT4X exciter, PhatBass; velvet/comb reverb "
                             "are already off in v1.61). Forces ENABLE_3D/FAT/PHATBASS/"
                             "VELVETREVERB/COMBREVERB = 0 and WINS over any individual "
                             "--surround/--phatbass/--fat4x passed alongside it. This is the "
                             "lightest Sound-tab path (no DSP private-vars) → best chance of "
                             "fitting ANGLE's per-GPU private-variable ceiling. Note: AA is a "
                             "resampler option, NOT part of the DSP chain — control it with --aa "
                             "(default off).")
    parser.add_argument('--fft-n', dest='fft_n',
                        type=int, choices=[64,128,256,512,1024,2048], default=None,
                        help="FFT size for Buffer A spectrum. Larger = more frequency "
                             "resolution but slower compile. Default: 1024 (or 128 if "
                             "--max-compat without override).")
    parser.add_argument('--max-compat', action='store_true', default=False,
                        help='[NO-OP — max-compat is now the DEFAULT in v1.40+ (current: v1.61)] '
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
    # No arguments at all → show full help (with the examples) and exit 0,
    # rather than argparse's terse "the following arguments are required".
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()

    # args.emit_json defaults True; pass --no-json to suppress JSON output.

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
            'phatbass':               False,      # off by default — adds private vars; opt in with --phatbass
            'fat4x':                  False,      # off by default — adds private vars; opt in with --fat4x
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
        'phatbass':               True,       # ON by default (user) — --no-phatbass to disable; ENABLE_PHATBASS=1
        'fat4x':                  True,       # ON by default (user) — --no-fat4x to disable; ENABLE_FAT=1
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
    # --no-dsp: master override — force the ENTIRE DSP effects chain OFF (3D
    # surround, FAT4X, PhatBass) regardless of individual flags. enableVelvet/
    # CombReverb are already hardwired false in the GLSL template, so the five
    # ENABLE_* gates all emit 0. Applied AFTER the per-flag derivation so it
    # wins over an explicit --surround/--phatbass/--fat4x. Lightest Sound tab.
    if getattr(args, 'no_dsp', False):
        args._compat_no_phatbass = True
        args._compat_no_fat      = True
        args._compat_no_surround = True
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
    if args.use_png: _flags.append('--png')
    _flags.append(f'--vec-dim {args.vec_dim}')
    _flags.append(f'--downsample {args.downsample}')
    _flags.append(f'--resampler {args.resampler}')
    _flags.append(f'--bitrate {args.bitrate}')
    print(_prefix + ' '.join(_flags))
    
    # Detect file format
    try:
        fmt = detect_module_format(args.modfile)
    except FileNotFoundError:
        sys.exit(f"\nError: module file {args.modfile!r} not found.\nVerify the path and try again.\n")
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
    elif fmt == 'IT':
        # IT support via mod_player.py's maintained ITFile loader (reused
        # by module import — ITFile has ~12 helper deps, so copying would
        # rot). ITFile translates IT letter-effects → MOD numbering AT
        # PARSE TIME (_xm_or_it_effect_to_mod, is_xm=False) and decodes
        # IT-packed samples, so downstream treats it exactly like a MOD
        # (is_s3m=False → no S3M re-remap; pattern-player path, no timeline
        # DSP). Phase-1: envelopes/NNA/filters/c5speed-precision dropped.
        import os as _os_it, sys as _sys_it
        _here_it = _os_it.path.dirname(_os_it.path.abspath(__file__))
        if _here_it not in _sys_it.path:
            _sys_it.path.insert(0, _here_it)
        import mod_player_archived as _mp_it
        it = _mp_it.ITFile(args.modfile)
        print(f"🎵 {it.title}")
        print(f"   Patterns: {it.num_patterns}, Channels: {it.num_channels}")
        print(f"   Speed: {it.initial_speed}, Tempo: {it.initial_tempo}")
        print("   ⚠️  IT support (reused from mod_player.py): IT-packed "
              "samples decoded, effects IT→MOD at parse.")
        mod = type('obj', (object,), {
            'title':          it.title,
            'samples':        it.samples,
            'num_patterns':   it.num_patterns,
            'song_length':    len(it.song_positions),
            'song_positions': it.song_positions,
            'orders':         it.song_positions,
            'patterns':       it.patterns,          # MOD-numbered effects
            'num_channels':   it.num_channels,
            'initial_speed':  it.initial_speed,
            'initial_tempo':  it.initial_tempo,
            'channel_settings': list(getattr(it, 'channel_settings', []) or []),
            'is_s3m':         False,
            'is_it':          True,
        })()
        # Attach the inst_table so the NNA envelope-follower port can build
        # PER-INSTRUMENT env arrays (not per-sample). IT envelopes are an
        # instrument property; ITFile's per-sample env_pts is a lossy
        # collapse (one sample referenced by N instruments → N different
        # envelopes; ITFile keeps only one). 34/99 jeff.it instruments
        # have note_to_sample[60] != instrument_index → sample-indexed
        # arrays would read the WRONG env. Also lets us honor env_on
        # per-instrument (e.g. jeff.it inst 1 env_on=False → no wooump).
        mod._it_inst_table = getattr(it, 'inst_table', None)
        # ── Variable-row pattern support (jeff.it 128-row root-cause fix) ─
        # ITFile keeps the legacy 64-row truncated form in `it.patterns`
        # (every MOD-shaped consumer needs fixed 64-row). IT files whose
        # patterns are NOT 64 rows (jeff.it: 128/120/112/80/32-row) had
        # rows 64+ silently DELETED and short patterns padded with silence
        # → the whole song desynced from the first pattern ("keys all over
        # the keyboard", ~0 corr vs the HTML oracle). Fix WITHOUT touching
        # the 64-row storage stride / getNote / VQ-crunch / GLSL: SPLIT each
        # >64-row pattern into ceil(R/64) sub-patterns of ≤64 rows and
        # EXPAND the order list so they play back-to-back. Any sub-pattern
        # with <64 real rows (a short final chunk, or a natively-short
        # pattern) gets a synthetic MOD pattern-break (effect 0x0D) on its
        # last real row in an effect-free channel — Dxx is already a
        # first-class effect for the rowStartTick walker, the
        # patStartRow/patRowOffset builder AND the GLSL, so the engine
        # advances after exactly the real row count with ZERO downstream
        # changes. Safe for jeff.it (its long patterns carry no B/D/loop —
        # verified); a >64-row pattern containing its own B/D is flagged.
        _PF = getattr(it, 'patterns_full', None)
        _PR = getattr(it, 'pattern_rows', None)
        if _PF and _PR and any(r > 64 or r < 64 for r in _PR):
            _nch = mod.num_channels
            _new_pats = []
            _map = []          # orig pattern idx -> [sub-pattern indices]
            _n_split = _n_brk = _n_clobber = _n_bd = 0
            for _p in range(len(_PF)):
                _cells = _PF[_p]
                _R = len(_cells)
                _chunks = [ _cells[_b:_b+64] for _b in range(0, max(1, _R), 64) ]
                if _R > 64:
                    _n_split += 1
                    for _row in _cells:                       # mid-pattern B/D?
                        for _cl in _row:
                            if _cl.get('effect', 0) in (0x0B, 0x0D):
                                _n_bd += 1
                _idxs = []
                for _seg in _chunks:
                    _sr = len(_seg)
                    _sub = [[dict(_cl) for _cl in _row] for _row in _seg]
                    if 0 < _sr < 64:                          # force advance
                        _last = _sub[_sr - 1]
                        _free = next((_c for _c in range(min(_nch, len(_last)))
                                      if _last[_c].get('effect', 0) == 0), None)
                        if _free is None:
                            _free = 0
                            _n_clobber += 1
                        if _free < len(_last):
                            _last[_free]['effect'] = 0x0D     # MOD pattern break
                            _last[_free]['param']  = 0
                            _n_brk += 1
                    _idxs.append(len(_new_pats))
                    _new_pats.append(_sub)
                _map.append(_idxs)
            _old_ord = list(mod.song_positions)
            _new_ord = []
            for _e in _old_ord:
                if isinstance(_e, int) and 0 <= _e < len(_map):
                    _new_ord.extend(_map[_e])
                else:
                    _new_ord.append(_e)                       # 254/255 end-marks
            mod.patterns      = _new_pats
            mod.num_patterns  = len(_new_pats)
            mod.song_positions = _new_ord
            mod.orders         = _new_ord
            mod.song_length    = len(_new_ord)
            print(f"   ✓ Variable-row split: {_n_split} >64-row patterns → "
                  f"{len(_new_pats)} sub-patterns, order {len(_old_ord)}→"
                  f"{len(_new_ord)}, {_n_brk} synthetic breaks"
                  + (f", ⚠{_n_bd} mid-pattern B/D in long patterns (split "
                     f"may mis-time those)" if _n_bd else "")
                  + (f", ⚠{_n_clobber} rows had no free effect slot"
                     if _n_clobber else ""))
        # ── MOD+ pattern-player path (the budget-viable IT route) ───────
        # The NNA timeline (load_it_native→tlGetOutput) was PROVEN an
        # architectural dead-end for loop/envelope-heavy IT (jeff.it):
        # corr 0.004 in GLSL AND 0.106 as a pure-Python replay of the exact
        # tlGetOutput math on render_pcm's own PCM/segments — so the lossy
        # compaction (fpos=pos0+dt·freq, single loop-wrap, 2-piece vol),
        # not VQ/index/downsample, is the defect, and a faithful per-tick
        # encoding (~190K states ≈ 740KB-1.5MB) busts the no-PNG embedded /
        # ANGLE budget 5-10×. See project_it_timeline_architectural_deadend.
        # MOD+ runs IT through the SAME proven pattern-player as S3M (1:1):
        # ITFile pre-translates IT effects→MOD at parse, the c5_speed→c2sp
        # rate-scale (just fixed) restores correct pitch, and the injected
        # stateless vol-envelope carries the key IT-over-MOD feature. It
        # fits the GPU like S3M (no timeline arrays / dead-strip / PNG).
        mod._it_timeline_glsl = None
        mod._it_timeline_tps  = None
        _ie = sum(1 for s in it.samples
                  if isinstance(s, dict) and s.get('env_pts'))
        _nc5 = sum(1 for s in it.samples
                   if isinstance(s, dict) and (s.get('c5_speed') or 8363) != 8363)
        print(f"   ✓ MOD+ pattern-player path (no timeline). "
              f"{_ie} samples carry a vol/pan envelope; {_nc5} carry a "
              f"non-8363 c5_speed → c2sp pitch rate-scale.")
    elif fmt == 'XM':
        # XM support via mod_player_archived.py's XMFile parser (reused
        # exactly like ITFile above).  XMFile: delta-decodes 8/16-bit samples
        # (16-bit → 8-bit), reads vol envelopes + fadeout + rel_note per
        # instrument, truncates/pads all patterns to 64 rows, and translates
        # XM letter-effects → MOD numbering at parse time via the shared
        # _xm_or_it_effect_to_mod table.  Output is MOD-compatible dicts so
        # the entire downstream (VQ encoder, pattern-player, GLSL generator)
        # needs no changes — the _XMITtoVQAdapter at line ~11182 already
        # handles both IT and XM.
        import os as _os_xm, sys as _sys_xm
        _here_xm = _os_xm.path.dirname(_os_xm.path.abspath(__file__))
        if _here_xm not in _sys_xm.path:
            _sys_xm.path.insert(0, _here_xm)
        import mod_player_archived as _mp_xm
        xm = _mp_xm.XMFile(args.modfile)
        print(f"🎵 {xm.title}")
        print(f"   Patterns: {xm.num_patterns}, Channels: {xm.num_channels}")
        print(f"   Speed: {xm.initial_speed}, Tempo: {xm.initial_tempo}")
        print("   ⚠️  XM support (via mod_player_archived.py): 16-bit→8-bit, "
              "vol envelopes + fadeout read, effects XM→MOD at parse.")
        mod = type('obj', (object,), {
            'title':          xm.title,
            'samples':        xm.samples,
            'num_patterns':   xm.num_patterns,
            'song_length':    len(xm.song_positions),
            'song_positions': xm.song_positions,
            'orders':         xm.song_positions,
            'patterns':       xm.patterns,
            'num_channels':   xm.num_channels,
            'initial_speed':  xm.initial_speed,
            'initial_tempo':  xm.initial_tempo,
            'channel_settings': [],
            'is_s3m':         False,
            'is_xm':          True,
            'is_it':          False,
        })()
        mod._it_timeline_glsl = None
        mod._it_timeline_tps  = None
        _ie_xm = sum(1 for s in xm.samples
                     if isinstance(s, dict) and s.get('env_pts'))
        print(f"   ✓ MOD+ pattern-player path (no timeline). "
              f"{_ie_xm} instruments carry a vol envelope.")
    elif fmt in ('STM', 'MTM'):
        raise ValueError(
            f"{fmt} format is not yet implemented in this player. "
            f"Currently supported: MOD (full), S3M (partial), XM (Phase-1), "
            f"IT (Phase-1). Patches welcome."
        )
    else:
        raise ValueError(
            f"Unknown module format for {args.modfile!r}: signature check failed and "
            f"file extension is not in the recognized set "
            f"(.mod / .s3m / .xm / .it / .stm / .mtm / .m15 / .nst / .wow). "
            f"If this really is a tracker module, rename it with the correct extension "
            f"or report the file so we can add a signature for it."
        )

    # Stash the intro-silence (seconds) on the mod so the Common-tab generator
    # (create_shadertoy_glsl f-string) can emit it into #define INTRO_SILENCE_S.
    mod._intro_silence_s = float(getattr(args, 'intro_silence', 0.0) or 0.0)

    # ── No PNG unless --png: remove any STALE data PNG from a prior --png run ────
    # The build never WRITES a *_player_data.png without --png, but a leftover one
    # from an earlier --png build can linger and look like it was just generated.
    # On a non-png parent/single build (not a child, which sets --output-base),
    # delete it so a non-png build leaves zero PNGs.
    if not getattr(args, 'use_png', False) and not getattr(args, 'output_base', None):
        _stale_png = os.path.splitext(os.path.basename(args.modfile))[0] + "_player_data.png"
        if os.path.exists(_stale_png):
            try:
                os.remove(_stale_png)
                print(f"   🧹 removed stale {_stale_png} (no --png → no PNG)")
            except Exception:
                pass

    # ── AUTO-SPLIT long songs into SEPARATE per-part bundles (SONG_shadertoy_partK_*) ──
    # ShaderToy's Sound tab is precomputed ONCE and cannot read any buffer,
    # texture, mouse or keyboard, so a single shader can't switch parts at runtime
    # (verified: "no texture access in sound shaders"). So a song longer than the
    # ~180s audio cap is emitted as several self-contained bundles, each a
    # contiguous order-position slice rendered by re-invoking this generator with
    # --positions (speed/tempo carried over → each plays at its true in-song
    # speed). Each part is ~1/N the data → each fits the GPU where the full song
    # would not. Import each part's .json into ShaderToy separately.
    #
    # Boundaries = ITERATIVE MAX-FILL: each child strict-trims (--cap-strict) its
    # slice to ≤180s and reports (via SONG_LENGTH) how many order positions it
    # kept; the next part begins exactly there → contiguous, gap-free, ≤180s BY
    # CONSTRUCTION (the child's row-time table is authoritative; a parent-side
    # estimate is used ONLY to decide whether to split at all).
    #
    # Triggers: a song > 180s auto-splits; --start POS forces a manual 2-way split.
    # Songs ≤ 180s → single normal build (no split).
    # SKIP auto-split when --positions is given: the user explicitly asked for a
    # specific order-position slice (e.g. one pattern) — render exactly that (with
    # global-effect carryover), don't auto-split the whole song. Also skip in the
    # per-part children (--output-base set) to avoid recursion.
    # SKIP auto-split for --png: the PNG data path holds the WHOLE song (a 1024²
    # RGBA texture = 4 MB, vs the embedded const-array path that forces the split),
    # so a --png build is a single bundle, never parts.
    _start_pos = getattr(args, 'start', None)
    if (not getattr(args, 'output_base', None) and not getattr(args, 'positions', None)
            and not getattr(args, 'use_png', False)):
        import subprocess as _subp, re as _re2, glob as _glob
        _base    = os.path.splitext(os.path.basename(args.modfile))[0]
        _sp_full = list(getattr(mod, 'song_positions', []) or [])
        _last    = len(_sp_full) - 1
        # Approximate full-song seconds — ONLY to decide whether to auto-split.
        _nch = getattr(mod, 'num_channels', 4) or 4
        _spd = getattr(mod, 'initial_speed', 6) or 6
        _tmp = getattr(mod, 'initial_tempo', 125) or 125
        # Native MOD patterns always carry 64 rows on disk but the VQ encoder
        # honours Effect D (Dxx pattern-break): it encodes only up to Dxx.
        # Match this estimate so we don't over-count and trigger a spurious split.
        _est_is_mod = not (getattr(mod,'is_s3m',False) or
                           getattr(mod,'is_it', False) or
                           getattr(mod,'is_xm', False))
        _est_next_start = 0   # row start carryover from Dxx
        _total = 0.0
        for _pat in _sp_full:
            _rows = mod.patterns[_pat] if 0 <= _pat < len(mod.patterns) else None
            if not _rows:
                _est_next_start = 0
                continue
            _est_row_start = _est_next_start
            _est_next_start = 0
            if _est_is_mod:
                _d_row = None
                for _ri in range(_est_row_start, len(_rows)):
                    for _ci in range(min(_nch, len(_rows[_ri]) if _rows[_ri] else 0)):
                        _c = _rows[_ri][_ci]
                        if _c.get('effect') == 0xD and _d_row is None:
                            _p = _c.get('param', 0)
                            _d_row = _ri
                            _est_next_start = ((_p >> 4) & 0xF) * 10 + (_p & 0xF)
                    if _d_row is not None:
                        break
                if _d_row is not None:
                    _rows = _rows[_est_row_start:_d_row + 1]
                else:
                    _rows = _rows[_est_row_start:]
            for _row in _rows:
                for _ci in range(min(_nch, len(_row))):
                    _c = _row[_ci]
                    _cmd = _c.get('command'); _inf = _c.get('info', 0)
                    _eff = _c.get('effect');  _prm = _c.get('param', 0)
                    if   _cmd == 1  and _inf > 0: _spd = _inf
                    elif _cmd == 20 and _inf > 0: _tmp = _inf
                    elif _eff == 0xF and _prm > 0:
                        if _prm < 0x20: _spd = _prm
                        else:           _tmp = _prm
                _total += float(_spd) / max(1.0, float(_tmp) * 0.4)
                if any((_c.get('command') in (2, 3)) or (_c.get('effect') in (0xB, 0xD))
                       for _c in _row):
                    break
        if _last >= 1 and (_start_pos is not None or _total > 180.0):
            # Pass-through argv: drop --start/--positions/--output-base (+values).
            _passthru = []; _skip = False
            for _tok in list(sys.argv[1:]):
                if _skip: _skip = False; continue
                if _tok in ('--start', '--positions', '--output-base'):
                    _skip = True; continue
                if any(_tok.startswith(_pre + '=') for _pre in
                       ('--start', '--positions', '--output-base')):
                    continue
                _passthru.append(_tok)

            def _emit_part(_K, _A, _B):
                # Render positions _A.._B as its own bundle, then rename to
                # {base}_shadertoy_part{K}_*  (part number AFTER 'shadertoy').
                # The child writes the isolated {base}_part{K}_* set (via
                # --output-base) so it never clobbers the default {base}_shadertoy_*
                # build. Returns how many order positions the child KEPT after its
                # strict ≤180s trim.
                _pbase = f"{_base}_part{_K}"
                _cmd = [sys.executable, sys.argv[0]] + _passthru + \
                       ['--positions', f'{_A}-{_B}', '--output-base', _pbase,
                        '--cap-strict', '--no-html']
                _r = _subp.run(_cmd)
                if _r.returncode != 0:
                    sys.exit(f"\n❌ part {_K} (positions {_A}-{_B}) failed (exit {_r.returncode}).\n")
                _kept = None
                try:
                    _m = _re2.search(r'#define\s+SONG_LENGTH\s+(\d+)',
                                     open(f"{_pbase}_shadertoy_common.glsl").read())
                    if _m: _kept = int(_m.group(1))
                except Exception:
                    pass
                # Rename {base}_part{K}_*  →  {base}_shadertoy_part{K}_*
                #   {base}_part{K}_shadertoy_common.glsl → {base}_shadertoy_part{K}_common.glsl
                #   {base}_part{K}_shadertoy.json        → {base}_shadertoy_part{K}.json
                #   {base}_part{K}_player.html           → {base}_shadertoy_part{K}_player.html
                for _src in _glob.glob(f"{_pbase}_*"):
                    _rest = _src[len(_pbase):]
                    if _rest.startswith("_shadertoy"):
                        _dst = f"{_base}_shadertoy_part{_K}" + _rest[len("_shadertoy"):]
                    else:
                        _dst = f"{_base}_shadertoy_part{_K}" + _rest
                    os.replace(_src, _dst)
                return _kept

            # ── Clean up ALL stale {base}_shadertoy* outputs from any prior build
            # (non-parts single-build files + old _partN files from runs that
            # emitted a different number of parts).  The current run will
            # re-generate everything fresh, so remove-then-write is safe.
            # Keep .wav / _exec.wav files (audio references, not regenerated).
            _stale_singles = [
                f"{_base}_shadertoy.json",           # non-parts single-build JSON
                f"{_base}_shadertoy_common.glsl",    # non-parts GLSL tabs
                f"{_base}_shadertoy_sound.glsl",
                f"{_base}_shadertoy_image.glsl",
                f"{_base}_shadertoy_bufferA.glsl",
            ]
            _stale_parts = [_sf for _sf in _glob.glob(f"{_base}_shadertoy_part*")
                            if not _sf.endswith('.wav')]
            _n_stale = 0
            for _sf2 in _stale_singles + _stale_parts:
                if os.path.exists(_sf2):
                    try: os.remove(_sf2); _n_stale += 1
                    except Exception: pass
            if _n_stale:
                print(f"   🧹 removed {_n_stale} stale {_base}_shadertoy_* file(s) (fresh build)")

            _emitted = []
            if _start_pos is not None:
                _M = max(1, min(int(_start_pos), _last))
                print(f"\n🧩 --start {_M}: 2-way split → part1 = positions 0-{_M-1}, "
                      f"part2 = positions {_M}-{_last}")
                for _K, (_A, _B) in enumerate([(0, _M - 1), (_M, _last)], 1):
                    print(f"\n   ── part {_K}/2: positions {_A}-{_B} → {_base}_shadertoy_part{_K}_* ──")
                    _kept = _emit_part(_K, _A, _B)
                    _emitted.append((_A, _B))
                    if _kept is not None and _kept < (_B - _A + 1):
                        print(f"   ⚠️  part {_K} exceeded 180s, trimmed to positions "
                              f"{_A}-{_A + _kept - 1}; positions {_A + _kept}-{_B} NOT covered. "
                              f"Omit --start for the gap-free auto-split.")
            else:
                print(f"\n🧩 auto-split: song ≈{_total:.0f}s > 180s cap → ≤180s part bundles")
                _cur = 0; _K = 0
                while _cur <= _last and _K < 64:
                    _K += 1
                    print(f"\n   ── part {_K}: positions {_cur}.. → {_base}_shadertoy_part{_K}_* ──")
                    _kept = _emit_part(_K, _cur, _last)
                    if not _kept or _kept < 1:
                        _kept = (_last - _cur + 1)
                    # STRICT trim kept exactly the positions that fit ≤180s and
                    # dropped the straddler → next part starts right after (gap-free,
                    # no overlap). The final part keeps the rest (≤180s), ends loop.
                    _end = min(_cur + _kept - 1, _last)
                    _emitted.append((_cur, _end))
                    _cur = _end + 1
            # ── If only 1 part ended up being emitted, strip the _part1 suffix ──
            # so the output looks identical to a normal (≤180s) single build.
            # Use explicit suffixes (_part1_.* and _part1.json) to avoid
            # accidentally matching stale _part10/_part11 files from prior runs.
            if len(_emitted) == 1:
                _p1_files = (sorted(_glob.glob(f"{_base}_shadertoy_part1_*")) +
                             _glob.glob(f"{_base}_shadertoy_part1.json"))
                for _src in _p1_files:
                    _rest = _src[len(f"{_base}_shadertoy_part1"):]
                    _dst  = f"{_base}_shadertoy" + _rest
                    os.replace(_src, _dst)

            if len(_emitted) == 1:
                _A, _B = _emitted[0]
                print(f"\n✅ Generated (positions {_A}-{_B}, fits in one ShaderToy bundle):")
                print(f"      {_base}_shadertoy.json   ← ShaderToy ▸ Import")
            else:
                print(f"\n✅ wrote {len(_emitted)} part bundle(s) covering ALL positions 0-{_last} "
                      f"(contiguous, gap-free, each ≤180s):")
                for _i, (_A, _B) in enumerate(_emitted, 1):
                    print(f"      {_base}_shadertoy_part{_i}.json   (positions {_A}-{_B})")
                print("   Import each part's .json into ShaderToy separately to hear the whole song.")
            # ── ONE full-song HTML player ───────────────────────────────────
            # The HTML player has no 180s limit, so a single file plays the
            # WHOLE song — no per-part HTMLs (the part children ran --no-html).
            print(f"\n🌐 full-song HTML player (entire song, single file):")
            try:
                _emit_html_player(mod, _base + "_player.html", args.downsample, args.vec_dim)
                print(f"      {_base}_player.html   (all positions 0-{_last} — open in a browser)")
            except Exception as _he:
                print(f"   ✗ full-song HTML generation failed: {_he}")
            return
        # (≤180s and no --start → fall through to the normal single build below)

    # ── --positions A-B: render only this order-position slice, with the
    # running speed/tempo carried over from positions 0..A-1 so the slice
    # plays at the SAME speed it has inside the full song (not the file
    # header default). MUST run before the trim/remap so A-B index the
    # full, original order (as shown on the Image tab). ──────────────────
    if getattr(args, 'positions', None):
        _sp = list(getattr(mod, 'song_positions', []) or [])
        if _sp:
            _pp = str(args.positions).split('-')
            try:
                _a = int(_pp[0]); _b = int(_pp[1]) if len(_pp) > 1 and _pp[1] != '' else _a
            except ValueError:
                _a = _b = 0
            _a = max(0, min(_a, len(_sp) - 1)); _b = max(_a, min(_b, len(_sp) - 1))
            # Walk positions 0..A-1 break-aware, tracking speed/tempo. Handles
            # BOTH S3M cells (command A=1 set-speed / T=20 set-tempo) and MOD/IT
            # cells (effect 0xF: param<0x20 → speed, else tempo). Position-break
            # = S3M B=2/C=3 or MOD Bxx=0xB/Dxx=0xD (stops the row walk early).
            _spd = getattr(mod, 'initial_speed', 6) or 6
            _tmp = getattr(mod, 'initial_tempo', 125) or 125
            _nch = getattr(mod, 'num_channels', 4) or 4
            for _si in range(_a):
                _pat = _sp[_si]
                _rows = mod.patterns[_pat] if 0 <= _pat < len(mod.patterns) else None
                if not _rows:
                    continue
                for _row in _rows:
                    for _ci in range(min(_nch, len(_row))):
                        _c = _row[_ci]
                        _cmd = _c.get('command'); _inf = _c.get('info', 0)
                        _eff = _c.get('effect');  _prm = _c.get('param', 0)
                        if   _cmd == 1  and _inf > 0: _spd = _inf
                        elif _cmd == 20 and _inf > 0: _tmp = _inf
                        elif _eff == 0xF and _prm > 0:
                            if _prm < 0x20: _spd = _prm
                            else:           _tmp = _prm
                    if any((_c.get('command') in (2, 3)) or (_c.get('effect') in (0xB, 0xD))
                           for _c in _row):
                        break
            mod.song_positions = _sp[_a:_b + 1]
            mod.orders = list(mod.song_positions)
            if hasattr(mod, 'song_length'):
                try: mod.song_length = len(mod.song_positions)
                except Exception: pass
            mod.initial_speed = _spd
            mod.initial_tempo = _tmp
            print(f"   ✂️  --positions {_a}-{_b}: rendering {_b - _a + 1} order "
                  f"position(s) {mod.song_positions} with carried-over "
                  f"speed={_spd} tempo={_tmp} (walked positions 0..{_a - 1})")

    # Trim the unplayable tail (audio past Shadertoy's ~180s buffer) so
    # the embedded build stays small. Conservative — keeps the position
    # straddling the cap. Runs before all encoding so the VQ pattern
    # crunch / rowStartTick / sample selection only cover what plays.
    # (A long song is auto-split earlier by the per-part driver, which re-invokes
    # this generator per part with --positions + --cap-strict and then exits — so
    # the auto-split parent never reaches here. The per-part children run with a
    # STRICT trim so each part ends at ≤180s exactly and the next part picks up at
    # the dropped straddling position — gap-free, no overlap.)
    # --png keeps the WHOLE song (the 4 MB PNG holds it all → no split, no trim).
    if not getattr(args, 'use_png', False):
        _trim_song_to_audio_cap(mod, strict=bool(getattr(args, 'cap_strict', False)))

    # ── --png as a DATA VEHICLE (not a separate codec) ──────────────────────────
    # The auto-split driver + trim above have already honored --png (no split, no
    # trim). From here on, build EXACTLY as a normal VQ build (whatever --vec-dim/
    # --bitrate/--downsample/--no-rvq2 were passed) — so all the encoders run and
    # the const-array data is produced. At the very end (before the JSON), if
    # --png was requested, `_pack_build_into_png` repackages every data array into
    # the PNG and points the fetchers at it. So flip args.use_png OFF now and
    # remember the request in _png_mode.
    _png_mode = bool(getattr(args, 'use_png', False))
    if _png_mode:
        args.use_png = False
        print("   🖼️  --png: building VQ data normally, then repackaging into the PNG "
              "(PNG = data vehicle; decoder unchanged)")

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
    
    # --output-base (used by the auto-split driver) gives each part child an
    # isolated base name ({base}_partK) so its files don't clobber the default
    # build; the driver then renames them to {base}_shadertoy_partK_*. Falls back
    # to the module's own filename for normal builds.
    base_name = getattr(args, 'output_base', None) or os.path.splitext(os.path.basename(args.modfile))[0]

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
    # Save post-trim disk pattern indices for the VQ encoder (plain MOD --positions fix).
    # At this point mod.song_positions is [0,1,...N-1] (dense), and used_pat_indices[k]
    # is the original disk pattern index for dense index k.  So the inverse mapping gives
    # the original disk indices in playback order — exactly what the VQ encoder needs to
    # build the right pattern_order list.  Doing this POST-trim means auto-split children
    # (--positions 0-63 --cap-strict, trimmed to ~23 positions) get the correct 23-entry
    # list instead of the stale pre-trim 64-entry list.
    mod._positions_disk_indices = [used_pat_indices[dense] for dense in mod.song_positions]

    pattern_bytes = []
    # mod.patterns is now in dense order (matches used_pat_indices) — iterate
    # by dense index, not original index.
    for dense_idx in range(len(used_pat_indices)):
        pattern = mod.patterns[dense_idx]
        for row in pattern:
            for ch_idx in range(num_channels):
                ch = row[ch_idx] if ch_idx < len(row) else {}
                vol_col = 0   # IT note-volume override (byte 4); 0 for MOD/S3M
                if mod.is_s3m:
                    sample = ch.get('instrument', 0)
                    period = s3m_note_to_mod_period(ch.get('note', 255))
                    effect = ch.get('command', 0)
                    param  = ch.get('info', 0)
                else:
                    sample  = ch.get('sample', 0)
                    period  = ch.get('period', 0)
                    effect  = ch.get('effect', 0)
                    param   = ch.get('param', 0)
                    vol_col = ch.get('vol_col', 0)   # IT note-volume override
                sample_hi = (sample & 0xF0)
                sample_lo = (sample & 0x0F) << 4
                period_hi = (period >> 8) & 0x0F
                period_lo = period & 0xFF
                pattern_bytes.append(sample_hi | period_hi)
                pattern_bytes.append(period_lo)
                pattern_bytes.append(sample_lo | (effect & 0x0F))
                pattern_bytes.append(param)
                pattern_bytes.append(vol_col & 0xFF)  # 5th byte: IT vol-col override

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
    # Only written when --png is requested (default OFF). The normal build is
    # embedded (const arrays in the GLSL), so the data PNG is dead weight unless
    # you're using the PNG-texture data path (--png / --png).
    png_file = base_name + "_player_data.png"
    png_size = 0
    if args.use_png and len(all_bytes) > 0:
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

    # Instruction .txt generation removed per user request ("no need for
    # instructions"). The f.write(...) block below is kept under `if False:`
    # (never runs, no .txt written) so this stays a minimal, low-risk diff.
    instructions_file = None
    if False:
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
    
    # Generate HTML player — RESTORED from mod_player_archived.py.
    # The user confirmed the archived player "played the tracks correctly" and
    # it has the features the stripped-down current one lost: horizontal
    # click-to-scroll tracks, a "tracks N of M" indicator, and the stronger
    # dying-voice crossfade declick. Route to the archived implementation; it's
    # compatible because both consume the same mod object (samples / patterns /
    # song_positions / num_channels) and the archived fn resolves its helpers
    # inside its own module. (Local current create_fixed_player_html kept for
    # reference but no longer the default path.)
    html_file = base_name + "_player.html"
    if getattr(args, 'no_html', False):
        # Auto-split GLSL child — the parent emits the single full-song HTML.
        print("   ⤷ skipping HTML (auto-split child; parent emits one full-song player)")
    else:
        _emit_html_player(mod, html_file, args.downsample, args.vec_dim)

    # ShaderToy Common tab: VQ-encoded via embedded vq_encoder_v2 (default),
    # or legacy PNG-loaded Common via create_shadertoy_glsl when --png.
    glsl_common_file = base_name + "_shadertoy_common.glsl"
    if args.use_png:
        print(f"\n\U0001f5bc\ufe0f  --png: skipping VQ encoder, generating PNG-loaded Common")
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
            # Capture carved-over speed/tempo so --positions builds start from the
            # correct values even when the VQ encoder's internal MODFile starts fresh.
            _pos_init_spd = int(getattr(mod, 'initial_speed', 6) or 6)
            _pos_init_tmp = int(getattr(mod, 'initial_tempo', 125) or 125)

            def _bpm_aware_compute_row_speed_table(_mod):
                speed = _pos_init_spd  # honours --positions carryover (was hardcoded 6)
                bpm   = _pos_init_tmp  # honours --positions carryover (was hardcoded 125)
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

            # ── --positions slice for plain MOD files ─────────────────────────
            # For S3M/IT/XM the adapters below replace _vqmod.MODFile entirely
            # and build pattern_order from the already-sliced mod.song_positions.
            # For plain MOD the encoder re-reads the file from disk — meaning
            # mod.song_positions slice (applied above) is silently ignored and
            # the GLSL receives all 64 positions instead of the requested slice.
            # Fix: after _parse_with_xxCN runs on the disk data, overwrite
            # song_length + pattern_order with the pre-sliced values.
            if (getattr(args, 'positions', None)
                    and not getattr(mod, 'is_s3m', False)
                    and not getattr(mod, 'is_it',  False)
                    and not getattr(mod, 'is_xm',  False)):
                # Use the POST-trim disk pattern indices saved by the trim step.
                # mod.song_positions is [0,1,...] (dense) by the time we get here;
                # _positions_disk_indices holds the original disk indices in the
                # same playback order (e.g. [9,10] for ENIGMA --positions 14-15,
                # or the correctly trimmed ~23 entries for an auto-split child).
                _sliced_po = list(getattr(mod, '_positions_disk_indices',
                                          mod.song_positions))
                _orig_parse_po = _vqmod.MODFile.parse
                def _parse_with_positions_slice(self):
                    _orig_parse_po(self)
                    self.song_length   = len(_sliced_po)
                    self.pattern_order = (_sliced_po + [0] * (128 - len(_sliced_po)))[:128]
                _vqmod.MODFile.parse = _parse_with_positions_slice
                print(f"   ✂️  --positions MOD: VQ encoder sliced to {_sliced_po} "
                      f"(song_length={len(_sliced_po)})")

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
            # Bound for ALL formats so the VQ-Common post-proc's
            # `if _vcol_side_capture[0] and fmt == 'S3M':` doesn't
            # UnboundLocalError on IT/XM/MOD (the S3M block repopulates it;
            # non-S3M paths legitimately have no dropped volcols).
            _vcol_side_capture = [{}]
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

                _vcol_side_capture = [{}]  # {(raw_pat, row, ch): vol} for dropped vol columns
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
                        # remap field names. Feed ALL S3M samples, not a
                        # hardcoded 31 — 2ND_PM has 54 instruments and the
                        # 31-cap silenced its voice/vocal samples (instr
                        # 32-54 were never encoded). See feedback_s3m_
                        # instrument_cap; this is that fix ported onto v1.61.
                        self.samples_info = []
                        self.sample_bytes = []
                        _n_smp = max(31, len(m.samples))
                        for i in range(_n_smp):
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
                        # S3M param memory: Dxx/Exx/Fxx with param==0 means
                        # "continue previous slide" (MikIT UniS3MEffectD etc.).
                        # MOD A00/200/100 are no-ops, so we pre-resolve here.
                        # Maintained across patterns (not reset per-pattern) so
                        # cross-boundary continuations work for linear orders.
                        _last_dxx = [0] * self.num_channels
                        _last_exx = [0] * self.num_channels
                        _last_fxx = [0] * self.num_channels
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
                                                # c2spd is applied as an EXACT float
                                                # rate-scale in _gcoBody (SampleInfo.c2sp),
                                                # NOT by integer-rounding the period here.
                                                # Rounding period*8363/c2 to a 12-bit int
                                                # lost ~0.3-0.5% pitch for non-8363 samples
                                                # — inaudible on short notes, but it
                                                # accumulated over loop iterations on a
                                                # sustained looped sample (2ND_PM inst33
                                                # pos-20 "out of phase lead"). Keep the
                                                # raw note period; only clamp to the 12-bit
                                                # pack range (octave-0 ultra-low notes,
                                                # already capped under the old code — so
                                                # this is unchanged for them).
                                                if _c2 != 8363 and period > 4095:
                                                    period = 4095   # 12-bit field cap
                                        # S3M param memory: cmd 4/5/6 (D/E/F)
                                        # with param==0 means "continue previous
                                        # slide" — substitute remembered param.
                                        if cmd == 4:
                                            if inf != 0: _last_dxx[ch_i] = inf
                                            else: inf = _last_dxx[ch_i]
                                        elif cmd == 5:
                                            if inf != 0: _last_exx[ch_i] = inf
                                            else: inf = _last_exx[ch_i]
                                        elif cmd == 6:
                                            if inf != 0: _last_fxx[ch_i] = inf
                                            else: inf = _last_fxx[ch_i]
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
                                        # Record volcol dropped due to coexisting effect.
                                        # mod_eff==0xC means volcol was promoted; anything
                                        # else means an effect (e.g. Gxx porta) won and
                                        # the per-row volume accent was lost.  The post-
                                        # process injects these as _volSide[] in Common
                                        # and patches _gcoBody to apply them in the
                                        # volume forward-scan (fixes inst-25 staccato).
                                        _raw_cv2 = cell.get('volume', 255)
                                        if 0 <= _raw_cv2 <= 64 and mod_eff != 0xC:
                                            _vcol_side_capture[0][(pat_idx, row_i, ch_i)] = _raw_cv2
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

            elif getattr(mod, 'is_it', False) or getattr(mod, 'is_xm', False):
                # ── IT / XM → VQ encoder bridge (ported from mod_player.py) ──
                # ITFile/XMFile already emit MOD-numbered cells (sample,
                # period, effect, param) — IT letter-effects were translated
                # IT→MOD at PARSE time (_xm_or_it_effect_to_mod, is_xm=False).
                # So this is a STRAIGHT PASSTHROUGH: no S3M-letter remap, no
                # S3M note table, no c2spd compensation. Without this bridge
                # the encoder re-reads the .it/.xm as raw MOD bytes, fails,
                # and falls back to the legacy non-embedded (PNG) path —
                # exactly the USE_EMBEDDED_DATA=0 breakage seen on JEFF93.IT.
                _outer_mod = mod

                class _XMITtoVQAdapter:
                    """Duck-typed mod the VQ encoder reads, built from the
                    already-parsed IT/XM shim. Same attribute set as
                    _S3MtoVQAdapter; cells packed straight to MOD 4-byte."""
                    def __init__(self, _path_unused):
                        m = _outer_mod
                        self.title = m.title
                        self.song_length  = len(m.song_positions)
                        po = list(m.song_positions[:128])
                        while len(po) < 128:
                            po.append(0)
                        self.pattern_order = po
                        self.num_channels  = m.num_channels
                        self.num_patterns  = m.num_patterns
                        self.channel_settings = list(getattr(m, 'channel_settings', []) or [])
                        self.magic = b'M.K.'   # encoder uses num_channels above
                        # samples_info/sample_bytes: ITFile/XMFile dicts use
                        # the same MODFile shape, so this is a passthrough
                        # (31→max(31,N): IT spec allows up to 99 samples).
                        self.samples_info = []
                        self.sample_bytes = []
                        _n_smp = max(31, len(m.samples))
                        for i in range(_n_smp):
                            si = m.samples[i] if (i < len(m.samples) and isinstance(m.samples[i], dict)) else None
                            if si is not None:
                                arr = si.get('data', None)
                                raw_length = si.get('length', 0)
                                # NOTE: the old _XMITToVQAdapter (ported from
                                # mod_player.py) here PREPENDED _pad zeros
                                # ("block-align onset") and ran a pre-VQ
                                # Butterworth LPF. Both are IT-ONLY (the
                                # working _S3MtoVQAdapter does NEITHER) and
                                # the playback never compensated: the pad
                                # shifts the sample data while loop_start/
                                # loop_len stay in ORIGINAL coords → every
                                # sustained/looped IT note wraps at the wrong
                                # phase forever → the corr-0.26→0.01 collapse
                                # vs mikIT. Feed the sample EXACTLY like the
                                # S3M adapter (raw int8 bytes, no shift, no
                                # filter) so the VQ stream matches the
                                # playhead/loop math. (root-cause fix #1)
                                raw = (arr.tobytes() if (arr is not None and getattr(arr, 'size', 0) > 0) else b'')
                                self.samples_info.append(dict(
                                    name       = si.get('name', ''),
                                    length     = raw_length,
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
                        # Patterns: 5-byte cells (4 MOD bytes + vol_col byte 4).
                        # self.data keeps 4-byte layout for actual_pattern_rows().
                        self.is_it = True
                        self.patterns = []
                        _data_pats = []  # 4-byte layout for self.data
                        for pat_idx in range(self.num_patterns):
                            pat = m.patterns[pat_idx] if pat_idx < len(m.patterns) else None
                            buf  = bytearray(64 * self.num_channels * 5)  # 5-byte cells
                            buf4 = bytearray(64 * self.num_channels * 4)  # 4-byte for data
                            if pat is not None:
                                for row_i in range(min(64, len(pat))):
                                    row = pat[row_i]
                                    for ch_i in range(self.num_channels):
                                        cell = row[ch_i] if ch_i < len(row) else None
                                        if cell is None:
                                            continue
                                        sample  = cell.get('sample', 0)
                                        period  = max(0, min(4095, cell.get('period', 0)))
                                        effect  = cell.get('effect', 0)
                                        param   = cell.get('param', 0)
                                        vol_col = cell.get('vol_col', 0)
                                        o5 = (row_i * self.num_channels + ch_i) * 5
                                        buf[o5]   = (sample & 0xF0) | ((period >> 8) & 0x0F)
                                        buf[o5+1] = period & 0xFF
                                        buf[o5+2] = ((sample & 0x0F) << 4) | (effect & 0x0F)
                                        buf[o5+3] = param & 0xFF
                                        buf[o5+4] = vol_col & 0xFF
                                        o4 = (row_i * self.num_channels + ch_i) * 4
                                        buf4[o4]   = buf[o5]
                                        buf4[o4+1] = buf[o5+1]
                                        buf4[o4+2] = buf[o5+2]
                                        buf4[o4+3] = buf[o5+3]
                            self.patterns.append(bytes(buf))
                            _data_pats.append(bytes(buf4))
                        # actual_pattern_rows() uses self.data (4-byte layout) to
                        # detect Dxx/Bxx pattern breaks — keep separate from patterns.
                        self.data = b'\x00' * 1084 + b''.join(_data_pats)

                _vqmod.MODFile = _XMITtoVQAdapter

            try:
                _vq_kw = dict(K=256, weighted=True,
                              downsample=args.downsample, bitrate=args.bitrate,
                              vec_dim=args.vec_dim, resampler=args.resampler,
                              no_rvq2=args.no_rvq2)
                try:
                    _vqmod.main(args.modfile, glsl_common_file, **_vq_kw)
                except ValueError as _vqe:
                    # sklearn KMeans needs n_samples >= n_clusters. A module
                    # with few samples (e.g. jeff.it = 99) vs the bitrate's
                    # RVQ K (hi → K=512, or the K=256 codebook) raises
                    # "n_samples=N should be >= n_clusters=K". Retry with
                    # the ×2 RVQ stage off and K clamped to the sample count
                    # so the VQ path STILL succeeds — the old behaviour fell
                    # through to a raw-embed path whose sample_bytes_data was
                    # empty → max(smp_chunk_sizes) hard-crash (no render).
                    _es = str(_vqe)
                    if ('n_clusters' in _es or 'n_samples' in _es):
                        _nsv = len(getattr(mod, 'samples', []) or []) or 1
                        _kc  = max(2, min(256, _nsv))
                        print(f"   ⚠️  VQ K>n_samples ({_es.strip()}); "
                              f"retry --no-rvq2 K={_kc} (≤{_nsv} samples)")
                        try:
                            _vqmod.main(args.modfile, glsl_common_file,
                                        **{**_vq_kw, 'no_rvq2': True,
                                           'K': _kc})
                        except ValueError as _vqe2:
                            print(f"   ⚠️  still failing ({str(_vqe2).strip()})"
                                  f"; retry bitrate=lo K={_kc}")
                            _vqmod.main(args.modfile, glsl_common_file,
                                        **{**_vq_kw, 'no_rvq2': True,
                                           'bitrate': 'lo', 'K': _kc})
                    else:
                        raise
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
                6: "Disco Inferno + UFOff Dancer (orblivius/finalman/Lallis — dance-floor scene with poi)",
                7: "Sparkly 4D (Philip Bertani — 4D IFS volumetric raymarcher)",
                8: "Skywalker (orblivius — synchronized flying-curve terrain + star field + cloud)",
                9: "Music is in the DNA (jaszunio15/enbe fork — DNA helix + parallax dunes)",
            }
            _vname = _viz_names.get(args.viz, f"viz{args.viz}")
            try:
                with open(glsl_common_file) as _cf: _ct = _cf.read()
                # Rename the SampleInfo.length field → .smpLen to avoid the
                # GLSL built-in name conflict with the array .length() method.
                # The VQ encoder bakes the struct with 'length'; do it once here
                # before any struct-rebuild patches so they all see 'smpLen'.
                _ct = _ct.replace('int start, length,', 'int start, smpLen,')
                _ct = _ct.replace('smp.length', 'smp.smpLen')   # field accesses in fn bodies
                # Force highp for all float/int in Common (prepended to every
                # tab → covers Sound/Image/BufferA too). ShaderToy injects its
                # own #version+precision before the Common; redeclaring the
                # default precision here is valid GLSL ES and pins highp for
                # the sample-position / period / timing math (some GPUs
                # default mediump → drift/roughness). User-requested.
                # precision highp: NOT prepended here; ShaderToy rejects it.
                # glsl_state_dump.py / sound_exec.py inject it themselves.
                _ct = _ct.replace(
                    "GLSL (The Last) MOD Player v1.42 (c) 2026 Orblivius",
                    "GLSL (The Last) MOD Player v1.61 (c) 2026 Orblivius", 1)
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
                        f"#define INTRO_SILENCE_S  {float(getattr(args,'intro_silence',0.0) or 0.0):.3f}\n"
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

                # ── Fix: TICKS_PER_SEC must be the song's TRUE tick rate ──
                # getPosition computes totalTickF = time * TICKS_PER_SEC and
                # the rowStartTick table is PURE integer ticks (fetchTick(r)
                # == cumulative ticks; the "BPM-scaled" comment is stale —
                # measured fetchTick(r)=SPEED*r). 50.0 is only correct at
                # BPM 125. MikIT's samples-per-tick is the integer
                # (125*SR)//(50*BPM); true ticks/sec = SR / that. At BPM 130
                # that's 52.0047, not 50 — using 50 lagged getPosition ~1
                # row every ~25 rows and desynced the lead (measured vs C++
                # MikIT: 573/600 rows wrong @50.0 → 4/600 with this fix).
                # Valid for BPM-constant songs (Txx tempo changes would need
                # per-segment rates; 2ND_PM has 58 Axx speed / 0 Txx → BPM
                # fixed, one rate exact). Effect _dt / _SAMPLES_PER_TICK also
                # key off TICKS_PER_SEC so this aligns DSP timing too.
                _SR_tps = 44100
                _bpm_i  = max(1, int(round(_bpm_actual)))
                _spt    = (125 * _SR_tps) // (50 * _bpm_i)
                _tps    = (_SR_tps / float(_spt)) if _spt else 50.0
                _ct = _ct.replace(
                    "#define TICKS_PER_SEC     50.0   // rowStartTick is BPM-scaled (see compute_row_speed_table patch)",
                    f"#define TICKS_PER_SEC     {_tps:.6f}   // SR/((125*SR)//(50*BPM)) MikIT-exact, BPM {_bpm_i}",
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
                # The VQ encoder emits only enable3D/enableFAT. The imported
                # mod_player.py DSP also needs enablePhatBass / enableVelvetReverb
                # / MASTER_GAIN — inject them right after the encoder's enableFAT
                # line so _flagdef can #define-gate them (else `#if
                # ENABLE_PHATBASS` references an undefined macro → ES error).
                if 'enablePhatBass' not in _ct:
                    # PhatBass is controlled by --phatbass (independent of --fat4x);
                    # --no-dsp force-disables it. (Was wrongly tied to _no_fat, so
                    # --phatbass did nothing.)
                    _pb_val = 'false' if bool(getattr(args, '_compat_no_phatbass', True)) else 'true'
                    _ct = _ct.replace(
                        'const bool  enableFAT     = true;',
                        'const bool  enableFAT     = true;\n'
                        f'const bool  enablePhatBass     = {_pb_val};\n'
                        'const bool  enableVelvetReverb = false;\n'
                        'const bool  enableCombReverb   = false;\n'
                        'const float MASTER_GAIN = 1.0;', 1
                    )
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

                # ── Fix: fetchTick must span ALL rowStartTick chunks ──────────
                # The VQ encoder emits a fetchTick() that only ever indexes
                # rowStartTick0. bytes_to_int32_be_array splits the 16-bit tick
                # table into 512-ivec4 chunks (= 4096 rows each), so any song
                # with >4096 rows (e.g. 2ND_PM = 5220) overflows into
                # rowStartTick1+, which the emitted fetchTick never reads —
                # past row 4096 it goes out of bounds / non-monotonic and
                # getPosition's binary search collapses (last ~22% of the song
                # desyncs/trainwrecks). Replace it with the same chunk-selector
                # idiom the encoder already uses for fetchCodesByte/fetchIdxByte
                # (ivec4Idx/512 chunk, %512 local), spanning every emitted
                # rowStartTick{k}. Correct for the single-chunk case too.
                import re as _re_ft
                _tk_chunks = sorted(set(int(x) for x in _re_ft.findall(
                    r'const ivec4 rowStartTick(\d+)\[', _ct)))
                _n_tk = (max(_tk_chunks) + 1) if _tk_chunks else 1
                _sel1 = "\n".join(
                    f'    {"if" if k == 0 else "else if"} (chunkIdx == {k}) v = rowStartTick{k}[localIvec4];'
                    for k in range(_n_tk))
                _sel2 = "\n".join(
                    f'    {"if" if k == 0 else "else if"} (chunkIdx2 == {k}) v2 = rowStartTick{k}[localIvec4_2];'
                    for k in range(_n_tk))
                _fixed_ft = (
                    "int fetchTick(int rowIdx) {\n"
                    "    int byteIdx = rowIdx * 2;\n"
                    "    int ivec4Idx = byteIdx >> 4;\n"
                    "    int byteInIvec4 = byteIdx & 15;\n"
                    "    int chunkIdx = ivec4Idx / 512;\n"
                    "    int localIvec4 = ivec4Idx % 512;\n"
                    "    ivec4 v = ivec4(0);\n"
                    f"{_sel1}\n"
                    "    int lo = _extractByte(v, byteInIvec4);\n"
                    "    int byteIdx2 = byteIdx + 1;\n"
                    "    int ivec4Idx2 = byteIdx2 >> 4;\n"
                    "    int byteInIvec4_2 = byteIdx2 & 15;\n"
                    "    int chunkIdx2 = ivec4Idx2 / 512;\n"
                    "    int localIvec4_2 = ivec4Idx2 % 512;\n"
                    "    ivec4 v2 = ivec4(0);\n"
                    f"{_sel2}\n"
                    "    int hi = _extractByte(v2, byteInIvec4_2);\n"
                    "    return lo | (hi << 8);\n"
                    "}"
                )
                _ct2, _nft = _re_ft.subn(
                    r'int fetchTick\(int rowIdx\)\s*\{.*?\n\}',
                    lambda _m: _fixed_ft, _ct, count=1, flags=_re_ft.DOTALL)
                if _nft == 1:
                    _ct = _ct2
                    print(f"   ✓ fetchTick patched to span {_n_tk} rowStartTick chunk(s) "
                          f"(encoder emitted rowStartTick0-only — >4096-row overflow bug)")
                else:
                    print("   WARNING: fetchTick block not found — multi-chunk timing patch skipped")

                # ── Fix: getPosition binary search must converge ──────────
                # Encoder hardcodes `for (int _bs = 0; _bs < 12; ...)` over
                # [0, NUM_SONG_ROWS]. 12 iters only covers ≤4096 rows
                # (log2). 2ND_PM has 5220 rows → search doesn't converge,
                # returns a row up to several off → getPosition lags MikIT
                # (measured: 12 iters=71% row-match vs MikIT, ≥16=91%).
                # Use enough iters for any song: ceil(log2(NUM_SONG_ROWS))+4.
                import math as _math_bs
                _m_nsr = _re_ft.search(r'#define\s+NUM_SONG_ROWS\s+(\d+)', _ct)
                _n_rows_bs = int(_m_nsr.group(1)) if _m_nsr else 5220
                _bs_iters = max(16, int(_math_bs.ceil(
                    _math_bs.log2(max(2, _n_rows_bs)))) + 4)
                _ct_bs, _nbs = _re_ft.subn(
                    r'for \(int _bs = 0; _bs < \d+; _bs\+\+\)',
                    f'for (int _bs = 0; _bs < {_bs_iters}; _bs++)', _ct, count=1)
                if _nbs == 1:
                    _ct = _ct_bs
                    print(f"   ✓ getPosition binary search → {_bs_iters} iters "
                          f"(was 12; {_n_rows_bs} rows need convergence)")
                else:
                    print("   WARNING: getPosition _bs loop not found — iter fix skipped")

                # ── Fix: SampleInfo samples[] array size + commas ─────────
                # The b64 vq-encoder hardcodes `samples[31]` and only
                # comma-separates the first 31 entries (`',' if i<30`).
                # With the _S3MtoVQAdapter 31-cap lifted, it now emits all
                # N instruments (2ND_PM=54) but with the wrong [31] size
                # and NO commas past entry 30 → malformed GLSL + silenced
                # voice samples. Rebuild the whole array: re-extract every
                # SampleInfo(...) tuple and re-emit with correct size and
                # proper comma separation. (feedback_s3m_instrument_cap)
                _m_si = _re_ft.search(
                    r'const SampleInfo samples\[\d+\]\s*=\s*SampleInfo\[\]\(\s*(.*?)\s*\);',
                    _ct, _re_ft.DOTALL)
                if _m_si:
                    _si_entries = _re_ft.findall(r'SampleInfo\([^)]*\)', _m_si.group(1))
                    _si_n = len(_si_entries)
                    # ── Clamp loop/envelope bounds to the (VQ-decimated) sample
                    # length. The encoder converts length/loopStart/loopLen by
                    # floor-÷sds; rounding (or a malformed source loop) can leave
                    # loopStart > length or loopStart+loopLen > length, so the
                    # GLSL loop-wrap `mod(pos-loopStart, loopLen)` reads PAST the
                    # sample's data into the next sample's bytes (audible as the
                    # wrong sample bleeding into a sustained looped note). These
                    # fields are already in compressed/decimated units, so the
                    # clamp IS the VQ adjustment. SampleInfo = (start, length,
                    # loopStart, loopLen, volume, bwFactor[, finetune]).
                    _si_fixed = []
                    _n_clamped = 0
                    _n_c2 = 0
                    _n_filt = 0
                    _env_pts_all = []   # flat packed (tick<<7|val0..64)
                    _n_env = 0
                    for _si_idx, _e in enumerate(_si_entries):
                        _nums = _re_ft.findall(r'-?\d+', _e)
                        if len(_nums) >= 4:
                            _v = [int(x) for x in _nums]
                            _len = max(0, _v[1])
                            _ls  = min(max(0, _v[2]), _len)            # 0 ≤ loopStart ≤ length
                            _ll  = max(0, _v[3])
                            if _ls + _ll > _len:                       # loop end ≤ length
                                _ll = _len - _ls
                            if (_ls, _ll) != (_v[2], _v[3]):
                                _n_clamped += 1
                            _v[1], _v[2], _v[3] = _len, _ls, _ll
                            # 8th field c2sp: the sample's S3M c2spd (samples/sec
                            # at C-5). Standard = 8363 → rate ×1.0 (byte-identical
                            # for MOD / standard-tuned samples, audits unaffected).
                            # Non-8363 → exact float rate-scale in _gcoBody,
                            # replacing the lossy integer period compensation that
                            # dropped ~0.5% pitch (inst33 pos-20 phase-drift fix).
                            while len(_v) < 7:
                                _v.append(0)
                            _v = _v[:7]
                            # IT sample dicts (ITFile) key this 'c5_speed'
                            # (samples/sec at C-5); S3M uses 'c2spd'. Both
                            # play the SAME role in the _gcoBody rate-scale
                            # (×c2sp/8363). Reading only 'c2spd' left every
                            # IT sample at the 8363 default → Amiga pitch →
                            # jeff.it ~6× too slow ("wrong scale/octave",
                            # corr 0.01). Fall back c2spd→c5_speed.
                            try:
                                _smp_d = (mod.samples[_si_idx]
                                          if (_si_idx < len(mod.samples)
                                              and isinstance(mod.samples[_si_idx], dict))
                                          else None)
                                _c2sp = int(((_smp_d.get('c2spd')
                                              or _smp_d.get('c5_speed')
                                              or 8363) if _smp_d else 8363)
                                            or 8363)
                            except Exception:
                                _c2sp = 8363
                            if _c2sp <= 0:
                                _c2sp = 8363
                            if _c2sp != 8363:
                                _n_c2 += 1
                            _v.append(_c2sp)
                            # 9th/10th fields itCut/itRes: IT instrument
                            # resonant-filter cutoff (0..127) & resonance
                            # (0..127). 127/0 = unfiltered (it2play
                            # filtOn=false) → S3M/MOD and no-filter IT
                            # samples are inert (the GLSL skips the 2-pole
                            # entirely for them, zero cost). Read from the
                            # already-parsed mod.samples like c2sp above.
                            try:
                                _fsmp = (mod.samples[_si_idx]
                                         if (_si_idx < len(mod.samples)
                                             and isinstance(mod.samples[_si_idx], dict))
                                         else {})
                                _itc = int(_fsmp.get('it_cutoff', 127))
                                _itr = int(_fsmp.get('it_res', 0))
                            except Exception:
                                _itc, _itr = 127, 0
                            _itc = max(0, min(127, _itc))
                            _itr = max(0, min(127, _itr))
                            if _itc < 127 or _itr > 0:
                                _n_filt += 1
                            _v.append(_itc)
                            _v.append(_itr)
                            # 11..14 (MOD+ vol envelope): eOff = start idx in
                            # _itEPt[], eN = #points (0 → inert; S3M/MOD/
                            # no-env IT unchanged), eSus = sustain pt idx
                            # (-1=none; held there while keyed-on), eLp =
                            # loopStart<<8|loopEnd pt idx (-1=none). Points
                            # = ITFile env_pts [tick,val0..64] packed
                            # tick<<7|val. This is the IT-over-MOD feature
                            # the HTML player has and the timeline lost.
                            try:
                                _esmp = (mod.samples[_si_idx]
                                         if (_si_idx < len(mod.samples)
                                             and isinstance(mod.samples[_si_idx], dict))
                                         else {})
                                _epts = _esmp.get('env_pts') or []
                                _eoff = len(_env_pts_all)
                                _en   = min(len(_epts), 25)
                                for _pp in _epts[:25]:
                                    _pt = max(0, min(0x1FFFF, int(_pp[0])))
                                    _pv = max(0, min(64, int(_pp[1])))
                                    _env_pts_all.append((_pt << 7) | _pv)
                                _esus = (int(_esmp.get('env_sus_pt', 0))
                                         if _esmp.get('env_sus') else -1)
                                if _esmp.get('env_loop'):
                                    _elp = ((int(_esmp.get('env_loop_st', 0)) & 0xFF) << 8) \
                                           | (int(_esmp.get('env_loop_en', 0)) & 0xFF)
                                else:
                                    _elp = -1
                                if _en > 0:
                                    _n_env += 1
                            except Exception:
                                _eoff, _en, _esus, _elp = 0, 0, -1, -1
                            _v.append(_eoff); _v.append(_en)
                            _v.append(_esus); _v.append(_elp)
                            # 15: nna — IT-ONLY. S3M/MOD/XM gets no extra
                            # field (14 ints exactly = pre-NNA baseline)
                            # because every function-local `SampleInfo smp`
                            # declaration in _gcoBody (~860 lines of locals)
                            # would grow by 4 bytes, and ANGLE's private-
                            # variable budget is what crashed 2ND_PM.s3m
                            # ("CONTEXT_LOST_WEBGL"). glslang doesn't catch
                            # this — only the user's GPU enforces it.
                            if getattr(mod, 'is_it', False):
                                try:
                                    _nna = int((_fsmp or {}).get('nna', 0))
                                except Exception:
                                    _nna = 0
                                _v.append(max(0, min(3, _nna)))
                            _si_fixed.append("SampleInfo(" + ", ".join(str(x) for x in _v) + ")")
                        else:
                            _si_fixed.append(_e)
                    _si_entries = _si_fixed
                    # MOD+ vol-envelope point pool, emitted right before
                    # samples[] (Common is prepended → _gcoBody sees it).
                    _ep = _env_pts_all if _env_pts_all else [0]
                    _itept_decl = (f"const int _itEPt[{len(_ep)}] = int[]("
                                   + ",".join(str(x) for x in _ep) + ");\n")
                    _si_new = (_itept_decl
                               + f"const SampleInfo samples[{_si_n}] = SampleInfo[](\n    "
                               + ",\n    ".join(_si_entries) + "\n);")
                    _ct = _ct[:_m_si.start()] + _si_new + _ct[_m_si.end():]
                    print(f"   ✓ SampleInfo samples[] rebuilt: {_si_n} entries "
                          f"(was declared [31]; commas repaired; "
                          f"{_n_clamped} loop-bound clamps)")
                    # ── Add the c2sp 8th struct field to match the rebuilt
                    # 8-value SampleInfo(...) constructors. c2sp carries the
                    # sample's exact c2spd; _gcoBody scales the playback rate
                    # by float(c2sp)/8363.0 (×1.0 for standard 8363 samples →
                    # byte-identical). Replaces the lossy integer period
                    # compensation that caused the inst33 pos-20 phase drift.
                    if 'c2sp' not in _ct.split('struct SampleInfo')[1].split('}')[0]:
                        _ct, _n_struct = _re_ft.subn(
                            r'(struct\s+SampleInfo\s*\{\s*int\s+[^;}]*?bwFactor,\s*finetune)\s*;',
                            r'\1, c2sp;', _ct, count=1)
                        if _n_struct == 1:
                            print(f"   ✓ SampleInfo struct → +c2sp field "
                                  f"({_n_c2} non-8363 samples get a rate-scale; "
                                  f"rest ×1.0)")
                        else:
                            print(f"   ✗ WARNING: SampleInfo struct def not "
                                  f"patched for c2sp — GLSL will not compile")
                    # ── Add itCut/itRes (9th/10th) struct fields to match the
                    # rebuilt 10-value SampleInfo(...) constructors. Carries
                    # the IT instrument resonant-filter cutoff/resonance so
                    # the GLSL can apply the it2play 2-pole LPF. 127/0 =
                    # unfiltered → the shader's filter is a no-op for those
                    # (S3M/MOD/no-filter-IT pay zero cost). Must run AFTER
                    # the c2sp patch so the struct already ends "…,c2sp".
                    if ('itCut' not in _ct.split('struct SampleInfo')[1].split('}')[0]):
                        _ct, _n_fstruct = _re_ft.subn(
                            r'(struct\s+SampleInfo\s*\{\s*int\s+[^;}]*?finetune,\s*c2sp)\s*;',
                            r'\1, itCut, itRes;', _ct, count=1)
                        if _n_fstruct == 1:
                            print(f"   ✓ SampleInfo struct → +itCut/itRes "
                                  f"({_n_filt} IT samples carry a resonant "
                                  f"filter; rest 127/0 = inert)")
                        else:
                            print(f"   ✗ WARNING: SampleInfo struct def not "
                                  f"patched for itCut/itRes — GLSL will not "
                                  f"compile")
                    # ── Add eOff/eN/eSus/eLp (11..14) for the MOD+ per-sample
                    # VOL ENVELOPE. eN==0 → _itVolEnv() returns 1.0 → S3M/
                    # MOD/no-env IT byte-identical (zero cost). Runs AFTER
                    # the itCut/itRes patch (struct ends "…,itCut,itRes").
                    if ('eOff' not in _ct.split('struct SampleInfo')[1].split('}')[0]):
                        _ct, _n_estruct = _re_ft.subn(
                            r'(struct\s+SampleInfo\s*\{\s*int\s+[^;}]*?itCut,\s*itRes)\s*;',
                            r'\1, eOff, eN, eSus, eLp;', _ct, count=1)
                        if _n_estruct == 1:
                            print(f"   ✓ SampleInfo struct → +eOff/eN/eSus/eLp "
                                  f"({_n_env} samples carry a vol envelope; "
                                  f"{len(_env_pts_all)} env points)")
                        else:
                            print(f"   ✗ WARNING: SampleInfo struct not "
                                  f"patched for envelope — GLSL won't compile")
                    # ── Add nna (15th) struct field — IT-ONLY. S3M/MOD/XM
                    # builds keep the 14-field struct so every function-
                    # local SampleInfo stays at 14 ints = no private-var
                    # budget growth (2ND_PM.s3m's "CONTEXT_LOST_WEBGL" was
                    # the +4 bytes × N locals pushing past ANGLE's GPU
                    # limit, even though glslang accepted it).
                    if (getattr(mod, 'is_it', False)
                            and 'nna' not in _ct.split('struct SampleInfo')[1].split('}')[0]):
                        _ct, _n_nstruct = _re_ft.subn(
                            r'(struct\s+SampleInfo\s*\{\s*int\s+[^;}]*?eOff,\s*eN,\s*eSus,\s*eLp)\s*;',
                            r'\1, nna;', _ct, count=1)
                        _n_nna = sum(1 for s in (getattr(mod,'samples',[]) or [])
                                     if isinstance(s, dict) and s.get('nna', 0))
                        if _n_nstruct == 1:
                            print(f"   ✓ SampleInfo struct → +nna "
                                  f"({_n_nna} samples carry NNA != cut)")
                        else:
                            print(f"   ✗ WARNING: SampleInfo struct not "
                                  f"patched for nna — GLSL won't compile")
                else:
                    print("   WARNING: SampleInfo samples[] block not found — cap fix skipped")

                # ── Vol side-channel: inject _volSide[] into Common ──────────
                # _vcol_side_capture[0] is {(raw_pat, row, ch): vol} for every
                # S3M cell whose volcol was dropped because an effect (e.g. Gxx
                # tone-porta) occupied the effect slot. Build a dense int array
                # packed 4 bytes/int, indexed by noteIdx = (densePat*64+row)*NC+ch.
                # _gcoBody's vol scan will check this alongside the normal Cxx
                # check to restore per-row volume accents on porta rows (inst-25).
                if _vcol_side_capture[0] and fmt == 'S3M':
                    try:
                        import re as _re_vs
                        _m_sp = _re_vs.search(
                            r'const int\s+songPositions\[(\d+)\]\s*=\s*int\[\]\(([^)]+)\)',
                            _ct)
                        if _m_sp:
                            _glsl_sp = [int(x.strip()) for x in _m_sp.group(2).split(',')]
                            # Build raw-pat → dense-id mapping from (song-pos, dense-id) pairs
                            _raw2d = {}
                            for _spi, _did in enumerate(_glsl_sp):
                                if _spi < len(mod.song_positions):
                                    _raw2d[mod.song_positions[_spi]] = _did
                            _nc_vs = mod.num_channels
                            _maxd  = max(_glsl_sp) if _glsl_sp else 0
                            _ncells = (_maxd + 1) * 64 * _nc_vs
                            _vsarr = [255] * _ncells
                            _n_applied = 0
                            for (rp, rw, rc), rv in _vcol_side_capture[0].items():
                                if rp in _raw2d:
                                    _di = _raw2d[rp]
                                    _vi = (_di * 64 + rw) * _nc_vs + rc
                                    if 0 <= _vi < _ncells:
                                        _vsarr[_vi] = rv
                                        _n_applied += 1
                            # Build sparse list of (noteIdx, vol) pairs
                            _sparse = []
                            for _ci in range(_ncells):
                                _bv = _vsarr[_ci]
                                if _bv != 255:
                                    _sparse.append((_ci << 7) | (_bv & 0x7F))
                            _sparse.sort()
                            _ns = len(_sparse)
                            # Switch-case lookup: no global const array → zero ANGLE
                            # private-variable register cost.  Dense/sparse const-array
                            # approaches flood ANGLE's "private variable" budget on many
                            # GPUs (ANGLE maps each global const int[] slot to a D3D
                            # constant register; total budget is GPU/driver-specific and
                            # can be as low as 256).  A switch statement generates branch
                            # instructions instead — no constant register allocation.
                            # 2-level nested switch.  A single flat switch
                            # with thousands of sparse cases (pod: 4857 on a
                            # ~100 KB line) is pathologically slow for the
                            # ShaderToy/ANGLE shader compiler (switch lowering
                            # is super-linear in cases-per-switch) → "Sound
                            # takes forever to load".  Splitting by the high
                            # bits into many SMALL inner switches keeps it
                            # pure-branch (zero const-register cost — still
                            # compiles on a 256-register-budget GPU, unlike a
                            # const array) but each sub-switch is tiny, so the
                            # compiler digests it far faster.  Identical
                            # results; only the source shape changes.
                            _pairs = sorted((e >> 7, e & 0x7F)
                                            for e in _sparse)
                            _mx = _pairs[-1][0] if _pairs else 0
                            _SH = max(0, _mx.bit_length() - 7)  # ≤~128 outer
                            _bk = {}
                            for _k, _v in _pairs:
                                _bk.setdefault(_k >> _SH, []).append((_k, _v))
                            _inner = ''.join(
                                f'case {_b}:switch(ni){{'
                                + ''.join(f'case {_k}:return {_v};'
                                          for _k, _v in _lst)
                                + 'default:return 255;}'
                                for _b, _lst in sorted(_bk.items()))
                            _vs_decl = (
                                "// Vol side-channel: 2-level switch "
                                "(budget-safe, fast compile).\n"
                                f"int _vsGet(int ni){{switch(ni>>{_SH}){{"
                                + _inner
                                + "default:return 255;}}\n")
                            _enc_mode = (f"2lvl-switch[{_ns}/"
                                         f"{len(_bk)}bkt>>{_SH}]")
                            # Inject after songPositions declaration
                            _sp_end = _ct.find('\n', _ct.find('const int   songPositions'))
                            if _sp_end < 0:
                                _sp_end = _ct.find('\n', _ct.find('const int songPositions'))
                            if _sp_end >= 0:
                                _ct = _ct[:_sp_end+1] + _vs_decl + _ct[_sp_end+1:]
                            else:
                                _ct = _ct + '\n' + _vs_decl
                            print(f"   ✓ _vsSparse {_enc_mode} injected ({_n_applied} vol overrides, "
                                  f"{len(_vcol_side_capture[0])} total dropped volcols)")
                        else:
                            print("   WARNING: songPositions not found — _volSide skip")
                    except Exception as _vs_err:
                        print(f"   WARNING: _volSide inject failed ({_vs_err})")

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
                                     'no_phatbass':   getattr(args, '_compat_no_phatbass', False),
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
                             'no_phatbass':   getattr(args, '_compat_no_phatbass', False),
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
    # Common: --png → keep the legacy Common (rename to final). Otherwise → delete it.
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
    # Skipped only for --png, which uses a totally different data path
    # (samples in a PNG texture, no VQ arrays exist to move).
    if args.use_png:
        print("   📌 --png: legacy PNG-loaded Common (getChannelOutput inline, samples via texelFetch)")
    else:
        try:
            _glsl_sound = base_name + "_shadertoy_sound.glsl"
            if _os2.path.exists(glsl_common_file) and _os2.path.exists(_glsl_sound):
                with open(glsl_common_file) as _f: _common_src = _f.read()
                with open(_glsl_sound)       as _f: _sound_src  = _f.read()
                # smpLen rename: Sound tab baked by VQ encoder still uses the
                # old field name 'length'; rename all struct field accesses.
                _sound_src = _sound_src.replace('smp.length', 'smp.smpLen')

                import re as _re3
                # (porta-timebase _dt→50.0 experiment reverted: the dual-rig
                # audit measured it WORSE — pos-3 inst-25 0.1%→3.9%, the
                # 52.0047/50 mismatch. pos-3 porta POSITION is already accurate;
                # the "mauled/staccato" symptom is volume/note-cut, not pitch.)

                # Capture all vqCodes/vqCodebook array decls (multi-line: const ivec4 vqCodes0[…] = …;)
                _arr_pat = _re3.compile(
                    r'(?:^//[^\n]*\n)?const\s+ivec4\s+(?:vqCodes|vqCodebook)\d+\[\d+\]\s*=\s*ivec4\[\]\([^;]+\);',
                    _re3.MULTILINE | _re3.DOTALL)
                _arrays = _arr_pat.findall(_common_src)
                _common_src = _arr_pat.sub('', _common_src)

                # Move _vsSparse[] + _vsGet() to Sound prelude — only used by _gcoBody.
                # Keeping in Common bloats Common past ShaderToy's tab size limit.
                # Brace-count extraction — robust to the 2-level nested
                # switch (the old regex `\{(?:\{[^}]*\}|[^{}])*\}` only
                # balanced ONE nesting level and truncated the nested
                # _vsGet, dropping its function-closing brace → broken
                # Sound).  Walk from _vsGet's first '{' to its matching '}'.
                _gi = _common_src.find('int _vsGet(')
                if _gi >= 0:
                    # absorb optional _vsSparse[] array + a leading // comment
                    _bstart = _gi
                    _mvs = _re3.search(
                        r'const\s+int\s+_vsSparse\[\d+\]\s*=\s*int\[\]\([^;]+\);\n',
                        _common_src[max(0, _gi-200000):_gi])
                    if _mvs:
                        _bstart = max(0, _gi-200000) + _mvs.start()
                    _ls = _common_src.rfind('\n', 0, _bstart)
                    _cl = _common_src[_ls+1:_bstart]
                    if _cl.lstrip().startswith('//'):
                        _bstart = _ls + 1
                    _ob = _common_src.find('{', _gi)
                    _d = 0
                    _vend = -1
                    for _p in range(_ob, len(_common_src)):
                        if _common_src[_p] == '{':
                            _d += 1
                        elif _common_src[_p] == '}':
                            _d -= 1
                            if _d == 0:
                                _vend = _p
                                break
                    if _vend > 0:
                        _vs_block = _common_src[_bstart:_vend+1]
                        _arrays = [_vs_block] + _arrays  # → Sound prelude
                        _common_src = (_common_src[:_bstart]
                                       + _common_src[_vend+1:])

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

                # ── Fix: _gcoBody instrument guard must match samples[] ───
                # The _gcoBody moved into Sound here was extracted from the
                # VQ common, whose copy still has the hardcoded `> 31` guard
                # (that create_shadertoy call saw _NSMP=31). samples[] was
                # rebuilt to the true count (2ND_PM=54); an under-count
                # guard silences instruments 32+ (voice/vocal samples).
                # Lift the guard (and any isBass cap) to the real count
                # read from the rebuilt samples[] array.  (feedback_s3m_
                # instrument_cap — final reconciliation across the split.)
                _m_n = _re3.search(r'const SampleInfo samples\[(\d+)\]', _common_src) \
                       or _re3.search(r'const SampleInfo samples\[(\d+)\]', _sound_src)
                if _m_n:
                    _real_n = int(_m_n.group(1))
                    _sound_src, _ng = _re3.subn(
                        r'(trigNote\.instrument <= 0 \|\| trigNote\.instrument > )\d+',
                        r'\g<1>' + str(_real_n), _sound_src)
                    _sound_src = _re3.sub(r'(inst >= 1 && inst <= )\d+',
                                          r'\g<1>' + str(_real_n), _sound_src)
                    if _ng:
                        print(f"   ✓ _gcoBody instrument guard → > {_real_n} "
                              f"(was 31; matches rebuilt samples[])")

                # ── Vol side-channel: patch _gcoBody to apply _volSide[] ────
                # For S3M files with dropped volcols, inject a _volSide[] lookup
                # at the 3 vol-scan sites: latch row, forward scan, trigger row.
                # The lookup extracts the packed byte at noteIdx = (densePat*64+row)*NC+ch
                # and calls VOL_SET if it != 255, restoring per-row volume accents
                # on porta rows where Cxx wasn't encoded (fixes inst-25 staccato).
                if '_vsGet' in _sound_src and fmt == 'S3M':
                    def _vs_lookup(pos_var, row_var, tick_var):
                        return (
                            f' {{ int _vsI=(songPositions[{pos_var}]*64+{row_var})*NUM_CHANNELS+ch;'
                            f' int _vsV=_vsGet(_vsI);'
                            f' if(_vsV<255) VOL_SET(_vsV,{tick_var}); }}'
                        )
                    # 1. Latch row: inject BEFORE if-else chain (appending after orphans the else-if)
                    _sound_src = _sound_src.replace(
                        "// Apply latch row's tick-0 vol effects (transitions stamped at row tick).\n"
                        "            if (_latchNote.effect == 0xC) VOL_SET(min(_latchNote.param, 64), _latchTickF);",
                        "// Apply latch row's tick-0 vol effects (transitions stamped at row tick).\n"
                        "            " + _vs_lookup('_instPat', '_instRow', '_latchTickF').strip() + "\n"
                        "            if (_latchNote.effect == 0xC) VOL_SET(min(_latchNote.param, 64), _latchTickF);")
                    # 2. Forward vol scan: inject BEFORE the if-else chain
                    _sound_src = _sound_src.replace(
                        'float _vTickF = float(fetchTick(_sgrV));\n'
                        '                if (_vn.effect == 0xC) VOL_SET(min(_vn.param, 64), _vTickF);',
                        'float _vTickF = float(fetchTick(_sgrV));\n'
                        '                ' + _vs_lookup('_vfp', '_vfr', '_vTickF').strip() + '\n'
                        '                if (_vn.effect == 0xC) VOL_SET(min(_vn.param, 64), _vTickF);')
                    # 3. Trigger row: inject BEFORE the if-else chain so we don't
                    #    break the `else if` that immediately follows the Cxx block.
                    _sound_src = _sound_src.replace(
                        'if (trigNote.effect == 0xC) {\n        VOL_SET(min(trigNote.param, 64), _triggerTickF);\n    }',
                        _vs_lookup('trigPat', 'trigRow', '_triggerTickF').lstrip()
                        + '\n    if (trigNote.effect == 0xC) {\n        VOL_SET(min(trigNote.param, 64), _triggerTickF);\n    }')
                    # 4. Continuation scan (_fn): rows trigRow+1..pos.row-1 — inject BEFORE if-else chain
                    _sound_src = _sound_src.replace(
                        'float _fnTickF = float(fetchTick(_sgr));\n'
                        '            if (_fn.effect == 0xC)',
                        'float _fnTickF = float(fetchTick(_sgr));\n'
                        '            ' + _vs_lookup('_fp', '_fr', '_fnTickF').strip() + '\n'
                        '            if (_fn.effect == 0xC)')
                    # 5. Current row partial (_pcr): pos.row itself.
                    #    The "if (_pcr.instrument <= 0 && _pcr.period <= 0)" guard skips Gxx+note
                    #    rows (period > 0), so hoist the volcol lookup BEFORE that guard with its
                    #    own tick variable so porta-target rows also get the accent.
                    _vs5_block = (
                        '        { int _vs5Sgr = patTickOffset[pos.songPos] + (pos.row - patStartRow[pos.songPos]);'
                        ' float _vs5T = float(fetchTick(_vs5Sgr));'
                        + _vs_lookup('pos.songPos', 'pos.row', '_vs5T').strip()
                        + ' }\n'
                        '        // Current row partial (non-trigger row only — trigger handled above)'
                    )
                    _sound_src = _sound_src.replace(
                        '        // Current row partial (non-trigger row only — trigger handled above)',
                        _vs5_block)
                    print(f"   ✓ _gcoBody patched with _volSide[] vol side-channel (5 sites)")

                # ── Fine vol slide fix: S3M DF2 (upper nibble=F) = fine DOWN,
                # D2F (lower nibble=F) = fine UP.  Fires once at tick 0, not
                # speed-1 times like a regular slide.  All 7 VOL_SET sites.
                def _fine_sub(expr_with_mult):
                    # expr_with_mult: '(_vu>0?_vu:-_vd) * MULT'
                    # returns fine-slide-aware ternary expression
                    mult = expr_with_mult.split('* ')[1]
                    return (f'(_vu==0xF&&_vd>0?-_vd:'
                            f'_vd==0xF&&_vu>0?_vu:'
                            f'(_vu>0?_vu:-_vd)*{mult})')
                for _mult in ('_ftV', '_full', '_pct'):
                    _old = f'(_vu>0?_vu:-_vd) * {_mult}'
                    _new = _fine_sub(f'x * {_mult}')
                    _sound_src = _sound_src.replace(_old, _new)
                # Latch-row slide (uses _vu/_vd directly with _ftL multiplier)
                _sound_src = _sound_src.replace(
                    'int _step = (_vu>0) ? _vu : -_vd;\n'
                    '                int _ftL  = fetchTick(_latchSgr + 1) - fetchTick(_latchSgr) - 1;\n'
                    '                VOL_SET(clamp(_volCurr + _step * _ftL, 0, 64), _latchTickF);',
                    'int _ftL  = fetchTick(_latchSgr + 1) - fetchTick(_latchSgr) - 1;\n'
                    '                int _fsDL = _vu==0xF&&_vd>0?-_vd:_vd==0xF&&_vu>0?_vu:(_vu>0?_vu:-_vd)*_ftL;\n'
                    '                VOL_SET(clamp(_volCurr + _fsDL, 0, 64), _latchTickF);')
                # Trigger-row slide (uses _su/_sd, two branches: current vs past row)
                _sound_src = _sound_src.replace(
                    '    } else if (trigNote.effect == 0xA || trigNote.effect == 0x6 || trigNote.effect == 0x5) {\n'
                    '        // 0x5 = tone+vol slide: pitch handled by 0x3-equivalent block, vol param same as 0xA\n'
                    '        int _su = (trigNote.param>>4)&0xF, _sd = trigNote.param&0xF;\n'
                    '        int _step = (_su>0) ? _su : -_sd;\n'
                    '        if (trigPat == pos.songPos && trigRow == pos.row) {\n'
                    '            VOL_SET(clamp(_volCurr + _step * _pct, 0, 64), _triggerTickF);\n'
                    '        } else {\n'
                    '            int _ts = fetchTick(patTickOffset[trigPat]+(trigRow-patStartRow[trigPat])+1)\n'
                    '                    - fetchTick(patTickOffset[trigPat]+(trigRow-patStartRow[trigPat]));\n'
                    '            VOL_SET(clamp(_volCurr + _step * (_ts-1), 0, 64), _triggerTickF);\n'
                    '        }\n'
                    '    }',
                    '    } else if (trigNote.effect == 0xA || trigNote.effect == 0x6 || trigNote.effect == 0x5) {\n'
                    '        // 0x5 = tone+vol slide: pitch handled by 0x3-equivalent block, vol param same as 0xA\n'
                    '        int _su = (trigNote.param>>4)&0xF, _sd = trigNote.param&0xF;\n'
                    '        bool _fsT = (_su==0xF&&_sd>0)||(_sd==0xF&&_su>0);\n'
                    '        int _fsTD = _su==0xF&&_sd>0?-_sd:_su;\n'
                    '        int _step = (_su>0) ? _su : -_sd;\n'
                    '        if (trigPat == pos.songPos && trigRow == pos.row) {\n'
                    '            VOL_SET(clamp(_volCurr + (_fsT?_fsTD:_step*_pct), 0, 64), _triggerTickF);\n'
                    '        } else {\n'
                    '            int _ts = fetchTick(patTickOffset[trigPat]+(trigRow-patStartRow[trigPat])+1)\n'
                    '                    - fetchTick(patTickOffset[trigPat]+(trigRow-patStartRow[trigPat]));\n'
                    '            VOL_SET(clamp(_volCurr + (_fsT?_fsTD:_step*(_ts-1)), 0, 64), _triggerTickF);\n'
                    '        }\n'
                    '    }')
                # Verify all 5 patterns were found (count replacements)
                _fs_checks = [
                    '(_vu==0xF&&_vd>0?-_vd:_vd==0xF&&_vu>0?_vu:(_vu>0?_vu:-_vd)*_ftV)',
                    '(_vu==0xF&&_vd>0?-_vd:_vd==0xF&&_vu>0?_vu:(_vu>0?_vu:-_vd)*_full)',
                    '(_vu==0xF&&_vd>0?-_vd:_vd==0xF&&_vu>0?_vu:(_vu>0?_vu:-_vd)*_pct)',
                    '_fsDL',
                    '_fsT',
                ]
                _fs_found = sum(1 for p in _fs_checks if p in _sound_src)
                print(f"   ✓ fine vol slide (DF2/D2F) patched at {_fs_found}/5 pattern groups")

                # ── Instrument-only vol-reset fix (S_nn P0 / XM inst-only rows) ──────
                # ProTracker: a row with instrument but no period (e.g. S01 P0 A0F) does
                # NOT restart the sample — it only RESETS the channel volume to the sample's
                # default, then applies the row's vol effects (Axx/Cxx etc.) from that reset.
                # The embedded GLSL's current-row partial skips instrument-only rows because
                # the old condition was `_pcr.instrument <= 0 && _pcr.period <= 0`.  That
                # means A0F on an instrument-only row is ignored and the vol stays at 0 from
                # the prior trigger's completed slide → silence.
                # Fix: widen condition to `_pcr.period <= 0` (includes inst-only rows), and
                # inject VOL_SET(sample_default_vol) before the row's vol effects.
                # Same fix in the forward scan for inst-only rows between trigger and current.
                _pcr_cond_old = (
                    '        // Current row partial (non-trigger row only — trigger handled above)\n'
                    '        if (_pcr.instrument <= 0 && _pcr.period <= 0) {\n'
                    '            // Tick stamp for current-row partial vol effects: the current row\'s\n'
                    '            // start tick. The 64-sample ramp completes well within the first\n'
                    '            // tick, so any vol change "happening at this row" reads as having\n'
                    '            // ramped to its final value almost immediately.\n'
                    '            int _curSgr = patTickOffset[pos.songPos] + (pos.row - patStartRow[pos.songPos]);\n'
                    '            float _curTickF = float(fetchTick(_curSgr));'
                )
                _pcr_cond_new = (
                    '        // Current row partial (non-trigger row only — trigger handled above).\n'
                    '        // Instrument-only rows (inst>0, period=0) also enter: ProTracker resets\n'
                    '        // vol to sample default before applying the row\'s effects.\n'
                    '        if (_pcr.period <= 0) {\n'
                    '            // Tick stamp for current-row partial vol effects: the current row\'s\n'
                    '            // start tick. The 64-sample ramp completes well within the first\n'
                    '            // tick, so any vol change "happening at this row" reads as having\n'
                    '            // ramped to its final value almost immediately.\n'
                    '            int _curSgr = patTickOffset[pos.songPos] + (pos.row - patStartRow[pos.songPos]);\n'
                    '            float _curTickF = float(fetchTick(_curSgr));\n'
                    '            // ProTracker instrument-only row: vol reset to sample default vol\n'
                    '            if (_pcr.instrument > 0 && _pcr.instrument <= 31) {\n'
                    '                VOL_SET(samples[_pcr.instrument - 1].volume, _curTickF);\n'
                    '            }'
                )
                if _pcr_cond_old in _sound_src:
                    _sound_src = _sound_src.replace(_pcr_cond_old, _pcr_cond_new)
                    print(f"   ✓ instrument-only vol-reset fix: current-row partial condition widened")
                else:
                    print(f"   ✗ WARNING: current-row partial condition not found for vol-reset fix")

                # Forward scan instrument-only vol reset (rows between trigger and current row).
                # Same ProTracker rule: any intermediate row with inst>0, period=0 resets vol.
                _fwd_volreset_old = (
                    '            // Volume effects (transition stamped at this row\'s start tick)\n'
                    '            float _fnTickF = float(fetchTick(_sgr));\n'
                    '            if (_fn.effect == 0xC)\n'
                    '                VOL_SET(min(_fn.param, 64), _fnTickF);'
                )
                _fwd_volreset_new = (
                    '            // Instrument-only row in forward scan: ProTracker vol reset\n'
                    '            if (_fn.instrument > 0 && _fn.period == 0 && _fn.instrument <= 31) {\n'
                    '                VOL_SET(samples[_fn.instrument - 1].volume, float(fetchTick(_sgr)));\n'
                    '            }\n'
                    '            // Volume effects (transition stamped at this row\'s start tick)\n'
                    '            float _fnTickF = float(fetchTick(_sgr));\n'
                    '            if (_fn.effect == 0xC)\n'
                    '                VOL_SET(min(_fn.param, 64), _fnTickF);'
                )
                if _fwd_volreset_old in _sound_src:
                    _sound_src = _sound_src.replace(_fwd_volreset_old, _fwd_volreset_new)
                    print(f"   ✓ instrument-only vol-reset fix: forward scan vol reset added")
                else:
                    print(f"   ✗ WARNING: forward scan vol block not found for vol-reset fix")

                # ── Float32 tick-boundary fix: pos.tick may compute as 0.9999…
                # instead of 1.0 when samppos/822 is exact, making int(pos.tick)
                # return the wrong (lower) tick count for _pct.  Add epsilon.
                _old_pct = 'int _pct = int(pos.tick);'
                _new_pct = 'int _pct = int(pos.tick + 0.0001);'
                if _old_pct in _sound_src:
                    _sound_src = _sound_src.replace(_old_pct, _new_pct)
                    print(f"   ✓ float32 tick-boundary epsilon applied to _pct")
                else:
                    print(f"   ✗ WARNING: could not find _pct definition for epsilon fix")

                # ── Vibrato/tremolo tick-0 phase fix ─────────────────────────
                # MikIT calls do_vibrato()/do_tremolo() on EVERY tick including
                # tick 0 (vibptr advances before the first sample is rendered).
                # The b64 encoder's phase accumulator counts T ticks elapsed
                # → computes pos.tick * speed (0-indexed).  Correct is
                # (pos.tick + 1) * speed so tick 0 already contributes one step,
                # matching MikIT's "advance then render" order.
                # Same +1 applies to completed rows: spd ticks fire, not spd-1.
                _vib_tre_patches = [
                    # vibrato — trigger row
                    ('_vibPos = int(pos.tick) * _vS;',
                     '_vibPos = (int(pos.tick) + 1) * _vS;'),
                    # vibrato — completed trigger row
                    ('_vibPos = _trigIsVibActive ? (_trSpd - 1) * _vS : 0;',
                     '_vibPos = _trigIsVibActive ? _trSpd * _vS : 0;'),
                    # vibrato — completed intermediate rows (loop body)
                    ('_vibPos += (_spd - 1) * _vS;',
                     '_vibPos += _spd * _vS;'),
                    # vibrato — current row partial (non-trigger)
                    ('_vibPos += int(pos.tick) * _vS;',
                     '_vibPos += (int(pos.tick) + 1) * _vS;'),
                    # tremolo — trigger row
                    ('_trePos = int(pos.tick) * _tS;',
                     '_trePos = (int(pos.tick) + 1) * _tS;'),
                    # tremolo — completed trigger row
                    ('_trePos = (_trSpd - 1) * _tS;',
                     '_trePos = _trSpd * _tS;'),
                    # tremolo — completed intermediate rows
                    ('_trePos += (_spd - 1) * _tS;',
                     '_trePos += _spd * _tS;'),
                    # tremolo — current row partial (non-trigger)
                    ('_trePos += int(pos.tick) * _tS;',
                     '_trePos += (int(pos.tick) + 1) * _tS;'),
                ]
                _vt_fixed = 0
                for _old, _new in _vib_tre_patches:
                    if _old in _sound_src:
                        _sound_src = _sound_src.replace(_old, _new)
                        _vt_fixed += 1
                if _vt_fixed == 8:
                    print(f"   ✓ vibrato/tremolo tick-0 phase fix applied (8/8 sites)")
                else:
                    print(f"   ✗ WARNING: vib/trem phase fix only matched {_vt_fixed}/8 sites")

                # ── Tremolo tick-boundary ramp (declick) ─────────────────────
                # Command-7 tremolo steps _tremoloDelta once PER TICK and it was
                # added to _effVol RAW (unlike base volume, which gets _vRamp).
                # So _effVol jumped at every tick boundary → a click each tick
                # (~52/s buzz) on fast tremolo leads, e.g. 2ND_PM pattern 30's
                # inst-25 run (verified: every click landed exactly on a tick
                # boundary). Fix = ramp the delta from the PREVIOUS tick's value
                # to the current over 64 samples right after the boundary — the
                # same noclick approach as _vRamp. The LFO itself is untouched
                # (64 samp ≈ 7% of an 848-sample tick), only the inter-tick step
                # is smoothed. fract(pos.tick) = position within the current
                # integer tick; *(SAMP_PER_TICK)/64 → 0→1 over the first 64 samp.
                _trem_old = '_tremoloDelta = (_tP < 32) ? _tDelta : -_tDelta;'
                _trem_new = (
                    'float _curTD = (_tP < 32) ? _tDelta : -_tDelta; '
                    'int _tPp = (_trePos - _tS) & 63; '
                    'float _tDp = (vibTab[_tPp & 31] * float(_tD)) / 64.0; '
                    'float _prevTD = (_tPp < 32) ? _tDp : -_tDp; '
                    '_tremoloDelta = mix(_prevTD, _curTD, '
                    'clamp(fract(pos.tick) * ((44100.0/TICKS_PER_SEC)/64.0), 0.0, 1.0));'
                )
                if _trem_old in _sound_src:
                    _sound_src = _sound_src.replace(_trem_old, _trem_new, 1)
                    print(f"   ✓ tremolo tick-boundary ramp applied (declick)")
                else:
                    print(f"   ✗ WARNING: tremolo delta line not found for ramp declick")

                # ── Arpeggio tick-boundary epsilon fix ──────────────────────
                # pos.tick is a float computed from sampleTime.  At the exact
                # tick-3 boundary pos.tick = 2.9999…  so int(pos.tick) = 2
                # instead of 3 → _arpStep = 2 instead of 0, phase-shifting
                # the whole 3-step arpeggio cycle from that point on.
                # _pct already has the +0.0001 epsilon fix applied above, so
                # use it for the modulo-3 calculation (integer math, no FP).
                _arp_old = 'int _arpStep = int(pos.tick) - int(pos.tick / 3.0) * 3;'
                _arp_new = 'int _arpStep = _pct - (_pct / 3) * 3;'
                if _arp_old in _sound_src:
                    _sound_src = _sound_src.replace(_arp_old, _arp_new)
                    print(f"   ✓ arpeggio tick-boundary epsilon fix applied")
                else:
                    print(f"   ✗ WARNING: arpeggio epsilon fix pattern not found")

                # ── c2spd exact float rate-scale (inst33 pos-20 phase-drift) ──
                # The encoder no longer integer-rounds period*8363/c2 (that
                # quantization lost ~0.5% pitch for non-8363 samples and
                # accumulated over loop iterations on sustained looped samples
                # — the "out of phase lead" at 2ND_PM pos 20). c2spd is now
                # carried exactly in SampleInfo.c2sp and applied here as a
                # float scale on the source-sample read rate. For standard
                # 8363 (and all MOD) samples float(c2sp)/8363.0 == 1.0 exactly
                # → the generated GLSL is byte-identical (audits unaffected).
                _cf_old = 'c4speeds[smp.finetune & 0xF] * 428.0;'
                _cf_new = ('c4speeds[smp.finetune & 0xF] * 428.0 '
                           '* (float(smp.c2sp) / 8363.0);')
                _cf_n = _sound_src.count(_cf_old)
                _frq_old = ('float freq = periodToFreqFt(max(1, int(basePeriod)), '
                            'smp.finetune);')
                _frq_new = ('float freq = periodToFreqFt(max(1, int(basePeriod)), '
                            'smp.finetune) * (float(smp.c2sp) / 8363.0);')
                _frq_n = _sound_src.count(_frq_old)
                if _cf_n >= 1:
                    _sound_src = _sound_src.replace(_cf_old, _cf_new)
                if _frq_n == 1:
                    _sound_src = _sound_src.replace(_frq_old, _frq_new)
                if _cf_n == 3 and _frq_n == 1:
                    print(f"   ✓ c2spd float rate-scale applied "
                          f"(3 _Cf accumulators + 1 freq)")
                else:
                    print(f"   ✗ WARNING: c2spd rate-scale matched "
                          f"{_cf_n}/3 _Cf + {_frq_n}/1 freq "
                          f"(expected 3 + 1) — pitch fix incomplete")

                # ── Targeted base-pitch de-quantization ────────────────────
                # The encoder routes note pitch through a 12-bit integer
                # Amiga period table.  For high notes the period shrinks to
                # 28-67 with no fractional headroom → up to ~21 cents error
                # vs MikIT's exact c5speed·2^((note-60)/12); ~0 for oct 0-5.
                # That register-dependent error is the audible "non-uniform"
                # pitch / sustained-note drift.  Recover the intended
                # semitone from trigNote.period and scale ONLY the 3 _Cf
                # rate accumulators by period/exactPeriod:
                #   rate = _Cf·corr·_dt/_Pt = _Cf·_dt/exactP
                # _Pt itself, the _fSamplePosAcc integrator, porta, vibrato,
                # arpeggio and the 100%-verified effect-count audits are left
                # bit-identical (only the absolute note pitch is corrected,
                # not the relative effect math).  Gated by DEQUANT_PITCH so
                # it can be flipped off in the Sound tab without rebuilding.
                _dq_anchor = ('samples[trigNote.instrument - 1];\n'
                              '    if (smp.smpLen == 0) return 0.0;')
                _dq_block = (
                    'samples[trigNote.instrument - 1];\n'
                    '    if (smp.smpLen == 0) return 0.0;\n'
                    '#define DEQUANT_PITCH 0\n'
                    '    float _pitchCorr = 1.0;\n'
                    '#if DEQUANT_PITCH\n'
                    '    // STATIC notes only: any pitch-bending effect (porta\n'
                    '    // up/dn 1/2, tone-porta 3, vibrato 4/6, arpeggio 0\n'
                    '    // w/param, 7) or an active tone-slide deliberately\n'
                    '    // bends the pitch off-grid — de-quantizing then would\n'
                    '    // double-count vs the effect-modulated _Pt (regresses\n'
                    '    // porta-heavy leads). Leave those exactly as-is.\n'
                    '    int _te = trigNote.effect;\n'
                    '    bool _tePitch = (_te == 0x1 || _te == 0x2 || _te == 0x3 ||\n'
                    '                     _te == 0x4 || _te == 0x6 || _te == 0x7 ||\n'
                    '                     (_te == 0x0 && trigNote.param != 0));\n'
                    '    if (!_tePitch && toneSlideTarget <= 0\n'
                    '            && float(trigNote.period) > 0.0) {\n'
                    '        float _bp = float(trigNote.period);\n'
                    '        int _ni = int(floor(-12.0 * log2(_bp / 214.0) + 0.5));\n'
                    '        float _exactP = 214.0 * exp2(-float(_ni) / 12.0);\n'
                    '        if (_exactP > 0.0) _pitchCorr = _bp / _exactP;\n'
                    '    }\n'
                    '#endif')
                _dq_cf_old = '* 428.0 * (float(smp.c2sp) / 8363.0);'
                _dq_cf_new = '* 428.0 * (float(smp.c2sp) / 8363.0) * _pitchCorr;'
                _dq_an = _sound_src.count(_dq_anchor)
                _dq_cfn = _sound_src.count(_dq_cf_old)
                if _dq_an == 1 and _dq_cfn == 3:
                    _sound_src = _sound_src.replace(_dq_anchor, _dq_block, 1)
                    _sound_src = _sound_src.replace(_dq_cf_old, _dq_cf_new)
                    print(f"   ✓ base-pitch de-quantization applied "
                          f"(DEQUANT_PITCH gate; 3 _Cf × _pitchCorr; "
                          f"_Pt/integrator/audits untouched)")
                else:
                    print(f"   ✗ WARNING: de-quant NOT applied "
                          f"(anchor×{_dq_an} exp 1, _Cf×{_dq_cfn} exp 3) "
                          f"— high-note pitch still 12-bit quantized")

                # ── Tone-porta current-row target fix (porta-lead declick) ──
                # The forward scan rebuilds pitch trigger→current, overwriting
                # targetPeriod with EACH intermediate row's tone-porta target.
                # By the time the current-row "head" integrates its glide,
                # targetPeriod is left at row(current-1)'s target, not the
                # CURRENT row's. For an all-tone-porta lead (2ND_PM inst25:
                # note+Gxx every row), the head then glides toward the WRONG
                # (previous) target, so the sample-read position it integrates
                # differs from what the forward scan integrates for that SAME
                # row one sample later (when the row flips to "completed").
                # The position therefore STEPS at every row boundary → audible
                # click on the lead (confirmed: GLSL maxΔ 0.63 vs oracle 0.30,
                # clicks land on Gxx glide rows, not note-ons; crossfade only
                # covers note-on triggers so it never smooths these).
                # Fix: re-apply the current row's target (toneSlideTarget) AFTER
                # the forward scan, right before the head's _pStartCur. Now head
                # and forward-scan integrate each row toward the SAME target →
                # position is continuous across the boundary → no step → no
                # click. Only fires when the current row carries a fresh target
                # (toneSlideTarget>0); continuation rows keep the scan's target.
                _tpfix_old = '    float _pStartCur = effectivePeriod;'
                _tpfix_new = ('    if (toneSlideTarget > 0) targetPeriod = float(toneSlideTarget);\n'
                              '    float _pStartCur = effectivePeriod;')
                _tpfix_n = _sound_src.count(_tpfix_old)
                if _tpfix_n == 1:
                    _sound_src = _sound_src.replace(_tpfix_old, _tpfix_new, 1)
                    print("   ✓ tone-porta current-row target fix applied "
                          "(declick on all-Gxx porta leads; head target == "
                          "forward-scan target → continuous fSamplePos)")
                else:
                    print(f"   ✗ WARNING: tone-porta target fix NOT applied "
                          f"(anchor×{_tpfix_n} exp 1) — porta-lead clicks remain")

                # ── --xfade: tunable retrigger/note-on declick crossfade window ──
                # getChannelOutput crossfades the OLD voice (s_prev, ramping down)
                # into the NEW voice (s_curr, ramping up) over N samples after each
                # trigger. Default 64 (1.45ms) can be too short to mask the
                # restart discontinuity on busy leads → raise via --xfade.
                _xf_n = int(getattr(args, 'xfade', 64) or 64)
                if _xf_n != 64:
                    _xf_a = 'if (ageSamples < 64.0 && ageSamples >= 0.0) {'
                    _xf_b = 'float t = ageSamples / 64.0;'
                    _xf_na, _xf_nb = _sound_src.count(_xf_a), _sound_src.count(_xf_b)
                    if _xf_na == 1 and _xf_nb == 1:
                        _sound_src = _sound_src.replace(
                            _xf_a, f'if (ageSamples < {float(_xf_n)} && ageSamples >= 0.0) {{')
                        _sound_src = _sound_src.replace(
                            _xf_b, f'float t = ageSamples / {float(_xf_n)};')
                        print(f"   ✓ retrigger declick crossfade window → {_xf_n} samples "
                              f"({_xf_n/44100.0*1000.0:.1f}ms)")
                    else:
                        print(f"   ✗ WARNING: --xfade patch NOT applied "
                              f"(anchors {_xf_na}/{_xf_nb}, expected 1/1)")

                # ── MikIT note-on volume reset (inst25 frozen-accent fix) ───
                # mikit_engine.py:560-575 — a cell carrying an INSTRUMENT
                # number unconditionally resets channel volume to that
                # sample's default volume (porta or kick), THEN a volume
                # column / Cxx on the same cell overrides.  v1.61's volume
                # forward-scan never did the inst→sample-default reset, so on
                # tone-porta leads (inst25: note+G every ~2 rows, volcol=20
                # on the off-rows) the volcol value froze and never returned
                # to full → MikIT oscillates vol 256↔80, GLSL stuck at ~80.
                #
                # FIX (2026-05-22): original approach prepended inst-default
                # VOL_SET BEFORE the vcol VOL_SET, leaving _volPrev=inst_default
                # (64) and _volCurr=vcol (1). At tick 0 of each portamento row,
                # vRamp=0 → effVol=64 instead of 1 → 64× amplitude spike = click.
                # Fix: check vcol first; apply inst-default only as a fallback
                # when no vcol is present. For the trigger row, use VOL_INIT
                # (both prev=curr=V, no ramp) so the crossfade in
                # getChannelOutput provides the declick instead of VOL_SET ramp.
                _miv_old = ('{ int _vsI=(songPositions[_vfp]*64+_vfr)'
                            '*NUM_CHANNELS+ch; int _vsV=_vsGet(_vsI); '
                            'if(_vsV<255) VOL_SET(_vsV,_vTickF); }')
                # Latch fwd scan: vcol first, then inst-default as fallback
                _miv_new = ('{ int _vsI=(songPositions[_vfp]*64+_vfr)'
                            '*NUM_CHANNELS+ch; int _vsV=_vsGet(_vsI); '
                            'if(_vsV<255) VOL_SET(_vsV,_vTickF); '
                            'else if(_vn.instrument > 0 && _vn.instrument <= 54) '
                            'VOL_SET(samples[_vn.instrument - 1].volume, _vTickF); }')
                _miv_n = _sound_src.count(_miv_old)
                # Same rule at the TRIGGER-row vol block (trigNote/_triggerTickF).
                # Use VOL_INIT (not VOL_SET) so no ramp is stamped; the
                # getChannelOutput crossfade provides the trigger declick.
                _miv_t_old = ('{ int _vsI=(songPositions[trigPat]*64+trigRow)'
                              '*NUM_CHANNELS+ch; int _vsV=_vsGet(_vsI); '
                              'if(_vsV<255) VOL_SET(_vsV,_triggerTickF); }')
                _miv_t_new = ('// Trigger-row vol: vcol wins (VOL_INIT—no ramp; '
                              'crossfade in getChannelOutput provides declick).\n'
                              '    { int _vsI=(songPositions[trigPat]*64+trigRow)'
                              '*NUM_CHANNELS+ch; int _vsV=_vsGet(_vsI); '
                              'if(_vsV<255) VOL_INIT(_vsV); '
                              'else if(trigNote.instrument > 0 && '
                              'trigNote.instrument <= 54) '
                              'VOL_INIT(samples[trigNote.instrument - 1].volume); }')
                _miv_tn = _sound_src.count(_miv_t_old)
                # Same rule at the trigger→current continuation scan
                # ("Forward scan: rows STRICTLY between trigger and current",
                # _fn/_fp/_fr/_fnTickF) — covers all-tone-porta leads (ch0
                # inst25: kick at row0, rows 2/4/6/8 = note+inst+G, latch=trig
                # =row0 so the latch+1..trigRow-1 fwd-scan is empty and these
                # inst rows live ONLY in this trig→current scan).
                _miv_c_old = ('{ int _vsI=(songPositions[_fp]*64+_fr)'
                              '*NUM_CHANNELS+ch; int _vsV=_vsGet(_vsI); '
                              'if(_vsV<255) VOL_SET(_vsV,_fnTickF); }')
                # Forward scan: vcol first, then inst-default as fallback
                _miv_c_new = ('{ int _vsI=(songPositions[_fp]*64+_fr)'
                              '*NUM_CHANNELS+ch; int _vsV=_vsGet(_vsI); '
                              'if(_vsV<255) VOL_SET(_vsV,_fnTickF); '
                              'else if(_fn.instrument > 0 && _fn.instrument <= 54) '
                              'VOL_SET(samples[_fn.instrument - 1].volume, _fnTickF); }')
                _miv_cn = _sound_src.count(_miv_c_old)
                # 4th site: the CURRENT row's volcol (_vs5T). When the current
                # playback row carries an instrument but is a tone-porta (not
                # the kick trigger), it is handled by NO inst-reset path —
                # current-row-partial is gated _pcr.instrument<=0, the trigger
                # is an earlier kick, and the continuation scan is rows
                # STRICTLY between trigger and current (excludes current).
                # ch0 inst25 r4/6/8/10 (note+inst+G, no volcol) froze here.
                # MikIT resets vol on ANY inst-bearing cell incl. the current.
                _miv_p_old = ('float _vs5T = float(fetchTick(_vs5Sgr));'
                              '{ int _vsI=(songPositions[pos.songPos]*64'
                              '+pos.row)*NUM_CHANNELS+ch; int _vsV='
                              '_vsGet(_vsI); if(_vsV<255) '
                              'VOL_SET(_vsV,_vs5T); }')
                # Current-row: vcol first, then inst-default as fallback
                _miv_p_new = ('float _vs5T = float(fetchTick(_vs5Sgr));'
                              '{ int _vsI=(songPositions[pos.songPos]*64'
                              '+pos.row)*NUM_CHANNELS+ch; int _vsV='
                              '_vsGet(_vsI); if(_vsV<255) '
                              'VOL_SET(_vsV,_vs5T); '
                              'else if(_pcr.instrument > 0 && _pcr.instrument '
                              '<= 54) VOL_SET(samples[_pcr.instrument - 1]'
                              '.volume, _vs5T); }')
                _miv_pn = _sound_src.count(_miv_p_old)
                if (_miv_n == 1 and _miv_tn == 1 and _miv_cn == 1
                        and _miv_pn == 1):
                    _sound_src = _sound_src.replace(_miv_old, _miv_new)
                    _sound_src = _sound_src.replace(_miv_t_old, _miv_t_new)
                    _sound_src = _sound_src.replace(_miv_c_old, _miv_c_new)
                    _sound_src = _sound_src.replace(_miv_p_old, _miv_p_new)
                    print(f"   ✓ MikIT inst-vol reset: vcol-priority (no-click) "
                          f"applied at fwd-scan + trigger(VOL_INIT) + "
                          f"trig→current + current-row")
                else:
                    print(f"   ✗ WARNING: inst-vol-reset targets "
                          f"fwd×{_miv_n} trig×{_miv_tn} cont×{_miv_cn} "
                          f"cur×{_miv_pn} (expected 1,1,1,1) — frozen")

                # ── pTrigPat=-1 fade-from-silence patch ───────────────────────
                # When no previous trigger exists for a channel (first note ever
                # on that channel), the old code returned s_curr hard (no fade).
                # If the sample doesn't start from 0, this creates an onset click.
                # Fix: within the 64-sample crossfade window, blend from silence
                # (s_prev=0) to s_curr using the same t-ramp — identical declick
                # budget to the pTrigPat>=0 case.
                _ptf_old = ('            return s_prev * (1.0 - t) + s_curr * t;\n'
                            '        }\n'
                            '    }\n'
                            '\n'
                            '    return s_curr;')
                _ptf_new = ('            return s_prev * (1.0 - t) + s_curr * t;\n'
                            '        } else {\n'
                            '            // First note ever on channel: fade from silence.\n'
                            '            float t = clamp(ageSamples / 64.0, 0.0, 1.0);\n'
                            '            return s_curr * t;\n'
                            '        }\n'
                            '    }\n'
                            '\n'
                            '    return s_curr;')
                _ptf_n = _sound_src.count(_ptf_old)
                if _ptf_n == 1:
                    _sound_src = _sound_src.replace(_ptf_old, _ptf_new)
                    print(f"   ✓ pTrigPat=-1 fade-from-silence patch applied")
                else:
                    print(f"   ✗ WARNING: pTrigPat=-1 fade patch: found {_ptf_n} "
                          f"occurrences of anchor (expected 1) — skipped")

                # ── gated stateless ratio-AA (box integrator) ─────────────
                # MikIT's C++ mixer is 2-tap LINEAR with NO anti-aliasing
                # (disassembled: Mix08/16StereoInterp/Noclick = s0+((s1-s0)*
                # frac>>12)); it aliases on pitch-up by design. So AA here is
                # a deliberate "cleaner than the oracle" knob → default OFF,
                # #define-gated, rig-invisible (glsl_vs_mikit audits control-
                # rate freq/vol/pan, not PCM). ShaderToy Sound is stateless
                # per-sample (no IIR recurrence possible), so the stateless
                # equivalent of OpenMPT's ratio-lowpass is a boxcar: average K
                # sub-taps spanning the per-output-sample step (= a moving-
                # average decimation LPF whose width tracks the resample
                # ratio). step = freq/(44100*bwFactor) source-samples/output-
                # sample; step<=1 (not pitched up) → K=1 → bit-identical to
                # getSampleF (zero cost / zero divergence). Works uniformly
                # for linear / bspline / lanczos3 (filters the reconstructed
                # signal, kernel-agnostic).  Enable with #define AA_RESAMPLE 1.
                _aa_sig = ("float getSampleF(int base, float fpos, "
                           "int smpLen, int loopStart, int loopLen) {")
                _aa_call_old = ("s = getSampleF(smp.start, fSamplePos, "
                                "smp.smpLen, smp.loopStart, smp.loopLen);")
                _aa_si = _sound_src.find(_aa_sig)
                _aa_ci = _sound_src.count(_aa_call_old)
                if _aa_si >= 0 and _aa_ci == 1:
                    # brace-count to end of getSampleF (body varies by
                    # resampler) so getSampleAA is defined right after it
                    # (before its only caller — GLSL needs decl-before-use).
                    _b = _sound_src.find('{', _aa_si)
                    _d = 0; _k = _b
                    while _k < len(_sound_src):
                        _c = _sound_src[_k]
                        if _c == '{': _d += 1
                        elif _c == '}':
                            _d -= 1
                            if _d == 0:
                                break
                        _k += 1
                    _aa_on = 1 if getattr(args, 'aa', False) else 0
                    _aa_block = (
                        f"\n\n#ifndef AA_RESAMPLE\n#define AA_RESAMPLE {_aa_on}\n#endif\n"
                        "#ifndef AA_MAX_TAPS\n#define AA_MAX_TAPS 4\n#endif\n"
                        "#if AA_RESAMPLE\n"
                        "// Ratio-width box integrator → stateless decimation\n"
                        "// LPF for pitched-up notes. Divergence from MikIT\n"
                        "// (oracle = plain linear, no AA) — opt-in only.\n"
                        "float getSampleAA(int base, float fpos, int smpLen,"
                        " int loopStart, int loopLen, float step){\n"
                        "    if (step <= 1.0001) return getSampleF(base, fpos,"
                        " smpLen, loopStart, loopLen);\n"
                        "    int K = int(min(ceil(step), float(AA_MAX_TAPS)));\n"
                        "    if (K < 2) return getSampleF(base, fpos, smpLen,"
                        " loopStart, loopLen);\n"
                        "    float acc = 0.0;\n"
                        "    for (int k = 0; k < AA_MAX_TAPS; k++){\n"
                        "        if (k >= K) break;\n"
                        "        float p = fpos + ((float(k)+0.5)/float(K)"
                        " - 0.5) * step;\n"
                        "        if (p < 0.0) p = 0.0;\n"
                        "        if (loopLen > 2 && p >= float(loopStart"
                        "+loopLen))\n"
                        "            p = float(loopStart) + mod(p -"
                        " float(loopStart), float(loopLen));\n"
                        "        acc += getSampleF(base, p, smpLen, loopStart,"
                        " loopLen);\n"
                        "    }\n"
                        "    return acc / float(K);\n"
                        "}\n#endif\n")
                    _sound_src = (_sound_src[:_k+1] + _aa_block
                                  + _sound_src[_k+1:])
                    _aa_call_new = (
                        "#if AA_RESAMPLE\n"
                        "        s = getSampleAA(smp.start, fSamplePos, "
                        "smp.smpLen, smp.loopStart, smp.loopLen, "
                        "freq / (44100.0 * float(max(1, smp.bwFactor))));\n"
                        "#else\n"
                        "        " + _aa_call_old + "\n"
                        "#endif")
                    _sound_src = _sound_src.replace(
                        _aa_call_old, _aa_call_new, 1)
                    if _aa_on:
                        print("   ✓ gated ratio-AA injected & ENABLED via "
                              "--aa (#define AA_RESAMPLE 1) — alias suppression "
                              "on pitched-up notes; deliberate divergence from "
                              "the linear/no-AA MikIT/IT oracle")
                    else:
                        print("   ✓ gated ratio-AA injected, OFF (oracle-1:1; "
                              "pass --aa to enable, or #define AA_RESAMPLE 1)")
                else:
                    print(f"   ⚠ ratio-AA skipped (getSampleF sig×"
                          f"{1 if _aa_si>=0 else 0} call×{_aa_ci}; "
                          f"expected 1,1) — playback unchanged")

                # ── IT instrument resonant filter (it2play 2-pole LPF) ────
                # Recovered algorithm (it2play it2drivers/hq.c, captured in
                # session a3158c73): coeffs from cutoff/res, mixrate-dependent
                # (filterStep=24). The IIR is recursive; v1.61's Sound shader
                # is stateless per sample, so realize it as a truncated
                # impulse-response FIR: h[k] = the 2-pole's response (run its
                # a/b/c K steps), convolved with the voice's dry source read
                # backward by the playback stride — the SAME stateless-
                # lookback the comb-reverb already uses. INERT when the
                # sample has no armed filter (itCut==127 && itRes==0) so
                # S3M/MOD and unfiltered-IT cost zero. Applied pre-volume
                # like it2play. ON by default (#define IT_FILTER 0 to skip);
                # IT_FILT_TAPS tunes accuracy/cost. Approximation: very high
                # resonance rings longer than K → softer peak vs the oracle.
                _itf_def_anchor = ("#ifndef AA_MAX_TAPS\n#define AA_MAX_TAPS 4"
                                   "\n#endif\n")
                _itf_defs = (_itf_def_anchor
                             + "#ifndef IT_FILTER\n#define IT_FILTER 1\n#endif\n"
                             + "#ifndef IT_FILT_TAPS\n#define IT_FILT_TAPS 32"
                               "\n#endif\n")
                _itf_anchor = "    // ── Anti-click ramps ──"
                _itf_block = (
                    "#if IT_FILTER\n"
                    "    // it2play 2-pole resonant LPF — stateless truncated\n"
                    "    // impulse-response FIR (see generator note). Inert\n"
                    "    // unless this sample carries an armed IT filter.\n"
                    "    if (smp.itCut < 127 || smp.itRes > 0) {\n"
                    "        float _fr = exp2((float(smp.itCut)*255.0) *"
                    " (-1.0/6144.0))\n"
                    "                    * ((1.0/(6.28318530718*110.0*"
                    "1.18920712))*44100.0);\n"
                    "        float _fp = pow(10.0, (-float(smp.itRes)*24.0)"
                    "/2560.0);\n"
                    "        float _fd = _fp*_fr + (_fp-1.0);\n"
                    "        float _fe = _fr*_fr;\n"
                    "        float _fA = 1.0/(1.0+_fd+_fe);\n"
                    "        float _fB = (_fd+_fe+_fe)*_fA;\n"
                    "        float _fC = 1.0 - _fA - _fB;\n"
                    "        float _fstride = freq / (44100.0 * float(max(1,"
                    " smp.bwFactor)));\n"
                    "        float _fy1 = 0.0, _fy2 = 0.0, _facc = 0.0;\n"
                    "        for (int _fk = 0; _fk < IT_FILT_TAPS; _fk++) {\n"
                    "            float _fh = ((_fk == 0) ? _fA : 0.0) +"
                    " _fB*_fy1 + _fC*_fy2;\n"
                    "            _fy2 = _fy1; _fy1 = _fh;\n"
                    "            float _fpp = fSamplePos - float(_fk)*"
                    "_fstride;\n"
                    "            if (_fpp < 0.0) break;\n"
                    "            if (smp.loopLen > 2 && _fpp >="
                    " float(smp.loopStart + smp.loopLen))\n"
                    "                _fpp = float(smp.loopStart)\n"
                    "                     + mod(_fpp - float(smp.loopStart),"
                    " float(smp.loopLen));\n"
                    "            _facc += _fh * getSampleF(smp.start, _fpp,\n"
                    "                          smp.smpLen, smp.loopStart,"
                    " smp.loopLen);\n"
                    "        }\n"
                    "        s = _facc;\n"
                    "    }\n"
                    "#endif\n"
                    + _itf_anchor)
                _itf_dn = _sound_src.count(_itf_def_anchor)
                _itf_an = _sound_src.count(_itf_anchor)
                if _itf_dn >= 1 and _itf_an == 1:
                    _sound_src = _sound_src.replace(
                        _itf_def_anchor, _itf_defs, 1)
                    _sound_src = _sound_src.replace(
                        _itf_anchor, _itf_block, 1)
                    print(f"   ✓ IT resonant filter injected (it2play 2-pole, "
                          f"stateless K=IT_FILT_TAPS FIR; ON by default, "
                          f"#define IT_FILTER 0 to skip; inert for non-filter "
                          f"samples)")
                else:
                    print(f"   ⚠ IT filter skipped (def-anchor×{_itf_dn} "
                          f"apply-anchor×{_itf_an}; expected ≥1,1) — "
                          f"playback unchanged")

                # ── MOD+ : per-sample VOL ENVELOPE in the pattern-player ──
                # The IT-over-MOD feature the HTML player (user's correct
                # oracle) has and the lossy timeline destroyed. _itVolEnv
                # evaluates the instrument vol envelope statelessly as a
                # pure function of elapsed-ticks-since-trigger (sustain-
                # hold while keyed-on — v1 assumes held, the dominant
                # case — + loop + piecewise-linear), reading the compact
                # _itEPt[] pool (Common) keyed by SampleInfo.eOff/eN/eSus/
                # eLp. eN<=0 → returns 1.0 → S3M/MOD/no-env IT byte-
                # identical (zero cost, always safe to inject). Defined
                # right before its only caller _gcoBody.
                _ve_sig  = "float _gcoBody("
                _ve_old  = "return s * (_effVol / 64.0) * declick * endFade;"
                _ve_fn   = (
                    "float _itVolEnv(SampleInfo smp, float etick){\n"
                    "  int n = smp.eN;\n"
                    "  if (n <= 0) return 1.0;\n"
                    "  int o = smp.eOff;\n"
                    "  if (smp.eSus >= 0 && smp.eSus < n){\n"
                    "    float st = float(_itEPt[o+smp.eSus] >> 7);\n"
                    "    if (etick > st) etick = st;\n"
                    "  } else if (smp.eLp >= 0){\n"
                    "    int ls=(smp.eLp>>8)&255, le=smp.eLp&255;\n"
                    "    if (le>ls && le<n){\n"
                    "      float ta=float(_itEPt[o+ls]>>7), tb=float(_itEPt[o+le]>>7);\n"
                    "      if (tb>ta && etick>tb) etick = ta + mod(etick-ta, tb-ta);\n"
                    "    }\n"
                    "  }\n"
                    "  if (etick <= float(_itEPt[o]>>7)) return float(_itEPt[o]&127)/64.0;\n"
                    "  for (int i=1;i<25;i++){\n"
                    "    if (i>=n) break;\n"
                    "    float t1=float(_itEPt[o+i]>>7);\n"
                    "    if (etick <= t1){\n"
                    "      float pt0=float(_itEPt[o+i-1]>>7);\n"
                    "      float pv0=float(_itEPt[o+i-1]&127), pv1=float(_itEPt[o+i]&127);\n"
                    "      float f=(t1>pt0)?(etick-pt0)/(t1-pt0):0.0;\n"
                    "      return (pv0+(pv1-pv0)*f)/64.0;\n"
                    "    }\n"
                    "  }\n"
                    "  return float(_itEPt[o+n-1]&127)/64.0;\n"
                    "}\n")
                # elapsed (sec since trigger) × TICKS_PER_SEC (Common
                # #define, BPM-derived player ticks/sec) = elapsed player
                # ticks — exactly the unit ITFile env_pts ticks are in.
                _ve_new  = ("return s * (_effVol / 64.0) * declick * endFade "
                            "* _itVolEnv(smp, elapsed * TICKS_PER_SEC);")
                # SKIP for IT: the NNA envelope-follower port (below)
                # multiplies s_curr by envValueAt() in getChannelOutput.
                # Applying _itVolEnv to _gcoBody's return TOO double-
                # applies the envelope — voices end up at env² (≪ env
                # for env<1) so sustained NNA ghost tails are squashed
                # to near-silence and the audible NNA effect vanishes.
                # envValueAt is strictly more capable (handles keyOffAge
                # release phase) so it supersedes _itVolEnv for IT.
                # Non-IT keeps _itVolEnv (inert: eN=0 → returns 1.0).
                _ve_ok = (_sound_src.count(_ve_sig) >= 1
                          and _sound_src.count(_ve_old) == 1
                          and not getattr(mod, 'is_it', False))
                if _ve_ok:
                    _sound_src = _sound_src.replace(
                        _ve_sig, _ve_fn + "\n" + _ve_sig, 1)
                    _sound_src = _sound_src.replace(_ve_old, _ve_new, 1)
                    print(f"   ✓ MOD+ vol envelope injected into _gcoBody "
                          f"(stateless f(elapsed-ticks); eN=0 ⇒ inert ⇒ "
                          f"S3M/MOD untouched)")
                elif getattr(mod, 'is_it', False):
                    print(f"   ⋯ MOD+ _itVolEnv skipped (NNA envelope-"
                          f"follower port covers env via envValueAt — "
                          f"avoids env² double-apply that silences ghosts)")
                else:
                    print(f"   ⚠ MOD+ vol-env skipped (sig×"
                          f"{_sound_src.count(_ve_sig)} ret×"
                          f"{_sound_src.count(_ve_old)}; expected ≥1,1) — "
                          f"envelopes not applied")

                # ── Full NNA envelope-follower port from mod_player.py ───
                # Ports envValueAt() + parallel aux arrays (sampleNNA,
                # sampleFadeout, sampleReleaseHold, sampleEnvBaseGain,
                # sampleEnvReleaseDur, envPointsX/Y, sampleEnvOff/Cnt/SusPt)
                # + the three crossfade-overlay regex patches from
                # mod_player.py:15797/15835/16409/16468/16484. Lifts the
                # original 31-instrument caps to len(mod.samples) so IT
                # files with >31 instruments (jeff.it has 99) actually
                # get NNA ghosts. Gated IT-only: S3M/MOD/XM skip the entire
                # port (zero added GPU cost, zero regression by construction).
                # #define NNA_GHOST_MODE 1 enables NNA=2/3 overlay; NNA=1
                # always engages (drum stacking). Provides a real ghost
                # release that follows the prev voice's IT vol envelope
                # past sustain, then fadeouts — vs. the simple sum-with-decay
                # widening (~+0.005 envelope corr) this replaces.
                if getattr(mod, 'is_it', False):
                    import re as _re_nna
                    # PER-INSTRUMENT env arrays (NOT per-sample). IT envelopes
                    # belong to the instrument; ITFile's per-sample copy is
                    # lossy and forces all-or-nothing for shared samples. By
                    # building from inst_table we (a) honor each instrument's
                    # env_on (jeff.it inst 1 env_on=False → no envelope = no
                    # wooump on first note), and (b) get the right env when
                    # multiple instruments share a sample with different
                    # envelopes. _curIns_/instIdx in envValueAt is already
                    # the instrument number (1..99) so no re-indexing needed
                    # at the GLSL site — just feed instrument-indexed arrays.
                    _it_tbl = getattr(mod, '_it_inst_table', None) or {}
                    _nsmp_nna = max(31, min(99, max(99, max(_it_tbl.keys()) if _it_tbl else 99)))
                    # Build array slot i → instrument (i+1). NNA still comes
                    # from the SAMPLE because NNA is a sample property in IT.
                    _xs_smp = list(mod.samples)[:99]
                    while len(_xs_smp) < _nsmp_nna:
                        _xs_smp.append({})
                    _xs_inst = []
                    for i in range(_nsmp_nna):
                        _xs_inst.append(_it_tbl.get(i + 1, {}))
                    def _sd(d): return d if isinstance(d, dict) else {}
                    def _env_on(d):
                        # Instrument envelope active = env_on==True AND points exist.
                        return bool(_sd(d).get('env_on') and _sd(d).get('env_pts'))
                    _fo_v = ', '.join(str(_sd(d).get('fadeout', 0)) for d in _xs_inst)
                    _rh_v = ', '.join(str(int(round(_sd(d).get('release_factor', 1.0) * 64))) for d in _xs_inst)
                    def _ebg(d):
                        # Pre-baked avg gain — only when env_on AND env_pts AND not sus-held.
                        if _env_on(d) and not _sd(d).get('env_sus', False):
                            return int(round(_sd(d).get('release_factor', 1.0) * 64))
                        return 64
                    _eg_v = ', '.join(str(_ebg(d)) for d in _xs_inst)
                    # NNA stays sample-indexed (per IT spec — it's a sample property
                    # not instrument property; the prev-trigger walk gets the
                    # SAMPLE that played, not the instrument).
                    _nna_v = ', '.join(str(_sd(s).get('nna', 0)) for s in _xs_smp)
                    def _erd(d):
                        if not (_env_on(d) and _sd(d).get('env_sus', False)):
                            return 0
                        sp = _sd(d).get('env_sus_pt', 0); pts = _sd(d).get('env_pts') or []
                        if sp < len(pts) and len(pts) > 1:
                            return max(0, pts[-1][0] - pts[sp][0])
                        return 0
                    _rd_v = ', '.join(str(_erd(d)) for d in _xs_inst)
                    _ex_f, _ey_f, _eo_a, _ec_a, _esp_a = [], [], [], [], []
                    _n_inst_env = 0
                    for d in _xs_inst:
                        # GATE: only emit env_pts when env_on=True. env_on=False
                        # → cnt=0 → envValueAt returns 1.0 → no envelope (correct).
                        if _env_on(d):
                            pts = _sd(d).get('env_pts') or []
                            sp = _sd(d).get('env_sus_pt', 0) if _sd(d).get('env_sus', False) else -1
                            _n_inst_env += 1
                        else:
                            pts = []; sp = -1
                        _eo_a.append(len(_ex_f)); _ec_a.append(len(pts)); _esp_a.append(sp)
                        for p in pts:
                            _ex_f.append(int(p[0])); _ey_f.append(int(p[1]))
                    if not _ex_f:
                        _ex_f, _ey_f = [0], [64]
                    _ex_v = ', '.join(str(v) for v in _ex_f)
                    _ey_v = ', '.join(str(v) for v in _ey_f)
                    _eo_v = ', '.join(str(v) for v in _eo_a)
                    _ec_v = ', '.join(str(v) for v in _ec_a)
                    _esp_v = ', '.join(str(v) for v in _esp_a)
                    _et = len(_ex_f)
                    _aux_block = (
                        f"\n// NNA envelope-follower aux arrays (parallel to samples[]).\n"
                        f"const int sampleFadeout[{_nsmp_nna}]      = int[]({_fo_v});\n"
                        f"const int sampleReleaseHold[{_nsmp_nna}]  = int[]({_rh_v});\n"
                        f"const int sampleEnvBaseGain[{_nsmp_nna}]  = int[]({_eg_v});\n"
                        f"const int sampleNNA[{_nsmp_nna}]          = int[]({_nna_v});\n"
                        f"const int sampleEnvReleaseDur[{_nsmp_nna}]= int[]({_rd_v});\n"
                        f"const int envPointsX[{_et}]  = int[]({_ex_v});\n"
                        f"const int envPointsY[{_et}]  = int[]({_ey_v});\n"
                        f"const int sampleEnvOff[{_nsmp_nna}]   = int[]({_eo_v});\n"
                        f"const int sampleEnvCnt[{_nsmp_nna}]   = int[]({_ec_v});\n"
                        f"const int sampleEnvSusPt[{_nsmp_nna}] = int[]({_esp_v});\n"
                        f"\nfloat envValueAt(int instIdx, float envTime, float keyOffAge) {{\n"
                        f"    if (instIdx < 0 || instIdx >= {_nsmp_nna}) return 1.0;\n"
                        f"    int cnt = sampleEnvCnt[instIdx];\n"
                        f"    if (cnt < 2) return 1.0;\n"
                        f"    int off = sampleEnvOff[instIdx];\n"
                        f"    int susPt = sampleEnvSusPt[instIdx];\n"
                        f"    float envX;\n"
                        f"    if (keyOffAge <= 0.0) {{\n"
                        f"        envX = envTime * TICKS_PER_SEC;\n"
                        f"        if (susPt >= 0 && susPt < cnt) {{\n"
                        f"            float susX = float(envPointsX[off + susPt]);\n"
                        f"            if (envX > susX) envX = susX;\n"
                        f"        }}\n"
                        f"    }} else {{\n"
                        f"        if (susPt >= 0 && susPt < cnt) {{\n"
                        f"            envX = float(envPointsX[off + susPt]) + keyOffAge * TICKS_PER_SEC;\n"
                        f"        }} else {{\n"
                        f"            envX = envTime * TICKS_PER_SEC;\n"
                        f"        }}\n"
                        f"    }}\n"
                        f"    float lastX = float(envPointsX[off + cnt - 1]);\n"
                        f"    if (envX >= lastX) return float(envPointsY[off + cnt - 1]) / 64.0;\n"
                        f"    if (envX <= float(envPointsX[off])) return float(envPointsY[off]) / 64.0;\n"
                        f"    for (int i = 0; i < 24; i++) {{\n"
                        f"        if (i + 1 >= cnt) break;\n"
                        f"        float x0 = float(envPointsX[off + i]);\n"
                        f"        float x1 = float(envPointsX[off + i + 1]);\n"
                        f"        if (envX >= x0 && envX <= x1) {{\n"
                        f"            float t = (x1 > x0) ? (envX - x0) / (x1 - x0) : 0.0;\n"
                        f"            return mix(float(envPointsY[off + i]),\n"
                        f"                       float(envPointsY[off + i + 1]), t) / 64.0;\n"
                        f"        }}\n"
                        f"    }}\n"
                        f"    return 1.0;\n"
                        f"}}\n"
                    )
                    # Inject after samples[] in Sound (Note: mod_player puts
                    # in Common (_ct), but we inject in Sound since
                    # envValueAt is only used by getChannelOutput in Sound).
                    # Actually safer: inject in Common so the function
                    # exists when Sound's getChannelOutput compiles.
                    _smp_end = _common_src.find(');', _common_src.find('const SampleInfo samples['))
                    if _smp_end >= 0:
                        _eol = _common_src.find('\n', _smp_end)
                        _common_src = _common_src[:_eol+1] + _aux_block + _common_src[_eol+1:]
                        _aux_ok = True
                    else:
                        _aux_ok = False
                    # #define NNA_GHOST_MODE 1
                    if '#define NNA_GHOST_MODE' not in _common_src:
                        if "#define USE_TIMELINE_DSP 0" in _common_src:
                            _common_src = _common_src.replace(
                                "#define USE_TIMELINE_DSP 0",
                                "#define USE_TIMELINE_DSP 0\n#define NNA_GHOST_MODE 1", 1)
                        else:
                            _common_src = "#define NNA_GHOST_MODE 1\n" + _common_src
                    # Patch 1: NNA-widen the crossfade window.
                    _p1_src = r'if \(ageSamples < 64\.0 && ageSamples >= 0\.0\) \{'
                    _p1_dst = (
                        'float _xfLen = 64.0;\n'
                        '    {\n'
                        '        int _pInst = 0;\n'
                        '        // FIX (May 16 bug recurrence): _r/_p MUST be declared\n'
                        '        // OUTSIDE the for body and DECREMENTED inside (matching\n'
                        '        // getChannelOutput\'s other walk-backs at 3611+/3717+).\n'
                        '        // The original `int _r = trigRow - 1 - _lb;` inside the\n'
                        '        // body re-initialised _r/_p every iteration, so when the\n'
                        '        // walk crossed a pattern boundary (_p--, _r=lastRow), the\n'
                        '        // NEXT iter reset it back to trigPat — the search died\n'
                        '        // after one row of the previous pattern, chopping every\n'
                        '        // NNA voice ringing more than ~1.5 ms.\n'
                        '        int _r = trigRow, _p = trigPat;\n'
                        '        for (int _lb = 0; _lb < 64; _lb++) {\n'
                        '            _r--;\n'
                        '            if (_r < 0) { if (_p > 0) { _p--; _r = patStartRow[_p] + (patRowOffset[_p+1] - patRowOffset[_p]) - 1; } else break; }\n'
                        '            Note _pp = getNote(_p, _r, ch);\n'
                        '            bool _isTone = ((_pp.effect == 0x3 || _pp.effect == 0x5) && _pp.period > 0);\n'
                        '            if (_pp.period > 0 && !_isTone) {\n'
                        '                _pInst = _pp.instrument;\n'
                        '                // FIX: inherited-instrument resolution. In IT,\n'
                        '                // a period-only retrigger has instrument==0 and\n'
                        '                // inherits from a PRIOR row. Without this walk,\n'
                        '                // _pInst==0 → NNA gate fails → ghost cuts after\n'
                        '                // 1.5 ms (user: "stops NNA instead of crossfading\n'
                        '                // gently"). Mirrors getChannelOutput\'s existing\n'
                        '                // instrument-inherit walk at lines 3678-3697.\n'
                        '                if (_pInst == 0) {\n'
                        '                    int _r2 = _r, _p2 = _p;\n'
                        '                    for (int _lb2 = 0; _lb2 < 64; _lb2++) {\n'
                        '                        _r2--;\n'
                        '                        if (_r2 < 0) { if (_p2 > 0) { _p2--; _r2 = patStartRow[_p2] + (patRowOffset[_p2+1] - patRowOffset[_p2]) - 1; } else break; }\n'
                        '                        Note _pp2 = getNote(_p2, _r2, ch);\n'
                        '                        if (_pp2.instrument > 0) { _pInst = _pp2.instrument; break; }\n'
                        '                    }\n'
                        '                }\n'
                        '                break;\n'
                        '            }\n'
                        '        }\n'
                        f'        if (_pInst >= 1 && _pInst <= {_nsmp_nna}) {{\n'
                        '            int _nna = sampleNNA[_pInst - 1];\n'
                        '            int _relTicks = sampleEnvReleaseDur[_pInst - 1];\n'
                        '            int _foAmt    = sampleFadeout[_pInst - 1];\n'
                        '            float _foSamp = (_foAmt > 0)\n'
                        '                ? (32768.0 / float(_foAmt)) * (44100.0 / TICKS_PER_SEC)\n'
                        '                : 220500.0;\n'
                        '            float _envSamp = (_relTicks > 0)\n'
                        '                ? float(_relTicks) * (44100.0 / TICKS_PER_SEC)\n'
                        '                : 220500.0;\n'
                        '            if (_nna == 1) {\n'
                        '                _xfLen = (_relTicks > 0) ? _envSamp : 220500.0;\n'
                        '            }\n'
                        '#if NNA_GHOST_MODE\n'
                        '            else if (_nna == 2) {\n'
                        '                _xfLen = (_relTicks > 0) ? _envSamp : _foSamp;\n'
                        '            } else if (_nna == 3) {\n'
                        '                _xfLen = min(_envSamp, _foSamp);\n'
                        '            }\n'
                        '#endif\n'
                        '        }\n'
                        '    }\n'
                        '    if (ageSamples < _xfLen && ageSamples >= 0.0) {'
                    )
                    _sound_src, _np1 = _re_nna.subn(_p1_src, _p1_dst, _sound_src, count=1)
                    # Patch 2: attack ramp on s_curr via envValueAt.
                    _p2_src = r'(float s_curr = _gcoBody\([^;]+\);\s*\n)'
                    _p2_dst = (
                        r'\1'
                        '    {\n'
                        '        int _curIns_ = trigNote.instrument;\n'
                        f'        if (_curIns_ >= 1 && _curIns_ <= {_nsmp_nna}) {{\n'
                        '            float _trigTimeF_ = float(fetchTick(patTickOffset[trigPat]+(trigRow-patStartRow[trigPat]))) / TICKS_PER_SEC;\n'
                        '            s_curr *= envValueAt(_curIns_ - 1, time - _trigTimeF_, 0.0);\n'
                        '        }\n'
                        '    }\n'
                    )
                    _sound_src, _np2 = _re_nna.subn(_p2_src, _p2_dst, _sound_src, count=1)
                    # Patch 3: ghost overlay (NNA blend, replaces crossfade).
                    _p3_src = r'float t = ageSamples / 64\.0;\s*\n\s*return s_prev \* \(1\.0 - t\) \+ s_curr \* t;'
                    _p3_dst = (
                        '            if (_xfLen > 64.5) {\n'
                        '                int _pIns2 = pTrigNote.instrument;\n'
                        f'                if (_pIns2 >= 1 && _pIns2 <= {_nsmp_nna}) {{\n'
                        '                    float _pTrigTime = float(fetchTick(patTickOffset[pTrigPat]+(pTrigRow-patStartRow[pTrigPat]))) / TICKS_PER_SEC;\n'
                        '                    float _curTrigTime = float(fetchTick(patTickOffset[trigPat]+(trigRow-patStartRow[trigPat]))) / TICKS_PER_SEC;\n'
                        '                    float _koAge = (sampleNNA[_pIns2 - 1] == 1) ? 0.0 : max(0.0, time - _curTrigTime);\n'
                        '                    s_prev *= envValueAt(_pIns2 - 1, time - _pTrigTime, _koAge);\n'
                        '                }\n'
                        '                return s_curr + s_prev;\n'
                        '            }\n'
                        '            float t = clamp(ageSamples / 64.0, 0.0, 1.0);\n'
                        '            return s_prev * (1.0 - t) + s_curr * t;'
                    )
                    _sound_src, _np3 = _re_nna.subn(_p3_src, _p3_dst, _sound_src, count=1)
                    print(f"   ✓ NNA envelope-follower port: aux_arrays={'OK' if _aux_ok else 'FAIL'}, "
                          f"patches=({_np1},{_np2},{_np3}) (expect 1,1,1), "
                          f"cap=31→{_nsmp_nna}, {_n_inst_env}/{_nsmp_nna} insts have env_on "
                          f"({_et} pts; sample-borrow bug FIXED — was 'wooump' on env_on=False insts)")
                    # ── REVERTED: Per-channel ghost-slot table (proper NNA architecture) ──
                    # This was triple-stacking (MOD+ live + crossfade overlay + ghost slots)
                    # and made jeff.it WORSE. Reverting until: (a) the existing crossfade
                    # overlay is dropped for IT so we don't double-render the prev voice,
                    # (b) ghost samp_pos/freq/inst-lookup is verified individually before
                    # stacking, and (c) the wooump root cause is properly verified (not
                    # just claimed). Slow down. Diagnose first, code second.
                    _disabled_ghost_block = """
                    # User: "modify MOD+ to handle ghost slots properly — pre-scan,
                    # allocate accordingly". ITPlayer already simulates NNA correctly
                    # and emits a VoiceSegment per voice trigger (3047 segs for jeff.it).
                    # Extract just the GHOST INTERVALS: for each channel sorted by
                    # start_tick, segment[i] is LIVE until segment[i+1] starts; after
                    # that, if seg.end_tick > supersede_tick, the interval
                    # [supersede_tick, end_tick] is a ghost. Capture the voice's
                    # state at supersede_tick (sample_pos, freq, vol, pan from
                    # tick_states) + carry instrument & NNA mode for envelope-
                    # following at GLSL replay time. jeff.it: 643 ghost events,
                    # peak 24 concurrent, ~20KB packed. Sort by (channel, start_tick)
                    # so getChannelOutput can iterate just its channel's range.
                    try:
                        _gh_pl = _mp_it.load_it_native(args.modfile)
                        _gh_segs = _gh_pl.run()
                        _gh_tps  = _gh_pl.initial_tempo * 2.0 / 5.0
                        import collections as _gh_coll
                        _gh_by_ch = _gh_coll.defaultdict(list)
                        for _s in _gh_segs:
                            if _s.tick_states:
                                _gh_by_ch[getattr(_s,'channel',-1)].append(_s)
                        _ghosts = []  # list of dicts
                        for _ch in range(mod.num_channels):
                            _cs = sorted(_gh_by_ch[_ch], key=lambda s: s.tick_states[0][0])
                            for _i, _s in enumerate(_cs):
                                _s_end = (_s.end_tick if _s.end_tick >= 0
                                          else _s.tick_states[-1][0] + 1)
                                _live_until = (_cs[_i+1].tick_states[0][0]
                                               if _i+1 < len(_cs) else _s_end)
                                if _s_end > _live_until:
                                    # Capture state at supersede_tick
                                    _state = _s.tick_states[0]
                                    for _ts in _s.tick_states:
                                        if _ts[0] <= _live_until: _state = _ts
                                        else: break
                                    _t, _f, _v, _p, _sp = _state
                                    # Find instrument: scan all instruments that
                                    # map to this sample at the played note. For
                                    # jeff.it instrument-mode IT, this requires
                                    # the inst_table — but for the ghost render we
                                    # just need ONE instrument index whose env
                                    # corresponds. ITPlayer tracks this per-seg
                                    # but doesn't expose; approximate with sample
                                    # index + 1 (works when inst==sample, which
                                    # is the common case after my per-inst env
                                    # arrays — for the 34 mismatches we lose a
                                    # bit of envelope fidelity on ghosts).
                                    _g_inst = _s.sample_idx + 1
                                    # NNA mode from sample dict
                                    _smp_d = (mod.samples[_s.sample_idx]
                                              if _s.sample_idx < len(mod.samples)
                                                 and isinstance(mod.samples[_s.sample_idx], dict)
                                              else {})
                                    _g_nna = int(_smp_d.get('nna', 0))
                                    _ghosts.append({
                                        'ch': _ch,
                                        'start': max(0, min(0xFFFF, _live_until)),
                                        'dur':   max(1, min(0xFFFF, _s_end - _live_until)),
                                        'sample': max(0, min(0x7F, _s.sample_idx)),
                                        'samp_pos': max(0, min(0x3FFFFF, int(_sp))),
                                        'freq':   float(_f),
                                        'vol':    max(0, min(255, int(round(_v * 255)))),
                                        'pan':    max(0, min(255, int(round(_p * 255)))),
                                        'inst':   max(0, min(0xFF, _g_inst)),
                                        'nna':    max(0, min(3, _g_nna)),
                                    })
                        # Sort by (channel, start) for per-channel index range
                        _ghosts.sort(key=lambda g: (g['ch'], g['start']))
                        _NCH = mod.num_channels
                        _ghostByCh = [0] * (_NCH + 1)
                        for _g in _ghosts:
                            _ghostByCh[_g['ch'] + 1] += 1
                        for _i in range(1, _NCH + 1):
                            _ghostByCh[_i] += _ghostByCh[_i-1]
                        # Pack ghosts as ivec4 (16 bytes each). Spec:
                        #  .x = start_tick(16) | dur(16)
                        #  .y = samp_pos(22) | sample(7) | (1 reserved)
                        #  .z = floatBitsToInt(freq) — lossless
                        #  .w = vol(8) | pan(8) | inst(8) | nna(2) | ch(5) | (1 reserved)
                        import struct as _gh_struct
                        def _f2i(f):
                            return _gh_struct.unpack('<i', _gh_struct.pack('<f', float(f)))[0]
                        def _i32(v):
                            v &= 0xFFFFFFFF
                            return v - 0x100000000 if v & 0x80000000 else v
                        _gh_strs = []
                        for _g in _ghosts:
                            _x = _i32(_g['start'] | (_g['dur'] << 16))
                            _y = _i32(_g['samp_pos'] | (_g['sample'] << 22))
                            _z = _f2i(_g['freq'])
                            _w = _i32(_g['vol'] | (_g['pan'] << 8) | (_g['inst'] << 16)
                                      | (_g['nna'] << 24) | (_g['ch'] << 26))
                            _gh_strs.append(f"ivec4({_x},{_y},{_z},{_w})")
                        if not _gh_strs:
                            _gh_strs = ["ivec4(0,0,0,0)"]
                        _NGH = len(_ghosts)
                        _gh_decl = (
                            f"\n// ── NNA ghost-slot table (proper polyphonic NNA) ──\n"
                            f"// Pre-scanned at build time from ITPlayer VoiceSegments.\n"
                            f"// jeff.it: {_NGH} ghosts, peak 24 concurrent, ~20KB.\n"
                            f"// Each ghost: ch + start_tick/duration + voice state at\n"
                            f"// supersede moment + instrument for envelope-following.\n"
                            f"#define NGHOSTS {_NGH}\n"
                            f"#define GHOST_TPS {_gh_tps:.4f}\n"
                            f"const ivec4 ghostI[{_NGH}] = ivec4[](\n  "
                                + ',\n  '.join(_gh_strs) + ");\n"
                            f"const int ghostByCh[{_NCH+1}] = int[]("
                                + ','.join(str(v) for v in _ghostByCh) + ");\n"
                        )
                        # Inject into Common right after the env aux arrays.
                        _ga_marker = "float envValueAt("
                        if _ga_marker in _common_src:
                            _common_src = _common_src.replace(
                                _ga_marker, _gh_decl + _ga_marker, 1)
                            _gh_inj = True
                        else:
                            _gh_inj = False
                        # Inject the ghost-render loop into getChannelOutput, replacing
                        # the final `return s_curr;` with sum of s_curr + active ghosts.
                        # Place AFTER the existing crossfade-overlay branch (patch3)
                        # but before the function's terminal `return s_curr;`.
                        _gho_src = "return s_curr;\n}"
                        _gho_dst = (
                            "    // ── Sum active ghost voices on this channel ──\n"
                            "    {\n"
                            "        int _tT = int(time * GHOST_TPS);\n"
                            "        int _gi0 = ghostByCh[ch];\n"
                            "        int _gi1 = ghostByCh[ch+1];\n"
                            "        for (int _gi = _gi0; _gi < _gi1; _gi++) {\n"
                            "            ivec4 _g = ghostI[_gi];\n"
                            "            int _gs = _g.x & 0xFFFF;\n"
                            "            int _gd = (_g.x >> 16) & 0xFFFF;\n"
                            "            if (_tT < _gs) break;            // sorted; future\n"
                            "            if (_tT >= _gs + _gd) continue;  // expired\n"
                            "            int _gsp0  = _g.y & 0x3FFFFF;\n"
                            "            int _gsmi  = (_g.y >> 22) & 0x7F;\n"
                            "            float _gfq = intBitsToFloat(_g.z);\n"
                            "            float _gvol = float(_g.w & 0xFF) / 255.0;\n"
                            "            float _gpan = float((_g.w >> 8) & 0xFF) / 255.0;\n"
                            "            int _gins = (_g.w >> 16) & 0xFF;\n"
                            "            int _gnna = (_g.w >> 24) & 0x3;\n"
                            "            float _gAge = (time - float(_gs) / GHOST_TPS);\n"
                            "            if (_gAge < 0.0) continue;\n"
                            "            float _gfpos = float(_gsp0) + _gAge * _gfq;\n"
                            f"            SampleInfo _gsmp = samples[clamp(_gsmi, 0, {_nsmp_nna - 1})];\n"
                            "            int _gls = _gsmp.loopStart, _gle = _gsmp.loopStart + _gsmp.loopLen;\n"
                            "            bool _glp = (_gsmp.loopLen > 0);\n"
                            "            if (_glp && _gfpos >= float(_gle))\n"
                            "                _gfpos = float(_gls) + mod(_gfpos - float(_gls), float(_gle - _gls));\n"
                            "            if (!_glp && _gfpos >= float(_gsmp.sampLen)) continue;\n"
                            "            float _gs_smp = getSampleF(_gsmp.start, _gfpos, _gsmp.sampLen, _gls, _glp ? _gle - _gls : 0);\n"
                            "            // Envelope on ghost: NNA=1 stays at sustain, NNA=2/3 releases.\n"
                            "            float _gko = (_gnna == 1) ? 0.0 : _gAge;\n"
                            "            float _genv = envValueAt(_gins - 1, _gAge, _gko);\n"
                            "            s_curr += _gs_smp * _gvol * _genv;\n"
                            "        }\n"
                            "    }\n"
                            "    return s_curr;\n}"
                        )
                        _n_gho = _sound_src.count(_gho_src)
                        if _n_gho >= 1:
                            _sound_src = _sound_src.replace(_gho_src, _gho_dst, 1)
                            _gh_sub_ok = True
                        else:
                            _gh_sub_ok = False
                        print(f"   ✓ Ghost-slot table: {_NGH} ghosts ({_NGH*16/1024:.1f}KB), "
                              f"sort=(ch,start), inject_common={_gh_inj}, "
                              f"inject_loop={_gh_sub_ok} (peak 24 concurrent → ~2 iter avg)")
                    except Exception as _gh_err:
                        print(f"   ⚠ Ghost-slot table failed: {_gh_err}")
                    """  # end disabled ghost-slot block
                else:
                    print(f"   ⋯ NNA envelope-follower port skipped (not IT — "
                          f"S3M/MOD/XM stay byte-identical, no GPU cost)")

                # ── IT NNA timeline graft (mod_player.py's refined engine) ─
                # mod_player.py's 64-voice ITPlayer baked NNA/DCT/DCA into a
                # VoiceSegment list (computed in the IT branch via reuse).
                # Inject the packed tlSegI/tlLoop/tlSlide arrays + the
                # SELF-CONTAINED tlGetOutput(T) (verbatim from mod_player.py,
                # only needs getSampleF + samples[], both present) after
                # getSampleF, gated #if USE_TIMELINE_DSP. The dry-mix swap is
                # a separate patch. Inert unless this is an IT build whose
                # NNA sim succeeded (mod._it_timeline_glsl set).
                # USE_TIMELINE_DSP must be DEFINED in EVERY build — GLSL ES
                # forbids an undefined macro inside #if (desktop GLSL allows
                # it, ShaderToy/ANGLE does NOT). It must live in COMMON: it
                # is the universally-prepended prelude, so a Common #define
                # covers BOTH Common's own `#if !USE_TIMELINE_DSP` (the
                # dead-pattern strip) AND Sound's `#if USE_TIMELINE_DSP`
                # (mainSound swap + tlGetOutput). Default 0 = pattern path;
                # the IT branch flips it 0→1. Anchor on USE_EMBEDDED_DATA
                # (Common, very early — before the first pattern array).
                if ("#define USE_TIMELINE_DSP" not in _common_src
                        and "#define USE_EMBEDDED_DATA" in _common_src):
                    _common_src = _common_src.replace(
                        "#define USE_EMBEDDED_DATA",
                        "#define USE_TIMELINE_DSP 0\n#define USE_EMBEDDED_DATA",
                        1)
                _tlg = getattr(mod, '_it_timeline_glsl', None)
                if _tlg:
                    # Bake the SampleInfo array size (mirrors the rebuild's
                    # _n_smp): the shipped code uses samples.length(), but
                    # sound_exec's `.length`→`.sampLen` glslang-compat rename
                    # mangles the array .length() method → bake a constant.
                    _tl_nsmp = max(31, len(getattr(mod, 'samples', []) or []))
                    _TLGETOUTPUT_GLSL = (
                        "\n// ── Timeline: sum active voice segments at T (NNA"
                        " baked in by mod_player.py ITPlayer) ──\n"
                        "#if USE_TIMELINE_DSP\n"
                        "vec2 tlGetOutput(float T) {\n"
                        "    vec2 out_lr = vec2(0.0);\n"
                        "    int tick_T = int(T * TL_TICKS_PER_SEC);\n"
                        "    int _nseg = TL_NUM_SEGS;\n"
                        "    for (int i = 0; i < _nseg; i++) {\n"
                        "        ivec4 ip = tlSegI[i];\n"
                        "        int start_tick = ip.x & 0xFFFF;\n"
                        "        int end_tick   = start_tick + ((ip.x >> 16) &"
                        " 0xFFFF);\n"
                        "        if (start_tick > tick_T) break;\n"
                        "        if (tick_T >= end_tick) continue;\n"
                        "        int sp0   = ip.y & 0x3FFFFF;\n"
                        "        int smIdx = (ip.y >> 22) & 0x7F;\n"
                        "        ivec4 _L = tlLoop[smIdx];\n"
                        "        int ls = _L.x, le = _L.y, lt = _L.z;\n"
                        "        float freq0 = intBitsToFloat(ip.z);\n"
                        "        float _vol0 = float(ip.w & 0xFF) / 255.0;\n"
                        "        float _pan  = float((ip.w >> 8) & 0xFF) /"
                        " 255.0;\n"
                        "        int _si = (ip.w >> 16) & 0xFFFF;\n"
                        "        float _fmul = 1.0, _vdelta = 0.0;\n"
                        "        if (_si > 0) {\n"
                        "            ivec4 _sv = tlSlide[(_si - 1) >> 1];\n"
                        "            if (((_si - 1) & 1) == 0) { _fmul ="
                        " intBitsToFloat(_sv.x); _vdelta ="
                        " intBitsToFloat(_sv.y); }\n"
                        "            else                     { _fmul ="
                        " intBitsToFloat(_sv.z); _vdelta ="
                        " intBitsToFloat(_sv.w); }\n"
                        "        }\n"
                        "        float seg_t0 = float(start_tick) /"
                        " TL_TICKS_PER_SEC;\n"
                        "        float dt = T - seg_t0;\n"
                        "        if (dt < 0.0) continue;\n"
                        "        float freq = freq0 * pow(_fmul, dt *"
                        " TL_TICKS_PER_SEC);\n"
                        "        float vol  = clamp(_vol0 + _vdelta * dt *"
                        " TL_TICKS_PER_SEC, 0.0, 1.0);\n"
                        "        float fpos = float(sp0) + dt * freq;\n"
                        "        if (lt > 0 && le > ls && fpos >= float(le))"
                        " {\n"
                        "            float span = float(le - ls);\n"
                        "            fpos = float(ls) + mod(fpos - float(ls),"
                        " span);\n"
                        "        }\n"
                        "        SampleInfo smp = samples[clamp(smIdx, 0, "
                        + str(_tl_nsmp - 1) + ")];\n"
                        "        bool _looping = (lt > 0 && le > ls);\n"
                        "        if (!_looping && (fpos >= float(smp.smpLen)"
                        " || fpos < 0.0)) continue;\n"
                        "        float s = getSampleF(smp.start, fpos,"
                        " smp.smpLen, ls, le > ls ? le - ls : 0);\n"
                        "        float panR = 0.125 + 0.75 * _pan;\n"
                        "        out_lr += s * vol * vec2(1.0 - panR,"
                        " panR);\n"
                        "    }\n"
                        "    return out_lr;\n"
                        "}\n"
                        "#endif // USE_TIMELINE_DSP\n")
                    _tl_block = _tlg + _TLGETOUTPUT_GLSL
                    _tl_anchor = "#ifndef AA_RESAMPLE\n"
                    # The define lives in COMMON now; flip it there. The
                    # tlGetOutput fn + segment arrays still go into SOUND
                    # (it's a Sound function), at the AA anchor.
                    _flip = _common_src.count("#define USE_TIMELINE_DSP 0")
                    if _sound_src.count(_tl_anchor) >= 1 and _flip >= 1:
                        _common_src = _common_src.replace(
                            "#define USE_TIMELINE_DSP 0",
                            "#define USE_TIMELINE_DSP 1", 1)
                        _sound_src = _sound_src.replace(
                            _tl_anchor, _tl_block + _tl_anchor, 1)
                        print(f"   ✓ NNA timeline injected (Common define "
                              f"0→1, Sound tlGetOutput + packed segs; "
                              f"{len(_tlg):,} B arrays)")
                    else:
                        print(f"   ⚠ NNA timeline NOT injected (AA anchor×"
                              f"{_sound_src.count(_tl_anchor)} Common-define×"
                              f"{_flip}) — pattern-player path kept")

                # ── Strip DEAD pattern data under the NNA timeline.
                # tlGetOutput never decodes patterns, so patDict0/patBitmap0
                # /patIdx0-2/patRowSeek0 (~95 KB — the bulk of the ANGLE-
                # capped Common, the jeff.it "doesn't fit" cause) are pure
                # dead weight. Rather than #if-wrap (the decoder region is
                # interleaved with pre-existing #if USE_EMBEDDED_DATA blocks
                # → unbalanced directives), just EMPTY the array LITERALS to
                # a 1-element zero placeholder. The decoder fns (getNote/
                # _gcoBody/getChannelOutput) stay as valid DEAD code (never
                # called under timeline — mainSound uses tlGetOutput; if
                # velvet/getMixedMono do touch them they read 0 → silence,
                # self-consistent). No #if added → zero balance risk; the
                # 95 KB of literal text is physically gone (real size win).
                # rowStartTick0 / songPositions / samples[] are UNTOUCHED
                # (getPosition + tlGetOutput need them). S3M/MOD never reach
                # this branch (gated on the IT timeline) → zero risk.
                if _tlg and "#define USE_TIMELINE_DSP 1" in _common_src:
                    import re as _re_ds
                    _ds_saved = 0; _ds_hit = []
                    for _an in ("patDict0", "patBitmap0", "patIdx0",
                                "patIdx1", "patIdx2", "patRowSeek0"):
                        # const <T> <name>[<N>] = <T>[]( …no ';' until end… );
                        _rx = (r'const\s+(\w+)\s+' + _an +
                               r'\s*\[\s*\d+\s*\]\s*=\s*\1\s*\[\]\s*\([^;]*\)\s*;')
                        _m = _re_ds.search(_rx, _common_src)
                        if _m:
                            _ty = _m.group(1)
                            _new = (f"const {_ty} {_an}[1] = "
                                    f"{_ty}[]({_ty}(0));")
                            _ds_saved += (_m.end() - _m.start()) - len(_new)
                            _common_src = (_common_src[:_m.start()] + _new
                                           + _common_src[_m.end():])
                            _ds_hit.append(_an)
                    if _ds_hit:
                        print(f"   ✂️  Dead pattern arrays emptied under "
                              f"timeline ({'+'.join(_ds_hit)}; "
                              f"~{_ds_saved//1024} KB off Common).")
                    else:
                        print(f"   ⚠ dead-pattern strip: no pattern arrays "
                              f"matched — build valid, just larger")

                    # ── GUT the dead pattern-player FUNCTION BODIES too.
                    # Keeping them as "inert dead-code" still pays their
                    # full LOCAL-variable cost — ANGLE/D3D counts every
                    # declared private var even in never-called functions,
                    # and _gcoBody alone is ~860 lines of locals → the
                    # "Total size of declared private variables exceeds
                    # implementation-defined limit" GPU error (glslang/the
                    # CPU executor do NOT enforce this D3D limit, so it only
                    # shows on the user's ANGLE GPU). Replace each dead
                    # function's BODY with a trivial stub via brace-count
                    # (keeps the signature so anything still referencing it
                    # compiles; under timeline nothing calls them — mainSound
                    # is tlGetOutput; velvet→getMixedMono stub→0 = silence,
                    # self-consistent). No #if → no balance risk.
                    def _ds_gut(src, sig, ret):
                        i = src.find(sig)
                        if i < 0:
                            return src, False
                        b = src.find('{', i)
                        if b < 0:
                            return src, False
                        d = 0; k = b
                        while k < len(src):
                            c = src[k]
                            if c == '{':
                                d += 1
                            elif c == '}':
                                d -= 1
                                if d == 0:
                                    break
                            k += 1
                        if d != 0:
                            return src, False
                        return src[:b] + "{ " + ret + " }" + src[k+1:], True
                    # ONLY the two pattern-player giants — they hold the
                    # bulk of the private-var cost AND are pattern-EXCLUSIVE
                    # (never on the sample-decode/timeline path). Do NOT gut
                    # _extractByte / fetch*Byte / getNote: _extractByte is
                    # the SHARED ivec4 byte-unpacker that getSample (the
                    # live sample decoder) also uses — gutting it returned
                    # 0 for every sample → total silence. Those small fns'
                    # locals are negligible vs _gcoBody's ~860 lines.
                    _gut_ok = []
                    for _sg, _rt in (("float _gcoBody(", "return 0.0;"),
                                     ("float getChannelOutput(",
                                      "return 0.0;")):
                        _sound_src, _g = _ds_gut(_sound_src, _sg, _rt)
                        if _g:
                            _gut_ok.append(_sg.split()[1].rstrip('('))
                    if _gut_ok:
                        print(f"   ✂️  Dead pattern fns gutted "
                              f"({'+'.join(_gut_ok)}) — frees the private-"
                              f"variable budget (the ANGLE GPU limit). "
                              f"_gcoBody (~860 lines of locals) was the hog.")

                # ── enable3D / enableFAT : const bool → #define + #if ─────
                # The b64 encoder bakes `const bool enableX = ...;` and
                # `if (enableX) { ... }`. A `const bool` false-branch can
                # still cost ANGLE private-var budget (inlined functions /
                # locals inside the dead block are accounted). Converting
                # to `#define ENABLE_X 0/1` + `#if` makes the GLSL
                # preprocessor PHYSICALLY delete the disabled feature's
                # code, shrinking the source/private-var footprint so big
                # builds (e.g. 2ND_PM ds1) can stay embedded. (User ask.)
                def _flagdef(src, sym, macro):
                    # Aligned column so the flag block reads cleanly, e.g.
                    #   #define ENABLE_3D        0
                    #   #define ENABLE_FAT       1
                    # NOTE: pad to 16 but ALWAYS keep ≥1 space — macros longer
                    # than 16 chars (ENABLE_VELVETREVERB) would otherwise glue
                    # name+value (`ENABLE_VELVETREVERB0`) → undefined-macro ES
                    # error at the matching `#if`.
                    src = _re3.sub(
                        r'const bool\s+%s\s*=\s*(true|false)\s*;' % sym,
                        lambda m: '#define %-15s %d' % (macro, 1 if m.group(1)=='true' else 0),
                        src)
                    key = 'if (%s) {' % sym
                    out = []; i = 0
                    while True:
                        j = src.find(key, i)
                        if j < 0:
                            out.append(src[i:]); break
                        out.append(src[i:j])
                        b = src.find('{', j); d = 0; k = b
                        while k < len(src):
                            c = src[k]
                            if c == '{': d += 1
                            elif c == '}':
                                d -= 1
                                if d == 0: break
                            k += 1
                        out.append('#if %s\n    %s\n#endif' % (macro, src[b:k+1]))
                        i = k + 1
                    return ''.join(out)
                for _sym, _mac in (('enable3D','ENABLE_3D'), ('enableFAT','ENABLE_FAT'),
                                   ('enablePhatBass','ENABLE_PHATBASS'),
                                   ('enableVelvetReverb','ENABLE_VELVETREVERB'),
                                   ('enableCombReverb','ENABLE_COMBREVERB')):
                    _common_src = _flagdef(_common_src, _sym, _mac)
                    _sound_src  = _flagdef(_sound_src,  _sym, _mac)
                print("   ✓ enable3D/enableFAT → #define + #if "
                      "(disabled features removed by preprocessor, not just dead-stripped)")

                # ── --preserve / --raw-perc: store instruments RAW (no VQ) ─
                # getSample(idx) intercepts each preserved instrument's
                # index range and reads a raw int8 array instead of VQ-
                # decoding → pristine lead / drums.  CRITICAL: the raw PCM is
                # stored FULL-RATE and the instrument's SampleInfo is rewritten
                # to a fresh non-colliding base with bwFactor=1 + full-rate
                # length/loop, so the voice phase (fSamplePos = elapsed*freq/
                # bwFactor) walks it at full resolution.  It is NOT resampled
                # down to the --downsample-decimated VQ length — "raw" means
                # raw; --downsample must never touch preserved/raw-perc.
                _presv_ids = sorted({int(x) for x in
                    str(getattr(args, 'preserve', '') or '').replace(' ', '').split(',')
                    if x.strip().isdigit()})
                # --raw-perc: auto-add percussion (NOISE-classified: kick/
                # snare/hat/clap/cymbal) to the raw/un-VQ'd set. Percussion
                # transients+noise are the worst-hit by RVQ; storing them raw
                # keeps drums crisp & HTML-matching. Reuses the --preserve
                # machinery below verbatim — just feeds it more indices.
                if getattr(args, 'raw_perc', True):
                    # Detect percussion by NAME (classifier==NOISE) OR by
                    # CONTENT — ported from mod_player.py's _bright(): the
                    # fraction of FFT energy above 2500 Hz.  Scene S3Ms
                    # (2ND_PM) put credits/greetings in sample-name fields,
                    # so name-match finds nothing; the spectral test catches
                    # the unnamed hats/snares/noise that names miss.  A
                    # one-shot (non-looped) short-ish sample with lots of HF
                    # energy is percussion (hat/snare/clap/cymbal); kicks are
                    # short one-shots with a sharp transient.
                    import numpy as _np_rp
                    def _rp_bright(_arr):
                        try:
                            _x = _np_rp.asarray(_arr, dtype=_np_rp.float64)
                            if _x.size < 32:
                                return 0.0
                            _x = _x - _x.mean()
                            _X = _np_rp.abs(_np_rp.fft.rfft(
                                _x * _np_rp.hanning(_x.size)))
                            _fr = _np_rp.fft.rfftfreq(_x.size, 1.0 / 8363.0)
                            return float(_X[_fr > 2500.0].sum()
                                         / (_X.sum() + 1e-9))
                        except Exception:
                            return 0.0
                    # SHORT is the key discriminator: kick/snare/hat/clap
                    # one-shots are brief.  Scene S3M sample-names are
                    # credits/scrolltext (useless), and a LONG bright sample
                    # is a melodic lead/pad, NOT a drum (pod s1=29490 /
                    # s18=39986 were bright false-positives → 104 KB bloat).
                    _PERC_MAX_LEN = 8000     # > this ⇒ melodic, never perc
                    _PERC_BR      = 0.30     # HF-energy fraction (hat/snare)
                    # User-tunable fit dial: lower it when full raw-perc's
                    # _presvPCM const array overflows the GPU (jeff.it).
                    _PERC_BUDGET  = max(0, int(getattr(args,
                                       'raw_perc_budget', 28672)))
                    _cand = []               # (length, 1-based idx)
                    for _i, _s in enumerate(mod.samples):
                        if not (isinstance(_s, dict)
                                and _s.get('length', 0) > 0):
                            continue
                        if _s.get('repeat_length', 0) > 2:
                            continue          # looped = melodic, never perc
                        _ln = _s.get('length', 0)
                        _nm = _classify_mod_sample_waveform(
                            _s.get('name', '')) == 4
                        _br = _rp_bright(_s.get('data'))
                        if _nm or (_ln <= _PERC_MAX_LEN and _br > _PERC_BR):
                            _cand.append((_ln, _i + 1))
                    # Budget guard (mod_player.py: default-on can never bloat
                    # / break a build).  Keep the SHORTEST candidates (the
                    # tightest, most byte-efficient real drums) until the
                    # raw-byte budget is spent; the rest stay VQ-compressed.
                    _cand.sort()
                    _perc = []; _acc = 0; _drop = 0
                    for _ln, _idx in _cand:
                        if _acc + _ln <= _PERC_BUDGET:
                            _perc.append(_idx); _acc += _ln
                        else:
                            _drop += 1
                    _perc = sorted(_perc)
                    if _perc:
                        _pn_new = len(set(_perc) - set(_presv_ids))
                        _presv_ids = sorted(set(_presv_ids) | set(_perc))
                        print(f"   ✓ --raw-perc: {len(_perc)} percussion "
                              f"sample(s) RAW {_perc} (~{_acc//1024}KB"
                              + (f", {_drop} over-budget→VQ" if _drop else "")
                              + f", +{_pn_new} beyond --preserve)")
                if _presv_ids:
                    _msi = _re3.search(
                        r'const SampleInfo samples\[\d+\] = SampleInfo\[\]\((.*?)\);',
                        _common_src, _re3.DOTALL)
                    _si = ([[int(v) for v in e.split(',')]
                            for e in _re3.findall(r'SampleInfo\(([^)]*)\)', _msi.group(1))]
                           if _msi else [])
                    # Parallel list of the *literal* SampleInfo(...) strings so a
                    # preserved instrument's entry can be rewritten in place.
                    _si_txt = (_re3.findall(r'SampleInfo\([^)]*\)', _msi.group(1))
                               if _msi else [])
                    # Fresh address space for full-rate preserved PCM, placed
                    # safely past the largest VQ-stream byte index so the
                    # getSample() hook range can never collide with a real VQ
                    # sample (or another preserved one — they're laid out
                    # cumulatively from this base).
                    _PRESV_BASE = (max((_e[0] + _e[1]) for _e in _si
                                       if len(_e) >= 2) + 1024) if _si else (1 << 26)
                    _ppcm = []; _prows = []; _prun = 0; _si_dirty = False
                    for _inst in _presv_ids:
                        if _inst < 1 or _inst > len(_si):
                            continue
                        _src = (mod.samples[_inst-1].get('data')
                                if (_inst-1) < len(mod.samples)
                                and isinstance(mod.samples[_inst-1], dict) else None)
                        if _src is None or len(_src) == 0:
                            print(f"   (--preserve: inst {_inst} skipped — "
                                  f"no data)")
                            continue
                        # FULL-RATE PCM — no np.interp to the decimated VQ
                        # length.  --downsample must not touch raw/preserved.
                        _a = np.asarray(_src)
                        if _a.dtype == np.uint8:
                            _a = _a.astype(np.int16) - 128
                        _a = np.clip(np.round(_a.astype(np.float64)),
                                     -128, 127).astype(np.int8)
                        _full = int(len(_a))
                        if _full <= 0:
                            continue
                        # Raw (un-decimated) loop bounds, clamped to full len.
                        _msd = mod.samples[_inst-1]
                        _rp = int(_msd.get('repeat_point', 0) or 0)
                        _rl = int(_msd.get('repeat_length', 0) or 0)
                        _rp = min(max(0, _rp), _full)
                        if _rp + _rl > _full:
                            _rl = max(0, _full - _rp)
                        # Rewrite this instrument's SampleInfo: new base,
                        # full-rate length/loop, bwFactor=1 (→ phase walks
                        # full resolution). Keep volume/finetune/c2sp.
                        _old = _si[_inst-1]
                        _vol = _old[4] if len(_old) >= 5 else 64
                        _ft  = _old[6] if len(_old) >= 7 else 0
                        _c2  = _old[7] if len(_old) >= 8 else 8363
                        # Fields 9/10 itCut/itRes — the SampleInfo struct is
                        # now 10-wide; keep this rewrite in lockstep or
                        # glslang rejects the mismatched constructor.
                        _itc = _old[8] if len(_old) >= 9 else 127
                        _itr = _old[9] if len(_old) >= 10 else 0
                        # 11..14 envelope fields — struct is now 14-wide;
                        # keep this rewrite in lockstep. Preserved/raw-perc
                        # = drums, no vol envelope → eN=0 (inert).
                        _eo = _old[10] if len(_old) >= 11 else 0
                        _eN = _old[11] if len(_old) >= 12 else 0
                        _eS = _old[12] if len(_old) >= 13 else -1
                        _eL = _old[13] if len(_old) >= 14 else -1
                        # Field 15 nna — IT-only; for S3M/MOD/XM the struct
                        # stays 14-wide (no +nna patch was applied) so the
                        # constructor must NOT emit a 15th arg or glslang
                        # rejects it as too-many-args.
                        _is_it_now = getattr(mod, 'is_it', False)
                        _na = _old[14] if (_is_it_now and len(_old) >= 15) else None
                        _pstart = _PRESV_BASE + _prun
                        _prun += _full
                        _fields14 = (f"SampleInfo({_pstart}, {_full}, {_rp}, {_rl}, "
                                     f"{_vol}, 1, {_ft}, {_c2}, {_itc}, {_itr}, "
                                     f"{_eo}, {_eN}, {_eS}, {_eL}")
                        _si_txt[_inst-1] = (_fields14 + f", {_na})") if _na is not None else (_fields14 + ")")
                        _si_dirty = True
                        _prows.append((_pstart, _full, len(_ppcm)))
                        _ppcm.extend(int(v) for v in _a)
                    if _si_dirty and _msi:
                        _new_arr = (f"const SampleInfo samples[{len(_si_txt)}] "
                                    f"= SampleInfo[](\n    "
                                    + ",\n    ".join(_si_txt) + "\n);")
                        _common_src = (_common_src[:_msi.start()]
                                       + _new_arr + _common_src[_msi.end():])
                    if _prows:
                        _pk = []
                        for _i in range(0, len(_ppcm), 4):
                            _b = [(_ppcm[_i+_k] & 0xFF) if _i+_k < len(_ppcm) else 0
                                  for _k in range(4)]
                            _v = _b[0] | (_b[1]<<8) | (_b[2]<<16) | (_b[3]<<24)
                            if _v >= (1<<31): _v -= (1<<32)
                            _pk.append(_v)
                        _decl = (
                            "// ── --preserve: raw (un-VQ'd) sample data ──\n"
                            f"const int _PRESV_N = {len(_prows)};\n"
                            f"const int _presvStart[{len(_prows)}] = int[]("
                            + ",".join(str(r[0]) for r in _prows) + ");\n"
                            f"const int _presvLen[{len(_prows)}] = int[]("
                            + ",".join(str(r[1]) for r in _prows) + ");\n"
                            f"const int _presvOff[{len(_prows)}] = int[]("
                            + ",".join(str(r[2]) for r in _prows) + ");\n"
                            f"const int _presvPCM[{len(_pk)}] = int[]("
                            + ",".join(str(v) for v in _pk) + ");\n"
                            "int _presvByte(int i){int w=_presvPCM[i>>2];"
                            "int b=(w>>((i&3)*8))&0xFF;return b<128?b:b-256;}\n")
                        # Inject _decl with its consumer (getSample) in
                        # SOUND, not Common.  Common has the tight ~130 KB
                        # ShaderToy/ANGLE cap; the raw percussion PCM (short
                        # samples, ~tens of KB) would blow it (127→152 KB
                        # observed).  Sound has headroom and already hosts
                        # the moved decoders.  Co-locating keeps Common slim
                        # and the build GPU-fitting with --raw-perc default-on.
                        _gsig = "float getSample(int sampleIdx) {"
                        if _gsig in _sound_src:
                            _sound_src = _sound_src.replace(
                                _gsig, _decl + _gsig, 1)
                        else:
                            _common_src = _common_src.replace(
                                "#define USE_EMBEDDED_DATA",
                                _decl + "#define USE_EMBEDDED_DATA", 1)
                        _ghook = (_gsig +
                            "\n    for (int _pk=0;_pk<_PRESV_N;_pk++){"
                            " int _ps=_presvStart[_pk];"
                            " if (sampleIdx>=_ps && sampleIdx<_ps+_presvLen[_pk])"
                            " return float(_presvByte(_presvOff[_pk]+sampleIdx-_ps))/128.0; }")
                        if _gsig in _sound_src:
                            _sound_src = _sound_src.replace(_gsig, _ghook, 1)
                        elif _gsig in _common_src:
                            _common_src = _common_src.replace(_gsig, _ghook, 1)
                        else:
                            print("   WARNING: getSample() not found — "
                                  "--preserve hook not installed")
                        print(f"   ✓ --preserve {_presv_ids}: "
                              f"{len(_prows)} instrument(s) stored RAW "
                              f"({len(_ppcm):,} int8, no VQ) — pristine")

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

    # ── --png: repackage the finished VQ build's data into the PNG ──────────────
    # Everything above produced a normal VQ build (const arrays). Now move ALL of
    # that array data into the PNG and point the fetchers at it (decoder unchanged).
    if _png_mode:
        png_file = base_name + "_player_data.png"
        _tot, _offs = _pack_build_into_png(glsl_common_file,
                                           base_name + "_shadertoy_sound.glsl", png_file)
        png_size = os.path.getsize(png_file)
        print(f"   🖼️  --png: packed {len(_offs)} data array(s) → {png_file} "
              f"({_tot:,} B data, {png_size:,} B file); fetchers read the PNG (USE_EMBEDDED_DATA 0)")
        for _lg in sorted(_offs):
            print(f"        {_lg}_PNG_OFF = {_offs[_lg]}")
        args.use_png = True   # restore so JSON wiring + summary reflect PNG mode

    # Summary
    bufA_file_short = base_name + "_shadertoy_bufferA.glsl"

    # ── ShaderToy import descriptor (DEFAULT-ON; --no-json to skip) ──
    # Schema is the VERIFIED real export format (bare {ver,renderpass,
    # flags,info}; NO "Shader" wrapper; input fields are type/filepath,
    # NOT ctype/src; pass order Image,Common,Buffer A,Sound). Buffer
    # linkage = matching id strings (Buffer A out.id == its own in.id ==
    # Image ch1 in.id). Font + noise texture media paths are the fixed
    # ShaderToy built-ins (Font 1 / RGBA-noise) captured from a known-good
    # export. Reads the FINAL written tabs so all post-proc (Common→Sound
    # move, AA, patches) is included. Built BEFORE the summary so it can be
    # listed in the file list right under the HTML player. Additive.
    _json_listing = None
    if getattr(args, 'emit_json', False):
        try:
            _stj = base_name + "_shadertoy.json"
            def _rf(_p):
                with open(_p, 'r') as _h:
                    return _h.read()
            _c_common = _rf(glsl_common_file)
            _c_sound  = _rf(base_name + "_shadertoy_sound.glsl")
            _c_image  = _rf(base_name + "_shadertoy_image.glsl")
            _c_bufA   = _rf(bufA_file_short)
            _SMP_TEX = {"filter": "mipmap", "wrap": "repeat",
                        "vflip": "true", "srgb": "false", "internal": "byte"}
            _SMP_BUF = {"filter": "linear", "wrap": "clamp",
                        "vflip": "true", "srgb": "false", "internal": "byte"}
            _IN_FONT = {"channel": 0, "type": "texture", "id": "4dXGzr",
                        "filepath": "/media/a/08b42b43ae9d3c0605da11d0e"
                                    "ac86618ea888e62cdd9518ee8b9097488b3"
                                    "1560.png", "sampler": _SMP_TEX}
            _IN_BUFA = {"channel": 1, "type": "buffer", "id": "4dXGR8",
                        "filepath": "/media/previz/buffer00.png",
                        "sampler": _SMP_BUF}
            _IN_NOISE = {"channel": 2, "type": "texture", "id": "Xsf3zn",
                         "filepath": "/media/a/f735bee5b64ef98879dc618b0"
                                     "16ecf7939a5756040c2cde21ccb15e69a6"
                                     "e1cfb.png", "sampler": _SMP_TEX}
            _IN_BUFA_SELF = {"channel": 0, "type": "buffer", "id": "4dXGR8",
                             "filepath": "/media/previz/buffer00.png",
                             "sampler": _SMP_BUF}
            # --png: the data PNG as a NEAREST / no-vflip / no-sRGB texture so
            # texelFetch reads the raw bytes exactly (any filtering/flip/sRGB
            # would corrupt the packed data). Wired to Sound iChannel0.
            _SMP_DATA = {"filter": "nearest", "wrap": "clamp",
                         "vflip": "false", "srgb": "false", "internal": "byte"}
            _IN_PNG = {"channel": 0, "type": "texture", "id": "dataPNG0",
                       "filepath": base_name + "_player_data.png", "sampler": _SMP_DATA}
            _st_name = (mod.title.strip() or base_name)[:64]
            _st_desc = f"{_st_name} — MOD2GLSL v1.61"
            # Image is placed first so ShaderToy's new-shader first-tab reset only
            # wipes Image code (user re-pastes from *_shadertoy_image.glsl).
            # Image inputs ARE included: channels survive the reset so they are
            # pre-wired even though Image code gets blanked.
            # Buffer A inputs stay empty: non-empty inputs would cause ShaderToy
            # to silently drop Buffer A's code on new-shader import.
            # IDs/filepaths/samplers taken verbatim from a real ShaderToy export
            # (enigma.json + 2ND_PM.json).  Buffer B output ID is XsXGR8 (not 4dXGRN).
            _SMP_BUF  = {"filter":"linear","wrap":"clamp","vflip":"true","srgb":"false","internal":"byte"}
            _SMP_TEX  = {"filter":"mipmap","wrap":"repeat","vflip":"true","srgb":"false","internal":"byte"}
            _renderpass = [
                # Buffer B FIRST → ShaderToy selects the first tab after import,
                # so Buffer B is selected and absorbs the reset. Delete it after.
                # Output ID XsXGR8 verified from real ShaderToy export (enigma.json).
                {"outputs": [{"channel": 0, "id": "XsXGR8"}],
                 "inputs": [],
                 "code": ("// *** DELETE THIS TAB after import ***\n"
                          "// After clicking ×, press Cmd+Z (Mac) / Ctrl+Z (Win) immediately!\n"
                          "// ShaderToy pastes this tab's code into Image on delete —\n"
                          "// Cmd+Z undoes that and restores Image's real code.\n"
                          "void mainImage(out vec4 o,in vec2 u){o=vec4(0);}"),
                 "name": "Buffer B", "description": "", "type": "buffer"},
                # Image, Common, Buffer A, Sound follow — all survive import intact.
                {"outputs": [{"channel": 0, "id": "4dfGRr"}],
                 "inputs": [
                     {"channel": 0, "id": "4dXGzr",
                      "filepath": "/media/a/08b42b43ae9d3c0605da11d0eac86618ea888e62cdd9518ee8b9097488b31560.png",
                      "type": "texture", "sampler": _SMP_TEX},
                     {"channel": 1, "id": "4dXGR8",
                      "filepath": "/media/previz/buffer00.png",
                      "type": "buffer", "sampler": _SMP_BUF},
                     {"channel": 2, "id": "Xsf3zn",
                      "filepath": "/media/a/f735bee5b64ef98879dc618b016ecf7939a5756040c2cde21ccb15e69a6e1cfb.png",
                      "type": "texture", "sampler": _SMP_TEX},
                 ],
                 "code": _c_image, "name": "Image",
                 "description": "", "type": "image"},
                {"outputs": [], "inputs": [],
                 "code": _c_common, "name": "Common",
                 "description": "", "type": "common"},
                {"outputs": [{"channel": 0, "id": "4dXGR8"}],
                 "inputs": [
                     {"channel": 0, "id": "4dXGR8",
                      "filepath": "/media/previz/buffer00.png",
                      "type": "buffer", "sampler": _SMP_BUF},
                 ],
                 "code": _c_bufA, "name": "Buffer A",
                 "description": "", "type": "buffer"},
                {"outputs": [], "inputs": [],
                 "code": _c_sound, "name": "Sound",
                 "description": "", "type": "sound"},
            ]
            _shader = {
                "ver": "0.1",
                "renderpass": _renderpass,
                "flags": {"mFlagVR": False, "mFlagWebcam": False,
                          "mFlagSoundInput": False, "mFlagSoundOutput": True,
                          "mFlagKeyboard": False, "mFlagMultipass": True,
                          "mFlagMusicStream": False},
                "info": {"id": "-1",
                         "date": "0", "viewed": 0, "name": _st_name,
                         "username": "", "description": _st_desc, "likes": 0,
                         "published": 0, "flags": 0, "usePreview": 0,
                         "tags": ["sound", "music", "tracker", "s3m", "mod"],
                         "hasliked": 0, "parentid": "", "parentname": ""},
            }
            with open(_stj, 'w') as _h:
                json.dump(_shader, _h, separators=(',', ':'))
            _jb = os.path.getsize(_stj)
            _json_listing = (f"   📦 Import JSON:   {_stj}  ({_jb:,} B)"
                             f"  ← ShaderToy ▸ Import (all 4 tabs in one)")
        except Exception as _je:
            _json_listing = f"   ⚠ json emit failed ({_je}); .glsl tabs are fine"

    print(f"\n✅ Generated:")
    print(f"   🌐 HTML Player:    {html_file}")
    if _json_listing:
        print(_json_listing)
    _enc_label = "PNG-loaded" if args.use_png else "VQ-encoded"
    print(f"   📁 ShaderToy tabs: {glsl_common_file}  ({_enc_label})")
    print(f"                      {base_name}_shadertoy_sound.glsl")
    print(f"                      {base_name}_shadertoy_image.glsl")
    print(f"                      {bufA_file_short}  ← Buffer A (FFT + state)")
    if args.use_png and png_size:
        print(f"   🖼️  Sample PNG:     {png_file} ({png_size} bytes)")
    print(f"   🗜️  Compression:    RLE (patterns ~50%)")
    if getattr(args, 'emit_json', False):
        print(f"")
        print(f"   🔗 After import:")
        print(f"      Image    → channels pre-wired (Alphabet / Buffer A / RGBA Noise)")
        print(f"      Buffer A → iChannel0 pre-wired (self-reference)")
        print(f"      Buffer B → absorbs first-tab reset (gets default stub code)")
        print(f"      ⚠️  To remove Buffer B: click its tab × to delete it,")
        print(f"         then immediately press Cmd+Z (Mac) / Ctrl+Z (Win) —")
        print(f"         ShaderToy pastes Buffer B's code into Image on delete;")
        print(f"         Cmd+Z undoes that overwrite, restoring Image code.")
        print(f"      ⚠️  If Image code is blank, paste from:")
        print(f"         Image    ← {base_name}_shadertoy_image.glsl")
        print(f"         Buffer A ← {base_name}_shadertoy_bufferA.glsl  (if also blank)")
        print(f"      Sound    → no channels needed")
    print(f"   🖱️  Click anywhere to toggle oscilloscope ↔ spectrum")

    if args.downsample > 1:
        print(f"   ⬇️  Downsampled:    {args.downsample}x")
    print(f"\n💡 ProTracker timing:")
    print(f"   - Tick-based playback")
    print(f"   - Notes trigger on tick 0 only")
    print(f"   - Persistent channel state")

    # --mp3: render the actual ShaderToy Sound tab to an .mp3 (CPU, via the
    # glslang->spirv-cross->clang toolchain). Friendly no-op if tools missing.
    if getattr(args, 'mp3', False):
        _render_mp3_via_toolchain(base_name, float(getattr(args, 'mp3_secs', 180.0) or 180.0))




if __name__ == '__main__':
    # Turn expected user-facing failures (missing file, unsupported/unknown
    # format, corrupt header) into a clean one-line message instead of a
    # Python traceback. Unexpected errors (real bugs) still raise normally.
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\n  Interrupted.\n")
    except FileNotFoundError as _e:
        sys.exit(f"\n❌  File not found: {_e}\n   Check the path and try again.\n")
    except ValueError as _e:
        sys.exit(f"\n❌  {_e}\n")
