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
import math
from dataclasses import dataclass, field
from typing import Optional, List

# Adaptive sample compression (BW analysis + anti-alias decimation)
try:
    from scipy.signal import resample_poly as _resample_poly
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ============================================================================
# MikIT tick engine (inlined from mikit_engine.py)
# ============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# MikIT lookup tables  (from mmod_it1.cpp — verbatim)
# ──────────────────────────────────────────────────────────────────────────────

PITCH_TABLE = [
    2048,   2170,   2299,   2435,   2580,   2734,
    2896,   3069,   3251,   3444,   3649,   3866,
    4096,   4340,   4598,   4871,   5161,   5468,
    5793,   6137,   6502,   6889,   7298,   7732,
    8192,   8679,   9195,   9742,  10321,  10935,
    11585,  12274,  13004,  13777,  14596,  15464,
    16384,  17358,  18390,  19484,  20643,  21870,
    23170,  24548,  26008,  27554,  29193,  30929,
    32768,  34716,  36781,  38968,  41285,  43740,
    46341,  49097,  52016,  55109,  58386,  61858,
    65536,  69433,  73562,  77936,  82570,  87480,
    92682,  98193, 104032, 110218, 116772, 123715,
   131072, 138866, 147123, 155872, 165140, 174960,
   185364, 196386, 208064, 220436, 233544, 247431,
   262144, 277732, 294247, 311744, 330281, 349920,
   370728, 392772, 416128, 440872, 467088, 494862,
   524288, 555464, 588493, 623487, 660561, 699841,
   741455, 785544, 832255, 881744, 934175, 989724,
  1048576,1110928,1176987,1246974,1321123,1399681,
  1482910,1571089,1664511,1763488,1868350,1979448,
]

FINE_SLIDE_UP = [
    65536, 65595, 65654, 65714, 65773, 65832, 65892, 65951,
    66011, 66071, 66130, 66190, 66250, 66309, 66369, 66429,
]

FINE_SLIDE_DN = [
    65535, 65477, 65418, 65359, 65300, 65241, 65182, 65359,
    65065, 65006, 64947, 64888, 64830, 64772, 64713, 64645,
]

LINEAR_SLIDE_UP = [
    65536,  65773,  66011,  66250,  66489,  66730,  66971,  67213,
    67456,  67700,  67945,  68191,  68438,  68685,  68933,  69183,
    69433,  69684,  69936,  70189,  70443,  70693,  70953,  71210,
    71468,  71726,  71985,  72246,  72507,  72769,  73032,  73297,
    73562,  73828,  74095,  73563,  74632,  74902,  75172,  75444,
    75717,  75991,  76266,  76542,  76819,  77096,  77375,  77655,
    77936,  78218,  78501,  78785,  79069,  79355,  79642,  79930,
    80220,  80510,  80801,  81093,  81386,  81681,  81976,  82273,
    82570,  82869,  83169,  83469,  83771,  84074,  84378,  84683,
    84990,  85297,  85606,  85915,  86226,  86538,  86851,  87165,
    87480,  87796,  88114,  88433,  88752,  89073,  89396,  89719,
    90043,  90369,  90696,  91024,  91353,  91684,  92015,  92348,
    92682,  93017,  93354,  93691,  94030,  94370,  94711,  95054,
    95398,  95743,  96089,  96436,  96784,  97135,  97487,  97839,
    98193,  98548,  98905,  99262,  99621,  99982, 100343, 100706,
   101070, 101436, 101803, 102171, 102540, 102911, 103283, 103657,
   104032, 104408, 104786, 105165, 105545, 105927, 106310, 106694,
   107080, 107468, 107856, 108246, 108638, 109031, 109425, 109821,
   110218, 110617, 111017, 111418, 111821, 112226, 112631, 113039,
   113453, 113858, 114270, 114683, 115098, 115514, 115932, 116351,
   116772, 117194, 117618, 118043, 118470, 118899, 119329, 119760,
   120194, 120628, 121065, 121502, 121942, 122383, 122825, 123270,
   123715, 124163, 124612, 125063, 125515, 125969, 126425, 126882,
   127341, 127801, 128263, 128727, 129193, 129660, 130129, 130600,
   131072, 131546, 132022, 132499, 132978, 133459, 133942, 134427,
   134913, 135399, 135890, 136382, 136875, 137370, 137867, 138366,
   138866, 139368, 139872, 140378, 140886, 141395, 141907, 142420,
   142935, 143452, 143971, 144491, 145014, 145539, 146065, 146593,
   147123, 147655, 148189, 148725, 149263, 149803, 150345, 150889,
   151434, 151982, 152532, 153083, 153637, 154193, 154750, 155310,
   155872, 156435, 157001, 156569, 158139, 158711, 159285, 159861,
   160439, 161019, 161602, 162186, 162773, 163361, 163952, 164545,
   165140,
]

LINEAR_SLIDE_DN = [
    65535, 65300, 65065, 64830, 64596, 64364, 64132, 63901,
    63670, 63441, 63212, 62984, 62757, 62531, 62306, 62081,
    61858, 61635, 61413, 61191, 60971, 60751, 60532, 60314,
    60097, 59880, 59664, 59449, 59235, 59022, 58809, 58597,
    58386, 58176, 57966, 57757, 57549, 57341, 57135, 56929,
    56724, 56519, 56316, 56113, 55911, 55709, 55508, 55308,
    55109, 54910, 54713, 54515, 54319, 54123, 53928, 53734,
    53540, 53347, 53155, 52963, 52773, 52582, 52393, 52204,
    52016, 51829, 51642, 51456, 51270, 51085, 50901, 50718,
    50535, 50353, 50172, 49991, 49811, 49631, 49452, 49274,
    49097, 48920, 48743, 48568, 48393, 48128, 48044, 47871,
    47699, 47527, 47356, 47185, 47015, 46846, 46677, 46509,
    46341, 46174, 46008, 45842, 45677, 45512, 45348, 45185,
    45022, 44859, 44698, 44537, 44376, 44216, 44057, 43898,
    43740, 43582, 43425, 43269, 43113, 42958, 42803, 42649,
    42495, 42342, 42189, 42037, 41886, 41735, 41584, 41434,
    41285, 41136, 40988, 10840, 40639, 40566, 40400, 40253,
    40110, 39965, 39821, 39678, 39535, 39392, 39250, 39109,
    38968, 38828, 38688, 38548, 38409, 38271, 38133, 37996,
    37859, 37722, 37586, 37451, 37316, 37181, 37047, 36914,
    36781, 36648, 36516, 36385, 36254, 36123, 35993, 35863,
    35734, 35605, 35477, 35349, 35221, 35095, 34968, 34842,
    34716, 34591, 34467, 34343, 34219, 34095, 33973, 33850,
    33728, 33607, 33486, 33365, 33245, 33125, 33005, 32887,
    32768, 32650, 32532, 32415, 32298, 32182, 32066, 31950,
    31835, 31720, 31606, 31492, 31379, 31266, 31153, 31041,
    30929, 30817, 30706, 30596, 30485, 30376, 30226, 30157,
    30048, 29940, 29832, 29725, 29618, 29511, 29405, 29299,
    29193, 29088, 28983, 28879, 28774, 28671, 28567, 28464,
    28362, 28260, 28158, 28056, 27955, 27855, 27754, 27654,
    27554, 27455, 27356, 27258, 27159, 27062, 26964, 26867,
    26770, 26674, 26577, 26482, 26386, 26291, 26196, 26102,
    26008,
]

# Vibrato tables: sine(0), ramp-down(1), square(2)  — values -64..64
VIB_SINE = [
     0,  2,  3,  5,  6,  8,  9, 11, 12, 14, 16, 17, 19, 20, 22, 23,
    24, 26, 27, 29, 30, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44,
    45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 56, 57, 58, 59,
    59, 60, 60, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 63, 63, 63, 62, 62, 62, 61, 61, 60, 60,
    59, 59, 58, 57, 56, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46,
    45, 44, 43, 42, 41, 39, 38, 37, 36, 34, 33, 32, 30, 29, 27, 26,
    24, 23, 22, 20, 19, 17, 16, 14, 12, 11,  9,  8,  6,  5,  3,  2,
     0, -2, -3, -5, -6, -8, -9,-11,-12,-14,-16,-17,-19,-20,-22,-23,
   -24,-26,-27,-29,-30,-32,-33,-34,-36,-37,-38,-39,-41,-42,-43,-44,
   -45,-46,-47,-48,-49,-50,-51,-52,-53,-54,-55,-56,-56,-57,-58,-59,
   -59,-60,-60,-61,-61,-62,-62,-62,-63,-63,-63,-64,-64,-64,-64,-64,
   -64,-64,-64,-64,-64,-64,-63,-63,-63,-62,-62,-62,-61,-61,-60,-60,
   -59,-59,-58,-57,-56,-56,-55,-54,-53,-52,-51,-50,-49,-48,-47,-46,
   -45,-44,-43,-42,-41,-39,-38,-37,-36,-34,-33,-32,-30,-29,-27,-26,
   -24,-23,-22,-20,-19,-17,-16,-14,-12,-11, -9, -8, -6, -5, -3, -2,
]
VIB_RAMP = [
    64, 63, 63, 62, 62, 61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 56,
    56, 55, 55, 54, 54, 53, 53, 52, 52, 51, 51, 50, 50, 49, 49, 48,
    48, 47, 47, 46, 46, 45, 45, 44, 44, 43, 43, 42, 42, 41, 41, 40,
    40, 39, 39, 38, 38, 37, 37, 36, 36, 35, 35, 34, 34, 33, 33, 32,
    32, 31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24,
    24, 23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16,
    16, 15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10,  9,  9,  8,
     8,  7,  7,  6,  6,  5,  5,  4,  4,  3,  3,  2,  2,  1,  1,  0,
     0, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, -6, -6, -7, -7, -8,
    -8, -9, -9,-10,-10,-11,-11,-12,-12,-13,-13,-14,-14,-15,-15,-16,
   -16,-17,-17,-18,-18,-19,-19,-20,-20,-21,-21,-22,-22,-23,-23,-24,
   -24,-25,-25,-26,-26,-27,-27,-28,-28,-29,-29,-30,-30,-31,-31,-32,
   -32,-33,-33,-34,-34,-35,-35,-36,-36,-37,-37,-38,-38,-39,-39,-40,
   -40,-41,-41,-42,-42,-43,-43,-44,-44,-45,-45,-46,-46,-47,-47,-48,
   -48,-49,-49,-50,-50,-51,-51,-52,-52,-53,-53,-54,-54,-55,-55,-56,
   -56,-57,-57,-58,-58,-59,-59,-60,-60,-61,-61,-62,-62,-63,-63,-64,
]
VIB_SQUARE = (
    [64]*128 + [0]*128
)
VIB_TABLES = [VIB_SINE, VIB_RAMP, VIB_SQUARE]

def _muldiv(a, b, c):
    """Integer multiply-divide: (a * b) // c  (MikIT's MMulDiv)."""
    return (a * b) // c if c else 0

def _frq_slide_up(frq, v, linear):
    v = min(v, 255)
    if linear:
        return _muldiv(frq, LINEAR_SLIDE_UP[v], 65536)
    else:
        # MikIT Amiga mode: period -= v*4 per tick, clamped to period >= 1.
        # Equivalent freq formula: BASE * frq / (BASE - frq*v*4).
        # When denominator <= 0 the period would underflow; clamp to period=1
        # which gives the maximum possible Amiga frequency = BASE.
        denom = (1712 * 8363) - frq * v * 4
        if denom <= 0:
            return 1712 * 8363   # period = 1 → max Amiga frequency
        return _muldiv(frq, 1712 * 8363, denom)

def _frq_slide_dn(frq, v, linear):
    v = min(v, 255)
    if linear:
        return _muldiv(frq, LINEAR_SLIDE_DN[v], 65536)
    else:
        return _muldiv(frq, 1712 * 8363, (1712 * 8363) + frq * v * 4)

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ITEnvPoint:
    tick: int
    value: int  # 0-64 for vol, -32..32 for pan, SWORD for pitch

@dataclass
class ITEnvelope:
    enabled: bool = False
    loop_enabled: bool = False
    sustain_enabled: bool = False
    loop_start: int = 0   # index into points
    loop_end: int = 0
    sus_start: int = 0    # index into points (SLB)
    sus_end: int = 0      # index into points (SLE)
    points: List[ITEnvPoint] = field(default_factory=list)

    def value_at_tick(self, t: float, keyon: bool) -> float:
        """Return interpolated envelope value (normalized 0.0-1.0 for vol)."""
        pts = self.points
        if len(pts) < 2:
            return 1.0
        # last point tick
        last_tick = pts[-1].tick
        # apply sustain loop while keyon
        tick = t
        if keyon and self.sustain_enabled and len(pts) > self.sus_end:
            slb = pts[self.sus_start].tick
            sle = pts[self.sus_end].tick
            if sle > slb:
                span = sle - slb
                if tick >= slb:
                    tick = slb + (tick - slb) % span
        # apply normal loop after sustain
        elif self.loop_enabled and len(pts) > self.loop_end:
            lb = pts[self.loop_start].tick
            le = pts[self.loop_end].tick
            if le > lb:
                span = le - lb
                if tick >= lb:
                    tick = lb + (tick - lb) % span
        # clamp to last
        if tick >= last_tick:
            return pts[-1].value / 64.0
        if tick <= pts[0].tick:
            return pts[0].value / 64.0
        # interpolate
        for i in range(len(pts) - 1):
            x0, x1 = pts[i].tick, pts[i+1].tick
            if x0 <= tick <= x1:
                if x1 == x0:
                    return pts[i].value / 64.0
                frac = (tick - x0) / (x1 - x0)
                return (pts[i].value + frac * (pts[i+1].value - pts[i].value)) / 64.0
        return pts[-1].value / 64.0

    def is_done(self, t: float, keyon: bool) -> bool:
        if not self.enabled or len(self.points) < 2:
            return False
        if keyon and self.sustain_enabled:
            return False
        if self.loop_enabled:
            return False
        return t >= self.points[-1].tick

@dataclass
class ITInstrument:
    name: str = ""
    nna: int = 0        # 0=cut 1=continue 2=noteoff 3=notefade
    dct: int = 0        # 0=off 1=note 2=sample 3=instrument
    dca: int = 0        # 0=cut 1=noteoff 2=notefade
    fadeout: int = 0    # 0..1024; per-tick = fadeout/512 of NFC
    global_vol: int = 128  # 0..128
    dfp: int = 0        # default panning; bit7=override
    pps: int = 0        # pan position separation
    ppc: int = 0        # pan position centre (note)
    rv: int = 0         # random volume variation
    rp: int = 0         # random panning variation
    note_to_sample: List[int] = field(default_factory=lambda: [0]*120)
    note_to_note:   List[int] = field(default_factory=lambda: list(range(120)))
    vol_env: ITEnvelope = field(default_factory=ITEnvelope)
    pan_env: ITEnvelope = field(default_factory=ITEnvelope)
    ptc_env: ITEnvelope = field(default_factory=ITEnvelope)

@dataclass
class ITSample:
    name: str = ""
    c5speed: int = 8363    # Hz at C-5
    vol: int = 64          # 0..64
    global_vol: int = 64   # 0..64
    dfp: int = 0           # default panning
    length: int = 0
    loop_start: int = 0
    loop_end: int = 0
    sus_loop_start: int = 0
    sus_loop_end: int = 0
    has_loop: bool = False
    has_sus_loop: bool = False
    bidi_loop: bool = False
    bidi_sus: bool = False
    data: object = None    # numpy array or None
    # Sample vibrato
    vib_speed: int = 0
    vib_depth: int = 0
    vib_rate: int = 0
    vib_type: int = 0

# ──────────────────────────────────────────────────────────────────────────────
# Pattern cell
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ITCell:
    note: int = 0    # 0=empty, 1-119=note, 254=cut, 255=noteoff
    inst: int = 0    # 0=empty
    vol: int = 255   # 255=empty; 0-64=set vol; 65-74=vol slide up; etc.
    eff: int = 0     # 0=none; 1=A..25=Y
    par: int = 0
    has_note: bool = False
    has_inst: bool = False
    has_vol: bool = False
    has_eff: bool = False

# ──────────────────────────────────────────────────────────────────────────────
# Voice (virtual channel) state — mirrors ITVIRTCH
# ──────────────────────────────────────────────────────────────────────────────

class ITVoice:
    def __init__(self, vid):
        self.vid = vid          # unique voice ID assigned at creation
        self.owner = -1         # column index that owns this voice (-1 = free)
        self.active = False
        self.background = True
        self.note = 0
        self.inst_idx = 0       # 1-based; 0 = none
        self.smp_idx = 0        # 1-based; 0 = none
        self.inst: Optional[ITInstrument] = None
        self.smp: Optional[ITSample] = None
        # Volume components
        self.nfc = 0            # note fade component 0..1024
        self.sv = 0             # sample global vol 0..64
        self.iv = 0             # instrument global vol 0..128
        self.cv = 0             # channel vol 0..64
        self.np = 32            # note panning 0..64
        self.vol = 0            # current column volume 0..64
        self.frq = 0            # current frequency (Hz-ish, MikIT units)
        self.vev = 64           # vol envelope value 0..64
        # Envelope processors (tick counters)
        self.env_tick_vol = 0.0
        self.env_tick_pan = 0.0
        self.env_tick_ptc = 0.0
        self.keyon = False
        self.dofade = False
        self.cut = False
        self.kick = False
        self.start_offset = 0
        # Sample playback tracking for timeline
        self.trigger_abs_tick = 0    # absolute tick when current note started
        self.trigger_samp_pos = 0    # sample position at trigger
        # Vibrato (sample-level)
        self.vibrunp = 0
        self.vibrund = 0

    def reset_for_new_note(self, abs_tick):
        self.nfc = 1024
        self.env_tick_vol = 0.0
        self.env_tick_pan = 0.0
        self.env_tick_ptc = 0.0
        self.keyon = True
        self.dofade = False
        self.cut = False
        self.vibrunp = 0
        self.vibrund = 0
        self.trigger_abs_tick = abs_tick
        self.trigger_samp_pos = self.start_offset

    def get_final_vol(self, run_gv):
        """Compute final output volume in 0-255 range (MikIT formula)."""
        if not self.active or self.cut:
            return 0
        endvol = run_gv          # 128
        endvol *= self.sv        # 64
        endvol *= self.iv        # 128
        endvol *= self.cv        # 64
        endvol >>= 17            # → 512 max
        endvol *= self.vol       # 64
        endvol *= self.vev       # 64
        endvol *= self.nfc       # 1024
        if endvol > 0:
            endvol //= 8421504   # → 255 max
        return min(255, max(0, endvol))

    def get_final_freq(self):
        """Return frequency in Hz (float)."""
        return float(self.frq)

    def advance_envelopes(self):
        """Advance envelope ticks by 1."""
        if self.inst and self.inst.vol_env.enabled:
            if self.keyon and self.inst.vol_env.sustain_enabled:
                pts = self.inst.vol_env.points
                if pts and self.env_tick_vol < pts[-1].tick:
                    sus_e = pts[self.inst.vol_env.sus_end].tick if len(pts) > self.inst.vol_env.sus_end else 999999
                    if self.env_tick_vol < sus_e:
                        self.env_tick_vol += 1
                    # Hold at sustain end point
            else:
                self.env_tick_vol += 1
        if self.inst and self.inst.pan_env.enabled:
            self.env_tick_pan += 1
        if self.inst and self.inst.ptc_env.enabled:
            self.env_tick_ptc += 1

    def update_vev(self):
        if self.inst and self.inst.vol_env.enabled:
            v = self.inst.vol_env.value_at_tick(self.env_tick_vol, self.keyon)
            self.vev = int(v * 64)
        else:
            self.vev = 64

    def do_nna(self, nna_action):
        """Apply NNA action (0=cut,1=continue,2=noteoff,3=fade)."""
        self.background = True
        self.owner = -1
        if nna_action == 0:
            self.cut = True
            self.active = False
        elif nna_action == 1:
            pass  # continue
        elif nna_action == 2:
            self.keyon = False
        elif nna_action == 3:
            self.dofade = True

# ──────────────────────────────────────────────────────────────────────────────
# Column (channel) state — mirrors ITCOLUMN
# ──────────────────────────────────────────────────────────────────────────────

class ITColumn:
    def __init__(self, col_idx, engine):
        self.col = col_idx
        self.eng = engine        # reference to ITPlayer
        self.voice: Optional[ITVoice] = None
        # Persistent column state
        self.note = 0
        self.inst = 0
        self.smp = 0
        self.smpnote = 0
        self.vol = 64
        self.frq = 0
        self.cv = 64             # channel volume
        self.cp = 32             # channel panning (default centre)
        self.np = 32             # note panning
        self.kick = False
        self.retrig = False
        self.own_vol = False
        self.own_frq = False
        self.own_ofs = False
        self.ovol = 0
        self.ofrq = 0
        self.start_offset = 0
        self.dest_frq = 0
        # Effect memory
        self.volslidespd = 0
        self.chvolslidespd = 0
        self.pitchslidespd = 0
        self.panslidespd = 0
        self.temposlidespd = 0
        self.gvolslidespd = 0
        self.vvolslidespd = 0
        self.vibspd = 0
        self.vibdpt = 0
        self.vibtyp = 0
        self.vibptr = 0
        self.trmspd = 0
        self.trmdpt = 0
        self.trmtyp = 0
        self.trmptr = 0
        self.tremorspd = 0
        self.tremorptr = 0
        self.arpeggio = 0
        self.qspeed = 0
        self.qptr = 0
        self.sfxdata = 0
        self.loopback_point = 0
        self.loopback_count = 0
        # Current row note data
        self.cell = ITCell()
        self.tick = 0

    def find_smp_note_freq(self):
        """Resolve sample, transposed note, and frequency for current inst+note."""
        eng = self.eng
        if self.inst > 0 and self.inst <= len(eng.instruments):
            i = eng.instruments[self.inst - 1]
            note_idx = max(0, min(119, self.note))  # IT notes are 0-based, 0=C-0 .. 119=B-9
            raw_smp = i.note_to_sample[note_idx]
            raw_note = i.note_to_note[note_idx]
            self.smp = raw_smp
            self.smpnote = raw_note
            # Frequency
            if 0 <= raw_note < 120:
                self.frq = PITCH_TABLE[raw_note]
            else:
                self.frq = PITCH_TABLE[60]
            if self.smp and self.smp <= len(eng.samples):
                s = eng.samples[self.smp - 1]
                self.frq = _muldiv(self.frq, s.c5speed, 65536)
            # Panning: sample override > instrument override > channel pan
            self.np = self.cp
            if self.smp and self.smp <= len(eng.samples):
                s = eng.samples[self.smp - 1]
                if s.dfp & 0x80:
                    self.np = s.dfp & 0x7F
                elif i.dfp & 0x80:
                    self.np = i.dfp & 0x7F
                t = self.np + (((self.note - 1 - i.ppc) * i.pps) >> 3)
                self.np = max(0, min(64, t))

    def process_tick0(self, abs_tick):
        """Row-start processing (tick == 0 only, or Sxd note-delay)."""
        cell = self.cell
        # Sxd note delay: only fire if tick == note_delay_val
        if cell.has_eff and cell.eff == 19 and (cell.par >> 4) == 0xD:
            if self.tick != (cell.par & 0xF):
                return
        else:
            if self.tick != 0:
                return

        doing_slide = False
        # Check for tone portamento (vol col G, or effect G)
        is_porta = (
            (cell.has_vol and 193 <= cell.vol <= 202) or
            (cell.has_eff and cell.eff == 7)
        )
        if is_porta and cell.has_note and cell.note <= 119:
            doing_slide = True
            slide_ins = self.inst
            if cell.has_inst:
                slide_ins = cell.inst
            if slide_ins > 0 and slide_ins <= len(self.eng.instruments):
                ii = self.eng.instruments[slide_ins - 1]
                raw_smp = ii.note_to_sample[max(0, min(119, cell.note))]
                raw_note = ii.note_to_note[max(0, min(119, cell.note))]
                if raw_smp and raw_smp <= len(self.eng.samples):
                    self.dest_frq = _muldiv(PITCH_TABLE[raw_note],
                                            self.eng.samples[raw_smp - 1].c5speed, 65536)
                else:
                    self.dest_frq = 0
                    cell.par = 0
            if not self.voice:
                doing_slide = False

        # Note column
        if cell.has_note:
            if cell.note == 255:      # note-off
                if self.voice:
                    self.voice.keyon = False
            elif cell.note == 254:    # note-cut
                if self.voice:
                    self.voice.cut = True
            elif cell.note <= 119:
                if not doing_slide:
                    self.note = cell.note
                    self.kick = True
                    self.find_smp_note_freq()

        # Instrument column
        if cell.has_inst and cell.inst <= len(self.eng.instruments):
            self.inst = cell.inst
            if not doing_slide and cell.has_note:
                self.kick = True
                self.find_smp_note_freq()
                self.tremorptr = 0
                self.vibptr = 0
                self.trmptr = 0
                self.qptr = 0
            if self.smp and self.smp <= len(self.eng.samples):
                self.vol = self.eng.samples[self.smp - 1].vol

        # Volume column set-vol
        if cell.has_vol:
            if cell.vol <= 64:
                self.vol = cell.vol
            elif 128 <= cell.vol <= 192:
                self.cp = cell.vol - 128
                self.np = self.cp

        # Safety: don't kick with no sample
        if self.inst == 0 or self.smp == 0:
            self.kick = False
        if self.inst > len(self.eng.instruments) or self.smp > len(self.eng.samples):
            self.kick = False

    # ── Effect helpers ────────────────────────────────────────────────────────

    def _vol_slide_up(self, v):
        self.vol = min(64, self.vol + v)

    def _vol_slide_dn(self, v):
        self.vol = max(0, self.vol - v)

    def _frq_up(self, v):
        self.frq = _frq_slide_up(self.frq, v, self.eng.linear_freq)

    def _frq_dn(self, v):
        self.frq = _frq_slide_dn(self.frq, v, self.eng.linear_freq)

    def _pan_left(self, v):
        self.np = max(0, self.np - v)

    def _pan_right(self, v):
        self.np = min(64, self.np + v)

    def do_vibrato(self):
        self.own_frq = True
        vtyp = self.vibtyp & 3
        if vtyp < 3:
            val = VIB_TABLES[vtyp][self.vibptr & 0xFF]
        else:
            import random
            val = random.randint(-64, 64)
        val = val * self.vibdpt >> 8
        self.ofrq = self.frq
        if val < 0:
            self.ofrq = _frq_slide_dn(self.frq, -val, self.eng.linear_freq)
        elif val > 0:
            self.ofrq = _frq_slide_up(self.frq, val, self.eng.linear_freq)
        self.vibptr = (self.vibptr + (self.vibspd << 2)) & 0xFF

    # ── Effect dispatch ───────────────────────────────────────────────────────

    def effects(self):
        cell = self.cell
        tick = self.tick

        # Volume column effects
        if cell.has_vol:
            v = cell.vol
            if 65 <= v <= 74:   self._eff_vol_slide_up(v - 65)
            elif 75 <= v <= 84: self._eff_vol_slide_dn(v - 75)
            elif 85 <= v <= 94: self._eff_fine_vol_up(v - 85)
            elif 95 <= v <= 104: self._eff_fine_vol_dn(v - 95)
            elif 105 <= v <= 114: self._eff_vib_depth(v - 105)
            elif 115 <= v <= 124: self._eff_vib_speed(v - 115)
            elif 193 <= v <= 202: self._eff_porta(v - 193, cell.par if cell.has_eff and cell.eff == 7 else None)
            elif 203 <= v <= 212: self._eff_vib_noteoff(v - 203)

        if not cell.has_eff:
            return

        eff = cell.eff
        par = cell.par

        if eff == 1:   self._eff_A(par)   # Set speed
        elif eff == 2: self._eff_B(par)   # Jump to order
        elif eff == 3: self._eff_C(par)   # Break to row
        elif eff == 4: self._eff_D(par)   # Vol slide
        elif eff == 5: self._eff_E(par)   # Pitch slide down
        elif eff == 6: self._eff_F(par)   # Pitch slide up
        elif eff == 7: self._eff_G(par)   # Tone portamento
        elif eff == 8: self._eff_H(par)   # Vibrato
        elif eff == 9: self._eff_I(par)   # Tremor
        elif eff == 10: self._eff_J(par)  # Arpeggio
        elif eff == 11:                   # K = H00 + D
            self.do_vibrato()
            self._eff_D(par)
        elif eff == 12:                   # L = G00 + D
            self._eff_G(0)
            self._eff_D(par)
        elif eff == 13: self._eff_M(par)  # Set channel vol
        elif eff == 14: self._eff_N(par)  # Slide channel vol
        elif eff == 15: self._eff_O(par)  # Sample offset
        elif eff == 16: self._eff_P(par)  # Pan slide
        elif eff == 17: self._eff_Q(par)  # Retrig
        elif eff == 18: self._eff_R(par)  # Tremolo
        elif eff == 19: self._eff_S(par)  # Special
        elif eff == 20: self._eff_T(par)  # Tempo
        elif eff == 21: self._eff_U(par)  # Fine vibrato
        elif eff == 22: self._eff_V(par)  # Set global vol
        elif eff == 23: self._eff_W(par)  # Slide global vol
        elif eff == 24: self._eff_X(par)  # Set pan
        elif eff == 25: self._eff_Y(par)  # Panbrello
        elif eff == 26: self._eff_K_delayed(par)  # XM K: key-off at tick par
        elif eff == 27: self._eff_XM_vib(par)    # XM vibrato (raw depth, no <<2)

    def post_effects(self):
        cell = self.cell
        if cell.has_eff and cell.eff == 19:
            self._post_S(cell.par)

    # ── Individual effects (MikIT mmod_it2.cpp) ───────────────────────────────

    def _eff_A(self, par):
        if self.tick == 0 and par:
            self.eng.speed = par

    def _eff_B(self, par):
        if self.tick == 0:
            self.eng.order_jump = par + 1

    def _eff_C(self, par):
        if self.tick == 0:
            self.eng.break_to_row = par + 1

    def _eff_D(self, par):
        if self.tick == 0 and par:
            self.volslidespd = par
        hi = (self.volslidespd >> 4) & 0xF
        lo = self.volslidespd & 0xF
        if self.tick == 0:
            if hi == 0xF:
                if lo: self._vol_slide_dn(lo)
                else:  self._vol_slide_up(0xF)
            elif lo == 0xF:
                if hi: self._vol_slide_up(hi)
                else:  self._vol_slide_dn(0xF)
        else:
            if hi == 0:   self._vol_slide_dn(lo)
            elif lo == 0: self._vol_slide_up(hi)

    def _eff_E(self, par):
        if self.tick == 0 and par:
            self.pitchslidespd = par
        hi = (self.pitchslidespd >> 4) & 0xF
        lo = self.pitchslidespd & 0xF
        if self.tick == 0:
            if hi == 0xF:
                self._frq_dn(lo)
            elif hi == 0xE:
                lin = self.eng.linear_freq
                self.frq = _muldiv(self.frq, FINE_SLIDE_DN[lo], 65536)
        else:
            if hi < 0xE:
                self._frq_dn(self.pitchslidespd)

    def _eff_F(self, par):
        if self.tick == 0 and par:
            self.pitchslidespd = par
        hi = (self.pitchslidespd >> 4) & 0xF
        lo = self.pitchslidespd & 0xF
        if self.tick == 0:
            if hi == 0xF:
                self._frq_up(lo)
            elif hi == 0xE:
                self.frq = _muldiv(self.frq, FINE_SLIDE_UP[lo], 65536)
        else:
            if hi < 0xE:
                self._frq_up(self.pitchslidespd)

    def _eff_G(self, par):
        if self.tick == 0 and par:
            self.pitchslidespd = par
        if not self.dest_frq:
            return
        if self.tick:
            if self.frq < self.dest_frq:
                self._frq_up(self.pitchslidespd)
                if self.frq > self.dest_frq:
                    self.frq = self.dest_frq
            elif self.frq > self.dest_frq:
                self._frq_dn(self.pitchslidespd)
                if self.frq < self.dest_frq:
                    self.frq = self.dest_frq

    def _eff_H(self, par):
        if self.tick == 0:
            if par >> 4:   self.vibspd = par >> 4
            if par & 0xF:  self.vibdpt = (par & 0xF) << 2
        self.do_vibrato()

    def _eff_I(self, par):
        if self.tick == 0 and par:
            self.tremorspd = par
        on  = max(1, self.tremorspd >> 4)
        off = max(1, self.tremorspd & 0xF)
        self.tremorptr %= (on + off)
        self.ovol = self.vol if self.tremorptr < on else 0
        self.tremorptr += 1
        self.own_vol = True

    def _eff_J(self, par):
        if self.tick == 0 and par:
            self.arpeggio = par
        phase = self.tick % 3
        if phase == 1:
            self.ofrq = _muldiv(self.frq, LINEAR_SLIDE_UP[16 * (self.arpeggio >> 4)], 65536)
            self.own_frq = True
        elif phase == 2:
            self.ofrq = _muldiv(self.frq, LINEAR_SLIDE_UP[16 * (self.arpeggio & 0xF)], 65536)
            self.own_frq = True

    def _eff_M(self, par):
        if self.tick == 0 and par <= 64:
            self.cv = par

    def _eff_N(self, par):
        if self.tick == 0 and par:
            self.chvolslidespd = par
        hi = (self.chvolslidespd >> 4) & 0xF
        lo = self.chvolslidespd & 0xF
        if self.tick == 0:
            if hi == 0xF:
                self.cv = max(0, self.cv - lo)
            elif lo == 0xF:
                self.cv = min(64, self.cv + hi)
        else:
            if hi == 0:   self.cv = max(0, self.cv - lo)
            elif lo == 0: self.cv = min(64, self.cv + hi)

    def _eff_O(self, par):
        if self.tick == 0 and par:
            self.start_offset = par << 8
        self.own_ofs = True

    def _eff_P(self, par):
        if self.tick == 0 and par:
            self.panslidespd = par
        hi = (self.panslidespd >> 4) & 0xF
        lo = self.panslidespd & 0xF
        if self.tick == 0:
            if hi == 0xF: self._pan_right(lo)
            elif lo == 0xF: self._pan_left(hi)
        else:
            if hi == 0:   self._pan_right(lo)
            elif lo == 0: self._pan_left(hi)
        self.cp = self.np

    def _eff_Q(self, par):
        if self.tick == 0 and par:
            self.qspeed = par
        qtyp = self.qspeed >> 4
        qspd = self.qspeed & 0xF
        val = self.vol
        if self.qptr >= qspd:
            vol_mods = {1:-1,2:-2,3:-4,4:-8,5:-16,6:None,7:None,9:1,10:2,11:4,12:8,13:16,14:None,15:None}
            if qtyp in vol_mods:
                dm = vol_mods[qtyp]
                if dm is not None:
                    val += dm
                elif qtyp == 6:
                    val = val * 2 // 3
                elif qtyp == 7:
                    val >>= 1
                elif qtyp == 14:
                    val = val * 3 >> 1
                elif qtyp == 15:
                    val <<= 1
            val = max(0, min(64, val))
            self.qptr = 0
            self.kick = True
            self.retrig = True
        self.qptr += 1
        self.vol = val

    def _eff_R(self, par):
        if self.tick == 0:
            if par >> 4:  self.trmspd = par >> 4
            if par & 0xF: self.trmdpt = par & 0xF
        vtyp = self.trmtyp & 3
        if vtyp < 3:
            val = VIB_TABLES[vtyp][self.trmptr & 0xFF]
        else:
            import random
            val = random.randint(-64, 64)
        val = val * self.trmdpt >> 6
        self.ovol = max(0, min(64, self.vol + val))
        self.own_vol = True
        self.trmptr = (self.trmptr + (self.trmspd << 2)) & 0xFF

    def _eff_S(self, par):
        hi = (par >> 4) & 0xF
        lo = par & 0xF
        if self.tick == 0:
            if hi == 0xB:   # pattern loop
                if lo == 0:
                    self.loopback_point = self.eng.row
                else:
                    if self.loopback_count == 0:
                        self.loopback_count = lo
                        self.eng.break_to_row = self.loopback_point + 1
                    else:
                        self.loopback_count -= 1
                        if self.loopback_count > 0:
                            self.eng.break_to_row = self.loopback_point + 1
            elif hi == 0xC:   # note cut at tick lo
                pass  # handled in post
            elif hi == 0xD:   # note delay at tick lo
                pass  # handled in process_tick0

    def _post_S(self, par):
        hi = (par >> 4) & 0xF
        lo = par & 0xF
        if hi == 0xC and self.tick == lo:
            if self.voice:
                self.voice.cut = True

    def _eff_T(self, par):
        if self.tick == 0:
            if par >= 0x20:
                self.eng.tempo = par
            else:
                if par >> 4 == 0:
                    self.eng.tempo = max(32, self.eng.tempo - (par & 0xF))
                elif par >> 4 == 1:
                    self.eng.tempo = min(255, self.eng.tempo + (par & 0xF))
                else:
                    if par:
                        self.temposlidespd = par
                    hi = (self.temposlidespd >> 4) & 0xF
                    lo = self.temposlidespd & 0xF
                    if hi:
                        self.eng.tempo = min(255, self.eng.tempo + hi)
                    elif lo:
                        self.eng.tempo = max(32, self.eng.tempo - lo)

    def _eff_U(self, par):
        if self.tick == 0:
            if par >> 4:  self.vibspd = par >> 4
            if par & 0xF: self.vibdpt = par & 0xF
        self.do_vibrato()

    def _eff_V(self, par):
        if self.tick == 0:
            self.eng.run_gv = max(0, min(128, par))

    def _eff_W(self, par):
        if self.tick == 0 and par:
            self.gvolslidespd = par
        hi = (self.gvolslidespd >> 4) & 0xF
        lo = self.gvolslidespd & 0xF
        if self.tick == 0:
            if hi == 0xF: self.eng.run_gv = min(128, self.eng.run_gv + lo)
            elif lo == 0xF: self.eng.run_gv = max(0, self.eng.run_gv - hi)
        else:
            if hi == 0:   self.eng.run_gv = max(0, self.eng.run_gv - lo)
            elif lo == 0: self.eng.run_gv = min(128, self.eng.run_gv + hi)

    def _eff_X(self, par):
        if self.tick == 0:
            self.cp = max(0, min(64, par))
            self.np = self.cp

    def _eff_Y(self, par):
        # Panbrello: like tremolo but on panning
        if self.tick == 0:
            if par >> 4:  self.vibspd = par >> 4   # reuse vibspd for panbrello
            if par & 0xF: self.vibdpt = par & 0xF
        # Not implementing full panbrello for now

    def _eff_K_delayed(self, par):
        """XM K effect: key-off at tick par (MikIT UNI_KEYFADE)."""
        if self.tick == par:
            if self.voice:
                self.voice.keyon = False

    def _eff_XM_vib(self, par):
        """XM effect 4 vibrato: depth = raw nibble (no <<2 scaling, unlike IT EffectH)."""
        if self.tick == 0:
            if par >> 4:  self.vibspd = par >> 4
            if par & 0xF: self.vibdpt = par & 0xF
        self.do_vibrato()

    # Vol-column effect sub-dispatchers
    def _eff_vol_slide_up(self, v):
        if self.tick: self._vol_slide_up(v)

    def _eff_vol_slide_dn(self, v):
        if self.tick: self._vol_slide_dn(v)

    def _eff_fine_vol_up(self, v):
        if self.tick == 0: self._vol_slide_up(v)

    def _eff_fine_vol_dn(self, v):
        if self.tick == 0: self._vol_slide_dn(v)

    def _eff_vib_depth(self, v):
        if self.tick == 0: self.vibdpt = v << 2
        self.do_vibrato()

    def _eff_vib_speed(self, v):
        if self.tick == 0: self.vibspd = v
        self.do_vibrato()

    def _eff_porta(self, v, override_par):
        # vol-col portamento (G command value 0-9 stored as speed)
        spd = override_par if override_par is not None else (v if v else self.pitchslidespd)
        self._eff_G(spd)

    def _eff_vib_noteoff(self, v):
        self._eff_H(v)

# ──────────────────────────────────────────────────────────────────────────────
# Voice segment (timeline output)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VoiceSegment:
    """One contiguous note-on event on a virtual voice."""
    voice_id: int         # unique voice ID
    channel: int          # physical channel index (-1 = NNA ghost)
    start_tick: int       # absolute tick when note triggered
    end_tick: int         # absolute tick when voice became silent (-1 = ongoing)
    sample_idx: int       # 0-based sample index
    # Per-tick state list: [(abs_tick, freq_hz, vol_f, pan_f, samp_pos)]
    # tick_states[0] is always at start_tick
    tick_states: List[tuple] = field(default_factory=list)
    # Derived loop info (from sample)
    loop_start: int = 0
    loop_end: int = 0
    loop_type: int = 0    # 0=none 1=fwd 2=bidi

# ──────────────────────────────────────────────────────────────────────────────
# Main player engine
# ──────────────────────────────────────────────────────────────────────────────

class ITPlayer:
    """
    Tick-accurate IT/XM/S3M player engine.
    Call run() to simulate the song and return the voice timeline.
    """
    MAX_VOICES = 64

    def __init__(self):
        self.instruments: List[ITInstrument] = []
        self.samples: List[ITSample] = []
        self.patterns: List[List[List[ITCell]]] = []  # [pat][row][ch]
        self.orders: List[int] = []
        self.num_channels = 0
        self.initial_speed = 6
        self.initial_tempo = 125
        self.global_volume = 128   # IT header GV
        self.mix_volume = 128      # IT header MV
        self.linear_freq = True    # IT_LINEAR_FREQ flag
        self.channel_pan: List[int] = []   # 0-64 per channel
        self.channel_vol: List[int] = []   # 0-64 per channel

        # Runtime state
        self.speed = 6
        self.tempo = 125
        self.tick = 0
        self.row = 0
        self.song_pos = 0
        self.abs_tick = 0
        self.run_gv = 128
        self.order_jump = 0
        self.break_to_row = 0
        self.pattern_delay = 0
        self.pattern_del_rq = 0

        self.voices: List[ITVoice] = []
        self.columns: List[ITColumn] = []
        self._next_vid = 0
        self._next_seg_id = 0            # unique per note-on event
        self._voice_segs: dict = {}      # seg_id -> VoiceSegment (in progress)
        self._voice_seg_id: dict = {}    # vid -> current seg_id
        self._finished_segs: List[VoiceSegment] = []
        # Per-row event log: (abs_tick, song_pos, row, pat_no) emitted at row
        # load time. The HTML segment player uses this to drive the tracker
        # view: bsearch abs_tick to find current row during audio playback.
        self.row_events: List[tuple] = []

    def _alloc_voice(self) -> ITVoice:
        vid = self._next_vid
        self._next_vid += 1
        v = ITVoice(vid)
        return v

    def _find_least_active(self) -> Optional[ITVoice]:
        """Find free or lowest-NFC background voice (MikIT vFindLeastActive)."""
        best = None
        best_nfc = 0xFFFF
        for v in self.voices:
            if not v.active:
                return v
            if v.background and v.nfc < best_nfc:
                best_nfc = v.nfc
                best = v
        return best

    def _do_nna(self, col: ITColumn):
        """Handle NNA when a new note triggers on a column that already has a voice."""
        v = col.voice
        if v is None:
            return
        if not v.active:
            return
        # Move old voice to background via its instrument's NNA
        nna = v.inst.nna if v.inst else 0
        if nna == 0:  # cut: close immediately — otherwise old voice plays a full extra tick at full
            self._close_segment(v)  # amplitude while new voice starts, producing an audible click
        v.do_nna(nna)

    def _find_or_steal_voice(self) -> Optional[ITVoice]:
        v = self._find_least_active()
        if v is None and self.voices:
            # steal the oldest background voice
            bg = [x for x in self.voices if x.background]
            if bg:
                v = min(bg, key=lambda x: x.trigger_abs_tick)
        return v

    def _dup_check(self, col: ITColumn):
        """Duplicate Check (MikIT DuplicateCheck)."""
        if not col.inst or col.inst > len(self.instruments):
            return
        inst = self.instruments[col.inst - 1]
        if inst.dct == 0:
            return
        for v in self.voices:
            if not v.active or not v.background:
                continue
            if v.owner != col.col:
                continue
            if v.inst_idx != col.inst:
                continue
            if inst.dct == 1 and v.note != col.note:
                continue
            if inst.dct == 2 and v.smp_idx != col.smp:
                continue
            v.do_nna(inst.dca)

    def _start_segment(self, v: ITVoice, col: ITColumn):
        """Begin recording a VoiceSegment for this voice (new note-on event)."""
        # Close any previous segment on this voice first
        self._close_segment(v)
        if v.smp_idx == 0 or v.smp_idx > len(self.samples):
            return
        smp = self.samples[v.smp_idx - 1]
        lt = 0
        if smp.has_loop:
            lt = 2 if smp.bidi_loop else 1
        seg_id = self._next_seg_id
        self._next_seg_id += 1
        seg = VoiceSegment(
            voice_id=seg_id,           # unique per note-on, not per physical voice
            channel=col.col if col else -1,
            start_tick=self.abs_tick,
            end_tick=-1,
            sample_idx=v.smp_idx - 1,
            loop_start=smp.loop_start,
            loop_end=smp.loop_end,
            loop_type=lt,
        )
        self._voice_segs[seg_id] = seg
        self._voice_seg_id[v.vid] = seg_id

    def _close_segment(self, v: ITVoice):
        seg_id = self._voice_seg_id.pop(v.vid, None)
        if seg_id is not None:
            seg = self._voice_segs.pop(seg_id, None)
            if seg:
                seg.end_tick = self.abs_tick
                self._finished_segs.append(seg)

    def _snapshot_voice(self, v: ITVoice, col: ITColumn):
        """Record current voice state into the active segment."""
        seg_id = self._voice_seg_id.get(v.vid)
        if seg_id is None:
            return
        seg = self._voice_segs.get(seg_id)
        if seg is None:
            return
        # Compute final volume as float 0-1
        vol_255 = v.get_final_vol(self.run_gv)
        vol_f = vol_255 / 255.0
        # Pan: NP 0-64 → 0.0-1.0
        pan_f = v.np / 64.0
        freq_hz = float(v.frq)
        # Sample position: approximate based on elapsed ticks
        dt_ticks = self.abs_tick - v.trigger_abs_tick
        ticks_per_sec = self.tempo * 2.0 / 5.0
        dt_sec = dt_ticks / ticks_per_sec
        if v.smp_idx and v.smp_idx <= len(self.samples):
            smp = self.samples[v.smp_idx - 1]
            samp_pos = int(v.trigger_samp_pos + dt_sec * freq_hz)
            # Apply looping for position estimate
            if smp.has_loop and smp.loop_end > smp.loop_start:
                span = smp.loop_end - smp.loop_start
                if samp_pos >= smp.loop_end and span > 0:
                    excess = samp_pos - smp.loop_start
                    samp_pos = smp.loop_start + (excess % span)
        else:
            samp_pos = 0
        seg.tick_states.append((self.abs_tick, freq_hz, vol_f, pan_f, samp_pos))

    def _touch_voice(self, v: ITVoice):
        """Deactivate voice if it's become silent (MikIT ITVIRTCH::Touch).
        Called at start of tick, BEFORE column processing."""
        if not v.active:
            return
        if v.cut:
            v.cut = False
            v.active = False
            self._close_segment(v)
            return
        if v.nfc == 0 or v.sv == 0 or v.iv == 0:
            v.active = False
            self._close_segment(v)
            return
        # When a volume envelope has run to completion and its final value is 0
        # (key-off → sustain released → tail plays out → last point = 0),
        # deactivate the voice.  The env_tick_vol > 1 guard prevents killing a
        # voice at tick 0 before the envelope has had a chance to run.
        if not v.keyon and v.vev == 0 and v.env_tick_vol > 1:
            v.active = False
            self._close_segment(v)

    def _update_voice(self, v: ITVoice, col: ITColumn):
        """Update voice state (MikIT ITVIRTCH::Update).
        Called AFTER column processing each tick."""
        if v.kick and v.inst and v.smp_idx:
            v.reset_for_new_note(self.abs_tick)
            if v.smp_idx <= len(self.samples):
                v.sv = self.samples[v.smp_idx - 1].global_vol
            else:
                v.sv = 64
            v.iv = v.inst.global_vol
            v.nna = v.inst.nna
            v.kick = False
            self._start_segment(v, col)

        # Advance envelopes first (MikIT ITPROCESS::Process increments tick before read),
        # then read the value at the new tick position.
        v.advance_envelopes()
        v.update_vev()

        # Key-off → start fade
        if not v.keyon:
            vol_env = v.inst.vol_env if v.inst else None
            has_env = vol_env and vol_env.enabled and len(vol_env.points) >= 2
            if not has_env or (vol_env and vol_env.loop_enabled):
                v.dofade = True

        # Vol envelope done → fade
        if v.inst and v.inst.vol_env.enabled:
            if v.inst.vol_env.is_done(v.env_tick_vol, v.keyon):
                v.dofade = True

        # Fade
        if v.inst and v.dofade:
            fo = v.inst.fadeout
            if fo > 0:
                v.nfc = max(0, v.nfc - fo)

    def _process_tick(self):
        """Process one tick (MikIT MMODULE_IT::Update inner body)."""
        # 1. Touch: deactivate voices that died in the PREVIOUS tick
        for v in self.voices:
            self._touch_voice(v)

        # Process each column
        for ci in range(self.num_channels):
            col = self.columns[ci]
            col.own_vol = False
            col.own_frq = False
            col.own_ofs = False
            col.retrig = False
            col.tick = self.tick

            col.process_tick0(self.abs_tick)
            col.effects()

            if col.kick:
                if not col.retrig:
                    # Duplicate check
                    self._dup_check(col)
                    # NNA
                    if col.voice and col.voice.active:
                        self._do_nna(col)
                        new_v = self._find_or_steal_voice()
                    elif col.voice:
                        new_v = col.voice  # reuse inactive voice
                    else:
                        new_v = self._find_or_steal_voice()
                        if new_v is None:
                            new_v = self._alloc_voice()
                            self.voices.append(new_v)
                else:
                    new_v = col.voice

                if new_v:
                    # Close old segment if stealing
                    if new_v.vid in self._voice_segs and new_v.vid != (col.voice.vid if col.voice else -1):
                        self._close_segment(new_v)
                    # Clear stale col.voice on the previous owner so it doesn't overwrite
                    # this voice's state when that column is processed later in the same tick.
                    old_owner = new_v.owner
                    if 0 <= old_owner < len(self.columns) and self.columns[old_owner].voice is new_v:
                        self.columns[old_owner].voice = None
                    new_v.owner = ci
                    new_v.active = True
                    new_v.keyon = True
                    new_v.dofade = False
                    new_v.background = False
                    new_v.cut = False
                    new_v.note = col.note
                    new_v.inst_idx = col.inst
                    new_v.smp_idx = col.smp
                    new_v.inst = self.instruments[col.inst - 1] if col.inst and col.inst <= len(self.instruments) else None
                    new_v.smp = self.samples[col.smp - 1] if col.smp and col.smp <= len(self.samples) else None
                    new_v.start_offset = col.start_offset if col.own_ofs else 0
                    new_v.kick = True
                    col.voice = new_v

                col.kick = False

            if col.voice:
                v = col.voice
                v.cv = col.cv
                v.np = col.np
                v.vol = col.ovol if col.own_vol else col.vol
                v.frq = col.ofrq if col.own_frq else col.frq
                if v.frq < 50:
                    v.frq = 50

            col.post_effects()

        if self.pattern_del_rq:
            self.pattern_delay = self.pattern_del_rq
            self.pattern_del_rq = 0

        # 3. Update all active voices (envelopes, fade)
        for v in self.voices:
            if v.active:
                col_ref = self.columns[v.owner] if 0 <= v.owner < len(self.columns) else None
                self._update_voice(v, col_ref)

        # 4. Snapshot all active voices AFTER update
        for v in self.voices:
            if v.active and v.vid in self._voice_seg_id:
                col_ref = self.columns[v.owner] if 0 <= v.owner < len(self.columns) else None
                self._snapshot_voice(v, col_ref)

    def _load_row(self, pat_idx, row_idx):
        """Load row data into column cells."""
        if pat_idx >= len(self.patterns):
            return
        pat = self.patterns[pat_idx]
        if row_idx >= len(pat):
            return
        row = pat[row_idx]
        for ci in range(min(self.num_channels, len(row))):
            self.columns[ci].cell = row[ci]

    def run(self, max_ticks=None) -> List[VoiceSegment]:
        """
        Simulate the song and return all voice segments.
        max_ticks: safety limit (default = 30 minutes at 60 ticks/sec).
        """
        if max_ticks is None:
            max_ticks = 30 * 60 * 60  # 30 min @ 60 ticks/sec

        # Init state
        self.speed = self.initial_speed
        self.tempo = self.initial_tempo
        self.run_gv = self.global_volume
        self.tick = self.speed  # force row load on first Update
        self.row = -1
        self.abs_tick = 0
        self.song_pos = 0
        self.order_jump = 0
        self.break_to_row = 0
        self.pattern_delay = 0
        self.pattern_del_rq = 0

        # Init voices pool
        self.voices = [self._alloc_voice() for _ in range(self.MAX_VOICES)]
        for v in self.voices:
            v.active = False
            v.background = True

        # Init columns
        self.columns = []
        for ci in range(self.num_channels):
            col = ITColumn(ci, self)
            col.cp = self.channel_pan[ci] if ci < len(self.channel_pan) else 32
            col.np = col.cp
            col.cv = self.channel_vol[ci] if ci < len(self.channel_vol) else 64
            self.columns.append(col)

        # Clear segments
        self._voice_segs = {}
        self._finished_segs = []
        self.row_events = []

        # Find first valid order
        self.song_pos = 0
        while self.song_pos < len(self.orders) and self.orders[self.song_pos] >= 0xFE:
            self.song_pos += 1
        if self.song_pos >= len(self.orders):
            return []

        pat_no = self.orders[self.song_pos]
        if pat_no >= len(self.patterns):
            return []
        current_pat_rows = len(self.patterns[pat_no])

        songs_done = False
        while self.abs_tick < max_ticks and not songs_done:
            # Advance tick counter
            self.tick += 1
            if self.tick >= self.speed:
                self.tick = 0

                if self.pattern_delay:
                    self.pattern_delay -= 1
                    if self.pattern_delay == 0:
                        self.row += 1
                else:
                    self.row += 1

                # Handle pattern boundary / order jump / break
                if self.row >= current_pat_rows or self.order_jump or self.break_to_row:
                    if self.order_jump:
                        self.song_pos = (self.order_jump - 1) % len(self.orders)
                        self.order_jump = 0
                        if self.song_pos == 0:
                            songs_done = True
                            break
                    else:
                        self.song_pos += 1

                    # Skip marker orders
                    while self.song_pos < len(self.orders) and self.orders[self.song_pos] == 0xFE:
                        self.song_pos += 1

                    if self.song_pos >= len(self.orders) or self.orders[self.song_pos] == 0xFF:
                        songs_done = True
                        break

                    pat_no = self.orders[self.song_pos]
                    if pat_no >= len(self.patterns):
                        songs_done = True
                        break

                    if self.break_to_row:
                        self.row = max(0, self.break_to_row - 1)
                        self.break_to_row = 0
                    else:
                        self.row = 0

                    current_pat_rows = len(self.patterns[pat_no])

                # Load row data into columns
                self._load_row(pat_no, self.row)
                self.row_events.append((self.abs_tick, self.song_pos, self.row, pat_no))

            self._process_tick()
            self.abs_tick += 1

        # Close any still-open segments
        for v in self.voices:
            if v.vid in self._voice_segs:
                self._close_segment(v)

        return self._finished_segs

# ──────────────────────────────────────────────────────────────────────────────
# IT file loader (native, not through existing mod_player conversion)
# ──────────────────────────────────────────────────────────────────────────────

def load_it_native(filename: str) -> ITPlayer:
    """Parse an .it file and return a configured ITPlayer ready to run()."""
    with open(filename, 'rb') as f:
        blob = f.read()

    if blob[:4] != b'IMPM':
        raise ValueError("Not an IT file")

    player = ITPlayer()
    # Song name lives at offset 4 (26 bytes, null-padded). load_xm_native does
    # the equivalent for XM via XMFile; do the same here so the HTML segment
    # player can show the song title instead of falling back to filename.
    player.title = blob[4:4+26].rstrip(b'\x00 \r\n').decode('latin-1', errors='ignore')

    # Header
    ord_num = struct.unpack_from('<H', blob, 32)[0]
    ins_num = struct.unpack_from('<H', blob, 34)[0]
    smp_num = struct.unpack_from('<H', blob, 36)[0]
    pat_num = struct.unpack_from('<H', blob, 38)[0]
    cmwt    = struct.unpack_from('<H', blob, 42)[0]
    flags   = struct.unpack_from('<H', blob, 44)[0]
    player.linear_freq = bool(flags & 8)
    player.initial_speed = max(1, blob[50])
    player.initial_tempo = max(32, blob[51])
    player.global_volume = blob[48]
    player.mix_volume    = blob[49]

    chn_pan = blob[64:128]
    chn_vol = blob[128:192]

    cur = 192
    orders = list(blob[cur:cur+ord_num]); cur += ord_num
    ins_offsets = list(struct.unpack_from(f'<{ins_num}I', blob, cur)); cur += 4*ins_num
    smp_offsets = list(struct.unpack_from(f'<{smp_num}I', blob, cur)); cur += 4*smp_num
    pat_offsets = list(struct.unpack_from(f'<{pat_num}I', blob, cur)); cur += 4*pat_num

    player.orders = [b for b in orders if b != 0xFE or True]  # keep 0xFF end marker
    player.orders = orders  # raw including 0xFE skip and 0xFF end

    # Channel config
    def _count_chans():
        mx = 0
        lm = [0]*64
        for po in pat_offsets:
            if not po or po+8 > len(blob): continue
            length = struct.unpack_from('<H', blob, po)[0]
            data = blob[po+8:po+8+length]
            for i in range(64): lm[i] = 0
            n = len(data); c = 0
            while c < n:
                cv = data[c]; c += 1
                if cv == 0: continue
                ch = (cv-1) & 0x3F
                if ch+1 > mx: mx = ch+1
                if cv & 0x80:
                    if c >= n: break
                    mask = data[c]; c += 1; lm[ch] = mask
                else:
                    mask = lm[ch]
                if mask & 0x01: c += 1
                if mask & 0x02: c += 1
                if mask & 0x04: c += 1
                if mask & 0x08: c += 2
        return mx
    try:
        highest = _count_chans()
    except Exception:
        highest = 0
        for i in range(64):
            if (chn_pan[i] & 0x80) == 0: highest = i+1
    player.num_channels = max(4, min(64, highest)) if highest else 4

    player.channel_pan = []
    player.channel_vol = []
    for i in range(player.num_channels):
        p = chn_pan[i] & 0x7F if i < 64 else 32
        player.channel_pan.append(p)
        v = chn_vol[i] if i < 64 else 64
        player.channel_vol.append(v)

    # Load instruments
    player.instruments = []
    if cmwt >= 0x200:
        for off in ins_offsets:
            if not off or off+64 > len(blob) or blob[off:off+4] != b'IMPI':
                player.instruments.append(ITInstrument())
                continue
            inst = ITInstrument()
            inst.name = blob[off+32:off+58].rstrip(b'\x00').decode('latin-1', errors='ignore')
            inst.nna     = blob[off+17]
            inst.dct     = blob[off+18]
            inst.dca     = blob[off+19]
            inst.fadeout = struct.unpack_from('<H', blob, off+20)[0]
            # IT New Instrument Header offsets (IMPI):
            #   0x14-0x15: Fadeout
            #   0x16: PPS (Pitch-Pan Separation, signed -32..+32)
            #   0x17: PPC (Pitch-Pan Centre, note 0..119)
            #   0x18: Global Vol (0..128)
            #   0x19: DfP (Default Pan, bit7=override, bits0-6=pan 0..64)
            # Bug fix: pps/ppc were being read from off+26/+27 which is
            # RandomVolVariation/RandomPanVariation — for jeff.it those bytes
            # were 5 and 4, interpreted as PPS=5/PPC=4 caused every note above
            # note 4 to clamp pan to 64 (full right).
            inst.pps = struct.unpack_from('<b', blob, off+22)[0] if off+22 < len(blob) else 0
            inst.ppc = blob[off+23] if off+23 < len(blob) else 60
            inst.global_vol = blob[off+24] if off+24 < len(blob) else 128
            inst.dfp = blob[off+25] if off+25 < len(blob) else 0
            # Note→sample map (120 entries at +64)
            n2s = [blob[off+64+n*2+1] for n in range(120)]
            n2n = [blob[off+64+n*2+0] for n in range(120)]
            inst.note_to_sample = n2s
            inst.note_to_note   = n2n
            # Volume envelope at +304
            for env_attr, env_base in [('vol_env', off+304), ('pan_env', off+386), ('ptc_env', off+468)]:
                if env_base + 82 > len(blob): continue
                flg = blob[env_base]
                num = blob[env_base+1]
                env = ITEnvelope()
                env.enabled        = bool(flg & 0x01) and num > 0
                env.loop_enabled   = bool(flg & 0x02)
                env.sustain_enabled= bool(flg & 0x04)
                env.loop_start     = blob[env_base+2]
                env.loop_end       = blob[env_base+3]
                env.sus_start      = blob[env_base+4]
                env.sus_end        = blob[env_base+5]
                if env.enabled:
                    for p in range(min(num, 25)):
                        v = blob[env_base+6+p*3]
                        t = struct.unpack_from('<H', blob, env_base+6+p*3+1)[0]
                        env.points.append(ITEnvPoint(t, v))
                setattr(inst, env_attr, env)
            player.instruments.append(inst)
    else:
        # Old format / sample mode: create identity instruments
        for i in range(smp_num):
            inst = ITInstrument()
            inst.note_to_sample = [i+1]*120
            inst.note_to_note   = list(range(120))
            player.instruments.append(inst)

    # If not using instruments mode, create identity mapping
    if not (flags & 0x04):  # IT_USE_INST bit
        player.instruments = []
        for i in range(smp_num):
            inst = ITInstrument()
            inst.note_to_sample = [i+1]*120
            inst.note_to_note   = list(range(120))
            player.instruments.append(inst)

    # Load samples (metadata only — PCM data managed by existing mod_player.py)
    import numpy as np
    player.samples = []
    for so in smp_offsets:
        if not so or so+80 > len(blob) or blob[so:so+4] != b'IMPS':
            player.samples.append(ITSample())
            continue
        s = ITSample()
        # IMPS layout: +4=dos filename(12), +16=0, +17=GvL, +18=Flg, +19=Vol,
        # +20=name(26), +46=Cvt, +47=DfP, +48=Length, +52=LoopBeg, +56=LoopEnd,
        # +60=C5Speed, +64=SusLoopBeg, +68=SusLoopEnd, +72=SmpOffset, +76=ViS..ViT
        s.global_vol = blob[so+17] if so+17 < len(blob) else 64   # GvL 0..64
        flags_s      = blob[so+18]
        s.vol        = blob[so+19]                                  # Vol 0..64
        s.name       = blob[so+20:so+46].rstrip(b'\x00').decode('latin-1', errors='ignore')
        cvt          = blob[so+46]
        s.dfp        = blob[so+47] if so+47 < len(blob) else 0
        s.length     = struct.unpack_from('<I', blob, so+48)[0]
        s.loop_start = struct.unpack_from('<I', blob, so+52)[0]
        s.loop_end   = struct.unpack_from('<I', blob, so+56)[0]
        s.c5speed    = struct.unpack_from('<I', blob, so+60)[0]
        s.sus_loop_start = struct.unpack_from('<I', blob, so+64)[0]
        s.sus_loop_end   = struct.unpack_from('<I', blob, so+68)[0]
        s.has_loop    = bool(flags_s & 0x10)
        s.has_sus_loop= bool(flags_s & 0x40)
        s.bidi_loop   = bool(flags_s & 0x08)
        s.bidi_sus    = bool(flags_s & 0x20)
        s.vib_speed  = blob[so+76] if so+76 < len(blob) else 0
        s.vib_depth  = blob[so+77] if so+77 < len(blob) else 0
        s.vib_rate   = blob[so+78] if so+78 < len(blob) else 0
        s.vib_type   = blob[so+79] if so+79 < len(blob) else 0
        if not s.c5speed: s.c5speed = 8363
        player.samples.append(s)

    # Decode patterns (native IT cells, not MOD conversion)
    player.patterns = []
    for po in pat_offsets:
        if not po:
            player.patterns.append(_empty_it_pattern(64, player.num_channels))
            continue
        length   = struct.unpack_from('<H', blob, po)[0]
        num_rows = struct.unpack_from('<H', blob, po+2)[0]
        data = blob[po+8:po+8+length]
        player.patterns.append(_decode_it_pattern(data, num_rows, player.num_channels))

    return player


# ──────────────────────────────────────────────────────────────────────────────
# S3M loader
# ──────────────────────────────────────────────────────────────────────────────

# S3M note byte: high nibble = octave (0–7), low nibble = semitone (0–11).
# c2spd is the sample rate at S3M C-4 (byte 0x40, period 428 = MOD C-2 ref).
# IT's c5speed is Hz at IT note 60 (C-5 = period 428 in Amiga terms).
# Map: IT_note = oct*12 + semi + 12  (S3M oct 4 → IT note 60).
_S3M_NOTE_OFFSET = 12

def _s3m_note_to_it(note_byte):
    """Convert raw S3M note byte to IT note number (0-based, 0=C-0)."""
    if note_byte == 0xFF or note_byte == 0xFE:
        return 255 if note_byte == 0xFF else 254   # no-note / note-off
    oct_  = (note_byte >> 4) & 0x0F
    semi  = note_byte & 0x0F
    return max(0, min(119, oct_ * 12 + semi + _S3M_NOTE_OFFSET))

# S3M command numbers (1=A, 2=B, …) map 1:1 to IT effect numbers (eff=1 is A).
# Only special cases need adjustment; the mapping is almost fully transparent.
def _s3m_eff_to_it(cmd, param):
    """Return (it_eff, it_par) for an S3M (command, param) pair."""
    if cmd == 0:
        return 0, 0
    # S3M C (pattern break) stores the row in BCD; IT C uses plain binary.
    if cmd == 3:
        par = ((param >> 4) * 10) + (param & 0xF)
        return 3, min(par, 63)
    # S3M A (set speed) and T (set tempo) are separate commands; IT uses the
    # same numbering so pass through as-is (A=eff1 speed, T=eff20 tempo).
    return cmd, param


def load_s3m_native(filename: str) -> 'ITPlayer':
    """Parse an S3M file and return an ITPlayer ready to run()."""
    with open(filename, 'rb') as f:
        blob = f.read()

    if len(blob) < 96 or blob[28] != 0x1A or blob[44:48] != b'SCRM':
        raise ValueError("Not an S3M file")

    player = ITPlayer()
    player.linear_freq = False   # S3M uses Amiga-style log slides

    title = blob[0:28].rstrip(b'\x00').decode('latin-1', errors='ignore')

    ord_num  = struct.unpack_from('<H', blob, 32)[0]
    ins_num  = struct.unpack_from('<H', blob, 34)[0]
    pat_num  = struct.unpack_from('<H', blob, 36)[0]

    player.global_volume = blob[48]
    player.initial_speed = max(1, blob[49])
    player.initial_tempo = max(32, blob[50])
    player.run_gv        = player.global_volume

    ffi = blob[0x2A]
    samples_unsigned = (ffi != 1)

    channel_settings = list(blob[64:96])

    # Count active channels and build pan map
    active_chans = []
    for i, cs in enumerate(channel_settings):
        if cs < 16:
            active_chans.append(i)
    player.num_channels = max(4, len(active_chans))

    # S3M channel pan: bits 3-0 of channel_settings encode L/R.
    # Values 0-7 = left channels, 8-15 = right channels, ≥16 = disabled.
    player.channel_pan = []
    player.channel_vol = []
    for i in range(player.num_channels):
        cs = channel_settings[i] if i < 32 else 0x08
        if cs < 8:      # left
            player.channel_pan.append(0)
        elif cs < 16:   # right
            player.channel_pan.append(64)
        else:
            player.channel_pan.append(32)
        player.channel_vol.append(64)

    # Orders
    cur = 96
    orders_raw = list(blob[cur:cur+ord_num]); cur += ord_num
    player.orders = [o for o in orders_raw if o < 254]  # drop skip(254)/end(255)

    ins_paras = list(struct.unpack_from(f'<{ins_num}H', blob, cur)); cur += 2*ins_num
    pat_paras = list(struct.unpack_from(f'<{pat_num}H', blob, cur)); cur += 2*pat_num

    # Load samples → ITSample + identity ITInstrument
    import numpy as np
    player.samples     = []
    player.instruments = []
    for i, para in enumerate(ins_paras):
        smp  = ITSample()
        inst = ITInstrument()
        inst.note_to_sample = [i+1]*120
        inst.note_to_note   = list(range(120))

        if para:
            off = para * 16
            if off + 80 <= len(blob) and blob[off] == 1:  # type 1 = PCM sample
                memseg_hi = blob[off+13]
                memseg    = struct.unpack_from('<H', blob, off+14)[0]
                samp_off  = ((memseg_hi << 16) | memseg) * 16
                smp.length     = struct.unpack_from('<I', blob, off+16)[0]
                loop_beg       = struct.unpack_from('<I', blob, off+20)[0]
                loop_end       = struct.unpack_from('<I', blob, off+24)[0]
                smp.vol        = blob[off+28]
                smp.global_vol = 64
                flags          = blob[off+31]
                c2spd          = struct.unpack_from('<I', blob, off+32)[0]
                smp.c5speed    = c2spd if c2spd else 8363
                smp.name       = blob[off+48:off+76].rstrip(b'\x00').decode('latin-1', errors='ignore')
                if flags & 1:
                    smp.has_loop   = True
                    smp.loop_start = loop_beg
                    smp.loop_end   = loop_end
                if flags & 4:
                    smp.bidi_loop  = True
        player.samples.append(smp)
        player.instruments.append(inst)

    # Decode patterns
    player.patterns = []
    for para in pat_paras:
        if not para:
            player.patterns.append(_empty_it_pattern(64, player.num_channels))
            continue
        off = para * 16
        if off + 2 > len(blob):
            player.patterns.append(_empty_it_pattern(64, player.num_channels))
            continue
        plen = struct.unpack_from('<H', blob, off)[0]
        data = blob[off+2 : off+2+plen]
        player.patterns.append(_decode_s3m_pattern(data, 64, player.num_channels))

    return player


def _decode_s3m_pattern(data, num_rows, num_chans):
    """Decode S3M packed pattern into list[row][ch] of ITCell."""
    rows = _empty_it_pattern(num_rows, num_chans)
    pos = 0; row = 0
    n = len(data)
    while row < num_rows and pos < n:
        what = data[pos]; pos += 1
        if what == 0:
            row += 1
            continue
        ch = what & 31
        cell = ITCell()
        if what & 32:  # note + instrument
            if pos + 1 >= n: break
            nb   = data[pos]; pos += 1
            inst = data[pos]; pos += 1
            if nb < 0xF0:   # real note
                cell.note     = _s3m_note_to_it(nb)
                cell.has_note = True
            elif nb == 0xFE:
                cell.note = 254; cell.has_note = True   # note-cut
            elif nb == 0xFF:
                pass                                     # no note
            if inst:
                cell.inst = inst; cell.has_inst = True
        if what & 64:  # volume
            if pos >= n: break
            vol = data[pos]; pos += 1
            if vol <= 64:
                cell.vol = vol; cell.has_vol = True
        if what & 128:  # effect
            if pos + 1 >= n: break
            cmd  = data[pos]; pos += 1
            par  = data[pos]; pos += 1
            eff, epar = _s3m_eff_to_it(cmd, par)
            if eff:
                cell.eff = eff; cell.par = epar; cell.has_eff = True
        if ch < num_chans:
            rows[row][ch] = cell
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# MOD loader
# ──────────────────────────────────────────────────────────────────────────────

# ProTracker period table (C-0 .. B-3), no finetune.  Index = note 0..47.
# In MOD, C-2 (index 24) = "natural" pitch, mapping to IT C-5 (note 60).
# Offset = 60 - 24 = 36.
_MOD_AMIGA_PERIODS = [
    1712,1616,1525,1440,1357,1281,1209,1141,1077,1017, 961, 907,
     856, 808, 763, 720, 678, 640, 604, 570, 538, 508, 480, 453,
     428, 404, 381, 360, 339, 320, 302, 285, 269, 254, 240, 226,
     214, 202, 190, 180, 170, 160, 151, 143, 135, 127, 120, 113,
]
_MOD_NOTE_OFFSET = 36   # IT_note = MOD_period_index + 36

# c4speeds[finetune] — effective C-5 playback rate per finetune nibble.
_MOD_C4SPEEDS = [8363,8413,8463,8529,8581,8651,8723,8757,
                 7895,7941,7985,8046,8107,8169,8232,8280]

def _mod_period_to_it_note(period: int, finetune: int) -> int:
    """Convert a raw MOD period to an IT note number (0-based).
    Finds the closest entry in the finetune-adjusted period table."""
    if period <= 0:
        return 0
    ft = finetune & 0xF
    c5 = _MOD_C4SPEEDS[ft]
    # period_for_note = _MOD_AMIGA_PERIODS[i] * 8363 / c5
    best_note = 0; best_err = 1 << 30
    for i, p0 in enumerate(_MOD_AMIGA_PERIODS):
        adj = int(p0 * 8363 / c5)
        err = abs(adj - period)
        if err < best_err:
            best_err = err; best_note = i
    return best_note + _MOD_NOTE_OFFSET   # → IT note 36..83

def _mod_eff_to_it(eff, par):
    """Convert MOD (eff_nibble, param) to (it_eff, it_par).
    Returns (0, 0) if no IT equivalent."""
    if eff == 0x0:
        return (10, par) if par else (0, 0)   # J Arpeggio (skip if par==0)
    elif eff == 0x1: return (6,  par)          # F Porta up
    elif eff == 0x2: return (5,  par)          # E Porta down
    elif eff == 0x3: return (7,  par)          # G Tone porta
    elif eff == 0x4: return (8,  par)          # H Vibrato
    elif eff == 0x5: return (12, par)          # L Tone porta + vol slide
    elif eff == 0x6: return (11, par)          # K Vibrato + vol slide
    elif eff == 0x7: return (18, par)          # R Tremolo
    elif eff == 0x8:                           # Set pan → X (0-255 → 0-64)
        return (24, max(0, min(64, par >> 2)))
    elif eff == 0x9: return (15, par)          # O Sample offset
    elif eff == 0xA: return (4,  par)          # D Vol slide
    elif eff == 0xB: return (2,  par)          # B Jump to order
    elif eff == 0xC: return (0,  0)            # Set vol — handled as vol column
    elif eff == 0xD:                           # Pattern break (BCD in MOD)
        row = ((par >> 4) * 10) + (par & 0xF)
        return (3, min(row, 63))
    elif eff == 0xE:                           # Extended effects
        sub = (par >> 4) & 0xF
        val = par & 0xF
        if sub == 0x1: return (6,  0xF0 | val)   # Fine porta up
        elif sub == 0x2: return (5,  0xF0 | val)  # Fine porta down
        elif sub == 0x9: return (17, 0x00 | val)  # Retrig (Q0x)
        elif sub == 0xA: return (4,  0xF0 | val)  # Fine vol up
        elif sub == 0xB: return (4,  (val << 4) | 0xF)  # Fine vol down
        elif sub == 0xC: return (19, 0xC0 | val)  # S Cut note (SCx)
        elif sub == 0xD: return (19, 0xD0 | val)  # S Note delay (SDx)
        elif sub == 0x6: return (19, 0x60 | val)  # S Pattern loop (S6x)
        return (0, 0)
    elif eff == 0xF:
        if par == 0:   return (0,  0)
        elif par < 32: return (1,  par)   # A Set speed
        else:          return (20, par)   # T Set tempo
    return (0, 0)


def load_mod_native(filename: str) -> 'ITPlayer':
    """Parse a MOD file and return an ITPlayer ready to run()."""
    with open(filename, 'rb') as f:
        blob = f.read()

    if len(blob) < 1084:
        raise ValueError("File too short for MOD")

    player = ITPlayer()
    player.linear_freq = False   # MOD uses Amiga log slides

    # Determine channel count from signature at offset 1080
    sig = blob[1080:1084]
    if sig in (b'M.K.', b'M!K!', b'M&K!', b'N.T.', b'FLT4', b'4CHN'):
        num_ch = 4
    elif sig in (b'FLT8', b'OCTA', b'CD81', b'OKTA'):
        num_ch = 8
    elif sig[1:4] == b'CHN' and sig[0:1].isdigit():
        num_ch = int(sig[0:1])
    elif sig[2:4] in (b'CH', b'CN') and sig[0:2].isdigit():
        num_ch = int(sig[0:2])
    elif sig[:3] == b'TDZ' and sig[3:4].isdigit():
        num_ch = int(sig[3:4])
    else:
        num_ch = 4
    player.num_channels = num_ch

    # Standard MOD 4-channel LRRL pan layout (0=L, 64=R, 32=C)
    _pan4 = [0, 64, 64, 0]
    player.channel_pan = [_pan4[i % 4] for i in range(num_ch)]
    player.channel_vol = [64] * num_ch

    # Read 31-sample headers
    import numpy as np
    samples_meta = []
    cur = 20
    for i in range(31):
        sname  = blob[cur:cur+22].rstrip(b'\x00').decode('latin-1', errors='ignore')
        slen   = struct.unpack_from('>H', blob, cur+22)[0] * 2
        ft_raw = blob[cur+24] & 0x0F
        finetune = ft_raw if ft_raw <= 7 else ft_raw - 16
        vol    = blob[cur+25]
        rpt    = struct.unpack_from('>H', blob, cur+26)[0] * 2
        rlen   = struct.unpack_from('>H', blob, cur+28)[0] * 2
        if rlen <= 2: rlen = 0
        samples_meta.append({'name': sname, 'length': slen, 'finetune': ft_raw,
                              'volume': vol, 'repeat_point': rpt, 'repeat_length': rlen})
        cur += 30

    song_length = blob[cur]; cur += 1
    cur += 1   # restart byte
    orders_raw = list(blob[cur:cur+128]); cur += 128
    player.orders = orders_raw[:song_length]
    num_patterns = max(player.orders) + 1

    cur += 4   # skip signature

    # Build patterns
    player.patterns = []
    for _ in range(num_patterns):
        rows = _empty_it_pattern(64, num_ch)
        for row in range(64):
            for ch in range(num_ch):
                if cur + 4 > len(blob): break
                word = struct.unpack_from('>I', blob, cur)[0]; cur += 4
                snum   = ((word >> 24) & 0xF0) | ((word >> 12) & 0x0F)
                period = (word >> 16) & 0x0FFF
                eff    = (word >> 8)  & 0x0F
                par    = word & 0xFF

                cell = ITCell()
                if period:
                    # Determine note from period using this sample's finetune
                    ft = samples_meta[snum - 1]['finetune'] if 1 <= snum <= 31 else 0
                    it_note = _mod_period_to_it_note(period, ft)
                    cell.note = it_note; cell.has_note = True
                if snum:
                    cell.inst = snum; cell.has_inst = True
                if eff == 0xC:   # Set volume → volume column
                    cell.vol = min(64, par); cell.has_vol = True
                else:
                    it_eff, it_par = _mod_eff_to_it(eff, par)
                    if it_eff:
                        cell.eff = it_eff; cell.par = it_par; cell.has_eff = True
                rows[row][ch] = cell
        player.patterns.append(rows)

    # Scan first pattern for initial speed/tempo (MOD Fxx in row 0)
    player.initial_speed = 6
    player.initial_tempo = 125
    if player.orders and num_patterns > 0:
        p0 = player.patterns[player.orders[0]]
        for row in p0[:8]:
            for cell in row:
                if cell.has_eff and cell.eff == 1 and cell.par:   # A = speed
                    player.initial_speed = cell.par
                elif cell.has_eff and cell.eff == 20 and cell.par: # T = tempo
                    player.initial_tempo = cell.par

    # Load sample PCM data and build ITSample + identity ITInstrument
    player.samples     = []
    player.instruments = []
    for i, sm in enumerate(samples_meta):
        smp  = ITSample()
        inst = ITInstrument()
        inst.note_to_sample = [i+1]*120
        inst.note_to_note   = list(range(120))
        smp.c5speed    = _MOD_C4SPEEDS[sm['finetune'] & 0xF]
        smp.vol        = sm['volume']
        smp.global_vol = 64
        smp.name       = sm['name']
        smp.length     = sm['length']
        if sm['repeat_length'] > 0:
            smp.has_loop   = True
            smp.loop_start = sm['repeat_point']
            smp.loop_end   = sm['repeat_point'] + sm['repeat_length']
        if sm['length'] > 0 and cur + sm['length'] <= len(blob):
            cur += sm['length']   # skip raw PCM (not used by mikit_engine)
        player.samples.append(smp)
        player.instruments.append(inst)

    return player


# ──────────────────────────────────────────────────────────────────────────────
# XM loader
# ──────────────────────────────────────────────────────────────────────────────

def _xm_volcol_to_it(vcol):
    """Map XM volume-column byte to IT volume-column byte (255 = empty)."""
    if vcol == 0:          return 255
    if 16 <= vcol <= 80:   return vcol - 16            # set vol 0-64
    if 96 <= vcol <= 111:  return 75 + (vcol - 96)     # vol slide down
    if 112 <= vcol <= 127: return 65 + (vcol - 112)    # vol slide up
    if 128 <= vcol <= 143: return 95 + (vcol - 128)    # fine vol down
    if 144 <= vcol <= 159: return 85 + (vcol - 144)    # fine vol up
    if 160 <= vcol <= 175: return 115 + (vcol - 160)   # vibrato speed
    if 176 <= vcol <= 191: return 105 + (vcol - 176)   # vibrato depth
    if 192 <= vcol <= 207: return 128 + (vcol - 192) * 4  # set pan → IT 128-192
    if 240 <= vcol <= 255: return 193 + (vcol - 240)   # tone porta speed
    return 255

def _xm_eff_to_it(eff, par):
    """Map XM (effect_byte, param) to (it_eff, it_par)."""
    if eff == 0:    return (10, par) if par else (0, 0)   # arpeggio
    elif eff == 1:  return (6,  par)     # porta up
    elif eff == 2:  return (5,  par)     # porta down
    elif eff == 3:  return (7,  par)     # tone porta
    elif eff == 4:  return (27, par)     # XM vibrato (raw depth)
    elif eff == 5:  return (12, par)     # tone porta + vol slide
    elif eff == 6:  return (11, par)     # vibrato + vol slide (uses stored vibdpt)
    elif eff == 7:  return (18, par)     # tremolo
    elif eff == 8:  return (24, min(64, par >> 2))   # set pan (0-255 → 0-64)
    elif eff == 9:  return (15, par)     # sample offset
    elif eff == 10: return (4,  par)     # vol slide
    elif eff == 11: return (2,  par)     # jump to order
    elif eff == 12: return (0,  0)       # set vol — handled as vol column
    elif eff == 13:                      # pattern break (BCD param)
        row = ((par >> 4) * 10) + (par & 0xF)
        return (3, min(row, 63))
    elif eff == 14:                      # extended Exx
        sub = (par >> 4) & 0xF; val = par & 0xF
        if sub == 0x1: return (6,  0xF0 | val)
        elif sub == 0x2: return (5, 0xF0 | val)
        elif sub == 0x6: return (19, 0x60 | val)
        elif sub == 0x9: return (17, val)
        elif sub == 0xA: return (4,  0xF0 | val)
        elif sub == 0xB: return (4,  (val << 4) | 0xF)
        elif sub == 0xC: return (19, 0xC0 | val)
        elif sub == 0xD: return (19, 0xD0 | val)
        return (0, 0)
    elif eff == 15:                      # set speed / BPM
        if par == 0:    return (0, 0)
        elif par < 32:  return (1, par)
        else:           return (20, par)
    elif eff == 16: return (22, min(128, par * 2))   # G set global vol (XM 0-64 → IT 0-128)
    elif eff == 17: return (23, par)     # H global vol slide
    elif eff == 20: return (0,  0)       # K key-off (handled as note=255 in pattern decoder)
    elif eff == 21: return (0,  0)       # L set envelope pos — skip
    elif eff == 23: return (16, par)     # P pan slide
    elif eff == 25: return (17, par)     # R multi-retrig
    elif eff == 27: return (9,  par)     # T tremor
    elif eff == 30:                      # X extra-fine porta
        sub = (par >> 4) & 0xF; val = par & 0xF
        if sub == 1: return (6, 0xE0 | val)
        elif sub == 2: return (5, 0xE0 | val)
        return (0, 0)
    return (0, 0)


def _decode_xm_pattern_to_it(blob, start, pack_size, num_rows, num_chans):
    """Decode XM packed pattern data into list[row][ch] of ITCell."""
    rows = _empty_it_pattern(max(1, min(num_rows, 256)), num_chans)
    cur = start
    end = start + pack_size
    for row in range(min(num_rows, 256)):
        for ch in range(num_chans):
            if cur >= end:
                break
            b = blob[cur]; cur += 1
            note = inst = vcol = eff = par = 0
            if b & 0x80:   # packed cell
                if b & 0x01:
                    if cur >= end: break
                    note = blob[cur]; cur += 1
                if b & 0x02:
                    if cur >= end: break
                    inst = blob[cur]; cur += 1
                if b & 0x04:
                    if cur >= end: break
                    vcol = blob[cur]; cur += 1
                if b & 0x08:
                    if cur >= end: break
                    eff = blob[cur]; cur += 1
                if b & 0x10:
                    if cur >= end: break
                    par = blob[cur]; cur += 1
            else:           # full 5-byte cell
                note = b
                if cur + 3 >= end: break
                inst = blob[cur]; cur += 1
                vcol = blob[cur]; cur += 1
                eff  = blob[cur]; cur += 1
                par  = blob[cur]; cur += 1

            cell = ITCell()
            if note == 97:          # key-off note
                cell.note = 255; cell.has_note = True
            elif 1 <= note <= 96:
                cell.note = note - 1; cell.has_note = True  # XM 1-based → IT 0-based
            if inst:
                cell.inst = inst; cell.has_inst = True
            if vcol:
                iv = _xm_volcol_to_it(vcol)
                if iv != 255:
                    cell.vol = iv; cell.has_vol = True
            # XM effect C = set volume → vol column
            if eff == 12:
                cell.vol = min(64, par); cell.has_vol = True
            # XM effect K = key-off at tick par (K00 → immediate note-off)
            elif eff == 20:
                if par == 0:
                    cell.note = 255; cell.has_note = True
                else:
                    # Delayed key-off: fire at tick par (stored as eff=26 = XM K delayed)
                    cell.eff = 26; cell.par = par; cell.has_eff = True
            elif eff:
                it_eff, it_par = _xm_eff_to_it(eff, par)
                if it_eff:
                    cell.eff = it_eff; cell.par = it_par; cell.has_eff = True
            if ch < num_chans:
                rows[row][ch] = cell
    return rows


def load_xm_native(filename: str) -> 'ITPlayer':
    """Parse an XM file and return an ITPlayer ready to run()."""
    with open(filename, 'rb') as f:
        blob = f.read()
    if blob[:17] != b'Extended Module: ':
        raise ValueError("Not an XM file")

    player = ITPlayer()
    # XM song title at offset 17 (20 bytes, null/space-padded). Mirrors XMFile.
    player.title = blob[17:17+20].rstrip(b'\x00 \r\n').decode('latin-1', errors='ignore')

    hdr_size      = struct.unpack_from('<I', blob, 60)[0]
    song_len      = struct.unpack_from('<H', blob, 64)[0]
    player.num_channels   = struct.unpack_from('<H', blob, 68)[0]
    num_patterns  = struct.unpack_from('<H', blob, 70)[0]
    num_insts     = struct.unpack_from('<H', blob, 72)[0]
    flags         = struct.unpack_from('<H', blob, 74)[0]
    player.linear_freq    = bool(flags & 1)
    player.initial_speed  = max(1, struct.unpack_from('<H', blob, 76)[0])
    player.initial_tempo  = max(32, struct.unpack_from('<H', blob, 78)[0])

    order_table = list(blob[80:80 + song_len])
    player.orders = [b for b in order_table if b < num_patterns]
    if not player.orders:
        player.orders = [0]

    # XM channel panning: standard LRRL per group of 4
    _pan4 = [0, 64, 64, 0]
    player.channel_pan = [_pan4[i % 4] for i in range(player.num_channels)]
    player.channel_vol = [64] * player.num_channels

    # ── Patterns ──────────────────────────────────────────────────────────────
    cur = 60 + hdr_size
    player.patterns = []
    for _ in range(num_patterns):
        ph_size   = struct.unpack_from('<I', blob, cur)[0]
        num_rows  = struct.unpack_from('<H', blob, cur + 5)[0]
        pack_size = struct.unpack_from('<H', blob, cur + 7)[0]
        pat_start = cur + ph_size
        player.patterns.append(
            _decode_xm_pattern_to_it(blob, pat_start, pack_size, num_rows, player.num_channels))
        cur = pat_start + pack_size

    # ── Instruments + Samples ─────────────────────────────────────────────────
    global_smp_idx = 0
    player.samples     = []
    player.instruments = []

    for inst_i in range(num_insts):
        inst_size = struct.unpack_from('<I', blob, cur)[0]
        if inst_size == 0: inst_size = 29
        inst_name = blob[cur+4:cur+26].rstrip(b'\x00 \r\n').decode('latin-1', errors='ignore')
        num_smp_in_inst = struct.unpack_from('<H', blob, cur+27)[0] if cur+29 <= len(blob) else 0

        inst = ITInstrument()
        inst.name = inst_name
        inst.nna  = 0   # XM: cut on new note (no NNA)

        if num_smp_in_inst == 0:
            # Empty instrument — identity note mapping, no samples
            inst.note_to_sample = [0] * 120
            inst.note_to_note   = list(range(120))
            player.instruments.append(inst)
            cur += inst_size
            continue

        # Note-to-sample map: 96 entries at cur+33 (0-based within this instrument)
        xm_n2s = list(blob[cur+33:cur+33+96]) if cur+129 <= len(blob) else [0]*96

        # Volume envelope
        vol_env = ITEnvelope()
        if cur + 241 <= len(blob):
            vol_type     = blob[cur+233]
            fadeout_raw  = struct.unpack_from('<H', blob, cur+239)[0]
            if vol_type & 0x01:
                n_pts = min(blob[cur+225], 12)
                vol_env.enabled          = True
                vol_env.sustain_enabled  = bool(vol_type & 0x02)
                vol_env.loop_enabled     = bool(vol_type & 0x04)
                vol_env.sus_start        = blob[cur+227]
                vol_env.sus_end          = blob[cur+227]
                vol_env.loop_start       = blob[cur+228]
                vol_env.loop_end         = blob[cur+229]
                for p in range(n_pts):
                    x = struct.unpack_from('<H', blob, cur+129+p*4)[0]
                    y = struct.unpack_from('<H', blob, cur+131+p*4)[0]
                    vol_env.points.append(ITEnvPoint(x, min(y, 64)))
        else:
            fadeout_raw = 0
        inst.vol_env = vol_env

        # Panning envelope
        pan_env = ITEnvelope()
        if cur + 241 <= len(blob):
            pan_type = blob[cur+234]
            if pan_type & 0x01:
                n_pts = min(blob[cur+226], 12)
                pan_env.enabled         = True
                pan_env.sustain_enabled = bool(pan_type & 0x02)
                pan_env.loop_enabled    = bool(pan_type & 0x04)
                pan_env.sus_start       = blob[cur+230]
                pan_env.sus_end         = blob[cur+230]
                pan_env.loop_start      = blob[cur+231]
                pan_env.loop_end        = blob[cur+232]
                for p in range(n_pts):
                    x = struct.unpack_from('<H', blob, cur+177+p*4)[0]
                    y = struct.unpack_from('<H', blob, cur+179+p*4)[0]
                    # XM pan env values are 0-64 (center=32); store as-is for now
                    pan_env.points.append(ITEnvPoint(x, min(y, 64)))
        inst.pan_env = pan_env

        # Fadeout: XM 0..0xFFF per tick; IT nfc starts at 1024, so scale by /64.
        # When no vol env and fadeout=0, XM holds notes indefinitely after key-off
        # (NNA=cut handles eventual release); set a minimal non-zero fadeout so
        # envelope-done voices still decay naturally.
        if vol_env.enabled:
            inst.fadeout = max(1, fadeout_raw >> 6)
        else:
            # No vol env: key-off has no effect; voice held until next note on channel.
            inst.fadeout = 1   # negligible; NNA=cut ends it on next trigger

        inst.global_vol = 128

        # Sample headers + data
        samp_hdr_size = struct.unpack_from('<I', blob, cur+29)[0]
        sh_base = cur + inst_size
        samp_meta = []
        for s_i in range(num_smp_in_inst):
            sh = sh_base + s_i * samp_hdr_size
            if sh + 40 > len(blob): break
            length   = struct.unpack_from('<I', blob, sh)[0]
            loop_st  = struct.unpack_from('<I', blob, sh+4)[0]
            loop_len = struct.unpack_from('<I', blob, sh+8)[0]
            vol      = blob[sh+12]
            ftune    = struct.unpack('b', bytes([blob[sh+13]]))[0]   # signed -128..127
            stype    = blob[sh+14]
            pan      = blob[sh+15]   # 0-255, center=128
            rel_note = struct.unpack('b', bytes([blob[sh+16]]))[0]   # signed semitones
            sname    = blob[sh+18:sh+40].rstrip(b'\x00 \r\n').decode('latin-1', errors='ignore')
            samp_meta.append({'length': length, 'loop_st': loop_st, 'loop_len': loop_len,
                               'vol': vol, 'ftune': ftune, 'stype': stype, 'pan': pan,
                               'rel_note': rel_note, 'name': sname})

        data_cur = sh_base + num_smp_in_inst * samp_hdr_size
        inst_smp_globals = []   # 1-based global sample indices for this instrument
        for sm in samp_meta:
            data_cur += sm['length']   # skip raw PCM (VQ encoder reads from XMFile)
            smp = ITSample()
            smp.name = sm['name'] or inst_name
            smp.vol  = min(sm['vol'], 64)
            smp.global_vol = 64
            # c5speed from finetune only; rel_note goes into note_to_note[]
            # ftune is ±128 per semitone → cents = ftune/128 * 100
            # XM reference rate is at C-4 (IT note 48); IT reference is C-5 (IT note 60).
            # C-5 = C-4 * 2, so multiply by 2 to shift reference up one octave.
            smp.c5speed = max(256, int(16726.0 * (2.0 ** (sm['ftune'] / (128.0 * 12.0)))))
            # Loop (byte offsets → sample offsets)
            bps = 2 if (sm['stype'] & 0x10) else 1
            smp.length = sm['length'] // bps
            loop_type  = sm['stype'] & 0x03
            if loop_type:
                smp.has_loop   = True
                smp.loop_start = sm['loop_st'] // bps
                smp.loop_end   = (sm['loop_st'] + sm['loop_len']) // bps
                smp.bidi_loop  = (loop_type == 2)
            # XM samples always have a panning byte; apply it (bit7=override).
            # pan 0-255 → IT 0-64, center=128→32
            smp.dfp = 0x80 | min(64, sm['pan'] >> 2)
            player.samples.append(smp)
            inst_smp_globals.append(global_smp_idx + 1)  # 1-based
            global_smp_idx += 1

        # Build note_to_sample and note_to_note from XM note-to-sample map
        # XM notes are 0-based here (decoded from 96-entry table)
        for it_note in range(120):
            xm_idx = min(it_note, 95)   # clamp; XM only has 96 entries
            s_idx = xm_n2s[xm_idx] if xm_idx < len(xm_n2s) else 0
            if s_idx < len(samp_meta) and s_idx < len(inst_smp_globals):
                inst.note_to_sample[it_note] = inst_smp_globals[s_idx]
                # rel_note transposes the pitch without changing the note column value
                rn = samp_meta[s_idx]['rel_note']
                inst.note_to_note[it_note] = max(0, min(119, it_note + rn))
            else:
                inst.note_to_sample[it_note] = 0
                inst.note_to_note[it_note]   = it_note

        player.instruments.append(inst)
        cur = data_cur

    return player


def _empty_it_pattern(num_rows, num_chans):
    return [[ITCell() for _ in range(num_chans)] for _ in range(num_rows)]


def _decode_it_pattern(data, num_rows, num_chans):
    """Decode IT compressed pattern into list[row][ch] of ITCell."""
    rows = _empty_it_pattern(num_rows, num_chans)
    last_mask = [0]*64
    last_note = [0]*64
    last_inst = [0]*64
    last_vol  = [0xFF]*64
    last_eff  = [0]*64
    last_par  = [0]*64
    # IT effect memory
    last_D = [0]*64; last_E = [0]*64; last_F = [0]*64; last_G = [0]*64
    n = len(data); cur = 0; row = 0
    while row < num_rows and cur < n:
        chvar = data[cur]; cur += 1
        if chvar == 0:
            row += 1
            continue
        ch = (chvar - 1) & 0x3F
        if chvar & 0x80:
            if cur >= n: break
            mask = data[cur]; cur += 1
            last_mask[ch] = mask
        else:
            mask = last_mask[ch]
        note = inst = vol = eff = par = 0
        has_note = has_inst = has_vol = has_eff = False
        if mask & 0x01:
            note = data[cur]; cur += 1
            last_note[ch] = note
            has_note = True
        if mask & 0x02:
            inst = data[cur]; cur += 1
            last_inst[ch] = inst
            has_inst = True
        if mask & 0x04:
            vol = data[cur]; cur += 1
            last_vol[ch] = vol
            has_vol = True
        if mask & 0x08:
            eff = data[cur]; par = data[cur+1]; cur += 2
            last_eff[ch] = eff; last_par[ch] = par
            has_eff = True
        if mask & 0x10:
            note = last_note[ch]; has_note = True
        if mask & 0x20:
            inst = last_inst[ch]
            # Don't set has_inst for inherited — no envelope/vol reset
        if mask & 0x40:
            vol = last_vol[ch]; has_vol = True
        if mask & 0x80:
            eff = last_eff[ch]; par = last_par[ch]; has_eff = True

        if ch >= num_chans:
            continue

        # IT effect memory (D/E/F/G with par=0 → reuse last)
        if has_eff:
            if eff == 4 and par == 0 and last_D[ch]: par = last_D[ch]
            elif eff == 5 and par == 0 and last_E[ch]: par = last_E[ch]
            elif eff == 6 and par == 0 and last_F[ch]: par = last_F[ch]
            elif eff == 7 and par == 0 and last_G[ch]: par = last_G[ch]
            if par:
                if eff == 4: last_D[ch] = par
                elif eff == 5: last_E[ch] = par
                elif eff == 6: last_F[ch] = par
                elif eff == 7: last_G[ch] = par

        cell = ITCell(
            note=note, inst=inst, vol=vol, eff=eff, par=par,
            has_note=has_note, has_inst=has_inst, has_vol=has_vol, has_eff=has_eff,
        )
        rows[row][ch] = cell

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Timeline → GLSL encoder
# ──────────────────────────────────────────────────────────────────────────────

def encode_timeline_glsl(segments: List[VoiceSegment], ticks_per_sec: float) -> dict:
    """
    Convert voice segments to compact GLSL-ready arrays.

    Returns a dict with keys:
        num_segs         int
        seg_start_tick   list[int]
        seg_end_tick     list[int]
        seg_sample       list[int]   (0-based)
        seg_loop_start   list[int]
        seg_loop_end     list[int]
        seg_loop_type    list[int]
        seg_freq         list[float] per segment start freq in Hz
        seg_vol          list[float] per segment start vol 0-1
        seg_pan          list[float] per segment start pan 0-1
        seg_samp_pos     list[int]   sample position at start_tick
        seg_freq_mul     list[float] per-tick freq multiplier (1.0 = constant)
        seg_vol_delta    list[float] per-tick vol delta (0.0 = constant)
    """
    out = {
        'num_segs': 0,
        'seg_start_tick': [], 'seg_end_tick': [],
        'seg_sample': [],
        'seg_loop_start': [], 'seg_loop_end': [], 'seg_loop_type': [],
        'seg_freq': [], 'seg_vol': [], 'seg_pan': [],
        'seg_samp_pos': [],
        'seg_freq_mul': [], 'seg_vol_delta': [],
    }

    for seg in segments:
        if not seg.tick_states:
            continue

        t0, f0, _, p0, pos0 = seg.tick_states[0]

        # Peak volume: XM envelopes start at 0 (attack) and end at 0 (release), so
        # first/last give delta=0 → silence.  Use max volume as the baseline; vol_delta
        # is then the per-tick decay rate from peak to final.  Attack ramp is collapsed
        # to instant (cosmetically acceptable; the ramp is usually very short).
        v_peak = max(s[2] for s in seg.tick_states)

        freq_mul = 1.0
        vol_delta = 0.0
        if len(seg.tick_states) > 1:
            t_last, f_last, v_last, _, _ = seg.tick_states[-1]
            dt = t_last - t0
            if dt > 0 and f0 > 0:
                freq_mul = (f_last / f0) ** (1.0 / dt)
            if dt > 0:
                vol_delta = (v_last - v_peak) / dt

        end_tick = seg.end_tick if seg.end_tick >= 0 else (
            seg.tick_states[-1][0] + 1 if seg.tick_states else seg.start_tick + 1
        )

        out['seg_start_tick'].append(seg.start_tick)
        out['seg_end_tick'].append(end_tick)
        out['seg_sample'].append(seg.sample_idx)
        out['seg_loop_start'].append(seg.loop_start)
        out['seg_loop_end'].append(seg.loop_end)
        out['seg_loop_type'].append(seg.loop_type)
        out['seg_freq'].append(f0)
        out['seg_vol'].append(v_peak)
        out['seg_pan'].append(p0)
        out['seg_samp_pos'].append(pos0)
        out['seg_freq_mul'].append(freq_mul)
        out['seg_vol_delta'].append(vol_delta)

    out['num_segs'] = len(out['seg_start_tick'])
    return out


def timeline_to_glsl_arrays(tl: dict, ticks_per_sec: float) -> str:
    """
    Emit GLSL const array declarations from a timeline dict.
    Returns a GLSL string fragment to inject into the Common tab.
    """
    n = tl['num_segs']
    if n == 0:
        return "// timeline: no segments\nconst int TL_NUM_SEGS = 0;\n"

    def _ints(arr, name):
        vals = ', '.join(str(int(x)) for x in arr)
        return f"const int {name}[{n}] = int[]({vals});\n"

    def _floats(arr, name, scale=1.0):
        vals = ', '.join(f"{float(x)*scale:.6f}" for x in arr)
        return f"const float {name}[{n}] = float[]({vals});\n"

    lines = [f"const int TL_NUM_SEGS = {n};\n"]
    lines.append(f"const float TL_TICKS_PER_SEC = {ticks_per_sec:.4f};\n")
    lines.append(_ints(tl['seg_start_tick'],  'tlSegStart'))
    lines.append(_ints(tl['seg_end_tick'],    'tlSegEnd'))
    lines.append(_ints(tl['seg_sample'],      'tlSegSample'))
    lines.append(_ints(tl['seg_loop_start'],  'tlSegLoopSt'))
    lines.append(_ints(tl['seg_loop_end'],    'tlSegLoopEn'))
    lines.append(_ints(tl['seg_loop_type'],   'tlSegLoopTy'))
    lines.append(_floats(tl['seg_freq'],      'tlSegFreq'))
    lines.append(_floats(tl['seg_vol'],       'tlSegVol'))
    lines.append(_floats(tl['seg_pan'],       'tlSegPan'))
    lines.append(_ints(tl['seg_samp_pos'],    'tlSegPos'))
    lines.append(_floats(tl['seg_freq_mul'],  'tlSegFreqMul'))
    lines.append(_floats(tl['seg_vol_delta'], 'tlSegVolDelta'))
    return ''.join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# GLSL Sound shader fragment for timeline playback
# ──────────────────────────────────────────────────────────────────────────────

GLSL_TIMELINE_SOUND = r"""
// ── Timeline-based channel output ─────────────────────────────────────────
// Replaces the old pattern-simulation getChannelOutput.
// At time T: find all active segments, sum their sample outputs.

float tlReadSample(int smpIdx, int pos) {
    // Delegates to the existing VQ sample reader from Common.
    return getSampleAt(smpIdx, pos);
}

vec2 tlGetOutput(float T) {
    vec2 out_lr = vec2(0.0);
    int tick_T = int(T * TL_TICKS_PER_SEC);

    for (int i = 0; i < TL_NUM_SEGS; i++) {
        if (tick_T < tlSegStart[i] || tick_T >= tlSegEnd[i])
            continue;

        float seg_start_sec = float(tlSegStart[i]) / TL_TICKS_PER_SEC;
        float dt = T - seg_start_sec;
        if (dt < 0.0) continue;

        // Reconstruct frequency at time T
        float freq = tlSegFreq[i] * pow(tlSegFreqMul[i],
            (T - seg_start_sec) * TL_TICKS_PER_SEC);

        // Reconstruct volume at time T
        float vol = clamp(tlSegVol[i] + tlSegVolDelta[i] *
            (T - seg_start_sec) * TL_TICKS_PER_SEC, 0.0, 1.0);

        float pan = tlSegPan[i];  // 0=L, 1=R

        // Compute sample position
        int samp_pos = tlSegPos[i] + int(dt * freq);

        // Apply looping
        int ls = tlSegLoopSt[i], le = tlSegLoopEn[i];
        int lt = tlSegLoopTy[i];
        if (lt > 0 && le > ls && samp_pos >= le) {
            int span = le - ls;
            samp_pos = ls + ((samp_pos - ls) % span);
        }

        float s = tlReadSample(tlSegSample[i], samp_pos);
        float panR = 0.25 + 0.5 * pan;
        out_lr += s * vol * vec2(1.0 - panR, panR);
    }
    return out_lr;
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────


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
    Analyzes frequency content via FFT, finds best power-of-2 downsample factor,
    applies a windowed-sinc anti-alias LPF, then decimates.
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

    # Anti-alias LPF: 128-tap windowed-sinc at 0.9× the new Nyquist.
    # Blackman window gives >70 dB stopband — eliminates aliasing foldback
    # that the old raw-stride code left in (audible as clicks/crackling).
    M   = 64                      # half-length; total taps = 2M+1 = 129
    fc  = 0.9 / best_factor       # normalised cutoff (1.0 = original Nyquist)
    idx = np.arange(-M, M + 1, dtype=np.float32)
    h   = np.sinc(2.0 * fc * idx).astype(np.float32)
    h  *= np.blackman(2 * M + 1).astype(np.float32)
    h  /= h.sum()                 # unity DC gain

    d_filt    = np.convolve(d, h, mode='same')
    compressed = d_filt[::best_factor]
    compressed = np.clip(np.round(compressed).astype(np.int32), -128, 127).astype(np.int8)
    return best_factor, compressed


def _compress_loop_offsets(repeat_point, repeat_length, bw_factor):
    """
    Convert original loop start/length to compressed-domain coordinates.
    Rounds both endpoints to the nearest decimated-grid position so the
    seam maps to the closest actual stored sample, minimising discontinuity.
    Returns (loop_start_c, loop_len_c) in compressed units.
    If the compressed loop would be < 3 samples (GLSL won't loop), returns (0, 0).
    """
    if repeat_length <= 2 or bw_factor <= 1:
        return repeat_point // max(1, bw_factor), repeat_length
    lp = int(round(repeat_point / bw_factor))
    le = int(round((repeat_point + repeat_length) / bw_factor))
    ll = max(0, le - lp)
    if ll < 3:          # too short — GLSL loopLen > 2 guard would skip it
        return 0, 0
    return lp, ll


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

# ─────────────────────────────────────────────────────────────────────────────
#  XM / IT helpers — note→period conversion, IT compressed-sample decoder
# ─────────────────────────────────────────────────────────────────────────────

def _compute_xm_release_factor(env_pts, sus_pt):
    """Return 0..1 = time-weighted average vol of envelope points after
    sustain, divided by 64. Used by the stateless GLSL Sound tab to give
    XM key-off voices a "post-envelope hold" level — a reasonable
    approximation when we can't simulate the full envelope per tick.
    Returns 1.0 when no envelope (so the GLSL path collapses to "no drop").
    """
    if not env_pts or sus_pt + 1 >= len(env_pts):
        return 1.0
    rel = env_pts[sus_pt:]
    if len(rel) < 2:
        return 1.0
    total_dur = max(1, rel[-1][0] - rel[0][0])
    wsum = 0.0
    for i in range(len(rel) - 1):
        d = rel[i+1][0] - rel[i][0]
        wsum += d * (rel[i][1] + rel[i+1][1]) / 2.0
    return max(0.0, min(1.0, (wsum / total_dur) / 64.0))


def _note_to_mod_period(note_xm_or_it):
    """Convert an XM (1..96) or IT (0..119) note number to a MOD-style period.

    Convention: note that maps to MOD period 428 = the sample's "natural"
    pitch (1× source sample rate). Each octave halves/doubles the period:
        period = 428 * 2^((49 - note) / 12)

    XM note 49 = A-4 = native; IT note 49 also = native (after IT's note=60-
    means-middle-C convention, but mikIT remaps so 49 stays native too). The
    JS engine clamps periods to [113, 856] in `getEffectivePeriod`, so notes
    more than ±1 octave from the native pitch get squashed to the rail —
    acceptable for a Phase-1 port; high-pitched leads / sub-bass may sit at
    the wrong octave in the worst case.

    Returns 0 for "no note" (note > 120), else an int period.
    Negative notes (IT C-0..B-0 after the IT→XM shift) are still valid
    musical notes and map to large periods; we let those through.
    """
    if note_xm_or_it > 120:
        return 0
    import math as _m
    return max(1, int(round(428.0 * (2.0 ** ((49 - note_xm_or_it) / 12.0)))))


def _xm_finetune_to_mod_nibble(xm_finetune):
    """XM finetune is a signed -128..+127 byte (-128 = -1 semitone, +127 ≈
    +1 semitone in 1/128-semitone units). MOD finetune is a 4-bit nibble
    where 0..7 = +1..+7 cents-ish and 8..15 = -8..-1 (two's-complement).
    Squash to nearest MOD nibble.
    """
    n = int(round(xm_finetune / 16.0))
    n = max(-8, min(7, n))
    return n & 0xF


def _it_c5speed_to_mod_nibble(c5_speed):
    """IT samples carry a C-5 playback rate (Hz). Convert that to a MOD
    finetune nibble using the same 16-entry c4speeds table the engine uses
    (the c4speeds[ft] * 428 / period frequency formula). MOD period 428
    corresponds to IT C-5 in our note→period mapping (IT note 60 → 428),
    so c5_speed maps DIRECTLY onto the c4_speeds table — no /2 conversion.
    The previous /2 bug made every IT sample default to nibble 8 (= c4=7895)
    instead of nibble 0 (= 8363) for the standard c5_speed=8363, dropping
    pitch by ~5% on every IT sample. Default 0 if input is invalid.
    """
    if c5_speed <= 0:
        return 0
    c4_speeds = [8363, 8413, 8463, 8529, 8581, 8651, 8723, 8757,
                 7895, 7941, 7985, 8046, 8107, 8169, 8232, 8280]
    target = float(c5_speed)
    best = 0; best_err = 1e30
    for i, c4 in enumerate(c4_speeds):
        err = abs(c4 - target)
        if err < best_err:
            best_err = err; best = i
    return best


def _it_decompress_sample(packed_bytes, num_samples, is_16bit, it215_delta):
    """Decompress an IT-packed sample block.

    Direct port of mikIT's MCONVERT::ReadITCompress8/ReadITCompress16
    (mod/mikIT/mdriver.cpp). Each block is prefixed by a 16-bit packed-byte
    count; we accept the packed_bytes as a single block (caller handles
    multi-block reads when num_samples > 0x4000 for 16-bit / 0x8000 for
    8-bit). Returns a list of int16 samples (already sign-extended;
    converted to 16-bit by left-shift 8 for 8-bit inputs to match mikIT's
    output convention).

    `it215_delta` corresponds to mikIT's `it215` flag — when True (IT 2.15+),
    the decoder accumulates twice (d1 + d2). For older IT files (which is
    most), it215_delta=False and the output is just the running sum d1.
    """
    out = []
    pos_byte = 0
    pos_bit = 0
    n_bytes = len(packed_bytes)
    bits = 17 if is_16bit else 9
    d1 = 0
    d2 = 0

    def _read_bits(n):
        nonlocal pos_byte, pos_bit
        x = 0
        have = 0
        while n > 0:
            if pos_byte >= n_bytes:
                # Out of input: return zeros (matches mikIT's `buf=0` fallback)
                buf_avail = 8 - pos_bit
                take = min(n, buf_avail)
                # contribute zeros of width 'take'
                pos_bit += take
                if pos_bit >= 8:
                    pos_byte += pos_bit // 8
                    pos_bit %= 8
                n -= take
                have += take
                continue
            buf = packed_bytes[pos_byte] >> pos_bit
            buf_avail = 8 - pos_bit
            take = min(n, buf_avail)
            x |= (buf & ((1 << take) - 1)) << have
            pos_bit += take
            if pos_bit >= 8:
                pos_byte += pos_bit // 8
                pos_bit %= 8
            n -= take
            have += take
        return x

    width = bits
    new_count = 0
    while len(out) < num_samples:
        needbits = (4 if is_16bit else 3) if new_count else width
        x = _read_bits(needbits)
        if new_count:
            new_count = 0
            x += 1
            if x >= width:
                x += 1
            width = x
            continue
        if is_16bit:
            if width < 7:
                if x == (1 << (width - 1)):
                    new_count = 1
                    continue
            elif width < 17:
                y = (0xFFFF >> (17 - width)) - 8
                if y < x <= y + 16:
                    x -= y
                    width = x if x < width else x + 1
                    continue
            elif width == 17:
                if x & 0x10000:
                    width = (x + 1) & 0xFF
                    continue
            else:
                return out  # error in compressed data
        else:  # 8-bit
            if width < 7:
                if x == (1 << (width - 1)):
                    new_count = 1
                    continue
            elif width < 9:
                y = (0xFF >> (9 - width)) - 4
                if y < x <= y + 8:
                    x -= y
                    if x >= width:
                        x += 1
                    width = x
                    continue
            elif width < 10:
                if x >= 0x100:
                    width = x - 0x100 + 1
                    continue
            else:
                return out

        # Sign-extend x as `width`-bit signed (matches mikIT's
        # `((BYTE)(x<<(8-bits)))>>(8-bits)` arithmetic-shift trick).
        # When width >= base bits, there's no truncation in C — it just
        # interprets the low `base` bits as signed via the BYTE/SWORD wrap.
        if is_16bit:
            ext = width if width < 16 else 16
        else:
            ext = width if width < 8 else 8
        if x & (1 << (ext - 1)):
            x -= (1 << ext)
        if is_16bit:
            d1 = (d1 + x) & 0xFFFF
            if d1 >= 0x8000: d1 -= 0x10000
            d2 = (d2 + d1) & 0xFFFF
            if d2 >= 0x8000: d2 -= 0x10000
            out.append(d2 if it215_delta else d1)
        else:
            d1 = (d1 + x) & 0xFF
            if d1 >= 0x80: d1 -= 0x100
            d2 = (d2 + d1) & 0xFF
            if d2 >= 0x80: d2 -= 0x100
            out.append(d1 << 8)
    return out


def _xm_or_it_effect_to_mod(eff, param, is_xm=True):
    """Translate XM/IT effect+param to MOD-compatible (effect_nibble, param).

    XM effect bytes 0x00..0x0F are MOD-compatible (pass through). Letters
    G..Z (encoded as eff=0x10..0x23 in the file = 'G'-55..'Z'-55) are XM-
    specific and have no MOD equivalent → return (0, 0) and they'll be
    counted as "skipped".

    IT uses letters A..Z (encoded as eff=1..26) — same scheme as S3M (which
    IT inherits from). The translation table below is inlined here so
    mod_player.py is fully self-contained (no s3m2mod.py dependency).
    """
    if eff == 0:
        return (0, param)
    if is_xm:
        # XM 0x0..0xF map directly to MOD 0..F.
        if eff <= 0xF:
            return (eff, param)
        # XM-specific letters → no-op for Phase 1.
        return (0, 0)

    # ── IT (S3M-style letter effects, 1=A .. 26=Z) → MOD numeric. ──
    # Same table-of-contents as the well-known S3M→MOD effect map. xx=param,
    # x=high nibble, y=low nibble. Anything without a clean MOD equivalent
    # returns (0, 0) and gets counted as a "skipped" cell.
    cmd = eff - 1            # 0=A, 1=B, …, 25=Z
    hi = (param >> 4) & 0xF
    lo =  param       & 0xF

    if cmd == 0:   return (0xF, param)                     # A: speed
    if cmd == 1:   return (0xB, param)                     # B: position jump
    if cmd == 2:                                           # C: pattern break
        row = min(param, 63)
        return (0xD, ((row // 10) << 4) | (row % 10))      # binary→BCD
    if cmd == 3:                                           # D: vol slide
        if lo == 0xF and hi > 0:  return (0xE, 0xA0 | hi)  # fine up
        if hi == 0xF and lo > 0:  return (0xE, 0xB0 | lo)  # fine down
        return (0xA, param)
    if cmd == 4:                                           # E: porta down
        if hi in (0xE, 0xF):  return (0xE, 0x20 | lo)      # (extra-)fine
        return (0x2, param)
    if cmd == 5:                                           # F: porta up
        if hi in (0xE, 0xF):  return (0xE, 0x10 | lo)
        return (0x1, param)
    if cmd == 6:   return (0x3, param)                     # G: tone porta
    if cmd == 7:   return (0x4, param)                     # H: vibrato
    if cmd == 8:   return (0, 0)                           # I: tremor (no MOD eq)
    if cmd == 9:   return (0x0, param)                     # J: arpeggio
    if cmd == 10:                                          # K: vib + volslide
        if lo == 0xF and hi > 0:  return (0xE, 0xA0 | hi)
        if hi == 0xF and lo > 0:  return (0xE, 0xB0 | lo)
        return (0x6, param)
    if cmd == 11:                                          # L: porta + volslide
        if lo == 0xF and hi > 0:  return (0xE, 0xA0 | hi)
        if hi == 0xF and lo > 0:  return (0xE, 0xB0 | lo)
        return (0x5, param)
    if cmd == 14:  return (0x9, param)                     # O: sample offset
    if cmd == 16:  return (0xE, 0x90 | lo)                 # Q: retrigger
    if cmd == 17:  return (0x7, param)                     # R: tremolo
    if cmd == 18:                                          # S: special subcmds
        return {
            0x3: (0xE, 0x40 | lo),    # S3 vibrato waveform → E4x
            0x4: (0xE, 0x70 | lo),    # S4 tremolo waveform → E7x
            0x6: (0xE, 0x60 | lo),    # S6 pattern loop     → E6x
            0xB: (0xE, 0x60 | lo),    # SB pattern loop     → E6x
            0xC: (0xE, 0xC0 | lo),    # SC note cut         → ECx
            0xD: (0xE, 0xD0 | lo),    # SD note delay       → EDx
            0xE: (0xE, 0xE0 | lo),    # SE pattern delay    → EEx
        }.get(hi, (0, 0))
    if cmd == 19:  return (0xF, param)                     # T: tempo (Fxx with xx>=0x20)
    if cmd == 20:  return (0xE, 0x40 | lo)                 # U: fine vibrato → E4x (approx)
    if cmd == 21:  return (0, 0)                           # V: global vol — no MOD eq
    if cmd == 23:  return (0x8, param)                     # X: set panning (param 0..FF)
    return (0, 0)                                          # M, N, P, W, Y, Z: no MOD eq


# ─────────────────────────────────────────────────────────────────────────────
#  XM PARSER (FastTracker 2 — Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

class XMFile:
    """FastTracker 2 .xm parser.

    Phase-1 scope: load header + flatten instruments to one sample each
    (first sample only) + read variable-row patterns (clamped/truncated to
    64 for engine compatibility) + delta-decode 8/16-bit samples (16-bit
    downconverted to 8-bit). Effect translation: 0x0..0xF pass through;
    XM-specific letter effects (Gxx, Hxx, Kxx, Lxx, Pxx, Rxx, Txx, Xxx) are
    silently dropped with a count printed once.

    The output object exposes the same shape as MODFile so the existing
    engine/ encoder/ HTML player code path needs no changes:
        title, num_channels, num_patterns, samples[], patterns[][][],
        song_positions[], initial_speed, initial_tempo
    """
    def __init__(self, filename):
        self.filename = filename
        self.title = ""
        self.num_channels = 4
        self.num_patterns = 0
        self.samples = []
        self.patterns = []
        self.song_positions = []
        self.initial_speed = 6
        self.initial_tempo = 125
        self.is_s3m = False         # for downstream code that flags S3M-isms
        self._unsupported_effects = 0
        self._parse(filename)

    def _parse(self, filename):
        with open(filename, 'rb') as f:
            blob = f.read()
        if blob[:17] != b'Extended Module: ':
            raise ValueError("Not an XM file (missing 'Extended Module: ' signature)")

        self.title = blob[17:17+20].rstrip(b'\x00 \r\n').decode('latin-1', errors='ignore')

        # ── Header (fields are little-endian) ──
        # offset 58: version (2B); 60: header size (4B); 64: song length (2B);
        # 66: restart pos (2B); 68: num channels (2B); 70: num patterns (2B);
        # 72: num instruments (2B); 74: flags (2B); 76: tempo/speed (2B);
        # 78: BPM (2B); 80: pattern order table (256B).
        version = struct.unpack_from('<H', blob, 58)[0]
        hdr_size = struct.unpack_from('<I', blob, 60)[0]
        song_len = struct.unpack_from('<H', blob, 64)[0]
        # restart   = struct.unpack_from('<H', blob, 66)[0]   # unused (Phase 1)
        self.num_channels = struct.unpack_from('<H', blob, 68)[0]
        num_patterns      = struct.unpack_from('<H', blob, 70)[0]
        num_instruments   = struct.unpack_from('<H', blob, 72)[0]
        # flags     = struct.unpack_from('<H', blob, 74)[0]   # bit 0 = linear freq (TODO Phase 2)
        self.initial_speed = max(1, struct.unpack_from('<H', blob, 76)[0])
        self.initial_tempo = max(32, struct.unpack_from('<H', blob, 78)[0])

        order_table = list(blob[80:80+song_len])
        self.song_positions = [b for b in order_table if b < num_patterns]
        if not self.song_positions:
            self.song_positions = [0]
        self.num_patterns = num_patterns

        # XM stores up to 256 patterns, but pattern indices in the order
        # table address into that pool. We must build all `num_patterns`
        # slots even if some are unreferenced, so engine indexing matches.

        # ── Patterns ──
        # File pointer starts at offset (60 + hdr_size). Each pattern has a
        # small header followed by `packed_size` bytes of compressed notes.
        cur = 60 + hdr_size
        self.patterns = []
        for pat_i in range(num_patterns):
            ph_size = struct.unpack_from('<I', blob, cur)[0]
            # ph[0..3]=size, ph[4]=packing, ph[5..6]=numrows (LE), ph[7..8]=packsize (LE)
            num_rows  = struct.unpack_from('<H', blob, cur+5)[0]
            pack_size = struct.unpack_from('<H', blob, cur+7)[0]
            pat_data_start = cur + ph_size
            pat = self._decode_pattern(blob, pat_data_start, pack_size,
                                       num_rows, self.num_channels)
            # Clamp rows to 64 for engine compatibility (truncate).
            if len(pat) > 64:
                pat = pat[:64]
            elif len(pat) < 64:
                # Pad with empty rows so engine's fixed-64 indexing is safe.
                empty_row = [{'sample':0,'period':0,'effect':0,'param':0,'vol_col':0}
                             for _ in range(self.num_channels)]
                pat = pat + [list(empty_row) for _ in range(64 - len(pat))]
            self.patterns.append(pat)
            cur = pat_data_start + pack_size

        # ── Instruments → flat samples list ──
        # JS engine uses 6-bit instrument numbers (bits 0-5 of sample byte),
        # supporting up to 63 instruments. Bits 6-7 are reserved for note-cut
        # and key-off flags respectively.
        self.samples = [self._empty_sample() for _ in range(63)]
        flatten_count = min(num_instruments, 63)
        warn_skipped_inst = max(0, num_instruments - 63)
        for inst_i in range(num_instruments):
            inst_size = struct.unpack_from('<I', blob, cur)[0]
            inst_name = blob[cur+4:cur+4+22].rstrip(b'\x00 \r\n').decode('latin-1', errors='ignore')
            num_samples_in_inst = struct.unpack_from('<H', blob, cur+27)[0]

            if num_samples_in_inst == 0:
                # Empty instrument slot — advance by inst_size only.
                if inst_i < flatten_count:
                    self.samples[inst_i] = self._empty_sample(inst_name)
                cur += inst_size
                continue

            # XMPATCHHEADER continues at cur+29, then sample headers follow
            # at (cur + inst_size). Each sample header is 40 bytes; sample
            # data follows after ALL sample headers.
            # Volume envelope (XM "vol_type" semantics):
            #   @+225  vol_pts_n     (1)  number of envelope points (≤12)
            #   @+227  sustain_pt    (1)  point index where env pauses while keyed-on
            #   @+228  loop_st_pt    (1)  loop start point index
            #   @+229  loop_en_pt    (1)  loop end point index
            #   @+233  vol_type      (1)  bit0=on, bit1=sustain, bit2=loop
            #   @+239  fadeout       (2)  per-tick volfade decrement (0..0xFFF)
            # Each env point is 4 bytes (x:2 LE, y:2 LE) at offsets +129..+177.
            vol_type = blob[cur+233] if inst_size > 233 else 0
            fadeout  = struct.unpack_from('<H', blob, cur+239)[0] if inst_size > 240 else 0
            xm_env_on    = bool(vol_type & 0x01)
            xm_env_sus   = bool(vol_type & 0x02)
            xm_env_loop  = bool(vol_type & 0x04)
            xm_env_pts = []
            xm_env_sus_pt = 0
            xm_env_loop_st = 0
            xm_env_loop_en = 0
            if xm_env_on and inst_size > 240:
                vol_pts_n      = min(blob[cur+225], 12)
                xm_env_sus_pt  = blob[cur+227]
                xm_env_loop_st = blob[cur+228]
                xm_env_loop_en = blob[cur+229]
                for p in range(vol_pts_n):
                    x = struct.unpack_from('<H', blob, cur+129+p*4)[0]
                    y = struct.unpack_from('<H', blob, cur+131+p*4)[0]
                    xm_env_pts.append([x, y])
            samp_hdr_size = struct.unpack_from('<I', blob, cur+29)[0]
            sh_base = cur + inst_size
            # Read all sample headers for this instrument
            samp_meta = []
            for s_i in range(num_samples_in_inst):
                sh = sh_base + s_i * samp_hdr_size
                length    = struct.unpack_from('<I', blob, sh+0)[0]
                loop_st   = struct.unpack_from('<I', blob, sh+4)[0]
                loop_len  = struct.unpack_from('<I', blob, sh+8)[0]
                vol       = blob[sh+12]
                ftune     = struct.unpack('b', bytes([blob[sh+13]]))[0]  # signed
                stype     = blob[sh+14]
                # pan       = blob[sh+15]                               # ignored Phase 1
                rel_note  = struct.unpack('b', bytes([blob[sh+16]]))[0]  # signed
                # reserved  = blob[sh+17]
                sname     = blob[sh+18:sh+40].rstrip(b'\x00 \r\n').decode('latin-1', errors='ignore')
                samp_meta.append(dict(length=length, loop_st=loop_st,
                                      loop_len=loop_len, vol=vol, ftune=ftune,
                                      stype=stype, rel_note=rel_note,
                                      name=sname))
            # Sample data follows after all headers; length is in BYTES
            # (not samples) for both 8 and 16-bit forms.
            data_cur = sh_base + num_samples_in_inst * samp_hdr_size
            for sm in samp_meta:
                raw = blob[data_cur:data_cur + sm['length']]
                data_cur += sm['length']
                sm['raw'] = raw

            # Phase 1: only flatten the first sample of each instrument.
            sm0 = samp_meta[0]
            decoded = self._decode_xm_sample(sm0['raw'], sm0['stype'])
            # Convert byte-domain loop points to sample-domain (16-bit halves
            # the index space).
            byte_per_samp = 2 if (sm0['stype'] & 0x10) else 1
            loop_st_smp  = sm0['loop_st']  // byte_per_samp
            loop_len_smp = sm0['loop_len'] // byte_per_samp
            length_smp   = sm0['length']   // byte_per_samp
            if not (sm0['stype'] & 0x03):
                # No loop set — clear loop fields so engine doesn't loop.
                loop_st_smp = 0
                loop_len_smp = 0

            # Combine XM finetune (-128..+127) and rel_note (signed semitones)
            # into MOD's 4-bit finetune. The rel_note semitone shift can't be
            # encoded in MOD finetune; it gets approximated by "always
            # transposing the note column by rel_note semitones" at convert
            # time — but since we don't know which channel will play this
            # sample at codegen time, we punt and store rel_note in a side
            # attribute the (currently MOD-centric) engine ignores.
            mod_ft = _xm_finetune_to_mod_nibble(sm0['ftune'])

            if inst_i < flatten_count:
                # XM raw samples are SIGNED; engine wants signed int8 too.
                slot = self._empty_sample(sm0['name'] or inst_name)
                slot.update(dict(
                    name=sm0['name'] or inst_name,
                    length=length_smp,
                    finetune=mod_ft,
                    volume=min(sm0['vol'], 64),    # MOD vol range is 0..64
                    repeat_point=loop_st_smp,
                    repeat_length=loop_len_smp,
                    data=np.frombuffer(bytes(decoded), dtype=np.int8),
                    _xm_rel_note=sm0['rel_note'],  # for note period biasing
                    # XM envelope/fadeout — only used when env is enabled.
                    # Engine treats fadeout=0 with env-off as hard cut on
                    # key-off; non-zero fadeout drives ghost-voice decay
                    # at fadeout/65536 of full vol per tick (mikIT/openMPT
                    # convention; we scale that into the engine's vol units
                    # at dispatch time).
                    fadeout=fadeout if xm_env_on else 0,
                    # Full XM volume envelope state — JS engine simulates per
                    # tick. env_pts is empty when env is off; engine treats
                    # that as constant vol = 64. sus_pt holds env_x while
                    # voice is keyed on; loop_st/en defines the loop range
                    # active when env_loop is set.
                    env_pts=xm_env_pts if xm_env_on else [],
                    env_sus=xm_env_sus,
                    env_loop=xm_env_loop,
                    env_sus_pt=xm_env_sus_pt,
                    # XM sustain is a single hold point — model it as a
                    # zero-length sustain "loop" (start==end) so the unified
                    # IT-style advance code in the engine still produces
                    # the same hold-at-point behavior.
                    env_sus_en=xm_env_sus_pt,
                    env_loop_st=xm_env_loop_st,
                    env_loop_en=xm_env_loop_en,
                    # release_factor: time-weighted avg of post-sustain env
                    # points / 64 (0..1). Used by the stateless GLSL Sound
                    # tab to approximate the envelope release "hold" level
                    # without per-voice state. Defaults to 1.0 (no drop) for
                    # samples without envelope.
                    release_factor=_compute_xm_release_factor(
                        xm_env_pts if xm_env_on else [], xm_env_sus_pt),
                ))
                self.samples[inst_i] = slot

            cur = data_cur

        if warn_skipped_inst:
            print(f"   ⚠️  XM has {num_instruments} instruments — engine cap is 63; "
                  f"skipped {warn_skipped_inst} (instruments {63}+)")
        if self._unsupported_effects:
            print(f"   ⚠️  XM has {self._unsupported_effects} XM-specific effect cells "
                  f"(G/H/K/L/P/R/T/X) — silently dropped (Phase 1)")

        # Apply each sample's _xm_rel_note as a bias to its period during
        # cell construction — we do that in _decode_pattern by remembering
        # the active sample number and biasing the note. But that requires
        # forward-knowing the sample. Cleanest: shift inside the cell after
        # the fact, using the sample's _xm_rel_note attribute.
        for pat in self.patterns:
            for row in pat:
                for ch in row:
                    if ch['sample'] > 0 and ch['period'] > 0:
                        s = self.samples[ch['sample']-1]
                        rn = s.get('_xm_rel_note', 0)
                        if rn:
                            # Each semitone = 2^(1/12) period ratio.
                            ch['period'] = max(1, int(round(
                                ch['period'] * (2.0 ** (-rn / 12.0)))))

    @staticmethod
    def _empty_sample(name=""):
        return {
            'name': name, 'length': 0, 'finetune': 0, 'volume': 64,
            'repeat_point': 0, 'repeat_length': 0,
            'data': np.zeros(0, dtype=np.int8),
            '_xm_rel_note': 0,
        }

    def _decode_pattern(self, blob, start, pack_size, num_rows, num_chan):
        """Decode XM packed pattern data into [row][ch] = {sample,period,effect,param} cells.
        See XM agent report §2 for the bit-7 prefix scheme."""
        rows = []
        if pack_size == 0:
            empty_row = [{'sample':0,'period':0,'effect':0,'param':0,'vol_col':0}
                         for _ in range(num_chan)]
            return [list(empty_row) for _ in range(num_rows)]
        cur = start
        for _r in range(num_rows):
            row = []
            for _c in range(num_chan):
                if cur >= start + pack_size:
                    note = inst = vol = eff = par = 0
                else:
                    prefix = blob[cur]; cur += 1
                    if prefix & 0x80:
                        note = blob[cur] if prefix & 0x01 else 0
                        if prefix & 0x01: cur += 1
                        inst = blob[cur] if prefix & 0x02 else 0
                        if prefix & 0x02: cur += 1
                        vol  = blob[cur] if prefix & 0x04 else 0
                        if prefix & 0x04: cur += 1
                        eff  = blob[cur] if prefix & 0x08 else 0
                        if prefix & 0x08: cur += 1
                        par  = blob[cur] if prefix & 0x10 else 0
                        if prefix & 0x10: cur += 1
                    else:
                        # Uncompressed: prefix IS the note byte.
                        note = prefix
                        inst = blob[cur]; cur += 1
                        vol  = blob[cur]; cur += 1
                        eff  = blob[cur]; cur += 1
                        par  = blob[cur]; cur += 1

                # Map XM cell → MOD cell.
                period = _note_to_mod_period(note) if (1 <= note <= 96) else 0
                # Note 97 = key off — encoded by setting bit 7 of the sample
                # byte. The engine reads bit 7 to dispatch the active voice
                # to a ghost (with fadeout-based decay if env is on, hard
                # cut otherwise). We retain the instrument bits so the
                # engine can look up the sample's fadeout, but skip the
                # period (no retrigger).
                keyoff = (note == 97)
                # Translate effect.
                m_eff, m_par = _xm_or_it_effect_to_mod(eff, par, is_xm=True)
                if eff != 0 and m_eff == 0 and m_par == 0:
                    self._unsupported_effects += 1

                # Volume column handling.  XM vol-col byte ranges:
                #   0x10-0x50 = set volume 0-64
                #   0x60-0x6F = vol slide down 0-15 (regular, skips tick 0)
                #   0x70-0x7F = vol slide up  0-15 (regular, skips tick 0)
                #   0x80-0x8F = fine vol down  0-15 (tick 0 only)
                #   0x90-0x9F = fine vol up    0-15 (tick 0 only)
                #   (vibrato, pan, tone-porta: Phase 1 drop)
                cell_vol_col = 0
                if vol >= 0x10 and vol <= 0x50:
                    set_vol = vol - 0x10
                    if m_eff == 0 and m_par == 0:
                        m_eff = 0xC
                        m_par = set_vol
                    elif set_vol > 0:
                        # Vol-col coexists with effect (e.g. tone portamento).
                        # Capture separately — same fix as ITFile (line ~4436).
                        cell_vol_col = set_vol
                elif vol >= 0x60 and vol <= 0x6F and m_eff == 0 and m_par == 0:
                    val = vol - 0x60
                    if val > 0:
                        m_eff = 0xA; m_par = val          # A0N = vol slide down
                elif vol >= 0x70 and vol <= 0x7F and m_eff == 0 and m_par == 0:
                    val = vol - 0x70
                    if val > 0:
                        m_eff = 0xA; m_par = val << 4     # AN0 = vol slide up
                elif vol >= 0x80 and vol <= 0x8F and m_eff == 0 and m_par == 0:
                    val = vol - 0x80
                    if val > 0:
                        m_eff = 0xE; m_par = 0xB0 | val   # EBx = fine vol dn
                elif vol >= 0x90 and vol <= 0x9F and m_eff == 0 and m_par == 0:
                    val = vol - 0x90
                    if val > 0:
                        m_eff = 0xE; m_par = 0xA0 | val   # EAx = fine vol up

                samp_byte = inst & 0x3F   # 6 bits: instruments 0-63
                if keyoff:
                    samp_byte |= 0x80     # bit 7 = key-off (note-cut uses bit 6)
                    period = 0
                row.append({
                    'sample':  samp_byte,
                    'period':  period,
                    'effect':  m_eff,
                    'param':   m_par,
                    'vol_col': cell_vol_col,
                })
            rows.append(row)
        return rows

    def _decode_xm_sample(self, raw, stype):
        """Δ-decode XM sample data (always delta-encoded). 16-bit samples are
        downconverted to 8-bit (high byte only) for engine compatibility."""
        if not raw:
            return b''
        out = bytearray()
        if stype & 0x10:  # 16-bit
            acc = 0
            n = len(raw) // 2
            for i in range(n):
                d = struct.unpack('<h', raw[i*2:i*2+2])[0]
                acc = (acc + d) & 0xFFFF
                if acc >= 0x8000: acc -= 0x10000
                # downconvert to int8 by taking high byte (signed)
                hi = (acc >> 8) & 0xFF
                if hi >= 0x80: hi -= 0x100
                out.append(hi & 0xFF)
        else:             # 8-bit
            acc = 0
            for b in raw:
                d = struct.unpack('b', bytes([b]))[0]
                acc = (acc + d) & 0xFF
                if acc >= 0x80: acc -= 0x100
                out.append(acc & 0xFF)
        return bytes(out)


# ─────────────────────────────────────────────────────────────────────────────
#  IT PARSER (Impulse Tracker — Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

class ITFile:
    """Impulse Tracker .it parser.

    Phase-1 scope: header + samples (8/16-bit, signed/unsigned, IT-packed
    decompression, NOT stereo) + uncompressed patterns + 'instrument-as-
    sample' flattening (use sample[0] of each instrument's note-map). IT-
    specific letter effects translate via the inline IT-effect table (IT inherits
    S3M's command set 1:1 plus extras we drop). Output object matches
    MODFile shape — same downstream code path applies.
    """
    def __init__(self, filename):
        self.filename = filename
        self.title = ""
        self.num_channels = 0
        self.num_patterns = 0
        self.samples = []
        self.patterns = []
        self.song_positions = []
        self.initial_speed = 6
        self.initial_tempo = 125
        self.is_s3m = False
        self._unsupported_effects = 0
        self._parse(filename)

    def _parse(self, filename):
        with open(filename, 'rb') as f:
            blob = f.read()
        if blob[:4] != b'IMPM':
            raise ValueError("Not an IT file (missing 'IMPM' signature)")

        # ── Fixed header (192 bytes) ──
        self.title = blob[4:4+26].rstrip(b'\x00 \r\n').decode('latin-1', errors='ignore')
        ord_num = struct.unpack_from('<H', blob, 32)[0]
        ins_num = struct.unpack_from('<H', blob, 34)[0]
        smp_num = struct.unpack_from('<H', blob, 36)[0]
        pat_num = struct.unpack_from('<H', blob, 38)[0]
        cmwt    = struct.unpack_from('<H', blob, 42)[0]
        # flags   = struct.unpack_from('<H', blob, 44)[0]  # not used Phase 1
        self.initial_speed = max(1, blob[50])
        self.initial_tempo = max(32, blob[51])
        # IT global volume (0..128) and mix volume (0..128). openMPT applies
        # both as multiplicative gain reducers — mix_vol is the MASTER level
        # the user set in the tracker UI ("output level"), distinct from
        # global_vol (which can be slid via 'V' effect during play). For
        # GADGET.IT mix_vol=48 (~37%) — without applying this, my engine
        # plays the song ~3× louder than openMPT.
        self.global_volume = blob[48]    # 0..128 (default 128 = unity)
        self.mix_volume    = blob[49]    # 0..128 (master/preamp)

        chn_pan = blob[64:128]

        # ── Order list (orders, then sample/pattern offset tables) ──
        cur = 192
        orders = list(blob[cur:cur+ord_num])
        cur += ord_num
        ins_offsets = struct.unpack_from(f'<{ins_num}I', blob, cur); cur += 4*ins_num
        smp_offsets = struct.unpack_from(f'<{smp_num}I', blob, cur); cur += 4*smp_num
        pat_offsets = struct.unpack_from(f'<{pat_num}I', blob, cur); cur += 4*pat_num

        # Channel count: scan all pattern data for the highest channel
        # index that's actually used. The header pan flags ('disabled bit'
        # 0x80, 'no channel' 0xFF) are routinely wrong — many IT files
        # have all 64 channels marked enabled even when only 8-18 are
        # used. Cap at 32 for the engine. Falls back to the header pan
        # scan if pre-decoding the patterns fails.
        try:
            highest = self._scan_pattern_max_channel(blob, pat_offsets)
        except Exception:
            highest = 0
            for i, p in enumerate(chn_pan):
                if (p & 0x80) == 0: highest = i + 1
        self.num_channels = max(4, min(32, highest)) if highest else 4
        self.channel_settings = [(0 if (chn_pan[i] & 0x40) == 0 else 1)
                                 for i in range(self.num_channels)]
        # IT per-channel default pan (0..64; 0=L, 32=center, 64=R). Bit 7
        # (0x80) is the MUTE flag; values 0..64 use bit 6 (0x40) as part
        # of the pan value, NOT a flag — masking with 0x3F clips 64 → 0
        # which silently broke channels panned hard-right. Use 0x7F.
        self.channel_pan = [chn_pan[i] & 0x7F for i in range(self.num_channels)]

        self.song_positions = [b for b in orders
                                if b not in (0xFE, 0xFF) and b < pat_num]
        if not self.song_positions:
            self.song_positions = [0]
        self.num_patterns = pat_num

        # ── Instrument headers (cmwt >= 0x200 = "new" format) ──
        # We extract NNA, DCT, DCA, fadeout and the note→sample table.
        # Older IT files (cmwt < 0x200) and "sample-mode" files (no
        # IMPI signature at the offset) get a default identity inst→
        # sample mapping with NNA=cut. inst_table is keyed 1-based to
        # match pattern cell instrument numbers.
        self.inst_table = {}
        if cmwt >= 0x200 and ins_num > 0:
            for i, off in enumerate(ins_offsets):
                if off == 0 or off + 64 > len(blob): continue
                if blob[off:off+4] != b'IMPI': continue
                nna     = blob[off+17]                                      # 0=cut 1=cont 2=noteoff 3=notefade
                dct     = blob[off+18]                                      # 0=off 1=note 2=sample 3=instrument
                dca     = blob[off+19]                                      # 0=cut 1=noteoff 2=notefade
                fadeout = struct.unpack_from('<H', blob, off+20)[0]
                # IMPI offset +24: GlobalVolume (0..128, default 128). Per
                # IT spec, the played voice volume is sample.vol * inst.gv /
                # 128. Without this, openMPT-mixed instruments at GV=90 (=
                # 70%) play at 100% in our engine, contributing to the
                # uniformly-loud-mix problem on songs like GADGET.IT.
                inst_gv = blob[off+24] if off+24 < len(blob) else 128
                # Note→sample mapping at offset 64: 120 entries × (note,sample)
                # Each entry is 2 bytes: [byte0=transposed note, byte1=sample].
                n2s = [blob[off + 64 + n*2 + 1] for n in range(120)]   # sample
                n2n = [blob[off + 64 + n*2 + 0] for n in range(120)]   # transposed note
                inst_name = blob[off+32:off+58].rstrip(b'\x00 \r\n').decode('latin-1', errors='ignore')
                # ── IT volume envelope ──────────────────────────────────
                # IMPI@+304: Volume envelope (82 bytes). Layout:
                #   @0  Flg (bit0=on, bit1=loop, bit2=sustain)
                #   @1  Num (point count, 0..25)
                #   @2  LpB / @3 LpE (loop begin/end point INDICES)
                #   @4  SLB / @5 SLE (sustain loop begin/end point INDICES)
                #   @6.. 25 nodes × 3 bytes (UBYTE value, UWORD tick LE)
                # Without parsing this, instruments named "Sinewave-envelope"
                # (gadget.it inst 27) play with no env shape — every staccato
                # note hits at full vol with no release tail, the same XM
                # symptom from earlier. Reuse the engine's XM envelope path.
                env_pts = []
                env_on = env_sus = env_loop = False
                env_sus_pt = env_loop_st = env_loop_en = 0
                # IMPI envelope offsets (mikIT mmod_it1.cpp ITINSTRUMENT::Load):
                # IMPI@4..16 filename, @17..30 NNA/DCT/DCA/FadeOut/PPS/PPC/
                # GbV/DfP/RV/RP/TrkVers/NoS, @31 pad, @32..57 name, @58..63 pad,
                # @64..303 notesample (120×2), @304..385 VOL ENV, @386..467 PAN
                # ENV, @468..549 PITCH ENV. Each envelope is 82 bytes:
                #   @0 Flg, @1 Num, @2 LpB, @3 LpE, @4 SLB, @5 SLE,
                #   @6.. 25 nodes × 3 bytes (UBYTE value, UWORD tick LE),
                #   @81 trailing pad.
                env_off = off + 304   # IMPI vol-env start (mikIT spec)
                env_sus_en = 0   # IT-only: sustain loop END point (SLE)
                if env_off + 82 <= len(blob):
                    flg = blob[env_off + 0]
                    num = blob[env_off + 1]
                    if (flg & 0x01) and num > 0:
                        env_on        = True
                        env_loop      = bool(flg & 0x02)
                        env_sus       = bool(flg & 0x04)
                        env_loop_st   = blob[env_off + 2]
                        env_loop_en   = blob[env_off + 3]
                        # IT sustain is a LOOP between SLB and SLE points,
                        # not a single hold (mikIT mmod_it1.cpp:228 — when
                        # tick > SLE, tick=SLB while keyOn). Store both.
                        env_sus_pt    = blob[env_off + 4]    # SLB
                        env_sus_en    = blob[env_off + 5]    # SLE
                        # 25 points × 3 bytes; only Num are valid
                        for p in range(min(num, 25)):
                            v = blob[env_off + 6 + p*3 + 0]   # signed-ish 0..64 for vol
                            t = struct.unpack_from('<H', blob, env_off + 6 + p*3 + 1)[0]
                            env_pts.append([t, v])
                self.inst_table[i+1] = dict(
                    name=inst_name, nna=nna, dct=dct, dca=dca,
                    fadeout=fadeout, note_to_sample=n2s, note_to_note=n2n,
                    global_volume=inst_gv,
                    env_pts=env_pts, env_on=env_on, env_sus=env_sus,
                    env_loop=env_loop, env_sus_pt=env_sus_pt,
                    env_sus_en=env_sus_en,
                    env_loop_st=env_loop_st, env_loop_en=env_loop_en,
                )

        # ── Samples ──
        # Each sample is loaded into a slot. The IT pattern instrument byte
        # uses 6-bit mask (0x3F = 63 max) plus we may have files with up to
        # 99 instrument slots per the IT spec. Lifted cap from 31 to 99 so
        # jeff.it (40 samples) and other rich IT files work in the segment
        # player; the legacy JS engine can address up to 63 via its 0x3F
        # mask, so files needing 64-98 only work via the segment player.
        IT_SAMPLE_CAP = 99
        self.samples = [self._empty_sample() for _ in range(IT_SAMPLE_CAP)]
        slot_count = min(smp_num, IT_SAMPLE_CAP)
        warn_skipped = max(0, smp_num - IT_SAMPLE_CAP)
        compressed_skipped = 0
        for s_i in range(smp_num):
            so = smp_offsets[s_i]
            if so == 0 or so + 80 > len(blob):
                continue
            if blob[so:so+4] != b'IMPS':
                continue
            sname  = blob[so+20:so+20+26].rstrip(b'\x00 \r\n').decode('latin-1', errors='ignore')
            flags  = blob[so+18]
            vol    = blob[so+19]
            cvt    = blob[so+46]
            length = struct.unpack_from('<I', blob, so+48)[0]
            loop_b = struct.unpack_from('<I', blob, so+52)[0]
            loop_e = struct.unpack_from('<I', blob, so+56)[0]
            c5spd  = struct.unpack_from('<I', blob, so+60)[0]
            data_off = struct.unpack_from('<I', blob, so+72)[0]

            is_16bit    = bool(flags & 0x02)
            is_compressed = bool(flags & 0x08)
            is_signed   = bool(cvt & 0x01)
            is_stereo   = bool(flags & 0x04)
            has_loop    = bool(flags & 0x10)
            byte_per_samp = 2 if is_16bit else 1
            channels    = 2 if is_stereo else 1

            if is_stereo:
                # Phase 1: skip stereo samples (rare; would need L+R merge)
                compressed_skipped += 1
                continue
            if length == 0 or data_off == 0:
                continue
            # Read raw / decompress
            if is_compressed:
                raw_decoded = self._read_it_compressed_sample(
                    blob, data_off, length, is_16bit, cmwt >= 0x215)
            else:
                # Uncompressed: data_off is absolute byte offset into file.
                # Length is in SAMPLES (not bytes) for IT.
                nbytes = length * byte_per_samp
                raw = blob[data_off:data_off+nbytes]
                raw_decoded = self._read_it_uncompressed_sample(
                    raw, length, is_16bit, is_signed)

            # Downconvert 16-bit → 8-bit (signed) for engine compatibility.
            int8_data = bytearray()
            if is_16bit:
                for v in raw_decoded:
                    if v >= 0x8000: v -= 0x10000
                    hi = (v >> 8)
                    if hi < -128: hi = -128
                    if hi > 127: hi = 127
                    int8_data.append(hi & 0xFF)
            else:
                # raw_decoded is already 16-bit shifted (left << 8) by the
                # decoder for uniformity — take high byte.
                for v in raw_decoded:
                    if v >= 0x8000: v -= 0x10000
                    hi = (v >> 8)
                    if hi < -128: hi = -128
                    if hi > 127: hi = 127
                    int8_data.append(hi & 0xFF)

            ftune_nibble = _it_c5speed_to_mod_nibble(c5spd)

            # IT loop range is in samples; if no loop, clear.
            if not has_loop:
                loop_b = 0
                loop_e = 0
            loop_len = max(0, loop_e - loop_b)

            if s_i < slot_count:
                self.samples[s_i] = {
                    'name': sname, 'length': length, 'finetune': ftune_nibble,
                    'volume': min(vol, 64), 'repeat_point': loop_b,
                    'repeat_length': loop_len,
                    'data': np.frombuffer(bytes(int8_data), dtype=np.int8),
                    '_xm_rel_note': 0,
                    # NNA fields populated by the annotation pass below.
                    'nna': 0, 'dct': 0, 'dca': 0, 'fadeout': 0,
                    # Raw IT c5_speed — engine prefers this over the lossy
                    # finetune-nibble lookup. Many IT samples (drums, leads,
                    # sinewaves) ship with c5_speed far above MOD's 8000-8800
                    # finetune range; the table-lookup path squashed them by
                    # multiple octaves. Engine uses `c5_speed * 428 / period`
                    # directly when this field is present and >0.
                    'c5_speed': c5spd,
                    # XM/IT envelope fields — populated from instrument
                    # data during annotation pass below. Empty here so
                    # samples without an instrument-driven envelope behave
                    # as straight-through (full vol, no release tail).
                    'env_pts': [], 'env_sus': False, 'env_loop': False,
                    'env_sus_pt': 0, 'env_sus_en': 0,
                    'env_loop_st': 0, 'env_loop_en': 0,
                }

        # ── Annotate samples with NNA/DCT/DCA/fadeout + apply inst gvol ──
        # For each instrument, look at every (note → sample) entry; when a
        # sample is referenced by an instrument, copy that instrument's
        # NNA settings AND apply its global volume to the sample's default
        # volume. When multiple instruments reference the same sample
        # (e.g. GADGET.IT has both inst 17 "Sinewave" with NO envelope and
        # inst 27 "Sinewave-envelope" both pointing to sample 17), prefer
        # the instrument that actually carries an envelope or fadeout —
        # otherwise the song-critical envelope gets silently dropped and
        # the sinewave plays as a continuous tone forever (Inspector Gadget
        # main melody → suspended low tone bug).
        # Sort: env-bearing instruments first, then by fadeout, then by idx
        # for determinism.
        _ord_insts = sorted(
            self.inst_table.items(),
            key=lambda kv: (
                -int(bool(kv[1].get('env_on') and kv[1].get('env_pts'))),
                -int(kv[1].get('fadeout', 0)),
                kv[0]))
        _annotated = set()
        for inst_idx, inst in _ord_insts:
            inst_gv = inst.get('global_volume', 128)
            for samp_n in inst['note_to_sample']:
                if 1 <= samp_n <= IT_SAMPLE_CAP and samp_n not in _annotated:
                    s = self.samples[samp_n - 1]
                    if s['nna'] == 0 and s['dct'] == 0 and s['fadeout'] == 0:
                        s['nna']     = inst['nna']
                        s['dct']     = inst['dct']
                        s['dca']     = inst['dca']
                        s['fadeout'] = inst['fadeout']
                    # Scale sample default volume by instrument global vol
                    # (IT spec: voice_vol = sample.vol * inst.gv / 128).
                    if inst_gv < 128 and s['volume'] > 0:
                        s['volume'] = max(1, (s['volume'] * inst_gv) // 128)
                    # Copy envelope from instrument if active. The engine
                    # already has the per-tick env-advance + sustain-loop
                    # path used for XM; this just hands IT data to it.
                    if inst.get('env_on') and inst.get('env_pts'):
                        s['env_pts']     = inst['env_pts']
                        s['env_sus']     = inst.get('env_sus', False)
                        s['env_loop']    = inst.get('env_loop', False)
                        s['env_sus_pt']  = inst.get('env_sus_pt', 0)
                        s['env_sus_en']  = inst.get('env_sus_en', 0)
                        s['env_loop_st'] = inst.get('env_loop_st', 0)
                        s['env_loop_en'] = inst.get('env_loop_en', 0)
                        # Compute release_factor for the GLSL fallback path
                        # (used when env can't be simulated stateful).
                        s['release_factor'] = _compute_xm_release_factor(
                            inst['env_pts'], inst.get('env_sus_pt', 0))
                    _annotated.add(samp_n)

        if warn_skipped:
            print(f"   ⚠️  IT has {smp_num} samples — cap is {IT_SAMPLE_CAP}; "
                  f"skipped {warn_skipped} (samples {IT_SAMPLE_CAP}+)")
        if compressed_skipped:
            print(f"   ⚠️  IT has {compressed_skipped} stereo sample(s) — "
                  f"silently dropped (Phase 1)")
        if self.inst_table:
            nna_counts = {0:0, 1:0, 2:0, 3:0}
            for s in self.samples:
                if s['length'] > 0:
                    nna_counts[s['nna']] = nna_counts.get(s['nna'], 0) + 1
            non_cut = nna_counts[1] + nna_counts[2] + nna_counts[3]
            if non_cut:
                names = {0:'cut', 1:'continue', 2:'noteoff', 3:'notefade'}
                summary = ', '.join(f"{names[k]}={v}" for k, v in sorted(nna_counts.items()) if v)
                print(f"   🎼 IT NNA distribution across samples: {summary}")

        # ── Patterns ──
        self.patterns = []
        for pat_i in range(pat_num):
            po = pat_offsets[pat_i]
            if po == 0:
                self.patterns.append(self._empty_pattern(64))
                continue
            length = struct.unpack_from('<H', blob, po)[0]
            num_rows = struct.unpack_from('<H', blob, po+2)[0]
            data = blob[po+8:po+8+length]
            pat = self._decode_pattern(data, num_rows)
            # Engine wants 64 rows per pattern.
            if len(pat) > 64:
                pat = pat[:64]
            elif len(pat) < 64:
                empty_row = [{'sample':0,'period':0,'effect':0,'param':0}
                             for _ in range(self.num_channels)]
                pat = pat + [list(empty_row) for _ in range(64 - len(pat))]
            self.patterns.append(pat)

        if self._unsupported_effects:
            print(f"   ⚠️  IT had {self._unsupported_effects} effect cells "
                  f"with no MOD equivalent — silently dropped (Phase 1)")

    @staticmethod
    def _empty_sample(name=""):
        return {
            'name': name, 'length': 0, 'finetune': 0, 'volume': 64,
            'repeat_point': 0, 'repeat_length': 0,
            'data': np.zeros(0, dtype=np.int8),
            '_xm_rel_note': 0,
            # NNA-related metadata. Default = cut (matches MOD/S3M/XM
            # implicit behaviour). For IT, these get overwritten with the
            # owning instrument's values during the inst→sample annotation
            # pass after sample loading.
            'nna': 0,        # 0=cut 1=continue 2=noteoff 3=notefade
            'dct': 0,        # 0=off 1=note 2=sample 3=instrument
            'dca': 0,        # 0=cut 1=noteoff 2=notefade
            'fadeout': 0,    # 0..1024; per-tick voice attenuation = fadeout/512
        }

    def _empty_pattern(self, num_rows):
        empty_row = [{'sample':0,'period':0,'effect':0,'param':0}
                     for _ in range(self.num_channels)]
        return [list(empty_row) for _ in range(num_rows)]

    @staticmethod
    def _scan_pattern_max_channel(blob, pat_offsets):
        """Walk all patterns' raw byte streams and return (max channel
        index used + 1). Doesn't decode cells fully — just follows the
        IT channel-mask compression to advance the cursor correctly so
        we can spot the highest channel byte. Cheap upfront pass."""
        max_ch = 0
        last_mask = [0] * 64
        for po in pat_offsets:
            if po == 0 or po + 8 > len(blob): continue
            length = struct.unpack_from('<H', blob, po)[0]
            data = blob[po+8:po+8+length]
            n = len(data); cur = 0
            # Reset per-pattern caches (mask is per-channel, scoped to pattern)
            for i in range(64): last_mask[i] = 0
            while cur < n:
                c = data[cur]; cur += 1
                if c == 0: continue          # row terminator
                ch = (c - 1) & 0x3F
                if ch + 1 > max_ch: max_ch = ch + 1
                if c & 0x80:
                    if cur >= n: break
                    mask = data[cur]; cur += 1
                    last_mask[ch] = mask
                else:
                    mask = last_mask[ch]
                # Skip the field bytes per mask bits (1=note, 2=inst,
                # 4=vol, 8=fx pair). Don't read them — just advance.
                if mask & 0x01: cur += 1
                if mask & 0x02: cur += 1
                if mask & 0x04: cur += 1
                if mask & 0x08: cur += 2
        return max_ch

    def _decode_pattern(self, data, num_rows):
        """Decode IT compressed pattern stream into [row][ch] cells.

        See IT agent report §3 — channel-mask byte with bit-7 'reuse last
        mask' optimization.
        """
        rows = self._empty_pattern(num_rows)
        # Per-channel last-mask + last-data caches (IT's compression trick:
        # if mask byte has bit-7 set, the LSB-only reuse-last-data flags
        # come from the cached mask).
        last_mask = [0] * 64
        last_note = [0] * 64
        last_inst = [0] * 64
        last_vol  = [0] * 64
        last_eff  = [0] * 64
        last_par  = [0] * 64
        # IT effect memory: per-channel last non-zero param for slide/porta
        # effects that use "continue last" semantics on param=0. PT/MOD A
        # (vol slide) does NOT have memory (A00 = no slide), but IT's D
        # (mapped to MOD A) DOES — same for E/F (pitch slides) and G (tone
        # porta). Without this, IT vol-slide chains like D04 D00 D00 D00
        # only slide on the first row and the volume curve flattens out.
        last_D = [0] * 64  # D: vol slide
        last_E = [0] * 64  # E: porta down
        last_F = [0] * 64  # F: porta up
        last_G = [0] * 64  # G: tone porta
        cur = 0; row = 0
        n = len(data)
        while row < num_rows and cur < n:
            chvar = data[cur]; cur += 1
            if chvar == 0:
                row += 1
                continue
            ch = (chvar - 1) & 0x3F
            if chvar & 0x80:
                if cur >= n: break
                mask = data[cur]; cur += 1
                last_mask[ch] = mask
            else:
                mask = last_mask[ch]
            note = inst = vol = eff = par = 0
            inst_explicit = False     # True only if mask bit 0x02 set
            if mask & 0x01:
                if cur >= n: break
                note = data[cur]; cur += 1
                last_note[ch] = note
            if mask & 0x02:
                if cur >= n: break
                inst = data[cur]; cur += 1
                last_inst[ch] = inst
                inst_explicit = True
            vol_col_present = False
            if mask & 0x04:
                if cur >= n: break
                vol = data[cur]; cur += 1
                last_vol[ch] = vol
                vol_col_present = True
            if mask & 0x40:
                vol_col_present = True
            if mask & 0x08:
                if cur+1 >= n: break
                eff = data[cur]; par = data[cur+1]; cur += 2
                last_eff[ch] = eff; last_par[ch] = par
            if mask & 0x10:
                note = last_note[ch]
            if mask & 0x20:
                inst = last_inst[ch]
                # NOT setting inst_explicit — cell only inherits, doesn't
                # introduce a "new instrument" event. mikIT semantic: a
                # note without explicit inst is a pitch change, no
                # envelope/channel-volume reset.
            if mask & 0x40:
                vol = last_vol[ch]
            if mask & 0x80:
                eff = last_eff[ch]; par = last_par[ch]

            if ch >= self.num_channels:
                continue   # cell is for a channel beyond our cap; skip

            # Translate: IT note 0..119 = note (with 60 = middle C); 254 =
            # note-cut, 255 = note-off. Map note 0..119 → MOD period; treat
            # cut/off as Cxx vol=0 if no other effect.
            period = 0
            if 0 < note < 120:
                # IT note 60 = C-5 = native pitch (period 428).
                # `_note_to_mod_period` is XM-anchored at note 49 (XM C-4 =
                # native = period 428), so shift IT notes down by 11 (= 60-49)
                # to put native at the right anchor. Without this, every IT
                # melody plays one octave too high.
                period = _note_to_mod_period(note - 11)
            elif note == 254:
                # IT note-cut: instant silence regardless of envelope.
                # Was: only translated to Cxx 0 when eff==0 — silently
                # dropped the cut whenever the cell had any other effect
                # (e.g. pat 8 r10 ch0 has note-cut + F.b0 tempo, so the
                # cut got swallowed and the bass sustained another 14s).
                # Fix: mark via bit 6 of the sample byte (mirrors the
                # bit-7 note-off marker), which the engine handles
                # independently of the effect column. Period also goes
                # to 0 to suppress retrigger logic.
                pass   # handled below via samp_byte |= 0x40
            elif note == 255:
                # IT note-off: triggers envelope release (NOT instant cut).
                # We mark this via the same bit-7 sample-byte marker that
                # the engine already handles for XM key-off — flips keyOn
                # off so the envelope leaves sustain and fadeout starts.
                # Without this, every IT note-off cell silenced the voice
                # immediately, producing the "staccato w/ no reverb tail"
                # complaint. (For samples without envelopes/fadeout, this
                # still cuts cleanly — the engine handles that case too.)
                pass   # handled below via samp_byte |= 0x80

            # ── IT effect memory: substitute last param when param==0 ──
            # IT semantics for D/E/F/G with param==0 = "continue with last
            # value". MOD/PT semantics for the equivalent MOD effects A/2/1/3
            # treat param==0 differently (A00 is no-op, etc.). To bridge,
            # rewrite IT param=0 cells into IT cells carrying the last
            # non-zero param BEFORE translating to MOD format.
            if eff == 4 and par == 0 and last_D[ch] != 0:    # IT D
                par = last_D[ch]
            elif eff == 5 and par == 0 and last_E[ch] != 0:  # IT E
                par = last_E[ch]
            elif eff == 6 and par == 0 and last_F[ch] != 0:  # IT F
                par = last_F[ch]
            elif eff == 7 and par == 0 and last_G[ch] != 0:  # IT G
                par = last_G[ch]
            # Save non-zero params into memory for next time.
            if par != 0:
                if eff == 4:   last_D[ch] = par
                elif eff == 5: last_E[ch] = par
                elif eff == 6: last_F[ch] = par
                elif eff == 7: last_G[ch] = par
            m_eff, m_par = _xm_or_it_effect_to_mod(eff, par, is_xm=False)
            if eff != 0 and m_eff == 0 and m_par == 0:
                self._unsupported_effects += 1

            # Volume column: 0..64 = set volume.
            # If the cell ALSO has an effect column (e.g. Sxx note delay),
            # we cannot encode both via MOD's single effect+param slot. So
            # we synthesize Cxx from vol-col only when the effect column
            # is empty. For cells with BOTH vol-col and effect, we capture
            # vol-col into a SEPARATE field (`vol_col`) so the engine can
            # apply it on top of the effect.
            # CRITICAL bug fix: previously vol-col on cells with non-empty
            # effect was silently dropped — channels with vol-col=2 + Sxx
            # delay triggered at the sample's default volume (e.g. 36)
            # instead of the intended 2/64 = -25 dB. On jeff pat 0 row 0,
            # ch1/ch3/ch5/ch6/ch8 had this combo → 4 channels at full vol
            # vs OpenMPT's correct quiet level → 3× loudness mismatch.
            # IT Mxx (Set Channel Volume): eff=13 (= cmd 12 = letter M).
            # MOD has no equivalent, so the translator at line 880 returns
            # (0, 0) and the effect was silently dropped. This is wrong —
            # jeff has 39 Mxx events and without them channel volumes
            # never compensate when vol-col=1 cells set the note volume
            # very low (pat 1 row 0). Capture Mxx into a separate cell
            # field so the engine can apply it as a per-channel multiplier
            # alongside the existing note-volume / Cxx path.
            # Encoding: 0 = no Mxx on this cell, 1..65 = set channel
            # volume to (chn_vol - 1) ∈ [0, 64].
            cell_chn_vol = 0
            if eff == 13 and par <= 0x40:
                cell_chn_vol = par + 1   # offset by 1 so 0 = "absent"
                # Drop the MOD-effect translation since we handled it.
                m_eff = 0
                m_par = 0

            cell_vol_col = 0
            if 0 < vol <= 64 and m_eff == 0 and m_par == 0:
                # Vol-col on its own (no effect col): fold into Cxx.
                # Applies on BOTH trigger rows (period > 0) and continuation
                # rows (period == 0) — continuation Cxx events drive the
                # vol crescendos in jeff (rows 1-31 ramp ch0 vol 2→64).
                # The earlier `period > 0` restriction was wrong: it
                # silently dropped every continuation Cxx, leaving channels
                # stuck at their initial trigger volume for entire patterns.
                m_eff = 0xC
                m_par = vol
            elif 0 < vol <= 64 and (m_eff != 0 or m_par != 0):
                # Vol-col coexists with another effect. Capture separately
                # so the engine applies it on top.
                cell_vol_col = vol
            # vol-col=0 with retrigger: tried treating as a "preserve
            # current channel volume" sentinel (cell_vol_col=65) — fixes
            # the +17 dB jump at jeff t=3.27s but locks subsequent
            # vol-col=1 cells (pat 1 row 0) at very low channel volume
            # for the rest of playback because the song relies on Mxx
            # events later to bring volumes back up. Trade-off chosen:
            # accept the small jump, keep overall loudness right.

            # Resolve sample number AND transposed note. In IT instrument-mode
            # (cmwt >= 0x200), each cell's "instrument" byte is an instrument
            # index, and `inst.notesample[note]` returns BOTH the sample to
            # play AND the transposed note to play it at. mikIT formula:
            #   smp     = inst.notesample[note].sample
            #   smpnote = inst.notesample[note].note
            #   frq     = PitchTable[smpnote] * sample.C5Speed / 65536
            # So the played pitch is determined by the TRANSPOSED note, not
            # the user-pressed note. Without applying this, IT files that
            # use note transposition to map drums/leads onto specific sample
            # regions play every note at the wrong pitch (often whole octaves
            # off — the "wrong samples + wrong pitch" GADGET.IT symptom).
            samp_resolved = inst & 0x3F
            if self.inst_table and 1 <= inst <= 128 and 0 < note < 120:
                inst_data = self.inst_table.get(inst)
                if inst_data:
                    table_samp = inst_data['note_to_sample'][note]
                    if 1 <= table_samp <= 63:
                        samp_resolved = table_samp & 0x3F
                    # Apply note transposition: recompute period from the
                    # table-resolved smpnote rather than the played note.
                    table_note = inst_data['note_to_note'][note]
                    if 0 < table_note < 120 and table_note != note:
                        period = _note_to_mod_period(table_note - 11)

            # IT note-off (255) and note-cut (254): mark via high bits of
            # the sample byte so the engine handles them independently of
            # the effect column. Period also goes to 0 to suppress
            # retrigger logic.
            #   bit 7 (0x80) = note-off → envelope release + fadeout
            #   bit 6 (0x40) = note-cut → instant silence (state.active=false)
            samp_byte = samp_resolved
            if note == 255:
                samp_byte |= 0x80
                period = 0
            elif note == 254:
                samp_byte |= 0x40
                period = 0
            # NOTE: tried zeroing samp_byte for cells with inherited
            # instrument (mask 0x20, no 0x02) to take the JS engine's
            # period-only retrigger path. Eliminated the +17 dB jump
            # at jeff t=3.27s but produced a 25 dB overall loudness
            # drop because jeff's patterns are 128 rows (Cxx crescendos
            # in rows 32-127) but our engine truncates to 64 rows.
            # Until variable-row patterns are supported, can't fix
            # both at once. Reverted; accepting the small jump.
            rows[row][ch] = {
                'sample': samp_byte,
                'period': period,
                'effect': m_eff,
                'param':  m_par,
                # vol_col != 0 means the cell carries a vol-col=N value the
                # engine should apply on TRIGGER (sets state.currentVolume
                # to N) in addition to whatever the effect column does.
                # Only emitted for IT cells where vol-col coexists with a
                # non-empty effect column; otherwise vol-col is folded into
                # the effect slot as Cxx and this stays 0.
                'vol_col': cell_vol_col,
                # chn_vol carries IT Mxx (Set Channel Volume), encoded as
                # 0 = no Mxx, 1..65 = set channel vol to (chn_vol - 1).
                # Multiplied in at the channel mix stage on top of the
                # note volume (mixVol).
                'chn_vol': cell_chn_vol,
            }
        return rows

    def _read_it_uncompressed_sample(self, raw, length, is_16bit, is_signed):
        """Read uncompressed IT sample data, return list of int16-shifted
        samples (signed). 8-bit samples are left-shifted by 8 to match the
        decompressor's output format."""
        out = []
        if is_16bit:
            n = min(length, len(raw)//2)
            for i in range(n):
                v = struct.unpack_from('<h', raw, i*2)[0]
                if not is_signed:
                    v = (v + 0x8000) & 0xFFFF
                    if v >= 0x8000: v -= 0x10000
                out.append(v)
        else:
            n = min(length, len(raw))
            for i in range(n):
                v = raw[i]
                if is_signed:
                    if v >= 0x80: v -= 0x100
                else:
                    v -= 128
                out.append(v << 8)
        return out

    def _read_it_compressed_sample(self, blob, data_off, length, is_16bit, it215):
        """Read IT-compressed sample data, decompressing in blocks. IT packs
        data in chunks of 0x8000 samples (8-bit) or 0x4000 samples (16-bit);
        each chunk starts with a 16-bit packed-byte count, then `count`
        bytes of compressed bits.
        """
        chunk_samples = 0x4000 if is_16bit else 0x8000
        out = []
        cur = data_off
        remaining = length
        while remaining > 0 and cur + 2 <= len(blob):
            packed_count = struct.unpack_from('<H', blob, cur)[0]
            cur += 2
            block = blob[cur:cur+packed_count]
            cur += packed_count
            n_this = min(remaining, chunk_samples)
            decoded = _it_decompress_sample(block, n_this, is_16bit, it215)
            out.extend(decoded[:n_this])
            remaining -= n_this
        return out


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
                'loop_start':  _compress_loop_offsets(sample['repeat_point'], sample['repeat_length'], bf)[0],
                'loop_length': _compress_loop_offsets(sample['repeat_point'], sample['repeat_length'], bf)[1],
                'bw_factor':   bf,
                'volume':      sample['volume'],
                'finetune':    sample['finetune'],
                'name':        sample['name'],
                # NNA per-sample fields (default 0/0/0/0 for non-IT). The
                # JS engine reads these to dispatch on note trigger:
                # nna=1 (continue) and nna=3 (notefade) move the OLD voice
                # to a ghost slot instead of cutting it. nna=2 (noteoff)
                # falls back to notefade since we have no envelope release.
                'nna':         sample.get('nna', 0),
                'dct':         sample.get('dct', 0),
                'dca':         sample.get('dca', 0),
                'fadeout':     sample.get('fadeout', 0),
                # XM volume envelope (empty for non-XM and env-off XM).
                # env_pts is a list of [x, y] pairs; engine advances env_x
                # by 1 per tick, holds at sus_pt while keyOn, loops between
                # loop_st_pt and loop_en_pt if env_loop is set.
                'env_pts':     sample.get('env_pts', []),
                'env_sus':     sample.get('env_sus', False),
                'env_loop':    sample.get('env_loop', False),
                'env_sus_pt':  sample.get('env_sus_pt', 0),
                # IT-only: sustain LOOP end-point index. For XM where
                # sustain is a single hold, this stays 0 and the engine
                # treats env_sus_pt as the hold point. For IT, env_sus_en
                # marks the OTHER end of the sustain loop range.
                'env_sus_en':  sample.get('env_sus_en', 0),
                'env_loop_st': sample.get('env_loop_st', 0),
                'env_loop_en': sample.get('env_loop_en', 0),
                # IT c5_speed (Hz at MOD period 428 = IT C-5). When > 0 the
                # engine uses this directly in periodToFreq instead of the
                # MOD finetune-nibble lookup. IT samples often have c5_speed
                # values (drums @ 33000 Hz, sinewaves @ 168000 Hz) far above
                # the finetune-table range (~7895-8757 Hz), and squashing
                # them through that table drops their pitch by 2-4 octaves.
                'c5_speed':    sample.get('c5_speed', 0),
            })

            all_samples.extend(data_float.tolist())
            individual_samples.append(compressed.tolist())
            current_pos += len(data_float)
        else:
            sample_map.append({
                'index': i, 'start': 0, 'length': 0,
                'loop_start': 0, 'loop_length': 0, 'bw_factor': 1,
                'volume': 0, 'finetune': 0, 'name': '',
                'nna': 0, 'dct': 0, 'dca': 0, 'fadeout': 0,
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

                cell = {
                    'sample': sample,
                    'period': period,
                    'effect': effect,
                    'param': param
                }
                vc = ch.get('vol_col', 0)
                if vc:
                    cell['vol_col'] = vc
                cv = ch.get('chn_vol', 0)
                if cv:
                    cell['chn_vol'] = cv
                pattern_data.append(cell)
    
    # Optional compression
    if compress:
        print("   Compressing patterns...")
        # Flatten to bytes for RLE. Cell layout (7 bytes):
        #   0: sample
        #   1: period >> 8
        #   2: period & 0xFF
        #   3: (effect << 4) | (param >> 4)
        #   4: param & 0x0F
        #   5: vol_col (0..64 set, 65 preserve sentinel; 0 = absent)
        #   6: chn_vol (0 = no Mxx, 1..65 = set channel vol to chn_vol-1)
        pattern_bytes = []
        for note in pattern_data:
            pattern_bytes.extend([
                note['sample'],
                note['period'] >> 8,
                note['period'] & 0xFF,
                (note['effect'] << 4) | (note['param'] >> 4),
                note['param'] & 0x0F,
                note.get('vol_col', 0) & 0x7F,
                note.get('chn_vol', 0) & 0x7F
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
    
    const _nc = modData.numChannels || 4;
    console.log('Decompressed bytes:', decompressed.length, 'expected:', modData.numPatterns * 64 * _nc * 7);

    // Reconstruct pattern objects (7-byte cells:
    //   sample / period_hi / period_lo / eff_par_hi / par_lo /
    //   vol_col / chn_vol).
    const patterns = [];
    let offset = 0;
    const totalNotes = modData.numPatterns * 64 * _nc;

    for (let n = 0; n < totalNotes && offset + 6 < decompressed.length; n++) {
        const s     = decompressed[offset++] || 0;
        const ph    = decompressed[offset++] || 0;
        const pl    = decompressed[offset++] || 0;
        const epHi  = decompressed[offset++] || 0;
        const pLo   = decompressed[offset++] || 0;
        const vcol  = decompressed[offset++] || 0;
        const cvol  = decompressed[offset++] || 0;
        patterns.push({
            sample: s,
            period: (ph << 8) | pl,
            effect: epHi >> 4,
            param: ((epHi & 0x0F) << 4) | pLo,
            vol_col: vcol,
            chn_vol: cvol
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
    numChannels: {num_channels},
    initialBPM: {mod.initial_tempo},
    initialSpeed: {mod.initial_speed},
    // IT global/mix volume (0..128). Defaults to 128 for non-IT formats
    // (no attenuation). Engine multiplies output by (gv*mv)/(128*128).
    globalVol: {getattr(mod, 'global_volume', 128)},
    mixVol:    {getattr(mod, 'mix_volume', 128)},
    // IT per-channel default pan (0..64). Empty for non-IT formats; in
    // that case the engine falls back to MOD-style ch%4 LRRL panning.
    channelPan: {json.dumps(list(getattr(mod, 'channel_pan', [])))},
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

// ── AdaptiveLimiter (port of adaptive_limiter) ──────────────────────────────
// Time-domain stateful limiter with adaptive attack/release. Stereo-linked
// (peak = max(|L|, |R|)) so L/R can't drift to different gains and break the
// stereo image. No lookahead; the 1ms minimum attack catches transients
// fast enough at typical sample rates.
//
// Adaptive coefficients:
//   exceed_ratio = clamp((|x| − maxLimit) / maxLimit, 0, 1)
//   attack_ms    = base_attack·(1 − r)  + min_attack·r        // faster when over-shoot is bigger
//   release_ms   = base_release·gain    + min_release         // slower as gain returns to unity
//
// The 70/30 smoothing kicks in when |Δgain| > 0.1 to avoid audible zipper
// noise on sudden bursts.
class AdaptiveLimiter {{
    constructor(sampleRate) {{
        this.sr             = sampleRate;
        // Loud mode: push the limiter ceiling far above unity so it only
        // engages on catastrophic peaks (FAT4X bursts). The downstream
        // post-limiter makeup gain + hard-clip stage handles the rest,
        // pinning the signal against ±1 for maximum perceived loudness.
        this.maxLimit       = 1.50;
        this.baseAttackMs   = 8.0;
        this.minAttackMs    = 1.0;
        this.baseReleaseMs  = 80.0;
        this.minReleaseMs   = 30.0;
        this.smoothThresh   = 0.1;
        this.gain           = 1.0;          // current_limit_gain
    }}
    process(l, r) {{
        const absMag = Math.max(Math.abs(l), Math.abs(r));   // stereo-linked
        const exceedAmount = Math.max(0.0, absMag - this.maxLimit);
        const exceedRatio  = Math.min(1.0, exceedAmount / this.maxLimit);
        const attackMs  = this.baseAttackMs  * (1.0 - exceedRatio) + this.minAttackMs * exceedRatio;
        const releaseMs = this.baseReleaseMs * (1.0 - Math.max(0.0, 1.0 - this.gain)) + this.minReleaseMs;
        const aCoeff = Math.exp(-1000.0 / (attackMs  * this.sr));
        const rCoeff = Math.exp(-1000.0 / (releaseMs * this.sr));
        const targetGain = Math.min(1.0, this.maxLimit / (absMag + 1e-6));
        // Branch: are we attacking (target < current → reduce more) or releasing?
        const nextGainRaw = (targetGain < this.gain)
            ? (aCoeff * this.gain + (1.0 - aCoeff) * targetGain)
            : (rCoeff * this.gain + (1.0 - rCoeff) * targetGain);
        const nextGain = Math.max(0.01, Math.min(1.0, nextGainRaw));
        // 70/30 smoothing on big jumps to suppress zipper noise
        const change = Math.abs(nextGain - this.gain);
        const finalGain = (change > this.smoothThresh)
            ? (0.7 * this.gain + 0.3 * nextGain)
            : nextGain;
        this.gain = finalGain;
        return [l * finalGain, r * finalGain];
    }}
}}

class MODPlayer {{
    constructor() {{
        this.audioCtx = null;
        this.isPlaying = false;
        this.bpm   = Math.max(32, modData.initialBPM   || 125);  // mikIT: bpm min=32
        this.speed = Math.min(32, modData.initialSpeed || 6);    // mikIT: speed max=32
        this.sampleRate = 44100;
        this._limiter = new AdaptiveLimiter(this.sampleRate);
        
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
                // IT Mxx (Set Channel Volume): 0..64, default 64 (= no scaling).
                // Updated by cells carrying chn_vol. Multiplied with mixVol in
                // the per-sample mix for proper IT note×channel volume math.
                channelVolume: 64,
                // mikMod-style click ramp: vlast = last output sample
                // value (pre-pan). On trigger, clickRemaining is set to
                // the window length (clickWindow); per-sample ramp is
                // (clickWindow - clickRemaining) / clickWindow.
                // For NNA=0/1 (percussive): 256-sample anti-click (~5.8ms
                // @44.1kHz) — matches mikMod default, kills bass/drum
                // retrigger clicks without softening transient attack.
                // For NNA=2/3 (sustained pad displaced): 4410 samples
                // (~100 ms equal-power crossfade) so the OLD voice
                // fades smoothly instead of cutting abruptly.
                vlast: 0.0, clickRemaining: 0, clickWindow: 256,
                loopbackPoint: 0, loopCount: 0, _delayedNote: null,
                // XM volume envelope state. envX is the current envelope-x
                // position in player ticks. keyOn=true while sustaining,
                // false after key-off. volFade is the 16-bit fixedpoint
                // fadeout multiplier (65536 = full, 0 = silent); only
                // decrements when keyOn=false. envFin=true means env has
                // reached its final point (no further advancement).
                envX: 0, keyOn: true, volFade: 65536, envFin: false,
                // ── NNA ghost voice array ───────────────────────────────
                // Up to 4 "ghost" voices per channel that keep playing
                // after a new note triggers, when the OLD sample's NNA
                // is `continue` (1) or `notefade` (3). Each ghost holds
                // a frozen snapshot of the previous voice (sample,
                // period, volume, samplePos) plus a fadeAmt that
                // attenuates the volume by `fadeAmt/512` per tick when
                // non-zero (notefade and noteoff modes).
                ghostVoices: [],
            }});

            // Dying channel for crossfade (matches C++ dying[] array)
            this.dyingChannels.push({{
                sample: 0, period: 0, samplePos: 0.0, volume: 64, active: false,
                volumeFade: 0, volumeFadeInc: 0, samplesLeft: 0
            }});
        }}
        // Max ghost voices kept alive per channel before the oldest gets
        // evicted. 4 is enough for typical IT NNA songs (drum stacking,
        // pad bloom). Higher = more polyphony but more shader/CPU cost.
        this._maxGhosts = 4;
        
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
        // Re-build the limiter with the actual hardware sample rate so its
        // attack/release coefficients are correct (likely 48000, not 44100).
        this._limiter = new AdaptiveLimiter(this.sampleRate);
        this.nextPlayTime = this.audioCtx.currentTime; // Track next buffer start time
        this.updateTiming();
        this.log('Audio initialized: ' + this.sampleRate + ' Hz');
    }}
    
    // Finetune table from mikIT (C4 playback speed for each of 16 finetune values)
    // finetune nibble 0-7 = positive (higher pitch), 8-15 = negative (lower pitch)
    // For IT samples we prefer raw c5_speed (passed via the 3rd arg) over
    // the 16-entry table, since IT files routinely use c5_speed values
    // (drums @ 33000 Hz, sinewaves @ 168000 Hz, leads @ 19800 Hz) that
    // can't be represented by MOD's finetune nibble at all.
    periodToFreq(period, finetune, c5_speed) {{
        if (period === 0) return 0;
        if (c5_speed && c5_speed > 0) {{
            return (c5_speed * 428) / period;
        }}
        const c4speeds = [8363,8413,8463,8529,8581,8651,8723,8757,
                          7895,7941,7985,8046,8107,8169,8232,8280];
        const c4 = c4speeds[(finetune || 0) & 0xF];
        return (c4 * 428) / period;
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
        // Wrap any cubic-tap index back into the legal range. For looping
        // samples, taps that fall before loop_start or at/after loop_end
        // must wrap THROUGH the loop boundary so the interpolation stays
        // continuous — otherwise p2/p3 read past the loop end into
        // unrelated bytes, producing a hard discontinuity once per loop
        // cycle (= audible high-harmonic buzz). For non-looping samples,
        // clamp to [0, length-1] so we don't read past the sample.
        const loopEnd = info.loop_start + info.loop_length;
        const wrap = (i) => {{
            if (looping) {{
                if (i < info.loop_start) {{
                    const off = ((i - info.loop_start) % info.loop_length + info.loop_length) % info.loop_length;
                    return info.loop_start + off;
                }}
                if (i >= loopEnd) {{
                    return info.loop_start + (i - info.loop_start) % info.loop_length;
                }}
                return i;
            }}
            return Math.max(0, Math.min(i, info.length - 1));
        }};
        const p0 = modData.samples[base + wrap(pos0 - 1)];
        const p1 = modData.samples[base + wrap(pos0)];
        const p2 = modData.samples[base + wrap(pos0 + 1)];
        const p3 = modData.samples[base + wrap(pos0 + 2)];
        
        const a = -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3;
        const b =      p0 - 2.5*p1 + 2.0*p2 - 0.5*p3;
        const c = -0.5*p0           + 0.5*p2;
        return ((a * frac + b) * frac + c) * frac + p1;
    }}
    
    getNote(pattern, row, channel) {{
        const nc = this.numChannels;
        const idx = (pattern * 64 * nc) + (row * nc) + channel;
        return modData.patterns[idx] || {{ sample: 0, period: 0, effect: 0, param: 0, vol_col: 0 }};
    }}

    // ── NNA dispatch: convert old voice to a ghost ────────────────────────
    // Called BEFORE overwriting channel state on a new note trigger. Reads
    // the OLD sample's NNA byte (set in modData.sampleMap[i].nna by the IT
    // parser; defaults to 0=cut for MOD/S3M/XM):
    //   0 cut       → no ghost; old voice silenced (current default)
    //   1 continue  → ghost with no fade (drum stacking, percussion)
    //   2 noteoff   → ghost with fade (no envelope release in Phase 2 —
    //                  approximated by treating like notefade)
    //   3 notefade  → ghost with fade rate fadeout/512 per tick
    _dispatchNNA(ch) {{
        const state = this.channels[ch];
        if (!state.active || state.sample < 0 || state.volume <= 0) return;
        const oldInfo = modData.sampleMap[state.sample];
        if (!oldInfo) return;
        const nna = oldInfo.nna || 0;
        if (nna === 0) return;     // cut — fall through to retrigger as before
        // All non-cut NNAs dispatch a ghost so the OLD voice plays through
        // its envelope release (NNA=2 linear release, NNA=3 fadeout, NNA=1
        // continue). The ghost+active overlap creates a slight amplitude
        // bump during the chord-change window but the envelope-shaped
        // release sound is clearly preferred over a clean cut.
        // Master gain reduced (line 1991, vol=0.5) and MAKEUP reduced
        // (line 4047) to keep the overlap below the limiter ceiling.
        // Snapshot full envelope state so the ghost continues advancing
        // through release/fadeout exactly like a live voice.
        // NNA=1 (continue): keyOn stays true, env keeps sustaining.
        // NNA=2 (off):      keyOn=false → env releases past sus point.
        // NNA=3 (fade):     keyOn=false + force fadeout even with no env.
        const ghost = {{
            sample:     state.sample,
            period:     state.period,
            basePeriod: state.basePeriod,
            samplePos:  state.samplePos,
            volume:     state.volume,
            // mikIT NNA mechanism (mmod_it0.cpp:1032): OLD voice stays in
            // place with keyon=0, no copy. Its envelope advances past
            // sustain via the existing per-tick advance, naturally
            // releasing over the envelope's release segment. No forced
            // down-ramp on ghost mixVol — the envelope multiplier alone
            // shapes the fade. NEW voice trigger ramps mixVol 0 → target
            // (kick semantic) which is the mikIT "fadecount" click ramp.
            // Overall amplitude is bounded by the lowered master volume
            // (mikIT uses pow(channels,0.52)/channels per-voice gain ≈
            // 0.137 for the 64-voice pool, giving headroom for overlap).
            // Ghost preserves the OLD voice's mixVol (no forced ramp-down).
            // The fade-out is shaped by the inherited envelope release
            // (envX continues advancing past sustain) and NFC fadeout — so
            // long-tail strings actually ring through. The NEW voice's
            // 30 ms mixVol ramp-up alone provides the crossfade against
            // this naturally-decaying ghost.
            mixVol:     state.mixVol,
            volInc:     0,
            envX:       state.envX,
            keyOn:      (nna === 1),
            volFade:    state.volFade,
            envFin:     state.envFin,
            // NNA=3 forces fadeout even when there's no envelope. NNA=2
            // relies on the envelope's release tail; only fades via
            // volFade if the envelope itself runs out.
            forceFadeout: (nna === 3),
            active:     true,
        }};
        if (state.ghostVoices.length >= this._maxGhosts) {{
            state.ghostVoices.shift();   // FIFO eviction of the oldest
        }}
        state.ghostVoices.push(ghost);
        // The ghost now carries the OLD voice's output continuity. Reset
        // the active state's vlast to 0 so the click ramp on the NEW voice
        // doesn't ALSO blend from the OLD signal — that would double-count
        // (ghost + active vlast both contributing OLD), producing the +6dB
        // swell every retrigger.
        state.vlast = 0;
    }}

    // ── DCT (Duplicate Check Type) on incoming note ───────────────────────
    // Scan ghosts across ALL channels for matches against the NEW note's
    // sample/note, then apply DCA (Duplicate Check Action). Implemented
    // per per-sample DCT/DCA byte (the IT parser annotates these).
    //   DCT: 0=off 1=note 2=sample 3=instrument (we map inst→sample 1:1)
    //   DCA: 0=cut 1=noteoff 2=notefade
    _applyDCT(newSample, newPeriod) {{
        const newInfo = modData.sampleMap[newSample];
        if (!newInfo) return;
        const dct = newInfo.dct || 0;
        if (dct === 0) return;
        const dca = newInfo.dca || 0;
        for (let ch = 0; ch < this.numChannels; ch++) {{
            const st = this.channels[ch];
            for (let i = st.ghostVoices.length - 1; i >= 0; i--) {{
                const g = st.ghostVoices[i];
                if (!g.active) continue;
                let match = false;
                if (dct === 1 && g.period === newPeriod)  match = true;   // by note
                if (dct === 2 && g.sample === newSample)  match = true;   // by sample
                if (dct === 3 && g.sample === newSample)  match = true;   // by instrument (proxy)
                if (!match) continue;
                if (dca === 0) {{
                    st.ghostVoices.splice(i, 1);                          // cut
                }} else {{
                    // DCA=1 (noteoff) / 2 (notefade): trigger key-off so
                    // the envelope-release / fadeout machinery in
                    // _decayGhosts takes over.
                    g.keyOn = false;
                    if (dca === 2) g.forceFadeout = true;
                }}
            }}
        }}
    }}

    // ── XM volume envelope: linear-interpolate y at env_x ────────────────
    // Returns the y value (0..64) interpolated between adjacent envelope
    // points. Clamps to first/last point's y outside the range.
    _xmEnvLookup(info, envX) {{
        const pts = info.env_pts;
        if (pts.length === 0) return 64;
        if (envX <= pts[0][0]) return pts[0][1];
        if (envX >= pts[pts.length - 1][0]) return pts[pts.length - 1][1];
        for (let i = 0; i < pts.length - 1; i++) {{
            const [x0, y0] = pts[i], [x1, y1] = pts[i + 1];
            if (envX >= x0 && envX <= x1) {{
                const t = x1 === x0 ? 0 : (envX - x0) / (x1 - x0);
                return y0 + (y1 - y0) * t;
            }}
        }}
        return pts[pts.length - 1][1];
    }}

    // ── Per-tick envelope advance + fadeout ───────────────────────────────
    // Advances each active voice's envX by 1 tick, honouring sustain (env
    // pauses at sus_pt while keyOn) and loop (jump from loop_en back to
    // loop_st when reached). After key-off, decrements volFade by
    // 2*fadeout per tick (mikIT/openMPT XM convention).
    _advanceEnvelopes() {{
        for (let ch = 0; ch < this.numChannels; ch++) {{
            const state = this.channels[ch];
            if (!state.active) continue;
            const info = modData.sampleMap[state.sample];
            if (!info) continue;

            const pts = info.env_pts;
            if (pts && pts.length >= 2 && !state.envFin) {{
                const susPt  = info.env_sus_pt | 0;
                const susEn  = info.env_sus_en | 0;       // IT: sustain LOOP end
                const lpStPt = info.env_loop_st | 0;
                const lpEnPt = info.env_loop_en | 0;
                const susStX = pts[Math.min(susPt,  pts.length - 1)][0];
                const susEnX = pts[Math.min(susEn,  pts.length - 1)][0];
                const lpStX  = pts[Math.min(lpStPt, pts.length - 1)][0];
                const lpEnX  = pts[Math.min(lpEnPt, pts.length - 1)][0];

                // mikIT IT semantics (mmod_it1.cpp:228 ITPROCESS::Process):
                //   tick++;
                //   if (keyon && env_sus): if (tick > SLE) tick = SLB
                //   else if (env_loop):    if (tick > LpE) tick = LpB
                //   if (tick > Lst):       tick = Lst, env DONE
                // For XM, sustain is a single hold point — represented as
                // SLB == SLE so the wrap is a no-op (tick stays clamped at
                // SLE = the hold point). This unified form covers both.
                let nextX = state.envX + 1;
                if (info.env_sus && state.keyOn && nextX > susEnX) {{
                    nextX = susStX;
                }} else if (info.env_loop && nextX > lpEnX && lpEnX > lpStX) {{
                    nextX = lpStX;
                }} else if (nextX > pts[pts.length - 1][0]) {{
                    nextX = pts[pts.length - 1][0];
                    state.envFin = true;
                }}
                state.envX = nextX;

                // Envelope-end deactivation. When the envelope has finished
                // at its final point AND that point's y is 0 (a "release-
                // to-silence" tail), mark the voice inactive. Without this,
                // the channel keeps mixing sample × envMul=0 forever — a
                // numerical no-op, but it also keeps the underlying sample
                // loop running and prevents the slot from being reclaimed.
                // Matches openMPT/mikIT behavior: an env that ends at 0
                // fully kills the voice. Fixes GADGET.IT inst 27
                // "Sinewave-envelope" leaking a sustained sinewave loop
                // past the envelope's natural end.
                if (state.envFin && pts[pts.length - 1][1] === 0) {{
                    state.active = false;
                }}
            }}

            // Fadeout: only after key-off, decrements volFade by 2*fadeout
            // per tick. When volFade hits 0 the voice goes silent and is
            // marked inactive.
            if (!state.keyOn) {{
                const fo = (info.fadeout | 0);
                if (fo > 0) {{
                    state.volFade -= 2 * fo;
                    if (state.volFade <= 0) {{
                        state.volFade = 0;
                        state.active = false;
                    }}
                }} else if (!info.env_pts || info.env_pts.length < 2) {{
                    // No env, no fadeout: key-off cuts cleanly.
                    state.active = false;
                }}
            }}
        }}
    }}

    // ── Per-tick ghost envelope advance + decay ──────────────────────────
    // Advances each ghost's envX (honouring sustain only when keyOn=true,
    // i.e. NNA=1 continue) and decrements volFade past key-off — same
    // formula as _advanceEnvelopes for active voices, so a ghost dies
    // exactly when an equivalent active voice would. Called once per tick.
    _decayGhosts() {{
        for (let ch = 0; ch < this.numChannels; ch++) {{
            const st = this.channels[ch];
            for (let i = st.ghostVoices.length - 1; i >= 0; i--) {{
                const g = st.ghostVoices[i];
                if (!g.active) {{ st.ghostVoices.splice(i, 1); continue; }}
                const info = modData.sampleMap[g.sample];
                if (!info) {{ st.ghostVoices.splice(i, 1); continue; }}

                // Envelope advance — same shape as _advanceEnvelopes.
                const pts = info.env_pts;
                if (pts && pts.length >= 2 && !g.envFin) {{
                    const susPt  = info.env_sus_pt | 0;
                    const susEn  = info.env_sus_en | 0;
                    const lpStPt = info.env_loop_st | 0;
                    const lpEnPt = info.env_loop_en | 0;
                    const susStX = pts[Math.min(susPt,  pts.length - 1)][0];
                    const susEnX = pts[Math.min(susEn,  pts.length - 1)][0];
                    const lpStX  = pts[Math.min(lpStPt, pts.length - 1)][0];
                    const lpEnX  = pts[Math.min(lpEnPt, pts.length - 1)][0];
                    let nextX = g.envX + 1;
                    if (info.env_sus && g.keyOn && nextX > susEnX) {{
                        nextX = susStX;
                    }} else if (info.env_loop && nextX > lpEnX && lpEnX > lpStX) {{
                        nextX = lpStX;
                    }} else if (nextX > pts[pts.length - 1][0]) {{
                        nextX = pts[pts.length - 1][0];
                        g.envFin = true;
                    }}
                    g.envX = nextX;
                    // Envelope ended at zero → kill ghost.
                    if (g.envFin && pts[pts.length - 1][1] === 0) {{
                        st.ghostVoices.splice(i, 1);
                        continue;
                    }}
                }}

                // Fadeout: applies after key-off (NNA=2/3). NNA=3 forces
                // fadeout even without an envelope; NNA=2 relies on the
                // envelope above and only fades if envFin landed >0.
                if (!g.keyOn) {{
                    const fo = info.fadeout | 0;
                    if (fo > 0 && (g.forceFadeout || g.envFin || !pts || pts.length < 2)) {{
                        g.volFade -= 2 * fo;
                        if (g.volFade <= 0) {{
                            st.ghostVoices.splice(i, 1);
                            continue;
                        }}
                    }} else if ((!pts || pts.length < 2) && fo === 0) {{
                        // No env, no fadeout: NNA=2/3 cuts cleanly.
                        st.ghostVoices.splice(i, 1);
                        continue;
                    }}
                }}
            }}
        }}
    }}
    
    processTick() {{
        const patternIdx = modData.songPositions[this.currentPattern];

        // ── Envelope advance happens BEFORE event processing ──────────────
        // mikIT/openMPT XM convention: at tick T, the envelope advances
        // using the keyOn state from tick T-1. Then tick-T events (note
        // triggers, key-offs) fire. So at the moment of key-off, envX is
        // still at the sustain point — the release doesn't begin until
        // tick T+1. Without this ordering, the first release tick fires
        // one tick early and the perceived envelope drop is too quick.
        // (Note triggers explicitly reset envX/keyOn/volFade afterwards,
        // so the advance done here is harmlessly overwritten on retrig.)
        this._advanceEnvelopes();

        // On tick 0, trigger new notes
        if (this.currentTick === 0) {{
            for (let ch = 0; ch < this.numChannels; ch++) {{
                const note = this.getNote(patternIdx, this.currentRow, ch);

                // IT Mxx (Set Channel Volume) fires on tick 0 of any row,
                // independent of trigger. chn_vol = 1..65 means set channel
                // volume to (chn_vol - 1) ∈ [0, 64]; 0 means absent.
                if (note.chn_vol && note.chn_vol >= 1 && note.chn_vol <= 65) {{
                    this.channels[ch].channelVolume = note.chn_vol - 1;
                }}

                // ── IT note-cut: bit 6 of the sample byte signals
                // "instant silence" (vs note-off bit 7 which only releases
                // the envelope). Required for cells like GADGET.IT pat 8
                // r10 ch0 where note=254 lives alongside an F-effect —
                // the cut MUST fire even when there's a non-empty effect
                // on the same row, AND the effect itself MUST still run
                // (e.g. pat 7 r0 ch0 has note-cut + F.59 setting BPM 89,
                // and skipping the effect leaves the song at 125 BPM —
                // pat 7 ends up 2× too long, bass line at 0:04 misses).
                // Cut the voice, then fall through so effect handlers
                // (and effect-memory updates) at the bottom of this branch
                // still execute.
                let _isNoteCut = false;
                if (note.sample & 0x40) {{
                    const _state = this.channels[ch];
                    _state.active = false;
                    _state.mixVol = 0;
                    _state.volume = 0;
                    _state.currentVolume = 0;
                    _state.volFade = 0;
                    _state.envFin = true;
                    _isNoteCut = true;
                }}

                // ── XM key-off: bit 7 of the sample byte signals "release"
                // Dispatch the active voice to a ghost (with fadeout-based
                // decay if the sample has fadeout>0, hard-cut otherwise),
                // then skip note-trigger logic for this cell. The remaining
                // bits 0..4 are still a valid sample index — used to look
                // up fadeout, but never retriggered.
                if (note.sample & 0x80) {{
                    // XM key-off: leave the voice playing but flip keyOn
                    // off so the envelope leaves sustain, and the per-tick
                    // fadeout begins multiplying volFade down. Voice goes
                    // silent naturally when volFade hits 0 OR when env
                    // reaches its end and envelope's final-y value is 0.
                    // No ghost needed — channel state continues until the
                    // next note retriggers it.
                    this.channels[ch].keyOn = false;
                    continue;
                }}
                const noteSample = note.sample & 0x3F;

                // Handle new sample trigger
                if (noteSample > 0) {{
                    const state = this.channels[ch];
                    // Capture pre-trigger envelope state for potential inheritance.
                    // When the channel was sustaining an envelope-equipped voice
                    // and we're cut-retriggering (NNA_GHOST_MODE=false), inherit
                    // envX so chord changes on samples with attack envelopes
                    // (Phase Strings env_pts=[[0,0],[8,64]…]) don't dip to silent
                    // for the first ~20 ms while envX rebuilds from 0.
                    const _wasActive = state.active;
                    const _oldInfo = modData.sampleMap[state.sample];
                    const _oldHadEnv = !!(_oldInfo && _oldInfo.env_pts && _oldInfo.env_pts.length >= 2);
                    const _oldEnvX = state.envX;
                    const _oldEnvFin = state.envFin;
                    this.dyingChannels[ch].active = false;
                    // XM cuts any lingering key-off-fade voice on retrigger.
                    // (IT key-off handling goes through the NNA path which
                    // produces tagged ghosts of its own — leave those alone.)
                    for (let gi = state.ghostVoices.length - 1; gi >= 0; gi--) {{
                        if (state.ghostVoices[gi].xmKeyoff) state.ghostVoices.splice(gi, 1);
                    }}

                    // EDx note delay: stash and trigger x ticks later
                    if (note.effect === 0xE && ((note.param >> 4) & 0xF) === 0xD) {{
                        state._delayedNote = note;
                        // Don't trigger sample now
                    }} else {{
                        const sampleInfo = modData.sampleMap[noteSample - 1];

                        // ── NNA dispatch on the OLD voice ─────────────────
                        // Before overwriting state, check the OLD sample's
                        // NNA. cut (0) → no ghost. continue (1) → ghost
                        // with no fade. notefade (3) / noteoff (2) → ghost
                        // that fades out by fadeout/512 per tick. Tone
                        // porta keeps the same voice, so skip dispatch.
                        if (note.effect !== 0x3) {{
                            this._dispatchNNA(ch);
                            // DCT: scan ALL channels' ghosts for matches on
                            // the NEW note's sample/instrument and apply DCA.
                            this._applyDCT(noteSample - 1, note.period);
                        }}

                        // IT volume-on-retrigger semantics:
                        //   - vol_col = 1..64: set vol to that value
                        //   - vol_col = 65 (sentinel): cell had vol-col=0,
                        //     i.e. "preserve current channel volume" — used
                        //     by IT files to retrigger without disturbing
                        //     the existing Cxx-set channel volume
                        //   - Cxx (folded vol-col, no other fx): set vol
                        //     to param
                        //   - None of the above: reset to sample.dv
                        state.sample = noteSample - 1;
                        let _trigVol = sampleInfo.volume;
                        if (note.vol_col === 65) {{
                            _trigVol = state.currentVolume;     // preserve
                        }} else if (note.vol_col && note.vol_col > 0 && note.vol_col <= 64) {{
                            _trigVol = note.vol_col;
                        }} else if (note.effect === 0xC) {{
                            _trigVol = Math.min(64, note.param);
                        }}
                        state.volume = _trigVol;
                        state.currentVolume = _trigVol;
                        state.active = true;
                        state.volumeRamping = false;

                        if (note.effect === 0x3) {{
                            // Tone portamento: keep current position, don't retrigger
                        }} else {{
                            state.samplePos = 0.0;
                            // mikMod-style click ramp: keep mixVol at target so
                            // output isn't quenched, but mark a window for
                            // vlast-based blending in the per-sample mix.
                            // For sustained-pad samples (NNA=2 note-off, NNA=3
                            // fade), use a LONG ~100 ms ramp so the OLD voice
                            // fades smoothly into the NEW one — matches the
                            // envelope-release feel without producing the +6dB
                            // overlap swell that ghost mode creates.
                            // For percussive (NNA=0/1) keep tight 64-sample
                            // anti-click.
                            // OpenMPT NNA mechanism: NEW voice's mix volume
                            // STARTS at 0 and ramps UP to currentVolume.
                            // Ghost (dispatched above in _dispatchNNA)
                            // inherited the OLD voice's prior mixVol and
                            // ramps DOWN to its envelope-release target.
                            // Sum of the two ramps ≈ constant during the
                            // overlap window — no +6 dB swell at trigger.
                            // Reference: OpenMPT Snd_fx.cpp CheckNNA:
                            //   srcChn.rightVol = srcChn.leftVol = 0;
                            //   chn = srcChn;  // ghost keeps prior amp
                            // Then ProcessRamping ramps each per voice.
                            // Anti-click ramp only: 64-sample fade from
                            // vlast → target. The crossfade itself comes
                            // from the envelope: NEW voice's envelope
                            // attack rises 0→peak over the row, while the
                            // ghost rides natural envelope release. mikIT
                            // does the same — new voice flvolmul=0 the
                            // entire chord row, ghost decays via NFC.
                            state.mixVol = state.currentVolume;
                            state.volInc = 0;
                            state.clickWindow = 256;
                            state.clickRemaining = 256;
                            state.volumeFade = 1.0;
                            state.volumeFadeInc = 0;
                            state.targetVolume = 1.0;
                            if (note.effect === 0x9) state.samplePos = note.param * 256;
                            // Envelope reset on retrigger (mikIT semantics):
                            // NEW voice envX=0, plays through the envelope's
                            // attack ramp (0→peak over env-defined ticks).
                            // For Phase Strings (env (0,0)→(8,64)→(48,0)) the
                            // new voice is essentially silent for the first
                            // ~130 ms then rises to peak — by then samplePos
                            // has moved past the DC-offset attack region of
                            // the sample data so no click. The ghost rides
                            // its envelope release in parallel and the two
                            // form a natural envelope-shaped crossfade.
                            state.envX = 0;
                            state.envFin = false;
                            state.keyOn = true;
                            state.volFade = 65536;
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
                        if (noteSample === 0 && this.channels[ch].active) {{
                            const state = this.channels[ch];
                            state.samplePos = 0.0;
                            state.mixVol = state.currentVolume;
                            state.clickWindow = 256; state.clickRemaining = 256;
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

        // ── NNA ghost-voice decay ─────────────────────────────────────────
        // Once per tick, fade notefade-marked ghosts. Continue ghosts
        // (fadeAmt=0) keep playing until their sample exhausts (handled in
        // the mixing loop).
        this._decayGhosts();
        // (Envelope advance is now at the TOP of processTick — see comment
        //  there for the ordering rationale.)

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
                    const _pMin = this.numChannels > 4 ? 13 : 113;
                    state.period = Math.max(_pMin, state.period - param);
                    state.basePeriod = state.period;
                }}
                break;

            case 0x2: // Portamento down
                if (!tick0 && param > 0) {{
                    const _pMax = this.numChannels > 4 ? 13696 : 856;
                    state.period = Math.min(_pMax, state.period + param);
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

            case 0x8: // Set Panning (IT X effect, MOD 8xx). Param 0..FF
                // where 0=hard L, 0x80=center, 0xFF=hard R. Stored on the
                // channel state and consumed by the mix loop's per-channel
                // pan override path. Tick-0 only — no slide variant here.
                if (tick0) {{
                    state.panOverride = Math.max(0, Math.min(255, param));
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
                                state.mixVol = state.currentVolume;
                                state.clickWindow = 256; state.clickRemaining = 256;
                            }}
                            break;
                        case 0xC: // ECx — Note cut after x ticks
                            if (!tick0 && this.currentTick === val) {{
                                state.currentVolume = 0;
                                state.volume = 0;
                                // Anti-click: ramp mixVol DOWN to 0 over 256
                                // output samples instead of setting it to 0
                                // immediately (which would click). The
                                // existing per-sample slew at the top of the
                                // mix loop will carry mixVol → currentVolume
                                // (=0) at this rate, then clamp.
                                if (state.mixVol > 0) {{
                                    state.volInc = -state.mixVol / 256.0;
                                }}
                            }}
                            break;
                        case 0xD: // EDx — Note delay: trigger note x ticks late
                            if (!tick0 && this.currentTick === val && state._delayedNote) {{
                                const dn = state._delayedNote;
                                state._delayedNote = null;
                                const dnSample = dn.sample & 0x3F;
                                const info = modData.sampleMap[dnSample - 1];
                                state.sample = dnSample - 1;
                                // Same vol-col handling as immediate trigger.
                                let _dnTrigVol = info.volume;
                                if (dn.vol_col === 65) {{
                                    _dnTrigVol = state.currentVolume;   // preserve
                                }} else if (dn.vol_col && dn.vol_col > 0 && dn.vol_col <= 64) {{
                                    _dnTrigVol = dn.vol_col;
                                }} else if (dn.effect === 0xC) {{
                                    _dnTrigVol = Math.min(64, dn.param);
                                }}
                                state.volume = _dnTrigVol;
                                state.currentVolume = _dnTrigVol;
                                state.active = true;
                                state.samplePos = 0.0;
                                state.mixVol = state.currentVolume;
                                state.clickWindow = 256; state.clickRemaining = 256;
                                state.envX = 0;
                                state.keyOn = true;
                                state.volFade = 65536;
                                state.envFin = false;
                                if (dn.period > 0) {{
                                    state.period = dn.period;
                                    state.basePeriod = dn.period;
                                }}
                            }}
                            break;
                        case 0xE: // EEx — Pattern delay: extra ticks per row
                            if (tick0 && !this._patDelayActive) {{
                                // PT spec: row plays for (1+val) row-times.
                                // The old `val * speed` was off by one row's
                                // ticks because the delay decrement runs on
                                // every tick (including tick 0..speed-1, the
                                // row's natural time). Initialise with
                                // (val+1)*speed - 1 so total ticks at this
                                // row come out to (1+val)*speed exactly.
                                // Caught with TINYTUNE.MOD pat 32 r49 (EE8
                                // at speed 15) — was 280 ms short per row.
                                this._patDelayTicks = (val + 1) * this.speed - 1;
                                this._patDelayActive = true;
                            }}
                            break;
                        case 0x1: // E1x — Fine porta up (TICK 0 ONLY)
                            if (tick0 && val > 0) {{
                                const _epMin = this.numChannels > 4 ? 13 : 113;
                                state.period = Math.max(_epMin, state.period - val);
                                state.basePeriod = state.period;
                            }}
                            break;
                        case 0x2: // E2x — Fine porta down (TICK 0 ONLY)
                            if (tick0 && val > 0) {{
                                const _epMax = this.numChannels > 4 ? 13696 : 856;
                                state.period = Math.min(_epMax, state.period + val);
                                state.basePeriod = state.period;
                            }}
                            break;
                        case 0xA: // EAx — Fine volume slide up (TICK 0 ONLY)
                            if (tick0 && val > 0) {{
                                const newVol = Math.min(64, state.volume + val);
                                state.volume = newVol;
                                state.currentVolume = newVol;
                                state.targetVolume2 = newVol;
                                state.volumeRamping = false;
                            }}
                            break;
                        case 0xB: // EBx — Fine volume slide down (TICK 0 ONLY)
                            if (tick0 && val > 0) {{
                                const newVol = Math.max(0, state.volume - val);
                                state.volume = newVol;
                                state.currentVolume = newVol;
                                state.targetVolume2 = newVol;
                                state.volumeRamping = false;
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
        
        // Period clamp. Original 4-channel MOD spec: [113, 856] (3 octaves
        // around C-2). XM/IT support 10 octaves; their notes routinely
        // produce periods outside that range. Widen to [13, 13696] for
        // multi-channel formats (= IT note range C-0 to B-9). 4-channel
        // MOD keeps the original tight clamp for protracker compatibility.
        if (this.numChannels > 4) {{
            return Math.max(13, Math.min(13696, effectivePeriod));
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
            // IT formats supply per-channel default pan in the header (0..64,
            // 32=center). When present, build pan-left/right arrays from
            // those values so multi-channel IT files (e.g. GADGET ch4-11
            // are CENTER not LRRL-cycled). Otherwise fall back to MOD-style
            // hardcoded ch%4 alternation that matches Amiga panning.
            let chPanLeft, chPanRight;
            if (modData.channelPan && modData.channelPan.length >= this.numChannels) {{
                chPanLeft  = new Array(this.numChannels);
                chPanRight = new Array(this.numChannels);
                for (let _ci = 0; _ci < this.numChannels; _ci++) {{
                    // IT pan 0=hard-L, 32=center, 64=hard-R. Compress around
                    // SEP so hard-pan still leaves a small bleed (matches
                    // Amiga 87.5/12.5 ratio at hard pan).
                    const _pIT = Math.max(0, Math.min(64, modData.channelPan[_ci])) / 64.0;
                    const _r = panL_right + (panL_left - panL_right) * _pIT;  // 0=0.125, 0.5=0.5, 1=0.875
                    chPanLeft[_ci]  = 1.0 - _r;
                    chPanRight[_ci] = _r;
                }}
            }} else {{
                chPanLeft  = [panL_left, panL_right, panL_right, panL_left];
                chPanRight = [panL_right, panL_left, panL_left, panL_right];
            }}
            
            // When the song supplies per-channel pan (IT files), every
            // channel mixes directly into the cent (= main) bus using its
            // own pan — the MOD-style surr/cent ch%4-routing assumption
            // doesn't hold for arbitrary IT pan layouts. Fall back to the
            // surr+cent split only for MOD/XM where pan is implicit.
            const _useITPan = (modData.channelPan && modData.channelPan.length >= this.numChannels);

            for (let ch = 0; ch < this.numChannels; ch++) {{
                const state = this.channels[ch];
            // Surround channel pair (1-indexed, matches GLSL surr_channels).
            // [1,4] = outer LEFT pair (ch0,ch3); [2,3] = inner RIGHT pair (ch1,ch2)
            const surroundPair = [1, 4];
            const isSurrCh = (!_useITPan) && (surroundPair.includes((ch % 4) + 1));

                if (state.active && state.period > 0) {{
                    const sample = this.getSampleData(state.sample, state.samplePos);
                    // Loop-wrap click suppression for bass / sustained loops.
                    // When a sample loops and its boundary samples don't match
                    // amplitudes, each wrap produces a click. At a 100 Hz bass
                    // note that's ~100 clicks/sec = audible buzz. Detect the
                    // wrap moment (current position in loop < previous position
                    // in loop, modulo loop length) and arm a 16-sample ramp.
                    // Skip very tight loops (< 64 samples = single-cycle synth
                    // waveforms) where 16-sample smoothing would dull the
                    // tone significantly.
                    {{
                        const _smiL = modData.sampleMap[state.sample];
                        if (_smiL && (_smiL.loop_length || 0) >= 64) {{
                            const _bfL = _smiL.bw_factor || 1;
                            const _inLoop = ((state.samplePos / _bfL) - _smiL.loop_start);
                            const _curMod = ((_inLoop % _smiL.loop_length) + _smiL.loop_length) % _smiL.loop_length;
                            if (state._prevLoopPos !== undefined
                                && _curMod + 1.0 < state._prevLoopPos
                                && state.clickRemaining === 0) {{
                                // Wrap detected — short ramp smooths the
                                // amplitude discontinuity at the loop boundary.
                                state.clickWindow = 16;
                                state.clickRemaining = 16;
                            }}
                            state._prevLoopPos = _curMod;
                        }} else {{
                            state._prevLoopPos = undefined;
                        }}
                    }}
                    state.mixVol += state.volInc;
                    if (state.volInc > 0 && state.mixVol > state.currentVolume) {{
                        state.mixVol = state.currentVolume; state.volInc = 0;
                    }} else if (state.volInc < 0 && state.mixVol < state.currentVolume) {{
                        state.mixVol = state.currentVolume; state.volInc = 0;
                    }}
                    // XM env+fadeout multiplier (1.0 for non-XM/env-off voices).
                    const _info = modData.sampleMap[state.sample];
                    let envMul = 1.0;
                    if (_info && _info.env_pts && _info.env_pts.length >= 2) {{
                        envMul = (this._xmEnvLookup(_info, state.envX) / 64.0)
                               * (state.volFade / 65536.0);
                    }} else if (!state.keyOn) {{
                        // No envelope but key-off was issued: just fade via volFade.
                        envMul = state.volFade / 65536.0;
                    }}
                    // Anti-alias gain compensation. When a sample's playback
                    // freq exceeds Nyquist (e.g. IT chip leads at c5_speed
                    // 33000 Hz playing C-7 → freq 133 kHz), my cubic interp
                    // produces aliased output at FULL sample amplitude while
                    // openMPT's lanczos-3 attenuates above-Nyquist content.
                    // Without compensation the chip-lead-heavy passages end
                    // up 2-3× louder than openMPT. Use sqrt(nyq/freq) — a
                    // gentle one-pole-ish rolloff that approximates lanczos's
                    // attenuation slope. Only kicks in when actually above
                    // Nyquist, so non-IT samples are unaffected.
                    if (_info && _info.c5_speed && _info.c5_speed > 0) {{
                        const _freq = (_info.c5_speed * 428) / state.period;
                        const _nyq = this.sampleRate * 0.5;
                        if (_freq > _nyq) {{
                            // Linear (6 dB/oct) rolloff above Nyquist —
                            // matches openMPT's lanczos-3 slope for
                            // chip-lead-heavy passages. EXCEPT when
                            // c5_speed is unusually high (> 50000 Hz),
                            // which signals a single-cycle synth waveform
                            // (e.g. inst 27 "Sinewave-envelope" at
                            // c5=168000): the source content is just a
                            // smooth fundamental with NO significant high-
                            // freq content, so cubic interp doesn't
                            // actually alias and the 95%+ rolloff buries
                            // the voice. Floor at 0.3 to keep inst 27
                            // audible — without it, GADGET.IT's main
                            // sinewave melody gets buried by the bass and
                            // is heard as an unwanted "suspended lower
                            // tone".
                            const _aaFloor = (_info.c5_speed > 50000) ? 0.3 : 0.0;
                            envMul *= Math.max(_aaFloor, _nyq / _freq);
                        }}
                    }}
                    const cv = (state.mixVol / 64.0) * (state.channelVolume / 64.0) * envMul;
                    const target = sample * cv;
                    // Post-end anti-click: when samplePos has crossed the
                    // non-looping sample's end (getSampleData now returns 0)
                    // and we still have a non-zero vlast, arm the existing
                    // click ramp. The ramp formula `vlast*(1-ct) + target*ct`
                    // with target=0 naturally decays vlast → 0 over 256
                    // samples. The sample plays through its full natural
                    // length first — this only kicks in *after* the data
                    // ends, so no track-path is cut short.
                    if (_info && (_info.loop_length || 0) <= 2 && _info.length > 0
                        && state.clickRemaining === 0
                        && Math.abs(state.vlast) > 1e-6) {{
                        const _bf = _info.bw_factor || 1;
                        if (state.samplePos / _bf >= _info.length) {{
                            state.clickWindow = 256;
                            state.clickRemaining = 256;
                        }}
                    }}
                    // mikMod-style click ramp: blend output from vlast (last
                    // sample's actual output) toward target over 256 samples
                    // post-trigger (~5.8 ms @44.1kHz). Eliminates the
                    // dip-to-silence that happened when state.mixVol was reset
                    // to 0 and the percussive click on bass/drum retriggers.
                    let s;
                    if (state.clickRemaining > 0) {{
                        const _w = state.clickWindow || 64;
                        const _ct = (_w - state.clickRemaining) / _w;
                        s = state.vlast * (1.0 - _ct) + target * _ct;
                        state.clickRemaining--;
                    }} else {{
                        s = target;
                    }}
                    state.vlast = s;
                    
                    // Per-channel pan: prefer panOverride from MOD 8xx /
                    // IT X effect when set, else fall back to header-default
                    // chPanLeft/chPanRight (IT-pan-aware or MOD-style ch%4).
                    let _pl, _pr;
                    if (state.panOverride !== undefined) {{
                        // Param 0..FF, 0=L, 0x80=center, 0xFF=R.
                        const _p = state.panOverride / 255.0;
                        _pr = panL_right + (panL_left - panL_right) * _p;
                        _pl = 1.0 - _pr;
                    }} else {{
                        _pl = chPanLeft[ch % chPanLeft.length];
                        _pr = chPanRight[ch % chPanRight.length];
                    }}
                    if (isSurrCh) {{
                        surrL += s * _pl;
                        surrR += s * _pr;
                    }} else {{
                        mixL += s * _pl;
                        mixR += s * _pr;
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
                    const _smInfo = modData.sampleMap[state.sample] || {{}};
                    const smpFt   = _smInfo.finetune || 0;
                    const smpC5   = _smInfo.c5_speed || 0;
                    // samplePos is in original (uncompressed) sample space.
                    // getSampleData maps it to compressed space via pos/bw_factor — so
                    // freq must NOT be divided by bw_factor here (would double-divide).
                    const freq = this.periodToFreq(effectivePeriod, smpFt, smpC5);
                    state.samplePos += freq / this.sampleRate;
                }}

                // ── Ghost voices (NNA) ────────────────────────────────────
                // Continue/notefade voices left over from previous note
                // triggers. We mix them in at their frozen pitch and volume,
                // panned to the same channel pair the voice belongs to.
                // Per-tick fade is applied in processTick (see below) — not
                // here per sample, so amplitude is constant within a tick.
                for (let gi = 0; gi < state.ghostVoices.length; gi++) {{
                    const g = state.ghostVoices[gi];
                    if (!g.active || g.period <= 0 || g.volume <= 0) continue;
                    const gs = this.getSampleData(g.sample, g.samplePos);
                    const _ginfo = modData.sampleMap[g.sample];
                    let _gEnvMul = 1.0;
                    if (_ginfo && _ginfo.env_pts && _ginfo.env_pts.length >= 2) {{
                        _gEnvMul = (this._xmEnvLookup(_ginfo, g.envX) / 64.0)
                                 * (g.volFade / 65536.0);
                    }} else if (!g.keyOn) {{
                        _gEnvMul = g.volFade / 65536.0;
                    }}
                    // Anti-alias gain compensation (mirrors active-voice path,
                    // line ~3858). Ghosts must rolloff above-Nyquist content
                    // the same way active voices do — otherwise a ghost from
                    // a high-c5_speed sample plays at full amplitude while its
                    // sibling active voice is attenuated to ~0.4×, and the
                    // ghost dominates the chord-change overlap by ~3 dB,
                    // producing the audible jump-at-trigger that doesn't
                    // exist in mikIT/OpenMPT renders.
                    if (_ginfo && _ginfo.c5_speed && _ginfo.c5_speed > 0) {{
                        const _gfreq = (_ginfo.c5_speed * 428) / g.period;
                        const _gnyq  = this.sampleRate * 0.5;
                        if (_gfreq > _gnyq) {{
                            const _gaaFloor = (_ginfo.c5_speed > 50000) ? 0.3 : 0.0;
                            _gEnvMul *= Math.max(_gaaFloor, _gnyq / _gfreq);
                        }}
                    }}
                    // Use frozen mixVol (where the active voice's ramp was at
                    // dispatch time), not the channel target volume — otherwise
                    // a mid-ramp dispatch causes the ghost to jump to full
                    // target instantly = audible click. Anti-ramp toward target
                    // continues per-sample for smooth completion of the ramp
                    // that was in progress.
                    g.mixVol += g.volInc;
                    if (g.volInc > 0 && g.mixVol > g.volume) {{ g.mixVol = g.volume; g.volInc = 0; }}
                    else if (g.volInc < 0 && g.mixVol < g.volume) {{ g.mixVol = g.volume; g.volInc = 0; }}
                    // Ghost output uses ramping mixVol (now ramping DOWN to
                    // 0 over click window per OpenMPT NNA dispatch).
                    const gcv = (g.mixVol / 64.0) * (state.channelVolume / 64.0) * _gEnvMul;
                    const gOut = gs * gcv;
                    if (isSurrCh) {{
                        surrL += gOut * chPanLeft[ch % chPanLeft.length];
                        surrR += gOut * chPanRight[ch % chPanRight.length];
                    }} else {{
                        mixL += gOut * chPanLeft[ch % chPanLeft.length];
                        mixR += gOut * chPanRight[ch % chPanRight.length];
                    }}
                    const _gInfo2 = modData.sampleMap[g.sample] || {{}};
                    const gFt = _gInfo2.finetune || 0;
                    const gC5 = _gInfo2.c5_speed || 0;
                    const gFreq = this.periodToFreq(g.period, gFt, gC5);
                    g.samplePos += gFreq / this.sampleRate;
                    // Deactivate ghost when its sample is exhausted (non-
                    // looping) — getSampleData returns 0 past the end, but
                    // keeping the ghost active wastes mix work.
                    const gInfo = modData.sampleMap[g.sample];
                    if (gInfo && gInfo.loop_length <= 2 &&
                        g.samplePos >= gInfo.length / (gInfo.bw_factor || 1)) {{
                        g.active = false;
                    }}
                }}
            }}

            // Mix normalization. For 4-channel MOD: 0.5 (keeps total ≤ 2.0
            // when 4 voices play at full vol — original tuning). For
            // multi-channel XM/IT/S3M formats: 1/N is too aggressive
            // (10ch XM ends up 2.5× quieter than MOD). openMPT scales by
            // ~1/sqrt(N) so songs maintain similar perceived loudness as
            // channel count grows. Use sqrt-based scaling for >4 channels,
            // capped so a 4-ch song still gets the original 0.5.
            // Normalization: 4-ch MOD keeps the original 0.5 (preserves the
            // long-tuned mix). For multi-channel XM/IT/S3M, scale so a single
            // voice at full volume reaches near unity — that's what
            // saturates the oscilloscope. The AdaptiveLimiter (1ms minimum
            // attack, 8ms base attack, stereo-linked) catches peaks when
            // multiple voices stack, so going hot here doesn't blow up.
            //
            // 1.0/sqrt(N) was conservative; with the limiter we can go to
            // ~0.7 (single voice fills the buffer at vol=64).
            const normFactor = (this.numChannels <= 4)
                ? 2.0 / this.numChannels
                : 0.85;
            // Master volume reduced to match mikIT-style headroom for NNA
            // ghost overlap. mikIT divides per-voice gain by total channels
            // (~64), giving low absolute output but room for many concurrent
            // voices. We use a lower master vol to give the same headroom.
            const vol = this._volume !== undefined ? this._volume : 0.45;
            // IT mix/global volume.
            // OpenMPT applies MixVol as a SCALAR with unity at MV=48 (the
            // IT default) — not at MV=128. So MV=40 → 40/48 = 0.833 (very
            // mild reduction), not 40/128 = 0.3125 (-10 dB cut).
            // The /128 normalization made our mix ~3-4× too quiet vs
            // OpenMPT on jeff (mv=40) and similar IT files. Reference:
            // OpenMPT MPTM_DefaultMixVol = 48.
            const _gv  = (modData.globalVol ?? 128) / 128.0;   // 0..1, default=1
            const _mv  = (modData.mixVol    ?? 48 ) / 48.0;    // 0..2.67, default=1
            const _gMix = _gv * _mv;

            // Apply Only3D to surround bus (ch0 + ch3 = outer LEFT pair = "Surround L/R")
            let sL = surrL * normFactor * vol * _gMix;
            let sR = surrR * normFactor * vol * _gMix;
            if (this._only3d && this._only3dDepth > 0) {{
                [sL, sR] = this._only3d.process(sL, sR);
            }}
            
            // Center bus (ch1 + ch2 = inner RIGHT pair) passes through dry
            const cL = mixL * normFactor * vol * _gMix;
            const cR = mixR * normFactor * vol * _gMix;
            
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
            // Soft-limit to 1.1 before fat_cs1 (polynomial diverges past |x|=1.1).
            // Knee at 0.95→1.1 preserves transient shape vs a hard clamp.
            const _softLim11 = (x) => {{
                const ax = Math.abs(x), T = 0.95, HEAD = 0.15, over = ax - T;
                return ax <= T ? x : Math.sign(x) * (T + (HEAD * over) / (over + HEAD));
            }};
            const _csInL = _softLim11(outL);
            const _csInR = _softLim11(outR);
            outL = outL * (1.0 + fat_cs1(_csInL) * FAT_AMOUNT);
            outR = outR * (1.0 + fat_cs1(_csInR) * FAT_AMOUNT);

            // ── AdaptiveLimiter (sits after PhatBass + FAT4X) ────────────
            // Stereo-linked, 1–8 ms adaptive attack, 30–110 ms adaptive
            // release. Catches the bass overshoot from PhatBass + the
            // saturation overshoot from FAT4X without zipper artefacts on
            // big transients (70/30 smoothing).
            const _lim = this._limiter.process(outL, outR);
            // Post-limiter makeup gain + hard-clip. Pushes the signal up
            // against the ±1 boundary for maximum perceived loudness.
            // The limiter has already softly tamed catastrophic peaks; this
            // gain stage hard-clips whatever's left to fill the rail.
            // Reduced from 1.7 → 1.0 to match mikIT's clean-output chain
            // (no post-limiter makeup; output sits below the rail).
            const _MAKEUP = 1.0;
            leftChannel[offset + i]  = Math.max(-1, Math.min(1, _lim[0] * _MAKEUP));
            rightChannel[offset + i] = Math.max(-1, Math.min(1, _lim[1] * _MAKEUP));
            
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
        
        for (let ch = 0; ch < this.numChannels; ch++) {{
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
      const y=H/2-Math.max(-1.0,Math.min(1.0,buf[Math.floor(x*step)]*64.0))*H*0.48;
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
    
    // Drag scrolling - only in tracker area, pixel-based offset
    let isDragging = false;
    let startX = 0;
    let trackOffset = 0;
    const hdrEl = document.getElementById('trkHeader');
    const oscCanvas = document.getElementById('oscCanvas');
    
    // Only start drag if clicking anywhere in tracker area (header, body, etc)
    const trackerParent = document.getElementById('tracker');
    trackerParent.addEventListener('mousedown', function(ev){{
      console.log('TRACKER MOUSEDOWN!', ev.clientX);
      isDragging = true;
      startX = ev.clientX;
      document.body.style.cursor = 'grabbing';
      ev.preventDefault();
      ev.stopPropagation();
      console.log('isDragging set to true');
    }});
    
    document.addEventListener('mousemove', function(ev){{
      if(!isDragging) return;
      trackOffset = ev.clientX - startX;
      console.log('DRAGGING:', trackOffset);
      // Apply transform to both header and body
      const tx = `translateX(${{trackOffset}}px)`;
      hdrEl.style.transform = tx;
      trackerEl.style.transform = tx;
    }});
    
    document.addEventListener('mouseup', function(){{
      if(isDragging){{
        isDragging = false;
        document.body.style.cursor = '';
        // Snap to nearest track
        const TRACK_WIDTH = 180;
        const snapDelta = Math.round(-trackOffset / TRACK_WIDTH);
        const next = Math.max(0, Math.min(maxFirstTrack, firstTrack + snapDelta));
        firstTrack = next;
        trackOffset = 0;
        hdrEl.style.transform = '';
        trackerEl.style.transform = '';
        updateHeaderAndRange();
        updateTracker();
      }}
    }});
    
    // Oscillo/Spectrum canvas click toggles mode (handled by shader)
    oscCanvas.addEventListener('click', function(ev){{
      ev.stopPropagation();  // Don't let this bubble to tracker
      // Shader reads mouse click from iMouse and toggles mode automatically
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

def create_segment_player_html(player, segs, mod_samples, filename, title=""):
    """Create HTML player using pre-baked ITPlayer VoiceSegments (render_pcm.py approach).

    No JS effect engine. Python ITPlayer produces tick-accurate (freq,vol,pan) data;
    JS mirrors render_pcm.py render loop using AudioBuffer — plays back once rendered.

    player      : ITPlayer with player.samples (ITSample, loop info)
    segs        : List[VoiceSegment] from player.run()
    mod_samples : list of sample dicts {'data': np.int8 array, ...} — raw PCM,
                  index-aligned with player.samples (XMFile/ITFile output)
    filename    : output HTML path
    title       : song title for display
    """
    import json as _json

    if not title:
        title = getattr(player, 'title', '') or os.path.splitext(os.path.basename(filename))[0]

    CHUNK_SIZE = 50000

    # ── Sample storage (bw_compress same as create_fixed_player_html) ──────────
    sample_map = []
    all_samples = []
    cur_pos = 0

    for i, smp in enumerate(player.samples):
        sdict = mod_samples[i] if i < len(mod_samples) else None
        raw = (sdict.get('data') if isinstance(sdict, dict) else
               getattr(sdict, 'data', None)) if sdict is not None else None

        if raw is not None and hasattr(raw, '__len__') and len(raw) > 1 and smp.length > 0:
            # Store RAW int8 PCM (no bw_compress) — bw_compression's LPF was
            # introducing aliasing artifacts when high-pitched samples
            # replayed at extreme rates (BUTTERFL 0:12 distorto-vibrato).
            # For the HTML player we don't have GLSL's tight storage budget,
            # so we keep full fidelity. The renderer's step formula simplifies
            # to step = freq / SR (bw_factor=1).
            if hasattr(raw, 'dtype') and raw.dtype == np.int16:
                raw_i8 = (raw.astype(np.float32) / 256.0).astype(np.int8)
            else:
                raw_i8 = np.asarray(raw, dtype=np.int8)

            data_f = np.concatenate([
                raw_i8.astype(np.float32) / 128.0,
                np.zeros(32, dtype=np.float32),   # guard
            ])

            ls_c = smp.loop_start if smp.has_loop else 0
            le_c = smp.loop_end   if smp.has_loop and smp.loop_end > smp.loop_start else 0
            lt   = (2 if smp.bidi_loop else 1) if smp.has_loop else 0

            sample_map.append({
                'start':      cur_pos,
                'length':     len(raw_i8),
                'loop_start': ls_c,
                'loop_end':   le_c,
                'loop_type':  lt,
                'bw_factor':  1,
            })
            all_samples.extend(data_f.tolist())
            cur_pos += len(data_f)
        else:
            sample_map.append({'start': 0, 'length': 0,
                               'loop_start': 0, 'loop_end': 0,
                               'loop_type': 0, 'bw_factor': 1})

    # ── Segment data: [si, ch, sp0, [[t,f,v,p],...], et] ──────────────────────
    # ch is the physical channel (-1 for NNA ghost). Used by the solo UI to
    # mute/unmute voices by channel for debugging which channel produces a bug.
    seg_data = []
    for seg in segs:
        if not seg.tick_states:
            continue
        si = seg.sample_idx
        if si < 0 or si >= len(sample_map) or sample_map[si]['length'] == 0:
            continue
        ts = [[int(t), round(float(f), 1), round(float(v), 5), round(float(p), 5)]
              for t, f, v, p, _ in seg.tick_states]
        seg_data.append([si, seg.channel, round(float(seg.tick_states[0][4]), 1), ts, seg.end_tick])

    # ── Metrics ──────────────────────────────────────────────────────────────
    tps = player.initial_tempo * 2.0 / 5.0
    max_tick = 0
    for seg in segs:
        if not seg.tick_states:
            continue
        et = seg.end_tick if seg.end_tick >= 0 else seg.tick_states[-1][0] + 1
        if et > max_tick:
            max_tick = et
    total_sec = max_tick / tps + 2.0

    print(f"   Segments: {len(seg_data)}, samples: {len([m for m in sample_map if m['length']>0])}, "
          f"duration: {total_sec-2:.1f}s")

    # ── Sample chunks ─────────────────────────────────────────────────────────
    sample_chunks = [all_samples[i:i+CHUNK_SIZE] for i in range(0, len(all_samples), CHUNK_SIZE)]
    chunk_decls   = "\n".join(f"const sampleChunk{i}={_json.dumps(c)};"
                              for i, c in enumerate(sample_chunks))
    chunk_concat  = "[..." + ", ...".join(f"sampleChunk{i}" for i in range(len(sample_chunks))) + "]"

    seg_json      = _json.dumps(seg_data, separators=(',', ':'))
    smap_json     = _json.dumps(sample_map, separators=(',', ':'))

    # ── Pattern + row event data for tracker UI ──────────────────────────────
    # Encode cells as [note, inst, vol, eff, par] or 0 if empty. ITCell has
    # has_note/has_inst/has_vol/has_eff flags; non-empty cells get the array.
    patterns_data = []
    for pat in player.patterns:
        pat_rows = []
        for row in pat:
            row_cells = []
            for c in row:
                if hasattr(c, 'note') and (c.has_note or c.has_inst or c.has_vol or c.has_eff):
                    row_cells.append([
                        c.note if c.has_note else 0,
                        c.inst if c.has_inst else 0,
                        c.vol if c.has_vol else 255,
                        c.eff if c.has_eff else 0,
                        c.par if c.has_eff else 0,
                    ])
                else:
                    row_cells.append(0)
            pat_rows.append(row_cells)
        patterns_data.append(pat_rows)
    patterns_json = _json.dumps(patterns_data, separators=(',', ':'))
    orders_json   = _json.dumps(list(player.orders), separators=(',', ':'))
    # Row events: (abs_tick, song_pos, row, pat_no) — emitted at row-load time
    rev = getattr(player, 'row_events', [])
    rev_json = _json.dumps([[t, sp, r, pn] for t, sp, r, pn in rev], separators=(',', ':'))
    chan_pan_json = _json.dumps(list(player.channel_pan), separators=(',', ':'))
    num_ch = player.num_channels

    fmt_fn = lambda s: (str(int(s//60)) + ':' + str(int(s%60)).zfill(2))
    dur_str = fmt_fn(total_sec - 2)

    # Pre-build channel panel HTML — CH# + S/M buttons on one row (CH# uses just
    # the number to fit narrow panels). Note and volume bar stack below.
    _ch_panel_html = "\n".join(
        f'    <div class="ch-panel" id="chP{i}" data-ch="{i}">'
        f'<div class="ch-header-row">'
        f'<span class="ch-num" title="Channel {i+1}">{i+1:02d}</span>'
        f'<span class="ch-btn ch-btn-s" id="chS{i}" title="Solo CH{i+1}">S</span>'
        f'<span class="ch-btn ch-btn-m" id="chM{i}" title="Mute CH{i+1}">M</span>'
        f'</div>'
        f'<div class="ch-note dim" id="chN{i}">---</div>'
        f'<div class="ch-bar-wrap"><div class="ch-bar" id="chB{i}"></div></div></div>'
        for i in range(num_ch)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title} — MOD Player</title>
<style>
:root{{--bg0:#0d0d12;--bg1:#13131a;--bg2:#1c1c26;--bg3:#252533;
  --acc:#3d8ef0;--acc2:#5af0c8;--txt:#c8ccd8;--dim:#555870;--bdr:#2a2a3c;
  --font:'Courier New',monospace}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg0);color:var(--txt);font-family:var(--font);
  display:flex;flex-direction:column;align-items:center;min-height:100vh}}
#topbar{{width:100%;background:var(--bg1);border-bottom:1px solid var(--bdr);
  display:flex;align-items:center;gap:16px;padding:10px 20px}}
.logo{{color:var(--acc);font-size:13px;letter-spacing:2px;font-weight:bold}}
.ttl{{color:var(--acc2);font-size:14px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.spc{{flex:1}}
.meta{{color:var(--dim);font-size:11px;white-space:nowrap}}
#loadSection{{width:100%;padding:40px 20px;display:flex;flex-direction:column;
  align-items:center;gap:16px}}
.load-label{{color:var(--dim);font-size:13px;letter-spacing:2px}}
#loadBar{{width:320px;height:5px;background:var(--bg3);border-radius:3px;overflow:hidden}}
#loadFill{{height:100%;background:var(--acc);width:0%;transition:width .1s}}
.load-pct{{color:var(--acc2);font-size:22px;font-weight:bold;letter-spacing:2px}}
#playerSection{{width:100%;display:none;flex-direction:column}}
#seekBar{{width:100%;height:5px;background:var(--bg3);cursor:pointer;position:relative}}
#seekFill{{height:100%;background:var(--acc);width:0%}}
#seekFill::after{{content:'';position:absolute;right:-5px;top:-3px;width:10px;height:10px;
  border-radius:50%;background:var(--acc)}}
#ctrls{{width:100%;background:var(--bg1);border-bottom:1px solid var(--bdr);
  display:flex;align-items:center;gap:8px;padding:8px 16px}}
.btn{{background:var(--bg2);border:1px solid var(--bdr);color:var(--txt);
  font-family:var(--font);font-size:13px;padding:6px 14px;cursor:pointer;border-radius:3px;
  transition:all .15s;letter-spacing:1px;display:inline-flex;align-items:center;justify-content:center;
  min-width:42px;height:30px}}
.btn:hover{{border-color:var(--acc)}}
.ico-play{{width:0;height:0;border-left:11px solid #3df0a0;border-top:7px solid transparent;border-bottom:7px solid transparent;margin-left:2px}}
.ico-pause{{width:4px;height:14px;background:#f0d040;box-shadow:8px 0 0 #f0d040;margin-right:8px}}
.ico-stop{{width:11px;height:11px;background:#e8e8e8}}
#timeDisp{{margin-left:auto;color:var(--dim);font-size:12px;letter-spacing:1px}}
#infoGrid{{width:100%;display:grid;grid-template-columns:repeat(4,1fr);
  border-bottom:1px solid var(--bdr)}}
.ic{{padding:10px 16px;border-right:1px solid var(--bdr);background:var(--bg1)}}
.ic:last-child{{border-right:none}}
.il{{font-size:10px;color:var(--dim);letter-spacing:2px;text-transform:uppercase;margin-bottom:4px}}
.iv{{font-size:22px;color:var(--acc2);letter-spacing:1px;font-weight:bold}}
.iv .sub{{color:var(--dim);font-size:13px;font-weight:normal}}
#volWrap{{display:flex;align-items:center;gap:8px;color:var(--dim);font-size:12px;margin-left:8px}}
#volSlider{{-webkit-appearance:none;width:80px;height:4px;border-radius:2px;background:var(--bg3);outline:none;cursor:pointer}}
#volSlider::-webkit-slider-thumb{{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:var(--acc);cursor:pointer}}
#channels{{width:100%;display:grid;grid-template-columns:repeat({num_ch},1fr);gap:1px;background:var(--bdr);border-bottom:1px solid var(--bdr)}}
.ch-panel{{background:var(--bg1);padding:5px 4px;min-width:0;transition:opacity .15s}}
.ch-panel.muted{{opacity:0.35}}
.ch-header-row{{display:flex;align-items:center;gap:2px;margin-bottom:4px;height:14px}}
.ch-num{{font-size:10px;color:#c8ccd8;font-weight:bold;letter-spacing:0;
  flex:1;min-width:0;overflow:hidden;text-overflow:clip;white-space:nowrap}}
.ch-btn-row{{display:flex;gap:2px;margin-bottom:4px;justify-content:flex-start}}
.ch-btn{{display:inline-flex;align-items:center;justify-content:center;
  width:11px;height:12px;font-size:8px;font-weight:bold;border-radius:2px;
  background:var(--bg2);color:var(--dim);cursor:pointer;
  border:1px solid var(--bdr);user-select:none;line-height:1;flex:0 0 auto}}
.ch-btn:hover{{background:var(--bg3);color:var(--txt)}}
.ch-btn-s.active{{background:#f0d040;color:#000;border-color:#f0d040}}
.ch-btn-m.active{{background:#e04040;color:#fff;border-color:#e04040}}
#renderPill{{display:none;position:absolute;right:20px;top:50%;
  transform:translateY(-50%);background:var(--bg2);border:1px solid var(--bdr);
  padding:2px 8px;border-radius:10px;font-size:10px;color:var(--acc2);
  letter-spacing:1px}}
#renderPill.show{{display:inline-block}}
#topbar{{position:relative}}
.ch-note{{font-size:14px;font-weight:bold;letter-spacing:1px;height:18px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.ch-bar-wrap{{height:3px;background:var(--bg3);border-radius:2px;overflow:hidden;margin-top:4px}}
.ch-bar{{height:100%;width:0%;border-radius:2px;transition:width .08s ease-out}}
.ch-note.dim{{color:#252535;font-weight:normal}}
#tracker{{width:100%;background:var(--bg0);border-bottom:1px solid var(--bdr);
  overflow:hidden;font-size:12px;font-family:var(--font)}}
.trk-header{{display:flex;background:var(--bg2);border-bottom:1px solid var(--bdr);padding:3px 0}}
.trk-col-hdr{{font-size:10px;letter-spacing:1px;color:#e8e8e8;font-weight:bold;text-transform:uppercase;padding:2px 6px;flex:1;min-width:0}}
.trk-col-hdr:first-child{{flex:0 0 44px;text-align:right;color:var(--dim);font-weight:normal}}
.trk-row{{display:flex;border-bottom:1px solid #0f0f18}}
.trk-row.current{{background:rgba(255,255,255,0.07)!important;border-left:3px solid var(--acc)}}
.trk-row:nth-child(even){{background:#0d0d14}}
.trk-row:nth-child(odd){{background:#101018}}
.trk-rownum{{flex:0 0 44px;color:var(--dim);font-size:10px;padding:2px 8px;align-self:center;text-align:right}}
.trk-row.current .trk-rownum{{color:var(--acc);font-weight:bold}}
.trk-cell{{flex:1;padding:2px 6px;font-family:var(--font);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}}
.trk-empty{{color:#252535}}
.trk-note{{color:var(--acc2)}}
.trk-samp{{color:#6677aa;font-size:10px}}
.trk-eff{{color:#557799;font-size:10px}}
#footer{{width:100%;padding:10px 20px;color:var(--dim);font-size:11px;
  display:flex;justify-content:space-between;border-top:1px solid var(--bdr);margin-top:auto}}
</style>
</head>
<body>
<div id="topbar">
  <span class="logo">MOD PLAYER</span>
  <span class="ttl">{title}</span>
  <span class="spc"></span>
  <span class="meta">{len(seg_data)} segs &bull; {dur_str}</span>
  <span id="renderPill">RERENDER <span id="renderPct">0</span>%</span>
</div>

<div id="loadSection">
  <div class="load-label">RENDERING AUDIO</div>
  <div id="loadBar"><div id="loadFill"></div></div>
  <div class="load-pct"><span id="loadPct">0</span>%</div>
  <div style="color:var(--dim);font-size:11px">Pre-baking {len(seg_data)} voice segments via Python ITPlayer (MikIT port)</div>
</div>

<div id="playerSection">
  <div id="seekBar"><div id="seekFill"></div></div>
  <div id="ctrls">
    <button class="btn" id="btnPlayPause" title="Play"><span class="ico-play" id="icoPP"></span></button>
    <button class="btn" id="btnStop" title="Stop"><span class="ico-stop"></span></button>
    <span id="volWrap">VOL <input type="range" id="volSlider" min="0" max="100" value="100"></span>
    <span id="timeDisp">0:00 / {dur_str}</span>
  </div>
  <div id="infoGrid">
    <div class="ic"><div class="il">POSITION</div><div class="iv"><span id="iPos">0</span><span class="sub">/{len(player.orders)}</span></div></div>
    <div class="ic"><div class="il">ROW</div><div class="iv"><span id="iRow">0</span><span class="sub">/64</span></div></div>
    <div class="ic"><div class="il">TEMPO</div><div class="iv">{player.initial_tempo} <span class="sub">BPM</span></div></div>
    <div class="ic"><div class="il">CHANNELS</div><div class="iv">{num_ch}</div></div>
  </div>
  <div id="channels">
{_ch_panel_html}
  </div>
  <div id="tracker">
    <div class="trk-header" id="trkHeader"></div>
    <div id="trkBody"></div>
  </div>
</div>

<div id="footer">
  <span>Python ITPlayer (MikIT port) &rarr; segment pre-bake &rarr; Web Audio API</span>
  <span>MOD2GLSL</span>
</div>

<script>
// ── Embedded data ─────────────────────────────────────────────────────────────
const INITIAL_TEMPO = {player.initial_tempo};
const NUM_CHANNELS  = {num_ch};
const sampleMap = {smap_json};
const segments  = {seg_json};
const patterns  = {patterns_json};
const orders    = {orders_json};
const rowEvents = {rev_json};
const channelPan = {chan_pan_json};
{chunk_decls}
const allSamples = {chunk_concat};

// Mute/solo state. channelMuted[i] is the FINAL gate used by the renderer;
// it's computed from channelSoloed[] + channelExplicitMuted[] (see below).
const channelMuted = new Array(NUM_CHANNELS).fill(false);
const channelSoloed = new Array(NUM_CHANNELS).fill(false);

// ── Renderer (mirrors render_pcm.py) ──────────────────────────────────────────
// FADE = voice start/end micro-fade length. 128 samples = 2.9ms — long enough
// to fully mask sample[0] != 0 click on note-trigger and to avoid the audible
// 32-sample step-down at the end of a faded chunk.
// NOCLICK_N = inter-tick vol/pan smoothing ramp length. Vol/pan stay constant
// within each tick (matches MikIT's behavior), with this many samples at the
// start of each tick ramping from the previous tick's vol to the current's.
// MikIT uses `frequency / 689 = 64` samples at 44100 Hz — see mdrv_mix.cpp:746.
const SR = 44100, GUARD = 32, FADE = 128, NOCLICK_N = 64;
const TPS = INITIAL_TEMPO * 2.0 / 5.0;

// Build extended sample buffers (float32 with guard-extension at loop boundary).
// For forward loops, crossfade the last CF samples of the loop into the first CF
// of the loop so ext[le-1] ≈ ext[ls]. Without this, the abrupt value jump every
// loop wrap repeats at freq/span Hz — for ~1845-sample loops at 18774 Hz that is
// 10.18 Hz, right in the audible click/tremolo range (the BUTTERFL CH1 bug and
// the 0:12 "mad vibrato" bug are both this same mechanism).
const smpExt = new Array(sampleMap.length).fill(null);
for (let si = 0; si < sampleMap.length; si++) {{
  const m = sampleMap[si];
  if (!m.length) continue;
  const ext = new Float32Array(m.length + GUARD);
  for (let k = 0; k < m.length; k++) ext[k] = allSamples[m.start + k];
  const ls = m.loop_start, le = m.loop_end, lt = m.loop_type;
  if (lt > 0 && le > ls) {{
    const span = le - ls;
    // Guard fill: GUARD samples past le for linear interpolation across wrap.
    for (let t = 0; t < GUARD; t++) {{
      ext[le + t] = (lt === 2)
        ? (le - 1 - t >= 0 ? ext[le - 1 - t] : 0.0)
        : ext[ls + (t % span)];
    }}
    // Forward-loop crossfade: blend ext[le-cf..le-1] toward ext[ls..ls+cf-1] so
    // the loop-wrap boundary is smooth. Uses a raised-cosine (Hann) window:
    // zero slope at both ends eliminates the slope corners that the old linear
    // ramp left at the crossfade entry/exit (which themselves could click at
    // the loop rate). cf=256 gives ~13 ms of smoothing at typical playback
    // speeds — enough to suppress loop-boundary clicks down to inaudibility
    // even on tight 1845-sample loops (BUTTERFL CH1 forward-loop / D-effects).
    if (lt === 1) {{
      const cf = Math.min(256, span >> 2);
      for (let t = 0; t < cf; t++) {{
        const f = 0.5 * (1.0 - Math.cos(Math.PI * (t + 1) / (cf + 1)));
        ext[le - cf + t] = ext[le - cf + t] * (1.0 - f) + ext[ls + t] * f;
      }}
    }}
  }}
  smpExt[si] = ext;
}}

// Compute song duration in output samples
let maxTick = 0;
for (const seg of segments) {{
  const ts = seg[3], et = seg[4];
  const end = et >= 0 ? et : (ts.length ? ts[ts.length-1][0]+1 : 0);
  if (end > maxTick) maxTick = end;
}}
const totalSamples = Math.ceil(maxTick / TPS * SR) + SR * 2;
const bufL = new Float32Array(totalSamples);
const bufR = new Float32Array(totalSamples);

function renderSeg(seg) {{
  const si = seg[0], ch = seg[1], sp0 = seg[2], ts = seg[3], et = seg[4];
  if (si < 0 || si >= smpExt.length || !smpExt[si]) return;
  if (!ts.length) return;
  // Solo/mute: skip segments on muted channels. NNA ghost voices (ch=-1)
  // follow their parent channel's mute state — they're stored with ch>=0 in
  // the segment encoder since they were spawned by that physical channel.
  if (ch >= 0 && ch < NUM_CHANNELS && channelMuted[ch]) return;
  const ext = smpExt[si], m = sampleMap[si];
  const bf = m.bw_factor, ls = m.loop_start, le = m.loop_end, lt = m.loop_type;
  const span = (le > ls) ? (le - ls) : 1.0;
  const smpLen = m.length, extLen = ext.length;

  // Pre-compute segment output range. NOTE: previously I extended segEnd by
  // FADE samples to overlap with the next segment's fade-in (crossfade), but
  // that added 128 samples of OLD-voice contribution past its logical end —
  // for samples that didn't actually have audio at that point (sample
  // exhausted past end) this could replay loop content unintentionally.
  // The MikIT-style noclick at tick boundaries handles the typical click
  // case (vol jumps); fade-in at segment start handles silence-to-audio.
  const segStart = Math.round(ts[0][0] / TPS * SR);
  const segEndTick = (et >= 0) ? et : (ts[ts.length-1][0] + 1);
  let segEnd = Math.round(segEndTick / TPS * SR);
  if (segEnd > totalSamples) segEnd = totalSamples;
  if (segStart >= segEnd || segStart >= totalSamples) return;
  const segLen = segEnd - segStart;
  const segL = new Float32Array(segLen);
  const segR = new Float32Array(segLen);

  let currentPos = sp0 / bf;
  // prev-tick vol/pan = the value at the END of the previous tick's render.
  // For the FIRST tick we initialise to the same as current so the noclick
  // ramp degenerates to a constant (no ramp); the segment-level fade-in
  // handles the actual silence-to-audio transition.
  let prevTickVol = ts[0][2], prevTickPan = ts[0][3];

  for (let i = 0; i < ts.length; i++) {{
    const st = ts[i];
    const tAbs = st[0], freq = st[1], vol = st[2], pan = st[3];
    let tNext, freqE, volE, panE;
    if (i + 1 < ts.length) {{
      tNext = ts[i+1][0]; freqE = ts[i+1][1]; volE = ts[i+1][2]; panE = ts[i+1][3];
    }} else {{
      tNext = et >= 0 ? et : tAbs + 1;
      freqE = freq; volE = vol; panE = pan;
    }}

    const outStart = Math.round(tAbs / TPS * SR);
    const outEnd   = Math.round(tNext / TPS * SR);
    const nOut     = Math.max(1, outEnd - outStart);
    if (outStart >= segEnd) break;
    const nAct = Math.min(nOut, segEnd - outStart);
    if (nAct <= 0) break;
    const relStart = outStart - segStart;   // index into segL/segR

    // Step size: CONSTANT for the whole tick (no LERP between tick freq values).
    // MikIT plays each tick at its discrete pitch — LERPing the freq across the
    // tick smears arpeggio (3-tick cycle) and discrete portamento jumps into
    // glides, perceived as "wrong vibrato speed". Volume/pan still LERP for
    // smooth dynamics, but pitch holds constant per tick like real trackers.
    const step = freq / (SR * bf);

    let pos = currentPos;
    for (let j = 0; j < nAct; j++) {{
      let p = pos;
      if (lt > 0 && le > ls && p >= le) {{
        let rel = (p - ls) % (2.0 * span);
        if (rel < 0) rel += 2.0 * span;
        p = (lt === 2) ? (rel < span ? ls + rel : le - (rel - span)) : ls + (rel % span);
      }} else if (lt === 0) {{
        if (p > smpLen - 0.001) p = smpLen - 0.001;
        if (p < 0) p = 0;
      }}
      if (p < 0) p = 0;
      if (p > extLen - 1.001) p = extLen - 1.001;
      const idx0 = p | 0;
      const frac = p - idx0;
      // Linear 2-tap interpolation. Cubic introduced HF overshoot at
      // 8-bit quantization steps (XM 16→8-bit truncation in XMFile) and
      // amplified those overshoots as crackling.
      const _i1 = (idx0 + 1 < extLen) ? idx0 + 1 : idx0;
      const pcmv = ext[idx0] * (1.0 - frac) + ext[_i1] * frac;
      // MikIT-style noclick: each tick plays at CONSTANT vol/pan, with a
      // 32-sample ramp at the START of each tick smoothing the jump from
      // the previous tick's value. Previously my LERP smeared each tick's
      // vol across the full tick duration, which (1) shifted the vol curve
      // by half a tick vs MikIT, and (2) produced slope corners at row
      // plateaus (D01/D02/D03 had a "no slide on tick 0" plateau every
      // 5 ticks → 11 Hz click pattern).
      let v, pa;
      if (j < NOCLICK_N) {{
        const rt = 0.5 * (1.0 - Math.cos(Math.PI * j / (NOCLICK_N - 1)));
        v  = prevTickVol + (vol - prevTickVol) * rt;
        pa = prevTickPan + (pan - prevTickPan) * rt;
      }} else {{
        v = vol; pa = pan;
      }}
      const ri = relStart + j;
      if (ri >= 0 && ri < segLen) {{
        // ProTracker/Amiga 75% stereo separation: hard-L = 87.5%L/12.5%R,
        // hard-R = 12.5%L/87.5%R.  pa=0=L, pa=0.5=C, pa=1=R.
        const _pr = 0.125 + 0.75 * pa;
        segL[ri] += pcmv * v * (1.0 - _pr);
        segR[ri] += pcmv * v * _pr;
      }}
      pos += step;
    }}

    // Save this tick's vol/pan as the "previous" for the next tick's noclick ramp.
    prevTickVol = vol;
    prevTickPan = pan;

    // Advance currentPos by exactly nAct * step (constant step within tick).
    currentPos += nAct * step;
    if (lt === 1 && le > ls && currentPos >= le) {{
      // Forward loop: fold into [ls, le)
      currentPos = ls + (currentPos - ls) % span;
    }} else if (lt === 2 && le > ls) {{
      // Bidi loop: keep currentPos MONOTONICALLY INCREASING in [ls, ls+2*span).
      // DO NOT fold to [ls, le) — that loses the direction phase (which half of
      // the bidi cycle we are in). The inner loop's modular math maps this
      // monotonic value to the correct reflected position for interpolation.
      // Modulo-reduce only to prevent float64 drift on very long notes.
      const twoSpan = 2.0 * span;
      if (currentPos < ls) currentPos = ls;
      else if (currentPos >= ls + twoSpan) currentPos = ls + (currentPos - ls) % twoSpan;
    }} else if (lt === 0 && currentPos > smpLen - 0.001) {{
      currentPos = smpLen - 0.001;
    }}
  }}

  // Micro-fade applied to THIS segment's contribution only. Hann window:
  // zero slope at both endpoints so no slope corners at fade entry/exit.
  // Linear fade had slope corners that clicked at the note-onset rate.
  const fn = Math.min(FADE, segLen >> 1);
  if (fn > 1) {{
    for (let j = 0; j < fn; j++) {{
      const f = 0.5 * (1.0 - Math.cos(Math.PI * j / (fn - 1)));
      segL[j] *= f;
      segR[j] *= f;
      const k = segLen - fn + j;
      segL[k] *= (1.0 - f);
      segR[k] *= (1.0 - f);
    }}
  }}

  // Mix into output buffers
  for (let j = 0; j < segLen; j++) {{
    bufL[segStart + j] += segL[j];
    bufR[segStart + j] += segR[j];
  }}
}}

// ── Batch renderer with progress ──────────────────────────────────────────────
let segIdx = 0;
const BATCH = 300;
let audioCtx = null, audioBuffer = null;
let sourceNode = null, playing = false;
let playOffset = 0, playStartTime = 0;
const totalDuration = totalSamples / SR;

function renderBatch() {{
  const end = Math.min(segIdx + BATCH, segments.length);
  for (let s = segIdx; s < end; s++) renderSeg(segments[s]);
  segIdx = end;
  const pct = Math.round(segIdx / Math.max(1, segments.length) * 100);
  document.getElementById('loadPct').textContent = pct;
  document.getElementById('loadFill').style.width = pct + '%';
  if (segIdx < segments.length) {{
    setTimeout(renderBatch, 0);
  }} else {{
    finishRender();
  }}
}}

let gainNode = null;
function finishRender() {{
  // Normalize
  let peak = 1e-9;
  for (let i = 0; i < totalSamples; i++) {{
    const al = Math.abs(bufL[i]), ar = Math.abs(bufR[i]);
    if (al > peak) peak = al;
    if (ar > peak) peak = ar;
  }}
  if (peak > 1.0) {{
    const g = 1.0 / peak;
    for (let i = 0; i < totalSamples; i++) {{ bufL[i] *= g; bufR[i] *= g; }}
  }}
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  gainNode = audioCtx.createGain();
  gainNode.gain.value = parseFloat(document.getElementById('volSlider').value) / 100;
  gainNode.connect(audioCtx.destination);
  audioBuffer = audioCtx.createBuffer(2, totalSamples, SR);
  audioBuffer.copyToChannel(bufL, 0);
  audioBuffer.copyToChannel(bufR, 1);
  document.getElementById('loadSection').style.display = 'none';
  document.getElementById('playerSection').style.display = 'flex';
  updateTimeDisplay();
  console.log('Render done. Duration:', (totalSamples/SR).toFixed(1)+'s, peak was:', peak.toFixed(4));
  // Autostart: kick off playback as soon as render finishes. Most browsers
  // allow this because the user already gestured to open the file/tab. If
  // the browser's autoplay policy blocks it, the audioCtx stays suspended
  // and the user just clicks Play.
  const _autoStart = () => {{
    try {{ startPlay(); setPPIcon(); }} catch (e) {{}}
  }};
  if (audioCtx.state === 'suspended') {{
    audioCtx.resume().then(_autoStart).catch(_autoStart);
  }} else {{
    _autoStart();
  }}
}}

function fmtTime(s) {{
  return String(Math.floor(s/60)).padStart(1,'0') + ':' + String(Math.floor(s%60)).padStart(2,'0');
}}
function currentTime() {{
  if (!playing) return playOffset;
  return Math.min(audioCtx.currentTime - playStartTime + playOffset, totalDuration);
}}

// ── Tracker / channel UI ──────────────────────────────────────────────────────
const NOTE_NAMES = ['C-','C#','D-','D#','E-','F-','F#','G-','G#','A-','A#','B-'];
function noteToStr(n) {{
  if (!n) return '---';
  if (n === 254) return '^^^';
  if (n === 255) return '===';
  return NOTE_NAMES[n % 12] + Math.floor(n / 12);
}}
function effChar(e) {{
  if (!e) return '.';
  // IT effect 1-25 → 'A'..'Y' (skip none = 0)
  return String.fromCharCode(64 + e);
}}
function hex2(v) {{ return ('00' + v.toString(16).toUpperCase()).slice(-2); }}

// Cell encoded as [note, inst, vol, eff, par] or 0 for empty
function fmtCell(c) {{
  if (!c || c === 0) return '<span class="trk-empty">--- .. .. ...</span>';
  const [n, i, vol, e, p] = c;
  const ns = noteToStr(n);
  const is = i ? hex2(i) : '..';
  let vs = '..';
  if (vol !== 255) {{
    if (vol <= 64) vs = hex2(vol);
    else vs = '?' + hex2(vol);
  }}
  const es = e ? effChar(e) + hex2(p) : '...';
  return `<span class="trk-note">${{ns}}</span> <span class="trk-samp">${{is}}</span> <span class="trk-samp">${{vs}}</span> <span class="trk-eff">${{es}}</span>`;
}}

function findRowAt(absTick) {{
  // Binary search rowEvents for the row currently active at absTick.
  // Returns the index in rowEvents (most recent row at-or-before absTick).
  if (!rowEvents.length) return -1;
  let lo = 0, hi = rowEvents.length - 1, ans = 0;
  while (lo <= hi) {{
    const mid = (lo + hi) >> 1;
    if (rowEvents[mid][0] <= absTick) {{ ans = mid; lo = mid + 1; }}
    else hi = mid - 1;
  }}
  return ans;
}}

// Build static tracker header once
(function buildTrackerHeader() {{
  let h = '<div class="trk-col-hdr">#</div>';
  for (let c = 0; c < NUM_CHANNELS; c++) h += `<div class="trk-col-hdr">CH${{c+1}}</div>`;
  document.getElementById('trkHeader').innerHTML = h;
}})();

const TRACKER_ROWS = 17;  // visible rows (current row + 8 above + 8 below)
let lastRowIdx = -1;

function updateTracker(absTick) {{
  const idx = findRowAt(absTick);
  if (idx < 0) return;
  const [, songPos, row, patNo] = rowEvents[idx];
  if (idx === lastRowIdx) return;   // no change
  lastRowIdx = idx;

  // Update info grid
  document.getElementById('iPos').textContent = songPos;
  const pat = (patNo >= 0 && patNo < patterns.length) ? patterns[patNo] : null;
  document.getElementById('iRow').textContent = row;
  if (pat) {{
    document.querySelector('#iRow + .sub') || null;
    // Update the row total in subtle
    const subEl = document.querySelector('.ic:nth-child(2) .sub');
    if (subEl) subEl.textContent = '/' + pat.length;
  }}

  // Render tracker body — show TRACKER_ROWS centered on current row
  const half = (TRACKER_ROWS - 1) >> 1;
  let html = '';
  if (pat) {{
    for (let off = -half; off <= half; off++) {{
      const r = row + off;
      const isCur = (off === 0);
      const cls = 'trk-row' + (isCur ? ' current' : '');
      if (r < 0 || r >= pat.length) {{
        html += `<div class="${{cls}}"><div class="trk-rownum">--</div>`;
        for (let c = 0; c < NUM_CHANNELS; c++) {{
          html += '<div class="trk-cell trk-empty">---</div>';
        }}
        html += '</div>';
        continue;
      }}
      const rowCells = pat[r];
      html += `<div class="${{cls}}"><div class="trk-rownum">${{r.toString(16).padStart(2,'0').toUpperCase()}}</div>`;
      for (let c = 0; c < NUM_CHANNELS; c++) {{
        const cell = (c < rowCells.length) ? rowCells[c] : 0;
        html += `<div class="trk-cell">${{fmtCell(cell)}}</div>`;
      }}
      html += '</div>';
    }}
  }}
  document.getElementById('trkBody').innerHTML = html;

  // Update channel panels — use the current row's cells for note/sample display
  if (pat && row >= 0 && row < pat.length) {{
    const rowCells = pat[row];
    for (let c = 0; c < NUM_CHANNELS; c++) {{
      const cell = (c < rowCells.length) ? rowCells[c] : 0;
      const noteEl = document.getElementById('chN' + c);
      if (cell && cell !== 0) {{
        const note = cell[0];
        noteEl.textContent = noteToStr(note);
        noteEl.classList.remove('dim');
        // Color hint by sample idx
        const inst = cell[1];
        if (inst) noteEl.style.color = `hsl(${{(inst * 47) % 360}},70%,65%)`;
      }}
    }}
  }}
}}

// Channel volume bars driven by active segments at current absTick
function updateChannelBars(absTick) {{
  // Build map: channel → max vol of currently-active segments
  const chVol = new Array(NUM_CHANNELS).fill(0);
  // Scan all segments — could be slow; for 2000 segs it's OK at 60fps
  for (let i = 0; i < segments.length; i++) {{
    const seg = segments[i];
    const ts = seg[3], et = seg[4];
    if (!ts.length) continue;
    const t0 = ts[0][0], t1 = (et >= 0) ? et : (ts[ts.length-1][0] + 1);
    if (t0 > absTick || t1 <= absTick) continue;
    // Find tick state at absTick
    let v = 0;
    for (let k = 0; k < ts.length; k++) {{
      if (ts[k][0] <= absTick) v = ts[k][2];
      else break;
    }}
    // We don't bake channel info per-seg here, so use segment.channel proxy via index
    // (rough: assume seg order ≈ channel order at trigger time — imperfect but visual)
    const ch = i % NUM_CHANNELS;
    if (v > chVol[ch]) chVol[ch] = v;
  }}
  for (let c = 0; c < NUM_CHANNELS; c++) {{
    const bar = document.getElementById('chB' + c);
    if (bar) bar.style.width = Math.min(100, chVol[c] * 100) + '%';
  }}
}}

function updateTimeDisplay() {{
  const c = currentTime();
  document.getElementById('timeDisp').textContent = fmtTime(c) + ' / ' + fmtTime(totalDuration);
  document.getElementById('seekFill').style.width = (c / totalDuration * 100) + '%';
  const absTick = Math.floor(c * TPS);
  updateTracker(absTick);
  if (playing) requestAnimationFrame(updateTimeDisplay);
}}

function startPlay(offset) {{
  if (!audioBuffer) return;
  if (sourceNode) {{ sourceNode.stop(); sourceNode.disconnect(); sourceNode = null; }}
  if (audioCtx.state === 'suspended') audioCtx.resume();
  sourceNode = audioCtx.createBufferSource();
  sourceNode.buffer = audioBuffer;
  sourceNode.connect(gainNode);
  playOffset    = (offset !== undefined) ? offset : playOffset;
  playStartTime = audioCtx.currentTime;
  sourceNode.start(0, playOffset);
  playing = true;
  sourceNode.onended = () => {{
    playing = false;
    setPPIcon();
    updateTimeDisplay();
  }};
  updateTimeDisplay();
}}
function doPause() {{
  if (!playing) return;
  playOffset += audioCtx.currentTime - playStartTime;
  if (sourceNode) {{ sourceNode.stop(); sourceNode.disconnect(); sourceNode = null; }}
  playing = false;
}}

function setPPIcon() {{
  const ico = document.getElementById('icoPP');
  const btn = document.getElementById('btnPlayPause');
  if (playing) {{
    ico.className = 'ico-pause';
    btn.title = 'Pause';
  }} else {{
    ico.className = 'ico-play';
    btn.title = 'Play';
  }}
}}
document.getElementById('btnPlayPause').onclick = () => {{
  if (!audioBuffer) return;
  playing ? doPause() : startPlay();
  setPPIcon();
}};
document.getElementById('btnStop').onclick = () => {{
  if (!audioBuffer) return;
  doPause(); playOffset = 0; updateTimeDisplay(); setPPIcon();
}};
document.getElementById('seekBar').onclick = (e) => {{
  if (!audioBuffer) return;
  const r = e.currentTarget.getBoundingClientRect();
  const f = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
  const off = f * totalDuration;
  if (playing) {{ doPause(); startPlay(off); }} else {{ playOffset = off; updateTimeDisplay(); }}
}};
document.getElementById('volSlider').addEventListener('input', (e) => {{
  if (gainNode) gainNode.gain.value = parseFloat(e.target.value) / 100;
}});

// ── Solo / mute: separate S and M buttons per channel ──────────────────────
// channelExplicitMuted[i] = user clicked M button on this channel
// channelSoloed[i]        = user clicked S button on this channel
// Final mute state = explicit mute OR (any solo && not soloed)
const channelExplicitMuted = new Array(NUM_CHANNELS).fill(false);
function recomputeMutedFromButtons() {{
  const anySolo = channelSoloed.some(s => s);
  for (let i = 0; i < NUM_CHANNELS; i++) {{
    channelMuted[i] = channelExplicitMuted[i] || (anySolo && !channelSoloed[i]);
  }}
}}
function refreshChannelPanelStyles() {{
  for (let c = 0; c < NUM_CHANNELS; c++) {{
    const p = document.getElementById('chP' + c);
    const s = document.getElementById('chS' + c);
    const m = document.getElementById('chM' + c);
    if (!p) continue;
    p.classList.toggle('muted', channelMuted[c]);
    if (s) s.classList.toggle('active', channelSoloed[c]);
    if (m) m.classList.toggle('active', channelExplicitMuted[c]);
  }}
}}
let rerenderPending = false, rerenderQueued = false;
function reRenderAudio() {{
  if (rerenderPending) {{ rerenderQueued = true; return; }}
  rerenderPending = true;
  const wasPlaying = playing;
  const resumeOffset = wasPlaying ? currentTime() : playOffset;
  if (wasPlaying) doPause();
  bufL.fill(0);
  bufR.fill(0);
  segIdx = 0;
  // Show subtle pill in topbar instead of swapping to full loading section
  const pill = document.getElementById('renderPill');
  pill.classList.add('show');
  document.getElementById('renderPct').textContent = '0';
  setTimeout(() => {{
    const renderBatchSolo = () => {{
      const end = Math.min(segIdx + BATCH, segments.length);
      for (let s = segIdx; s < end; s++) renderSeg(segments[s]);
      segIdx = end;
      const pct = Math.round(segIdx / Math.max(1, segments.length) * 100);
      document.getElementById('renderPct').textContent = pct;
      if (segIdx < segments.length) {{
        setTimeout(renderBatchSolo, 0);
      }} else {{
        let peak = 1e-9;
        for (let i = 0; i < totalSamples; i++) {{
          const al = Math.abs(bufL[i]), ar = Math.abs(bufR[i]);
          if (al > peak) peak = al; if (ar > peak) peak = ar;
        }}
        if (peak > 1.0) {{
          const g = 1.0 / peak;
          for (let i = 0; i < totalSamples; i++) {{ bufL[i] *= g; bufR[i] *= g; }}
        }}
        audioBuffer = audioCtx.createBuffer(2, totalSamples, SR);
        audioBuffer.copyToChannel(bufL, 0);
        audioBuffer.copyToChannel(bufR, 1);
        pill.classList.remove('show');
        rerenderPending = false;
        if (wasPlaying) startPlay(resumeOffset);
        else {{ playOffset = resumeOffset; updateTimeDisplay(); }}
        setPPIcon();
        if (rerenderQueued) {{ rerenderQueued = false; reRenderAudio(); }}
      }}
    }};
    renderBatchSolo();
  }}, 10);
}}
function onSoloClick(c) {{
  channelSoloed[c] = !channelSoloed[c];
  recomputeMutedFromButtons();
  refreshChannelPanelStyles();
  reRenderAudio();
}}
function onMuteClick(c) {{
  channelExplicitMuted[c] = !channelExplicitMuted[c];
  recomputeMutedFromButtons();
  refreshChannelPanelStyles();
  reRenderAudio();
}}
for (let c = 0; c < NUM_CHANNELS; c++) {{
  const s = document.getElementById('chS' + c);
  const m = document.getElementById('chM' + c);
  if (s) s.addEventListener('click', (e) => {{ e.stopPropagation(); onSoloClick(c); }});
  if (m) m.addEventListener('click', (e) => {{ e.stopPropagation(); onMuteClick(c); }});
}}

// Kick off rendering after the page has loaded
setTimeout(renderBatch, 50);
</script>
</body>
</html>"""

    with open(filename, 'w') as f:
        f.write(html)
    print(f"   📄 Segment player written → {filename}  ({os.path.getsize(filename)//1024} KB)")


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
                          compat=None, timeline_glsl=None):
    """Generate ShaderToy GLSL code with texture-based OR embedded data.
    viz: 0=None, 1=Reactive 001 (default), 2=Fluxline Surfer, 3=Zuvuya,
         4=Maya tunnel-warp, 5=Dodecahedron (Philip Bertani),
         6=Disco Combined (orblivius/finalman — smoke spotlights + lasers/clouds),
         7=Sparkly 4D (Philip Bertani — 4D IFS fractal raymarcher)
    compat: optional dict of compatibility overrides from --max-compat. Keys:
            no_surround, no_fat, reverb_2x2, fft_n, extra_pragmas. Missing
            keys default to permissive values (full-quality mode).
    timeline_glsl: pre-baked voice timeline string from mikit_engine (IT files).
                   When provided, Sound shader uses tlGetOutput instead of
                   getChannelOutput, skipping stateless pattern re-simulation."""

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
    _use_timeline = timeline_glsl is not None

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
            'repeat_point':   _compress_loop_offsets(smp['repeat_point'], smp['repeat_length'], bw_factor)[0],
            'repeat_length':  _compress_loop_offsets(smp['repeat_point'], smp['repeat_length'], bw_factor)[1],
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
            # they affect the duration of the row they appear in). Also scan
            # for EEx (pattern delay) — extends the row's duration by val
            # extra row-times. Bug fix: TINYTUNE.MOD pat 32 r49 has EE8 at
            # speed 15 → row should occupy (1+8)*15 = 135 ticks; the walker
            # was giving it 15 ticks, drifting the GLSL rowStartTick table
            # by ~280 ms across each EEx row.
            try:
                _row = mod.patterns[_pi][_ri]
            except Exception:
                _row = []
            _ee_val = 0
            for _ch in range(_num_ch):
                try:
                    _n = _row[_ch]
                    _eff = _n.get('effect', 0)
                    _par = _n.get('param', 0)
                    if _eff == 0xF and _par > 0:
                        if _par < 0x20: _cur_speed = _par
                        else:           _cur_tempo = _par
                    elif _eff == 0xE and ((_par >> 4) & 0xF) == 0xE:
                        _ee_val = max(_ee_val, _par & 0xF)
                except Exception:
                    pass
            _tps = _cur_tempo * 2.0 / 5.0
            _row_dur = (_cur_speed * (1 + _ee_val)) / _tps if _tps > 0 else 0.0
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
    # USE_142_DSP toggle — emit as 0/1 for the GLSL #define
    use_142_dsp_int = 1 if compat.get('use_142_dsp', False) else 0
    common_glsl = f"""/* ============================================================================
   GLSL (The Last) MOD Player v1.55 (c) 2026 Orblivius
   
   32 Tracks support, IT/XM/S3M/MOD loader, 3D Surround, PHAT Bass, Velvet Reverb, 
   Comb Reverb, FAT, W1 Limiter, RVQ sample compression, configurable downsample
   
   Visualizer: {viz_name}
 
   Git Home: https://github.com/mewza/mod2glsl
   Contact:  subband@gmail.com or
             subband@protonmail.com
  ============================================================================ */
// Generated from: {mod.title}
// {data_source_comment}

#define USE_EMBEDDED_DATA {1 if use_embedded else 0}
// 3-tap FIR low-pass on RVQ-decoded samples (1=ON, 0=OFF). Suppresses
// HF quantization noise from --no-rvq2 / --bitrate lo at 3× decode cost.
#define RVQ_LPF 1
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

// ── DSP MODE SWITCH (compile-time) ────────────────────────────────────────────
// USE_142_DSP = 0 → full v1.45 path: per-sample bass routing, 2-tap Only3D,
//                   PhatBass with 64-iter inst-walkback (best fidelity)
// USE_142_DSP = 1 → v1.42-style simpler path: surr_channels routing, 1-tap
//                   Only3D, single-row PhatBass inst lookup. Lower shader
//                   complexity → compiles on weaker drivers (ANGLE/Mali/older
//                   Adreno) that may otherwise crash on the v1.45 path.
//                   All correctness fixes (EAx/EBx, EEx, etc.) and the
//                   master softLimit are preserved either way.
#define USE_142_DSP {use_142_dsp_int}
// USE_TIMELINE_DSP=1: Sound shader reads pre-baked mikit_engine voice segments
// instead of re-simulating pattern effects via getChannelOutput. IT files only.
#define USE_TIMELINE_DSP {1 if _use_timeline else 0}

// ── Audio effects — toggle here in Common tab ─────────────────────────────────
// Each module is independently toggleable. Flip false to disable.
//   enable3D           : Only3D surround widening (surr_channels pair only)
//   enablePhatBass     : PhatBass Hilbert allpass cross-pan on bass instruments
//   enableFAT          : FAT4X harmonic exciter on master output (cs1 polynomial)
//   enableVelvetReverb : sparse-tap velvet-noise reverb (6 random-sign taps in
//                        an ~80 ms tail). Cheaper and smoother than the
//                        disabled Freeverb path — adds ~24 getChannelOutput
//                        calls/sample (4 channels × 6 taps). Default OFF.
// surr_channels: 1-indexed channel pair that gets Only3D (the other two = dry center)
//   ivec2(1,4) = outer LEFT pair (ch0,ch3) — default Amiga layout
//   ivec2(2,3) = inner RIGHT pair (ch1,ch2) — swap surround and center
const bool  enable3D           = {str(not _compat["no_surround"]).lower()};
const bool  enablePhatBass     = {str(not _compat["no_fat"]).lower()};
const bool  enableFAT          = {str(not _compat["no_fat"]).lower()};
const bool  enableVelvetReverb = false;   // opt-in; flip true to engage

// PhatBass routing — flip here to switch without re-running the encoder:
//   0 → per-sample: cross-pan ONLY isBass[]-tagged instruments (cleanest)
//   1 → mix-wide:   cross-pan the WHOLE mix (wider stereo, every voice
//                   gets the Hilbert character, mids may smear slightly)
// Defined in Common with #ifndef so a Sound-tab override is also possible
// (Sound's prelude only defines it if Common didn't).
#define PHATBASS_MIX_MODE __PHATBASS_MIX_MODE__
// Surround: AUTO-DETECT — applied to NON-bass channels (leads/pads get width, bass stays centered)

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
        # IT: use channel_pan (range 0..64, where 0=L, 32=center, 64=R).
        # IT files carry per-channel pan in the header; without this, the
        # GLSL falls through to S3M's channel_settings logic which IT files
        # also have but populate as mute flags (0/1) — making every IT
        # channel map to '0.0' (full LEFT). Fix: prefer channel_pan when
        # the file provides it (length matches num_channels and any value
        # exceeds 1, which distinguishes pan data from S3M mute flags).
        f'{(min(64, max(0, getattr(mod, "channel_pan", [32]*32)[i])) / 64.0):.3f}'
        if (getattr(mod, 'channel_pan', None) and
            len(mod.channel_pan) > i and
            max(mod.channel_pan[:mod.num_channels] or [0]) > 1)
        # S3M: use file-specified pan when available (channel_settings list
        # exists and contains valid entries: <8=L, 8..15=R, else fall back).
        else ('0.0' if (getattr(mod, 'channel_settings', None) and
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
    # XM key-off support — parallel arrays (kept separate from SampleInfo so
    # the VQ encoder's struct layout doesn't need to change).
    # sampleFadeout[i]    = XM volfade decrement per tick × 2 / 65536, 0 if no fade
    # sampleReleaseHold[i] = post-sustain env-vol average, 0..64 (64 = no drop)
    _fo_list = ', '.join(str(s.get('fadeout', 0)) for s in sample_map[:31])
    _rh_list = ', '.join(str(int(round(s.get('release_factor', 1.0) * 64))) for s in sample_map[:31])
    common_glsl += f"const int sampleFadeout[31]     = int[]({_fo_list});\n"
    common_glsl += f"const int sampleReleaseHold[31] = int[]({_rh_list});\n\n"
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
    // XM key-off cells (instrument byte ≥ 128, no period) are NOT triggers —
    // skip them in the trigger-search so the backward scan finds the real
    // note that's still being released.
    bool _trigIsKO = (trigNote.instrument >= 128) && (trigNote.period <= 0);
    if (_trigIsKO || trigNote.instrument <= 0 || trigNote.period <= 0) {{
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
            // XM key-off cells (instrument byte bit 7 set, period == 0)
            // are not triggers — keep scanning past them.
            bool _prevIsKO = (prev.instrument >= 128) && (prev.period <= 0);
            if (!_prevIsKO && (prev.instrument > 0 || prev.period > 0)) {{
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

    float s;
    if (smp.loopLen > 2) {{
        if (fSamplePos >= float(smp.loopStart + smp.loopLen))
            fSamplePos = float(smp.loopStart) + mod(fSamplePos - float(smp.loopStart), float(smp.loopLen));
        if (fSamplePos < 0.0) return 0.0;
        s = getSampleF(smp.start, fSamplePos, smp.length, smp.loopStart, smp.loopLen);

        // Loop-wrap anti-click: blend with the wave's "other side" near each
        // boundary. Bass samples often have mismatched loop_end and loop_start
        // amplitudes — without this, every loop wrap clicks. 8-sample window
        // each side (in compressed sample-domain — the discontinuity lives
        // there). Skip very short loops (<32 samples) where this would dull
        // single-cycle synth tones.
        if (smp.loopLen >= 32) {{
            const float _LF = 8.0;
            float _posInLoop = fSamplePos - float(smp.loopStart);
            float _distFromStart = _posInLoop;
            float _distFromEnd   = float(smp.loopLen) - _posInLoop;
            if (_distFromStart < _LF) {{
                float _partnerPos = float(smp.loopStart + smp.loopLen) - (_LF - _distFromStart);
                float _partner = getSampleF(smp.start, _partnerPos, smp.length, smp.loopStart, smp.loopLen);
                float _fade = 0.5 * (1.0 - _distFromStart / _LF);
                s = mix(s, _partner, _fade);
            }} else if (_distFromEnd < _LF) {{
                float _partnerPos = float(smp.loopStart) + (_LF - _distFromEnd);
                float _partner = getSampleF(smp.start, _partnerPos, smp.length, smp.loopStart, smp.loopLen);
                float _fade = 0.5 * (1.0 - _distFromEnd / _LF);
                s = mix(s, _partner, _fade);
            }}
        }}
    }} else if (fSamplePos >= float(smp.length)) {{
        // Anti-click post-end tail: hold the sample's last value and fade
        // it linearly over 64 OUTPUT samples (~1.45 ms @ 44.1k) — NOT 64
        // compressed-sample-domain units, which would be wildly wrong for
        // samples with bw_factor>1 played at non-c5 pitches. Convert via
        // freq and bw_factor: time-past-end = (fSamplePos - length) * bwFactor / freq
        // (in seconds), then × 44100 for output samples.
        float _postEndOut = (fSamplePos - float(smp.length)) * float(smp.bwFactor) * 44100.0 / freq;
        if (_postEndOut >= 64.0) return 0.0;
        s = getSample(smp.start + smp.length - 1) * (1.0 - _postEndOut / 64.0);
    }} else {{
        if (fSamplePos < 0.0) return 0.0;
        s = getSampleF(smp.start, fSamplePos, smp.length, smp.loopStart, smp.loopLen);
    }}

    // ── Volume: forward scan trigger→current to honour Cxx cuts & Axx slides ─
    // ProTracker volume slide (Effect A/6) SKIPS tick 0 → applies (SPEED-1) ticks per row.
    int volume = smp.volume;
    // ── XM key-off detection ────────────────────────────────────────
    // Stateless approximation: scan rows trigRow+1..pos.row for a key-off
    // marker (instrument byte has bit 7 set, period == 0). If found, record
    // its time so we can apply env-release + fadeout for `currentTime -
    // keyOffTime` seconds at the end. JS engine has true per-voice envelope
    // state; this is a "snap to release-hold then fade" approximation that
    // matches openMPT closely enough for the typical staccato-with-tail use.
    float keyOffTime = -1.0;
    // Trigger-row effect
    if (trigNote.effect == 0xC) {{
        volume = min(trigNote.param, 64);
    }} else if (trigNote.effect == 0xA || trigNote.effect == 0x6) {{
        int _su=(trigNote.param>>4)&0xF, _sd=trigNote.param&0xF;
        // (speed-1) ticks because tick 0 is skipped — use trigger row's
        // actual speed from the per-row table, not the initial SPEED constant
        volume = clamp(volume + (_su>0?_su:-_sd)*(trigSpeed-1), 0, 64);
    }} else if (trigNote.effect == 0xE) {{
        // EAx fine vol slide up / EBx fine vol slide down — TICK 0 ONLY (one-shot)
        int _esub = (trigNote.param >> 4) & 0xF;
        int _eval =  trigNote.param        & 0xF;
        if      (_esub == 0xA) volume = min(64, volume + _eval);
        else if (_esub == 0xB) volume = max(0,  volume - _eval);
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
            // XM key-off cell: instrument byte ≥ 128 (bit 7) and no period.
            // Don't break — just record the time and keep scanning so a real
            // following note can still terminate the trigger window.
            bool _isKO = (_fn.instrument >= 128) && (_fn.period <= 0);
            if (_isKO) {{
                if (keyOffTime < 0.0) {{
                    int _gRow = patRowOffset[_fp] + (_fr - patStartRow[_fp]);
                    keyOffTime = rowStartTime[_gRow];
                }}
                _fr++;
                if (_fr >= 64) {{ _fr=0; _fp++; }}
                continue;
            }}
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
            else if (_fn.effect == 0xE) {{
                // EAx / EBx — fine vol slide (TICK 0 only, one-shot per row)
                int _esub = (_fn.param >> 4) & 0xF;
                int _eval =  _fn.param        & 0xF;
                if      (_esub == 0xA) volume = min(64, volume + _eval);
                else if (_esub == 0xB) volume = max(0,  volume - _eval);
            }}
            _fr++;
            if (_fr >= 64) {{ _fr=0; _fp++; }}
        }}
        // Current row: Cxx fully, Axx for elapsed ticks (tick 0 skipped)
        Note _cr = getNote(pos.songPos, pos.row, ch);
        bool _crIsKO = (_cr.instrument >= 128) && (_cr.period <= 0);
        if (_crIsKO && keyOffTime < 0.0) {{
            // Key-off on the current row — record its start time.
            keyOffTime = rowStartTime[curGlobalRow];
        }} else if (_cr.instrument<=0 && _cr.period<=0) {{
            if (_cr.effect==0xC)
                volume = min(_cr.param, 64);
            else if (_cr.effect==0xA || _cr.effect==0x6) {{
                int _su=(_cr.param>>4)&0xF, _sd=_cr.param&0xF;
                int _ticks = max(0, int(pos.tick) - 1);  // tick 0 skipped
                volume = clamp(volume+(_su>0?_su:-_sd)*_ticks, 0, 64);
            }}
            else if (_cr.effect == 0xE) {{
                // EAx / EBx — fine vol slide (TICK 0 only, applied as soon as
                // pos enters this row; one-shot, no per-tick accumulation)
                int _esub = (_cr.param >> 4) & 0xF;
                int _eval =  _cr.param        & 0xF;
                if      (_esub == 0xA) volume = min(64, volume + _eval);
                else if (_esub == 0xB) volume = max(0,  volume - _eval);
            }}
        }}
    }}

    // ── Apply XM key-off envelope release + fadeout multiplier ─────────
    // The first ~40ms after key-off snaps from full vol down to releaseHold
    // (matching the rapid post-sustain env drop in openMPT). Then fadeout
    // linearly decays the multiplier toward 0 over 65536/(2*fadeout) ticks.
    // For samples without an envelope (fadeout=0): hard cut on key-off,
    // matching MOD/non-XM behaviour where key-off has no fade.
    float releaseMul = 1.0;
    if (keyOffTime > 0.0) {{
        float relTime = currentTime - keyOffTime;
        if (relTime > 0.0) {{
            int _fo = sampleFadeout[trigNote.instrument - 1];
            if (_fo <= 0) {{
                // Anti-click: ramp releaseMul 1→0 over 64 output samples
                // (~1.45 ms @ 44.1k). Used to be 256 — too long.
                float _kSamp = relTime * 44100.0;
                releaseMul = max(0.0, 1.0 - _kSamp / 64.0);
            }} else {{
                // Stage 1: snap to releaseHold over 40ms (env release shape).
                float drop = clamp(relTime * 25.0, 0.0, 1.0);
                float hold = float(sampleReleaseHold[trigNote.instrument - 1]) / 64.0;
                float envMul = mix(1.0, hold, drop);
                // Stage 2: fadeout — volfade -= 2*fadeout per tick.
                float ticksKO = relTime * float(pos.speed) / pos.rowTime;
                float fadeMul = max(0.0, 1.0 - 2.0 * float(_fo) * ticksKO / 65536.0);
                releaseMul = envMul * fadeMul;
            }}
        }}
    }}
    // ── mikIT-style 64-sample crossfade with previous trigger ─────────────
    //
    // mdrv_mix.cpp `dying[]` mechanism: when a new note kicks on channel t,
    // mikIT copies the OLD channel state to dying[t], sets dying[t].volume=0
    // and ramps it down over 64 samples while the new note ramps up from 0.
    // True parallel crossfade — output is continuous across the trigger.
    //
    // Stateless port: scan back to find the trigger BEFORE current, replay
    // its FULL effect chain (volume slides, Cxx, EAx/EBx, key-off envelope)
    // to compute the prev note's frozen volume at the moment of the new
    // trigger, look up its sample value at currentTime, and blend:
    //
    //   weight = clamp(elapsed * 44100 / 64, 0, 1)
    //   at trigger T (elapsed=0):    weight=0 → 100% prev (continuous w/ T-1)
    //   at T+64:                     weight=1 → 100% new
    //
    // Earlier mikIT attempt failed because prev used `_pSmp.volume` (static
    // default vol), not the actual playing volume. If old note was at vol 32
    // and prev-output was computed at vol 64, the boundary jumped 2x = click.
    // This version fixes that by running the full vol/release chain for prev.
    float _newOutput = s * float(volume) / 64.0 * releaseMul;
    float _xfWeight  = clamp(elapsed * 44100.0 / 64.0, 0.0, 1.0);
    float _prevOutput = 0.0;

    if (_xfWeight < 1.0) {{
        // ── Step 1: backward scan from trigRow-1 to find prev trigger ──
        int _pScanRow = trigRow - 1;
        int _pScanPat = trigPat;
        Note _pNote;
        bool _pFound = false;
        for (int _plb = 0; _plb < 64; _plb++) {{
            if (_pScanRow < 0) {{
                if (_pScanPat > 0) {{
                    _pScanPat--;
                    int _prCount = patRowOffset[_pScanPat+1] - patRowOffset[_pScanPat];
                    _pScanRow = patStartRow[_pScanPat] + _prCount - 1;
                }} else {{ break; }}
            }}
            _pNote = getNote(_pScanPat, _pScanRow, ch);
            bool _pIsKO = (_pNote.instrument >= 128) && (_pNote.period <= 0);
            if (!_pIsKO && (_pNote.instrument > 0 || _pNote.period > 0)) {{
                _pFound = true;
                break;
            }}
            _pScanRow--;
        }}
        if (_pFound && _pNote.instrument > 0 && _pNote.instrument <= 31 && _pNote.period > 0) {{
            SampleInfo _pSmp = samples[_pNote.instrument - 1];
            if (_pSmp.length > 0) {{
                int   _pTrigGlobalRow = patRowOffset[_pScanPat] + (_pScanRow - patStartRow[_pScanPat]);
                int   _pTrigSpeed     = rowSpeed[_pTrigGlobalRow];
                float _pTrigTime      = rowStartTime[_pTrigGlobalRow];
                float _pElapsed       = currentTime - _pTrigTime;
                if (_pElapsed > 0.0) {{
                    // ── Step 2: prev note's full volume effect chain ───
                    // Identical structure to the current note's chain but
                    // scoped to (_pScanPat,_pScanRow) → (trigPat,trigRow-1).
                    int _pVolume = _pSmp.volume;
                    float _pKeyOffTime = -1.0;
                    // Prev trigger row effects
                    if (_pNote.effect == 0xC) {{
                        _pVolume = min(_pNote.param, 64);
                    }} else if (_pNote.effect == 0xA || _pNote.effect == 0x6) {{
                        int _psu=(_pNote.param>>4)&0xF, _psd=_pNote.param&0xF;
                        _pVolume = clamp(_pVolume + (_psu>0?_psu:-_psd)*(_pTrigSpeed-1), 0, 64);
                    }} else if (_pNote.effect == 0xE) {{
                        int _pesub = (_pNote.param >> 4) & 0xF;
                        int _peval =  _pNote.param        & 0xF;
                        if      (_pesub == 0xA) _pVolume = min(64, _pVolume + _peval);
                        else if (_pesub == 0xB) _pVolume = max(0,  _pVolume - _peval);
                    }}
                    // Forward scan from _pScanRow+1 to (trigPat,trigRow-1)
                    {{
                        int _pfp = _pScanPat, _pfr = _pScanRow + 1;
                        int _pPosRows = (_pfp < SONG_LENGTH) ? patRowOffset[_pfp+1] - patRowOffset[_pfp] : 0;
                        if (_pfr >= patStartRow[_pfp] + _pPosRows) {{
                            _pfp++;
                            if (_pfp < SONG_LENGTH) {{
                                _pfr = patStartRow[_pfp];
                                _pPosRows = patRowOffset[_pfp+1] - patRowOffset[_pfp];
                            }}
                        }}
                        for (int _pfi = 0; _pfi < 64; _pfi++) {{
                            // Stop when we reach trigRow on trigPat (exclusive — new note replaces old at start of trigRow)
                            if (_pfp > trigPat || (_pfp == trigPat && _pfr >= trigRow)) break;
                            if (_pfp >= SONG_LENGTH) break;
                            if (_pfr >= patStartRow[_pfp] + _pPosRows) {{
                                _pfp++;
                                _pfr = (_pfp < SONG_LENGTH) ? patStartRow[_pfp] : 0;
                                if (_pfp < SONG_LENGTH) _pPosRows = patRowOffset[_pfp+1] - patRowOffset[_pfp];
                                continue;
                            }}
                            Note _pfn = getNote(_pfp, _pfr, ch);
                            bool _pfIsKO = (_pfn.instrument >= 128) && (_pfn.period <= 0);
                            if (_pfIsKO) {{
                                if (_pKeyOffTime < 0.0) {{
                                    int _pgRow = patRowOffset[_pfp] + (_pfr - patStartRow[_pfp]);
                                    _pKeyOffTime = rowStartTime[_pgRow];
                                }}
                                _pfr++;
                                continue;
                            }}
                            if (_pfn.instrument > 0 || _pfn.period > 0) break;  // shouldn't happen (we stopped at trigRow), but safety
                            if (_pfn.effect == 0xC) {{
                                _pVolume = min(_pfn.param, 64);
                            }} else if (_pfn.effect == 0xA || _pfn.effect == 0x6) {{
                                int _psu=(_pfn.param>>4)&0xF, _psd=_pfn.param&0xF;
                                int _pfGRow = patRowOffset[_pfp] + (_pfr - patStartRow[_pfp]);
                                int _pfSpd = rowSpeed[_pfGRow];
                                _pVolume = clamp(_pVolume + (_psu>0?_psu:-_psd)*(_pfSpd-1), 0, 64);
                            }} else if (_pfn.effect == 0xE) {{
                                int _pesub = (_pfn.param >> 4) & 0xF;
                                int _peval =  _pfn.param        & 0xF;
                                if      (_pesub == 0xA) _pVolume = min(64, _pVolume + _peval);
                                else if (_pesub == 0xB) _pVolume = max(0,  _pVolume - _peval);
                            }}
                            _pfr++;
                        }}
                    }}
                    // ── Step 3: prev note's release / fadeout multiplier ──
                    float _pReleaseMul = 1.0;
                    if (_pKeyOffTime > 0.0) {{
                        float _pRelTime = currentTime - _pKeyOffTime;
                        if (_pRelTime > 0.0) {{
                            int _pFo = sampleFadeout[_pNote.instrument - 1];
                            if (_pFo <= 0) {{
                                float _pKSamp = _pRelTime * 44100.0;
                                _pReleaseMul = max(0.0, 1.0 - _pKSamp / 64.0);
                            }} else {{
                                float _pDrop  = clamp(_pRelTime * 25.0, 0.0, 1.0);
                                float _pHold  = float(sampleReleaseHold[_pNote.instrument - 1]) / 64.0;
                                float _pEnvM  = mix(1.0, _pHold, _pDrop);
                                float _pTksKO = _pRelTime * float(pos.speed) / pos.rowTime;
                                float _pFadeM = max(0.0, 1.0 - 2.0 * float(_pFo) * _pTksKO / 65536.0);
                                _pReleaseMul = _pEnvM * _pFadeM;
                            }}
                        }}
                    }}
                    // ── Step 4: prev note's sample value at currentTime ──
                    // (vibrato ignored — modulation over 1.45ms is sub-cent)
                    float _pFreq       = periodToFreq(max(1, _pNote.period));
                    float _pFSamplePos = _pElapsed * _pFreq / float(_pSmp.bwFactor);
                    float _pS = 0.0;
                    if (_pSmp.loopLen > 2) {{
                        if (_pFSamplePos >= float(_pSmp.loopStart + _pSmp.loopLen))
                            _pFSamplePos = float(_pSmp.loopStart) + mod(_pFSamplePos - float(_pSmp.loopStart), float(_pSmp.loopLen));
                        if (_pFSamplePos >= 0.0)
                            _pS = getSampleF(_pSmp.start, _pFSamplePos, _pSmp.length, _pSmp.loopStart, _pSmp.loopLen);
                    }} else if (_pFSamplePos < float(_pSmp.length) && _pFSamplePos >= 0.0) {{
                        _pS = getSampleF(_pSmp.start, _pFSamplePos, _pSmp.length, _pSmp.loopStart, _pSmp.loopLen);
                    }} else if (_pFSamplePos >= float(_pSmp.length)) {{
                        // Post-end tail (64 output samples)
                        float _pPostOut = (_pFSamplePos - float(_pSmp.length)) * float(_pSmp.bwFactor) * 44100.0 / _pFreq;
                        if (_pPostOut < 64.0)
                            _pS = getSample(_pSmp.start + _pSmp.length - 1) * (1.0 - _pPostOut / 64.0);
                    }}
                    _prevOutput = _pS * float(_pVolume) / 64.0 * _pReleaseMul;
                }}
            }}
        }}
    }}

    return mix(_prevOutput, _newOutput, _xfWeight);
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
    # IT/XM: NNA-retrigger routing makes per-sample bass detection unreliable.
    # Silently promote the default 'sample' to 'mix' for these formats so
    # PhatBass always runs mix-wide unless the user explicitly forces 'sample'.
    if _pb_mode == 'sample' and (getattr(mod, 'is_it', False) or getattr(mod, 'is_xm', False)):
        _pb_mode = 'mix'
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

    # Common's PhatBass mix-mode #define was emitted with a deferred token
    # because phatbass_mix_mode is computed AFTER bass detection (which
    # happens after common_glsl was built). Substitute the real value now.
    common_glsl = common_glsl.replace('__PHATBASS_MIX_MODE__', str(phatbass_mix_mode))

    # ── Only3D allpass: precompute coefficients at generation time ─────────
    # The per-sample code used to compute tan/sin/cos every audio sample
    # (~441,000× per song-second). Since freq + SR are constants, the
    # coefficients are too — compute them here and emit literals.
    #
    # For each freq, the closed-form derivation is:
    #     d        = tan(freq * pi / SR)
    #     p0       = sin(d) / (cos(d) + sin(d))
    #     (p1+p2)  = cos(d) / (cos(d) + sin(d))   [since p1=p0 and p2 = ... - p1]
    #     delay    = 0.5 / freq
    # Emit AP_P0, AP_P1_PLUS_P2, AP_DELAY as const vec2 in the GLSL.
    _ONLY3D_FREQ1 = 700.0    # tweaked from original 500 — wider stereo image
    _ONLY3D_FREQ2 = 2500.0
    _ONLY3D_DEPTH = 0.2      # shuffle gain (was 0.12)
    _ONLY3D_SAT   = 1.0      # soft saturation strength (was 0.8)
    _ONLY3D_SR    = 44100.0
    def _only3d_coeffs(freq, sr=_ONLY3D_SR):
        d        = math.tan(freq * math.pi / sr)
        sin_d    = math.sin(d)
        cos_d    = math.cos(d)
        denom    = cos_d + sin_d
        p0       = sin_d / denom
        p1_p2    = cos_d / denom        # p1 + p2 simplifies to this
        delay    = 0.5 / freq
        return p0, p1_p2, delay
    _ap_p0_1, _ap_p1p2_1, _ap_delay_1 = _only3d_coeffs(_ONLY3D_FREQ1)
    _ap_p0_2, _ap_p1p2_2, _ap_delay_2 = _only3d_coeffs(_ONLY3D_FREQ2)

    # ── Timeline injection for IT files (mikit_engine pre-baked voice segments) ──
    # If timeline_glsl is provided, inject the TL const arrays + tlGetOutput into
    # the Sound shader right after getSampleF (which tlGetOutput calls).
    # The plain-string concatenation keeps GLSL braces from being misread as
    # Python f-string markers.
    _TLGETOUTPUT_GLSL = (
        "\n// ── Timeline output: sum active voice segments at time T ─────────────────\n"
        "// Replaces the channel loop for IT files.  Each VoiceSegment stores start freq,\n"
        "// vol, pan, sample position and per-tick derivatives; this function integrates\n"
        "// them in continuous time so the loop runs over segments (TL_NUM_SEGS), not\n"
        "// channels×rows.  Loop count is bounded by a non-const local so ANGLE/D3D11\n"
        "// does not unroll it (runtime loop, data-dependent early-continue).\n"
        "#if USE_TIMELINE_DSP\n"
        "vec2 tlGetOutput(float T) {\n"
        "    vec2 out_lr = vec2(0.0);\n"
        "    int tick_T = int(T * TL_TICKS_PER_SEC);\n"
        "    int _nseg = TL_NUM_SEGS;  // non-const var prevents ANGLE unrolling\n"
        "    for (int i = 0; i < _nseg; i++) {\n"
        "        if (tick_T < tlSegStart[i] || tick_T >= tlSegEnd[i])\n"
        "            continue;\n"
        "        float seg_t0 = float(tlSegStart[i]) / TL_TICKS_PER_SEC;\n"
        "        float dt = T - seg_t0;\n"
        "        if (dt < 0.0) continue;\n"
        "        float freq = tlSegFreq[i] * pow(tlSegFreqMul[i], dt * TL_TICKS_PER_SEC);\n"
        "        float vol  = clamp(tlSegVol[i] + tlSegVolDelta[i] * dt * TL_TICKS_PER_SEC,\n"
        "                           0.0, 1.0);\n"
        "        float fpos = float(tlSegPos[i]) + dt * freq;\n"
        "        int ls = tlSegLoopSt[i], le = tlSegLoopEn[i];\n"
        "        if (tlSegLoopTy[i] > 0 && le > ls && fpos >= float(le)) {\n"
        "            float span = float(le - ls);\n"
        "            fpos = float(ls) + mod(fpos - float(ls), span);\n"
        "        }\n"
        "        SampleInfo smp = samples[clamp(tlSegSample[i], 0, 30)];\n"
        "        float s = getSampleF(smp.start, fpos, smp.length, ls,\n"
        "                             le > ls ? le - ls : 0);\n"
        "        float panR = 0.25 + 0.5 * tlSegPan[i];\n"
        "        out_lr += s * vol * vec2(1.0 - panR, panR);\n"
        "    }\n"
        "    return out_lr;\n"
        "}\n"
        "#endif // USE_TIMELINE_DSP\n"
    )
    if _use_timeline:
        _tl_injection = timeline_glsl + _TLGETOUTPUT_GLSL
    else:
        _tl_injection = ""

    sound_glsl = f"""/* ============================================================================
   GLSL (The Last) MOD Player v1.55 (c) 2026 Orblivius
   
   32 Tracks support, IT/XM/S3M/MOD loader, 3D Surround, PHAT Bass, Velvet Reverb, 
   Comb Reverb, FAT, W1 Limiter, RVQ sample compression, configurable downsample
   
   Visualizer: {viz_name}
 
   Git Home: https://github.com/mewza/mod2glsl
   Contact:  subband@gmail.com or
             subband@protonmail.com
  ============================================================================ */
// USE_TIMELINE_DSP defined here as well as Common — the VQ-encoded Common
// generator doesn't preserve this define, and Sound's #if blocks below need
// it to compile. Safe to define in both places (same value).
#ifndef USE_TIMELINE_DSP
#define USE_TIMELINE_DSP {1 if _use_timeline else 0}
#endif
// getByte / getPatternByte / getSample / getSampleF / getNote / getChannelOutput are in Common.
// tlGetOutput (if USE_TIMELINE_DSP) is injected below after getSampleF is prepended by VQ post-proc.
{_tl_injection}
// Bass sample flags (true = instrument detected as bass) — for PhatBass
const bool isBass[31] = bool[]({bass_flags_str});

// PhatBass routing — set in Common (#ifndef guard so manual edits there win):
//   PHATBASS_MIX_MODE 0 → per-sample (only isBass[]-tagged instruments get
//                          the Hilbert cross-pan; cleanest, leaves leads alone)
//   PHATBASS_MIX_MODE 1 → mix-wide (applies the cross-pan to the entire mix —
//                          wider stereo + bass enhancement on everything,
//                          can lightly smear mid/high transients)
// Default is whatever --phatbass-mode the encoder ran with. Edit the line
// in Common to override at compile time on Shadertoy without re-running.
#ifndef PHATBASS_MIX_MODE
#define PHATBASS_MIX_MODE {phatbass_mix_mode}
#endif

// ── FAT4X harmonic exciter helper ────────────────────────────────────────
// cs1 polynomial waveshaper (even harmonics only) from FAT4X by Orblivius.
// Even-power series → zero at x=0, adds warm even harmonics, soft clip near ±1.
float fat_cs1(float x) {{
    float x2=x*x, x4=x2*x2, x6=x4*x2, x8=x4*x4, x10=x4*x6, x12=x6*x6;
    return 0.4375 - 0.3228759765625*x2 + 0.1123046875*x4
         - 0.50537109375*x6 + 0.1993408203125*x8
         + 0.634521484375*x10 - 0.6513671875*x12;
}}
// vec2 overload: same polynomial applied per-channel in one call. GLSL
// supports overloading by signature, so this coexists with the float version.
vec2 fat_cs1(vec2 x) {{
    vec2 x2=x*x, x4=x2*x2, x6=x4*x2, x8=x4*x4, x10=x4*x6, x12=x6*x6;
    return vec2(0.4375) - 0.3228759765625*x2 + 0.1123046875*x4
         - 0.50537109375*x6 + 0.1993408203125*x8
         + 0.634521484375*x10 - 0.6513671875*x12;
}}

// ── Master limiter (rational soft-knee, stateless) ────────────────────────
// Stateless equivalent of the JS AdaptiveLimiter — Shadertoy's mainSound is
// per-sample with no persistent state, so we can't run a true envelope/
// attack/release loop here. Instead we use a soft-knee curve tuned to the
// same ceiling (0.995 ≈ −0.04 dB) as AdaptiveLimiter::maxLimit, with the
// knee starting at 0.95 so signal under that level is bit-identical
// pass-through.
//
// Math:
//   over     = max(|x| − T, 0)
//   reduced  = HEAD · over / (over + HEAD)
//   y        = sign(x) · (min(|x|, T) + reduced)
// where T = 0.95 (knee) and HEAD = ceiling − T = 0.045. As over→∞ the
// reduced term asymptotes to HEAD, so |y|→ceiling=0.995. At over=0 the
// function is exactly |x|, untouched. No exp/tanh — one mul, one div,
// abs/min/max each. Catches PhatBass + FAT4X overshoot before the audio
// context hard-clips, without compressing musical dynamics below ±0.95.
vec2 softLimit(vec2 x) {{
    // Fully robust soft-clipper. Always converges to ±CEIL regardless of T.
    //   • T < CEIL → soft-knee from T to CEIL (musical, no clicks)
    //   • T ≥ CEIL → hard ceiling at CEIL (no headroom for a knee, but no
    //     NaN/Inf either; peaks just clamp to ±CEIL)
    // Guarantees: output is always in [−CEIL, +CEIL]. No divide-by-zero
    // even if T is set to 1.0 or above. Catches PhatBass / FAT4X overshoot
    // before WebAudio hard-clips. NaN in → NaN out is impossible because
    // every branch only does mul/div on bounded magnitudes.
    // CEIL=1.0: fat_cs1 polynomial is only well-behaved on |x|<=1.0. Past 1.0
    // it goes non-monotonic (at x=1.1, fat_cs1≈-1.71, flipping FAT4X's sign
    // and crushing a 1.1 peak to ~0.63 — audible as crunchy distortion on
    // bright mixes). Knee 0.85→1.0 keeps PhatBass/reverb transients clean.
    const float T    = 0.85;    // knee start (signals below this are bit-perfect)
    const float CEIL = 1.0;     // ceiling — keep fat_cs1 strictly in [-1,1]
    vec2 ax = abs(x);
    if (T >= CEIL) {{
        // No soft-knee headroom available — hard-clip at CEIL.
        return sign(x) * min(ax, vec2(CEIL));
    }}
    const float HEAD = CEIL - T;                      // > 0 here
    vec2 over    = max(ax - T, vec2(0.0));
    vec2 reduced = (HEAD * over) / (over + HEAD);     // 0 ≤ reduced < HEAD
    return sign(x) * (min(ax, vec2(T)) + reduced);
}}

// Helper: Get mixed mono output at a given time (for reverb optimization)
// Mixes all channels once instead of re-mixing N_COMB×N_ITER times
// Mono input + pan coefficients = stereo output (standard reverb design)
float getMixedMono(float time_offset, Position pos, float rowTime) {{
    float mix = 0.0;
#if USE_TIMELINE_DSP
    vec2 _tm = tlGetOutput(time_offset);
    mix = (_tm.x + _tm.y) * 0.5;
#else
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
        mix += getChannelOutput(ch, time_offset, pos, rowTime);
    }}
#endif
    const float normFactor = (NUM_CHANNELS <= 4) ? (2.0 / float(NUM_CHANNELS)) : 0.85;
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
    // OPT: vec2 (.x = L, .y = R) — modern GPUs vectorize 2-lane FP ops at
    // the same cost as scalar, and the source is half as noisy as the
    // separate L/R floats it replaces.
    // OPT: const-qualified — NUM_CHANNELS is a #define.
    // Loud mode: 4-ch MOD keeps the original 2/N; multi-ch formats get
    // a flat 0.45 (single voice at vol=64 reaches 45% of full scale).
    // softLimit / FAT4X handle multi-voice peaks downstream.
    const float normFactor = (NUM_CHANNELS <= 4) ? (2.0 / float(NUM_CHANNELS)) : 0.85;
#if USE_TIMELINE_DSP
    // IT pre-baked timeline path: tlGetOutput sums all active voice segments.
    // Pan is already baked per-segment; no surr/cent split needed.
    vec2 surr = tlGetOutput(playbackTime) * normFactor;
    vec2 cent = vec2(0.0);
#else
    vec2 surr = vec2(0.0);
    vec2 cent = vec2(0.0);

    for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
        float s = getChannelOutput(ch, playbackTime, pos, rowTime);
        // pan: .x = panL (0.75..0.25), .y = panR (0.25..0.75)
        float panR = 0.25 + 0.5 * channelPan[ch];
        vec2  pan  = vec2(1.0 - panR, panR);

        vec2 panned = s * pan;
#if USE_142_DSP
        // Fixed surr_channels routing (v1.42 path). No per-channel getNote.
        // surr_channels.x/.y are 1-indexed channels in the outer "surround"
        // pair (ch0+ch3 for 4-chan); the inner pair feeds the dry centre.
        int ch1 = (ch % 4) + 1;
        bool isSurr = (ch1 == surr_channels.x || ch1 == surr_channels.y);
        if (isSurr) surr += panned;
        else        cent += panned;
#else
        // Per-note bass detection (v1.45 path): non-bass channels get
        // surround widening, bass-tagged samples stay centered.
        Note n = getNote(pos.songPos, pos.row, ch);
        int  inst = n.instrument;
        bool bassFlag = (inst >= 1 && inst <= 31) ? isBass[inst - 1] : false;
        if (bassFlag) cent += panned;
        else          surr += panned;
#endif
    }}
    surr *= normFactor;
    cent *= normFactor;
#endif // USE_TIMELINE_DSP
    
    // ── Only3D — proper allpass technique with precomputed coefficients ──────
    // Direct port from Only3D.h by Dmitry Boldyrev / mss
    // Two 1st-order allpass filters at different frequencies create
    // phase-shifted copies of the stereo difference, which are then
    // cross-mixed to widen the stereo image.
    //
    // OPT (compile-time): all coefficients are pure functions of freq + SR,
    // so they're computed in Python (mod_player.py) and emitted as literal
    // const vec2. Saves ~10 trig ops + a divide per audio sample
    // (~441,000 ops/sec eliminated) vs the old per-sample tan/sin/cos.
    //
    // OPT (vec2): the two parallel allpass channels share identical math,
    // so they're computed in vec2 form (one vector op instead of two
    // scalars at every step).
    const float ONLY3D_DEPTH = {_ONLY3D_DEPTH};
    const float SATURATION   = {_ONLY3D_SAT};

    // Precomputed for freq=({_ONLY3D_FREQ1:.0f}Hz, {_ONLY3D_FREQ2:.0f}Hz) @ SR={_ONLY3D_SR:.0f}Hz
    //   d        = tan(freq*PI/SR)
    //   p0       = sin(d) / (cos(d) + sin(d))
    //   (p1+p2)  = cos(d) / (cos(d) + sin(d))   [p1 = p0, p2 = denom - p1, sum = denom]
    //   delay    = 0.5 / freq
    const vec2 AP_P0         = vec2({_ap_p0_1:.6f}, {_ap_p0_2:.6f});
    const vec2 AP_P1_PLUS_P2 = vec2({_ap_p1p2_1:.6f}, {_ap_p1p2_2:.6f});
    const vec2 AP_DELAY      = vec2({_ap_delay_1:.7f}, {_ap_delay_2:.7f});

    if (enable3D) {{
#if USE_TIMELINE_DSP
        // ── Only3D, timeline path: call tlGetOutput at the allpass delay offsets ──
        // The stereo difference between delayed and current output forms the
        // allpass input. No per-channel loop needed.
        vec2 t3d = vec2(playbackTime) - AP_DELAY;
        if (t3d.x >= 0.0 && t3d.y >= 0.0) {{
            float diffNow = surr.x - surr.y;
            vec2 out1 = tlGetOutput(t3d.x) * normFactor;
            vec2 out2 = tlGetOutput(t3d.y) * normFactor;
            vec2 diffDelayed = vec2(out1.x - out1.y, out2.x - out2.y);
            vec2 ap = diffNow * AP_P0 + diffDelayed * AP_P1_PLUS_P2;
            vec2 dd = ap * inversesqrt(1.0 + ap * ap * SATURATION);
            float shuffle = (dd.x - dd.y) * ONLY3D_DEPTH;
            surr += vec2(shuffle, -shuffle);
        }}
#elif USE_142_DSP
        // ── Only3D, single-tap (v1.42 path) — lower shader complexity ──
        float tW = playbackTime - AP_DELAY.x;
        if (tW >= 0.0) {{
            Position posW = getPosition(tW);
            float wL = 0.0, wR = 0.0;
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
                int ch1 = (ch % 4) + 1;
                if (ch1 == surr_channels.x || ch1 == surr_channels.y) {{
                    int  _inst142   = getNote(posW.songPos, posW.row, ch).instrument;
                    bool _isBass142 = (_inst142 >= 1 && _inst142 <= 31) ? isBass[_inst142 - 1] : false;
                    if (_isBass142) continue;
                    float sw   = getChannelOutput(ch, tW, posW, rowTime);
                    float panR = 0.25 + 0.5 * channelPan[ch];
                    wL += sw * (1.0 - panR);
                    wR += sw * panR;
                }}
            }}
            wL *= normFactor;
            wR *= normFactor;
            float diff = (wL - wR) * ONLY3D_DEPTH;
            surr += vec2(diff, -diff);
        }}
#else
        // ── Only3D, 2-tap parallel allpass (v1.45 path) — wider, smoother ──
        vec2 t = vec2(playbackTime) - AP_DELAY;
        if (t.x >= 0.0 && t.y >= 0.0) {{
            Position pos1 = getPosition(t.x);
            Position pos2 = getPosition(t.y);
            float diffNow = surr.x - surr.y;
            vec2 wL = vec2(0.0);
            vec2 wR = vec2(0.0);
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
                int  inst1   = getNote(pos1.songPos, pos1.row, ch).instrument;
                bool isBass1 = (inst1 >= 1 && inst1 <= 31) ? isBass[inst1 - 1] : false;
                if (isBass1) continue;
                float panR = 0.25 + 0.5 * channelPan[ch];
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
            // Soft saturation: x / sqrt(1 + x²·SAT). inversesqrt() is one HW op.
            vec2 dd = ap * inversesqrt(1.0 + ap * ap * SATURATION);
            float shuffle = (dd.x - dd.y) * ONLY3D_DEPTH;
            surr += vec2(shuffle, -shuffle);
        }}
#endif
    }}
    
    // ── PhatBass — low-shelf bass boost + Haas-delayed cross-pan widening ───
    // Replaces the previous truncated-Hilbert (h1, h3 coefficients with 80-
    // sample tap spacing) which had heavy comb-aliasing (bandpass peak at
    // ~137 Hz + alias lobes above) and sounded gritty on bright transients.
    //
    // New design — stateless per-sample, cheap, clean:
    //   1. 2-tap boxcar LPF on the bass signal (taps at t and t-0.5ms).
    //      First null at ~1 kHz, gentle rolloff — passes bass, kills mids.
    //   2. SAME-pan: add LPF'd signal at SHELF_DEPTH → low-shelf boost.
    //   3. OPPOSITE-pan: add 8 ms Haas-delayed dry signal at HAAS_DEPTH →
    //      classic stereo widening without comb-aliasing or phase issues.
    //
    // Two modes (selected at encode time via PHATBASS_MIX_MODE):
    //  0 = Per-sample: only channels playing bass-tagged instruments.
    //  1 = Mix-wide: applied to the full mix (auto when no bass detected).
    const float PHAT_SHELF_T     = 0.0005;  // 0.5 ms (22 sample) LPF spacing
    const float PHAT_HAAS_T      = 0.008;   // 8 ms Haas widening delay
    const float PHAT_SHELF_DEPTH = 0.7;     // same-pan low-shelf gain
    const float PHAT_HAAS_DEPTH  = 0.4;     // opposite-pan Haas gain
    vec2 _phatPB = vec2(0.0);
    if (enablePhatBass) {{
        float tA = playbackTime - PHAT_SHELF_T;
        float tH = playbackTime - PHAT_HAAS_T;
        // Guard: skip until both taps are within rendered audio (Haas tap
        // is the longest, so checking it covers tA too).
        if (tH < 0.0) {{
            // Skip PhatBass; Haas tap not yet available.
        }} else {{
#if USE_TIMELINE_DSP
        // Mix-wide via timeline taps. LPF same-pan + Haas opposite-pan (.yx).
        vec2 dryNow = tlGetOutput(playbackTime) * normFactor;
        vec2 dryA   = tlGetOutput(tA)           * normFactor;
        vec2 dryH   = tlGetOutput(tH)           * normFactor;
        vec2 lp     = 0.5 * (dryNow + dryA);
        _phatPB     = lp * PHAT_SHELF_DEPTH + dryH.yx * PHAT_HAAS_DEPTH;
#else
        Position posA = getPosition(tA);
        Position posH = getPosition(tH);
        vec2 pb = vec2(0.0);
#if PHATBASS_MIX_MODE
        // Mix-wide: sum all channels at the three tap times.
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
            float panR = 0.25 + 0.5 * channelPan[ch];
            vec2  pan  = vec2(1.0 - panR, panR);
            float s0 = getChannelOutput(ch, playbackTime, pos,  rowTime);
            float sA = getChannelOutput(ch, tA, posA, rowTime);
            float sH = getChannelOutput(ch, tH, posH, rowTime);
            float lp = 0.5 * (s0 + sA);
            pb += lp * PHAT_SHELF_DEPTH * pan + sH * PHAT_HAAS_DEPTH * pan.yx;
        }}
#else
        // Per-sample: only bass-detected instruments.
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {{
#if USE_142_DSP
            int inst2 = 0;
            {{
                int _sR = pos.row, _sP = pos.songPos;
                for (int _lb = 0; _lb < 8; _lb++) {{
                    Note _n2 = getNote(_sP, _sR, ch);
                    if (_n2.instrument > 0 && _n2.instrument <= 31) {{ inst2 = _n2.instrument; break; }}
                    _sR--;
                    if (_sR < 0) {{
                        if (_sP > 0) {{
                            _sP--;
                            _sR = patStartRow[_sP] + (patRowOffset[_sP+1] - patRowOffset[_sP]) - 1;
                        }} else {{ break; }}
                    }}
                }}
            }}
            bool bass  = (inst2 >= 1 && inst2 <= 31) ? isBass[inst2 - 1] : false;
#else
            int inst2 = 0;
            int sR = pos.row, sP = pos.songPos;
            for (int lb = 0; lb < 64; lb++) {{
                Note n2 = getNote(sP, sR, ch);
                if (n2.instrument > 0) {{ inst2 = n2.instrument; break; }}
                sR--;
                if (sR < 0) {{
                    if (sP > 0) {{
                        sP--;
                        sR = patStartRow[sP] + (patRowOffset[sP+1] - patRowOffset[sP]) - 1;
                    }} else {{ break; }}
                }}
            }}
            bool bass = (inst2 >= 1 && inst2 <= 31) ? isBass[inst2 - 1] : false;
#endif
            if (bass) {{
                float panR = 0.25 + 0.5 * channelPan[ch];
                vec2  pan  = vec2(1.0 - panR, panR);
                float s0 = getChannelOutput(ch, playbackTime, pos,  rowTime);
                float sA = getChannelOutput(ch, tA, posA, rowTime);
                float sH = getChannelOutput(ch, tH, posH, rowTime);
                float lp = 0.5 * (s0 + sA);
                pb += lp * PHAT_SHELF_DEPTH * pan + sH * PHAT_HAAS_DEPTH * pan.yx;
            }}
        }}
#endif
        _phatPB = pb * normFactor;
#endif // USE_TIMELINE_DSP
        }}  // end else (tH >= 0)
    }}
    
    vec2 _out = surr + cent;

    // ── PhatBass — add cross-panned bass signal (chain: 3D → PhatBass → ...) ─
    _out += _phatPB;

    // ── Velvet-noise reverb (sparse-tap convolution) ─────────────────────
    // 6 random-sign taps spread across an ~80 ms tail with exponential
    // amplitude decay. Polarity is flipped between L/R to widen the wet
    // image without using any allpass tricks. Math: out += sum(±decay_i ·
    // dry(t − tap_i)). Per output sample this costs N_TAPS getPosition +
    // N_TAPS getChannelOutput-equivalent (via getMixedMono), which is ~half
    // the disabled Freeverb's per-sample work and avoids the comb-filter
    // peaks that color a comb-network's tail. Toggle in Common via
    // enableVelvetReverb (default off — opt-in).
    if (enableVelvetReverb) {{
        const int   _VELV_N    = 6;
        // Taps placed by hand to be roughly logarithmic in time, with no
        // pair too close together (avoids audible flutter). All within
        // the 180 s buffer so getPosition() never wraps to song-end.
        const float _VELV_T[6] = float[6](0.011, 0.019, 0.027, 0.038, 0.052, 0.071);
        const float _VELV_S[6] = float[6](+1.0, -1.0, +1.0, +1.0, -1.0, +1.0);
        const float _VELV_RT60 = 0.060;   // ~60 ms decay constant
        const float _VELV_WET  = 0.18;
        vec2 _wet = vec2(0.0);
        for (int _vi = 0; _vi < _VELV_N; _vi++) {{
            float _vtt = playbackTime - _VELV_T[_vi];
            if (_vtt < 0.0) continue;
#if USE_TIMELINE_DSP
            float _vdry = dot(tlGetOutput(_vtt), vec2(0.5, 0.5)) * normFactor;
#else
            Position _vp = getPosition(_vtt);
            float _vdry = getMixedMono(_vtt, _vp, rowTime);
#endif
            float _vamp = exp(-_VELV_T[_vi] / _VELV_RT60) * _VELV_S[_vi];
            _wet.x += _vdry *  _vamp;
            _wet.y -= _vdry *  _vamp;     // L/R polarity flip → stereo width
        }}
        _out += _wet * _VELV_WET;
    }}

    // Soft-limit to 1.1 before FAT4X: fat_cs1 polynomial diverges past |x|=1.1.
    // Soft knee from 0.95→1.1 lets PhatBass/reverb transients breathe without
    // hard-clipping; the polynomial then colors the result, not the limiter.
    _out = softLimit(_out);

    // ── FAT4X harmonic exciter (stateless) ─────────────────────────────────
    // Applied BEFORE reverb (reverb currently disabled — see block below).
    // cs1 produces even harmonics → adds warmth/presence, soft-limits peaks.
    // FAT_AMOUNT: 0.0=off  0.5=half  1.0=full FAT4X-equivalent  >1.0=heavy
    // Uses the vec2 overload of fat_cs1 so both channels go through one
    // polynomial call instead of two separate scalar invocations.
    if (enableFAT) {{
        // FAT_AMOUNT was 1.5 but combined with the end-of-chain ×1.3 crank
        // and the 1.1-ceiling soft-limit it produced ~6dB sustained
        // clipping on bass-heavy 12-channel IT mixes (peak=1.0 the entire
        // song). Dialed back to 0.5 for IT-style content; chip leads still
        // get presence without stacking gain past the ceiling.
        const float FAT_AMOUNT = 0.5;
        _out = _out * (1.0 + 0.5 * fat_cs1(_out) * FAT_AMOUNT);
    }}

    // ── End-of-chain soft-limit at 1.0 (clean ceiling) ─────────────────
    // Previously: ×1.3 crank + 1.1 ceiling → "intentional 0.1 overdrive".
    // For 12-channel IT mixes with PhatBass + FAT4X stacked above, that
    // crank pushed the recorded output to peak=1.0 every second of the
    // song (sustained DAC clipping audible as harsh distortion). Dialed
    // back to a clean unity-ceiling soft-knee — softLimit at the start
    // of this chain already catches per-voice peaks, this is a final
    // safety net that asymptotes cleanly to ±1.0.
    if (NUM_CHANNELS > 4) {{
        vec2 _ax = abs(_out);
        float _T = 0.85;
        float _C = 1.0;
        float _H = _C - _T;
        vec2 _over    = max(_ax - _T, vec2(0.0));
        vec2 _reduced = (_H * _over) / (_over + _H);
        _out = sign(_out) * (min(_ax, vec2(_T)) + _reduced);
    }}

    // ── Freeverb-inspired parallel comb reverb — DISABLED ──────────────────
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
    //
    // DISABLED by default — the dry mix sits well without it, and the comb
    // loop is the hottest path in the shader. To re-enable, remove the /*
    // and */ markers below AND insert before the block:
    //     float outL = _out.x; float outR = _out.y;
    // and after the block:
    //     _out = vec2(outL, outR);
    // (The body still uses outL/outR scalars — keeping it that way means
    // re-enable is a 4-line edit, not a rewrite.)
    /*
    {(
        '''const int   N_ITER  = 2;   // (--max-compat: was 3)
    const float RT60    = 1.6;  // shorter tail (was 2.4) - less smear on hihats
    const float _decay  = 8.9078 / RT60;
    const float _D[2]  = float[](0.0253, 0.0338);   // shortest + longest only
    const float _pL[2] = float[](0.85, 0.45);
    const float _pR[2] = float[](0.40, 0.80);
    const int   N_COMB  = 2;
    const float COMB_DIV = 2.0;'''
        if _compat["reverb_2x2"] else
        '''const int   N_ITER  = 3;   // iterations per comb (was 5)
    const float RT60    = 1.6;  // shorter tail (was 2.4) - less smear on hihats
    const float _decay  = 8.9078 / RT60;

    // Freeverb-inspired comb delays (seconds), mutually prime in samples.
    // Kept original indices {0,1,4,5} → shortest pair + longest pair.
    const float _D[4]  = float[](0.0253, 0.0269, 0.0322, 0.0338);
    const float _pL[4] = float[](0.85, 0.40, 0.80, 0.45);
    const float _pR[4] = float[](0.40, 0.85, 0.45, 0.80);
    const int   N_COMB  = 4;
    const float COMB_DIV = 4.0;'''
    )}
    const float RV_WET  = 0.08;  // less wet (was 0.15) - cleaner transients

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
            // OPTIMIZED: Use pre-mixed mono (4× faster!)
            // Before: N_COMB×N_ITER×4 channels = 64 getChannelOutput calls
            // After:  N_COMB×N_ITER = 16 getMixedMono calls
            // Mono input + different pan coefficients = stereo output (standard reverb)
            float _m = getMixedMono(_tw, _rp, rowTime) * _gk;
            _wet.x += _pL[_c] * _m;
            _wet.y += _pR[_c] * _m;
        }}
    }}
    _wet /= COMB_DIV;       // RMS match
    outL += _wet.x * RV_WET;
    outR += _wet.y * RV_WET;
    */

    // ── Buffer-end fade-out ─────────────────────────────────────────────
    // Shadertoy's audio buffer is ~180 seconds (precomputed at compile time,
    // not streamed). When `time` reaches that limit, audio just stops — if
    // the last sample is mid-waveform at high amplitude, the cut is audible
    // as a "halt" or click. Fade the last 0.4s smoothly to zero so the
    // ending sounds intentional rather than truncated. 0.4s ≈ 17640 samples
    // at 44.1kHz — long enough to be smooth, short enough that user only
    // loses a fraction of a row at the very end.
    // Buffer-end fade DISABLED — user removed it. The 180 s buffer cap
    // still exists (Shadertoy renders Sound once into a fixed buffer), but
    // the trailing 0.4 s cosine fade-out is no longer applied; audio just
    // stops abruptly at the cap. Re-enable by uncommenting the block below.
    // const float BUFFER_CAP   = 180.0;
    // const float FADE_LEN     = 0.4;
    // float _fadeT = clamp((BUFFER_CAP - time) / FADE_LEN, 0.0, 1.0);
    // float _bufFade = 0.5 - 0.5 * cos(_fadeT * 3.14159265);
    // _out *= _bufFade;

    // IT master volume (global_volume × mix_volume / 128²). Defined in
    // Common — defaults to 1.0 for MOD/XM, < 1.0 for IT files that ship
    // with a mix attenuation. Without this, IT files play 2-3× too hot
    // and slam every downstream limiter into sustained clipping.
    _out *= MASTER_GAIN;

    return _out;
}}
"""
    
    
    # ========== IMAGE TAB ==========
    raw_title   = mod.title.strip() or "UNTITLED"
    title_text  = raw_title[:20]
    title_chars = to_glsl_font_chars(title_text)
    title_len   = len(title_text)
    # Format suffix rendered SEPARATELY in WHITE right after the title
    # (in YELLOW). main() wraps XM/IT into a duck-typed namespace with
    # `is_xm` / `is_it` flags; check those first, then is_s3m, else MOD.
    if getattr(mod, 'is_xm', False):
        fmt_text = " (XM)"
    elif getattr(mod, 'is_it', False):
        fmt_text = " (IT)"
    elif getattr(mod, 'is_s3m', False):
        fmt_text = " (S3M)"
    else:
        fmt_text = " (MOD)"
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
        # Image tab can't call getChannelOutput (it's emitted only into Sound
        # after the Common→Sound split). Instead, read 4 distinct positions
        # from Buffer A's row-0 mix waveform — gives 4 audio-reactive amps
        # (not per-channel, but visually equivalent for the curtain effect).
        viz_setup_block = (
            "    vec2 _uv=(C*2.-iResolution.xy)/iResolution.y;\n"
            "    float _scrollX=texelFetch(iChannel1,ivec2(5,2),0).r;\n"
            "    float _va0=abs(texelFetch(iChannel1,ivec2( 64,0),0).r);\n"
            "    float _va1=abs(texelFetch(iChannel1,ivec2(192,0),0).r);\n"
            "    float _va2=abs(texelFetch(iChannel1,ivec2(320,0),0).r);\n"
            "    float _va3=abs(texelFetch(iChannel1,ivec2(448,0),0).r);\n"
            "    vec3 col = _visCurtain(vec2(_uv.y, abs(_uv.x)), _va0, _va1, _va2, _va3) + _visBG(C);"
        )
    else:  # 1, 2, 4, 5, 6, 7, 8 all use _VizScene
        viz_setup_block = (
            "    vec2 _uv=(C*2.-iResolution.xy)/iResolution.y;\n"
            "    float _scrollX=texelFetch(iChannel1,ivec2(5,2),0).r;\n"  # Read scroll offset from Buffer A
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

    else:  # viz == 8
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


    image_glsl = f"""/* ============================================================================
   GLSL (The Last) MOD Player v1.55 (c) 2026 Orblivius
   
   32 Tracks support, IT/XM/S3M/MOD loader, 3D Surround, PHAT Bass, Velvet Reverb, 
   Comb Reverb, FAT, W1 Limiter, RVQ sample compression, configurable downsample
   
   Visualizer: {viz_name}
   iChannel0: alphabet texture (shadertoy.com/view/4sf3RB)
   iChannel1: Buffer A (audio + FFT + smoothed bands)
   iChannel2: RGBA Noise Small  ← required for viz 6 smoke turbulence
 
   Git Home: https://github.com/mewza/mod2glsl
   Contact:  subband@gmail.com or
             subband@protonmail.com
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
makeStr(printHdr)   _NUM _NUM _NUM _ _G _L _S _L _ _M _O _D _ _P _L _A _Y _E _R _ _V _1 _DOT _5 _5 _ _NUM _NUM _NUM _end
makeStr(printCredit) _COPY _2 _0 _2 _6 _ _O _R _B _L _I _V _I _U _S _end
makeStr(printLoad)   _L _O _A _D _I _N _G _DOT _DOT _DOT _end
makeStr(printSpec)   _S _P _E _C _T _R _U _M _end
makeStr(printOsci)   _O _S _C _I _L _L _O _S _C _O _P _E _end
makeStr(printNoSnd)  _S _E _T _ _I _C _H _A _N _1 _ _EQ _ _S _O _U _N _D _ _O _U _T _P _U _T _end
makeStr(printPatt)  _P _A _T _T _E _R _N _COL _end
makeStr(printRow)   _R _O _W _COL _end
makeStr(printTracks) _T _R _A _C _K _S _COL _end
makeStr(printBPM)   _B _P _M _COL _end
makeStr(printSpd)   _S _P _E _E _D _COL _end
makeStr(printTrk1)  _T _R _A _C _K _ _NUM _1 _end
makeStr(printTrk2)  _T _R _A _C _K _ _NUM _2 _end
makeStr(printTrk3)  _T _R _A _C _K _ _NUM _3 _end
makeStr(printTrk4)  _T _R _A _C _K _ _NUM _4 _end
makeStr(printTrack) _T _R _A _C _K _ _NUM _end

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
    float rx  = iResolution.x*0.42;

    trk += BLUE   * printPatt(pUV(fp, ML, iy, CH));
    trk += WHITE  * drawNum(pos.songPos, 2, ML+10.*CW, iy, CW,CH,fp);
    trk += BLUE   * drawCh(47,fp, ML+11.*CW, iy, CW,CH);
    trk += YELLOW * drawNum({mod.num_patterns-1}, 2, ML+13.*CW, iy, CW,CH,fp);

    trk += BLUE   * printRow(pUV(fp, rx, iy, CH));
    trk += WHITE  * drawNum(pos.row, 2, rx+5.*CW, iy, CW,CH,fp);
    trk += BLUE   * drawCh(47,fp, rx+6.*CW, iy, CW,CH);
    trk += YELLOW * drawCh(54,fp, rx+7.*CW, iy, CW,CH);
    trk += YELLOW * drawCh(52,fp, rx+8.*CW, iy, CW,CH);
    
    // TRACKS: N — show total channel count to the right of ROW
    float tx_lbl = rx + 12.*CW;
    trk += BLUE   * printTracks(pUV(fp, tx_lbl, iy, CH));
    trk += YELLOW * drawNum(NUM_CHANNELS, 2, tx_lbl + 8.*CW, iy, CW, CH, fp);

    trk += BLUE   * printBPM(pUV(fp, ML,  iy2, CH));
    trk += YELLOW * printBPMVal(pUV(fp, ML+5.*CW, iy2, CH));
    trk += BLUE   * printSpd(pUV(fp, rx,  iy2, CH));
    trk += YELLOW * printSpdVal(pUV(fp, rx+7.*CW, iy2, CH));

    trk += BLUE * 0.55 * hline(fp, iy2+CH+4., 1., iResolution.x-1.);

    // ============ TRACKER ============
    float ty   = iy2+CH+10.;
    float TW   = 9.*CW+6.;    // 9 chars per cell + 6px gap
    float rNW  = 2.*CW;
    float txOff= ML+rNW+8. + _scrollX;  // ← scroll offset applied to ALL tracks!
    const int HVR = 4;  // 9 visible rows → more room for oscilloscope

    // Track headers - "TRACK#N" CENTERED in each column
    vec3 trackColors[4]; trackColors[0]=TC0; trackColors[1]=TC1; trackColors[2]=TC2; trackColors[3]=TC3;
    for(int tc=0; tc<NUM_CHANNELS; tc++) {{
        vec3 tCol = trackColors[tc % 4];
        float tx = txOff + float(tc)*TW;
        // "TRACK #" then digit(s) — both digits AFTER #, no overlap
        int digits = (tc+1 >= 10) ? 2 : 1;
        float numCW = CW * 0.65;  // tight digit spacing
        // # is at column 6. First digit at column ~6.6 (right after #).
        // For 1-digit: rightmost (only) digit at +6.6*CW
        // For 2-digit: rightmost at +6.6 + numCW, leftmost at +6.6 (right after #)
        float firstDigitX = 6.6 * CW;  // position after the #
        float rightmostX = firstDigitX + float(digits-1) * numCW;
        float textW = rightmostX + numCW;  // total approximate width
        float xCenter = tx + (TW - textW) * 0.5;
        trk += tCol * printTrack(pUV(fp, xCenter, ty, CH));
        trk += tCol * drawNum(tc+1, digits, xCenter + rightmostX, ty, numCW, CH, fp);
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
        trk += BLUE*0.55*vline(fp, txOff+float(tc)*TW-4., ty, ty+float(2*HVR+2)*CH+7.);

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

        // BLACK MASK: hide track content that scrolled under row numbers
        if(fp.x < ML+rNW+8.) {{
            col = vec3(0.0);  // black out left column area
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
                else        nc = (ri==0 ? TCols[tc%4] : TCols[tc%4]*fade);
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
    // MASK: zero out trk in row number column area (left of tracks)
    // ONLY within tracker Y range (don't break BPM/PATTERN labels above!)
    if(fp.x < ML+rNW+8. && fp.y >= ty+CH+1. && fp.y < tBot+5.) {{
        // Save the row number contribution before zeroing
        vec3 rowNumPart = vec3(0.0);
        if(fp.y>=tTop && fp.y<tBot) {{
            int ri_abs2=int((fp.y-tTop)/CH);
            int rn2 = pageStart + ri_abs2;
            if(rn2<0)   rn2+=64;
            if(rn2>=64) rn2-=64;
            bool on4_2 = (rn2%4)==0;
            int frameRow2 = pos.row - pageStart;
            vec3 rnc2 = ri_abs2==frameRow2 ? WHITE : (on4_2 ? YELLOW*0.7 : TC3*0.7);
            rowNumPart = rnc2 * drawNum(rn2, 2, ML+rNW-CW*1.2, tTop+float(ri_abs2)*CH, CW*0.75,CH,fp);
        }}
        trk = rowNumPart;  // Replace trk with only row numbers in this area
        col = vec3(0.0);   // Black background
    }}
    
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
                    barColor = TCols[ch%4];
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
   GLSL (The Last) MOD Player v1.55 (c) 2026 Orblivius
   
   32 Tracks support, IT/XM/S3M/MOD loader, 3D Surround, PHAT Bass, Velvet Reverb, 
   Comb Reverb, FAT, W1 Limiter, RVQ sample compression, configurable downsample
   
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
 
   Git Home: https://github.com/mewza/mod2glsl
   Contact:  subband@gmail.com or
             subband@protonmail.com
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
            float att = clamp(age / 0.300, 0.0, 1.0); float env = att * att * (3.0 - 2.0 * att);
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
            // Spectrum toggle: only in BOTTOM 30% (canvas/visualization area)
            float clickY = iMouse.y / iResolution.y;
            bool inCanvasArea = clickY < 0.3;
            float currMouse = (iMouse.z > 0.0 && inCanvasArea) ? 1.0 : 0.0;
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
        }} else if (px == 5) {{
            // ── Relative grabbing scroll with out-of-bounds release ────────
            float scrollOffset = texelFetch(iChannel0, ivec2(5, 2), 0).r;
            float scrollAnchor = texelFetch(iChannel0, ivec2(6, 2), 0).r;
            float prevPressed  = texelFetch(iChannel0, ivec2(7, 2), 0).r;
            float currPressed  = iMouse.z > 0.0 ? 1.0 : 0.0;
            
            // Check Y bounds - tracker area is approximately Y 0.18-0.82
            // (above oscilloscope at bottom, below header at top)
            float mouseY = iMouse.y / iResolution.y;
            bool inBounds = mouseY > 0.18 && mouseY < 0.82;
            
            // MAX_SCROLL based on actual screen width and track count
            const float TW_PX = 231.0;
            float visibleWidth = iResolution.x - 68.0;
            float totalWidth = float(NUM_CHANNELS) * TW_PX;
            float hiddenWidth = max(0.0, totalWidth - visibleWidth);
            float MAX_SCROLL = ceil(hiddenWidth / TW_PX) * TW_PX;
            
            // LOGIC:
            // - Pressed + in bounds → drag normally (1:1 mouse)
            // - Pressed + out of bounds → TREAT AS RELEASE: snap based on gap
            //   • Gap on left → flush left (track 0)
            //   • Gap on right → flush right (track N)
            //   • No gap → stay put
            // - Released → same as out-of-bounds: snap based on gap
            // Read drag-dead flag (set when mouse leaves view, cleared on click)
            float dragDead = texelFetch(iChannel0, ivec2(8, 2), 0).r;
            
            // New click → reset everything
            if (currPressed > 0.5 && prevPressed < 0.5) {{
                dragDead = 0.0;
                scrollAnchor = scrollOffset;
            }}
            
            // Out of bounds while pressed → KILL DRAG
            if (currPressed > 0.5 && !inBounds) {{
                dragDead = 1.0;
            }}
            
            // Only update scroll if drag is alive AND in bounds
            if (currPressed > 0.5 && inBounds && dragDead < 0.5) {{
                scrollOffset = scrollAnchor + (iMouse.x - abs(iMouse.z));
            }}
            
            // ALWAYS clamp/snap (no over-scroll allowed)
            scrollOffset = clamp(scrollOffset, -MAX_SCROLL, 0.0);
            // Released or out of bounds: leaves where it was
            O = vec4(scrollOffset, 0., 0., 1.);
        }} else if (px == 6) {{
            // Scroll anchor - just save on click
            float scrollAnchor = texelFetch(iChannel0, ivec2(6, 2), 0).r;
            float scrollOffset = texelFetch(iChannel0, ivec2(5, 2), 0).r;
            float prevPressed  = texelFetch(iChannel0, ivec2(7, 2), 0).r;
            float currPressed  = iMouse.z > 0.0 ? 1.0 : 0.0;
            if (currPressed > 0.5 && prevPressed < 0.5) {{
                scrollAnchor = scrollOffset;
            }}
            O = vec4(scrollAnchor, 0., 0., 1.);
        }} else if (px == 7) {{
            // ── Previous mouse pressed state ───
            O = vec4(iMouse.z > 0.0 ? 1.0 : 0.0, 0., 0., 1.);
        }} else if (px == 8) {{
            // ── Drag-dead flag ───
            float dragDead = texelFetch(iChannel0, ivec2(8, 2), 0).r;
            float prevPressed = texelFetch(iChannel0, ivec2(7, 2), 0).r;
            float currPressed = iMouse.z > 0.0 ? 1.0 : 0.0;
            float mouseY = iMouse.y / iResolution.y;
            bool inBounds = mouseY > 0.18 && mouseY < 0.82;
            
            // New click clears dead
            if (currPressed > 0.5 && prevPressed < 0.5) dragDead = 0.0;
            // Out of bounds while pressed → kill
            if (currPressed > 0.5 && !inBounds) dragDead = 1.0;
            // Released clears dead
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
                float att = clamp(age / 0.300, 0.0, 1.0); float env = att * att * (3.0 - 2.0 * att);
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
            float att = clamp(age / 0.300, 0.0, 1.0); float env = att * att * (3.0 - 2.0 * att);
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


def _repack_pat_indices(src):
    """Repack the dictionary index stream in a generated common.glsl from
    byte-stream layout (16 bits per index across patIdx0/1/2) to a single
    patIdx0 with three 10-bit indices per int (4 ints per ivec4 = 12 indices
    each). Replaces fetchIdxByte with fetchIdx10 and patches getNote.

    Saves ~14 KB on a typical 100 KB common.glsl. Requires DICT_NOTES ≤ 1024
    (anything larger needs more than 10 bits per index — function returns
    `src` unchanged with a warning in that case).

    Identical logic to repack_patidx.py — kept here so it runs as part of
    `python3 mod_player.py` instead of as a separate post-step. Errors are
    non-fatal: any failure leaves `src` unchanged so generation still
    completes.
    """
    import re as _re_rp

    def _g(name):
        m = _re_rp.search(rf"#define\s+{name}\s+(\d+)", src)
        return int(m.group(1)) if m else None

    # Idempotency: if fetchIdx10 already exists, the file's already packed.
    if "int fetchIdx10(" in src:
        return src

    DICT_NOTES    = _g("DICT_NOTES")
    IDX_BYTES     = _g("IDX_BYTES")
    IDX_BYTES_PER = _g("IDX_BYTES_PER")
    if DICT_NOTES is None or IDX_BYTES is None or IDX_BYTES_PER is None:
        print("   ⚠️  patIdx repack skipped — required #defines not found")
        return src
    if DICT_NOTES > 1024:
        print(f"   ⚠️  patIdx repack skipped — DICT_NOTES={DICT_NOTES} exceeds 1024 "
              f"(would need >10 bits per index)")
        return src

    total_indices = IDX_BYTES // IDX_BYTES_PER

    def _parse_ivec4_array(name, optional=False):
        m = _re_rp.search(
            rf"const\s+ivec4\s+{name}\s*\[\s*\d+\s*\]\s*=\s*"
            rf"ivec4\s*\[\s*\]\s*\((.*?)\)\s*;",
            src, _re_rp.DOTALL)
        if m is None:
            if optional: return []
            raise RuntimeError(f"patIdx repack: array {name} not found")
        return [tuple(int(x) for x in tup) for tup in _re_rp.findall(
            r"ivec4\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
            m.group(1))]

    try:
        p0 = _parse_ivec4_array("patIdx0")
        p1 = _parse_ivec4_array("patIdx1", optional=True)
        p2 = _parse_ivec4_array("patIdx2", optional=True)
        old_total = len(p0) + len(p1) + len(p2)

        def _ivec4s_to_bytes(lst):
            out = bytearray()
            for v in lst:
                for comp in v:
                    u = comp & 0xFFFFFFFF
                    out.append((u >> 24) & 0xFF)
                    out.append((u >> 16) & 0xFF)
                    out.append((u >>  8) & 0xFF)
                    out.append( u        & 0xFF)
            return out

        byte_stream = (_ivec4s_to_bytes(p0) +
                       _ivec4s_to_bytes(p1) +
                       _ivec4s_to_bytes(p2))[:IDX_BYTES]
        if len(byte_stream) != IDX_BYTES:
            raise RuntimeError(
                f"patIdx repack: byte stream is {len(byte_stream)}, expected {IDX_BYTES}")

        indices = []
        for i in range(0, len(byte_stream), 2):
            lo = byte_stream[i]
            hi = byte_stream[i+1] if i+1 < len(byte_stream) else 0
            indices.append(lo | (hi << 8))
        indices = indices[:total_indices]

        if max(indices) >= 1024:
            raise RuntimeError(
                f"patIdx repack: max index {max(indices)} doesn't fit in 10 bits")

        # Pack 3 × 10-bit indices into each int; bits 30..31 unused.
        packed_ints = []
        for i in range(0, len(indices), 3):
            a = indices[i]
            b = indices[i+1] if i+1 < len(indices) else 0
            c = indices[i+2] if i+2 < len(indices) else 0
            packed_ints.append(a | (b << 10) | (c << 20))

        new_ivec4_count = (len(packed_ints) + 3) // 4
        while len(packed_ints) < new_ivec4_count * 4:
            packed_ints.append(0)
        new_ivec4s = [tuple(packed_ints[i:i+4])
                      for i in range(0, len(packed_ints), 4)]

        # Roundtrip check — must decode every index back to its original value.
        for r, expected in enumerate(indices):
            intIdx, subIdx = divmod(r, 3)
            got = (packed_ints[intIdx] >> (subIdx * 10)) & 0x3FF
            if got != expected:
                raise RuntimeError(
                    f"patIdx repack: roundtrip failed at rank {r} "
                    f"(expected {expected}, got {got})")

        def _fmt(v): return f"ivec4({v[0]},{v[1]},{v[2]},{v[3]})"
        lines = []
        for i in range(0, len(new_ivec4s), 2):
            pair = new_ivec4s[i:i+2]
            lines.append("    " + ", ".join(_fmt(v) for v in pair))
        new_array_src = (f"const ivec4 patIdx0[{len(new_ivec4s)}] = ivec4[](\n"
                         + ",\n".join(lines) + "\n);\n")

        out = src

        def _kill_array(text, name):
            pat = (rf"const\s+ivec4\s+{name}\s*\[\s*\d+\s*\]\s*=\s*"
                   rf"ivec4\s*\[\s*\]\s*\(.*?\)\s*;")
            return _re_rp.sub(pat, "", text, count=1, flags=_re_rp.DOTALL)

        out = _kill_array(out, "patIdx0")
        out = _kill_array(out, "patIdx1")
        out = _kill_array(out, "patIdx2")

        # Splice the packed array right after patBitmap0 (its semantic neighbour).
        m = _re_rp.search(
            r"(const\s+ivec4\s+patBitmap0\s*\[\s*\d+\s*\]\s*=\s*"
            r"ivec4\s*\[\s*\]\s*\(.*?\)\s*;)", out, _re_rp.DOTALL)
        if m is None:
            raise RuntimeError("patIdx repack: patBitmap0 anchor not found")
        out = (out[:m.end()]
               + "\n\n// 10-bit-packed dictionary indices (3 per int, 12 per ivec4)\n"
               + new_array_src + out[m.end():])

        new_fetch = (
            "// ── Index stream 10-bit fetch (consolidated single chunk) ────────────────\n"
            "// Returns the dictionary index directly (0..1023). Replaces the previous\n"
            "// fetchIdxByte + 2-byte combine in getNote with a single fetch + shift + mask.\n"
            "int fetchIdx10(int rank) {\n"
            "    int intIdx   = rank / 3;\n"
            "    int subIdx   = rank - intIdx * 3;          // rank % 3\n"
            "    int ivec4Idx = intIdx >> 2;\n"
            "    int compIdx  = intIdx & 3;\n"
            "    int packed   = patIdx0[ivec4Idx][compIdx];\n"
            "    return (packed >> (subIdx * 10)) & 0x3FF;\n"
            "}")

        old_fetch = _re_rp.search(
            r"// ── Index stream byte fetch.*?int\s+fetchIdxByte\s*\([^)]*\)"
            r"\s*\{[^}]*\}", out, _re_rp.DOTALL)
        if old_fetch is None:
            raise RuntimeError("patIdx repack: fetchIdxByte not found to replace")
        out = out[:old_fetch.start()] + new_fetch + out[old_fetch.end():]

        old_block = _re_rp.search(
            r"// 3\) Look up index and fetch note from dictionary\s*\n"
            r"\s*int dictIdx;\s*\n"
            r"#if IDX_BYTES_PER == 1\s*\n"
            r"\s*dictIdx = fetchIdxByte\(rank\);\s*\n"
            r"#else\s*\n"
            r"\s*int lo = fetchIdxByte\(rank \* 2\);\s*\n"
            r"\s*int hi = fetchIdxByte\(rank \* 2 \+ 1\);\s*\n"
            r"\s*dictIdx = lo \| \(hi << 8\);\s*\n"
            r"#endif", out)
        if old_block is None:
            print("   ⚠️  patIdx repack: getNote dictIdx block not in expected form — "
                  "edit getNote manually so dictIdx = fetchIdx10(rank);")
        else:
            out = (out[:old_block.start()]
                   + "// 3) Look up 10-bit index and fetch note from dictionary\n"
                     "    int dictIdx = fetchIdx10(rank);"
                   + out[old_block.end():])

        # Remove any leftover progressive-load helper from older patches and
        # leave a no-op stub so any caller still compiles.
        out = _re_rp.sub(
            r"// ── Progressive pattern-index loading.*?"
            r"int g_curSongPos = SONG_LENGTH;",
            "", out, flags=_re_rp.DOTALL)
        out = _re_rp.sub(
            r"// ═══ Progressive pattern-load setter.*?"
            r"void setProgressiveLoadPosition\(float time\) \{\s*"
            r"g_curSongPos = getPosition\(time\)\.songPos;\s*\}",
            "", out, flags=_re_rp.DOTALL)
        if "void setProgressiveLoadPosition" not in out:
            out = out.rstrip() + (
                "\n\n"
                "// No-op stub for any tab still calling setProgressiveLoadPosition().\n"
                "// Progressive loading was tied to the old patIdx chunking, which\n"
                "// doesn't exist in the 10-bit-packed layout. Safe to delete once\n"
                "// the call sites in BufferA / Image / Sound are gone.\n"
                "void setProgressiveLoadPosition(float time) {}\n"
            )

        out = _re_rp.sub(
            r"#define IDX_INTS\s+\d+",
            f"#define IDX_INTS          {len(new_ivec4s)}   "
            f"// 10-bit packed: 3 indices per int, 12 per ivec4",
            out)

        saved = len(src) - len(out)
        pct = 100 * saved / max(1, len(src))
        print(f"   🗜️  patIdx repacked: {old_total} → {len(new_ivec4s)} ivec4s, "
              f"saved {saved:,} bytes ({pct:.1f}%)")
        return out
    except Exception as _e:
        print(f"   ⚠️  patIdx repack skipped: {_e}")
        return src


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
    parser.add_argument('--bitrate', choices=['lo','med','hi','ultra'], default='hi',
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
    parser.add_argument('--viz', type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8], default=1,
                        help='Image-tab visualizer:\n'
                             '  0 = None             (black backdrop, fastest compile)\n'
                             '  1 = Reactive 001     (PAEz fork — SDF circles + cosmic web)  ← default\n'
                             '  2 = Fluxline Surfer  (mrange — DR2 dodecahedron + glowtracer)\n'
                             '  3 = Zuvuya           (city/stars + audio-reactive curtain)\n'
                             '  4 = Maya             (raymarched fractal tunnel-warp)\n'
                             '  5 = Dodecahedron     (Philip Bertani — DR2 IFS fractal raymarcher)\n'
                             '  6 = Disco Combined   (smoke spotlights + lasers/clouds, time-driven)\n'
                             '  7 = Sparkly 4D       (Philip Bertani — 4D IFS volumetric raymarcher)\n'
                             '  8 = Skywalker        (orblivius — flying-curve terrain + sync stars)')
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
    parser.add_argument('--142', '--dsp142', dest='use_142_dsp', action='store_true', default=False,
                        help='Generate Sound tab with USE_142_DSP=1 — reverts the '
                             'expensive v1.45 DSP paths (per-note isBass routing in '
                             'main mix, 2-tap Only3D, 64-iter PhatBass walkback) to '
                             'their simpler v1.42 forms. Lower shader complexity, '
                             'compiles on weaker drivers (ANGLE/Mali/older Adreno) '
                             'that crash on the full v1.45 path. Toggleable at '
                             "compile time inside the GLSL — flip the #define in "
                             'Common to switch without re-running the encoder.')
    parser.add_argument('--max-compat', action='store_true', default=False,
                        help='Maximum-compatibility build: implies --142 '
                             '(USE_142_DSP=1 → simpler v1.42-style DSP path) '
                             'plus the lanczos3/small-reverb/no-surround/'
                             'fft_n=512/extra-pragmas preset that was always '
                             'the project default. The DSP bit is what '
                             'actually matters now — strict GLSL parsers '
                             '(ANGLE/Mali/older Adreno) crash on the full '
                             'v1.45 path. Use this when the user reports '
                             '"shader fails to compile" or a black audio '
                             'tab. Flips can be overridden by individual '
                             'knobs (--surround, --reverb-size full, etc.).')
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
            # Flip the Sound tab to the simpler v1.42-compatible DSP path —
            # this is what actually keeps the shader compiling on ANGLE/Mali/
            # older Adreno. Without this, --max-compat would be a near-no-op
            # since the other knobs are already the project defaults.
            'use_142_dsp':            True,
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
    # Timeline GLSL — populated by mikit_engine for IT/S3M/MOD; None on fallback.
    _it_timeline_glsl = None

    if fmt == 'S3M':
        s3m = S3MFile(args.modfile)
        print(f"🎵 {s3m.title}")
        print(f"   Instruments: {s3m.num_instruments}, Patterns: {s3m.num_patterns}, Channels: {s3m.num_channels}")
        print(f"   Speed: {s3m.initial_speed}, Tempo: {s3m.initial_tempo}")
        
        # ── mikit_engine: tick-accurate S3M simulation ─────────────────────
        try:
            print("   🔬 Running mikit_engine tick simulation...")
            _mk_player = load_s3m_native(args.modfile)
            _mk_segs   = _mk_player.run()
            _mk_tps    = _mk_player.initial_tempo * 2.0 / 5.0
            print(f"   ✓  {len(_mk_segs)} voice segments, {_mk_tps:.1f} ticks/sec")
            _180_max_tick = int(180.0 * _mk_tps)
            _mk_segs_st = [s for s in _mk_segs if s.tick_states and s.tick_states[0][0] < _180_max_tick]
            if len(_mk_segs_st) < len(_mk_segs):
                print(f"   ⏱️  180s clip: {len(_mk_segs)} → {len(_mk_segs_st)} segs for ShaderToy")
            _mk_tl = encode_timeline_glsl(_mk_segs_st, _mk_tps)
            _it_timeline_glsl = timeline_to_glsl_arrays(_mk_tl, _mk_tps)
            print(f"   ✓  Timeline: {_mk_tl['num_segs']} segments encoded to GLSL")
        except Exception as _mk_err:
            print(f"   ⚠️  mikit_engine failed ({_mk_err}), using Phase-1 GLSL sim")
            _it_timeline_glsl = None

        # Convert S3M to MOD-compatible structure for the existing GLSL path.
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
        print(f"🎵 {mod.title}")
        print(f"   Patterns: {mod.num_patterns}, Channels: {mod.num_channels}")
        print(f"   Speed: {mod.initial_speed}, Tempo: {mod.initial_tempo}")
        # ── mikit_engine: tick-accurate MOD simulation ──────────────────────
        try:
            print("   🔬 Running mikit_engine tick simulation...")
            _mk_player = load_mod_native(args.modfile)
            _mk_segs   = _mk_player.run()
            _mk_tps    = _mk_player.initial_tempo * 2.0 / 5.0
            print(f"   ✓  {len(_mk_segs)} voice segments, {_mk_tps:.1f} ticks/sec")
            _180_max_tick = int(180.0 * _mk_tps)
            _mk_segs_st = [s for s in _mk_segs if s.tick_states and s.tick_states[0][0] < _180_max_tick]
            if len(_mk_segs_st) < len(_mk_segs):
                print(f"   ⏱️  180s clip: {len(_mk_segs)} → {len(_mk_segs_st)} segs for ShaderToy")
            _mk_tl = encode_timeline_glsl(_mk_segs_st, _mk_tps)
            _it_timeline_glsl = timeline_to_glsl_arrays(_mk_tl, _mk_tps)
            print(f"   ✓  Timeline: {_mk_tl['num_segs']} segments encoded to GLSL")
        except Exception as _mk_err:
            print(f"   ⚠️  mikit_engine failed ({_mk_err}), using Phase-1 GLSL sim")
            _it_timeline_glsl = None
    elif fmt == 'XM':
        xm = XMFile(args.modfile)
        print(f"🎵 {xm.title}")
        print(f"   Patterns: {xm.num_patterns}, Channels: {xm.num_channels}")
        print(f"   Speed: {xm.initial_speed}, Tempo: {xm.initial_tempo}")
        # ── mikit_engine: tick-accurate XM simulation ────────────────────────
        # Full envelopes, multi-sample per instrument, key-off, rel_note, NNA.
        try:
            print("   🔬 Running mikit_engine tick simulation...")
            _mk_player = load_xm_native(args.modfile)
            _mk_segs   = _mk_player.run()
            _mk_tps    = _mk_player.initial_tempo * 2.0 / 5.0
            print(f"   ✓  {len(_mk_segs)} voice segments, {_mk_tps:.1f} ticks/sec")
            _180_max_tick = int(180.0 * _mk_tps)
            _mk_segs_st = [s for s in _mk_segs if s.tick_states and s.tick_states[0][0] < _180_max_tick]
            if len(_mk_segs_st) < len(_mk_segs):
                print(f"   ⏱️  180s clip: {len(_mk_segs)} → {len(_mk_segs_st)} segs for ShaderToy")
            _mk_tl = encode_timeline_glsl(_mk_segs_st, _mk_tps)
            _it_timeline_glsl = timeline_to_glsl_arrays(_mk_tl, _mk_tps)
            print(f"   ✓  Timeline: {_mk_tl['num_segs']} segments encoded to GLSL")
        except Exception as _mk_err:
            import traceback; traceback.print_exc()
            print(f"   ⚠️  mikit_engine failed ({_mk_err}), using Phase-1 GLSL sim")
            _it_timeline_glsl = None
        mod = type('obj', (object,), {
            'title':         xm.title,
            'samples':       xm.samples,
            'num_patterns':  xm.num_patterns,
            'song_length':   len(xm.song_positions),
            'song_positions': xm.song_positions,
            'orders':        xm.song_positions,
            'patterns':      xm.patterns,
            'num_channels':  xm.num_channels,
            'initial_speed': xm.initial_speed,
            'initial_tempo': xm.initial_tempo,
            'channel_settings': [],
            'is_s3m':        False,
            'is_xm':         True,
            'is_it':         False,
        })()
    elif fmt == 'IT':
        # Phase-1 IT support — parser handles uncompressed + IT-packed
        # samples (mikIT-port decompressor), reads compressed pattern
        # streams with the channel-mask scheme, translates IT letter
        # effects via the inline _xm_or_it_effect_to_mod table, drops
        # envelopes/NNA/filters.
        it = ITFile(args.modfile)
        print(f"🎵 {it.title}")
        print(f"   Patterns: {it.num_patterns}, Channels: {it.num_channels}")
        print(f"   Speed: {it.initial_speed}, Tempo: {it.initial_tempo}")
        mod = type('obj', (object,), {
            'title':         it.title,
            'samples':       it.samples,
            'num_patterns':  it.num_patterns,
            'song_length':   len(it.song_positions),
            'song_positions': it.song_positions,
            'orders':        it.song_positions,
            'patterns':      it.patterns,
            'num_channels':  it.num_channels,
            'initial_speed': it.initial_speed,
            'initial_tempo': it.initial_tempo,
            'channel_settings': it.channel_settings,
            'is_s3m':        False,
            'is_xm':         False,
            'is_it':         True,
            # IT mix/global volume — applied as a gain reduction in the
            # audio mix so we match openMPT's level.
            'global_volume': it.global_volume,
            'mix_volume':    it.mix_volume,
            # IT per-channel default pan (0..64). Used to override the
            # MOD-style hardcoded LRRL pattern in the audio mix.
            'channel_pan':   it.channel_pan,
        })()
        # ── mikit_engine: run tick-accurate IT simulation → pre-baked timeline ──
        # Replaces stateless GLSL getChannelOutput with pre-computed voice segments.
        # On failure, falls back to the old Phase-1 approach (USE_TIMELINE_DSP=0).
        try:
            print("   🔬 Running mikit_engine tick simulation...")
            _mk_player = load_it_native(args.modfile)
            _mk_segs = _mk_player.run()
            _mk_tps = _mk_player.initial_tempo * 2.0 / 5.0
            print(f"   ✓  {len(_mk_segs)} voice segments, {_mk_tps:.1f} ticks/sec")
            _180_max_tick = int(180.0 * _mk_tps)
            _mk_segs_st = [s for s in _mk_segs if s.tick_states and s.tick_states[0][0] < _180_max_tick]
            if len(_mk_segs_st) < len(_mk_segs):
                print(f"   ⏱️  180s clip: {len(_mk_segs)} → {len(_mk_segs_st)} segs for ShaderToy")
            _mk_tl = encode_timeline_glsl(_mk_segs_st, _mk_tps)
            _it_timeline_glsl = timeline_to_glsl_arrays(_mk_tl, _mk_tps)
            print(f"   ✓  Timeline: {_mk_tl['num_segs']} segments encoded to GLSL")
        except Exception as _mk_err:
            print(f"   ⚠️  mikit_engine failed ({_mk_err}), using Phase-1 GLSL sim")
            _it_timeline_glsl = None
    elif fmt in ('STM', 'MTM'):
        raise ValueError(
            f"{fmt} format is not yet implemented in this player. "
            f"Currently supported: MOD (full), S3M (partial), XM/IT (Phase 1)."
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
        #
        # CRITICAL: PRESERVE GLOBAL EFFECTS from muted channels. Effects
        # like F (set speed/tempo), D (pattern break), B (position jump)
        # affect playback timing and song structure — stripping them from
        # muted channels makes --solo play at wrong tempo / skip pattern
        # transitions. Specifically, GADGET.IT pat 7 row 0 ch1 has A.03
        # (set speed 3) which halves row time; without it --solo plays
        # pat 7 at 2× the duration → bass on track 1 lands at 0:08 instead
        # of 0:04. Keep those cells' EFFECT (and effect-only cells stay
        # intact); just zero out the audio side (sample/period/volume).
        _global_fx = {0xF, 0xD, 0xB}   # MOD: F=tempo/speed, D=patbreak, B=jump
        muted_cells = 0
        for pat_idx, pat in enumerate(mod.patterns):
            if pat is None:
                continue
            for row in pat:
                upper = min(len(row), mod.num_channels)
                for ch in range(upper):
                    if ch == solo_ch:
                        continue
                    cell = row[ch]
                    if is_s3m:
                        # S3M command codes: 1=A(set speed), 2=B(jump),
                        # 3=C(pattern break), 14=T(set tempo).
                        if cell.get('command') in (1, 2, 3, 14):
                            # keep effect — silence audio side only
                            row[ch] = {'note': 255, 'instrument': 0,
                                       'volume': 255,
                                       'command': cell['command'],
                                       'info':    cell['info']}
                        else:
                            row[ch] = dict(empty)
                    else:
                        if cell.get('effect') in _global_fx:
                            row[ch] = {'sample': 0, 'period': 0,
                                       'effect': cell['effect'],
                                       'param':  cell['param']}
                        else:
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
    
    # ── Truncate song_positions to 180s before pattern encoding ─────────────────
    # Segments are already clipped above; do the same for the pattern-bytes path
    # so patterns referenced only beyond 180s are excluded from the GLSL output.
    _mk_player_ref = locals().get('_mk_player')
    if _mk_player_ref is not None and hasattr(_mk_player_ref, 'row_events'):
        _sp_tps = _mk_player_ref.initial_tempo * 2.0 / 5.0
        _sp_max_tick = int(180.0 * _sp_tps)
        _sp_last = 0
        for (_t, _sp, _r, _pn) in _mk_player_ref.row_events:
            if _t >= _sp_max_tick:
                break
            if _sp > _sp_last:
                _sp_last = _sp
        _sp_n = _sp_last + 1
        if _sp_n < len(mod.song_positions):
            _sp_orig = len(mod.song_positions)
            mod.song_positions = list(mod.song_positions[:_sp_n])
            if hasattr(mod, 'orders'):
                mod.orders = mod.song_positions
            if hasattr(mod, 'song_length'):
                mod.song_length = _sp_n
            print(f"   ✂️  180s clip: song_positions {_sp_orig} → {_sp_n} (patterns beyond 180s excluded)")

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

    
    # Generate README.md (single concise instructions file)
    readme_file = base_name + "_README.md"
    with open(readme_file, 'w') as f:
        f.write(f"""# {mod.title} - ShaderToy MOD Player

## Quick Start

### Method 1: ShaderToy Plugin (Recommended)
1. Install [ShaderToy Unofficial Plugin](https://github.com/patuwwy/ShaderToy-Unofficial-Plugin)
2. Load shader in ShaderToy
3. Plugin → Custom Textures → Add `{png_file}`
4. Common tab: iChannel0 → Custom → `{png_file}`

### Method 2: Standalone HTML
Just open `{base_name}_player.html` in a browser - works offline!

## Files
- `{base_name}_player.html` - Standalone HTML player
- `{base_name}_shadertoy_common.glsl` - MOD data + helpers
- `{base_name}_shadertoy_sound.glsl` - Audio engine
- `{base_name}_shadertoy_image.glsl` - Visualizer
- `{base_name}_shadertoy_bufferA.glsl` - FFT + UI state
- `{png_file}` - Sample data (PNG texture)

## ShaderToy Setup
1. Create new shader at shadertoy.com
2. **Common tab**: paste `{base_name}_shadertoy_common.glsl`
3. **Buffer A**: paste `{base_name}_shadertoy_bufferA.glsl`
4. **Sound**: paste `{base_name}_shadertoy_sound.glsl`
5. **Image**: paste `{base_name}_shadertoy_image.glsl`

## Loop Modes (PNG pixel 0 alpha)
- `0` = Normal (full song loop)
- `255` = Testing (10 second loop)

Generated by MOD2GLSL v1.55
""")
    
    # Generate HTML player — segment-based (Python ITPlayer → JS replay) with
    # fallback to the legacy JS-engine player when mikit segments are unavailable.
    html_file = base_name + "_player.html"
    _smk_player = locals().get('_mk_player')
    _smk_segs   = locals().get('_mk_segs')
    if _smk_player is not None and _smk_segs is not None:
        print(f"\n🎵 Generating segment-based HTML player ({len(_smk_segs)} segs)...")
        try:
            create_segment_player_html(_smk_player, _smk_segs, mod.samples, html_file, mod.title)
        except Exception as _seg_err:
            import traceback as _tb; _tb.print_exc()
            print(f"   ⚠️  Segment player failed ({_seg_err}), falling back to fixed player")
            create_fixed_player_html(mod, html_file, args.downsample, compress=True, vec_dim=args.vec_dim)
    else:
        create_fixed_player_html(mod, html_file, args.downsample, compress=True, vec_dim=args.vec_dim)

    # ── ShaderToy 180s hard limit: truncate song_positions to fit ────────────
    # ShaderToy caps Sound shader runtime at 180 seconds. Songs longer than
    # this just get cut off mid-pattern, so we drop trailing orders that
    # start past TIME_LIMIT and emit a #define TIME_LIMIT into Common. The
    # HTML player is unaffected (already generated above).
    TIME_LIMIT_SEC = 180.0
    if _smk_player is not None and _smk_segs is not None:
        _tl_tps = _smk_player.initial_tempo * 2.0 / 5.0
        _tl_max_tick = int(TIME_LIMIT_SEC * _tl_tps)
        _last_keep_pos = 0
        for (_t, _sp, _r, _pn) in _smk_player.row_events:
            if _t > _tl_max_tick:
                break
            if _sp > _last_keep_pos:
                _last_keep_pos = _sp
        _n_keep = _last_keep_pos + 1
        if hasattr(mod, 'song_positions') and _n_keep < len(mod.song_positions):
            _orig_n = len(mod.song_positions)
            print(f"   ⏱️  TIME_LIMIT={TIME_LIMIT_SEC}s: truncating {_orig_n} → {_n_keep} song positions for ShaderToy")
            mod.song_positions = list(mod.song_positions[:_n_keep])
            if hasattr(mod, 'orders'):
                mod.orders = mod.song_positions
            if hasattr(mod, 'song_length'):
                mod.song_length = _n_keep

    # ShaderToy Common tab: VQ-encoded via embedded vq_encoder_v2 (default),
    # or legacy PNG-loaded Common via create_shadertoy_glsl when --use-png.
    glsl_common_file = base_name + "_shadertoy_common.glsl"

    # Cap timeline for ShaderToy: 12 arrays × N entries fills Sound tab fast.
    # 2000 segs ≈ 210 KB of timeline data; combined with VQ codec (~200 KB) the
    # Sound tab stays under ~500 KB. Songs over this limit fall back to the
    # traditional pattern player (USE_TIMELINE_DSP=0). HTML player unaffected.
    _ST_TL_MAX_SEGS = 2000
    _st_timeline_glsl = _it_timeline_glsl
    if _it_timeline_glsl is not None:
        import re as _re_tl
        _tl_m = _re_tl.search(r'TL_NUM_SEGS\s*=\s*(\d+)', _it_timeline_glsl)
        if _tl_m and int(_tl_m.group(1)) > _ST_TL_MAX_SEGS:
            print(f"   ⚠️  Timeline {_tl_m.group(1)} segs > {_ST_TL_MAX_SEGS} ShaderToy limit "
                  f"(Sound tab would be ~{int(_tl_m.group(1))*12*9//1024} KB). "
                  f"ShaderToy uses pattern player; HTML keeps full timeline.")
            _st_timeline_glsl = None

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
                speed = getattr(_mod, 'initial_speed', 6)
                bpm   = getattr(_mod, 'initial_tempo', 125)
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

            elif getattr(mod, 'is_xm', False) or getattr(mod, 'is_it', False):
                # ── XM / IT → VQ encoder bridge ────────────────────────────
                # XMFile / ITFile already produce MOD-style cells (period,
                # sample, effect, param), so the adapter is a straight
                # passthrough — no S3M-letter→MOD-effect remap, no c2spd
                # compensation, no S3M note-table lookup. Without this
                # bridge the encoder tries to re-parse the .xm/.it file as
                # raw MOD bytes, fails, and we fall through to a
                # create_shadertoy_glsl path that doesn't emit
                # `patTickOffset`/`patStartRow`/`surr_channels` —
                # producing GLSL that fails to compile in Buffer A and
                # Sound. The adapter keeps the VQ success path live.
                _outer_mod = mod

                class _XMITToVQAdapter:
                    """Duck-typed adapter that exposes the attributes the
                    VQ encoder reads (title, samples_info, sample_bytes,
                    song_length, pattern_order, num_channels, num_patterns,
                    patterns, data) — built from already-parsed XM/IT data.
                    Cells go through the MOD 4-byte packed format so the
                    encoder's pattern walker sees them as native MOD."""
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
                        self.magic = b'M.K.'   # placeholder; encoder uses
                                               # num_channels above instead.
                        # samples_info: 31 entries with the encoder-expected
                        # field names (XMFile/ITFile use the same dict shape
                        # as MODFile so this is mostly a passthrough).
                        self.samples_info = []
                        self.sample_bytes = []
                        for i in range(31):
                            si = m.samples[i] if (i < len(m.samples) and isinstance(m.samples[i], dict)) else None
                            if si is not None:
                                arr = si.get('data', None)
                                raw_length = si.get('length', 0)
                                # Block-align onset: if the first non-zero sample
                                # in the first VQ block is at position p > 0, prepend
                                # (vec_dim - p) zeros so the transient starts on a
                                # fresh block boundary — eliminates the mixed
                                # silence+transient block that causes clicking.
                                if (arr is not None and getattr(arr, 'size', 0) > 0
                                        and raw_length > 0 and args.vec_dim > 1):
                                    _vd = args.vec_dim
                                    _scan = arr[:min(_vd, len(arr))]
                                    _nz = (_scan != 0).nonzero()[0]
                                    if len(_nz) > 0 and int(_nz[0]) > 0:
                                        _pad = _vd - int(_nz[0])
                                        arr = np.concatenate([np.zeros(_pad, dtype=arr.dtype), arr])
                                        raw_length += _pad
                                # Pre-VQ LPF: if lag-1 autocorr is strongly
                                # negative (near-Nyquist oscillation), apply a
                                # 4th-order Butterworth LPF at 60% Nyquist before
                                # encoding so VQ gets a smoother signal → smaller
                                # block-boundary reconstruction errors → less clicking.
                                if arr is not None and getattr(arr, 'size', 0) > 64:
                                    _af = arr.astype(np.float32)
                                    # Trigger: std of first-differences > 25 means
                                    # adjacent samples swing wildly (percussive
                                    # transient or near-Nyquist oscillation), which
                                    # VQ encodes poorly.  LPF at 0.60 Nyquist before
                                    # encoding; smooth melodic samples are unaffected.
                                    if float(np.std(np.diff(_af))) > 25:
                                        from scipy.signal import butter, sosfilt
                                        _sos = butter(4, 0.60, btype='low', output='sos')
                                        _af2 = sosfilt(_sos, _af)
                                        arr = np.clip(_af2, -128, 127).astype(np.int8)
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
                        # Pack cells into MOD's 4-byte format. Each cell:
                        #   b0 = (sample & 0xF0) | ((period >> 8) & 0x0F)
                        #   b1 = period & 0xFF
                        #   b2 = ((sample & 0x0F) << 4) | (effect & 0x0F)
                        #   b3 = param & 0xFF
                        # Period field is 12 bits (max 4095), sample is
                        # 5-bit (1..31), effect is 4-bit (0..F).
                        self.patterns = []
                        for pat_idx in range(self.num_patterns):
                            pat = m.patterns[pat_idx] if pat_idx < len(m.patterns) else None
                            buf = bytearray(64 * self.num_channels * 4)
                            if pat is not None:
                                for row_i in range(min(64, len(pat))):
                                    row = pat[row_i]
                                    for ch_i in range(self.num_channels):
                                        cell = row[ch_i] if ch_i < len(row) else None
                                        if cell is None: continue
                                        sample = cell.get('sample', 0)
                                        period = max(0, min(4095, cell.get('period', 0)))
                                        effect = cell.get('effect', 0)
                                        param  = cell.get('param', 0)
                                        o = (row_i * self.num_channels + ch_i) * 4
                                        buf[o]   = (sample & 0xF0) | ((period >> 8) & 0x0F)
                                        buf[o+1] = period & 0xFF
                                        buf[o+2] = ((sample & 0x0F) << 4) | (effect & 0x0F)
                                        buf[o+3] = param & 0xFF
                            self.patterns.append(bytes(buf))
                        # Encoder's actual_pattern_rows() reads cell bytes
                        # from `self.data` to detect Dxx/Bxx — synthesize a
                        # MOD-shaped blob with 1084-byte header padding.
                        self.data = b'\x00' * 1084 + b''.join(self.patterns)

                _vqmod.MODFile = _XMITToVQAdapter

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
                6: "Disco Inferno + UFOff Dancer (orblivius/finalman/Lallis — dance-floor scene with poi)",
                7: "Sparkly 4D (Philip Bertani — 4D IFS volumetric raymarcher)",
                8: "Skywalker (orblivius — synchronized flying-curve terrain + star field + cloud)",
            }
            _vname = _viz_names.get(args.viz, f"viz{args.viz}")
            try:
                with open(glsl_common_file) as _cf: _ct = _cf.read()
                # The VQ encoder's emitted Common carries its own (older)
                # banner. Replace the whole banner block with our current
                # one — version, feature list, visualizer line, Git Home,
                # Contact — so the regenerated file matches the source
                # templates exactly.
                import re as _re_v
                _new_banner = (
                    "/* ============================================================================\n"
                    "   GLSL (The Last) MOD Player v1.55 (c) 2026 Orblivius\n"
                    "   \n"
                    "   32 Tracks support, IT/XM/S3M/MOD loader, 3D Surround, PHAT Bass, Velvet Reverb, \n"
                    "   Comb Reverb, FAT, W1 Limiter, RVQ sample compression, configurable downsample\n"
                    "   \n"
                    f"   Visualizer: {_vname}\n"
                    " \n"
                    "   Git Home: https://github.com/mewza/mod2glsl\n"
                    "   Contact:  subband@gmail.com or\n"
                    "             subband@protonmail.com\n"
                    "  ============================================================================ */"
                )
                _ct = _re_v.sub(
                    r"/\* =+\n   GLSL \(The Last\) MOD Player.*?=+ \*/",
                    lambda _m: _new_banner,
                    _ct, count=1, flags=_re_v.DOTALL)

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

                # Inject USE_142_DSP toggle into the VQ-emitted Common. The
                # Sound tab tests this with #if USE_142_DSP to switch between
                # v1.45 fancy DSP and v1.42-compatible simpler DSP. Default
                # tracks the --142 CLI flag.
                if 'USE_142_DSP' not in _ct:
                    _use142_int = 1 if getattr(args, 'use_142_dsp', False) else 0
                    _ct = (
                        "// DSP mode switch — 0 = v1.45 (fancy), 1 = v1.42-compatible (simpler).\n"
                        f"#define USE_142_DSP {_use142_int}\n\n"
                    ) + _ct

                # Inject TIME_LIMIT — ShaderToy hard-caps Sound shader at 180s.
                # song_positions has already been truncated above so the song
                # fits; this constant lets the Sound tab also clamp its output.
                if 'TIME_LIMIT' not in _ct:
                    _ct = (
                        "// ShaderToy Sound tab is capped at 180 seconds of playback.\n"
                        f"#define TIME_LIMIT {TIME_LIMIT_SEC}\n"
                    ) + _ct

                # Also inject USE_TIMELINE_DSP so the Sound tab's #if blocks
                # have a defined value when Common gets prepended. The VQ
                # encoder's Common template doesn't emit it; we infer the
                # value from whether mikit segments were produced.
                # Use _st_timeline_glsl (not raw _smk_segs) so that songs
                # whose segment count exceeded _ST_TL_MAX_SEGS emit 0 here,
                # keeping Common in sync with Sound's actual #if path.
                _tldsp_int = 1 if _st_timeline_glsl is not None else 0
                if 'USE_TIMELINE_DSP' not in _ct:
                    _ct = (
                        "// USE_TIMELINE_DSP=1: Sound reads pre-baked mikit voice segments.\n"
                        "// USE_TIMELINE_DSP=0: Sound uses the traditional GLSL pattern player.\n"
                        f"#define USE_TIMELINE_DSP {_tldsp_int}\n"
                    ) + _ct
                else:
                    import re as _re_tldsp
                    _ct = _re_tldsp.sub(
                        r'#define\s+USE_TIMELINE_DSP\s+[01]',
                        f'#define USE_TIMELINE_DSP {_tldsp_int}',
                        _ct, count=1)

                # Inject PHATBASS_MIX_MODE into the VQ-emitted Common (encoder
                # doesn't emit it). 0 = per-sample (only isBass[] taps), 1 =
                # mix-wide (entire mix gets Hilbert cross-pan). Default
                # tracks --phatbass-mode resolved against bass detection.
                if 'PHATBASS_MIX_MODE' not in _ct:
                    _pb_mode_cli = getattr(args, 'phatbass_mode', 'sample')
                    _pb_mix_int = 0 if _pb_mode_cli == 'sample' else (1 if _pb_mode_cli == 'mix' else 0)
                    _ct = _ct + (
                        "\n// PhatBass routing — 0 = per-sample, 1 = mix-wide. Flip to override.\n"
                        f"#define PHATBASS_MIX_MODE {_pb_mix_int}\n"
                    )

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

                # ── IT master volume (#define MASTER_GAIN) ─────────────────
                # IT files carry a global_volume + mix_volume pair (header
                # offsets 0x30/0x31). openMPT applies their product as a
                # master attenuation: master = global * mix / 128² so a
                # GADGET.IT mix_vol=48 → 0.375× output. The VQ-encoded
                # Sound has no master gain — runs everything at full scale,
                # then stacks PhatBass + FAT4X + ×1.3 crank on top until
                # the DAC clips on every sample. Inject MASTER_GAIN here
                # so the mix sits where openMPT puts it. For non-IT files
                # both fields default to 128 → MASTER_GAIN = 1.0 (no
                # attenuation, MOD/XM behavior preserved).
                _gv = int(getattr(mod, 'global_volume', 128))
                _mv = int(getattr(mod, 'mix_volume', 128))
                _master_gain = (_gv * _mv) / (128.0 * 128.0)
                if 'MASTER_GAIN' not in _ct:
                    # Splice right after the BPM/SPEED defines.
                    _bpm_def = f"#define BPM               {_bpm_actual}"
                    _ct = _ct.replace(
                        _bpm_def,
                        f"#define MASTER_GAIN       {_master_gain:.4f}f   "
                        f"// IT gv={_gv} mv={_mv} → {_master_gain:.4f}\n"
                        + _bpm_def,
                        1)

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
                # Inject enablePhatBass + enableVelvetReverb right after the
                # encoder's enableFAT line (the encoder doesn't emit them).
                # PhatBass + FAT4X used to be gated together by enableFAT —
                # now they're separate. Velvet reverb is opt-in (default off).
                _phatbass_val = "false" if _no_fat else "true"
                if 'enablePhatBass' not in _ct:
                    _ct = _ct.replace(
                        'const bool  enableFAT     = true;',
                        f'const bool  enableFAT     = true;\n'
                        f'const bool  enablePhatBass = {_phatbass_val};\n'
                        f'const bool  enableVelvetReverb = false;', 1
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
                # ── IT channel pan: replace VQ encoder's channelPan when IT carries pan data ──
                # The VQ encoder uses S3M-style channel_settings (mute flags
                # for IT) and emits all-LEFT panning for IT files. Replace
                # the channelPan const with values derived from IT's
                # channel_pan (range 0..64, where 0=L, 32=center, 64=R)
                # so the GLSL stereo image matches what openMPT plays.
                _it_pan = list(getattr(mod, 'channel_pan', []) or [])
                if _it_pan and max(_it_pan or [0]) > 1:
                    import re as _re
                    _nc = int(getattr(mod, 'num_channels', len(_it_pan)))
                    _vals = []
                    for i in range(32):
                        if i < min(_nc, len(_it_pan)):
                            _v = max(0, min(64, _it_pan[i])) / 64.0
                            _vals.append(f'{_v:.3f}')
                        else:
                            _vals.append('0.5')
                    _new_pan_line = (
                        'const float channelPan[32] = float[]('
                        + ', '.join(_vals) + ');')
                    _pan_pat = _re.compile(
                        r'const\s+float\s+channelPan\s*\[\s*32\s*\]\s*=\s*'
                        r'float\s*\[\s*\]\s*\([^)]*\)\s*;',
                        _re.MULTILINE)
                    _ct, _n = _pan_pat.subn(_new_pan_line, _ct, count=1)

                # ── XM key-off support: inject sampleFadeout / sampleReleaseHold ──
                # The VQ encoder's Common doesn't carry XM envelope info. Splice
                # parallel const-int arrays right after the `samples[]` decl so
                # the patched getChannelOutput (below) can index them. For non-XM
                # samples both arrays are zero/64 → key-off path becomes a no-op.
                if 'sampleFadeout[31]' not in _ct:
                    _xm_samples = list(getattr(mod, 'samples', []))[:31]
                    while len(_xm_samples) < 31:
                        _xm_samples.append({})
                    _fo_v = ', '.join(str(s.get('fadeout', 0)) for s in _xm_samples)
                    _rh_v = ', '.join(str(int(round(s.get('release_factor', 1.0) * 64)))
                                      for s in _xm_samples)
                    # sampleEnvBaseGain: pre-baked average envelope gain for
                    # samples whose envelope plays through to end without a
                    # sustain hold (env_sus=False with env_pts). The GLSL is
                    # stateless and can't simulate per-tick envelope advance
                    # like the JS engine does — so we apply the time-weighted
                    # average y as a constant gain for these voices. Without
                    # this, GADGET.IT inst 27 "Sinewave-envelope" plays at
                    # FULL volume in the GLSL while the JS engine modulates
                    # it down to ~22% average. For samples with sustain
                    # (env_sus=True) or no envelope, gain stays 64 (= 1.0,
                    # no attenuation) and xmReleaseMul handles release tail.
                    def _env_base_gain(s):
                        if (s.get('env_pts') and not s.get('env_sus', False)):
                            return int(round(s.get('release_factor', 1.0) * 64))
                        return 64
                    _eg_v = ', '.join(str(_env_base_gain(s)) for s in _xm_samples)
                    # sampleC5Speed: raw c5_speed (samples/sec at C-5) for IT
                    # samples that ship with values outside MOD's 8000-8800
                    # finetune range — most IT drum/lead/sinewave samples.
                    # The encoder squashes c5_speed into a 4-bit finetune
                    # nibble, capping at 8757 Hz; without this array, IT's
                    # inst 27 sinewave (c5=168000) plays 4 octaves too low.
                    # 0 = use the c4speeds[finetune] fallback (MOD/XM path).
                    # S3M instruments store c2spd (= Hz at S3M C-4 / period 428)
                    # instead of c5_speed.  The GLSL formula `freq = c5spd *
                    # 428 / period` uses period 428 as its reference, which
                    # matches S3M C-4 exactly → store c2spd directly.
                    def _get_c5_speed(s):
                        c5 = s.get('c5_speed', 0)
                        if c5:
                            return c5
                        c2 = s.get('c2spd', 0)
                        if c2:
                            return c2
                        return 0
                    _c5_v = ', '.join(str(_get_c5_speed(s)) for s in _xm_samples)
                    # sampleNNA: New-Note-Action byte for each sample (0=cut,
                    # 1=continue, 2=note-off, 3=note-fade). The GLSL needs
                    # this to know when an old voice should fade through its
                    # envelope release tail instead of being instantly cut
                    # by a new trigger. inst 8 (Siner lead) has nna=2 and
                    # an envelope ending at 0 — without ghost release the
                    # lead notes click off instead of decaying smoothly.
                    _nna_v = ', '.join(str(s.get('nna', 0)) for s in _xm_samples)
                    # sampleEnvReleaseDur: length of the envelope's release
                    # segment in TICKS (env_pts[-1].x - env_pts[sus_pt].x).
                    # 0 means no envelope or no sustain → no ghost-release
                    # window. Used to size the previous-note crossfade for
                    # NNA-noteoff samples; we render the prev voice with a
                    # linear fade across this duration in lieu of stateful
                    # per-tick envelope simulation.
                    def _env_rel_dur(s):
                        if not (s.get('env_pts') and s.get('env_sus', False)):
                            return 0
                        sus_pt = s.get('env_sus_pt', 0)
                        pts = s['env_pts']
                        if sus_pt < len(pts) and len(pts) > 1:
                            return max(0, pts[-1][0] - pts[sus_pt][0])
                        return 0
                    _rd_v = ', '.join(str(_env_rel_dur(s)) for s in _xm_samples)
                    # sampleEnvAttackDur: tick count from env_pts[0] to the
                    # sustain point. Non-zero ONLY for samples whose envelope
                    # starts at silence (env_pts[0].y == 0) — for these the
                    # JS engine plays the new voice silently during the
                    # attack ramp (envX advances 0→sus over this many ticks).
                    # The GLSL is stateless so it ramps s_curr's amplitude
                    # from 0 to 1 over this window in the trigger overlay,
                    # eliminating the "burst" of full-amplitude new voice
                    # before the envelope catches up.
                    def _env_attack_dur(s):
                        pts = s.get('env_pts')
                        if not pts or len(pts) < 2:
                            return 0
                        if pts[0][1] != 0:
                            return 0  # env starts at peak — no attack ramp
                        sus_pt = s.get('env_sus_pt', 0)
                        if sus_pt >= len(pts):
                            return 0
                        return max(0, pts[sus_pt][0] - pts[0][0])
                    _ad_v = ', '.join(str(_env_attack_dur(s)) for s in _xm_samples)
                    # Pack env_pts (x,y pairs) into flat arrays with per-
                    # sample offset/count for the GLSL synthesis-time
                    # envelope lookup. Each point is two ints (tick, value);
                    # values are 0..64. envOffset[i]/envCount[i] index into
                    # envPointsX/Y for sample i. Samples without an envelope
                    # get count=0 → engine treats envValue as 1.0.
                    _env_x_flat = []
                    _env_y_flat = []
                    _env_off_v = []
                    _env_cnt_v = []
                    _env_sus_pt_v = []
                    for s in _xm_samples:
                        pts = s.get('env_pts') or []
                        sus_pt = s.get('env_sus_pt', 0) if s.get('env_sus', False) else -1
                        _env_off_v.append(len(_env_x_flat))
                        _env_cnt_v.append(len(pts))
                        _env_sus_pt_v.append(sus_pt)
                        for p in pts:
                            _env_x_flat.append(int(p[0]))
                            _env_y_flat.append(int(p[1]))
                    # Pad to non-empty for GLSL int[](…) syntax.
                    if not _env_x_flat:
                        _env_x_flat = [0]
                        _env_y_flat = [64]
                    _ex_v = ', '.join(str(v) for v in _env_x_flat)
                    _ey_v = ', '.join(str(v) for v in _env_y_flat)
                    _eo_v = ', '.join(str(v) for v in _env_off_v)
                    _ec_v = ', '.join(str(v) for v in _env_cnt_v)
                    _esp_v = ', '.join(str(v) for v in _env_sus_pt_v)
                    _env_total = len(_env_x_flat)
                    _arrays_block = (
                        f"\n// XM key-off envelope helpers (parallel to samples[]).\n"
                        f"const int sampleFadeout[31]     = int[]({_fo_v});\n"
                        f"const int sampleReleaseHold[31] = int[]({_rh_v});\n"
                        f"// Pre-baked average envelope gain for non-sustained\n"
                        f"// envelopes (env_sus=False). Applied as a constant\n"
                        f"// gain to approximate the JS engine's per-tick\n"
                        f"// envelope advance in this stateless shader.\n"
                        f"const int sampleEnvBaseGain[31] = int[]({_eg_v});\n"
                        f"// Raw c5_speed for IT samples (overrides finetune-\n"
                        f"// table lookup when > 0). Required for IT samples\n"
                        f"// with c5_speed outside MOD's 8000-8800 range.\n"
                        f"const int sampleC5Speed[31]     = int[]({_c5_v});\n"
                        f"// IT NNA byte (0=cut 1=cont 2=off 3=fade) and\n"
                        f"// envelope release duration in ticks. Together\n"
                        f"// they let the previous-note crossfade widen to a\n"
                        f"// proper ghost release for NNA=2/3 voices.\n"
                        f"const int sampleNNA[31]         = int[]({_nna_v});\n"
                        f"const int sampleEnvReleaseDur[31] = int[]({_rd_v});\n"
                        f"// ── Per-sample volume envelopes (env_pts) ──\n"
                        f"// Flat (x,y) pairs indexed by sample via offset/count.\n"
                        f"// sampleEnvSusPt[i] is the sustain point INDEX into the\n"
                        f"// sample's envelope (or -1 = no sustain). envValueAt()\n"
                        f"// linearly interpolates between adjacent points; while\n"
                        f"// keyOn it clamps envX at the sustain point's tick.\n"
                        f"const int envPointsX[{_env_total}]  = int[]({_ex_v});\n"
                        f"const int envPointsY[{_env_total}]  = int[]({_ey_v});\n"
                        f"const int sampleEnvOff[31]    = int[]({_eo_v});\n"
                        f"const int sampleEnvCnt[31]    = int[]({_ec_v});\n"
                        f"const int sampleEnvSusPt[31]  = int[]({_esp_v});\n"
                        f"\n"
                        f"// envValueAt — sample-defined envelope at envTime (seconds).\n"
                        f"// envTime: total seconds since trigger.\n"
                        f"// keyOffAge: seconds since key-off; <= 0 means keyOn=true\n"
                        f"//   (sustain at sus point). > 0 means voice has been released\n"
                        f"//   for that long — envX advances past sus by that much\n"
                        f"//   regardless of envTime.\n"
                        f"// Returns y/64.0 in [0,1]. instIdx is 0-based sample index.\n"
                        f"float envValueAt(int instIdx, float envTime, float keyOffAge) {{\n"
                        f"    if (instIdx < 0 || instIdx >= 31) return 1.0;\n"
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
                    # Find end of `samples[]` array initializer and inject after.
                    _smp_end = _ct.find(');', _ct.find('const SampleInfo samples['))
                    if _smp_end >= 0:
                        _eol = _ct.find('\n', _smp_end)
                        _ct = _ct[:_eol+1] + _arrays_block + _ct[_eol+1:]
                # ── BPM-normalised rowStartTick0 replacement ──────────────────────
                # The VQ encoder emits rowStartTick0 from raw speed ticks at the
                # default BPM=125/speed=6, ignoring the song's actual header values.
                # IT/XM/S3M files with non-default header BPM/speed (e.g. elk_hitch:
                # speed=3, BPM=120) end up with 6-tick-per-row tables making the
                # song play ~2× too slow. Post-process: recompute from outer `mod`.
                _tspd = int(getattr(mod, 'initial_speed', 6))
                _tbpm = int(getattr(mod, 'initial_tempo', 125))
                if (_tspd != 6 or _tbpm != 125) and 'rowStartTick0' in _ct:
                    import re as _re_tk
                    _spd2, _bpm2 = _tspd, _tbpm
                    _cumF3 = 0.0
                    _tv3 = [0]
                    for _si3 in range(len(mod.song_positions)):
                        _pi3 = mod.song_positions[_si3]
                        _pat3 = mod.patterns[_pi3]
                        for _ri3 in range(len(_pat3)):
                            _row3 = _pat3[_ri3]
                            for _ch3 in range(mod.num_channels):
                                _n3 = _row3[_ch3] if _ch3 < len(_row3) else {}
                                _e3 = _n3.get('effect', 0)
                                _p3 = _n3.get('param', 0)
                                if _e3 == 0xF and _p3 > 0:
                                    if _p3 < 0x20: _spd2 = _p3
                                    else:          _bpm2 = _p3
                            _cumF3 += _spd2 * 125.0 / max(_bpm2, 1)
                            _tv3.append(int(round(_cumF3)))
                    _tot3 = _tv3[-1]
                    _b3 = []
                    for _t3 in _tv3:
                        _b3 += [_t3 & 0xFF, (_t3 >> 8) & 0xFF]
                    while len(_b3) % 16:
                        _b3.append(0)
                    _ic3 = len(_b3) // 16
                    _iv3 = []
                    for _i3 in range(_ic3):
                        _o3 = _i3 * 16
                        _ints3 = []
                        for _j3 in range(4):
                            _bs3 = _b3[_o3 + _j3*4 : _o3 + _j3*4 + 4]
                            _v3 = (_bs3[0]<<24)|(_bs3[1]<<16)|(_bs3[2]<<8)|_bs3[3]
                            if _v3 >= 0x80000000: _v3 -= 0x100000000
                            _ints3.append(_v3)
                        _iv3.append(f'ivec4({_ints3[0]},{_ints3[1]},{_ints3[2]},{_ints3[3]})')
                    _new_ta = ', '.join(_iv3)
                    _ct = _re_tk.sub(
                        r'const ivec4 rowStartTick0\[\d+\] = ivec4\[\]\([^;]+\);',
                        f'const ivec4 rowStartTick0[{_ic3}] = ivec4[]({_new_ta});',
                        _ct)
                    _ct = _re_tk.sub(
                        r'#define TOTAL_TICKS\s+\d+',
                        f'#define TOTAL_TICKS       {_tot3}',
                        _ct)
                    print(f"   🔧 rowStartTick0 patched: {_tspd} ticks/row @ BPM={_tbpm} → TOTAL_TICKS={_tot3}")
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
                                 timeline_glsl=_st_timeline_glsl,
                                 compat={
                                     'no_surround':   getattr(args, '_compat_no_surround', False),
                                     'no_fat':        getattr(args, '_compat_no_fat', False),
                                     'reverb_2x2':    getattr(args, '_compat_reverb_2x2', False),
                                     'fft_n':         getattr(args, '_compat_fft_n', 256),
                                     'extra_pragmas': getattr(args, '_compat_extra_pragmas', False),
                                     'phatbass_mode': args.phatbass_mode,
                                     'use_142_dsp':   args.use_142_dsp,
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
                         timeline_glsl=_st_timeline_glsl,
                         compat={
                             'no_surround':   getattr(args, '_compat_no_surround', False),
                             'no_fat':        getattr(args, '_compat_no_fat', False),
                             'reverb_2x2':    getattr(args, '_compat_reverb_2x2', False),
                             'fft_n':         getattr(args, '_compat_fft_n', 256),
                             'extra_pragmas': getattr(args, '_compat_extra_pragmas', False),
                             'phatbass_mode': args.phatbass_mode,
                             'use_142_dsp':   args.use_142_dsp,
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
                # Rename captured getSample → _decodeSample, append RVQ_LPF wrapper
                # so HF quantization noise can be filtered without regenerating.
                _wrapped = []
                for _f in _gs_fns:
                    _wrapped.append(_f.replace('float getSample(', 'float _decodeSample(', 1))
                    _wrapped.append(
                        'float getSample(int sampleIdx) {\n'
                        '    if (sampleIdx < 0 || sampleIdx >= TOTAL_SAMPLES) return 0.0;\n'
                        '#if RVQ_LPF\n'
                        '    // 5-tap triangular FIR: reduces RVQ block-boundary jumps by ~80%\n'
                        '    // vs raw decode, at the cost of 2 extra _decodeSample calls.\n'
                        '    int _a2 = max(0, sampleIdx - 2);\n'
                        '    int _a1 = max(0, sampleIdx - 1);\n'
                        '    int _c1 = min(TOTAL_SAMPLES - 1, sampleIdx + 1);\n'
                        '    int _c2 = min(TOTAL_SAMPLES - 1, sampleIdx + 2);\n'
                        '    return 0.0625 * _decodeSample(_a2)\n'
                        '         + 0.25   * _decodeSample(_a1)\n'
                        '         + 0.375  * _decodeSample(sampleIdx)\n'
                        '         + 0.25   * _decodeSample(_c1)\n'
                        '         + 0.0625 * _decodeSample(_c2);\n'
                        '#else\n'
                        '    return _decodeSample(sampleIdx);\n'
                        '#endif\n'
                        '}')
                _gs_fns = _wrapped
    
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
                # ── XM key-off release multiplier ─────────────────────────────
                # Stateless backward-scan helper. For each sample, returns
                # 0..1 representing the env-release × fadeout multiplier when
                # a key-off cell (instrument byte ≥ 128, period == 0) is the
                # most-recent event on the channel BEFORE the current trigger.
                # Reads the parallel arrays sampleFadeout/sampleReleaseHold
                # injected into Common above.
                _prelude_parts.append("""\
// XM key-off envelope-release helper — applied as a multiplier on the
// channel output. Combines three effects:
//   1. ENV BASE GAIN — pre-baked time-weighted average env y for samples
//      whose envelope plays through to end without sustain (env_sus=False).
//      The shader is stateless and can't simulate the JS engine's per-tick
//      envelope advance, so we apply the average as a constant gain.
//      For sustained-env or env-less samples, this is 1.0 (no effect).
//   2. ANTI-ALIAS GAIN — when playback freq > Nyquist (chip-leads at
//      c5=33000 playing C-7, sinewaves at c5=168000 playing low notes),
//      lanczos-3 still aliases above-Nyquist content into the audible
//      range. Apply linear `nyq/freq` rolloff with a 0.3 floor for
//      single-cycle synth waveforms (c5 > 50000) — matches the JS
//      engine's anti-alias compensation.
//   3. RELEASE FADE — when the most recent event was a key-off (instrument
//      byte ≥ 128, period == 0), drop ~40 ms toward releaseHold then linear
//      fadeout decay.
// All three are folded into a single multiplier so the per-sample mix loop
// only pays the cost of one trigger search.
float xmReleaseMul(int ch, Position pos, float curTime) {
    int sR = pos.row, sP = pos.songPos;
    int koGlobalRow = -1;
    int trigGlobalRow = -1;
    int trigInst = 0;
    int trigPeriod = 0;
    // ── IT note-cut detection ──
    // While scanning back for the most recent trigger, also watch for any
    // note-cut event (instrument byte bit 6 = 0x40, period=0). A cut found
    // BEFORE the trigger when walking backward means the cut happened
    // AFTER the trigger in song time — voice should be silenced. Without
    // this the GLSL ignores note-cut cells entirely (e.g. GADGET.IT pat 8
    // r10 ch0 has note-cut alongside an F-effect; my parser marks it
    // with bit 6, the JS engine handles it, but the GLSL kept playing
    // the bass for 14 seconds afterward).
    bool sawCutAfterTrig = false;
    for (int lb = 0; lb < 128; lb++) {
        Note n = getNote(sP, sR, ch);
        bool isKO   = (n.instrument >= 128) && (n.period <= 0);
        bool isCut  = ((n.instrument & 0x40) != 0) && (n.instrument < 128) && (n.period <= 0);
        bool isTrig = !isKO && !isCut && (n.period > 0);
        if (isKO && koGlobalRow < 0) {
            koGlobalRow = patRowOffset[sP] + (sR - patStartRow[sP]);
        }
        if (isCut && trigInst == 0) {
            // Note: trigInst==0 means we haven't found a prior trigger yet,
            // so this cut comes AFTER the most-recent trigger in song time.
            sawCutAfterTrig = true;
        }
        if (isTrig) {
            trigPeriod = n.period;
            trigGlobalRow = patRowOffset[sP] + (sR - patStartRow[sP]);
            trigInst = (n.instrument >= 1 && n.instrument < 128 && (n.instrument & 0x40) == 0) ? n.instrument : 0;
            if (trigInst <= 0) {
                int sR2 = sR, sP2 = sP;
                for (int lb2 = 1; lb2 < 128; lb2++) {
                    sR2--;
                    if (sR2 < 0) {
                        if (sP2 > 0) { sP2--; sR2 = patStartRow[sP2] + (patRowOffset[sP2+1] - patRowOffset[sP2]) - 1; }
                        else break;
                    }
                    Note p2 = getNote(sP2, sR2, ch);
                    if (p2.instrument > 0 && p2.instrument < 128) { trigInst = p2.instrument; break; }
                }
            }
            break;
        }
        sR--;
        if (sR < 0) {
            if (sP > 0) { sP--; sR = patStartRow[sP] + (patRowOffset[sP+1] - patRowOffset[sP]) - 1; }
            else break;
        }
    }
    // Note-cut takes priority over everything else — voice is silenced
    // from the cut row onward, regardless of envelope/release/fadeout.
    if (sawCutAfterTrig) return 0.0;
    // Resolve env base gain (1.0 for samples without continuous-env; pre-
    // baked avg for inst 27-style sinewave-envelopes). Applied to every
    // return path so even voices without a key-off get the right average.
    float baseGain = (trigInst > 0 && trigInst <= 31)
        ? float(sampleEnvBaseGain[trigInst - 1]) / 64.0
        : 1.0;
    // Anti-alias gain compensation. lanczos-3 helps but doesn't fully
    // suppress aliasing when stride > 1 source-sample-per-output (e.g.
    // sinewave at c5=168000 playing period 143 → stride 11.4). Without
    // this, IT chip leads / sinewaves come through 2-5× too loud and
    // fizzy. Floor at 0.3 for c5 > 50000 — single-cycle synth waveforms
    // have no real high-freq content to alias, so 95%+ rolloff buries
    // them unnecessarily.
    if (trigInst > 0 && trigInst <= 31 && trigPeriod > 0) {
        int _c5 = sampleC5Speed[trigInst - 1];
        float _c5f = (_c5 > 0) ? float(_c5) : c4speeds[samples[trigInst - 1].finetune & 0xF];
        float _freq = _c5f * 428.0 / float(trigPeriod);
        const float _nyq = 22050.0;
        if (_freq > _nyq) {
            float _aaFloor = (_c5 > 50000) ? 0.3 : 0.0;
            baseGain *= max(_aaFloor, _nyq / _freq);
        }
    }
    if (koGlobalRow < 0 || trigInst <= 0 || trigInst > 31) return baseGain;
    int fo = sampleFadeout[trigInst - 1];
    if (fo <= 0) return 0.0;
    // VQ-emitted Common uses fetchTick(globalRow) → cumulative tick count;
    // divide by TICKS_PER_SEC to get seconds. The fallback Common layout
    // uses rowStartTime[] directly — both end up as seconds.
    float koTime = float(fetchTick(koGlobalRow)) / TICKS_PER_SEC;
    float relTime = curTime - koTime;
    if (relTime <= 0.0) return baseGain;
    float drop = clamp(relTime * 25.0, 0.0, 1.0);
    float hold = float(sampleReleaseHold[trigInst - 1]) / 64.0;
    float envMul = mix(1.0, hold, drop);
    // ticks elapsed since key-off = seconds × ticks/sec.
    float ticksKO = relTime * TICKS_PER_SEC;
    float fadeMul = max(0.0, 1.0 - 2.0 * float(fo) * ticksKO / 65536.0);
    return envMul * fadeMul * baseGain;
}
""")
                _prelude_parts.append("// ═══ end sample decoders ═══════════════════════════════════════════════\n\n")
                _prelude = '\n'.join(_prelude_parts)

                # Multiply each getChannelOutput call by the XM key-off release
                # mul. Pattern is `mix += getChannelOutput(ch, T, pos, rowTime)`
                # — splice in `* xmReleaseMul(ch, pos, T)` at the same call.
                _sound_src = _re3.sub(
                    r'getChannelOutput\(\s*(\w+)\s*,\s*([^,]+?)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)',
                    r'(getChannelOutput(\1, \2, \3, \4) * xmReleaseMul(\1, \3, \2))',
                    _sound_src)

                # ── NNA ghost release: widen previous-note crossfade ──────
                # The default 64-sample crossfade is just an anti-click
                # ramp. For samples with NNA=noteoff/notefade (e.g. inst 8
                # "Siner lead" with env_pts ending at 0), a NEW trigger
                # should leave the OLD voice playing through its envelope
                # release tail. Replace the fixed 64-sample window with one
                # sized by the previous sample's release duration when its
                # NNA calls for a fade — and use a "ghost overlay" mix (new
                # at full + old fading linearly to 0) instead of a blend
                # crossfade. Without this, every Siner lead note clicks
                # off into the next instead of decaying smoothly.
                _prelude = _re3.sub(
                    r'if \(ageSamples < 64\.0 && ageSamples >= 0\.0\) \{',
                    'float _xfLen = 64.0;\n'
                    '    {\n'
                    '        // peek prev trigger inst — quick scan for NNA window\n'
                    '        int _pInst = 0;\n'
                    '        for (int _lb = 0; _lb < 64; _lb++) {\n'
                    '            int _r = trigRow - 1 - _lb;\n'
                    '            int _p = trigPat;\n'
                    '            if (_r < 0) { if (_p > 0) { _p--; _r = patStartRow[_p] + (patRowOffset[_p+1] - patRowOffset[_p]) - 1; } else break; }\n'
                    '            Note _pp = getNote(_p, _r, ch);\n'
                    '            bool _isTone = ((_pp.effect == 0x3 || _pp.effect == 0x5) && _pp.period > 0);\n'
                    '            if (_pp.period > 0 && !_isTone) { _pInst = _pp.instrument; break; }\n'
                    '        }\n'
                    '        if (_pInst >= 1 && _pInst <= 31) {\n'
                    '            int _nna = sampleNNA[_pInst - 1];\n'
                    '            int _relTicks = sampleEnvReleaseDur[_pInst - 1];\n'
                    '            int _foAmt    = sampleFadeout[_pInst - 1];\n'
                    '            float _foSamp = (_foAmt > 0)\n'
                    '                ? (32768.0 / float(_foAmt)) * (44100.0 / TICKS_PER_SEC)\n'
                    '                : 220500.0;\n'
                    '            float _envSamp = (_relTicks > 0)\n'
                    '                ? float(_relTicks) * (44100.0 / TICKS_PER_SEC)\n'
                    '                : 220500.0;\n'
                    '            // NNA=1 (continue) ALWAYS uses long ghost — drum stacking\n'
                    '            // depends on it regardless of NNA_GHOST_MODE.\n'
                    '            // NNA=2/3 only use long ghost when NNA_GHOST_MODE=1; otherwise\n'
                    '            // _xfLen stays at the 64-sample anti-click default.\n'
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
                    '    if (ageSamples < _xfLen && ageSamples >= 0.0) {',
                    _prelude, count=1)
                # Attack ramp on s_curr — multiplies the NEW voice by 0→1 over
                # 130 ms when its sample has a sustained envelope starting at
                # silence (sampleEnvReleaseDur > 0). Done at the s_curr
                # definition site so EVERY return path (overlay branches,
                # bare `return s_curr;`) inherits the attenuation. The ghost
                # `s_prev` is computed separately later and unaffected.
                _prelude = _re3.sub(
                    r'(float s_curr = _gcoBody\([^;]+\);\s*\n)',
                    r'\1'
                    r'    // Apply the sample-defined envelope to s_curr. NO extra\n'
                    r'    // smoothstep — stacking ramps causes a mid-transition\n'
                    r'    // amplitude DIP (new voice rises slower than ghost decays\n'
                    r'    // → both partial in the middle = swoosh). Just envelope.\n'
                    r'    {\n'
                    r'        int _curIns_ = trigNote.instrument;\n'
                    r'        if (_curIns_ >= 1 && _curIns_ <= 31) {\n'
                    r'            float _trigTimeF_ = float(fetchTick(patTickOffset[trigPat]+(trigRow-patStartRow[trigPat]))) / TICKS_PER_SEC;\n'
                    r'            s_curr *= envValueAt(_curIns_ - 1, time - _trigTimeF_, 0.0);\n'
                    r'        }\n'
                    r'    }\n',
                    _prelude, count=1)

                _prelude = _re3.sub(
                    r'float t = ageSamples / 64\.0;\s*\n\s*return s_prev \* \(1\.0 - t\) \+ s_curr \* t;',
                    '            // Ghost overlay engages when _xfLen > 64.5. _xfLen was\n'
                    '            // set long for NNA=1 ALWAYS (drum stacking — additive\n'
                    '            // is correct here) and for NNA=2/3 only when\n'
                    '            // NNA_GHOST_MODE=1. NNA=2/3 use a CROSSFADE rather\n'
                    '            // than additive overlay so chord-change retriggers\n'
                    '            // stay near the pre-trigger amplitude instead of\n'
                    '            // swelling +6 dB during the ghost-release window\n'
                    '            // (matches the JS engine where the new voice is\n'
                    '            // silent during env-attack while the ghost rides\n'
                    '            // its envelope release).\n'
                    '            if (_xfLen > 64.5) {\n'
                    '                // Ghost (s_prev) gets envelope release: it was\n'
                    '                // key-offed at the NEW trigger time, so its envX\n'
                    '                // progresses past the sustain point from then on.\n'
                    '                // envValueAt with keyOn=false advances envX into\n'
                    '                // the release segment — the natural envelope decay.\n'
                    '                int _pIns2 = pTrigNote.instrument;\n'
                    '                if (_pIns2 >= 1 && _pIns2 <= 31) {\n'
                    '                    float _pTrigTime = float(fetchTick(patTickOffset[pTrigPat]+(pTrigRow-patStartRow[pTrigPat]))) / TICKS_PER_SEC;\n'
                    '                    float _curTrigTime = float(fetchTick(patTickOffset[trigPat]+(trigRow-patStartRow[trigPat]))) / TICKS_PER_SEC;\n'
                    '                    // NNA=1 (continue): ghost stays in sustain — not key-offed\n'
                    '                    // NNA=2/3 (noteoff/fade): key-offs at new trigger time\n'
                    '                    float _koAge = (sampleNNA[_pIns2 - 1] == 1) ? 0.0 : max(0.0, time - _curTrigTime);\n'
                    '                    s_prev *= envValueAt(_pIns2 - 1, time - _pTrigTime, _koAge);\n'
                    '                }\n'
                    '                return s_curr + s_prev;\n'
                    '            }\n'
                    '            // Equal-power crossfade over 64 samples for the anti-click\n'
                    '            // window. clamp(t) prevents t>1 from producing negative\n'
                    '            // s_prev + over-amped s_curr.\n'
                    '            float t = clamp(ageSamples / 64.0, 0.0, 1.0);\n'
                    '            return s_prev * (1.0 - t) + s_curr * t;',
                    _prelude, count=1)

                # ── IT c5_speed override: replace c4speeds[finetune] lookups ──
                # The VQ encoder emits `c4speeds[smp.finetune & 0xF] * 428.0`
                # for the playback-rate multiplier. For IT samples whose
                # c5_speed is outside the c4speeds table's 7895–8757 range
                # (e.g. inst 27 sinewave at c5=168000, chiprez at c5=33448),
                # the finetune nibble caps and the GLSL plays the sample
                # multiple octaves too low (= the "samples sound distorted"
                # symptom). When sampleC5Speed[i] > 0, prefer it; otherwise
                # fall back to the finetune lookup (MOD/XM compatibility).
                # NOTE: the c4speeds expressions live inside _gcoBody which
                # is part of the PRELUDE string, not _sound_src yet — apply
                # the rewrite to the prelude before it gets injected below.
                _prelude = _re3.sub(
                    r'c4speeds\[\s*smp\.finetune\s*&\s*0xF\s*\]\s*\*\s*428\.0',
                    r'((sampleC5Speed[trigNote.instrument - 1] > 0 ? '
                    r'float(sampleC5Speed[trigNote.instrument - 1]) : '
                    r'c4speeds[smp.finetune & 0xF]) * 428.0)',
                    _prelude)
                # And the periodToFreqFt() call site uses the same lookup
                # internally — replace the call with a direct computation
                # that honors sampleC5Speed.
                _prelude = _re3.sub(
                    r'periodToFreqFt\(\s*max\(\s*1\s*,\s*int\(basePeriod\)\s*\)\s*,\s*smp\.finetune\s*\)',
                    r'((sampleC5Speed[trigNote.instrument - 1] > 0 ? '
                    r'float(sampleC5Speed[trigNote.instrument - 1]) : '
                    r'c4speeds[smp.finetune & 0xF]) * 428.0 / max(1.0, basePeriod))',
                    _prelude)

                # ── Loud-mode normFactor + post-softLimit makeup gain ──────────
                # The VQ encoder bakes `normFactor = 2.0 / float(NUM_CHANNELS)`
                # into the Sound mix, which drops to 0.167 for 12-channel XM —
                # most of the audible loudness disappears. Replace with the
                # 4-ch=2/N / >4-ch=0.65 split, matching the JS engine. Then
                # splice a hard-clip makeup gain right after the softLimit
                # call (or any reasonable late stage) for >4-channel formats.
                _nc_actual = mod.num_channels if hasattr(mod, 'num_channels') else 4
                if _nc_actual > 4:
                    _sound_src = _sound_src.replace(
                        'const float normFactor = 2.0 / float(NUM_CHANNELS);',
                        f'const float normFactor = 0.85;   // loud mode (>4 channels)')
                    # NOTE: the chain restructure (fat_cs1 → phatbass → softlim)
                    # is now baked into mod_player.py's mainSound emission, so
                    # the VQ-emitted Sound already has the correct order plus
                    # final softLimit. No post-VQ tail-limit splice needed.

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

                # Inject runtime-toggleable knobs after USE_EMBEDDED_DATA
                # if not already present (idempotent).
                if 'RVQ_LPF' not in _common_src:
                    _common_src = _re3.sub(
                        r'(#define\s+USE_EMBEDDED_DATA\s+[01]\s*\n)',
                        r'\1// 3-tap FIR low-pass on RVQ-decoded samples (1=ON, 0=OFF). Suppresses\n'
                        r'// HF quantization noise from --no-rvq2 / --bitrate lo at 3× decode cost.\n'
                        r'#define RVQ_LPF 1\n'
                        r'// NNA mode: 0=clean equal-power crossfade over 64 samples for ALL NNAs\n'
                        r'// (no +6dB swell at trigger, no long ghost overlay — VLC-style).\n'
                        r'// 1=ghost overlay per NNA byte (IT semantics; envelope release tail\n'
                        r'// for NNA=2/3, drum stacking for NNA=1) — matches the JS engine.\n'
                        r'#define NNA_GHOST_MODE 1\n',
                        _common_src, count=1)

                with open(glsl_common_file, 'w') as _f: _f.write(_common_src)
                with open(_glsl_sound,       'w') as _f: _f.write(_sound_src)
                _moved_kb = sum(len(a) for a in _arrays) // 1024
                _moved_fn_count = (len(_fetch_fns) + len(_gs_fns) + len(_gsf_fns) + len(_lz_fns)
                                   + (1 if _gco_fn else 0) + (1 if _gcobody_fn else 0))
                print(f"   ✂️  Moved {len(_arrays)} arrays + {_moved_fn_count} fns "
                      f"({_moved_kb} KB) from Common → Sound prelude")
        except Exception as _splerr:
            print(f"   WARNING: Common→Sound split failed ({_splerr}); leaving Common intact")

    # 10-bit-packed dictionary index repack — saves ~14% on common.glsl.
    # Runs after the Common→Sound split so it sees the final form. The
    # helper is a no-op (returns src unchanged) when DICT_NOTES > 1024 or
    # when the expected anchors aren't present, so it's safe to invoke
    # unconditionally.
    try:
        import os as _os3
        if _os3.path.exists(glsl_common_file):
            _orig = open(glsl_common_file).read()
            _packed = _repack_pat_indices(_orig)
            if _packed is not _orig:
                with open(glsl_common_file, 'w') as _f: _f.write(_packed)
    except Exception as _re_err:
        print(f"   WARNING: patIdx repack failed ({_re_err}); Common left as-is")


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
