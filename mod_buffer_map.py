#!/usr/bin/env python3
"""
MOD Buffer Map Generator
Creates detailed memory maps showing where each sample is stored in textures
"""

import struct
import numpy as np
from PIL import Image
import sys
import os
import json

class MODFile:
    def __init__(self, filename):
        self.filename = filename
        self.samples = []
        self.patterns = []
        self.song_positions = []
        self.num_patterns = 0
        
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

def create_buffer_map(mod, texture_width=1024):
    """Create detailed buffer memory map"""
    
    sample_map = []
    current_position = 0
    
    print("="*80)
    print(f"SAMPLE BUFFER MEMORY MAP - {mod.title}")
    print("="*80)
    print()
    
    # Build the map
    for i, sample in enumerate(mod.samples):
        if sample['length'] > 0:
            data_float = sample['data'].astype(np.float32) / 128.0
            length = len(data_float)
            
            # Calculate texture coordinates
            start_pixel = current_position // 4
            start_channel = current_position % 4
            start_x = start_pixel % texture_width
            start_y = start_pixel // texture_width
            
            end_position = current_position + length - 1
            end_pixel = end_position // 4
            end_channel = end_position % 4
            end_x = end_pixel % texture_width
            end_y = end_pixel // texture_width
            
            sample_info = {
                'index': i + 1,  # 1-based for MOD
                'name': sample['name'][:20],
                'byte_start': current_position,
                'byte_end': current_position + length - 1,
                'length': length,
                'loop_start': sample['repeat_point'],
                'loop_length': sample['repeat_length'],
                'volume': sample['volume'],
                'finetune': sample['finetune'],
                'texture_start': {
                    'pixel': start_pixel,
                    'x': start_x,
                    'y': start_y,
                    'channel': ['R', 'G', 'B', 'A'][start_channel]
                },
                'texture_end': {
                    'pixel': end_pixel,
                    'x': end_x,
                    'y': end_y,
                    'channel': ['R', 'G', 'B', 'A'][end_channel]
                }
            }
            
            sample_map.append(sample_info)
            current_position += length
    
    # Print detailed map
    print(f"Texture Dimensions: {texture_width}x{((current_position + 3) // 4 + texture_width - 1) // texture_width}")
    print(f"Total Samples: {current_position}")
    print(f"Total Pixels Used: {(current_position + 3) // 4}")
    print()
    print("="*80)
    print("SAMPLE INDEX TABLE")
    print("="*80)
    print()
    
    for info in sample_map:
        print(f"Sample #{info['index']:2d}: {info['name']:20s}")
        print(f"  Name:        '{info['name']}'")
        print(f"  Byte Range:  {info['byte_start']:8d} - {info['byte_end']:8d}  (length: {info['length']:6d})")
        print(f"  Start:       Pixel {info['texture_start']['pixel']:6d} @ ({info['texture_start']['x']:4d}, {info['texture_start']['y']:4d}).{info['texture_start']['channel']}")
        print(f"  End:         Pixel {info['texture_end']['pixel']:6d} @ ({info['texture_end']['x']:4d}, {info['texture_end']['y']:4d}).{info['texture_end']['channel']}")
        
        if info['loop_length'] > 2:
            loop_end = info['loop_start'] + info['loop_length']
            print(f"  Loop:        {info['loop_start']:6d} - {loop_end:6d}  (length: {info['loop_length']:6d})")
        else:
            print(f"  Loop:        None (one-shot)")
        
        print(f"  Volume:      {info['volume']:3d}/64")
        print(f"  Finetune:    {info['finetune']:+2d}")
        print()
    
    print("="*80)
    print("GLSL CONSTANT LOOKUP TABLE")
    print("="*80)
    print()
    print("// Sample lookup table: [start, length, loop_start, loop_length, volume, finetune]")
    print("const int SAMPLE_TABLE[186] = int[186](")
    
    # Pad to 31 samples
    all_entries = []
    for i in range(31):
        if i < len(sample_map):
            info = sample_map[i]
            all_entries.append(f"    {info['byte_start']:6d}, {info['length']:6d}, {info['loop_start']:6d}, {info['loop_length']:6d}, {info['volume']:3d}, {info['finetune']:+2d}")
        else:
            all_entries.append(f"         0,      0,      0,      0,   0,  0")
    
    print(",\n".join(all_entries))
    print(");")
    print()
    
    print("="*80)
    print("TEXTURE COORDINATE CALCULATION")
    print("="*80)
    print()
    print("""
To read sample N at position P:

1. Get sample info from table:
   int base = (N - 1) * 6;  // N is 1-based in MOD
   int start = SAMPLE_TABLE[base + 0];
   int length = SAMPLE_TABLE[base + 1];

2. Calculate absolute position:
   int absolutePos = start + P;

3. Convert to pixel coordinates:
   int pixelIndex = absolutePos / 4;          // 4 samples per pixel (RGBA)
   int channel = absolutePos % 4;             // Which channel (0=R, 1=G, 2=B, 3=A)
   int x = pixelIndex % TEXTURE_WIDTH;        // X coordinate
   int y = pixelIndex / TEXTURE_WIDTH;        // Y coordinate

4. Calculate UV coordinates:
   vec2 uv = (vec2(x, y) + 0.5) / vec2(TEXTURE_WIDTH, TEXTURE_HEIGHT);

5. Read sample:
   vec4 pixel = texture(iChannel0, uv);
   float sample = pixel[channel];             // Get specific channel
   sample = (sample - 0.5) * 2.0;             // Convert [0,1] to [-1,1]
""")
    
    print("="*80)
    print("MEMORY LAYOUT VISUALIZATION")
    print("="*80)
    print()
    print("Pixel Layout (each pixel = 4 samples):")
    print()
    print("  Pixel 0: [R G B A] <- Samples 0, 1, 2, 3")
    print("  Pixel 1: [R G B A] <- Samples 4, 5, 6, 7")
    print("  Pixel 2: [R G B A] <- Samples 8, 9, 10, 11")
    print("  ...")
    print()
    print(f"Row Layout ({texture_width} pixels per row):")
    print()
    print(f"  Row 0: Pixels 0 - {texture_width-1} (Samples 0 - {texture_width*4-1})")
    print(f"  Row 1: Pixels {texture_width} - {texture_width*2-1} (Samples {texture_width*4} - {texture_width*8-1})")
    print(f"  ...")
    print()
    
    return sample_map, current_position

def generate_buffer_map_json(sample_map, total_samples, texture_width, output_file):
    """Generate JSON buffer map for programmatic use"""
    
    texture_height = ((total_samples + 3) // 4 + texture_width - 1) // texture_width
    
    buffer_map = {
        'texture': {
            'width': texture_width,
            'height': texture_height,
            'total_samples': total_samples,
            'total_pixels': (total_samples + 3) // 4,
            'format': 'RGBA8',
            'sample_encoding': 'float32 mapped to [0,255]',
            'samples_per_pixel': 4
        },
        'samples': sample_map,
        'usage': {
            'glsl_read_function': '''
float readSample(sampler2D tex, int position) {
    int pixelIdx = position / 4;
    int channel = position % 4;
    int x = pixelIdx % TEXTURE_WIDTH;
    int y = pixelIdx / TEXTURE_WIDTH;
    vec2 uv = (vec2(x, y) + 0.5) / vec2(TEXTURE_WIDTH, TEXTURE_HEIGHT);
    vec4 pixel = texture(tex, uv);
    return (pixel[channel] - 0.5) * 2.0;  // Convert [0,1] to [-1,1]
}
''',
            'get_sample_info': '''
void getSampleInfo(int sampleIdx, out int start, out int length, 
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
'''
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(buffer_map, f, indent=2)
    
    return buffer_map

def create_html_buffer_map(sample_map, texture_width, total_samples, output_file):
    """Create interactive HTML visualization"""
    
    total_pixels = (total_samples + 3) // 4
    texture_height = (total_pixels + texture_width - 1) // texture_width
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>MOD Buffer Map</title>
    <style>
        body {{ font-family: monospace; margin: 20px; background: #1e1e1e; color: #d4d4d4; }}
        h1 {{ color: #4ec9b0; }}
        .info {{ background: #252526; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .grid {{ display: inline-block; border: 2px solid #3e3e42; }}
        .pixel {{ 
            width: 8px; 
            height: 8px; 
            display: inline-block; 
            cursor: pointer;
            border: 1px solid #1e1e1e;
        }}
        .row {{ line-height: 0; }}
        #tooltip {{ 
            position: fixed; 
            background: #252526; 
            border: 1px solid #3e3e42; 
            padding: 10px; 
            border-radius: 5px;
            display: none;
            pointer-events: none;
            max-width: 400px;
            z-index: 1000;
        }}
        .sample-legend {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 10px;
            margin: 20px 0;
        }}
        .sample-item {{ 
            background: #252526; 
            padding: 8px; 
            border-radius: 3px;
            border-left: 4px solid;
        }}
    </style>
</head>
<body>
    <h1>MOD Sample Buffer Map</h1>
    <div class="info">
        <strong>Texture Dimensions:</strong> {texture_width} x {texture_height} pixels<br>
        <strong>Total Samples:</strong> {total_samples}<br>
        <strong>Total Pixels:</strong> {total_pixels}<br>
        <strong>Format:</strong> RGBA (4 samples per pixel)
    </div>
    
    <h2>Sample Legend</h2>
    <div class="sample-legend">
"""
    
    colors = [
        '#e06c75', '#98c379', '#e5c07b', '#61afef', '#c678dd', '#56b6c2',
        '#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#f0932b', '#eb4d4b',
        '#6ab04c', '#7ed6df', '#e056fd', '#686de0'
    ]
    
    for i, info in enumerate(sample_map):
        color = colors[i % len(colors)]
        html += f"""
        <div class="sample-item" style="border-color: {color}">
            <strong>#{info['index']}: {info['name']}</strong><br>
            <small>
                Bytes: {info['byte_start']} - {info['byte_end']} (len: {info['length']})<br>
                Pixels: {info['texture_start']['pixel']} - {info['texture_end']['pixel']}
            </small>
        </div>
"""
    
    html += """
    </div>
    
    <h2>Buffer Visualization</h2>
    <p>Hover over pixels to see details. Each pixel = 4 samples (RGBA)</p>
    <div class="grid" id="grid"></div>
    <div id="tooltip"></div>
    
    <script>
"""
    
    # Generate pixel data
    html += f"const textureWidth = {texture_width};\n"
    html += f"const textureHeight = {texture_height};\n"
    html += "const samples = [\n"
    
    for info in sample_map:
        html += "  " + json.dumps({
            'index': info['index'],
            'name': info['name'],
            'start': info['byte_start'],
            'end': info['byte_end'],
            'length': info['length'],
            'color': colors[(info['index'] - 1) % len(colors)]
        }) + ",\n"
    
    html += """];\n
    
const grid = document.getElementById('grid');
const tooltip = document.getElementById('tooltip');

// Create pixel map
const pixelMap = new Array(textureWidth * textureHeight).fill(null);

samples.forEach(sample => {
    const startPixel = Math.floor(sample.start / 4);
    const endPixel = Math.floor(sample.end / 4);
    
    for (let p = startPixel; p <= endPixel; p++) {
        if (p < pixelMap.length) {
            pixelMap[p] = sample;
        }
    }
});

// Render grid
for (let y = 0; y < textureHeight; y++) {
    const row = document.createElement('div');
    row.className = 'row';
    
    for (let x = 0; x < textureWidth; x++) {
        const idx = y * textureWidth + x;
        const pixel = document.createElement('div');
        pixel.className = 'pixel';
        
        const sample = pixelMap[idx];
        if (sample) {
            pixel.style.backgroundColor = sample.color;
        } else {
            pixel.style.backgroundColor = '#2d2d30';
        }
        
        pixel.addEventListener('mouseenter', (e) => {
            if (sample) {
                const pixelIdx = y * textureWidth + x;
                const baseSample = pixelIdx * 4;
                
                const channels = ['R', 'G', 'B', 'A'];
                let channelInfo = '';
                for (let c = 0; c < 4; c++) {
                    const samplePos = baseSample + c;
                    if (samplePos >= sample.start && samplePos <= sample.end) {
                        const offset = samplePos - sample.start;
                        channelInfo += channels[c] + ': sample[' + offset + ']<br>';
                    }
                }
                
                tooltip.innerHTML = `
                    <strong>Pixel ${pixelIdx} @ (${x}, ${y})</strong><br>
                    <strong>Sample #${sample.index}: ${sample.name}</strong><br>
                    Byte range: ${sample.start} - ${sample.end}<br>
                    Length: ${sample.length} samples<br>
                    <br>
                    <strong>This pixel contains:</strong><br>
                    ${channelInfo}
                `;
                tooltip.style.display = 'block';
                tooltip.style.left = e.pageX + 10 + 'px';
                tooltip.style.top = e.pageY + 10 + 'px';
            }
        });
        
        pixel.addEventListener('mouseleave', () => {
            tooltip.style.display = 'none';
        });
        
        row.appendChild(pixel);
    }
    
    grid.appendChild(row);
}
    </script>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html)

def main():
    if len(sys.argv) < 2:
        print("Usage: python mod_buffer_map.py <modfile.mod>")
        print("\nGenerates detailed memory maps for MOD sample buffers")
        sys.exit(1)
    
    mod_filename = sys.argv[1]
    
    if not os.path.exists(mod_filename):
        print(f"Error: File not found: {mod_filename}")
        sys.exit(1)
    
    # Parse MOD
    mod = MODFile(mod_filename)
    
    # Create buffer map
    sample_map, total_samples = create_buffer_map(mod)
    
    # Save JSON map
    output_base = os.path.splitext(os.path.basename(mod_filename))[0]
    json_file = f"{output_base}_buffer_map.json"
    
    buffer_map = generate_buffer_map_json(sample_map, total_samples, 1024, json_file)
    
    # Generate HTML visualization
    html_file = f"{output_base}_buffer_map.html"
    create_html_buffer_map(sample_map, 1024, total_samples, html_file)
    
    print("="*80)
    print(f"Files generated:")
    print(f"  - {json_file} (machine-readable)")
    print(f"  - {html_file} (interactive visualization)")
    print("="*80)
    print(f"\nOpen {html_file} in your browser to explore the buffer layout!")

if __name__ == '__main__':
    main()
