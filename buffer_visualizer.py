#!/usr/bin/env python3
"""
Buffer Layout Visualizer
Creates visual diagrams of sample buffer layout in texture
"""

import sys
import os

def create_ascii_buffer_diagram(sample_map, texture_width, total_samples):
    """Create ASCII art visualization of buffer layout"""
    
    total_pixels = (total_samples + 3) // 4
    texture_height = (total_pixels + texture_width - 1) // texture_width
    
    print("="*80)
    print("TEXTURE BUFFER LAYOUT (ASCII Visualization)")
    print("="*80)
    print()
    
    # Create a simplified view showing which samples occupy which regions
    # Each character represents a pixel
    
    # Build pixel map
    pixel_map = [' '] * (texture_width * texture_height)
    sample_chars = '123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    for info in sample_map:
        sample_idx = info['index'] - 1
        char = sample_chars[sample_idx % len(sample_chars)]
        
        start_pixel = info['byte_start'] // 4
        end_pixel = info['byte_end'] // 4
        
        for p in range(start_pixel, end_pixel + 1):
            if p < len(pixel_map):
                pixel_map[p] = char
    
    # Print header
    print(f"Texture: {texture_width}x{texture_height} pixels")
    print(f"Each character = 1 pixel = 4 audio samples (RGBA)")
    print()
    print("Legend:")
    for info in sample_map[:min(len(sample_map), 35)]:
        sample_idx = info['index'] - 1
        char = sample_chars[sample_idx % len(sample_chars)]
        print(f"  {char} = Sample #{info['index']:2d} - {info['name'][:30]}")
    print()
    
    # Print grid (show first few rows)
    rows_to_show = min(20, texture_height)
    
    # Column headers
    print("     ", end="")
    for x in range(0, texture_width, 10):
        print(f"{x:10d}", end="")
    print()
    print("     " + "".join([str(i % 10) for i in range(texture_width)]))
    print("    +" + "-" * texture_width + "+")
    
    for y in range(rows_to_show):
        print(f"{y:3d} |", end="")
        for x in range(texture_width):
            idx = y * texture_width + x
            print(pixel_map[idx], end="")
        print("|")
    
    if texture_height > rows_to_show:
        print(f"    | ... ({texture_height - rows_to_show} more rows) ...")
    
    print("    +" + "-" * texture_width + "+")
    print()
    
    # Detailed pixel breakdown for first few samples
    print("="*80)
    print("DETAILED PIXEL BREAKDOWN")
    print("="*80)
    print()
    
    for info in sample_map[:5]:  # First 5 samples
        print(f"Sample #{info['index']}: {info['name']}")
        print(f"  Occupies {(info['length'] + 3) // 4} pixels")
        print(f"  Start: Pixel {info['texture_start']['pixel']:6d} @ ({info['texture_start']['x']:4d},{info['texture_start']['y']:4d}).{info['texture_start']['channel']}")
        print(f"  End:   Pixel {info['texture_end']['pixel']:6d} @ ({info['texture_end']['x']:4d},{info['texture_end']['y']:4d}).{info['texture_end']['channel']}")
        
        # Show first few pixels
        start_pixel = info['byte_start'] // 4
        num_pixels = min(4, (info['length'] + 3) // 4)
        
        print(f"  First {num_pixels} pixels:")
        for i in range(num_pixels):
            pixel_idx = start_pixel + i
            x = pixel_idx % texture_width
            y = pixel_idx // texture_width
            
            # Calculate which samples are in this pixel
            base_sample = pixel_idx * 4
            samples_in_pixel = []
            for ch in range(4):
                sample_pos = base_sample + ch
                if info['byte_start'] <= sample_pos <= info['byte_end']:
                    offset = sample_pos - info['byte_start']
                    samples_in_pixel.append(f"{['R','G','B','A'][ch]}:sample[{offset}]")
            
            print(f"    Pixel {pixel_idx} @ ({x:4d},{y:4d}): {' '.join(samples_in_pixel)}")
        print()
    
    print("="*80)
    print("CHANNEL DISTRIBUTION")
    print("="*80)
    print()
    print("How samples are distributed across RGBA channels:")
    print()
    
    channel_usage = {'R': 0, 'G': 0, 'B': 0, 'A': 0}
    channel_names = ['R', 'G', 'B', 'A']
    
    for pos in range(total_samples):
        channel = channel_names[pos % 4]
        channel_usage[channel] += 1
    
    for ch in ['R', 'G', 'B', 'A']:
        count = channel_usage[ch]
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        bar = '#' * (count // (total_samples // 50 + 1))
        print(f"  {ch}: {count:6d} samples ({percentage:5.1f}%) {bar}")
    
    print()

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
        html += "  " + str({
            'index': info['index'],
            'name': info['name'],
            'start': info['byte_start'],
            'end': info['byte_end'],
            'length': info['length'],
            'color': colors[(info['index'] - 1) % len(colors)]
        }).replace("'", '"') + ",\n"
    
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
                
                tooltip.innerHTML = `
                    <strong>Pixel ${pixelIdx} @ (${x}, ${y})</strong><br>
                    <strong>Sample #${sample.index}: ${sample.name}</strong><br>
                    Byte range: ${sample.start} - ${sample.end}<br>
                    Length: ${sample.length} samples<br>
                    <br>
                    <strong>This pixel contains:</strong><br>
                    R: sample[${baseSample - sample.start}]<br>
                    G: sample[${baseSample + 1 - sample.start}]<br>
                    B: sample[${baseSample + 2 - sample.start}]<br>
                    A: sample[${baseSample + 3 - sample.start}]
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
    
    print(f"\nInteractive HTML map saved to: {output_file}")
    print("Open in browser to explore buffer layout visually")

if __name__ == '__main__':
    # This would be called from the main converter
    print("This module provides visualization functions.")
    print("Use mod_buffer_map.py to generate buffer maps.")
