#!/usr/bin/env python3
"""
Generate a compressed self-contained HTML report.
Images are resized and converted to JPEG with compression.
"""

import base64
import io
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).parent

# Compression settings
MAX_WIDTH = 1400  # Max width in pixels
JPEG_QUALITY = 82  # JPEG quality (0-100)

def compress_image(image_path: Path) -> str:
    """Compress image and convert to base64 data URI."""
    img = Image.open(image_path)

    # Convert RGBA to RGB (JPEG doesn't support alpha)
    if img.mode in ('RGBA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize if too large
    if img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / img.width
        new_height = int(img.height * ratio)
        img = img.resize((MAX_WIDTH, new_height), Image.LANCZOS)

    # Save to JPEG buffer with compression
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=JPEG_QUALITY, optimize=True)
    data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return f"data:image/jpeg;base64,{data}"

def generate_report():
    """Generate the compressed HTML report."""
    images = {}

    # Root directory images
    root_images = [
        'phase_diagram_sparsity.png',
        'slope_eigenvalue_correlation.png',
        'slope_eigenvalue_summary.png'
    ]

    for img in root_images:
        path = BASE_DIR / img
        if path.exists():
            images[img] = compress_image(path)
            print(f"Compressed: {img}")

    # Dynamic hopping plots
    dh_plots_dir = BASE_DIR / 'dynamic_hopping' / 'plots'
    if dh_plots_dir.exists():
        for img_path in sorted(dh_plots_dir.glob('*.png')):
            key = f"dynamic_hopping/plots/{img_path.name}"
            images[key] = compress_image(img_path)
            print(f"Compressed: {key}")

    # Temporal dynamics plots
    td_dir = BASE_DIR / 'temporal_dynamics'
    if td_dir.exists():
        for img_path in sorted(td_dir.glob('*.png')):
            key = f"temporal_dynamics/{img_path.name}"
            images[key] = compress_image(img_path)
            print(f"Compressed: {key}")

    print(f"\nTotal images: {len(images)}")

    html = generate_html(images)

    output_path = BASE_DIR / 'spectral_superposition_report_compressed.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024*1024)
    print(f"\nReport: {output_path}")
    print(f"Size: {size_mb:.2f} MB")

def generate_html(images: dict) -> str:
    """Generate HTML with embedded compressed images."""

    def img_tag(key: str, alt: str, caption: str, fig_num: str) -> str:
        if key in images:
            return f'''<div class="fig"><img src="{images[key]}" alt="{alt}"><p class="cap"><b>{fig_num}:</b> {caption}</p></div>'''
        return f'<p><em>Missing: {key}</em></p>'

    samples = [
        ('n1024_m112_s0.000000_seed0', 'm=112, s=0.0'),
        ('n1024_m112_s0.020204_seed0', 'm=112, s=0.02'),
        ('n1024_m112_s0.040408_seed0', 'm=112, s=0.04'),
        ('n1024_m160_s0.545510_seed1', 'm=160, s=0.55'),
        ('n1024_m208_s0.101020_seed0', 'm=208, s=0.10'),
        ('n1024_m256_s0.666735_seed0', 'm=256, s=0.67'),
        ('n1024_m320_s0.202041_seed1', 'm=320, s=0.20'),
        ('n1024_m352_s0.767755_seed1', 'm=352, s=0.77'),
        ('n1024_m416_s0.323265_seed0', 'm=416, s=0.32'),
        ('n1024_m464_s0.888980_seed0', 'm=464, s=0.89'),
        ('n1024_m512_s0.424286_seed1', 'm=512, s=0.42'),
        ('n1024_m96_s0.990000_seed1', 'm=96, s=0.99'),
    ]

    sample_html = ''
    for idx, (name, config) in enumerate(samples, 1):
        fn = f"3.{idx+3}"
        sample_html += f'''<h4>Sample {idx}: {name} ({config})</h4>
{img_tag(f"dynamic_hopping/plots/{name}_rayleigh_heatmap.png", "Heatmap", f"Rayleigh heatmap ({config})", f"Fig {fn}a")}
{img_tag(f"dynamic_hopping/plots/{name}_jump_events.png", "Jumps", f"Jump events ({config})", f"Fig {fn}b")}
{img_tag(f"dynamic_hopping/plots/{name}_volatility_evolution.png", "Volatility", f"Volatility ({config})", f"Fig {fn}c")}
'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Spectral Superposition Report</title>
<style>
body{{font-family:sans-serif;max-width:1000px;margin:0 auto;padding:20px;line-height:1.6;color:#333}}
h1{{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px}}
h2{{color:#2c3e50;border-bottom:1px solid #ddd;margin-top:30px}}
h3{{color:#3498db;margin-top:25px}}
h4{{color:#555;margin-top:20px}}
.fig{{margin:20px 0;text-align:center}}
.fig img{{max-width:100%;border:1px solid #ddd;border-radius:4px}}
.cap{{font-style:italic;color:#666;font-size:0.9em;margin-top:8px}}
pre{{background:#f5f5f5;padding:12px;border-radius:4px;overflow-x:auto;font-size:0.85em}}
code{{background:#f5f5f5;padding:2px 5px;border-radius:3px;font-size:0.9em}}
table{{border-collapse:collapse;width:100%;margin:15px 0}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left}}
th{{background:#f5f5f5}}
.box{{padding:12px 15px;margin:15px 0;border-radius:0 4px 4px 0}}
.def{{background:#e3f2fd;border-left:4px solid #3498db}}
.key{{background:#e8f5e9;border-left:4px solid #4caf50}}
.dark{{background:#fce4ec;border-left:4px solid #e74c3c}}
.eq{{text-align:center;margin:15px 0;padding:10px;background:#f5f5f5;border-radius:4px}}
hr{{border:none;border-top:2px solid #eee;margin:30px 0}}
</style>
</head>
<body>

<h1>Spectral Superposition: Comprehensive Analysis Report</h1>
<p><em>Understanding Dark Matter Features Through Dynamic and Temporal Analysis</em><br>
<small>Version: v1-temporal5 | January 2025</small></p>

<h2>Contents</h2>
<ol>
<li><a href="#s1">Experimental Setup</a></li>
<li><a href="#s2">Phase Stratification & Spectral Analysis</a></li>
<li><a href="#s3">Dynamic Hopping Analysis</a></li>
<li><a href="#s4">Temporal Dynamics Analysis</a></li>
</ol>

<hr>

<h1 id="s1">1. Experimental Setup</h1>

<p>Large-scale parameter sweep studying superposition in neural networks - how networks represent more features than dimensions.</p>

<div class="box def">
<b>Model:</b> ReLU autoencoder<br>
<code>h = W @ x</code> (R<sup>n</sup> → R<sup>m</sup>)<br>
<code>x' = ReLU(W<sup>T</sup> @ h + b)</code> (R<sup>m</sup> → R<sup>n</sup>)<br>
<code>Loss = MSE(x, x')</code><br>
<b>Input:</b> x<sub>i</sub> = 0 with prob S, else Uniform(0,1)
</div>

<table>
<tr><th>Parameter</th><th>Values</th></tr>
<tr><td>Features (n)</td><td>1024</td></tr>
<tr><td>Hidden (m)</td><td>16-512 (32 values)</td></tr>
<tr><td>Sparsity (S)</td><td>0.0-0.99 (50 values)</td></tr>
<tr><td>Seeds</td><td>0, 1</td></tr>
<tr><td><b>Total</b></td><td><b>3,200 experiments</b></td></tr>
</table>

<p><b>Training:</b> 25K steps, batch=1024, lr=1e-3, Adam, 56 checkpoints<br>
<b>Compute:</b> 8× NVIDIA L4 GPUs</p>

<hr>

<h1 id="s2">2. Phase Stratification & Spectral Analysis</h1>

<h2>2.1 Phase Diagram</h2>

<div class="box def">
<b>Fractional Dimensionality:</b> D<sub>i</sub> = (M<sub>ii</sub>)² / (M²)<sub>ii</sub> where M = W<sup>T</sup>W
</div>

{img_tag('phase_diagram_sparsity.png', 'Phase Diagram', 'D_i vs ||W_i||². Features cluster along rays with slopes = inverse eigenvalues.', 'Fig 2.1')}

<h2>2.2 Slope-Eigenvalue Correlation</h2>

<div class="box key">
<b>Conjecture:</b> Ray slope κ ≈ 1/λ of frame operator S = WW<sup>T</sup>
</div>

{img_tag('slope_eigenvalue_correlation.png', 'Correlation', 'For λ > 1: κ ≈ 1/λ with R² = 0.94, r = 0.97', 'Fig 2.2')}

{img_tag('slope_eigenvalue_summary.png', 'Summary', 'Spectral interpretation breaks down for λ < 1', 'Fig 2.3')}

<h2>2.3 The Dark Matter Regime</h2>

<div class="box dark">
<b>Dark Matter Features (λ < 1):</b>
<ul>
<li>73-84% have R² < 0.9 for D<sub>i</sub> vs ||W<sub>i</sub>||²</li>
<li>Lower eigenspace concentration</li>
<li>Higher entropy, negative curvature in trajectories</li>
</ul>
</div>

<hr>

<h1 id="s3">3. Dynamic Hopping Analysis</h1>

<p><b>Goal:</b> Quantify feature "hopping" via Rayleigh quotient time-variation.</p>

<h2>3.1 Mathematical Foundation</h2>

<div class="eq">
κ<sub>i</sub>(t) = w<sub>i</sub><sup>T</sup> S w<sub>i</sub> / ||w<sub>i</sub>||² where S = WW<sup>T</sup>
</div>

<p><b>Jump detection:</b> Using MAD (Median Absolute Deviation):<br>
σ<sub>robust</sub> = 1.4826 × median(|Δx - median(Δx)|)<br>
Jump if |Δx<sub>i</sub>| > 4 × σ<sub>robust</sub></p>

<h2>3.2 Global Results</h2>

{img_tag('dynamic_hopping/plots/dynamic_hopping_summary.png', 'Summary', 'Comprehensive summary: jump frequency, volatility, synchrony, patterns by sparsity', 'Fig 3.1')}

{img_tag('dynamic_hopping/plots/feature_classification_summary.png', 'Classification', 'Volatility classes, temporal patterns, synchrony, trends', 'Fig 3.2')}

{img_tag('dynamic_hopping/plots/sparsity_comparison.png', 'Sparsity', 'Hopping metrics across sparsity regimes', 'Fig 3.3')}

<h2>3.3 Sample Experiments (12 configurations)</h2>

<p>Each sample shows: Rayleigh heatmap, jump events, volatility evolution.</p>

{sample_html}

<hr>

<h1 id="s4">4. Temporal Dynamics Analysis</h1>

<div class="box key">
<b>Key Question:</b> Is dark matter transient (features "in transit") or persistent?
</div>

<h2>4.1 Dark Matter Evolution</h2>

<p>Track fraction with R² < 0.9 over training.</p>

{img_tag('temporal_dynamics/temporal_dark_matter_evolution.png', 'Evolution', 'Dark matter fraction over training; transition statistics', 'Fig 4.1')}

<h2>4.2 Eigenspace Stability</h2>

<p>Feature migration between eigenspace clusters.</p>

{img_tag('temporal_dynamics/temporal_eigenspace_stability.png', 'Stability', 'Hopping rate, concentration, λ>1 vs λ≤1 comparison', 'Fig 4.2')}

<h2>4.3 Instantaneous Linearity</h2>

<p>Cumulative R² vs instantaneous slope analysis.</p>

{img_tag('temporal_dynamics/temporal_instantaneous_linearity.png', 'Linearity', 'R² by slope stability, convergence, trajectories', 'Fig 4.3')}

<h2>4.4 Slope-Eigenvalue Temporal</h2>

<p>Track κ ≈ 1/λ at each checkpoint.</p>

{img_tag('temporal_dynamics/temporal_slope_eigenvalue_temporal.png', 'κλ Temporal', 'Mean κλ over training, regime comparison, distribution', 'Fig 4.4')}

<h2>4.5 Trajectory Classification</h2>

<p>Classify by trajectory shape: linear, curved, oscillatory, transient, collapsed.</p>

{img_tag('temporal_dynamics/temporal_trajectory_classification.png', 'Classification', 'Distribution by sparsity, sample trajectories, R² by type', 'Fig 4.5')}

<hr>

<p style="text-align:center;color:#888;margin-top:40px">
<b>End of Report</b><br>
Spectral Superposition v1-temporal5 | January 2025
</p>

</body>
</html>'''

if __name__ == '__main__':
    generate_report()
