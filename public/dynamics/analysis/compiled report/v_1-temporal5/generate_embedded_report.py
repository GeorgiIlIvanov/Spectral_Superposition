#!/usr/bin/env python3
"""
Generate a self-contained HTML report with all images embedded as base64.
This creates a single file that can be uploaded to Claude or other LLMs.
"""

import base64
from pathlib import Path

BASE_DIR = Path(__file__).parent

def image_to_base64(image_path: Path) -> str:
    """Convert an image file to base64 data URI."""
    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')

    suffix = image_path.suffix.lower()
    if suffix == '.png':
        mime = 'image/png'
    elif suffix in ['.jpg', '.jpeg']:
        mime = 'image/jpeg'
    elif suffix == '.gif':
        mime = 'image/gif'
    else:
        mime = 'image/png'

    return f"data:{mime};base64,{data}"

def generate_report():
    """Generate the self-contained HTML report."""

    # Collect all images
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
            images[img] = image_to_base64(path)
            print(f"Loaded: {img}")

    # Dynamic hopping plots
    dh_plots_dir = BASE_DIR / 'dynamic_hopping' / 'plots'
    if dh_plots_dir.exists():
        for img_path in sorted(dh_plots_dir.glob('*.png')):
            key = f"dynamic_hopping/plots/{img_path.name}"
            images[key] = image_to_base64(img_path)
            print(f"Loaded: {key}")

    # Temporal dynamics plots
    td_dir = BASE_DIR / 'temporal_dynamics'
    if td_dir.exists():
        for img_path in sorted(td_dir.glob('*.png')):
            key = f"temporal_dynamics/{img_path.name}"
            images[key] = image_to_base64(img_path)
            print(f"Loaded: {key}")

    print(f"\nTotal images loaded: {len(images)}")

    # Generate HTML with embedded images
    html = generate_html(images)

    output_path = BASE_DIR / 'spectral_superposition_report_embedded.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nReport generated: {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

def generate_html(images: dict) -> str:
    """Generate the full HTML content with embedded images."""

    def img_tag(key: str, alt: str, caption: str, fig_num: str) -> str:
        if key in images:
            return f'''
<div class="figure-container">
    <img src="{images[key]}" alt="{alt}">
    <div class="figure-caption">
        <span class="figure-number">{fig_num}:</span> {caption}
    </div>
</div>'''
        else:
            return f'<p><em>Image not found: {key}</em></p>'

    # Sample experiment configurations
    samples = [
        ('n1024_m112_s0.000000_seed0', 'm=112, sparsity=0.0 (dense), seed=0'),
        ('n1024_m112_s0.020204_seed0', 'm=112, sparsity=0.02, seed=0'),
        ('n1024_m112_s0.040408_seed0', 'm=112, sparsity=0.04, seed=0'),
        ('n1024_m160_s0.545510_seed1', 'm=160, sparsity=0.55, seed=1'),
        ('n1024_m208_s0.101020_seed0', 'm=208, sparsity=0.10, seed=0'),
        ('n1024_m256_s0.666735_seed0', 'm=256, sparsity=0.67, seed=0'),
        ('n1024_m320_s0.202041_seed1', 'm=320, sparsity=0.20, seed=1'),
        ('n1024_m352_s0.767755_seed1', 'm=352, sparsity=0.77, seed=1'),
        ('n1024_m416_s0.323265_seed0', 'm=416, sparsity=0.32, seed=0'),
        ('n1024_m464_s0.888980_seed0', 'm=464, sparsity=0.89, seed=0'),
        ('n1024_m512_s0.424286_seed1', 'm=512, sparsity=0.42, seed=1'),
        ('n1024_m96_s0.990000_seed1', 'm=96, sparsity=0.99 (extreme), seed=1'),
    ]

    # Generate sample sections
    sample_html = ''
    for idx, (name, config) in enumerate(samples, 1):
        fig_base = f"3.{idx+3}"
        sample_html += f'''
<h3>Sample {idx}: {name}</h3>
<p><em>Configuration: {config}</em></p>

{img_tag(f"dynamic_hopping/plots/{name}_rayleigh_heatmap.png", "Rayleigh Heatmap", f"Rayleigh quotient heatmap ({config})", f"Figure {fig_base}a")}

{img_tag(f"dynamic_hopping/plots/{name}_jump_events.png", "Jump Events", f"Jump events analysis ({config})", f"Figure {fig_base}b")}

{img_tag(f"dynamic_hopping/plots/{name}_volatility_evolution.png", "Volatility Evolution", f"Volatility evolution ({config})", f"Figure {fig_base}c")}
'''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spectral Superposition: Comprehensive Analysis Report</title>
    <style>
        :root {{
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --bg-light: #ecf0f1;
            --text-color: #2c3e50;
            --code-bg: #f8f9fa;
            --border-color: #ddd;
        }}

        * {{ box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.7;
            color: var(--text-color);
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px 40px;
            background-color: #fff;
        }}

        h1 {{
            color: var(--primary-color);
            border-bottom: 4px solid var(--secondary-color);
            padding-bottom: 15px;
            font-size: 2.2em;
            margin-top: 40px;
        }}

        h2 {{
            color: var(--primary-color);
            border-bottom: 2px solid var(--bg-light);
            padding-bottom: 10px;
            margin-top: 40px;
            font-size: 1.6em;
        }}

        h3 {{
            color: var(--secondary-color);
            margin-top: 30px;
            font-size: 1.3em;
        }}

        .title-page {{
            text-align: center;
            padding: 60px 20px;
            border-bottom: 4px double var(--secondary-color);
            margin-bottom: 40px;
        }}

        .title-page h1 {{
            font-size: 2.5em;
            border: none;
            margin-bottom: 20px;
        }}

        .title-page .subtitle {{
            font-size: 1.3em;
            color: #666;
            margin-bottom: 30px;
        }}

        .toc {{
            background-color: var(--bg-light);
            padding: 25px 35px;
            border-radius: 8px;
            margin: 30px 0 40px 0;
        }}

        .toc h2 {{ margin-top: 0; border-bottom: none; }}
        .toc ul {{ list-style: none; padding-left: 0; }}
        .toc li {{ margin: 6px 0; }}
        .toc a {{ color: var(--secondary-color); text-decoration: none; }}
        .toc a:hover {{ text-decoration: underline; }}
        .toc .toc-section {{ font-weight: bold; font-size: 1.05em; margin-top: 12px; }}
        .toc .toc-subsection {{ padding-left: 20px; font-size: 0.9em; }}

        .figure-container {{
            margin: 25px 0;
            text-align: center;
        }}

        .figure-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid var(--border-color);
            border-radius: 4px;
        }}

        .figure-caption {{
            margin-top: 10px;
            font-style: italic;
            color: #666;
            font-size: 0.9em;
        }}

        .figure-number {{ font-weight: bold; color: var(--primary-color); }}

        pre {{
            background-color: var(--code-bg);
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.8em;
            border: 1px solid var(--border-color);
            line-height: 1.4;
        }}

        code {{
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            background-color: var(--code-bg);
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 0.85em;
        }}

        pre code {{ padding: 0; background: none; }}

        .equation {{
            text-align: center;
            margin: 15px 0;
            padding: 12px;
            background-color: var(--bg-light);
            border-radius: 4px;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}

        th, td {{
            border: 1px solid var(--border-color);
            padding: 10px 12px;
            text-align: left;
        }}

        th {{ background-color: var(--bg-light); font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}

        .highlight-box {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12px 15px;
            margin: 15px 0;
            border-radius: 0 4px 4px 0;
        }}

        .definition-box {{
            background-color: #e3f2fd;
            border-left: 4px solid var(--secondary-color);
            padding: 12px 15px;
            margin: 15px 0;
            border-radius: 0 4px 4px 0;
        }}

        .key-finding {{
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 12px 15px;
            margin: 15px 0;
            border-radius: 0 4px 4px 0;
        }}

        .dark-matter-box {{
            background-color: #fce4ec;
            border-left: 4px solid var(--accent-color);
            padding: 12px 15px;
            margin: 15px 0;
            border-radius: 0 4px 4px 0;
        }}

        .section-divider {{
            border: none;
            border-top: 2px solid var(--bg-light);
            margin: 40px 0;
        }}
    </style>
</head>
<body>

<div class="title-page">
    <h1>Spectral Superposition</h1>
    <div class="subtitle">Comprehensive Analysis Report</div>
    <div class="subtitle" style="font-size: 1.1em;">Understanding Dark Matter Features Through Dynamic and Temporal Analysis</div>
    <div style="color: #888;">
        <p><strong>Version:</strong> v1-temporal5 | <strong>Date:</strong> January 2025</p>
    </div>
</div>

<div class="toc">
    <h2>Table of Contents</h2>
    <ul>
        <li class="toc-section"><a href="#sec-1">1. Experimental Setup</a></li>
        <li class="toc-subsection"><a href="#sec-1-1">1.1 Model Architecture</a></li>
        <li class="toc-subsection"><a href="#sec-1-2">1.2 Experimental Grid</a></li>
        <li class="toc-subsection"><a href="#sec-1-3">1.3 Training Configuration</a></li>
        <li class="toc-section"><a href="#sec-2">2. Phase Stratification and Spectral Analysis</a></li>
        <li class="toc-subsection"><a href="#sec-2-1">2.1 Phase Diagram</a></li>
        <li class="toc-subsection"><a href="#sec-2-2">2.2 Slope-Eigenvalue Correlation</a></li>
        <li class="toc-subsection"><a href="#sec-2-3">2.3 The Dark Matter Regime</a></li>
        <li class="toc-section"><a href="#sec-3">3. Dynamic Hopping Analysis</a></li>
        <li class="toc-subsection"><a href="#sec-3-1">3.1 Mathematical Foundation</a></li>
        <li class="toc-subsection"><a href="#sec-3-2">3.2 Analysis Pipeline</a></li>
        <li class="toc-subsection"><a href="#sec-3-3">3.3 Global Results</a></li>
        <li class="toc-subsection"><a href="#sec-3-4">3.4 Sample Experiment Analysis</a></li>
        <li class="toc-section"><a href="#sec-4">4. Temporal Dynamics Analysis</a></li>
        <li class="toc-subsection"><a href="#sec-4-1">4.1 Dark Matter Evolution</a></li>
        <li class="toc-subsection"><a href="#sec-4-2">4.2 Eigenspace Stability</a></li>
        <li class="toc-subsection"><a href="#sec-4-3">4.3 Instantaneous Linearity</a></li>
        <li class="toc-subsection"><a href="#sec-4-4">4.4 Slope-Eigenvalue Temporal</a></li>
        <li class="toc-subsection"><a href="#sec-4-5">4.5 Trajectory Classification</a></li>
    </ul>
</div>

<hr class="section-divider">

<h1 id="sec-1">1. Experimental Setup</h1>

<p>This report documents a large-scale parameter sweep for the Toy Models of Superposition experiment, studying how neural networks learn to represent more features than they have dimensions (superposition).</p>

<h2 id="sec-1-1">1.1 Model Architecture</h2>

<div class="definition-box">
<p><strong>Model:</strong> ReLU output autoencoder</p>
</div>

<div class="equation">
    <code>h = W @ x</code> (Projection: R<sup>n</sup> → R<sup>m</sup>)<br>
    <code>x' = ReLU(W<sup>T</sup> @ h + b)</code> (Reconstruction: R<sup>m</sup> → R<sup>n</sup>)<br>
    <code>Loss = MSE(x, x')</code>
</div>

<p><strong>Input distribution:</strong> <code>x_i = 0</code> with probability <code>S</code> (sparsity), else <code>x_i ~ Uniform(0,1)</code></p>

<h2 id="sec-1-2">1.2 Experimental Grid</h2>

<table>
    <tr><th>Parameter</th><th>Values</th><th>Description</th></tr>
    <tr><td>N_FEATURES</td><td>1024</td><td>Number of input features</td></tr>
    <tr><td>M_VALUES</td><td>16 to 512 (32 values)</td><td>Hidden dimension</td></tr>
    <tr><td>S_VALUES</td><td>0.0 to 0.99 (50 values)</td><td>Sparsity levels</td></tr>
    <tr><td>SEEDS</td><td>[0, 1]</td><td>Random seeds</td></tr>
</table>

<p><strong>Total experiments:</strong> 32 x 50 x 2 = <strong>3,200 experiments</strong></p>

<h2 id="sec-1-3">1.3 Training Configuration</h2>

<table>
    <tr><th>Parameter</th><th>Value</th></tr>
    <tr><td>Total Steps</td><td>25,000</td></tr>
    <tr><td>Batch Size</td><td>1,024</td></tr>
    <tr><td>Learning Rate</td><td>1e-3</td></tr>
    <tr><td>Optimizer</td><td>Adam</td></tr>
    <tr><td>Checkpoints</td><td>56 (adaptive schedule)</td></tr>
</table>

<p><strong>Compute:</strong> 8x NVIDIA L4 GPUs (24GB each) on GCP</p>

<hr class="section-divider">

<h1 id="sec-2">2. Phase Stratification and Spectral Analysis</h1>

<h2 id="sec-2-1">2.1 Phase Diagram</h2>

<div class="definition-box">
<p><strong>Fractional Dimensionality:</strong> D<sub>i</sub> = (M<sub>ii</sub>)<sup>2</sup> / (M<sup>2</sup>)<sub>ii</sub> where M = W<sup>T</sup>W</p>
</div>

{img_tag('phase_diagram_sparsity.png', 'Phase Diagram', 'Phase diagram showing D_i vs ||W_i||^2. Features cluster along distinct rays with slopes corresponding to inverse eigenvalues.', 'Figure 2.1')}

<h2 id="sec-2-2">2.2 Slope-Eigenvalue Correlation</h2>

<div class="key-finding">
<p><strong>Key Conjecture:</strong> The linear slope of each ray corresponds to 1/lambda of the frame operator S = WW<sup>T</sup>.</p>
</div>

{img_tag('slope_eigenvalue_correlation.png', 'Slope-Eigenvalue Correlation', 'For lambda > 1, kappa = 1/lambda holds with R^2 = 0.94, Pearson r = 0.97.', 'Figure 2.2')}

{img_tag('slope_eigenvalue_summary.png', 'Slope-Eigenvalue Summary', 'Summary showing breakdown of spectral interpretation for lambda < 1.', 'Figure 2.3')}

<h2 id="sec-2-3">2.3 The Dark Matter Regime</h2>

<div class="dark-matter-box">
<p><strong>Dark Matter Features:</strong> For lambda < 1, where majority of features lie, the spectral interpretation breaks down.</p>
<ul>
    <li>73-84% of features have R^2 < 0.9 for D_i vs ||W_i||^2</li>
    <li>Lower eigenspace concentration (spread across multiple eigenspaces)</li>
    <li>Higher eigenspace entropy</li>
    <li>Negative curvature in D_i(t) vs ||W_i(t)||^2 trajectories</li>
</ul>
</div>

<hr class="section-divider">

<h1 id="sec-3">3. Dynamic Hopping Analysis</h1>

<p><strong>Goal:</strong> Quantify "dynamic hopping" via time-variation of the Rayleigh quotient.</p>

<h2 id="sec-3-1">3.1 Mathematical Foundation</h2>

<div class="equation">
    kappa_i(t) = (w_i(t)<sup>T</sup> S(t) w_i(t)) / ||w_i(t)||<sup>2</sup>
</div>

<p>where S(t) = W(t) W(t)<sup>T</sup> is the feature covariance matrix.</p>

<p><strong>Jump Detection:</strong> Using robust statistics with MAD (Median Absolute Deviation):</p>
<ul>
    <li>sigma_robust = 1.4826 x median(|Delta_x - median(Delta_x)|)</li>
    <li>Jump threshold: |Delta_x_i(t)| > z x sigma_robust (z = 4.0)</li>
</ul>

<h2 id="sec-3-2">3.2 Analysis Pipeline</h2>

<table>
    <tr><th>Script</th><th>Purpose</th></tr>
    <tr><td>01_rayleigh_quotients.py</td><td>Core Rayleigh quotient computation</td></tr>
    <tr><td>02_jump_detection.py</td><td>Detect jumps using robust statistics</td></tr>
    <tr><td>03_temporal_patterns.py</td><td>Analyze temporal structure</td></tr>
    <tr><td>04_visualizations.py</td><td>Generate all plots</td></tr>
</table>

<h2 id="sec-3-3">3.3 Global Results</h2>

{img_tag('dynamic_hopping/plots/dynamic_hopping_summary.png', 'Dynamic Hopping Summary', 'Comprehensive summary: (Row 1) Jump frequency, volatility, trend; (Row 2) Synchrony, volatility classes, temporal patterns; (Row 3) Scatter analysis.', 'Figure 3.1')}

{img_tag('dynamic_hopping/plots/feature_classification_summary.png', 'Feature Classification', 'Volatility class distribution, temporal patterns, synchrony ratios, and trend directions.', 'Figure 3.2')}

{img_tag('dynamic_hopping/plots/sparsity_comparison.png', 'Sparsity Comparison', 'Hopping metrics across sparsity: total jumps, global sigma, late/early ratio, jumps vs volatility.', 'Figure 3.3')}

<h2 id="sec-3-4">3.4 Sample Experiment Analysis</h2>

<p>Detailed analysis for 12 sample experiments across different configurations. Each includes: Rayleigh Heatmap, Jump Events, Volatility Evolution.</p>

{sample_html}

<hr class="section-divider">

<h1 id="sec-4">4. Temporal Dynamics Analysis</h1>

<div class="highlight-box">
<p><strong>Key Question:</strong> Is dark matter transient (features "in transit" between eigenspaces) or persistent?</p>
</div>

<h2 id="sec-4-1">4.1 Dark Matter Evolution</h2>

<p>Track dark matter fraction (R^2 < 0.9) as a function of training step.</p>

{img_tag('temporal_dynamics/temporal_dark_matter_evolution.png', 'Dark Matter Evolution', 'Cumulative/windowed dark matter fraction over training; feature transition statistics.', 'Figure 4.1')}

<h2 id="sec-4-2">4.2 Eigenspace Stability</h2>

<p>Track feature migration between eigenspace clusters during training.</p>

{img_tag('temporal_dynamics/temporal_eigenspace_stability.png', 'Eigenspace Stability', 'Hopping rate, concentration, regime comparison (lambda>1 vs lambda<=1), hop-R^2 correlation.', 'Figure 4.2')}

<h2 id="sec-4-3">4.3 Instantaneous Linearity</h2>

<p>Compare cumulative R^2 vs instantaneous slope to determine if poor R^2 is due to slope changes or intrinsic nonlinearity.</p>

{img_tag('temporal_dynamics/temporal_instantaneous_linearity.png', 'Instantaneous Linearity', 'R^2 by slope stability, convergence metrics, sample slope trajectories, phase space paths.', 'Figure 4.3')}

<h2 id="sec-4-4">4.4 Slope-Eigenvalue Temporal Relationship</h2>

<p>Track kappa = 1/lambda conjecture at each checkpoint.</p>

{img_tag('temporal_dynamics/temporal_slope_eigenvalue_temporal.png', 'Slope-Eigenvalue Temporal', 'Mean kappa*lambda over training, regime comparison, distribution evolution, final values.', 'Figure 4.4')}

<h2 id="sec-4-5">4.5 Trajectory Classification</h2>

<p>Classify features by trajectory shape: linear, curved, oscillatory, transient, collapsed.</p>

{img_tag('temporal_dynamics/temporal_trajectory_classification.png', 'Trajectory Classification', 'Distribution by sparsity, linear vs non-linear, sample trajectories, R^2 by type.', 'Figure 4.5')}

<hr class="section-divider">

<div style="text-align: center; margin-top: 50px; padding: 25px; background-color: var(--bg-light); border-radius: 8px;">
    <h2 style="border: none; margin-top: 0;">End of Report</h2>
    <p style="color: #888; font-size: 0.9em;">Spectral Superposition Project v1-temporal5 | January 2025</p>
</div>

</body>
</html>'''

    return html

if __name__ == '__main__':
    generate_report()
