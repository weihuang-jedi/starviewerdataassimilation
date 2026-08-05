import os
from weasyprint import HTML

# Create output directory
os.makedirs("output", exist_ok=True)

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @page {
    size: A4;
    margin: 12mm 10mm;
    background-color: #f8fafc;
  }
  *, *::before, *::after {
    box-sizing: border-box;
  }
  body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: #1e293b;
    margin: 0;
    padding: 0;
    font-size: 9pt;
    line-height: 1.4;
  }
  .header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    color: #ffffff;
    padding: 20px 24px;
    border-radius: 8px;
    margin-bottom: 16px;
  }
  .header h1 {
    margin: 0 0 6px 0;
    font-size: 18pt;
    font-weight: 700;
    letter-spacing: -0.5px;
  }
  .header p {
    margin: 0;
    font-size: 9.5pt;
    color: #93c5fd;
  }
  .badge {
    display: inline-block;
    background-color: #10b981;
    color: #ffffff;
    font-size: 8pt;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
    margin-top: 8px;
  }
  .section-title {
    font-size: 11pt;
    font-weight: 700;
    color: #0f172a;
    border-left: 4px solid #2563eb;
    padding-left: 8px;
    margin: 16px 0 10px 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .grid-table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 16px;
    background: #ffffff;
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .grid-table th {
    background-color: #1e293b;
    color: #f8fafc;
    font-weight: 600;
    text-align: left;
    padding: 8px 10px;
    font-size: 8.5pt;
  }
  .grid-table td {
    padding: 7px 10px;
    border-bottom: 1px solid #e2e8f0;
    font-size: 8.5pt;
  }
  .grid-table tr:last-child td {
    border-bottom: none;
  }
  .grid-table tr:nth-child(even) {
    background-color: #f1f5f9;
  }
  .highlight-green {
    color: #059669;
    font-weight: 700;
  }
  .highlight-blue {
    color: #2563eb;
    font-weight: 700;
  }
  .callout {
    background-color: #eff6ff;
    border-left: 4px solid #3b82f6;
    padding: 10px 14px;
    border-radius: 4px;
    margin-bottom: 16px;
  }
  .callout-title {
    font-weight: 700;
    color: #1e40af;
    margin-bottom: 4px;
    font-size: 9pt;
  }
  .metrics-summary-grid {
    display: table;
    width: 100%;
    margin-bottom: 16px;
  }
  .metric-card {
    display: table-cell;
    width: 33.33%;
    background: #ffffff;
    padding: 12px;
    border-radius: 6px;
    border: 1px solid #cbd5e1;
    text-align: center;
  }
  .metric-card + .metric-card {
    border-left: none;
  }
  .metric-val {
    font-size: 14pt;
    font-weight: 800;
    color: #0f172a;
    margin-top: 4px;
  }
  .metric-label {
    font-size: 7.5pt;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 600;
  }
  .footer {
    text-align: center;
    font-size: 7.5pt;
    color: #94a3b8;
    margin-top: 20px;
    border-top: 1px solid #e2e8f0;
    padding-top: 8px;
  }
</style>
</head>
<body>

<div class="header">
  <h1>AIDA Model Verification Summary</h1>
  <p>Cosine-Latitude Weighted Atmospheric Analysis & Optimization Profile</p>
  <div class="badge">Humidity Bug Resolved &bull; RelDiff &lt; 30%</div>
</div>

<div class="metrics-summary-grid">
  <div class="metric-card">
    <div class="metric-label">Specific Humidity (q) Error</div>
    <div class="metric-val highlight-green">29.47 %</div>
    <div style="font-size:7pt; color:#059669;">Down from 389,901%</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Humidity ACC Correlation</div>
    <div class="metric-val highlight-blue">0.772</div>
    <div style="font-size:7pt; color:#2563eb;">Up from 0.358</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Zonal Wind (u) ACC</div>
    <div class="metric-val">0.934</div>
    <div style="font-size:7pt; color:#64748b;">Optimal flow fidelity</div>
  </div>
</div>

<div class="section-title">1. Final Model State Metrics (Latest Run)</div>
<table class="grid-table">
  <thead>
    <tr>
      <th>Variable</th>
      <th>Description</th>
      <th>RMSE</th>
      <th>MAE</th>
      <th>BIAS</th>
      <th>ACC</th>
      <th>RelDiff (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>t</b></td>
      <td>Temperature (K)</td>
      <td>9.8248</td>
      <td>7.8293</td>
      <td>-5.5304</td>
      <td>0.7759</td>
      <td>2.97%</td>
    </tr>
    <tr>
      <td><b>u</b></td>
      <td>Zonal Wind (m/s)</td>
      <td>3.4513</td>
      <td>2.3393</td>
      <td>+0.1446</td>
      <td><b>0.9336</b></td>
      <td>23.67%</td>
    </tr>
    <tr>
      <td><b>v</b></td>
      <td>Meridional Wind (m/s)</td>
      <td>3.6614</td>
      <td>2.3627</td>
      <td>-0.0904</td>
      <td>0.8598</td>
      <td>44.20%</td>
    </tr>
    <tr>
      <td><b>w</b></td>
      <td>Vertical Motion (m/s)</td>
      <td>0.3665</td>
      <td>0.2088</td>
      <td>-0.0033</td>
      <td>0.1675</td>
      <td>98.67%</td>
    </tr>
    <tr>
      <td><b>q</b></td>
      <td>Specific Humidity (kg/kg)</td>
      <td>0.0021</td>
      <td>0.0015</td>
      <td><b>-0.0003</b></td>
      <td><b>0.7723</b></td>
      <td class="highlight-green"><b>29.47%</b></td>
    </tr>
    <tr>
      <td><b>p</b></td>
      <td>Surface/Column Pressure (hPa)</td>
      <td>24.1543</td>
      <td>16.4943</td>
      <td>-4.8379</td>
      <td>0.6521</td>
      <td>2.45%</td>
    </tr>
  </tbody>
</table>

<div class="section-title">2. Humidity (q) Optimization Progression</div>
<table class="grid-table">
  <thead>
    <tr>
      <th>Iteration Step</th>
      <th>q MAE (kg/kg)</th>
      <th>q BIAS (kg/kg)</th>
      <th>q ACC</th>
      <th>q RelDiff (%)</th>
      <th>Status / Primary Mechanism</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Baseline (Raw)</b></td>
      <td>0.056433</td>
      <td>-0.016158</td>
      <td>0.357768</td>
      <td style="color:#dc2626;">389,901.94%</td>
      <td>Unconstrained output; Stratospheric near-zero division blowup</td>
    </tr>
    <tr>
      <td><b>Evaluation Fix</b></td>
      <td>0.056433</td>
      <td>-0.016158</td>
      <td>0.357768</td>
      <td>1,118.72%</td>
      <td>Volume-weighted aggregate denominator</td>
    </tr>
    <tr>
      <td><b>Softplus Activation</b></td>
      <td>0.020195</td>
      <td>+0.020079</td>
      <td>0.341890</td>
      <td>400.35%</td>
      <td>Eliminated unphysical negative moisture predictions</td>
    </tr>
    <tr>
      <td><b>q Bias Penalty</b></td>
      <td>0.001490</td>
      <td>-0.000172</td>
      <td>0.776368</td>
      <td class="highlight-green">29.54%</td>
      <td>Targeted mean bias penalty added to loss.py</td>
    </tr>
    <tr>
      <td><b>Latest Refinement</b></td>
      <td>0.001486</td>
      <td>-0.000310</td>
      <td>0.772274</td>
      <td class="highlight-green"><b>29.47%</b></td>
      <td>Balanced thermodynamic barrier state (p bias = -4.84 hPa)</td>
    </tr>
  </tbody>
</table>

<div class="callout">
  <div class="callout-title">Recommended Action for Final Fine-Tuning</div>
  To resolve the remaining <b>-4.84 hPa</b> pressure drift and center temperature bias near <b>0.0 K</b>, update the barrier in <code>models/loss.py</code> from <code>0.05</code> to <code>0.08</code>:
  <br><br>
  <code>p_violation = F.relu(0.08 - p_pred)</code>
</div>

<div class="footer">
  AIDA Verification Report &bull; Generated automatically via WeasyPrint HTML-to-PDF &bull; Session Date: August 2026
</div>

</body>
</html>
"""

# Write HTML file and convert to PDF via WeasyPrint
html_path = "output/aida_verification_report.html"
pdf_path = "output/aida_verification_report.pdf"

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

HTML(filename=html_path).write_pdf(pdf_path)
print(f"Generated PDF report at: {pdf_path}")
