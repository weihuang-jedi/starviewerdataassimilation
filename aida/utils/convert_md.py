#!/usr/bin/env python3
"""
HPC-Compatible Markdown to DOCX & PDF Converter.
Requires 'pandoc' (via conda) and optionally 'weasyprint' for PDF generation.
"""

import sys
import shutil
from pathlib import Path
import pypandoc


def check_dependencies():
    """Verify system/conda binaries for Pandoc and PDF engines."""
    pandoc_path = shutil.which("pandoc")
    if not pandoc_path:
        print("[ERROR] 'pandoc' executable not found in PATH!")
        print("  Fix: Run 'conda install -c conda-forge pandoc'")
        sys.exit(1)
    else:
        print(f"[FOUND] Pandoc binary: {pandoc_path}")


def convert_markdown(
    input_md_path: str,
    output_docx_path: str = None,
    output_pdf_path: str = None,
    pdf_engine: str = "weasyprint"
):
    """
    Converts Markdown files containing LaTeX math equations to DOCX and PDF.
    """
    check_dependencies()

    input_path = Path(input_md_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Source file not found: {input_md_path}")

    if output_docx_path is None:
        output_docx_path = input_path.with_suffix(".docx")
    if output_pdf_path is None:
        output_pdf_path = input_path.with_suffix(".pdf")

    extra_args = ["--standalone", "--mathjax"]

    # --- 1. Convert to DOCX ---
    print(f"\n[CONVERTING] -> {output_docx_path} ...")
    try:
        pypandoc.convert_file(
            source_file=str(input_path),
            to="docx",
            outputfile=str(output_docx_path),
            extra_args=extra_args
        )
        print(f"[SUCCESS] Created DOCX: {output_docx_path}")
    except Exception as e:
        print(f"[ERROR] Failed to generate DOCX: {e}")

    # --- 2. Convert to PDF ---
    print(f"[CONVERTING] -> {output_pdf_path} using engine '{pdf_engine}' ...")
    
    # Verify if chosen PDF engine exists in environment
    if not shutil.which(pdf_engine):
        print(f"[WARNING] PDF engine '{pdf_engine}' not found in PATH.")
        print(f"  Fix: Run 'pip install weasyprint' or 'conda install -c conda-forge xelatex'")
        return

    pdf_extra_args = extra_args + [f"--pdf-engine={pdf_engine}"]
    try:
        pypandoc.convert_file(
            source_file=str(input_path),
            to="pdf",
            outputfile=str(output_pdf_path),
            extra_args=pdf_extra_args
        )
        print(f"[SUCCESS] Created PDF: {output_pdf_path}")
    except Exception as e:
        print(f"[ERROR] Failed to generate PDF: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        md_file = sys.argv[1]
    else:
        md_file = "sample_response.md"
        sample_content = """# Atmospheric State Summary

The ideal gas law in log-state space is expressed as:

$$\\text{ln\\_p} = \\text{ln\\_rho} + \\ln(R_d) + \\text{ln\\_t}$$

Where:
* $R_d = 287.058 \\text{ J/(kg K)}$
* $\\text{ln\\_p}$ is the natural log of pressure $p$.

| Variable | Description |
| :--- | :--- |
| `ln_t` | Log Temperature |
| `ln_p` | Log Pressure |
| `ln_rho` | Log Density |
"""
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(sample_content)
        print(f"[INFO] Created sample markdown file: {md_file}")

    convert_markdown(md_file)
