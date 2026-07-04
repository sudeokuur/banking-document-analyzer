#!/usr/bin/env python
"""CLI entrypoint: analyze a banking/financial PDF report and write structured reports.

Usage:
    python scripts/analyze_report.py --input sample_data/sample_bank_report.pdf --output outputs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/analyze_report.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from dotenv import load_dotenv

from analyzer.pipeline import run_pipeline
from analyzer.report import generate_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a banking/financial PDF report.")
    parser.add_argument("--input", required=True, help="Path to the PDF report to analyze.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--output", default="outputs/", help="Output directory for analysis.json / analysis.md.")
    parser.add_argument("--force-ocr", action="store_true", help="Run every page through OCR, not just scanned-looking ones.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel chunk-extraction threads.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Analyzing {args.input} with {config['provider']['name']}/{config['provider']['model']}...", file=sys.stderr)

    result = run_pipeline(
        pdf_path=args.input,
        config=config,
        force_ocr=args.force_ocr,
        workers=args.workers,
    )

    paths = generate_reports(result, args.output)

    print("\nDone. Reports written to:", file=sys.stderr)
    print(f"  {paths['json']}", file=sys.stderr)
    print(f"  {paths['markdown']}", file=sys.stderr)
    print(file=sys.stderr)
    print(f"KPIs extracted: {len(result.kpis)}", file=sys.stderr)
    print(f"Risks identified: {len(result.risks)}", file=sys.stderr)
    print(f"Overall risk level: {result.summary.overall_risk_level}", file=sys.stderr)
    if result.metadata.get("errors"):
        print(f"\n{len(result.metadata['errors'])} warning(s) -- see analysis.md for details.", file=sys.stderr)


if __name__ == "__main__":
    main()
