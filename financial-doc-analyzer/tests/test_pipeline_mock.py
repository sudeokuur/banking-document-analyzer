"""End-to-end pipeline test against the real sample PDF, with a fake LLM
provider standing in for real API calls (so this test runs with no API keys
and no network access, in CI or locally).
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyzer.providers as providers
from analyzer.providers.base import GenerationResult
from analyzer.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = REPO_ROOT / "sample_data" / "sample_bank_report.pdf"

FAKE_KPI_RESPONSE = (
    '[{"name": "Return on Equity", "value": "11.4%", "unit": "%", "period": "Q3 2025", '
    '"category": "profitability", "confidence": 0.9}, '
    '{"name": "Net Interest Margin", "value": "3.42%", "unit": "%", "period": "Q3 2025", '
    '"category": "profitability", "confidence": 0.85}]'
)
FAKE_RISK_RESPONSE = (
    '[{"risk_type": "credit_risk", "description": "Commercial real estate concentration in the office sector", '
    '"severity": "high", "evidence_quote": "34% of the bank\'s commercial loan portfolio", "section": "Risk Factors"}]'
)
FAKE_SUMMARY_RESPONSE = (
    '{"overview": "Northbridge posted solid but moderating profitability with rising CRE-related risk.", '
    '"key_takeaways": ["ROE improved to 11.4%", "CRE office concentration is a key watch item"], '
    '"overall_risk_level": "medium"}'
)


class FakeProvider:
    name = "fake"

    def generate(self, prompt, model, temperature=0.1, max_tokens=4096, system=None, **kwargs):
        # Match on phrases unique to each prompt template's opening line --
        # NOT the generic substring "risk factor", which also appears in the
        # summarizer prompt's "## Extracted risk factors" section header and
        # would otherwise misroute summary calls to the risk-extraction reply.
        lowered = prompt.lower()
        if "extract every financial kpi" in lowered:
            text = FAKE_KPI_RESPONSE
        elif "identify risk factors discussed" in lowered:
            text = FAKE_RISK_RESPONSE
        elif "write an executive summary" in lowered:
            text = FAKE_SUMMARY_RESPONSE
        else:
            raise AssertionError(f"FakeProvider received an unrecognized prompt: {prompt[:80]!r}")
        return GenerationResult(text=text, model=model, provider=self.name, latency_seconds=0.001, input_tokens=10, output_tokens=10)


@pytest.fixture(autouse=True)
def ensure_sample_pdf_and_fake_provider():
    if not SAMPLE_PDF.exists():
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "generate_sample_pdf.py")], check=True)
    for key in ("openai", "anthropic", "google"):
        providers._instances[key] = FakeProvider()
    yield
    providers._instances.clear()


def _config():
    return {
        "provider": {"name": "anthropic", "model": "fake-model", "temperature": 0.1, "max_tokens": 4096},
        "chunking": {"max_chars_per_chunk": 12000, "overlap_chars": 500},
        "ocr": {"min_chars_for_text_page": 40, "dpi": 150},
        "kpi_categories": ["profitability", "capital_adequacy", "asset_quality", "liquidity", "growth", "other"],
        "risk_categories": ["credit_risk", "market_risk", "cybersecurity", "other"],
    }


def test_pipeline_produces_kpis_risks_and_summary():
    result = run_pipeline(SAMPLE_PDF, _config(), workers=2)

    assert len(result.kpis) >= 1
    assert any(k.name == "Return on Equity" for k in result.kpis)
    assert len(result.risks) >= 1
    assert result.risks[0].severity in ("low", "medium", "high", "critical")
    assert result.summary.overview
    assert result.summary.overall_risk_level == "medium"
    assert result.metadata["total_pages"] == 2


def test_pipeline_dedups_kpis_across_chunks_when_document_is_chunked():
    config = _config()
    config["chunking"]["max_chars_per_chunk"] = 1000  # force multiple chunks
    config["chunking"]["overlap_chars"] = 100
    result = run_pipeline(SAMPLE_PDF, config, workers=2)

    # Even though every chunk returns the same fake KPI list, merge_kpis should
    # dedup by (name, period) rather than duplicating per chunk.
    roe_matches = [k for k in result.kpis if k.name == "Return on Equity"]
    assert len(roe_matches) == 1
