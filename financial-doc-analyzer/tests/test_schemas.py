"""Unit tests for schema parsing and KPI/risk merge-dedup logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.schemas import KPI, RiskFactor, merge_kpis, merge_risks


def test_kpi_from_dict_parses_numeric_value():
    kpi = KPI.from_dict({"name": "Return on Equity", "value": "11.4%", "period": "Q3 2025", "category": "profitability"})
    assert kpi.numeric_value == 11.4
    assert kpi.category == "profitability"


def test_kpi_from_dict_requires_name():
    try:
        KPI.from_dict({"value": "10%"})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_kpi_from_dict_clamps_confidence():
    kpi = KPI.from_dict({"name": "NIM", "value": "3.4%", "confidence": 5})
    assert kpi.confidence == 1.0
    kpi2 = KPI.from_dict({"name": "NIM", "value": "3.4%", "confidence": -1})
    assert kpi2.confidence == 0.0


def test_risk_from_dict_normalizes_severity():
    r = RiskFactor.from_dict({"risk_type": "Credit_Risk", "description": "CRE concentration", "severity": "SEVERE"})
    assert r.severity == "medium"  # invalid severity falls back to medium
    r2 = RiskFactor.from_dict({"risk_type": "credit_risk", "description": "x", "severity": "high"})
    assert r2.severity == "high"


def test_merge_kpis_dedups_same_name_and_period_keeps_highest_confidence():
    kpis = [
        KPI(name="Net Interest Margin", value="3.4%", period="Q3 2025", confidence=0.6, source_pages=[1]),
        KPI(name="net interest margin", value="3.42%", period="Q3 2025", confidence=0.9, source_pages=[2]),
    ]
    merged = merge_kpis(kpis)
    assert len(merged) == 1
    assert merged[0].value == "3.42%"
    assert merged[0].source_pages == [1, 2]


def test_merge_kpis_keeps_distinct_periods_separate():
    kpis = [
        KPI(name="ROE", value="11%", period="Q2 2025", confidence=0.8, source_pages=[1]),
        KPI(name="ROE", value="11.4%", period="Q3 2025", confidence=0.8, source_pages=[1]),
    ]
    merged = merge_kpis(kpis)
    assert len(merged) == 2


def test_merge_risks_dedups_similar_descriptions_and_keeps_higher_severity():
    risks = [
        RiskFactor(risk_type="credit_risk", description="Commercial real estate concentration is elevated in the office sector", severity="medium", source_pages=[3]),
        RiskFactor(risk_type="credit_risk", description="Commercial real estate concentration in the office sector could worsen", severity="high", source_pages=[3, 4]),
    ]
    merged = merge_risks(risks)
    assert len(merged) == 1
    assert merged[0].severity == "high"
    assert merged[0].source_pages == [3, 4]


def test_merge_risks_keeps_distinct_risk_types_separate():
    risks = [
        RiskFactor(risk_type="credit_risk", description="CRE concentration", severity="high"),
        RiskFactor(risk_type="cybersecurity", description="Phishing and ransomware exposure", severity="medium"),
    ]
    merged = merge_risks(risks)
    assert len(merged) == 2
    # sorted with highest severity first
    assert merged[0].severity == "high"
