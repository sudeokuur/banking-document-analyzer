# Financial Document Analyzer

An AI-powered pipeline that reads banking/financial reports (10-Ks, quarterly earnings reports, annual reports) and produces structured output: extracted KPIs, flagged risk factors, and an executive summary.

Handles both text-based PDFs and scanned/image PDFs (via OCR fallback), splits long documents into chunks, runs extraction per chunk through an LLM, then merges and deduplicates the results into one structured report.

## Features

- **Pluggable provider layer** (`analyzer/providers/`) — OpenAI, Anthropic, and Google out of the box; add another by subclassing `BaseProvider`.
- **Robust ingestion** — text extraction via `pdfplumber` with a `pypdf` fallback, and automatic OCR (via `pytesseract` + `pdf2image`) for pages with little or no extractable text (scanned documents).
- **Chunking** — long reports are split into overlapping chunks so extraction stays within model context limits; results are merged and deduplicated afterward.
- **Structured extraction** — KPIs (name, value, unit, period, category, confidence), risk factors (type, description, severity, evidence quote), and an executive summary, all validated against a defined schema.
- **Reports** — writes `analysis.json` (full structured data) and `analysis.md` (human-readable report with KPI tables and risks grouped by severity).
- **Sample data included** — a synthetic bank report (`sample_data/`) so you can run the pipeline immediately without a real document.

## Project layout

```
financial-doc-analyzer/
├── analyzer/
│   ├── providers/           # OpenAI, Anthropic, Google integrations
│   ├── ingestion/           # PDF text extraction, OCR fallback, chunking
│   ├── extraction/          # KPI extractor, risk extractor, summarizer
│   ├── schemas.py           # KPI / RiskFactor / ExecutiveSummary / AnalysisResult
│   ├── pipeline.py          # orchestrates ingest -> chunk -> extract -> merge -> summarize
│   ├── report.py            # writes JSON + Markdown reports
│   └── utils.py             # retry decorator, JSON-from-LLM-output parsing
├── config/
│   └── config.yaml          # provider/model settings, KPI + risk taxonomy
├── sample_data/
│   ├── sample_bank_report.txt
│   └── sample_bank_report.pdf
├── scripts/
│   ├── analyze_report.py    # CLI entrypoint
│   └── generate_sample_pdf.py
├── outputs/                 # analysis.json / analysis.md land here
├── tests/
└── requirements.txt
```

## Setup

```bash
git clone <this-repo>
cd financial-doc-analyzer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API key(s)
```

Required environment variables (in `.env`):

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

You only need a key for the provider you configure in `config/config.yaml`.

### OCR prerequisites (system packages, not pip)

OCR fallback (for scanned PDFs) needs Tesseract and Poppler installed on your system:

```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils
```

If you never expect scanned documents, you can skip this — the pipeline only invokes OCR on pages where direct text extraction comes back empty/near-empty.

## Configuring the model

Edit `config/config.yaml`:

```yaml
provider:
  name: anthropic          # openai | anthropic | google
  model: claude-sonnet-5
  temperature: 0.1
  max_tokens: 4096

chunking:
  max_chars_per_chunk: 12000
  overlap_chars: 500
```

The KPI and risk taxonomies in the same file are passed into the extraction prompts as guidance — the model isn't restricted to only those categories, but it groups the metrics/risks it finds under them where they fit, and uses `"other"` otherwise.

## Running it

```bash
python scripts/analyze_report.py --input sample_data/sample_bank_report.pdf --output outputs/
```

Options:

- `--input path/to/report.pdf` — the document to analyze (required)
- `--config config/config.yaml` — path to config file (default shown)
- `--output outputs/` — directory to write `analysis.json` and `analysis.md`
- `--force-ocr` — run every page through OCR instead of only pages with little/no extractable text
- `--workers 4` — parallel chunk-extraction threads

This produces `outputs/analysis.json` (machine-readable) and `outputs/analysis.md` (a formatted report with a KPI table and risks grouped by severity, high/critical first).

## Extending

- **New provider**: subclass `BaseProvider` in `analyzer/providers/base.py`, implement `_call(...)`, and register it in `analyzer/providers/__init__.py`.
- **New KPI/risk categories**: edit the taxonomy lists in `config/config.yaml` — no code changes needed, they're interpolated into the extraction prompts.
- **Different chunking strategy**: `analyzer/ingestion/chunker.py` currently does simple character-based chunking with overlap; swap in a token-aware or section-aware splitter if your documents have exploitable structure (e.g., splitting on report section headers).

## How merging works

Each chunk is processed independently (in parallel) for KPIs and risks. Afterward:

- **KPIs** are deduplicated by normalized name + period; when the same metric appears in multiple chunks, the highest-confidence extraction wins.
- **Risks** are deduplicated by normalized risk type + description similarity (simple token-overlap threshold); duplicates are merged, keeping the higher severity.
- The **executive summary** is generated last, from the merged KPI/risk data (not the raw document text), to keep it grounded in what was actually extracted and to keep token usage down on large documents.

## Cost, accuracy, and safety notes

This pipeline makes real LLM API calls — cost scales with document length (number of chunks) and model choice. LLMs can misread numbers or hallucinate figures, especially from OCR'd text with recognition errors; treat extracted KPIs as a first pass that should be spot-checked against the source document before being used for any financial decision, disclosure, or compliance purpose. This tool does not constitute financial, investment, or accounting advice.

## License

MIT — see `LICENSE`.
