# banking-document-analyzer
Financial Document Analyzer
An AI-powered pipeline that reads banking/financial reports (10-Ks, quarterly earnings reports, annual reports) and produces structured output: extracted KPIs, flagged risk factors, and an executive summary.

Handles both text-based PDFs and scanned/image PDFs (via OCR fallback), splits long documents into chunks, runs extraction per chunk through an LLM, then merges and deduplicates the results into one structured report.

Features
Pluggable provider layer (analyzer/providers/) — OpenAI, Anthropic, and Google out of the box; add another by subclassing BaseProvider.
Robust ingestion — text extraction via pdfplumber with a pypdf fallback, and automatic OCR (via pytesseract + pdf2image) for pages with little or no extractable text (scanned documents).
Chunking — long reports are split into overlapping chunks so extraction stays within model context limits; results are merged and deduplicated afterward.
Structured extraction — KPIs (name, value, unit, period, category, confidence), risk factors (type, description, severity, evidence quote), and an executive summary, all validated against a defined schema.
Reports — writes analysis.json (full structured data) and analysis.md (human-readable report with KPI tables and risks grouped by severity).
Sample data included — a synthetic bank report (sample_data/) so you can run the pipeline immediately without a real document.
