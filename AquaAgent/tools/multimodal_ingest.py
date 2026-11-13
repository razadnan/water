"""
Multi-Modal Document & Telemetry Ingestion Parser.
Extracts structured water telemetry from:
- CSV & TSV data files
- PDF audit reports & utility bills (extracts tables and key-value readings)
- DOCX operational logs & work orders (extracts tables and textual metrics)
- TXT telemetry dumps & SCADA ASCII logs
- Images (bills, meter photos, sensor dashboard screenshots via OCR/Vision)

Transforms arbitrary unstructured document data into clean normalized DataFrame
format compatible with AquaAgent's water_tools, anomaly detection, and agent pipeline.
"""
import io
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd


def _parse_tabular_or_csv(content_bytes: bytes) -> pd.DataFrame:
    """Attempts direct CSV, TSV, or whitespace-delimited parsing."""
    try:
        df = pd.read_csv(io.BytesIO(content_bytes))
        if len(df.columns) >= 2:
            return df
    except Exception:
        pass

    try:
        df = pd.read_csv(io.BytesIO(content_bytes), sep=r"\s+")
        if len(df.columns) >= 2:
            return df
    except Exception:
        pass

    return pd.DataFrame()


def _extract_text_from_pdf(content_bytes: bytes) -> str:
    """Extracts raw text from PDF documents using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(content_bytes))
    text_parts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text_parts.append(t)
    return "\n".join(text_parts)


def _extract_text_from_docx(content_bytes: bytes) -> str:
    """Extracts text and table rows from DOCX documents."""
    import docx
    doc = docx.Document(io.BytesIO(content_bytes))
    lines = []
    for p in doc.paragraphs:
        if p.text.strip():
            lines.append(p.text.strip())
    for table in doc.tables:
        for row in table.rows:
            row_text = ",".join(cell.text.strip() for cell in row.cells)
            lines.append(row_text)
    return "\n".join(lines)


def _extract_text_from_image(content_bytes: bytes) -> str:
    """Extracts text from images using pytesseract OCR if available, else regex patterns."""
    from PIL import Image
    try:
        import pytesseract
        img = Image.open(io.BytesIO(content_bytes))
        return pytesseract.image_to_string(img)
    except Exception:
        # Fallback if tesseract binary is not installed on system path
        return "IMAGE_OCR_UNAVAILABLE: Please provide text or PDF"


def _text_to_telemetry_df(text: str, default_zone: str = "Zone Uploaded") -> pd.DataFrame:
    """
    Parses arbitrary free-form document text (utility bills, SCADA logs, audit reports)
    to find dates, zones, consumption, pressure, and quality parameters.
    """
    rows = []
    lines = text.splitlines()

    # Pattern matchers
    date_regex = re.compile(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?)")
    zone_regex = re.compile(r"(Zone\s+[A-Za-z0-9]+|District\s+[A-Za-z0-9]+|DMA\s+[0-9]+)", re.I)
    num_regex = re.compile(r"[-+]?\d*\.?\d+")

    current_zone = default_zone
    base_date = datetime.now()

    for idx, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean or len(line_clean) < 3:
            continue

        # Check if line contains a zone name
        zm = zone_regex.search(line_clean)
        if zm:
            current_zone = zm.group(1).title()

        # Check for date in line
        dm = date_regex.search(line_clean)
        if dm:
            try:
                row_date = pd.to_datetime(dm.group(1))
            except Exception:
                row_date = base_date - timedelta(hours=len(rows))
        else:
            row_date = base_date - timedelta(hours=len(rows))

        # Look for numbers (consumption, pressure, pH)
        numbers = [float(n) for n in num_regex.findall(line_clean) if n not in ('.', '-')]
        if len(numbers) >= 1:
            # First significant number is usually consumption or meter reading
            cons_val = numbers[0]
            # Heuristic sanity mapping
            pressure = numbers[1] if len(numbers) > 1 and 20 <= numbers[1] <= 120 else 52.0
            ph = numbers[2] if len(numbers) > 2 and 4.0 <= numbers[2] <= 10.0 else 7.2
            turb = numbers[3] if len(numbers) > 3 and 0.0 <= numbers[3] <= 15.0 else 2.0

            rows.append({
                "timestamp": row_date.strftime("%Y-%m-%d %H:%M:%S"),
                "zone": current_zone,
                "consumption_l": abs(cons_val),
                "flow_lpm": round(abs(cons_val) / 60.0, 2),
                "pressure_psi": round(pressure, 2),
                "ph": round(ph, 2),
                "turbidity_ntu": round(turb, 2)
            })

    # If text format was pure tabular CSV embedded in PDF/DOCX
    if not rows or len(rows) < 2:
        try:
            csv_df = pd.read_csv(io.StringIO(text))
            if len(csv_df.columns) >= 2:
                return csv_df
        except Exception:
            pass

    return pd.DataFrame(rows)


def ingest_file(filename: str, content_bytes: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Unified entry point for CSV, PDF, DOCX, TXT, and Images.
    Returns: (standardized_dataframe, metadata_dict)
    """
    ext = Path(filename).suffix.lower()
    meta = {
        "filename": filename,
        "format": ext.replace(".", "").upper(),
        "file_size_bytes": len(content_bytes),
        "extracted_text_preview": ""
    }

    if ext in (".csv", ".tsv"):
        df = _parse_tabular_or_csv(content_bytes)
        meta["extraction_method"] = "Direct Tabular Ingestion"

    elif ext == ".pdf":
        raw_text = _extract_text_from_pdf(content_bytes)
        meta["extracted_text_preview"] = raw_text[:300]
        meta["extraction_method"] = "PyPDF Document Extraction"
        df = _text_to_telemetry_df(raw_text)

    elif ext in (".docx", ".doc"):
        raw_text = _extract_text_from_docx(content_bytes)
        meta["extracted_text_preview"] = raw_text[:300]
        meta["extraction_method"] = "DOCX Document & Table Parser"
        df = _text_to_telemetry_df(raw_text)

    elif ext in (".txt", ".log", ".asc"):
        raw_text = content_bytes.decode("utf-8", errors="ignore")
        meta["extracted_text_preview"] = raw_text[:300]
        meta["extraction_method"] = "ASCII SCADA Log Parser"
        df = _text_to_telemetry_df(raw_text)

    elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        raw_text = _extract_text_from_image(content_bytes)
        meta["extracted_text_preview"] = raw_text[:300]
        meta["extraction_method"] = "OCR Visual Telemetry Extractor"
        df = _text_to_telemetry_df(raw_text)

    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: CSV, PDF, DOCX, TXT, PNG, JPG.")

    if df.empty or len(df) == 0:
        raise ValueError(f"Could not extract valid water telemetry records from {filename}.")

    meta["parsed_rows"] = len(df)
    meta["detected_zones"] = sorted(df["zone"].unique().tolist()) if "zone" in df.columns else ["Default Zone"]

    return df, meta
