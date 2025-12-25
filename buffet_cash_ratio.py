import re
import io
import requests

import numpy as np
import pdfplumber
from typing import List, Dict, Optional, Any

PDF_URL = "https://www.berkshirehathaway.com/qtrly/3rdqtr24.pdf"
# PDF_URL = "https://www.berkshirehathaway.com/2024ar/2024ar.pdf"

SECTION_HEADERS = [
    r"Insurance and Other:",
    r"Railroad, Utilities and Energy:",
]

NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+")  # e.g. 72,156 or 1,225,963


def extract_first_labeled_row_on_pages(
    pdf_bytes: bytes,
    label_regex: str,
    page_indices: List[int],
    lookahead: int = 10,
) -> Dict[str, Any]:
    """Same as extract_first_labeled_row but only scans selected 0-indexed pages."""
    label_pat = re.compile(label_regex, re.IGNORECASE)

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_idx in page_indices:
            page = pdf.pages[page_idx]
            text = page.extract_text() or ""
            text = re.sub(r"[ \t]+", " ", text)
            lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]

            for i, ln in enumerate(lines):
                if label_pat.search(ln):
                    vals = find_two_numbers_forward(lines, i, lookahead=lookahead)
                    if vals:
                        return {"values": vals, "page": page_idx + 1, "line": ln}

    raise RuntimeError(f"Could not find label on selected pages: {label_regex}")

def find_pages_with_phrase(pdf_bytes: bytes, phrase_regex: str) -> List[int]:
    """Return 0-indexed page indices whose extracted text matches phrase_regex."""
    pat = re.compile(phrase_regex, re.IGNORECASE)
    hits: List[int] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if pat.search(text):
                hits.append(page_idx)

    return hits


def extract_largest_two_number_pair_on_pages(
    pdf_bytes: bytes,
    page_indices: List[int],
) -> Dict[str, Any]:
    """
    On given pages only, pick the line with >=2 comma-numbers and the largest first number.
    """
    best = None  # (i0, i1, page_idx, line, [v0, v1])

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_idx in page_indices:
            text = pdf.pages[page_idx].extract_text() or ""
            text = re.sub(r"[ \t]+", " ", text)
            lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]

            for ln in lines:
                nums = NUM_RE.findall(ln)
                if len(nums) >= 2:
                    v0, v1 = nums[0], nums[1]
                    i0, i1 = int(v0.replace(",", "")), int(v1.replace(",", ""))
                    if (best is None) or (i0 > best[0]):
                        best = (i0, i1, page_idx, ln, [v0, v1])

    if best is None:
        raise RuntimeError("Fallback failed: no 2-number lines found on BALANCE SHEET pages.")

    return {
        "values": best[4],
        "page": best[2] + 1,
        "line": best[3],
        "heuristic": "largest_two_number_pair_on_balance_sheet_pages",
    }

def download_pdf_bytes(url: str) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; brk-crawler/1.0)"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.content


def find_two_numbers_forward(lines: List[str], start_idx: int, lookahead: int = 10) -> Optional[List[str]]:
    """From start_idx, scan forward and return the first 2 comma-numbers found."""
    found: List[str] = []
    for j in range(start_idx, min(len(lines), start_idx + lookahead)):
        found.extend(NUM_RE.findall(lines[j]))
        if len(found) >= 2:
            return found[:2]
    return None


def extract_all_labeled_rows_with_sections(
    pdf_bytes: bytes,
    label_regex: str,
    section_headers: List[str],
    lookahead: int = 10,
) -> List[Dict[str, Any]]:
    """
    Extract ALL occurrences of label_regex, attaching the nearest preceding section header.
    Returns a list of hits.
    """
    label_pat = re.compile(label_regex, re.IGNORECASE)
    header_pats = [(re.compile(h, re.IGNORECASE), h) for h in section_headers]

    hits: List[Dict[str, Any]] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = re.sub(r"[ \t]+", " ", text)
            lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]

            current_section: Optional[str] = None

            for i, ln in enumerate(lines):
                # Update section
                for hp, raw in header_pats:
                    if hp.search(ln):
                        current_section = raw.rstrip(":")
                        break

                # Match label
                if label_pat.search(ln):
                    vals = find_two_numbers_forward(lines, i, lookahead=lookahead)
                    if vals:
                        hits.append({
                            "section": current_section,
                            "values": vals,
                            "page": page_idx + 1,
                            "line": ln,
                        })

    return hits


def extract_first_labeled_row(
    pdf_bytes: bytes,
    label_regex: str,
    lookahead: int = 10,
) -> Dict[str, Any]:
    """
    Extract the FIRST occurrence of a label (for singletons like treasury bills / total assets).
    Robust to line breaks: once label is found, grab next 2 numbers across following lines.
    """
    label_pat = re.compile(label_regex, re.IGNORECASE)

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = re.sub(r"[ \t]+", " ", text)
            lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]

            for i, ln in enumerate(lines):
                if label_pat.search(ln):
                    vals = find_two_numbers_forward(lines, i, lookahead=lookahead)
                    if vals:
                        return {
                            "values": vals,
                            "page": page_idx + 1,
                            "line": ln,
                        }

    raise RuntimeError(f"Could not find label: {label_regex}")


def parse_ints(values: List[str]) -> List[int]:
    return np.array([int(v.replace(",", "")) for v in values])


def extract_balance_sheet_items(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Returns a dict like:
      {
        "cash_and_cash_equivalents": [
            {"section": "...", "values": [...], ...},
            {"section": "...", "values": [...], ...},
        ],
        "treasury_bills": {"values": [...], ...},
        "total_assets": {"values": [...], ...},
      }
    """
    out: Dict[str, Any] = {}

    # duplicated label -> list
    out["cash_and_cash_equivalents"] = extract_all_labeled_rows_with_sections(
        pdf_bytes,
        label_regex=r"Cash and cash equivalents",
        section_headers=SECTION_HEADERS,
        lookahead=10,
    )

    # Treasury bills: try precise label first, then fall back to balance-sheet-only scan
    try:
        out["treasury_bills"] = extract_first_labeled_row(
            pdf_bytes,
            # label_regex=r"Short[-\s]*term\s+investments.*U\.S\.?\s*Treasury\s*Bills?",
            label_regex=r"Fixed maturity securities",
            lookahead=12,
        )
        out["treasury_bills"]["heuristic"] = "label_match"
    except RuntimeError:
        bs_pages = find_pages_with_phrase(pdf_bytes, r"CONSOLIDATED\s+BALANCE\s+SHEETS")
        if not bs_pages:
            raise RuntimeError("Could not locate any page with 'CONSOLIDATED BALANCE SHEETS' for treasury-bills fallback.")

        out["treasury_bills"] = extract_first_labeled_row_on_pages(
            pdf_bytes,
            label_regex=r"(U\.S\.?)?\s*Treasury\s*Bills?",
            page_indices=bs_pages,
            lookahead=20,
        )
        out["treasury_bills"]["heuristic"] = "balance_sheet_pages_fallback"

    # 1) try label-based Total assets
    try:
        out["total_assets"] = extract_first_labeled_row(
            pdf_bytes,
            label_regex=r"Total assets",
            lookahead=12,
        )
        out["total_assets"]["heuristic"] = "label_match"
        return out
    except RuntimeError:
        pass

    # 2) fallback: ONLY on pages containing "CONSOLIDATED BALANCE SHEETS"
    bs_pages = find_pages_with_phrase(
        pdf_bytes,
        phrase_regex=r"CONSOLIDATED\s+BALANCE\s+SHEETS",
    )
    if not bs_pages:
        raise RuntimeError("Could not locate any page with 'CONSOLIDATED BALANCE SHEETS'.")

    out["total_assets"] = extract_largest_two_number_pair_on_pages(pdf_bytes, bs_pages)

    return out


def main():

    # for yr in np.arange(16, 0, -1):
    for yr in np.arange(14, 0, -1):
        # for qtr in ['3rd', '2nd', '1st']:
        for qtr in ['3rd']:
            pdf_url = f"https://www.berkshirehathaway.com/qtrly/{qtr}qtr{yr}.pdf"
            pdf_bytes = download_pdf_bytes(pdf_url)
            data = extract_balance_sheet_items(pdf_bytes)

            cash_totals = 0
            #print("\nCASH rows:")
            for h in data["cash_and_cash_equivalents"]:
                if h["section"] != None:
                    #print(" ", h["section"], "=>", h["values"], "page", h["page"], "ints", parse_ints(h["values"]))
                    cash_totals += parse_ints(h["values"])

            #print("\nTreasury bills:", data["treasury_bills"]["values"], "ints", parse_ints(data["treasury_bills"]["values"]))
            cash_totals += parse_ints(data["treasury_bills"]["values"])
            #print("Total assets:", data["total_assets"]["values"], "ints", parse_ints(data["total_assets"]["values"]))
            cash_ratios = cash_totals / parse_ints(data["total_assets"]["values"])
            print(f"{yr} {qtr} Cash ratio: {cash_ratios[0]*100:.1f}%")

if __name__ == "__main__":
    main()