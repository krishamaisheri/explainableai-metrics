"""
PDF Text Extractor.

Extracts all PDFs in data/ → extracted/<category>/<name>.md
Uses PyMuPDF native OCR at 300 DPI for scanned documents.
"""

import os
import time
import fitz  # PyMuPDF
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXTRACTED_DIR = os.path.join(BASE_DIR, "extracted")

# Chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ────────────────────────────────────────────────
#   PDF TEXT EXTRACTION (PyMuPDF native OCR)
# ────────────────────────────────────────────────
def extract_pdf_text(pdf_path: str, max_pages: int = 10) -> str:
    """
    Phase 1: standard text layer.
    Phase 2: native OCR at 300 DPI if standard extraction yields < 100 chars.
    Returns cleaned text.
    """
    doc = fitz.open(pdf_path)
    pages = min(max_pages, len(doc))

    # Try standard extraction first
    text = ""
    for i in range(pages):
        page = doc.load_page(i)
        text += page.get_text()

    if len(text.strip()) > 100:
        doc.close()
        return clean_text(text)

    # OCR fallback at 300 DPI
    print(f"    🔬 OCR scanning {os.path.basename(pdf_path)}...")
    text = ""
    for i in range(pages):
        page = doc.load_page(i)
        ocr_tp = page.get_textpage_ocr(flags=0, language="eng", dpi=300, full=True)
        page_text = page.get_text(textpage=ocr_tp)
        text += page_text + "\n"
        del ocr_tp  # Explicit cleanup

    doc.close()
    return clean_text(text)


def clean_text(text: str) -> str:
    """Remove junk lines, excessive whitespace."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) > 2:
            lines.append(line)
    return "\n".join(lines)


# ────────────────────────────────────────────────
#   CHUNKER
# ────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks for vector storage."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# ────────────────────────────────────────────────
#   MAIN EXTRACTION PIPELINE
# ────────────────────────────────────────────────
def extract_all_pdfs(data_dir: str = DATA_DIR) -> list[dict]:
    """
    Walk data/, extract text from every PDF, save as .md,
    chunk for vector storage.

    Returns list of dicts with keys: text, source, category, page, chunk_index
    """
    pdf_files = sorted(Path(data_dir).rglob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDFs found in {data_dir}")
        return []

    print(f"📂 {len(pdf_files)} PDFs detected\n")

    all_chunks = []
    extracted_count = 0

    for i, pdf_path in enumerate(pdf_files):
        category = pdf_path.parent.name
        fname = pdf_path.stem
        doc_id = f"{category}/{pdf_path.name}"

        md_dir = os.path.join(EXTRACTED_DIR, category)
        md_path = os.path.join(md_dir, f"{fname}.md")

        # If .md already exists, read from it instead of re-extracting
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                raw = f.read()
            # Strip the markdown header we added
            text = raw.split("---\n\n", 1)[-1] if "---\n\n" in raw else raw
            print(f"  [{i+1}/{len(pdf_files)}] {doc_id} → cached .md ({len(text)} chars)")
        else:
            t0 = time.time()
            text = extract_pdf_text(str(pdf_path), max_pages=10)
            dt = time.time() - t0

            if not text or len(text) < 50:
                print(f"  [{i+1}/{len(pdf_files)}] {doc_id} → ⚠️  SKIPPED (no content)")
                continue

            # Save .md file
            os.makedirs(md_dir, exist_ok=True)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {fname}\n\n")
                f.write(f"**Source**: `{doc_id}`\n\n")
                f.write(f"---\n\n")
                f.write(text)

            print(f"  [{i+1}/{len(pdf_files)}] {doc_id} → {len(text)} chars ({dt:.1f}s)")

        # Chunk the text
        chunks = chunk_text(text)
        for j, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc_id,
                "category": category,
                "page": 0,
                "chunk_index": j,
            })

        extracted_count += 1

    print(f"\n✅ Extracted {len(all_chunks)} chunks from {extracted_count}/{len(pdf_files)} PDFs")
    print(f"📁 Markdown files saved to: {EXTRACTED_DIR}")
    return all_chunks


if __name__ == "__main__":
    chunks = extract_all_pdfs()
    for c in chunks[:5]:
        print(f"\n--- [{c['category']}] {c['source']} chunk {c['chunk_index']} ---")
        preview = c["text"][:200]
        print(preview + "…" if len(c["text"]) > 200 else preview)
