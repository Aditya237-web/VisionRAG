# scripts/make_scan.py
# Create a scanned PDF by rasterizing an existing PDF (removes text layer)
import fitz
import sys

def make_scan(input_url_or_path: str, output_path: str, dpi: int = 130):
    """Rasterize a PDF to create a scanned version with no text layer."""
    doc = fitz.open(input_url_or_path)
    pdf_bytes = doc.convert_to_pdf(dpi=dpi)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"Created {output_path} with {doc.page_count} pages at {dpi} DPI")

if __name__ == "__main__":
    # BERT paper from arXiv
    make_scan("https://arxiv.org/pdf/1810.04805", "BERT_paper01.pdf", dpi=130)