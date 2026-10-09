# VisionRAG PDF Ingestion
# Renders PDF pages to images using PyMuPDF (no Poppler needed)
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
from tqdm import tqdm

from src.utils import load_config, ensure_dir, save_jsonl


def render_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 130,
    image_format: str = "png"
) -> List[Dict[str, Any]]:
    """Render each page of a PDF to an image file.
    
    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save page images.
        dpi: Rendering DPI (default 130 for memory efficiency).
        image_format: Output image format (default png).
    
    Returns:
        List of page metadata dicts with keys:
        doc, page_num, image_path, width, height
    """
    doc_name = pdf_path.stem
    ensure_dir(output_dir)
    
    metadata = []
    doc = fitz.open(pdf_path)
    
    try:
        for page_num, page in enumerate(doc):
            # Render page to pixmap at specified DPI
            pix = page.get_pixmap(dpi=dpi)
            
            # Save image
            img_filename = f"page_{page_num + 1:04d}.{image_format}"
            img_path = output_dir / img_filename
            pix.save(str(img_path))
            
            # Record metadata - use relative path from project root
            try:
                rel_path = img_path.relative_to(Path.cwd())
            except ValueError:
                # If not relative to cwd, use as-is
                rel_path = img_path
            
            metadata.append({
                "doc": pdf_path.name,
                "doc_name": doc_name,
                "page_num": page_num + 1,  # 1-indexed for human readability
                "image_path": str(rel_path),
                "width": pix.width,
                "height": pix.height,
            })
    finally:
        doc.close()
    
    return metadata


def process_pdf_dir(
    pdf_dir: Path,
    pages_dir: Path,
    dpi: int = 130,
    image_format: str = "png"
) -> None:
    """Process all PDFs in a directory, render pages, save metadata.
    
    Args:
        pdf_dir: Directory containing PDF files.
        pages_dir: Root directory for output page images.
        dpi: Rendering DPI.
        image_format: Output image format.
    """
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    all_metadata = []
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        doc_name = pdf_path.stem
        output_dir = pages_dir / doc_name
        
        page_metadata = render_pdf_to_images(
            pdf_path, output_dir, dpi=dpi, image_format=image_format
        )
        all_metadata.extend(page_metadata)
        print(f"  {pdf_path.name}: {len(page_metadata)} pages")
    
    # Save combined metadata
    metadata_path = pages_dir / "metadata.jsonl"
    save_jsonl(all_metadata, metadata_path)
    print(f"\nSaved metadata for {len(all_metadata)} pages to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Render PDF pages to images")
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="data/pdfs",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--pages_dir",
        type=str,
        default="data/pages",
        help="Output directory for page images"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=130,
        help="Rendering DPI (default: 130)"
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image format (default: png)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml"
    )
    
    args = parser.parse_args()
    
    # Load config for defaults, override with CLI args
    config = load_config(args.config)
    ingest_cfg = config.get("ingest", {})
    
    dpi = args.dpi if args.dpi != 130 else ingest_cfg.get("dpi", 130)
    image_format = args.image_format if args.image_format != "png" else ingest_cfg.get("image_format", "png")
    
    pdf_dir = Path(args.pdf_dir)
    pages_dir = Path(args.pages_dir)
    
    if not pdf_dir.exists():
        print(f"Error: PDF directory not found: {pdf_dir}")
        return
    
    process_pdf_dir(pdf_dir, pages_dir, dpi=dpi, image_format=image_format)


if __name__ == "__main__":
    main()
