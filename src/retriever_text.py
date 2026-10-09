# VisionRAG Text Baseline Retriever
# PyMuPDF text extraction + sentence-transformers embedding + FAISS indexing
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils import load_config, ensure_dir, load_jsonl, save_jsonl
from src.page_ids import build_page_id


class TextRetriever:
    """Text baseline retriever using sentence-transformers + FAISS."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        rt_cfg = config.get("retriever_text", {})

        self.embedder_name = rt_cfg.get("embedder_name", "BAAI/bge-small-en-v1.5")
        self.chunk_size = rt_cfg.get("chunk_size", 500)
        self.chunk_overlap = rt_cfg.get("chunk_overlap", 50)
        self.batch_size = rt_cfg.get("batch_size", 16)
        self.device_str = rt_cfg.get("device", "auto")
        self.query_prefix = rt_cfg.get(
            "query_prefix",
            "Represent this sentence for searching relevant passages: "
        )

        self.device = "cuda" if (self.device_str == "auto" and torch.cuda.is_available()) else "cpu"

        print(f"Loading embedder: {self.embedder_name} on {self.device}")
        self.embedder = SentenceTransformer(self.embedder_name, device=self.device)
        self.tokenizer = self.embedder.tokenizer
        self.dim = self.embedder.get_sentence_embedding_dimension()
        print(f"Embedder loaded. Dimension: {self.dim}")

        # Index components (loaded on demand)
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict] = []
        self.chunk_to_page: List[str] = []  # chunk_idx -> page_id
        self.page_order: List[str] = []

    def _get_page_id(self, doc_name: str, page_num: int) -> str:
        """Generate consistent page_id matching visual retriever format."""
        return build_page_id(doc_name, page_num)

    def extract_page_text(self, pdf_path: Path, page_num: int) -> str:
        """Extract text from a single PDF page using PyMuPDF."""
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_num - 1]  # 0-indexed
            text = page.get_text("text")
        finally:
            doc.close()
        return text.strip()

    def chunk_page_text(self, text: str, doc_name: str, page_num: int) -> List[Dict]:
        """Chunk a page's text by tokens with overlap.
        
        Uses tokenizer offset_mapping to slice original text by character offsets.
        Falls back to encode/decode if offsets unavailable.
        Never chunks across pages.
        """
        if not text:
            return []

        page_id = self._get_page_id(doc_name, page_num)
        chunks = []

        # Try tokenizing with offset mapping
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=False,
            add_special_tokens=False
        )
        input_ids = encoding["input_ids"]
        offsets = encoding.get("offset_mapping")

        if offsets is None or len(offsets) == 0:
            # Fallback: encode/decode
            return self._chunk_by_encode_decode(text, doc_name, page_num)

        chunk_id = 0
        stride = self.chunk_size - self.chunk_overlap

        for start in range(0, len(input_ids), stride):
            end = min(start + self.chunk_size, len(input_ids))
            if start >= len(input_ids):
                break

            # Get character offsets for this token window
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            chunk_text = text[char_start:char_end]

            chunks.append({
                "text": chunk_text,
                "doc_name": doc_name,
                "page_num": page_num,
                "chunk_id": chunk_id,
                "page_id": page_id,
                "token_start": start,
                "token_end": end,
            })
            chunk_id += 1

            if end == len(input_ids):
                break

        return chunks

    def _chunk_by_encode_decode(self, text: str, doc_name: str, page_num: int) -> List[Dict]:
        """Fallback chunking using encode/decode when offsets unavailable."""
        page_id = self._get_page_id(doc_name, page_num)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        chunk_id = 0
        stride = self.chunk_size - self.chunk_overlap

        for start in range(0, len(tokens), stride):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)

            chunks.append({
                "text": chunk_text,
                "doc_name": doc_name,
                "page_num": page_num,
                "chunk_id": chunk_id,
                "page_id": page_id,
            })
            chunk_id += 1

            if end == len(tokens):
                break

        return chunks

    def build_index(self, pdf_dir: Path, index_dir: Path) -> None:
        """Build text index from PDFs with FAISS."""
        ensure_dir(index_dir)

        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return

        all_chunks = []
        chunk_to_page = []
        page_order = []
        seen_pages = set()

        pages_with_text = 0
        pages_no_text = 0
        chunks_per_page_counts = []

        start_time = time.time()

        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            doc_name = pdf_path.stem
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            doc.close()

            page_chunks_list = []

            for page_num in range(1, num_pages + 1):
                page_id = self._get_page_id(doc_name, page_num)

                if page_id in seen_pages:
                    continue

                text = self.extract_page_text(pdf_path, page_num)

                if not text:
                    pages_no_text += 1
                    continue

                chunks = self.chunk_page_text(text, doc_name, page_num)
                if not chunks:
                    pages_no_text += 1
                    continue

                pages_with_text += 1
                page_chunks_list.extend(chunks)
                chunks_per_page_counts.append(len(chunks))

                if page_id not in seen_pages:
                    seen_pages.add(page_id)
                    page_order.append(page_id)

            all_chunks.extend(page_chunks_list)

        if not all_chunks:
            print("No text chunks extracted from any PDF")
            return

        # Embed all chunks in batches
        print(f"Embedding {len(all_chunks)} chunks in batches of {self.batch_size}...")
        chunk_texts = [c["text"] for c in all_chunks]

        embeddings_list = []
        for i in tqdm(range(0, len(chunk_texts), self.batch_size), desc="Embedding chunks"):
            batch = chunk_texts[i:i + self.batch_size]
            batch_emb = self.embedder.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.batch_size
            )
            embeddings_list.append(batch_emb)

        embeddings = np.vstack(embeddings_list).astype("float32")

        # Build FAISS index (IndexFlatIP for cosine similarity with normalized vectors)
        print("Building FAISS index...")
        index = faiss.IndexFlatIP(self.dim)
        index.add(embeddings)

        # Build chunk_to_page mapping
        chunk_to_page = [c["page_id"] for c in all_chunks]

        # Save index components
        faiss.write_index(index, str(index_dir / "faiss.index"))
        save_jsonl(all_chunks, index_dir / "chunks.jsonl")

        with open(index_dir / "chunk_to_page.json", "w") as f:
            json.dump(chunk_to_page, f)

        with open(index_dir / "page_order.json", "w") as f:
            json.dump(page_order, f)

        # Save build stats
        elapsed = time.time() - start_time
        index_size = (index_dir / "faiss.index").stat().st_size / 1e6

        stats = {
            "pages_with_text": pages_with_text,
            "pages_no_text": pages_no_text,
            "total_chunks": len(all_chunks),
            "chunks_per_page": {
                "min": min(chunks_per_page_counts) if chunks_per_page_counts else 0,
                "max": max(chunks_per_page_counts) if chunks_per_page_counts else 0,
                "mean": sum(chunks_per_page_counts) / len(chunks_per_page_counts) if chunks_per_page_counts else 0,
            },
            "embedding_dim": self.dim,
            "embedder_name": self.embedder_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "elapsed_seconds": elapsed,
            "index_size_mb": index_size,
        }

        with open(index_dir / "build_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # Store in memory for immediate use
        self.index = index
        self.chunks = all_chunks
        self.chunk_to_page = chunk_to_page
        self.page_order = page_order

        # Print summary
        print(f"\nText index build complete:")
        print(f"  Pages with text: {pages_with_text}")
        print(f"  Pages no text (scanned): {pages_no_text}")
        print(f"  Total chunks: {len(all_chunks)}")
        print(f"  Chunks/page: min={stats['chunks_per_page']['min']}, max={stats['chunks_per_page']['max']}, mean={stats['chunks_per_page']['mean']:.1f}")
        print(f"  Build time: {elapsed:.1f}s")
        print(f"  Index size: {index_size:.2f} MB")
        print(f"  Saved to: {index_dir}")

    def load_index(self, index_dir: Path) -> None:
        """Load index components from disk."""
        index_path = index_dir / "faiss.index"
        chunks_path = index_dir / "chunks.jsonl"
        chunk_to_page_path = index_dir / "chunk_to_page.json"
        page_order_path = index_dir / "page_order.json"

        if not all(p.exists() for p in [index_path, chunks_path, chunk_to_page_path, page_order_path]):
            raise FileNotFoundError(f"Incomplete index in {index_dir}")

        self.index = faiss.read_index(str(index_path))
        self.chunks = load_jsonl(chunks_path)

        with open(chunk_to_page_path, "r") as f:
            self.chunk_to_page = json.load(f)

        with open(page_order_path, "r") as f:
            self.page_order = json.load(f)

        print(f"Loaded text index: {len(self.chunks)} chunks, {len(self.page_order)} pages")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query with the query prefix."""
        query_text = self.query_prefix + query
        emb = self.embedder.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return emb.astype("float32")

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve top-k pages for a query.
        
        Returns page-level results by taking max score across chunks per page.
        """
        if self.index is None:
            # Try default index dir
            index_dir = Path(self.config["paths"]["index_dir"]) / "text"
            self.load_index(index_dir)

        query_emb = self.embed_query(query)  # [1, dim]

        # Search more chunks to ensure k distinct pages
        search_k = min(k * 3, len(self.chunks))
        scores, indices = self.index.search(query_emb, search_k)

        # Aggregate by page_id: max score per page
        page_scores: Dict[str, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_to_page):
                continue
            page_id = self.chunk_to_page[idx]
            if page_id not in page_scores or score > page_scores[page_id]:
                page_scores[page_id] = float(score)

        # Sort and return top-k
        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_pages[:k]


def main():
    parser = argparse.ArgumentParser(description="Text Baseline Retriever for VisionRAG")
    parser.add_argument(
        "--build_index", action="store_true", help="Build text index from PDFs"
    )
    parser.add_argument(
        "--query", type=str, help="Query string to search"
    )
    parser.add_argument(
        "--k", type=int, default=3, help="Number of results to return"
    )
    parser.add_argument(
        "--pdf_dir", type=str, default="data/pdfs", help="Directory with PDF files"
    )
    parser.add_argument(
        "--index_dir", type=str, default="data/index/text", help="Directory for index files"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config.yaml"
    )

    args = parser.parse_args()

    config = load_config(args.config)
    retriever = TextRetriever(config)

    pdf_dir = Path(args.pdf_dir)
    index_dir = Path(args.index_dir)

    if args.build_index:
        retriever.build_index(pdf_dir, index_dir)
    elif args.query:
        results = retriever.retrieve(args.query, args.k)
        print(f"\nTop-{args.k} results for: {args.query}")
        for i, (page_id, score) in enumerate(results, 1):
            print(f"  {i}. {page_id}  score={score:.4f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
