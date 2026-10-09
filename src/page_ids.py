# VisionRAG page id helpers
# Shared by heavy (retriever) and light (evaluate/scripts) consumers,
# so this module must only import from the standard library.
from __future__ import annotations

from typing import Tuple


def build_page_id(doc_name: str, page_num: int) -> str:
    """Build a page id, e.g. build_page_id('f_paper01', 1) -> 'f_paper01_page_0001'.

    Doc names may contain underscores; only the trailing '_page_XXXX' is special.
    """
    return f"{doc_name}_page_{int(page_num):04d}"


def parse_page_id(page_id: str) -> Tuple[str, int]:
    """Inverse of build_page_id: parse 'f_paper01_page_0001' -> ('f_paper01', 1).

    Splits on the LAST '_page_' so doc names containing underscores (or even
    the substring '_page_') parse correctly.
    """
    doc_name, sep, page_str = page_id.rpartition("_page_")
    if not sep or not doc_name or not page_str.isdigit():
        raise ValueError(f"Invalid page_id: {page_id!r}")
    return doc_name, int(page_str)
