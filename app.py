# VisionRAG - Visual Document QA with ColPali + Ollama
# Streamlit app: upload PDFs, build indexes, QA with visual/text/OCR retrievers
import streamlit as st
import time
import json
from pathlib import Path
from typing import Tuple
from PIL import Image

st.set_page_config(
    page_title="VisionRAG - Visual Document QA",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stImage > img { border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)


def _imports():
    from src.utils import load_config
    from src.retriever_visual import VisualRetriever
    from src.retriever_text import TextRetriever
    from src.retriever_ocr_text import OCRTextRetriever
    from src.retriever_hybrid import HybridRetriever
    from src.generator import Generator, GeneratorError
    from src.build_compressed_visual import CompressedVisualRetriever
    from src.page_ids import parse_page_id
    return dict(load_config=load_config, VisualRetriever=VisualRetriever,
                TextRetriever=TextRetriever, OCRTextRetriever=OCRTextRetriever,
                HybridRetriever=HybridRetriever, Generator=Generator,
                GeneratorError=GeneratorError,
                CompressedVisualRetriever=CompressedVisualRetriever,
                parse_page_id=parse_page_id)


@st.cache_resource
def get_config():
    from src.utils import load_config
    return load_config("config.yaml")


RETRIEVERS = {
    "visual": ("📊 Visual (ColPali)", "ColSmol visual retriever — best for charts, diagrams, layouts"),
    "visual_int8": ("📦 Visual INT8", "INT8 quantized visual (smaller index)"),
    "visual_mean_pool_64": ("📦 Visual MeanPool-64", "Mean-pooled to 64 vectors"),
    "visual_mean_pool_256": ("📦 Visual MeanPool-256", "Mean-pooled to 256 vectors"),
    "visual_fp16_local": ("🔍 Visual Local Patches", "Local patches only, 36% smaller index"),
    "text": ("📝 Text", "BGE embeddings + FAISS"),
    "ocr_text": ("🔍 OCR + Text", "Text + RapidOCR for scanned pages"),
    "hybrid": ("🔀 Hybrid (RRF)", "Reciprocal Rank Fusion of Visual + OCR+Text"),
}


def index_dir(name: str) -> Path:
    return {
        "visual": Path("data/index"),
        "hybrid": Path("data/index"),
        "text": Path("data/index/text"),
        "ocr_text": Path("data/index/ocr_text"),
        "visual_int8": Path("data/index/visual_int8"),
        "visual_mean_pool_64": Path("data/index/visual_mean_pool_64"),
        "visual_mean_pool_256": Path("data/index/visual_mean_pool_256"),
        "visual_fp16_local": Path("data/index/visual_fp16_local"),
    }[name]


@st.cache_resource(show_spinner=False)
def get_retriever(name: str):
    imp = _imports()
    cfg = get_config()
    if name == "visual":
        r = imp["VisualRetriever"](cfg); r.load_index(Path("data/index")); r.load_model(); return r
    if name == "text":
        r = imp["TextRetriever"](cfg); r.load_index(Path("data/index/text")); return r
    if name == "ocr_text":
        r = imp["OCRTextRetriever"](cfg); r.load_index(Path("data/index/ocr_text")); return r
    if name == "hybrid":
        r = imp["HybridRetriever"](cfg); return r
    if name == "visual_int8":
        r = imp["CompressedVisualRetriever"](cfg, compression_mode="int8")
        r.load_index(Path("data/index/visual_int8")); return r
    if name == "visual_mean_pool_64":
        r = imp["CompressedVisualRetriever"](cfg, compression_mode="mean_pool_64")
        r.load_index(Path("data/index/visual_mean_pool_64")); return r
    if name == "visual_mean_pool_256":
        r = imp["CompressedVisualRetriever"](cfg, compression_mode="mean_pool_256")
        r.load_index(Path("data/index/visual_mean_pool_256")); return r
    if name == "visual_fp16_local":
        r = imp["VisualRetriever"](cfg)
        r.config["retriever_visual"]["use_local_patches_only"] = True
        r.load_index(Path("data/index/visual_fp16_local")); r.load_model(); return r
    raise ValueError(name)


def page_image(page_id: str):
    imp = _imports()
    doc, num = imp["parse_page_id"](page_id)
    return Path("data/pages") / doc / f"page_{num:04d}.png"


def check_ollama(host: str = "http://localhost:11434") -> Tuple[bool, str]:
    import httpx
    try:
        r = httpx.get(f"{host}/api/tags", timeout=5.0)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return True, "Ollama running. Models: " + ", ".join(models)
        return False, f"Ollama status {r.status_code}"
    except Exception as e:
        return False, f"Cannot connect to Ollama at {host}: {e}. Start it with `ollama serve`."


def upload_tab():
    st.subheader("📤 Upload PDFs & Build Indexes")
    from src import ingest
    files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("📂 Save & Build Indexes", disabled=not files) and files:
        pdf_dir = Path("data/pdfs"); pdf_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            (pdf_dir / Path(f.name).name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(files)} file(s) to data/pdfs/")
        progress = st.progress(0.0, text="Ingesting pages...")
        try:
            import subprocess, sys
            steps = [
                ("Ingesting PDFs → page images", [sys.executable, "-m", "src.ingest", "--pdf_dir", "data/pdfs"]),
                ("Building visual index", [sys.executable, "-m", "src.retriever_visual", "--build_index"]),
                ("Building text index", [sys.executable, "-m", "src.retriever_text", "--build_index"]),
                ("Building OCR text index", [sys.executable, "-m", "src.retriever_ocr_text", "--build_index"]),
            ]
            for i, (label, cmd) in enumerate(steps):
                progress.progress(i / len(steps), text=label)
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
                if proc.returncode != 0:
                    st.error(f"{label} failed:\n```\n{(proc.stderr or proc.stdout)[-2000:]}\n```")
                    return
            progress.progress(1.0, text="Done")
            st.cache_resource.clear()
            st.success("Indexes rebuilt.")
        except Exception as e:
            st.error(f"Build failed: {e}")


def main():
    st.title("📄 VisionRAG - Visual Document QA")
    st.caption("ColPali visual retriever + Ollama VLM — fully local")

    ollama_ok, ollama_msg = check_ollama()

    with st.sidebar:
        st.header("⚙️ Settings")
        if ollama_ok:
            st.success(f"🟢 {ollama_msg}")
        else:
            st.error(f"🔴 {ollama_msg}")

        st.divider()
        st.subheader("🔍 Retriever")
        selected = st.selectbox(
            "Retriever", list(RETRIEVERS),
            format_func=lambda x: RETRIEVERS[x][0])
        st.caption(RETRIEVERS[selected][1])

        st.divider()
        st.subheader("📋 Retrieval")
        k = st.slider("Top-K pages", 1, 10, 1)

        st.divider()
        st.subheader("🤖 Generation")
        retrieval_only = st.checkbox(
            "Retrieval only (skip VLM)", value=not ollama_ok,
            help="Skip Ollama generation; show retrieved pages only")
        gen_k = st.slider("Pages sent to VLM", 1, 3, 1)
        gen_size = st.slider("Max image size (px)", 256, 1536, 768, step=128)

        st.divider()
        st.subheader("📚 Compare retrievers")
        compare = st.checkbox("Side-by-side comparison", value=False)
        compare_with = st.selectbox(
            "Compare with", [r for r in RETRIEVERS if r != selected],
            format_func=lambda x: RETRIEVERS[x][0],
            disabled=not compare)

        st.divider()
        st.subheader("📤 Corpus")
        upload_tab()

    if compare:
        st.header("🆚 Side-by-side comparison")

    tab_a, tab_b = (st.tabs([f"Retriever A: {RETRIEVERS[selected][0]}",
                             f"Retriever B: {RETRIEVERS[compare_with][0]}"])
                    if compare else [st.container(), None])

    def run_qa(name, container):
        with container:
            question = st.text_area(
                "Question", key=f"q_{name}",
                placeholder="What does Figure 3 show?")
            if not st.button("🔍 Ask", type="primary", key=f"ask_{name}"):
                return
            if not question.strip():
                st.warning("Enter a question."); return
            with st.spinner(f"Retrieving with {RETRIEVERS[name][0]}..."):
                t0 = time.time()
                try:
                    retriever = get_retriever(name)
                    results = retriever.retrieve(question, k)
                except Exception as e:
                    st.error(f"Retriever failed: {e}"); return
                rtime = time.time() - t0
            if not results:
                st.warning("No pages found."); return
            st.success(f"{len(results)} pages in {rtime:.2f}s")
            items = []
            for pid, score in results:
                p = page_image(pid)
                if p.exists():
                    items.append((pid, score, p))
            cols = st.columns(min(3, max(len(items), 1)))
            for i, (pid, score, p) in enumerate(items):
                with cols[i % 3]:
                    img = Image.open(p)
                    img.thumbnail((400, 600), Image.LANCZOS)
                    st.image(img, caption=f"{pid} ({score:.3f})",
                             use_container_width=True)

            if not retrieval_only:
                st.divider()
                st.subheader("🤖 Generated Answer")
                if not ollama_ok:
                    st.error(ollama_msg)
                    st.info("Start Ollama or enable 'Retrieval only'.")
                    return
                with st.spinner("Generating... (may take ~2 min on CPU)"):
                    imp = _imports()
                    try:
                        gen = imp["Generator"](get_config())
                        paths = [p for _, _, p in items[:gen_k]]
                        gen.config = getattr(gen, "config", {})
                        if isinstance(gen.config, dict):
                            gen.config.setdefault("generator", {})["max_image_size"] = gen_size
                        answer, _ = gen.generate(question, paths, [pid for pid, _, _ in items[:gen_k]])
                        st.markdown(f"**Answer:** {answer}")
                    except imp["GeneratorError"] as e:
                        st.error(f"Generation error: {e}")
                    except Exception as e:
                        st.error(f"Generation failed: {e}")

    run_qa(selected, tab_a)
    if compare and tab_b is not None:
        run_qa(compare_with, tab_b)


if __name__ == "__main__":
    main()
