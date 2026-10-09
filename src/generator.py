# VisionRAG Generator
# Calls Ollama VLM with page images and question, returns answer with citations
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from PIL import Image

from src.utils import load_config


class GeneratorError(Exception):
    """Custom exception for generator errors."""
    pass


class Generator:
    """Generator that calls Ollama VLM with page images and question."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        gen_cfg = config.get("generator", {})

        self.model_name = gen_cfg.get("model_name", "qwen2.5vl:3b")
        self.host = gen_cfg.get("host", "http://localhost:11434")
        self.max_image_size = gen_cfg.get("max_image_size", 1024)
        self.num_ctx = gen_cfg.get("num_ctx", 4096)
        self.timeout_seconds = gen_cfg.get("timeout_seconds", 120)
        self.keep_alive = gen_cfg.get("keep_alive", "5m")
        self.temperature = gen_cfg.get("temperature", 0.1)
        self.top_p = gen_cfg.get("top_p", 0.9)
        self.generation_k = gen_cfg.get("generation_k", 2)

        self.client = httpx.Client(timeout=self.timeout_seconds)
        self._api_base = f"{self.host}/api"

    def _downscale_image(self, image_path: Path) -> str:
        """Load image, downscale if needed, return base64 encoded PNG."""
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Downscale if longest side exceeds max_image_size
            max_dim = max(img.width, img.height)
            if max_dim > self.max_image_size:
                scale = self.max_image_size / max_dim
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Encode to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_prompt(self, question: str, page_ids: List[str]) -> str:
        """Build the prompt for the VLM."""
        pages_list = ", ".join(page_ids)
        return f"""You are a helpful assistant that answers questions based ONLY on the provided page images.

Question: {question}

You are given {len(page_ids)} page image(s) from a document. The page identifiers are: {pages_list}.

Instructions:
1. Answer the question using ONLY the information visible in the provided page images.
2. Cite the specific page identifier(s) for each piece of information you use (e.g., "According to paper1_page_0004...").
3. If the answer is not found in the provided pages, respond with exactly: "not found in the provided pages"
4. Do not use any external knowledge.
5. Be concise but complete.

Answer:"""

    def _call_ollama(self, prompt: str, images: List[str]) -> str:
        """Call Ollama API and return the response text."""
        url = f"{self._api_base}/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": images,
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
            "keep_alive": self.keep_alive,
        }

        try:
            response = self.client.post(url, json=payload)
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise GeneratorError(
                f"Cannot connect to Ollama at {self.host}. Is Ollama running?"
            ) from e
        except httpx.TimeoutException as e:
            raise GeneratorError(
                f"Request to Ollama timed out after {self.timeout_seconds} seconds"
            ) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise GeneratorError(
                    f"Model '{self.model_name}' not found in Ollama. Run: ollama pull {self.model_name}"
                ) from e
            raise GeneratorError(f"Ollama API error: {e.response.status_code} {e.response.text}") from e

        try:
            data = response.json()
            return data.get("response", "").strip()
        except json.JSONDecodeError as e:
            raise GeneratorError(f"Invalid JSON response from Ollama: {e}") from e

    def generate(
        self,
        question: str,
        page_image_paths: List[Path],
        page_ids: List[str],
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """Generate answer from page images and question.

        Returns:
            Tuple of (answer_text, list of (page_id, score))
        """
        if not page_image_paths:
            return "not found in the provided pages", []

        # Limit to generation_k images
        k = min(self.generation_k, len(page_image_paths))
        selected_paths = page_image_paths[:k]
        selected_ids = page_ids[:k]

        # Downscale and encode images
        images_b64 = []
        for img_path in selected_paths:
            try:
                images_b64.append(self._downscale_image(img_path))
            except Exception as e:
                raise GeneratorError(f"Failed to process image {img_path}: {e}") from e

        # Build prompt
        prompt = self._build_prompt(question, selected_ids)

        # Call Ollama
        answer = self._call_ollama(prompt, images_b64)

        # Return answer with dummy scores (scores come from retriever)
        # The actual scores are tracked by the pipeline
        return answer, [(pid, 0.0) for pid in selected_ids]

    def close(self):
        """Close the HTTP client."""
        self.client.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="VisionRAG Generator")
    parser.add_argument("--question", type=str, required=True, help="Question to answer")
    parser.add_argument(
        "--images", type=str, nargs="+", required=True, help="Paths to page images"
    )
    parser.add_argument(
        "--page_ids", type=str, nargs="+", required=True, help="Page IDs corresponding to images"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    generator = Generator(config)

    try:
        image_paths = [Path(p) for p in args.images]
        answer, _ = generator.generate(args.question, image_paths, args.page_ids)
        print(answer)
    finally:
        generator.close()


if __name__ == "__main__":
    main()
