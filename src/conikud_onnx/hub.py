"""Locating the model file: a local path, or the Hub copy fetched on first use."""

from __future__ import annotations

from pathlib import Path

REPO = "conikud/conikud-onnx"
FILENAME = "conikud_int8.onnx"


def resolve(model_path: str | Path | None = None) -> str:
    """Return a local .onnx path, downloading the Hub model when none is given.

    The download is cached by huggingface_hub, so only the first call needs a
    network; set HF_HOME to move the cache.
    """
    if model_path is not None:
        return str(model_path)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=REPO, filename=FILENAME)
