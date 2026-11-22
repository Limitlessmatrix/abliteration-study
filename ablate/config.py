"""Paths and service endpoints, read from environment variables with local defaults.

Nothing in the library hard-codes a cluster, a volume or a host. Everything that used to be a fixed path in the
original research setup is resolved here:

  ABLATE_DATA         directory with the built search sets (sets/*.jsonl); default <repo>/data
  ABLATE_OLLAMA_HOST  base URL of an Ollama server, used only by the optional twin generator and the optional
                      Ollama judge; default http://localhost:11434
  ABLATE_K_MAX        top of the rank range the search may pick (read in ablate.search); default 4

The judge model used by the study is pinned here so the search and the evaluation use the same revision.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Qwen3Guard-Gen-8B at the revision every reported number was judged with.
GUARD_REPO = "Qwen/Qwen3Guard-Gen-8B"
GUARD_REVISION = "4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb"


def data_dir() -> Path:
    return Path(os.environ.get("ABLATE_DATA") or (ROOT / "data"))


def ollama_host() -> str:
    return (os.environ.get("ABLATE_OLLAMA_HOST") or "http://localhost:11434").rstrip("/")


def resolve_model_dir(spec: str) -> str:
    """A local directory is returned as is. "org/name@revision" or "org/name" is downloaded from the Hugging Face
    Hub (snapshot_download, cached in HF_HOME) and the local snapshot path is returned, so a pinned revision is
    what actually gets loaded."""
    if not spec or Path(spec).exists():
        return spec
    repo, _, rev = spec.partition("@")
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=rev or None)
