from __future__ import annotations

from pathlib import Path


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def external_root() -> Path:
    return workspace_root() / "external"
