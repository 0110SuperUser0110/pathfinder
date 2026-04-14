from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from .models import slugify, utc_now_iso


@dataclass(slots=True)
class RunArtifactRecord:
    artifact_type: str
    path: str
    role: str = ""
    format: str = ""
    parent_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "path": self.path,
            "role": self.role,
            "format": self.format,
            "parent_paths": list(self.parent_paths),
            "metadata": dict(self.metadata),
        }


class StructuredRunLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, level: str, stage: str, message: str, **context: Any) -> None:
        payload = {
            "timestamp_utc": utc_now_iso(),
            "level": level,
            "stage": stage,
            "message": message,
            "context": context,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


DEFAULT_OPTIONAL_DEPENDENCIES = ["mne", "numpy", "torch"]


def make_run_id(prefix: str, requested: str = "") -> str:
    if requested.strip():
        return slugify(requested, prefix)
    timestamp = utc_now_iso().replace(":", "").replace("-", "")
    return slugify(f"{prefix}_{timestamp}", prefix)


def _json_write(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _git_commit(cwd: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return completed.stdout.strip()


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return ""


def environment_snapshot(
    *,
    code_root: str | Path,
    optional_dependencies: list[str] | None = None,
) -> dict[str, Any]:
    code_path = Path(code_root).resolve()
    dependencies = optional_dependencies or DEFAULT_OPTIONAL_DEPENDENCIES
    return {
        "captured_at_utc": utc_now_iso(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": _git_commit(code_path),
        "optional_dependencies": {
            name: {
                "available": find_spec(name) is not None,
                "version": _package_version(name),
            }
            for name in dependencies
        },
    }


def artifact_lineage_payload(
    *,
    source_artifacts: list[RunArtifactRecord],
    generated_artifacts: list[RunArtifactRecord],
) -> dict[str, Any]:
    return {
        "captured_at_utc": utc_now_iso(),
        "source_artifacts": [item.to_dict() for item in source_artifacts],
        "generated_artifacts": [item.to_dict() for item in generated_artifacts],
    }


def write_run_bundle(
    *,
    run_root: str | Path,
    run_id: str,
    operation: str,
    command: str,
    config_snapshot: dict[str, Any],
    source_artifacts: list[RunArtifactRecord],
    generated_artifacts: list[RunArtifactRecord],
    code_root: str | Path,
    rng_seed: int | None = None,
    active_branches: list[str] | None = None,
    warnings: list[str] | None = None,
    status: str = "success",
    notes: list[str] | None = None,
    optional_dependencies: list[str] | None = None,
) -> dict[str, str]:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    config_path = _json_write(root / "config_snapshot.json", config_snapshot)
    environment_path = _json_write(
        root / "environment.json",
        environment_snapshot(code_root=code_root, optional_dependencies=optional_dependencies),
    )
    lineage_path = _json_write(
        root / "artifact_lineage.json",
        artifact_lineage_payload(source_artifacts=source_artifacts, generated_artifacts=generated_artifacts),
    )
    warnings_list = sorted(set(warnings or []))
    warnings_path = root / "warnings.json"
    if warnings_list:
        _json_write(
            warnings_path,
            {
                "run_id": run_id,
                "warnings": warnings_list,
            },
        )
    log_path = root / "run.log.jsonl"
    if not log_path.exists():
        log_path.touch()
    manifest = {
        "run_id": run_id,
        "operation": operation,
        "command": command,
        "status": status,
        "created_at_utc": utc_now_iso(),
        "rng_seed": rng_seed,
        "active_branches": list(active_branches or []),
        "config_snapshot_path": str(config_path),
        "environment_path": str(environment_path),
        "artifact_lineage_path": str(lineage_path),
        "warnings_path": str(warnings_path) if warnings_list else "",
        "log_path": str(log_path),
        "generated_artifact_count": len(generated_artifacts),
        "source_artifact_count": len(source_artifacts),
        "notes": list(notes or []),
    }
    manifest_path = _json_write(root / "run_manifest.json", manifest)
    return {
        "run_manifest_path": str(manifest_path),
        "config_snapshot_path": str(config_path),
        "environment_path": str(environment_path),
        "artifact_lineage_path": str(lineage_path),
        "warnings_path": str(warnings_path) if warnings_list else "",
        "log_path": str(log_path),
    }
