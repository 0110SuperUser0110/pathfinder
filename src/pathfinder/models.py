from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RECOMMENDED_FORMATS = {
    "bdf",
    "csv",
    "edf",
    "fif",
    "hdf5",
    "json",
    "jsonl",
    "md",
    "npy",
    "npz",
    "parquet",
    "png",
    "tsv",
    "zarr",
}

SIGNAL_REPRESENTATIONS = {"raw_eeg", "processed_epoch"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str, default: str = "unknown") -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return collapsed or default


@dataclass(slots=True)
class PartitionSpec:
    study_ids: list[str] = field(default_factory=list)
    event_family: str = ""
    target_label: str = ""
    event_subtype: str = ""
    label_namespace: str = ""
    biological_sex: str = ""
    gender_identity: str = ""
    stimulus_modality: str = ""
    age_band: str = ""
    cohort_label: str = ""

    def to_path_parts(self) -> list[str]:
        return [
            slugify(self.event_family, "unspecified_event_family"),
            slugify(self.biological_sex, "all_sexes"),
            slugify(self.stimulus_modality, "all_modalities"),
        ]

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.event_family:
            errors.append("partition.event_family is required")
        if not self.target_label:
            errors.append("partition.target_label is required")
        return errors

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PartitionSpec":
        payload = dict(data)
        if "event_family" not in payload and "sensory_domain" in payload:
            payload["event_family"] = payload.pop("sensory_domain")
        if "target_label" not in payload and "condition_label" in payload:
            payload["target_label"] = payload.pop("condition_label")
        if "event_subtype" not in payload and "arousal_type" in payload:
            payload["event_subtype"] = payload.pop("arousal_type")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_ids": list(self.study_ids),
            "event_family": self.event_family,
            "target_label": self.target_label,
            "event_subtype": self.event_subtype,
            "label_namespace": self.label_namespace,
            "biological_sex": self.biological_sex,
            "gender_identity": self.gender_identity,
            "stimulus_modality": self.stimulus_modality,
            "age_band": self.age_band,
            "cohort_label": self.cohort_label,
        }


@dataclass(slots=True)
class AnalysisSpec:
    discovery_mode: str
    source_models: list[str]
    created_at_utc: str = field(default_factory=utc_now_iso)
    notes: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.discovery_mode:
            errors.append("analysis.discovery_mode is required")
        if not self.source_models:
            errors.append("analysis.source_models must not be empty")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovery_mode": self.discovery_mode,
            "source_models": list(self.source_models),
            "created_at_utc": self.created_at_utc,
            "notes": self.notes,
        }


@dataclass(slots=True)
class PatternSummary:
    candidate_signature: str = ""
    bands: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    temporal_notes: str = ""
    reproducibility_score: float | None = None
    cross_model_support: float | None = None
    cross_subject_support: float | None = None
    artifact_risk: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        for field_name in ("reproducibility_score", "cross_model_support", "cross_subject_support"):
            value = getattr(self, field_name)
            if value is not None and not 0.0 <= value <= 1.0:
                errors.append(f"summary.{field_name} must be between 0 and 1 when provided")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_signature": self.candidate_signature,
            "bands": list(self.bands),
            "channels": list(self.channels),
            "temporal_notes": self.temporal_notes,
            "reproducibility_score": self.reproducibility_score,
            "cross_model_support": self.cross_model_support,
            "cross_subject_support": self.cross_subject_support,
            "artifact_risk": self.artifact_risk,
        }


@dataclass(slots=True)
class ArtifactRecord:
    artifact_id: str
    role: str
    representation: str
    format: str
    path: str
    description: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.artifact_id:
            errors.append("artifact.artifact_id is required")
        if not self.role:
            errors.append(f"artifact {self.artifact_id!r} role is required")
        if not self.representation:
            errors.append(f"artifact {self.artifact_id!r} representation is required")
        if not self.format:
            errors.append(f"artifact {self.artifact_id!r} format is required")
        if not self.path:
            errors.append(f"artifact {self.artifact_id!r} path is required")
        if Path(self.path).is_absolute():
            errors.append(f"artifact {self.artifact_id!r} path must be relative inside the package")
        if self.format.lower() not in RECOMMENDED_FORMATS:
            errors.append(
                f"artifact {self.artifact_id!r} format {self.format!r} is not in the recommended format set"
            )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "role": self.role,
            "representation": self.representation,
            "format": self.format,
            "path": self.path,
            "description": self.description,
        }


@dataclass(slots=True)
class PatternManifest:
    pattern_id: str
    partition: PartitionSpec
    analysis: AnalysisSpec
    artifacts: list[ArtifactRecord]
    summary: PatternSummary = field(default_factory=PatternSummary)
    schema_version: str = "1.1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "partition": self.partition.to_dict(),
            "analysis": self.analysis.to_dict(),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "summary": self.summary.to_dict(),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatternManifest":
        partition = PartitionSpec.from_dict(data["partition"])
        analysis = AnalysisSpec(**data["analysis"])
        artifacts = [ArtifactRecord(**item) for item in data["artifacts"]]
        summary = PatternSummary(**data.get("summary", {}))
        return cls(
            pattern_id=data["pattern_id"],
            partition=partition,
            analysis=analysis,
            artifacts=artifacts,
            summary=summary,
            schema_version=data.get("schema_version", "1.0"),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.pattern_id:
            errors.append("pattern_id is required")
        errors.extend(self.partition.validate())
        errors.extend(self.analysis.validate())
        errors.extend(self.summary.validate())

        if not self.artifacts:
            errors.append("at least one artifact is required")

        seen_ids: set[str] = set()
        seen_paths: set[str] = set()
        has_signal_artifact = False
        for artifact in self.artifacts:
            errors.extend(artifact.validate())
            if artifact.artifact_id in seen_ids:
                errors.append(f"duplicate artifact_id {artifact.artifact_id!r}")
            seen_ids.add(artifact.artifact_id)
            if artifact.path in seen_paths:
                errors.append(f"duplicate artifact path {artifact.path!r}")
            seen_paths.add(artifact.path)
            if artifact.representation in SIGNAL_REPRESENTATIONS:
                has_signal_artifact = True

        if not has_signal_artifact:
            errors.append("at least one artifact must be a signal artifact")
        return errors
