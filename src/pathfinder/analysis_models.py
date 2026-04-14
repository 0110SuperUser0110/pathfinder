from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import slugify, utc_now_iso


def _coerce_path(value: str | Path) -> str:
    return str(Path(value))


@dataclass(slots=True)
class BaselineWindowSpec:
    start_offset_seconds: float
    end_offset_seconds: float

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.end_offset_seconds <= self.start_offset_seconds:
            errors.append("baseline_window end_offset_seconds must be greater than start_offset_seconds")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_offset_seconds": self.start_offset_seconds,
            "end_offset_seconds": self.end_offset_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaselineWindowSpec":
        return cls(
            start_offset_seconds=float(data["start_offset_seconds"]),
            end_offset_seconds=float(data["end_offset_seconds"]),
        )


@dataclass(slots=True)
class RecordingReference:
    recording_id: str
    subject_id: str
    session_id: str
    label_namespace: str
    source_path: str
    source_format: str
    channel_names: list[str]
    sampling_rate_hz: float
    n_samples: int
    duration_seconds: float
    source_provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.recording_id:
            errors.append("recording.recording_id is required")
        if not self.source_path:
            errors.append("recording.source_path is required")
        if not self.source_format:
            errors.append("recording.source_format is required")
        if not self.channel_names:
            errors.append("recording.channel_names must not be empty")
        if self.sampling_rate_hz <= 0:
            errors.append("recording.sampling_rate_hz must be greater than 0")
        if self.n_samples <= 0:
            errors.append("recording.n_samples must be greater than 0")
        if self.duration_seconds <= 0:
            errors.append("recording.duration_seconds must be greater than 0")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "label_namespace": self.label_namespace,
            "source_path": self.source_path,
            "source_format": self.source_format,
            "channel_names": list(self.channel_names),
            "sampling_rate_hz": self.sampling_rate_hz,
            "n_samples": self.n_samples,
            "duration_seconds": self.duration_seconds,
            "source_provenance": dict(self.source_provenance),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecordingReference":
        return cls(
            recording_id=data["recording_id"],
            subject_id=data.get("subject_id", ""),
            session_id=data.get("session_id", ""),
            label_namespace=data.get("label_namespace", ""),
            source_path=_coerce_path(data["source_path"]),
            source_format=data["source_format"],
            channel_names=list(data["channel_names"]),
            sampling_rate_hz=float(data["sampling_rate_hz"]),
            n_samples=int(data["n_samples"]),
            duration_seconds=float(data["duration_seconds"]),
            source_provenance=dict(data.get("source_provenance", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class EventRecord:
    event_id: str
    recording_id: str
    onset_seconds: float
    duration_seconds: float
    event_family: str
    target_label: str
    event_subtype: str = ""
    label_namespace: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def end_seconds(self) -> float:
        return self.onset_seconds + self.duration_seconds

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.event_id:
            errors.append("event.event_id is required")
        if not self.recording_id:
            errors.append(f"event {self.event_id!r} recording_id is required")
        if self.onset_seconds < 0:
            errors.append(f"event {self.event_id!r} onset_seconds must be >= 0")
        if self.duration_seconds < 0:
            errors.append(f"event {self.event_id!r} duration_seconds must be >= 0")
        if not self.event_family:
            errors.append(f"event {self.event_id!r} event_family is required")
        if not self.target_label:
            errors.append(f"event {self.event_id!r} target_label is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "recording_id": self.recording_id,
            "onset_seconds": self.onset_seconds,
            "duration_seconds": self.duration_seconds,
            "event_family": self.event_family,
            "target_label": self.target_label,
            "event_subtype": self.event_subtype,
            "label_namespace": self.label_namespace,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventRecord":
        return cls(
            event_id=data["event_id"],
            recording_id=data["recording_id"],
            onset_seconds=float(data["onset_seconds"]),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            event_family=data["event_family"],
            target_label=data["target_label"],
            event_subtype=data.get("event_subtype", ""),
            label_namespace=data.get("label_namespace", ""),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class EpochWindowConfig:
    pre_event_seconds: float = 0.0
    onset_seconds: float = 0.0
    offset_seconds: float = 0.0
    post_event_seconds: float = 0.0
    sustained_seconds: float | None = None
    baseline_window: BaselineWindowSpec | None = None

    def validate(self) -> list[str]:
        errors: list[str] = []
        for field_name in ("pre_event_seconds", "onset_seconds", "offset_seconds", "post_event_seconds"):
            value = getattr(self, field_name)
            if value < 0:
                errors.append(f"window_config.{field_name} must be >= 0")
        if self.sustained_seconds is not None and self.sustained_seconds < 0:
            errors.append("window_config.sustained_seconds must be >= 0 when provided")
        if self.baseline_window is not None:
            errors.extend(self.baseline_window.validate())
        return errors

    def slug(self) -> str:
        parts = [
            f"pre_{self.pre_event_seconds:g}s",
            f"on_{self.onset_seconds:g}s",
            f"off_{self.offset_seconds:g}s",
            f"post_{self.post_event_seconds:g}s",
        ]
        if self.sustained_seconds is not None:
            parts.append(f"sustain_{self.sustained_seconds:g}s")
        if self.baseline_window is not None:
            parts.append(
                f"base_{self.baseline_window.start_offset_seconds:g}_{self.baseline_window.end_offset_seconds:g}s"
            )
        return slugify("__".join(parts), "epoch_window")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pre_event_seconds": self.pre_event_seconds,
            "onset_seconds": self.onset_seconds,
            "offset_seconds": self.offset_seconds,
            "post_event_seconds": self.post_event_seconds,
            "sustained_seconds": self.sustained_seconds,
        }
        if self.baseline_window is not None:
            payload["baseline_window"] = self.baseline_window.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpochWindowConfig":
        baseline_data = data.get("baseline_window")
        baseline = BaselineWindowSpec.from_dict(baseline_data) if baseline_data else None
        sustained_seconds = data.get("sustained_seconds")
        return cls(
            pre_event_seconds=float(data.get("pre_event_seconds", 0.0)),
            onset_seconds=float(data.get("onset_seconds", 0.0)),
            offset_seconds=float(data.get("offset_seconds", 0.0)),
            post_event_seconds=float(data.get("post_event_seconds", 0.0)),
            sustained_seconds=None if sustained_seconds is None else float(sustained_seconds),
            baseline_window=baseline,
        )


@dataclass(slots=True)
class EpochPhaseRange:
    phase_name: str
    requested_start_seconds: float
    requested_end_seconds: float
    actual_start_seconds: float
    actual_end_seconds: float
    start_sample: int
    end_sample: int
    derived_samples: int
    clipped: bool = False
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "requested_start_seconds": self.requested_start_seconds,
            "requested_end_seconds": self.requested_end_seconds,
            "actual_start_seconds": self.actual_start_seconds,
            "actual_end_seconds": self.actual_end_seconds,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "derived_samples": self.derived_samples,
            "clipped": self.clipped,
            "warning": self.warning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpochPhaseRange":
        return cls(
            phase_name=data["phase_name"],
            requested_start_seconds=float(data["requested_start_seconds"]),
            requested_end_seconds=float(data["requested_end_seconds"]),
            actual_start_seconds=float(data["actual_start_seconds"]),
            actual_end_seconds=float(data["actual_end_seconds"]),
            start_sample=int(data["start_sample"]),
            end_sample=int(data["end_sample"]),
            derived_samples=int(data.get("derived_samples", int(data["end_sample"]) - int(data["start_sample"]))),
            clipped=bool(data.get("clipped", False)),
            warning=data.get("warning", ""),
        )


@dataclass(slots=True)
class EpochArtifactReference:
    event: EventRecord
    signal_path: str
    format: str
    phase_ranges: list[EpochPhaseRange]
    baseline_range: EpochPhaseRange | None = None
    phase_shapes: dict[str, list[int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event": self.event.to_dict(),
            "signal_path": self.signal_path,
            "format": self.format,
            "phase_ranges": [item.to_dict() for item in self.phase_ranges],
            "phase_shapes": {key: list(value) for key, value in self.phase_shapes.items()},
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }
        if self.baseline_range is not None:
            payload["baseline_range"] = self.baseline_range.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpochArtifactReference":
        baseline_data = data.get("baseline_range")
        return cls(
            event=EventRecord.from_dict(data["event"]),
            signal_path=_coerce_path(data["signal_path"]),
            format=data["format"],
            phase_ranges=[EpochPhaseRange.from_dict(item) for item in data["phase_ranges"]],
            baseline_range=EpochPhaseRange.from_dict(baseline_data) if baseline_data else None,
            phase_shapes={key: list(value) for key, value in data.get("phase_shapes", {}).items()},
            warnings=list(data.get("warnings", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class EpochCollection:
    collection_id: str
    recording: RecordingReference
    window_config: EpochWindowConfig
    channel_names: list[str]
    sampling_rate_hz: float
    artifacts: list[EpochArtifactReference]
    created_at_utc: str = field(default_factory=utc_now_iso)
    source_event_table_path: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors = self.recording.validate()
        errors.extend(self.window_config.validate())
        if not self.collection_id:
            errors.append("epoch_collection.collection_id is required")
        if not self.channel_names:
            errors.append("epoch_collection.channel_names must not be empty")
        if self.sampling_rate_hz <= 0:
            errors.append("epoch_collection.sampling_rate_hz must be greater than 0")
        if not self.artifacts:
            errors.append("epoch_collection.artifacts must not be empty")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_id": self.collection_id,
            "recording": self.recording.to_dict(),
            "window_config": self.window_config.to_dict(),
            "channel_names": list(self.channel_names),
            "sampling_rate_hz": self.sampling_rate_hz,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "created_at_utc": self.created_at_utc,
            "source_event_table_path": self.source_event_table_path,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpochCollection":
        return cls(
            collection_id=data["collection_id"],
            recording=RecordingReference.from_dict(data["recording"]),
            window_config=EpochWindowConfig.from_dict(data["window_config"]),
            channel_names=list(data["channel_names"]),
            sampling_rate_hz=float(data["sampling_rate_hz"]),
            artifacts=[EpochArtifactReference.from_dict(item) for item in data["artifacts"]],
            created_at_utc=data.get("created_at_utc", utc_now_iso()),
            source_event_table_path=data.get("source_event_table_path", ""),
            notes=data.get("notes", ""),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class TransformRecord:
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    applied_at_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": dict(self.parameters),
            "warnings": list(self.warnings),
            "applied_at_utc": self.applied_at_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformRecord":
        return cls(
            name=data["name"],
            parameters=dict(data.get("parameters", {})),
            warnings=list(data.get("warnings", [])),
            applied_at_utc=data.get("applied_at_utc", utc_now_iso()),
        )


@dataclass(slots=True)
class ChannelRemapReport:
    original_channels: list[str]
    output_channels: list[str]
    missing_channels: list[str] = field(default_factory=list)
    dropped_channels: list[str] = field(default_factory=list)
    reused_channels: list[str] = field(default_factory=list)
    fill_value: str = "nan"

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_channels": list(self.original_channels),
            "output_channels": list(self.output_channels),
            "missing_channels": list(self.missing_channels),
            "dropped_channels": list(self.dropped_channels),
            "reused_channels": list(self.reused_channels),
            "fill_value": self.fill_value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChannelRemapReport":
        return cls(
            original_channels=list(data.get("original_channels", [])),
            output_channels=list(data.get("output_channels", [])),
            missing_channels=list(data.get("missing_channels", [])),
            dropped_channels=list(data.get("dropped_channels", [])),
            reused_channels=list(data.get("reused_channels", [])),
            fill_value=data.get("fill_value", "nan"),
        )


@dataclass(slots=True)
class PreprocessBranchConfig:
    branch_name: str
    notch_hz: list[float] = field(default_factory=list)
    notch_bandwidth_hz: float = 1.0
    resample_hz: float | None = None
    baseline_mode: str = "none"
    align_channels: list[str] = field(default_factory=list)
    rereference_mode: str = "none"
    reference_channels: list[str] = field(default_factory=list)
    scale_factor: float | None = None

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.branch_name:
            errors.append("preprocess.branch_name is required")
        if self.notch_bandwidth_hz <= 0:
            errors.append("preprocess.notch_bandwidth_hz must be greater than 0")
        if self.resample_hz is not None and self.resample_hz <= 0:
            errors.append("preprocess.resample_hz must be greater than 0 when provided")
        if self.baseline_mode not in {"none", "metadata_only", "subtract_mean"}:
            errors.append("preprocess.baseline_mode must be one of: none, metadata_only, subtract_mean")
        if self.rereference_mode not in {"none", "average", "channels"}:
            errors.append("preprocess.rereference_mode must be one of: none, average, channels")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_name": self.branch_name,
            "notch_hz": list(self.notch_hz),
            "notch_bandwidth_hz": self.notch_bandwidth_hz,
            "resample_hz": self.resample_hz,
            "baseline_mode": self.baseline_mode,
            "align_channels": list(self.align_channels),
            "rereference_mode": self.rereference_mode,
            "reference_channels": list(self.reference_channels),
            "scale_factor": self.scale_factor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessBranchConfig":
        resample_hz = data.get("resample_hz")
        scale_factor = data.get("scale_factor")
        return cls(
            branch_name=data["branch_name"],
            notch_hz=[float(item) for item in data.get("notch_hz", [])],
            notch_bandwidth_hz=float(data.get("notch_bandwidth_hz", 1.0)),
            resample_hz=None if resample_hz is None else float(resample_hz),
            baseline_mode=data.get("baseline_mode", "none"),
            align_channels=list(data.get("align_channels", [])),
            rereference_mode=data.get("rereference_mode", "none"),
            reference_channels=list(data.get("reference_channels", [])),
            scale_factor=None if scale_factor is None else float(scale_factor),
        )


@dataclass(slots=True)
class PreprocessBranchResult:
    branch_name: str
    source_collection_path: str
    output_collection_path: str
    config: PreprocessBranchConfig
    transforms: list[TransformRecord]
    warnings: list[str] = field(default_factory=list)
    channel_report: ChannelRemapReport | None = None
    created_at_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "branch_name": self.branch_name,
            "source_collection_path": self.source_collection_path,
            "output_collection_path": self.output_collection_path,
            "config": self.config.to_dict(),
            "transforms": [item.to_dict() for item in self.transforms],
            "warnings": list(self.warnings),
            "created_at_utc": self.created_at_utc,
        }
        if self.channel_report is not None:
            payload["channel_report"] = self.channel_report.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessBranchResult":
        channel_report_data = data.get("channel_report")
        return cls(
            branch_name=data["branch_name"],
            source_collection_path=_coerce_path(data["source_collection_path"]),
            output_collection_path=_coerce_path(data["output_collection_path"]),
            config=PreprocessBranchConfig.from_dict(data["config"]),
            transforms=[TransformRecord.from_dict(item) for item in data["transforms"]],
            warnings=list(data.get("warnings", [])),
            channel_report=ChannelRemapReport.from_dict(channel_report_data) if channel_report_data else None,
            created_at_utc=data.get("created_at_utc", utc_now_iso()),
        )


@dataclass(slots=True)
class SubjectTrialGroup:
    subject_id: str
    target_label: str
    event_ids: list[str]
    artifact_paths: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "target_label": self.target_label,
            "event_ids": list(self.event_ids),
            "artifact_paths": list(self.artifact_paths),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubjectTrialGroup":
        return cls(
            subject_id=data["subject_id"],
            target_label=data["target_label"],
            event_ids=list(data.get("event_ids", [])),
            artifact_paths=list(data.get("artifact_paths", [])),
            metadata=dict(data.get("metadata", {})),
        )
