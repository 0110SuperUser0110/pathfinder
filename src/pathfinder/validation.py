from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .analysis_models import EpochCollection, EventRecord, PreprocessBranchResult, RecordingReference
from .artifact_contracts import ARTIFACT_CONTRACTS
from .discovery_models import CandidatePattern, DiscoveryRunSummary
from .epochs import load_epoch_collection
from .ingest import load_normalized_event_table, load_recording_reference
from .package import PatternPackageBuilder
from .preprocess import load_branch_result


@dataclass(slots=True)
class ValidationFinding:
    level: str
    code: str
    message: str
    location: str = ""
    path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "path": self.path,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ValidationReport:
    artifact_type: str
    target_path: str
    errors: list[ValidationFinding] = field(default_factory=list)
    warnings: list[ValidationFinding] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, code: str, message: str, *, location: str = "", path: str = "", metadata: dict[str, Any] | None = None) -> None:
        self.errors.append(
            ValidationFinding(
                level="error",
                code=code,
                message=message,
                location=location,
                path=path,
                metadata=dict(metadata or {}),
            )
        )

    def add_warning(self, code: str, message: str, *, location: str = "", path: str = "", metadata: dict[str, Any] | None = None) -> None:
        self.warnings.append(
            ValidationFinding(
                level="warning",
                code=code,
                message=message,
                location=location,
                path=path,
                metadata=dict(metadata or {}),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "target_path": self.target_path,
            "ok": self.ok,
            "contract": ARTIFACT_CONTRACTS[self.artifact_type].to_dict() if self.artifact_type in ARTIFACT_CONTRACTS else {},
            "errors": [item.to_dict() for item in self.errors],
            "warnings": [item.to_dict() for item in self.warnings],
            "summary": dict(self.summary),
        }

    def render_lines(self) -> list[str]:
        lines = [f"artifact_type: {self.artifact_type}", f"target_path: {self.target_path}", f"ok: {'yes' if self.ok else 'no'}"]
        if self.summary:
            lines.append("summary:")
            for key in sorted(self.summary):
                lines.append(f"  {key}: {self.summary[key]}")
        if self.errors:
            lines.append("errors:")
            for item in self.errors:
                prefix = f" [{item.code}]" if item.code else ""
                location = f" ({item.location})" if item.location else ""
                lines.append(f"  -{prefix} {item.message}{location}")
        if self.warnings:
            lines.append("warnings:")
            for item in self.warnings:
                prefix = f" [{item.code}]" if item.code else ""
                location = f" ({item.location})" if item.location else ""
                lines.append(f"  -{prefix} {item.message}{location}")
        return lines


def _ensure_relative_paths_exist(base_dir: Path, relative_paths: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for relative_path in relative_paths:
        if not (base_dir / relative_path).exists():
            missing.append(relative_path)
    return missing


def _duplicate_values(values: Iterable[str]) -> list[str]:
    counter = Counter(item for item in values if item)
    return sorted([item for item, count in counter.items() if count > 1])


def _channel_name_errors(channel_names: list[str]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not channel_names:
        errors.append("channel_names must not be empty")
        return errors, warnings
    duplicates = _duplicate_values(channel_names)
    if duplicates:
        errors.append("duplicate channel names: " + ", ".join(duplicates))
    blanks = [repr(name) for name in channel_names if not str(name).strip()]
    if blanks:
        errors.append("blank channel names are not allowed")
    malformed = [name for name in channel_names if any(ch.isspace() for ch in str(name))]
    if malformed:
        warnings.append("channel names contain whitespace: " + ", ".join(malformed))
    return errors, warnings


def _looks_like_bids_eeg_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return name.startswith("sub-") and "_eeg" in name and any(part.startswith("sub-") for part in parts)


def _bids_expected_sidecars(path: Path) -> list[Path]:
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.stem
    base_name = stem[:-4] if stem.endswith("_eeg") else stem
    return [
        path.with_name(f"{stem}.json"),
        path.with_name(f"{base_name}_channels.tsv"),
        path.with_name(f"{base_name}_events.tsv"),
    ]


def _load_candidate(path: str | Path) -> CandidatePattern:
    return CandidatePattern.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _load_run_summary(path: str | Path) -> DiscoveryRunSummary:
    path_obj = Path(path)
    if path_obj.is_dir():
        path_obj = path_obj / "run_summary.json"
    return DiscoveryRunSummary.from_dict(json.loads(path_obj.read_text(encoding="utf-8")))


def validate_recording_reference_artifact(path: str | Path) -> ValidationReport:
    target = Path(path).resolve()
    report = ValidationReport("recording_reference", str(target))
    recording = load_recording_reference(target)
    for message in recording.validate():
        report.add_error("recording.invalid", message)
    channel_errors, channel_warnings = _channel_name_errors(recording.channel_names)
    for message in channel_errors:
        report.add_error("recording.channels", message)
    for message in channel_warnings:
        report.add_warning("recording.channels", message)
    source_path = Path(recording.source_path)
    if not source_path.exists():
        report.add_error("recording.source_missing", "source EEG file does not exist", path=str(source_path))
    if not recording.source_provenance:
        report.add_warning("recording.provenance_missing", "source provenance is missing")
    if source_path.exists() and _looks_like_bids_eeg_path(source_path):
        for sidecar_path in _bids_expected_sidecars(source_path):
            if not sidecar_path.exists():
                report.add_warning(
                    "recording.bids_sidecar_missing",
                    f"expected BIDS sidecar is missing: {sidecar_path.name}",
                    path=str(sidecar_path),
                )
    report.summary = {
        "recording_id": recording.recording_id,
        "subject_id": recording.subject_id,
        "session_id": recording.session_id,
        "channel_count": len(recording.channel_names),
        "sampling_rate_hz": recording.sampling_rate_hz,
        "duration_seconds": recording.duration_seconds,
        "bids_like_source": bool(source_path.exists() and _looks_like_bids_eeg_path(source_path)),
    }
    return report


def validate_event_table_artifact(path: str | Path, *, recording: RecordingReference | None = None) -> ValidationReport:
    target = Path(path).resolve()
    report = ValidationReport("event_table", str(target))
    events = load_normalized_event_table(target)
    if not events:
        report.add_error("events.empty", "event table contains no events")
        return report
    duplicate_ids = _duplicate_values(event.event_id for event in events)
    if duplicate_ids:
        report.add_error("events.duplicate_id", "duplicate event IDs: " + ", ".join(duplicate_ids))
    by_recording: dict[str, list[EventRecord]] = defaultdict(list)
    for event in events:
        for message in event.validate():
            report.add_error("events.invalid", message, location=event.event_id)
        if recording is not None and event.recording_id and event.recording_id != recording.recording_id:
            report.add_error(
                "events.recording_mismatch",
                f"event belongs to recording {event.recording_id!r}, expected {recording.recording_id!r}",
                location=event.event_id,
            )
        if recording is not None and event.end_seconds > recording.duration_seconds:
            report.add_error(
                "events.out_of_bounds",
                "event extends beyond recording duration",
                location=event.event_id,
            )
        by_recording[event.recording_id or "<unknown>"].append(event)
    for recording_id, recording_events in by_recording.items():
        ordered = sorted(recording_events, key=lambda item: (item.onset_seconds, item.end_seconds, item.event_id))
        for left, right in zip(ordered, ordered[1:], strict=False):
            allow_overlap = bool(left.metadata.get("allow_overlap")) or bool(right.metadata.get("allow_overlap"))
            if not allow_overlap and right.onset_seconds < left.end_seconds:
                report.add_error(
                    "events.overlap",
                    f"events {left.event_id!r} and {right.event_id!r} overlap in recording {recording_id!r}",
                    location=recording_id,
                )
    label_counts = Counter((event.label_namespace, event.event_family, event.target_label) for event in events)
    if label_counts:
        counts = sorted(label_counts.values())
        if counts and counts[0] > 0 and counts[-1] / counts[0] >= 3.0:
            report.add_warning("events.class_imbalance", "major label imbalance detected in event counts")
    report.summary = {
        "event_count": len(events),
        "label_count": len(label_counts),
        "recording_ids": sorted({event.recording_id for event in events if event.recording_id}),
    }
    return report


def _validate_epoch_npz(path: Path, channel_count: int, sampling_rate_hz: float) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    with np.load(path, allow_pickle=False) as payload:
        if "sampling_rate_hz" in payload:
            actual_rate = float(np.asarray(payload["sampling_rate_hz"]).reshape(-1)[0])
            if not np.isclose(actual_rate, sampling_rate_hz):
                errors.append(f"sampling rate mismatch in {path.name}: {actual_rate} != {sampling_rate_hz}")
        phase_arrays = [key for key in payload.files if key not in {"channel_names", "phase_names", "band_names", "sampling_rate_hz", "event_onset_seconds", "event_duration_seconds", "subject_ids", "event_ids"}]
        if not phase_arrays:
            errors.append(f"epoch artifact {path.name} contains no signal arrays")
        for key in phase_arrays:
            value = np.asarray(payload[key])
            if value.ndim not in {2, 3}:
                errors.append(f"signal array {key!r} in {path.name} must be 2D or 3D")
                continue
            channel_axis = value.shape[-2] if value.ndim == 3 else value.shape[0]
            if channel_axis != channel_count:
                errors.append(f"signal array {key!r} in {path.name} has channel count {channel_axis}, expected {channel_count}")
            if 0 in value.shape:
                warnings.append(f"signal array {key!r} in {path.name} includes an empty axis")
    return errors, warnings


def validate_epoch_collection_artifact(path: str | Path) -> ValidationReport:
    target = Path(path).resolve()
    report = ValidationReport("epoch_collection", str(target))
    collection = load_epoch_collection(target)
    for message in collection.validate():
        report.add_error("epochs.invalid", message)
    channel_errors, channel_warnings = _channel_name_errors(collection.channel_names)
    for message in channel_errors:
        report.add_error("epochs.channels", message)
    for message in channel_warnings:
        report.add_warning("epochs.channels", message)
    if not collection.artifacts:
        report.add_error("epochs.empty", "epoch collection contains no artifacts")
    duplicate_event_ids = _duplicate_values(artifact.event.event_id for artifact in collection.artifacts)
    if duplicate_event_ids:
        report.add_error("epochs.duplicate_event", "duplicate event IDs: " + ", ".join(duplicate_event_ids))
    collection_dir = target.parent
    for artifact in collection.artifacts:
        signal_path = collection_dir / artifact.signal_path
        if not signal_path.exists():
            report.add_error("epochs.missing_signal", "missing epoch signal artifact", location=artifact.event.event_id, path=str(signal_path))
            continue
        if artifact.event.recording_id and artifact.event.recording_id != collection.recording.recording_id:
            report.add_error("epochs.recording_mismatch", "artifact event recording ID does not match collection recording", location=artifact.event.event_id)
        errors, warnings = _validate_epoch_npz(signal_path, len(collection.channel_names), collection.sampling_rate_hz)
        for message in errors:
            report.add_error("epochs.signal_contract", message, location=artifact.event.event_id, path=str(signal_path))
        for message in warnings:
            report.add_warning("epochs.signal_contract", message, location=artifact.event.event_id, path=str(signal_path))
        for phase_name, shape in artifact.phase_shapes.items():
            if len(shape) != 2:
                report.add_error("epochs.phase_shape", f"phase {phase_name!r} metadata must be [channels, samples]", location=artifact.event.event_id)
    report.summary = summarize_collection(target)
    return report


def validate_preprocess_branch_artifact(path: str | Path) -> ValidationReport:
    target = Path(path).resolve()
    report = ValidationReport("preprocess_branch", str(target))
    branch = load_branch_result(target)
    for message in branch.config.validate():
        report.add_error("preprocess.invalid", message)
    if branch.config.rereference_mode == "channels" and not branch.config.reference_channels:
        report.add_error("preprocess.reference_missing", "channel-based rereference requires at least one reference channel")
    if _duplicate_values(branch.config.align_channels):
        report.add_error("preprocess.align_duplicate", "align_channels contains duplicates")
    if Path(branch.source_collection_path).resolve() == Path(branch.output_collection_path).resolve():
        report.add_error("preprocess.overwrite_raw", "output collection path must not overwrite the source collection")
    if branch.branch_name == "raw_preserving" and (
        branch.config.notch_hz
        or branch.config.resample_hz is not None
        or branch.config.align_channels
        or branch.config.rereference_mode != "none"
        or branch.config.scale_factor is not None
    ):
        report.add_warning("preprocess.raw_branch_modified", "raw_preserving branch carries transforms that may change the signal")
    if not Path(branch.source_collection_path).exists():
        report.add_error("preprocess.source_missing", "source collection path does not exist", path=branch.source_collection_path)
    if not Path(branch.output_collection_path).exists():
        report.add_error("preprocess.output_missing", "output collection path does not exist", path=branch.output_collection_path)
    else:
        output_report = validate_epoch_collection_artifact(branch.output_collection_path)
        report.errors.extend(output_report.errors)
        report.warnings.extend(output_report.warnings)
    report.summary = {
        "branch_name": branch.branch_name,
        "warning_count": len(branch.warnings),
        "transform_count": len(branch.transforms),
    }
    return report


def validate_candidate_pattern_artifact(path: str | Path) -> ValidationReport:
    target = Path(path).resolve()
    report = ValidationReport("candidate_pattern", str(target))
    candidate = _load_candidate(target)
    if not candidate.pattern_id:
        report.add_error("candidate.pattern_id", "pattern_id is required")
    if not candidate.target_label:
        report.add_error("candidate.target_label", "target_label is required")
    if candidate.cross_subject_agreement.n_subjects < 2:
        report.add_error("candidate.subject_support", "candidate has too few subjects for cross-subject analysis")
    if candidate.cross_subject_agreement.n_events < 2:
        report.add_warning("candidate.event_support", "candidate has very few events")
    candidate_root = Path(candidate.candidate_root or target.parent)
    required_artifacts = ["prototype_epoch", "subject_prototypes", "exemplar_epochs"]
    for artifact_key in required_artifacts:
        if artifact_key not in candidate.artifact_paths:
            report.add_error("candidate.artifact_missing", f"required artifact {artifact_key!r} is missing")
            continue
        artifact_path = candidate_root / candidate.artifact_paths[artifact_key]
        if not artifact_path.exists():
            report.add_error("candidate.artifact_missing", f"artifact {artifact_key!r} does not exist", path=str(artifact_path))
            continue
        errors, warnings = _validate_epoch_npz(artifact_path, len(candidate.channel_names), candidate.sampling_rate_hz)
        for message in errors:
            report.add_error("candidate.signal_contract", message, path=str(artifact_path))
        for message in warnings:
            report.add_warning("candidate.signal_contract", message, path=str(artifact_path))
    if "topography_summary" in candidate.artifact_paths:
        topography_path = candidate_root / candidate.artifact_paths["topography_summary"]
        if not topography_path.exists():
            report.add_warning("candidate.topography_missing", "topography derivative is referenced but missing", path=str(topography_path))
    if not candidate.run_id:
        report.add_warning("candidate.run_id_missing", "candidate does not record the discovery run ID")
    if candidate.reliability is None:
        report.add_warning("candidate.reliability_missing", "candidate does not include a reliability assessment")
    else:
        if candidate.reliability.confidence_tier in {"unstable", "insufficient"}:
            report.add_warning("candidate.low_confidence", f"candidate confidence tier is {candidate.reliability.confidence_tier}")
    for evidence in candidate.backbone_evidence:
        for relative_path in evidence.artifact_paths.values():
            evidence_path = candidate_root / relative_path
            if not evidence_path.exists():
                report.add_warning("candidate.backbone_artifact_missing", f"backbone artifact is referenced but missing for {evidence.model_id}", path=str(evidence_path))
    report.summary = summarize_candidate(target)
    return report


def validate_pattern_package_artifact(path: str | Path) -> ValidationReport:
    target = Path(path).resolve()
    report = ValidationReport("pattern_package", str(target))
    builder = PatternPackageBuilder(target.parent)
    errors = builder.validate_package(target)
    for message in errors:
        report.add_error("package.invalid", message)
    manifest = builder.load_manifest(target / "manifest.json") if (target / "manifest.json").exists() else None
    if manifest is not None:
        if not any(artifact.representation in {"raw_eeg", "processed_epoch"} for artifact in manifest.artifacts):
            report.add_error("package.signal_missing", "package contains no EEG-backed artifact")
        artifact_roles = {artifact.role for artifact in manifest.artifacts}
        if "run_manifest" not in artifact_roles:
            report.add_warning("package.run_manifest_missing", "package does not embed the originating run manifest")
        if "config_snapshot" not in artifact_roles:
            report.add_warning("package.config_snapshot_missing", "package does not embed the originating config snapshot")
    report.summary = summarize_package(target)
    return report


def validate_discovery_run_artifact(path: str | Path) -> ValidationReport:
    target = Path(path).resolve()
    report = ValidationReport("discovery_run", str(target))
    summary = _load_run_summary(target)
    required_paths = [
        summary.run_manifest_path,
        summary.config_snapshot_path,
        summary.environment_path,
        summary.artifact_lineage_path,
        summary.log_path,
    ]
    for required_path in [item for item in required_paths if item]:
        if not Path(required_path).exists():
            report.add_error("run.path_missing", "referenced run file does not exist", path=required_path)
    if summary.status != "success":
        report.add_warning("run.status", f"run completed with status {summary.status!r}")
    if summary.backbone_evaluation is not None:
        for required_path in (
            summary.backbone_evaluation.run_manifest_path,
            summary.backbone_evaluation.config_snapshot_path,
            summary.backbone_evaluation.environment_path,
            summary.backbone_evaluation.artifact_lineage_path,
            summary.backbone_evaluation.log_path,
        ):
            if required_path and not Path(required_path).exists():
                report.add_error("run.backbone_eval_path_missing", "referenced backbone evaluation file does not exist", path=required_path)
    for candidate in summary.candidates:
        candidate_path = Path(candidate.candidate_root) / candidate.artifact_paths.get("candidate_json", "candidate.json")
        if candidate_path.exists():
            candidate_report = validate_candidate_pattern_artifact(candidate_path)
            report.errors.extend(candidate_report.errors)
            report.warnings.extend(candidate_report.warnings)
    for failure in summary.failures:
        if failure.severity == "error":
            report.add_warning("run.failure_recorded", failure.message, location=failure.stage)
    report.summary = summarize_run(target)
    return report


def validate_study(collection_paths: Iterable[str | Path], *, min_subjects: int = 2, min_trials_per_label: int = 2) -> ValidationReport:
    resolved = [Path(path).resolve() for path in collection_paths]
    report = ValidationReport("epoch_collection", ",".join(str(path) for path in resolved))
    if not resolved:
        report.add_error("study.empty", "no collections were provided")
        return report
    collections: list[EpochCollection] = []
    for path in resolved:
        try:
            collections.append(load_epoch_collection(path))
        except Exception as exc:
            report.add_error("study.collection_unreadable", str(exc), path=str(path))
    collection_ids = _duplicate_values(collection.collection_id for collection in collections)
    if collection_ids:
        report.add_warning("study.duplicate_collection_id", "duplicate collection IDs detected: " + ", ".join(collection_ids))
    recording_map: dict[str, set[str]] = defaultdict(set)
    sample_rates: dict[tuple[str, str, str, str], set[float]] = defaultdict(set)
    label_subjects: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    label_trials: Counter[tuple[str, str, str, str]] = Counter()
    branch_channels: dict[str, list[set[str]]] = defaultdict(list)
    for collection in collections:
        branch_name = str(collection.metadata.get("branch_name", "source_epoch")) or "source_epoch"
        recording_map[collection.recording.recording_id].add(collection.recording.source_path)
        branch_channels[branch_name].append(set(collection.channel_names))
        for artifact in collection.artifacts:
            key = (
                artifact.event.label_namespace or collection.recording.label_namespace,
                artifact.event.event_family,
                artifact.event.target_label,
                branch_name,
            )
            sample_rates[key].add(float(collection.sampling_rate_hz))
            label_subjects[key].add(collection.recording.subject_id or collection.recording.recording_id)
            label_trials[key] += 1
    for recording_id, source_paths in recording_map.items():
        if len(source_paths) > 1:
            report.add_error("study.recording_collision", f"recording_id {recording_id!r} maps to multiple source files")
    for key, rates in sample_rates.items():
        if len(rates) > 1:
            report.add_error("study.sample_rate_mismatch", f"mixed sampling rates within discovery context {key!r}: {sorted(rates)}")
    for key, subjects in label_subjects.items():
        if len(subjects) < min_subjects:
            report.add_error("study.too_few_subjects", f"{key!r} has only {len(subjects)} subject(s)")
        if label_trials[key] < min_trials_per_label:
            report.add_warning("study.too_few_trials", f"{key!r} has only {label_trials[key]} trial(s)")
    branch_context_labels: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    branch_context_subjects: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for (label_namespace, event_family, target_label, branch_name), subjects in label_subjects.items():
        context_key = (label_namespace, event_family, branch_name)
        branch_context_labels[context_key].add(target_label)
        branch_context_subjects[context_key].update(subjects)
    for context_key, labels in branch_context_labels.items():
        if len(labels) < 2:
            report.add_warning("study.weak_controls", f"{context_key!r} has fewer than two labels, so negative controls are weak")
    for branch_name, channel_sets in branch_channels.items():
        if channel_sets:
            union = set().union(*channel_sets)
            intersection = set(channel_sets[0])
            for item in channel_sets[1:]:
                intersection &= item
            if union != intersection:
                report.add_warning("study.montage_inconsistent", f"branch {branch_name!r} has montage inconsistency across collections")
    label_counts = sorted(label_trials.values())
    if label_counts and label_counts[0] > 0 and label_counts[-1] / label_counts[0] >= 3.0:
        report.add_warning("study.class_imbalance", "major label imbalance detected across the provided collections")
    label_to_subjects: dict[str, set[str]] = defaultdict(set)
    for (_, _, target_label, _), subjects in label_subjects.items():
        label_to_subjects[target_label].update(subjects)
    labels = sorted(label_to_subjects)
    for left_index in range(len(labels)):
        for right_index in range(left_index + 1, len(labels)):
            if label_to_subjects[labels[left_index]] & label_to_subjects[labels[right_index]]:
                report.add_warning(
                    "study.subject_leakage_risk",
                    f"subjects appear in multiple labels ({labels[left_index]!r} and {labels[right_index]!r}); downstream recognition must split subject-wise",
                )
                break
    report.summary = {
        "collection_count": len(collections),
        "branch_count": len(branch_context_labels),
        "label_count": len({key[:3] for key in label_subjects}),
        "subject_count": len({collection.recording.subject_id or collection.recording.recording_id for collection in collections}),
    }
    return report


def summarize_collection(path: str | Path) -> dict[str, Any]:
    collection = load_epoch_collection(path)
    branch_name = str(collection.metadata.get("branch_name", "source_epoch")) or "source_epoch"
    label_counts = Counter(artifact.event.target_label for artifact in collection.artifacts)
    return {
        "collection_id": collection.collection_id,
        "recording_id": collection.recording.recording_id,
        "subject_id": collection.recording.subject_id,
        "branch_name": branch_name,
        "artifact_count": len(collection.artifacts),
        "channel_count": len(collection.channel_names),
        "sampling_rate_hz": collection.sampling_rate_hz,
        "labels": dict(sorted(label_counts.items())),
    }


def summarize_candidate(path: str | Path) -> dict[str, Any]:
    candidate = _load_candidate(path)
    return {
        "pattern_id": candidate.pattern_id,
        "target_label": candidate.target_label,
        "branch_name": candidate.branch_name,
        "subject_count": len(candidate.subject_ids),
        "event_count": len(candidate.event_ids),
        "backbone_ids": list(candidate.backbone_ids),
        "backbone_evidence_count": len(candidate.backbone_evidence),
        "backbone_consensus": candidate.backbone_consensus.overall_status if candidate.backbone_consensus is not None else "",
        "dominant_bands": list(candidate.dominant_bands),
        "dominant_channels": list(candidate.dominant_channels),
        "confidence_tier": candidate.reliability.confidence_tier if candidate.reliability else "unscored",
        "backbone_stability": candidate.reliability.backbone_stability if candidate.reliability else "",
        "backbone_support_score": candidate.reliability.backbone_support_score if candidate.reliability else None,
    }


def summarize_package(path: str | Path) -> dict[str, Any]:
    builder = PatternPackageBuilder(Path(path).resolve().parent)
    manifest = builder.load_manifest(Path(path).resolve() / "manifest.json")
    return {
        "pattern_id": manifest.pattern_id,
        "artifact_count": len(manifest.artifacts),
        "event_family": manifest.partition.event_family,
        "target_label": manifest.partition.target_label,
        "signal_artifact_count": sum(1 for artifact in manifest.artifacts if artifact.representation in {"raw_eeg", "processed_epoch"}),
        "embedded_run_bundle_count": sum(1 for artifact in manifest.artifacts if artifact.role in {"run_manifest", "config_snapshot", "environment", "artifact_lineage", "warnings", "run_log"}),
    }


def summarize_run(path: str | Path) -> dict[str, Any]:
    summary = _load_run_summary(path)
    return {
        "run_id": summary.run_id,
        "status": summary.status,
        "candidate_count": len(summary.candidates),
        "branch_count": len(summary.branch_names),
        "backbone_count": len(summary.backbone_ids),
        "backbone_evaluated": summary.backbone_evaluation is not None,
        "backbone_completed_count": 0 if summary.backbone_evaluation is None else len(summary.backbone_evaluation.completed_model_ids),
        "packaged_pattern_count": len(summary.packaged_pattern_paths),
        "failure_count": len(summary.failures),
        "rng_seed": summary.rng_seed,
    }



