from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

from .analysis_models import (
    EpochArtifactReference,
    EpochCollection,
    EpochPhaseRange,
    EpochWindowConfig,
    EventRecord,
    RecordingReference,
)
from .ingest import (
    LoadedRecording,
    load_event_table,
    load_normalized_event_table,
    load_recording,
    load_recording_reference,
)
from .models import slugify
from .run_tracking import RunArtifactRecord, StructuredRunLogger, make_run_id, write_run_bundle

PHASE_NAMES = ["pre_event", "onset", "sustained", "offset", "post_event"]


class EpochBuildError(RuntimeError):
    pass


def _json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_epoch_collection(collection: EpochCollection, path: str | Path) -> Path:
    output_path = Path(path)
    _json_write(output_path, collection.to_dict())
    return output_path


def load_epoch_collection(path: str | Path) -> EpochCollection:
    return EpochCollection.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _resolve_loaded_recording(recording: RecordingReference | LoadedRecording | str | Path) -> LoadedRecording:
    if isinstance(recording, LoadedRecording):
        return recording
    if isinstance(recording, RecordingReference):
        loaded = load_recording(
            recording.source_path,
            recording_id=recording.recording_id,
            subject_id=recording.subject_id,
            session_id=recording.session_id,
            label_namespace=recording.label_namespace,
        )
        merged_reference = RecordingReference(
            recording_id=loaded.reference.recording_id,
            subject_id=loaded.reference.subject_id,
            session_id=loaded.reference.session_id,
            label_namespace=loaded.reference.label_namespace,
            source_path=loaded.reference.source_path,
            source_format=loaded.reference.source_format,
            channel_names=list(loaded.reference.channel_names),
            sampling_rate_hz=loaded.reference.sampling_rate_hz,
            n_samples=loaded.reference.n_samples,
            duration_seconds=loaded.reference.duration_seconds,
            source_provenance={**loaded.reference.source_provenance, **recording.source_provenance},
            metadata={**loaded.reference.metadata, **recording.metadata},
        )
        return LoadedRecording(reference=merged_reference, data=loaded.data)
    path = Path(recording)
    if path.suffix.lower() == ".json":
        reference = load_recording_reference(path)
        return _resolve_loaded_recording(reference)
    return load_recording(path)


def _resolve_events(events: Iterable[EventRecord] | str | Path, recording: RecordingReference) -> tuple[list[EventRecord], str]:
    if isinstance(events, (str, Path)):
        event_path = Path(events)
        if event_path.suffix.lower() == ".json":
            try:
                loaded = load_normalized_event_table(event_path)
            except Exception:
                loaded = load_event_table(event_path, recording=recording, label_namespace=recording.label_namespace)
            return loaded, str(event_path.resolve())
        loaded = load_event_table(event_path, recording=recording, label_namespace=recording.label_namespace)
        return loaded, str(event_path.resolve())
    loaded_events = list(events)
    return loaded_events, ""


def _time_range(
    *,
    phase_name: str,
    requested_start: float,
    requested_end: float,
    sampling_rate_hz: float,
    total_samples: int,
) -> EpochPhaseRange:
    clipped = False
    warning = ""
    if requested_end < requested_start:
        requested_end = requested_start
        clipped = True
        warning = f"{phase_name} requested end preceded start; collapsed to zero-length"
    start_seconds = max(0.0, requested_start)
    end_seconds = max(start_seconds, requested_end)
    if start_seconds != requested_start or end_seconds != requested_end:
        clipped = True
        warning = warning or f"{phase_name} window was clipped to the recording bounds"

    start_sample = max(0, min(total_samples, int(math.floor(start_seconds * sampling_rate_hz))))
    end_sample = max(start_sample, min(total_samples, int(math.ceil(end_seconds * sampling_rate_hz))))
    actual_start_seconds = start_sample / sampling_rate_hz
    actual_end_seconds = end_sample / sampling_rate_hz
    if end_sample == total_samples and requested_end > total_samples / sampling_rate_hz:
        clipped = True
        warning = warning or f"{phase_name} window exceeded the recording duration and was clipped"
    return EpochPhaseRange(
        phase_name=phase_name,
        requested_start_seconds=requested_start,
        requested_end_seconds=requested_end,
        actual_start_seconds=actual_start_seconds,
        actual_end_seconds=actual_end_seconds,
        start_sample=start_sample,
        end_sample=end_sample,
        derived_samples=end_sample - start_sample,
        clipped=clipped,
        warning=warning,
    )


def _event_phase_requests(event: EventRecord, window_config: EpochWindowConfig) -> dict[str, tuple[float, float]]:
    event_start = event.onset_seconds
    event_end = event.end_seconds
    onset_end = event_start + window_config.onset_seconds
    if event.duration_seconds > 0:
        onset_end = min(event_end, onset_end)
    offset_start = event_end - window_config.offset_seconds if window_config.offset_seconds else event_end
    if event.duration_seconds > 0:
        offset_start = max(event_start, offset_start)

    sustained_start = min(onset_end, event_end)
    sustained_end = max(sustained_start, offset_start)
    if window_config.sustained_seconds is not None:
        sustained_end = min(sustained_end, sustained_start + window_config.sustained_seconds)

    return {
        "pre_event": (event_start - window_config.pre_event_seconds, event_start),
        "onset": (event_start, onset_end),
        "sustained": (sustained_start, sustained_end),
        "offset": (offset_start, event_end),
        "post_event": (event_end, event_end + window_config.post_event_seconds),
    }


def _baseline_request(event: EventRecord, window_config: EpochWindowConfig) -> tuple[float, float] | None:
    if window_config.baseline_window is None:
        return None
    return (
        event.onset_seconds + window_config.baseline_window.start_offset_seconds,
        event.onset_seconds + window_config.baseline_window.end_offset_seconds,
    )


def _slice_data(data: np.ndarray, phase_range: EpochPhaseRange) -> np.ndarray:
    return np.asarray(data[:, phase_range.start_sample : phase_range.end_sample], dtype=np.float32)


def build_event_epochs(
    recording: RecordingReference | LoadedRecording | str | Path,
    events: Iterable[EventRecord] | str | Path,
    *,
    output_root: str | Path,
    window_config: EpochWindowConfig,
    collection_id: str = "",
    run_id: str = "",
    command: str = "",
    overwrite: bool = False,
) -> tuple[Path, EpochCollection]:
    loaded_recording = _resolve_loaded_recording(recording)
    resolved_events, source_event_table_path = _resolve_events(events, loaded_recording.reference)
    if not resolved_events:
        raise EpochBuildError("No events were available for epoch extraction")

    errors = window_config.validate()
    if errors:
        raise EpochBuildError("invalid epoch window config:\n- " + "\n- ".join(errors))

    collection_name = collection_id or f"{loaded_recording.reference.recording_id}__{window_config.slug()}"
    collection_slug = slugify(collection_name, "epoch_collection")
    collection_dir = Path(output_root) / "epochs" / slugify(loaded_recording.reference.recording_id) / collection_slug
    if collection_dir.exists() and not overwrite:
        raise FileExistsError(f"epoch collection already exists: {collection_dir}")
    if collection_dir.exists() and overwrite:
        shutil.rmtree(collection_dir)
    collection_dir.mkdir(parents=True, exist_ok=True)
    events_dir = collection_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    operation_run_id = make_run_id("epoch", run_id or collection_name)
    logger = StructuredRunLogger(collection_dir / "run.log.jsonl")
    logger.log(level="info", stage="epoch", message="starting epoch extraction", collection_id=collection_name)

    artifacts: list[EpochArtifactReference] = []
    total_samples = loaded_recording.reference.n_samples
    sampling_rate_hz = loaded_recording.reference.sampling_rate_hz
    warnings: list[str] = []

    for event in resolved_events:
        if event.recording_id and event.recording_id != loaded_recording.reference.recording_id:
            raise EpochBuildError(
                f"event {event.event_id!r} belongs to recording {event.recording_id!r}, expected {loaded_recording.reference.recording_id!r}"
            )
        phase_ranges: list[EpochPhaseRange] = []
        phase_shapes: dict[str, list[int]] = {}
        artifact_warnings: list[str] = []
        phase_arrays: dict[str, np.ndarray] = {}
        for phase_name, (start_seconds, end_seconds) in _event_phase_requests(event, window_config).items():
            phase_range = _time_range(
                phase_name=phase_name,
                requested_start=start_seconds,
                requested_end=end_seconds,
                sampling_rate_hz=sampling_rate_hz,
                total_samples=total_samples,
            )
            phase_ranges.append(phase_range)
            if phase_range.warning:
                artifact_warnings.append(phase_range.warning)
            phase_data = _slice_data(loaded_recording.data, phase_range)
            phase_arrays[phase_name] = phase_data
            phase_shapes[phase_name] = [int(size) for size in phase_data.shape]

        baseline_range = None
        baseline_request = _baseline_request(event, window_config)
        if baseline_request is not None:
            baseline_range = _time_range(
                phase_name="baseline",
                requested_start=baseline_request[0],
                requested_end=baseline_request[1],
                sampling_rate_hz=sampling_rate_hz,
                total_samples=total_samples,
            )
            if baseline_range.warning:
                artifact_warnings.append(baseline_range.warning)
            baseline_data = _slice_data(loaded_recording.data, baseline_range)
            phase_arrays["baseline"] = baseline_data
            phase_shapes["baseline"] = [int(size) for size in baseline_data.shape]

        event_file_name = f"{slugify(event.event_id, 'event')}.npz"
        event_path = events_dir / event_file_name
        np.savez_compressed(
            event_path,
            channel_names=np.asarray(loaded_recording.reference.channel_names),
            sampling_rate_hz=np.asarray([sampling_rate_hz], dtype=np.float32),
            event_onset_seconds=np.asarray([event.onset_seconds], dtype=np.float32),
            event_duration_seconds=np.asarray([event.duration_seconds], dtype=np.float32),
            **phase_arrays,
        )
        artifacts.append(
            EpochArtifactReference(
                event=event,
                signal_path=str(event_path.relative_to(collection_dir)),
                format="npz",
                phase_ranges=phase_ranges,
                baseline_range=baseline_range,
                phase_shapes=phase_shapes,
                warnings=artifact_warnings,
                metadata={"run_id": operation_run_id},
            )
        )
        warnings.extend(artifact_warnings)

    collection = EpochCollection(
        collection_id=collection_name,
        recording=loaded_recording.reference,
        window_config=window_config,
        channel_names=list(loaded_recording.reference.channel_names),
        sampling_rate_hz=sampling_rate_hz,
        artifacts=artifacts,
        source_event_table_path=source_event_table_path,
        metadata={"run_id": operation_run_id},
    )
    collection_errors = collection.validate()
    if collection_errors:
        raise EpochBuildError("invalid epoch collection:\n- " + "\n- ".join(collection_errors))
    collection_path = save_epoch_collection(collection, collection_dir / "collection.json")

    run_paths = write_run_bundle(
        run_root=collection_dir,
        run_id=operation_run_id,
        operation="epoch",
        command=command,
        config_snapshot={
            "collection_id": collection_name,
            "recording_id": loaded_recording.reference.recording_id,
            "window_config": window_config.to_dict(),
            "source_event_table_path": source_event_table_path,
            "event_count": len(resolved_events),
        },
        source_artifacts=[
            RunArtifactRecord(
                artifact_type="recording_reference",
                path=loaded_recording.reference.source_path,
                role="source_recording",
                format=loaded_recording.reference.source_format,
            ),
            *(
                [
                    RunArtifactRecord(
                        artifact_type="event_table",
                        path=source_event_table_path,
                        role="source_events",
                        format=Path(source_event_table_path).suffix.lstrip("."),
                    )
                ]
                if source_event_table_path
                else []
            ),
        ],
        generated_artifacts=[
            RunArtifactRecord(
                artifact_type="epoch_collection",
                path=str(collection_path),
                role="collection_index",
                format="json",
                parent_paths=[loaded_recording.reference.source_path],
            ),
            *[
                RunArtifactRecord(
                    artifact_type="epoch_artifact",
                    path=str(collection_dir / artifact.signal_path),
                    role=artifact.event.event_id,
                    format=artifact.format,
                    parent_paths=[loaded_recording.reference.source_path],
                )
                for artifact in artifacts
            ],
        ],
        code_root=Path(__file__).resolve().parents[2],
        warnings=warnings,
        optional_dependencies=["mne", "numpy"],
    )
    collection.metadata = {**collection.metadata, **run_paths}
    collection_path = save_epoch_collection(collection, collection_path)
    logger.log(level="info", stage="epoch", message="epoch extraction completed", artifact_count=len(artifacts))
    return collection_path, collection
