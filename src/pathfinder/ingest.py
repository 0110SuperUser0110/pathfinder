from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .analysis_models import EventRecord, RecordingReference
from .models import slugify
from .run_tracking import RunArtifactRecord, StructuredRunLogger, make_run_id, write_run_bundle

SUPPORTED_SOURCE_SUFFIXES = {".npz", ".edf", ".bdf", ".fif"}
_MNE_SUFFIXES = {".edf", ".bdf", ".fif"}


@dataclass(slots=True)
class LoadedRecording:
    reference: RecordingReference
    data: np.ndarray


class RecordingLoadError(RuntimeError):
    pass


class EventTableError(RuntimeError):
    pass


KNOWN_EVENT_FIELDS = {
    "event_id",
    "recording_id",
    "onset_seconds",
    "onset",
    "start_seconds",
    "duration_seconds",
    "duration",
    "length_seconds",
    "event_family",
    "family",
    "target_label",
    "label",
    "event_subtype",
    "subtype",
    "label_namespace",
}


def _utc_iso_from_stat(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()


def _source_provenance(path: Path, loader_id: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "loader_id": loader_id,
        "size_bytes": stat.st_size,
        "modified_time_utc": _utc_iso_from_stat(path),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _load_mne_module():
    try:
        import mne  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RecordingLoadError(
            "Loading FIF/EDF/BDF recordings requires the optional 'mne' package. "
            "Install it in the Pathfinder runtime before ingesting those formats."
        ) from exc
    return mne


def _normalize_npz_data(data: np.ndarray, channel_names: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    if data.ndim != 2:
        raise RecordingLoadError("NPZ recordings must store a 2D 'data' array shaped [channels, samples]")
    if data.shape[0] == len(channel_names):
        return np.asarray(data, dtype=np.float32), {"transposed_on_load": False}
    if data.shape[1] == len(channel_names):
        return np.asarray(data.T, dtype=np.float32), {"transposed_on_load": True}
    raise RecordingLoadError(
        "Channel count does not match NPZ data shape. Expected one axis to match channel_names length."
    )


def _recording_defaults(
    source_path: Path,
    *,
    recording_id: str,
    subject_id: str,
    session_id: str,
    label_namespace: str,
) -> dict[str, str]:
    stem = slugify(source_path.stem, "recording")
    return {
        "recording_id": recording_id or stem,
        "subject_id": subject_id,
        "session_id": session_id,
        "label_namespace": label_namespace,
    }


def inspect_recording(
    source_path: str | Path,
    *,
    recording_id: str = "",
    subject_id: str = "",
    session_id: str = "",
    label_namespace: str = "",
) -> RecordingReference:
    path = Path(source_path).resolve()
    suffix = path.suffix.lower()
    defaults = _recording_defaults(
        path,
        recording_id=recording_id,
        subject_id=subject_id,
        session_id=session_id,
        label_namespace=label_namespace,
    )
    if suffix not in SUPPORTED_SOURCE_SUFFIXES:
        raise RecordingLoadError(
            f"Unsupported recording format {suffix!r}. Supported suffixes: {', '.join(sorted(SUPPORTED_SOURCE_SUFFIXES))}"
        )
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "data" not in payload or "channel_names" not in payload or "sampling_rate_hz" not in payload:
                raise RecordingLoadError(
                    "NPZ recordings must include 'data', 'channel_names', and 'sampling_rate_hz' arrays"
                )
            channel_names = [str(item) for item in payload["channel_names"].tolist()]
            data, provenance = _normalize_npz_data(np.asarray(payload["data"]), channel_names)
            sampling_rate_hz = float(np.asarray(payload["sampling_rate_hz"]).reshape(-1)[0])
            metadata = {}
            for key in ("subject_id", "session_id", "recording_id", "label_namespace"):
                if key in payload:
                    metadata[key] = str(np.asarray(payload[key]).reshape(-1)[0])
            return RecordingReference(
                recording_id=defaults["recording_id"] or metadata.get("recording_id", defaults["recording_id"]),
                subject_id=defaults["subject_id"] or metadata.get("subject_id", ""),
                session_id=defaults["session_id"] or metadata.get("session_id", ""),
                label_namespace=defaults["label_namespace"] or metadata.get("label_namespace", ""),
                source_path=str(path),
                source_format="npz",
                channel_names=channel_names,
                sampling_rate_hz=sampling_rate_hz,
                n_samples=int(data.shape[1]),
                duration_seconds=float(data.shape[1] / sampling_rate_hz),
                source_provenance={**_source_provenance(path, "npz"), **provenance},
            )
    mne = _load_mne_module()
    if suffix == ".fif":
        raw = mne.io.read_raw_fif(str(path), preload=False, verbose="ERROR")
    else:
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    try:
        channel_names = [str(name) for name in raw.ch_names]
        sampling_rate_hz = float(raw.info["sfreq"])
        n_samples = int(raw.n_times)
        return RecordingReference(
            recording_id=defaults["recording_id"],
            subject_id=defaults["subject_id"],
            session_id=defaults["session_id"],
            label_namespace=defaults["label_namespace"],
            source_path=str(path),
            source_format=suffix.lstrip("."),
            channel_names=channel_names,
            sampling_rate_hz=sampling_rate_hz,
            n_samples=n_samples,
            duration_seconds=float(n_samples / sampling_rate_hz),
            source_provenance=_source_provenance(path, "mne"),
            metadata={"measurement_date": str(raw.info.get("meas_date") or "")},
        )
    finally:
        raw.close()


def load_recording(
    source_path: str | Path,
    *,
    recording_id: str = "",
    subject_id: str = "",
    session_id: str = "",
    label_namespace: str = "",
) -> LoadedRecording:
    path = Path(source_path).resolve()
    reference = inspect_recording(
        path,
        recording_id=recording_id,
        subject_id=subject_id,
        session_id=session_id,
        label_namespace=label_namespace,
    )
    if reference.source_format == "npz":
        with np.load(path, allow_pickle=False) as payload:
            data, _ = _normalize_npz_data(np.asarray(payload["data"]), reference.channel_names)
        return LoadedRecording(reference=reference, data=data)

    mne = _load_mne_module()
    if reference.source_format == "fif":
        raw = mne.io.read_raw_fif(str(path), preload=True, verbose="ERROR")
    else:
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    try:
        data = np.asarray(raw.get_data(), dtype=np.float32)
        return LoadedRecording(reference=reference, data=data)
    finally:
        raw.close()


def save_recording_reference(recording: RecordingReference, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(recording.to_dict(), indent=2), encoding="utf-8")
    return output_path


def load_recording_reference(path: str | Path) -> RecordingReference:
    return RecordingReference.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _normalize_event_row(
    row: dict[str, Any],
    *,
    index: int,
    recording: RecordingReference | None,
    label_namespace: str,
) -> EventRecord:
    recording_id = _safe_text(row.get("recording_id"), recording.recording_id if recording else "")
    onset_seconds = _safe_float(row.get("onset_seconds", row.get("onset", row.get("start_seconds"))), 0.0)
    duration_seconds = _safe_float(
        row.get("duration_seconds", row.get("duration", row.get("length_seconds"))),
        0.0,
    )
    event_family = _safe_text(row.get("event_family", row.get("family")))
    target_label = _safe_text(row.get("target_label", row.get("label")))
    event_subtype = _safe_text(row.get("event_subtype", row.get("subtype")))
    namespace = _safe_text(row.get("label_namespace"), label_namespace or (recording.label_namespace if recording else ""))
    event_id = _safe_text(row.get("event_id"), f"{recording_id or 'recording'}_event_{index:04d}")
    metadata = {key: value for key, value in row.items() if key not in KNOWN_EVENT_FIELDS}
    event = EventRecord(
        event_id=event_id,
        recording_id=recording_id,
        onset_seconds=onset_seconds,
        duration_seconds=duration_seconds,
        event_family=event_family,
        target_label=target_label,
        event_subtype=event_subtype,
        label_namespace=namespace,
        metadata=metadata,
    )
    errors = event.validate()
    if errors:
        raise EventTableError("invalid event row:\n- " + "\n- ".join(errors))
    return event


def load_event_table(
    event_table_path: str | Path,
    *,
    recording: RecordingReference | None = None,
    label_namespace: str = "",
) -> list[EventRecord]:
    path = Path(event_table_path).resolve()
    suffix = path.suffix.lower()
    rows: list[dict[str, Any]] = []
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "events" in payload:
            rows = [dict(item) for item in payload["events"]]
        elif isinstance(payload, list):
            rows = [dict(item) for item in payload]
        else:
            raise EventTableError("JSON event tables must be a list of events or a dict with an 'events' key")
    elif suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            rows = [dict(row) for row in reader]
    else:
        raise EventTableError("Unsupported event table format. Use JSON, CSV, or TSV")

    events = [
        _normalize_event_row(row, index=index, recording=recording, label_namespace=label_namespace)
        for index, row in enumerate(rows, start=1)
    ]
    return sorted(events, key=lambda item: (item.onset_seconds, item.event_id))


def save_event_table(events: list[EventRecord], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"events": [event.to_dict() for event in events]}, indent=2),
        encoding="utf-8",
    )
    return output_path


def load_normalized_event_table(path: str | Path) -> list[EventRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    events = payload["events"] if isinstance(payload, dict) else payload
    return [EventRecord.from_dict(item) for item in events]


def ingest_recording(
    source_path: str | Path,
    *,
    output_root: str | Path,
    recording_id: str = "",
    subject_id: str = "",
    session_id: str = "",
    label_namespace: str = "",
    event_table_path: str | Path | None = None,
    run_id: str = "",
    command: str = "",
    overwrite: bool = False,
) -> tuple[Path, RecordingReference, list[EventRecord]]:
    recording = inspect_recording(
        source_path,
        recording_id=recording_id,
        subject_id=subject_id,
        session_id=session_id,
        label_namespace=label_namespace,
    )
    operation_run_id = make_run_id("ingest", run_id or recording.recording_id)
    recording_dir = Path(output_root) / "recordings" / slugify(recording.recording_id, "recording")
    if recording_dir.exists() and not overwrite:
        raise FileExistsError(f"recording ingest output already exists: {recording_dir}")
    if recording_dir.exists() and overwrite:
        shutil.rmtree(recording_dir)
    recording_dir.mkdir(parents=True, exist_ok=True)

    logger = StructuredRunLogger(recording_dir / "run.log.jsonl")
    logger.log(level="info", stage="ingest", message="starting ingest", recording_id=recording.recording_id)

    recording.metadata = {
        **recording.metadata,
        "run_id": operation_run_id,
    }
    recording_path = save_recording_reference(recording, recording_dir / "recording.json")
    logger.log(level="info", stage="ingest", message="recording index written", path=str(recording_path))

    events: list[EventRecord] = []
    warnings: list[str] = []
    event_output_path = recording_dir / "events.json"
    if event_table_path is not None:
        events = load_event_table(event_table_path, recording=recording, label_namespace=label_namespace)
        save_event_table(events, event_output_path)
        logger.log(level="info", stage="ingest", message="event table normalized", event_count=len(events), path=str(event_output_path))
    else:
        warnings.append("no event table was provided during ingest")
        logger.log(level="warning", stage="ingest", message="no event table was provided")

    run_paths = write_run_bundle(
        run_root=recording_dir,
        run_id=operation_run_id,
        operation="ingest",
        command=command,
        config_snapshot={
            "source_path": str(Path(source_path).resolve()),
            "recording_id": recording.recording_id,
            "subject_id": recording.subject_id,
            "session_id": recording.session_id,
            "label_namespace": recording.label_namespace,
            "event_table_path": str(Path(event_table_path).resolve()) if event_table_path is not None else "",
        },
        source_artifacts=[
            RunArtifactRecord(
                artifact_type="recording_source",
                path=str(Path(source_path).resolve()),
                role="raw_recording",
                format=recording.source_format,
            ),
            *(
                [
                    RunArtifactRecord(
                        artifact_type="event_table_source",
                        path=str(Path(event_table_path).resolve()),
                        role="event_table",
                        format=Path(event_table_path).suffix.lstrip("."),
                    )
                ]
                if event_table_path is not None
                else []
            ),
        ],
        generated_artifacts=[
            RunArtifactRecord(
                artifact_type="recording_reference",
                path=str(recording_path),
                role="recording_index",
                format="json",
                parent_paths=[str(Path(source_path).resolve())],
            ),
            *(
                [
                    RunArtifactRecord(
                        artifact_type="event_table",
                        path=str(event_output_path),
                        role="normalized_event_table",
                        format="json",
                        parent_paths=[str(Path(event_table_path).resolve())],
                    )
                ]
                if event_table_path is not None
                else []
            ),
        ],
        code_root=Path(__file__).resolve().parents[2],
        warnings=warnings,
        optional_dependencies=["mne", "numpy"],
    )
    recording.metadata = {
        **recording.metadata,
        **run_paths,
    }
    save_recording_reference(recording, recording_path)
    logger.log(level="info", stage="ingest", message="ingest completed", run_id=operation_run_id)
    return recording_path, recording, events
