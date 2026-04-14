from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .analysis_models import (
    ChannelRemapReport,
    EpochArtifactReference,
    EpochCollection,
    PreprocessBranchConfig,
    PreprocessBranchResult,
    TransformRecord,
)
from .epochs import load_epoch_collection, save_epoch_collection
from .models import slugify
from .run_tracking import RunArtifactRecord, StructuredRunLogger, make_run_id, write_run_bundle


class PreprocessError(RuntimeError):
    pass


PHASE_KEYS = {"pre_event", "onset", "sustained", "offset", "post_event", "baseline"}


def save_branch_result(result: PreprocessBranchResult, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return output_path


def load_branch_result(path: str | Path) -> PreprocessBranchResult:
    return PreprocessBranchResult.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _load_epoch_arrays(collection_dir: Path, artifact: EpochArtifactReference) -> dict[str, np.ndarray]:
    with np.load(collection_dir / artifact.signal_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key], dtype=np.float32) for key in payload.files if key in PHASE_KEYS}
    return arrays


def _save_epoch_arrays(path: Path, *, arrays: dict[str, np.ndarray], channel_names: list[str], sampling_rate_hz: float) -> None:
    np.savez_compressed(
        path,
        channel_names=np.asarray(channel_names),
        sampling_rate_hz=np.asarray([sampling_rate_hz], dtype=np.float32),
        **arrays,
    )


def _apply_notch(data: np.ndarray, sampling_rate_hz: float, freq_hz: float, bandwidth_hz: float) -> np.ndarray:
    if data.shape[1] == 0:
        return data
    fft = np.fft.rfft(data, axis=1)
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / sampling_rate_hz)
    mask = (freqs >= max(0.0, freq_hz - bandwidth_hz / 2.0)) & (freqs <= freq_hz + bandwidth_hz / 2.0)
    fft[:, mask] = 0
    filtered = np.fft.irfft(fft, n=data.shape[1], axis=1)
    return np.asarray(filtered, dtype=np.float32)


def _apply_resample(data: np.ndarray, old_rate_hz: float, new_rate_hz: float) -> np.ndarray:
    if data.shape[1] == 0 or old_rate_hz == new_rate_hz:
        return data
    duration_seconds = data.shape[1] / old_rate_hz
    new_samples = max(1, int(round(duration_seconds * new_rate_hz)))
    old_index = np.linspace(0.0, duration_seconds, num=data.shape[1], endpoint=False)
    new_index = np.linspace(0.0, duration_seconds, num=new_samples, endpoint=False)
    resampled = np.empty((data.shape[0], new_samples), dtype=np.float32)
    for channel_index in range(data.shape[0]):
        resampled[channel_index] = np.interp(new_index, old_index, data[channel_index]).astype(np.float32)
    return resampled


def _align_arrays(
    arrays: dict[str, np.ndarray],
    current_channels: list[str],
    target_channels: list[str],
) -> tuple[dict[str, np.ndarray], ChannelRemapReport]:
    channel_index = {name: index for index, name in enumerate(current_channels)}
    missing_channels = [name for name in target_channels if name not in channel_index]
    dropped_channels = [name for name in current_channels if name not in set(target_channels)]
    reused_channels = [name for name in target_channels if target_channels.count(name) > 1]
    aligned: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        new_value = np.full((len(target_channels), value.shape[1]), np.nan, dtype=np.float32)
        for target_index, channel_name in enumerate(target_channels):
            if channel_name in channel_index:
                new_value[target_index] = value[channel_index[channel_name]]
        aligned[key] = new_value
    report = ChannelRemapReport(
        original_channels=list(current_channels),
        output_channels=list(target_channels),
        missing_channels=missing_channels,
        dropped_channels=dropped_channels,
        reused_channels=sorted(set(reused_channels)),
        fill_value="nan",
    )
    return aligned, report


def _rereference_arrays(
    arrays: dict[str, np.ndarray],
    channel_names: list[str],
    *,
    mode: str,
    reference_channels: list[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    warnings: list[str] = []
    if mode == "none":
        return arrays, warnings
    if mode == "average":
        reference_indices = list(range(len(channel_names)))
    else:
        lookup = {name: index for index, name in enumerate(channel_names)}
        reference_indices = [lookup[name] for name in reference_channels if name in lookup]
        missing = [name for name in reference_channels if name not in lookup]
        if missing:
            warnings.append("missing reference channels: " + ", ".join(missing))
    if not reference_indices:
        warnings.append("no valid reference channels were available; rereference was skipped")
        return arrays, warnings

    rereferenced: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if value.shape[1] == 0:
            rereferenced[key] = value
            continue
        reference = np.nanmean(value[reference_indices], axis=0, keepdims=True)
        rereferenced[key] = np.asarray(value - reference, dtype=np.float32)
    return rereferenced, warnings


def _baseline_subtract(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[str]]:
    warnings: list[str] = []
    baseline = arrays.get("baseline")
    if baseline is None or baseline.shape[1] == 0:
        warnings.append("baseline subtraction was requested but no baseline samples were available")
        return arrays, warnings
    valid_counts = np.sum(~np.isnan(baseline), axis=1, keepdims=True)
    baseline_sum = np.nansum(baseline, axis=1, keepdims=True)
    baseline_mean = np.zeros((baseline.shape[0], 1), dtype=np.float32)
    np.divide(baseline_sum, valid_counts, out=baseline_mean, where=valid_counts > 0)
    if np.any(valid_counts == 0):
        warnings.append("baseline subtraction encountered channels with no valid baseline samples")
    adjusted = {key: np.asarray(value - baseline_mean, dtype=np.float32) for key, value in arrays.items()}
    return adjusted, warnings


def _update_artifact_reference(
    artifact: EpochArtifactReference,
    *,
    signal_path: str,
    arrays: dict[str, np.ndarray],
    warnings: list[str],
    run_id: str,
) -> EpochArtifactReference:
    new_artifact = copy.deepcopy(artifact)
    new_artifact.signal_path = signal_path
    new_artifact.warnings = sorted(set(list(new_artifact.warnings) + list(warnings)))
    new_artifact.phase_shapes = {key: [int(size) for size in value.shape] for key, value in arrays.items()}
    phase_lookup = {phase.phase_name: phase for phase in new_artifact.phase_ranges}
    for phase_name, phase in phase_lookup.items():
        if phase_name in arrays:
            phase.derived_samples = int(arrays[phase_name].shape[1])
    if new_artifact.baseline_range is not None and "baseline" in arrays:
        new_artifact.baseline_range.derived_samples = int(arrays["baseline"].shape[1])
    new_artifact.metadata = {**new_artifact.metadata, "run_id": run_id}
    return new_artifact


def preprocess_epoch_collection(
    collection_path: str | Path,
    *,
    output_root: str | Path,
    config: PreprocessBranchConfig,
    run_id: str = "",
    command: str = "",
    overwrite: bool = False,
) -> tuple[Path, PreprocessBranchResult]:
    config_errors = config.validate()
    if config_errors:
        raise PreprocessError("invalid preprocess config:\n- " + "\n- ".join(config_errors))

    source_collection_path = Path(collection_path).resolve()
    collection = load_epoch_collection(source_collection_path)
    collection_dir = source_collection_path.parent
    branch_dir = (
        Path(output_root)
        / "preprocess"
        / slugify(collection.recording.recording_id)
        / slugify(collection.collection_id)
        / slugify(config.branch_name)
    )
    if branch_dir.exists() and not overwrite:
        raise FileExistsError(f"preprocess branch already exists: {branch_dir}")
    if branch_dir.exists() and overwrite:
        shutil.rmtree(branch_dir)
    branch_dir.mkdir(parents=True, exist_ok=True)
    events_dir = branch_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    operation_run_id = make_run_id("preprocess", run_id or f"{collection.collection_id}_{config.branch_name}")
    logger = StructuredRunLogger(branch_dir / "run.log.jsonl")
    logger.log(level="info", stage="preprocess", message="starting preprocessing branch", branch_name=config.branch_name)

    current_channels = list(collection.channel_names)
    output_channels = list(current_channels)
    channel_report = None
    branch_warnings: list[str] = []
    transforms: list[TransformRecord] = [
        TransformRecord(
            name="branch_created",
            parameters={
                "branch_name": config.branch_name,
                "source_collection_path": str(source_collection_path),
                "run_id": operation_run_id,
            },
        )
    ]

    if config.branch_name == "raw_preserving":
        if config.notch_hz or config.resample_hz or config.align_channels or config.rereference_mode != "none":
            branch_warnings.append("raw_preserving branch received transform options; raw-preserving intent should be reviewed")

    if config.align_channels:
        output_channels = list(config.align_channels)
        _, channel_report = _align_arrays({"baseline": np.zeros((len(current_channels), 0), dtype=np.float32)}, current_channels, output_channels)
        transforms.append(
            TransformRecord(
                name="align_channels",
                parameters={"target_channels": list(output_channels)},
                warnings=list(channel_report.missing_channels) + list(channel_report.dropped_channels),
            )
        )
        if channel_report.missing_channels:
            branch_warnings.append("missing aligned channels were filled with NaN values")

    if config.rereference_mode != "none":
        transforms.append(
            TransformRecord(
                name="rereference",
                parameters={
                    "mode": config.rereference_mode,
                    "reference_channels": list(config.reference_channels),
                },
            )
        )
    if config.scale_factor is not None:
        transforms.append(TransformRecord(name="scale", parameters={"scale_factor": config.scale_factor}))
    for freq in config.notch_hz:
        transforms.append(
            TransformRecord(
                name="notch_filter",
                parameters={"freq_hz": freq, "bandwidth_hz": config.notch_bandwidth_hz},
            )
        )
    if config.resample_hz is not None:
        transforms.append(TransformRecord(name="resample", parameters={"target_rate_hz": config.resample_hz}))
    if config.baseline_mode != "none":
        transforms.append(TransformRecord(name="baseline", parameters={"mode": config.baseline_mode}))

    output_sampling_rate_hz = config.resample_hz or collection.sampling_rate_hz
    output_artifacts: list[EpochArtifactReference] = []

    for artifact in collection.artifacts:
        arrays = _load_epoch_arrays(collection_dir, artifact)
        artifact_warnings: list[str] = []

        if config.align_channels:
            arrays, _ = _align_arrays(arrays, current_channels, output_channels)
        if config.rereference_mode != "none":
            arrays, reref_warnings = _rereference_arrays(
                arrays,
                output_channels,
                mode=config.rereference_mode,
                reference_channels=config.reference_channels,
            )
            artifact_warnings.extend(reref_warnings)
        if config.scale_factor is not None:
            arrays = {key: np.asarray(value * config.scale_factor, dtype=np.float32) for key, value in arrays.items()}
        for freq in config.notch_hz:
            arrays = {
                key: _apply_notch(value, collection.sampling_rate_hz, freq, config.notch_bandwidth_hz)
                for key, value in arrays.items()
            }
        if config.resample_hz is not None:
            arrays = {
                key: _apply_resample(value, collection.sampling_rate_hz, config.resample_hz)
                for key, value in arrays.items()
            }
        if config.baseline_mode == "subtract_mean":
            arrays, baseline_warnings = _baseline_subtract(arrays)
            artifact_warnings.extend(baseline_warnings)
        elif config.baseline_mode == "metadata_only" and artifact.baseline_range is None:
            artifact_warnings.append("baseline metadata was requested but no baseline window exists for this event")

        output_path = events_dir / Path(artifact.signal_path).name
        _save_epoch_arrays(output_path, arrays=arrays, channel_names=output_channels, sampling_rate_hz=output_sampling_rate_hz)
        output_artifacts.append(
            _update_artifact_reference(
                artifact,
                signal_path=str(output_path.relative_to(branch_dir)),
                arrays=arrays,
                warnings=artifact_warnings,
                run_id=operation_run_id,
            )
        )
        branch_warnings.extend(artifact_warnings)

    output_collection = EpochCollection(
        collection_id=f"{collection.collection_id}__{config.branch_name}",
        recording=collection.recording,
        window_config=collection.window_config,
        channel_names=output_channels,
        sampling_rate_hz=output_sampling_rate_hz,
        artifacts=output_artifacts,
        source_event_table_path=collection.source_event_table_path,
        metadata={
            **collection.metadata,
            "branch_name": config.branch_name,
            "source_collection_path": str(source_collection_path),
            "run_id": operation_run_id,
        },
    )
    output_collection_path = save_epoch_collection(output_collection, branch_dir / "collection.json")
    result = PreprocessBranchResult(
        branch_name=config.branch_name,
        source_collection_path=str(source_collection_path),
        output_collection_path=str(output_collection_path),
        config=config,
        transforms=transforms,
        warnings=sorted(set(branch_warnings)),
        channel_report=channel_report,
    )
    branch_result_path = save_branch_result(result, branch_dir / "branch.json")

    run_paths = write_run_bundle(
        run_root=branch_dir,
        run_id=operation_run_id,
        operation="preprocess",
        command=command,
        config_snapshot={
            "branch_name": config.branch_name,
            "source_collection_path": str(source_collection_path),
            "config": config.to_dict(),
        },
        source_artifacts=[
            RunArtifactRecord(
                artifact_type="epoch_collection",
                path=str(source_collection_path),
                role="source_collection",
                format="json",
            )
        ],
        generated_artifacts=[
            RunArtifactRecord(
                artifact_type="preprocess_branch",
                path=str(branch_result_path),
                role="branch_result",
                format="json",
                parent_paths=[str(source_collection_path)],
            ),
            RunArtifactRecord(
                artifact_type="epoch_collection",
                path=str(output_collection_path),
                role="output_collection",
                format="json",
                parent_paths=[str(source_collection_path)],
            ),
            *[
                RunArtifactRecord(
                    artifact_type="epoch_artifact",
                    path=str(branch_dir / artifact.signal_path),
                    role=artifact.event.event_id,
                    format=artifact.format,
                    parent_paths=[str(source_collection_path)],
                )
                for artifact in output_artifacts
            ],
        ],
        code_root=Path(__file__).resolve().parents[2],
        active_branches=[config.branch_name],
        warnings=branch_warnings,
        optional_dependencies=["numpy"],
    )
    output_collection.metadata = {**output_collection.metadata, **run_paths}
    output_collection_path = save_epoch_collection(output_collection, output_collection_path)
    logger.log(level="info", stage="preprocess", message="preprocessing completed", branch_name=config.branch_name)
    return branch_result_path, result
