from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .discovery import load_discovery_run_summary, package_discovery_run, save_discovery_run_summary
from .discovery_models import (
    BackboneConsensusSummary,
    BackboneEvaluationSummary,
    BackboneEvidenceSummary,
    CandidatePattern,
    DiscoveryRunSummary,
    RunIssueRecord,
)
from .eeg_registry import EEGPTAdapter, LoadedModel, ModelLoadError, default_registry
from .models import slugify
from .reliability import augment_reliability_with_backbone_consensus
from .run_tracking import RunArtifactRecord, StructuredRunLogger, make_run_id, write_run_bundle

BIOT_18_CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T3", "C3", "CZ", "C4", "T4",
    "T5", "P3", "P4", "T6", "O1", "O2",
]


class BackboneDiscoveryError(RuntimeError):
    pass


@dataclass(slots=True)
class PreparedBackboneInput:
    model_id: str
    prepared: np.ndarray
    prepared_channel_names: list[str]
    target_sampling_rate_hz: float
    selected_phases: list[str]
    notes: list[str]
    extras: dict[str, np.ndarray] = field(default_factory=dict)


EEGPT_TARGET_SAMPLES = 512
_BRAINOMNI_MIN_CHANNELS = 4

_CHANNEL_ALIAS_MAP = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
    "M1": "A1",
    "M2": "A2",
}

_ROW_Y_MAP = {
    "FP": 0.92,
    "AF": 0.78,
    "F": 0.58,
    "FT": 0.38,
    "FC": 0.32,
    "T": 0.02,
    "C": 0.0,
    "TP": -0.34,
    "CP": -0.30,
    "P": -0.56,
    "PO": -0.78,
    "O": -0.92,
    "A": -0.10,
}

_LATERAL_X_MAP = {
    0: 0.0,
    1: 0.18,
    2: 0.18,
    3: 0.34,
    4: 0.34,
    5: 0.50,
    6: 0.50,
    7: 0.74,
    8: 0.74,
    9: 0.88,
    10: 0.88,
}


def _normalize_channel_name(name: str) -> str:
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


def _canonical_eeg_name(name: str) -> str:
    normalized = _normalize_channel_name(name)
    return _CHANNEL_ALIAS_MAP.get(normalized, normalized)


def _approximate_eeg_xyz(name: str) -> np.ndarray | None:
    canonical = _canonical_eeg_name(name)
    if canonical in {"A1", "A2"}:
        x = -0.96 if canonical.endswith("1") else 0.96
        return np.asarray([x, -0.12, 0.0], dtype=np.float32)

    row = ""
    suffix = ""
    for prefix in sorted(_ROW_Y_MAP, key=len, reverse=True):
        if canonical.startswith(prefix):
            row = prefix
            suffix = canonical[len(prefix):]
            break
    if not row or not suffix:
        return None

    if suffix == "Z":
        x = 0.0
    else:
        try:
            lateral_index = int(suffix)
        except ValueError:
            return None
        magnitude = _LATERAL_X_MAP.get(abs(lateral_index))
        if magnitude is None:
            return None
        x = -magnitude if lateral_index % 2 else magnitude

    y = _ROW_Y_MAP[row]
    radial = min(0.995, math.sqrt((x * x) + (y * y)))
    z = math.sqrt(max(0.0, 1.0 - (radial * radial)))
    if row in {"P", "PO", "O", "A", "TP"}:
        z *= 0.8
    return np.asarray([x, y, z], dtype=np.float32)


def _normalize_brainomni_positions(pos: np.ndarray) -> np.ndarray:
    normalized = np.asarray(pos, dtype=np.float32).copy()
    if normalized.size == 0:
        return normalized
    mean_xyz = np.mean(normalized[:, :3], axis=0, keepdims=True)
    normalized[:, :3] -= mean_xyz
    scale = float(np.sqrt(3.0 * np.mean(np.sum(normalized[:, :3] ** 2, axis=1))))
    if scale > 0.0:
        normalized[:, :3] /= scale
    return normalized


def _brainomni_sensor_metadata(channel_names: list[str]) -> tuple[list[str], np.ndarray, list[int], list[str]]:
    kept_names: list[str] = []
    kept_positions: list[np.ndarray] = []
    kept_indices: list[int] = []
    dropped: list[str] = []
    for channel_index, channel_name in enumerate(channel_names):
        xyz = _approximate_eeg_xyz(channel_name)
        if xyz is None:
            dropped.append(channel_name)
            continue
        kept_names.append(channel_name)
        kept_indices.append(channel_index)
        kept_positions.append(np.concatenate([xyz, np.zeros(3, dtype=np.float32)]))
    if not kept_positions:
        return [], np.zeros((0, 6), dtype=np.float32), [], dropped
    normalized_positions = _normalize_brainomni_positions(np.stack(kept_positions, axis=0))
    return kept_names, normalized_positions.astype(np.float32), kept_indices, dropped


def _safe_corr(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    a = np.asarray(vector_a, dtype=np.float64).ravel()
    b = np.asarray(vector_b, dtype=np.float64).ravel()
    if a.size != b.size or a.size == 0:
        return 0.0
    mask = ~np.isnan(a) & ~np.isnan(b)
    if int(mask.sum()) < 2:
        return 0.0
    a = a[mask]
    b = b[mask]
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _mean_pairwise_similarity(vectors: np.ndarray) -> float | None:
    if vectors.shape[0] < 2:
        return None
    scores: list[float] = []
    for left_index in range(vectors.shape[0]):
        for right_index in range(left_index + 1, vectors.shape[0]):
            scores.append(_safe_corr(vectors[left_index], vectors[right_index]))
    return float(np.mean(scores)) if scores else None


def _mean_cross_similarity(left_vectors: np.ndarray, right_vectors: np.ndarray) -> float | None:
    if left_vectors.size == 0 or right_vectors.size == 0:
        return None
    scores = [_safe_corr(left, right) for left in left_vectors for right in right_vectors]
    return float(np.mean(scores)) if scores else None


def _candidate_context_key(candidate: CandidatePattern) -> tuple[str, str, str]:
    return candidate.label_namespace, candidate.event_family, candidate.branch_name


def _candidate_phase_names(candidate: CandidatePattern) -> list[str]:
    preferred = [phase for phase in candidate.strongest_phases if phase in set(candidate.phase_names)]
    if preferred:
        return preferred
    fallback = [phase for phase in ("onset", "sustained") if phase in set(candidate.phase_names)]
    if fallback:
        return fallback
    return list(candidate.phase_names)


def _load_subject_prototypes(candidate: CandidatePattern) -> tuple[np.ndarray, list[str], list[str], float]:
    path = Path(candidate.candidate_root) / candidate.artifact_paths["subject_prototypes"]
    with np.load(path, allow_pickle=False) as payload:
        channel_names = [str(item) for item in payload["channel_names"].tolist()]
        phase_names = [str(item) for item in payload["phase_names"].tolist()]
        subject_ids = [str(item) for item in payload["subject_ids"].tolist()]
        sampling_rate_hz = float(np.asarray(payload["sampling_rate_hz"]).reshape(-1)[0])
        selected_phases = _candidate_phase_names(candidate)
        stacks = []
        for phase_name in selected_phases:
            key = f"{phase_name}_stack"
            if key not in payload:
                continue
            stacks.append(np.asarray(payload[key], dtype=np.float32))
        if not stacks:
            raise BackboneDiscoveryError(f"candidate {candidate.pattern_id} has no subject prototype stacks for the selected phases")
        return np.concatenate(stacks, axis=-1), channel_names, subject_ids, sampling_rate_hz


def _resample_batch(data: np.ndarray, target_samples: int) -> np.ndarray:
    if data.shape[-1] == target_samples:
        return np.asarray(data, dtype=np.float32)
    source_positions = np.linspace(0.0, 1.0, data.shape[-1], dtype=np.float64)
    target_positions = np.linspace(0.0, 1.0, target_samples, dtype=np.float64)
    result = np.empty((data.shape[0], data.shape[1], target_samples), dtype=np.float32)
    for batch_index in range(data.shape[0]):
        for channel_index in range(data.shape[1]):
            result[batch_index, channel_index] = np.interp(
                target_positions,
                source_positions,
                np.asarray(data[batch_index, channel_index], dtype=np.float64),
            ).astype(np.float32)
    return result


def _align_channels(data: np.ndarray, source_channels: list[str], target_channels: list[str], *, fill_value: float = 0.0) -> tuple[np.ndarray, list[str]]:
    source_lookup = {_normalize_channel_name(name): index for index, name in enumerate(source_channels)}
    aligned = np.full((data.shape[0], len(target_channels), data.shape[-1]), fill_value, dtype=np.float32)
    missing: list[str] = []
    for channel_index, channel_name in enumerate(target_channels):
        lookup_key = _normalize_channel_name(channel_name)
        if lookup_key not in source_lookup:
            missing.append(channel_name)
            continue
        aligned[:, channel_index] = data[:, source_lookup[lookup_key]]
    return aligned, missing


def _prepare_biot(candidate: CandidatePattern) -> PreparedBackboneInput:
    subject_data, source_channels, _, _ = _load_subject_prototypes(candidate)
    aligned, missing = _align_channels(subject_data, source_channels, BIOT_18_CHANNELS, fill_value=0.0)
    prepared = _resample_batch(aligned, 2000)
    notes: list[str] = []
    if missing:
        notes.append("missing channels were zero-filled: " + ", ".join(missing))
    return PreparedBackboneInput(
        model_id="biot",
        prepared=prepared,
        prepared_channel_names=list(BIOT_18_CHANNELS),
        target_sampling_rate_hz=200.0,
        selected_phases=_candidate_phase_names(candidate),
        notes=notes,
    )


def _prepare_cbramod(candidate: CandidatePattern) -> PreparedBackboneInput:
    subject_data, source_channels, _, _ = _load_subject_prototypes(candidate)
    resampled = _resample_batch(subject_data, 2000)
    prepared = resampled.reshape(resampled.shape[0], resampled.shape[1], 10, 200)
    return PreparedBackboneInput(
        model_id="cbramod",
        prepared=prepared,
        prepared_channel_names=list(source_channels),
        target_sampling_rate_hz=200.0,
        selected_phases=_candidate_phase_names(candidate),
        notes=[],
    )


def _prepare_eegpt(candidate: CandidatePattern) -> PreparedBackboneInput:
    subject_data, source_channels, _, sampling_rate_hz = _load_subject_prototypes(candidate)
    aligned, missing = _align_channels(subject_data, source_channels, EEGPTAdapter.CANONICAL_CHANNELS, fill_value=0.0)
    prepared = _resample_batch(aligned, EEGPT_TARGET_SAMPLES)
    notes = [
        f"resampled concatenated subject prototypes from {sampling_rate_hz:.3f} Hz phase timing into {EEGPT_TARGET_SAMPLES} samples for EEGPT patch geometry",
    ]
    if missing:
        notes.append("missing canonical channels were zero-filled: " + ", ".join(missing))
    return PreparedBackboneInput(
        model_id="eegpt",
        prepared=prepared,
        prepared_channel_names=list(EEGPTAdapter.CANONICAL_CHANNELS),
        target_sampling_rate_hz=256.0,
        selected_phases=_candidate_phase_names(candidate),
        notes=notes,
    )


def _prepare_brainomni(candidate: CandidatePattern) -> PreparedBackboneInput:
    subject_data, source_channels, _, sampling_rate_hz = _load_subject_prototypes(candidate)
    kept_names, positions, kept_indices, dropped = _brainomni_sensor_metadata(source_channels)
    if len(kept_indices) < _BRAINOMNI_MIN_CHANNELS:
        raise BackboneDiscoveryError(
            f"candidate {candidate.pattern_id} only had {len(kept_indices)} channels with BrainOmni sensor metadata; at least {_BRAINOMNI_MIN_CHANNELS} are required"
        )
    prepared = subject_data[:, kept_indices, :]
    sensor_type = np.zeros((prepared.shape[0], prepared.shape[1]), dtype=np.int64)
    notes = [
        f"BrainOmni used {len(kept_indices)} of {len(source_channels)} channels with deterministic 10-20 position inference",
        f"preserved source sampling rate {sampling_rate_hz:.3f} Hz for tokenizer windowing",
    ]
    if dropped:
        notes.append("channels without deterministic BrainOmni position mapping were excluded: " + ", ".join(dropped))
    return PreparedBackboneInput(
        model_id="brainomni",
        prepared=prepared,
        prepared_channel_names=kept_names,
        target_sampling_rate_hz=sampling_rate_hz,
        selected_phases=_candidate_phase_names(candidate),
        notes=notes,
        extras={
            "pos": np.broadcast_to(positions[np.newaxis, :, :], (prepared.shape[0], positions.shape[0], positions.shape[1])).copy(),
            "sensor_type": sensor_type,
        },
    )


SUPPORTED_PREPARERS = {
    "biot": _prepare_biot,
    "cbramod": _prepare_cbramod,
    "eegpt": _prepare_eegpt,
    "brainomni": _prepare_brainomni,
}


def _extract_embeddings(loaded_model: LoadedModel, prepared: PreparedBackboneInput) -> np.ndarray:
    import torch

    model_id = loaded_model.model_id
    tensor = torch.as_tensor(prepared.prepared, dtype=torch.float32, device=loaded_model.device)
    with torch.no_grad():
        if model_id == "biot":
            embeddings = loaded_model.model(tensor)
        elif model_id == "cbramod":
            embeddings = loaded_model.model(tensor).mean(dim=(1, 2))
        elif model_id == "eegpt":
            chan_ids = loaded_model.metadata.get("channel_ids")
            embeddings = loaded_model.model(tensor, chan_ids=chan_ids).mean(dim=(1, 2))
        elif model_id == "brainomni":
            pos = torch.as_tensor(prepared.extras["pos"], dtype=torch.float32, device=loaded_model.device)
            sensor_type = torch.as_tensor(prepared.extras["sensor_type"], dtype=torch.long, device=loaded_model.device)
            embeddings = loaded_model.model.encode(tensor, pos, sensor_type).mean(dim=(1, 2))
        else:
            raise BackboneDiscoveryError(f"no extraction path is implemented for backbone {model_id!r}")
    return np.asarray(embeddings.detach().cpu(), dtype=np.float32)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _compute_consensus(evidence_items: list[BackboneEvidenceSummary], requested_model_ids: list[str]) -> BackboneConsensusSummary:
    successful = [item for item in evidence_items if item.status == "used"]
    successful_model_ids = [item.model_id for item in successful]
    positive_models = [item.model_id for item in successful if item.margin_vs_rest is not None and item.margin_vs_rest > 0.0]
    margins_by_model = {item.model_id: item.margin_vs_rest for item in evidence_items}
    if not successful:
        return BackboneConsensusSummary(
            requested_model_ids=list(requested_model_ids),
            successful_model_ids=[],
            agreeing_model_ids=[],
            overall_status="insufficient",
            notes=["no requested backbone produced usable evidence"],
            margins_by_model=margins_by_model,
        )
    mean_margin = float(np.mean([item.margin_vs_rest for item in successful if item.margin_vs_rest is not None])) if any(
        item.margin_vs_rest is not None for item in successful
    ) else None
    if len(successful) >= 2 and len(positive_models) == len(successful):
        status = "strong_agreement"
    elif positive_models:
        status = "partial_agreement"
    else:
        status = "weak_agreement"
    return BackboneConsensusSummary(
        requested_model_ids=list(requested_model_ids),
        successful_model_ids=successful_model_ids,
        agreeing_model_ids=positive_models,
        overall_status=status,
        mean_margin=mean_margin,
        margins_by_model=margins_by_model,
    )


def evaluate_backbones_for_run(
    run_summary: DiscoveryRunSummary | str | Path,
    *,
    backbone_ids: list[str],
    device: str = "cpu",
    package_root: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, DiscoveryRunSummary]:
    summary = load_discovery_run_summary(run_summary) if isinstance(run_summary, (str, Path)) else run_summary
    requested_model_ids = sorted({item.strip().lower() for item in backbone_ids if item.strip()})
    if not requested_model_ids:
        raise BackboneDiscoveryError("at least one backbone ID is required")

    run_dir = Path(summary.output_root)
    evaluation_id = make_run_id("backbone_eval", f"{summary.run_id}_backbones")
    evaluation_dir = run_dir / "backbone_evaluation" / evaluation_id
    if evaluation_dir.exists() and overwrite:
        shutil.rmtree(evaluation_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    logger = StructuredRunLogger(evaluation_dir / "run.log.jsonl")
    registry = default_registry()
    notes: list[str] = []
    failures: list[RunIssueRecord] = []
    generated_artifacts: list[RunArtifactRecord] = []
    source_artifacts: list[RunArtifactRecord] = [
        RunArtifactRecord(artifact_type="discovery_run", path=str(run_dir / "run_summary.json"), role="source_run", format="json")
    ]

    loaded_models: dict[str, LoadedModel] = {}
    completed_model_ids: list[str] = []
    failed_model_ids: list[str] = []
    for model_id in requested_model_ids:
        if model_id not in SUPPORTED_PREPARERS:
            failed_model_ids.append(model_id)
            notes.append(f"backbone {model_id} is not yet supported for automated evaluation")
            continue
        try:
            loaded_models[model_id] = registry.load(model_id, device=device)
            completed_model_ids.append(model_id)
            logger.log(level="info", stage="backbone_load", message="loaded backbone", model_id=model_id, device=device)
        except (KeyError, ModelLoadError) as exc:
            failed_model_ids.append(model_id)
            notes.append(f"backbone {model_id} could not be loaded: {exc}")
            logger.log(level="warning", stage="backbone_load", message=str(exc), model_id=model_id)

    embeddings_by_model: dict[str, dict[str, np.ndarray]] = {model_id: {} for model_id in completed_model_ids}
    evidence_by_candidate: dict[str, list[BackboneEvidenceSummary]] = {candidate.pattern_id: [] for candidate in summary.candidates}

    for candidate in summary.candidates:
        candidate_dir = Path(candidate.candidate_root)
        for model_id in requested_model_ids:
            if model_id not in loaded_models:
                evidence_by_candidate[candidate.pattern_id].append(
                    BackboneEvidenceSummary(
                        model_id=model_id,
                        status="skipped",
                        notes=[f"backbone {model_id} was unavailable for this evaluation run"],
                    )
                )
                continue
            try:
                prepared = SUPPORTED_PREPARERS[model_id](candidate)
                embeddings = _extract_embeddings(loaded_models[model_id], prepared)
                model_dir = candidate_dir / "backbones" / slugify(model_id)
                embeddings_path = model_dir / "embeddings.npz"
                _write_npz(
                    embeddings_path,
                    {
                        "embeddings": embeddings.astype(np.float32),
                        "subject_ids": np.asarray(candidate.subject_ids),
                        "selected_phases": np.asarray(prepared.selected_phases),
                        "channel_names": np.asarray(prepared.prepared_channel_names),
                        "sampling_rate_hz": np.asarray([prepared.target_sampling_rate_hz], dtype=np.float32),
                    },
                )
                embeddings_by_model[model_id][candidate.pattern_id] = embeddings
                evidence = BackboneEvidenceSummary(
                    model_id=model_id,
                    variant_id=loaded_models[model_id].variant_id,
                    status="used",
                    subject_count=embeddings.shape[0],
                    embedding_dim=embeddings.shape[1] if embeddings.ndim == 2 else 0,
                    selected_phases=list(prepared.selected_phases),
                    artifact_paths={
                        "embeddings_npz": str(embeddings_path.relative_to(candidate_dir)),
                    },
                    notes=list(prepared.notes),
                )
                evidence_by_candidate[candidate.pattern_id].append(evidence)
                candidate.artifact_paths[f"backbone_{model_id}_embeddings"] = str(embeddings_path.relative_to(candidate_dir))
                generated_artifacts.append(
                    RunArtifactRecord(
                        artifact_type="backbone_embeddings",
                        path=str(embeddings_path),
                        role=f"{candidate.pattern_id}:{model_id}:embeddings",
                        format="npz",
                        parent_paths=[str(candidate_dir / candidate.artifact_paths["subject_prototypes"])],
                    )
                )
            except Exception as exc:
                failures.append(
                    RunIssueRecord(
                        stage="backbone_eval",
                        severity="warning",
                        message=str(exc),
                        target_label=candidate.target_label,
                        branch_name=candidate.branch_name,
                        path=str(candidate_dir),
                    )
                )
                logger.log(level="warning", stage="backbone_eval", message=str(exc), model_id=model_id, pattern_id=candidate.pattern_id)
                evidence_by_candidate[candidate.pattern_id].append(
                    BackboneEvidenceSummary(
                        model_id=model_id,
                        status="failed",
                        notes=[str(exc)],
                    )
                )

    candidates_by_context: dict[tuple[str, str, str], list[CandidatePattern]] = {}
    for candidate in summary.candidates:
        candidates_by_context.setdefault(_candidate_context_key(candidate), []).append(candidate)

    for context_candidates in candidates_by_context.values():
        for candidate in context_candidates:
            evidence_items = evidence_by_candidate[candidate.pattern_id]
            for evidence in evidence_items:
                if evidence.status != "used":
                    continue
                current_embeddings = embeddings_by_model[evidence.model_id].get(candidate.pattern_id)
                if current_embeddings is None:
                    continue
                other_candidates = [item for item in context_candidates if item.pattern_id != candidate.pattern_id]
                other_scores: dict[str, float] = {}
                for other in other_candidates:
                    other_embeddings = embeddings_by_model[evidence.model_id].get(other.pattern_id)
                    if other_embeddings is None:
                        continue
                    similarity = _mean_cross_similarity(current_embeddings, other_embeddings)
                    if similarity is not None:
                        other_scores[other.target_label] = similarity
                evidence.within_label_similarity = _mean_pairwise_similarity(current_embeddings)
                evidence.target_vs_rest_similarity = float(np.mean(list(other_scores.values()))) if other_scores else None
                evidence.margin_vs_rest = (
                    None
                    if evidence.within_label_similarity is None or evidence.target_vs_rest_similarity is None
                    else float(evidence.within_label_similarity - evidence.target_vs_rest_similarity)
                )
                if other_scores:
                    evidence.strongest_negative_control_label = max(other_scores, key=other_scores.get)
                    evidence.strongest_negative_control_similarity = other_scores[evidence.strongest_negative_control_label]
                evidence_path = Path(candidate.candidate_root) / "backbones" / slugify(evidence.model_id) / "evidence.json"
                _write_json(evidence_path, evidence.to_dict())
                evidence.artifact_paths["evidence_json"] = str(evidence_path.relative_to(Path(candidate.candidate_root)))
                candidate.artifact_paths[f"backbone_{evidence.model_id}_evidence_json"] = str(evidence_path.relative_to(Path(candidate.candidate_root)))
                generated_artifacts.append(
                    RunArtifactRecord(
                        artifact_type="backbone_evidence",
                        path=str(evidence_path),
                        role=f"{candidate.pattern_id}:{evidence.model_id}:evidence",
                        format="json",
                        parent_paths=[str(run_dir / "run_summary.json")],
                    )
                )
            candidate.backbone_evidence = evidence_items
            candidate.backbone_consensus = _compute_consensus(evidence_items, requested_model_ids)
            consensus_path = Path(candidate.candidate_root) / "backbone_consensus.json"
            _write_json(consensus_path, candidate.backbone_consensus.to_dict())
            candidate.artifact_paths["backbone_consensus_json"] = str(consensus_path.relative_to(Path(candidate.candidate_root)))
            generated_artifacts.append(
                RunArtifactRecord(
                    artifact_type="backbone_consensus",
                    path=str(consensus_path),
                    role=f"{candidate.pattern_id}:consensus",
                    format="json",
                    parent_paths=[str(run_dir / "run_summary.json")],
                )
            )
            candidate.reliability = augment_reliability_with_backbone_consensus(
                candidate.reliability,
                backbone_consensus=candidate.backbone_consensus,
                backbone_evidence=evidence_items,
            )
            if candidate.reliability is not None:
                reliability_path = Path(candidate.candidate_root) / candidate.artifact_paths.get("reliability_json", "reliability.json")
                _write_json(reliability_path, candidate.reliability.to_dict())
                candidate.artifact_paths["reliability_json"] = str(reliability_path.relative_to(Path(candidate.candidate_root)))
                generated_artifacts.append(
                    RunArtifactRecord(
                        artifact_type="candidate_reliability",
                        path=str(reliability_path),
                        role=f"{candidate.pattern_id}:reliability",
                        format="json",
                        parent_paths=[str(consensus_path)],
                    )
                )
            candidate_json_path = Path(candidate.candidate_root) / candidate.artifact_paths.get("candidate_json", "candidate.json")
            candidate_json_path.write_text(json.dumps(candidate.to_dict(), indent=2), encoding="utf-8")

    evaluation_status = "partial" if failed_model_ids or failures else "success"
    run_bundle = write_run_bundle(
        run_root=evaluation_dir,
        run_id=evaluation_id,
        operation="backbone_evaluation",
        command="",
        config_snapshot={
            "source_run_id": summary.run_id,
            "requested_model_ids": requested_model_ids,
            "device": device,
            "package_root": str(Path(package_root).resolve()) if package_root is not None else "",
        },
        source_artifacts=source_artifacts,
        generated_artifacts=generated_artifacts,
        code_root=Path(__file__).resolve().parents[2],
        active_branches=list(summary.branch_names),
        warnings=notes + [item.message for item in failures if item.severity == "warning"],
        status=evaluation_status,
        optional_dependencies=["numpy", "torch"],
    )

    summary.backbone_ids = requested_model_ids
    summary.backbone_evaluation = BackboneEvaluationSummary(
        evaluation_id=evaluation_id,
        output_root=str(evaluation_dir),
        requested_model_ids=requested_model_ids,
        completed_model_ids=completed_model_ids,
        failed_model_ids=failed_model_ids,
        candidate_count=len(summary.candidates),
        status=evaluation_status,
        run_manifest_path=run_bundle["run_manifest_path"],
        config_snapshot_path=run_bundle["config_snapshot_path"],
        environment_path=run_bundle["environment_path"],
        artifact_lineage_path=run_bundle["artifact_lineage_path"],
        warnings_path=run_bundle["warnings_path"],
        log_path=run_bundle["log_path"],
        notes=notes,
    )
    summary.failures.extend(failures)
    if evaluation_status == "partial" and summary.status == "success":
        summary.status = "partial"
    summary_path = save_discovery_run_summary(summary, run_dir / "run_summary.json")
    logger.log(level="info", stage="backbone_eval", message="backbone evaluation completed", completed_model_ids=completed_model_ids, failed_model_ids=failed_model_ids)

    if package_root is not None:
        package_discovery_run(summary, package_root=package_root, overwrite=overwrite)
        summary_path = save_discovery_run_summary(summary, run_dir / "run_summary.json")

    return summary_path, summary
