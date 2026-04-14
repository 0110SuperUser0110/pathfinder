from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .analysis_models import EpochCollection
from .discovery_models import (
    BranchAgreement,
    CandidatePattern,
    ControlComparisonSummary,
    CrossSubjectAgreement,
    DiscoveryRunSummary,
    RunIssueRecord,
)
from .epochs import load_epoch_collection
from .models import AnalysisSpec, PartitionSpec, PatternSummary, slugify
from .package import ArtifactInput, PatternPackageBuilder
from .reliability import ReliabilityContextGroup, assess_candidate_reliability
from .run_tracking import RunArtifactRecord, StructuredRunLogger, make_run_id, write_run_bundle
from .validation import validate_study

PHASE_ORDER = ["pre_event", "onset", "sustained", "offset", "post_event"]
DEFAULT_BAND_DEFINITIONS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
]


@dataclass(slots=True)
class _EpochObservation:
    collection_path: Path
    collection: EpochCollection
    branch_name: str
    subject_id: str
    session_id: str
    cohort_label: str
    channel_names: list[str]
    event_id: str
    event_family: str
    target_label: str
    event_subtype: str
    label_namespace: str
    arrays: dict[str, np.ndarray]


@dataclass(slots=True)
class _ComputedGroup:
    label_namespace: str
    event_family: str
    target_label: str
    branch_name: str
    event_subtypes: list[str]
    subject_ids: list[str]
    event_ids: list[str]
    channel_names: list[str]
    phase_names: list[str]
    available_bands: list[str]
    sampling_rate_hz: float
    prototype_arrays: dict[str, np.ndarray]
    subject_stacks: dict[str, np.ndarray]
    exemplar_stacks: dict[str, np.ndarray]
    exemplar_subject_ids: list[str]
    exemplar_event_ids: list[str]
    subject_bandpower: np.ndarray
    event_bandpower: np.ndarray
    mean_bandpower: np.ndarray
    subject_similarity_matrix: np.ndarray
    event_similarity_matrix: np.ndarray
    subject_features: np.ndarray
    event_features: np.ndarray
    profile_matrix: np.ndarray
    within_label_similarity: float | None
    subject_consistency_score: float | None
    trial_consistency_score: float | None
    session_ids: list[str]
    cohort_labels: list[str]
    rest_mean_bandpower: np.ndarray | None
    difference_bandpower: np.ndarray | None
    candidate_dir: Path
    artifact_paths: dict[str, str] = field(default_factory=dict)
    summary_notes: list[str] = field(default_factory=list)
    control_summary: ControlComparisonSummary | None = None
    cross_subject_agreement: CrossSubjectAgreement | None = None
    dominant_bands: list[str] = field(default_factory=list)
    dominant_channels: list[str] = field(default_factory=list)
    strongest_phases: list[str] = field(default_factory=list)
    candidate: CandidatePattern | None = None


class DiscoveryError(RuntimeError):
    pass


def save_discovery_run_summary(summary: DiscoveryRunSummary, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return output_path


def load_discovery_run_summary(path: str | Path) -> DiscoveryRunSummary:
    return DiscoveryRunSummary.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _infer_branch_name(collection: EpochCollection) -> str:
    branch_name = str(collection.metadata.get("branch_name", "")).strip()
    return branch_name or "source_epoch"


def _load_artifact_arrays(collection_dir: Path, signal_path: str) -> dict[str, np.ndarray]:
    with np.load(collection_dir / signal_path, allow_pickle=False) as payload:
        return {
            key: np.asarray(payload[key], dtype=np.float32)
            for key in payload.files
            if key in PHASE_ORDER and np.asarray(payload[key]).ndim == 2
        }


def _ordered_intersection(sequences: list[list[str]]) -> list[str]:
    if not sequences:
        return []
    common = set(sequences[0])
    for sequence in sequences[1:]:
        common &= set(sequence)
    return [item for item in sequences[0] if item in common]


def _safe_corr(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    a = np.asarray(vector_a, dtype=np.float64).ravel()
    b = np.asarray(vector_b, dtype=np.float64).ravel()
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
    for left_index, right_index in combinations(range(vectors.shape[0]), 2):
        scores.append(_safe_corr(vectors[left_index], vectors[right_index]))
    return float(np.mean(scores)) if scores else None


def _mean_cross_similarity(left_vectors: np.ndarray, right_vectors: np.ndarray) -> float | None:
    if left_vectors.size == 0 or right_vectors.size == 0:
        return None
    scores = [_safe_corr(left, right) for left in left_vectors for right in right_vectors]
    return float(np.mean(scores)) if scores else None


def _similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    matrix = np.eye(vectors.shape[0], dtype=np.float32)
    for left_index in range(vectors.shape[0]):
        for right_index in range(left_index + 1, vectors.shape[0]):
            similarity = _safe_corr(vectors[left_index], vectors[right_index])
            matrix[left_index, right_index] = similarity
            matrix[right_index, left_index] = similarity
    return matrix


def _subject_consistency(vectors: np.ndarray) -> float | None:
    if vectors.shape[0] == 0:
        return None
    centroid = np.nanmean(vectors, axis=0)
    scores = [_safe_corr(vector, centroid) for vector in vectors]
    return float(np.mean(scores)) if scores else None


def _available_bands(sampling_rate_hz: float) -> list[tuple[str, float, float]]:
    nyquist = sampling_rate_hz / 2.0
    bands = [band for band in DEFAULT_BAND_DEFINITIONS if band[2] <= nyquist]
    if bands:
        return bands
    fallback_high = max(0.5, nyquist)
    return [("broadband", 0.0, fallback_high)]


def _bandpower(signal: np.ndarray, sampling_rate_hz: float, bands: list[tuple[str, float, float]]) -> np.ndarray:
    if signal.shape[1] == 0:
        return np.zeros((signal.shape[0], len(bands)), dtype=np.float32)
    fft = np.fft.rfft(signal, axis=1)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(signal.shape[1], d=1.0 / sampling_rate_hz)
    bandpower = np.zeros((signal.shape[0], len(bands)), dtype=np.float32)
    for band_index, (_, low_hz, high_hz) in enumerate(bands):
        mask = (freqs >= low_hz) & (freqs < high_hz)
        if not np.any(mask):
            continue
        bandpower[:, band_index] = np.mean(power[:, mask], axis=1, dtype=np.float64).astype(np.float32)
    return bandpower


def _phase_bandpower(
    arrays_by_phase: dict[str, np.ndarray],
    phase_names: list[str],
    sampling_rate_hz: float,
    bands: list[tuple[str, float, float]],
) -> np.ndarray:
    tensors = [_bandpower(arrays_by_phase[phase_name], sampling_rate_hz, bands) for phase_name in phase_names]
    return np.stack(tensors, axis=0).astype(np.float32)


def _flatten_feature(tensor: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(tensor, dtype=np.float32)).reshape(-1)


def _aligned_event_arrays(
    observations: list[_EpochObservation],
) -> tuple[list[dict[str, np.ndarray]], list[str], list[str], dict[str, int]]:
    common_channels = _ordered_intersection([observation.channel_names for observation in observations])
    if not common_channels:
        raise DiscoveryError("No common channels were available across the candidate observations")
    phase_names = [
        phase_name
        for phase_name in PHASE_ORDER
        if all(phase_name in observation.arrays for observation in observations)
    ]
    if not phase_names:
        raise DiscoveryError("No common event phases were available across the candidate observations")
    common_lengths = {
        phase_name: min(observation.arrays[phase_name].shape[1] for observation in observations)
        for phase_name in phase_names
    }
    aligned: list[dict[str, np.ndarray]] = []
    for observation in observations:
        channel_lookup = {name: index for index, name in enumerate(observation.channel_names)}
        channel_indices = [channel_lookup[name] for name in common_channels]
        arrays_by_phase = {
            phase_name: np.asarray(
                observation.arrays[phase_name][channel_indices, : common_lengths[phase_name]],
                dtype=np.float32,
            )
            for phase_name in phase_names
        }
        aligned.append(arrays_by_phase)
    return aligned, common_channels, phase_names, common_lengths


def _observation_vector(arrays_by_phase: dict[str, np.ndarray], phase_names: list[str]) -> np.ndarray:
    return np.concatenate([arrays_by_phase[phase_name].reshape(-1) for phase_name in phase_names]).astype(np.float32)


def _select_exemplars(
    aligned_events: list[dict[str, np.ndarray]],
    subject_ids: list[str],
    event_ids: list[str],
    phase_names: list[str],
    prototype_vector: np.ndarray,
    max_exemplars: int,
) -> list[int]:
    similarities = [
        _safe_corr(_observation_vector(arrays_by_phase, phase_names), prototype_vector)
        for arrays_by_phase in aligned_events
    ]
    subject_best: list[int] = []
    for subject_id in sorted(set(subject_ids)):
        subject_indices = [index for index, item in enumerate(subject_ids) if item == subject_id]
        best_index = max(subject_indices, key=lambda index: similarities[index])
        subject_best.append(best_index)
    ranked = sorted(range(len(aligned_events)), key=lambda index: similarities[index], reverse=True)
    selected: list[int] = []
    for index in subject_best + ranked:
        if index not in selected:
            selected.append(index)
        if len(selected) >= max_exemplars:
            break
    return selected


def _write_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _group_key(observation: _EpochObservation) -> tuple[str, str, str, str]:
    return (
        observation.label_namespace,
        observation.event_family,
        observation.target_label,
        observation.branch_name,
    )


def _compute_group(
    observations: list[_EpochObservation],
    *,
    candidate_dir: Path,
    max_exemplars: int,
) -> _ComputedGroup:
    aligned_events, common_channels, phase_names, _ = _aligned_event_arrays(observations)
    sampling_rate_hz = float(observations[0].collection.sampling_rate_hz)
    bands = _available_bands(sampling_rate_hz)
    band_names = [name for name, _, _ in bands]
    event_ids = [observation.event_id for observation in observations]
    subject_ids = [observation.subject_id for observation in observations]
    subject_to_session = {
        observation.subject_id: observation.session_id
        for observation in observations
        if observation.subject_id and observation.session_id
    }
    subject_to_cohort = {
        observation.subject_id: observation.cohort_label
        for observation in observations
        if observation.subject_id and observation.cohort_label
    }

    subject_event_arrays: dict[str, list[dict[str, np.ndarray]]] = {}
    for subject_id, arrays_by_phase in zip(subject_ids, aligned_events, strict=True):
        subject_event_arrays.setdefault(subject_id, []).append(arrays_by_phase)

    ordered_subject_ids = sorted(subject_event_arrays)
    subject_prototypes: dict[str, dict[str, np.ndarray]] = {}
    for subject_id in ordered_subject_ids:
        subject_prototypes[subject_id] = {
            phase_name: np.mean(
                np.stack([event_arrays[phase_name] for event_arrays in subject_event_arrays[subject_id]], axis=0),
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)
            for phase_name in phase_names
        }

    prototype_arrays = {
        phase_name: np.mean(
            np.stack([subject_prototypes[subject_id][phase_name] for subject_id in ordered_subject_ids], axis=0),
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        for phase_name in phase_names
    }
    prototype_vector = _observation_vector(prototype_arrays, phase_names)
    exemplar_indices = _select_exemplars(
        aligned_events,
        subject_ids,
        event_ids,
        phase_names,
        prototype_vector,
        max_exemplars=max_exemplars,
    )

    subject_stacks = {
        phase_name: np.stack([subject_prototypes[subject_id][phase_name] for subject_id in ordered_subject_ids], axis=0)
        for phase_name in phase_names
    }
    exemplar_stacks = {
        phase_name: np.stack([aligned_events[index][phase_name] for index in exemplar_indices], axis=0)
        for phase_name in phase_names
    }
    exemplar_subject_ids = [subject_ids[index] for index in exemplar_indices]
    exemplar_event_ids = [event_ids[index] for index in exemplar_indices]

    subject_bandpower = np.stack(
        [
            _phase_bandpower(subject_prototypes[subject_id], phase_names, sampling_rate_hz, bands)
            for subject_id in ordered_subject_ids
        ],
        axis=0,
    )
    event_bandpower = np.stack(
        [_phase_bandpower(arrays_by_phase, phase_names, sampling_rate_hz, bands) for arrays_by_phase in aligned_events],
        axis=0,
    )
    mean_bandpower = np.mean(subject_bandpower, axis=0, dtype=np.float64).astype(np.float32)
    subject_features = np.stack([_flatten_feature(item) for item in subject_bandpower], axis=0)
    event_features = np.stack([_flatten_feature(item) for item in event_bandpower], axis=0)
    subject_similarity_matrix = _similarity_matrix(subject_features)
    event_similarity_matrix = _similarity_matrix(event_features)
    within_label_similarity = _mean_pairwise_similarity(subject_features)
    subject_consistency_score = _subject_consistency(subject_features)
    trial_consistency_score = _subject_consistency(event_features)
    profile_matrix = np.mean(mean_bandpower, axis=1, dtype=np.float64).astype(np.float32)

    summary_notes: list[str] = []
    if len(observations) != len(ordered_subject_ids):
        summary_notes.append("multiple events per subject were averaged into subject-level prototypes")

    return _ComputedGroup(
        label_namespace=observations[0].label_namespace,
        event_family=observations[0].event_family,
        target_label=observations[0].target_label,
        branch_name=observations[0].branch_name,
        event_subtypes=sorted({observation.event_subtype for observation in observations if observation.event_subtype}),
        subject_ids=ordered_subject_ids,
        event_ids=event_ids,
        channel_names=common_channels,
        phase_names=phase_names,
        available_bands=band_names,
        sampling_rate_hz=sampling_rate_hz,
        prototype_arrays=prototype_arrays,
        subject_stacks=subject_stacks,
        exemplar_stacks=exemplar_stacks,
        exemplar_subject_ids=exemplar_subject_ids,
        exemplar_event_ids=exemplar_event_ids,
        subject_bandpower=subject_bandpower,
        event_bandpower=event_bandpower,
        mean_bandpower=mean_bandpower,
        subject_similarity_matrix=subject_similarity_matrix,
        event_similarity_matrix=event_similarity_matrix,
        subject_features=subject_features,
        event_features=event_features,
        profile_matrix=profile_matrix,
        within_label_similarity=within_label_similarity,
        subject_consistency_score=subject_consistency_score,
        trial_consistency_score=trial_consistency_score,
        session_ids=[subject_to_session.get(subject_id, "") for subject_id in ordered_subject_ids],
        cohort_labels=[subject_to_cohort.get(subject_id, "") for subject_id in ordered_subject_ids],
        rest_mean_bandpower=None,
        difference_bandpower=None,
        candidate_dir=candidate_dir,
        summary_notes=summary_notes,
    )

def _align_mean_bandpower(
    source: _ComputedGroup,
    phase_names: list[str],
    channel_names: list[str],
    band_names: list[str],
) -> np.ndarray:
    aligned = np.full((len(phase_names), len(channel_names), len(band_names)), np.nan, dtype=np.float32)
    phase_lookup = {name: index for index, name in enumerate(source.phase_names)}
    channel_lookup = {name: index for index, name in enumerate(source.channel_names)}
    band_lookup = {name: index for index, name in enumerate(source.available_bands)}
    for phase_index, phase_name in enumerate(phase_names):
        if phase_name not in phase_lookup:
            continue
        for channel_index, channel_name in enumerate(channel_names):
            if channel_name not in channel_lookup:
                continue
            for band_index, band_name in enumerate(band_names):
                if band_name not in band_lookup:
                    continue
                aligned[phase_index, channel_index, band_index] = source.mean_bandpower[
                    phase_lookup[phase_name],
                    channel_lookup[channel_name],
                    band_lookup[band_name],
                ]
    return aligned


def _align_subject_bandpower(
    source: _ComputedGroup,
    phase_names: list[str],
    channel_names: list[str],
    band_names: list[str],
) -> np.ndarray:
    aligned = np.full(
        (source.subject_bandpower.shape[0], len(phase_names), len(channel_names), len(band_names)),
        np.nan,
        dtype=np.float32,
    )
    phase_lookup = {name: index for index, name in enumerate(source.phase_names)}
    channel_lookup = {name: index for index, name in enumerate(source.channel_names)}
    band_lookup = {name: index for index, name in enumerate(source.available_bands)}
    for phase_index, phase_name in enumerate(phase_names):
        if phase_name not in phase_lookup:
            continue
        for channel_index, channel_name in enumerate(channel_names):
            if channel_name not in channel_lookup:
                continue
            for band_index, band_name in enumerate(band_names):
                if band_name not in band_lookup:
                    continue
                aligned[:, phase_index, channel_index, band_index] = source.subject_bandpower[
                    :,
                    phase_lookup[phase_name],
                    channel_lookup[channel_name],
                    band_lookup[band_name],
                ]
    return aligned


def _group_common_axes(left: _ComputedGroup, right: _ComputedGroup) -> tuple[list[str], list[str], list[str]]:
    phase_names = [phase for phase in left.phase_names if phase in set(right.phase_names)]
    channel_names = [channel for channel in left.channel_names if channel in set(right.channel_names)]
    band_names = [band for band in left.available_bands if band in set(right.available_bands)]
    return phase_names, channel_names, band_names


def _group_cross_similarity(left: _ComputedGroup, right: _ComputedGroup) -> float | None:
    phase_names, channel_names, band_names = _group_common_axes(left, right)
    if not phase_names or not channel_names or not band_names:
        return None
    left_features = _align_subject_bandpower(left, phase_names, channel_names, band_names).reshape(left.subject_bandpower.shape[0], -1)
    right_features = _align_subject_bandpower(right, phase_names, channel_names, band_names).reshape(right.subject_bandpower.shape[0], -1)
    return _mean_cross_similarity(left_features, right_features)


def _group_profile_similarity(left: _ComputedGroup, right: _ComputedGroup) -> float | None:
    phase_names = [phase for phase in left.phase_names if phase in set(right.phase_names)]
    band_names = [band for band in left.available_bands if band in set(right.available_bands)]
    if not phase_names or not band_names:
        return None
    left_phase_lookup = {name: index for index, name in enumerate(left.phase_names)}
    right_phase_lookup = {name: index for index, name in enumerate(right.phase_names)}
    left_band_lookup = {name: index for index, name in enumerate(left.available_bands)}
    right_band_lookup = {name: index for index, name in enumerate(right.available_bands)}
    left_vector = np.array(
        [left.profile_matrix[left_phase_lookup[phase], left_band_lookup[band]] for phase in phase_names for band in band_names],
        dtype=np.float32,
    )
    right_vector = np.array(
        [right.profile_matrix[right_phase_lookup[phase], right_band_lookup[band]] for phase in phase_names for band in band_names],
        dtype=np.float32,
    )
    return _safe_corr(left_vector, right_vector)


def _populate_controls(target_group: _ComputedGroup, other_groups: list[_ComputedGroup]) -> None:
    other_label_similarities: dict[str, float] = {}
    notes: list[str] = []
    for other_group in other_groups:
        similarity = _group_cross_similarity(target_group, other_group)
        if similarity is not None:
            other_label_similarities[other_group.target_label] = similarity
    target_vs_rest = float(np.mean(list(other_label_similarities.values()))) if other_label_similarities else None
    strongest_label = ""
    strongest_similarity = None
    if other_label_similarities:
        strongest_label = max(other_label_similarities, key=other_label_similarities.get)
        strongest_similarity = other_label_similarities[strongest_label]
    else:
        notes.append("no negative-control labels were available in the same branch context")

    margin = None
    if target_group.within_label_similarity is not None and target_vs_rest is not None:
        margin = target_group.within_label_similarity - target_vs_rest

    target_group.control_summary = ControlComparisonSummary(
        target_label=target_group.target_label,
        branch_name=target_group.branch_name,
        target_vs_rest_similarity=target_vs_rest,
        target_vs_other_labels=other_label_similarities,
        strongest_negative_control_label=strongest_label,
        strongest_negative_control_similarity=strongest_similarity,
        notes=notes,
    )
    target_group.cross_subject_agreement = CrossSubjectAgreement(
        subject_ids=list(target_group.subject_ids),
        event_ids=list(target_group.event_ids),
        n_subjects=len(target_group.subject_ids),
        n_events=len(target_group.event_ids),
        within_label_similarity=target_group.within_label_similarity,
        between_label_similarity=target_vs_rest,
        margin_vs_rest=margin,
        subject_consistency_score=target_group.subject_consistency_score,
        trial_consistency_score=target_group.trial_consistency_score,
    )

    target_mean = target_group.mean_bandpower
    rest_mean = None
    if other_groups:
        aligned_rest = [
            _align_mean_bandpower(other_group, target_group.phase_names, target_group.channel_names, target_group.available_bands)
            for other_group in other_groups
        ]
        rest_mean = np.nanmean(np.stack(aligned_rest, axis=0), axis=0)

    if rest_mean is not None and np.any(~np.isnan(rest_mean)):
        difference = target_mean - rest_mean
    else:
        difference = target_mean
    target_group.rest_mean_bandpower = None if rest_mean is None else rest_mean.astype(np.float32)
    target_group.difference_bandpower = difference.astype(np.float32)

    band_scores = np.nanmean(difference, axis=(0, 1))
    channel_scores = np.nanmean(difference, axis=(0, 2))
    phase_scores = np.nanmean(difference, axis=(1, 2))

    band_order = np.argsort(np.nan_to_num(band_scores, nan=-np.inf))[::-1]
    channel_order = np.argsort(np.nan_to_num(channel_scores, nan=-np.inf))[::-1]
    phase_order = np.argsort(np.nan_to_num(phase_scores, nan=-np.inf))[::-1]

    target_group.dominant_bands = [target_group.available_bands[index] for index in band_order[: min(3, len(band_order))]]
    target_group.dominant_channels = [target_group.channel_names[index] for index in channel_order[: min(6, len(channel_order))]]
    target_group.strongest_phases = [target_group.phase_names[index] for index in phase_order[: min(3, len(phase_order))]]


def _unit_interval(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, (float(value) + 1.0) / 2.0))


def _candidate_signature_from_parts(
    target_label: str,
    dominant_bands: list[str],
    dominant_channels: list[str],
    strongest_phases: list[str],
) -> str:
    band_text = ", ".join(dominant_bands[:2]) if dominant_bands else "unspecified bands"
    channel_text = ", ".join(dominant_channels[:4]) if dominant_channels else "unspecified channels"
    phase_text = ", ".join(strongest_phases[:2]) if strongest_phases else "unspecified phases"
    return f"shared {target_label} structure centered on {band_text} across {channel_text} during {phase_text}"


def _write_candidate_artifacts(group: _ComputedGroup) -> None:
    group.candidate_dir.mkdir(parents=True, exist_ok=True)
    prototype_path = group.candidate_dir / "prototype_epoch.npz"
    subject_path = group.candidate_dir / "subject_prototypes.npz"
    exemplar_path = group.candidate_dir / "exemplar_epochs.npz"
    spectral_path = group.candidate_dir / "spectral_summary.npz"
    topography_path = group.candidate_dir / "topography_summary.npz"
    similarity_path = group.candidate_dir / "similarity_matrices.npz"

    _write_npz(
        prototype_path,
        {
            "channel_names": np.asarray(group.channel_names),
            "phase_names": np.asarray(group.phase_names),
            "sampling_rate_hz": np.asarray([group.sampling_rate_hz], dtype=np.float32),
            "subject_ids": np.asarray(group.subject_ids),
            **group.prototype_arrays,
        },
    )
    _write_npz(
        subject_path,
        {
            "channel_names": np.asarray(group.channel_names),
            "phase_names": np.asarray(group.phase_names),
            "subject_ids": np.asarray(group.subject_ids),
            "sampling_rate_hz": np.asarray([group.sampling_rate_hz], dtype=np.float32),
            **{f"{phase_name}_stack": stack for phase_name, stack in group.subject_stacks.items()},
        },
    )
    _write_npz(
        exemplar_path,
        {
            "channel_names": np.asarray(group.channel_names),
            "phase_names": np.asarray(group.phase_names),
            "subject_ids": np.asarray(group.exemplar_subject_ids),
            "event_ids": np.asarray(group.exemplar_event_ids),
            "sampling_rate_hz": np.asarray([group.sampling_rate_hz], dtype=np.float32),
            **{f"{phase_name}_stack": stack for phase_name, stack in group.exemplar_stacks.items()},
        },
    )
    _write_npz(
        spectral_path,
        {
            "band_names": np.asarray(group.available_bands),
            "phase_names": np.asarray(group.phase_names),
            "channel_names": np.asarray(group.channel_names),
            "subject_bandpower": group.subject_bandpower.astype(np.float32),
            "event_bandpower": group.event_bandpower.astype(np.float32),
            "mean_bandpower": group.mean_bandpower.astype(np.float32),
        },
    )
    _write_npz(
        topography_path,
        {
            "band_names": np.asarray(group.available_bands),
            "phase_names": np.asarray(group.phase_names),
            "channel_names": np.asarray(group.channel_names),
            "mean_bandpower": group.mean_bandpower.astype(np.float32),
            "difference_bandpower": (
                group.difference_bandpower.astype(np.float32)
                if group.difference_bandpower is not None
                else np.zeros_like(group.mean_bandpower, dtype=np.float32)
            ),
            "rest_mean_bandpower": (
                group.rest_mean_bandpower.astype(np.float32)
                if group.rest_mean_bandpower is not None
                else np.zeros_like(group.mean_bandpower, dtype=np.float32)
            ),
            "channel_profile": np.nanmean(group.mean_bandpower, axis=2, dtype=np.float64).astype(np.float32),
            "band_profile": np.nanmean(group.mean_bandpower, axis=1, dtype=np.float64).astype(np.float32),
        },
    )
    _write_npz(
        similarity_path,
        {
            "subject_similarity_matrix": group.subject_similarity_matrix.astype(np.float32),
            "event_similarity_matrix": group.event_similarity_matrix.astype(np.float32),
            "subject_ids": np.asarray(group.subject_ids),
            "event_ids": np.asarray(group.event_ids),
        },
    )

    group.artifact_paths = {
        "prototype_epoch": str(prototype_path.relative_to(group.candidate_dir)),
        "subject_prototypes": str(subject_path.relative_to(group.candidate_dir)),
        "exemplar_epochs": str(exemplar_path.relative_to(group.candidate_dir)),
        "spectral_summary": str(spectral_path.relative_to(group.candidate_dir)),
        "topography_summary": str(topography_path.relative_to(group.candidate_dir)),
        "similarity_matrices": str(similarity_path.relative_to(group.candidate_dir)),
    }

def _build_candidate(group: _ComputedGroup, *, run_id: str, backbone_ids: list[str] | None = None) -> CandidatePattern:
    if group.cross_subject_agreement is None or group.control_summary is None:
        raise DiscoveryError("control summaries must be populated before building a candidate")
    summary_notes = list(group.summary_notes)
    if group.control_summary.strongest_negative_control_label:
        summary_notes.append(
            f"strongest negative control was {group.control_summary.strongest_negative_control_label}"
        )
    pattern_id = slugify(
        f"{group.label_namespace}_{group.event_family}_{group.target_label}_{group.branch_name}",
        "candidate_pattern",
    )
    return CandidatePattern(
        pattern_id=pattern_id,
        label_namespace=group.label_namespace,
        event_family=group.event_family,
        target_label=group.target_label,
        branch_name=group.branch_name,
        event_subtypes=list(group.event_subtypes),
        subject_ids=list(group.subject_ids),
        event_ids=list(group.event_ids),
        sampling_rate_hz=group.sampling_rate_hz,
        channel_names=list(group.channel_names),
        phase_names=list(group.phase_names),
        available_bands=list(group.available_bands),
        dominant_bands=list(group.dominant_bands),
        dominant_channels=list(group.dominant_channels),
        strongest_phases=list(group.strongest_phases),
        artifact_paths=dict(group.artifact_paths),
        candidate_root=str(group.candidate_dir),
        run_id=run_id,
        backbone_ids=sorted(set(backbone_ids or [])),
        cross_subject_agreement=group.cross_subject_agreement,
        control_summary=group.control_summary,
        summary_notes=summary_notes,
    )


def _write_candidate_metadata(group: _ComputedGroup) -> None:
    if group.candidate is None or group.control_summary is None:
        raise DiscoveryError("candidate metadata cannot be written before the candidate object exists")
    candidate_path = group.candidate_dir / "candidate.json"
    control_path = group.candidate_dir / "control_summary.json"
    branch_path = group.candidate_dir / "branch_agreement.json"
    reliability_path = group.candidate_dir / "reliability.json"
    report_path = group.candidate_dir / "report.md"
    _write_text(candidate_path, json.dumps(group.candidate.to_dict(), indent=2))
    _write_text(control_path, json.dumps(group.control_summary.to_dict(), indent=2))
    if group.candidate.branch_agreement is not None:
        _write_text(branch_path, json.dumps(group.candidate.branch_agreement.to_dict(), indent=2))
    if group.candidate.reliability is not None:
        _write_text(reliability_path, json.dumps(group.candidate.reliability.to_dict(), indent=2))
    lines = [
        f"# Candidate Pattern: {group.candidate.pattern_id}",
        "",
        f"- Target label: {group.target_label}",
        f"- Branch: {group.branch_name}",
        f"- Subjects: {len(group.subject_ids)}",
        f"- Events: {len(group.event_ids)}",
        f"- Dominant bands: {', '.join(group.dominant_bands) if group.dominant_bands else 'not determined'}",
        f"- Dominant channels: {', '.join(group.dominant_channels) if group.dominant_channels else 'not determined'}",
        f"- Strongest phases: {', '.join(group.strongest_phases) if group.strongest_phases else 'not determined'}",
    ]
    if group.cross_subject_agreement is not None:
        lines.extend(
            [
                f"- Within-label similarity: {group.cross_subject_agreement.within_label_similarity}",
                f"- Target-vs-rest similarity: {group.cross_subject_agreement.between_label_similarity}",
                f"- Margin vs rest: {group.cross_subject_agreement.margin_vs_rest}",
            ]
        )
    if group.candidate.branch_agreement is not None:
        lines.append(f"- Branch agreement: {group.candidate.branch_agreement.overall_status}")
    if group.candidate.reliability is not None:
        lines.append(f"- Confidence tier: {group.candidate.reliability.confidence_tier}")
    if group.candidate.backbone_ids:
        lines.append(f"- Backbone provenance: {', '.join(group.candidate.backbone_ids)}")
    lines.extend(["", "## Notes"])
    lines.extend(f"- {note}" for note in group.candidate.summary_notes or ["No additional notes"])
    _write_text(report_path, "\n".join(lines) + "\n")
    group.artifact_paths["candidate_json"] = str(candidate_path.relative_to(group.candidate_dir))
    group.artifact_paths["control_summary_json"] = str(control_path.relative_to(group.candidate_dir))
    if group.candidate.branch_agreement is not None:
        group.artifact_paths["branch_agreement_json"] = str(branch_path.relative_to(group.candidate_dir))
    if group.candidate.reliability is not None:
        group.artifact_paths["reliability_json"] = str(reliability_path.relative_to(group.candidate_dir))
    group.artifact_paths["candidate_report"] = str(report_path.relative_to(group.candidate_dir))
    group.candidate.artifact_paths = dict(group.artifact_paths)
    _write_text(candidate_path, json.dumps(group.candidate.to_dict(), indent=2))


def _reference_group(groups: list[_ComputedGroup]) -> _ComputedGroup:
    for group in groups:
        if group.branch_name == "raw_preserving":
            return group
    for group in groups:
        if group.branch_name == "source_epoch":
            return group
    def margin_value(item: _ComputedGroup) -> float:
        if item.cross_subject_agreement is None or item.cross_subject_agreement.margin_vs_rest is None:
            return float("-inf")
        return float(item.cross_subject_agreement.margin_vs_rest)
    return max(groups, key=margin_value)


def _classify_branch_status(reference: _ComputedGroup, candidate: _ComputedGroup, similarity: float | None) -> str:
    if candidate is reference:
        return "preserved"
    if similarity is None:
        return "branch-sensitive"
    reference_margin = reference.cross_subject_agreement.margin_vs_rest if reference.cross_subject_agreement else None
    candidate_margin = candidate.cross_subject_agreement.margin_vs_rest if candidate.cross_subject_agreement else None
    if reference_margin in (None, 0):
        margin_ratio = 1.0
    elif candidate_margin is None:
        margin_ratio = 0.0
    else:
        margin_ratio = candidate_margin / reference_margin if reference_margin != 0 else 1.0
    if similarity >= 0.8 and margin_ratio >= 0.8:
        return "preserved"
    if similarity >= 0.55 and margin_ratio >= 0.45:
        return "weakened"
    if similarity >= 0.35 and margin_ratio > 0.2:
        return "shifted"
    return "branch-sensitive"


def _compute_branch_agreements(groups: list[_ComputedGroup]) -> list[BranchAgreement]:
    grouped: dict[tuple[str, str, str], list[_ComputedGroup]] = {}
    for group in groups:
        grouped.setdefault((group.label_namespace, group.event_family, group.target_label), []).append(group)

    agreements: list[BranchAgreement] = []
    for (label_namespace, event_family, target_label), branch_groups in grouped.items():
        if len(branch_groups) < 2:
            continue
        reference = _reference_group(branch_groups)
        branch_status: dict[str, str] = {}
        pairwise_similarity: dict[str, float] = {}
        margin_by_branch: dict[str, float | None] = {}
        notes: list[str] = [f"reference branch: {reference.branch_name}"]
        for group in branch_groups:
            similarity = _group_profile_similarity(reference, group)
            status = _classify_branch_status(reference, group, similarity)
            branch_status[group.branch_name] = status
            if similarity is not None:
                pairwise_similarity[f"{reference.branch_name}__{group.branch_name}"] = similarity
            margin_by_branch[group.branch_name] = (
                None if group.cross_subject_agreement is None else group.cross_subject_agreement.margin_vs_rest
            )
        if any(status == "branch-sensitive" for status in branch_status.values()):
            overall_status = "branch-sensitive"
        elif any(status == "shifted" for status in branch_status.values()):
            overall_status = "shifted"
        elif any(status == "weakened" for status in branch_status.values()):
            overall_status = "weakened"
        else:
            overall_status = "preserved"
        agreement = BranchAgreement(
            target_label=target_label,
            label_namespace=label_namespace,
            event_family=event_family,
            reference_branch=reference.branch_name,
            overall_status=overall_status,
            branch_status=branch_status,
            pairwise_similarity=pairwise_similarity,
            margin_by_branch=margin_by_branch,
            notes=notes,
        )
        agreements.append(agreement)
        for group in branch_groups:
            if group.candidate is not None:
                group.candidate.branch_agreement = agreement
    return agreements


def _attach_reliability(
    group: _ComputedGroup,
    *,
    context_groups: list[_ComputedGroup],
    rng_seed: int,
    null_iterations: int,
    subsample_iterations: int,
) -> None:
    if group.candidate is None or group.cross_subject_agreement is None:
        return
    other_groups = [other for other in context_groups if other.target_label != group.target_label]
    group.candidate.reliability = assess_candidate_reliability(
        target_label=group.target_label,
        n_subjects=len(group.subject_ids),
        n_events=len(group.event_ids),
        margin_vs_rest=group.cross_subject_agreement.margin_vs_rest,
        target_features=group.subject_features,
        subject_ids=list(group.subject_ids),
        session_ids=list(group.session_ids),
        cohort_ids=list(group.cohort_labels),
        other_groups=[
            ReliabilityContextGroup(
                target_label=other.target_label,
                subject_ids=list(other.subject_ids),
                subject_features=other.subject_features,
                session_ids=list(other.session_ids),
                cohort_ids=list(other.cohort_labels),
                margin_vs_rest=(None if other.cross_subject_agreement is None else other.cross_subject_agreement.margin_vs_rest),
            )
            for other in other_groups
        ],
        branch_agreement=group.candidate.branch_agreement,
        rng_seed=rng_seed,
        null_iterations=null_iterations,
        subsample_iterations=subsample_iterations,
    )


def _artifact_input(candidate: CandidatePattern, artifact_key: str, *, role: str, representation: str, description: str) -> ArtifactInput:
    source_path = Path(candidate.candidate_root) / candidate.artifact_paths[artifact_key]
    return ArtifactInput(
        artifact_id=artifact_key,
        role=role,
        representation=representation,
        format=source_path.suffix.lstrip("."),
        source_path=source_path,
        description=description,
    )


def _predicted_package_dir(candidate: CandidatePattern, package_root: str | Path) -> Path:
    event_subtype = "mixed"
    if len(candidate.event_subtypes) == 1:
        event_subtype = candidate.event_subtypes[0]
    partition = PartitionSpec(
        study_ids=list(candidate.subject_ids),
        event_family=candidate.event_family,
        target_label=candidate.target_label,
        event_subtype=event_subtype,
        label_namespace=candidate.label_namespace,
        stimulus_modality=candidate.branch_name,
    )
    return Path(package_root) / "patterns" / Path(*partition.to_path_parts()) / slugify(candidate.pattern_id)


def _run_bundle_artifacts(run_summary: DiscoveryRunSummary) -> list[ArtifactInput]:
    run_bundle_specs = [
        ("run_manifest", run_summary.run_manifest_path, "run_manifest", "support", "Discovery run manifest"),
        ("config_snapshot", run_summary.config_snapshot_path, "config_snapshot", "support", "Discovery config snapshot"),
        ("environment", run_summary.environment_path, "environment", "support", "Runtime environment snapshot"),
        ("artifact_lineage", run_summary.artifact_lineage_path, "artifact_lineage", "support", "Artifact lineage snapshot"),
        ("warnings", run_summary.warnings_path, "warnings", "support", "Structured warning index"),
        ("run_log", run_summary.log_path, "run_log", "support", "Structured discovery log"),
    ]
    artifacts: list[ArtifactInput] = []
    for artifact_id, source_path, role, representation, description in run_bundle_specs:
        if not source_path:
            continue
        source = Path(source_path)
        if not source.exists():
            continue
        artifacts.append(
            ArtifactInput(
                artifact_id=artifact_id,
                role=role,
                representation=representation,
                format=source.suffix.lstrip("."),
                source_path=source,
                description=description,
            )
        )
    return artifacts


def package_discovery_run(
    summary: DiscoveryRunSummary | str | Path,
    *,
    package_root: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    run_summary = load_discovery_run_summary(summary) if isinstance(summary, (str, Path)) else summary
    builder = PatternPackageBuilder(Path(package_root))
    packaged_paths: list[Path] = []
    for candidate in run_summary.candidates:
        try:
            event_subtype = "mixed"
            if len(candidate.event_subtypes) == 1:
                event_subtype = candidate.event_subtypes[0]
            branch_status = candidate.branch_agreement.overall_status if candidate.branch_agreement else "single_branch_only"
            partition = PartitionSpec(
                study_ids=list(candidate.subject_ids),
                event_family=candidate.event_family,
                target_label=candidate.target_label,
                event_subtype=event_subtype,
                label_namespace=candidate.label_namespace,
                stimulus_modality=candidate.branch_name,
            )
            analysis_notes = [f"branch={candidate.branch_name}", f"branch_agreement={branch_status}", f"run_id={run_summary.run_id}"]
            if candidate.reliability is not None:
                analysis_notes.append(f"confidence_tier={candidate.reliability.confidence_tier}")
            analysis = AnalysisSpec(
                discovery_mode="cross_subject_discovery",
                source_models=["PathfinderInterpretableDiscoveryV1", *candidate.backbone_ids],
                notes="; ".join(analysis_notes),
            )
            artifact_risk = "branch-sensitive" if branch_status == "branch-sensitive" else ""
            if candidate.reliability is not None and candidate.reliability.status_flags:
                artifact_risk = ",".join(candidate.reliability.status_flags)
            summary_model = PatternSummary(
                candidate_signature=_candidate_signature_from_parts(
                    candidate.target_label,
                    candidate.dominant_bands,
                    candidate.dominant_channels,
                    candidate.strongest_phases,
                ),
                bands=list(candidate.dominant_bands),
                channels=list(candidate.dominant_channels),
                temporal_notes=(
                    f"strongest phases: {', '.join(candidate.strongest_phases) if candidate.strongest_phases else 'unspecified'}; "
                    f"branch agreement: {branch_status}; "
                    f"confidence: {candidate.reliability.confidence_tier if candidate.reliability else 'unscored'}"
                ),
                reproducibility_score=_unit_interval(candidate.cross_subject_agreement.margin_vs_rest),
                cross_subject_support=_unit_interval(candidate.cross_subject_agreement.within_label_similarity),
                artifact_risk=artifact_risk,
            )
            artifacts = [
                _artifact_input(candidate, "prototype_epoch", role="prototype_epoch", representation="processed_epoch", description="Cross-subject prototype epoch artifact"),
                _artifact_input(candidate, "exemplar_epochs", role="exemplar_epochs", representation="processed_epoch", description="Representative aligned epochs supporting the candidate"),
                _artifact_input(candidate, "subject_prototypes", role="subject_prototypes", representation="processed_epoch", description="Per-subject prototype epochs for the candidate"),
                _artifact_input(candidate, "spectral_summary", role="spectral_summary", representation="time_frequency", description="Bandpower tensor backing the candidate summary"),
                _artifact_input(candidate, "topography_summary", role="topography_summary", representation="topography", description="Channel-phase-band topography derivative for candidate inspection"),
                _artifact_input(candidate, "similarity_matrices", role="similarity_matrices", representation="support", description="Cross-subject and cross-event similarity matrices"),
                _artifact_input(candidate, "candidate_json", role="candidate_metadata", representation="support", description="Candidate metadata index"),
                _artifact_input(candidate, "control_summary_json", role="control_summary", representation="support", description="Negative-control comparison summary"),
                _artifact_input(candidate, "candidate_report", role="candidate_report", representation="report", description="Readable candidate summary"),
            ]
            if "branch_agreement_json" in candidate.artifact_paths:
                artifacts.append(
                    _artifact_input(candidate, "branch_agreement_json", role="branch_agreement", representation="support", description="Branch agreement summary")
                )
            if "reliability_json" in candidate.artifact_paths:
                artifacts.append(
                    _artifact_input(candidate, "reliability_json", role="reliability", representation="support", description="Reliability assessment")
                )
            if "backbone_consensus_json" in candidate.artifact_paths:
                artifacts.append(
                    _artifact_input(candidate, "backbone_consensus_json", role="backbone_consensus", representation="support", description="Cross-backbone consensus summary")
                )
            for evidence in candidate.backbone_evidence:
                if "embeddings_npz" in evidence.artifact_paths:
                    artifacts.append(
                        _artifact_input(
                            candidate,
                            f"backbone_{evidence.model_id}_embeddings",
                            role=f"backbone_{evidence.model_id}_embeddings",
                            representation="embedding",
                            description=f"{evidence.model_id} embedding support for the candidate",
                        )
                    )
                if "evidence_json" in evidence.artifact_paths:
                    artifacts.append(
                        _artifact_input(
                            candidate,
                            f"backbone_{evidence.model_id}_evidence_json",
                            role=f"backbone_{evidence.model_id}_evidence",
                            representation="support",
                            description=f"{evidence.model_id} evidence summary for the candidate",
                        )
                    )
            artifacts.extend(_run_bundle_artifacts(run_summary))
            package_dir, _ = builder.create(
                pattern_id=candidate.pattern_id,
                partition=partition,
                analysis=analysis,
                summary=summary_model,
                artifacts=artifacts,
                overwrite=overwrite,
            )
            candidate.packaged_pattern_path = str(package_dir)
            packaged_paths.append(package_dir)
        except Exception as exc:
            run_summary.failures.append(
                RunIssueRecord(
                    stage="packaging",
                    severity="error",
                    message=str(exc),
                    target_label=candidate.target_label,
                    branch_name=candidate.branch_name,
                    path=candidate.candidate_root,
                )
            )
    run_summary.packaged_pattern_paths = [str(path) for path in packaged_paths]
    for candidate in run_summary.candidates:
        candidate_path = Path(candidate.candidate_root) / candidate.artifact_paths["candidate_json"]
        candidate_path.write_text(json.dumps(candidate.to_dict(), indent=2), encoding="utf-8")
    save_discovery_run_summary(run_summary, Path(run_summary.output_root) / "run_summary.json")
    return packaged_paths

def discover_shared_patterns(
    collection_paths: list[str | Path],
    *,
    output_root: str | Path,
    run_id: str = "",
    target_labels: list[str] | None = None,
    branches: list[str] | None = None,
    min_subjects: int = 2,
    max_exemplars: int = 6,
    package_root: str | Path | None = None,
    backbone_ids: list[str] | None = None,
    rng_seed: int = 0,
    null_iterations: int = 64,
    subsample_iterations: int = 32,
    overwrite: bool = False,
) -> tuple[Path, DiscoveryRunSummary]:
    resolved_collection_paths = [Path(path).resolve() for path in collection_paths]
    if not resolved_collection_paths:
        raise DiscoveryError("At least one collection path is required for discovery")

    run_name = make_run_id("discovery", run_id)
    run_dir = Path(output_root) / "discovery_runs" / run_name
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"discovery run already exists: {run_dir}")
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = StructuredRunLogger(run_dir / "run.log.jsonl")
    logger.log(level="info", stage="discovery", message="starting discovery run", run_id=run_name)

    normalized_labels = {label.strip() for label in target_labels or [] if label.strip()}
    normalized_branches = {branch.strip() for branch in branches or [] if branch.strip()}
    normalized_backbones = sorted({backbone.strip() for backbone in backbone_ids or [] if backbone.strip()})

    observations: list[_EpochObservation] = []
    notes: list[str] = []
    failures: list[RunIssueRecord] = []

    study_report = validate_study(resolved_collection_paths, min_subjects=min_subjects, min_trials_per_label=2)
    for item in study_report.errors:
        failures.append(RunIssueRecord(stage="validation", severity="error", message=item.message, path=item.path or item.location))
        logger.log(level="error", stage="validation", message=item.message, code=item.code, path=item.path or item.location)
    for item in study_report.warnings:
        notes.append(item.message)
        logger.log(level="warning", stage="validation", message=item.message, code=item.code, path=item.path or item.location)

    for collection_path in resolved_collection_paths:
        try:
            collection = load_epoch_collection(collection_path)
        except Exception as exc:
            failures.append(RunIssueRecord(stage="loading", severity="error", message=str(exc), path=str(collection_path)))
            logger.log(level="error", stage="loading", message=str(exc), path=str(collection_path))
            continue
        branch_name = _infer_branch_name(collection)
        if normalized_branches and branch_name not in normalized_branches:
            continue
        collection_dir = collection_path.parent
        subject_id = collection.recording.subject_id or collection.recording.recording_id
        for artifact in collection.artifacts:
            if normalized_labels and artifact.event.target_label not in normalized_labels:
                continue
            arrays = _load_artifact_arrays(collection_dir, artifact.signal_path)
            if not arrays:
                failures.append(
                    RunIssueRecord(
                        stage="loading",
                        severity="warning",
                        message="artifact contained no common event-phase arrays",
                        target_label=artifact.event.target_label,
                        branch_name=branch_name,
                        subject_id=subject_id,
                        path=str(collection_dir / artifact.signal_path),
                    )
                )
                continue
            observations.append(
                _EpochObservation(
                    collection_path=collection_path,
                    collection=collection,
                    branch_name=branch_name,
                    subject_id=subject_id,
                    session_id=collection.recording.session_id,
                    cohort_label=str(collection.recording.metadata.get("cohort_label", "") or collection.recording.metadata.get("cohort", "")),
                    channel_names=list(collection.channel_names),
                    event_id=artifact.event.event_id,
                    event_family=artifact.event.event_family,
                    target_label=artifact.event.target_label,
                    event_subtype=artifact.event.event_subtype,
                    label_namespace=artifact.event.label_namespace or collection.recording.label_namespace,
                    arrays=arrays,
                )
            )

    if not observations:
        raise DiscoveryError("No observations matched the discovery filters")

    grouped_observations: dict[tuple[str, str, str, str], list[_EpochObservation]] = {}
    for observation in observations:
        grouped_observations.setdefault(_group_key(observation), []).append(observation)

    computed_groups: list[_ComputedGroup] = []
    for (_, _, target_label, branch_name), group_observations in grouped_observations.items():
        subject_ids = sorted({observation.subject_id for observation in group_observations})
        if len(subject_ids) < min_subjects:
            message = f"skipped {target_label} on branch {branch_name} because only {len(subject_ids)} subject(s) were available"
            notes.append(message)
            failures.append(RunIssueRecord(stage="grouping", severity="warning", message=message, target_label=target_label, branch_name=branch_name))
            continue
        sampling_rates = {float(observation.collection.sampling_rate_hz) for observation in group_observations}
        if len(sampling_rates) > 1:
            message = f"skipped {target_label} on branch {branch_name} because sampling rates were inconsistent: {sorted(sampling_rates)}"
            failures.append(RunIssueRecord(stage="grouping", severity="error", message=message, target_label=target_label, branch_name=branch_name))
            logger.log(level="error", stage="grouping", message=message, target_label=target_label, branch_name=branch_name)
            continue
        candidate_dir = (
            run_dir
            / "candidates"
            / slugify(group_observations[0].label_namespace, "default_namespace")
            / slugify(group_observations[0].event_family, "event_family")
            / slugify(group_observations[0].target_label, "target_label")
            / slugify(group_observations[0].branch_name, "branch")
        )
        try:
            computed_group = _compute_group(
                group_observations,
                candidate_dir=candidate_dir,
                max_exemplars=max_exemplars,
            )
            computed_groups.append(computed_group)
        except Exception as exc:
            failures.append(RunIssueRecord(stage="grouping", severity="error", message=str(exc), target_label=target_label, branch_name=branch_name))
            logger.log(level="error", stage="grouping", message=str(exc), target_label=target_label, branch_name=branch_name)

    if not computed_groups:
        raise DiscoveryError("No candidate groups met the minimum subject requirement")

    context_groups: dict[tuple[str, str, str], list[_ComputedGroup]] = {}
    for group in computed_groups:
        context_groups.setdefault((group.label_namespace, group.event_family, group.branch_name), []).append(group)

    for groups_in_context in context_groups.values():
        for group in groups_in_context:
            try:
                other_groups = [other for other in groups_in_context if other.target_label != group.target_label]
                _populate_controls(group, other_groups)
                _write_candidate_artifacts(group)
                group.candidate = _build_candidate(group, run_id=run_name, backbone_ids=normalized_backbones)
            except Exception as exc:
                failures.append(RunIssueRecord(stage="candidate_build", severity="error", message=str(exc), target_label=group.target_label, branch_name=group.branch_name, path=str(group.candidate_dir)))
                logger.log(level="error", stage="candidate_build", message=str(exc), target_label=group.target_label, branch_name=group.branch_name)

    active_groups = [group for group in computed_groups if group.candidate is not None]
    if not active_groups:
        raise DiscoveryError("No candidate groups survived discovery processing")

    branch_agreements = _compute_branch_agreements(active_groups)
    for context in context_groups.values():
        for group in context:
            if group.candidate is None:
                continue
            _attach_reliability(
                group,
                context_groups=context,
                rng_seed=rng_seed,
                null_iterations=null_iterations,
                subsample_iterations=subsample_iterations,
            )
            _write_candidate_metadata(group)

    candidates = [group.candidate for group in active_groups if group.candidate is not None]
    summary = DiscoveryRunSummary(
        run_id=run_name,
        output_root=str(run_dir),
        collection_paths=[str(path) for path in resolved_collection_paths],
        branch_names=sorted({group.branch_name for group in active_groups}),
        candidates=candidates,
        status="partial" if any(item.severity == "error" for item in failures) else "success",
        rng_seed=rng_seed,
        backbone_ids=normalized_backbones,
        branch_agreements=branch_agreements,
        failures=failures,
        notes=notes,
    )
    summary_path = save_discovery_run_summary(summary, run_dir / "run_summary.json")

    generated_artifacts = [
        RunArtifactRecord(
            artifact_type="candidate_pattern",
            path=str(Path(candidate.candidate_root) / candidate.artifact_paths["prototype_epoch"]),
            role=f"{candidate.target_label}:{candidate.branch_name}:prototype",
            format="npz",
            parent_paths=[str(path) for path in resolved_collection_paths],
        )
        for candidate in candidates
    ]
    for candidate in candidates:
        if "topography_summary" in candidate.artifact_paths:
            generated_artifacts.append(
                RunArtifactRecord(
                    artifact_type="candidate_derivative",
                    path=str(Path(candidate.candidate_root) / candidate.artifact_paths["topography_summary"]),
                    role=f"{candidate.target_label}:{candidate.branch_name}:topography",
                    format="npz",
                    parent_paths=[str(path) for path in resolved_collection_paths],
                )
            )
    generated_artifacts.append(
        RunArtifactRecord(
            artifact_type="discovery_run",
            path=str(summary_path),
            role="run_summary",
            format="json",
            parent_paths=[str(path) for path in resolved_collection_paths],
        )
    )
    predicted_package_paths: list[str] = []
    if package_root is not None:
        predicted_package_paths = [str(_predicted_package_dir(candidate, package_root)) for candidate in candidates]
    for package_path in predicted_package_paths:
        generated_artifacts.append(
            RunArtifactRecord(
                artifact_type="pattern_package",
                path=package_path,
                role="packaged_candidate",
                format="dir",
                parent_paths=[str(path) for path in resolved_collection_paths],
            )
        )
    run_paths = write_run_bundle(
        run_root=run_dir,
        run_id=run_name,
        operation="discovery",
        command="",
        config_snapshot={
            "target_labels": sorted(normalized_labels),
            "branches": sorted(normalized_branches),
            "min_subjects": min_subjects,
            "max_exemplars": max_exemplars,
            "rng_seed": rng_seed,
            "null_iterations": null_iterations,
            "subsample_iterations": subsample_iterations,
            "package_root": str(Path(package_root).resolve()) if package_root is not None else "",
            "backbone_ids": normalized_backbones,
        },
        source_artifacts=[RunArtifactRecord(artifact_type="epoch_collection", path=str(path), role="input_collection", format="json") for path in resolved_collection_paths],
        generated_artifacts=generated_artifacts,
        code_root=Path(__file__).resolve().parents[2],
        rng_seed=rng_seed,
        active_branches=summary.branch_names,
        warnings=notes,
        status=summary.status,
        optional_dependencies=["numpy"],
    )
    summary.run_manifest_path = run_paths["run_manifest_path"]
    summary.config_snapshot_path = run_paths["config_snapshot_path"]
    summary.environment_path = run_paths["environment_path"]
    summary.artifact_lineage_path = run_paths["artifact_lineage_path"]
    summary.warnings_path = run_paths["warnings_path"]
    summary.log_path = run_paths["log_path"]
    if package_root is not None:
        package_discovery_run(summary, package_root=package_root, overwrite=overwrite)
    summary_path = save_discovery_run_summary(summary, run_dir / "run_summary.json")
    logger.log(level="info", stage="discovery", message="discovery completed", candidate_count=len(candidates), status=summary.status)
    return summary_path, summary









