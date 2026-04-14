from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .discovery_models import (
    BackboneConsensusSummary,
    BackboneEvidenceSummary,
    BranchAgreement,
    GroupedHoldoutSummary,
    LeaveOneSubjectOutSummary,
    NullTestSummary,
    ReliabilityAssessment,
    SubsampleStabilitySummary,
)


@dataclass(slots=True)
class ReliabilityContextGroup:
    target_label: str
    subject_ids: list[str]
    subject_features: np.ndarray
    session_ids: list[str] | None = None
    cohort_ids: list[str] | None = None
    margin_vs_rest: float | None = None


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
    for left_index in range(vectors.shape[0]):
        for right_index in range(left_index + 1, vectors.shape[0]):
            scores.append(_safe_corr(vectors[left_index], vectors[right_index]))
    return float(np.mean(scores)) if scores else None


def _mean_cross_similarity(left_vectors: np.ndarray, right_vectors: np.ndarray) -> float | None:
    if left_vectors.size == 0 or right_vectors.size == 0:
        return None
    scores = [_safe_corr(left, right) for left in left_vectors for right in right_vectors]
    return float(np.mean(scores)) if scores else None


def compute_leave_one_subject_out(
    *,
    subject_ids: list[str],
    target_features: np.ndarray,
    other_groups: list[ReliabilityContextGroup],
) -> LeaveOneSubjectOutSummary:
    if target_features.shape[0] < 2:
        return LeaveOneSubjectOutSummary(
            n_folds=int(target_features.shape[0]),
            notes=["leave-one-subject-out requires at least two subjects"],
        )
    margins: list[float] = []
    failures: list[str] = []
    for index, subject_id in enumerate(subject_ids):
        keep_mask = np.ones(target_features.shape[0], dtype=bool)
        keep_mask[index] = False
        centroid = np.mean(target_features[keep_mask], axis=0, dtype=np.float64)
        holdout = target_features[index]
        target_similarity = _safe_corr(holdout, centroid)
        negative_scores: list[float] = []
        for other in other_groups:
            if other.subject_features.size == 0:
                continue
            other_centroid = np.mean(other.subject_features, axis=0, dtype=np.float64)
            negative_scores.append(_safe_corr(holdout, other_centroid))
        best_negative = max(negative_scores) if negative_scores else 0.0
        margin = float(target_similarity - best_negative)
        margins.append(margin)
        if margin <= 0.0:
            failures.append(subject_id)
    success_rate = 1.0 - (len(failures) / len(subject_ids)) if subject_ids else None
    return LeaveOneSubjectOutSummary(
        n_folds=len(subject_ids),
        success_rate=success_rate,
        mean_margin=float(np.mean(margins)) if margins else None,
        min_margin=float(np.min(margins)) if margins else None,
        held_out_subject_ids=list(subject_ids),
        failed_subject_ids=failures,
    )


def compute_label_shuffle_null(
    *,
    target_label: str,
    target_features: np.ndarray,
    other_groups: list[ReliabilityContextGroup],
    actual_margin: float | None,
    rng_seed: int,
    iterations: int,
) -> NullTestSummary:
    if actual_margin is None:
        return NullTestSummary(iterations=iterations, notes=["actual margin was unavailable"])
    label_to_features = {target_label: np.asarray(target_features, dtype=np.float32)}
    for group in other_groups:
        label_to_features[group.target_label] = np.asarray(group.subject_features, dtype=np.float32)
    unique_labels = [label for label, features in label_to_features.items() if features.shape[0] > 0]
    if len(unique_labels) < 2:
        return NullTestSummary(iterations=iterations, actual_margin=actual_margin, notes=["null test requires at least two labels"])

    pooled_features = np.concatenate([label_to_features[label] for label in unique_labels], axis=0)
    labels: list[str] = []
    for label in unique_labels:
        labels.extend([label] * label_to_features[label].shape[0])
    label_array = np.asarray(labels)
    rng = np.random.default_rng(rng_seed)
    null_margins: list[float] = []
    for _ in range(iterations):
        shuffled = np.asarray(label_array)
        rng.shuffle(shuffled)
        target_mask = shuffled == target_label
        rest_mask = ~target_mask
        if int(np.sum(target_mask)) < 2 or int(np.sum(rest_mask)) < 1:
            continue
        target_like = pooled_features[target_mask]
        rest_like = pooled_features[rest_mask]
        within = _mean_pairwise_similarity(target_like)
        between = _mean_cross_similarity(target_like, rest_like)
        if within is None or between is None:
            continue
        null_margins.append(float(within - between))
    if not null_margins:
        return NullTestSummary(iterations=iterations, actual_margin=actual_margin, notes=["no valid null-test permutations were produced"])
    null_array = np.asarray(null_margins, dtype=np.float64)
    empirical_p_value = float((1 + int(np.sum(null_array >= actual_margin))) / (len(null_array) + 1))
    return NullTestSummary(
        iterations=iterations,
        actual_margin=actual_margin,
        null_mean_margin=float(np.mean(null_array)),
        null_std_margin=float(np.std(null_array)),
        null_p95_margin=float(np.quantile(null_array, 0.95)),
        empirical_p_value=empirical_p_value,
    )


def compute_grouped_holdout(
    *,
    grouping_name: str,
    subject_ids: list[str],
    group_ids: list[str] | None,
    target_features: np.ndarray,
    other_groups: list[ReliabilityContextGroup],
) -> GroupedHoldoutSummary:
    if group_ids is None or len(group_ids) != len(subject_ids):
        return GroupedHoldoutSummary(
            grouping_name=grouping_name,
            n_groups=0,
            notes=[f"{grouping_name} holdout metadata was unavailable"],
        )
    normalized_groups = [group_id.strip() for group_id in group_ids if str(group_id).strip()]
    unique_groups = sorted(set(normalized_groups))
    if len(unique_groups) < 2:
        return GroupedHoldoutSummary(
            grouping_name=grouping_name,
            n_groups=len(unique_groups),
            notes=[f"{grouping_name} holdout requires at least two distinct groups"],
        )
    margins: list[float] = []
    failures: list[str] = []
    held_out_group_ids: list[str] = []
    group_array = np.asarray([group_id.strip() for group_id in group_ids], dtype=object)
    for group_id in unique_groups:
        holdout_mask = group_array == group_id
        keep_mask = ~holdout_mask
        if int(np.sum(holdout_mask)) < 1 or int(np.sum(keep_mask)) < 1:
            continue
        centroid = np.mean(target_features[keep_mask], axis=0, dtype=np.float64)
        held_out_vectors = target_features[holdout_mask]
        holdout_scores = [_safe_corr(vector, centroid) for vector in held_out_vectors]
        target_similarity = float(np.mean(holdout_scores)) if holdout_scores else 0.0
        negative_scores: list[float] = []
        for other in other_groups:
            if other.subject_features.size == 0:
                continue
            other_centroid = np.mean(other.subject_features, axis=0, dtype=np.float64)
            negative_scores.extend(_safe_corr(vector, other_centroid) for vector in held_out_vectors)
        best_negative = max(negative_scores) if negative_scores else 0.0
        margin = float(target_similarity - best_negative)
        margins.append(margin)
        held_out_group_ids.append(group_id)
        if margin <= 0.0:
            failures.append(group_id)
    success_rate = 1.0 - (len(failures) / len(held_out_group_ids)) if held_out_group_ids else None
    return GroupedHoldoutSummary(
        grouping_name=grouping_name,
        n_groups=len(unique_groups),
        success_rate=success_rate,
        mean_margin=float(np.mean(margins)) if margins else None,
        min_margin=float(np.min(margins)) if margins else None,
        held_out_group_ids=held_out_group_ids,
        failed_group_ids=failures,
    )


def compute_subsample_stability(
    *,
    subject_features: np.ndarray,
    rng_seed: int,
    iterations: int,
    subsample_fraction: float = 0.75,
) -> SubsampleStabilitySummary:
    if subject_features.shape[0] < 3:
        return SubsampleStabilitySummary(
            iterations=iterations,
            subsample_fraction=subsample_fraction,
            notes=["subsample stability requires at least three subjects"],
        )
    sample_size = max(2, int(math.ceil(subject_features.shape[0] * subsample_fraction)))
    full_centroid = np.mean(subject_features, axis=0, dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    scores: list[float] = []
    for _ in range(iterations):
        indices = rng.choice(subject_features.shape[0], size=sample_size, replace=False)
        centroid = np.mean(subject_features[indices], axis=0, dtype=np.float64)
        scores.append(_safe_corr(full_centroid, centroid))
    return SubsampleStabilitySummary(
        iterations=iterations,
        subsample_fraction=subsample_fraction,
        mean_similarity=float(np.mean(scores)) if scores else None,
        min_similarity=float(np.min(scores)) if scores else None,
    )


def _branch_score(branch_agreement: BranchAgreement | None) -> float:
    if branch_agreement is None:
        return 0.5
    mapping = {
        "preserved": 1.0,
        "weakened": 0.7,
        "shifted": 0.4,
        "branch-sensitive": 0.1,
    }
    return mapping.get(branch_agreement.overall_status, 0.4)


def _backbone_score(backbone_consensus: BackboneConsensusSummary | None) -> float | None:
    if backbone_consensus is None:
        return None
    mapping = {
        "strong_agreement": 1.0,
        "partial_agreement": 0.72,
        "weak_agreement": 0.38,
        "insufficient": 0.18,
    }
    return mapping.get(backbone_consensus.overall_status, 0.38)


def assess_candidate_reliability(
    *,
    target_label: str,
    n_subjects: int,
    n_events: int,
    margin_vs_rest: float | None,
    target_features: np.ndarray,
    subject_ids: list[str],
    other_groups: list[ReliabilityContextGroup],
    session_ids: list[str] | None,
    cohort_ids: list[str] | None,
    branch_agreement: BranchAgreement | None,
    rng_seed: int,
    null_iterations: int,
    subsample_iterations: int,
) -> ReliabilityAssessment:
    loso = compute_leave_one_subject_out(
        subject_ids=subject_ids,
        target_features=target_features,
        other_groups=other_groups,
    )
    session_holdout = compute_grouped_holdout(
        grouping_name="session",
        subject_ids=subject_ids,
        group_ids=session_ids,
        target_features=target_features,
        other_groups=other_groups,
    )
    cohort_holdout = compute_grouped_holdout(
        grouping_name="cohort",
        subject_ids=subject_ids,
        group_ids=cohort_ids,
        target_features=target_features,
        other_groups=other_groups,
    )
    null_test = compute_label_shuffle_null(
        target_label=target_label,
        target_features=target_features,
        other_groups=other_groups,
        actual_margin=margin_vs_rest,
        rng_seed=rng_seed,
        iterations=null_iterations,
    )
    subsample = compute_subsample_stability(
        subject_features=target_features,
        rng_seed=rng_seed + 1009,
        iterations=subsample_iterations,
    )

    flags: list[str] = []
    notes: list[str] = []
    if n_subjects < 3 or n_events < 4:
        flags.append("low-support")
        notes.append("candidate support is limited by subject or trial count")
    control_margin_ok = margin_vs_rest is not None and margin_vs_rest > 0.05
    if not control_margin_ok:
        flags.append("control-weak")
    if loso.success_rate is not None and loso.success_rate < 0.67:
        flags.append("holdout-weak")
    if session_holdout.success_rate is not None and session_holdout.success_rate < 0.67:
        flags.append("session-holdout-weak")
    if cohort_holdout.success_rate is not None and cohort_holdout.success_rate < 0.67:
        flags.append("cohort-holdout-weak")
    if branch_agreement is not None and branch_agreement.overall_status in {"shifted", "branch-sensitive"}:
        flags.append("branch-sensitive")
    if null_test.empirical_p_value is not None and null_test.empirical_p_value >= 0.2:
        flags.append("unstable")
    if subsample.min_similarity is not None and subsample.min_similarity < 0.5:
        flags.append("unstable")

    branch_score = _branch_score(branch_agreement)
    margin_score = max(0.0, min(1.0, ((margin_vs_rest or 0.0) + 0.2) / 0.4))
    loso_score = 0.0 if loso.success_rate is None else loso.success_rate
    session_score = 0.5 if session_holdout.success_rate is None else session_holdout.success_rate
    cohort_score = 0.5 if cohort_holdout.success_rate is None else cohort_holdout.success_rate
    null_score = 0.5 if null_test.empirical_p_value is None else max(0.0, min(1.0, 1.0 - null_test.empirical_p_value))
    subsample_score = 0.5 if subsample.mean_similarity is None else max(0.0, min(1.0, subsample.mean_similarity))
    robustness = float(np.mean([branch_score, margin_score, loso_score, session_score, cohort_score, null_score, subsample_score]))
    fragility = float(max(0.0, min(1.0, 1.0 - robustness)))

    if n_subjects < 2 or n_events < 2:
        confidence_tier = "insufficient"
    elif "unstable" in flags or (loso.success_rate is not None and loso.success_rate < 0.5):
        confidence_tier = "unstable"
    elif not flags and robustness >= 0.78:
        confidence_tier = "strong"
    elif margin_vs_rest is not None and margin_vs_rest > 0.08 and robustness >= 0.58:
        confidence_tier = "moderate"
    else:
        confidence_tier = "exploratory"

    return ReliabilityAssessment(
        confidence_tier=confidence_tier,
        status_flags=sorted(set(flags)),
        fragility_score=fragility,
        branch_stability=branch_agreement.overall_status if branch_agreement is not None else "single_branch_only",
        backbone_stability="not_evaluated",
        control_margin_ok=control_margin_ok,
        leave_one_subject_out=loso,
        session_holdout=session_holdout,
        cohort_holdout=cohort_holdout,
        null_test=null_test,
        subsample_stability=subsample,
        notes=notes,
    )


def augment_reliability_with_backbone_consensus(
    reliability: ReliabilityAssessment | None,
    *,
    backbone_consensus: BackboneConsensusSummary | None,
    backbone_evidence: list[BackboneEvidenceSummary],
) -> ReliabilityAssessment | None:
    if reliability is None or backbone_consensus is None:
        return reliability

    updated = ReliabilityAssessment.from_dict(reliability.to_dict())
    support_score = _backbone_score(backbone_consensus)
    updated.backbone_stability = backbone_consensus.overall_status
    updated.backbone_support_score = support_score
    updated.supporting_backbone_ids = list(backbone_consensus.agreeing_model_ids)

    used_evidence = [item for item in backbone_evidence if item.status == "used"]
    if not used_evidence:
        updated.status_flags = sorted(set([*updated.status_flags, "backbone-insufficient"]))
        updated.notes.append("no usable backbone evidence was produced during ensemble evaluation")
        return updated

    if backbone_consensus.overall_status == "insufficient":
        updated.status_flags = sorted(set([*updated.status_flags, "backbone-insufficient"]))
        updated.notes.append("backbone consensus was insufficient to strengthen the candidate")
    elif backbone_consensus.overall_status == "weak_agreement":
        updated.status_flags = sorted(set([*updated.status_flags, "backbone-weak"]))
        updated.notes.append("backbone agreement was weak across the evaluated ensemble")
    elif backbone_consensus.overall_status == "partial_agreement":
        updated.notes.append("backbone agreement was partial across the evaluated ensemble")
    elif backbone_consensus.overall_status == "strong_agreement":
        updated.notes.append("backbone agreement remained positive across the evaluated ensemble")

    if any((item.margin_vs_rest is not None and item.margin_vs_rest <= 0.0) for item in used_evidence):
        updated.status_flags = sorted(set([*updated.status_flags, "backbone-sensitive"]))
        updated.notes.append("at least one evaluated backbone did not preserve a positive control margin")

    if updated.fragility_score is not None and support_score is not None:
        prior_robustness = max(0.0, min(1.0, 1.0 - updated.fragility_score))
        blended_robustness = float(np.mean([prior_robustness, support_score]))
        updated.fragility_score = float(max(0.0, min(1.0, 1.0 - blended_robustness)))

    serious_flags = {"unstable", "holdout-weak", "session-holdout-weak", "cohort-holdout-weak", "control-weak"}
    current_tier = updated.confidence_tier
    if backbone_consensus.overall_status == "strong_agreement":
        if current_tier == "moderate" and not serious_flags.intersection(updated.status_flags):
            updated.confidence_tier = "strong"
    elif backbone_consensus.overall_status == "weak_agreement":
        if current_tier == "strong":
            updated.confidence_tier = "moderate"
        elif current_tier == "moderate":
            updated.confidence_tier = "exploratory"
    elif backbone_consensus.overall_status == "insufficient":
        if current_tier == "strong":
            updated.confidence_tier = "moderate"
        elif current_tier == "moderate":
            updated.confidence_tier = "exploratory"

    return updated
