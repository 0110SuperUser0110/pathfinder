from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import utc_now_iso


def _coerce_path(value: str | Path) -> str:
    return str(Path(value))


@dataclass(slots=True)
class CrossSubjectAgreement:
    subject_ids: list[str]
    event_ids: list[str]
    n_subjects: int
    n_events: int
    within_label_similarity: float | None = None
    between_label_similarity: float | None = None
    margin_vs_rest: float | None = None
    subject_consistency_score: float | None = None
    trial_consistency_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_ids": list(self.subject_ids),
            "event_ids": list(self.event_ids),
            "n_subjects": self.n_subjects,
            "n_events": self.n_events,
            "within_label_similarity": self.within_label_similarity,
            "between_label_similarity": self.between_label_similarity,
            "margin_vs_rest": self.margin_vs_rest,
            "subject_consistency_score": self.subject_consistency_score,
            "trial_consistency_score": self.trial_consistency_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrossSubjectAgreement":
        return cls(
            subject_ids=list(data.get("subject_ids", [])),
            event_ids=list(data.get("event_ids", [])),
            n_subjects=int(data.get("n_subjects", 0)),
            n_events=int(data.get("n_events", 0)),
            within_label_similarity=data.get("within_label_similarity"),
            between_label_similarity=data.get("between_label_similarity"),
            margin_vs_rest=data.get("margin_vs_rest"),
            subject_consistency_score=data.get("subject_consistency_score"),
            trial_consistency_score=data.get("trial_consistency_score"),
        )


@dataclass(slots=True)
class ControlComparisonSummary:
    target_label: str
    branch_name: str
    target_vs_rest_similarity: float | None = None
    target_vs_other_labels: dict[str, float] = field(default_factory=dict)
    strongest_negative_control_label: str = ""
    strongest_negative_control_similarity: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_label": self.target_label,
            "branch_name": self.branch_name,
            "target_vs_rest_similarity": self.target_vs_rest_similarity,
            "target_vs_other_labels": dict(self.target_vs_other_labels),
            "strongest_negative_control_label": self.strongest_negative_control_label,
            "strongest_negative_control_similarity": self.strongest_negative_control_similarity,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControlComparisonSummary":
        return cls(
            target_label=data.get("target_label", ""),
            branch_name=data.get("branch_name", ""),
            target_vs_rest_similarity=data.get("target_vs_rest_similarity"),
            target_vs_other_labels={str(key): float(value) for key, value in data.get("target_vs_other_labels", {}).items()},
            strongest_negative_control_label=data.get("strongest_negative_control_label", ""),
            strongest_negative_control_similarity=data.get("strongest_negative_control_similarity"),
            notes=list(data.get("notes", [])),
        )


@dataclass(slots=True)
class BranchAgreement:
    target_label: str
    label_namespace: str
    event_family: str
    reference_branch: str
    overall_status: str
    branch_status: dict[str, str] = field(default_factory=dict)
    pairwise_similarity: dict[str, float] = field(default_factory=dict)
    margin_by_branch: dict[str, float | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_label": self.target_label,
            "label_namespace": self.label_namespace,
            "event_family": self.event_family,
            "reference_branch": self.reference_branch,
            "overall_status": self.overall_status,
            "branch_status": dict(self.branch_status),
            "pairwise_similarity": dict(self.pairwise_similarity),
            "margin_by_branch": dict(self.margin_by_branch),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BranchAgreement":
        return cls(
            target_label=data.get("target_label", ""),
            label_namespace=data.get("label_namespace", ""),
            event_family=data.get("event_family", ""),
            reference_branch=data.get("reference_branch", ""),
            overall_status=data.get("overall_status", ""),
            branch_status={str(key): str(value) for key, value in data.get("branch_status", {}).items()},
            pairwise_similarity={str(key): float(value) for key, value in data.get("pairwise_similarity", {}).items()},
            margin_by_branch={str(key): (None if value is None else float(value)) for key, value in data.get("margin_by_branch", {}).items()},
            notes=list(data.get("notes", [])),
        )


@dataclass(slots=True)
class LeaveOneSubjectOutSummary:
    n_folds: int
    success_rate: float | None = None
    mean_margin: float | None = None
    min_margin: float | None = None
    held_out_subject_ids: list[str] = field(default_factory=list)
    failed_subject_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "success_rate": self.success_rate,
            "mean_margin": self.mean_margin,
            "min_margin": self.min_margin,
            "held_out_subject_ids": list(self.held_out_subject_ids),
            "failed_subject_ids": list(self.failed_subject_ids),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LeaveOneSubjectOutSummary":
        return cls(
            n_folds=int(data.get("n_folds", 0)),
            success_rate=data.get("success_rate"),
            mean_margin=data.get("mean_margin"),
            min_margin=data.get("min_margin"),
            held_out_subject_ids=list(data.get("held_out_subject_ids", [])),
            failed_subject_ids=list(data.get("failed_subject_ids", [])),
            notes=list(data.get("notes", [])),
        )


@dataclass(slots=True)
class NullTestSummary:
    iterations: int
    actual_margin: float | None = None
    null_mean_margin: float | None = None
    null_std_margin: float | None = None
    null_p95_margin: float | None = None
    empirical_p_value: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "actual_margin": self.actual_margin,
            "null_mean_margin": self.null_mean_margin,
            "null_std_margin": self.null_std_margin,
            "null_p95_margin": self.null_p95_margin,
            "empirical_p_value": self.empirical_p_value,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NullTestSummary":
        return cls(
            iterations=int(data.get("iterations", 0)),
            actual_margin=data.get("actual_margin"),
            null_mean_margin=data.get("null_mean_margin"),
            null_std_margin=data.get("null_std_margin"),
            null_p95_margin=data.get("null_p95_margin"),
            empirical_p_value=data.get("empirical_p_value"),
            notes=list(data.get("notes", [])),
        )


@dataclass(slots=True)
class SubsampleStabilitySummary:
    iterations: int
    subsample_fraction: float
    mean_similarity: float | None = None
    min_similarity: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "subsample_fraction": self.subsample_fraction,
            "mean_similarity": self.mean_similarity,
            "min_similarity": self.min_similarity,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubsampleStabilitySummary":
        return cls(
            iterations=int(data.get("iterations", 0)),
            subsample_fraction=float(data.get("subsample_fraction", 0.0)),
            mean_similarity=data.get("mean_similarity"),
            min_similarity=data.get("min_similarity"),
            notes=list(data.get("notes", [])),
        )


@dataclass(slots=True)
class GroupedHoldoutSummary:
    grouping_name: str
    n_groups: int
    success_rate: float | None = None
    mean_margin: float | None = None
    min_margin: float | None = None
    held_out_group_ids: list[str] = field(default_factory=list)
    failed_group_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grouping_name": self.grouping_name,
            "n_groups": self.n_groups,
            "success_rate": self.success_rate,
            "mean_margin": self.mean_margin,
            "min_margin": self.min_margin,
            "held_out_group_ids": list(self.held_out_group_ids),
            "failed_group_ids": list(self.failed_group_ids),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GroupedHoldoutSummary":
        return cls(
            grouping_name=data.get("grouping_name", ""),
            n_groups=int(data.get("n_groups", 0)),
            success_rate=data.get("success_rate"),
            mean_margin=data.get("mean_margin"),
            min_margin=data.get("min_margin"),
            held_out_group_ids=list(data.get("held_out_group_ids", [])),
            failed_group_ids=list(data.get("failed_group_ids", [])),
            notes=list(data.get("notes", [])),
        )


@dataclass(slots=True)
class ReliabilityAssessment:
    confidence_tier: str
    status_flags: list[str] = field(default_factory=list)
    fragility_score: float | None = None
    branch_stability: str = ""
    backbone_stability: str = ""
    backbone_support_score: float | None = None
    supporting_backbone_ids: list[str] = field(default_factory=list)
    control_margin_ok: bool | None = None
    leave_one_subject_out: LeaveOneSubjectOutSummary | None = None
    session_holdout: GroupedHoldoutSummary | None = None
    cohort_holdout: GroupedHoldoutSummary | None = None
    null_test: NullTestSummary | None = None
    subsample_stability: SubsampleStabilitySummary | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "confidence_tier": self.confidence_tier,
            "status_flags": list(self.status_flags),
            "fragility_score": self.fragility_score,
            "branch_stability": self.branch_stability,
            "backbone_stability": self.backbone_stability,
            "backbone_support_score": self.backbone_support_score,
            "supporting_backbone_ids": list(self.supporting_backbone_ids),
            "control_margin_ok": self.control_margin_ok,
            "notes": list(self.notes),
        }
        if self.leave_one_subject_out is not None:
            payload["leave_one_subject_out"] = self.leave_one_subject_out.to_dict()
        if self.session_holdout is not None:
            payload["session_holdout"] = self.session_holdout.to_dict()
        if self.cohort_holdout is not None:
            payload["cohort_holdout"] = self.cohort_holdout.to_dict()
        if self.null_test is not None:
            payload["null_test"] = self.null_test.to_dict()
        if self.subsample_stability is not None:
            payload["subsample_stability"] = self.subsample_stability.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReliabilityAssessment":
        loso_data = data.get("leave_one_subject_out")
        session_data = data.get("session_holdout")
        cohort_data = data.get("cohort_holdout")
        null_data = data.get("null_test")
        subsample_data = data.get("subsample_stability")
        return cls(
            confidence_tier=data.get("confidence_tier", "exploratory"),
            status_flags=list(data.get("status_flags", [])),
            fragility_score=data.get("fragility_score"),
            branch_stability=data.get("branch_stability", ""),
            backbone_stability=data.get("backbone_stability", ""),
            backbone_support_score=data.get("backbone_support_score"),
            supporting_backbone_ids=list(data.get("supporting_backbone_ids", [])),
            control_margin_ok=data.get("control_margin_ok"),
            leave_one_subject_out=LeaveOneSubjectOutSummary.from_dict(loso_data) if loso_data else None,
            session_holdout=GroupedHoldoutSummary.from_dict(session_data) if session_data else None,
            cohort_holdout=GroupedHoldoutSummary.from_dict(cohort_data) if cohort_data else None,
            null_test=NullTestSummary.from_dict(null_data) if null_data else None,
            subsample_stability=SubsampleStabilitySummary.from_dict(subsample_data) if subsample_data else None,
            notes=list(data.get("notes", [])),
        )


@dataclass(slots=True)
class BackboneEvidenceSummary:
    model_id: str
    variant_id: str = ""
    status: str = "used"
    subject_count: int = 0
    embedding_dim: int = 0
    selected_phases: list[str] = field(default_factory=list)
    within_label_similarity: float | None = None
    target_vs_rest_similarity: float | None = None
    margin_vs_rest: float | None = None
    strongest_negative_control_label: str = ""
    strongest_negative_control_similarity: float | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "variant_id": self.variant_id,
            "status": self.status,
            "subject_count": self.subject_count,
            "embedding_dim": self.embedding_dim,
            "selected_phases": list(self.selected_phases),
            "within_label_similarity": self.within_label_similarity,
            "target_vs_rest_similarity": self.target_vs_rest_similarity,
            "margin_vs_rest": self.margin_vs_rest,
            "strongest_negative_control_label": self.strongest_negative_control_label,
            "strongest_negative_control_similarity": self.strongest_negative_control_similarity,
            "artifact_paths": dict(self.artifact_paths),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackboneEvidenceSummary":
        return cls(
            model_id=data.get("model_id", ""),
            variant_id=data.get("variant_id", ""),
            status=data.get("status", "used"),
            subject_count=int(data.get("subject_count", 0)),
            embedding_dim=int(data.get("embedding_dim", 0)),
            selected_phases=list(data.get("selected_phases", [])),
            within_label_similarity=data.get("within_label_similarity"),
            target_vs_rest_similarity=data.get("target_vs_rest_similarity"),
            margin_vs_rest=data.get("margin_vs_rest"),
            strongest_negative_control_label=data.get("strongest_negative_control_label", ""),
            strongest_negative_control_similarity=data.get("strongest_negative_control_similarity"),
            artifact_paths={str(key): _coerce_path(value) for key, value in data.get("artifact_paths", {}).items()},
            notes=list(data.get("notes", [])),
        )


@dataclass(slots=True)
class BackboneConsensusSummary:
    requested_model_ids: list[str]
    successful_model_ids: list[str]
    agreeing_model_ids: list[str]
    overall_status: str
    mean_margin: float | None = None
    margins_by_model: dict[str, float | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_model_ids": list(self.requested_model_ids),
            "successful_model_ids": list(self.successful_model_ids),
            "agreeing_model_ids": list(self.agreeing_model_ids),
            "overall_status": self.overall_status,
            "mean_margin": self.mean_margin,
            "margins_by_model": dict(self.margins_by_model),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackboneConsensusSummary":
        return cls(
            requested_model_ids=list(data.get("requested_model_ids", [])),
            successful_model_ids=list(data.get("successful_model_ids", [])),
            agreeing_model_ids=list(data.get("agreeing_model_ids", [])),
            overall_status=data.get("overall_status", ""),
            mean_margin=data.get("mean_margin"),
            margins_by_model={str(key): (None if value is None else float(value)) for key, value in data.get("margins_by_model", {}).items()},
            notes=list(data.get("notes", [])),
        )


@dataclass(slots=True)
class BackboneEvaluationSummary:
    evaluation_id: str
    output_root: str
    requested_model_ids: list[str]
    completed_model_ids: list[str]
    failed_model_ids: list[str]
    candidate_count: int
    status: str = "success"
    run_manifest_path: str = ""
    config_snapshot_path: str = ""
    environment_path: str = ""
    artifact_lineage_path: str = ""
    warnings_path: str = ""
    log_path: str = ""
    notes: list[str] = field(default_factory=list)
    created_at_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "output_root": self.output_root,
            "requested_model_ids": list(self.requested_model_ids),
            "completed_model_ids": list(self.completed_model_ids),
            "failed_model_ids": list(self.failed_model_ids),
            "candidate_count": self.candidate_count,
            "status": self.status,
            "run_manifest_path": self.run_manifest_path,
            "config_snapshot_path": self.config_snapshot_path,
            "environment_path": self.environment_path,
            "artifact_lineage_path": self.artifact_lineage_path,
            "warnings_path": self.warnings_path,
            "log_path": self.log_path,
            "notes": list(self.notes),
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackboneEvaluationSummary":
        return cls(
            evaluation_id=data.get("evaluation_id", ""),
            output_root=_coerce_path(data.get("output_root", "")),
            requested_model_ids=list(data.get("requested_model_ids", [])),
            completed_model_ids=list(data.get("completed_model_ids", [])),
            failed_model_ids=list(data.get("failed_model_ids", [])),
            candidate_count=int(data.get("candidate_count", 0)),
            status=data.get("status", "success"),
            run_manifest_path=_coerce_path(data.get("run_manifest_path", "")),
            config_snapshot_path=_coerce_path(data.get("config_snapshot_path", "")),
            environment_path=_coerce_path(data.get("environment_path", "")),
            artifact_lineage_path=_coerce_path(data.get("artifact_lineage_path", "")),
            warnings_path=_coerce_path(data.get("warnings_path", "")),
            log_path=_coerce_path(data.get("log_path", "")),
            notes=list(data.get("notes", [])),
            created_at_utc=data.get("created_at_utc", utc_now_iso()),
        )


@dataclass(slots=True)
class RunIssueRecord:
    stage: str
    severity: str
    message: str
    target_label: str = ""
    branch_name: str = ""
    subject_id: str = ""
    path: str = ""
    created_at_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "severity": self.severity,
            "message": self.message,
            "target_label": self.target_label,
            "branch_name": self.branch_name,
            "subject_id": self.subject_id,
            "path": self.path,
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunIssueRecord":
        return cls(
            stage=data.get("stage", ""),
            severity=data.get("severity", "warning"),
            message=data.get("message", ""),
            target_label=data.get("target_label", ""),
            branch_name=data.get("branch_name", ""),
            subject_id=data.get("subject_id", ""),
            path=_coerce_path(data.get("path", "")),
            created_at_utc=data.get("created_at_utc", utc_now_iso()),
        )


@dataclass(slots=True)
class CandidatePattern:
    pattern_id: str
    label_namespace: str
    event_family: str
    target_label: str
    branch_name: str
    event_subtypes: list[str]
    subject_ids: list[str]
    event_ids: list[str]
    sampling_rate_hz: float
    channel_names: list[str]
    phase_names: list[str]
    available_bands: list[str]
    dominant_bands: list[str] = field(default_factory=list)
    dominant_channels: list[str] = field(default_factory=list)
    strongest_phases: list[str] = field(default_factory=list)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    candidate_root: str = ""
    run_id: str = ""
    backbone_ids: list[str] = field(default_factory=list)
    cross_subject_agreement: CrossSubjectAgreement = field(default_factory=lambda: CrossSubjectAgreement([], [], 0, 0))
    control_summary: ControlComparisonSummary = field(default_factory=lambda: ControlComparisonSummary("", ""))
    branch_agreement: BranchAgreement | None = None
    reliability: ReliabilityAssessment | None = None
    backbone_evidence: list[BackboneEvidenceSummary] = field(default_factory=list)
    backbone_consensus: BackboneConsensusSummary | None = None
    summary_notes: list[str] = field(default_factory=list)
    packaged_pattern_path: str = ""
    created_at_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pattern_id": self.pattern_id,
            "label_namespace": self.label_namespace,
            "event_family": self.event_family,
            "target_label": self.target_label,
            "branch_name": self.branch_name,
            "event_subtypes": list(self.event_subtypes),
            "subject_ids": list(self.subject_ids),
            "event_ids": list(self.event_ids),
            "sampling_rate_hz": self.sampling_rate_hz,
            "channel_names": list(self.channel_names),
            "phase_names": list(self.phase_names),
            "available_bands": list(self.available_bands),
            "dominant_bands": list(self.dominant_bands),
            "dominant_channels": list(self.dominant_channels),
            "strongest_phases": list(self.strongest_phases),
            "artifact_paths": dict(self.artifact_paths),
            "candidate_root": self.candidate_root,
            "run_id": self.run_id,
            "backbone_ids": list(self.backbone_ids),
            "cross_subject_agreement": self.cross_subject_agreement.to_dict(),
            "control_summary": self.control_summary.to_dict(),
            "summary_notes": list(self.summary_notes),
            "packaged_pattern_path": self.packaged_pattern_path,
            "created_at_utc": self.created_at_utc,
        }
        if self.branch_agreement is not None:
            payload["branch_agreement"] = self.branch_agreement.to_dict()
        if self.reliability is not None:
            payload["reliability"] = self.reliability.to_dict()
        if self.backbone_evidence:
            payload["backbone_evidence"] = [item.to_dict() for item in self.backbone_evidence]
        if self.backbone_consensus is not None:
            payload["backbone_consensus"] = self.backbone_consensus.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidatePattern":
        branch_agreement_data = data.get("branch_agreement")
        reliability_data = data.get("reliability")
        backbone_consensus_data = data.get("backbone_consensus")
        return cls(
            pattern_id=data["pattern_id"],
            label_namespace=data.get("label_namespace", ""),
            event_family=data.get("event_family", ""),
            target_label=data.get("target_label", ""),
            branch_name=data.get("branch_name", ""),
            event_subtypes=list(data.get("event_subtypes", [])),
            subject_ids=list(data.get("subject_ids", [])),
            event_ids=list(data.get("event_ids", [])),
            sampling_rate_hz=float(data.get("sampling_rate_hz", 0.0)),
            channel_names=list(data.get("channel_names", [])),
            phase_names=list(data.get("phase_names", [])),
            available_bands=list(data.get("available_bands", [])),
            dominant_bands=list(data.get("dominant_bands", [])),
            dominant_channels=list(data.get("dominant_channels", [])),
            strongest_phases=list(data.get("strongest_phases", [])),
            artifact_paths={str(key): _coerce_path(value) for key, value in data.get("artifact_paths", {}).items()},
            candidate_root=_coerce_path(data.get("candidate_root", "")),
            run_id=data.get("run_id", ""),
            backbone_ids=list(data.get("backbone_ids", [])),
            cross_subject_agreement=CrossSubjectAgreement.from_dict(data["cross_subject_agreement"]),
            control_summary=ControlComparisonSummary.from_dict(data["control_summary"]),
            branch_agreement=BranchAgreement.from_dict(branch_agreement_data) if branch_agreement_data else None,
            reliability=ReliabilityAssessment.from_dict(reliability_data) if reliability_data else None,
            backbone_evidence=[BackboneEvidenceSummary.from_dict(item) for item in data.get("backbone_evidence", [])],
            backbone_consensus=BackboneConsensusSummary.from_dict(backbone_consensus_data) if backbone_consensus_data else None,
            summary_notes=list(data.get("summary_notes", [])),
            packaged_pattern_path=_coerce_path(data.get("packaged_pattern_path", "")),
            created_at_utc=data.get("created_at_utc", utc_now_iso()),
        )


@dataclass(slots=True)
class DiscoveryRunSummary:
    run_id: str
    output_root: str
    collection_paths: list[str]
    branch_names: list[str]
    candidates: list[CandidatePattern]
    status: str = "success"
    rng_seed: int = 0
    backbone_ids: list[str] = field(default_factory=list)
    backbone_evaluation: BackboneEvaluationSummary | None = None
    branch_agreements: list[BranchAgreement] = field(default_factory=list)
    packaged_pattern_paths: list[str] = field(default_factory=list)
    failures: list[RunIssueRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    run_manifest_path: str = ""
    config_snapshot_path: str = ""
    environment_path: str = ""
    artifact_lineage_path: str = ""
    warnings_path: str = ""
    log_path: str = ""
    created_at_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output_root": self.output_root,
            "collection_paths": list(self.collection_paths),
            "branch_names": list(self.branch_names),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "status": self.status,
            "rng_seed": self.rng_seed,
            "backbone_ids": list(self.backbone_ids),
            "backbone_evaluation": None if self.backbone_evaluation is None else self.backbone_evaluation.to_dict(),
            "branch_agreements": [agreement.to_dict() for agreement in self.branch_agreements],
            "packaged_pattern_paths": list(self.packaged_pattern_paths),
            "failures": [item.to_dict() for item in self.failures],
            "notes": list(self.notes),
            "run_manifest_path": self.run_manifest_path,
            "config_snapshot_path": self.config_snapshot_path,
            "environment_path": self.environment_path,
            "artifact_lineage_path": self.artifact_lineage_path,
            "warnings_path": self.warnings_path,
            "log_path": self.log_path,
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscoveryRunSummary":
        return cls(
            run_id=data["run_id"],
            output_root=_coerce_path(data["output_root"]),
            collection_paths=[_coerce_path(path) for path in data.get("collection_paths", [])],
            branch_names=list(data.get("branch_names", [])),
            candidates=[CandidatePattern.from_dict(item) for item in data.get("candidates", [])],
            status=data.get("status", "success"),
            rng_seed=int(data.get("rng_seed", 0)),
            backbone_ids=list(data.get("backbone_ids", [])),
            backbone_evaluation=BackboneEvaluationSummary.from_dict(data["backbone_evaluation"]) if data.get("backbone_evaluation") else None,
            branch_agreements=[BranchAgreement.from_dict(item) for item in data.get("branch_agreements", [])],
            packaged_pattern_paths=[_coerce_path(path) for path in data.get("packaged_pattern_paths", [])],
            failures=[RunIssueRecord.from_dict(item) for item in data.get("failures", [])],
            notes=list(data.get("notes", [])),
            run_manifest_path=_coerce_path(data.get("run_manifest_path", "")),
            config_snapshot_path=_coerce_path(data.get("config_snapshot_path", "")),
            environment_path=_coerce_path(data.get("environment_path", "")),
            artifact_lineage_path=_coerce_path(data.get("artifact_lineage_path", "")),
            warnings_path=_coerce_path(data.get("warnings_path", "")),
            log_path=_coerce_path(data.get("log_path", "")),
            created_at_utc=data.get("created_at_utc", utc_now_iso()),
        )
