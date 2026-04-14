from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ArtifactContract:
    artifact_type: str
    description: str
    required_metadata: list[str]
    optional_metadata: list[str] = field(default_factory=list)
    supported_formats: list[str] = field(default_factory=list)
    signal_backed: bool = False
    channel_axis: str = ""
    time_axis: str = ""
    label_fields: list[str] = field(default_factory=list)
    provenance_fields: list[str] = field(default_factory=list)
    parent_link_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "description": self.description,
            "required_metadata": list(self.required_metadata),
            "optional_metadata": list(self.optional_metadata),
            "supported_formats": list(self.supported_formats),
            "signal_backed": self.signal_backed,
            "channel_axis": self.channel_axis,
            "time_axis": self.time_axis,
            "label_fields": list(self.label_fields),
            "provenance_fields": list(self.provenance_fields),
            "parent_link_fields": list(self.parent_link_fields),
            "notes": list(self.notes),
        }


RECORDING_CONTRACT = ArtifactContract(
    artifact_type="recording_reference",
    description="Canonical index for a raw EEG recording. The referenced source file remains the source of truth.",
    required_metadata=[
        "recording_id",
        "subject_id",
        "session_id",
        "label_namespace",
        "source_path",
        "source_format",
        "channel_names",
        "sampling_rate_hz",
        "n_samples",
        "duration_seconds",
        "source_provenance",
    ],
    supported_formats=["json", "npz", "edf", "bdf", "fif"],
    signal_backed=True,
    channel_axis="channel_names",
    time_axis="n_samples",
    provenance_fields=["source_provenance", "source_path"],
    parent_link_fields=["source_path"],
    notes=["The JSON index is not the EEG itself; it must point to a file-backed raw recording artifact."],
)

EVENT_TABLE_CONTRACT = ArtifactContract(
    artifact_type="event_table",
    description="Normalized event metadata aligned to a source recording.",
    required_metadata=[
        "event_id",
        "recording_id",
        "onset_seconds",
        "duration_seconds",
        "event_family",
        "target_label",
        "label_namespace",
    ],
    supported_formats=["json", "csv", "tsv"],
    signal_backed=False,
    label_fields=["event_family", "target_label", "event_subtype", "label_namespace"],
    parent_link_fields=["recording_id"],
)

EPOCH_COLLECTION_CONTRACT = ArtifactContract(
    artifact_type="epoch_collection",
    description="Event-centered EEG segments derived from a full recording while preserving time-bounded phase metadata.",
    required_metadata=[
        "collection_id",
        "recording",
        "window_config",
        "channel_names",
        "sampling_rate_hz",
        "artifacts",
    ],
    supported_formats=["json", "npz"],
    signal_backed=True,
    channel_axis="channel_names",
    time_axis="phase sample axis",
    label_fields=["event_family", "target_label", "event_subtype", "label_namespace"],
    provenance_fields=["source_event_table_path", "recording", "metadata"],
    parent_link_fields=["recording.source_path", "artifact.signal_path"],
    notes=["Each artifact entry must reference a file-backed EEG segment artifact."],
)

PREPROCESS_BRANCH_CONTRACT = ArtifactContract(
    artifact_type="preprocess_branch",
    description="Branch-specific derived epoch collection with explicit transform lineage and warnings.",
    required_metadata=[
        "branch_name",
        "source_collection_path",
        "output_collection_path",
        "config",
        "transforms",
    ],
    supported_formats=["json", "npz"],
    signal_backed=True,
    channel_axis="output collection channel_names",
    time_axis="phase sample axis",
    provenance_fields=["source_collection_path", "config", "transforms", "channel_report"],
    parent_link_fields=["source_collection_path", "output_collection_path"],
    notes=["No branch may overwrite the source collection or replace the raw-preserving branch."],
)

CANDIDATE_PATTERN_CONTRACT = ArtifactContract(
    artifact_type="candidate_pattern",
    description="Cross-subject discovered pattern with EEG-backed prototype and exemplar artifacts plus lightweight summaries.",
    required_metadata=[
        "pattern_id",
        "label_namespace",
        "event_family",
        "target_label",
        "branch_name",
        "subject_ids",
        "event_ids",
        "sampling_rate_hz",
        "channel_names",
        "phase_names",
        "artifact_paths",
        "cross_subject_agreement",
        "control_summary",
    ],
    supported_formats=["json", "npz", "md"],
    signal_backed=True,
    channel_axis="channel_names",
    time_axis="phase sample axis",
    label_fields=["event_family", "target_label", "event_subtype", "label_namespace"],
    provenance_fields=["artifact_paths", "branch_agreement", "reliability", "run_id"],
    parent_link_fields=["candidate_root", "artifact_paths"],
    notes=["The primary scientific result must remain in file-backed EEG artifacts such as prototype and exemplar epochs."],
)

PATTERN_PACKAGE_CONTRACT = ArtifactContract(
    artifact_type="pattern_package",
    description="Portable package for a discovered pattern, containing EEG-backed artifacts plus a manifest and report.",
    required_metadata=["pattern_id", "partition", "analysis", "artifacts", "summary"],
    supported_formats=["json", "md", "npz", "edf", "fif", "zarr"],
    signal_backed=True,
    label_fields=["partition.event_family", "partition.target_label", "partition.event_subtype", "partition.label_namespace"],
    provenance_fields=["analysis.source_models", "analysis.notes"],
    parent_link_fields=["artifacts.path"],
)

DISCOVERY_RUN_CONTRACT = ArtifactContract(
    artifact_type="discovery_run",
    description="Run-level summary and provenance bundle for a Pathfinder discovery execution.",
    required_metadata=[
        "run_id",
        "output_root",
        "collection_paths",
        "branch_names",
        "candidates",
        "status",
        "rng_seed",
    ],
    supported_formats=["json", "jsonl"],
    signal_backed=False,
    provenance_fields=[
        "run_manifest_path",
        "config_snapshot_path",
        "environment_path",
        "artifact_lineage_path",
        "warnings_path",
        "log_path",
    ],
    parent_link_fields=["collection_paths", "packaged_pattern_paths"],
)


ARTIFACT_CONTRACTS = {
    contract.artifact_type: contract
    for contract in [
        RECORDING_CONTRACT,
        EVENT_TABLE_CONTRACT,
        EPOCH_COLLECTION_CONTRACT,
        PREPROCESS_BRANCH_CONTRACT,
        CANDIDATE_PATTERN_CONTRACT,
        PATTERN_PACKAGE_CONTRACT,
        DISCOVERY_RUN_CONTRACT,
    ]
}


def get_artifact_contract(artifact_type: str) -> ArtifactContract:
    try:
        return ARTIFACT_CONTRACTS[artifact_type]
    except KeyError as exc:
        raise KeyError(f"unknown artifact contract: {artifact_type}") from exc
