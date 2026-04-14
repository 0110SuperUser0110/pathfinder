# Candidate Pattern Artifact Contract

A candidate pattern is Pathfinder's first cross-subject discovery artifact.

## Required Metadata

- `pattern_id`
- `label_namespace`
- `event_family`
- `target_label`
- `branch_name`
- `subject_ids`
- `event_ids`
- `sampling_rate_hz`
- `channel_names`
- `phase_names`
- `artifact_paths`
- `cross_subject_agreement`
- `control_summary`

## Required EEG-Backed Outputs

A valid candidate must retain file-backed EEG-derived artifacts such as:

- `prototype_epoch.npz`
- `subject_prototypes.npz`
- `exemplar_epochs.npz`

Additional supporting artifacts may include:

- `spectral_summary.npz`
- `topography_summary.npz`
- `similarity_matrices.npz`
- `branch_agreement.json`
- `reliability.json`
- `backbones/<model_id>/embeddings.npz`
- `backbones/<model_id>/evidence.json`
- `backbone_consensus.json`
- `report.md`

## Reliability Layer

Candidates may include a reliability assessment with:

- leave-one-subject-out results
- grouped session holdout results
- grouped cohort holdout results when cohort metadata exists
- null-test summary
- subsampling stability
- fragility score
- confidence tier

These fields support operator trust. They do not replace the EEG-backed artifacts.

## Ensemble Support Layer

When a candidate is evaluated by Pathfinder's ensemble layer, the contract also allows:

- per-backbone embedding artifacts
- per-backbone evidence summaries
- a candidate-level backbone consensus summary
- reliability updates carrying `backbone_stability`, `backbone_support_score`, and `supporting_backbone_ids`

Backbone support is secondary evidence. It must not replace the candidate's EEG-backed prototype, exemplar, and subject-prototype artifacts.
