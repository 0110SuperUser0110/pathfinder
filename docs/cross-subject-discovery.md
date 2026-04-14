# Cross-Subject Discovery

Pathfinder's Phase 2 discovery engine is designed to find shared EEG structure across subjects without collapsing the system into a feature-only classifier.

## Core Rule

The scientific result must remain EEG-backed.

That means discovery outputs must include signal-derived artifacts such as:

- prototype epochs
- representative exemplar epochs
- per-subject prototype stacks
- spectral summary tensors derived from EEG windows
- topography-style channel and phase derivatives

JSON manifests, scores, and markdown reports are supporting metadata only.

## Current Discovery Method

Pathfinder currently uses an interpretable baseline discovery engine.

For each branch and each target label it:

1. groups event-centered epochs across subjects
2. aligns common channels and phases conservatively
3. builds subject-level prototype epochs
4. builds a cross-subject prototype epoch
5. selects representative exemplar epochs
6. computes bandpower summaries and cross-subject similarity
7. compares within-label similarity against other labels in the same branch context
8. records branch agreement across preprocessing branches when available

## Negative Controls

Negative controls are explicit.

At minimum Pathfinder records:

- within-label similarity
- target-vs-rest similarity
- target-vs-other-label similarity
- strongest negative-control label
- margin between within-label and target-vs-rest similarity

This helps separate "something happened" from "this labeled condition has shared structure".

## Branch Agreement

A discovery candidate is stronger when it survives across preprocessing branches.

Pathfinder compares branch-specific candidates and classifies them as:

- `preserved`
- `weakened`
- `shifted`
- `branch-sensitive`

This is a guardrail against false confidence from heavy preprocessing.

## Output Layout

A discovery run writes candidates under:

```text
discovery_runs/<run_id>/candidates/<label_namespace>/<event_family>/<target_label>/<branch_name>/
```

Each candidate directory includes:

```text
prototype_epoch.npz
subject_prototypes.npz
exemplar_epochs.npz
spectral_summary.npz
topography_summary.npz
similarity_matrices.npz
candidate.json
control_summary.json
report.md
```

The first four files are the core result layer.

## Current Limits

This discovery engine is a first-pass baseline.

It is designed to be:

- interpretable
- signal-preserving
- branch-aware
- package-compatible

It is not yet:

- a final neuroscience claim engine
- a black-box foundation-model discovery workflow
- a substitute for replication and scientific validation

## Phase 3 Reliability Layer

Discovery candidates may now include:

- `reliability.json`
- leave-one-subject-out support
- grouped session holdout support
- grouped cohort holdout support when cohort metadata exists
- label-shuffle null summaries
- subsampling stability summaries
- confidence tiers and fragility scores

These are support layers around the EEG-backed candidate artifacts, not replacements for them.

## Discovery Run Provenance

Each discovery run now writes a provenance bundle at the run root:

- `run_manifest.json`
- `config_snapshot.json`
- `environment.json`
- `artifact_lineage.json`
- `run.log.jsonl`
- `warnings.json` when applicable

When candidates are packaged, that provenance bundle is also copied into each package as support metadata so the package stays auditable on its own.

## Phase 4 Ensemble Layer

Pathfinder now also supports an automated multi-backbone evaluation pass for `BIOT`, `CBraMod`, `EEGPT`, and `BrainOmni`. This layer operates on discovered candidates and adds support artifacts such as embedding summaries, cross-backbone consensus, and reliability updates without replacing the underlying EEG-backed outputs.
