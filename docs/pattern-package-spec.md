# Pattern Package Specification

This specification defines how Pathfinder stores discovered EEG patterns.

## Purpose

The output of a discovery run is the pattern itself, not a JSON summary. A
pattern package therefore combines signal artifacts with a lightweight manifest
that makes the package searchable, comparable, and reproducible.

## Generic Labeling Rule

Pathfinder must never hardcode a specific state such as `happy`, `fear`,
`sexual_arousal`, or `pain` into the processing logic. Those are data labels,
not code branches.

Use generic manifest fields instead:

- `event_family`: broad domain such as `affect`, `vision`, `pain`, `reward`, `memory`
- `target_label`: the state or condition label from the study or normalized ontology
- `event_subtype`: optional refinement such as `self_report`, `stimulus_onset`, `task_block`
- `label_namespace`: the ontology or study-specific namespace that produced the label

The code should operate the same way regardless of the label values.

## Event-Centered Timeline Rule

The full EEG recording is the source of truth. Pattern search should operate on
an event-centered timeline derived from that full recording, not on destructive
fixed chopping alone.

Recommended event context for each labeled event:

- pre-event context
- onset transition
- in-event interval
- offset transition
- post-event context

## Package Principles

1. Raw EEG and processed EEG remain in signal-native formats such as `EDF`,
   `BDF`, `FIF`, `NPY`, `NPZ`, `HDF5`, or `Zarr`.
2. Derived artifacts such as spectrograms, topographies, and connectivity
   matrices are stored in array-native formats such as `NPY`, `NPZ`, `HDF5`, or
   `Parquet`.
3. The manifest points to those files and records the metadata required to
   compare them across studies.
4. The report is descriptive only. It does not replace the stored signal
   artifacts.

## Required Metadata

Each pattern package should capture, at minimum:

- `pattern_id`
- `study_ids`
- `event_family`
- `target_label`
- `event_subtype`
- `label_namespace`
- `biological_sex`
- `gender_identity`
- `stimulus_modality`
- `age_band`
- `cohort_label`
- `discovery_mode`
- `source_models`

Keep sex and gender separate. If one field is unavailable in a source dataset,
leave it blank rather than collapsing them into one field.

## Partitioning Rules

Discovery runs should be stratified before comparison whenever possible.

Recommended partition dimensions:

- `study_id`
- `subject_id`
- `session_id`
- `event_family`
- `target_label`
- `event_subtype`
- `label_namespace`
- `biological_sex`
- `gender_identity`
- `stimulus_modality`
- `age_band`
- `device_family`
- `sampling_rate`

Never compare pooled groups without preserving enough metadata to detect
confounds from acquisition hardware, study protocol, or label imbalance.

## Required Artifact Types

Each pattern package should include at least one signal artifact:

- `prototype_epoch`
- `exemplar_segments`
- `raw_segment`
- `processed_epoch`

Useful derived artifacts include:

- `spectrogram`
- `band_power_map`
- `coupling_map`
- `topomap`
- `connectivity_matrix`
- `state_transition_matrix`
- `embedding_projection`

## Filesystem Layout

Packages are stored under:

```text
patterns/<event_family>/<biological_sex>/<stimulus_modality>/<pattern_id>/
```

Inside each package:

```text
manifest.json
report.md
signals/
  raw/
  processed/
derived/
reports/
support/
```

## Manifest Scope

`manifest.json` stores:

- package metadata
- relative paths to packaged artifacts
- summary descriptors
- validation and reproducibility notes

`manifest.json` does not store the EEG signal arrays themselves.

## Example Manifest

```json
{
  "pattern_id": "pattern_003",
  "partition": {
    "study_ids": ["study_alpha"],
    "event_family": "affect",
    "target_label": "positive_valence",
    "event_subtype": "self_report",
    "label_namespace": "research_ontology_v1",
    "biological_sex": "female",
    "gender_identity": "",
    "stimulus_modality": "self_report",
    "age_band": "25_34",
    "cohort_label": "adult"
  },
  "analysis": {
    "discovery_mode": "stratified",
    "source_models": ["EEGPT", "BrainOmni", "BIOT", "CBraMod"]
  },
  "artifacts": [
    {
      "artifact_id": "proto",
      "role": "prototype_epoch",
      "representation": "processed_epoch",
      "format": "fif",
      "path": "signals/processed/prototype_epoch.fif",
      "description": "Representative event-centered segment"
    }
  ]
}
```

## Validation Rules

A valid pattern package must satisfy all of the following:

1. `manifest.json` exists.
2. At least one artifact is a signal artifact.
3. Every artifact path is relative to the package root.
4. Every referenced file or directory exists.
5. `source_models` is non-empty.
6. `event_family` and `target_label` are non-empty.
