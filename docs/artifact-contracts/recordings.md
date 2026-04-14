# Recording Artifact Contract

A recording artifact in Pathfinder is the canonical index for a raw EEG recording.

## Required Metadata

- `recording_id`
- `subject_id`
- `session_id`
- `label_namespace`
- `source_path`
- `source_format`
- `channel_names`
- `sampling_rate_hz`
- `n_samples`
- `duration_seconds`
- `source_provenance`

## Contract Rules

- The JSON reference is not the EEG itself.
- `source_path` must point to a file-backed raw EEG artifact.
- Raw EEG remains the source of truth.
- Channel names must be present and non-duplicate.
- Sampling rate and duration must be positive.
- Source provenance must identify how the recording was loaded.

## Acceptable Source Formats

- `NPZ`
- `EDF`
- `BDF`
- `FIF`

## Production Notes

Pathfinder never overwrites the source recording during ingest. Ingest only creates a local index and optional normalized event table plus a run provenance bundle.

When a source path looks BIDS-like, Pathfinder validation also checks for expected local sidecars such as the EEG JSON, `channels.tsv`, and `events.tsv`. These are validation hooks, not a requirement that every recording be stored in BIDS.
