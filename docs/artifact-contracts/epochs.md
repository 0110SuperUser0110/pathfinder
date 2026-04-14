# Epoch Artifact Contract

An epoch collection is an event-centered EEG artifact set derived from a full recording.

## Required Metadata

- `collection_id`
- `recording`
- `window_config`
- `channel_names`
- `sampling_rate_hz`
- `artifacts`

Each artifact entry must include:

- `event`
- `signal_path`
- `format`
- `phase_ranges`
- `phase_shapes`

## Signal Contract

- Epoch outputs must remain EEG-backed files.
- Pathfinder currently stores epoch artifacts as `NPZ`.
- Each signal artifact must retain channel and sample axes.
- Exact phase timing metadata must remain attached.
- Baseline windows may be recorded without forcing subtraction.

## Required Linkage

- Each epoch collection must link back to its source recording.
- Each event artifact must link back to the event metadata and recording ID.
- Source event table provenance must remain available.

## Validation Focus

Validation checks:

- unreadable or missing signal files
- duplicate event IDs
- malformed phase shapes
- channel-count mismatches
- sample-rate mismatches
- empty collections
- out-of-contract metadata
