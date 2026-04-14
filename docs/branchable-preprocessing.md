# Branchable Preprocessing

Pathfinder does not treat preprocessing as a single mandatory pipeline.

## Why

For unknown-pattern discovery, aggressive cleaning can remove the very structure we are trying to discover.

Examples of risky defaults:

- aggressive denoising without preserved raw branches
- silent channel dropping
- forced rereferencing without provenance
- z-scoring by default
- collapsing recordings into feature tables before retaining signal artifacts

## Pathfinder Rule

Every preprocessing path must be branch-based and provenance-tracked.

That means:

- the raw-preserving branch always remains available
- every transform is logged
- every branch writes its own EEG artifacts
- missing channels are recorded explicitly
- channel alignment is auditable
- transformed artifacts never overwrite the source collection

## Phase 1 Branches

### `raw_preserving`

Use for minimal technical handling only.

Expected behavior:

- preserve timing
- preserve channel set unless the user explicitly requests a remap
- avoid aggressive denoising
- keep the closest available representation to the extracted event-centered EEG

### `light_clean`

Use for mild cleanup while preserving the raw-preserving branch.

Supported options now:

- notch filtering
- resampling
- baseline metadata retention
- optional baseline subtraction

### `comparison_safe`

Use when a later comparison requires more alignment across recordings or cohorts.

Supported options now:

- channel alignment to a named target montage
- rereferencing
- scaling
- optional notch filtering
- optional resampling

## Provenance Outputs

Each branch writes:

- `collection.json`
- `branch.json`
- one signal artifact per event under `events/`

`branch.json` records:

- source collection path
- branch name
- transform log
- branch warnings
- channel remap report when alignment occurs

## Scientific Guardrail

Normalization is useful for comparison, but dangerous as a universal default.

Pathfinder's baseline stance is:

- preserve the raw branch
- allow comparison branches
- never assume the cleaned branch is the only truth

## Phase 3 Operational Additions

Preprocessing branches now also write local run provenance files alongside `branch.json` and `collection.json`. This makes branch creation reproducible and auditable without changing the underlying EEG-first output rule.
