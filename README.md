# Pathfinder

Pathfinder is a local-first EEG discovery system.

Its operating rule is simple:

- EEG in
- EEG out

Inputs are raw EEG recordings plus event metadata or Phase 1 epoch artifacts. Outputs remain EEG-backed artifacts. JSON manifests, reports, and scores are supporting indexes around those artifacts, not replacements for them.

## Core Philosophy

- local-first
- signal-native
- event-centered
- generic across labels and states
- full recordings remain the source of truth
- raw data is never overwritten or discarded
- unknown patterns must not be normalized away by default
- shared-pattern discovery must preserve branch-aware evidence

## What Exists Now

Pathfinder currently includes:

- a pattern package specification in `docs/`
- package creation and validation for file-backed pattern artifacts
- a local model registry for EEG-native backbones
- a raw EEG ingestion layer
- event table normalization
- event-centered epoch extraction
- branchable preprocessing with provenance tracking
- an interpretable cross-subject discovery engine
- negative-control comparison logic
- branch agreement scoring across preprocessing branches
- packaging integration for discovered shared patterns
- tests for ingestion, epoching, preprocessing, discovery, packaging, and model registry status
- run manifests, config snapshots, environment snapshots, and artifact lineage for file-backed workflows
- strict validation and inspection tooling for studies, collections, runs, candidates, and packages
- a first reliability layer with subject, session, and cohort holdouts plus label-shuffle null testing and subsampling stability
- an initial automated multi-backbone orchestration layer for supported local backbones
- safe automated ensemble prep for `BIOT`, `CBraMod`, `EEGPT`, and `BrainOmni`

## EEG In, EEG Out

Pathfinder does not treat scores, embeddings, or reports as the scientific result.

The core outputs remain on disk as signal-native or signal-derived EEG artifacts such as:

- source recordings: `FIF`, `EDF`, `BDF`, or `NPZ`
- event-centered epoch artifacts: `NPZ`
- preprocessing branches: `NPZ`
- discovery prototypes: `NPZ`
- representative cross-subject exemplars: `NPZ`
- subject prototype stacks: `NPZ`
- spectral summary tensors: `NPZ`
- topography-style channel and phase derivatives: `NPZ`

JSON manifests and markdown reports exist to make those artifacts searchable, reproducible, and packageable.

## Why Branchable Preprocessing Matters

Aggressive normalization can erase the very structure Pathfinder is supposed to discover.

Pathfinder therefore preserves multiple preprocessing branches instead of forcing one canonical pipeline:

- `raw_preserving`: minimal handling, no destructive defaults
- `light_clean`: optional notch, optional resample, optional baseline handling
- `comparison_safe`: optional alignment, rereference, and scaling for comparison

A candidate that survives across branches is stronger than one that appears only after heavy transformation.

## Current Recording Support

Implemented now:

- `NPZ` EEG recordings out of the box for deterministic local workflows and tests
- optional `FIF`, `EDF`, and `BDF` loading through the `mne` package when installed

The ingestion layer remains file-format first, but validation now includes lightweight BIDS-aware sidecar checks when a source path looks BIDS-like.

## Local Workflow

### 1. Ingest a recording

```powershell
$env:PYTHONPATH='E:\pathfinder\src'
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli ingest `
  E:\data\recording_alpha.npz `
  --output-root E:\pathfinder\output `
  --event-table E:\data\events.csv
```

### 2. Build event-centered epochs

```powershell
$env:PYTHONPATH='E:\pathfinder\src'
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli epoch `
  --recording E:\pathfinder\output\recordings\recording_alpha\recording.json `
  --events E:\pathfinder\output\recordings\recording_alpha\events.json `
  --output-root E:\pathfinder\output `
  --pre-event-seconds 2.0 `
  --onset-seconds 1.0 `
  --offset-seconds 1.0 `
  --post-event-seconds 2.0 `
  --baseline-start-offset -1.0 `
  --baseline-end-offset 0.0
```

### 3. Create preprocessing branches

```powershell
$env:PYTHONPATH='E:\pathfinder\src'
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli preprocess `
  --collection E:\pathfinder\output\epochs\recording_alpha\...\collection.json `
  --output-root E:\pathfinder\output `
  --branch comparison_safe `
  --target-channel F3 `
  --target-channel F4 `
  --target-channel C3 `
  --target-channel C4 `
  --rereference-mode average `
  --notch-hz 60 `
  --resample-hz 128
```

### 4. Discover shared patterns across subjects

```powershell
$env:PYTHONPATH='E:\pathfinder\src'
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli discover `
  --collection E:\pathfinder\output\preprocess\subject_01\...\raw_preserving\collection.json `
  --collection E:\pathfinder\output\preprocess\subject_02\...\raw_preserving\collection.json `
  --collection E:\pathfinder\output\preprocess\subject_03\...\raw_preserving\collection.json `
  --collection E:\pathfinder\output\preprocess\subject_01\...\light_clean\collection.json `
  --collection E:\pathfinder\output\preprocess\subject_02\...\light_clean\collection.json `
  --collection E:\pathfinder\output\preprocess\subject_03\...\light_clean\collection.json `
  --output-root E:\pathfinder\output `
  --run-id strawberry_discovery `
  --min-subjects 3
```

This writes a discovery run under `output/discovery_runs/<run_id>/` with branch-specific candidate directories containing EEG-backed artifacts such as:

- `prototype_epoch.npz`
- `subject_prototypes.npz`
- `exemplar_epochs.npz`
- `spectral_summary.npz`
- `topography_summary.npz`
- `similarity_matrices.npz`
- `candidate.json`
- `control_summary.json`
- `branch_agreement.json`
- `reliability.json`
- `report.md`

The run root also now contains a provenance bundle:

- `run_manifest.json`
- `config_snapshot.json`
- `environment.json`
- `artifact_lineage.json`
- `warnings.json` when warnings exist
- `run.log.jsonl`
- `run_summary.json`

If `--package-root` is provided, Pathfinder also packages each candidate into the existing pattern-package layout and embeds the discovery run bundle into each package as support provenance.

### 5. Run multi-backbone ensemble evaluation

```powershell
$env:PYTHONPATH='E:\pathfinder\src'
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli discover-ensemble `
  --collection E:\pathfinder\output\preprocess\subject_01\...\raw_preserving\collection.json `
  --collection E:\pathfinder\output\preprocess\subject_02\...\raw_preserving\collection.json `
  --collection E:\pathfinder\output\preprocess\subject_03\...\raw_preserving\collection.json `
  --output-root E:\pathfinder\output `
  --run-id strawberry_ensemble `
  --backbone biot `
  --backbone cbramod `
  --backbone eegpt `
  --backbone brainomni `
  --device cpu `
  --package-root E:\pathfinder\packages
```

This keeps the EEG-backed discovery artifacts primary, then adds support artifacts such as:

- `backbones/<model_id>/embeddings.npz`
- `backbones/<model_id>/evidence.json`
- `backbone_consensus.json`
- an updated `reliability.json` carrying backbone stability and support score

## What Phase 2 Actually Discovers

The current discovery engine is intentionally interpretable and conservative.

It does the following:

- groups event-centered epochs by `target_label` across subjects
- computes subject-level prototypes within each branch
- builds cross-subject prototype epochs and representative exemplars
- compares within-label similarity against other labels in the same branch context
- records target-vs-rest control margins
- compares branch-specific candidates across `raw_preserving`, `light_clean`, and `comparison_safe` when available
- classifies branch survival as `preserved`, `weakened`, `shifted`, or `branch-sensitive`
- emits topography-style derived inspection artifacts alongside the core EEG-backed outputs

It does not claim final scientific validity. It provides a first real signal-preserving baseline discovery layer.

## Pattern Packaging

Pathfinder's package system remains artifact-first. Discovery candidates can be packaged because the underlying outputs stay file-backed and EEG-derived.

A packaged candidate includes the pattern itself in EEG-backed form, not just a statement that a pattern was found.

## Model Registry

Inspect local model readiness:

```powershell
$env:PYTHONPATH='E:\pathfinder\src'
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli models list
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli models show eegpt
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli models probe biot --device cuda
```

Current GPU-loadable engines on this Windows machine:

- `BIOT`
- `EEGPT`
- `BrainOmni`
- `CBraMod`

`EEGMamba` remains downloaded but not runnable natively on this setup because its official `mamba-ssm` dependency does not currently install cleanly on Windows.

## Current Boundary

Pathfinder still does not include:

- a web UI
- an API server
- a database
- a final report-writing layer
- a claim of final scientific validity beyond the implemented methods

Current orchestration limits:

- automated backbone evaluation is implemented for `BIOT`, `CBraMod`, `EEGPT`, and `BrainOmni`
- `EEGPT` uses deterministic remapping into the released 58-channel canonical order and records any zero-filled canonical channels in the backbone evidence notes
- `BrainOmni` uses deterministic inferred 10-20 positions when explicit sensor geometry is absent, and it records any excluded channels in the backbone evidence notes
- `EEGMamba` remains outside the automated ensemble flow on this Windows setup

What it does include now is a real local path from raw EEG to event-centered artifacts to branch-aware cross-subject discovery candidates that remain EEG-backed on disk.




## Phase 3 Hardening

Phase 3 adds the first production trust layer without changing Pathfinder's scientific identity.

What is new now:

- deterministic run IDs and RNG seeds for discovery
- run manifests and environment/config snapshots
- artifact lineage tracking
- explicit validation for studies, collections, runs, candidates, and packages
- structured JSONL run logging
- partial-failure recording instead of silent continuation
- reliability scoring tied to measurable checks rather than opaque confidence claims

## Reliability Layer

Pathfinder now attaches a first reliability assessment to each discovered candidate.

Current checks include:

- leave-one-subject-out validation
- grouped session holdout checks
- grouped cohort holdout checks when cohort metadata exists
- branch stability via branch agreement
- ensemble backbone stability via cross-backbone consensus
- label-shuffle null testing hooks
- subsampling stability checks
- fragility scoring and confidence tiers

Confidence tiers are intentionally modest:

- `exploratory`
- `moderate`
- `strong`
- `unstable`
- `insufficient`

These are operational trust labels, not final scientific truth claims.

## Validation And Inspection

New operator commands:

```powershell
$env:PYTHONPATH='E:\pathfinder\src'
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli validate-study --collection E:\pathfinder\output\preprocess\...\collection.json
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli validate-collection E:\pathfinder\output\preprocess\...\collection.json
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli validate-run E:\pathfinder\output\discovery_runs\strawberry_discovery\run_summary.json
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli validate-package E:\pathfinder\packages\patterns\...
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli inspect-collection E:\pathfinder\output\preprocess\...\collection.json
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli inspect-candidate E:\pathfinder\output\discovery_runs\...\candidate.json
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli inspect-run E:\pathfinder\output\discovery_runs\strawberry_discovery\run_summary.json
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli inspect-package E:\pathfinder\packages\patterns\...
```

These commands are local and file-backed. They exist to answer:

- what is in this artifact
- what labels and branches are present
- how many subjects and trials support it
- what warnings or failures were recorded
- which run generated it

## Artifact Contracts

Phase 3 formalizes artifact contracts for:

- recording references
- event-centered epoch collections
- preprocessing branches
- candidate patterns
- packaged patterns
- discovery runs

See:

- `docs/artifact-contracts/recordings.md`
- `docs/artifact-contracts/epochs.md`
- `docs/artifact-contracts/candidate-patterns.md`
- `docs/artifact-contracts/packages.md`
- `docs/production-workflow.md`



