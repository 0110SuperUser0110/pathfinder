# Multi-Backbone Orchestration

Phase 4 adds the first automated ensemble layer to Pathfinder.

## Core Rule

Backbones are support analyzers around the EEG-backed discovery result.

They do not replace:

- prototype epochs
- exemplar epochs
- subject prototypes
- spectral or topography derivatives

Those remain the primary scientific artifacts.

## Current Scope

Pathfinder now supports a first automated backbone evaluation pass over discovered candidates.

The workflow is:

1. run the interpretable cross-subject discovery engine
2. load supported local backbones
3. prepare candidate subject prototypes into backbone-specific input tensors
4. compute subject-level embeddings
5. compare within-label and between-label similarity in backbone space
6. attach candidate-level backbone evidence and cross-backbone consensus

## Current Supported Automated Backbones

The current automated prep paths are implemented for:

- `BIOT`
- `CBraMod`
- `EEGPT`
- `BrainOmni`

Safety behavior for the broader two:

- `EEGPT` is fed through deterministic remapping into its released 58-channel canonical order. Missing canonical channels are zero-filled and recorded in the evidence notes.
- `BrainOmni` is fed through deterministic 10-20 position inference when explicit sensor geometry is unavailable in Pathfinder artifacts. Channels without a deterministic mapping are excluded and recorded in the evidence notes.

## Candidate Outputs

Backbone evaluation writes support artifacts under each candidate directory:

```text
backbones/<model_id>/embeddings.npz
backbones/<model_id>/evidence.json
backbone_consensus.json
reliability.json
```

These are secondary support layers around the candidate, not replacements for the EEG-backed artifacts. `reliability.json` is updated only when the candidate already had a Pathfinder reliability assessment from the interpretable discovery run.

## CLI

Run discovery and backbone evaluation together:

```powershell
$env:PYTHONPATH='E:\pathfinder\src'
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli discover-ensemble `
  --collection E:\pathfinder\output\preprocess\...\collection.json `
  --output-root E:\pathfinder\output `
  --run-id study_ensemble `
  --backbone biot `
  --backbone cbramod `
  --backbone eegpt `
  --backbone brainomni
```

Evaluate an existing run:

```powershell
$env:PYTHONPATH='E:\pathfinder\src'
& 'E:\pathfinder\.venv\Scripts\python.exe' -m pathfinder.cli backbone-evaluate `
  E:\pathfinder\output\discovery_runs\study_run\run_summary.json `
  --backbone biot `
  --backbone cbramod `
  --backbone eegpt `
  --backbone brainomni
```

## Consensus Meaning

Current cross-backbone consensus is intentionally simple and interpretable:

- `strong_agreement`
- `partial_agreement`
- `weak_agreement`
- `insufficient`

This is a support layer for operator judgment. It is not a claim of final scientific truth.

## Reliability Interaction

When a candidate already has a Pathfinder reliability assessment, ensemble evaluation updates that assessment with:

- `backbone_stability`
- `backbone_support_score`
- `supporting_backbone_ids`

Weak or insufficient ensemble agreement can demote an overconfident candidate. Strong agreement can strengthen a moderate candidate when the rest of the reliability checks are already healthy.
