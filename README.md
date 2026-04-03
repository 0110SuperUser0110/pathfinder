# Pathfinder

Pathfinder is a local-first EEG pattern discovery system.

The system is intended to search for recurring EEG signatures across sensory,
emotional, interoceptive, and other study domains without assuming that any
target pattern already exists. Raw EEG stays in signal-native formats. JSON is
used only as a manifest layer that points to packaged artifacts.

## Design Rules

- Do not serialize EEG signals into JSON.
- Store raw and processed signal artifacts in EEG-native or array-native formats.
- Keep the full EEG recording as the source of truth.
- Analyze event-centered timelines derived from the full EEG instead of only isolated slices.
- Partition discovery runs by metadata only as an analysis view, not as destructive preprocessing.
- Keep sex and gender as separate fields.
- Treat labels like `happy`, `pain`, or `sexual_arousal` as data values, never as hardcoded program logic.
- Treat the output as a pattern package, not a single score.

## Pattern Package Layout

Each discovered pattern should live in a self-contained directory under
`patterns/`.

```text
patterns/
  affect/
    female/
      self_report/
        pattern_003/
          manifest.json
          report.md
          signals/
            processed/
              prototype_epoch.fif
              exemplar_segments.zarr/
          derived/
            spectrogram.npz
            topomap.npz
            connectivity.npz
```

`manifest.json` stores metadata and file references only. The EEG signal itself
lives in the packaged artifact files.

## Included Scaffold

This repository includes:

- a pattern package specification in `docs/`
- a small Python package for manifests and package creation
- a CLI to create and validate pattern packages
- tests for the package builder
- a local model registry for the EEG engines

## Quick Start

Create a package from existing artifacts:

```powershell
python -m pathfinder.cli create `
  --root E:\pathfinder\output `
  --pattern-id pattern_003 `
  --study-id study_alpha `
  --event-family affect `
  --target-label positive_valence `
  --event-subtype self_report `
  --label-namespace research_ontology_v1 `
  --biological-sex female `
  --stimulus-modality self_report `
  --discovery-mode stratified `
  --source-model EEGPT `
  --source-model BrainOmni `
  --source-model BIOT `
  --source-model CBraMod `
  --band alpha `
  --band beta `
  --channel F3 `
  --channel F4 `
  --artifact "proto|prototype_epoch|processed_epoch|fif|E:\data\prototype_epoch.fif|Representative epoch" `
  --artifact "segments|exemplar_segments|processed_epoch|zarr|E:\data\segments.zarr|Supporting segments" `
  --artifact "spec|spectrogram|time_frequency|npz|E:\data\spectrogram.npz|Time-frequency power map"
```

Validate an existing package:

```powershell
python -m pathfinder.cli validate E:\pathfinder\output\patterns\affect\female\self_report\pattern_003
```

## Model Registry

Inspect local model readiness:

```powershell
$env:PYTHONPATH='E:\pathfinder\src'
python -m pathfinder.cli models list
python -m pathfinder.cli models show eegpt
python -m pathfinder.cli models check
```

Attempt to load a model once its Python stack is installed:

```powershell
$env:PYTHONPATH='E:\pathfinder\src'
python -m pathfinder.cli models probe biot
python -m pathfinder.cli models probe brainomni --variant tiny
```

## Local Runtime

Pathfinder now has a dedicated local runtime at:

- `E:\pathfinder\.python311\python.exe`
- `E:\pathfinder\.venv\Scripts\python.exe`

Current GPU-loadable engines on this Windows machine:

- `BIOT`
- `EEGPT`
- `BrainOmni`
- `CBraMod`

`EEGMamba` remains downloaded but not runnable natively on this setup because its official `mamba-ssm` dependency does not currently install cleanly on Windows.

## Next Build Steps

- add ingestion for EDF/BDF/FIF and BIDS EEG datasets
- add event-centered timeline extraction from full recordings
- add metadata partitioning and cohort balancing
- wire in the four runnable EEG-native analyzers
- add a local science-tuned report model for summarization only
