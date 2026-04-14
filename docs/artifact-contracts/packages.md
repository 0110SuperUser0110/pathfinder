# Package Artifact Contract

A Pathfinder package is a portable artifact-first wrapper around a discovered pattern.

## Required Metadata

- `pattern_id`
- `partition`
- `analysis`
- `artifacts`
- `summary`

## Package Rules

- At least one packaged artifact must be EEG-backed.
- Paths inside the manifest must remain relative.
- The package may include reports and JSON metadata, but those are supporting layers only.
- The scientific result remains in packaged EEG-derived artifacts.

## Typical Packaged Artifacts

- prototype epoch
- exemplar epochs
- subject prototypes
- spectral summary
- topography summary
- candidate metadata
- control summary
- reliability summary
- backbone consensus summary
- backbone evidence summaries
- backbone embedding artifacts
- report
- embedded run manifest
- embedded config and environment snapshots
- embedded artifact lineage and run log

## Provenance Expectations

The package analysis block should identify:

- the discovery mode
- the producing model or engine label
- branch context
- run ID or other production provenance when available
