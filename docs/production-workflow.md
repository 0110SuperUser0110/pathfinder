# Production Workflow

Pathfinder's production workflow is designed around reproducibility, validation, and EEG artifact preservation.

## Doctrine

- EEG in, EEG out
- raw EEG remains the source of truth
- preprocessing stays branchable and non-destructive
- pattern timing matters
- summaries are indexes, not replacements for signal artifacts

## Recommended Operator Sequence

1. Ingest the raw recording and normalized events.
2. Validate the study inputs before discovery.
3. Build event-centered epochs.
4. Create one or more preprocessing branches.
5. Run discovery with an explicit run ID and seed.
6. If needed, run supported backbone evaluation on the discovery run with `BIOT`, `CBraMod`, `EEGPT`, and/or `BrainOmni`.
7. Inspect the resulting run, candidates, and packages.
8. Package only the candidates that retain valid EEG-backed artifacts.

## Run Provenance Bundle

Each major file-backed workflow now writes a local provenance bundle that can include:

- `run_manifest.json`
- `config_snapshot.json`
- `environment.json`
- `artifact_lineage.json`
- `warnings.json`
- `run.log.jsonl`

Discovery packaging can now embed this run bundle into each packaged pattern so a candidate package remains auditable away from the original run directory.

## Validation Philosophy

Pathfinder prefers:

- explicit errors over silent continuation on scientific integrity failures
- warnings over hidden caveats on weaker evidence
- partial completion over total failure when bad inputs can be isolated safely

## Reliability Philosophy

Reliability tiers are operational labels derived from measurable checks.

Current checks include:

- leave-one-subject-out validation
- grouped session holdout validation
- grouped cohort holdout validation when cohort metadata exists
- branch agreement
- backbone consensus when ensemble evidence exists
- label-shuffle null testing
- subsampling stability
- fragility scoring

These outputs help an operator decide whether a candidate is robust, branch-sensitive, low-support, control-weak, holdout-weak, unstable, or merely exploratory.

## Current Guarantees

Phase 3 improves:

- reproducibility
- provenance
- auditability
- structural validation
- operator inspection
- failure visibility

It does not guarantee final scientific truth or replace replication.
