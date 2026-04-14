from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .analysis_models import BaselineWindowSpec, EpochWindowConfig, PreprocessBranchConfig
from .backbone_discovery import BackboneDiscoveryError, evaluate_backbones_for_run
from .discovery import DiscoveryError, discover_shared_patterns
from .eeg_registry import ModelLoadError, default_registry
from .epochs import build_event_epochs
from .ingest import ingest_recording
from .models import AnalysisSpec, PartitionSpec, PatternSummary
from .package import ArtifactInput, PatternPackageBuilder
from .preprocess import preprocess_epoch_collection
from .validation import (
    summarize_candidate,
    summarize_collection,
    summarize_package,
    summarize_run,
    validate_candidate_pattern_artifact,
    validate_discovery_run_artifact,
    validate_epoch_collection_artifact,
    validate_pattern_package_artifact,
    validate_study,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pathfinder EEG pattern package tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Inspect a raw EEG recording and write a local recording index")
    ingest_parser.add_argument("source_path", type=Path, help="Raw EEG recording path")
    ingest_parser.add_argument("--output-root", type=Path, required=True, help="Output root for recording indexes")
    ingest_parser.add_argument("--recording-id", default="")
    ingest_parser.add_argument("--subject-id", default="")
    ingest_parser.add_argument("--session-id", default="")
    ingest_parser.add_argument("--label-namespace", default="")
    ingest_parser.add_argument("--event-table", type=Path, default=None, help="Optional CSV/TSV/JSON event table")
    ingest_parser.add_argument("--run-id", default="")
    ingest_parser.add_argument("--overwrite", action="store_true")

    epoch_parser = subparsers.add_parser("epoch", help="Build event-centered EEG epoch artifacts from a recording")
    epoch_parser.add_argument("--recording", type=Path, required=True, help="Path to a recording.json index or source EEG file")
    epoch_parser.add_argument("--events", type=Path, required=True, help="Path to a normalized or source event table")
    epoch_parser.add_argument("--output-root", type=Path, required=True, help="Output root for epoch collections")
    epoch_parser.add_argument("--collection-id", default="")
    epoch_parser.add_argument("--pre-event-seconds", type=float, default=0.0)
    epoch_parser.add_argument("--onset-seconds", type=float, default=0.0)
    epoch_parser.add_argument("--offset-seconds", type=float, default=0.0)
    epoch_parser.add_argument("--post-event-seconds", type=float, default=0.0)
    epoch_parser.add_argument("--sustained-seconds", type=float, default=None)
    epoch_parser.add_argument("--baseline-start-offset", type=float, default=None)
    epoch_parser.add_argument("--baseline-end-offset", type=float, default=None)
    epoch_parser.add_argument("--run-id", default="")
    epoch_parser.add_argument("--overwrite", action="store_true")

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Create a named preprocessing branch from an epoch collection without overwriting the source branch",
    )
    preprocess_parser.add_argument("--collection", type=Path, required=True, help="Path to epoch collection.json")
    preprocess_parser.add_argument("--output-root", type=Path, required=True, help="Output root for preprocessing branches")
    preprocess_parser.add_argument(
        "--branch",
        required=True,
        choices=["raw_preserving", "light_clean", "comparison_safe"],
        help="Named preprocessing branch",
    )
    preprocess_parser.add_argument("--notch-hz", dest="notch_hz", action="append", type=float, default=[])
    preprocess_parser.add_argument("--notch-bandwidth-hz", type=float, default=1.0)
    preprocess_parser.add_argument("--resample-hz", type=float, default=None)
    preprocess_parser.add_argument("--baseline-mode", default="none", choices=["none", "metadata_only", "subtract_mean"])
    preprocess_parser.add_argument("--target-channel", dest="target_channels", action="append", default=[])
    preprocess_parser.add_argument("--target-channels-file", type=Path, default=None)
    preprocess_parser.add_argument("--rereference-mode", default="none", choices=["none", "average", "channels"])
    preprocess_parser.add_argument("--reference-channel", dest="reference_channels", action="append", default=[])
    preprocess_parser.add_argument("--scale-factor", type=float, default=None)
    preprocess_parser.add_argument("--run-id", default="")
    preprocess_parser.add_argument("--overwrite", action="store_true")

    discover_parser = subparsers.add_parser(
        "discover",
        help="Run interpretable cross-subject discovery on Phase 1 epoch collections or preprocessing branches",
    )
    discover_parser.add_argument("--collection", dest="collections", action="append", default=[])
    discover_parser.add_argument("--collection-list", type=Path, default=None, help="Optional text file listing collection.json paths")
    discover_parser.add_argument("--output-root", type=Path, required=True, help="Output root for discovery runs")
    discover_parser.add_argument("--run-id", default="")
    discover_parser.add_argument("--target-label", dest="target_labels", action="append", default=[])
    discover_parser.add_argument("--branch-filter", dest="branches", action="append", default=[])
    discover_parser.add_argument("--min-subjects", type=int, default=2)
    discover_parser.add_argument("--max-exemplars", type=int, default=6)
    discover_parser.add_argument("--package-root", type=Path, default=None)
    discover_parser.add_argument("--backbone", dest="backbone_ids", action="append", default=[], help="Optional provenance-only EEG backbone IDs associated with this discovery run")
    discover_parser.add_argument("--seed", type=int, default=0)
    discover_parser.add_argument("--null-iterations", type=int, default=64)
    discover_parser.add_argument("--subsample-iterations", type=int, default=32)
    discover_parser.add_argument("--overwrite", action="store_true")

    discover_ensemble_parser = subparsers.add_parser(
        "discover-ensemble",
        help="Run interpretable discovery first, then evaluate discovered candidates across supported local EEG backbones",
    )
    discover_ensemble_parser.add_argument("--collection", dest="collections", action="append", default=[])
    discover_ensemble_parser.add_argument("--collection-list", type=Path, default=None, help="Optional text file listing collection.json paths")
    discover_ensemble_parser.add_argument("--output-root", type=Path, required=True, help="Output root for discovery runs")
    discover_ensemble_parser.add_argument("--run-id", default="")
    discover_ensemble_parser.add_argument("--target-label", dest="target_labels", action="append", default=[])
    discover_ensemble_parser.add_argument("--branch-filter", dest="branches", action="append", default=[])
    discover_ensemble_parser.add_argument("--min-subjects", type=int, default=2)
    discover_ensemble_parser.add_argument("--max-exemplars", type=int, default=6)
    discover_ensemble_parser.add_argument("--package-root", type=Path, default=None)
    discover_ensemble_parser.add_argument("--backbone", dest="backbone_ids", action="append", default=[], help="Backbone IDs to evaluate, for example biot, cbramod, eegpt, or brainomni")
    discover_ensemble_parser.add_argument("--device", default="cpu")
    discover_ensemble_parser.add_argument("--seed", type=int, default=0)
    discover_ensemble_parser.add_argument("--null-iterations", type=int, default=64)
    discover_ensemble_parser.add_argument("--subsample-iterations", type=int, default=32)
    discover_ensemble_parser.add_argument("--overwrite", action="store_true")

    backbone_eval_parser = subparsers.add_parser(
        "backbone-evaluate",
        help="Evaluate an existing discovery run across supported local EEG backbones and attach consensus metadata",
    )
    backbone_eval_parser.add_argument("run_path", type=Path, help="Path to run_summary.json or the discovery run directory")
    backbone_eval_parser.add_argument("--backbone", dest="backbone_ids", action="append", default=[], help="Backbone IDs to evaluate, for example biot, cbramod, eegpt, or brainomni")
    backbone_eval_parser.add_argument("--device", default="cpu")
    backbone_eval_parser.add_argument("--package-root", type=Path, default=None)
    backbone_eval_parser.add_argument("--overwrite", action="store_true")

    create_parser = subparsers.add_parser("create", help="Create a pattern package from existing artifacts")
    create_parser.add_argument("--root", type=Path, required=True, help="Output root for packaged patterns")
    create_parser.add_argument("--pattern-id", required=True)
    create_parser.add_argument("--study-id", dest="study_ids", action="append", default=[])
    create_parser.add_argument("--event-family", required=True)
    create_parser.add_argument("--target-label", required=True)
    create_parser.add_argument("--event-subtype", default="")
    create_parser.add_argument("--label-namespace", default="")
    create_parser.add_argument("--biological-sex", default="")
    create_parser.add_argument("--gender-identity", default="")
    create_parser.add_argument("--stimulus-modality", default="")
    create_parser.add_argument("--age-band", default="")
    create_parser.add_argument("--cohort-label", default="")
    create_parser.add_argument("--discovery-mode", required=True)
    create_parser.add_argument("--source-model", dest="source_models", action="append", required=True)
    create_parser.add_argument("--candidate-signature", default="")
    create_parser.add_argument("--band", dest="bands", action="append", default=[])
    create_parser.add_argument("--channel", dest="channels", action="append", default=[])
    create_parser.add_argument("--temporal-notes", default="")
    create_parser.add_argument("--notes", default="")
    create_parser.add_argument("--overwrite", action="store_true")
    create_parser.add_argument("--artifact", action="append", required=True, help="artifact_id|role|representation|format|source_path|description")

    validate_parser = subparsers.add_parser("validate", help="Validate an existing pattern package")
    validate_parser.add_argument("package_dir", type=Path)
    validate_parser.add_argument("--json", action="store_true")

    validate_collection_parser = subparsers.add_parser("validate-collection", help="Validate an epoch collection artifact")
    validate_collection_parser.add_argument("collection_path", type=Path)
    validate_collection_parser.add_argument("--json", action="store_true")

    validate_package_parser = subparsers.add_parser("validate-package", help="Validate a pattern package")
    validate_package_parser.add_argument("package_dir", type=Path)
    validate_package_parser.add_argument("--json", action="store_true")

    validate_run_parser = subparsers.add_parser("validate-run", help="Validate a discovery run summary and its referenced artifacts")
    validate_run_parser.add_argument("run_path", type=Path)
    validate_run_parser.add_argument("--json", action="store_true")

    validate_study_parser = subparsers.add_parser("validate-study", help="Validate study integrity across epoch collections")
    validate_study_parser.add_argument("--collection", dest="collections", action="append", default=[])
    validate_study_parser.add_argument("--collection-list", type=Path, default=None)
    validate_study_parser.add_argument("--min-subjects", type=int, default=2)
    validate_study_parser.add_argument("--json", action="store_true")

    inspect_collection_parser = subparsers.add_parser("inspect-collection", help="Inspect an epoch collection")
    inspect_collection_parser.add_argument("collection_path", type=Path)
    inspect_collection_parser.add_argument("--json", action="store_true")

    inspect_candidate_parser = subparsers.add_parser("inspect-candidate", help="Inspect a candidate pattern artifact")
    inspect_candidate_parser.add_argument("candidate_path", type=Path)
    inspect_candidate_parser.add_argument("--json", action="store_true")

    inspect_package_parser = subparsers.add_parser("inspect-package", help="Inspect a packaged pattern")
    inspect_package_parser.add_argument("package_dir", type=Path)
    inspect_package_parser.add_argument("--json", action="store_true")

    inspect_run_parser = subparsers.add_parser("inspect-run", help="Inspect a discovery run")
    inspect_run_parser.add_argument("run_path", type=Path)
    inspect_run_parser.add_argument("--json", action="store_true")

    models_parser = subparsers.add_parser("models", help="Inspect local EEG engine readiness")
    models_subparsers = models_parser.add_subparsers(dest="models_command", required=True)
    models_subparsers.add_parser("list", help="List all configured local EEG engines")
    show_parser = models_subparsers.add_parser("show", help="Show detailed status for a single EEG engine")
    show_parser.add_argument("model_id")
    check_parser = models_subparsers.add_parser("check", help="Return non-zero if a model is not runnable")
    check_parser.add_argument("model_id", nargs="?", default="")
    probe_parser = models_subparsers.add_parser("probe", help="Attempt to load a model with the current environment")
    probe_parser.add_argument("model_id")
    probe_parser.add_argument("--variant", default="")
    probe_parser.add_argument("--device", default="cpu")

    return parser


def parse_artifact(argument: str) -> ArtifactInput:
    parts = [part.strip() for part in argument.split("|")]
    if len(parts) < 5:
        raise ValueError("artifact must have at least 5 pipe-separated fields")
    artifact_id, role, representation, fmt, source_path, *rest = parts
    description = rest[0] if rest else ""
    return ArtifactInput(
        artifact_id=artifact_id,
        role=role,
        representation=representation,
        format=fmt,
        source_path=Path(source_path),
        description=description,
    )


def _load_channels(args: argparse.Namespace) -> list[str]:
    channels = list(args.target_channels)
    if args.target_channels_file is not None:
        channels.extend(
            line.strip()
            for line in args.target_channels_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return channels


def _load_collection_inputs(args: argparse.Namespace) -> list[Path]:
    collection_paths = [Path(path) for path in args.collections]
    if args.collection_list is not None:
        collection_paths.extend(
            Path(line.strip())
            for line in args.collection_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return collection_paths


def _emit_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _emit_summary(summary: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        _emit_json(summary)
        return
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


def _emit_validation(report, *, as_json: bool) -> int:
    if as_json:
        _emit_json(report.to_dict())
    else:
        for line in report.render_lines():
            print(line)
    return 0 if report.ok else 1


def handle_ingest(args: argparse.Namespace) -> int:
    recording_path, _, events = ingest_recording(
        args.source_path,
        output_root=args.output_root,
        recording_id=args.recording_id,
        subject_id=args.subject_id,
        session_id=args.session_id,
        label_namespace=args.label_namespace,
        event_table_path=args.event_table,
        run_id=args.run_id,
        command="pathfinder ingest",
        overwrite=args.overwrite,
    )
    print(recording_path)
    if args.event_table is not None:
        print(recording_path.parent / "events.json")
    else:
        print(f"events_loaded: {len(events)}")
    return 0


def handle_epoch(args: argparse.Namespace) -> int:
    baseline_window = None
    if (args.baseline_start_offset is None) != (args.baseline_end_offset is None):
        print("baseline_start_offset and baseline_end_offset must be provided together", file=sys.stderr)
        return 1
    if args.baseline_start_offset is not None and args.baseline_end_offset is not None:
        baseline_window = BaselineWindowSpec(
            start_offset_seconds=args.baseline_start_offset,
            end_offset_seconds=args.baseline_end_offset,
        )
    window_config = EpochWindowConfig(
        pre_event_seconds=args.pre_event_seconds,
        onset_seconds=args.onset_seconds,
        offset_seconds=args.offset_seconds,
        post_event_seconds=args.post_event_seconds,
        sustained_seconds=args.sustained_seconds,
        baseline_window=baseline_window,
    )
    collection_path, _ = build_event_epochs(
        args.recording,
        args.events,
        output_root=args.output_root,
        window_config=window_config,
        collection_id=args.collection_id,
        run_id=args.run_id,
        command="pathfinder epoch",
        overwrite=args.overwrite,
    )
    print(collection_path)
    return 0


def handle_preprocess(args: argparse.Namespace) -> int:
    config = PreprocessBranchConfig(
        branch_name=args.branch,
        notch_hz=args.notch_hz,
        notch_bandwidth_hz=args.notch_bandwidth_hz,
        resample_hz=args.resample_hz,
        baseline_mode=args.baseline_mode,
        align_channels=_load_channels(args),
        rereference_mode=args.rereference_mode,
        reference_channels=list(args.reference_channels),
        scale_factor=args.scale_factor,
    )
    branch_path, _ = preprocess_epoch_collection(
        args.collection,
        output_root=args.output_root,
        config=config,
        run_id=args.run_id,
        command="pathfinder preprocess",
        overwrite=args.overwrite,
    )
    print(branch_path)
    return 0


def handle_discover(args: argparse.Namespace) -> int:
    collection_paths = _load_collection_inputs(args)
    if not collection_paths:
        print("at least one --collection or --collection-list entry is required", file=sys.stderr)
        return 1
    try:
        summary_path, summary = discover_shared_patterns(
            collection_paths=collection_paths,
            output_root=args.output_root,
            run_id=args.run_id,
            target_labels=args.target_labels,
            branches=args.branches,
            min_subjects=args.min_subjects,
            max_exemplars=args.max_exemplars,
            package_root=args.package_root,
            backbone_ids=args.backbone_ids,
            rng_seed=args.seed,
            null_iterations=args.null_iterations,
            subsample_iterations=args.subsample_iterations,
            overwrite=args.overwrite,
        )
    except (DiscoveryError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(summary_path)
    for package_path in summary.packaged_pattern_paths:
        print(package_path)
    return 0


def handle_discover_ensemble(args: argparse.Namespace) -> int:
    collection_paths = _load_collection_inputs(args)
    if not collection_paths:
        print("at least one --collection or --collection-list entry is required", file=sys.stderr)
        return 1
    if not args.backbone_ids:
        print("at least one --backbone is required for discover-ensemble", file=sys.stderr)
        return 1
    try:
        summary_path, _ = discover_shared_patterns(
            collection_paths=collection_paths,
            output_root=args.output_root,
            run_id=args.run_id,
            target_labels=args.target_labels,
            branches=args.branches,
            min_subjects=args.min_subjects,
            max_exemplars=args.max_exemplars,
            package_root=None,
            backbone_ids=[],
            rng_seed=args.seed,
            null_iterations=args.null_iterations,
            subsample_iterations=args.subsample_iterations,
            overwrite=args.overwrite,
        )
        summary_path, summary = evaluate_backbones_for_run(
            summary_path,
            backbone_ids=args.backbone_ids,
            device=args.device,
            package_root=args.package_root,
            overwrite=args.overwrite,
        )
    except (BackboneDiscoveryError, DiscoveryError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(summary_path)
    for package_path in summary.packaged_pattern_paths:
        print(package_path)
    return 0


def handle_backbone_evaluate(args: argparse.Namespace) -> int:
    if not args.backbone_ids:
        print("at least one --backbone is required for backbone-evaluate", file=sys.stderr)
        return 1
    try:
        summary_path, summary = evaluate_backbones_for_run(
            args.run_path,
            backbone_ids=args.backbone_ids,
            device=args.device,
            package_root=args.package_root,
            overwrite=args.overwrite,
        )
    except BackboneDiscoveryError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(summary_path)
    for package_path in summary.packaged_pattern_paths:
        print(package_path)
    return 0


def handle_create(args: argparse.Namespace) -> int:
    partition = PartitionSpec(
        study_ids=args.study_ids,
        event_family=args.event_family,
        target_label=args.target_label,
        event_subtype=args.event_subtype,
        label_namespace=args.label_namespace,
        biological_sex=args.biological_sex,
        gender_identity=args.gender_identity,
        stimulus_modality=args.stimulus_modality,
        age_band=args.age_band,
        cohort_label=args.cohort_label,
    )
    analysis = AnalysisSpec(discovery_mode=args.discovery_mode, source_models=args.source_models, notes=args.notes)
    summary = PatternSummary(candidate_signature=args.candidate_signature, bands=args.bands, channels=args.channels, temporal_notes=args.temporal_notes)
    artifacts = [parse_artifact(item) for item in args.artifact]
    builder = PatternPackageBuilder(args.root)
    package_dir, _ = builder.create(
        pattern_id=args.pattern_id,
        partition=partition,
        analysis=analysis,
        summary=summary,
        artifacts=artifacts,
        overwrite=args.overwrite,
    )
    print(package_dir)
    return 0


def handle_validate(args: argparse.Namespace) -> int:
    return _emit_validation(validate_pattern_package_artifact(args.package_dir), as_json=args.json)


def handle_validate_collection(args: argparse.Namespace) -> int:
    return _emit_validation(validate_epoch_collection_artifact(args.collection_path), as_json=args.json)


def handle_validate_package(args: argparse.Namespace) -> int:
    return _emit_validation(validate_pattern_package_artifact(args.package_dir), as_json=args.json)


def handle_validate_run(args: argparse.Namespace) -> int:
    return _emit_validation(validate_discovery_run_artifact(args.run_path), as_json=args.json)


def handle_validate_study(args: argparse.Namespace) -> int:
    collection_paths = _load_collection_inputs(args)
    if not collection_paths:
        print("at least one --collection or --collection-list entry is required", file=sys.stderr)
        return 1
    report = validate_study(collection_paths, min_subjects=args.min_subjects)
    return _emit_validation(report, as_json=args.json)


def handle_inspect_collection(args: argparse.Namespace) -> int:
    _emit_summary(summarize_collection(args.collection_path), as_json=args.json)
    return 0


def handle_inspect_candidate(args: argparse.Namespace) -> int:
    _emit_summary(summarize_candidate(args.candidate_path), as_json=args.json)
    return 0


def handle_inspect_package(args: argparse.Namespace) -> int:
    _emit_summary(summarize_package(args.package_dir), as_json=args.json)
    return 0


def handle_inspect_run(args: argparse.Namespace) -> int:
    _emit_summary(summarize_run(args.run_path), as_json=args.json)
    return 0


def handle_models_list() -> int:
    registry = default_registry()
    for status in registry.statuses():
        runnable = "yes" if status.runnable else "no"
        deps_ready = "ok" if status.dependencies_ready else "missing"
        assets_ready = "ok" if status.common_assets_ready and status.default_variant_ready else "missing"
        print(f"{status.model_id:<10} runnable={runnable:<3} assets={assets_ready:<7} deps={deps_ready:<7} default={status.default_variant}")
    return 0


def handle_models_show(args: argparse.Namespace) -> int:
    registry = default_registry()
    status = registry.get(args.model_id).status()
    print(f"model_id: {status.model_id}")
    print(f"display_name: {status.display_name}")
    print(f"repo_root: {status.repo_root}")
    print(f"entrypoint: {status.entrypoint}")
    print(f"default_variant: {status.default_variant}")
    print(f"runnable: {'yes' if status.runnable else 'no'}")
    print(f"input_contract: {status.input_contract}")
    print(f"output_contract: {status.output_contract}")
    print("dependencies:")
    for item in status.dependencies:
        state = "installed" if item.installed else "missing"
        suffix = f" ({item.purpose})" if item.purpose else ""
        print(f"  - {item.package_name}: {state}{suffix}")
    print("common_assets:")
    for item in status.common_assets:
        state = "present" if item.exists else "missing"
        print(f"  - {item.label}: {state} -> {item.path}")
    print("variants:")
    for variant in status.variants:
        ready = "ready" if variant.ready else "missing_assets"
        print(f"  - {variant.variant_id}: {ready} ({variant.label})")
        for item in variant.assets:
            state = "present" if item.exists else "missing"
            print(f"      {item.label}: {state} -> {item.path}")
        if variant.notes:
            print(f"      notes: {variant.notes}")
    if status.notes:
        print("notes:")
        for note in status.notes:
            print(f"  - {note}")
    return 0


def handle_models_check(args: argparse.Namespace) -> int:
    registry = default_registry()
    if args.model_id:
        status = registry.get(args.model_id).status()
        if not status.runnable:
            print(f"not runnable: {status.model_id}", file=sys.stderr)
            if status.missing_dependencies:
                print("missing dependencies: " + ", ".join(status.missing_dependencies), file=sys.stderr)
            if status.missing_assets:
                print("missing assets: " + ", ".join(status.missing_assets), file=sys.stderr)
            return 1
        print(f"runnable: {status.model_id}")
        return 0
    not_ready = [status for status in registry.statuses() if not status.runnable]
    if not_ready:
        for status in not_ready:
            print(f"not runnable: {status.model_id}", file=sys.stderr)
        return 1
    print("all models runnable")
    return 0


def handle_models_probe(args: argparse.Namespace) -> int:
    registry = default_registry()
    try:
        loaded = registry.load(model_id=args.model_id, variant_id=args.variant or None, device=args.device)
    except (ModelLoadError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"loaded: {loaded.model_id}")
    print(f"variant: {loaded.variant_id}")
    print(f"device: {loaded.device}")
    for key, value in loaded.metadata.items():
        print(f"{key}: {value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        return handle_ingest(args)
    if args.command == "epoch":
        return handle_epoch(args)
    if args.command == "preprocess":
        return handle_preprocess(args)
    if args.command == "discover":
        return handle_discover(args)
    if args.command == "discover-ensemble":
        return handle_discover_ensemble(args)
    if args.command == "backbone-evaluate":
        return handle_backbone_evaluate(args)
    if args.command == "create":
        return handle_create(args)
    if args.command == "validate":
        return handle_validate(args)
    if args.command == "validate-collection":
        return handle_validate_collection(args)
    if args.command == "validate-package":
        return handle_validate_package(args)
    if args.command == "validate-run":
        return handle_validate_run(args)
    if args.command == "validate-study":
        return handle_validate_study(args)
    if args.command == "inspect-collection":
        return handle_inspect_collection(args)
    if args.command == "inspect-candidate":
        return handle_inspect_candidate(args)
    if args.command == "inspect-package":
        return handle_inspect_package(args)
    if args.command == "inspect-run":
        return handle_inspect_run(args)
    if args.command == "models":
        if args.models_command == "list":
            return handle_models_list()
        if args.models_command == "show":
            return handle_models_show(args)
        if args.models_command == "check":
            return handle_models_check(args)
        if args.models_command == "probe":
            return handle_models_probe(args)
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
