from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .eeg_registry import ModelLoadError, default_registry
from .models import AnalysisSpec, PartitionSpec, PatternSummary
from .package import ArtifactInput, PatternPackageBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pathfinder EEG pattern package tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

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
    create_parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="artifact_id|role|representation|format|source_path|description",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate an existing pattern package")
    validate_parser.add_argument("package_dir", type=Path)

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
    analysis = AnalysisSpec(
        discovery_mode=args.discovery_mode,
        source_models=args.source_models,
        notes=args.notes,
    )
    summary = PatternSummary(
        candidate_signature=args.candidate_signature,
        bands=args.bands,
        channels=args.channels,
        temporal_notes=args.temporal_notes,
    )
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
    builder = PatternPackageBuilder(args.package_dir.parent)
    errors = builder.validate_package(args.package_dir)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"valid: {args.package_dir}")
    return 0


def handle_models_list() -> int:
    registry = default_registry()
    for status in registry.statuses():
        runnable = "yes" if status.runnable else "no"
        deps_ready = "ok" if status.dependencies_ready else "missing"
        assets_ready = "ok" if status.common_assets_ready and status.default_variant_ready else "missing"
        print(
            f"{status.model_id:<10} runnable={runnable:<3} assets={assets_ready:<7} deps={deps_ready:<7} default={status.default_variant}"
        )
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
        loaded = registry.load(
            model_id=args.model_id,
            variant_id=args.variant or None,
            device=args.device,
        )
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
    if args.command == "create":
        return handle_create(args)
    if args.command == "validate":
        return handle_validate(args)
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
