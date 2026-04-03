from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .models import AnalysisSpec, ArtifactRecord, PartitionSpec, PatternManifest, PatternSummary, slugify


@dataclass(slots=True)
class ArtifactInput:
    artifact_id: str
    role: str
    representation: str
    format: str
    source_path: Path
    description: str = ""


class PatternPackageBuilder:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def create(
        self,
        pattern_id: str,
        partition: PartitionSpec,
        analysis: AnalysisSpec,
        summary: PatternSummary,
        artifacts: list[ArtifactInput],
        overwrite: bool = False,
    ) -> tuple[Path, PatternManifest]:
        package_dir = self.root / "patterns" / Path(*partition.to_path_parts()) / slugify(pattern_id)
        if package_dir.exists():
            if not overwrite:
                raise FileExistsError(f"package already exists: {package_dir}")
            shutil.rmtree(package_dir)
        package_dir.mkdir(parents=True, exist_ok=True)

        packaged_records: list[ArtifactRecord] = []
        for artifact in artifacts:
            if not artifact.source_path.exists():
                raise FileNotFoundError(f"artifact source does not exist: {artifact.source_path}")
            relative_path = self._destination_for(artifact)
            destination = package_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._copy_artifact(artifact.source_path, destination)
            packaged_records.append(
                ArtifactRecord(
                    artifact_id=artifact.artifact_id,
                    role=artifact.role,
                    representation=artifact.representation,
                    format=artifact.format.lower(),
                    path=relative_path.as_posix(),
                    description=artifact.description,
                )
            )

        manifest = PatternManifest(
            pattern_id=pattern_id,
            partition=partition,
            analysis=analysis,
            artifacts=packaged_records,
            summary=summary,
        )
        errors = manifest.validate()
        if errors:
            raise ValueError("invalid package manifest:\n- " + "\n- ".join(errors))

        self._write_manifest(package_dir, manifest)
        self._write_report(package_dir, manifest)
        return package_dir, manifest

    def validate_package(self, package_dir: Path) -> list[str]:
        package_dir = Path(package_dir)
        manifest_path = package_dir / "manifest.json"
        if not manifest_path.exists():
            return [f"missing manifest: {manifest_path}"]

        manifest = self.load_manifest(manifest_path)
        errors = manifest.validate()
        for artifact in manifest.artifacts:
            artifact_path = package_dir / artifact.path
            if not artifact_path.exists():
                errors.append(f"missing artifact path: {artifact.path}")
        return errors

    @staticmethod
    def load_manifest(path: Path) -> PatternManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return PatternManifest.from_dict(data)

    @staticmethod
    def _copy_artifact(source: Path, destination: Path) -> None:
        if source.is_dir():
            shutil.copytree(source, destination)
            return
        shutil.copy2(source, destination)

    @staticmethod
    def _destination_for(artifact: ArtifactInput) -> Path:
        extension = artifact.format.lower()
        if artifact.representation == "raw_eeg":
            base_dir = Path("signals") / "raw"
        elif artifact.representation == "processed_epoch":
            base_dir = Path("signals") / "processed"
        elif artifact.representation in {"time_frequency", "topography", "connectivity", "embedding"}:
            base_dir = Path("derived")
        elif artifact.representation in {"report", "figure"}:
            base_dir = Path("reports")
        else:
            base_dir = Path("support")

        safe_name = slugify(artifact.role)
        if artifact.source_path.is_dir() or extension == "zarr":
            file_name = safe_name
        else:
            file_name = f"{safe_name}.{extension}"
        return base_dir / file_name

    @staticmethod
    def _write_manifest(package_dir: Path, manifest: PatternManifest) -> None:
        manifest_path = package_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def _write_report(package_dir: Path, manifest: PatternManifest) -> None:
        lines = [
            f"# Pattern Report: {manifest.pattern_id}",
            "",
            "## Partition",
            f"- Event family: {manifest.partition.event_family or 'unspecified'}",
            f"- Target label: {manifest.partition.target_label or 'unspecified'}",
            f"- Event subtype: {manifest.partition.event_subtype or 'unspecified'}",
            f"- Label namespace: {manifest.partition.label_namespace or 'unspecified'}",
            f"- Biological sex: {manifest.partition.biological_sex or 'unspecified'}",
            f"- Gender identity: {manifest.partition.gender_identity or 'unspecified'}",
            f"- Stimulus modality: {manifest.partition.stimulus_modality or 'unspecified'}",
            "",
            "## Summary",
            f"- Candidate signature: {manifest.summary.candidate_signature or 'not provided'}",
            f"- Bands: {', '.join(manifest.summary.bands) if manifest.summary.bands else 'not provided'}",
            f"- Channels: {', '.join(manifest.summary.channels) if manifest.summary.channels else 'not provided'}",
            f"- Temporal notes: {manifest.summary.temporal_notes or 'not provided'}",
            "",
            "## Source Models",
        ]
        lines.extend(f"- {model}" for model in manifest.analysis.source_models)
        lines.extend(["", "## Artifacts"])
        lines.extend(
            f"- {artifact.artifact_id}: {artifact.role} -> {artifact.path}" for artifact in manifest.artifacts
        )
        (package_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
