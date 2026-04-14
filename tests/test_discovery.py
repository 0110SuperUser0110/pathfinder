from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathfinder.analysis_models import BaselineWindowSpec, EpochWindowConfig, PreprocessBranchConfig
from pathfinder.cli import main
from pathfinder.discovery import discover_shared_patterns, load_discovery_run_summary
from pathfinder.epochs import build_event_epochs
from pathfinder.ingest import ingest_recording
from pathfinder.package import PatternPackageBuilder
from pathfinder.preprocess import preprocess_epoch_collection


class DiscoveryEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_path = Path(tempfile.mkdtemp(prefix=".test_tmp_discovery_", dir=str(ROOT)))
        self.output_root = self.tmp_path / "out"
        self.package_root = self.tmp_path / "packages"
        self.collection_paths = self._build_fixture()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    def _write_recording(self, path: Path, *, subject_index: int) -> None:
        sampling_rate_hz = 100.0
        duration_seconds = 30.0
        samples = int(sampling_rate_hz * duration_seconds)
        time = np.arange(samples, dtype=np.float32) / sampling_rate_hz
        phase_shift = subject_index * 0.17
        base = np.vstack(
            [
                0.08 * np.sin(2.0 * np.pi * 6.0 * time + phase_shift),
                0.08 * np.sin(2.0 * np.pi * 6.5 * time + phase_shift),
                0.08 * np.sin(2.0 * np.pi * 5.5 * time + phase_shift),
                0.08 * np.sin(2.0 * np.pi * 7.0 * time + phase_shift),
            ]
        ).astype(np.float32)
        noise = np.random.default_rng(subject_index).normal(0.0, 0.015, size=base.shape).astype(np.float32)
        data = base + noise

        strawberry_events = [(5.0, 2.0), (15.0, 2.0)]
        citrus_events = [(10.0, 2.0), (22.0, 2.0)]
        for onset_seconds, duration in strawberry_events:
            mask = (time >= onset_seconds) & (time < onset_seconds + duration)
            burst_time = time[mask] - onset_seconds
            data[0, mask] += (0.9 + subject_index * 0.04) * np.sin(2.0 * np.pi * 10.0 * burst_time)
            data[1, mask] += (0.85 + subject_index * 0.04) * np.sin(2.0 * np.pi * 10.0 * burst_time)
            data[2, mask] += 0.12 * np.sin(2.0 * np.pi * 4.0 * burst_time)
        for onset_seconds, duration in citrus_events:
            mask = (time >= onset_seconds) & (time < onset_seconds + duration)
            burst_time = time[mask] - onset_seconds
            data[2, mask] += (0.95 + subject_index * 0.03) * np.sin(2.0 * np.pi * 20.0 * burst_time)
            data[3, mask] += (0.9 + subject_index * 0.03) * np.sin(2.0 * np.pi * 20.0 * burst_time)
            data[0, mask] += 0.08 * np.sin(2.0 * np.pi * 5.0 * burst_time)

        np.savez_compressed(
            path,
            data=data.astype(np.float32),
            channel_names=np.asarray(["F3", "F4", "C3", "C4"]),
            sampling_rate_hz=np.asarray([sampling_rate_hz], dtype=np.float32),
            subject_id=np.asarray([f"subject_{subject_index:02d}"]),
            session_id=np.asarray(["session_a"]),
            label_namespace=np.asarray(["fruit_study_v1"]),
        )

    def _write_events(self, path: Path) -> None:
        rows = [
            {"event_id": "strawberry_001", "onset_seconds": "5.0", "duration_seconds": "2.0", "event_family": "gustation", "target_label": "strawberry", "event_subtype": "stimulus"},
            {"event_id": "citrus_001", "onset_seconds": "10.0", "duration_seconds": "2.0", "event_family": "gustation", "target_label": "citrus", "event_subtype": "stimulus"},
            {"event_id": "strawberry_002", "onset_seconds": "15.0", "duration_seconds": "2.0", "event_family": "gustation", "target_label": "strawberry", "event_subtype": "stimulus"},
            {"event_id": "citrus_002", "onset_seconds": "22.0", "duration_seconds": "2.0", "event_family": "gustation", "target_label": "citrus", "event_subtype": "stimulus"},
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _build_fixture(self) -> list[Path]:
        collection_paths: list[Path] = []
        for subject_index in range(1, 4):
            source_path = self.tmp_path / f"subject_{subject_index:02d}.npz"
            event_path = self.tmp_path / f"subject_{subject_index:02d}_events.csv"
            self._write_recording(source_path, subject_index=subject_index)
            self._write_events(event_path)
            recording_path, _, _ = ingest_recording(
                source_path,
                output_root=self.output_root,
                event_table_path=event_path,
                overwrite=True,
            )
            collection_path, _ = build_event_epochs(
                recording_path,
                recording_path.parent / "events.json",
                output_root=self.output_root,
                window_config=EpochWindowConfig(
                    pre_event_seconds=1.0,
                    onset_seconds=1.0,
                    offset_seconds=0.5,
                    post_event_seconds=1.0,
                    baseline_window=BaselineWindowSpec(start_offset_seconds=-1.0, end_offset_seconds=0.0),
                ),
                overwrite=True,
            )
            raw_branch_path, _ = preprocess_epoch_collection(
                collection_path,
                output_root=self.output_root,
                config=PreprocessBranchConfig(branch_name="raw_preserving"),
                overwrite=True,
            )
            light_branch_path, _ = preprocess_epoch_collection(
                collection_path,
                output_root=self.output_root,
                config=PreprocessBranchConfig(
                    branch_name="light_clean",
                    notch_hz=[60.0],
                    resample_hz=80.0,
                    baseline_mode="metadata_only",
                ),
                overwrite=True,
            )
            comparison_branch_path, _ = preprocess_epoch_collection(
                collection_path,
                output_root=self.output_root,
                config=PreprocessBranchConfig(
                    branch_name="comparison_safe",
                    align_channels=["F3", "F4", "C3", "C4"],
                    rereference_mode="average",
                    scale_factor=1.0,
                    baseline_mode="subtract_mean",
                ),
                overwrite=True,
            )
            collection_paths.extend(
                [
                    Path(raw_branch_path).parent / "collection.json",
                    Path(light_branch_path).parent / "collection.json",
                    Path(comparison_branch_path).parent / "collection.json",
                ]
            )
        return collection_paths

    def _candidate(self, summary, target_label: str, branch_name: str):
        for candidate in summary.candidates:
            if candidate.target_label == target_label and candidate.branch_name == branch_name:
                return candidate
        raise AssertionError(f"candidate not found for {target_label} on {branch_name}")

    def test_discovery_groups_labels_and_negative_controls(self) -> None:
        summary_path, summary = discover_shared_patterns(
            self.collection_paths,
            output_root=self.output_root,
            run_id="fruit_discovery",
            min_subjects=3,
            overwrite=True,
        )
        self.assertTrue(summary_path.exists())
        strawberry = self._candidate(summary, "strawberry", "raw_preserving")
        self.assertEqual(strawberry.cross_subject_agreement.n_subjects, 3)
        self.assertEqual(strawberry.cross_subject_agreement.n_events, 6)
        self.assertIsNotNone(strawberry.cross_subject_agreement.within_label_similarity)
        self.assertIsNotNone(strawberry.control_summary.target_vs_rest_similarity)
        self.assertGreater(
            strawberry.cross_subject_agreement.within_label_similarity,
            strawberry.control_summary.target_vs_rest_similarity,
        )
        self.assertIn("alpha", strawberry.dominant_bands)
        self.assertTrue(strawberry.strongest_phases)

    def test_discovery_artifacts_and_branch_agreement(self) -> None:
        _, summary = discover_shared_patterns(
            self.collection_paths,
            output_root=self.output_root,
            run_id="fruit_branch_discovery",
            min_subjects=3,
            overwrite=True,
        )
        strawberry = self._candidate(summary, "strawberry", "comparison_safe")
        self.assertIsNotNone(strawberry.branch_agreement)
        self.assertIn(strawberry.branch_agreement.overall_status, {"preserved", "weakened", "shifted"})
        candidate_root = Path(strawberry.candidate_root)
        prototype_path = candidate_root / strawberry.artifact_paths["prototype_epoch"]
        exemplar_path = candidate_root / strawberry.artifact_paths["exemplar_epochs"]
        self.assertTrue(prototype_path.exists())
        self.assertTrue(exemplar_path.exists())
        with np.load(prototype_path, allow_pickle=False) as payload:
            self.assertIn("onset", payload.files)
            self.assertIn("sustained", payload.files)
            self.assertEqual(payload["onset"].shape[0], 4)
        with np.load(exemplar_path, allow_pickle=False) as payload:
            self.assertGreaterEqual(payload["onset_stack"].shape[0], 3)
            self.assertLessEqual(payload["onset_stack"].shape[0], 6)

    def test_package_integration_and_cli_smoke(self) -> None:
        args = [
            "discover",
            "--output-root",
            str(self.output_root),
            "--run-id",
            "cli_discovery",
            "--min-subjects",
            "3",
            "--package-root",
            str(self.package_root),
            "--backbone",
            "biot",
            "--backbone",
            "brainomni",
            "--overwrite",
        ]
        for collection_path in self.collection_paths:
            args.extend(["--collection", str(collection_path)])
        exit_code = main(args)
        self.assertEqual(exit_code, 0)
        summary_path = self.output_root / "discovery_runs" / "cli_discovery" / "run_summary.json"
        self.assertTrue(summary_path.exists())
        summary = load_discovery_run_summary(summary_path)
        self.assertTrue(summary.packaged_pattern_paths)
        self.assertEqual(summary.backbone_ids, ["biot", "brainomni"])
        builder = PatternPackageBuilder(self.package_root)
        for package_path in summary.packaged_pattern_paths:
            self.assertEqual(builder.validate_package(Path(package_path)), [])
            manifest = builder.load_manifest(Path(package_path) / "manifest.json")
            artifact_roles = {artifact.role for artifact in manifest.artifacts}
            self.assertIn("run_manifest", artifact_roles)
            self.assertIn("config_snapshot", artifact_roles)


if __name__ == "__main__":
    unittest.main()
