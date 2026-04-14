from __future__ import annotations

import csv
import json
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
from pathfinder.discovery import discover_shared_patterns
from pathfinder.ingest import ingest_recording
from pathfinder.epochs import build_event_epochs
from pathfinder.preprocess import preprocess_epoch_collection
from pathfinder.validation import validate_discovery_run_artifact, validate_recording_reference_artifact, validate_study


class Phase3HardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_path = Path(tempfile.mkdtemp(prefix=".test_tmp_phase3_", dir=str(ROOT)))
        self.output_root = self.tmp_path / "out"
        self.package_root = self.tmp_path / "packages"
        self.collection_paths = self._build_fixture()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    def _write_recording(self, path: Path, *, subject_index: int) -> None:
        sampling_rate_hz = 100.0
        duration_seconds = 24.0
        samples = int(sampling_rate_hz * duration_seconds)
        time = np.arange(samples, dtype=np.float32) / sampling_rate_hz
        base = np.vstack(
            [
                0.04 * np.sin(2.0 * np.pi * 6.0 * time + subject_index * 0.1),
                0.04 * np.sin(2.0 * np.pi * 7.0 * time + subject_index * 0.1),
                0.04 * np.sin(2.0 * np.pi * 5.5 * time + subject_index * 0.1),
                0.04 * np.sin(2.0 * np.pi * 8.0 * time + subject_index * 0.1),
            ]
        ).astype(np.float32)
        noise = np.random.default_rng(subject_index).normal(0.0, 0.01, size=base.shape).astype(np.float32)
        data = base + noise
        for onset_seconds in (4.0, 12.0):
            mask = (time >= onset_seconds) & (time < onset_seconds + 2.0)
            burst_time = time[mask] - onset_seconds
            data[0, mask] += (0.7 + subject_index * 0.03) * np.sin(2.0 * np.pi * 10.0 * burst_time)
            data[1, mask] += (0.65 + subject_index * 0.03) * np.sin(2.0 * np.pi * 10.0 * burst_time)
        for onset_seconds in (8.0, 18.0):
            mask = (time >= onset_seconds) & (time < onset_seconds + 2.0)
            burst_time = time[mask] - onset_seconds
            data[2, mask] += (0.75 + subject_index * 0.02) * np.sin(2.0 * np.pi * 18.0 * burst_time)
            data[3, mask] += (0.7 + subject_index * 0.02) * np.sin(2.0 * np.pi * 18.0 * burst_time)
        np.savez_compressed(
            path,
            data=data.astype(np.float32),
            channel_names=np.asarray(["F3", "F4", "C3", "C4"]),
            sampling_rate_hz=np.asarray([sampling_rate_hz], dtype=np.float32),
            subject_id=np.asarray([f"subject_{subject_index:02d}"]),
            session_id=np.asarray(["session_a"]),
            label_namespace=np.asarray(["phase3_study"]),
        )

    def _write_events(self, path: Path) -> None:
        rows = [
            {"event_id": "strawberry_001", "onset_seconds": "4.0", "duration_seconds": "2.0", "event_family": "gustation", "target_label": "strawberry", "event_subtype": "stimulus"},
            {"event_id": "citrus_001", "onset_seconds": "8.0", "duration_seconds": "2.0", "event_family": "gustation", "target_label": "citrus", "event_subtype": "stimulus"},
            {"event_id": "strawberry_002", "onset_seconds": "12.0", "duration_seconds": "2.0", "event_family": "gustation", "target_label": "strawberry", "event_subtype": "stimulus"},
            {"event_id": "citrus_002", "onset_seconds": "18.0", "duration_seconds": "2.0", "event_family": "gustation", "target_label": "citrus", "event_subtype": "stimulus"},
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
            recording_path, _, _ = ingest_recording(source_path, output_root=self.output_root, event_table_path=event_path, overwrite=True)
            payload = json.loads(recording_path.read_text(encoding="utf-8"))
            payload["session_id"] = "session_a" if subject_index % 2 else "session_b"
            payload["metadata"] = {
                **payload.get("metadata", {}),
                "cohort_label": "cohort_red" if subject_index == 1 else "cohort_blue",
            }
            recording_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
            collection_paths.append(Path(raw_branch_path).parent / "collection.json")
        return collection_paths

    def test_discovery_writes_run_bundle_and_reliability(self) -> None:
        summary_path, summary = discover_shared_patterns(
            self.collection_paths,
            output_root=self.output_root,
            run_id="phase3_run",
            rng_seed=13,
            null_iterations=8,
            subsample_iterations=6,
            overwrite=True,
        )
        self.assertTrue(summary_path.exists())
        self.assertEqual(summary.status, "success")
        self.assertTrue(Path(summary.run_manifest_path).exists())
        self.assertTrue(Path(summary.config_snapshot_path).exists())
        self.assertTrue(Path(summary.environment_path).exists())
        self.assertTrue(Path(summary.artifact_lineage_path).exists())
        self.assertTrue(Path(summary.log_path).exists())
        self.assertTrue(summary.candidates)
        candidate = summary.candidates[0]
        self.assertIsNotNone(candidate.reliability)
        self.assertIn(candidate.reliability.confidence_tier, {"exploratory", "moderate", "strong", "unstable", "insufficient"})
        self.assertIn("reliability_json", candidate.artifact_paths)
        self.assertIn("topography_summary", candidate.artifact_paths)
        self.assertTrue((Path(candidate.candidate_root) / candidate.artifact_paths["reliability_json"]).exists())
        self.assertTrue((Path(candidate.candidate_root) / candidate.artifact_paths["topography_summary"]).exists())
        self.assertIsNotNone(candidate.reliability.session_holdout)
        self.assertIsNotNone(candidate.reliability.cohort_holdout)
        self.assertEqual(candidate.reliability.session_holdout.grouping_name, "session")
        self.assertEqual(candidate.reliability.cohort_holdout.grouping_name, "cohort")
        self.assertGreaterEqual(candidate.reliability.session_holdout.n_groups, 2)
        self.assertGreaterEqual(candidate.reliability.cohort_holdout.n_groups, 2)
        report = validate_discovery_run_artifact(summary_path)
        self.assertTrue(report.ok)

    def test_discovery_handles_bad_collection_path_as_partial_failure(self) -> None:
        bad_path = self.tmp_path / "missing_collection.json"
        summary_path, summary = discover_shared_patterns(
            self.collection_paths + [bad_path],
            output_root=self.output_root,
            run_id="phase3_partial",
            rng_seed=7,
            null_iterations=6,
            subsample_iterations=4,
            overwrite=True,
        )
        self.assertTrue(summary_path.exists())
        self.assertEqual(summary.status, "partial")
        self.assertTrue(summary.failures)
        self.assertTrue(any(item.path == str(bad_path) for item in summary.failures))

    def test_validation_and_inspection_cli_smoke(self) -> None:
        summary_path, summary = discover_shared_patterns(
            self.collection_paths,
            output_root=self.output_root,
            run_id="phase3_cli",
            package_root=self.package_root,
            rng_seed=5,
            null_iterations=6,
            subsample_iterations=4,
            overwrite=True,
        )
        self.assertEqual(main(["inspect-run", str(summary_path), "--json"]), 0)
        self.assertEqual(main(["validate-run", str(summary_path), "--json"]), 0)
        self.assertEqual(main(["inspect-collection", str(self.collection_paths[0]), "--json"]), 0)
        args = ["validate-study", "--json"]
        for collection_path in self.collection_paths:
            args.extend(["--collection", str(collection_path)])
        self.assertEqual(main(args), 0)
        self.assertTrue(summary.packaged_pattern_paths)
        self.assertEqual(main(["inspect-package", summary.packaged_pattern_paths[0], "--json"]), 0)
        self.assertEqual(main(["validate-package", summary.packaged_pattern_paths[0], "--json"]), 0)

    def test_bids_validation_warns_on_missing_sidecars(self) -> None:
        bids_dir = self.tmp_path / "bids" / "sub-01" / "ses-01" / "eeg"
        bids_dir.mkdir(parents=True, exist_ok=True)
        eeg_path = bids_dir / "sub-01_ses-01_task-taste_eeg.edf"
        eeg_path.write_bytes(b"placeholder")
        reference_path = self.tmp_path / "bids_recording.json"
        reference_path.write_text(
            json.dumps(
                {
                    "recording_id": "sub01_ses01_task_taste",
                    "subject_id": "sub-01",
                    "session_id": "ses-01",
                    "label_namespace": "bids_test",
                    "source_path": str(eeg_path),
                    "source_format": "edf",
                    "channel_names": ["F3", "F4"],
                    "sampling_rate_hz": 100.0,
                    "n_samples": 200,
                    "duration_seconds": 2.0,
                    "source_provenance": {"backend": "synthetic"},
                    "metadata": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        report = validate_recording_reference_artifact(reference_path)
        self.assertTrue(report.ok)
        self.assertTrue(any(item.code == "recording.bids_sidecar_missing" for item in report.warnings))

    def test_validate_study_flags_low_support(self) -> None:
        report = validate_study([self.collection_paths[0]], min_subjects=2)
        self.assertFalse(report.ok)
        self.assertTrue(any(item.code == "study.too_few_subjects" for item in report.errors))


if __name__ == "__main__":
    unittest.main()
