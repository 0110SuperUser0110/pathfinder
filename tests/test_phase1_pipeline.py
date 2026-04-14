from __future__ import annotations

import csv
import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathfinder.analysis_models import BaselineWindowSpec, EpochWindowConfig, PreprocessBranchConfig
from pathfinder.cli import main
from pathfinder.epochs import build_event_epochs, load_epoch_collection
from pathfinder.ingest import ingest_recording, load_normalized_event_table, load_recording_reference
from pathfinder.preprocess import load_branch_result, preprocess_epoch_collection


class Phase1PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_path = ROOT / ".test_tmp"
        shutil.rmtree(self.tmp_path, ignore_errors=True)
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        self.source_path = self.tmp_path / "recording_alpha.npz"
        self.events_path = self.tmp_path / "events.csv"
        self._write_recording_fixture()
        self._write_event_fixture()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    def _write_recording_fixture(self) -> None:
        sampling_rate_hz = 100.0
        duration_seconds = 20.0
        samples = int(sampling_rate_hz * duration_seconds)
        time = np.arange(samples, dtype=np.float32) / sampling_rate_hz
        data = np.vstack(
            [
                np.sin(2.0 * np.pi * 8.0 * time),
                np.sin(2.0 * np.pi * 10.0 * time),
                np.sin(2.0 * np.pi * 12.0 * time),
                np.sin(2.0 * np.pi * 15.0 * time),
            ]
        ).astype(np.float32)
        np.savez_compressed(
            self.source_path,
            data=data,
            channel_names=np.asarray(["F3", "F4", "C3", "C4"]),
            sampling_rate_hz=np.asarray([sampling_rate_hz], dtype=np.float32),
            subject_id=np.asarray(["subject_01"]),
            session_id=np.asarray(["session_a"]),
            label_namespace=np.asarray(["study_v1"]),
        )

    def _write_event_fixture(self) -> None:
        rows = [
            {
                "event_id": "event_001",
                "onset_seconds": "5.0",
                "duration_seconds": "4.0",
                "event_family": "affect",
                "target_label": "happy_candidate",
                "event_subtype": "self_report",
            },
            {
                "event_id": "event_002",
                "onset_seconds": "12.0",
                "duration_seconds": "3.0",
                "event_family": "affect",
                "target_label": "calm_candidate",
                "event_subtype": "self_report",
            },
        ]
        with self.events_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def test_ingest_recording_writes_reference_and_events(self) -> None:
        recording_path, recording, events = ingest_recording(
            self.source_path,
            output_root=self.tmp_path / "out",
            event_table_path=self.events_path,
        )
        self.assertTrue(recording_path.exists())
        self.assertEqual(recording.recording_id, "recording_alpha")
        self.assertEqual(recording.subject_id, "subject_01")
        self.assertEqual(recording.session_id, "session_a")
        self.assertEqual(recording.label_namespace, "study_v1")
        self.assertEqual(recording.channel_names, ["F3", "F4", "C3", "C4"])
        self.assertEqual(len(events), 2)
        persisted = load_recording_reference(recording_path)
        self.assertEqual(persisted.source_path, str(self.source_path.resolve()))
        normalized_events = load_normalized_event_table(recording_path.parent / "events.json")
        self.assertEqual([event.event_id for event in normalized_events], ["event_001", "event_002"])

    def test_build_event_epochs_creates_signal_artifacts(self) -> None:
        recording_path, _, _ = ingest_recording(
            self.source_path,
            output_root=self.tmp_path / "out",
            event_table_path=self.events_path,
        )
        collection_path, collection = build_event_epochs(
            recording_path,
            recording_path.parent / "events.json",
            output_root=self.tmp_path / "out",
            window_config=EpochWindowConfig(
                pre_event_seconds=1.5,
                onset_seconds=1.0,
                offset_seconds=1.0,
                post_event_seconds=1.5,
                baseline_window=BaselineWindowSpec(start_offset_seconds=-1.0, end_offset_seconds=0.0),
            ),
        )
        self.assertTrue(collection_path.exists())
        self.assertEqual(len(collection.artifacts), 2)
        event_artifact = collection.artifacts[0]
        event_file = collection_path.parent / event_artifact.signal_path
        self.assertTrue(event_file.exists())
        with np.load(event_file, allow_pickle=False) as payload:
            self.assertIn("pre_event", payload.files)
            self.assertIn("onset", payload.files)
            self.assertIn("sustained", payload.files)
            self.assertIn("offset", payload.files)
            self.assertIn("post_event", payload.files)
            self.assertIn("baseline", payload.files)
            self.assertEqual(payload["pre_event"].shape[0], 4)
        reloaded = load_epoch_collection(collection_path)
        self.assertEqual(reloaded.recording.recording_id, "recording_alpha")
        self.assertEqual(reloaded.channel_names, ["F3", "F4", "C3", "C4"])

    def test_preprocess_branch_tracks_provenance_and_channel_alignment(self) -> None:
        recording_path, _, _ = ingest_recording(
            self.source_path,
            output_root=self.tmp_path / "out",
            event_table_path=self.events_path,
        )
        collection_path, _ = build_event_epochs(
            recording_path,
            recording_path.parent / "events.json",
            output_root=self.tmp_path / "out",
            window_config=EpochWindowConfig(
                pre_event_seconds=1.0,
                onset_seconds=1.0,
                offset_seconds=1.0,
                post_event_seconds=1.0,
                baseline_window=BaselineWindowSpec(start_offset_seconds=-1.0, end_offset_seconds=0.0),
            ),
        )
        branch_path, branch_result = preprocess_epoch_collection(
            collection_path,
            output_root=self.tmp_path / "out",
            config=PreprocessBranchConfig(
                branch_name="comparison_safe",
                notch_hz=[60.0],
                resample_hz=50.0,
                baseline_mode="subtract_mean",
                align_channels=["F3", "F4", "C3", "CZ"],
                rereference_mode="average",
                scale_factor=0.5,
            ),
        )
        self.assertTrue(branch_path.exists())
        self.assertIsNotNone(branch_result.channel_report)
        self.assertIn("CZ", branch_result.channel_report.missing_channels)
        transform_names = [item.name for item in branch_result.transforms]
        self.assertIn("align_channels", transform_names)
        self.assertIn("notch_filter", transform_names)
        self.assertIn("resample", transform_names)
        self.assertIn("baseline", transform_names)
        branch_manifest = load_branch_result(branch_path)
        self.assertEqual(branch_manifest.branch_name, "comparison_safe")
        output_collection = load_epoch_collection(branch_manifest.output_collection_path)
        self.assertEqual(output_collection.channel_names, ["F3", "F4", "C3", "CZ"])
        self.assertEqual(output_collection.sampling_rate_hz, 50.0)
        event_file = Path(branch_manifest.output_collection_path).parent / output_collection.artifacts[0].signal_path
        with np.load(event_file, allow_pickle=False) as payload:
            self.assertEqual(payload["pre_event"].shape[0], 4)
            self.assertLess(payload["pre_event"].shape[1], 100)

    def test_cli_smoke_workflow(self) -> None:
        out_root = self.tmp_path / "cli_out"
        exit_code = main(
            [
                "ingest",
                str(self.source_path),
                "--output-root",
                str(out_root),
                "--event-table",
                str(self.events_path),
            ]
        )
        self.assertEqual(exit_code, 0)
        recording_path = out_root / "recordings" / "recording_alpha" / "recording.json"
        events_path = out_root / "recordings" / "recording_alpha" / "events.json"
        self.assertTrue(recording_path.exists())
        self.assertTrue(events_path.exists())

        exit_code = main(
            [
                "epoch",
                "--recording",
                str(recording_path),
                "--events",
                str(events_path),
                "--output-root",
                str(out_root),
                "--pre-event-seconds",
                "1.0",
                "--onset-seconds",
                "1.0",
                "--offset-seconds",
                "1.0",
                "--post-event-seconds",
                "1.0",
                "--baseline-start-offset",
                "-1.0",
                "--baseline-end-offset",
                "0.0",
            ]
        )
        self.assertEqual(exit_code, 0)
        epoch_root = out_root / "epochs" / "recording_alpha"
        collection_path = next(epoch_root.rglob("collection.json"))
        self.assertTrue(collection_path.exists())

        exit_code = main(
            [
                "preprocess",
                "--collection",
                str(collection_path),
                "--output-root",
                str(out_root),
                "--branch",
                "light_clean",
                "--notch-hz",
                "60",
                "--resample-hz",
                "50",
                "--baseline-mode",
                "metadata_only",
            ]
        )
        self.assertEqual(exit_code, 0)
        branch_path = next((out_root / "preprocess").rglob("branch.json"))
        self.assertTrue(branch_path.exists())
        branch_payload = json.loads(branch_path.read_text(encoding="utf-8"))
        self.assertEqual(branch_payload["branch_name"], "light_clean")


if __name__ == "__main__":
    unittest.main()
