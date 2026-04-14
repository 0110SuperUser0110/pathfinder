from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathfinder.analysis_models import BaselineWindowSpec, EpochWindowConfig, PreprocessBranchConfig
from pathfinder.backbone_discovery import (
    PreparedBackboneInput,
    _prepare_brainomni,
    _prepare_eegpt,
    evaluate_backbones_for_run,
)
from pathfinder.cli import main
from pathfinder.discovery import discover_shared_patterns, load_discovery_run_summary
from pathfinder.eeg_registry import LoadedModel
from pathfinder.epochs import build_event_epochs
from pathfinder.ingest import ingest_recording
from pathfinder.package import PatternPackageBuilder
from pathfinder.preprocess import preprocess_epoch_collection


class _FakeRegistry:
    def load(self, model_id: str, variant_id: str | None = None, device: str = "cpu") -> LoadedModel:
        return LoadedModel(
            model_id=model_id,
            display_name=model_id.upper(),
            variant_id=variant_id or "mock_variant",
            device=device,
            model=object(),
            metadata={"mock": True},
        )


def _mock_prepare(candidate):
    subject_path = Path(candidate.candidate_root) / candidate.artifact_paths["subject_prototypes"]
    with np.load(subject_path, allow_pickle=False) as payload:
        phase_name = "onset" if "onset_stack" in payload else payload["phase_names"].tolist()[0]
        key = f"{phase_name}_stack"
        stack = np.asarray(payload[key], dtype=np.float32)
        channel_names = [str(item) for item in payload["channel_names"].tolist()]
    return PreparedBackboneInput(
        model_id="mockbone",
        prepared=stack,
        prepared_channel_names=channel_names,
        target_sampling_rate_hz=100.0,
        selected_phases=[phase_name],
        notes=[],
    )


def _mock_extract_embeddings(_loaded_model, prepared: PreparedBackboneInput) -> np.ndarray:
    data = np.asarray(prepared.prepared, dtype=np.float32)
    channel_means = np.mean(data, axis=-1)
    channel_stds = np.std(data, axis=-1)
    return np.concatenate([channel_means, channel_stds], axis=1).astype(np.float32)


class BackboneDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_path = Path(tempfile.mkdtemp(prefix=".test_tmp_backbone_", dir=str(ROOT)))
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
                0.05 * np.sin(2.0 * np.pi * 6.0 * time + subject_index * 0.1),
                0.05 * np.sin(2.0 * np.pi * 6.5 * time + subject_index * 0.1),
                0.05 * np.sin(2.0 * np.pi * 12.0 * time + subject_index * 0.1),
                0.05 * np.sin(2.0 * np.pi * 12.5 * time + subject_index * 0.1),
            ]
        ).astype(np.float32)
        noise = np.random.default_rng(subject_index).normal(0.0, 0.01, size=base.shape).astype(np.float32)
        data = base + noise
        for onset_seconds in (4.0, 12.0):
            mask = (time >= onset_seconds) & (time < onset_seconds + 2.0)
            burst_time = time[mask] - onset_seconds
            data[0, mask] += (0.8 + subject_index * 0.03) * np.sin(2.0 * np.pi * 10.0 * burst_time)
            data[1, mask] += (0.75 + subject_index * 0.03) * np.sin(2.0 * np.pi * 10.0 * burst_time)
        for onset_seconds in (8.0, 18.0):
            mask = (time >= onset_seconds) & (time < onset_seconds + 2.0)
            burst_time = time[mask] - onset_seconds
            data[2, mask] += (0.85 + subject_index * 0.02) * np.sin(2.0 * np.pi * 18.0 * burst_time)
            data[3, mask] += (0.8 + subject_index * 0.02) * np.sin(2.0 * np.pi * 18.0 * burst_time)
        np.savez_compressed(
            path,
            data=data.astype(np.float32),
            channel_names=np.asarray(["F3", "F4", "C3", "C4"]),
            sampling_rate_hz=np.asarray([sampling_rate_hz], dtype=np.float32),
            subject_id=np.asarray([f"subject_{subject_index:02d}"]),
            session_id=np.asarray(["session_a"]),
            label_namespace=np.asarray(["backbone_study"]),
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

    def _candidate(self, summary, target_label: str, branch_name: str):
        for candidate in summary.candidates:
            if candidate.target_label == target_label and candidate.branch_name == branch_name:
                return candidate
        raise AssertionError(f"candidate not found for {target_label} on {branch_name}")

    def test_backbone_evaluate_run_writes_evidence_and_consensus(self) -> None:
        summary_path, _ = discover_shared_patterns(
            self.collection_paths,
            output_root=self.output_root,
            run_id="backbone_base",
            overwrite=True,
        )
        with patch("pathfinder.backbone_discovery.default_registry", return_value=_FakeRegistry()), patch.dict(
            "pathfinder.backbone_discovery.SUPPORTED_PREPARERS",
            {"mockbone": _mock_prepare},
            clear=False,
        ), patch("pathfinder.backbone_discovery._extract_embeddings", side_effect=_mock_extract_embeddings):
            summary_path, summary = evaluate_backbones_for_run(
                summary_path,
                backbone_ids=["mockbone"],
                package_root=self.package_root,
                overwrite=True,
            )
        self.assertTrue(summary_path.exists())
        self.assertIsNotNone(summary.backbone_evaluation)
        self.assertEqual(summary.backbone_evaluation.completed_model_ids, ["mockbone"])
        self.assertTrue(summary.packaged_pattern_paths)
        candidate = summary.candidates[0]
        self.assertEqual(len(candidate.backbone_evidence), 1)
        self.assertEqual(candidate.backbone_evidence[0].model_id, "mockbone")
        self.assertEqual(candidate.backbone_evidence[0].status, "used")
        self.assertIn("backbone_consensus_json", candidate.artifact_paths)
        self.assertTrue((Path(candidate.candidate_root) / candidate.artifact_paths["backbone_consensus_json"]).exists())
        builder = PatternPackageBuilder(self.package_root)
        for package_path in summary.packaged_pattern_paths:
            self.assertEqual(builder.validate_package(Path(package_path)), [])

    def test_discover_ensemble_cli_smoke(self) -> None:
        args = [
            "discover-ensemble",
            "--output-root",
            str(self.output_root),
            "--run-id",
            "ensemble_cli",
            "--backbone",
            "mockbone",
            "--package-root",
            str(self.package_root),
            "--overwrite",
        ]
        for collection_path in self.collection_paths:
            args.extend(["--collection", str(collection_path)])
        with patch("pathfinder.backbone_discovery.default_registry", return_value=_FakeRegistry()), patch.dict(
            "pathfinder.backbone_discovery.SUPPORTED_PREPARERS",
            {"mockbone": _mock_prepare},
            clear=False,
        ), patch("pathfinder.backbone_discovery._extract_embeddings", side_effect=_mock_extract_embeddings):
            exit_code = main(args)
        self.assertEqual(exit_code, 0)
        summary = load_discovery_run_summary(self.output_root / "discovery_runs" / "ensemble_cli" / "run_summary.json")
        self.assertIsNotNone(summary.backbone_evaluation)
        self.assertEqual(summary.backbone_ids, ["mockbone"])
        self.assertTrue(summary.packaged_pattern_paths)

    def test_prepare_eegpt_uses_canonical_58_channel_layout(self) -> None:
        _, summary = discover_shared_patterns(
            self.collection_paths,
            output_root=self.output_root,
            run_id="prep_eegpt",
            overwrite=True,
        )
        candidate = self._candidate(summary, "strawberry", "raw_preserving")
        prepared = _prepare_eegpt(candidate)
        self.assertEqual(prepared.model_id, "eegpt")
        self.assertEqual(prepared.prepared.shape[1], 58)
        self.assertEqual(prepared.prepared.shape[-1], 512)
        self.assertEqual(len(prepared.prepared_channel_names), 58)
        self.assertTrue(any("zero-filled" in note for note in prepared.notes))

    def test_prepare_brainomni_infers_sensor_positions(self) -> None:
        _, summary = discover_shared_patterns(
            self.collection_paths,
            output_root=self.output_root,
            run_id="prep_brainomni",
            overwrite=True,
        )
        candidate = self._candidate(summary, "strawberry", "raw_preserving")
        prepared = _prepare_brainomni(candidate)
        self.assertEqual(prepared.model_id, "brainomni")
        self.assertEqual(prepared.prepared.shape[1], 4)
        self.assertEqual(prepared.extras["pos"].shape, (3, 4, 6))
        self.assertEqual(prepared.extras["sensor_type"].shape, (3, 4))
        self.assertTrue(np.all(prepared.extras["sensor_type"] == 0))
        self.assertTrue(np.allclose(prepared.extras["pos"][..., 3:], 0.0))
        self.assertGreater(float(np.abs(prepared.extras["pos"][..., :3]).sum()), 0.0)

    def test_backbone_consensus_updates_reliability(self) -> None:
        summary_path, _ = discover_shared_patterns(
            self.collection_paths,
            output_root=self.output_root,
            run_id="backbone_reliability",
            overwrite=True,
        )
        with patch("pathfinder.backbone_discovery.default_registry", return_value=_FakeRegistry()), patch.dict(
            "pathfinder.backbone_discovery.SUPPORTED_PREPARERS",
            {"mockbone_a": _mock_prepare, "mockbone_b": _mock_prepare},
            clear=False,
        ), patch("pathfinder.backbone_discovery._extract_embeddings", side_effect=_mock_extract_embeddings):
            _, summary = evaluate_backbones_for_run(
                summary_path,
                backbone_ids=["mockbone_a", "mockbone_b"],
                overwrite=True,
            )
        candidate = self._candidate(summary, "strawberry", "raw_preserving")
        self.assertIsNotNone(candidate.reliability)
        self.assertEqual(candidate.backbone_consensus.overall_status, "strong_agreement")
        self.assertEqual(candidate.reliability.backbone_stability, "strong_agreement")
        self.assertIsNotNone(candidate.reliability.backbone_support_score)
        self.assertEqual(sorted(candidate.reliability.supporting_backbone_ids), ["mockbone_a", "mockbone_b"])


if __name__ == "__main__":
    unittest.main()
