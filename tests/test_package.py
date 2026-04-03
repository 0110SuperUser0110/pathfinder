from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathfinder.models import AnalysisSpec, PartitionSpec, PatternSummary
from pathfinder.package import ArtifactInput, PatternPackageBuilder


class PatternPackageBuilderTests(unittest.TestCase):
    def test_create_and_validate_package(self) -> None:
        tmp_path = ROOT / ".test_tmp"
        shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            signal_file = tmp_path / "prototype_epoch.fif"
            signal_file.write_bytes(b"signal")

            zarr_dir = tmp_path / "segments.zarr"
            zarr_dir.mkdir()
            (zarr_dir / ".zarray").write_text("{}", encoding="utf-8")

            builder = PatternPackageBuilder(tmp_path / "out")
            package_dir, manifest = builder.create(
                pattern_id="pattern_003",
                partition=PartitionSpec(
                    study_ids=["study_alpha"],
                    event_family="affect",
                    target_label="positive_valence",
                    event_subtype="self_report",
                    biological_sex="female",
                    stimulus_modality="self_report",
                ),
                analysis=AnalysisSpec(
                    discovery_mode="stratified",
                    source_models=["EEGPT", "BrainOmni", "BIOT", "CBraMod"],
                ),
                summary=PatternSummary(
                    candidate_signature="alpha suppression with beta increase",
                    bands=["alpha", "beta"],
                    channels=["F3", "F4"],
                ),
                artifacts=[
                    ArtifactInput(
                        artifact_id="proto",
                        role="prototype_epoch",
                        representation="processed_epoch",
                        format="fif",
                        source_path=signal_file,
                    ),
                    ArtifactInput(
                        artifact_id="segments",
                        role="exemplar_segments",
                        representation="processed_epoch",
                        format="zarr",
                        source_path=zarr_dir,
                    ),
                ],
            )

            self.assertEqual(manifest.pattern_id, "pattern_003")
            self.assertTrue((package_dir / "manifest.json").exists())
            self.assertTrue((package_dir / "report.md").exists())

            errors = builder.validate_package(package_dir)
            self.assertEqual(errors, [])
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
