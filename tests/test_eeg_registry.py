from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathfinder.eeg_registry import build_default_registry


class EEGRegistryTests(unittest.TestCase):
    def test_registry_contains_expected_models(self) -> None:
        registry = build_default_registry()
        self.assertEqual(registry.model_ids(), ["biot", "brainomni", "cbramod", "eegmamba", "eegpt"])

    def test_default_variants_have_local_assets(self) -> None:
        registry = build_default_registry()
        for model_id in registry.model_ids():
            status = registry.get(model_id).status()
            self.assertTrue(status.common_assets_ready, model_id)
            self.assertTrue(status.default_variant_ready, model_id)
            self.assertTrue(status.repo_root.exists(), model_id)

    def test_status_contract_is_consistent(self) -> None:
        registry = build_default_registry()
        for model_id in registry.model_ids():
            status = registry.get(model_id).status()
            self.assertEqual(status.dependencies_ready, len(status.missing_dependencies) == 0)
            self.assertEqual(status.runnable, status.dependencies_ready and status.common_assets_ready and status.default_variant_ready)


if __name__ == "__main__":
    unittest.main()

