"""Unit tests for Part 3 config validation, incl. the frozen test split."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cslr.translation.config import Part3ConfigError, load_config


def _write_config(path: Path, split: str) -> Path:
    body = {
        "root": ".",
        "split": split,
        "data": {
            "manifest": "manifests/ce-csl.csv",
            "label_dir": "raw/CE-CSL/label",
        },
        "model": {
            "hidden_dim": 32,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.1,
            "gloss_aux_weight": 0.0,
            "decoder": {"num_layers": 2, "num_heads": 4},
        },
        "fusion": {"modalities": ["rgb", "motion", "landmark"], "num_heads": 4, "num_layers": 2},
        "smoke": {"train_steps": 15, "dev_limit": 3, "seed": 0},
    }
    body["split"] = split
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


class ConfigValidationTest(unittest.TestCase):
    def _cfg(self, split: str = "train") -> Path:
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / "cfg.yaml"
        return _write_config(path, split)

    def test_train_accepted(self) -> None:
        cfg = load_config(self._cfg("train"))
        self.assertEqual(cfg.effective_split(), "train")

    def test_dev_alias_normalised(self) -> None:
        cfg = load_config(self._cfg("dev"))
        self.assertEqual(cfg.effective_split(), "validation")

    def test_test_split_rejected_before_file_access(self) -> None:
        # The rejection happens at config parse, in a directory without any data.
        tmp = Path(tempfile.mkdtemp())
        path = _write_config(tmp / "cfg.yaml", "test")
        with self.assertRaises(Part3ConfigError) as ctx:
            load_config(path)
        self.assertIn("frozen", str(ctx.exception))

    def test_missing_manifest(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        cfg_path = _write_config(tmp / "cfg.yaml", "train")
        # break manifest
        body = json.loads(cfg_path.read_text(encoding="utf-8"))
        del body["data"]["manifest"]
        cfg_path.write_text(json.dumps(body), encoding="utf-8")
        with self.assertRaises(Part3ConfigError):
            load_config(cfg_path)

    def test_empty_modalities_rejected(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        cfg_path = _write_config(tmp / "cfg.yaml", "train")
        body = json.loads(cfg_path.read_text(encoding="utf-8"))
        body["fusion"]["modalities"] = []
        cfg_path.write_text(json.dumps(body), encoding="utf-8")
        with self.assertRaises(Part3ConfigError):
            load_config(cfg_path)


if __name__ == "__main__":
    unittest.main()