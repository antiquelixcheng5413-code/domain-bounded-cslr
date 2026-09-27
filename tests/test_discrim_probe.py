import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.discrim_probe import build_labels, encode_rows, load_gloss_tokens, run_probe


def _write_manifest(path: Path, train_ids: list[str], dev_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["sample_id,video,label,signer,session,split"]
    for sample_id in train_ids:
        lines.append(f"{sample_id},video/{sample_id}.mp4,test,a,train,train")
    for sample_id in dev_ids:
        lines.append(f"{sample_id},video/{sample_id}.mp4,test,a,dev,validation")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_labels(label_dir: Path, rows: dict[str, list[str]], name: str) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    lines = ["Number,Translator,Chinese Sentences,Gloss,Note"]
    for sample_id, tokens in rows.items():
        lines.append(f"{sample_id},A,参考句.,{'/'.join(tokens)},x")
    (label_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_features(root: Path, spec: dict[str, tuple[int, int]], seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for sample_id, (length, dim) in spec.items():
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / f"{sample_id}.npy", rng.standard_normal((length, dim)).astype(np.float32))


class LabelParsingTests(unittest.TestCase):
    def test_reads_gloss_tokens_by_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            label_dir = Path(tmp) / "label"
            _write_labels(label_dir, {"dev-00001": ["今天", "好"]}, "dev.csv")
            tokens = load_gloss_tokens(label_dir, "dev.csv")
        self.assertEqual(tokens["dev-00001"], ["今天", "好"])

    def test_missing_label_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_gloss_tokens(Path(tmp), "dev.csv")


class LabelBuildTests(unittest.TestCase):
    def test_filters_rare_glosses_and_encodes_rows(self) -> None:
        tokens = {f"t-{i:02d}": [f"gloss-{i % 3}"] for i in range(30)}
        labels, index = build_labels(tokens, min_freq=5, max_labels=50)
        self.assertEqual(labels, ["gloss-0", "gloss-1", "gloss-2"])
        rows = encode_rows(tokens, labels, index)
        self.assertTrue(np.allclose(rows["t-00"], [1.0, 0.0, 0.0]))

    def test_max_labels_truncates(self) -> None:
        tokens = {f"t-{i:02d}": [f"g-{i}"] for i in range(12)}
        labels, _ = build_labels(tokens, min_freq=1, max_labels=3)
        self.assertEqual(len(labels), 3)

    def test_shared_label_set_encodes_both_splits(self) -> None:
        train_tokens = {f"t-{i}": [f"g-{i % 3}"] for i in range(30)}
        eval_tokens = {f"e-{i}": [f"g-{i % 3}"] for i in range(10)}
        labels, index = build_labels(train_tokens, min_freq=1, max_labels=10)
        train_rows = encode_rows(train_tokens, labels, index)
        eval_rows = encode_rows(eval_tokens, labels, index)
        self.assertEqual(set(labels), {"g-0", "g-1", "g-2"})
        self.assertEqual(train_rows["t-0"].shape, (3,))
        self.assertEqual(eval_rows["e-0"].shape, (3,))


class ProbeSignalTests(unittest.TestCase):
    def _run_probe(self, feature_dim: int, signal: bool) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_ids = [f"train-{i:03d}" for i in range(40)]
            dev_ids = [f"dev-{i:03d}" for i in range(20)]
            _write_manifest(root / "manifest.csv", train_ids, dev_ids)

            all_ids = train_ids + dev_ids
            tokens = {sample_id: [f"gloss-{i % 4}"] for i, sample_id in enumerate(all_ids)}
            _write_labels(root / "label", {sid: tokens[sid] for sid in train_ids}, "train.csv")
            _write_labels(root / "label", {sid: tokens[sid] for sid in dev_ids}, "dev.csv")

            rng = np.random.default_rng(0)
            feature_root = root / "features"
            train_feature_dir = feature_root / "train"
            eval_feature_dir = feature_root / "validation"
            train_feature_dir.mkdir(parents=True, exist_ok=True)
            eval_feature_dir.mkdir(parents=True, exist_ok=True)
            for sample_id in all_ids:
                gloss = tokens[sample_id][0]
                if signal:
                    # encode the gloss index into every frame -> linearly separable
                    vector = np.zeros((8, feature_dim), dtype=np.float32)
                    vector[:, int(gloss.split("-")[1]) % 4] = 5.0
                    vector += 0.1 * rng.standard_normal((8, feature_dim)).astype(np.float32)
                else:
                    vector = rng.standard_normal((8, feature_dim)).astype(np.float32)
                if sample_id.startswith("train"):
                    np.save(train_feature_dir / f"{sample_id}.landmark.npy", vector)
                else:
                    np.save(eval_feature_dir / f"{sample_id}.landmark.npy", vector)

            args = argparse.Namespace(
                manifest=root / "manifest.csv",
                label_dir=root / "label",
                landmark_root=feature_root,
                rgb_root=None,
                motion_root=None,
                vl48_root=None,
                min_freq=1,
                max_labels=10,
                max_iter=500,
                out=root / "receipt.json",
            )
            run_probe(args)
            import json

            receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
            return receipt["results"]["landmark"]

    def test_signal_features_yield_high_auc(self) -> None:
        result = self._run_probe(feature_dim=8, signal=True)
        self.assertGreater(result["macro_auc"], 0.8)

    def test_random_features_yield_chance_auc(self) -> None:
        result = self._run_probe(feature_dim=8, signal=False)
        self.assertLess(result["macro_auc"], 0.65)


if __name__ == "__main__":
    unittest.main()
