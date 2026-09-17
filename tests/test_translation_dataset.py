"""Unit tests for the Part 3 CE-CSL SLT dataset."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cslr.translation.dataset import (
    Part3DataError,
    load_part3_dataset,
    read_label_csv,
)


def _make_manifest(dir_path: Path, rows: list[tuple[str, str]]) -> Path:
    manifest = dir_path / "manifest.csv"
    lines = ["sample_id,video,label,signer,session,split"]
    for sample_id, split in rows:
        letter = sample_id.split("-")[1][:1]
        lines.append(
            f"{sample_id},video/{split}/{letter}/{sample_id}.mp4,标/签,A,1,{split}"
        )
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def _make_label(dir_path: Path, split: str, rows: list[tuple[str, str]]) -> Path:
    label_dir = dir_path / "label"
    label_dir.mkdir(parents=True, exist_ok=True)
    label = label_dir / f"{'dev' if split == 'validation' else split}.csv"
    lines = ["Number,Translator,Chinese Sentences,Gloss,Note"]
    for sample_id, sentence in rows:
        lines.append(f"{sample_id},A,{sentence},标/签,")
    label.write_text("\n".join(lines), encoding="utf-8")
    return label


class ReadLabelCsvTest(unittest.TestCase):
    def test_reads_chinese_sentences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_label(Path(tmp), "train", [("train-00001", "2023年高考到了。")])
            sources = read_label_csv(path, "train")
            self.assertEqual(sources["train-00001"].chinese_sentences, "2023年高考到了。")

    def test_duplicate_id_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "label"
            path.mkdir()
            label = path / "train.csv"
            label.write_text(
                "Number,Chinese Sentences,Gloss\n"
                "train-00001,句子一,gloss\ntrain-00001,句子二,gloss\n",
                encoding="utf-8",
            )
            with self.assertRaises(Part3DataError):
                read_label_csv(label, "train")

    def test_missing_column_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "label"
            path.mkdir()
            label = path / "train.csv"
            label.write_text("Number,Gloss\ntrain-00001,x\n", encoding="utf-8")
            with self.assertRaises(Part3DataError):
                read_label_csv(label, "train")


class LoadDatasetTest(unittest.TestCase):
    def _make_dataset(self, split: str = "train", label_split: str | None = None) -> tuple[Path, str]:
        tmp = Path(tempfile.mkdtemp())
        rows = [(f"train-{i:05d}", "train") for i in range(1, 3)] if split == "train" else [(f"dev-{i:05d}", "dev") for i in range(1, 3)]
        manifest = _make_manifest(tmp, rows)
        split_key = "dev" if split == "dev" or split == "validation" else "train"
        _make_label(tmp, split_key, [(sample_id, "一句中文。") for sample_id, _ in rows])
        return tmp, str(split_key)

    def test_load_train(self) -> None:
        tmp, label_split = self._make_dataset("train")
        cfg_like = type("cfg", (), {})()
        ds = load_part3_dataset(
            tmp / "manifest.csv", tmp / "label", split="train", root=tmp
        )
        self.assertEqual(len(ds), 2)
        item = ds[0]
        self.assertIn("sample_id", item)
        self.assertIn("chinese_sentences", item)

    def test_missing_label_id_rejected(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        _make_manifest(tmp, [(f"train-{i:05d}", "train") for i in range(1, 3)])
        # label CSV lacks one id
        _make_label(tmp, "train", [("train-00001", "句子")])
        with self.assertRaises(Part3DataError) as ctx:
            load_part3_dataset(tmp / "manifest.csv", tmp / "label", split="train", root=tmp)
        self.assertIn("missing", str(ctx.exception))

    def test_test_split_never_loads(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        _make_manifest(tmp, [("test-00001", "test")])
        with self.assertRaises(Part3DataError):
            load_part3_dataset(tmp / "manifest.csv", tmp / "label", split="test", root=tmp)


if __name__ == "__main__":
    unittest.main()