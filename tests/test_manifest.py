import tempfile
import unittest
from pathlib import Path

from cslr.data.adapters import CECSLVideoAdapter, NationalCSLDPImageSequenceAdapter
from cslr.data.manifest import read_manifest, validate_manifest


class ManifestTests(unittest.TestCase):
    def test_reads_valid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text(
                "sample_id,video,label,signer,session,split\n"
                "s1,clips/a.mp4,REGISTER,p01,day1,train\n",
                encoding="utf-8",
            )
            records = read_manifest(path)
            validate_manifest(records)
            self.assertEqual(records[0].label, "REGISTER")

    def test_rejects_duplicate_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text(
                "sample_id,video,label,signer,session,split\n"
                "s1,a.mp4,A,p01,day1,train\n"
                "s1,b.mp4,B,p02,day1,test\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_manifest(read_manifest(path))

    def test_nationalcsl_dp_adapter_scans_image_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "Participant_02" / "front" / "1660"
            sequence.mkdir(parents=True)
            (sequence / "00001.jpg").write_bytes(b"frame")
            adapter = NationalCSLDPImageSequenceAdapter(
                data_root=root,
                intent_labels={"REGISTER": ["1660"], "HELP": ["5094"]},
                split_by_signer={"Participant_02": "train", "Participant_03": "test"},
                views=["front"],
            )

            records = list(adapter.records())

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].sample_id, "Participant_02_front_1660")
        self.assertEqual(records[0].video.as_posix(), "Participant_02/front/1660")
        self.assertEqual(records[0].label, "REGISTER")
        self.assertEqual(records[0].split, "train")

    def test_ce_csl_adapter_reads_label_csv_and_video_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "label").mkdir()
            (root / "video" / "train" / "A").mkdir(parents=True)
            (root / "video" / "dev" / "B").mkdir(parents=True)
            (root / "video" / "train" / "A" / "train-00001.mp4").write_bytes(b"video")
            (root / "video" / "dev" / "B" / "dev-00001.mp4").write_bytes(b"video")
            (root / "label" / "train.csv").write_text(
                "Number,Translator,Chinese Sentences,Gloss,Note\n"
                "train-00001,A,2023年高考到了。,2/0/2/3/高/考/时间/到/。,\n",
                encoding="utf-8",
            )
            (root / "label" / "dev.csv").write_text(
                "Number,Translator,Chinese Sentences,Gloss,Note\n"
                "dev-00001,B,上车请买票。,上/汽车/票/买/要/。,\n",
                encoding="utf-8",
            )
            adapter = CECSLVideoAdapter(
                data_root=root,
                split_files={"train": "label/train.csv", "validation": "label/dev.csv"},
                video_splits={"train": "train", "validation": "dev"},
            )

            records = list(adapter.records())

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].sample_id, "train-00001")
        self.assertEqual(records[0].video.as_posix(), "video/train/A/train-00001.mp4")
        self.assertEqual(records[0].label, "2023年高考到了。")
        self.assertEqual(records[0].signer, "A")
        self.assertEqual(records[1].split, "validation")


if __name__ == "__main__":
    unittest.main()
