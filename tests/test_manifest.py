import tempfile
import unittest
from pathlib import Path

from cslr.data.adapters import NationalCSLDPImageSequenceAdapter
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


if __name__ == "__main__":
    unittest.main()
