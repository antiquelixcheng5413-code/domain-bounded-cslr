import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
