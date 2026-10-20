"""Regression checks for byte order, orientation, units and dataset pairing."""

import csv
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p3_depth.data.middlebury import (
    MANIFEST_FIELDS, discover_samples, disparity_to_depth_m, load_sample,
    read_calibration, read_disparity_pfm, write_manifest,
)


CALIBRATION = """cam0=[100 0 1; 0 100 1; 0 0 1]
cam1=[100 0 11; 0 100 1; 0 0 1]
baseline=200
doffs=10
width=2
height=2
ndisp=64
"""


class MiddleburyTests(unittest.TestCase):
    def setUp(self):
        test_root = (Path(__file__).resolve().parents[1] / "outputs/test-temp").resolve()
        test_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.temporary.name).resolve()
        if not self.root.is_relative_to(test_root):
            raise RuntimeError("Temporary test directory escaped the workspace")
        self.addCleanup(self.temporary.cleanup)

    def calibration(self):
        path = self.root / "calib.txt"
        path.write_text(CALIBRATION)
        return read_calibration(path)

    def scene(self, name="Example", variant="trainingQ", wrapper="inputs"):
        inputs = self.root / wrapper / "MiddEval3" / variant / name
        gt = self.root / "ground_truth" / "MiddEval3" / variant / name
        inputs.mkdir(parents=True)
        gt.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), (10, 20, 30)).save(inputs / "im0.png")
        Image.new("RGB", (2, 2), (11, 21, 31)).save(inputs / "im1.png")
        (inputs / "calib.txt").write_text(CALIBRATION)
        # PFM rows on disk: bottom [inf, 30], top [10, 20].
        (gt / "disp0GT.pfm").write_bytes(b"Pf\n2 2\n-0.003922\n" + struct.pack("<4f", float("inf"), 30, 10, 20))
        Image.fromarray(np.array([[255, 128], [0, 255]], dtype=np.uint8)).save(gt / "mask0nocc.png")
        return inputs, gt

    def test_pfm_endianness_orientation_and_nonunit_header(self):
        for scale, byte_order in [(-0.003922, "<"), (0.003922, ">")]:
            with self.subTest(scale=scale):
                path = self.root / "test.pfm"
                path.write_bytes(f"Pf\n2 2\n{scale}\n".encode() + struct.pack(byte_order + "4f", 30, 40, 10, 20))
                actual, header = read_disparity_pfm(path)
                np.testing.assert_array_equal(actual, [[10, 20], [30, 40]])
                self.assertEqual(actual.dtype, np.float32)
                self.assertEqual(header, scale)

    def test_pfm_rejects_corrupt_payload_and_header(self):
        cases = [b"Pf\n2 2\n-1\n" + b"\x00" * 15,
                 b"Pf\n2 2\n-1\n" + b"\x00" * 17,
                 b"PF\n2 2\n-1\n" + b"\x00" * 48,
                 b"Pf\n2 2\n0\n" + b"\x00" * 16]
        for index, payload in enumerate(cases):
            with self.subTest(index=index):
                path = self.root / "bad.pfm"
                path.write_bytes(payload)
                with self.assertRaises(ValueError):
                    read_disparity_pfm(path)

    def test_depth_units_doffs_and_invalids(self):
        depth = disparity_to_depth_m(np.array([[0, 10, -10, np.inf, np.nan]]), self.calibration())
        np.testing.assert_allclose(depth[0, :2], [2.0, 1.0])
        self.assertTrue(np.isnan(depth[0, 2:]).all())
        self.assertEqual(depth.dtype, np.float32)

    def test_calibration_rejects_inconsistent_doffs(self):
        path = self.root / "calib.txt"
        path.write_text(CALIBRATION.replace("doffs=10", "doffs=20"))
        with self.assertRaisesRegex(ValueError, "principal-point"):
            read_calibration(path)

    def test_discovery_pairs_separate_archives_and_excludes_other_sets(self):
        self.scene()
        self.scene("Hidden", "testQ")
        self.scene("Full", "trainingF")
        samples = discover_samples(self.root)
        self.assertEqual([sample.sample_id for sample in samples], ["Example"])
        sample = load_sample(samples[0])
        np.testing.assert_array_equal(sample.nonoccluded, [[True, False], [False, True]])
        np.testing.assert_array_equal(sample.valid_gt, [[True, True], [False, True]])
        np.testing.assert_allclose(sample.depth_m, [[1, 2/3], [np.nan, .5]], equal_nan=True)
        np.testing.assert_array_equal(sample.left_rgb[0, 0], [10, 20, 30])

    def test_duplicate_and_missing_scene_files_fail(self):
        inputs, _ = self.scene()
        (inputs / "im1.png").unlink()
        with self.assertRaises(FileNotFoundError):
            discover_samples(self.root)
        Image.new("RGB", (2, 2)).save(inputs / "im1.png")
        self.scene(wrapper="second_copy")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            discover_samples(self.root)

    def test_mismatched_gt_scene_fails(self):
        _, gt = self.scene()
        gt.rename(gt.with_name("WrongScene"))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            discover_samples(self.root)

    def test_wrong_mask_orientation_or_labels_fail(self):
        _, gt = self.scene()
        paths = discover_samples(self.root)[0]
        for values in [[[0, 255], [128, 255]], [[255, 127], [0, 255]]]:
            with self.subTest(values=values):
                Image.fromarray(np.array(values, dtype=np.uint8)).save(gt / "mask0nocc.png")
                with self.assertRaises(ValueError):
                    load_sample(paths)

    def test_wrong_image_dimensions_fail(self):
        inputs, _ = self.scene()
        Image.new("RGB", (3, 2)).save(inputs / "im1.png")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            load_sample(discover_samples(self.root)[0])

    def test_manifest_preserves_manual_split_and_notes(self):
        self.scene()
        samples = discover_samples(self.root)
        destination = self.root / "manifest.csv"
        write_manifest(samples, self.root, destination)
        with destination.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["split"], "unassigned")
        self.assertFalse(Path(rows[0]["left_path"]).is_absolute())
        rows[0]["split"], rows[0]["notes"] = "dev", "Chosen by the team"
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        write_manifest(samples, self.root, destination)
        with destination.open(newline="", encoding="utf-8") as handle:
            updated = list(csv.DictReader(handle))
        self.assertEqual(updated[0]["split"], "dev")
        self.assertEqual(updated[0]["notes"], "Chosen by the team")


if __name__ == "__main__":
    unittest.main()
