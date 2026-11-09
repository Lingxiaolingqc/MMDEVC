"""Preserve the original delivery and create a separately verified extended ZIP."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/submission_20261109_extension"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    source = ROOT / "outputs/submission_20261109/depth_results.zip"
    target = OUT / "depth_results_extended.zip"
    if target.exists():
        raise FileExistsError("Preserve completed package")
    previous = json.loads(source.with_suffix(".verification.json").read_text(encoding="utf-8"))
    if sha(source) != previous["archive_sha256"]:
        raise ValueError("Original package changed")
    manifest = {"schema": "project3_extended_demo_v1", "date": "2026-11-09",
                "original_archive_sha256": sha(source), "files": {},
                "scope": "Original main delivery plus optional streaming and bounded refinement; no training or metric-video; no personal acceptance guide."}
    additions = {}
    pdf = OUT / "MMDEVC_Project3_Sample_Report_Extended.pdf"
    additions["report/"+pdf.name] = pdf
    for p in (ROOT / "report/sample_20261109_extension").rglob("*"):
        if p.is_file() and p.name != "report_provenance.json":
            additions["report/source/" + p.relative_to(ROOT / "report/sample_20261109_extension").as_posix()] = p
    for split, clip in (("dev", "dev_tokyo_100"), ("test", "course_rollercoaster_70")):
        modes = ROOT / "outputs/part2_models/vda_modes" / split / clip
        study = ROOT / "outputs/part3_improvement/confidence_refinement" / split / clip
        for label, base in (("streaming", modes), ("refinement", study)):
            for p in base.rglob("*"):
                if p.is_file() and p.suffix in (".json", ".csv", ".md", ".png", ".mp4"):
                    additions[f"extensions/{label}/{split}/{clip}/"+p.relative_to(base).as_posix()] = p
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        old_manifest = json.loads(old.read("manifest.json"))
        for name, record in old_manifest["files"].items():
            data = old.read(name)
            digest = hashlib.sha256(data).hexdigest()
            if digest != record["sha256"]:
                raise ValueError("Original payload changed: " + name)
            z.writestr(name, data)
            manifest["files"][name] = {"sha256": digest, "bytes": len(data), "origin": "original_verified_package"}
        for name, path in sorted(additions.items()):
            if name in manifest["files"]:
                raise ValueError("Duplicate payload")
            z.write(path, name)
            manifest["files"][name] = {"sha256": sha(path), "bytes": path.stat().st_size,
                                       "source": path.relative_to(ROOT).as_posix()}
        z.writestr("README.txt", "Extended Project 3 sample delivery, 2026-11-09.\n"
                   "Start with report/MMDEVC_Project3_Sample_Report_Extended.pdf.\n"
                   "Original main comparisons remain under clips/, application/ and metrics paths.\n"
                   "extensions/streaming contains matched offline/streaming videos and timing.\n"
                   "extensions/refinement contains original/global/selective controls and independent analyses.\n"
                   "Original E4 remains selected; this bounded attempt showed no clear sampled visual gain.\n"
                   "The official course clip has 70 frames, not 100. Video has no depth GT.\n"
                   "Full NPY arrays, weights and datasets remain local; raw source hashes are recorded.\n"
                   "This is a sample for author review, not an automatic course submission.\n")
        z.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None or len(z.namelist()) != len(manifest["files"])+2:
            raise ValueError("ZIP CRC or inventory mismatch")
        for name, record in manifest["files"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != record["sha256"]:
                raise ValueError("ZIP payload mismatch")
        if any("personal_acceptance" in n for n in z.namelist()):
            raise ValueError("Personal guide must stay local")
    receipt = dict(status="verified", archive_sha256=sha(target), payload_files=len(manifest["files"]),
                   bytes=target.stat().st_size, original_archive_sha256=sha(source), all_payload_hashes_verified=True,
                   implementation_sha256=sha(Path(__file__)))
    target.with_suffix(".verification.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print("PASS extended archive:", receipt)


if __name__ == "__main__":
    main()
