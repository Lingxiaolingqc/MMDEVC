"""Verify the separately versioned extended report against current saved evidence."""
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report/sample_20261109_extension"
DELIVERY = ROOT / "outputs/submission_20261109_extension"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compact(s):
    return re.sub(r"\s+", "", re.sub(r"\\textbf\{([^{}]+)\}", r"\1", s))


def rows(name):
    with (REPORT / "extension" / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    old = read(REPORT / "asset_provenance.json")
    extension = read(REPORT / "extension_assets.json")
    require(sha(ROOT / "scripts/build_sample_report_assets.py") == old["implementation_sha256"], "Original asset builder changed")
    require(sha(ROOT / "scripts/build_extension_report_assets.py") == extension["implementation_sha256"], "Extension asset builder changed")
    for path, digest in {**old["sources_sha256"], **extension["source_sha256"]}.items():
        require(sha(ROOT / path) == digest, "Changed source: " + path)
    for name, digest in {**old["images_sha256"], **extension["images_sha256"]}.items():
        require(sha(REPORT / "images" / name) == digest, "Changed report figure")
    for name, digest in extension["copies_sha256"].items():
        require(sha(REPORT / "extension" / name) == digest, "Changed extension copy")
    for p in (REPORT / "data").iterdir():
        require(sha(p) in set(old["sources_sha256"].values()), "Original data copy lacks matching source")
    tex = (REPORT / "main.tex").read_text(encoding="utf-8")
    flat = compact(tex)
    old_receipt = read(ROOT / "report/sample_20261109/report_provenance.json")
    verified = list(old_receipt["numeric_table_rows"])
    for row in verified:
        require(compact(row) in flat, "Previously verified numeric row changed: " + row)
    for split, label in (("dev", "Tky"), ("test", "RC")):
        for r in rows(f"streaming_{split}_comparison.csv"):
            numbers = [f"{float(r[k]):.{n}f}" for k, n in (
                ("median_first_api_return_ms", 1), ("median_amortized_ms", 2), ("max_allocated_mib", 1), ("T_own_span", 6))]
            numbers[-1] = numbers[-1].lstrip("0")
            row = label + "&" + ("Offline" if r["mode"] == "offline" else "Stream") + "&" + "&".join(numbers) + r"\\"
            require(compact(row) in flat, "Streaming numeric row differs: " + row); verified.append(row)
    sources = [{r["id"]: r for r in rows(f"refinement_{split}_comparison.csv")} for split in ("dev", "test")]
    for mode in ("original", "global", "selective"):
        values = []
        for source in sources:
            values += [f"{float(source[mode]['capped_normalized_l1_full_reference']):.6f}".lstrip("0"),
                       f"{float(source[mode]['fusion_seconds']):.3f}"]
        row = mode.capitalize() + "&" + "&".join(values) + r"\\"
        require(compact(row) in flat, "Refinement numeric row differs: " + row); verified.append(row)
    reviews = [p for p in list((REPORT / "data").glob("*_review.json")) + list((REPORT / "extension").glob("*_review.json"))
               if not p.name.endswith("quantitative_review.json")]
    require(len(reviews) == 16, "Expected 12 original plus four independent extension reviews")
    for p in reviews:
        m = read(p)
        require(m["status"].startswith("complete") and m.get("independent_reanalysis", True), "Review incomplete")
    expected = {"sgm", "kitti", "dpt", "dav1", "dav2", "unidepth", "unidepthv2", "depthpro", "raft", "vda", "nyu", "middlebury", "opencv", "farneback"}
    cited = {key for x in re.findall(r"\\cite\{([^}]+)\}", tex) for key in x.split(",")}
    require(cited == expected, "Reading references missing")
    log = (DELIVERY / "report_build/main.log").read_text(encoding="utf-8")
    require(not re.search("Overfull|Underfull|undefined|multiply defined", log, re.I), "LaTeX layout/reference warning")
    require((DELIVERY / "report_build/main.bbl").read_text(encoding="utf-8").count(r"\bibitem[") == 14, "Unresolved bibliography")
    pdf = DELIVERY / "MMDEVC_Project3_Sample_Report_Extended.pdf"
    require(sha(pdf) == sha(DELIVERY / "report_build/main.pdf"), "Delivered PDF differs")
    r = PdfReader(pdf); texts = [p.extract_text() or "" for p in r.pages]
    require(len(texts) == 9 and texts[8].lstrip().startswith("References"), "Require 8 body pages plus references")
    require(all("??" not in t for t in texts), "Unresolved PDF reference")
    fonts = set()
    for p in r.pages:
        for ref in p["/Resources"].get_object().get("/Font", {}).get_object().values():
            f = ref.get_object()
            for face in f.get("/DescendantFonts", [f]):
                face = face.get_object(); descriptor = face.get("/FontDescriptor")
                require(descriptor is not None or face.get("/Subtype") == "/Type3", "Missing font descriptor")
                if descriptor:
                    require(any(k in descriptor.get_object() for k in ("/FontFile", "/FontFile2", "/FontFile3")), "Unembedded font")
                fonts.add(str(face.get("/BaseFont", "Type3")))
    visual = read(DELIVERY / "visual_review.json")
    require(visual["pdf_sha256"] == sha(pdf) and visual["pages_inspected"] == list(range(1, 10)), "PDF not visually verified")
    for name, digest in visual["rendered_sha256"].items():
        require(sha(DELIVERY / name) == digest, "Page render changed")
    archive = DELIVERY / "depth_results_extended.zip"; receipt = read(archive.with_suffix(".verification.json"))
    require(receipt["archive_sha256"] == sha(archive), "Archive changed")
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, "ZIP CRC")
        manifest = json.loads(z.read("manifest.json"))
        require(len(z.namelist()) == len(manifest["files"])+2, "ZIP inventory")
        for name, record in manifest["files"].items():
            require(hashlib.sha256(z.read(name)).hexdigest() == record["sha256"], "ZIP payload")
    result = dict(status="verified", date="2026-11-09", body_pages=8, reference_pages=1, references=14,
        numeric_table_rows=verified, independent_reviews=16, embedded_fonts=sorted(fonts),
        source_file_count=len({**old["sources_sha256"], **extension["source_sha256"]}),
        pdf_sha256=sha(pdf), pdf_path=pdf.relative_to(ROOT).as_posix(), archive=receipt,
        report_files_sha256={p.relative_to(REPORT).as_posix(): sha(p) for p in REPORT.rglob("*") if p.is_file() and p.name != "report_provenance.json"},
        verifier_sha256=sha(Path(__file__)), visual_review_sha256=sha(DELIVERY / "visual_review.json"),
        limits=["Numerical checks cover retained and new tables; prose and visual claims reviewed separately.", "No author acceptance or course submission is implied."])
    (REPORT / "report_provenance.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(f"PASS: {len(verified)} numeric rows, 16 independent reviews, 8+1 pages, 14 references, embedded fonts and all ZIP hashes")


if __name__ == "__main__":
    main()
