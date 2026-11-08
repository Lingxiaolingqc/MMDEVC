"""Verify the dated sample report against saved evidence (requires pypdf)."""
import csv
import hashlib
import json
import re
import statistics
import zipfile
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report/sample_20261109"
DELIVERY = ROOT / "outputs/submission_20261109"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(name):
    with (REPORT / "data" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compact(text):
    text = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", text)
    return re.sub(r"\s+", "", text)


def number(value, places, omit_zero=True):
    text = f"{float(value):.{places}f}"
    return text[1:] if omit_zero and text.startswith("0.") else text


def main():
    tex = (REPORT / "main.tex").read_text(encoding="utf-8")
    flat = compact(tex)
    verified_rows = []

    def table_row(label, values):
        row = label + "&" + "&".join(values) + r"\\"
        require(compact(row) in flat, "Report row differs from evidence: " + row)
        verified_rows.append(row)

    assets = read_json(REPORT / "asset_provenance.json")
    require(digest(ROOT / "scripts/build_sample_report_assets.py") == assets["implementation_sha256"],
            "Asset builder changed since generation")
    for path, sha in assets["sources_sha256"].items():
        require(digest(ROOT / path) == sha, "Changed source: " + path)
    for path, sha in assets["images_sha256"].items():
        require(digest(REPORT / "images" / path) == sha, "Changed figure: " + path)
    source_digests = set(assets["sources_sha256"].values())
    for path in (REPORT / "data").iterdir():
        require(digest(path) in source_digests, "Data copy lacks original hash: " + path.name)
    reviews = list((REPORT / "data").glob("*_review.json"))
    require(len(reviews) == 12, "Expected twelve completed analysis records")
    for path in reviews:
        record = read_json(path)
        complete = record["status"] in {
            "complete_for_declared_quantitative_and_visual_scope",
            "complete_for_declared_quantitative_and_sampled_visual_scope"}
        if path.name.startswith("confidence_"):
            complete = (record["status"] == "completed_with_scope_limits"
                        and record["independent_reanalysis"] is True
                        and record["user_observations_used_as_proof"] is False
                        and len(record["visual_evidence"]) == 4)
        require(complete,
                "Incomplete independent review: " + path.name)

    for r in rows("stereo_summary.csv")[:4]:
        label = ("All" if r["region"] == "all_gt" else "Nonocc.") + "&" + r["custom_lr"]
        table_row(label, [number(r["macro_disparity_mae_px"], 3),
                         number(100 * float(r["pooled_coverage"]), 2),
                         number(r["macro_depth_rmse_m"], 3),
                         number(100 * float(r["pooled_bad1_invalid_as_bad"]), 2)])
    nyu = read_json(REPORT / "data/nyu_summary.json")
    for label, key in [("Unclipped diagnostic", "raw_unclipped_macro"), ("Course-clipped", "course")]:
        table_row(label, [number(nyu[key][k], 5) for k in ("abs_rel", "rmse_m", "delta1")])
    for r in rows("vda_comparison.csv"):
        label = "DA V2 Small" if r["model"] == "DA" else "VDA Small offline"
        table_row(label, [number(r["ms_per_frame"], 3), number(r["fps"], 3),
                         number(r["peak_cuda_allocated_mib"], 2)])
    nu_run = read_json(ROOT / "outputs/part2_models/unidepth/test/summary.json")
    nu_meta = read_json(ROOT / "outputs/part2_models/unidepth/test/metadata.json")
    table_row("UniDepthV2 ViT-S", [number(nu_run["mean_infer_seconds"] * 1000, 3),
                                 number(nu_run["throughput_fps_excluding_io"], 3),
                                 number(nu_meta["peak_cuda_allocated_bytes"] / 1024**2, 2)])
    main_rows = [{r["variant"]: r for r in rows(f"main_{clip}_comparison.csv")}
                 for clip in ("dev", "course")]
    for variant in ("E0", "E1", "E2", "E3", "E4"):
        values = []
        for source in main_rows:
            r = source[variant]
            values += [number(r["capped_normalized_l1_full_reference"], 6),
                       "0" if variant == "E0" else number(r["mean_departure_from_DA_normalized"], 6)]
        table_row(variant, values)
    for label, clip in [("Tokyo", "dev"), ("Rollercoaster", "course")]:
        table_row(label, [number(r["recursive_T_reduction_percent"], 2, omit_zero=False) + r"\%"
                          for r in rows(f"history_{clip}_contrasts.csv")])
    confidence = [{r["id"]: r for r in rows(f"confidence_{clip}_summary.csv")}
                  for clip in ("dev", "course")]
    for label, key in [("Full", "E4_recursive"), ("No photo", "E4_no_photo"),
                       ("No flow", "E4_no_flow"), ("No depth", "E4_no_depth"),
                       ("No edge", "E4_no_edge")]:
        values = []
        for source in confidence:
            values += [number(source[key][k], 6) for k in ("T", "departure_from_DA")]
        table_row(label, values)
    application = rows("application_per_frame.csv")
    require(len(application) == 70, "Application frame count")
    for model, setting in [("DA", "old"), ("E4", "old"), ("DA", "new"), ("E4", "new")]:
        prefix = model + "_" + setting
        mean_weight = statistics.mean(float(r[prefix + "_mean_weight"]) for r in application)
        sharp = 100 * statistics.mean(float(r[prefix + "_fully_sharp_fraction"]) for r in application)
        table_row(model + ", " + setting, [number(mean_weight, 6), number(sharp, 4)])

    expected_citations = {"sgm", "kitti", "dpt", "dav1", "dav2", "unidepth",
                          "unidepthv2", "depthpro", "raft", "vda",
                          "nyu", "middlebury", "opencv", "farneback"}
    cited = {key for match in re.findall(r"\\cite\{([^}]+)\}", tex) for key in match.split(",")}
    require(cited == expected_citations, "Expected reading/implementation citations")
    build = DELIVERY / "report_build"
    bbl = (build / "main.bbl").read_text(encoding="utf-8")
    require(bbl.count(r"\bibitem[") == 14, "Expected fourteen resolved bibliography entries")
    log = (build / "main.log").read_text(encoding="utf-8")
    require(not re.search(r"Overfull|Underfull|undefined|multiply defined", log, re.I),
            "Unresolved LaTeX or layout warning")
    pdf = DELIVERY / "MMDEVC_Project3_Sample_Report.pdf"
    require(digest(pdf) == digest(build / "main.pdf"), "Final PDF differs from inspected build")
    reader = PdfReader(pdf)
    texts = [page.extract_text() or "" for page in reader.pages]
    require(len(texts) == 9 and texts[8].lstrip().startswith("References"),
            "Expected eight body pages and one references page")
    require(all("??" not in text for text in texts), "Unresolved PDF text")
    font_names = set()
    for page in reader.pages:
        for ref in page["/Resources"].get_object().get("/Font", {}).get_object().values():
            font = ref.get_object()
            for face in font.get("/DescendantFonts", [font]):
                face = face.get_object()
                descriptor = face.get("/FontDescriptor")
                if descriptor is None:
                    require(face.get("/Subtype") == "/Type3", "Missing font descriptor")
                else:
                    descriptor = descriptor.get_object()
                    require(any(k in descriptor for k in ("/FontFile", "/FontFile2", "/FontFile3")),
                            "Unembedded font")
                font_names.add(str(face.get("/BaseFont", face.get("/Name", "Type3"))))
    visual = read_json(DELIVERY / "visual_review.json")
    require(visual["pdf_sha256"] == digest(pdf) and visual["pages_inspected"] == list(range(1, 10)),
            "Visual record does not match final PDF")
    for name, sha in visual["rendered_sha256"].items():
        require(digest(DELIVERY / name) == sha, "Changed rendered page")

    archive = DELIVERY / "depth_results.zip"
    receipt = read_json(archive.with_suffix(".verification.json"))
    require(digest(archive) == receipt["archive_sha256"], "Archive changed after verification")
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, "ZIP CRC failed")
        manifest = json.loads(z.read("manifest.json"))
        require(len(manifest["files"]) == 80 and len(z.namelist()) == 82, "ZIP count")
        for name, record in manifest["files"].items():
            require(hashlib.sha256(z.read(name)).hexdigest() == record["sha256"], "ZIP payload: " + name)
        require(not any("acceptance" in n.lower() for n in z.namelist()), "Personal guide in ZIP")

    source_files = [p for p in REPORT.rglob("*") if p.is_file()
                    and p.name != "report_provenance.json" and "__pycache__" not in p.parts]
    result = {
        "date": "2026-11-09", "status": "verified",
        "body_pages": 8, "reference_pages": 1, "cited_references": 14,
        "embedded_font_names": sorted(font_names),
        "verified_numeric_table_rows": len(verified_rows),
        "numeric_table_rows": verified_rows,
        "complete_independent_analysis_records": len(reviews),
        "asset_source_files_verified": len(assets["sources_sha256"]),
        "report_files_sha256": {p.relative_to(REPORT).as_posix(): digest(p) for p in sorted(source_files)},
        "verifier_sha256": digest(Path(__file__)),
        "pdf_sha256": digest(pdf), "pdf_path": pdf.relative_to(ROOT).as_posix(),
        "visual_review_sha256": digest(DELIVERY / "visual_review.json"),
        "archive": receipt,
        "limits": ["Numeric check covers all seven result tables; prose is separately reviewed.",
                   "Visual inspection is explicitly recorded; this script cannot assess appearance.",
                   "No member acceptance, Canvas submission, arXiv publication, or blind-test claim."]
    }
    (REPORT / "report_provenance.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(verified_rows)} numeric rows, 12 analysis records, 8+1 PDF pages, "
          f"14 references, {len(font_names)} embedded fonts, 80 ZIP payload hashes")


if __name__ == "__main__":
    main()
