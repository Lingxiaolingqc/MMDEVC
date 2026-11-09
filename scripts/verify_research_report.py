"""Verify the research-ordered report without rewriting prior receipts."""
import re
from pathlib import Path

from pypdf import PdfReader

from verify_extension_report import compact, read, require, sha

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report/sample_20261109_research"
DELIVERY = ROOT / "outputs/submission_20261109_research"


def main():
    import json

    revision = read(REPORT / "revision_manifest.json")
    parent = ROOT / revision["parent"]
    receipt = read(REPORT / "inherited_report_provenance.json")
    require(sha(parent / "report_provenance.json") == revision["parent_receipt_sha256"], "Parent receipt changed")
    require(sha(REPORT / "inherited_report_provenance.json") == revision["parent_receipt_sha256"], "Inherited receipt changed")
    for name, digest in receipt["report_files_sha256"].items():
        require(sha(parent / name) == digest, "Prior report changed: " + name)
        if name not in {"main.tex", "README.md"}:
            require(sha(REPORT / name) == digest, "Copied evidence changed: " + name)

    original = read(REPORT / "asset_provenance.json")
    extension = read(REPORT / "extension_assets.json")
    require(sha(ROOT / "scripts/build_sample_report_assets.py") == original["implementation_sha256"], "Original asset builder changed")
    require(sha(ROOT / "scripts/build_extension_report_assets.py") == extension["implementation_sha256"], "Extended asset builder changed")
    sources = {**original["sources_sha256"], **extension["source_sha256"]}
    for path, digest in sources.items():
        require(sha(ROOT / path) == digest, "Saved experiment source changed: " + path)

    tex = (REPORT / "main.tex").read_text(encoding="utf-8")
    flat = compact(tex)
    numeric_rows = receipt["numeric_table_rows"]
    require(len(numeric_rows) == 32, "Expected 32 inherited numeric result rows")
    for row in numeric_rows:
        require(compact(row) in flat, "Numeric row changed or missing: " + row)
    require(re.findall(r"\\section\{([^}]+)\}", tex) == revision["top_level_sections"], "Top-level order differs")
    experiments = tex.split(r"\section{Experiments}", 1)[1].split(r"\section{Discussion and Conclusion}", 1)[0]
    require(re.findall(r"\\subsection\{([^}]+)\}", experiments) == revision["experiment_sections"], "Five experiment sections differ")
    expected_keys = {"sgm", "kitti", "dpt", "dav1", "dav2", "unidepth", "unidepthv2", "depthpro", "raft", "vda", "nyu", "middlebury", "opencv", "farneback"}
    cited = {k for group in re.findall(r"\\cite\{([^}]+)\}", tex) for k in group.split(",")}
    require(cited == expected_keys, "Citations differ from required and supporting references")
    reviews = [p for directory in ("data", "extension") for p in (REPORT / directory).glob("*_review.json")
               if not p.name.endswith("quantitative_review.json")]
    require(len(reviews) == 16, "Expected 16 completed reviews")
    for path in reviews:
        review = read(path)
        require(review["status"].startswith("complete") and review.get("independent_reanalysis", True), "Incomplete review")

    build = DELIVERY / "report_build"
    log = (build / "main.log").read_text(encoding="utf-8")
    require(not re.search(r"Overfull|Underfull|undefined|multiply defined", log, re.I), "LaTeX warning")
    require((build / "main.bbl").read_text(encoding="utf-8").count(r"\bibitem[") == 14, "Bibliography incomplete")
    pdf = DELIVERY / "MMDEVC_Project3_Research_Report.pdf"
    require(sha(pdf) == sha(build / "main.pdf"), "Delivered PDF differs from build")
    reader = PdfReader(pdf)
    texts = [p.extract_text() or "" for p in reader.pages]
    require(len(texts) == 9 and texts[8].lstrip().startswith("References"), "Expected 8 body plus 1 reference page")
    require(all("??" not in text for text in texts), "PDF has unresolved references")
    fonts = set()
    for page in reader.pages:
        for ref in page["/Resources"].get_object().get("/Font", {}).get_object().values():
            font = ref.get_object()
            for face in font.get("/DescendantFonts", [font]):
                face = face.get_object()
                descriptor = face.get("/FontDescriptor")
                require(descriptor is not None or face.get("/Subtype") == "/Type3", "Font descriptor missing")
                if descriptor:
                    require(any(k in descriptor.get_object() for k in ("/FontFile", "/FontFile2", "/FontFile3")), "Unembedded font")
                fonts.add(str(face.get("/BaseFont", "Type3")))

    visual = read(DELIVERY / "visual_review.json")
    require(visual["pdf_sha256"] == sha(pdf), "Visual review does not match PDF")
    require(visual["pages_inspected"] == list(range(1, 10)), "Not all pages visually inspected")
    require(len(visual["rendered_sha256"]) == 9, "Expected nine inspected page renders")
    for name, digest in visual["rendered_sha256"].items():
        require(sha(DELIVERY / name) == digest, "Inspected page render changed")
    preservation = DELIVERY / "preservation_snapshot.json"
    if preservation.exists():
        for name, digest in read(preservation).items():
            require(sha(ROOT / name) == digest, "Previously saved or user-owned file changed: " + name)

    result = dict(status="verified", date="2026-11-09", body_pages=8, reference_pages=1,
                  references=14, independent_reviews=16, source_file_count=len(sources),
                  numeric_table_rows=numeric_rows, embedded_fonts=sorted(fonts),
                  pdf_path=pdf.relative_to(ROOT).as_posix(), pdf_sha256=sha(pdf),
                  parent_receipt_sha256=revision["parent_receipt_sha256"],
                  ordered_experiments=revision["experiment_sections"],
                  report_files_sha256={p.relative_to(REPORT).as_posix(): sha(p) for p in REPORT.rglob("*")
                                       if p.is_file() and p.name != "report_provenance.json"},
                  verifier_sha256=sha(Path(__file__)),
                  verifier_helper_sha256=sha(ROOT / "scripts/verify_extension_report.py"),
                  visual_review_sha256=sha(DELIVERY / "visual_review.json"),
                  limits=["No new experiments; numeric checks retain previously verified rows and live source hashes.",
                          "Prose and sampled visual interpretations reviewed separately from numeric checks.",
                          "The preserved earlier video ZIP contains the previous PDF, not this revision.",
                          "No author acceptance or course submission is implied."])
    (REPORT / "report_provenance.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"PASS: five ordered experiments, {len(numeric_rows)} numeric rows, {len(sources)} sources, 16 reviews, 8+1 pages, 14 references and embedded fonts")


if __name__ == "__main__":
    main()
