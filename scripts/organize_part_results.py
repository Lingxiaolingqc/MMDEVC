"""Create/verify part-grouped browsing copies without moving experiment originals."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DESTINATIONS = {"table": ROOT / "results/by_part", "media": ROOT / "outputs/by_part"}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def media_readmes(entries):
    """Build portable local navigation when another member materializes media."""
    media = [row for row in entries if row["kind"] == "media"]
    groups = sorted({row["destination"].split("/")[0] for row in media})
    result = {}
    root_text = "# 按 Part 浏览图片、视频和交付文件\n\n[项目总导航](../../PARTS.md) · [小型指标](../../results/by_part/README.md) · [副本哈希](index.json)\n\n"
    root_text += "\n".join(f"- [{group}]({group}/README.md)" for group in groups)
    root_text += "\n\n这些是本机阅读副本，未纳入 Git。数据集、权重、原始预测数组和个人验收指南未复制。\n"
    result[DESTINATIONS["media"] / "README.md"] = root_text.encode("utf-8")
    for group in groups:
        text = f"# {group}：本机媒体\n\n[总导航](../../../PARTS.md) · [本部分指标、解释与原始预测](../../../results/by_part/{group}/README.md)\n\n"
        text += "表中副本与原件逐项核对 SHA256；这不是新实验。\n\n| 浏览副本 | 原件 |\n| --- | --- |\n"
        for row in media:
            if row["destination"].startswith(group + "/"):
                name = row["destination"].split("/", 1)[1]
                text += f'| [{name}]({name}) | [原件](../../../{row["source"]}) |\n'
        result[DESTINATIONS["media"] / group / "README.md"] = text.encode("utf-8")
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tables-only", action="store_true", help="Use only small result records available through Git.")
    p.add_argument("--check-only", action="store_true", help="Verify existing copies without writing.")
    a = p.parse_args()
    entries = json.loads((ROOT / "configs/part_results.json").read_text(encoding="utf-8"))["artifacts"]
    selected = [row for row in entries if not a.tables_only or row["kind"] == "table"]
    checked = []
    # Validate the complete selected plan before creating any copies.
    seen = set()
    for row in selected:
        source = (ROOT / row["source"]).resolve()
        base = DESTINATIONS[row["kind"]].resolve()
        target = (base / row["destination"]).resolve()
        if not source.is_relative_to(ROOT.resolve()) or not target.is_relative_to(base) or target == base:
            raise ValueError("Artifact path escapes its intended directory")
        if target in seen:
            raise ValueError(f"Duplicate destination: {target}")
        seen.add(target)
        if not source.is_file():
            raise FileNotFoundError(f"Missing original: {source}; use --tables-only on a clone without local outputs.")
        digest = sha256(source)
        if target.exists() and (not target.is_file() or sha256(target) != digest):
            raise ValueError(f"Existing browsing copy differs; preserve it and investigate: {target}")
        if a.check_only and not target.is_file():
            raise FileNotFoundError(f"Missing browsing copy: {target}")
        checked.append((row, source, target, digest))
    for row, source, target, digest in checked:
        if not a.check_only and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if sha256(target) != digest:
            raise ValueError(f"Copied bytes differ: {target}")
    manifests = {}
    for kind in {row["kind"] for row, *_ in checked}:
        records = [{"source": row["source"], "destination": row["destination"], "sha256": digest,
                    "bytes": source.stat().st_size} for row, source, target, digest in checked if row["kind"] == kind]
        manifests[DESTINATIONS[kind] / "index.json"] = (json.dumps({"schema_version": 1, "artifacts": records},
                                                               ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if not a.tables_only:
        manifests.update(media_readmes(selected))
    for path, content in manifests.items():
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Existing index differs; preserve it and investigate: {path}")
        if a.check_only:
            if not path.exists():
                raise FileNotFoundError(path)
        elif not path.exists():
            path.write_bytes(content)
    print(f"PASS: {len(checked)} part-grouped artifacts match their originals; originals unchanged.")

if __name__ == "__main__":
    main()
