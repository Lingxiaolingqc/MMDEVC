"""Audit the saved old/new blur comparison and export descriptive diagnostics.

No depth inference, parameter search, or overwrite of original application output.
The generated review is numeric evidence; visual acceptance must follow inspection.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import json
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import ROOT, repo_path, checksum, write_json, write_csv
from part3_improvement.run_fusion_suite import load_run
from p3_depth.application import background_blur


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def equal(actual, expected, name):
    if not np.isclose(actual, float(expected), rtol=1e-9, atol=1e-9):
        raise ValueError(f"{name} differs: {actual} != {expected}")


def effects_at(rgb, predictions, masks, parameters, frame):
    result, weights = [], []
    for params in parameters:
        for pred, valid in zip(predictions, masks):
            img, w = background_blur(rgb, pred[frame], valid[frame], params["limits"],
                                     sigma=params["sigma_px"], transition=params["transition_normalized"])
            result.append(img); weights.append(w)
    return result, weights


def sheet(frames, predictions, masks, params, times, selected, target, roi=None):
    h, w = frames.shape[1:3]
    x0, y0, x1, y1 = roi or (0, 0, w, h)
    scale = 1 if roi else .5
    pw, ph = int((x1-x0)*scale), int((y1-y0)*scale)
    canvas = Image.new("RGB", (5*pw, len(selected)*(ph+28)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default(size=15)
    for row, t in enumerate(selected):
        imgs, _ = effects_at(frames[t], predictions, masks, params, t)
        for col, (name, img) in enumerate(zip(["RGB", "DA old", "E4 old", "DA new", "E4 new"],
                                              [frames[t], *imgs])):
            crop = Image.fromarray(img[y0:y1, x0:x1]).resize((pw,ph), Image.Resampling.LANCZOS)
            top = row*(ph+28)
            canvas.paste(crop, (col*pw, top+28))
            draw.text((col*pw+3, top+4), f"{name} | f{t} {times[t]:.2f}s", fill="black", font=font)
    canvas.save(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Saved new application with settings_comparison")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--improved", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    run, cache, baseline, improved = map(repo_path, (args.run,args.cache,args.baseline,args.improved))
    out = repo_path(args.output_dir) if args.output_dir else run/"analysis"
    if out.exists():
        raise FileExistsError("Use a new analysis directory")
    meta, cm = read(run/"metadata.json"), read(cache/"metadata.json")
    ref = repo_path(meta["settings_comparison"]["reference_application_path"])
    old = read(ref/"metadata.json")
    if meta["status"]!="complete" or old["status"]!="complete":
        raise ValueError("Both application exports must be complete")
    if checksum(ref/"metadata.json") != meta["settings_comparison"]["reference_application_metadata_sha256"]:
        raise ValueError("Old application metadata changed")
    for folder, m in ((run,meta),(ref,old)):
        if m["cache_metadata_sha256"]!=checksum(cache/"metadata.json") or m["methods"]!=meta["methods"]:
            raise ValueError("Applications do not have identical sources")
        if m["implementation_sha256"]["src/p3_depth/application.py"]!=checksum(ROOT/"src/p3_depth/application.py"):
            raise ValueError("Application effect implementation changed")
        for name, digest in m["files"].items():
            if checksum(folder/name)!=digest:
                raise ValueError("Saved application file changed: "+name)
    for name, key in (("frames.npy","frames_sha256"),("frames.csv","index_sha256")):
        if checksum(cache/name)!=cm[key]:
            raise ValueError("RGB or timestamps changed")
    bm, base, bv = load_run(baseline, cm)
    im, imp, iv = load_run(improved, cm)
    if [checksum(p/"metadata.json") for p in (baseline,improved)]!=[m["metadata_sha256"] for m in meta["methods"]]:
        raise ValueError("Prediction identities differ")
    if any(m["prediction_type"]!="inverse_depth_like" for m in (bm,im)) or im["config_status"]!="frozen":
        raise ValueError("Require relative predictions and frozen E4")
    params = [old["parameters"],meta["parameters"]]
    if params[0]["limits"]!=params[1]["limits"] or params[0]["sigma_px"]!=params[1]["sigma_px"]:
        raise ValueError("This analysis isolates transition thresholds only")
    limits = list(map(float,np.quantile(base[bv],[.01,.99])))
    if params[0]["limits"]!=limits or im["normalization_reference"]["limits"]!=limits:
        raise ValueError("Fixed baseline display scale differs")
    if meta["settings_comparison"]["previous_parameters"]!=params[0] or meta["settings_comparison"]["adjusted_parameters"]!=params[1]:
        raise ValueError("Recorded comparison parameters differ")
    frames = np.load(cache/"frames.npy",mmap_mode="r",allow_pickle=False)
    times = [float(r["timestamp_s"]) for r in csv_rows(cache/"frames.csv")]
    if base.shape!=frames.shape[:3] or imp.shape!=base.shape or len(times)!=len(base):
        raise ValueError("Mismatched frames")
    old_rows, new_rows, comparison = [csv_rows(p) for p in
        (ref/"blur_strength.csv",run/"blur_strength.csv",run/"settings_comparison.csv")]
    if any(len(rows)!=len(frames) for rows in (old_rows,new_rows,comparison)):
        raise ValueError("Incomplete saved statistics")
    for folder, name in ((ref,"demo.mp4"),(run,"demo.mp4"),(run,"settings_comparison.mp4")):
        with av.open(str(folder/name)) as container:
            decoded = [float(f.pts*f.time_base) for f in container.decode(video=0)]
        if len(decoded)!=len(times) or not np.allclose(decoded,np.array(times)-times[0],rtol=0,atol=2e-6):
            raise ValueError("Movie timestamps differ")
    rows = []
    names = ("DA_old","E4_old","DA_new","E4_new")
    for t, rgb in enumerate(frames):
        imgs, weights = effects_at(rgb,(base,imp),(bv,iv),params,t)
        row = {"frame":t,"timestamp_s":times[t]}
        for name, weight in zip(names,weights):
            row[name+"_mean_weight"] = float(weight.mean())
            row[name+"_fully_sharp_fraction"] = float((weight==0).mean())
            row[name+"_fully_blurred_fraction"] = float((weight==1).mean())
        rgb_difference = lambda a,b: float(np.abs(a.astype(float)-b.astype(float)).mean())
        row["DA_settings_rgb_change"] = rgb_difference(imgs[0],imgs[2])
        row["E4_settings_rgb_change"] = rgb_difference(imgs[1],imgs[3])
        row["old_model_rgb_change"] = rgb_difference(imgs[0],imgs[1])
        row["new_model_rgb_change"] = rgb_difference(imgs[2],imgs[3])
        for saved in (old_rows[t],new_rows[t],comparison[t]):
            equal(t,saved["frame"],"frame"); equal(times[t],saved["timestamp_s"],"time")
        for saved, offset in ((old_rows[t],0),(new_rows[t],2)):
            equal(weights[offset].mean(),saved["baseline_mean_blur_weight"],"DA weight")
            equal(weights[offset+1].mean(),saved["improved_mean_blur_weight"],"E4 weight")
            equal(rgb_difference(imgs[offset],imgs[offset+1]),saved["mean_absolute_rgb_effect_change"],"model RGB")
        for j,name in enumerate(("DA","E4")):
            equal(weights[j].mean(),comparison[t][name+"_previous_mean_blur_weight"],"old weight")
            equal(weights[j+2].mean(),comparison[t][name+"_adjusted_mean_blur_weight"],"new weight")
            equal(row[name+"_settings_rgb_change"],comparison[t][name+"_mean_absolute_rgb_settings_change"],"settings RGB")
        # For this lower-threshold candidate, blur must not increase anywhere.
        if params[1]["transition_normalized"][1] <= params[0]["transition_normalized"][0]:
            if any(np.any(weights[j+2]>weights[j]+1e-7) for j in (0,1)):
                raise ValueError("Lower transition unexpectedly increased blur")
        rows.append(row)
    averages = {k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("frame","timestamp_s")}
    out.mkdir(parents=True); figures=out/"figures"; figures.mkdir()
    write_csv(out/"per_frame.csv",rows)
    fig,axs=plt.subplots(2,2,figsize=(12,7),layout="constrained")
    colors=["#808080","#287c9f","#d18932","#699547"]
    for name,color in zip(names,colors):
        axs[0,0].plot(times,[r[name+"_mean_weight"] for r in rows],label=name,color=color)
        axs[0,1].plot(times,[r[name+"_fully_sharp_fraction"] for r in rows],label=name,color=color)
    axs[0,0].set(title="Mean blur weight (not image quality)",ylabel="Weight")
    axs[0,1].set(title="Pixels with exactly zero blur weight",ylabel="Fraction")
    for key,color in zip(("DA_settings_rgb_change","E4_settings_rgb_change"),colors[2:]):
        axs[1,0].plot(times,[r[key] for r in rows],label=key,color=color)
    for key,color in zip(("old_model_rgb_change","new_model_rgb_change"),colors[1:3]):
        axs[1,1].plot(times,[r[key] for r in rows],label=key,color=color)
    axs[1,0].set(title="Changing thresholds at fixed predictions",ylabel="Mean RGB difference (0-255)")
    axs[1,1].set(title="Changing DA to E4 at fixed thresholds",ylabel="Mean RGB difference (0-255)")
    for ax in axs.flat:
        ax.legend(fontsize=8); ax.grid(alpha=.2); ax.set_xlabel("Original time (s)")
    fig.suptitle("Application controls | observed output change, not a quality score")
    fig.savefig(figures/"application_diagnostics.png",dpi=180)
    plt.close(fig)
    overview=sorted(set(np.linspace(0,len(frames)-1,6).round().astype(int).tolist()))
    sheet(frames,(base,imp),(bv,iv),params,times,overview,figures/"overview.png")
    max_model=int(np.argmax([r["new_model_rgb_change"] for r in rows]))
    starts=sorted(set((max(0,min(len(frames)-6,max_model-2)),32)))
    windows=[list(range(s,min(s+6,len(frames)))) for s in starts]
    for win in windows:
        sheet(frames,(base,imp),(bv,iv),params,times,win,figures/f"consecutive_{win[0]}_{win[-1]}.png")
    sheet(frames,(base,imp),(bv,iv),params,times,[0,14],figures/"early_subject.png",roi=(100,60,450,200))
    summary={"status":"quantitative_complete_visual_pending","frames":len(frames),"parameters":{"old":params[0],"new":params[1]},
        "averages":averages,"largest_new_model_difference_frame":max_model,
        "source_metadata_sha256":{str(p.relative_to(ROOT)):checksum(p) for p in
           (run/"metadata.json",ref/"metadata.json",cache/"metadata.json",baseline/"metadata.json",improved/"metadata.json")},
        "analysis_script_sha256":checksum(Path(__file__)),
        "verification":"All 70x4 effects recomputed from saved predictions; two original CSVs and the settings CSV matched; input/artifact hashes and three movie timestamp sequences checked.",
        "definition":"Blur weight=1 fully blurred, 0 original RGB. RGB differences on raw uint8 effects are output changes, not accuracy/perceptual scores. Equal frames have equal image sizes.",
        "visual_evidence":{"overview":overview,"consecutive_windows":windows,"early_subject_frames":[0,14],
                           "early_subject_roi_xyxy":[100,60,450,200],"selection":"Uniform overview; maximum DA/E4-new difference window and fixed frames 32-37; two early subject crops. Post-hoc diagnostics."},
        "artifacts_sha256":{p.relative_to(out).as_posix():checksum(p) for p in out.rglob("*") if p.is_file()},
        "original_predictions_modified":False,"application_parameters_changed":False}
    write_json(out/"quantitative_review.json",summary)
    print(json.dumps({"frames":len(frames),"averages":averages,"max_new_model_frame":max_model},indent=2))


if __name__=="__main__":
    main()
