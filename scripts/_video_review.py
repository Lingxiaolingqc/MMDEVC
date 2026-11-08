"""Shared read-only video evidence helpers for per-Part analysis scripts."""
import csv
import json
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from _common import checksum
from _media import checked_comparison
from render_depth_video import colorize


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def verify_preview(cache, run, times, limits):
    visual = run/'visualizations'
    if not visual.is_dir():
        raise FileNotFoundError('Saved preview required: '+str(visual))
    checked_comparison(cache, [run], visual)
    m = read_json(visual/'metadata.json')
    if [m['inputs'][0]['low'], m['inputs'][0]['high']] != limits:
        raise ValueError('Preview scale differs from fixed whole-clip range')
    with av.open(str(visual/'comparison.mp4')) as video:
        pts = [float(f.pts*f.time_base) for f in video.decode(video=0)]
    if len(pts) != len(times) or not np.allclose(pts, np.asarray(times)-times[0], rtol=0, atol=2e-6):
        raise ValueError('Saved preview timestamps differ')
    return {'metadata_sha256': checksum(visual/'metadata.json'), 'frames': len(pts), 'limits': limits}


def contact_sheet(frames, predictions, masks, limits, labels, times, indices, path, roi=None):
    """Render matched raw arrays, with one fixed scale per method for every frame."""
    h, w = frames.shape[1:3]
    x0, y0, x1, y1 = roi or (0, 0, w, h)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('ROI outside frame')
    scale = 1 if roi else .5
    pw, ph = round((x1-x0)*scale), round((y1-y0)*scale)
    cols = 1+len(predictions)
    sheet = Image.new('RGB', (cols*pw, len(indices)*(ph+30)), 'white')
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('arial.ttf', 15)
    except OSError:
        font = ImageFont.load_default(size=15)
    for row, t in enumerate(indices):
        images = [frames[t]] + [colorize(d[t], v[t], *span) for d, v, span in zip(predictions, masks, limits)]
        for col, (label, img) in enumerate(zip(['RGB', *labels], images)):
            crop = Image.fromarray(img[y0:y1, x0:x1]).resize((pw, ph), Image.Resampling.LANCZOS)
            top = row*(ph+30)
            sheet.paste(crop, (col*pw, top+30))
            draw.text((col*pw+3, top+5), f'{label} | f{t} {times[t]:.3f}s', fill='black', font=font)
    sheet.save(path)


def selected_windows(pair_rows, count, size=6):
    ranked = sorted(pair_rows, key=lambda r: r['capped_normalized_l1_full_reference'], reverse=True)
    starts = []
    for row in ranked:
        start = max(0, min(count-size, row['frame']-2))
        if all(abs(start-other) >= size for other in starts):
            starts.append(start)
        if len(starts) == 2:
            break
    starts.append(max(0, min(count-size, count//2-3)))
    return [list(range(start, min(start+size, count))) for start in sorted(set(starts))]
