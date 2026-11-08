"""Explicit submission inventory with frame-count and compliance checks."""
import json
import os
from pathlib import Path
import tempfile
import zipfile

from .cache import load_predictions
from .records import sha256_file
from .video import inspect_video


def build_submission(run_dir,output_zip,allow_short=False,report_path=None):
    run_dir=Path(run_dir);output_zip=Path(output_zip)
    if output_zip.exists():
        raise FileExistsError("Refusing to overwrite an existing submission archive")
    summary=json.loads((run_dir/'summary.json').read_text())
    count=summary['processed_frames']
    compliant=bool(summary['compliant_100_frames']) and count>=100
    if not compliant and not allow_short:
        raise ValueError("100 genuine frames are required; use explicit allow_short for a provisional preview")
    files=[(run_dir/'comparison.mp4','comparison.mp4'),(run_dir/'background-blur.mp4','background-blur.mp4'),(run_dir/'COMPLIANCE.txt','COMPLIANCE.txt'),(run_dir/'summary.json','summary.json')]
    for name in ['comparison.mp4','background-blur.mp4']:
        info=inspect_video(run_dir/name,required_frames=max(count,1))
        if info.decoded_frames!=count or abs(info.fps-summary['fps'])>0.01:
            raise ValueError(f"Submission video frame count/FPS mismatch: {name}")
    raw_paths=summary.get('raw_prediction_files',[])
    if not raw_paths:
        raise ValueError("Submission requires raw floating-point predictions")
    for index,raw in enumerate(raw_paths):
        raw=Path(raw)
        values,metadata=load_predictions(raw)
        if len(values)!=count:
            raise ValueError("Raw depth/video frame counts differ")
        files.append((raw,f'raw_predictions/{index:02d}_{raw.name}'))
    if report_path is not None:
        report=Path(report_path)
        if not report.is_file() or report.suffix.lower()!='.pdf':
            raise ValueError("Report must be an existing PDF")
        files.append((report,'report/independent-report.pdf'))
    inventory={'submission_ready':compliant,'processed_frames':count,'requested_frames':summary['requested_frames'],'fps':summary['fps'],'warning':None if compliant else 'Provisional: the official source is shorter than the required 100 frames. Obtain a longer course copy or TA clarification.','files':[{'path':name,'bytes':path.stat().st_size,'sha256':sha256_file(path)} for path,name in files]}
    output_zip.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_zip.parent,suffix='.zip',delete=False) as stream:
        temporary=Path(stream.name)
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED) as archive:
            for path,name in files:archive.write(path,arcname=name)
            archive.writestr('manifest.json',json.dumps(inventory,indent=2,allow_nan=False))
        with zipfile.ZipFile(temporary) as archive:
            bad=archive.testzip()
            if bad is not None:raise ValueError(f"Archive CRC failure: {bad}")
        os.replace(temporary,output_zip)
    finally:
        temporary.unlink(missing_ok=True)
    return inventory
