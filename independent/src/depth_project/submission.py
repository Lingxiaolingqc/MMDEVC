"""Explicit submission inventory with frame-count and compliance checks."""
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

from .cache import load_predictions
from .records import sha256_file
from .video import inspect_video
from .render import comparison_canvas_hw


def build_submission(run_dir,output_zip,allow_short=False,report_path=None,supplemental_files=None):
    run_dir=Path(run_dir);output_zip=Path(output_zip)
    if output_zip.exists():
        raise FileExistsError("Refusing to overwrite an existing submission archive")
    summary=json.loads((run_dir/'summary.json').read_text())
    count=summary['processed_frames']
    frame_ids=summary.get('frame_ids')
    if frame_ids!=list(range(count)):
        raise ValueError("Submission frame IDs must match the first processed source frames")
    output_hw=summary.get('output_hw')
    if not isinstance(output_hw,list) or len(output_hw)!=2 or any(not isinstance(value,int) or value<1 for value in output_hw):
        raise ValueError("Submission output shape must be explicit positive HW")
    source_sha=summary.get('source_sha256')
    if not isinstance(source_sha,str) or len(source_sha)!=64:
        raise ValueError("Submission source SHA-256 is missing")
    compliant=bool(summary['compliant_100_frames']) and count==100 and frame_ids==list(range(100))
    if not compliant and not allow_short:
        raise ValueError("100 genuine frames are required; use explicit allow_short for a provisional preview")
    files=[(run_dir/'comparison.mp4','comparison.mp4'),(run_dir/'background-blur.mp4','background-blur.mp4'),(run_dir/'COMPLIANCE.txt','COMPLIANCE.txt'),(run_dir/'summary.json','summary.json')]
    for name in ['comparison.mp4','background-blur.mp4']:
        info=inspect_video(run_dir/name,required_frames=max(count,1))
        if info.decoded_frames!=count or abs(info.fps-summary['fps'])>0.01:
            raise ValueError(f"Submission video frame count/FPS mismatch: {name}")
        expected_canvas=comparison_canvas_hw(output_hw,4 if name=='comparison.mp4' else 2)
        if (info.height,info.width)!=expected_canvas:
            raise ValueError(f"Submission rendered video dimensions mismatch: {name}")
    raw_paths=summary.get('raw_prediction_files',[])
    if len(raw_paths)!=3:
        raise ValueError("Submission requires baseline, video-reference, and fused raw predictions")
    for index,raw in enumerate(raw_paths):
        raw=Path(raw)
        values,metadata=load_predictions(raw)
        if len(values)!=count:
            raise ValueError("Raw depth/video frame counts differ")
        if metadata.get('source_sha256')!=source_sha:
            raise ValueError("Raw prediction source identity differs from the summary")
        if metadata.get('frame_ids')!=frame_ids:
            raise ValueError("Raw prediction frame IDs differ from the summary")
        if list(values.shape[1:])!=output_hw or metadata.get('output_hw')!=output_hw:
            raise ValueError("Raw prediction output shape differs from the summary")
        if metadata.get('kind')!='relative':
            raise ValueError("Video raw prediction kind must be relative")
        if index==0 and metadata.get('key')!=summary.get('baseline_cache_key'):
            raise ValueError("Raw baseline cache key differs from the summary")
        if index==1 and metadata.get('key')!=summary.get('video_reference',{}).get('key'):
            raise ValueError("Raw video-reference cache key differs from the summary")
        if index==2:
            if metadata.get('parent_baseline_key')!=summary.get('baseline_cache_key'):
                raise ValueError("Fused raw prediction has the wrong parent baseline key")
            if metadata.get('configuration_sha256')!=summary.get('configuration_sha256'):
                raise ValueError("Fused raw prediction configuration differs from the summary")
        files.append((raw,f'raw_predictions/{index:02d}_{raw.name}'))
    if report_path is not None:
        report=Path(report_path)
        if not report.is_file() or report.suffix.lower()!='.pdf':
            raise ValueError("Report must be an existing PDF")
        files.append((report,'report/independent-report.pdf'))
    names = {name.casefold() for _, name in files} | {'manifest.json'}
    for name, source in (supplemental_files or {}).items():
        archive_path = PurePosixPath(name)
        if not name or archive_path.is_absolute() or '..' in archive_path.parts or '\\' in name or ':' in name:
            raise ValueError('Supplemental archive path must be a safe relative path')
        canonical = archive_path.as_posix()
        if canonical.casefold() in names:
            raise ValueError('Duplicate supplemental archive path')
        source = Path(source)
        if not source.is_file():
            raise ValueError('Supplemental file must exist')
        files.append((source, canonical))
        names.add(canonical.casefold())
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
