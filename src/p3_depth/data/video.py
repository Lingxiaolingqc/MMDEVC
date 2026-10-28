"""Shared RGB video cache built from decoded presentation timestamps."""

import math
import numpy as np


def output_size(width, height, max_long_edge):
    scale = min(1., max_long_edge / max(width, height))
    # Even sizes are suitable for common video encoders; record this once for all methods.
    return max(2, int(round(width * scale / 2)) * 2), max(2, int(round(height * scale / 2)) * 2)


def decode_video(path, target_fps=10., max_long_edge=640, max_frames=None):
    import av
    import cv2
    if target_fps <= 0 or max_long_edge < 14 or (max_frames is not None and max_frames < 2):
        raise ValueError('Positive FPS, edge >=14 and at least two frames required')
    frames, index = [], []
    origin, previous, next_sample = None, None, 0.
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate) if stream.average_rate else None
        source_duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        source_size = [stream.codec_context.height, stream.codec_context.width]
        width, height = output_size(source_size[1], source_size[0], max_long_edge)
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError('Missing PTS: remux with preserved timestamps before evaluation')
            absolute = float(frame.pts * frame.time_base)
            origin = absolute if origin is None else origin
            timestamp = absolute - origin
            if previous is not None and timestamp <= previous:
                raise ValueError('Non-increasing decoded timestamps')
            previous = timestamp
            if timestamp + 1e-8 < next_sample:
                continue
            rgb = frame.to_ndarray(format='rgb24')
            if rgb.shape[:2] != (source_size[0], source_size[1]):
                raise ValueError('Variable-resolution stream not supported')
            if rgb.shape[:2] != (height, width):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
            index.append({'cache_index': len(frames)-1, 'source_frame_index': frame_index,
                          'timestamp_s': timestamp, 'pts': frame.pts, 'time_base': str(frame.time_base)})
            next_sample = (math.floor((timestamp + 1e-8) * target_fps) + 1) / target_fps
            if max_frames is not None and len(frames) >= max_frames:
                break
    if len(frames) < 2:
        raise ValueError('Need at least two decoded frames for video evaluation')
    timestamps = [r['timestamp_s'] for r in index]
    return np.stack(frames), index, {'source_fps_nominal': source_fps, 'source_duration_s': source_duration, 'source_size_hw': source_size,
        'output_size_hw': [height, width], 'requested_fps': target_fps,
        'effective_fps': (len(frames)-1)/(timestamps[-1]-timestamps[0]), 'sampled_duration_s': timestamps[-1]-timestamps[0],
        'max_frames': max_frames, 'timestamp_rule': 'first decoded frame at or after each target interval; original PTS retained',
        'spatial_resize': 'aspect-preserving INTER_AREA, even dimensions, no upsampling', 'orientation': 'decoded pixels; no implicit rotation correction'}
