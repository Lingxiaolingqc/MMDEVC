"""A minimal depth-aware background-blur application."""
import cv2
import numpy as np

from .models import validate_rgb
from .render import normalize_depth


def background_blur(rgb,relative_depth,config):
    validate_rgb(rgb)
    if rgb.shape[:2]!=relative_depth.shape:
        raise ValueError("Application RGB/depth sizes differ")
    normalized=normalize_depth(relative_depth,config['display_range'])
    threshold=float(config.get('focus_threshold',0.6))
    transition=float(config.get('transition',0.1))
    if transition<=0:
        raise ValueError("Focus transition must be positive")
    # DA V2 relative output increases toward nearer objects.
    foreground=np.clip((normalized-(threshold-transition))/(2*transition),0,1)[:,:,None]
    blurred=cv2.GaussianBlur(rgb,(0,0),float(config.get('blur_sigma',8)))
    return np.rint(foreground*rgb+(1-foreground)*blurred).astype(np.uint8)
