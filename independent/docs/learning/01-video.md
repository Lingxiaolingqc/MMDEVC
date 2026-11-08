# Stage 1: what a decoded video frame is

Imagine a video as a flipbook. Its FPS is the number of pages shown each second; its width and height are the size of each page. A file's advertised page count can be wrong, so our inspector reads the pages and counts those it actually decodes.

A red pixel is `[255, 0, 0]` in RGB. OpenCV returns BGR, so the same red pixel arrives as `[0, 0, 255]`. We explicitly convert it before giving it to a model. A frame has shape `(height, width, 3)` and dtype `uint8`; each channel is an integer from 0 to 255.

In the three-frame test video, FPS is 12. Frame IDs are 0, 1, and 2; their nominal times are 0, 0.08333, and 0.16667 seconds. Requesting 100 frames still gives three real frames and a short-source warning.

Commands, run from `independent/`:

```powershell
.\.venv\Scripts\python.exe -m depth_project.cli inspect-video --input third_party/Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4 --required-frames 100 --preview-frames 3
```

Inspect the generated `runs/video-inspection/frame_0000.png` through `frame_0002.png` and `inspection.json`. Exit status 2 means the file contains fewer real frames than required; it does not mean frames should be duplicated.

Learning checkpoint for the user: explain why BGR/RGB conversion matters, why the decoded count is authoritative, and why a frame array is not yet a depth prediction. This checkpoint is offered, not recorded as personally mastered until the user explains it.
