# SightBridge Spatial Awareness Demonstrator

A Python/OpenCV research prototype that combines:

- **YOLO11n** object detection and ByteTrack-based object IDs across consecutive frames.
- **Direction estimation** from horizontal position in the camera frame (far left / left / ahead / right / far right).
- **Depth Anything V2 Metric Outdoor Small** to estimate an approximate metric depth map and sample the depth inside each object's bounding box.
- **Temporal depth trend** to label an object as possibly approaching only when its estimated range decreases sufficiently over several observations.
- **Speech feedback** through `pyttsx3` on Windows.

## Important limitations

This is a demonstration only, not a certified mobility aid. It does **not** guarantee detection of every object, exact metric distance, or reliable collision warnings. YOLO11n recognizes classes covered by its pretrained dataset, not arbitrary objects. Monocular metric-depth estimates vary with scene, lighting, object size, and camera viewpoint. A moving camera can make a static object appear to approach. The `APPROACHING?` label is only a short-window depth trend, not time-to-collision or collision prediction. Never use this software for real navigation or in traffic.

## Requirements

- Windows 10/11, 64-bit.
- Recommended: Python 3.11 (64-bit), working internet for initial model/dependency downloads.
- NVIDIA GPU helps, but CPU mode is supported and may be slow.
- Webcam or a local video file.

## Run on Windows

1. Extract this folder to a writable location (for example `C:\Users\YourName\Documents\SightBridge_Spatial_Awareness`).
2. Install Python 3.11 from python.org if it is not installed. Enable **Add Python to PATH**.
3. Double-click `start-windows.bat`.
4. Wait for dependency installation and model downloads. First run may take several minutes and needs internet.
5. Webcam window opens when both models load. Use controls below.

The equivalent manual commands in PowerShell are:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

If PowerShell blocks activation, use the provided `.bat` launcher instead; do not change system-wide execution policy just for this project.

## Controls

- `Q`: quit
- `V`: toggle voice feedback
- `P`: pause/resume analysis
- Use another camera index or a saved clip, e.g. `python app.py --source 1` or `python app.py --source "C:\\demo\\traffic.mp4"`.
- `--depth-every 8` runs depth estimation less frequently on slow hardware. Depth/approach estimates become less temporally fresh.
- `--no-audio` disables speech.

## What the output means

- **Direction** is relative to the camera's current orientation, not a compass direction.
- **Distance** is sampled from a metric depth model's map within the inner part of the object's bounding box and is shown as an estimate. It is not independently calibrated against a range sensor.
- **Approaching?** means the estimated range decreased enough within a short time window. It can be wrong when the wearer/camera moves or depth estimation fluctuates.
- If depth inference fails, the app says distance is unavailable rather than making up a distance.

## How to explain the deep learning part

1. YOLO11n performs object detection and returns class, confidence, and bounding box.
2. ByteTrack maintains object identities across video frames.
3. Depth Anything V2 Metric Outdoor estimates per-pixel depth in a metric scale learned from synthetic outdoor training data.
4. The app samples a robust median in the central part of the detected box, smooths the distance by track ID, and observes the range trend over time.
5. A rule-based feedback layer converts the detected class, direction, approximate distance and short-window trend into spoken output.

This is pretrained-model inference and integration, not model training from scratch. Report latency/FPS measured on the actual laptop rather than inventing benchmark values.

## Troubleshooting

- **Camera failed to open:** close Teams/Zoom/Camera app; check Windows Settings → Privacy & security → Camera. Try `python app.py --source 1`.
- **Model download fails:** confirm internet access and run `python app.py` again. Both model files are downloaded on first use.
- **Very low FPS:** use `python app.py --depth-every 10`; close GPU-heavy apps. CPU-only depth inference may be slow.
- **Torch does not use NVIDIA GPU:** check the official PyTorch installation selector for a CUDA-compatible build matching the installed driver. CPU mode remains available.
- **Speech fails:** check Windows audio and reinstall dependencies; visual inference can continue without speech.
