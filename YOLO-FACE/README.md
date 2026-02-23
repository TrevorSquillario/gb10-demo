# YOLO Face Tracking with BoT-SORT Re-ID

This application provides real-time face detection and tracking using YOLO and the BoT-SORT tracker with Re-Identification (Re-ID) support to "keep tabs" on people across frames.

## Features

- **YOLO Face Detection**: Uses `yolo26t.pt` model for accurate face detection
- **BoT-SORT Tracker**: High-accuracy tracking with motion prediction and Kalman filtering
- **Persistent IDs**: `persist=True` ensures faces maintain the same ID across frames
- **Re-Identification**: Recognizes faces that leave and return to the scene
- **GPU Acceleration**: CUDA support for real-time processing
- **Web Interface**: Flask-based web UI with live statistics
- **Multi-Source Support**: Webcam, video files, RTSP/HLS streams

## Quick Start

### Test Your Video Source First

Before running the full application, test that your video source works:

```bash
# Test webcam
python examples/test_video_source.py 0

# Test local video file
python examples/test_video_source.py /path/to/video.mp4

# Test RTSP stream
python examples/test_video_source.py rtsp://192.168.1.100:554/stream
```

This will verify connectivity and show video properties (resolution, FPS, etc.).

### Using Docker Compose

```bash
docker compose up -d
```

Access the stream at: http://localhost:5000

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Configuration

### Environment Variables

- `MODEL_PATH`: Path to YOLO model (default: `yolo26t.pt`)
- `VIDEO_SOURCE`: Video source - webcam index, file path, or RTSP URL
- `DETECT_STREAM_URL`: HLS/HTTP stream URL requiring ffmpeg conversion
- `SKIP_FRAMES`: Process every Nth frame (default: 1)

### Example Configurations

**USB Webcam:**
```bash
# Default webcam (device 0)
export VIDEO_SOURCE=0
python app.py

# Specific webcam device
export VIDEO_SOURCE=1
python app.py
```

**Local Video File:**
```bash
export VIDEO_SOURCE=/path/to/video.mp4
python app.py

# Relative path
export VIDEO_SOURCE=./videos/sample.mp4
python app.py
```

**Direct RTSP Stream:**
```bash
# IP Camera or RTSP source
export VIDEO_SOURCE=rtsp://192.168.1.100:554/stream
python app.py

# RTSP with authentication
export VIDEO_SOURCE=rtsp://username:password@192.168.1.100:554/stream
python app.py
```

**HLS/HTTP Stream (via ffmpeg):**
```bash
# For HLS streams that need conversion
export DETECT_STREAM_URL=https://example.com/stream.m3u8
python app.py
```

## Advanced: Enabling Re-Identification (Re-ID)

Re-ID allows the tracker to recognize faces that leave the frame and return later by using visual embeddings instead of just motion tracking.

### Step 1: Locate the BoT-SORT Configuration

The `botsort.yaml` configuration file is located in the Ultralytics package:

```bash
# Find the Ultralytics config directory
python -c "from ultralytics.cfg import get_config_dir; print(get_config_dir())"

# Navigate to trackers directory
cd $(python -c "from ultralytics.cfg import get_config_dir; print(get_config_dir())")/trackers
```

### Step 2: Enable Re-ID in botsort.yaml

Edit `botsort.yaml` and set `with_reid: True`:

```yaml
tracker_type: botsort

# BoT-SORT settings
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 30
match_thresh: 0.8

# Enable Re-Identification
with_reid: True  # ← Set this to True

# CMC method for camera motion compensation
cmc_method: sparseOptFlow

# Fuse scores during association
fuse_score: True
```

### Step 3: Understanding Re-ID

When `with_reid: True`:

1. **Visual Embeddings**: Each detected face gets a visual "fingerprint" (embedding)
2. **Appearance Matching**: When a face returns, it's matched by appearance, not just location
3. **Persistent Identity**: IDs are maintained even after faces leave and return
4. **Better Occlusion Handling**: Handles temporary occlusions better

### Re-ID Performance Considerations

- **Computation**: Re-ID adds ~10-20% overhead but significantly improves tracking
- **Memory**: Stores embeddings for active tracks
- **Accuracy**: Best for scenarios where people leave and return to frame

## Tracking Statistics

The application tracks and logs:

- **Current Faces**: Number of faces in the current frame
- **Total Tracked**: Total unique face IDs seen during session
- **Track IDs**: Unique identifier for each person
- **Last Seen**: Frame number when each face was last detected

Access real-time statistics at: http://localhost:5000/stats

## How It Works

### Video Source Detection

The application automatically detects the type of video source and uses the appropriate cv2 backend:

| Source Type | Detection | cv2 Backend | Example |
|-------------|-----------|-------------|---------|
| **USB Webcam** | Numeric value | `CAP_V4L2` (Linux) or default | `VIDEO_SOURCE=0` |
| **RTSP Stream** | Starts with `rtsp://` | `CAP_FFMPEG` | `VIDEO_SOURCE=rtsp://cam.local/stream` |
| **Local File** | File path | Default | `VIDEO_SOURCE=/path/video.mp4` |
| **HTTP Stream** | Starts with `http://` or `https://` | `CAP_FFMPEG` | `VIDEO_SOURCE=http://stream.url` |
| **HLS via ffmpeg** | Set `DETECT_STREAM_URL` | `CAP_FFMPEG` (local RTSP) | `DETECT_STREAM_URL=https://...m3u8` |

### 1. Detection
```python
model = YOLO('yolo26t.pt')
```

### 2. Tracking with BoT-SORT
```python
results = model.track(
    source=video_source,
    tracker="botsort.yaml",
    persist=True,  # Maintain IDs across frames
    conf=0.3,
    iou=0.3
)
```

### 3. Processing Results
```python
for det in results[0].boxes.data:
    track_id = int(det[4])
    tracked_faces.add(track_id)
    print(f"Tracking Face ID: {track_id}")
```

## BoT-SORT vs ByteTrack

| Feature | BoT-SORT | ByteTrack |
|---------|----------|-----------|
| **Accuracy** | Higher | Good |
| **Re-ID Support** | Yes | No |
| **Speed** | Slightly slower | Faster |
| **Use Case** | Long-term tracking | Real-time performance |

## Troubleshooting

### Video Source Issues

**Webcam Not Found:**
```bash
# List available video devices on Linux
ls -l /dev/video*

# Test webcam with ffplay
ffplay /dev/video0

# Try different device index
export VIDEO_SOURCE=1
```

**RTSP Stream Connection Failed:**
```bash
# Test RTSP stream with ffplay
ffplay rtsp://192.168.1.100:554/stream

# Check network connectivity
ping 192.168.1.100

# Increase retry attempts in code if needed
# Verify RTSP URL format and credentials
```

**Local File Not Found:**
```bash
# Check file exists
ls -la /path/to/video.mp4

# Use absolute path
export VIDEO_SOURCE=/home/user/videos/sample.mp4

# Or mount directory in Docker
# See compose.yaml volumes section
```

**HTTP/HLS Stream Issues:**
```bash
# For HLS streams (.m3u8), use DETECT_STREAM_URL
export DETECT_STREAM_URL=https://example.com/stream.m3u8

# For direct HTTP streams, try VIDEO_SOURCE
export VIDEO_SOURCE=http://example.com/stream

# Test stream availability
curl -I https://example.com/stream.m3u8
```

### Model Not Found
Download or provide the `yolo26t.pt` model file. You can also use community face detection models like `yolov8n-face.pt`.

### Re-ID Not Working
1. Verify `with_reid: True` in `botsort.yaml`
2. Check that Ultralytics version supports Re-ID (>=8.0.0)
3. Ensure sufficient GPU memory for embedding computation

### Low FPS
1. Increase `SKIP_FRAMES` to process fewer frames
2. Reduce `max_det` in the code
3. Use smaller YOLO model variant
4. Disable Re-ID if not needed

## License

This project uses Ultralytics YOLO which is licensed under AGPL-3.0.

## References

- [Ultralytics Tracking Docs](https://docs.ultralytics.com/modes/track/)
- [BoT-SORT Paper](https://arxiv.org/abs/2206.14651)
- [YOLO Documentation](https://docs.ultralytics.com/)
