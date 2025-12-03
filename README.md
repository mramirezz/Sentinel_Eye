# Sentinel Eye - Industrial Camera Quality & Stability Monitor

> **Sample Data & Outputs:** [Google Drive - Test Videos, Models & Results](https://drive.google.com/drive/folders/1iVfWtvqK9Hc_ZPepfytH61CzicCTpE0d?usp=sharing)  
> **Contains: test videos (`data/`), trained models (`models/`), processed outputs (`outputs/`), and execution logs (`logs/`)**

---

## Executive Summary

Sentinel Eye is a real-time computer vision system designed for industrial camera monitoring in harsh environments (mining, construction, remote facilities). The system performs three critical functions:

1. **Image Quality Control (QC)**: Multi-factor quality scoring (0-100) evaluating sharpness, occlusion, lighting, and lens cleanliness
2. **Camera Stability Analysis**: Vibration detection and self-healing ROI tracking that compensates for camera movement
3. **Object Detection**: YOLOv8-based detection with TensorRT acceleration for real-time performance

**Key Performance Metrics:**
- Processing Speed: 30+ FPS (effective) on NVIDIA GPU with frame_skip=2
- Quality Score: Sub-metric breakdown (sharpness, occlusion, lighting, cleanliness)
- Stability Tracking: <0.5px drift detection threshold with optical flow
- Detection: TensorRT FP16 optimization (~3x faster than PyTorch)

**Technology Stack:**
- YOLOv8l with TensorRT (FP16)
- OpenCV + PyTorch (GPU-accelerated QC)
- Sparse Optical Flow (Lucas-Kanade) for ROI tracking
- Docker containerization with NVIDIA GPU support

---

## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- Docker Compose v3.8+

### Installation

1. Clone repository:
```bash
git clone https://github.com/mramirezz/Sentinel_Eye.git
cd Sentinel_Eye
```

2. Build Docker image:
```bash
docker compose build
```

### Usage Examples

#### Process all videos in data folder:
```bash
docker compose run --rm sentinel-eye python src/main.py --no-display
```

#### Process specific video:
```bash
docker compose run --rm sentinel-eye python src/main.py --video data/earthquake2.mp4 --no-display
```

#### Use custom configuration:
```bash
docker compose run --rm -v ${PWD}/config_custom.yaml:/app/config.yaml sentinel-eye python src/main.py --video data/video_2_vibration.mp4 --no-display
```

#### Select initial ROI for a video:
```bash
python select_roi.py data/earthquake2.mp4
```
This writes ROI coordinates to `initial_rois.json` for self-healing reference.

**IMPORTANT**: **This script is NOT part of the core processing pipeline**. It's a **helper tool for interactive ROI selection** during initial setup. The main processing (`main.py`) runs entirely in Docker and reads the pre-configured ROIs from `initial_rois.json`.

**Why not in Docker?** Docker containers don't have GUI access, so this tool must run locally with display support.

**Requirements for local execution**:
- **Python libraries**: `opencv-python` (for GUI), `numpy`
- **Option 1 (Recommended)**: Use Conda environment
  ```bash
  conda activate base  # or any conda env with opencv-python installed
  pip install opencv-python  # if not already installed
  python select_roi.py data/video.avi
  ```
- **Option 2**: Create virtual environment with Python ≤3.11 (Python 3.13+ has numpy 1.x compatibility issues)
  ```bash
  conda create -n sentinel_roi python=3.11 opencv-python -y
  conda activate sentinel_roi
  python select_roi.py data/video.avi
  ```

### Outputs

**Video Output** (`outputs/<video_name>_processed.mp4`):
- Top-left: QC Score overlay with sub-metrics + Processing/Effective FPS
- Top-right: Stability status (STABLE/VIBRATING) with X/Y drift
- Center: Tracked feature points with motion vectors
- ROI visualization: Green box (current adjusted ROI), cyan markers (tracked features)

**Metrics Plot** (`outputs/<video_name>_metrics.png`):
- Top-left: 2D camera drift trajectory (spatial visualization)
- Top-right: QC Score temporal evolution with quality thresholds
- Bottom-left: X/Y drift over time (vibration patterns)
- Bottom-right: Summary statistics (avg/min/max QC, FPS, total frames)

**Logs** (`logs/sentinel_eye_<timestamp>.log`):
- Detailed execution trace with module-level performance metrics

---

## Configuration Parameters

### Video Processing (`video`)
- **`input_path`**: Directory containing input videos (default: `data/`)
- **`output_path`**: Directory for processed outputs (default: `outputs/`)
- **`target_resolution`**: `[width, height]` - Target resolution for frame processing (default: `[640, 480]`)
  - **Only used when `optimization.enable_resize=true`**
  - **Ignored when `enable_resize=false`** (processes at original resolution)
- **`frame_skip`**: `int` - Process every Nth frame (default: `2` for 30+ effective FPS)

### Quality Control (`qc_score`)
- **`weights`**: Sub-metric weights (must sum to 1.0)
  - `sharpness`: 0.25 (Laplacian variance - focus quality)
  - `occlusion`: 0.25 (edge density analysis - blockage detection)
  - `lighting`: 0.25 (histogram brightness/contrast)
  - `cleanliness`: 0.25 (spot detection - rain/dust/dirt)
- **`thresholds`**: Quality status levels
  - `excellent`: 80+ (green indicator)
  - `good`: 60-79 (yellow)
  - `warning`: 40-59 (orange)
  - `critical`: <40 (red)
- **`hyperparameters`**: Algorithm sensitivity tuning (configure strictness for harsh environments)
  - **Sharpness**:
    - `sharpness_divisor`: Laplacian variance normalization (default: `6.0`, higher = stricter)
  - **Occlusion**:
    - `min_edge_density`: Minimum edge density threshold (default: `0.03`)
    - `edge_density_multiplier`: Penalty for low edge density (default: `1500`)
  - **Lighting**:
    - `ideal_brightness`: Target mean brightness 0-255 (default: `120`)
    - `brightness_divisor`: Brightness tolerance (default: `1.0`, lower = stricter)
    - `contrast_multiplier`: Contrast score multiplier (default: `1.8`)
  - **Cleanliness**:
    - `spot_threshold`: Spot detection sensitivity (default: `14`, lower = more sensitive)
    - `spot_penalty_multiplier`: Penalty for dust/rain spots (default: `900`, higher = stricter)

### Stability Tracking (`stability`)
- **`history_size`**: `int` - Sliding window for drift history (default: `30` frames)
- **`vibration_threshold`**: `float` - Movement threshold in pixels for vibration detection (default: `0.8px`)
- **`enable_self_healing`**: `bool` - Enable dynamic ROI adjustment (default: `true`)
- **`initial_roi_file`**: Path to JSON with pre-defined ROIs per video

### Optimization (`optimization`)
- **`use_gpu`**: `bool` - Enable CUDA acceleration (default: `true`)
- **`enable_resize`**: `bool` - Downscale frames to `video.target_resolution` before QC/Stability processing (default: `true`)
  - **When `true`**: Frames resized to `target_resolution` for faster processing (~2x speedup)
  - **When `false`**: Process at original video resolution (slower, higher quality for QC/Stability)
  - **Note**: YOLO always resizes independently to `yolo_imgsz` regardless of this setting

### Detection (`detection`)
- **`use_yolo`**: `bool` - Enable YOLOv8 detector (default: `true`)
- **`yolo_model`**: `str` - Model size: `n`/`s`/`m`/`l`/`x` (default: `l`)
  - **Trade-off**: `l` provides best accuracy/speed balance for 30+ FPS target with frame_skip=2
- **`yolo_imgsz`**: `int` - YOLO input resolution, must be multiple of 32 (default: `640`)
  - **Recommended**: `640` (speed), `1024` (balanced), `1280` (precision)
- **`confidence_threshold`**: `float` - Detection confidence filter (default: `0.5`)
- **`show_yolo`**: `bool` - Draw detection bounding boxes on output video

### Logging (`logging`)
- **`level`**: Log verbosity (`DEBUG`/`INFO`/`WARNING`/`ERROR`)
- **`save_plots`**: `bool` - Generate metrics plot PNG
- **`save_videos`**: `bool` - Save processed video output

---

## Technical Architecture

### Module 1: Image Quality Control

**Objective**: Real-time assessment of image health across four dimensions.

**Algorithm Details:**

1. **Sharpness (25% weight)**
   - Method: Laplacian variance on grayscale
   - GPU-accelerated using PyTorch convolution
   - Normalization: variance / sharpness_divisor (configurable, default: 6.0)
   - Kernel:
     ```
     [[ 0,  1,  0],
      [ 1, -4,  1],
      [ 0,  1,  0]]
     ```

2. **Occlusion Detection (25% weight)**
   - Method: Canny edge density analysis
   - Edge threshold: 50/150 (low/high)
   - Scoring: Configurable min_edge_density threshold (default: 0.03)
   - Penalty below threshold: edge_density × edge_density_multiplier (default: 1500)

3. **Lighting Assessment (25% weight)**
   - Method: Histogram statistics (mean/std on GPU)
   - Target: mean brightness ~ideal_brightness (configurable, default: 120)
   - Tolerance: brightness_divisor (default: 1.0), contrast_multiplier (default: 1.8)
   - Score combines brightness deviation and contrast level

4. **Lens Cleanliness (25% weight)**
   - Method: Median blur difference (5x5 kernel)
   - Detects spots/dirt via intensity difference
   - Threshold: spot_threshold (configurable, default: 14, lower = more sensitive)
   - Score penalty: spot_density × spot_penalty_multiplier (default: 900)

**GPU Optimization:**
- PyTorch tensor operations on CUDA for all metrics
- Single CPU↔GPU transfer per frame (grayscale conversion)
- Avoids NumPy overhead in critical path

**Output:** Overall score (0-100) and per-metric breakdown.

---

### Module 2: Stability Analysis & Self-Healing ROI

**Objective**: Detect camera vibration and dynamically adjust ROI to track the same physical area despite camera movement.

**Algorithm: Sparse Optical Flow (Lucas-Kanade)**

1. **Feature Detection**
   - Method: `cv2.goodFeaturesToTrack()` with Shi-Tomasi corner detector
   - Parameters:
     - `maxCorners`: 100 (maximum corners to detect, returns best quality)
     - `qualityLevel`: 0.01 (minimum quality = 1% of best corner, lower = more permissive)
     - `minDistance`: 7px (minimum distance between detected corners)
     - `blockSize`: 7 (window size for corner detection matrix, 7×7 pixels)
   - **Tuning Guide:**
     - Uniform textures (sky, walls): use `qualityLevel=0.01` to find weaker features
     - High detail scenes: increase to `qualityLevel=0.1` for better quality
     - Small ROIs: reduce `minDistance` to 5-7px for denser features
   - Initial features detected within ROI bounds

2. **Feature Tracking**
   - Method: Pyramidal Lucas-Kanade optical flow
   - Window size: 21×21
   - Pyramid levels: 3
   - Termination: 30 iterations or ε=0.01

3. **Motion Estimation**
   - Calculate mean displacement vector (dx, dy) from tracked features
   - Accumulate offset: `camera_offset_x += dx`, `camera_offset_y += dy`
   - ROI adjustment: `new_roi = (x + offset_x, y + offset_y, w, h)`

4. **Vibration Detection**
   - Sliding window (10 frames) of movement magnitudes
   - Threshold: Average movement > 0.3px triggers "VIBRATING" status

5. **Feature Regeneration**
   - Minimum features required: 3 (tracking continues with 3+ features)
   - When features drop to 2 or fewer, auto-recalculate within adjusted ROI
   - Prevents tracking failure over long sequences
   - Auto-recovery mechanism for low-texture scenes

**Resolution Handling:**
- Dynamically detects frame dimensions (no hardcoded 1280×720)
- Bounds checking ensures ROI stays within frame after adjustment

**Output:** Displacement (dx, dy), cumulative offset, vibration flag, adjusted ROI coordinates.

---

### Module 3: Object Detection (YOLOv8 + TensorRT)

**Objective**: Real-time object detection with GPU-optimized inference.

**Model Pipeline:**

1. **Model Selection**
   - YOLOv8l (large) chosen for accuracy/speed balance
   - With `frame_skip=2`, achieves 30+ effective FPS
   - Smaller models (`n`/`s`) available for constrained hardware

2. **Export Pipeline: PyTorch → ONNX → TensorRT**

   **Step 1: PyTorch to ONNX**
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8l.pt')
   model.export(format='onnx', opset_version=13, dynamic=True)
   ```
   - Opset 13 for broad compatibility
   - Dynamic batch size support

   **Step 2: ONNX to TensorRT**
   ```python
   model.export(format='engine', device=0, half=True, imgsz=640)
   ```
   - **FP16 precision**: Halves memory, ~3× faster on Tensor Cores
   - **Fixed input shape**: `(1, 3, 640, 640)` for optimal kernel selection
   - **Engine naming**: `yolov8l_640.engine` (includes imgsz for multi-size support)

3. **Inference Optimization**
   - TensorRT engine cached in `models/` directory
   - First run: 5-10 min compilation (GPU-specific optimization)
   - Subsequent runs: <1s engine load
   - INT64→INT32 automatic casting (safe for detection indices)

**Image Size Handling:**

- **Original Video**: 1280×720 (or any resolution)
- **Processing Path**:
  1. Optional resize to `target_resolution` (640×480) if `enable_resize=true`
  2. YOLO resizes internally to `yolo_imgsz` (640×640 square)
  3. Detections mapped back to original resolution coordinates

**Warning System:**
- If `enable_resize=true` and `max(target_resolution) < yolo_imgsz × 0.5`:
  - Logs warning about upscaling quality loss
  - Suggests: increase `target_resolution`, decrease `yolo_imgsz`, or disable resize

**Output:** Bounding boxes (x, y, w, h), class labels, confidence scores.

---

## Performance Optimization Strategies

### 1. GPU Computation (PyTorch vs NumPy)
- **QC Metrics**: PyTorch tensor operations on CUDA
- **Rationale**: Avoid CPU↔GPU transfers for simple operations
- **Speedup**: ~2× faster than NumPy + OpenCV for Laplacian/stats

### 2. Frame Skip Strategy
- **Configuration**: `frame_skip=2` processes every 2nd frame
- **Effect**: 2× effective throughput (e.g., 30 FPS → 60 FPS effective)
- **Trade-off**: Temporal resolution vs processing speed

### 3. Resolution Scaling
- **QC/Stability**: Process at 640×480 (`enable_resize=true`)
- **YOLO**: Fixed 640×640 input (independent of resize setting)
- **Output**: Always rendered at original resolution with scaled overlays

### 4. TensorRT Optimization
- **FP16 Precision**: Half-precision computation on Tensor Cores
- **Graph Optimization**: Fused operations, kernel auto-tuning
- **Memory**: ~50% reduction vs FP32

### 5. Optical Flow Efficiency
- **Sparse tracking**: 50 features vs dense flow (millions of pixels)
- **Pyramidal LK**: Multi-scale for large displacements without exhaustive search

---

## Production Monitoring 
### What to Monitor

**1. Camera Health Alerts**
- **QC Score drops below 60**: Camera needs cleaning or has obstruction
- **Vibration detected**: Physical inspection needed (loose mount, mechanical issue)
- **Sharpness declining**: Lens degradation or focus drift

**2. System Performance**
- **FPS below 15**: System overloaded, reduce resolution or frame_skip
- **Processing delays**: Check GPU availability

### Easy Monitoring Setup

**Option 1: Log Files (Simplest)**
```bash
# Check logs for warnings
tail -f logs/sentinel_eye.log | grep -E "WARNING|ERROR|Low QC"

# Daily summary
grep "FINAL STATISTICS" logs/*.log
```

**Option 2: CSV Export (Spreadsheet-Friendly)**
- Add simple CSV writer to save: timestamp, video, QC score, vibration flag
- Import to Excel/Google Sheets for weekly reports

**Option 3: Basic Dashboard (Recommended)**
- Use free tools like Grafana Cloud (free tier: 10k metrics)
- Parse logs and send to Prometheus
- Set alerts: email when QC < 60 for >5 minutes

### Model Updates
- Keep models in `models/` folder
- Docker rebuild picks up new `.engine` files automatically
- No code changes needed, just replace model file

### Multi-Camera Deployment
- Run one Docker container per camera
- Different `config.yaml` for each camera's ROI
- Centralize logs to shared network folder for easy monitoring

---

## Code Quality & Design

- **Modular**: 3 independent modules (QC, Stability, Detection)
- **Configurable**: All parameters in `config.yaml`, no hardcoded values
- **GPU-Optimized**: PyTorch + TensorRT for real-time performance
- **Type-Safe**: Python type hints throughout
- **Logged**: All warnings/errors captured for debugging

---

**Author**: Mauricio Ramirez  
**Repository**: [github.com/mramirezz/Sentinel_Eye](https://github.com/mramirezz/Sentinel_Eye)

For technical questions or collaboration inquiries, open a GitHub issue.


