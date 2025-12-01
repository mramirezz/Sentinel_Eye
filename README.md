# Sentinel Eye 🎥

Sistema avanzado de monitoreo de calidad de video y detección de objetos con **TensorRT**, **ORB feature tracking** y optimizaciones GPU.

## 🚀 Características

### Módulo 1: Quality Control (QC) Score
- **Sharpness Detection**: Análisis de nitidez con Laplacian variance
- **Occlusion Detection**: Detección de obstrucciones de cámara
- **Lighting Analysis**: Evaluación de condiciones de iluminación
- **Cleanliness Check**: Detección de suciedad en lente
- **Score ponderado**: Sistema configurable de pesos por métrica

### Módulo 2: Stability Analysis
- **Camera Vibration Detection**: Detección de vibraciones con optical flow
- **Movement Tracking**: Análisis de movimiento de cámara
- **Feature Caching**: Optimización con cache de features cada 10 frames
- **Self-Healing**: Recuperación automática de errores

### Módulo 3: Motion Detection
- **YOLOv8n + TensorRT**: Detección de objetos optimizada con TensorRT FP16
- **ORB Feature Tracking**: Seguimiento de ROI con ORB features (persistencia en movimiento de cámara)
- **Adaptive ROI**: ROI que persigue objetos detectados
- **Background Subtraction**: Detección de movimiento por sustracción de fondo

## ⚡ Optimizaciones

### GPU Acceleration
- **TensorRT Engine**: YOLOv8n compilado con FP16 para inferencia ultra-rápida (8-14ms)
- **PyTorch CUDA**: QC Score con tensores GPU (Laplacian, mean, std)
- **Feature Caching**: Vibration features recalculadas cada 10 frames

### Performance
- **Frame Skip**: Procesamiento cada 2 frames con duplicación para mantener 30 FPS
- **Frame Duplication**: Frames intermedios duplicados para video suave
- **Batch Processing**: Pipeline optimizado por lotes

### Rendimiento Actual
```
QC Score:    9-10 ms
Stability:   2 ms (con caching)
YOLO:        8-14 ms (TensorRT FP16)
Viz:         6-10 ms
Total:       25-30 ms/frame → 30-40 FPS
```

## 🐳 Docker Setup

### Build y Run
```bash
# Build imagen
docker-compose build

# Procesar video
docker-compose run --rm sentinel-eye python3 src/main.py --video data/your_video.mp4 --no-display

# Procesar con output custom
docker-compose run --rm sentinel-eye python3 src/main.py --video data/input.mp4 --output outputs/result.mp4 --no-display
```

### Requisitos
- **Docker** con soporte NVIDIA GPU
- **NVIDIA Container Toolkit**
- **GPU NVIDIA** con compute capability ≥ 6.1
- **CUDA 11.8+** (incluido en imagen base)

## 📁 Estructura del Proyecto

```
Sentinel_Eye/
├── src/
│   ├── main.py                    # Pipeline principal
│   ├── modules/
│   │   ├── qc_score.py           # Módulo 1: QC Score (PyTorch GPU)
│   │   ├── stability.py          # Módulo 2: Stability (feature caching)
│   │   ├── motion_detection.py   # Módulo 3: YOLO + ORB tracking
│   │   ├── optimization.py       # Performance optimizer
│   │   └── model_export.py       # TensorRT export utilities
│   ├── utils/
│   │   ├── config.py             # Configuration loader
│   │   ├── logger.py             # Logging setup
│   │   └── visualization.py      # Video output con overlays
├── config.yaml                    # Configuración principal
├── Dockerfile                     # TensorRT + PyTorch CUDA image
├── docker-compose.yml
├── requirements.txt
└── OPTIMIZATION.md               # Documentación de optimizaciones

```

## ⚙️ Configuración (config.yaml)

```yaml
video:
  frame_skip: 2  # Process every 2 frames

qc_score:
  weights:
    sharpness: 0.35
    occlusion: 0.25
    lighting: 0.20
    cleanliness: 0.20

stability:
  history_size: 30
  vibration_threshold: 5.0

detection:
  use_yolo: true
  frame_skip: 2
  show_yolo: true

optimization:
  use_gpu: true
  enable_tensorrt: false  # TensorRT se carga automáticamente si existe .engine
```

## 🎯 ROI Tracking con ORB

El sistema usa **ORB (Oriented FAST and Rotated BRIEF)** para trackear features en el ROI:

1. **Detección inicial**: YOLO detecta objeto → crea ROI
2. **Feature extraction**: ORB extrae ~50-100 features del ROI
3. **Frame-to-frame tracking**: ORB matcher + RANSAC para homografía
4. **ROI persistence**: ROI se mueve siguiendo las features aunque la cámara se mueva

### Ventajas vs Lucas-Kanade
- ✅ **Robusto a rotación** (rotation invariant)
- ✅ **Robusto a escala** (scale invariant)
- ✅ **Rápido** (binary descriptors)
- ✅ **Funciona con movimiento de cámara** (homografía global)

## 📊 Outputs

### Video Procesado
- **Overlays**: QC Score, Stability metrics, YOLO detections, ROI tracking
- **Gráficos**: QC history, vibration plot
- **Bounding boxes**: YOLO detections con clase y confianza
- **ROI tracker**: Rectángulo verde siguiendo features ORB

### Logs
```
2025-12-01 22:48:01 - INFO - Timings [ms]: QC=9.2 | Stability=1.9 | YOLO=8.1 | Viz=8.7 | Total=28.2
2025-12-01 22:48:01 - INFO - Progress: 9.9% | Frame: 30/302 | FPS: 27.8
2025-12-01 22:48:01 - INFO - ORB: 52 features tracked | Homography valid
```

## 🔧 TensorRT Export

El modelo YOLO se exporta automáticamente a TensorRT en el primer run:

```bash
# Manual export (opcional)
docker-compose run --rm sentinel-eye python3 -c \
  "from ultralytics import YOLO; \
   model = YOLO('yolov8n.pt'); \
   model.export(format='engine', device=0, half=True, imgsz=640)"
```

El engine generado (`yolov8n.engine`) se guarda en `models/` y se reutiliza automáticamente.

## 📈 Optimizaciones Aplicadas

1. **TensorRT FP16**: YOLO compilado con half precision
2. **PyTorch GPU**: QC Score con operaciones CUDA
3. **Feature Caching**: Vibration features cada 10 frames
4. **Frame Skip**: Procesar cada 2 frames (mantener 30 FPS output)
5. **Frame Duplication**: Duplicar frames procesados para suavidad

Ver `OPTIMIZATION.md` para detalles técnicos completos.

## 🐛 Troubleshooting

### YOLO no detecta nada
- Verifica que `models/yolov8n.engine` exista y sea válido
- Prueba bajando `conf_threshold` en `config.yaml`

### FPS bajo
- Verifica que GPU esté disponible: `docker-compose run --rm sentinel-eye nvidia-smi`
- Aumenta `frame_skip` en `config.yaml`

### ORB no trackea ROI
- Verifica que haya suficientes features: mínimo 10 features para homografía
- Ajusta `max_features` en `motion_detection.py`

## 📝 License

MIT

## 👤 Author

mramirezz

