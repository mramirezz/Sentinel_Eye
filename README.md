# Sentinel Eye

Sistema de monitoreo de calidad de video y detección de objetos con TensorRT, optical flow tracking y optimizaciones GPU para ambientes industriales.

## Características Principales

### Módulo 1: Quality Control (QC) Score
Sistema de diagnóstico de salud de imagen que analiza 4 métricas críticas:
- **Sharpness**: Laplacian variance para detectar desenfoque
- **Occlusion**: Edge density para detectar suciedad/obstrucciones
- **Lighting**: Análisis de histograma RGB para low light/glare
- **Cleanliness**: Variance de textura para detectar polvo acumulado

Score ponderado 0-100 con alertas automáticas si QC < 60.

### Módulo 2: Stability & Self-Healing
Detección de vibración y ajuste dinámico de ROI:
- **Vibration Detection**: Optical flow con goodFeaturesToTrack (threshold 0.3px promedio en 10 frames)
- **Self-Healing ROI**: ROI persigue zona física aunque cámara se mueva (pan/tilt/vibración)
- **Dual Visualization**: ROI original (gris) vs ROI tracked (amarillo)

### Módulo 3: Motion Detection
YOLOv8n optimizado con TensorRT FP16 para detección de objetos (8-14ms por frame).

## Rendimiento

```
QC Score:    9-10 ms (PyTorch CUDA)
Stability:   1-2 ms (optical flow)
YOLO:        8-14 ms (TensorRT FP16)
Visualization: 6-10 ms
────────────────────────────
Total:       25-30 ms
```

## Arquitectura Técnica

### Pipeline de Procesamiento

El sistema usa **procesamiento secuencial single-threaded** por diseño:

**¿Por qué no multiprocessing/threading?**
- **GPU Bottleneck**: El cuello de botella es la GPU, no CPU. Paralelizar frames no ayuda porque compiten por misma GPU.
- **Memory Efficiency**: Compartir contexto CUDA entre threads es complejo y propenso a race conditions.
- **Simplicidad**: Pipeline secuencial es más fácil de debuggear y mantener.
- **Frame Dependencies**: Self-healing ROI requiere estado del frame anterior (optical flow frame-to-frame).

**Alternativa considerada**: AsyncIO para I/O (lectura de video) pero el procesamiento GPU es síncrono por naturaleza.

**Optimización elegida**: Frame skip + caching de features en lugar de paralelización.

### Decisiones de Optimización

**1. TensorRT FP16 sobre ONNX Runtime**

Razón: Hardware target es NVIDIA Jetson/GPU.
- TensorRT es nativo para NVIDIA → mejor integración con CUDA
- FP16 aprovecha Tensor Cores (3x speedup vs FP32)
- ONNX Runtime es agnóstico pero más lento en NVIDIA hardware

Métricas:
```
PyTorch FP32:     ~35ms/frame (baseline)
ONNX Runtime:     ~28ms/frame
TensorRT FP16:    8-14ms/frame (3x mejora)
```

**2. Frame Skip = 2**

Inicialmente usamos frame_skip=1 (todos los frames) pero tracking era inestable.

Hallazgo: Con skip=2, las features persisten más tiempo entre frames → mejor tracking ROI.

Trade-off: Latencia aumenta pero tracking mejora significativamente (ROI no "tiembla").

**3. Unified Vibration Architecture**

Decisión: Usar mismas features del ROI para vibración + self-healing (en lugar de feature sets separados).

Beneficio: Coherencia - vibración se calcula sobre zona de interés, no imagen completa.

Resultado: Vibration detection más precisa y relevante para la zona monitoreada.

## Estrategia MLOps (Escalado a Producción)

### Deployment en 1000 Dispositivos Edge

**Arquitectura**:
```
Edge (Jetson Xavier NX) → Regional Hub (TimescaleDB) → Cloud (AWS)
```

**Edge Layer**: 
- Docker + K3s para orquestación
- 24h buffer local (256GB NVMe)
- Prometheus exporter para métricas

**Monitoring de Data Drift**:

El problema: Cámaras se ensucian progresivamente → QC score baja, sharpness cae, occlusion aumenta.

**Estrategia de detección**:

1. **Baseline (30 días)**: Establecer distribución normal de QC metrics por cámara
2. **Detection diaria**: Kolmogorov-Smirnov test comparando últimos 7 días vs baseline
3. **Threshold**: p-value < 0.05 indica drift significativo

**Query de análisis** (TimescaleDB):
```sql
WITH baseline AS (
  SELECT camera_id, AVG(qc_score) as baseline_mean
  FROM metrics
  WHERE timestamp >= NOW() - INTERVAL '30 days'
    AND timestamp < NOW() - INTERVAL '7 days'
  GROUP BY camera_id
),
recent AS (
  SELECT camera_id, AVG(qc_score) as recent_mean
  FROM metrics WHERE timestamp >= NOW() - INTERVAL '7 days'
  GROUP BY camera_id
)
SELECT 
  b.camera_id,
  (b.baseline_mean - r.recent_mean) as drift_magnitude
FROM baseline b JOIN recent r ON b.camera_id = r.camera_id
WHERE drift_magnitude > 5;
```

**Auto-Remediation**:
- QC < 40 por 24h → Ticket automático "limpiar cámara" (ServiceNow)
- Vibración > 5px por 6h → Ticket "revisar montaje mecánico"
- Drift detectado 3 días consecutivos → Trigger pipeline de re-entrenamiento

**Model Retraining**:
1. Collect edge cases (frames con QC < 60 o vibración > 3px) de últimos 90 días
2. SageMaker training job: 80% data histórico + 20% edge cases recientes
3. A/B testing: Deploy a 10% de dispositivos por 24 horas
4. Promote si t-test muestra mejora significativa (p < 0.05)

**Observability**:
- Prometheus metrics: `qc_score`, `vibration_magnitude`, `frame_processing_ms`
- Grafana dashboards: Fleet overview (heatmap), cámaras bajo threshold, latency p95
- Alertas críticas: QC < 40 (30m), vibración > 5px (1h), latency p95 > 100ms

**CI/CD Pipeline**:
```
GitHub Actions → pytest → docker build → ECR push → 
canary deploy (10%) → monitor 1h → rollout/rollback
```

## Quick Start

## Quick Start

### Docker Setup

```bash
# Build imagen
docker-compose build

# Procesar video
docker-compose run --rm sentinel-eye python src/main.py --video data/video.mp4 --no-display

# Procesar todos los videos en data/
docker-compose run --rm sentinel-eye python src/main.py --no-display
```

**Requisitos**: Docker + NVIDIA Container Toolkit + GPU NVIDIA (compute capability ≥ 6.1)

## Configuración (config.yaml)

## Configuración (config.yaml)

```yaml
video:
  frame_skip: 2  # Mejor tracking con skip=2

qc_score:
  weights:
    sharpness: 0.35
    occlusion: 0.25
    lighting: 0.20
    cleanliness: 0.20

stability:
  vibration_threshold: 0.3  # px promedio en 10 frames
  enable_self_healing: true
  initial_roi_file: initial_rois.json

detection:
  use_yolo: true
  confidence_threshold: 0.5
```

## ROI Tracking

## ROI Tracking

Define ROI inicial en `initial_rois.json`:

```json
{
  "earthquake.mp4": {
    "x": 599, "y": 399,
    "width": 245, "height": 321,
    "description": "Silla de oficina"
  }
}
```

El sistema usa **optical flow** (goodFeaturesToTrack + calcOpticalFlowPyrLK) para:
1. Detectar ~100 features en ROI inicial
2. Trackear features frame-to-frame
3. Calcular desplazamiento medio → ajustar ROI automáticamente
4. ROI persigue zona física aunque cámara se mueva

Ver `ROI_GUIDE.md` para más detalles.

## Output

## Output

**Video procesado** (`outputs/video_output.mp4`):
- QC Score en tiempo real (top-right)
- Gráficos de vibración (bottom-left)
- ROI original (gris) vs ROI tracked (amarillo)
- YOLO detections con bounding boxes

**Métricas** (`outputs/video_output_metrics.png`):
- Gráfico de vibración (X/Y drift) vs tiempo
- Gráfico de QC Score vs tiempo

**Logs**:
```
2025-12-02 11:24:09 - INFO - Timings [ms]: QC=9.2 | Stability=1.3 | YOLO=8.1 | Viz=8.5 | Total=28.2
2025-12-02 11:24:09 - INFO - Progress: 7.9% | Frame: 60/757 | FPS: 27.4
2025-12-02 11:24:09 - WARNING - Frame 54: Vibration detected (dx=0.23, dy=0.42)
```

## Troubleshooting

## Troubleshooting

**GPU no detectada**: `docker-compose run --rm sentinel-eye nvidia-smi`

**FPS bajo**: Aumenta `frame_skip` en config.yaml

**ROI no trackea**: Verifica que `initial_rois.json` tenga entrada para tu video

**YOLO no detecta**: Baja `confidence_threshold` en config.yaml

## Documentación Adicional

- `OPTIMIZATION.md`: Detalles de optimizaciones TensorRT/CUDA
- `ROI_GUIDE.md`: Configuración de ROI tracking
- `MLOPS_STRATEGY.md`: Estrategia de deployment a escala
- `EVALUATION_CHECKLIST.md`: Cumplimiento de requisitos

## Stack Tecnológico

- **Core**: Python 3.10, OpenCV 4.x, PyTorch 2.1.0
- **Inference**: TensorRT 8.6.1 FP16, YOLOv8n
- **GPU**: CUDA 11.8, cuDNN 8.x
- **Container**: Docker, NVIDIA Container Toolkit
- **Monitoring**: Prometheus (production), Grafana (dashboards)

## License

MIT

## Author

mramirezz


