# Optimización TensorRT - Sentinel Eye

## Por Qué YOLOv8n + TensorRT FP16

**Modelo elegido**: YOLOv8n (nano)
- **Tamaño**: 3.2M parámetros, 6.2 MB
- **Speed/Accuracy trade-off**: Más rápido que YOLOv8s/m/l/x, suficiente precisión para detección general
- **Edge-friendly**: Cabe en GPUs pequeñas (Jetson Nano con 2GB VRAM)

**Pipeline de conversión**: PyTorch → ONNX → TensorRT
1. **PyTorch (.pt)**: Modelo original de Ultralytics, descargado automáticamente si no existe
2. **ONNX (.onnx)**: Formato intermedio estándar, facilita la conversión a TensorRT
3. **TensorRT (.engine)**: Ejecutable optimizado específico para tu GPU

**Por qué FP16 (half precision)**:
- **Tensor Cores**: RTX 3050/Jetson tienen Tensor Cores que aceleran FP16 (2-3x speedup)
- **Memoria**: Usa mitad de VRAM (crítico en GPUs edge con 4GB)
- **Precisión**: Pérdida mínima en mAP (<1%) para detección de objetos
- **Trade-off**: FP32 es 0.5% más preciso pero 3x más lento

**Alternativas descartadas**:
- ONNX Runtime: Agnóstico pero 2x más lento que TensorRT en NVIDIA GPUs
- YOLOv5: Arquitectura más vieja, YOLOv8 tiene mejor accuracy/speed ratio
- YOLOv8s/m: Más precisos pero 2-4x más lentos (innecesario para uso general)

## Auto-Generación de TensorRT Engine

### Primera Ejecución (Automática)

El sistema genera automáticamente el TensorRT engine optimizado la primera vez que lo ejecutas:

```bash
# Build de la imagen con TensorRT incluido
docker-compose build

# Primera ejecución: auto-genera models/yolov8n.engine
docker-compose run --rm sentinel-eye python3 src/main.py --video data/video.mp4 --output outputs/resultado.mp4 --no-display
```

**Proceso automático** (solo primera vez, ~4 minutos):
1. Descarga YOLOv8n.pt (6.2 MB) desde Ultralytics
2. Exporta a ONNX con opset 17
3. Genera TensorRT engine FP16 con kernel optimization
4. Guarda `models/yolov8n.engine` (9.5 MB)
5. Ejecuta inferencia con el engine generado

**Ejecuciones posteriores**: Usa el engine existente (carga instantánea, <1s)

### Qué Hace la Auto-Generación

```python
# src/modules/motion_detection.py - YOLODetector.__init__()

if Path('models/yolov8n.engine').exists():
    # Carga engine existente
    self._load_tensorrt('models/yolov8n.engine')
else:
    # AUTO-GENERA engine en primera ejecución
    model = YOLO('yolov8n.pt')
    model.export(
        format='engine',
        device=0,          # GPU 0
        half=True,         # FP16 precision
        imgsz=640,         # Input size
        workspace=4        # 4GB workspace
    )
    # Mueve engine a models/
    shutil.move('yolov8n.engine', 'models/yolov8n.engine')
```

**Optimizaciones aplicadas por TensorRT**:
- Layer fusion (Conv + BatchNorm + ReLU)
- Kernel auto-tuning para RTX 3050
- FP16 precision (half precision)
- Dynamic tensor memory allocation
- Vertical/horizontal layer fusion

### Fallback Automático

Si TensorRT falla (GPU no disponible, driver incompatible, etc.), el sistema automáticamente usa PyTorch:

```python
try:
    self._load_tensorrt(engine_path)
    logger.info("Using TensorRT engine")
except Exception as e:
    logger.warning(f"TensorRT failed: {e}, falling back to PyTorch")
    self._load_yolo()  # PyTorch backend
```

## Benchmarks YOLOv8n

| Método | Latencia (ms/frame) | Tamaño | Hardware |
|--------|---------------------|--------|----------|
| PyTorch FP32 | ~60 ms | 6.2 MB | RTX 3050 |
| **TensorRT FP16** | **<10 ms** | 9.5 MB | RTX 3050 |

*Pipeline completo: 14.6 FPS (limitado por I/O de video, no por inferencia)*

## Troubleshooting

### Engine no se genera

**Problema**: Primera ejecución no crea `models/yolov8n.engine`

**Solución**:
```bash
# Verifica GPU disponible
docker-compose run --rm sentinel-eye nvidia-smi

# Logs completos
docker-compose run --rm sentinel-eye python3 src/main.py --video data/video.mp4 --output outputs/test.mp4 --no-display 2>&1 | grep -i tensorrt

# Si falla, revisa driver NVIDIA (requiere 530+)
nvidia-smi
```

### Engine incompatible con otra GPU

**Problema**: `models/yolov8n.engine` generado en RTX 3050 no funciona en T4/V100

**Solución**: El engine es específico para cada arquitectura GPU. Elimina y regenera:
```bash
# Eliminar engine existente
rm models/yolov8n.engine

# Regenerar para GPU actual
docker-compose run --rm sentinel-eye python3 src/main.py --video data/video.mp4 --output outputs/test.mp4 --no-display
```

### Fallback a PyTorch inesperado

**Problema**: Logs muestran "falling back to PyTorch" aunque GPU está disponible

**Causas comunes**:
- Driver NVIDIA desactualizado (requiere 530+)
- CUDA version mismatch (imagen usa CUDA 11.8)
- Memoria GPU insuficiente (<2GB libre)

**Diagnóstico**:
```bash
# Check CUDA disponible
docker-compose run --rm sentinel-eye python3 -c "import torch; print(torch.cuda.is_available())"

# Check TensorRT instalado
docker-compose run --rm sentinel-eye python3 -c "import tensorrt; print(tensorrt.__version__)"
```

## Deployment en Producción

### Jetson AGX/Xavier (Edge)

```bash
# Usar imagen ARM64 con TensorRT pre-instalado
FROM nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime

# Regenerar engine en Jetson (diferente arquitectura)
rm models/yolov8n.engine
python3 src/main.py --video test.mp4 --output out.mp4 --no-display
```

Performance esperado: 30-60 FPS en Jetson AGX Xavier

### Cloud (T4/V100/A100)

```bash
# Imagen base para datacenter GPUs
FROM nvcr.io/nvidia/tensorrt:23.08-py3

# Engine se auto-genera para GPU específica
# T4: ~50 FPS | V100: ~80 FPS | A100: ~150 FPS
```

### CPU-Only (Sin GPU)

El sistema detecta automáticamente ausencia de GPU y usa PyTorch CPU:

```yaml
# config.yaml
optimization:
  use_gpu: false  # Fuerza CPU mode
```

Performance: ~5-10 FPS en CPU moderno (i7/i9)

## Referencias

- **TensorRT Developer Guide**: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/
- **Ultralytics TensorRT Export**: https://docs.ultralytics.com/modes/export/#tensorrt
- **NVIDIA Container Toolkit**: https://github.com/NVIDIA/nvidia-container-toolkit
- **TensorRT Docker Images**: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt
