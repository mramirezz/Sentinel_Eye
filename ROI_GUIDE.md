# Guía de ROI Dinámico en Sentinel Eye

## Concepto: Self-Healing ROI Tracking

El sistema usa **UN SOLO ROI** que se **ajusta dinámicamente** para seguir la misma zona física cuando la cámara se mueve.

### ¿Cómo funciona?

1. **Frame de Referencia**: El primer frame se usa para detectar features (ORB) en toda la imagen
2. **Tracking de Cámara**: Cada frame se compara con la referencia para calcular desplazamiento de cámara
3. **Ajuste de ROI**: El ROI se mueve automáticamente para compensar el movimiento de cámara
4. **Resultado**: El ROI siempre apunta a la misma zona física, aunque la cámara se mueva

### Ejemplo Visual

```
Frame 1 (referencia):
┌──────────────────────────────┐
│                              │
│        ┌─────────┐           │  <- ROI inicial (100, 100, 200x150)
│        │  ZONA   │           │     Zona donde pasan camiones
│        │ CAMIÓN  │           │
│        └─────────┘           │
│                              │
└──────────────────────────────┘

Frame 50 (cámara se movió 50px a la derecha):
┌──────────────────────────────┐
│                              │
│              ┌─────────┐     │  <- ROI ajustado (150, 100, 200x150)
│              │  ZONA   │     │     Sigue apuntando a la misma zona física!
│              │ CAMIÓN  │     │
│              └─────────┘     │
│                              │
└──────────────────────────────┘
```

## Workflow Completo

### Primera vez con un video:
```bash
# 1. Seleccionar ROI INICIAL (zona de interés)
python select_roi.py data/earthquake.mp4
# -> Dibuja rectángulo sobre zona donde quieres detectar objetos
# -> Se guarda automáticamente en initial_rois.json

# 2. Procesar video (ROI se ajusta automáticamente)
docker-compose run --rm sentinel-eye python3 src/main.py --video data/earthquake.mp4 --no-display
```

### Videos subsecuentes:
```bash
# Solo procesar - el ROI se carga y ajusta automáticamente
docker-compose run --rm sentinel-eye python3 src/main.py --video data/earthquake.mp4 --no-display
```

## Módulo 2: Self-Healing ROI

**Objetivo**: Que el ROI "persiga" la misma zona física aunque la cámara se mueva (pan/tilt)

### Componentes:

1. **Feature Detection**: ORB features en toda la imagen del primer frame
2. **Feature Matching**: Compara frame actual vs referencia (BFMatcher + Lowe's ratio test)
3. **Transformación Afín**: Estima desplazamiento de cámara con RANSAC
4. **ROI Adjustment**: Aplica offset al ROI original

### Métricas:

- **displacement_x/y**: Movimiento instantáneo de la cámara (frame a frame)
- **camera_offset_x/y**: Desplazamiento acumulado desde frame de referencia
- **is_vibrating**: Detecta vibraciones (threshold configurable)
- **ROI ajustado**: Se visualiza en verde siguiendo la zona física

## Archivos

- `initial_rois.json` - Base de datos de ROIs iniciales por video
- `select_roi.py` - Tool para seleccionar ROI inicial
- `src/modules/stability_tracking.py` - Implementación del tracking
- `config.yaml` - Configuración global (threshold, enable_self_healing, etc)

## Tips

1. **ROI debe cubrir zona de interés**: Área donde esperas ver objetos/eventos
2. **Tamaño recomendado**: Al menos 15-20% del frame para buena detección
3. **Posición**: Puede ser anywhere - el sistema lo seguirá
4. **Features**: El sistema detecta automáticamente objetos estáticos en toda la imagen
5. **Self-healing**: Funciona SIEMPRE, no solo durante vibraciones

## Ventajas

✅ **Un solo ROI**: No confundir múltiples ROIs  
✅ **Automático**: Se ajusta sin intervención manual  
✅ **Robusto**: Usa RANSAC para ignorar outliers  
✅ **Visualizable**: Offset de cámara se muestra en pantalla  
✅ **Configurable**: Threshold y habilitación en config.yaml
