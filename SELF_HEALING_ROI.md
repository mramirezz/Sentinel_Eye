# Self-Healing ROI: Explicación Detallada

## 🎯 Concepto Principal

El **Self-Healing ROI** es un sistema que ajusta automáticamente la región de interés (ROI) para seguir la misma zona física del mundo real, incluso cuando la cámara se mueve por vibración.

### Problema que Resuelve

Imagina que tienes una cámara mirando una carretera y defines un ROI para contar vehículos:

```
Frame 1 (cámara estable):
┌─────────────────────────┐
│                         │
│     ┌─────────┐        │
│     │  ROI    │        │  ← El ROI cubre la zona de interés
│     │ ■ ■ ■   │        │
│     └─────────┘        │
│                         │
└─────────────────────────┘
```

Si la cámara vibra 10 píxeles hacia la derecha:

```
Frame 2 (cámara vibró →):
┌─────────────────────────┐
│                         │
│ ┌─────────┐            │  
│ │  ROI    │            │  ← El ROI sigue en la misma posición de píxeles
│ │ ✗ ✗ ✗   │            │     pero ahora cubre una zona física diferente!
│ └─────────┘            │
│                         │
└─────────────────────────┘
        ↑ La carretera se movió en la imagen
```

**Sin self-healing**: El ROI cubre la zona incorrecta → Pierdes vehículos o detectas falsos positivos

**Con self-healing**: El ROI se ajusta automáticamente para seguir la misma zona física:

```
Frame 2 (con self-healing):
┌─────────────────────────┐
│                         │
│         ┌─────────┐    │  
│         │  ROI    │    │  ← El ROI se movió para seguir la zona física
│         │ ■ ■ ■   │    │
│         └─────────┘    │
│                         │
└─────────────────────────┘
```

---

## 🔧 Cómo Funciona: Paso a Paso

### 1. Inicialización (Primera Frame)

Cuando llamas a `set_reference_frame()`:

```python
# Frame inicial: 1280x720
# ROI definido manualmente: (400, 200, 300, 200)
#   x=400, y=200, ancho=300, alto=200

analyzer.set_reference_frame(frame, roi=(400, 200, 300, 200))
```

**Qué hace internamente:**

1. **Detecta features** (puntos característicos) dentro del ROI:
   ```
   ROI en (400, 200):
   ┌──────────────┐
   │ ●            │  ← 100 puntos detectados
   │    ●  ●      │     (esquinas, bordes, texturas)
   │  ●        ●  │
   │      ●    ●  │
   └──────────────┘
   ```

2. **Guarda las coordenadas** de esos puntos:
   ```python
   self.tracked_features = [
       [450, 210],  # Punto 1
       [520, 230],  # Punto 2
       [480, 250],  # Punto 3
       ...
   ]
   ```

3. **Inicializa offsets en cero**:
   ```python
   self.camera_offset_x = 0.0
   self.camera_offset_y = 0.0
   ```

---

### 2. Tracking Frame a Frame (Optical Flow)

En cada frame siguiente, llamas a `analyze_frame()`:

#### Frame 2: Cámara vibró 3px → y 2px ↓

**Paso 2.1: Calcular Optical Flow**

```python
# OpenCV busca dónde están ahora los mismos puntos
p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, tracked_features, ...)
```

Resultado:
```
Punto 1: [450, 210] → [453, 212]  (movimiento: +3, +2)
Punto 2: [520, 230] → [523, 232]  (movimiento: +3, +2)
Punto 3: [480, 250] → [483, 252]  (movimiento: +3, +2)
...
```

**Paso 2.2: Filtrar Movimientos Válidos**

```python
# FILTRO 1: Rechazar movimientos > 15px (objetos en movimiento, no cámara)
movements = [[3, 2], [3, 2], [3, 2], ...]
magnitudes = [√(3²+2²) = 3.6px, 3.6px, 3.6px, ...]

valid_mask = magnitudes < 15.0  # Todos pasan ✓
```

Si hubiera un camión cruzando:
```
Punto 50 (en el camión): [600, 300] → [650, 305]  (movimiento: +50, +5)
Magnitud: √(50²+5²) = 50.2px > 15px  → RECHAZADO ✗
```

**Paso 2.3: RANSAC - Consenso de Movimiento**

```python
# Calcular movimiento consensuado (median)
median_dx = median([3, 3, 3, ...]) = 3.0
median_dy = median([2, 2, 2, ...]) = 2.0

# Identificar outliers (puntos que se desvían del consenso)
deviations = [
    √((3-3)² + (2-2)²) = 0px,  ✓ Inlier
    √((3-3)² + (2-2)²) = 0px,  ✓ Inlier
    ...
]

# Si algún punto se desvía > 2px del consenso → Outlier → Rechazado
```

**Paso 2.4: Calcular Offset Acumulado**

```python
roi_dx = 3.0  # Movimiento consensuado en X
roi_dy = 2.0  # Movimiento consensuado en Y

# Acumular offset (es la suma de TODOS los movimientos desde el inicio)
self.camera_offset_x += roi_dx  # 0.0 + 3.0 = 3.0
self.camera_offset_y += roi_dy  # 0.0 + 2.0 = 2.0
```

---

#### Frame 3: Cámara vibró -2px ← y 1px ↓

```python
# Mismos pasos, nuevos movimientos:
roi_dx = -2.0
roi_dy = 1.0

# Acumular
self.camera_offset_x += -2.0  # 3.0 + (-2.0) = 1.0
self.camera_offset_y += 1.0   # 2.0 + 1.0 = 3.0
```

**El offset acumulado guarda el desplazamiento total de la cámara desde el frame 1.**

---

### 3. Ajuste del ROI

Cuando llamas a `adjust_roi(roi_original)`:

```python
# ROI original (definido en frame 1)
roi_original = (400, 200, 300, 200)

# Offset acumulado (cuánto se movió la cámara)
camera_offset_x = 1.0
camera_offset_y = 3.0

# ROI ajustado
new_x = 400 + 1.0 = 401
new_y = 200 + 3.0 = 203

# ROI ajustado final
roi_adjusted = (401, 203, 300, 200)
```

**Visualización:**

```
Frame 1:                    Frame 3 (cámara en +1, +3):
┌──────────────┐           ┌──────────────┐
│              │           │              │
│  ┌────────┐  │           │   ┌────────┐ │  ← ROI ajustado
│  │ROI orig│  │           │   │ROI adj │ │     se movió +1, +3
│  └────────┘  │           │   └────────┘ │
│              │           │              │
└──────────────┘           └──────────────┘
```

**Resultado**: El ROI ajustado cubre la **misma zona física del mundo real**, aunque la cámara se haya movido.

---

## 📊 Ejemplo Completo con Números

### Configuración Inicial

```python
# Frame 1 (1280x720)
# Defino ROI para monitorear carretera
roi_original = (500, 300, 400, 200)  # x, y, w, h

analyzer.set_reference_frame(frame1, roi_original)
# Detecta 87 features dentro del ROI
# camera_offset_x = 0.0
# camera_offset_y = 0.0
```

### Procesamiento de 5 Frames

| Frame | Movimiento (dx, dy) | Offset Acumulado (x, y) | ROI Ajustado          |
|-------|---------------------|-------------------------|-----------------------|
| 1     | -                   | (0.0, 0.0)              | (500, 300, 400, 200)  |
| 2     | (+3, +2)            | (3.0, 2.0)              | (503, 302, 400, 200)  |
| 3     | (-1, +1)            | (2.0, 3.0)              | (502, 303, 400, 200)  |
| 4     | (+2, -2)            | (4.0, 1.0)              | (504, 301, 400, 200)  |
| 5     | (-1, +3)            | (3.0, 4.0)              | (503, 304, 400, 200)  |

**En el Frame 5:**
- El ROI original estaba en (500, 300)
- El ROI ajustado está en (503, 304)
- **Esto compensa los +3px y +4px que la cámara se movió**
- El ROI sigue cubriendo la misma zona física de la carretera

---

## 🚗 Caso Real: Camión Cruzando

### Sin Filtrado (Problema Original)

```
Frame 10: Camión entra en el ROI
┌──────────────────┐
│  ┌────────┐      │
│  │🚛 ROI  │      │  Features detectan el camión
│  └────────┘      │
└──────────────────┘

Frame 11: Camión se mueve +50px →
┌──────────────────┐
│  ┌────────┐   🚛│  
│  │  ROI   │      │  Features del camión: movimiento +50px
│  └────────┘      │  Sistema piensa "cámara se movió +50px"
└──────────────────┘  

ROI se ajusta incorrectamente:
roi_adjusted = (500 + 50, 300, 400, 200) = (550, 300, 400, 200)
                      ↑ ERROR!
```

### Con Filtrado (Solución Implementada)

```
Frame 11: Camión se mueve +50px →

Features detectadas:
- Features del fondo: movimiento (+1, +0.5) ✓
- Features del camión: movimiento (+50, +2)  ✗ RECHAZADO (>15px)

FILTRO 1: Rechaza movimientos > 15px
  Features válidas: Solo las del fondo (80 de 87)

RANSAC: Identifica consenso
  Median: (+1, +0.5)
  Inliers: 78 features
  
Resultado:
  roi_dx = +1.0 (no +50!)
  roi_adjusted = (501, 300, 400, 200) ✓ CORRECTO
```

---

## 🔄 Recalculación de Features

### ¿Por qué se necesita?

Las features se pierden cuando:
- Salen del campo de visión
- Quedan cubiertas por objetos (camión pasa por enfrente)
- Hay polvo, niebla, cambios de iluminación

### Cuándo se activa

```python
if len(good_features) < 5:
    logger.warning("Solo 4 features, recalculando...")
    self._recalculate_roi_features(gray)
```

### Qué hace

1. **Calcula la posición actual del ROI ajustado**:
   ```python
   # ROI original: (500, 300, 400, 200)
   # Offset acumulado: (3.0, 4.0)
   adjusted_x = 500 + 3 = 503
   adjusted_y = 300 + 4 = 304
   ```

2. **Detecta nuevas features en esa zona**:
   ```python
   gray_roi = frame[304:504, 503:903]  # Zona del ROI ajustado
   new_features = cv2.goodFeaturesToTrack(gray_roi, maxCorners=100, ...)
   
   # Convierte coordenadas locales → globales
   self.tracked_features = new_features + [503, 304]
   ```

3. **Continúa el tracking con las nuevas features**:
   ```python
   # Frame siguiente usa estas nuevas features
   # El offset acumulado NO se resetea (sigue siendo 3.0, 4.0)
   ```

**Importante**: El recálculo **NO resetea el offset acumulado**. Solo refresca los puntos que se están rastreando.

---

## 📈 Ejemplo Visual Completo

### Video de 10 segundos (300 frames) con vibración

```
Frame 1: Inicialización
  ROI: (500, 300, 400, 200)
  Features: 87
  Offset: (0, 0)

Frames 2-50: Vibración normal
  Movimientos: ±1-3px por frame
  Offset evoluciona: (0,0) → (2,1) → (1,3) → (3,2) → ...
  Features: 87 → 82 (algunas se pierden)

Frame 51: Camión cruza
  Movimientos detectados:
    - 60 features fondo: (+1, -1)  ✓
    - 22 features camión: (+45, +3) ✗ RECHAZADO
  Offset acumulado: (5, -2)
  ROI ajustado: (505, 298, 400, 200)

Frame 52: Camión sigue cruzando
  Features fondo: 60 → 45 (camión las tapa)
  Offset: (4, -1)
  
Frame 53: Camión salió
  Features: 45 → 38 (se perdieron algunas)
  
Frame 54: Recalculación (< 40 features)
  logger: "38 features remaining, recalculating..."
  Detecta 91 nuevas features en ROI ajustado (504, 298)
  Offset NO cambia: sigue siendo (4, -1)

Frames 55-300: Continúa normalmente
  Offset evoluciona: (4,-1) → (3,0) → (2,1) → ...
  ROI siempre ajustado para seguir la zona física
```

---

## 🎓 Resumen Conceptual

### La Clave

**El self-healing ROI es un contador acumulativo del movimiento de la cámara.**

- Cada frame calcula: "¿Cuánto se movió la cámara?"
- Suma ese movimiento al offset total
- Usa el offset total para ajustar el ROI original

### Analogía

Imagina que estás parado en una cancha mirando una portería:

1. **Frame 1**: Estás en la línea central (posición 0)
2. **Frame 2**: Caminas 3 pasos a la derecha (posición +3)
3. **Frame 3**: Caminas 2 pasos a la izquierda (posición +1)
4. **Frame 4**: Caminas 1 paso a la derecha (posición +2)

Para seguir mirando la **misma portería**, debes ajustar tu vista según tu posición acumulada:
- Si estás en +2, miras "portería + 2 pasos"
- El self-healing hace exactamente esto pero con píxeles

### Ventajas

✅ **Robustez**: Detecta movimientos reales de cámara, ignora objetos
✅ **Precisión**: Sigue la zona física exacta
✅ **Auto-recuperación**: Recalcula features si se pierden
✅ **Transparente**: El usuario define ROI una vez, el sistema lo mantiene

### Limitaciones

⚠️ **No maneja zoom**: Solo traslación (dx, dy), no escala
⚠️ **No maneja rotación**: Solo movimiento lineal
⚠️ **Deriva posible**: Errores pequeños se acumulan (pero RANSAC minimiza esto)

---

## 🧪 Para Verificar que Funciona

### Test 1: Video con Vibración

```bash
docker-compose run --rm sentinel-eye python src/main.py --video data/video_2_vibration.mp4
```

**Observa**:
- El ROI dibujado (magenta) se mueve sutilmente frame a frame
- Pero siempre cubre la misma zona física del suelo
- Los offsets en pantalla (camera_offset_x, camera_offset_y) cambian

### Test 2: Video con Vehículos

```bash
docker-compose run --rm sentinel-eye python src/main.py --video data/earthquake2.mp4
```

**Observa**:
- Cuando pasa un vehículo, el ROI NO se va con él
- Los logs muestran: "85 inliers" (features del fondo)
- No muestra: "All movements excessive" (filtro funcionando)

### Test 3: Video con Polvo

```bash
docker-compose run --rm sentinel-eye python src/main.py --video data/video_1_dust.avi
```

**Observa**:
- Logs: "Only X features remaining, recalculating..." (cada varios frames)
- El ROI se mantiene en la zona correcta
- No se "pierde" ni deriva significativamente

---

## 💡 Configuración Avanzada

Si quieres ajustar el comportamiento, modifica estos parámetros en `stability_tracking.py`:

```python
# Umbral de movimiento máximo (rechaza objetos)
self.max_movement_per_frame = 15.0  # Aumenta si cámara se mueve rápido
                                     # Disminuye si objetos lentos causan problemas

# Umbral RANSAC (outlier detection)
self.ransac_threshold = 2.0  # Aumenta si vibración es muy errática
                             # Disminuye para mayor precisión

# Cantidad de features
maxCorners=100  # Más features = más robusto pero más CPU
                # Menos features = más rápido pero menos robusto

# Calidad de features
qualityLevel=0.2  # Menor = más features en zonas con poca textura
                  # Mayor = solo features muy nítidas
```
