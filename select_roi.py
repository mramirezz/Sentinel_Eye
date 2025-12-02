"""
Script para seleccionar ROI INICIAL (zona de interés para detección)
Este ROI se ajustará dinámicamente cuando la cámara se mueva.
"""
import cv2
import sys
import json
import os

def load_rois(json_path='initial_rois.json'):
    """Carga el archivo de ROIs existente"""
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}

def save_roi_to_json(video_name, roi, description, json_path='initial_rois.json'):
    """Guarda el ROI en el archivo JSON"""
    rois = load_rois(json_path)
    
    x, y, w, h = roi
    rois[video_name] = {
        'x': int(x),
        'y': int(y),
        'width': int(w),
        'height': int(h),
        'description': description
    }
    
    with open(json_path, 'w') as f:
        json.dump(rois, f, indent=2)
    
    print(f"\n✓ ROI guardado en {json_path}")
    print(f"  Video: {video_name}")
    print(f"  ROI: ({x}, {y}, {w}x{h})")
    print(f"  Descripción: {description}")

def select_initial_roi(video_path):
    """Abre el primer frame y permite seleccionar ROI inicial"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir {video_path}")
        return None
    
    # Leer primer frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: No se pudo leer el primer frame")
        return None
    
    print("\n" + "="*70)
    print("SELECCIÓN DE ROI INICIAL (Zona de Interés)")
    print("="*70)
    print("\nInstrucciones:")
    print("1. Dibuja un rectángulo sobre la ZONA donde quieres detectar objetos")
    print("   (ej: área por donde pasan camiones, entrada de túnel, etc)")
    print("2. Esta zona será TRACKED automáticamente si la cámara se mueve")
    print("3. Presiona ENTER para confirmar")
    print("4. Presiona C para cancelar y reintentar")
    print("\n" + "="*70 + "\n")
    
    # Seleccionar ROI
    roi = cv2.selectROI("Selecciona ROI INICIAL (zona de detección)", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    
    if roi[2] == 0 or roi[3] == 0:
        print("ROI cancelada")
        return None
    
    x, y, w, h = roi
    print(f"\n✓ ROI seleccionada: x={x}, y={y}, w={w}, h={h}")
    
    # Mostrar preview con ROI
    preview = frame.copy()
    cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(preview, "ZONA DE DETECCION", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("Preview - Presiona cualquier tecla para continuar", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return roi

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python select_roi.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    video_name = os.path.basename(video_path)
    
    roi = select_initial_roi(video_path)
    
    if roi:
        # Pedir descripción del objeto
        description = input("\nDescripción de la zona seleccionada (ej: 'Entrada túnel', 'Zona de carga'): ").strip()
        if not description:
            description = "Zona de interés"
        
        # Guardar en JSON
        save_roi_to_json(video_name, roi, description)
        
        print("\n✓ Listo! El ROI se cargará automáticamente y se ajustará si la cámara se mueve.")
