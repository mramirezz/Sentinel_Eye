"""
Module 2: Stability Analysis with Reference Frame
Detecta vibración de cámara comparando contra un frame de referencia estático.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class StabilityAnalyzer:
    """
    Detecta vibración comparando features actuales vs frame de referencia.
    Tracking automático de ROI estática aunque la cámara se mueva.
    """
    
    def __init__(self, history_size: int = 30, vibration_threshold: float = 2.0):
        """
        Args:
            history_size: Frames en historial de movimiento
            vibration_threshold: Umbral en px para detectar vibración
        """
        self.history_size = history_size
        self.vibration_threshold = vibration_threshold
        
        # Movement history
        self.dx_history = deque(maxlen=history_size)
        self.dy_history = deque(maxlen=history_size)
        
        # Reference frame (primer frame del stream)
        self.reference_frame = None
        self.reference_kpts = None
        self.reference_desc = None
        
        # Accumulated camera displacement desde frame de referencia
        self.camera_offset_x = 0.0
        self.camera_offset_y = 0.0
        
        # Frame counter
        self.frame_count = 0
        
        # ORB detector (robusto a rotación/escala)
        self.orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31
        )
        
        # Matcher para ORB (Hamming distance)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Transformación afín previa (para suavizado)
        self.M_prev = None
        
        # Para visualización
        self.tracked_features = None
        self.prev_tracked_features = None
    
    def set_reference_frame(self, frame: np.ndarray):
        """
        Establece el frame de referencia (llamar una vez al inicio).
        Detecta features SOLO en el ROI estático de referencia.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reference_frame = gray.copy()
        
        # Si hay ROI de referencia, detectar features solo ahí
        if self.reference_roi:
            x = self.reference_roi['x']
            y = self.reference_roi['y']
            w = self.reference_roi['width']
            h = self.reference_roi['height']
            
            # Extraer ROI
            roi_gray = gray[y:y+h, x:x+w]
            
            # Detectar features en el ROI
            kpts, desc = self.orb.detectAndCompute(roi_gray, None)
            
            # Ajustar coordenadas de keypoints al frame completo
            if kpts:
                self.reference_kpts = [cv2.KeyPoint(kp.pt[0] + x, kp.pt[1] + y, kp.size, kp.angle, 
                                                     kp.response, kp.octave, kp.class_id) for kp in kpts]
                self.reference_desc = desc
                logger.info(f"Frame de referencia: {len(self.reference_kpts)} features en ROI ({x},{y},{w},{h})")
            else:
                self.reference_kpts = None
                self.reference_desc = None
                logger.warning(f"No se detectaron features en ROI de referencia ({x},{y},{w},{h})")
        else:
            # Sin ROI específico, detectar en toda la imagen
            self.reference_kpts, self.reference_desc = self.orb.detectAndCompute(gray, None)
            logger.info(f"Frame de referencia: {len(self.reference_kpts) if self.reference_kpts else 0} features (frame completo)")
        
        if self.reference_desc is None or len(self.reference_kpts) < 10:
            logger.warning(f"Solo {len(self.reference_kpts) if self.reference_kpts else 0} features en frame de referencia")
    
    def analyze_frame(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Tuple[float, float, bool]:
        """
        Analiza movimiento de cámara comparando con frame de referencia.
        
        Returns:
            (dx, dy, is_vibrating) - movimiento instantáneo y flag de vibración
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Establecer referencia si es el primer frame
        if self.reference_frame is None:
            self.set_reference_frame(frame)
            return 0.0, 0.0, False
        
        self.frame_count += 1
        
        # Detectar features en frame actual
        kpts_t, desc_t = self.orb.detectAndCompute(gray, None)
        
        if desc_t is None or len(kpts_t) < 10:
            logger.warning(f"Pocos features en frame actual: {len(kpts_t) if kpts_t else 0}")
            return 0.0, 0.0, False
        
        # Matching features: referencia → actual
        matches = self.matcher.match(self.reference_desc, desc_t)
        
        if len(matches) < 10:
            logger.debug(f"Solo {len(matches)} matches")
            return 0.0, 0.0, False
        
        # Ordenar por calidad (distancia menor = mejor match)
        matches = sorted(matches, key=lambda m: m.distance)
        good = matches[:min(100, len(matches))]  # Mejores 100 matches
        
        # Extraer puntos matched
        pts_ref = np.float32([self.reference_kpts[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_curr = np.float32([kpts_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # Estimar transformación afín con RANSAC
        M, inliers = cv2.estimateAffinePartial2D(
            pts_ref, pts_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0
        )
        
        if M is None:
            logger.debug("No se pudo estimar transformación afín")
            return 0.0, 0.0, False
        
        # Suavizado con transformación previa
        if self.M_prev is not None:
            alpha = 0.7  # 70% frame actual, 30% frame anterior
            M = alpha * M + (1 - alpha) * self.M_prev
        
        self.M_prev = M
        
        # Extraer traslación (dx, dy) de la matriz afín
        # M = [[cos(θ)*s, -sin(θ)*s, dx],
        #      [sin(θ)*s,  cos(θ)*s, dy]]
        dx = M[0, 2]
        dy = M[1, 2]
        
        # Actualizar offset acumulado
        self.roi_offset_x = dx
        self.roi_offset_y = dy
        
        # Calcular movimiento instantáneo (diferencia con frame anterior)
        if len(self.dx_history) > 0:
            dx_instant = dx - self.dx_history[-1] if len(self.dx_history) > 0 else 0
            dy_instant = dy - self.dy_history[-1] if len(self.dy_history) > 0 else 0
        else:
            dx_instant = dx
            dy_instant = dy
        
        # Update history con movimiento instantáneo
        self.dx_history.append(dx_instant)
        self.dy_history.append(dy_instant)
        
        # Detectar vibración
        is_vibrating = self._detect_vibration()
        
        # Log cada 30 frames
        if self.frame_count % 30 == 0:
            logger.info(f"Offset desde ref: ({self.roi_offset_x:.1f}, {self.roi_offset_y:.1f}) px | Movimiento: dx={dx_instant:.2f}, dy={dy_instant:.2f}")
        
        if is_vibrating:
            logger.warning(f"VIBRATING - Movimiento instantaneo: dx={dx_instant:.2f}, dy={dy_instant:.2f}")
        
        # Guardar features para visualización
        self.prev_tracked_features = self.tracked_features
        self.tracked_features = pts_curr.reshape(-1, 2)
        
        return dx_instant, dy_instant, is_vibrating
    
    def adjust_roi(self, original_roi: Tuple[int, int, int, int], 
                   frame_shape: Tuple[int, int] = None, 
                   frame: np.ndarray = None) -> Tuple[int, int, int, int]:
        """
        Ajusta la ROI original según el movimiento de la cámara.
        La ROI sigue los puntos estáticos aunque la cámara se mueva.
        """
        if self.M_prev is None:
            return original_roi
        
        x, y, w, h = original_roi
        
        # Coordenadas de las esquinas del ROI original
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # Aplicar transformación afín para seguir el movimiento
        transformed = cv2.transform(corners, self.M_prev)
        
        # Calcular nuevo bounding box
        pts = transformed.reshape(-1, 2)
        new_x = int(np.min(pts[:, 0]))
        new_y = int(np.min(pts[:, 1]))
        new_w = int(np.max(pts[:, 0]) - new_x)
        new_h = int(np.max(pts[:, 1]) - new_y)
        
        return (new_x, new_y, new_w, new_h)
    
    def _detect_vibration(self) -> bool:
        """
        Detecta vibración basándose en oscilaciones en el historial.
        """
        if len(self.dx_history) < self.history_size // 2:
            return False
        
        # Magnitud de movimientos
        movements = np.sqrt(np.array(self.dx_history)**2 + np.array(self.dy_history)**2)
        
        # Ratio de movimientos que superan threshold
        high_movement_ratio = np.sum(movements > self.vibration_threshold) / len(movements)
        
        # Desviación estándar (indica oscilación)
        dx_std = np.std(self.dx_history)
        dy_std = np.std(self.dy_history)
        oscillation = (dx_std > 1.0) or (dy_std > 1.0)
        
        return high_movement_ratio > 0.2 or oscillation
