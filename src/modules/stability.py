"""
Module 2: Stability Analysis and Self-Healing
Detects camera vibration/movement and dynamically adjusts ROIs.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class StabilityAnalyzer:
    """
    Detects vibration patterns and provides self-healing ROI adjustments.
    """
    
    def __init__(self, history_size: int = 30, vibration_threshold: float = 5.0):
        """
        Args:
            history_size: Number of frames to keep in history
            vibration_threshold: Pixel movement threshold to detect vibration
        """
        self.history_size = history_size
        self.vibration_threshold = vibration_threshold
        
        # Movement history
        self.dx_history = deque(maxlen=history_size)
        self.dy_history = deque(maxlen=history_size)
        
        # Previous frame for optical flow
        self.prev_gray = None
        
        # Feature tracking for ROI self-healing (persigue objeto estático físico)
        self.tracked_features = None  # Features en ROI inicial (coordenadas absolutas en frame)
        self.prev_tracked_features = None  # Features del frame anterior (para visualizar movimiento)
        self.prev_full_gray = None    # Frame completo previo (para tracking de features)
        self.roi_offset_x = 0.0       # Desplazamiento acumulado del ROI en X
        self.roi_offset_y = 0.0       # Desplazamiento acumulado del ROI en Y
        
        # ORB detector para features más robustos que goodFeaturesToTrack
        self.orb = cv2.ORB_create(
            nfeatures=100,        # Detectar hasta 100 features
            scaleFactor=1.2,      # Escala entre niveles de pirámide
            nlevels=8,            # Niveles de pirámide
            edgeThreshold=15,     # Borde para evitar features en bordes
            patchSize=31          # Tamaño de patch para descriptor
        )
        
        # Cache para vibration detection (evitar recalcular features cada frame)
        self.vibration_features = None
        self.vibration_feature_counter = 0
        
        # Parámetros de optical flow (backup si ORB falla)
        self.feature_params = dict(
            maxCorners=50,
            qualityLevel=0.3,
            minDistance=10,
            blockSize=7
        )
        
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
    def initialize_tracking(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        """
        Inicializar feature tracking en el ROI usando ORB para detectar features robustos.
        ORB es más resistente a polvo y cambios de iluminación que goodFeaturesToTrack.
        
        Args:
            frame: Primer frame del video
            roi: ROI inicial (x, y, w, h) sobre región estática (ej: rocas)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = roi
        
        # Extraer ROI para detectar features
        gray_roi = gray[y:y+h, x:x+w]
        
        # Aplicar CLAHE para mejorar contraste en la región (mejor detección en polvo)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_roi_enhanced = clahe.apply(gray_roi)
        
        # Detectar keypoints ORB en ROI (más robustos que Shi-Tomasi)
        keypoints = self.orb.detect(gray_roi_enhanced, None)
        
        if keypoints and len(keypoints) > 10:
            # Convertir keypoints a array de coordenadas
            features_roi = np.array([kp.pt for kp in keypoints[:50]], dtype=np.float32)  # Top 50
            # Convertir a coordenadas absolutas del frame
            self.tracked_features = features_roi + np.array([x, y], dtype=np.float32)
            print(f"✓ ORB tracking inicializado con {len(self.tracked_features)} features robustos en ROI")
            logger.info(f"ORB tracking inicializado con {len(self.tracked_features)} features robustos")
        else:
            # Fallback a goodFeaturesToTrack si ORB falla
            logger.warning("ORB no encontró suficientes features, usando Shi-Tomasi como fallback")
            features_roi = cv2.goodFeaturesToTrack(gray_roi_enhanced, mask=None, **self.feature_params)
            
            if features_roi is not None and len(features_roi) > 0:
                features_roi = features_roi.reshape(-1, 2)
                self.tracked_features = features_roi + np.array([x, y], dtype=np.float32)
                logger.info(f"Shi-Tomasi tracking con {len(self.tracked_features)} features")
            else:
                self.tracked_features = None
                logger.error("No se detectaron features en ROI inicial")
        
        # Guardar frame completo para tracking
        self.prev_full_gray = gray
        self.prev_gray = gray_roi
        
        # Reset offsets
        self.roi_offset_x = 0.0
        self.roi_offset_y = 0.0
    
    def analyze_frame(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Tuple[float, float, bool]:
        """
        Analyze frame for camera movement.
        
        Args:
            frame: Current BGR frame
            roi: Region of Interest (x, y, w, h) for optical flow - should be static area
            
        Returns:
            Tuple of (dx, dy, is_vibrating)
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Extract ROI region if specified
            if roi is not None:
                x, y, w, h = roi
                gray_roi = gray[y:y+h, x:x+w]
            else:
                gray_roi = gray
            
            if self.prev_gray is None:
                self.prev_gray = gray_roi
                self.prev_full_gray = gray
                return 0.0, 0.0, False
            
            # Calculate optical flow only in ROI (static region) para medir vibración
            dx, dy = self._calculate_movement(self.prev_gray, gray_roi)
            
            # Update history
            self.dx_history.append(dx)
            self.dy_history.append(dy)
            
            # Detect vibration
            is_vibrating = self._detect_vibration()
            
            # Update previous frames
            self.prev_gray = gray_roi
            self.prev_full_gray = gray
            
            return dx, dy, is_vibrating
            
        except Exception as e:
            logger.error(f"Error analyzing stability: {e}")
            return 0.0, 0.0, False
    
    def _calculate_movement(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> Tuple[float, float]:
        """
        Calculate camera movement using optical flow.
        Uses cached features and only recalculates every 10 frames for performance.
        """
        # Recalcular features solo cada 10 frames (optimización)
        if self.vibration_features is None or self.vibration_feature_counter % 10 == 0:
            self.vibration_features = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)
        
        self.vibration_feature_counter += 1
        
        if self.vibration_features is None or len(self.vibration_features) < 10:
            return 0.0, 0.0
        
        # Calculate optical flow from cached features
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, self.vibration_features, None, **self.lk_params)
        
        if p1 is None:
            self.vibration_features = None
            return 0.0, 0.0
        
        # Select good points
        good_new = p1[st == 1]
        good_old = self.vibration_features[st == 1]
        
        if len(good_new) < 5:
            self.vibration_features = None
            return 0.0, 0.0
        
        # Update cached features for next frame
        self.vibration_features = good_new.reshape(-1, 1, 2)
        
        # Calculate median movement
        movements = good_new - good_old
        dx = np.median(movements[:, 0])
        dy = np.median(movements[:, 1])
        
        return float(dx), float(dy)
    
    def _detect_vibration(self) -> bool:
        """
        Detect if camera is vibrating based on movement history.
        """
        if len(self.dx_history) < self.history_size // 2:
            return False
        
        # Calculate movement magnitude
        movements = np.sqrt(np.array(self.dx_history)**2 + np.array(self.dy_history)**2)
        
        # Check for frequent high-magnitude movements
        high_movement_ratio = np.sum(movements > self.vibration_threshold) / len(movements)
        
        # Also check for oscillation pattern
        dx_std = np.std(self.dx_history)
        dy_std = np.std(self.dy_history)
        oscillation = (dx_std > 3.0) or (dy_std > 3.0)
        
        return high_movement_ratio > 0.3 or oscillation
    
    def adjust_roi(self, roi: Tuple[int, int, int, int], frame_shape: Tuple[int, int], 
                   current_frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Adjust ROI based on feature tracking (self-healing).
        El ROI "persigue" features estáticos físicos aunque la cámara se mueva (pan/tilt).
        
        Args:
            roi: Current ROI (x, y, w, h)
            frame_shape: Frame shape (height, width)
            current_frame: Current BGR frame for tracking
            
        Returns:
            Adjusted ROI (x, y, w, h) que sigue los features estáticos
        """
        if self.tracked_features is None or self.prev_full_gray is None:
            # No hay features tracked, usar método antiguo (fallback)
            return self._adjust_roi_fallback(roi, frame_shape)
        
        x, y, w, h = roi
        height, width = frame_shape
        
        try:
            # Convertir frame actual a gris
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Convertir features a formato esperado por calcOpticalFlowPyrLK (N, 1, 2)
            features_for_tracking = self.tracked_features.reshape(-1, 1, 2)
            
            # Trackear features desde frame anterior al actual
            new_features, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_full_gray, gray, features_for_tracking, None, **self.lk_params
            )
            
            if new_features is None or status is None:
                logger.warning("Feature tracking failed, using fallback")
                return self._adjust_roi_fallback(roi, frame_shape)
            
            # Reshape a (N, 2) para operaciones
            new_features = new_features.reshape(-1, 2)
            old_features = self.tracked_features
            
            # Filtrar solo features bien trackeados
            good_mask = status.flatten() == 1
            good_old = old_features[good_mask]
            good_new = new_features[good_mask]
            
            if len(good_new) < 10:  # Necesitamos al menos 10 features
                logger.warning(f"Solo {len(good_new)} features válidos, re-inicializando tracking")
                # Re-detectar features en ROI actual
                self._reinitialize_features(current_frame, roi)
                return roi
            
            # Calcular desplazamiento mediano de los features (robusto a outliers)
            displacement = good_new - good_old
            median_dx = float(np.median(displacement[:, 0]))
            median_dy = float(np.median(displacement[:, 1]))
            
            # Guardar features previos para visualización de movimiento
            self.prev_tracked_features = good_old.copy()
            
            # CRÍTICO: NO actualizar tracked_features aquí
            # Los features deben permanecer en las coordenadas físicas de las rocas
            # Solo usamos good_new para calcular el desplazamiento y mover el ROI
            # La actualización de features se hace solo cuando se mueve el ROI (abajo)
            
            # Acumular offset (el ROI debe moverse en dirección opuesta al movimiento de features)
            # Si los features se movieron a la derecha, la cámara hizo pan a la izquierda
            # entonces el ROI debe moverse a la derecha para seguir los mismos features físicos
            self.roi_offset_x += median_dx
            self.roi_offset_y += median_dy
            
            # Calcular nueva posición del ROI
            new_x = int(x + median_dx)
            new_y = int(y + median_dy)
            
            # Asegurar que ROI permanezca dentro del frame
            new_x = max(0, min(new_x, width - w))
            new_y = max(0, min(new_y, height - h))
            
            # Actualizar features trackeados SOLO con los que se movieron con el ROI
            # Mantener las coordenadas relativas dentro del ROI ajustado
            # Los features deben "seguir" las rocas físicas en el nuevo ROI
            self.tracked_features = good_new
            
            # Debug: Verificar que los features se actualizaron
            if len(good_new) > 0:
                logger.debug(f"Features actualizados de {good_old[0]} a {good_new[0]} (dx={median_dx:.2f}, dy={median_dy:.2f})")
            
            # Log solo si hubo movimiento significativo
            if abs(median_dx) > 1.0 or abs(median_dy) > 1.0:
                logger.info(f"ROI ajustado: dx={median_dx:.1f}px, dy={median_dy:.1f}px | "
                           f"Offset acumulado: ({self.roi_offset_x:.1f}, {self.roi_offset_y:.1f})px | "
                           f"Features: {len(good_new)}")
            
            return (new_x, new_y, w, h)
            
        except Exception as e:
            logger.error(f"Error en feature tracking: {e}")
            return self._adjust_roi_fallback(roi, frame_shape)
    
    def _adjust_roi_fallback(self, roi: Tuple[int, int, int, int], 
                            frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Fallback: ajustar ROI usando promedio de drift (método antiguo).
        """
        if len(self.dx_history) == 0:
            return roi
        
        x, y, w, h = roi
        height, width = frame_shape
        
        # Calculate cumulative drift
        cumulative_dx = -np.mean(list(self.dx_history)[-10:])
        cumulative_dy = -np.mean(list(self.dy_history)[-10:])
        
        # Adjust ROI position
        new_x = int(x + cumulative_dx)
        new_y = int(y + cumulative_dy)
        
        # Ensure ROI stays within frame bounds
        new_x = max(0, min(new_x, width - w))
        new_y = max(0, min(new_y, height - h))
        
        return (new_x, new_y, w, h)
    
    def _reinitialize_features(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        """
        Re-detectar features en el ROI actual cuando se pierden, usando ORB.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = roi
        gray_roi = gray[y:y+h, x:x+w]
        
        # CLAHE para mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_roi_enhanced = clahe.apply(gray_roi)
        
        # Intentar ORB primero
        keypoints = self.orb.detect(gray_roi_enhanced, None)
        
        if keypoints and len(keypoints) > 10:
            features_roi = np.array([kp.pt for kp in keypoints[:50]], dtype=np.float32)
            self.tracked_features = features_roi + np.array([x, y], dtype=np.float32)
            logger.info(f"Features ORB re-inicializados: {len(self.tracked_features)} puntos")
        else:
            # Fallback a Shi-Tomasi
            features_roi = cv2.goodFeaturesToTrack(gray_roi_enhanced, mask=None, **self.feature_params)
            
            if features_roi is not None and len(features_roi) > 0:
                features_roi = features_roi.reshape(-1, 2)
                self.tracked_features = features_roi + np.array([x, y], dtype=np.float32)
                logger.info(f"Features Shi-Tomasi re-inicializados: {len(self.tracked_features)} puntos")
            else:
                self.tracked_features = None
                logger.warning("No se pudieron re-inicializar features")
    
    def get_stability_metrics(self) -> dict:
        """Get current stability metrics."""
        if len(self.dx_history) == 0:
            return {'status': 'INITIALIZING'}
        
        avg_dx = np.mean(self.dx_history)
        avg_dy = np.mean(self.dy_history)
        std_dx = np.std(self.dx_history)
        std_dy = np.std(self.dy_history)
        
        magnitude = np.sqrt(avg_dx**2 + avg_dy**2)
        
        return {
            'avg_dx': float(avg_dx),
            'avg_dy': float(avg_dy),
            'std_dx': float(std_dx),
            'std_dy': float(std_dy),
            'magnitude': float(magnitude),
            'is_stable': magnitude < self.vibration_threshold
        }
