"""
Module 2: Stability Analysis with Dynamic ROI Tracking
Detecta vibración de cámara y ajusta ROI dinámicamente para seguir la misma zona física.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from collections import deque
import logging

logger = logging.getLogger(__name__)


class StabilityAnalyzer:
    """Detects vibration patterns and provides self-healing ROI adjustments."""
    
    def __init__(self, history_size: int = 30, vibration_threshold: float = 1.0):
        self.history_size = history_size
        self.vibration_threshold = vibration_threshold
        
        # Movement history
        self.dx_history = deque(maxlen=history_size)
        self.dy_history = deque(maxlen=history_size)
        
        # Previous frame
        self.prev_gray = None
        
        # Tracking features para visualización Y vibración (ahora son los mismos)
        self.tracked_features = None
        self.prev_tracked_features = None
        
        # Reference ROI para transformación
        self.reference_roi_corners = None
        self.M_current = None
        self.current_roi = None  # Guardar ROI actual para recalcular features
        
        # Optical flow parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7
        )
        
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        self.camera_offset_x = 0.0
        self.camera_offset_y = 0.0
        self.frame_count = 0
    
    def set_reference_frame(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        """Initialize tracking with first frame and ROI."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = roi
        self.current_roi = roi
        
        # Detect features in ROI (se usan para tracking Y vibración)
        gray_roi = gray[y:y+h, x:x+w]
        features_roi = cv2.goodFeaturesToTrack(gray_roi, mask=None, **self.feature_params)
        
        if features_roi is not None and len(features_roi) > 0:
            features_roi = features_roi.reshape(-1, 2)
            self.tracked_features = features_roi + np.array([x, y], dtype=np.float32)
            self.prev_tracked_features = self.tracked_features.copy()
            logger.info(f"ROI tracking initialized with {len(self.tracked_features)} features in ROI")
        
        # Save reference
        self.prev_gray = gray
        self.reference_roi_corners = np.float32([
            [x, y], [x + w, y], [x + w, y + h], [x, y + h]
        ]).reshape(-1, 1, 2)
        
        return True
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze frame for vibration using sparse optical flow."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return self._get_default_result()
        self.frame_count += 1
        
        # Update tracked features usando optical flow DIRECTO
        # Ahora los tracked_features se usan para TODO (vibración + self-healing)
        # Update tracked features usando optical flow DIRECTO (no el promedio)
        roi_dx, roi_dy = 0.0, 0.0
        if self.tracked_features is not None and self.prev_tracked_features is not None:
            # Track features directamente
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, 
                self.prev_tracked_features.reshape(-1, 1, 2), 
                None, **self.lk_params
            )
            
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = self.prev_tracked_features[st.flatten() == 1]
                
                if len(good_new) > 2:  # Reducido de 5 a 2 features mínimas
                    # Calcular el movimiento promedio de las features tracked
                    movements = good_new - good_old
                    roi_dx = float(np.mean(movements[:, 0]))
                    roi_dy = float(np.mean(movements[:, 1]))
                    
                    self.tracked_features = good_new.reshape(-1, 2)
                    self.prev_tracked_features = self.tracked_features.copy()
                else:
                    # Si perdimos demasiadas features, recalcular en el ROI
                    logger.warning(f"Only {len(good_new)} features tracked, recalculating...")
                    self._recalculate_roi_features(gray)
        
        # Update history - USAR ROI DX/DY para vibración
        self.dx_history.append(roi_dx)
        self.dy_history.append(roi_dy)
        
        # Detect vibration
        is_vibrating = self._detect_vibration()
        
        # Update offsets - USAR EL MOVIMIENTO DE LAS FEATURES TRACKED para ROI
        self.camera_offset_x += roi_dx
        self.camera_offset_y += roi_dy
        
        # Update transformation matrix
        if self.M_current is None:
            self.M_current = np.float32([[1, 0, self.camera_offset_x], [0, 1, self.camera_offset_y]])
        else:
            self.M_current[0, 2] = self.camera_offset_x
            self.M_current[1, 2] = self.camera_offset_y
        
        self.prev_gray = gray
        
        return {
            'displacement_x': float(roi_dx),
            'displacement_y': float(roi_dy),
            'camera_offset_x': float(self.camera_offset_x),
            'camera_offset_y': float(self.camera_offset_y),
            'is_vibrating': is_vibrating,
            'good_matches': 0,
            'frame_count': self.frame_count
        }
    
    def _recalculate_roi_features(self, gray: np.ndarray):
        """Recalculate features dentro del ROI actual ajustado."""
        if self.current_roi is None:
            return
        
        # Ajustar ROI con offset actual
        x, y, w, h = self.current_roi
        adjusted_x = int(x + self.camera_offset_x)
        adjusted_y = int(y + self.camera_offset_y)
        
        # Asegurar que esté dentro del frame
        h_frame, w_frame = gray.shape[:2]
        adjusted_x = max(0, min(adjusted_x, w_frame - w))
        adjusted_y = max(0, min(adjusted_y, h_frame - h))
        
        # Detectar nuevos features en el ROI ajustado
        gray_roi = gray[adjusted_y:adjusted_y+h, adjusted_x:adjusted_x+w]
        features_roi = cv2.goodFeaturesToTrack(gray_roi, mask=None, **self.feature_params)
        
        if features_roi is not None and len(features_roi) > 0:
            features_roi = features_roi.reshape(-1, 2)
            self.tracked_features = features_roi + np.array([adjusted_x, adjusted_y], dtype=np.float32)
            self.prev_tracked_features = self.tracked_features.copy()
            logger.debug(f"Recalculated {len(self.tracked_features)} features in ROI")
    
    def _detect_vibration(self) -> bool:
        """Detect if camera is vibrating (simple: avg últimos 10 frames > 0.3px)."""
        if len(self.dx_history) < 5:
            return False
        
        dx_list = list(self.dx_history)
        dy_list = list(self.dy_history)
        
        recent_window = 10
        recent_dx = dx_list[-recent_window:] if len(dx_list) >= recent_window else dx_list
        recent_dy = dy_list[-recent_window:] if len(dy_list) >= recent_window else dy_list
        
        movements = np.sqrt(np.array(recent_dx)**2 + np.array(recent_dy)**2)
        recent_avg = np.mean(movements)
        
        return recent_avg > 0.3
    
    def adjust_roi(self, roi_original: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Adjust ROI based on camera offset."""
        if self.M_current is None:
            return roi_original
        
        x, y, w, h = roi_original
        
        # Apply offset
        new_x = int(x + self.camera_offset_x)
        new_y = int(y + self.camera_offset_y)
        
        # Keep within bounds (assuming 720p)
        new_x = max(0, min(new_x, 1280 - w))
        new_y = max(0, min(new_y, 720 - h))
        
        return (new_x, new_y, w, h)
    
    def get_transformed_roi_corners(self) -> Optional[np.ndarray]:
        """Get transformed ROI corners for visualization."""
        if self.M_current is None or self.reference_roi_corners is None:
            return None
        
        return cv2.transform(self.reference_roi_corners, self.M_current)
    
    def _get_default_result(self) -> Dict:
        """Default result when analysis fails."""
        return {
            'displacement_x': 0.0,
            'displacement_y': 0.0,
            'camera_offset_x': self.camera_offset_x,
            'camera_offset_y': self.camera_offset_y,
            'is_vibrating': False,
            'good_matches': 0,
            'frame_count': self.frame_count
        }
    
    def get_stats(self) -> Dict:
        """Get stability statistics."""
        if len(self.dx_history) == 0:
            return {'avg_displacement': 0.0, 'max_displacement': 0.0}
        
        magnitudes = [np.sqrt(dx**2 + dy**2) for dx, dy in zip(self.dx_history, self.dy_history)]
        
        return {
            'avg_displacement': np.mean(magnitudes),
            'max_displacement': np.max(magnitudes)
        }