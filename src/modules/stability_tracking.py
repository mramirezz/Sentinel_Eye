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
    """
    Detects vibration patterns and provides self-healing ROI adjustments.
    """
    
    def __init__(self, history_size: int = 30, vibration_threshold: float = 1.0):
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
        
        # Previous frame for optical flow (vibration detection)
        self.prev_gray = None
        
        # Feature tracking for ROI self-healing
        self.tracked_features = None
        self.prev_tracked_features = None
        self.prev_full_gray = None
        self.roi_offset_x = 0.0
        self.roi_offset_y = 0.0
        
        # Reference ROI
        self.reference_roi = None
        self.reference_roi_corners = None
        
        # Transformation matrix for ROI
        self.M_current = None
        self.M_prev = None
        self.alpha = 0.7
        
        # ORB detector for robust features
        self.orb = cv2.ORB_create(
            nfeatures=100,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31
        )
        
        # Cache for vibration detection
        self.vibration_features = None
        self.vibration_feature_counter = 0
        
        # Optical flow parameters
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
        
        self.camera_offset_x = 0.0
        self.camera_offset_y = 0.0
        self.frame_count = 0
    
    def set_reference_frame(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        """Initialize tracking with first frame and ROI."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = roi
        
        # Extract ROI
        gray_roi = gray[y:y+h, x:x+w]
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_roi_enhanced = clahe.apply(gray_roi)
        
        # Detect ORB keypoints in ROI
        keypoints = self.orb.detect(gray_roi_enhanced, None)
        
        if keypoints and len(keypoints) > 10:
            features_roi = np.array([kp.pt for kp in keypoints[:50]], dtype=np.float32)
            self.tracked_features = features_roi + np.array([x, y], dtype=np.float32)
            logger.info(f"ORB tracking initialized with {len(self.tracked_features)} features")
        else:
            # Fallback to goodFeaturesToTrack
            features_roi = cv2.goodFeaturesToTrack(gray_roi_enhanced, mask=None, **self.feature_params)
            
            if features_roi is not None and len(features_roi) > 0:
                features_roi = features_roi.reshape(-1, 2)
                self.tracked_features = features_roi + np.array([x, y], dtype=np.float32)
                logger.info(f"Shi-Tomasi tracking with {len(self.tracked_features)} features")
            else:
                self.tracked_features = None
                logger.error("No features detected in ROI")
                return False
        
        # Save frames
        self.prev_full_gray = gray
        self.prev_gray = gray_roi
        self.reference_roi = roi
        
        # Save ROI corners for transformation
        self.reference_roi_corners = np.float32([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ]).reshape(-1, 1, 2)
        
        self.roi_offset_x = 0.0
        self.roi_offset_y = 0.0
        
        return True
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze frame for vibration using sparse optical flow.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_full_gray = gray
            return self._get_default_result()
        
        self.frame_count += 1
        
        # Calculate movement using cached features
        dx, dy = self._calculate_movement(self.prev_gray, gray)
        
        # Update history
        self.dx_history.append(dx)
        self.dy_history.append(dy)
        
        # Detect vibration
        is_vibrating = self._detect_vibration()
        
        # Update offsets
        self.camera_offset_x += dx
        self.camera_offset_y += dy
        
        # Update transformation matrix
        if self.M_current is None:
            self.M_current = np.float32([[1, 0, self.camera_offset_x], [0, 1, self.camera_offset_y]])
        else:
            self.M_current[0, 2] = self.camera_offset_x
            self.M_current[1, 2] = self.camera_offset_y
        
        # Update previous frame
        self.prev_gray = gray
        self.prev_full_gray = gray
        
        return {
            'displacement_x': float(dx),
            'displacement_y': float(dy),
            'camera_offset_x': float(self.camera_offset_x),
            'camera_offset_y': float(self.camera_offset_y),
            'is_vibrating': is_vibrating,
            'good_matches': 0,
            'frame_count': self.frame_count
        }
    
    def _calculate_movement(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> Tuple[float, float]:
        """Calculate camera movement using sparse optical flow with feature caching."""
        # Recalculate features every 5 frames (antes era 10, ahora más frecuente)
        if self.vibration_features is None or self.vibration_feature_counter % 5 == 0:
            self.vibration_features = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)
        
        self.vibration_feature_counter += 1
        
        if self.vibration_features is None or len(self.vibration_features) < 10:
            return 0.0, 0.0
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, self.vibration_features, None, **self.lk_params)
        
        if p1 is None:
            self.vibration_features = None
            return 0.0, 0.0
        
        # Select good points
        good_new = p1[st == 1]
        good_old = self.vibration_features[st == 1]
        
        # Si perdimos muchas features (< 50%), recalcular inmediatamente
        tracking_ratio = len(good_new) / len(self.vibration_features)
        if tracking_ratio < 0.5:
            self.vibration_features = None
            return 0.0, 0.0
        
        if len(good_new) < 5:
            self.vibration_features = None
            return 0.0, 0.0
        
        # Update cached features
        self.vibration_features = good_new.reshape(-1, 1, 2)
        
        # Calculate movement - usar mean de los mejores puntos en vez de median global
        # Esto captura mejor la magnitud real de vibraciones grandes
        movements = good_new - good_old
        movement_magnitudes = np.sqrt(movements[:, 0]**2 + movements[:, 1]**2)
        
        # Filtrar outliers (> 10px son probablemente errores)
        valid_mask = movement_magnitudes < 10.0
        if np.sum(valid_mask) < 3:
            self.vibration_features = None
            return 0.0, 0.0
        
        valid_movements = movements[valid_mask]
        
        # Usar mean en vez de median para capturar mejor las vibraciones fuertes
        dx = np.mean(valid_movements[:, 0])
        dy = np.mean(valid_movements[:, 1])
        
        return float(dx), float(dy)
    
    def _detect_vibration(self) -> bool:
        """Detect if camera is vibrating."""
        if len(self.dx_history) < 5:
            return False
        
        # Convertir deque a lista
        dx_list = list(self.dx_history)
        dy_list = list(self.dy_history)
        
        # Usar últimos 10 frames
        recent_window = 10
        recent_dx = dx_list[-recent_window:] if len(dx_list) >= recent_window else dx_list
        recent_dy = dy_list[-recent_window:] if len(dy_list) >= recent_window else dy_list
        
        # Calculate movement magnitude
        movements = np.sqrt(np.array(recent_dx)**2 + np.array(recent_dy)**2)
        
        # Si HAY CUALQUIER movimiento promedio > 0.3px en últimos 10 frames = VIBRATING
        # Esto asegura que si la gráfica muestra movimiento, el STATUS también lo muestre
        recent_avg = np.mean(movements)
        
        return recent_avg > 0.3  # Simple y directo: movimiento = vibración
    
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
