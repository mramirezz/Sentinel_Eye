"""
Module 3: Object Detection with YOLOv8
Fast and accurate object detection with ONNX export capability.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLOv8 object detector with TensorRT support."""
    
    def __init__(self, engine_path: str = 'models/yolov8n.engine'):
        self.model = None
        self.engine_path = engine_path
        
        if Path(engine_path).exists():
            logger.info(f"Loading existing TensorRT engine: {engine_path}")
            self._load_tensorrt(engine_path)
        else:
            logger.info(f"TensorRT engine not found. Generating: {engine_path}")
            self._generate_tensorrt_engine()
            if Path(engine_path).exists():
                self._load_tensorrt(engine_path)
            else:
                logger.warning("TensorRT generation failed, using PyTorch")
                self._load_yolo()
    
    def _generate_tensorrt_engine(self):
        """Generate TensorRT engine from YOLOv8 model."""
        try:
            from ultralytics import YOLO
            logger.info("Exporting YOLOv8n to TensorRT (this may take a few minutes)...")
            model = YOLO('yolov8n.pt')
            model.export(format='engine', device=0, half=True, imgsz=640)
            
            # Move generated engine to models directory
            import shutil
            import os
            os.makedirs('models', exist_ok=True)
            if Path('yolov8n.engine').exists():
                shutil.move('yolov8n.engine', self.engine_path)
                logger.info(f"TensorRT engine generated successfully: {self.engine_path}")
        except Exception as e:
            logger.error(f"Failed to generate TensorRT engine: {e}")
    
    def _load_tensorrt(self, engine_path: str):
        """Load YOLOv8 with TensorRT engine."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(engine_path, task='detect')
            logger.info(f"TensorRT engine loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TensorRT: {e}")
            self._load_yolo()
    
    def _load_yolo(self):
        """Load YOLOv8 PyTorch model (fallback)."""
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            logger.info("YOLOv8n PyTorch model loaded (fallback)")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.1) -> List[Dict]:
        """
        Detect objects using YOLO (filtered to truck class only).
        Returns: List of detections with boxes, class, and confidence
        """
        if self.model is None:
            return []
        
        try:
            # YOLO inference con TensorRT engine (640x640 default)
            results = self.model(frame, conf=conf_threshold, verbose=False)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = result.names[cls] if hasattr(result, 'names') else str(cls)
                        
                        detection = {
                            'box': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                            'confidence': conf,
                            'class': class_name,
                            'class_id': cls
                        }
                        detections.append(detection)
            
            return detections
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []


class OptimizedDetectionPipeline:
    """YOLO-based object detection pipeline."""
    
    def __init__(self, 
                 use_yolo: bool = True,
                 use_background_sub: bool = False,
                 frame_skip: int = 1):
        """
        Args:
            use_yolo: Enable YOLO detector
            use_background_sub: Ignored (kept for compatibility)
            frame_skip: Ignored (kept for compatibility)
        """
        self.yolo = YOLODetector() if use_yolo else None
        logger.info("YOLO detection pipeline initialized")
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process frame with YOLO detector.
        
        Returns:
            dict with 'yolo' bounding boxes
        """
        results = {'yolo': []}
        
        if self.yolo:
            results['yolo'] = self.yolo.detect(frame)
        
        return results


def draw_detections(frame: np.ndarray, detections: List[Dict], 
                    color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw bounding boxes with labels and confidence on frame."""
    output = frame.copy()
    
    for detection in detections:
        box = detection['box']
        conf = detection['confidence']
        cls = detection['class']
        
        x, y, w, h = box
        
        # Draw bounding box
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        
        # Draw label with confidence
        label = f"{cls} {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background for label
        cv2.rectangle(output, (x, y - label_size[1] - 10), 
                     (x + label_size[0] + 10, y), color, -1)
        cv2.putText(output, label, (x + 5, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return output
