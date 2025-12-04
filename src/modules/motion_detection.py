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
    
    def __init__(self, model_size: str = 's', engine_path: str = None, imgsz: int = 640):
        """
        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            engine_path: Path to TensorRT engine (auto-generated if None)
            imgsz: Input image size for YOLO (must be multiple of 32: 640, 1024, 1280)
        """
        self.model = None
        self.model_size = model_size
        self.imgsz = imgsz
        self.model_name = f'yolov8{model_size}'
        # Include imgsz in engine name so different sizes don't clash
        self.engine_path = engine_path or f'models/{self.model_name}_{imgsz}.engine'
        
        if Path(self.engine_path).exists():
            logger.info(f"Loading existing TensorRT engine: {self.engine_path}")
            self._load_tensorrt(self.engine_path)
        else:
            logger.info(f"TensorRT engine not found. Generating: {self.engine_path}")
            self._generate_tensorrt_engine()
            if Path(self.engine_path).exists():
                self._load_tensorrt(self.engine_path)
            else:
                logger.warning("TensorRT generation failed, using PyTorch")
                self._load_yolo()
    
    def _generate_tensorrt_engine(self):
        """Generate TensorRT engine from YOLOv8 model."""
        try:
            from ultralytics import YOLO
            logger.info(f"Exporting {self.model_name} to TensorRT with imgsz={self.imgsz} (this may take a few minutes)...")
            model = YOLO(f'{self.model_name}.pt')
            model.export(format='engine', device=0, half=True, imgsz=self.imgsz)
            
            # Move generated engine to models directory
            import shutil
            import os
            os.makedirs('models', exist_ok=True)
            engine_file = f'{self.model_name}.engine'
            if Path(engine_file).exists():
                shutil.move(engine_file, self.engine_path)
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
            self.model = YOLO(f'{self.model_name}.pt')
            logger.info(f"{self.model_name} PyTorch model loaded (fallback)")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.3) -> List[Dict]:
        """
        Detect objects using YOLO.
        Returns: List of detections with boxes, class, and confidence
        """
        if self.model is None:
            return []
        
        # Mining vehicle classes: truck, car, bus (ignore boat, airplane)
        MINING_RELEVANT_CLASSES = {'truck', 'car', 'bus'}
        TRUCK_THRESHOLD = 0.10  # Lower threshold for trucks (mining vehicles)
        
        try:
            # Use higher IoU threshold for NMS to reduce duplicate boxes
            results = self.model(frame, conf=0.08, imgsz=self.imgsz, iou=0.5, verbose=False)  # iou=0.5 removes more duplicates
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = result.names[cls] if hasattr(result, 'names') else str(cls)
                        
                        # Filter: only mining-relevant vehicles
                        if class_name not in MINING_RELEVANT_CLASSES:
                            continue
                        
                        # Apply class-specific thresholds
                        if class_name == 'truck' and conf < TRUCK_THRESHOLD:
                            continue
                        elif class_name != 'truck' and conf < conf_threshold:
                            continue
                        
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
    
    def __init__(self, use_yolo: bool = True, yolo_model: str = 's', yolo_imgsz: int = 640):
        """
        Args:
            use_yolo: Enable YOLO detector
            yolo_model: YOLO model size ('n', 's', 'm', 'l', 'x')
            yolo_imgsz: YOLO input image size (640, 1024, 1280, etc.)
        """
        self.yolo = YOLODetector(model_size=yolo_model, imgsz=yolo_imgsz) if use_yolo else None
        logger.info(f"YOLO detection pipeline initialized (model: yolov8{yolo_model}, imgsz: {yolo_imgsz})")
    
    def process_frame(self, frame: np.ndarray, conf_threshold: float = 0.25) -> dict:
        """
        Process frame with YOLO detector.
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold for YOLO (0.0-1.0)
        
        Returns:
            dict with 'yolo' bounding boxes
        """
        results = {'yolo': []}
        
        if self.yolo:
            results['yolo'] = self.yolo.detect(frame, conf_threshold=conf_threshold)
        
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
