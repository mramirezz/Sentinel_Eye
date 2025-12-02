"""
Module 3: Performance Optimization
Implements various optimization techniques to maximize FPS.
"""

import cv2
import numpy as np
import torch
from typing import Optional, List
import logging
import time

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Handles performance optimization for real-time processing.
    """
    
    def __init__(self, 
                 target_resolution: Optional[tuple] = None,
                 use_gpu: bool = True):
        """
        Args:
            target_resolution: Target resolution for downscaling (width, height)
            use_gpu: Whether to use GPU acceleration
        """
        self.target_resolution = target_resolution or (640, 480)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Performance metrics
        self.frame_times = []
        self.processing_times = []
        
        logger.info(f"Performance Optimizer initialized")
        logger.info(f"GPU Available: {torch.cuda.is_available()}")
        logger.info(f"Using GPU: {self.use_gpu}")
        if self.use_gpu:
            logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    def preprocess_frame(self, frame: np.ndarray, resize: bool = True) -> np.ndarray:
        """
        Optimize frame preprocessing.
        
        Args:
            frame: Input BGR frame
            resize: Whether to resize to target resolution
            
        Returns:
            Preprocessed frame
        """
        start_time = time.time()
        
        try:
            if resize:
                # Use INTER_LINEAR for faster resizing (vs INTER_CUBIC)
                frame = cv2.resize(frame, self.target_resolution, interpolation=cv2.INTER_LINEAR)
            
            # Record processing time
            self.processing_times.append(time.time() - start_time)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return frame
    
    def optimize_cv_operations(self):
        """
        Configure OpenCV for optimal performance.
        """
        # Set number of threads for OpenCV
        cv2.setNumThreads(4)
        
        # Enable OpenCL if available
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            logger.info("OpenCL enabled for OpenCV")
    
    def get_fps(self) -> float:
        """Calculate current FPS based on recent frame times."""
        if len(self.frame_times) < 2:  # Need at least 2 frames
            return 0.0
        
        # Use last 10 frames for more stable calculation
        recent_times = self.frame_times[-10:] if len(self.frame_times) >= 10 else self.frame_times
        
        if len(recent_times) < 2:
            return 0.0
        
        time_span = recent_times[-1] - recent_times[0]
        
        # Avoid division by zero, but allow very fast processing
        # Changed from 0.1 to 0.001 (1ms minimum)
        if time_span < 0.001:
            return 0.0
        
        fps = (len(recent_times) - 1) / time_span
        
        # Cap FPS at reasonable maximum (100 FPS)
        # Return at least 0.1 FPS to avoid showing 0 in graphs
        return max(0.1, min(fps, 100.0))
    
    def record_frame_time(self):
        """Record timestamp for FPS calculation."""
        self.frame_times.append(time.time())
        
        # Keep only last 100 frames
        if len(self.frame_times) > 100:
            self.frame_times = self.frame_times[-100:]
    
    def get_performance_metrics(self) -> dict:
        """Get current performance statistics."""
        fps = self.get_fps()
        
        avg_processing_time = 0
        if self.processing_times:
            avg_processing_time = np.mean(self.processing_times[-30:]) * 1000  # ms
        
        return {
            'fps': fps,
            'avg_processing_time_ms': avg_processing_time,
            'gpu_enabled': self.use_gpu,
            'target_resolution': self.target_resolution
        }
    
    def apply_frame_skip(self, frame_count: int, skip_factor: int = 2) -> bool:
        """
        Decide whether to skip frame for performance.
        
        Args:
            frame_count: Current frame number
            skip_factor: Process every Nth frame
            
        Returns:
            True if should process, False if should skip
        """
        return frame_count % skip_factor == 0
    
    def release_resources(self):
        """Clean up resources."""
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
