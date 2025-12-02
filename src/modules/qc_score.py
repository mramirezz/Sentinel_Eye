"""
Module 1: Image Quality Control (QC) Score
Evaluates image health across multiple factors:
- Sharpness (35%): Laplacian variance
- Occlusion Detection (25%): Edge density analysis
- Lighting Level (20%): Histogram analysis
- Lens Cleanliness (20%): Blur and spot detection
"""

import cv2
import numpy as np
import torch
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ImageQualityChecker:
    """
    Evaluates image quality and returns a weighted QC score (0-100).
    GPU-accelerated with PyTorch.
    """
    
    def __init__(self, config: dict = None):
        # Load weights from config or use defaults
        if config and 'qc_score' in config and 'weights' in config['qc_score']:
            self.weights = config['qc_score']['weights']
        else:
            self.weights = {
                'sharpness': 0.35,
                'occlusion': 0.25,
                'lighting': 0.20,
                'cleanliness': 0.20
            }
        
        # Load hyperparameters from config or use defaults
        if config and 'qc_score' in config and 'hyperparameters' in config['qc_score']:
            hp = config['qc_score']['hyperparameters']
            self.sharpness_divisor = hp.get('sharpness_divisor', 8.0)
            self.min_edge_density = hp.get('min_edge_density', 0.04)
            self.edge_density_multiplier = hp.get('edge_density_multiplier', 500)
            self.ideal_brightness = hp.get('ideal_brightness', 115)
            self.brightness_divisor = hp.get('brightness_divisor', 0.8)
            self.contrast_multiplier = hp.get('contrast_multiplier', 1.6)
            self.spot_threshold = hp.get('spot_threshold', 15)
            self.spot_penalty_multiplier = hp.get('spot_penalty_multiplier', 800)
        else:
            # Default values (strict)
            self.sharpness_divisor = 8.0
            self.min_edge_density = 0.04
            self.edge_density_multiplier = 500
            self.ideal_brightness = 115
            self.brightness_divisor = 0.8
            self.contrast_multiplier = 1.6
            self.spot_threshold = 15
            self.spot_penalty_multiplier = 800
        
        # Use CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"QC Score usando: {self.device}")
        
    def compute_qc_score(self, frame: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Compute overall QC score and individual metrics.
        
        Args:
            frame: Input BGR image
            
        Returns:
            Tuple of (overall_score, metrics_dict)
        """
        try:
            if frame is None or frame.size == 0:
                logger.warning("Empty frame received")
                return 0.0, {}
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Compute individual metrics
            sharpness_score = self._check_sharpness(gray)
            occlusion_score = self._check_occlusion(gray)
            lighting_score = self._check_lighting(gray)
            cleanliness_score = self._check_cleanliness(gray)
            
            # Weighted overall score
            overall_score = (
                sharpness_score * self.weights['sharpness'] +
                occlusion_score * self.weights['occlusion'] +
                lighting_score * self.weights['lighting'] +
                cleanliness_score * self.weights['cleanliness']
            )
            
            metrics = {
                'sharpness': sharpness_score,
                'occlusion': occlusion_score,
                'lighting': lighting_score,
                'cleanliness': cleanliness_score,
                'overall': overall_score
            }
            
            return overall_score, metrics
            
        except Exception as e:
            logger.error(f"Error computing QC score: {e}")
            return 0.0, {}
    
    def _check_sharpness(self, gray: np.ndarray) -> float:
        """
        Check image sharpness using Laplacian variance (GPU-accelerated).
        Higher variance = sharper image.
        """
        # Convert to PyTorch tensor on GPU
        gray_tensor = torch.from_numpy(gray).float().to(self.device)
        
        # Laplacian kernel
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        # Apply convolution
        gray_4d = gray_tensor.unsqueeze(0).unsqueeze(0)
        laplacian = torch.nn.functional.conv2d(gray_4d, laplacian_kernel, padding=1)
        variance = torch.var(laplacian).item()
        
        # Normalize to 0-100 (typical range: 0-500) - Configurable strictness
        score = min(100, (variance / self.sharpness_divisor))
        return score
    
    def _check_occlusion(self, gray: np.ndarray) -> float:
        """
        Detect occlusion by analyzing edge density (GPU-accelerated).
        Too few edges might indicate occlusion/blockage.
        """
        # Use Canny on CPU (OpenCV is fast enough for this)
        edges = cv2.Canny(gray, 50, 150)
        
        # Move to GPU for sum calculation
        edges_tensor = torch.from_numpy(edges).to(self.device)
        edge_density = (edges_tensor > 0).sum().item() / edges_tensor.numel()
        
        # Normalize: configurable edge density threshold
        if edge_density < self.min_edge_density:
            score = edge_density * self.edge_density_multiplier
        elif edge_density > 0.25:
            score = max(0, 100 - (edge_density - 0.25) * 200)
        else:
            score = 100
            
        return min(100, max(0, score))
    
    def _check_lighting(self, gray: np.ndarray) -> float:
        """
        Check if lighting is adequate using histogram analysis (GPU-accelerated).
        """
        # Convert to GPU tensor
        gray_tensor = torch.from_numpy(gray).float().to(self.device)
        
        mean_brightness = torch.mean(gray_tensor).item()
        std_brightness = torch.std(gray_tensor).item()
        
        # Configurable brightness and contrast sensitivity
        brightness_score = max(0, 100 - abs(mean_brightness - self.ideal_brightness) / self.brightness_divisor)
        contrast_score = min(100, std_brightness * self.contrast_multiplier)
        
        # Average both factors
        score = (brightness_score * 0.6 + contrast_score * 0.4)
        return max(0, min(100, score))
    
    def _check_cleanliness(self, gray: np.ndarray) -> float:
        """
        Detect lens dirt/spots using blur and blob detection.
        """
        # Apply median blur to detect spots
        blurred = cv2.medianBlur(gray, 5)
        diff = cv2.absdiff(gray, blurred)
        
        # Configurable spot detection threshold and penalty
        _, thresh = cv2.threshold(diff, self.spot_threshold, 255, cv2.THRESH_BINARY)
        spot_density = np.sum(thresh > 0) / thresh.size
        
        # Lower spot density = cleaner lens (configurable penalty)
        score = max(0, 100 - (spot_density * self.spot_penalty_multiplier))
        return min(100, score)
    
    def get_status_message(self, score: float) -> str:
        """Get human-readable status based on QC score."""
        if score >= 80:
            return "EXCELLENT - Optimal conditions"
        elif score >= 60:
            return "GOOD - Acceptable quality"
        elif score >= 40:
            return "WARNING - Degraded quality"
        else:
            return "CRITICAL - Poor image quality"
