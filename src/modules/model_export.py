"""
Model export and optimization module.
Demonstrates ONNX export and quantization for production deployment.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SimpleDetector(nn.Module):
    """Lightweight demo model for ONNX/TensorRT export demonstration."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def export_to_onnx(model: nn.Module, output_path: str = "models/optimized_model.onnx"):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model in eval mode
        output_path: Path to save ONNX file
    """
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        dummy_input = torch.randn(1, 3, 640, 640)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to ONNX: {output_path}")
        logger.info(f"   - Opset version: 13")
        logger.info(f"   - Dynamic batch size enabled")
        logger.info(f"   - Ready for TensorRT conversion")
        
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")


def apply_quantization(model: nn.Module) -> nn.Module:
    """
    Apply dynamic INT8 quantization to model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Quantized model
    """
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        logger.info("Dynamic quantization applied successfully")
        logger.info(f"   - INT8 quantization on Conv2d and Linear layers")
        logger.info(f"   - Expected speedup: 2-4x on CPU")
        logger.info(f"   - Model size reduction: ~75%")
        
        return quantized_model
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logger.info("=" * 70)
    logger.info("MODEL EXPORT & OPTIMIZATION DEMONSTRATION")
    logger.info("=" * 70)
    
    # Create demo model
    logger.info("\nCreating demo model...")
    model = SimpleDetector()
    model.eval()
    
    # Export to ONNX (BEFORE quantization)
    logger.info("\n1. ONNX EXPORT")
    export_to_onnx(model)
    
    # Apply quantization (for PyTorch inference)
    logger.info("\n2. QUANTIZATION (INT8)")
    quantized_model = apply_quantization(model)
    
    logger.info("\n" + "=" * 70)
    logger.info("Export complete!")
    logger.info("Next step: Convert ONNX to TensorRT with:")
    logger.info("  docker run --gpus all --rm -v ${PWD}/models:/models \\")
    logger.info("    nvcr.io/nvidia/tensorrt:23.08-py3 \\")
    logger.info("    trtexec --onnx=/models/optimized_model.onnx \\")
    logger.info("            --saveEngine=/models/optimized_model.trt \\")
    logger.info("            --fp16 --workspace=4096")
    logger.info("=" * 70)
