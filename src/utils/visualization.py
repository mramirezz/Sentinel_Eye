"""
Visualization utilities for displaying results.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def draw_qc_metrics(frame: np.ndarray, metrics: Dict, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """
    Draw QC metrics on frame.
    
    Args:
        frame: Input frame
        metrics: Dictionary of QC metrics
        position: Starting position for text
        
    Returns:
        Frame with metrics drawn
    """
    frame_copy = frame.copy()
    x, y = position
    
    # Determine color based on overall score
    overall = metrics.get('overall', 0)
    if overall >= 80:
        color = (0, 255, 0)  # Green
    elif overall >= 60:
        color = (0, 255, 255)  # Yellow
    elif overall >= 40:
        color = (0, 165, 255)  # Orange
    else:
        color = (0, 0, 255)  # Red
    
    # Draw semi-transparent background
    overlay = frame_copy.copy()
    cv2.rectangle(overlay, (x-5, y-25), (x+400, y+150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame_copy, 0.4, 0, frame_copy)
    
    # Draw metrics
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    cv2.putText(frame_copy, f"QC SCORE: {overall:.1f}", (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame_copy, f"Sharpness: {metrics.get('sharpness', 0):.1f}", (x, y+30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_copy, f"Occlusion: {metrics.get('occlusion', 0):.1f}", (x, y+55), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_copy, f"Lighting: {metrics.get('lighting', 0):.1f}", (x, y+80), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_copy, f"Cleanliness: {metrics.get('cleanliness', 0):.1f}", (x, y+105), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame_copy


def draw_stability_info(frame: np.ndarray, dx: float, dy: float, is_vibrating: bool, 
                       position: Tuple[int, int] = (10, 200)) -> np.ndarray:
    """
    Draw stability information on frame.
    
    Args:
        frame: Input frame
        dx: X-axis drift
        dy: Y-axis drift
        is_vibrating: Whether vibration detected
        position: Starting position for text
        
    Returns:
        Frame with stability info drawn
    """
    frame_copy = frame.copy()
    x, y = position
    
    # Color based on vibration status
    color = (0, 0, 255) if is_vibrating else (0, 255, 0)
    status = "VIBRATING!" if is_vibrating else "STABLE"
    
    # Draw semi-transparent background
    overlay = frame_copy.copy()
    cv2.rectangle(overlay, (x-5, y-25), (x+350, y+80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame_copy, 0.4, 0, frame_copy)
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame_copy, f"STATUS: {status}", (x, y), font, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(frame_copy, f"Drift X: {dx:.2f}px", (x, y+30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_copy, f"Drift Y: {dy:.2f}px", (x, y+55), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame_copy


def draw_tracked_features(frame: np.ndarray, features: np.ndarray, 
                         prev_features: np.ndarray = None,
                         roi: Tuple[int, int, int, int] = None,
                         color: Tuple[int, int, int] = (0, 255, 255),
                         show_motion_vectors: bool = True) -> np.ndarray:
    """
    Draw tracked features (points) on frame to show what the ROI is following.
    Features se dibujan siguiendo las rocas físicas en el frame.
    
    Args:
        frame: Input frame
        features: Array of feature points (N, 2) in absolute frame coordinates
        prev_features: Previous feature positions to draw motion vectors
        roi: Optional ROI to only draw features inside it
        color: Color for feature points (default: cyan)
        show_motion_vectors: Whether to draw arrows showing feature movement
        
    Returns:
        Frame with features drawn
    """
    if features is None or len(features) == 0:
        return frame
    
    frame_copy = frame.copy()
    features_drawn = 0
    total_movement = 0.0
    
    for i, (x, y) in enumerate(features):
        x, y = int(x), int(y)
        
        # Verificar que el feature esté dentro del frame
        h, w = frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            continue
        
        # Solo dibujar si dentro del ROI (si se especificó)
        if roi is not None:
            rx, ry, rw, rh = roi
            if not (rx <= x < rx + rw and ry <= y < ry + rh):
                continue
        
        features_drawn += 1
        
        # Draw motion vector if previous features available
        if show_motion_vectors and prev_features is not None and i < len(prev_features):
            prev_x, prev_y = int(prev_features[i][0]), int(prev_features[i][1])
            # Solo dibujar si el movimiento es significativo (>0.5px)
            dx = x - prev_x
            dy = y - prev_y
            movement = np.sqrt(dx**2 + dy**2)
            total_movement += movement
            
            if movement > 0.5:
                # Flecha que muestra el movimiento del feature (DE LA ROCA FÍSICA)
                # Color según magnitud: verde (poco) -> amarillo (medio) -> rojo (mucho)
                if movement > 3.0:
                    arrow_color = (0, 0, 255)  # Rojo - movimiento grande
                elif movement > 1.5:
                    arrow_color = (0, 165, 255)  # Naranja - movimiento medio
                else:
                    arrow_color = (0, 255, 0)  # Verde - movimiento pequeño
                
                # Dibujar flecha DESDE posición anterior HACIA posición actual
                # Esto muestra hacia dónde se movió la roca en el frame
                cv2.arrowedLine(frame_copy, (prev_x, prev_y), (x, y), 
                               arrow_color, 2, tipLength=0.4)
        
        # Draw feature point (small circle) - marca la roca física
        cv2.circle(frame_copy, (x, y), 3, color, -1)
        # Draw cross for better visibility
        cv2.drawMarker(frame_copy, (x, y), color, cv2.MARKER_CROSS, 8, 1)
        
        # Numerar los primeros 5 features para tracking visual
        if i < 5:
            cv2.putText(frame_copy, str(i+1), (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    return frame_copy


def draw_roi(frame: np.ndarray, roi: Tuple[int, int, int, int], 
             label: str = "ROI", color: Tuple[int, int, int] = (0, 255, 0),
             thickness: int = 2) -> np.ndarray:
    """
    Draw ROI rectangle on frame.
    
    Args:
        frame: Input frame
        roi: ROI coordinates (x, y, w, h)
        label: Label text
        color: Rectangle color
        thickness: Line thickness
        
    Returns:
        Frame with ROI drawn
    """
    frame_copy = frame.copy()
    x, y, w, h = roi
    
    # Draw rectangle
    cv2.rectangle(frame_copy, (x, y), (x+w, y+h), color, thickness)
    
    # Draw label with background - asegurar que esté dentro del frame
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    # Si el ROI está muy arriba, poner el label dentro del ROI
    if y < label_size[1] + 15:
        # Label dentro del ROI (arriba)
        label_y = y + label_size[1] + 5
        cv2.rectangle(frame_copy, (x, y), (x+label_size[0]+10, y+label_size[1]+10), color, -1)
        cv2.putText(frame_copy, label, (x+5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    else:
        # Label fuera del ROI (arriba)
        cv2.rectangle(frame_copy, (x, y-label_size[1]-10), (x+label_size[0]+10, y), color, -1)
        cv2.putText(frame_copy, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    
    return frame_copy


def draw_performance_metrics(frame: np.ndarray, fps: float, 
                            position: Tuple[int, int] = None) -> np.ndarray:
    """
    Draw performance metrics (FPS) on frame.
    
    Args:
        frame: Input frame
        fps: Current FPS
        position: Position for text (default: top-right)
        
    Returns:
        Frame with FPS drawn
    """
    frame_copy = frame.copy()
    
    # Don't show FPS during initial warm-up (less than 0.5 FPS is clearly wrong)
    if fps < 0.5:
        return frame_copy
    
    if position is None:
        h, w = frame.shape[:2]
        position = (w - 150, 30)
    
    x, y = position
    
    # Color based on FPS
    if fps >= 25:
        color = (0, 255, 0)  # Green
    elif fps >= 15:
        color = (0, 255, 255)  # Yellow
    else:
        color = (0, 0, 255)  # Red
    
    # Draw semi-transparent background
    overlay = frame_copy.copy()
    cv2.rectangle(overlay, (x-10, y-25), (x+140, y+10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame_copy, 0.4, 0, frame_copy)
    
    # Draw FPS
    cv2.putText(frame_copy, f"FPS: {fps:.1f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
    return frame_copy


def create_comparison_view(original: np.ndarray, processed: np.ndarray, 
                          title1: str = "Original", title2: str = "Processed") -> np.ndarray:
    """
    Create side-by-side comparison view.
    
    Args:
        original: Original frame
        processed: Processed frame
        title1: Title for original
        title2: Title for processed
        
    Returns:
        Combined comparison frame
    """
    # Ensure same size
    h1, w1 = original.shape[:2]
    h2, w2 = processed.shape[:2]
    
    if (h1, w1) != (h2, w2):
        processed = cv2.resize(processed, (w1, h1))
    
    # Add titles
    cv2.putText(original, title1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(processed, title2, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Concatenate horizontally
    combined = np.hstack([original, processed])
    
    return combined


def draw_vibration_graph(frame: np.ndarray, dx_history: list, dy_history: list, 
                        max_points: int = 150, 
                        position: Tuple[int, int] = None,
                        size: Tuple[int, int] = (350, 120),
                        vibration_threshold: float = 0.3) -> np.ndarray:
    """
    Draw real-time vibration graph overlay on frame.
    
    Args:
        frame: Input frame
        dx_history: List of recent X-axis drift values
        dy_history: List of recent Y-axis drift values
        max_points: Maximum points to show in graph
        position: Bottom-left corner position (default: bottom-left of frame)
        size: Graph size (width, height)
        vibration_threshold: Threshold for vibration detection (default: 0.3px)
        
    Returns:
        Frame with vibration graph overlay
    """
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    
    # Default position: bottom-left corner
    if position is None:
        position = (10, h - size[1] - 10)
    
    x_start, y_start = position
    graph_w, graph_h = size
    
    # Get recent history (last max_points)
    recent_dx = list(dx_history[-max_points:]) if len(dx_history) > 0 else [0]
    recent_dy = list(dy_history[-max_points:]) if len(dy_history) > 0 else [0]
    
    # Draw semi-transparent background
    overlay = frame_copy.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_start + graph_w, y_start + graph_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame_copy, 0.3, 0, frame_copy)
    
    # Draw border
    cv2.rectangle(frame_copy, (x_start, y_start), (x_start + graph_w, y_start + graph_h), (100, 100, 100), 2)
    
    # Title
    cv2.putText(frame_copy, "Vibration History", (x_start + 5, y_start + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Calculate scale
    max_drift = max(max(abs(d) for d in recent_dx), max(abs(d) for d in recent_dy), 1.0)
    scale = (graph_h - 40) / (2 * max_drift)  # Scale to fit in graph
    
    # Center line (zero drift)
    center_y = y_start + graph_h // 2
    cv2.line(frame_copy, (x_start, center_y), (x_start + graph_w, center_y), (100, 100, 100), 1, cv2.LINE_AA)
    
    # Draw vibration threshold lines (yellow dashed)
    threshold_offset = int(vibration_threshold * scale)
    threshold_y_upper = center_y - threshold_offset
    threshold_y_lower = center_y + threshold_offset
    
    # Draw dashed threshold lines
    if y_start + 25 <= threshold_y_upper <= y_start + graph_h - 5:
        for i in range(x_start, x_start + graph_w, 10):
            cv2.line(frame_copy, (i, threshold_y_upper), (min(i + 5, x_start + graph_w), threshold_y_upper), (0, 255, 255), 1, cv2.LINE_AA)
    if y_start + 25 <= threshold_y_lower <= y_start + graph_h - 5:
        for i in range(x_start, x_start + graph_w, 10):
            cv2.line(frame_copy, (i, threshold_y_lower), (min(i + 5, x_start + graph_w), threshold_y_lower), (0, 255, 255), 1, cv2.LINE_AA)
    
    # Draw grid lines
    for i in range(1, 3):
        y_pos = y_start + (graph_h * i // 3)
        cv2.line(frame_copy, (x_start, y_pos), (x_start + graph_w, y_pos), (50, 50, 50), 1, cv2.LINE_AA)
    
    # Plot drift lines
    if len(recent_dx) > 1:
        points_per_pixel = max(1, len(recent_dx) / (graph_w - 20))
        
        for i in range(1, len(recent_dx)):
            # X drift (red)
            x1 = int(x_start + 10 + (i - 1) * (graph_w - 20) / max_points)
            y1 = int(center_y - recent_dx[i - 1] * scale)
            x2 = int(x_start + 10 + i * (graph_w - 20) / max_points)
            y2 = int(center_y - recent_dx[i] * scale)
            
            # Clamp to graph bounds
            y1 = max(y_start + 25, min(y_start + graph_h - 5, y1))
            y2 = max(y_start + 25, min(y_start + graph_h - 5, y2))
            
            cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)  # Red for X
            
            # Y drift (green)
            y1_dy = int(center_y - recent_dy[i - 1] * scale)
            y2_dy = int(center_y - recent_dy[i] * scale)
            
            y1_dy = max(y_start + 25, min(y_start + graph_h - 5, y1_dy))
            y2_dy = max(y_start + 25, min(y_start + graph_h - 5, y2_dy))
            
            cv2.line(frame_copy, (x1, y1_dy), (x2, y2_dy), (0, 255, 0), 2, cv2.LINE_AA)  # Green for Y
    
    # Legend
    cv2.line(frame_copy, (x_start + graph_w - 80, y_start + 15), (x_start + graph_w - 65, y_start + 15), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_copy, "X", (x_start + graph_w - 60, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.line(frame_copy, (x_start + graph_w - 40, y_start + 15), (x_start + graph_w - 25, y_start + 15), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_copy, "Y", (x_start + graph_w - 20, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Threshold indicator (bottom-right of legend area)
    cv2.putText(frame_copy, f"Threshold: {vibration_threshold:.1f}px", 
                (x_start + graph_w - 120, y_start + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Current values
    current_dx = recent_dx[-1] if recent_dx else 0
    current_dy = recent_dy[-1] if recent_dy else 0
    cv2.putText(frame_copy, f"X:{current_dx:+.2f}px Y:{current_dy:+.2f}px", 
                (x_start + 5, y_start + graph_h - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame_copy


def draw_qc_score_graph(frame: np.ndarray, qc_history: list,
                        max_points: int = 150,
                        position: Tuple[int, int] = None,
                        size: Tuple[int, int] = (350, 120)) -> np.ndarray:
    """
    Draw real-time QC Score graph overlay on frame.
    
    Args:
        frame: Input frame
        qc_history: List of recent QC score values
        max_points: Maximum points to show in graph
        position: Top-left corner position (default: top-left of frame)
        size: Graph size (width, height)
        
    Returns:
        Frame with QC score graph overlay
    """
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    
    # Default position: top-left corner
    if position is None:
        position = (10, 10)
    
    x_start, y_start = position
    graph_w, graph_h = size
    
    # Get recent history (last max_points)
    recent_qc = list(qc_history[-max_points:]) if len(qc_history) > 0 else [0]
    
    # Draw semi-transparent background
    overlay = frame_copy.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_start + graph_w, y_start + graph_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame_copy, 0.3, 0, frame_copy)
    
    # Draw border
    cv2.rectangle(frame_copy, (x_start, y_start), (x_start + graph_w, y_start + graph_h), (100, 100, 100), 2)
    
    # Title
    cv2.putText(frame_copy, "QC Score History", (x_start + 5, y_start + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Threshold lines
    scale = (graph_h - 40) / 100  # Scale 0-100 to graph height
    
    # Excellent threshold (80)
    y_excellent = int(y_start + graph_h - 10 - (80 * scale))
    cv2.line(frame_copy, (x_start, y_excellent), (x_start + graph_w, y_excellent), (0, 255, 0), 1, cv2.LINE_AA)
    
    # Good threshold (60)
    y_good = int(y_start + graph_h - 10 - (60 * scale))
    cv2.line(frame_copy, (x_start, y_good), (x_start + graph_w, y_good), (0, 165, 255), 1, cv2.LINE_AA)
    
    # Warning threshold (40)
    y_warning = int(y_start + graph_h - 10 - (40 * scale))
    cv2.line(frame_copy, (x_start, y_warning), (x_start + graph_w, y_warning), (0, 0, 255), 1, cv2.LINE_AA)
    
    # Plot QC line
    if len(recent_qc) > 1:
        for i in range(1, len(recent_qc)):
            x1 = int(x_start + 10 + (i - 1) * (graph_w - 20) / max_points)
            y1 = int(y_start + graph_h - 10 - (recent_qc[i - 1] * scale))
            x2 = int(x_start + 10 + i * (graph_w - 20) / max_points)
            y2 = int(y_start + graph_h - 10 - (recent_qc[i] * scale))
            
            # Clamp to graph bounds
            y1 = max(y_start + 25, min(y_start + graph_h - 10, y1))
            y2 = max(y_start + 25, min(y_start + graph_h - 10, y2))
            
            # Color based on score
            if recent_qc[i] >= 80:
                color = (0, 255, 0)  # Green
            elif recent_qc[i] >= 60:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            cv2.line(frame_copy, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    
    # Current value
    current_qc = recent_qc[-1] if recent_qc else 0
    cv2.putText(frame_copy, f"Current: {current_qc:.1f}", 
                (x_start + 5, y_start + graph_h - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame_copy


def save_metrics_plot(metrics_history: Dict, output_path: str):
    """
    Save metrics plot to file - simplified with 2 clear temporal graphs.
    
    Args:
        metrics_history: Dictionary with lists of metrics over time
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Sentinel Eye - Performance Metrics', fontsize=16, fontweight='bold')
    
    # 1. Camera Vibration (drift temporal)
    if 'dx' in metrics_history and 'dy' in metrics_history:
        frames = range(len(metrics_history['dx']))
        
        axes[0].plot(frames, metrics_history['dx'], label='X Drift', color='red', linewidth=1.5, alpha=0.8)
        axes[0].plot(frames, metrics_history['dy'], label='Y Drift', color='green', linewidth=1.5, alpha=0.8)
        axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        axes[0].set_title('Camera Vibration - Drift Over Time', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Frame Number', fontsize=11)
        axes[0].set_ylabel('Drift (pixels)', fontsize=11)
        axes[0].legend(loc='upper right', fontsize=10)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
    
    # 2. QC Score temporal
    if 'qc_scores' in metrics_history:
        frames = range(len(metrics_history['qc_scores']))
        
        axes[1].plot(frames, metrics_history['qc_scores'], color='blue', linewidth=1.5, alpha=0.8)
        axes[1].axhline(y=80, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Excellent (80+)')
        axes[1].axhline(y=60, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Good (60+)')
        axes[1].axhline(y=40, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Warning (40+)')
        
        axes[1].set_title('Image Quality Score Over Time', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Frame Number', fontsize=11)
        axes[1].set_ylabel('QC Score (0-100)', fontsize=11)
        axes[1].set_ylim([0, 100])
        axes[1].legend(loc='lower right', fontsize=9)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
