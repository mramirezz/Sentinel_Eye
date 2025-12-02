"""
Main entry point for Sentinel Eye system.
Integrates all modules: QC Score, Stability Analysis, and Performance Optimization.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import Optional
import sys
import json
import os

# Add src to path
sys.path.append(str(Path(__file__).parent))

from modules.qc_score import ImageQualityChecker
from modules.stability_tracking import StabilityAnalyzer
from modules.optimization import PerformanceOptimizer
from modules.motion_detection import OptimizedDetectionPipeline, draw_detections
from utils.logger import setup_logger
from utils.visualization import (
    draw_qc_metrics, draw_stability_info, draw_roi,
    draw_performance_metrics, create_comparison_view, save_metrics_plot,
    draw_vibration_graph, draw_qc_score_graph, draw_tracked_features
)
from utils.config import Config

logger = setup_logger()


class SentinelEye:
    """
    Main Sentinel Eye system integrating all modules.
    """
    
    def __init__(self, config: Config):
        """
        Initialize Sentinel Eye system.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize modules
        logger.info("Initializing Sentinel Eye modules...")
        
        self.qc_checker = ImageQualityChecker()
        
        # Stability analyzer will be initialized in process_video
        self.stability_config = {
            'history_size': config.get('stability.history_size', 30),
            'vibration_threshold': config.get('stability.vibration_threshold', 5.0)
        }
        self.initial_roi_file = config.get('stability.initial_roi_file', 'initial_rois.json')
        
        target_res = tuple(config.get('video.target_resolution', [640, 480]))
        self.optimizer = PerformanceOptimizer(
            target_resolution=target_res,
            use_gpu=config.get('optimization.use_gpu', True),
            batch_size=config.get('optimization.batch_size', 1)
        )
        
        # Module 3: Motion Detection
        try:
            self.motion_detector = OptimizedDetectionPipeline(
                use_yolo=config.get('detection.use_yolo', True),
                use_background_sub=config.get('detection.use_background_sub', True),
                frame_skip=config.get('detection.frame_skip', 2)
            )
            logger.info("Motion Detection Pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize motion detector: {e}")
            self.motion_detector = None

        
        # Configure OpenCV optimizations
        self.optimizer.optimize_cv_operations()
        
        # Metrics history
        self.metrics_history = {
            'qc_scores': [],
            'dx': [],
            'dy': [],
            'fps': [],
            'processing_time': []
        }
        
        # ROI - será calculado dinámicamente por frame
        self.roi_percentage = 0.6  # 60% del tamaño del frame
        self.current_roi = None
        self.original_roi = None
        
        logger.info("Sentinel Eye initialized successfully")
    
    def _load_initial_roi(self, video_name: str) -> Optional[dict]:
        """
        Load initial ROI from JSON file for the given video.
        
        Args:
            video_name: Name of the video file (e.g., 'earthquake.mp4')
            
        Returns:
            Initial ROI dict with x, y, width, height or None if not found
        """
        try:
            with open(self.initial_roi_file, 'r') as f:
                rois = json.load(f)
            
            if video_name in rois:
                roi = rois[video_name]
                logger.info(f"Loaded initial ROI for {video_name}: ({roi['x']}, {roi['y']}, {roi['width']}x{roi['height']})")
                return roi
            else:
                logger.warning(f"No initial ROI found for {video_name}. Will use auto-calculated ROI.")
                return None
        except FileNotFoundError:
            logger.warning(f"Initial ROIs file not found: {self.initial_roi_file}")
            return None
        except Exception as e:
            logger.error(f"Error loading initial ROI: {e}")
            return None
    
    def process_video(self, video_path: str, output_path: Optional[str] = None, display: bool = True):
        """
        Process a video file with all modules.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            display: Whether to display video during processing
        """
        logger.info(f"Processing video: {video_path}")
        
        # Load initial ROI for this specific video
        video_name = os.path.basename(video_path)
        self.initial_roi_config = self._load_initial_roi(video_name)
        
        # Initialize stability analyzer
        self.stability_analyzer = StabilityAnalyzer(
            history_size=self.stability_config['history_size'],
            vibration_threshold=self.stability_config['vibration_threshold']
        )
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Saving output to: {output_path}")
        
        frame_count = 0
        frame_skip = self.config.get('video.frame_skip', 1)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Optional frame skipping for performance
                if not self.optimizer.apply_frame_skip(frame_count, frame_skip):
                    # Write last processed frame to maintain output FPS
                    if writer and hasattr(self, '_last_processed'):
                        writer.write(self._last_processed)
                    continue
                
                # Record frame time for FPS calculation
                self.optimizer.record_frame_time()
                
                # Process at full resolution
                processed_frame = self._process_frame(frame, frame_count)
                
                # Cache for skipped frames
                self._last_processed = processed_frame
                
                # Save to video
                if writer:
                    writer.write(processed_frame)
                
                # Display
                if display:
                    cv2.imshow('Sentinel Eye', processed_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User stopped processing")
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        screenshot_path = f"outputs/screenshot_{frame_count}.jpg"
                        cv2.imwrite(screenshot_path, processed_frame)
                        logger.info(f"Screenshot saved: {screenshot_path}")
                
                # Log progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    current_fps = self.optimizer.get_fps()
                    logger.info(f"Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f}")
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        
        except Exception as e:
            logger.error(f"Error during processing: {e}", exc_info=True)
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Save metrics plot
            if self.config.get('logging.save_plots', True):
                plot_path = output_path.replace('.mp4', '_metrics.png') if output_path else 'outputs/metrics.png'
                save_metrics_plot(self.metrics_history, plot_path)
                logger.info(f"Metrics plot saved: {plot_path}")
            
            # Final statistics
            self._log_final_statistics(frame_count)
    
    def _process_frame(self, frame: np.ndarray, frame_count: int) -> np.ndarray:
        """
        Process a single frame through all modules.
        
        Args:
            frame: Input frame
            frame_count: Frame number
            
        Returns:
            Processed frame with visualizations
        """
        import time
        t_start = time.time()
        
        processed = frame.copy()
        
        # Module 1: QC Score
        t1 = time.time()
        qc_score, qc_metrics = self.qc_checker.compute_qc_score(frame)
        self.metrics_history['qc_scores'].append(qc_score)
        t_qc = time.time() - t1
        
        # Initialize ROI if not set
        if self.current_roi is None:
            # Try to load from initial_roi_config (passed from process_video)
            if hasattr(self, 'initial_roi_config') and self.initial_roi_config:
                roi_x = self.initial_roi_config['x']
                roi_y = self.initial_roi_config['y']
                roi_w = self.initial_roi_config['width']
                roi_h = self.initial_roi_config['height']
                self.roi_label = self.initial_roi_config.get('description', 'ROI')
                logger.info(f"Using saved initial ROI: ({roi_x}, {roi_y}, {roi_w}x{roi_h})")
            else:
                # Default: bottom-right corner
                h, w = frame.shape[:2]
                roi_w = int(w * 0.20)  # 20% width
                roi_h = int(h * 0.20)  # 20% height
                roi_x = w - roi_w - 10  # Bottom-right, 10px margin
                roi_y = h - roi_h - 10
                self.roi_label = "Auto-ROI (bottom-right 20%)"
                logger.info(f"Using auto-calculated ROI: ({roi_x}, {roi_y}, {roi_w}x{roi_h})")
            
            self.current_roi = (roi_x, roi_y, roi_w, roi_h)
            self.original_roi = self.current_roi
            
            # Set reference frame with initial ROI
            self.stability_analyzer.set_reference_frame(frame, self.current_roi)
        
        # Module 2: Stability analysis
        t2 = time.time()
        stability_result = self.stability_analyzer.analyze_frame(frame)
        dx = stability_result['displacement_x']
        dy = stability_result['displacement_y']
        is_vibrating = stability_result['is_vibrating']
        
        self.metrics_history['dx'].append(dx)
        self.metrics_history['dy'].append(dy)
        
        # Self-healing: Adjust ROI to track same physical location despite camera movement
        if self.config.get('stability.enable_self_healing', True):
            self.current_roi = self.stability_analyzer.adjust_roi(self.original_roi)
        
        t_stability = time.time() - t2
        
        # Module 3: Motion/Object Detection
        t3 = time.time()
        detections = self.motion_detector.process_frame(frame)
        t_yolo = time.time() - t3
        
        # Module 4: Performance metrics
        perf_metrics = self.optimizer.get_performance_metrics()
        self.metrics_history['fps'].append(perf_metrics['fps'])
        
        # Visualization - Reorganized layout
        t4 = time.time()
        h, w = processed.shape[:2]
        
        # Right side: QC metrics (top-right)
        processed = draw_qc_metrics(processed, qc_metrics, position=(w - 350, 30))
        
        # Right side: Stability info (below QC metrics)
        processed = draw_stability_info(processed, dx, dy, is_vibrating, position=(w - 350, 180))
        
        # Top-right corner: FPS
        processed = draw_performance_metrics(processed, perf_metrics['fps'], position=(w - 150, 30))
        
        # Show ROI tracking status (top-center for visibility)
        camera_offset_x = stability_result['camera_offset_x']
        camera_offset_y = stability_result['camera_offset_y']
        total_offset = np.sqrt(camera_offset_x**2 + camera_offset_y**2)
        tracking_text = f"Camera Offset: ({camera_offset_x:.1f}, {camera_offset_y:.1f}) = {total_offset:.1f}px"
        cv2.putText(processed, tracking_text, (w//2 - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Draw tracked features (puntos físicos que el ROI persigue)
        tracked_features = self.stability_analyzer.tracked_features
        if tracked_features is not None and len(tracked_features) > 0:
            for pt in tracked_features:
                cv2.circle(processed, tuple(pt.astype(int)), 3, (0, 255, 255), -1)
        
        # Draw ROI ORIGINAL (gris) - referencia inicial
        cv2.rectangle(processed,
                     (self.original_roi[0], self.original_roi[1]),
                     (self.original_roi[0] + self.original_roi[2], self.original_roi[1] + self.original_roi[3]),
                     (128, 128, 128), 2)
        # Usar el label guardado (de initial_rois.json o auto-calculado)
        roi_label = getattr(self, 'roi_label', 'ROI Original')
        # Fuente más chica y posición segura
        label_y = max(15, self.original_roi[1] - 5)
        cv2.putText(processed, roi_label, 
                   (self.original_roi[0], label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1, cv2.LINE_AA)
        
        # Draw ROI transformado (polígono cyan + bounding box rojo)
        roi_corners = self.stability_analyzer.get_transformed_roi_corners()
        if roi_corners is not None:
            # Polígono de esquinas transformadas (cyan brillante)
            roi_corners_int = np.int32(roi_corners)
            cv2.polylines(processed, [roi_corners_int], isClosed=True, color=(255, 255, 0), thickness=3)
            
            # Bounding box axis-aligned (rojo)
            cv2.rectangle(processed, 
                         (self.current_roi[0], self.current_roi[1]),
                         (self.current_roi[0] + self.current_roi[2], self.current_roi[1] + self.current_roi[3]),
                         (0, 0, 255), 2)
            
            # Label del ROI transformado - fuente más chica y posición segura
            tracked_label_y = max(15, self.current_roi[1] - 5)
            cv2.putText(processed, "ROI Tracked (Self-healing)", 
                       (self.current_roi[0], tracked_label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
            
            # Flecha desde centro original a centro actual
            orig_center = (self.original_roi[0] + self.original_roi[2]//2, 
                          self.original_roi[1] + self.original_roi[3]//2)
            curr_center = (self.current_roi[0] + self.current_roi[2]//2,
                          self.current_roi[1] + self.current_roi[3]//2)
            if orig_center != curr_center:
                cv2.arrowedLine(processed, orig_center, curr_center, (255, 0, 255), 2, tipLength=0.3)
        else:
            # Fallback: dibujar ROI simple
            cv2.rectangle(processed,
                         (self.current_roi[0], self.current_roi[1]),
                         (self.current_roi[0] + self.current_roi[2], self.current_roi[1] + self.current_roi[3]),
                         (255, 255, 0), 2)
        
        # Draw YOLO detections (on top of everything)
        yolo_detections = detections.get('yolo', [])
        if yolo_detections:
            processed = draw_detections(processed, yolo_detections)
        
        # Draw vibration graph overlay (bottom-left)
        processed = draw_vibration_graph(
            processed, 
            self.metrics_history['dx'], 
            self.metrics_history['dy'],
            max_points=150,
            vibration_threshold=self.stability_config['vibration_threshold']
        )
        
        # Draw QC Score graph overlay (top-left position)
        qc_position = (10, 10)  # Top-left corner
        processed = draw_qc_score_graph(
            processed,
            self.metrics_history['qc_scores'],
            max_points=150,
            position=qc_position
        )
        t_viz = time.time() - t4
        
        # Log timings cada 30 frames
        if frame_count % 30 == 0:
            t_total = time.time() - t_start
            logger.info(f"Timings [ms]: QC={t_qc*1000:.1f} | Stability={t_stability*1000:.1f} | YOLO={t_yolo*1000:.1f} | Viz={t_viz*1000:.1f} | Total={t_total*1000:.1f}")
        
        # Log warnings
        if qc_score < self.config.get('qc_score.thresholds.warning', 40):
            logger.warning(f"Frame {frame_count}: Low QC score ({qc_score:.1f})")
        
        if is_vibrating:
            logger.warning(f"Frame {frame_count}: Vibration detected (dx={dx:.2f}, dy={dy:.2f})")
        
        return processed
    
    def _log_final_statistics(self, total_frames: int):
        """Log final processing statistics."""
        logger.info("=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 60)
        
        if self.metrics_history['qc_scores']:
            avg_qc = np.mean(self.metrics_history['qc_scores'])
            min_qc = np.min(self.metrics_history['qc_scores'])
            max_qc = np.max(self.metrics_history['qc_scores'])
            logger.info(f"QC Score - Avg: {avg_qc:.1f}, Min: {min_qc:.1f}, Max: {max_qc:.1f}")
        
        if self.metrics_history['fps']:
            avg_fps = np.mean(self.metrics_history['fps'])
            logger.info(f"Average FPS: {avg_fps:.1f}")
        
        if self.metrics_history['dx'] and self.metrics_history['dy']:
            movements = np.sqrt(np.array(self.metrics_history['dx'])**2 + 
                              np.array(self.metrics_history['dy'])**2)
            avg_movement = np.mean(movements)
            max_movement = np.max(movements)
            logger.info(f"Camera Movement - Avg: {avg_movement:.2f}px, Max: {max_movement:.2f}px")
        
        logger.info(f"Total frames processed: {total_frames}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Sentinel Eye - Industrial Computer Vision System')
    parser.add_argument('--video', '-v', type=str, help='Path to input video file')
    parser.add_argument('--input-dir', '-i', type=str, help='Directory containing videos to process')
    parser.add_argument('--output', '-o', type=str, help='Path to output video file')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config if Path(args.config).exists() else None)
    
    # Initialize system
    sentinel = SentinelEye(config)
    
    # Process videos
    if args.video:
        # Single video
        video_path = args.video
        output_path = args.output or f"outputs/{Path(video_path).stem}_output.mp4"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        sentinel.process_video(video_path, output_path, display=not args.no_display)
    
    elif args.input_dir:
        # Multiple videos
        input_dir = Path(args.input_dir)
        video_files = list(input_dir.glob('*.mp4')) + list(input_dir.glob('*.avi')) + list(input_dir.glob('*.mov'))
        
        logger.info(f"Found {len(video_files)} videos in {input_dir}")
        
        for video_file in video_files:
            output_path = f"outputs/{video_file.stem}_output.mp4"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            sentinel.process_video(str(video_file), output_path, display=not args.no_display)
    
    else:
        # Default: process all videos in data/
        data_dir = Path(config.get('video.input_path', 'data'))
        if data_dir.exists():
            video_files = list(data_dir.glob('*.mp4')) + list(data_dir.glob('*.avi')) + list(data_dir.glob('*.mov'))
            
            if video_files:
                logger.info(f"Found {len(video_files)} videos in {data_dir}")
                
                for video_file in video_files:
                    output_path = f"outputs/{video_file.stem}_output.mp4"
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    
                    sentinel.process_video(str(video_file), output_path, display=not args.no_display)
            else:
                logger.error("No video files found in data/ directory")
                logger.info("Usage: python main.py --video path/to/video.mp4")
        else:
            logger.error(f"Data directory not found: {data_dir}")
            logger.info("Usage: python main.py --video path/to/video.mp4")


if __name__ == "__main__":
    main()
