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
    draw_qc_metrics, draw_stability_info, save_metrics_plot,
    draw_vibration_graph, draw_qc_score_graph
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
            self.yolo_confidence = config.get('detection.confidence_threshold', 0.25)
            self.motion_detector = OptimizedDetectionPipeline(
                use_yolo=config.get('detection.use_yolo', True),
                yolo_model=config.get('detection.yolo_model', 's')
            )
            logger.info("Motion Detection Pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize motion detector: {e}")
            self.motion_detector = None

        
        # Configure OpenCV optimizations
        self.optimizer.optimize_cv_operations()
        
        # Initialize FPS tracking
        self._last_perf_metrics = {'fps': 0.0}
        self._last_effective_fps = 0.0
        
        # Metrics history
        self.metrics_history = {
            'frame_numbers': [],  # Real frame numbers from video
            'timestamps': [],     # Time in seconds
            'qc_scores': [],
            'dx': [],
            'dy': [],
            'fps': [],            # Processing FPS
            'effective_fps': [],  # Effective FPS (fps * frame_skip)
            'processing_time': []
        }
        
        # Video FPS (will be set when processing starts)
        self.video_fps = 30.0
        
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
        
        # Reset ROI for new video (critical for multi-video processing)
        self.current_roi = None
        self.original_roi = None
        
        # Reset metrics history for new video
        self.metrics_history = {
            'frame_numbers': [],
            'timestamps': [],     # Time in seconds
            'qc_scores': [],
            'dx': [],
            'dy': [],
            'fps': [],
            'effective_fps': [],
            'processing_time': []
        }
        
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
        self.video_fps = fps  # Store for timestamp calculation
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
        enable_resize = self.config.get('optimization.enable_resize', False)
        
        # Store original dimensions for upscaling output
        self.original_width = width
        self.original_height = height
        self.resize_enabled = enable_resize  # Store for ROI scaling
        
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
                
                # Optional downscaling for faster processing
                if enable_resize:
                    frame_resized = self.optimizer.preprocess_frame(frame, resize=True)
                    # Process analysis at low resolution (faster)
                    self._process_frame_analysis(frame_resized, frame_count)
                    # Draw visualizations at original resolution (better quality)
                    processed_frame = self._draw_visualizations(frame.copy(), frame_count)
                else:
                    # Process at full resolution
                    processed_frame = self._process_frame(frame, frame_count)
                
                # Record frame time AFTER processing (for accurate FPS calculation)
                self.optimizer.record_frame_time()
                
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
                save_metrics_plot(self.metrics_history, plot_path, video_fps=self.video_fps)
                logger.info(f"Metrics plot saved: {plot_path}")
            
            # Final statistics
            self._log_final_statistics(frame_count)
    
    def _process_frame_analysis(self, frame: np.ndarray, frame_count: int):
        """
        Analyze frame through all modules (without visualization).
        Stores results in instance variables for later visualization.
        
        Args:
            frame: Input frame (possibly resized)
            frame_count: Frame number
        """
        import time
        self._analysis_start_time = time.time()
        
        # Module 1: QC Score
        t1 = time.time()
        qc_score, qc_metrics = self.qc_checker.compute_qc_score(frame)
        self.metrics_history['frame_numbers'].append(frame_count)
        self.metrics_history['timestamps'].append(frame_count / self.video_fps)
        self.metrics_history['qc_scores'].append(qc_score)
        self._last_qc_score = qc_score
        self._last_qc_metrics = qc_metrics
        self._t_qc = time.time() - t1
        
        # Initialize ROI if not set
        if self.current_roi is None:
            h, w = frame.shape[:2]
            
            if hasattr(self, 'initial_roi_config') and self.initial_roi_config:
                roi_x = self.initial_roi_config['x']
                roi_y = self.initial_roi_config['y']
                roi_w = self.initial_roi_config['width']
                roi_h = self.initial_roi_config['height']
                
                # Scale ROI if frame was resized
                if hasattr(self, 'original_width') and hasattr(self, 'resize_enabled') and self.resize_enabled:
                    scale_x = w / self.original_width
                    scale_y = h / self.original_height
                    roi_x = int(roi_x * scale_x)
                    roi_y = int(roi_y * scale_y)
                    roi_w = int(roi_w * scale_x)
                    roi_h = int(roi_h * scale_y)
                    logger.info(f"Scaled ROI for resized frame: ({roi_x}, {roi_y}, {roi_w}x{roi_h})")
                
                self.roi_label = self.initial_roi_config.get('description', 'ROI')
                logger.info(f"Using saved initial ROI: ({roi_x}, {roi_y}, {roi_w}x{roi_h})")
            else:
                roi_w = int(w * 0.20)
                roi_h = int(h * 0.20)
                roi_x = w - roi_w - 10
                roi_y = h - roi_h - 10
                self.roi_label = "Auto-ROI (bottom-right 20%)"
                logger.info(f"Using auto-calculated ROI: ({roi_x}, {roi_y}, {roi_w}x{roi_h})")
            
            self.current_roi = (roi_x, roi_y, roi_w, roi_h)
            self.original_roi = self.current_roi
            self.stability_analyzer.set_reference_frame(frame, self.current_roi)
        
        # Module 2: Stability analysis
        t2 = time.time()
        stability_result = self.stability_analyzer.analyze_frame(frame)
        dx = stability_result['displacement_x']
        dy = stability_result['displacement_y']
        is_vibrating = stability_result['is_vibrating']
        
        self.metrics_history['dx'].append(dx)
        self.metrics_history['dy'].append(dy)
        
        if self.config.get('stability.enable_self_healing', True):
            self.current_roi = self.stability_analyzer.adjust_roi(self.original_roi)
        
        self._last_stability_result = stability_result
        self._last_dx = dx
        self._last_dy = dy
        self._last_is_vibrating = is_vibrating
        self._t_stability = time.time() - t2
        
        # Module 3: Motion/Object Detection
        t3 = time.time()
        detections = self.motion_detector.process_frame(frame, conf_threshold=self.yolo_confidence)
        self._last_detections = detections
        self._t_yolo = time.time() - t3
        
        # Module 4: Performance metrics
        perf_metrics = self.optimizer.get_performance_metrics()
        processing_fps = perf_metrics['fps']
        # Calculate effective FPS (how fast we're moving through the video)
        frame_skip = self.config.get('video.frame_skip', 1)
        effective_fps = processing_fps * frame_skip
        
        self.metrics_history['fps'].append(processing_fps)
        self.metrics_history['effective_fps'].append(effective_fps)
        self._last_perf_metrics = perf_metrics
        self._last_effective_fps = effective_fps
    
    def _draw_visualizations(self, frame: np.ndarray, frame_count: int) -> np.ndarray:
        """
        Draw visualizations on frame at original resolution.
        Uses analysis results from _process_frame_analysis.
        
        Args:
            frame: Input frame at original resolution
            frame_count: Frame number
            
        Returns:
            Frame with visualizations
        """
        import time
        t4 = time.time()
        
        processed = frame.copy()
        h, w = processed.shape[:2]
        
        # Scale ROI coordinates back to original resolution if needed
        if hasattr(self, 'resize_enabled') and self.resize_enabled:
            scale_x = w / self.optimizer.target_resolution[0]
            scale_y = h / self.optimizer.target_resolution[1]
            
            # Scale current ROI
            current_roi_scaled = (
                int(self.current_roi[0] * scale_x),
                int(self.current_roi[1] * scale_y),
                int(self.current_roi[2] * scale_x),
                int(self.current_roi[3] * scale_y)
            )
            original_roi_scaled = (
                int(self.original_roi[0] * scale_x),
                int(self.original_roi[1] * scale_y),
                int(self.original_roi[2] * scale_x),
                int(self.original_roi[3] * scale_y)
            )
        else:
            current_roi_scaled = self.current_roi
            original_roi_scaled = self.original_roi
        
        # Right side: QC metrics with Processing FPS (top-right)
        processing_fps = self._last_perf_metrics['fps']
        processed = draw_qc_metrics(processed, self._last_qc_metrics, fps=processing_fps, 
                                   effective_fps=self._last_effective_fps, position=(w - 260, 30))
        
        # Right side: Stability info (below QC metrics)
        processed = draw_stability_info(processed, self._last_dx, self._last_dy, self._last_is_vibrating, 
                                       position=(w - 220, 220))
        
        # Show ROI tracking status (top-center for visibility)
        camera_offset_x = self._last_stability_result['camera_offset_x']
        camera_offset_y = self._last_stability_result['camera_offset_y']
        total_offset = np.sqrt(camera_offset_x**2 + camera_offset_y**2)
        tracking_text = f"Camera Offset: ({camera_offset_x:.1f}, {camera_offset_y:.1f}) = {total_offset:.1f}px"
        cv2.putText(processed, tracking_text, (w//2 - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Draw tracked features (scaled if needed)
        tracked_features = self.stability_analyzer.tracked_features
        if tracked_features is not None and len(tracked_features) > 0:
            if hasattr(self, 'resize_enabled') and self.resize_enabled:
                for pt in tracked_features:
                    # Handle both (2,) and (1,2) shapes
                    if len(pt.shape) > 1:
                        x, y = pt[0]
                    else:
                        x, y = pt
                    scaled_pt = (int(x * scale_x), int(y * scale_y))
                    cv2.circle(processed, scaled_pt, 3, (0, 255, 255), -1)
            else:
                for pt in tracked_features:
                    cv2.circle(processed, tuple(pt.astype(int)), 3, (0, 255, 255), -1)
        
        # Draw ROI ORIGINAL (gris) - using scaled coordinates
        cv2.rectangle(processed,
                     (original_roi_scaled[0], original_roi_scaled[1]),
                     (original_roi_scaled[0] + original_roi_scaled[2], original_roi_scaled[1] + original_roi_scaled[3]),
                     (128, 128, 128), 2)
        roi_label = getattr(self, 'roi_label', 'ROI Original')
        label_y = max(15, original_roi_scaled[1] - 5)
        cv2.putText(processed, roi_label, 
                   (original_roi_scaled[0], label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1, cv2.LINE_AA)
        
        # Draw ROI transformado (using scaled coordinates)
        roi_corners = self.stability_analyzer.get_transformed_roi_corners()
        if roi_corners is not None:
            if hasattr(self, 'resize_enabled') and self.resize_enabled:
                # Scale corners - cv2.transform returns shape (n, 1, 2)
                roi_corners_scaled = []
                for pt in roi_corners:
                    # pt shape is (1, 2), extract the point
                    x, y = pt[0] if len(pt.shape) > 1 else pt
                    roi_corners_scaled.append([int(x * scale_x), int(y * scale_y)])
                roi_corners_scaled = np.array(roi_corners_scaled, dtype=np.int32)
                cv2.polylines(processed, [roi_corners_scaled], isClosed=True, color=(255, 255, 0), thickness=3)
            else:
                roi_corners_int = np.int32(roi_corners)
                cv2.polylines(processed, [roi_corners_int], isClosed=True, color=(255, 255, 0), thickness=3)
            
            # Bounding box axis-aligned
            cv2.rectangle(processed, 
                         (current_roi_scaled[0], current_roi_scaled[1]),
                         (current_roi_scaled[0] + current_roi_scaled[2], current_roi_scaled[1] + current_roi_scaled[3]),
                         (0, 0, 255), 2)
            
            tracked_label_y = max(15, current_roi_scaled[1] - 5)
            cv2.putText(processed, "ROI Tracked (Self-healing)", 
                       (current_roi_scaled[0], tracked_label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
            
            # Flecha desde centro original a centro actual
            orig_center = (original_roi_scaled[0] + original_roi_scaled[2]//2, 
                          original_roi_scaled[1] + original_roi_scaled[3]//2)
            curr_center = (current_roi_scaled[0] + current_roi_scaled[2]//2,
                          current_roi_scaled[1] + current_roi_scaled[3]//2)
            if orig_center != curr_center:
                cv2.arrowedLine(processed, orig_center, curr_center, (255, 0, 255), 2, tipLength=0.3)
        else:
            cv2.rectangle(processed,
                         (current_roi_scaled[0], current_roi_scaled[1]),
                         (current_roi_scaled[0] + current_roi_scaled[2], current_roi_scaled[1] + current_roi_scaled[3]),
                         (255, 255, 0), 2)
        
        # Draw YOLO detections (scaled if needed)
        yolo_detections = self._last_detections.get('yolo', [])
        if yolo_detections and hasattr(self, 'resize_enabled') and self.resize_enabled:
            # Scale detection boxes
            scaled_detections = []
            for det in yolo_detections:
                scaled_det = det.copy()
                box = det['box']  # (x, y, w, h)
                scaled_det['box'] = (
                    int(box[0] * scale_x),
                    int(box[1] * scale_y),
                    int(box[2] * scale_x),
                    int(box[3] * scale_y)
                )
                scaled_detections.append(scaled_det)
            processed = draw_detections(processed, scaled_detections)
        elif yolo_detections:
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
        qc_position = (10, 10)
        processed = draw_qc_score_graph(
            processed,
            self.metrics_history['qc_scores'],
            max_points=150,
            position=qc_position
        )
        
        self._t_viz = time.time() - t4
        
        # Log timings
        if frame_count % 100 == 0:
            t_total = time.time() - self._analysis_start_time
            logger.info(f"Timings [ms]: QC={self._t_qc*1000:.1f} | Stability={self._t_stability*1000:.1f} | YOLO={self._t_yolo*1000:.1f} | Viz={self._t_viz*1000:.1f} | Total={t_total*1000:.1f}")
        
        # Log warnings
        if self._last_qc_score < self.config.get('qc_score.thresholds.warning', 40):
            logger.warning(f"Frame {frame_count}: Low QC score ({self._last_qc_score:.1f})")
        
        if self._last_is_vibrating and frame_count % 20 == 0:
            if abs(self._last_dx) > self.stability_config['vibration_threshold'] * 2 or abs(self._last_dy) > self.stability_config['vibration_threshold'] * 2:
                logger.warning(f"Frame {frame_count}: Significant vibration (dx={self._last_dx:.2f}, dy={self._last_dy:.2f})")
        
        return processed
    
    def _process_frame(self, frame: np.ndarray, frame_count: int) -> np.ndarray:
        """
        Process a single frame through all modules (legacy method for full resolution).
        
        Args:
            frame: Input frame
            frame_count: Frame number
            
        Returns:
            Processed frame with visualizations
        """
        self._process_frame_analysis(frame, frame_count)
        return self._draw_visualizations(frame, frame_count)
    
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
