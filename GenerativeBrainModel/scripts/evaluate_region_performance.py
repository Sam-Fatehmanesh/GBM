"""
Main script for region-specific performance evaluation of GBM models.

This script orchestrates the complete evaluation pipeline:
1. Load test data and predictions from experiment results
2. Detect z-zero frames and extract valid chunks
3. Group frames into brain volumes  
4. Run both next-frame and long-horizon evaluations
5. Calculate region-specific performance metrics
6. Generate comprehensive visualizations
7. Save detailed results and summaries

Usage:
    python evaluate_region_performance.py --experiment_path <path> [options]
"""

import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
import sys
import traceback
import pdb
import h5py


# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.region_analysis import (
    TestDataLoader,
    FrameProcessor, 
    PredictionRunner,
    VolumeGrouper,
    RegionPerformanceEvaluator,
    RegionPerformanceVisualizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

# Enable debug logging for prediction runner and evaluator
logging.getLogger('evaluation.region_analysis.prediction_runner').setLevel(logging.DEBUG)
logging.getLogger('evaluation.region_analysis.region_performance_evaluator').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


def setup_output_directory(base_dir: str = "experiments/region_eval") -> Path:
    """
    Create timestamped output directory for results.
    
    Args:
        base_dir: Base directory for region evaluation outputs
        
    Returns:
        Path to created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    return output_dir


def save_configuration(output_dir: Path, args: argparse.Namespace):
    """Save evaluation configuration to JSON."""
    config = {
        'experiment_path': args.experiment_path,
        'masks_path': args.masks_path,
        'target_shape': args.target_shape,
        'threshold': args.threshold,
        'chunk_size': args.chunk_size,
        'time_window': args.time_window,
        'device': args.device,
        'evaluation_types': {
            'next_frame': args.eval_next_frame,
            'long_horizon': args.eval_long_horizon
        },
        'video_options': {
            'create_videos': args.create_videos,
            'fps': args.video_fps,
            'max_frames': args.video_max_frames,
            'max_sequences': args.video_max_sequences
        },
        'timestamp': datetime.now().isoformat(),
        'output_directory': str(output_dir)
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved configuration to {config_path}")


def evaluate_next_frame_performance(test_loader: TestDataLoader,
                                  evaluator: RegionPerformanceEvaluator,
                                  visualizer: RegionPerformanceVisualizer,
                                  output_dir: Path,
                                  args: argparse.Namespace) -> dict:
    """
    Run next-frame prediction evaluation.
    
    Args:
        test_loader: TestDataLoader instance
        evaluator: RegionPerformanceEvaluator instance  
        visualizer: RegionPerformanceVisualizer instance
        output_dir: Output directory
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Starting next-frame prediction evaluation")
    
    try:
        # Get next-frame data using all test samples in a batch format
        input_frames, ground_truth, predictions = test_loader.get_next_frame_data()
        
        logger.info(f"Next-frame data shapes:")
        logger.info(f"  Input: {input_frames.shape}")
        logger.info(f"  Ground truth: {ground_truth.shape}")
        logger.info(f"  Predictions: {predictions.shape}")
        
        # Evaluate performance across regions
        results = evaluator.evaluate_next_frame_predictions(predictions, ground_truth)
        
        # Get summary statistics
        summary = evaluator.get_region_summary(results)
        
        # Create visualizations - now returns dictionary of metric plots
        plot_paths = {}
        metric_plots = visualizer.plot_next_frame_performance(results)
        plot_paths.update(metric_plots)  # Add precision, recall, f1 plot paths
        
        # Create comparison video for next-frame predictions if enabled
        if args.create_videos:
            logger.info("Creating next-frame comparison video")
            video_path = visualizer.create_comparison_video(
                predictions=predictions,
                ground_truth=ground_truth,
                input_frames=input_frames,
                filename="next_frame_comparison.mp4",
                num_frames=predictions.shape[0],  # Use full length of predictions
                fps=args.video_fps,
                max_sequences=args.video_max_sequences
            )
            if video_path:
                plot_paths['comparison_video'] = video_path
        
        # Save results to next_frame subdirectory
        next_frame_results_dir = output_dir / "results" / "next_frame"
        next_frame_results_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = next_frame_results_dir / "next_frame_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        summary_path = next_frame_results_dir / "next_frame_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save results table to next_frame plots directory
        table_path = visualizer.next_frame_dir / "next_frame_performance_table.csv"
        import pandas as pd
        df = pd.DataFrame.from_dict(results, orient='index')
        df = df.sort_values('f1', ascending=False)
        df.to_csv(table_path, float_format='%.4f')
        logger.info(f"Saved results table to {table_path}")
        
        logger.info("Next-frame evaluation completed successfully")
        
        return {
            'results': results,
            'summary': summary,
            'plot_paths': plot_paths,
            'results_path': str(results_path),
            'summary_path': str(summary_path),
            'table_path': table_path
        }
        
    except Exception as e:
        logger.error(f"Next-frame evaluation failed: {e}")
        logger.error(traceback.format_exc())
        return {}


def evaluate_long_horizon_performance(test_loader: TestDataLoader,
                                    frame_processor: FrameProcessor,
                                    volume_grouper: VolumeGrouper,
                                    prediction_runner: PredictionRunner,
                                    evaluator: RegionPerformanceEvaluator,
                                    visualizer: RegionPerformanceVisualizer,
                                    output_dir: Path,
                                    args: argparse.Namespace) -> dict:
    """
    Run long-horizon prediction evaluation.
    
    Args:
        test_loader: TestDataLoader instance
        frame_processor: FrameProcessor instance
        volume_grouper: VolumeGrouper instance
        prediction_runner: PredictionRunner instance
        evaluator: RegionPerformanceEvaluator instance
        visualizer: RegionPerformanceVisualizer instance
        output_dir: Output directory
        time_window: Time window size for temporal evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Starting long-horizon prediction evaluation")
    
    try:
        # Get long-horizon data
        initial_frames, ground_truth_continuation = test_loader.get_long_horizon_data()
        
        logger.info(f"Long-horizon data shapes:")
        logger.info(f"  Initial frames: {initial_frames.shape}")
        logger.info(f"  Ground truth continuation: {ground_truth_continuation.shape}")
        
        # Compute long-horizon volume boundaries from recorded sequence_z_start
        # Get sequence parameters
        seq_len = test_loader.metadata.get('param_seq_len')
        if seq_len is None:
            seq_len = initial_frames.shape[0] + ground_truth_continuation.shape[0]
        seq_len = int(seq_len)
        
        # Number of z-planes per volume (mask depth) from subject metadata
        exp_info_file = Path(args.experiment_path) / "experiment_info.txt"
        if not exp_info_file.exists():
            raise FileNotFoundError(f"experiment_info.txt not found at {exp_info_file}")
        subject = None
        preaug_dir = None
        # Parse target_subject and preaugmented_dir
        with open(exp_info_file) as ef:
            for line in ef:
                stripped = line.strip()
                if stripped.startswith("target_subject:"):
                    subject_value = stripped.split(":", 1)[1].strip()
                    # Handle pretrain-only mode where target_subject might be None
                    if subject_value.lower() != 'none':
                        subject = subject_value
                elif stripped.startswith("preaugmented_dir:"):
                    preaug_dir = stripped.split(":", 1)[1].strip()
        
        # For pretrain-only experiments, we might not have a target subject
        # In this case, use any available subject for metadata extraction
        if subject is None:
            logger.info("No target subject found (pretrain-only mode), using first available subject for metadata")
            if preaug_dir is None:
                raise RuntimeError("preaugmented_dir not found in experiment_info.txt")
            # Find the first available subject directory
            preaug_path = Path(preaug_dir)
            if not preaug_path.exists():
                raise FileNotFoundError(f"Preaugmented directory not found: {preaug_dir}")
            
            available_subjects = []
            for subject_dir in preaug_path.iterdir():
                if subject_dir.is_dir() and subject_dir.name.startswith('subject_'):
                    metadata_h5 = subject_dir / "metadata.h5"
                    if metadata_h5.exists():
                        available_subjects.append(subject_dir.name)
            
            if not available_subjects:
                raise RuntimeError(f"No valid subject directories found in {preaug_dir}")
            
            subject = available_subjects[0]  # Use first available subject for metadata
            logger.info(f"Using subject '{subject}' for metadata extraction")
        
        if preaug_dir is None:
            raise RuntimeError("preaugmented_dir not found in experiment_info.txt")
            
        metadata_h5 = Path(preaug_dir) / subject / "metadata.h5"
        if not metadata_h5.exists():
            raise FileNotFoundError(f"Subject metadata not found: {metadata_h5}")
        # Read num_z_planes from subject metadata
        with h5py.File(metadata_h5, 'r') as mf:
            Z = int(mf['num_z_planes'][()])
        logger.info(f"Using num_z_planes={Z} from subject metadata")
        
        # Number of volumes in the full sequence
        n_vols = seq_len // Z
        
        # Starting z-plane index for this sequence
        seq_z = getattr(test_loader, 'sequence_z_start', 0)
        
        # First volume boundary occurs when we reach z-plane 0 from the starting z-plane
        # (-seq_z) % Z gives us the frame offset to reach z=0 from seq_z
        first_idx = (-seq_z) % Z
        
        # Build all volume boundaries in full sequence coordinates
        full_boundaries = [(first_idx + i * Z, first_idx + (i + 1) * Z) for i in range(n_vols)]
        
        logger.info(f"Sequence z_start={seq_z}, volumes={n_vols}, z_planes={Z}")
        logger.info(f"First volume boundary at frame {first_idx} (to align with z=0)")
        
        # Map boundaries into continuation frame indices (subtract initial frames length)
        init_len = initial_frames.shape[0]
        cont_len = ground_truth_continuation.shape[0]
        boundaries = []
        
        for i, (start, end) in enumerate(full_boundaries):
            # Convert from full-sequence indices to continuation indices
            cs = start - init_len  # Continuation start
            ce = end - init_len    # Continuation end
            
            # Only include volumes that overlap with continuation frames
            if ce <= 0 or cs >= cont_len:
                continue
            
            # Clip to continuation frame range
            clipped_start = max(cs, 0)
            clipped_end = min(ce, cont_len)
            
            boundaries.append((clipped_start, clipped_end))
            logger.info(f"Volume {i+1}: full_seq[{start}:{end}] -> continuation[{clipped_start}:{clipped_end}] (size: {clipped_end - clipped_start})")
        
        logger.info(f"Computed {len(boundaries)} continuation volume boundaries")
        
        # Batch long-horizon predictions and evaluation over all sequences
        initial_batch, cont_batch = test_loader.get_long_horizon_data()
        N = initial_batch.shape[0]
        logger.info(f"Running autoregressive predictions for {N} sequences in batch")
        
        # Prepare aggregated tp/fp/fn/tn per volume per region across all batches
        volume_aggregated_counts = {}
        for vol_idx in range(len(boundaries)):
            volume_aggregated_counts[vol_idx] = {region: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'total_pixels': 0} 
                                               for region in evaluator.region_names}
        
        for i in range(N):
            init_i = initial_batch[i]  # (initial_size, H, W)
            cont_i = cont_batch[i]     # (continuation_size, H, W)
            
            # Compute sample-specific boundaries
            seq_z_all = getattr(test_loader, 'sequence_z_starts', None)
            seq_z_i = int(seq_z_all[i]) if seq_z_all is not None else getattr(test_loader, 'sequence_z_start', 0)
            first_idx_i = (-seq_z_i) % Z
            full_bds_i = [(first_idx_i + j * Z, first_idx_i + (j + 1) * Z) for j in range(n_vols)]
            
            # Map to continuation indices
            bds_i = []
            init_len = init_i.shape[0]
            cont_len = cont_i.shape[0]
            for (s, e) in full_bds_i:
                cs = s - init_len
                ce = e - init_len
                if ce <= 0 or cs >= cont_len:
                    continue
                cs_clipped = max(cs, 0)
                ce_clipped = min(ce, cont_len)
                bds_i.append((cs_clipped, ce_clipped))
            
            if not bds_i:
                raise RuntimeError(f"No valid volumes for sequence {i}, debug z-start logic.")
            
            # Run prediction for this sample
            logger.info(f"Sequence {i+1}/{N}: generating predictions of shape {cont_i.shape}")
            preds_i = prediction_runner.run_long_horizon_predictions(
                init_i, num_steps=cont_i.shape[0]
            )
            
            # Evaluate by volumes
            sample_res = evaluator.evaluate_long_horizon_predictions_by_volumes(
                preds_i, cont_i, bds_i
            )
            
            # Aggregate tp/fp/fn/tn counts per volume per region
            for region_name, metrics_list in sample_res.items():
                for volume_metrics in metrics_list:
                    vol_idx = volume_metrics['volume_idx']
                    if vol_idx in volume_aggregated_counts:
                        counts = volume_aggregated_counts[vol_idx][region_name]
                        counts['tp'] += volume_metrics['tp']
                        counts['fp'] += volume_metrics['fp']
                        counts['fn'] += volume_metrics['fn']
                        counts['tn'] += volume_metrics['tn']
                        counts['total_pixels'] += volume_metrics['total_pixels']
        
        # Now compute final metrics once per volume per region
        temporal_results = {region: [] for region in evaluator.region_names}
        for vol_idx in sorted(volume_aggregated_counts.keys()):
            for region_name in evaluator.region_names:
                counts = volume_aggregated_counts[vol_idx][region_name]
                tp = counts['tp']
                fp = counts['fp']
                fn = counts['fn']
                tn = counts['tn']
                total_pixels = counts['total_pixels']
                
                # Calculate aggregated metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 0.0
                support = tp + fn
                
                aggregated_metrics = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'accuracy': float(accuracy),
                    'support': float(support),
                    'tp': float(tp),
                    'fp': float(fp),
                    'fn': float(fn),
                    'tn': float(tn),
                    'total_pixels': float(total_pixels),
                    'volume_idx': vol_idx,
                    'aggregated_across_batches': N
                }
                
                temporal_results[region_name].append(aggregated_metrics)
        
        # Get temporal summary
        temporal_summary = evaluator.get_temporal_summary(temporal_results)
        
        # Create visualizations - save to long_horizon directory
        plot_paths = {}
        plot_paths['heatmap_f1'] = visualizer.plot_long_horizon_heatmap(
            temporal_results, metric='f1'
        )
        plot_paths['heatmap_precision'] = visualizer.plot_long_horizon_heatmap(
            temporal_results, metric='precision'
        )
        plot_paths['heatmap_recall'] = visualizer.plot_long_horizon_heatmap(
            temporal_results, metric='recall'
        )
        plot_paths['temporal_trends'] = visualizer.plot_temporal_trends(temporal_summary)
        
        # Create comparison video for long-horizon predictions if enabled
        if args.create_videos:
            logger.info("Creating long-horizon comparison video")
            #pdb.set_trace()
            video_path = visualizer.create_comparison_video(
                predictions=preds_i,
                ground_truth=cont_i,
                input_frames=cont_i,
                filename="long_horizon_comparison.mp4",
                num_frames=preds_i.shape[0],  # Use full length of predictions
                fps=args.video_fps,
                max_sequences=args.video_max_sequences
            )
            if video_path:
                plot_paths['comparison_video'] = video_path
        
        # Save results to long_horizon subdirectory
        long_horizon_results_dir = output_dir / "results" / "long_horizon"
        long_horizon_results_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = long_horizon_results_dir / "long_horizon_results.json"
        with open(results_path, 'w') as f:
            # Convert lists to regular Python lists for JSON serialization
            json_safe_results = {}
            for region, region_results in temporal_results.items():
                json_safe_results[region] = region_results
            json.dump(json_safe_results, f, indent=2)
        
        summary_path = long_horizon_results_dir / "long_horizon_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(temporal_summary, f, indent=2)
        
        # Save processing information to long_horizon directory
        processing_info = {
            # Starting z-plane index for this sequence
            'sequence_z_start': seq_z,
            # Number of volumes per sequence
            'volumes_per_sequence': n_vols,
            # Number of z-planes per volume
            'z_planes_per_volume': Z,
            # Boundaries in full sequence frame indices
            'full_volume_boundaries': full_boundaries,
            # Boundaries mapped to continuation (prediction) frame indices
            'continuation_volume_boundaries': boundaries
        }
        
        processing_path = long_horizon_results_dir / "processing_info.json"
        with open(processing_path, 'w') as f:
            json.dump(processing_info, f, indent=2)
        
        logger.info("Long-horizon evaluation completed successfully")
        
        return {
            'temporal_results': temporal_results,
            'temporal_summary': temporal_summary,
            'plot_paths': plot_paths,
            'results_path': str(results_path),
            'summary_path': str(summary_path),
            'processing_info': processing_info
        }
        
    except Exception as e:
        logger.error(f"Long-horizon evaluation failed: {e}")
        logger.error(traceback.format_exc())
        return {}


def create_comprehensive_dashboard(next_frame_results: dict,
                                 long_horizon_results: dict,
                                 visualizer: RegionPerformanceVisualizer,
                                 output_dir: Path):
    """Create comprehensive dashboard combining all results."""
    logger.info("Creating comprehensive performance dashboard")
    
    try:
        if next_frame_results and long_horizon_results:
            dashboard_path = visualizer.create_summary_dashboard(
                next_frame_results.get('results', {}),
                long_horizon_results.get('temporal_results', {}),
                long_horizon_results.get('temporal_summary', {})

                
            )
            logger.info(f"Saved comprehensive dashboard to {dashboard_path}")
        else:
            logger.warning("Insufficient results for comprehensive dashboard")
            
    except Exception as e:
        logger.error(f"Dashboard creation failed: {e}")


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate GBM region-specific performance")
    
    # Required arguments
    parser.add_argument('--experiment_path', type=str, required=True,
                       help='Path to experiment directory containing test_data_and_predictions.h5')
    
    # Optional arguments
    parser.add_argument('--masks_path', type=str, default='masks',
                       help='Path to brain region masks directory (default: masks)')
    parser.add_argument('--output_dir', type=str, default='experiments/region_eval',
                       help='Base output directory (default: experiments/region_eval)')
    parser.add_argument('--target_shape', type=int, nargs=3, default=[30, 256, 128],
                       help='Target shape for mask downsampling (default: 30 256 128)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary conversion (default: 0.5)')
    parser.add_argument('--chunk_size', type=int, default=330,
                       help='Expected chunk size (default: 330)')
    parser.add_argument('--time_window', type=int, default=10,
                       help='Time window for temporal evaluation (default: 10)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device for computation (default: cuda)')
    
    # Evaluation type flags
    parser.add_argument('--eval_next_frame', action='store_true', default=True,
                       help='Evaluate next-frame predictions (default: True)')
    parser.add_argument('--eval_long_horizon', action='store_true', default=True,
                       help='Evaluate long-horizon predictions (default: True)')
    parser.add_argument('--skip_next_frame', action='store_true', 
                       help='Skip next-frame evaluation')
    parser.add_argument('--skip_long_horizon', action='store_true',
                       help='Skip long-horizon evaluation')
    
    # Video generation options
    parser.add_argument('--create_videos', action='store_true', default=True,
                       help='Create comparison videos (default: True)')
    parser.add_argument('--skip_videos', action='store_true',
                       help='Skip video generation')
    parser.add_argument('--video_fps', type=int, default=2,
                       help='Frames per second for videos (default: 2)')
    parser.add_argument('--video_max_frames', type=int, default=100,
                       help='Maximum frames per video (default: 100)')
    parser.add_argument('--video_max_sequences', type=int, default=3,
                       help='Maximum sequences per video (default: 3)')
    
    args = parser.parse_args()
    
    # Handle evaluation type flags
    if args.skip_next_frame:
        args.eval_next_frame = False
    if args.skip_long_horizon:
        args.eval_long_horizon = False
    
    # Handle video generation flags
    if args.skip_videos:
        args.create_videos = False
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    
    # Setup file logging
    log_file = output_dir / "logs" / "evaluation.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Starting region performance evaluation")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Experiment path: {args.experiment_path}")
    
    # Save configuration
    save_configuration(output_dir, args)
    
    try:
        # Initialize components
        logger.info("Initializing evaluation components")
        
        test_loader = TestDataLoader(args.experiment_path, device=args.device)
        
        frame_processor = FrameProcessor(
            chunk_size=args.chunk_size,
            marker_threshold=0.3,  # More lenient for real brain data
            clear_area_threshold=0.5  # More lenient clear area
        )
        
        volume_grouper = VolumeGrouper(
            expected_volume_size=None,  # Will be inferred
            tolerance=2
        )
        
        prediction_runner = PredictionRunner(
            args.experiment_path,
            device=args.device,
            threshold=args.threshold
        )
        
        evaluator = RegionPerformanceEvaluator(
            masks_path=args.masks_path,
            target_shape=tuple(args.target_shape),
            threshold=args.threshold,
            device=args.device
        )
        
        visualizer = RegionPerformanceVisualizer(
            output_dir=output_dir / "plots"
        )
        
        # Log component status
        logger.info(f"Components initialized:")
        logger.info(f"  Test data: {test_loader.get_data_info()}")
        logger.info(f"  Model loaded: {prediction_runner.is_model_loaded()}")
        logger.info(f"  Available regions: {len(evaluator.get_available_regions())}")
        
        # Run evaluations
        results = {}
        
        if args.eval_next_frame:
            logger.info("=" * 60)
            logger.info("NEXT-FRAME PREDICTION EVALUATION")
            logger.info("=" * 60)
            
            next_frame_results = evaluate_next_frame_performance(
                test_loader, evaluator, visualizer, output_dir, args
            )
            results['next_frame'] = next_frame_results
        
        if args.eval_long_horizon:
            logger.info("=" * 60)
            logger.info("LONG-HORIZON PREDICTION EVALUATION")
            logger.info("=" * 60)
            
            long_horizon_results = evaluate_long_horizon_performance(
                test_loader, frame_processor, volume_grouper, prediction_runner,
                evaluator, visualizer, output_dir, args
            )
            results['long_horizon'] = long_horizon_results
        
        # Create comprehensive dashboard
        if args.eval_next_frame and args.eval_long_horizon:
            logger.info("=" * 60)
            logger.info("CREATING COMPREHENSIVE DASHBOARD")
            logger.info("=" * 60)
            
            create_comprehensive_dashboard(
                results.get('next_frame', {}),
                results.get('long_horizon', {}),
                visualizer,
                output_dir
            )
        
        # Save final summary
        final_summary = {
            'evaluation_completed': True,
            'experiment_path': args.experiment_path,
            'output_directory': str(output_dir),
            'evaluations_run': {
                'next_frame': args.eval_next_frame,
                'long_horizon': args.eval_long_horizon
            },
            'results_files': {}
        }
        
        # Add result file paths
        if 'next_frame' in results:
            final_summary['results_files']['next_frame'] = results['next_frame'].get('results_path', '')
        if 'long_horizon' in results:
            final_summary['results_files']['long_horizon'] = results['long_horizon'].get('results_path', '')
        
        summary_path = output_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Summary: {summary_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error(traceback.format_exc())
        
        # Save error information
        error_info = {
            'evaluation_failed': True,
            'error_message': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
        
        error_path = output_dir / "error_log.json"
        with open(error_path, 'w') as f:
            json.dump(error_info, f, indent=2)
        
        return 1


if __name__ == "__main__":
    exit_code = main() 