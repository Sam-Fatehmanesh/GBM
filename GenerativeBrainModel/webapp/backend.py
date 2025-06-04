import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import h5py
import numpy as np

# --- Path Setup ---
# Assuming backend.py is in GenerativeBrainModel/webapp/backend.py
# PROJECT_ROOT will be 'GBM/'
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Local Imports ---
# These are expected to be in GenerativeBrainModel/webapp/ or accessible via PROJECT_ROOT in sys.path
try:
    from .config import settings # Relative import for config.py in the same directory (webapp)
    from .pipeline import job_manager
    from .pipeline.run_job import run_simulation
    from .pipeline import baseline
    from .pipeline.analysis import analyze_region_differences
except ImportError: # Fallback for environments where relative imports might be tricky with script execution
    from config import settings
    from pipeline import job_manager
    from pipeline.run_job import run_simulation
    from pipeline import baseline
    from pipeline.analysis import analyze_region_differences


# --- GBM Module Imports ---
try:
    from GenerativeBrainModel.utils.masks import ZebrafishMaskLoader, load_zebrafish_masks
except ImportError as e:
    print(f"CRITICAL: Could not import ZebrafishMaskLoader from GenerativeBrainModel.utils.visualizations_utils. Error: {e}")
    print(f"PROJECT_ROOT (added to sys.path): {PROJECT_ROOT}")
    print(f"Current sys.path: {sys.path}")
    # Placeholder to allow app to start for debugging, but /regions will fail or return dummy data.
    class ZebrafishMaskLoader:
        def __init__(self, masks_path: str):
            print(f"WARNING: Using DUMMY ZebrafishMaskLoader. Masks path configured: {masks_path}")
            if not Path(masks_path).exists():
                print(f"WARNING: Dummy ZebrafishMaskLoader: Provided masks_path does not exist: {masks_path}")
            self.masks_path = masks_path
        def get_available_region_names(self) -> List[str]:
            print("WARNING: Dummy ZebrafishMaskLoader returning empty list for regions because main import failed.")
            # Optionally, try to list .npy files if path exists, as a very basic fallback
            try:
                if Path(self.masks_path).exists() and Path(self.masks_path).is_dir():
                    return sorted([p.stem for p in Path(self.masks_path).glob('*.npy')])
            except Exception as ex:
                print(f"Dummy ZebrafishMaskLoader: Error trying to list npy files: {ex}")
            return ["dummy_region_import_failed"]

app = FastAPI()
# Global cache for mask loader and preloaded z-start and Z
mask_loader_global = None
baseline_zStart_global = None
baseline_Z_global = None
# Mount static directory for frontend assets
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
async def preload_masks():
    global mask_loader_global, baseline_zStart_global, baseline_Z_global
    """
    Preload the zebrafish masks using the dynamic Z, Y, X from the experiment so they are cached before any simulate calls.
    """
    try:
        # Determine masks directory
        masks_dir_str = settings.masks_dir or str(PROJECT_ROOT / "masks")
        # Parse experiment_info to find preaugmented_dir and subject
        exp_root = Path(settings.experiments_root)
        exp_info = exp_root / 'experiment_info.txt'
        preaug_dir = None
        subject = None
        if exp_info.exists():
            with open(exp_info, 'r') as ef:
                for line in ef:
                    lt = line.strip()
                    if lt.lower().startswith('preaugmented_dir:'):
                        preaug_dir = lt.split(':',1)[1].strip()
                    elif lt.lower().startswith('target_subject:') or lt.lower().startswith('target subject:'):
                        subject = lt.split(':',1)[1].strip()
        if not preaug_dir or not subject:
            raise RuntimeError(f"Cannot preload masks: missing preaugmented_dir or target_subject in {exp_info}")
        # Read number of z-planes from subject metadata
        metadata_h5 = Path(preaug_dir) / subject / 'metadata.h5'
        with h5py.File(metadata_h5, 'r') as mf:
            Z = int(mf['num_z_planes'][()])
            # Cache global Z
            baseline_Z_global = Z
        # Read H, W from a test data HDF5 if available
        test_h5 = exp_root / 'finetune' / 'test_data' / 'test_data_and_predictions.h5'
        if test_h5.exists():
            with h5py.File(test_h5, 'r') as tf:
                # dataset shape: (num_samples, seq_len, H, W)
                shape = tf['test_data'].shape
                H = int(shape[2])
                W = int(shape[3])
                # Preload sequence_z_starts
                if 'sequence_z_starts' not in tf:
                    raise RuntimeError(f"Missing 'sequence_z_starts' in {test_h5}")
                baseline_zStart_global = int(tf['sequence_z_starts'][0])
        else:
            # Fallback to default mask loader dimensions
            _, default_Y, default_X = load_zebrafish_masks().target_shape
            H, W = default_Y, default_X
        # Preload masks for dynamic shape and cache loader
        loader = load_zebrafish_masks(masks_dir_str, target_shape=(Z, H, W))
        mask_loader_global = loader
        print(f"Preloaded masks with shape (Z={Z}, Y={H}, X={W}) into cache")
    except Exception as e:
        print(f"Warning: could not preload masks: {e}")

@app.get("/")
async def serve_frontend():
    return FileResponse(str(static_dir / "index.html"))

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Models ---
class SimulateRequest(BaseModel):
    selected_regions: List[str]
    activation_fraction: float
    prediction_steps: int # Kept for future use in inference, not used by baseline.py

# --- API Router ---
router = APIRouter()

@router.get("/regions")
async def get_regions():
    """
    Returns a list of available brain region names from the masks directory.
    The path to the masks directory is determined by MASKS_DIR in .env,
    with a fallback to 'PROJECT_ROOT/masks'.
    """
    # If masks have been preloaded, reuse the global loader
    if mask_loader_global:
        names = mask_loader_global.list_masks()
        if not names or "dummy_region_import_failed" in names:
            raise HTTPException(status_code=500, detail="Cached ZebrafishMaskLoader unavailable or failed to load regions.")
        return {"regions": names}
    # Fallback: load masks on demand (rare)
    masks_dir = settings.masks_dir or str(PROJECT_ROOT / "masks")
    path = Path(masks_dir)
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=500, detail=f"Masks directory not found: {masks_dir}")
    try:
        loader = load_zebrafish_masks(str(path))
        names = loader.list_masks()
        if not names:
            raise HTTPException(status_code=500, detail="Failed to load masks on demand.")
        return {"regions": names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading regions: {e}")

@router.get("/regions/{region_name}/mask_json")
async def get_region_mask_json(region_name: str):
    """
    Return 3D coordinates for a specific brain region mask.
    """
    try:
        # Use the global mask loader if available
        global mask_loader_global, baseline_Z_global
        if not mask_loader_global:
            raise HTTPException(status_code=500, detail="Mask loader not available")
        
        if baseline_Z_global is None:
            raise HTTPException(status_code=500, detail="Z dimension not available")
        
        # Get the region mask
        region_mask = mask_loader_global.get_mask(region_name).cpu().numpy().astype(bool)
        Z, H, W = region_mask.shape
        
        # Find all voxel coordinates where the region is active
        coords = np.argwhere(region_mask)  # returns [z, y, x] coordinates
        coords_list = coords.tolist()
        
        return JSONResponse({
            'region_name': region_name,
            'coords': coords_list,  # [[z, y, x], ...]
            'Z': Z,
            'H': H, 
            'W': W
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading region {region_name}: {str(e)}")

@router.post("/simulate")
async def simulate_optogenetics(request: SimulateRequest):
    """
    Submits an optogenetic simulation job (baseline modification).
    The base experiment path is taken from EXPERIMENTS_ROOT in .env.
    Uses the global preloaded mask loader to avoid reloading masks.
    """
    global mask_loader_global
    
    exp_path_str = None
    try:
        exp_path_str = settings.experiments_root
        if not exp_path_str: # Should be handled by Settings default if not in .env
             raise HTTPException(status_code=500, detail="EXPERIMENTS_ROOT not configured.")
    except AttributeError:
        raise HTTPException(status_code=500, detail="EXPERIMENTS_ROOT setting is missing in configuration.")

    exp_path = Path(exp_path_str)
    if not exp_path.exists() or not exp_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Configured EXPERIMENTS_ROOT is not a valid directory: {exp_path_str}")

    # Check if mask loader is available
    if mask_loader_global is None:
        raise HTTPException(status_code=500, detail="Mask loader not preloaded. Server may still be starting up.")

    try:
        job_id = job_manager.submit(
            run_simulation,
            regions=request.selected_regions,
            fraction=request.activation_fraction,
            sample_idx=0,
            num_steps=request.prediction_steps,
            mask_loader=mask_loader_global
        )
        return {"job_id": job_id, "message": "Simulation job submitted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit simulation job: {e}")


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Returns the status of a submitted job.
    """
    try:
        status = job_manager.status(job_id)
        return {"job_id": job_id, "status": status}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving status for job '{job_id}': {str(e)}")


@router.get("/result/{job_id}")
async def get_job_result(job_id: str):
    """
    Returns the result of a completed job (paths to generated files).
    """
    try:
        current_status = job_manager.status(job_id) # Checks if job_id is valid
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking status for job '{job_id}' before getting result: {str(e)}")
    if current_status["status"] != "done":
        return {"job_id": job_id, "status": current_status["status"], "message": "Job is not yet completed."}

    try:
        # Retrieve the job result (baseline + predictions)
        job_res = job_manager.result(job_id)
        # Unpack baseline and predictions directories
        if isinstance(job_res, dict):
            baseline_dir = Path(job_res.get('baseline_dir', ''))
            predictions_dir = Path(job_res.get('predictions_dir', ''))
        else:
            baseline_dir = Path(job_res)
            predictions_dir = None

        files_generated = {}
        # Baseline outputs
        baseline_file = baseline_dir / "baseline_sequence.npy"
        activation_mask_file = baseline_dir / "activation_mask.npy"
        files_generated["baseline_sequence"] = str(baseline_file) if baseline_file.exists() else None
        files_generated["activation_mask"] = str(activation_mask_file) if activation_mask_file.exists() else None
        # Include sequence z-start index file if generated
        sequence_start_file = baseline_dir / "sequence_z_start.npy"
        files_generated["sequence_z_start"] = str(sequence_start_file) if sequence_start_file.exists() else None

        # Prediction outputs
        if predictions_dir and predictions_dir.exists():
            pred_seq_file = predictions_dir / "predicted_sequence.npy"
            pred_prob_file = predictions_dir / "predicted_probabilities.npy"
            files_generated["predicted_sequence"] = str(pred_seq_file) if pred_seq_file.exists() else None
            files_generated["predicted_probabilities"] = str(pred_prob_file) if pred_prob_file.exists() else None
        # Analysis outputs
        analysis_dir = None
        if isinstance(job_res, dict) and 'analysis_dir' in job_res:
            analysis_dir = Path(job_res['analysis_dir'])
        if analysis_dir and analysis_dir.exists():
            summary_json = analysis_dir / 'summary.json'
            summary_csv = analysis_dir / 'summary.csv'
            heatmap_png = analysis_dir / 'heatmap.png'
            files_generated['summary_json'] = str(summary_json) if summary_json.exists() else None
            files_generated['summary_csv'] = str(summary_csv) if summary_csv.exists() else None
            files_generated['heatmap'] = str(heatmap_png) if heatmap_png.exists() else None
        return {
            "job_id": job_id,
            "status": current_status,
            "baseline_dir": str(baseline_dir),
            "predictions_dir": str(predictions_dir) if predictions_dir else None,
            "analysis_dir": str(analysis_dir) if analysis_dir else None,
            "files": files_generated
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving results for completed job '{job_id}': {str(e)}")

@router.post("/analyze/{job_id}")
async def analyze_job(job_id: str):
    """
    Run region-difference analysis for a completed simulation job on all regions.
    """
    status = job_manager.status(job_id)
    if status.get("status") != "done":
        raise HTTPException(status_code=400, detail=f"Job {job_id} not completed: {status.get('status')}")
    job_res = job_manager.result(job_id)
    if not job_res or not isinstance(job_res, dict):
        raise HTTPException(status_code=500, detail=f"Invalid job result for {job_id}")
    baseline_dir = job_res.get('baseline_dir')
    predictions_dir = job_res.get('predictions_dir')
    if not baseline_dir or not predictions_dir:
        raise HTTPException(status_code=500, detail=f"Missing baseline or predictions directory for job {job_id}")
    try:
        # Analyze differences across all regions
        analysis_dir = analyze_region_differences(
            baseline_dir,
            predictions_dir
        )
        return {"job_id": job_id, "analysis_dir": analysis_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/results/{job_id}/summary.json")
async def get_summary_json(job_id: str):
    """Fetch the analysis summary JSON for a given job."""
    job_res = job_manager.result(job_id)
    if not job_res or not isinstance(job_res, dict):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found or not completed")
    analysis_dir = Path(job_res['predictions_dir']) / 'analysis'
    file_path = analysis_dir / 'summary.json'
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="summary.json not found")
    return FileResponse(str(file_path), media_type='application/json', filename='summary.json')

@router.get("/results/{job_id}/summary.csv")
async def get_summary_csv(job_id: str):
    """Fetch the analysis summary CSV for a given job."""
    job_res = job_manager.result(job_id)
    if not job_res or not isinstance(job_res, dict):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found or not completed")
    analysis_dir = Path(job_res['predictions_dir']) / 'analysis'
    file_path = analysis_dir / 'summary.csv'
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="summary.csv not found")
    return FileResponse(str(file_path), media_type='text/csv', filename='summary.csv')

@router.get("/results/{job_id}/heatmap.png")
async def get_heatmap(job_id: str):
    """Fetch the region-volume heatmap PNG for a given job."""
    job_res = job_manager.result(job_id)
    if not job_res or not isinstance(job_res, dict):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found or not completed")
    analysis_dir = Path(job_res['predictions_dir']) / 'analysis'
    file_path = analysis_dir / 'heatmap.png'
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="heatmap.png not found")
    return FileResponse(str(file_path), media_type='image/png', filename='heatmap.png')

@router.get("/results/{job_id}/predicted_sequence.npy")
async def get_predicted_sequence(job_id: str):
    job_res = job_manager.result(job_id)
    if not job_res or not isinstance(job_res, dict):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found or not completed")
    predictions_dir = Path(job_res['predictions_dir'])
    file_path = predictions_dir / 'predicted_sequence.npy'
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="predicted_sequence.npy not found")
    return FileResponse(str(file_path), media_type='application/octet-stream', filename='predicted_sequence.npy')

@router.get("/results/{job_id}/predicted_probabilities.npy")
async def get_predicted_probabilities(job_id: str):
    job_res = job_manager.result(job_id)
    if not job_res or not isinstance(job_res, dict):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found or not completed")
    predictions_dir = Path(job_res['predictions_dir'])
    file_path = predictions_dir / 'predicted_probabilities.npy'
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="predicted_probabilities.npy not found")
    return FileResponse(str(file_path), media_type='application/octet-stream', filename='predicted_probabilities.npy')

@router.get("/results/{job_id}/activation_mask_json")
async def get_activation_mask_json(job_id: str):
    """
    Return a list of active voxel coordinates for the activation mask as JSON.
    """
    try:
        job_res = job_manager.result(job_id)
        if not job_res or not isinstance(job_res, dict):
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or not completed")
        baseline_dir = Path(job_res['baseline_dir'])
        mask_file = baseline_dir / 'activation_mask.npy'
        if not mask_file.exists():
            raise HTTPException(status_code=404, detail="activation_mask.npy not found")
        import numpy as np
        mask = np.load(mask_file)
        # Find active voxel coordinates (z, y, x)
        coords = np.argwhere(mask)
        coords_list = coords.tolist()
        return JSONResponse({'coords': coords_list})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/baseline_mask_json")
async def get_baseline_mask_json():
    """
    Return active voxel coordinates for the baseline brain activity (sample 0) from the test data HDF5.
    """
    # Locate test_data HDF5 in finetune folder
    exp_root = Path(settings.experiments_root)
    test_h5 = exp_root / 'finetune' / 'test_data' / 'test_data_and_predictions.h5'
    if not test_h5.exists():
        raise HTTPException(status_code=404, detail=f"Baseline test HDF5 not found: {test_h5}")
    try:
        # Ensure zStart and Z were preloaded at startup
        global baseline_zStart_global, baseline_Z_global
        if baseline_zStart_global is None or baseline_Z_global is None:
            raise HTTPException(status_code=500, detail="Baseline sequence metadata not preloaded at startup")
        zStart = baseline_zStart_global
        Z = baseline_Z_global
        # Load baseline test sequence (flattened slices) from HDF5
        with h5py.File(test_h5, 'r') as f:
            data = f['test_data'][0]  # shape: (seq_len, H, W)
            # Convert to boolean mask and get coords for all slices
            mask = data > 0
            coords = np.argwhere(mask)  # returns [slice_idx, y, x]
            coords_list = coords.tolist()
            return JSONResponse({'coords': coords_list, 'Z': Z, 'zStart': zStart})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{job_id}/predicted_sequence_json")
async def get_predicted_sequence_json(job_id: str):
    """
    Return predicted brain activity coordinates for time-based animation.
    Similar to baseline_mask_json but for predicted probabilities.
    """
    try:
        job_res = job_manager.result(job_id)
        if not job_res or not isinstance(job_res, dict):
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or not completed")
        
        predictions_dir = Path(job_res['predictions_dir'])
        baseline_dir = Path(job_res['baseline_dir'])
        
        # Load predicted probabilities
        probs_file = predictions_dir / 'predicted_probabilities.npy'
        if not probs_file.exists():
            raise HTTPException(status_code=404, detail="predicted_probabilities.npy not found")
        
        # Load sequence z-start for alignment
        seq_start_file = baseline_dir / 'sequence_z_start.npy'
        if not seq_start_file.exists():
            raise HTTPException(status_code=404, detail="sequence_z_start.npy not found")
        
        # Use global Z from startup
        global baseline_Z_global, baseline_zStart_global
        if baseline_Z_global is None:
            raise HTTPException(status_code=500, detail="Z dimension not available")
        
        Z = baseline_Z_global
        zStart = int(np.load(seq_start_file))
        
        # Load predicted probabilities (shape: num_steps, H, W)
        probabilities = np.load(probs_file)
        num_steps, H, W = probabilities.shape
        
        # Convert probabilities to binary mask via Bernoulli sampling for visualization
        samples = np.random.binomial(1, probabilities)
        mask = samples.astype(bool)
        
        # Convert to coordinate list like baseline data
        coords = []
        for t in range(num_steps):
            active_pixels = np.argwhere(mask[t])  # returns [y, x] coordinates
            for y, x in active_pixels:
                coords.append([t, int(y), int(x)])  # [slice_idx, y, x] format
        
        return JSONResponse({
            'coords': coords, 
            'Z': Z, 
            'zStart': zStart
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{job_id}/heatmap_top12.png")
async def get_heatmap_top12(job_id: str):
    """Fetch the top 12 regions heatmap PNG for a given job."""
    job_res = job_manager.result(job_id)
    if not job_res or not isinstance(job_res, dict):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found or not completed")
    analysis_dir = Path(job_res['predictions_dir']) / 'analysis'
    file_path = analysis_dir / 'heatmap_top12.png'
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="heatmap_top12.png not found")
    return FileResponse(str(file_path), media_type='image/png', filename='heatmap_top12.png')

app.include_router(router)

# --- Main Entry Point ---
if __name__ == "__main__":
    print("Attempting to start FastAPI server...")
    try:
        print(f"  FastAPI host: 0.0.0.0, port: 8000")
        print(f"  Project Root (for sys.path): {PROJECT_ROOT}")
        
        # Access settings to ensure they are loaded and print them
        s = settings
        print(f"  Configured EXPERIMENTS_ROOT: {s.experiments_root}")
        if not Path(s.experiments_root).exists():
             print(f"  WARNING: EXPERIMENTS_ROOT path does not exist: {s.experiments_root}")
        
        print(f"  Configured MASKS_DIR: {s.masks_dir}")
        if not Path(s.masks_dir).exists():
             print(f"  WARNING: MASKS_DIR path does not exist: {s.masks_dir}")
             print(f"    (Note: /regions endpoint might use fallback: {PROJECT_ROOT / 'masks'})")

    except Exception as e:
        print(f"Error during pre-run configuration check: {e}")

    uvicorn.run(app, host="0.0.0.0", port=8000) 