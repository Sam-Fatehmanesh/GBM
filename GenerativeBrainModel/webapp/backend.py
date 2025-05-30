import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

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
    from .pipeline import baseline
except ImportError: # Fallback for environments where relative imports might be tricky with script execution
    from config import settings
    from pipeline import job_manager
    from pipeline import baseline


# --- GBM Module Imports ---
try:
    from GenerativeBrainModel.utils.masks import ZebrafishMaskLoader
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
    masks_path_str = None
    try:
        masks_path_str = settings.masks_dir
        if not masks_path_str:
            print("settings.masks_dir is not set, will use default.") # Should not happen if default is set in Settings
            raise ValueError("MASKS_DIR not configured.") # Trigger fallback
    except (AttributeError, ValueError): # AttributeError if masks_dir not in settings, ValueError if forced
        default_masks_path = PROJECT_ROOT / "masks"
        print(f"Using fallback masks_dir: {default_masks_path}")
        masks_path_str = str(default_masks_path)

    masks_path = Path(masks_path_str)
    if not masks_path.exists() or not masks_path.is_dir():
        raise HTTPException(
            status_code=500,
            detail=f"Masks directory not found or is not a directory. Path: '{masks_path_str}'"
        )

    try:
        loader = ZebrafishMaskLoader(str(masks_path))
        region_names = loader.list_masks()
        if not region_names and "dummy_region_import_failed" in region_names : # check if dummy was used
             raise HTTPException(status_code=500, detail="ZebrafishMaskLoader failed to initialize properly or load regions.")
        return {"regions": region_names}
    except Exception as e:
        # Catch-all for other ZebrafishMaskLoader or file issues
        raise HTTPException(status_code=500, detail=f"Error loading regions from '{masks_path_str}': {str(e)}")


@router.post("/simulate")
async def simulate_optogenetics(request: SimulateRequest):
    """
    Submits an optogenetic simulation job (baseline modification).
    The base experiment path is taken from EXPERIMENTS_ROOT in .env.
    """
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

    try:
        job_id = job_manager.submit(
            baseline.modified_baseline_sequence,
            experiment_path=str(exp_path),
            regions=request.selected_regions,
            fraction=request.activation_fraction,
            sample_idx=0  # Default sample index, can be parameterized later
        )
        return {"job_id": job_id, "message": "Simulation job submitted."}
    except Exception as e:
        print(f"Error submitting job: {e}") # Log to server console
        raise HTTPException(status_code=500, detail=f"Failed to submit simulation job: {str(e)}")


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
        # Retrieve the output directory path from job_manager
        output_dir_str = job_manager.result(job_id)
        output_dir = Path(output_dir_str)

        # Files saved by baseline.py
        baseline_file = output_dir / "baseline_sequence.npy"
        activation_mask_file = output_dir / "activation_mask.npy"

        files_generated = {}
        if baseline_file.exists():
            files_generated["baseline_sequence_path"] = str(baseline_file)
        else:
            files_generated["baseline_sequence_path"] = "File not found"
        
        if activation_mask_file.exists():
            files_generated["activation_mask_path"] = str(activation_mask_file)
        else:
            files_generated["activation_mask_path"] = "File not found"
            
        return {
            "job_id": job_id,
            "status": current_status,
            "output_directory": str(output_dir),
            "files": files_generated
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving results for completed job '{job_id}': {str(e)}")


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