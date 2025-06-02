import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

class Settings:
    # Base directory where experiments are stored
    experiments_root: str = os.getenv("EXPERIMENTS_ROOT", "")
    # Directory containing zebrafish mask files
    masks_dir: str = os.getenv("MASKS_DIR", "")

settings = Settings() 