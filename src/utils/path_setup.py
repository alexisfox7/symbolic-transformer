import os
import sys
from pathlib import Path

def setup_project_paths():
    """
    Setup project paths to ensure imports work from anywhere.
    Call this at the top of any script that needs project imports.
    """
    # Get the project root (where this file's parent's parent is)
    current_file = Path(__file__).resolve()
    
    # Go up to find src directory
    src_dir = current_file.parent.parent  # src/utils/path_setup.py -> src/
    project_root = src_dir.parent          # src/ -> project_root/
    
    # Add src to Python path if not already there
    src_path = str(src_dir)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Also add project root for good measure
    root_path = str(project_root)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    
    return src_dir, project_root

def get_project_root():
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent  # src/utils/path_setup.py -> project_root/

def get_src_dir():
    """Get the src directory."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent  # src/utils/path_setup.py -> src/