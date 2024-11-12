from ensure import ensure_annotations
from pathlib import Path
from src import logger
import os

def get_project_root() -> Path:

    i = 3 #Since project structure are max at 3 level
    parent_path = Path()
    
    while i > 0:  

        if Path.exists(parent_path / 'src'):
            return parent_path
        else:
            parent_path = parent_path.resolve().parent
        
        i = i - 1

    return parent_path

@ensure_annotations
def create_directories(path_to_directories: list, verbose = True):
    """
    create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:

        path = get_project_root().joinpath(path)

        if not path.exists():
            
            os.makedirs(path, exist_ok = True)

            if verbose:
                logger.info(f"Created directory at: {path}")
