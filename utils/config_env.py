import os
import sys
import subprocess
import logging
from pathlib import Path
import matplotlib as mpl

###########################################################
# Note: This script has been mostly written by an LLM and #
# is refined with some modifications by the author(s).    #
###########################################################

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _install_packages(requirements_file='requirements.txt'):
    """
    Installs packages from a requirements.txt file using pip.
    
    Arguments:
        requirements_file (str): Path to the requirements.txt file.
    
    Returns:
        None
    """
    # Checking whether the path exists
    if not os.path.exists(requirements_file):
        logger.error(f"Requirements file '{requirements_file}' not found.")
        return
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
        logger.info("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing packages: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

def _setup_matplotlib_defaults():
    """
    Sets up default parameters for Matplotlib to improve plot aesthetics.
    
    Returns:
        None
    """
    mpl.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    logger.info("Pyplot defaults set up.")

def _setup_paths(input_dir='data/input', output_dir='data/output', qc_dir='qc', logs_dir='logs'):
    """
    Sets up input/output paths for the project.
    
    Arguments:
        input_dir (str): Base input directory.
        output_dir (str): Base output directory.
        qc_dir (str): QC subdirectory.
        logs_dir (str): Logs subdirectory.
    
    Returns:
        dict: Dictionary of paths.
    """
    paths = {
        'input': Path(input_dir).resolve(),
        'output': Path(output_dir).resolve(),
        'qc': Path(qc_dir).resolve(),
        'logs': Path(logs_dir).resolve(),
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    # Logging success
    logger.info("Project paths set up.")
    return paths

def project_setup(requirements_file='requirements.txt'):
    """
    Runs full project setup: install packages, matplotlib defaults, and paths.
    
    Arguments:
        requirements_file (str): Path to requirements.txt.
    
    Returns:
        dict: Setup status.
    """
    _install_packages(requirements_file)
    _setup_matplotlib_defaults()
    paths = _setup_paths()
    
    status = {
        'packages_installed': True,
        'matplotlib_set': True,
        'paths_set': True,
    }
    logger.info("Full project setup completed.")
    return status

# Example usage (commented out / removeed in production)
#if __name__ == "__main__":
#    project_setup()
