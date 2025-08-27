import os
import sys
import subprocess
import logging
from pathlib import Path
import matplotlib as mpl
import mne

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_packages(requirements_file='requirements.txt'):
    """
    Install packages from a requirements.txt file using pip.
    
    Parameters:
    requirements_file (str): Path to the requirements.txt file.
    
    Returns:
    None
    """
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

def setup_matplotlib_defaults():
    """
    Set up default parameters for Matplotlib to improve plot aesthetics.
    
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
    logger.info("Matplotlib defaults set up.")

def get_default_epoch_params():
    """
    Get default parameters for creating MNE Epochs.
    
    Returns:
    dict: Dictionary of default parameters.
    """
    defaults = {
        'tmin': -0.5,  # Start time before event (in seconds)
        'tmax': 1.5,   # End time after event (in seconds)
        'baseline': (None, 0),  # Baseline correction interval
        'preload': True,
        'event_id': None,  # To be set based on dataset (e.g., {'hand': 1, 'tongue': 2})
        'reject': {'ecog': 1e-3},  # Rejection criteria for bad epochs
        'flat': {'ecog': 1e-6},    # Flat signal rejection
        'proj': True,
        'decim': 1,
    }
    logger.info("Default epoch parameters retrieved.")
    return defaults

def calculate_baseline(epochs, mode='logratio', baseline=(None, 0)):
    """
    Calculate and apply baseline correction to MNE Epochs.
    
    Parameters:
    epochs (mne.Epochs): The epochs object to baseline.
    mode (str): Baseline mode ('mean', 'ratio', 'logratio', 'percent', 'zscore', 'zlogratio').
    baseline (tuple): Time interval for baseline (start, end).
    
    Returns:
    mne.Epochs: Baslined epochs.
    """
    if not isinstance(epochs, mne.Epochs):
        logger.error("Input must be an mne.Epochs object.")
        return None
    
    try:
        epochs.apply_baseline(baseline=baseline, mode=mode)
        logger.info(f"Baseline applied with mode '{mode}' over interval {baseline}.")
        return epochs
    except Exception as e:
        logger.error(f"Error applying baseline: {e}")
        return None

def perform_qc(epochs, plot_dir='qc_plots', n_jobs=1):
    """
    Perform quality control on MNE Epochs, including plots and stats.
    
    Parameters:
      epochs (mne.Epochs): The epochs to QC.
      plot_dir (str): Directory to save QC plots.
      n_jobs (int): Number of jobs for parallel processing.
    
    Returns:
      dict: QC statistics (e.g., dropped epochs, variance).
    """
    if not isinstance(epochs, mne.Epochs):
        logger.error("Input must be an mne.Epochs object.")
        return None
    
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    
    qc_stats = {
        'n_epochs_original': len(epochs),
        'n_epochs_dropped': 0,
        'channel_variances': {},
    }
    
    try:
        # Drop bad epochs
        epochs.drop_bad()
        qc_stats['n_epochs_dropped'] = qc_stats['n_epochs_original'] - len(epochs)
        
        # Compute variances per channel
        data = epochs.get_data()
        for ch_idx, ch_name in enumerate(epochs.ch_names):
            qc_stats['channel_variances'][ch_name] = np.var(data[:, ch_idx, :])
        
        # Plot PSD
        fig = epochs.plot_psd(fmax=100, n_jobs=n_jobs)
        fig.savefig(os.path.join(plot_dir, 'psd_plot.png'))
        plt.close(fig)
        
        # Plot epochs
        fig = epochs.plot(n_epochs=10, n_channels=10)
        fig.savefig(os.path.join(plot_dir, 'epochs_plot.png'))
        plt.close(fig)
        
        logger.info("QC performed and plots saved.")
    except Exception as e:
        logger.error(f"Error during QC: {e}")
    
    return qc_stats

def setup_paths(input_dir='data/input', output_dir='data/output', qc_dir='qc', logs_dir='logs'):
    """
    Set up input/output paths for the project.
    
    Parameters:
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
    
    logger.info("Project paths set up.")
    return paths

def project_setup(requirements_file='requirements.txt'):
    """
    Run full project setup: install packages, matplotlib defaults, and paths.
    
    Parameters:
    requirements_file (str): Path to requirements.txt.
    
    Returns:
    dict: Setup status.
    """
    install_packages(requirements_file)
    setup_matplotlib_defaults()
    paths = setup_paths()
    
    status = {
        'packages_installed': True,
        'matplotlib_set': True,
        'paths_set': True,
    }
    logger.info("Full project setup completed.")
    return status

# Example usage (comment out or remove in production)
if __name__ == "__main__":
    project_setup()
    # Example: Load your data and use functions
    # raw = mne.io.read_raw('path/to/raw.fif')
    # events = mne.find_events(raw)
    # epoch_params = get_default_epoch_params()
    # epoch_params['event_id'] = {'hand': 1, 'tongue': 2}
    # epochs = mne.Epochs(raw, events, **epoch_params)
    # epochs = calculate_baseline(epochs)
    # qc_stats = perform_qc(epochs)
    # print(qc_stats)
