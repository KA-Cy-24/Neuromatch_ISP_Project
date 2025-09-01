###########################################
#     Functional Connectivity Module      #
###########################################


# Defining a function for performing wPLI FC analysis on Epoch arrays resulting from the filters
def perform_fc_analysis(epochs_dict : dict, band_keys : list, num_subjects : int = 7, method = "wpli", t_min=0, t_max=3):
    """
    Performs functional connectivity analysis (wPLI) on epoched data for different frequency bands.
    It is better to set up overlapping epochs to account for the difference between the activity from the stimuli.

    Args:
        epochs_dict (dict): A dictionary where keys are frequency band names and values are lists of MNE Epochs objects for each subject.
        band_keys (list): A list of frequency band names to analyze.
        num_subjects (int): The number of subjects.
        method (str): The connectivity method to use (default is `"wpli"`).
        t_min (float): The lower bound of the time window for connectivity analysis (default is 0).
        t_max (float): The upper bound of the time window for connectivity analysis (default is 3).

    Returns:
        dict: A dictionary where keys are frequency band names and values are lists of connectivity matrices for each subject.
    """
    connectivity_results = {}
    for band in band_keys:
        print(f"Performing {method} connectivity analysis for {band} band...")
        band_epochs = epochs_dict[band]
        subject_connectivity = []
        for i in range(num_subjects):
            print(f"  Subject {i + 1}/{num_subjects}...")
            # Compute connectivity for the current subject and band
            connectivity_obj = spectral_connectivity_epochs(
                band_epochs[i],
                method=method,
                mode='fourier', # Use fourier mode for frequency-domain connectivity
                sfreq=band_epochs[i].info['sfreq'],
                fmin=freq_bands[band][0], # Use the frequency range of the band
                fmax=freq_bands[band][1],
                faverage=True, # Average connectivity over the frequency band
                tmin=t_min, # Use the specified time window
                tmax=t_max,
                n_jobs=1 # Use 1 job for now, can increase for parallel processing
            )
            # Extracting the connectivity matrix from the Connectivity object
            subject_connectivity.append(con.get_data(output='dense'))
        connectivity_results[band] = subject_connectivity
    return connectivity_results
