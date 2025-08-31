# Importing MNE packages
import mne
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Defining constants
_OUTPUT_MODE = [0, 1, 2]

# Downloading the data
def download_data(url = 'https://osf.io/ksqv8/download', save_path = 'alldat.npy'):
    try:
        response = requests.get(url, stream = True)
        response.raise_for_status() # This raises HTTP error for bad responses
        # Writing file in chunks
        with open(save_path, 'wb'):
            for chunk in response.iter_content(chunk_size = 8192):
                if chunk:
                    file.write(chunk)
        print(f"File successfully downloaded as {save_path}")
    # Exceptions generated using Copilot
    #except requests.exceptions.Timeout:
    #    print("The request timed out. Please try again later.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    except IOError as io_error:
        print(f"File write error: {io_error}")

# Loading the data using MNE from the dictionary
# Note that alldat requires transposition in order to have the shape (2, 7)
# 7 patients, 2 experiments (01 = motor execution, 02 = motor imagery)
def _split_data_into_exp(alldat, output_mode = 0, store_brodmanns = False):
    # Declaring variables to separate the two experiments
    exp01 = alldat.T[0]
    exp02 = alldat.T[1]
    
    # Storing the dimensions for the dataset and the number of channels
    num_subjects = exp01.shape[0]
    num_channels = [exp01[i]['V'].shape[1] for i in range(num_subjects)]
    time_steps = [exp01[i]['V'].shape[0] for i in range(num_subjects)]

    if (output_mode == 0):
        # 0: Scalars only
        return (num_subjects, num_channels, time_steps)
    
    # Extracting data matrices
    data_exp01 = [exp01[i]['V'] for i in range(num_subjects)]
    data_exp02 = [exp02[i]['V'] for i in range(num_subjects)]
    
    # Converting the data into microvolt units
    data_exp01 = [data_exp01[i] * exp01[i]['scale_uv'] for i in range(num_subjects)]
    data_exp02 = [data_exp02[i] * exp02[i]['scale_uv'] for i in range(num_subjects)]
    
    # Transposing data to fit the MNE standards
    data_exp01 = [x.T for x in data_exp01]
    data_exp02 = [x.T for x in data_exp02]

    # Naming channels and setting the sampling frequency
    sampling_freq = exp01[0]['srate']

    if (store_brodmanns):
        # Storing the information for the Brodmann areas
        brodmann_areas_exp01 = [exp01[i]['Brodmann_Area'] for i in range(num_subjects)]
        brodmann_areas_exp02 = [exp02[i]['Brodmann_Area'] for i in range(num_subjects)]

    if output_mode == 1:
        # 1: Arrays only
        return (data_exp01, data_exp02)
    else:
        # 2 or other numbers: All of the extracted data
        return (
            num_subjects, num_channels, time_steps,
            data_exp01, data_exp02
        )

# Declaring a function for creating epochs
def _create_epochs_single_exp(rawarray_input, num_subjects, events, event_ids : dict, t_min, t_max, padding = (0, 0)) -> tuple:
    """
    Creates epochs from raw arrays.
    Args:
        rawarray (list): The input raw array.
        num_subjects (int): Number of subjects.
        events (list): List of events.
        event_ids (dict): Dictionary of event ids.
    Returns:
        tuple: Tuple of epochs.
    """
    # Creating epoch list
    epochs_exp = []
    # Creating Epochs for each of the subjects
    for i in range(num_subjects):
        print(f'\nSubject {i + 1}/{num_subjects}:')
        epochs_exp.append(mne.Epochs(rawarray_input[i], events=events[i],
                            event_id=event_ids, tmin=t_min,
                            tmax=t_max, baseline = (0, 0), preload=True))
    return epochs_exp
