"""
Tools for using PandExo
"""

import numpy as np
import pandas as pd
from . import myutils

def format_spectrum_for_pandexo(wavs, Rt_RJ, filename):
    """
    Formats a spectrum into a file to be read by PandExo.
    Uses stellar parameters of TRAPPIST-1
    Output is Delta = (R_p / R_star)**2
    """
    R_star = 0.12 * 6.96e8  # TRAPPIST-1: 0.12R_sun
    Rt = Rt_RJ * 6.99e7  # R_J = 7e7
    Delta = (Rt / R_star) ** 2
    filetext = ""
    for w, D in zip(wavs, Delta):
        filetext += f"{w:.8f}\t{D:.8f}\n"
    filetext = filetext[:-1]  # Removes trailing \n
    myutils.overwrite_to(filename, filetext)


def extract_spectrum_from_pandexo(filepath, random_noise=False):
    """
    Extracts the wavelengths, spectrum, and errors from a
     PandExo output file
    Can optionally add PandExo's random noise to the spectrum
     (though not too sure how it does that)
    """
    spec_dict = pd.read_pickle(filepath)["FinalSpectrum"]
    wavelengths = spec_dict["wave"]
    if random_noise:
        spectrum_w_noise = spec_dict["spectrum_w_rand"]
        return wavelengths, spectrum_w_noise
    spectrum = spec_dict["spectrum"]
    error = spec_dict["error_w_floor"]
    return wavelengths, spectrum, error


def create_spectrum_datafile(filepath, N_transits=1):
    """
    Creates a .txt file to be used by pRT for retrieval
    """
    wavelengths, spectrum, error = extract_spectrum_from_pandexo(filepath)
    error /= np.sqrt(N_transits)
    np.savetxt(
        f'{filepath[:-2]}_{N_transits}.txt',
        np.array([wavelengths, spectrum, error]).T,
        fmt=['%.7f', '%.9f', '%.9f'],
        header = 'Wavelength [micron], Flux [(Rp/Rstar)^2], Flux Error [(Rp/Rstar)^2]'
    )
