"""
fOxhunter.py
Varies oxygen abundance to achieve a desired fO2 at a particular p, T
Assumes Venus abundances of all other elements
This program should be run from just outside the GGchem folder
"""
import os
import numpy as np
import pandas as pd
from scipy import optimize

# Reading and writing utils
def read_from(filename):
    """Returns text in `filename`
    This is just here because python is annoying me"""
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def write_to(filename, text):
    """Overwrites `text` to `filename"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)


def prepare_input_file(p, T):
    """
    Prepares the input file on which GGchem will run
    by setting the desired p and T
    """
    input_file = "./GGchem/input/model_0D_venus.in"
    input_filetext = read_from(input_file)
    new_input_filetext = "\n".join(input_filetext.split("\n")[:-2])
    new_input_filetext += f"\n{T}\t\t\t! Tmax [K]\n{p}\t\t\t! pmax [bar]"
    write_to(input_file, new_input_filetext)


def prepare_abund_file(epsO=19.559078521):
    """
    Prepares the abundance file, for the desired epsO
    NB: Venus has epsO = 19.559078521; this is the default value
    """
    eps_df = pd.read_csv(
        "./GGchem/abund_Venus_O.in",
        delim_whitespace=True,
        index_col=0,
        names=["element", "eps"],
    )
    eps_df.loc["O", "eps"] = epsO
    eps_df.to_csv("./GGchem/abund_Venus_O.in", sep="\t", header=False)


def get_log_fO2(p, T, epsO):
    """
    Runs GGchem on for the given elemental composition, p, T;
    Hacks the mole fraction of O2 from the GGchem output
    """
    prepare_input_file(p, T)
    prepare_abund_file(epsO)
    ggchem_output = os.popen(
        "cd ./GGchem && ./ggchem input/model_0D_venus.in && cd .."
    ).read()  # Runs GGchem and captures output
    mf_O2 = float(
        ggchem_output.split("nO2")[1].split("\n")[0].split()[2]
    )  # This is the hacky bit!
    return np.log10(mf_O2 * p)


def fOxhunt(target_log_fO2, p, T):
    """
    Calculates the epsO such that, for a given p, T,
    log_fO2 reaches the target value
    Takes ~40min for target_log_fO2=-20
    """
    epsO_ontarget = optimize.bisect(
        lambda epsO: get_log_fO2(p, T, epsO) - target_log_fO2,
        19.52,
        19.57,
        xtol=10**target_log_fO2,
    )
    return epsO_ontarget
