"""
Calculating gas and mineral abundances over a pT grid
    for general composition.
A generalisation of venus_grid.py
"""

import os
import io
import numpy as np
import pandas as pd
from tqdm import tqdm

# Grid parameters
PMIN = 0.1  # bar
PMAX = 1e4  # bar
PBOUNDS = [PMIN, PMAX]
TMIN = 650  # K
TMAX = 2000  # K
TBOUNDS = [TMIN, TMAX]
NPOINTS = 100  # grid fineness

ABUND_FILE = "abund_Venus.in"


def read_from(filename):
    """
    Extracts the text from `filename`
    """
    with open(filename, "r", encoding="utf-8") as f:
        filetext = f.read()
    return filetext


def overwrite_to(filename, text):
    """
    Overwrites `filename`'s text with `text`
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)


def write_to(filename, text):
    """
    Appends `text` to the text in `filename`
    """
    with open(filename, "a", encoding="utf-8") as f:
        f.write(text)


def create_GGchem_file_pT(p, Tbounds=None, Npoints=NPOINTS, abund_file=ABUND_FILE):
    """
    Creates the GGchem input file for a particular p,
    for a given temperature range, precision, and
    set of elemental abundances
    """
    if Tbounds is None:
        Tbounds = TBOUNDS

    template_text = read_from("./GGchem/input/model_Venus_template.in")
    template_lines = template_text.split("\n")
    filetext = ""
    filetext += "\n".join(template_lines[:11])
    filetext += "\n" + abund_file
    filetext += "\n".join(template_lines[12:28])
    filetext += "\n" + str(Npoints) + "\t\t! Npoints"
    filetext += "\n" + str(Tbounds[0]) + "\t\t! Tmin [K]"
    filetext += "\n" + str(Tbounds[1]) + "\t\t! Tmax [K]"
    filetext += "\n" + str(p) + "\t! pmin [bar]"
    filetext += "\n" + str(p) + "\t! pmax [bar]" + "\n"

    overwrite_to("./GGchem/input/grid_line_p.in", filetext)


def run_ggchem_gridline():
    """
    Runs GGchem on the input file `grid_line_p.in`
    """
    os.system("cd ./GGchem && ./ggchem input/grid_line_p.in > /dev/null && cd ..")


def run_ggchem_grid(pbounds=None, Tbounds=None, Npoints=NPOINTS, abund_file=ABUND_FILE):
    """
    Repeatedly runs `run_ggchem_gridline()` to find the
    abundances over the entire pT grid
    """
    if Tbounds is None:
        Tbounds = TBOUNDS
    if pbounds is None:
        pbounds = PBOUNDS
    results_file = "/data/ajnb3/results/summer/venus_grid.txt"
    overwrite_to(results_file, "")  # Initialises if doesn't exist

    pressures = np.logspace(np.log10(pbounds[0]), np.log10(pbounds[1]), Npoints)

    for p in tqdm(pressures, total=len(pressures)):
        create_GGchem_file_pT(
            p, Tbounds=Tbounds, Npoints=Npoints, abund_file=abund_file
        )

        run_ggchem_gridline()  # Runs GGchem at pressure p and T between Tbounds

        write_to(results_file, read_from("./GGchem/Static_Conc.dat"))


# Managing concatenated .dat files


def create_ggchem_results_df(results_file):
    """
    Creates a pandas df from the concatenated results file
    """
    filetext = read_from(results_file)
    # Extracting param numbers, ensuring that they're the same on each line
    grid_line_data = filetext.split("eps( H)")[1:]
    param_lists = np.array(
        [
            [int(param) for param in line_data.split("\n")[1].split()]
            for line_data in grid_line_data
        ]
    )
    params = param_lists[0]
    assert np.all(param_lists == params)

    # Extracting gas species names, ensuring that they're also the same
    header_list = np.array(
        [[line_data.split("\n")[2].split()] for line_data in grid_line_data]
    )
    assert np.all(header_list == header_list[0])

    df_string = grid_line_data[0] + "\n".join(
        ["\n".join(line_data.split("\n")[3:]) for line_data in grid_line_data[1:]]
    )  # First gridline plus numbers from rest
    df_raw = pd.read_csv(io.StringIO(df_string), skiprows=2, delim_whitespace=True)

    ## Processing df, removing absent species etc
    # Removing supersaturation ratios, element ratios
    column_mask = np.full_like(df_raw.columns, False)
    column_mask[0] = True
    column_mask[2] = True
    column_mask[4 : 4 + params[0] + params[1]] = True
    column_mask[
        4
        + params[0]
        + params[1]
        + params[2] : 4
        + params[0]
        + params[1]
        + 2 * params[2]
    ] = True

    df_masked = df_raw.iloc[:, column_mask]
    # Removing absent species
    df_masked = df_masked.loc[:, (df_masked != -300.0).any()]
    # Converting to VMRs and MFs
    gas_data = df_masked.iloc[:, 2 : 2 + params[0] + params[1]]
    cond_data = df_masked.iloc[:, 2 + params[0] + params[1] : 2 + sum(params)]
    gas_df = (10**gas_data).div((10**gas_data).sum(axis=1), axis=0)
    cond_df = (10**cond_data).div((10**cond_data).sum(axis=1), axis=0)

    df_processed = df_masked.copy()
    df_processed.loc[gas_df.index, gas_df.columns] = gas_df
    df_processed.loc[cond_df.index, cond_df.columns] = cond_df

    # Non-crucial relabelling/conversions of temperature/pressure columns
    df_processed.pgas = df_processed.pgas.div(1e6)  # Converting to bar
    df_processed = df_processed.rename(columns={"Tg": "T_K", "pgas": "p_bar"})

    return df_processed
