"""
Basic utilities
"""
from collections import namedtuple
import io
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
mpl.style.use("~/xbyrne.mplstyle")


## ------------------------------------
## Reading + Writing files


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


## ------------------------------------
## GGchem tools


def correct_GGchem_names(ggchem_names):
    """
    Corrects GGchem chemical formulae
        which sometimes have capitalised elements like NA for some reason
    """
    capitalised_elements = ["Na", "Mg", "Al", "Si", "Cl", "Ca", "Ti", "Mn", "Fe"]

    def correct_name(name):
        """
        Corrects a single GGchem name
        """
        for cap_el in capitalised_elements:
            if cap_el.upper() in name:
                name = name.replace(cap_el.upper(), cap_el)
        return name

    if isinstance(ggchem_names, str):
        return correct_name(ggchem_names)

    if isinstance(ggchem_names, np.ndarray):
        correct_names = np.copy(ggchem_names)
        for i, ggchem_name in enumerate(ggchem_names):
            correct_names[i] = correct_name(ggchem_name)
        return correct_names


def create_GGchem_file_pT(
    filename="./GGchem/input/grid_line_p.in",
    pbounds=None,
    Tbounds=None,
    Npoints=100,
    abund_file="abund_venus.in",
):
    """
    Creates the GGchem input file for a particular p,
    for a given temperature range, precision, and
    set of elemental abundances
    """
    if Tbounds is None:
        Tbounds = [650, 2000]
    if pbounds is None:
        pbounds = [0.1, 1e4]

    template_text = read_from("./GGchem/input/model_Venus_template.in")
    template_lines = template_text.split("\n")
    filetext = ""
    filetext += "\n".join(template_lines[:11])
    filetext += "\n" + abund_file
    filetext += "\n".join(template_lines[12:28])
    filetext += "\n" + str(Npoints) + "\t\t! Npoints"
    filetext += "\n" + str(Tbounds[0]) + "\t\t! Tmin [K]"
    filetext += "\n" + str(Tbounds[1]) + "\t\t! Tmax [K]"
    filetext += "\n" + str(pbounds[0]) + "\t! pmin [bar]"
    filetext += "\n" + str(pbounds[1]) + "\t! pmax [bar]" + "\n"

    overwrite_to(filename, filetext)


def run_ggchem_gridline():
    """
    Runs GGchem on the input file `grid_line_p.in`
    """
    os.system("cd ./GGchem && ./ggchem input/grid_line_p.in > /dev/null && cd ..")


def run_ggchem_grid(
    results_file, abund_file="abund_venus.in", pbounds=None, Tbounds=None, Npoints=100
):
    """
    Repeatedly runs `run_ggchem_gridline()` to find the
    abundances over the entire pT grid
    Saves to `results_file`
    """
    if Tbounds is None:
        Tbounds = [650, 2000]
    if pbounds is None:
        pbounds = [0.1, 1e4]
    overwrite_to(results_file, "")  # Initialises if doesn't exist

    pressures = np.logspace(np.log10(pbounds[0]), np.log10(pbounds[1]), Npoints)

    for p in tqdm(pressures, total=len(pressures)):
        create_GGchem_file_pT(
            p, Tbounds=Tbounds, Npoints=Npoints, abund_file=abund_file
        )

        run_ggchem_gridline()  # Runs GGchem at pressure p and T between Tbounds

        write_to(results_file, read_from("./GGchem/Static_Conc.dat"))


# Managing resulting .dat files


def gather_GGchem_results(results_file="./GGchem/Static_Conc.dat"):
    """
    Extracts arrays of temperature, pressure, gas VMRs and names,
     from GGchem's output file.
    Also calculates condensate number densities and names,
     if it detects that condensation was switched on.
    """
    GGchem_df = pd.read_csv(results_file, skiprows=2, delim_whitespace=True)
    N_elements, N_molecules, N_condensates, _ = [
        int(par) for par in read_from(results_file).split("\n")[1].split()
    ]
    column_mask = np.full_like(GGchem_df.columns, False)
    column_mask[0] = True
    column_mask[2] = True
    column_mask[4 : 4 + N_elements + N_molecules] = True
    column_mask[
        4
        + N_elements
        + N_molecules
        + N_condensates : 4
        + N_elements
        + N_molecules
        + 2 * N_condensates
    ] = True

    df_masked = GGchem_df.iloc[:, column_mask]
    # Removing masked minerals
    df_masked = df_masked.loc[:, (df_masked != -300.0).any()]
    # Converting to VMRs and MFs
    gas_data = df_masked.iloc[:, 2 : 2 + N_elements + N_molecules]
    cond_data = df_masked.iloc[
        :, 2 + N_elements + N_molecules : 2 + N_elements + N_molecules + N_condensates
    ]
    gas_df = (10**gas_data).div((10**gas_data).sum(axis=1), axis=0)
    cond_df = (10**cond_data).div((10**cond_data).sum(axis=1), axis=0)

    df_processed = df_masked.copy()
    df_processed.loc[gas_df.index, gas_df.columns] = gas_df
    df_processed.loc[cond_df.index, cond_df.columns] = cond_df
    # Flooring <1e-300s
    df_processed[df_processed < 1e-300] = 0
    # Non-crucial relabelling/conversions of temperature/pressure columns
    df_processed.pgas = df_processed.pgas.div(1e6)  # Converting to bar
    df_processed = df_processed.rename(columns={"Tg": "T_K", "pgas": "p_bar"})

    return df_processed


def create_ggchem_results_df(results_file):
    """
    Creates a pandas df from the concatenated results file
    created by `run_ggchem_grid`
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
    # Flooring <1e-300s
    df_processed[df_processed < 1e-300] = 0
    # Non-crucial relabelling/conversions of temperature/pressure columns
    df_processed.pgas = df_processed.pgas.div(1e6)  # Converting to bar
    df_processed = df_processed.rename(columns={"Tg": "T_K", "pgas": "p_bar"})

    return df_processed


def df_from_abund(abund_code, **kwargs):
    """
    Gets a df of all the elemental abundances in `./GGchem/{abund_code}.in`
    Kwargs are passed to pd.read_csv
    """
    return pd.read_csv(
        io.StringIO(read_from(f"./GGchem/{abund_code}.in")),
        delim_whitespace=True,
        names=["Element", "epsilon"],
        index_col=0,
        **kwargs,
    )


def df_to_abund(df, abund_code, **kwargs):
    """
    Writes a df of elemental abundances to `./GGchem/{abund_code}.in`
    Kwargs are passed to df.to_csv
    """
    df.to_csv(f"./GGchem/{abund_code}.in", sep=" ", header=False, **kwargs)


## ------------------------------------
## petitRADTRANS tools

Molecule = namedtuple("Molecule", "GGchem_name pRT_filename mmw")

CO2 = Molecule("CO2", "CO2", 44.01)
SO2 = Molecule("SO2", "SO2", 64.066)
O2 = Molecule("O2", "O2", 31.999)
SO3 = Molecule("SO3", "SO3", 80.06)
NO = Molecule("NO", "NO", 30.01)
H2O = Molecule("H2O", "H2O_HITEMP", 18.01528)
CO = Molecule("CO", "CO_all_iso_HITEMP", 28.01)
O = Molecule("O", "O", 15.999)
OH = Molecule("OH", "OH", 17.01)
NaCl = Molecule("NACL", "NaCl", 58.44)
KCl = Molecule("KCL", "KCl", 74.55)
KF = Molecule("KF", "KF", 58.10)
NaF = Molecule("NAF", "NaF", 41.99)
Na = Molecule("Na", "Na_allard", 22.99)
NaOH = Molecule("NAOH", "NaOH", 40.00)
O3 = Molecule("O3", "O3", 48.00)
K = Molecule("K", "K_allard", 39.098)
N2 = Molecule("N2", "N2", 28.02)

ALL_MOLECULES = [
    CO2,
    SO2,
    O2,
    SO3,
    H2O,
    CO,
    O,
    OH,
    NaCl,
    KCl,
    KF,
    NaF,
    Na,
    O3,
    K,
    N2,
    NO,
    NaOH,
]
all_molecules_GGchem_names = np.array(
    [molecule.GGchem_name for molecule in ALL_MOLECULES]
)
all_molecules_mmws = np.array([molecule.mmw for molecule in ALL_MOLECULES])

RADIATIVE_MOLECULES = [
    CO2,
    SO2,
    O2,
    SO3,
    H2O,
    CO,
    O,
    OH,
    NaCl,
    KCl,
    KF,
    NaF,
    Na,
    O3,
    K,
    # NO, NaOH [something wrong with the opacity files]
]

RAYLEIGH_MOLECULES = [CO2, H2O, O2, N2, CO]

CONTINUUM_OPACITIES = [
    "N2-N2",
    "O2-O2",
    "N2-O2",
    "CO2-CO2",
]


def calc_MMW(molecules_GGchem_names, VMRs):
    """
    Calculates the overall MMW of a gas, given the
    mixing ratios of the major components.

    Accepts a list of the VMRs of gases, along with a list of their names.
    This list must include all of the gases in all_molecules above
    """
    if len(VMRs) != len(molecules_GGchem_names):
        raise IndexError(
            f"{len(VMRs)} VMRs given, but {len(molecules_GGchem_names)} names."
        )

    sum_VMRi_mui = sum(
        VMRs[molecules_GGchem_names == mol_name] * mol_mmw
        for mol_name, mol_mmw in zip(all_molecules_GGchem_names, all_molecules_mmws)
    )
    sum_VMRi = sum(
        VMRs[molecules_GGchem_names == mol_name]
        for mol_name in all_molecules_GGchem_names
    )

    mmw = sum_VMRi_mui / sum_VMRi
    return mmw


## -----------------------
## Consistent plotting tools
def plot_spectra(wavelengths_um, spectra, cols, labels=None):
    """
    Plots a set of spectra in my standard format,
    with inputted colours (and optionally labels)
    """
    fg, ax = plt.subplots(figsize=(15, 5))
    if labels:
        for spectrum, col, label in zip(spectra, cols, labels):
            ax.plot(wavelengths_um, spectrum, c=col, label=label)
        ax.legend()
    else:
        for spectrum, col in zip(spectra, cols):
            ax.plot(wavelengths_um, spectrum, c=col)
    ax.set_xlim([0.3, 15])
    ax.set_xscale("log")
    xtix = [0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14]
    ax.set_xticks(xtix)
    ax.set_xticklabels(xtix)
    ax.set_xlabel(r"$\lambda/\mu$m")
    ax.set_ylabel(r"Transit Radius / $R_J$")
    return fg


def plot_grid(df, gas=None, cond=None, **kwargs):
    """
    Plots abundance grids. Don't know why I didn't define a function for this earlier
    """
    grid_size = int(np.sqrt(len(df)))

    def squarsh(df, key):
        return df[key].to_numpy().reshape(grid_size, grid_size)

    if gas:
        field = gas
        cmp = cm.Greens
    elif cond:
        field = f"n{cond}"
        cmp = cm.Oranges

    fg, ax = plt.subplots()
    contf = ax.pcolormesh(
        squarsh(df, "T_K"),
        squarsh(df, "p_bar"),
        squarsh(df, field),
        vmin=0,
        vmax=np.max(df[field]),
        cmap=cmp,
        **kwargs,
    )
    ax.set_yscale("log")
    fg.colorbar(contf, ax=ax)
    return fg
