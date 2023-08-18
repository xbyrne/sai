"""
Basic utilities
"""
import io
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from chempy import Substance
import matplotlib.pyplot as plt
import matplotlib as mpl

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


# Utils for managing elemental abundance files


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


def create_GGchem_input_file(
    filename="grid_line_p.in",
    pbounds=None,
    Tbounds=None,
    Npoints=100,
    abund_code="abund_venus",
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
    filetext += "\n" + f"{abund_code}.in"
    filetext += "\n".join(template_lines[12:28])
    filetext += "\n" + str(Npoints) + "\t\t! Npoints"
    filetext += "\n" + str(Tbounds[0]) + "\t\t! Tmin [K]"
    filetext += "\n" + str(Tbounds[1]) + "\t\t! Tmax [K]"
    filetext += "\n" + str(pbounds[0]) + "\t! pmin [bar]"
    filetext += "\n" + str(pbounds[1]) + "\t! pmax [bar]" + "\n"

    overwrite_to(f"./GGchem/input/{filename}", filetext)


# Managing resulting .dat files


def get_params_from_df(df):
    """
    Extracts number of gas and mineral species from a df
    Returns `N_gases` and `N_conds`
    """
    gas_cols = [col for col in df.columns if col[0] != "n"][
        3:  # Excluding unnamed column, T_K, p_bar
    ]
    cond_cols = [col for col in df.columns if col[0]][1:]  # Excluding nHtot
    return len(gas_cols), len(cond_cols)


def get_params(results_file="./GGchem/Static_Conc.dat"):
    """
    Extracts number of gas and mineral species from `Static_Conc.dat`
    Returns `N_gases` and `N_conds`
    """
    filetext = read_from(results_file)
    params = [int(param) for param in filetext.split("\n")[1].split()]
    N_gases = params[0] + params[1]
    N_conds = params[2]
    return N_gases, N_conds


def gather_GGchem_results(results_file="./GGchem/Static_Conc.dat"):
    """
    Gathers number densities of gas and condensate species from GGchem's output file.
    In each case, number densities are given as log10(n / cm-3)
    Removes supersaturation ratios, epsilons, and other various
    """
    GGchem_df = pd.read_csv(results_file, skiprows=2, delim_whitespace=True)
    N_gases, N_conds = get_params(results_file)
    column_mask = np.full_like(GGchem_df.columns, False)
    column_mask[0:3] = True
    column_mask[4 : 4 + N_gases] = True
    column_mask[4 + N_gases + N_conds : 4 + N_gases + 2 * N_conds] = True

    df_masked = GGchem_df.iloc[:, column_mask]
    # Removing absent minerals
    df_masked = df_masked.loc[:, (df_masked != -300.0).any()]
    # Converting condensate stats to log(n/cm-3)
    df_masked.iloc[:, 3 + N_gases :] = df_masked.iloc[:, 3 + N_gases :].add(
        np.log10(df_masked.nHges), axis=0
    )
    # Fixing chemical names, relabelling columns, putting p in bars bc it's annoying
    df_masked.rename(columns=correct_GGchem_names, inplace=True)
    df_masked.pgas = df_masked.pgas.div(1e6)
    df_masked.rename(
        columns={"Tg": "T_K", "pgas": "p_bar", "nHges": "nHtot"}, inplace=True
    )

    return df_masked


def CaO_mass_fraction(df):
    """
    Produces a ``CaO solid mass fraction`` column for data in a given df
    Assumes mineral abundances are stored logarithmically
    and all the mineral columns start with `n`, as GGchem gives them.
    """
    mineral_df = df[
        [header for header in df.columns if header[0] == "n" and header != "nHtot"]
    ]
    mineral_df = mineral_df.rename(columns=lambda header: header[1:])
    CaO_mmw = Substance.from_formula("CaO").mass
    CaO_mass = np.zeros_like(mineral_df.index, dtype=np.float64)
    total_mass = np.zeros_like(mineral_df.index, dtype=np.float64)
    for mineral_formula in mineral_df.columns:
        species = Substance.from_formula(mineral_formula)
        if 20 in species.composition:
            CaO_mass += (
                10 ** mineral_df[mineral_formula]
                * species.composition[20]
                * 1
                * CaO_mmw
            )
        total_mass += 10 ** mineral_df[mineral_formula] * species.mass
    return CaO_mass / total_mass


def MgO_mass_fraction(df):
    """
    Produces a ``MgO solid mass fraction`` column for data in a given df
    Assumes mineral abundances are stored logarithmically
    and all the mineral columns start with `n`, as GGchem gives them.
    """
    mineral_df = df[
        [header for header in df.columns if header[0] == "n" and header != "nHtot"]
    ]
    mineral_df = mineral_df.rename(columns=lambda header: header[1:])
    MgO_mmw = Substance.from_formula("MgO").mass
    MgO_mass = np.zeros_like(mineral_df.index, dtype=np.float64)
    total_mass = np.zeros_like(mineral_df.index, dtype=np.float64)
    for mineral_formula in mineral_df.columns:
        species = Substance.from_formula(mineral_formula)
        if 12 in species.composition:
            MgO_mass += (
                10 ** mineral_df[mineral_formula]
                * species.composition[12]
                * 1  # 1/Stoichiometric coeff of Mg in MgO
                * MgO_mmw
            )
        total_mass += 10 ** mineral_df[mineral_formula] * species.mass
    return MgO_mass / total_mass


def Al2O3_mass_fraction(df):
    """
    Produces a ``Al2O3 solid mass fraction``` column for data in a given df
    Assumes mineral abundances are stored logarithmically
    and all the mineral columns start with `n` as GGchem gives them
    """
    mineral_df = df[
        [header for header in df.columns if header[0] == "n" and header != "nHtot"]
    ]
    mineral_df = mineral_df.rename(columns=lambda header: header[1:])
    Al2O3_mmw = Substance.from_formula("Al2O3").mass
    Al2O3_mass = np.zeros_like(mineral_df.index, dtype=np.float64)
    total_mass = np.zeros_like(mineral_df.index, dtype=np.float64)
    for mineral_formula in mineral_df.columns:
        species = Substance.from_formula(mineral_formula)
        if 13 in species.composition:
            Al2O3_mass += (
                10 ** mineral_df[mineral_formula]
                * species.composition[13]
                * 0.5  # 1/Stoichiometric coeff of Al in Al2O3
                * Al2O3_mmw
            )
        total_mass += 10 ** mineral_df[mineral_formula] * species.mass
    return Al2O3_mass / total_mass


def SiO2_mass_fraction(df):
    """
    Produces an ``SiO2 solid mass fraction``` column for data in a given df
    Assumes mineral abundances are stored logarithmically
    and all the mineral columns start with `n` as GGchem gives them
    """
    mineral_df = df[
        [header for header in df.columns if header[0] == "n" and header != "nHtot"]
    ]
    mineral_df = mineral_df.rename(columns=lambda header: header[1:])
    SiO2_mmw = Substance.from_formula("SiO2").mass
    SiO2_mass = np.zeros_like(mineral_df.index, dtype=np.float64)
    total_mass = np.zeros_like(mineral_df.index, dtype=np.float64)
    for mineral_formula in mineral_df.columns:
        species = Substance.from_formula(mineral_formula)
        if 14 in species.composition:
            SiO2_mass += (
                10 ** mineral_df[mineral_formula]
                * species.composition[14]
                * 1  # 1/Stoichiometric coeff of Si in SiO2
                * SiO2_mmw
            )
        total_mass += 10 ** mineral_df[mineral_formula] * species.mass
    return SiO2_mass / total_mass


def fractionate(df, element):
    """
    Returns a df containing only the species with the specified element in it,
    and with the fields being the fraction of the atoms of that element which are
    in each of those species
    """
    atomic_number = list(Substance.from_formula(element).composition.keys())[0]
    element_df = df[
        [
            col
            for col in df.columns[4:]
            if atomic_number
            in Substance.from_formula(col[1:] if col[0] == "n" else col).composition
        ]
    ]
    stoic_element = [stoic(col, atomic_number) for col in element_df.columns]
    element_cm3_df = 10**element_df * stoic_element
    return element_cm3_df.div(element_cm3_df.sum(axis=1), axis=0)


def VMR(df, gas_species):
    """
    Returns a series containing the VMRs of the given gas species
    for each row in the df
    """
    gas_cm3 = 10 ** df[[col for col in df.columns if col[0] != "n"][3:]]
    return gas_cm3[gas_species].div(gas_cm3.sum(axis=1), axis=0)


def MF(df, mineral="CaSO4"):
    """
    Returns a series containing the mole fraction of the specified mineral
    (default: CaSO4) among the condensates for each row in the df
    """
    mineral_cm3 = 10 ** df[[col for col in df.columns if col[0] == "n"][1:]]
    return mineral_cm3[f"n{mineral}"].div(mineral_cm3.sum(axis=1), axis=0)


def log_f(df, gas_species):
    """
    Returns a series containing the log(fugacity/bar) of the given
    gas species for each row in the df
    """
    vmr = VMR(df, gas_species)
    fugacity = vmr * df.p_bar
    return np.log10(fugacity)


## ------------------------------------
## Chemical tools
def mr(species_name):
    """
    Returns the molecular weight of a single species using chempy
    Given in g/mol
    """
    return Substance.from_formula(species_name).mass


def stoic(species_name, element_Ar):
    """
    Returns the number of the element with atomic number `element_Ar` in `species_name`
    Crops leading `n`s from name, to allow for GGchem condensate namings
    """
    if species_name[0] == "n":
        species_name = species_name[1:]
    try:
        return Substance.from_formula(species_name).composition[element_Ar]
    except KeyError:
        return 0


def unicodify(raw_formula_string):
    """Converts chemical formula into unicode, allowing subscripts"""
    return Substance.from_formula(raw_formula_string).unicode_name


def chemlatex(raw_formula_string):
    """Converts chemical formula into latex string; chempy does it slightly wrong"""
    chempy_string = Substance.from_formula(raw_formula_string).latex_name
    return chempy_string.replace("_", "$_").replace("}", "}$")


## -----------------------
## Consistent plotting tools


def squarsh(df, key, grid_size=100):
    """
    Squashes a df into a square and returns a np array
    """
    df2 = df.copy()
    return df2[key].to_numpy().reshape(grid_size, grid_size)


def atm_demo(dfs, title_list=None, figheight=None):
    """
    For a set of GGchem results dfs from 0.1 to 1e2bar, creates a figure showing
    the atmospheric composition along the transect
    """
    if title_list is None:
        title_list=[''] * len(dfs)
    if figheight is None:
        figheight = 2.5 * len(dfs)

    col_dict = {"N2": "#aaaaaa", "O2": "#7570b3", "CO2": "#e7298a", "SO2": "#66a61e", "SO3": "#1b9e77"}

    fg, axs = plt.subplots(
        len(dfs), 1, figsize=(7, figheight), gridspec_kw={"hspace": 0}
    )
    for j, ax in enumerate(axs):
        df = dfs[j]
        vmr_srs = [VMR(df, gas) for gas in col_dict]
        ax.stackplot(
            df.p_bar, vmr_srs, labels=col_dict.keys(), colors=col_dict.values()
        )
        ax2 = ax.twinx()
        ax2.plot(
            df.p_bar, MF(df, 'CaSO4'), 'k'
        )
        # ax2.plot(
        #     df.p_bar, MF(df, 'CaMgSi2O6'), 'white'
        # )
        ax2.set_xscale('log')
        ax2.set_ylim(0,.13)

        ax.set_xscale("log")
        ax.set_xlim(.1,1e2)
        ax.set_ylim(0, 1)
        ax.set_title(title_list[j], y=0.7, x=0.035, loc="left")

        ax.set_xlabel(r"$p_0$ / bar", fontsize=24)
        if ax == axs[-1]:
            ax.set_yticks([0.0, 0.5, 1.0])
            leg = ax.legend(fontsize=15, loc="lower right", reverse=True)
            leg.remove()
            ax2.add_artist(leg)
            ax.tick_params(axis='x', which='major', pad=10)
        else:
            ax.set_xticklabels([])
            ax.set_yticks([0.5, 1.0])
        ax.set_yticks([.1,.2,.3,.4,.6,.7,.8,.9], minor=True)
        ax2.set_yticks([.01, .02,.03,.04,.06,.07,.08,.09,.1,.11,.12,.13],
                       minor=True)
        ax.tick_params(which='major', axis='x', length=5)
        ax.tick_params(which='minor', axis='x', length=3)
    
    fg.supylabel("VMRs", fontsize=26, x=-.01)
    fg.text(1.02, .5, f'Mole Fraction of {chemlatex("CaSO4")}', rotation=90, verticalalignment='center', fontsize=24)

    return fg


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


## ------------------------------------
## petitRADTRANS tools

pRT_filename_dict = {
    "CO2": "CO2",
    "SO2": "SO2",
    "O2": "O2",
    "SO3": "SO3",
    "NO": "NO",
    "H2O": "H2O_HITEMP",
    "CO": "CO_all_iso_HITEMP",
    "O": "O",
    "OH": "OH",
    "NaCl": "NaCl",
    "KCl": "KCl",
    "KF": "KF",
    "NaF": "NaF",
    "Na": "Na_allard",
    "NaOH": "NaOH",
    "O3": "O3",
    "K": "K_allard",
    "N2": "N2",
}

RADIATIVE_MOLECULES = [
    "CO2",
    "SO2",
    "O2",
    "SO3",
    "H2O",
    "CO",
    "O",
    "OH",
    "NaCl",
    "KCl",
    "KF",
    "NaF",
    "Na",
    "O3",
    "K",
    # "NO", "NaOH" [something wrong with the opacity files]
]

RAYLEIGH_MOLECULES = ["CO2", "H2O", "O2", "N2", "CO"]

CONTINUUM_OPACITIES = [
    "N2-N2",
    "O2-O2",
    "N2-O2",
    "CO2-CO2",
]


## Utils for running a very specific grid thing I needed once


def run_ggchem_gridline():
    """
    Runs GGchem on the input file `grid_line_p.in`
    """
    os.system("cd ./GGchem && ./ggchem input/grid_line_p.in > /dev/null && cd ..")


def run_ggchem_grid(
    results_file, abund_code="abund_venus", pbounds=None, Tbounds=None, Npoints=100
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
        create_GGchem_input_file(
            p, Tbounds=Tbounds, Npoints=Npoints, abund_code=abund_code
        )

        run_ggchem_gridline()  # Runs GGchem at pressure p and T between Tbounds

        write_to(results_file, read_from("./GGchem/Static_Conc.dat"))


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
