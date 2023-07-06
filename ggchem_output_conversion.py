"""
Converts GGchem output into VMRs and MFs
"""
import numpy as np
import pandas as pd
from mymodules import myutils

# OUTPUT_FILE = '/data/ajnb3/results/grid/grid_output.dat'
OUTPUT_FILE = '/data/ajnb3/results/grid/grid_output_moreO.dat'
# OUTPUT_FILE = '/data/ajnb3/results/grid/grid_output_lessO.dat'

print('Importing GGchem results')
# Importing results file metadata
params = myutils.read_from(OUTPUT_FILE).split("\n")[1].split()
N_elements = int(params[0])
N_molecules = int(params[1])
N_condensates = int(params[2])

# Importing abundance data from results file
# Gases are in cm^-3
# Condensates are in concentration relative to H (I think)
df = pd.read_csv(OUTPUT_FILE, skiprows=2, delim_whitespace=True)

N_P = 100  # number of points along p axis
N_T = 100  # number of points along T axis
T_K = np.array(df["Tg"]).reshape((N_P, N_T))  # Tg column is in K
p_bar = (
    np.array(df["pgas"]).reshape((N_P, N_T)) / 1e6
)  # pgas column is in ubar for some reason

print('Calculating VMRs')
## Packaging Gas species
gas_species_names = np.array(
    df.columns[3 : 4 + N_elements + N_molecules]
)  # Names of gas species (inc. elemental species)
df_mol = df[gas_species_names]
gas_species_cm3 = np.array(df_mol).reshape(
    (
        N_P,
        N_T,
        1 + N_elements + N_molecules,
    )  # 1+ for the electrons, which are actually absent but oh well
)
gas_species_cm3 = np.rollaxis(gas_species_cm3, 2)  # put species axis first
gas_species_mr = 10**gas_species_cm3 / np.sum(
    10**gas_species_cm3, axis=0
)  # Converting to VMR

## Packaging Condensates
condensate_colnames = np.array(
    df.columns[
        4
        + N_elements
        + N_molecules
        + N_condensates : 4
        + N_elements
        + N_molecules
        + 2 * N_condensates
    ]  # Skipping supersaturation ratios
)
df_cond = df[condensate_colnames]
condensates_orig_units = np.array(df_cond).reshape((N_P, N_T, N_condensates))
condensates_orig_units = np.rollaxis(condensates_orig_units, 2)  # species axis first
condensates_mf = 10**condensates_orig_units / np.sum(
    10**condensates_orig_units, axis=0
)  # Converting to MF

## Removing absentees
# Gases
gas_species_abundance_cutoff = (
    -300
)  # Gas species must have n>={gas_species_abundance_cutoff} to be included here
is_above_cutoff = np.any(
    gas_species_cm3 > gas_species_abundance_cutoff, axis=(1, 2)
)  # Will be all except 'el'
gas_species_above_cutoff = gas_species_names[is_above_cutoff]  # Masking out absentees
gas_species_present_mr = gas_species_mr[is_above_cutoff, :, :]

# Condensates
condensate_abundance_cutoff = (
    -300
)  # Changing this up to as high as -3 doesn't change anything
is_above_cutoff = np.any(
    condensates_orig_units > condensate_abundance_cutoff, axis=(1, 2)
)  # 29 mineral species present
condensates_above_cutoff = np.array(
    [cn[1:] for cn in condensate_colnames[is_above_cutoff]]  # Masking out absentees
)
condensates_present_mf = condensates_mf[is_above_cutoff, :, :]

## Saving data
# npz_file = '/data/ajnb3/results/grid/grid_results.npz'
npz_file = '/data/ajnb3/results/grid/grid_results_moreO.npz'
# npz_file = '/data/ajnb3/results/grid/grid_results_lessO.npz'
np.savez_compressed(
    npz_file,
    T_K=T_K,
    p_bar=p_bar,
    gas_species_names=gas_species_above_cutoff,
    gas_species_mr=gas_species_present_mr,
    condensates_names=condensates_above_cutoff,
    condensates_mf=condensates_present_mf,
)
