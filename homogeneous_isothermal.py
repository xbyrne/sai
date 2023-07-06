"""
Calculates atmospheres using homogeneous isothermal models
"""

import numpy as np
from tqdm import tqdm
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
from mymodules import myutils

# -----------------------------------------
# Loading and dealing out data
print("Loading Data...")
DATA_DIR = '/data/ajnb3/results/grid'
fl = np.load(f"{DATA_DIR}/grid_results.npz", allow_pickle=True)
T_K = fl["T_K"]
p_bar = fl["p_bar"]
gas_species_names = fl["gas_species_names"]
gas_species_mr = fl["gas_species_mr"]

MMW_grid = myutils.calc_MMW(gas_species_names, gas_species_mr)

atmosphere = Radtrans(
    line_species=[mol.pRT_filename for mol in myutils.RADIATIVE_MOLECULES],
    #     rayleigh_species=["H2", "He"],
    #     continuum_opacities=["H2-H2", "H2-He"],
    wlen_bords_micron=[0.3, 15],
)

R_pl = 0.95 * nc.r_earth
GRAVITY = 10**2.94

print("Beginning calculations")
for i, j in tqdm(np.ndindex(T_K.shape), total=np.prod(T_K.shape)):
    T = T_K[i, j]
    p0 = p_bar[i, j]
    pressures = np.logspace(-6, np.log10(p0), 100)
    one = np.ones_like(pressures)
    temperature = T * one
    MMW = MMW_grid[i, j] * one

    atmosphere.setup_opa_structure(pressures)

    mass_fractions = {}
    for molecule in myutils.RADIATIVE_MOLECULES:
        mass_fractions[molecule.pRT_filename] = (
            gas_species_mr[gas_species_names == molecule.GGchem_name, :, :][0, i, j]
            * molecule.mmw
            / MMW
        )

    atmosphere.calc_transm(
        temperature, mass_fractions, GRAVITY, MMW, R_pl=R_pl, P0_bar=p0
    )

    if (i == 0) & (j == 0):
        wavelengths_um = nc.c / atmosphere.freq / 1e-4
        transit_radius_RJ_grid = np.zeros(wavelengths_um.shape + T_K.shape)

    transit_radius_RJ = atmosphere.transm_rad / nc.r_jup_mean
    transit_radius_RJ_grid[:, i, j] = transit_radius_RJ

print("Saving Data")
np.savez_compressed(
    "/data/ajnb3/results/petitRADTRANS/homogeneous_isothermal_spectra.npz",
    T_K=T_K,
    p_bar=p_bar,
    wavelengths_um=wavelengths_um,
    transit_radius_RJ=transit_radius_RJ_grid,
)
print("Done")
