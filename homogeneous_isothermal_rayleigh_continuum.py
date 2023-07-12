"""
Calculates spectra for homogeneous isothermal atmospheres,
this time with Rayleigh scattering and continuum emission
included
"""

import numpy as np
from tqdm import tqdm
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
from . import myutils

# -----------------------------------------
# Loading and dealing out data
print("Loading Data...")
fl = np.load("/data/ajnb3/results/grid/grid_results.npz", allow_pickle=True)
T_K = fl["T_K"]
p_bar = fl["p_bar"]
gas_species_names = fl["gas_species_names"]
gas_species_mr = fl["gas_species_mr"]
mmw = myutils.calc_MMW(gas_species_names, gas_species_mr)

# Selecting data only along trans-boundary transect
transect_3d = np.s_[:, 15:55, 68]
transect = transect_3d[1:]
gas_mr_transect = gas_species_mr[transect_3d]
T_transect = T_K[transect]
p_transect = p_bar[transect]
mmw_transect = mmw[transect]


atmosphere = Radtrans(
    line_species=[mol.pRT_filename for mol in myutils.RADIATIVE_MOLECULES],
    rayleigh_species=[mol.pRT_filename for mol in myutils.RAYLEIGH_MOLECULES],
    continuum_opacities=[
        "N2-N2",
        "O2-O2",
        "N2-O2",
        "CO2-CO2",
    ],
    wlen_bords_micron=[0.3, 15],
)

R_pl = 0.95 * nc.r_earth
GRAVITY = 10**2.94

print("Beginning calculations")
for i, (T, p, mmw, gmr) in tqdm(
    enumerate(zip(T_transect, p_transect, mmw_transect, gas_mr_transect.T)),
    total=len(T_transect),
):
    pressures = np.logspace(-8, np.log10(p), 100)
    one = np.ones_like(pressures)
    temperature = T * one
    MMW = mmw * one

    atmosphere.setup_opa_structure(pressures)

    mass_fractions = {}
    for molecule in myutils.ALL_MOLECULES:
        mass_fractions[molecule.pRT_filename] = (
            gmr[gas_species_names == molecule.GGchem_name][0] * molecule.mmw / mmw
        )

    atmosphere.calc_transm(
        temperature, mass_fractions, GRAVITY, MMW, R_pl=R_pl, P0_bar=p
    )

    if i == 0:
        wavelengths_um = nc.c / atmosphere.freq / 1e-4
        transit_radius_RJ_grid = np.zeros(wavelengths_um.shape + T_transect.shape)

    transit_radius_RJ = atmosphere.transm_rad / nc.r_jup_mean
    transit_radius_RJ_grid[:, i] = transit_radius_RJ

print("Saving Data")
np.savez_compressed(
    "/data/ajnb3/results/petitRADTRANS/homogeneous_isothermal_rayleigh_continuum_spectra.npz",
    T_K=T_transect,
    p_bar=p_transect,
    wavelengths_um=wavelengths_um,
    transit_radius_RJ=transit_radius_RJ_grid,
)
print("Done")
