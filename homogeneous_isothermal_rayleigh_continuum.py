"""
Calculates spectra for homogeneous isothermal atmospheres,
this time with Rayleigh scattering and continuum emission
included
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
import myutils

# -----------------------------------------
# Loading and dealing out dat
surface_df = pd.read_csv("/data/ajnb3/results/summer/venus_1400.csv")
gas_df = surface_df[[col for col in surface_df.columns if col[0] != "n"][3:]]
vmr_df = (10**gas_df).div((10**gas_df).sum(axis=1), axis=0)
rel_mass_df = vmr_df * [myutils.mr(col) for col in vmr_df.columns]
mmw_srs = rel_mass_df.sum(axis=1)
mass_frac_df = rel_mass_df.div(mmw_srs, axis=0)

atmosphere = Radtrans(
    line_species=[
        myutils.pRT_filename_dict[mol] for mol in myutils.RADIATIVE_MOLECULES
    ],
    rayleigh_species=[
        myutils.pRT_filename_dict[mol] for mol in myutils.RAYLEIGH_MOLECULES
    ],
    continuum_opacities=[
        "N2-N2",
        "O2-O2",
        "N2-O2",
        "CO2-CO2",
    ],
    wlen_bords_micron=[0.3, 15],
)

R_VENUS = 0.95 * nc.r_earth
GRAVITY = 10**2.94
pRT_folders = myutils.pRT_filename_dict

print("Beginning calculations")
for i, (p, mmw, (_, mass_fracs)) in tqdm(
    enumerate(zip(surface_df.p_bar, mmw_srs, mass_frac_df.iterrows())), total=100
):
    atm_pressures = np.logspace(-8, np.log10(p), 100)
    atm_temperatures = 1400.0 * np.ones_like(atm_pressures)
    atm_MMWs = mmw * np.ones_like(atm_pressures)
    mass_fractions = {
        mol: mf * np.ones_like(atm_pressures)
        for mol, mf in zip(pRT_folders.values(), mass_fracs[pRT_folders.keys()])
    }

    atmosphere.setup_opa_structure(atm_pressures)
    atmosphere.calc_transm(
        atm_temperatures, mass_fractions, GRAVITY, atm_MMWs, R_pl=R_VENUS, P0_bar=p
    )

    if i == 0:
        wavelengths_um = nc.c / atmosphere.freq / 1e-4
        transit_radius_RJ_grid = np.zeros(wavelengths_um.shape + surface_df.p_bar.shape)

    transit_radius_RJ = atmosphere.transm_rad / nc.r_jup_mean
    transit_radius_RJ_grid[:, i] = transit_radius_RJ

spectra_df = pd.DataFrame(data=transit_radius_RJ_grid.T, columns=wavelengths_um)

print("Saving Data")
spectra_df.to_csv("/data/ajnb3/results/summer/spectra_1400.csv")
print("Done")
