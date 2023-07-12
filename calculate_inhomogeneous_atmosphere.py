"""
Calculating gas MRs for inhomogeneous atmospheres.
For a range of base pT conditions (and hence surface compositions), a
    pT profile is devised that tends to an isotherm of ~540K~ ~390K~ 335K
    (as might be experienced on TRAPPIST-~1b~ 1c).
GGchem is then run on the elemental composition corresponding to that 
    at the base of the atmosphere, assuming this elemental composition
    is uniform.
The compositional profile for each surface pressure is saved at the end
"""

import numpy as np
from tqdm import tqdm
from petitRADTRANS import Radtrans, nat_cst as nc
from . import myutils

## Loading grid abundances
print("Loading grid abundances...")
fl = np.load("/data/ajnb3/results/grid/grid_results.npz", allow_pickle=True)
T_K = fl["T_K"]
p_bar = fl["p_bar"]
gas_species_names = fl["gas_species_names"]
gas_species_mr = fl["gas_species_mr"]

## Initialising atmosphere
print("Initialising atmosphere...")
atmosphere = Radtrans(
    line_species=[mol.pRT_filename for mol in myutils.RADIATIVE_MOLECULES],
    rayleigh_species=[mol.pRT_filename for mol in myutils.RAYLEIGH_MOLECULES],
    continuum_opacities=myutils.CONTINUUM_OPACITIES,
    wlen_bords_micron=[0.3, 15],
)

GRAVITY = 10**2.94  # Venusian params
R_PLANET = 0.95 * nc.r_earth

def pRT_spectrum(p_profile, T_profile, gas_names, gas_mr_profile):
    """
    Uses petitRADTRANS to calculate the transmission spectrum
    for a given pT profile and gas composition profile
    """
    p_profile, T_profile = myutils.sort_pT_profile(p_profile, T_profile)
    mmw_profile = myutils.calc_MMW(gas_names, gas_mr_profile)
    atmosphere.setup_opa_structure(p_profile)
    mass_fractions = myutils.mass_frac_dict(gas_names, gas_mr_profile)

    atmosphere.calc_transm(
        T_profile,
        mass_fractions,
        GRAVITY,
        mmw_profile,
        R_pl=R_PLANET,
        P0_bar=np.max(p_profile),
    )
    wav_um = nc.c / atmosphere.freq / 1e-4
    tr_RJ = atmosphere.transm_rad / nc.r_jup_mean
    return wav_um, tr_RJ

# ISOTHERM_TEMPERATURE = 540  # Irradiation temperature at TRAPPIST-1b
# ISOTHERM_TEMPERATURE = 390  # Equilibrium temperature at TRAPPIST-1b
ISOTHERM_TEMPERATURE = 335 # Equilibrium temperature at TRAPPIST-1c
# grid_dims = p_bar.shape
j = 68
(i1,i2)=(20,44)
grid_dims = (i2-i1,)
# for i, j in tqdm(np.ndindex(grid_dims), total=np.prod(grid_dims)):
# for i,j in tqdm(np.ndindex((2,2))):
for i in tqdm(range(i1,i2)):
    p_base = p_bar[i, j]
    T_base = T_K[i, j]
    pressures, temperatures = myutils.isotherm_adiabat_stitcher(
        p_base, T_base, ISOTHERM_TEMPERATURE
    )
    base_gas_mrs = gas_species_mr[:, i, j]
    (
        pressure_profile,
        temperature_profile,
        new_gas_species_names,
        new_gas_species_mr,
        _,
        _,  # No condensation (see below)
    ) = myutils.run_GGchem(
        pressures,
        temperatures,
        gas_species_names,
        base_gas_mrs,
        False  # Running with condensates off to save time
        # (doesn't seem to affect atmospheric chemistry much)
    )
    if i == i1:
        global_gas_names = new_gas_species_names  # (~348,)
    if np.any(new_gas_species_names != global_gas_names):
        raise AssertionError(
            "Gas species in this atmosphere are different to in the first atm"
        )
    wavelength_um, transit_radius_RJ = pRT_spectrum(
        pressure_profile, temperature_profile,
        new_gas_species_names, new_gas_species_mr
    )
    if i == i1:
        pressure_profiles = np.zeros(grid_dims + (len(pressure_profile),))  # (100,100,100)
        temperature_profiles = np.zeros(grid_dims + (len(temperature_profile),))  # (100,100,100)
        gas_mr_profiles = np.zeros(
            grid_dims + new_gas_species_mr.shape  # (100,100,348,)
        )
        spectra = np.zeros(grid_dims + (len(transit_radius_RJ),))
    pressure_profiles[i-i1, :] = pressure_profile
    temperature_profiles[i-i1, :] = temperature_profile
    gas_mr_profiles[i-i1, :, :] = new_gas_species_mr
    spectra[i-i1,:] = transit_radius_RJ

np.savez_compressed(
#     "/data/ajnb3/results/inhomogeneous/grid_inhomogeneous_atms.npz",
    "/data/ajnb3/results/inhomogeneous/transect_inhomogeneous_atms_spectra_1c.npz",
#     p_bar=p_bar,
#     T_K=T_K,
    p_profiles=pressure_profiles,
    T_profiles=temperature_profiles,
    gas_species_names=global_gas_names,
    gas_mr_profiles=gas_mr_profiles,
    wavelength_um = wavelength_um,
    transit_radius_RJ = spectra
)
