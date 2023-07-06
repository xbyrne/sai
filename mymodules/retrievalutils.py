"""
Trying out retrievals
"""
import glob
import json
import numpy as np
from petitRADTRANS import nat_cst as nc
from petitRADTRANS.retrieval import Retrieval, RetrievalConfig
from petitRADTRANS.retrieval.util import calc_MMW
from . import myutils
from . import pandexoutils

RETRIEVAL_SPECIES = [
    myutils.CO2,
    myutils.SO3,
    myutils.SO2,
    myutils.O2,
    #     myutils.N2
]
RAYLEIGH_SPECIES = [
    myutils.CO2,
    #     myutils.N2,
    myutils.O2,
]


def run_retrieval(atm_code, N_transits):
    """
    Runs a retrieval for the atmosphere,
    and number of transits, selected
    ```atm_code```: either 'red' or 'blue'
    ```N_transits```: integer >= 1
    """

    RunDefinitionSimple = RetrievalConfig(
        retrieval_name=f"{atm_code}_{N_transits}_again", run_mode="retrieval", AMR=False
    )
    # Adding parameters - not sure how many of them do anything
    # RunDefinitionSimple.add_parameter("Rstar", False, value=0.1192 * nc.r_sun)
    # RunDefinitionSimple.add_parameter("log_g", False, value=2.94)
    # RunDefinitionSimple.add_parameter("R_pl", False, value=0.95 * nc.r_earth)
    # RunDefinitionSimple.add_parameter("Temperature", False, value=1407)
    # This one does do something! It will be varied by MultiNest
    RunDefinitionSimple.add_parameter(
        "logp0",
        True,
        transform_prior_cube_coordinate=lambda x: -1 + 5 * x,  # From -1 to 4
    )
    RunDefinitionSimple.add_parameter(
        "T0",
        True,
        transform_prior_cube_coordinate=lambda x: 650 + 1350 * x,  # From 650 to 2000
    )
    RunDefinitionSimple.add_parameter(
        "Tiso",
        True,
        transform_prior_cube_coordinate=lambda x: 100 + 550 * x,  # From 100 to 650
    )
    # Adding molecules
    RunDefinitionSimple.set_rayleigh_species(
        [mol.pRT_filename for mol in RAYLEIGH_SPECIES]
    )
    RunDefinitionSimple.set_continuum_opacities(
        [
            # "N2-N2", "N2-O2",
            "O2-O2",
            "CO2-CO2",
        ]
    )
    RunDefinitionSimple.set_line_species(
        [mol.pRT_filename for mol in RETRIEVAL_SPECIES], eq=False, abund_lim=(-6.0, 0.0)
    )

    # Adding data
    pandexoutils.create_spectrum_datafile(
        f"/home/ajnb3/sai/pandexo_files/{atm_code}.p", N_transits=N_transits
    )
    RunDefinitionSimple.add_data(
        "JWST",
        f"/home/ajnb3/sai/pandexo_files/{atm_code}_{N_transits}.txt",
        model_generating_function=mgf,
    )

    # Running retrieval
    OUTPUT_DIR = f"/data/ajnb3/results/retrieval/{atm_code}_{N_transits}/"
    retrieval = Retrieval(
        RunDefinitionSimple,
        output_dir=OUTPUT_DIR,
        sample_spec=False,
        pRT_plot_style=False,
        ultranest=False,
    )
    retrieval.run(n_live_points=400, const_efficiency_mode=False, resume=False)

    return retrieval


# Spectrum generating model
def mgf(pRT_object, parameters, PT_plot_mode=False, AMR=False):
    """
    `parameters` should be a dictionary containing:
        `logp0`: surface pressure,
        `T0`: surface temperature,
        `Tiso`: isotherm temperature at top of atmosphere,
        `spec`: log mass abundance of spec species
    """
    pressures, temperatures = myutils.isotherm_adiabat_stitcher(
        10 ** (parameters["logp0"].value),
        parameters["T0"].value,
        parameters["Tiso"].value,
    )
    pressures, temperatures = myutils.sort_pT_profile(pressures, temperatures)

    abundances = {}
    for species in pRT_object.line_species:
        abundances[species] = 10 ** parameters[species].value * np.ones_like(pressures)

    MMW = calc_MMW(abundances)  # pRT's function, not mine

    pRT_object.setup_opa_structure(pressures)
    pRT_object.calc_transm(
        temperatures,
        abundances,
        10**2.94,  # Gravity
        MMW,
        R_pl=0.95 * nc.r_earth,
        P0_bar=10 ** (parameters["logp0"].value),
    )
    wlen_model = nc.c / pRT_object.freq / 1e-4  # in um
    spectrum_model = (pRT_object.transm_rad / (0.1192 * nc.r_sun)) ** 2.0  # (R/R*)^2
    return wlen_model, spectrum_model


## Functions to read results of retrievals
def get_multinest_samples(atm_code, N_transits):
    """
    Extracts the multinest samples from the results file(s)
    """
    file_code = f"{atm_code}_{N_transits}"
    file_list = glob.glob(f'/data/ajnb3/results/retrieval/{file_code}/out_PMN/{file_code}*post_equal_weights.dat')
    samples_list = [np.loadtxt(samples_file) for samples_file in file_list]
    samples = np.concatenate(samples_list, axis=0)
    
    return samples


def get_multinest_stats_dict(atm_code, N_transits):
    """
    Grabs the dictionary (in a json file) for the statistics
    of a multinest run
    """
    file_code = f"{atm_code}_{N_transits}"
    big_stats_dict = json.loads(
        myutils.read_from(
            f"/data/ajnb3/results/retrieval/{file_code}/out_PMN/{file_code}_stats.json"
        )
    )
    stats_dicts = big_stats_dict['marginals']
    return stats_dicts


def get_sigmas(atm_code, N_transits, index, n_sigma):
    """
    Gets the ```n_sigma``` limits for the ```{atm_code}_{N_transits}``` run
    """
    assert n_sigma in [1,2,3,5]
    sigma_key = f'{n_sigma}sigma'
    stats_dicts = get_multinest_stats_dict(
        atm_code, N_transits
    )
    sigmas = stats_dicts[index][sigma_key]
    return sigmas