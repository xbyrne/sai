"""
Calculates gas and mineral abundances over a pT grid
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from . import myutils

# Pressures at which GGchem will run
pmin, pmax = 0.1, 1e4  # bar
NPOINTS = 100
p_points = np.logspace(np.log10(pmin), np.log10(pmax), NPOINTS)

# file_code = sys.argv[1]  # e.g. lessO, moreS, noCa

# abund_file = "abund_Venus.in"  # Relative to ../GGchem
# abund_file = 'abund_Venus_lessO.in'
# abund_file = 'abund_Venus_moreO.in'

# output_file = "/data/ajnb3/results/grid/grid_output.csv"
# output_file = '/data/ajnb3/results/grid/grid_output_lessO.csv'
# output_file = '/data/ajnb3/results/grid/grid_output_moreO.csv'


def calculate_GGchem_grid(file_code):
    """
    Calculates an abundance grid using GGchem for a given output_file,
    which must be:
    /data/ajnb3/results/grid/grid_results_{file_code}.npz
    """
    abund_file = f"abund_Venus_{file_code}.in"
    output_file = f"/data/ajnb3/results/grid/grid_results_{file_code}.npz"

    for i, p in tqdm(enumerate(p_points), total=len(p_points)):
        # Reading in template input file
        input_text = myutils.create_GGchem_input_file(abund_file, p)
        # Writing to input file, to be actually run on by GGchem
        myutils.overwrite_to("./GGchem/input/model_Venus_p.in", input_text)
        # Running GGchem
        os.system("bash ./run_ggchem_p.sh > /dev/null")  # Mute

        this_p_df = myutils.extract_GGchem_df()

        if i == 0:
            all_p_df = this_p_df.copy()
        else:
            union_columns = all_p_df.columns.union(this_p_df.columns, sort=False)
            all_p_df = all_p_df.reindex(columns=union_columns, fill_value=-300.0)
            this_p_df = this_p_df.reindex(columns=union_columns, fill_value=-300.0)
            all_p_df = pd.concat([all_p_df, this_p_df])

    T_K = all_p_df.Tg.to_numpy().reshape((100, 100))
    p_bar = all_p_df.pgas.to_numpy().reshape((100, 100)) * 1e-6  # was in ubar fsr

    gas_start = all_p_df.columns.get_loc("H")
    gas_end = [i for i, col in enumerate(all_p_df.columns) if col[0] == "n"][1]
    gas_species_names = all_p_df.columns[gas_start:gas_end]
    cond_colnames = all_p_df.columns[gas_end:]
    condensates_names = np.array([cn[1:] for cn in cond_colnames])

    gas_species_cm3 = (
        all_p_df[gas_species_names]
        .to_numpy()
        .reshape((100, 100, len(gas_species_names)))
    )  # This is the correct reshaping!!
    gas_species_cm3 = np.moveaxis(gas_species_cm3, 2, 0)
    gas_species_mr = 10**gas_species_cm3 / np.sum(10**gas_species_cm3, axis=0)

    condensates_cm3 = (
        all_p_df[cond_colnames].to_numpy().reshape((100, 100, len(cond_colnames)))
    )
    condensates_cm3 = np.moveaxis(condensates_cm3, 2, 0)
    condensates_mf = 10**condensates_cm3 / np.sum(10**condensates_cm3, axis=0)

    np.savez_compressed(
        output_file,
        T_K=T_K,
        p_bar=p_bar,
        gas_species_names=gas_species_names,
        gas_species_mr=gas_species_mr,
        condensates_names=condensates_names,
        condensates_mf=condensates_mf,
    )

def calculate_GGchem_grid_moreS():
    """
    Same as below, but just for moreS which is bad
    """
    abund_file = "abund_Venus_moreS.in"
    results_file = "/data/ajnb3/results/grid/grid_output_moreS.dat"
    output_file = "/data/ajnb3/results/grid/grid_results_moreS.npz"

    for i, p in tqdm(enumerate(p_points), total=len(p_points)):
        input_text = myutils.create_GGchem_input_file(abund_file, p)
        myutils.overwrite_to("./GGchem/input/model_Venus_p.in", input_text)
        os.system("bash ./run_ggchem_p.sh > /dev/null")
        if i == 0:
            myutils.overwrite_to(
                results_file, myutils.read_from("./GGchem/Static_Conc.dat")
            )
        else:
            myutils.write_to(
                results_file,
                "\n".join(
                    myutils.read_from("./GGchem/Static_Conc.dat").split("\n")[3:]
                ),
            )
    (
        p_bar,
        T_K,
        gas_species_names,
        gas_species_mr,
        condensates_names,
        condensates_mf,
    ) = myutils.gather_GGchem_results(results_file=results_file)
    np.savez_compressed(
        output_file,
        p_bar=p_bar,
        T_K=T_K,
        gas_species_names=gas_species_names,
        gas_species_mr=gas_species_mr,
        condensates_names=condensates_names,
        condensates_mf=condensates_mf,
    )