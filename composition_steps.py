"""
Taking steps away from Venus in compositional space,
and seeing if the mineral-gas associations hold
"""
import os
import numpy as np
import myutils

# Altering Ca
def alter_Ca():
    """
    Alters Ca composition, generates lines at 1400K
    """
    for Ca_abund in np.linspace(18.23, 18.25, 21):
        abund_number = round((Ca_abund - 18) * 1000)  # From 230 to 250
        abund_code = f"abund_Ca{abund_number}"
        print(f"Currently running for number {abund_number}")
        abund_df = myutils.df_from_abund("abund_Venus")
        abund_df.at["Ca", "epsilon"] = Ca_abund
        myutils.df_to_abund(abund_df, abund_code)
        myutils.create_GGchem_input_file(
            filename="grid_line_1400.in", Tbounds=[1400, 1400], abund_code=abund_code
        )
        os.system(
            "cd ./GGchem && ./ggchem input/grid_line_1400.in > /dev/null && cd .."
        )
        df = myutils.gather_GGchem_results()
        df.to_csv(f"/data/ajnb3/results/summer/Ca_lines/Ca{abund_number}.csv")


# Altering O
def alter_O():
    """
    Alters O composition, generates lines at 1400K
    """
    for O_abund in np.linspace(
        # 19.559, 19.560, 11
        19.559, 19.570, 12,
    ):
        # abund_number = round((O_abund - 19) * 10000)  # From 5590 to 5600
        abund_number = round((O_abund - 19) * 1000)  # From 559 to 570
        abund_code = f"abund_O{abund_number}"
        print(f"Currently running for number {abund_number}")
        abund_df = myutils.df_from_abund("abund_Venus")
        abund_df.at["O", "epsilon"] = O_abund
        myutils.df_to_abund(abund_df, abund_code)
        myutils.create_GGchem_input_file(
            filename="grid_line_1400.in", Tbounds=[1400, 1400], abund_code=abund_code
        )
        os.system(
            "cd ./GGchem && ./ggchem input/grid_line_1400.in > /dev/null && cd .."
        )
        df = myutils.gather_GGchem_results()
        df.to_csv(f"/data/ajnb3/results/summer/O_lines/O{abund_number}.csv")


# Altering Ca, O
def depsilon_O(depsilon_Ca):
    """
    For a given alteration in Ca's conventional elemental abundance
    parameter epsilon, gives the alteration in O's value of epsilon
    such that the same amount of Ca and O atoms are added
    (relative to Venus)
    """
    df_venus = myutils.df_from_abund("abund_Venus")
    epsilon_Ca = df_venus.at["Ca", "epsilon"]  # ~18.24
    epsilon_O = df_venus.at["O", "epsilon"]  # ~19.56

    return (
        np.log10(10 ** (epsilon_Ca + depsilon_Ca) + 10**epsilon_O - 10**epsilon_Ca)
        - epsilon_O
    )


def alter_Ca_O():
    """
    Alters Ca and O composition simultaneously, such that relative to Venus we have
    just added CaO
    """
    abs_deps_Cas = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.08,
        0.1,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]
    depsilons_Ca = abs_deps_Cas + [-ep for ep in abs_deps_Cas]
    for deps_Ca in depsilons_Ca:
        print(deps_Ca)
        deps_O = depsilon_O(deps_Ca)
        abund_df = myutils.df_from_abund("abund_Venus")
        abund_df.at["Ca", "epsilon"] += deps_Ca
        abund_df.at["O", "epsilon"] += deps_O
        myutils.df_to_abund(abund_df, "abund_Venus_CaO")
        myutils.create_GGchem_input_file(
            filename="grid_line_1400.in",
            Tbounds=[1400, 1400],
            abund_code="abund_Venus_CaO",
        )
        os.system(
            "cd ./GGchem && ./ggchem input/grid_line_1400.in > /dev/null && cd .."
        )
        df = myutils.gather_GGchem_results()
        df.to_csv(f"/data/ajnb3/results/summer/Ca_O_lines/CaO{deps_Ca}.csv")
