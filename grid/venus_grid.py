import numpy as np
import os

# File reading/writing in python
def read_from(filename):
    with open(filename, "r") as f:
        filetext = f.read()
    return filetext


def overwrite_to(filename, text):
    with open(filename, "w") as f:
        f.write(text)


def write_to(filename, text):
    with open(filename, "a") as f:
        f.write(text)


# Preparing output file
overwrite_to("./grid_output.dat", read_from("./grid_output_header.dat"))

# Pressures at which GGchem will run
pmin, pmax = 0.1, 1e4  # bar
Npoints = 100
p_points = np.logspace(np.log10(pmin), np.log10(pmax), Npoints)

for p in p_points:
    # Reading in template input file
    template_text = read_from("./model_Venus_pblank.in")
    # Amendment to template file, specifying the pressure to run this time
    p_string = f"{p:.5e}\t! pmin [bar]\n{p:.5e}\t! pmax [bar]\n"
    input_text = template_text + p_string
    # Writing to input file, to be actually run on by GGchem
    overwrite_to("../GGchem/input/model_Venus_p.in", input_text)
    # Running GGchem
    os.system("bash ./run_ggchem_p.sh")

    this_p_output = read_from("../GGchem/Static_Conc.dat")
    this_p_output = '\n'.join(this_p_output.split('\n')[3:])
    write_to("./grid_output.dat", this_p_output)
