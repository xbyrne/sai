module unload anaconda3
module load anaconda2
python GGchem/tools/make_abund.py
module unload anaconda2
module load anaconda3
mv abund_Venus.in mfrac_Venus.in GGchem
