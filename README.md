# rmtchem: A library for numerical continuation of steady states in random networks of chemical reactions

The rmtchem library contains the Python file rmtchem.py and a Jupyter notbook file rmtchem.ipny for individual runs, along with scripts for running batches and a Jupyter notbook file plot.ipny for plotting results.

The required Python packages can be installed in anaconda environment with `conda create -n rmtchem_env -c conda-forge numpy scipy cantera networkx matplotlib jupyter` followed by `conda activate rmtchem_env`.

Running `./rmtchem.py -h` produces the following usage message:

```
usage: rmtchem.py [-h] --filebase FILEBASE [--n N] [--nr NR] [--nd ND]
                  [--dmax DMAX] [--type TYPE] [--d0 D0] [--seed SEED]
                  [--dep DEP] [--na NA] [--skip SKIP] [--output OUTPUT]
                  [--quasistatic QUASI] [--integrate INTEG] [--rank RANK]

Random chemical reaction networks.

optional arguments:
  -h, --help           show this help message and exit
  --filebase FILEBASE  Base string for file output
  --n N                Number of species
  --nr NR              Number of reactions
  --nd ND              Number of drives
  --dmax DMAX          Maximum drive
  --type TYPE          Type of adjacency matrix. 0 for chemical networks, 1
                       for ER networks.
  --d0 D0              Drive timescale
  --seed SEED          Random seed
  --dep DEP            Step size for driving
  --na NA              Number of autocatalytic reactions
  --skip SKIP          Steps to skip for output
  --output OUTPUT      1 for matrix output, 0 for none
  --quasistatic QUASI  1 for quasistatic
  --integrate INTEG    1 for integrate
  --rank RANK          1 for rank calculation
```
