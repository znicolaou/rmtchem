# rmtchem
Random networks of chemical reactions

The rmtchem library contains the Python file rmtchem.py and a Jupyter notbook file rmtchem.ipny for plotting results, along with scripts for running batches.

Running `./rmtchem.py -h` produces the following usage message:

```
usage: rmtchem.py [-h] --filebase FILEBASE [--n N] [--nr NR] [--nd ND]
                  [--dmax DMAX] [--type TYPE] [--d0 D0] [--seed SEED]
                  [--steps STEPS] [--skip SKIP] [--output OUTPUT]
                  [--quasistatic QUASI] [--rank RANK]

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
  --steps STEPS        Steps for driving
  --skip SKIP          Steps to skip for output
  --output OUTPUT      1 for matrix output, 0 for none
  --quasistatic QUASI  1 for quasistatic
  --rank RANK          1 for rank calculation
```
