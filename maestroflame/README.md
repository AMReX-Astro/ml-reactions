## The example directory notebook requires two pre requisite steps.

## 1. Data Generation
To generate data we use the aprox13 nuclear reaction network included in Microphysics.

Use this fork/branch of MAESTROeX to run a flame https://github.com/dwillcox/maestroex/tree/save_reactions

This allows you to set maestro.save_react_int in an input file which determines how many time steps between each batch of saved input/output nuclear reaction data. This is 3 sets of input/output data per timestep (due to the nature of the coupling method).

## 2. Install this package
pip install .

Now that you have your data, you're ready to go. Navigate to the examples/ directory to get started.
