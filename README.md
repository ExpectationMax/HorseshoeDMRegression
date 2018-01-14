Read Me
=======

This project aims at providing a fast and flexible solution for the estimation of covariate effects on individual microbes of a sequenced metagenomic sample.


Project structure
-----------------

```
├── data
│   ├── __init__.py
│   ├── biological                <-- biological datasets
│   └── simulated                 <-- simulated datasets
│
├── lib                           <-- dmbvs binary location
│
├── utils                         <-- package with internal functions             
│   ├── __init__.py
│   ├── cli.py                    <-- command line interface
│   ├── data.py                   <-- data processing and internal representation
│   ├── distributions.py          <-- implementation of custom distributions (Dirichlet-Multinomial)
│   ├── dmbvs.py                  <-- wrapping functions to call dmbvs
│   ├── result_analysis.py        <-- computation of summary statistics for traces
│   ├── resultfile_processing.py  <-- formatting and parsing of result file names
│   └── sampling.py               <-- functions to setup hmc sampling
│
├── README.md                     <-- this file
├── analyse_single_trace.py       <-- compute summary statistics and inclusion probabilities
|                                     for a single trace
├── benchmark_oracle_guess.py     <-- run sampling on multiple datasets and with(out) oracle guess
├── combine_dmbvs_results.py      <-- read results from individual dmbvs runs and combine them into
|                                     a joint dataset contain information about dataset parameters
├── combine_hmc_results.py        <-- read results from multiple hmc sampling runs of the devised model and
|                                     combine them into a joint dataset
├── dm_regression_model.py        <-- definition of multiple models based on the sparsity inducing horseshoe prior
|                                     and the Dirichlet-Multinomial distribution
├── dmbvs_wrapper.py              <-- run dmbvs algorithm on multiple datasets
└── run_sampling.py               <-- run hmc sampling on model
```

Getting started
---------------

To get started after installing the software, it is required to provide per sample species wise read counts and covariates in a tab separated values file (tsv).

Alternatively, the files in the folder `data/simulated` can be used for testing and benchmarking. In this folder simulated simulated datasets of different dimensionality and data availability can be found. The folder names follow the scheme `<#species/OTUs>O_<#covaraites>C_<#covariates relevant>p0_<#samples>S_<replica>R` and contain the files `alphas.tsv` (groundtruth regression intercepts), `betas.tsv` (groundtruth regression coefficients), `YY.tsv` (sample wise read counts of species abundances) and `XX.tsv` (sample wise covariate values).

Using the `run_sampling.py` script, the parameters of a statistical model can be inferred using MCMC sampling.
This script additionally generates a file with summary statistics and inclusion probabilities of individual covariates.

The detailed usage of this script is described below.


Requirements
------------

This project is written in python 3.6, thus a python interpreter with at least this version should be installed for the software to function.
The software was tested on UNIX based operating systems (Linux and Mac OS X).
Although much effort was put into writing platform independent code, due to insufficient testing problems could arise on Windows operating systems.


Installation
------------

Clone the repository and change to the root directory.
Add the root directory to your `PYTHONPATH` environment variable (e.g. via the .bashrc file) like this:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/HorseshoeDMRegression
```

Activate your virtual environment, if you intend to use one (recommended).
Install the requirements with

```bash
pip install -r requirements.txt
```

Running sampling for a statistical model
---------------------------------------

The parameters of statistical models are inferred using the `run_sampling.py` script:

```
usage: run_sampling.py [-h] [--transpose-counts] --estimated_covariates
                       ESTIMATED_COVARIATES -o OUTPUT [--n_chains N_CHAINS]
                       [--n_tune N_TUNE] [--n_draws N_DRAWS] [--seed SEED]
                       [--model_type {DMRegression,DMRegressionMixed,DMRegressionDMixed,MvNormalDMRegression,SoftmaxRegression}]
                       [--traceplot] [--save_model] [--save_trace]
                       countdata metadata

Infer covariate effects using a dirichlet multinomial regression model with
horseshoe sparsity induction.

positional arguments:
  countdata             Read counts associated with individual microbes in tab
                        separated format with layout: Samples (rows) x Microbes
                        (columns).
  metadata              Covariates associated with samples in tab separated
                        format with layout: Samples (rows) x Covariates
                        (columns).

optional arguments:
  -h, --help            show this help message and exit
  --transpose-counts    Accept read counts in tab separated format with
                        Microbes (rows) x Samples (columns) layout instead.
  --estimated_covariates ESTIMATED_COVARIATES
                        Guess on number of relevant covariates, provide -1 if
                        no guess should be used (scale of cauchy on tau = 1).
  -o OUTPUT, --output OUTPUT
                        Directory to store results.

Sampling options:
  --n_chains N_CHAINS   Number of chains to runn in parallel.
  --n_tune N_TUNE       Number of tuning steps that should be descarded as
                        "burn-in".
  --n_draws N_DRAWS     Number of samples to generate after tuning
  --seed SEED           Random seed
  --model_type {DMRegression,DMRegressionMixed,DMRegressionDMixed,MvNormalDMRegression,SoftmaxRegression}
                        Model type to use, further details see readme.

Output options:
  --traceplot           Save traveplot of sampling for diagnostics.
  --save_model          Save model.
  --save_trace          Store trace for later analysis.
```

Example running sampling on the `40O_10C_20p0_100S_1R` dataset:
```bash
python run_sampling.py data/simulated/40O_10C_20p0_100S_1R/YY.tsv data/simulated/40O_10C_20p0_100S_1R/XX.tsv -o sampling_testrun --save_trace --traceplot
```

Running this command will generate three files, `trace.pck` containing sampler statistics and generated samples, `traceplot.pdf` containing histograms of the sampled values for diagnostic purposes and `variable_statistics.xlsx` containing summary statistics of inferred parameters and inclusion probabilities.

Model types
-----------

 * **DMRegression**: Dirichlet multinomial regression model with horseshoe sparsity induction. Regression coefficients as well as intercepts are estimated jointly for all samples.
 * **DMRegressionMixed**: Dirichlet multinomial regression model with horseshoe sparsity induction, hierarchical mixed model. Regression coefficients as well as intercepts are estimated jointly. For each patient, additional mixed effects are estimated. Requires additional *patient* column in covariate file to determine repeated measurements form same patient
 * **DMRegressionDMixed**: Dirichlet multinomial regression model with horseshoe sparsity induction, hierarchical mixed model. Regression coefficients as well as intercepts are estimated jointly. Additional dirichlet clustered random effects are included to in the model, no patient column required. Assumes that certain groups can have distinct microbiome compositions (enterotypes).
 * **MvNormalDMRegression**: Dirichlet multinomial regression model with horseshoe sparsity induction, intercepts sampled from Multivariate Normal. Regression coefficients are estimated jointly, intercepts are modeled to be independently drawn from a Multivariate Gaussian distribution of which the Covariance matrix and the Mean are estimated jointly for all samples.


Additional information
----------------------

The *analyse_single_trace.py* script, allows to compute summary statistics using a provided model, trace and input data.

The *benchmark_oracle_guess.py* script can be used to run sampling on multiple simulated datasets for performance benchmarking.

The *combine_dmbvs_results.py* and *combine_hmc_results.py* scripts were written to crawl all runs in a path that were either executed using *benchmark_oracle_guess.py* for hmc sampling or *dmbvs_wrapper.py* for the dmbvs algorithm, compute summary statistics of all runs and variables and store these computations in a table format to allow easy comparison of performance benchmarks.

The *dmbvs_wrapper.py* script can be used to run a competitor algorithm (dmbvs) for performance comparison. The script was implemented to remove the requirement of installing `R`.

All scripts give further information on their usage using the command `script.py --help`.
