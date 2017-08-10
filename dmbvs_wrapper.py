import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

def compute_alpha_init(countdata):
    return pd.DataFrame(scale(np.log(countdata.sum(axis=0))))


def compute_beta_init(countdata, metadata):
    composition = countdata.div(countdata.sum(axis=1), axis=0)
    nOTUs = countdata.shape[1]
    nCovariates = metadata.shape[1]
    correlations = np.zeros((nOTUs, nCovariates))
    pvalues = np.zeros((nOTUs, nCovariates))
    for i in range(nOTUs):
        for j in range(nCovariates):
            cor, pval = spearmanr(composition.iloc[:, i], metadata.iloc[:, j])
            correlations[i, j] = cor
            pvalues[i, j] = pval

    masking = (multipletests(pvalues.flatten(), method='fdr_bh')[1].reshape(pvalues.shape) <= 0.2).astype(float)  + 0
    beta_init = correlations * masking
    return pd.DataFrame(beta_init, index=countdata.columns, columns=metadata.columns)


def scale(data):
    return (data - data.mean())/data.std()


def run_dmbvs(metadata, countdata, GG, thin, burn, output_location, intercept_variance=10, slab_variance=10,
              bb_alpha=0.02, bb_beta=1.98, proposal_alpha = 0.5, proposal_beta = 0.5,
              executable = os.path.join("lib","dmbvs.x"), r_seed = None, cleanup=True):
    os.makedirs(output_location, exist_ok=True)
    float_parameters = {'decimal':'.', 'index':False, 'header':False, 'sep':' ', 'float_format':'%.15g'}
    metadata.to_csv(os.path.join(output_location, 'covariates.txt'), **float_parameters)
    countdata.astype(int).to_csv(os.path.join(output_location, 'count_matrix.txt'), **float_parameters)

    nsamples = countdata.shape[0]
    notus = countdata.shape[1]
    ncovariates = metadata.shape[1]

    # initialization
    alpha_init = compute_alpha_init(countdata)
    alpha_init.to_csv(os.path.join(output_location, 'init_alpha.txt'), **float_parameters)

    beta_init = compute_beta_init(countdata, metadata)
    pd.DataFrame(beta_init.values.flatten()).to_csv(os.path.join(output_location, 'init_beta.txt'), **float_parameters)

    # write parameters
    pd.DataFrame([proposal_alpha]*countdata.shape[1]).to_csv(os.path.join(output_location, 'proposal_alpha.txt'), **float_parameters)
    pd.DataFrame([proposal_beta]*(countdata.shape[1]*metadata.shape[1]))\
        .to_csv(os.path.join(output_location, 'proposal_beta.txt'), **float_parameters)

    if r_seed == None:
        r_seed = np.random.randint(1, 1e6)

    proc = subprocess.Popen([executable] + list(map(str,
                                          [GG, thin, burn, intercept_variance, slab_variance, bb_alpha, bb_beta, notus,
                                           nsamples, ncovariates, r_seed, output_location]
                                           )
                                       ),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = proc.communicate()

    # read results
    alpha = pd.read_table(os.path.join(output_location, 'alpha.out'), skipinitialspace=True, index_col=None, header=None, sep='\s+')
    alpha.columns = countdata.columns
    alpha_mean = alpha.mean(axis=0)

    beta = pd.read_table(os.path.join(output_location, 'beta.out'), skipinitialspace=True, index_col=None, header=None, sep='\s+')
    beta_reshaped = beta.values.reshape((-1, notus, ncovariates))
    beta_mean = pd.DataFrame(beta_reshaped.mean(axis=0), index=countdata.columns, columns=metadata.columns)

    if cleanup:
        shutil.rmtree(output_location)

    return {'alpha': alpha_mean, 'beta':beta_mean, 'alpha_trace':alpha, 'beta_trace': beta_reshaped, 'rseed': r_seed,
            'stdout': stdout, 'stderr':stderr}


