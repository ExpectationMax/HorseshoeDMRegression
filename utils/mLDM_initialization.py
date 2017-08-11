import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.linalg import cholesky

def r_scale(data):
    return (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)


def relativeAbundance(data):
    return data.div(data.sum(axis=1), axis=0)


def calculate_B_init(countdata, metadata):
    n_otu = countdata.shape[1]
    relative_abundance = relativeAbundance(countdata)
    metadata_scaled = r_scale(metadata)
    B_init = spearmanr(relative_abundance, metadata_scaled)[0][n_otu:, :n_otu]
    return pd.DataFrame(B_init, columns=countdata.columns, index=metadata.columns)


def calculate_z_init(countdata, z_mean=10):
    return np.log(countdata + 1) + z_mean


def calculate_B0_init(countdata, slac=0.001):
    b_init = calculate_z_init(countdata).mean(axis=0)
    #if np.any(b_init < 0):
    #    b_init = b_init - b_init.min() + slac

    return b_init


def calculate_cholesky_theta_init(countdata):
    n_otu = countdata.shape[1]
    correlations = spearmanr(countdata)[0]
    while np.linalg.det(correlations) < n_otu:
        correlations[np.diag_indices_from(correlations)] += 1

    correlations_chol = cholesky(correlations, lower=True)
    return correlations_chol[np.tril_indices_from(correlations_chol)]


