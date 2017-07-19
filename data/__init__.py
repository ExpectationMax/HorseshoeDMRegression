import pandas as pd
import os
basepath = os.path.dirname(os.path.abspath(__file__))


def get_simulated_data(dataname='5O_5C_6p0_5S'):
    out = {}
    out['covariates'] = pd.read_table(os.path.join(basepath, dataname, 'XX.tsv'))
    out['counts'] = pd.read_table(os.path.join(basepath, dataname, 'YY.tsv'))
    out['beta'] = pd.read_table(os.path.join(basepath, dataname, 'betas.tsv'))
    out['alpha'] = pd.read_table(os.path.join(basepath, dataname, 'alphas.tsv'))
    return out