import pandas as pd
import os
basepath = os.path.dirname(os.path.abspath(__file__))


def get_simulated_data():
    out = {}
    out['covariates'] = pd.read_table(os.path.join(basepath, 'XX.tsv'))
    out['counts'] = pd.read_table(os.path.join(basepath, 'YY.tsv'))
    out['beta'] = pd.read_table(os.path.join(basepath, 'betas.tsv'))
    out['alpha'] = pd.read_table(os.path.join(basepath, 'alphas.tsv'))
    return out