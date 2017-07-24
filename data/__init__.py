import pandas as pd
import os
basepath = os.path.dirname(os.path.abspath(__file__))


def get_simulated_data(dataname='5O_5C_6p0_5S'):
    out = {}
    out['covariates'] = pd.read_table(os.path.join(basepath, dataname, 'XX.tsv')).values
    out['counts'] = pd.read_table(os.path.join(basepath, dataname, 'YY.tsv')).values
    out['beta'] = pd.read_table(os.path.join(basepath, dataname, 'betas.tsv')).T.values
    out['alpha'] = pd.read_table(os.path.join(basepath, dataname, 'alphas.tsv')).T.values
    return out

def get_available_datasets():
    return [filename for filename in os.listdir(basepath) if os.path.isdir(os.path.join(basepath, filename)) and not filename.endswith('__')]