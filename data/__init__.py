import pandas as pd
import os
basepath = os.path.dirname(os.path.abspath(__file__))


def get_simulated_data(dataname='5O_5C_6p0_5S'):
    out = {}
    out['covariates'] = pd.read_table(os.path.join(basepath, 'simulated', dataname, 'XX.tsv')).values
    out['counts'] = pd.read_table(os.path.join(basepath, 'simulated', dataname, 'YY.tsv')).values
    out['beta'] = pd.read_table(os.path.join(basepath, 'simulated', dataname, 'betas.tsv')).T.values
    out['alpha'] = pd.read_table(os.path.join(basepath, 'simulated', dataname, 'alphas.tsv')).T.values
    out['repetition'] = int(dataname.split('_')[-1][:-1])
    return out

def get_available_datasets():
    return [filename for filename in os.listdir(os.path.join(basepath, 'simulated')) if os.path.isdir(os.path.join(basepath, 'simulated', filename)) and not filename.endswith('__')]


def get_biological_datasets():
    return [filename for filename in os.listdir(os.path.join(basepath, 'biological')) if os.path.isdir(os.path.join(basepath, 'biological', filename)) and not filename.endswith('__')]

def get_biological_data(dataname):
    return {'counts': pd.read_table(os.path.join(basepath, 'biological', dataname, 'X.tsv'), index_col=0),
            'covariates': pd.read_table(os.path.join(basepath, 'biological', dataname, 'M.tsv'), index_col=0)}
