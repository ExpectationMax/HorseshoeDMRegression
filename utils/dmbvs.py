import numpy as np
import pandas as pd
import pickle
from joblib import Parallel, delayed
from data import get_available_datasets, get_simulated_data
from os.path import join
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


def compute_alpha_init(countdata):
    return pd.Series(scale(np.log(countdata.sum(axis=0))))


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

    masking = (multipletests(pvalues.flatten(), method='fdr_bh')[1].reshape(pvalues.shape) <= 0.2).astype(float) + 0
    beta_init = correlations * masking
    return pd.DataFrame(beta_init, index=countdata.columns, columns=metadata.columns)


def scale(data):
    return (data - data.mean())/data.std(ddof=1)


def get_model_data(folder, dataset):
    inputfile = join(folder, dataset, 'results.pck')
    with open(inputfile, 'rb') as f:
        data = pickle.load(f)

    return data


def read_result(folder, dataset, variables, functions):
    try:
        data = get_simulated_data(dataset, as_dataframe=True)
        modeldata = get_model_data(folder, dataset)
    except Exception as e:
        print('Error reading results form {}'.format(dataset))
        print(e)
        return None, dataset

    result = pd.DataFrame(
        columns=['Dataset', 'Variable', 'Groundtruth', 'Prediction (mean)', 'Prediction (std)'])

    if 'Sampling finished!' not in modeldata['stdout'].decode():
        print('Sampling for dataset {} ended prematurely. Stderr: \n{}\n Skipping...'.format(dataset, modeldata['stderr'].decode()))
        return None, dataset

    for var in variables:
        try:
            n_values = modeldata[var].size
            if var in data.keys():
                if hasattr(modeldata[var], 'columns'):
                    if data[var].shape[0] != modeldata[var].shape[0]:
                        assert np.all(data[var].columns == modeldata[var].index)
                        data[var] = data[var].T
                    else:
                        assert np.all(data[var].columns == modeldata[var].columns) and np.all(data[var].index == modeldata[var].index)
                else:
                    assert np.all(data[var].columns == modeldata[var].index)

                r = pd.DataFrame(
                    {'Dataset': [dataset] * n_values, 'Variable': [var] * n_values,
                     'Groundtruth': data[var].values.flatten(),
                     'Prediction (mean)': modeldata[var].values.flatten()})
            else:
                r = pd.DataFrame(
                    {'Dataset': [dataset] * n_values, 'Variable': [var] * n_values,
                     'Groundtruth': [np.NaN] * n_values,
                     'Prediction (mean)': modeldata[var].values.flatten()})

            result = result.append(r, ignore_index=True)
        except KeyError as e:
            print('Unable to calculate {} for dataset {}.'.format(var, dataset))
            print(e)

    for func in functions:
        try:
            r = result.append(func(dataset, data, modeldata), ignore_index=True)
        except Exception as e:
            print(e)
    result = result.append(r, ignore_index=True)

    return result, dataset


def compute_MPPI(dataset, data, modeldata):
    gt_beta = data['beta'].T.values.flatten()
    n_values = gt_beta.size
    masked_beta_trace = np.ma.array(modeldata['beta_trace'], mask=modeldata['beta_trace']==0)
    predicted_beta = masked_beta_trace.mean(axis=0).flatten().filled(0)
    mppi = modeldata['MPPI'].values.flatten()
    return pd.DataFrame(
        {'Dataset': [dataset] * n_values, 'Variable': ['beta'] * n_values,
         'Groundtruth': gt_beta,
         'Prediction (mean)': predicted_beta,
         'Prediction (std)': mppi})


def read_results(folder, n_jobs):
    results = Parallel(n_jobs=n_jobs)(delayed(read_result)(folder, dataset, ['alpha', 'MPPI'], [compute_MPPI]) for dataset in get_available_datasets())
    fitting_results = pd.concat([res[0] for res in results if res[0] is not None])
    failed = [res[1] for res in results if res[0] is None]
    return failed, fitting_results
