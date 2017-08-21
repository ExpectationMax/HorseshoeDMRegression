import argparse
import pickle
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from data import get_available_datasets, get_simulated_data
from os.path import exists, join
from normalize_parameters import split_datasetname_into_parameters

def get_model_data(folder, dataset):
    inputfile = join(folder, dataset, 'results.pck')
    with open(inputfile, 'rb') as f:
        data = pickle.load(f)

    return data


def read_result(folder, dataset, variables, functions):
    try:
        data = get_simulated_data(dataset, as_dataframe=True)
        modeldata = get_model_data(folder, dataset)
    except:
        print('Error reading results form {}'.format(dataset))
        return None, dataset

    result = pd.DataFrame(
        columns=['Dataset', 'Variable', 'Groundtruth', 'Prediction (mean)', 'Prediction (std)'])

    if modeldata['stderr'] != b'':
        print('Run on dataset {} had output on stderr:'.format(dataset))
        print(modeldata['stderr'])
        print('skipping...')
        return None, dataset

    for var in variables:
        try:
            n_values = data[var].size
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
        except KeyError:
            print('Unable to calculate {} for dataset {}.'.format(var, dataset))

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_folder')
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--n_jobs', default=10, type=int)

    args = parser.parse_args()
    failed, results = read_results(args.results_folder, args.n_jobs)

    print('failed datasets:')
    print(failed)

    _, res_dataset_details = split_datasetname_into_parameters(results['Dataset'])
    res = pd.concat([res_dataset_details, results], axis=1)

    with open(args.output, 'wb') as f:
        pickle.dump(res, f)
