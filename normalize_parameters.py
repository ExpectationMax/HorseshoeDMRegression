import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import logging
logging.basicConfig(level=logging.DEBUG)
import pickle
from glob import glob
import pandas as pd
import numpy as np
import pymc3 as pm
from data import get_simulated_data
from utils.result_analysis import compute_pseudo_inclusion_probability
from joblib import Parallel, delayed


def get_model_data(dataset, modelname):
    inputfile = os.path.join('results', dataset, '{}_sampling.pck'.format(modelname))
    with open(inputfile, 'rb') as f:
        trace = pickle.load(f)

    #data = {'beta': trace['beta'].mean(axis=0), 'alpha': trace['alpha'].mean(axis=0)}
    return trace#, data

def compute_ra(alpha, beta, covariates):
    if len(beta.shape) == 3:
        res = np.array([np.exp(a + np.dot(covariates, b)) for a, b in zip(alpha, beta)])
    else:
        res = np.exp(alpha + np.dot(covariates, beta))
    return res/res.sum(axis=-1, keepdims=True)

def get_sucessful_runs(dataset):
    datafiles = [os.path.basename(filepath)[:-len('_sampling.pck')] for filepath in glob(os.path.join('results', dataset, '*_sampling.pck'))]
    return datafiles



def analyse_run(dataset, model, variables, functions, save_traceplot):
    result = pd.DataFrame(
        columns=['Dataset', 'Model', 'Variable', 'Groundtruth', 'Prediction (mean)', 'Prediction (std)'])
    statistics = pd.DataFrame(
        columns=['Dataset', 'Model', 'depth', 'diverging', 'mean_tree_accept', 'step_size', 'tree_size'])
    logging.info('Dataset: %s, Model: %s', dataset, model)
    try:
        data = get_simulated_data(dataset)
        trace = get_model_data(dataset, model)
    except Exception as e:
        failed.append((dataset, model))
        logging.error('Error while processing (Dataset: %s, Model: %s):\n%s', dataset, model, e)
        return None, (dataset, model)

    stats = pd.Series({'Dataset': dataset, 'Model': model, 'depth': trace.get_sampler_stats('depth').mean(),
                       'diverging': trace.get_sampler_stats('diverging').sum(),
                       'mean_tree_accept': trace.get_sampler_stats('mean_tree_accept').mean(),
                       'step_size': trace.get_sampler_stats('step_size').mean(),
                       'tree_size': trace.get_sampler_stats('tree_size').mean()})
    statistics = statistics.append(stats, ignore_index=True)

    for var in variables:
        try:
            means = trace[var].mean(axis=0)
            stds = trace[var].std(axis=0)
            n_values = means.size
            if var in data.keys():
                r = pd.DataFrame(
                    {'Dataset': [dataset] * n_values, 'Model': [model] * n_values, 'Variable': [var] * n_values,
                     'Groundtruth': data[var].flatten(),
                     'Prediction (mean)': means.flatten(),
                     'Prediction (std)': stds.flatten()})
            else:
                r = pd.DataFrame(
                    {'Dataset': [dataset] * n_values, 'Model': [model] * n_values, 'Variable': [var] * n_values,
                     'Groundtruth': [0] * n_values,
                     'Prediction (mean)': means.flatten(),
                     'Prediction (std)': stds.flatten()})
            result = result.append(r, ignore_index=True)
        except KeyError:
            print('Unable to calculate {} for model {}.'.format(var, model))

    # for name, (inputs, function) in derived_variables.items():
    #    inputgt = [data[i] for i in inputs]
    #    inputmodel = [trace[]]
    for func in functions:
        try:
            result = result.append(func(dataset, data, model, trace), ignore_index=True)
        except Exception as e:
            print(e)

    if save_traceplot:
        os.makedirs('traceplots', exist_ok=True)
        try:
            # Complete models do not have tau
            pm.traceplot(trace, variables + ['tau'])
        except:
            pm.traceplot(trace, variables)
        finally:
            plt.savefig(os.path.join('traceplots', '{}-{}-traceplot.pdf'.format(dataset, model)))
        plt.close('all')

    return result, statistics


def create_performance_dataframe(datasets, variables, functions, save_traceplot=True): #derived_variables={}):
    #result = pd.DataFrame(columns=['Dataset', 'Model', 'Variable', 'Groundtruth', 'Prediction (mean)', 'Prediction (std)'])
    #statistics = pd.DataFrame(columns=['Dataset', 'Model', 'depth', 'diverging', 'mean_tree_accept', 'step_size', 'tree_size'])
    failed = []
    with Parallel(n_jobs=10) as parallel:
        dataset_model_combinations = []
        for dataset in datasets:
            models = get_sucessful_runs(dataset)
            for model in models:
                dataset_model_combinations.append((dataset, model))

        combined_results = parallel(delayed(analyse_run)(dataset, model, variables, functions, save_traceplot) for dataset, model in dataset_model_combinations)
        fitting_results = pd.concat([res[0] for res in combined_results if res[0] is not None])
        statistics_results = pd.concat([res[1] for res in combined_results if res[0] is not None])
        failed = [res[1] for res in combined_results if res[0] is None]

    return failed, fitting_results, statistics_results


model_lookup = {}
def split_modelname_into_parameters(data):
    model_parameters = ['Type', 'Nu', 'Parametrization', 'Hyperprior', 'Model p0']
    def split_model_name(modelname):
        if modelname in model_lookup.keys():
            return model_lookup[modelname]
        fragments = modelname.split('_')
        if len(fragments) == 2:
            ret = pd.Series(index=model_parameters, data=[fragments[1], '', '', '', ''])
        else:
            ret = pd.Series(index=model_parameters, data=[fragments[0], int(fragments[1][2:]), fragments[2], fragments[3], int(fragments[4][2:])])
        model_lookup[modelname] = ret
        return ret

    return model_parameters, data.apply(split_model_name)


dataset_lookup = {}
def split_datasetname_into_parameters(data):
    dataset_parameters = ['OTUs', 'Covariates', 'Data p0', 'Samples']
    def split_dataset_name(dataset):
        if dataset in dataset_lookup.keys():
            return dataset_lookup[dataset]
        frags = dataset.split('_')
        ret = pd.Series(index=dataset_parameters,
                        data=[int(frags[0][:-1]), int(frags[1][:-1]), int(frags[2][:-2]), int(frags[3][:-1])])
        dataset_lookup[dataset] = ret
        return ret

    return dataset_parameters, data.apply(split_dataset_name)


def compute_ra_performance(dataset, data, model, trace):
    gt_ra = compute_ra(data['alpha'], data['beta'], data['covariates'])
    model_ra = compute_ra(trace['alpha'], trace['beta'], data['covariates'])
    means = model_ra.mean(axis=0)
    stds = model_ra.std(axis=0)
    n_values = means.size
    return pd.DataFrame(
        {'Dataset': [dataset] * n_values, 'Model': [model] * n_values, 'Variable': ['Relative Abundance'] * n_values,
         'Groundtruth': gt_ra.flatten(),
         'Prediction (mean)': means.flatten(),
         'Prediction (std)': stds.flatten()})


def compute_pip_values(dataset, data, model, trace):
    pip = compute_pseudo_inclusion_probability(trace).flatten()
    gt_beta = data['beta'].flatten()
    beta_mean = trace['beta'].mean(axis=0).flatten()
    n_values = pip.size
    return pd.DataFrame(
        {'Dataset': [dataset] * n_values, 'Model': [model] * n_values, 'Variable': ['pip'] * n_values,
         'Groundtruth': gt_beta,
         'Prediction (mean)': beta_mean,
         'Prediction (std)': pip})

if __name__ == '__main__':
    from data import get_available_datasets
    #models = ['explicit_horseshoe_nu1', 'explicit_horseshoe_nu3', 'implicit_horseshoe_nu1', 'implicit_horseshoe_nu3',
    #          'explicit_complete', 'implicit_complete']

    functions = [
        #compute_ra_performance,
        compute_pip_values
    ]
    failed, res, stats = create_performance_dataframe(get_available_datasets(), ['alpha','beta', 'tau'], functions)
    print('Failed datasets:')
    print(failed)
    _, res_model_details = split_modelname_into_parameters(res['Model'])
    _, res_dataset_details = split_datasetname_into_parameters(res['Dataset'])
    res = pd.concat([res_dataset_details, res_model_details, res], axis=1)

    _, stats_model_details = split_modelname_into_parameters(stats['Model'])
    _, stats_dataset_details = split_datasetname_into_parameters(stats['Dataset'])
    stats = pd.concat([stats_dataset_details, stats_model_details, stats], axis=1)

    res.to_pickle('performance_comparison.pck')
    stats.to_pickle('sampler_statistics.pck')
