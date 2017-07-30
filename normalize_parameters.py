import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import pickle
from glob import glob
import pandas as pd
import numpy as np
import pymc3 as pm
from data import get_simulated_data


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


def create_performance_dataframe(datasets, variables, compute_ra_performance=True, save_traceplot=True): #derived_variables={}):
    result = pd.DataFrame(columns=['Dataset', 'Model', 'Variable', 'Groundtruth', 'Prediction (mean)', 'Prediction (std)'])
    statistics = pd.DataFrame(columns=['Dataset', 'Model', 'depth', 'diverging', 'mean_tree_accept', 'step_size', 'tree_size'])
    for dataset in datasets:
        data = get_simulated_data(dataset)
        models = get_sucessful_runs(dataset)
        for model in models:
            print(dataset, model)
            try:
                trace = get_model_data(dataset, model)
            except:
                print('Error')
                continue

            stats = pd.Series({'Dataset': dataset, 'Model': model, 'depth': trace.get_sampler_stats('depth').mean(),
                                  'diverging': trace.get_sampler_stats('diverging').sum(),
                                  'mean_tree_accept': trace.get_sampler_stats('mean_tree_accept').mean(),
                                  'step_size': trace.get_sampler_stats('step_size').mean(),
                                  'tree_size': trace.get_sampler_stats('mean_tree_accept').mean()})
            statistics = statistics.append(stats, ignore_index=True)

            for var in variables:
                means = trace[var].mean(axis=0)
                stds = trace[var].std(axis=0)
                n_values = means.size
                r = pd.DataFrame({'Dataset':[dataset]*n_values,'Model':[model]*n_values, 'Variable':[var]*n_values,
                              'Groundtruth': data[var].flatten(),
                              'Prediction (mean)': means.flatten(),
                              'Prediction (std)': stds.flatten()})
                result = result.append(r, ignore_index=True)
            #for name, (inputs, function) in derived_variables.items():
            #    inputgt = [data[i] for i in inputs]
            #    inputmodel = [trace[]]
            if compute_ra_performance:
                gt_ra = compute_ra(data['alpha'], data['beta'], data['covariates'])
                model_ra = compute_ra(trace['alpha'], trace['beta'], data['covariates'])
                means = model_ra.mean(axis=0)
                stds = model_ra.std(axis=0)
                n_values = means.size
                r = pd.DataFrame(
                    {'Dataset': [dataset] * n_values, 'Model': [model] * n_values, 'Variable': ['Relative Abundance'] * n_values,
                     'Groundtruth': gt_ra.flatten(),
                     'Prediction (mean)': means.flatten(),
                     'Prediction (std)': stds.flatten()})
                result = result.append(r, ignore_index=True)

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

            del trace
    return result, statistics

if __name__ == '__main__':
    from data import get_available_datasets
    #models = ['explicit_horseshoe_nu1', 'explicit_horseshoe_nu3', 'implicit_horseshoe_nu1', 'implicit_horseshoe_nu3',
    #          'explicit_complete', 'implicit_complete']
    res, stats = create_performance_dataframe(get_available_datasets(), ['alpha','beta'])
    res.to_pickle('performance_comparison.pck')
    stats.to_pickle('sampler_statistics.pck')
    print()