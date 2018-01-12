import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import logging
logging.basicConfig(level=logging.DEBUG)
import pandas as pd
import pymc3 as pm
from data import get_simulated_data
from utils.cli import nonexistant_file
from utils.resultfile_processing import get_model_data, get_sucessful_runs, get_sample_size_from_dataset, \
    split_datasetname_into_parameters, split_modelname_into_parameters
from utils.result_analysis import compute_pseudo_inclusion_probability
from joblib import Parallel, delayed


def analyse_run(dataset, model, variables, functions, save_traceplot, resultspath='results'):
    result = pd.DataFrame(
        columns=['Dataset', 'Model', 'Variable', 'Groundtruth', 'Prediction (mean)', 'Prediction (std)'])
    statistics = pd.DataFrame(
        columns=['Dataset', 'Model', 'depth', 'diverging', 'mean_tree_accept', 'step_size', 'tree_size'])
    logging.info('Dataset: %s, Model: %s', dataset, model)
    try:
        data = get_simulated_data(dataset)
        trace = get_model_data(dataset, model, resultspath=resultspath)
    except Exception as e:
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


def create_performance_dataframe(datasets, variables, functions, save_traceplot=True, resultspath='results'):
    with Parallel(n_jobs=-2) as parallel:
        dataset_model_combinations = []
        for dataset in datasets:
            models = get_sucessful_runs(dataset, resultspath)
            for model in models:
                dataset_model_combinations.append((dataset, model))

        combined_results = parallel(delayed(analyse_run)(dataset, model, variables, functions, save_traceplot, resultspath) for dataset, model in dataset_model_combinations)
        fitting_results = pd.concat([res[0] for res in combined_results if res[0] is not None])
        statistics_results = pd.concat([res[1] for res in combined_results if res[0] is not None])
        failed = [res[1] for res in combined_results if res[0] is None]

    return failed, fitting_results, statistics_results


def compute_pip_values(dataset, data, model, trace):
    pip = compute_pseudo_inclusion_probability(trace, get_sample_size_from_dataset(dataset)).flatten()
    gt_beta = data['beta'].flatten()
    beta_mean = trace['beta'].mean(axis=0).flatten()
    n_values = pip.size
    return pd.DataFrame(
        {'Dataset': [dataset] * n_values, 'Model': [model] * n_values, 'Variable': ['pip'] * n_values,
         'Groundtruth': gt_beta,
         'Prediction (mean)': beta_mean,
         'Prediction (std)': pip})


def variable_selected(dataset, data, model, trace):
    pip = compute_pseudo_inclusion_probability(trace, get_sample_size_from_dataset(dataset)).flatten()
    include = pip > 0.5
    gt_beta = data['beta'].flatten() != 0
    n_values = include.size
    return pd.DataFrame(
        {'Dataset': [dataset] * n_values, 'Model': [model] * n_values, 'Variable': ['include_covariate'] * n_values,
         'Groundtruth': gt_beta,
         'Prediction (mean)': include,
         'Prediction (std)': pip})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('resultsfolder', type=str, help='Folder with results from benchmark_oracle_guess.py run.')
    parser.add_argument('-o','--output', type=nonexistant_file, required=True, help='Filename to store summary statistics of all traces.')
    parser.add_argument('--sampler-statistics', type=nonexistant_file, help='Filename to store sampler statistics of traces.')

    args = parser.parse_args()

    from data import get_available_datasets

    functions = [
        variable_selected,
        compute_pip_values
    ]
    failed, res, stats = create_performance_dataframe(get_available_datasets(), ['alpha','beta', 'tau'], functions,
                                                      save_traceplot=True, resultspath=args.resultsfolder)
    print('Failed datasets:')
    print(failed)
    _, res_model_details = split_modelname_into_parameters(res['Model'])
    _, res_dataset_details = split_datasetname_into_parameters(res['Dataset'])
    res = pd.concat([res_dataset_details, res_model_details, res], axis=1)

    _, stats_model_details = split_modelname_into_parameters(stats['Model'])
    _, stats_dataset_details = split_datasetname_into_parameters(stats['Dataset'])
    stats = pd.concat([stats_dataset_details, stats_model_details, stats], axis=1)

    res.to_pickle(args.output)
    stats.to_pickle(args.sampler_statistics)
