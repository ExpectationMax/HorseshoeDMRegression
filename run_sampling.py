import matplotlib
matplotlib.use('Agg')
import pymc3 as pm
import os
import math
import dm_regression_model
from dirichlet_multinomial import init_nuts_advi_map
import seaborn as sns
import pickle
import numpy as np
from data import get_simulated_data

def traceplot_with_priors(trace, model):
    selected_vars = [varname for varname in model.named_vars.keys() if varname in trace.varnames and not varname.endswith('__') and hasattr(model, varname) and hasattr(model[varname], 'distribution')]
    remove = []
    for i, var in enumerate(selected_vars):
        try:
            model[var].distribution.logp(0).eval()
        except:
            remove.append(i)

    for rem in remove[::-1]:
        selected_vars.pop(rem)
    print(selected_vars)
    pm.traceplot(trace, varnames=selected_vars, priors=[model[var].distribution for var in selected_vars])

def run_sampling_with_model(name, model, outputpath, rseed, njobs=1, tune=1500, draws=1000):
    print(name)
    with open(os.path.join(outputpath, '{}_model.pck'.format(name)), 'wb') as f:
        pickle.dump(model, f)
        print('Stored model')

    if name == 'explicit_complete' or name == 'explicit_horseshoe_nu3' or name == 'explicit_horseshoe_nu1':
        with model:
            start, cov = init_nuts_advi_map(100000, njobs, rseed, model)
            step = pm.NUTS(scaling=cov, is_cov=True, target_accept=0.9)
            trace = pm.sample(draws=draws, tune=tune, start=start, step=step, njobs=njobs)
    else:
        with model:
            trace = pm.sample(draws=draws, njobs=njobs, tune=tune, nuts_kwargs={'target_accept': 0.9})

    with open(os.path.join(outputpath, '{}_sampling.pck'.format(name)), 'wb') as f:
        pickle.dump(trace, f)
        print('Stored trace')

    pm.traceplot(trace)
    sns.plt.savefig(os.path.join(outputpath, '{}_traceplot.pdf'.format(name)))


if __name__ == '__main__':
    import argparse
    from joblib import Parallel, delayed
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+', choices=
                        ['5O_5C_6p0_5S', '5O_5C_6p0_25S', '5O_5C_6p0_50S', '5O_5C_6p0_100S', '5O_5C_6p0_150S',
                         '10O_10C_24p0_5S', '10O_10C_24p0_25S', '10O_10C_24p0_50S', '10O_10C_24p0_100S',
                         '10O_10C_24p0_150S'])
    args = parser.parse_args()
    with Parallel(n_jobs=4) as parallel:
        for dataset in args.datasets:
            outputpath = os.path.join('results', dataset)
            os.makedirs(outputpath, exist_ok=True)
            data = get_simulated_data(dataset)
            S, O = data['counts'].shape
            S, C = data['covariates'].shape

            rseed = 35424353

            p0 = (data['beta'] != 0).sum().sum()
            sigma = 1
            t0 = (p0 / (C * O)) * (sigma / math.sqrt(S))
            print('p0 =', p0, 'sigma =', sigma, 'tau0 =', t0)
            with open(os.path.join(outputpath, 'parameters.txt'), 'w') as f:
                f.write('p0 = {}, sigma = {}, tau0 = {}'.format(p0, sigma, t0))

            models = {}
            models['explicit_horseshoe_nu1'] = dm_regression_model.DMRegressionModelExplicit(S, C, O, t0, data['counts'], data['covariates'], nu=1)
            models['implicit_horseshoe_nu1'] = dm_regression_model.DMRegressionModel(S, C, O, t0, nu=1)
            models['implicit_horseshoe_nu1'].set_counts_and_covariates(data['counts'], data['covariates'])
            #models['implicit_complete'] = dm_regression_model.DMRegressionModelNonsparseImplicit(S, C, O, data['counts'], data['covariates'])
            ##models['implicit_complete'].set_counts_and_covariates(data['counts'], data['covariates'])
            #models['explicit_complete'] = dm_regression_model.DMRegressionModelNonsparseExplicit(S, C, O, data['counts'], data['covariates'])

            parallel(delayed(run_sampling_with_model)(name, model, outputpath, rseed) for name, model in models.items())


