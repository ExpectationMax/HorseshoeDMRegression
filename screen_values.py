#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import pymc3 as pm
import os
import math
import dm_regression_model
from utils.sampling import compute_tau
import pickle
import numpy as np
from data import get_simulated_data, get_available_datasets
from itertools import product
from utils.sampling import init_nuts

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

def run_sampling_with_model(name, model, outputpath, rseed, njobs=1, tune=2000, draws=2000):
    print(name)
    if os.path.isfile(os.path.join(outputpath, '{}_sampling.pck'.format(name))):
        return

    with open(os.path.join(outputpath, '{}_model.pck'.format(name)), 'wb') as f:
        pickle.dump(model, f)
        print('Stored model')
    try:
        #if name == 'explicit_complete' or name.startswith('explicit_horseshoe'):
        #    with model:
        #            start, cov = init_nuts_advi_map(100000, njobs, rseed, model)
        #            step = pm.NUTS(scaling=cov, is_cov=True, target_accept=0.9)
        #            trace = pm.sample(draws=draws, tune=tune, start=start, step=step, njobs=njobs)
        #else:
        with model:
            start, step = init_nuts(njobs, random_seed=rseed)
            trace = pm.sample(draws=draws, njobs=njobs, tune=tune, start=start, step=step, random_seed=rseed)

    except Exception as e:
        print('Error occured:', e)
        return

    with open(os.path.join(outputpath, '{}_sampling.pck'.format(name)), 'wb') as f:
        pickle.dump(trace, f)
        print('Stored trace')


if __name__ == '__main__':
    import argparse
    from joblib import Parallel, delayed
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+', choices=get_available_datasets())
    parser.add_argument('--njobs', type=int, default=10)
    args = parser.parse_args()
    with Parallel(n_jobs=args.njobs) as parallel:
        for dataset in args.datasets:
            outputpath = os.path.join('results', dataset)
            os.makedirs(outputpath, exist_ok=True)
            data = get_simulated_data(dataset)
            S, O = data['counts'].shape
            S, C = data['covariates'].shape

            rseed = 35424353

            p0 = (data['beta'] != 0).sum().sum()
            repeat = data['repetition']
            print(dataset)

            models = {}

            #models['implicit_complete'] = dm_regression_model.DMRegressionModelNonsparseImplicit(S, C, O,
            #                                                                                     data['counts'],
            #                                                                                     data['covariates'])

            nus = [1]
            centereds = [False]
            cauchys = [True]
            p0s = [p0, -1]

            sigma = 1
            #sigmas = [1, 2, 3]

            for nu, centered, cauchy, p0 in product(nus, centereds, cauchys, p0s):
                name = 'horseshoe_nu{}_{}_{}_p0{}_R{}'.format(nu, 'centered' if centered else 'noncentered', 'cauchy' if cauchy else 'normal', p0, repeat)
                if p0 != -1:
                    t0 = compute_tau(O, C, S, p0, sigma)
                else:
                    t0 = 1

                print('p0 =', p0, 'sigma =', sigma, 'tau0 =', t0, 'C =', C, 'O = ', O, 'S = ', S, 'nu =', nu, 'cauchy =', cauchy)
                with open(os.path.join(outputpath, name+'_parameters.txt'), 'w') as f:
                    f.write('p0 = {}, sigma = {}, tau0 = {}'.format(p0, sigma, t0))

                models[name] = dm_regression_model.DMRegressionModel(S, C, O, t0, nu=nu, centered=centered, cauchy=cauchy)
                models[name].set_counts_and_covariates(data['counts'].astype(np.uint), data['covariates'])

            parallel(delayed(run_sampling_with_model)(name, model, outputpath, rseed) for name, model in models.items())


