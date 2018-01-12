#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import pymc3 as pm
import os
import dm_regression_model
from utils.sampling import compute_tau
import pickle
import numpy as np
from data import get_simulated_data, get_available_datasets
from itertools import product


def run_sampling_with_model(name, model, outputpath, rseed, njobs=1, tune=2000, draws=2000):
    print(name)
    if os.path.isfile(os.path.join(outputpath, '{}_sampling.pck'.format(name))):
        return

    with open(os.path.join(outputpath, '{}_model.pck'.format(name)), 'wb') as f:
        pickle.dump(model, f)
        print('Stored model')
    try:
        with model:
            trace = pm.sample(draws=draws, njobs=njobs, tune=tune, random_seed=rseed, init='jitter+diag')

    except Exception as e:
        print('Error occured:', e)
        return

    with open(os.path.join(outputpath, '{}_sampling.pck'.format(name)), 'wb') as f:
        pickle.dump(trace, f)
        print('Stored trace')


if __name__ == '__main__':
    import argparse
    from joblib import Parallel, delayed
    parser = argparse.ArgumentParser(description='Run sampling with and without oracle guess on multiple datasets.')
    parser.add_argument('datasets', nargs='+', choices=get_available_datasets(), help='Datasets on which sampling should be run.')
    parser.add_argument('-o', '--output', default='results', type=str, help='Path to store sampling trace.')
    parser.add_argument('--njobs', type=int, default=10, help='Number of jobs to run in parallel.')
    parser.add_argument('--rseed', type=int, default=35424353, help='Random seed used for HMC initialization.')
    args = parser.parse_args()
    with Parallel(n_jobs=args.njobs) as parallel:
        for dataset in args.datasets:
            outputpath = os.path.join(args.output, dataset)
            os.makedirs(outputpath, exist_ok=True)
            data = get_simulated_data(dataset)
            S, O = data['counts'].shape
            S, C = data['covariates'].shape

            rseed = args.rseed

            p0 = (data['beta'] != 0).sum().sum()
            repeat = data['repetition']
            print(dataset)

            models = {}

            nus = [1]
            centereds = [False]
            cauchys = [True]
            p0s = [p0, -1]

            sigma = 1

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


