import pymc3 as pm
import numpy as np
import logging

from .data import get_input_specs


def run_hmc_sampling(countdata, metadata, patients, p0, n_chains, n_tune, n_draws, seed,  model_type):
    import dm_regression_model

    O, C, S = get_input_specs(countdata.T, metadata)
    if p0 == -1:
        tau0 = 1
    else:
        tau0 = compute_tau(O, C, S, p0)

    sampling_logger = logging.getLogger('Sampling')
    nu = 5

    sampling_logger.info(
        'Running sampling with parameters: tau0 = %f, nu = %i, n_chains = %i, n_tune = %i, n_draws = %i',
        tau0, nu, n_chains, n_tune, n_draws)

    if seed == -1:
        sampling_logger.warning('Random seed not set, please note following value to ensure reproducibility.')
    rseed, seeds = get_random_seeds(seed, n_chains)
    sampling_logger.info('Random seed used for sampling: %i', rseed)

    if model_type == 'DMRegression':
        model = dm_regression_model.DMRegressionModel(S, C, O, tau0, nu=nu, centered=False)
        model.set_counts_and_covariates(countdata, metadata)
    elif model_type == 'MvNormalDMRegression':
        model = dm_regression_model.DMRegressionMVNormalModel(countdata.values, pm.floatX(metadata.values), tau0, nu=nu,
                                                              centered=False)
    elif model_type == 'MvNormalDiagDMRegression':
        model = dm_regression_model.DMRegressionMvNormalDiagModel(
            countdata.values, pm.floatX(metadata.values), patients, tau0, nu=nu)
    elif model_type == 'DMRegressionMixed':
        model = dm_regression_model.DMRegressionMixedCovariates(countdata.values, pm.floatX(metadata.values), patients, tau0, nu=nu, centered=False)
    elif model_type == 'DMRegressionDMixed':
        model = dm_regression_model.DMRegressionMixedDP(countdata.values, pm.floatX(metadata.values), pm.floatX(tau0),
                                                        DP_components=6, alpha=pm.floatX(0.6), nu=pm.floatX(nu), centered=False)
    else:
        raise ValueError('Model type not correctly specified')

    try:
        with model:
            # make sure all logp are finite
            print(model.check_test_point())
            trace = pm.sample(n_draws, tune=n_tune, chains=n_chains,
                              random_seed=seeds, cores=n_chains,
                              init='jitter+adapt_diag')
    except Exception as e:
        sampling_logger.error('Error during initialisation or sampling:')
        sampling_logger.error('%s', e)
        raise e

    return model, trace


def compute_tau(O, C, S, p0, sigma=1):
    return (p0 / (C * O - p0)) * (sigma / np.sqrt(S))


def get_random_seeds(rseed, njobs):
    get_randseed = lambda: np.random.randint(0, 2**32 - 1)
    from theano.sandbox.rng_mrg import M2
    get_thanoseed = lambda: np.random.randint(0, M2 -1)
    if rseed == -1:
        np.random.seed()
        rseed = get_randseed()
    np.random.seed(rseed)

    return rseed, [get_thanoseed() for i in range(njobs)]
