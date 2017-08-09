import pymc3 as pm
import numpy as np
import logging
from pymc3.step_methods.hmc import quadpotential
from pymc3.variational.callbacks import Callback

from .data import get_input_specs

def run_hmc_sampling(countdata, metadata, p0, n_chains, n_tune, n_draws, seed):
    import dm_regression_model

    O, C, S = get_input_specs(countdata.T, metadata)
    tau0 = compute_tau(O, C, S, p0)
    sampling_logger = logging.getLogger('Sampling')
    nu = 1

    sampling_logger.info(
        'Running sampling with parameters: tau0 = %f, nu = %i, n_chains = %i, n_tune = %i, n_draws = %i',
        tau0, nu, n_chains, n_tune, n_draws)

    if seed == -1:
        sampling_logger.warning('Random seed not set, please note following value to ensure reproducibility.')
    rseed, seeds = get_random_seeds(seed, n_chains)
    sampling_logger.info('Random seed used for sampling: %i', rseed)

    model = dm_regression_model.DMRegressionModel(S, C, O, tau0, nu=nu, centered=False)
    model.set_counts_and_covariates(countdata, metadata)
    try:
        with model:
            start, step = init_nuts(njobs=n_chains, random_seed=seeds)
            trace = pm.sample(n_draws, tune=n_tune, start=start, step=step, njobs=n_chains, random_seed=seeds)
    except Exception as e:
        sampling_logger.error('Error during initialisation or sampling:')
        sampling_logger.error('%s', e)
        raise e

    return model, trace


def compute_tau(O, C, S, p0, sigma=1):
    return (p0 / (C * O)) * (sigma / np.sqrt(S))


def get_random_seeds(rseed, njobs):
    get_randseed = lambda: np.random.randint(0, 2**32 - 1)
    from theano.sandbox.rng_mrg import M2
    get_thanoseed = lambda: np.random.randint(0, M2 -1)
    if rseed == -1:
        np.random.seed()
        rseed = get_randseed()
    np.random.seed(rseed)

    return rseed, [get_thanoseed() for i in range(njobs)]


class EarlyStopping(Callback):
    def __init__(self, every=100, tolerance=1e-2, patience=50):
        self.every = every
        self.min = None
        self.tolerance = tolerance
        self.patience = patience
        self.patience_count = 0


    def __call__(self, _, scores, i):
        if self.min is None:
            self.min = scores[max(0, i - 1000):i + 1].mean()
            return
        if i % self.every or i < self.every:
            return
        if i < 1000:
            return
        current = scores[max(0, i - 1000):i + 1].mean()

        if current < self.min:
            self.min = current
            self.patience_count = 0
        elif (current - self.min) > self.tolerance*self.min:
            self.patience_count += 1
            if self.patience_count > self.patience:
                raise StopIteration('Stopping fitting at %d due to increasing loss.' % i)


def init_nuts(njobs=1, n_init=200000, model=None,
              random_seed=-1, progressbar=True, start_at_map=False, **kwargs):
    model = pm.modelcontext(model)
    vars = kwargs.get('vars', model.vars)
    if set(vars) != set(model.vars):
        raise ValueError('Must use init_nuts on all variables of a model.')
    if not pm.model.all_continuous(vars):
        raise ValueError('init_nuts can only be used for models with only '
                         'continuous variables.')

    pm._log.info('Initializing NUTS using map+advi+adapt_diag...')

    random_seed = int(np.atleast_1d(random_seed)[0])

    cb = [
        pm.callbacks.CheckParametersConvergence(tolerance=1e-2, diff='absolute'),
        pm.callbacks.CheckParametersConvergence(tolerance=1e-2, diff='relative'),
        EarlyStopping(tolerance=1e-2)
    ]

    if start_at_map:
        start = pm.find_MAP()
        approx = pm.MeanField(model=model, start=start)
        method = pm.ADVI.from_mean_field(approx)
    else:
        method='advi'

    approx = pm.fit(
        random_seed=random_seed,
        n=n_init, method=method, model=model,
        callbacks=cb,
        progressbar=progressbar,
        obj_optimizer=pm.adagrad_window,
    )
    start = approx.sample(draws=njobs)
    start = list(start)
    stds = approx.gbij.rmap(approx.std.eval())
    cov = model.dict_to_array(stds) ** 2
    mean = approx.gbij.rmap(approx.mean.get_value())
    mean = model.dict_to_array(mean)
    weight = 50
    potential = quadpotential.QuadPotentialDiagAdapt(
        model.ndim, mean, cov, weight)
    if njobs == 1:
        start = start[0]

    step = pm.NUTS(potential=potential, **kwargs)
    return start, step