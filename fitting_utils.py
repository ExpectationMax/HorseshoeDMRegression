import numpy as np
import pymc3 as pm
from pymc3.step_methods.hmc import quadpotential
from pymc3.variational.callbacks import Callback

class EarlyStopping(Callback):
    def __init__(self, every=100, tolerance=5e-2, patience=20):
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
              random_seed=-1, progressbar=True, **kwargs):
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
    start = pm.find_MAP()
    approx = pm.MeanField(model=model, start=start)
    approx = pm.fit(
        random_seed=random_seed,
        n=n_init, method=pm.ADVI.from_mean_field(approx), model=model,
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
