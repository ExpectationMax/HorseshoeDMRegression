import matplotlib.pyplot as plt
import pymc3 as pm
from pymc3.distributions import generate_samples, draw_values
from pymc3.distributions.dist_math import gammaln, bound, factln
import theano
import theano.tensor as tt
import numpy as np


class DirichletMultinomial(pm.Discrete):
    def __init__(self, n, a, *args, **kwargs):
        super(DirichletMultinomial, self).__init__(*args, **kwargs)

        self.K = tt.as_tensor_variable(a.shape[-1])
        self.n = tt.as_tensor_variable(n[:, np.newaxis])

        if a.ndim == 1:
            self.alphas = tt.as_tensor_variable(a[np.newaxis, :])  # alphas[1, #classes]
        else:
            self.alphas = tt.as_tensor_variable(a)  # alphas[#samples, #classes]

        self.A = self.alphas.sum(axis=-1, keepdims=True)  # A[#samples]
        self.mean = self.n * (self.alphas / self.A)

        self.mode = tt.cast(pm.math.tround(self.mean), 'int32')

    def logp(self, value):
        printing = False
        k = self.K
        a = self.alphas
        A = self.A
        n = self.n
        res = bound(tt.squeeze(factln(n) + gammaln(A) - gammaln(A + n) +
                               tt.sum(gammaln(a + value) - gammaln(a) - factln(value), keepdims=True, axis=-1)),
                    tt.all(value >= 0),
                    tt.all(tt.eq(tt.sum(value, axis=-1, keepdims=True), n)),
                    tt.all(a > 0),
                    k > 1,
                    tt.all(tt.ge(n, 0)),
                    broadcast_conditions=False
                    )
        return res


    def _random(self, n, alphas, size=None):
        n = np.squeeze(n)
        if size == alphas.shape:
            size = None
        if alphas.shape[0] == 1:
            p = np.random.dirichlet(np.squeeze(alphas), size=size[0])
            res = np.array([np.random.multinomial(cur_n, cur_p) for cur_n, cur_p in zip(n, p)])
        else:
            res = np.array([np.random.multinomial(cur_n, np.random.dirichlet(cur_a)) for cur_n, cur_a in zip(n, alphas)])

        return res

    def random(self, point=None, size=None):
        n, a = draw_values([self.n, self.alphas], point=point)
        samples = generate_samples(self._random, n, a,
                                   dist_shape=self.shape,
                                   size=size)
        return samples


    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        a = dist.a
        return r'${} \sim \text{{Dirichlet}}(\mathit{{a}}={})$'.format(name,
                                                                       get_variable_name(a))


def run_explicit_model_gamma(data, random_seed):
    S, O = data.shape
    sums = data.sum(axis=1)
    with pm.Model() as model:
        BoundedFlat = pm.Bound(pm.Flat, lower=0)
        a = BoundedFlat('alpha')
        b = BoundedFlat('beta')
        alphas = pm.Gamma('gamma', a, b, shape=O)
        dirichlets = pm.Dirichlet('dirichlets', alphas, shape=(S, O))
        obs = pm.Multinomial('obs', sums, dirichlets, observed=data, shape=(S, O))

    with model:
        njobs = 2
        start, cov = init_nuts_advi_map(100000, njobs, random_seed, model)
        step = pm.NUTS(scaling=cov, is_cov=True)
        trace = pm.sample(2000, tune=1000, start=start, step=step, njobs=njobs)

    pm.traceplot(trace)
    plt.savefig('Explicit_model_gamma.pdf')
    pm.summary(trace, to_file='Explicit_model_gamma_summary.txt')



def run_implicit_model_gamma(data, r, random_seed):
    S, O = data.shape
    sums = data.sum(axis=1)
    with pm.Model() as model:
        #alpha = pm.Normal('alpha', 1.14, 0.26, transform=pm.distributions.transforms.lowerbound(0))
        #beta = pm.Normal('beta', 3.15, 0.89, transform=pm.distributions.transforms.lowerbound(0))
        BoundedFlat = pm.Bound(pm.Flat, lower=0)
        a = BoundedFlat('alpha')
        b = BoundedFlat('beta')
        alphas = pm.Gamma('gamma', a, b , shape=O)
        obs = DirichletMultinomial('obs', sums, alphas, observed=data, shape=(S, O))

    with model:
        njobs = 2
        start, cov = init_nuts_advi_map(100000, njobs, random_seed, model)
        step = pm.NUTS(scaling=cov, is_cov=True)
        trace = pm.sample(2000, tune=1000, start=start, step=step, njobs=njobs)

    return model, trace
    #pm.traceplot(trace)
    #plt.savefig('Implicit_model_gamma.pdf')
    #pm.summary(trace, to_file='Implicit_model_gamma_summary.txt')


def run_explicit_model(data, random_seed):
    S, O = data.shape
    sums = data.sum(axis=1)
    with pm.Model() as model:
        BoundedFlat = pm.Bound(pm.Flat, lower=0)
        alphas = BoundedFlat('alphas', shape=O)
        dirichlets = pm.Dirichlet('dirichlets', alphas, shape=(S, O))
        obs = pm.Multinomial('obs', sums, dirichlets, observed=data, shape=(S, O))

    with model:
        njobs = 2
        start, cov = init_nuts_advi_map(100000, njobs, random_seed, model)
        step = pm.NUTS(scaling=cov, is_cov=True)
        trace = pm.sample(2000, tune=1000, start=start, step=step, njobs=njobs)

    pm.traceplot(trace)
    plt.savefig('Explicit_model_{}.pdf'.format(datatype))
    pm.summary(trace, to_file='Explicit_model_summary_{}.txt'.format(datatype))


def run_implicit_model(data, datatype, random_seed):
    S, O = data.shape
    sums = data.sum(axis=1)
    with pm.Model() as model:
        #alpha = pm.Normal('alpha', 1.14, 0.26, transform=pm.distributions.transforms.lowerbound(0))
        #beta = pm.Normal('beta', 3.15, 0.89, transform=pm.distributions.transforms.lowerbound(0))
        BoundedFlat = pm.Bound(pm.Flat, lower=0)
        alphas = BoundedFlat('alphas', shape=O)
        obs = DirichletMultinomial('obs', sums, alphas, observed=data, shape=(S, O))

    with model:
        njobs = 2
        start, cov = init_nuts_advi_map(100000, njobs, random_seed, model)
        step = pm.NUTS(scaling=cov, is_cov=True)
        trace = pm.sample(2000, tune=1000, start=start, step=step, njobs=njobs)

    #pm.traceplot(trace)
    #plt.savefig('Implicit_model_{}.pdf'.format(datatype))
    #pm.summary(trace, to_file='Implicit_model_summary_{}.txt'.format(datatype))
    return model, trace


def init_nuts_advi_map(n_init, njobs, rseed, model):
    cb = [
        pm.callbacks.CheckParametersConvergence(tolerance=1e-2, diff='absolute'),
        pm.callbacks.CheckParametersConvergence(tolerance=1e-2, diff='relative'),
    ]
    #start = pm.find_MAP(model=model)
    #approx = pm.MeanField(model=model, start=start)
    approx = pm.fit(random_seed=rseed,
                    n=n_init, method=pm.ADVI(), progressbar=True,
                    obj_optimizer=pm.adagrad_window(learning_rate=2e-4), total_grad_norm_constraint=10,
                    model=model, callbacks=cb
                    )
    start = approx.sample(draws=njobs)
    stds = approx.gbij.rmap(approx.std.eval())
    cov = model.dict_to_array(stds) ** 2
    if njobs == 1:
        start = start[0]
    return start, cov

if __name__ == '__main__':
    import sys
    #sys.setrecursionlimit(1000000)
    datatype = 'real'
    random_seed = 124324223
    if datatype == 'real':
        from data import get_real_data
        s = get_real_data()
    elif datatype == 'simulated':
        from data import get_simulated_data
        s = get_simulated_data(random_seed=random_seed)
    else:
        raise(Exception('datatype not defined correctly'))


    #K = 4

    np.random.randint()
    # with pm.Model() as model:
    #     alpha = pm.Normal('alpha', 1.14, 0.26, transform=pm.distributions.transforms.lowerbound(0))
    #     beta = pm.Normal('beta', 3.15, 0.89, transform=pm.distributions.transforms.lowerbound(0))
    #     alphas = pm.Gamma('gamma', alpha, beta, shape=O)
    #     obs = DirichletMultinomial('obs', sums, alphas, observed=s)

    from sys import argv
    if sys.argv[1] == 'implicit':
        run_implicit_model_gamma(s, datatype, random_seed)
    elif sys.argv[1] == 'explicit':
        run_explicit_model(s, datatype, random_seed)
    else:
        print('Error')

