import matplotlib.pyplot as plt
import pymc3 as pm
from pymc3.distributions import generate_samples, draw_values
from pymc3.distributions.dist_math import gammaln, bound, factln
import theano
import theano.tensor as tt
import numpy as np
import math


class SoftmaxRegression(pm.Model):
    def __init__(self, n_samples, n_covariates, n_otus, t0, data, covariates, nu=3, centered_lambda=True, centered_beta=True,
                 name='', model=None):
        super(SoftmaxRegression, self).__init__(name, model)
        self.S = n_samples
        self.C = n_covariates
        self.O = n_otus
        self.covariates = covariates#theano.shared(np.zeros((self.S, self.C)), 'covariates')
        self.data = data# theano.shared(np.ones((self.S, self.O), dtype=np.uint), 'data')
        self.n = self.data.sum(axis=1)
        #self.n = self.S
        pm.HalfCauchy('tau', t0)
        if centered_lambda:
            pm.HalfStudentT('lambda', nu=nu, mu=0, shape=(self.C, self.O))
        else:
            lamb_normal = pm.HalfNormal.dist(1, shape=(self.C, self.O))
            lamb_invGamma = pm.InverseGamma.dist(0.5 * nu, 0.5 * nu, shape=(self.C, self.O))
            pm.Deterministic('lambda', lamb_normal * tt.sqrt(lamb_invGamma))

        if centered_beta:
            pm.Normal('beta', 0, self['lambda']*self.tau, shape=(self.C, self.O))
        else:
            z = pm.Normal.dist(0, 1, shape=(self.C, self.O))
            pm.Deterministic('beta', z*self['lambda']*self.tau)

        self.coefficient_mask = theano.shared(np.ones((self.C, self.O), dtype=np.uint8), 'beta_mask')
        pm.Normal('alpha', 0, 10, shape=self.O)
        self.intermediate = tt.exp(self.alpha + tt.dot(self.covariates, self.beta*self.coefficient_mask))
        self.intermediate = tt.nnet.softmax(self.alpha + tt.dot(self.covariates, self.beta*self.coefficient_mask))
        #self.intermediate.name = 'intermediate'

        #dirichlet = pm.Dirichlet('dirchlet', self.intermediate, shape=(self.S, self.O))
        pm.Multinomial('counts', self.n, self.intermediate, shape=(self.S, self.O), observed=self.data)


if __name__ == '__main__':
    from data import get_simulated_data
    data = get_simulated_data('5O_5C_6p0_50S')
    S, O = data['counts'].shape
    S, C = data['covariates'].shape

    rseed = 35424353

    p0 = (data['beta'] != 0).sum().sum()
    sigma = 1
    t0 = (p0 / (C * O)) * (sigma / math.sqrt(S))
    print('p0 =', p0, 'sigma =', sigma, 'tau0 =', t0)
    model = SoftmaxRegression(S, C, O, t0, data['counts'], data['covariates'])

    with model:
        trace = pm.sample()

    pm.traceplot(trace)
    plt.show()
    print()