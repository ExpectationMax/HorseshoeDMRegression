import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt
from dirichlet_multinomial import DirichletMultinomial

class DMRegressionModel(pm.Model):
    def __init__(self, n_samples, n_covariates, n_otus, t0, nu=3, centered_lambda=True, centered_beta=True,
                 name='', model=None):
        super(DMRegressionModel, self).__init__(name, model)
        self.S = n_samples
        self.C = n_covariates
        self.O = n_otus
        self.covariates = theano.shared(np.zeros((self.S, self.C)), 'covariates')
        self.data = theano.shared(np.ones((self.S, self.O), dtype=np.uint), 'data')
        self.n = self.data.sum(axis=-1)

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
        self.intermediate = tt.exp(self.alpha[np.newaxis, :] + tt.dot(self.covariates, self.beta*self.coefficient_mask))
        self.intermediate.name = 'intermediate'
        DirichletMultinomial('counts', self.n, self.intermediate, shape=(self.S, self.O), observed=self.data)

    def set_counts_and_covariates(self, counts, covariates):
        if counts.shape[0] == covariates.shape[0]:
            self.data.set_value(counts)
            self.covariates.set_value(covariates)
        else:
            raise ValueError('Counts and covariates must have same sample dimension ({} vs {})'
                             .format(counts.shape[0], covariates.shape[0]))

    def set_counts(self, counts, borrow=False):
        data_shape = self.data.shape.eval()
        if counts.shape[0] == data_shape[0] and counts.shape[1] == data_shape[1]:
            self.data.set_value(counts, borrow=borrow)
        else:
            raise ValueError('Counts must have same dimension ({}x{} vs {}x{})'
                             .format(counts.shape[0], counts.shape[1], data_shape[0], data_shape[1]))

    def set_coefficient_mask(self, coefficient_mask):
        if coefficient_mask.shape[0] == self.C and coefficient_mask.shape[1] == self.O:
            self.coefficient_mask.set_value(coefficient_mask)
        else:
            raise ValueError('Coefficient mask must have dimensions ({}x{})'.format(self.C, self.O))


class MaskableDMRegressionModel(pm.Model):
    def __init__(self, n_samples, n_covariates, n_otus, t0, nu=3, centered_lambda=True, centered_beta=True,
                 name='', mask=None, model=None):
        super(MaskableDMRegressionModel, self).__init__(name, model)
        self.S = n_samples
        self.C = n_covariates
        self.O = n_otus
        self.covariates = theano.shared(np.zeros((self.S, self.C)), 'covariates')
        self.data = theano.shared(np.ones((self.S, self.O), dtype=np.uint), 'data')
        self.n = self.data.sum(axis=-1)
        if mask is None:
            self.ncoefficients = self.C * self.O
        else:
            self.ncoefficients = int(np.sum(mask))
            self.unmasked_coefficients = np.where(mask)

        pm.HalfCauchy('tau', t0)
        if centered_lambda:
            pm.HalfStudentT('lambda', nu=nu, mu=0, shape=(self.ncoefficients,))
        else:
            lamb_normal = pm.HalfNormal.dist(1, shape=(self.ncoefficients,))
            lamb_invGamma = pm.InverseGamma.dist(0.5 * nu, 0.5 * nu, shape=self.ncoefficients)
            pm.Deterministic('lambda', lamb_normal * tt.sqrt(lamb_invGamma))

        if centered_beta:
            prebeta = pm.Normal('prebeta', 0, self['lambda']*self.tau, shape=(self.ncoefficients,))
            if mask is not None:
                reshaped_beta = tt.zeros((self.C, self.O))
                pm.Deterministic('beta', tt.set_subtensor(reshaped_beta[self.unmasked_coefficients], prebeta))
            else:
                pm.Deterministic('beta', tt.reshape(prebeta, (self.C, self.O)))
        else:
            z = pm.Normal.dist(0, 1, shape=(self.ncoefficients,))
            pm.Deterministic('beta', z*self['lambda']*self.tau)

        self.coefficient_mask = theano.shared(np.ones((self.C, self.O), dtype=np.uint8), 'beta_mask')
        pm.Normal('alpha', 0, 10, shape=self.O)
        self.intermediate = tt.exp(self.alpha[np.newaxis, :] + tt.dot(self.covariates, self.beta*self.coefficient_mask))
        #self.intermediate.name = 'intermediate'
        DirichletMultinomial('counts', self.n, self.intermediate, shape=(self.S, self.O), observed=self.data)

    def set_counts_and_covariates(self, counts, covariates):
        if counts.shape[0] == covariates.shape[0]:
            self.data.set_value(counts)
            self.covariates.set_value(covariates)
        else:
            raise ValueError('Counts and covariates must have same sample dimension ({} vs {})'
                             .format(counts.shape[0], covariates.shape[0]))

    def set_counts(self, counts, borrow=False):
        data_shape = self.data.shape.eval()
        if counts.shape[0] == data_shape[0] and counts.shape[1] == data_shape[1]:
            self.data.set_value(counts, borrow=borrow)
        else:
            raise ValueError('Counts must have same dimension ({}x{} vs {}x{})'
                             .format(counts.shape[0], counts.shape[1], data_shape[0], data_shape[1]))

    def set_coefficient_mask(self, coefficient_mask):
        if coefficient_mask.shape[0] == self.C and coefficient_mask.shape[1] == self.O:
            self.coefficient_mask.set_value(coefficient_mask)
        else:
            raise ValueError('Coefficient mask must have dimensions ({}x{})'.format(self.C, self.O))


