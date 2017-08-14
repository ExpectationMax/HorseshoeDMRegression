import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt
from dirichlet_multinomial import DirichletMultinomial

class DMRegressionModel(pm.Model):
    def __init__(self, n_samples, n_covariates, n_otus, t0, nu=3, centered=True, cauchy=True, alpha_init=None,
                 beta_init=None, name='', model=None):
        super(DMRegressionModel, self).__init__(name, model)
        self.S = n_samples
        self.C = n_covariates
        self.O = n_otus
        self.covariates = theano.shared(np.zeros((self.S, self.C)), 'covariates')
        self.data = theano.shared(np.ones((self.S, self.O), dtype=np.uint), 'data')
        self.n = self.data.sum(axis=-1)

        if beta_init is None:
            z_init = None
        else:
            z_init = beta_init/beta_init.std()

        if cauchy:
            if centered:
                pm.HalfCauchy('tau', t0)
            else:
                tau_normal = pm.HalfNormal('tau-normal', t0)
                tau_invGamma = pm.InverseGamma('tau-invGamma', alpha=0.5, beta=0.5, testval=(0.5/(0.5+1)))
                pm.Deterministic('tau', tau_normal*tt.sqrt(tau_invGamma))
        else:
            pm.HalfNormal('tau', t0)

        if centered:
            pm.HalfStudentT('lambda', nu=nu, mu=0, shape=(self.C, self.O))
        else:
            lamb_normal = pm.HalfNormal('lamb-Normal', sd=1, shape=(self.C, self.O))
            lamb_invGamma = pm.InverseGamma('lamb-invGamma', alpha=0.5 * nu, beta=0.5 * nu, shape=(self.C, self.O), testval=np.full((self.C, self.O), (0.5*nu)/(0.5*nu + 1)))
            pm.Deterministic('lambda', lamb_normal * tt.sqrt(lamb_invGamma))

        if centered:
            pm.Normal('beta', mu=0, sd=self['lambda']*self.tau, shape=(self.C, self.O), testval=beta_init)
        else:
            z = pm.Normal('z', mu=0, sd=1, shape=(self.C, self.O), testval=z_init)
            pm.Deterministic('beta', z*self['lambda']*self.tau)

        self.coefficient_mask = theano.shared(np.ones((self.C, self.O), dtype=np.uint8), 'beta_mask')
        pm.Normal('alpha', 0, 10, shape=self.O, testval=alpha_init)
        self.intermediate = tt.exp(self.alpha + tt.dot(self.covariates, self.beta*self.coefficient_mask))
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


class DMRegressionMVNormalModel(pm.Model):
    def __init__(self, countdata, metadata, t0, nu=3, centered=True, cauchy=True, alpha_init=None,
                 beta_init=None, z_init=None, chol_init=None, name='', model=None):
        super(DMRegressionMVNormalModel, self).__init__(name, model)
        self.S, self.O = countdata.shape
        self.S, self.C = metadata.shape
        self.covariates = metadata
        self.data = countdata.astype(np.uint)
        self.n = self.data.sum(axis=-1)

        if cauchy:
            if centered:
                pm.HalfCauchy('tau', t0)
            else:
                tau_normal = pm.HalfNormal('tau-normal', t0)
                tau_invGamma = pm.InverseGamma('tau-invGamma', alpha=0.5, beta=0.5, testval=(0.5/(0.5+1)))
                pm.Deterministic('tau', tau_normal*tt.sqrt(tau_invGamma))
        else:
            pm.HalfNormal('tau', t0)

        if centered:
            pm.HalfStudentT('lambda', nu=nu, mu=0, shape=(self.C, self.O))
        else:
            lamb_normal = pm.HalfNormal('lamb-Normal', sd=1, shape=(self.C, self.O))
            lamb_invGamma = pm.InverseGamma('lamb-invGamma', alpha=0.5 * nu, beta=0.5 * nu, shape=(self.C, self.O), testval=np.full((self.C, self.O), (0.5*nu)/(0.5*nu + 1)))
            pm.Deterministic('lambda', lamb_normal * tt.sqrt(lamb_invGamma))

        pm.Normal('beta', mu=0, sd=self['lambda']*self.tau, shape=(self.C, self.O), testval=beta_init)

        pm.Normal('alpha', 0, 10, shape=self.O, testval=alpha_init) # this is basically B0

        sd_dist = pm.HalfCauchy.dist(beta=2.5, shape=self.O)
        chol_packed = pm.LKJCholeskyCov('chol_packed', n=self.O, eta=1, sd_dist=sd_dist, testval=chol_init)
        chol = pm.expand_packed_triangular(self.O, chol_packed)
        z_raw = pm.Normal('z_raw', mu=0, sd=1, shape=(self.S, self.O))
        z = pm.Deterministic('z', self.alpha[np.newaxis, :] + tt.dot(chol, z_raw.T).T)
        #vals = pm.MvNormal('z', mu=self.alpha, chol=chol, testval=z_init, shape=(self.S, self.O))  #

        self.intermediate = tt.exp(z + tt.dot(self.covariates, self.beta))
        DirichletMultinomial('counts', self.n, self.intermediate, shape=(self.S, self.O), observed=self.data)


class DMRegressionMixed(pm.Model):
    def __init__(self, countdata, metadata, patients, t0, nu=3, centered=True, cauchy=True,
                 name='', model=None):
        super(DMRegressionMixed, self).__init__(name, model)
        self.S, self.O = countdata.shape
        self.S, self.C = metadata.shape

        self.covariates = metadata
        self.data = countdata.astype(int)
        #if patients is None:
        #    patients = np.arange(self.S)

        unique_patients, patient_indexes = np.unique(patients, return_inverse=True)
        self.n_patients = len(unique_patients)
        self.patientindexes = patient_indexes
        self.n = self.data.sum(axis=-1)

        if cauchy:
            if centered:
                pm.HalfCauchy('tau', t0)
            else:
                tau_normal = pm.HalfNormal('tau-normal', t0)
                tau_invGamma = pm.InverseGamma('tau-invGamma', 0.5, 0.5, testval=(0.5/(0.5+1)))
                pm.Deterministic('tau', tau_normal*tt.sqrt(tau_invGamma))
        else:
            pm.HalfNormal('tau', t0)

        if centered:
            pm.HalfStudentT('lambda', nu=nu, mu=0, shape=(self.C, self.O))
        else:
            lamb_normal = pm.HalfNormal('lamb-Normal', 1, shape=(self.C, self.O))
            lamb_invGamma = pm.InverseGamma('lamb-invGamma', 0.5 * nu, 0.5 * nu, shape=(self.C, self.O), testval=np.full((self.C, self.O), (0.5*nu)/(0.5*nu + 1)))
            pm.Deterministic('lambda', lamb_normal * tt.sqrt(lamb_invGamma))

        if centered:
            pm.Normal('beta', 0, self['lambda']*self.tau, shape=(self.C, self.O))
        else:
            z = pm.Normal('z', 0, 1, shape=(self.C, self.O))
            pm.Deterministic('beta', z*self['lambda']*self.tau)

        pm.Normal('alpha', 0, 10, shape=self.O)
        deviation = pm.HalfCauchy('sigma_alphas', 1)
        alpha_offsets = pm.Normal('alpha_offsets', mu=0, sd=1, shape=(self.n_patients, self.O))

        pm.Deterministic('alphas', self.alpha[np.newaxis, :] + alpha_offsets[self.patientindexes]*deviation)

        self.intermediate = tt.exp(self.alphas + tt.dot(self.covariates, self.beta))
        DirichletMultinomial('counts', self.n, self.intermediate, shape=(self.S, self.O), observed=self.data)


class SoftmaxRegression(pm.Model):
    def __init__(self, countdata, metadata, patients, t0, nu=3, centered=True, cauchy=True,
                 name='', model=None):
        super(SoftmaxRegression, self).__init__(name, model)
        self.S, self.O = countdata.shape
        self.S, self.C = metadata.shape

        self.covariates = metadata
        self.data = countdata.astype(int)
        if patients is None:
            patients = np.arange(self.S)

        unique_patients, patient_indexes = np.unique(patients, return_inverse=True)
        self.n_patients = len(unique_patients)
        self.patientindexes = patient_indexes
        self.n = self.data.sum(axis=-1)

        if cauchy:
            if centered:
                pm.HalfCauchy('tau', t0)
            else:
                tau_normal = pm.HalfNormal('tau-normal', 1)
                tau_invGamma = pm.InverseGamma('tau-invGamma', 0.5, 0.5, testval=(0.5 / (0.5 + 1)))
                pm.Deterministic('tau', tau_normal * t0 * tt.sqrt(tau_invGamma))
        else:
            pm.HalfNormal('tau', t0)

        if centered:
            pm.HalfStudentT('lambda', nu=nu, mu=0, shape=(self.C, self.O))
        else:
            lamb_normal = pm.HalfNormal('lamb-Normal', 1, shape=(self.C, self.O))
            lamb_invGamma = pm.InverseGamma('lamb-invGamma', 0.5 * nu, 0.5 * nu, shape=(self.C, self.O),
                                            testval=np.full((self.C, self.O), (0.5 * nu) / (0.5 * nu + 1)))
            pm.Deterministic('lambda', lamb_normal * tt.sqrt(lamb_invGamma))

        if centered:
            pm.Normal('beta', 0, self['lambda'] * self.tau, shape=(self.C, self.O))
        else:
            z = pm.Normal('z', 0, 1, shape=(self.C, self.O))
            pm.Deterministic('beta', z * self['lambda'] * self.tau)

        pm.Normal('alpha', 0, 5, shape=self.O)

        pm.Deterministic('composition', tt.nnet.softmax(self.alpha + tt.dot(self.covariates, self.beta)))
        precisions = pm.HalfCauchy('precision', 5, shape=self.n_patients)

        pm.Deterministic('alphas', self.composition*(precisions[self.patientindexes][:, np.newaxis]))

        DirichletMultinomial('counts', self.n, self.alphas, shape=(self.S, self.O), observed=self.data)


class MaskableDMRegressionModel(DMRegressionModel):
    def __init__(self, n_samples, n_covariates, n_otus, t0, nu=3, centered_lambda=True, centered_beta=True, cauchy=True,
                 name='', mask=None, model=None):
        super(DMRegressionModel, self).__init__(name, model)
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

        if cauchy:
            pm.HalfCauchy('tau', t0)
        else:
            pm.HalfNormal('tau', t0)

        if centered_lambda:
            pm.HalfStudentT('lambda', nu=nu, mu=0, shape=(self.ncoefficients,))
        else:
            lamb_normal = pm.HalfNormal.dist(1, shape=(self.ncoefficients,))
            lamb_invGamma = pm.InverseGamma.dist(0.5 * nu, 0.5 * nu, shape=self.ncoefficients)
            pm.Deterministic('lambda', lamb_normal * tt.sqrt(lamb_invGamma))

        if centered_beta:
            prebeta = pm.Normal('prebeta', 0, self['lambda']*self.tau, shape=self.ncoefficients)
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
        self.intermediate = tt.exp(self.alpha + tt.dot(self.covariates, self.beta*self.coefficient_mask))
        DirichletMultinomial('counts', self.n, self.intermediate, shape=(self.S, self.O), observed=self.data)


class DMRegressionModelNonsparseImplicit(DMRegressionModel):
    def __init__(self, n_samples, n_covariates, n_otus, data, covariates, name='', model=None):
        super(DMRegressionModel, self).__init__(name, model)
        self.S = n_samples
        self.C = n_covariates
        self.O = n_otus
        self.covariates = covariates #theano.shared(np.zeros((self.S, self.C)), 'covariates')
        self.data = data #theano.shared(np.ones((self.S, self.O), dtype=np.uint), 'data')
        self.n = self.data.sum(axis=1)

        pm.Normal('beta', 0, 10, shape=(self.C, self.O))
        self.coefficient_mask = theano.shared(np.ones((self.C, self.O), dtype=np.uint8), 'beta_mask')
        pm.Normal('alpha', 0, 10, shape=self.O)
        self.intermediate = tt.exp(self.alpha + tt.dot(self.covariates, self.beta*self.coefficient_mask))

        DirichletMultinomial('counts', self.n, self.intermediate, shape=(self.S, self.O), observed=self.data)

