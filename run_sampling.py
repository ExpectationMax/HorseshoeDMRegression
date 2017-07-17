import pymc3 as pm
import math
import dm_regression_model
import seaborn as sns
import pickle
import numpy as np
from data import get_simulated_data

if __name__ == '__main__':
    data = get_simulated_data()
    S, O = data['counts'].shape
    S, C = data['covariates'].shape

    p0 = 6
    sigma = 1
    t0 = (p0 / (C * O)) * (sigma / math.sqrt(S))
    print('p0 =', p0, 'sigma =', sigma, 'tau0 =', t0)
    mask = np.zeros((C, O), dtype=np.uint8)
    mask[0, 0] = 1
    mask[1, 1] = 1
    model = dm_regression_model.MaskableDMRegressionModel(S, C, O, t0, mask=mask)
    model.set_counts_and_covariates(data['counts'], data['covariates'])


    with open('NUTS_model.pck', 'wb') as f:
        pickle.dump(model, f)
        print('Stored model')

    with model:
        trace = pm.sample(draws=500, njobs=1, tune=1500, nuts_kwargs={'target_accept': 0.9})

    with open('NUTS_sampling.pck', 'wb') as f:
        pickle.dump(trace, f)
        print('Stored trace')

    pm.traceplot(trace)
    sns.plt.savefig('NUTS_traceplot.pdf')


    print()
    #masked = pm.sample_ppc(trace, 500, model=model, random_seed=47583793)
