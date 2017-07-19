import numpy as np
import theano
import theano.tensor as tt
from collections import defaultdict
from pymc3 import modelcontext
from tqdm import tqdm
from numpy.random import randint, seed
from pymc3.distributions import draw_values
from dirichlet_multinomial import DirichletMultinomial
from scipy.special import gammaln, psi
from tqdm import tqdm
from joblib import Parallel, delayed
from fast_parameter_sampling import draw_values_fast

def _dirichlet_KL_divergence():
    pre_alphas = tt.tensor3('alphas')
    #x.tag.test_value = np.abs(np.random.rand(40, )
    betas = tt.tensor4('betas')
    alphas = tt.tile(pre_alphas, [betas.shape[0]] + [1] * (betas.ndim - 1))
    alpha0 = tt.sum(alphas, axis=-1)
    beta0 = tt.sum(betas, axis=-1)
    res = tt.gammaln(alpha0) - tt.sum(tt.gammaln(alphas), axis=-1) -\
          tt.gammaln(beta0) + tt.sum(tt.gammaln(betas), axis=-1) +\
          tt.sum((alphas - betas)*(tt.psi(alphas) - tt.shape_padright(tt.psi(alpha0))), axis=-1)
    return theano.function([pre_alphas, betas], res)

def _dirichlet_KL_divergence_fast():
    alphas = tt.tensor3('alphas')
    #x.tag.test_value = np.abs(np.random.rand(40, )
    betas = tt.tensor4('betas')
    alpha0 = tt.sum(alphas, axis=-1)
    beta0 = tt.sum(betas, axis=-1)
    res = tt.shape_padleft(tt.gammaln(alpha0)) - tt.shape_padleft(tt.sum(tt.gammaln(alphas), axis=-1)) -\
          tt.gammaln(beta0) + tt.sum(tt.gammaln(betas), axis=-1) +\
          tt.sum((tt.shape_padleft(alphas) - betas)*tt.shape_padleft(tt.psi(alphas) - tt.shape_padright(tt.psi(alpha0))), axis=-1)
    return theano.function([alphas, betas], res)

dirichlet_KL_divergence_theano = _dirichlet_KL_divergence_fast()

def dirichlet_KL_divergence_np(alphas, betas):
    alpha0 = np.sum(alphas, axis=-1)
    beta0 = np.sum(betas, axis=-1)
    return gammaln(alpha0) - np.sum(gammaln(alphas), axis=-1) -\
           gammaln(beta0) + np.sum(gammaln(betas), axis=-1) +\
           np.sum((alphas - betas)*(psi(alphas) - psi(alpha0)[..., np.newaxis]), axis=-1)


def sample_RV_using_trace(RV, trace):
    return np.concatenate([draw_values([RV], point=p) for p in trace], axis=0)


def KL_divergence_to_masked_model(trace, model, mask, unmasked_parameters=None):
    if unmasked_parameters is None:
        unmasked_parameters = sample_from_model_using_mask(trace, model, np.ones_like(mask, dtype=np.uint8))

    masked_parameters = sample_from_model_using_mask(trace, model, mask)
    return dirichlet_KL_divergence_theano(unmasked_parameters, masked_parameters).mean()


def sample_from_model_using_mask(trace, model, mask):
    model.set_coefficient_mask(mask)
    return draw_values_fast(model.intermediate, trace) #sample_RV_using_trace(model.intermediate, trace)


def compute_multiple_divergences(xs, ys, trace, model, mask, unmasked_parameters):
    accumulated_parameters = []
    for x, y in zip(xs, ys):
        mymask = mask.copy()
        mymask[x, y] = 1
        accumulated_parameters.append(sample_from_model_using_mask(trace, model, mymask))

    accumulated_parameters = np.array(accumulated_parameters)
    theano_res = dirichlet_KL_divergence_theano(unmasked_parameters, accumulated_parameters)
    return theano_res.mean(axis=-1).mean(axis=-1)


def compute_single_divergence(x, y, trace, model, mask, unmasked_parameters):
    mymask = mask.copy()
    mymask[x, y] = 1
    return KL_divergence_to_masked_model(trace, model, mymask, unmasked_parameters=unmasked_parameters)

def mask_next_value(mask, trace, model, full_model_parameters=None, parallel_pool=None):
    mask = mask.copy()
    positions = np.where(mask == 0)
    kl_divergences = np.ma.masked_array(np.zeros_like(mask), mask=mask, dtype=np.float64)
    n_arrays = 10 if len(positions[0]) > 10 else len(positions[0])
    if parallel_pool is None:
            #for xs, ys in zip(np.array_split(positions[0], n_arrays), np.array_split(positions[1], n_arrays)):
        results = compute_multiple_divergences(positions[0], positions[1], trace, model, mask, full_model_parameters)
        kl_divergences[positions] = results
    else:
        results = parallel_pool(
            delayed(compute_multiple_divergences)(xs, ys, trace, model, mask, unmasked_parameters=full_model_parameters)
            for xs, ys in zip(np.array_split(positions[0], n_arrays), np.array_split(positions[1], n_arrays)))
        results = np.concatenate(results)
        kl_divergences[positions] = results

    next_index = np.unravel_index(np.argmin(kl_divergences), kl_divergences.shape)
    mask[next_index] = 1
    return kl_divergences.min(), mask


def execute_variable_selection(trace, model, njobs=1):
    masks = []
    kl_divergences = []
    full_model_parameters = sample_RV_using_trace(model.intermediate, trace)
    mask_shape = model.coefficient_mask.shape.eval()
    previous_mask = np.zeros(mask_shape, dtype=np.uint8)
    if njobs == 1:
        parallel = None
    else:
        parallel = Parallel(n_jobs=njobs, max_nbytes=None)

    for i in tqdm(range(mask_shape[0]*mask_shape[1]), total=mask_shape[0]*mask_shape[1]):
        divergence, mask = mask_next_value(previous_mask, trace, model, full_model_parameters=full_model_parameters, parallel_pool=parallel)
        masks.append(mask)
        kl_divergences.append(divergence)
        previous_mask = mask

    return masks, kl_divergences




if __name__ == '__main__':
    import pymc3 as pm
    import pickle
    import seaborn as sns

    with open('NUTS_sampling.pck', 'rb') as f:
        trace = pickle.load(f)

    with open('NUTS_model.pck', 'rb') as f:
        model = pickle.load(f)

    masks, kl_divergences = execute_variable_selection(trace, model)
    #res1, future_samples = compute_mean_KL_divergence_to_full_model(trace, model, np.zeros((model.C, model.O), dtype=np.uint8))
    #res2 = compute_mean_KL_divergence_to_full_model2(trace, model, np.zeros((model.C, model.O), dtype=np.uint8), future_samples=future_samples)
    print(res1)