import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt

from pymc3.math import logsumexp
from pymc3.distributions.dist_math import bound
from pymc3.distributions.transforms import t_stick_breaking

class MyMixture(pm.Distribution):
    def __init__(self, w, comp_dists, *args, **kwargs):
        shape = kwargs.pop('shape', ())

        self.w = w = tt.as_tensor_variable(w)
        self.comp_dists = comp_dists

        defaults = kwargs.pop('defaults', [])

        dtype = kwargs.pop('dtype', 'float64')

        try:
            comp_means = self._comp_means()
            self.mean = (tt.dot(w, comp_means.T)).sum(axis=-1)

            if 'mean' not in defaults:
                defaults.append('mean')
        except AttributeError:
            pass

        super(MyMixture, self).__init__(shape, dtype, defaults=defaults,
                                        *args, **kwargs)

    def _comp_logp(self, value):
        comp_dists = self.comp_dists
        try:
            value_ = value if value.ndim > 1 else tt.shape_padright(value)
            return comp_dists.logp(value_)
        except AttributeError:
            return tt.stack([comp_dist.logp(value) for comp_dist in comp_dists],
                            axis=1)

    def _comp_means(self):
        try:
            return tt.as_tensor_variable(self.comp_dists.mean)
        except AttributeError:
            return tt.stack([comp_dist.mean for comp_dist in self.comp_dists],
                            axis=1)

    def logp(self, value):
        w = self.w
        comp_logp = self._comp_logp(value)

        w_sum = w.sum(axis=-1)
        return bound(logsumexp(tt.log(w) + comp_logp.sum(axis=-1), axis=-1).sum(),
                     w >= 0, w <= 1, tt.allclose(w_sum, 1),
                     broadcast_conditions=False)

def sensible_stickbreacking():
    return 