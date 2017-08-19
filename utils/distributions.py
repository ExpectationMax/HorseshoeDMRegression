import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt

from pymc3.distributions import generate_samples, draw_values
from pymc3.distributions.dist_math import gammaln, bound, factln
from pymc3.util import get_variable_name

from pymc3.math import logsumexp, logit, invlogit
from pymc3.distributions.transforms import t_stick_breaking
from pymc3.distributions.transforms import Transform

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
        return r'${} \sim \text{{Dirichlet}}(\mathit{{a}}={})$'.format(name, get_variable_name(a))


class Mixture(pm.Distribution):
    def __init__(self, w, comp_dists, *args, **kwargs):
        shape = kwargs.pop('shape', ())

        self.w = w = tt.as_tensor_variable(w)
        self.comp_dists = comp_dists

        defaults = kwargs.pop('defaults', [])

        dtype = kwargs.pop('dtype', theano.config.floatX)

        try:
            comp_means = self._comp_means()
            self.mean = (tt.dot(w, comp_means.T)).sum(axis=-1)

            if 'mean' not in defaults:
                defaults.append('mean')
        except AttributeError:
            pass

        super(Mixture, self).__init__(shape, dtype, defaults=defaults,
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


class StickBreaking(Transform):
    """Transforms K dimensional simplex space (values in [0,1] and sum to 1) to K - 1 vector of real values.
    Primarily borrowed from the STAN implementation.

    Parameters
    ----------
    eps : float, positive value
        A small value for numerical stability in invlogit.
    """

    name = "stickbreaking"

    def __init__(self, eps=0.0):
        self.eps = eps

    def forward(self, x_):
        x = x_.T
        # reverse cumsum
        x0 = x[:-1]
        s = tt.extra_ops.cumsum(x0[::-1], 0)[::-1] + x[-1]
        z = x0 / s
        Km1 = x.shape[0] - 1
        k = tt.arange(Km1)[(slice(None), ) + (None, ) * (x.ndim - 1)]
        eq_share = logit(1. / (Km1 + 1 - k).astype(str(x_.dtype)))
        y = logit(z) - eq_share
        return pm.floatX(y.T)

    def forward_val(self, x, point=None):
        return self.forward(x)

    def backward(self, y_):
        y = y_.T
        Km1 = y.shape[0]
        k = tt.arange(Km1)[(slice(None), ) + (None, ) * (y.ndim - 1)]
        eq_share = logit(1. / (Km1 + 1 - k).astype(str(y_.dtype)))
        z = invlogit(y + eq_share, self.eps)
        yl = tt.concatenate([z, tt.ones(y[:1].shape)])
        yu = tt.concatenate([tt.ones(y[:1].shape), 1 - z])
        S = tt.extra_ops.cumprod(yu, 0)
        x = S * yl
        return pm.floatX(x.T)

    def jacobian_det(self, y_):
        y = y_.T
        Km1 = y.shape[0]
        k = tt.arange(Km1)[(slice(None), ) + (None, ) * (y.ndim - 1)]
        eq_share = logit(1. / (Km1 + 1 - k).astype(str(y_.dtype)))
        yl = y + eq_share
        yu = tt.concatenate([tt.ones(y[:1].shape), 1 - invlogit(yl, self.eps)])
        S = tt.extra_ops.cumprod(yu, 0)
        return tt.sum(tt.log(S[:-1]) - tt.log1p(tt.exp(yl)) - tt.log1p(tt.exp(-yl)), 0).T

sensible_stick_breaking = StickBreaking(eps=pm.floatX(np.finfo(theano.config.floatX).eps))