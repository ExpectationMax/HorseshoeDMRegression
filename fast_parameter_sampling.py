import numbers
import numpy as np
import theano.tensor as tt
from pymc3.distributions.distribution import _compile_theano_function

def draw_values_fast(param, points=None):
    from pymc3.model import get_named_nodes
    """
    Draw (fix) parameter values. Handles a number of cases:

        1) The parameter is a scalar
        2) The parameter is an *RV

            a) parameter can be fixed to the value in the point
            b) parameter can be fixed by sampling from the *RV
            c) parameter can be fixed using tag.test_value (last resort)

        3) The parameter is a tensor variable/constant. Can be evaluated using
        theano.function, but a variable may contain nodes which

            a) are named parameters in the point
            b) are *RVs with a random method

    """
    # Distribution parameters may be nodes which have named node-inputs
    # specified in the point. Need to find the node-inputs to replace them.
    givens = {}
    if hasattr(param, 'name'):
        named_nodes = get_named_nodes(param)
        if param.name in named_nodes:
            named_nodes.pop(param.name)
        for name, node in named_nodes.items():
            if not isinstance(node, (tt.sharedvar.SharedVariable,
                                     tt.TensorConstant)):
                givens[name] = (node, _draw_value(node, points=points))

    return _draw_value(param, points=points, givens=givens.values())

def _draw_value(param, points=None, givens=None):
    """Draw a random value from a distribution or return a constant.

    Parameters
    ----------
    param : number, array like, theano variable or pymc3 random variable
        The value or distribution. Constants or shared variables
        will be converted to an array and returned. Theano variables
        are evaluated. If `param` is a pymc3 random variables, draw
        a new value from it and return that, unless a value is specified
        in `point`.
    point : dict, optional
        A dictionary from pymc3 variable names to their values.
    givens : dict, optional
        A dictionary from theano variables to their values. These values
        are used to evaluate `param` if it is a theano variable.
    """
    if isinstance(param, numbers.Number):
        return param
    elif isinstance(param, np.ndarray):
        return param
    elif isinstance(param, tt.TensorConstant):
        return param.value
    elif isinstance(param, tt.sharedvar.SharedVariable):
        return param.get_value()
    elif isinstance(param, tt.TensorVariable):
        if points is not None and hasattr(param, 'model') and param.name in points.varnames:
            return points[param.name]
        elif hasattr(param, 'random') and param.random is not None:
            return np.array([param.random(point=p, size=None) for p in points])
        else:
            if givens:
                variables, values = list(zip(*givens))
            else:
                variables = values = []
            func = _compile_theano_function(param, variables)
            return np.array([func(*v) for v in zip(*values)])
    else:
        raise ValueError('Unexpected type in draw_value: %s' % type(param))

if __name__ == '__main__':
    import pickle
    from projection_predictive_variable_selection import sample_RV_using_trace
    with open('NUTS_sampling.pck', 'rb') as f:
        trace = pickle.load(f)

    with open('NUTS_model.pck', 'rb') as f:
        model = pickle.load(f)

    def method1():
        values = draw_values_fast(model.intermediate, trace)

    def method2():
        values2 = sample_RV_using_trace(model.intermediate, trace)
    print('test')