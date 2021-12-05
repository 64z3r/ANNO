# -*- coding: utf-8 -*-
r"""
"""

### ======================================================================== ###

import cost
import sample
import format

import theano      as _theano
import numpy       as _np
import scipy       as _sp

import collections as _collections

from _misc import Mapped       as _Mapped
from _misc import shared_as    as _shared_as
from _misc import flatten      as _flatten
from _misc import floor_vector as _floor_vector
from _misc import floor_matrix as _floor_matrix

### ======================================================================== ###

class Context(object):
    r"""
    Implementation of a context object that can store optimization relevant
    variables, basic routines and relations that are necessary for compiling
    theano functions.
    """

    ### -------------------------------------------------------------------- ###

    def __init__(self, inputs=[], wrt=[], givens={}, updates={}, cc=[], context=None):
        r"""
        Parameters
        ----------
        inputs : [*theano variable*, ...]
            Variables that are used as parameters when compiling a theano
            function (see :func:`theano.function`).
        wrt : [*shared theano variable*, ...]
            List of parameters that need to be optimized. An optimization
            method will use this list of parameters to compute the gradient
            with respect to these parameters to determine the resiudal that
            minimizes the error objective (see :func:`theano.grad`).
        givens : {*theano variable* : *theano variable*, ...}
            Replacement mapping of variables to other input variables that are
            part of the symbolic graph that represents the output of a
            function. This is necessary if you want to replace an input
            variable that would be used as a parameter of a function with an
            other theano expression (see :func:`theano.function`).
        updates : {*theano variable* : *theano variable*, ...}
            Update mapping of variables that will be updated after each call to
            the function (see :func:`theano.function`).
        cc : [*theano variable*, ...]
            List of parameters that are considered to be constant when
            computing the gradient of an optimization objective (see
            :func:`theano.grad`).
        context : :class:`Context`
            Context object that is used to extend the current optimization
            context with optimization relevant variables.
        """

        # update context attributes if possible
        if context is not None:

            # update function related attributes
            inputs  = context.inputs          + inputs
            givens  = context.givens.items()  + givens.items()
            updates = context.updates.items() + updates.items()

            # update optimization related attributes
            wrt = context.wrt + wrt
            cc  = context.cc  + cc

        # set function related attributes
        self.inputs  = inputs
        self.givens  = givens
        self.updates = updates

        # set optimization related attributes
        self.wrt = wrt
        self.cc  = cc

    ### -------------------------------------------------------------------- ###

    def __repr__( self ):
        return "Context(inputs={self.inputs!r}, "  \
                          "wrt={self.wrt!r}, "     \
                       "givens={self.givens!r}, "  \
                      "updates={self.updates!r}, " \
                           "cc={self.cc!r})".format(self=self)

    ### -------------------------------------------------------------------- ###

    inputs = property(doc="list of inputs with respect to some symbolic "
                          "expression (see :func:`theano.function`)")

    @inputs.getter
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = list(value)

    @inputs.deleter
    def inputs(self):
        raise AttributeError("cannot delete attribute 'inputs'")

    ### -------------------------------------------------------------------- ###

    givens = property(doc="dictionary of given inputs with respect to some "
                          "symbolic expression (see :func:`theano.function`)")

    @givens.getter
    def givens(self):
        return self._givens

    @givens.setter
    def givens(self, value):
        self._givens = dict(value)

    @givens.deleter
    def givens( self ):
        raise AttributeError("cannot delete attribute 'givens'")

    ### -------------------------------------------------------------------- ###

    updates = property(doc="dictionary of update-mappings that are going to "
                           "be applied after each function call (see :func:"
                           "`theano.function`)")

    @updates.getter
    def updates(self):
        return self._updates

    @updates.setter
    def updates(self, value):
        self._updates = _collections.OrderedDict(value)

    @updates.deleter
    def updates(self):
        raise AttributeError("cannot delete attribute 'updates'")

    ### -------------------------------------------------------------------- ###

    wrt = property(doc="list of parameters that have to be optimized (see "
                       ":func:`theano.grad`)")

    @wrt.getter
    def wrt(self):
        return self._wrt

    @wrt.setter
    def wrt(self, value):
        self._wrt = list(value)

    @wrt.deleter
    def wrt(self):
        raise AttributeError("cannot delete attribute 'wrt'")

    ### -------------------------------------------------------------------- ###

    cc = property(doc="list of parameters that should be considered constant "
                      "during the optimization (see :func:`theano.grad`)")

    @cc.getter
    def cc(self):
        return self._cc

    @cc.setter
    def cc(self, value):
        self._cc = list(value)

    @cc.deleter
    def cc(self):
        raise AttributeError("cannot delete attribute 'cc'")

    ### -------------------------------------------------------------------- ###

### ======================================================================== ###

class Objective(Context):
    r"""
    Implementation of an optimization objective that is used as a parameter
    for various optimization methods. An optimization objective can store
    optimization relavant variables -- just like :class:`Context` -- but it
    provides some additional methods for updating optimization specific
    parameters, see :func:`Objective.get`, :func:`Obective.set`,
    :func:`Objective.pull` and :func:`Objective.push`.
    """

    ### -------------------------------------------------------------------- ###

    def __init__(self, costs, outputs=[], inputs=[], wrt=[], givens={}, updates={}, cc=[], context=None):
        r"""
        Parameters
        ----------
        costs : [*theano variable*, ...]
            List of costs that represent the error objective of an optimization
            problem (see :func:`theano.grad`).
        outputs : [*theano variable*, ...]
            List of outputs that represent the outputs of a model that needs to
            be optimized to satisfy the optimization objective (see
            :func:`theano.function`).
        inputs : [*theano variable*, ...]
            Variables that are used as parameters when compiling a theano
            function (see :func:`theano.function`).
        wrt : [*shared theano variable*, ...]
            List of parameters that need to be optimized. An optimization
            method will use this list of parameters to compute the gradient
            with respect to these parameters to determine the resiudal that
            minimizes the error objective (see :func:`theano.grad`).
        givens : {*theano variable* : *theano variable*, ...}
            Replacement mapping of variables to other input variables that are
            part of the symbolic graph that represents the output of a
            function. This is necessary if you want to replace an input
            variable that would be used as a parameter of a function with an
            other theano expression (see :func:`theano.function`).
        updates : {*theano variable* : *theano variable*, ...}
            Update mapping of variables that will be updated after each call to
            the function (see :func:`theano.function`).
        cc : [*theano variable*, ...]
            List of parameters that are considered to be constant when
            computing the gradient of an optimization objective (see
            :func:`theano.grad`).
        context : :class:`Context`
            Context object that is used to extend the current optimization
            context with optimization relevant variables.
        """

        # set context related attributes
        super(Objective, self).__init__(inputs, wrt, givens, updates, cc, context)

        # set attributes to objective
        self.costs   = costs
        self.outputs = outputs

        # define basic types w.r.t. number of dimensions
        types = [_theano.tensor.scalar,
                 _theano.tensor.vector,
                 _theano.tensor.matrix,
                 _theano.tensor.tensor3,
                 _theano.tensor.tensor4]

        # define input parameters that match the optimization parameters
        self._in_params = [types[p.ndim]("input" + (":%s" % p.name if p.name else "")) for p in self.wrt]

        # compile theano functions for parameter manipulations
        self._get_list_params = _theano.function([], self.wrt)
        self._set_list_params = _theano.function(self._in_params, updates=zip(self.wrt, self._in_params))
        self._put_list_params = _theano.function(self._in_params, updates=zip(self.wrt, map(lambda a, b: a + b, self.wrt, self._in_params)))

        # define flatten vector and container for it's split
        # and reshaped result
        self._in_vector = _theano.tensor.vector("v", dtype=_theano.config.floatX)
        self._as_params = []

        # re-assign flatten vector in order to split
        v = self._in_vector

        # split and reshape flatten vector to match number
        # and shape of parameters
        for p in self.wrt:
            x, v = v[:p.shape.prod()], v[p.shape.prod():]
            self._as_params.append(x.reshape(p.shape))

        # compile flatten theano functions for parameter manipulations
        self._get_flat_params = _theano.function([], _flatten(self.wrt))
        self._set_flat_params = _theano.function([self._in_vector], updates=zip(self.wrt, self._as_params))
        self._put_flat_params = _theano.function([self._in_vector], updates=zip(self.wrt, map(lambda a, b: a + b, self.wrt, self._as_params)))

    ### -------------------------------------------------------------------- ###

    def __repr__( self ):
        return "Objective(const={self.costs!r}, "   \
                       "outputs={self.outputs!r}, " \
                        "inputs={self.inputs!r}, "  \
                           "wrt={self.wrt!r}, "     \
                        "givens={self.givens!r}, "  \
                       "updates={self.updates!r}, " \
                            "cc={self.cc!r})".format(self=self)

    ### -------------------------------------------------------------------- ###

    def get(self, flatten=True):
        r"""
        Method that returns all parameters as a vectorized numpy array or a
        list of numpy arrays.

        Parameters
        ----------
        flatten : *bool*
            If this method should return flatten (i.e. vectorized) parameters
            or a list of parameters with the original shapes.

        Returns
        -------
        out : *numpy array* or [*numpy array*, ...]
            The parameters.
        """

        return self._get_flat_params() if flatten else self._get_list_params()

    def set(self, x):
        r"""
        Method that sets all parameters to the values of a vectorized numpy
        array or a list of numpy arrays.

        Parameters
        ----------
        x : *numpy array* or [*numpy array*, ...]
            Parameters that need to be set.
        """

        if isinstance(x, _np.ndarray):
            self._set_flat_params( x)
        else:
            self._set_list_params(*x)

    ### -------------------------------------------------------------------- ###

    def pull(self, delta=None):
        r"""
        Method that subtracts a vectorized numpy array or a list of numpy arrays
        from the parameters, instead of simply assigning it to the parameters.

        This method can be used in combination with a :samp:`with`-statement,
        which adds the :samp:`delta` back to the parameters on exit:

            >>> p = objective.get()
            >>> with objective.pull(p):
            ...     pass
            >>> assert numpy.testing.assert_almost_equal(p, objective.get())

        Parameters
        ----------
        delta : *numpy array* or [*numpy array*, ...]
            Constants that need to be subtracted from the parameters. If
            :samp:`delta` is None then this method is going to do nothing.

        Returns
        -------
        delegate : :class:`_AtExit`
            Object that implements :samp:`__enter__` and :samp:`__exit__`
            python methods. If used with a :samp:`with`-statement, then the
            :samp:`delta` will be added back to the parameters on exit.
        """

        # do nothing if delta is None
        if delta is None:
            return self._AtExit()

        # subtract delta to parameters
        if isinstance(delta, _np.ndarray):
            self._put_flat_params(-delta)
        else:
            self._put_list_params(*(-x for x in delta))

        # return at-exit delegate to add delta to parameters
        # if necessary
        return self._AtExit(lambda: self.push(delta))

    def push(self, delta=None):
        r"""
        Method that adds a vectorized numpy array or a list of numpy arrays to
        the parameters, instead of simply assigning it to the parameters.

        This method can be used in combination with a :samp:`with`-statement,
        which subtracts the :samp:`delta` back from the parameters on exit:

            >>> p = objective.get()
            >>> with objective.push(p):
            ...     pass
            >>> assert numpy.testing.assert_almost_equal(p, objective.get())

        Parameters
        ----------
        delta : *numpy array* or [*numpy array*, ...]
            Constants that need to be added to the parameters. If :samp:`delta`
            is None then this method is going to do nothing.

        Returns
        -------
        delegate : :class:`_AtExit`
            Object that implements :samp:`__enter__` and :samp:`__exit__`
            python methods. If used with a :samp:`with`-statement, then the
            :samp:`delta` will be subtracted back from the parameters.
        """

        # do nothing if delta is None
        if delta is None:
            return self._AtExit()

        # add delta to parameters
        if isinstance(delta, _np.ndarray):
            self._put_flat_params( delta)
        else:
            self._put_list_params(*delta)

        # return at-exit delegate to subtract delta from parameters
        # if necessary
        return self._AtExit(lambda: self.pull(delta))

    ### -------------------------------------------------------------------- ###

    def evaluate(self, d_set, delta=None, cost="all"):
        r"""
        Method that computes the cost error for a given data set.

        Parameters
        ----------
        d_set : instance of :samp:`sample.Abstract` or [*numpy array*, ...]
            The data set that can be an instance of :samp:`sample.Abstract`
            if one wants to compute the mean cost error over batches of a
            data set or a list of data set parameters that will be used as
            call arguments for the actual theano function.
        delta : *numpy array* or [*numpy array*, ...]
            Parameter delta that will be temporarily added to the parameters
            while computing the cost error. If None then this function will
            use the parameters as is.
        costs : "main" or "all"
            If :samp:`costs="main"` then this method will just compute the
            main error, if :samp:`costs="all"` it will compute all cost errors.

        Returns
        -------
        cost : *numpy array*
            Mean cost error over all batches.
        """

        # compile cost function if missing
        if not hasattr(self, "_cost_func"):
            self._cost_func = _get_cost_func(self)

        # compute and return cost
        return self._cost_func(d_set, delta, cost)

    ### -------------------------------------------------------------------- ###

    costs = property(doc="list of costs that need to be minimized according to "
                         "this objective.")

    @costs.getter
    def costs(self):
        return self._costs

    @costs.setter
    def costs(self, value):
        self._costs = list(value)

    @costs.deleter
    def costs(self):
        raise AttributeError("cannot delete attribute 'costs'")

    ### -------------------------------------------------------------------- ###

    outputs = property(doc="outputs that are defined by this objective.")

    @outputs.getter
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        self._outputs = list(value)

    @outputs.deleter
    def outputs(self):
        raise AttributeError("cannot delete attribute 'outputs'")

    ### -------------------------------------------------------------------- ###

    class _AtExit(object):
        r"""
        Delegate object that executes a given callback at the end of a
        :samp:`with`-statement.
        """

        def __init__(self, callback=lambda: None):
            self._callback = callback

        def __enter__(self):
            pass

        def __exit__(self, type, value, traceback):
            self._callback()

    ### -------------------------------------------------------------------- ###

### ======================================================================== ###

def _get_cost_func(objective):
    r"""
    Compiles and returns a function for computing the mean cost error w.r.t. an
    objective.

    Parameters
    ----------
    objective : :class:`Objective`
        Optimization objective that needs to be fulfilled.

    Returns
    -------
    cost : callable of cost(...) -> *numpy array*
        Function of :samp:`cost(d_set, delta=None, cost="all")` that computes
        the cost error, whereas the call arguments are defined as follow:

        d_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
            The data set that can can be an instance of
            :class:`sample.Abstract` if one wants to compute the mean cost
            error over batches of a data set or a list of data set parameters
            that will be used as call arguments for the actual theano function.
        delta : *numpy array* or [*numpy array*, ...]
            Parameter delta that will be temporarily added to the parameters
            while computing the cost error. If None then this function will
            use the parameters as is.
        costs : "main" or "all"
            If :samp:`costs="main"` then this function will just compute the
            main error, if :samp:`costs="all"` it will compute all cost errors.

        This function will compute and return the mean cost error over all
        batches.
    """

    # compile theano function for computing the cost error
    fn_all  = _theano.function(objective.inputs, _theano.tensor.stack(*objective.costs),
                               givens=objective.givens,
                               on_unused_input="ignore")
    fn_main = _theano.function(objective.inputs, objective.costs[0],
                               givens=objective.givens,
                               on_unused_input="ignore")

    # define function for computing the cost error
    def cost(d_set, delta=None, costs="all"):

        # get costs function
        if   costs == "all":
            fn = fn_all
        elif costs == "main":
            fn = fn_main
        else:
            raise ValueError("got an unexpected value for 'costs': %r" % costs)

        # compute mean cost over all batches
        with objective.push(delta):
            if not isinstance(d_set, sample.Abstract):
                return fn(*d_set)
            else:
                return _np.sum((fn(*x) for x in d_set), axis=0) / len(d_set)

    # return function for computing the cost error
    return cost

### ------------------------------------------------------------------------ ###

def _get_grad_func(objective):
    r"""
    Compiles and returns a function for computing the flatten mean gradient and
    the mean cost error w.r.t. an objective.

    Parameters
    ----------
    objective : :class:`Objective`
        Optimization objective that needs to be fulfilled.

    Returns
    -------
    grad : callable of grad(...) -> *numpy array* or (*numpy array*, [*numpy array*, ...])
        Function of :samp:`grad(d_set, delta=None, costs=None)` that computes
        the flatten mean gradient. The call arguments of the function are
        defined as follow:

        d_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
            The data set that can can be an instance of
            :class:`sample.Abstract` if one wants to compute the mean
            gradient over batches of a data set or a list of data set
            parameters that will be used as call arguments for the actual
            theano function.
        delta : *numpy array* or [*numpy array*, ...]
            Parameter delta that will be temporarily added to the parameters
            while computing the gradient. If None then this function will
            use the parameters as is.
        costs : "main" or "all"
            If this function should return the cost error. If None then this
            function is not going to compute the error and it will only return
            the mean gradient, if "main" then this function is going to compute
            and return only the main error with the mean gradient and if "all"
            then this function is going to compute and return all errors over
            the training set, together with the mean gradient.

        This function will either just return the mean gradient over all
        batches if :samp:`costs=None`, or a tuple of the mean gradient and the
        mean error over the training set.
    """

    # apply updates before cost is being evaluated
    costs = _theano.clone(objective.costs, replace=objective.updates)

    # compile theano function for computing the flatten gradient
    # and the cost error
    gr      = _theano.grad(costs[0], objective.wrt, objective.cc)
    fn_none = _theano.function(objective.inputs, [_flatten(gr), _theano.tensor.constant(0)],
                               givens=objective.givens, updates=objective.updates,
                               on_unused_input="ignore")
    fn_main = _theano.function(objective.inputs, [_flatten(gr), costs[0]],
                               givens=objective.givens, updates=objective.updates,
                               on_unused_input="ignore")
    fn_all  = _theano.function(objective.inputs, [_flatten(gr), _theano.tensor.stack(*costs)],
                               givens=objective.givens, updates=objective.updates,
                               on_unused_input="ignore")

    # define function for computing flatten gradient and cost error
    def grad(d_set, delta=None, costs=None):

        # get gradient/cost function
        if   costs is None:
            fn = fn_none
        elif costs == "main":
            fn = fn_main
        elif costs == "all":
            fn = fn_all
        else:
            raise ValueError("got an unexpected value for 'costs': %r" % costs)

        # compute mean gradient and cost error
        with objective.push(delta):
            if not isinstance(d_set, sample.Abstract):
                return fn(*d_set)
            else:
                grad, cost = 0.0, 0.0
                for x in d_set:
                    a, b = fn(*x)
                    grad += a
                    cost += b
                if costs is None:
                    return grad / len(d_set)
                else:
                    return grad / len(d_set), cost / len(d_set)

    # return function for computing flatten gradient and cost error
    return grad

### ------------------------------------------------------------------------ ###

def _get_step_iter(objective, momentum=None):
    r"""
    Compiles and returns a function that iterates over the baches of a data set
    and applies a single update to the parameters on each batch. The returned
    function further returns an iterator that yields the current cost error of
    each batch.

    Parameters
    ----------
    objective : :class:`Objective`
        Optimization objective that needs to be fulfilled.
    momentum : "standard" or "nesterov"
        If the returned function should be compiled with momentum or not. If
        None then this function will return a function that does not implement
        a momentum method. If "standard" then the returned function will be
        compiled with a standard momentum. If "nesterov" then the returned
        function will be compiled with a momentum method that was inspired by
        the Accelerated Nesterov Gradient method.

    Returns
    -------
    step : callable of step(...) -> iterable
        Function of :samp:`step(d_set, eta, mu=0.0, costs=None)`, whereas the
        call arguments are defined as:

        d_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
            The data set that can can be an instance of
            :class:`sample.Abstract` if one wants to iterate over batches of
            a data set or a list of data set parameters that will be used as
            call arguments for the actual
            theano function.
        eta : *float*
            The learning rate that is going to be used.
        mu : *float*
            The momentum rate, which can be ignored if the function was
            compiled without a momentum method.
        costs : "main" or "all"
            If this function should compute the batch cost error. If None then
            this function is not going to compute the error and will just yield
            0 as default, if "main" then this function is going to compute only
            the main error of the current batch and if "all" then this function
            will compute all errors of the current batch.

        This function will return an iterator that updates the parameters of
        the trained model and that yields the errors.
    """

    # assert value of momentum
    assert momentum in (None, "standard", "nesterov")

    # define variables for update rate and momentum
    eta = _theano.tensor.scalar("eta", dtype=_theano.config.floatX)
    mu  = _theano.tensor.scalar("mu" , dtype=_theano.config.floatX)

    # compute mapped gradient of current objective and get mapped parameters
    g = _Mapped(_theano.grad(objective.costs[0], objective.wrt, objective.cc))
    p = _Mapped(objective.wrt)

    # define function context
    context = Context(context=objective)
    context.inputs.insert(0, _theano.Param(eta, allow_downcast=True))
    context.inputs.insert(1, _theano.Param(mu , allow_downcast=True))

    # update function context according to momentum
    if momentum is None:

        # update function context
        context.updates.update(zip(p, p - eta * g))

    else:

        # define velocity matrices and residual of step function
        v = _Mapped.apply(_shared_as, p, 0, "velocity[{!r}]")
        r = v * mu - g * eta

        # pre-apply velocity if necessary
        if momentum == "nesterov":
            r = _Mapped.apply(_theano.clone, r, zip(p, p + mu * v))

        # update function context
        context.updates.update(zip(v, r) + zip(p, p + r))

    # compile theano functions
    fn_none = _theano.function(context.inputs, _theano.tensor.constant(0),
                               givens=context.givens, updates=context.updates,
                               on_unused_input="ignore")
    fn_main = _theano.function(context.inputs, objective.costs[0],
                               givens=context.givens, updates=context.updates,
                               on_unused_input="ignore")
    fn_all  = _theano.function(context.inputs, objective.costs,
                               givens=context.givens, updates=context.updates,
                               on_unused_input="ignore")

    # define function for computing steps
    def step(d_set, eta, mu=0.0, costs=None):

        # get step/cost function
        if   costs is None:
            fn = fn_none
        elif costs == "main":
            fn = fn_main
        elif costs == "all":
            fn = fn_all
        else:
            raise ValueError("got an unexpected value for 'costs': %r" % costs)

        # create iterable data set if necessary
        if not isinstance(d_set, sample.Abstract):
            d_set = [d_set]

        # return iterator for computing steps and yielding cost errors
        return (fn(eta, mu, *x) for x in d_set)

    # return function for computing steps and cost error
    return step

### ------------------------------------------------------------------------ ###

def _get_Avec_func(objective, type="G", struct=lambda dc: 0.0, alt_Avec=False):
    r"""
    Compiles and returns a function for computing the flatten curvature matrix-\
    vector product w.r.t. an objective.

    Parameters
    ----------
    objective : :class:`Objective`
        Optimization objective that needs to be fulfilled.
    type : "H", "G" or callable of type(...) -> *numpy array*
        If a string then this argument can be "G" for the Gauss-Newton
        matrix-vector product :math:`\textbf{Gv}`, or "H" for the Hessian
        matrix-vector product :math:`\textbf{Hv}`. If callable then this
        function must be of :samp:`type(d_set, v, c=0.0, delta=None)` and
        return the flatten curvature matrix-vector product.
    struct : callable of struct(*theano scalar*) -> *theano scalar*
        Function of :samp:`struct(dc)` that returns the structural damping
        penalty as described in [2]_.
    alt_Avec : *bool*
        If Ture then this function will use an alternative definition for
        computing the curvature matrix-vector products.

    Returns
    -------
    Avec : callable of Avec(...) -> *numpy array*
        Function of :samp:`Avec(d_set, v, c=0.0, delta=None)`. The call
        arguments of the function are defined as follow:

        d_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
            The data set that can can be an instance of
            :class:`sample.Abstract` if one wants to compute the mean
            curvature matrix-vector product over all batches of a data
            set or a list of data set parameters that will be used as call
            arguments for the actual theano function.
        v : *numpy array*
            The vector that is used for computing the curvature matrix-\
            vector product.
        dc : *float*
            The damping value that dictates the ammount of damping as described
            in [1]_ and [2]_.
        delta : *numpy array* or [*numpy array*, ...]
            Parameter delta that will be temporarily added to the parameters
            while computing the curvature matrix-vector product. If None then
            this function will use the parameters as is.

        This function will compute and return the flatten matrix-vector product.

    References
    ----------
    .. [1] James Martens. Deep Learning via Hessian-free Optimization. In
        Proceedings of the 27th International Conference on Machine Learning
        (ICML), 2010.
    .. [2] James Martens and Ilya Sutskever. Learning Recurrent Neural Networks
        with Hessian-Free Optimization. In Proceedings of the 28th
        International Conference on Machine Learning (ICML-11), 2011.
    """

    # function for computing the flatten matrix-vector product
    # was defined somewhere else
    if callable(type):
        return type

    # define default damping coefficient
    dc = _theano.tensor.scalar("dc", dtype=_theano.config.floatX)

    # get input vector and it's corresponding split variables that
    # match the opimization parameters in number and shape
    v =  objective._in_vector
    x =  objective._as_params

    # define basic types w.r.t. number of dimensions
    types = [_theano.tensor.scalar,
             _theano.tensor.vector,
             _theano.tensor.matrix,
             _theano.tensor.tensor3,
             _theano.tensor.tensor4]

    # define consider constant parameters that macht the originals
    cc = [types[c.ndim]("copy" + (":%s" % c.name if c.name else "")) for c in objective.cc]

    # substitute consider constants with variables
    clones  = _theano.clone(objective.outputs + [objective.costs[0] + struct(dc)], replace=zip(objective.cc, cc))
    outputs = clones[:-1]
    cost    = clones[ -1]

    # proceede according to matrix type
    if   type == "G":
        if not alt_Avec:
            Jx  = _theano.tensor.Rop(outputs, objective.wrt, x)
            HJx = _theano.grad(_theano.tensor.dot(_flatten(_theano.grad(cost, outputs)), _flatten(Jx)), outputs, Jx)
            Ax  = _theano.tensor.Lop(outputs, objective.wrt, HJx, HJx + Jx)
        else:
            Jx  = _theano.tensor.Rop(outputs, objective.wrt, x)
            HJx = _theano.tensor.Rop(_theano.grad(cost, outputs), outputs, Jx)
            Ax  = _theano.tensor.Lop(outputs, objective.wrt, HJx)
    elif type == "H":
        if not alt_Avec:
            Ax  = _theano.grad(_theano.tensor.dot(_flatten(_theano.grad(cost, objective.wrt)), _flatten(x)), objective.wrt)
        else:
            Ax  = _theano.Rop(_theano.grad(cost, objective.wrt), objective.wrt, x)
    else:
        raise ValueError("got an unexpected value for the curvature matrix "
                         "type %r" % type)

    # compile theano function
    fn = _theano.function([dc, v] + objective.inputs, _flatten(Ax),
                          givens=objective.givens.items()
                                + zip(cc, objective.cc),
                          on_unused_input="ignore")

    # define function for computing the flatten matrix-vector products
    def Avec(d_set, v, c=0.0, delta=None):

        # define initial curvature matrix-vector product
        Av = 0.0 if not c else c * v

        # compute mean matrix-vector product
        with objective.push(delta):
            if not isinstance(d_set, sample.Abstract):
                Av += fn(c, v, *d_set)
            else:
                Av += _np.sum((fn(c, v, *x) for x in d_set), axis=0) / len(d_set)

        # return curvature matrix-vector product
        return Av

    # return function for computing flatten matrix-vector product
    return Avec

### ------------------------------------------------------------------------ ###

def _get_pcon_func(objective, type=None, struct=lambda c: 0.0):
    r"""
    Compiles and returns a function for computing the flattened diagonal of a
    precodition matrix :math:`\text{diag}(\textbf{M})^{-1}`.

    Parameters
    ----------
    objective : :class:`Objective`
        Optimization objective that needs to be fulfilled.
    type : "martens", "jacobi" or callable of type(...) -> *numpy array*
        If string then this argument can be "martens" for the pre-conditioner
        described in [1]_, which basically approximates the inverted diagonal
        of the Fisher information matrix or "jacobi" for the pre-conditioner
        described in [3]_, which basically approximates the inverted diagonal
        of the Gauss-Newton matrix. If callable then this function must be of
        :samp:`type(d_set, c=0.0, delta=None)` and return the inverted diagonal
        of the pre-condition matrix :math:`\textbf{m} =
        \text{diag}(\textbf{M}^{-1})`.
    struct : callable of struct(*theano scalar*) -> *theano scalar*
        Function of :samp:`struct(dc)` that returns the structural damping penalty
        as described in [1]_.

    Returns
    -------
    pcon : callable of pcon(...) -> *numpy array*
        Function of :samp:`pcon(d_set, c=0.0, delta=None)` that will compute the
        flatten and inverted diagonal of a precondition matrix. The call
        arguments of the function are defined as follow:

        d_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
            The data set that can can be an instance of
            :class:`sample.Abstract` if one wants to compute the mean
            preconditioner matrix over all batches of a data set or a list of
            data set parameters that will be used as call arguments for the
            actual theano function.
        dc : *float*
            The damping value that dictates the ammount of damping as described
            in [1]_ and [2]_.
        delta : *numpy array* or [*numpy array*, ...]
            Parameter delta that will be temporarily added to the parameters
            while computing the curvature matrix vector product. If None then
            this function will use the parameters as is.

        This function will compute and return the flatten diagonal of the
        preconditioner matrix.

    Note
    ----
    The pre-condition function that is being returned when :samp:`type="jacobi"`
    contains one additional call argument that dictates the number of iterations
    to approximate the inverted diagonal of the pre-condition matrix. This value
    is defined as and set to :samp:`n_times=10`.

    References
    ----------
    .. [1] James Martens. Deep Learning via Hessian-Free Optimization. In
        Proceedings of the 27th International Conference on Machine Learning
        (ICML), 2010.
    .. [2] James Martens and Ilya Sutskever. Learning Recurrent Neural Networks
        with Hessian-Free Optimization. In Proceedings of the 28th
        International Conference on Machine Learning (ICML-11), 2011.
    .. [3] Olivier Chapelle and Dumitru Erhan. Improved Preconditioner for
        Hessian Free Optimization. In NIPS Workshop on Deep Learning and
        Unsupervised Feature Learning, 2011.
    """

    # do not return any precondition matrix
    if type is None:
        return lambda d_set, dc=0.0, delta=None: _np.cast[_theano.config.floatX](1.0)

    # function for computing the inverted diagonal of a
    # precondition matrix was defined somewhere else
    if callable(type):
        return type

    # define default damping coefficient
    dc = _theano.tensor.scalar("dc", dtype=_theano.config.floatX)

    # compile theano function for computing the flatten martens
    # precondition matrix
    if type == "martens":

        # compile squared gradient function
        gr = _theano.grad(objective.costs[0] + struct(dc), objective.wrt, objective.cc)
        fn = _theano.function([dc] + objective.inputs, _flatten(gr)**2,
                              givens=objective.givens,
                              on_unused_input="ignore")

        # define function for computing the flatten precondition matrix
        def martens(d_set, dc=0.0, delta=None):

            # define initial diagonal of precondition matrix
            pc = dc

            # compute accumulated diagonal of precondition matrix
            with objective.push(delta):
                if not isinstance(d_set, sample.Abstract):
                    pc += fn(dc, *d_set)
                else:
                    pc += _np.sum((fn(dc, *x) for x in d_set), axis=0)

            # return inverted diagonal of precondition matrix
            return _floor_vector(pc) ** -0.75

        # return function for computing the flatten diagonal of
        # precondition matrix
        return martens

    # compile theano function for computing the flatten diagonal of
    # jacobi precondition matrix
    if type == "jacobi":

        # define shared random number generator
        srng = _theano.tensor.shared_randomstreams.RandomStreams()

        # define random permutations of dimensions that match the number
        # and shape of the outputs
        r = map(lambda out: srng.binomial(out.shape, dtype=out.dtype) * 2.0 - 1.0, objective.outputs)

        # compute approximated Gauss-Newton diagonal of matrix
        Hv      = _theano.grad(_theano.tensor.sum(_flatten(_theano.grad(objective.costs[0] + struct(dc), objective.outputs, objective.cc))), objective.outputs, objective.cc)
        sqrtHv  = map(lambda a, b: _theano.tensor.sqrt(a) * b, Hv, r)
        JsqrtHv = _theano.tensor.Lop(objective.outputs, objective.wrt, sqrtHv, objective.cc)

        # compile theano function of J'sqrt(diag(H))r
        fn = _theano.function([dc] + objective.inputs, _flatten(JsqrtHv),
                              givens=objective.givens,
                              on_unused_input="ignore")

        # define function for computing the flatten diagonal of
        # precondition matrix
        def jacobi(d_set, dc=0.0, delta=None, n_times=10):

            # define initial precondition matrix
            pc = dc

            # compute accumulated precondition matrix
            with objective.push(delta):
                for i in xrange(n_times):
                    if not isinstance(d_set, sample.Abstract):
                        pc += fn(dc, *d_set)
                    else:
                        pc += _np.sum((fn(dc, *x) for x in d_set), axis=0)

            # return inverted diagonal of jacobi precondition matrix
            return _floor_vector(pc) ** -0.75

        # return function for computing the flatten diagonal of
        # jacobi precondition matrix
        return jacobi

    # raise error if nothing else to do
    raise ValueError("got an unexpected value for precondtion type %r" % type)

### ======================================================================== ###

def sgd(objective, d_set, v_set=None, n_epoch=250, eta=0.01, mu=0.25,
        t_error=None, epsilon=1e-4, phi=1.0, bestof=0,
        momentum=None, interrupt=None,
        return_info=False, callback=None, formatter=format.sgd):
    r"""
    Function that implements the well known **Stochastic Gradient Descent**
    method for optimizing neural networks on large data sets.

    Note that the learning rate argument (i.e. :samp:`eta`) and momentum rate
    argument (i.e. :samp:`mu`) can as well be callable functions of :samp:`f(\
    epoch, error_t, error_v)`. In this case they will be called on every epoch
    to compute the next learning rate and/or momentum rate value.

    As a recomendation one should consider the following function for computing
    a constantly decaying learning rate:

    .. math::
        \eta_0 (t + 1)^{-\alpha} \rightarrow \eta_t \text{,}

    wheras :math:`\eta_0` denotes the initial learning rate, :math:`\alpha` the
    rate at which the learning rate will decay (use a value inbetween 0 and 1)
    and :math:`t` the current epoch starting from 0.

    Parameters
    ----------
    objective : :class:`Objective`
        Optimization objective that needs to be fulfilled.
    d_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
        Training set used for computing the gradient. In case of a Stochastic
        Gradient Descent method one can choose a batch size of 1, but it is
        recomended to use mini batches of approximately the size of 10 to 100.
    v_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
        Validation set used for estimating the predictive power of the trained
        model on future data samples. This should be independent of
        :samp:`d_set`.
    n_epoch : *int*
        Minimum number of epochs. If the number of iterations exceeds this
        value then the optimizer will quit and return the best observed
        parameters.
    eta : *float* or callable of eta(...) -> *float*
        The learning rate at which this optimizer will update the residual
        direction. This argument can be an instance of float or a callable
        function of :samp:`eta(epoch, error_t, error_v)`, which will be called
        before starting a new epoch.
    mu : *float* or callable of mu(...) -> *float*
        The momentum rate, which will only be used if this optimizer is given a
        momentum directive. This argument can be -- just like :samp:`eta` --
        an instance of float or a callable function.
    t_error : *float*
        Target error. If the error on the validation set is lesser than this
        value, then this optimizer will quit the optimization and return the
        best observed parameters.
    epsilon : *float*
        This value defines the minimum improvement of the objective error
        that is considered to be significant. This value is used to determine
        after which epoch we have observed the best paramters, which only
        applies if you provide a validation set (i.e. :samp:`v_set`) to this
        optimization method.
    phi : *float*
        Factor that governs the patience of this optimization method. If this
        optimization method has observed an improvement w.r.t. the objective
        error (i.e. a new best epoch) then the number of minimum epochs will be
        updated regarding to :math:`N_\min = \max(\phi t, N_\min)`, whereas
        :math:`t` is the index of the current epoch update, :math:`N_\min` is
        the minimum number of epochs and :math:`\phi` is the factor that
        increases the patience. You can as well omit this stopping behaviour if
        you set :samp:`phi` to 1.0.
    bestof : *int*
        Which error from the list of validation set errors should be considered
        when evaluating the best epoch. Default is :samp:`bestof=0`, which will
        consider the main error that is used for optimizing the objective.
    momentum : "standard" or "nesterov"
        If this optimizer should apply a momentum to the parameter updates. If
        you set this value to None then this optimizer will not use any kind of
        momentum, if you set this value to "standard" then this optimizer will
        use the standard momentum and if you set this value to "nesterov" then
        this optimizer will use a momentum that was inspired by the Nesterov
        Accelerated Gradient method.
    interrupt : *int*
        Number of steps that this optimizer will take during one epoch to re-\
        evaluate the training set and validation set errors. This value should
        be smaller than the number of batches in :samp:`d_set` and if set to
        None the training set and validation set error will only be evaluated
        at the end of every epoch.
    return_info : *bool*
        If this optimizer should return an info dictionary in addition to the
        best observed parameters.
    callback : callable
        Function of :samp:`callback(epoch, step, error_t, error_v, info)` that
        will be called after each interrupt, whereas the call arguments are
        defined as follow:

        epoch : *int*
            The current epoch of this optimization method.
        step : *int*
            The current step of this optimization method.
        error_t : *numpy array*
            The errors on the training set taken from the last interrupt.
        error_v : *numpy array*
            The errors on the validation set taken from the last interrupt.
        info : *dict*
            A dictionary that contains some additional information about the
            last iteration.

        Note that the first call of this function might contain some dummy
        values. This function can as well stop the training by raising a
        :class:`StopIteration` exception.
    formatter : callable of formatter(...) -> callable
        Function of :samp:`formatter(n_error, n_epoch, n_step, info)` that is
        used for outputing the progress of this optimization method. The call
        arguments are defined as follow:

        n_error : *int*
            The number of errors that should be displayed.
        n_epoch : *int*
            The number of epochs of this optimizer.
        n_step : *int*
            The number of steps of this optimizer.
        info : *dict*
            The original call arguments of this optimization method.

        This function must return a row-print function :samp:`r_print(epoch,
        step, error_t, error_v=None, info={})`, whereas the call arguments of
        this function are defined as follow:

        epoch : *int*
            The current epoch of this optimization method.
        step : *int*
            The current step of this optimization method.
        error_t : *numpy array*
            The errors on the training set taken from the last interrupt.
        error_v : *numpy array*
            The errors on the validation set taken from the last interrupt.
        info : *dict*
            A dictionary that contains some additional information about the
            last iteration.

        This optimization method will not generate any output if you set the
        formatter to None.

    Returns
    -------
    out : [*numpy array*, ...]
        List of parameters that result from optimizing the objective with this
        optimization method.
    info : *dict*
        Optional output that will be returned if this function was called with
        :samp:`return_info=True`. This output contains some additional
        information of every single epoch/step that this optimizer has
        undergone.

    See Also
    --------

        Class :class:`Objective`
            Implementation of an optimization objective that basically
            describes the optimizaton relevant context.
        Class :class:`sample.Constant`
            Implementation of a constant data sampler, used to draw samples in
            form of batches from an existing data set. This sampler instance
            actually dosn't update its state, even when calling the update
            method.
        Class :class:`sample.Random`
            Implementation of a random data sampler, used to draw samples in
            form of batches from an existing data set. When updated then this
            sampler will shuffle all samples across all its batches.
        Class :class:`sample.Sequence`
            Implementation of a sequence data sampler that is used to draw
            time series in form of batches from an existing time series data
            set. When updated then this sampler will shuffle all its batches
            but withoud shuffling the samples within its batches.
        Class :class:`sample.Interrupt`
            Implementation of a interrupt sampler that wrapps other samplers
            and that calls a callback function when drawing samples in form
            of a batch.
    """

    # define number of errors and steps when iterating over the baches
    n_error = len(objective.costs)
    n_step  = len(d_set) if isinstance(d_set, sample.Abstract) else 1

    # get row- and progress-print functions from formatter function
    if formatter is not None:
        r_print = formatter(n_error, n_epoch, n_step,
                            info=dict(objective = objective,
                                      d_set     = d_set    ,
                                      v_set     = v_set    ,
                                      n_epoch   = n_epoch  ,
                                      eta       = eta      ,
                                      mu        = mu       ,
                                      t_error   = t_error  ,
                                      epsilon   = epsilon  ,
                                      phi       = phi      ,
                                      bestof    = bestof   ,
                                      momentum  = momentum ,
                                      interrupt = interrupt,
                                      callback  = callback ))
    else:
        r_print = lambda *args, **kwargs: None

    # set interrupt if necessary
    if interrupt is None:
        interrupt = n_step

    # define functions for updating learning and momentum rates
    compute_eta = eta if callable(eta) else lambda *args: eta
    compute_mu  = mu  if callable(mu ) else lambda *args: mu

    # compile theano functions
    cost      = _get_cost_func(objective          )
    step_iter = _get_step_iter(objective, momentum)

    # define initial target and validation error
    error_t = [[float("inf")] * n_error]
    error_v = [[float("inf")] * n_error]

    # define initial minimum error and epoch
    m_error = float("inf")
    m_epoch = None

    # define indicator for best iteration
    best = False

    # define initial parameters
    params = []

    # print header to stdout
    r_print(-1, -1, error_t[-1], error_v[-1],
            info=dict(n_epoch = n_epoch,
                      n_step  = n_step ,
                      m_epoch = m_epoch,
                      m_error = m_error,
                      best    = best   ,
                      eta     = eta    ,
                      mu      = mu     ))

    # define main loop of optimizer
    for epoch in xrange(2**32 - 1):

        # compute current learning and momentum rates
        eta = compute_eta(epoch, error_t[-1], error_v[-1])
        mu  = compute_mu (epoch, error_t[-1], error_v[-1])

        # get step iterator of current epoch
        step  = step_iter(d_set, eta, mu, costs=None)

        # step through all batches and update parameters
        for i in xrange(n_step):

            # skip current step if not interrupted
            if not (i % interrupt or i == n_step - 1):

                # append current parameters to buffer
                params.append(objective.get())

                # compute training set error
                error_t.append(cost(d_set))

                # update optimizer w.r.t. validation set
                if v_set is not None:

                    # compute validation set error
                    error_v.append(cost(v_set))

                    # update best error and parameters if necessary
                    if error_v[-1][bestof] < m_error - epsilon:
                        n_epoch = max(n_epoch, int(phi * epoch))
                        m_epoch = epoch
                        m_error = error_v[-1][bestof]
                        best    = True
                    else:
                        best    = False

                # print current state to stdout
                r_print(epoch, i, error_t[-1], error_v[-1],
                        info=dict(n_epoch = n_epoch,
                                  n_step  = n_step ,
                                  m_epoch = m_epoch,
                                  m_error = m_error,
                                  best    = best   ,
                                  eta     = eta    ,
                                  mu      = mu     ))

                # call callback function
                try:
                    if callback is not None:
                        callback(epoch, i, error_t[-1], error_v[-1],
                                 info=dict(n_epoch = n_epoch,
                                           n_step  = n_step ,
                                           m_epoch = m_epoch,
                                           m_error = m_error,
                                           best    = best   ,
                                           eta     = eta    ,
                                           mu      = mu     ))
                except StopIteration:
                    n_epoch = epoch + 1

            # compute step on next batch
            step.next()

        # test if patience is exceeded
        if n_epoch is not None and n_epoch <= epoch + 1:
            break

        # test if target error was accomplished
        if t_error is not None and m_error <= t_error:
            break

        # update data sets if possible
        if isinstance(d_set, sample.Abstract):
            d_set.update()
        if isinstance(v_set, sample.Abstract):
            v_set.update()

    # append current parameters to buffer
    params.append(objective.get())

    # compute training set error
    error_t.append(cost(d_set))

    # update optimizer w.r.t. validation set
    if v_set is not None:

        # compute validation set error
        error_v.append(cost(v_set))

        # update best error and parameters if necessary
        if error_v[-1][bestof] < m_error - epsilon:
            m_epoch = epoch + 1
            m_error = error_v[-1][bestof]
            best    = True
        else:
            best    = False

        # update optimizer w.r.t. validation set
        objective.set(params[m_epoch])

    # print current state to stdout
    r_print(epoch + 1, 0, error_t[-1], error_v[-1],
            info=dict(n_epoch = n_epoch,
                      n_step  = n_step ,
                      m_epoch = m_epoch,
                      m_error = m_error,
                      best    = best   ,
                      eta     = eta    ,
                      mu      = mu     ))

    # call callback function
    try:
        if callback is not None:
            callback(epoch + 1, 0, error_t[-1], error_v[-1],
                     info=dict(n_epoch = n_epoch,
                               n_step  = n_step ,
                               m_epoch = m_epoch,
                               m_error = m_error,
                               best    = best   ,
                               eta     = eta    ,
                               mu      = mu     ))
    except StopIteration:
        pass

    # return params from objective
    if return_info:
        return (objective.get(flatten=False),
                dict(error_t = error_t[1:],
                     error_v = error_v[1:],
                     params  = params     ))
    else:
        return  objective.get(flatten=False)

### ======================================================================== ###

def irprop(objective, d_set, v_set=None, n_epoch=250,
           eta=(0.5, 1.2), delta=(1e-6, 50),
           t_error=None, epsilon=1e-4, phi=1.0, bestof=0, kind="-",
           return_info=False, callback=None, formatter=format.irprop):
    r"""
    Function that implements the magnificent **Improved Resilient Propagation**
    (i.e. :math:`\text{iRprop}^-` and :math:`\text{iRprop}^+`) method for
    optimizing neural networks.

    This specific version is a batched version that can use batch samplers to
    compute the mean gradient over all batches. Further, you can choose if you
    want to have error backtracking, by setting :samp:`kind="+"` (i.e. use
    error backtracking) or :samp:`kind="-"` (i.e. do not use error
    backtracking).

    Please refer to [1]_, for a deeper insight on how this optimization method
    works.

    Parameters
    ----------
    objective : :class:`Objective`
        Optimization objective that needs to be fulfilled.
    d_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
        Training set used for computing the gradient.
    v_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
        Validation set used for estimating the predictive power of the trained
        model on future data samples. This should be independent of
        :samp:`d_set`.
    n_epoch : *int*
        Minimum number of epochs. If the number of iterations exceeds this
        value then the optimizer will quit and return the best observed
        parameters.
    eta : (*float*, *float*)
        Tuple containing :math:`\eta^-` and :math:`\eta^+` for decreasing or
        increasing the step update :math:`\Delta`.
    delta : (*float*, *float*)
        Tuple containing :math:`\Delta_\min` and :math:`\Delta_\max` for the
        minimum and maximum step update.
    t_error : *float*
        Target error. If the error on the validation set is lesser than this
        value, then this optimizer will quit the optimization and return the
        best observed parameters.
    epsilon : *float*
        This value defines the minimum improvement of the objective error
        that is considered to be significant. This value is used to determine
        after which epoch we have observed the best paramters, which only
        applies if you provide a validation set (i.e. :samp:`v_set`) to this
        optimization method.
    phi : *float*
        Factor that governs the patience of this optimization method. If this
        optimization method has observed an improvement w.r.t. the objective
        error (i.e. a new best epoch) then the number of minimum epochs will be
        updated regarding to :math:`N_\min = \max(\phi t, N_\min)`, whereas
        :math:`t` is the index of the current epoch update, :math:`N_\min` is
        the minimum number of epochs and :math:`\phi` is the factor that
        increases the patience. You can as well omit this stopping behaviour if
        you set :samp:`phi` to 1.0.
    bestof : *int*
        Which error from the list of validation set errors should be considered
        when evaluating the best epoch. Default is :samp:`bestof=0`, which will
        consider the main error that is used for optimizing the objective.
    kind : "+" or "-"
        If this optimizer should use error backtracking in case the objective
        error increases compared to the previous epoch. A value of "+" is
        equivalent to :math:`\text{iRprop}^+` and this optimizer will use
        error backtracking and a value of "-" is equivalent to
        :math:`\text{iRprop}^-` and this optimizer will not use any error
        backtracking. Even thought this optimizer implements both flavors,
        it should be pointed out that the convergences are almost the same,
        just that :math:`\text{iRprop}^-` is slightly more efficient than
        :math:`\text{iRprop}^+` in terms of computational demand.
    return_info : *bool*
        If this optimizer should return an info dictionary in addition to the
        best observed parameters.
    callback : callable
        Function of :samp:`callback(epoch, error_t, error_v, info)` that will
        be called after each interrupt, whereas the call arguments are defined
        as follow:

        epoch : *int*
            The current epoch of this optimization method.
        error_t : *numpy array*
            The errors on the training set taken from the last interrupt.
        error_v : *numpy array*
            The errors on the validation set taken from the last interrupt.
        info : *dict*
            A dictionary that contains some additional information about the
            last iteration.

        Note that the first call of this function might contain some dummy
        values. This function can as well stop the training by raising a
        :class:`StopIteration` exception.
    formatter : callable of formatter(...) -> callable
        Function of :samp:`formatter(n_error, n_epoch, info)` that is used for
        outputing the progress of this optimization method. The call arguments
        are defined as follow:

        n_error : *int*
            The number of errors that should be displayed.
        n_epoch : *int*
            The number of epochs of this optimizer.
        info : *dict*
            The original call arguments of this optimization method.

        This function must return a row-print function :samp:`r_print(epoch,
        error_t, error_v=None, info={})`, whereas the call arguments of this
        function are defined as follow:

        epoch : *int*
            The current epoch of this optimization method.
        error_t : *numpy array*
            The errors on the training set taken from the last interrupt.
        error_v : *numpy array*
            The errors on the validation set taken from the last interrupt.
        info : *dict*
            A dictionary that contains some additional information about the
            last iteration.

        This optimization method will not generate any output if you set the
        formatter to None.

    Returns
    -------
    out : [*numpy array*, ...]
        List of parameters that result from optimizing the objective with this
        optimization method.
    info : *dict*
        Optional output that will be returned if this function was called with
        :samp:`return_info=True`. This output contains some additional
        information of every single epoch that this optimizer has undergone.

    References
    ----------
    .. [1] Christian Igel and Michael Hüsken. Improving the Rprop Learning
        Algorithm. In Proceedings of the Second International ICSC Symposium on
        Neural Computation (NC 2000), 2000.

    See Also
    --------

        Class :class:`Objective`
            Implementation of an optimization objective that basically
            describes the optimizaton relevant context.
        Class :class:`sample.Constant`
            Implementation of a constant data sampler, used to draw samples in
            form of batches from an existing data set. This sampler instance
            actually dosn't update its state, even when calling the update
            method.
        Class :class:`sample.Random`
            Implementation of a random data sampler, used to draw samples in
            form of batches from an existing data set. When updated then this
            sampler will shuffle all samples across all its batches.
        Class :class:`sample.Sequence`
            Implementation of a sequence data sampler that is used to draw
            time series in form of batches from an existing time series data
            set. When updated then this sampler will shuffle all its batches
            but withoud shuffling the samples within its batches.
        Class :class:`sample.Interrupt`
            Implementation of a interrupt sampler that wrapps other samplers
            and that calls a callback function when drawing samples in form
            of a batch.
    """

    # test if kind has an expected value
    if not kind in ("-", "+"):
        raise ValueError("call argument 'kind' must be '-' or '+', but got "
                         "%r" % kind)

    # cast rate arguments to theano's config and pre-process eta
    try:
        eta_dec, eta_inc = _np.array(eta, dtype=_theano.config.floatX) - 1.0
    except (ValueError, TypeError):
        raise ValueError("call argument 'eta' must be a tuple/list of two "
                         "floats, but got %r" % eta)
    try:
        delta_min, delta_max = _np.array(delta, dtype=_theano.config.floatX)
    except (ValueError, TypeError):
        raise ValueError("call argument 'delta' must be a tuple/list of two "
                         "floats, but got %r" % delta)

    # define number of errors and steps when iterating over the baches
    n_error = len(objective.costs)

    # get row- and progress-print functions from formatter function
    if formatter is not None:
        r_print = formatter(n_error, n_epoch,
                            info=dict(objective = objective,
                                      d_set     = d_set    ,
                                      v_set     = v_set    ,
                                      n_epoch   = n_epoch  ,
                                      eta       = eta      ,
                                      delta     = delta    ,
                                      t_error   = t_error  ,
                                      epsilon   = epsilon  ,
                                      phi       = phi      ,
                                      bestof    = bestof   ,
                                      callback  = callback ))
    else:
        r_print = lambda *args, **kwargs: None

    # compile theano functions
    cost = _get_cost_func(objective)
    grad = _get_grad_func(objective)

    # define initial target and validation error
    error_t = [[float("inf")] * n_error]
    error_v = [[float("inf")] * n_error]

    # define initial minimum error and epoch
    m_error = float("inf")
    m_epoch = None

    # define indicator for best iteration
    best = False

    # define initial parameters
    params = [objective.get()]

    # define initial parameter delta, gradient and update directon
    dw = _np.ones_like(params[-1]) * 0.0000
    b  = _np.ones_like(params[-1]) * 0.0000
    d  = _np.ones_like(params[-1]) * 0.0001

    # print header to stdout
    r_print(-1, error_t[-1], error_v[-1],
            info=dict(n_epoch = n_epoch,
                      m_epoch = m_epoch,
                      m_error = m_error,
                      best    = best   ,
                      b       = b      ,
                      d       = d      ))

    # define main loop of optimizer
    for epoch in xrange(2**32 - 1):

        # compute current gradient and training set error
        c, error = grad(d_set, costs="all")

        # append target set error to errors
        error_t.append(error)

        # update optimizer w.r.t. validation set
        if v_set is not None:

            # compute validation set error
            error_v.append(cost(v_set))

            # update best error and parameters if necessary
            if error_v[-1][bestof] < m_error - epsilon:
                n_epoch = max(n_epoch, int(phi * epoch))
                m_epoch = epoch
                m_error = error_v[-1][bestof]
                best    = True
            else:
                best    = False

        # print current state to stdout
        r_print(epoch, error_t[-1], error_v[-1],
                info=dict(n_epoch = n_epoch,
                          m_epoch = m_epoch,
                          m_error = m_error,
                          best    = best   ,
                          b       = b      ,
                          d       = d      ))

        # call callback function
        try:
            if callback is not None:
                callback(epoch, error_t[-1], error_v[-1],
                         info=dict(n_epoch = n_epoch,
                                   m_epoch = m_epoch,
                                   m_error = m_error,
                                   best    = best   ,
                                   b       = b      ,
                                   d       = d      ))
        except StopIteration:
            n_epoch = epoch + 1

        # compute product of old and new gradient
        prod = b * c

        # compute masks for positive and negative products
        p = prod > 0.0
        n = prod < 0.0

        # compute new update delta and clip to min and max
        _np.clip(d * (n * eta_dec + p * eta_inc + 1.0), delta_min, delta_max, d)

        # apply backtracking to parameters if error has increased
        if kind == "+" and error_t[-1][0] > error_t[-2][0]:
            objective.pull(n * dw)

        # set gradients with changing sign to zero and prepare
        # for next iteration
        b  = c * ~n

        # compute delta update of parameters
        dw = -_np.sign(b) * d

        # subtract delta from parameters
        objective.push(dw)

        # append current parameters to buffer
        params.append(objective.get())

        # test if patience is exceeded
        if n_epoch is not None and n_epoch <= epoch + 1:
            break

        # test if target error was accomplished
        if t_error is not None and m_error <= t_error:
            break

        # update data sets if possible
        if isinstance(d_set, sample.Abstract):
            d_set.update()
        if isinstance(v_set, sample.Abstract):
            v_set.update()

    # compute final training set error
    error_t.append(cost(d_set))

    # update optimizer w.r.t. validation set
    if v_set is not None:

        # compute validation set error
        error_v.append(cost(v_set))

        # update best error and parameters if necessary
        if error_v[-1][bestof] < m_error - epsilon:
            m_epoch = epoch + 1
            m_error = error_v[-1][bestof]
            best    = True
        else:
            best    = False

        # update optimizer w.r.t. validation set
        objective.set(params[m_epoch])

    # print current state to stdout
    r_print(epoch + 1, error_t[-1], error_v[-1],
            info=dict(n_epoch = n_epoch,
                      m_epoch = m_epoch,
                      m_error = m_error,
                      best    = best   ,
                      b       = b      ,
                      d       = d      ))

    # call callback function
    try:
        if callback is not None:
            callback(epoch, error_t[-1], error_v[-1],
                     info=dict(n_epoch = n_epoch,
                               m_epoch = m_epoch,
                               m_error = m_error,
                               best    = best   ,
                               b       = b      ,
                               d       = d      ))
    except StopIteration:
        pass

    # return params from objective
    if return_info:
        return (objective.get(flatten=False),
                dict(error_t = error_t[1:],
                     error_v = error_v[1:],
                     params  = params     ))
    else:
        return  objective.get(flatten=False)

### ======================================================================== ###

def _hf_iter_cg(A, b, x=None, M=lambda x: x, n_step=2**31-1):
    r"""
    Implementation of the Conjugate(d) Gradient method that iterates through
    all conjugated directions. This is an efficient method for solving large
    linear systems of

    .. math::
        \textbf{Ax} = \textbf{b}

    without the necessety of computing the inverse of :math:`\textbf{A}`. The
    matrix :math:`\textbf{A}` and the vector :math:`\textbf{b}` are known and
    :math:`\textbf{x}` is a unknown in this linear system.

    Please refer to [1]_, for a deeper insight on how this optimization method
    works.

    Parameters
    ----------
    A : callable of A(*numpy array*) -> *numpy array* or *numpy array*
        Function that computes the matrix-vector product of a matrix
        :math:`\textbf{A}` and a vector :math:`\textbf{v}` or the matrix
        :math:`\textbf{A}` itself that is part of the linear system.
    b : *numpy array*
        Vector of :math:`\textbf{b}` that is part of the linear system.
    x : *numpy array*
        Initial guess of the solution :math:`\mathbf{x}`. If None then a
        zero vector will be assumed.
    M : callable of M(*numpy array*) -> *numpy array* or *numpy array*
        Function that computes the pre-conditioned version of an update
        direction, e.g. :math:`\text{diag}(\textbf{M})^{-1}\textbf{r}`,
        or the diagonal :math:`\text{diag}(\textbf{M})^{-1}` of the pre-\
        condition matrix itself. If None then this method will not apply
        any pre-conditioning to the residuals.
    n_steps : *int*
        Maximum number of steps for this optimizer.

    Yields
    ------
    x : *numpy array*
        Current solution of the linear system :math:`\textbf{Ax} = \textbf{b}`.
    r : *numpy array*
        Current residual, pointing towards a better solution.

    Note
    ----
    If you want to save a record of all partial solutions of :samp:`x` and
    :samp:`r` throughout all steps, then please make a copy of these vectors,
    since this method will update them in-place and you will only end up with
    the last solution.

    References
    ----------
    .. [1] Jonathan Richard Shewchuk. An Introduction to the Conjugate Gradient
        Method Without the Agonizing Pain. Technical Report, 1994.
    """

    # default values for callables and initial solution
    if not callable(A):
        A = lambda v: _np.dot(A, v)
    if not callable(M):
        M = lambda v: v / M
    if x is None:
        x = _np.zeros_like(b)

    # compute initial residual and update direction
    r = b - A(x)
    d = M(r)

    # compute norm of residual
    delta = _np.dot(r, d)

    # iterate through conjugate directions
    for i in xrange(2**32 - 1):

        # compute product of update direction with curvature matrix
        q  = A(d)

        # compute magnitude of optimal line-search (i.e. A-norm of residual)
        alpha = delta / _np.dot(d, q)

        # update solution and the residual according to optimal line-search
        x += alpha * d
        r -= alpha * q

        # yield current solution, residual and direction
        yield x, r

        # break look if number of steps is reached
        if n_step - 1 <= i:
            break

        # update pre-conditioned residual
        s  = M(r)

        # compute new delta value
        delta_new = _np.dot(r, s)

        # compute next update direction (Gram-Schmidt conjugation in
        # Krylov sub-spaces)
        d *= delta_new / delta
        d += s

        # set delta to new value
        delta = delta_new

### ------------------------------------------------------------------------ ###

def hf(objective, d_set, c_set=None, v_set=None, n_epoch=100, n_step=100,
       pc_type=None, struct=lambda dc: 0.0,
       dc_init=0.01, dc_eta=1.5, dc_delta=0.25,
       tau=5e-5, gamma=1.3,
       t_error=None, epsilon=1e-4, phi=1.0, bestof=0,
       return_info=False, callback=None, formatter=format.hf):
    r"""
    Function that implements the notorious **Hessian-Free (Newton)** method for
    optimizing deep and recurrent neural networks.

    You can as well use structural damping with this implementation of
    Hessan-Free, by giving it a callback function (see argument :samp:`struct`)
    that takes the damping coefficient as an arguemnt and returns the
    additional penalty as described in [2]_. The damping coefficient in this
    case will be a theano variable and the returned expression should also be
    a symbolic theano expression. Any other values (e.g. givens, updates etc.)
    that the structural penalty expression might depend on can be -- as usual
    -- handed over to the optimization objective.

    Please refer to [1]_ and [2]_, for a deeper insight on how this
    optimization method works.

    Parameters
    ----------
    objective : :class:`Objective`
        Optimization objective that needs to be fulfilled.
    d_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
        Training set used for computing the gradient.
    c_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
        Training set used for computing the curvature matrix-vector products.
        If you leave this argument out then it will be the same as the gradient
        training set :samp:`d_set`.
    v_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
        Validation set used for estimating the predictive power of the trained
        model on future data samples. This should be independent of
        :samp:`d_set`.
    n_epoch : *int*
        Minimum number of epochs. If the number of iterations exceeds this
        value then the optimizer will quit and return the best observed
        parameters.
    n_step : *int*
        Maximum number of steps for solving the sub-objective.
    pc_type : "martens", "jacobi" or callable of pc_type(...) -> *numpy array*
        Pre-conditioner type. This can be "martens" for the pre-conditioner
        described in [1]_, which basically approximates the inverse diagonal
        of the Fisher information matrix, "jacobi" for the pre-conditioner
        described in [3]_, which basically approximates the inverse diagonal
        of the Gauss-Newton matrix, or any callable of :samp:`pc_type(b_set,
        dc=0.0, delta=None)` that returns the inverted diagonal of a pre-\
        condition matrix :math:`\textbf{m} = \text{diag}(\textbf{M})^{-1}`.
        If set to None then this optimizer will not use any pre-conditioning
        on the residuals of the sub-objective.
    struct : callable of struct(*theano scalar*) -> *theano scalar*
        Function of :samp:`struct(dc)` that takes the damping coefficient and
        returns the structural penalty as described in [2]_.
    dc_init : *float*
        The initial damping coefficient used by this optimizer that dictates
        the steepness of the ridge.
    dc_eta : *float*
        Factor by which the damping coefficient is being updated.
    dc_delta : *float*
        Margin which dictates when to update the damping coefficient. An
        :math:`\rho` that is smaller than this margin will lead to an increase
        of the damping coefficient by the value defined in :samp:`dc_eta`, an
        value of :math:`\rho` that is bigger than 1.0 minus the margin will
        on the other hand result into an decline of the damping coefficient by
        the same ammount that is defined in :samp:`dc_eta`. The value of
        :math:`\rho` is basically the ratio of the estimated error reduction
        defined by the sub-objective and the actual error reduction measured
        on the gradient training set.
    tau : *float*
        Value that governs the stop criterion of the sub-objective. The smaller
        this value is the more likely it will be that the sub-objective will
        stop early, and vice versa.
    gamma : *float*
        Value that governs how sparse the sampling of the error backtracking is
        going to be. The smaller this value the more dense is the sampling
        going to be, which could also lead to a longer runtime of this
        optimizer.
    t_error : *float*
        Target error. If the error on the validation set is lesser than this
        value, then this optimizer will quit the optimization and return the
        best observed parameters.
    epsilon : *float*
        This value defines the minimum improvement of the objective error
        that is considered to be significant. This value is used to determine
        after which epoch we have observed the best paramters, which only
        applies if you provide a validation set (i.e. :samp:`v_set`) to this
        optimization method.
    phi : *float*
        Factor that governs the patience of this optimization method. If this
        optimization method has observed an improvement w.r.t. the objective
        error (i.e. a new best epoch) then the number of minimum epochs will be
        updated regarding to :math:`N_\min = \max(\phi t, N_\min)`, whereas
        :math:`t` is the index of the current epoch update, :math:`N_\min` is
        the minimum number of epochs and :math:`\phi` is the factor that
        increases the patience. You can as well omit this stopping behaviour if
        you set :samp:`phi` to 1.0.
    bestof : *int*
        Which error from the list of validation set errors should be considered
        when evaluating the best epoch. Default is :samp:`bestof=0`, which will
        consider the main error that is used for optimizing the objective.
    return_info : *bool*
        If this optimizer should return an info dictionary in addition to the
        best observed parameters.
    callback : callable
        Function of :samp:`callback(epoch, step, error_t, error_v, info)` that
        will be called after each epoch, whereas the call arguments are defined
        as follow:

        epoch : *int*
            The current epoch of this optimization method.
        step : *int*
            The current step of this optimization method.
        error_t : *numpy array*
            The errors on the training set taken from the last interrupt.
        error_v : *numpy array*
            The errors on the validation set taken from the last interrupt.
        info : *dict*
            A dictionary that contains some additional information about the
            last iteration.

        Note that the first call of this function might contain some dummy
        values. This function can as well stop the training by raising a
        :class:`StopIteration` exception.
    formatter : callable of formatter(...) -> (callable, callable)
        Function of :samp:`formatter(n_error, n_epoch, n_step, info)` that is
        used for outputing the progress of this optimization method. The call
        arguments are defined as follow:

        n_error : *int*
            The number of errors that should be displayed.
        n_epoch : *int*
            The number of epochs of this optimizer.
        n_step : *int*
            The number of steps of this optimizer.
        info : *dict*
            The original call arguments of this optimization method.

        This function must return a row-print function :samp:`r_print(epoch,
        step, error_t, error_v=None, info={})` and a progress-print
        function :samp:`p_print(epoch, step, error_t=None, error_v=None,
        info={})`. The call arguments of this two functions are defined as
        follow:

        epoch : *int*
            The current epoch of this optimization method.
        step : *int*
            The current step of this optimization method.
        error_t : *numpy array*
            The errors on the training set taken from the last interrupt.
        error_v : *numpy array*
            The errors on the validation set taken from the last interrupt.
        info : *dict*
            A dictionary that contains some additional information about the
            last iteration.

        This optimization method will not generate any output if you set the
        formatter to None.

    Returns
    -------
    out : [*numpy array*, ...]
        List of parameters that result from optimizing the objective with this
        optimization method.
    info : *dict*
        Optional output that will be returned if this function was called with
        :samp:`return_info=True`. This output contains some additional
        information of every single epoch that this optimizer has undergone.

    References
    ----------
    .. [1] James Martens. Deep Learning via Hessian-free Optimization. In
        Proceedings of the 27th International Conference on Machine Learning
        (ICML), 2010.
    .. [2] James Martens and Ilya Sutskever. Learning Recurrent Neural Networks
        with Hessian-Free Optimization. In Proceedings of the 28th
        International Conference on Machine Learning (ICML-11), 2011.
    .. [3] Olivier Chapelle and Dumitru Erhan. Improved Preconditioner for
        Hessian Free Optimization. In NIPS Workshop on Deep Learning and
        Unsupervised Feature Learning, 2011.

    See Also
    --------

        Class :class:`Objective`
            Implementation of an optimization objective that basically
            describes the optimizaton relevant context.
        Class :class:`sample.Constant`
            Implementation of a constant data sampler, used to draw samples in
            form of batches from an existing data set. This sampler instance
            actually dosn't update its state, even when calling the update
            method.
        Class :class:`sample.Random`
            Implementation of a random data sampler, used to draw samples in
            form of batches from an existing data set. When updated then this
            sampler will shuffle all samples across all its batches.
        Class :class:`sample.Sequence`
            Implementation of a sequence data sampler that is used to draw
            time series in form of batches from an existing time series data
            set. When updated then this sampler will shuffle all its batches
            but withoud shuffling the samples within its batches.
        Class :class:`sample.Interrupt`
            Implementation of a interrupt sampler that wrapps other samplers
            and that calls a callback function when drawing samples in form
            of a batch.
    """

    # define number of errors
    n_error = len(objective.costs)

    # get row- and progress-print functions from formatter function
    if formatter is not None:
        r_print, p_print = formatter(n_error, n_epoch, n_step,
                                     info=dict(objective = objective,
                                               d_set     = d_set    ,
                                               c_set     = c_set    ,
                                               v_set     = v_set    ,
                                               n_epoch   = n_epoch  ,
                                               n_step    = n_step   ,
                                               pc_type   = pc_type  ,
                                               t_error   = t_error  ,
                                               epsilon   = epsilon  ,
                                               phi       = phi      ,
                                               bestof    = bestof   ,
                                               dc_init   = dc_init  ,
                                               struct    = struct   ,
                                               callback  = callback ))
    else:
        r_print = lambda *args, **kwargs: None
        p_print = lambda *args, **kwargs: None

    # define c_set if missing
    if c_set is None:
        c_set = d_set

    # compile theano functions
    cost = _get_cost_func(objective                 )
    grad = _get_grad_func(objective                 )
    Avec = _get_Avec_func(objective, "G"    , struct)
    pcon = _get_pcon_func(objective, pc_type, struct)

    # define the initial damping coefficient
    # (i.e. labmda in original paper)
    dc = dc_init

    # define gradient and previous update direction
    b = None
    p = None

    # define initial value of quadratic
    # objective (i.e. 0.5 x'Ax - b'x)
    q = float("inf")

    # define initial target and validation error
    error_t = [[float("inf")] * n_error]
    error_v = [[float("inf")] * n_error]

    # define initial minimum error and epoch
    m_error = float("inf")
    m_epoch = None

    # define indicator for best iteration
    best = False

    # define initial parameters
    params = [objective.get()]

    # print header to stdout
    r_print(-1, 0, error_t[-1], error_v[-1],
            info=dict(n_epoch = n_epoch,
                      n_step  = n_step ,
                      m_epoch = m_epoch,
                      m_error = m_error,
                      best    = best   ,
                      dc      = dc     ,
                      b       = b      ,
                      p       = p      ,
                      q       = q      ))

    # compute training set error
    error_t.append(cost(d_set))

    # update optimizer w.r.t. validation set
    if v_set is not None:

        # compute validation set error
        error_v.append(cost(v_set))

        # update best error and parameters if necessary
        if error_v[-1][bestof] < m_error - epsilon:
            m_epoch = 0
            m_error = error_v[-1][bestof]
            best    = True
        else:
            best    = False

    # print current state to stdout
    r_print(0, 0, error_t[-1], error_v[-1],
            info=dict(n_epoch = n_epoch,
                      n_step  = n_step ,
                      m_epoch = m_epoch,
                      m_error = m_error,
                      best    = best   ,
                      dc      = dc     ,
                      b       = b      ,
                      p       = p      ,
                      q       = q      ))

    # call callback function
    try:
        if callback is not None:
            callback(0, 0, error_t[-1], error_v[-1],
                     info=dict(n_epoch = n_epoch,
                               n_step  = n_step ,
                               m_epoch = m_epoch,
                               m_error = m_error,
                               best    = best   ,
                               dc      = dc     ,
                               b       = b      ,
                               p       = p      ,
                               q       = q      ))
    except StopIteration:
        n_epoch = 0

    # define main loop of optimizer
    for epoch in xrange(1, 2**32 - 1):

        # compute gradient and inverted diagonal of precondition matrix
        b  = grad(d_set, costs=None)
        pc = pcon(c_set, dc)

        # define buffers to store results of the quadratic
        # objective
        q_all = []

        # define buffers to store results of the quadratic
        # objective, update directions and step index at which
        # these values were taken for backtracking
        q_bt_all = []
        p_bt_all = []
        j_bt_all = []

        # optimize sub-objective and iterate through all conjugated directions
        for j, (p, r) in enumerate(_hf_iter_cg(A = lambda v: Avec(c_set, v, dc),
                                               b = -b,
                                               x =  p,
                                               M = lambda v: pc * v,
                                          n_step = n_step)):

            # compute quadratic error of the sub-objective
            # (i.e. 0.5 x'Ax - b'x)
            q = -0.5 * _np.dot(p, -b + r)

            # print progress info
            p_print(epoch, j, info=dict(n_epoch = n_epoch,
                                        n_step  = n_step ,
                                        m_epoch = m_epoch,
                                        m_error = m_error,
                                        dc      = dc     ,
                                        b       = b      ,
                                        p       = p      ,
                                        q       = q      ))

            # append current quadratic error to buffers
            q_all.append(q)

            # break if progress seems to stall
            if 8 <= j and q < 0.0 and (q - q_all[int(0.9 * j)]) / q < tau * j:
                break

            # store curent direction and quadratic error only if we have
            # made (according to gamma) exponentially more steps than
            # the previous time
            if len(j_bt_all) <= _np.ceil(_np.log(j) / _np.log(gamma)):
                p_bt_all.append(p.copy())
                q_bt_all.append(q       )
                j_bt_all.append(j       )

        # compute error of last step and define initial
        # backtracking error
        error_last = cost(d_set, delta=p)
        error_bt   = error_last

        # define initial update direction, quadratic error
        # and interrupt index for backtracking
        p_bt = p
        q_bt = q
        j_bt = j

        # backtrack partial solution until best observed direction
        for k in reversed(xrange(len(j_bt_all))):

            # print progress info of previous backtracking step
            p_print(epoch, j_bt, error_bt, info=dict(n_epoch = n_epoch,
                                                     n_step  = n_step ,
                                                     m_epoch = m_epoch,
                                                     m_error = m_error,
                                                     dc      = dc     ,
                                                     b       = b      ,
                                                     p       = p_bt   ,
                                                     q       = q_bt   ))

            # compute error of current sub-step
            error_k = cost(d_set, delta=p_bt)

            # stop backtracking if the previous error is smaller
            # then the current error
            if error_bt[0] < error_k[0]:
                break

            # set new backtracked error
            error_bt = error_k

            # update initial update direction, quadratic error
            # and interrupt index for backtracking
            p_bt = p_bt_all[k]
            q_bt = q_bt_all[k]
            j_bt = j_bt_all[k]

        # update objective's parameters with backtracked parameters
        objective.push(p_bt)

        # update parameters buffer
        params.append(objective.get())

        # append backtracked training set error to buffer
        error_t.append(error_bt)

        # update optimizer w.r.t. validation set
        if v_set is not None:

            # compute validation set error
            error_v.append(cost(v_set))

            # update best error and parameters if necessary
            if error_v[-1][bestof] < m_error - epsilon:
                n_epoch = max(n_epoch, int(phi * epoch))
                m_epoch = epoch
                m_error = error_v[-1][bestof]
                best    = True
            else:
                best    = False

        # print current state to stdout
        r_print(epoch, j_bt, error_t[-1], error_v[-1],
                info=dict(n_epoch = n_epoch,
                          n_step  = n_step ,
                          m_epoch = m_epoch,
                          m_error = m_error,
                          best    = best   ,
                          dc      = dc     ,
                          b       = b      ,
                          p       = p_bt   ,
                          q       = q_bt   ))

        # call callback function
        try:
            if callback is not None:
                callback(epoch, j_bt, error_t[-1], error_v[-1],
                         info=dict(n_epoch = n_epoch,
                                   n_step  = n_step ,
                                   m_epoch = m_epoch,
                                   m_error = m_error,
                                   best    = best   ,
                                   dc      = dc     ,
                                   b       = b      ,
                                   p       = p_bt   ,
                                   q       = q_bt   ))
        except StopIteration:
            n_epoch = epoch + 1

        # test if patience is exceeded
        if n_epoch is not None and n_epoch <= epoch + 1:
            break

        # test if target error was accomplished
        if t_error is not None and m_error <= t_error:
            break

        # compute ratio between actual error and approximation
        rho = (error_last[0] - error_t[-2][0]) / q

        # update damping coefficient
        if 1.0 - dc_delta < rho:
            dc /= dc_eta
        elif     dc_delta > rho:
            dc *= dc_eta

        # update data sets if possible
        if isinstance(d_set, sample.Abstract):
            d_set.update()
        if isinstance(c_set, sample.Abstract):
            c_set.update()
        if isinstance(v_set, sample.Abstract):
            v_set.update()

    # update optimizer w.r.t. validation set
    if m_epoch is not None:
        objective.set(params[m_error])

    # return params from objective
    if return_info:
        return (objective.get(flatten=False),
                dict(error_t = error_t[1:],
                     error_v = error_v[1:],
                     params  = params     ))
    else:
        return  objective.get(flatten=False)

### ======================================================================== ###

def _ksd_compute_kb(A, b, d=None, M=lambda x: x, n_dim=20,
                    d_type=_theano.config.floatX,
                    callback=lambda p: None):
    r"""
    Function that computes a Krylov basis over the subspace of

    .. math::
        \textbf{P} = [\textbf{b}, \textbf{Ab}, \dots,
                      \textbf{A}^{K-1}\textbf{b}, \textbf{d}_\text{prev}] \text{, }

    whereas :math:`\textbf{A}` is considered to be a curvature matrix (e.g.
    Gauss-Newton or Hessian), :math:`\mathbf{b}` is considered to be the
    gradient of a particular optimization problem, :math:`\textbf{d}_\text{prev}`
    is the previous update direction (i.e. argument :samp:`d`) and :math:`K` is
    the number of dimensions for the Krylov basis (i.e. argument :samp:`n_dim`).

    Please note that in the original paper :math:`\textbf{A}` and
    :math:`\textbf{b}` are defined as :math:`\textbf{H}` and
    :math:`\textbf{g}`, which is different in this implementation due to some
    naming conflicts.

    For further details on how the Krylov basis is being computed, please
    refer to [1]_.

    Parameters
    ----------
    A : callable of A(*numpy array*) -> *numpy array* or *numpy array*
        If callable then this function should return the matrix-vector product
        of :math:`\textbf{Av}`, whereas :math:`\textbf{v}` will be handed over
        to the function as :samp:`A(v)`. If numpy array then this should be a
        matrix :math:`\textbf{A}` that usually refers to some sort of curvature
        matrix (e.g. Gauss-Newton or Hessian).
    b : *numpy array*
        Initial vector that is used for computing the Krylov basis. This vector
        is usually considered to be the gradient of a particular optimization
        problem.
    d : *numpy array*
        If given then :samp:`d` describes the previous update direction
        :math:`\textbf{d}_\text{prev}`, otherwise a random vector will be
        choosen for :math:`\textbf{d}_\text{prev}`. In both cases
        :math:`\textbf{d}_\text{prev}` will be the last dimension of the Krylov
        basis after it was orthogonalized w.r.t. the previous basis vectors.
    M : callable of M(*numpy array*) -> *numpy array* or *numpy array*
        If callable then this function should return the pre-conditioned
        version of :math:`\textbf{v}`, i.e. :math:`\textbf{v}' =
        \text{diag}(\textbf{M})^{-1}\textbf{v}`, whereas :math:`\textbf{v}`
        will be handed over to the function as :samp:`M(v)`. If numpy array
        then this should be the inverted diagonal of the pre-condition matrix
        :math:`\textbf{M}`. By default, this function will not use any pre-\
        conditioning for computing the Krylov basis.
    n_dim : *int*
        Number of dimensions that the Krylov basis should have.
    d_type : *str*
        Data-type that this function should use (e.g. "float32" or "float64").
        By default, this function will use whatever is defined in
        :samp:`theano.config.floatX`.
    callback : callable of callback(*numpy array*)
        Function of :samp:`callback(p)` that is going to be called after each
        dimension that was computed, whereas :samp:`p` is the current
        basis vector.

    Returns
    -------
    P : *numpy array*
        Krylov basis of the subspace.
    H : *numpy array*
        Hessian substitute of the subspace.

    References
    ----------
    .. [1] Oriol Vinyals and Daniel Povey. Krylov Subspace Descent for Deep
        Learning. In Proceedings of the 15th International Conference on
        Artificial Intelligence and Statistics (AISTATS-12), 2012.
    """

    # define default values for callables and previous direction
    if not callable(A):
        A = lambda v: _np.dot(A, v)
    if not callable(M):
        M = lambda v: v / M
    if d is None:
        #d = _np.random.uniform(-1, 1, b.shape).astype(d_type)
        d = _np.ones(b.shape, b.dtype)

    # define euclidean norm and gram-schmidt conjugation functions
    norm = lambda v: _np.sqrt(_np.dot(v, v))
    conj = lambda P, u: reduce(lambda u, p: u - _np.dot(u, p) * p, P, u)

    # define matrices for basis vectors and the hessian of the subspace
    P = _np.zeros((n_dim, b.shape[0]), dtype=d_type)
    H = _np.zeros((n_dim, n_dim     ), dtype=d_type)

    # compute initial basis vector
    p  = M(b)
    p /= norm(p)

    P[0] = p

    # call callback function
    callback(p)

    # compute remaining basis vectors
    for i, j in enumerate(xrange(1, n_dim)):

        # compute curvature matrix-vector product
        w = A(p)
        u = M(w)

        # compute gram-schmidt conjugate of basis vector
        u = conj(P[:j], u)

        # compute row of hessian in the subspace and normalize basis vector
        H[i] = _np.dot(P, w)
        P[j] = p = u / norm(u)

        # call callback function
        callback(p)

    # compute gram-schmidt conjugate of previous direction
    u = conj(P, d)

    # compute final hessian in the subspace and basis vector
    H[-1] = _np.dot(P, d)
    H    += _np.tril(H, -1).T
    P[-1] = p = u / norm(u)

    # call callback function
    callback(p)

    # return result of computation
    return P.T, H

### ------------------------------------------------------------------------ ###

def _ksd_run_bfgs(P, H, d_set, cost, grad, n_step=100,
                  d_type=_theano.config.floatX,
                  callback=lambda p: None):
    r"""
    Function that runs the BFGS method for solving the sub-objective. Please
    refer to [1]_ for a deeper insight on how the objective (returns error)
    and primary (return gradient) functions are being implemented.

    The BFGS method that is used in this function is implemented in SciPy and
    can be found in :func:`scipy.optimize.fmin_bfgs`.

    Parameters
    ----------
    P : *numpy array*
        Matrix that defines the Krylov sub-space.
    H : *numpy array*
        Hessian matrix (or a substitude) that spawn in combination with an
        initial direction (i.e. the gradient) the Krylov sub-space.
    d_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
        Training set used for running the BFGS method.
    cost : callable (see: :func:`_get_cost_func`)
        Function that computes the error of the objective on the training set.
    grad : callable (see: :func:`_get_grad_func`)
        Function that computes the gradient of the objective on the training
        set.
    n_step : *int*
        Maximum number of steps of this sub-optimizer.

    d_type : *str*
        Data-type that this function should use (e.g. "float32" or "float64").
        By default, this function will use whatever is defined in
        :samp:`theano.config.floatX`.
    callback : *callable*
        Function of :samp:`f(p)` that is going to be called after each
        update of the BFGS method, whereas :samp:`p` is the current *parameter
        vector*.

    Returns
    -------
    d : *numpy array*
        Update direction that minimizes the error on the objective.
    accuracy : *float*
        Quadratic norm of the gradient that indicates how well the BFGS
        method has converged on the sub-objective.
    info : *tuple*
        Original return arguments of SciPy's BFGS method.

    References
    ----------
    .. [1] Oriol Vinyals and Daniel Povey. Krylov Subspace Descent for Deep
        Learning. In Proceedings of the 15th International Conference on
        Artificial Intelligence and Statistics (AISTATS-12), 2012.
    """

    # floor hessian substitute matrix and compute cholesky factorization
    L = _np.linalg.cholesky(_floor_matrix(H))

    # define functions for computing search direction
    comp_PC = lambda x: _np.dot(P, _sp.linalg.solve_triangular(L, x.astype(d_type), lower=True, trans="T"))
    comp_g  = lambda x: grad(d_set, delta=comp_PC(x.astype(d_type)))

    # define BFGS functions
    f_obj   = lambda x: cost(d_set, delta=comp_PC(x.astype(d_type)), costs="main")
    f_prime = lambda x: _sp.linalg.solve_triangular(L, _np.dot(P.T, comp_g(x.astype(d_type))), lower=True)

    # define solution vector
    x = _np.zeros(len(L), dtype=d_type)

    # run BFGS on subspace
    info = _sp.optimize.fmin_bfgs(f_obj, x, f_prime, maxiter=n_step,
                                  full_output=True, disp=False, callback=callback)

    # return final update direction, accuracy and entire BFGS result
    return comp_PC(info[0]), _np.dot(info[2], info[2]), info

### ------------------------------------------------------------------------ ###

def ksd(objective, a_set, b_set=None, c_set=None, v_set=None,
        n_epoch=100, n_step=100, n_dim=20,
        m_type="G", pc_type=None,
        t_error=None, epsilon=1e-4, phi=1.0, bestof=0,
        return_info=False, callback=None, formatter=format.ksd):
    r"""
    Function that implements the mind-bending **Krylov Subspace Descent**
    method for optimizing deep and recurrent neural networks.

    Just as described in [1]_ this methods requires three overlapping data
    sets. The first one, which is :samp:`a_set` and that we will refer to as
    the gradient training set, is used to compute the gradient and it should
    cover the entire data set, the second, which is :samp:`b_set` and that we
    will refer to as the curvature matrix training set, is used for computing
    the curvature matrix-vector product and the inverted diagonal of the pre-\
    conditioner matrix, the third and last one, which is :samp:`c_set` and that
    we will refer to as the sub-objective training set, is used while running
    the BFGS method, for cumputing the sub-steps. Both, the curvature matrix
    and the sub-objective training set should overlap with the gradient
    training set, but not with each other.

    The two additional data sets (curvature matrix and sub-objective training
    set) are optional. If you leaving them out this implementation will thake
    the first :math:`1/K` of all samples from the gradient training set for
    the curvature matrix training set and the last :math:`1/K` of all samples
    for the sub-objective training set, whereas :math:`K` is the number of
    dimensions for the subspace, and create random samplers (i.e. instance of
    :class:`sample.Random`) with a batch size of 1000.

    If any of the data sets is an instance of :class:`sample.Abstract` then
    this implementation will update the sample instances of that data set.
    Please note that even if you define the curvature matrix and sub-objective
    training sets to be instances of :class:`sample.Constant` (i.e. an
    implementation of a batched sampler that does not update its sample
    instances) the sample instances of the curvature matrix and sub-objective
    training sets will be updated, since these two data sets overlap with
    with the gradient training set. In fact, the curvature matrix and sub-\
    objective training sets will receive new sample instances from the gradient
    training set and drop others. This is due to the inplace shuffling method
    implemented by numpy.

    Please refer to [1]_, for a deeper insight on how this optimization method
    works.

    Parameters
    ----------
    objective : :class:`Objective`
        Optimization objective that needs to be fulfilled.
    a_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
        Training set used for computing the gradient. This should according to
        [1]_ consist of the full training set.
    b_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
        Training set used for computing the curvature matrix. This should
        according to [1]_ partially consist of the gradient training set
        :samp:`a_set` (i.e. :math:`1/K` of the full training set, whereas
        :math:`K` is the number of dimensions of the subspace), but be
        independent of the BFGS set :samp:`c_set`.
    c_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
        Training set used for running the BFGS optimizer. This should according
        to [1]_ partially consist of the gradient training set :samp:`a_set`
        (i.e. :math:`1/K` of the full training set, whereas :math:`K` is the
        number of dimensions of the subspace), but be independent of the
        curvature set :samp:`b_set`.
    v_set : instance of :class:`sample.Abstract` or [*numpy array*, ...]
        Validation set used for estimating the predictive power of the trained
        model on future data samples. This should be independent of
        :samp:`a_set` and therefore independend of :samp:`b_set` and
        :samp:`c_set`.
    n_epoch : *int*
        Minimum number of update iterations. If the number of iterations
        exceeds this value then the optimizer will quit and return the best
        observed parameters.
    n_step : *int*
        Maximum number of iterations for the BFGS optimizer, that is being used
        to optimize the partial objective, i.e. within the given subspace.
    n_dim : *int*
        Number of dimensions that are being computed for the Krylov subspace.
    m_type : "H", "G" or callable of m_type(...) -> *numpy array*
        Matrix type for the curvature matrix. This can be "G" for the Gauss-\
        Newton matrix, "H" for the Hessian matrix or any callable function of
        :samp:`m_type(b_set, v, c=0.0, delta=None)` that returns a matirx-\
        vector product :math:`\textbf{A}\textbf{v}`, whereas :math:`\textbf{A}`
        denotes the curvature matrix and :math:`\textbf{v}` the vector.
    pc_type : "martens", "jacobi" or callable of pc_type(...) -> *numpy array*
        Pre-conditioner type. This can be "martens" for the pre-conditioner
        described in [2]_, which basically approximates the inverse diagonal
        of the Fisher information matrix, "jacobi" for the pre-conditioner
        described in [3]_, which basically approximates the inverse diagonal
        of the Gauss-Newton matrix, or any callable of :samp:`pc_type(b_set,
        dc=0.0, delta=None)` that returns the inverted diagonal of a pre-\
        condition matrix :math:`\textbf{m} = \text{diag}(\textbf{M})^{-1}`.
        If set to None then this optimizer will not use any pre-conditioning
        on the residuals of the sub-objective.
    t_error : *float*
        Target error. If the error on the validation set is lesser than this
        value, then this optimizer will quit the optimization and return the
        best observed parameters.
    epsilon : *float*
        This value defines the minimum improvement of the objective error
        that is considered to be significant. This value is used to determine
        after which epoch we have observed the best paramters, which only
        applies if you provide a validation set (i.e. :samp:`v_set`) to this
        optimization method.
    phi : *float*
        Factor that governs the patience of this optimization method. If this
        optimization method has observed an improvement w.r.t. the objective
        error (i.e. a new best epoch) then the number of minimum epochs will be
        updated regarding to :math:`N_\min = \max(\phi t, N_\min)`, whereas
        :math:`t` is the index of the current epoch update, :math:`N_\min` is
        the minimum number of epochs and :math:`\phi` is the factor that
        increases the patience. You can as well omit this stopping behaviour if
        you set :samp:`phi` to 1.0.
    bestof : *int*
        Which error from the list of validation set errors should be considered
        when evaluating the best epoch. Default is :samp:`bestof=0`, which will
        consider the main error that is used for optimizing the objective.
    return_info : *bool*
        If this optimizer should return an info dictionary in addition to the
        best observed parameters.
    callback : callable
        Function of :samp:`callback(epoch, accuracy, error_t, error_v, info)`
        that will be called after each epoch (i.e. BFGS run), whereas the call
        arguments are defined as follow:

        epoch : *int*
            The current epoch of this optimization method.
        accuracy : *float*
            The euclidean norm of the gradient that inicates how well the BFGS
            method has converged during the last epoch.
        error_t : *numpy array*
            The errors on the training set taken from the last interrupt.
        error_v : *numpy array*
            The errors on the validation set taken from the last interrupt.
        info : *dict*
            A dictionary that contains some additional information about the
            last iteration.

        Note that the first call of this function might contain some dummy
        values. This function can as well stop the training by raising a
        :class:`StopIteration` exception.
    formatter : callable of formatter(...) -> (callable, callable)
        Function of :samp:`formatter(n_error, n_epoch, n_step, info)` that is
        used for outputing the progress of this optimization method. The call
        arguments are defined as follow:

        n_error : *int*
            The number of errors that should be displayed.
        n_epoch : *int*
            The number of epochs of this optimizer.
        n_step : *int*
            The number of steps of this optimizer.
        info : *dict*
            The original call arguments of this optimization method.

        This function must return a row-print function :samp:`r_print(epoch,
        accuracy, error_t, error_v=None, info={})` and a progress-print
        function :samp:`p_print(step, message=u"", info={})`. The call
        arguments of this two functions are defined as follow:

        epoch : *int*
            The current epoch of this optimization method.
        step : *int*
            The current step of this optimization method.
        accuracy : *float*
            The euclidean norm of the gradient that inicates how well the BFGS
            method has converged during the last epoch.
        error_t : *numpy array*
            The errors on the training set taken from the last interrupt.
        error_v : *numpy array*
            The errors on the validation set taken from the last interrupt.
        info : *dict*
            A dictionary that contains some additional information about the
            last iteration.
        message : *str*
            A message that indicates the current state of the sub-objective.

        Note that currently the :samp:`info` argument in the progress-print
        function is not being used, but might be of use in a future version.
        This optimization method will further not generate any output if you
        set the formatter to None.

    Returns
    -------
    out : [*numpy array*, ...]
        List of parameters that result from optimizing the objective with this
        optimization method.
    info : *dict*
        Optional output that will be returned if this function was called with
        :samp:`return_info=True`. This output contains some additional
        information of every single epoch/step that this optimizer has
        undergone.

    References
    ----------
    .. [1] Oriol Vinyals and Daniel Povey. Krylov Subspace Descent for Deep
        Learning. In Proceedings of the 15th International Conference on
        Artificial Intelligence and Statistics (AISTATS-12), 2012.
    .. [2] James Martens. Deep Learning via Hessian-free Optimization. In
        Proceedings of the 27th International Conference on Machine Learning
        (ICML), 2010.
    .. [3] Olivier Chapelle and Dumitru Erhan. Improved Preconditioner for
        Hessian Free Optimization. In NIPS Workshop on Deep Learning and
        Unsupervised Feature Learning, 2011.

    See Also
    --------

        Class :class:`Objective`
            Implementation of an optimization objective that basically
            describes the optimizaton relevant context.
        Class :class:`sample.Constant`
            Implementation of a constant data sampler, used to draw samples in
            form of batches from an existing data set. This sampler instance
            actually dosn't update its state, even when calling the update
            method.
        Class :class:`sample.Random`
            Implementation of a random data sampler, used to draw samples in
            form of batches from an existing data set. When updated then this
            sampler will shuffle all samples across all its batches.
        Class :class:`sample.Sequence`
            Implementation of a sequence data sampler that is used to draw
            time series in form of batches from an existing time series data
            set. When updated then this sampler will shuffle all its batches
            but withoud shuffling the samples within its batches.
        Class :class:`sample.Interrupt`
            Implementation of a interrupt sampler that wrapps other samplers
            and that calls a callback function when drawing samples in form
            of a batch.
    """

    # define number of errors and number of steps for computing
    # the sub-objective
    n_error = len(objective.costs)
    s_step  = n_dim + n_step + 3

    # get row- and progress-print functions from formatter function
    if formatter is not None:
        r_print, p_print = formatter(n_error, n_epoch, s_step,
                                     info=dict(objective = objective,
                                               a_set     = a_set    ,
                                               b_set     = b_set    ,
                                               c_set     = c_set    ,
                                               v_set     = v_set    ,
                                               n_epoch   = n_epoch  ,
                                               n_step    = n_step   ,
                                               n_dim     = n_dim    ,
                                               m_type    = m_type   ,
                                               pc_type   = pc_type  ,
                                               t_error   = t_error  ,
                                               epsilon   = epsilon  ,
                                               phi       = phi      ,
                                               bestof    = bestof   ,
                                               callback  = callback ))
    else:
        r_print = lambda *args, **kwargs: None
        p_print = lambda *args, **kwargs: None

    # define b_set and/or c_set if missing
    if b_set is None or c_set is None:
        if isinstance(a_set, sample.Abstract):
            smp = a_set.samples
        else:
            smp = a_set
        if b_set is None:
            b_set = sample.Constant([x[:(len(x) / n_dim + 1) ] for x in smp], 1000)
        if c_set is None:
            c_set = sample.Constant([x[-(len(x) / n_dim + 1):] for x in smp], 1000)

    # compile theano functions
    cost = _get_cost_func(objective         )
    grad = _get_grad_func(objective         )
    Avec = _get_Avec_func(objective,  m_type)
    pcon = _get_pcon_func(objective, pc_type)

    # define gradient and previous update direction
    b = None
    d = None

    # define accuracy of BFGS run
    accuracy =  float("inf")

    # define initial target and validation error
    error_t = [[float("inf")] * n_error]
    error_v = [[float("inf")] * n_error]

    # define initial minimum error and epoch
    m_error = float("inf")
    m_epoch = None

    # define indicator for best iteration
    best = False

    # define initial parameters
    params = [objective.get()]

    # define BFGS update info
    info = ()

    # print header to stdout
    r_print(-1, accuracy, error_t[-1], error_v[-1],
            info=dict(n_epoch = n_epoch,
                      m_epoch = m_epoch,
                      m_error = m_error,
                      best    = best   ,
                      info    = info   ,
                      b       = b      ,
                      d       = d      ))

    # define main loop of optimizer
    for epoch in xrange(2**32 - 1):

        # print progress-bar info
        p_print(s_step - 3, u"preparing next iteration")

        # compute gradient and training set error
        b, error = grad(a_set, costs="all")

        # append training set error to buffer
        error_t.append(error)

        # update optimizer w.r.t. validation set
        if v_set is not None:

            # print progress-bar info
            p_print(s_step - 2, u"preparing next iteration")

            # compute validation set error
            error_v.append(cost(v_set))

            # update best error and parameters if necessary
            if error_v[-1][bestof] < m_error - epsilon:
                n_epoch = max(n_epoch, int(phi * epoch))
                m_epoch = epoch
                m_error = error_v[-1][bestof]
                best    = True
            else:
                best    = False

        # print progress-bar info
        p_print(s_step - 1, u"preparing next iteration")

        # compute inverted diagonal of precondition matrix
        pc = pcon(b_set)

        # print current state to stdout
        r_print(epoch, accuracy, error_t[-1], error_v[-1],
                info=dict(n_epoch = n_epoch,
                          m_epoch = m_epoch,
                          m_error = m_error,
                          best    = best   ,
                          info    = info   ,
                          b       = b      ,
                          d       = d      ))

        # call callback function
        try:
            if callback is not None:
                callback(epoch, accuracy, error_t[-1], error_v[-1],
                         info=dict(n_epoch = n_epoch,
                                   m_epoch = m_epoch,
                                   m_error = m_error,
                                   best    = best   ,
                                   info    = info   ,
                                   b       = b      ,
                                   d       = d      ))
        except StopIteration:
            n_epoch = epoch + 1

        # define iteration counter for sub-objective
        count = iter(xrange(0, s_step))

        # compute krylov subspace
        P, H = _ksd_compute_kb(A = lambda v: Avec(b_set, v),
                               M = lambda v: pc * v,
                               b = -b,
                               d =  d,
                           n_dim = n_dim,
                        callback = lambda p: p_print(count.next(), u"computing krylov subspace"))

        # run BFGS on subspace
        d, accuracy, info = _ksd_run_bfgs(P, H, c_set, cost, grad, n_step,
                                          callback=lambda p: p_print(count.next(), u"running bfgs on subspace"))

        # update objective's parameters
        objective.push(d)

        # update parameters buffer
        params.append(objective.get())

        # test if patience is exceeded
        if n_epoch is not None and n_epoch <= epoch + 1:
            break

        # test if target error was accomplished
        if t_error is not None and m_error <= t_error:
            break

        # update data sets if possible
        if isinstance(a_set, sample.Abstract):
            a_set.update()
        if isinstance(b_set, sample.Abstract):
            b_set.update()
        if isinstance(c_set, sample.Abstract):
            c_set.update()
        if isinstance(v_set, sample.Abstract):
            v_set.update()

    # print progress-bar info
    p_print(s_step - 2, u"computing final error")

    # compute final training set error
    error_t.append(cost(a_set))

    # update optimizer w.r.t. validation set
    if v_set is not None:

        # print progress-bar info
        p_print(s_step - 1, u"computing final error")

        # compute final validation set error
        error_v.append(cost(v_set))

        # update best error and parameters if necessary
        if error_v[-1][bestof] < m_error - epsilon:
            m_epoch = epoch + 1
            m_error = error_v[-1][bestof]
            best    = True
        else:
            best    = False

        # set best parameters to objective
        objective.set(params[m_epoch])

    # print current state to stdout
    r_print(epoch + 1, accuracy, error_t[-1], error_v[-1],
            info=dict(n_epoch = n_epoch,
                      m_epoch = m_epoch,
                      m_error = m_error,
                      best    = best   ,
                      info    = info   ,
                      b       = b      ,
                      d       = d      ))

    # call callback function
    try:
        if callback is not None:
            callback(epoch + 1, accuracy, error_t[-1], error_v[-1],
                     info=dict(n_epoch = n_epoch,
                               m_epoch = m_epoch,
                               m_error = m_error,
                               best    = best   ,
                               info    = info   ,
                               b       = b      ,
                               d       = d      ))
    except StopIteration:
        pass

    # return params from objective
    if return_info:
        return (objective.get(flatten=False),
                dict(error_t = error_t[1:],
                     error_v = error_v[1:],
                     params  = params     ))
    else:
        return  objective.get(flatten=False)

### ======================================================================== ###
