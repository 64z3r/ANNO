# -*- coding: utf-8 -*-
r"""
"""

### ======================================================================== ###

import activation

import theano    as _theano
import numpy     as _np

import functools as _functools

from _misc import assert_activation as _assert_activation
from _misc import assert_shape      as _assert_shape
from _misc import to_func           as _to_func
from _misc import to_shared         as _to_shared

from utils import Sandboxed

### ======================================================================== ###

class ExpressionType(type):
    r"""
    Implementation of a meta-class that auto-decorates :samp:`__call__` methods 
    of any class instance that inherits from this type.
    
    Class instances that inherit from this meta-class must implement a 
    :samp:`__call__(self, x)` method with the exact same call signature. This 
    method must further accept theano variables as its call argument or lists 
    of such as its call argument. If further must return a theano variable as 
    its result.
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __new__(cls, name, base, attr):
        r"""
        Method used to override the class instance creation of expression 
        types.
        
        Paramerters
        -----------
        cls : :class:`ExpressionType`
            Class instance of this type implementation.
        name : *str*
            Name of the class instance that needs to be created.
        base : *tuple*
            List of base classes for the class instance that needs to be 
            created.
        attr : *dict*
            Dictionary of attributes for the class instance that needs to 
            be created.
        
        Returns
        -------
        cls : instance of :class:`type`
            Class instance that needs to be created.
        """
        
        # auto-decorate call-method for expression-types
        if "__call__" in attr:
            attr["__call__"] = cls._decorate(attr["__call__"])
        
        # return type-instance of the class
        return super(ExpressionType, cls).__new__(cls, name, base, attr)
    
    ### -------------------------------------------------------------------- ###
    
    @staticmethod
    def _decorate(call):
        r"""
        Implementation of a decorator method for expression-type classes. This 
        method accepts a single callable, wich is supposed to be the 
        :samp:`__call__` method of the targeted expression-type class and 
        returns a decorated version of this method.
        
        The decorated :samp:`__call__` method implements an expression cache 
        so that it will return the same instance of an expression if called 
        with the same input argument. It further implements a function cache, 
        which is used to store compiled functions, in case one calls this 
        method with a numpy array as an input argument.
        
        Parameters
        ----------
        call : callable of call(self, x) -> y
            Methdo that needs to be decorated. This methdo can accept a theano 
            variable as its input or a list/tuple of theano variables and 
            should return a single theano variable or a list of theano 
            variables as its result. Any other input instance will cause the 
            wrapper of this method to raise an error.
        
        Returns
        -------
        wrap : callable of wrap(self, x) -> y
            Wrapper method that decorates the :func:`call` method. This method 
            can accept -- just like the original method -- a theano variable or 
            a list/tuple of theano variables as its input and/or output, but it 
            can as well accept a numpy array or a list/tuple of numpy arrays as 
            its input and it can as well return a single numpy array or a list/\
            tuple of a numpy array, depending on the definition of the 
            expression-type class.
        """
        
        ### ---------------------------------------------------------------- ###
        
        # define function for getting symbolic instances
        def symbolic_of(x, name=None):
            
            # assert correctness of symbolic dimensions
            if not 1 <= x.ndim <= 4:
                raise ValueError("input cannot have lesser than 1 or more "
                                "than 4 dimensions, but got %d" % x.ndim)
            
            # define symbolic types for different dimensions
            symbolic = [_theano.tensor.scalar ,
                        _theano.tensor.vector ,
                        _theano.tensor.matrix ,
                        _theano.tensor.tensor3,
                        _theano.tensor.tensor4]
            
            # return instance of symbolic type
            return symbolic[x.ndim](name, x.dtype)
        
        ### ---------------------------------------------------------------- ###
        
        # define function for matching input instance types
        def is_instance(value, type):
            
            # return test-result of input instance type
            return isinstance(value, type         ) or  \
                   isinstance(value, (list, tuple)) and \
               any(isinstance(x    , type         ) for x in value)
        
        ### ---------------------------------------------------------------- ###
        
        # define function for computing predictions
        def prediction_of(self, value, memo={}):
            
            if isinstance(value, (list, tuple)):
                
                # define the key for accessing the function cache
                key = tuple((x.ndim, x.dtype) for x in value)
                
                # update the function cache if necessary
                if key not in memo:
                    
                    # get symbolic input with respect to the number 
                    # of input dimensions and define output 
                    inputs = [symbolic_of(x, "x") for x in value]
                    output = call(self, inputs)
                    
                    # compile function with respect to the number of 
                    # input dimensions
                    memo[key] = _theano.function(inputs, output, 
                                                 allow_input_downcast=True)
                
                # compute and return the prediction
                return memo[key](*value)
                
            else:
                
                # define the key for accessing the function cache
                key = (value.ndim, value.dtype)
                
                # update the function cache if necessary
                if key not in memo:
                    
                    # get symbolic input with respect to the number 
                    # of input dimensions and define output 
                    input  = symbolic_of(value, "x")
                    output = call(self, input)
                    
                    # compile function with respect to the number of 
                    # input dimensions
                    memo[key] = _theano.function([input], output, 
                                                 allow_input_downcast=True)
                
                # compute and return the prediction
                return memo[key](value)
        
        ### ---------------------------------------------------------------- ###
        
        # define function for computing expressions
        def expression_of(self, value, memo={}):
            
            # define the key for accessing the expression cache
            key = tuple(value) if isinstance(value, (list, tuple)) else \
                        value
            
            # update expression cache
            if key not in memo:
                memo[key] = call(self, value)
            
            # return expression
            return memo[key]
        
        ### ---------------------------------------------------------------- ###
        
        # define method wrapper
        @_functools.wraps(call)
        def wrap(self, x):
            
            # return prediction if possible
            if is_instance(x, _np.ndarray):
                return prediction_of(self, x, self._func_cache)
            
            # return expression if possible
            if is_instance(x, _theano.tensor.Variable):
                return expression_of(self, x, self._expr_cache)
            
            # raise error if input type is unmatched
            raise TypeError("argument 'x' must be an instance of a numpy "
                            "array, a theano variable or a list of either "
                            "one of them")
        
        ### ---------------------------------------------------------------- ###
        
        # return wrapped method
        return wrap
    
    ### -------------------------------------------------------------------- ###

### ======================================================================== ###

class Abstract(object):
    r"""
    Base class that must be implemented by any neural network model and that 
    provides the basic interface that this models must implement.
    """
    
    ### -------------------------------------------------------------------- ###
    
    # all instances of this class are expression-types, meaning that 
    # their call-method will be deocrated with a wrapper that enables 
    # them to cache symbolic expressions and compute predictions
    __metaclass__ = ExpressionType
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, type):
        r"""
        Parameters
        ----------
        type : instance of :class:`Abstract`
            Actual type of the current neural network model.
        """
        
        # define cache attributes, these attributes are required 
        # by the meta-class instance
        self._expr_cache = {}
        self._func_cache = {}
        
        # set instance attributes
        self._type = type
    
    ### -------------------------------------------------------------------- ###
    
    def __call__(self, x):
        r"""
        Method that implements the callable interface and that returns a 
        prediction if :samp:`x` is a numpy array or a symbolic expression if 
        :samp:`x` is a theano variable.
        
        Parameters
        ----------
        x : *numpy array* or *theano variable*
            Input of the model that can be a numpy array or a theano variable. 
            If :samp:`x` is a numpy array then this method will return a 
            prediction :samp:`y` of the input, if on the other hand :samp:`x` 
            is a theano variable then it will return a symbolic expression.
        
        Returns
        -------
        y : *numpy array* or *theano variable*
            Output of the model that can be a numpy array if the input :samp:`x` 
            is an instance of a numpy array or a symbolic expression if the 
            input :samp:`x` is a theano variable.
        
        Note
        ----
        This function will raise a :samp:`NotImplementedError` if invoked.
        """
        
        raise NotImplementedError("cannot call abstract model with input x")
    
    ### -------------------------------------------------------------------- ###
    
    def __getstate__(self):
        r"""
        Implementation of the Pyhton built-in method for getting this 
        model's state.
        """
        
        return dict(type       = self._type      , 
                    expr_cache = self._expr_cache,
                    func_cache = self._func_cache)
    
    def __setstate__(self, state):
        r"""
        Implementation of the Pyhton built-in method for setting this 
        model's state.
        """
        
        self._type       = state["type"      ]
        self._expr_cache = state["expr_cache"]
        self._func_cache = state["func_cache"]
    
    ### -------------------------------------------------------------------- ###
    
    parameters = property(doc="optimization paramters of current model")
    
    @parameters.getter
    def parameters(self):
        raise NotImplementedError("abstract model does not have optimization "
                                  "parameters")
    
    @parameters.setter
    def parameters(self, value):
        raise AttributeError("assignment to read-only attribute 'parameters'")
    
    @parameters.deleter
    def parameters(self):
        raise AttributeError("deletion of read-only attribute 'parameters'")
    
    ### -------------------------------------------------------------------- ###

### ======================================================================== ###

class Primitive(Abstract):
    r"""
    Base class for all primitive neural network models, that can be used to 
    build more complex models.
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, parameters, activation, type):
        r"""
        Parameters
        ----------
        parameters : [*shared theano variable*, ...]
            List of parameters that can be optimized in order to improve this 
            model's performance.
        activation : callable of activation(*theano variable*) -> *theano variable*
            The activation function used by this model.
        type : instance of :class:`Abstract`
            Actual type of current neural network model.
        """
        
        super(Primitive, self).__init__(type)
        self._parameters = parameters
        self._activation = activation
    
    ### -------------------------------------------------------------------- ###
    
    def __getstate__(self):
        r"""
        Implementation of the Pyhton built-in method for getting this 
        model's state.
        """
        
        state = super(Primitive, self).__getstate__()
        state["parameters"] = self._parameters
        state["activation"] = self._activation
        
        return state
    
    def __setstate__(self, state):
        r"""
        Implementation of the Pyhton built-in method for setting this 
        model's state.
        """
        
        super(Primitive, self).__setstate__(state)
        self._parameters = state["parameters"]
        self._activation = state["activation"]
    
    ### -------------------------------------------------------------------- ###
    
    parameters = property(doc="optimization parameters of current model")
    
    @parameters.getter
    def parameters(self):
        return list(self._parameters)
    
    @parameters.setter
    def parameters(self, value):
        raise AttributeError("assignment to read-only attribute 'parameters'")
    
    @parameters.deleter
    def parameters(self):
        raise AttributeError("deletion of read-only attribute 'parameters'")
    
    ### -------------------------------------------------------------------- ###
    
    activation = property(doc="activation function of current model")
    
    @activation.getter
    def activation(self):
        return self._activation
    
    @activation.setter
    def activation(self, value):
        raise AttributeError("assignment to read-only attribute 'activation'")
    
    @activation.deleter
    def activation(self):
        raise AttributeError("deletion of read-only attribute 'activation'")
    
    ### -------------------------------------------------------------------- ###

### ======================================================================== ###

class Composite(Abstract):
    r"""
    Base class for all composite neural network models, that use other model 
    instances to build more complex models.
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, functions, type):
        r"""
        Parameters
        ----------
        functions : [instance of :class:`Abstract`, ...]
            List of neural network models that this decorator composes into 
            a more complex model.
        type : instance of :class:`Composite`
            Actual instance of current neural network model.
        """
        
        super(Composite, self).__init__(type)
        self._functions = functions
    
    ### -------------------------------------------------------------------- ###
    
    def __getstate__(self):
        r"""
        Implementation of the Pyhton built-in method for getting this 
        model's state.
        """
        
        state = super(Composite, self).__getstate__()
        state["functions"] = self._functions
        
        return state
    
    def __setstate__(self, state):
        r"""
        Implementation of the Pyhton built-in method for setting this 
        model's state.
        """
        
        super(Composite, self).__setstate__(state)
        self._functions = state["functions"]
    
    ### -------------------------------------------------------------------- ###
    
    parameters = property(doc="optimization parameters of decorated models")
    
    @parameters.getter
    def parameters(self):
        
        parameters = []
        
        for fn in self._functions:
            for p in fn.parameters:
                if p not in parameters:
                    parameters.append(p)
        
        return parameters
    
    @parameters.setter
    def parameters(self, value):
        raise AttributeError("assignment to read-only attribute 'parameters'")
    
    @parameters.deleter
    def parameters(self):
        raise AttributeError("deletion of read-only attribute 'parameters'")
    
    ### -------------------------------------------------------------------- ###

### ======================================================================== ###

class Collection(Composite):
    r"""
    Base class for all neural network models that provide a iterable collection 
    of other models.
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, functions, type):
        r"""
        Parameters
        ----------
        functions : [instance of :class:`Abstract`, ...]
            List of neural network models that this decorator composes into 
            a more complex model.
        type : derived type of :class:`Collection`
            The class instance of the actual decorator implementation.
        """
        
        super(Collection, self).__init__(functions, type)
    
    ### -------------------------------------------------------------------- ###
    
    def __getitem__(self, index):
        r"""
        Implementation of the Python built-in method for getting a single 
        function with respect to its index from this decorator.
        """
        
        if index is Ellipsis:
            return       list(self._functions)
        if isinstance(index, int):
            return            self._functions.__getitem__(index)
        if isinstance(index, slice):
            return self._type(self._functions.__getitem__(index))
        
        raise TypeError("index must be an ellipsis, integer or slice, but "
                        "got an instance of %r" % type(index).__name__)
    
    def __setitem__(self, index, value):
        r"""
        Implementation of the Python built-in method for setting a single 
        function with respect to its index to this decorator.
        
        Note
        ----
        Decorated functions are defined to be read-only and therefor this 
        method will always raise a :class:`TypeError` exception.
        """
        
        raise TypeError("item assignment to read-only object")
    
    def __delitem__(self, index):
        r"""
        Implementation of the Python built-in method for deleting a single 
        function with respect to its index from this decorator.
        
        Note
        ----
        Decorator functions are defined to be read-only and therefor this 
        method will always raise a :class:`TypeError` exception.
        """
        
        raise TypeError("item deletion of read-only object")
    
    ### -------------------------------------------------------------------- ###
    
    def __len__(self):
        r"""
        Implementation of the Python built-in method for returning the number 
        of functions in this decorator.
        """
        
        return self._functions.__len__()
    
    ### -------------------------------------------------------------------- ###
    
    def __iter__(self):
        r"""
        Implementation of the Python built-in method for iterating over all 
        functions in this decorator.
        """
        
        return self._functions.__iter__()
    
    ### -------------------------------------------------------------------- ###

### ======================================================================== ###

class Fxy(Abstract):
    r"""
    Implementation of a decorator class that can decorate arbitrary neural 
    networks models that implement an :class:`Abstract` interface and 
    transform the input and/or output of a given model. 
    
    Please note that this decorator only provides a limited set of attributes 
    compared to the original instance, but you can access the original 
    instance with :samp:`fx.perceptron`, whereas :class:`fx` denotes the 
    decorator.
    
    An toy-example of how to use this decorator would be:
        
        >>> import numpy
        >>> import anno
        >>> from utils import Sandboxed
        >>> x  = numpy.zeros((5, 1), dtype="float32")
        >>> fn = anno.nnet.Fxy(anno.nnet.ff((3, 2), "tanh"), 
        ...                    Sandboxed(lambda x: x.repeat(3, -1)),
        ...                    Sandboxed(lambda y: y.dimshuffle(1, 0)))
        >>> print fn(x)
        [[ 0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.]]
    
    We could have defined the foregoing decorator without using sandboxed 
    lambda expressions, but in that case we wouldn't be able to pickle this 
    neural network model.
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, fn, fx=Sandboxed(lambda x: x), fy=Sandboxed(lambda y: y)):
        r"""
        Parameters
        ----------
        fn : instance of :class:`Abstract`
            Original model that this class is decorating.
        fx : callable of fx(*theano variable*) -> *theano variable*
            Function that will be called on the input before piping it 
            through the original model.
        fy : callable of fy(*theano variable*) -> *theano variable*
            Function that will be called on the output after pipiing it
            through the original model.
        
        Warning
        -------
        The callables :func:`fx` and :func:`fy` must be pickleable instances 
        if you want to pickle this decorator instance. You can implement your 
        own callable instances for this purpose, or use the 
        :class:`utils.Sandboxed` implementation to serialize and pickle almost 
        any function or lambda expression.
        
        See Also
        --------
        Class :class:`utils.Sandboxed`
            Implementation of a sandboxed callable, that can turn almost any 
            function or lambda expression into a picklable instance.
        """
        
        super(Fxy, self).__init__(Fxy)
        self._fn = fn
        self._fx = fx
        self._fy = fy
    
    ### -------------------------------------------------------------------- ###
    
    def __call__(self, x):
        r"""
        Method that implements the callable interface and that returns a 
        prediction if :samp:`x` is a numpy array or a symbolic expression if 
        :samp:`x` is a theano variable.
        
        Parameters
        ----------
        x : *numpy array* or *theano variable*
            Input of the model that can be a numpy array or a theano variable. 
            If :samp:`x` is a numpy array then this method will return a 
            prediction :samp:`y` of the input, if on the other hand :samp:`x` 
            is a theano variable then it will return a symbolic expression.
        
        Returns
        -------
        y : *numpy array* or *theano variable*
            Output of the model that can be a numpy array if the input :samp:`x` 
            is an instance of a numpy array or a symbolic expression if the 
            input :samp:`x` is a theano variable.
        """
        
        return self._fy(self._fn(self._fx(x)))
    
    ### -------------------------------------------------------------------- ###
    
    def __getstate__(self):
        r"""
        Implementation of the Pyhton built-in method for getting this 
        model's state.
        """
        
        state = super(Fxy, self).__getstate__()
        state["fn"] = self._fn
        state["fx"] = self._fx
        state["fy"] = self._fy
        
        return state
    
    def __setstate__(self, state):
        r"""
        Implementation of the Pyhton built-in method for setting this 
        model's state.
        """
        
        super(Fxy, self).__setstate__(state)
        self._fn = state["fn"]
        self._fx = state["fx"]
        self._fy = state["fy"]
    
    ### -------------------------------------------------------------------- ###
    
    parameters = property(doc="optimization parameters of decorated model")
    
    @parameters.getter
    def parameters(self):
        return self._fn.parameters
    
    @parameters.setter
    def parameters(self, value):
        self._fn.parameters = value
    
    @parameters.deleter
    def parameters(self):
        del self._fn.parameters
    
    ### -------------------------------------------------------------------- ###
    
    fn = property(doc="original model that this class decorates")
    
    @fn.getter
    def fn(self):
        return self._fn
    
    @fn.setter
    def fn(self, value):
        raise AttributeError("assignment to read-only attribute 'fn'")
    
    @fn.deleter
    def fn(self):
        raise AttributeError("deletion of read-only attribute 'fn'")
    
    ### -------------------------------------------------------------------- ###
    
    fx = property(doc="input transformation function")
    
    @fx.getter
    def fx(self):
        return self._fx
    
    @fx.setter
    def fx(self, value):
        raise AttributeError("assignment to read-only attribute 'fx'")
    
    @fx.deleter
    def fx(self):
        raise AttributeError("deletion of read-only attribute 'fx'")
    
    ### -------------------------------------------------------------------- ###
    
    fy = property(doc="output transformation function")
    
    @fy.getter
    def fy(self):
        return self._fy
    
    @fy.setter
    def fy(self, value):
        raise AttributeError("assignment to read-only attribute 'fy'")
    
    @fy.deleter
    def fy(self):
        raise AttributeError("deletion of read-only attribute 'fy'")
    
    ### -------------------------------------------------------------------- ###

### ======================================================================== ###

class Fork(Collection):
    r"""
    Implementation of a decorator class that can fork an input into multiple 
    outputs:
        
        >>> import numpy
        >>> import anno
        >>> x  = numpy.zeros((5, 2), dtype="float32")
        >>> fn = anno.nnet.Fork([anno.nnet.ff((2, 1 + i), "tanh") for i in xrange(3)])
        >>> for y in fn(x):
        ...     print y
        [[ 0.]
         [ 0.]
         [ 0.]
         [ 0.]
         [ 0.]]
        [[ 0.  0.]
         [ 0.  0.]
         [ 0.  0.]
         [ 0.  0.]
         [ 0.  0.]]
        [[ 0.  0.  0.]
         [ 0.  0.  0.]
         [ 0.  0.  0.]
         [ 0.  0.  0.]
         [ 0.  0.  0.]]
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, functions):
        r"""
        Parameters
        ----------
        functions : [instance of :class:`Abstract`, ...]
            List of neural network models that will be used to fork the input 
            to multiple outputs.
        
        See Also
        --------
        Class :class:`Join`
            Implementation of a decorator that can join multiple inputs into 
            a single output.
        """
        
        super(Fork, self).__init__(functions, Fork)
    
    ### -------------------------------------------------------------------- ###
    
    def __call__(self, x):
        r"""
        Method that implements the callable interface and that returns a 
        prediction if :samp:`x` is a numpy array or a symbolic expression if 
        :samp:`x` is a theano variable. 
        
        Note
        ----
        In contrast to other instances of :class:`Abstract` this decorator 
        will map the input to a list of multiple outputs.
        
        Parameters
        ----------
        x : *numpy array* or *theano variable*
            Input of the model that can be a numpy array or a theano variable. 
            If :samp:`x` is a numpy array then this method will return a 
            prediction :samp:`y` of the input, if on the other hand :samp:`x` 
            is a theano variable then it will return a symbolic expression.
        
        Returns
        -------
        y : [*numpy array* or *theano variable*, ...]
            Output of the model that can be a list of numpy arrays if the input 
            :samp:`x` is an instance of a numpy array or a list of symbolic 
            expressions if the input :samp:`x` is a theano variable.
        """
        
        return [fn(x) for fn in self]
    
    ### -------------------------------------------------------------------- ###

### ------------------------------------------------------------------------ ###

class Join(Collection):
    r"""
    Implementation of a decorator class that can join multiple inputs into a 
    single output:
        
        >>> import theano
        >>> import numpy
        >>> import anno
        >>> from utils import Sandboxed
        >>> x  = [numpy.zeros((5, 1 + i), dtype="float32") for i in xrange(3)]
        >>> fn = anno.nnet.Join([anno.nnet.ff((1 + i, 2), "tanh") for i in xrange(3)])
        >>> print fn(x)
        [[ 0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.]]
    
    This decorator will by default concatenate the inner most dimensions of all 
    input-to-output mappings defined in :samp:`functions`, but you can change 
    this behaviour by providing a custom combinator function:
        
        >>> fn = anno.nnet.Join([anno.nnet.ff((1 + i, 2), "tanh") for i in xrange(3)], 
        ...                     combinator=Sandboxed(lambda x: tensor.sum(x, axis=0), 
        ...                                          tensor=theano.tensor))
        >>> print fn(x)
        [[ 0.  0.]
         [ 0.  0.]
         [ 0.  0.]
         [ 0.  0.]
         [ 0.  0.]]
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, functions, combinator=Sandboxed(lambda x: tensor.concatenate(x, axis=-1), tensor=_theano.tensor)):
        r"""
        Parameters
        ----------
        functions : [instance of :class:`Abstract`, ...]
            List of neural network models, which outputs will be used to join 
            them to a single common output.
        combinator : callable of combinator([*theano variable*, ...]) -> *theano variable*
            Function that combines all outputs to a single common output.
        
        Warning
        -------
        The callable :func:`combinator` must be pickleable instance if you want 
        to pickle this decorator instance. You can implement your own callable 
        instance for this purpose, or use the :class:`utils.Sandboxed` 
        implementation to serialize and pickle almost any function or lambda 
        expression.
        
        See Also
        --------
        Class :class:`Fork`
            Implementation of a decorator that can fork an input into multiple 
            outputs.
        Class :class:`utils.Sandboxed`
            Implementation of a sandboxed callable, that can turn almost any 
            function or lambda expression into a picklable instance.
        """
        
        super(Join, self).__init__(functions, Join)
        self._combinator = combinator
    
    ### -------------------------------------------------------------------- ###
    
    def __call__(self, x):
        r"""
        Method that implements the callable interface and that returns a 
        prediction if :samp:`x` is a numpy array or a symbolic expression if 
        :samp:`x` is a theano variable.
        
        Note
        ----
        In contrast to other instances of :class:`Abstract` this decorator 
        will map multiple inputs to a single output.
        
        Parameters
        ----------
        x : [*numpy array* or *theano variable*, ...]
            Input of the model that can be a list of numpy arrays or a theano 
            variables. If :samp:`x` is a list of numpy arrays then this method 
            will return a prediction :samp:`y` of the input, if on the other 
            hand :samp:`x` is a theano variable then it will return a symbolic 
            expression.
        
        Returns
        -------
        y : *numpy array* or *theano variable*
            Output of the model that can be a numpy array if the input 
            :samp:`x` is a list of numpy array instances or a symbolic 
            expression if the input :samp:`x` is a list of theano variables.
        """
        
        return self._combinator([f(c) for f, c in zip(self, x)])
    
    ### -------------------------------------------------------------------- ###
    
    def __getstate__(self):
        r"""
        Implementation of the Pyhton built-in method for getting this 
        decorators state.
        """
        
        state = super(Join, self).__getstate__()
        state["combinator"] = self._combinator
        
        return state
    
    def __setstate__(self, state):
        r"""
        Implementation of the Pyhton built-in method for setting this 
        decorators state.
        """
        
        super(Join, self).__setstate__(state)
        self._combinator = state["combinator"]
    
    ### -------------------------------------------------------------------- ###
    
    combinator = property(doc="combinator function of this decorator")
    
    @combinator.getter
    def combinator(self):
        return self._combinator
    
    @combinator.setter
    def combinator(self, value):
        raise AttributeError("assignment to read-only attribute 'combinator'")
    
    @combinator.deleter
    def combinator(self):
        raise AttributeError("deletion of read-only attribute 'combinator'")
    
    ### -------------------------------------------------------------------- ###

### ======================================================================== ###

class FF(Primitive):
    r"""
    This model represents a single layer of a feedforward neural network that 
    can be used as is or as a basic building block for complex models.
    
    A single-layer feedforward neural network is defined as
    
    .. math::
        
        \mathbf{y} = f_\alpha\left(\mathbf{W}'\mathbf{x} + \mathbf{b}\right).
    
    The activation function of the network is depicted as :math:`f_\alpha`, the 
    weight matrix and bias vector as :math:`\mathbf{W}` and :math:`\mathbf{b}` 
    and the input and output of the network as :math:`\mathbf{x}` and 
    :math:`\mathbf{y}`.
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, W, b, activation, parameters=None):
        r"""
        Parameters
        ----------
        W : *(shared) theano variable*
            The input-to-output weight matrix of the single-layer feedforward 
            neural network.
        b : *(shared) theano variable*
            The bias vector of the single-layer feedforward neural network.
        activation : callable of activation(*theano variable*) -> *theano variable*
            The activation function of the single-layer feedforward neural 
            network.
        parameters : [*shared theano variable*, ...]
            Alternative list of parameters that will be used if not None. This 
            is sometimes necessary, since the parameters can be theano 
            expressions that actually depend on other variables that need to be 
            optimized (e.g. weight matrix of a decoder that is part of an 
            auto-encoder with tied weights). If this argument is None then this 
            model will use the original parameters (i.e. :samp:`W` and 
            :samp:`b`) as its parameters.
        
        See Also
        --------
        Function :func:`ff`
            This function initializes the weight matrix and bias vector of a 
            single-layer feedforward neural network.
        """
        
        # use actual parameters of this model if not defined
        if parameters is None:
            super(FF, self).__init__([W, b]    , activation, FF)
        else:
            super(FF, self).__init__(parameters, activation, FF)
        
        # set parameters of this model
        self._W = W
        self._b = b
    
    ### -------------------------------------------------------------------- ###
    
    def __call__(self, x):
        r"""
        Method that implements the callable interface and that returns a 
        prediction if :samp:`x` is a numpy array or a symbolic expression if 
        :samp:`x` is a theano variable.
        
        Parameters
        ----------
        x : *numpy array* or *theano variable*
            Input of the model that can be a numpy array or a theano variable. 
            If :samp:`x` is a numpy array then this method will return a 
            prediction :samp:`y` of the input, if on the other hand :samp:`x` 
            is a theano variable then it will return a symbolic expression.
        
        Returns
        -------
        y : *numpy array* or *theano variable*
            Output of the model that can be a numpy array if the input :samp:`x` 
            is an instance of a numpy array or a symbolic expression if the 
            input :samp:`x` is a theano variable.
        """
        
        # assert that input x has correct number of dimensions
        if x.ndim > 4:
            raise ValueError("input x cannot have more than 4 dimensions")
        
        # get activation function and model parameters
        activation = self._activation
        W = self._W
        b = self._b
        
        # return expression
        return activation(_theano.tensor.dot(x, W) + b)
    
    ### -------------------------------------------------------------------- ###
    
    def __getstate__(self):
        r"""
        Implementation of the Pyhton built-in method for getting this 
        model's state.
        """
        
        state = super(FF, self).__getstate__()
        state["W"] = self._W
        state["b"] = self._b
        
        return state
    
    def __setstate__(self, state):
        r"""
        Implementation of the Pyhton built-in method for setting this 
        model's state.
        """
        
        super(FF, self).__setstate__(state)
        self._W = state["W"]
        self._b = state["b"]
    
    ### -------------------------------------------------------------------- ###
    
    W = property(doc="weight matrix of current model")
    
    @W.getter
    def W(self):
        return self._W
    
    @W.setter
    def W(self, value):
        raise AttributeError("assignment to read-only attribute 'W'")
    
    @W.deleter
    def W(self):
        raise AttributeError("deletion of read-only attribute 'W'")
    
    ### -------------------------------------------------------------------- ###
    
    b = property(doc="bias vector of current model")
    
    @b.getter
    def b(self):
        return self._b
    
    @b.setter
    def b(self, value):
        raise AttributeError("assignment to read-only attribute 'b'")
    
    @b.deleter
    def b(self):
        raise AttributeError("deletion of read-only attribute 'b'")
    
    ### -------------------------------------------------------------------- ###

### ------------------------------------------------------------------------ ###

def ff(shape, activation="linear", fmt="{param}", rs=_np.random):
    r"""
    Function that initializes the parameters of a single-layer feedforward 
    neural network and that returns its model.
    
    Parameters
    ----------
    shape : [*int*, *int*]
        The shape of a single-layer feedforward neural network. This argument 
        can be an instance of tuple or list and must define two non-zero 
        dimensions. The first dimension defines the input dimensionality (i.e. 
        features) and the second dimension the output dimensionality (i.e. 
        activations).
    activation : *str* or callable of activation(*theano variable*) -> *theano variable*
        The activation function of a single-layer feedforward neural network. 
        This can be "linear", "sigmoid", "tanh", "relu", "softmax" or a callable 
        function. You can as well define a different initialization of the 
        weights, as if you would use an activation function that is different 
        from your actual activation function, e.g. :samp:`"linear<sigmoid>"` 
        will use a :samp:`"linear"` activation function for the output 
        activation and :samp:`"sigmoid"` for the *hypothetical* activation 
        used to initialize the weighs. In case that this argument is a 
        callable function, the weights will always be initialized as if you 
        would be using a linear activation, except when using one of the 
        functions defined in this module.
    fmt : *str*
        String that is used to format parameter names. This function provides 
        during the formatting only one argument named :samp:`param` that 
        basically stands for the parameter name itself. 
        See new-style formatting in Python on how to format parameter names.
    rs : *numpy random state*
        Random state instance that is used to initialize the parameters of this 
        model.
    
    Returns
    -------
    model : :class:`FF`
        Model instance of a single-layer feedforward neural network.
    
    See Also
    --------
    Class :class:`FF`
        Model class of a single-layer feedforward neural network.
    """
    
    # assert shape and actiavtion function
    shape  = _assert_shape(shape, 2)
    fn, to = _assert_activation(activation)
    
    # define symetric range of uniform interval for random initialization
    # w.r.t. the activation function
    if   to == "sigmoid":
        r = _np.sqrt(6.0 / _np.sum(shape)) * 4.0
    elif to == "tanh":
        r = _np.sqrt(6.0 / _np.sum(shape))
    elif to == "relu":
        r = 1.0 / shape[0]
    else:
        r = 1.0
    
    # get actual activation function
    fn = _to_func(fn)
    
    # initialize weights of current perceptron
    W = _to_shared(rs.uniform(-r, r, shape), fmt.format(param="W"))
    b = _to_shared(_np.zeros(shape[1])     , fmt.format(param="b"))
    
    # return single layer perceptron
    return FF(W, b, fn)

### ======================================================================== ###

class R(Primitive):
    r"""
    This model represents a single layer of a recurrent neural network that 
    can be used as is or as a basic building block for complex models.
    
    A single-layer recurrent neural network over a time-series input 
    :math:`\mathbf{x}` is defined as
    
    .. math::
        \mathbf{y}_0 &= \mathbf{h} \\
        \mathbf{y}_t &= f_\alpha\left(\mathbf{W}'\mathbf{x}_t 
                      + \mathbf{U}'\mathbf{y}_{t-1} + \mathbf{b}\right)
    
    The activation function of the network is depicted as :math:`f_\alpha`, the 
    input-to-output and dynamics weight matrices as :math:`\mathbf{W}` and 
    :math:`\mathbf{U}`, the bias vector as :math:`\mathbf{b}` and the input 
    and output of the network as :math:`\mathbf{x}` and :math:`\mathbf{y}`.
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, W, U, b, h, activation, parameters=None):
        r"""
        Parameters
        ----------
        W : *(shared) theano variable*
            The input-to-output weight matrix of the single-layer recurrent 
            neural network.
        U : *(shared) theano variable*
            The dynamics weight matrix of the single-layer recurrent neural 
            network.
        b : *(shared) theano variable*
            The bias vector of the single-layer recurrent neural network.
        h : *(shared) theano variable*
            The initial state of the single-layer recurrent neural network.
        activation : callable of activation(*theano variable*) -> *theano variable*
            The activation function of the single-layer recurrent neural 
            network.
        parameters : [*shared theano variable*, ...]
            Alternative list of parameters that will be used if not None. This 
            is sometimes necessary, since the parameters can be theano 
            expressions that actually depend on other variables that need to be 
            optimized (e.g. weight matrix of a decoder that is part of an 
            auto-encoder with tied weights). If this argument is None then this 
            model will use the original parameters (i.e. :samp:`W`, :samp:`U`, 
            :samp:`b` and :samp:`h`) as its parameters.
        
        See Also
        --------
        Function :func:`r`
            This function initializes the input-to-output and dynamics weight 
            matrices and the bias vector of a single-layer recurrent neural 
            network.
        """
        
        # use actual parameters of this model if not defined
        if parameters is None:
            super(R, self).__init__([W, U, b, h], activation, R)
        else:
            super(R, self).__init__(parameters  , activation, R)
        
        # set parameters of this model
        self._W = W
        self._U = U
        self._b = b
        self._h = h
    
    ### -------------------------------------------------------------------- ###
    
    def __call__(self, x):
        r"""
        Method that implements the callable interface and that returns a 
        prediction if :samp:`x` is a numpy array or a symbolic expression if 
        :samp:`x` is a theano variable.
        
        Parameters
        ----------
        x : *numpy array* or *theano variable*
            Input of the model that can be a numpy array or a theano variable. 
            If :samp:`x` is a numpy array then this method will return a 
            prediction :samp:`y` of the input, if on the other hand :samp:`x` 
            is a theano variable then it will return a symbolic expression.
        
        Returns
        -------
        y : *numpy array* or *theano variable*
            Output of the model that can be a numpy array if the input :samp:`x` 
            is an instance of a numpy array or a symbolic expression if the 
            input :samp:`x` is a theano variable.
        """
        
        # assert that input x has correct number of dimensions
        if x.ndim < 2:
            raise ValueError("input x must have at least 2 dimensions")
        if x.ndim > 4:
            raise ValueError("input x cannot have more than 4 dimensions")
        
        # get activation function and model parameters
        activation = self._activation
        W = self._W
        U = self._U
        b = self._b
        h = self._h
        
        # broadcast initial state to all inputs
        if   x.ndim == 3:
            h += _theano.tensor.zeros((x.shape[1],             h.shape[0]), dtype=h.dtype)
        elif x.ndim == 4:
            h += _theano.tensor.zeros((x.shape[1], x.shape[2], h.shape[0]), dtype=h.dtype)
        
        # define iteration function
        f = lambda x, h: activation(_theano.tensor.dot(x, W) + _theano.tensor.dot(h, U) + b)
        
        # return expression
        return _theano.scan(f, x, h)[0]
    
    ### -------------------------------------------------------------------- ###
    
    def __getstate__(self):
        r"""
        Implementation of the Pyhton built-in method for getting this 
        model's state.
        """
        
        state = super(R, self).__getstate__()
        state["W"] = self._W
        state["U"] = self._U
        state["b"] = self._b
        state["h"] = self._h
        
        return state
    
    def __setstate__(self, state):
        r"""
        Implementation of the Pyhton built-in method for setting this 
        model's state.
        """
        
        super(R, self).__setstate__(state)
        self._W = state["W"]
        self._U = state["U"]
        self._b = state["b"]
        self._h = state["h"]
    
    ### -------------------------------------------------------------------- ###
    
    W = property(doc="input-to-output weight matrix of current model")
    
    @W.getter
    def W(self):
        return self._W
    
    @W.setter
    def W(self, value):
        raise AttributeError("assignment to read-only attribute 'W'")
    
    @W.deleter
    def W(self):
        raise AttributeError("deletion of read-only attribute 'W'")
    
    ### -------------------------------------------------------------------- ###
    
    U = property(doc="dynamics weight matrix of current model")
    
    @U.getter
    def U(self):
        return self._U
    
    @U.setter
    def U(self, value):
        raise AttributeError("assignment to read-only attribute 'U'")
    
    @U.deleter
    def U(self):
        raise AttributeError("deletion of read-only attribute 'U'")
    
    ### -------------------------------------------------------------------- ###
    
    b = property(doc="bias vector of current model")
    
    @b.getter
    def b(self):
        return self._b
    
    @b.setter
    def b(self, value):
        raise AttributeError("assignment to read-only attribute 'b'")
    
    @b.deleter
    def b(self):
        raise AttributeError("deletion of read-only attribute 'b'")
    
    ### -------------------------------------------------------------------- ###
    
    h = property(doc="initial state of current model")
    
    @h.getter
    def h(self):
        return self._h
    
    @h.setter
    def h(self, value):
        raise AttributeError("assignment to read-only attribute 'h'")
    
    @h.deleter
    def h(self):
        raise AttributeError("deletion of read-only attribute 'h'")
    
    ### -------------------------------------------------------------------- ###

### ------------------------------------------------------------------------ ###

def r(shape, activation="linear", fmt="{param}", rs=_np.random):
    r"""
    Function that initializes the parameters of a single-layer recurrent neural 
    network and that returns its model.
    
    Parameters
    ----------
    shape : [*int*, *int*]
        The shape of a single-layer recurrent neural network. This argument 
        can be an instance of tuple or list and must define two non-zero 
        dimensions. The first dimension defines the input dimensionality (i.e. 
        features) and the second dimension the output dimensionality (i.e. 
        activations).
    activation : *str* or callable of activation(*theano variable*) -> *theano variable*
        The activation function of a single-layer recurrent neural network. 
        This can be "linear", "sigmoid", "tanh", "relu", "softmax" or a callable 
        function. You can as well define a different initialization of the 
        weights, as if you would use an activation function that is different 
        from your actual activation function, e.g. :samp:`"linear<sigmoid>"` 
        will use a :samp:`"linear"` activation function for the output 
        activation and :samp:`"sigmoid"` for the *hypothetical* activation 
        used to initialize the weighs. In case that this argument is a 
        callable function, the weights will always be initialized as if you 
        would be using a linear activation, except when using one of the 
        functions defined in this module.
    fmt : *str*
        String that is used to format parameter names. This function provides 
        during the formatting only one argument named :samp:`param` that 
        basically stands for the parameter name itself. 
        See new-style formatting in Python on how to format parameter names.
    rs : *numpy random state*
        Random state instance that is used to initialize the parameters of this 
        model.
    
    Returns
    -------
    model : :class:`R`
        Model instance of a single-layer recurrent neural network.
    
    See Also
    --------
    Class :class:`R`
        Model class of a single-layer recurrent neural network.
    """
    
    # assert shape and actiavtion function
    shape  = _assert_shape(shape, 2)
    fn, to = _assert_activation(activation)
    
    # define symetric range of uniform interval for random initialization
    # w.r.t. the activation function, note: there's no evidence that the 
    # square of the input to hidden dimensions improve the initialization
    # of the network, but choosing the weights to be smaller than the 
    # weights of the dynamics matrix is a good practice.
    if   to == "sigmoid":
        a = _np.sqrt(6.0 / (shape[0]**2 + shape[1]**2)) * 4.0
        b = _np.sqrt(6.0 / (shape[1]    + shape[1]   )) * 4.0
    elif to == "tanh":
        a = _np.sqrt(6.0 / (shape[0]**2 + shape[1]**2))
        b = _np.sqrt(6.0 / (shape[1]    + shape[1]   ))
    elif to == "relu":
        a = 1.0 / shape[0]
        b = 1.0 / shape[1]
    else:
        a = 1.0
        b = 1.0
    
    # get actual activation function
    fn = _to_func(fn)
    
    # initialize weights of current perceptron
    W = _to_shared(rs.uniform(-a, a,  shape              ), fmt.format(param="W"))
    U = _to_shared(rs.uniform(-b, b, (shape[1], shape[1])), fmt.format(param="U"))
    b = _to_shared(_np.zeros(shape[1])                    , fmt.format(param="b"))
    h = _to_shared(_np.zeros(shape[1])                    , fmt.format(param="h"))
    
    # return single layer perceptron
    return R(W, U, b, h, fn)

### ======================================================================== ###

class BD(Primitive):
    r"""
    This model represents a single layer of a bi-directional recurrent neural 
    network that can be used as is or as a basic building block for complex 
    models.
    
    A single-layer bi-directional recurrent neural network over a time-series 
    input :math:`\mathbf{x}` is defined as
    
    .. math::
        \mathbf{h}_0^+ &= \mathbf{u} \\
        \mathbf{h}_n^- &= \mathbf{v} \\
        \mathbf{h}_t\; &= \mathbf{W}'\mathbf{x}_t + \mathbf{b} \\
        \mathbf{h}_t^+ &= f_\alpha\left(\mathbf{h}_t 
                        + \mathbf{U}'\mathbf{h}_{t-1}^+\right) \\
        \mathbf{h}_t^- &= f_\alpha\left(\mathbf{h}_t 
                        + \mathbf{V}'\mathbf{h}_{t+1}^-\right) \\
        \mathbf{y}_t\; &= \mathbf{h}_t^+ +\; \mathbf{h}_t^-
    
    The activation function of the network is depicted as :math:`f_\alpha`, the 
    input-to-output and dynamics weight matrices as :math:`\mathbf{W}`, 
    :math:`\mathbf{U}` and :math:`\mathbf{V}`, the bias vector as 
    :math:`\mathbf{b}` and the input and output of the network as 
    :math:`\mathbf{x}` and :math:`\mathbf{y}`.
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, W, U, V, b, u, v, activation, parameters=None):
        r"""
        Parameters
        ----------
        W : *(shared) theano variable*
            The input-to-output weight matrix of the single-layer 
            bi-directional recurrent neural network.
        U : *(shared) theano variable*
            The forward dynamics weight matrix of the single-layer 
            bi-directional recurrent neural network.
        V : *(shared) theano variable*
            The backward dynamics weight matrix of the single-layer 
            bi-directional recurrent neural network.
        b : *(shared) theano variable*
            The bias vector of the single-layer bi-directional recurrent 
            neural network.
        u : *(shared) theano variable*
            The initial forward state of the single-layer bi-directional 
            recurrent neural network.
        v : *(shared) theano variable*
            The initial backward state of the single-layer bi-directional 
            recurrent neural network.
        activation : callable of activation(*theano variable*) -> *theano variable*
            The activation function of the single-layer bi-directional 
            recurrent neural network.
        parameters : [*shared theano variable*, ...]
            Alternative list of parameters that will be used if not None. This 
            is sometimes necessary, since the parameters can be theano 
            expressions that actually depend on other variables that need to be 
            optimized (e.g. weight matrix of a decoder that is part of an 
            auto-encoder with tied weights). If this argument is None then this 
            model will use the original parameters (i.e. :samp:`W`, :samp:`U`, 
            :samp:`V`, :samp:`b`, :samp:`u` and :samp:`v`) as its parameters.
        
        See Also
        --------
        Function :func:`bd`
            This function initializes the input-to-output and dynamics weight 
            matrices and the bias vector of a single-layer bi-directional 
            recurrent neural network.
        """
        
        # use actual parameters of this model if not defined
        if parameters is None:
            super(BD, self).__init__([W, U, V, b, u, v], activation, BD)
        else:
            super(BD, self).__init__(parameters        , activation, BD)
        
        # set parameters of this model
        self._W = W
        self._U = U
        self._V = V
        self._b = b
        self._u = u
        self._v = v
    
    ### -------------------------------------------------------------------- ###
    
    def __call__(self, x):
        r"""
        Method that implements the callable interface and that returns a 
        prediction if :samp:`x` is a numpy array or a symbolic expression if 
        :samp:`x` is a theano variable.
        
        Parameters
        ----------
        x : *numpy array* or *theano variable*
            Input of the model that can be a numpy array or a theano variable. 
            If :samp:`x` is a numpy array then this method will return a 
            prediction :samp:`y` of the input, if on the other hand :samp:`x` 
            is a theano variable then it will return a symbolic expression.
        
        Returns
        -------
        y : *numpy array* or *theano variable*
            Output of the model that can be a numpy array if the input :samp:`x` 
            is an instance of a numpy array or a symbolic expression if the 
            input :samp:`x` is a theano variable.
        """
        
        # assert that input x has correct number of dimensions
        if x.ndim < 2:
            raise ValueError("input x must have at least 2 dimensions")
        if x.ndim > 4:
            raise ValueError("input x cannot have more than 4 dimensions")
        
        # get activation function and model parameters
        activation = self._activation
        W = self._W
        U = self._U
        V = self._V
        b = self._b
        u = self._u
        v = self._v
        
        # broadcast initial state to all inputs
        if   x.ndim == 3:
            u += _theano.tensor.zeros((x.shape[1],             u.shape[0]), dtype=u.dtype)
            v += _theano.tensor.zeros((x.shape[1],             v.shape[0]), dtype=v.dtype)
        elif x.ndim == 4:
            u += _theano.tensor.zeros((x.shape[1], x.shape[2], u.shape[0]), dtype=u.dtype)
            v += _theano.tensor.zeros((x.shape[1], x.shape[2], v.shape[0]), dtype=v.dtype)
        
        # define iteration functions
        f = lambda h, p, W: activation(h + _theano.tensor.dot(p, W))
        
        # define initial hiddenstate
        h = _theano.tensor.dot(x, W) + b
        
        # return final (propagated) expression
        return _theano.scan(f, h, u, U                   )[0] \
             + _theano.scan(f, h, v, V, go_backwards=True)[0]
    
    ### -------------------------------------------------------------------- ###
    
    def __getstate__(self):
        r"""
        Implementation of the Pyhton built-in method for getting this 
        model's state.
        """
        
        state = super(BD, self).__getstate__()
        state["W"] = self._W
        state["U"] = self._U
        state["V"] = self._V
        state["b"] = self._b
        state["u"] = self._u
        state["v"] = self._v
        
        return state
    
    def __setstate__(self, state):
        r"""
        Implementation of the Pyhton built-in method for setting this 
        model's state.
        """
        
        super(BD, self).__setstate__(state)
        self._W = state["W"]
        self._U = state["U"]
        self._V = state["V"]
        self._b = state["b"]
        self._u = state["u"]
        self._v = state["v"]
    
    ### -------------------------------------------------------------------- ###
    
    W = property(doc="input-to-output weight matrix of current model")
    
    @W.getter
    def W(self):
        return self._W
    
    @W.setter
    def W(self, value):
        raise AttributeError("assignment to read-only attribute 'W'")
    
    @W.deleter
    def W(self):
        raise AttributeError("deletion of read-only attribute 'W'")
    
    ### -------------------------------------------------------------------- ###
    
    U = property(doc="forward dynamics weight matrix of current model")
    
    @U.getter
    def U(self):
        return self._U
    
    @U.setter
    def U(self, value):
        raise AttributeError("assignment to read-only attribute 'U'")
    
    @U.deleter
    def U(self):
        raise AttributeError("deletion of read-only attribute 'U'")
    
    ### -------------------------------------------------------------------- ###
    
    V = property(doc="backward dynamics weight matrix of current model")
    
    @V.getter
    def V(self):
        return self._V
    
    @V.setter
    def V(self, value):
        raise AttributeError("assignment to read-only attribute 'V'")
    
    @V.deleter
    def V(self):
        raise AttributeError("deletion of read-only attribute 'V'")
    
    ### -------------------------------------------------------------------- ###
    
    b = property(doc="bias vector of current model")
    
    @b.getter
    def b(self):
        return self._b
    
    @b.setter
    def b(self, value):
        raise AttributeError("assignment to read-only attribute 'b'")
    
    @b.deleter
    def b(self):
        raise AttributeError("deletion of read-only attribute 'b'")
    
    ### -------------------------------------------------------------------- ###
    
    u = property(doc="initial forward state of current model")
    
    @u.getter
    def u(self):
        return self._u
    
    @u.setter
    def u(self, value):
        raise AttributeError("assignment to read-only attribute 'u'")
    
    @u.deleter
    def u(self):
        raise AttributeError("deletion of read-only attribute 'u'")
    
    ### -------------------------------------------------------------------- ###
    
    v = property(doc="initial backward state of current model")
    
    @v.getter
    def v(self):
        return self._v
    
    @v.setter
    def v(self, value):
        raise AttributeError("assignment to read-only attribute 'v'")
    
    @v.deleter
    def v(self):
        raise AttributeError("deletion of read-only attribute 'v'")
    
    ### -------------------------------------------------------------------- ###

### ------------------------------------------------------------------------ ###

def bd(shape, activation="linear", tied=True, fmt="{param}", rs=_np.random):
    r"""
    Function that initializes the parameters of a single-layer bi-directional 
    recurrent neural network and that returns its model.
    
    Parameters
    ----------
    shape : [*int*, *int*]
        The shape of a single-layer bi-directional recurrent neural network. 
        This argument can be an instance of tuple or list and must define two 
        non-zero dimensions. The first dimension defines the input 
        dimensionality (i.e. features) and the second dimension the output 
        dimensionality (i.e. activations).
    activation : *str* or callable of activation(*theano variable*) -> *theano variable*
        The activation function of a single-layer bi-directional recurrent 
        neural network. This can be "linear", "sigmoid", "tanh", "relu", "softmax" 
        or a callable function. You can as well define a different 
        initialization of the weights, as if you would use an activation 
        function that is different from your actual activation function, e.g. 
        :samp:`"linear<sigmoid>"` will use a :samp:`"linear"` activation 
        function for the output activation and :samp:`"sigmoid"` for the 
        *hypothetical* activation used to initialize the weighs. In case that 
        this argument is a callable function, the weights will always be 
        initialized as if you would be using a linear activation, except when 
        using one of the functions defined in this module.
    tied : *bool*
        If True then this function will initialize the dynamics matrices as a 
        single matrix, by defining the backward dynamics matrix as the 
        transposed of the forward dynamics matrix.
    fmt : *str*
        String that is used to format parameter names. This function provides 
        during the formatting only one argument named :samp:`param` that 
        basically stands for the parameter name itself. 
        See new-style formatting in Python on how to format parameter names.
    rs : *numpy random state*
        Random state instance that is used to initialize the parameters of this 
        model.
    
    Returns
    -------
    model : :class:`BD`
        Model instance of a single-layer bi-directional recurrent neural 
        network.
    
    See Also
    --------
    Class :class:`BD`
        Model class of a single-layer bi-directional recurrent neural network.
    """
    
    # assert shape and actiavtion function
    shape  = _assert_shape(shape, 2)
    fn, to = _assert_activation(activation)
    
    # define symetric range of uniform interval for random initialization
    # w.r.t. the activation function, note: there's no evidence that the 
    # square of the input to hidden dimensions improve the initialization
    # of the network, but choosing the weights to be smaller than the 
    # weights of the dynamics matrix is a good practice.
    if   to == "sigmoid":
        a = _np.sqrt(6.0 / (shape[0]**2 + shape[1]**2)) * 4.0
        b = _np.sqrt(6.0 / (shape[1]    + shape[1]   )) * 4.0
    elif to == "tanh":
        a = _np.sqrt(6.0 / (shape[0]**2 + shape[1]**2))
        b = _np.sqrt(6.0 / (shape[1]    + shape[1]   ))
    elif to == "relu":
        a = 1.0 / shape[0]
        b = 1.0 / shape[1]
    else:
        a = 1.0
        b = 1.0
    
    # get actual activation function
    fn = _to_func(fn)
    
    # initialize weights of current perceptron
    W = _to_shared(rs.uniform(-a, a,  shape              ), fmt.format(param="W"))
    U = _to_shared(rs.uniform(-b, b, (shape[1], shape[1])), fmt.format(param="U"))
    V = _to_shared(rs.uniform(-b, b, (shape[1], shape[1])), fmt.format(param="V")) if not tied else U.T
    b = _to_shared(_np.zeros(shape[1])                    , fmt.format(param="b"))
    u = _to_shared(_np.zeros(shape[1])                    , fmt.format(param="u"))
    v = _to_shared(_np.zeros(shape[1])                    , fmt.format(param="v"))
    
    # define parameter list
    if tied:
        params = [W, U,    b, u, v]
    else:
        params = [W, U, V, b, u, v]
    
    # return single layer perceptron
    return BD(W, U, V, b, u, v, fn, params)

### ======================================================================== ###

class AE(Composite):
    r"""
    This complex model can be used to concatenate two basic models to form an 
    auto-encode neural network and it can be used as a building block for other 
    complex models.
    
    Instances of an auto-encoder neural network are -- like all other instances 
    of :class:`Abstract` -- callable functions that can return a prediction if 
    the input is a numpy array:
    
        >>> import numpy
        >>> import anno
        >>> x  = numpy.random.uniform(-1, 1, (10, 5)).astype("float32")
        >>> fn = anno.nnet.ae((5, 10), ("tanh", "linear"))
        >>> print fn(x)
        [[ 0.97237408  1.00813651  0.9005146   0.88410503 -0.61758912]
         [-0.8699525  -0.4089416  -0.19028392  0.35464552  0.48619726]
         [-0.43080559 -0.23137969  0.41973996  0.76087856  0.20538573]
         [ 1.17796278  1.0344063   1.29984081  0.63868254 -0.13285083]
         [ 1.69219983  1.37676501  1.09395599  0.73955178 -0.79146886]
         [-1.49235773 -1.11560214 -1.33291292 -0.68163711  0.4969312 ]
         [ 1.77924418  0.90828133  1.38461316  0.2303012  -0.84406221]
         [-0.15545847  0.45110205 -0.20935348  0.80708271 -0.05539536]
         [-1.11132979 -0.56336826 -0.92359054 -0.83929408  0.67961287]
         [ 0.3968266   0.11142822 -0.24104252 -0.09834703 -0.41611543]]
    
    or a symbolic variable of the output if the input is a theano variable:
    
        >>> import theano
        >>> x = theano.tensor.matrix("x")
        >>> y = fn(x)
        >>> print theano.pp(y)
        ((tanh(((x \dot encode::W) + encode::b)) \dot encode::W.T) + decode::b)
    
    You can as well encode and decode inputs on their own:
    
        >>> h = fn.encode(x)
        >>> print theano.pp(h)
        tanh(((x \dot encode::W) + encode::b))
        >>> h = theano.tensor.matrix("h")
        >>> y = fn.decode(h)
        >>> print theano.pp(y)
        ((h \dot encode::W.T) + decode::b)
    
    Further more, you can access all optimization specific parameters through 
    the :samp:`fn.parameters` propertiy:
    
        >>> fn.parameters
        [encode::W, encode::b, decode::b]
    
    Note that in the upper examle one might think that we're missing a 
    parameter, namely the weight matrix of the decoder, but the initializer 
    function :func:`ae` will by default a create auto-encoder neural network 
    with tied weights, i.e. the weight matrix of the decoder is just the 
    transposed weight matrix of the encoder:
    
        >>> fn.encode.parameters
        [encode::W, encode::b]
        >>> fn.decode.parameters
        [encode::W, decode::b]
    
    See :func:`ae` for more detail on how to initialize an auto-encoder neural 
    network.
    
    The upper examples use the :func:`ae` initializer function that returns a 
    pre-initialized auto-encoder neural network, but -- just like with multi-\
    layer neural networks -- you can use any instances of :class:`Abstract` 
    to initialize your own, custom auto-encoder neural network:
    
        >>> fn = anno.nnet.AE(anno.nnet.ml((5, 10, 15), "tanh"  , "encode::{param}[{index}]"),
        ...                   anno.nnet.ff((15, 5)    , "linear", "decode::{param}"         ))
        >>> fn.parameters
        [encode::W[0], encode::b[0], encode::W[1], encode::b[1], decode::W, decode::b]
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, encode, decode):
        r"""
        Parameters
        ----------
        encode : instance of :class:`Abstract`
            The encoder of the auto-encoder neural network.
        decode : instance of :class:`Abstract`
            The decoder of the auto-encoder neural network.
        
        See Also
        --------
        Function :func:`ae`
            This function initializes an auto-encoder neural network with basic 
            single-layer neural networks.
        """
        
        super(AE, self).__init__([encode, decode], AE)
        self._encode = encode
        self._decode = decode
    
    ### -------------------------------------------------------------------- ###
    
    def __call__(self, x):
        r"""
        Method that implements the callable interface and that returns a 
        prediction if :samp:`x` is a numpy array or a symbolic expression if 
        :samp:`x` is a theano variable.
        
        Parameters
        ----------
        x : *numpy array* or *theano variable*
            Input of the model that can be a numpy array or a theano variable. 
            If :samp:`x` is a numpy array then this method will return a 
            prediction :samp:`y` of the input, if on the other hand :samp:`x` 
            is a theano variable then it will return a symbolic expression.
        
        Returns
        -------
        y : *numpy array* or *theano variable*
            Output of the model that can be a numpy array if the input :samp:`x` 
            is an instance of a numpy array or a symbolic expression if the 
            input :samp:`x` is a theano variable.
        """
        
        return self._decode(self._encode(x))
    
    ### -------------------------------------------------------------------- ###
    
    def __getstate__(self):
        r"""
        Implementation of the Pyhton built-in method for getting this 
        model's state.
        """
        
        state = super(AE, self).__getstate__()
        state["encode"] = self._encode
        state["decode"] = self._decode
        
        return state
    
    def __setstate__(self, state):
        r"""
        Implementation of the Pyhton built-in method for setting this 
        model's state.
        """
        
        super(AE, self).__setstate__(state)
        self._encode = state["encode"]
        self._decode = state["decode"]
    
    ### -------------------------------------------------------------------- ###
    
    encode = property(doc="encoder of current model")
    
    @encode.getter
    def encode(self):
        return self._encode
    
    @encode.setter
    def encode(self, value):
        raise AttributeError("assignment to read-only attribute 'encode'")
    
    @encode.deleter
    def encode(self):
        raise AttributeError("deletion of read-only attribute 'encode'")
    
    ### -------------------------------------------------------------------- ###
    
    decode = property(doc="decoder of current model")
    
    @decode.getter
    def decode(self):
        return self._decode
    
    @decode.setter
    def decode(self, value):
        raise AttributeError("assignment to read-only attribute 'decode'")
    
    @decode.deleter
    def decode(self):
        raise AttributeError("deletion of read-only attribute 'decode'")
    
    ### -------------------------------------------------------------------- ###

### ------------------------------------------------------------------------ ###

def ae(shape, activation="linear", tied=True, fmt="{type}::{param}", rs=_np.random):
    r"""
    Function that initializes an auto-encoder neural network with simple 
    single-layer neural networks as its encoder and decoder and that returns 
    the newly created auto-encoder model.
    
    Parameters
    ----------
    shape : [*int*, *int*]
        The shape of an auto-encoder neural network. This argument can be an 
        instance of tuple or list and must define two non-zero dimensions. The 
        first dimension defines the input dimensionality (i.e. features) and 
        the second dimension the hidden-state dimensionality (i.e. activations
        of the hidden-states).
    activation: *str* or callable of activation(*theano variable*) -> *theano variable*
        The activation function of an auto-encoder neural network. This can be 
        "linear", "sigmoid", "tanh", "relu", "softmax" or a callable function. 
        All the previous examples will be broadcasted to both, the encoder and 
        the decoder of the auto-encoder neural network (i.e. they will use the 
        same activation function). You can as well define the activation 
        function for the encoder and decoder independently, by using a list or 
        tuple, e.g. :samp:`["tanh", "linear"]`, with a :samp:`"tanh"` 
        activation function for the encoder and a :samp:`"linear"` for the 
        decoder. Further, you can as well define a different initialization 
        of the weights, as if you would use an activation function that is 
        different from your actual activation function, e.g. 
        :samp:`"linear<sigmoid>"` will use a :samp:`"linear"` activation 
        function for the current activation and :samp:`"sigmoid"` for the 
        *hypothetical* activation used to initialize the weighs. In case that 
        this argument is a callable function, the weights will always be 
        initialized as if you would be using a linear activation, except when 
        using one of the functions defined in this module.
    tied : *bool*
        If True then this function will initialize the auto-encoder neural 
        network with tied weight matrices (i.e. the encoder and decoder will 
        share the same weight matrix, so that the weight matrix of the decoder 
        is just the transposed of the encoder's weight matrix). If False then 
        this function will initialize the auto-encoder neural network with 
        independend weight matrices.
    fmt : *str*
        String that is used to format parameter names. This function provides 
        during the formatting two arguments named :samp:`param` and 
        :samp:`type`, whereas :samp:`param` basically stands for the parameter 
        name itself and :samp:`type` can be :samp:`'encode'` if the parameter 
        belongs to the encoder or :samp:`'decode'` if the parameter belongs to 
        the decoder.
        See new-style formatting in Python on how to format parameter names.
    rs : *numpy random state*
        Random state instance that is used to initialize the parameters of this 
        model.
    
    Returns
    -------
    model : :class:`AE`
        Model instance of an auto-encoder neural network.
    
    See Also
    --------
    Class :class:`AE`
        Model class of an auto-encoder neural network.
    """
    
    # assert shape and activation function
    shape  = _assert_shape(shape, 2)
    fn, to = _assert_activation(activation, 2)
    
    # define symetric range of uniform interval for random initialization
    # w.r.t. the activation function
    if   to[0] == "sigmoid":
        r_0 = _np.sqrt(6.0 / _np.sum(shape)) * 4.0
    elif to[0] == "tanh":
        r_0 = _np.sqrt(6.0 / _np.sum(shape))
    elif to[0] == "relu":
        r_0 = 1.0 / shape[0]
    else:
        r_0 = 1.0
    if   to[1] == "sigmoid":
        r_1 = _np.sqrt(6.0 / _np.sum(shape)) * 4.0
    elif to[1] == "tanh":
        r_1 = _np.sqrt(6.0 / _np.sum(shape))
    elif to[1] == "relu":
        r_1 = 1.0 / shape[1]
    else:
        r_1 = 1.0
    
    # get actual activation function
    fn_0 = _to_func(fn[0])
    fn_1 = _to_func(fn[1])
    
    # initialize weights of current perceptron
    W_0 = _to_shared(rs.uniform(-r_0, r_0, shape      ), fmt.format(type="encode", param="W"))
    W_1 = _to_shared(rs.uniform(-r_1, r_1, shape[::-1]), fmt.format(type="decode", param="W")) if not tied else W_0.T
    b_0 = _to_shared(_np.zeros(shape[1])               , fmt.format(type="encode", param="b"))
    b_1 = _to_shared(_np.zeros(shape[0])               , fmt.format(type="decode", param="b"))
    
    # define optimization parameters
    params_0 = [W_0, b_0]
    params_1 = [W_1, b_1] if not tied else [W_0, b_1]
    
    # return auto encoder
    return AE(FF(W_0, b_0, fn_0, params_0),
              FF(W_1, b_1, fn_1, params_1))

### ======================================================================== ###

class ML(Collection):
    r"""
    This complex model can be used to stack other basic models to form a multi-\
    layer neural network and it can be used as a building block for other 
    complex models.
    
    Instances of a multi-layer neural network are -- like all other instances 
    of :class:`Abstract` -- callable functions that can return a prediction 
    if the input is a numpy array:
    
        >>> import numpy
        >>> import anno
        >>> x  = numpy.random.uniform(-1, 1, (10, 5)).astype("float32")
        >>> fn = anno.nnet.ml((5, 10, 1), ("tanh", "linear"))
        >>> print fn(x)
        [[-0.00745437]
         [-1.51680887]
         [-0.17474225]
         [-1.24331784]
         [ 0.43335542]
         [-0.40937495]
         [-0.44526345]
         [-0.08229574]
         [-0.06250013]
         [-0.88697016]]

    or a symbolic variable of the output if the input is a theano variable:
    
        >>> import theano
        >>> x = theano.tensor.matrix("x")
        >>> y = fn(x)
        >>> print theano.pp(y)
        ((tanh(((x \dot W[0]) + b[0])) \dot W[1]) + b[1])
    
    Further more, you can access all optimization specific parameters through 
    the :samp:`fn.parameters` propertiy:
    
        >>> fn.parameters
        [W[0], b[0], W[1], b[1]]
    
    Besides being a callable function, this model implements as well an 
    iterator and an index operator interface to access any layer of the model:
    
        >>> print len(fn)
        2
        >>> for f in fn:
        ...     print repr(f)
        <anno.nnet.FF object at 0x10acd81d0>
        <anno.nnet.FF object at 0x10acd84d0>
        >>> print repr(fn[0])
        <anno.nnet.FF object at 0x10acd81d0>
        >>> print repr(fn[:-1])
        <anno.nnet.ML object at 0x104f76cd0>    
        >>> print len(fn[:-1])
        1
    
    The upper examples use the :func:`ml` initializer function that returns a 
    pre-initialized multi-layer neural network, but you can use any instances 
    of :class:`Abstract` to initialize your own, custom multi-layer neural 
    network:
    
        >>> fn = anno.nnet.ML([anno.nnet.r (( 5, 10), "tanh"  , "{param}[0]"),
        ...                    anno.nnet.ff((10, 10), "tanh"  , "{param}[1]"),
        ...                    anno.nnet.ff((10,  1), "linear", "{param}[2]")])
        >>> fn.parameters
        [W[0], U[0], b[0], h[0], W[1], b[1], W[2], b[2]]
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, functions):
        r"""
        Parameters
        ----------
        functions : [instance of :class:`Abstract`, ...]
            List of :class:`Abstract` instances, which represent the layers 
            that need to be stacked to form a multi-layer neural network.
        
        See Also
        --------
        Function :func:`ml`
            This function initializes a multi-layer neural network with basic 
            single-layer neural networks.
        """
        
        super(ML, self).__init__(functions, ML)
    
    ### -------------------------------------------------------------------- ###
    
    def __call__(self, x):
        r"""
        Method that implements the callable interface and that returns a 
        prediction if :samp:`x` is a numpy array or a symbolic expression if 
        :samp:`x` is a theano variable.
        
        Parameters
        ----------
        x : *numpy array* or *theano variable*
            Input of the model that can be a numpy array or a theano variable. 
            If :samp:`x` is a numpy array then this method will return a 
            prediction :samp:`y` of the input, if on the other hand :samp:`x` 
            is a theano variable then it will return a symbolic expression.
        
        Returns
        -------
        y : *numpy array* or *theano variable*
            Output of the model that can be a numpy array if the input :samp:`x` 
            is an instance of a numpy array or a symbolic expression if the 
            input :samp:`x` is a theano variable.
        """
        
        return reduce(lambda h, fn: fn(h), self, x)
    
    ### -------------------------------------------------------------------- ###

### ------------------------------------------------------------------------ ###

def ml(shape, activation="linear", fmt="{param}[{index}]", rs=_np.random):
    r"""
    Function that initializes a multi-layer neural network with simple single-\
    layer neural networks as its layers and that returns its model.
    
    Parameters
    ----------
    shape : [*int*, *int*, ...]
        The shape of a multi-layer neural network. This argument can be an 
        instance of tuple or list and must define at least two non-zero 
        dimensions. The first dimension defines the input dimensionality 
        (i.e. features), the last dimension the output dimensionality (i.e. 
        activations) and all other dimensions define the hidden-layer 
        dimensionalities of this model.
    activation : *str* or callable of activation(*theano variable*) -> *theano variable*
        The activation function of a multi-layer neural network. This can be 
        "linear", "sigmoid", "tanh", "relu", "softmax" or a callable function. 
        All the previous examples will be broadcasted to all layers of a multi-\
        layer neural network (i.e. they will all use the same activation 
        function). You can as well define the activation function for every 
        single layer as well, by using a list or tuple, e.g. :samp:`["tanh", 
        "tanh", "linear"]` for a 3-layer neural network with :samp:`"tanh"` 
        activation functions for the hidden-layers and :samp:`"linear"` for 
        the output-layer. Further, you can as well define a different 
        initialization of the weights, as if you would use an activation 
        function that is different from your actual activation function, e.g. 
        :samp:`"linear<sigmoid>"` will use a :samp:`"linear"` activation 
        function for the current layer and :samp:`"sigmoid"` for the 
        *hypothetical* activation used to initialize the weighs of that layer. 
        In case that this argument is a callable function, the weights will 
        always be initialized as if you would be using a linear activation, 
        except when using one of the functions defined in this module.
    fmt : *str*
        String that is used to format parameter names. This function provides 
        during the formatting two arguments named :samp:`param` and 
        :samp:`index`, whereas :samp:`param` basically stands for the parameter 
        name itself and :samp:`index` for the index of the layer that these
        parameters belong to. 
        See new-style formatting in Python on how to format parameter names.
    rs : *numpy random state*
        Random state instance that is used to initialize the parameters of this 
        model.
    
    Returns
    -------
    model : :class:`ML`
        Model instance of a multi-layer neural network.
    
    See Also
    --------
    Class :class:`ML`
        Model class of a multi-layer neural network.
    """
    
    # assert shape and activation function
    shape  = _assert_shape(shape)
    fn, to = _assert_activation(activation, len(shape) - 1)
    
    # define layerwise shapes and broadcast activation
    shape  = zip(shape[:-1], shape[1:])
    
    # define container for layers
    functions = []
    
    # initialize all layers
    for i in xrange(len(shape)):
        
        # define symetric range of uniform interval for random initialization
        # w.r.t. the activation function
        if   to[i] == "sigmoid":
            r = _np.sqrt(6.0 / _np.sum(shape[i])) * 4.0
        elif to[i] == "tanh":
            r = _np.sqrt(6.0 / _np.sum(shape[i]))
        elif to[i] == "relu":
            r = 1.0 / shape[i][0]
        else:
            r = 1.0
        
        # initialize weights of current perceptron
        W = _to_shared(rs.uniform(-r, r, shape[i]), fmt.format(index=i, param="W"))
        b = _to_shared(_np.zeros(shape[i][1])     , fmt.format(index=i, param="b"))
        
        # define current layer
        functions.append(FF(W, b, _to_func(fn[i])))
    
    # return multi layer perceptron
    return ML(functions)

### ======================================================================== ###