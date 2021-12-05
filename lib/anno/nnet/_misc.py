# -*- coding: utf-8 -*-
r"""
"""

### ======================================================================== ###

import theano    as _theano
import numpy     as _np

import functools as _functools
import marshal   as _marshal
import types     as _types
import re        as _re

from activation import Activation as _Activation

### ======================================================================== ###

def assert_activation(activation, broadcast=None):
    r"""
    Function that asserts an activation function argument and broadcasts it 
    to a given number of instances if necessary. Further, this function will 
    return a tuple containing the actual activation(s) used for the actual 
    activation function(s) and the *hypothetical* activation(s) used for 
    initializing the weights.
    
    Parameters
    ----------
    activation : *str* or callable of activation(*theano variable*) -> *theano variable*
        The activation argument that must be "linear", "sigmoid", "tanh", 
        "relu", "softmax" or a callable function. If the broadcast argument is 
        not None then this argument can be an instance of a tuple or list as 
        well. You can as well define a different initialization of the weights, 
        as if you would use an activation function that is different from your 
        actual activation function, e.g. :samp:`"linear<sigmoid>"` returns 
        :samp:`"linear"` for your actual activation function and 
        :samp:`"sigmoid"` for the *hypothetical* activation function used to 
        initialize the weighs. In case this argument is a callable function, 
        then this function will always return :samp:`"linear"` for the 
        *hypothetical* activation function, unless it has an :samp:`act_type` 
        attribute, which will be returned as the *hypothetical* activation.
    broadcast : *int*
        Number of instances that the activation function must be broadcasted to. 
        If None then this function will not broadcast the activation function, 
        if an instance of int then it will broadcast the activation function to 
        the number of instances provided by the broadcast argument.
    
    Returns
    -------
    fn, to : (*str*, *str*) or ([*str*, ...], [*str*, ...])
        If the broadcast argument is None then this function will return a 
        tuple containing the actual activation and the *hypothetical* 
        activation. If the broadcast argument is an instance of int 
        then this function will try to broadcast all activations to 
        the number provided by the broadcast argument.
    """
    
    # compile regular expression for matching activations
    __activation_rx = _re.compile(r"^(?P<fn>.*?)(?:<(?P<to>.*?)>)?$")
    
    # define function for activation testing
    def __assert_activation(fn, position=None):
        
        # define possible type of error
        if isinstance(fn, str):
            error_type = ValueError
        else:
            error_type = TypeError
        
        # define possible message of error 
        if position is None:
            error_msg = "activation is expected to be 'linear', 'sigmoid', "   \
                        "'tanh', 'relu', 'softmax' or a callable function, "   \
                        "but got %r" % fn
        else:
            error_msg = "activation at position #%d is expected to be "        \
                        "'linear', 'sigmoid', 'tanh', 'relu', 'softmax' or a " \
                        "callabe function, but got %r" % (position, fn)
        
        # return activation as is with linear initializcation
        if callable(fn):
            return fn, getattr(fn, "_act_type", "linear")
        
        # retrieve values from matched expression
        exp = __activation_rx.match(fn)
        fn  = exp.group("fn")
        to  = exp.group("to")
        
        # set weignt activation to be the same as activation
        if to is None:
            to = fn
        
        # assert if activation is valid
        if fn not in ("linear", "sigmoid", "tanh", "relu", "softmax"):
            raise error_type(error_msg)
        if to not in ("linear", "sigmoid", "tanh", "relu", "softmax"):
            raise error_type("weight " + error_msg)
        
        # return value of current activation
        return fn, to
    
    # test activation and broadcast if necessary
    if broadcast is not None:
        if isinstance(activation, (list, tuple)):
            if len(activation) == 1:
                return zip(*[__assert_activation(activation[0])] * broadcast)
            if len(activation) == broadcast:
                return zip(*[__assert_activation(activation[i], i + 1) for i in xrange(broadcast)])
            if   broadcast   == 1:
                raise ValueError("cannot broadcast activation to one layer")
            elif broadcast == 2:
                raise ValueError("cannot broadcast activation to two layers")
            elif broadcast == 3:
                raise ValueError("cannot broadcast activation to three layers")
            else:
                raise ValueError("cannot broadcast activation to %d layers" % broadcast)
        return zip(*[__assert_activation(activation)] * broadcast)
    else:
        return __assert_activation(activation)

### ------------------------------------------------------------------------ ###

def assert_shape(shape, ndim=None):
    r"""
    Function that asserts a given shape argument.
    
    Parameters
    ----------
    shape : [*int*, *int*, ...]
        The shape argument that must contain at least two (or more) non-zero
        dimensions.
    ndim : *int*
        Number of dimensions of the shape, if None then this function will 
        match any shape that has at least two non-zero dimensions, if ndim 
        is an instance of int then this function will match a shape that has 
        exactly the number of ndim non-zero dimensions.
    
    Returns
    -------
    shape : [*int*, *int*, ...]
        The original shape argument.
    """
    
    # assert that size has a correct value
    assert ndim is None or 2 <= ndim
    
    # test instance of shape
    if not isinstance(shape, (list, tuple)):
        raise TypeError("shape must be an instance of list or tuple")
    
    # test number of dimensions
    if ndim is None:
        if not 2 <= len(shape):
            raise ValueError("shape must define at least two dimensions")
    else:
        if not len(shape) == ndim:
            if   ndim == 2:
                raise ValueError("shape must define two dimensions")
            elif ndim == 3:
                raise ValueError("shape must define three dimensions")
            else:
                raise ValueError("shape must define %d dimensions" % ndim)
    
    # test dimensions of shape
    for i, a in enumerate(shape, 1):
        if not isinstance(a, int):
            raise TypeError("dimension #%d of shape must be an instance of int "
                            "but got %s" % (i, type(a).__name__))
        if not 0 < a:
            raise ValueError("dimension #%d of shape must be bigger than zero" % i)
    
    # return value of shape
    return shape

### ======================================================================== ###

def to_func(activation):
    r"""
    Function that returns a callable activation function.
    
    Parameters
    ----------
    activation : *str* or callable of activation(*theano variable*) -> *theano variable*
        The activation argument that can be "linear", "sigmoid", "tanh", 
        "relu", "softmax" or a callable function.
    
    Returns
    -------
    activation : callable of activation(*theano variable*) -> *theano variable*
        Callable function that implements the activation function.
    """
    
    if callable(activation):
        return activation
    if activation in ("linear", "sigmoid", "tanh", "relu", "softmax"):
        return _Activation(activation)
    
    raise ValueError("activation must be 'sigmoid', 'tanh', 'relu', 'softmax' "
                     "or a callable function")

### ------------------------------------------------------------------------ ###

def to_shared(x, name=None, dtype=_theano.config.floatX):
    r"""
    Function that returns a shared theano variable of :samp:`x`. This function 
    will in addition cast :samp:`x` to the correct type.
    
    Parameters
    ----------
    x : *numpy array*
        The numpy array that is used to create the shared variable.
    name : *str*
        The name of the shared variable.
    dtype : *str*
        The type of the shared variable.
    
    Returns
    -------
    out : *shared theano variable*
        The shared variable that was created from :samp:`x`.
    """
    
    return _theano.shared(_theano._asarray(x, dtype), name, borrow=True)

### ======================================================================== ###