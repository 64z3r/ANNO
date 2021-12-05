# -*- coding: utf-8 -*-
r"""
"""

### ======================================================================== ###

import theano as _theano

### ======================================================================== ###

class Activation(object):
    r"""
    Implementation of an object that implements multiple activation function. 
    This is necessary since it is not possible to pickle functions in Python.
    
    Currently implemented activation functions are:
    
    ============= =============================================================
     Name          Expression
    ============= =============================================================
     **linear**    :math:`f_\text{linear}(x) \rightarrow x`
     **sigmoid**   :math:`f_\text{sigm}(x) \rightarrow \frac{1}{1 + 
                   \epsilon^{-x}}`
     **tanh**      :math:`f_\text{tanh}(x) \rightarrow \frac{\epsilon^{2x} + 1}
                   {\epsilon^{2x} - 1}`
     **relu**      :math:`f_\text{relu}(x) \rightarrow \max(0, x)`
     **softmax**   :math:`f_\text{softmax}(\textbf{x})_i \rightarrow 
                   \frac{\epsilon^{\textbf{x}_i}}
                   {\sum_{j=1}^{N}\epsilon^{\textbf{x}_j}}`
    ============= =============================================================
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, name):
        r"""
        Parameters
        ----------
        name : str
            Type of activation function. Can be "linear", "sigmoid", "tanh", 
            "relu" or "softmax".
        """
        
        # assert that activation type as the correct value
        if name not in ("linear", "sigmoid", "tanh", "relu", "softmax"):
            raise ValueError("activation type (i.e. type) must be 'linear', "
                             "'sigmoid', 'tanh', 'relu' or 'softmax', "
                             "but got %r" % name)
        
        # set activation type argument
        self._name = name
    
    ### -------------------------------------------------------------------- ###
    
    def __call__(self, x):
        r"""
        Implementation of the Python built-in method for a callable interface 
        that causes this object to behave like a function.
        
        Parameters
        ----------
        x : theano variable or numpy array
            Input arguments used to compute the current activation.
        
        Returns
        -------
        y : theano variable or numpy array
            Result of current activation function given the inputs of :samp:`x`.
        """
        
        # evaluate activation function
        if self._name == "linear":
            return self._linear(x)
        if self._name == "sigmoid":
            return self._sigmoid(x)
        if self._name == "tanh":
            return self._tanh(x)
        if self._name == "relu":
            return self._relu(x)
        if self._name == "softmax":
            return self._softmax(x)
        
        # raise error if we have an unexpected activation type
        raise ValueError("unexpected activation type %r" % self._name)
    
    ### -------------------------------------------------------------------- ###
    
    def __getstate__(self):
        r"""
        Implementation of the Python built-in method for pickling objects.
        """
        
        return self._name
    
    def __setstate__(self, state):
        r"""
        Implementation of the Python built-in method for unpickling objects.
        """
        
        self._name = state
    
    ### -------------------------------------------------------------------- ###
    
    @staticmethod
    def _linear(x):
        r"""
        Implementation of a linear activation function:
        
        .. math::
            f_\text{linear}(x) \rightarrow x
        
        Parameters
        ----------
        x : theano variable or numpy array
            Input arguments used to compute the activation.
        
        Returns
        -------
        y : theano variable or numpy array
            Result of activation function given the inputs of :samp:`x`.
        """
        
        return x
    
    ### -------------------------------------------------------------------- ###
    
    @staticmethod
    def _sigmoid(x):
        r"""
        Implementation of a sigmoid activation function:
        
        .. math::
            f_\text{sigmoid}(x) \rightarrow \frac{1}{1 + \epsilon^{-x}}
        
        Parameters
        ----------
        x : theano variable or numpy array
            Input arguments used to compute the activation.
        
        Returns
        -------
        y : theano variable or numpy array
            Result of activation function given the inputs of :samp:`x`.
        """
        
        return _theano.tensor.nnet.sigmoid(x)
    
    ### -------------------------------------------------------------------- ###
    
    @staticmethod
    def _tanh(x):
        r"""
        Implementation of a tangens hyperbolicus activation function:
        
        .. math::
            f_\text{tanh}(x) \rightarrow \frac{\epsilon^{2x} + 1}
            {\epsilon^{2x} - 1}
        
        Parameters
        ----------
        x : theano variable or numpy array
            Input arguments used to compute the activation.
        
        Returns
        -------
        y : theano variable or numpy array
            Result of activation function given the inputs of :samp:`x`.
        """
        
        return _theano.tensor.tanh(x)
    
    ### -------------------------------------------------------------------- ###
    
    @staticmethod
    def _relu(x):
        r"""
        Implementation of a rectified linear unit:
        
        .. math::
            f_\text{relu}(x) \rightarrow \max(0, x)
        
        Parameters
        ----------
        x : theano variable or numpy array
            Input arguments used to compute the activation.
        
        Returns
        -------
        y : theano variable or numpy array
            Result of activation function given the inputs of :samp:`x`.
        """
        
        return _theano.tensor.maximum(0, x)
    
    ### -------------------------------------------------------------------- ###
    
    @staticmethod
    def _softmax(x):
        r"""
        Implementation of a softmax activation function:
        
        .. math::
            f_\text{softmax}(\textbf{x})_i \rightarrow 
            \frac{\epsilon^{\textbf{x}_i}}
            {\sum_{j=1}^{N}\epsilon^{\textbf{x}_j}}
        
        Parameters
        ----------
        x : theano variable or numpy array
            Input arguments used to compute the activation.
        
        Returns
        -------
        y : theano variable or numpy array
            Result of activation function given the inputs of :samp:`x`.
        """
        
        raise NotImplementedError()
    
    ### -------------------------------------------------------------------- ###
    
    name = property("name of activation function")
    
    @name.getter
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        raise AttributeError("assignment to read-only attribute 'name'")
    
    @name.deleter
    def name(self):
        raise AttributeError("deletion of read-only attribute 'name'")
    
    ### -------------------------------------------------------------------- ###

### ------------------------------------------------------------------------ ###

# instantiate all known activations
linear  = Activation("linear" )
sigmoid = Activation("sigmoid")
tanh    = Activation("tanh"   )
relu    = Activation("relu"   )
softmax = Activation("softmax")

### ======================================================================== ###