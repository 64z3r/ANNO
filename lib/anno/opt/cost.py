# -*- coding: utf-8 -*-
r"""
"""

### ======================================================================== ###

import theano as _theano
import numpy  as _np

### ======================================================================== ###

def mbcee(y, t):
    r"""
    Mean Binary Cross-Entropy Error Cost-Function:
    
    .. math::
        E_\text{MBCEE}(\textbf{y}, \textbf{t}) 
            = \frac{1}{N} \sum -      \textbf{t}  \cdot \log{     \textbf{y}} 
                               - (1 - \textbf{t}) \cdot \log{(1 - \textbf{y})}
    
    Parameters
    ----------
    y : *numpy array* or *theano variable*
        Prediction.
    
    t : *numpy array* or *theano variable*
        Target.
    
    Returns
    -------
    error : *float* or *theano variable*
        Mean binary cross-entropy error of y and t.
    """
    
    # assert that both inputs are compatible
    assert isinstance(y, (_np.ndarray, _theano.tensor.Variable)) \
       and isinstance(t, (_np.ndarray, _theano.tensor.Variable))
    
    # get log-function w.r.t. type
    if isinstance(t, _np.ndarray):
        log = _np.log
    else:
        log = _theano.tensor.log
    
    # return mean error
    return (-t * log(y) - (1 - t) * log(1 - y)).sum(axis=-1).mean()

### ------------------------------------------------------------------------ ###

def mse(y, t):
    r"""
    Mean Squared Error Cost-Function:
    
    .. math::
        E_\text{MSE}(\textbf{y}, \textbf{t}) 
            = \frac{1}{N}\sum(\textbf{y} - \textbf{t})^2
    
    Parameters
    ----------
    y : *numpy array* or *theano variable*
        Prediction.
    
    t : *numpy array* or *theano variable*
        Target.
    
    Returns
    -------
    error : *float* or *theano variable*
        Mean squared error of y and t.
    """
    
    # assert that both inputs are compatible
    assert isinstance(y, (_np.ndarray, _theano.tensor.Variable)) \
       and isinstance(t, (_np.ndarray, _theano.tensor.Variable))
    
    # return error
    return ((y - t)**2).sum(axis=-1).mean()

### ------------------------------------------------------------------------ ###

def rmse(y, t):
    r"""
    Root Mean Squared Error Cost-Function:
    
    .. math::
        E_\text{RMSE}(\textbf{y}, \textbf{t}) 
            = \frac{1}{N}\sqrt{\sum(\textbf{y} - \textbf{t})^2}
    
    Parameters
    ----------
    y : *numpy array* or *theano variable*
        Prediction.
    
    t : *numpy array* or *theano variable*
        Target.
    
    Returns
    -------
    error : *float* or *theano variable*
        Root mean squared error of y and t.
    """
    
    # assert that both inputs are compatible
    assert isinstance(y, (_np.ndarray, _theano.tensor.Variable)) \
       and isinstance(t, (_np.ndarray, _theano.tensor.Variable))
    
    # get sqrt-function w.r.t. type
    if isinstance(t, _np.ndarray):
        sqrt = _np.sqrt
    else:
        sqrt = _theano.tensor.sqrt
    
    # return error
    return sqrt(mse(y, t))

### ======================================================================== ###

def l1(x):
    r"""
    Function used to compute the initial step of the :math:`L_1`-loss:
    
    .. math::
        \bar{L}_1(\textbf{x}) = \left|\textbf{x}\right|
    
    Parameters
    ----------
    x : *numpy array* or *theano variable*
        Value that is used for computing the initial step of the 
        :math:`L_1`-loss.
    
    Returns
    -------
    y : *numpy array* or *theano variable*
        Result of the initial step of the :math:`L_1`-loss.
    
    Note
    ----
    If you want to compute final :math:`L_1`-loss of this function then you 
    will need to sum the result, i.e.
    
    .. math::
        L_1(\textbf{x}) = \sum \bar{L}_1(\textbf{x}) \text{.}
    """
    
    # assert that both inputs are compatible
    assert isinstance(x, (_np.ndarray, _theano.tensor.Variable))
    
    # return the result of the L1-loss
    return abs(x)

### ------------------------------------------------------------------------ ###

def l2(x):
    r"""
    Function used to compute the initial step of the :math:`L_2`-loss:
    
    .. math::
        \bar{L}_2(\textbf{x}) = \left|\textbf{x}\right|
    
    Parameters
    ----------
    x : *numpy array* or *theano variable*
        Value that is used for computing the initial step of the 
        :math:`L_2`-loss.
    
    Returns
    -------
    y : *numpy array* or *theano variable*
        Result of the initial step of the :math:`L_2`-loss.
    
    Note
    ----
    If you want to compute final :math:`L_2`-loss of this function then you 
    will need to sum the result, i.e.
    
    .. math::
        L_2(\textbf{x}) = \sum \bar{L}_2(\textbf{x}) \text{.}
    """
    
    # assert that both inputs are compatible
    assert isinstance(x, (_np.ndarray, _theano.tensor.Variable))
    
    # return the result of the L2-loss
    return x**2

### ------------------------------------------------------------------------ ###

def student(x, kappa=1.0):
    r"""
    Function used to compute the initial step of the student-loss:
    
    .. math::
        \bar{L}_\kappa(\textbf{x}) 
            = \log\left( 1 - \frac{\textbf{x}^2}{\kappa} \right)
    
    Parameters
    ----------
    x : *numpy array* or *theano variable*
        Value that is used for computing the initial step of the student-loss.
    kappa: float
        Variable that controls the spread of the student-loss.
    
    Returns
    -------
    y : *numpy array* or *theano variable*
        Result of the initial step of the studen-loss.
    
    Note
    ----
    If you want to compute final student-loss of this function then you will 
    need to sum the result, i.e.
    
    .. math::
        L_\kappa(\textbf{x}) = \sum \bar{L}_\kappa(\textbf{x}) \text{.}
    """
    
    # assert that both inputs are compatible
    assert isinstance(x, (_np.ndarray, _theano.tensor.Variable))
    
    # get log-function w.r.t. type
    if isinstance(x, _np.ndarray):
        log = _np.log
    else:
        log = _theano.tensor.log
    
    # return the result of the student-loss
    return log(1.0 + x**2/kappa)

### ------------------------------------------------------------------------ ###

def huber(x, delta=0.1):
    r"""
    Function used to compute the initial step of the huber-loss:
    
    .. math::
        \bar{L}_\delta(\textbf{x}) =
        \begin{cases}
            \tfrac{1}{2}\textbf{x}^2, 
                & \text{if } \left|\textbf{x}\right| \leq \delta\\
            \delta(\left|\textbf{x}\right| - \tfrac{1}{2}\delta), 
                & \text{otherwise}
        \end{cases} 
    
    Parameters
    ----------
    x : *numpy array* or *theano variable*
        Value that is used for computing the initial step of the huber-loss.
    delta : *float*
        Variable that controls the ratio between :math:`L^1` and :math:`L^2` 
        losses.
    
    Returns
    -------
    y : *numpy array* or *theano variable*
        Result of the initial step of the huber-loss.
    
    Note
    ----
    If you want to compute final huber-loss of this function then you will need 
    to sum the result, i.e. 
    
    .. math::
        L_\delta(\textbf{x}) = \sum \bar{L}_\delta(\textbf{x}) \text{.}
    """
    
    # assert that both inputs are compatible
    assert isinstance(x, (_np.ndarray, _theano.tensor.Variable))
    
    # get switch-function w.r.t. type
    if isinstance(x, _np.ndarray):
        switch = lambda c, a, b: _np.select([c], [a], b)
    else:
        switch = _theano.tensor.switch
    
    # return the result of the huber-loss
    return switch(abs(x) <= delta, 0.5 * x**2, delta * (abs(x) - 0.5 * delta))

### ------------------------------------------------------------------------ ###

def hybrid(x, delta=0.1):
    r"""
    Function used to compute the initial step of the hybrid-loss:
    
    .. math::
        \bar{L}_\delta(\textbf{x}) 
            \approx \delta^2 \left(
                \sqrt{1 + \left(\frac{\textbf{x}}{\delta}\right)^2} - 1 \right)
    
    This loss function is a differentiable approximation of the huber-loss 
    function.
    
    Parameters
    ----------
    x : *numpy array* or *theano variable*
        Value that is used for computing the initial step of the hybrid-loss.
    delta : *float*
        Variable that controls the ratio between :math:`L^1` and :math:`L^2` 
        losses.
    
    Returns
    -------
    y : *numpy array* or *theano variable*
        Result of the initial step of the hybrid-loss.
    
    Note
    ----
    If you want to compute final hybrid-loss of this function then you will 
    need to sum the result, i.e. 
    
    .. math::
        L_\delta(\textbf{x}) = \sum \bar{L}_\delta(\textbf{x}) \text{.}
    """
    
    # assert that both inputs are compatible
    assert isinstance(x, (_np.ndarray, _theano.tensor.Variable))
    
    # get sqrt-function w.r.t. type
    if isinstance(x, _np.ndarray):
        sqrt = _np.sqrt
    else:
        sqrt = _theano.tensor.sqrt
    
    # return the result of the hybrid-loss
    return delta**2 * (sqrt(1.0 + (x / delta)**2) - 1.0)

### ======================================================================== ###

def contractive(W, h, x, activation="linear", norm=l2, sigma=None, seed=None):
    r"""
    Function used to compute the norm of the Jacobian matrix over all samples:
    
    .. math::
        \left\| \textbf{J} \right\|_2
    
    if :samp:`sigma=None`, otherwise
    
    .. math::
        \left\| \tilde{\textbf{H}} \right\|_2 \text{,}
    
    which is the norm of the approximated Hessian matrix over all samples.
    
    Parameters
    ----------
    W : *theano variable*
        Weight matrix of model.
    h : *theano variable*
        Hidden state activations of model.
    activation : *str*
        Activation function of model, can be "linear", "sigmoid" or "tanh".
    norm : callable of norm(*theano variable*) -> *theano variable*
        Function that computes the norm of the Jacobian/Hessian.
    sigma : *float*
        Variation of normal distribution. If not None then this function will 
        approximate the Hessian instead of computing the Jacobian.
    seed : *int*
        Initial seed for the random number generator. This is only necessary 
        if you want to have reproducable results when approximating the 
        Hessian.
    
    Returns
    -------
    out : *theano variable*
        Contractive penalty for the given model parameters.
    """
    
    # initialize random number generator
    rng = _theano.tensor.shared_randomstreams.RandomStreams(seed)
    
    # compute jacobian matrix
    if   activation == "linear":
        J = W
    elif activation == "sigmoid":
        J = ( h      * (1 - h)).dimshuffle(0, "x", 1) * W.dimshuffle("x", 0, 1)
    elif activation == "tanh":
        J = ((h + 1) * (h - 1)).dimshuffle(0, "x", 1) * W.dimshuffle("x", 0, 1)
    else:
        raise NotImplementedError("unknown activation function '%s'" % activation)
    
    # aproximate hessian matrix
    if sigma is not None:
        n  = rng.normal(size=x.shape, avg=0.0, std=sigma, dtype=x.dtype)
        J -= _theano.clone(J, replace=[(x, x + n)])
    
    # return cost of contractive norm penalty
    return norm(J).sum() / J.shape[0]

### ======================================================================== ###

def orthogonal(a, b):
    r"""
    Function that computes an orthogonal penalty between two sets:
    
    .. math::
        \left\| \bar{\textbf{A}}'\bar{\textbf{B}} \right\|_\textbf{F}^2 \text{,}
    
    where :math:`\bar{\textbf{A}}` is the mean-normalized version of 
    :math:`\textbf{A}` and :math:`\bar{\textbf{B}}` is the mean-normalized 
    version of :math:`\textbf{B}`.
    
    This function basically computes the frobenius norm over all inner products 
    of all feature columns in :math:`\bar{\textbf{A}}` and 
    :math:`\bar{\textbf{B}}`.
    
    Note that this function will only compute the feature orthogonality between 
    corresponding samples of the two sets, i.e. samples that appear at the 
    sampe position.
    
    Parameters
    ----------
    a, b : *numpy array* or *theano variable*
        The data sets used to compute the orthogonal penalty. Both sets must 
        have the same number of samples, i.e. row-entries.
    
    Returns
    -------
    c : *float* or *theano variable*
        Orthogonal penalty of :samp:`a` and :samp:`b`.
    """
    
    # assert that both inputs are compatible
    assert isinstance(a, (_np.ndarray, _theano.tensor.Variable)) \
       and isinstance(b, (_np.ndarray, _theano.tensor.Variable))
    
    # get functions w.r.t. type
    if isinstance(a, _np.ndarray):
        sum = _np.sum
        dot = _np.dot
    else:
        sum = _theano.tensor.sum
        dot = _theano.tensor.dot
    
    # center columns
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    
    # compute distance matrix (ATTENTION: numpy doesn't have dimshuffle!)
    return sum(dot(a.T, b)**2)

### ======================================================================== ###

def orthonormal(a, b):
    r"""
    Function that computes an orthonormal penalty between two sets:
    
    .. math::
        \sum \rho^2 = \sum \left(\frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii}\Sigma_{jj}}}\right)^2
                    = \sum \frac{\Sigma_{ij}^2}{\Sigma_{ii}\Sigma_{jj}} \text{,}
    
    where :math:`\Sigma_{ij} = \bar{\textbf{A}}'\bar{\textbf{B}}`, 
    :math:`\Sigma_{ii} = (\bar{\textbf{A}}'\bar{\textbf{A}})_{ii}`, 
    :math:`\Sigma_{jj} = (\bar{\textbf{B}}'\bar{\textbf{B}})_{jj}`, 
    :math:`\bar{\textbf{A}}` is the mean-normalized version of 
    :math:`\textbf{A}` and :math:`\bar{\textbf{B}}` is the mean-normalized 
    version of :math:`\textbf{B}`.
    
    Note that this function will only compute the feature orthonormality 
    between corresponding samples of the two sets, i.e. samples that appear at 
    the sampe position.
    
    Parameters
    ----------
    a, b : *numpy array* or *theano variable*
        The data sets used to compute the orthonormal penalty. Both sets must 
        have the same number of samples, i.e. row-entries.
    
    Returns
    -------
    c : *float* or *theano variable*
        Orthonormal penalty of :samp:`a` and :samp:`b`.
    """
    
    # assert that both inputs are compatible
    assert isinstance(a, (_np.ndarray, _theano.tensor.Variable)) \
       and isinstance(b, (_np.ndarray, _theano.tensor.Variable))
    
    # get functions w.r.t. type
    if isinstance(a, _np.ndarray):
        sum = _np.sum
        dot = _np.dot
    else:
        sum = _theano.tensor.sum
        dot = _theano.tensor.dot
    
    # center columns
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    
    # compute distance matrix (ATTENTION: numpy doesn't have dimshuffle!)
    d  = dot(a.T, b)**2
    d /= dot(sum(a**2, axis=0).dimshuffle( 0 , "x"),
             sum(b**2, axis=0).dimshuffle("x",  0 )) + 1e-12
    
    # return orthonomal cost
    return sum(d)

### ======================================================================== ###

def mutual(a, b):
    r"""
    Function that computes the mutual information up to the **second order** 
    between two sets:
    
    .. math::
        I(\textbf{A}; \textbf{B}) 
            & = \frac{1}{2}\log\left(\frac{1}{\prod_{ij}(1 - \rho_{ij}^2)}\right) \\
            & = \frac{1}{2}\sum_{ij}\log\left(\frac{1}{1 - \rho_{ij}^2}\right) \text{,}
    
    where
    
    .. math::
        \rho_{ij}^2 = \frac{\Sigma_{ij}^2}{\Sigma_{ii}\Sigma_{jj}}
    
    and :math:`\Sigma_{ij} = \bar{\textbf{A}}'\bar{\textbf{B}}`, 
    :math:`\Sigma_{ii} = (\bar{\textbf{A}}'\bar{\textbf{A}})_{ii}`, 
    :math:`\Sigma_{jj} = (\bar{\textbf{B}}'\bar{\textbf{B}})_{jj}`, 
    :math:`\bar{\textbf{A}}` is the mean-normalized version of 
    :math:`\textbf{A}` and :math:`\bar{\textbf{B}}` is the mean-normalized 
    version of :math:`\textbf{B}`.
    
    Being a **second order** version of the actual mutual information, this 
    function will only produce *reasonable* results if the samples of 
    :math:`\textbf{A}` and :math:`\textbf{B}` are drawn from an gaussian 
    distribution, but you can use this function as well to approximate the 
    actual mutual information.
    
    Note that this function will only compute the feature based mutual 
    information between corresponding samples of the two sets, i.e. samples 
    that appear at the sampe position.
    
    Parameters
    ----------
    a, b : *numpy array* or *theano variable*
        The data sets used to compute the mutual information. Both sets must 
        have the same number of samples, i.e. row-entries.
    
    Returns
    -------
    c : *float* or *theano variable*
        Mutual information of :samp:`a` and :samp:`b`.
    """
    
    # assert that both inputs are compatible
    assert isinstance(a, (_np.ndarray, _theano.tensor.Variable)) \
       and isinstance(b, (_np.ndarray, _theano.tensor.Variable))
    
    # get functions w.r.t. type
    if isinstance(a, _np.ndarray):
        sum = _np.sum
        dot = _np.dot
        log = _np.log
    else:
        sum = _theano.tensor.sum
        dot = _theano.tensor.dot
        log = _theano.tensor.log
    
    # center columns
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    
    # compute distance matrix (ATTENTION: numpy doesn't have dimshuffle!)
    d  = dot(a.T, b)**2
    d /= dot(sum(a**2, axis=0).dimshuffle( 0 , "x"),
             sum(b**2, axis=0).dimshuffle("x",  0 )) + 1e-12
    
    # return orthonomal cost
    return sum(-log(1.0 - d / 2.0))

### ======================================================================== ###

def laplace(A=None, x=None, sigma=2.0, normalized=False):
    r"""
    Function used to compute the graph-laplacian matrix :math:`\textbf{L}`.
    """
    
    # assert that A and x are valid
    assert not (A is None and x is None), "missing input A or x"
    assert     (A is None or  x is None), "A and x are exclusive inputs (either or)"
    
    # get functions w.r.t. type
    if isinstance(A, _np.ndarray) or isinstance(x, _np.ndarray):
        sum  = _np.sum
        exp  = _np.exp
        eye  = _np.eye
        diag = _np.diag
    else:
        sum  = _theano.tensor.sum
        exp  = _theano.tensor.exp
        eye  = _theano.tensor.eye
        diag = _theano.tensor.diag
    
    # compute adjacency matrix
    if A is None:
        A = exp(-sum( (x.dimshuffle(0, "x", 1) 
                     - x.dimshuffle("x", 0, 1))**2, axis=-1) / sigma)
    
    # compute adjacency matrix and degree vector
    A *= 1.0 - eye(*A.shape)
    d  = sum(A, axis=1) + 1e-32
    
    # compute laplacian matrix
    if normalized:
        L = eye(d.size) - ((d**-0.5 * A).T * d**-0.5).T
    else:
        L = diag(d) - A
    
    # return laplacian matrix
    return L

### ======================================================================== ###
