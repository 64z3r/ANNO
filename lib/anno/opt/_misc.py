
### ======================================================================== ###

import theano      as _theano
import numpy       as _np

import operator    as _operator

### ======================================================================== ###

class Mapped(object):
    r"""
    Implementation of a mapping object that maps arithmetic operators and 
    method calls to items of a mapper sequence:
        
        >>> import numpy as np
        >>> from anno.opt._misc import Mapped
        >>> x = Mapped([np.ones((4, 5)), np.ones((3, 5)) * 10])
        >>> x
        Mapped([array([[ 1.,  1.,  1.,  1.,  1.],
                       [ 1.,  1.,  1.,  1.,  1.],
                       [ 1.,  1.,  1.,  1.,  1.],
                       [ 1.,  1.,  1.,  1.,  1.]]),
                array([[ 10.,  10.,  10.,  10.,  10.],
                       [ 10.,  10.,  10.,  10.,  10.],
                       [ 10.,  10.,  10.,  10.,  10.]])])
        >>> x.shape
        Mapped([(4, 5), (3, 5)])
        >>> y = x.sum(axis=0)
        >>> y
        Mapped([array([ 4.,  4.,  4.,  4.,  4.]), array([ 30.,  30.,  30.,  30.,  30.])])
        >>> y + 1
        Mapped([array([ 5.,  5.,  5.,  5.,  5.]), array([ 31.,  31.,  31.,  31.,  31.])])
        >>> y.dot(np.arange(5))
        Mapped([40.0, 300.0])
    
    In addition to mapping arithmetic operators and method calls, you can as 
    well map index operations, but with some restrictions: the first index is 
    always going to be used for indexing items in the mapper sequence:
        
        >>> y
        Mapped([array([ 4.,  4.,  4.,  4.,  4.]), array([ 30.,  30.,  30.,  30.,  30.])])
        >>> y[0]
        array([ 4.,  4.,  4.,  4.,  4.])
    
    You can as well use slices, which will return you an list of all items:
        
        >>> y[:-1]
        [array([ 4.,  4.,  4.,  4.,  4.])
    
    If you use an ellipsis, you will get an other :class:`Mapped` instance. 
    This is especially usefull if you want to index items in the 
    :class:`Mapped` instance:
        
        >>> y[..., 0:3]
        Mapped([array([ 4.,  4.,  4.]), array([ 30.,  30.,  30.])])
    
    Operands that are not an instance of :class:`Mapped` will be broadcasted to 
    all items of an :class:`Mapped` instance:
        
        >>> a = Mapped([np.ones((3, 2)), np.ones((1, 2))])
        >>> a
        Mapped([array([[ 1.,  1.],
                       [ 1.,  1.],
                       [ 1.,  1.]]),
                array([[ 1.,  1.]])])
        >>> a.dot(np.arange(1, 3))
        Mapped([array([ 3.,  3.,  3.]), array([ 3.])])
    
    But if both operands are instances of :class:`Mapped` then the behaviour is 
    more like what you would expect from :func:`map`:
        
        >>> b = Mapped([np.arange(1, 3), np.arange(3, 5)])
        Mapped([array([1, 2]), array([3, 4])])
        >>> a.dot(b)
        Mapped([array([ 3.,  3.,  3.]), array([ 7.])])
        >>> map(lambda x, y: x.dot(y), a, b)
        [array([ 3.,  3.,  3.]), array([ 7.])]
    
    By the same means you can as well map or broadcast arguments in method 
    calls:
        
        >>> a.sum(axis=0)
        Mapped([array([ 3.,  3.]), array([ 1.,  1.])])
        >>> a.sum(axis=Mapped([0, 1]))
        Mapped([array([ 3.,  3.]), array([ 2.])])
    
    Certain function are not being mapped to the items of a :class:`Mapped` 
    instance. These are lenght of a mapper, iterator operations on a mapper 
    and contains operations on a mapper:
        
        >>> len(a)
        2
        >>> for x in a:
        ...     print x
        [[ 1.  1.]
         [ 1.  1.]
         [ 1.  1.]]
        [[ 1.  1.]]
        >>> a[0] in a
        True
        >>> 0 in a:
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "anno/opt/_misc.py", line 339, in __contains__"
            string representation of this mapper.
        ValueError: The truth value of an array with more than one element is 
        ambiguous. Use a.any() or a.all()
    
    The last error may appear surprising, but this is exactly what one would 
    expect when working with numpy arrays:
        
        >>> b = a[:]
        >>> b
        [array([[ 1.,  1.],
                [ 1.,  1.],
                [ 1.,  1.]]), array([[ 1.,  1.]])]
        >>> 0 in b
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        ValueError: The truth value of an array with more than one element is 
        ambiguous. Use a.any() or a.all()
    
    As we have seen, certain functions cannot be mapped or broadcasted to the 
    items of a :class:`Mapped` instance. Therefor the :class:`Mapped` instance 
    implements a method that will let you broadcast arbitrary functions to the 
    items defined in a :class:`Mapped` instance:
        
        >>> Mapped.apply(len, a)
         Mapped([3, 1])
    
    You can also use this method to call functions on mapped items that are not 
    implemented as methods:
        
        >>> Mapped.apply(np.concatenate, Mapped.apply(zip, a, a), axis=0)
        Mapped([array([[ 1.,  1.],
                       [ 1.,  1.],
                       [ 1.,  1.],
                       [ 1.,  1.],
                       [ 1.,  1.],
                       [ 1.,  1.]]),
                array([[ 1.,  1.],
                       [ 1.,  1.]])])
    
    Or a little bit simpler:
        
        >>> Mapped.apply(np.append, a, a, axis=0)
        Mapped([array([[ 1.,  1.],
                       [ 1.,  1.],
                       [ 1.,  1.],
                       [ 1.,  1.],
                       [ 1.,  1.],
                       [ 1.,  1.]]),
                array([[ 1.,  1.],
                       [ 1.,  1.]])])
    """
    
    ### -------------------------------------------------------------------- ###
    
    # define unary operations
    __abs__       = lambda self: self.apply(_operator.__abs__, self)
    __pos__       = lambda self: self.apply(_operator.__pos__, self)
    __neg__       = lambda self: self.apply(_operator.__neg__, self)
    __not__       = lambda self: self.apply(_operator.__not__, self)
    __inv__       = lambda self: self.apply(_operator.__inv__, self)
    
    # define comparison operations
    __eq__        = lambda self, other: self.apply(_operator.__eq__       , self, other)
    __ne__        = lambda self, other: self.apply(_operator.__ne__       , self, other)
    __ge__        = lambda self, other: self.apply(_operator.__ge__       , self, other)
    __gt__        = lambda self, other: self.apply(_operator.__gt__       , self, other)
    __le__        = lambda self, other: self.apply(_operator.__le__       , self, other)
    __lt__        = lambda self, other: self.apply(_operator.__lt__       , self, other)

    # define bit operations
    __and__       = lambda self, other: self.apply(_operator.__and__      , self, other)
    __or__        = lambda self, other: self.apply(_operator.__or__       , self, other)
    __xor__       = lambda self, other: self.apply(_operator.__xor__      , self, other)
    __lshift__    = lambda self, other: self.apply(_operator.__lshift__   , self, other)
    __rshift__    = lambda self, other: self.apply(_operator.__rshift__   , self, other)
    
    # define arithmetic operation
    __add__       = lambda self, other: self.apply(_operator.__add__      , self, other)
    __sub__       = lambda self, other: self.apply(_operator.__sub__      , self, other)
    __mul__       = lambda self, other: self.apply(_operator.__mul__      , self, other)
    __div__       = lambda self, other: self.apply(_operator.__div__      , self, other)
    __mod__       = lambda self, other: self.apply(_operator.__mod__      , self, other)
    __pow__       = lambda self, other: self.apply(_operator.__pow__      , self, other)
    __divmod__    = lambda self, other: self.apply(_operator.__divmod__   , self, other)
    __truediv__   = lambda self, other: self.apply(_operator.__truediv__  , self, other)
    __floordiv__  = lambda self, other: self.apply(_operator.__floordiv__ , self, other)
    
    # define right-hand bit operations
    __rand__      = lambda self, other: self.apply(_operator.__and__      , self, other)
    __ror__       = lambda self, other: self.apply(_operator.__or__       , self, other)
    __rxor__      = lambda self, other: self.apply(_operator.__xor__      , self, other)
    __rlshift__   = lambda self, other: self.apply(_operator.__lshift__   , self, other)
    __rrshift__   = lambda self, other: self.apply(_operator.__rshift__   , self, other)
    
    # define right-hand arithmetic operations
    __radd__      = lambda self, other: self.apply(_operator.__add__      , self, other)
    __rsub__      = lambda self, other: self.apply(_operator.__sub__      , self, other)
    __rmul__      = lambda self, other: self.apply(_operator.__mul__      , self, other)
    __rdiv__      = lambda self, other: self.apply(_operator.__div__      , self, other)
    __rmod__      = lambda self, other: self.apply(_operator.__mod__      , self, other)
    __rpow__      = lambda self, other: self.apply(_operator.__pow__      , self, other)
    __rdivmod__   = lambda self, other: self.apply(_operator.__divmod__   , self, other)
    __rtruediv__  = lambda self, other: self.apply(_operator.__truediv__  , self, other)
    __rfloordiv__ = lambda self, other: self.apply(_operator.__floordiv__ , self, other)
    
    # define in-place bit operations
    __iand__      = lambda self, other: self.apply(_operator.__iand__     , self, other)
    __ior__       = lambda self, other: self.apply(_operator.__ior__      , self, other)
    __ixor__      = lambda self, other: self.apply(_operator.__ixor__     , self, other)
    __ilshift__   = lambda self, other: self.apply(_operator.__ilshift__  , self, other)
    __irshift__   = lambda self, other: self.apply(_operator.__irshift__  , self, other)
    
    # define in-place arithmetic operations
    __iadd__      = lambda self, other: self.apply(_operator.__iadd__     , self, other)
    __isub__      = lambda self, other: self.apply(_operator.__isub__     , self, other)
    __imul__      = lambda self, other: self.apply(_operator.__imul__     , self, other)
    __idiv__      = lambda self, other: self.apply(_operator.__idiv__     , self, other)
    __imod__      = lambda self, other: self.apply(_operator.__imod__     , self, other)
    __ipow__      = lambda self, other: self.apply(_operator.__ipow__     , self, other)
    __itruediv__  = lambda self, other: self.apply(_operator.__itruediv__ , self, other)
    __ifloordiv__ = lambda self, other: self.apply(_operator.__ifloordiv__, self, other)
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, sequence):
        r"""
        Parameters
        ----------
        sequence : [...]
            Sequence of items that has to be mapped.
        """
        
        # assert that sequence is iterable
        try:
            sequence = list(sequence)
        except TypeError:
            raise TypeError("sequence must be an iterable instance")
        
        # assert that all instances are of the sampe type
        if not all(type(sequence[0]) == type(x) for x in sequence[1:]):
            raise ValueError("items in sequence must be of the same type")
        
        # add sequence attribute
        self.__dict__["_sequence"] = list(sequence)
    
    ### -------------------------------------------------------------------- ###
    
    def __repr__(self):
        r"""
        Implementation of the Python built-in method for getting the 
        representation string of this object.
        """
        
        # define output and pre-format buffer
        out = []
        fmt = []
        
        # format all items in this sequence
        for x in self._sequence:
            fmt.append(repr(x).split("\n"))
        
        # format final content of mapper
        for i in xrange(len(fmt)):
            
            # append seperator with respect to lines
            if not i == 0:
                if 1 < len(fmt[i]) or 1 < len(fmt[i-1]):
                    out.append(",\n        ")
                else:
                    out.append(", "         )
            
            # append formated representation to buffer
            out.append("\n        ".join(fmt[i]))
        
        # return formatter string
        return "Mapped([%s])" % "".join(out)
    
    def __str__(self):
        r"""
        Implementation of the Python built-in method for getting the 
        string representation of this object.
        """
        
        # define output and pre-format buffer
        out = []
        fmt = []
        
        # format all items in this sequence
        for x in self._sequence:
            fmt.append( str(x).split("\n"))
        
        # format final content of mapper
        for i in xrange(len(fmt)):
            
            # append seperator with respect to lines
            if not i == 0:
                if 1 < len(fmt[i]) or 1 < len(fmt[i-1]):
                    out.append(",\n        ")
                else:
                    out.append(", "         )
            
            # append formated representation to buffer
            out.append("\n        ".join(fmt[i]))
        
        # return formatter string
        return "Mapped([%s])" % "".join(out)
    
    ### -------------------------------------------------------------------- ###
    
    def __iter__(self):
        r"""
        Implementation of the Python built-in method for iterating over mapped 
        items.
        """
        
        return self._sequence.__iter__()
    
    def __len__(self):
        r"""
        Implementation of the Pyhton built-in method for returning the number 
        of items in this mapper instance.
        """
        
        return self._sequence.__len__()
    
    ### -------------------------------------------------------------------- ###
    
    def __contains__(self, x):
        r"""
        Implementation of the Python built-in method for returning the truth 
        value if an object is also an item of this mapper instance.
        """
        
        return self._sequence.__contains__(x)
    
    ### -------------------------------------------------------------------- ###
    
    def __getitem__(self, x):
        r"""
        Implementation of the Python built-in method for returning items of 
        this mapper instance.
        """
        
        # try to handle like ordinary list
        if isinstance(x, int  ):
            return      self._sequence[x]
        if isinstance(x, slice):
            return list(self._sequence[x])
        
        # return mapper itself
        if x is Ellipsis:
            return self
        
        # try to slice items as well
        if isinstance(x, tuple):
            
            # try to handle like ordinary list
            if isinstance(x[0], int  ):
                return      self.apply(_operator.getitem,        self._sequence[x[0]] , x[1:])
            if isinstance(x[0], slice):
                return list(self.apply(_operator.getitem, Mapped(self._sequence[x[0]]), x[1:]))
            
            # return mapped items
            if x[0] is Ellipsis:
                return self.apply(_operator.getitem, self, x[1:])
        
        # raise error as last resort
        raise TypeError("'Mapped' indices must be integers, not %s" % type(x).__name__)
    
    ### -------------------------------------------------------------------- ###
    
    def __setitem__(self, x, value):
        r"""
        Implementation of the Python built-in method for setting items in this 
        mapper instance.
        """
        
        # try to handle like ordinary list
        if   isinstance(x, (int, slice)):
            self._sequence[x] = value
        
        # sampe as empty slice
        elif x is Ellipsis:
            self._sequence[:] = value
        
        # try to slice items as well
        elif isinstance(x, tuple):
            
            # try to handle like ordinary list
            if   isinstance(x[0], int  ):
                self.apply(_operator.setitem,        self._sequence[x[0]] , x[1:], value)
            elif isinstance(x[0], slice):
                self.apply(_operator.setitem, Mapped(self._sequence[x[0]]), x[1:], value)
        
            # return mapped items
            elif x[0] is Ellipsis:
                self.apply(_operator.setitem, self, x[1:], value)
        
        # raise error as last resort
        raise TypeError("'Mapped' indices must be integers, not %s" % type(x).__name__)
    
    ### -------------------------------------------------------------------- ###
    
    def __delitem__(self, x):
        r"""
        Implementation of the Pyhton built-in method for deleting items in the 
        mapper instance.
        """
        
        # try to handle like ordinary list
        if   isinstance(x, (int, slice)):
            del self._sequence[x]
        
        # sampe as empty slice
        elif x is Ellipsis:
            del self._sequence[:]
        
        # try to slice items as well
        elif isinstance(x, tuple):
            
            # try to handle like ordinary list
            if   isinstance(x[0], int  ):
                self.apply(_operator.delitem,        self._sequence[x[0]] , x[1:])
            elif isinstance(x[0], slice):
                self.apply(_operator.delitem, Mapped(self._sequence[x[0]]), x[1:])
        
            # return mapped items
            elif x[0] is Ellipsis:
                self.apply(_operator.delitem, self, x[1:])
        
        # raise error as last resort
        raise TypeError("'Mapped' indices must be integers, not %s" % type(x).__name__)
    
    ### -------------------------------------------------------------------- ###
    
    def __getattr__(self, name):
        r"""
        Implementation of the Python built-in method for getting attributes of 
        a mapper instance.
        """
        
        if name in self.__dict__:
            return self.__dict__[name]
        
        if "_sequence" in self.__dict__:
            if all(self.apply(hasattr, self, name)):
                attr = self.apply(getattr, self, name)
                if not all(self.apply(callable, attr)):
                    return attr
                return lambda *args, **kwargs: self.apply(attr, *args, **kwargs)
        
        raise AttributeError("'Mapped' instance has no attribute '%s'" % name)
    
    ### -------------------------------------------------------------------- ###
    
    def __setattr__(self, name, value):
        r"""
        Implementation of the Pyhton built-in method for setting attributes of 
        a mapper instance.
        """
        
        if "_sequence" in self.__dict__:
            if all( self.apply(hasattr, self, name)):
                self.apply(setattr, self, name, value)
        else:
            self.__dict__[name] = value
    
    ### -------------------------------------------------------------------- ###
    
    def __delattr__(self, name):
        r"""
        Implementation of the Python built-in method for deleting attributes of 
        a mapper instance.
        """
        
        if "_sequence" in self.__dict__:
            if all(self.apply(hasattr, self, name)):
                self.apply(delattr, self, name)
        else:
            try:
                del self.__dict__[name]
            except KeyError:
                raise AttributeError("'Mapped' instance has no attribute '%s'" % name)
    
    ### -------------------------------------------------------------------- ###
    
    @staticmethod
    def apply(fn, *args, **kwargs):
        r"""
        Function that mapps a function to its call arguments. Arguments that 
        are not an instance of :class:`Mapped` will be broadcasted to all items 
        of the :class:`Mapped` instance arguments.
        
        Please note that all arguments that are an instance of :class:`Mapped` 
        have to have the same number of items, or else they cannot be mapped.
        
        Parameters
        ----------
        fn : callable
            Function that is used to map all call arguments.
        *args : *tuple*
            Positional call arguments.
        **kwargs : *dict*
            Key-word call arguments.
        """
        
        # get all mapped arguments
        mapped_args = filter(lambda x: isinstance(x, Mapped), [fn] + list(args) + list(kwargs.itervalues()))
        
        # test if all mapped arguments have the same length
        if not all(len(mapped_args[0]) == len(x) for x in mapped_args[1:]):
            raise ValueError("all mapped arguments must have the same number "
                             "of items")
        
        # return single instance if necessary
        if not mapped_args:
            return fn(*args, **kwargs)
        
        # define result of computation
        result = []
        
        # do element-wise operation
        for i in xrange(len(mapped_args[0])):
            
            # get current function
            f = fn[i] if isinstance(fn, Mapped) else fn
            
            # define positional and key-word arguments for operation call
            a = []
            b = {}
            
            # create positional arguments
            for x in args:
                a.append( x[i] if isinstance(x, Mapped) else x)
            
            # create key-word arguments
            for (name, x) in kwargs.iteritems():
                b[name] = x[i] if isinstance(x, Mapped) else x
                    
            # apply operation and append to results
            result.append(f(*a, **b))
        
        # return new mapped instance with results
        return Mapped(result)
    
    ### -------------------------------------------------------------------- ###

### ======================================================================== ###

def shared_as(x, init=None, name=None, dtype=None):
    r"""
    Function that initializes a single shared variable based on a *template* 
    variable (i.e. :samp:`x`).
    
    Parameters
    ----------
    x : *shared theano variable*
        The *template* shared variable, which will be used to create a new 
        shared variable with the same properties.
    init : *int* or *float*
        The value that the new shared variable should be initialized with. If 
        None then this function will create a copy of the *template* variable.
    name : *str*
        The name of the new shared variable. If None then the name will be the 
        same as with the *template* variable. This can also be a formatting 
        string, e.g. ``"{!s}_new"``.
    dtype : *str*
        The data type of the new shared variable. If None then the data type 
        will be the same as with the *template* variable.
    
    Returns
    -------
    out : *shared theano variable*
        The newly created shared variable.
    """
    
    # assert that the first argument is a shared theano variable
    if not isinstance(x, _theano.tensor.sharedvar.SharedVariable):
        raise ValueError("argument 'x' must be a shared theano variable")
    
    # define default values for arguments if necessary
    if name  is None: name  = x.name
    if dtype is None: dtype = x.dtype
    
    # format name w.r.t. value attributes
    name = name.format(x)
    
    # initialize newly created shared variable
    if init is None:
        value = _np.copy (x.get_value()        )
    else:
        value = _np.empty(x.shape.eval(), dtype)
        value[...] = init
    
    # return newly created shared variable
    return _theano.shared(_theano._asarray(value, dtype), name, borrow=True)

### ======================================================================== ###

def flatten(A):
    r"""
    Function that flattens and concatenates all variables in a list.
    """
    return _theano.tensor.concatenate([x.flatten() for x in A])

### ======================================================================== ###

def floor_vector(v, eps=1e-4):
    r"""
    Function that returns a floored vector:
    
    .. math::
        \hat{\textbf{v}} = \max(\textbf{v}, \epsilon) \text{.}
    
    Parameters
    ----------
    v : *numpy array*
        Vector that needs to be floored.
    eps : *float*
        Minimum coefficient of the floored vector.
    
    Returns
    -------
    out : *numpy array*
        Floored vector.
    """
    
    return _np.maximum(v, eps * _np.max(v))

### ------------------------------------------------------------------------ ###

def floor_matrix(H, eps=1e-4):
    r"""
    Function that returns a floored matrix:
    
    .. math::
        \hat{\textbf{H}} = \textbf{U}\hat{\boldsymbol\Sigma}\textbf{U}' \text{, }
    
    whereas :math:`\hat{\boldsymbol\Sigma} = \max(\boldsymbol\Sigma, \epsilon)` 
    and :math:`\overline{\textbf{H}} = \textbf{U}\boldsymbol\Sigma\textbf{V}'` 
    (i.e. singular value decomposition).
    
    Parameters
    ----------
    H : *numpy array*
        Matrix that needs to be floored.
    eps : *float*
        Minimum coefficient of the floored matrix.
    
    Returns
    -------
    out : *numpy array*
        Floored matrix.
    """
    
    u, s, _ = _np.linalg.svd(H)
    return _np.dot(u * floor_vector(s), u.T)

### ======================================================================== ###