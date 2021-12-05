# -*- coding: utf-8 -*-
r"""
"""

### ======================================================================== ###

import numpy as _np

### ======================================================================== ###

class Abstract(object):
    r"""
    Implementation of an abstract class that is used for data sampling. 
    Instances of this class are especially usefull for batched optimization 
    methods, so that these don't have to implement the sampling routines.
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, samples, size, step):
        r"""
        Parameters
        ----------
        samples : [*numpy array*, ...]
            List of actual parameter values in row major form, i.e. rows equal 
            to samples and columns to features. These data set parameters will 
            be used to draw samples from to form batches that are used for 
            computing update steps when training a model to fulfill its 
            optimization objective.
        size : *int*
            Number of samples that a batch should have.
        step : *int*
            Offset of a batch w.r.t. to other batches, i.e. if 
            :samp:`size <= step` then samples of one batch cannot appear in an 
            other batch.
        
        Note
        ----
        This is an abstract class and it should only be used as a base class that 
        can be inherited by actual data sampling instances.
        """
        
        # do some argument test
        if not samples:
            raise ValueError("'samples' cannot be empty")
        if not all(len(samples[0]) == len(x) for x in samples[1:]):
            raise ValueError("number of samples in 'samples' cannot differ")
        if not 0 < size:
            raise ValueError("argument 'size' must be bigger than 0")
        if not 0 < step:
            raise ValueError("argument 'step' must be bigger than 0")
        
        # compute number of samples per batch and number of batches
        self._s_num = size
        self._b_num = (len(samples[0]) - size) // step + 1
        
        # define attributes for samples and batches
        self._samples =  tuple(samples)
        self._batches = [tuple(x[i:i+size] for x in samples) for i in xrange(0, self._b_num * step, step)]
    
    ### -------------------------------------------------------------------- ###
    
    def __iter__(self):
        r"""
        Implementation of the Python built-in method for iterating over 
        batches.
        """
        
        return iter(self._batches)
    
    def __len__(self):
        r"""
        Implementation of the Python build-in method for returning the number 
        of batches.
        """
        
        return  len(self._batches)
    
    ### -------------------------------------------------------------------- ###
    
    def __getitem__(self, i):
        r"""
        Implementation of the Python built-in method for returning a batch with 
        respect to its index.
        """
        
        return self._batches[i]
    
    def __setitem__(self, i, value):
        r"""
        Implementation of the Python built-in method for setting the values of 
        a batch.
        
        Note
        ----
        This method always raises a :class:`TypeError` since all batches are 
        defined as read-only.
        """
        
        raise TypeError("cannot set items in sampler instances")
    
    def __delitem__(self, i):
        r"""
        Implementation of the Python built-in method for deleting the values of 
        a batch.
        
        Note
        ----
        This method always raises a :class:`TypeError` since all batches are 
        defined as read-only.
        """
        
        raise TypeError("cannot delete items in sampler instances")
    
    ### -------------------------------------------------------------------- ###
    
    def update(self):
        """
        Method that updates the internal state of a data sampler, i.e. it 
        shuffles data samples.
        """
        
        pass
    
    ### -------------------------------------------------------------------- ###
    
    shape = property(doc="shape of the sampler instance that is a tuple and "
                         "looks like :samp:`(<number of batches>, <number of "
                         "parameters>, <size of batches>)`")
    
    @shape.getter
    def shape(self):
        return (self._b_num, len(self._samples), self._s_num)
    
    @shape.setter
    def shape(self, value):
        raise AttributeError("assignment to read-only attribute 'shape'")
    
    @shape.deleter
    def shape( self ):
        raise AttributeError("deletion of read-only attribute 'shape'")
    
    ### -------------------------------------------------------------------- ###
    
    samples = property(doc="original samples of the sampler instance")
    
    @samples.getter
    def samples(self):
        return self._samples
    
    @samples.setter
    def samples(self, value):
        raise AttributeError("assignment to read-only attribute 'samples'")
    
    @samples.deleter
    def samples(self):
        raise AttributeError("deletion of read-only attribute 'samples'")
    
    ### -------------------------------------------------------------------- ###

### ======================================================================== ###

class Interrupt(Abstract):
    r"""
    Implementation of an interrupt data sampler that wrapps other data sampler 
    instances. This sampler instance will take two additional callback 
    functions as an argument, one that is being called on every batch that is 
    drawn from the original sampler and another one that is being called every 
    time before the original sampler updates its state. Both must implement 
    some call arguments that will be used for passing over the samples. In case 
    of the sample callback function, these samples will be the one from the 
    current batch and in case of the update callback, these samples will be the 
    original data set samples, stored within the wrapped sampler instance. 
    Further, the sample callback function must return the samples of the 
    current batch, these can be modified or as they are.
    
    The following short example shows how to initialize and use an instance of 
    :class:`Interrupt` with an instance of :class:`Constant` as its actual 
    data sampler:
    
        >>> from anno.opt.sample import Interrupt, Constant
        >>> import numpy
        >>> x = numpy.reshape(numpy.arange(6*2, dtype="float32"), (6, 2))
        >>> t = numpy.reshape(numpy.arange(6*1, dtype="float32"), (6, 1))
        >>> def scb(x, t):
        ...     return (numpy.dot(x.T),)
        >>> def ucb(x, t):
        ...     x[...] += 1
        >>> smp = Interrupt(Constant([x, t], 3), scb, ucb)
    
    We can as well print out the lenght and the shape of the current sampler 
    instance:
    
        >>> len(smp)
        2
        >>> smp.shape
        (2, 1, 2)
    
    One can as well access single batch instances of this sampler or entire 
    slices, but in the case of the current sampler instance these batches will 
    basically contain the output of the callback function that was given as an 
    argument to the sampler instance:
    
        >>> smp[0]
        (array([[ 10.],
                [ 13.]], dtype=float32),)
        >>> smp[1:]
        [(array([[ 100.],
                 [ 112.]], dtype=float32),)]
    
    But you can still access the original sampler of this sampler instance and 
    therefor the original parameter values of a batch that were used to call 
    the callback function:
    
        >>> smp.sampler[0]
        (array([[ 0.,  1.],
                [ 2.,  3.],
                [ 4.,  5.]], dtype=float32), array([[ 0.],
                [ 1.],
                [ 2.]], dtype=float32))
        >>> smp.sampler[1:]
        [(array([[  6.,   7.],
                 [  8.,   9.],
                 [ 10.,  11.]], dtype=float32), array([[ 3.],
                 [ 4.],
                 [ 5.]], dtype=float32))]
    
    By updating this sampler, it will add 1 to the samples of x:
    
        >>> smp.update()
        >>> smp[0]
        (array([[ 13.],
                [ 16.]], dtype=float32),)
        >>> smp.sampler[0]
        (array([[ 1.,  2.],
                [ 3.,  4.],
                [ 5.,  6.]], dtype=float32), array([[ 0.],
                [ 1.],
                [ 2.]], dtype=float32))
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, sampler, on_sample=lambda *x: x, on_update=lambda *x: None):
        r"""
        Parameters
        ----------
        sampler : instance of :class:`Abstract`
            Sampler instance that is being wrapped by this sampler 
            implementation.
        on_sample : callable of on_sample(*numpy array*, ...) -> (*numpy array*, ...)
            Callback function that is being called every time a batch is being 
            drawn from the original sampler. This function must further provide 
            some call arguments for the data set samples that are being stored 
            within the batch and it must return the final batch samples too.
        on_update : callable of on_update(*numpy array*, ...)
            Callback function that is being called every time before an update. 
            This function must some call arguments for the data set parameters 
            that are going to be updated, but must not return any value.
        """
        
        # set attributes of decorator instance
        self._sampler   = sampler
        self._on_sample = on_sample
        self._on_update = on_update
    
    ### -------------------------------------------------------------------- ###
    
    def __iter__(self):
        r"""
        Implementation of the Python built-in method for iterating over 
        batches.
        """
        
        for x in self._sampler:
            yield self._on_sample(*x)
    
    def __len__(self):
        r"""
        Implementation of the Python build-in method for returning the number 
        of batches.
        """
        
        return len(self._sampler)
    
    ### -------------------------------------------------------------------- ###
    
    def __getitem__(self, i):
        r"""
        Implementation of the Python built-in method for returning a batch with 
        respect to its index.
        """
        
        if isinstance(i, int):
            return  self._on_sample(*self._sampler[i])
        else:
            return [self._on_sample(*args) for args in self._sampler[i]]
    
    ### -------------------------------------------------------------------- ###
    
    def update(self, *args, **kwargs):
        """
        Method that updates the internal state of a data sampler, i.e. it 
        shuffles data samples.
        """
        
        self._on_update(*self._sampler.samples)
        return self._sampler.update(*args, **kwargs)
    
    ### -------------------------------------------------------------------- ###
    
    shape = property(doc="shape of the sampler instance that is a tuple and "
                         "looks like :samp:`(<number of batches>, <number of "
                         "parameters>, <min. size over all batches>)`")
    
    @shape.getter
    def shape(self):
        return (len(self), len(self[0]), min(len(a) for a in self[0]))
    
    @shape.setter
    def shape(self, value):
        self._sampler.shape = value
    
    @shape.deleter
    def shape(self):
        del self._sampler.shape
    
    ### -------------------------------------------------------------------- ###
    
    samples = property(doc="original samples of the sampler instance")
    
    @samples.getter
    def samples(self):
        return self._sampler.samples
    
    @samples.setter
    def samples(self, value):
        self._sampler.samples = value
    
    @samples.deleter
    def samples(self):
        del self._sampler.samples
    
    ### -------------------------------------------------------------------- ###
    
    sampler = property(doc="original sampler that has been decorated by this "
                           "sampler instance")
    
    @sampler.getter
    def sampler(self):
        return self._sampler
    
    @sampler.setter
    def sampler(self, value):
        raise AttributeError("assignment to read-only attribute 'sampler'")
    
    @sampler.deleter
    def sampler(self):
        raise AttributeError("deletion of read-only attribute 'sampler'")
    
    ### -------------------------------------------------------------------- ###

### ======================================================================== ###

class Constant(Abstract):
    r"""
    Implementation of a constand data sampler. Instances of this class will not 
    update its internal state, even when calling :func:`update`.
    
    The following example shows how to initialize and use an instance of 
    :class:`Constant`:
    
        >>> from anno.opt.sample import Constant
        >>> import numpy
        >>> x = numpy.reshape(numpy.arange(6*2, dtype="float32"), (6, 2))
        >>> t = numpy.reshape(numpy.arange(6*1, dtype="float32"), (6, 1))
        >>> smp = Constant([x, t], 3)
    
    We can as well print out the lenght and the shape of the current sampler 
    instance:
    
        >>> len(smp)
        2
        >>> smp.shape
        (2, 2, 3)
    
    One can as well access single batch instances of this sampler or entire 
    slices:
    
        >>> smp[0]
        (array([[ 0.,  1.],
                [ 2.,  3.],
                [ 4.,  5.]], dtype=float32), array([[ 0.],
                [ 1.],
                [ 2.]], dtype=float32))
        >>> smp[1:]
        [(array([[  6.,   7.],
                 [  8.,   9.],
                 [ 10.,  11.]], dtype=float32), array([[ 3.],
                 [ 4.],
                 [ 5.]], dtype=float32))]
    
    We can as well print out all batches. This is being done with respect to 
    the actual parameter values for the sake of a better visualization:
     
        >>> xsmp, tsmp = zip(*smp)
        >>> numpy.array(xsmp)
        array([[[  0.,   1.],
                [  2.,   3.],
                [  4.,   5.]],
               [[  6.,   7.],
                [  8.,   9.],
                [ 10.,  11.]]], dtype=float32)
        >>> numpy.array(tsmp)
        array([[[ 0.],
                [ 1.],
                [ 2.]],
               [[ 3.],
                [ 4.],
                [ 5.]]], dtype=float32)
    
    After updating this sampler instance, we will have the following output, 
    which is basically the same compared to the previous output:
    
        >>> smp.update()
        >>> xsmp, tsmp = zip(*smp)
        >>> numpy.array(xsmp)
        array([[[  0.,   1.],
                [  2.,   3.],
                [  4.,   5.]],
        
               [[  6.,   7.],
                [  8.,   9.],
                [ 10.,  11.]]], dtype=float32)
        >>> numpy.array(tsmp)
        array([[[ 0.],
                [ 1.],
                [ 2.]],
               [[ 3.],
                [ 4.],
                [ 5.]]], dtype=float32)
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, samples, size):
        r"""
        Parameters
        ----------
        samples : [*numpy array*, ...]
            List of actual parameter values in row major form, i.e. rows equal 
            to samples and columns to features. These data set parameters will 
            be used to draw samples from to form batches that are used for 
            computing update steps when training a model to fulfill its 
            optimization objective.
        size : *int*
            Number of samples that a batch should have. When using this sampler 
            instance, samples of one batch cannot reappear in another batch.
        """
        
        # initialize sampler instance
        super(Constant, self).__init__(samples, size, size)

    ### -------------------------------------------------------------------- ###

### ======================================================================== ###

class Random(Abstract):
    r"""
    Implementation of a random data sampler. Instances of this class will 
    shuffle their samples uniformly across all batches, when calling 
    :func:`update`.
    
    The following example shows how to initialize and use an instance of 
    :class:`Random`:
    
        >>> from anno.opt.sample import Random
        >>> import numpy
        >>> x = numpy.reshape(numpy.arange(6*2, dtype="float32"), (6, 2))
        >>> t = numpy.reshape(numpy.arange(6*1, dtype="float32"), (6, 1))
        >>> smp = Random([x, t], 3)
    
    We can as well print out the lenght and the shape of the current sampler 
    instance:
    
        >>> len(smp)
        2
        >>> smp.shape
        (2, 2, 3)
    
    One can as well access single batch instances of this sampler or entire 
    slices:
    
        >>> smp[0]
        (array([[ 0.,  1.],
                [ 2.,  3.],
                [ 4.,  5.]], dtype=float32), array([[ 0.],
                [ 1.],
                [ 2.]], dtype=float32))
        >>> smp[1:]
        [(array([[  6.,   7.],
                 [  8.,   9.],
                 [ 10.,  11.]], dtype=float32), array([[ 3.],
                 [ 4.],
                 [ 5.]], dtype=float32))]
    
    We can as well print out all batches. This is being done with respect to 
    the actual parameter values for the sake of a better visualization:
    
        >>> xsmp, tsmp = zip(*smp)
        >>> numpy.array(xsmp)
        array([[[  0.,   1.],
                [  2.,   3.],
                [  4.,   5.]],
               [[  6.,   7.],
                [  8.,   9.],
                [ 10.,  11.]]], dtype=float32)
        >>> numpy.array(tsmp)
        array([[[ 0.],
                [ 1.],
                [ 2.]],
               [[ 3.],
                [ 4.],
                [ 5.]]], dtype=float32)
    
    After updating this sampler instance, we will have the following output:
    
        >>> smp.update()
        >>> xsmp, tsmp = zip(*smp)
        >>> numpy.array(xsmp)
        array([[[  0.,   1.],
                [  6.,   7.],
                [  2.,   3.]],
               [[  8.,   9.],
                [  4.,   5.],
                [ 10.,  11.]]], dtype=float32)
        >>> numpy.array(tsmp)
        array([[[ 0.],
                [ 3.],
                [ 1.]],
               [[ 4.],
                [ 2.],
                [ 5.]]], dtype=float32)
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, samples, size, seed=None):
        r"""
        Parameters
        ----------
        samples : [*numpy array*, ...]
            List of actual parameter values in row major form, i.e. rows equal 
            to samples and columns to features. These data set parameters will 
            be used to draw samples from to form batches that are used for 
            computing update steps when training a model to fulfill its 
            optimization objective.
        size : *int*
            Number of samples that a batch should have. When using this sampler 
            instance, samples of one batch cannot reappear in another batch.
        seed : *int*
            Initial seed value to the random function that is used for 
            shuffling data samples.
        """
        
        # initialize sampler instance
        super(Random, self).__init__(samples, size, size)
        
        # define random seed if necessary
        if seed is None:
            seed = _np.random.randint(0, 2**32-1)
        
        # define random states for sampling
        self._random_state = [_np.random.RandomState(seed) for i in xrange(len(samples))]
    
    ### -------------------------------------------------------------------- ###
    
    def update(self):
        r"""
        Method that updates the internal state of the data sampler, i.e. it 
        will shuffle its samples uniformly across all batches. This is carried 
        out in-place (see :func:`numpy.random.shuffle`) so that any parameter 
        variable instance of a batch is just a memory mapped pointer to the 
        coalescent data set that stores the actual sample instances.
        """
        
        for i in xrange(len(self._samples)):
            self._random_state[i].shuffle(self._samples[i])
    
    ### -------------------------------------------------------------------- ###

### ======================================================================== ###

class Sequence(Abstract):
    r"""
    Implementation of a sequence data sampler. Instances of this class are used 
    to sample samples from time series, so that they will shuffle entire 
    batches but not the samples within these batches, when calling 
    :func:`update`.
    
    The following example shows how to initialize and use an instance of 
    :class:`Sequence`:
    
        >>> from anno.opt.sample import Sequence
        >>> import numpy
        >>> x = numpy.reshape(numpy.arange(6*2, dtype="float32"), (6, 2))
        >>> t = numpy.reshape(numpy.arange(6*1, dtype="float32"), (6, 1))
        >>> smp = Sequence([x, t], 3, 1)
    
    We can as well print out the lenght and the shape of the current sampler 
    instance:
    
        >>> len(smp)
        4
        >>> smp.shape
        (4, 2, 3)
    
    One can as well access single batch instances of this sampler or entire 
    slices:
    
        >>> smp[0]
        (array([[ 0.,  1.],
                [ 2.,  3.],
                [ 4.,  5.]], dtype=float32), array([[ 0.],
                [ 1.],
                [ 2.]], dtype=float32))
        >>> smp[1:3]
        [(array([[ 2.,  3.],
                 [ 4.,  5.],
                 [ 6.,  7.]], dtype=float32), array([[ 1.],
                 [ 2.],
                 [ 3.]], dtype=float32)), (array([[ 4.,  5.],
                 [ 6.,  7.],
                 [ 8.,  9.]], dtype=float32), array([[ 2.],
                 [ 3.],
                 [ 4.]], dtype=float32))]
    
    We can as well print out all batches. This is being done with respect to 
    the actual parameter values for the sake of a better visualization:
    
        >>> xsmp, tsmp = zip(*smp)
        >>> numpy.array(xsmp)
        array([[[  0.,   1.],
                [  2.,   3.],
                [  4.,   5.]],
               [[  2.,   3.],
                [  4.,   5.],
                [  6.,   7.]],
               [[  4.,   5.],
                [  6.,   7.],
                [  8.,   9.]],
               [[  6.,   7.],
                [  8.,   9.],
                [ 10.,  11.]]], dtype=float32)
        >>> numpy.array(tsmp)
        array([[[ 0.],
                [ 1.],
                [ 2.]],
               [[ 1.],
                [ 2.],
                [ 3.]],
               [[ 2.],
                [ 3.],
                [ 4.]],
               [[ 3.],
                [ 4.],
                [ 5.]]], dtype=float32)
    
    After updating this sampler instance, we will have the following output:
    
        >>> smp.update()
        >>> xsmp, tsmp = zip(*smp)
        >>> numpy.array(xsmp)
        array([[[  2.,   3.],
                [  4.,   5.],
                [  6.,   7.]],
               [[  6.,   7.],
                [  8.,   9.],
                [ 10.,  11.]],
               [[  0.,   1.],
                [  2.,   3.],
                [  4.,   5.]],
               [[  4.,   5.],
                [  6.,   7.],
                [  8.,   9.]]], dtype=float32)
        >>> numpy.array(tsmp)
        array([[[ 1.],
                [ 2.],
                [ 3.]],
               [[ 3.],
                [ 4.],
                [ 5.]],
               [[ 0.],
                [ 1.],
                [ 2.]],
               [[ 2.],
                [ 3.],
                [ 4.]]], dtype=float32)
    """
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, samples, size, step=1, seed=None):
        r"""
        Parameters
        ----------
        samples : [*numpy array*, ...]
            List of actual parameter values in row major form, i.e. rows equal 
            to samples and columns to features. These data set parameters will 
            be used to draw samples from to form batches that are used for 
            computing update steps when training a model to fulfill its 
            optimization objective.
        size : *int*
            Number of samples that a batch should have. In this case the number 
            of samples represents the time stamps that one draw from the time 
            series should have.
        step : *int*
            Offset of a batch w.r.t. to other batches, i.e. if 
            :samp:`size <= step` then samples of one batch cannot appear in an 
            other batch. This variable basically dictates the overlap of drawn 
            samples from the time series.
        seed : *int*
            Initial seed value to the random function that is used for 
            shuffling batches.
        """
        
        # initialize sampler instance
        super(Sequence, self).__init__(samples, size, step)
        
        # define random state for sampling
        self._random_state = _np.random.RandomState(seed)
    
    ### -------------------------------------------------------------------- ###
    
    def update(self):
        """
        Method that updates the internal state of the data sampler, i.e. it 
        will shuffle its draws from the time series (i.e. batches) but not the 
        samples within the draw. Note that the batches are again just memory 
        mapped instances that point to the original data set, but instead of 
        shuffling the samples withing the data set, this method only shuffles 
        the memory mapped instances (i.e. batches), which doesn not affect the 
        order of samples within the original data set.
        """
        
        self._random_state.shuffle(self._batches)
    
    ### -------------------------------------------------------------------- ###

### ======================================================================== ###