import anno
import numpy
import theano
import sklearn.datasets as ds

USE_DROPOUT=False

def dropout(fn, q=[0.2, 0.5], scale=True, rs=None):

    # define random state if undefined
    if rs is None or isinstance(rs, int):
        rs = theano.tensor.shared_randomstreams.RandomStreams(rs)

    # assert type of neural network
    if not isinstance(fn, anno.nnet.Abstract):
        raise TypeError("fn must be an instance of anno.nnet.Abstract")

    # assert type and value of dropout probability
    if not isinstance(q, (list, tuple)):
        raise TypeError("drop must be an instance of list")

    # create copy of dropout ratio
    q = list(q)

    # define function that decorates the activation function with
    # drop-out function
    def drop(f):

        # define drop-out for primitive perceptrons
        if isinstance(f, (anno.nnet.FF, anno.nnet.R, anno.nnet.BD)):

            # define current dropout rate
            if 1 < len(q):
                r = q.pop(0)
            else:
                r = q[0]

            # define weight scale
            if scale:
                s = 1.0 / (1.0 - r)
            else:
                s = 1.0

            # define drop-out for primitive perceptrons
            if isinstance(f, anno.nnet.FF):
                return anno.nnet.Fxy(anno.nnet.FF(f.W * s, f.b,
                    f.activation, f.parameters),
                    lambda x: x * rs.binomial(x.shape, 1, 1.0 - r, dtype=x.dtype))
            if isinstance(f, anno.nnet.R ):
                return anno.nnet.Fxy(anno.nnet.R (f.W * s, f.U, f.b, f.h,
                    f.activation, f.parameters),
                    lambda x: x * rs.binomial(x.shape, 1, 1.0 - r, dtype=x.dtype))
            if isinstance(f, anno.nnet.BD):
                return anno.nnet.Fxy(anno.nnet.BD(f.W * s, f.U, f.V, f.b, f.u, f.v,
                    f.activation, f.parameters),
                    lambda x: x * rs.binomial(x.shape, 1, 1.0 - r, dtype=x.dtype))

        # define drop-out for auto-encoder
        if isinstance(f, anno.nnet.AE):
            return anno.nnet.AE(drop(f.encode), drop(f.decode))

        # define drop-out for multi-layer perceptron
        if isinstance(f, anno.nnet.ML):
            return anno.nnet.ML(map(drop, f))

        # define drop-out for decorator perceptrons
        if isinstance(f, anno.nnet.Fxy):
            return anno.nnet.Fxy(drop(f.fn), f.fx, f.fy)
        if isinstance(f, anno.nnet.Fork):
            return anno.nnet.Fork([drop(p) for p in f])
        if isinstance(fn, anno.nnet.Join):
            return anno.nnet.Join([drop(p) for p in f], f.combinator)

        # return unknown perceptron as is
        return f

    # return model with drop-out
    return fn if not q else drop(fn)

# define random-stream (necessary for deterministic sampling)
rs    = theano.tensor.shared_randomstreams.RandomStreams(0)

# define ANN model
f     = anno.nnet.ml((32, 128, 64, 4), ("tanh", "tanh", "linear"), rs=numpy.random.RandomState(0))

# define auxiliary model for optimisation
if   USE_DROPOUT:
    f_opt = dropout(f, [0.05, 0.20], rs=rs)
else:
    f_opt = f

# define inputs and targets
x     = theano.tensor.matrix("x", dtype="float32")
t     = theano.tensor.vector("t", dtype= "uint8" )

# define predictions
y     = theano.tensor.argmax(f_opt(x), axis=1).astype("uint8")
y_opt = theano.tensor.nnet.softmax(f_opt(x))

# define objective cost and accuracy (disclaimer: since this is a multi-class problem
# we will consider TP/#samples to be the accuracy)
c     = -theano.tensor.mean(theano.tensor.log(y_opt)[theano.tensor.arange(y_opt.shape[0]), t])
p     =  theano.tensor.sum([abs(p).sum() for p in f_opt.parameters]) *  1e-3
acc   =  theano.tensor.switch(theano.tensor.eq(t, y), 1, 0).astype("float32").mean()

# define optimisation objective
objective = anno.opt.Objective([c + p, c, p, acc], [y_opt], [x, t], f_opt.parameters)

# generate synthetic training scenario
x_smp, t_smp = ds.make_classification(
        n_samples     = 8000,
        n_features    = 32,
        n_informative = 5,
        n_redundant   = 2,
        n_repeated    = 1,
        n_classes     = 4,
        random_state  = 0
)

# cast features and targets to required types
x_smp = x_smp.astype("float32")
t_smp = t_smp.astype( "uint8" )

# define index lookup for samples
i_smp = numpy.arange(5000, dtype=int)

# split dataset into training, validation and test dataset
x_smp_train = x_smp[    :4000]
t_smp_train = t_smp[    :4000]
i_smp_train = i_smp[    :4000]
x_smp_valid = x_smp[4000:5000]
t_smp_valid = t_smp[4000:5000]
i_smp_valid = i_smp[4000:5000]
x_smp_test  = x_smp[5000:    ]
t_smp_test  = t_smp[5000:    ]

# define reset function for seed update
if USE_DROPOUT:
    def reset(x, t, i):

        # update seed of random streams state
        # -> this is necessary due to the way
        #    how HF and KSD work, but not so
        #    much for SGD or iRprop
        # => better solutions for this proble
        #    do exist, but I kind of feel lazy ...
        rs.seed(hash(str(i.data)) % (2**32-1))

        # return input samples as tuple
        return x, t
else:
    def reset(x, t, i):
        return x, t

# define sampler for training data set
a_set = anno.opt.sample.Random  ([x_smp_train,
                                  t_smp_train,
                                  i_smp_train], 1000, 0)

# define auxiliary sampler for curvature and sub-objective datasets
# -> these sets are defined as constant, but in fact
#    their samples are being sampled by the previous
#    sampler instance (=> in-place shuffling)
b_set = anno.opt.sample.Constant([x_smp_train[:1000],
                                  t_smp_train[:1000],
                                  i_smp_train[:1000]], 500)
c_set = anno.opt.sample.Constant([x_smp_train[1000:],
                                  t_smp_train[1000:],
                                  i_smp_train[1000:]], 500)

# define validation dataset sampler
v_set = anno.opt.sample.Constant([x_smp_valid,
                                  t_smp_valid,
                                  i_smp_valid], 500)

# decorate sampler instances with interrupt decorator
# -> necessary for resetting random-state due to
#    determinism of dropout network
a_set = anno.opt.sample.Interrupt(a_set, on_sample=reset)
b_set = anno.opt.sample.Interrupt(b_set, on_sample=reset)
c_set = anno.opt.sample.Interrupt(c_set, on_sample=reset)
v_set = anno.opt.sample.Interrupt(v_set, on_sample=reset)

try:
    #parameters, info = anno.opt.ksd(objective, a_set, b_set, c_set, v_set,
    #        n_epoch=50, n_step=25, n_dim=20, pc_type=None, epsilon=1e-6,
    #        return_info=True)
    parameters, info = anno.opt.irprop(objective, a_set, v_set,
            n_epoch=250, epsilon=1e-6, return_info=True)
except:
    print
    raise
finally:
    y     = theano.tensor.argmax(f(x), axis=1).astype("uint8")
    y_opt = theano.tensor.nnet.softmax(f(x))
    c     = -theano.tensor.mean(theano.tensor.log(y_opt)[theano.tensor.arange(y.shape[0]), t])
    acc   = theano.tensor.switch(theano.tensor.eq(t, y), 1, 0).astype("float32").mean()
    error = anno.opt.Objective([c, acc], [y_opt], [x, t], f.parameters).evaluate([x_smp_test, t_smp_test])

    print " softmax: %12.6e" % error[0]
    print "accuracy: %12.6e" % error[1]

    print   t_smp_test[:20]
    print f(x_smp_test[:20]).argmax(axis=1)
