# -*- coding: utf-8 -*-
r"""
"""

### ======================================================================== ###

import numpy    as _np

import sys      as _sys
import datetime as _datetime

### ======================================================================== ###

def sgd(n_error, n_epoch, n_step, out=_sys.stdout, info={}):
    r"""
    Function that returns a row-print function. This print function will cause 
    the sgd-optimizer to output something like
    ::
        ─────────────────────────────────────────────────────────────────────────
          epoch  ¦    step   ¦      time stamp     ¦      training set error     
         ─────────────────────────────────────────────────────────────────────── 
           0/100 ¦    0/1000 ¦ 2014-09-13 17:02:16 ¦  1.875286e+00  7.306722e-01 
           0/100 ¦  250/1000 ¦ 2014-09-13 17:02:17 ¦  1.858011e+00  7.305308e-01 
           0/100 ¦  500/1000 ¦ 2014-09-13 17:02:17 ¦  1.840416e+00  7.303914e-01 
           0/100 ¦  750/1000 ¦ 2014-09-13 17:02:18 ¦  1.824284e+00  7.302456e-01 
           1/100 ¦    0/1000 ¦ 2014-09-13 17:02:18 ¦  1.808678e+00  7.301076e-01 
           1/100 ¦  250/1000 ¦ 2014-09-13 17:02:19 ¦  1.793647e+00  7.299805e-01 
           1/100 ¦  500/1000 ¦ 2014-09-13 17:02:19 ¦  1.779103e+00  7.298540e-01 
           1/100 ¦  750/1000 ¦ 2014-09-13 17:02:20 ¦  1.764851e+00  7.297190e-01 
           2/100 ¦    0/1000 ¦ 2014-09-13 17:02:20 ¦  1.750963e+00  7.295932e-01 
           2/100 ¦  250/1000 ¦ 2014-09-13 17:02:21 ¦  1.737558e+00  7.294515e-01 
           2/100 ¦  500/1000 ¦ 2014-09-13 17:02:21 ¦  1.725219e+00  7.293292e-01 
           2/100 ¦  750/1000 ¦ 2014-09-13 17:02:21 ¦  1.713029e+00  7.292042e-01 
           3/100 ¦    0/1000 ¦ 2014-09-13 17:02:22 ¦  1.701496e+00  7.290821e-01 
           3/100 ¦  250/1000 ¦ 2014-09-13 17:02:23 ¦  1.690264e+00  7.289602e-01 
           3/100 ¦  500/1000 ¦ 2014-09-13 17:02:23 ¦  1.679724e+00  7.288410e-01 
           3/100 ¦  750/1000 ¦ 2014-09-13 17:02:23 ¦  1.669346e+00  7.287201e-01 
           4/100 ¦    0/1000 ¦ 2014-09-13 17:02:24 ¦  1.659384e+00  7.286166e-01 
    onto stdout, whereas the type and width of colums may vary.
    
    Parameters
    ----------
    n_error : *int*
        Number of errors that should be displayed.
    n_epoch : *int*
        Maximum number of epoch updates that are used during the optimization.
    n_step : *int*
        Maximum number of step updates that are used during one epoch.
    out : *file*
        Output stream used for writing output.
    info : *dict*
        Original call arguments used for calling :func:`sgd`, excluding the 
        :samp:`formatter` and some other arguments.
    
    Returns
    -------
    r_print : callable
        Row-print function of :samp:`r_print(epoch, step, error_t, 
        error_v=None, info={})`.
    """
    
    # define default column formats
    c_fmt = dict(
        time    = dict(w=19, h=u"time stamp"          , b=u"{time:%Y-%0m-%0d %H:%M:%S}"),
        epoch   = dict(w= 5, h=u"epoch"               , b=u"{epoch:>5d}"               ),
        step    = dict(w= 4, h=u"step"                , b=u"{step:>4d}"                ),
        best    = dict(w= 3, h=u"min"                 , b=u"{best:^3}"                 ),
        eta     = dict(w=12, h=u"eta"                 , b=u"{eta:^12.6e}"              ),
        mu      = dict(w=12, h=u"mu"                  , b=u"{mu:^12.6e}"               ),
        error_t = dict(w=20, h=u"training set error"  , b=u"{error_t:^ 20.6e}"         ),
        error_v = dict(w=20, h=u"validation set error", b=u"{error_v:^ 20.6e}"         )
    )
    
    # re-define minimum epoch column width and format
    if n_epoch:
        epoch_w = max(len(str(n_epoch)), 3)
        epoch_b = u"{{epoch:>{w}d}}/{{n_epoch:<{w}d}}".format(w=epoch_w)
        epoch_w = epoch_w * 2 + 1
        c_fmt["epoch"].update(b=epoch_b, w=epoch_w)
    
    # re-define minimum step column width and format
    if n_step:
        step_w = max(len(str(n_step)), 3)
        step_b = u"{{step:>{w}d}}/{{n_step:<{w}d}}".format(w=step_w)
        step_w = step_w * 2 + 1
        c_fmt["step"].update(b=step_b, w=step_w)
    
    # re-define minimum error column width and format
    if n_error:
        
        # define maximum error rows
        max_rows = 5
        
        # re-define maximum error rows
        if callable(info["eta"]):
            max_rows -= 1
        if callable(info["mu" ]):
            max_rows -= 1
        
        # re-define error columns
        if max_rows <= n_error:
            error_w = 10
            error_a =  3
        elif 2 <= n_error:
            error_w = 13
            error_a =  6
        else:
            error_w = 20
            error_a =  6
        error_t = u" ".join(u"{{error_t[{}]:^ {w}.{a}e}}".format(i, w=error_w, a=error_a) for i in xrange(n_error))
        error_v = u" ".join(u"{{error_v[{}]:^ {w}.{a}e}}".format(i, w=error_w, a=error_a) for i in xrange(n_error))
        error_w = (error_w + 1) * n_error - 1
        c_fmt["error_t"].update(b=error_t, w=error_w)
        c_fmt["error_v"].update(b=error_v, w=error_w)
        
        # re-define rate columns
        if max_rows <= n_error:
            float_w =  9
            float_a =  3
        else:
            float_w = 12
            float_a =  6
        c_fmt["eta"].update(b=u"{{eta:^{w}.{a}e}}".format(w=float_w, a=float_a), w=float_w)
        c_fmt["mu" ].update(b= u"{{mu:^{w}.{a}e}}".format(w=float_w, a=float_a), w=float_w)
    
    # define all output columns
    cols = [u"epoch", u"step", u"time", u"best", u"eta", u"mu", u"error_t", u"error_v"]
    
    # remove unnecessary columns
    if n_step is None or info["interrupt"] is None or n_step <= info["interrupt"]:
        cols.remove(u"step")
    if not callable(info["eta"]):
        cols.remove(u"eta")
    if not callable(info["mu"]):
        cols.remove(u"mu")
    if info["v_set"] is None:
        cols.remove(u"best")
        cols.remove(u"error_v")
    
    # compute tty width
    tty_width = sum(c_fmt[c]["w"] for c in cols) + len(cols) * 3 - 1
    
    # define output of head, body and progress
    head = u"\r " + u" \u00a6 ".join(c_fmt[c]["h"].center(c_fmt[c]["w"]) for c in cols) + u" \n"
    body = u"\r " + u" \u00a6 ".join(c_fmt[c]["b"]                       for c in cols) + u" \n"
    
    # define row-print function
    def r_print(epoch, step, error_t, error_v=None, info={}):
        if epoch < 0:
            out.write(u"\r"  + u"\u2500" *  tty_width       + u"\n")
            out.write(head)
            out.write(u"\r " + u"\u2500" * (tty_width - 2) + u" \n")
        else:
            out.write(body.format(time    = _datetime.datetime.now(),
                                  epoch   = epoch,
                                  step    = step,
                                  error_t = error_t,
                                  error_v = error_v,
                                  n_epoch = info.get("n_epoch", n_epoch),
                                  n_step  = info.get("n_step" , n_step ),
                                  eta     = info.get("eta"    , 0.0    ),
                                  mu      = info.get("mu"     , 0.0    ),
                                  best    = "*" if info.get("best", False) else ""))
        out.flush()
    
    # return callable print functions
    return r_print

### ======================================================================== ###

def irprop(n_error, n_epoch, out=_sys.stdout, info={}):
    r"""
    Function that returns a row-print function. This print function will cause 
    the irprop-optimizer to output something like
    ::
        ─────────────────────────────────────────────────────────────
          epoch  ¦      time stamp     ¦      training set error     
         ─────────────────────────────────────────────────────────── 
           0/100 ¦ 2014-09-13 17:08:48 ¦  1.875286e+00  7.306775e-01 
           1/100 ¦ 2014-09-13 17:08:49 ¦  1.771726e+00  7.286435e-01 
           2/100 ¦ 2014-09-13 17:08:50 ¦  1.665497e+00  7.260883e-01 
           3/100 ¦ 2014-09-13 17:08:50 ¦  1.564256e+00  7.226604e-01 
           4/100 ¦ 2014-09-13 17:08:51 ¦  1.478593e+00  7.182116e-01 
           5/100 ¦ 2014-09-13 17:08:51 ¦  1.424197e+00  7.127677e-01 
           6/100 ¦ 2014-09-13 17:08:52 ¦  1.411723e+00  7.053242e-01 
           7/100 ¦ 2014-09-13 17:08:53 ¦  1.400089e+00  6.958197e-01 
           8/100 ¦ 2014-09-13 17:08:53 ¦  1.385632e+00  6.847752e-01 
           9/100 ¦ 2014-09-13 17:08:54 ¦  1.370207e+00  6.717612e-01 
          10/100 ¦ 2014-09-13 17:08:55 ¦  1.354342e+00  6.564629e-01 
          11/100 ¦ 2014-09-13 17:08:55 ¦  1.336300e+00  6.389711e-01 
    onto stdout, whereas the type and width of colums may vary.
    
    Parameters
    ----------
    n_error : *int*
        Number of errors that should be displayed.
    n_epoch : *int*
        Maximum number of epoch updates that are used during the optimization.
    out : *file*
        Output stream used for writing output.
    info : *dict*
        Original call arguments used for calling :func:`irprop`, excluding the 
        :samp:`formatter` and some other arguments.
    
    Returns
    -------
    r_print : callable
        Row-print function of :samp:`r_print(epoch, error_t, error_v=None, 
        info={})`.
    """
    
    # define default column formats
    c_fmt = dict(
        time     = dict(w=19, h=u"time stamp"          , b=u"{time:%Y-%0m-%0d %H:%M:%S}"),
        epoch    = dict(w= 5, h=u"epoch"               , b=u"{epoch:>5d}"               ),
        best     = dict(w= 3, h=u"min"                 , b=u"{best:^3}"                 ),
        error_t  = dict(w=20, h=u"training set error"  , b=u"{error_t:^ 20.6e}"         ),
        error_v  = dict(w=20, h=u"validation set error", b=u"{error_v:^ 20.6e}"         )
    )
    
    # re-define minimum epoch column width and format
    if n_epoch:
        epoch_w = max(len(str(n_epoch)), 3)
        epoch_b = u"{{epoch:>{w}d}}/{{n_epoch:<{w}d}}".format(w=epoch_w)
        epoch_w = epoch_w * 2 + 1
        c_fmt["epoch"].update(b=epoch_b, w=epoch_w)
    
    # re-define minimum error column width and format
    if n_error:
        if   5 <= n_error:
            error_w = 10
            error_a =  3
        elif 2 <= n_error:
            error_w = 13
            error_a =  6
        else:
            error_w = 20
            error_a =  6
        error_t = u" ".join(u"{{error_t[{}]:^ {w}.{a}e}}".format(i, w=error_w, a=error_a) for i in xrange(n_error))
        error_v = u" ".join(u"{{error_v[{}]:^ {w}.{a}e}}".format(i, w=error_w, a=error_a) for i in xrange(n_error))
        error_w = (error_w + 1) * n_error - 1
        c_fmt["error_t"].update(b=error_t, w=error_w)
        c_fmt["error_v"].update(b=error_v, w=error_w)
    
    # define output columns
    if info["v_set"] is None:
        cols = [u"epoch", u"time", u"error_t"]
    else:
        cols = [u"epoch", u"time", u"best", u"error_t", u"error_v"]
    
    # compute tty width
    tty_width = sum(c_fmt[c]["w"] for c in cols) + len(cols) * 3 - 1
    
    # define output of head and body
    head = u"\r " + u" \u00a6 ".join(c_fmt[c]["h"].center(c_fmt[c]["w"]) for c in cols) + u" \n"
    body = u"\r " + u" \u00a6 ".join(c_fmt[c]["b"]                       for c in cols) + u" \n"
    
    # define row-print function
    def r_print(epoch, error_t, error_v=None, info={}):
        if epoch < 0:
            out.write(u"\r"  + u"\u2500" *  tty_width       + u"\n")
            out.write(head)
            out.write(u"\r " + u"\u2500" * (tty_width - 2) + u" \n")
        else:
            out.write(body.format(time    = _datetime.datetime.now(),
                                  epoch   = epoch,
                                  error_t = error_t,
                                  error_v = error_v,
                                  n_epoch = info.get("n_epoch", n_epoch),
                                  best    = "*" if info.get("best", False) else ""))
        out.flush()
    
    # return callable print function
    return r_print

### ======================================================================== ###

def hf(n_error, n_epoch, n_step, out=_sys.stdout, info={}):
    r"""
    Function that returns a row- and a progress-print function. These print 
    functions will cause the hf-optimizer to output something like
    ::
        ──────────────────────────────────────────────────────────────────────────────────────────────────────
          epoch  ¦   step  ¦      time stamp     ¦    lambda    ¦     phi(x)    ¦      training set error     
         ──────────────────────────────────────────────────────────────────────────────────────────────────── 
           0/100 ¦   0/100 ¦ 2014-09-13 17:14:06 ¦ 1.000000e-01 ¦      inf      ¦  1.875286e+00  7.306775e-01 
           1/100 ¦  15/100 ¦ 2014-09-13 17:14:12 ¦ 1.000000e-01 ¦ -1.046153e+00 ¦  1.851088e+00  4.805207e-01 
           2/100 ¦   0/100 ¦ 2014-09-13 17:14:21 ¦ 1.500000e-01 ¦  0.000000e+00 ¦  1.851088e+00  4.805207e-01 
           3/100 ¦   0/100 ¦ 2014-09-13 17:14:24 ¦ 2.250000e-01 ¦  0.000000e+00 ¦  1.851088e+00  4.805207e-01 
           4/100 ¦   0/100 ¦ 2014-09-13 17:14:29 ¦ 3.375000e-01 ¦  0.000000e+00 ¦  1.851088e+00  4.805207e-01 
           5/100 ¦   0/100 ¦ 2014-09-13 17:14:33 ¦ 5.062500e-01 ¦  0.000000e+00 ¦  1.851088e+00  4.805207e-01 
           6/100 ¦   0/100 ¦ 2014-09-13 17:14:36 ¦ 7.593750e-01 ¦  0.000000e+00 ¦  1.851088e+00  4.805207e-01 
           7/100 ¦   0/100 ¦ 2014-09-13 17:14:40 ¦ 1.139063e+00 ¦  0.000000e+00 ¦  1.851088e+00  4.805207e-01 
           8/100 ¦   8/100 ¦ 2014-09-13 17:14:43 ¦ 1.708594e+00 ¦ -8.697326e-01 ¦  1.567557e+00  5.307326e-01 
           9/100 ¦   8/100 ¦ 2014-09-13 17:14:49 ¦ 1.708594e+00 ¦ -2.907739e-02 ¦  1.225070e+00  5.164487e-01 
          10/100 ¦  53/100 ¦ 2014-09-13 17:14:56 ¦ 1.708594e+00 ¦ -2.459071e-02 ¦  1.201091e+00  5.060568e-01 
          11/100 ¦  15/100 ¦ 2014-09-13 17:15:02 ¦ 1.139063e+00 ¦ -8.790827e-03 ¦  1.182268e+00  4.883099e-01 
          12/100 ¦  15/100 ¦ 2014-09-13 17:15:07 ¦ 7.593750e-01 ¦ -1.284442e-02 ¦  1.159223e+00  4.655194e-01 
          13/100 ¦  15/100 ¦ 2014-09-13 17:15:13 ¦ 5.062500e-01 ¦ -1.816406e-02 ¦  1.124960e+00  4.315440e-01 
          14/100 ¦  15/100 ¦ 2014-09-13 17:15:17 ¦ 3.375000e-01 ¦ -2.534608e-02 ¦  1.081060e+00  3.879037e-01 
          15/100 ¦  15/100 ¦ 2014-09-13 17:15:23 ¦ 2.250000e-01 ¦ -3.326596e-02 ¦  1.016564e+00  3.236564e-01 
          16/100 ¦   9/100 ¦ 2014-09-13 17:15:25 ¦ 1.500000e-01 ¦ -4.284574e-02 ¦                             
    onto stdout, whereas the type and width of colums may vary.
    
    Parameters
    ----------
    n_error : *int*
        Number of errors that should be displayed.
    n_epoch : *int*
        Maximum number of epoch updates that are used during the optimization.
    n_step : *int*
        Maximum number of step updates for computing the partial objective.
    out : *file*
        Output stream used for writing output.
    info : *dict*
        Original call arguments used for calling :func:`hf`, excluding the 
        :samp:`formatter` and some other arguments.
    
    Returns
    -------
    r_print : callable
        Row-print function of :samp:`r_print(epoch, step, error_t, 
        error_v=None, info={})`.
    p_print : callable
        Progress-print function of :samp:`p_print(epoch, step, error_t=None, 
        error_v=None, info={})`.
    """
    
    # define default column formats
    c_fmt = dict(
        time     = dict(w=19, h=u"time stamp"          , b=u"{time:%Y-%0m-%0d %H:%M:%S}"),
        epoch    = dict(w= 5, h=u"epoch"               , b=u"{epoch:>5d}"               ),
        step     = dict(w= 4, h=u"step"                , b=u"{step:>4d}"                ),
        best     = dict(w= 3, h=u"min"                 , b=u"{best:^3}"                 ),
        dc       = dict(w=12, h=u"lambda"              , b=u"{dc:^12.6e}"               ),
        q        = dict(w=13, h=u"phi(x)"              , b=u"{q:^ 13.6e}"               ),
        error_t  = dict(w=20, h=u"training set error"  , b=u"{error_t:^ 20.6e}"         ),
        error_v  = dict(w=20, h=u"validation set error", b=u"{error_v:^ 20.6e}"         )
    )
    
    # re-define minimum epoch column width and format
    if n_epoch:
        epoch_w = max(len(str(n_epoch)), 3)
        epoch_b = u"{{epoch:>{w}d}}/{{n_epoch:<{w}d}}".format(w=epoch_w)
        epoch_w = epoch_w * 2 + 1
        c_fmt["epoch"].update(b=epoch_b, w=epoch_w)
    
    # re-define minimum step column width and format
    if n_step:
        step_w = max(len(str(n_step)), 3)
        step_b = u"{{step:>{w}d}}/{{n_step:<{w}d}}".format(w=step_w)
        step_w = step_w * 2 + 1
        c_fmt["step"].update(b=step_b, w=step_w)
    
    # re-define minimum error column width and format
    if n_error:
        
        # re-define error column width
        if   4 <= n_error:
            error_w = 10
            error_a =  3
        elif 2 <= n_error:
            error_w = 13
            error_a =  6
        else:
            error_w = 20
            error_a =  6
        error_t = u" ".join(u"{{error_t[{}]:^ {w}.{a}e}}".format(i, w=error_w, a=error_a) for i in xrange(n_error))
        error_v = u" ".join(u"{{error_v[{}]:^ {w}.{a}e}}".format(i, w=error_w, a=error_a) for i in xrange(n_error))
        error_w = (error_w + 1) * n_error - 1
        c_fmt["error_t"].update(b=error_t, w=error_w)
        c_fmt["error_v"].update(b=error_v, w=error_w)
        
        # re-define minimum damping coefficient and lambda 
        # widht and format
        if   4 <= n_error:
            float_w =  9
            float_a =  3
        else:
            float_w = 12
            float_a =  6
        c_fmt["dc"].update(b=u"{{dc:^{w}.{a}e}}".format(w=float_w    , a=float_a), w=float_w    )
        c_fmt["q" ].update(b=u"{{q:^ {w}.{a}e}}".format(w=float_w + 1, a=float_a), w=float_w + 1)
            
    
    # define column format for progress
    pc_fmt = dict((c, dict(c_fmt[c])) for c in c_fmt)
    pc_fmt["error_t"]["b"] = "{{error_t:^{w}s}}".format(w=pc_fmt["error_t"]["w"])
    pc_fmt["error_v"]["b"] = "{{error_v:^{w}s}}".format(w=pc_fmt["error_v"]["w"])
    
    # define output columns
    if info["v_set"] is None:
        cols = [u"epoch", u"step", u"time", u"dc", u"q", u"error_t"]
    else:
        cols = [u"epoch", u"step", u"time", u"best", u"dc", u"q", u"error_t", u"error_v"]
    
    # compute tty width
    tty_width = sum(c_fmt[c]["w"] for c in cols) + len(cols) * 3 - 1
    
    # define output of head, body and progress
    head = u"\r " + u" \u00a6 ".join( c_fmt[c]["h"].center(c_fmt[c]["w"]) for c in cols) + u" \n"
    body = u"\r " + u" \u00a6 ".join( c_fmt[c]["b"]                       for c in cols) + u" \n"
    prog = u"\r " + u" \u00a6 ".join(pc_fmt[c]["b"]                       for c in cols)
    
    # define row-print function
    def r_print(epoch, step, error_t, error_v=None, info={}):
        if epoch < 0:
            out.write(u"\r"  + u"\u2500" *  tty_width       + u"\n")
            out.write(head)
            out.write(u"\r " + u"\u2500" * (tty_width - 2) + u" \n")
        else:
            out.write(body.format(time    = _datetime.datetime.now(),
                                  epoch   = epoch,
                                  step    = step,
                                  error_t = error_t,
                                  error_v = error_v,
                                  n_epoch = info.get("n_epoch", n_epoch),
                                  n_step  = info.get("n_step" , n_step ),
                                  dc      = info.get("dc"     , 0.0    ),
                                  q       = info.get("q"      , 0.0    ),
                                  best    = "*" if info.get("best", False) else ""))
        out.flush()
    
    # define progress-print function
    def p_print(epoch, step, error_t=None, error_v=None, info={}):
        if error_t is None:
            error_t = u""
        else:
            error_t = c_fmt["error_t"]["b"].format(error_t=error_t)
        if error_v is None:
            error_v = u""
        else:
            error_v = c_fmt["error_v"]["b"].format(error_v=error_v)
        out.write(prog.format(time    = _datetime.datetime.now(),
                              epoch   = epoch,
                              step    = step,
                              error_t = error_t,
                              error_v = error_v,
                              n_epoch = info.get("n_epoch", n_epoch),
                              n_step  = info.get("n_step" , n_step ),
                              dc      = info.get("dc"     , 0.0    ),
                              q       = info.get("q"      , 0.0    ),
                              best    = "*" if info.get("best", False) else ""))
        out.flush()
    
    # return callable print functions
    return r_print, p_print

### ======================================================================== ###

def ksd(n_error, n_epoch, n_step, out=_sys.stdout, info={}):
    r"""
    Function that returns a row- and a progress-print function. These print 
    functions will cause the ksd-optimizer to output something like
    ::
        ────────────────────────────────────────────────────────────────────────────
          epoch  ¦      time stamp     ¦     ‖∇f‖     ¦      training set error     
         ────────────────────────────────────────────────────────────────────────── 
           0/100 ¦ 2014-09-14 02:53:28 ¦     inf      ¦  1.875286e+00  7.306775e-01 
           1/100 ¦ 2014-09-14 02:53:34 ¦ 1.506670e-08 ¦  1.076530e+00  3.655692e-01 
           2/100 ¦ 2014-09-14 02:53:41 ¦ 9.410248e-08 ¦  8.542557e-01  1.558756e-01 
           3/100 ¦ 2014-09-14 02:53:47 ¦ 1.256310e-06 ¦  7.593527e-01  6.466950e-02 
           4/100 ¦ 2014-09-14 02:53:55 ¦ 2.565331e-07 ¦  7.065590e-01  1.335151e-02 
           5/100 ¦ 2014-09-14 02:54:05 ¦ 1.552956e-08 ¦  6.988602e-01  5.723802e-03 
         ==> running bfgs on subspace ......... [###############          ]  64.15% 
    onto stdout, whereas the type and width of colums may vary.
    
    Parameters
    ----------
    n_error : *int*
        Number of errors that should be displayed.
    n_epoch : *int*
        Maximum number of epoch updates that are used during the optimization.
    n_step : *int*
        Maximum number of step updates for computing the partial objective. In 
        contrast to the :samp:`n_step` argument from :func:`ksd` this argument 
        covers all the necessary steps for computing the partial objective and 
        not just the maximum steps for the BFGS optimizer.
    out : *file*
        Output stream used for writing output.
    info : *dict*
        Original call arguments used for calling :func:`ksd`, excluding the 
        :samp:`formatter` and some other arguments.
    
    Returns
    -------
    r_print : callable
        Row-print function of :samp:`r_print(epoch, accuracy, error_t, 
        error_v=None, info={})`.
    p_print : callable
        Progress-print function of :samp:`p_print(step, message=u"", info={})`.
    """
    
    # define default column formats
    c_fmt = dict(
        time     = dict(w=19, h=u"time stamp"          , b=u"{time:%Y-%0m-%0d %H:%M:%S}"),
        epoch    = dict(w= 5, h=u"epoch"               , b=u"{epoch:>5d}"               ),
        best     = dict(w= 3, h=u"min"                 , b=u"{best:^3}"                 ),
        accuracy = dict(w=12, h=u"\u2016\u2207f\u2016" , b=u"{accuracy:^12.6e}"         ),
        error_t  = dict(w=20, h=u"training set error"  , b=u"{error_t:^ 20.6e}"         ),
        error_v  = dict(w=20, h=u"validation set error", b=u"{error_v:^ 20.6e}"         )
    )
    
    # re-define minimum epoch column width and format
    if n_epoch:
        epoch_w = max(len(str(n_epoch)), 3)
        epoch_b = u"{{epoch:>{w}d}}/{{n_epoch:<{w}d}}".format(w=epoch_w)
        epoch_w = epoch_w * 2 + 1
        c_fmt["epoch"].update(b=epoch_b, w=epoch_w)
    
    # re-define minimum error column width and format
    if n_error:
        if   5 <= n_error:
            error_w = 10
            error_a =  3
        elif 2 <= n_error:
            error_w = 13
            error_a =  6
        else:
            error_w = 20
            error_a =  6
        error_t = u" ".join(u"{{error_t[{}]:^ {w}.{a}e}}".format(i, w=error_w, a=error_a) for i in xrange(n_error))
        error_v = u" ".join(u"{{error_v[{}]:^ {w}.{a}e}}".format(i, w=error_w, a=error_a) for i in xrange(n_error))
        error_w = (error_w + 1) * n_error - 1
        c_fmt["error_t"].update(b=error_t, w=error_w)
        c_fmt["error_v"].update(b=error_v, w=error_w)
    
    # define output columns
    if info["v_set"] is None:
        cols = [u"epoch", u"time", u"accuracy", u"error_t"]
    else:
        cols = [u"epoch", u"time", u"best", u"accuracy", u"error_t", u"error_v"]
    
    # compute tty width
    tty_width = sum(c_fmt[c]["w"] for c in cols) + len(cols) * 3 - 1
    
    # define output of head, body and progress
    head = u"\r " + u" \u00a6 ".join(c_fmt[c]["h"].center(c_fmt[c]["w"]) for c in cols) + u" \n"
    body = u"\r " + u" \u00a6 ".join(c_fmt[c]["b"]                       for c in cols) + u" \n"
    prog = u"\r ==> {{m:.<34.31}} [{{b:<{w}.{w}}}] {{p:7.2%}} ".format(w=tty_width-51)
    
    # define row-print function
    def r_print(epoch, accuracy, error_t, error_v=None, info={}):
        if epoch < 0:
            out.write(u"\r"  + u"\u2500" *  tty_width       + u"\n")
            out.write(head)
            out.write(u"\r " + u"\u2500" * (tty_width - 2) + u" \n")
        else:
            out.write(body.format(time     = _datetime.datetime.now(),
                                  epoch    = epoch,
                                  accuracy = accuracy,
                                  error_t  = error_t,
                                  error_v  = error_v,
                                  n_epoch  = info.get("n_epoch", n_epoch),
                                  best     = "*" if info.get("best", False) else ""))
        out.flush()
    
    # compute conversion factors for progress bar
    n = 1.0 / n_step
    m = (tty_width - 51) * n
    
    # define progress-print function
    def p_print(step, message=u"", info={}):
        out.write(prog.format(m=message+u" ", p=step*n, b=(u"#" * _np.ceil(step*m))))
        out.flush()
    
    # return callable print functions
    return r_print, p_print

### ======================================================================== ###