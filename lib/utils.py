# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 19:00:27 2014

@author: darko
"""

### ======================================================================== ###

import __builtin__

import marshal as _marshal
import types   as _types
import imp     as _imp
import os      as _os

### ======================================================================== ###

class Sandboxed(object):
    
    ### -------------------------------------------------------------------- ###
    
    def __init__(self, function, name=None, **sandbox):
        
        # assert type of function
        if not isinstance(function, _types.FunctionType):
            raise TypeError("function must be a function type, but got "
                            "%r" % type(function).__name__)
        
        # set default python environment
        sandbox.setdefault("__builtin__", __builtin__)
        
        # set attributes of this instance
        self._function = _types.FunctionType(function.func_code, sandbox, name)
    
    ### -------------------------------------------------------------------- ###
    
    def __call__(self, *args, **kwargs):
        
        return self._function(*args, **kwargs)
    
    ### -------------------------------------------------------------------- ###
    
    def __getstate__(self):
        
        # filter out modules from sandbox
        sandbox = filter(lambda (name, x): not isinstance(x, _types.ModuleType), 
                         self._function.func_globals.iteritems())
        modules = filter(lambda (name, x):     isinstance(x, _types.ModuleType), 
                         self._function.func_globals.iteritems())
        
        # define final pickleable instances of sandbox
        sandbox = dict(sandbox)
        modules = [(name, self._dump_module(x)) for name, x in modules]
        
        # return sate of this sandbox instance
        return dict(code=_marshal.dumps(self._function.func_code), 
                    sandbox=sandbox, modules=modules)
    
    def __setstate__(self, state):
        
        # define sandbox and modules of this instance
        sandbox =                                              state["sandbox"]
        modules = [(name, self._load_module(x)) for name, x in state["modules"]]
        
        # update sandbox of this instance
        sandbox.update(modules)
        
        # update state of this instance
        self._function = _types.FunctionType(_marshal.loads(state["code"]), sandbox)
    
    ### -------------------------------------------------------------------- ###
    
    @staticmethod
    def _dump_module(module):
        
        if "." in module.__name__:
            
            # retrieve name of module
            name = module.__name__.split(".")[-1]
            
            # retrieve path of module
            path = map(_os.path.dirname, 
                 module.__path__  if hasattr(module, "__path__") else \
                [module.__file__] if hasattr(module, "__file__") else [])
            
        else:
            
            # retrieve default name and path
            name = module.__name__
            path = None
        
        # return full module description
        return (module.__name__,) + _imp.find_module(name, path)
    
    @staticmethod
    def _load_module(args):
        
        # return initialized moduel from description
        return _imp.load_module(*args)
    
    ### -------------------------------------------------------------------- ###

### ======================================================================== ###