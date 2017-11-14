#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import methods
import scipy.linalg
import logging
import matplotlib.pyplot as pyplot

def find_best(A,tol=1e-20,maxiter=100):
    S = None
    res = numpy.inf
    used = None
    for (name,func) in [ ('direct',scipy.linalg.sqrtm),
                         ('proddb',lambda X : methods.proddb(X,tol,maxiter,raiseifbig=True)),
                         ('cr',lambda X : methods.cr(X,tol,maxiter,raiseifbig=True)) ]:
        try:
            thisS = func(A)
        except:
            logging.warning("Exception during the evaluation of {name}".format(name=name))
            continue
        else:
            thisres = numpy.linalg.norm(numpy.dot(thisS,thisS)-A)
            if thisres < res:
                S = thisS
                res = thisres
                used = name
    if S is None:
        logging.critical("No valid approximation found")
        raise Exception("Square root not found")
    else:
        logging.info("Using {name} approximation".format(name=used))
        return S

def try_one(A,func,S=None,tol=1e-16,maxiter=100):
    if S is None:
        S = find_best(A,tol/100,maxiter*10)
    d = func(A,tol,maxiter)
    errabs = numpy.linalg.norm(d['approximations'] - S,axis=(1,2))
    errrel = errabs/numpy.linalg.norm(S)
    bestiter = numpy.argmin(errabs)
    fig1 = pyplot.figure()
    pyplot.semilogy(d['residues'],label='Residues')
    pyplot.semilogy(d['commutativity'],label='Commutativity')
    pyplot.semilogy(errabs,label='Absolute error')
    pyplot.legend(loc='best')
    fig2 = pyplot.figure()
    pyplot.semilogy(errrel,label='Relative error')
    pyplot.legend(loc='best')
    return {
        'final': {
            'index': d['iterations'],
            'value': d['result'],
            'errabs': errabs[-1],
            'errrel': errrel[-1],
            'residue': numpy.linalg.norm(numpy.dot(d['result'],d['result'])-A)
        },
        'best': {
            'index': bestiter,
            'value': d['approximations'][bestiter],
            'errabs': errabs[bestiter],
            'errrel': errrel[bestiter],
            'residue': numpy.linalg.norm(numpy.dot(d['approximations'][bestiter],d['approximations'][bestiter])-A)
        },
        'absplot': fig1,
        'relplot': fig2,
    }

def try_all(A,S=None,tol=1e-16,maxiter=100):
    data = {}
    if S is None:
        S = find_best(A,tol/100,maxiter*10)
    for (name,func) in [ ('Newton',methods.newton_diagnostic),
                         ('DB',methods.db_diagnostic),
                         ('Product DB',methods.proddb_diagnostic),
                         ('CR',methods.cr_diagnostic) ]:
        exp = try_one(A,func,S,tol,maxiter)
        data[name] = exp
    return data
