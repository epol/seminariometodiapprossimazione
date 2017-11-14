#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import methods
import scipy.linalg
import logging
import matplotlib.pyplot as pyplot
import prettytable
import os

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
    pyplot.axvline(x=bestiter,label='Best iteration')
    pyplot.legend(loc='best')
    fig2 = pyplot.figure()
    pyplot.semilogy(errrel,label='Relative error')
    pyplot.axvline(x=bestiter,label='Best iteration')
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

def psi_n(eigs):
    sqrteigs = numpy.sqrt(eigs)
    return numpy.max(numpy.abs(1 - numpy.dot(sqrteigs[:,None],sqrteigs[None,:]) ))

def pretty_experiments(A,outradix):
    S = find_best(A,1e-30,1000)
    data = try_all(A,S)
    table = prettytable.PrettyTable(['Method','Type','Iteration','Absolute error','Relative error','Square residue'])
    for name in data.keys():
        exp = data[name]
        table.add_row([name,'final',exp['final']['index'],exp['final']['errabs'],exp['final']['errrel'],exp['final']['residue']])
        table.add_row([name,'best',exp['best']['index'],exp['best']['errabs'],exp['best']['errrel'],exp['best']['residue']])
    print(table)
    latextable = """\\begin{tabular}{r| c c c c}
Method & Iteration & Absolute error & Relative error & Square residue \\\\
"""
    for name in data.keys():
        exp = data[name]
        latextable +="\\hline\n"
        for t in ['final','best']:
            latextable += "{name} & ${index}$ & ${errabs:.2e}$ & ${errrel:.2e}$ & ${res:.2e}$ \\\\\n".format(name=name,type=t,index=exp[t]['index'],errabs=exp[t]['errabs'],errrel=exp[t]['errrel'],res=exp[t]['residue'])
    latextable += "\\end{tabular}"

    try:
        os.mkdir(outradix)
    except OSError:
        pass
    for name in data.keys():
        exp = data[name]
        exp['absplot'].savefig(outradix+'/'+name+' - '+'absplot.png')
        exp['relplot'].savefig(outradix+'/'+name+' - '+'relplot.png')

    eigs = numpy.linalg.eig(A)[0]
    m = 1.2*numpy.max(numpy.abs(eigs))
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    pyplot.ylim(-m,m)
    pyplot.xlim(-m,m)
    ax.plot(numpy.real(eigs),numpy.imag(eigs),'o')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('white')
    ax.spines['top'].set_color('white')
    fig.savefig(outradix+'/eigs.png')
    
    return { 'latex': latextable,
             'data': data,
             'psi_n': psi_n(eigs),
             'scond': numpy.linalg.cond(S)
    }
