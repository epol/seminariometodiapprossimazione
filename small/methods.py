#!/usr/bin/env python3

import logging

import numpy
import numpy.linalg


def newton(A,tol,maxiter,norm=numpy.linalg.norm):
    i = 0
    X = A
    try:
        res = norm(numpy.dot(X,X) - A)
        (res > tol).all()
    except:
        logging.exception('Invalid norm function')
        return None
    while (res > tol).all() and (i < maxiter):
        try:
            X = 0.5 * (X + numpy.linalg.solve(X,A))
        except numpy.linalg.LinAlgError:
            logging.critical("Singularity reached")
            break
        res = norm( numpy.dot(X,X) - A )
        i = i+1
    if (res> tol).all():
        logging.warning("Method hasn't converged with enough iterations")
    return X

def newton_diagnostic(A,tol,maxiter,norm=numpy.linalg.norm):
    i = 0
    X = [ A]
    try:
        res = [ norm(numpy.dot(X,X) - A) ]
        (res[-1] > tol).all()
    except:
        logging.exception('Invalid norm function')
        return None
    commutativity = [ norm(numpy.dot(A,X[-1]) - numpy.dot(X[-1],A)) ]
    while (res[-1] > tol).all() and (i < maxiter):
        try:
            X_new = 0.5* ( X[-1] + numpy.linalg.solve(X[-1],A))
        except numpy.linalg.LinAlgError:
            logging.critical("Singularity reached")
            break            
        X.append(X_new)
        res.append(norm( numpy.dot(X_new,X_new) - A ))
        commutativity.append(norm(numpy.dot(A,X_new) - numpy.dot(X_new,A)))
        i = i+1
    if (res[-1]> tol).all():
        logging.warning("Method hasn't converged with enough iterations")
    return { 'result': X[-1],
             'approximations': X,
             'residues': res,
             'iterations': i,
             'commutativity': commutativity
    }

def db(A,tol,maxiter,norm=numpy.linalg.norm):
    i = 0
    X = A
    Y = numpy.eye(*(A.shape))
    try:
        res = norm(numpy.dot(X,X) - A)
        (res > tol).all()
    except:
        logging.exception('Invalid norm function')
        return None
    while (res > tol).all() and (i < maxiter):
        try:
            Xinv = numpy.linalg.inv(X)
            Yinv = numpy.linalg.inv(Y)
        except numpy.linalg.LinAlgError:
            logging.critical("Singularity reached")
            break
        X = 0.5 * (X + Yinv)
        Y = 0.5 * (Y + Xinv)
        res = norm( numpy.dot(X,X) - A )
        i = i+1
    if (res> tol).all():
        logging.warning("Method hasn't converged with enough iterations")
    return X

def db_diagnostic(A,tol,maxiter,norm=numpy.linalg.norm):
    i = 0
    X = [ A]
    Y = [ numpy.eye(*(A.shape)) ]
    try:
        res = [ norm(numpy.dot(X,X) - A) ]
        (res[-1] > tol).all()
    except:
        logging.exception('Invalid norm function')
        return None
    commutativity = [ norm(numpy.dot(Y[-1],X[-1]) - numpy.dot(X[-1],Y[-1])) ]
    while (res[-1] > tol).all() and (i < maxiter):
        try:
            Xinv = numpy.linalg.inv(X[-1])
            Yinv = numpy.linalg.inv(Y[-1])
        except numpy.linalg.LinAlgError:
            logging.critical("Singularity reached")
            break            
        X_new = 0.5* ( X[-1] + Yinv)
        Y_new = 0.5* ( Y[-1] + Xinv)
        X.append(X_new)
        Y.append(Y_new)
        res.append(norm( numpy.dot(X_new,X_new) - A ))
        commutativity.append(norm(numpy.dot(Y_new,X_new) - numpy.dot(X_new,Y_new)))
        i = i+1
    if (res[-1]> tol).all():
        logging.warning("Method hasn't converged with enough iterations")
    return { 'result': X[-1],
             'approximations': X,
             'invapproximations': Y,
             'residues': res,
             'iterations': i,
             'commutativity': commutativity
    }
