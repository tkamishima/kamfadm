#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two Class logistic regression module with Unfairness Hater

the number of sensitive features is restricted to one, and the feature must
be binary.

Attributes
----------
EPSILON : floast
    small positive constant
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

#==============================================================================
# Module metadata variables
#==============================================================================

#==============================================================================
# Imports
#==============================================================================

import logging
import numpy as np
from scipy.optimize.optimize import fmin_cg
from sklearn.base import BaseEstimator, ClassifierMixin

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['LogisticRegressionWithUnfairnessHaterType1',
           'LogisticRegressionWithUnfairnessHaterType2']

#==============================================================================
# Constants
#==============================================================================

EPSILON = 1.0e-10
SIGMOID_RANGE = np.log((1.0 - EPSILON) / EPSILON)

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Functions
#==============================================================================

def sigmoid(x, w):
    """ sigmoid(w^T x)
    To suppress the warnings at np.exp, do "np.seterr(all='ignore')

    Parameters
    ----------
    x : array, shape=(d)
        input vector
    w : array, shape=(d)
        weight

    Returns
    -------
    sigmoid : float
        sigmoid(w^T x)
    """

    s = np.clip(np.dot(w, x), -SIGMOID_RANGE, SIGMOID_RANGE)

    return 1.0 / (1.0 + np.exp(-s))

#==============================================================================
# Classes
#==============================================================================

class LikelihoodType1Mixin(object):
    """ mixin for singe type 1 likelihood

    Parameters
    ----------
    X : array, shape=(n_samples, n_features)
        feature vectors of samples
    y : array, shape=(n_samples)
        target class of samples
    ns : int
        number of sensitive variables (fix to 1)
    ufc : MyLogisticRegression
        unfair classifier
    ignore_sensitive : bool
        if True, clear the weights for sensitive features
    params : any
        arguments to optmizer
    """

    def fit(self, X, y, ns, ufc, ignore_sensitive=False, **params):

        # fix ns to 1 in current version
        ns = 1

        # compute weights
        Xw = np.array([[0.0], [1.0]])
        self.w_ = ufc.predict_proba(Xw)[:, 1]

        # add a constanet term
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # check optimization parameters
        if not 'disp' in params:
            params['disp'] = False
        if not 'maxiter' in params:
            params['maxiter'] = 100

        self.coef_ = np.zeros(X.shape[1])
        self.coef_ = fmin_cg(self.loss,
                             self.coef_,
                             fprime=self.grad_loss,
                             args=(X, y, ns),
                             **params)

        # clear the weights for sensitive features
        if ignore_sensitive:
            self.coef_[-ns:] = 0.0

    def predict_proba(self, X):
        """ predict probabilities

        Parameters
        ----------
        X : array, shape=(n_samples, n_features)
            feature vectors of samples

        Returns
        -------
        y_proba : array, shape=(n_samples, n_classes), dtype=float
            array of predicted class
        """

        # add a constanet term
        X = np.array(X)
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        prob = np.empty((X.shape[0], 2))
        prob[:, 1] = [sigmoid(X[i, :], self.coef_)
                      for i in xrange(X.shape[0])]
        prob[:, 0] = 1.0 - prob[:, 1]

        return prob

class LogisticRegressionWithUnfairnessHater(BaseEstimator, ClassifierMixin):
    """ Two class LogisticRegression with Unfairness Hater

    Parameters
    ----------
    C : float
        regularization parameter
    eta : float
        penalty parameter
    fit_intercept : bool
        use a constant term
    penalty : str
        fixed to 'l2'

    Attributes
    ----------
    minor_type : int
        type of likelihood fitting
    `coef_` : array, shape=(n_features)
        parameters for logistic regression model
    `w_` : array-like, shpae(dom_sensitive_feature)
        probabilities of the positive class given sensitive features
    """

    def __init__(self, C=1.0, eta=1.0, fit_intercept=True, penalty='l2'):

        if C < 0.0:
            raise TypeError
        self.fit_intercept = fit_intercept
        self.penalty = 'l2'
        self.C = C
        self.coef_ = None

        self.eta = eta
        self.lossfunc_type = 0

    def predict(self, X):
        """ predict classes

        Parameters
        ----------
        X : array, shape=(n_samples, n_features)
            feature vectors of samples

        Returns
        -------
        y : array, shape=(n_samples), dtype=int
            array of predicted class
        """

        return np.argmax(self.predict_proba(X), 1)

class LogisticRegressionWithUnfairnessHaterType1\
    (LogisticRegressionWithUnfairnessHater,
     LikelihoodType1Mixin):
    """ Two class LogisticRegression with Unfairness Hater

    Loss function type 1: sensitive feature is included in the model of
    Pr[y | x, s]
    
    ::
    
        Likelihood = sum{(y,x,s)\in D} Pr[y | x,s; [w,v]]

    Parameters
    ----------
    C : float
        regularization parameter
    eta : float
        penalty parameter
    fit_intercept : bool
        use a constant term
    penalty : str
        fixed to 'l2'
    """

    def __init__(self, C=1.0, eta=1.0, fit_intercept=True, penalty='l2'):

        super(LogisticRegressionWithUnfairnessHaterType1, self).\
            __init__(C=C, eta=eta,
                     fit_intercept=fit_intercept, penalty=penalty)
        lossfunc_type = 1

    def loss(self, coef, X, y, ns):
        """ loss function: negative log-likelihood with l2 regularizer
        To suppress the warnings at np.log, do "np.seterr(all='ignore')

        Parameters
        ----------
        coef : array, shape=(n_features)
            coefficients of model
        X : array, shape=(n_samples, n_features)
            feature vectors of samples
        y : array, shape=(n_samples)
            target class of samples
        ns : int
            number of sensitive variables

        Returns
        -------
        loss : float
            loss function value
        """

#        print >> sys.stderr, "loss:", coef,

        #likelihood
        # sigma := sigmoid([w,v]^T, [x,s])
        # \sum_{x,s,y in D} y log(sigma) + (1 - y) log(1 - sigma)
        p = np.array([sigmoid(X[i, :], coef)
                      for i in xrange(X.shape[0])])
        l = np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

        # penalty term
        # KL-div from the most unfair estimator
        # sigma := sigmoid([w,v]^T, [x,s])
        # psi := sigmoid([ws*]^T s)
        # \sum_{x,s in D} \sum_y psi log(sigma) + (1 -psi) log(1 - sigma)
        q = self.w_[X[:, -ns:].ravel().astype(int)]
        f = np.sum(q * np.log(p) + (1.0 - q) * np.log(1.0 - p))

        # l2 regularizer
        reg = np.dot(coef, coef)

#        print >> sys.stderr, - l + self.eta * f + 0.5 * self.C * reg
        return -l + self.eta * f + 0.5 * self.C * reg

    def grad_loss(self, coef, X, y, ns):
        """ first derivative of loss function

        Parameters
        ----------
        coef : array, shape=(n_features)
            coefficients of model
        X : array, shape=(n_samples, n_features)
            feature vectors of samples
        y : array, shape=(n_samples)
            target class of samples
        ns : int
            number of sensitive variables

        Returns
        -------
        grad_loss : float
            first derivative of loss function
        """

#        print >> sys.stderr, "diff:", coef,

        #likelihood
        # sigma := sigmoid([w,v]^T, [x,s])
        # \sum_{x,s,y in D} (y - sigma) x
        p = np.array([sigmoid(X[i, :], coef)
                      for i in xrange(X.shape[0])])
        l = np.sum((y - p)[:, np.newaxis] * X, axis=0)

        # fairness penalty term
        # sigma := sigmoid([w,v]^T, [x,s])
        # psi := sigmoid([ws*]^T s)
        # (psi - sigma) [x,s]
        q = self.w_[X[:, -ns:].ravel().astype(int)]
        f = np.sum((q - p)[:, np.newaxis] * X, axis=0)

        # l2 regularizer
        reg = coef

#        print >> sys.stderr, -l + self.eta * f + self.C * coef

        return -l + self.eta * f + self.C * reg

class LogisticRegressionWithUnfairnessHaterType2\
    (LogisticRegressionWithUnfairnessHater,
     LikelihoodType1Mixin):
    """ Two class LogisticRegression with Unfairness Hater

    Loss function type 2: sensitive feature is excluded in the model of
    Pr[y | x, s].
    
    ::
    
        Likelihood = sum{(y,x\in D} Pr[y | x; w]

    Parameters
    ----------
    C : float
        regularization parameter
    eta : float
        penalty parameter
    fit_intercept : bool
        use a constant term
    penalty : str
        fixed to 'l2'
    """

    def __init__(self, C=1.0, eta=1.0, fit_intercept=True, penalty='l2'):

        super(LogisticRegressionWithUnfairnessHaterType2, self).\
            __init__(C=C, eta=eta,
                     fit_intercept=fit_intercept, penalty=penalty)
        lossfunc_type = 2

    def loss(self, coef, X, y, ns):
        """ loss function: negative log-likelihood with l2 regularizer
        To suppress the warnings at np.log, do "np.seterr(all='ignore')

        Parameters
        ----------
        coef : array, shape=(n_features)
            coefficients of model
        X : array, shape=(n_samples, n_features)
            feature vectors of samples
        y : array, shape=(n_samples)
            target class of samples
        ns : int
            number of sensitive variables

        Returns
        -------
        loss : float
            loss function value
        """

#        print >> sys.stderr, "loss:", coef,

        #likelihood
        # sigma := sigmoid([w,v]^T, [x,s])
        # \sum_{x,s,y in D} y log(sigma) + (1 - y) log(1 - sigma)
        p = np.array([sigmoid(X[i, :-ns], coef[:-ns])
                      for i in xrange(X.shape[0])])
        l = np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

        # penalty term
        # KL-div from the most unfair estimator
        # sigma := sigmoid([w,v]^T, [x,s])
        # psi := sigmoid([ws*]^T s)
        # \sum_{x,s in D} \sum_y psi log(sigma) + (1 -psi) log(1 - sigma)
        q = self.w_[X[:, -ns:].ravel().astype(int)]
        f = np.sum(q * np.log(p) + (1.0 - q) * np.log(1.0 - p))

        # l2 regularizer
        reg = np.dot(coef, coef)

#        print >> sys.stderr, - l + self.eta * f + 0.5 * self.C * reg
        return -l + self.eta * f + 0.5 * self.C * reg

    def grad_loss(self, coef, X, y, ns):
        """ first derivative of loss function

        Parameters
        ----------
        coef : array, shape=(n_features)
            coefficients of model
        X : array, shape=(n_samples, n_features)
            feature vectors of samples
        y : array, shape=(n_samples)
            target class of samples
        ns : int
            number of sensitive variables

        Returns
        -------
        grad_loss : float
            first derivative of loss function
        """

#        print >> sys.stderr, "diff:", coef,

        #likelihood
        # sigma := sigmoid([w,v]^T, [x,s])
        # \sum_{x,s,y in D} (y - sigma) x
        p = np.array([sigmoid(X[i, :-ns], coef[:-ns])
                      for i in xrange(X.shape[0])])
        l = np.sum((y - p)[:, np.newaxis] * X[:, :-ns], axis=0)
        l = np.r_[l, np.zeros(ns)]

        # fairness penalty term
        # sigma := sigmoid([w,v]^T, [x,s])
        # psi := sigmoid([ws*]^T s)
        # (psi - sigma) [x,s]
        q = self.w_[X[:, -ns:].ravel().astype(int)]
        f = np.sum((q - p)[:, np.newaxis] * X, axis=0)

        # l2 regularizer
        reg = coef

#        print >> sys.stderr, -l + self.eta * f + self.C * coef

        return -l + self.eta * f + self.C * reg

#==============================================================================
# Module initialization
#==============================================================================

# init logging system

logger = logging.getLogger('fadm')
if not logger.handlers:
    logger.addHandler(logging.NullHandler)

#==============================================================================
# Test routine
#==============================================================================

def _test():
    """ test function for this module
    """

    # perform doctest
    import sys
    import doctest

    doctest.testmod()

    sys.exit(0)

# Check if this is call as command script

if __name__ == '__main__':
    _test()
