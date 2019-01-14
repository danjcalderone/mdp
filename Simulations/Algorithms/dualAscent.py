# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:04:16 2018

@author: craba
"""
import numpy as np
import Algorithms.frankWolfe as fw
import numpy.linalg as la
def dualAscent(lambda0, 
               y0, 
               p0, 
               P, 
               cState,
               cThresh, 
               R,
               C,
               maxErr = 1.0,
               optVar = None):
    states, actions, time = y0.shape;
    it= 0;
    maxIterations = 500;
    lamb = lambda0;
    lambHist = np.zeros(maxIterations); 
    yHist = np.zeros((states, actions, time, maxIterations));
    certificate = np.zeros(maxIterations);
    yCState = np.zeros(maxIterations);
    err = maxErr *2;
    while it < maxIterations and err >= maxErr:
#        print "at iteration ",it;
        def gradF(x): 
            return -np.multiply(R,x) + C + lamb;
        ytCThresh, ytCHist, dk = fw.FW(y0, p0, P, gradF, True, maxError = 1e-1,returnLastGrad = True);
        for i in range(3,time):
            #alpha factor is one
            if lamb[cState,0,i] + cThresh - np.sum(ytCThresh[cState,:,i])  > 0:
                lamb[cState,:,i] += cThresh - np.sum(ytCThresh[cState,:,i]) ;
            else:
                lamb[cState,:,i] = np.zeros(actions);
            
        lambHist[it] = la.norm(lamb - optVar);
        yCState[it] =  np.sum(ytCThresh[cState,:,time-3]);
        yHist[:,:,:,it] = ytCThresh;
        certificate[it] = np.sum(np.multiply(ytCThresh - dk, gradF(ytCThresh)))
        it +=1;
    return certificate, yCState, lambHist, lamb;
 
def admm(lambda0, 
         rho,
         y0, 
         p0, 
         P, 
         cState,
         cThresh, 
         R,
         C,
         maxErr = 1.0,
         optVar = None):
    states, actions, time = y0.shape;
    it= 0;
    maxIterations = 100;
    lamb = lambda0;
    certificate = np.zeros(maxIterations);
    lambHist = np.zeros(maxIterations); 
    yHist = np.zeros(maxIterations);
    err = maxErr *2;
    while it < maxIterations and err >= maxErr:
        if it%20 == 0:    
            print "at iteration ",it;
        slack = np.zeros((states,actions,time));
        def gradF(x): 
            ell = -np.multiply(R,x) + C;
            penalty = lamb;
            residual = np.zeros((states,actions,time));
            for i in range(3, time):
                if np.sum(x[cState,:,i]) < cThresh:
                    residual[cState,:,i] = cThresh - np.sum(x[cState,:,i]);
            return ell + penalty + rho*(residual + slack);
        ytCThresh, ytCHist, dk = fw.FW(y0, p0, P, gradF, True, maxError = 1e-1, returnLastGrad= True);
        for i in range(3,time):
            # alpha factor is one
            if lamb[cState,0,i]/rho + cThresh - np.sum(ytCThresh[cState,:,i])  > 0:
                # slack is zero
                lamb[cState,:,i] += rho*(cThresh - np.sum(ytCThresh[cState,:,i]) );
            else:
                lamb[cState,:,i] = np.zeros(actions);
                slack[cState,:,i] = np.sum(ytCThresh[cState,:,i]) - lamb[cState,0,i]/rho - cThresh; 
                
        lambHist[it] = la.norm(lamb - optVar);
        yHist[it] =  np.sum(ytCThresh[cState,:,time-3]);
        certificate[it] = np.sum(np.multiply(ytCThresh - dk, gradF(ytCThresh)))
        it +=1;
    return yHist, lambHist,certificate, lamb;
               