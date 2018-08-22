# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:03:22 2018

@author: craba
"""
import numpy as np
import dynamicProgramming as dp

def ALM(states, actions, time, p0, R, C, P, maxErr = 1e-1):
    maxIterations = 8e5;
    it = 1;
    err = 1000.;
    ytsa = np.zeros((states, actions, time));
    cost = np.zeros((states, actions, time))
    while it <= maxIterations and err >= maxErr:
        step = 1./it;
        lastCost = cost;
        # run value iteration on costs to generate new flow
        V, valNext, yNext =  dp.instantDP(R, C, P,ytsa, p0)
        # merge the flows
        ytsa = (1. - step)*ytsa + step*yNext;
        # calculate new costs
        cost =-np.multiply(R, ytsa) + C;
        err = np.linalg.norm(lastCost - cost);
        it += 1;
        
    print " ------------ MSA summary -----------";
    print "number of iterations = ", it;
    print "total error in cost function = ", err;
    return cost, ytsa;