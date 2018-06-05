# -*- coding: utf-8 -*-
"""
Created on Mon Jun 04 17:43:16 2018

@author: craba
"""
import numpy as np;

def constrainedReward(c,toll,constrainedState, time):
    states, actions = c.shape;
    constrainedC = np.zeros((states,actions,time));
    
    for t in range(time):
        constrainedC[:,:,t] = c;
        if toll[t] > 1e-8:
            for a in range(actions):
                constrainedC[constrainedState,a,t] += -toll[t];
    return constrainedC;
