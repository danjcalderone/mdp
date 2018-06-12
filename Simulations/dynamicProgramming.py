# -*- coding: utf-8 -*-
"""
Created on Mon Jun 04 15:40:26 2018

@author: craba
"""
import numpy as np;

def dynamicP(c, P, p0):
    states,actions,time = c.shape;
    V = np.zeros((states, time));
    policy = np.zeros((states, time)); # pi_t(state) = action;
    trajectory = np.zeros((states,time));
    # construct optimal value function and policy

    for tIter in range(time):
        t = time-1-tIter;
        
        if t == time-1:
            cCurrent = c[:,:, t];
            V[:,t] = np.max(cCurrent, axis = 1);
        else:
            cCurrent = c[:,:,t];
            # solve Bellman operators
            Vt = V[:,t+1];
            obj = cCurrent + np.einsum('ijk,i',P,Vt);
            V[:,t] = np.max(obj, axis=1);
            pol = np.argmax(obj, axis =1);
            policy[:,t+1] = pol;

            
    for t in range(time):
        # construct next trajectory
        if t == 0:
            trajectory[:,t] = p0;
        else:
            # construct y
            pol = policy[:,t];
#            print pol;
            y = np.zeros((states,actions));
            traj = trajectory[:,t-1];
            for s in range(states):
                y[s,int(pol[s])] = traj[s];
            trajectory[:,t] =  np.einsum('ijk,jk',P,y);
    
    print sum([p0[state]*V[state,0] for state in range(states)]);
    return V, trajectory;   
        
