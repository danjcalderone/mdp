# -*- coding: utf-8 -*-
"""
Created on Mon Jun 04 15:40:26 2018

@author: craba
"""
import numpy as np;
    
def dynamicPLinearCost(c,P,yt, p0, hasToll =False, toll = None, tollState = None):
    states,actions,time = c.shape;
    V = np.zeros((states, time));
    policy = np.zeros((states, time)); # pi_t(state) = action;
    trajectory = np.zeros((states,time));
    # construct optimal value function and policy
    for tIter in range(time-1):
        t = time-1-tIter;   
        print "----------------------t = ", t," -----------------";
        if t == time-2:
            cCurrent =-np.multiply(c[:,:,t], yt[:,:,t]);
            if hasToll:    
                V[tollState,t] = V[tollState,t] + toll[t];
#            print cCurrent.shape;
            V[:,t] = np.max(cCurrent, axis = 1);
            print cCurrent;
            print V[:,t]
        else:
            cCurrent =-np.multiply(c[:,:,t], yt[:,:,t]);
            if hasToll:    
                V[tollState,t] = V[tollState,t] + toll[t];
            # solve Bellman operators
            Vt = V[:,t+1];
            obj = cCurrent + np.einsum('ijk,i',P,Vt);
            V[:,t] = np.max(obj, axis=1);
            pol = np.argmax(obj, axis=1);
            policy[:,t+1] = pol;
            print obj
            print V[:,t];
            print pol;
            

            
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
    
    print -sum([p0[state]*V[state,0] for state in range(states)])-0.5*sum([sum([c[state,0,t]*trajectory[state,t]*trajectory[state,t] for t in range(time)])  for state in range(states)]);
    return V, trajectory;

    
def dynamicP(c, P, p0, isQuad = False):
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
        
