# -*- coding: utf-8 -*-
"""
Created on Fri Jun 01 12:09:47 2018

@author: sarah
These are helpers for mdpRouting game class
"""
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
import networkx as nx
# ----------generate new constrained reward based on exact penalty-------------
# (for 2D reward matrix)
def constrainedReward2D(c,toll,constrainedState, time):
    states, actions = c.shape;
    constrainedC = np.zeros((states,actions,time));
    
    for t in range(time):
        constrainedC[:,:,t] = c;
        if toll[t] > 1e-8:
            for a in range(actions):
                constrainedC[constrainedState,a,t] += -toll[t];
    return constrainedC;
# (for 3D reward matrix):
def constrainedReward3D(c,toll,constrainedState):
    states, actions,time = c.shape;
    constrainedC = 1.0*c;

    for t in range(time):
        if toll[t] > 1e-8:
            for a in range(actions):
                constrainedC[constrainedState,a,t] += -toll[t];
                
    return constrainedC;   

def drawOptimalPopulation(time,pos,G,optRes, is2D = False, 
                          constrainedState = None, 
                          startAtOne = False, 
                          constrainedUpperBound = 0.2):
    frameNumber = time;
    v = G.number_of_nodes();
    fig = plt.figure();
    #ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    #line, = ax.plot([], [], lw=2)
    iStart = -5;
    mag = 1000;
    cap = mag* np.ones(v);  
    if(constrainedState != None):
        # Draw the red circle
        cap[constrainedState]= cap[constrainedState]*(constrainedUpperBound+0.3); 
    nx.draw(G, pos=pos, node_color='w',with_labels=True, font_weight='bold');
    nx.draw_networkx_nodes(G,pos,node_size=3.2/3*cap,node_color='r',alpha=1);
    if(constrainedState != None):
        # Draw the white circle
        cap[constrainedState]= cap[constrainedState]*(constrainedUpperBound+ 0.2); 
    dontStop = True; 
    try:
        
        print('running')
    
    except KeyboardInterrupt:
        print('paused')
        inp =input('continue? (y/n)')
    
    for i in range(iStart,frameNumber):
        try:
            if is2D:
                if i < 0:
                    frame = optRes[:,0];
                else:
                    frame = optRes[:,i];
            else:
                if i < 0:
                    frame = np.einsum('ij->i', optRes[:,:,0]);
                else:   
                    frame = np.einsum('ij->i', optRes[:,:,i]);
            if startAtOne:
                nodesize=[frame[f-1]*mag for f in G];
            else:
                nodesize=[frame[f]*frame[f]*mag for f in G];
            nx.draw_networkx_nodes(G,pos,node_size=cap,node_color='w',alpha=1)
            nx.draw_networkx_nodes(G,pos,node_size=nodesize,node_color='c',alpha=1)  
        except KeyboardInterrupt:
            dontStop = False;
        plt.show();
        plt.pause(0.5);
       
def generateGridMDP(v,a,G,p = 0.8,test = False):
    """
    Generates a grid MDP based on given graph. p is the probability of reaching the target state given an action.
    
    Parameters
    ----------
    v : int
        Cardinality of state space.
    a : int
        Cardinality of input space.
        
    Returns
    -------
    P : (S,S,A) array
        Transition probability tensor such that ``P[i,j,k]=prob(x_next=i | x_now=j, u_now=k)``.
    """
    debug = False;
    # making the transition matrix
    P = np.zeros((v,v,a));
    for node in range(v):#x_now = node
        neighbours = list(G.neighbors(node));
        totalN = len(neighbours);
        # chance of not reaching action
        pNot = (1.-p)/(totalN-1);
        actionIter = 0;
        if debug: 
            print neighbours;
        for neighbour in neighbours: # neighbour = x_next
            P[neighbour,node,actionIter] = p;
            for scattered in neighbours:
                if debug:
                    print scattered;
                if scattered != neighbour:
                    P[scattered,node,actionIter] = pNot;
            actionIter += 1;
        while actionIter < a:         
            P[node, node, actionIter] = p;
            pNot = (1.-p)/(totalN);
            for scattered in neighbours: 
                P[scattered,node,actionIter] = pNot;
            actionIter += 1;
    # making the cost function
    if test:
        c = np.zeros((v,a));
        c[12] = 10.;
    else:
        c = np.random.uniform(size=(v,a))
    return P,c;
#----------convert cvx variable dictionary into an array of dictated shape
def cvxDict2Arr(optDict, shapeList):
    arr = np.zeros(shapeList);
    for DIter, key in enumerate(optDict):
        arr[key] = optDict[key].value;
    return arr;
#----------convert cvx variable list into an array of dictated shape,
# mostly used for dual variables, since the cvx constraints are in lists
def cvxList2Arr(optList,shapeList,isDual):
    arr = np.zeros(shapeList);
    it = np.nditer(arr, flags=['f_index'], op_flags=['writeonly'])    
    for pos, item in enumerate(optList):
        if isDual:
            it[0] = item.dual_value;
        else:
            it[0] = item.value;
        
        it.iternext();                    
    return arr;
# truncate one D array to something more readable
def truncate(tau):
    for i in range(len(tau)):
        if abs(tau[i]) <= 1e-8:
            tau[i] = 0.0;
    return tau;
        
def generateMDP(v,a,G, p =0.8):
    """
    Generates a random MDP with finite sets X and U such that |X|=S and |U|=A.
    each action will take a state to one of its neighbours with p = 0.7
    rest of the neighbours will get p =0.3/(n-1) where n is the number of 
    neighbours of this state
    Parameters
    ----------
    S : int
        Cardinality of state space.
    A : int
        Cardinality of input space.
        
    Returns
    -------
    P : (S,S,A) array
        Transition probability tensor such that ``P[i,j,k]=prob(x_next=i | x_now=j, u_now=k)``.
    c : (S,A) array
        Cost such that ``c[i,j]=cost(x_now=i,u_now=j)``.
    """
    debug = False;
    P= np.zeros((v,v,a))
    for node in range(v):#x_now = node
        nodeInd = node+1;
        neighbours = list(G.neighbors(nodeInd));
        totalN = len(neighbours);
        # chance of not reaching action
        pNot = (1.-p)/(totalN);
        actionIter = 0;
        if debug: 
            print neighbours;
        for neighbour in neighbours: # neighbour = x_next
            neighbourInd = neighbour - 1;
            P[neighbourInd,node,actionIter] = p;
            for scattered in neighbours:
                scatteredInd = scattered -1;
                if debug:
                    print scattered;
                if scattered != neighbour:
                    # probablity of ending up at a neighbour
                    P[scatteredInd,node,actionIter] = pNot;
            # some probability of staying stationary
            P[node,node,actionIter] =pNot;
            actionIter += 1;        
        while actionIter < a:         
            P[node, node, actionIter] = p;
            pNot = (1.-p)/(totalN);
            for scattered in neighbours: 
                scatteredInd = scattered -1;
                P[scatteredInd,node,actionIter] = pNot;
            actionIter += 1;
    # test the cost function
    c = np.zeros((v,a))
    c[6] = 10.;
    return P,c
