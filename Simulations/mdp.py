# -*- coding: utf-8 -*-
"""
Created on Fri Jun 01 12:09:47 2018

@author: craba
"""
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
import networkx as nx

def drawOptimalPopulation(time,pos,G,optRes, is2D = False, constrainedState = None):
    frameNumber = time;
    v = G.number_of_nodes();
    fig = plt.figure();
    #ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    #line, = ax.plot([], [], lw=2)
    iStart = -5;
    mag = 3000;
    cap = mag* np.ones(v);  
    if(constrainedState != None):
        cap[constrainedState]= cap[constrainedState]/5.; 
    nx.draw(G, pos=pos, node_color='w',with_labels=True, font_weight='bold');
    nx.draw_networkx_nodes(G,pos,node_size=3.2/3*cap,node_color='r',alpha=1);
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
            nodesize=[frame[f]*mag for f in G]
            nx.draw_networkx_nodes(G,pos,node_size=cap,node_color='w',alpha=1)
            nx.draw_networkx_nodes(G,pos,node_size=nodesize,node_color='c',alpha=1)  
        except KeyboardInterrupt:
            dontStop = False;
        plt.show();
        plt.pause(0.5);
        
            
def solveMDP(time, P, c, tau = None, constrainedState = None, 
             initDist = None, returnDual = False,verbose=False):
    # Construct the problem.
    #----------------MDP Routing Game--------------------
    states,actions,time = c.shape;
#    R = np.zeros((states, actions,time))
    R = c;
#    #Construct the time dependent reward
#    for t in range(time):
#        R[:,:,t] = 1.0*c;
    # y_ijt is 3D array with dimensions p x q x r.
    y_ijt = {}   
    for i in range(states):
        for j in range(actions):
            for t in range(time):
                y_ijt[(i,j,t)] = cvx.Variable() 
    # construct LP objective  
    objF = sum([sum([sum([y_ijt[(i,j,t)]*R[i,j,t] for i in range(states) ]) for j in range(actions)]) for t in range(time)])
    
    if constrainedState != None:
        epsilon = 0.1; # make optimal solution an interior solution
        objF += -sum([(tau[t] + epsilon)*cvx.pos(sum([y_ijt[(constrainedState,j,t)] - 0.04 
                       for j in range(actions)]))
                for t in range(time)])
    mdpObj = cvx.Maximize(objF)    
    # construct constraints
    positivity = [];
    massConservation = []; #constraintCounter = 0;
    initialCondition = [];
    
    for i in range(states):
        # Enforce initial conditoin
        initState = sum([y_ijt[(i,j,0)] for j in range(actions)]);
        if initDist == None:
            if i == 0:
                initialCondition.append(initState == 1.)
            else:
                initialCondition.append(initState == 0.)
        else: 
            initialCondition.append(initState == initDist[i]);     
            
        for t in range(time):  
            if t < time-1:
                # mass conservation constraints between timesteps
                prevProb = sum([sum([y_ijt[(iLast,j,t)]*P[i,iLast,j] for iLast in range(states) ]) for j in range(actions)]) ;
                newProb = sum([y_ijt[(i,j,t+1)] for j in range(actions)]);
                massConservation.append(newProb == prevProb);
                
            for j in range(actions):
                # positivity constraints
                positivity.append(y_ijt[(i,j,t)] > 0.)

                
    mdpPolicy = cvx.Problem(mdpObj,positivity+massConservation+initialCondition);
    
    mdpRes = mdpPolicy.solve(verbose=verbose)
    
    print mdpRes
    optRes = cvxDict2Arr(y_ijt,[states,actions,time]);
    optDual = cvxList2Arr(massConservation,[states,time-1],returnDual);
    
    return optRes,optDual;

#-------------------------Solving constrained MDP-------------------------------------------
def solveCMDP(time, P, c, constrainedState, initDist = None,returnDual = False,verbose = False):
    # Construct the problem.
    #----------------MDP Routing Game--------------------
    states,actions,time = c.shape;
    R = 1.0*c
    # Construct the time dependent reward
#    for t in range(time):
#        R[:,:,t] = 1.0*c;
    # y_ijt is 3D array with dimensions p x q x r.
    y_ijt = {}   
    for i in range(states):
        for j in range(actions):
            for t in range(time):
                y_ijt[(i,j,t)] = cvx.Variable() 
    # construct LP objective    
    mdpObj = cvx.Maximize(sum([sum([sum([y_ijt[(i,j,t)]*R[i,j,t] for i in range(states) ]) for j in range(actions)]) for t in range(time)]))
    
    # construct constraints
    positivity = [];
    massConservation = []; #constraintCounter = 0;
    initialCondition = [];
    densityConstraints = [];
    
    for i in range(states):
        # Enforce initial conditoin
        initState = sum([y_ijt[(i,j,0)] for j in range(actions)]);
        if initDist == None:
            if i == 0:
                initialCondition.append(initState == 1.)
            else:
                initialCondition.append(initState == 0.)
        else: 
            initialCondition.append(initState == initDist[i]);     
            
        for t in range(time):  
            if t < time-1:
                # mass conservation constraints between timesteps
                prevProb = sum([sum([y_ijt[(iLast,j,t)]*P[i,iLast,j] for iLast in range(states) ]) for j in range(actions)]) ;
                newProb = sum([y_ijt[(i,j,t+1)] for j in range(actions)]);
                massConservation.append(newProb == prevProb);
                
            for j in range(actions):
                # positivity constraints
                positivity.append(y_ijt[(i,j,t)] > 0.)
                         
            
 # NOTE EXTRA DENSITY CONSTRAINT   
    for t in range(time):
        densityConstraints.append(sum([y_ijt[(constrainedState,j,t)] for j in range(actions)])  <= 0.2);

    
    mdpPolicy = cvx.Problem(mdpObj,positivity+massConservation+initialCondition+densityConstraints);

    
    mdpRes = mdpPolicy.solve(verbose=verbose)
    
    print mdpRes
    optRes = cvxDict2Arr(y_ijt,[states,actions,time]);
    optDual = cvxList2Arr(densityConstraints,[time],returnDual);
       
    return optRes, optDual;

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

def cvxDict2Arr(optDict, shapeList):
    arr = np.zeros(shapeList);
    for DIter, key in enumerate(optDict):
        arr[key] = optDict[key].value;
    return arr;

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

def generateMDP(S,A):
    """
    Generates a random MDP with finite sets X and U such that |X|=S and |U|=A.
    
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
    P, c = np.zeros((S,S,A)), np.random.uniform(size=(S,A))
    for j in range(S):
        for k in range(A):
            P[:,j,k] = np.random.uniform(size=S)
            P[:,j,k] /= np.sum(P[:,j,k])
    return P, c
