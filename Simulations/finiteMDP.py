# -*- coding: utf-8 -*-
"""
Created on Thu May 31 22:49:38 2018

@author: craba
"""
import routes as route
import mdp as mdp
import constrainedMDP as cMDP
import dynamicProgramming as dp

import cvxpy as cvx
import numpy as np
import networkx as nx
import scipy.linalg as sla
import matplotlib.pyplot as plt



# networks Set up 
# Create Graph Object
rowSize = 3; colSize = 5;

G = nx.grid_graph([rowSize,colSize])
e = G.number_of_edges();
v = G.number_of_nodes();
G = nx.convert_node_labels_to_integers(G)
pos=nx.spring_layout(G);
#plt.figure();
#nx.draw(G, pos=pos, node_color='g', edge_color='k', with_labels=True, font_weight='bold')
#plt.show();
time = 20;
states = rowSize*colSize;
actions = 5;

P,c = mdp.generateGridMDP(states,actions,G,test= True)
R = np.zeros((states, actions,time))
p0 = np.zeros((states));
p0[0] = 1.0;
#Construct the time dependent reward
for t in range(time):
    R[:,:,t] = 1.0*c;
# Primal, completely unconstrained case    
print "Solving primal unconstrained case";
optRes,optDual = mdp.solveMDP(time, P, R,returnDual=True,verbose = False);
#mdp.drawOptimalPopulation(time,pos,G,optRes, is2D = False, constrainedState = None);

# solve constrained version: 
constrainedState = 6;
print "Solving constrained case, state 6 <= 0.2 case";
optCRes, tau = mdp.solveCMDP(time, P, R, constrainedState = constrainedState,returnDual = True,verbose=False);    
#mdp.drawOptimalPopulation(time,pos,G,optCRes, is2D = False, constrainedState = constrainedState);
# clean up tau
for i in range(len(tau)):
    if abs(tau[i]) <= 1e-8:
        tau[i] = 0.0;
#print tau; 




# solve problem again using unconstrained case: 
print "Solving unconstrained problem with new Toll";
optCSol,optCTau = mdp.solveMDP(time, P, R,
                               tau = tau,
                               constrainedState = constrainedState,
                               returnDual=True,verbose=True);
#mdp.drawOptimalPopulation(time,pos,G,optCSol,constrainedState=constrainedState)

toll = optCSol[constrainedState,:,:];
toll = cvx.pos(np.einsum('at->t',toll) - 0.2).value.A1; 
toll = np.multiply(toll,tau +0.1); 
tau[2] = tau[2] + 0.01;
tau[3] = tau[3] + 0.01;
# create new reward;
cR = cMDP.constrainedReward(c,tau,constrainedState,time);                            
 #solve problem again with dynamic programming 
print "Solving dynamic problem with new Toll";
dpVC, dpSolC = dp.dynamicP(cR,P,p0);
#mdp.drawOptimalPopulation(time,pos,G,dpSolC,is2D=True,constrainedState=constrainedState)


print 0.2*sum(tau)


#---------------------------------Junk---------------------------------------------
#final = np.sum(optRes[:,:,time-1],axis=1);

#plt.figure()
#timeLine = np.linspace(0,time,time);
#for i in range(states):
#    plt.plot(timeLine,np.sum(optRes[i,:,:], axis=0),label=(r'node %i'%(i+1)));
#plt.xlabel('Time [s]')
#plt.ylabel('Node States')
#plt.title('MDP Node Evolution Over time')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show();
#
#
#
#mag = 3000;
#nodesize=[final[f]*mag for f in G]
#cap = mag* np.ones(len(nodesize));

#plt.figure();
#nx.draw(G, pos=pos, node_color='w',with_labels=True, font_weight='bold')
#nx.draw_networkx_nodes(G,pos,node_size=3.2/3*cap,node_color='r',alpha=1)
#nx.draw_networkx_nodes(G,pos,node_size=cap,node_color='w',alpha=1)
#
#nx.draw_networkx_nodes(G,pos,node_size=nodesize,node_color='c',alpha=1)
#
#plt.show();   
   


