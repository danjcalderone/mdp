# -*- coding: utf-8 -*-
"""
Created on Thu May 31 22:49:38 2018

@author: craba
"""
import routes as route
from cvxpy import *
import numpy as np
import networkx as nx
import scipy.linalg as sla
import mosek
import matplotlib.pyplot as plt
import matplotlib

# networks Set up 
# Create Graph Object
G = nx.grid_graph([3,5])
e = G.number_of_edges();
v = G.number_of_nodes();


# --------Source Vector set up ----------------#
sourceNode = 14; 
sinkNode = 7;
sourceVec = np.zeros((v));
sourceVec[sourceNode] = -1;
sourceVec[sinkNode] = 1;

#---------Draw Graph---------#
G = nx.convert_node_labels_to_integers(G)
plt.figure()
nx.draw(G, node_color='g', edge_color='k', with_labels=True, font_weight='bold')
edgeList = list(G.edges);
plt.show()

RouteMat = route.RouteGen(G,edgeList,sourceNode,sinkNode)
e,r = RouteMat.shape
#---------Construct the latency functions-----------
scale = 5;
A = np.diag(scale * np.random.random_sample((e)));
b = scale * np.random.random_sample((e));

# Construct the problem.
#----------------WARDROP EQUILIBRIUM--------------------
mass = 10.0;
x = Variable(e); 
z = Variable(r);
warPot =  b*x + quad_form(x,A)*0.5;
warObj = Minimize(warPot);
warConstraints = []
for i in range(10):
   warConstraints += [0 <= z]
                  RouteMat*z == x,
                  sum(z) == mass]
wardrop = Problem(warObj, warConstraints)
warRes = wardrop.solve(solver=MOSEK)
print(warRes)

# make flow the colours
flow = x.value.A1
# create weight dictionary
edgeDictionary = {}
for edge in range(0,len(edgeList)):
    edgeDictionary[edgeList[edge]] = round(flow[edge]/mass,2)


# Equilibrium flow of graph
pos=nx.spring_layout(G);
plt.figure()
nx.draw(G, pos=pos, node_color='g', edge_cmap=plt.cm.Blues, width=8,edge_color=(flow), with_labels=True, font_weight='bold')
nx.draw_networkx_edge_labels(G,pos,edge_labels=edgeDictionary);
nx.draw_networkx_nodes(G,pos, nodelist = {sourceNode,sinkNode},nodeColor='r')

# Equilibrium costs of graph
# make flow the colours
cost = A.dot(x.value).A1 + b;
# create weight dictionary
costDictionary = {}
for edge in range(0,len(edgeList)):
    costDictionary[edgeList[edge]] = round(cost[edge],2)
    
pos=nx.spring_layout(G);
plt.figure()
nx.draw(G, pos=pos, node_color='g', edge_cmap=plt.cm.Blues, width=8,edge_color=flow, with_labels=True, font_weight='bold')
nx.draw_networkx_edge_labels(G,pos,edge_labels=costDictionary);
nx.draw_networkx_nodes(G,pos, nodelist = {sourceNode,sinkNode},nodeColor='r')

