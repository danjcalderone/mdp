# -*- coding: utf-8 -*-
"""
Created on Fri Jun 01 12:39:33 2018

@author: craba
"""

import cvxpy as cvx
import matplotlib.pyplot as plt
# adding subroutines
import util.utilities as ut
#--------------------a cvx problem-------------------------------------#
A,B = 1.,1.

x = cvx.Variable(10)
u = cvx.Variable(9)
constraints = [x[0] == 1.]
for k in range(9):
    constraints.append(x[k+1] == A*x[k]+B*u[k])
    constraints.append(-10 <= u[k])
    constraints.append(u[k] <= 10)
cost = cvx.Minimize(sum([x[k]**2 for k in range(10)])+sum([u[k]**2 for k in range(9)]))
pbm = cvx.Problem(cost, constraints)
sol = pbm.solve(verbose=True)

#--------------------plots using python -------------------------------------#
fig = plt.figure(1)
plt.clf()
ax1 = fig.add_subplot(211)
ax1.plot(x.value)
ax1.set_ylabel('x')
ax1.legend(); # add legend
ax1.grid(); # add grid
ax2 = fig.add_subplot(212)
ax2.plot(u.value)
ax2.set_ylabel('u')
ax2.legend(); # add legend
ax2.grid(); # add grid
plt.show()