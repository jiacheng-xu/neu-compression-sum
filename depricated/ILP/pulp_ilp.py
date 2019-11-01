import cvxpy
from pulp import *
# * Solver <class 'pulp.solvers.PULP_CBC_CMD'> passed.
# * Solver <class 'pulp.solvers.GLPK_CMD'> passed.

x = LpVariable("x",  cat=LpBinary)

prob = LpProblem("Maximize",LpMaximize)

prob += lpSum([vars[i]*cost for i in PatternNames]),"Production Cost"
