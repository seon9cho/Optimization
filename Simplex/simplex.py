# simplex.py
"""Volume 2: Simplex.
Seong-Eun Cho
Math 323
3/14/18

Problems 1-6 give instructions on how to build the SimplexSolver class.
The grader will test your class by solving various linear optimization
problems and will only call the constructor and the solve() methods directly.
Write good docstrings for each of your class methods and comment your code.

prob7() will also be tested directly.
"""

import numpy as np


# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        maximize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the tableau.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        # Save the dimensions for future use
        self.n = len(c)
        self.m = len(b)
        # Check if the origin is feasible
        x0 = np.zeros(self.n)
        if not (A@x0 <= b).all():
            raise ValueError("The problem is not feasible at the origin.")
        # Save each of the arrays
        self.c = c
        self.A = A
        self.b = b
        # Create an index list
        self.L = [self.n + i for i in range(self.m)] + \
                 [i for i in range(self.n)]
        
    def init_tableau(self):
        """Create an initial tableau using the given arrays.
        
        Returns:
            ((m+1,m+n+2) ndarray): The initial tableau.
        """
        # Exactly follow the form given in 14.1
        Im = np.eye(self.m)
        A_bar = np.column_stack((self.A,Im))
        c_bar = np.concatenate((self.c, np.zeros(self.m)))
        T1 = np.concatenate(([0], -c_bar, [1]))
        T2 = np.column_stack((self.b, A_bar, np.zeros(self.m)))
        return np.row_stack((T1, T2))
    
    def pivot(self, T):
        """Update the dictionary (tableau) using the pivoting mechanism.
        
        Parameters:
            ((m+1,m+n+2) ndarray): The tableau to be updated.
            
        Raises:
            ValueError: if the given system is unbounded
        """
        for i,t in enumerate(T[0][1:]): # Search through the entries of the first row
            if t < 0: # Stop when encountering the first negative value
                R = T[1:, 0]/T[1:,i+1] # Calculate the ratios
                # Raise a ValueError if there are no positive entries in the column
                if len(R[R>0]) == 0:
                    raise ValueError("The problem is unbounded")
                # Find the row index, stopping at the first one found following Bland's rule
                p = np.min(R[R>0])
                j = np.where(R==p)[0][0]
                break
        # Update the index list
        self.L[j], self.L[self.m + i] = self.L[self.m + i], self.L[j]
        #Update the tableau
        T[j+1] /= T[j+1][i+1]
        for k in range(len(T)):
            if k != j+1:
                T[k] -= T[k,i+1]*T[j+1]
    
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The maximum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        # Initialize our tableau
        table = self.init_tableau()
        obj = table[0]
        # Repeat until there are no negative value in the objective function
        while len(obj[obj<0]) != 0:
            # Update the tableau using the pivot function
            self.pivot(table)
            obj = table[0]
        # Find the optimum value, and the values for the basic and non basic variables
        optimum = table[0,0]
        basic = dict()
        nonbasic = dict()
        for i,v in enumerate(self.L[:self.m]):
            basic[v] = table[i+1,0]
        for v in self.L[self.m:]:
            nonbasic[v] = 0
        return optimum, basic, nonbasic


# Problem 7
def prob7(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        The minimizer of the problem (as an array).
    """
    # Load the data and extract the arrays
    data = np.load(filename)
    A = data['A']
    p = data['p']
    m = data['m']
    d = data['d']
    n = len(p)
    # Combine the contraints so we have Mx <= b
    M = np.row_stack((A, np.eye(n)))
    b = np.concatenate((m, d))
    # Use our SimplexSolver to find the minimizers
    S = SimplexSolver(p, M, b)
    opt, d1, d2 = S.solve()
    # Extract the minimizers from the basic dictionary
    minimizers = []
    for i in range(n):
        minimizers.append(d1[i])
    return minimizers

c = np.array([3,2])
b = np.array([2,5,7])
A = np.array([[1,-1],[3,1],[4,3]])
S = SimplexSolver(c, A, b)
print(S.solve())
prob7()