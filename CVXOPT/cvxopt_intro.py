# cvxopt_intro.py
"""Volume 2: Intro to CVXOPT.
<Name>
<Class>
<Date>
"""
import cvxopt as cvx
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x + 2y          >= 3
                    2x + 10y + 3z   >= 10
                    x               >= 0
                    y               >= 0
                    z               >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    c = cvx.matrix([2.,1.,3.])
    G = cvx.matrix([[-1.,-2.,-1.,0.,0.],[-2.,-10.,0.,-1.,0.],[0.,-3.,0.,0.,-1.]])
    h = cvx.matrix([-3.,-10.,0.,0.,0.])
    sol = cvx.solvers.lp(c, G, h)
    return np.ravel(sol['x']), sol['primal objective']

# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray), without any slack variables u
        The optimal value (sol['primal objective'])
    """
    m = len(A)
    n = len(A[0])
    c = cvx.matrix(np.concatenate([np.ones(n), np.zeros(n)]))
    iden = np.eye(n)
    r1 = np.column_stack([-iden, iden])
    r2 = np.column_stack([-iden, -iden])
    G = cvx.matrix(np.row_stack([r1, r2]))
    h = cvx.matrix(np.zeros(2*n))
    Z = np.zeros((m,n))
    Am = cvx.matrix(np.column_stack([Z, A]))
    print(c)
    print(G)
    print(h)
    print(Am)
    print(b)
    sol = cvx.solvers.lp(c,G,h,Am,cvx.matrix(b))
    return np.ravel(sol['x'][n:]), sol['primal objective']

# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    c = cvx.matrix([4.,7.,6.,8.,8.,9.])
    g1 = -1*np.eye(6)
    g2 = np.array([0.,1.,0.,1.,0.,1.])
    g = np.row_stack([g1, g2, -g2])
    G = cvx.matrix(g)
    h = cvx.matrix(np.concatenate([np.zeros(6), [8.,-8.]]))
    A = cvx.matrix(np.array([[1.,1.,0.,0.,0.,0.],
                             [0.,0.,1.,1.,0.,0.],
                             [0.,0.,0.,0.,1.,1.],
                             [1.,0.,1.,0.,1.,0.]]))
    b = cvx.matrix([7.,2.,4.,5.])
    sol = cvx.solvers.lp(c,G,h,A,b)
    return np.ravel(sol['x']), sol['primal objective']

# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    P = cvx.matrix(np.array([[3.,2.,1.],[2.,4.,2.],[1.,2.,3.]]))
    q = cvx.matrix([3.,0.,1.])
    sol = cvx.solvers.qp(P,q)
    return np.ravel(sol['x']), sol['primal objective']

# Problem 5
def l2Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_2
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    m = len(A)
    n = len(A[0])
    print(m,n)
    P = cvx.matrix(2*np.eye(n))
    q = cvx.matrix(np.zeros(n))
    A = cvx.matrix(A)
    b = cvx.matrix(b)
    sol = cvx.solvers.qp(P,q, A=A, b=b)
    return np.ravel(sol['x']), sol['primal objective']

# Problem 6
def prob6():
    """Solve the allocation model problem in 'ForestData.npy'.
    Note that the first three rows of the data correspond to the first
    analysis area, the second group of three rows correspond to the second
    analysis area, and so on.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective']*-1000)
    """
    data = np.load('ForestData.npy')
    s = data[:,1]
    b = cvx.matrix(s[::3].astype(np.float))
    p = data[:,3]
    c = cvx.matrix(-p)
    t = data[:,4]
    g = data[:,5]
    w = data[:,6]
    
    G1 = np.row_stack([t,g,w])
    h1 = np.array([40000., 5., 55160.])
    G2 = np.eye(21).astype(np.float)
    h2 = np.zeros(21).astype(np.float)
    G = cvx.matrix(np.row_stack([-G1, -G2]))
    h = cvx.matrix(np.concatenate([-h1, -h2]))
    R = [np.concatenate([np.array([0]*3*i), np.ones(3), np.array([0]*3*(6-i))]) for i in range(7)]
    A = cvx.matrix(np.row_stack(R))
    sol = cvx.solvers.lp(c, G, h, A, b)
    return np.ravel(sol['x']), sol['primal objective']*-1000

p,q = prob6()
print(p,q)