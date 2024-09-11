# interior_point_quadratic.py
"""Volume 2: Interior Point for Quadratic Programs.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse import spdiags
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import cvxopt as opt

def startingPoint(G, c, A, b, guess):
    """
    Obtain an appropriate initial point for solving the QP
    .5 x^T Gx + x^T c s.t. Ax >= b.
    Parameters:
        G -- symmetric positive semidefinite matrix shape (n,n)
        c -- array of length n
        A -- constraint matrix shape (m,n)
        b -- array of length m
        guess -- a tuple of arrays (x, y, mu) of lengths n, m, and m, resp.
    Returns:
        a tuple of arrays (x0, y0, l0) of lengths n, m, and m, resp.
    """
    m,n = A.shape
    x0, y0, l0 = guess

    # Initialize linear system
    N = np.zeros((n+m+m, n+m+m))
    N[:n,:n] = G
    N[:n, n+m:] = -A.T
    N[n:n+m, :n] = A
    N[n:n+m, n:n+m] = -np.eye(m)
    N[n+m:, n:n+m] = np.diag(l0)
    N[n+m:, n+m:] = np.diag(y0)
    rhs = np.empty(n+m+m)
    rhs[:n] = -(G.dot(x0) - A.T.dot(l0)+c)
    rhs[n:n+m] = -(A.dot(x0) - y0 - b)
    rhs[n+m:] = -(y0*l0)

    sol = la.solve(N, rhs)
    dx = sol[:n]
    dy = sol[n:n+m]
    dl = sol[n+m:]

    y0 = np.maximum(1, np.abs(y0 + dy))
    l0 = np.maximum(1, np.abs(l0+dl))

    return x0, y0, l0


# Problems 1-2
def qInteriorPoint(Q, c, A, b, guess, niter=20, tol=1e-16, verbose=False):
    """Solve the Quadratic program min .5 x^T Q x +  c^T x, Ax >= b
    using an Interior Point method.

    Parameters:
        Q ((n,n) ndarray): Positive semidefinite objective matrix.
        c ((n, ) ndarray): linear objective vector.
        A ((m,n) ndarray): Inequality constraint matrix.
        b ((m, ) ndarray): Inequality constraint vector.
        guess (3-tuple of arrays of lengths n, m, and m): Initial guesses for
            the solution x and lagrange multipliers y and eta, respectively.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    m = len(b)
    n = len(c)
    # Define the function that returns matrix F
    def KKT_F(x, y, mu):
        F_r1 = Q@x - A.T@mu + c
        F_r2 = A@x - y - b
        F_r3 = y*mu
        return np.concatenate([F_r1,F_r2,F_r3])
    # Save the first and second block rows of DF since they don't change
    DF_r1 = np.column_stack([Q, np.zeros((n,m)), -A.T])
    DF_r2 = np.column_stack([A, -np.eye(m), np.zeros((m,m))])
    # Define the function that returns the search direction
    def searchDirection(x, y, mu, s=0.1):
        F = KKT_F(x, y, mu)
        DF_r3 = np.column_stack([np.zeros((m,n)), np.diag(mu), np.diag(y)])
        DF = np.row_stack([DF_r1, DF_r2, DF_r3])
        nu = np.dot(y,mu)/m
        p = np.concatenate([np.zeros(m+n), s*nu*np.ones(m)])
        return la.lu_solve(la.lu_factor(DF), -F + p)
    # Defind the function that returns the stepsize, along with the delta vector
    def stepSize(x, y, mu):
        direction = searchDirection(x, y, mu)
        d_x = direction[:n]
        d_y = direction[n:n+m]
        d_mu = direction[n+m:]
        beta = np.min([1, np.min((-mu/d_mu)[d_mu<0])])
        delta = np.min([1, np.min((-y/d_y)[d_y<0])])
        return np.min([0.95*beta, 0.95*delta]), d_x, d_y, d_mu
    # Use the predefined function startingPoint to get the initial point
    x, y, mu = startingPoint(Q, c, A, b, guess)
    # Repeat the following for niter times
    for i in range(niter):
        alpha, d_x, d_y, d_mu = stepSize(x, y, mu)
        # Update each of the variables
        x += alpha*d_x
        y += alpha*d_y
        mu += alpha*d_mu
        nu = np.dot(y,mu) / m
        # Stopping criteria 
        if abs(nu) < tol:
            return x, (1/2)*np.dot(x,Q@x) + np.dot(c,x)
    return x, (1/2)*np.dot(x,Q@x) + np.dot(c,x)

def laplacian(n):
    """Construct the discrete Dirichlet energy matrix H for an n x n grid."""
    data = -1*np.ones((5, n**2))
    data[2,:] = 4
    data[1, n-1::n] = 0
    data[3, ::n] = 0
    diags = np.array([-n, -1, 0, 1, n])
    return spdiags(data, diags, n**2, n**2).toarray()


# Problem 3
def circus(n=15):
    """Solve the circus tent problem for grid size length 'n'.
    Display the resulting figure.
    """
    # Initialize L
    L = np.zeros((n,n))
    L[n//2-1:n//2+1,n//2-1:n//2+1] = 0.5
    m = [n//6-1, n//6, int(5*(n/6.))-1, int(5*(n/6.))]
    mask1, mask2 = np.meshgrid(m,m)
    L[mask1, mask2] = .3
    L = L.ravel()
    # Initialize H, c, A
    H = laplacian(n)
    c = -(n-1)**(-2) * np.ones(n**2)
    A = np.eye(n**2)
    # Initial guesses
    x = np.ones((n,n)).ravel()
    y = np.ones(n**2)
    mu = np.ones(n**2)
    # Solve, then plot the function
    z = qInteriorPoint(H, c, A, L, (x,y,mu))[0].reshape((n,n))
    domain = np.arange(n)
    X, Y = np.meshgrid(domain, domain)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, z, rstride=1, cstride=1, color='r')
    plt.show()

# Problem 4
def portfolio(filename="portfolio.txt"):
    """Markowitz Portfolio Optimization

    Parameters:
        filename (str): The name of the portfolio data file.

    Returns:
        (ndarray) The optimal portfolio with short selling.
        (ndarray) The optimal portfolio without short selling.
    """
    n = 8
    R = 1.13
    # Load data
    data = np.loadtxt(filename)
    # Estimate Q and mu
    Q = opt.matrix(np.cov(data.T[1:]))
    p = opt.matrix(np.zeros(n))
    mu = np.mean(data[:,1:], axis=0)
    # Define the contraints
    G = opt.matrix(np.eye(n))
    h = opt.matrix(np.zeros(n))
    A = opt.matrix(np.stack([np.ones(n), mu]))
    b = opt.matrix(np.array([1, R]))
    # With shortselling
    sol1 = opt.solvers.qp(Q, p, A=A, b=b)
    # Without shortselling
    sol2 = opt.solvers.qp(Q, p, -G, h, A, b)
    
    return np.ravel(sol1['x']), np.ravel(sol2['x'])
    
Q = np.array([[1,-1],[-1,2]])
c = np.array([-2,-6])
A = np.array([[-1,-1],[1,-2],[-2,-1],[1,0],[0,1]])
b = np.array([-2,-2,-3,0,0])
guess = np.array([[0.5, 0.5], np.ones(5), np.ones(5)])
print(qInteriorPoint(Q, c, A, b, guess))
print(portfolio())
