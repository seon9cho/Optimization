# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt


# Auxiliary Functions ---------------------------------------------------------
def startingPoint(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(m):
    """Generate a 'square' linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add slack variables.
    Parameters:
        m -- positive integer: the number of desired constraints
             and the dimension of space in which to optimize.
    Returns:
        A -- array of shape (m,n).
        b -- array of shape (m,).
        c -- array of shape (n,).
        x -- the solution to the LP.
    """
    n = m
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(n)*10
    b = A.dot(x)
    c = A.sum(axis=0)/float(n)
    return A, b, -c, x

# This random linear program generator is more general than the first.
def randomLP2(m,n):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        m -- positive integer >= n, number of desired constraints
        n -- dimension of space in which to optimize
    Returns:
        A -- array of shape (m,n+m)
        b -- array of shape (m,)
        c -- array of shape (n+m,), with m trailing 0s
        v -- the solution to the LP
    """
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    v = np.random.random(n)*10
    k = n
    b = np.zeros(m)
    b[:k] = A[:k,:].dot(v)
    b[k:] = A[k:,:].dot(v) + np.random.random(m-k)*10
    c = np.zeros(n+m)
    c[:n] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(m)))
    return A, b, -c, v


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    # Define m,n
    m = len(b)
    n = len(c)
    # Define the function that returns matrix F
    def KKT_F(x, l, mu):
        F_r1 = A.T@l + mu - c
        F_r2 = A@x - b
        F_r3 = mu*x
        return np.concatenate([F_r1,F_r2,F_r3])
    # Save the first and second block rows of DF since they don't change
    DF_r1 = np.column_stack([np.zeros((n,n)), A.T, np.eye(n)])
    DF_r2 = np.column_stack([A, np.zeros((m,m)), np.zeros((m,n))])
    # Define the function that returns the search direction
    def searchDirection(x, l, mu, s=0.1):
        F = KKT_F(x, l, mu)
        DF_r3 = np.column_stack([np.diag(mu), np.zeros((n,m)), np.diag(x)])
        DF = np.row_stack([DF_r1, DF_r2, DF_r3])
        nu = np.dot(x,mu)/n
        p = np.concatenate([np.zeros(m+n), s*nu*np.ones(n)])
        return la.lu_solve(la.lu_factor(DF), -F + p)
    # Defind the function that returns the stepsize, along with the delta vector
    def stepSize(x, l, mu):
        direction = searchDirection(x, l, mu)
        d_x = direction[:n]
        d_l = direction[n:n+m]
        d_mu = direction[n+m:]
        alpha = np.min([1, np.min((-mu/d_mu)[d_mu<0])])
        delta = np.min([1, np.min((-x/d_x)[d_x<0])])
        return 0.95*alpha, 0.95*delta, d_x, d_l, d_mu
    # Use the predefined function startingPoint to get the initial point
    x, l, mu = startingPoint(A, b, c)
    # Repeat the following for niter times
    for i in range(niter):
        alpha, delta, d_x, d_l, d_mu = stepSize(x, l, mu)
        # Update each of the variables
        x += delta*d_x
        l += alpha*d_l
        mu += alpha*d_mu
        nu = np.dot(x,mu) / n
        # Stopping criteria 
        if abs(nu) < tol:
            return x, np.dot(c,x)
    return x, np.dot(c,x)

def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    # Load the data
    data = np.loadtxt(filename)
    # Define m,n,c,y,x
    m = data.shape[0]
    n = data.shape[1] - 1
    c = np.zeros(3*m + 2*(n+1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:,0]
    y[1::2] = data[:,0]
    x = data[:,1:]
    # Define the matrix A
    A = np.ones((2*m, 3*m + 2*(n+1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)
    # Solve for the system
    sol = interiorPoint(A, y, c, niter=10)[0]
    # Extract the necessary data beta and b
    beta = sol[m:m+n] - sol[m+n:m+2*n]
    b = sol[m+2*n] - sol[m+2*n+1]
    # Necessary data for least square line obtained by the scipy function
    slope, intercept = linregress(data[:,1], data[:,0])[:2]
    domain = np.linspace(0, 10, 200)
    # Plot the results
    plt.plot(domain, domain*beta + b, 'r', label='LAD')
    plt.plot(domain, domain*slope + intercept, 'm', label='least square')
    plt.scatter(data[:,1], data[:,0], label='data points')
    plt.legend()
    plt.show()

def randomLP(m, n):
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(n)*10
    b = np.zeros(m)
    b[:n] = A[:n,:] @ x
    b[n:] = A[n:,:] @ x + np.random.random(m-n)*10
    c = np.zeros(n+m)
    c[:n] = A[:n,:].sum(axis=0)/n
    A = np.hstack((A, np.eye(m)))
    return A, b, -c, x
m, n = 7, 5
A, b, c, x = randomLP(m, n)
point, value = interiorPoint(A, b, c)
print(np.allclose(x, point[:n]))

leastAbsoluteDeviations()
