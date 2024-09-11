# solutions.py
"""Volume 2: Newton and Quasi-Newton Methods.
<Name>
<Class>
<Date>
"""
import sys
import numpy as np
import scipy.linalg as la
from scipy.optimize import *
import matplotlib.pyplot as plt
import time

# Problem 1
def newton(Df, D2f, x0, maxiter=20, tol=1e-5):
    """Use Newton's method to minimize a function f:R^n -> R.

    Parameters:
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        D2f (function): The second dirivative (Hessian) of f. Accepts a NumPy
            array of shape (n,) and returns a NumPy array of shape (n,n).
        x0 ((n,) ndarray): The initial guess.
        maxiter (int): The maximum number of iterations to compute.
        tol (float): The stopping tolerance.

    Returns:
        ((n,) ndarray): The approximate optimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # Keep track of convergence and number of iterations
    converged = False
    iterations = 0
    # Loop maxiter number of times before convergence 
    for i in range(maxiter):
        iterations += 1
        # Store Df(x0)
        dfx = Df(x0)
        # Equation 12.4, find D2f-1@Df using la.solve
        x1 = x0 - la.solve(D2f(x0), dfx)
        #Condition for convergence
        if la.norm(dfx) < tol:
            converged = True
            break
        # Update x0
        x0 = x1

    return x1, converged, iterations


# Problem 2
def bfgs(Df, x0, maxiter=80, tol=1e-5):
    """Use BFGS to minimize a function f:R^n -> R.

    Parameters:
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        maxiter (int): The maximum number of iterations to compute.
        tol (float): The stopping tolerance.

    Returns:
        ((n,) ndarray): The approximate optimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # Keep track of convergence and number of iterations
    converged = False
    iterations = 0
    n = len(x0)
    A_inv = np.eye(n)
    dfx = Df(x0)
    # Loop maxiter number of times before convergence 
    for i in range(maxiter):
        iterations += 1
        # Equation 12.6
        x1 = x0 - A_inv@dfx
        # Condition for convergence
        if la.norm(dfx) < tol:
            converged = True
            break
        # Store all necessary values for efficiency
        s = x1 - x0
        dfx1 = Df(x1)
        y = dfx1 - dfx
        dfx = dfx1
        sy = np.dot(s,y)
        # Avoid dividing by 0
        if sy == 0:
            break
        # Equation 12.7
        A_inv += ((np.dot(s,y) + np.dot(y, A_inv@y))*np.outer(s,s))/(sy**2) \
                 - (A_inv@np.outer(y,s) + np.outer(s,y)@A_inv)/sy
        # Update x0
        x0 = x1

    return x1, converged, iterations


# Problem 3
def prob3(N=100):
    """Compare newton(), bfgs(), and scipy.optimize.fmin_bfgs() by repeating
    the following N times.
        1. Sample a random initial guess x0 from the 2-D uniform distribution
            over [-3,3]x[-3,3].
        2. Time (separately) newton(), bfgs(), and scipy.optimize.bfgs_fmin()
            for minimizing the Rosenbrock function with an initial guess of x0.
        3. Record the number of iterations from each method.
    Plot the computation times versus the number of iterations with a log-log
    scale, using different colors for each method.
    """
    # Initialize all lists to store necessary values
    newton_x = []
    newton_y = []
    bfgs_x = []
    bfgs_y = []
    scipy_x = []
    scipy_y = []

    for i in range(N):
        # Store the time and number of iterations of each method
        x0 = np.random.uniform(-3,3,2)
        t1 = time.time()
        i1 = newton(rosen_der, rosen_hess, x0)[2]
        t2 = time.time()
        i2 = bfgs(rosen_der, x0, maxiter=150)[2]
        t3 = time.time()
        i3 = fmin_bfgs(rosen, x0, disp=False, retall=True)[1]
        t4 = time.time()
        # Append the above values in their respective list
        newton_x.append(t2-t1)
        newton_y.append(i1)
        bfgs_x.append(t3-t2)
        bfgs_y.append(i2)
        scipy_x.append(t4-t3)
        scipy_y.append(len(i3))
    # Plot the resulting lists
    plt.scatter(newton_x, newton_y, alpha=0.5, label="Newton's method")
    plt.scatter(bfgs_x, bfgs_y, alpha=0.5, label="BFGS")
    plt.scatter(scipy_x, scipy_y, alpha=0.5, label="scipy.optimize.fmin_bfgs()")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(5e-4, 1e-1)
    plt.ylim(0, 1e3)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Iterations")
    plt.legend()
    plt.show()

# Problem 4
def gauss_newton(J, r, x0, maxiter=10, tol=1e-5):
    """Solve a nonlinear least squares problem with the Gauss-Newton method.

    Parameters:
        J (function): Jacobian of the residual function. Accepts a NumPy array
            of shape (n,) and returns a NumPy array of shape (m,n).
        r (function): Residual vector function. Accepts a NumPy array of shape
            (n,) and returns an array of shape (m,).
        x0 ((n,) ndarray): The initial guess.
        maxiter (int): The maximum number of iterations to compute.
        tol (float): The stopping tolerance.

    Returns:
        ((n,) ndarray): The approximate optimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # Keep track of convergence and number of iterations
    converged = False
    iterations = 0
    # Loop maxiter number of times before convergence 
    for i in range(maxiter):
        iterations += 1
        # Store J.T@r and J.T@J for efficiency
        Jr = J(x0).T @ r(x0)
        JJ = J(x0).T @ J(x0)
        # Equation 12.8
        x1 = x0 - la.solve(JJ, Jr)
        # Condition for convergence
        if la.norm(Jr) < tol:
            converged = True
            break
        # Update x0
        x0 = x1

    return x1, converged, iterations


# Problem 5
def prob5(filename="population.npy"):
    """Load the data from the given file. Fit the data to an exponential model

        phi(x1, x2, x3, t) = x1 exp(x2(t + x3))

    and to a logistic model

        phi(x1, x2, x3, t) = x1 / (1 + exp(-x2(t + x3))).

    Plot the resulting curves along with the data points.
    """
    # Load the data and split the attributes
    data = np.load(filename)
    T = data[:,0]
    y = data[:,1]
    # Exponential model, the residual, and its Jacobian
    x0 = np.array([1.5, .4, 2.5])
    phi = lambda x, t: x[0] * np.exp(x[1]*(t + x[2]))
    residual = lambda x: phi(x, T) - y
    jac = lambda x: np.column_stack((np.exp(x[1]*(T+x[2])), \
                                    x[0]*(T+x[2])*np.exp(x[1]*(T+x[2])), \
                                    x[0]*x[1]*np.exp(x[1]*(T+x[2]))))
    m2 = leastsq(func=residual, x0=x0, Dfun=jac)
    # Logistic model, the residual, and its Jacobian
    x1 = np.array([150, .4, -15])
    phi2 = lambda x, t: x[0] / (1 + np.exp(-x[1]*(t + x[2])))
    res2 = lambda x: phi2(x, T) - y
    jac2 = lambda x: np.column_stack((1 / (1 + np.exp(-x[1]*(T + x[2]))), \
                                     x[0]*(T+x[2])*np.exp(-x[1]*(T+x[2]))*(1+np.exp(-x[1]*(T+x[2])))**(-2) , \
                                     x[0]*x[1]*np.exp(-x[1]*(T+x[2]))*(1+np.exp(-x[1]*(T+x[2])))**(-2)))
    m4 = leastsq(func=res2, x0=x1, Dfun=jac2)
    # Plot the data and the models together
    dom = np.linspace(0, 15, 200)
    plt.plot(T, y, '*', label="Data points")
    plt.plot(dom, phi(m2[0], dom), label="Exponential model")
    plt.plot(dom, phi2(m4[0], dom), label="Logistic model")
    plt.xlabel("Time (decade)")
    plt.ylabel("Population (millions)")
    plt.legend()
    plt.show()

# Test cases
def main(prob):
    # Problem 1
    if prob == "prob1":
        x0 = (-2,2)
        # Solve for the minimizer using problem 1 and scipy
        m1 = newton(rosen_der, rosen_hess, x0)
        m2 = fmin_bfgs(rosen, x0, disp=False, retall=True)
        # Print the results
        print(m1)
        print(m2[0], len(m2[1]))
        # Converges at i = 6

    # Problem 2
    elif prob == "prob2":
        x0 = (-2,2)
        # Solve for the minimizer using problem 2
        m = bfgs(rosen_der, x0, maxiter=150)
        # Print the results
        print(m)
        # Converges at i = 108

    # Problem 3
    elif prob == "prob3":
        prob3()

    # Problem 4
    elif prob == "prob4":
        x0 = np.array([2.5, .6])
        T = np.arange(10)
        y = 3*np.sin(0.5*T) + 0.5*np.random.randn(10)
        # Model function, residual, and its Jacobian
        model = lambda x, t: x[0]*np.sin(x[1]*t)
        residual = lambda x: model(x, T) - y
        jac = lambda x: np.column_stack((np.sin(x[1]*T), x[0]*T*np.cos(x[1]*T)))
        # Solve for the least square using problem 4 and scipy
        minx = leastsq(func=residual, x0=x0, Dfun=jac)
        m1 = gauss_newton(jac, residual, x0, maxiter=10, tol=1e-3)
        # Plot the data points, the model, and the least square
        dom = np.linspace(0, 10, 200)
        plt.plot(T, y, '*', label="Data points")
        plt.plot(dom, 3*np.sin(.5*dom), '--', label="Data-generating curve")
        plt.plot(dom, model(m1[0], dom), label="Fitted model")
        plt.xlabel("Time")
        plt.ylabel("Output")
        plt.legend()
        plt.show()
        # Print the results
        print(m1)
        print(minx)

    # Problem 5
    elif prob == "prob5":
        prob5()

if __name__ == "__main__":
    plt.ion()
    main(sys.argv[1])
