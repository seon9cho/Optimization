# solutions.py
"""Volume 2: One-Dimensional Optimization.
<Name>
<Class>
<Date>
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from autograd import numpy as anp
from autograd import grad

# Problem 1
def golden_section(f, a, b, tol=1e-5, maxiters=15):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiters (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # Algorithm 11.1
    x0 = (a + b) / 2
    r = (1 + np.sqrt(5))/2
    # Keep track of convergence and the number of iterations
    converged = False
    iterations = 0
    for i in range(maxiters):
        iterations += 1
        c = (b - a) / r
        a_1 = b - c
        b_1 = a + c
        if f(a_1) <= f(b_1):
            b = b_1
        else:
            a = a_1
        x1 = (a + b) / 2
        if abs(x0 - x1) < tol:
            converged = True
            break
        x0 = x1

    return x1, converged, iterations


# Problem 2
def newton1d(df, d2f, x0, tol=1e-5, maxiters=15):
    """Use Newton's method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiters (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    converged = False
    iterations = 0
    for i in range(maxiters):
        iterations += 1
        # Newton's method, equation 11.1
        x1 = x0 - df(x0)/d2f(x0)
        if abs(x0 - x1) < tol:
            converged = True
            break
        # Update the x value
        x0 = x1

    return x1, converged, iterations


# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiters=15):
    """Use the secant method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        x1 (float): Another guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiters (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    converged = False
    iterations = 0
    # Save df(x0) and df(x1) so that we don't need to call them every time
    dfx0 = df(x0)
    dfx1 = df(x1)
    for i in range(maxiters):
        iterations += 1
        # Secant method, equation 11.3
        x2 = (x0*dfx1 - x1*dfx0)/(dfx1 - dfx0)
        if abs(x2 - x1) < tol:
            converged = True
            break
        # Update the x values and their derivatives
        dfx0 = dfx1
        dfx1 = df(x2)
        x0 = x1
        x1 = x2

    return x2, converged, iterations

# Problem 4
def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """
    # Algorithm 11.2
    Dfp = np.dot(Df(x), p)
    fx = f(x)
    while f(x + alpha*p) > fx + c*alpha*Dfp:
        alpha = rho*alpha
    return alpha

def main():
    # Test functions for all of the problems.
    plt.ion()
    g = lambda x: np.exp(x) - 4*x
    a = 0
    b = 3
    domain = np.linspace(a, b, 200)
    # Plot prob1
    #plt.plot(domain, g(domain))
    print("Problem 1: ")
    print(golden_section(g, a, b, maxiters=100))
    print(opt.golden(g, brack=(0,3), tol=1e-5))

    dfn = lambda x: 2*x + 5*np.cos(5*x)
    d2fn = lambda x: 2 - 25*np.sin(5*x)
    print("\nProblem 2: ")
    print(newton1d(dfn, d2fn, 0, tol=1e-10, maxiters=500))
    print(opt.newton(dfn, x0=0, fprime=d2fn, tol=1e-10, maxiter=500))

    fs = lambda x: x**2 + np.sin(x) + np.sin(10*x)
    dfs = lambda x: 2*x + np.cos(x) + 10*np.cos(10*x)
    dom_s = np.linspace(-6, 0, 200)
    # Plot prob3
    plt.plot(dom_s, fs(dom_s))
    plt.grid()
    print("\nProblem 3: ")
    s = secant1d(dfs, 0, -1, tol=1e-10, maxiters=500)[0]
    n = opt.newton(dfs, x0=0, tol=1e-10, maxiter=500)
    print(s)
    print(n)
    print(fs(s), fs(n))

    fb = lambda x: x[0]**2 + x[1]**2 + x[2]**2
    Dfb = lambda x: np.array([2*x[0], 2*x[1], 2*x[2]])
    x = anp.array([150., .03, 40.])
    p = anp.array([-.5, -100., -4.5])
    phi = lambda alpha: fb(x + alpha*p)
    dphi = grad(phi)
    print("\nProblem 4: ")
    alpha, _ = opt.linesearch.scalar_search_armijo(phi, phi(0.), dphi(0.))
    print(alpha)
    print(backtracking(fb, Dfb, x, p))

if __name__ == main():
    main()
