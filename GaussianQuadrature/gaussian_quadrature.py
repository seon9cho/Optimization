# solutions.py
"""Volume 2: Gaussian Quadrature.
<Name>
<Class>
<Date>
"""
import numpy as np
import scipy.linalg as la
from scipy.integrate import quad
from scipy.integrate import nquad
from scipy.stats import norm
import matplotlib.pyplot as plt

class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        if polytype == "legendre":
            self.w_inv = lambda x: 1
        elif polytype == "chebyshev":
            self.w_inv = lambda x: np.sqrt(1 - x**2)
        else:
            raise ValueError("Polynomial type must be either legendre or chebyshev.")
        self.polytype = polytype
        self.points, self.weights = self.points_weights(n)

    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        B = None
        mu = 1
        if self.polytype == "chebyshev":
            B = [1/4 for i in range(n-1)]
            B[0] = 1/2
            mu = np.pi
        elif self.polytype == "legendre":
            B = [k**2/(4*k**2 - 1) for k in range(1, n)]
            mu = 2
        J1 = np.diag(np.sqrt(B), k=1)
        J2 = np.diag(np.sqrt(B), k=-1)
        J = J1 + J2
        x, eigvects = la.eig(J)
        w = [mu * v**2 for v in eigvects[0]]
        return np.real(x), np.real(w)


    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        g = f(self.points) * self.w_inv(self.points)
        return np.dot(g, self.weights)

    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        h = f((b-a)/2 * self.points + (a+b)/2)
        g = h * self.w_inv(self.points)
        return np.dot(g, self.weights) * (b-a)/2

    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        h = np.array([f((b1-a1)/2 * self.points + (a1+b1)/2, \
                        (b2-a2)/2 * y + (a2+b2)/2) for y in self.points])
        g = h.T * self.w_inv(self.points)
        g = (g.T * self.w_inv(self.points)).T
        return (b1-a1)*(b2-a2)/4*np.dot(self.weights, g@self.weights)

# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    f = lambda x: 1/np.sqrt(2*np.pi) * np.exp((-x**2)/2)
    F = norm.cdf(2) - norm.cdf(-3)
    N = [5 * i for i in range(1, 11)]
    err_l = []
    err_c = []
    for n in N:
        Q1 = GaussianQuadrature(n, polytype="legendre")
        err_l.append(abs(F - Q1.integrate(f, -3, 2)))
        Q2 = GaussianQuadrature(n, polytype="chebyshev")
        err_c.append(abs(F - Q2.integrate(f, -3, 2)))
    err_sp = np.ones_like(N) * abs(F - quad(f, -3, 2)[0])
    plt.ion()
    plt.plot(N, err_l, label="Legendre")
    plt.plot(N, err_c, label="Chebyshev")
    plt.plot(N, err_sp, label="scipy")
    plt.legend()
    plt.xlabel("n")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.show()


def test():
    Q = GaussianQuadrature(30, polytype="chebyshev")
    f = lambda x, y: x**2 * y**2
    return nquad(f, [[-1,2], [-3, 4]])[0], Q.integrate2d(f, -1, 2, -3, 4)

