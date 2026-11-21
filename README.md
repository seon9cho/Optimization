# Optimization

My solutions to the optimization and quadrature labs from BYU's Applied and Computational Mathematics Emphasis (ACME), drawn from *Volume 1* and *Volume 2* of the lab manuals. Each lab is a notebook that pairs the spec's problem statements with my solutions and their output, plus condensed notes on the concepts the lab covers.

| Lab | Volume | Notebook |
|-----|--------|----------|
| Newton's Method | Vol. 1, Lab 9 | Not completed |
| Gaussian Quadrature | Vol. 2, Lab 10 | [GaussianQuadrature/gaussian_quadrature.ipynb](GaussianQuadrature/gaussian_quadrature.ipynb) |
| One-dimensional Optimization | Vol. 2, Lab 11 | [OneD_Optimization/OneD_Optimization.ipynb](OneD_Optimization/OneD_Optimization.ipynb) |
| Newton and Quasi-Newton Methods | Vol. 2, Lab 12* | [QuasiNewton/quasi_newton.ipynb](QuasiNewton/quasi_newton.ipynb) |
| Gradient Descent Methods | Vol. 2, Lab 12 | Not preserved** |
| The Simplex Method | Vol. 2, Lab 13 | [Simplex/simplex.ipynb](Simplex/simplex.ipynb) |
| CVXOPT | Vol. 2, Lab 15 | [CVXOPT/cvxopt_intro.ipynb](CVXOPT/cvxopt_intro.ipynb) |
| Interior Point 1: Linear Programs | Vol. 2, Lab 16 | [InteriorPoint_Linear/interior_point_linear.ipynb](InteriorPoint_Linear/interior_point_linear.ipynb) |
| Interior Point 2: Quadratic Programs | Vol. 2, Lab 17 | [InteriorPoint_Quadratic/interior_point_quadratic.ipynb](InteriorPoint_Quadratic/interior_point_quadratic.ipynb) |

\* Numbered from the lab's own specification, which comes from an earlier edition of Volume 2 than the other labs.<br>
\*\* I submitted this lab in 2018, but my solution file wasn't kept.

Each folder also keeps the grader feedback from my original submissions (`*_feedback.txt`).

## Running the notebooks

Each notebook runs from its own folder. The data each lab uses sits next to its notebook. The notebooks need NumPy, SciPy and Matplotlib. The CVXOPT and Interior Point 2 labs also need [CVXOPT](https://cvxopt.org/) (`pip install cvxopt`), and One-dimensional Optimization needs [autograd](https://github.com/HIPS/autograd) (`pip install autograd`).

## Credits

The lab specifications come from the [Foundations of Applied Mathematics](https://foundations-of-applied-mathematics.github.io/) lab manuals, © Dr. J. Humpherys, licensed under [CC BY 3.0 US](http://creativecommons.org/licenses/by/3.0/us/). The data files come from the original lab materials. Problem statements are quoted from the specs; the solutions are my own.
