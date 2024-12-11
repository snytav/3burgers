import numpy as np

def rosen(x):
    """The Rosenbrock function"""
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0, axis=0)


if __name__ == '__main__':
    from scipy.optimize import minimize
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    res = minimize(rosen, x0, method='nelder-mead',
        options={'xtol': 1e-8, 'disp': True})
    print(res.x)