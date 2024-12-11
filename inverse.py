import numpy as np
import torch
from convection_diffusion import convection_diffusion



e_num = 32
import numpy as np

nu = np.ones(e_num + 1)
c = np.ones(e_num + 1)
nu[int(e_num/2):] = 0.0*np.ones(int(e_num/2)+1)
u_l = 0.0
u_r = 0.0
xx = np.linspace(0, 1, e_num + 1)
u0 = np.sin(np.pi * xx)

res = convection_diffusion( e_num, nu,c,u_l,u_r,u0 )
qq = 0

# starting inverse problem solution
lf = 1e6*torch.ones(1)



e_num = 32
import numpy as np

nu = np.ones(e_num + 1)
c = np.ones(e_num + 1)
nu[int(e_num/2):] = 0.0*np.ones(int(e_num/2)+1)
u_l = 0.0
u_r = 0.0
xx = np.linspace(0, 1, e_num + 1)
u0 = np.sin(np.pi * xx)

res = convection_diffusion( e_num, nu,c,u_l,u_r,u0 )

def rosen(x):
    """The Rosenbrock function"""
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0, axis=0)

def conv_diff(x):
   N = int(x.shape[0]/2)
   A = x[:N]
   B = x[N:]
   A = np.abs(A)
   r = convection_diffusion(e_num, A, B, u_l, u_r, u0)

   eps = np.max(np.abs(res-r))
   print(eps)
   return eps


if __name__ == '__main__':
    from scipy.optimize import minimize

    x0 = np.random.random(2*(e_num+1))
    N = int(x0.shape[0] / 2)
    A = x0[:N]
    B = x0[N:]

    res = minimize(conv_diff, x0, method='nelder-mead',
        options={'xtol': 1e-8, 'disp': True})
    print(res.x)


