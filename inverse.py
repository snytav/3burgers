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

A = torch.abs(torch.randn(e_num+1))
A.requires_grad=True
B = torch.randn(e_num+1,requires_grad=True)
optim = torch.optim.SGD([A,B],lr=0.01)
while lf.item() > 1e0:
   optim.zero_grad()
   A = torch.abs(A)
   r = convection_diffusion( e_num, A.detach().numpy(),B.detach().numpy(),u_l,u_r,u0)
   lf = torch.max(torch.abs(torch.from_numpy(r)-torch.from_numpy(res)))
   lf.backward()
   optim.step()
qq = 0


