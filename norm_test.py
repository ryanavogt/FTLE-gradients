import torch
from torch.linalg import norm
import matplotlib.pyplot as plt

max_exp = 20
norm_list = []
norm_list_B = []
exp_list = torch.arange(max_exp)
for i in exp_list:
    print(i)
    A = torch.randn((int(10**(i/5)), int(10**(i/5))))
    B = torch.randn((10**4, int(10**(i/5))))
    norm_list.append(norm(A, ord = 2))
    norm_list_B.append(norm(B, ord = 2))
plt.scatter(exp_list/5, norm_list, label = 'Square 2-norm')
plt.scatter(exp_list/5, norm_list_B, label = 'Rect 2-norm')
plt.plot(exp_list/5, 2*torch.sqrt(10**(exp_list/5)), 'r', label = r'$2*\sqrt{10^{i}}$')
plt.plot(exp_list/5, torch.sqrt(10**(exp_list/5))+100, 'y', label = r'$\sqrt{10^{i}}+100$')
plt.xlabel('Exponent i')
plt.legend()
plt.savefig('SMNIST/Plots/norm_scale.png', dpi = 400, bbox_inches = 'tight')
print(norm_list)