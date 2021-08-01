import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def scaling_func(x, f, barrier1, barrier2, scaling, smallest_num=torch.tensor([1e-30])):

    bump1 = f(barrier1) - f(scaling*torch.log(barrier1)) #makes left barrier continuous
    bump2 = f(barrier2) - f(scaling*torch.log(barrier2)) #makes right barrier continuous


    y1 = torch.heaviside(x - barrier1, torch.tensor([0.]))
    y2 = -torch.heaviside(x - barrier2, torch.tensor([0.])) + 1
    y3 = -torch.heaviside(x - barrier1, torch.tensor([0.])) + 1
    y4 = torch.heaviside(x - barrier2, torch.tensor([0.]))

    A = y1*y2
    B = np.abs(y3-y4)
    C = y3
    D = y4

    #asserts everything is above zero.
    #we can't really know the smallest loss function value before we do ML.
    #But the default (1e-30) is defintley below it.
    bump3 = f(scaling*torch.log(smallest_num)+bump1)

    return (C*f(scaling*torch.log(x)+bump1) + f(x)*A + D*f(scaling*torch.log(x)+bump2))-bump3


x = torch.logspace(-30, 32)

def f(x):
    return x


b1 = torch.tensor([.1])
b2 = torch.tensor([1.])
scaling = torch.tensor([.01])

y = scaling_func(x, f, b1, b2, scaling)

plt.figure()
#plt.plot(x, (C*f(scaling*torch.log(x)+bump1) + f(x)*A + D*f(scaling*torch.log(x)+bump2))-bump3)
plt.plot(x,y)
plt.xscale('log')
plt.xlabel("Loss Function Output (True Value)")
plt.ylabel("Loss Function Scaled Value")
plt.title("Scaling for rates loss function")
plt.xlim([1e-30, 1e30])


txt = " - Monotonically decreasing \n \
- Same scale $10^{-30}$ - $10^{30}$ \n \
- Continuous "

plt.text(1e-25, 2, txt, size=10, rotation=0.,
         ha="left", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
plt.show()