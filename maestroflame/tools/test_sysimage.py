
import os


from julia.api import LibJulia
api = LibJulia.load()
#api.sysimage = "sys_plots.so"
api.sysimage = "sys.so"
api.init_julia()

#from julia import Base
from julia import Plots

import numpy as np


x = np.zeros((5,5))
y = np.zeros_like(x)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        x[i,j] = i
        y[i,j] = j

Plots.scatter(x,y, title="A Julia plot made with numpy data run from python!", dpi=600)
Plots.savefig('test.png')


print("Success! Created plot: test.png")
os.system("open test.png")
