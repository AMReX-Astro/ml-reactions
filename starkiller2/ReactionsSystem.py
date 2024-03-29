import numpy as np
import random
from StarKiller.initialization import starkiller_initialize
from StarKiller.interfaces import BurnType, EosType
from StarKiller.integration import Integrator
from StarKiller.network import Network
from StarKiller.eos import Eos

class ReactionsSystem(object):
    def __init__(self, dens, temp, end_time):
        starkiller_initialize("probin_aprox13")
        self.network = Network()
        self.integrator = Integrator()
        self.init_state(dens, temp, end_time)
        
        self.numDependent = self.network.nspec + 1
        self.numIndependent = self.network.nspec + 2
        self.ienuc = self.network.nspec
        self.idens = self.network.nspec + 1
        self.itemp = self.network.nspec + 2

    def init_state(self, dens, temp, end_time):
        # Input sampling domain & scaling
        self.dens = dens
        self.temp = temp
        self.xhe = 1.0
        
        self.end_time = end_time

    # Get the solution given time t
    # t is a vector containing the times at which
    # we want to evaluate the System solution.
    def sol(self, t, initData=False):
        num_time = len(t)
        y = np.zeros((num_time, self.numIndependent+1)) # nspec+energy+density + temperature for rhs
        
        for i, time in enumerate(t):
            # get the time
            time = time.item()
            
            # construct a burn type
            state_in = BurnType()

            # set density & temperature
            state_in.state.rho = self.dens
            state_in.state.t = self.temp

            # mass fractions
            state_in.state.xn = np.zeros(self.network.nspec)
            state_in.state.xn[:] = (1.0-self.xhe)/(self.network.nspec-1)
            state_in.state.xn[self.network.species_map["he4"]] = self.xhe

            # integrate to get the output state
            state_out = self.integrator.integrate(state_in, time)
            
            # set the solution values
            for n in range(self.network.nspec):
                y[i][n] = state_out.state.xn[n]
            y[i][self.ienuc] = state_out.state.e
            y[i][self.idens] = state_out.state.rho
            y[i][self.itemp] = state_out.state.t
        
        if initData:
            # want all variables
            return y
        else:
            return y[:,0:self.numDependent]

    # Get the solution rhs given state y
    def rhs(self, y):
        num_y = len(y)
        dydt = np.zeros((num_y, self.numDependent)) # nspec+energy

        for i, yi in enumerate(y):
#             # construct an eos type to compute temperature
#             eos_state = EosType()
            
#             # set density, energy, and mass fractions
#             eos_state.state.rho = max(yi[self.idens], 0.0)
#             eos_state.state.e = max(yi[self.ienuc], 0.0)
#             for n in range(self.network.nspec):
#                 eos_state.state.xn[n] = max(yi[n], 0.0)

#             # get temperature
#             eos = Eos()
#             eos.evaluate(eos_state.eos_input_re, eos_state)
#             temp = eos_state.state.t
            
#             if i<5:
#                 print(eos_state.state.rho, eos_state.state.e, eos_state.state.t)
            
            # construct a burn type
            state = BurnType()

            # set density & temperature
            state.state.rho = max(yi[self.idens], 0.0)
            state.state.t = max(yi[self.itemp], 0.0)

            # mass fractions
            for n in range(self.network.nspec):
                state.state.xn[n] = max(yi[n], 0.0)

            # evaluate the rhs
            self.network.rhs(state)
            
            # get rhs
            f = self.network.rhs_to_x(state.ydot)
            for n in range(self.network.nspec):
                dydt[i][n] = f[n]
 
            dydt[i][self.ienuc] = f[self.network.net_ienuc]
            
        return dydt

    # take random pairs of states in-between time interval
    def getPair(self, t, y):
        index1 = random.randint(0,len(t)-1)
        index2 = random.randint(0,len(t)-1)
    
        if t[index1] == t[index2]:
            y0 = np.concatenate(([t[index1]], y[0,:]), axis=None)
            return (y0, y[index1,:], t[index1])
    
        # return (dt, y0, yn)
        if t[index1] < t[index2]:
            y0 = np.concatenate(([t[index2]-t[index1]], y[index1,:]),
                                axis=None)
            return (y0, y[index2,:], t[index2])
        else: 
            y0 = np.concatenate(([t[index1]-t[index2]], y[index2,:]),
                                axis=None)
            return (y0, y[index1,:], t[index1])
    
    # compute truth solutions from t=0 and generate pairs of solutions to 
    # use as input and output data
    # input = (dt, solution @ t1), output = (solution @ t2) where t1 < t2 <= end_time
    def generateData(self, NumSamples):
        # time
        t0 = np.linspace(0, self.end_time, NumSamples)
        
        # get the truth solution as a function of t
        y0 = self.sol(t0, initData=True)
        
        # get pairs of truth solutions (input state + dt, output truth state, time)
        x = np.empty((2*NumSamples, y0.shape[1]+1))
        y = np.empty((2*NumSamples, y0.shape[1])) 
        t = np.empty((2*NumSamples, 1))

        for i in range(2*NumSamples):
            x[i,:], y[i,:], t[i] = self.getPair(t0, y0)
        
        return (x, y, t)