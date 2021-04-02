import torch
import numpy as np
from StarKiller.initialization import starkiller_initialize
from StarKiller.interfaces import BurnType, EosType
from StarKiller.integration import Integrator
from StarKiller.network import Network
from StarKiller.eos import Eos

class ReactionsSystem(object):
    def __init__(self):
        starkiller_initialize("probin_aprox13")
        self.network = Network()
        self.integrator = Integrator()
        self.init_state()
        self.numDependent = self.network.nspec + 2

    def init_state(self):
        # Input sampling domain & scaling
        self.dens = 1.0e8
        self.temp = 4.0e8
        self.xhe = 1.0

        self.end_time = 1.0

        self.time_scale = 1.0e-6
        self.density_scale = self.dens
        self.temperature_scale = self.temp * 10

        # do an eos call to set the internal energy scale
        eos = Eos()
        eos_state = EosType()

        eos_state.state.t = self.temp
        eos_state.state.rho = self.dens

        # pick a composition for normalization of Ye = 0.5 w/ abar = 12, zbar = 6
        eos_state.state.abar = 12.0
        eos_state.state.zbar = 6.0
        eos_state.state.y_e = eos_state.state.zbar / eos_state.state.abar
        eos_state.state.mu_e = 1.0 / eos_state.state.y_e

        # use_raw_inputs uses only abar, zbar, y_e, mu_e for the EOS call
        # instead of setting those from the mass fractions
        eos.evaluate(eos_state.eos_input_rt, eos_state, use_raw_inputs=True)

        self.energy_scale = eos_state.state.e

        print("density_scale = ", self.density_scale)
        print("temperature_scale = ", self.temperature_scale)
        print("energy_scale = ", self.energy_scale)

    # Get the solution given time t
    # t is a PyTorch tensor containing the times at which
    # we want to evaluate the System solution.
    def sol(self, t):
        num_time = t.data.nelement()
        y = torch.zeros(num_time, self.network.nspec+2)
        
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
            state_out = self.integrator.integrate(state_in, time * self.time_scale)
            
            # set the solution values
            for n in range(self.network.nspec):
                y[i][n] = state_out.state.xn[n]
            y[i][self.network.net_itemp] = state_out.state.t / self.temperature_scale
            y[i][self.network.net_ienuc] = state_out.state.e / self.energy_scale
        
        return y

    # Get the solution rhs given state y
    # scaled solution: ys = y / y_scale
    # scaled time: ts = t / t_scale
    # f = dys/dts = (dy/y_scale) / (dt/t_scale) = (dy/dt) * (t_scale / y_scale)
    def rhs(self, y):
        num_y = list(y.size())[0]
        dydt = torch.zeros(num_y, self.network.nspec+2)

        for i, yi in enumerate(y):
            # construct a burn type
            state = BurnType()

            # set density & temperature
            state.state.rho = self.dens
            state.state.t = max(yi[self.network.net_itemp] * self.temperature_scale, 0.0)

            # mass fractions
            for n in range(self.network.nspec):
                state.state.xn[n] = max(yi[n], 0.0)

            # evaluate the rhs
            self.network.rhs(state)
            
            # get rhs
            f = self.network.rhs_to_x(state.ydot)
            for n in range(self.network.nspec):
                dydt[i][n] = f[n] * self.time_scale

            dydt[i][self.network.net_itemp] = f[self.network.net_itemp] * self.time_scale / self.temperature_scale
            dydt[i][self.network.net_ienuc] = f[self.network.net_ienuc] * self.time_scale / self.energy_scale
                
        return dydt
