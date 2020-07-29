from __future__ import print_function

import numpy as np
from scipy.integrate import ode
import cno_rhs as cno
import matplotlib.pyplot as plt

# integrate the reaction network forward by tmax
def burn(X0, rho, T, tmax, nsave):

    r = ode(cno.rhs).set_integrator("vode", method="bdf",
                                    with_jacobian=False,
                                    atol=1.e-8, rtol=1.e-8,
                                    nsteps = 1500000, order=5) #, min_step=dt)

    t = 0.0

    r.set_initial_value(X0, t)

    r.set_f_params(rho, T)

    dt = tmax/nsave

    t_out = []
    H_out = []
    He_out = []
    O14_out = []
    O15_out = []

    print(t, X0[cno.ip], X0[cno.io14])

    istep = 1
    while r.successful() and istep <= nsave:
        r.integrate(t+dt*istep)

        if r.successful():
            print(r.t, r.y[cno.ip], r.y[cno.io14])
            t_out.append(r.t)
            H_out.append(r.y[cno.ip])
            He_out.append(r.y[cno.ihe4])
            O14_out.append(r.y[cno.io14])
            O15_out.append(r.y[cno.io15])
            istep = istep + 1
        else:
            print("An integration error occurred at time {}".format(r.t))

    return t_out, H_out, He_out, O14_out, O15_out


if __name__ == "__main__":

    # initialize as mass fractions first
    X0 = np.zeros((cno.nnuc), dtype=np.float64)

    X0[cno.ip] = 0.7
    X0[cno.ihe4] = 0.28
    X0[cno.ic12] = 0.02

    rho = 10000.0
    T = 1.e8

    # estimate the H destruction time
    Xdot = cno.rhs(0.0, X0, rho, T)

    tmax = 10.0*np.abs(X0[cno.ip]/Xdot[cno.ip])
    print("tmax: {}".format(tmax))

    nsteps = 100

    t, X_H, X_He, X_O14, X_O15 = burn(X0, rho, T, tmax, nsteps)

    plt.loglog(t, np.array(X_H), label="H1")
    plt.loglog(t, np.array(X_He), label="He4")
    plt.loglog(t, np.array(X_O14), label="O14")
    plt.loglog(t, np.array(X_O15), label="O15")

    plt.xlabel("time [s]")
    plt.ylabel("mass fraction, X")

    plt.legend(frameon=False)

    plt.savefig("burn.png")
