import numpy as np
from pynucastro.rates import Tfactors
import numba
import tensorflow as tf


ip = 0
ihe4 = 1
ic12 = 2
ic13 = 3
in13 = 4
in14 = 5
in15 = 6
io14 = 7
io15 = 8
nnuc = 9

A = np.zeros((nnuc), dtype=np.int32)

A[ip] = 1
A[ihe4] = 4
A[ic12] = 12
A[ic13] = 13
A[in13] = 13
A[in14] = 14
A[in15] = 15
A[io14] = 14
A[io15] = 15

Z = np.zeros((nnuc), dtype=np.int32)

Z[ip] = 1
Z[ihe4] = 2
Z[ic12] = 6
Z[ic13] = 6
Z[in13] = 7
Z[in14] = 7
Z[in15] = 7
Z[io14] = 8
Z[io15] = 8

# @numba.njit()
def ye(Y):
    return np.sum(Z * Y)/np.sum(A * Y)

# @numba.njit()
def p_c12__n13(Tf):
    # c12 + p --> n13
    rate = 0.0
    
    # ls09n
    rate += tf.math.exp(  17.1482 + -13.692*Tf.T913i + -0.230881*Tf.T913
                  + 4.44362*Tf.T9 + -3.15898*Tf.T953 + -0.666667*Tf.lnT9)
    # ls09r
    rate += tf.math.exp(  17.5428 + -3.77849*Tf.T9i + -5.10735*Tf.T913i + -2.24111*Tf.T913
                  + 0.148883*Tf.T9 + -1.5*Tf.lnT9)
    
    return rate

# @numba.njit()
def p_c13__n14(Tf):
    # c13 + p --> n14
    
    # nacrn
    rate = tf.math.exp(  18.5155 + -13.72*Tf.T913i + -0.450018*Tf.T913
                  + 3.70823*Tf.T9 + -1.70545*Tf.T953 + -0.666667*Tf.lnT9)
    # nacrr
    rate += tf.math.exp(  13.9637 + -5.78147*Tf.T9i + -0.196703*Tf.T913
                  + 0.142126*Tf.T9 + -0.0238912*Tf.T953 + -1.5*Tf.lnT9)
    # nacrr
    rate += tf.math.exp(  15.1825 + -13.5543*Tf.T9i
                  + -1.5*Tf.lnT9)
    
    return rate

# @numba.njit()
def n13__c13__weak__wc12(Tf):
    # n13 --> c13
    
    # wc12w
    rate = tf.cast(tf.math.exp(  -6.7601),tf.float64)
    
    return rate

# @numba.njit()
def p_n13__o14(Tf):
    # n13 + p --> o14
    rate = 0.0
    
    # lg06n
    rate += tf.math.exp(  18.1356 + -15.1676*Tf.T913i + 0.0955166*Tf.T913
                  + 3.0659*Tf.T9 + -0.507339*Tf.T953 + -0.666667*Tf.lnT9)
    # lg06r
    rate += tf.math.exp(  10.9971 + -6.12602*Tf.T9i + 1.57122*Tf.T913i
                  + -1.5*Tf.lnT9)
    
    return rate

# @numba.njit()
def p_n14__o15(Tf):
    # n14 + p --> o15
    rate = 0.0
    
    # im05n
    rate += tf.math.exp(  17.01 + -15.193*Tf.T913i + -0.161954*Tf.T913
                  + -7.52123*Tf.T9 + -0.987565*Tf.T953 + -0.666667*Tf.lnT9)
    # im05r
    rate += tf.math.exp(  6.73578 + -4.891*Tf.T9i
                  + 0.0682*Tf.lnT9)
    # im05r
    rate += tf.math.exp(  7.65444 + -2.998*Tf.T9i
                  + -1.5*Tf.lnT9)
    # im05n
    rate += tf.math.exp(  20.1169 + -15.193*Tf.T913i + -4.63975*Tf.T913
                  + 9.73458*Tf.T9 + -9.55051*Tf.T953 + 0.333333*Tf.lnT9)
    
    return rate

# @numba.njit()
def p_n15__he4_c12(Tf):
    # n15 + p --> he4 + c12
    rate = 0.0
    
    # nacrn
    rate += tf.math.exp(  27.4764 + -15.253*Tf.T913i + 1.59318*Tf.T913
                  + 2.4479*Tf.T9 + -2.19708*Tf.T953 + -0.666667*Tf.lnT9)
    # nacrr
    rate += tf.math.exp(  -6.57522 + -1.1638*Tf.T9i + 22.7105*Tf.T913
                  + -2.90707*Tf.T9 + 0.205754*Tf.T953 + -1.5*Tf.lnT9)
    # nacrr
    rate += tf.math.exp(  20.8972 + -7.406*Tf.T9i
                  + -1.5*Tf.lnT9)
    # nacrr
    rate += tf.math.exp(  -4.87347 + -2.02117*Tf.T9i + 30.8497*Tf.T913
                  + -8.50433*Tf.T9 + -1.54426*Tf.T953 + -1.5*Tf.lnT9)
    
    return rate

# @numba.njit()
def o14__n14__weak__wc12(Tf):
    # o14 --> n14
    
    # wc12w
    rate = tf.cast(tf.math.exp(  -4.62354),tf.float64)
    
    return rate

# @numba.njit()
def o15__n15__weak__wc12(Tf):
    # o15 --> n15
    
    # wc12w
    rate = tf.cast(tf.math.exp(  -5.17053),tf.float64)
    
    return rate

def rhs(t, Y, rho, T):
    return rhs_eq(t, Y, rho, T)

class Tfactors(object):
    """ precompute temperature factors for speed """

    def __init__(self, T):
        """ return the Tfactors object.  Here, T is temperature in Kelvin """
        self.T9 = T/1.e9
        self.T9i = 1.0/self.T9
        self.T913i = self.T9i**(1./3.)
        self.T913 = self.T9**(1./3.)
        self.T953 = self.T9**(5./3.)
        self.lnT9 = tf.math.log(self.T9)

# @numba.njit()
def rhs_eq(t, X, rho, T):

    ip = 0
    ihe4 = 1
    ic12 = 2
    ic13 = 3
    in13 = 4
    in14 = 5
    in15 = 6
    io14 = 7
    io15 = 8
    nnuc = 9

    Y = X[:]/A[:]

    Tf = Tfactors(T)

    lambda_p_c12__n13 = p_c12__n13(Tf)
    lambda_p_c13__n14 = p_c13__n14(Tf)
    lambda_n13__c13__weak__wc12 = n13__c13__weak__wc12(Tf)
    lambda_p_n13__o14 = p_n13__o14(Tf)
    lambda_p_n14__o15 = p_n14__o15(Tf)
    lambda_p_n15__he4_c12 = p_n15__he4_c12(Tf)
    lambda_o14__n14__weak__wc12 = o14__n14__weak__wc12(Tf)
    lambda_o15__n15__weak__wc12 = o15__n15__weak__wc12(Tf)

    dYdt = np.zeros((nnuc), dtype=np.float64)

    dYdt[ip] = (
       -rho*Y[ip]*Y[ic12]*lambda_p_c12__n13
       -rho*Y[ip]*Y[ic13]*lambda_p_c13__n14
       -rho*Y[ip]*Y[in13]*lambda_p_n13__o14
       -rho*Y[ip]*Y[in14]*lambda_p_n14__o15
       -rho*Y[ip]*Y[in15]*lambda_p_n15__he4_c12
       )

    dYdt[ihe4] = (
       +rho*Y[ip]*Y[in15]*lambda_p_n15__he4_c12
       )

    dYdt[ic12] = (
       -rho*Y[ip]*Y[ic12]*lambda_p_c12__n13
       +rho*Y[ip]*Y[in15]*lambda_p_n15__he4_c12
       )

    dYdt[ic13] = (
       -rho*Y[ip]*Y[ic13]*lambda_p_c13__n14
       +Y[in13]*lambda_n13__c13__weak__wc12
       )

    dYdt[in13] = (
       -Y[in13]*lambda_n13__c13__weak__wc12
       -rho*Y[ip]*Y[in13]*lambda_p_n13__o14
       +rho*Y[ip]*Y[ic12]*lambda_p_c12__n13
       )

    dYdt[in14] = (
       -rho*Y[ip]*Y[in14]*lambda_p_n14__o15
       +rho*Y[ip]*Y[ic13]*lambda_p_c13__n14
       +Y[io14]*lambda_o14__n14__weak__wc12
       )

    dYdt[in15] = (
       -rho*Y[ip]*Y[in15]*lambda_p_n15__he4_c12
       +Y[io15]*lambda_o15__n15__weak__wc12
       )

    dYdt[io14] = (
       -Y[io14]*lambda_o14__n14__weak__wc12
       +rho*Y[ip]*Y[in13]*lambda_p_n13__o14
       )

    dYdt[io15] = (
       -Y[io15]*lambda_o15__n15__weak__wc12
       +rho*Y[ip]*Y[in14]*lambda_p_n14__o15
       )

    dXdt = dYdt * A[:]

    return dXdt
