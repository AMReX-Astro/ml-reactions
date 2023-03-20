import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def my_heaviside(x, input):
    y = torch.zeros_like(x)
    y[x < 0] = 0
    y[x > 0] = 1
    y[x == 0] = input
    return y

def component_loss_f(prediction, targets):
    #Takes the MSE of each component and returns the array of losses
    loss = torch.zeros(prediction.shape[1])
    L = nn.MSELoss()
    for i in range(prediction.shape[1]):
        loss[i] = L(prediction[:, i], targets[:,i])
    return loss


def component_loss_f_L1(prediction, targets):
    #Takes the MSE of each component and returns the array of losses
    loss = torch.zeros(prediction.shape[1])
    L = nn.L1Loss()
    for i in range(prediction.shape[1]):
        loss[i] = L(prediction[:, i], targets[:,i])
    return loss


def log_loss(prediction, target, nnuc=13):
    # Log Loss Function for standard ML.
    # If there are negative values in X we use MSE
    #Enuc stays with MSE because its normalized

    #X is not allowed to be negative. Enuc is
    X = prediction[:, :nnuc]
    X_target = target[:, :nnuc]
    enuc = prediction[:, nnuc]
    enuc_target = target[:, nnuc]

    L = nn.MSELoss()
    F = nn.L1Loss()

    enuc_loss = L(enuc, enuc_target) + F(torch.sign(enuc), torch.sign(enuc_target))


    #if there are negative numbers we cant use log on mass fractions
    if torch.sum(X < 0) > 0:
        #how much do we hate negative numbers?
        factor = 1000 # a lot
        return enuc_loss + factor*L(X, X_target)

    elif torch.sum(X == 0) > 0:
        return enuc_loss + L(X, X_target)
    
    else:
        Xlog = -0.3/torch.log10(X)
        Xlog_target = -0.3/torch.log10(X_target)
        
        barrier = torch.tensor([.001], device=device)
        value = torch.tensor([0.], device=device)

        #greater than barrier we apply mse loss
        #less then barier we apply log of mse loss
        A = my_heaviside(X_target - barrier, value)
        B = -my_heaviside(X_target - barrier, value) + 1

        X_loss =  torch.sum(A * L(X, X_target) + B * 0.01 * L(Xlog, Xlog_target))

    return enuc_loss + X_loss

def log_loss_w_enuc(prediction, target, nnuc=13):
    # Log Loss Function for standard ML.
    # If there are negative values in X we use MSE
    #Enuc stays with MSE because its normalized

    #X is not allowed to be negative. Enuc is
    X = prediction[:, :nnuc]
    X_target = target[:, :nnuc]
    enuc = prediction[:, nnuc]
    enuc_target = target[:, nnuc]

    L = nn.MSELoss()
    F = nn.L1Loss()

    if torch.sum(enuc <= 0) > 0:
        enuc_loss = L(enuc, enuc_target) + F(torch.sign(enuc), torch.sign(enuc_target))

    else:
        alpha = 1.0  # prevents computing log(enuc=1)
        
        elog = -0.1/torch.log10(alpha*enuc)
        elog_target = -0.1/torch.log10(alpha*enuc_target)
        
        barrier = torch.tensor([.1], device=device)
        value = torch.tensor([0.], device=device)

        #greater than barrier we apply mse loss
        #less then barier we apply log of mse loss
        A = my_heaviside(enuc_target - barrier, value)
        B = -my_heaviside(enuc_target - barrier, value) + 1

        enuc_loss =  torch.sum(A * L(enuc, enuc_target) + B * 0.1 * L(elog, elog_target))

    
    #if there are negative numbers we cant use log on mass fractions
    if torch.sum(X < 0) > 0:
        #how much do we hate negative numbers?
        factor = 1000 # a lot
        return enuc_loss + factor*L(X, X_target)

    elif torch.sum(X == 0) > 0:
        return enuc_loss + L(X, X_target)
    
    else:
        Xlog = -0.3/torch.log10(X)
        Xlog_target = -0.3/torch.log10(X_target)
        
        barrier = torch.tensor([.001], device=device)
        value = torch.tensor([0.], device=device)

        #greater than barrier we apply mse loss
        #less then barier we apply log of mse loss
        A = my_heaviside(X_target - barrier, value)
        B = -my_heaviside(X_target - barrier, value) + 1

        X_loss =  torch.sum(A * L(X, X_target) + B * 0.01 * L(Xlog, Xlog_target))

    return enuc_loss + X_loss

def logX_loss(prediction, target, nnuc=13):
    # We are working with mass fractions in the form of -1/log(X_k)

    X = prediction[:, :nnuc]
    X_target = target[:, :nnuc]
    enuc = prediction[:, nnuc]
    enuc_target = target[:, nnuc]

    L = nn.MSELoss()
    F = nn.L1Loss()

    # enuc is allowed to be negative
    # but penalty should be given if prediction is of different signs
    enuc_fac = 1
    enuc_loss = L(enuc, enuc_target) + enuc_fac * F(torch.sign(enuc), torch.sign(enuc_target))

    # we do not want negative values for mass fractions
    if torch.sum(X < 0) > 0:
        #how much do we hate negative numbers?
        factor = 1000  #a lot
    else:
        factor = 1

    return factor * L(X, X_target) + enuc_loss


def rms_weighted_error(input, target, solution, atol=1e-6, rtol=1e-6):
    error_weight = atol + rtol * torch.abs(solution)
    weighted_error = (input - target) / error_weight
    rms_weighted_error = torch.sqrt((weighted_error**2).sum() / input.data.nelement())
    return rms_weighted_error

def loss_pinn(input, prediction, target, enuc_fac, enuc_dot_fac,
                log_option=False, nnuc=13):

    #input mass fractions/density/temp at t=n
    dt     = input[:, 0]
    X_n    = input[:, 1:nnuc+1]
    rho_n  = input[:, nnuc+1]
    temp_n = input[:, nnuc+2]

    #output mass fractions (NN) at t=n+1
    X_nn_1    = prediction[:, :nnuc]
    enuc_nn_1 = prediction[:, nnuc]
    #rhs_nn_1  = prediction

    #truth mass fractions (calculated) at t=n+1
    X_t_1    = target[:, :nnuc]
    enuc_t_1 = target[:, nnuc]
    #Asssume rho is constant, call eos to get updated T. then call rhs on updated variables.
    #This will all be done within maestro and the output is just loaded here.

    #rhs_t_1 = target[:, 15:28]

    dpndx = torch.zeros_like(prediction)
    for n in range(nnuc+1):
        if input.grad is not None:
            #sets componnents to zero if not already zero
            input.grad.data.zero_()

        prediction[:, n].backward(torch.ones_like(prediction[:,n]), retain_graph=True)
        dpndx[:, n] = input.grad.clone()[:, 0]
        input.grad.data.zero_()



    #Scale derivative of solution to be same scale as normalized values
    # for i in range(dpndx.shape[0]):
    #     dpndx[i, 0:nnuc] = dpndx[i, 0:nnuc]/rate_factors
    #both enuc and enucdot were scaled so we have to apply both factors
    dpndx[:, nnuc] = dpndx[:, nnuc] * enuc_fac/enuc_dot_fac
    #dpndx = dpndx/rate_factors
    #dpndx[enuc_dot] = dpndx[enuc_dot] * enuc_fac/enuc_dot_fac

    if log_option:

        loss_sum = torch.tensor([0.])
        for var in range(prediction.shape[1]):

            L = nn.MSELoss()

            #if there are negative numbers we cant use log
            if torch.sum(prediction[:, var] < 0) > 0:

                #how much do we hate negative numbers?
                # factor = 1 #  negative numbers aren't bad here. we just need a better way to handle it.
                # return factor*L(prediction[:,var], target[:, nnuc+1+var])

                pred = torch.abs(prediction[:, var])

                barrier = torch.tensor([.1], device=device)
                #greater than barrier we apply mse loss
                #less then barier we apply log of mse loss

                barrier1 = torch.tensor([.1], device=device)
                barrier2 = torch.tensor([100], device=device)
                value = torch.tensor([0.], device=device)


                y1 = my_heaviside(target[:, nnuc+1:] - barrier1, value)
                y2 = -my_heaviside(target[:, nnuc+1:] - barrier2, value) + 1
                y3 = -my_heaviside(target[:, nnuc+1:] - barrier1, value) + 1
                y4 = my_heaviside(target[:, nnuc+1:] - barrier2, value)

                A = y1*y2
                B = torch.abs(y3-y4)

                L =  torch.sum(B * L(pred, target[:, nnuc+1+var]) + A* torch.abs(.01*L(torch.log(target[:, nnuc+1+var]), torch.log(pred))))

            else:
                barrier = torch.tensor([.1], device=device)
                #greater than barrier we apply mse loss
                #less then barier we apply log of mse loss

                barrier1 = torch.tensor([.1], device=device)
                barrier2 = torch.tensor([100], device=device)
                value = torch.tensor([0.], device=device)


                y1 = my_heaviside(target[:, nnuc+1:] - barrier1, value)
                y2 = -my_heaviside(target[:, nnuc+1:] - barrier2, value) + 1
                y3 = -my_heaviside(target[:, nnuc+1:] - barrier1, value) + 1
                y4 = my_heaviside(target[:, nnuc+1:] - barrier2, value)

                A = y1*y2
                B = torch.abs(y3-y4)

                L =  torch.sum(B * L(prediction[:, var], target[:, nnuc+1+var]) + A* torch.abs(.01*L(torch.log(target[:, nnuc+1+var]), torch.log(prediction[:, var]))))

        return loss_sum
    else:

        L = nn.MSELoss()

        loss = L(dpndx, target[:, nnuc+1:])

        return loss

def loss_mass_fraction(prediction, nnuc=13):
    return 10* torch.abs(1 - torch.sum(prediction[:, :nnuc]))

def loss_mass_fraction_log(prediction, totsum=1.0, nnuc=13):
    L = nn.MSELoss()
    total = totsum * torch.ones(prediction.shape[0], device=device)
    mass_fraction = torch.exp(-0.5/prediction[:, :nnuc])
    
    return L(torch.sum(mass_fraction, 1), total)

def loss_mass_fraction_half(prediction, nnuc=2):
    F = nn.L1Loss()
    total = 0.5*torch.ones(prediction.shape[0], device=device)
    
    return F(torch.sum(prediction[:, :nnuc], 1), total)

def loss_mass_fraction_half_L(prediction, nnuc=2):
    L = nn.MSELoss()
    total = 0.5*torch.ones(prediction.shape[0], device=device)
    
    return L(torch.sum(prediction[:, :nnuc], 1), total)

def loss_pure(prediction, target, log_option = False):

    if log_option:
        L = nn.MSELoss()

        #if there are negative numbers we cant use log
        if torch.sum(prediction < 0) > 0:
            #how much do we hate negative numbers?
            factor = 1000 # a lot
            return factor*L(prediction, target[:, :nnuc+1])

        else:
            barrier = torch.tensor([.1], device=device)
            value = torch.tensor([0.], device=device)
            #greater than barrier we apply mse loss
            #less then barier we apply log of mse loss

            A = my_heaviside(target[:, :nnuc+1] - barrier, value)
            B = -my_heaviside(target[:, :nnuc+1] - barrier, value) + 1

            L =  torch.sum(A * L(target[:, :nnuc+1], prediction) + B* torch.abs(.01*L(torch.log(target[:, :nnuc+1]), torch.log(prediction))))

            return L

    else:
        L = nn.MSELoss()
        return L(prediction, target[:, :nnuc+1])


def tanh_loss(dxdt, prediction):
    L = nn.MSELoss()
    out = L(dxdt, prediction)

    return torch.tanh(out)


from .tools import scaling_func

def derivative_loss_piecewise(dxdt, actual, enuc_fac, enuc_dot_fac, nnuc=13):

    b1 = torch.tensor([.1])
    b2 = torch.tensor([100.])
    scaling = torch.tensor([.01])


    dxdt[:, nnuc] = dxdt[:, nnuc] * enuc_fac/enuc_dot_fac

    scaled_dxdt = scaling_func(torch.abs(dxdt), lambda x : x, b1, b2, scaling)
    scaled_actual = scaling_func(torch.abs(actual), lambda x : x, b1, b2, scaling)

    F = nn.L1Loss()

    return F(scaled_dxdt, scaled_actual)

def signed_loss_function(pred, actual):
    F = nn.L1Loss()
    return F(torch.sign(pred), torch.sign(actual))


def relative_loss(prediction, target):


    threshold = target.clone()
    threshold[target<1.e-15] = 1.e-15

    L = torch.mean(torch.abs(prediction-target)/threshold)

    #If has nan
    if L.isnan().sum().item() > 0:
        return torch.tensor([1.0])
    else:
        return torch.mean(torch.abs(prediction-target)/threshold)
