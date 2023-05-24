import torch
from torch.autograd import Variable
from torch.nn import RNN, GRU, LSTM
import numpy as np
from tqdm import tqdm
import pickle as pkl
import gc
import math


def get_device(X):
    if X.is_cuda:
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def sech(X):
    device = get_device(X)
    return torch.diag_embed(1 / (torch.cosh(X)))


def oneStep(*params, model):
    # Params is a tuple including h, x, and c (if LSTM)
    l = len(params)
    if l < 2:
        print('Params must be a tuple containing at least (x_t, h_t)')
        return None
    elif l > 2:
        states = (params[1], params[2])
        return model(params[0], states)
    else:
        return model(*params)


def oneStepVarQR(J, Q):
    # Z = torch.matmul(torch.transpose(J, 1, 2), Q)  # Linear extrapolation of the network in many directions
    Z = torch.matmul(J, Q)
    q, r = torch.linalg.qr(Z, mode='reduced')  # QR decomposition of new directions
    s = torch.diag_embed(torch.sign(torch.diagonal(r, dim1=1, dim2=2)))  # extract sign of each leading r value
    return {'Q': torch.matmul(q, s), 'R': torch.matmul(s, r),
            'S': s}  # return positive r values and corresponding vectors


def jac(rnn_layer, ht, xt, bias=False, nonlin = 'tanh'):
    device = get_device(ht)
    U = rnn_layer.U.weight
    V = rnn_layer.V.weight
    if rnn_layer.U.bias is not None:
        b_u = rnn_layer.U.bias
    else:
        b_u = torch.zeros(U.shape[0],).to(device)
    if rnn_layer.V.bias is not None:
        b_v = rnn_layer.V.bias
    else:
        b_v = torch.zeros(V.shape[0],).to(device)
    b = b_u + b_v
    batch_size, hidden_size = ht.shape
    h_in = ht.transpose(-2, -1).detach()
    x_in = xt.squeeze(dim=1).t()  # New shape: (input_shape, batch_size)
    # J = torch.zeros(batch_size, num_layers * hidden_size, num_layers * hidden_size).to(device)
    y = (U @ x_in + V @ h_in + b.repeat(batch_size, 1).t()).t()
    if nonlin == 'relu':
        J_h = 1.0*(y>0) @ V
    elif nonlin == 'tanh':
        J_h = sech(y) @ V
    return J_h


def FTLE(*params, model, k_LE=100000, rec_layer=0, kappa=10, diff=10, warmup=200, T_ons=1, save_s=False,
         bias=False, nonlin='tanh'):
    '''
    Calculates the FTLE of an RNN. Found by taking expansion factor at each time step and calculating each LE.
    Inputs:
    Params - either x, h for RNN, GRU; or x, (h, c) for LSTM
    Model: Pytorch RNN model with parameters
    rec_layer = ['rnn', 'lstm, 'gru', 'asrnn'] - Determines which model (and therefore Jacobian) to use
    '''
    cuda = next(model.parameters()).is_cuda
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    x_in = Variable(params[0], requires_grad=False).to(device)
    hc = params[1]
    h0 = Variable(hc, requires_grad=False).to(device)
    num_layers, batch_size, hidden_size = h0.shape
    _, feed_seq, input_size = x_in.shape
    L = num_layers * hidden_size
    ht_list = []
    xt_list = []

    k_LE = max(min(L, k_LE), 1)
    Q = torch.reshape(torch.eye(L), (1, L, L)).repeat(batch_size, 1, 1).to(device)
    Q = Q[:, :, :k_LE]  # Choose how many exponents to track

    ht = h0
    states = (ht,)  # make tuple for easier generalization
    rvals = torch.eye(k_LE).unsqueeze(0).unsqueeze(0).repeat(batch_size, feed_seq, 1, 1).to(device)  # storage
    qvect = torch.zeros(batch_size, feed_seq, L, k_LE)  # storage
    if save_s:
        s_list = torch.eye(k_LE).unsqueeze(0).unsqueeze(0).repeat(batch_size, feed_seq, 1, 1).to(
            device)  # storage of S (sign multipliers)
    t = 0
    # Warmup
    with torch.no_grad():
        for xt in tqdm(x_in.transpose(0, 1)[:warmup]):
            xt = torch.unsqueeze(xt, 1)  # select t-th element in the fed sequence
            J = jac(model.rnn_layer, ht.squeeze(), xt, bias=bias)
            _, states = oneStep(xt, ht.squeeze(), model=model)
            ht = states
            Q = torch.matmul(torch.transpose(J, 1, 2), Q)
            # print(f"Q shape: {Q.shape}")
            # Q = torch.matmul(J, Q)
        Q, _ = torch.linalg.qr(Q, mode='reduced')

        T_pred = math.log2(kappa / diff)

        # Actual Evolution (storing expansion and contraction)
        t_QR = t
        Jacs = []
        for xt in tqdm(x_in.transpose(0, 1)):
            if (t - t_QR) >= T_ons or t == 0 or t == feed_seq:
                QR = True
            else:
                QR = False
            xt = torch.unsqueeze(xt, 1)  # select t-th element in the fed sequence
            J = jac(model.rnn_layer, ht, xt, bias=bias, nonlin=nonlin)
            # Jacs.append(J)
            _, states = oneStep(xt, ht, model=model)
            if QR:
                qr_dict = oneStepVarQR(J, Q)
                Q, r = qr_dict['Q'], qr_dict['R']
                t_QR = t
                if save_s:
                    S = qr_dict['S']

            else:
                Q = torch.matmul(torch.transpose(J, 1, 2), Q)
                r = torch.eye(hidden_size).unsqueeze(0).repeat(batch_size, 1, 1)
                if save_s:
                    S = torch.eye(hidden_size).unsqueeze(0).repeat(batch_size, 1, 1)
            ht = states
            # ht_list.append(ht)
            # xt_list.append(xt)
            rvals[:, t, :] = r
            qvect[:, t, :, :] = Q
            if save_s:
                s_list[:, t, :, :] = S

            t = t + 1
        rtotal = torch.cumsum(torch.log(torch.diagonal(rvals.detach(), dim1=-2, dim2=-1)), dim=1)
        denom = torch.arange(start=1, end=rtotal.shape[1] + 0.1).to(device)
        FTLEs = torch.div(rtotal,
                          denom.unsqueeze(0).unsqueeze(-1).repeat((rtotal.shape[0], 1, rtotal.shape[2])))
        # Jacs = torch.stack(Jacs)
    # ftle_dict = {'FTLE': FTLEs, 'rvals': torch.diagonal(rvals, dim1=-2, dim2=-1), 'qvects': qvect}
    ftle_dict = {'rvals': torch.diagonal(rvals, dim1=-2, dim2=-1)}
    ftle_dict = {'rvals': torch.diagonal(rvals, dim1=-2, dim2=-1), 'FTLE': FTLEs}
    if save_s:
        ftle_dict['s'] = s_list
    # all_h = torch.Tensor(feed_seq, batch_size, hidden_size, hidden_size).to(device)
    # torch.cat(ht_list, out=all_h)
    # all_x = torch.Tensor(batch_size, feed_seq, input_size).to(device)
    # torch.cat(xt_list, out=all_x, dim=1)
    # ftle_dict['h_list'] = all_h
    # ftle_dict['x_list'] = all_x
    # ftle_dict['Jacs'] = Jacs
    # print(all_h)
    return ftle_dict
