import torch
from torch import linalg, nn
from torch.nn import functional as f
from config import *
from lyapunov import *
from models import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors, cm
import imageio
from torch.distributions import Categorical
import os
import math
from training import *
from lyapunov import param_split
from collections import OrderedDict



if __name__ == '__main__':
    print('Starting')
    batch_size = 128
    le_batch_size = 25
    output_size = 10
    max_epoch = 4
    learning_rate = 0.002
    dropout = 0.1
    hidden_size = 128
    save_interval = 1
    model_type = 'rnn'
    p = 0.001
    seq_length = 28
    input_size = 28
    max_sub_epoch = 2
    in_epoch_saves = 4
    lr = 0.1
    
    params = torch.linspace(0.005, end= 0.025, steps = 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    dcon = DataConfig('../Dataset/', input_size = input_size, batch_size= batch_size, input_seq_length = seq_length, 
                                                target_seq_length = 1, val_frac = 0.2, 
                                                test_frac = 0, name_params = {'insize':input_size})
    mcon = ModelConfig(model_type, 1, hidden_size, dcon.input_size, output_size = output_size, dropout=dropout, 
                        init_type = 'normal', init_params = {'mean':0, 'std':p},
                        device = device, bias = False, id_init_param = 'std')                                            
    tcon = TrainConfig(model_dir = 'SMNIST/Models', batch_size = batch_size, max_epoch = max_epoch, 
                                                        optimizer = 'adam', learning_rate = learning_rate)
    fcon = FullConfig(dcon, tcon, mcon)
    
    
    if in_epoch_saves >0:
        if os.path.exists('SMNIST/training_saveIdcs.p'):
            save_idcs = torch.load('SMNIST/training_saveIdcs.p')
        else:
            train_dataloader = torch.utils.data.DataLoader(fcon.data.datasets['train_set'], 
                                                            batch_size = fcon.train.batch_size)
            epoch_samples = len(list(train_dataloader))
            save_idcs = part_equal(epoch_samples, in_epoch_saves)
            torch.save(save_idcs, 'SMNIST/training_saveIdcs.p')
    else:
        save_idcs = []
        
    val_dl = torch.utils.data.DataLoader(fcon.data.datasets['val_set'], 
                                                    batch_size = le_batch_size)
    le_input, le_target = next(iter(val_dl))
    le_input = le_input.to(fcon.device).squeeze(1)
    le_target = le_target.to(fcon.device)
    print(f'Target Shape {le_target.shape}')
    
    h_le = torch.randn(1, le_batch_size, hidden_size).to(fcon.device)
    
    if in_epoch_saves >0:
        epoch_samples = len(list(train_dataloader))
        save_idcs = part_equal(epoch_samples, in_epoch_saves)
    else:
        save_idcs = []
        
    for p in params[:1]:
        p = float(int(p*1000))*1.0/1000
        fcon.model.init_params['std'] = p
        print(f'Parameter = {p}')
        for epoch in range(1, max_epoch+1, save_interval):
            print(f"Epoch {epoch}")
            for it in range(in_epoch_saves + 1):
                if it == in_epoch_saves:
                    model, optimizer, train_loss, _ = load_checkpoint(fcon, epoch)
                    suffix = ''
                    it_lab = ''
                elif epoch == 0 or epoch >= max_sub_epoch:
                    continue
                else:
                    ind = save_idcs[it]
                    print(f'Iteration: {ind}')
                    model, optimizer, train_loss = load_iter_checkpoint(fcon, epoch, save_idcs[it])
                    suffix = f'_iter{ind}'
                    it_lab = f', Iteration {ind}'
                ftle_dict = torch.load(f'SMNIST/LEs/{fcon.name()}_e{epoch}{suffix}_FTLE.p')
                gradV_list, gradW_list, loss_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_grads.p')
                pred_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_logits.p')
                h_list = ftle_dict['h_list']
                x_list = ftle_dict['x_list']
                h_le = ftle_dict['h'].to(fcon.device)
                
                outputs, h_t = model(le_input, h_le)
                
                #Load/Generate Grad SVDs
                fname = f'SMNIST/Eigs/{fcon.name()}_e{epoch}{suffix}_svds.p'
                if os.path.exists(fname):
                    u_list, s_list, v_list = torch.load(fname)
                else:
                    u_list = torch.zeros(seq_length, le_batch_size, hidden_size,hidden_size)
                    v_list = torch.zeros(seq_length, le_batch_size, hidden_size,hidden_size)
                    s_list = torch.zeros(seq_length, le_batch_size, hidden_size)
                    for i, M in tqdm(zip(range(seq_length), gradV_list)):
                        U, S, Vh = linalg.svd(M)
                        u_list[i] = U
                        s_list[i] = S
                        v_list[i] = Vh
                    torch.save((u_list, s_list, v_list), fname)
                
                fname = f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_dVs.p'
                ranks = torch.arange(hidden_size)
                state_dict = model.rnn_layer.state_dict()
                V = state_dict['weight_hh_l0'].to(device)
                
                if os.path.exists(fname):
                    dV_list = torch.load(fname)
                else:
                    dV_list = torch.zeros(len(ranks), seq_length, le_batch_size, hidden_size,hidden_size)
                    for r in tqdm(ranks):
                        for sequence in range(seq_length):
                            temp_mat = torch.bmm(u_list[sequence, :, :, r].unsqueeze(-1), v_list[sequence, :, r, :].unsqueeze(-2))
                            # print(temp_mat.shape)
                            # print(s_list[sequence, :, r].shape)
                            temp_mat = temp_mat*s_list[sequence, :, r].view(-1, 1, 1)
                            # print(temp_mat.shape)
                            prev_dVr = torch.zeros_like(temp_mat) if r == 0 else dV_list[r-1, sequence]
                            dV_list[r, sequence] = prev_dVr + temp_mat
                        # print((V.unsqueeze(0).unsqueeze(0).repeat(seq_length, le_batch_size, 1,1).to(device) - dV_list[r,sequence].to(device)).mean(dim = (-1, -2, -3)))
                    torch.save(dV_list, fname)
                
                # print((grad_list.unsqueeze(0).repeat(len(ranks), 1,1,1,1).cpu() - dV_list.cpu()).mean(dim= (-1, -2, -3, -4)))
                criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
                # print(state_dict)
                fname = f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_baseLosses.p'
                if os.path.exists(fname):
                    base_loss = torch.load(fname)
                else:
                    base_loss = torch.zeros(seq_length, le_batch_size)
                    for sequence in range(seq_length):
                        preds, _ = model(le_input[:, :sequence+1], h_le)
                        temp_loss = criterion(preds, le_target)
                        base_loss[sequence] = temp_loss
                    torch.save(base_loss, fname)
                
                fname = f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_fullRankLosses.p'
                if os.path.exists(fname):
                    full_red = torch.load(fname)
                else:
                    full_red = torch.zeros(seq_length, le_batch_size).to(device)
                    for sequence in range(seq_length):
                            V = state_dict['weight_hh_l0'].unsqueeze(0).repeat(le_batch_size, 1, 1)
                            V = V - lr*gradV_list[sequence].to(device)
                            for batch in range(le_batch_size):
                                new_state = OrderedDict([(k, V[batch]) if k == 'weight_hh_l0' else (k, v) for k, v in state_dict.items()])
                                model.rnn_layer.load_state_dict(new_state, strict = False)
                                preds, _ = model(le_input[batch, :sequence+1].unsqueeze(0), h_le[:, batch].unsqueeze(1))
                                temp_loss = criterion(preds, le_target[batch].unsqueeze(0))
                                full_red[sequence, batch] = temp_loss
                    torch.save(full_red, fname)
                
                
                fname = f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_rankLosses.p'
                
                # overwrite = False
                if os.path.exists(fname):
                    dV_losses = torch.load(fname).to(device)
                else:
                    
                    dV_losses = torch.zeros(len(ranks), seq_length, le_batch_size).to(device)
                    
                    #Calculate loss reduction from rank-r delta-V's
                    
                    model.eval()
                    with torch.no_grad():
                        for r in tqdm(ranks):
                            for sequence in range(seq_length):
                                V = state_dict['weight_hh_l0'].unsqueeze(0).repeat(le_batch_size, 1, 1)
                                V = V - lr*dV_list[r, sequence].to(device)
                                for batch in range(le_batch_size):
                                    # model = RNNModel(mcon).to(device)
                                    new_state = OrderedDict([(k, V[batch]) if k == 'weight_hh_l0' else (k, v) for k, v in state_dict.items()])
                                    model.rnn_layer.load_state_dict(new_state, strict = False)
                                    preds, _ = model(le_input[batch, :sequence+1].unsqueeze(0), h_le[:, batch].unsqueeze(1))
                                    # print(preds.shape)
                                    temp_loss = criterion(preds, le_target[batch].unsqueeze(0))
                                # print(temp_loss)
                                    dV_losses[r, sequence, batch] = temp_loss
                        torch.save(dV_losses, f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_rankLosses.p')
                    
                print(f'dV Losses {dV_losses[-1, -1]}')
                print(f'Initial Losses {base_loss[-1]}')
                rank_differences = dV_losses - base_loss.unsqueeze(0).repeat(len(ranks), 1, 1).to(device)
                print(f'Mean Difference (batch) {rank_differences[-1, -1]}')
                # print(full_red)
                # print(f'Standard Deviation (batch) {rank_differences.std(dim = -1)}')
            