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
    
    val_dl = torch.utils.data.DataLoader(fcon.data.datasets['val_set'], 
                                                    batch_size = le_batch_size)
    le_input, le_target = next(iter(val_dl))
    le_input = le_input.to(fcon.device).squeeze(1)
    le_target = le_target.to(fcon.device)
    
    h_le = torch.randn(1, le_batch_size, hidden_size).to(fcon.device)

    print('Dataloader')
    train_dataloader = torch.utils.data.DataLoader(fcon.data.datasets['train_set'], 
                                                    batch_size = fcon.train.batch_size)
    if in_epoch_saves >0:
        epoch_samples = len(list(train_dataloader))
        save_idcs = part_equal(epoch_samples, in_epoch_saves)
    else:
        save_idcs = []
        
    for p in params[:1]:
        p = float(int(p*1000))*1.0/1000
        fcon.model.init_params['std'] = p
        print(f'Parameter = {p}')
        pearcs = []
        pearcs1 = []
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
                grad_list, loss_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_grads.p')
                pred_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_logits.p')
                print(grad_list.shape)
                
                #Load/Generate Grad SVDs
                fname = f'SMNIST/Eigs/{fcon.name()}_e{epoch}_svds.p'
                if os.path.exists(fname):
                    u_list, s_list, v_list = torch.load(fname)
                else:
                    u_list = torch.zeros(seq_length, le_batch_size, hidden_size,hidden_size)
                    v_list = torch.zeros(seq_length, le_batch_size, hidden_size,hidden_size)
                    s_list = torch.zeros(seq_length, le_batch_size, hidden_size)
                    for i, M in tqdm(zip(range(seq_length), grad_list)):
                        U, S, Vh = linalg.svd(M)
                        u_list[i] = U
                        s_list[i] = S
                        v_list[i] = Vh
                    torch.save((u_list, s_list, v_list), f'SMNIST/Eigs/{fcon.name()}_e{epoch}_svds.p')
                
                fname = f'SMNIST/Grads/{fcon.name()}_e{epoch}{it_lab}_dVs.p'
                ranks = torch.arange(hidden_size)
                state_dict = model.rnn_layer.state_dict()
                V = state_dict['weight_hh_l0'].to(device)
                
                if os.path.exists(fname):
                    dV_list = torch.load(fname)
                else:
                    dV_list = torch.zeros(len(ranks), seq_length, le_batch_size, hidden_size,hidden_size)
                    for r in tqdm(ranks):
                        for sequence in range(seq_length):
                            dV_list[r, sequence] = torch.bmm(u_list[sequence, :, :, :r+1], torch.matmul(s_list[sequence, :, :r+1].diag_embed(), v_list[sequence, :, :r+1, :]))
                        # print((V.unsqueeze(0).unsqueeze(0).repeat(seq_length, le_batch_size, 1,1).to(device) - dV_list[r,sequence].to(device)).mean(dim = (-1, -2, -3)))
                    torch.save(dV_list, fname)
                
                print((V.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(len(ranks),(seq_length, le_batch_size, 1,1).to(device) - dV_list.to(device)).mean(dim= (-1, -2, -3, -4)))
                criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
                # print(state_dict)
                fname = f'SMNIST/Grads/{fcon.name()}_e{epoch}{it_lab}_baseLosses.p'
                if os.path.exists(fname):
                    base_loss = torch.load(fname)
                else:
                    base_loss = torch.zeros(seq_length, le_batch_size)
                    for sequence in range(seq_length):
                        preds, _ = model(le_input[:, :sequence+1], h_le)
                        temp_loss = criterion(preds, le_target)
                        base_loss[sequence] = temp_loss
                    torch.save(base_loss, fname)

                fname = f'SMNIST/Grads/{fcon.name()}_e{epoch}{it_lab}_rankLosses.p'
                
                if os.path.exists(fname):
                    dV_losses = torch.load(fname).to(device)
                else:
                    
                    dV_losses = torch.zeros(len(ranks), seq_length, le_batch_size).to(device)
                    
                    #Calculate loss reduction from rank-r delta-V's
                    
                    model.eval()
                    for r in tqdm(ranks):
                        for sequence in range(seq_length):
                            V = state_dict['weight_hh_l0'].unsqueeze(0).repeat(le_batch_size, 1, 1)
                            V -= dV_list[r, sequence].to(device)
                            new_state = OrderedDict([(k, V) if k == 'weight_hh_0' else (k, v) for k, v in state_dict.items()])
                            model.rnn_layer.load_state_dict(new_state, strict = False)
                            preds, _ = model(le_input[:, :sequence+1], h_le)
                            temp_loss = criterion(preds, le_target)
                            # print(temp_loss)
                            dV_losses[r, sequence] = temp_loss
                    torch.save(dV_losses, f'SMNIST/Grads/{fcon.name()}_e{epoch}{it_lab}_rankLosses.p')
                    
                rank_differences = dV_losses - base_loss.unsqueeze(0).repeat(len(ranks), 1, 1).to(device)
                # print(base_loss.unsqueeze(0).repeat(len(ranks), 1, 1).shape)
                # print(rank_differences)
                # print((V.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(len(ranks), seq_length, le_batch_size, 1, 1).cpu() - dV_list.cpu()).sum(dim = (-1,-2)))
                # print(f'Mean Difference (batch) {rank_differences.mean(dim = -1)}')
                # print(f'Standard Deviation (batch) {rank_differences.std(dim = -1)}')
            