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

from numpy import cov, sqrt

def calc_entropy(input_tensor, dim= -1):
    lsm = nn.LogSoftmax(dim = dim)
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum(dim = dim)
    return entropy

if __name__ == '__main__':
    print('Starting')
    batch_size = 128
    le_batch_size = 25
    output_size = 10
    max_epoch = 6
    learning_rate = 0.001
    dropout = 0.1
    hidden_size = 128
    save_interval = 1
    model_type = 'rnn'
    p = 0.001
    seq_length = 28
    input_size = 28
    max_sub_epoch = 0
    in_epoch_saves = 4

    #Tasks
    gif = False
    eigs = False
    svds = True
    svd_projections = True
    conditional = False
    corr = False
    confidence = False
    entropy = False
    qangles = False
    dims = False
    ftle = False
    train_loss_plot = False
    le_mode = 'Random'
    plot_proj = True
    svd_mode = 'Last'
    w_change = False

    qs = 10
    evs = 5

    init_type = 'xav_normal'
    init_param = {'gain': 1}
    id_init_params = {'asrnn': 'b', 'rnn': 'gain'}
    nonlinearity = 'tanh'



    print('Configs')
    params = torch.linspace(0.005, end= 0.025, steps = 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    dcon = DataConfig('../Dataset/', input_size = input_size, batch_size= batch_size, input_seq_length = seq_length, 
                                                target_seq_length = 1, val_frac = 0.2, 
                                                test_frac = 0, name_params = {'insize':input_size})
    mcon = ModelConfig(model_type, 1, hidden_size, dcon.input_size, output_size=output_size, dropout=dropout,
                       init_type=init_type, init_params=init_param,
                       device=device, bias=False, id_init_param=id_init_params[model_type], nonlinearity=nonlinearity)
    tcon = TrainConfig(model_dir = 'SMNIST/Models', batch_size = batch_size, max_epoch = max_epoch, 
                                                        optimizer = 'sgd', learning_rate = learning_rate)
    fcon = FullConfig(dcon, tcon, mcon)

    print('Dataloader')
    train_dataloader = torch.utils.data.DataLoader(fcon.data.datasets['train_set'], 
                                                    batch_size = fcon.train.batch_size)
    if in_epoch_saves >0:
        epoch_samples = len(list(train_dataloader))
        save_idcs = part_equal(epoch_samples, in_epoch_saves)
    else:
        save_idcs = []
    
    model = RNNModel(mcon)
    ckpt = load_checkpoint(fcon, load_epoch = 0, overwrite=True)
    model = ckpt[0]
    weightW_list = [model.fc.weight]
    weightW_diff = []
    weightV_list = [model.rnn_layer.state_dict()['weight_hh_l0']]
    weightV_diff = []
    
    print('Loop')
    ymax = 0
    inds = torch.randperm(hidden_size-2*evs)+evs
    all_stds = []
    # epochs = range(0, max_epoch+1, save_interval)
    epochs = [0, 5]
    for p in params[:1]:
        p = float(int(p*1000))*1.0/1000
        if model_type == 'asrnn':
            fcon.model.init_params = {'a': -p, 'b': p}
        elif init_type == 'xav_normal':
            fcon.model.init_params = {'gain': p}
        else:
            fcon.model.init_params = {'mean': 0, 'std': p}
        # fcon.model.init_params['std'] = p
        print(f'Parameter = {p}')
        pearcs = []
        pearcs1 = []
        for epoch in epochs:
            print(f"Epoch {epoch}")
            for it in range(in_epoch_saves + 1):
                    if it == in_epoch_saves:
                        model, optimizer, train_loss, _ = load_checkpoint(fcon, epoch, overwrite=True)
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
                    gradV_list, gradU_list, gradW_list, loss_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_grads.p')
                    pred_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_logits.p')
                    normalizer = colors.Normalize()
                    normalizer.autoscale(loss_list)
                    normalizer2 = colors.Normalize()
                    normalizer2.autoscale(range(le_batch_size))
                    # print(train_loss)
                # print(f"Grads shape: {grad_list.shape}")
                    
                    ftles = ftle_dict['FTLE']
                    qvects = ftle_dict['qvects']
                    rvals = ftle_dict['rvals']
                    print(rvals.shape)
                    if w_change:
                        weightW_list.append(model.fc.weight)
                        # print(model.fc.weight.shape)
                        weightW_diff.append(weightW_list[-1] - weightW_list[-2])
                        weightV_list.append(model.rnn_layer.state_dict()['weight_hh_l0'])
                        weightV_diff.append(weightV_list[-1] - weightV_list[-2])
                    if svds:
                        fname = f'SMNIST/Eigs/{fcon.name()}_e{epoch}{it_lab}_svds.p'
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
                            torch.save((u_list, s_list, v_list), f'SMNIST/Eigs/{fcon.name()}_e{epoch}{it_lab}_svds.p')
                        epoch
                        # print(s_list.sum(dim = -1).shape)
                        s_norm = s_list/s_list.sum(dim = -1).unsqueeze(-1).repeat(1,1,s_list.shape[-1])
                        s_cum = s_norm.cumsum(dim = -1)
                        s_sq = s_list**2
                        s_sq_sum = s_sq.sum(dim = -1)
                        s_sum_squared = (s_list.sum(dim = -1))**2
                        s_part_ratio = s_sq_sum/s_sum_squared
                        
                        co_sim = nn.CosineSimilarity(dim = -2)
                        sims = co_sim(v_list[:-1], v_list[1:])
                        # print(f"sim shape: {sims.shape}")
                        angles = torch.acos(sims)    

                    if qangles:
                        #Find angle between Q vectors
                        co_sim = nn.CosineSimilarity(dim = -1)
                        q_list = ftle_dict['qvects']
                        # print(q_list.shape)
                        qsims = co_sim(q_list[:,:-1], q_list[:, 1:])
                        angles_q = torch.acos(qsims)
                        # print(angles)
                        
                    if entropy:
                        e_list = calc_entropy(pred_list)
                    if entropy and qangles:
                        keep = 1
                        # print(e_list[1:].unsqueeze(-1).repeat((1,1, keep)).shape)
                        # print(angles[:,:,keep].shape)
                        plt.figure()
                        for i in range(keep):
                            plt.scatter(e_list[1:], angles_q[:,:,i], label = i)
                        plt.ylim([0, math.pi])
                        plt.xlabel('Entropy')
                        plt.ylabel('Angle')
                        plt.legend(title = "Q-vector")
                        plt.title(f'Entropy vs. first {keep} Q angles, Epoch {epoch}{it_lab}')
                        # plt.show()
                        plt.savefig(f'SMNIST/Plots/e{epoch}/QAngles{suffix}.png', bbox_inches = 'tight', dpi = 400)

                    if entropy and svds:
                        keep = 1
                        # print(e_list[1:].unsqueeze(-1).repeat((1,1, keep)).shape)
                        # print(angles[:,:,keep].shape)
                        plt.figure()
                        plt.scatter(e_list[1:].unsqueeze(-1).repeat((1,1, keep)), angles[:,:,:keep])
                        plt.xlabel('Entropy')
                        plt.ylabel('Angle')
                        plt.title(f'Entropy vs. first {keep} SV angles, Epoch {epoch}{it_lab}')
                        # plt.show()
                        plt.savefig(f'SMNIST/Plots/e{epoch}/SVDAngles{suffix}.png')

                    if svd_projections:
                        
                        fname = f'SMNIST/Eigs/{fcon.name()}_e{epoch}_projections.p'
                        if os.path.exists(fname):
                            print('Loading Projections')
                            u_projs, v_projs = torch.load(fname)
                        else:						
                            # full_projs = torch.zeros(seq_length, le_batch_size, hidden_size, hidden_size)
                            u_projs = torch.zeros(seq_length, le_batch_size, hidden_size, hidden_size)
                            v_projs = torch.zeros(seq_length, le_batch_size, hidden_size, hidden_size)
                            for i, (Q, G, Vh) in enumerate(zip(qvects.transpose(0,1), gradV_list, v_list)):
                                # full_projs[i] = torch.bmm(G,Q).detach()
                                # print(Q.transpose(-2,-1).shape)
                                # print(Vh.transpose(-2,-1).shape)
                                v_projs[i] = torch.bmm(Q.transpose(-2,-1), Vh.transpose(-2,-1)).detach()
                            print('Saving Projections')
                            torch.save((u_projs, v_projs), fname)
                                
                        std_list = torch.zeros(seq_length, hidden_size)

                        normalizer = colors.Normalize()
                        normalizer.autoscale(range(seq_length))
                        for svd_mode in ['First', 'Last']:
                            plt.figure()
                            print(f'SV mode: {svd_mode}')
                            if svd_mode == 'First':
                                sv_list = torch.arange(evs)
                            if svd_mode == 'Last':
                                sv_list = torch.arange(hidden_size-evs, hidden_size)
                            std_list = torch.std(v_projs[:, :, :, sv_list], dim = (1,3))
                            rand_std = torch.std(v_projs[:, :, inds[:qs]][:,:,:,sv_list])
                            print(f'Random St. Dev = {rand_std}')
                            c_vals = torch.arange(seq_length).unsqueeze(1).repeat(1, hidden_size)
                            x_vals = torch.arange(hidden_size).unsqueeze(0).repeat(seq_length, 1)
                            plt.plot([1, hidden_size], [rand_std, rand_std], color='red')
                            torch.save(std_list, f'SMNIST/svds/std_evs{evs}_{svd_mode}{evs}SingVals_e{epoch}.p')
                            plt.scatter(x_vals, std_list, c = c_vals, norm = normalizer)
                            plt.xlabel('Q Vector Index')
                            plt.ylabel('St. Deviation')
                            plt.title(f'Projection STDs, Epoch {epoch}{it_lab}, {svd_mode} {evs} Sing Vals')
                            plt.colorbar(cm.ScalarMappable(norm = normalizer), label = 'Step No.')
                            # plt.show()
                            all_stds.append(std_list.mean(dim =0))
                            plt.savefig(f'SMNIST/Plots/p{p}/p{p}_e{epoch}{suffix}_{svd_mode}{evs}_STDs.png', bbox_inches = 'tight', dpi = 400)

                            if plot_proj:
                                plot_steps = False
                                for le_mode in ['First', 'Last', 'Random']:
                                    print(f'LE Mode: {le_mode}')
                                    hist_fname = f'SMNIST/Eigs/{fcon.name()}_e{epoch}_{le_mode}_{svd_mode}_hist.p'
                                    proj_fname = f'SMNIST/Eigs/{fcon.name()}_e{epoch}_{le_mode}_{svd_mode}_stds.p'
                                    # Generate Singular Vector Indices
                                    if svd_mode == 'First':
                                        sv_list = torch.arange(evs)
                                    if svd_mode == 'Last':
                                        sv_list = torch.arange(hidden_size - evs, hidden_size)
                                    if svd_mode == 'Random':
                                        sv_list = inds[:evs]

                                    # Calculate Projections
                                    if os.path.exists(proj_fname):
                                        all_hist = torch.load(hist_fname)
                                        stds = torch.load(proj_fname)
                                    else:
                                        stds = torch.zeros(seq_length, evs)
                                        filenames = []
                                        all_hist = []
                                        for step in tqdm(range(seq_length)):
                                            h_svd_full = torch.zeros((evs, 100))
                                            if plot_steps:
                                                plt.figure()
                                            for idx, i in enumerate(sv_list):
                                                if le_mode == 'First':
                                                    v_red = v_projs[step,:, :qs, i]
                                                elif le_mode == 'Last':
                                                    v_red = v_projs[step,:, -qs:, i]
                                                elif le_mode == 'Random':
                                                    v_red = v_projs[step,:, inds[:qs], i]
                                                else:
                                                    print(f'LE Mode {le_mode} not recognized')
                                                    break
                                                std = torch.std(v_red)
                                                stds[step, idx] = std
                                                h = torch.histc(v_red, min = -1, max = 1)
                                                if plot_steps:
                                                    plt.bar(torch.linspace(-1,1,101)[:-1], h, bottom = h_svd_full.sum(dim=0), width = 0.02, label = i.item()+1)
                                                h_svd_full[idx] = h
                                            all_hist.append(h_svd_full)
                                            if plot_steps:
                                                ymax = max(ymax, h_svd_full.sum(dim=0).max().item())
                                                plt.legend(title = 'Grad SV')
                                                plt.ylim([0,ymax])
                                                plt.ylabel('Count')
                                                plt.xlabel('Alignment')
                                                plt.title(f'Epoch {epoch}{it_lab}, Step {step} \n {svd_mode} {evs} Grad SVs, {le_mode} {qs} Q-vects, Std = {std:.4f}')
                                                fname = f'SMNIST/Plots/p{p}/SVDAlignment_{le_mode}q{qs}_ev{evs}_e{epoch}{suffix}_step{step}.png'
                                                filenames.append(fname)
                                                plt.savefig(fname, bbox_inches='tight', dpi=400)
                                                plt.close()
                                    # with imageio.get_writer(f'SMNIST/Plots/p{p}/SVDAlignment_p{p}_e{epoch}{suffix}_{le_mode}LEs_{svd_mode}SVs.gif', mode='I', fps = 4) as writer:
                                    #     for filename in filenames:
                                    #         image = imageio.imread(filename)
                                    #         writer.append_data(image)
                                        for filename in set(filenames):
                                            os.remove(filename)
                                        all_hist = torch.stack(all_hist)
                                        torch.save(all_hist, hist_fname)
                                        torch.save(stds, proj_fname)

                                    # Plot full histogram across all time steps

                                    plt.figure(figsize = (4,2))
                                    all_hist_plot = all_hist.sum(dim=0)
                                    print(f'Mean Alignment std: {stds.mean()}')
                                    for idx, i in enumerate(sv_list):
                                        plt.bar(torch.linspace(-1, 1, 101)[:-1], all_hist_plot[idx],
                                                bottom=all_hist_plot[:idx].sum(dim=0), width=0.02,
                                                label=i.item() + 1)
                                    plt.legend(title='Grad SV')
                                    plt.ylim([0, all_hist_plot.sum(dim=-1).max()])
                                    plt.ylabel('Count')
                                    plt.xlabel('Alignment')
                                    plt.title(
                                        f'Epoch {epoch}{it_lab}, All Steps \n {svd_mode} {evs} Grad SVs, {le_mode} {qs} Q-vects, Std = {stds.mean():.4f}')
                                    fname = f'SMNIST/Plots/p{p}/SVDAlignment_{le_mode}q{qs}_ev{evs}_{svd_mode}{evs}SVs_e{epoch}{suffix}_All_steps.png'
                                    plt.savefig(fname, bbox_inches='tight', dpi=400)
                                    plt.close()

                        if svds and dims:
                            filenames = []
                            # for step in range(seq_length):
                                # plt.figure()
                                # h = torch.histc(s_part_ratio[step,:], min = 0, max = 1, bins = 10)
                                # plt.bar(torch.linspace(0, 1, 10), h, width = 0.1)
                                # ymax = h.max().item()
                                # plt.ylim([0,25])
                                # plt.xlim([0, 1])
                                # plt.ylabel('Count')
                                # plt.xlabel('Partipation Ratio')
                                # if epoch > 0:
                                    # loss_append = f'\n Training Loss = {train_loss[-1]:.3f}'
                                # else:
                                    # loss_append = ''
                                # plt.title(f'Epoch {epoch}{it_lab}, Step {step}{loss_append}')
                                # fname = f'SMNIST/Plots/e{epoch}/{fcon.name()}_e{epoch}{suffix}_step{step}_svdDim.png'
                                # filenames.append(fname)
                                # plt.savefig(fname, bbox_inches = 'tight', dpi = 400)
                                # plt.close()

                            plt.figure()
                            fname = f'SMNIST/Plots/e{epoch}/p{p}_e{epoch}{suffix}_svdRatio_mean.png'
                            y= s_part_ratio.mean(dim = -1)
                            err = s_part_ratio.std(dim = -1)
                            plt.errorbar(torch.arange(1, seq_length+1), y, yerr=err, fmt='o')
                            plt.title(f'Participation Ratio, Epoch {epoch}{it_lab}')
                            plt.xlabel('Step No.')
                            plt.ylabel('Partipation Ratio Mean')
                            plt.ylim([0,1])
                            plt.savefig(fname, bbox_inches = 'tight', dpi = 400)
                    if ftle:
                        f_dim = torch.sum(ftles>-1, dim = -1).cpu()
                        # print(f_dim.shape)
                        filenames = []
                        for step in range(seq_length):
                            plt.figure()
                            h = torch.histc(f_dim[:,step]*1.0, min = 0, max = hidden_size, bins = hidden_size)
                            plt.bar(torch.linspace(1,hidden_size+1, hidden_size), h, width = 1)
                            ymax = h.max().item()
                            plt.ylim([0,25])
                            plt.xlim([0, 25])
                            plt.title(f'Epoch {epoch}, Step {step} \n FTLE Dimension')
                            fname = f'SMNIST/Plots/e{epoch}/{fcon.name()}_e{epoch}{suffix}_step{step}_FTLEDim.png'
                            filenames.append(fname)
                            plt.savefig(fname)
                            plt.close()
                            
                        with imageio.get_writer(f'SMNIST/Plots/e{epoch}/FTLEDim_e{epoch}{suffix}.gif', mode='I', fps = 4) as writer:
                            for filename in filenames:
                                image = imageio.imread(filename)
                                writer.append_data(image)
                        for filename in set(filenames):
                            os.remove(filename)
                    if eigs:
                        # ev_list = torch.zeros(seq_length, le_batch_size, hidden_size,hidden_size, dtype = torch.cfloat)
                        # eig_list = torch.zeros(seq_length, le_batch_size, hidden_size)
                        # for i, M in tqdm(zip(range(seq_length), grad_list)):
                            # e, v = linalg.eig(M)
                            # ev_list[i] = v
                            # eig_list[i] = e
                        
                        # torch.save((ev_list, eig_list), f'SMNIST/Eigs/{fcon.name()}_e{epoch}_eigs.p')
                        ev_list, eig_list = torch.load(f'SMNIST/Eigs/{fcon.name()}_e{epoch}_eigs.p')
                        full_projs = torch.zeros(seq_length, le_batch_size, hidden_size, hidden_size)
                        ev_projs = torch.zeros(seq_length, le_batch_size, hidden_size, hidden_size)
                        for i, (Q, G, V) in enumerate(zip(qvects.transpose(0,1), gradV_list, ev_list)):
                            full_projs[i] = torch.bmm(G,Q).detach()
                            ev_projs[i] = torch.bmm(V.transpose(-2,-1), torch.complex(Q, torch.zeros_like(Q))).detach()
                    if confidence:
                        ev_trunc = ev_projs[:, :, :evs, :qs]
                        c = torch.zeros_like(ev_trunc)
                        logits = f.softmax(pred_list, dim = -1).max(dim = -1)[0]
                        print(logits)
                        for seq in range(seq_length):
                            for b in range(le_batch_size):
                                c[seq, b] = torch.ones(ev_trunc.shape[-2:])*logits[seq, b]
                        
                        C = cov(c.flatten(), ev_trunc.flatten())
                        P = C[0,1]/(sqrt(C[1,1]*C[0,0]))
                        print(cov(c.flatten(), ev_trunc.abs().flatten()))
                        C1 = cov(c.flatten(), ev_trunc.abs().flatten())
                        P1 = C1[0,1]/(sqrt(C1[1,1]*C1[0,0]))
                        print(f'Pearsons: {P}')
                        print(f'Pearsons1: {P1}')
                        pearcs.append(P)
                        pearcs1.append(P1)
                        plt.figure()
                        plt.scatter(ev_trunc.flatten(), c.flatten(), c= 'b', alpha = 0.5)
                        plt.ylabel('confidence')
                        plt.xlabel('alignment')
                        plt.title(f'Epoch {epoch}{it_lab}, EV = {evs} Q-vects = {qs}')
                        plt.savefig(f'SMNIST/Plots/AlVConf_ev{evs}_qs{qs}_e{epoch}{suffix}.png', bbox_inches=  'tight')
                        plt.close()
                    if conditional:
                        # print(pred_list[:, 0,:])
                        c_scores = f.softmax(pred_list, dim=-1).max(dim = -1)
                        bins = [0, 0.4, 0.7, 0.8, 0.9, 0.95, 1]
                        con_filter = torch.zeros((len(bins)-1, seq_length, le_batch_size),  dtype = torch.bool)
                        for i in range(len(bins)-1):
                            con_filter[i] = (c_scores[0]>bins[i])*(c_scores[0]<bins[i+1])
                        # print(sum(con_filter.flatten()))
                        # print(torch.where(c_scores[0] <0.4))
                        # print(torch.where(con_filter[0]))
                        filenames = []
                        for i in range(len(bins)-1):
                            plt.figure()
                            filt = con_filter[i]
                            # print(sum(filt.flatten()))
                            x = ev_projs.flatten(start_dim = 0, end_dim = 1)[filt.flatten()]
                            # print(x.shape)
                            if x.shape[0]>0:
                                h_ev_full = torch.zeros((evs, 100))
                                for j in range(evs):
                                    h = torch.histc(x[:, j, :qs], min = -1, max = 1)
                                    # print(sum(h.flatten()))
                                    h_ev_full[j] = h
                                # print(h_ev_full.shape)
                                max_h = h_ev_full.sum(dim=0).flatten().max()
                                # print(max_h)
                                for j in range(evs):
                                    plt.bar(torch.linspace(-1,1,101)[:-1], h_ev_full[j]/max_h, bottom = h_ev_full[:j].sum(dim=0)/max_h, width = 0.02, label = j)
                            ymax = max(ymax, h_ev_full.sum(dim=0).max().item())
                            # plt.legend(title = 'Grad EV')
                            if epoch == 0:
                                plt.ylim([0,1])
                            else:
                                plt.ylim([0, 1])
                            plt.title(f'Epoch {epoch}{it_lab}, Logit {bins[i]} - {bins[i+1]} \n First {evs} Grad Evs, First {qs} Q-vects')
                            filenames.append(f'SMNIST/Plots/e{epoch}/{fcon.name()}_e{epoch}{suffix}_CondAlignment_q{qs}_ev{evs}_max{bins[i+1]}.png')
                            plt.savefig(f'SMNIST/Plots/e{epoch}/{fcon.name()}_e{epoch}_CondAlignment_q{qs}_ev{evs}_max{bins[i+1]}.png')
                            plt.close()
                        with imageio.get_writer(f'SMNIST/Plots/e{epoch}/CondAlignment_e{epoch}{suffix}.gif', mode='I', fps = 1) as writer:
                            for filename in filenames:
                                image = imageio.imread(filename)
                                writer.append_data(image)
                        for filename in set(filenames):
                                os.remove(filename)
                    if gif:
                        filenames = []
                        for step in range(seq_length):
                            h_ev_full = torch.zeros((evs, 100))
                            plt.figure()
                            for i in range(evs):
                                h = torch.histc(ev_projs[step,:, i, :qs], min = -1, max = 1)
                                plt.bar(torch.linspace(-1,1,101)[:-1], h, bottom = h_ev_full.sum(dim=0), width = 0.02, label = i+1)
                                h_ev_full[i] = h
                            ymax = max(ymax, h_ev_full.sum(dim=0).max().item())
                            plt.legend(title = 'Grad EV')
                            plt.ylim([0,ymax])
                            plt.title(f'Epoch {epoch}{it_lab}, Step {step} \n First {evs} Grad Evs, First {qs} Q-vects')
                            filenames.append(f'SMNIST/Plots/e{epoch}/{fcon.name()}_e{epoch}{suffix}_step{step}_Alignment_q{qs}_ev{evs}.png')
                            plt.savefig(f'SMNIST/Plots/e{epoch}/{fcon.name()}_e{epoch}{suffix}_step{step}_Alignment_q{qs}_ev{evs}.png')
                            plt.close()
                            
                        with imageio.get_writer(f'SMNIST/Plots/e{epoch}/Alignment_e{epoch}{suffix}.gif', mode='I', fps = 4) as writer:
                            for filename in filenames:
                                image = imageio.imread(filename)
                                writer.append_data(image)
                        for filename in set(filenames):
                                os.remove(filename)
        if w_change:
            l = len(weightW_list)
            diff_normW = [torch.linalg.norm(dW, 2).item() for dW in weightW_diff]
            diff_normV = [torch.linalg.norm(dV, 2).item() for dV in weightV_diff]
            x1 = torch.linspace(1.0/in_epoch_saves, max_sub_epoch, (in_epoch_saves+2)*(max_sub_epoch-1))
            x2 = torch.arange(start = max_sub_epoch, end = max_epoch + 1)
            x = torch.cat((x1, x2))
            plt.xlabel('Epoch')
            plt.xticks(torch.arange(1, 11))
            plt.plot(x[1:], diff_normW[1:], label = 'W')
            plt.plot(x[1:], diff_normV[1:], label = 'V')
            plt.legend()
            plt.xlabel('Training Epoch')
            plt.ylabel(r'2-Norm')
            plt.title(r'2-Norm of $\Delta W$ and $\Delta V$ over training')
            plt.savefig(f'SMNIST/Plots/p{p}_DeltaNorms.png', bbox_inches = 'tight', dpi = 400)
        if corr:
            plt.figure()
            plt.plot(range(0, 11, 2), pearcs, label = 'base')
            plt.plot(range(0, 11, 2), pearcs1, label = 'abs')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Pearson Coefficient')
            plt.title(f'Pearson, EV = {evs}, Q-vects = {qs}')
            plt.savefig(f'SMNIST/Plots/Pearson_ev{evs}_qs{qs}.png', bbox_inches=  'tight')
            # plt.show()
        # if svd_projections:
        #     plt.figure()
        #     x1 = torch.linspace(1.0/in_epoch_saves, max_sub_epoch, (in_epoch_saves+2)*max_sub_epoch)
        #     x2 = torch.arange(start = max_sub_epoch, end = max_epoch+1)
        #     x = torch.cat((x1, x2))
        #     x_vals = x.unsqueeze(0).repeat(hidden_size, 1)
        #     normalizer = colors.Normalize()
        #     normalizer.autoscale(range(max_epoch))
        #     c_vals = x.unsqueeze(1).repeat(1, hidden_size)
        #     # print(x.shape)
        #     b = torch.Tensor(x.shape[0], hidden_size)
        #     torch.vstack(all_stds, out=b)
        #     plt.scatter(x.unsqueeze(1).repeat((1, hidden_size)), b, c= c_vals, norm = normalizer2)
        #     plt.xticks(torch.arange(1, max_epoch+1))
        #     plt.legend()
        #     plt.xlabel('Training Epoch')
        #     plt.savefig(f'SMNIST/Plots/p{p}/STDs.png', bbox_inches=  'tight')
        #     # plt.show()
            
            
        if train_loss_plot:
            _, _, train_loss, _ = load_checkpoint(fcon, max_epoch)
            x1 = torch.linspace(1.0/in_epoch_saves, max_sub_epoch, (in_epoch_saves+2)*max_sub_epoch)
            x2 = torch.arange(start = max_sub_epoch, end = max_epoch)
            print(x1.shape)
            print(x2.shape)
            x = torch.cat((x1, x2))
            plt.plot(x, train_loss)
            plt.xlabel('Epoch')
            plt.xticks(torch.arange(1, 11))
            plt.ylabel('Train Loss')
            plt.title('Training Loss')
            plt.savefig(f'SMNIST/Plots/p{p}_trainLoss.png')
            print('test')