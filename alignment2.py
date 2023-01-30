import torch
from torch import linalg
from config import *
from lyapunov import *
from models import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors, cm

if __name__ == '__main__':
    batch_size = 128
    le_batch_size = 25
    output_size = 10
    max_epoch = 10
    learning_rate = 0.002
    dropout = 0.1
    hidden_size = 128
    save_interval = 2
    model_type = 'rnn'
    p = 0.001
    seq_length = 28
    input_size = 28

    params = torch.linspace(0.005, end= 0.025, steps = 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    dcon = DataConfig('../Dataset/', batch_size= batch_size, input_seq_length = seq_length, 
                                                target_seq_length = 1, val_frac = 0.2, 
                                                test_frac = 0, name_params = {'insize':input_size})
    mcon = ModelConfig(model_type, 1, hidden_size, dcon.input_size, output_size = output_size, dropout=dropout, 
                        init_type = 'normal', init_params = {'mean':0, 'std':p},
                        device = device, bias = False, id_init_param = 'std')                                            
    tcon = TrainConfig(model_dir = 'SMNIST/Models', batch_size = batch_size, max_epoch = max_epoch, 
                                                        optimizer = 'adam', learning_rate = learning_rate)
    fcon = FullConfig(dcon, tcon, mcon)
    
    epoch = 4
    for p in params[:1]:
        p = float(int(p*1000))*1.0/1000
        fcon.model.init_params['std'] = p
        print(f'Parameter = {p}')
        for epoch in range(0, max_epoch+1, 2):
            print(f"Epoch {epoch}")
            ftle_dict = torch.load(f'SMNIST/LEs/{fcon.name()}_e{epoch}_FTLE.p')
            grad_list, loss_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}_grads.p')
            normalizer = colors.Normalize()
            normalizer.autoscale(pred_list)
            # print(f"Grads shape: {grad_list.shape}")
            
            ftles = ftle_dict['FTLE']
            qvects = ftle_dict['qvects']
            # print(qvects.transpose(0,1).shape)
            ev_list = torch.zeros(seq_length, le_batch_size, hidden_size,hidden_size)
            eig_list = torch.zeros(seq_length, le_batch_size, hidden_size)
            for i, M in tqdm(zip(range(seq_length), grad_list)):
                e, v = linalg.eig(M)
                ev_list[i] = v
                eig_list[i] = e
            
            full_projs = torch.zeros(seq_length, le_batch_size, hidden_size, hidden_size)
            ev_projs = torch.zeros(seq_length, le_batch_size, hidden_size, hidden_size)
            for i, (Q, G, V) in enumerate(zip(qvects.transpose(0,1), grad_list, ev_list)):
                full_projs[i] = torch.bmm(G,Q).detach()
                ev_projs[i] = torch.bmm(V, Q).detach()
            
            fig, ax = plt.subplots()
            # scatter = ax.scatter(ev_projs[:, 0, 0, 0], ev_projs[:, 0, 0, 1], c = loss_list[:,0].detach())
            scatter = ax.scatter(ev_projs[:, 0, 0, 0], ev_projs[:, 0, 0, 1], c = range(28))
            normalizer2 = colors.Normalize()
            normalizer2.autoscale(range(28))
            cmap2 = cm.ScalarMappable(norm = normalizer2)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            cb = fig.colorbar(cmap2)
            cb.set_label("Step")
            # cb.set_label("Loss Gradient")
            print(loss_list)
            
            def plot(a, data, loss_list):
                scatter.set_offsets(data[:,a,0,:2])
                # scatter.set_facecolor(cmap.to_rgba(loss_list[:,a].detach()))
                scatter.set_facecolor(cmap2.to_rgba(range(28)))
                ax.set_title(f'Sample No. {a+1}')
                ax.set_xlabel('Gradient EV 1')
                ax.set_ylabel('Gradient EV 2')
                return scatter
            anim = FuncAnimation(fig, plot, fargs = (ev_projs,loss_list), interval=500, frames = le_batch_size)
            anim.save(f"Grad Projection Sample, std = {p}, epoch = {epoch}.mp4")
            
            # for step in [0, 10, 25]:
                # f = plt.figure()
                # for j in range(5):
                    # plt.scatter(ev_projs[step, j, 0, 0], ev_projs[step, j, 0, 1])
                # plt.title(f"Sequence Step {step}")
                # plt.xlabel("Grad EV 1")
                # plt.ylabel("Grad EV 2")
                # plt.savefig(f"Grad Projection, std = {p}, epoch = {epoch}, Step {step},.png")
                # plt.close()
                
                
    

    