import torch
from torch import linalg
from torch.nn import functional as f
from config import *
from lyapunov import *
from models import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors, cm
import imageio

from numpy import cov, sqrt

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
    
    qs = 5
    evs = 10
    
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
    
    ymax = 0
    for p in params[:1]:
        p = float(int(p*1000))*1.0/1000
        fcon.model.init_params['std'] = p
        print(f'Parameter = {p}')
        # for epoch in range(0, max_epoch+1, 2):
        for epoch in [0, 10]:
            print(f"Epoch {epoch}")
            ftle_dict = torch.load(f'SMNIST/LEs/{fcon.name()}_e{epoch}_FTLE.p')
            grad_list, loss_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}_grads.p')
            pred_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}_logits.p')
            normalizer = colors.SymLogNorm(linthresh=0.000001, linscale=0.000001,
                                              base=10, vmax = 1, vmin = -1)
            normalizer.autoscale(pred_list)
            
            print(grad_list.shape)
            filenames = []
            # cmap1 = cm.ScalarMappable(norm = normalizer, cmap = 'magma')
            for step in range(seq_length):
                plt.figure()
                plt.pcolor(grad_list[step, 0], cmap = 'magma', norm = normalizer)
                plt.colorbar()
                plt.title(f'Epoch {epoch}, Step {step+1}')
                filenames.append(f'SMNIST/Plots/e{epoch}/{fcon.name()}_e{epoch}_step{step}_gradmap.png')
                plt.savefig(f'SMNIST/Plots/e{epoch}/{fcon.name()}_e{epoch}_step{step}_gradmap.png')
                plt.close()
                
            with imageio.get_writer(f'SMNIST/Plots/e{epoch}/Heatmap_e{epoch}.gif', mode='I', fps = 1) as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)