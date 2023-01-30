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

if __name__ == '__main__':
    batch_size = 128
    le_batch_size = 25
    output_size = 10
    max_epoch = 10
    learning_rate = 0.002
    dropout = 0.1
    hidden_size = 128
    save_interval = 1
    model_type = 'rnn'
    p = 0.001
    seq_length = 28
    input_size = 28
    in_epoch_saves = 4
    
    #Tasks
    gif = True
    eigs = False
    svds = True
    svd_projections = False
    conditional = False
    corr = False
    confidence = False
    entropy = True
    qangles = False
    dims = True
    
    qs = 10
    evs = 5
    
    
    params = torch.linspace(0.005, end= 0.025, steps = 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    
    
    for std in [0.01, 0.1, 1]:
        ymax = 0
        rand1 = torch.empty((seq_length, le_batch_size, hidden_size, hidden_size)).normal_(mean = 0, std = std)
        rand2 = torch.empty((seq_length, le_batch_size, hidden_size, hidden_size)).normal_(mean = 0, std = std)
        
        if gif:
            filenames = []
            for step in range(seq_length):
                u1, s1, v1 = linalg.svd(rand1[step])
                u2, s2, v2 = linalg.svd(rand2[step])
                projs = torch.bmm(v1, v2)
                h_full = torch.zeros((evs, 100))
                plt.figure()
                dist_std = torch.std(projs)
                for i in range(evs):
                    h = torch.histc(projs[:, i, :qs], min = -1, max = 1)
                    plt.bar(torch.linspace(-1,1,101)[:-1], h, bottom = h_full.sum(dim=0), width = 0.02, label = i+1)
                    h_full[i] = h
                ymax = max(ymax, h_full.sum(dim=0).max().item())
                # plt.legend(title = '')
                plt.ylim([0,ymax])
                plt.title(f'Random Vector Alignment, Rand Std = {std}, Dist Std = {dist_std:.4f}')
                filenames.append(f'SMNIST/Plots/Random/Alignment_std{std}_step{step}.png')
                plt.savefig(f'SMNIST/Plots/Random/Alignment_std{std}_step{step}.png')
                plt.close()
                
            with imageio.get_writer(f'SMNIST/Plots/Random/Alignment_std{std}.gif', mode='I', fps = 4) as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            for filename in set(filenames):
                    os.remove(filename)