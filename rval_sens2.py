import os

import matplotlib.pyplot as plt
import torch
from config import *
import numpy as np
from models import *
from tqdm import tqdm
from training import *
import time
import imageio
import torchvision as T
import argparse
# from sMNIST import Model#, test_model
from utils import select_network, select_optimizer
from SMNIST_FTLE import FTLE

## Args
parser = argparse.ArgumentParser(description='auglang parameters')

parser.add_argument('--net-type', type=str, default='RNN',
                    choices=['RNN', 'nnRNN', 'LSTM', 'expRNN'],
                    help='options: RNN, nnRNN, expRNN, LSTM')
parser.add_argument('--nhid', type=int,
                    default=512,
                    help='hidden size of recurrent net')
parser.add_argument('--cuda', action='store_true',
                    default=False, help='use cuda')
parser.add_argument('--random-seed', type=int,
                    default=400, help='random seed')
parser.add_argument('--permute', action='store_true',
                    default=True, help='permute the order of sMNIST')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--save-freq', type=int,
                    default=50, help='frequency to save data')
parser.add_argument('--batch', type=int, default=100)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_orth', type=float, default=2e-5)
parser.add_argument('--optimizer', type=str, default='RMSprop',
                    choices=['Adam', 'RMSprop'],
                    help='optimizer: choices Adam and RMSprop')
parser.add_argument('--alpha', type=float,
                    default=0.99, help='alpha value for RMSprop')
parser.add_argument('--betas', type=tuple,
                    default=(0.9, 0.999), help='beta values for Adam')

parser.add_argument('--rinit', type=str, default="xavier",
                    choices=['random', 'cayley', 'henaff', 'xavier'],
                    help='recurrent weight matrix initialization')
parser.add_argument('--iinit', type=str, default="xavier",
                    choices=['xavier', 'kaiming'],
                    help='input weight matrix initialization')
parser.add_argument('--nonlin', type=str, default='tanh',
                    choices=['none', 'modrelu', 'tanh', 'relu', 'sigmoid'],
                    help='non linearity none, relu, tanh, sigmoid')
parser.add_argument('--alam', type=float, default=0.0001,
                    help='decay for gamma values nnRNN')
parser.add_argument('--Tdecay', type=float,
                    default=0, help='weight decay on upper T')

args = parser.parse_args()


def inverse_permutation(perm):
    perm_tensor = torch.LongTensor(perm)
    inv = torch.empty_like(perm_tensor)
    inv[perm_tensor] = torch.arange(perm_tensor.size(0), device=perm_tensor.device)
    return inv


lr = args.lr
lr_orth = args.lr_orth
random_seed = args.random_seed
NET_TYPE = args.net_type
CUDA = args.cuda
SAVEFREQ = args.save_freq
inp_size = 1
alam = args.alam
Tdecay = args.Tdecay
permute = True

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
# device = torch.device('cpu')
le_batch_size = 15
hidden_size = args.nhid
output_size = 10

torch.cuda.manual_seed(args.random_seed)
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

if permute:
    rng = np.random.RandomState(1234)
    order = rng.permutation(784)
else:
    order = np.arange(784)

inverse_order = inverse_permutation(order)

class Model(nn.Module):
    def __init__(self, hidden_size, rnn, eval=True):
        super(Model, self).__init__()
        self.rnn_layer = rnn
        self.hidden_size = hidden_size
        self.lin = nn.Linear(hidden_size, 10)
        self.loss_func = nn.CrossEntropyLoss()
        self.evaluate = eval

    def forward(self, inputs, y=None, order=None, h=None):
        inputs = inputs[:, order]
        inputs = inputs.squeeze(dim=1)
        for input in torch.unbind(inputs, dim=1):
            h = self.rnn_layer(input.view(-1, inp_size), h)
        out = self.lin(h)

        if self.evaluate:
            loss = self.loss_func(out, y)
            preds = torch.argmax(out, dim=1)
            correct = torch.eq(preds, y).sum().item()
            return loss, correct
        else:
            return out, h


def test_model(net, dataloader, order):
    accuracy = 0
    loss = 0
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            x, y = data
            x = x.view(-1, 784)
            if CUDA:
                x = x.cuda()
                y = y.cuda()
            if NET_TYPE == 'LSTM':
                net.rnn_layer.init_states(x.shape[0])
            loss, c = net.forward(x, y, order)
            accuracy += c
    accuracy /= len(testset)
    return loss, accuracy


# h_0 = torch.randn(1, le_batch_size, args.nhid).to(device)
h_0 = torch.zeros(1, le_batch_size, args.nhid).to(device)
# targets = testset.targets[:trunc].view(le_batch_size, -1, 1)

le_loader_fname = f'SMNIST/le_loader_b{le_batch_size}.p'
if os.path.exists(le_loader_fname):
    print('Loading Dataloader')
    le_loader, le_input, le_target = torch.load(le_loader_fname)
else:
    print('Saving Dataloader')
    testset = T.datasets.MNIST(root='./MNIST',
                               train=False,
                               download=True,
                               transform=T.transforms.ToTensor())
    le_input = testset.data[:le_batch_size].view(-1, 784, inp_size) / 255
    le_target = testset.targets[:le_batch_size].view(-1)
    le_dataset = torch.utils.data.TensorDataset(le_input, le_target)

    le_loader = torch.utils.data.DataLoader(le_dataset,
                                            batch_size=le_batch_size)
    torch.save((le_loader, le_input, le_target), le_loader_fname)

# testloader = torch.utils.data.DataLoader(testset,
#                                          batch_size=args.batch,
#                                          num_workers=2)

rnn = select_network(args, inp_size)
rnn.batch_first = True
model = Model(hidden_size, rnn).to(device)
# model.rnn_layer.batch_first = False
epoch = 50
model_name = f'RNN_{epoch}.pth.tar'
best_state = torch.load(f'SMNIST/Models/{model_name}')
model.load_state_dict(best_state['state_dict'])

# model2 = Model2(hidden_size, rnn).to(device)
# model2.load_state_dict(best_state['state_dict'])

# loss1, correct1 = model2(le_input, le_target, order, h=h_0, batch_size=le_batch_size)
# loss2, accuracy2 = test_model(model, le_loader)
# loss3, accuracy3 = test_model(model, testloader, order)
model.evaluate = False
# out_preds, h_preds = model(le_input, h=h_0[0])

# loss3, accuracy3 = test_model(model, testloader, h=h_0[:, :testloader.batch_size])

# print(f'Loss 1: {loss2}')
# print(f'Loss 2: {loss3}')

hidden_size = 512
input_seq_length = 784
grads = True
r_plot = False
r_thresh = True
calc_grads = False
load_grads = True
le_load = True

if permute:
    perm_suffix = '_permuted'
    perm_plot = ', Permuted'
else:
    perm_suffix = ''
    perm_plot = ''
fname = f'SMNIST/ftle_dict_{args.nonlin}{perm_suffix}_b{le_batch_size}_e{epoch}.p'

model.evaluate = False

if os.path.exists(fname) and le_load:
    print('Loading FTLEs')
    ftle_dict = torch.load(fname, map_location=device)
    print('FTLES Loaded')
else:
    print('Calculating FTLEs')
    ftle_dict = FTLE(le_input[:, order], h_0, model=model, rec_layer='rnn')
    torch.save(ftle_dict, fname)

if grads:
    grads_fname = f'SMNIST/Grads/trainedRNN_grads{perm_suffix}_e{epoch}.p'
    logits_fname = f'SMNIST/Grads/trainedRNN_logits{perm_suffix}_e{epoch}.p'
    if os.path.exists(logits_fname) and load_grads:
        if calc_grads:
            gradV_list, gradU_list, gradW_list, loss_list = torch.load(grads_fname)
        pred_list = torch.load(logits_fname)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        h_le = h_0
        # le_target =
        print("Calculating Gradients")
        if calc_grads:
            gradV_list = torch.zeros(input_seq_length, le_batch_size, hidden_size,
                                     hidden_size)
            gradW_list = torch.zeros(input_seq_length, le_batch_size, hidden_size, output_size)
            gradU_list = torch.zeros(input_seq_length, le_batch_size, hidden_size,
                                     inp_size)
        loss_list = torch.zeros(input_seq_length, le_batch_size)
        pred_list = torch.zeros(input_seq_length, le_batch_size, output_size)
        # h_list = [h_0.squeeze()]
        grad_dict = {}
        layer_names = ['W', 'U', 'b_i', 'b_h']
        model.train()
        for t in tqdm(torch.arange(1, input_seq_length + 1)):
            preds, h = model(le_input[:, order][:, :t], h_0.squeeze())
            # h_list.append(h)
            loss = criterion(preds, le_target.squeeze())
            loss_list[t - 1] = loss.detach()
            pred_list[t - 1] = preds.detach()
            if calc_grads:
                for i in range(le_batch_size):
                    optimizer.zero_grad()
                    loss[i].backward(retain_graph=True)
                    for layer, name in zip(model.rnn_layer.parameters(), layer_names):
                        grad_dict[name] = layer.grad
                    gradV_list[t - 1, i] = grad_dict['U']
                    gradU_list[t - 1, i] = grad_dict['W']
                    # temp = model.fc.parameters()[0].grad
                    # print(temp.shape)
                    gradW_list[t - 1, i] = model.lin.weight.grad.t()
                torch.save((gradV_list, gradU_list, gradW_list, loss_list),
                           grads_fname)
        torch.save(pred_list, logits_fname)

if r_plot:
    torch.manual_seed(31)
    plot_dir = f'SMNIST/Plots/rplot'
    # le_target = targets[:, 0].squeeze()
    # le_input = x_set[:, 0]
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    suffix = ''
    # print(f'Best Loss: {model.best_loss}')
    # ftle_dict = torch.load(f'SMNIST/LEs/{fcon.name()}_e{epoch}{suffix}_FTLE.p', map_location=device)
    # pred_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_logits.p', map_location=device)

    # pred_logits = pred_list.softmax(dim=-1)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    best_idx = torch.argmin(criterion(pred_list[-1], le_target))
    plt.figure()
    plt.pcolor(le_input.cpu()[best_idx].reshape((28, 28)))
    plt.gca().invert_yaxis()
    plt.title(f'Input for RNN, trained')
    plt.savefig(f'{plot_dir}/input_plot_trainedRNN_h{hidden_size}_len{input_seq_length}.png', bbox_inches='tight')

    plt.figure(figsize=(4, 1))
    plt.pcolor(le_input.cpu()[best_idx].T)
    plt.gca().invert_yaxis()
    plt.savefig(f'{plot_dir}/input_plot_sequence_trainedRNN_h{hidden_size}_len{input_seq_length}.png',
                bbox_inches='tight')
    plt.figure()
    plt.plot(pred_list[-1, best_idx].softmax(dim=-1).cpu())
    plt.xlabel('Label')
    plt.ylabel('Logit')
    plt.title(f'Class Logits for trained RNN, seq len = {input_seq_length}\n Correct label: {le_target[best_idx]}')
    plt.savefig(f'{plot_dir}/classLogits_trainedRNN_h{hidden_size}_len{input_seq_length}.png',
                bbox_inches='tight')
    plt.close()
    filenames = []

    plt.figure()
    plt.plot(pred_list[:, best_idx].cpu().softmax(dim=-1)[:, le_target[best_idx]])
    plt.xlabel('t')
    plt.ylabel('Correct Logit')
    plt.savefig(f'{plot_dir}/logits_plot_trainedRNN_h{hidden_size}_len{input_seq_length}.png',
                bbox_inches='tight')
    plt.close()

    r_vals = ftle_dict['rvals']
    r_diag = r_vals[best_idx]
    ns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n = ns[-1]
    plot_min = (torch.log(r_diag[:, :n]).cumsum(dim=-1) / torch.arange(1, n + 1)).min()
    plot_max = (torch.log(r_diag[:, :n]).cumsum(dim=-1) / torch.arange(1, n + 1)).max()
    for first_n in ns:
        plt.figure()
        x = torch.arange(r_vals.shape[1])
        plt.plot(x, torch.log(r_diag[:, :first_n]).mean(dim=-1), label='Rvals')
        plt.xlabel('t')
        plt.ylabel(f'Sum of first {first_n} Rval logs')
        plt.title(
            f'R value evolution over time for trained RNN, \nSequence Length = {input_seq_length}, First {first_n}')
        # if model_type == 'rnn':
        #     plt.ylim([plot_min, plot_max])
        fig_filename = f'SMNIST/Plots/rplot/rvals_plot_trainedRNN_h{hidden_size}_len{input_seq_length}_n{first_n}.png'
        filenames.append(fig_filename)
        plt.savefig(fig_filename, bbox_inches='tight')
        # plt.figure()
        plt.plot(x, torch.log(torch.linalg.norm(gradV_list[:, 0], dim=(-2, -1))).cpu(), label='GradV Norm')
        # plt.scatter(x, torch.linalg.norm(gradU_list[:, 0], dim=(-2, -1)), label='GradU Norm')
        # plt.scatter(x, torch.linalg.norm(gradW_list[:, 0], dim=(-2, -1)), label='GradW Norm')
        plt.legend()
        plt.ylabel('LogNorm')
        plt.xlabel('t')
        plt.title(f'Gradient Component Norms over time, \nSequence Length = {input_seq_length}, First {first_n}')
        plt.savefig(f'SMNIST/Plots/rplot/gradnorms_plot_trainedRNN_h{hidden_size}_len{input_seq_length}_n{first_n}.png',
                    bbox_inches='tight')
        plt.close()

    with imageio.get_writer(
            f'SMNIST/Plots/rplot/rvals_vid_trainedRNN_h{hidden_size}_len{input_seq_length}.gif',
            mode='I', fps=4) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            os.remove(filename)
        # for filename in set(filenames):
        #     os.remove(filename)

if r_thresh:
    torch.manual_seed(31)
    plot_dir = f'SMNIST/Plots/rthresh/Epoch{epoch}'
    # le_target = targets[:, 0].squeeze()
    # le_input = x_set[:, 0]
    # model.evaluate = True
    # le_loss, le_acc = model(le_input[:, order], y=le_target)
    # le_test_loss, le_test_acc = test_model(model, le_loader, order)
    # print(f'Model LE Loss: {le_loss}, Acc: {le_acc/5.}')
    # full_test_loss, full_test_acc = test_model(model, testloader, order)
    # print(f'Model Test Loss: {full_test_loss}, Acc: {full_test_acc}')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    suffix = ''

    pred_logits = pred_list.softmax(dim=-1)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    base_loss = criterion(pred_list[-1], le_target)
    # best_idx = torch.argmin(criterion(pred_list[-1], le_target))

    r_vals = ftle_dict['rvals']
    x = torch.arange(r_vals.shape[1])
    plot_evolution = False
    # n_ev_plots = 5
    plot_flips = False
    plot_input = False

    # Flip pixels based on mean value of first n FTLEs
    n_list = [1, 2, 5, 10, 20, 50, 100, 150, 250, 400, 512]
    std_list = [0.5, 1, 2, 3]
    flipped_losses = torch.zeros(len(std_list), len(n_list), le_batch_size)
    flip_counts = torch.zeros(len(std_list), len(n_list), 2, le_batch_size)
    flip_le_inputs = torch.zeros(len(std_list), len(n_list), le_batch_size, input_seq_length, inp_size)

    #Store Random Flips
    no_randoms = 50
    random_losses = torch.zeros(len(std_list), len(n_list), no_randoms, le_batch_size)
    random_inputs = torch.zeros(len(std_list), len(n_list), no_randoms, le_batch_size, input_seq_length, inp_size)

    # random_loss_means = torch.zeros(len(std_list), len(n_list))
    # random_loss_max = torch.zeros(len(std_list), len(n_list))
    # random_loss_min = torch.zeros(len(std_list), len(n_list))
    n_select = 6
    std_select = 0
    idx_select = 3

    for n_idx, n in enumerate(n_list):
        thresh_dir = f'{plot_dir}/Rvals{n}'
        if not os.path.exists(thresh_dir):
            os.mkdir(thresh_dir)

        a = torch.log(r_vals[:, :, :n]).mean(dim=-1)
        sort_order = a.argsort(dim=1, desecending=True)
        a_ordered = a[sort_order]

        # a_median, a_std = a.median(), a.std()
        # a_thresholds = [a_median - i * a.std() for i in std_list]
        # print(f'n = {n}, Thresholds = {a_thresholds}')

        if plot_evolution:
            plt.figure()
            idcs = rng.permutation(le_batch_size)[:5]
            for idx in idcs:
                plt.plot(x, a[idx], alpha=0.6, label=idx)
            for a_thresh in a_thresholds:
                plt.plot([0, 784], [a_thresh, a_thresh], c='k')
            plt.xlabel('t')
            plt.ylabel(f'Sum of first {n} Rval logs{perm_plot}')
            plt.title(
                f'R value evolution over time for trained RNN, First {n} Rval(s)')
            fig_filename = f'{thresh_dir}/rvals_plot_n{n}{perm_suffix}.png'
            # plt.legend()
            plt.savefig(fig_filename, bbox_inches='tight', dpi=400)
            plt.close()

        for thresh_idx, threshold in enumerate(a_thresholds):
            load_thresh = True
            thresh_suffix = '_less'
            check_thresh = torch.where(torch.log(r_vals[:, inverse_order, :n]).mean(dim=-1) < threshold)
            # check_thresh0 = torch.log(r_vals[:, inverse_order, :n]).mean(dim=-1) < a_median-0.1*a_std
            check_thresh0 = True
            check_thresh2 = torch.log(r_vals[:, inverse_order, :n]).mean(dim=-1) < threshold
            check_thresh1 = check_thresh0*check_thresh2
            num_flips = check_thresh1.sum(dim=-1)
            # print(f'Mean num flips for threshhold = {threshold:.2f}: {num_flips.float().mean():.2f}')

            flipped_fname = f'{thresh_dir}/flipped_losses_thresh{threshold:.2f}{thresh_suffix}.p'
            if os.path.exists(flipped_fname) and load_thresh:
                # print(f'Loading flips for threshold {threshold}')
                flipped_loss, flipped_le_input, flip_count = torch.load(flipped_fname)
            else:
                # print(f'Calculating flips for threshold {threshold}')
                flipped_le_input = torch.clone(le_input)
                # flipped_le_input[0, :, :] =

                up_thresh = check_thresh1*(le_input<0.001).squeeze()
                down_thresh = check_thresh1*(le_input>0.001).squeeze()
                flipped_up_counts = up_thresh.sum(dim=-1)
                flipped_down_counts = down_thresh.sum(dim=-1)
                flip_count = torch.stack([flipped_up_counts, flipped_down_counts])
                flipped_le_input[up_thresh] = 1.0
                flipped_le_input[down_thresh] = 0.0
                # flipped_le_input[check_thresh1] = 1.0 - flipped_le_input[check_thresh1]

                outs, h = model(flipped_le_input[:, order])
                flipped_loss = criterion(outs, le_target)
                flipped_logits = outs.softmax(dim=-1)
                torch.save((flipped_loss, flipped_le_input, flip_count), flipped_fname)
            flipped_losses[thresh_idx, n_idx] = flipped_loss
            flip_counts[thresh_idx, n_idx] = flip_count
            flip_le_inputs[thresh_idx, n_idx] = flipped_le_input

            if plot_flips:
                for idx in range(le_batch_size):
                    # # Flat Input Image (permuted)
                    plt.figure()
                    plt.title(f'Flat Flipped Input Plot for RNN, n= {n}, Threshold= {threshold:.2f} \n Label = {le_target[idx]}{perm_plot}')
                    plt.plot(flipped_le_input[idx, order].detach())
                    plt.savefig(f'{plot_dir}/Inputs/flat_input_plot_idx{idx}{perm_suffix}_n{n}_thresh{threshold:.2f}{thresh_suffix}.png',
                                bbox_inches='tight')
                    plt.close()

            # Compare to Randomly Flipped Pixels
            rand_flip_fname = f'{thresh_dir}/rand_n{no_randoms}_thresh{threshold:.2f}{thresh_suffix}.p'
            if os.path.exists(rand_flip_fname) and load_thresh:
                # print(f'Loading random flips for threshold {threshold:.2f}')
                rand_losses, rand_flipped_inputs = torch.load(rand_flip_fname)
            else:
                print(f'Calculating random flips for threshold {threshold:.2f}')
                rng2 = np.random.RandomState(5678)
                rand_losses = []
                rand_inputs = []
                for i in range(no_randoms):
                    rand_order = rng.permutation(784)
                    rand_flipped_le_input = torch.clone(le_input)
                    rand_flips = check_thresh1[:, rand_order]
                    rand_flipped_le_input[rand_flips * (le_input < 0.001).squeeze()] = 1.0
                    rand_flipped_le_input[rand_flips * (le_input > 0.001).squeeze()] = 0.0
                    # rand_flipped_le_input[rand_flips] = 1. - rand_flipped_le_input[rand_flips]
                    out_rand, _ = model(rand_flipped_le_input[:, order])
                    rand_loss = criterion(out_rand, le_target)
                    rand_losses.append(rand_loss)
                    rand_inputs.append(rand_flipped_le_input)
                rand_losses = torch.stack(rand_losses)
                rand_flipped_inputs = torch.stack(rand_inputs)
                torch.save((rand_losses, rand_flipped_inputs), rand_flip_fname)
            random_losses[thresh_idx, n_idx] = rand_losses
            random_inputs[thresh_idx, n_idx] = rand_flipped_inputs
            # random_loss_means[thresh_idx, n_idx] = rand_losses.mean()
            # random_loss_min[thresh_idx, n_idx] = rand_losses.min()
            # random_loss_max[thresh_idx, n_idx] = rand_losses.max()



            # rand_logits = out_rand.softmax(dim=-1)
            if plot_input:
                for idx in torch.arange(le_batch_size):
                    # #Input Image (permuted)
                    # plt.figure()
                    # plt.pcolor(le_input.cpu()[idx][order].reshape((28, 28)))
                    # plt.gca().invert_yaxis()
                    # plt.title(f'Input for RNN, Label = {le_target[idx]}{perm_plot}')
                    # plt.savefig(f'{plot_dir}/input_plot_idx{idx}{perm_suffix}.png', bbox_inches='tight')
                    # plt.close()

                    # Flat Input Image (permuted)

                    plt.figure()
                    plt.plot(le_input.cpu()[idx][order])
                    plt.title(f'Flat Input Plot for RNN, Label = {le_target[idx]}{perm_plot}')
                    plt.savefig(f'{plot_dir}/Inputs/flat_input_plot_idx{idx}{perm_suffix}.png', bbox_inches='tight')
                    plt.close()

                # #Softmax Evolution
                # plt.figure()
                # plt.plot(pred_list[:, idx].cpu().softmax(dim=-1)[:, le_target[idx]])
                # plt.xlabel('t')
                # plt.ylabel(f'Correct Logit, label = {le_target[idx]}{perm_plot}')
                # plt.savefig(f'{plot_dir}/logits_plot_idx{idx}{perm_suffix}.png',
                #             bbox_inches='tight')

                # Plot of First Rvals

                # plt.figure()
                # plt.plot(x, a[idx], label='Rvals')
                # plt.xlabel('t')
                # plt.ylabel(f'Sum of first {n} Rval logs{perm_plot}')
                # plt.title(
                #     f'R value evolution over time for trained RNN, \nSequence Length = {input_seq_length}, First {n} Rval(s)')
                # fig_filename = f'{thresh_dir}/rvals_plot_n{n}_idx{idx}{perm_suffix}.png'
                # plt.savefig(fig_filename, bbox_inches='tight')

                # Final Logits Flipped
                # plt.figure()
                # plt.plot(pred_list[-1, idx].softmax(dim=-1).cpu(), label=f'original')
                # plt.plot(flipped_logits[idx].detach().cpu(), label=f'rvals')
                # plt.plot(rand_logits[idx].detach().cpu(), label=f'random')
                # plt.xlabel('Label')
                # plt.ylabel('Logit')
                # plt.legend()
                # plt.title(
                #     f'Class Logits for trained RNN, Flip thresh: {threshold:.2f}, \nCorrect label: {le_target[idx]}{perm_plot}, flips = {num_flips[idx]}')
                # plt.savefig(f'{thresh_dir}/classLogits_idx{idx}{perm_suffix}_thresh{threshold:.2f}.png',
                #             bbox_inches='tight')
                # plt.close()
    plot_losses = False
    for thresh_idx, std in enumerate(std_list):
        print(f'Thresh Idx: {thresh_idx+1} of {len(std_list)}')
        if plot_losses:
            plt.figure()
            flip_mean = flipped_losses[thresh_idx].mean(dim=-1).detach()
            flip_std = flipped_losses[thresh_idx].std(dim=-1).detach()
            plt.plot([n_list[0], n_list[-1]], [base_loss.mean(), base_loss.mean()], label='base', c='k')
            plt.plot(n_list, flip_mean, label=f'Rvals', color='r')
            plt.fill_between(n_list, flip_mean - flip_std, flip_mean+flip_std,
                             alpha=0.2, facecolor='r', linewidth=0)
            # plt.plot(n_list, flipped_losses[thresh_idx].std(dim=-1).detach(), label=f'Rvals', color='r')
            plt.plot(n_list, random_losses[thresh_idx].mean(dim=(-2,-1)).detach(), label=f'Random', color='y')
            plt.yscale('log')
            rl_std = random_losses[thresh_idx].mean(dim=-2).std(dim=-1)
            # plt.fill_between(n_list, random_losses[thresh_idx].mean(dim=-1).min(dim=-1)[0].detach(),
            #                  random_losses[thresh_idx].mean(dim=-1).max(dim=-1)[0].detach(),
            #                  alpha=0.2, facecolor='y', linewidth=0)
            plt.fill_between(n_list, (random_losses[thresh_idx].mean(dim=(-2,-1)) - rl_std).detach(),
                                                (random_losses[thresh_idx].mean(dim=(-2,-1)) + rl_std).detach(),
                                                alpha=0.2, facecolor='y', linewidth=0)
            plt.title(f'Rval vs. Random Flip loss, Stds = {std}, Epoch {epoch}')
            # plt.legend()
            loss_dir = f'{plot_dir}/loss_plots'
            if not os.path.exists(loss_dir):
                os.mkdir(loss_dir)
            plt.savefig(f'{plot_dir}/loss_plots/LossValues{perm_suffix}_std{std}_rands{no_randoms}{thresh_suffix}.png',
                    bbox_inches='tight')

    plot_input = True
    in_plot_dir = f'SMNIST/Plots/rthresh/Epoch{epoch}/Inputs'
    if not os.path.exists(in_plot_dir):
        os.mkdir(in_plot_dir)
    original_input = le_input[idx_select]
    plt.figure(figsize=(3,3))
    plt.imshow(original_input.reshape(28,28))
    plt.savefig(f'{in_plot_dir}/original_input_{idx_select}.png', dpi=400, bbox_inches='tight')

    plt.figure(figsize=(3,3))
    flipped_input = flip_le_inputs[std_select, n_select, idx_select]
    plt.imshow(flipped_input.reshape(28,28))
    plt.savefig(f'{in_plot_dir}/flipped_input_{idx_select}.png', dpi=400, bbox_inches='tight')


    flip_mean_thresh = flipped_losses.mean(dim=0).detach()
    rand_mean_thresh = random_losses.mean(dim=0).detach()
    rl_thresh_mean = random_losses.mean(dim=-2).detach()
    rl_thresh_std = random_losses.std(dim=-2).detach()

    plot_ind_losses = False
    if plot_ind_losses:
        print(f'Individual Loss Plots')
        for thresh_idx, std in enumerate(std_list):
            print(f'Thresh = {std}')
            for input_idx in tqdm(range(le_batch_size)):
                plt.figure(figsize=(3,3))
                plt.plot([n_list[0], n_list[-1]], [base_loss[input_idx], base_loss[input_idx]], label='base', c='k')
                plt.plot(n_list, flipped_losses[thresh_idx, :, input_idx].detach(), label=f'Rvals', color='b')
                plt.plot(n_list, rl_thresh_mean[thresh_idx, :, input_idx], label=f'Random', color='y')
                plt.fill_between(n_list, rl_thresh_mean[thresh_idx, :, input_idx]-rl_thresh_std[thresh_idx, :, input_idx],
                                 rl_thresh_mean[thresh_idx, :, input_idx]+rl_thresh_std[thresh_idx, :, input_idx],
                                 alpha=0.2, facecolor='y', linewidth=0)
                # plt.title(f'Rval vs. Random Flip loss, index = {input_idx}, Correct = {le_target[input_idx]}, Epoch {epoch}, thresh_idx: {thresh_idx}')
                loss_dir = f'{plot_dir}/loss_plots'
                if not os.path.exists(loss_dir):
                    os.mkdir(loss_dir)
                plt.savefig(f'{plot_dir}/loss_plots/LossValues{perm_suffix}_idx{input_idx}_thresh_idx{thresh_idx}.png',
                            bbox_inches='tight', dpi=400)
                plt.close()

    plot_logits = False
    if plot_logits:
        for thresh_idx, std in enumerate(std_list):
            print(f'Thresh = {std}')
            for input_idx in tqdm(range(le_batch_size)):
                for n_idx, n in enumerate(n_list):
                    plt.figure(figsize=(3,3))
                    plt.plot(pred_logits[-1, input_idx], color='k')
                    f_logits = model(flip_le_inputs[thresh_idx, n_idx, input_idx, order].unsqueeze(0), h_0)[0].softmax(dim=-1).detach().t()
                    rand_logits = model(random_inputs[thresh_idx, n_idx, :, input_idx, order], h_0)[0].softmax(dim=-1).detach().t()
                    plt.plot(f_logits, c='b')
                    plt.plot(rand_logits.mean(dim=-1), color='y')
                    plt.fill_between(range(10), rand_logits.min(dim=-1)[0], rand_logits.max(dim=-1)[0], alpha=0.2,
                                     color='y')
                    # plt.title(f'Random Logits for n ={n}, index = {input_idx}, Correct = {le_target[input_idx]}, Epoch {epoch}')
                    plt.savefig(f'{plot_dir}/Rvals{n}/randLogits{perm_suffix}_idx{input_idx}_std{thresh_idx}.png',
                                bbox_inches='tight', dpi=400)
                    plt.close()

    plt.figure(figsize=(4,2))
    y = (flipped_losses - base_loss.unsqueeze(0).unsqueeze(0).repeat(len(std_list), len(n_list), 1)).detach()
    x = flip_counts
    plt.scatter(x[:, :, 0], y, label='up', alpha=0.5)
    plt.scatter(x[:, :, 1], y, label='down', alpha=0.2)
    # plt.legend()
    # plt.title(f'Change in loss as a function of number of pixel flips, Epoch {epoch}')
    plt.xlabel(f'Number of Pixel Flips')
    plt.ylabel(f'Loss')
    plt.savefig(f'{plot_dir}/Loss_vs_flips.png', bbox_inches='tight', dpi=400)