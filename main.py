import matplotlib.pyplot as plt
import torch
from config import *
from lyapunov import *
from models import *
from tqdm import tqdm
from training import *
import time
import imageio


def main(input_seq_length=784, train=True):
    input_size = 784 // input_seq_length
    batch_size = 128
    le_batch_size = 25
    output_size = 10
    max_epoch = 10
    learning_rate = 0.002
    dropout = 0.1
    hidden_size = 128
    save_interval = 1
    in_epoch_saves = 4
    max_sub_epoch = 0
    fix_W = False
    load_LE = False
    train = False
    lyap = True
    grads = True
    flat_grads = False
    r_plot = True

    # params = torch.linspace(0.005, end= 0.025, steps = 2)
    # params = [0.001, 0.005, 0.025, 0.05, 0.25, 0.5]
    params = [0.001, 0.005]
    p = 0.005
    model_type = 'rnn'
    id_init_params = {'asrnn': 'b', 'rnn': 'std'}
    init_params = {'asrnn': {'a': -p, 'b': p}, 'rnn': {'mean': 0, 'std': p}}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dcon = DataConfig('../Dataset/', batch_size=batch_size, input_seq_length=input_seq_length, input_size=input_size,
                      target_seq_length=1, val_frac=0.2, test_frac=0, name_params={'insize': input_size})
    mcon = ModelConfig(model_type, 1, hidden_size, dcon.input_size, output_size=output_size, dropout=dropout,
                       init_type='normal', init_params=init_params[model_type],
                       device=device, bias=False, id_init_param=id_init_params[model_type])
    tcon = TrainConfig(model_dir='SMNIST/Models', batch_size=batch_size, max_epoch=max_epoch,
                       optimizer='sgd', learning_rate=learning_rate)
    fcon = FullConfig(dcon, tcon, mcon)

    train_dataloader = torch.utils.data.DataLoader(fcon.data.datasets['train_set'],
                                                   batch_size=fcon.train.batch_size)
    epoch_samples = len(list(train_dataloader))
    if in_epoch_saves > 0:
        save_idcs = part_equal(epoch_samples, in_epoch_saves)

    indices = dcon.datasets['val_set'].indices
    # le_input = dcon.datasets['val_set'].dataset.data[indices][:le_batch_size].unsqueeze(-1).to(fcon.device)
    h_le = torch.randn(1, le_batch_size, hidden_size).to(fcon.device)
    val_dl = torch.utils.data.DataLoader(fcon.data.datasets['val_set'],
                                         batch_size=le_batch_size)
    if load_LE:
        le_input, le_target = torch.load(f'SMNIST/le_setup_seq{input_seq_length}.p')
    else:
        le_input, le_target = next(iter(val_dl))
        le_input = le_input.to(fcon.device).squeeze(1)
        torch.save((le_input, le_target), f'SMNIST/le_setup_seq{input_seq_length}.p')
    le_target = le_target.to(fcon.device)
    print(f'LE target: {le_target[:20]}')
    le_input = le_input.to(fcon.device)
    model = RNNModel(fcon.model).to(fcon.device)
    if fix_W:
        optimizer = fcon.train.get_optimizer(model.rnn_layer.parameters())
    else:
        optimizer = fcon.train.get_optimizer(model.parameters())

    if in_epoch_saves > 0:
        epoch_samples = len(list(train_dataloader))
        save_idcs = part_equal(epoch_samples, in_epoch_saves)
    else:
        save_idcs = []

    if train:
        print('Training SMNIST RNNs')
        for i, p in enumerate(params):
            start_time = time.time()
            p = float(int(p * 1000)) * 1.0 / 1000
            params[i] = p
            print(f"Init Parameter: {p}")
            if model_type == 'asrnn':
                fcon.model.init_params = {'a': -p, 'b': p}
            else:
                fcon.model.init_params = {'mean': 0, 'std': p}
            model = RNNModel(fcon.model).to(fcon.device)
            optimizer = fcon.train.get_optimizer(model.parameters())
            train_loss, val_loss, accuracy = train_model(fcon, model, optimizer, verbose=True,
                                                         save_interval=save_interval, in_epoch_saves=in_epoch_saves,
                                                         max_sub_epoch=max_sub_epoch, overwrite=True, reg=False)
            end_time = time.time()
            time_diff = end_time - start_time
            remaining_time = (len(params) - (i + 1)) * time_diff
            print(f'Training time {time_diff:.2f}s, Expected remaining time: {remaining_time:.2f}s')

    model.train()
    lcon = LyapConfig(batch_size=le_batch_size, seq_length=input_seq_length)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    print(model.fc)

    if lyap:
        for p in params:
            p = float(int(p * 1000)) * 1.0 / 1000
            print(f"p = {p}")
            for epoch in range(0, max_epoch + 1, save_interval):
                print(f"Calculating FTLEs for epoch {epoch}")
                fcon.model.init_params[id_init_params[model_type]] = p
                for it in range(in_epoch_saves + 1):
                    if it == in_epoch_saves:
                        model, optimizer, _, _ = load_checkpoint(fcon, epoch)
                        suffix = ''
                        it_lab = ''
                    elif epoch == 0 or epoch >= max_sub_epoch:
                        continue
                    else:
                        ind = save_idcs[it]
                        print(f'Iteration: {ind}')
                        model, optimizer, _ = load_iter_checkpoint(fcon, epoch, save_idcs[it])
                        suffix = f'_iter{ind}'
                        it_lab = f'Iteration {ind}'
                    model, optimizer, _, _ = load_checkpoint(fcon, epoch)

                    h = torch.randn(1, le_batch_size, fcon.model.rnn_atts['hidden_size'])
                    ftle_dict = lcon.calc_FTLE(le_input, model, fcon, h=h, save_s=True)
                    ftle_dict['h'] = h
                    ftle_dict['targets'] = le_target
                    torch.save(ftle_dict, f'SMNIST/LEs/{fcon.name()}_e{epoch}{suffix}_FTLE.p')

                    if grads:
                        print("Calculating Gradients")
                        gradV_list = torch.zeros(input_seq_length, le_batch_size, mcon.gate_sizes[model_type],
                                                 hidden_size)
                        gradW_list = torch.zeros(input_seq_length, le_batch_size, hidden_size, output_size)
                        gradU_list = torch.zeros(input_seq_length, le_batch_size, mcon.gate_sizes[model_type],
                                                 input_size)
                        loss_list = torch.zeros(input_seq_length, le_batch_size)
                        pred_list = torch.zeros(input_seq_length, le_batch_size, output_size)
                        grad_dict = {}
                        layer_names = ['W', 'U', 'b_i', 'b_h']
                        model.train()
                        for t in tqdm(torch.arange(1, input_seq_length + 1)):
                            preds, _ = model(le_input[:, :t], h_le)
                            loss = criterion(preds[:, -1], le_target)
                            loss_list[t - 1] = loss.detach()
                            pred_list[t - 1] = preds[:, -1].detach()
                            for i in range(le_batch_size):
                                optimizer.zero_grad()
                                loss[i].backward(retain_graph=True)
                                for layer, name in zip(model.rnn_layer.parameters(), layer_names):
                                    grad_dict[name] = layer.grad
                                if model_type == 'asrnn':
                                    gradV_list[t - 1, i] = grad_dict['W']
                                    gradU_list[t - 1, i] = grad_dict['U']
                                else:
                                    gradV_list[t - 1, i] = grad_dict['U']
                                    gradU_list[t - 1, i] = grad_dict['W']
                                # temp = model.fc.parameters()[0].grad
                                # print(temp.shape)
                                gradW_list[t - 1, i] = model.fc.weight.grad.t()

                        torch.save((gradV_list, gradU_list, gradW_list, loss_list),
                                   f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_grads.p')
                        torch.save(pred_list, f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_logits.p')

    if flat_grads:
        for p in params:
            p = float(int(p * 1000)) * 1.0 / 1000
            print(f"p = {p}")
            for epoch in range(0, max_epoch + 1, save_interval):
                fcon.model.init_params['std'] = p
                for it in range(in_epoch_saves + 1):
                    if it == in_epoch_saves:
                        model, optimizer, _, _ = load_checkpoint(fcon, epoch)
                        suffix = ''
                        it_lab = ''
                    elif epoch == 0 or epoch >= max_sub_epoch:
                        continue
                    else:
                        ind = save_idcs[it]
                        print(f'Iteration: {ind}')
                        model, optimizer, _ = load_iter_checkpoint(fcon, epoch, save_idcs[it])
                        suffix = f'_iter{ind}'
                        it_lab = f'Iteration {ind}'
                    model, optimizer, _, _ = load_checkpoint(fcon, epoch)
                    print(f"Calculating (flat) Gradients for epoch {epoch}")
                    gradV_list = torch.zeros(le_batch_size, mcon.gate_sizes[model_type], hidden_size)
                    gradW_list = torch.zeros(le_batch_size, hidden_size, output_size)
                    gradU_list = torch.zeros(le_batch_size, mcon.gate_sizes[model_type], input_size)
                    loss_list = torch.zeros(le_batch_size)
                    pred_list = torch.zeros(le_batch_size, output_size)
                    grad_dict = {}
                    layer_names = ['W', 'U', 'b_i', 'b_h']
                    model.train()
                    ftle_dict = torch.load(f'SMNIST/LEs/{fcon.name()}_e{epoch}{suffix}_FTLE.p')
                    # print(f'File name: SMNIST/LEs/{fcon.name()}_e{epoch}{suffix}_FTLE.p')
                    h = ftle_dict['h'].to(device)
                    # print(f'H: {h[0, 0]}')
                    # Calculate Gradients
                    model.dropout = nn.Dropout(p=0)
                    preds, _ = model(le_input, h)
                    # print(f'Preds: {preds[0]}')
                    loss = criterion(preds, le_target)
                    loss_list = loss.detach()
                    pred_list = preds.detach()
                    V = model.rnn_layer.all_weights[0][1]
                    U = model.rnn_layer.all_weights[0][0]
                    W = model.fc.weight
                    # print(f'LE input: {le_input}')
                    # print(f'RNN Layer Params: {model.rnn_layer.all_weights}')
                    param_list = [V, U, W]
                    # print(f'V: {V}')
                    # print(f'U: {U}')
                    # print(f'W: {W}')
                    for i in range(le_batch_size):
                        optimizer.zero_grad()
                        loss[i].backward(retain_graph=True)
                        for layer, name in zip(model.rnn_layer.parameters(), layer_names):
                            grad_dict[name] = layer.grad
                        # print(f'Grad Dict: {grad_dict}')
                        gradV_list[i] = V.grad
                        gradU_list[i] = U.grad
                        gradW_list[i] = W.grad.t()
                    # gradV_list[i] = grad_dict['U']
                    # gradU_list[i] = grad_dict['W']
                    # gradW_list[i] = model.fc.weight.grad.t()

                    torch.save((gradV_list, gradU_list, gradW_list, loss_list),
                               f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_Fullgrads.p')

    if r_plot:
        torch.manual_seed(31)
        for p in params:
            plot_dir = f'SMNIST/Plots/p{p}'
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)
            if model_type == 'asrnn':
                fcon.model.init_params = {'a': -p, 'b': p}
            else:
                fcon.model.init_params = {'mean': 0, 'std': p}
            model, optimizer, _, _ = load_checkpoint(fcon, max_epoch)
            suffix = ''
            epoch = model.best_epoch
            print(f'Best Epoch: {epoch}')
            print(f'Best Loss: {model.best_loss}')
            ftle_dict = torch.load(f'SMNIST/LEs/{fcon.name()}_e{epoch}{suffix}_FTLE.p', map_location=device)
            pred_list = torch.load(f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_logits.p', map_location=device)
            # pred_logits = pred_list.softmax(dim=-1)
            best_idx = torch.argmin(criterion(pred_list[-1], le_target))
            plt.figure()
            plt.pcolor(le_input.cpu()[best_idx].reshape((28,28)))
            plt.gca().invert_yaxis()
            plt.title(f'Input for {model_type}, p={p}')
            plt.savefig(f'{plot_dir}/input_plot_{model_type}_h{hidden_size}_len{input_seq_length}.png', bbox_inches='tight')

            plt.figure(figsize = (4,1))
            plt.pcolor(le_input.cpu()[best_idx].T)
            plt.gca().invert_yaxis()
            plt.savefig(f'{plot_dir}/input_plot_sequence_{model_type}_h{hidden_size}_len{input_seq_length}.png', bbox_inches='tight')
            plt.figure()
            plt.plot(pred_list[-1, best_idx].softmax(dim=-1).cpu())
            plt.xlabel('Label')
            plt.ylabel('Logit')
            plt.title(f'Class Logits for {model_type}, p={p}, seq len = {input_seq_length}\n Correct label: {le_target[best_idx]}')
            plt.savefig(f'{plot_dir}/classLogits_{model_type}_h{hidden_size}_len{input_seq_length}.png',
                        bbox_inches='tight')
            plt.close()

            gradV_list, gradU_list, gradW_list, loss_list = torch.load(
                f'SMNIST/Grads/{fcon.name()}_e{epoch}{suffix}_grads.p', map_location=device)
            filenames = []

            plt.figure()
            plt.plot(pred_list[:, best_idx].cpu().softmax(dim=-1)[:, le_target[best_idx]])
            plt.xlabel('t')
            plt.ylabel('Correct Logit')
            plt.savefig(f'{plot_dir}/logits_plot_{model_type}_h{hidden_size}_len{input_seq_length}.png',
                        bbox_inches='tight')
            plt.close()

            r_vals = ftle_dict['rvals']
            r_diag = torch.diagonal(r_vals[best_idx], dim1=-2, dim2=-1).cpu()
            ns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
            n = ns[-1]
            plot_min = torch.log(r_diag[:, :n].cumsum(dim=-1).min())
            plot_max = torch.log(r_diag[:, :n].cumsum(dim=-1).max())
            for first_n in ns:
                plt.figure()
                x = torch.arange(r_vals.shape[1])
                plt.plot(x, torch.log(r_diag[:, :first_n].sum(dim=-1)), label='Rvals')
                plt.xlabel('t')
                plt.ylabel(f'Sum of first {first_n} Rvals')
                plt.title(f'R value evolution over time for RNN, p = {p}\nSequence Length = {input_seq_length}, First {first_n}')
                if model_type == 'rnn':
                    plt.ylim([plot_min, plot_max])
                fig_filename = f'SMNIST/Plots/p{p}/rvals_plot_{model_type}_h{hidden_size}_len{input_seq_length}_n{first_n}.png'
                filenames.append(fig_filename)
                plt.savefig(fig_filename, bbox_inches='tight')
                # plt.figure()
                plt.plot(x, torch.log(torch.linalg.norm(gradV_list[:, 0], dim=(-2, -1))).cpu(), label='GradV Norm')
                # plt.scatter(x, torch.linalg.norm(gradU_list[:, 0], dim=(-2, -1)), label='GradU Norm')
                # plt.scatter(x, torch.linalg.norm(gradW_list[:, 0], dim=(-2, -1)), label='GradW Norm')
                plt.legend()
                plt.ylabel('LogNorm')
                plt.xlabel('t')
                plt.title(f'Gradient Component Norms over time, p = {p}\nSequence Length = {input_seq_length}, First {first_n}')
                plt.savefig(f'SMNIST/Plots/p{p}/gradnorms_plot_{model_type}_h{hidden_size}_len{input_seq_length}_n{first_n}.png', bbox_inches='tight')
                plt.close()

            with imageio.get_writer(
                    f'SMNIST/Plots/p{p}/rvals_vid_{model_type}_h{hidden_size}_len{input_seq_length}.gif',
                    mode='I', fps=4) as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            # for filename in set(filenames):
            #     os.remove(filename)

if __name__ == '__main__':
    main(112, False)


def part_equal(n, splits):
    size = (n * 1.0) / (splits + 1)
    splits = [int(round(size * (i + 1))) for i in range(n)]
    return splits
