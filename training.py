import torch
from config import *
from models import RNNModel
from torch import nn
import lyapunov as lyap

import os

def load_checkpoint(full_con, load_epoch, verbose = False, overwrite = False):
    if verbose:
        print("Train Directory:", full_con.train.model_dir)
    device = full_con.model.device
    model = RNNModel(full_con.model).to(device)
    optimizer = full_con.train.get_optimizer(model.parameters())
    ckpt_name = '{}/{}_e{}.ckpt'.format(full_con.train.model_dir, full_con.name(), load_epoch)
    if load_epoch > 0:
        if os.path.isfile(ckpt_name):
            ckpt = torch.load(ckpt_name, map_location = device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            train_loss = ckpt['train_loss']
            val_loss = ckpt['val_loss']
        else:
            print("Expected file name {}".format(ckpt_name))
            raise ValueError("Asked to load checkpoint at epoch {0}, but checkpoint does not exist.".format(load_epoch))
    elif load_epoch == 0 and os.path.isfile(ckpt_name) and overwrite == False:
        ckpt = torch.load(ckpt_name, map_location = device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        train_loss = ckpt['train_loss']
        val_loss = ckpt['val_loss']
    else:
        if verbose:
            print("Epoch = 0. Creating new checkpoint for untrained model")
        train_loss = []
        val_loss = 0.0
        accuracy = 0.0
        ckpt = {'model_state_dict': model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(), 'train_loss':train_loss, 'val_loss':val_loss, 'accuracy': accuracy}
        torch.save(ckpt, ckpt_name)
    return model, optimizer, train_loss, val_loss

def load_iter_checkpoint(full_con, load_epoch, iter_no, verbose = False):
    if verbose:
        print("Train Directory:", full_con.train.model_dir)
    device = full_con.model.device
    model = RNNModel(full_con.model).to(device)
    # print(full_con.model.rnn_atts['input_size'])
    optimizer = full_con.train.get_optimizer(model.parameters())
    ckpt_name = '{}/{}_e{}_iter{}.ckpt'.format(full_con.train.model_dir, full_con.name(), load_epoch, iter_no)
    if os.path.isfile(ckpt_name):
        ckpt = torch.load(ckpt_name, map_location = device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        train_loss = ckpt['train_loss']
    else:
        print("Expected file name {}".format(ckpt_name))
        raise ValueError("Asked to load checkpoint at epoch {0}, but checkpoint does not exist.".format(load_epoch))
    return model, optimizer, train_loss
    
def save_checkpoint(full_con, model, optimizer, train_loss, val_loss, accuracy, save_epoch, additional_dict = {}, suffix = ''):
    ckpt_name = '{}/{}_e{}{}.ckpt'.format(full_con.train.model_dir, full_con.name(), save_epoch, suffix)
    ckpt = {'model_state_dict': model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(), 'train_loss':train_loss, 'val_loss':val_loss, 'accuracy': accuracy}
    if len(additional_dict) > 0:
        ckpt.update(additional_dict)
    torch.save(ckpt, ckpt_name)

def save_iter_checkpoint(full_con, model, optimizer, train_loss, save_epoch, iteration, additional_dict = {}, suffix = ''):
    ckpt_name = '{}/{}_e{}_iter{}{}.ckpt'.format(full_con.train.model_dir, full_con.name(), save_epoch, iteration, suffix)
    ckpt = {'model_state_dict': model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(), 'train_loss':train_loss}
    if len(additional_dict) > 0:
        ckpt.update(additional_dict)
    torch.save(ckpt, ckpt_name)

def regularizer(fcon, model, x, h):
    seq_len = x.shape[1]
    batch_size = x.shape[0]
    # print(f'h shape: {h.shape}')
    # print(f'h: {h}')
    reg_list = torch.zeros(batch_size, seq_len-1)
    for k in range(seq_len-1):
        # print(f'hk shape {h[k].shape}')
        # print(f'xk shape {x[k].shape}')
        jac = lyap.rnn_jac(model.rnn_layer.all_weights, h[k], x[:,k], bias = fcon.model.rnn_atts['bias'])
        # print(f'h_k grad: {h[k+1].grad}')
    return reg_list

def train_model(full_con, model, optimizer, start_epoch= 0, print_interval = 1, save_interval = 1, verbose = True, keep_amount = 1.0, in_epoch_saves = 0, max_sub_epoch = 0, overwrite = False, reg = False, reg_factor = 0.1):
    device = full_con.device
    #ckpt = load_checkpoint(full_con, start_epoch)
    model, optimizer, train_loss, _ = load_checkpoint(full_con, start_epoch, verbose, overwrite = overwrite)
    #model.load_state_dict(ckpt['model_state_dict'])
    # optimizer.load_state_dict(ckpt['model_state_dict'])
    # train_loss = ckpt['loss']
    criterion = nn.CrossEntropyLoss(reduction = 'sum')
    scheduler = full_con.train.get_scheduler(optimizer)
    
    train_dataloader = torch.utils.data.DataLoader(full_con.data.datasets['train_set'], 
                                                    batch_size = full_con.train.batch_size)
    val_dataloader = torch.utils.data.DataLoader(full_con.data.datasets['val_set'], 
                                                    batch_size = full_con.train.batch_size*4)

    
    train_dataloader = DeviceDataLoader(train_dataloader, device)
    val_dataloader = DeviceDataLoader(val_dataloader, device)
    
    
    
#     data = dl.create_dataset(full_con.data)
    # train_input, train_target = (full_con.data.datasets['train_set'][0].to(device), full_con.data.datasets['train_set'][1].to(device))
    # val_input, val_target = (full_con.data.datasets['val_set'][0].to(device), full_con.data.datasets['val_set'][1].to(device))
    if verbose:
        print('Training ...')
    
    if in_epoch_saves >0:
        epoch_samples = len(list(train_dataloader))
        save_idcs = part_equal(epoch_samples, in_epoch_saves)
    else:
        save_idcs = []
    
    for epoch in range(start_epoch+1, full_con.train.max_epoch+1):
        if epoch%print_interval == 0 and verbose:
            print('Training epoch {} of {}'.format(epoch, full_con.train.max_epoch), end = '')
        running_loss = 0.0
        iter_loss = 0.0
        hidden = model.init_hidden(full_con.train.batch_size)

        #train for all batches in the training set
        model.train()
        total_samples = 0
        iter_samples = 0
        i = 0
        
        for batch_in, batch_target in train_dataloader:
            total_samples += len(batch_in)
            iter_samples += len(batch_in)
            optimizer.zero_grad()
            loss = 0.0
            batch_out, h_out = model(batch_in.squeeze(1), hidden[:, :len(batch_in)])
            loss = criterion(batch_out.view(-1, full_con.model.output_size), 
                                                batch_target.view(-1,)).to(device)
            loss.backward()
            print(f'h out grad {h_out.grad}')
            if reg:
                # print(f'h0 shape: {hidden[:, :len(batch_in)].shape}')
                # print(f'ht shape: {h_out.shape}')
                reg_loss = reg_factor * regularizer(full_con, model, batch_in.squeeze(1), torch.cat((hidden[:, :len(batch_in)].unsqueeze(0), h_out.transpose(0,1).unsqueeze(1))))
            optimizer.step()
#             print(loss.item())
            iter_loss += loss.item()
            running_loss += loss.item()
            i += 1
            if i in save_idcs and max_sub_epoch >= epoch:
                train_loss.append(iter_loss/iter_samples)
                save_iter_checkpoint(full_con, model, optimizer, train_loss, epoch,
                                        additional_dict = {'train_samples': epoch_samples}, iteration = i)
                iter_loss = 0.0
                iter_samples = 0
                
        train_loss.append(running_loss/(total_samples))

        #Find validation loss
        val_loss, accuracy = evaluate(model, full_con, criterion, val_dataloader)
        scheduler.step()
        
        if epoch%print_interval == 0 and verbose:
            print(', Training Loss: {:.4f}, Val Accuracy {:.3f}'.format(train_loss[-1], accuracy))

        #save model checkpoint
        if epoch%save_interval == 0:
            save_checkpoint(full_con, model, optimizer, train_loss, val_loss, accuracy, epoch, additional_dict = {'train_samples': epoch_samples})
    return train_loss, val_loss, accuracy

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
    
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)
def __len__(self):
        """Number of batches"""
        return len(self.dl)
        
def evaluate(model, full_con, criterion, val_dataloader):
    hidden = model.init_hidden(full_con.train.batch_size*4)
    total_samples = 0
    device = full_con.device
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        correct= 0
        for batch_in, batch_target in val_dataloader:
            total_samples += len(batch_in)
            batch_out, _ = model(batch_in.squeeze(1), hidden[:, :len(batch_in)])
            preds = batch_out.view(-1, full_con.model.output_size)
            loss = criterion(preds, batch_target.view(-1,)).to(device)
            digit_preds = torch.argmax(preds, dim = 1)
            correct += torch.sum(digit_preds == batch_target)
            # print(f"{correct} Correct out of {total_samples}")
            
#             print(loss.item())
            val_loss += loss.item()
        
        val_loss = val_loss/float(total_samples)
        accuracy = correct/float(total_samples)
    return val_loss, accuracy
    
def part_equal(n, splits):
    size = (n*1.0)/(splits+1)
    splits = [int(round(size*(i+1))) for i in range(n)]
    return splits