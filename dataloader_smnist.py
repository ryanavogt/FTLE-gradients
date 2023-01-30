import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class MNISTDataset():
    def __init__(self, train=True, download = False, root = ''):
        # MNIST dataset
        apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if train:
            self.dataset = torchvision.datasets.MNIST(root=root,
                                                   train=True,
                                                   transform=apply_transform,
                                                   download=download)
        else:
            self.dataset = torchvision.datasets.MNIST(root=root,
                                                  train=False,
                                                  transform=apply_transform)
    def __len__(self):
        return len(self.dataset)
        
        
def create_dataset(config, seed = 42, download = False):
    train_dataset = MNISTDataset(train = True, root = config.data_dir, download = download).dataset
    test_dataset = MNISTDataset(train = False, root = config.data_dir, download = download).dataset
    seq_length = config.input_seq_length
    input_size = config.input_size
    total_size = input_size*seq_length
    
    t_split, v_split = (config.train_frac/(config.train_frac+config.val_frac), 
                            config.val_frac/(config.train_frac+config.val_frac))
    train_len = len(train_dataset)
    train_dataset.data= torch.reshape(train_dataset.data.flatten(start_dim = 1)[:,:total_size],
                                                        (-1, seq_length, input_size))
    test_dataset.data = torch.reshape(test_dataset.data.flatten(start_dim = 1)[:,:total_size],
                                                        (-1, seq_length, input_size))
    
    # if pixel:
        # train_dataset.data= train_dataset.data.flatten(start_dim = 1)[:,:seq_length].unsqueeze(-1)
        # test_dataset.data = test_dataset.data.flatten(start_dim = 1)[:,:seq_length].unsqueeze(-1)
    # else: 
        # train_dataset.data= train_dataset.data[:,:seq_length].unsqueeze(-1)
        # test_dataset.data = test_dataset.data[:,:seq_length].unsqueeze(-1)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, 
                                    [int(t_split*train_len), int(v_split*train_len)],) 
                                    # generator=torch.Generator().manual_seed(seed))
    
    return {'train_set':train_dataset, 'val_set':val_dataset, 'test_set':test_dataset}