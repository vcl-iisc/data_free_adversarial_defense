import os
import sys
import tqdm
import os.path as osp
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader, TensorDataset

from config import *
from models.resnet import resnet18
from frequencyHelper import generateDataWithDifferentFrequencies_3Channel as freq_3t
from frequencyHelper import generateDataWithDifferentFrequencies_GrayScale as freq_t


### Data Utils

def get_adv_data(args, batch_size=None, data=False):
    
    ## Get Adversarial Data Loader

    if batch_size is None:
        batch_size = args.batch_size

    save_path = os.path.join('./adv_data', args.dataset + '/')
    save_path += args.model_name + '_' + args.attack + '.pt'

    adv_images, adv_labels = torch.load(save_path)
    adv_data = TensorDataset(adv_images.float()/255, adv_labels)
    adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=False)

    if data:
        return adv_data, adv_loader

    return adv_loader

def load_data(root='./input/cifar10/', batch_size=32, valid_size=0.2, data=False):
    
    dataset = root.split('/')[-2]

    ## Normalization will go in model construction
    preprocess = transforms.Compose([transforms.Resize((32,32)),
                                        transforms.ToTensor()])
    
    ## Init the data
    if dataset == 'cifar10':
        print('Loading Cifar')
        train_data = torchvision.datasets.CIFAR10(root, train=True, transform=preprocess)
        test_data = torchvision.datasets.CIFAR10(root=root, train=False, transform=preprocess)
    elif dataset == 'mnist':
        print('Loading MNIST')
        train_data = torchvision.datasets.MNIST(root, train=True, transform=preprocess, download=True)
        test_data = torchvision.datasets.MNIST(root=root, train=False, transform=preprocess, download=True)
    elif dataset == 'fmnist':
        print('Loading FMNIST')
        train_data = torchvision.datasets.FashionMNIST(root, train=True, transform=preprocess, download=True)
        test_data = torchvision.datasets.FashionMNIST(root=root, train=False, transform=preprocess, download=True)
    else:
        print(f'{root} / {dataset} doesnt exist')

    ## Split Train and Valid Data
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size*num_train))
    train_idx,valid_idx = indices[split:],indices[:split]
    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
        
    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    if data:
        return test_data, test_loader
    
    return train_loader,valid_loader,test_loader


def load_data_source_detector(root='./input/cifar10/', batch_size=32, valid_size=0.2):
    
    dataset = root.split('/')[-2]

    ## Normalization will go in model construction
    preprocess = transforms.Compose([transforms.Resize((32,32)),
                                        transforms.ToTensor()])
    
    ## Init the data
    if dataset == 'cifar10':
        print('Loading Cifar')
        train_data = torchvision.datasets.CIFAR10(root, train=True, transform=preprocess)
        test_data = torchvision.datasets.CIFAR10(root=root, train=False, transform=preprocess)
    elif dataset == 'mnist':
        print('Loading MNIST')
        train_data = torchvision.datasets.MNIST(root, train=True, transform=preprocess, download=True)
        test_data = torchvision.datasets.MNIST(root=root, train=False, transform=preprocess, download=True)
    elif dataset == 'fmnist':
        print('Loading FMNIST')
        train_data = torchvision.datasets.FashionMNIST(root, train=True, transform=preprocess, download=True)
        test_data = torchvision.datasets.FashionMNIST(root=root, train=False, transform=preprocess, download=True)
    else:
        print(f'{root} / {dataset} doesnt exist')

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

    return train_data, train_loader


class CombDataset(torch.utils.data.Dataset):
    """Dataset wrapper to induce class-imbalance"""

    def __init__(self, clean_dataset, adv_dataset, pre_clf, device=None, transform=None, return_idx=False):
        
        self.clean_dataset = clean_dataset ## Clean Dataset

        if type(adv_dataset) == list:
            self.combined_dataset = torch.utils.data.ConcatDataset([clean_dataset, adv_dataset[0]])
            for idx in range(1, len(adv_dataset)):
                self.combined_dataset = torch.utils.data.ConcatDataset([self.combined_dataset, adv_dataset[idx]])
            self.combined_labels = np.concatenate(([0]*len(self.clean_dataset), [1]*(len(self.combined_dataset) - len(self.clean_dataset))))
        else:
            self.adv_dataset = adv_dataset ## Clean Dataset
            self.combined_dataset = torch.utils.data.ConcatDataset([clean_dataset, adv_dataset])
            self.combined_labels = np.concatenate(([0]*len(self.clean_dataset), [1]*len(self.adv_dataset)))

        print(f'Clean : {sum(self.combined_labels == 0)} \t|\t Adv : {sum(self.combined_labels == 1)}')
        self.device = device

        self.pre_clf = pre_clf

        self.transform = transform
        self.return_idx = return_idx

    def get_logits(self, x):
        x = x.to(self.device)
        
        self.pre_clf.eval()
        op = self.pre_clf(x.unsqueeze(0))
        return op.detach()
        
    def __getitem__(self, i):
        (x,cls_), y = self.combined_dataset[i], self.combined_labels[i] ## Original Sample

        if self.transform == 'logits':
            logits = self.get_logits(x) 

        if self.return_idx:
            return (x, logits, int(cls_), int(y), int(i))
        else: 
            return (x, logits, int(cls_), int(y))

    def __len__(self):
        return len(self.combined_dataset)


def get_loader_source_detector(args, batch_size=None, data=False):
    
    ## Get Adversarial Data Loader

    if batch_size is None:
        batch_size = args.batch_size

    save_path = os.path.join('./adv_data', args.dataset + '/train_')
    save_path += args.model_name + '_' + args.attack + '.pt'

    adv_images, adv_labels = torch.load(save_path)
    adv_data = TensorDataset(adv_images.float()/255, adv_labels)
    adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=False)

    if data:
        return adv_data, adv_loader

    return adv_loader

### Model Utils

class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        if len(self.mean)>1:
            mean = self.mean.reshape(1, 3, 1, 1)
            std = self.std.reshape(1, 3, 1, 1)
        else:
            mean, std = self.mean, self.std
        return (input - mean) / std



def load_model(args):
    
    if args.dataset == 'cifar10':
        save_path = None
        MEAN, STD = CIFAR_MEAN, CIFAR_STD 
    else: 
        save_path = './'+args.dataset+'_state_dict/'+args.model_name+'.pt'
        MEAN, STD = MNIST_MEAN, MNIST_STD 

    base_model = resnet18(pretrained=True, save_path=save_path, p=args.p)    
    norm_layer = Normalize(mean=MEAN, std=STD)

    model = nn.Sequential(
        norm_layer,
        base_model
    ).to(args.device)
        
    return model


### Metric Utils

def get_correct(outputs, labels):
    
    ## Correct Predicitons in a given batch

    _, pred = torch.max(outputs, 1)
    correct = (pred == labels).float().sum(0).item()
    return correct


def evaluate_attack(loader, model, args):

    ## Performance on a given attack

    print(f'Evaluating : {args.attack} on {args.model_name}')

    ## Go in eval mode
    model.eval()

    metrics = {'clean_acc':{'correct':0, 'total':0}, 'adv_acc':{'correct':0, 'total':0}}
    adv_loader = get_adv_data(args)
    pbar = tqdm.tqdm(zip(loader, adv_loader), unit="batches", leave=False, total=len(loader), ascii=True, ncols=150)

    for (data,labels),(adv_data, adv_labels) in pbar:
        
        data, adv_data, labels = data.to(args.device), adv_data.to(args.device), labels.to(args.device)
        
        clean_output = model(data)
        metrics['clean_acc']['correct'] += get_correct(clean_output, labels)
        metrics['clean_acc']['total']   += clean_output.size(0)

        adv_output = model(adv_data)
        metrics['adv_acc']['correct'] += get_correct(adv_output, labels)
        metrics['adv_acc']['total']   += adv_output.size(0)

    metrics['clean_acc']['acc'] = (metrics['clean_acc']['correct'] / metrics['clean_acc']['total']) * 100.
    metrics['adv_acc']['acc'] = (metrics['adv_acc']['correct'] / metrics['adv_acc']['total']) * 100.

    print(f'Clean Accuracy : {metrics["clean_acc"]["acc"]:.2f} \t|\t Correct : {metrics["clean_acc"]["correct"]} \t|\t Total : {metrics["clean_acc"]["total"]}')
    print(f'{args.attack} Accuracy : {metrics["adv_acc"]["acc"]:.2f} \t|\t Correct : {metrics["adv_acc"]["correct"]} \t|\t Total : {metrics["adv_acc"]["total"]}')

    return metrics


### Misc.

def get_freq(data, r, dataset='cifar10'):
    
    images = data.detach().cpu()

    if dataset == 'cifar10':
        images = images.permute(0,2,3,1)
        img_l, img_h = freq_3t(images, r=r)
        img_l, img_h = torch.from_numpy(np.transpose(img_l, (0,3,1,2))), torch.from_numpy(np.transpose(img_h, (0,3,1,2)))
        return img_l, img_h

    img_l, img_h = freq_t(images, r=r)
    img_l, img_h = torch.from_numpy(img_l).view(-1, 32, 32).unsqueeze(1), torch.from_numpy(img_h).view(-1, 32, 32).unsqueeze(1)
    return img_l, img_h


class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise   

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()