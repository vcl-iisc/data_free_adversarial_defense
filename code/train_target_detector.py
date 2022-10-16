import os
import sys
import numpy as np
from torch.serialization import load, save
import tqdm
import argparse
from scipy.spatial.distance import cdist
import random

import torch
from torch._C import device 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
import torch.optim as optim


from utils import load_model, load_data, CombDataset, get_adv_data, evaluate_attack, Logger
import loss_adapt


torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True



class Net(nn.Module):
    def __init__(self, flat_dim=10):
        super(Net, self).__init__()
        self.flat_dim = flat_dim

        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = weightNorm(nn.Linear(128,2), name='weight')

        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)
        self.batchnorm = nn.BatchNorm1d(128)

    def forward(self, x):

        ## Corresponding to the FE part
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        ## Batch Norm Bottleneck Part
        x = F.relu(self.fc2(x))
        x = self.batchnorm(x)
        x = self.dropout(x)

        self.features_test = x
        
        ## Locked Classifier.
        x = self.fc3(x)

        return x



def load_source_detector(args):

    detector = Net().to(args.device)
    ckpt = torch.load(args.load_source_path)

    print(f'Loading Source Detector @ {args.load_source_path}')
    detector.load_state_dict(ckpt['detector_state_dict'])

    return detector

def load_detector(args):
    
    detector = Net().to(args.device)
    ckpt = torch.load(args.load_target_path)

    print(f'Loading Target Detector @ {args.load_target_path}')
    detector.load_state_dict(ckpt['detector_state_dict'])

    return detector


def eval_dec(detector, loader, device=None):
    ## Evaluate Detector on Cifar
    detector.eval()

    correct, total = 0,0
    for data, logits, _, labels, _ in tqdm.tqdm(loader, leave=False, ascii=True, ncols=150):

        data, logits, labels = data.to(device), logits.to(device), labels.to(device)
        logits = logits.view((data.size(0), -1))
        output = detector(logits.float())
        _, pred = output.max(1)
        
        correct += (pred == labels).float().sum(0).item()
        total += data.size(0)

    print(f"Detector Accuracy : {(correct/total)*100:.2f}")
                
    return correct, total

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def adapt_detector(detector, loader, args, device=None):

    epochs = 20

    best_acc = -np.Inf

    max_iter = epochs * len(loader)
    interval_iter = len(loader)
    iter_num = 0


    ## Freeze Params of last layer
    for param in detector.fc3.parameters():
        param.requires_grad = False

    ## Training on for other two layers
    param_group = []
    for k, v in detector.fc1.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in detector.fc2.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    # for k, v in detector.fc3.named_parameters():
    #     param_group += [{'params': v, 'lr': args.lr}]
    for k, v in detector.batchnorm.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]


    optimizer = optim.SGD(param_group)
    # optimizer = op_copy(optimizer)

    while iter_num < max_iter:

        data, logits, _, labels, idxs = iter(loader).next()

        # for tar_idx, (data, logits, labels) in enumerate(loader):
        optimizer.zero_grad()

        data, logits, labels  = data.to(device), logits.to(device), labels.to(device)
        logits = logits.view((data.size(0), -1))

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            detector.eval()
            mem_label = obtain_label(loader, detector, args, device=device)
            mem_label = torch.from_numpy(mem_label).to(device)
            detector.train()

        iter_num += 1
        # lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)


        outputs_test = detector(logits.float())
        features_test = detector.features_test


        if args.cls_par > 0:
            pred = mem_label[idxs]
            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_test, pred)
        else:
            classifier_loss = torch.tensor(0.0).to(device)

            
        ## Check sign of ENTROPY LOSS
        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss_adapt.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0:
            if iter_num // interval_iter == 1:
                calc_metrics(loader, args, detector=detector)
            print(f'IM-Loss : {im_loss}')
            correct, total = eval_dec(detector, loader, device)
            print(f'Iteration : {iter_num} \t|\t Acc : {(correct/total)*100.}')
            if (correct/total)*100. > best_acc:
                best_acc = (correct/total)*100.
                best_iter = iter_num

                if args.issave:

                    ckpt = {'detector_state_dict':detector.state_dict(),
                            'acc': best_acc,
                            'iter_num': best_iter}
                    torch.save(ckpt, args.save_path)

            detector.train()



def obtain_label(loader, detector, args, c=None, device=None):
    start_test = True
    with torch.no_grad():
        for data, logits, _, labels, idxs in loader:
            logits = logits.view((data.size(0), -1))
            outputs = detector(logits.float())
            feas = detector.features_test
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    print(log_str)
    return pred_label.astype('int')






def calc_metrics(loader, args, detector=None):

    if detector is None:
        detector = Net()

        print(f'Loading Detector From: {args.save_path}')
        try:
            ckpt = torch.load(args.save_path)
        except:
            print(f'Model Not found at: {args.save_path}')

        print(f'Best Acc: {ckpt["acc"]}')
        detector.load_state_dict(ckpt['detector_state_dict'])
        detector.to(args.device)
    
    else: 
        print(f'Metric Mid-Training...')

    detector.eval()
    all_op = []
    start_test = True


    for data, logits, _, labels, idxs in loader:

        data, logits, labels = data.to(args.device), logits.to(args.device), labels.to(args.device)
        logits = logits.view((data.size(0), -1))
        op = detector(logits.float())
        
        if start_test:
            all_op = op.float().cpu()
            all_lbl = labels.float().cpu()
            start_test = False
        else:
            all_op = torch.cat((all_op, op.float().cpu()), 0)
            all_lbl = torch.cat((all_lbl, labels.float().cpu()), 0)

    ## once you have all outputs
    all_op = nn.Softmax(dim=1)(all_op)
    _, predict = torch.max(all_op, 1)
    print('='*100)

    benign_rate = 0
    benign_guesses = 0
    ad_guesses = 0
    ad_rate = 0
    for i in range(len(predict)):
        if predict[i] == 0:
            benign_guesses +=1
            if all_lbl[i]==0:
                benign_rate +=1
        else:
            ad_guesses +=1
            if all_lbl[i]==1:
                ad_rate +=1

    acc = (benign_rate+ad_rate)/len(predict)        
    TP = 2*ad_rate/len(predict)
    TN = 2*benign_rate/len(predict)
    precision = ad_rate/ad_guesses
    print('True positive rate/adversarial detetcion rate/recall/sensitivity is ', round(100*TP,2))
    print('True negative rate/normal detetcion rate/selectivity is ', round(100*TN,2))
    print('Precision',round(100*precision,1))
    print('The accuracy is',round(100*acc,2))
    print('='*100)
    
    


def main():

    ## Add Arguments
    parser = argparse.ArgumentParser(description='Adapt Source Detector to Target Detector')
    
    parser.add_argument('--dataset',help='Target Dataset',default='cifar10')
    parser.add_argument('--s_dataset',help='Source Dataset',default='fmnist')
    parser.add_argument('--batch_size',help='Batch Size',default=64,type=int)
    parser.add_argument('--model_name',help='Model Choice', default='resnet18')
    parser.add_argument("--attack",help='Attack choice', default = "auto_attack", choices=['fgsm', 'pgd', 'auto_attack', 'mifgsm', 'deepfool', 'bim'],type=str)
    parser.add_argument('--gpu',help='Model Choice', default='0')
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--cls_par', type=float, default=0.5)
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")    
    parser.add_argument('--p', help='dropout rate', default=0.65, type = float)


    args = parser.parse_args()
    args.root = './input/' + args.dataset + '/'
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    args.device = device

    log_path = './logs/detection/'
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    suffix = args.dataset + "_" + args.model_name + '_' + args.attack + '_ent' + str(args.ent_par) + '_cls' + str(args.cls_par)
    sys.stdout = Logger(log_path + suffix + '.txt')

    args.load_source_path = f'./source_detectors/{args.s_dataset}_resnet18_pgd_source_detector.pt'
    print(args)

    print(f"Model : {args.model_name} \t|\t Dataset : {args.dataset} \t|\t Attack : {args.attack}")
    clean_data, dataloaders = load_data(args.root, args.batch_size, data=True)
    adv_data, adv_loader = get_adv_data(args, data=True)

    pre_clf = load_model(args)
    print('Data and Model Loaded...') 
    evaluate_attack(dataloaders, pre_clf, args)

    combined_test_data = CombDataset(clean_data, adv_data, pre_clf, device=device, transform='logits', return_idx=True)
    combined_test_loader = torch.utils.data.DataLoader(combined_test_data, batch_size=args.batch_size, shuffle=True)
    print(f'Combined Data Size : ',len(combined_test_data))
    
    ## Load Detector
    detector = load_source_detector(args)

    ## Evaluate Performance Without Adaptation
    print('-'*100)
    print('Performance of T-I detector w/0 adaptation (Initialized with Source Weights)')
    detector.to(args.device)
    detector.eval()
    eval_dec(detector, combined_test_loader, device)
    print('-'*100)

    args.save_path = './target_detectors/'
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    args.save_path += f'{args.s_dataset}_{args.dataset}_{args.attack}_{args.model_name}_target_detector.pt'
    adapt_detector(detector, combined_test_loader, args, device)

    calc_metrics(combined_test_loader, args)
    print('= '*75)





if __name__ == '__main__':
    main()