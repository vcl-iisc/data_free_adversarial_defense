import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm


import os
import argparse
import tqdm

from utils import load_model, load_data_source_detector, CombDataset, get_loader_source_detector, evaluate_attack, AverageMeter


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


def train_detector(loader, args, device=None):

    epochs = 40
    detector = Net().to(device)
    
    criterion_id = nn.CrossEntropyLoss().to(device)

    optimizer = optim.RMSprop(detector.parameters(), lr=0.0005)
    # optimizer = optim.SGD([{'params':detector.parameters(), 'lr':0.1}],
    #             weight_decay=5e-4, momentum=0.9, nesterov=True)

    detector_acc = AverageMeter()
    detector_loss = AverageMeter()

    
    print(f'saving @ : {args.save_path}')

    pbar = tqdm.tqdm(range(epochs), leave=False)

    for epoch in pbar:
        pbar.set_description(f'Acc : {detector_acc.avg} | Loss : {detector_loss.avg}')

        for data, logits, _, labels in loader:

            optimizer.zero_grad()

            data, logits, labels = data.to(device), logits.to(device), labels.to(device)
            logits = logits.view((data.size(0), -1))

            output = detector(logits.float())

            loss = criterion_id(output, labels)
            loss.backward()
            optimizer.step()

            _, pred = output.max(1)
            acc = (pred.eq(labels).sum().item() / pred.size(0)) * 100.
            detector_acc.update(acc, data.size(0)) ## Detector Accuracy Update
            detector_loss.update(loss.item(), data.size(0))

            torch.save({
            'detector_state_dict': detector.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': detector_acc.avg,
            'loss': detector_loss.avg
            }, args.save_path)


    print(f'Accuracy : {detector_acc.avg} \t|\t Loss : {detector_loss.avg}')



def calc_metrics(loader, args):

    detector = Net()

    try:
        ckpt = torch.load(args.save_path)
    except:
        print(f'Loading Model From: {args.save_path} | args={args}')

    detector.load_state_dict(ckpt['detector_state_dict'])
    detector.to(args.device)

    detector.eval()
    all_op = []
    start_test = True


    for data, logits, _, labels in tqdm.tqdm(loader):

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
    accuracy = torch.sum(torch.squeeze(predict).float() == all_lbl).item() / float(all_lbl.size()[0])

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
    parser = argparse.ArgumentParser(description='Train Source Detector')
    
    parser.add_argument('--dataset',help='Dataset',default='fmnist') ## 'source/arbitrary' dataset
    parser.add_argument('--batch_size',help='Batch Size',default=100,type=int) 
    parser.add_argument('--model_name',help='Model Choice', default='resnet18') ## 'model' -> F_s
    parser.add_argument("--attack",help='Attack choice', default = "pgd", choices=['fgsm', 'pgd', 'auto_attack', 'mifgsm', 'deepfool'],type=str) ## Same attack for all arbitrary dataset (With Diff. params though)
    parser.add_argument('--gpu',help='GPU Choice', default='0')
    parser.add_argument('--only_eval',help='Only Do Detector Evaluation', action='store_true')
    parser.add_argument('--p', help='dropout rate', default=0.65, type = float)


    args = parser.parse_args()
    args.root = './input/' + args.dataset + '/'
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    args.device = device

    save_path = './source_detectors/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    save_path += f'{args.dataset}_{args.model_name}_{args.attack}_source_detector.pt'
    args.save_path = save_path

    print(f"Model : {args.model_name} \t|\t Dataset : {args.dataset} \t|\t Attack : {args.attack}")
    img_dataset, dataloaders = load_data_source_detector(args.root, args.batch_size)
    adv_data, _ = get_loader_source_detector(args, data=True) ## Get Adversarial Dataset

    pre_clf = load_model(args)
    print('Data and Model Loaded...') 
    
    evaluate_attack(dataloaders, pre_clf, args)

    combined_test_data = CombDataset(img_dataset, adv_data, pre_clf, device=device, transform='logits')
    combined_test_loader = torch.utils.data.DataLoader(combined_test_data, batch_size=args.batch_size, shuffle=True)
    print(f'Combined Dataset Length: {len(combined_test_data)}')
    
    skip_train = False
    if os.path.exists(args.save_path):
        print(f'Detector already trained..')
        if args.only_eval:
            skip_train = True
            print(f'\tSkipping Training')
        else: 
            print(f'\tOverwritting Saved Detector @ {args.save_path}')

    if not skip_train:
        print(f'Training Detector...')
        train_detector(combined_test_loader, args, device=device)

    print(f'Evaluating Detector...')
    calc_metrics(combined_test_loader, args)


if __name__ == '__main__':
    main()