import os
import sys
import tqdm
import torch
import argparse

from correction import correct_data, correct_module
from train_target_detector import eval_dec, load_detector
from utils import get_correct, get_adv_data, load_data, load_model, evaluate_attack, Logger, CombDataset


def calc_comb_acc(loader, model, detector, args):
    
    model.eval()
    detector.eval()

    correct, total = 0,0 ## Overall Acc. -> Detector's Accuracy
    metrics = {'clean':{'correct':0, 'total':0+1e-5}, 'adv':{'correct':0, 'total':0+1e-5}} ## Classifier Acc. (Not Detector)
    pbar = tqdm.tqdm(loader, unit='batches', leave=False, total=len(loader), ascii=True, ncols=150)

    for (data, logits, labels, det_labels, _) in pbar:


        total += data.size(0)
        acc = (correct/total)*100.

        data, labels, logits, det_labels = data.to(args.device), labels.to(args.device), logits.to(args.device), det_labels.to(args.device)
        logits = logits.view((data.size(0), -1))


        if det_labels.item() == 1:
            key = 'adv'
        else: 
            key = 'clean'

        ## Check Detector's Prediction
        out_detect = detector(logits)
        _, pred_detect = torch.max(out_detect, 1)
        correct += get_correct(out_detect, det_labels)

        if pred_detect.item() == 1:
            data = correct_data(args, model, data, labels) ## Get Corrected Data
        
        model.eval() ## Since it could be coming in from dropout-enabled

        output = model(data)
        metrics[key]['correct'] += get_correct(output, labels)
        metrics[key]['total'] += data.size(0)

        pbar.set_description(f'D-Acc : {acc:.2f} | Clean-Acc : {(metrics["clean"]["correct"]/metrics["clean"]["total"])*100.:.2f} | Adv-Acc : {(metrics["adv"]["correct"]/metrics["adv"]["total"])*100.:.2f}')

    
    print(f'D-Acc : {acc:.2f} | Clean-Acc : {(metrics["clean"]["correct"]/metrics["clean"]["total"])*100.:.2f} | Adv-Acc : {(metrics["adv"]["correct"]/metrics["adv"]["total"])*100.:.2f}')
    print(f'Total: Detector : {total} \t|\t Clean : {metrics["clean"]["total"]} \t|\t Adv : {metrics["adv"]["total"]}')


def main():

    ## Add Arguments
    parser = argparse.ArgumentParser(description='Check Combined Performance')
    
    parser.add_argument('--dataset',help='Dataset',default='cifar10')
    parser.add_argument('--s_dataset',help='Arbitrary Dataset Choice', default='fmnist')
    parser.add_argument('--batch_size',help='Batch Size',default=1,type=int)
    parser.add_argument('--model_name',help='Model Choice', default='resnet18')
    parser.add_argument("--attack",help='Attack choice', default = "auto_attack", choices=['pgd', 'auto_attack', 'bim'],type=str)
    parser.add_argument('--r_range', help='max radius range', default=16, type = int)
    parser.add_argument('--pop', help='population count for each radius', default=10, type = int)
    parser.add_argument('--gpu',help='Model Choice', default='0')

    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--cls_par', type=float, default=0.5)
    parser.add_argument('--p', help='dropout rate', default=0.65, type = float)

    args = parser.parse_args()
    args.root = './input/' + args.dataset + '/'
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    log_path = './logs/combined/'
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    suffix = args.dataset + "_" + args.model_name + '_' + args.attack + '_ent' + str(args.ent_par) + '_cls' + str(args.cls_par)
    sys.stdout = Logger(log_path + suffix + '.txt')


    print(f"Model : {args.model_name} \t|\t Dataset : {args.dataset} \t|\t Arb. Dataset : {args.s_dataset} \t|\t Attack : {args.attack}")
    clean_data, clean_loader = load_data(args.root, args.batch_size, data=True)
    adv_data, _ = get_adv_data(args, data=True)
    
    pre_clf = load_model(args)
    print('Data and Model Loaded...')
    print('='*100)
    
    evaluate_attack(clean_loader, pre_clf, args)
    print('='*100)

    # Check Performance Assuming Ideal Detector
    # args.return_corr_data = False
    # acc, corr, total = correct_module(pre_clf, args)
    # print(f'Accuracy : {acc} \t|\t Correct : {corr} \t|\t Total : {total}  (Assuming Ideal Detector)')
    # print('='*100)

    combined_test_data = CombDataset(clean_data, adv_data, pre_clf, device=args.device, transform='logits', return_idx=True)
    combined_test_loader = torch.utils.data.DataLoader(combined_test_data, batch_size=args.batch_size, shuffle=True)
    print(f'Combined Data Size : ',len(combined_test_data))


    ## Load Detector
    args.load_target_path = f'./target_detectors/{args.s_dataset}_{args.dataset}_{args.attack}_{args.model_name}_target_detector.pt'
    detector = load_detector(args)

    ## Check Performance of detector
    detector.to(args.device)
    correct, total = eval_dec(detector, combined_test_loader, args.device)
    print('='*100)

    ## Check Combined Peformance
    args.return_corr_data = True
    calc_comb_acc(combined_test_loader, pre_clf, detector, args)


if __name__ == '__main__':
    main()