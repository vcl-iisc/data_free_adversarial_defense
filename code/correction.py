import tqdm
import argparse

import torch

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from utils import get_freq, evaluate_attack, get_adv_data, load_model, load_data


def enable_dropout(m):

    ## Enable Dropout for populations

    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()


def get_ssim(X, Y):

    ## Calculate SSIM between data and data component

    ssim_val = ssim( X, Y, data_range=torch.max(X).item(), size_average=False,  nonnegative_ssim=True) # return (N,)
    return ssim_val


def correct_data(args, model, data, labels):

    ## Get Precition of Original Sample
    model.eval()
    output_adv = model(data)
    _, pred_adv = torch.max(output_adv, 1)
    pred_adv = pred_adv.item()

    metrics = {str(r):0 for r in range(2, args.r_range+1, 2)}
    ssim_metric = {str(r):0 for r in range(2, args.r_range+1, 2)}

    ## For each radius
    for r in range(2, args.r_range, 2):

        pred_low_list = []
        enable_dropout(model)
        torch.manual_seed(42) ## Same seeds for different radiuses

        ## For each population
        for idx in range(1, args.pop+1):
            
            ## Run multiple forward passes -> different classifiers
            data_l, data_h = get_freq(data, r, args.dataset) ## Get Low-High Componenets
            data_l = data_l.to(args.device, dtype=torch.float)
            output_low = model(data_l)
            _, pred_low = torch.max(output_low, 1)
            pred_low = pred_low.item() ## Prediction on Low Frequency Sample
            pred_low_list.append(pred_low) ## Append low prediction

            ssim_metric[str(r)] += (get_ssim(data, data_l).item() / args.pop)


        ## Check for how many models out of populations did the label predicted differ from original adversarial pertubation
        lcr_rad = sum([p_low != pred_adv for p_low in pred_low_list])
        metrics[str(r)] = lcr_rad 
    
    ## Pick the maximum non-zero lcr radius
    best_r = 4
    for idx, (r, lcr) in enumerate(metrics.items()):
        lcr_temp =  (args.pop-lcr) / args.pop
        if ssim_metric[r] - lcr_temp <= 0:
            break
        else:
            best_r = int(list(metrics.keys())[idx]) ## Max non-zero r
            
    ## Get prediction on low-pass version of non-dropout model
    model.eval()
    data_best_r, _ = get_freq(data, r=best_r, dataset=args.dataset)
    data_best_r = data_best_r.to(args.device, dtype=torch.float)
    if args.return_corr_data:
        return data_best_r

    output_best_r = model(data_best_r)
    _, pred_best_r = torch.max(output_best_r, 1)
    corr_best_r = (pred_best_r == labels).float().sum(0).item()
    return corr_best_r, best_r



def correct_module(model, args):
    
    loader = get_adv_data(args, 1)

    pbar = tqdm.tqdm(enumerate(loader), unit='batches', leave=False, total=len(loader), ascii=True, ncols=150)
    correct, total = 0.,0.

    rad_count = {str(r):0 for r in range(2, args.r_range+1, 2)}

    ## For each sample
    for batch_idx, (data, labels) in pbar:

        data, labels = data.to(args.device), labels.to(args.device)
        total += data.size(0)

        pbar.set_description(f"Acc : {(correct/total)*100.:.2f}")

        corr_best_r, best_r = correct_data(args, model, data, labels)
        
        rad_count[str(best_r)] += 1 
        correct += corr_best_r

    acc = (correct/total)*100.
    return acc, correct, total


def main():
    
    ## Add Arguments
    parser = argparse.ArgumentParser(description='Check Feature Score')

    parser.add_argument('--dataset',help='Dataset',default='cifar10')
    parser.add_argument('--batch_size',help='Batch Size',default=1,type=int)
    parser.add_argument("--attack",help='Attack choice', default = "pgd", choices=['pgd', 'auto_attack', 'bim'],type=str)
    parser.add_argument('--model_name',help='Model Choice', default='resnet18')
    parser.add_argument('--r_range', help='max radius range', default=16, type = int)
    parser.add_argument('--pop', help='population count for each radius', default=10, type = int)
    parser.add_argument('--gpu',help='Model Choice', default='0')
    parser.add_argument('--return_corr_data',help='Only Return Correct Data', action='store_true')
    parser.add_argument('--p', help='dropout rate', default=0.65, type = float)


    args = parser.parse_args()
    args.root = './input/' + args.dataset + '/'
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    args.device = device


    print(f"Model : {args.model_name} \t|\t Dataset : {args.dataset} \t|\t Attack : {args.attack}")

    _, _, test_loader = load_data(args.root, args.batch_size)
    model = load_model(args)

    # Check Baseline Accuracy
    evaluate_attack(test_loader, model, args)
    print('='*100)

    acc, corr, total = correct_module(model, args)
    print(f'Accuracy : {acc} \t|\t Correct : {corr} \t|\t Total : {total}  (Assuming Ideal Detector)')
    print('='*100)


if __name__  == '__main__':
    main()