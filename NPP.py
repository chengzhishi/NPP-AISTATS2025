import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tools.plot_utils import plot_and_save
from tools.data_utils import *
from tools.losses import NPPLoss
from tools.models import Autoencoder
from tools.optimization import EarlyStoppingCallback, evaluate_model, train_model
import matplotlib.pyplot as plt
import argparse
import time
from tools.models import *
from torch.utils.data import Subset
from tools.NeuralPointProcess import NeuralPointProcesses

def run_experiments(config, train_loader, val_loader,
                    eval_loader, input_channel, input_shape, device):  
    best_val_loss_MSE = float('inf')
    best_val_loss_NPP = float('inf')
    timestamp = int(time.time())
    experiment_id = f"{timestamp}"
    dataset = config['dataset']
    deeper = config['deeper']
    kernel = config['kernel']
    manual_lr = config['manual_lr']
    num_runs = config['num_runs']
    val_every_epoch = config['val_every_epoch']
    epochs = config['epochs']
    exp_name = config['experiment_name']
    lr = config['lr']
    num_encoder = config['num_encoder']
    num_decoder = config['num_decoder']
    kernel = config['kernel']
    kernel_mode = config['kernel_mode']
    kernel_param = config['kernel_param']
    seed = config['seed']

    if dataset == "COWC":
        input_shape = 200
    elif dataset == "Building":
        input_shape = 100
    else:
        input_shape = 28
        
    losses = {}
    # Create storage directory and store the experiment configuration
    path = os.path.join(".", "history", exp_name, str(experiment_id))
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, "config.json"), "w") as outfile:
        json.dump(config, outfile)

    # run the experiments for num_rums times and save the best-performing model
    for run in range(num_runs):
        if kernel_param == 0:
            # run MSE train
            NPP = NeuralPointProcesses(identity=True, num_encoder=num_encoder, num_decoder=num_decoder, input_shape=input_shape, input_channel=input_channel, deeper=deeper, lr=lr, device=device)
            train_losses, val_losses, best_val_loss = NPP.train_model(train_loader, val_loader,
                                                                         epochs, experiment_id, exp_name, 
                                                                         best_val_loss_MSE, val_every_epoch)
            losses[f"MSE_run{run}_train"] = train_losses
            losses[f"MSE_run{run}_val"] = val_losses
            if best_val_loss < best_val_loss_MSE:
                best_val_loss_MSE = best_val_loss
            MSE_test_loss, MSE_test_R2, MSE_test_GR2 = NPP.evaluate_model(eval_loader)
            print(f"MSE Loss| Loss: {MSE_test_loss}, R2: {MSE_test_R2}, GR2: {MSE_test_GR2} ")
            
        else:
            # run NPP train
            NPP = NeuralPointProcesses(kernel=kernel, kernel_mode=kernel_mode, num_encoder=num_encoder, num_decoder=num_decoder, input_shape=input_shape, input_channel=input_channel, deeper=deeper, kernel_param=kernel_param, lr=lr)
            train_losses, val_losses, best_val_loss = NPP.train_model(train_loader, val_loader,
                                                                         epochs, experiment_id, exp_name, 
                                                                         best_val_loss_NPP, val_every_epoch)
            
            losses[f"NPP_run{run}_train"] = train_losses
            losses[f"NPP_run{run}_val"] = val_losses
            if best_val_loss < best_val_loss_NPP:
                best_val_loss_NPP = best_val_loss
               
            for percent in [0.00, 0.25, 0.50, 0.75, 1.00]:
                NPP_test_loss, NPP_test_R2, NPP_test_GR2 = NPP.evaluate_model(eval_loader, partial_percent=percent)
                print(f"Percent: {percent}| Loss: {NPP_test_loss}, R2: {NPP_test_R2}, GR2: {NPP_test_GR2}")
   
    if kernel_param == 0:
        # MSE test
        NPP.load_best_model(experiment_id, exp_name)
        MSE_test_loss, MSE_test_R2 = NPP.evaluate_model(eval_loader)
        f = open(f"./history/{exp_name}/{experiment_id}/results.txt", "w")
        f.write(f"Results {experiment_id}: Best Val: {best_val_loss_MSE} \n MSE: {MSE_test_loss}, R2: {MSE_test_R2}, GR2: {MSE_test_GR2}")
        f.close()
        print("metrics saved")
    else:
        # NPP test
        NPP.load_best_model(experiment_id, exp_name)
        f = open(f"./history/{exp_name}/{experiment_id}/results.txt", "a")
        f.write(f"Results {experiment_id}: Best Val: {best_val_loss_NPP} \n")
        for percent in [0.00, 0.25, 0.50, 0.75, 1.00]:
            print(f'Percent testing {percent}')  
            NPP_test_loss, NPP_test_R2, NPP_test_GR2 = NPP.evaluate_model(eval_loader, partial_percent=percent)
            print(f"Percent: {percent}| Loss: {NPP_test_loss}, R2: {NPP_test_R2}, GR2: {NPP_test_GR2} ")
            f.write(f"Percent: {percent}| Loss: {NPP_test_loss}, R2: {NPP_test_R2}, GR2: {NPP_test_GR2}  \n")
        f.close()
        print("metrics saved")
        
    print("start saving losses!")
    # Save losses
    save_loss(losses, f'./history/{exp_name}/{experiment_id}/losses.npy')
    return experiment_id


def parse_args():
    parser = argparse.ArgumentParser(description="Your script description here.")

    # Datasets and hyperparameters
    parser.add_argument("--dataset", type=str, default="PinMNIST", help="Dataset name")
    parser.add_argument("--modality", type=str, default="PS-RGBNIR", help="Building dataset modality")
    parser.add_argument("--feature", type=str, default="AE", help="feature from 'AE' or 'DDPM'")
    parser.add_argument("--mode", type=str, default="mesh", help="mode for 'mesh' or 'random'")
    parser.add_argument("--lr", type=float, default=0.001, help="Value for initial learning rate")
    parser.add_argument("--d", type=int, default=10, help="Value for 'd'")
    parser.add_argument("--n_pins", type=int, default=500, help="Value for 'n_pins'")
    parser.add_argument("--r", type=int, default=3, help="Value for 'r'")
    parser.add_argument("--seed", type=int, default=1, help="seed for pytorch and np")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--val_every_epoch", type=int, default=5, help="Number of epochs in between validations")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of different trainings to do per model and sigma")
    parser.add_argument("--manual_lr", action='store_true', default=False, help="Do not use Custom LR Finder")

    # List of kernel_param values
    parser.add_argument("--kernel_param", type=float, default=[1],
                        help="a kernel param: Q=1 (SMK) or sigma =1 (RBF)")

    # Model configuration
    parser.add_argument("--num_encoder", nargs="+", type=int, default=[64, 32], help="List of encoder kernel sizes")
    parser.add_argument("--num_decoder", nargs="+", type=int, default=[64], help="List of decoder kernel sizes")
    parser.add_argument("--deeper", action='store_true', default=False, help="Add extra convolutional layer for the model")
    parser.add_argument("--kernel_mode", type=str, default="fixed", help="fixed or learned, or predicted")
    parser.add_argument("--kernel", type=str, default="RBF", help="RBF or SM")
    
    # Experiment title
    parser.add_argument("--experiment_name", type=str, default=None, help="Define if you want to save the generated experiments in an specific folder")

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    # Set a random seed for PyTorch
    seed = args.seed  # You can use any integer value as the seed
    torch.manual_seed(seed)
    # Set a random seed for NumPy (if you're using NumPy operations)
    np.random.seed(seed)

    config = vars(args)
    dataset = config['dataset']
    input_channel, input_shape, train_loader, val_loader, eval_loader = data_prepare(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Run and save the pipeline data
    experiment_id = run_experiments(config, train_loader, val_loader, eval_loader, input_channel, input_shape, device)

    end_time = time.time()
    print(f"Time Elapsed: {(end_time - start_time) } seconds")
    f = open(f"./history/{config['experiment_name']}/{experiment_id}/results.txt", "a")
    f.write(f"Time Elapsed: {(end_time - start_time) } seconds")
    
    
if __name__ == "__main__":
    main()
