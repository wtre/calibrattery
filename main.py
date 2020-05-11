# Importing Libraries
import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import seaborn as sns
import torch.nn.init as init
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
from pytorchtools import EarlyStopping

# Custom Libraries
import utils
import time

# Tensorboard initialization
writer = SummaryWriter()

# Plotting Style
sns.set_style('darkgrid')

# Main
def main(args, ITE=0):
    import pandas as pd
    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinit = True if args.prune_type=="reinit" else False
    layerwise = True if args.prune_type == "layerwise" else False

    # Data Loader
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    transform_cifar10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if args.dataset == "mnist":
        traindataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar10":
        traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform_cifar10)
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform_cifar10)
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet, minivgg

    elif args.dataset == "fashionmnist":
        traindataset = datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet 

    elif args.dataset == "cifar100":
        traindataset = datasets.CIFAR100('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR100('../data', train=False, transform=transform)   
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet  
    
    # If you want to add extra datasets paste here

    else:
        print("\nWrong Dataset choice \n")
        exit()

    # obtain training indices that will be used for validation
    if args.early_stopping:
        print(' Splitting Validation sets ')
        trainset_size = int((1 - args.valid_size) * len(traindataset))
        valset_size = len(traindataset) - trainset_size
        trainset, valset = torch.utils.data.random_split(traindataset, [trainset_size, valset_size])

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                                  drop_last=False)
        valid_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                                  drop_last=False)

    else:
        print(' Eww, no validation set? ')
        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0,drop_last=False)

    # train_loader = cycle(train_loader)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0,drop_last=True)

    
    # Importing Network Architecture
    global model
    if args.arch_type == "fc1":
       model = fc1.fc1().to(device)
    elif args.arch_type == "lenet5":
        model = LeNet5.LeNet5().to(device)
    elif args.arch_type == "alexnet":
        model = AlexNet.AlexNet().to(device)
    elif args.arch_type == "vgg16":
        model = vgg.vgg16().to(device)  
    elif args.arch_type == "resnet18":
        model = resnet.resnet18().to(device)   
    elif args.arch_type == "densenet121":
        model = densenet.densenet121().to(device)   
    # If you want to add extra model paste here
    elif args.arch_type == "conv2":
        model = minivgg.conv2().to(device)
    elif args.arch_type == "conv4":
        model = minivgg.conv4().to(device)
    elif args.arch_type == "conv6":
        model = minivgg.conv6().to(device)

    else:
        print("\nWrong Model choice\n")
        exit()

    # Weight Initialization. Warning! This drops test acc, so i'm examining this function.
    model.apply(weight_init)

    # get time for file path
    import datetime
    now = datetime.datetime.now()
    now_ = now.strftime("%02m%02d%02H%02M_")

    # Copying and Saving Initial State
    print('  saving initial model... ')
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/saves/{now_}{args.arch_type}/{args.dataset}/")
    torch.save(model, f"{os.getcwd()}/saves/{now_}{args.arch_type}/{args.dataset}/initial_state_dict_{args.prune_type}.pth.tar")
    print("  initial model saved in ", f"{os.getcwd()}/saves/{now_}{args.arch_type}/{args.dataset}/initial_state_dict_{args.prune_type}.pth.tar")

    # Making Initial Mask
    make_mask(model)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss() # Default was F.nll_loss

    # Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size())

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION,float)
    bestacc = np.zeros(ITERATION,float)
    step = 0
    all_loss = np.zeros(args.end_iter,float)
    all_vloss = np.zeros(args.end_iter, float)
    all_accuracy = np.zeros(args.end_iter,float)

    for _ite in range(args.start_iter, ITERATION):

        # Early stopping parameter for each pruning iteration
        early_stopping = EarlyStopping(patience=99, verbose=True) ######### we don't stop, party all night

        if not _ite == 0:
            prune_by_percentile(args.prune_percent, args.fc_prune_percent, resample=resample,
                                reinit=reinit, layerwise=layerwise, if_split=args.split_conv_and_fc)
            if reinit:
                model.apply(weight_init)
                #if args.arch_type == "fc1":
                #    model = fc1.fc1().to(device)
                #elif args.arch_type == "lenet5":
                #    model = LeNet5.LeNet5().to(device)
                #elif args.arch_type == "alexnet":
                #    model = AlexNet.AlexNet().to(device)
                #elif args.arch_type == "vgg16":
                #    model = vgg.vgg16().to(device)  
                #elif args.arch_type == "resnet18":
                #    model = resnet.resnet18().to(device)   
                #elif args.arch_type == "densenet121":
                #    model = densenet.densenet121().to(device)   
                #else:
                #    print("\nWrong Model choice\n")
                #    exit()
                step = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        weight_dev = param.device
                        param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
                        step = step + 1
                step = 0
            else:
                original_initialization(mask, initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        time.sleep(0.25)
        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1

        # pbar = range(args.end_iter)
        pbar = tqdm(range(args.end_iter))

        stop_flag = False
        for iter_ in pbar:

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy = test(model, test_loader, criterion)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    # We don't save model per test-acc, will use validation-acc!
                    # utils.checkdir(f"{os.getcwd()}/saves/{now_}{args.arch_type}/{args.dataset}/")
                    # torch.save(model,f"{os.getcwd()}/saves/{now_}{args.arch_type}/{args.dataset}/{_ite}_model_{args.prune_type}.pth.tar")

            # Training
            loss = train(model, train_loader, optimizer, criterion)

            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy

            # Validating
            valid_loss, loss_v = validate(model, valid_loader, optimizer, criterion)
            all_vloss[iter_] = valid_loss #loss_v

            # early stopping
            checkpoint_path = f"{os.getcwd()}/saves/{now_}{args.arch_type}/{args.dataset}/"
            save_path = f"{os.getcwd()}/saves/{now_}{args.arch_type}/{args.dataset}/{_ite}_model_{args.prune_type}.pth.tar"
            # msg = early_stopping(valid_loss, model, checkpoint_path, save_path)
            early_stopping(valid_loss, model, checkpoint_path, save_path)

            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                time.sleep(0.25)
                pbar.set_description(
                    # f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}% \t' + msg)
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} V-Loss: {valid_loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')
                if iter_ % 5 == 4:
                    print('')

            if early_stopping.early_stop and not stop_flag:
                print("Early stopping")
                stop_flag = True
                # break


        writer.add_scalar('Accuracy/test', best_accuracy, comp1)
        bestacc[_ite]=best_accuracy

        # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
        #NOTE Loss is computed for every iteration while Accuracy is computed only for every {args.valid_freq} iterations. Therefore Accuracy saved is constant during the uncomputed iterations.
        #NOTE Normalized the accuracy to [0,100] for ease of plotting.
        plt.plot(np.arange(1,(args.end_iter)+1), 100*(all_loss - np.min(all_loss))/np.ptp(all_loss).astype(float), c="blue", label="Train loss")
        plt.plot(np.arange(1, (args.end_iter) + 1), 100 * (all_vloss - np.min(all_vloss)) / np.ptp(all_vloss).astype(float), c="green", label="Valid loss")
        plt.plot(np.arange(1,(args.end_iter)+1), all_accuracy, c="red", label="Accuracy") 
        plt.title(f"Loss Vs Accuracy Vs Iterations ({args.dataset},{now_}{args.arch_type})")
        plt.xlabel("Iterations") 
        plt.ylabel("Loss and Accuracy") 
        plt.legend() 
        plt.grid(color="gray") 
        utils.checkdir(f"{os.getcwd()}/plots/lt/{now_}{args.arch_type}/{args.dataset}/")
        plt.savefig(f"{os.getcwd()}/plots/lt/{now_}{args.arch_type}/{args.dataset}/{args.prune_type}_LossVsAccuracy_{comp1}.png", dpi=300)
        plt.close()

        # Dump Plot values
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{now_}{args.arch_type}/{args.dataset}/")
        all_loss.dump(f"{os.getcwd()}/dumps/lt/{now_}{args.arch_type}/{args.dataset}/{args.prune_type}_all_loss_{comp1}.dat")
        all_accuracy.dump(f"{os.getcwd()}/dumps/lt/{now_}{args.arch_type}/{args.dataset}/{args.prune_type}_all_accuracy_{comp1}.dat")
        
        # Dumping mask
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{now_}{args.arch_type}/{args.dataset}/")
        with open(f"{os.getcwd()}/dumps/lt/{now_}{args.arch_type}/{args.dataset}/{args.prune_type}_mask_{comp1}.pkl", 'wb') as fp:
            pickle.dump(mask, fp)
        
        # Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(args.end_iter,float)
        all_accuracy = np.zeros(args.end_iter,float)

        # Dumping Values for Plotting
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{now_}{args.arch_type}/{args.dataset}/")
        comp.dump(f"{os.getcwd()}/dumps/lt/{now_}{args.arch_type}/{args.dataset}/{args.prune_type}_compression.dat")
        bestacc.dump(f"{os.getcwd()}/dumps/lt/{now_}{args.arch_type}/{args.dataset}/{args.prune_type}_bestaccuracy.dat")

        # Plotting
        a = np.arange(args.prune_iterations)
        plt.plot(a, bestacc, c="blue", label="Winning tickets")
        plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{now_}{args.arch_type})")
        plt.xlabel("Unpruned Weights Percentage")
        plt.ylabel("test accuracy")
        plt.xticks(a, comp, rotation ="vertical")
        plt.ylim(0,100)
        plt.legend()
        plt.grid(color="gray")
        utils.checkdir(f"{os.getcwd()}/plots/lt/{now_}{args.arch_type}/{args.dataset}/")
        plt.savefig(f"{os.getcwd()}/plots/lt/{now_}{args.arch_type}/{args.dataset}/{args.prune_type}_AccuracyVsWeights.png", dpi=300)
        plt.close()

    print('Training ended~~~')

    # # Dumping Values for Plotting
    # utils.checkdir(f"{os.getcwd()}/dumps/lt/{now_}{args.arch_type}/{args.dataset}/")
    # comp.dump(f"{os.getcwd()}/dumps/lt/{now_}{args.arch_type}/{args.dataset}/{args.prune_type}_compression.dat")
    # bestacc.dump(f"{os.getcwd()}/dumps/lt/{now_}{args.arch_type}/{args.dataset}/{args.prune_type}_bestaccuracy.dat")
    #
    # # Plotting
    # a = np.arange(args.prune_iterations)
    # plt.plot(a, bestacc, c="blue", label="Winning tickets")
    # plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{now_}{args.arch_type})")
    # plt.xlabel("Unpruned Weights Percentage")
    # plt.ylabel("test accuracy")
    # plt.xticks(a, comp, rotation ="vertical")
    # plt.ylim(0,100)
    # plt.legend()
    # plt.grid(color="gray")
    # utils.checkdir(f"{os.getcwd()}/plots/lt/{now_}{args.arch_type}/{args.dataset}/")
    # plt.savefig(f"{os.getcwd()}/plots/lt/{now_}{args.arch_type}/{args.dataset}/{args.prune_type}_AccuracyVsWeights.png", dpi=300)
    # plt.close()


# Function for Training
def train(model, train_loader, optimizer, criterion):
    global mask
    if_prune_by_threshold = False   # True(original) has unexpected behavior
    if not if_prune_by_threshold:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tensored_mask = []
        for i in range(len(mask)):
            tensored_mask.append(torch.from_numpy(mask[i]).float().to(device))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_loss = 0.0

    train_losses = []
    EPS = 1e-6
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        if if_prune_by_threshold:
            for name, p in model.named_parameters():
                if 'weight' in name:
                    tensor = p.data.cpu().numpy()
                    grad_tensor = p.grad.data.cpu().numpy()
                    grad_tensor = np.where(abs(tensor) < EPS, 0, grad_tensor)
                    p.grad.data = torch.from_numpy(grad_tensor).to(device)
            optimizer.step()
        else:
            step = 0
            for name, p in model.named_parameters():
                if 'weight' in name:
                    p.grad.data = torch.where(tensored_mask[step] < 1, 0*p.grad.data, p.grad.data)
                    step = step + 1
            optimizer.step()

    return loss.item()

    #### Old training  below ####
    # EPS = 1e-6
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.train()
    # for batch_idx, (imgs, targets) in enumerate(train_loader):
    #     optimizer.zero_grad()
    #     #imgs, targets = next(train_loader)
    #     imgs, targets = imgs.to(device), targets.to(device)
    #     output = model(imgs)
    #     train_loss = criterion(output, targets)
    #     train_loss.backward()
    #
    #     # Freezing Pruned weights by making their gradients Zero
    #     for name, p in model.named_parameters():
    #         if 'weight' in name:
    #             tensor = p.data.cpu().numpy()
    #             grad_tensor = p.grad.data.cpu().numpy()
    #             grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
    #             p.grad.data = torch.from_numpy(grad_tensor).to(device)
    #     optimizer.step()
    # return train_loss.item()

# Function for Vadlidating
def validate(model, valid_loader, optimizer, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_losses = []
    model.eval()
    for i, (inputs, labels) in enumerate(valid_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        valid_losses.append(loss.item())

    valid_loss = np.average(valid_losses)

    return valid_loss, loss.item()



# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

# Prune by Percentile module
def prune_by_percentile(percent, fc_percent, resample=False, reinit=False, layerwise=False, if_split=True, **kwargs):
    global step
    global mask
    global model

    # Calculate percentile value
    step = 0
    if if_split:
        # print('yay! lettts split!!')
        # i won't implement split & layerwise case for now, it's useless

        # split % global case
        alive_conv = np.empty(shape=0)
        alive_fc = np.empty(shape=0)  # init 0d array
        step = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                tensor = np.multiply(param.data.cpu().numpy(), mask[step])   # this line is edited
                if 'features' in name:
                    alive_conv = np.concatenate((alive_conv, tensor[np.nonzero(tensor)]))
                elif 'classifier' in name:
                    alive_fc = np.concatenate((alive_fc, tensor[np.nonzero(tensor)]))
                step = step + 1
        conv_percentile_value = np.percentile(abs(alive_conv), percent)
        fc_percentile_value = np.percentile(abs(alive_fc), fc_percent)
        # print(f'>> conv threshold : {conv_percentile_value:15} (cut from {percent:3}%) | fc threshold : {fc_percentile_value:15} (cut from {fc_percent:3}%)')

        # Convert Tensors to numpy and calculate
        step = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                weight_dev = param.device
                if 'features' in name:
                    new_mask = np.where(abs(tensor) <= conv_percentile_value, 0, mask[step])
                elif 'classifier' in name:
                    new_mask = np.where(abs(tensor) <= fc_percentile_value, 0, mask[step])

                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                # print(f'  {step:2}th layer sum is {np.sum(new_mask):8}, ')
                mask[step] = new_mask
                step += 1

    else:   # use single threshold for pruning conv and FC.
        if layerwise:
            for name, param in model.named_parameters():

                # We do not prune bias term
                if 'weight' in name:
                    tensor = param.data.cpu().numpy()
                    alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                    percentile_value = np.percentile(abs(alive), percent)

                    # Convert Tensors to numpy and calculate
                    weight_dev = param.device
                    new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

                    # Apply new weight and mask
                    param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                    mask[step] = new_mask
                    step += 1
        else:
            alive = np.empty(shape=0)    # init 0d array
            for name, param in model.named_parameters():
                if 'weight' in name:
                    tensor = param.data.cpu().numpy()
                    alive = np.concatenate((alive,tensor[np.nonzero(tensor)]))  # flattened array of nonzero values
            percentile_value = np.percentile(abs(alive), percent)

            # Convert Tensors to numpy and calculate
            for name, param in model.named_parameters():
                if 'weight' in name:
                    tensor = param.data.cpu().numpy()
                    weight_dev = param.device
                    new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

                    # Apply new weight and mask
                    param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                    mask[step] = new_mask
                    step += 1
    step = 0


# Function to make an empty mask of the same size as the model
def make_mask(model):
    global step
    global mask
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            step = step + 1
    mask = [None]* step 
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0


def original_initialization(mask_temp, initial_state_dict):
    global model
    
    step = 0
    for name, param in model.named_parameters(): 
        if "weight" in name: 
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0


# Function for Initialization
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    # print('init type '+init_type)
    init_type = "uber"

    if init_type == "rahul":
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
                init.constant(m.bias.data, 0)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.weight.data, 1)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

    elif init_type == "uber":
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight.data, 1)
            init.constant_(m.bias.data, 0)




if __name__=="__main__":
    
    #from gooey import Gooey
    #@Gooey      
    
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 1.2e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=100, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=10, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=35, type=int, help="Pruning iterations count")

    parser.add_argument("--early_stopping", action="store_true", help="Split validation set to measure early stopping point")
    parser.add_argument("--valid_size", default=0.1, type=float, help="Size of validation set")
    parser.add_argument("--prune_scope", default="global", type=str, help="global | layerwise")
    parser.add_argument("--split_conv_and_fc", action="store_true", help="use separate prune rate for FC layer")
    parser.add_argument("--fc_prune_percent", default=10, type=int, help="Used only when --split_conv_and_fc==True")

    parser.add_argument("--init_type", default="uber", type=str, help="rahul | uber | pytorch")

    
    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    
    #FIXME resample
    resample = False

    # Looping Entire process
    #for i in range(0, 5):
    main(args, ITE=1)
