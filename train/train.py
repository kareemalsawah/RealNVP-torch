import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_epoch(model, optimizer, train_loader):
    device = next(model.parameters()).device

    model.train()
    train_losses = []

    for imgs, labels in tqdm(train_loader,position=0,leave=True):
        imgs, labels = imgs.to(device), labels.to(device)

        log_prob_x = model.log_prob(imgs)

        loss = -1*torch.mean(log_prob_x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
    
    return train_losses

def test(model, test_loader):
    device = next(model.parameters()).device

    model.eval()
    test_losses = 0
    num_test = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            log_prob_x = model.log_prob(imgs)

            loss = -1*torch.sum(log_prob_x)

            test_losses += loss.item()
            num_test += imgs.shape[0]
    
    return test_losses/num_test

def train(model, optimizer, n_epochs, train_loader, test_loader, plot=False):
    train_losses = []
    test_losses = []

    for e in range(1, n_epochs+1):
        test_loss_epoch = test(model,test_loader)
        if e > 1:
            test_losses.append(test_loss_epoch)
        
        print("Epoch {}, Test Loss: {}".format(e, test_loss_epoch))

        train_loss_epoch = train_epoch(model,optimizer,train_loader)
        if e > 1:
            train_losses.extend(train_loss_epoch)
    
    test_loss_epoch = test(model,test_loader)
    test_losses.append(test_loss_epoch)

    print("Final Test Loss: {}".format(test_loss_epoch))

    if plot:
        num_tr = len(train_losses)
        num_ts = len(test_losses)
        itrs_per_epoch = num_tr/(num_ts-1)

        plt.title("Training and testing bits/dim throughout training")
        plt.plot(np.arange(0,num_tr),train_losses,label="Train bits/dim")
        plt.plot(np.arange(0,num_ts)*itrs_per_epoch,test_losses,label="Test bits/dim")
        plt.legend()
        plt.show()

    return train_losses, test_losses

def generate_samples(model,num_samples=32,floor=True,num_to_plot=0,samples_per_row=5,figsize=(15,8)):
    device = next(model.parameters()).device

    z = model.z_dist.sample((num_samples,*model.in_shape)).to(device)

    model.eval()
    with torch.no_grad():
        x, _ = model.forward(z, invert=True)

    if floor:
        x = torch.clamp(torch.floor(x),min=0,max=1)

    x = x.cpu().numpy()

    if num_to_plot>0:
        num_samples = num_to_plot
        num_rows = int(np.ceil(num_samples/samples_per_row))
        fig, ax = plt.subplots(num_rows,samples_per_row,figsize=figsize)

        # Draw samples
        for i in range(num_samples):
            ax[i//samples_per_row, i%samples_per_row].imshow(x[i][0], cmap="gray")

    return x