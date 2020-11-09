from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from Autoencoder import Autoencoder 
import argparse
import numpy as np 
import config
import matplotlib.pyplot as plt

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        
        # As target is same as data
        target = data.detach().clone()
        
        # To match input dimension flatten target
        target = torch.flatten(target, start_dim=1)      
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        # ======================================================================
        # Compute loss based on criterion
        # ======================================================================        
        loss = criterion(output, target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
        
        
    train_loss = float(np.mean(losses))
    print('Train set: Average loss: {:.4f}\n'.format(float(np.mean(losses))))
    return train_loss
    


def test(model, device, test_loader, criterion):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            
            # As target is same as data
            target = data.detach().clone()
        
            # To match input dimension flatten target
            target = torch.flatten(target, start_dim=1)  
        
            data, target = data.to(device), target.to(device)
             
            # Predict for data by doing forward pass
            output = model(data)

            # ======================================================================
            # Compute loss based on same criterion as training
            # ======================================================================         
            loss = criterion(output, target)

            # Append loss to overall test loss
            losses.append(loss.item())
 
    test_loss = float(np.mean(losses))

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    
    return test_loss
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def find_reconstruction(model, device, data, criterion):
    '''
    Returns reconstruction of the given image
    model - trained model
    device - cuda or cpu
    data - image tensor to reconstruct
    criterion - loss function to calculate reconstruction loss
    '''
    model.eval()
    
    # As target is same as data
    target = data.detach().clone()
    
    #flat target to match input dimension
    target = torch.flatten(target, start_dim=1)
    
    # Push data/label to correct device
    data, target = data.to(device), target.to(device)
    
    # Predict for data by doing forward pass
    output = model(data)
    
    # ======================================================================
    # Compute loss based on criterion
    # ======================================================================   
    loss = criterion(output, target)
    
    #reshape to original size
    output = output.view(-1, 28, 28).cpu().squeeze_().detach()
    target = target.view(-1, 28, 28).cpu().squeeze_().detach()
    
    loss = loss.cpu().data
    
    return output, target, loss
    
    
def generate_reconstruction_dict(reconstruction_test_loader):
    '''
    Generate 2 reconstructions for 2 samples for each class
    A reconstruction_test_loader is a new data loader with batch_size as 1
    trained model
    device
    loss function to store losses
    returns image data & rescontruction data dicts ordered by class labels
    '''

    with torch.no_grad():
        datadict = dict()
        for key in range(10):
            datadict[key] = {'count':0, 'datalist':[]}
        #collect the samples two per class
        for sample in reconstruction_test_loader:
            data, target = sample
            #print(data.shape)
            key = int(target[0])
            if datadict[key]['count'] < 2:
                datadict[key]['datalist'].append(data)
                datadict[key]['count'] += 1
            isfull = True
            for i in range(10):
                if datadict[i]['count'] < 2:
                    isfull = False
            if isfull:
                break
        return datadict
        

def plot_reconstructions(model, device, criterion, datadict=None):
    '''
    Plotting the reconstructions 
    '''
    if datadict is not None:
        reconstruction_dict = dict()
        for key in range(10):
            reconstruction_dict[key] = {'losses':[], 'datalist':[]}
        for key in datadict:
            for count in range(datadict[key]['count']):
                data = datadict[key]['datalist'][count]
               
                reconstructed_data, _, loss = find_reconstruction(model, device, data, criterion)
                reconstruction_dict[key]['datalist'].append(reconstructed_data)
                reconstruction_dict[key]['losses'].append(loss)
    
    lossesdict = {} #collect losses per sample
    overallloss = 0.0
    fig, axs = plt.subplots(10, 4, figsize=(28, 28))
    for key in datadict:
        lossesdict[key] = 0.0
        datas = datadict[key]['datalist']
        reconstructions = reconstruction_dict[key]['datalist']
        losses = reconstruction_dict[key]['losses']
        lossesdict[key] = float((losses[0] + losses[1])/2.0)
        overallloss += float((losses[0] + losses[1])/2.0)
        
        axs[key][0].imshow(datas[0].squeeze_(dim=0).squeeze_(dim=0).numpy(), cmap="gray")
        axs[key][0].xaxis.set_visible(False)
        axs[key][0].yaxis.set_visible(False)
        axs[key][0].set_title('Sample 1')

        axs[key][1].imshow(reconstructions[0].numpy(), cmap="gray")
        axs[key][1].xaxis.set_visible(False)
        axs[key][1].yaxis.set_visible(False)
        axs[key][1].set_title('Sample 1 loss {:.4f}'.format(float(losses[0])) )

        axs[key][2].imshow(datas[1].squeeze_(dim=0).squeeze_(dim=0).numpy(), cmap="gray")
        axs[key][2].xaxis.set_visible(False)
        axs[key][2].yaxis.set_visible(False)
        axs[key][2].set_title('Sample 2')

        axs[key][3].imshow(reconstructions[1].numpy(), cmap="gray")
        axs[key][3].xaxis.set_visible(False)
        axs[key][3].yaxis.set_visible(False)
        axs[key][3].set_title('Sample 2 loss {:.4f}'.format(float(losses[1])) )
    
    plt.show()
    return lossesdict, overallloss/20.0


def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    # Initialize the model and send to device 
    model = Autoencoder(FLAGS.mode).to(device)
    
    # ======================================================================
    # Define loss function.
    # ======================================================================
    criterion = nn.MSELoss()
    # ======================================================================
    # Define optimizer function.
    # ======================================================================   
    optimizer = optim.SGD(model.parameters(), lr = FLAGS.learning_rate)    
    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = datasets.MNIST('./data/', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False,
                       transform=transform)
    train_loader = DataLoader(dataset1, batch_size = FLAGS.batch_size, 
                                shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size = FLAGS.batch_size, 
                                shuffle=False, num_workers=4)
                                
    print('Network in ' +FLAGS.mode)
    
    encoder_params, decoder_params = model.get_no_of_parameters(FLAGS.mode)
    total_params = count_parameters(model)
    
    print('Total parameters is = ', total_params)

    print('Encoder parameters is = ', encoder_params)

    print('Decoder parameters is = ', decoder_params)

    best_loss = np.Inf
    save_path = './best_weights/' + FLAGS.mode + '_best.pth'
    log_path = './logs/' + FLAGS.mode
    
    # Create summary writer object in specified folder. 
    writer = SummaryWriter(log_path, comment="Test_01_LR_1e-3")

    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch, FLAGS.batch_size)
        test_loss = test(model, device, test_loader, criterion)
       
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        if test_loss < best_loss:
            best_loss = test_loss
            states = {
                'epoch' : epoch,
                'state_dict' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_loss' : best_loss
            }
            torch.save(states, save_path)
        print('Model Saved')

    torch.save(model, 'Output.txt')
    print("Training and evaluation finished")
    
    writer.flush()
    writer.close()
    
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('Autoencoder Exercise.')
    parser.add_argument('--mode',
                        type=str, default='conv',
                        help='Select mode between fc and conv.')
    parser.add_argument('--learning_rate',
                        type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=10,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
    
    