import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
    
        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 40 convolutional features, with a square kernel size of 5 and stride 1       
        self.conv1 = nn.Conv2d(in_channels=1, 
                                out_channels=40,
                                kernel_size=5,
                                stride=1)
        # Second 2D convolutional layer, taking in 40 input channel,
        # outputting 40 convolutional features, with a square kernel size of 5 and stride 1                             
        self.conv2 = nn.Conv2d(40, 40, (5,5), (1,1))
        
        # Add dropout of rate 0.5
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        
        # 2D convolutional layer, taking in 1 input channel (image),
        # outputting 40 convolutional features, with a square kernel size of 5, stride 1 abd padding 2  
        self.conv51 = nn.Conv2d(in_channels=1, 
                                out_channels=40,
                                kernel_size=5,
                                stride=1, padding = 2)
                                
        # 2D convolutional layer, taking in 40 input channel,
        # outputting 40 convolutional features, with a square kernel size of 5, stride 1 abd padding 2                                   
        self.conv52 = nn.Conv2d(in_channels=40, 
                                out_channels=40,
                                kernel_size=5,
                                stride=1, padding = 2)
                                
        #Input size is 28 X 28 and 1 is the depth of the given image = 1 * 28 * 28
        #Fully Connected layers with neurons 100
        self.fc1 = nn.Linear(784, 100);
        
        #Fully Connected layers with input_neurons=40*4*4, neurons=100
        self.fc2 = nn.Linear(640, 100);
        
        #Fully Connected layers with input_neurons=40*7*7, neurons=100
        self.fc51 = nn.Linear(1960, 1000);
        
        #Fully Connected layers with both input and output neurons 1000
        self.fc52 = nn.Linear(1000, 1000);
       
        #Fully Connected output layers for 10 class prediction
        self.fc5out = nn.Linear(1000, 10);
        
        #Fully Connected layers with both input and output neurons 100
        self.fc3 = nn.Linear(100, 100);
        
        #Fully Connected output layers for 10 class prediction
        self.fcout = nn.Linear(100, 10);

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connnected layer.
        # ======================================================================
      
        x = torch.flatten(X, start_dim = 1)

        x = self.fc1(x);
        x = torch.sigmoid(x);
        x = self.fcout(x);
        output = x;
        return output;
        
    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        # ======================================================================
      
        x = self.conv1(X)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc2(x);
        x = torch.sigmoid(x);
        x = self.fcout(x);
        
        
        output = x;
        return output;

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        # ======================================================================
        
        x = self.conv1(X)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, start_dim = 1)
        x = self.fc2(x);
        x = F.relu(x);
        x = self.fcout(x);
        
        
        output = x;
        return output;
        
    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # ======================================================================
        
        x = self.conv1(X)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, start_dim = 1)
        x = self.fc2(x);
        x = F.relu(x);
        x = self.fc3(x);
        x = F.relu(x);
        x = self.fcout(x);

        output = x;
        return output;
        
    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        # ======================================================================
        
        x = self.conv51(X)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv52(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc51(x);
        x = F.relu(x);
        x = self.dropout2(x)
        x = self.fc52(x);
        x = F.relu(x);
        x = self.fc5out(x);

        output = x;
        return output;    
    
