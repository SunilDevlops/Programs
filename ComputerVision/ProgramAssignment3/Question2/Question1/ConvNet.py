import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()   
        
        if mode == 'small':
        
            # First 2D convolutional layer, taking in 3 input channel (image),
            # outputting 32 convolutional features, with a square kernel size of 3, stride 1 and padding 1     
            self.conv1 =  nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding = 1)

            # Second 2D convolutional layer, taking in 32 input channel (image),
            # outputting 64 convolutional features, with a square kernel size of 3, stride 1 and padding 1 
            self.conv2 =  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding = 1)

            # Third 2D convolutional layer, taking in 64 input channel (image),
            # outputting 128 convolutional features, with a square kernel size of 3, stride 1 and padding 1 
            self.conv3 =  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1)

            #Average pooling
            self.avgpool = nn.AdaptiveAveragePool(4,4)
            
            #Fully Connected 1st layer with input_neurons = 2048, output_neurons=2048
            self.fc1 = nn.Linear(128*4*4, 2048)
            
            #Dropout1 of rate 0.5
            self.dropout1 = nn.Dropout2d(0.5)

            #Fully Connected 2nd layer with input_neurons=2048, output_neurons=512
            self.fc2 = nn.Linear(2048, 512)
            
            #Dropout2 of rate 0.5
            self.dropout2 = nn.Dropout2d(0.5)
 
            #Fully Connected final layer with input_neurons=512, output_neurons=10
            self.fc3 = nn.Linear(512, 10)

            self.forward = self.model_1
            
        elif mode == 'large':

            # First 2D convolutional layer, taking in 3 input channel (image),
            # outputting 32 convolutional features, with a square kernel size of 3, stride 1 and padding 1     
            self.conv1 =  nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding = 1)

            # Second 2D convolutional layer, taking in 32 input channel (image),
            # outputting 64 convolutional features, with a square kernel size of 3, stride 1 and padding 1 
            self.conv2 =  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding = 1)

            # Third 2D convolutional layer, taking in 64 input channel (image),
            # outputting 128 convolutional features, with a square kernel size of 3, stride 1 and padding 1 
            self.conv3 =  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1)

            # Fourth 2D convolutional layer, taking in 128 input channel (image),
            # outputting 256 convolutional features, with a square kernel size of 3, stride 1 and padding 1 
            self.conv4 =  nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding = 1)
            
            # Fifth 2D convolutional layer, taking in 256 input channel (image),
            # outputting 256 convolutional features, with a square kernel size of 3, stride 1 and padding 1 
            self.conv5 =  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding = 1)

            #Average pooling
            self.avgpool = nn.AdaptiveAveragePool(4,4)
            
            #Fully Connected 1st layer with input_neurons = 4096, output_neurons=2048
            self.fc1 = nn.Linear(256*4*4, 2048)
            
            #Dropout1 of rate 0.5
            self.dropout1 = nn.Dropout2d(0.5)
            
            #Fully Connected 2nd layer with input_neurons=2048, output_neurons=512
            self.fc2 = nn.Linear(2048, 512)
            
            #Dropout2 of rate 0.5
            self.dropout2 = nn.Dropout2d(0.5)

            #Fully Connected final layer with both input_neurons=512, output_neurons=10
            self.fc3 = nn.Linear(512, 10)
  
            self.forward = self.model_2
            
        else: 
        
            print("Invalid mode ", mode, "selected. Select between fc and conv")
            exit(0)
            
            
    # mode = 'small'
    def model_1(self, X):
        
        x = self.conv1(X)    # 128 x 128 x 32
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # 64 x 64 x 32

        x = self.conv2(x)    # 64 x 64 x 64
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # 32 x 32 x 64

        x = self.conv3(x)    # 32 x 32 x 128
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # 16 x 16 x 128
        
        x = self.avgpool(x)  # 4 x 4 x 128

        x = torch.flatten(x, start_dim = 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        x = torch.sigmoid(x);
        output = x;
        return output;
        
    # mode = 'large'
    def model_2(self, X):
        
        x = self.conv1(X)    # 128 x 128 x 32
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # 64 x 64 x 32

        x = self.conv2(x)    # 64 x 64 x 64
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # 32 x 32 x 64

        x = self.conv3(x)    # 32 x 32 x 128
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # 16 x 16 x 128

        x = self.conv4(x)    # 16 x 16 x 256
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # 8 x 8 x 256
        
        x = self.conv5(x)    # 8 x 8 x 256
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # 4 x 4 x 256
        
        x = self.avgpool(x)  # 4 x 4 x 256
        x = torch.flatten(x, start_dim = 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        x = torch.sigmoid(x);
        output = x;
        return output;

