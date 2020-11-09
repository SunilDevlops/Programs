import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, mode):
        super(Autoencoder, self).__init__()
        print(mode)
        self.mode = mode
        if mode == 'fc':
        
            #Input size is 28 X 28 and 1 is the depth of the given image = 1 * 28 * 28
            #Fully Connected Encoder 1st layer with neurons 256
            self.fcEncoder1 = nn.Linear(784, 256);
            
            #Fully Connected Encoder 2nd layer with input_neurons=256, output_neurons=128
            self.fcEncoder2 = nn.Linear(256, 128);
            
            #Fully Connected Decoder 1st layer with input_neurons=128, output_neurons=256
            self.fcDecoder1 = nn.Linear(128, 256);
            
            #Fully Connected Decoder 2nd layer with both input_neurons=256, output_neurons=784
            self.fcDecoder2 = nn.Linear(256, 784);
     
            self.forward = self.model_1
            
        elif mode == 'conv':
        
            # First 2D convolutional Encoder 1st layer, taking in 1 input channel (image),
            # outputting 32 convolutional features, with a square kernel size of 3 and padding 1       
            self.convEncoder1 =  nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding = 1)

            # First 2D convolutional Encoder 2nd layer, taking in 32 input channel (image),
            # outputting 64 convolutional features, with a square kernel size of 3 and padding 1 
            self.convEncoder2 =  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding = 1)

            # First 2D convolutional Decoder 1st layer, taking in 64 input channel (image),
            # outputting 64 convolutional features, with a square kernel size of 3 and padding 1 
            self.convDecoder1 =  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding = 1)

            # First 2D convolutional Decoder 2nd layer, taking in 64 input channel (image),
            # outputting 32 convolutional features, with a square kernel size of 3 and padding 1 
            self.convDecoder2 =  nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding = 1)

            # First 2D convolutional Decoder 3rd layer, taking in 32 input channel (image),
            # outputting 1 convolutional features, with a square kernel size of 3 and padding 1 
            self.convDecoder3 =  nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding = 1)
        
            # Upsampling layer
            self.upsample =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
         
            self.forward = self.model_2
            
        else: 
        
            print("Invalid mode ", mode, "selected. Select between fc and conv")
            exit(0)
        
        
    # Autoencoder using FC
    def model_1(self, X):
        # =========================================================================
        # Two Encoder fully connnected layer and Two Decoder fully connected layer
        # =========================================================================
      
        x = torch.flatten(X, start_dim = 1)

        x = self.fcEncoder1(x)
        x = F.relu(x)
        x = self.fcEncoder2(x)
        x = F.relu(x)
        x = self.fcDecoder1(x)
        x = F.relu(x)
        x = self.fcDecoder2(x)
        x = torch.sigmoid(x);
        output = x;
        return output;
        
    # Autoencoder using CONV
    def model_2(self, X):
        # ============================================================================================
        # Two Encoder fully connnected layer + Three Decoder fully connected layer + Upsampling layer
        # ============================================================================================
      
        x = self.convEncoder1(X)    # 28 x 28 x 32
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # 14 x 14 x 32
        x = self.convEncoder2(x)    # 14 x 14 x 64
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # 7 x 7 x 64
        x = self.convDecoder1(x)    # 7 x 7 x 64
        x = F.relu(x)
        x = self.upsample(x)           # 14 x 14 x 64
        x = self.convDecoder2(x)    # 14 x 14 x 32
        x = F.relu(x)
        x = self.upsample(x)           # 28 x 28 x 32
        x = self.convDecoder3(x)    # 28 x 28 x 1
        x = torch.sigmoid(x);
        x = torch.flatten(x, start_dim = 1)
        output = x;
        return output;
    
    
    # Get the number of parameters in the encoder and the decoder
    def get_no_of_parameters(self, mode):
        if mode == 'fc':
            encoder_params = 0.0
            decoder_params = 0.0
            for name, p in self.named_parameters():         #Returns an iterator over module parameters, yielding both the name of the parameter(name) as well as the parameter(p)
                if name.find("fcDecoder") != -1:
                    decoder_params += p.numel() if p.requires_grad == True else 0
                else:
                    encoder_params += p.numel() if p.requires_grad == True else 0
        
            return(encoder_params, decoder_params)
        elif mode == 'conv':
            encoder_params = 0.0
            decoder_params = 0.0
            for name, p in self.named_parameters():         #Returns an iterator over module parameters, yielding both the name of the parameter(name) as well as the parameter(p)
                if name.find('convDecode') != -1:
                    decoder_params += p.numel() if p.requires_grad == True else 0
                else:
                    encoder_params += p.numel() if p.requires_grad == True else 0
        
            return(encoder_params, decoder_params)
