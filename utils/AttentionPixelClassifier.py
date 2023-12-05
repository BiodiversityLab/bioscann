from utils.AttentionBlock import *
import torch

from torch.nn import MaxPool2d

from torch.nn import Sigmoid

class AttentionPixelClassifierLiteDeep(nn.Module):
    def __init__(self, input_numChannels, output_numChannels):
        super(AttentionPixelClassifierLiteDeep, self).__init__()

        depth1 = 4
        depth2 = 8
        depth3 = 8
        depth4 = 8
        depth5 = 8
        
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(input_numChannels, depth1)
        self.Conv2 = ConvBlock(depth1, depth2)
        self.Conv3 = ConvBlock(depth2, depth3)
        self.Conv4 = ConvBlock(depth3, depth4)
        self.Conv5 = ConvBlock(depth4, depth5)

        self.Up5 = UpConv(int(depth5), int(depth5))
        self.Att5 = AttentionBlock(F_g=int(depth5), F_l=int(depth4), n_coefficients=int(depth3))
        self.UpConv5 = ConvBlock(int(depth4 + depth5), int(depth4))

        self.Up4 = UpConv(int(depth4), int(depth4))
        self.Att4 = AttentionBlock(F_g=int(depth4), F_l=int(depth3), n_coefficients=int(depth2))
        self.UpConv4 = ConvBlock(int(depth3 + depth4), int(depth3))

        self.Up3 = UpConv(int(depth3), int(depth3))
        self.Att3 = AttentionBlock(F_g=int(depth3), F_l=int(depth2), n_coefficients=int(depth1))
        self.UpConv3 = ConvBlock(int(depth2 + depth3), int(depth2))

        self.Up2 = UpConv(int(depth2), int(depth2))
        self.Att2 = AttentionBlock(F_g=int(depth2), F_l=int(depth1), n_coefficients=int(depth2))
        self.UpConv2 = ConvBlock(int(depth1 + depth2), int(depth3))

        self.Conv = nn.Conv2d(int(depth3), output_numChannels, kernel_size=1, stride=1, padding=0)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv3(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv4(e5)

        d5 = self.Up5(e5)
        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(e4)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(e3)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)
        out = self.sigmoid(out)
        
        return out


class AttentionPixelClassifierLite(nn.Module):
    def __init__(self, input_numChannels, output_numChannels):
        super(AttentionPixelClassifierLite, self).__init__()

        depth1 = 32
        depth2 = 16
        depth3 = 8
        
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(input_numChannels, depth1)
        self.Conv2 = ConvBlock(depth1, depth2)
        self.Conv3 = ConvBlock(depth2, depth3)

        self.Up3 = UpConv(int(depth3), int(depth3))
        self.Att3 = AttentionBlock(F_g=int(depth3), F_l=int(depth2), n_coefficients=int(depth1))
        self.UpConv3 = ConvBlock(int(depth2 + depth3), int(depth2))

        self.Up2 = UpConv(int(depth2), int(depth2))
        self.Att2 = AttentionBlock(F_g=int(depth2), F_l=int(depth1), n_coefficients=int(depth2))
        self.UpConv2 = ConvBlock(int(depth1 + depth2), int(depth3))

        self.Conv = nn.Conv2d(int(depth3), output_numChannels, kernel_size=1, stride=1, padding=0)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        d3 = self.Up3(e3)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)
        out = self.sigmoid(out)
        
        return out


class AttentionPixelClassifierMedium(nn.Module):
    def __init__(self, input_numChannels, output_numChannels):
        super(AttentionPixelClassifierMedium, self).__init__()


        divide_depth = 2

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(input_numChannels, 64//divide_depth)
        self.Conv2 = ConvBlock(64//divide_depth, 128//divide_depth)
        self.Conv3 = ConvBlock(128//divide_depth, 256//divide_depth)
      #  self.Conv4 = ConvBlock(256//divide_depth, 512//divide_depth)
        #self.Conv5 = ConvBlock(512, 1024)

        #self.Up5 = UpConv(1024, 512)
        #self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        #self.UpConv5 = ConvBlock(1024, 512)

     #   self.Up4 = UpConv(int(512//divide_depth), int(256/divide_depth))
     #   self.Att4 = AttentionBlock(F_g=int(256//divide_depth), F_l=int(256//divide_depth), n_coefficients=int(128//divide_depth))
      #  self.UpConv4 = ConvBlock(int(512//divide_depth), int(256//divide_depth))

        self.Up3 = UpConv(int(256//divide_depth), int(128//divide_depth))
        self.Att3 = AttentionBlock(F_g=int(128//divide_depth), F_l=int(128//divide_depth), n_coefficients=int(64//divide_depth))
        self.UpConv3 = ConvBlock(int(256//divide_depth), int(128//divide_depth))

        self.Up2 = UpConv(int(128//divide_depth), int(64//divide_depth))
        self.Att2 = AttentionBlock(F_g=int(64//divide_depth), F_l=int(64//divide_depth), n_coefficients=int(32//divide_depth))
        self.UpConv2 = ConvBlock(int(128//divide_depth), int(64//divide_depth))

        self.Conv = nn.Conv2d(int(64//divide_depth), output_numChannels, kernel_size=1, stride=1, padding=0)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

      #  e4 = self.MaxPool(e3)
       # e4 = self.Conv4(e4)

       # e5 = self.MaxPool(e4)
      #  e5 = self.Conv5(e5)

       # d5 = self.Up5(e5)

      #  s4 = self.Att5(gate=d5, skip_connection=e4)
     #   d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
      #  d5 = self.UpConv5(d5)

        #d4 = self.Up4(d5)
         #d4 = self.Up4(e4)
        #print(d4)
        # s3 = self.Att4(gate=d4, skip_connection=e3)
        # d4 = torch.cat((s3, d4), dim=1)
        # d4 = self.UpConv4(d4)

        

       # d3 = self.Up3(d4)
        d3 = self.Up3(e3)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)
        out = self.sigmoid(out)
        
        return out



class AttentionPixelClassifier(nn.Module):
    def __init__(self, input_numChannels, output_numChannels):
        super(AttentionPixelClassifier, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(input_numChannels, 32)
        self.Conv2 = ConvBlock(32, 64)
        self.Conv3 = ConvBlock(64, 128)
        self.Conv4 = ConvBlock(128, 256)
        #self.Conv5 = ConvBlock(512, 1024)

        #self.Up5 = UpConv(1024, 512)
        #self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        #self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(int(512//2), int(256/2))
        self.Att4 = AttentionBlock(F_g=int(256//2), F_l=int(256//2), n_coefficients=int(128//2))
        self.UpConv4 = ConvBlock(int(512//2), int(256//2))

        self.Up3 = UpConv(int(256//2), int(128//2))
        self.Att3 = AttentionBlock(F_g=int(128//2), F_l=int(128//2), n_coefficients=int(64//2))
        self.UpConv3 = ConvBlock(int(256//2), int(128//2))

        self.Up2 = UpConv(int(128//2), int(64//2))
        self.Att2 = AttentionBlock(F_g=int(64//2), F_l=int(64//2), n_coefficients=int(32//2))
        self.UpConv2 = ConvBlock(int(128//2), int(64//2))

        self.Conv = nn.Conv2d(int(64//2), output_numChannels, kernel_size=1, stride=1, padding=0)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

       # e5 = self.MaxPool(e4)
      #  e5 = self.Conv5(e5)

       # d5 = self.Up5(e5)

      #  s4 = self.Att5(gate=d5, skip_connection=e4)
     #   d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
      #  d5 = self.UpConv5(d5)

        #d4 = self.Up4(d5)
        d4 = self.Up4(e4)
        #print(d4)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)
        out = self.sigmoid(out)
        
        return out


class AttentionPixelClassifierFlex(nn.Module):
    def __init__(self, input_numChannels, output_numChannels, n_channels_per_layer, n_coefficients_per_upsampling_layer=None):
        super(AttentionPixelClassifierFlex, self).__init__()
        self.n_channels_per_layer = np.array(n_channels_per_layer)

        if not len(self.n_channels_per_layer) % 2 == 1:
            print('Warning: Channel counts must be specified for an uneven number of layers. The provided array',self.n_channels_per_layer, 'is uneven, therefore last provided value is being ignored.')
            self.n_channels_per_layer = self.n_channels_per_layer[:-1]
        self.n_layers_half = (len(self.n_channels_per_layer)+1)//2
        if n_coefficients_per_upsampling_layer == None:
            self.n_coefficients_per_upsampling_layer = (np.array(self.n_channels_per_layer[-(self.n_layers_half-1):])/2).astype(int)
        else:
            self.n_coefficients_per_upsampling_layer = n_coefficients_per_upsampling_layer

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = ConvBlock(input_numChannels, self.n_channels_per_layer[0])
        if self.n_layers_half > 1:
            self.Conv2 = ConvBlock(self.n_channels_per_layer[0], self.n_channels_per_layer[1])
        if self.n_layers_half > 2:
            self.Conv3 = ConvBlock(self.n_channels_per_layer[1], self.n_channels_per_layer[2])
        if self.n_layers_half > 3:
            self.Conv4 = ConvBlock(self.n_channels_per_layer[2], self.n_channels_per_layer[3])
        if self.n_layers_half > 4:
            self.Conv5 = ConvBlock(self.n_channels_per_layer[3], self.n_channels_per_layer[4])
        if self.n_layers_half > 5:
            self.Conv6 = ConvBlock(self.n_channels_per_layer[4], self.n_channels_per_layer[5])
        if self.n_layers_half > 6:
            self.Conv7 = ConvBlock(self.n_channels_per_layer[5], self.n_channels_per_layer[6])

        if self.n_layers_half > 6:
            self.Up7 = UpConv(int(self.n_channels_per_layer[6]), int(self.n_channels_per_layer[-6]))
            self.Att7 = AttentionBlock(F_g=int(self.n_channels_per_layer[5]), F_l=int(self.n_channels_per_layer[-6]), n_coefficients=int(self.n_coefficients_per_upsampling_layer[-6]))
            self.UpConv7 = ConvBlock(int(self.n_channels_per_layer[-6]*2), int(self.n_channels_per_layer[-6]))
        if self.n_layers_half > 5:
            self.Up6 = UpConv(int(self.n_channels_per_layer[-6]), int(self.n_channels_per_layer[-5]))
            self.Att6 = AttentionBlock(F_g=int(self.n_channels_per_layer[4]), F_l=int(self.n_channels_per_layer[-5]), n_coefficients=int(self.n_coefficients_per_upsampling_layer[-5]))
            self.UpConv6 = ConvBlock(int(self.n_channels_per_layer[-5]*2), int(self.n_channels_per_layer[-5]))
        if self.n_layers_half > 4:
            self.Up5 = UpConv(int(self.n_channels_per_layer[-5]), int(self.n_channels_per_layer[-4]))
            self.Att5 = AttentionBlock(F_g=int(self.n_channels_per_layer[3]), F_l=int(self.n_channels_per_layer[-4]), n_coefficients=int(self.n_coefficients_per_upsampling_layer[-4]))
            self.UpConv5 = ConvBlock(int(self.n_channels_per_layer[-4]*2), int(self.n_channels_per_layer[-4]))
        if self.n_layers_half > 3:
            self.Up4 = UpConv(int(self.n_channels_per_layer[-4]), int(self.n_channels_per_layer[-3]))
            self.Att4 = AttentionBlock(F_g=int(self.n_channels_per_layer[2]), F_l=int(self.n_channels_per_layer[-3]), n_coefficients=int(self.n_coefficients_per_upsampling_layer[-3]))
            self.UpConv4 = ConvBlock(int(self.n_channels_per_layer[-3]*2), int(self.n_channels_per_layer[-3]))
        if self.n_layers_half > 2:
            self.Up3 = UpConv(int(self.n_channels_per_layer[-3]), int(self.n_channels_per_layer[-2]))
            self.Att3 = AttentionBlock(F_g=int(self.n_channels_per_layer[1]), F_l=int(self.n_channels_per_layer[-2]), n_coefficients=int(self.n_coefficients_per_upsampling_layer[-2]))
            self.UpConv3 = ConvBlock(int(self.n_channels_per_layer[-2]*2), int(self.n_channels_per_layer[-2]))
        if self.n_layers_half > 1:
            self.Up2 = UpConv(int(self.n_channels_per_layer[-2]), int(self.n_channels_per_layer[-1]))
            self.Att2 = AttentionBlock(F_g=int(self.n_channels_per_layer[0]), F_l=int(self.n_channels_per_layer[-1]), n_coefficients=int(self.n_coefficients_per_upsampling_layer[-1]))
            self.UpConv2 = ConvBlock(int(self.n_channels_per_layer[-1]*2), int(self.n_channels_per_layer[-1]))

        self.Conv = nn.Conv2d(int(self.n_channels_per_layer[-1]), output_numChannels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        output_layers=[]
        e1 = self.Conv1(x)
        output_layers.append(e1)
        if self.n_layers_half > 1:
            e2 = self.MaxPool(e1)
            e2 = self.Conv2(e2)
            output_layers.append(e2)
        if self.n_layers_half > 2:
            e3 = self.MaxPool(e2)
            e3 = self.Conv3(e3)
            output_layers.append(e3)
        if self.n_layers_half > 3:
            e4 = self.MaxPool(e3)
            e4 = self.Conv4(e4)
            output_layers.append(e4)
        if self.n_layers_half > 4:
            e5 = self.MaxPool(e4)
            e5 = self.Conv5(e5)
            output_layers.append(e5)
        if self.n_layers_half > 5:
            e6 = self.MaxPool(e5)
            e6 = self.Conv6(e6)
            output_layers.append(e6)
        if self.n_layers_half > 6:
            e7 = self.MaxPool(e6)
            e7 = self.Conv7(e7)
            output_layers.append(e7)
        # going up again
        if self.n_layers_half > 6:
            d7 = self.Up7(output_layers[-1])
            s6 = self.Att7(gate=e6, skip_connection=d7)
            d7 = torch.cat((s6, d7), dim=1)
            d7 = self.UpConv7(d7)
            output_layers.append(d7)
        if self.n_layers_half > 5:
            d6 = self.Up6(output_layers[-1])
            s5 = self.Att6(gate=e5, skip_connection=d6)
            d6 = torch.cat((s5, d6), dim=1)
            d6 = self.UpConv6(d6)
            output_layers.append(d6)
        if self.n_layers_half > 4:
            d5 = self.Up5(output_layers[-1])
            s4 = self.Att5(gate=e4, skip_connection=d5)
            d5 = torch.cat((s4, d5), dim=1)
            d5 = self.UpConv5(d5)
            output_layers.append(d5)
        if self.n_layers_half > 3:
            d4 = self.Up4(output_layers[-1])
            s3 = self.Att4(gate=e3, skip_connection=d4)
            d4 = torch.cat((s3, d4), dim=1)
            d4 = self.UpConv4(d4)
            output_layers.append(d4)
        if self.n_layers_half > 2:
            d3 = self.Up3(output_layers[-1])
            s2 = self.Att3(gate=e2, skip_connection=d3)
            d3 = torch.cat((s2, d3), dim=1)
            d3 = self.UpConv3(d3)
            output_layers.append(d3)
        if self.n_layers_half > 1:
            d2 = self.Up2(output_layers[-1])
            s1 = self.Att2(gate=e1, skip_connection=d2)
            d2 = torch.cat((s1, d2), dim=1)
            d2 = self.UpConv2(d2)
            output_layers.append(d2)
        out = self.Conv(output_layers[-1])
        out = self.sigmoid(out)
        return out



