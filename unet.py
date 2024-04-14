import torch 
import torch.nn as nn
import math
from torch.nn import functional as F 

# code reference: https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        conv_size = 5
        stride = 2
        padding = int((conv_size-1)/2)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, conv_size, stride=stride, padding=padding)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, stride, padding)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, conv_size, stride=stride, padding=padding)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, stride, padding)
        self.conv2 = nn.Conv2d(out_ch, out_ch, conv_size, stride=stride, padding=padding)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))

        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        conv_size = 3
        stride = 1
        padding = int((conv_size-1)/2)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, conv_size, padding=padding)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, conv_size, padding=padding)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, padding)
        self.conv2 = nn.Conv2d(out_ch, out_ch, conv_size, padding=padding)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)
     
class UNet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, down_channels=[2, 4, 8, 4, 2], img_size=(100, 160)):
        super().__init__()
        image_channels = 1
        down_channels = down_channels
        up_channels = down_channels[::-1]
        out_dim = 1
        time_emb_dim = 32
        self.img_size = img_size

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 5, padding=2)

        down_list = [Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)]

        # Downsample
        self.downs = nn.ModuleList(down_list)
        
        # generate the test samples to determine whether 
        device = next(self.parameters()).device
        test_data = torch.rand([1, 1, img_size[0], img_size[1]]).to(device)
        x = self.conv0(test_data)
        residual_inputs = []
        for down in self.downs:
            x = down(x)
            residual_inputs.append(x)
        
        self.embd_size = x.shape[-1] * x.shape[-2]

        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])
        # self.var_layer = Block(up_channels[len(up_channels)-2], up_channels[len(up_channels)-1], time_emb_dim, up=True)

        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def encode(self, x):
        # Initial conv
        x = self.conv0(x)
        # Unet
        self.residual_inputs = []
        for down in self.downs:
            x = down(x)
            self.residual_inputs.append(x)

        return x
    
    

    def decode(self, x):
        for up in self.ups:
            residual_x = self.residual_inputs.pop()

            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x)

        return self.output(x)
    
    def forward(self, x):
        enc = self.encode(x)
        return self.decode(enc)
    
class SSM5311(nn.Module):
    def __init__(self, img_size=(80, 80)):
        super().__init__()

        self.unet = UNet(img_size=img_size)


        test_data = torch.randn(1, 1, img_size[0], img_size[1])
        x = self.unet.encode(test_data)


        self.z_dim = math.prod(x.shape)

        # self.last_conv_down = nn.Conv2d(x.shape[1], 2, kernel_size=1)
        # self.last_conv_up = nn.Conv2d(2, x.shape[1], kernel_size=1)

        self.rnn = nn.GRU(self.z_dim, self.z_dim*2, batch_first=True)
        self.h0 = nn.Parameter(torch.rand(1, self.z_dim*2), requires_grad=True)

    def forward(self, x):
        B, T, H, W = x.shape
        x = x.reshape(B*T, H, W).unsqueeze(1) # [B*T, 1, H, W]

        # get embedding
        embd = self.unet.encode(x)
        # embd = self.last_conv_down(embd)
        _, Cemb, Hemb, Wemb = embd.shape

        # process embd with rnn
        embd = embd.reshape(B*T, Cemb*Hemb*Wemb).reshape(B, T, -1) # [B, T, N]
        N = embd.shape[-1]

        rnn_out, _ = self.rnn(embd, self.h0.repeat(repeats=(1, embd.shape[0], 1))) # [B, T, 2*N]
        mu, logvar = rnn_out[..., :N], rnn_out[..., N:]

        if self.training: # apply the reparameterization trick
            eps = torch.randn_like(mu)
            std_dev = torch.exp(0.5 * logvar)
            embd_new = mu + std_dev * eps
        else:
            embd_new = mu

        embd_new = embd_new.reshape(B*T, Cemb, Hemb, Wemb)
        # embd_new = self.last_conv_up(embd_new)


        dec =  self.unet.decode(embd_new)
 
        return dec.reshape(B, T, H, W)

if __name__ == "__main__":
    from utils import * 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SSM5311().to(device)
    print(f"UNetnumber: {sum(p.numel() for p in model.unet.parameters() if p.requires_grad)}")
    print(f"Total param number: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # exit()
    sample_data = torch.randn(128, 10, 101, 161).to(device)
    sample_data = reshape_to_square(sample_data)
    model.train()
    print("a")
    a = model(sample_data)
    print("b")
    a =reshape_back(a)
    print(a.shape)
    # print(model(sample_data).shape)