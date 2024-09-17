import torch
from torch import nn
from Config import Config
class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )



    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x

def img_to_patch(x, patch_size, flatten_channels=False):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x
    
class DownBlock(nn.Module):
    def __init__(self , input_channel , out_channel ):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channel , out_channel , kernel_size = 3 , padding=1 , stride=1)
        self.ac1 = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channel , out_channel , kernel_size = 3 , padding=1 , stride=1)
        self.ac2 = nn.ReLU()
        self.conv3 = nn.Conv3d(out_channel , out_channel , kernel_size = 3 , padding=1 , stride=1)
        self.ac3 = nn.ReLU()
        self.BatchNorm = nn.BatchNorm3d(out_channel)
        self.Maxpool = nn.MaxPool3d(kernel_size=(2,2,2) , stride=(2,2,2) , padding=(0,0,0))
    def forward(self,x):
        ## INPUT : (B , C , T , P , P)
        x1 = self.ac1(self.conv1(x))
        x2 = self.ac2(self.conv2(self.BatchNorm(x1)))
        x2 = self.ac3(self.conv3(x2))
        ## OUTPUT : (B , C' , T /2 , P /2 , P/2)
        return self.Maxpool(torch.add(x1 , x2))
    
class  UPBlock(nn.Module):
    def __init__(self , input_channel , out_channel ):
        super().__init__()
        self.convT = nn.ConvTranspose3d(input_channel,out_channel,kernel_size=(2,2,2),stride=(2,2,2))
        self.conv1 = nn.Conv3d(out_channel , out_channel , kernel_size = 3 , padding=1 , stride=1)
        self.ac1 = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channel , out_channel , kernel_size = 3 , padding=1 , stride=1)
        self.ac2 = nn.ReLU()
        self.conv3 = nn.Conv3d(out_channel , out_channel , kernel_size = 3 , padding=1 , stride=1)
        self.ac3 = nn.ReLU()
        self.BatchNorm = nn.BatchNorm3d(out_channel)
        
    def forward(self,x):
        ## INPUT : (B , C , T , P , P)
        x1 = self.convT(x)
        x2 = self.ac1(self.conv1(x1))
        x2 = self.ac2(self.conv2(self.BatchNorm(x2)))
        x2 = self.ac3(self.conv3(x2))
        ## OUTPUT : (B , C' , T * 2 , P * 2 , P * 2)
        return torch.add(x1 , x2)
        


class ViUnet(nn.Module):
    def __init__(self , input_channel = 1):
        super().__init__()
        ## List for skip conections
        self.skip = nn.ParameterList()
        ## Encoder
        self.down_block1 = DownBlock(input_channel = Config.out0 , out_channel = Config.out1)
        self.down_block2 = DownBlock(input_channel = Config.out1 , out_channel = Config.out2)
        self.down_block3 = DownBlock(input_channel = Config.out2 , out_channel = Config.out2)
        ## Neck
        self.neck = nn.Sequential(*[AttentionBlock(int(Config.out2 * ((Config.PATCH_SIZE[0]/16)**2)), 128, 2, dropout=0.1) for _ in range(Config.NUM_TR)])
        self.flat = nn.Flatten(2,-1)
        ## Decoder
        self.up_block1 = UPBlock(input_channel = Config.out2 * 2 , out_channel = Config.out2)
        self.up_block2 = UPBlock(input_channel = Config.out1 + Config.out2 , out_channel = Config.out0)
        ## Input Convolution
        self.in_conv = nn.Conv2d(in_channels=1,out_channels=Config.out0,kernel_size=3,padding=1,stride=1)
        ## Out Convolution
        self.out_conv = nn.Conv2d(in_channels=Config.out0,out_channels=2,kernel_size=3,padding=1,stride=1)
        self.sig = nn.Sigmoid()
        self.Laten_norm = 0
    def forward(self,x):
        B , C , H , W = x.shape
        ## Input (B , C , H , W)
        x = self.in_conv(x)    ## OUT : [B , out0 , H , W]
        x = img_to_patch(x,Config.PATCH_SIZE[0],flatten_channels=False).transpose(1,2)   ## OUT : [B, out0 , T, P, P]
        ## Encoder 
        x = self.down_block1(x)  ## OUT : [B , out1 , T /2 , P/2 , P/2]
        self.skip.append(x)
        x = self.down_block2(x)  ## OUT : [B , out2 , T /4 , P/4 , P/4]
        self.skip.append(x)
        x = self.down_block3(x)  ## OUT : [B , out2 , T /8 , P/8 , P/8]
        self.skip.append(x)
        x = self.down_block3(x)  ## OUT : [B , out2 , T /16 , P/16 , P/16]
        self.skip.append(x)
        ## Neck
        x = self.flat(x.transpose(1,2))
        ## IN : [B , T/16 , out2 * P/16 * P/16]
        x = self.neck(x)
        self.Laten = torch.norm(x)
        x = x.reshape(x.size(0) ,x.size(1) , Config.out2 , Config.PATCH_SIZE[0] // 16 , Config.PATCH_SIZE[0] // 16)
        x = x.transpose(1,2)
        ## OUT : [B , out2,  T/16 , P/16 , P/16]

        ## Decoder
        x = torch.cat([x , self.skip[3]],dim = 1)  ## OUT : [B , out2*2 , T/16 , P/16 , P/16]
        x = self.up_block1(x)                 ## OUT : [B , out2 , T/8 , P/8 , P/8]

        x = torch.cat([x , self.skip[2]],dim = 1)  ## OUT : [B , out2*2 , T/8 , P/8 , P/8]
        x = self.up_block1(x)                 ## OUT : [B , out2 , T/4, P/4 , P/4]

        x = torch.cat([x , self.skip[1]],dim = 1)  ## OUT : [B , out2*2 , T/4 , P/4 , P/4]
        x = self.up_block1(x)                 ## OUT : [B , out2 , T/2, P/2 , P/2]

        x = torch.cat([x , self.skip[0]],dim = 1)  ## OUT : [B , out1*out2 , T/2 , P/2 , P/2]
        x = self.up_block2(x)                 ## OUT : [B , out0 , T , P , P]
        x = x.reshape(B , Config.out0 , H , W)
        return self.sig(self.out_conv(x))  ## OUT  : [B , 1 , H , W]
