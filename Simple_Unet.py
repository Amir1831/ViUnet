import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import Config
class MultiUNetModel(nn.Module):
    def __init__(self, n_classes=3, img_height=Config.IMG_SIZE[0], img_width=Config.IMG_SIZE[1], img_channels=1):
        super(MultiUNetModel, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        
        # Contraction path
        self.conv1 = self.conv_block(img_channels, 16)
        self.dropout1 = nn.Dropout(0.1)
        self.conv1_2 = self.conv_block(16, 16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = self.conv_block(16, 32)
        self.dropout2 = nn.Dropout(0.1)
        self.conv2_2 = self.conv_block(32, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = self.conv_block(32, 64)
        self.dropout3 = nn.Dropout(0.2)
        self.conv3_2 = self.conv_block(64, 64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = self.conv_block(64, 128)
        self.dropout4 = nn.Dropout(0.2)
        self.conv4_2 = self.conv_block(128, 128)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = self.conv_block(128, 256)
        self.dropout5 = nn.Dropout(0.3)
        self.conv5_2 = self.conv_block(256, 256)
        
        # Expansive path
        self.trans_conv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv6 = self.conv_block(256, 128)
        self.dropout6 = nn.Dropout(0.2)
        self.conv6_2 = self.conv_block(128, 128)
        
        self.trans_conv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv7 = self.conv_block(128, 64)
        self.dropout7 = nn.Dropout(0.2)
        self.conv7_2 = self.conv_block(64, 64)
        
        self.trans_conv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv8 = self.conv_block(64, 32)
        self.dropout8 = nn.Dropout(0.1)
        self.conv8_2 = self.conv_block(32, 32)
        
        self.trans_conv9 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv9 = self.conv_block(32, 16)
        self.dropout9 = nn.Dropout(0.1)
        self.conv9_2 = self.conv_block(16, 16)
        
        self.final_conv = nn.Conv2d(16, n_classes, 1)
        
    def forward(self, x):
        # Contraction path
        ## INPUT : [B , 1 , H , W]
        conv1 = self.conv1(x)  ## OUT  [B , 16 , H , W]
        conv1 = self.dropout1(conv1)
        conv1 = self.conv1_2(conv1)  ## OUT  [B , 16 , H , W]
        pool1 = self.pool1(conv1)    ## OUT  [B , 16 , H / 2 , W / 2]
        
        conv2 = self.conv2(pool1)    ## OUT  [B , 32 , H  / 2, W / 2]
        conv2 = self.dropout2(conv2)
        conv2 = self.conv2_2(conv2)  ## OUT  [B , 32 , H  / 2, W / 2]
        pool2 = self.pool2(conv2)    ## OUT  [B , 32 , H  / 4, W / 4]
        
        conv3 = self.conv3(pool2)    ## OUT  [B , 64 , H  / 4 , W / 4]
        conv3 = self.dropout3(conv3)
        conv3 = self.conv3_2(conv3)  ## OUT  [B , 64 , H  / 4 , W / 4]
        pool3 = self.pool3(conv3)    ## OUT  [B , 64 , H  / 8 , W / 8]
        
        conv4 = self.conv4(pool3)    ## OUT  [B , 128 , H  / 8 , W / 8]
        conv4 = self.dropout4(conv4)
        conv4 = self.conv4_2(conv4)  ## OUT  [B , 128 , H  / 8 , W / 8]
        pool4 = self.pool4(conv4)    ## OUT  [B , 128 , H  / 16 , W / 16]
         
        conv5 = self.conv5(pool4)    ## OUT  [B , 256 , H  / 16 , W / 16]
        conv5 = self.dropout5(conv5)
        conv5 = self.conv5_2(conv5)  ## OUT  [B , 256 , H  / 16 , W / 16]
        
        # Expansive path
        trans_conv6 = self.trans_conv6(conv5)   ## OUT  [B , 128 , H  / 8 , W / 8]
        concat6 = torch.cat([trans_conv6, conv4], dim=1)  ## OUT  [B , 256 , H  / 8 , W / 8]
        conv6 = self.conv6(concat6)             ## OUT  [B , 128 , H  / 8 , W / 8]
        conv6 = self.dropout6(conv6)
        conv6 = self.conv6_2(conv6)             ## OUT  [B , 128 , H  / 8 , W / 8]
        
        trans_conv7 = self.trans_conv7(conv6)   ## OUT  [B , 64 , H  / 4 , W / 4]
        concat7 = torch.cat([trans_conv7, conv3], dim=1)  ## OUT  [B , 128 , H  / 4 , W / 4]
        conv7 = self.conv7(concat7)             ## OUT  [B , 64 , H  / 4 , W / 4]
        conv7 = self.dropout7(conv7)
        conv7 = self.conv7_2(conv7)             ## OUT  [B , 64 , H  / 4 , W / 4]
         
        trans_conv8 = self.trans_conv8(conv7)  ## OUT  [B , 32 , H  / 2 , W / 2]
        concat8 = torch.cat([trans_conv8, conv2], dim=1)  ## OUT  [B , 64 , H  / 2 , W / 2]
        conv8 = self.conv8(concat8)            ## OUT  [B , 32 , H  / 2 , W / 2]
        conv8 = self.dropout8(conv8)
        conv8 = self.conv8_2(conv8)            ## OUT  [B , 32 , H  / 2 , W / 2]
        
        trans_conv9 = self.trans_conv9(conv8)  ## OUT  [B , 16 , H  , W ]
        concat9 = torch.cat([trans_conv9, conv1], dim=1) ## OUT  [B , 32 , H   , W ]
        conv9 = self.conv9(concat9)            ## OUT  [B , 16 , H   , W ]
        conv9 = self.dropout9(conv9)
        conv9 = self.conv9_2(conv9)           ## OUT  [B , 16 , H  , W ]
        
        output = self.final_conv(conv9)       ## OUT  [B , num_class , H  , W ]
        output = F.softmax(output, dim=1)
        # print(output.shape)
        return output
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )