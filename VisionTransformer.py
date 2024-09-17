BATCH_SIZE = 64
IMG_SIZE = (128,128)
PATCH_SIZE = (8,8)
# Number of patch in one sequence
NUM_PATCH = int((IMG_SIZE[0] / PATCH_SIZE[0])**2)
# Number of patch out after feature extracting
T = int(NUM_PATCH / 4)
# Number of attention head
NUM_H = 4
# Number of Transformer Block
NUM_TR = 11

# For this notebook to run with updated APIs, we need torch 1.12+ and torchvision 0.13+

import torch
import torchvision
assert int(torch.__version__.split(".")[1]) >= 12 or int(torch.__version__.split(".")[0]) == 2, "torch version should be 1.12+"
assert int(torchvision.__version__.split(".")[1]) >= 13, "torchvision version should be 0.13+"
print(f"torch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")



# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms



device = "cuda" if torch.cuda.is_available() else "cpu"


my_transform = transforms.Compose([
                                    transforms.Resize((IMG_SIZE)),
                                    transforms.ToTensor()
])

def img_to_patch(x, patch_size, flatten_channels=True):
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


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class VisionTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim,n_feature, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        # self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        # self.input_layer = FeatureExtractor(n_feature,embed_dim)
        self.input_layer = nn.Sequential(
            nn.Conv3d(3 , n_feature*3 , kernel_size = 3 , padding=1 , stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,1,1) , stride=(2,1,1) , padding=(0,0,0)),

            nn.Conv3d(n_feature*3 , n_feature*6 , 3 , padding=1 , stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,1,1) , stride=(2,1,1) , padding=(0,0,0)),
            nn.Conv3d(n_feature*6 , n_feature*8 , 3 , padding=1 , stride=1),

            Transpose(2,1),
            nn.Flatten(2,-1),
            nn.Linear(n_feature*8*PATCH_SIZE[0]*PATCH_SIZE[0] ,embed_dim )


        )
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
            # torch.nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+int(NUM_PATCH / 4),embed_dim))


    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size , flatten_channels=False).transpose(1,2)
        # print(x.shape)
        B, _, _ , _ , _= x.shape
        T = int(NUM_PATCH / 4)
        x = self.input_layer(x)
        # print(x.shape)
        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out