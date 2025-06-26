import math
import torch
from torch import nn

class OverLapPatchEmbed(nn.Module):
    """
    Overlap Patch Embedding
    Args:
        img_size: size of the image (default: 224)
        patch_size: size of the patch (default: 16)
        in_chans: number of channels of the input (default: 3)
        embed_dim: dimension of the embedding (default: 768)
        norm_layer: normalization layer (default: nn.LayerNorm)
        stride: stride of the convolution (default: 4)
        padding: padding of the convolution (default: 2)
    """
    def __init__(self, in_chans=3, embed_dim=768, patch_size=16, norm_layer=nn.LayerNorm, stride=4, padding=2):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x):
        x = self.proj(x)
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        return x, H, W


class EfficientMultiHeadAttention(nn.Module):
    """
    Efficient Multi-Head Attention
    Args:
        dim: dimension of the input 
        heads: number of heads
        sr_ratio: ratio of spatial reduction (default: 1)
        norm_layer: normalization layer (default: nn.LayerNorm)
    """
    def __init__(self,dim,heads,sr_ratio=1,norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.sr_ratio = sr_ratio
        
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_sr = norm_layer(dim) if norm_layer else nn.Identity()


    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.sr_ratio > 1:
            kv = x.transpose(1,2).reshape(B,self.dim,H,W)
            kv = self.sr(kv)
            kv = kv.reshape(B,self.dim,-1).transpose(1,2)
            kv = self.norm_sr(kv)
        else:
            kv = x
        
        out, _ = self.attn(x, kv, kv,need_weights=False)

        return out

    
class MixMLP(nn.Module):
    """
    MixMLP
    Args:
        dim: dimension of the input
        hidden: dimension of the hidden layer
        dwconv: whether to use depthwise convolution (default: True)
    """
    def __init__(self,dim,hidden,dwconv=True):
        super().__init__()
        self.fc1 = nn.Linear(dim,hidden)
        self.dwconv = nn.Conv2d(hidden,hidden,kernel_size=3,padding=1,groups=hidden) if dwconv else nn.Identity()
        self.fc2 = nn.Linear(hidden,dim)
        self.act = nn.GELU()

    def forward(self,x,H,W):
        x = self.fc1(x)
        B,N,C = x.shape
        x = x.transpose(1,2).reshape(B,C,H,W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1,2)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer Block
    Args:
        dim: dimension of the input
        num_heads: number of heads
        mlp_ratio: ratio of the MLP hidden dimension
        sr_ratio: ratio of spatial reduction (default: 1)
    """
    def __init__(self,dim,num_heads, mlp_ratio ,sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientMultiHeadAttention(dim,num_heads,sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixMLP(dim,int(dim * mlp_ratio))
        
    def forward(self,x,H,W):
        x = x + self.attn(self.norm1(x),H,W)
        x = x + self.mlp(self.norm2(x),H,W)
        return x

CONF = {  # C1..C4, heads, blocks, R
 'B0': dict(C=[32,64,160,256],  L=[2,2,2,2], H=[1,2,5,8], R=[8,4,2,1]),
 'B1': dict(C=[64,128,320,512], L=[2,2,2,2], H=[1,2,5,8], R=[8,4,2,1]),
 'B2': dict(C=[64,128,320,512], L=[3,4,6,3], H=[1,2,5,8], R=[8,4,2,1]),
 'B3': dict(C=[64,128,320,512], L=[3,4,18,3],H=[1,2,5,8], R=[8,4,2,1]),
 'B4': dict(C=[64,128,320,512], L=[3,8,27,3],H=[1,2,5,8], R=[8,4,2,1]),
 'B5': dict(C=[64,128,320,512], L=[3,6,40,3],H=[1,2,5,8], R=[8,4,2,1]),
}

class MixVisionTransformer(nn.Module):
    """
    MixVisionTransformer
    Args:
        variant: variant of the model (default: 'B0')
    """
    def __init__(self, variant='B0'):
        super().__init__()
        cfg = CONF[variant]
        self.stages, in_ch = nn.ModuleList(), 3
        for i in range(4):
            patch = OverLapPatchEmbed(in_ch, cfg["C"][i], patch_size= 7 if i==0 else 3,
                                      stride=4 if i==0 else 2, padding=3 if i==0 else 1)
            blocks = nn.ModuleList([TransformerBlock(cfg["C"][i],
                                                     cfg["H"][i],
                                                     mlp_ratio=4,
                                                     sr_ratio=cfg["R"][i])
                                    for _ in range(cfg["L"][i])])
            self.stages.append(nn.ModuleDict(dict(patch=patch, blocks=blocks)))
            in_ch = cfg["C"][i]

    def forward(self, x):
        outs = []
        for stage in self.stages:
            x,H,W = getattr(stage, 'patch')(x)            # (B,N,C)
            for blk in getattr(stage, 'blocks'):
                x = blk(x,H,W)
            outs.append((x,H,W))
            if len(x.shape)==3:                  # restore for next stage
                x = x.transpose(1,2).reshape(x.size(0),-1,H,W)
        return outs 

if __name__ == '__main__':
    model = MixVisionTransformer(variant='B0')
    x = torch.randn(1,3,224,224)
    outs = model(x)
    print(outs[0][0].shape)


    