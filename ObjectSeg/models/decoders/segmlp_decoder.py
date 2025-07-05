import torch
import torch.nn as nn
from torchvision.transforms.functional import resize


class MLPDecoder(nn.Module):
    def __init__(self, dims, embed_dim=256, num_classes=12):
        super().__init__()
        self.proj   = nn.ModuleList([nn.Linear(d, embed_dim) for d in dims])
        self.fuse   = nn.Linear(embed_dim*4, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, xs):
        feats = []
        target_size = xs[0][1:] 
        for (x, H, W), proj in zip(xs, self.proj):
            x_proj = proj(x)
            x_proj = x_proj.transpose(1,2).reshape(x.size(0),-1,H,W)  # (B, embed_dim, H, W)
            x_proj = resize(x_proj, size=target_size, antialias=True)
            x_proj = x_proj.flatten(2).transpose(1, 2)  # (B, HW, embed_dim)
            feats.append(x_proj)
        x = torch.cat(feats, dim=2)  # (B, HW, 4*embed_dim)
        x = self.fuse(x)
        B, N, C = x.shape
        H, W = target_size
        x = self.classifier(x).transpose(1, 2).reshape(B, -1, H, W)
        return x

if __name__ == '__main__':
    model = MLPDecoder(dims=[32,64,160,256])
    x = torch.randn(1, 49, 32)      # 7x7
    x2 = torch.randn(1, 196, 64)    # 14x14
    x3 = torch.randn(1, 784, 160)   # 28x28
    x4 = torch.randn(1, 3136, 256)  # 56x56
    y = model([(x,7,7),(x2,14,14),(x3,28,28),(x4,56,56)])
    print(y.shape)