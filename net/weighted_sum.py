import torch
import torch.nn as nn
import torch.nn.functional as F

class Weighted_Sum(nn.Module):
    def __init__(self):
        super(Weighted_Sum, self).__init__()

        weight_dim = 25
        self.dim = 1024

        self.weights = nn.Parameter(torch.ones(weight_dim))
        self.softmax = nn.Softmax(-1)
        layer_norm  = []
        for _ in range(weight_dim):
            layer_norm.append(nn.LayerNorm(self.dim))
        self.layer_norm = nn.Sequential(*layer_norm)

    def forward(self, layer_reps):
        ssl = torch.cat(layer_reps,2)

        lms  = torch.split(ssl, self.dim, dim=2)
        for i,(lm,layer,weight) in enumerate(zip(lms,self.layer_norm,self.softmax(self.weights))):
            lm = layer(lm)
            if i==0:
                out = lm*weight
            else:
                out = out+lm*weight

        return out