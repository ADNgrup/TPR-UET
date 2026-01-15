import torch
import torch.nn as nn
import torch.nn.functional as F
from model.clip_model import Transformer, QuickGELU, LayerNorm
from collections import OrderedDict


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    """https://github.com/woodfrog/vse_infty, thanks!"""
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results

def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)

def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) from https://github.com/woodfrog/vse_infty, thanks!"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B * N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x
 
class TexualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=1024,ratio=0.3):
        super(TexualEmbeddingLayer, self).__init__()
        self.embed_dim= embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.ratio = ratio

    def forward(self, features, text, atten):
        # print(atten) 64 x 77 x 77
        mask =  ((text != 0) + 0)
        lengths = mask.sum(1).view(-1) - 2 # -2 for SOS token and EOS token
        k = int((atten.size(1)-2)*self.ratio)
        bs = features.size(0)
        atten[torch.arange(bs), :, text.argmax(dim=-1)] = -1 # last token 
        atten[torch.arange(bs), :, 0] = -1 # first token 
        atten = atten[torch.arange(bs), text.argmax(dim=-1), :] # 64 x 77
        atten = atten * mask
        
        atten_topK = atten.topk(dim=-1,k = k)[1].unsqueeze(-1).expand(bs,k,features.size(2)) # 64 x k x 512
        features = torch.gather(input=features,dim=1,index=atten_topK)  # 64 x k x 512
        features = l2norm(features, dim=-1)

        lengths = torch.Tensor([lengths[i] if lengths[i] < k else k for i in range(bs)]) # Keep at least K
        
        cap_emb = self.linear(features.half())
        features = self.mlp(features) + cap_emb
        features = maxk_pool1d_var(features, 1, 1, lengths.to(cap_emb.device))  # max 
        
        return features.float()

class VisualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=1024,ratio=0.3):
        super(VisualEmbeddingLayer, self).__init__()
        self.embed_dim= embed_dim
        self.ratio = ratio
        self.fc = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
    
    def forward(self, base_features, atten):
        k = int((atten.size(1)-1)*self.ratio) # 192
        
        bs = base_features.size(0)
        atten[torch.arange(bs), :, 0] = -1 # CLS token   
        atten_topK = atten[:,0].topk(dim=-1,k = k)[1]
        
        atten_topK = atten_topK.unsqueeze(-1).expand(bs, k, base_features.size(2)) # 64 x k x 512
        base_features = torch.gather(input=base_features,dim=1,index=atten_topK)  # 64 x k x 512
        base_features = l2norm(base_features, dim=-1) 
        base_features = base_features.half()
        feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device).half()
        feat_lengths[:] = base_features.size(1)
        
        features = self.fc(base_features)
        features = self.mlp(base_features) + features 
        features = maxk_pool1d_var(features, 1, 1, feat_lengths)   # max 
 
        return features.float()
    
class TexualSelectedEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=1024,ratio=0.5):
        super(TexualSelectedEmbeddingLayer, self).__init__()
        self.embed_dim= embed_dim
        self.fc = nn.Linear(input_dim, embed_dim)
        # self.attention = nn.MultiheadAttention(self.embed_dim, self.embed_dim // 64, batch_first=True)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.ratio = ratio

    def forward(self, text_feats, text, attenscore):
        """
        text_feats: BxLxD
        attenscore: BxLxL
        """
        mask =  ((text != 0) + 0)
        lengths = mask.sum(1).view(-1) - 2 # -2 for SOS token and EOS token
        k = int((attenscore.size(1)-2)*self.ratio)
        bs = text_feats.size(0)
        attenscore[torch.arange(bs), :, text.argmax(dim=-1)] = -1 # last token 
        attenscore[torch.arange(bs), :, 0] = -1 # first token 
        attenscore = attenscore[torch.arange(bs), text.argmax(dim=-1), :] # 64 x 77
        attenscore = attenscore * mask
        
        atten_topK = attenscore.topk(dim=-1,k = k)[1].unsqueeze(-1).expand(bs,k,text_feats.size(2)) # 64 x k x 512
        base_features = torch.gather(input=text_feats,dim=1,index=atten_topK)  # 64 x k x 512
        base_features = l2norm(base_features, dim=-1)

        lengths = torch.Tensor([lengths[i] if lengths[i] < k else k for i in range(bs)]) # Keep at least K
        
        # cap_emb = self.attention(            
        #     base_features,
        #     base_features,
        #     base_features,
        #     need_weights=False
        # )[0]
        
        # localfeatures = self.mlp(base_features.half()) + cap_emb.half()
        # localfeatures = self.fc(base_features.half()) + cap_emb.half()
        cap_emb = self.mlp(base_features.half())
        localfeatures = self.fc(base_features.half()) + cap_emb
        localfeatures = maxk_pool1d_var(localfeatures, 1, 1, lengths.to(cap_emb.device))  # max 
        
        return localfeatures.float()

class VisualSelectedEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=1024,ratio=0.5):
        super(VisualSelectedEmbeddingLayer, self).__init__()
        self.embed_dim= embed_dim
        self.ratio = ratio
        self.fc = nn.Linear(input_dim, embed_dim)
        # self.attention = nn.MultiheadAttention(self.embed_dim, self.embed_dim // 64, batch_first=True)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
    
    def forward(self, base_features, atten):
        """
        image_feats: BxLxD
        attenscore: BxLxL
        """
        k = int((atten.size(1)-1)*self.ratio) # 192
        
        bs = base_features.size(0)
        atten[torch.arange(bs), :, 0] = -1 # CLS token   
        atten_topK = atten[:,0].topk(dim=-1,k = k)[1]
        
        atten_topK = atten_topK.unsqueeze(-1).expand(bs, k, base_features.size(2)) # 64 x k x 512
        base_features = torch.gather(input=base_features,dim=1,index=atten_topK)  # 64 x k x 512
        base_features = l2norm(base_features, dim=-1) 
        base_features = base_features.half()
        feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device).half()
        feat_lengths[:] = base_features.size(1)

        # x_att = self.attention(
        #     base_features,
        #     base_features,
        #     base_features,
        #     need_weights=False
        # )[0]
        # localfeatures = self.mlp(base_features.half()) + x_att.half()
        # localfeatures = self.fc(base_features.half()) + x_att.half()
        x_att  = self.mlp(base_features.half())
        localfeatures = self.fc(base_features.half()) + x_att
        
        feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device).half()
        feat_lengths[:] = base_features.size(1)
        localfeatures = maxk_pool1d_var(localfeatures, 1, 1, feat_lengths)   # max 
 
        return localfeatures.float()
    
class ResidualSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head:int):
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            # ("c_fc", nn.Linear(d_model, d_model * 4)),
            # ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model, d_model))
        ]))
        self.gap = nn.AdaptiveMaxPool2d((1, None)) #nn.AdaptiveAvgPool2d
        nn.init.normal_(self.mlp.c_proj.weight.data, std=0.001)
        nn.init.constant_(self.mlp.c_proj.bias.data, val=0.0)
        nn.init.normal_(self.attention.in_proj_weight, std=0.1)
        nn.init.normal_(self.attention.out_proj.weight, std=0.1)

    def forward(self, x: torch.Tensor):
        x_ = self.attention(
            self.ln_1(x),
            self.ln_1(x),
            self.ln_1(x),
            need_weights=False
        )[0]
        x = x_ + self.mlp(self.ln_2(x))
        x = self.gap(x).squeeze(1).squeeze(-1) #get highlight features
        return x