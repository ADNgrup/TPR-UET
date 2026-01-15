import math
import torch
import torch.nn.functional as F
from typing import Tuple, Union
def get_gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array."""
    def _gaussian_fn(kernlen, std):
        n = torch.arange(0, kernlen).float()
        n -= n.mean()
        n /= std
        w = torch.exp(-0.5 * n**2)
        return w
    gkern1d = _gaussian_fn(kernlen, std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d / gkern2d.sum()


class HOGLayerC(torch.nn.Module):
    def __init__(self, nbins=9, pool=8, gaussian_window=0, norm_pix_loss=True):
        super(HOGLayerC, self).__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer("weight_x", weight_x)
        self.register_buffer("weight_y", weight_y)
        self.norm_pix_loss = norm_pix_loss

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gkern = get_gkern(gaussian_window, gaussian_window//2)
            self.register_buffer("gkern", gkern)

    @torch.no_grad()
    def forward(self, x):
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")

        #norm image 
        gx_rgb = F.conv2d(x, self.weight_x, bias=None, stride=1, padding=0, groups=3)
        gy_rgb = F.conv2d(x, self.weight_y, bias=None, stride=1, padding=0, groups=3)
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)   #torch.Size([2, 3, 64, 64])
       
        phase = torch.atan2(gx_rgb, gy_rgb) #calculate HOG ?????
        phase = phase / math.pi * self.nbins  # [-9, 9]
        b, c, h, w = norm_rgb.shape
        
        out = torch.zeros((b, c, self.nbins, h, w), dtype=torch.float, device=x.device)
        phase = phase.reshape(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)

        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, "h {} gw {}".format(h, self.gaussian_window)
                repeat_rate = h // self.gaussian_window
                temp_gkern = self.gkern.repeat([repeat_rate, repeat_rate])
            else:
                temp_gkern = self.gkern
            norm_rgb *= temp_gkern
        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb) #HOG-Ground truth = B x C x bins x H x W
       
        out = out.unfold(3, self.pool, self.pool)   #HOG-Ground truth = B x C x bins x (H/pool)  x  W x pool
        out = out.unfold(4, self.pool, self.pool)   #HOG-Ground truth = B x C x bins x (H/pool)  x  (W/pool) x pool x pool
        out = out.sum(dim=[-1, -2])                 #HOG-Ground truth = B x (C*bins) x pool x  pool 

        if self.norm_pix_loss:
            out = torch.nn.functional.normalize(out, p=2, dim=2)

        out = out.flatten(1, 2) 
        
        return out
 


class L2MIMLoss(torch.nn.Module):
    def __init__(self, patch_size, norm_pix_loss=False):
        super(L2MIMLoss, self).__init__()
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->npqchw', x)
        x = x.reshape(shape=(imgs.shape[0], p**2 * 3, h, w))
        return x
    
    def forward(self, imgs, pred, mask):
        B, N, C = pred.shape
        H = imgs.shape[2] // self.patch_size
        W = imgs.shape[3] // self.patch_size
        pred = pred.transpose(-1,-2).reshape(B, C, H, W)
        
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=1, keepdim=True)
            var = target.var(dim=1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        mask = mask.repeat(1, C, 1, 1).bool()

        loss = (pred[mask] - target[mask]) ** 2        
        return loss