import torch
import torch.nn as nn

from pose import Pose


def scale_intrinsics(K, x_scale, y_scale):
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
    return K

def image_grid(B, H, W, dtype, device):
    xs, ys = meshgrid(B, H, W, dtype, device)
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=1)
    return grid

def meshgrid(B, H, W, dtype, device):
    xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
    ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    ys, xs = torch.meshgrid([ys, xs])
    return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1])


class Camera(nn.Module):
    def __init__(self, K, Tcw=None):
        super().__init__()
        self.K = K
        self.Tcw = Pose.identity(len(K)) if Tcw is None else Tcw

    def __len__(self):
        return len(self.K)

    def to(self, *args, **kwargs):
        self.K = self.K.to(*args, **kwargs)
        self.Tcw = self.Tcw.to(*args, **kwargs)
        return self

    @property
    def fx(self):
        return self.K[:, 0, 0]

    @property
    def fy(self):
        return self.K[:, 1, 1]

    @property
    def cx(self):
        return self.K[:, 0, 2]

    @property
    def cy(self):
        return self.K[:, 1, 2]

    @property
    def Twc(self):
        return self.Tcw.inverse()

    @property
    def Kinv(self):
        Kinv = self.K.clone()
        Kinv[:, 0, 0] = 1. / self.fx
        Kinv[:, 1, 1] = 1. / self.fy
        Kinv[:, 0, 2] = -1. * self.cx / self.fx
        Kinv[:, 1, 2] = -1. * self.cy / self.fy
        return Kinv

    def scaled(self, x_scale, y_scale=None):
        if y_scale is None:
            y_scale = x_scale
        if x_scale == 1. and y_scale == 1.:
            return self
        K = scale_intrinsics(self.K.clone(), x_scale, y_scale)
        return Camera(K, Tcw=self.Tcw)

    def reconstruct(self, depth, frame='w'):
        B, C, H, W = depth.shape
        grid = image_grid(B, H, W, depth.dtype, depth.device)
        flat_grid = grid.view(B, 3, -1)
        xnorm = (self.Kinv.bmm(flat_grid)).view(B, 3, H, W)
        Xc = xnorm * depth
        if frame == 'c':
            return Xc
        else:
            return self.Twc @ Xc

    def project(self, X, frame='w'):
        B, C, H, W = X.shape
        if frame == 'c':
            Xc = self.K.bmm(X.view(B, 3, -1))
        else:
            Xc = self.K.bmm((self.Tcw @ X).view(B, 3, -1))
        X = Xc[:, 0]
        Y = Xc[:, 1]
        Z = Xc[:, 2].clamp(min=1e-5)
        Xnorm = 2 * (X / Z) / (W - 1) - 1.
        Ynorm = 2 * (Y / Z) / (H - 1) - 1.
        return torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2)