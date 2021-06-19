import torch
import torch.nn as nn

from camera import Camera
from pose import Pose
from utils import inv2depth, view_synthesis, calc_smoothness, match_scales


def SSIM(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d

    return ssim


class MultiViewPhotometricLoss(nn.Module):
    def __init__(self, num_scales=4, ssim_loss_weight=0.85, smooth_loss_weight=0.05):
        super().__init__()
        self.n = num_scales
        self.ssim_loss_weight = ssim_loss_weight
        self.smooth_loss_weight = smooth_loss_weight

    def warp_ref_image(self, inv_depths, ref_image, K, pose):
        B, _, H, W = ref_image.shape
        device = ref_image.get_device()
        cams, ref_cams = [], []
        for i in range(self.n):
            _, _, DH, DW = inv_depths[i].shape
            scale_factor = DW / float(W)
            cams.append(Camera(K=K.float()).scaled(scale_factor).to(device))
            ref_cams.append(Camera(K=K.float(), Tcw=pose).scaled(scale_factor).to(device))
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        ref_images = match_scales(ref_image, inv_depths, self.n)
        ref_warped = [view_synthesis(ref_images[i], depths[i], ref_cams[i],
                                     cams[i]) for i in range(self.n)]
        return ref_warped

    def SSIM(self, x, y):
        ssim_value = SSIM(x, y)
        return torch.clamp((1. - ssim_value) / 2., 0., 1.)

    def calc_photometric_loss(self, t_est, images):
        l1_loss = [torch.abs(t_est[i] - images[i])
                   for i in range(self.n)]
        if self.ssim_loss_weight > 0.0:
            ssim_loss = [self.SSIM(t_est[i], images[i])
                         for i in range(self.n)]
            photometric_loss = [self.ssim_loss_weight * ssim_loss[i].mean(1, True) +
                                (1 - self.ssim_loss_weight) * l1_loss[i].mean(1, True)
                                for i in range(self.n)]
        else:
            photometric_loss = l1_loss
        return photometric_loss

    def reduce_photometric_loss(self, photometric_losses):
        def reduce_function(losses):
            return torch.cat(losses, 1).min(1, True)[0].mean()
        
        photometric_loss = sum([reduce_function(photometric_losses[i])
                                for i in range(self.n)]) / self.n
        return photometric_loss

    def calc_smoothness_loss(self, inv_depths, images):
        smoothness_x, smoothness_y = calc_smoothness(inv_depths, images, self.n)
        smoothness_loss = sum([(smoothness_x[i].abs().mean() +
                                smoothness_y[i].abs().mean()) / 2 ** i
                               for i in range(self.n)]) / self.n
        smoothness_loss = self.smooth_loss_weight * smoothness_loss
        return smoothness_loss

    def forward(self, image, context, inv_depths, K, poses):
        photometric_losses = [[] for _ in range(self.n)]
        images = match_scales(image, inv_depths, self.n)     
        warped = []
        for j, (ref_image, pose) in enumerate(zip(context, poses)):
            pose = Pose(pose)
            ref_warped = self.warp_ref_image(inv_depths, ref_image, K, pose)
            warped.append(ref_warped)
            photometric_loss = self.calc_photometric_loss(ref_warped, images)
            for i in range(self.n):
                photometric_losses[i].append(photometric_loss[i])

            ref_images = match_scales(ref_image, inv_depths, self.n)
            unwarped_image_loss = self.calc_photometric_loss(ref_images, images)
            for i in range(self.n):
                photometric_losses[i].append(unwarped_image_loss[i])

        loss = self.reduce_photometric_loss(photometric_losses)
        if self.smooth_loss_weight > 0.0:
            loss += self.calc_smoothness_loss(inv_depths, images)
        return {
            'loss': loss.unsqueeze(0),
            'warped_context': warped
        }
