from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvBlockDepth(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlockDepth, self).__init__()

        self.conv = DepthConv3x3(in_channels, out_channels)
        self.nonlin = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class DepthConv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(DepthConv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        # self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, groups=int(out_channels), bias=False)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        # self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=3 // 2, groups=int(out_channels), bias=False)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x, scale_factor=2, mode="bilinear"):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

# BR-LiteDepth: tensor-only helpers; no new package/directory is required.
def br_masked_mean(x, mask):
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def br_erode(mask, radius=1):
    if radius == 0:
        return mask
    invalid = F.pad(1.0 - mask, (radius,) * 4, value=1.0)
    return 1.0 - F.max_pool2d(invalid, 2 * radius + 1, stride=1)


def br_smooth_loss(disp, image, mask):
    mean = (disp * mask).sum((2, 3), keepdim=True)
    mean = mean / mask.sum((2, 3), keepdim=True).clamp_min(1.0)
    d = disp / (mean + 1e-7)
    dx = (d[..., 1:] - d[..., :-1]).abs()
    dy = (d[..., 1:, :] - d[..., :-1, :]).abs()
    ix = (image[..., 1:] - image[..., :-1]).abs().mean(1, True)
    iy = (image[..., 1:, :] - image[..., :-1, :]).abs().mean(1, True)
    mx = mask[..., 1:] * mask[..., :-1]
    my = mask[..., 1:, :] * mask[..., :-1, :]
    return br_masked_mean(dx * torch.exp(-ix), mx) + br_masked_mean(dy * torch.exp(-iy), my)


def br_gaussian(x, sigma_x, sigma_y):
    # sigma is determined by known image geometry, not learned.
    for axis, sigma in [('x', sigma_x), ('y', sigma_y)]:
        if sigma < 1e-5:
            continue
        radius = int(math.ceil(3.0 * sigma))
        t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        w = torch.exp(-0.5 * (t / sigma).square())
        w = w / w.sum()
        kernel = w.view(1, 1, 1, -1) if axis == 'x' else w.view(1, 1, -1, 1)
        pad = (0, radius) if axis == 'x' else (radius, 0)
        x = F.conv2d(x, kernel, padding=pad)
    return x


def br_cross_scale(za, zb, ka, kb, ma, mb, mode):
    """Return loss and common-grid valid fraction for a batch.

    mode=ra: fine common grid, unfiltered SSIM+L1 depth control.
    mode=log: coarse common grid, unfiltered log-depth control.
    mode=band: SAME coarse grid/log loss as log, with Gaussian prefilter.
    """
    assert mode in ('ra', 'log', 'band')
    assert za.shape == zb.shape and za.shape[1] == 1
    losses, fractions = [], []
    h, w = za.shape[-2:]
    for n in range(za.shape[0]):
        aa, bb = za[n:n+1], zb[n:n+1]
        va, vb = ma[n:n+1], mb[n:n+1]
        ca, cb = ka[n, :3, :3], kb[n, :3, :3]
        fxa, fya, cxa, cya = [float(ca[i, j]) for i, j in [(0,0),(1,1),(0,2),(1,2)]]
        fxb, fyb, cxb, cyb = [float(cb[i, j]) for i, j in [(0,0),(1,1),(0,2),(1,2)]]
        assert min(fxa, fya, fxb, fyb) > 0
        choose = max if mode == 'ra' else min
        fx, fy = choose(fxa, fxb), choose(fya, fyb)
        # Find rectangular real-image support, before filtering.
        boxes = []
        for vm, fxx, fyy, cx, cy in [(va,fxa,fya,cxa,cya),(vb,fxb,fyb,cxb,cyb)]:
            yy, xx = torch.where(vm[0,0] > 0.5)
            if xx.numel() == 0:
                raise RuntimeError('BR consistency: empty real-image support')
            boxes.append(((float(xx.min())-cx)/fxx, (float(xx.max())-cx)/fxx,
                          (float(yy.min())-cy)/fyy, (float(yy.max())-cy)/fyy))
        left, right = max(b[0] for b in boxes), min(b[1] for b in boxes)
        top, bottom = max(b[2] for b in boxes), min(b[3] for b in boxes)
        nx = int(math.floor((right-left)*fx)) + 1
        ny = int(math.floor((bottom-top)*fy)) + 1
        if min(nx, ny) < 4:
            raise RuntimeError('BR consistency: insufficient overlapping field of view')
        xs = left + torch.arange(nx,device=za.device,dtype=za.dtype)/fx
        ys = top + torch.arange(ny,device=za.device,dtype=za.dtype)/fy
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')

        samples, supports = [], []
        for depth, valid, fxx, fyy, cx, cy in [(aa,va,fxa,fya,cxa,cya),(bb,vb,fxb,fyb,cxb,cyb)]:
            if mode in ('log','band'):
                sx = 0.5 * math.sqrt(max((fxx/fx)**2-1.0, 0.0))
                sy = 0.5 * math.sqrt(max((fyy/fy)**2-1.0, 0.0))
                weight = br_gaussian(valid, sx, sy)
                if mode == 'band':
                    depth = br_gaussian(depth*valid, sx, sy) / weight.clamp_min(1e-6)
                # log/band compare exactly the same conservative support.
                valid = (weight > 0.9999).to(depth.dtype)
            grid = torch.stack((2*(xx*fxx+cx)/(w-1)-1,
                                2*(yy*fyy+cy)/(h-1)-1), -1)[None]
            samples.append(F.grid_sample(depth,grid,mode='bilinear',padding_mode='zeros',align_corners=True))
            supports.append(F.grid_sample(valid,grid,mode='bilinear',padding_mode='zeros',align_corners=True))
        mask = ((supports[0] > 0.9999) & (supports[1] > 0.9999)).to(za.dtype)
        # Same erosion for all modes; SSIM has a 3x3 support.
        mask = br_erode(mask, 1)
        if float(mask.sum()) < 4:
            raise RuntimeError('BR consistency: no reliable common pixels')
        a, b = samples
        if mode == 'ra':
            error = 0.85 * SSIM()(a,b) + 0.15 * (a-b).abs()
        else:
            delta = a.clamp_min(1e-6).log() - b.clamp_min(1e-6).log()
            error = torch.sqrt(delta.square()+1e-6)
        losses.append(br_masked_mean(error,mask))
        fractions.append(mask.mean().detach())
    return torch.stack(losses).mean(), torch.stack(fractions).mean()
