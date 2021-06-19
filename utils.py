import torch, cv2, os
import numpy as np
import torch.nn.functional as funct
from collections import defaultdict
from torchvision.transforms import Compose, Resize, Normalize, ToTensor


def transform():
    return Compose([Resize(288), ToTensor()])

def transform_midas():
    return Compose(
        [
            Resize(288),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

def depth_to_rgb(depth):
    device = depth.device
    depth = (depth - depth.min()) / (depth.max() - depth.min())      
    depth = depth.squeeze(0).detach().cpu().numpy()
    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
    depth = np.transpose(depth, (2,0,1))
    return torch.from_numpy(depth).to(device)


def read_files(directory, ext=('.png', '.jpg', '.jpeg'), skip_empty=True):
    files = defaultdict(list)
    for entry in os.scandir(directory):
        relpath = os.path.relpath(entry.path, directory)
        if entry.is_dir():
            d_files = read_files(entry.path, ext=ext, skip_empty=skip_empty)
            if skip_empty and not len(d_files):
                continue
            files[relpath] = d_files[entry.path]
        elif entry.is_file():
            if ext is None or entry.path.lower().endswith(tuple(ext)):
                files[directory].append(relpath)
    return files

    
def inv2depth(inv_depth):
    if isinstance(inv_depth, tuple) or isinstance(inv_depth, list):
        return [inv2depth(item) for item in inv_depth]
    else:
        return 1. / inv_depth.clamp(min=1e-6)


def view_synthesis(ref_image, depth, ref_cam, cam,
                   mode='bilinear', padding_mode='zeros'):
    assert depth.size(1) == 1
    world_points = cam.reconstruct(depth, frame='w')
    ref_coords = ref_cam.project(world_points, frame='w')
    return funct.grid_sample(ref_image, ref_coords, mode=mode,
                             padding_mode=padding_mode, align_corners=True)


def calc_smoothness(inv_depths, images, num_scales):
    inv_depths_norm = inv_depths_normalize(inv_depths)
    inv_depth_gradients_x = [gradient_x(d) for d in inv_depths_norm]
    inv_depth_gradients_y = [gradient_y(d) for d in inv_depths_norm]

    image_gradients_x = [gradient_x(image) for image in images]
    image_gradients_y = [gradient_y(image) for image in images]

    weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
    weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

    smoothness_x = [inv_depth_gradients_x[i] * weights_x[i] for i in range(num_scales)]
    smoothness_y = [inv_depth_gradients_y[i] * weights_y[i] for i in range(num_scales)]
    return smoothness_x, smoothness_y


def inv_depths_normalize(inv_depths):
    mean_inv_depths = [inv_depth.mean(2, True).mean(3, True) for inv_depth in inv_depths]
    return [inv_depth / mean_inv_depth.clamp(min=1e-6)
            for inv_depth, mean_inv_depth in zip(inv_depths, mean_inv_depths)]


def gradient_x(image):
    return image[:, :, :, :-1] - image[:, :, :, 1:]


def gradient_y(image):
    return image[:, :, :-1, :] - image[:, :, 1:, :]


def match_scales(image, targets, num_scales):
    images = []
    image_shape = image.shape[-2:]
    for i in range(num_scales):
        target_shape = targets[i].shape
        if same_shape(image_shape, target_shape):
            images.append(image)
        else:
            images.append(interpolate_image(image, target_shape))
    return images


def same_shape(shape1, shape2):
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False
    return True


def interpolate_image(image, shape, mode='bilinear', align_corners=True):
    if len(shape) > 2:
        shape = shape[-2:]
    if same_shape(image.shape[-2:], shape):
        return image
    else:
        return funct.interpolate(image, size=shape, mode=mode,
                                 align_corners=align_corners)