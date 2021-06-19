import os, torch, random
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from utils import read_files, transform
from camera import scale_intrinsics


class ImageDataset(Dataset):
    def __init__(self, root_dir, images_dir, poses_dir, intrinsics_dir,
                  context=1, stride=4, step=1):
        super().__init__()
        self.root_dir = root_dir
        self.images_dir = os.path.join(self.root_dir, images_dir)
        self.poses_dir = os.path.join(self.root_dir, poses_dir)
        self.intrinsics = np.genfromtxt(os.path.join(self.root_dir, intrinsics_dir), delimiter=' ')
        self.context = context
        self.stride = stride
        self.step = step
        self.transform = transform()
        self.poses = sorted(read_files(self.poses_dir, ext=('.txt'))[self.poses_dir])
        self.files = sorted(read_files(self.images_dir)[self.images_dir])

    def __len__(self):
        return len(self.files)

    def _get_context_file_paths(self, idx):
        context_left = list(np.arange(idx-self.stride*self.step,idx,self.step))
        context_right = list(np.arange(idx+self.step,idx+self.stride*self.step+self.step,self.step))
        context_left = [ctx for ctx in context_left if ctx >= 0]
        context_right = [ctx for ctx in context_right if ctx < len(self.files)]
        if len(context_left) > 0 and len(context_right) > 0:
            return random.sample(context_left,self.context)+random.sample(context_right,self.context)
        elif len(context_left) > 0:
            return np.random.choice(context_left,self.context*2).tolist()
        else:
            return np.random.choice(context_right,self.context*2).tolist()

    def _read_rgb_context_files(self, idx):
        context_idxs = self._get_context_file_paths(idx)
        images = []
        poses = []
        for idx in context_idxs:
            image = self._read_rgb_file(self.files[idx])
            images.append(image)
            pose = self._read_pose_file(self.poses[idx])
            poses.append(pose)
        return images, poses
    
    def _read_rgb_file(self, filename):
        return Image.open(os.path.join(self.images_dir, filename))

    def _read_pose_file(self, filename):
        return np.genfromtxt(os.path.join(self.poses_dir, filename), delimiter=' ')

    def _get_transition_mtx(self, target, source):
        transform_mtx = np.linalg.inv(source) @ target
        return torch.from_numpy(transform_mtx).float()
    
    def __getitem__(self, idx):
        img_target = self._read_rgb_file(self.files[idx])
        pose_target = self._read_pose_file(self.poses[idx])
        imgs_source, poses_source = self._read_rgb_context_files(idx)
        
        trans_matrices = []
        for pose_source in poses_source:
            trans_matrix = self._get_transition_mtx(pose_target, pose_source)
            trans_matrices.append(trans_matrix)
        
        img_target_resized = self.transform(img_target).squeeze()
        imgs_source_resized = [self.transform(src).squeeze() for src in imgs_source]

        intrinsics = torch.from_numpy(self.intrinsics)
        
        old_W, old_H = img_target.size
        _, new_H, new_W = img_target_resized.shape

        x_scale = new_W / old_W
        y_scale = new_H / old_H

        intrinsics = scale_intrinsics(intrinsics.clone(), x_scale, y_scale)

        return img_target_resized, imgs_source_resized, trans_matrices, intrinsics
