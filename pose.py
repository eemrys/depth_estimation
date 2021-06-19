import torch


def invert_pose(T):
    Tinv = torch.eye(4, device=T.device, dtype=T.dtype).repeat([len(T), 1, 1])
    Tinv[:, :3, :3] = torch.transpose(T[:, :3, :3], -2, -1)
    Tinv[:, :3, -1] = torch.bmm(-1. * Tinv[:, :3, :3], T[:, :3, -1].unsqueeze(-1)).squeeze(-1)
    return Tinv


class Pose:
    def __init__(self, mat):
        assert tuple(mat.shape[-2:]) == (4, 4)
        if mat.dim() == 2:
            mat = mat.unsqueeze(0)
        assert mat.dim() == 3
        self.mat = mat

    def __len__(self):
        return len(self.mat)

    @classmethod
    def identity(cls, N=1, device=None, dtype=torch.float):
        return cls(torch.eye(4, device=device, dtype=dtype).repeat([N,1,1]))

    @property
    def shape(self):
        return self.mat.shape

    def item(self):
        return self.mat

    def repeat(self, *args, **kwargs):
        self.mat = self.mat.repeat(*args, **kwargs)
        return self

    def inverse(self):
        return Pose(invert_pose(self.mat))

    def to(self, *args, **kwargs):
        self.mat = self.mat.to(*args, **kwargs)
        return self

    def transform_pose(self, pose):
        assert tuple(pose.shape[-2:]) == (4, 4)
        return Pose(self.mat.bmm(pose.item()))

    def transform_points(self, points):
        assert points.shape[1] == 3
        B, _, H, W = points.shape
        out = self.mat[:,:3,:3].bmm(points.view(B, 3, -1)) + \
              self.mat[:,:3,-1].unsqueeze(-1)
        return out.view(B, 3, H, W)

    def __matmul__(self, other):
        if isinstance(other, Pose):
            return self.transform_pose(other)
        elif isinstance(other, torch.Tensor):
            if other.shape[1] == 3 and other.dim() > 2:
                assert other.dim() == 3 or other.dim() == 4
                return self.transform_points(other)
            else:
                raise ValueError()
        else:
            raise NotImplementedError()