import torch
import numpy as np
import sys

class BackprojectDepth(torch.nn.Module):
    """
    Backproject absolute depth from ToF to point cloud
    (adapted from https://github.com/nianticlabs/monodepth2)
    """
    def __init__(self, opt):
        super(BackprojectDepth, self).__init__()

        self.batchSize = opt.numPairs
        self.height = opt.height
        self.width = opt.width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.idCoords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.idCoords = torch.tensor(self.idCoords, requires_grad=False)

        self.ones = torch.ones(self.batchSize, 1, self.height * self.width, requires_grad=False)
        self.pixCoords = torch.unsqueeze(torch.stack(
            [self.idCoords[0].view(-1), self.idCoords[1].view(-1)], 0), 0)
        self.pixCoords = self.pixCoords.repeat(self.batchSize, 1, 1)
        self.pixCoords = torch.cat([self.pixCoords, self.ones], 1)

    def forward(self, depth, K):
        invK = torch.linalg.pinv(K)
        camPoints = torch.matmul(invK[:, :3, :3], self.pixCoords)
        camPoints = depth.view(self.batchSize, 1, -1) * camPoints
        camPoints = torch.cat([camPoints, self.ones], 1)
        return camPoints.float()

class ProjectDepth(torch.nn.Module):
    """
    Project point cloud into color camera
     (adapted from https://github.com/nianticlabs/monodepth2)
    """
    def __init__(self, opt, eps=1e-7):
        super(ProjectDepth, self).__init__()
        self.batchSize = opt.numPairs
        self.height = opt.height
        self.width = opt.width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batchSize, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

class Camera(torch.nn.Module):
    """
    projective geometry model
    fx = lens * sensor size"""
    def __init__(self, opt):
        super(Camera, self).__init__()
        self.opt = opt
        
        # projection layers
        self.backprojectDepth = BackprojectDepth(opt)
        self.projectDepth = ProjectDepth(opt)

        # initialize color camera intrinsics K and extrinsics E
        self.initColorCamera()
        # initialize ToF camera intrinsics K
        self.initTofCamera()


    def initTofCamera(self):
        # NYU Depth V2 has calibrated camera parameters
        # self.opt.refine is if you have already have decently good cailibration but 
        # still want to tune it
        if self.opt.refine:
            self.tofK = torch.tensor(
                [[582.624, 0, 0.0313, 0], 
                [0, 582.691, 0.024, 0], 
                [0, 0, 1, 0], 
                [0, 0, 0, 1]], requires_grad=True)[None]
        else:
            # randomly generate focal lengths in a range
            # randomly generate remaining intrinsic parameters between 0 and 1
            f = (400 - 600) * torch.rand(1, 1, requires_grad=True) + 600
            offsets = torch.tensor([[0.5]], requires_grad=True)
            col1 = torch.cat((f, torch.zeros(3, 1, requires_grad=False)))
            col2 = torch.cat( (torch.cat((offsets, f), dim=0), torch.zeros(2,1, requires_grad=False)) )
            col3 = torch.cat((offsets, offsets), dim=0)
            col3 = torch.cat((col3, torch.tensor([[1], [0]], requires_grad=False)), dim=0)
            col4 = torch.tensor([[0], [0], [0], [1]], requires_grad=False)
            self.tofK = torch.nn.Parameter(
                torch.cat((col1, col2, col3, col4), dim=1)[None], 
                requires_grad=True)          

    def initColorCamera(self):
        # NYU Depth V2 has calibrated camera parameters
        if self.opt.refine:
            self.colorK = torch.tensor(
                [[518.858, 0, 0.033, 0], 
                [0, 519.470, 0.024, 0], 
                [0, 0, 1, 0], 
                [0, 0, 0, 1]], requires_grad=True)[None]
            self.colorEgt = torch.tensor(
                [[0.999, 0.0051, 0.0043, 0.025], 
                [-0.0050, 0.999, -0.0037, -0.000293], 
                [-0.00432, 0.0037, 0.999, 0.000662], 
                [0, 0, 0, 1]])[None]
            self.colorEgt = self.colorEgt.transpose(1,2)
            self.colorEgt = torch.linalg.inv(self.colorEgt)
            print(self.colorEgt); sys.exit()
            self.colorEgt = torch.nn.Parameter(self.colorEgt, requires_grad=True)
        else:
            # randomly generate focal lengths in a range
            # randomly generate remaining intrinsic parameters between 0 and 1
            f = (400 - 600) * torch.rand(1, 1, requires_grad=True) + 600
            offsets = torch.tensor([[0.5]], requires_grad=True)
            col1 = torch.cat((f, torch.zeros(3, 1, requires_grad=False)))
            col2 = torch.cat( (torch.cat((offsets, f), dim=0), torch.zeros(2,1, requires_grad=False)) )
            col3 = torch.cat((offsets, offsets), dim=0)
            col3 = torch.cat((col3, torch.tensor([[1], [0]], requires_grad=False)), dim=0)
            col4 = torch.tensor([[0], [0], [0], [1]], requires_grad=False)
            self.colorK = torch.nn.Parameter(
                torch.cat((col1, col2, col3, col4), dim=1)[None], 
                requires_grad=True)

            # randomly generate translation vector and assume identity rotation matrix
            # rotation matrix and translation vector are optimized
            a = torch.eye(3) # rotation matrix
            a = torch.cat((a, torch.zeros(1, 3)), dim=0)
            t = torch.tensor([[.1], [0.], [0.]], requires_grad=True) # translation vec
            t = torch.cat((t, torch.tensor([[1.]])))
            self.colorEgt = torch.cat((a, t), dim=1)[None]
            self.colorEgt = self.colorEgt.transpose(1, 2)
            self.colorEgt = torch.linalg.inv(self.colorEgt)
            self.colorEgt = torch.nn.Parameter(self.colorEgt, requires_grad=True)

    def forward(self, tofDepth, color):
        pointCloud = self.backprojectDepth(tofDepth, self.tofK)
        predCoords = self.projectDepth(pointCloud, self.colorK, self.colorEgt)
        predColor = torch.nn.functional.grid_sample(color, 
                                                    predCoords, 
                                                    padding_mode="border", 
                                                    align_corners=False)
        return predColor
