import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

class Sphere():
    def __init__(self, center = (0,0,0), radius = 1, color = (1,0,0), density = 10, device = "cpu") -> None:
        self.center = torch.Tensor(center).to(device)
        self.color = torch.Tensor(color).reshape(1,-1).to(device)
        self.radius = radius
        self.density = density
        
        self.device = device
    
    def intersect(self, X):
        """ 
        X : 3D points in space
        """
        
        N = X.shape[0]
        color   = torch.zeros((N,3), device=self.device)
        density = torch.zeros((N,1), device=self.device)
        
        cond = ( (X[:,0] - self.center[0])**2 +  (X[:,1] - self.center[1])**2 +  (X[:,2] - self.center[2])**2 ) <= self.radius**2
        
        color[cond]   = self.color
        density[cond] = self.density
        
        return color,density


class Voxels(nn.Module):
    
    def __init__(self, nb_voxels=100, scale=1, device="cpu") -> None:
        super(Voxels, self).__init__()
        
        self.nb_voxels = nb_voxels
        self.scale = scale
        self.device = device
        
    def forward(self, xyz):
        x = xyz[:,0]
        y = xyz[:,1]
        z = xyz[:,2]
        
        cond = ()
        
        
        

def initialize_rays(H,W, device="cpu"):
    f = 1200

    u,v = np.arange(W), np.arange(H)
    u,v = np.meshgrid(u,v)

    rays_o = np.zeros((H*W, 3))
    rays_d = np.stack((
        (u - W/2),
        -(v - H/2),
        -f * np.ones_like(u)
    ), axis=-1)

    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    rays_d = rays_d.reshape(-1, 3)

    rays_o = torch.Tensor(rays_o).to(device)
    rays_d = torch.Tensor(rays_d).to(device)
    
    return rays_o, rays_d


def apply_camera_transformation(rays_o, rays_d, cam2world):
    rays_d = cam2world[:3,:3] @ rays_d.unsqueeze(-1)
    rays_d = rays_d.squeeze(-1)
    rays_o += cam2world[:3, 3]
    return rays_o, rays_d

def accumulated_transmittance(b):
    acc = torch.cumprod(b, 1)
    acc[:, 0] = 1.
    return acc

def rendering(models, rays_o, rays_d, tn = 0.2, tf = 1.8, nb_bins = 100, device="cpu"):
    t  = torch.linspace(tn, tf, nb_bins, device=device)
    dt = torch.cat((t[1:] - t[:-1], torch.Tensor([1e10]).to(device)))
    
    x = rays_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_d.unsqueeze(1) #[nb_rays, nb_bins, 3]
    
    N = torch.numel(x) // 3
    colors    = torch.zeros((N,3),device=device)
    densities = torch.zeros((N,1),device=device)
    
    for model in models:
        color,density = model.intersect(x.reshape(-1,3))
        colors += color
        densities += density
    
    colors    = colors.reshape(-1, nb_bins, 3)
    densities = densities.reshape(-1, nb_bins)
    
    alpha = 1 - torch.exp(- densities * dt.unsqueeze(0))
    T = accumulated_transmittance(1-alpha)
    img = (T.unsqueeze(-1) * colors * alpha.unsqueeze(-1)).sum(1)
    return img

def euler_to_rotation_matrix(ypr):
    R_ned = Rotation.from_euler("ZYX", ypr, degrees=True)
    return R_ned.as_matrix()

def create_homogeneous_matrix(rotation_matrix, translation_vector=None):
    if len(rotation_matrix.shape) == 2:
        rotation_matrix = rotation_matrix[None,...]
        
    # Concat R and t
    if translation_vector is None:
        if rotation_matrix.shape[0] == 1:
            translation_vector = np.zeros((1,3))
        else:
            translation_vector = np.zeros((rotation_matrix.shape[0],1,3))
    else:
        if len(translation_vector.shape) == 1:
              translation_vector = translation_vector[...,None]
    H1  = np.concatenate((rotation_matrix, translation_vector[...,None]),axis=-1)
    # Put in 4x4 HTM Format
    H1  = np.concatenate((H1,np.zeros((H1.shape[0],1,4))),axis=1)
    H1[:,3,3] += 1
    
    return H1
    