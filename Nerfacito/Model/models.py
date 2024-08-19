import numpy as np
import torch
import torch.nn as nn

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
    
    
# TODO: 
#   - Apply mean centering given the poses, or focus centering
#   - Apply rotation given the poses (mean rotating)

class Voxels(nn.Module):
    
    def __init__(self, nb_voxels=100, scale=1, device="cpu") -> None:
        super(Voxels, self).__init__()
        
        self.nb_voxels = nb_voxels
        self.voxels    = torch.nn.Parameter(torch.rand((self.nb_voxels,self.nb_voxels,self.nb_voxels,4), device=device),requires_grad=True) # XYZ,RGBA
        self.scale = scale
        self.device = device
        
    def forward(self, X):
        """
        X: nb_rays*nb_bins,3
        """
        x = X[:,0]
        y = X[:,1]
        z = X[:,2]
        
        # Are given points inside voxel grid?
        cond = (x.abs() < (self.scale/2.)) & (y.abs() <(self.scale/2.)) & (z.abs() <(self.scale/2.)) 
        # Digitize (Coordinate -> Index)
        indx = (x[cond] / (self.scale/self.nb_voxels) + self.nb_voxels/2).floor().type(torch.long) - 1
        indy = (y[cond] / (self.scale/self.nb_voxels) + self.nb_voxels/2).floor().type(torch.long) - 1
        indz = (z[cond] / (self.scale/self.nb_voxels) + self.nb_voxels/2).floor().type(torch.long) - 1
        
        colors_densities = torch.zeros((X.shape[0],4),dtype=torch.float32, device=self.device)
        colors_densities[cond, :3] = self.voxels[indx,indy,indz,:3]
        colors_densities[cond, -1] = self.voxels[indx,indy,indz,-1] * 10
        
        return torch.sigmoid(colors_densities[:,:3]), torch.relu(colors_densities[:,-1:])
        
    def intersect(self,X):
        return self.forward(X)