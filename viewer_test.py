
from utils import initialize_rays, rendering, apply_camera_transformation
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
import nerfview
import viser
from typing import Tuple

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

def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
    global image
    W, H = img_wh
    c2w = camera_state.c2w
    #c2w = np.linalg.inv(c2w)
    c2w = torch.from_numpy(c2w).float().to("cpu")
    K = camera_state.get_K(img_wh)
    fx,fy = K[0,0],K[1,1]

    if image is None:
        ro,rd = initialize_rays(H,W,fx,fy,device="cpu")
        ro,rd = apply_camera_transformation(ro,rd,c2w)
        render_colors = rendering([model], ro,rd, tn,tf,nb_bins,device="cpu").detach().numpy()
        render_colors = render_colors.reshape(H,W,3)
        image = render_colors
    
    print(image.shape)
    return image

with open("./saved_model_2024_08_16.pkl", "rb") as f:
    load_dict = pickle.load(f)
    
model = load_dict["model"]
tn,tf,nb_bins = load_dict["tn"],load_dict["tf"],load_dict["nb_bins"]
image = np.random.rand()

def main():
    port = 7891
    server = viser.ViserServer(port=port, verbose=False)
    viewer = nerfview.Viewer(
                server=server,
                render_fn=viewer_render_fn,
                mode="rendering",
            )

    print("Viewer running... Ctrl+C to exit.")
    time.sleep(1000000)
    print("Done")

if __name__ == "__main__":
    main()