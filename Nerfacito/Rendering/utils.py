import os, json, random
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import imageio
from pathlib import Path,PosixPath 

def initialize_rays(H,W,fx, fy, device="cpu"):
    u,v = np.arange(W), np.arange(H)
    u,v = np.meshgrid(u,v)

    rays_o = np.zeros((H*W, 3))
    rays_d = np.stack((
        (u - W/2)   /fx,
        -(v - H/2) / fy,
        -np.ones_like(u)
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
    return torch.cat((
                torch.ones(acc.shape[0],1, device=acc.device),
                acc[:,:-1]
            ),dim=1)

def rendering(models, rays_o, rays_d, tn = 0.2, tf = 1.8, nb_bins = 100, device="cpu"):
    """
    Given array of models along with ray origins and ray directions, computes volumetric rendering
    and returns the color and depth
    
    @Args:
    - model: List of models with _intersect()_ method.
    - rays_o : Tensor of ray origins (W*H,3)
    - rays_d : Tensor of ray directions (W*H,3)
    - tn     : Near plane distance
    - tf     : Far plane distance
    - nb_bins : Nuber of bins along ray direction
    - device  : cpu or cuda
        
    @Returns:
    - colors: Colors computed from volumetric rendering (H*W,3)
    - depths: Depths computed from alpha (H*W,)
    """
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
    weights = accumulated_transmittance(1-alpha) * alpha # T(r) * sigma(r)
    img = (weights.unsqueeze(-1) * colors).sum(1)
    return img, alpha.sum(1)

def euler_to_rotation_matrix(ypr):
    R_ned = Rotation.from_euler("ZYX", ypr, degrees=True)
    return R_ned.as_matrix()
    
def create_homogeneous_matrix(rotation_matrix, translation_vector=None):
    if len(rotation_matrix.shape) == 2:
        H = np.eye(4)
        H[:3,:3] = rotation_matrix
        if translation_vector is not None:
            H[:3,3] = translation_vector
        return H 
      
    else:
        H = np.zeros((rotation_matrix.shape[0],4,4))
        H[:,3,3] = 1.
        H[:,:3,:3] = rotation_matrix
        if translation_vector is not None:
            H[:,:3,3] = translation_vector
        return H
    

def get_rays(DIR : PosixPath, image_downscale_factor = 2, max_num_images=None, random_sample=False):
    ### Find json file and image folders
    json_path = Path.joinpath(DIR, "transforms.json")
    images_path = Path.joinpath(DIR, f"images_{image_downscale_factor}")
    
    assert os.path.isfile(json_path), f"transforms.json does not exists in directory {DIR}"
    assert os.path.isdir(images_path), f"There are no images_{image_downscale_factor} folders in directory {DIR}"
    
    ### Read json file and images
    with open(json_path, "r") as jfile:
        json_file_contents =  json.load(jfile)
    
    image_paths = [os.path.join(images_path, i) for i in os.listdir(images_path)]
    assert len(json_file_contents["frames"]) == len(image_paths), "Number of camera poses must match the number of images"
    
    framenames_transforms = [(json_file_contents["frames"][i]["file_path"],json_file_contents["frames"][i]["transform_matrix"]) for i in range(len(json_file_contents["frames"]))]
    
    get_image = lambda fname:  image_paths[image_paths.index(os.path.join(DIR.as_posix(),f"images_{image_downscale_factor}",fname.split("/")[-1]))]
    fname_poseimagepath = [(pose,get_image(fname)) for fname,pose in framenames_transforms]
    
    if random_sample:
        assert max_num_images is not None, "In random subsampling, max_num_images must be an int."
        fname_poseimagepath = random.sample(fname_poseimagepath, max_num_images)
    
    frames,poses = [],[]
    for pose,imagepath in fname_poseimagepath:
        frame = imageio.imread(imagepath) / 255. # To map to [0,1]
        pose  = np.array(pose)
        pose  = np.linalg.inv(pose)
        frames.append(frame)
        poses.append(pose)       
        
    frames,poses = np.array(frames),np.array(poses)
    
    # RGBA -> RGB
    if frames.shape[3] == 4: 
        frames = frames[...:3] * frames[...:-1:] + (1 - frames[...:-1:])
    
    N = len(frames)

    H,W = frames[0].shape[0], frames[0].shape[1]
    rays_o_t, rays_d_t = torch.zeros((N,W*H,3)), torch.zeros((N,W*H,3))
    ground_truths = torch.Tensor(frames.reshape(N,H*W,3))
    for i in range(N): # TODO: Apply intrinsic transformations when scaled images are used! 
        c2w = torch.Tensor(poses[i])
        fx,fy = json_file_contents["fl_x"], json_file_contents["fl_y"]         
        rays_o, rays_d = initialize_rays(H,W,fx,fy, device="cpu")  
        rays_o, rays_d = apply_camera_transformation(rays_o, rays_d, c2w)
        rays_o_t[i] = rays_o
        rays_d_t[i] = rays_d
        
        
    infos_dict = {"H": H, "W":W, "fname_poseimagepath":fname_poseimagepath}
    return rays_o_t, rays_d_t,ground_truths, infos_dict
