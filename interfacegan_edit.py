from tqdm import tqdm 
from PIL import Image 
import torch 
import dnnlib
import legacy
import numpy as np 
import argparse 

class InterfaceGAN:
    def __init__(self, stylegan_model, boundary, device) -> None:
        self.device = device
        with dnnlib.util.open_url(stylegan_model) as f:
            self.stylegan = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        self.attr_norm = torch.load(boundary)['weight']
        
    def edit_w(self, z_np, alpha, truncation_psi=1.0):
        with torch.no_grad():
            z_tensor = torch.from_numpy(z_np).to(self.device)
            label = torch.zeros([1, self.stylegan.c_dim], device=self.device)
            ws_tensor = self.stylegan.mapping(z_tensor, label, truncation_psi)
            w_edit = ws_tensor[:, 0, :] + alpha * self.attr_norm
            w_edit = torch.repeat_interleave(w_edit, self.stylegan.num_ws, 0).unsqueeze(0)
            img_out = self.stylegan.synthesis(w_edit, noise_mode="const")
            img_out = (img_out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
        return img_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stylegan", type=str, default="stylegan3-t-ffhq-1024x1024.pkl", help="stylegan model path")
    parser.add_argument("--boundary", type=str, default="resources/interfacegan/boundary_young.pth")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    interfacegan = InterfaceGAN(args.stylegan, args.boundary, args.device)
    z_np = np.random.randn(1, interfacegan.stylegan.z_dim)
    outs = []
    for i in tqdm(range(8)):
        edited_out = interfacegan.edit_w(z_np, alpha=0.5 * i)
        outs.append(edited_out)
    outs = np.concatenate(outs, axis=1)
    Image.fromarray(np.uint8(outs)).save("interfacegan_edited.jpg")
    
