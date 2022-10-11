from tqdm import tqdm 
from PIL import Image 
import torch 
import dnnlib
import legacy
import numpy as np 
import os 
import argparse 


def generate_data(n_samples, gen_path, np_save_path, stylegan_model, device):
    if not os.path.exists(gen_path):
        os.mkdir(gen_path)
    device = "cuda"
    truncation_psi = 1.0
    with dnnlib.util.open_url(stylegan_model) as f:
        stylegan = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    z_save = np.zeros([n_samples, 512])
    w_save = np.zeros([n_samples, 512])
    for i in tqdm(range(n_samples)):
        z_np = np.random.randn(1, stylegan.z_dim)
        z_tensor = torch.from_numpy(z_np).to(device)
        label = torch.zeros([1, stylegan.c_dim], device=device)
        ws_tensor = stylegan.mapping(z_tensor, label, truncation_psi)
        ws_np = ws_tensor.cpu().numpy()[:, 0, :]
        z_save[i] = z_np[0]
        w_save[i] = ws_np[0]
        img_out = stylegan.synthesis(ws_tensor, noise_mode="const")
        img_out = (img_out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
        Image.fromarray(np.uint8(img_out)).save(gen_path + str(i) + ".jpg")
    np.save(np_save_path + "z_space", z_save)
    np.save(np_save_path + "w_space", w_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200000, help="generated images counts")
    parser.add_argument("--stylegan", type=str, default="stylegan3-t-ffhq-1024x1024.pkl", help="stylegan model path")
    parser.add_argument("--gen_path", type=str, default="resources/interfacegan/generated_imgs/", help="save the generated images from stylegan")
    parser.add_argument("--np_save_path", type=str, default="resources/interfacegan/", help="save the numpy format z and w space feature vectors")
    parser.add_argument("--device", type=str, default="cuda",)

    args = parser.parse_args()

    generate_data(args.n_samples, gen_path=args.gen_path, np_save_path=args.np_save_path, stylegan_model=args.stylegan, device=args.device)
