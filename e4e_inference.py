from pspEncoder import pspEncoder
from PIL import Image 
import torch
import dnnlib
import numpy as np
import legacy
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="1.jpg")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--pretrain", type=str, default="saved_models/10_5000_pspEncoder.pth")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    network_pkl = "stylegan3-t-ffhq-1024x1024.pkl"
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(args.device) # type: ignore

    Encoder = pspEncoder(n_style=G.num_ws)
    Encoder.to(args.device)
    Encoder.eval()
    Encoder.load_state_dict(torch.load(args.pretrain))

    inputs = np.array(Image.open(args.img_path).resize([args.input_size, args.input_size]))
    inputs = inputs / 127.5 - 1.0
    inputs = torch.tensor(inputs, dtype=torch.float32).cuda()[None].permute(0, 3, 1, 2)
    with torch.no_grad():
        wplus = Encoder(inputs)
        fake_img = G.synthesis(wplus, noise_mode="const")
        fake_img = fake_img.detach().clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy()[0]
        fake_img = (fake_img + 1.0) * 127.5
        Image.fromarray(np.uint8(fake_img)).save("e4e_inverse.jpg")
    