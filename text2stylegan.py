import clip 
from PIL import Image 
import torch 
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F
import dnnlib
import legacy
import numpy as np 
import argparse 


def clip2stylegan(text, opt_space, learning_rate, stylegan_model, truncation_psi=0.5, device="cuda"):
    with dnnlib.util.open_url(stylegan_model) as f:
        stylegan = legacy.load_network_pkl(f)['G_ema'].to("cuda") # type: ignore
    device ="cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    model = model.to(device)
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    
    label = torch.zeros([1, stylegan.c_dim], device=device)
    if opt_space == "z":
        z_np = np.random.randn(1, stylegan.z_dim)
        z_tensor = torch.from_numpy(z_np).to(device)
        z_tensor = nn.Parameter(z_tensor.detach())
        opt = optim.Adam([z_tensor], lr=learning_rate)
    else:
        z_np = np.random.randn(1, stylegan.z_dim)
        z_tensor = torch.from_numpy(z_np).to(device)
        ws = stylegan.mapping(z_tensor, label, truncation_psi)
        ws = ws.detach()[:, 0:1, :]
        ws = nn.Parameter(ws)
        opt = optim.Adam([ws], lr=learning_rate)

    mu = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1).to(device)
    sigma = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1).to(device)
    imgs = []
    for i in range(501):
        if opt_space == "z":
            fake_img = stylegan(z_tensor, label, truncation_psi=truncation_psi)
        else:
            ws_plus = torch.repeat_interleave(ws, stylegan.num_ws, 1)
            fake_img = stylegan.synthesis(ws_plus, noise_mode="const")
        img_out = (fake_img + 1) / 2
        img_out = F.interpolate(img_out, [224, 224], mode="bilinear")
        img_out = (img_out - mu) / sigma
        image_features = model.encode_image(img_out)
        loss = (1 - torch.cosine_similarity(image_features, text_features)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 100 == 0:
            imgs.append(fake_img.clamp(-1, 1).permute(0, 2, 3, 1).detach().cpu().numpy()[0])
            Image.fromarray(np.uint8((fake_img.clamp(-1, 1).permute(0, 2, 3, 1).detach().cpu().numpy()[0] + 1)*127.5)).save("clip2stylegan.jpg")
            print(f"Iteration: {str(i)}, Loss: {loss.item()}")
    imgs = np.concatenate(imgs, axis=1)
    Image.fromarray(np.uint8((imgs + 1)*127.5)).save("clip2stylegan.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--opt_space", type=str, default="w", help="z | w")
    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--stylegan_model", type=str, default="stylegan3-t-ffhq-1024x1024.pkl") 
    parser.add_argument("--truncation_psi", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    clip2stylegan(args.text, args.opt_space, args.learning_rate, args.stylegan_model, args.truncation_psi, args.device)
    # python text2stylegan.py --text "a woman with blue eyes" --opt_space w --learning_rate 0.02
