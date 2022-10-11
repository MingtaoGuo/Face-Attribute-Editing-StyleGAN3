import dnnlib
import numpy as np
import torch
from sklearn.decomposition import PCA
from PIL import Image 
from tqdm import tqdm 
import legacy
import argparse 


class GANSpace:
    def __init__(self, network_pkl, n_sample, n_components, is_gpu_free, device) -> None:
        print('Loading networks from "%s"...' % network_pkl)
        self.truncation_psi = 1.0
        self.device = device
        self.n_components = n_components
        with dnnlib.util.open_url(network_pkl) as f:
            self.stylegan = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        if is_gpu_free:
            z = torch.from_numpy(np.random.randn(n_sample, self.stylegan.z_dim)).to(device)
            label = torch.zeros([n_sample, self.stylegan.c_dim], device=device)
            ws_space = self.stylegan.mapping(z, label, self.truncation_psi) # 1 x 16 x 512
            ws_space = ws_space.cpu().numpy()[:, 0, :]
        else:
            # GPU memory constraint
            ws_space = np.zeros([n_sample, 512])
            for i in tqdm(range(n_sample)):
                with torch.no_grad():
                    z = torch.from_numpy(np.random.randn(1, self.stylegan.z_dim)).to(device)
                    label = torch.zeros([1, self.stylegan.c_dim], device=device)
                    ws = self.stylegan.mapping(z, label, self.truncation_psi) # 1 x 16 x 512
                    ws_space[i, :] = ws.cpu().numpy()[0, 0]
        self.pca = PCA(n_components=n_components)
        self.pca = self.pca.fit(ws_space)
        self.basis = self.pca.components_  # n_components x 512
        self.mean_tensor = torch.tensor(self.pca.mean_, dtype=torch.float32).to(device)
        self.basis_tensor = torch.tensor(self.pca.components_, dtype=torch.float32).to(device)

    def layer_wise_edit(self, w, v_idxs, layer_idxs):
        x = torch.zeros([1, self.n_components]).to(self.device)
        for v_idx in v_idxs:
            x[0, v_idx] = 1
        imgs = []
        for i in tqdm(range(9)):
            alpha = (i - 5) * 2
            w_ = torch.matmul(x, self.basis_tensor) * alpha + self.mean_tensor + w[:, 0, :]
            w_copy = w * 1.0
            for layer_idx in layer_idxs:
                w_copy[:, layer_idx] = w_
            img_out = self.stylegan.synthesis(w_copy, noise_mode="const")
            img_out = (img_out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
            imgs.append(img_out)
        imgs = np.concatenate(imgs, axis=1)
        return imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stylegan", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="ganspace.png")
    parser.add_argument("--v_idxs", type=int, default=0)
    parser.add_argument("--layer_idxs", type=str, default="0-18")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_components", type=int, default=256)
    parser.add_argument("--is_gpu_free", type=bool, default=False, help="If gpu memory is constrait set False")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    ganspace = GANSpace(args.stylegan, args.n_samples, args.n_components, is_gpu_free=args.is_gpu_free, device=args.device)

    z = torch.from_numpy(np.random.randn(1, ganspace.stylegan.z_dim)).to(ganspace.device)
    label = torch.zeros([1, ganspace.stylegan.c_dim], device=ganspace.device)
    w = ganspace.stylegan.mapping(z, label, ganspace.truncation_psi) # 1 x 16 x 512
    v_idxs = [args.v_idxs]
    start_l = max(int(args.layer_idxs.split("-")[0]), 0)
    end_l = min(int(args.layer_idxs.split("-")[1]), ganspace.stylegan.num_ws)
    layer_idxs = [i for i in range(start_l, end_l)]
    result = ganspace.layer_wise_edit(w, v_idxs=v_idxs, layer_idxs=layer_idxs)
    Image.fromarray(np.uint8(result)).save(args.save_path)