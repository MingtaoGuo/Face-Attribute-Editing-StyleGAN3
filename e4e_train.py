from pspEncoder import pspEncoder
from Discriminator import Discriminator
from arcface import iresnet50
from lpips.lpips import LPIPS
from Dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import torch
import time
import torchvision
import cv2
import dnnlib
import numpy as np
import legacy
import argparse


def get_grid_image(X):
    X = X[:8]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]) * 0.5 + 0.5
    return X


def make_image(Xs, Xt, Y):
    Xs = get_grid_image(Xs)
    Xt = get_grid_image(Xt)
    Y = get_grid_image(Y)
    return torch.cat((Xs, Xt, Y), dim=1).numpy()

def train(G, D, Encoder, arcface, lpips_loss, opt_D, opt_G, dataloader, args):
    f = open("loss.txt", "w")
    Encoder.progress_stage = 0
    total_itr = 0
    for epoch in range(args.epoch):
        for iteration, data in enumerate(dataloader):
            start_time = time.time()
            low, high = data
            low = low.to(args.device)
            high = high.to(args.device)
            if total_itr >= 20000 and (total_itr - 20000) % 2000 == 0 and Encoder.progress_stage < G.num_ws - 1:
                Encoder.progress_stage += 1
            # train D
            opt_D.zero_grad()
            wplus = Encoder(low)
            z = torch.from_numpy(np.random.randn(args.batch_size, G.z_dim)).to(args.device)
            label = torch.zeros([1, G.c_dim], device=args.device)
            ws = G.mapping(z, label, truncation_psi=1.0)
            fake_pred = D(wplus.detach()[:, 0, :])
            real_pred = D(ws.detach()[:, 0, :])
            real_loss = F.softplus(-real_pred).mean()
            fake_loss = F.softplus(fake_pred).mean()
            D_loss = real_loss + fake_loss
            D_loss.backward()
            opt_D.step()
            # train Encoder
            opt_G.zero_grad()
            fake_pred = D(wplus[:, 0, :])
            L_adv = F.softplus(-fake_pred).mean()

            fake = G.synthesis(wplus, noise_mode="const")
            fake_z_id = arcface(F.interpolate(fake, [143, 143], mode="nearest")[..., 15:127, 15:127])
            with torch.no_grad():
                z_id = arcface(F.interpolate(high, [143, 143], mode="nearest")[..., 15:127, 15:127])
            L_sim = (1 - torch.cosine_similarity(z_id, fake_z_id)).mean()
            L_lpips = lpips_loss(fake, high)
            L_pix = F.mse_loss(fake, high)
            L_reg = torch.tensor(0.).cuda()
            first_w = wplus[:, 0, :]
            for j in range(1, Encoder.progress_stage + 1):
                delta_j = wplus[:, j, :] - first_w
                L_reg += torch.norm(delta_j, 2, dim=1).mean()
            L_dist = L_pix * args.lambda_pix + L_lpips * args.lambda_lpips + L_sim * args.lambda_sim
            L_edit = L_adv * args.lambda_adv + L_reg * args.lambda_reg
            G_loss = L_dist + L_edit
            G_loss.backward()
            opt_G.step()

            batch_time = time.time() - start_time
            if total_itr % 100 == 0:
                image = make_image(F.interpolate(low, [args.resolution, args.resolution]), fake, high)
                cv2.imwrite('./results/'+str(epoch)+"_"+str(iteration)+".jpg", image.transpose([1,2,0])[..., ::-1]*255)
                print(f'epoch: {epoch}    {iteration} / {len(dataloader)}')
                print(f'Iteration: {total_itr} D_loss: {D_loss.item():.4f} G_loss: {G_loss.item():.4f} L_adv: {L_adv.item():.4f} L_pix: {L_pix.item():.4f} L_lpips: {L_lpips.item():.4f} L_sim: {L_sim.item():.4f} L_reg: {L_reg.item():.4f} batch_time: {batch_time:.4f}s')
                f.write(f'Iteration: {total_itr} D_loss: {D_loss.item():.4f} G_loss: {G_loss.item():.4f} L_adv: {L_adv.item():.4f} L_pix: {L_pix.item():.4f} L_lpips: {L_lpips.item():.4f} L_sim: {L_sim.item():.4f} L_reg: {L_reg.item():.4f} batch_time: {batch_time:.4f}s\n')
                f.flush()
            if total_itr % 10000 == 0:
                torch.save(Encoder.state_dict(), './saved_models/'+str(epoch)+'_'+str(iteration)+'_'+'pspEncoder.pth')
            total_itr += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data1/GMT/Dataset/FFHQ1024/")
    parser.add_argument("--resolution", type=int, default=1024, help="512 | 1024")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--arcface", type=str, default="backbone.pth")
    parser.add_argument("--epoch", type=int, default=20)

    parser.add_argument("--lambda_pix", type=float, default=1.0)
    parser.add_argument("--lambda_lpips", type=float, default=0.8)
    parser.add_argument("--lambda_sim", type=float, default=0.1)
    parser.add_argument("--lambda_adv", type=float, default=0.1)
    parser.add_argument("--lambda_reg", type=float, default=2e-4)

    parser.add_argument("--lr_D", type=float, default=2e-5)
    parser.add_argument("--lr_G", type=float, default=1e-4)

    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    network_pkl = "stylegan3-t-ffhq-1024x1024.pkl"
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(args.device) # type: ignore

    Encoder = pspEncoder(n_style=G.num_ws, pretrain=args.arcface)
    Encoder.to(args.device)
    Encoder.train()

    D = Discriminator()
    D.to(args.device)
    D.train()

    arcface = iresnet50().to(args.device)
    arcface.eval()
    arcface.load_state_dict(torch.load(args.arcface))

    lpips_loss = LPIPS(net_type="alex").to(args.device).eval()

    opt_D = optim.Adam(D.parameters(), lr=args.lr_D)
    opt_G = optim.Adam(Encoder.parameters(), lr=args.lr_G)

    dataset = Dataset(args.data_path, args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    
    train(G, D, Encoder, arcface, lpips_loss, opt_D, opt_G, dataloader, args)
    