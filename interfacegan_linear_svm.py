import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset import Dataset_svm


def validation(model, val_data, val_label):
    model.eval()
    val_data = torch.tensor(val_data, dtype=torch.float32).cuda()
    ys = []
    for y in val_label:
        y = -1 if int(y.strip().split()[1]) == 0 else 1
        ys.append(y)
    val_label = torch.tensor(np.array(ys), dtype=torch.float32).cuda()
    pred = model(val_data)
    pred[pred>0] = 1
    pred[pred<=0] = -1
    acc = (pred == val_label).float().mean()
    model.train()
    return acc 

def train(val_data, val_label, model, opt, dataloader, C, epoch, boundary_save, device):
    max_val_acc = -1
    for epoch in range(epoch):
        for i, data in enumerate(dataloader):
            x, y = data[0].to(device), data[1].to(device)

            opt.zero_grad()
            pred = model(x).squeeze()
            weight = model.weight.squeeze()

            loss = torch.mean(torch.clamp(1 - y * pred, min=0))
            loss += C * (weight.t() @ weight) / 2.0

            loss.backward()
            opt.step()
            if i % 100 == 0:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item():.4f}")
        acc = validation(model, val_data, val_label)
        if acc.item() > max_val_acc:
            max_val_acc = acc.item()
            torch.save(model.state_dict(), boundary_save)
        print(f"Epoch: {epoch}, Iteration: {i}, Val accuracy: {acc.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_npy", type=str, default="resources/interfacegan/w_space.npy")
    parser.add_argument("--path_txt", type=str, default="resources/interfacegan/young_label.txt")
    parser.add_argument("--boundary_save", type=str, default="resources/interfacegan/boundary_young.pth")
    parser.add_argument("--c", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data = np.load(args.path_npy)
    label = open(args.path_txt).readlines()

    train_data, train_label = data[:190000], label[:190000]
    val_data, val_label = data[190000:], label[190000:]
    dataset = Dataset_svm(train_data, train_label)
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)

    model = nn.Linear(512, 1, bias=False)
    model.to(args.device)
    model.train()

    opt = optim.Adam(model.parameters(), lr=args.lr)

    train(val_data, val_label, model, opt, dataloader, args.c, args.epoch, args.boundary_save, args.device)
