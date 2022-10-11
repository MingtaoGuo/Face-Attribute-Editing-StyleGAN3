import torch
import clip
from PIL import Image
import numpy as np 
from tqdm import tqdm 
import argparse


def predict(attr_text, path):
    device ="cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    model = model.to(device)
    text = clip.tokenize(["a face don't have " + attr_text, "a face haves " + attr_text]).to(device)

    f = open(path + attr_text + "_label.txt", "w")
    with torch.no_grad():
        for i in tqdm(range(200000)):
            # normalized features
            file = str(i) + ".jpg"
            image = preprocess(Image.open(path + "/generated_imgs/" + file)).unsqueeze(0).to(device)
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            f.write(file + " " + str(np.argmax(probs, axis=1)[0]) + " " + str(np.max(probs[0])) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attr_text", type=str, required=True, help="glasses | smile | male | young | old | etc.")
    parser.add_argument("--gen_data_path", type=str, default="resources/interfacegan/")

    args = parser.parse_args()

    predict(args.attr_text, args.gen_data_path)