# Face-Attribute-Editing-StyleGAN3
Face-Attribute-Editing-Encoder4Editing-Text2StyleGAN-InterfaceGAN-GANSpace-Based-on-StyleGAN3

## Description   
--------------

This repo is mainly to re-implement the follow face-editing papers based on stylegan
- Encoder4Editing: [Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766)
- InterfaceGAN: [InterFaceGAN: Interpreting the Disentangled Face Representation Learned by GANs](https://arxiv.org/abs/2005.09635)
- GANSpace: [GANSpace: Discovering Interpretable GAN Controls](https://arxiv.org/abs/2004.02546)

## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN
- Python 3

### Installation
- Clone the repository:
``` 
git clone https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3.git
cd Face-Attribute-Editing-StyleGAN3
wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl
```
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment.yaml`.

### Inference e4e
- Download the e4e pretrained model [FFHQ-e4e-stylegan3](https://drive.google.com/file/d/11OKcGJniqmvf_J6Mym_erqCy5Mm1wfmO/view?usp=sharing)
``` 
python e4e_inference.py --img_path resources/imgs/1.png --pretrain saved_models/13_12500_pspEncoder.pth
```
|Image|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/resources/imgs/1.png)|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/resources/imgs/2.png)|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/resources/imgs/3.png)|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/resources/imgs/4.png)|
|-|-|-|-|-|
|**Inverse**|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/IMGS/inverse1.jpg)|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/IMGS/inverse2.jpg)|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/IMGS/inverse3.jpg)|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/IMGS/inverse4.jpg)|

### Train e4e
- Download the FFHQ dataset from [NVlabs](https://github.com/NVlabs/ffhq-dataset)
- Download the face recognition model arcface from [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) Go to  Baidu Drive **->** arcface_torch **->** glint360k_cosface_r50_fp16_0.1 **->** backbone.pth
``` 
python e4e_train.py --data_path FFHQ --batch_size 4 --epoch 20 --arcface saved_models/backbone.pth
```
### Inference Text2StyleGAN
``` 
python text2stylegan.py --text "a woman with blue eyes" --opt_space w --learning_rate 0.02
```
|Description|a woman with blue eyes|an old asian man with gray hair|an asian woman with black long straight hair||
|-|-|-|-|-|
|Generated|||||

# To be continued
## Author 
Mingtao Guo
E-mail: gmt798714378@hotmail.com

