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
|Description|an european woman with blue eyes|an old asian man with gray hair|an asian woman with black long straight hair| a woman with blond hair is smiling|
|-|-|-|-|-|
|**Text2img**|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/IMGS/clip2stylegan1.jpg)|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/IMGS/clip2stylegan2.jpg)|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/IMGS/clip2stylegan3.jpg)|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/IMGS/clip2stylegan4.jpg)|

### Inference InterfaceGAN
``` 
python interfacegan_edit.py --stylegan stylegan3-t-ffhq-1024x1024.pkl --boundary resources/interfacegan/boundary_glasses.pth
```
|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/IMGS/interfacegan_input1.jpg)|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/IMGS/interfacegan_input2.jpg)|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/IMGS/interfacegan_input3.jpg)|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/IMGS/interfacegan_input4.jpg)|
|-|-|-|-|
|**--------glasses---------**|**---------beard---------**|**---------young---------**|**---------smile---------**|
|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/IMGS/interfacegan_edit1.jpg)|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/IMGS/interfacegan_edit2.jpg)|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/IMGS/interfacegan_edit3.jpg)|![](https://github.com/MingtaoGuo/Face-Attribute-Editing-StyleGAN3/blob/main/IMGS/interfacegan_edit4.jpg)|

### Inference GANSpace
``` 
python ganspace.py --stylegan stylegan3-t-ffhq-1024x1024.pkl --v_idxs 0 --layer_idxs 0-18 --n_samples 10000
```
|v_0, layer_1_18 (gender)|
|-|
|![](https://github.com/MingtaoGuo/DDPM_pytorch/blob/main/resources/v_0_l_all_.jpg)|

|v_1, layer_1_18 (rotate)|
|-|
|![](https://github.com/MingtaoGuo/DDPM_pytorch/blob/main/resources/v_1_l_all_.jpg)|

|v_2, layer_1_18 (rotate + age+ bkg)|
|-|
|![](https://github.com/MingtaoGuo/DDPM_pytorch/blob/main/resources/v_2_l_all_.jpg)|

## Author 
Mingtao Guo
E-mail: gmt798714378@hotmail.com

## Reference
[1]. Tov, Omer, et al. "Designing an encoder for stylegan image manipulation." ACM Transactions on Graphics (TOG) 40.4 (2021): 1-14.

[2]. Shen, Yujun, et al. "Interfacegan: Interpreting the disentangled face representation learned by gans." IEEE transactions on pattern analysis and machine intelligence (2020).

[3]. Härkönen, Erik, et al. "Ganspace: Discovering interpretable gan controls." Advances in Neural Information Processing Systems 33 (2020): 9841-9850.
