# Dual-Path AI-Generated Image Detection: Leveraging Texture-Rich and Texture-Poor Patches with Global Semantic Features
<b> Jiapeng Lin,Qihao Zhou,Jie Li </b>


## Abstract
> The rapid advancement of image generation models, such as Generative Adversarial Networks (GANs) and diffusion models, has led to AI-generated images that are often indistinguishable from authentic photographs. This development poses significant challenges in verifying the authenticity of visual content, necessitating robust detectors capable of identifying AI-generated images. While numerous detection techniques have been proposed, most suffer from limited generalization due to their reliance on either domain-specific artifacts or global semantic features alone. In this paper, we address these limitations by introducing a novel detection framework centered on extracting local artifact features from both texture-rich and texture-poor regions. Our approach is motivated by the observation that during the image synthesis process, regions with poor textures exhibit lower reconstruction losses, indicating varying generation difficulties across different areas. By strategically selecting patches with diverse texture complexities, we capture critical fine-grained artifacts that enhance the detector’s sensitivity to AI-specific patterns. To further improve detection performance, our framework integrates extracted global semantic features  using CLIP-ViT from the entire image, forming a dual-path architecture that combines both local and global information. Within this framework, an attention-based Feature Fusion Module is introduced to dynamically model the relationships between local patches and global semantics, enabling robust and generalizable classification. Extensive experiments conducted on two benchmark datasets—GenImage and UniversalFakeDetect—demonstrate that our method achieves state-of-the-art performance in ID accuracy, OOD accuracy, and average precision metrics, highlighting its strong cross-generator generalization ability.
<p align="center">
<img src=" width=60%>
</p>

## pipeline
![Image Description](https://github.com/ljppp117/Dual-Path-AI-Generated-Image-Detection/blob/main/pipiline.PNG?raw=true)



## Requirements
```
conda create -n clipfordetection python=3.9
conda activate clipfordetection
pip install torch==1.12.0+cu113  torchvision==0.13.0+cu113 -f 
pip install -r requirements.txt

```
## Genimage Dataset and UFD Dataset
The Genimage Dataset can be downloaded from [here](https://pan.baidu.com/share/init?surl=i0OFqYN5i6oFAxeK6bIwRQ). The code is ztf1. 

Specifically, you can download the simplified version of the Genimage Dataset from  [here](), which only includes the SD1.4 for training and the other eight val datasets.

The UFD Dataset can be downloaded from [here](https://github.com/WisconsinAIVision/UniversalFakeDetect). 


The dataset is organized as follows:
```
images
└── train/val
    ├── ADM
    │   ├── 0_real
    │   │   └──img1.png...
    │   ├── 1_fake
    │   │   └──img1.png...
    ├── biggan
    │   ├── 0_real
    │   │   └──img1.png...
    │   ├── 1_fake
    │   │   └──img1.png...
    ├── VQDM
    │   ├── 0_real
    │   │   └──img1.png...
    │   ├── 1_fake
    │   │   └──img1.png...
    │   ...
 

```
## Training
Before training, you should link the training real and fake images to the `/train` folder. You need to modify the correct paths for `train_dataset` and `test_dataset` in `train.py`.
```
python train.py
```
## Evaluation
We provide the pre-trained model in [here]().
Before evaluating, you should link the training real and fake images to the `/train` folder. You need to modify the correct paths for `test_dataset` in `test.py`.You can evaluate the model by running the following command:
```
python test.py
```

## Acknowledgments
Our code is developed based on [AIDE](https://github.com/shilinyan99/AIDE) and [CNNDetection](https://github.com/peterwang512/CNNDetection). Thanks for their sharing codes and models.


