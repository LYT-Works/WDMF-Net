# <p align=center> A Lightweight Wavelet-Aligned Difference and Mask-Guided Fusion Network for Change Detection <p>
> Yuting Liu<sup>1</sup>, Shihua Li<sup>1</sup>, Lorenzo Bruzzone<sup>2</sup>
\
> <sup>1</sup>University of Electronic Science and Technology of China
\
> <sup>2</sup>University of Trento

📄Paper at arXiv:


## Abstract
<p align="justify">
Remote sensing change detection (CD) plays an important role in urban monitoring, resource management, and disaster assessment. To meet the practical demand for both high accuracy and efficiency, this article proposes a lightweight wavelet-aligned difference and mask-guided feature fusion network (WDMF-Net), which optimizes the CD process from three aspects: temporal alignment, difference modeling, and change and non-change decoupling. WDMF-Net adopts MobileNetV2 as the Siamese backbone and incorporates an improved adjacent-level aggregation strategy to exploit the  complementary information. A wavelet-enhanced feature alignment module (WFAM) is introduced to enhance local boundary details and global structural components in the frequency domain, while spatial alignment is further performed to alleviate bitemporal misregistration. In addition, a collaborative feature difference module (CFDM) is designed to model the complementary relationship between channel-wise and spatial-wise features, enabling effective capture of local and global differences. Furthermore, a mask-guided feature enhancement module (MFEM) is employed, where mask priors are progressively involved in feature fusion to explicitly decouple changed and unchanged regions, enhancing change separability and discriminability. Experiments demonstrate that WDMF-Net outperforms 18 state-of-the-art approaches on five public CD datasets, and achieves a favorable balance among detection accuracy, computational efficiency, and generalization capability.<p>
  
## Overall Architecture 
<p align="center">
  <img src="images/Overall Architecture.png">
</p>

## Getting Started
### 1. Dataset Download & Pre-processing
We evaluate the proposed method on five widely used change detection benchmarks:
- LEVIR-CD: https://justchenhao.github.io/LEVIR
- WHU-CD: official version at http://gpcv.whu.edu.cn/data/building_dataset.html, pre-processed version at [Google Drive](https://drive.google.com/file/d/1c93Y0ioe16rxEkVIJJyUpJdKkMmPOTxR/view) 
- SYSU-CD: https://github.com/liumency/SYSU-CD
- LEVIR-CD+: https://github.com/S2Looking/Dataset
- CLCD-CD: https://github.com/liumency/CropLand-CD

Please ensure that each image in the dataset is cropped to patches of 256×256 pixels.
### 2. Dataset Organization
Prepare the dataset into the following structure and set its path in the [config](https://github.com/LYT-Works/WDMF-Net/blob/main/train.py#L181-L190) file.
 
    ├─Train
        ├─A        ...jpg/png
        ├─B        ...jpg/png
        ├─label    ...jpg/png
        └─list     ...txt
    ├─Val
        ├─A
        ├─B
        ├─label
        └─list
    ├─Test
        ├─A
        ├─B
        ├─label
        └─list
Generate list files using:
`ls -R ./label/* > test.txt`
### 3. Environment Setup
Create a virtual environment and install all dependencies:
`pip install -r requirements.txt`
### 4. Training & Testing
Training:
`bash train.sh`
\
Testing:
`bash test.sh`
\
Pretrained weights are provided and can be used directly for evaluation.
## Main Results
The proposed method (WDMF-Net) achieves state-of-the-art performance on five public remote sensing change detection benchmarks, demonstrating strong robustness and generalization capability. Training logs and pretrained weights of the compared methods are available at: [Baidu Disk](https://pan.baidu.com/s/17tHry4XBErbrk-XURdWRNA?pwd=w6c6) (Password: w6c6)
<p align="center">
  <img src="images/Main Results.png">
</p>

## Acknowledgement
This repository is built upon [A2Net](https://github.com/guanyuezhen/A2Net) and [ChangeViT](https://github.com/zhuduowang/ChangeViT).
We sincerely thank the authors for their well-organized and open-sourced codebases.



