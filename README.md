## Getting Started
### 1. Dataset Download & Pre-processing
We evaluate the proposed method on five widely used change detection benchmarks:
- LEVIR-CD: https://justchenhao.github.io/LEVIR
- WHU-CD: http://gpcv.whu.edu.cn/data/building_dataset.html
- SYSU-CD: https://github.com/liumency/SYSU-CD
- LEVIR-CD+: https://github.com/S2Looking/Dataset
- CLCD-CD: https://github.com/liumency/CropLand-CD

Above five preprocessed datasets are available at [Baidu Disk](https://pan.baidu.com/s/1XtL511v12h27PtN7O3D_pQ?pwd=0000) (Password: 0000) 
\
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
Training logs and pretrained weights of the compared methods are available at: [Baidu Disk](https://pan.baidu.com/s/17tHry4XBErbrk-XURdWRNA?pwd=w6c6) (Password: w6c6)


