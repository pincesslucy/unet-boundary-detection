# Unet boundary detection  
Unet을 이용해 이미지 boundary detection을 하는 학습 코드입니다.

## Dataset
Berkeley Segmentation Dataset 500 (BSDS500)  
https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500
 
## Model  
UNET  
![image](https://github.com/pincesslucy/unet-boundary-detection/assets/98650288/f8a74370-332b-4f41-b175-8ff06ad42644)

## Training  
batch size: 4  
loss function: BCEwithLogitsLoss  
optimizer: Adam  
num_epochs: 20  

## Usage  
python version == 3.10
```bash
pip install requirements.txt
python train.py
```
## Results  
### input  
![apple](https://github.com/pincesslucy/unet-boundary-detection/assets/98650288/28be0d27-1915-41da-a658-a3cc4d4d4db6)
![image](https://github.com/pincesslucy/unet-boundary-detection/assets/98650288/38fa30ac-293d-4d93-9142-242066b69700)
### output  
![apple_output](https://github.com/pincesslucy/unet-boundary-detection/assets/98650288/c5673fbd-7ad4-4b01-8ea0-5feac132af1a)
![image_output](https://github.com/pincesslucy/unet-boundary-detection/assets/98650288/7cd149c5-099c-48a3-9b54-6a1568e71313)
### loss
![image](https://github.com/pincesslucy/unet-boundary-detection/assets/98650288/9ffad2d1-2b03-4b0c-ba66-c6e172fc27d6)
