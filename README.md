# PrivateSNN


Pytorch Implementation for [PrivateSNN: Privacy-Preserving Spiking Neural Networks]


## Prerequisites
* Python 3.9    
* PyTorch 1.10.0     
* NVIDIA GPU (>= 12GB)      
* CUDA 10.2 (optional)         

## Getting Started

### Conda Environment Setting
```
conda create -n PrivateSNN 
conda activate PrivateSNN
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

## Training and testing



### [STEP 1] Training ANN (ref: https://github.com/nitin-rathi/hybrid-snn-conversion)


```
python ann.py --dataset 'CIFAR10' -a 'VGG16' --optimizer 'SGD' --batch_size 256 -lr 1e-2 --lr_interval '0.40 0.60 0.80'
```


### [STEP 2] Generate synthetic training samples

