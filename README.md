# PrivateSNN


Pytorch Implementation for [PrivateSNN: Privacy-Preserving Spiking Neural Networks].

Accepted in AAAI2022.


## Prerequisites
* Python 3.8    
* PyTorch 1.5.0     
* NVIDIA GPU (>= 12GB)      

## Getting Started

## Training and testing



### [STEP 1] Training ANN (ref: https://github.com/nitin-rathi/hybrid-snn-conversion)
Trainig ANN architecture with Backpropagation.

```
python ann.py --dataset 'CIFAR10' -a 'VGG16' --optimizer 'SGD' --batch_size 256 -lr 1e-2 --lr_interval '0.40 0.60 0.80'
```


### [STEP 2] Generate synthetic training samples
Generating synthetic images from pre-trained ANN model.

```
python ann_image_gen.py --dataset 'CIFAR10' --num_synimage 1000
```

### [STEP 3] Datafree Conversion
Converting pre-trained ANN model to SNN model with synthetic dataset (from STEP2).
```
python datafree_conversion.py -a 'VGG16' --dataset 'CIFAR10' --timesteps 100 --batch_size 1024 
```




## Citation
 
Please consider citing our paper:
 ```
@article{kim2021privatesnn,
  title={Privatesnn: Privacy-preserving spiking neural networks},
  author={Kim, Youngeun and Venkatesha, Yeshwanth and Panda, Priyadarshini},
  journal={arXiv preprint arXiv:2104.03414},
  year={2021}
}
 ```
 
 
 
