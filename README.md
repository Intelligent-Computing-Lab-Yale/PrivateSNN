# PrivateSNN


Pytorch Implementation for [PrivateSNN: Privacy-Preserving Spiking Neural Networks]

Code will be released soon.


## Training and testing


### [STEP 1] Training ANN

'''
python ann.py --dataset 'CIFAR10' -a 'VGG16' --optimizer 'SGD' --batch_size 256 -lr 1e-2 --lr_interval '0.40 0.60 0.80'
'''

