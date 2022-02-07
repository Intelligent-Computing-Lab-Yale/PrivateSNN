import torch
import torch.nn as nn


cfg = {
    'VGG5' : [64, 'A', 128, 128, 'A'],
    'VGG9':  [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A'],
    'VGG11': [64, 'A', 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512, 'A'],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512,'A']
}


class VGG(nn.Module):
    def __init__(self, vgg_name='VGG16', labels=10, dataset = 'CIFAR10', kernel_size=3, dropout=0.2, bias = False):
        super(VGG, self).__init__()
        
        self.dataset        = dataset
        self.kernel_size    = kernel_size
        self.dropout        = dropout
        self.bias = bias

        self.features       = self._make_layers(cfg[vgg_name])
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        if vgg_name == 'VGG5' and dataset == 'MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(128*7*7, 4096, bias=self.bias ),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=self.bias ),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=self.bias )
                            )
        elif vgg_name == 'VGG16' and dataset== 'CIFAR10':
            print(vgg_name, dataset)
            self.classifier = nn.Sequential(
                nn.Linear(512*1*1, 4096, bias=self.bias),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 10, bias=self.bias)
            )
        elif vgg_name == 'VGG16' and dataset== 'CIFAR100':
            print(vgg_name, dataset)
            self.classifier = nn.Sequential(
                nn.Linear(512*1*1, 4096, bias=self.bias),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 100, bias=self.bias)
            )
        elif vgg_name == 'VGG16' and dataset== 'Tinyimagenet':
            print(vgg_name, dataset)
            self.classifier = nn.Sequential(nn.Linear(512*2*2, 4096, bias=self.bias), nn.ReLU(), nn.Dropout2d(p=0.5),
                                            # nn.Linear(4096, 4096, bias=self.bias), nn.ReLU(), nn.Dropout2d(p=0.5),
                                            nn.Linear(4096, 200, bias=self.bias)
                                            )
        elif vgg_name == 'VGG16' and dataset== 'ImageNet':
            print(vgg_name, dataset)
            self.imgnet_pool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(nn.Linear(512*7*7, 4096, bias=self.bias), nn.ReLU(), nn.Dropout2d(p=0.5),
                                            nn.Linear(4096, 4096, bias=self.bias), nn.ReLU(), nn.Dropout2d(p=0.5),
                                            nn.Linear(4096, 1000, bias=self.bias)
                                            )

        self._initialize_weights2()
        

    def forward(self, x):
        out = self.features(x)

        if self.dataset== 'ImageNet':
            out = self.imgnet_pool(out)
        # print (out.size())
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _initialize_weights2(self):
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def _make_layers(self, cfg):
        layers = []

        if self.dataset == 'MNIST':
            in_channels = 1
        else:
            in_channels = 3
        
        for x in cfg:
            stride = 1
            if x == 'A':
                layers.pop()
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2,
                                         stride=stride, bias=self.bias ),nn.ReLU(inplace=True)]

                layers += [nn.Dropout(self.dropout)]
                in_channels = x

        
        return nn.Sequential(*layers)

