#---------------------------------------------------
# Imports - this file is for data-free conversion
# ref: https://github.com/nitin-rathi/hybrid-snn-conversion
#---------------------------------------------------
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import DataLoader
import numpy as np
import datetime
from self_models import  vgg_spiking
import sys
import argparse
from syndataloader import *


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def find_threshold(batch_size=512, timesteps=2500, architecture='VGG16'):


    small_batch_size = 1024
    batch_iter_stop = batch_size // small_batch_size

    syntrain_loader    = DataLoader(syn_trainset, batch_size=small_batch_size, shuffle=True)

    model.module.network_update(timesteps=timesteps, leak=1.0)

    pos=0
    thresholds=[]
    def find(layer, pos):
        max_act=0
        with torch.no_grad():

            f.write('\n Finding threshold for layer {}'.format(layer))
            for batch_idx, (data, target) in enumerate(syntrain_loader):
                if torch.cuda.is_available() and args.gpu:
                    data, target = data.cuda(), target.cuda()

                    model.eval()
                    output = model(data, find_max_mem=True, max_mem_layer=layer)

                    if output>max_act:
                        max_act = output.item()

                    if batch_idx==batch_iter_stop:
                        thresholds.append(max_act)
                        pos = pos+1

                        f.write(' {}'.format(thresholds))
                        model.module.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
                        break
            return pos

    if architecture.lower().startswith('vgg'):


        for l in model.module.features.named_children(): # l = (no.layer, type)
            if isinstance(l[1], nn.Conv2d):
                pos = find(int(l[0]), pos)


        for c in model.module.classifier.named_children():
            if isinstance(c[1], nn.Linear):
                if (int(l[0])+int(c[0])+1) == (len(model.module.features) + len(model.module.classifier) -1):
                    pass
                else:
                    pos = find(int(l[0])+int(c[0])+1, pos)

    if architecture.lower().startswith('res'):
        for l in model.module.pre_process.named_children():
            if isinstance(l[1], nn.Conv2d):
                pos = find(int(l[0]), pos)
    f.write('\n ANN thresholds: {}'.format(thresholds))
    return thresholds



def test(epoch):

    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')


    with torch.no_grad():
        model.eval()
        global max_accuracy

        for batch_idx, (data, target) in enumerate(test_loader):
            print ("{} / {}".format(batch_idx, len(test_loader)))

            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()

            output  = model(data)


            loss    = F.cross_entropy(output,target)
            pred    = output.max(1,keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()

            losses.update(loss.item(),data.size(0))
            top1.update(correct.item()/data.size(0), data.size(0))
        print ("accc", top1.avg)
        temp1 = []
        for value in model.module.threshold.values():
            temp1 = temp1+[value.item()]


        return top1.avg



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='Tinyimagenet',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100','Tinyimagenet'])
    parser.add_argument('--batch_size',             default=1024,                 type=int,       help='minibatch size')
    parser.add_argument('-a','--architecture',      default='VGG16',            type=str,       help='network architecture', choices=['VGG5','VGG9','VGG11','VGG13','VGG16','VGG19'])
    parser.add_argument('-lr','--learning_rate',    default=1e-4,               type=float,     help='initial learning_rate')
    parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained ANN model')
    parser.add_argument('--pretrained_snn',         default='',                 type=str,       help='pretrained SNN for inference')
    parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('--epochs',                 default=60,                type=int,       help='number of training epochs')
    parser.add_argument('--lr_interval',            default='0.60 0.80 0.90',   type=str,       help='intervals at which to reduce lr, expressed as %%age of total epochs')
    parser.add_argument('--lr_reduce',              default=10,                 type=int,       help='reduction factor for learning rate')
    parser.add_argument('--timesteps',              default=250,                type=int,       help='simulation timesteps')
    parser.add_argument('--leak',                   default=1.0,                type=float,     help='membrane leak')
    parser.add_argument('--scaling_factor',         default=1.0,                type=float,     help='scaling factor for thresholds at reduced timesteps')
    parser.add_argument('--default_threshold',      default=1.0,                type=float,     help='intial threshold to train SNN from scratch')
    parser.add_argument('--activation',             default='Linear',           type=str,       help='SNN activation function', choices=['Linear', 'STDB'])
    parser.add_argument('--alpha',                  default=0.3,                type=float,     help='parameter alpha for STDB')
    parser.add_argument('--beta',                   default=0.01,               type=float,     help='parameter beta for STDB')
    parser.add_argument('--optimizer',              default='Adam',             type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--weight_decay',           default=5e-4,               type=float,     help='weight decay parameter for the optimizer')
    parser.add_argument('--momentum',               default=0.95,                type=float,     help='momentum parameter for the SGD optimizer')
    parser.add_argument('--amsgrad',                default=True,               type=bool,      help='amsgrad parameter for Adam optimizer')
    parser.add_argument('--dropout',                default=0.3,                type=float,     help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
    parser.add_argument('--test_acc_every_batch',   action='store_true',                        help='print acc of every batch during inference')
    parser.add_argument('--train_acc_batches',      default=200,                type=int,       help='print training progress after this many batches')
    parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    # Seed random number
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    dataset             = args.dataset
    batch_size          = args.batch_size
    architecture        = args.architecture
    learning_rate       = args.learning_rate
    pretrained_ann      = args.pretrained_ann
    pretrained_snn      = args.pretrained_snn
    epochs              = args.epochs
    lr_reduce           = args.lr_reduce
    timesteps           = args.timesteps
    leak                = args.leak
    scaling_factor      = args.scaling_factor
    default_threshold   = args.default_threshold
    activation          = args.activation
    alpha               = args.alpha
    beta                = args.beta
    optimizer           = args.optimizer
    weight_decay        = args.weight_decay
    momentum            = args.momentum
    amsgrad             = args.amsgrad
    dropout             = args.dropout
    kernel_size         = args.kernel_size
    test_acc_every_batch= args.test_acc_every_batch
    train_acc_batches   = args.train_acc_batches

    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value)*args.epochs))

    log_file = './logs/snn/'
    try:
        os.mkdir(log_file)
    except OSError:
        pass

    identifier = 'snn_'+architecture.lower()+'_'+dataset.lower()+'_'+str(timesteps)
    log_file+=identifier+'.log'

    if args.log:
        f = open(log_file, 'w', buffering=1)
    else:
        f = sys.stdout

    if not pretrained_ann:
        ann_file = './trained_models/ann_revision/ann_'+architecture.lower()+'_'+dataset.lower()+'.pth'

        if os.path.exists(ann_file):
            print ("pretrained_weight exist!!")
            pretrained_ann = ann_file



    f.write('\n Run on time: {}'.format(datetime.datetime.now()))

    f.write('\n\n Arguments: ')
    for arg in vars(args):
        if arg == 'lr_interval':
            f.write('\n\t {:20} : {}'.format(arg, lr_interval))
        elif arg == 'pretrained_ann':
            f.write('\n\t {:20} : {}'.format(arg, pretrained_ann))
        else:
            f.write('\n\t {:20} : {}'.format(arg, getattr(args,arg)))

    # Training settings

    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    normalize       = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])


    if dataset in ['CIFAR10', 'CIFAR100']:
        transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize
        ])
        transform_test  = transforms.Compose([transforms.ToTensor(), normalize])

    if dataset == 'CIFAR10':
        syn_trainset = syn_dataset(data_path = 'synimg/cifar10', transform = transform_train)
        testset     = datasets.CIFAR10(root='~/Datasets/cifar_data', train=False, download=True, transform = transform_test)
        labels      = 10

    elif dataset == 'CIFAR100':
        syn_trainset = syn_dataset(data_path = 'synimg/cifar100_data', transform = transform_train)
        testset     = datasets.CIFAR100(root='~/Datasets/cifar_data', train=False, download=True, transform = transform_test)
        labels      = 100


    syntrain_loader    = DataLoader(syn_trainset, batch_size=batch_size, shuffle=True)
    test_loader     = DataLoader(testset, batch_size=1024, shuffle=False)


    print ("architecture : {}".format(architecture))
    model = vgg_spiking.VGG_SNN_STDB(vgg_name = architecture, activation = activation, labels=labels, timesteps=timesteps, leak=leak, default_threshold=default_threshold, alpha=alpha, beta=beta, dropout=dropout, kernel_size=kernel_size, dataset=dataset)

    model = nn.DataParallel(model)


    if pretrained_ann:

        state = torch.load(pretrained_ann, map_location='cpu')
        cur_dict = model.state_dict()
        new_satedict = state['state_dict'].keys()

        for key in new_satedict:
            if key in cur_dict:
                if (state['state_dict'][key].shape == cur_dict[key].shape):
                    cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
                    f.write('\n Success: Loaded {} from {}'.format(key, pretrained_ann))
                else:
                    f.write('\n Error: Size mismatch, size of loaded model {}, size of current model {}'.format(state['state_dict'][key].shape, model.state_dict()[key].shape))
            else:
                f.write('\n Error: Loaded weight {} not present in current model'.format(key))
        model.load_state_dict(cur_dict)
        f.write('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))




    thresholds = find_threshold(batch_size=batch_size, timesteps=timesteps, architecture=architecture)
    model.module.threshold_update(scaling_factor=scaling_factor, thresholds=thresholds[:])

    top1_acc = test(epoch=0)
    print("after conversion accuracy:", top1_acc)

