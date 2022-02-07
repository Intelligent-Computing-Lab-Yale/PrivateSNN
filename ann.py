#############################
#@author: Nitin Rathi  (ref: https://github.com/nitin-rathi/hybrid-snn-conversion)
#############################
import argparse
import torch.optim as optim
import datetime
import os
import numpy as np
from self_models import *
from torchvision import datasets, transforms

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

def train(epoch, loader):

    global learning_rate
    
    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    if epoch in lr_interval:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / lr_reduce
            learning_rate = param_group['lr']
    
    model.train()
    for batch_idx, (data, target) in enumerate(loader):

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        pred = output.max(1,keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()

        losses.update(loss.item(), data.size(0))
        top1.update(correct.item()/data.size(0), data.size(0))
        
    print('\n Epoch: {}, lr: {:.1e}, train_loss: {:.4f}, train_acc: {:.4f}'.format(
            epoch,
            learning_rate,
            losses.avg,
            top1.avg
            )
        )

def test(loader):

    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    with torch.no_grad():
        model.eval()
        total_loss = 0
        correct = 0
        global max_accuracy, start_time
        
        for batch_idx, (data, target) in enumerate(loader):

            data, target = data.cuda(), target.cuda()
            
            output = model(data)
            loss = F.cross_entropy(output,target)
            total_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            losses.update(loss.item(), data.size(0))
            top1.update(correct.item()/data.size(0), data.size(0))

        print ("test_acc:", top1.avg)

        if epoch>30 and top1.avg<0.15:
            print('\n Quitting as the training is not progressing')
            exit(0)

        if top1.avg>max_accuracy:
            max_accuracy = top1.avg
            state = {
                    'accuracy'      : max_accuracy,
                    'epoch'         : epoch,
                    'state_dict'    : model.state_dict(),
                    'optimizer'     : optimizer.state_dict()
            }
            if os.path.exists('./trained_models/ann/') is False:
                os.mkdir('./trained_models')
                os.mkdir('./trained_models/ann')

            
            filename = './trained_models/ann/'+identifier+'.pth'
            torch.save(state,filename)
            
        print(' test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f}, time: {}'.  format(
            losses.avg, 
            top1.avg,
            max_accuracy,
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            )
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ANN to be later converted to SNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='CIFAR100',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100','Tinyimagenet'])
    parser.add_argument('--batch_size',             default=512,                 type=int,       help='minibatch size')
    parser.add_argument('-a','--architecture',      default='VGG16',            type=str,       help='network architecture', choices=['VGG5','VGG9','VGG11','VGG13','VGG16','VGG19','RESNET12','RESNET20','RESNET34'])
    parser.add_argument('-lr','--learning_rate',    default=1e-1,               type=float,     help='initial learning_rate')
    parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    parser.add_argument('--epochs',                 default=90,                type=int,       help='number of training epochs')
    parser.add_argument('--lr_interval',            default='0.60 0.80 0.90',   type=str,       help='intervals at which to reduce lr, expressed as %%age of total epochs')
    parser.add_argument('--lr_reduce',              default=10,                 type=int,       help='reduction factor for learning rate')
    parser.add_argument('--optimizer',              default='SGD',              type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--weight_decay',           default=5e-4,               type=float,     help='weight decay parameter for the optimizer')
    parser.add_argument('--momentum',               default=0.9,                type=float,     help='momentum parameter for the SGD optimizer')
    parser.add_argument('--amsgrad',                default=True,               type=bool,      help='amsgrad parameter for Adam optimizer')
    parser.add_argument('--dropout',                default=0.0,                type=float,     help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
    parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')
        
    args=parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    
    # Seed random number
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    dataset         = args.dataset
    batch_size      = args.batch_size
    architecture    = args.architecture
    learning_rate   = args.learning_rate
    epochs          = args.epochs
    lr_reduce       = args.lr_reduce
    optimizer       = args.optimizer
    weight_decay    = args.weight_decay
    momentum        = args.momentum
    amsgrad         = args.amsgrad
    dropout         = args.dropout
    kernel_size     = args.kernel_size

    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value)*args.epochs))
    
    
    log_file = './logs/ann/'
    try:
        os.mkdir(log_file)
    except OSError:
        pass 


    identifier = 'ann_'+architecture.lower()+'_'+dataset.lower()+'_'+str(datetime.datetime.now())
    log_file+=identifier+'.log'
    print (identifier)

    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Loading Dataset
    if dataset == 'CIFAR100':
        normalize   = transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
        labels      = 100 
    elif dataset == 'CIFAR10':
        normalize   = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        labels      = 10


    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])




    if dataset == 'CIFAR100':
        train_dataset   = datasets.CIFAR100(root='~/Datasets/cifar_data', train=True, download=True,transform =transform_train)
        test_dataset    = datasets.CIFAR100(root='~/Datasets/cifar_data', train=False, download=True, transform=transform_test)
    
    elif dataset == 'CIFAR10': 
        train_dataset   = datasets.CIFAR10(root='~/Datasets/cifar_data', train=True, download=True,transform=transform_train)
        test_dataset    = datasets.CIFAR10(root='~/Datasets/cifar_data', train=False, download=True, transform=transform_test)




    train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader     = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    model = VGG(vgg_name=architecture, labels=labels, dataset=dataset, kernel_size=kernel_size, dropout=dropout)


    model = nn.DataParallel(model).cuda()

    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay)

    print('\n {}'.format(optimizer))

    max_accuracy = 0

    for epoch in range(1, epochs):

        start_time = datetime.datetime.now()
        train(epoch, train_loader)
        if (epoch+1) % 5 ==0:
            test(test_loader)

    print('\n Highest accuracy: {:.4f}'.format(max_accuracy))
