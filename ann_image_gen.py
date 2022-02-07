#---------------------------------------------------
# code reference: https://github.com/utkuozbulak/pytorch-cnn-visualizations
#---------------------------------------------------
from __future__ import print_function


import argparse

from torch.distributions.dirichlet import Dirichlet
from self_models import *
from image_gen_function import *

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('--gpu',                    default=True, type=bool,      help='use gpu')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100','Tinyimagenet'])
    parser.add_argument('-a','--architecture',      default='VGG16',            type=str,       help='network architecture', choices=['VGG5','VGG9','VGG11','VGG13','VGG16','VGG19'])
    parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained ANN model')
    parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')


    parser.add_argument('--num_synimage',             default=100,                 type=int,       help='num_synimage')



    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    # Seed random number
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    dataset             = args.dataset
    architecture        = args.architecture
    pretrained_ann = args.pretrained_ann



    log_file = './logs/snn/'
    try:
        os.mkdir(log_file)
    except OSError:
        pass


    if not pretrained_ann:
        ann_file = './trained_models/ann/ann_'+architecture.lower()+'_'+dataset.lower()+'.pth'

        if os.path.exists(ann_file):
            print ("pretrained_weight exist!!")
            pretrained_ann = ann_file


    if dataset == 'CIFAR10':
        labels      = 10
    elif dataset == 'CIFAR100':
        labels      = 100



    model = VGG(vgg_name=architecture, labels=labels, dataset=dataset)
    model = nn.DataParallel(model)

    state = torch.load(pretrained_ann, map_location='cpu')
    model.load_state_dict(state['state_dict'])

    # correlation matrix
    cor_vec =model.module.classifier[-1].weight
    cor_mag = torch.norm(cor_vec, p=2, dim=1)
    cor_mat = torch.matmul(cor_vec, cor_vec.permute(1,0))


    normalized_cormat = torch.zeros_like(cor_mat)
    for i in range(labels):
        for j in range(labels):
            normalized_cormat[i,j] = cor_mat[i, j] / (cor_mag[i]*cor_mag[j])

    min_mat = ((torch.min(normalized_cormat,dim=1,keepdim=True))[0]).repeat(1,labels)
    normalized_cormat = normalized_cormat - min_mat
    max_mat = ((torch.max(normalized_cormat,dim=1,keepdim=True))[0]).repeat(1,labels)
    normalized_cormat = normalized_cormat/max_mat

    diri_list = []
    for i in range(labels):
        m = Dirichlet((normalized_cormat[i,:]))
        diri_list.append(m)


    img_idxes = range(args.num_synimage)
    target_classes = range(labels)



    for target_class in target_classes:
        print ('----------class', target_class)
        path = 'synimg/cls'+str(target_class)
        if not os.path.exists(path):
            os.makedirs(path)
        import time
        time_list  = []
        for img_idx in img_idxes:
            s_time = time.time()

            csig = Data_impression(model, target_class, img_idx,diri_list, dataset)
            if dataset == "MNIST":
                csig.generate(iterations=1500)
            else:
                csig.generate(iterations=2500)

            time_list.append(time.time()-s_time)
