import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from dataset import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def set_break():
    import pdb 
    pdb.set_trace()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'streetdance245'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()


if args.dataset == 'ucf101':
    num_class = 101
elif args.dataset == 'hmdb51':
    num_class = 51
elif args.dataset == 'kinetics':
    num_class = 400
elif args.dataset == 'streetdance245':
    num_class = 245
else:
    raise ValueError('Unknown dataset '+args.dataset)

net = TSN(num_class, args.test_segments, args.modality,
          base_model=args.arch,
          consensus_type=args.crop_fusion_type,
          dropout=args.dropout)

checkpoint = torch.load(args.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
# print(base_dict)
# set_break()
net.load_state_dict(base_dict)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

data_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.test_list, num_segments=args.test_segments,
                   new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   image_tmpl="{:d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=4, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))


net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []


def eval_video(video_data):
    i, data, label = video_data
    # print('label size: ', label.shape)
    num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)

    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
                                        volatile=True)
    # rst = net(input_var).data.cpu().numpy().copy()
    rst = net(input_var)
    # rst shape here is (batch_size, label)
    # print('rst.shape: ', rst.shape)
    # set_break()
    # return i, rst.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape(
    #     (args.test_segments, 1, num_class)
    # ), label[0]
    return i, rst, label


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate_video():
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    avg_prec1 = 0.0
    avg_prec5 = 0.0
    cnt = 0

    for i, (data, label) in data_gen:
        if i >= max_num:
            break
        label = label.cuda()
        output = net(data) 
        prec1, prec5 = accuracy(output.data, label, topk=(1,5))
        print('Top 1: {}, Top 5: {}'.format(prec1, prec5))

        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))
        
        cnt += 1
        avg_prec1 += prec1.item()
        avg_prec5 += prec5.item()
        # print(i)
        # print(data.shape)
        # import pdb
        # pdb.set_trace()
        # rst = eval_video((i, data, label))

        # output.append(rst[1:])
        # cnt_time = time.time() - proc_start_time
        # print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
        #                                                                 total_num,
        #                                                                 float(cnt_time) / (i+1)))
    print('Average Top 1: {}, Average Top 5: {}'.format(avg_prec1 / cnt, avg_prec5 / cnt))


validate_video()
# video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]

# video_pred = [np.mean(x[0], axis=0) for x in output]

# video_labels = [x[1] for x in output]

# video_pred = torch.Tensor(video_pred)
# video_pred = video_pred.squeeze(dim=1)
# video_labels = torch.Tensor(video_labels)

# cf = confusion_matrix(video_labels, video_pred).astype(float)

# cls_cnt = cf.sum(axis=1)
# cls_hit = np.diag(cf)

# cls_acc = cls_hit.sum() / cls_cnt.sum()

# print('video_pred: ', video_pred.shape)
# print('video_labels: ', video_labels.shape)

# import pdb
# pdb.set_trace()


# print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

if args.save_scores is not None:

    # reorder before saving
    name_list = [x.strip().split()[0] for x in open(args.test_list)]

    order_dict = {e:i for i, e in enumerate(sorted(name_list))}

    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)

    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]

    np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)


