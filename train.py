import argparse

import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
import numpy as np
import torch.optim as optim
import torchvision.utils as vutils
import time
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import sys
# sys.path.append('./netwroks')  

from dataset.datasets_rgb import LIPDataSet
import torchvision.transforms as transforms
import timeit
from tensorboardX import SummaryWriter
from utils.utils import decode_parsing, inv_preprocess
from utils.lovasz_losses import LovaszSoftmaxDSN
from utils.criterion2 import CriterionDSN
from utils.loss import OhemCrossEntropy2d
from utils.encoding import DataParallelModel, DataParallelCriterion 
from utils.miou import compute_mean_ioU
from config import get_arguments
from ModelDefinition import create_model
from loss.criterion import Seg_Loss

start = timeit.default_timer()
  

args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, total_iters):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, total_iters, args.power)
    optimizer.param_groups[0]['lr'] = lr
    # for i in range(1,len( optimizer.param_groups)):
        # optimizer.param_groups[i]['lr'] = lr
    return lr



def model_init(model,optimizer, args):
    saved_state_dict = torch.load(args.restore_from)
    if args.start_epoch >0:
        model = DataParallelModel(model)
        model.load_state_dict(saved_state_dict['state_dict'])
        if 'optimizer' in saved_state_dict:
            optimizer.load_state_dict(saved_state_dict['optimizer'])
            print ('========Load Optimizer',args.restore_from)
    else:
        new_params = model.state_dict().copy()
        #state_dict_pretrain = saved_state_dict #['state_dict']

        for state_name in saved_state_dict:
            if state_name in new_params:
                new_params[state_name] = saved_state_dict[state_name]
            else:
                print ('Model Missed',state_name)
        for state_name in new_params:
            if state_name not in saved_state_dict:
                print ('Model Increased',state_name)
        model.load_state_dict(new_params)
        model = DataParallelModel(model)
    print ('-------Load Weight',args.restore_from)

def main():
    """Create the model and start the training."""
    print (args)
    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    writer = SummaryWriter(args.snapshot_dir)

    gpus = [int(i) for i in args.gpu.split(',')]
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.enabled = True
    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    print('Create Dataset')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    trainloader = data.DataLoader(LIPDataSet(args,crop_size=input_size, transform=transform,list_path=args.list_path),
                                  batch_size=args.batch_size * len(gpus), shuffle=True, num_workers=8,
                                  pin_memory=True)

    num_samples = 5000 

    model = create_model(args)
    criterion = Seg_Loss(args, input_size)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    if args.restore_from!='':
        model_init(model,optimizer, args)
    else:
        model = DataParallelModel(model)
    model.cuda()
    criterion.cuda()


    # dump_input = torch.rand((args.batch_size, 3, input_size[0], input_size[1]))
    # writer.add_graph(model.cuda(), dump_input.cuda(), verbose=False)

    
    '''
    list_map = []

    for part in model.path_list:
        list_map = list_map + list(map(id, part.parameters()))
    
    base_params = filter(lambda p: id(p) not in list_map,
                         model.parameters())
    params_list = []
    params_list.append({'params': base_params, 'lr':args.learning_rate*0.1})
    for part in model.path_list:
        params_list.append({'params': part.parameters()})
    print ('len(params_list)',len(params_list))
    '''

    total_iters = args.epochs * len(trainloader)
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        for i_iter, batch in enumerate(trainloader):
            i_iter += len(trainloader) * epoch
            lr = adjust_learning_rate(optimizer, i_iter, total_iters)

            images, labels, _ = batch
            labels = labels.long().cuda(non_blocking=True)
            preds = model(images)

            losses = criterion(preds,labels)
            loss_total = sum(losses)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            if i_iter % 100 == 0:
                writer.add_scalar('learning_rate', lr, i_iter)
                writer.add_scalar('total_loss', loss_total.data.cpu().numpy(), i_iter)
                for i_loss in range(len(losses)):
                    name_loss = 'loss' + '_' + str(i_loss)
                    writer.add_scalar(name_loss, losses[i_loss].data.cpu().numpy(), i_iter)

            print('epoch = {}, iter = {} of {} completed,lr={:.4f}, loss = {:.4f}, BCE_loss = {:.4f}, IoU_loss = {:.4f}'
                .format(epoch, i_iter, total_iters,lr, loss_total.data.cpu().numpy(),losses[0].data.cpu().numpy(),losses[-1].data.cpu().numpy())) 
        if epoch%args.save_step == 0 or epoch==args.epochs:
            time.sleep(10)
            save_checkpoint(model,epoch,optimizer)

    time.sleep(10)
    save_checkpoint(model,epoch,optimizer)
    end = timeit.default_timer()
    print(end - start, 'seconds')

def save_checkpoint(model,epoch,optimizer):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    filepath =  osp.join(args.snapshot_dir, 'LIP_epoch_' + str(epoch) + '.pth')
    torch.save(state, filepath)

def valid(model, valloader, input_size, num_samples, gpus):
    model.eval()

    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]),
                             dtype=np.uint8)

    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            image, meta = batch
            num_images = image.size(0)
            if index % 10 == 0:
                print('%d  processd' % (index * num_images))

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]

            outputs = model(image.cuda())
            if gpus > 1:
                for output in outputs:
                    parsing = output[0][-1]
                    nums = len(parsing)
                    parsing = interp(parsing).data.cpu().numpy()
                    parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                    parsing_preds[idx:idx + nums, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

                    idx += nums
            else:
                parsing = outputs[0][-1]
                parsing = interp(parsing).data.cpu().numpy()
                parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

                idx += num_images

    parsing_preds = parsing_preds[:num_samples, :, :]


    return parsing_preds, scales, centers


if __name__ == '__main__':
    main()
