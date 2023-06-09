import os
import torch
from dataset import CIFAR10
from network import ConvNet
import tqdm 
import torch.optim as optim
from util import evaluate, AverageMeter
import argparse
from torch.utils.tensorboard  import SummaryWriter

def MyCELoss(pred, gt):
    
    # Implement CE loss here
    one_hot = torch.zeros_like(pred)
    lines = torch.arange(pred.shape[0])
    one_hot[lines, gt] = 1
    loss = (one_hot * torch.log(pred)).sum(dim = -1).mean(dim = 0)
    return -loss 

def validate(epoch, model, val_loader, writer, lr):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for imgs, labels in tqdm.tqdm(val_loader):
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        bsz = labels.shape[0]
        output = model(imgs)
        if torch.cuda.is_available():
            output = output.cpu()
        # update metric
        acc1, acc5 = evaluate(output, labels, topk=(1, 5))
        top1.update(acc1.item(), bsz)
        top5.update(acc5.item(), bsz)

    # draw accuracy curve!
    writer.add_scalars('val/top1', {str(lr): top1.avg}, epoch)
    writer.add_scalars('val/top5', {str(lr): top5.avg}, epoch)
    
    print(' Val Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' Val Acc@5 {top5.avg:.3f}'.format(top5=top5))
    return 

def train(epoch, model, optimizer, train_loader, writer, lr):
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    iteration = len(train_loader) * epoch
    for imgs, labels in tqdm.tqdm(train_loader):
        bsz = labels.shape[0]
        
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        output = model(imgs)
        loss = MyCELoss(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = evaluate(output, labels, topk=(1, 5))
        top1.update(acc1.item(), bsz)
        top5.update(acc5.item(), bsz)

        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 50 == 0:
            # draw loss curve and accuracy curve!
            writer.add_scalars('train/loss', {str(lr): loss.item()}, iteration)
            writer.add_scalars('train/top1', {str(lr): top1.avg}, iteration)
            writer.add_scalars('train/top5', {str(lr): top5.avg}, iteration)
            

    print(' Epoch: %d'%(epoch))
    print(' Train Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' Train Acc@5 {top5.avg:.3f}'.format(top5=top5))
    return 

def run(args):
    save_folder = os.path.join('../experiments',  args.exp_name)
    ckpt_folder = os.path.join(save_folder,  'ckpt')
    log_folder = os.path.join(save_folder,  'log')
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_folder)

    # define dataset and dataloader
    train_dataset = CIFAR10(train = True, aug = args.aug)
    val_dataset = CIFAR10(train=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
         val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=2)
    
    # define network 
    model = ConvNet()
    if torch.cuda.is_available():
        model = model.cuda()

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cont:
        # load latest checkpoint
        ckpt_lst = os.listdir(ckpt_folder)
        ckpt_lst.sort(key=lambda x: int(x.split('_')[-1]))
        read_path = os.path.join(ckpt_folder, ckpt_lst[-1])
        print('load checkpoint from %s'%(read_path))
        checkpoint = torch.load(read_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.total_epoch):
        train(epoch, model, optimizer, train_loader, writer, args.lr)
        
        if epoch % args.save_freq == 0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(ckpt_folder, 'ckpt_epoch_%s'%(str(epoch)))
            torch.save(state, save_file)

        with torch.no_grad():
            validate(epoch, model, val_loader, writer, args.lr)
    return 

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_name', '-e', type=str, required=True, help="The checkpoints and logs will be save in ./checkpoint/$EXP_NAME")
    arg_parser.add_argument('--lr', '-l', type=float, default=1e-4, help="Learning rate")
    arg_parser.add_argument('--save_freq', '-s', type=int, default=1, help="frequency of saving model")
    arg_parser.add_argument('--total_epoch', '-t', type=int, default=10, help="total epoch number for training")
    arg_parser.add_argument('--cont', '-c', action='store_true', help="whether to load saved checkpoints from $EXP_NAME and continue training")
    arg_parser.add_argument('--batchsize', '-b', type=int, default=20, help="batch size")
    arg_parser.add_argument('--aug', '-a', action = 'store_true', help = 'whether to use data augmentation')
    args = arg_parser.parse_args()

    run(args)