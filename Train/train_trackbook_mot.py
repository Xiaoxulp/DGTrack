import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os.path as osp
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.backends import cudnn
from tensorboardX import SummaryWriter
from datetime import datetime
from datasets.reiddataset import ReidDataset
from datasets.RandomIdentitySampler import RandomIdentitySampler
from torch.utils.data import DataLoader
from reid.models.rga_model_clip import resnet50_rga
from reid.models.TrackBook import LearnableTrackBook 
from clip import clip 
from reid.utils.serialization import load_checkpoint,save_checkpoint
from reid.loss.loss_set import CrossEntropyLabelSmoothLoss,TripletHardLoss,Li2tceLoss 
from torch.optim.lr_scheduler import CosineAnnealingLR
from Train.engine.reid_trainers_trackbook import ReidTrainer
from torch.utils.data import Subset
from datasets.ValBuilder import ValBuilder
from Train.eval.reid_valers import ReidValer

def parse_args():
    
    parser = argparse.ArgumentParser(description='Train MOT Reid')
    # base config
    parser.add_argument('--json', type=str, default='/home/st3/UnifyTrack/data/reid_meta.json')
    parser.add_argument('-d', '--dataset', type=str, default='/home/st3/UnifyTrack/data')
    parser.add_argument('-b', '--batch_size', type=int, default=72)  # 16
    parser.add_argument('-p', '--path', type=str,default='/home/st3/UnifyTrack/datasets/data_path/data_path.txt')
    parser.add_argument("--num_classes", type=str,default='/home/st3/UnifyTrack/datasets/data_path/total_len.txt')
    # trackbook config
    parser.add_argument('--clip_model_path', type=str, default='/home/st3/UnifyTrack/reid/weight/ViT-B-32.pt')
    parser.add_argument('--trackbook_ckpt', type=str, default='/home/st3/UnifyTrack/Train/logs_trackbook_old/trackbook_exp230_230.pth.tar')
    # train config
    parser.add_argument('-j', '--workers', type=int, default=32) 
    parser.add_argument('--num_instances', type=int, default=4)
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--features', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.5)
    # loss
    parser.add_argument('--margin', type=float, default=0.3,help="margin of the triplet loss, default: 0.3")
    # optimizer
    parser.add_argument('-opt', '--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-4,help="learning rate of new parameters, for pretrained ")
    # training configs
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to use (auto select)')  
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--start_save', type=int, default=70,help="start saving checkpoints after specific epoch")  
    parser.add_argument('--start_show', type=int, default=30)  
    parser.add_argument('--seed', type=int, default=202611)
    parser.add_argument('--random_erasing', type=bool, default=True)
    
    parser.add_argument('--val_ids', type=int, default=100, help='Number of IDs per domain for validation')
    parser.add_argument('--val_imgs', type=int, default=2, help='Number of images per ID for validation')
    parser.add_argument("--output_dir", type=str, default='/home/st3/UnifyTrack/appearance/vis2/u4')
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='cosine', choices=['euclidean', 'kissme'])  # cosine
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs-file', type=str, metavar='PATH',default='log.txt')
    return parser.parse_args()

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    
    with open(args.num_classes, 'r') as file:
        data = file.read().strip()  
        number = int(data)  
        num_classes = number
    Dataset = ReidDataset(args.json)
    
    sampler = RandomIdentitySampler(Dataset, num_instances=args.num_instances)
    dataloader = DataLoader(
        Dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
        pin_memory=True, drop_last=True
    )
    print("Building Validation Set...")
    
    print("Building Validation Set...")
    val_builder = ValBuilder(Dataset)
    val_indices = val_builder.build(
        num_ids_per_domain=args.val_ids,
        num_imgs_per_id=args.val_imgs,
        seed=args.seed
    )
    val_dataset = Subset(Dataset, val_indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  
        num_workers=0,
        pin_memory=True
    )
    print(f"Validation Set Ready: {len(val_indices)} images.")
   
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    summary_writer = SummaryWriter(osp.join(args.logs_dir, 'tensorboard_log' + TIMESTAMP))
    model = resnet50_rga( 
        pretrained=True,
        num_feat=args.features,
        height=args.height,
        width=args.width,
        dropout=args.dropout,
        num_classes=num_classes
    )
    # TrackBook (Frozen)
    print("Loading TrackBook (Frozen)...")
    clip_model, _ = clip.load(args.clip_model_path, device='cuda')
    clip_model.float()
    for p in clip_model.parameters():
        p.requires_grad = False
    trackbook = LearnableTrackBook(clip_model, n_ctx=12, n_ids=num_classes).to('cuda') 
    
    if os.path.exists(args.trackbook_ckpt):
        print(f"Loading TrackBook from {args.trackbook_ckpt}")
        tb_ckpt = torch.load(args.trackbook_ckpt)
        trackbook.load_state_dict(tb_ckpt['state_dict'])
    else:
        print("Warning: TrackBook checkpoint not found! Using random init.")
    trackbook.eval()
    for p in trackbook.parameters():
        p.requires_grad = False
    ## Load from checkpoint
    start_epoch = best_top1 = 0  
    # model = model.to('cuda')  
    model = nn.DataParallel(model).cuda() 
    print(model.device_ids)
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        # model.load_state_dict(checkpoint['state_dict'])
        model.module.load_state_dict(checkpoint['state_dict'])  
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    # loss
    criterion_cls = CrossEntropyLabelSmoothLoss(num_classes).cuda()
    criterion_tri = TripletHardLoss(margin=args.margin)
    criterion_clip = Li2tceLoss().cuda()
    criterion = [criterion_cls, criterion_tri, criterion_clip]
    ## Optimizer
    # param_groups = filter(lambda p: p.requires_grad, model.module.parameters())
    param_groups = filter(lambda p: p.requires_grad, model.module.parameters())
    # backbone_params = []
    # reid_head_params = []
    # for name, p in model.module.named_parameters():
    #     if not p.requires_grad:
    #         continue
    #     elif 'cls' in name or 'feat_bn' in name:
    #         reid_head_params.append(p)
    #     else:
    #         backbone_params.append(p)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            param_groups, lr=args.lr, weight_decay=args.weight_decay
        )
        # optimizer = torch.optim.Adam([
        #     {'params': backbone_params, 'lr': args.lr},
        #     {'params': reid_head_params, 'lr': args.lr},
        # ], weight_decay=args.weight_decay)
    else:
        raise NameError
    if args.resume and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    total_epochs = args.epochs  
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,  
        eta_min=1e-6 
    )
    if args.resume and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    trainer = ReidTrainer(model=model, trackbook=trackbook, criterion=criterion, summary_writer=summary_writer)
    
    valer = ReidValer(model, args.output_dir)
    # valer.val(1, val_loader)
    # valer.run_cam(1, val_loader) 
    for epoch in range(start_epoch, total_epochs):
        print(f"Epoch {epoch + 1}")
        
        current_lr = optimizer.param_groups[0]['lr']
        print('[Info] Epoch [{}] learning rate: {:.3e}'.format(epoch, current_lr))
        trainer.train(epoch, dataloader, optimizer, random_erasing=args.random_erasing, empty_cache=False)
        scheduler.step()
        if (epoch +1)% 15 == 0 and (epoch +1) >= args.start_show:
           
            print('t-sne')
            # valer.val(epoch+1, val_loader)
            # print('save - grad_gam')
            # valer.run_cam(epoch+1, val_loader)
        
        if (epoch + 1) % 10 == 0 and (epoch + 1) >= args.start_save:
            is_best = False
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
                'best_top1': best_top1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),  
            }, epoch + 1, is_best, save_interval=1, fpath=osp.join(args.logs_dir, 'u4.pth.tar'))

if __name__ == '__main__':
    args = parse_args()
    main(args)

