import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os.path as osp
import numpy as np
import argparse
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from datetime import datetime
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from datasets.MOTDataset import MOTDataset
from datasets.mot_collate_fn import mot_collate_fn
from fusion.model.unified_model import UnifiedModel 
from fusion.loss.loss import SparseMatchingLoss,BCEWeightLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.serialization import save_checkpoint
from fusion.engine.fusion_trainer import FusionTrainer 
from datasets.MOTValBuilder import MOTValBuilder
from torch.utils.data import Subset
from fusion.eval.fusion_valers import FusionValer

def parse_args():
    parser = argparse.ArgumentParser(description='Train Motion Matching')
    parser.add_argument('--data_txt', type=str,default='/home/st3/DGTrack/datasets/data_path/data_path.txt')
    parser.add_argument('--root', type=str, default='/home/st3/DGTrack/data') 
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument("--num_classes", type=str,default='/home/st3/DGTrack/datasets/data_path/total_len.txt')
    parser.add_argument('--ckpt', type=str,default='/home/st3/DGTrack/Train/logs/UnifyTrack5T_best.pth.tar')
    parser.add_argument('--reid_pkl_paths', nargs='+', default=[
        '/home/st3/DGTrack/data/Domain1/gt_embeddings/train/gt_embeddings.pkl',
        '/home/st3/DGTrack/data/Domain2/gt_embeddings/train/gt_embeddings.pkl',
        '/home/st3/DGTrack/data/Domain3/gt_embeddings/train/gt_embeddings.pkl',
    ])
    parser.add_argument('--ema_alpha', type=float, default=0.95)
    parser.add_argument('--max_history', type=int, default=100)
   
    parser.add_argument('-j', '--workers', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--start_save', type=int, default=1)
    parser.add_argument('--seed', type=int, default=202611)
    
    parser.add_argument('--val_frames', type=int, default=35, help='Number of frames per domain for validation')
    parser.add_argument('--output_dir', type=str, default='/home/st3/chenruxu/DGTrack/appearance/vis/vis_fusion1')
    parser.add_argument('--start_show', type=int, default=1, help='Start visualization after specific epoch')
    '''
        misc
    '''
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, default=osp.join(working_dir, 'logs'))
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--num_gpu', type=int, default=1)
    return parser.parse_args()

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    with open(args.num_classes, 'r') as file:
        data = file.read().strip()
        number = int(data)
        num_classes = number
    print("Building Motion & Vision Train Set...")
    
    train_dataset = MOTDataset(args=args, data_txt_path=args.data_txt, seqs_folder=args.root)
    collate_fn = mot_collate_fn
    sampler_train = RandomSampler(train_dataset)
    batch_sampler_train = BatchSampler(sampler_train, args.batch_size, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, shuffle=False, num_workers=args.workers,
                              collate_fn=collate_fn, pin_memory=True)
    print('Finsh Motion & Vision Set...')
    val_builder = MOTValBuilder(train_dataset)
    val_indices = val_builder.build(
        num_frames_per_domain=args.val_frames,
        seed=args.seed
    )
    val_dataset = Subset(train_dataset, val_indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    print(f"Validation Set Ready: {len(val_indices)} frames loaded.")
    
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    summary_writer = SummaryWriter(osp.join(args.logs_dir, 'tensorboard_' + TIMESTAMP))
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    # print("Loading Reid Backbone...")
    # backbone = ReidExtractor(args.ckpt,
    #                          num_classes=num_classes,
    #                          num_domains=3).to(device)  
    print("Initializing Fusion Model...")
    model = UnifiedModel(input_dim=2048, hidden_dim=256, output_dim=256).to(device)
    # Resume
    start_epoch = 0
    if args.resume:
        print(f"=> Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
    
    criterion = BCEWeightLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-7)
    if args.resume and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    trainer = FusionTrainer(model,criterion, summary_writer)
    valer = FusionValer(model,args.output_dir)
    # valer.val(1, val_loader)
    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch + 1}/{args.epochs}  lr:{current_lr:.2e}")
        trainer.train(epoch, train_loader, optimizer,device)
        scheduler.step()
        if (epoch + 1) % 1 == 0 and (epoch + 1) >= args.start_show:
            print(f"\n Visualizing Epoch {epoch + 1} ")
            print("Running TSNE + Similarity Matrix Visualization...")
            valer.val(epoch + 1, val_loader)  
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= args.start_save:
            save_checkpoint(
                state={  
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                },
                epoch=epoch + 1,  
                is_best=False,  
                fpath=osp.join(args.logs_dir, f'model_{epoch + 1}.pth.tar')
            )
            print('saving checkpoint...')
    print("Training finished.")

if __name__ == '__main__':
    args = parse_args()
    main(args)
