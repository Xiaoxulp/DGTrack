import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os.path as osp
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from clip import clip
from datetime import datetime
from tensorboardX import SummaryWriter
from datasets.reiddataset import ReidDataset
from datasets.RandomIdentitySampler import RandomIdentitySampler
from datasets.ValBuilder import ValBuilder 
from reid.models.TrackBook import LearnableTrackBook
from reid.loss.loss_set import IPMOT_ContrastiveLoss
from reid.utils.serialization import save_checkpoint, load_checkpoint

from Train.engine.trackbook_trainer import TrackBookTrainer
from Train.eval.trackbook_valer import TrackBookValer
from torch.optim.lr_scheduler import CosineAnnealingLR

# CLIP 参数
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
def parse_args():
    parser = argparse.ArgumentParser(description='Train TrackBook')
    
    parser.add_argument('--json', type=str, default='/home/st3/UnifyTrack/data/reid_meta.json')
    parser.add_argument('--clip_model_path', type=str,default='/home/st3/UnifyTrack/reid/weight/ViT-B-32.pt')
    parser.add_argument('--logs_dir', type=str, default='./logs_trackbook')
    parser.add_argument('--resume', type=str, default='')
    
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('--num_instances', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=240)
    parser.add_argument('--lr', type=float, default=3.5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=202611)
    parser.add_argument('--start_save', type=int, default=20)
    parser.add_argument('--start_show', type=int, default=60)  
    
    parser.add_argument('--val_ids', type=int, default=100, help='Number of IDs per domain for validation')
    parser.add_argument('--val_imgs', type=int, default=2, help='Number of images per ID for validation')
    parser.add_argument("--output_dir", type=str, default='/home/st3/UnifyTrack/clip/ExpT_1')
    return parser.parse_args()

def main(args):
    if not osp.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Initializing Data...")
    
    Dataset = ReidDataset(args.json)
    sampler = RandomIdentitySampler(Dataset, num_instances=args.num_instances)
    dataloader = DataLoader(
        Dataset, batch_size=args.batch_size, num_workers=0,
        sampler=sampler, pin_memory=True, drop_last=True
    )
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
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    print(f"Validation Set Ready: {len(val_indices)} images.")

    
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    summary_writer = SummaryWriter(osp.join(args.logs_dir, 'tensorboard_log' + TIMESTAMP))
    print("Loading Models...")
    clip_model, _ = clip.load(args.clip_model_path, device=device)
    clip_model.float()
    for p in clip_model.parameters():
        p.requires_grad = False 
    
    trackbook = LearnableTrackBook(clip_model, n_ctx=12, n_ids=Dataset.num_pids).to(device)
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        ckpt = load_checkpoint(args.resume)
        trackbook.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']
    optimizer = torch.optim.AdamW([trackbook.ctx], lr=args.lr, weight_decay=args.weight_decay)
    if args.resume and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    
    total_epochs = args.epochs  
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,  
        eta_min=1e-6  
    )
    if args.resume and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    loss_fn = IPMOT_ContrastiveLoss().to(device)
    trainer = TrackBookTrainer(clip_model, trackbook, loss_fn, summary_writer, device, CLIP_MEAN, CLIP_STD)
    valer = TrackBookValer(clip_model, trackbook, args.output_dir, device, CLIP_MEAN, CLIP_STD)
    
    valer.val(1, val_loader)
    print("Start Training...")
    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print('[Info] Epoch [{}] learning rate: {:.3e}'.format(epoch, current_lr))
        trainer.train(epoch, dataloader, optimizer)
        scheduler.step()
        
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= args.start_show:
            valer.val(epoch + 1, val_loader)
        
        if (epoch + 1) % 10 == 0 and (epoch + 1) >= args.start_save:
            save_path = osp.join(args.logs_dir, f"trackbook_exp{epoch + 1}.pth.tar")
            save_checkpoint({
                'state_dict': trackbook.state_dict(), 
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, epoch + 1, is_best=False, fpath=save_path)
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

