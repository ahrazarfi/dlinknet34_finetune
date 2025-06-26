
import os, random, time, argparse, itertools
import torch
from torch.utils.data import DataLoader
import numpy as np

from networks.dinknet import DinkNet34
from roads_dataset import RoadsDataset, default_aug
from losses import ComboLoss
from framework import AMPFrame
from torch.optim.lr_scheduler import ReduceLROnPlateau

# optional wandb
try:
    import wandb
    WANDB_OK = True
except ImportError:
    WANDB_OK = False

# ------------------------------------------------------------------- utils
def make_split(root, val_frac:float=0.2, seed:int=42):
    ids = [f.split('_sat')[0] for f in os.listdir(root) if '_sat' in f]
    random.Random(seed).shuffle(ids)
    vlen = int(len(ids)*val_frac)
    return ids[vlen:], ids[:vlen]

def make_loaders(root, batch, workers=4, val_frac=0.2):
    train_ids, val_ids = make_split(root, val_frac)
    train_ds = RoadsDataset(train_ids, root, aug=default_aug())
    val_ds   = RoadsDataset(val_ids,   root, aug=default_aug())
    dl_kw = dict(batch_size=batch, num_workers=workers,
                 pin_memory=True, drop_last=True)
    return (DataLoader(train_ds, shuffle=True, **dl_kw),
            DataLoader(val_ds,   shuffle=False, **dl_kw))

@torch.no_grad()
def batch_metrics(logits, masks, thresh=0.5):
    preds = (torch.sigmoid(logits) > thresh).float()
    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum()
    iou_den = preds.sum() + masks.sum() - intersection
    return intersection, union, iou_den

def log_images(frame, val_loader, epoch, max_samples=3):
    if not WANDB_OK: return
    frame.model.eval()
    data_iter = iter(val_loader)
    imgs, masks = next(data_iter)
    imgs = imgs.cuda(non_blocking=True)
    with torch.cuda.amp.autocast():
        logits = frame.model(imgs)
    preds = (torch.sigmoid(logits) > 0.5).float().cpu()
    imgs = imgs.cpu()*0.229 + 0.485  # denorm roughly
    table = wandb.Table(columns=["image", "prediction", "ground_truth"])
    for i in range(min(max_samples, imgs.size(0))):
        img = imgs[i].permute(1,2,0).numpy()
        pr  = preds[i][0].numpy()
        gt  = masks[i][0].numpy()
        table.add_data(wandb.Image(img),
                       wandb.Image(pr, caption="pred"),
                       wandb.Image(gt, caption="gt"))
    wandb.log({f"val_samples_epoch_{epoch}": table})

# ------------------------------------------------------------------- main loop
def run(cfg):
    device_cnt = torch.cuda.device_count()
    batch = device_cnt * cfg['batch_per_gpu']
    train_loader, val_loader = make_loaders(cfg['data_root'],
                                            batch, val_frac=cfg['val_frac'])
    frame = AMPFrame(DinkNet34, ComboLoss(use_focal=True))
    if cfg.get('pretrained') and os.path.exists(cfg['pretrained']):
        frame.load(cfg['pretrained'])
        print('Loaded weights')

    scheduler = ReduceLROnPlateau(frame.optimizer, factor=0.5,
                                  patience=2, verbose=True)
    best_val = np.inf
    os.makedirs(cfg['weights_dir'], exist_ok=True)

    # wandb init
    use_wb = cfg['wandb'] and WANDB_OK
    if use_wb:
        wandb.init(project=cfg['wandb_project'],
                   name=cfg['exp'],
                   config=cfg)
        wandb.watch(frame.model, log="gradients", log_freq=100)

    for epoch in range(1, cfg['epochs']+1):
        t0=time.time(); frame.model.train(); train_loss=0
        for imgs, masks in train_loader:
            train_loss += frame.train_batch(imgs, masks)
        train_loss /= len(train_loader)

        # validation
        frame.model.eval()
        val_loss, inter, un, iou_dem = 0,0,0,0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs_cuda = imgs.cuda(non_blocking=True)
                masks_cuda = masks.cuda(non_blocking=True)
                val_logits = frame.model(imgs_cuda)
                val_loss += frame.loss_fn(val_logits, masks_cuda).item()
                a,b,c = batch_metrics(val_logits, masks_cuda)
                inter += a.item(); un += b.item(); iou_dem += c.item()
        val_loss /= len(val_loader)
        val_dice = 2*inter/(un+1e-8)
        val_iou  = inter/(iou_dem+1e-8)
        scheduler.step(val_loss)

        lr = frame.optimizer.param_groups[0]['lr']
        print(f"Ep{epoch:03d} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | dice {val_dice:.4f} | "
              f"iou {val_iou:.4f} | lr {lr:.2e} | {int(time.time()-t0)}s")

        if use_wb:
            wandb.log({'epoch':epoch,'train_loss':train_loss,
                       'val_loss':val_loss,'val_dice':val_dice,
                       'val_iou':val_iou,'lr':lr})
            if epoch % cfg['img_log_freq']==0:
                log_images(frame, val_loader, epoch)

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(cfg['weights_dir'], f"{cfg['exp']}_best.th")
            frame.save(ckpt_path)
            if use_wb:
                artifact = wandb.Artifact(f"{cfg['exp']}_model", type="model")
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact)

        if lr < 5e-7:
            print("LR too low, stopping"); break

    if use_wb: wandb.finish()

# ------------------------------------------------------------------- CLI
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--pretrained', default='')
    ap.add_argument('--exp', default='finetune')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch_per_gpu', type=int, default=4)
    ap.add_argument('--weights_dir', default='weights')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--wandb_project', default='dink-rural-roads')
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--img_log_freq', type=int, default=5)
    args = ap.parse_args()
    cfg = vars(args)
    run(cfg)
