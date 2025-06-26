# === train_finetune_minimal.py ===============================================
"""
Minimal finetuning script with *progressive unfreezing*.
All original training logic is preserved; only safe additions & bug-fixes.
"""
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
import argparse
from time import time

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not available → running without online logging.")

# -- original imports ---------------------------------------------------------
from networks.dinknet import DinkNet34
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------  
# 1.  Progressive-unfreezing wrapper around MyFrame
# -----------------------------------------------------------------------------
class ProgressiveUnfreezingFrame(MyFrame):
    """
    Extends MyFrame with   progressive-unfreezing + discriminative LRs
    """
    def __init__(self, net_cls, loss_fn, lr=2e-4, evalmode=False,
                 lr_ratios=None,  # let caller override the LR ladder
                 max_unfreeze_level=3):
        super().__init__(net_cls, loss_fn, lr, evalmode)

        self.current_unfreeze_level = 0
        self.max_unfreeze_level = max_unfreeze_level

        # LR ladder (decoder : encoder4 : encoder3 : encoder2) — override via config
        self.lr_ratios = lr_ratios or {
            'decoder': 1.0,
            'encoder4': 0.1,
            'encoder3': 0.03,
            'encoder2': 0.01,
        }

        # ── layer groups -------------------------------------------------------
        self.layer_groups = {
            'always_frozen': [
                'firstconv', 'firstbn', 'firstrelu',
                'firstmaxpool', 'encoder1'
            ],
            'decoder': [
                'dblock', 'decoder4', 'decoder3', 'decoder2', 'decoder1',
                'finaldeconv1', 'finalrelu1',
                'finalconv2',  'finalrelu2',  'finalconv3'
            ],
            'encoder4': ['encoder4'],
            'encoder3': ['encoder3'],
            'encoder2': ['encoder2'],
        }

        # ── 1) freeze backbone, leave decoder trainable -----------------------
        self._freeze_encoder_layers()

        # ── 2) freeze *all* BN running-stats & weights ------------------------
        for m in self.net.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()                           # stop running-stats update
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

        # ── 3) build optimiser once with differential LRs ---------------------
        self.optimizer = torch.optim.AdamW([])     # empty; we’ll add groups
        self._add_or_update_param_groups()

    # -------------------------------------------------------------------------
    # internal helpers
    # -------------------------------------------------------------------------
    def _freeze_encoder_layers(self):
        """Freeze everything except decoder."""
        frozen = ['always_frozen', 'encoder2', 'encoder3', 'encoder4']
        for group in frozen:
            for lname in self.layer_groups[group]:
                layer = getattr(self.net.module, lname, None)
                if layer is None:
                    continue
                for p in layer.parameters():
                    p.requires_grad = False
        print("Initial freeze complete ➜ decoder only.")

    def _iter_group_params(self, group_name):
        for lname in self.layer_groups.get(group_name, []):
            layer = getattr(self.net.module, lname, None)
            if layer is None:
                continue
            for p in layer.parameters():
                if p.requires_grad:
                    yield p

    def _add_or_update_param_groups(self):
        """
        Add freshly-unfrozen parameters to the existing AdamW optimiser,
        respecting the LR ladder. Existing groups keep their state.
        """
        base_lr = self.old_lr
        already_in_optim = {id(p)
                            for g in self.optimizer.param_groups
                            for p in g['params']}

        for gname, ratio in self.lr_ratios.items():
            new_params = [p for p in self._iter_group_params(gname)
                          if id(p) not in already_in_optim]
            if not new_params:
                continue
            self.optimizer.add_param_group({
                'params': new_params,
                'lr': base_lr * ratio,
                'name': gname,              # keep for pretty printing
            })

        # log current LR per group
        for g in self.optimizer.param_groups:
            tag = g.get('name', '<unnamed>')
            print(f"  ├─ {tag:10s}: lr={g['lr']:.2e}")

    # -------------------------------------------------------------------------
    # public API
    # -------------------------------------------------------------------------
    def progressive_unfreeze(self, new_level: int):
        """
        0 → decoder only (default)  
        1 → + encoder4  
        2 → + encoder3  
        3 → + encoder2
        """
        if new_level <= self.current_unfreeze_level:
            return                                  # nothing to do
        new_level = min(new_level, self.max_unfreeze_level)
        self.current_unfreeze_level = new_level

        to_unfreeze = {
            1: ['encoder4'],
            2: ['encoder3'],
            3: ['encoder2'],
        }.get(new_level, [])
        for gname in to_unfreeze:
            for lname in self.layer_groups[gname]:
                layer = getattr(self.net.module, lname, None)
                if layer is None:
                    continue
                for p in layer.parameters():
                    p.requires_grad_(True)

        print(f"✓ Unfroze encoder level {new_level}: {to_unfreeze}")
        self._add_or_update_param_groups()

    # -------------------------------------------------------------------------
    # override: proportional LR update for *all* groups
    # -------------------------------------------------------------------------
    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr

        for g in self.optimizer.param_groups:
            ratio = g['lr'] / self.old_lr
            g['lr'] = new_lr * ratio
        mylog.write(f'update learning rate: {self.old_lr:.6f} → {new_lr:.6f}\n')
        print(f"update learning rate: {self.old_lr:.6f} → {new_lr:.6f}")
        self.old_lr = new_lr


# -----------------------------------------------------------------------------  
# 2.  Training driver (unchanged except bug-fixes)
# -----------------------------------------------------------------------------
def create_data_loader(data_root, batch_size_per_gpu=4):
    """Original dataset logic."""
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    imagelist = [f for f in os.listdir(data_root) if '_sat' in f]
    trainlist = [f.rsplit('_sat', 1)[0] for f in imagelist]

    if not trainlist:
        raise ValueError(f"No '*_sat.*' images in {data_root}")

    dataset = ImageFolder(trainlist, data_root)
    batchsize = torch.cuda.device_count() * batch_size_per_gpu
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batchsize,
                                         shuffle=True,
                                         num_workers=4)
    return loader, len(trainlist)


def train_with_progressive_unfreezing(config):
    """Preserves original train.py flow, adds safe PU hooks."""
    SHAPE = (1024, 1024)
    NAME  = config.get('experiment_name', 'finetune_progressive')
    BATCH_PER_GPU = config.get('batch_size_per_gpu', 4)

    solver = ProgressiveUnfreezingFrame(
        DinkNet34, dice_bce_loss,
        lr=config.get('learning_rate', 2e-4),
        lr_ratios=config.get('lr_ratios'),   # ← allow override
    )

    # ── load weights if present
    if (w_path := config.get('pretrained_weights')) and os.path.exists(w_path):
        print(f"Loading pretrained weights from {w_path}")
        solver.load(w_path)
    else:
        print("⚠ No pretrained weights supplied → training from scratch")

    # ── data
    loader, n_samples = create_data_loader(config['data_root'], BATCH_PER_GPU)
    batchsize = torch.cuda.device_count() * BATCH_PER_GPU

    # ── logging
    os.makedirs(ldir := config.get('log_dir', 'logs'), exist_ok=True)
    os.makedirs(wdir := config.get('weights_dir', 'weights'), exist_ok=True)
    mylog = open(os.path.join(ldir, f'{NAME}.log'), 'w', buffering=1)

    use_wandb = WANDB_AVAILABLE and config.get('use_wandb', False)
    if use_wandb:
        wandb.init(project=config.get('wandb_project', 'dink-finetune'),
                   name=NAME, config=config)

    # ── training loop (original)
    tic = time()
    no_optim = 0
    best_loss = float('inf')
    total_epoch = config.get('total_epochs', 100)
    unfreeze_sched = config.get('unfreeze_schedule', {5:1, 10:2, 15:3})

    print(f"[INFO] samples={n_samples}, batch={batchsize}, "
          f"epochs={total_epoch}, unfreeze={unfreeze_sched}")

    for epoch in range(1, total_epoch + 1):

        if epoch in unfreeze_sched:
            solver.progressive_unfreeze(unfreeze_sched[epoch])

        epoch_loss = 0.0
        for img, mask in loader:
            solver.set_input(img, mask)
            epoch_loss += solver.optimize()

        epoch_loss /= len(loader)               # ← fixed

        # ── logs
        mylog.write(f"epoch {epoch:03d} | time {int(time()-tic)}s | "
                    f"loss {epoch_loss:.6f}\n")
        print(f"epoch {epoch:03d} | loss {epoch_loss:.6f}")

        if use_wandb:
            wandb.log({'epoch': epoch,
                       'train_loss': epoch_loss,
                       'learning_rate': solver.old_lr,
                       'unfreeze_level': solver.current_unfreeze_level,
                       'no_optim': no_optim})

        # ── early-stop / LR-schedule (kept from original)
        if epoch_loss >= best_loss:
            no_optim += 1
        else:
            no_optim = 0
            best_loss = epoch_loss
            solver.save(os.path.join(wdir, f'{NAME}.th'))

        if no_optim > 6:
            print(f"Early stop at epoch {epoch}")
            break
        if no_optim > 3:
            if solver.old_lr < 5e-7:
                break
            solver.load(os.path.join(wdir, f'{NAME}.th'))
            solver.update_lr(5.0, mylog=mylog, factor=True)

    mylog.write("Finish!\n")
    mylog.close()
    if use_wandb:
        wandb.finish()
    print(f"Best training loss: {best_loss:.6f}")


# -----------------------------------------------------------------------------  
# 3.  CLI
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="DinkNet34 progressive unfreeze finetuner")
    p.add_argument('--data_root', required=True)
    p.add_argument('--pretrained_weights',
                   default='weights/log01_dink34.th')
    p.add_argument('--experiment_name', default='finetune_progressive')
    p.add_argument('--learning_rate', type=float, default=2e-4)
    p.add_argument('--batch_size_per_gpu', type=int, default=4)
    p.add_argument('--total_epochs', type=int, default=100)
    p.add_argument('--use_wandb', action='store_true')
    p.add_argument('--wandb_project', default='dlink-rural-roads')
    args = p.parse_args()

    cfg = vars(args) | {
        'log_dir':    'logs',
        'weights_dir':'weights',
        'unfreeze_schedule': {5:1, 10:2, 15:3},
    }
    print("Starting training with config:", cfg)
    train_with_progressive_unfreezing(cfg)


if __name__ == '__main__':
    main()
