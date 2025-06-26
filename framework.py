
import torch, os, math, contextlib
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

class AMPFrame:
    """ Generic training wrapper with AMP, clipping, and param-groups"""
    def __init__(self, net_cls, loss_fn, lr=2e-4):
        self.model = net_cls().cuda()
        self.model = torch.nn.DataParallel(self.model)
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scaler = GradScaler()
        self.clip = 1.0

    def train_batch(self, imgs, masks):
        imgs, masks = imgs.cuda(non_blocking=True), masks.cuda(non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = self.model(imgs)
            loss = self.loss_fn(logits, masks)
        self.scaler.scale(loss).backward()
        clip_grad_norm_(self.model.parameters(), self.clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.detach().item()

    @torch.no_grad()
    def eval_batch(self, imgs, masks):
        self.model.eval()
        with autocast():
            logits = self.model(imgs.cuda(non_blocking=True))
            loss = self.loss_fn(logits, masks.cuda(non_blocking=True))
        self.model.train()
        return loss.detach().item()

    def save(self, path): torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
