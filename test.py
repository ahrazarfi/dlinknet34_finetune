#!/usr/bin/env python3
"""
test_finetune.py – inference for fine-tuned D-LinkNet-34 checkpoints

Features
────────
• reads GeoTIFF/JPG/PNG of any bit-depth
• test-time augmentation (TTA) 1 / 2 / 4 / 8 views
• AMP for speed + lower VRAM
• robust radiometry stretch -> uint8 BGR 1024²
• CLI flags for all important knobs
"""

from __future__ import annotations
import argparse, os, time, cv2, numpy as np, rasterio, torch
from glob import glob
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List
from networks.dinknet import DinkNet34
from torch import nn
from torch.cuda.amp import autocast

# ─────────────────────────── utils ──────────────────────────────────────────
def load_tile(path: str, size: int = 1024) -> np.ndarray:
    """Robust: any dtype -> uint8 BGR, resized/padded to `size`²."""
    if path.lower().endswith((".tif", ".tiff")):
        with rasterio.open(path) as src:
            rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
    else:
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)
        lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1))
        hi = np.where(hi - lo < 1e-6, lo + 1e-6, hi)
        rgb = np.clip((rgb - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if bgr.shape[:2] != (size, size):
        bgr = cv2.resize(bgr, (size, size), cv2.INTER_LINEAR)
    return bgr


def _forward(
    net: nn.Module,
    imgs: np.ndarray,
    div: float = 255.0,
) -> np.ndarray:
    """Run network on N HWC uint8 images -> N HxW probs."""
    t = torch.from_numpy(imgs.transpose(0, 3, 1, 2)).float().cuda()
    t = t / div * 3.2 - 1.6            # DeepGlobe normalisation
    with torch.no_grad(), autocast():
        return torch.sigmoid(net(t)).cpu().numpy()[:, 0]


# ─────────────────────────── TTA kernels ────────────────────────────────────
def tta_1(net: nn.Module, img: np.ndarray) -> np.ndarray:
    return _forward(net, img[None])[0]

def tta_2(net: nn.Module, img: np.ndarray) -> np.ndarray:
    imgs = np.stack([img, np.rot90(img)])
    p = _forward(net, imgs)
    return p[0] + np.rot90(p[1], 3)

def tta_4(net: nn.Module, img: np.ndarray) -> np.ndarray:
    r = np.rot90(img); f = img[:, ::-1]; rf = r[:, ::-1]
    imgs = np.stack([img, r, f, rf])
    p = _forward(net, imgs)
    return (
        p[0] + np.rot90(p[1], 3) +
        p[2][:, ::-1] + np.rot90(p[3], 3)[:, ::-1]
    )

def tta_8(net: nn.Module, img: np.ndarray) -> np.ndarray:
    r = np.rot90(img); f = img[:, ::-1]; rf = r[:, ::-1]
    v = img[::-1]; rv = r[::-1]; fv = f[::-1]; rfv = rf[::-1]
    imgs = np.stack([img, r, f, rf, v, rv, fv, rfv])
    p = _forward(net, imgs)
    aligned = [
        p[0], np.rot90(p[1], 3), p[2][:, ::-1], np.rot90(p[3], 3)[:, ::-1],
        p[4][::-1], np.rot90(p[5], 3)[::-1],
        p[6][::-1, ::-1], np.rot90(p[7], 3)[::-1, ::-1],
    ]
    return sum(aligned)


TTA_FUNCS = {1: tta_1, 2: tta_2, 4: tta_4, 8: tta_8}

# ─────────────────────────── main ───────────────────────────────────────────
def infer(
    model_w: str,
    src_dir: str,
    dst_dir: str,
    tta: int = 4,
    thresh: float = 0.30,
    batch_per_gpu: int = 4,
):
    assert tta in (1, 2, 4, 8), "TTA must be 1/2/4/8"

    net = nn.DataParallel(DinkNet34().cuda())
    net.load_state_dict(torch.load(model_w, map_location="cuda"))
    net.eval()

    os.makedirs(dst_dir, exist_ok=True)
    paths = sorted(
        p for ext in ("*.tif", "*.tiff", "*.png", "*.jpg")
        for p in glob(os.path.join(src_dir, ext))
    )
    print(f"Inferencing {len(paths)} images | TTA={tta} | batch/GPU={batch_per_gpu}")

    t0 = time.time()
    tta_fn = TTA_FUNCS[tta]

    for pth in tqdm(paths, unit="img"):
        img = load_tile(pth)
        summed = tta_fn(net, img)          # SUM already handled inside
        prob  = summed / tta               # average prob
        mask  = (prob > thresh).astype(np.uint8) * 255
        out   = np.stack([mask] * 3, 2)    # BGR for cv2
        cv2.imwrite(
            os.path.join(dst_dir, Path(pth).stem + "_mask.png"),
            out,
        )

    print(f"Done in {time.time()-t0:.1f}s → {dst_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Inference for fine-tuned D-LinkNet-34")
    ap.add_argument("--weights", required=True, help="checkpoint .th")
    ap.add_argument("--source",  required=True, help="image folder")
    ap.add_argument("--dst",     default="out_masks", help="save folder")
    ap.add_argument("--tta",     type=int, default=4, choices=[1,2,4,8],
                    help="test-time augmentation views")
    ap.add_argument("--th",      type=float, default=0.30,
                    help="probability threshold")
    ap.add_argument("--bs",      type=int, default=4,
                    help="virtual batch per GPU (affects AMP/TTA mem)")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    infer(args.weights, args.source, args.dst, args.tta, args.th, args.bs)
