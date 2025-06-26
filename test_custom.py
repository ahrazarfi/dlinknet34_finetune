# ---------------------------------------------------------------------------
#  test.py – inference script for the DeepGlobe-trained DLinkNet34
#            * works on GeoTIFF/JPG/PNG of any bit-depth
#            * automatic test-time augmentation (1 / 2 / 4 / 8 views)
#            * correct probability calibration & thresholding
# ---------------------------------------------------------------------------

import os, time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import rasterio                                     # <-- NEW : robust TIFF I/O

from networks.dinknet import DinkNet34              # same network as training

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change "0" to desired GPU index


# ------------------------------- constants ----------------------------------
BATCHSIZE_PER_CARD = 4
TARGET_SIZE        = 1024     # size of DeepGlobe chips (50 cm/pixel)
THRESHOLD          = 0.3     # probability cutoff → tweak on your own AOI

# --------------------- helper : match DeepGlobe radiometry ------------------
import numpy as np
import cv2, rasterio

def load_like_deepglobe(path: str, target_size: int = 1024) -> np.ndarray:
    """
    Read an arbitrary GeoTIFF/PNG/JPG and return an 8-bit BGR tile that
    mimics DeepGlobe radiometry.  Robust to 32-bit float reflectance.
    """
    # ------------------------------------------------------------------ read
    if path.lower().endswith((".tif", ".tiff")):
        with rasterio.open(path) as src:
            rgb = src.read([1, 2, 3]).transpose(1, 2, 0)   # HWC, any dtype
    else:
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    # ----------------------------------------- dtype-agnostic  →  uint8  BGR
    if rgb.dtype == np.uint8:                # already fine
        bgr8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    else:
        # ------- 1.  convert to float32 for uniform processing
        rgb_f = rgb.astype(np.float32)

        # ------- 2.  compute robust min/max per band (2nd–98th percentile)
        lo = np.percentile(rgb_f,  2, axis=(0, 1))
        hi = np.percentile(rgb_f, 98, axis=(0, 1))
        hi = np.where(hi - lo < 1e-6, lo + 1e-6, hi)       # avoid /0

        # ------- 3.  linear stretch each band to 0-255 and clip
        rgb_stretched = (rgb_f - lo) / (hi - lo) * 255.0
        rgb_stretched = np.clip(rgb_stretched, 0, 255).astype(np.uint8)

        bgr8 = cv2.cvtColor(rgb_stretched, cv2.COLOR_RGB2BGR)

    # --------------------------------------------- resize / pad 1024×1024
    if bgr8.shape[:2] != (target_size, target_size):
        bgr8 = cv2.resize(bgr8, (target_size, target_size),
                          interpolation=cv2.INTER_LINEAR)

    return bgr8            # uint8 , H×W×3  (BGR)

# ---------------------------------------------------------------------------

class TTAFrame:
    """
    Thin wrapper that:
      1. Holds a DataParallel network.
      2. Provides test-time augmentation (TTA) with 1/2/4/8 views.
      3. Keeps track of how many views are summed so we can turn the
         SUM back into a proper probability map.
    """
    def __init__(self, net_cls):
        self.net = net_cls().cuda()
        self.net = nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

        # bs = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        bs = 8
        if bs >= 8:
            self._tester  = self._test8     # 8-view TTA
            self.n_views  = 8
        elif bs >= 4:
            self._tester  = self._test4     # 4-view TTA
            self.n_views  = 4
        elif bs >= 2:
            self._tester  = self._test2     # 2-view TTA
            self.n_views  = 2
        else:
            self._tester  = self._test1     # no TTA
            self.n_views  = 1

    # --------------------------- 1-view (no TTA) ----------------------------
    def _test1(self, img_bgr: np.ndarray) -> np.ndarray:
        img = img_bgr.transpose(2, 0, 1)[None]          # NCHW
        img = (torch.from_numpy(img).float().cuda() / 255. * 3.2 - 1.6)
        with torch.no_grad():
            mask = self.net(img).squeeze().cpu().numpy()  # prob 0-1
        return mask                                       # H×W

    # --------------------------- 2-view (rot90) -----------------------------
    def _test2(self, img_bgr: np.ndarray) -> np.ndarray:
        img90 = np.rot90(img_bgr)                        # CCW 90°
        imgs  = np.stack([img_bgr, img90])               # 2 views
        imgs  = imgs.transpose(0, 3, 1, 2)               # NCHW
        imgs  = (torch.from_numpy(imgs).float().cuda() / 255. * 3.2 - 1.6)
        with torch.no_grad():
            masks = self.net(imgs).squeeze().cpu().numpy()  # 2×H×W prob

        mask_sum = masks[0] + np.rot90(masks[1], k=3)    # rotate back
        return mask_sum                                  # SUM of 2 views

    # --------------------------- 4-view (rot + H-flip) ----------------------
    def _test4(self, img_bgr: np.ndarray) -> np.ndarray:
        img90  = np.rot90(img_bgr)
        img_fl = img_bgr[:, ::-1]                        # horizontal flip
        img90f = img90[:, ::-1]

        imgs   = np.stack([img_bgr, img90, img_fl, img90f])  # 4 views
        imgs   = imgs.transpose(0, 3, 1, 2)
        imgs   = (torch.from_numpy(imgs).float().cuda() / 255. * 3.2 - 1.6)
        with torch.no_grad():
            masks = self.net(imgs).squeeze().cpu().numpy()

        mask_sum = (masks[0] +
                    np.rot90(masks[1], k=3) +           # undo rot90
                    masks[2][:, ::-1] +                 # undo flip
                    np.rot90(masks[3], k=3)[:, ::-1])   # undo both
        return mask_sum                                 # SUM of 4 views

    # --------------------------- 8-view (rot + H/V flips) -------------------
    def _test8(self, img_bgr: np.ndarray) -> np.ndarray:
        r  = np.rot90(img_bgr)
        f  = img_bgr[:, ::-1]
        rf = r[:, ::-1]
        v  = img_bgr[::-1]
        rv = r[::-1]
        fv = f[::-1]
        rfv= rf[::-1]

        imgs = np.stack([img_bgr, r, f, rf, v, rv, fv, rfv])  # 8 views
        imgs = imgs.transpose(0, 3, 1, 2)
        imgs = (torch.from_numpy(imgs).float().cuda() / 255. * 3.2 - 1.6)
        with torch.no_grad():
            masks = self.net(imgs).squeeze().cpu().numpy()

        # align each augmented mask back to original orientation
        aligned = [
            masks[0],                                # original
            np.rot90(masks[1], k=3),                 # rot-back
            masks[2][:, ::-1],                       # H-flip back
            np.rot90(masks[3], k=3)[:, ::-1],        # rot & flip back
            masks[4][::-1],                          # V-flip back
            np.rot90(masks[5], k=3)[::-1],           # rot & V back
            masks[6][::-1, ::-1],                    # H+V back
            np.rot90(masks[7], k=3)[::-1, ::-1]      # rot + H+V back
        ]
        return sum(aligned)                          # SUM of 8 views

    # -------------------------- public interface ----------------------------
    def test(self, img_path: str) -> np.ndarray:
        """Return SUM of probabilities for n_views augmentations."""
        self.net.eval()
        img = load_like_deepglobe(img_path)
        return self._tester(img)                     # H×W SUM

    def load(self, weight_path: str):
        self.net.load_state_dict(torch.load(weight_path, map_location='cuda'))

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    source_dir = "DATA/50cm_patches_1024_1024/163788"
    target_dir = "submits/50cm_patches_1024_1024_163788"
    os.makedirs(target_dir, exist_ok=True)

    solver = TTAFrame(DinkNet34)
    print("Loading model weights...")
    load_start = time.time()
    solver.load("weights/log01_dink34.th")
    print(f"Model loaded in {time.time() - load_start:.1f}s")
    print(f"Model device: {next(solver.net.parameters()).device}")
    print(f"Number of GPUs used: {torch.cuda.device_count()}")

    tic = time.time()
    for i, fname in enumerate(os.listdir(source_dir), 1):
        if not fname.lower().endswith((".tif", ".tiff", ".jpg", ".png")):
            continue
        if i % 10 == 0:
            print(f"[{i:4d}]  elapsed {time.time()-tic:.1f}s")
            
        # Add GPU memory info
        if torch.cuda.is_available():
            print(f"GPU Memory before processing: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            
        img_start = time.time()
        summed = solver.test(os.path.join(source_dir, fname))   # H×W  SUM
        img_time = time.time() - img_start
        print(f"Image {fname} processed in {img_time:.1f}s")
        
        if torch.cuda.is_available():
            print(f"GPU Memory after processing: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            
        prob   = summed / solver.n_views                        # → probability
        
        print(fname, "min/mean/max prob:", prob.min(), prob.mean(), prob.max())
        binary = (prob > THRESHOLD).astype(np.uint8) * 255      # threshold

        out = np.stack([binary] * 3, axis=2)                    # BGR for cv2
        out_name = os.path.splitext(fname)[0] + "_mask.png"
        cv2.imwrite(os.path.join(target_dir, out_name), out)

    print("Done masks saved to", target_dir)
