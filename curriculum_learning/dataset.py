from utils import *
import os
import os.path as osp
import json
import hashlib
import logging
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter

from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage import uniform_filter

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Seeding helpers (reproducibility & multi-worker friendliness)
# -----------------------------------------------------------------------------

def set_global_seed(seed: int = 1234):
    """Set seeds for python, numpy, torch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_worker_init_fn(base_seed: int = 1234):
    """
    Returns a function suitable for DataLoader(worker_init_fn=...). Ensures each worker
    has a distinct, reproducible RNG state derived from base_seed.
    """
    def _init_fn(worker_id: int):
        seed = base_seed + worker_id
        set_global_seed(seed)
        logger.info(f"[worker_init_fn] worker_id={worker_id}, seed={seed}")
    return _init_fn

# -----------------------------------------------------------------------------
# Local metrics: Ring-background SNR & negative-sample noise metric
# -----------------------------------------------------------------------------

def _extract_local_patch(img: np.ndarray, center_y: int, center_x: int, window_size: int) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    H, W = img.shape
    half = window_size // 2
    y0 = max(0, center_y - half)
    y1 = min(H, center_y + half + 1)  # +1 because Python slicing is right-open
    x0 = max(0, center_x - half)
    x1 = min(W, center_x + half + 1)
    return img[y0:y1, x0:x1], (y0, y1, x0, x1)


def compute_local_snr_ring(
    img: np.ndarray,
    mask: np.ndarray,
    window_size: int = 11,
    bg_margin: int = 3,
    use_abs_snr: bool = True,
) -> float:
    """
    Compute local SNR using a ring-shaped background around the target.
    - img: float32 [H, W]
    - mask: float32/uint8 [H, W], >0 indicates target
    - window_size: local patch size (odd preferred)
    - bg_margin: dilation iterations for background ring thickness
    - use_abs_snr: take abs(SNR) to accommodate dark or bright targets
    Returns: scalar SNR (optionally abs)
    """
    if mask.sum() <= 0:
        return 0.0

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0.0

    min_y, max_y = int(ys.min()), int(ys.max())
    min_x, max_x = int(xs.min()), int(xs.max())
    cy = (min_y + max_y) // 2
    cx = (min_x + max_x) // 2

    local, (y0, y1, x0, x1) = _extract_local_patch(img, cy, cx, window_size)
    local_mask = (mask[y0:y1, x0:x1] > 0)

    if not local_mask.any():
        return 0.0

    # ring background via dilation
    if bg_margin > 0:
        st = generate_binary_structure(2, 1)
        dil = binary_dilation(local_mask, structure=st, iterations=bg_margin)
        bg_region = ~dil
    else:
        bg_region = ~local_mask

    bg = local[bg_region]
    if bg.size == 0:
        return 0.0

    tgt = float(local[local_mask].mean())
    bg_mean = float(bg.mean())
    bg_std = float(bg.std())
    if bg_std <= 0:
        return 0.0

    snr = (tgt - bg_mean) / (bg_std + 1e-8)
    return float(abs(snr) if use_abs_snr else snr)


def compute_target_size(mask: np.ndarray, binarize: bool = True, thresh: float = 0.5) -> float:
    if binarize:
        m = (mask > thresh).astype(np.uint8)
    else:
        m = (mask > 0).astype(np.uint8)
    target_pixels = int(m.sum())
    total_pixels = mask.shape[0] * mask.shape[1]
    return float(target_pixels) / float(total_pixels + 1e-8)


def compute_local_noise_std(
    img: np.ndarray,
    window_size: int = 11,
    stride: int = 8,
    summary: str = "median",
) -> float:
    """
    For negative samples (no target), estimate a local background noise level.
    We approximate local std over a coarse grid using integral-image style with uniform_filter.
    summary: one of {"min","median","mean","p25","p75"} to summarize the grid stds.
    """
    # compute local mean & mean of squares
    img = img.astype(np.float32)
    k = window_size
    mu = uniform_filter(img, k)
    mu2 = uniform_filter(img * img, k)
    var = np.maximum(mu2 - mu * mu, 0.0)
    std = np.sqrt(var + 1e-8)

    # downsample by stride to reduce cost
    std_grid = std[::stride, ::stride]

    if summary == "min":
        val = float(np.min(std_grid))
    elif summary == "mean":
        val = float(np.mean(std_grid))
    elif summary == "p25":
        val = float(np.percentile(std_grid, 25))
    elif summary == "p75":
        val = float(np.percentile(std_grid, 75))
    else:  # median
        val = float(np.median(std_grid))
    return val

# -----------------------------------------------------------------------------
# Easiness scoring (unified for positives & negatives) with dataset-level norm
# -----------------------------------------------------------------------------

def normalize_feature(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    z = (x - lo) / (hi - lo)
    return float(np.clip(z, 0.0, 1.0))


def compute_easiness_score(
    img: np.ndarray,
    mask: np.ndarray,
    norm_cfg: Dict[str, float],
    window_size: int = 11,
    bg_margin: int = 3,
    use_abs_snr: bool = True,
    snr_weight: float = 0.88,
    size_weight: float = 0.12,
    neg_window_size: int = 11,
    neg_stride: int = 8,
    neg_summary: str = "median",
) -> Tuple[float, Dict[str, Any]]:
    """
    Returns easiness in [0,1]. Larger means easier.
    Positive samples (mask>0): easiness = weighted normalized SNR & size.
    Negative samples (mask==0): define a noise metric (local std), then easiness = 1 - normalized_noise.
    """
    stats = {}
    if mask.sum() > 0:  # positive
        snr = compute_local_snr_ring(img, mask, window_size=window_size, bg_margin=bg_margin, use_abs_snr=use_abs_snr)
        sz = compute_target_size(mask)
        snr_n = normalize_feature(snr, norm_cfg['snr_min'], norm_cfg['snr_max'])
        sz_n  = normalize_feature(sz,  norm_cfg['size_min'], norm_cfg['size_max'])
        easiness = snr_weight * snr_n + size_weight * sz_n
        stats.update(dict(kind='pos', snr=snr, size=sz, snr_n=snr_n, size_n=sz_n))
    else:  # negative
        noise = compute_local_noise_std(img, window_size=neg_window_size, stride=neg_stride, summary=neg_summary)
        noise_n = normalize_feature(noise, norm_cfg['noise_min'], norm_cfg['noise_max'])
        easiness = 1.0 - noise_n  # lower noise => easier
        stats.update(dict(kind='neg', noise=noise, noise_n=noise_n))

    easiness = float(np.clip(easiness, 0.0, 1.0))
    return easiness, stats

# -----------------------------------------------------------------------------
# Adaptive Curriculum (pace controller)
# -----------------------------------------------------------------------------
class AdaptiveCurriculumLearning:
    def __init__(self,
                 total_epochs: int,
                 initial_pace: float = 0.05,
                 min_pace: float = 0.01,
                 max_pace: float = 0.2,
                 warmup_epochs: int = 5,
                 patience: int = 5,
                 improvement_threshold: float = 0.01,
                 strategy: str = 'hybrid'):
        self.total_epochs = total_epochs
        self.initial_pace = initial_pace
        self.current_pace = initial_pace
        self.min_pace = min_pace
        self.max_pace = max_pace
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.strategy = strategy

        self.performance_history: List[float] = []
        self.loss_history: List[float] = []
        self.difficulty_threshold_history: List[float] = []
        self.pace_history: List[float] = []

        self.stagnant_epochs = 0
        self.best_performance = 0.0
        self.recent_avg_performance = 0.0

    def update_performance(self, current_metric: float, current_loss: Optional[float] = None) -> float:
        self.performance_history.append(current_metric)
        if current_loss is not None:
            self.loss_history.append(current_loss)

        epoch = len(self.performance_history)

        if epoch <= self.warmup_epochs:
            self.pace_history.append(self.current_pace)
            return self.current_pace

        if current_metric > self.best_performance:
            improvement = current_metric - self.best_performance
            self.best_performance = current_metric
            if improvement > self.improvement_threshold and self.strategy in ['performance_based', 'hybrid']:
                self._increase_pace()
                self.stagnant_epochs = 0
            else:
                self.stagnant_epochs += 1
        else:
            self.stagnant_epochs += 1

        recent_window = min(5, len(self.performance_history))
        self.recent_avg_performance = float(np.mean(self.performance_history[-recent_window:]))

        if self.stagnant_epochs >= self.patience and self.strategy in ['performance_based', 'hybrid']:
            self._decrease_pace()
            self.stagnant_epochs = 0

        if current_loss is not None and len(self.loss_history) >= 3:
            self._adjust_pace_by_loss()

        self._adjust_pace_by_progress(epoch)

        self.pace_history.append(self.current_pace)
        return self.current_pace

    def _increase_pace(self):
        old = self.current_pace
        self.current_pace = min(self.current_pace * 1.2, self.max_pace)
        if self.current_pace != old:
            logger.info(f"Curriculum pace increased: {old:.4f} -> {self.current_pace:.4f}")

    def _decrease_pace(self):
        old = self.current_pace
        self.current_pace = max(self.current_pace * 0.8, self.min_pace)
        if self.current_pace != old:
            logger.info(f"Curriculum pace decreased: {old:.4f} -> {self.current_pace:.4f}")

    def _adjust_pace_by_loss(self):
        if len(self.loss_history) < 3:
            return
        recent = self.loss_history[-3:]
        trend = (recent[-1] - recent[0]) / 2.0
        if self.strategy in ['loss_based', 'hybrid']:
            if trend > 0.01:
                self.current_pace = max(self.current_pace * 0.9, self.min_pace)
            elif trend < -0.01:
                self.current_pace = min(self.current_pace * 1.1, self.max_pace)

    def _adjust_pace_by_progress(self, epoch: int):
        progress = epoch / float(self.total_epochs)
        if progress < 0.3:
            target = self.initial_pace * 0.8
        elif progress < 0.7:
            target = self.initial_pace
        else:
            target = self.initial_pace * 1.5
        self.current_pace = float(np.clip(self.current_pace * 0.9 + target * 0.1, self.min_pace, self.max_pace))

# -----------------------------------------------------------------------------
# Dataset with adaptive curriculum & caching
# -----------------------------------------------------------------------------
class AdaptiveCurriculumDataSetLoader(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 dataset_name: str,
                 patch_size: int,
                 mode: str,
                 img_norm_cfg: Optional[Dict[str, Any]] = None,
                 curriculum_learning: bool = False,
                 curriculum_epoch_start: int = 0,
                 curriculum_pace: float = 0.1,
                 total_epochs: int = 400,
                 adaptive_curriculum: bool = True,
                 seed: int = 1234,
                 cache_scores: bool = True,
                 window_size: int = 11,
                 bg_margin: int = 3,
                 use_abs_snr: bool = True,
                 neg_window_size: int = 11,
                 neg_stride: int = 8,
                 neg_summary: str = 'median'):
        super().__init__()  # FIX: correct super call

        set_global_seed(seed)
        self.base_seed = seed
        self.rng = np.random.RandomState(seed)

        self.dataset_name = dataset_name
        dataset_dir = osp.join(dataset_dir, dataset_name)
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size
        self.mode = mode

        self.curriculum_learning = curriculum_learning and mode == 'train'
        self.curriculum_epoch_start = curriculum_epoch_start
        self.current_epoch = 0
        self.current_threshold = 1.0  # for easiness (>= threshold)
        self.adaptive_curriculum = adaptive_curriculum

        self.window_size = window_size
        self.bg_margin = bg_margin
        self.use_abs_snr = use_abs_snr
        self.neg_window_size = neg_window_size
        self.neg_stride = neg_stride
        self.neg_summary = neg_summary

        self.files: List[str] = []
        if mode == 'train':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        elif mode == 'val':
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        elif mode == 'test':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        else:
            raise NotImplementedError
        logger.info(f"{len(self.files)} samples from {dataset_name} for {mode}")

        self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir) if img_norm_cfg is None else img_norm_cfg
        self.transform = augumentation()

        # Curriculum strategy
        if self.curriculum_learning and self.adaptive_curriculum:
            self.curriculum_strategy = AdaptiveCurriculumLearning(
                total_epochs=total_epochs,
                initial_pace=curriculum_pace,
                min_pace=0.005,
                max_pace=0.2,
                warmup_epochs=max(5, curriculum_epoch_start),
                patience=8,
                improvement_threshold=0.005,
                strategy='hybrid'
            )
        else:
            self.curriculum_strategy = None
            self.curriculum_pace = curriculum_pace

        # Precompute and cache easiness scores
        self.cache_scores = cache_scores
        self.scores_cache_path = osp.join(self.dataset_dir, f"curriculum_scores_{dataset_name}.npz")

        self.easiness_scores: List[float] = []
        self.valid_indices: List[int] = []  # indices into self.files for samples we computed
        self.norm_cfg: Dict[str, float] = {}
        self.sample_meta: List[Dict[str, Any]] = []  # for debugging/analysis

        if self.curriculum_learning:
            self._prepare_easiness_scores()

        # Histories for visualization
        self.threshold_history: List[float] = []

    # ----------------------- IO helpers -----------------------
    def _hash_settings(self) -> str:
        payload = {
            'files': self.files,
            'window_size': self.window_size,
            'bg_margin': self.bg_margin,
            'use_abs_snr': self.use_abs_snr,
            'neg_window_size': self.neg_window_size,
            'neg_stride': self.neg_stride,
            'neg_summary': self.neg_summary,
            'img_norm_cfg': self.img_norm_cfg,
        }
        blob = json.dumps(payload, sort_keys=True).encode('utf-8')
        return hashlib.md5(blob).hexdigest()

    def _load_cache(self) -> bool:
        if not (self.cache_scores and osp.exists(self.scores_cache_path)):
            return False
        try:
            data = np.load(self.scores_cache_path, allow_pickle=True)
            if data.get('settings_hash', None) is None:
                return False
            if str(data['settings_hash']) != self._hash_settings():
                return False
            self.easiness_scores = data['easiness_scores'].tolist()
            self.valid_indices = data['valid_indices'].tolist()
            self.norm_cfg = data['norm_cfg'].item()
            self.sample_meta = data['sample_meta'].tolist()
            logger.info(f"Loaded cached easiness scores: {len(self.easiness_scores)} samples from {self.scores_cache_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False

    def _save_cache(self):
        if not self.cache_scores:
            return
        try:
            np.savez_compressed(
                self.scores_cache_path,
                settings_hash=self._hash_settings(),
                easiness_scores=np.array(self.easiness_scores, dtype=np.float32),
                valid_indices=np.array(self.valid_indices, dtype=np.int32),
                norm_cfg=self.norm_cfg,
                sample_meta=np.array(self.sample_meta, dtype=object),
            )
            logger.info(f"Saved easiness scores cache to {self.scores_cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    # ----------------------- Scoring preparation -----------------------
    def _prepare_easiness_scores(self):
        if self._load_cache():
            return

        logger.info("Computing easiness scores (positives + negatives) with dataset-level normalization...")

        # 1) quick sampling to estimate percentiles
        sample_snrs, sample_sizes, sample_noises = [], [], []
        sample_count = min(100, len(self.files))
        if sample_count == 0:
            logger.warning("No files to score.")
            return
        sample_indices = self.rng.choice(len(self.files), size=sample_count, replace=False)

        for idx in sample_indices:
            try:
                filename = self.files[idx]
                img, mask = self._load_image_and_mask(filename)
                img_array = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
                mask_array = np.array(mask, dtype=np.float32) / 255.0
                if mask_array.ndim > 2:
                    mask_array = mask_array[:, :, 0]

                if mask_array.sum() > 0:
                    snr = compute_local_snr_ring(img_array, mask_array, self.window_size, self.bg_margin, self.use_abs_snr)
                    sz  = compute_target_size(mask_array)
                    sample_snrs.append(abs(snr))
                    sample_sizes.append(sz)
                else:
                    noise = compute_local_noise_std(img_array, self.neg_window_size, self.neg_stride, self.neg_summary)
                    sample_noises.append(noise)
            except Exception as e:
                logger.warning(f"[sample scan] {self.files[idx]} failed: {e}")

        # 2) dataset-level normalization config via percentiles (5-95)
        def q(arr, p, default):
            return float(np.percentile(arr, p)) if len(arr) > 0 else default

        snr_min = q(sample_snrs, 5, 0.0)
        snr_max = q(sample_snrs, 95, 10.0)
        size_min, size_max = 0.0, q(sample_sizes, 95, 0.01)
        noise_min = q(sample_noises, 5, 0.0)
        noise_max = q(sample_noises, 95, 10.0)

        self.norm_cfg = dict(
            snr_min=snr_min, snr_max=snr_max,
            size_min=size_min, size_max=size_max,
            noise_min=noise_min, noise_max=noise_max,
        )
        logger.info(f"Norm percentiles: SNR[{snr_min:.4f},{snr_max:.4f}], SIZE[0,{size_max:.6f}], NOISE[{noise_min:.4f},{noise_max:.4f}]")

        # 3) score all samples (positives & negatives)
        easiness_scores, valid_indices, metas = [], [], []
        for idx, filename in enumerate(self.files):
            try:
                img, mask = self._load_image_and_mask(filename)
                img_array = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
                mask_array = np.array(mask, dtype=np.float32) / 255.0
                if mask_array.ndim > 2:
                    mask_array = mask_array[:, :, 0]

                ease, stats = compute_easiness_score(
                    img_array, mask_array, self.norm_cfg,
                    window_size=self.window_size, bg_margin=self.bg_margin, use_abs_snr=self.use_abs_snr,
                    snr_weight=0.88, size_weight=0.12,
                    neg_window_size=self.neg_window_size, neg_stride=self.neg_stride, neg_summary=self.neg_summary,
                )
                easiness_scores.append(ease)
                valid_indices.append(idx)
                metas.append({"file": filename, **stats})
            except Exception as e:
                logger.warning(f"[score all] {filename} failed: {e}")

        if len(easiness_scores) == 0:
            logger.warning("No valid samples scored for curriculum.")
            return

        # sort by easiness ascending just for consistent ordering (not strictly needed)
        pairs = sorted(zip(easiness_scores, valid_indices, metas), key=lambda x: x[0])
        self.easiness_scores = [p[0] for p in pairs]
        self.valid_indices = [p[1] for p in pairs]
        self.sample_meta = [p[2] for p in pairs]

        logger.info(f"Computed easiness for {len(self.valid_indices)} samples: range {min(self.easiness_scores):.3f} - {max(self.easiness_scores):.3f}")
        self._save_cache()

    # ----------------------- Epoch-wise threshold update -----------------------
    def update_curriculum_with_metrics(self, current_metric: float, current_loss: Optional[float] = None) -> Optional[float]:
        if self.curriculum_strategy is not None:
            current_pace = self.curriculum_strategy.update_performance(current_metric, current_loss)
            return current_pace
        return None

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        if self.curriculum_learning and epoch >= self.curriculum_epoch_start and len(self.easiness_scores) > 0:
            if self.curriculum_strategy is not None:
                current_pace = self.curriculum_strategy.current_pace
            else:
                current_pace = self.curriculum_pace

            # progress in [0,1]
            progress = min(1.0, (epoch - self.curriculum_epoch_start) * current_pace)

            # Easiness threshold from high -> low over time? We want >= threshold to include easy first then harder.
            # So threshold should start high and decrease over time.
            e_min, e_max = self.easiness_scores[0], self.easiness_scores[-1]
            threshold = e_max - progress * (e_max - e_min)  # decreases with progress
            self.current_threshold = float(threshold)
            self.threshold_history.append(self.current_threshold)

            available = sum(1 for e in self.easiness_scores if e >= self.current_threshold)
            logger.info(f"Epoch {epoch}: Using {available}/{len(self.valid_indices)} samples (easiness >= {self.current_threshold:.3f}, pace={current_pace:.4f})")

            # also push into strategy history for unified logging/plotting
            if self.curriculum_strategy is not None:
                self.curriculum_strategy.difficulty_threshold_history.append(self.current_threshold)

    # ----------------------- Sampling under curriculum -----------------------
    def _get_curriculum_sample_index(self) -> int:
        if (not self.curriculum_learning) or (self.current_epoch < self.curriculum_epoch_start) or (len(self.easiness_scores) == 0):
            return int(self.rng.randint(0, len(self.files)))

        # available set: easiness >= threshold (easy to hard)
        curr_idx = [i for i, e in enumerate(self.easiness_scores) if e >= self.current_threshold]
        if not curr_idx:
            # fallback: pick the easiest one (largest easiness -> last index after sorting asc)
            curr_idx = [len(self.easiness_scores) - 1]

        selected = int(curr_idx[self.rng.randint(0, len(curr_idx))])
        original_idx = self.valid_indices[selected]
        return original_idx

    # ----------------------- IO: load image & mask -----------------------
    def _load_image_and_mask(self, stem: str) -> Tuple[Image.Image, Image.Image]:
        # prefer .png else fallback to .bmp
        img_png = osp.join(self.dataset_dir, 'images', stem + '.png')
        msk_png = osp.join(self.dataset_dir, 'masks', stem + '.png')
        if osp.exists(img_png) and osp.exists(msk_png):
            img = Image.open(img_png).convert('I')
            mask = Image.open(msk_png)
            return img, mask
        img_bmp = osp.join(self.dataset_dir, 'images', stem + '.bmp')
        msk_bmp = osp.join(self.dataset_dir, 'masks', stem + '.bmp')
        img = Image.open(img_bmp).convert('I')
        mask = Image.open(msk_bmp)
        return img, mask

    # ----------------------- PyTorch Dataset API -----------------------
    def __getitem__(self, idx: int):
        if self.mode == 'train' and self.curriculum_learning:
            actual_idx = self._get_curriculum_sample_index()
        else:
            actual_idx = idx % len(self.files)

        img, mask = self._load_image_and_mask(self.files[actual_idx])
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if mask.ndim > 2:
            mask = mask[:, :, 0]

        if self.mode == 'train':
            img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)
            img_patch, mask_patch = self.transform(img_patch, mask_patch)

            img_patch = img_patch[np.newaxis, :]
            mask_patch = mask_patch[np.newaxis, :]
            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
            return img_patch, mask_patch, 0, 0
        else:
            h, w = img.shape
            img = PadImg(img)
            mask = PadImg(mask)
            img = img[np.newaxis, :]
            mask = mask[np.newaxis, :]
            img = torch.from_numpy(np.ascontiguousarray(img))
            mask = torch.from_numpy(np.ascontiguousarray(mask))
            return img, mask, [h, w], self.files[actual_idx]

    def __len__(self) -> int:
        return len(self.files)

# -----------------------------------------------------------------------------
# Convenience exports
# -----------------------------------------------------------------------------
__all__ = [
    'AdaptiveCurriculumDataSetLoader',
    'AdaptiveCurriculumLearning',
    'set_global_seed',
    'get_worker_init_fn',
    'compute_local_snr_ring',
    'compute_local_noise_std',
    'compute_target_size',
    'compute_easiness_score',
]



class CurriculumDataSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, mode, img_norm_cfg=None, 
                 curriculum_learning=True, curriculum_epoch_start=0, curriculum_pace=0.1):
        super(CurriculumDataSetLoader).__init__()
        self.dataset_name = dataset_name
        dataset_dir = osp.join(dataset_dir, dataset_name)
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size
        self.mode = mode
        
        # 课程学习参数
        self.curriculum_learning = curriculum_learning and mode == 'train'
        self.curriculum_epoch_start = curriculum_epoch_start
        self.curriculum_pace = curriculum_pace
        self.current_epoch = 10
        self.current_difficulty_threshold = 0.0  # 从最简单的开始

        """  # 新增：区分基础进度和当前进度
        self.base_pace = curriculum_pace  # 基础进度（初始值）
        self.current_pace = curriculum_pace  # 当前进度（动态调整） """

        self.files = []
        if mode == 'train':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        elif mode == 'val':
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        elif mode == 'test':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        else:
            raise NotImplementedError
            
        print(f'{len(self.files)} samples from {dataset_name} for {mode}')

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.transform = augumentation()
        
        # 预计算所有样本的难度分数（仅在训练模式和启用课程学习时）
        if self.curriculum_learning:
            self._compute_difficulty_scores()
    
    def _compute_difficulty_scores(self):
        """预计算所有样本的难度分数"""
        print("Computing difficulty scores for curriculum learning...")
        self.difficulty_scores = []
        self.valid_indices = []
        
        for idx, filename in enumerate(self.files):
            try:
                # 加载图像和掩码
                try:
                    img = Image.open(osp.join(self.dataset_dir, 'images', filename + '.png')).convert('I')
                    mask = Image.open(osp.join(self.dataset_dir, 'masks', filename + '.png'))
                except:
                    img = Image.open(osp.join(self.dataset_dir, 'images', filename + '.bmp')).convert('I')
                    mask = Image.open(osp.join(self.dataset_dir, 'masks', filename + '.bmp'))
                
                img_array = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
                mask_array = np.array(mask, dtype=np.float32) / 255.0
                if len(mask_array.shape) > 2:
                    mask_array = mask_array[:,:,0]
                
                # 计算难度分数
                if mask_array.sum() > 0:  # 只考虑有目标的样本
                    difficulty = compute_difficulty_score(img_array, mask_array)
                    self.difficulty_scores.append(difficulty)
                    self.valid_indices.append(idx)
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        # 根据难度分数排序
        sorted_pairs = sorted(zip(self.difficulty_scores, self.valid_indices))
        self.difficulty_scores, self.valid_indices = zip(*sorted_pairs)
        self.difficulty_scores = list(self.difficulty_scores)
        self.valid_indices = list(self.valid_indices)
        
        print(f"Computed difficulty scores for {len(self.valid_indices)} valid samples")
        print(f"Difficulty range: {min(self.difficulty_scores):.3f} - {max(self.difficulty_scores):.3f}")

    # 新增：用于动态调整的变量
        #self.current_epoch = 0
        #self.current_difficulty_threshold = 0.0
        #self.val_metrics = []  # 记录历史验证指标
        #self.best_val_metric = 0.0  # 最佳验证指标
        #self.patience = 3  # 连续多少epoch无提升则减慢进度 
        
    
    def set_epoch(self, epoch,val_metric=None):
        self.current_epoch = epoch
        if self.curriculum_learning and epoch >= self.curriculum_epoch_start:
            # 随着训练进行，逐渐增加难度阈值
            progress = min(1.0, (epoch - self.curriculum_epoch_start) * self.curriculum_pace)
            max_difficulty = max(self.difficulty_scores) if self.difficulty_scores else 1.0
            self.current_difficulty_threshold = progress * max_difficulty
            
            # 计算当前可用的样本数量
            available_samples = sum(1 for score in self.difficulty_scores 
                                  if score <= self.current_difficulty_threshold)
            print(f"Epoch {epoch}: Using {available_samples}/{len(self.valid_indices)} samples "
                  f"(difficulty <= {self.current_difficulty_threshold:.3f})")
        
        """修改：添加动态进度调整逻辑"""
        """ self.current_epoch = epoch
        if not (self.curriculum_learning and epoch >= self.curriculum_epoch_start):
            return
        
        # 新增：根据验证指标调整进度
        if val_metric is not None:
            self.val_metrics.append(val_metric)
            # 若指标提升超过2%，加快进度
            if val_metric > self.best_val_metric * 1.02:
                self.best_val_metric = val_metric
                self.current_pace = min(self.base_pace * 1.5, 0.5)  # 限制最大进度
            else:
                # 连续patience个epoch无提升，减慢进度
                if len(self.val_metrics) >= self.patience:
                    recent_metrics = self.val_metrics[-self.patience:]
                    if all(m <= self.best_val_metric for m in recent_metrics):
                        self.current_pace = max(self.base_pace * 0.5, 0.01)  # 限制最小进度
        
        # 基于当前进度计算难度阈值（核心逻辑修改）
        progress = min(1.0, (epoch - self.curriculum_epoch_start) * self.current_pace)
        max_difficulty = max(self.difficulty_scores) if self.difficulty_scores else 1.0
        self.current_difficulty_threshold = progress * max_difficulty
        
        # 新增：打印进度信息（方便调试）
        available_samples = sum(1 for s in self.difficulty_scores if s <= self.current_difficulty_threshold)
        print(f"课程进度调整：Epoch {epoch} | 当前进度速率={self.current_pace:.3f} | "
              f"难度阈值={self.current_difficulty_threshold:.3f} | 可用样本={available_samples}/{len(self.valid_indices)}")
     """
    def _get_curriculum_sample_index(self):
        """根据课程学习策略选择样本索引"""
        if not self.curriculum_learning or self.current_epoch < self.curriculum_epoch_start:
            # 不使用课程学习，随机选择
            return random.randint(0, len(self.files) - 1)
        
        # 找到满足当前难度阈值的样本
        available_indices = [i for i, score in enumerate(self.difficulty_scores) 
                           if score <= self.current_difficulty_threshold]
        
        if not available_indices:
            # 如果没有满足条件的样本，使用最简单的样本
            available_indices = [0]
        
        # 从可用样本中随机选择
        selected_curriculum_idx = random.choice(available_indices)
        original_idx = self.valid_indices[selected_curriculum_idx]
        
        return original_idx

    def __getitem__(self, idx):
        # 根据课程学习策略选择实际的样本索引
        if self.mode == 'train' and self.curriculum_learning:
            actual_idx = self._get_curriculum_sample_index()
        else:
            actual_idx = idx % len(self.files)
        
        try:
            img = Image.open(osp.join(self.dataset_dir, 'images', self.files[actual_idx] + '.png')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', self.files[actual_idx] + '.png'))
        except:
            img = Image.open(osp.join(self.dataset_dir, 'images', self.files[actual_idx] + '.bmp')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', self.files[actual_idx] + '.bmp'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
            
        if self.mode == 'train':
            img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5) 
            img_patch, mask_patch = self.transform(img_patch, mask_patch)
            
            img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

            return img_patch, mask_patch, 0, 0
        else:
            h, w = img.shape
            img = PadImg(img)
            mask = PadImg(mask)
            
            img, mask = img[np.newaxis,:], mask[np.newaxis,:]
            
            img = torch.from_numpy(np.ascontiguousarray(img))
            mask = torch.from_numpy(np.ascontiguousarray(mask))
            return img, mask, [h, w], self.files[actual_idx]
    
    def __len__(self):
        return len(self.files)


class IRSTD_Dataset(Dataset):
    def __init__(self, dataset, params={}, mode='train', root='/home/pc/work/BasicIRSTD-main/data'):
        
        dataset_dir = osp.join(root, dataset)

        self.imgs_dir = osp.join(dataset_dir, 'images')
        self.label_dir = osp.join(dataset_dir, 'masks')
        self.names = []

        if mode == 'train':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset}.txt'), 'r') as f:
                self.names += [line.strip() for line in f.readlines()]
        elif mode == 'val':
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset}.txt'), 'r') as f:
                self.names += [line.strip() for line in f.readlines()]
        elif mode == 'test':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset}.txt'), 'r') as f:
                self.names += [line.strip() for line in f.readlines()]
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset}.txt'), 'r') as f:
                self.names += [line.strip() for line in f.readlines()]
        else:
            raise NotImplementedError
        
        print(f'{len(self.names)} samples from {dataset} for {mode}')

        self.mode = mode
        self.crop_size = params.crop_size
        self.base_size = params.base_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        else:
            img, mask = self._testval_sync_transform(img, mask)

        
        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, mask

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask

class DataSetLoader_Mixdecoder(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, mode, img_norm_cfg=None):
        super(DataSetLoader_Mixdecoder).__init__()
        self.dataset_name = dataset_name
        dataset_dir = osp.join(dataset_dir, dataset_name)
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size

        self.mode = mode

        self.files = []
        if mode == 'train':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]   # self.train_list = f.read().splitlines()
        elif mode == 'val':
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        elif mode == 'test':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        else:
            raise NotImplementedError
        print(f'{len(self.files)} samples from {dataset_name} for {mode}')

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
    
    def generate_contrastive_mask(self, mask0, region_size=16, mask_ratio=0.2):  # mask0: [C, H, W]
        dh, dw = int(mask0.shape[0] // region_size), int(mask0.shape[0] // region_size)
        mask = np.zeros(dh * dw, dtype=np.float32)  
        mask[:round(dh * dw * mask_ratio)] = 1
        np.random.shuffle(mask)

        mask = mask.reshape(dh, dw)
        mask = mask.repeat(region_size, axis=0).repeat(region_size, axis=1)

        # 膨胀
        mask0 = nn.functional.max_pool2d(torch.from_numpy(mask0).unsqueeze(0), kernel_size=7, stride=1, padding=3).numpy()[0]

        mask[mask0 == 1] = 0  # positive 目标区域不进行mask
        return mask, mask0
    
    def __getitem__(self, idx):
        try:
            img = Image.open(osp.join(self.dataset_dir, 'images', self.files[idx] + '.png')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', self.files[idx] + '.png'))
        except:
            img = Image.open(osp.join(self.dataset_dir, 'images', self.files[idx] + '.bmp')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', self.files[idx] + '.bmp'))


        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
            
        if self.mode == 'train':
            img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5) 
            img_patch, mask_patch = self.tranform(img_patch, mask_patch)
            
            img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

            filename = self.files[idx]
            if filename.startswith('XDU'):
                branch_id = 0
            elif filename.startswith('Misc'):
                branch_id = 1
            else:  # 默认纯数字
                branch_id = 2
            return img_patch, mask_patch,branch_id ,0

        else:
            h, w = img.shape
            img = PadImg(img)
            mask = PadImg(mask)
            
            img, mask = img[np.newaxis,:], mask[np.newaxis,:]
            
            img = torch.from_numpy(np.ascontiguousarray(img))
            mask = torch.from_numpy(np.ascontiguousarray(mask))
            if self.dataset_name == 'IRSTD-1K':
                branch_id = 0
            elif self.dataset_name == 'NUAA-SIRST':
                branch_id = 1
            elif self.dataset_name == 'NUDT-SIRST':
                branch_id = 2
            else:
                branch_id = -1  # 其它情况
            return img, mask, [h, w], self.files[idx], branch_id
    
    def __len__(self):
        return len(self.files)



class DataSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, mode, img_norm_cfg=None):
        super(DataSetLoader).__init__()
        self.dataset_name = dataset_name
        dataset_dir = osp.join(dataset_dir, dataset_name)
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size

        self.mode = mode

        self.files = []
        if mode == 'train':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]   # self.train_list = f.read().splitlines()
        elif mode == 'val':
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        elif mode == 'test':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        else:
            raise NotImplementedError
        print(f'{len(self.files)} samples from {dataset_name} for {mode}')

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
    
    def generate_contrastive_mask(self, mask0, region_size=16, mask_ratio=0.2):  # mask0: [C, H, W]
        dh, dw = int(mask0.shape[0] // region_size), int(mask0.shape[0] // region_size)
        mask = np.zeros(dh * dw, dtype=np.float32)  
        mask[:round(dh * dw * mask_ratio)] = 1
        np.random.shuffle(mask)

        mask = mask.reshape(dh, dw)
        mask = mask.repeat(region_size, axis=0).repeat(region_size, axis=1)

        # 膨胀
        mask0 = nn.functional.max_pool2d(torch.from_numpy(mask0).unsqueeze(0), kernel_size=7, stride=1, padding=3).numpy()[0]

        mask[mask0 == 1] = 0  # positive 目标区域不进行mask
        return mask, mask0
    
    # def generate_contrastive_mask(self,mask0, region_size=16, mask_ratio=0.2):  # mask0: [C, H, W]
    #     # 获取 mask0 的形状
    #     H, W = mask0.shape
        
    #     # 计算 dh 和 dw，使用 np.ceil() 以处理非整除的情况
    #     dh = int(np.ceil(H / region_size))
    #     dw = int(np.ceil(W / region_size))

    #     mask = np.zeros(dh * dw, dtype=np.float32)
    #     mask[:round(dh * dw * mask_ratio)] = 1
    #     np.random.shuffle(mask)

    #     mask = mask.reshape(dh, dw)
    #     mask = np.repeat(mask, region_size, axis=0)[:H, :]
    #     mask = np.repeat(mask, region_size, axis=1)[:, :W]

    #     # 膨胀
    #     mask0 = nn.functional.max_pool2d(torch.from_numpy(mask0).unsqueeze(0), kernel_size=7, stride=1, padding=3).numpy()[0]

    #     mask[mask0 == 1] = 0   # positive 目标区域不进行mask
    #     return mask, mask0
    
    def __getitem__(self, idx):
        try:
            img = Image.open(osp.join(self.dataset_dir, 'images', self.files[idx] + '.png')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', self.files[idx] + '.png'))
        except:
            img = Image.open(osp.join(self.dataset_dir, 'images', self.files[idx] + '.bmp')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', self.files[idx] + '.bmp'))

        # print(self.files[idx])  # check for randomness, passed

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        # img = np.array(img, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
            
        if self.mode == 'train':
            img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5) 
            img_patch, mask_patch = self.tranform(img_patch, mask_patch)

            # # ====================================================================================
            # construct contrastive masking strategy
            # 1. 根据固定比例随机生成基于16*16 / 32*32 / 8*8的mask索引, 创建mask矩阵
            # 2. 构建正样本：将masking区域置为图像的均值, gt部分保留, label = label;
            # 3. 构建负样本：将gt区域膨胀5个像素点(基本充分掩盖目标), 将膨胀后的gt区域置为图像的均值, label = 0. 
            # if self.dataset_name == 'IRSTD-1K':
            #     mask_positive, mask_negative = self.generate_contrastive_mask(mask_patch.copy(), region_size=16, mask_ratio=0.2)
            # elif self.dataset_name == 'NUAA-SIRST':
            #     mask_positive, mask_negative = self.generate_contrastive_mask(mask_patch.copy(), region_size=16, mask_ratio=0.05)
            # else:    
            #     mask_positive, mask_negative = self.generate_contrastive_mask(mask_patch.copy(), region_size=48, mask_ratio=0.05)
    #         mask_positive, mask_negative = self.generate_contrastive_mask(mask_patch.copy(), region_size=16, mask_ratio=0.2)
    # # # === 计算局部块均值 ===
    #         region = 16
    #         h, w = img_patch.shape
    #         dh, dw = h // region, w // region

    #         # reshape + mean over patches
    #         patch_means = img_patch.reshape(dh, region, dw, region).mean(axis=(1, 3))  # shape: (dh, dw)

    #         # 恢复为原图大小，每个 patch 的均值填满16x16区域
    #         mean_full = np.repeat(np.repeat(patch_means, region, axis=0), region, axis=1)  # shape: (H, W)

    #         # === 构造正负样本 ===
    #         img_positive = img_patch.copy()
    #         img_positive[mask_positive == 1] = mean_full[mask_positive == 1]

    #         img_negative = img_patch.copy()
    #         img_negative[mask_negative == 1] = mean_full[mask_negative == 1]


    #         # fill_ = img_patch.mean()
    #         # fill_ = 0
    #         # img_positive = img_patch.copy()
    #         # img_positive[mask_positive == 1] = fill_
    #         # img_negative = img_patch.copy()
    #         # img_negative[mask_negative == 1] = fill_

    #         # import cv2
    #         # cv2.imwrite('img.png', img_patch.astype(np.uint8))
    #         # cv2.imwrite('img_pos.png', img_positive.astype(np.uint8))
    #         # cv2.imwrite('img_neg.png', img_negative.astype(np.uint8))

    #         img_positive, img_negative = img_positive[np.newaxis, :], img_negative[np.newaxis, :]
    #         img_positive = torch.from_numpy(np.ascontiguousarray(img_positive))
    #         img_negative = torch.from_numpy(np.ascontiguousarray(img_negative))

    #         # # # ====================================================================================
            
            img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

            return img_patch, mask_patch,0 ,0
            #return img_patch, mask_patch, img_positive, img_negative
        else:
            h, w = img.shape
            img = PadImg(img)
            mask = PadImg(mask)
            
            img, mask = img[np.newaxis,:], mask[np.newaxis,:]
            
            img = torch.from_numpy(np.ascontiguousarray(img))
            mask = torch.from_numpy(np.ascontiguousarray(mask))
            return img, mask, [h, w], self.files[idx]
    
    def __len__(self):
        return len(self.files)




class DataSetDomainLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, mode, img_norm_cfg=None):
        super(DataSetDomainLoader).__init__()
        self.dataset_name = dataset_name 
        dataset_dir = osp.join(dataset_dir, dataset_name)
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size

        self.mode = mode

        self.files = []
        if mode == 'train':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]   # self.train_list = f.read().splitlines()
                self.img_norm_cfg_train = get_img_norm_cfg(dataset_name, dataset_dir)
        elif mode == 'val':
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
                self.img_norm_cfg_test = get_img_norm_cfg(dataset_name, dataset_dir)
        elif mode == 'test':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        else:
            raise NotImplementedError
        print(f'{len(self.files)} samples from {dataset_name} for {mode}')

        # if img_norm_cfg == None:
        #     self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        # else:
        #     self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
    

    def __getitem__(self, idx):
        try:
            img = Image.open(osp.join(self.dataset_dir, 'images', self.files[idx] + '.png')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', self.files[idx] + '.png'))
        except:
            img = Image.open(osp.join(self.dataset_dir, 'images', self.files[idx] + '.bmp')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', self.files[idx] + '.bmp'))


        # img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        # mask = np.array(mask, dtype=np.float32)  / 255.0
        # if len(mask.shape) > 2:
        #     mask = mask[:,:,0]
            
        if self.mode == 'train':
            img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg_train)
            mask = np.array(mask, dtype=np.float32)  / 255.0
            if len(mask.shape) > 2:
                mask = mask[:,:,0]
            img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5) 
            img_patch, mask_patch = self.tranform(img_patch, mask_patch)
            
            img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

            return img_patch, mask_patch,0 ,0
        
        else:
            img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg_test)
            mask = np.array(mask, dtype=np.float32)  / 255.0
            if len(mask.shape) > 2:
                mask = mask[:,:,0]
            h, w = img.shape
            img = PadImg(img)
            mask = PadImg(mask)
            
            img, mask = img[np.newaxis,:], mask[np.newaxis,:]
            
            img = torch.from_numpy(np.ascontiguousarray(img))
            mask = torch.from_numpy(np.ascontiguousarray(mask))
            return img, mask, [h, w], self.files[idx]
    
    def __len__(self):
        return len(self.files)



import math
class MixedDomainBatchLoader(Dataset):
    def __init__(self, syn_dataset, real_dataset, syn_ratio=0.75, batchsize=16):
        self.syn_dataset = syn_dataset
        self.real_dataset = real_dataset
        self.syn_ratio = syn_ratio
        self.real_ratio = 1.0 - syn_ratio

        self.syn_len = len(syn_dataset)
        self.real_len = len(real_dataset)
        self.batch_size = batchsize

        # 每 batch 中合成/真实样本数
        self.b_syn = int(self.syn_ratio * self.batch_size)
        self.b_real = self.batch_size - self.b_syn

        # 保证每类样本至少用一遍，计算总 batch 数
        self.num_batches = max(
            math.ceil(self.syn_len / self.b_syn),
            math.ceil(self.real_len / self.b_real)
        )
        self.total_len = self.num_batches * self.batch_size

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        batch_position = idx % self.batch_size
        if batch_position < self.b_syn:
            sample_idx = (idx // self.batch_size * self.b_syn + batch_position) % self.syn_len
            img, mask, _, _ = self.syn_dataset[sample_idx]
            domain_label = 0
        else:
            sample_idx = (idx // self.batch_size * self.b_real + (batch_position - self.b_syn)) % self.real_len
            img, mask, _, _ = self.real_dataset[sample_idx]
            domain_label = 1

        return img, mask, domain_label

class BalancedPairedDomainDataset(Dataset):
    def __init__(self, syn_dataset, real_dataset):
        """
        合成和真实数据按1:1配对加载。
        较小数据集将被重复采样以匹配大数据集。
        """
        assert len(syn_dataset) > 0 and len(real_dataset) > 0, "两个数据集都不能为空！"
        self.syn_dataset = syn_dataset
        self.real_dataset = real_dataset

        self.syn_len = len(syn_dataset)
        self.real_len = len(real_dataset)

        self.dataset_len = max(self.syn_len, self.real_len)

        self.set_epoch(0)

    def set_epoch(self, epoch):
        random.seed(epoch)
        # 确定长短
        if self.syn_len >= self.real_len:
            self.long_dataset = self.syn_dataset
            self.short_dataset = self.real_dataset
            self.long_indices = list(range(self.syn_len))  # 不重复
            self.short_indices = self._build_balanced_indices(self.real_len, self.syn_len)
        else:
            self.long_dataset = self.real_dataset
            self.short_dataset = self.syn_dataset
            self.long_indices = list(range(self.real_len))
            self.short_indices = self._build_balanced_indices(self.syn_len, self.real_len)

        # 打乱长短索引顺序
        random.shuffle(self.long_indices)
        random.shuffle(self.short_indices)

    def _build_balanced_indices(self, short_len, target_len):
        # 每个样本至少重复 floor(n) 次
        base_repeats = target_len // short_len
        remainder = target_len % short_len

        indices = list(range(short_len)) * base_repeats
        indices += random.sample(range(short_len), remainder)
        return indices

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        long_item = self.long_dataset[self.long_indices[idx]]
        short_item = self.short_dataset[self.short_indices[idx]]

        # 按原始配对顺序返回
        if self.syn_len >= self.real_len:
            syn_img, syn_mask, *_ = long_item
            real_img, real_mask, *_ = short_item
        else:
            syn_img, syn_mask, *_ = short_item
            real_img, real_mask, *_ = long_item

        return {
            'syn_img': syn_img,
            'syn_mask': syn_mask,
            'real_img': real_img,
            'real_mask': real_mask,
        }
    

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//','/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//','/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
        
        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)
        
        img, mask = img[np.newaxis,:], mask[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 

class InferenceSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(InferenceSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')).convert('I')
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//','/')).convert('I')
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        
        h, w = img.shape
        img = PadImg(img)
        
        img = img[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        return img, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 


class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(EvalSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name
        with open(self.dataset_dir+'/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()

    def __getitem__(self, idx):
        mask_pred = Image.open((self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/' + self.test_list[idx] + '.png').replace('//','/'))
        mask_gt = Image.open(self.dataset_dir + '/masks/' + self.test_list[idx] + '.png')

        mask_pred = np.array(mask_pred, dtype=np.float32)  / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32)  / 255.0
        
        if len(mask_pred.shape) == 3:
            mask_pred = mask_pred[:,:,0]
        
        h, w = mask_pred.shape
        
        mask_pred, mask_gt = mask_pred[np.newaxis,:], mask_gt[np.newaxis,:]
        
        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, [h,w]
    def __len__(self):
        return len(self.test_list) 

# class EvalSetLoader(Dataset):
#     def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name):
#         super(EvalSetLoader).__init__()
#         self.dataset_dir = '/home/pc/work/BasicIRSTD-main/data'
#         # self.mask_pred_dir = '/home/pc/work/IRSTD-Ablation/BasicIRSTD-main'
#         self.mask_pred_dir = '/home/pc/work/IRSTD-Ablation/BasicIRSTD-main/Traditional_methods/NRAM-master/Demo_NRAM/results'
#         self.test_dataset_name = test_dataset_name

#         with open(self.dataset_dir+'/'+test_dataset_name+'/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
#             self.test_list = f.read().splitlines()

#     def __getitem__(self, idx):
#         # mask_pred = Image.open((self.mask_pred_dir + '/'+ self.test_dataset_name + '/' +'IPI'+'/'+'E' + '/' + self.test_list[idx] + '.png').replace('//','/'))
#         mask_pred = Image.open((self.mask_pred_dir + '/'+ self.test_dataset_name + '/' + 'target/' + self.test_list[idx] + '.png').replace('//','/'))
#         mask_gt = Image.open(self.dataset_dir +'/'+self.test_dataset_name + '/masks/' + self.test_list[idx] + '.png')

#         mask_pred = np.array(mask_pred, dtype=np.float32)  / 255.0
#         mask_gt = np.array(mask_gt, dtype=np.float32)  / 255.0
#         if len(mask_gt.shape) > 2:
#             mask_gt = mask_gt[:,:,0]
#         if len(mask_pred.shape) == 3:
#             mask_pred = mask_pred[:,:,0]
#         # if len(mask_gt.shape) == 5:
#         #     print('1')
        
#         h, w = mask_pred.shape
        
#         mask_pred, mask_gt = mask_pred[np.newaxis,:], mask_gt[np.newaxis,:]
        
#         mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
#         mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
#         # if len(mask_gt.shape) == 5:
#         #     print('1')
#         # if idx == 90:
#         #     print('1')
#         return mask_pred, mask_gt, [h,w]
#     def __len__(self):
#         return len(self.test_list) 

class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]

            # print('aug: 1') # check for randomness, passed
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
            # print('aug: 2') # check for randomness, passed
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
            # print('aug: 3') # check for randomness, passed
        return input, target
