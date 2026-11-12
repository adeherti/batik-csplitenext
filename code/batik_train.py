"""
batik_train.py — Scaffold pelatihan & evaluasi (PyTorch + timm).

Dua mode evaluasi:
  1. train_one(cfg)   -> split tetap train/val/test (cara lama)
  2. run_kfold(cfg)   -> stratified k-fold CV pada SELURUH gambar (disarankan)

Perbaikan stabilitas (mengatasi variansi antar-run):
  - ckpt_select="ema"  : bobot EMA (exponential moving average) dipakai utk evaluasi.
                         Menghilangkan kerapuhan "pilih checkpoint terbaik dari val"
                         yang sangat noisy saat val set kecil.
  - ckpt_select="last" : bobot epoch terakhir.
  - ckpt_select="best_val": cara lama (rapuh pada dataset kecil).

Pada mode k-fold, TIDAK ada val set: tiap fold melatih pada k-1 bagian dan menguji
pada 1 bagian, sehingga setiap gambar diuji tepat sekali (766 sampel uji efektif,
bukan 78). Estimasi jauh lebih presisi.

Butuh: pip install torch torchvision timm scikit-learn thop
Perlu cspconvnext.py di direktori yang sama.
"""

import os
import csv
import copy
import time
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ============================ KONFIGURASI ============================
@dataclass
class Config:
    data_root: str = "/kaggle/input/batik-clean-v1/batik_clean"
    num_classes: int = 20
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 300
    lr: float = 3e-4
    weight_decay: float = 0.05
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    label_smoothing: float = 0.1
    drop_path: float = 0.1
    # --- augmentasi ---
    aug_geometric: bool = True
    aug_mixup_cutmix: bool = False
    # --- model ---
    model_name: str = "cspconvnext"
    csp_skip_ratio: float = 0.5
    stem_type: str = "stepped"
    block_norm: str = "bn"
    use_ese: bool = True
    pretrained: bool = False
    # --- resep pelatihan ---
    warmup_epochs: int = 20
    scale_lr_by_batch: bool = False
    # --- stabilisasi checkpoint (perbaikan variansi) ---
    ckpt_select: str = "ema"      # "ema" | "last" | "best_val"
    ema_decay: float = 0.999
    # --- k-fold ---
    n_folds: int = 5
    fold: int = -1                # -1 = semua fold (dipakai run_kfold)
    # --- lain-lain ---
    seed: int = 42
    num_workers: int = 2
    results_csv: str = "/kaggle/working/results.csv"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
# ====================================================================


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# --------------------------- EMA ---------------------------
class ModelEMA:
    """Exponential moving average bobot model. Mengurangi variansi hasil secara drastis
    karena evaluasi tidak lagi bergantung pada satu checkpoint yang kebetulan bagus."""

    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1.0 - self.decay)
            else:
                e.copy_(m)


# --------------------------- DATA ---------------------------
class ListImageDataset(Dataset):
    """Dataset dari daftar path + label (dipakai k-fold)."""

    def __init__(self, files, labels, transform):
        self.files, self.labels, self.transform = files, labels, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.transform(img), int(self.labels[i])


def build_transforms(cfg):
    if cfg.aug_geometric:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(cfg.img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize(int(cfg.img_size * 1.14)),
            transforms.CenterCrop(cfg.img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(cfg.img_size * 1.14)),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def pool_all_images(data_root):
    """Gabungkan SEMUA gambar dari train/val/test menjadi satu pool (untuk k-fold)."""
    splits = [s for s in ("train", "val", "test")
              if os.path.isdir(os.path.join(data_root, s))]
    classes = sorted({c for s in splits
                      for c in os.listdir(os.path.join(data_root, s))
                      if os.path.isdir(os.path.join(data_root, s, c))})
    cls2idx = {c: i for i, c in enumerate(classes)}
    files, labels = [], []
    for s in splits:
        for c in sorted(os.listdir(os.path.join(data_root, s))):
            d = os.path.join(data_root, s, c)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(IMG_EXTS):
                    files.append(os.path.join(d, f))
                    labels.append(cls2idx[c])
    return np.array(files), np.array(labels), classes


def get_loaders(cfg):
    """Loader untuk split tetap (mode lama)."""
    train_tf, eval_tf = build_transforms(cfg)
    train_ds = datasets.ImageFolder(os.path.join(cfg.data_root, "train"), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(cfg.data_root, "val"), eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(cfg.data_root, "test"), eval_tf)
    assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx
    g = torch.Generator(); g.manual_seed(cfg.seed)
    tl = DataLoader(train_ds, cfg.batch_size, shuffle=True, drop_last=True,
                    num_workers=cfg.num_workers, pin_memory=True, generator=g)
    vl = DataLoader(val_ds, cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
    sl = DataLoader(test_ds, cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
    return tl, vl, sl, train_ds.classes


# --------------------------- MODEL ---------------------------
def build_model(cfg):
    name = cfg.model_name.lower()
    if name in ("cspconvnext", "convnext_ours"):
        from cspconvnext import CSPConvNeXt
        return CSPConvNeXt(num_classes=cfg.num_classes,
                           csp=(name == "cspconvnext"),
                           skip_ratio=cfg.csp_skip_ratio,
                           stem_type=cfg.stem_type,
                           block_norm=cfg.block_norm,
                           use_ese=cfg.use_ese,
                           drop_path_rate=cfg.drop_path)
    import timm
    try:
        return timm.create_model(name, pretrained=cfg.pretrained,
                                 num_classes=cfg.num_classes, drop_path_rate=cfg.drop_path)
    except TypeError:
        return timm.create_model(name, pretrained=cfg.pretrained, num_classes=cfg.num_classes)


def measure_efficiency(model, img_size=224):
    params = sum(p.numel() for p in model.parameters()) / 1e6
    gflops = None
    try:
        from thop import profile
        macs, _ = profile(model.to("cpu").eval(),
                          inputs=(torch.randn(1, 3, img_size, img_size),), verbose=False)
        gflops = macs / 1e9
    except Exception as e:
        print(f"[WARN] gagal hitung FLOPs: {e}")
    return params, gflops


# --------------------------- EVALUASI ---------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, trues = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
        probs.append(p); trues.append(np.asarray(y))
    y_prob = np.concatenate(probs); y_true = np.concatenate(trues)
    y_pred = y_prob.argmax(1)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        m["auc_macro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        m["auc_macro"] = float("nan")
    return m, (y_true, y_pred, y_prob)


# --------------------------- OPTIMIZER / SCHEDULER ---------------------------
def _effective_lr(cfg):
    if cfg.scale_lr_by_batch:
        return 1.25e-4 * (cfg.batch_size / 128)
    return cfg.lr


def _make_optimizer(cfg, model):
    lr = _effective_lr(cfg)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                           weight_decay=cfg.weight_decay, nesterov=True)


def _make_scheduler(cfg, optimizer):
    from torch.optim.lr_scheduler import (CosineAnnealingLR, StepLR, LinearLR,
                                          SequentialLR, ConstantLR)
    warm = min(cfg.warmup_epochs, max(0, cfg.epochs - 1))
    if cfg.scheduler == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs - warm))
    elif cfg.scheduler == "step":
        main = StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.1)
    else:
        if warm <= 0:
            return None
        main = ConstantLR(optimizer, factor=1.0, total_iters=0)
    if warm > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warm)
        return SequentialLR(optimizer, [warmup, main], milestones=[warm])
    return main


# --------------------------- LOOP PELATIHAN INTI ---------------------------
def _train_loop(cfg, model, train_loader, val_loader=None, verbose=False):
    """Melatih model; mengembalikan model siap-evaluasi sesuai cfg.ckpt_select."""
    device = cfg.device
    mixup_fn, criterion = None, nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    if cfg.aug_mixup_cutmix:
        from timm.data import Mixup
        from timm.loss import SoftTargetCrossEntropy
        mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
                         mode="batch", label_smoothing=cfg.label_smoothing,
                         num_classes=cfg.num_classes)
        criterion = SoftTargetCrossEntropy()

    optimizer = _make_optimizer(cfg, model)
    scheduler = _make_scheduler(cfg, optimizer)
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ckpt_select == "ema" else None

    best_f1, best_state = -1.0, None
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if mixup_fn is not None:
                x, y = mixup_fn(x, y)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
        if scheduler:
            scheduler.step()
        if cfg.ckpt_select == "best_val" and val_loader is not None:
            vm, _ = evaluate(model, val_loader, device)
            if vm["f1_macro"] > best_f1:
                best_f1 = vm["f1_macro"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch + 1) % 25 == 0:
            print(f"  epoch {epoch+1}/{cfg.epochs}", flush=True)

    if cfg.ckpt_select == "ema":
        return ema.ema
    if cfg.ckpt_select == "best_val" and best_state is not None:
        model.load_state_dict(best_state)
    return model


def _append_row(path, row):
    exists = os.path.exists(path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def _base_row(cfg):
    return {
        "model": cfg.model_name,
        "csp_skip_ratio": cfg.csp_skip_ratio if cfg.model_name == "cspconvnext" else None,
        "stem": cfg.stem_type, "block_norm": cfg.block_norm, "ese": cfg.use_ese,
        "warmup": cfg.warmup_epochs, "ckpt": cfg.ckpt_select,
        "epochs": cfg.epochs, "lr": round(_effective_lr(cfg), 8),
        "aug_geom": cfg.aug_geometric, "aug_mix": cfg.aug_mixup_cutmix,
        "seed": cfg.seed,
    }


# --------------------------- MODE 1: split tetap ---------------------------
def train_one(cfg, verbose=False):
    set_seed(cfg.seed)
    train_loader, val_loader, test_loader, classes = get_loaders(cfg)
    model = build_model(cfg).to(cfg.device)
    t0 = time.time()
    model = _train_loop(cfg, model, train_loader, val_loader, verbose)
    train_time = time.time() - t0

    test_metrics, arrays = evaluate(model, test_loader, cfg.device)
    _, eval_tf = build_transforms(cfg)
    tr_eval = DataLoader(datasets.ImageFolder(os.path.join(cfg.data_root, "train"), eval_tf),
                         cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    train_metrics, _ = evaluate(model, tr_eval, cfg.device)

    params, gflops = measure_efficiency(build_model(cfg), cfg.img_size)
    row = {**_base_row(cfg), "fold": None,
           "train_accuracy": round(train_metrics["accuracy"], 4),
           **{f"test_{k}": round(v, 4) for k, v in test_metrics.items()},
           "params_M": round(params, 3),
           "gflops": round(gflops, 3) if gflops else None,
           "train_time_s": round(train_time, 1)}
    _append_row(cfg.results_csv, row)
    if verbose:
        print(row)
    return row, arrays, classes


# --------------------------- MODE 2: k-fold CV ---------------------------
def run_kfold(cfg, verbose=True):
    """Stratified k-fold CV pada SELURUH gambar. Tiap gambar diuji tepat sekali."""
    files, labels, classes = pool_all_images(cfg.data_root)
    if verbose:
        print(f"Pool: {len(files)} gambar, {len(classes)} kelas, {cfg.n_folds}-fold")

    train_tf, eval_tf = build_transforms(cfg)
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    params, gflops = measure_efficiency(build_model(cfg), cfg.img_size)

    all_true, all_pred, fold_rows = [], [], []
    for k, (tr_idx, te_idx) in enumerate(skf.split(files, labels)):
        if cfg.fold >= 0 and k != cfg.fold:
            continue
        set_seed(cfg.seed + k)
        tr_ds = ListImageDataset(files[tr_idx], labels[tr_idx], train_tf)
        te_ds = ListImageDataset(files[te_idx], labels[te_idx], eval_tf)
        g = torch.Generator(); g.manual_seed(cfg.seed + k)
        tl = DataLoader(tr_ds, cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.num_workers, pin_memory=True, generator=g)
        el = DataLoader(te_ds, cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

        model = build_model(cfg).to(cfg.device)
        t0 = time.time()
        model = _train_loop(cfg, model, tl, None, verbose=False)
        dt = time.time() - t0

        m, (yt, yp, _) = evaluate(model, el, cfg.device)
        all_true.append(yt); all_pred.append(yp)
        row = {**_base_row(cfg), "fold": k,
               "train_accuracy": None,
               **{f"test_{kk}": round(v, 4) for kk, v in m.items()},
               "params_M": round(params, 3),
               "gflops": round(gflops, 3) if gflops else None,
               "train_time_s": round(dt, 1)}
        _append_row(cfg.results_csv, row)
        fold_rows.append(row)
        if verbose:
            print(f"  fold {k}: acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} "
                  f"({dt/60:.1f} menit)", flush=True)

    if not fold_rows:
        return None, None
    accs = [r["test_accuracy"] for r in fold_rows]
    f1s = [r["test_f1_macro"] for r in fold_rows]
    summary = {
        "model": cfg.model_name, "n_folds": len(fold_rows),
        "acc_mean": round(float(np.mean(accs)), 4), "acc_std": round(float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, 4),
        "f1_mean": round(float(np.mean(f1s)), 4), "f1_std": round(float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0, 4),
        "params_M": round(params, 3), "gflops": round(gflops, 3) if gflops else None,
    }
    if verbose:
        print(f"  >>> {cfg.model_name}: acc {summary['acc_mean']:.4f}+/-{summary['acc_std']:.4f} | "
              f"f1 {summary['f1_mean']:.4f}+/-{summary['f1_std']:.4f}", flush=True)
    pooled = (np.concatenate(all_true), np.concatenate(all_pred)) if all_true else None
    return summary, pooled


def sanity_check(cfg):
    set_seed(cfg.seed)
    train_loader, *_ = get_loaders(cfg)
    model = build_model(cfg).to(cfg.device)
    x, y = next(iter(train_loader))
    x, y = x.to(cfg.device), y.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    for i in range(60):
        opt.zero_grad()
        out = model(x)
        crit(out, y).backward(); opt.step()
        if (i + 1) % 15 == 0:
            print(f"step {i+1:3d} | batch_acc {(out.argmax(1) == y).float().mean():.3f}")


def plot_confusion(pooled, classes, out="/kaggle/working/confusion.png"):
    import matplotlib.pyplot as plt
    y_true, y_pred = pooled
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im); plt.tight_layout(); plt.savefig(out, dpi=150)
    print("disimpan:", out)


if __name__ == "__main__":
    print("Gunakan run_kfold(cfg) untuk evaluasi k-fold, atau train_one(cfg) untuk split tetap.")