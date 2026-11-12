"""
CSPConvNeXt / E-ConvNeXt untuk klasifikasi motif batik.

Mengadaptasi perbaikan dari:
  F. Wang et al., "E-ConvNeXt: combining CSPNet with ConvNeXt" (preprint, Elsevier)

Setiap perbaikan dibuat TOGGLEABLE agar bisa dilakukan studi ablasi per komponen:
  1. stem_type="stepped"  -> Stepped Stem (ganti patchify 4x4 stride-4)
  2. block_norm="bn"      -> BatchNorm2d + Conv1x1 (ganti LayerNorm + transpose + FC)
  3. use_ese=True         -> ESE channel attention (ganti LayerScale)
  4. skip_ratio           -> rasio split CSP (0.5 = split naif 50/50, versi "original")

Kombinasi "original CSPConvNeXt" (versi terlemah di ablasi paper):
    stem_type="patchify", block_norm="ln", use_ese=False, skip_ratio=0.5
Kombinasi "improved" (adaptasi E-ConvNeXt):
    stem_type="stepped", block_norm="bn", use_ese=True

Verifikasi lokal:
    python cspconvnext.py
"""

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Utilitas
# ----------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    """LayerNorm pada dimensi channel untuk tensor (B, C, H, W) (channels-first)."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class DropPath(nn.Module):
    """Stochastic depth (drop path) per-sampel."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x / keep * mask


class ESEBlock(nn.Module):
    """Effective Squeeze-and-Excitation: GAP -> satu Conv1x1 (tanpa reduksi) -> HardSigmoid.
    Dipilih paper karena akurasi tertinggi & tercepat dibanding SE/ECA/CBAM."""

    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        w = x.mean((2, 3), keepdim=True)
        return x * self.act(self.fc(w))


# ----------------------------------------------------------------------------
# Blok
# ----------------------------------------------------------------------------
class ConvNeXtBlockLN(nn.Module):
    """Blok ConvNeXt ASLI: dwconv7x7 -> LN(channels-last) -> FC -> GELU -> FC -> LayerScale."""

    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init * torch.ones(dim))
            if layer_scale_init > 0 else None
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)          # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)          # NHWC -> NCHW
        return inp + self.drop_path(x)


class ConvNeXtBlockBN(nn.Module):
    """Blok versi E-ConvNeXt: dwconv7x7 -> BN -> Conv1x1 -> GELU -> Conv1x1 -> BN -> ESE.
    Tanpa transpose (lebih cepat menurut paper). BN sebelum ESE mencegah
    ketidakstabilan/collapse saat channel attention ditambahkan."""

    def __init__(self, dim, drop_path=0.0, use_ese=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(dim)   # normalisasi sebelum attention (perbaikan paper)
        self.ese = ESEBlock(dim) if use_ese else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        x = self.norm1(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.norm2(x)
        if self.ese is not None:
            x = self.ese(x)
        return inp + self.drop_path(x)


def make_block(dim, drop_path, block_norm, use_ese):
    if block_norm == "bn":
        return ConvNeXtBlockBN(dim, drop_path, use_ese=use_ese)
    return ConvNeXtBlockLN(dim, drop_path)


# ----------------------------------------------------------------------------
# Stage
# ----------------------------------------------------------------------------
class ConvNeXtStage(nn.Module):
    """Stage STANDAR: semua channel diproses seluruh blok (baseline non-CSP)."""

    def __init__(self, dim, depth, drop_paths, block_norm="ln", use_ese=False):
        super().__init__()
        self.blocks = nn.Sequential(
            *[make_block(dim, drop_paths[i], block_norm, use_ese) for i in range(depth)]
        )

    def forward(self, x):
        return self.blocks(x)


class CSPConvNeXtStage(nn.Module):
    """Stage CSP: sebagian channel dilewatkan langsung, sisanya diproses blok, lalu
    concat + transisi 1x1.

    skip_ratio: fraksi channel yang DILEWATKAN LANGSUNG (0.5 = split naif 50/50)."""

    def __init__(self, dim, depth, drop_paths, use_transition=True, skip_ratio=0.5,
                 block_norm="ln", use_ese=False):
        super().__init__()
        assert 0.0 < skip_ratio < 1.0, "skip_ratio harus di antara 0 dan 1"
        self.skip_ch = max(1, round(dim * skip_ratio))
        self.proc_ch = dim - self.skip_ch
        self.blocks = nn.Sequential(
            *[make_block(self.proc_ch, drop_paths[i], block_norm, use_ese)
              for i in range(depth)]
        )
        self.use_transition = use_transition
        if use_transition:
            self.transition = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        part1, part2 = x[:, : self.skip_ch], x[:, self.skip_ch:]
        part2 = self.blocks(part2)
        out = torch.cat([part1, part2], dim=1)
        if self.use_transition:
            out = self.transition(out)
        return out


# ----------------------------------------------------------------------------
# Stem & downsample
# ----------------------------------------------------------------------------
class PatchifyStem(nn.Module):
    """Stem ConvNeXt asli: Conv 4x4 stride 4 (224 -> 56)."""

    def __init__(self, in_ch=3, dim=96, norm="ln"):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d(dim) if norm == "bn" else LayerNorm2d(dim)

    def forward(self, x):
        return self.norm(self.proj(x))


class SteppedStem(nn.Module):
    """Stepped Stem (E-ConvNeXt): satu Conv 2x2 s2 + dua Conv 3x3, channel bertahap
    3 -> 32 -> 64 -> dim. Mengurangi kehilangan informasi akibat downsampling mendadak.
    Output spasial sama dengan patchify (224 -> 56)."""

    def __init__(self, in_ch=3, dim=96, norm="bn"):
        super().__init__()

        def nl(c):
            return nn.BatchNorm2d(c) if norm == "bn" else LayerNorm2d(c)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=2, stride=2, padding=0),  # 224 -> 112
            nl(32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),     # 112 -> 56
            nl(64), nn.GELU(),
            nn.Conv2d(64, dim, kernel_size=3, stride=1, padding=1),    # 56 -> 56
            nl(dim),
        )

    def forward(self, x):
        return self.stem(x)


class Downsample(nn.Module):
    """Norm + Conv 2x2 stride 2 (H,W -> H/2,W/2 ; channel in->out)."""

    def __init__(self, in_dim, out_dim, norm="ln"):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_dim) if norm == "bn" else LayerNorm2d(in_dim)
        self.reduce = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        return self.reduce(self.norm(x))


# ----------------------------------------------------------------------------
# Model penuh
# ----------------------------------------------------------------------------
class CSPConvNeXt(nn.Module):
    def __init__(
        self,
        num_classes,
        dims=(96, 192, 384, 768),
        depths=(3, 3, 9, 3),
        drop_path_rate=0.1,
        csp=True,                 # True = CSP; False = ConvNeXt standar (baseline ablasi)
        skip_ratio=0.5,           # fraksi channel dilewatkan langsung (hanya bila csp=True)
        use_transition=True,
        stem_type="stepped",      # "stepped" (E-ConvNeXt) | "patchify" (ConvNeXt asli)
        block_norm="bn",          # "bn" (E-ConvNeXt) | "ln" (ConvNeXt asli)
        use_ese=True,             # ESE channel attention (berlaku bila block_norm="bn")
        head_hidden=512,
        head_dropout=0.2,
    ):
        super().__init__()
        assert len(dims) == 4 and len(depths) == 4
        assert stem_type in ("stepped", "patchify")
        assert block_norm in ("bn", "ln")
        self.cfg_summary = dict(csp=csp, skip_ratio=skip_ratio, stem_type=stem_type,
                                block_norm=block_norm, use_ese=use_ese)

        if stem_type == "stepped":
            self.stem = SteppedStem(3, dims[0], norm=block_norm)
        else:
            self.stem = PatchifyStem(3, dims[0], norm=block_norm)

        total = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total)]

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        cur = 0
        for i in range(4):
            dp = dpr[cur:cur + depths[i]]
            if csp:
                stage = CSPConvNeXtStage(dims[i], depths[i], dp, use_transition,
                                         skip_ratio, block_norm, use_ese)
            else:
                stage = ConvNeXtStage(dims[i], depths[i], dp, block_norm, use_ese)
            self.stages.append(stage)
            cur += depths[i]
            if i < 3:
                self.downsamples.append(Downsample(dims[i], dims[i + 1], norm=block_norm))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        if head_hidden:
            self.head = nn.Sequential(
                nn.Linear(dims[-1], head_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for i in range(4):
            x = self.stages[i](x)
            if i < 3:
                x = self.downsamples[i](x)
        x = x.mean([-2, -1])   # Global Average Pooling
        x = self.norm(x)
        return self.head(x)


# ----------------------------------------------------------------------------
# Helper metrik
# ----------------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_size=(1, 3, 224, 224)):
    try:
        from thop import profile
        model.eval()
        inp = torch.randn(*input_size)
        flops, _ = profile(model, inputs=(inp,), verbose=False)
        return flops
    except ImportError:
        return None


# ----------------------------------------------------------------------------
# Self-test: ablasi tiap perbaikan
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    N_CLASSES = 20
    x = torch.randn(2, 3, 224, 224)

    configs = [
        ("ConvNeXt baseline (LN, patchify)",
         dict(csp=False, stem_type="patchify", block_norm="ln", use_ese=False)),
        ("CSP original (LN, patchify, 50/50)",
         dict(csp=True, skip_ratio=0.5, stem_type="patchify", block_norm="ln", use_ese=False)),
        ("CSP + BN block",
         dict(csp=True, skip_ratio=0.5, stem_type="patchify", block_norm="bn", use_ese=False)),
        ("CSP + BN + ESE",
         dict(csp=True, skip_ratio=0.5, stem_type="patchify", block_norm="bn", use_ese=True)),
        ("CSP + BN + ESE + Stepped (full)",
         dict(csp=True, skip_ratio=0.5, stem_type="stepped", block_norm="bn", use_ese=True)),
        ("Full, skip_ratio=0.25",
         dict(csp=True, skip_ratio=0.25, stem_type="stepped", block_norm="bn", use_ese=True)),
    ]
    for name, kwargs in configs:
        model = CSPConvNeXt(num_classes=N_CLASSES, **kwargs)
        model.eval()
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, N_CLASSES), f"bentuk output salah: {y.shape}"
        params = count_params(model) / 1e6
        flops = count_flops(model)
        fs = f"{flops / 1e9:5.2f} GFLOPs" if flops else "n/a"
        print(f"{name:36s} | {params:6.2f} M params | {fs}")

    print("\nSanity check lolos: semua konfigurasi forward pass OK.")