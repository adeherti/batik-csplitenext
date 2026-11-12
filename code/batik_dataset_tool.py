"""
batik_dataset_tool.py
Alat semi-otomatis untuk membangun dataset batik gabungan yang bersih & bebas leakage.

DUA MODE:
  MODE = "propose"  -> temukan sumber OTOMATIS, normalkan nama kelas, gabung otomatis
                       yang jelas identik, tulis draft peta kelas ke class_map.json,
                       dan cetak DAFTAR REVIEW singkat (pasangan ambigu).
  MODE = "build"    -> pakai class_map.json + OVERRIDE, dedup (perceptual hash),
                       split 80/10/10 per gambar asli, salin file + manifest.csv.

Penemuan path sumber OTOMATIS: Anda tidak perlu mengisi path manual. Cukup Add Input
dataset di Kaggle, skrip mencari sendiri folder yang berisi kelas-kelas gambar.

Butuh: pip install pillow imagehash   (matplotlib utk show_pair)
"""

import os
import re
import csv
import json
import shutil
import random
import difflib
import collections
from PIL import Image
import imagehash

# ============================ KONFIGURASI ============================
BASE_INPUT = "/kaggle/input"       # tempat Kaggle me-mount semua dataset
AUTO_DISCOVER = True               # True: cari path sumber otomatis (disarankan)
EXCLUDE = ("augment",)             # lewati folder yg mengandung kata ini (cegah leakage)
MIN_CLASSES = 4                    # folder dianggap "root kelas" bila punya >= sekian kelas
# Dipakai HANYA bila AUTO_DISCOVER=False:
SOURCES_MANUAL = [
    # ("/kaggle/input/indonesian-batik-motifs", "dh"),
    # ("/kaggle/input/batik-nusantara-batik-indonesia-dataset", "hb"),
]

OUT_DIR = "/kaggle/working/batik_clean"
MAP_JSON = "/kaggle/working/class_map.json"
MODE = "propose"                   # "propose" dulu, lalu "build"

STOPWORDS = {"batik", "motif", "motifs", "corak", "kain", "dataset", "the", "of"}
DISTINCTIVE_DF = 2
NAME_SIM_THRESHOLD = 0.60
PHASH_THRESHOLD = 5

OVERRIDE = {
    # diisi SETELAH review; nilai = nama kanonik (gabung) atau None (buang)
}
DROP_UNMAPPED = True
SEED = 42
SPLIT = (0.8, 0.1, 0.1)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# ====================================================================


def auto_find_sources(base=None, min_classes=None, exclude=None):
    """Cari 'root kelas' otomatis: folder dgn >= min_classes subfolder yg berisi gambar.
    Ambil root terluar; folder ber-kata-'exclude' dilewati (mis. augmented)."""
    base = BASE_INPUT if base is None else base
    min_classes = MIN_CLASSES if min_classes is None else min_classes
    exclude = EXCLUDE if exclude is None else exclude
    roots = []
    if not os.path.isdir(base):
        return roots
    for r, dirs, files in os.walk(base):
        low = r.lower()
        if any(x in low for x in exclude):
            dirs[:] = []
            continue
        n_class = 0
        for d in dirs:
            p = os.path.join(r, d)
            try:
                if any(f.lower().endswith(IMG_EXTS) for f in os.listdir(p)):
                    n_class += 1
            except Exception:
                pass
        if n_class >= min_classes:
            roots.append(r)
            dirs[:] = []   # jangan turun lebih dalam; root sudah ketemu
    sources, seen = [], set()
    for root in roots:
        rel = os.path.relpath(root, base).split(os.sep)
        base_tag = re.sub(r"[^a-z0-9]", "", rel[0].lower())[:8] or "src"
        tag, k = base_tag, 1
        while tag in seen:
            k += 1
            tag = f"{base_tag}{k}"
        seen.add(tag)
        sources.append((root, tag))
    return sources


def resolve_sources():
    if AUTO_DISCOVER:
        srcs = auto_find_sources()
        print("=== SUMBER TERDETEKSI OTOMATIS ===")
        if not srcs:
            print(f"  (tidak ada) Pastikan dataset sudah Add Input di {BASE_INPUT},")
            print("  dan bukan hanya folder ber-kata 'augment'.")
        for root, tag in srcs:
            n = sum(1 for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
            print(f"  [{tag}] {root}  (~{n} kelas)")
        print()
        return srcs
    return SOURCES_MANUAL


def discover(sources):
    data = collections.defaultdict(list)
    for root, tag in sources:
        if not os.path.isdir(root):
            print(f"[WARN] sumber tidak ditemukan: {root}")
            continue
        for r, _, files in os.walk(root):
            imgs = [f for f in files if f.lower().endswith(IMG_EXTS)]
            if imgs:
                raw = os.path.basename(r.rstrip("/"))
                data[(tag, raw)].extend(os.path.join(r, f) for f in imgs)
    return data


def normalize(raw):
    toks = [t for t in re.split(r"[-_\s]+", raw.lower()) if t and t not in STOPWORDS]
    return frozenset(toks)


class UF:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def compute_hashes(files):
    hashes, corrupt = {}, []
    for fp in files:
        try:
            with Image.open(fp) as im:
                hashes[fp] = imagehash.phash(im.convert("RGB"))
        except Exception:
            corrupt.append(fp)
    return hashes, corrupt


def dup_clusters(hashes, threshold):
    uniq = collections.defaultdict(list)
    for fp, h in hashes.items():
        uniq[h].append(fp)
    uh = list(uniq.keys())
    uf = UF(range(len(uh)))
    if threshold > 0:
        for i in range(len(uh)):
            for j in range(i + 1, len(uh)):
                if uh[i] - uh[j] <= threshold:
                    uf.union(i, j)
    comp = collections.defaultdict(list)
    for idx, h in enumerate(uh):
        comp[uf.find(idx)].extend(uniq[h])
    return list(comp.values())


def pick_rep(cluster):
    def score(fp):
        try:
            with Image.open(fp) as im:
                w, h = im.size
            return (w * h, os.path.getsize(fp))
        except Exception:
            return (0, 0)
    return max(cluster, key=score)


def cross_image_sim(files_a, files_b, sample=8):
    ha, _ = compute_hashes(files_a[:sample])
    hb, _ = compute_hashes(files_b[:sample])
    if not ha or not hb:
        return None, 0
    best, near = 999, 0
    for x in ha.values():
        for y in hb.values():
            d = x - y
            best = min(best, d)
            if d <= PHASH_THRESHOLD:
                near += 1
    return best, near


def propose():
    data = discover(resolve_sources())
    raw_classes = sorted({raw for (_, raw) in data})
    if not raw_classes:
        print("Tidak ada kelas ditemukan. Cek Add Input dataset di Kaggle.")
        return
    files_by_raw = collections.defaultdict(list)
    for (tag, raw), v in data.items():
        files_by_raw[raw].extend(v)
    norm = {raw: normalize(raw) for raw in raw_classes}

    uf = UF(raw_classes)
    by_norm = collections.defaultdict(list)
    for raw in raw_classes:
        by_norm[norm[raw]].append(raw)
    for group in by_norm.values():
        for other in group[1:]:
            uf.union(group[0], other)

    groups = collections.defaultdict(list)
    for raw in raw_classes:
        groups[uf.find(raw)].append(raw)

    def canonical(members):
        best = min(members, key=lambda r: (len(norm[r]), len(r)))
        return "_".join(w.capitalize() for w in re.split(r"[-_\s]+", best)
                        if w.lower() not in STOPWORDS) or best

    class_map, canon_of = {}, {}
    for root, members in groups.items():
        c = canonical(members)
        canon_of[root] = c
        for m in members:
            class_map[m] = c
    auto_merged = {canon_of[r]: sorted(m) for r, m in groups.items() if len(m) > 1}

    df = collections.Counter()
    for raw in raw_classes:
        for t in norm[raw]:
            df[t] += 1
    reps = list(groups.keys())
    reviews = []
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            a, b = reps[i], reps[j]
            na = set().union(*[norm[m] for m in groups[a]])
            nb = set().union(*[norm[m] for m in groups[b]])
            reasons = []
            shared = {t for t in (na & nb) if df[t] <= DISTINCTIVE_DF}
            if na and nb and (na <= nb or nb <= na) and shared:
                reasons.append(f"subset+token '{','.join(sorted(shared))}'")
            elif shared:
                reasons.append(f"berbagi token '{','.join(sorted(shared))}'")
            ratio = difflib.SequenceMatcher(None, " ".join(sorted(na)),
                                            " ".join(sorted(nb))).ratio()
            if ratio >= NAME_SIM_THRESHOLD:
                reasons.append(f"nama mirip {ratio:.2f}")
            if not reasons:
                continue
            best, near = cross_image_sim(files_by_raw[groups[a][0]], files_by_raw[groups[b][0]])
            if best is not None and (best <= PHASH_THRESHOLD or near > 0):
                reasons.append(f"gambar mirip (min dist={best}, near={near})")
            reviews.append((canon_of[a], canon_of[b], groups[a], groups[b], reasons))

    with open(MAP_JSON, "w") as f:
        json.dump(class_map, f, indent=2, ensure_ascii=False)

    print(f"Kelas mentah: {len(raw_classes)} -> kelas kanonik: {len(groups)}")
    print(f"Peta kelas otomatis ditulis ke: {MAP_JSON}\n")
    print("=== DIGABUNG OTOMATIS (aman, nama identik) ===")
    if auto_merged:
        for c, members in sorted(auto_merged.items()):
            print(f"  {c:22s} <- {members}")
    else:
        print("  (tidak ada)")

    print("\n=== PERLU REVIEW ANDA (ambigu, TIDAK digabung otomatis) ===")
    if reviews:
        for a, b, ga, gb, reasons in reviews:
            print(f"  [{a}] vs [{b}]  | {'; '.join(reasons)}")
            print(f"       folder: {ga}  <->  {gb}")
        print("\n  Aturan cepat: ada 'gambar mirip' dist rendah -> GABUNG; tidak ada -> pisah.")
        print("  Lihat gambar: import batik_dataset_tool as t; t.show_pair('<folderA>','<folderB>')")
        print("  Jika sama -> tambahkan keduanya ke OVERRIDE dgn nama kanonik yg sama.")
    else:
        print("  (tidak ada pasangan ambigu)")
    print("\nLangkah berikut: isi OVERRIDE bila perlu, set MODE='build', jalankan lagi.")


def show_pair(raw_a, raw_b, n=4):
    import matplotlib.pyplot as plt
    data = discover(resolve_sources())
    files = collections.defaultdict(list)
    for (_, raw), v in data.items():
        files[raw].extend(v)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    fig.suptitle(f"ATAS: {raw_a}   |   BAWAH: {raw_b}")
    for row, raw in enumerate([raw_a, raw_b]):
        for col in range(n):
            axes[row, col].axis("off")
            if col < len(files[raw]):
                try:
                    axes[row, col].imshow(Image.open(files[raw][col]).convert("RGB"))
                except Exception:
                    pass
    plt.tight_layout()
    plt.show()


def build():
    with open(MAP_JSON) as f:
        class_map = json.load(f)
    class_map.update(OVERRIDE)
    random.seed(SEED)
    data = discover(resolve_sources())
    file2canon = {}
    for (tag, raw), v in data.items():
        canon = class_map.get(raw, None if DROP_UNMAPPED else raw)
        if canon is None:
            continue
        for fp in v:
            file2canon[fp] = canon
    print(f"File setelah mapping kelas: {len(file2canon)}")

    hashes, corrupt = compute_hashes(list(file2canon))
    for fp in corrupt:
        file2canon.pop(fp, None)
    clusters = dup_clusters({fp: hashes[fp] for fp in file2canon}, PHASH_THRESHOLD)
    kept, dropped = [], 0
    for c in clusters:
        if len(set(file2canon[fp] for fp in c)) > 1:
            dropped += len(c)
            continue
        kept.append(pick_rep(c))
    print(f"Gambar unik setelah dedup: {len(kept)} (buang cluster konflik-label: {dropped})")

    by_class = collections.defaultdict(list)
    for fp in kept:
        by_class[file2canon[fp]].append(fp)
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    manifest = []
    counts = {s: collections.Counter() for s in ("train", "val", "test")}
    for canon, files in sorted(by_class.items()):
        random.shuffle(files)
        n = len(files)
        if n < 3:
            print(f"[WARN] kelas '{canon}' hanya {n} gambar -> semua ke train")
            parts = {"train": files, "val": [], "test": []}
        else:
            n_va = max(1, round(n * SPLIT[1]))
            n_te = max(1, round(n * SPLIT[2]))
            parts = {"train": files[:n - n_va - n_te],
                     "val": files[n - n_va - n_te:n - n_te],
                     "test": files[n - n_te:]}
        for split, fl in parts.items():
            for i, src in enumerate(fl):
                ext = os.path.splitext(src)[1].lower()
                d = os.path.join(OUT_DIR, split, canon)
                os.makedirs(d, exist_ok=True)
                shutil.copy2(src, os.path.join(d, f"{canon}_{split}_{i:05d}{ext}"))
                manifest.append((src, canon, split))
                counts[split][canon] += 1

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_path", "class", "split"])
        w.writerows(manifest)

    print("\n=== HASIL SPLIT ===")
    classes = sorted(by_class)
    print(f"{'kelas':28s} train  val  test")
    for c in classes:
        print(f"{c:28s} {counts['train'][c]:5d} {counts['val'][c]:4d} {counts['test'][c]:4d}")
    tot = {s: sum(counts[s].values()) for s in counts}
    print(f"\nTOTAL train={tot['train']} val={tot['val']} test={tot['test']} | kelas={len(classes)}")
    print(f"Output: {OUT_DIR} (+ manifest.csv). Leakage dicegah: dedup SEBELUM split.")


if __name__ == "__main__":
    if MODE == "propose":
        propose()
    elif MODE == "build":
        build()
    else:
        print("MODE harus 'propose' atau 'build'")
