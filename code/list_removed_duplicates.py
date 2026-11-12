# Satu sel untuk notebook Kaggle 1792026 (input: dataset asli hendryhb; batik-clean-v1 opsional untuk verifikasi).
# Mengulang langkah dedup batik_dataset_tool.build() persis (pHash, jarak Hamming <= 5,
# wakil = resolusi tertinggi, cluster lintas-label dibuang) lalu menulis daftar gambar yang dibuang.
# Hasil: /kaggle/working/removed_duplicates.csv
import os, glob, json, subprocess, sys, collections
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imagehash"], check=False)
import pandas as pd, imagehash
from PIL import Image

PHASH_THRESHOLD, MIN_CLASSES, EXCLUDE = 5, 4, ("augment",)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASS_MAP = {"Aceh_Pintu_Aceh": "Aceh_Pintu_Aceh", "Bali_Barong": "Bali_Barong", "Bali_Merak": "Bali_Merak",
  "DKI_Ondel_Ondel": "Dki_Ondel_Ondel", "JawaBarat_Megamendung": "Jawabarat_Megamendung",
  "JawaTimur_Pring": "Jawatimur_Pring", "Kalimantan_Dayak": "Kalimantan_Dayak", "Lampung_Gajah": "Lampung_Gajah",
  "Madura_Mataketeran": "Madura_Mataketeran", "Maluku_Pala": "Maluku_Pala", "NTB_Lumbung": "Ntb_Lumbung",
  "Papua_Asmat": "Papua_Asmat", "Papua_Cendrawasih": "Papua_Cendrawasih", "Papua_Tifa": "Papua_Tifa",
  "Solo_Parang": "Solo_Parang", "SulawesiSelatan_Lontara": "Sulawesiselatan_Lontara",
  "SumateraBarat_Rumah_Minang": "Sumaterabarat_Rumah_Minang", "SumateraUtara_Boraspati": "Sumaterautara_Boraspati",
  "Yogyakarta_Kawung": "Yogyakarta_Kawung", "Yogyakarta_Parang": "Yogyakarta_Parang"}

# 1. dataset asli saja (batik-clean-v1 tidak boleh ikut terbaca sebagai sumber)
# Dataset hendryhb memuat dua versi (raw_batik_v2 dan raw_batik_v2.1); paper memakai raw_batik_v2.1 (800 gambar).
hits = glob.glob("/kaggle/input/**/raw_batik_v2.1", recursive=True)
assert hits, "Folder raw_batik_v2.1 tidak ditemukan: tambahkan dataset hendryhb lewat Add Input"
BASE = hits[0]
roots = []
for r, dirs, files in os.walk(BASE):
    if any(x in r.lower() for x in EXCLUDE):
        dirs[:] = []; continue
    n = sum(1 for d in dirs if any(f.lower().endswith(IMG_EXTS) for f in os.listdir(os.path.join(r, d))))
    if n >= MIN_CLASSES:
        roots.append(r); dirs[:] = []
print("Sumber:", roots)
file2canon = {}
for root in roots:
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS) and os.path.basename(r) in CLASS_MAP:
                file2canon[os.path.join(r, f)] = CLASS_MAP[os.path.basename(r)]
print("Gambar asli:", len(file2canon))

# 2. hash, cluster, wakil (logika sama dengan batik_dataset_tool)
hashes, corrupt = {}, []
for fp in file2canon:
    try:
        with Image.open(fp) as im:
            hashes[fp] = imagehash.phash(im.convert("RGB"))
    except Exception:
        corrupt.append(fp)
uniq = collections.defaultdict(list)
for fp, h in hashes.items():
    uniq[h].append(fp)
uh = list(uniq); parent = list(range(len(uh)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for i in range(len(uh)):
    for j in range(i + 1, len(uh)):
        if uh[i] - uh[j] <= PHASH_THRESHOLD:
            a, b = find(i), find(j)
            if a != b: parent[a] = b
comp = collections.defaultdict(list)
for i, h in enumerate(uh):
    comp[find(i)].extend(uniq[h])
def score(fp):
    with Image.open(fp) as im:
        w, h = im.size
    return (w * h, os.path.getsize(fp))

# Gambar identik (hash dan ukuran sama) bisa membuat pilihan wakil bergantung pada urutan file;
# bila manifest.csv tersedia, wakil diambil dari anggota cluster yang tercatat di manifest.
tail = lambda p: p.split("raw_batik_v2.1", 1)[-1]
man = glob.glob("/kaggle/input/**/batik_clean/manifest.csv", recursive=True)
man_tails = {tail(p) for p in pd.read_csv(man[0])["source_path"]} if man else set()
rows, kept, cid = [], set(), 0
for c in comp.values():
    if len(c) == 1:
        kept.add(c[0]); continue
    cid += 1
    conflict = len({file2canon[f] for f in c}) > 1
    rep = None if conflict else max(c, key=score)
    if rep and man_tails and tail(rep) not in man_tails:
        alt = [f for f in c if tail(f) in man_tails and score(f) == score(rep)]
        if alt: rep = alt[0]
    if rep: kept.add(rep)
    for f in c:
        if f == rep: continue
        w, h = Image.open(f).size
        rows.append({"cluster": cid, "removed_path": f, "class": file2canon[f], "width": w, "height": h,
                     "reason": "label conflict" if conflict else "duplicate",
                     "retained_path": rep or "", "hamming_to_retained": int(hashes[f] - hashes[rep]) if rep else ""})
for f in corrupt:
    rows.append({"cluster": "", "removed_path": f, "class": file2canon[f], "width": "", "height": "",
                 "reason": "unreadable", "retained_path": "", "hamming_to_retained": ""})
out = pd.DataFrame(rows).sort_values(["class", "cluster"])
out.to_csv("/kaggle/working/removed_duplicates.csv", index=False)
print(f"\nDipertahankan: {len(kept)} | dibuang: {len(out)} | cluster duplikat: {cid}")
print(out["reason"].value_counts().to_string())
print(out.groupby("class").size().to_string())

# 3. verifikasi terhadap manifest.csv (bila batik-clean-v1 terpasang)
if man:
    same = {tail(p) for p in kept} == man_tails
    print("\nCocok dengan manifest.csv:", same)
