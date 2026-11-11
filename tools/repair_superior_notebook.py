"""Repair Superior cells and save a separately named copy for VS Code."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "DAT301m_DataCentric_AI_Roman_Numerals_Master_AllInOne.ipynb"
nb = json.loads(PATH.read_text(encoding="utf-8"))

builder = '''# 4. SUPERIOR FIX V2: SINH DỮ LIỆU, CHIA ẢNH GỐC TRƯỚC AUGMENT
import json
import hashlib
from pathlib import Path

SUPERIOR_DIR = "superior_data_9900"

def build_superior_dataset(target_per_class=990):
    data = Path("data")
    label_book = next((p for p in (Path("label_book"), Path("dataset/label_book"))
                       if p.is_dir()), None)
    if not data.is_dir() or label_book is None:
        raise FileNotFoundError(
            f"Thiếu data/ hoặc label_book/ trên runtime tại {Path.cwd()}. "
            "Chạy cell khởi tạo/giải nén dữ liệu ở đầu notebook trước."
        )
    excluded = {
        "aac26b40-ce5d-11eb-b317-38f9d35ea60f.png",
        "b0a13faa-ce5d-11eb-b317-38f9d35ea60f.png",
        "b030ba5a-ce5d-11eb-b317-38f9d35ea60f.png",
        "aec401ea-ce5d-11eb-b317-38f9d35ea60f.png",
        "aec04e56-ce5d-11eb-b317-38f9d35ea60f.png",
        "adf5ad04-ce5d-11eb-b317-38f9d35ea60f.png",
    }
    fingerprint = hashlib.sha256()
    def read_image(f):
        fingerprint.update(str(f).encode())
        fingerprint.update(f.read_bytes())
        with Image.open(f) as im:
            gray = im.convert("L")
            digest = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
            canvas = preprocess_highres_canvas(gray)
        return digest, canvas

    lb = {c: [] for c in CLASS_NAMES}
    lb_hashes = set()
    raw = {c: {} for c in CLASS_NAMES}
    owners = {}
    for c in CLASS_NAMES:
        for f in sorted((label_book / c).glob("*.png")):
            h, canvas = read_image(f)
            lb_hashes.add(h)
            if canvas is not None:
                lb[c].append((f.name, canvas))
        if not lb[c]:
            raise ValueError(f"Label Book thiếu ảnh hợp lệ của lớp {c}.")
    for c in CLASS_NAMES:
        for split in ("train", "val"):
            for f in sorted((data / split / c).glob("*.png")):
                if f.name in excluded:
                    continue
                h, canvas = read_image(f)
                if canvas is not None and h not in lb_hashes:
                    owners.setdefault(h, set()).add(c)
                    raw[c][h] = canvas

    config = dict(version=2, seed=RANDOM_SEED, size=list(IMAGE_SIZE),
                  classes=CLASS_NAMES, target=target_per_class,
                  source=fingerprint.hexdigest())
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(SUPERIOR_DIR) / ("split_v2_" + key)
    marker = root / "manifest.json"
    manifest = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
    if manifest is not None and not all(
        len(list((root / split / c).glob("*.png"))) == count
        for split, counts in manifest["counts"].items() for c, count in counts.items()
    ):
        manifest = None
    if manifest is None:
        rng = random.Random(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        counts = {s: {} for s in ("train", "val", "label_book")}
        sources = {s: {} for s in ("train", "val")}
        for c in CLASS_NAMES:
            items = sorted((h, im) for h, im in raw[c].items() if len(owners[h]) == 1)
            if not 2 <= len(items) <= target_per_class:
                raise ValueError(f"Lớp {c}: cần 2..{target_per_class} ảnh gốc, có {len(items)}.")
            rng.shuffle(items)
            n_val = max(1, round(len(items) * 0.2))
            groups = {"train": items[n_val:], "val": items[:n_val]}
            for split, group in groups.items():
                dest = root / split / c
                dest.mkdir(parents=True, exist_ok=True)
                sources[split][c] = [h for h, _ in group]
                def save(im, name):
                    im.resize(IMAGE_SIZE, Image.Resampling.LANCZOS).convert("RGB").save(dest / name)
                for i, (_, im) in enumerate(group):
                    save(im, f"original_{i:04d}.png")
                if split == "train":
                    for i in range(target_per_class - n_val - len(group)):
                        save(domain_handwriting_augment(group[i % len(group)][1]), f"aug_{i:04d}.png")
                counts[split][c] = target_per_class - n_val if split == "train" else n_val
            dest = root / "label_book" / c
            dest.mkdir(parents=True, exist_ok=True)
            for name, im in lb[c]:
                im.resize(IMAGE_SIZE, Image.Resampling.LANCZOS).convert("RGB").save(dest / name)
            counts["label_book"][c] = len(lb[c])
            print(f"{c}: train={counts['train'][c]}, val={n_val}", flush=True)
        train_hashes = {h for group in sources["train"].values() for h in group}
        val_hashes = {h for group in sources["val"].values() for h in group}
        assert train_hashes.isdisjoint(val_hashes)
        assert (train_hashes | val_hashes).isdisjoint(lb_hashes)
        manifest = dict(config=config, counts=counts, source_hashes=sources)
        marker.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        print("Tái sử dụng dữ liệu Superior đã tạo.")
    print("SUPERIOR FIX V2:", root.resolve())
    print({s: sum(counts.values()) for s, counts in manifest["counts"].items()})
    return str(root / "train"), str(root / "val"), str(root / "label_book")

SUPERIOR_TRAIN_DIR, SUPERIOR_VAL_DIR, CLEAN_LB_DIR = build_superior_dataset()
'''

loader = '''# 5. SUPERIOR FIX V2: NẠP HAI TẬP ĐÃ CHIA TRƯỚC AUGMENT
if "SUPERIOR_TRAIN_DIR" not in globals():
    raise RuntimeError("Chạy cell '4. SUPERIOR FIX V2' ngay phía trên trước.")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    SUPERIOR_TRAIN_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, shuffle=True, seed=RANDOM_SEED,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    SUPERIOR_VAL_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, shuffle=False,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
clean_lb_ds = tf.keras.preprocessing.image_dataset_from_directory(
    CLEAN_LB_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, shuffle=False,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = (train_ds.cache().unbatch()
            .shuffle(1024, seed=RANDOM_SEED, reshuffle_each_iteration=True)
            .batch(BATCH_SIZE).prefetch(AUTOTUNE))
val_ds = val_ds.cache().prefetch(AUTOTUNE)
clean_lb_ds = clean_lb_ds.cache().prefetch(AUTOTUNE)
print("SUPERIOR FIX V2: nạp dữ liệu thành công.")
'''

def put(index, source):
    cell = nb["cells"][index]
    cell["source"] = source.splitlines(keepends=True)
    cell["execution_count"] = None
    cell["outputs"] = []

s = "".join(nb["cells"][32]["source"])
if "from scipy import ndimage as ndi" not in s:
    s = "import random\nfrom scipy import ndimage as ndi\n\n" + s
put(32, s)
put(33, builder)
put(34, loader)
s = "".join(nb["cells"][35]["source"])
if 'superior_model.load_weights("superior_best_model.weights.h5")' not in s:
    s = s.replace("val_loss, val_acc = superior_model.evaluate", 'superior_model.load_weights("superior_best_model.weights.h5")\nval_loss, val_acc = superior_model.evaluate')
put(35, s)
nb["cells"][31]["source"].insert(0,
    "**SUPERIOR FIX V2 — VS Code + Colab:** Chạy các cell High-Res → tạo dữ liệu → nạp dữ liệu → huấn luyện bên dưới. Nếu runtime mới, chạy hai cell khởi tạo ở đầu notebook trước. Các accuracy ghi trong phần mô tả là số tham khảo từ lần chạy cũ; cần đo lại.\n\n")

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]), "notebook", "exec")
import nbformat
nbformat.validate(nbformat.from_dict(nb))
payload = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
PATH.write_text(payload, encoding="utf-8")
fixed = PATH.with_name(PATH.stem + "_FIXED.ipynb")
fixed.write_text(payload, encoding="utf-8")
print(fixed.name)
