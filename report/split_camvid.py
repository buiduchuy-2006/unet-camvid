from pathlib import Path
import random
import csv

# Sửa hai dòng này nếu thư mục của bạn có tên khác
IMAGE_DIR = Path("report/mau_anh_mask.png")
MASK_DIR = Path("report/mau_anh_mask.png")
OUT_DIR = Path("data/splits")

EXPECTED = {"train": 367, "val": 101, "test": 233}
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(folder):
    if not folder.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục: {folder.resolve()}\n"
            "Hãy sửa IMAGE_DIR/MASK_DIR ở đầu file cho đúng tên thư mục."
        )
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in EXTS)

def key(path):
    # Bỏ các hậu tố thường dùng cho mask, nếu có
    s = path.stem
    for suffix in ("_L", "_label", "_labels", "_mask", "_gt"):
        if s.lower().endswith(suffix.lower()):
            s = s[:-len(suffix)]
            break
    return s

def main():
    images = list_images(IMAGE_DIR)
    masks = list_images(MASK_DIR)

    image_by_key = {}
    for p in images:
        k = key(p)
        if k in image_by_key:
            raise ValueError(f"Trùng tên ảnh sau khi chuẩn hóa: {k}")
        image_by_key[k] = p

    mask_by_key = {}
    for p in masks:
        k = key(p)
        if k in mask_by_key:
            raise ValueError(f"Trùng tên mask sau khi chuẩn hóa: {k}")
        mask_by_key[k] = p

    missing_masks = sorted(set(image_by_key) - set(mask_by_key))
    orphan_masks = sorted(set(mask_by_key) - set(image_by_key))
    pairs = [(image_by_key[k], mask_by_key[k])
             for k in sorted(set(image_by_key) & set(mask_by_key))]

    print(f"Số ảnh tìm thấy: {len(images)}")
    print(f"Số mask tìm thấy: {len(masks)}")
    print(f"Số cặp ghép được: {len(pairs)}")
    print(f"Ảnh thiếu mask: {len(missing_masks)}")
    print(f"Mask không có ảnh: {len(orphan_masks)}")

    if len(images) != 701 or len(masks) != 701 or len(pairs) != 701:
        print("\nCHƯA ĐẠT: cần đúng 701 ảnh, 701 mask và 701 cặp.")
        if missing_masks:
            print("Một số ảnh thiếu mask:", missing_masks[:10])
        if orphan_masks:
            print("Một số mask không có ảnh:", orphan_masks[:10])
        print("Chưa tạo danh sách chia tập để tránh tạo kết quả sai.")
        return

    # Xáo trộn tái lập được. Đây là chia ngẫu nhiên, không phải danh sách split chính thức.
    random.Random(42).shuffle(pairs)
    train = pairs[:367]
    val = pairs[367:367+101]
    test = pairs[367+101:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, group in (("train", train), ("val", val), ("test", test)):
        # Mỗi dòng gồm tên ảnh và tên mask, cách nhau bằng TAB.
        with (OUT_DIR / f"{split}.txt").open("w", encoding="utf-8") as f:
            for image, mask in sorted(group, key=lambda pair: pair[0].name):
                f.write(f"{image.name}\t{mask.name}\n")

    # Ghi báo cáo kiểm tra đơn giản
    with (OUT_DIR / "pair_check.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "mask", "pair_exists"])
        for image, mask in pairs:
            writer.writerow([image.name, mask.name, True])

    print("\nĐÃ TẠO DANH SÁCH:")
    print(f"train.txt: {len(train)} cặp")
    print(f"val.txt:   {len(val)} cặp")
    print(f"test.txt:  {len(test)} cặp")
    print(f"Thư mục đầu ra: {OUT_DIR.resolve()}")
    print("Lưu ý: chia ngẫu nhiên với seed 42; nếu bài yêu cầu split chuẩn chính thức, hãy dùng danh sách split đi kèm bộ CamVid.")

if __name__ == "__main__":
    main()
