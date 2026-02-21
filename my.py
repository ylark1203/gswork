import argparse
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

METHODS = [
    "bbw_2000_covarience_shear0125",
    "bbw_2000_covarience_shear0125_dX",
    "bbw_2000_covarience_dX",
]

DATASETS = ["HR", "INSTA"]


def list_images_sorted(folder: Path):
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files, key=lambda p: p.name)


def safe_copy(src: Path, dst_dir: Path, overwrite: bool):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if overwrite or not dst.exists():
        shutil.copy2(src, dst)
        return

    # 不覆盖：同名则加后缀
    stem, suf = src.stem, src.suffix
    k = 1
    while True:
        cand = dst_dir / f"{stem}_{k}{suf}"
        if not cand.exists():
            shutil.copy2(src, cand)
            return
        k += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", default="/mnt/data/lyl/codes/RGBAvatar/output", required=True, help="源根目录，包含 HR/ 和 INSTA/")
    ap.add_argument("--dst_root", default="/mnt/data/lyl/codes/RGBAvatar/download", required=True, help="目标根目录，会在其下创建 ours/")
    ap.add_argument("--n", type=int, default=350, help="复制最后 N 张")
    ap.add_argument("--overwrite", action="store_true", help="允许同名覆盖")
    ap.add_argument("--dry_run", action="store_true", help="只打印不复制")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    ours_root = Path(args.dst_root) / "ours"

    total_copied = 0
    total_missing = 0

    for ds in DATASETS:
        ds_path = src_root / ds
        if not ds_path.exists():
            print(f"[WARN] dataset not found: {ds_path}")
            continue

        # ds 下一级目录就是角色（HR/subject1, INSTA/bala ...)
        roles = sorted([p for p in ds_path.iterdir() if p.is_dir()])
        for role_dir in roles:
            role = role_dir.name

            for method in METHODS:
                src_render = role_dir / method / "render_image"
                if not src_render.exists():
                    total_missing += 1
                    print(f"[MISS] {ds}/{role}/{method}/render_image")
                    continue

                imgs = list_images_sorted(src_render)
                if not imgs:
                    print(f"[EMPTY] {ds}/{role}/{method}/render_image")
                    continue

                pick = imgs[-args.n:] if len(imgs) > args.n else imgs
                dst_dir = ours_root / method / role  # ✅ ours/方法/角色/

                if args.dry_run:
                    print(f"[DRY] {ds}/{role}/{method}: would copy {len(pick)}/{len(imgs)} -> {dst_dir}")
                    continue

                for img in pick:
                    safe_copy(img, dst_dir, overwrite=args.overwrite)

                total_copied += len(pick)
                print(f"[OK] {ds}/{role}/{method}: {len(pick)}/{len(imgs)} copied -> {dst_dir}")

    print("\n==== Summary ====")
    print(f"Total copied images: {total_copied}")
    print(f"Missing render_image folders: {total_missing}")
    if args.dry_run:
        print("Dry-run mode: no files were actually copied.")


if __name__ == "__main__":
    main()
