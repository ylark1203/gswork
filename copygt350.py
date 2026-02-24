import os
import shutil
from pathlib import Path

# ====== 你需要改这里 ======
SRC_ROOT = Path("/mnt/data/lyl/datasets/HR")     # 角色目录所在位置（包含 bala/biden/...）
DST_ROOT = Path("/mnt/data/lyl/codes/RGBAvatar/download/gt")       # 你要复制到的“指定路径”
N_LAST = 350
# =========================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"SRC_ROOT not found: {SRC_ROOT}")
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    # 遍历 SRC_ROOT 下的所有角色目录
    role_dirs = [d for d in SRC_ROOT.iterdir() if d.is_dir()]

    for role_dir in sorted(role_dirs, key=lambda x: x.name):
        images_dir = role_dir / "images.HQ"
        if not images_dir.exists():
            print(f"[SKIP] {role_dir.name}: no images/ folder")
            continue

        # 收集图片并按文件名排序（INSTA 一般是帧号命名，排序即时间顺序）
        imgs = sorted([p for p in images_dir.iterdir() if is_image(p)], key=lambda p: p.name)
        if not imgs:
            print(f"[SKIP] {role_dir.name}: images/ is empty")
            continue

        selected = imgs[-N_LAST:] if len(imgs) > N_LAST else imgs

        # 目标：DST_ROOT/角色名/ 下
        out_dir = DST_ROOT / role_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for p in selected:
            dst = out_dir / p.name
            shutil.copy2(p, dst)

        print(f"[OK] {role_dir.name}: copied {len(selected)} images -> {out_dir}")

if __name__ == "__main__":
    main()
