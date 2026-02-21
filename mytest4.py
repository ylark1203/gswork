#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
from typing import List, Literal

def list_images(src_dir: Path) -> List[Path]:
    """列出src_dir下的图片文件（常见扩展名），不递归。"""
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    files = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return files

def copy_last_n_images(
    src_dir: Path,
    dst_dir: Path,
    n: int = 350,
    sort_by: Literal["name", "mtime"] = "name",
) -> int:
    """
    将src_dir中的最后n张图片复制到dst_dir。
    sort_by:
      - "name": 按文件名排序（适合帧号递增）
      - "mtime": 按修改时间排序（适合生成时间递增）
    返回实际复制的数量。
    """
    files = list_images(src_dir)
    if not files:
        return 0

    if sort_by == "name":
        files.sort(key=lambda p: p.name)
    elif sort_by == "mtime":
        files.sort(key=lambda p: p.stat().st_mtime)
    else:
        raise ValueError(f"Unknown sort_by: {sort_by}")

    selected = files[-n:] if len(files) > n else files

    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in selected:
        # copy2会尽量保留时间戳等元信息
        shutil.copy2(f, dst_dir / f.name)

    return len(selected)

def main():
    src_root = Path("/mnt/data/lyl/codes/RGBAvatar/output")
    dst_root = Path("./download/rgbavatar")

    # 你可以改成 "mtime"：按修改时间取最后350张
    sort_by: Literal["name", "mtime"] = "name"
    n_last = 350

    groups = ["INSTA", "HR"]
    total_copied = 0

    for group in groups:
        group_dir = src_root / group
        if not group_dir.exists():
            print(f"[WARN] group not found: {group_dir}")
            continue

        for role_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()]):
            role = role_dir.name
            src_dir = role_dir / "reproduction" / "render_image"
            if not src_dir.exists():
                print(f"[SKIP] no render_image: {src_dir}")
                continue

            dst_dir = dst_root / role
            copied = copy_last_n_images(src_dir, dst_dir, n=n_last, sort_by=sort_by)
            total_copied += copied
            print(f"[OK] {group}/{role}: copied {copied} -> {dst_dir}")

    print(f"\nDone. Total copied: {total_copied}")

if __name__ == "__main__":
    main()
