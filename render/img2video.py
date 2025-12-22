import argparse
import os, glob, zipfile
from tqdm import tqdm

def images_to_video(path, fps=25, video_format='DIVX'):
    import cv2
    img_array = []
    for filename in tqdm(sorted(glob.glob(f'{path}/*.png'))):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    img_array = img_array[-350:]
    if len(img_array) > 0:
        # out = cv2.VideoWriter(f'{path}/video.avi', cv2.VideoWriter_fourcc(*video_format), fps, size)
        out = cv2.VideoWriter('/mnt/data/lyl/codes/RGBAvatar/render/rgbavatar/wojtek_1.avi', cv2.VideoWriter_fourcc(*video_format), fps, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        
images_to_video("/mnt/data/lyl/codes/RGBAvatar/output/INSTA/wojtek_1/reproduction/render_image")