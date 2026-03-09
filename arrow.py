import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
import shutil

input_dir = "input/"
output_dir = "output/"
base_img = cv2.imread("base_track.png")

def detect_arrow_and_orientation(image_name):
    global base_img
    img = cv2.imread(os.path.join(output_dir, image_name))
    if img is None:
        print("错误：无法读取图片")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])

    # 创建掩膜 (Mask)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    np.savetxt('mask.txt', mask, fmt='%d', delimiter='\t')
    points = np.where(mask != 0)
    center = np.mean(points, axis=1)
    center = (round(center[1]), round(center[0]))

    # 绘制质心 绘制叠加
    mask = np.zeros_like(base_img)
    cv2.circle(mask, center, 3, (255 - 0, 255 - 255, 255 - 0), -1)
    cv2.imwrite(output_dir + image_name[:-9]+'mask.png', mask)

    base_img = cv2.bitwise_not(base_img)
    base_img = cv2.add(base_img, mask)

    cv2.imwrite(output_dir + image_name[:-9]+'_white.png', base_img)
    base_img = cv2.bitwise_not(base_img)
    cv2.imwrite(output_dir + image_name[:-9]+'_dark.png', base_img) # 原版
def clip_track(image_name):
    image_path = os.path.join(input_dir, image_name)
    img = Image.open(image_path)
    width, height = img.size
    track_map = img.crop((width * 0.82, height * 0.7, width, height))
    crop_name = image_name[:-4] + "_crop.png"
    track_map.save(os.path.join(output_dir, crop_name))
    return crop_name

if os.path.exists(output_dir):
    shutil.rmtree(output_dir) # 清空已有输出
os.mkdir(output_dir)

for root, dirs, files in os.walk(input_dir):
    for name in files:
        if name.endswith((".png", ".jpg")):
            crop_name = clip_track(name)
            detect_arrow_and_orientation(crop_name)
