# -*- coding: utf-8-*-
# Author: Jack Cui
import sys
from turtle import width
import cv2
from PIL import Image
import numpy as np
import struct

def parse_from_img(img_path):
    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape

    stepx = 10
    stepy = 10

    sml_w = img_w // stepx
    sml_h = img_h // stepy

    res_img = np.zeros((sml_h, sml_w, 3), np.uint8)

    for m in range(0, sml_w):
        for n in range(0, sml_h):
            map_col = int(m * stepx + stepx * 0.5)
            map_row = int(n * stepy + stepy * 0.5)
            res_img[n, m] = img[map_row, map_col]

    return res_img

def generate_img(big_img_path, small_img_path):

    big_img = cv2.imread(big_img_path)
    sml_img = cv2.imread(small_img_path)

    dst_img = big_img.copy()

    big_h, big_w, _ = big_img.shape
    sml_h, sml_w, _ = sml_img.shape

    stepx = big_w / sml_w
    stepy = big_h / sml_h

    for m in range(0, sml_w):
        for n in range(0, sml_h):
            map_col = int(m * stepx + stepx * 0.5)
            map_row = int(n * stepy + stepy * 0.5)

            if map_col < big_w and map_row < big_h:
                dst_img[map_row, map_col] = sml_img[n, m]

    return dst_img

def Img2Text(img_fname):
    img = cv2.imread(img_fname)
    height, width, _ = img.shape
    text_list = []
    for h in range(height):
        for w in range(width):
            R, G, B = img[h, w]
            if R | G | B == 0:
                break
            idx = (G << 8) + B
            text_list.append(chr(idx))
    text = "".join(text_list)
    with open("斗破苍穹_dec.txt", "a", encoding="utf-8") as f:
        f.write(text)

def Text2Img(txt_fname, save_fname):
    with open(txt_fname, "r", encoding="utf-8") as f:
        text = f.read()
    text_len = len(text)
    img_w = 1000
    img_h = 1680
    img = np.zeros((img_h, img_w, 3))
    x = 0
    y = 0
    for each_text in text:
        idx = ord(each_text)
        rgb = (0, (idx & 0xFF00) >> 8, idx & 0xFF)
        img[y, x] = rgb
        if x == img_w - 1:
            x = 0
            y += 1
        else:
            x += 1
    cv2.imwrite(save_fname, img)

def big_with_small(big_img_path, small_img_path, res_img_path):
    """
    大图里藏小图
    """
    dst_img = generate_img(big_img_path, small_img_path)
    cv2.imwrite(res_img_path, dst_img)

def npy2img(npy_ptr, img_ptr):
    npy = np.load(npy_ptr)
    bigimg = cv2.imread(img_ptr)
    print(npy)
    img_w = 36
    img_h = 80
    img = np.zeros((img_h, img_w, 3))
    x = 0
    y = 0
    iRGB = 0
    for i in npy:
        for idx in i:
            idx = struct.pack("f", float(idx))
            idx = struct.unpack("i", idx)[0]
            for j in range(4):
                tFF = 0xFF << (j*8)
                itoadd = (idx & tFF) >> (j*8)
                img[y,x][iRGB] = itoadd
                if iRGB == 2:
                    iRGB = 0
                    if x == img_w - 1:
                        x = 0
                        y += 1
                    else:
                        x += 1
                else:
                    iRGB += 1

    print("then")
    cv2.imwrite("dpcq.jpg", img)
    # big_img_path = "1.png"
    # small_img_path = "2.png"
    res_img_path = "res.jpg"
    big_with_small("0001_c07s1.jpg", "dpcq.jpg", res_img_path)
    img2 = cv2.imread("0001_c07s1.jpg")
    cv2.imwrite("000001.jpg", img2)


    height, width, _ = img.shape
    text_list = []
    for h in range(height):
        for w in range(width):
            R, G, B = img[h, w]
            R = int(R)
            G = int(G)
            B = int(B)

            # print(R)
            # print(G)
            if R == 0 and G == 0 and B == 0:
                break
            text_list.append(R)
            text_list.append(G)
            text_list.append(B)

    n = len(text_list)
    ans_list = []
    for x in range (0,int(n/4)*4, 4):
        # print(text_list[0])
        # print(text_list[1])
        # print(text_list[2])
        # print(text_list[3])
        idx = 0xFFFFFFFF
        idx = idx & (0xFFFFFF00 | text_list[x+0])
        idx = idx & (0xFFFF00FF |text_list[x+1] << 8)
        idx = idx & (0xFF00FFFF |text_list[x+2] << 16)
        idx = idx & (0x00FFFFFF |text_list[x+3] << 24)
        idx = struct.pack("I", idx)
        idx = struct.unpack("f", idx)[0]
        ans_list.append(idx)
        #print("trmp:",tmp)

    print(ans_list)

if __name__ == "__main__":
    npy2img("./1.npy", "./1.jpg")
    # # 将小图藏于大图中，并保存结果
    # big_img_path = "1.png"
    # small_img_path = "2.png"
    # res_img_path = "res.png"
    # big_with_small(big_img_path, small_img_path, res_img_path)

    # # 从大图中，解析出小图
    # parsed_img_fname = "parsed_img.png"
    # parsed_img = parse_from_img(res_img_path)
    # cv2.imwrite(parsed_img_fname, parsed_img)

    # # 文本生成图片
    # txt_fname = "斗破苍穹.txt"
    # txt_img_fname = "dpcq.png"
    # txt_res_img_path = "text_res.png"
    # Text2Img(txt_fname, txt_img_fname)

    # # 将生成的文本图片，藏于大图中，并保存结果
    # big_with_small(big_img_path, txt_img_fname, txt_res_img_path)

    # # 从藏有文本的大图中，解析出文本小图
    # parsed_img_text_fname = "pares_text_img.png"
    # parsed_img_text = parse_from_img(txt_res_img_path)
    # cv2.imwrite(parsed_img_text_fname, parsed_img_text)

    # # 从文本图片中，解析出文字
    # Img2Text(parsed_img_text_fname)
    # # 从含有特征的图片，解析出特征
    # # 从图片中计算
