import time
import xml.dom.minidom
from typing import List, Dict, Union, Tuple
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import hashlib
import os
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import clip
from transformers import BertTokenizer, RobertaTokenizerFast
import numpy as np
from pycocotools import mask as maskUtils



def rle_decode(rle_str, height, width):
    rle_encoding = maskUtils.frPyObjects(rle_str, height, width)
    segmentation_mask = maskUtils.decode(rle_encoding)
    return np.transpose(segmentation_mask, (1, 0, 2))[:, :, 0]

def generate_gaussian_mask_region(shape, center, sigma,v):

    mask = np.zeros(shape, dtype=np.float32)
    cy, cx = center

    for x in range(shape[1]):
        for y in range(shape[0]):
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            weight = np.exp(-0.5 * (distance / sigma) ** 2)
            mask[y, x] = weight


    min_val = np.min(mask)
    max_val = np.max(mask)
    normalized_data = (mask - min_val+0.000001) / (max_val+0.000001 - min_val) * v
    normalized_data = torch.from_numpy(normalized_data)

    return normalized_data


def transform_box2token_base(box,img,v):
    size = 224
    scale0 = size / img.width
    scale1 = size / img.height

    x0 = min(box[0]*scale0, size)
    y0 = min(box[1]*scale1, size)
    x1 = min(box[2]*scale0, size)
    y1 = min(box[3]*scale1, size)

    start = max(math.ceil(x0/16),1)-1+(max(math.ceil(y0/16),1)-1)*14+1
    end = max(math.ceil(x1/16),1)+(max(math.ceil(y0/16),1)-1)*14+1
    h = (max(math.ceil(y1/16),1)-1)-(max(math.ceil(y0/16),1)-1)

    list_se = [(start, end)]
    for i in range(h):
        list_se.append((list_se[0+i][0]+14, list_se[0+i][1]+14))

    gauss = True
    if gauss:
        width = end-start
        height = h+1
        gauss = generate_gaussian_mask_region((height,width),((height-1)/2,(width-1)/2),100,v)

        return list_se, gauss
    else:
        return list_se

def transform_box2token_large(box,img,v):
    size = 224
    scale0 = size / img.width
    scale1 = size / img.height

    x0 = min(box[0] * scale0, size)
    y0 = min(box[1] * scale1, size)
    x1 = min(box[2] * scale0, size)
    y1 = min(box[3] * scale1, size)

    start = max(math.ceil(x0/14),1)-1+(max(math.ceil(y0/14),1)-1)*16+1
    end = max(math.ceil(x1/14),1)+(max(math.ceil(y0/14),1)-1)*16+1
    h = (max(math.ceil(y1/14),1)-1)-(max(math.ceil(y0/14),1)-1)
    list_se = [(start,end)]
    for i in range(h):
        list_se.append((list_se[0+i][0]+16,list_se[0+i][1]+16))

    gauss = True
    if gauss:
        width = end - start
        height = h + 1
        gauss = generate_gaussian_mask_region((height, width), ((height - 1) / 2, (width - 1) / 2), 100, v)

        return list_se, gauss
    return list_se

def transform_box2token_32(box, img, v):
    patch_size = 32
    tokens_per_row = 224 // patch_size  # = 7

    x0 = min(box[0] * (224 / 224), 224)
    y0 = min(box[1] * (224 / 224), 224)
    x1 = min(box[2] * (224 / 224), 224)
    y1 = min(box[3] * (224 / 224), 224)

    # 패치 기준으로 나누기
    x0_patch = max(math.ceil(x0 / patch_size), 1) - 1
    y0_patch = max(math.ceil(y0 / patch_size), 1) - 1
    x1_patch = max(math.ceil(x1 / patch_size), 1) - 1
    y1_patch = max(math.ceil(y1 / patch_size), 1) - 1

    # 1D 토큰 index 계산
    start = x0_patch + y0_patch * tokens_per_row + 1
    end   = x1_patch + y0_patch * tokens_per_row + 1
    h     = y1_patch - y0_patch

    # 토큰 row별 인덱스 리스트 생성
    list_se = [(start, end)]
    for i in range(h):
        list_se.append((list_se[i][0] + tokens_per_row, list_se[i][1] + tokens_per_row))

    # Gaussian mask 생성
    gauss = True
    if gauss:
        width = end - start
        height = h + 1
        gauss = generate_gaussian_mask_region(
            (height, width),
            center=((height - 1) / 2, (width - 1) / 2),
            sigma=100,
            v=v
        )
        return list_se, gauss
    return list_se

def generate_gaussian_mask_from_bbox(
    x_min, y_min, x_max, y_max,
    original_size,   # (width, height)
    patch_size=32,   # ViT-B/32 기준
    sigma=100,
    v=1.0
):
    orig_w, orig_h = original_size
    scale_x = 224 / orig_w
    scale_y = 224 / orig_h

    # 원본 bbox → 224 기준으로 정규화
    x_min_224 = int(x_min * scale_x)
    y_min_224 = int(y_min * scale_y)
    x_max_224 = int(x_max * scale_x)
    y_max_224 = int(y_max * scale_y)

    # 224 기준 bbox → patch 단위
    x0_patch = max(math.ceil(x_min_224 / patch_size), 1) - 1
    y0_patch = max(math.ceil(y_min_224 / patch_size), 1) - 1
    x1_patch = max(math.ceil(x_max_224 / patch_size), 1) - 1
    y1_patch = max(math.ceil(y_max_224 / patch_size), 1) - 1

    width = x1_patch - x0_patch + 1
    height = y1_patch - y0_patch + 1
    center = ((height - 1) / 2, (width - 1) / 2)

    gauss_mask = generate_gaussian_mask_region((height, width), center, sigma, v)

    return gauss_mask, (x0_patch, y0_patch)
