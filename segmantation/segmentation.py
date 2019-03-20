#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:57:27 2019

@author: shuichi

パラメータ 	
sigma 	境界線の滑らかさ（複雑な境界線には小さい値、滑らかな境界線には大きい値）
k 	たぶん、どれくらい候補領域を統合するか（値が小さいと多くの小さい領域、大きいと少ない大きな領域に分割？）
min_size 	領域の最小サイズ（たぶん領域のピクセル数）


opencvのがそのあつかい：http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_core/py_basic_ops/py_basic_ops.html
"""

import cv2
import numpy as np

segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=300, min_size=1000)
src = cv2.imread('./lightdesign/02.jpg')
segment = segmentator.processImage(src)

mask = segment.reshape(list(segment.shape) + [1]).repeat(3, axis=2)
masked = np.ma.masked_array(src, fill_value=0)
for i in range(np.max(segment)):
    print("{0}/{1} process done".format(i,np.max(segment)))
    masked.mask = mask != i
    y, x = np.where(segment == i)
    top, bottom, left, right = min(y), max(y), min(x), max(x)
    dst = masked.filled()[top : bottom + 1, left : right + 1]
    cv2.imwrite('./segment_pic/segment_{num}.jpg'.format(num=i), dst)


segment = segment*(int(255/segment.max())-1)
cv2.imwrite('./segment_pic/segment_all.jpg', segment)
