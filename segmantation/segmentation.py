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
import os
import shutil


import imageio
gif_images = []

def not_exist_mkdir( output_path ):
    if( not os.path.exists(output_path) ):
        os.mkdir( output_path )



if __name__ == '__main__':
    shutil.rmtree("./segment_pic")
    not_exist_mkdir("./segment_pic")
    
    segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=400, min_size=500)
    src = cv2.imread('./lightdesign/03.jpg')
    src = cv2.resize(src,(640,480) )
    segment = segmentator.processImage(src)

    # グレースケールへの変換(1chになる)
    tmp_gray=cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.merge((tmp_gray,tmp_gray,tmp_gray))

    mask = segment.reshape(list(segment.shape) + [1]).repeat(3, axis=2)
    masked = np.ma.masked_array(src, fill_value=0)
    for i in range(np.max(segment)):
        print("{0}/{1} process done".format(i,np.max(segment)-1))
        masked.mask = mask != i
        y, x = np.where(segment == i)
        top, bottom, left, right = min(y), max(y), min(x), max(x)
        dst = masked.filled()[top : bottom + 1, left : right + 1]  ## masked.maskで定義したマスク部分に0をかぶせる
        cv2.imwrite('./segment_pic/segment_{num}.jpg'.format(num=i), dst)

        #get gif
        gif_imagesrc = np.where(masked.mask,gray_image,src)
        gif_images.append(gif_imagesrc)
        
    #save to gif
    imageio.mimsave('segment.gif', gif_images,duration=0.2)

    segment = segment*(int(255/segment.max())-1)
    cv2.imwrite('./segment_pic/segment_all.jpg', segment)

    
