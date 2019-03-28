#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:57:27 2019

@author: shuichi

opencvのがそのあつかい：http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_core/py_basic_ops/py_basic_ops.html
"""

import cv2
import numpy as np
import copy

def main():
    # 入力画像の読み込み
    #img = cv2.imread("./lightdesign/myroom2.jpg")
    img = cv2.imread("./lightdesign/05.jpg")
    img = cv2.resize(img,(640,480) )
    # 方法2(OpenCVで実装)       
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #hsv分割
    Hchanel,Schanel,Vchanel=cv2.split(hsv)
    hsv_transed=copy.deepcopy(hsv)
    
    hsv_transed[:,:,1] = 255
    hsv_transed[:,:,2] = 255#int(np.mean(Vchanel))
    # 結果を出力
    #cv2.imwrite("hsv.jpg", hsv)
    cv2.imshow('hsv_transed',cv2.cvtColor(hsv_transed, cv2.COLOR_HSV2BGR))

    cv2.imshow('image',img)
    #hsv output
    
    cv2.imshow('h',Hchanel)
    cv2.imshow('s',Schanel)
    cv2.imshow('v',Vchanel)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

    # hsv変換
    """
    green = np.uint8([[[0,255,0 ]]])
    hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
    print(hsv_green)
    """
