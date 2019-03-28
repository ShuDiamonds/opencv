# -*- coding: utf-8 -*-
import cv2
import numpy as np

import copy
import pylab as plt

import os
import shutil
import imageio
gif_images = []

def not_exist_mkdir( output_path ):
    if( not os.path.exists(output_path) ):
        os.mkdir( output_path )

# 減色処理
def sub_color(src, K):

    # 次元数を1落とす
    Z = src.reshape((-1,3))

    # float32型に変換
    Z = np.float32(Z)

    # 基準の定義
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # K-means法で減色
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # UINT8に変換
    center = np.uint8(center)

    res = center[label.flatten()]

    # 配列の次元数と入力画像と同じに戻す
    return res.reshape((src.shape))

def show_histogram(im):
    if im.ndim == 2:
        # グレースケール
        plt.hist(im.lavel(), 256, range=(0, 255), fc='k')
        plt.show()

    elif im.ndim == 3:
        # カラー
        fig = plt.figure()
        fig.add_subplot(311)
        plt.hist(im[:,:,0].ravel(), 256, range=(0, 255), fc='b')
        plt.xlim(0,255)
        fig.add_subplot(312)
        plt.hist(im[:,:,1].ravel(), 256, range=(0, 255), fc='g')
        plt.xlim(0,255)
        fig.add_subplot(313)
        plt.hist(im[:,:,2].ravel(), 256, range=(0, 255), fc='r')
        plt.xlim(0,255)
        plt.show()


def segmentation(inputimg):
    shutil.rmtree("./segment_pic")
    not_exist_mkdir("./segment_pic")
    
    segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=200, min_size=1000)
    #src = cv2.imread(path)
    src = inputimg
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

def main():

    # 入力画像とスクリーントーン画像を取得
    img = cv2.imread("./lightdesign/myroom.jpg") 
    img = cv2.resize(img,(640,480) )
    # ノイズ除去
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,15,7,21)
    
    

    # 減色処理(三値化)
    dst = sub_color(dst, K=40)
    
    #segmentation(dst)

    #hsv
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    Hchanel,Schanel,Vchanel=cv2.split(hsv)
    hsv_transed=copy.deepcopy(hsv)
    hsv_transed[:,:,1] = 255
    hsv_transed[:,:,2] = 255#int(np.mean(Vchanel))
    show_histogram(hsv)
    
    # 結果を出力
    cv2.imwrite("kmeans_output.jpg", dst)

    cv2.imshow('stack',np.hstack((img,dst)))
    cv2.imshow('hsv_transed',cv2.cvtColor(hsv_transed, cv2.COLOR_HSV2BGR))
    cv2.imshow('h',Hchanel)
    cv2.imshow('s',Schanel)
    cv2.imshow('v',Vchanel)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()