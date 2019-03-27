

import numpy as np
import cv2 as c
import glob
import os

# 8近傍の定義
neiborhood8 = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                            np.uint8)

path="./lightdesign/02.jpg"
img = c.imread(path, 0) # 0なしでカラー
img_dilate = c.dilate(img, neiborhood8, iterations=1)
img_diff = c.absdiff(img, img_dilate)
img_diff_not = c.bitwise_not(img_diff)
#gray = c.cvtColor(img_diff_not, c.COLOR_RGB2GRAY)

#at = c.adaptiveThreshold(img_diff_not, 255, c.ADAPTIVE_THRESH_GAUSSIAN_C, c.THRESH_BINARY, 7, 8) # intをいい感じに調整する
#c.imwrite(os.path.dirname(path) + '_clean_senga_color_gray/' + os.path.basename(path), img_diff_not)

c.imshow('test',img)
c.imshow('test2',img_dilate)
c.imshow('test3',img_diff)
c.imshow('test4',img_diff_not)
c.waitKey(10000)

c.destroyAllWindows()