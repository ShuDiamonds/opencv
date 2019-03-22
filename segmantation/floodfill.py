import numpy as np
import matplotlib.pyplot as plt
import cv2

#必ずUTF-8にて保存のこと

def main():

    # *** 閉曲線の作成 *** 
    x=[]
    y=[]
    for itht in range(361):
        tht = itht*np.pi/180.0
        num_polygon = 7
        r = 2+1*np.sin(tht*num_polygon)
        x.append(r*np.cos(tht))
        y.append(r*np.sin(tht))
    # *** プロット処理 *** 
    plt.figure( figsize=(3,3) )
    plt.plot( x, y )
    # *** プロット画像の保存 ***
    plt.savefig("tak.png")
    # *** 画像をOpenCV画像としてロード *** 
    img = cv2.imread("tak.png")
    print(img.shape) #300,300,3
    # *** エッジ検出画像をマスク画像として用意 ***
    mask = cv2.Canny(img, 100, 200) 
    print(mask.shape) #
    # *** マスク画像の拡張。（画像の上下左右に1ラインずつ拡張）
    mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, 
    cv2.BORDER_REPLICATE)
    print(mask.shape) #
    # *** 領域塗りつぶしと結果の表示
    cv2.floodFill( img, mask, (120,120), (0,255,255)) #third one is the start point
    sz = img.shape[:2]
    cv2.floodFill( img, mask, (int(sz[0]/2), int(sz[1]/2)), (0,255,0))
    cv2.imshow("canny",mask)
    cv2.imshow("flood",img)
    cv2.waitKey()

if __name__=='__main__':
    main()
