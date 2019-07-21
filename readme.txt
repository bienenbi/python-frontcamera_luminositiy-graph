　このプログラムは、Surface Pro 4のフロントカメラから画像を取得し,
その画像の輝度値をグラフとしてプロットするプログラムである.
  プログラムを起動するとフロントカメラから画像を検出する. そして,一度Qを押すとその時点までのグラフを
プロットし,もういちどQを押すと終了する.

##############################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

video_capture = cv2.VideoCapture(1)        		// Surface Pro 4 のフロントカメラを用いる場合は2にすること.
x = np.array([])
y = np.array([])

while True:
    ret, frame = video_capture.read()
    cv2.imshow('Video', frame)				// カメラが撮影している画像を映すコード.
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)	// カメラ画像をRGBからグレースケール画像にするコード.
    gray_mean = np.mean(gray)				// グレーの画像の輝度を平均するコード.
    y = np.append(y, gray_mean)				// 輝度を配列に収めるコード.

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.plot(y)						// 上のwhile文が終わった後に,配列に収められた輝度をグラフ化する.
plt.show()
video_capture.release()
cv2.destroyAllWindows()

#####################################################################################

	参考にしたサイト

http://tadaoyamaoka.hatenablog.com/entry/2017/02/21/201919 (最終閲覧日　2019/07/19)