'''
make_img.py
read all the images and combine into a single one. 
'''
import os
import numpy as np
import cv2

# 获取下载的所有图片
files = list(os.walk('img'))[0][-1]

# 结果图片的大小和网格数量
img_size = (100, 100)
result_grid = (35, 14)

for i in range(result_grid[1]):
    try:
        img_f = cv2.imread('img/'+files[i * result_grid[0]])
        # 下载下来的图片大小不同，拼接之前需要统一分辨率
        img_f = cv2.resize(img_f, img_size, interpolation=cv2.INTER_CUBIC)
    except:
        # 如果读不出来图像，就用黑色块代替
        img_f = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    for j in range(1, result_grid[0]):
        try:
            img_t = cv2.imread('img/'+files[i * result_grid[0] + j])
            img_t = cv2.resize(img_t, img_size, interpolation=cv2.INTER_CUBIC)
        except:
            img_t = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        img_f = np.append(img_f, img_t, axis=1)
    if i == 0:
        img = img_f
    else:
        img = np.append(img, img_f, axis=0)

# 保存结果
cv2.imwrite('result.png', img)
