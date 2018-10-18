# 你的微信好友都长啥样？

![](result.png)





### 系统要求

- Python 3.7
- 一个微信账号



### 安装 itchat 和 OpenCV

使用 Python 的 pip 指令安装

按 `Windows + R` 键打开命令行窗口，输入

```
pip3 install itchat numpy opencv-python
```



### itchat 登陆微信获取图片

itchat 有一个很棒的方法：`get_head_img`， 可以直接下载图片。

其中 `picDir` 参数是图片文件路径名，不是目录名 【这个起名很有误导性啊，研究了半天.....

```python
'''
extract_img.py
get images from WeChat friends list.
'''
import itchat

# 登陆
itchat.auto_login(hotReload=True)

# 获取通讯录列表
friends = itchat.get_friends(update=True)

# 编列列表中的字典，通过微信的用户唯一标识符 UserName 下载头像，头像的链接在 HeadImageUrl 中
for i, friend in enumerate(friends):
    itchat.get_head_img(userName=friend['UserName'], picDir='img/%d.png' % i)
```



运行时会先弹出二维码，和桌面版微信一样，直接扫码登陆即可



### OpenCV 拼接图片

这里的算法不是很优化，但我的态度是好不好看，能用就行~

```
'''
make_image.py
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
```

