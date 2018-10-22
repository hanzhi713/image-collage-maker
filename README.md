# 你的微信好友都长啥样？

![](result-rand.png)


## 系统要求

- Python >= 3.5
- 一个微信账号


## 安装 itchat, OpenCV, tqdm 和 scikit-learn

使用 Python 的 pip 指令安装

按 `Windows + R` 键打开命令行窗口，输入

```
pip3 install itchat opencv-python tqdm scikit-learn
```

NumPy和SciPy会作为Dependency自动安装

## 使用方法

首先使用[extract_img.py](extract_img.py)下载图片

下载自己所有好友的头像（--dir的参数是下载目录）：
```bash
python3 extract_img.py --dir img --type self
```
下载某个群聊里所有成员的头像（请把```name```换成群聊的名字并保留双引号）：
```bash
python3 extract_img.py --dir img2 --type chatroom --name "name"
```

再使用[make_img.py](make_img.py)合成

```bash
python3 make_img.py --path img --sort pca_lab --size 100
```

使用```--ratio w h```可修改横纵比，默认16:9。
如```--ratio 21 9```可改为21:9。

使用```python3 make_img.py --help```来查看更多选项

## 部分排序方法展示

平均Hue排序

![av_hue](result-av_hue.png)

BGR值求和并排序

![bgr_sum](result-bgr_sum.png)

平均Saturation排序

![sat](result-av_sat.png)

LAB颜色空间排序

![lab](result-lab.png)

Luminosity排序

![lum](result-lum.png)

PCA-平均BGR排序

![PCA-BGR](result-pca_bgr.png)

PCA-平均HSV排序

![PCA-HSV](result-pca_hsv.png)

PCA-平均LAB排序

![PCA-LAB](result-pca_lab.png)

