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

NumPy 和 SciPy会作为Dependency自动安装

## 使用方法

首先使用[extract_img.py](extract_img.py)下载图片

```bash
python3 extract_img.py
```

再使用[make_img.py](make_img.py)合成

```bash
python3 make_img.py --sort pca_lab --pic_size 100
```

使用```python3 make_img.py --help```来查看更多选项

## 部分排序方法展示

平均Hue排序

![av_hue](result-av_hue.png)

BGR值求和并排序

![bgr_sum](result-bgr_sum.png)

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

