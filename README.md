# Image Collage Maker

![](result-rand.png)


## System Requirements

- Python >= 3.5
- A WeChat account with a lot of friends

Note: The collage maker can be applied to any folder which contains sufficient images

### Install itchat, OpenCV, tqdm and scikit-learn

Open the terminal and type

```
pip3 install itchat opencv-python tqdm scikit-learn lap
```

If lap cannot be successfully installed, try to install lapsolver instead.
 
Because SciPy's linear sum assignment is implemented in Python, it is slow. So I chose lap/lapsolver whose kernels are implemented in C++

## How to use

1\. Use [extract_img.py](extract_img.py) to download head images of your friends

Download all your friends' head images(--dir specifies the directory to store these images):
```bash
python3 extract_img.py --dir img --type self
```

Or, download the group members' images in a group chat(replace ```name``` with the group chat's name and keep the double quotes):
```bash
python3 extract_img.py --dir img2 --type chatroom --name "name"
```

2\. Use [make_img.py](make_img.py) to make the collage

#### Option 1: Sorting

```bash
python3 make_img.py --path img --sort pca_lab --size 100
```

Use ```--ratio w h``` to change the aspect ratio, whose default is 16:9

Example: use ```--ratio 21 9``` to change aspect ratio to 21:9

Result:
![PCA-LAB](result-tsne_bgr.png)

#### Option 2: Fit a particular image

```bash
python3 make_img.py --path img --collage img/0.png --size 25 --dup 4 --out collage.png
```

```--dup4``` allows each image to be used four times. Increase that number if you don't have enough friends or you want a better fitting result. Note that a large number of images may result in long computational time.

Result: 
![collage.png](collage.png)

#### Other options

Use ```python3 make_img.py --help``` to get other optional arguments



